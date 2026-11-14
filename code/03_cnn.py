"""Part 7 - Custom CNNs.

Stage 1: the baseline CNN trained in Part 5 (`best_cnn_stage1.keras`) is loaded unchanged and
         scored on the validation and test images (validation scores are needed for threshold tuning).
Stage 2: a deeper CNN is trained from scratch on the corroded images (eight corrosion types).
"""
import os, sys, time
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import tensorflow as tf
from tensorflow import keras
from common import *

keras.utils.set_random_seed(SEED)
BATCH = 32
EPOCHS = int(os.environ.get('CNN_EPOCHS', 40))
m = load_manifest()
Y = {s: labels(m, s) for s in ('train', 'val', 'test')}
IMG = {s: load_images(s) for s in ('train', 'val', 'test')}
PRED = OUT / 'preds'; PRED.mkdir(exist_ok=True)
MODELS = OUT / 'models'; MODELS.mkdir(exist_ok=True)


class Batches(keras.utils.PyDataset):
    """Feeds uint8 images from the memory-mapped array, scaled to [0, 1] at the model input."""

    def __init__(self, images, y=None, idx=None, shuffle=False, **kw):
        super().__init__(**kw)
        self.images, self.y, self.shuffle = images, y, shuffle
        self.idx = np.arange(len(images)) if idx is None else np.asarray(idx)
        self.rng = np.random.default_rng(SEED)
        self.order = self.idx.copy()
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.idx) / BATCH))

    def __getitem__(self, i):
        b = np.sort(self.order[i * BATCH:(i + 1) * BATCH]) if not self.shuffle else self.order[i * BATCH:(i + 1) * BATCH]
        x = np.stack([self.images[j] for j in b]).astype(np.float32) / 255.0
        return (x,) if self.y is None else (x, self.y[b])

    def on_epoch_end(self):
        if self.shuffle:
            self.order = self.rng.permutation(self.idx)


def predict(model, images, idx=None):
    return model.predict(Batches(images, idx=idx), verbose=0)


# ============================================================ Stage 1: Part 5 baseline (no retraining)
base1 = keras.models.load_model(IN_DIR / 'part5' / 'best_cnn_stage1.keras')
s1_val = predict(base1, IMG['val']).reshape(-1)
s1_test = predict(base1, IMG['test']).reshape(-1)
old = pd.read_csv(IN_DIR / 'part5' / 'cnn_test_predictions.csv')
print('Baseline CNN reproduced on rebuilt test images | max |prob diff| vs Part 5 file =',
      float(np.abs(old['prob_corroded'].to_numpy() - s1_test).max()),
      '| identical predictions:', bool(((s1_test >= 0.5).astype(int) == old['predicted'].to_numpy()).all()), flush=True)

# ============================================================ Stage 2: deeper CNN, eight classes
tr = np.where(Y['train'][1] >= 0)[0]; va = np.where(Y['val'][1] >= 0)[0]


def conv_bn_relu(x, filters, separable=True, strides=1):
    conv = keras.layers.SeparableConv2D if separable else keras.layers.Conv2D
    x = conv(filters, 3, strides=strides, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    return keras.layers.ReLU()(x)


# Stem (standard 3x3 convolution, stride 2) + four depthwise-separable blocks (1, 1, 2, 2 convolutions).
# Separable convolutions keep the network deeper than the Stage 1 baseline (7 convolution layers
# instead of 4) but light enough to train without a GPU.
inp = keras.Input((IMG_SIZE, IMG_SIZE, 3))
x = keras.layers.RandomFlip('horizontal_and_vertical', seed=SEED)(inp)       # active in training only
x = conv_bn_relu(x, 32, separable=False, strides=2)
for f, n_conv in ((64, 1), (128, 1), (256, 2), (384, 2)):
    for _ in range(n_conv):
        x = conv_bn_relu(x, f)
    x = keras.layers.MaxPooling2D()(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.Dropout(0.4)(x)
out = keras.layers.Dense(len(S2_CLASSES), activation='softmax')(x)
cnn2 = keras.Model(inp, out, name='deeper_cnn_stage2')
cnn2.compile(optimizer=keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn2.summary(print_fn=lambda s: print(s, flush=True))

ckpt = MODELS / 'cnn_stage2.keras'
# CNN_EVAL_ONLY=1 skips training and scores the saved best checkpoint (used when training was stopped early).
EVAL_ONLY = os.environ.get('CNN_EVAL_ONLY') == '1'
t0 = time.time()
if not EVAL_ONLY:
    hist = cnn2.fit(Batches(IMG['train'], Y['train'][1], tr, shuffle=True),
                    validation_data=Batches(IMG['val'], Y['val'][1], va),
                    epochs=EPOCHS, verbose=2,
                    callbacks=[keras.callbacks.ModelCheckpoint(str(ckpt), monitor='val_loss', save_best_only=True),
                               keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
                               keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
                               keras.callbacks.CSVLogger(str(OUT / 'cnn_stage2_history.csv'))])
    print(f'Stage 2 CNN training time: {(time.time() - t0) / 60:.1f} min', flush=True)

best2 = keras.models.load_model(ckpt)
p_val = predict(best2, IMG['val']); p_test = predict(best2, IMG['test'])      # all images (end-to-end needs it)
np.savez(PRED / 'cnn.npz',
         s1_val_score=s1_val, s1_val_pred=(s1_val >= 0.5).astype(int),
         s1_test_score=s1_test, s1_test_pred=(s1_test >= 0.5).astype(int),
         s2_val_pred=p_val.argmax(1), s2_test_pred=p_test.argmax(1), s2_test_proba=p_test)
y2 = Y['test'][1]
h = pd.read_csv(OUT / 'cnn_stage2_history.csv')                      # written by CSVLogger during training
save_json({'stage2_params': int(cnn2.count_params()), 'stage2_epochs_run': len(h),
           'stage2_best_epoch': int(h['val_loss'].idxmin()) + 1, 'stage2_best_val_loss': float(h['val_loss'].min()),
           'stage2_best_val_accuracy': float(h.loc[h['val_loss'].idxmin(), 'val_accuracy']),
           'stopped_manually': EVAL_ONLY}, PRED / 'cnn_info.json')
print('Deeper CNN Stage 2 test accuracy:', float((p_test.argmax(1)[y2 >= 0] == y2[y2 >= 0]).mean()), flush=True)
