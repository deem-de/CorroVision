"""Part 10 - Export the two small classification heads used by the demo app.

Retrains phase 1 of 04_transfer.py (frozen ResNet50 + new head) from the cached pooled features with
the same settings and seed, checks the test accuracy, and saves the heads to app/models/.
The heads are about 2 MB each, so they can be stored in the repository; ResNet50 itself is
downloaded by Keras the first time the app starts.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
from tensorflow import keras
from common import *

m = load_manifest()
Y = {s: labels(m, s) for s in ('train', 'val', 'test')}
FEAT = {s: np.load(WORK / f'tl_feat_{s}.npy') for s in ('train', 'val', 'test')}
DST = ROOT / 'app' / 'models'; DST.mkdir(parents=True, exist_ok=True)

for stage in (1, 2):
    yi = 0 if stage == 1 else 1
    ytr, yva, yte = (Y[s][yi] for s in ('train', 'val', 'test'))
    tr, va, te = (np.arange(len(y)) if stage == 1 else np.where(y >= 0)[0] for y in (ytr, yva, yte))
    sw = None
    if stage == 1:
        cnt = np.bincount(ytr, minlength=2); sw = (len(ytr) / (2 * cnt))[ytr].astype(np.float32)
    keras.utils.set_random_seed(SEED)
    head = keras.Sequential([keras.Input((2048,)), keras.layers.Dropout(0.3),
                             keras.layers.Dense(256, activation='relu'), keras.layers.Dropout(0.4),
                             keras.layers.Dense(1, activation='sigmoid') if stage == 1
                             else keras.layers.Dense(len(S2_CLASSES), activation='softmax')], name='head')
    head.compile(keras.optimizers.Adam(1e-3), metrics=['accuracy'],
                 loss='binary_crossentropy' if stage == 1 else 'sparse_categorical_crossentropy')
    head.fit(FEAT['train'][tr], ytr[tr], sample_weight=None if sw is None else sw[tr],
             validation_data=(FEAT['val'][va], yva[va]), epochs=60, batch_size=32, verbose=0,
             callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)])
    p = head.predict(FEAT['test'][te], verbose=0)
    pred = (p.reshape(-1) >= 0.5).astype(int) if stage == 1 else p.argmax(1)
    print(f'Stage {stage} head: test accuracy = {(pred == yte[te]).mean():.4f}', flush=True)
    head.save(DST / f'head_stage{stage}.keras')
save_json({'stage2_classes': S2_NAMES, 'backbone': 'ResNet50 (ImageNet, frozen, global average pooling)'}, DST / 'labels.json')
print('saved to', DST)
