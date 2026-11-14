"""Part 8 - Transfer learning with ResNet50 (ImageNet weights), both stages.

Phase 1 (feature extraction): the whole backbone is frozen and only a new classification head
        (dropout - dense 256 - dropout - output) is trained on the 2,048 pooled features.
Phase 2 (fine-tuning): the last residual stage (conv5_x, 9 conv layers) is unfrozen and trained
        together with the head at a low learning rate; BatchNorm layers stay in inference mode.

Because conv1-conv4 never change, their output (14x14x1024) is computed once per image and cached.
This is mathematically identical to running the full network with those layers frozen, and it makes
fine-tuning feasible without a GPU.
"""
import os, sys, time
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import tensorflow as tf
from tensorflow import keras
from common import *

keras.utils.set_random_seed(SEED)
BATCH = 32
m = load_manifest()
Y = {s: labels(m, s) for s in ('train', 'val', 'test')}
PRED = OUT / 'preds'; PRED.mkdir(exist_ok=True)
MODELS = OUT / 'models'; MODELS.mkdir(exist_ok=True)
CUT = 'conv4_block6_out'                                   # last frozen layer in phase 2
C4_SHAPE = (14, 14, 1024)

base = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False
lower = keras.Model(base.input, base.get_layer(CUT).output)            # conv1 ... conv4 (always frozen)


def build_top():
    """conv5_x as a separate model that starts from the cached conv4 activations."""
    full = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    top = keras.Model(full.get_layer(CUT).output, full.output, name='resnet50_conv5')
    return top


# ------------------------------------------------------------------ cache conv4 activations + pooled features
top_frozen = build_top(); top_frozen.trainable = False
for split in ('val', 'test', 'train'):
    c4_path, f_path = WORK / f'c4_{split}.f16', WORK / f'tl_feat_{split}.npy'
    if f_path.exists():
        continue
    imgs = load_images(split); n = len(imgs)
    c4 = np.lib.format.open_memmap(c4_path, mode='w+', dtype=np.float16, shape=(n,) + C4_SHAPE)
    feat = np.zeros((n, 2048), np.float32)
    t0 = time.time()
    for i in range(0, n, 32):
        x = keras.applications.resnet50.preprocess_input(np.asarray(imgs[i:i + 32]).astype(np.float32))
        a = lower.predict_on_batch(x)
        c4[i:i + 32] = a.astype(np.float16)
        feat[i:i + 32] = top_frozen.predict_on_batch(a).mean(axis=(1, 2))
        if i % 1920 == 0:
            print(f'  caching {split}: {i}/{n} ({time.time() - t0:.0f}s)', flush=True)
    c4.flush(); del c4
    np.save(f_path, feat)
FEAT = {s: np.load(WORK / f'tl_feat_{s}.npy') for s in ('train', 'val', 'test')}
C4 = {s: np.load(WORK / f'c4_{s}.f16', mmap_mode='r') for s in ('train', 'val', 'test')}


class C4Batches(keras.utils.PyDataset):
    def __init__(self, split, y=None, idx=None, shuffle=False, sample_weight=None, **kw):
        super().__init__(**kw)
        self.a, self.y, self.shuffle, self.sw = C4[split], y, shuffle, sample_weight
        self.idx = np.arange(len(self.a)) if idx is None else np.asarray(idx)
        self.rng = np.random.default_rng(SEED); self.order = self.idx.copy(); self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.idx) / BATCH))

    def __getitem__(self, i):
        b = np.sort(self.order[i * BATCH:(i + 1) * BATCH])              # sorted = faster memmap reads
        x = np.asarray(self.a[b]).astype(np.float32)
        if self.y is None: return (x,)
        if self.sw is None: return x, self.y[b]
        return x, self.y[b], self.sw[b]

    def on_epoch_end(self):
        if self.shuffle: self.order = self.rng.permutation(self.idx)


def make_head(n_out):
    act, units = ('sigmoid', 1) if n_out == 2 else ('softmax', n_out)
    return keras.Sequential([keras.Input((2048,)), keras.layers.Dropout(0.3),
                             keras.layers.Dense(256, activation='relu'), keras.layers.Dropout(0.4),
                             keras.layers.Dense(units, activation=act)], name='head')


def callbacks(tag, patience):
    return [keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(1, patience // 3), min_lr=1e-6),
            keras.callbacks.CSVLogger(str(OUT / f'{tag}_history.csv'))]


def run_stage(stage):
    y_idx = 0 if stage == 1 else 1
    ytr, yva = Y['train'][y_idx], Y['val'][y_idx]
    tr = np.arange(len(ytr)) if stage == 1 else np.where(ytr >= 0)[0]
    va = np.arange(len(yva)) if stage == 1 else np.where(yva >= 0)[0]
    n_out = 2 if stage == 1 else len(S2_CLASSES)
    loss = 'binary_crossentropy' if stage == 1 else 'sparse_categorical_crossentropy'
    sw = None
    if stage == 1:                                                      # balanced class weights (as in Part 5)
        cnt = np.bincount(ytr, minlength=2); w = len(ytr) / (2 * cnt); sw = w[ytr].astype(np.float32)
        print('Stage 1 class weights:', w, flush=True)

    # ---- phase 1: frozen backbone, train head on cached pooled features
    keras.utils.set_random_seed(SEED)
    head = make_head(n_out)
    head.compile(keras.optimizers.Adam(1e-3), loss=loss, metrics=['accuracy'])
    t0 = time.time()
    h1 = head.fit(FEAT['train'][tr], ytr[tr], sample_weight=None if sw is None else sw[tr],
                  validation_data=(FEAT['val'][va], yva[va]), epochs=60, batch_size=BATCH, verbose=0,
                  callbacks=callbacks(f'tl_head_stage{stage}', 8))
    t1 = time.time() - t0
    print(f'[TL phase 1, stage {stage}] epochs {len(h1.history["loss"])}, best val loss '
          f'{min(h1.history["val_loss"]):.4f}, val acc at best {h1.history["val_accuracy"][int(np.argmin(h1.history["val_loss"]))]:.4f}', flush=True)
    p1 = {s: head.predict(FEAT[s], verbose=0) for s in ('val', 'test')}

    # ---- phase 2: unfreeze conv5_x and fine-tune with the phase-1 head
    top = build_top()
    for layer in top.layers:
        layer.trainable = not isinstance(layer, keras.layers.BatchNormalization)
    inp = keras.Input(C4_SHAPE)
    z = top(inp, training=False)                                        # BatchNorm in inference mode
    z = keras.layers.GlobalAveragePooling2D()(z)
    ft = keras.Model(inp, head(z), name=f'resnet50_ft_stage{stage}')
    ft.compile(keras.optimizers.Adam(1e-5), loss=loss, metrics=['accuracy'])
    t0 = time.time()
    h2 = ft.fit(C4Batches('train', ytr, tr, shuffle=True, sample_weight=sw),
                validation_data=C4Batches('val', yva, va),
                epochs=int(os.environ.get('FT_EPOCHS', 8)), verbose=2,
                callbacks=callbacks(f'tl_ft_stage{stage}', 3))
    t2 = time.time() - t0
    ft.save(MODELS / f'tl_ft_stage{stage}.keras')
    p2 = {s: ft.predict(C4Batches(s), verbose=0) for s in ('val', 'test')}
    info = {'phase1_epochs': len(h1.history['loss']), 'phase1_minutes': t1 / 60,
            'phase2_epochs': len(h2.history['loss']), 'phase2_best_epoch': int(np.argmin(h2.history['val_loss'])) + 1,
            'phase2_minutes': t2 / 60, 'head_params': int(head.count_params()),
            'phase2_trainable_params': int(sum(np.prod(w.shape) for w in ft.trainable_weights))}
    return p1, p2, info


res = {st: run_stage(st) for st in (1, 2)}
for k, name in ((0, 'tl_head'), (1, 'tl_ft')):
    s1, s2 = res[1][k], res[2][k]
    np.savez(PRED / f'{name}.npz',
             s1_val_score=s1['val'].reshape(-1), s1_val_pred=(s1['val'].reshape(-1) >= 0.5).astype(int),
             s1_test_score=s1['test'].reshape(-1), s1_test_pred=(s1['test'].reshape(-1) >= 0.5).astype(int),
             s2_val_pred=s2['val'].argmax(1), s2_test_pred=s2['test'].argmax(1), s2_test_proba=s2['test'])
    y1, y2, _ = Y['test']; p = np.load(PRED / f'{name}.npz')
    print(f"{name}: Stage 1 test acc = {(p['s1_test_pred'] == y1).mean():.4f} | "
          f"Stage 2 test acc = {(p['s2_test_pred'][y2 >= 0] == y2[y2 >= 0]).mean():.4f}", flush=True)
save_json({'stage1': res[1][2], 'stage2': res[2][2]}, PRED / 'tl_info.json')
