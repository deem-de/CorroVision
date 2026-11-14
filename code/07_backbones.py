"""Part 8b - Stronger pretrained backbones, feature fusion and test-time augmentation (TTA).

Same recipe as phase 1 of 04_transfer.py (frozen ImageNet backbone + new classification head):
  * ConvNeXt and EfficientNetV2 backbones are compared with ResNet50
  * Fusion: the pooled features of several backbones are concatenated before the head
  * TTA: validation/test images are also scored after a horizontal and a vertical flip, and the
    three probability vectors are averaged
Every choice (which backbone, fusion or not, TTA or not) is made on the VALIDATION set only.

Usage: python 07_backbones.py convnext_tiny effnetv2_s [convnext_base ...]
"""
import os, sys, time, itertools
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import tensorflow as tf
from tensorflow import keras
from common import *

A = keras.applications
BACKBONES = {  # name: (constructor, preprocessing applied to 0-255 RGB input)
    'resnet50': (A.ResNet50, A.resnet50.preprocess_input),
    'convnext_tiny': (A.ConvNeXtTiny, None),            # ConvNeXt / EfficientNetV2 normalise inside the model
    'convnext_small': (A.ConvNeXtSmall, None),
    'convnext_base': (A.ConvNeXtBase, None),
    'effnetv2_s': (A.EfficientNetV2S, None),
    'effnetv2_m': (A.EfficientNetV2M, None),
}
FLIPS = {'': lambda x: x, '_hflip': lambda x: x[:, :, ::-1], '_vflip': lambda x: x[:, ::-1]}
names = sys.argv[1:] or ['convnext_tiny', 'effnetv2_s']
m = load_manifest()
Y = {s: labels(m, s) for s in ('train', 'val', 'test')}
PRED = OUT / 'preds'; PRED.mkdir(exist_ok=True)
BATCH = 32


# ------------------------------------------------------------------ 1. cache pooled features
def extract(name):
    ctor, prep = BACKBONES[name]
    todo = [(s, f) for s in ('val', 'test', 'train') for f in FLIPS
            if not (s == 'train' and f) and not (WORK / f'bb_{name}_{s}{f}.npy').exists()]
    if not todo:
        return
    net = ctor(weights='imagenet', include_top=False, pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    net.trainable = False
    for s, f in todo:
        imgs = load_images(s); t0 = time.time(); out = []
        for i in range(0, len(imgs), BATCH):
            x = FLIPS[f](np.asarray(imgs[i:i + BATCH])).astype(np.float32)
            out.append(net.predict_on_batch(prep(x) if prep else x))
        np.save(WORK / f'bb_{name}_{s}{f}.npy', np.concatenate(out).astype(np.float32))
        print(f'  {name}: cached {s}{f} {np.concatenate(out).shape} in {time.time() - t0:.0f}s', flush=True)
    keras.backend.clear_session()


for n in names:
    extract(n)


def feats(combo, split, flip=''):
    return np.hstack([np.load(WORK / f'bb_{n}_{split}{flip}.npy') for n in combo])


# ------------------------------------------------------------------ 2. head on (fused) features
def run(combo, stage):
    Xtr = feats(combo, 'train'); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6      # standardise with TRAIN statistics
    z = lambda X: (X - mu) / sd
    yi = 0 if stage == 1 else 1
    ytr, yva = Y['train'][yi], Y['val'][yi]
    tr = np.arange(len(ytr)) if stage == 1 else np.where(ytr >= 0)[0]
    va = np.arange(len(yva)) if stage == 1 else np.where(yva >= 0)[0]
    sw = None
    if stage == 1:
        cnt = np.bincount(ytr, minlength=2); sw = (len(ytr) / (2 * cnt))[ytr].astype(np.float32)
    keras.utils.set_random_seed(SEED)
    head = keras.Sequential([keras.Input((Xtr.shape[1],)), keras.layers.Dropout(0.3),
                             keras.layers.Dense(256, activation='relu'), keras.layers.Dropout(0.4),
                             keras.layers.Dense(1, activation='sigmoid') if stage == 1
                             else keras.layers.Dense(len(S2_CLASSES), activation='softmax')])
    head.compile(keras.optimizers.Adam(1e-3), metrics=['accuracy'],
                 loss='binary_crossentropy' if stage == 1 else 'sparse_categorical_crossentropy')
    Xva = z(feats(combo, 'val'))
    h = head.fit(z(Xtr)[tr], ytr[tr], sample_weight=None if sw is None else sw[tr],
                 validation_data=(Xva[va], yva[va]), epochs=60, batch_size=BATCH, verbose=0,
                 callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)])
    out = {}
    for s in ('val', 'test'):
        p = {f: head.predict(z(feats(combo, s, f)), verbose=0) for f in FLIPS}
        out[s] = p['']; out[s + '_tta'] = np.mean(list(p.values()), axis=0)
    out['epochs'] = len(h.history['loss'])
    return out


def acc(p, y, stage):
    pred = (p.reshape(-1) >= 0.5).astype(int) if stage == 1 else p.argmax(1)
    keep = np.ones(len(y), bool) if stage == 1 else y >= 0
    return float((pred[keep] == y[keep]).mean())


have = [n for n in BACKBONES if (WORK / f'bb_{n}_train.npy').exists()]
combos = [(n,) for n in have] + [c for r in (2, 3) for c in itertools.combinations(have, r)]
rows, RES = [], {}
for combo in combos:
    r = {st: run(combo, st) for st in (1, 2)}; RES[combo] = r
    rows.append({'Backbone(s)': ' + '.join(combo), 'Features': feats(combo, 'val').shape[1],
                 'S1 val': acc(r[1]['val'], Y['val'][0], 1), 'S1 val TTA': acc(r[1]['val_tta'], Y['val'][0], 1),
                 'S2 val': acc(r[2]['val'], Y['val'][1], 2), 'S2 val TTA': acc(r[2]['val_tta'], Y['val'][1], 2),
                 'S1 test': acc(r[1]['test'], Y['test'][0], 1), 'S1 test TTA': acc(r[1]['test_tta'], Y['test'][0], 1),
                 'S2 test': acc(r[2]['test'], Y['test'][1], 2), 'S2 test TTA': acc(r[2]['test_tta'], Y['test'][1], 2)})
    print(pd.DataFrame(rows[-1:]).round(4).to_string(index=False, header=len(rows) == 1), flush=True)
tab = pd.DataFrame(rows); tab.to_csv(OUT / 'backbone_comparison.csv', index=False)

# ------------------------------------------------------------------ 3. export the validation-selected models
def export(name, combo, tta):
    k = '_tta' if tta else ''
    s1, s2 = RES[combo][1], RES[combo][2]
    np.savez(PRED / f'{name}.npz',
             s1_val_score=s1['val' + k].reshape(-1), s1_val_pred=(s1['val' + k].reshape(-1) >= 0.5).astype(int),
             s1_test_score=s1['test' + k].reshape(-1), s1_test_pred=(s1['test' + k].reshape(-1) >= 0.5).astype(int),
             s2_val_pred=s2['val' + k].argmax(1), s2_test_pred=s2['test' + k].argmax(1), s2_test_proba=s2['test' + k])
    save_json({'backbones': list(combo), 'tta': tta}, PRED / f'{name}_info.json')


new = [c for c in combos if c != ('resnet50',)]
if new:
    single = max([c for c in new if len(c) == 1], key=lambda c: tab.loc[combos.index(c), 'S2 val'], default=None)
    if single:
        export('best_single', single, False); print('Best new single backbone by Stage 2 validation accuracy:', single)
    best = max(new, key=lambda c: max(tab.loc[combos.index(c), 'S2 val'], tab.loc[combos.index(c), 'S2 val TTA']))
    i = combos.index(best); tta = bool(tab.loc[i, 'S2 val TTA'] > tab.loc[i, 'S2 val'])
    export('best_fusion', best, tta); print('Best overall by Stage 2 validation accuracy:', best, '| TTA:', tta)
