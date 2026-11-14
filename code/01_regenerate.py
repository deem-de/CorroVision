"""Part 6a - Rebuild the Part 2 outputs (processed images + features) from the shared manifest.

Only `features_test.npz` and `processed_images.csv` were retained from Part 2, so the processed
images and the train/validation features are rebuilt with the unchanged Part 2 functions.
The rebuilt TEST features are compared against the retained `features_test.npz` to prove that
the rebuild is identical to what the earlier results were computed on.
"""
import sys
from joblib import Parallel, delayed
from common import *

DATA_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ['CORRO_DATA'])
m = load_manifest()
print('Manifest rows:', len(m))
print(pd.crosstab(m['split'], m['stage1']))


def process_one(rel_path, rows):
    """rows: list of (processed_path, is_aug, idx) that come from one original image."""
    img = preprocess(load_rgb(DATA_ROOT / rel_path))
    out = []
    for processed_path, is_aug, idx in sorted(rows, key=lambda r: r[1]):   # original first
        if is_aug:
            im = augment(img, np.random.default_rng(SEED + idx))            # same seed rule as Part 2
        else:
            im = img
        dst = PROC_DIR / processed_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        out.append((processed_path, im, hog_features(im), glcm_features(im), color_lbp_features(im)))
    return out


for split in ['test', 'val', 'train']:
    sub = split_frame(m, split)
    groups = [(rp, list(zip(g['processed_path'], g['is_aug'], g['idx'])))
              for rp, g in sub.groupby('rel_path', sort=False)]
    res = Parallel(n_jobs=-1, batch_size=16, verbose=2)(delayed(process_one)(rp, rows) for rp, rows in groups)
    by_path = {it[0]: it for r in res for it in r}
    items = [by_path[p] for p in sub['processed_path']]                     # back to manifest order
    np.save(WORK / f'images_{split}.npy', np.stack([it[1] for it in items]))
    np.savez_compressed(WORK / f'features_{split}.npz',
                        hog=np.stack([it[2] for it in items]), glcm=np.stack([it[3] for it in items]),
                        color_lbp=np.stack([it[4] for it in items]))
    print(split, '| images', len(items), '| HOG', items[0][2].shape, '| GLCM', items[0][3].shape,
          '| colour+LBP', items[0][4].shape)

# ---- proof of identity with the retained Part 2 file
old = np.load(IN_DIR / 'features_test.npz')
new = np.load(WORK / 'features_test.npz')
t = split_frame(m, 'test')
assert (old['rel_path'] == t['rel_path'].to_numpy()).all()
check = {'hog_max_abs_diff': float(np.abs(old['hog'] - new['hog']).max()),
         'glcm_max_abs_diff': float(np.abs(old['glcm'] - new['glcm']).max()),
         'hog_identical': bool(np.array_equal(old['hog'], new['hog'])),
         'glcm_identical': bool(np.array_equal(old['glcm'], new['glcm']))}
print('Rebuild check against retained features_test.npz:', check)
save_json(check, OUT / 'rebuild_check.json')
