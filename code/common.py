"""CorroVision - shared configuration and helpers (Parts 6-9, Weeks 9-11).

Everything here reuses the team's earlier decisions unchanged:
the 70/15/15 split and labels come from `processed_images.csv` (Part 2),
and the preprocessing / augmentation / HOG / GLCM functions are copied from Part 2.
"""
import os, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

SEED = 42
IMG_SIZE = 224
random.seed(SEED); np.random.seed(SEED)

ROOT = Path(os.environ.get('CORRO_ROOT', Path(__file__).resolve().parent.parent))
IN_DIR = ROOT / 'inputs'                      # processed_images.csv, features_test.npz, part5/
WORK = ROOT / 'work'                          # regenerated images, features, cached activations
OUT = ROOT / 'outputs'                        # predictions, metrics, figures
FIG = OUT / 'figures'
PROC_DIR = WORK / 'processed'
for d in (WORK, OUT, FIG):
    d.mkdir(parents=True, exist_ok=True)

HEALTHY = 'healthy_structure'
S1_NAMES = ['Healthy', 'Corroded']
S2_CLASSES = ['crevice_corrosion', 'erosion_corrosion', 'galvanic_corrosion', 'mic_corrosion',
              'pitting_corrosion', 'stress_corrosion', 'under_insulation_corrosion', 'uniform_corrosion']
ALL9 = [HEALTHY] + S2_CLASSES


def nice(c):
    return (c.replace('_corrosion', '').replace('_structure', '').replace('_', ' ')
             .title().replace('Mic', 'MIC'))


S2_NAMES = [nice(c) for c in S2_CLASSES]
ALL9_NAMES = [nice(c) for c in ALL9]


def load_manifest():
    """Shared manifest written by Part 2 (one row per processed image)."""
    m = pd.read_csv(IN_DIR / 'processed_images.csv', keep_default_na=False)
    m['is_aug'] = m['is_aug'].astype(str).str.lower().map({'true': True, 'false': False})
    # Part 2 named every file after the image's position in the clean dataframe
    # (train/<class>/00001.png, train_aug/<class>/00001_aug1.png), so the index that
    # seeded the augmentation (SEED + idx) can be read back from the file name.
    m['idx'] = m['processed_path'].map(lambda p: int(Path(p).stem.split('_')[0]))
    assert set(m['split']) == {'train', 'val', 'test'}
    assert m.loc[m['split'] != 'train', 'is_aug'].eq(False).all()
    return m


# ----------------------------------------------------------------------------- Part 2 (unchanged)
def load_rgb(path):
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def resize(rgb):
    return cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def denoise(rgb):
    return cv2.bilateralFilter(rgb, 5, 20, 3)


def color_correct(rgb, gain_range=(0.5, 3.0)):
    x = rgb.astype(np.float32)
    means = x.reshape(-1, 3).mean(axis=0) + 1e-6
    gains = np.clip(means.mean() / means, *gain_range)
    return np.clip(x * gains, 0, 255).astype(np.uint8)


def enhance_contrast(rgb, clip=2.0, grid=(8, 8)):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess(rgb):
    return enhance_contrast(color_correct(denoise(resize(rgb))))


def to_unit(rgb_uint8):
    return rgb_uint8.astype(np.float32) / 255.0


def augment(rgb_uint8, rng):
    img = to_unit(rgb_uint8)
    M = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), rng.uniform(-30, 30), 1.0)
    img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    if rng.random() < 0.5: img = img[:, ::-1]
    if rng.random() < 0.5: img = img[::-1, :]
    img = np.clip(img * rng.uniform(0.7, 1.3) + rng.uniform(-0.08, 0.08), 0, 1)
    return (img * 255).round().astype(np.uint8)


HOG_ORIENT, HOG_PPC, HOG_CPB = 9, (16, 16), (2, 2)
GLCM_LEVELS, GLCM_DIST, GLCM_ANGLES = 32, [1, 2, 4], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPS = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'ASM', 'correlation', 'entropy']


def hog_features(rgb_uint8):
    from skimage.feature import hog
    gray = cv2.cvtColor(to_unit(rgb_uint8), cv2.COLOR_RGB2GRAY)
    return hog(gray, orientations=HOG_ORIENT, pixels_per_cell=HOG_PPC,
               cells_per_block=HOG_CPB, block_norm='L2-Hys').astype(np.float32)


def glcm_features(rgb_uint8):
    from skimage.feature import graycomatrix, graycoprops
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    q = (gray.astype(np.uint16) * GLCM_LEVELS // 256).astype(np.uint8)
    P = graycomatrix(q, GLCM_DIST, GLCM_ANGLES, levels=GLCM_LEVELS, symmetric=True, normed=True)
    feats = [graycoprops(P, p).ravel() for p in GLCM_PROPS if p != 'entropy']
    feats.append(-np.sum(P * np.log2(P + 1e-12), axis=(0, 1)).ravel())
    return np.concatenate(feats).astype(np.float32)


# ----------------------------------------------------------------------------- new in Part 6
def color_lbp_features(rgb_uint8):
    """Extension proposed in the progress report: colour + LBP texture (92 values).

    HSV histograms (32 hue + 16 saturation + 16 value bins) describe colour, which
    neither HOG nor GLCM uses; uniform LBP histograms at two radii describe micro-texture.
    """
    from skimage.feature import local_binary_pattern
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    n = hsv.shape[0] * hsv.shape[1]
    feats = [np.bincount((hsv[..., 0].ravel().astype(np.int32) * 32) // 180, minlength=32)[:32] / n,
             np.bincount(hsv[..., 1].ravel() // 16, minlength=16) / n,
             np.bincount(hsv[..., 2].ravel() // 16, minlength=16) / n]
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    for P, R in ((8, 1), (16, 2)):
        lbp = local_binary_pattern(gray, P, R, method='uniform').astype(np.int32)
        feats.append(np.bincount(lbp.ravel(), minlength=P + 2) / n)
    return np.concatenate(feats).astype(np.float32)


# ----------------------------------------------------------------------------- data access
def split_frame(m, split):
    return m[m['split'] == split].reset_index(drop=True)


def load_images(split):
    """uint8 array (N, 224, 224, 3) written by 01_regenerate.py, in manifest order."""
    return np.load(WORK / f'images_{split}.npy', mmap_mode='r')


def labels(m, split):
    s = split_frame(m, split)
    y1 = (s['stage1'] == 'Corroded').astype(np.int64).to_numpy()            # 0 Healthy, 1 Corroded
    y2 = s['stage2'].map({c: i for i, c in enumerate(S2_CLASSES)}).fillna(-1).astype(np.int64).to_numpy()
    y9 = s['class'].map({c: i for i, c in enumerate(ALL9)}).astype(np.int64).to_numpy()
    return y1, y2, y9


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
