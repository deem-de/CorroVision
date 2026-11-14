"""Part 6b - Classical methods on the rebuilt features (same split, same settings as Parts 3-4).

* HOG + SVM            : RBF kernel, C = 10, balanced class weights (the configuration that the
                         Part 3 validation grid selected for both stages).
* GLCM + Random Forest : Part 4 grid (trees 100/200, min leaf 1/2) selected by validation macro-F1.
* Colour-Texture + RF  : extension - GLCM (84) + HSV histograms and LBP (92) = 176 features,
                         same Random Forest grid.
Per-image test predictions are saved so that McNemar's test and the end-to-end pipeline can be
computed later (07_evaluate.py).

Run with scikit-learn 1.6.1 (the version on Colab when Parts 3-4 were run): Random Forest results
depend on the scikit-learn version, and 1.6.1 reproduces the progress-report numbers exactly.
Usage: python 02_classical.py [rf] [svm]   (default: both)
"""
import sys, time
import sklearn
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from common import *

m = load_manifest()
F = {s: np.load(WORK / f'features_{s}.npz') for s in ('train', 'val', 'test')}
Y = {s: labels(m, s) for s in ('train', 'val', 'test')}
PRED = OUT / 'preds'; PRED.mkdir(exist_ok=True)
PARTS = set(sys.argv[1:]) or {'rf', 'svm'}
print('scikit-learn', sklearn.__version__, '| parts:', sorted(PARTS), flush=True)


def stage_data(X, stage):
    """Stage 1 uses every image; Stage 2 is trained/validated on corroded images only."""
    out = {}
    for s in ('train', 'val', 'test'):
        y1, y2, _ = Y[s]
        out[s] = (X[s], y1) if stage == 1 else (X[s][y2 >= 0], y2[y2 >= 0])
    return out


def fit_svm(X, stage):
    d = stage_data(X, stage)
    t0 = time.time()
    clf = SVC(kernel='rbf', gamma='scale', C=10, class_weight='balanced', random_state=SEED)
    clf.fit(*d['train'])
    val_f1 = f1_score(d['val'][1], clf.predict(d['val'][0]), average='macro')
    print(f'[HOG+SVM stage {stage}] val macro-F1 = {val_f1:.4f} ({time.time() - t0:.0f}s)', flush=True)
    return clf, {'kernel': 'rbf', 'C': 10, 'val_macro_f1': val_f1}


def fit_rf(X, stage, tag):
    d = stage_data(X, stage)
    best = (-1, None, None)
    for n in (100, 200):
        for leaf in (1, 2):
            clf = RandomForestClassifier(n_estimators=n, min_samples_leaf=leaf, class_weight='balanced',
                                         random_state=SEED, n_jobs=-1).fit(*d['train'])
            f1 = f1_score(d['val'][1], clf.predict(d['val'][0]), average='macro')
            if f1 > best[0]:
                best = (f1, clf, {'n_estimators': n, 'min_samples_leaf': leaf, 'val_macro_f1': f1})
    print(f'[{tag} stage {stage}] best {best[2]}', flush=True)
    return best[1], best[2]


def export(name, clf1, clf2, X, info):
    """Stage 1 scores/predictions + Stage 2 predictions for ALL test images (needed end to end)."""
    score = (lambda Z: clf1.decision_function(Z)) if hasattr(clf1, 'decision_function') \
        else (lambda Z: clf1.predict_proba(Z)[:, 1])
    np.savez(PRED / f'{name}.npz',
             s1_val_pred=clf1.predict(X['val']), s1_val_score=score(X['val']),
             s1_test_pred=clf1.predict(X['test']), s1_test_score=score(X['test']),
             s2_val_pred=clf2.predict(X['val']), s2_test_pred=clf2.predict(X['test']))
    save_json({**info, 'sklearn': sklearn.__version__}, PRED / f'{name}_info.json')
    y1, y2, _ = Y['test']
    p = np.load(PRED / f'{name}.npz')
    print(f"{name}: Stage 1 test acc = {accuracy_score(y1, p['s1_test_pred']):.4f} | "
          f"Stage 2 test acc = {accuracy_score(y2[y2 >= 0], p['s2_test_pred'][y2 >= 0]):.4f}", flush=True)


# ---- GLCM + RF and Colour-Texture + RF (fast)
Xg = {s: F[s]['glcm'] for s in F}
Xc = {s: np.hstack([F[s]['glcm'], F[s]['color_lbp']]) for s in F}
for name, X in ((('glcm_rf', Xg), ('colortex_rf', Xc)) if 'rf' in PARTS else ()):
    c1, i1 = fit_rf(X, 1, name); c2, i2 = fit_rf(X, 2, name)
    export(name, c1, c2, X, {'stage1': i1, 'stage2': i2})

# ---- HOG + SVM (slow: two stages fitted in parallel)
Xh = {s: F[s]['hog'] for s in F}
if 'svm' in PARTS:
    (c1, i1), (c2, i2) = Parallel(n_jobs=2, backend='threading')(delayed(fit_svm)(Xh, st) for st in (1, 2))
    export('hog_svm', c1, c2, Xh, {'stage1': i1, 'stage2': i2})
