"""Part 9 - Final evaluation on the shared held-out test split (Week 11).

* Stage 1 and Stage 2 metrics for every method (weighted + macro averages, per-class recall)
* Stage 1 decision-threshold tuning on the validation set
* McNemar's exact test for every pair of methods (Holm-corrected)
* End-to-end two-stage pipeline evaluated on all nine classes
* All report figures
"""
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, confusion_matrix,
                             balanced_accuracy_score, roc_auc_score, roc_curve, f1_score)
from common import *

m = load_manifest()
y1, y2, y9 = labels(m, 'test')
v1, v2, _ = labels(m, 'val')
cor = y2 >= 0                                                # corroded test images (Stage 2 test set)
PRED = OUT / 'preds'
METHODS = {'hog_svm': 'HOG + SVM', 'glcm_rf': 'GLCM + RF', 'colortex_rf': 'Colour-Texture + RF',
           'cnn': 'Custom CNN', 'tl_head': 'ResNet50 (frozen)', 'tl_ft': 'ResNet50 (fine-tuned)',
           'best_fusion': 'Fusion of 3 backbones + TTA'}
P = {k: np.load(PRED / f'{k}.npz') for k in METHODS if (PRED / f'{k}.npz').exists()}
NAMES = {k: METHODS[k] for k in P}
# Fixed colour per method (colour follows the method in every figure)
COL = dict(zip(METHODS, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6', '#008300']))
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e1e0d9'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 110,
                     'savefig.dpi': 200, 'savefig.bbox': 'tight', 'axes.titlesize': 10})


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def summary(y, p):
    wp, wr, wf, _ = precision_recall_fscore_support(y, p, average='weighted', zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y, p, average='macro', zero_division=0)
    lo, hi = wilson(int((y == p).sum()), len(y))
    return {'Accuracy': accuracy_score(y, p), 'Acc CI low': lo, 'Acc CI high': hi,
            'Precision (w)': wp, 'Recall (w)': wr, 'F1 (w)': wf,
            'Precision (macro)': mp, 'Recall (macro)': mr, 'F1 (macro)': mf}


# ============================================================ Stage 1
rows = []
for k, p in P.items():
    r = {'Method': NAMES[k], **summary(y1, p['s1_test_pred'])}
    cm = confusion_matrix(y1, p['s1_test_pred'], labels=[0, 1])
    r.update({'Healthy recall': cm[0, 0] / cm[0].sum(), 'Corroded recall': cm[1, 1] / cm[1].sum(),
              'Balanced accuracy': balanced_accuracy_score(y1, p['s1_test_pred']),
              'ROC-AUC': roc_auc_score(y1, p['s1_test_score']),
              'TN (H->H)': cm[0, 0], 'FP (H->C)': cm[0, 1], 'FN (C->H)': cm[1, 0], 'TP (C->C)': cm[1, 1]})
    rows.append(r)
s1 = pd.DataFrame(rows).set_index('Method'); s1.to_csv(OUT / 'stage1_results.csv')
print('=== Stage 1 (test) ===\n', s1.round(4).to_string(), flush=True)

# ---- threshold tuning on validation (maximise macro-F1), then applied once to the test set
rows, THR = [], {}
for k, p in P.items():
    sv, st = p['s1_val_score'], p['s1_test_score']
    default = 0.0 if k == 'hog_svm' else 0.5
    cand = np.unique(np.quantile(sv, np.linspace(0.005, 0.995, 399)))
    f = np.array([f1_score(v1, (sv >= t).astype(int), average='macro') for t in cand])
    t_best = float(cand[int(np.argmax(f))])
    THR[k] = (cand, f, t_best, default)
    for tag, t in (('default', default), ('tuned', t_best)):
        pr = (st >= t).astype(int); cm = confusion_matrix(y1, pr, labels=[0, 1])
        rows.append({'Method': NAMES[k], 'Threshold': tag, 'Value': t, 'Accuracy': accuracy_score(y1, pr),
                     'F1 (macro)': f1_score(y1, pr, average='macro'), 'F1 (w)': f1_score(y1, pr, average='weighted'),
                     'Healthy recall': cm[0, 0] / cm[0].sum(), 'Corroded recall': cm[1, 1] / cm[1].sum(),
                     'Val F1 (macro)': f1_score(v1, (sv >= t).astype(int), average='macro')})
thr = pd.DataFrame(rows); thr.to_csv(OUT / 'stage1_threshold.csv', index=False)
print('\n=== Stage 1 threshold tuning ===\n', thr.round(4).to_string(index=False), flush=True)

# ============================================================ Stage 2
rows, rec = [], {}
for k, p in P.items():
    pr = p['s2_test_pred'][cor]
    rows.append({'Method': NAMES[k], **summary(y2[cor], pr)})
    cm = confusion_matrix(y2[cor], pr, labels=range(8)); rec[NAMES[k]] = cm.diagonal() / cm.sum(1)
s2 = pd.DataFrame(rows).set_index('Method'); s2.to_csv(OUT / 'stage2_results.csv')
rec = pd.DataFrame(rec, index=S2_NAMES).T; rec.to_csv(OUT / 'stage2_per_class_recall.csv')
print('\n=== Stage 2 (test, corroded images) ===\n', s2.round(4).to_string(),
      '\n\nPer-class recall:\n', rec.round(3).to_string(), flush=True)

# ============================================================ End-to-end (nine classes)
rows, E2E = [], {}
for k, p in P.items():
    pred9 = np.where(p['s1_test_pred'] == 0, 0, p['s2_test_pred'] + 1)
    E2E[k] = pred9
    r = {'Method': NAMES[k], **summary(y9, pred9)}
    # how the errors split between the two stages
    r['Stage-1 errors'] = int((p['s1_test_pred'] != y1).sum())
    r['Stage-2 errors (after correct Stage 1)'] = int(((p['s1_test_pred'] == y1) & (pred9 != y9)).sum())
    rows.append(r)
e2e = pd.DataFrame(rows).set_index('Method'); e2e.to_csv(OUT / 'end_to_end_results.csv')
print('\n=== End-to-end, 9 classes ===\n', e2e.round(4).to_string(), flush=True)


# ============================================================ McNemar (exact, two-sided) + Holm
def mcnemar_table(correct, tag):
    rows = []
    for a, b in itertools.combinations(correct, 2):
        n01 = int((correct[a] & ~correct[b]).sum()); n10 = int((~correct[a] & correct[b]).sum())
        pval = 1.0 if n01 + n10 == 0 else binomtest(n01, n01 + n10, 0.5).pvalue
        rows.append({'Comparison': tag, 'Method A': NAMES[a], 'Method B': NAMES[b],
                     'A right, B wrong': n01, 'A wrong, B right': n10, 'p': pval})
    d = pd.DataFrame(rows).sort_values('p').reset_index(drop=True)
    n = len(d); adj = np.maximum.accumulate([min(1.0, pv * (n - i)) for i, pv in enumerate(d['p'])])
    d['p (Holm)'] = adj; d['Significant (0.05)'] = d['p (Holm)'] < 0.05
    return d


mc = pd.concat([mcnemar_table({k: p['s1_test_pred'] == y1 for k, p in P.items()}, 'Stage 1'),
                mcnemar_table({k: p['s2_test_pred'][cor] == y2[cor] for k, p in P.items()}, 'Stage 2'),
                mcnemar_table({k: E2E[k] == y9 for k in P}, 'End-to-end')])
mc.to_csv(OUT / 'mcnemar.csv', index=False)
pd.set_option('display.width', 250)
print('\n=== McNemar exact tests ===\n', mc.to_string(index=False), flush=True)


# ============================================================ Figures
def draw_cm(ax, y, p, names, title, small=False):
    cm = confusion_matrix(y, p, labels=range(len(names))); cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40 if len(names) > 2 else 0, ha='right' if len(names) > 2 else 'center')
    ax.set_yticklabels(names); ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    for i in range(len(names)):
        for j in range(len(names)):
            txt = f'{cm[i, j]}' if small else f'{cm[i, j]}\n{cmn[i, j] * 100:.0f}%'
            ax.text(j, i, txt, ha='center', va='center', fontsize=6.5 if small else 9,
                    color='white' if cmn[i, j] > 0.5 else INK)


keys = list(P)
# Fig: Stage 1 confusion matrices
fig, axes = plt.subplots(1, len(keys), figsize=(2.7 * len(keys), 3.0))
for ax, k in zip(axes, keys):
    draw_cm(ax, y1, P[k]['s1_test_pred'], S1_NAMES, NAMES[k])
    if k != keys[0]: ax.set_ylabel('')
fig.tight_layout(); fig.savefig(FIG / 'stage1_confusion.png'); plt.close(fig)

# Fig: Stage 2 confusion matrices
fig, axes = plt.subplots(2, 4, figsize=(20, 9.6))
for ax, k in zip(axes.flat, keys):
    draw_cm(ax, y2[cor], P[k]['s2_test_pred'][cor], S2_NAMES,
            f"{NAMES[k]}  (accuracy {accuracy_score(y2[cor], P[k]['s2_test_pred'][cor]) * 100:.1f}%)", small=True)
for ax in axes.flat[len(keys):]: ax.axis('off')
fig.tight_layout(); fig.savefig(FIG / 'stage2_confusion.png'); plt.close(fig)

# Fig: overall comparison (one measure per panel, one axis each)
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
panels = [('Stage 1: Healthy recall', s1['Healthy recall'], None),
          ('Stage 2: accuracy (8 corrosion types)', s2['Accuracy'], 1 / 8),
          ('End-to-end: accuracy (9 classes)', e2e['Accuracy'], None)]
for ax, (title, ser, chance) in zip(axes, panels):
    yy = np.arange(len(ser))[::-1]
    ax.barh(yy, ser.values, height=0.62, color=[COL[k] for k in keys])
    for yv, v in zip(yy, ser.values):
        ax.text(v + 0.012, yv, f'{v * 100:.1f}%', va='center', fontsize=9, color=INK)
    ax.set_yticks(yy); ax.set_yticklabels(ser.index if ax is axes[0] else [])
    ax.set_xlim(0, 1.12); ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(['0', '25%', '50%', '75%', '100%'])
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=0)
    ax.set_title(title, loc='left')
    if chance:
        ax.axvline(chance, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.text(chance + 0.012, -0.47, 'chance 12.5%', fontsize=7.5, color=MUTED)
fig.tight_layout(); fig.savefig(FIG / 'comparison.png'); plt.close(fig)

# Fig: per-class recall heat-map (Stage 2)
fig, ax = plt.subplots(figsize=(8.6, 3.2))
im = ax.imshow(rec.values, cmap='Blues', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(8)); ax.set_xticklabels(S2_NAMES, rotation=30, ha='right')
ax.set_yticks(range(len(rec))); ax.set_yticklabels(rec.index); ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
for i in range(rec.shape[0]):
    for j in range(8):
        ax.text(j, i, f'{rec.values[i, j] * 100:.0f}', ha='center', va='center', fontsize=8.5,
                color='white' if rec.values[i, j] > 0.5 else INK)
ax.set_title('Stage 2 recall per corrosion type (%) on the test set', loc='left')
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02).outline.set_visible(False)
fig.tight_layout(); fig.savefig(FIG / 'stage2_per_class_recall.png'); plt.close(fig)

# Fig: ROC (Stage 1) and threshold tuning
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
for k in keys:
    fpr, tpr, _ = roc_curve(1 - y1, -P[k]['s1_test_score'])          # positive class = Healthy (minority)
    axes[0].plot(fpr, tpr, color=COL[k], lw=2, label=f"{NAMES[k]} (AUC {s1.loc[NAMES[k], 'ROC-AUC']:.3f})")
axes[0].plot([0, 1], [0, 1], color=GRID, lw=1)
axes[0].set(xlabel='False-positive rate', ylabel='True-positive rate (Healthy detected)', xlim=(0, 1), ylim=(0, 1.01))
axes[0].set_title('Stage 1 ROC curves on the test set', loc='left'); axes[0].legend(frameon=False, fontsize=8, loc='lower right')
deep = [k for k in ('cnn', 'tl_head', 'tl_ft', 'best_fusion') if k in P]
for k in deep:
    cand, f, t_best, default = THR[k]
    axes[1].plot(cand, f, color=COL[k], lw=2, label=f'{NAMES[k]} (best {t_best:.2f})')
    axes[1].scatter([t_best], [f.max()], s=42, color=COL[k], edgecolor='white', linewidth=1.5, zorder=3)
axes[1].axvline(0.5, color=MUTED, lw=1, ls=(0, (3, 3))); axes[1].text(0.51, 0.52, 'default 0.5', fontsize=7.5, color=MUTED)
axes[1].set(xlabel='Decision threshold on P(Corroded)', ylabel='Validation macro-F1', xlim=(0, 1), ylim=(0.5, 1.0))
axes[1].set_title('Stage 1 threshold tuning on the validation set', loc='left'); axes[1].legend(frameon=False, fontsize=8, loc='lower left')
for ax in axes: ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(FIG / 'roc_threshold.png'); plt.close(fig)

# Fig: training curves of the new deep models
curves = [('cnn_stage2_history.csv', 'Custom CNN, Stage 2'), ('tl_ft_stage1_history.csv', 'ResNet50 fine-tuning, Stage 1'),
          ('tl_ft_stage2_history.csv', 'ResNet50 fine-tuning, Stage 2')]
curves = [(f, t) for f, t in curves if (OUT / f).exists()]
if curves:
    fig, axes = plt.subplots(2, len(curves), figsize=(4.4 * len(curves), 5.6), squeeze=False)
    for j, (f, title) in enumerate(curves):
        h = pd.read_csv(OUT / f); ep = np.arange(1, len(h) + 1)
        for i, (a, b, lab) in enumerate((('accuracy', 'val_accuracy', 'Accuracy'), ('loss', 'val_loss', 'Loss'))):
            ax = axes[i, j]
            ax.plot(ep, h[a], color='#2a78d6', lw=2, label='Train'); ax.plot(ep, h[b], color='#eb6834', lw=2, label='Validation')
            ax.set(xlabel='Epoch', ylabel=lab); ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            if i == 0: ax.set_title(title, loc='left')
            if i == 0 and j == 0: ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / 'training_curves.png'); plt.close(fig)

# Fig: end-to-end confusion matrix of the best pipeline
best = e2e['Accuracy'].idxmax(); bk = [k for k in P if NAMES[k] == best][0]
fig, ax = plt.subplots(figsize=(7.4, 6.4))
draw_cm(ax, y9, E2E[bk], ALL9_NAMES, f'End-to-end pipeline, {best} (nine classes, test set)', small=True)
fig.tight_layout(); fig.savefig(FIG / 'end_to_end_confusion.png'); plt.close(fig)

# Fig: error examples of the best pipeline
imgs = load_images('test'); pb = P[bk]
groups = [('Healthy predicted as Corroded', np.where((y1 == 0) & (pb['s1_test_pred'] == 1))[0]),
          ('Corroded predicted as Healthy', np.where((y1 == 1) & (pb['s1_test_pred'] == 0))[0]),
          ('Wrong corrosion type', np.where(cor & (pb['s1_test_pred'] == 1) & (pb['s2_test_pred'] != y2))[0])]
rng = np.random.default_rng(SEED)
fig, axes = plt.subplots(3, 6, figsize=(13, 7.2))
for r, (title, idx) in enumerate(groups):
    pick = rng.choice(idx, size=min(6, len(idx)), replace=False) if len(idx) else []
    for c in range(6):
        ax = axes[r, c]; ax.axis('off')
        if c < len(pick):
            i = pick[c]; ax.imshow(imgs[i])
            ax.set_title(f'true: {ALL9_NAMES[y9[i]]}\npred: {ALL9_NAMES[E2E[bk][i]]}', fontsize=8)
    axes[r, 0].text(-0.08, 0.5, f'{title}\n(n = {len(idx)})', transform=axes[r, 0].transAxes, ha='right', va='center', fontsize=9)
fig.suptitle(f'Examples of test errors, {best}', x=0.02, ha='left', fontsize=11)
fig.tight_layout(); fig.savefig(FIG / 'error_examples.png'); plt.close(fig)
print('\nBest end-to-end pipeline:', best, '| figures saved to', FIG, flush=True)
