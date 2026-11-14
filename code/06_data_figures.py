"""Report figures that describe the data: class examples, class distribution, preprocessing steps
and augmentation examples (same drawing code as Parts 1-2)."""
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *

DATA_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ['CORRO_DATA'])
m = load_manifest()
orig = m[~m['is_aug']]
plt.rcParams.update({'font.size': 9, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
                     'axes.spines.top': False, 'axes.spines.right': False})
classes = sorted(orig['class'].unique())

# ---- one example per class
fig, axes = plt.subplots(1, 9, figsize=(18, 2.6))
for ax, c in zip(axes, classes):
    rp = orig[orig['class'] == c].sample(1, random_state=7)['rel_path'].iloc[0]
    ax.imshow(resize(load_rgb(DATA_ROOT / rp))); ax.set_title(nice(c), fontsize=11); ax.axis('off')
fig.tight_layout(); fig.savefig(FIG / 'class_examples.png'); plt.close(fig)

# ---- class distribution after cleaning
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={'width_ratios': [1, 3]})
s1 = orig['stage1'].value_counts()[['Healthy', 'Corroded']]
axes[0].bar(s1.index, s1.values, color=['#1baf7a', '#2a78d6'], width=0.6)
axes[0].set_title('Stage 1: Healthy vs. Corroded', loc='left'); axes[0].set_ylabel('Number of images')
for i, v in enumerate(s1.values): axes[0].text(i, v + 90, f'{v:,}', ha='center')
s2 = orig['class'].value_counts().sort_index()
axes[1].bar([nice(c) for c in s2.index], s2.values, width=0.7,
            color=['#1baf7a' if c == HEALTHY else '#2a78d6' for c in s2.index])
axes[1].set_title('All nine classes (Healthy in green)', loc='left'); axes[1].set_ylim(0, 1150)
axes[1].tick_params(axis='x', rotation=30)
for i, v in enumerate(s2.values): axes[1].text(i, v + 15, str(v), ha='center', fontsize=8)
fig.tight_layout(); fig.savefig(FIG / 'class_distribution.png'); plt.close(fig)

# ---- preprocessing steps
picked = [classes[i] for i in np.linspace(0, len(classes) - 1, 4).astype(int)]
titles = ['Original', 'Resized 224x224', 'Denoised (bilateral)', 'Colour corrected', 'Contrast enhanced (final)']
fig, axes = plt.subplots(len(picked), 5, figsize=(13, 2.7 * len(picked)))
for r, c in enumerate(picked):
    row = orig[orig['class'] == c].sample(1, random_state=SEED).iloc[0]
    o = load_rgb(DATA_ROOT / row['rel_path'])
    a = resize(o); b = denoise(a); d = color_correct(b); e = enhance_contrast(d)
    for k, im in enumerate([o, a, b, d, e]):
        axes[r, k].imshow(im); axes[r, k].axis('off')
        if r == 0: axes[r, k].set_title(titles[k], fontsize=10)
    axes[r, 0].text(-0.05, 0.5, nice(c), transform=axes[r, 0].transAxes, ha='right', va='center', fontsize=9)
fig.tight_layout(); fig.savefig(FIG / 'preprocessing_steps.png'); plt.close(fig)

# ---- augmentation examples (processed image and its stored augmented copy)
tr = split_frame(m, 'train'); imgs = load_images('train')
pairs = tr[tr['is_aug']].sample(5, random_state=SEED)
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for j, (_, r) in enumerate(pairs.iterrows()):
    i_aug = r.name
    i_org = tr.index[(tr['rel_path'] == r['rel_path']) & (~tr['is_aug'])][0]
    axes[0, j].imshow(imgs[i_org]); axes[1, j].imshow(imgs[i_aug])
    axes[0, j].set_title(nice(r['class']), fontsize=10)
    for a in axes[:, j]: a.axis('off')
axes[0, 0].text(-0.05, 0.5, 'Processed', transform=axes[0, 0].transAxes, ha='right', va='center')
axes[1, 0].text(-0.05, 0.5, 'Augmented', transform=axes[1, 0].transAxes, ha='right', va='center')
fig.tight_layout(); fig.savefig(FIG / 'augmentation_examples.png'); plt.close(fig)
print('saved data figures to', FIG)
