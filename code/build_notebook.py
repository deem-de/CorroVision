"""Builds CorroVision_Final_Pipeline.ipynb (Colab) from the scripts in this folder, so the notebook
and the scripts that produced the report results can never drift apart."""
import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).parent
nb = nbf.v4.new_notebook()
md = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: nb.cells.append(nbf.v4.new_code_cell(s))

md("""# CorroVision - Final Pipeline (Weeks 9-11)
### Marine corrosion detection (Stage 1) and corrosion-type classification (Stage 2)

Continues from Parts 1-5. It uses the shared `processed_images.csv` (same 70/15/15 split) and:

1. rebuilds the processed images and the HOG / GLCM features, and proves they match the retained `features_test.npz`
2. retrains HOG + SVM and GLCM + Random Forest, and adds the Colour-Texture extension
3. scores the Part 5 baseline CNN (Stage 1) and trains a deeper CNN for Stage 2
4. transfer learning with ResNet50: phase 1 (frozen backbone) and phase 2 (fine-tuning conv5); then ConvNeXt / EfficientNetV2 features, fusion and TTA
5. final evaluation: metrics, threshold tuning, McNemar's test, end-to-end pipeline, all figures

**Run the cells top to bottom.** A GPU runtime is recommended for steps 3-4 (they also run on CPU; about 2.5 hours).
Put these files in one Drive folder and set `INPUT_DIR` below: `processed_images.csv`, `features_test.npz`,
and a `part5/` folder with `best_cnn_stage1.keras` and `cnn_test_predictions.csv`.""")

code("""# scikit-learn is pinned: Random Forest results change between versions, and 1.6.1 is the
# version that produced the progress-report numbers.
!pip -q install kagglehub opencv-python-headless scikit-image "scikit-learn==1.6.1" joblib scipy

import os, shutil
from pathlib import Path
from google.colab import drive
drive.mount('/content/drive')

INPUT_DIR = Path('/content/drive/MyDrive/corrovision_inputs')     # <- change if needed
ROOT = Path('/content/corrovision')
(ROOT / 'code').mkdir(parents=True, exist_ok=True)
if not (ROOT / 'inputs').exists():
    shutil.copytree(INPUT_DIR, ROOT / 'inputs')
os.environ['CORRO_ROOT'] = str(ROOT)

import kagglehub
os.environ['CORRO_DATA'] = kagglehub.dataset_download('ebrahim007/marine-corrosion-dataset')
print('Dataset:', os.environ['CORRO_DATA'])
%cd /content/corrovision/code""")

steps = [('common.py', None, 'Shared configuration and the unchanged Part 2 functions'),
         ('01_regenerate.py', '01_regenerate.py', '1. Rebuild processed images and features'),
         ('06_data_figures.py', '06_data_figures.py', 'Data figures'),
         ('02_classical.py', '02_classical.py', '2. Classical methods'),
         ('03_cnn.py', '03_cnn.py', '3. Custom CNNs'),
         ('04_transfer.py', '04_transfer.py', '4. Transfer learning (ResNet50)'),
         ('07_backbones.py', '07_backbones.py resnet50 effnetv2_s convnext_tiny', '4b. Stronger backbones, feature fusion and TTA'),
         ('05_evaluate.py', '05_evaluate.py', '5. Final evaluation and figures')]
import re, base64
LOGS = HERE.parent / 'logs'; FIGS = HERE.parent / 'outputs' / 'figures'
LOG_OF = {'01_regenerate.py': ['01.log'], '02_classical.py': ['02_rf.log', '02_svm.log'], '03_cnn.py': ['03.log', '03_eval_tail.log'],
          '04_transfer.py': ['04.log'], '07_backbones.py resnet50 effnetv2_s convnext_tiny': ['07.log'], '05_evaluate.py': ['05.log'], '06_data_figures.py': []}
NOISE = re.compile(r'I0000|E0000|W0000|WARNING|Parallel\(|^\s*$|UserWarning|axes\[1\]\.set_title|\x1b|━|cuda|TF-TRT|oneDNN|rebuild TensorFlow|AVX')


def real_output(run):
    """Outputs recorded when these exact scripts were run (attached so the notebook opens with results)."""
    text = ''
    for f in LOG_OF.get(run, []):
        if (LOGS / f).exists():
            text += ''.join(l for l in (LOGS / f).read_text(errors='ignore').splitlines(True) if not NOISE.search(l))
    return [nbf.v4.new_output('stream', name='stdout', text=text)] if text.strip() else []


for fname, run, title in steps:
    md(f'## {title}')
    code(f'%%writefile {fname}\n' + (HERE / fname).read_text())
    if run:
        code(f'!python {run}')
        nb.cells[-1].outputs = real_output(run)

md('## Save results to Drive')
code("""SAVE = Path('/content/drive/MyDrive/CorroVision_Final_Results')
shutil.copytree(ROOT / 'outputs', SAVE, dirs_exist_ok=True)
print('Saved to', SAVE)
from IPython.display import Image, display
for f in sorted((ROOT / 'outputs' / 'figures').glob('*.png')):
    print(f.name); display(Image(str(f), width=900))""")

if FIGS.exists():
    outs = []
    for f in sorted(FIGS.glob('*.png')):
        outs.append(nbf.v4.new_output('stream', name='stdout', text=f.name + '\n'))
        outs.append(nbf.v4.new_output('display_data', data={'image/png': base64.b64encode(f.read_bytes()).decode()},
                                      metadata={'image/png': {'width': 900}}))
    nb.cells[-1].outputs = outs
out = HERE.parent / 'CorroVision_Final_Pipeline.ipynb'
nbf.write(nb, out)
print('wrote', out, len(nb.cells), 'cells')
