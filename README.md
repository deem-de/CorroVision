# CorroVision - final pipeline (Weeks 9-11)

Code that completes the project after the progress report: transfer learning, the Stage 2 CNN,
the final evaluation (McNemar's test, threshold tuning, end-to-end pipeline) and all report figures.

## Files

| File | What it does |
| --- | --- |
| `CorroVision_Final_Pipeline.ipynb` | Colab notebook that contains and runs every script below, top to bottom |
| `code/common.py` | Shared configuration + the unchanged Part 2 preprocessing / augmentation / HOG / GLCM functions |
| `code/01_regenerate.py` | Rebuilds processed images and features from `processed_images.csv`; checks them against the retained `features_test.npz` |
| `code/02_classical.py` | HOG + SVM, GLCM + Random Forest, Colour-Texture + Random Forest (extension) |
| `code/03_cnn.py` | Scores the Part 5 baseline CNN (Stage 1); trains the deeper CNN (Stage 2) |
| `code/04_transfer.py` | ResNet50 transfer learning: phase 1 (frozen) and phase 2 (fine-tune conv5), both stages |
| `code/07_backbones.py` | ConvNeXt-Tiny / EfficientNetV2-S features, feature fusion and test-time augmentation (run before `05_evaluate.py`) |
| `code/05_evaluate.py` | Metrics, threshold tuning, McNemar's tests, end-to-end pipeline, result figures |
| `code/06_data_figures.py` | Data figures (class examples, distribution, preprocessing, augmentation) |
| `code/build_notebook.py` | Regenerates the notebook from the scripts |
| `results/` | Result tables (`*.csv`) and figures from the reported run |
| `outputs/` | Created when you run the code: predictions, figures, trained models (not tracked in git) |

## Run locally

```bash
pip install kagglehub opencv-python-headless scikit-image "scikit-learn==1.6.1" joblib scipy tensorflow matplotlib pandas
export CORRO_ROOT=$PWD            # folder that contains inputs/ and code/
export CORRO_DATA=/path/to/kaggle/marine-corrosion-dataset
cd code
python 01_regenerate.py $CORRO_DATA
python 06_data_figures.py $CORRO_DATA
python 02_classical.py
python 03_cnn.py
python 04_transfer.py
python 07_backbones.py resnet50 effnetv2_s convnext_tiny
python 05_evaluate.py
```

`inputs/` must contain `processed_images.csv`, `features_test.npz` and `part5/best_cnn_stage1.keras`
(+ `part5/cnn_test_predictions.csv`).

## Notes

* **scikit-learn version.** Random Forest results depend on the scikit-learn version. Version 1.6.1
  reproduces the progress-report GLCM numbers exactly (0.9281 / 0.3536); newer versions give slightly
  different forests. The SVM is unaffected.
* **Rebuild is exact.** `01_regenerate.py` reads the image index that seeded each augmentation from the
  processed file name, so the rebuilt images are identical to Part 2. The rebuilt test features equal
  `features_test.npz` bit for bit (`outputs/rebuild_check.json`).
* **Dataset:** https://www.kaggle.com/datasets/ebrahim007/marine-corrosion-dataset
* **Stage 2 CNN.** Training was stopped manually after epoch 16 of 40 (CPU time); the reported result uses the best saved
  checkpoint. `CNN_EVAL_ONLY=1 python 03_cnn.py` scores a saved checkpoint without retraining.
* **CPU run.** Everything here was produced on a 4-core CPU without a GPU.
