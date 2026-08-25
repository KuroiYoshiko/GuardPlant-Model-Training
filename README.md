# GuardPlant Model Training

Training, evaluation, and export pipeline for the GuardPlant Android application. GuardPlant performs offline plant identification across 70 species.

The project provides a Google Colab notebook that trains and compares exactly two image-classification models:

- Ultralytics YOLO26 Nano Classifier (`yolo26n-cls.pt`)
- Torchvision EfficientNet-B0 with `EfficientNet_B0_Weights.DEFAULT`

## Project goals

The benchmark is designed to produce reproducible evidence for selecting a mobile plant-identification model. It covers:

- Dataset integrity and class-index validation
- Pretrained model fine-tuning with early stopping
- Held-out test-set evaluation
- Top-1 and Top-5 accuracy
- Per-class precision, recall, and F1 score
- Confusion analysis and training diagnostics
- Development-environment latency and checkpoint-size comparison
- ONNX and LiteRT/TFLite export for future Android deployment
- Persistent artifact backup to Google Drive

## Repository contents

```text
GuardPlant-Model-Training/
├── guardplant_model_benchmark.ipynb  # Complete Colab workflow
├── PROMPT_LOG.md                      # Development and prompt history
└── README.md                          # Project documentation
```

Generated checkpoints, metrics, plots, and exports are intentionally created when the notebook runs and are not required in the repository.

## Dataset

The current dataset contains approximately 28,000 images across 70 plant species, with roughly 400 images per species before splitting.

It must already be divided into 80% training, 10% validation, and 10% test data using an ImageFolder-compatible structure:

```text
dataset/
├── train/
│   ├── <plant_id>/
│   └── ...
├── val/
│   ├── <plant_id>/
│   └── ...
└── test/
    ├── <plant_id>/
    └── ...
```

Each split must contain exactly the same 70 non-empty class directories. Directory names are sorted alphabetically to create the canonical numeric label mapping in `class_names.json`.

The notebook expects the compressed dataset at:

```text
/content/drive/MyDrive/dataset.zip
```

The ZIP may contain `train`, `val`, and `test` directly or place them inside one top-level `dataset` directory.

## Benchmark configuration

The main configuration is defined in a single notebook cell:

```python
SEED = 42
IMAGE_SIZE = 224
MAX_EPOCHS = 100
PATIENCE = 15
EXPECTED_NUM_CLASSES = 70

YOLO_BATCH = -1
EFFICIENTNET_BATCH_SIZE = 64
```

`YOLO_BATCH = -1` lets Ultralytics choose a batch size based on available GPU memory. EfficientNet uses CUDA automatic mixed precision when CUDA is available.

## Running in Google Colab

1. Upload `dataset.zip` to the root of Google Drive.
2. Open `guardplant_model_benchmark.ipynb` in Google Colab.
3. Select a GPU runtime from **Runtime → Change runtime type**.
4. Run the notebook cells in order.
5. Authorize Google Drive access when prompted.

The notebook installs its Python dependencies, mounts Drive, extracts the dataset, validates every split, and builds the training pipelines automatically.

Training can take a substantial amount of time. Colab GPU availability and runtime limits vary, so best checkpoints and histories are copied to Google Drive as soon as training completes.

## Training or loading saved models

The notebook supports fresh training and later evaluation sessions through two workflow flags:

```python
TRAIN_YOLO26 = True
TRAIN_EFFICIENTNET = True
```

Leave a flag set to `True` to train that model. Set it to `False` to load the corresponding checkpoint from Google Drive:

```text
/content/drive/MyDrive/GuardPlant_Artifacts/yolo26n_best.pt
/content/drive/MyDrive/GuardPlant_Artifacts/efficientnet_b0_best.pth
```

The loading cells restore the same model and history variables used by evaluation, plotting, benchmarking, and export. This allows a new Colab session to continue without retraining.

## Training details

### YOLO26n-cls

- ImageNet-pretrained `yolo26n-cls.pt`
- Maximum 100 epochs
- Early-stopping patience of 15 epochs
- 224×224 input size
- Automatic batch sizing and optimizer selection
- Deterministic seed configuration
- Best checkpoint saved as `yolo26n_best.pt`

### EfficientNet-B0

- ImageNet-pretrained `EfficientNet_B0_Weights.DEFAULT`
- 70-output classification head
- Cross-entropy loss
- AdamW optimizer
- Cosine annealing learning-rate schedule
- CUDA autocast and gradient scaling when available
- Early stopping based only on validation Top-1 accuracy
- Best checkpoint saved as `efficientnet_b0_best.pth`

The held-out test split is never used for early stopping or model selection.

## Fair evaluation

Both models are evaluated against one canonical ordered manifest of test image paths and labels. Each model still receives its correct native preprocessing:

- YOLO uses the Ultralytics classification prediction pipeline.
- EfficientNet uses the evaluation transform supplied by its pretrained Torchvision weights.

The notebook does not pass EfficientNet-normalized tensors into YOLO or attempt to reverse one model's preprocessing for the other.

Development latency includes image decoding, native preprocessing, and batch-size-one inference in the active Colab runtime. It is useful for relative comparison but does not represent Android device performance.

## Generated outputs

### Checkpoints and histories

```text
yolo26n_best.pt
efficientnet_b0_best.pth
guardplant_artifacts/yolo26n_history.csv
guardplant_artifacts/yolo26n_metadata.json
guardplant_artifacts/efficientnet_b0_history.json
```

### Metrics

```text
benchmark_results.csv
dataset_stats.csv
per_class_metrics.csv
top_confusions.csv
class_names.json
```

`benchmark_results.csv` includes test Top-1, test Top-5, best validation Top-1, best epoch, parameter count, checkpoint size, and clearly labelled Colab development latency.

### Plots

```text
plots/training_curves.png
plots/yolo26n_confusion_matrix.png
plots/efficientnet_b0_confusion_matrix.png
plots/worst_classes_by_recall.png
```

The confusion matrices are full row-normalized 70×70 matrices. The worst-class report ranks the 15 lowest-recall classes independently for each model.

### Production exports

```text
yolo26n_guardplant.onnx
efficientnet_b0_guardplant.onnx
yolo26n_guardplant.tflite
```

Both ONNX graphs are validated after export. The YOLO LiteRT/TFLite export is attempted when supported by the active Colab runtime.

## Google Drive backup

Important artifacts are copied to:

```text
/content/drive/MyDrive/GuardPlant_Artifacts/
```

This backup includes checkpoints, histories, class mapping, CSV reports, ONNX exports, a supported LiteRT/TFLite export, and important diagnostic plots.

## Reproducibility notes

- Dataset folder names define the canonical class IDs.
- The same alphabetical class order is checked across all splits and saved with exported artifacts.
- Random number generators use seed 42.
- Deterministic execution is requested where supported.
- GPU kernels and Colab hardware can still introduce small numerical or latency differences between sessions.
- Final mobile performance must be measured on representative Android hardware using the intended inference runtime.

