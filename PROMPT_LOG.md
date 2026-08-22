# Prompt Log

## 2026-08-22 — Step 1: Environment Setup & Data Pipeline

- **Assistant model:** OpenAI Codex (GPT-5)
- **Benchmark models:** Ultralytics YOLOv8 Nano Classifier (`yolov8n-cls`) and Torchvision EfficientNet-B0
- **Requested role:** Senior Machine Learning Engineer specializing in PyTorch, Ultralytics YOLO, and Google Colab workflows
- **Deliverable:** `guardplant_model_benchmark.ipynb`
- **Summary:** Created the GuardPlant benchmark introduction and Colab environment setup; added Google Drive mounting and idempotent extraction of `dataset.zip` into `/content/dataset/`; validated the train, validation, and test ImageFolder splits; enforced a shared set of exactly 200 non-empty plant ID classes; saved the canonical alphabetical label order to `class_names.json`; configured ImageNet-normalized 224×224 PyTorch datasets and data loaders with light training augmentation; and added a one-batch pipeline smoke test. The validated dataset root is also exposed for the later YOLOv8 classification step.

## 2026-08-22 — Step 2: Model Training Loops

- **Assistant model:** OpenAI Codex (GPT-5)
- **Benchmark models:** Ultralytics YOLOv8 Nano Classifier (`yolov8n-cls.pt`) and pretrained Torchvision EfficientNet-B0
- **Requested role:** Senior Machine Learning Engineer
- **Deliverable updated:** `guardplant_model_benchmark.ipynb`
- **Training configuration:** 20 epochs per model, 224×224 inputs, shared 200-class label mapping, GPU when available
- **Summary:** Added Ultralytics Python API training with native run artifacts and copying of the best validation weights to `yolov8_best.pt`; verified the YOLO checkpoint label order; initialized ImageNet-pretrained EfficientNet-B0 with a 200-class output head; configured cross-entropy loss, AdamW (`lr=1e-3`), and cosine annealing; implemented native PyTorch training and validation passes with sample-weighted loss and Top-1/Top-5 accuracy; persisted epoch history for later plots; and saved the best validation Top-1 checkpoint to `efficientnet_best.pth`.

## 2026-08-22 — Step 3: Evaluation, Comparative Benchmarking, Visualizations & Export

- **Assistant model:** OpenAI Codex (GPT-5)
- **Evaluated checkpoints:** `yolov8_best.pt` and `efficientnet_best.pth`
- **Evaluation protocol:** Complete held-out `test_loader`; Top-1/Top-5 accuracy; warmed-up batch-size-1 forward latency with CUDA synchronization; checkpoint file size
- **Outputs:** `benchmark_results.csv`, `plots/training_curves.png`, `plots/confusion_matrix.png`, `yolov8n_guardplant.onnx`, and `efficientnet_b0_guardplant.onnx`
- **Summary:** Added fresh best-checkpoint loading and common-test-loader evaluation for both models; generated a Markdown/Pandas comparison table and CSV; added comparative training curves from Ultralytics and EfficientNet histories; created row-normalized side-by-side confusion matrices for the 15 most-supported test classes plus an `Other` bucket; exported and validated both ONNX graphs; and added Google Drive backup to `/content/drive/MyDrive/GuardPlant_Artifacts/` for both ONNX models, `class_names.json`, and `benchmark_results.csv`.
