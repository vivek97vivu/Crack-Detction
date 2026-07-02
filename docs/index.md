# Crack Detection Pipeline Documentation

Welcome to the documentation for the multi-stage Crack Detection and Severity Alerting system. This documentation is organized into the following sections:

## Documentation Sections

- **[Inference Pipeline](inference.md)**: Details on running the end-to-end pipeline, processing inputs, cascading inference models, and post-processing results.
- **[Model Training](training.md)**: Configuration and scripts for training the MobileNetV3 gate classifier, RF-DETR detector, and CrackFormer/U-Net segmenter.
- **[Edge Deployment](deploy.md)**: Performance tuning, TensorRT export, and Jetson-specific throughput optimization.
- **[Configuration Guide](config.md)**: API parameters, severity levels, and threshold configurations.

---

## Stage-by-Stage Reference

### 0. Data Prep (Roboflow)
* **Objective**: Instance segmentation annotations (polygons) on Roboflow project "Crack".
* **Classes**: `crack`, `rebar`, `spall` (3 classes total).
* **Hard Negatives**: Include weld seams, scratches, rust patches, and paint cracks to minimize false positives.

### 1. Label Derivation
* **Objective**: Automate label derivation to prevent re-annotating.
* **Outputs**: Image classifier flags (crack present vs. not), bounding boxes for RF-DETR, and pixel masks for U-Net/CrackFormer.

### 2. Stage 1: CNN Classifier Gate
* **Objective**: Filter frames without crack signals to optimize Jetson throughput.
* **Architecture**: MobileNetV3-Small (lightweight binary classifier).
* **Gate Threshold**: `P(crack) > 0.4` (adjustable depending on target validation recall).

### 3. Stage 2: RF-DETR Detection
* **Objective**: Localize instances of crack, rebar, and spall.
* **Model**: RF-DETR Base with `dinov2_windowed_small` encoder.
* **Weights**: Localized at `checkpoint_best_ema(4).pth`.

### 4. Stage 3: Segmentation
* **Objective**: Crop ROIs from detector bounding boxes and run pixel-level segmentation.
* **Loss Function**: Combined Dice + BCE loss to handle extreme class imbalance (hairline cracks).

### 5. Geometry Post-Processing
* **Objective**: Extract crack skeleton centerline and measure length and width.
* **Measurements**: Max distance transform for width; path length summation for length.
* **blob rejection**: Reject circular/isolated blobs (e.g. circles are not cracks).

### 6. Severity & Alerting
* **Objective**: Map width and length measurements to API 570/579 severity classes.
* **Actions**: Level 1 (Monitor), Level 2 (Schedule Repair), Level 3 (Immediate Alert).
* **alert stack**: Send notifications via Slack/SMS/Email with cooldown deduplication.
