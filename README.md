# Crack Detection Pipeline

High-level project overview for the Oil and Gas Crack Detection Pipeline. This project implements a multi-stage machine learning pipeline for detecting cracks, rebars, and spalls in steel and concrete infrastructure, with real-time optimization for Jetson edge platforms and API 570/579 compliance reporting.

## Pipeline Architecture

The pipeline consists of 7 sequential stages:

```
[0. Data Preparation] ──> [1. Label Derivation] ──> [2. Stage 1: CNN Gate]
                                                             │
                                                     (Crack Present?)
                                                             │
                                                  ┌──────────┴──────────┐
                                                  ▼ (Yes)               ▼ (No)
                                         [3. Stage 2: RF-DETR]       [Drop Frame]
                                                  │
                                             (Bounding Box)
                                                  │
                                                  ▼
                                         [4. Stage 3: Segmenter]
                                                  │
                                               (Mask)
                                                  │
                                                  ▼
                                         [5. Geometry Extraction]
                                                  │
                                             (Width, Length)
                                                  │
                                                  ▼
                                         [6. Severity & Alerting]
```

1. **Stage 0: Data Prep (Roboflow)**: Polygon-based instance segmentation annotations of cracks, rebars, and spalls.
2. **Stage 1: Label Derivation**: Automatic derivation of pixel masks, bounding boxes, and image-level classification flags.
3. **Stage 2: Classifier Gate (MobileNetV3)**: Edge-optimized binary gate filtering out negative frames (60-80% drop rate).
4. **Stage 3: RF-DETR Detector**: Bounding box localization using the `checkpoint_best_ema(4).pth` model.
5. **Stage 4: Segmentation (U-Net / CrackFormer)**: Pixel-level crack isolation inside bounding box crops.
6. **Stage 5: Geometry Extraction**: Crack centerline skeletonization, length and width measurements.
7. **Stage 6: Severity & Alerting**: API 570/579 fitness-for-service rule evaluation, MLflow logging, and Slack/SMS notifications.

---

## Directory Structure

```
.
├── README.md                  # High-level project overview
├── docs/                      # Centralized documentation
│   ├── index.md               # Landing page & stage overview
│   ├── inference.md           # Inference pipeline guide
│   ├── training.md            # Training configurations & guidelines
│   ├── deploy.md              # Jetson deployment instructions
│   └── config.md              # Configuration and parameter reference
│
└── src/myproj/                # Python package source code
    ├── __init__.py
    ├── inference/             # Inference and pipeline coordination
    │   └── __init__.py
    ├── training/              # Model training scripts and pipelines
    │   └── __init__.py
    ├── deploy/                # Edge optimization & deployment code
    │   └── __init__.py
    └── utils/                 # Geometry, parsing, and shared utilities
        └── __init__.py
```

## Quick Start

Refer to the [Documentation Landing Page](docs/index.md) to get started with setting up training, running inference, or deploying the pipeline.
