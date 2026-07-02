# Inference Pipeline

This document explains the end-to-end inference flow for the Crack Detection Pipeline.

## Pipeline Workflow

```
Input Video/Frame
       │
       ▼
┌──────────────────────────────┐
│  MobileNetV3 Classifier Gate │
└──────────────┬───────────────┘
               │
         P(crack) > 0.4?
               ├───────────────────────┐
            (Yes)                     (No)
               ▼                       ▼
┌──────────────────────────────┐  ┌──────────────┐
│  RF-DETR Detector (3 Class)   │  │  Drop Frame  │
└──────────────┬───────────────┘  └──────────────┘
               │
         Cropped Bounding Boxes
               ▼
┌──────────────────────────────┐
│   Segmentation Network       │
└──────────────┬───────────────┘
               │
          Binary Mask
               ▼
┌──────────────────────────────┐
│     Geometry Extraction      │
└──────────────┬───────────────┘
               │
       Width & Length (mm)
               ▼
┌──────────────────────────────┐
│    API 570/579 Severity      │
└──────────────────────────────┘
```

## Running Inference

To run the pipeline on a source video or a folder of images:

```bash
python -m myproj.inference.pipeline \
    --input /path/to/source \
    --detector-checkpoint checkpoint_best_ema\(4\).pth \
    --gate-threshold 0.4 \
    --output /path/to/results
```

### 1. Gate Stage (MobileNetV3)
The gate runs on every frame at resolution $224 \times 224$. 
* If prediction is lower than the threshold (e.g. 0.4), execution for the frame terminates.
* This filters out approximately 60%–80% of frames in standard pipeline environments.

### 2. Detection Stage (RF-DETR)
Frames passing the gate are sent to the RF-DETR object detector.
* Encoder: `dinov2_windowed_small`
* Input resolution: 560px (multi-scale enabled)
* Bounding boxes are generated for `crack`, `rebar`, and `spall`.

### 3. Segmentation Stage (U-Net / CrackFormer)
For each detected `crack` bounding box:
* The bounding box region is cropped and resized to $256 \times 256$ or $512 \times 512$.
* The segmenter outputs a binary pixel mask where $1$ denotes a crack pixel and $0$ background.

### 4. Post-processing & Geometry
* **Skeletonization**: Reduces the binary mask to a single-pixel width centerline.
* **Width measurement**: Extracted via distance transform at perpendicular intervals along the centerline.
* **Length measurement**: Determined by path length summation along the centerline.
* **False Positive Rejection**: Connected components are checked for eccentricity to reject circular or non-elongated blobs.
