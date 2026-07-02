<div align="center">

# 🧱 **Crack Detection Pipeline**

### 🚨 Real-Time Structural Defect and Crack Detection for Industrial Infrastructure

A **production-grade AI pipeline** built for **real-time structural health monitoring**, combining **MobileNetV3 gating + RF-DETR detection + U-Net segmentation + centerline geometry extraction + tracking** for API 570/579 compliance reporting.

> ⚙️ Powered by **MobileNetV3, RF-DETR, and U-Net**
> 🧠 Designed for **low false positives, high reliability edge deployments**
> 🧩 Part of the **CampNeuron AI Series** — engineered by the **Algosium AI Team**

---

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](#)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green?logo=nvidia&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch&logoColor=white)](#)
[![API-579](https://img.shields.io/badge/Compliance-API%20570%2F579-red)](#)
[![Platform](https://img.shields.io/badge/Platform-Linux%20|%20x86__64-lightgrey?logo=linux&logoColor=white)](#)

</div>

---

## 📷 Visual Demonstrations

Here is a preview of the Crack Detection Pipeline in action, showing the detection, instance segmentation, and persistent tracking capabilities on actual concrete and steel structures.

<div align="center">
  <table border="0">
    <tr>
      <td align="center"><b>1. Object Detection (RF-DETR)</b></td>
      <td align="center"><b>2. Instance Segmentation Mask</b></td>
    </tr>
    <tr>
      <td><img src="assets/crack_detection_demo.jpg" alt="RF-DETR Object Detection" width="400"/></td>
      <td><img src="assets/crack_segmentation_demo.jpg" alt="Crack Instance Segmentation" width="400"/></td>
    </tr>
    <tr>
      <td align="center" colspan="2"><b>3. Real-Time Tracking & ID Assignment (IOU Tracker)</b></td>
    </tr>
    <tr>
      <td align="center" colspan="2"><img src="assets/crack_tracking_demo.jpg" alt="BBox Multi-Object Tracking" width="600"/></td>
    </tr>
  </table>
</div>

---

## ⚡ Core Stack

| Component | Purpose |
|---|---|
| 🔍 **MobileNetV3 Binary Gate** | Filters out negative frames (60-80% drop rate) to optimize latency |
| 🤖 **RF-DETR Detector** | Bounding box localization of target classes (`crack`, `rebar`, `spall`) |
| 🧩 **U-Net Segmenter** | Pixel-level crack segmentation inside cropped bounding boxes |
| 📐 **Geometry Extraction** | Centerline skeletonization path traversal to measure width, length, and dominant angle |
| 🔁 **Unique Track Filtering** | Deduplication to save exactly one crop snapshot and one JSON report per unique track ID |
| 📝 **API 570/579 Severity** | Maps physical geometry measurements to inspection compliance categories |
| ⚙️ **YAML Config Engine** | Centrally managed settings in `config/config.yaml` for camera inputs, thresholds, and outputs |

---

## 🚀 Pipeline Overview

```text
Camera (RTSP Stream / Webcam / Video File / Static Image / Synthetic)
                          ↓
         MobileNetV3 Binary Gating Classifier
                          │
                  (Crack Present?)
                  /              \
            (Yes)/                \(No)
                ▼                  ▼
       RF-DETR Detector       Drop Frame
                │
         (Bounding Box)
                │
                ▼
         U-Net Segmenter
                │
             (Mask)
                │
                ▼
      Geometry Extraction (Centerline, Length, Width)
                │
                ▼
      API 570/579 Severity & Alerting (Level 1 / 2 / 3)
                │
                ▼
      Simple BBox Tracker (Assign persistent Track ID)
                │
                ▼
  🚨 ALERT + Save (Cropped JPEG, JSON Log per Track ID)
```

---

## 🎯 Key Features

* 🧱 **Real-Time Crack Detection**: Multi-stage deep learning pipeline for localization, segmentation, and classification of structural defects.
* 🔍 **MobileNetV3 Gating**: Lowers edge hardware execution costs by dynamically filtering out negative frames.
* 📏 **Connected Component Geometry**: Labels individual crack segments using `skimage` to measure physical length, mean/max width, and orientation.
* 🔁 **Redundancy Filter**: Prevents alert flooding by saving exactly one crop JPEG and one JSON metadata report per unique track ID.
* 📝 **Compliance Mapping**: Automatically determines API 570/579 fitness-for-service severity rankings (`LEVEL_1` to `LEVEL_3`) and recommended maintenance intervals.
* 📂 **Structured JSON Logging**: Saves individual JSON reports and session history logs tracking timestamps, coordinates, geometries, and severity levels.
* 🏷️ **Track ID Annotation**: Displays track IDs directly on the live overlay and saved screenshots for easy auditing.

---

## 📂 Project Structure

```bash
crack_detection_oilgas/
├── config/
│   └── config.yaml             # Main configuration file (checkpoints, thresholds, API rules)
│
├── model/
│   └── checkpoint_best_ema(4).pth  # RF-DETR model checkpoint
│
├── runs/
│   └── snapshots/              # Event-triggered crop images and JSON alert logs
│
├── src/myproj/
│   ├── inference/
│   │   ├── gate.py             # MobileNetV3 gating classifier inference
│   │   ├── detector.py         # RF-DETR object detector wrapper
│   │   ├── segmenter.py        # U-Net segmenter wrapper (with morphological fallback)
│   │   └── pipeline.py         # Coordinates frame gating, tracking, and snapshots
│   │
│   ├── training/
│   │   ├── train_gate.py       # Gate classifier training script
│   │   └── train_segmenter.py  # U-Net segmenter training script
│   │
│   ├── deploy/
│   │   └── trt_export.py       # Jetson TensorRT ONNX/engine compiler
│   │
│   └── utils/
│       ├── config.py           # YAML config loader and path resolver
│       ├── geometry.py         # Skeleton centerline path and widths extractor
│       ├── severity.py         # API 570/579 fitness-for-service mapper
│       ├── tracking.py         # Simple IOU-based bounding box tracker
│       └── visualization.py    # Overlays HUD, masks, and severity badges
│
├── run_demo.py                 # Pipeline demonstration entry point
└── README.md
```

---

## ⚙️ Configuration

All system behavior is controlled via `config/config.yaml`. No code changes needed.

```yaml
pipeline:
  px_to_mm: 0.15
  alerts_log: "alerts.log"
  fallback_to_heuristic: true
  snapshot_dir: "runs/snapshots"
  save_snapshots: true

gate:
  checkpoint: null
  threshold: 0.2
  input_size: [224, 224]

detector:
  checkpoint: "model/checkpoint_best_ema(4).pth"
  threshold: 0.1
  input_size: [560, 560]
  target_classes: ["crack", "rebar", "spall"]

segmenter:
  checkpoint: null
  input_size: [256, 256]
  fallback_to_heuristic: true

geometry:
  pixel_per_mm: 10.0
  min_length_px: 20
  min_area_px: 50
  sample_interval: 5

severity:
  level_1:
    max_width_mm: 0.2
    max_length_mm: 20.0
    action: "Monitor and log during routine checkups"
    reinspection_days: 180
  level_2:
    max_width_mm: 0.5
    max_length_mm: 50.0
    action: "Schedule repair/maintenance within 30 days"
    reinspection_days: 30
  level_3:
    action: "Immediate shutdown or emergency maintenance"
    reinspection_days: 0
```

---

## 🚀 Installation

```bash
git clone https://github.com/vivek97vivu/Crack-Detction.git
cd Crack-Detction

# Activate your conda environment (e.g., crack)
conda activate crack

# Install required dependencies
pip install -r requirements.txt
```

### Requirements

* NVIDIA GPU (CUDA support recommended)
* PyTorch / Torchvision
* scikit-image & scipy
* OpenCV
* Python 3.12

---

## ▶️ Run

```bash
# Run with PYTHONPATH pointing to src
PYTHONPATH=src python run_demo.py
```

---

## 🚨 Alert System

### Stage 1 — Detection & Segmentation
MobileNetV3 filters negative frames. Passing frames are processed by RF-DETR to detect cracks, rebars, or spalls, followed by U-Net crop-level segmentation.

### Stage 2 — Geometry & Severity Analysis
Connected components are extracted, and physical width and length metrics are calculated. If the measurements violate inspection parameters:
* **Image Alert**: Saves a JPEG crop highlighting the crack area with a 20px margin to `runs/snapshots/` (exactly once per track ID, named `track_{track_id}_crop_{timestamp}.jpg`).
* **JSON Alert**: Saves a detailed JSON report describing the geometry coordinates, orientation, aspect ratio, and mapped severity action to `runs/snapshots/` (exactly once per track ID, named `track_{track_id}_alert_{timestamp}.json`).

---

## 📸 Output

| Folder | Contents |
|---|---|
| `runs/snapshots/` | Track-specific alert reports (.json) and high-quality cropped crack snapshots (.jpg) |
| `alerts.log` | Central text log appending timestamped severity details and action recommendations |

---

## ⚡ Performance

| Metric | Value |
|---|---|
| Gating Frame Filter | ~2–4ms |
| Detector Inference | ~10–18ms (on CUDA GPUs) |
| Segmentation & Geometry | ~5–12ms |
| GPU Memory Footprint | ~1.5 GB to 2.2 GB |

---

## 🧪 Engineering Decisions

| Decision | Reason |
|---|---|
| **Binary Classifier Gate** | Gating drops up to 80% of negative frames, saving massive edge hardware compute. |
| **Deduplicated Alerting** | Enforces saving exactly one crop JPEG and one JSON log per track ID, preventing alert flooding on persistent cracks. |
| **Scikit-Image Labeling** | Connectivity labeling allows the pipeline to measure and catalog multiple independent cracks inside a single bounding box crop. |

---

<div align="center">
Engineered by the <b>Algosium AI Team</b> · CampNeuron AI Series
</div>
