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
| 🔍 **MobileNetV3 Binary Gate** | Filters out negative frames (60-80% drop rate) to optimize edge latency |
| 🤖 **RF-DETR Detector** | Bounding box localization of target classes. Silences non-crack detections (`rebar`, `spall`) to focus exclusively on `crack` anomalies |
| 🧩 **U-Net Segmenter** | Pixel-level crack segmentation inside cropped bounding boxes |
| 📐 **Geometry Extraction** | Centerline skeletonization path traversal to measure width, length, and dominant angle |
| 🔁 **Unique Track Filtering** | Deduplication to save exactly one full-frame annotated JPEG snapshot and one JSON report per unique track ID |
| 📝 **API 570/579 Severity** | Maps physical geometry measurements (width, length) to inspection compliance categories (`LEVEL_1` to `LEVEL_3`) |
| 📹 **Multi-Input Streamer** | Ingests USB Webcams, video files, and remote RTSP streams (with GStreamer and automatic FFMPEG fallback) |
| ⚙️ **YAML Config Engine** | Centrally managed settings in `config/config.yaml` for camera inputs, speed metrics, thresholds, and outputs |

---

## 🚀 Pipeline Overview

```text
Camera (RTSP Stream / Webcam / Video File / Static Image / Synthetic)
                          ↓
         MobileNetV3 Binary Gating Classifier (Optional)
                          │
                   (Crack Present?)
                   /              \
             (Yes)/                \(No)
                 ▼                  ▼
        RF-DETR Detector       Drop Frame
                 │
      (Filter: Keep Crack Only)
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
  🚨 ALERT + Save (Full Annotated Frame, JSON Log per Track ID)
```

---

## 🎯 Key Features

* 🧱 **Real-Time Crack Detection**: Multi-stage deep learning pipeline for localization, segmentation, and classification of structural defects.
* 🔍 **Dual-Format Wrapper**: Seamlessly loads both PyTorch (`.pth`) and ONNX (`.onnx`) checkpoints, automatically adapting the pre- and post-processing steps.
* ⚡ **PyTorch Inference Optimization**: Pre-compiles and optimizes PyTorch checkpoints on load using `optimize_for_inference` to fuse layers and remove edge latency overheads.
* 💾 **VRAM Downscaling Control**: Automatically downsamples high-resolution frames before running the PyTorch forward pass, performing the upsampling steps on CPU. This prevents `CUDA out of memory` errors on 1080p and 4K streams.
* 🛡️ **Robust Grayscale Handling**: Automatically converts 2D grayscale camera feeds to 3-channel BGR frames on ingest to prevent overlay shape mismatch crashes.
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
│   └── config.yaml             # Main configuration file (checkpoints, camera streams, threshold rules)
│
├── model/
│   ├── det.pth                 # RF-DETR model checkpoint (best_ema weights)
│   └── seg.pth                 # U-Net segmenter model checkpoint
│
├── alerts/
│   ├── json/                   # Event-triggered structured alert reports (.json)
│   └── snapshot/               # Event-triggered full annotated snapshot images (.jpg)
│
├── log/
│   └── alerts.log              # Appended event logs
│
├── docs/                       # Technical reports and implementation details
│
├── src/
│   ├── inference/
│   │   ├── gate.py             # MobileNetV3 gating classifier inference
│   │   ├── detector.py         # RF-DETR object detector wrapper (with ONNX/PyTorch and downscaling support)
│   │   ├── segmenter.py        # U-Net segmenter wrapper (with crop & mask processing and morphology fallback)
│   │   └── pipeline.py         # Decoupled orchestrator coordinating tracking, geometry, and snapshots
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
├── main.py                     # CLI entry point (handles multiple configurations, camera streams, and self-tests)
└── README.md
```

---

## ⚙️ Configuration

All system behavior is controlled via `config/config.yaml`. No code changes needed.

```yaml
pipeline:
  px_to_mm: 0.15
  alerts_log: "log/alerts.log"
  fallback_to_heuristic: true
  save_snapshots: true
  alerts_json_dir: "alerts/json"
  alerts_snapshot_dir: "alerts/snapshot"
  min_consecutive_frames: 4
  force_split_segmentation: false # Force crop segmenter on det.pth
  enable_detection: true          # Toggle detector stage
  enable_segmentation: true       # Toggle segmenter stage
  enable_gate: true               # Toggle gating classifier stage

gate:
  checkpoint: null                # null uses pretrained MobileNetV3
  threshold: 0.6                  # Configure to filter background noise
  input_size: [224, 224]

detector:
  checkpoint_ema: "model/seg.pth" # Load .pth or .onnx models
  threshold: 0.45                 # Detection threshold
  input_size: [560, 560]
  target_classes: ["crack", "rebar", "spall"]

segmenter:
  checkpoint: null                # null uses traditional morphology
  input_size: [256, 256]
  fallback_to_heuristic: true

geometry:
  pixel_per_mm: 10.0
  min_length_px: 20
  min_area_px: 50
  sample_interval: 5
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

The entrypoint script `main.py` supports CLI arguments to run custom configs or camera channels side-by-side:

```bash
# Run default camera using default config
python main.py

# Run a specific camera configuration in config.yaml
python main.py --camera cam_1

# Run with a custom config file
python main.py --config config/custom_config.yaml
```

* **Note**: If any camera is `enabled: true` in your active `config.yaml`, the pipeline immediately boots the live stream. If all cameras are disabled, it falls back to a synthetic self-test run generating `test_input.jpg` and `test_output.jpg`.

---

## 🚨 Alert System

### Stage 1 — Gating & Crack Filtering
MobileNetV3 filters negative frames (if `enable_gate` is active). Passing frames are processed by RF-DETR. Any detections that are not class `"crack"` (such as rebar or spall) are discarded immediately to keep the system silent on non-defect structures.

### Stage 2 — Measurement & Severity Analysis
For each unique `track_id`, physical width and length metrics are calculated. If the measurements exceed severity thresholds:
* **Image Alert**: Saves the **full annotated frame** highlighting the crack path, bounding box, track ID, and severity badge to `alerts/snapshot/track_{track_id}.jpg`.
* **JSON Alert**: Saves a detailed JSON metadata log detailing the crack location, length, orientation, and severity action recommendation to `alerts/json/track_{track_id}.json`.

---

## 📸 Output

| Directory / File | Contents |
|---|---|
| `alerts/snapshot/` | Full-frame annotated snapshots (.jpg) showing marked crack paths |
| `alerts/json/` | Track-specific alert data reports (.json) containing exact geometry and severity levels |
| `alerts.log` | Central text log appending timestamped severity details and action recommendations |

---

## 🧪 Key Engineering Decisions

* **FFMPEG RTSP Fallback**: Protects production environments by automatically switching from GStreamer pipelines to direct OpenCV FFMPEG readers if local network or plugin issues occur.
* **Warning Suppression**: Blocks package deprecation output to keep terminal streams clean and readable.
* **Frame Skipping (`frame_skip`)**: Processes every $N$-th frame (e.g., 1 out of 5 frames), reducing CPU/GPU load to guarantee real-time performance on high-resolution video streams.
* **Deduplicated Alerting**: Prevents alert flooding by logging exactly one snapshot and JSON report per unique track ID.
* **Grayscale Auto-Conversion**: Prevents overlay dimension mismatches by converting grayscale camera streams to BGR formats on stream ingest.
* **GPU Memory Optimization**: Implements high-resolution frame downscaling and CPU-based upsampling during PyTorch inference to prevent CUDA out-of-memory errors.

---

<div align="center">
Engineered by the <b>Algosium AI Team</b> · CampNeuron AI Series
</div>
