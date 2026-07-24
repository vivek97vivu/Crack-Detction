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

## 📊 Jetson AGX Orin Performance & Scaling Observations

The pipeline has been extensively benchmarked and optimized for high-scale multi-camera deployments on the **NVIDIA Jetson AGX Orin (64 GB Unified Memory)**.

---

### 1. Multi-Camera Scaling Benchmark (15 to 120 Camera Streams)

All tests ran on the **TensorRT FP16 Dynamic Batch Engine** (`rfdetr-seg-medium-fp16.engine`) using hardware-accelerated GStreamer video decoders (`nvv4l2decoder`):

| Camera Count | Optimal Worker Batch Size | Total System FPS | FPS / Camera | Sampling Interval (Sec/Cam) | Avg CPU % | Avg GPU % | System Memory |
|---|---|---|---|---|---|---|---|
| **15 Cameras** | **Batch = 5** | **43.88 FPS** | **2.93 FPS** | ~0.34s | 45.5% | 88.8% | 1.8 GB |
| **20 Cameras** | **Batch = 4** | **21.88 FPS** | **1.09 FPS** | ~0.92s | 53.5% | 84.1% | 2.1 GB |
| **60 Cameras** | **Batch = 12** | **30.04 FPS** | **0.50 FPS** | ~2.00s | 67.9% | 57.7% | 3.1 GB |
| **80 Cameras** | **Batch = 4** | **31.79 FPS** | **0.40 FPS** | ~2.50s | 57.7% | 75.9% | 3.6 GB |
| **100 Cameras** | **Batch = 4** | **26.30 FPS** | **0.26 FPS** | ~3.85s | 59.5% | 67.1% | 3.9 GB |
| **120 Cameras** | **Batch = 16** | **33.95 FPS** | **0.28 FPS** | ~3.57s | 29.4% | 55.8% | 4.2 GB |

> 💡 **Memory Efficiency**: Even at 120 parallel decoders, single-engine memory usage is only ~4.2 GB.

---

### 2. Multi-Engine Parallel Execution (RAM-Utilized 2x–3x Acceleration)

By leveraging Jetson's 64 GB Unified RAM to instantiate **3 parallel TensorRT engine contexts** (`num_engines=3`) across separate CUDA streams, GPU Tensor Core saturation jumps from 55.4% to **89.2%**, boosting total throughput by **+35.4% to +71.5%**:

| Camera Scale | Architecture | TRT Engines | Total RAM | Total System FPS | FPS / Camera | Avg GPU % | Performance Gain |
|---|---|---|---|---|---|---|---|
| **120 Cameras** | Single Engine | 1 Engine | ~4.2 GB | **19.62 – 33.72 FPS** | 0.16 – 0.28 FPS | 55.4% | Baseline |
| **120 Cameras** | **3x Engine Pool** | **3 Engines** | **~6.2 GB** | **26.56 – 57.84 FPS** 🚀 | **0.22 – 0.48 FPS** | **88.2%** | **+35.4% to +71.5% Faster** |
| **30 Cameras** | **3x Engine Pool** | **3 Engines** | **~6.2 GB** | **53.64 FPS** | **1.79 FPS** | **89.2%** | **High Throughput** |

---

### 3. Precision & Quantization Study (FP16 vs. INT8 vs. FP8)

* **FP16 (Optimal Choice)**: Peak throughput (**43.88 FPS**), 100.0% prediction accuracy retention, and native Ampere Tensor Core acceleration.
* **INT8 Quantization Study**:
  - Implemented static post-training quantization with graph surgery to restore FP32 biases on 12 Conv/Gemm nodes and excluded 3,360 transformer nodes to prevent TensorRT Myelin compiler assertions (`CHECK(is_tensor()) failed`).
  - **Finding**: INT8 engine runs at **28.92 FPS** (slower than FP16's **43.88 FPS**) due to FP16 ↔ INT8 casting overhead between transformer attention blocks and quantized convolutions.
* **FP8 / INT16**: FP8 hardware units are not present on Ampere SM87 (introduced in Ada Lovelace SM89 / Hopper SM90). INT16 lacks hardware Tensor Core support.
* **NVDLA Compatibility**: RF-DETR's deformable cross-attention layers cannot run on NVDLA cores; mixed offloading introduces DLA-to-GPU memory transfer bottlenecks.

---

### 3. Hardware Acceleration & Unthrottling Command

To achieve maximum performance on Jetson AGX Orin, ensure the board is set to **MAX-N Uncapped Mode** to prevent memory bus throttling (from 204 MHz to 3.2 GHz):

```bash
# Unlock 60W+ MAX-N performance profile
sudo nvpmodel -m 0

# Lock GPU clock to 1.30 GHz and CPU to 2.20 GHz
sudo jetson_clocks
```

For complete technical logs, graph surgery python scripts, and detailed architectural notes, see **[observation_document.md](file:///home/algosium/.gemini/antigravity-ide/brain/2361cef8-d143-4b13-95ed-78ee76fdb08c/observation_document.md)**.

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

