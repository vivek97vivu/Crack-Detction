# Edge Deployment Instructions

This document covers deploying the crack detection pipeline on edge devices, particularly NVIDIA Jetson platforms.

---

## Jetson Optimization

Executing multiple deep learning models sequentially is resource-intensive. The following techniques are applied to ensure real-time performance:

### 1. Throughput Gate (MobileNetV3)
The lightweight classification gate is the primary mechanism for saving compute. By dropping 60%–80% of empty frames at $224 \times 224$ resolution, we prevent the heavy RF-DETR detector and segmentation models from running on negative frames.

### 2. TensorRT Conversion
All models should be exported to ONNX and compiled using TensorRT on the target edge device:

```bash
# Example command to compile to TensorRT engine
/usr/src/tensorrt/bin/trtexec \
    --onnx=model.onnx \
    --saveEngine=model.engine \
    --fp16
```

* **Gate (MobileNetV3)**: Export with FP16 precision.
* **RF-DETR Detector**: Export with FP16 precision.
* **Segmentation Network**: Use FP16 or INT8 precision where calibration datasets are available.

---

## Alerts and Severity Mapping

Extracted crack geometries are classified against API 570/579 fitness-for-service criteria. Alerts are fired with cooldown deduplication:

```
Severity Level 1 (Monitor)       ──> Log to MLflow
Severity Level 2 (Repair)        ──> Email/Slack alert (cooldown: 2 hours)
Severity Level 3 (Immediate)     ──> Email/Slack/SMS alert (cooldown: 10 mins)
```

> [!NOTE]
> VLM confirmation has been removed from the alerting loop. The pipeline relies directly on the metrics computed by the segmentation and geometry stages to evaluate severity level and recommended action.
