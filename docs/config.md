# Configuration and Parameters Guide

This file defines the primary configurable parameters across the pipeline stages.

---

## 1. Class Definitions
* `0`: `crack`
* `1`: `rebar`
* `2`: `spall`

---

## 2. Gate Configuration
* `input_size`: `[224, 224]`
* `threshold`: `0.4` (adjustable depending on target validation recall)
* `model`: `mobilenetv3_small`

---

## 3. Detector Configuration
* `input_size`: `[560, 560]`
* `multi_scale`: `true`
* `expanded_scales`: `true`
* `detector_checkpoint`: `checkpoint_best_ema(4).pth`

---

## 4. Segmentation Configuration
* `input_size`: `[256, 256]` (or `[512, 512]`)
* `model`: `unet` or `crackformer`

---

## 5. Severity Mapping Thresholds (API 570/579 Compliance)

| Level | Severity | Criteria | Action Recommended |
| :--- | :--- | :--- | :--- |
| **Level 1** | Minor | Width $\le 0.2\text{ mm}$ and Length $\le 20\text{ mm}$ | Monitor and log during routine checkups |
| **Level 2** | Moderate | Width $> 0.2\text{ mm}$ or Length $> 20\text{ mm}$ (and not Level 3) | Schedule repair/maintenance within 30 days |
| **Level 3** | Critical | Width $> 0.5\text{ mm}$ or Length $> 50\text{ mm}$ | Immediate shutdown or emergency maintenance |
