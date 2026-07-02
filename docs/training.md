# Model Training Guide

This guide details the configurations and procedures for training each model in the crack detection pipeline.

---

## 1. Stage 1: CNN Classifier Gate (MobileNetV3)

The binary gate classifier is trained on full-frame images flagged with `crack_present = 1` or `0` based on the existence of annotated polygons.

### Training Details
* **Model**: MobileNetV3-Small or ResNet18
* **Input Resolution**: $224 \times 224$ (no cropping)
* **Loss**: Binary Cross-Entropy (BCE) with 0.1 label smoothing
* **Target Objective**: Maximize recall on the validation set. False negatives must be kept $< 2\%$ because missing a crack is more critical than a false gate trigger.

---

## 2. Stage 2: RF-DETR Object Detector

The RF-DETR object detector localizes classes (`crack`, `rebar`, `spall`) on images that contain cracks/defects.

### Key Configurations
* **Model Base**: RFDETRBase with a `dinov2_windowed_small` encoder
* **Resolution**: 560px with multi-scale training enabled (`multi_scale: true` and `expanded_scales: true`)
* **Classes Configuration**: Exactly 3 classes (`crack`, `rebar`, `spall`).
  > [!IMPORTANT]
  > Resolve config issues: ensure class names list has no duplicate entries and `dataset_dir` is correctly configured before training.

### Local Weights
* A pre-trained checkpoint is available in the project root: `checkpoint_best_ema(4).pth`.

---

## 3. Stage 3: Crack Segmentation

The segmenter operates on cropped bounding box crops of cracks, producing a pixel-level binary mask.

### Training Details
* **Loss Function**: Combined Dice Loss + Binary Cross-Entropy (BCE). This handles class imbalance from very thin hairline cracks.
* **Input Resolution**: Resized crops at $256 \times 256$ or $512 \times 512$.
* **Augmentations**: Elastic distortion, random rotation, brightness/contrast jitter, and random flipping.

### PyTorch Loss Implementation Example
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Flatten label and prediction tensors
        inputs = torch.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
```
