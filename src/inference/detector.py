import rfdetr
import numpy as np
import cv2
import onnxruntime
import torch
from dataclasses import dataclass
from typing import Optional, List
from src.utils.geometry import CrackGeometry
from src.utils.severity import SeverityResult

@dataclass
class Detection:
    bbox_xyxy: np.ndarray # shape (4,) [x1, y1, x2, y2]
    class_name: str
    confidence: float
    class_id: int
    track_id: Optional[int] = None
    geometry: List[CrackGeometry] = None
    severity: Optional[SeverityResult] = None
    mask: Optional[np.ndarray] = None

class DetectorInference:
    """
    Wrapper for RF-DETR object detector that handles model loading,
    inference, and mapping of the 4 trained classes down to 3 clean target classes.
    Supports both PyTorch (.pth) checkpoints and ONNX (.onnx) exports.
    """
    def __init__(self, checkpoint_path, threshold=0.3):
        self.threshold = threshold
        self.target_classes = ["crack", "rebar", "spall"]
        
        if checkpoint_path.endswith(".onnx"):
            print(f"Loading RF-DETR model from ONNX checkpoint: {checkpoint_path}")
            self.use_onnx = True
            self.session = onnxruntime.InferenceSession(checkpoint_path)
            self.input_name = self.session.get_inputs()[0].name
            input_shape = self.session.get_inputs()[0].shape
            self.input_h = input_shape[2]
            self.input_w = input_shape[3]
        else:
            print(f"Loading RF-DETR model from PyTorch checkpoint: {checkpoint_path}")
            self.use_onnx = False
            self.model = rfdetr.from_checkpoint(checkpoint_path)
            
            # Call rfdetr optimization to prevent PyTorch inference delay
            try:
                print("Optimizing PyTorch model for inference...")
                self.model.optimize_for_inference(compile=False)
            except Exception as e:
                print(f"Warning: Could not optimize PyTorch model: {e}")
        
    def predict(self, image):
        """
        Runs object detection on the image.
        Returns:
            list[dict]: A list of detections with mapped class IDs and labels.
        """
        if self.use_onnx:
            return self._predict_onnx(image)
        else:
            return self._predict_pytorch(image)
            
    def process(self, frame_rgb, tracker):
        """
        Runs prediction, maps raw outputs to Detection objects, and updates tracker.
        Returns:
            list[Detection]: Tracked detection objects.
        """
        h, w = frame_rgb.shape[:2]
        detector_outputs = self.predict(frame_rgb)
        
        raw_detections = []
        for det in detector_outputs:
            if det["class_name"] != "crack":
                continue
            box = det["box"]
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            raw_detections.append(Detection(
                bbox_xyxy=np.array([x1, y1, x2, y2], dtype=int),
                class_name=det["class_name"],
                confidence=det["confidence"],
                class_id=det["class_id"],
                mask=det.get("mask")
            ))
            
        return tracker.update(raw_detections)

    def _predict_pytorch(self, image):
        h_orig, w_orig = image.shape[:2]
        
        # Scale down large images to prevent GPU VRAM OOM during mask upsampling in postprocess.py
        max_dim = 960
        if max(h_orig, w_orig) > max_dim:
            scale = max_dim / max(h_orig, w_orig)
            h_new, w_new = int(h_orig * scale), int(w_orig * scale)
            image_resized = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
        else:
            image_resized = image
            scale = 1.0
            
        # Run prediction via rfdetr PyTorch wrapper
        detections = self.model.predict(image_resized, threshold=self.threshold)
        
        results = []
        if len(detections) == 0:
            return results
            
        xyxy = detections.xyxy
        confidences = detections.confidence
        class_ids = detections.class_id
        masks = detections.mask if (hasattr(detections, "mask") and detections.mask is not None) else None
        
        for i in range(len(xyxy)):
            orig_cid = class_ids[i]
            # Rescale boxes back to original image size
            box = [
                xyxy[i][0] / scale,
                xyxy[i][1] / scale,
                xyxy[i][2] / scale,
                xyxy[i][3] / scale
            ]
            conf = float(confidences[i])
            
            # Map original 4 classes to target 3 classes
            if orig_cid in (0, 1):
                mapped_cid = 0
            elif orig_cid == 2:
                mapped_cid = 1
            elif orig_cid == 3:
                mapped_cid = 2
            else:
                continue # ignore any invalid classes
                
            det_dict = {
                "box": box, # [x1, y1, x2, y2]
                "confidence": conf,
                "class_id": mapped_cid,
                "class_name": self.target_classes[mapped_cid]
            }
            if masks is not None:
                # Upsample the mask to original image resolution in CPU memory
                mask_resized = cv2.resize(masks[i].astype(np.uint8), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                det_dict["mask"] = mask_resized.astype(bool)
                
            results.append(det_dict)
            
        return results

    def _predict_onnx(self, image):
        h_orig, w_orig = image.shape[:2]
        
        # 1. Preprocessing: BGR to RGB, Resize, Float, Normalize
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        img_float = img_resized.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_float - mean) / std
        
        inp_tensor = np.transpose(img_normalized, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
        
        # 2. Run session
        raw_outputs = self.session.run(None, {self.input_name: inp_tensor})
        
        # Identify outputs by shape and name matching
        output_names = [out.name for out in self.session.get_outputs()]
        boxes_idx = next((i for i, name in enumerate(output_names) if "dets" in name), 0)
        logits_idx = next((i for i, name in enumerate(output_names) if "labels" in name), 1)
        masks_idx = next((i for i, name in enumerate(output_names) if "masks" in name), None)
        
        boxes_cwh = raw_outputs[boxes_idx][0]  # (Q, 4) normalized center-x, center-y, width, height
        logits = raw_outputs[logits_idx][0, :, :-1]  # (Q, num_classes)
        
        # Apply sigmoid to logits to get scores
        one = np.asarray(1, dtype=logits.dtype)
        scores_all = one / (one + np.exp(-logits.clip(-88, 88)))
        scores = scores_all.max(axis=-1)
        cls = scores_all.argmax(axis=-1)
        
        # Keep detections above threshold
        keep = scores >= self.threshold
        results = []
        if not np.any(keep):
            return results
            
        kept_boxes = boxes_cwh[keep]
        kept_scores = scores[keep]
        kept_classes = cls[keep]
        
        # Convert cx, cy, w, h to x1, y1, x2, y2
        cx, cy, bw, bh = kept_boxes.T
        x1 = (cx - bw / 2) * w_orig
        y1 = (cy - bh / 2) * h_orig
        x2 = (cx + bw / 2) * w_orig
        y2 = (cy + bh / 2) * h_orig
        
        # Parse masks if model has segmentation output
        has_masks = masks_idx is not None
        if has_masks:
            raw_masks = raw_outputs[masks_idx][0]  # (Q, Hm, Wm)
            kept_masks = raw_masks[keep]
            
            decoded_masks = []
            for idx in range(len(kept_boxes)):
                mask_logit = kept_masks[idx]
                # Resize mask to original input size
                mask_resized = cv2.resize(mask_logit, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                # Convert logit to binary mask
                mask_binary = mask_resized > 0.0
                decoded_masks.append(mask_binary)
                
        # 3. Build result dictionaries
        for i in range(len(kept_boxes)):
            orig_cid = int(kept_classes[i])
            conf = float(kept_scores[i])
            
            # Map original classes to target classes
            if orig_cid in (0, 1):
                mapped_cid = 0
            elif orig_cid == 2:
                mapped_cid = 1
            elif orig_cid == 3:
                mapped_cid = 2
            else:
                continue
                
            x1_clip = max(0, min(w_orig, int(x1[i])))
            y1_clip = max(0, min(h_orig, int(y1[i])))
            x2_clip = max(0, min(w_orig, int(x2[i])))
            y2_clip = max(0, min(h_orig, int(y2[i])))
            box = [x1_clip, y1_clip, x2_clip, y2_clip]
            
            det_dict = {
                "box": box,
                "confidence": conf,
                "class_id": mapped_cid,
                "class_name": self.target_classes[mapped_cid]
            }
            if has_masks:
                det_dict["mask"] = decoded_masks[i]
                
            results.append(det_dict)
            
        return results
