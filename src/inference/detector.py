import rfdetr
import numpy as np
import cv2
import onnxruntime
import torch
import torch.nn.functional as F
import os
import logging
import queue
import threading
from dataclasses import dataclass
from typing import Optional, List
from src.utils.geometry import CrackGeometry
from src.utils.severity import SeverityResult
from src.inference.trt_engine import TRTEngineBackend, is_available as trt_is_available

logger = logging.getLogger(__name__)

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


def _build_ort_session(onnx_path: str, trt_cache_dir: str = "model/trt_cache") -> onnxruntime.InferenceSession:
    """
    Build an ONNX Runtime InferenceSession with the best available GPU provider.
    Priority: TensorrtExecutionProvider > CUDAExecutionProvider > CPU.

    TensorRT EP compiles the model to a TensorRT engine on first run (slow, ~1-3 min)
    then caches it — subsequent runs load instantly and are 3-10x faster.
    """
    available = onnxruntime.get_available_providers()
    logger.info("ORT available providers: %s", available)

    # Shared session options: maximize graph optimization
    sess_opts = onnxruntime.SessionOptions()
    sess_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

    # ── 1. Try TensorRT Execution Provider ──
    if "TensorrtExecutionProvider" in available:
        try:
            os.makedirs(trt_cache_dir, exist_ok=True)
            trt_options = {
                "device_id": "0",
                "trt_max_workspace_size": str(4 * 1024 * 1024 * 1024),  # 4 GB
                "trt_fp16_enable": "1",
                "trt_engine_cache_enable": "1",
                "trt_engine_cache_path": trt_cache_dir,
                "trt_dla_enable": "0",
            }
            logger.info("Initializing ORT with TensorRT EP (cache: %s)...", trt_cache_dir)
            sess = onnxruntime.InferenceSession(
                onnx_path,
                sess_options=sess_opts,
                providers=[("TensorrtExecutionProvider", trt_options), "CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            active = sess.get_providers()[0]
            logger.info("ORT session active EP: %s", active)
            print(f"  [GPU] ORT active provider: {active}")
            return sess
        except Exception as e:
            logger.warning("TensorRT EP init failed (%s) — trying CUDA EP", e)

    # ── 2. Fallback to CUDA Execution Provider ──
    if "CUDAExecutionProvider" in available:
        try:
            cuda_options = {
                "device_id": "0",
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": str(4 * 1024 * 1024 * 1024),
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": "1",
            }
            logger.info("Initializing ORT with CUDA EP...")
            sess = onnxruntime.InferenceSession(
                onnx_path,
                sess_options=sess_opts,
                providers=[("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"],
            )
            active = sess.get_providers()[0]
            logger.info("ORT session active EP: %s", active)
            print(f"  [GPU] ORT active provider: {active}")
            return sess
        except Exception as e:
            logger.warning("CUDA EP init failed (%s) — falling back to CPU", e)

    # ── 3. CPU fallback ──
    print("  [WARN] Running on CPU — no GPU execution provider available")
    return onnxruntime.InferenceSession(
        onnx_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]
    )


class DetectorInference:
    """
    Wrapper for RF-DETR object detector.
    Supports:
      - TRT (.engine): Native TensorRT — fastest, instant startup, no ORT overhead
      - ONNX (.onnx):  GPU via ORT TensorRT/CUDA EP — auto-compiles TRT on first run
      - PyTorch (.pth): CPU-only fallback
    """
    def __init__(self, checkpoint_path, threshold=0.3, num_engines=3):
        self.threshold = threshold
        self.target_classes = ["crack", "rebar", "spall"]
        self.use_trt    = False
        self.use_onnx   = False
        self.trt_engine = None
        self.trt_engine_pool = queue.Queue()
        self.session    = None
        self.model      = None

        if checkpoint_path.endswith(".engine"):
            # ── Native TRT engine (fastest — pre-compiled, no ORT overhead) ──
            if not trt_is_available():
                raise RuntimeError(
                    "TRT engine loading failed: TensorRT Python bindings or "
                    "libcudart.so not found. Use the .onnx checkpoint instead."
                )
            print(f"Loading RF-DETR model ({num_engines}x parallel engines) from TensorRT engine: {checkpoint_path}")
            self.use_trt = True
            for i in range(num_engines):
                eng = TRTEngineBackend(checkpoint_path)
                self.trt_engine_pool.put(eng)
            self.trt_engine = eng
            # Derive input shape from first input tensor
            inp = self.trt_engine._inputs[0]
            shape = inp["shape"]
            self.input_h = int(shape[2]) if shape[2] > 0 else 576
            self.input_w = int(shape[3]) if shape[3] > 0 else 576
            print(f"  [GPU] {num_engines}x TRT engine pool loaded — input: {self.input_h}x{self.input_w}")

        elif checkpoint_path.endswith(".onnx"):
            # ── ONNX via ORT TensorRT/CUDA EP ──
            print(f"Loading RF-DETR model from ONNX (GPU via ORT): {checkpoint_path}")
            self.use_onnx = True
            trt_cache = os.path.join(os.path.dirname(checkpoint_path), "trt_cache")
            self.session = _build_ort_session(checkpoint_path, trt_cache_dir=trt_cache)
            self.input_name = self.session.get_inputs()[0].name
            input_shape = self.session.get_inputs()[0].shape
            self.input_h = input_shape[2] if isinstance(input_shape[2], int) else 576
            self.input_w = input_shape[3] if isinstance(input_shape[3], int) else 576

        else:
            # ── PyTorch fallback ──
            print(f"Loading RF-DETR model from PyTorch checkpoint: {checkpoint_path}")
            self.model = rfdetr.from_checkpoint(checkpoint_path)
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
        if self.use_trt:
            return self._predict_trt(image)
        elif self.use_onnx:
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

    def predict_batch(self, images):
        """
        Runs batched object detection on a list of images.
        Uses GPU-accelerated PyTorch preprocessing and zero-copy TRT inference.

        Args:
            images (list[np.ndarray]): List of BGR images.
        Returns:
            list[list[dict]]: List of detections for each image.
        """
        if not self.use_trt or not torch.cuda.is_available():
            # Fallback: run sequentially for ONNX/PyTorch backends
            return [self.predict(img) for img in images]

        batch_size = len(images)
        if batch_size == 0:
            return []

        # 1. Preprocess batch on GPU via PyTorch
        gpu_tensors = []
        orig_shapes = []
        for img in images:
            h_orig, w_orig = img.shape[:2]
            orig_shapes.append((h_orig, w_orig))

            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]

            # Upload BGR frame to GPU (very fast uint8 transfer)
            t = torch.from_numpy(img).to('cuda', non_blocking=True) # H x W x 3 (uint8)
            # Flip BGR -> RGB
            t = t.flip(2)
            # Permute to C x H x W
            t = t.permute(2, 0, 1).float() # C x H x W (float32)
            gpu_tensors.append(t)

        # Stack into N x C x H x W
        stacked = torch.stack(gpu_tensors)

        # Resize to input_h, input_w
        resized = F.interpolate(
            stacked, size=(self.input_h, self.input_w),
            mode='bilinear', align_corners=False
        )

        # Normalize: (pixel / 255.0 - mean) / std
        resized /= 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device='cuda').view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device='cuda').view(1, 3, 1, 1)
        normalized = (resized - mean) / std

        inp_tensor = normalized.contiguous()

        # 2. Run batched TRT inference using PyTorch pointers
        engine = self.trt_engine_pool.get() if hasattr(self, "trt_engine_pool") and not self.trt_engine_pool.empty() else self.trt_engine
        try:
            input_name = engine.input_names[0]
            raw_outputs = engine.infer_pytorch({input_name: inp_tensor})
        finally:
            if hasattr(self, "trt_engine_pool"):
                self.trt_engine_pool.put(engine)

        # Map outputs by name
        output_names = list(raw_outputs.keys())
        boxes_key  = next((k for k in output_names if "dets"   in k), output_names[0])
        logits_key = next((k for k in output_names if "labels" in k),
                          output_names[1] if len(output_names) > 1 else output_names[0])
        masks_key  = next((k for k in output_names if "masks"  in k), None)

        boxes_cwh = raw_outputs[boxes_key]           # (B, Q, 4) cx,cy,w,h normalised
        logits    = raw_outputs[logits_key][..., :-1] # (B, Q, num_classes)
        raw_masks = raw_outputs[masks_key] if masks_key else None  # (B, Q, Hm, Wm) float32 logits

        # 3. Post-process (sigmoid, thresholding done on GPU)
        scores_all = torch.sigmoid(logits.clamp(-88, 88))
        scores, cls = scores_all.max(dim=-1) # (B, Q), (B, Q)

        batch_results = []
        for b in range(batch_size):
            h_orig, w_orig = orig_shapes[b]
            b_scores = scores[b]
            b_cls = cls[b]
            b_boxes = boxes_cwh[b]
            b_masks = raw_masks[b] if raw_masks is not None else None

            keep = b_scores >= self.threshold
            keep_idx = torch.where(keep)[0]

            if len(keep_idx) == 0:
                batch_results.append([])
                continue

            # Copy only filtered predictions to host CPU memory
            kept_scores = b_scores[keep_idx].cpu().numpy()
            kept_classes = b_cls[keep_idx].cpu().numpy()
            kept_boxes = b_boxes[keep_idx].cpu().numpy()
            kept_masks = b_masks[keep_idx].cpu().numpy() if b_masks is not None else None

            cx, cy, bw, bh = kept_boxes.T
            x1 = (cx - bw / 2) * w_orig
            y1 = (cy - bh / 2) * h_orig
            x2 = (cx + bw / 2) * w_orig
            y2 = (cy + bh / 2) * h_orig

            results = []
            for i in range(len(keep_idx)):
                orig_cid = int(kept_classes[i])
                conf     = float(kept_scores[i])

                if orig_cid in (0, 1):
                    mapped_cid = 0
                elif orig_cid == 2:
                    mapped_cid = 1
                elif orig_cid == 3:
                    mapped_cid = 2
                else:
                    continue

                x1_c = max(0, min(w_orig, int(x1[i])))
                y1_c = max(0, min(h_orig, int(y1[i])))
                x2_c = max(0, min(w_orig, int(x2[i])))
                y2_c = max(0, min(h_orig, int(y2[i])))

                det_dict = {
                    "box":        [x1_c, y1_c, x2_c, y2_c],
                    "confidence": conf,
                    "class_id":   mapped_cid,
                    "class_name": self.target_classes[mapped_cid],
                }
                if kept_masks is not None:
                    det_dict["mask"] = kept_masks[i]

                results.append(det_dict)

            batch_results.append(results)

        return batch_results

    def post_predict_and_track(self, detector_outputs, tracker, img_shape):
        """
        Runs tracking on pre-computed detector outputs.
        """
        h, w = img_shape[:2]
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

    def _predict_trt(self, image):
        """Native TRT inference — same pre/post-processing as _predict_onnx."""
        h_orig, w_orig = image.shape[:2]

        # Pre-process (identical to ONNX path)
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        img_float = img_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_float - mean) / std
        inp_tensor = np.ascontiguousarray(
            np.transpose(img_normalized, (2, 0, 1))[np.newaxis, ...], dtype=np.float32
        )

        # Run native TRT inference
        engine = self.trt_engine_pool.get() if hasattr(self, "trt_engine_pool") and not self.trt_engine_pool.empty() else self.trt_engine
        try:
            input_name = engine.input_names[0]
            raw_outputs = engine.infer({input_name: inp_tensor})
        finally:
            if hasattr(self, "trt_engine_pool"):
                self.trt_engine_pool.put(engine)

        # Map outputs by name (same convention as ONNX path)
        output_names = list(raw_outputs.keys())
        boxes_key  = next((k for k in output_names if "dets"   in k), output_names[0])
        logits_key = next((k for k in output_names if "labels" in k),
                          output_names[1] if len(output_names) > 1 else output_names[0])
        masks_key  = next((k for k in output_names if "masks"  in k), None)

        boxes_cwh = raw_outputs[boxes_key][0]           # (Q, 4) cx,cy,w,h normalised
        logits    = raw_outputs[logits_key][0, :, :-1]  # (Q, num_classes)
        raw_masks = raw_outputs[masks_key][0] if masks_key else None  # (Q, Hm, Wm) float32 logits

        # Post-process (identical to ONNX path)
        one = np.asarray(1, dtype=logits.dtype)
        scores_all = one / (one + np.exp(-logits.clip(-88, 88)))
        scores = scores_all.max(axis=-1)
        cls    = scores_all.argmax(axis=-1)

        keep = scores >= self.threshold
        if not np.any(keep):
            return []

        kept_boxes   = boxes_cwh[keep]
        kept_scores  = scores[keep]
        kept_classes = cls[keep]
        kept_masks   = raw_masks[keep] if raw_masks is not None else None  # (K, Hm, Wm)

        # Convert cx,cy,w,h → x1,y1,x2,y2
        cx, cy, bw, bh = kept_boxes.T
        x1 = (cx - bw / 2) * w_orig
        y1 = (cy - bh / 2) * h_orig
        x2 = (cx + bw / 2) * w_orig
        y2 = (cy + bh / 2) * h_orig

        results = []
        for i in range(len(kept_boxes)):
            orig_cid = int(kept_classes[i])
            conf     = float(kept_scores[i])

            # Same class ID mapping as _predict_onnx
            if orig_cid in (0, 1):
                mapped_cid = 0   # crack
            elif orig_cid == 2:
                mapped_cid = 1   # rebar
            elif orig_cid == 3:
                mapped_cid = 2   # spall
            else:
                continue

            x1_c = max(0, min(w_orig, int(x1[i])))
            y1_c = max(0, min(h_orig, int(y1[i])))
            x2_c = max(0, min(w_orig, int(x2[i])))
            y2_c = max(0, min(h_orig, int(y2[i])))

            det_dict = {
                "box":        [x1_c, y1_c, x2_c, y2_c],
                "confidence": conf,
                "class_id":   mapped_cid,
                "class_name": self.target_classes[mapped_cid],
            }
            # Store raw float32 logit mask (Hm, Wm) — pipeline.py resizes on demand
            if kept_masks is not None:
                det_dict["mask"] = kept_masks[i]

            results.append(det_dict)

        return results


    def _predict_onnx(self, image):
        h_orig, w_orig = image.shape[:2]
        
        # 1. Preprocessing: BGR to RGB, Resize, Float, Normalize
        # Guard: nvvidconv on Jetson may output BGRx (4-channel) even when
        # pipeline requests format=BGR.  Drop the alpha channel if present.
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
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
            # Keep raw logits (typically 144x144). Will resize directly to bounding box crop later.
            decoded_masks = list(kept_masks)
                
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
