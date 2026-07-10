import cv2
import numpy as np
import time
import os
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Tuple

from inference.detector import DetectorInference, Detection
from inference.segmenter import SegmenterInference
from inference.gate import GateInference
from utils.geometry import CrackGeometry, draw_geometry
from utils.severity import APISeverityMapper, SeverityResult, SeverityLevel
from utils.visualization import draw_detections, draw_mask_overlay, draw_severity_badge, draw_pipeline_hud
from utils.tracking import SimpleBBoxTracker
from utils.config import load_config, resolve_path

def detection_to_dict(det: Detection):
    geom_list = []
    if det.geometry:
        for g in det.geometry:
            geom_list.append({
                "length_mm": g.length_mm,
                "width_mean_mm": g.width_mean_mm,
                "width_max_mm": g.width_max_mm,
                "aspect_ratio": g.aspect_ratio,
                "orientation_deg": g.orientation_deg,
                "pixel_area": g.pixel_area,
                "skeleton_path": g.skeleton_path,
                "is_valid_crack": g.is_valid_crack
            })
            
    sev_dict = None
    if det.severity:
        sev_dict = {
            "level": int(det.severity.level),
            "level_name": det.severity.level.name,
            "action": det.severity.action,
            "reinspection_days": det.severity.reinspection_days,
            "width_mm": det.severity.width_mm,
            "length_mm": det.severity.length_mm,
            "trigger_reason": det.severity.trigger_reason,
            "surface_only_note": det.severity.surface_only_note
        }
        
    return {
        "bbox_xyxy": det.bbox_xyxy.tolist(),
        "class_name": det.class_name,
        "confidence": det.confidence,
        "class_id": det.class_id,
        "track_id": det.track_id,
        "geometry": geom_list,
        "severity": sev_dict
    }

class CrackDetectionPipeline:
    def __init__(self, 
                 detector_checkpoint=None, 
                 gate_checkpoint=None,
                 segmenter_checkpoint=None,
                 gate_threshold=None,
                 detector_threshold=None,
                 alerts_log=None,
                 fallback_to_heuristic=None,
                 config_path=None):
        
        # Load configuration
        config = load_config(config_path)
        self.config = config
        
        p_cfg = config.get("pipeline", {})
        self.enable_gate = p_cfg.get("enable_gate", True)
        g_cfg = config.get("gate", {})
        d_cfg = config.get("detector", {})
        s_cfg = config.get("segmenter", {})
        geom_cfg = config.get("geometry", {})
        
        # Resolve detector checkpoint
        det_checkpoint = (
            detector_checkpoint if detector_checkpoint is not None 
            else (d_cfg.get("checkpoint") or d_cfg.get("checkpoint_ema") or "checkpoint_best_ema(4).pth")
        )
        det_checkpoint = resolve_path(det_checkpoint)
        det_threshold = detector_threshold if detector_threshold is not None else d_cfg.get("threshold", 0.3)
        
        # Resolve gate checkpoint
        g_checkpoint = gate_checkpoint if gate_checkpoint is not None else g_cfg.get("checkpoint", None)
        g_checkpoint = resolve_path(g_checkpoint)
        g_threshold = gate_threshold if gate_threshold is not None else g_cfg.get("threshold", 0.4)
        
        # Resolve segmenter checkpoint
        seg_checkpoint = segmenter_checkpoint if segmenter_checkpoint is not None else s_cfg.get("checkpoint", None)
        seg_checkpoint = resolve_path(seg_checkpoint)
        fallback = fallback_to_heuristic if fallback_to_heuristic is not None else s_cfg.get("fallback_to_heuristic", True)
        
        # 1. Gate Stage
        if self.enable_gate:
            self.gate = GateInference(
                checkpoint_path=g_checkpoint, 
                threshold=g_threshold
            )
        else:
            self.gate = None
        
        # 2. Detector Stage
        self.detector = DetectorInference(
            checkpoint_path=det_checkpoint, 
            threshold=det_threshold
        )
        if "target_classes" in d_cfg:
            self.detector.target_classes = d_cfg["target_classes"]
            
        # 3. Segmenter Stage
        self.segmenter = SegmenterInference(
            checkpoint_path=seg_checkpoint,
            fallback_to_heuristic=fallback
        )
        
        # 4. Severity Mapper
        self.severity_mapper = APISeverityMapper(config)
        
        # 5. Tracker
        self.tracker = SimpleBBoxTracker()
        
        # Other pipeline parameters
        self.px_per_mm = geom_cfg.get("pixel_per_mm", 10.0)
        self.min_length_px = geom_cfg.get("min_length_px", 20)
        self.min_area_px = geom_cfg.get("min_area_px", 50)
        self.sample_interval = geom_cfg.get("sample_interval", 5)
        
        self.alerts_log = resolve_path(alerts_log if alerts_log is not None else p_cfg.get("alerts_log", "log/alerts.log"))
        self.save_snapshots = p_cfg.get("save_snapshots", True)
        self.alerted_track_ids = set()
        
        # Load minimum consecutive frames configuration (default to 1)
        self.min_consecutive_frames = p_cfg.get("min_consecutive_frames", 1)
        self.track_frame_counts = {}
        
        # Load force split segmentation option (default to False)
        self.force_split_segmentation = p_cfg.get("force_split_segmentation", False)
        
        # Load stage enabled/disabled flags (default to True)
        self.enable_detection = p_cfg.get("enable_detection", True)
        self.enable_segmentation = p_cfg.get("enable_segmentation", True)
        
        # Generate a unique run timestamp prefix for the output filenames
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.alerts_json_dir = resolve_path(p_cfg.get("alerts_json_dir", "alerts/json"))
        self.alerts_snapshot_dir = resolve_path(p_cfg.get("alerts_snapshot_dir", "alerts/snapshot"))

    def process_frame(self, frame_bgr, frame_id, pixel_per_mm=None):
        if len(frame_bgr.shape) == 2:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
            
        if pixel_per_mm is None:
            pixel_per_mm = self.px_per_mm
            
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_bgr.shape[:2]
        
        metadata = {
            "frame_id": frame_id,
            "gate_passed": False,
            "gate_probability": 0.0,
            "detections": [],
            "processing_time_ms": 0.0
        }
        
        start_time = time.time()
        
        # --- Stage 1: Gate ---
        if self.enable_gate and self.gate is not None:
            gate_passed, prob = self.gate.predict(frame_rgb)
        else:
            gate_passed, prob = True, 1.0
            
        metadata["gate_passed"] = bool(gate_passed)
        metadata["gate_probability"] = float(prob)
        
        annotated_frame = frame_bgr.copy()
        
        if not gate_passed:
            annotated_frame = draw_pipeline_hud(annotated_frame, prob, 0, 0)
            metadata["processing_time_ms"] = float((time.time() - start_time) * 1000)
            return annotated_frame, metadata
        
        # --- Stage 2 & 3 Dynamic Paths ---
        if self.enable_detection:
            # Stage 2: Detector and tracker (handled inside detector.py)
            detections = self.detector.process(frame_rgb, self.tracker)
            
            # Update track frame counts
            for det in detections:
                if det.track_id is not None:
                    self.track_frame_counts[det.track_id] = self.track_frame_counts.get(det.track_id, 0) + 1
                    
            # Clean up inactive track counts to save memory
            active_track_ids = set(self.tracker.tracked_dets.keys())
            self.track_frame_counts = {
                tid: count for tid, count in self.track_frame_counts.items() if tid in active_track_ids
            }
            
            # Count variables
            n_detections = len(detections)
            n_cracks = sum(1 for d in detections if d.class_name == "crack")
            
            # Draw bounding boxes and HUD
            annotated_frame = draw_detections(annotated_frame, detections)
            annotated_frame = draw_pipeline_hud(annotated_frame, prob if self.enable_gate else None, n_detections, n_cracks)
            
            # Process each detection for segmentation if enabled
            for det in detections:
                if self.enable_segmentation:
                    x1, y1, x2, y2 = det.bbox_xyxy
                    
                    # --- Stage 3: Segmentation (only for cracks) ---
                    if det.class_name == "crack" and (x2 > x1) and (y2 > y1):
                        # Determine if we should use detector's mask or UNet segmenter
                        if det.mask is not None and not self.force_split_segmentation:
                            mask = det.mask.astype(np.uint8) * 255
                            mask = mask[y1:y2, x1:x2]
                            # Run geometry and severity mapping on the mask (handled inside segmenter.py)
                            geom_list, worst_sev = self.segmenter.process_mask(
                                mask, pixel_per_mm, self.min_length_px, self.min_area_px, self.sample_interval, self.severity_mapper
                            )
                        else:
                            crop = frame_rgb[y1:y2, x1:x2]
                            # Run crop segmentation, geometry, and severity mapping (handled inside segmenter.py)
                            mask, geom_list, worst_sev = self.segmenter.process_crop(
                                crop, pixel_per_mm, self.min_length_px, self.min_area_px, self.sample_interval, self.severity_mapper
                            )
                        
                        if geom_list:
                            det.geometry = geom_list
                            det.severity = worst_sev
                            
                            # Draw mask overlay
                            annotated_frame = draw_mask_overlay(annotated_frame, mask, det.bbox_xyxy)
                            
                            # Draw severity badge
                            annotated_frame = draw_severity_badge(annotated_frame, worst_sev, det.bbox_xyxy)
                            
                            # Draw centerline skeleton and width/length labels
                            for geom in geom_list:
                                # Draw geometry relative to bounding box offset
                                annotated_frame = draw_geometry(annotated_frame, geom, offset_xy=(x1, y1))
                                
                            # Save snapshot and trigger alerts
                            self._handle_alert_and_snapshots(det, annotated_frame, frame_id)
                            
                metadata["detections"].append(detection_to_dict(det))
                
        elif self.enable_segmentation:
            # --- Segmentation-only Mode: run directly on the full frame (handled inside segmenter.py) ---
            mask, geom_list, worst_sev = self.segmenter.process_crop(
                frame_rgb, pixel_per_mm, self.min_length_px, self.min_area_px, self.sample_interval, self.severity_mapper
            )
            
            annotated_frame = draw_pipeline_hud(annotated_frame, prob if self.enable_gate else None, 0, 1 if geom_list else 0)
            
            if geom_list:
                # Draw mask overlay on full frame
                annotated_frame = draw_mask_overlay(annotated_frame, mask, np.array([0, 0, w, h], dtype=int))
                
                # Draw severity badge (fixed location at top-left)
                annotated_frame = draw_severity_badge(annotated_frame, worst_sev, np.array([10, 50, 150, 100], dtype=int))
                
                # Draw centerline and labels
                for geom in geom_list:
                    annotated_frame = draw_geometry(annotated_frame, geom, offset_xy=(0, 0))
                    
                # Create a dummy detection object for logging / alerting in segmentation-only mode
                dummy_det = Detection(
                    bbox_xyxy=np.array([0, 0, w, h], dtype=int),
                    class_name="crack",
                    confidence=1.0,
                    class_id=0,
                    track_id=None
                )
                dummy_det.severity = worst_sev
                dummy_det.geometry = geom_list
                
                # Save snapshot and trigger alerts
                self._handle_alert_and_snapshots(dummy_det, annotated_frame, frame_id)
                metadata["detections"].append(detection_to_dict(dummy_det))
        else:
            # Both disabled (video passthrough)
            annotated_frame = draw_pipeline_hud(annotated_frame, prob if self.enable_gate else None, 0, 0)

        metadata["processing_time_ms"] = float((time.time() - start_time) * 1000)
        return annotated_frame, metadata

    def _handle_alert_and_snapshots(self, det: Detection, frame_bgr: np.ndarray, frame_id: str):
        if not det.severity:
            return
            
        track_id = det.track_id
        if track_id is not None:
            # If not tracked via process_frame, default to the threshold to allow immediate triggering
            frame_count = self.track_frame_counts.get(track_id, self.min_consecutive_frames)
            if frame_count < self.min_consecutive_frames:
                return
                
            if track_id in self.alerted_track_ids:
                return
            self.alerted_track_ids.add(track_id)
            
        level = int(det.severity.level)
        status = det.severity.level.name
        action = det.severity.action
        w_mm = det.severity.width_mm
        l_mm = det.severity.length_mm
        reason = det.severity.trigger_reason
        
        # Format logging output
        log_line = (
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"ALERT {status} (Level {level}) - Frame ID: {frame_id} - Track ID: {track_id} - "
            f"Width: {w_mm:.3f}mm, Length: {l_mm:.1f}mm - "
            f"Trigger Reason: {reason} - Recommended Action: {action}\n"
        )
        print(log_line.strip())
        
        # Append log
        try:
            log_dir = os.path.dirname(self.alerts_log)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(self.alerts_log, "a") as f:
                f.write(log_line)
        except Exception as e:
            print(f"Error writing to alerts log: {e}")
            
        # Snapshot saving
        if self.save_snapshots:
            os.makedirs(self.alerts_json_dir, exist_ok=True)
            os.makedirs(self.alerts_snapshot_dir, exist_ok=True)
            
            # Form clean filename prefix
            if track_id is not None:
                file_prefix = f"{self.run_timestamp}_track_{track_id}"
            else:
                frame_id_clean = os.path.basename(str(frame_id)).replace(".", "_").replace("/", "_").replace("\\", "_")
                file_prefix = f"{self.run_timestamp}_frame_{frame_id_clean}"
            
            # JSON file snapshot
            json_filename = f"{file_prefix}.json"
            json_path = os.path.join(self.alerts_json_dir, json_filename)
            
            snapshot_data = {
                "track_id": track_id,
                "frame_id": frame_id,
                "timestamp": datetime.now().isoformat(),
                "detection": detection_to_dict(det)
            }
            
            try:
                with open(json_path, "w") as f:
                    json.dump(snapshot_data, f, indent=2)
            except Exception as e:
                print(f"Error saving alert JSON: {e}")
                
            # JPEG Image snapshot (save full frame with marked detections)
            crop_filename = f"{file_prefix}.jpg"
            crop_path = os.path.join(self.alerts_snapshot_dir, crop_filename)
            
            try:
                cv2.imwrite(crop_path, frame_bgr)
            except Exception as e:
                print(f"Error saving full snapshot image: {e}")

    def process_image_file(self, input_path, output_path, pixel_per_mm=None):
        """
        Processes a single image file, saves output image, and returns metadata.
        """
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {input_path}")
            
        annotated, meta = self.process_frame(img, frame_id=os.path.basename(input_path), pixel_per_mm=pixel_per_mm)
        cv2.imwrite(output_path, annotated)
        return meta
