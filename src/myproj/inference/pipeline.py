import cv2
import numpy as np
import time
import os

from myproj.inference.gate import GateInference
from myproj.inference.detector import DetectorInference
from myproj.inference.segmenter import SegmenterInference
from myproj.utils.geometry import extract_geometry
from myproj.utils.alerting import map_severity, AlertSystem

class CrackDetectionPipeline:
    def __init__(self, 
                 detector_checkpoint, 
                 gate_checkpoint=None, 
                 segmenter_checkpoint=None,
                 gate_threshold=0.4, 
                 detector_threshold=0.3,
                 alerts_log="alerts.log",
                 fallback_to_heuristic=True):
        
        # 1. Gate Stage
        self.gate = GateInference(
            checkpoint_path=gate_checkpoint, 
            threshold=gate_threshold
        )
        
        # 2. Detector Stage (RF-DETR)
        self.detector = DetectorInference(
            checkpoint_path=detector_checkpoint, 
            threshold=detector_threshold
        )
        
        # 3. Segmenter Stage (U-Net)
        self.segmenter = SegmenterInference(
            checkpoint_path=segmenter_checkpoint,
            fallback_to_heuristic=fallback_to_heuristic
        )
        
        # 4. Alert &Severity System
        self.alert_system = AlertSystem(log_path=alerts_log)
        
    def process_frame(self, frame_bgr, frame_id, px_to_mm=0.1):
        """
        Processes a single frame.
        Input is a BGR OpenCV image.
        Returns:
            processed_frame: Annotated BGR frame.
            pipeline_metadata: Summary dictionary of detections, severity, and status.
        """
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
        gate_passed, prob = self.gate.predict(frame_rgb)
        metadata["gate_passed"] = bool(gate_passed)
        metadata["gate_probability"] = float(prob)
        
        annotated_frame = frame_bgr.copy()
        
        if not gate_passed:
            # Draw gate drop HUD
            hud_text = f"GATE: DROP (P={prob:.3f})"
            cv2.putText(annotated_frame, hud_text, (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            metadata["processing_time_ms"] = float((time.time() - start_time) * 1000)
            return annotated_frame, metadata
            
        # --- Stage 2: RF-DETR Detector ---
        detections = self.detector.predict(frame_rgb)
        metadata["detections"] = detections
        
        # Draw gate pass HUD
        cv2.putText(annotated_frame, f"GATE: PASS (P={prob:.3f})", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        for det in detections:
            box = det["box"]
            cls_name = det["class_name"]
            conf = det["confidence"]
            
            x1, y1, x2, y2 = map(int, box)
            # Clip boxes to image dimensions
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Draw bounding box
            color = (255, 0, 0) if cls_name == "crack" else (0, 255, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, f"{cls_name} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # --- Stage 3: Segmentation (only for cracks) ---
            if cls_name == "crack" and (x2 > x1) and (y2 > y1):
                crop = frame_rgb[y1:y2, x1:x2]
                mask = self.segmenter.predict(crop)
                
                # --- Stage 4: Geometry ---
                geom = extract_geometry(mask, px_to_mm=px_to_mm)
                if geom:
                    # Map box coordinates back to original frame
                    geom["bounding_box"][0] += x1
                    geom["bounding_box"][1] += y1
                    
                    # Store geometry in detection dictionary
                    det["geometry"] = geom
                    
                    # --- Stage 5: Severity mapping & Alerting ---
                    severity = map_severity(geom["max_width_mm"], geom["length_mm"])
                    det["severity"] = severity
                    
                    self.alert_system.trigger_alert(
                        severity_info=severity,
                        frame_id=frame_id,
                        max_width_mm=geom["max_width_mm"],
                        length_mm=geom["length_mm"]
                    )
                    
                    # Overlay binary mask on frame
                    # Make colored overlay
                    mask_bgr = np.zeros_like(crop)
                    mask_bgr[mask > 0] = [0, 0, 255] # Red overlay for cracks
                    crop_bgr = annotated_frame[y1:y2, x1:x2]
                    cv2.addWeighted(crop_bgr, 0.7, mask_bgr, 0.3, 0, crop_bgr)
                    
                    # Overlay metrics on frame
                    metrics_text = f"W:{geom['max_width_mm']:.2f}mm L:{geom['length_mm']:.1f}mm S:{severity['level']}"
                    cv2.putText(annotated_frame, metrics_text, (x1, y2 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                                
        metadata["processing_time_ms"] = float((time.time() - start_time) * 1000)
        return annotated_frame, metadata

    def process_image_file(self, input_path, output_path, px_to_mm=0.1):
        """
        Processes a single image file, saves output image, and returns metadata.
        """
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {input_path}")
            
        annotated, meta = self.process_frame(img, frame_id=os.path.basename(input_path), px_to_mm=px_to_mm)
        cv2.imwrite(output_path, annotated)
        return meta
