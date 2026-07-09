import cv2
import numpy as np
import os
import sys
import json
import time
import warnings

# Suppress deprecation and future warnings from third-party libraries
warnings.filterwarnings("ignore")

# Add src/ to the python path so modules can be imported directly
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from inference.pipeline import CrackDetectionPipeline, Detection, detection_to_dict
from utils.geometry import extract_geometry, draw_geometry
from utils.severity import SeverityLevel
from utils.visualization import draw_mask_overlay, draw_severity_badge
from utils.config import load_config

def get_video_capture(camera_config):
    source = camera_config.get("source")
    use_gstreamer = camera_config.get("use_gstreamer", False)
    
    # If source is a string representing a local file, verify it exists first
    if isinstance(source, str) and not (source.startswith("rtsp://") or source.startswith("http://") or source.startswith("https://")):
        expanded_source = os.path.expanduser(source)
        if not os.path.exists(expanded_source):
            print(f"\n[Error] Local video or image source file not found at: {source}")
            print(f"Please check the file path configuration in config/config.yaml.")
            return cv2.VideoCapture()
        source = expanded_source
        
    if use_gstreamer and isinstance(source, str) and source.startswith("rtsp://"):
        # Build GStreamer RTSP pipeline string for OpenCV
        codec = camera_config.get("codec", "h265")
        latency = camera_config.get("latency", 0)
        protocols = camera_config.get("protocols", "tcp")
        
        depay = "rtph265depay" if codec == "h265" else "rtph264depay"
        parser = "h265parse" if codec == "h265" else "h264parse"
        decoder = "avdec_h265" if codec == "h265" else "avdec_h264"
        
        gstreamer_str = (
            f"rtspsrc location=\"{source}\" latency={latency} protocols={protocols} ! "
            f"{depay} ! {parser} ! {decoder} ! videoconvert ! appsink drop=true sync=false"
        )
        print(f"Opening RTSP stream using GStreamer: {gstreamer_str}")
        cap = cv2.VideoCapture(gstreamer_str, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap
            
        print("[Warning] GStreamer pipeline failed to open. Falling back to FFMPEG reader...")
        
    # Standard OpenCV source (int or string)
    print(f"Opening video source via FFMPEG: {source}")
    return cv2.VideoCapture(source)

def generate_synthetic_image(output_path):
    """
    Generates a synthetic concrete-like image with a drawn crack.
    """
    print(f"Generating synthetic test image at {output_path}...")
    # Create base gray image (concrete texture)
    img = np.full((600, 800, 3), 180, dtype=np.uint8)
    
    # Add noise to simulate concrete texture
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Draw a crack (dark, jagged line)
    points = [
        (100, 150), (180, 190), (270, 220), (350, 290), 
        (420, 310), (510, 390), (590, 420), (680, 480)
    ]
    for i in range(len(points) - 1):
        pt1 = points[i]
        pt2 = points[i+1]
        # Draw main crack line
        cv2.line(img, pt1, pt2, (40, 40, 40), thickness=4, lineType=cv2.LINE_AA)
        # Draw some tiny branching lines
        if i % 2 == 0:
            cv2.line(img, pt2, (pt2[0] + 15, pt2[1] + 10), (50, 50, 50), thickness=2)
            
    cv2.imwrite(output_path, img)
    return img

def main():
    # Check for enabled cameras in configuration first
    config = load_config()
    cameras = config.get("cameras", [])
    enabled_cameras = [cam for cam in cameras if cam.get("enabled", True)]
    
    # Initialize pipeline
    # Configuration will be loaded automatically from config/config.yaml
    print("\nInitializing Crack Detection Pipeline using config/config.yaml...")
    pipeline = CrackDetectionPipeline()
    
    if enabled_cameras:
        camera_cfg = enabled_cameras[0]
        cam_id = camera_cfg.get("id")
        cam_name = camera_cfg.get("name", cam_id)
        print(f"\n[Live Mode] Starting live stream for camera: {cam_name} ({cam_id}). Press 'q' to quit.")
        
        cap = get_video_capture(camera_cfg)
        if not cap.isOpened():
            print(f"Error: Could not open video source for camera {cam_id}")
            return
            
        frame_skip = camera_cfg.get("frame_skip", 1)
        if frame_skip < 1:
            frame_skip = 1
        if frame_skip > 1:
            print(f"Frame skip enabled: processing 1 frame every {frame_skip} frames to accelerate playback.")
            
        # Get playback FPS to control display speed (0 or unset means no artificial delay)
        playback_fps = camera_cfg.get("playback_fps", 0)
        delay_ms = max(1, int(1000 / playback_fps)) if playback_fps > 0 else 1
        if playback_fps > 0:
            print(f"Target playback speed: {playback_fps} FPS (Frame delay: {delay_ms}ms)")
            
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream or failed to fetch frame.")
                    break
                    
                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue
                    
                frame_id = f"{cam_id}_frame_{frame_count}"
                
                # Process the frame
                annotated, meta = pipeline.process_frame(frame, frame_id=frame_id)
                
                # Try to show the live feed if GUI backend is available
                try:
                    cv2.imshow(f"Crack Detection Live - {cam_name}", annotated)
                    if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
                        break
                except Exception:
                    # Headless mode fallback: print progress periodically
                    if frame_count % 30 == 0:
                        print(f"[Headless Live] Processed {frame_count} frames...")
        finally:
            cap.release()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
    else:
        # No cameras enabled: run synthetic self-test demo
        input_img_path = "test_input.jpg"
        output_img_path = "test_output.jpg"
        
        # Generate synthetic input
        generate_synthetic_image(input_img_path)
        
        # Run the pipeline on the input image
        print("\nProcessing test image...")
        # Load image for processing
        img_bgr = cv2.imread(input_img_path)
        frame_id = "test_input.jpg"
        
        # Process
        annotated_frame, metadata = pipeline.process_frame(img_bgr, frame_id=frame_id)
        
        # Check if detector found anything on this synthetic image.
        if len(metadata["detections"]) == 0:
            print("\n[Demo Note] RF-DETR did not find cracks on the synthetic image (expected for simple line drawings).")
            print("Injecting a simulated crack detection bounding box to demonstrate the downstream segmenter, geometry, and alerting stages...")
            
            # Inject mock crack detection box corresponding to our drawn crack
            mock_box = [80, 130, 700, 500]
            x1, y1, x2, y2 = mock_box
            
            # Manually run downstream segmenter on the crop
            crop_rgb = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            mask = pipeline.segmenter.predict(crop_rgb)
            
            # Geometry
            geom_list = extract_geometry(
                mask, 
                pixel_per_mm=pipeline.px_per_mm, 
                min_length_px=pipeline.min_length_px,
                min_area_px=pipeline.min_area_px,
                sample_interval=pipeline.sample_interval
            )
            
            if geom_list:
                # Severity mapping & Alerting
                severity_results = [
                    pipeline.severity_mapper.classify(g.width_mean_mm, g.length_mm)
                    for g in geom_list
                ]
                worst_sev = pipeline.severity_mapper.worst_level(severity_results)
                
                det = Detection(
                    bbox_xyxy=np.array(mock_box, dtype=int),
                    class_name="crack",
                    confidence=0.92,
                    class_id=0,
                    track_id=1,
                    geometry=geom_list,
                    severity=worst_sev
                )
                
                # Draw HUD & overlays manually on annotated_frame for the output image
                annotated_frame = draw_mask_overlay(annotated_frame, mask, det.bbox_xyxy)
                annotated_frame = draw_severity_badge(annotated_frame, worst_sev, det.bbox_xyxy)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"crack ID:1 0.92", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                for geom in geom_list:
                    annotated_frame = draw_geometry(annotated_frame, geom, offset_xy=(x1, y1))
                    
                # Trigger alert and save snapshots
                pipeline._handle_alert_and_snapshots(det, annotated_frame, frame_id)
                    
                # Update metadata
                metadata["detections"].append(detection_to_dict(det))
                
        # Save output image
        cv2.imwrite(output_img_path, annotated_frame)
        print(f"Annotated output image saved to {output_img_path}")
        
        # Print results
        print("\n--- Pipeline Execution Metadata ---")
        print(json.dumps(metadata, indent=2))
        
        print("\n--- Severity Alerting Log (alerts.log) ---")
        if os.path.exists("alerts.log"):
            with open("alerts.log", "r") as f:
                print(f.read())
            
if __name__ == "__main__":
    main()
