import cv2
import numpy as np
import os
import json
import time

from myproj.inference.pipeline import CrackDetectionPipeline, Detection, detection_to_dict
from myproj.utils.geometry import extract_geometry, draw_geometry
from myproj.utils.severity import SeverityLevel
from myproj.utils.visualization import draw_mask_overlay, draw_severity_badge

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
    input_img_path = "test_input.jpg"
    output_img_path = "test_output.jpg"
    
    # Generate synthetic input
    generate_synthetic_image(input_img_path)
    
    # Initialize pipeline
    # Configuration will be loaded automatically from config/config.yaml
    print("\nInitializing Crack Detection Pipeline using config/config.yaml...")
    pipeline = CrackDetectionPipeline()
    
    # Run the pipeline on the input image
    print("\nProcessing test image...")
    # Load image for processing
    img_bgr = cv2.imread(input_img_path)
    frame_id = "test_input.jpg"
    
    # Process
    annotated_frame, metadata = pipeline.process_frame(img_bgr, frame_id=frame_id)
    
    # Check if detector found anything on this synthetic image.
    # If not, we simulate a mock detection of a crack to demonstrate the segmenter + geometry + alerting stages.
    if len(metadata["detections"]) == 0:
        print("\n[Demo Note] RF-DETR did not find cracks on the synthetic image (expected for simple line drawings).")
        print("Injecting a simulated crack detection bounding box to demonstrate the downstream segmenter, geometry, and alerting stages...")
        
        # Inject mock crack detection box corresponding to our drawn crack
        # Box format: [x1, y1, x2, y2]
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
            
            # Trigger alert and save snapshots
            pipeline._handle_alert_and_snapshots(det, img_bgr, frame_id)
            
            # Draw HUD & overlays manually on annotated_frame for the output image
            annotated_frame = draw_mask_overlay(annotated_frame, mask, det.bbox_xyxy)
            annotated_frame = draw_severity_badge(annotated_frame, worst_sev, det.bbox_xyxy)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"crack ID:1 0.92", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            for geom in geom_list:
                annotated_frame = draw_geometry(annotated_frame, geom, offset_xy=(x1, y1))
                
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
