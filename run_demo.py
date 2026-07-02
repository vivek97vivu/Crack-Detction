import cv2
import numpy as np
import os
import json
import time

from myproj.inference.pipeline import CrackDetectionPipeline
from myproj.utils.alerting import map_severity

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
    # We use checkpoint_best_ema(4).pth for detector.
    # We set gate_checkpoint=None to initialize a standard MobileNetV3 with pre-trained ImageNet weights.
    # We set segmenter_checkpoint=None to use the heuristic-based segmenter for the demo.
    print("\nInitializing Crack Detection Pipeline...")
    abs_detector_checkpoint = os.path.abspath("checkpoint_best_ema(4).pth")
    pipeline = CrackDetectionPipeline(
        detector_checkpoint=abs_detector_checkpoint,
        gate_checkpoint=None,
        segmenter_checkpoint=None,
        gate_threshold=0.2,       # Low gate threshold for demo
        detector_threshold=0.1,   # Low detector threshold for demo
        alerts_log="alerts.log",
        fallback_to_heuristic=True
    )
    
    # Run the pipeline on the input image
    print("\nProcessing test image...")
    # Load image for processing
    img_bgr = cv2.imread(input_img_path)
    frame_id = "test_input.jpg"
    
    # Process
    annotated_frame, metadata = pipeline.process_frame(img_bgr, frame_id=frame_id, px_to_mm=0.15)
    
    # Check if detector found anything on this synthetic image.
    # If not, we simulate a mock detection of a crack to demonstrate the segmenter + geometry + alerting stages.
    if len(metadata["detections"]) == 0:
        print("\n[Demo Note] RF-DETR did not find cracks on the synthetic image (expected for simple line drawings).")
        print("Injecting a simulated crack detection bounding box to demonstrate the downstream segmenter, geometry, and alerting stages...")
        
        # Inject mock crack detection box corresponding to our drawn crack
        # Box format: [x1, y1, x2, y2]
        mock_box = [80, 130, 700, 500]
        
        # Manually run downstream segmenter on the crop
        x1, y1, x2, y2 = mock_box
        crop_rgb = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        
        # Segmenter
        mask = pipeline.segmenter.predict(crop_rgb)
        
        # Geometry
        geom = extract_geometry(mask, px_to_mm=0.15) # 0.15 mm per pixel calibration
        if geom:
            geom["bounding_box"][0] += x1
            geom["bounding_box"][1] += y1
            
            # Severity mapping & Alerting
            severity = map_severity(geom["max_width_mm"], geom["length_mm"])
            
            # Trigger alert
            pipeline.alert_system.trigger_alert(
                severity_info=severity,
                frame_id=frame_id,
                max_width_mm=geom["max_width_mm"],
                length_mm=geom["length_mm"]
            )
            
            # Draw HUD elements manually on annotated_frame for the output image
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_frame, f"crack (simulated) 0.92", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            mask_bgr = np.zeros_like(img_bgr[y1:y2, x1:x2])
            mask_bgr[mask > 0] = [0, 0, 255]
            crop_bgr = annotated_frame[y1:y2, x1:x2]
            cv2.addWeighted(crop_bgr, 0.7, mask_bgr, 0.3, 0, crop_bgr)
            
            metrics_text = f"W:{geom['max_width_mm']:.2f}mm L:{geom['length_mm']:.1f}mm S:{severity['level']}"
            cv2.putText(annotated_frame, metrics_text, (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            
            # Update metadata
            metadata["detections"].append({
                "box": mock_box,
                "confidence": 0.92,
                "class_id": 0,
                "class_name": "crack",
                "geometry": geom,
                "severity": severity
            })
            
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
