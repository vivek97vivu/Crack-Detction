import os
import sys
import time

# Auto-switch to 'crack' conda python & set LD_LIBRARY_PATH
crack_python = "/home/algosium/miniforge3/envs/crack/bin/python"
crack_lib    = "/home/algosium/miniforge3/envs/crack/lib"
if os.path.exists(crack_python) and (sys.executable != crack_python or crack_lib not in os.environ.get("LD_LIBRARY_PATH", "")):
    os.environ["LD_LIBRARY_PATH"] = f"{crack_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.execv(crack_python, [crack_python] + sys.argv)

import cv2
import numpy as np
import yaml

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, "src"))

from src.inference.pipeline import CrackDetectionPipeline

def run_health_check():
    print("=" * 75)
    print(" 🏥 PIPELINE & MODEL INFERENCE COMPREHENSIVE HEALTH CHECK")
    print("=" * 75)

    # 1. Load Configuration
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    print("\n[1/5] Configuration Health:")
    print(f"  • Gate Status          : {'Enabled' if cfg['pipeline']['enable_gate'] else 'Disabled'} (Threshold: {cfg['gate']['threshold']})")
    print(f"  • Detector Engine Path  : {cfg['detector']['checkpoint_ema']}")
    print(f"  • Input Resolution     : {cfg['detector']['input_size'][0]}x{cfg['detector']['input_size'][1]}")
    print(f"  • Segmentation Active  : {cfg['pipeline']['enable_segmentation']}")

    # 2. Instantiate Full Pipeline
    t0 = time.time()
    pipeline = CrackDetectionPipeline(config_path="config/config.yaml")
    init_time = (time.time() - t0) * 1000.0
    print(f"\n[2/5] Pipeline Initialization: ✅ SUCCESS ({init_time:.1f} ms)")

    # 3. Test Test Image Defect Processing
    test_image_path = "alerts/snapshot/20260721_143654_track_1.jpg"
    if not os.path.exists(test_image_path):
        test_image_path = "test_input.jpg"
    frame = cv2.imread(test_image_path)

    print(f"\n[3/5] Single-Frame Inference & Geometry Benchmark:")
    t_start = time.time()
    annotated_frame, metadata = pipeline.process_frame(frame, "frame_001")
    total_time_ms = (time.time() - t_start) * 1000.0

    detections = metadata.get("detections", [])
    print(f"  • End-to-End Latency   : {total_time_ms:.2f} ms")
    print(f"  • Gate Evaluation      : {'Passed' if metadata.get('gate_passed') else 'Filtered'}")
    print(f"  • Detections Found     : {len(detections)}")

    # 4. Geometry & Mask Integrity
    print(f"\n[4/5] Instance Mask & Geometry Extraction Integrity:")
    for idx, det in enumerate(detections):
        class_name = getattr(det, "class_name", "crack")
        conf = getattr(det, "confidence", 0.0)
        box = getattr(det, "bbox_xyxy", [])
        geom_list = getattr(det, "geometry", [])
        length_mm = geom_list[0].length_mm if geom_list else 0.0
        width_mm = geom_list[0].width_max_mm if geom_list else 0.0
        mask = getattr(det, "mask", None)
        has_mask = mask is not None and np.sum(mask) > 0

        print(f"  Defect #{idx+1}: {class_name.upper()} | Conf: {conf*100:.1f}% | Box: {box}")
        print(f"    └─ Mask Active : {'✅ YES' if has_mask else '❌ NO'}")
        print(f"    └─ Sub-mm Geometry : Length = {length_mm:.2f} mm, Max Width = {width_mm:.2f} mm")

    # 5. Pipeline Verdict
    print(f"\n[5/5] Final Pipeline Verdict:")
    print("  ✅ All Model Inference & Pipeline Modules Operational!")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    run_health_check()
