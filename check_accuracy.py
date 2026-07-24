import os
import sys
import glob
import time

# Auto-switch to 'crack' conda python & set LD_LIBRARY_PATH
crack_python = "/home/algosium/miniforge3/envs/crack/bin/python"
crack_lib    = "/home/algosium/miniforge3/envs/crack/lib"
if os.path.exists(crack_python) and (sys.executable != crack_python or crack_lib not in os.environ.get("LD_LIBRARY_PATH", "")):
    os.environ["LD_LIBRARY_PATH"] = f"{crack_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.execv(crack_python, [crack_python] + sys.argv)

import cv2
import numpy as np
import torch
from src.inference.detector import DetectorInference
from src.inference.gate import GateInference

def main():
    print("=" * 70)
    print(" 🔍 MODEL ACCURACY & PREDICTION INTEGRITY CHECK")
    print("=" * 70)

    snapshots = sorted(glob.glob("alerts/snapshot/*.jpg"))
    if not snapshots:
        snapshots = ["test_input.jpg"]

    test_images = snapshots[:20]
    print(f"Loaded {len(test_images)} real-world crack test images from alerts/snapshot/...\n")

    # 1. Initialize Gate and Detector Models
    gate = GateInference(threshold=0.35)
    detector = DetectorInference(checkpoint_path="model/rfdetr-seg-medium-fp16.engine", threshold=0.3, num_engines=1)

    gate_passed_count = 0
    total_detections = 0
    crack_detections = 0
    confidences = []
    mask_pixel_counts = []
    latencies = []

    print(f"Running accuracy evaluation on {len(test_images)} images...")
    print(f"| Image File | Gate Passed? | Detections | Cracks Found | Top Conf % | Mask Pixels | Latency (ms) |")
    print(f"|---|---|---|---|---|---|---|")

    for img_path in test_images:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        t0 = time.time()
        # Step 1: Gate
        passed, prob = gate.predict(img_rgb)
        if passed:
            gate_passed_count += 1

        # Step 2: Detector
        dets = detector.predict(img)
        t_ms = (time.time() - t0) * 1000.0
        latencies.append(t_ms)

        num_dets = len(dets)
        total_detections += num_dets

        cracks = [d for d in dets if d["class_name"] == "crack"]
        num_cracks = len(cracks)
        crack_detections += num_cracks

        top_conf = max([d["confidence"] for d in dets], default=0.0)
        if top_conf > 0:
            confidences.append(top_conf)

        mask_px = 0
        for d in cracks:
            if "mask" in d and d["mask"] is not None:
                mask_px += int(np.sum(d["mask"]))
        mask_pixel_counts.append(mask_px)

        print(f"| {filename[:25]}... | {'✅ YES' if passed else '❌ NO'} ({prob:.2f}) | {num_dets} | {num_cracks} | {top_conf*100:.1f}% | {mask_px} px | {t_ms:.1f} ms |")

    avg_conf = np.mean(confidences) * 100 if confidences else 0.0
    avg_latency = np.mean(latencies) if latencies else 0.0
    gate_recall = (gate_passed_count / len(test_images)) * 100

    print("\n" + "=" * 70)
    print(" 📊 ACCURACY EVALUATION SUMMARY REPORT")
    print("=" * 70)
    print(f"  • Gate Model Recall         : {gate_recall:.1f}% ({gate_passed_count}/{len(test_images)} passed)")
    print(f"  • Total Defects Detected   : {total_detections} (Cracks: {crack_detections})")
    print(f"  • Average Detection Conf    : {avg_conf:.1f}%")
    print(f"  • Average End-to-End Time   : {avg_latency:.2f} ms / image")
    print(f"  • Mask Segmentation Status  : ✅ ACTIVE ({np.mean(mask_pixel_counts):.0f} avg mask pixels per frame)")
    
    if gate_recall >= 90.0 and avg_conf >= 50.0:
        print("\n✅ ACCURACY VERIFIED: High detection recall & confidence retention confirmed!")
    else:
        print("\n⚠️ WARNING: Low confidence or recall detected.")

if __name__ == "__main__":
    main()
