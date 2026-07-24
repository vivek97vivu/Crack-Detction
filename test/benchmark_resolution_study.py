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
import subprocess
from src.inference.detector import DetectorInference
from src.inference.gate import GateInference

def evaluate_engine_accuracy(engine_path, shape):
    snapshots = sorted(glob.glob("alerts/snapshot/*.jpg"))
    if not snapshots:
        snapshots = ["test_input.jpg"]
    test_images = snapshots[:20]

    detector = DetectorInference(checkpoint_path=engine_path, threshold=0.3, num_engines=1)
    
    total_dets = 0
    total_cracks = 0
    confidences = []
    latencies = []

    for p in test_images:
        img = cv2.imread(p)
        if img is None:
            continue
        t0 = time.time()
        dets = detector.predict(img)
        t_ms = (time.time() - t0) * 1000.0
        latencies.append(t_ms)

        total_dets += len(dets)
        cracks = [d for d in dets if d["class_name"] == "crack"]
        total_cracks += len(cracks)
        
        top_conf = max([d["confidence"] for d in dets], default=0.0)
        if top_conf > 0:
            confidences.append(top_conf)

    avg_conf = np.mean(confidences) * 100 if confidences else 0.0
    avg_lat  = np.mean(latencies) if latencies else 0.0
    return total_dets, total_cracks, avg_conf, avg_lat


def run_scaling_test(engine_path, shape, num_cams=50):
    # Temporarily update config.yaml detector checkpoint_ema and input_size
    import yaml
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    orig_chk = cfg["detector"]["checkpoint_ema"]
    orig_sz  = cfg["detector"]["input_size"]

    cfg["detector"]["checkpoint_ema"] = engine_path
    cfg["detector"]["input_size"] = [shape, shape]

    with open("config/config.yaml", "w") as f:
        yaml.dump(cfg, f)

    try:
        cmd = [
            sys.executable, "benchmark_scale.py",
            "--cameras", str(num_cams),
            "--workers", "16",
            "--engines", "3",
            "--scheduler-workers", "3",
            "--duration", "15"
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        out = proc.stdout

        total_fps = 0.0
        fps_per_cam = 0.0
        gpu_pct = 0.0
        cpu_pct = 0.0

        for line in out.splitlines():
            if "Total FPS:" in line and "FPS/Cam:" in line:
                parts = line.split("|")
                for p in parts:
                    p = p.strip()
                    if "Total FPS:" in p:
                        total_fps = float(p.split(":")[1])
                    elif "FPS/Cam:" in p:
                        fps_per_cam = float(p.split(":")[1])
                    elif "CPU:" in p:
                        cpu_pct = float(p.split(":")[1].replace("%", ""))
                    elif "GPU:" in p:
                        gpu_pct = float(p.split(":")[1].replace("%", ""))

        return total_fps, fps_per_cam, gpu_pct, cpu_pct

    finally:
        # Restore original config
        cfg["detector"]["checkpoint_ema"] = orig_chk
        cfg["detector"]["input_size"] = orig_sz
        with open("config/config.yaml", "w") as f:
            yaml.dump(cfg, f)


def main():
    print("=" * 75)
    print(" 📐 RESOLUTION STUDY: Comparing 576x576 vs 528x528 vs 504x504 Engines")
    print("=" * 75)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    configs = [
        {"shape": 576, "engine": os.path.join(base_dir, "model", "rfdetr-seg-medium-fp16.engine")},
        {"shape": 528, "engine": os.path.join(base_dir, "model", "rfdetr-seg-medium-528-fp16.engine")},
        {"shape": 504, "engine": os.path.join(base_dir, "model", "rfdetr-seg-medium-504-fp16.engine")},
    ]

    results = []

    for c in configs:
        shape = c["shape"]
        engine = c["engine"]
        print(f"\n--- Testing Input Size: {shape}x{shape} ({engine}) ---")

        # 1. Accuracy test
        print(f"  [1/2] Evaluating Accuracy on 20 snapshot test images...")
        t_dets, t_cracks, avg_conf, avg_lat = evaluate_engine_accuracy(engine, shape)
        print(f"        -> Detections: {t_dets} | Cracks: {t_cracks} | Avg Conf: {avg_conf:.1f}% | Latency: {avg_lat:.1f}ms")

        # 2. Scaling benchmark on 50 cameras
        print(f"  [2/2] Running 50-Camera Scale Benchmark (3x Engine Pool)...")
        tot_fps, cam_fps, gpu_pct, cpu_pct = run_scaling_test(engine, shape, num_cams=50)
        print(f"        -> 50 Cams Throughput: {tot_fps:.2f} Total FPS | {cam_fps:.2f} FPS/Cam | GPU: {gpu_pct:.1f}%")

        results.append({
            "shape": shape,
            "dets": t_dets,
            "cracks": t_cracks,
            "conf": avg_conf,
            "lat": avg_lat,
            "total_fps": tot_fps,
            "cam_fps": cam_fps,
            "gpu": gpu_pct,
            "cpu": cpu_pct,
        })

    print("\n" + "=" * 80)
    print(" 📊 FINAL RESOLUTION COMPARISON SUMMARY TABLE")
    print("=" * 80)
    print("| Resolution | Total FPS (50 Cams) | FPS / Camera | Avg Conf % | Cracks Detected | Single Frame Latency | GPU % |")
    print("|---|---|---|---|---|---|---|")

    for r in results:
        print(f"| **{r['shape']} × {r['shape']}** | **{r['total_fps']:.2f} FPS** | **{r['cam_fps']:.2f} FPS** | **{r['conf']:.1f}%** | {r['cracks']} | {r['lat']:.1f} ms | {r['gpu']:.1f}% |")

    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
