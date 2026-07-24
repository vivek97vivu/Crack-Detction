import os
import sys
import subprocess
import time

# Auto-switch to 'crack' conda python & set LD_LIBRARY_PATH
crack_python = "/home/algosium/miniforge3/envs/crack/bin/python"
crack_lib    = "/home/algosium/miniforge3/envs/crack/lib"
if os.path.exists(crack_python) and (sys.executable != crack_python or crack_lib not in os.environ.get("LD_LIBRARY_PATH", "")):
    os.environ["LD_LIBRARY_PATH"] = f"{crack_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.execv(crack_python, [crack_python] + sys.argv)

def main():
    counts_to_test = [30, 40, 50, 54, 60]
    duration_per_test = 20  # seconds

    print(f"\n============================================================")
    print(f" 🎯 1.0 FPS CAMERA CAPACITY SWEEP (Isolated Subprocess Benchmarks)")
    print(f" Testing Camera Counts: {counts_to_test}")
    print(f" Target: Find Maximum Camera Scale with FPS/Camera >= 1.0 FPS")
    print(f"============================================================\n")

    results = []

    for num_cams in counts_to_test:
        print(f"\n--- Running Isolated Test for {num_cams} Cameras ---")
        cmd = [
            sys.executable, "benchmark_scale.py",
            "--cameras", str(num_cams),
            "--workers", "16",
            "--engines", "3",
            "--scheduler-workers", "3",
            "--duration", str(duration_per_test)
        ]
        
        proc = subprocess.run(cmd, capture_output=True, text=True)
        out = proc.stdout
        
        # Parse result line: "Total FPS: XX.XX | FPS/Cam: X.XX | CPU: XX.X% | GPU: XX.X%"
        total_fps = 0.0
        fps_per_cam = 0.0
        cpu_pct = 0.0
        gpu_pct = 0.0

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

        meets_target = "✅ YES (>= 1.0 FPS)" if fps_per_cam >= 1.0 else "❌ NO (< 1.0 FPS)"
        results.append({
            "cameras": num_cams,
            "total_fps": total_fps,
            "fps_per_cam": fps_per_cam,
            "target_met": meets_target,
            "cpu_percent": cpu_pct,
            "gpu_percent": gpu_pct,
        })

        print(f"  Result: {num_cams} Cams -> {total_fps:.2f} Total FPS | {fps_per_cam:.2f} FPS/Cam | GPU: {gpu_pct:.1f}% | Meets 1.0 FPS Target: {meets_target}")
        time.sleep(3)  # Allow hardware decoders to fully release

    print(f"\n============================================================")
    print(f" 📊 FINAL 1.0 FPS CAPACITY SUMMARY TABLE")
    print(f"============================================================")
    print(f"| Camera Count | Total System FPS | FPS / Camera | Meets 1.0 FPS Target? | Avg CPU % | Avg GPU % |")
    print(f"|---|---|---|---|---|---|")
    
    max_cams_1fps = 0
    for r in results:
        print(f"| {r['cameras']} Cameras | **{r['total_fps']:.2f} FPS** | **{r['fps_per_cam']:.2f} FPS** | {r['target_met']} | {r['cpu_percent']:.1f}% | {r['gpu_percent']:.1f}% |")
        if r['fps_per_cam'] >= 1.0:
            max_cams_1fps = max(max_cams_1fps, r['cameras'])

    print(f"\n🏆 MAXIMUM CAMERA CAPACITY FOR AT LEAST 1.0 FPS PER CAMERA: **{max_cams_1fps} CAMERAS**\n")

if __name__ == "__main__":
    main()
