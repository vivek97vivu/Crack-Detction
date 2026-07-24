
import os
import sys

# Auto-switch to 'crack' conda python & set LD_LIBRARY_PATH for C++ GLIBCXX compatibility
crack_python = "/home/algosium/miniforge3/envs/crack/bin/python"
crack_lib    = "/home/algosium/miniforge3/envs/crack/lib"
if os.path.exists(crack_python) and (sys.executable != crack_python or crack_lib not in os.environ.get("LD_LIBRARY_PATH", "")):
    os.environ["LD_LIBRARY_PATH"] = f"{crack_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.execv(crack_python, [crack_python] + sys.argv)

import cv2
import time
import queue
import logging
import threading
import warnings
import argparse
import psutil
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from inference.pipeline import CrackDetectionPipeline
from inference.detector import DetectorInference
from inference.segmenter import SegmenterInference
from inference.gate import GateInference
from inference.scheduler import InferenceScheduler
from utils.config import load_config, resolve_path
from utils.capture import ThreadedVideoCapture
from utils.gstreamer import build_gstreamer_capture

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("benchmark_scale")


def load_shared_models(config, num_engines=3):
    p_cfg = config.get("pipeline", {})
    g_cfg = config.get("gate", {})
    d_cfg = config.get("detector", {})
    s_cfg = config.get("segmenter", {})

    enable_gate  = p_cfg.get("enable_gate", False)
    g_checkpoint = resolve_path(g_cfg.get("checkpoint"))
    g_threshold  = g_cfg.get("threshold", 0.4)
    shared_gate  = GateInference(checkpoint_path=g_checkpoint, threshold=g_threshold) if enable_gate else None

    det_checkpoint  = resolve_path(d_cfg.get("checkpoint") or d_cfg.get("checkpoint_ema"))
    det_threshold   = d_cfg.get("threshold", 0.3)
    shared_detector = DetectorInference(checkpoint_path=det_checkpoint, threshold=det_threshold, num_engines=num_engines)
    if "target_classes" in d_cfg:
        shared_detector.target_classes = d_cfg["target_classes"]

    seg_checkpoint   = resolve_path(s_cfg.get("checkpoint"))
    fallback         = s_cfg.get("fallback_to_heuristic", True)
    shared_segmenter = SegmenterInference(checkpoint_path=seg_checkpoint, fallback_to_heuristic=fallback)

    return shared_gate, shared_detector, shared_segmenter


def open_camera(camera_cfg):
    source  = camera_cfg.get("source")
    use_gst = camera_cfg.get("use_gstreamer", False)

    if use_gst and isinstance(source, str) and source.startswith("rtsp://"):
        raw_cap = build_gstreamer_capture(camera_cfg)
        if raw_cap.isOpened():
            return ThreadedVideoCapture(raw_cap)

    if isinstance(source, str):
        source = os.path.expanduser(source)
    cap = cv2.VideoCapture(source)
    return ThreadedVideoCapture(cap) if cap.isOpened() else None


class BenchmarkCameraWorker(threading.Thread):
    def __init__(self, camera_cfg, config_path,
                 shared_gate, shared_detector, shared_segmenter,
                 scheduler, stop_event):
        super().__init__(daemon=True)
        self.camera_cfg       = camera_cfg
        self.cam_id           = camera_cfg.get("id", "cam")
        self.config_path      = config_path
        self.shared_gate      = shared_gate
        self.shared_detector  = shared_detector
        self.shared_segmenter = shared_segmenter
        self.scheduler        = scheduler
        self.stop_event       = stop_event
        self.total_frames     = 0
        self.start_time       = None
        self.end_time         = None

    def run(self):
        pipeline = CrackDetectionPipeline(
            config_path=self.config_path,
            shared_gate=self.shared_gate,
            shared_detector=self.shared_detector,
            shared_segmenter=self.shared_segmenter,
        )
        cap = open_camera(self.camera_cfg)
        if cap is None or not cap.isOpened():
            return

        frame_skip  = max(1, self.camera_cfg.get("frame_skip", 1))
        frame_count = 0
        self.start_time = time.time()

        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.05)
                    continue

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                # If gating is enabled, check gate classifier first (0.8 ms)
                detector_outputs = None
                if pipeline.enable_gate and pipeline.gate is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    gate_passed, _ = pipeline.gate.predict(frame_rgb)
                    if not gate_passed:
                        # Non-defect frame filtered out by gate — skip heavy TRT detector
                        self.total_frames += 1
                        continue

                # 1. Submit frame to scheduler for GPU batch inference if gate passed
                detector_outputs = self.scheduler.submit(frame)

                # 2. Run pipeline tracking, geometry and alerts
                frame_id = f"{self.cam_id}_frame_{frame_count}"
                pipeline.process_frame(frame, frame_id, detector_outputs=detector_outputs)
                self.total_frames += 1

        finally:
            self.end_time = time.time()
            cap.release()

    @property
    def avg_fps(self):
        elapsed = max(0.1, (self.end_time or time.time()) - (self.start_time or time.time()))
        return self.total_frames / elapsed

    @property
    def elapsed(self):
        return max(0.1, (self.end_time or time.time()) - (self.start_time or time.time()))


def get_gpu_usage():
    try:
        with open("/sys/devices/platform/bus@0/17000000.gpu/load", "r") as f:
            val = int(f.read().strip())
            return val / 10.0  # Permille to percent
    except Exception:
        return 0.0


class HardwareMonitor(threading.Thread):
    def __init__(self, interval=1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_event = threading.Event()
        self.cpu_readings = []
        self.gpu_readings = []

    def run(self):
        # Prime psutil
        psutil.cpu_percent(interval=None)
        while not self.stop_event.is_set():
            time.sleep(self.interval)
            self.cpu_readings.append(psutil.cpu_percent(interval=None))
            self.gpu_readings.append(get_gpu_usage())

    def stop(self):
        self.stop_event.set()

    @property
    def avg_cpu(self):
        return sum(self.cpu_readings) / len(self.cpu_readings) if self.cpu_readings else 0.0

    @property
    def avg_gpu(self):
        return sum(self.gpu_readings) / len(self.gpu_readings) if self.gpu_readings else 0.0


def run_one_config(config, config_path, pool_size, cameras, duration_s, num_engines=3, scheduler_workers=3):
    shared_gate, shared_detector, shared_segmenter = load_shared_models(config, num_engines=num_engines)
    
    # Initialize scheduler with batch_size = pool_size and num_workers = scheduler_workers
    scheduler = InferenceScheduler(shared_detector, batch_size=pool_size, timeout_ms=5.0, num_workers=scheduler_workers)
    scheduler.start()
    
    stop_event = threading.Event()
    workers = [
        BenchmarkCameraWorker(c, config_path, shared_gate, shared_detector,
                             shared_segmenter, scheduler, stop_event)
        for c in cameras
    ]
    
    monitor = HardwareMonitor(interval=1.0)
    monitor.start()

    for w in workers:
        w.start()

    time.sleep(duration_s)
    
    stop_event.set()
    for w in workers:
        w.join(timeout=5.0)
    
    scheduler.stop()
    monitor.stop()
    monitor.join(timeout=2.0)

    return workers, monitor.avg_cpu, monitor.avg_gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",            default=None)
    ap.add_argument("--duration",          type=int, default=30)
    ap.add_argument("--cameras",           type=int, default=30, help="Target camera streams (replicated in memory)")
    ap.add_argument("--workers",           default="16", help="Comma-separated pool/batch sizes to test")
    ap.add_argument("--engines",           type=int, default=3, help="Number of parallel TRT engines in GPU RAM")
    ap.add_argument("--scheduler-workers", type=int, default=3, help="Number of parallel scheduler worker threads")
    args = ap.parse_args()

    config = load_config(args.config)
    cameras_cfg = config.get("cameras", [])
    enabled_cameras = [c for c in cameras_cfg if c.get("enabled", True)]

    if not enabled_cameras:
        print("[Error] No enabled cameras found. Check config.yaml.")
        return

    # Replicate cameras in memory to hit target scaling count
    target_count = args.cameras
    cameras = []
    for i in range(target_count):
        orig = enabled_cameras[i % len(enabled_cameras)]
        copy_cfg = orig.copy()
        copy_cfg["id"] = f"{orig['id']}_{i}"
        copy_cfg["name"] = f"{orig['name']}_{i}"
        cameras.append(copy_cfg)

    pool_sizes = [int(x) for x in args.workers.split(",")]
    duration = args.duration

    print(f"\n============================================================")
    print(f"  SCALING BENCHMARK: {len(cameras)} Replicated Camera Streams")
    print(f"  Duration: {duration}s per run  |  TRT Parallel Engine Pool: {args.engines}x")
    print(f"============================================================\n")

    summary_rows = []

    for pw in pool_sizes:
        print(f"Running batch size = {pw} with {args.engines}x TRT Engines...")
        workers, avg_cpu, avg_gpu = run_one_config(
            config, args.config, pw, cameras, duration,
            num_engines=args.engines, scheduler_workers=args.scheduler_workers
        )

        total_frames = sum(w.total_frames for w in workers)
        elapsed      = max(1.0, max(w.elapsed for w in workers))
        total_fps    = total_frames / elapsed
        per_cam_fps  = total_fps / len(workers)

        summary_rows.append((pw, len(workers), total_frames, elapsed, total_fps, per_cam_fps, avg_cpu, avg_gpu))
        print(f"  Total FPS: {total_fps:.2f} | FPS/Cam: {per_cam_fps:.2f} | CPU: {avg_cpu:.1f}% | GPU: {avg_gpu:.1f}%\n")

    # Output Markdown summary table
    div = "|---|---|---|---|---|---|---|---|---|"
    print(f"\n### Scaling Benchmark Summary (TRT Dynamic Batch FP16)")
    print(f"| Worker Pool (Batch) | Camera Count | Total Frames | Elapsed (s) | Total FPS | FPS / Camera | Avg CPU % | Avg GPU % |")
    print(f"|---|---|---|---|---|---|---|---|")
    
    file_lines = [
        "Worker Pool (Batch) | Camera Count | Total Frames | Elapsed (s) | Total FPS | FPS / Camera | Avg CPU % | Avg GPU %",
        "-------------------------------------------------------------------------------------------------------"
    ]
    
    for (pw, cc, tf, el, tfps, pcfps, cpu, gpu) in summary_rows:
        row = f"| {pw} | {cc} | {tf} | {el:.1} | {tfps:.2f} | {pcfps:.2f} | {cpu:.1f}% | {gpu:.1f}% |"
        print(row)
        file_lines.append(f"{pw:>19} | {cc:>12} | {tf:>12} | {el:>11.1f} | {tfps:>9.2f} | {pcfps:>12.2f} | {cpu:>9.1f}% | {gpu:>9.1f}%")

    out = "benchmark_scale_results.txt"
    with open(out, "w") as f:
        f.write("\n".join(file_lines) + "\n")
    print(f"\nResults successfully saved to {out}\n")


if __name__ == "__main__":
    main()
