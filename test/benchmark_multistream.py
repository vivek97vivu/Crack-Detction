"""
benchmark_multistream.py — 60-second FPS benchmark for different pool_workers configs.

Usage:
    LD_PRELOAD=... conda run --no-capture-output -n crack \
        python benchmark_multistream.py [--config config/config.yaml] [--duration 60]

Tests pool_workers = [1, 2, 4, 6] against all enabled cameras.
Each run lasts --duration seconds. Results saved to benchmark_results.txt.
"""

import cv2
import os
import sys
import time
import logging
import threading
import warnings
import argparse
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from inference.pipeline import CrackDetectionPipeline
from inference.detector import DetectorInference
from inference.segmenter import SegmenterInference
from inference.gate import GateInference
from utils.config import load_config, resolve_path
from utils.capture import ThreadedVideoCapture
from utils.gstreamer import build_gstreamer_capture

logging.basicConfig(level=logging.WARNING)


def load_shared_models(config):
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
    shared_detector = DetectorInference(checkpoint_path=det_checkpoint, threshold=det_threshold)
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


class BenchmarkWorker(threading.Thread):
    def __init__(self, camera_cfg, config_path,
                 shared_gate, shared_detector, shared_segmenter,
                 inference_pool, stop_event):
        super().__init__(daemon=True)
        self.camera_cfg       = camera_cfg
        self.cam_id           = camera_cfg.get("id", "cam")
        self.config_path      = config_path
        self.shared_gate      = shared_gate
        self.shared_detector  = shared_detector
        self.shared_segmenter = shared_segmenter
        self.inference_pool   = inference_pool
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
                frame_id = f"{self.cam_id}_frame_{frame_count}"
                future = self.inference_pool.submit(
                    pipeline.process_frame, frame.copy(), frame_id
                )
                try:
                    future.result(timeout=5.0)
                    self.total_frames += 1
                except Exception:
                    pass
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


def run_one(config, config_path, pool_workers, cameras, duration_s):
    shared_gate, shared_detector, shared_segmenter = load_shared_models(config)
    pool  = ThreadPoolExecutor(max_workers=pool_workers, thread_name_prefix="bm")
    stop  = threading.Event()
    workers = [
        BenchmarkWorker(c, config_path, shared_gate, shared_detector,
                        shared_segmenter, pool, stop)
        for c in cameras
    ]
    for w in workers:
        w.start()
    time.sleep(duration_s)
    stop.set()
    for w in workers:
        w.join(timeout=5.0)
    pool.shutdown(wait=False)
    return workers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",   default=None)
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--workers",  default="1,2,4,6")
    args = ap.parse_args()

    config  = load_config(args.config)
    enabled = [c for c in config.get("cameras", []) if c.get("enabled", True)]
    pool_workers_list = [int(x) for x in args.workers.split(",")]
    duration = args.duration

    print(f"\nBenchmark: {len(enabled)} cameras | {duration}s per run | pools={pool_workers_list}")
    print("Model: rfdetr-seg-medium-fp16.engine (TRT FP16, Jetson AGX Orin)\n")

    summary_rows = []

    for pw in pool_workers_list:
        print(f"{'─'*60}")
        print(f"Running: pool_workers={pw}, cameras={len(enabled)}, duration={duration}s")
        workers = run_one(config, args.config, pw, enabled, duration)

        total_frames = sum(w.total_frames for w in workers)
        elapsed      = max(1.0, max(w.elapsed for w in workers))
        total_fps    = total_frames / elapsed
        per_cam_fps  = total_fps / max(1, len(workers))

        summary_rows.append((pw, len(enabled), total_frames, elapsed, total_fps, per_cam_fps))

        print(f"\n  {'Camera':<12} {'Frames':>8} {'Elapsed':>10} {'FPS':>8}")
        print(f"  {'─'*42}")
        for w in workers:
            print(f"  {w.cam_id:<12} {w.total_frames:>8} {w.elapsed:>10.1f} {w.avg_fps:>8.2f}")
        print(f"\n  --> Total FPS: {total_fps:.2f} | FPS/camera: {per_cam_fps:.2f}\n")

    # Final summary table
    div = "─" * 72
    print(f"\n{'='*72}")
    print(f"  FINAL SUMMARY  |  Cameras: {len(enabled)}  |  Duration: {duration}s")
    print(f"{'='*72}")
    print(f"  {'Pool Workers':>14} {'Cameras':>9} {'Total Frames':>14} {'Total FPS':>11} {'FPS/Camera':>11}")
    print(div)
    lines = []
    for (pw, cc, tf, el, tfps, pcfps) in summary_rows:
        row = f"  {pw:>14} {cc:>9} {tf:>14} {tfps:>11.2f} {pcfps:>11.2f}"
        print(row)
        lines.append(f"{pw:>4}  {cc:>7}  {tf:>12}  {el:>8.1f}  {tfps:>9.2f}  {pcfps:>9.2f}")
    print(div)

    out = "benchmark_results.txt"
    with open(out, "w") as f:
        f.write("Pool Workers | Cameras | Total Frames | Elapsed(s) | Total FPS | FPS/Camera\n")
        f.write("─" * 64 + "\n")
        for line in lines:
            f.write(line + "\n")
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
