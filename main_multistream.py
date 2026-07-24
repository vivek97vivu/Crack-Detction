

import cv2
import numpy as np
import os
import sys
import json
import time
import logging
import threading
import warnings
import argparse
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from inference.pipeline import CrackDetectionPipeline, Detection, detection_to_dict
from inference.detector import DetectorInference
from inference.segmenter import SegmenterInference
from inference.gate import GateInference
from utils.config import load_config, resolve_path
from utils.capture import ThreadedVideoCapture
from utils.gstreamer import build_gstreamer_capture

# Enable gstreamer decoder selection logging so we can verify HW decoder is used
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("multistream")


# ─────────────────────────────────────────────────────────────────────────────
# Shared model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_shared_models(config: dict):
    """Load gate, detector, segmenter once — shared across all camera threads."""
    p_cfg  = config.get("pipeline", {})
    g_cfg  = config.get("gate", {})
    d_cfg  = config.get("detector", {})
    s_cfg  = config.get("segmenter", {})

    # Gate
    enable_gate  = p_cfg.get("enable_gate", True)
    g_checkpoint = resolve_path(g_cfg.get("checkpoint", None))
    g_threshold  = g_cfg.get("threshold", 0.4)
    shared_gate  = GateInference(checkpoint_path=g_checkpoint, threshold=g_threshold) if enable_gate else None

    # Detector
    det_checkpoint  = resolve_path(d_cfg.get("checkpoint") or d_cfg.get("checkpoint_ema"))
    det_threshold   = d_cfg.get("threshold", 0.3)
    shared_detector = DetectorInference(checkpoint_path=det_checkpoint, threshold=det_threshold)
    if "target_classes" in d_cfg:
        shared_detector.target_classes = d_cfg["target_classes"]

    # Segmenter
    seg_checkpoint   = resolve_path(s_cfg.get("checkpoint", None))
    fallback         = s_cfg.get("fallback_to_heuristic", True)
    shared_segmenter = SegmenterInference(checkpoint_path=seg_checkpoint, fallback_to_heuristic=fallback)

    logger.info("Shared models loaded — gate=%s detector=%s segmenter=%s",
                type(shared_gate).__name__,
                type(shared_detector).__name__,
                type(shared_segmenter).__name__)
    return shared_gate, shared_detector, shared_segmenter


# ─────────────────────────────────────────────────────────────────────────────
# Per-camera capture helper
# ─────────────────────────────────────────────────────────────────────────────

def open_camera(camera_cfg: dict):
    """Open a camera using HW-accelerated GStreamer (with threaded capture) or FFMPEG."""
    source  = camera_cfg.get("source")
    use_gst = camera_cfg.get("use_gstreamer", False)
    cam_id  = camera_cfg.get("id", "camera")

    if use_gst and isinstance(source, str) and source.startswith("rtsp://"):
        logger.info("[%s] Building GStreamer HW-decoder pipeline...", cam_id)
        raw_cap = build_gstreamer_capture(camera_cfg)
        if raw_cap.isOpened():
            logger.info("[%s] GStreamer OK — wrapping in ThreadedVideoCapture", cam_id)
            return ThreadedVideoCapture(raw_cap)
        logger.warning("[%s] GStreamer failed — falling back to FFMPEG", cam_id)

    # Local file / webcam fallback
    if isinstance(source, str):
        source = os.path.expanduser(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("[%s] Failed to open source: %s", cam_id, source)
        return None
    return ThreadedVideoCapture(cap)


# ─────────────────────────────────────────────────────────────────────────────
# Per-camera worker thread
# ─────────────────────────────────────────────────────────────────────────────

class CameraWorker(threading.Thread):
    def __init__(self, camera_cfg, config_path,
                 shared_gate, shared_detector, shared_segmenter,
                 latest_frames, stop_event, inference_pool):
        super().__init__(daemon=True)
        self.camera_cfg       = camera_cfg
        self.cam_id           = camera_cfg.get("id", "cam")
        self.cam_name         = camera_cfg.get("name", self.cam_id)
        self.config_path      = config_path
        self.shared_gate      = shared_gate
        self.shared_detector  = shared_detector
        self.shared_segmenter = shared_segmenter
        self.latest_frames    = latest_frames   # shared dict: cam_id -> annotated_frame
        self.stop_event       = stop_event
        self.inference_pool   = inference_pool  # ThreadPoolExecutor shared across all cameras
        self.fps              = 0.0
        self.total_frames     = 0              # for 60s benchmark summary
        self.start_time       = None

    def run(self):
        cam_id = self.cam_id

        # Build a per-camera pipeline that reuses shared models
        pipeline = CrackDetectionPipeline(
            config_path=self.config_path,
            shared_gate=self.shared_gate,
            shared_detector=self.shared_detector,
            shared_segmenter=self.shared_segmenter,
        )

        cap = open_camera(self.camera_cfg)
        if cap is None or not cap.isOpened():
            logger.error("[%s] Could not open camera — thread exiting.", cam_id)
            return

        frame_skip  = max(1, self.camera_cfg.get("frame_skip", 1))
        frame_count = 0
        fps_count   = 0
        fps_t0      = time.time()
        self.start_time = fps_t0

        logger.info("[%s] Stream started (frame_skip=%d)", cam_id, frame_skip)

        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning("[%s] Failed to read frame — retrying...", cam_id)
                    time.sleep(0.1)
                    continue

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                frame_id = f"{cam_id}_frame_{frame_count}"

                # Submit inference to shared pool — serialises GPU calls cleanly
                future = self.inference_pool.submit(
                    pipeline.process_frame, frame.copy(), frame_id
                )
                try:
                    annotated, _ = future.result(timeout=5.0)
                except Exception as exc:
                    logger.warning("[%s] Inference error: %s", cam_id, exc)
                    continue

                # FPS accounting
                self.total_frames += 1
                fps_count += 1
                elapsed = time.time() - fps_t0
                if elapsed >= 1.0:
                    self.fps  = fps_count / elapsed
                    fps_count = 0
                    fps_t0    = time.time()
                    logger.info("[%s] Inference %.1f FPS", cam_id, self.fps)

                # Overlay FPS
                cv2.putText(
                    annotated,
                    f"{self.cam_name} | INF {self.fps:.1f} FPS",
                    (10, annotated.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2, cv2.LINE_AA,
                )

                self.latest_frames[cam_id] = annotated

        finally:
            cap.release()
            logger.info("[%s] Stream stopped.", cam_id)


# ─────────────────────────────────────────────────────────────────────────────
# Tiled display helper
# ─────────────────────────────────────────────────────────────────────────────

def tile_frames(frames, target_w=1280):
    """Tile a list of frames side-by-side, scaled to target_w total width."""
    if not frames:
        return np.zeros((360, target_w, 3), dtype=np.uint8)
    n      = len(frames)
    cell_w = target_w // n
    resized = []
    for f in frames:
        h, w   = f.shape[:2]
        cell_h = int(h * cell_w / w)
        resized.append(cv2.resize(f, (cell_w, cell_h)))
    max_h  = max(r.shape[0] for r in resized)
    padded = []
    for r in resized:
        pad = np.zeros((max_h, cell_w, 3), dtype=np.uint8)
        pad[: r.shape[0], :] = r
        padded.append(pad)
    return np.hstack(padded)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Multi-camera crack detection on Jetson")
    ap.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = ap.parse_args()

    config      = load_config(args.config)
    cameras_cfg = config.get("cameras", [])
    enabled     = [c for c in cameras_cfg if c.get("enabled", True)]
    p_cfg       = config.get("pipeline", {})
    pool_workers = max(1, p_cfg.get("pool_workers", 4))

    if not enabled:
        print("[Error] No enabled cameras found in config. Exiting.")
        return

    print(f"\nInitializing shared models for {len(enabled)} camera(s)...")
    shared_gate, shared_detector, shared_segmenter = load_shared_models(config)

    # Shared inference pool — throttles concurrent GPU submissions
    inference_pool = ThreadPoolExecutor(
        max_workers=pool_workers,
        thread_name_prefix="inf_worker",
    )
    logger.info("Inference pool created — pool_workers=%d cameras=%d",
                pool_workers, len(enabled))

    latest_frames = {}
    stop_event    = threading.Event()
    workers       = []

    for cam_cfg in enabled:
        w = CameraWorker(
            camera_cfg=cam_cfg,
            config_path=args.config,
            shared_gate=shared_gate,
            shared_detector=shared_detector,
            shared_segmenter=shared_segmenter,
            latest_frames=latest_frames,
            stop_event=stop_event,
            inference_pool=inference_pool,
        )
        workers.append(w)

    print(f"Starting {len(workers)} camera thread(s)... Press 'q' to quit.\n")
    for w in workers:
        w.start()

    cam_ids     = [c.get("id") for c in enabled]
    has_display = True

    try:
        while not stop_event.is_set():
            frames = [latest_frames[cid] for cid in cam_ids if cid in latest_frames]
            if frames:
                if has_display:
                    try:
                        tiled = tile_frames(frames, target_w=1280 * min(len(frames), 2))
                        cv2.imshow("Crack Detection — Multi-Stream", tiled)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break
                    except Exception:
                        has_display = False   # switch to headless mode

            if not has_display:
                time.sleep(0.5)
                fps_strs = [f"{w.cam_name}: {w.fps:.1f} FPS" for w in workers]
                print("[Multistream] " + " | ".join(fps_strs))

            time.sleep(0.001)

    finally:
        print("\nShutting down...")
        stop_event.set()
        for w in workers:
            w.join(timeout=3.0)
        inference_pool.shutdown(wait=False)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        # ── 60-second benchmark summary ───────────────────────────────────
        now = time.time()
        print("\n" + "─" * 68)
        print(f"  BENCHMARK SUMMARY  |  pool_workers={pool_workers}  |  cameras={len(workers)}")
        print("─" * 68)
        print(f"  {'Camera':<10} {'Frames':>8} {'Elapsed(s)':>12} {'Avg FPS':>10}")
        print("─" * 68)
        total_f = 0
        for w in workers:
            elapsed = max(1.0, (now - w.start_time) if w.start_time else 1.0)
            avg_fps = w.total_frames / elapsed
            total_f += w.total_frames
            print(f"  {w.cam_id:<10} {w.total_frames:>8} {elapsed:>12.1f} {avg_fps:>10.2f}")
        print("─" * 68)
        total_elapsed = max(1.0, (now - min(
            (w.start_time for w in workers if w.start_time), default=now
        )))
        print(f"  {'TOTAL':<10} {total_f:>8} {total_elapsed:>12.1f} {total_f/total_elapsed:>10.2f}")
        print("─" * 68)
        print("All streams stopped.")



if __name__ == "__main__":
    main()
