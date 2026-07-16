import sys
import os
import time
import threading
import cv2
import torch
import psutil
import numpy as np

# Add the project directories to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from utils.config import load_config
from main import get_video_capture
from inference.pipeline import CrackDetectionPipeline

gpu_lock = threading.Lock()

# Statistics and frame buffer dictionary
stats = {
    "frames_processed": 0,
    "lock": threading.Lock(),
    "cam_fps": {},
    "latest_frames": {}
}

class CameraThread(threading.Thread):
    def __init__(self, camera_cfg, shared_pipeline_args):
        super().__init__()
        self.camera_cfg = camera_cfg
        self.cam_id = camera_cfg.get("id")
        self.cam_name = camera_cfg.get("name", self.cam_id)
        self.stopped = False
        self.daemon = True
        
        # Instantiate a camera-specific pipeline with shared models to isolate tracking state
        self.pipeline = CrackDetectionPipeline(**shared_pipeline_args)
        
    def run(self):
        print(f"[Thread-{self.cam_id}] Connecting to source...")
        cap = get_video_capture(self.camera_cfg)
        if not cap.isOpened():
            print(f"[Thread-{self.cam_id}] Error: Could not open stream.")
            return
            
        frame_skip = self.camera_cfg.get("frame_skip", 1)
        if frame_skip < 1:
            frame_skip = 1
            
        fps = self.camera_cfg.get("playback_fps", 25)
        if fps <= 0:
            fps = 25
        # Target interval between frame processing (e.g. 1.0 second for 25 FPS with frame_skip=25)
        sleep_time = frame_skip / fps
        print(f"[Thread-{self.cam_id}] Connected! Running. Target interval: {sleep_time:.2f}s")
        
        frame_count = 0
        last_time = time.time()
        local_processed = 0
        
        try:
            while not self.stopped:
                # Blocks until a new frame is fetched by GStreamer/FFMPEG thread
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.1)
                    continue
                    
                frame_count += 1
                frame_id = f"{self.cam_id}_frame_{frame_count}"
                
                # Run crack detection pipeline with global GPU lock to serialize CUDA calls
                with gpu_lock:
                    annotated, meta = self.pipeline.process_frame(frame, frame_id=frame_id)
                
                local_processed += 1
                with stats["lock"]:
                    stats["frames_processed"] += 1
                    stats["latest_frames"][self.cam_id] = annotated
                    
                # Calculate local FPS occasionally
                now = time.time()
                if now - last_time >= 5.0:
                    fps_val = local_processed / (now - last_time)
                    with stats["lock"]:
                        stats["cam_fps"][self.cam_id] = fps_val
                    local_processed = 0
                    last_time = now
                    
                # Sleep to regulate camera thread loop and free up CPU/GPU scheduling time
                time.sleep(sleep_time)
                
        except Exception as e:
            print(f"[Thread-{self.cam_id}] Error in run loop: {e}")
        finally:
            cap.release()
            print(f"[Thread-{self.cam_id}] Stream closed.")
            
    def stop(self):
        self.stopped = True

def print_resource_usage():
    cpu_percent = psutil.cpu_percent(interval=None)
    memory = psutil.virtual_memory()
    
    gpu_memory_used = 0
    gpu_memory_total = 0
    gpu_name = "N/A"
    gpu_load_str = "N/A"
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_used = torch.cuda.memory_allocated(0) / (1024 * 1024)  # MB
        gpu_memory_reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)  # MB
        
    print(f"\n================ RESOURCE USAGE ================")
    print(f"CPU Load (Overall): {cpu_percent}%")
    print(f"RAM Usage: {memory.percent}% ({memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB)")
    if torch.cuda.is_available():
        print(f"GPU: {gpu_name}")
        print(f"GPU VRAM Active: {gpu_memory_used:.1f} MB (Reserved: {gpu_memory_reserved:.1f} MB)")
    print(f"================================================\n")

def build_grid(frames, grid_size, cell_size):
    rows, cols = grid_size
    w, h = cell_size
    
    # Create empty black grid
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    
    cam_ids = sorted(list(frames.keys()))
    for idx, cam_id in enumerate(cam_ids):
        if idx >= rows * cols:
            break
        r = idx // cols
        c = idx % cols
        
        frame = frames[cam_id]
        if frame is not None:
            # Resize frame to fit cell in the grid
            resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Draw camera ID text on top
            cv2.putText(resized, cam_id, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            grid[r*h:(r+1)*h, c*w:(c+1)*w] = resized
            
    return grid

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Stream Crack Detection")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml file")
    parser.add_argument("--duration", type=int, default=600, help="Duration to run the benchmark in seconds")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI window)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    cameras = config.get("cameras", [])
    enabled_cameras = [cam for cam in cameras if cam.get("enabled", True)]
    
    if not enabled_cameras:
        print("Error: No enabled cameras found in configuration.")
        return
        
    print(f"Loaded {len(enabled_cameras)} enabled cameras from config.")
    
    # Load model weights ONCE to share them across threads
    print("Loading models once into GPU VRAM...")
    base_pipeline = CrackDetectionPipeline(config_path=args.config)
    
    shared_args = {
        "config_path": args.config,
        "shared_gate": base_pipeline.gate,
        "shared_detector": base_pipeline.detector,
        "shared_segmenter": base_pipeline.segmenter
    }
    
    # Calculate grid layout if displaying GUI
    if not args.headless:
        n_cams = len(enabled_cameras)
        cols = int(np.ceil(np.sqrt(n_cams)))
        rows = int(np.ceil(n_cams / cols))
        grid_size = (rows, cols)
        
        # Scale cell sizes dynamically so the grid fits on the screen
        if n_cams <= 4:
            cell_size = (640, 480)
        elif n_cams <= 9:
            cell_size = (480, 360)
        elif n_cams <= 16:
            cell_size = (320, 240)
        else:
            cell_size = (240, 180) # 30 cameras fits on a 1440x900 grid
            
        print(f"Display Mode: Grid window enabled ({rows} rows x {cols} cols, cell: {cell_size[0]}x{cell_size[1]})")
    
    # Create and start threads
    threads = []
    for cam_cfg in enabled_cameras:
        thread = CameraThread(cam_cfg, shared_args)
        threads.append(thread)
        
    print(f"Starting {len(threads)} camera stream threads...")
    for t in threads:
        t.start()
        time.sleep(0.05) # brief stagger
        
    # Monitor and GUI loop
    start_time = time.time()
    last_print = start_time
    
    try:
        while time.time() - start_time < args.duration:
            if args.headless:
                time.sleep(1.0)
            else:
                # Retrieve latest frames and display grid
                with stats["lock"]:
                    current_frames = {k: v.copy() for k, v in stats["latest_frames"].items() if v is not None}
                
                if current_frames:
                    grid = build_grid(current_frames, grid_size, cell_size)
                    cv2.imshow("Crack Detection Multi-Stream Grid (Press 'q' to Quit)", grid)
                    
                # OpenCV GUI waitKey handles rendering and keyboard events
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    print("\nGUI Window closed by user.")
                    break
            
            # Print console stats every 5 seconds
            now = time.time()
            if now - last_print >= 5.0:
                with stats["lock"]:
                    total_processed = stats["frames_processed"]
                    cam_speeds = list(stats["cam_fps"].items())
                
                elapsed = now - start_time
                overall_fps = total_processed / elapsed
                
                print(f"\n--- Progress: {elapsed:.1f}s / {args.duration}s ---")
                print(f"Total Frames Processed: {total_processed}")
                print(f"Overall Processing FPS: {overall_fps:.2f} frames/sec")
                print_resource_usage()
                last_print = now
                
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    finally:
        print("Stopping camera threads...")
        for t in threads:
            t.stop()
        for t in threads:
            t.join(timeout=1.0)
        
        if not args.headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("All threads stopped. Benchmark finished.")

if __name__ == "__main__":
    main()
