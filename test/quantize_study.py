"""
quantize_study.py — Quantization Balance Study for RF-DETR Crack Detection

This script quantizes the RF-DETR model using three calibration methods:
  1. MinMax (Default, simple scaling)
  2. Percentile (Clips outliers, default at 99.99%)
  3. Entropy (KL-divergence minimization, balances clipping vs rounding noise)

To prevent Jetson memory exhaustion (OOM), each calibration method is executed
in a separate Python subprocess, guaranteeing 100% of memory is reclaimed by the OS.
"""

from __future__ import annotations

import os
import sys
import time
import glob
import argparse
import subprocess
import json
import numpy as np
import cv2

# Locate ORT from system path if not present
_TRT_SYSTEM_PATH = "/usr/lib/python3.10/dist-packages"
if _TRT_SYSTEM_PATH not in sys.path:
    sys.path.insert(0, _TRT_SYSTEM_PATH)

import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    quant_pre_process,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
    CalibrationMethod,
)

# ── Preprocessing ────────────────────────────────────────────────────────────
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 3 and bgr.shape[2] == 4:
        bgr = bgr[:, :, :3]
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res  = cv2.resize(rgb, (576, 576), interpolation=cv2.INTER_LINEAR)
    flt  = res.astype(np.float32) / 255.0
    norm = (flt - _MEAN) / _STD
    return np.ascontiguousarray(np.transpose(norm, (2, 0, 1))[np.newaxis, ...])


class SimpleCalibrationReader(CalibrationDataReader):
    def __init__(self, image_paths, input_name):
        super().__init__()
        self.image_paths = image_paths
        self.input_name = input_name
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.image_paths):
            return None
        p = self.image_paths[self.idx]
        self.idx += 1
        bgr = cv2.imread(p)
        if bgr is None:
            return self.get_next()
        return {self.input_name: preprocess(bgr)}

    def rewind(self):
        self.idx = 0


def calculate_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten()
    b_flat = b.flatten()
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom == 0:
        return 1.0 if np.allclose(a_flat, b_flat) else 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def run_single_method(name: str, calib_method: CalibrationMethod, extra_opts: dict, args):
    """Runs a single method in the current process. Called via subprocess."""
    image_paths = sorted(glob.glob(os.path.join(args.calib, "*.jpg")) +
                         glob.glob(os.path.join(args.calib, "*.png")))
    if not image_paths:
        sys.exit(f"Error: No images found in {args.calib}")

    # Use 15 images for calibration and evaluation to keep memory and runtime low
    eval_images = image_paths[:15]
    calib_images = image_paths[:15]

    preprocessed_onnx = "model/rfdetr-seg-medium-preprocessed.onnx"
    
    # Get input name
    import onnx
    model = onnx.load(preprocessed_onnx)
    input_name = model.graph.input[0].name
    del model

    out_onnx = f"model/rfdetr-int8-{name.lower()}.onnx"
    print(f"\n[Subprocess] Starting quantization for {name}...")
    
    reader = SimpleCalibrationReader(calib_images, input_name)
    
    t0 = time.time()
    quantize_static(
        model_input=preprocessed_onnx,
        model_output=out_onnx,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        calibrate_method=calib_method,
        extra_options=extra_opts,
    )
    quant_time = time.time() - t0
    print(f"[Subprocess] Quantization for {name} completed in {quant_time:.1f}s")

    # Evaluate accuracy retention
    print(f"[Subprocess] Evaluating {name} accuracy retention on snapshots...")
    providers = ["CPUExecutionProvider"]
    baseline_sess = ort.InferenceSession(preprocessed_onnx, providers=providers)
    quant_sess = ort.InferenceSession(out_onnx, providers=providers)

    cos_sims_dets = []
    cos_sims_masks = []
    mae_boxes = []

    for p in eval_images:
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        inp = preprocess(bgr)
        
        # Baseline predictions
        base_out = baseline_sess.run(None, {input_name: inp})
        # Quantized predictions
        quant_out = quant_sess.run(None, {input_name: inp})

        dets_base = base_out[0]
        dets_quant = quant_out[0]
        
        cos_sims_dets.append(calculate_cosine_similarity(dets_base, dets_quant))
        mae_boxes.append(float(np.mean(np.abs(dets_base - dets_quant))))

        if len(base_out) > 2 and len(quant_out) > 2:
            masks_base = base_out[2]
            masks_quant = quant_out[2]
            cos_sims_masks.append(calculate_cosine_similarity(masks_base, masks_quant))

    mean_cos_dets = np.mean(cos_sims_dets) * 100
    mean_cos_masks = np.mean(cos_sims_masks) * 100 if cos_sims_masks else 100.0
    mean_mae_box = np.mean(mae_boxes)

    # Save results to a temporary JSON file to be read by coordinator
    results = {
        "Method": name,
        "Box Cosine Sim (%)": mean_cos_dets,
        "Mask Cosine Sim (%)": mean_cos_masks,
        "Box MAE": mean_mae_box,
        "File Size (MB)": os.path.getsize(out_onnx) / 1e6,
        "Time (s)": quant_time
    }
    
    with open(f"model/study_result_{name.lower()}.json", "w") as f:
        json.dump(results, f)
    print(f"[Subprocess] Done with {name}. Saved temporary results.")


def run_coordinator(args):
    print("=" * 70)
    print("  RF-DETR INT8 Quantization Balance Study (TRT 10.3)")
    print("=" * 70)
    print(f"  ONNX FP32 model: {args.onnx}")
    print(f"  Snapshots dir  : {args.calib}")
    print("=" * 70)

    # Pre-process model (crucial for shape inference and quantization accuracy)
    preprocessed_onnx = "model/rfdetr-seg-medium-preprocessed.onnx"
    if not os.path.exists(preprocessed_onnx):
        print(f"Running model pre-processing (shape inference & fusion)...")
        quant_pre_process(args.onnx, preprocessed_onnx, skip_symbolic_shape=True)
        print(f"Pre-processed model saved -> {preprocessed_onnx}")
    else:
        print(f"Using existing pre-processed model: {preprocessed_onnx}")

    methods = ["MinMax", "ConvOnly"]
    study_results = []

    for method in methods:
        print(f"\n>>> Launching subprocess for {method} calibration...")
        cmd = [
            sys.executable,
            __file__,
            "--onnx", args.onnx,
            "--calib", args.calib,
            "--run-method", method
        ]
        # Inherit LD_PRELOAD environment variables
        env = os.environ.copy()
        
        t0 = time.time()
        # Execute the subprocess
        proc = subprocess.run(cmd, env=env)
        elapsed = time.time() - t0
        
        if proc.returncode != 0:
            print(f"[Coordinator] Error: Subprocess for {method} failed (exit code {proc.returncode})")
            continue
            
        # Read the generated result file
        res_file = f"model/study_result_{method.lower()}.json"
        if os.path.exists(res_file):
            with open(res_file, "r") as f:
                res_data = json.load(f)
                study_results.append(res_data)
            os.remove(res_file) # Clean up
        else:
            print(f"[Coordinator] Warning: Could not find results file for {method}")

    # Print final Markdown comparison
    print("\n" + "=" * 75)
    print("  SUMMARY STUDY COMPARISON TABLE")
    print("=" * 75)
    print("| Calibration Method | Box Cosine Sim (%) | Mask Cosine Sim (%) | Box MAE | Size (MB) |")
    print("|---|---|---|---|---|")
    for res in study_results:
        print(f"| {res['Method']:18s} | {res['Box Cosine Sim (%)']:.2f}% | {res['Mask Cosine Sim (%)']:.2f}% | {res['Box MAE']:.6f} | {res['File Size (MB)']:.1f} MB |")
    print("=" * 75)
    
    # Determine the best balanced method
    if study_results:
        # We define "best balance" as the one with high box similarity and high mask similarity.
        # Entropy is usually the best, followed by Percentile.
        best = max(study_results, key=lambda x: (x["Box Cosine Sim (%)"] + x["Mask Cosine Sim (%)"]) / 2)
        print(f"\n[Balance Pick]: **{best['Method']}** has the best balance of accuracy and quantization scale.")
        print(f"  Recommended engine build command:")
        print(f"  python build_int8_engine.py --quant model/rfdetr-int8-{best['Method'].lower()}.onnx")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Study balance of INT8 calibration methods")
    ap.add_argument("--onnx", default="model/rfdetr-seg-medium.onnx")
    ap.add_argument("--calib", default="alerts/snapshot")
    ap.add_argument("--run-method", default=None, help="Internal use: run a single method")
    args = ap.parse_args()

    if args.run_method is not None:
        # Run single method in subprocess
        methods = {
            "MinMax": (CalibrationMethod.MinMax, {}),
            "ConvOnly": (CalibrationMethod.MinMax, {"op_types_to_quantize": ["Conv"]}),
            "Percentile_9999": (CalibrationMethod.Percentile, {"CalibPercentile": 99.99}),
            "Entropy": (CalibrationMethod.Entropy, {}),
        }
        if args.run_method in methods:
            method, opts = methods[args.run_method]
            run_single_method(args.run_method, method, opts, args)
        else:
            sys.exit(f"Unknown method {args.run_method}")
    else:
        # Run coordinator
        run_coordinator(args)
