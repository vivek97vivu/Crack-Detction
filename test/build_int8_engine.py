"""
build_int8_engine.py — Build a TensorRT INT8 engine for RF-DETR crack detection.

This script uses the recommended TensorRT 10.3 + ONNX static quantization workflow:
  1. Pre-processes the ONNX model (node fusion & shape inference)
  2. Excludes the entire transformer encoder and decoder blocks (which contain
     unsupported dynamic-shape attention layers) from quantization. This keeps
     them in FP16/FP32 to prevent Myelin compilation errors and retain accuracy.
  3. Excludes all Add, LayerNorm, and Softmax layers.
  4. Performs graph surgery to restore any remaining quantized biases back to
     original FP32 parameters, avoiding UINT8 zero-point parser failures.
  5. Compiles the quantized ONNX model to a TensorRT engine via trtexec.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import onnx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("build_int8")

# ── Preprocessing (identical to detector.py._predict_trt) ────────────────────
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(bgr: np.ndarray, h: int = 576, w: int = 576) -> np.ndarray:
    if bgr.ndim == 3 and bgr.shape[2] == 4:
        bgr = bgr[:, :, :3]
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res  = cv2.resize(rgb, (576, 576), interpolation=cv2.INTER_LINEAR)
    flt  = res.astype(np.float32) / 255.0
    norm = (flt - _MEAN) / _STD
    return np.ascontiguousarray(np.transpose(norm, (2, 0, 1))[np.newaxis, ...])


class SimpleCalibrationReader:
    def __init__(self, image_paths, input_name):
        from onnxruntime.quantization import CalibrationDataReader
        self.__class__ = type(
            "SimpleCalibrationReader",
            (CalibrationDataReader,),
            dict(self.__class__.__dict__),
        )
        CalibrationDataReader.__init__(self)
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


def build_pipeline(args):
    # 1. Pre-process model
    preprocessed_onnx = "model/rfdetr-seg-medium-preprocessed.onnx"
    if not os.path.exists(preprocessed_onnx):
        print(f"\n[Step 1/4] Running model pre-processing (node fusion & shape inference)...")
        from onnxruntime.quantization import quant_pre_process
        quant_pre_process(args.onnx, preprocessed_onnx, skip_symbolic_shape=True)
        print(f"  Pre-processed model saved -> {preprocessed_onnx}")
    else:
        print(f"\n[Step 1/4] Using existing pre-processed model: {preprocessed_onnx}")

    # 2. Identify nodes to exclude (biases, additions, transformer attention layers)
    print(f"  Loading model to identify sensitive and transformer nodes...")
    model = onnx.load(preprocessed_onnx)
    input_name = model.graph.input[0].name
    
    # Exclude all nodes belonging to:
    #   - Transformer encoder: '/backbone/backbone.0/encoder/' (contains 'encoder')
    #   - Transformer decoder: '/transformer/' (contains 'transformer')
    #   - Sensitive ops: Add, LayerNorm, Softmax, ReduceMean
    nodes_to_exclude = []
    excluded_types = ("Add", "BiasAdd", "LayerNormalization", "ReduceMean", "Softmax")
    for node in model.graph.node:
        name_lower = node.name.lower()
        if any(x in name_lower for x in ("encoder", "transformer")) or node.op_type in excluded_types:
            nodes_to_exclude.append(node.name)
            
    print(f"  Total graph nodes: {len(model.graph.node)}")
    print(f"  Excluding {len(nodes_to_exclude)} nodes (biases, transformer, and non-convolutional ops).")
    del model

    # 3. Collect calibration images
    image_paths = sorted(glob.glob(os.path.join(args.calib, "*.jpg")) +
                         glob.glob(os.path.join(args.calib, "*.png")))
    if not image_paths:
        sys.exit(f"Error: No images found in {args.calib}")
    
    # Use 15 representative snapshots to avoid Jetson RAM OOM while keeping high calibration quality
    calib_images = image_paths[:15]
    print(f"  Using {len(calib_images)} snapshot images for INT8 calibration.")

    # 4. Run static quantization with strict GPU/TRT settings
    print(f"\n[Step 2/4] Quantizing model (FP32 -> QDQ INT8)...")
    from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationMethod
    
    extra_options = {
        "ActivationSymmetric": True,        # Required for TRT compatibility
        "WeightSymmetric": True,            # Symmetrize weights
        "CalibTensorRangeSymmetric": True,  # Required for dynamic ranges on GPU
        "MatMulConstBOnly": True,           # Safely handles transformer attention layers
    }

    reader = SimpleCalibrationReader(calib_images, input_name)
    
    t0 = time.time()
    quantize_static(
        model_input=preprocessed_onnx,
        model_output=args.quant,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=nodes_to_exclude,
        extra_options=extra_options,
    )
    print(f"  Static quantization completed in {time.time() - t0:.1f}s")

    # 5. Graph surgery on the quantized model to restore remaining FP32 biases
    print(f"\n[Step 3/4] Performing graph surgery to restore FP32 biases...")
    quant_model = onnx.load(args.quant)
    fp32_model = onnx.load(preprocessed_onnx)
    
    fp32_initializers = {init.name: init for init in fp32_model.graph.initializer}
    dequant_nodes = {node.output[0]: node for node in quant_model.graph.node if node.op_type == 'DequantizeLinear'}
    
    nodes_to_fix = []
    for node in quant_model.graph.node:
        if node.op_type in ('Conv', 'Gemm') and len(node.input) > 2:
            bias_name = node.input[2]
            if bias_name in dequant_nodes:
                dequant_node = dequant_nodes[bias_name]
                orig_bias_name = dequant_node.output[0]
                if orig_bias_name in fp32_initializers:
                    nodes_to_fix.append((node, dequant_node, orig_bias_name))

    removed_nodes = set()
    for node, dequant_node, orig_bias_name in nodes_to_fix:
        # Copy original FP32 bias initializer back to graph
        orig_init = fp32_initializers[orig_bias_name]
        quant_model.graph.initializer.append(orig_init)
        
        # Connect node directly to the FP32 bias tensor
        node.input[2] = orig_bias_name
        
        # Remove the DequantizeLinear node
        removed_nodes.add(dequant_node.name)

    # Clean up graph nodes
    new_nodes = [n for n in quant_model.graph.node if n.name not in removed_nodes]
    quant_model.graph.ClearField('node')
    quant_model.graph.node.extend(new_nodes)

    onnx.save(quant_model, args.quant)
    print(f"  Graph surgery complete! Restored {len(nodes_to_fix)} FP32 biases. Saved -> {args.quant}")

    # 6. Compile to TRT via trtexec
    print(f"\n[Step 4/4] Compiling quantized ONNX to TensorRT INT8 engine...")
    trtexec = "/usr/src/tensorrt/bin/trtexec"
    if not os.path.exists(trtexec):
        sys.exit(f"Error: trtexec not found at {trtexec}")

    cmd = [
        trtexec,
        f"--onnx={args.quant}",
        f"--saveEngine={args.out}",
        "--fp16",                       # Fallback precision
        "--int8",                       # Use scale factors from QDQ nodes
        "--minShapes=input:1x3x576x576",
        "--optShapes=input:4x3x576x576",
        "--maxShapes=input:16x3x576x576",
        f"--memPoolSize=workspace:{args.workspace}G",
        "--persistentCacheRatio=0.5",   # Pin weights in L2 cache
        "--avgRuns=10",
        "--warmUp=1000",
        "--duration=15",
    ]

    print("\n  Executing trtexec:")
    print("  " + " \\\n    ".join(cmd))
    print()

    t0 = time.time()
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n[ERROR] trtexec failed with exit code {result.returncode}")
        sys.exit(1)

    print(f"\n[SUCCESS] Engine successfully compiled in {time.time() - t0:.1f}s!")
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"  Engine file size: {size_mb:.1f} MB")
    print(f"  Saved -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description="Build TRT INT8 engine for RF-DETR")
    ap.add_argument("--onnx", default="model/rfdetr-seg-medium.onnx")
    ap.add_argument("--out", default="model/rfdetr-seg-medium-int8.engine")
    ap.add_argument("--quant", default="model/rfdetr-int8-gpu.onnx")
    ap.add_argument("--calib", default="alerts/snapshot")
    ap.add_argument("--workspace", type=int, default=4)
    args = ap.parse_args()

    build_pipeline(args)


if __name__ == "__main__":
    main()
