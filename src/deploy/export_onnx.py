"""
export_onnx.py — Export RF-DETR seg.pth to ONNX for GPU inference via ORT.

Uses rfdetr's native .export() which handles all custom ops correctly.
Shape must be divisible by 24 (patch_size=12 × num_windows=2).
Closest valid size to 560: 576.

Usage (from project root, crack env):
    python src/deploy/export_onnx.py

After export, set detector checkpoint in config.yaml to 'model/rfdetr-seg-medium.onnx'
and the detector will automatically use ORT + TensorRT/CUDAExecutionProvider.
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import rfdetr
import torch
import torch.nn.functional as F


def _patch_antialias():
    """
    Patch torch.nn.functional.interpolate to disable antialias.

    rfdetr's DINOv2 backbone calls F.interpolate(..., antialias=True) which
    creates aten::_upsample_bicubic2d_aa — an op not supported by the legacy
    TorchScript ONNX exporter (torch 2.13 / opset ≤ 20).  Forcing antialias=False
    makes the export use the standard aten::upsample_bicubic2d which is fully
    supported.  The accuracy difference is negligible for inference.
    """
    _orig = F.interpolate
    def _patched(input, size=None, scale_factor=None, mode='nearest',
                 align_corners=None, recompute_scale_factor=None, antialias=False):
        return _orig(input, size=size, scale_factor=scale_factor, mode=mode,
                     align_corners=align_corners,
                     recompute_scale_factor=recompute_scale_factor,
                     antialias=False)
    F.interpolate = _patched
    print("  [patch] F.interpolate antialias disabled for ONNX export compatibility")


def export(checkpoint_path: str = "model/seg.pth",
           output_dir: str = "model",
           shape: int = 576,
           opset: int = 17):
    """
    Export RF-DETR seg.pth → ONNX using rfdetr's native exporter.

    Parameters
    ----------
    checkpoint_path : str
        Path to the .pth checkpoint.
    output_dir : str
        Output directory for the .onnx file.
    shape : int
        Square input size. Must be divisible by 24 (patch_size=12 × num_windows=2).
        Valid values near 560: 552, 576 (recommended), 600.
    opset : int
        ONNX opset version. Use 17 (patch applied to support bicubic upsample).
    """
    if shape % 24 != 0:
        raise ValueError(f"shape={shape} must be divisible by 24. Valid near 560: 552, 576, 600.")

    # Must patch before importing rfdetr model (DINOv2 backbone uses antialias=True)
    _patch_antialias()

    print(f"Loading RF-DETR checkpoint (on CPU): {checkpoint_path}")
    model = rfdetr.from_checkpoint(checkpoint_path)
    model.optimize_for_inference(compile=False)


    print(f"Exporting to ONNX  shape=({shape},{shape})  opset={opset} ...")
    out = model.export(
        output_dir=output_dir,
        format="onnx",
        opset_version=opset,
        shape=(shape, shape),
        batch_size=1,
        dynamic_batch=False,
        verbose=False,
    )

    size_mb = os.path.getsize(str(out)) / 1024 / 1024
    print(f"\n✅ ONNX model saved: {out}  ({size_mb:.1f} MB)")
    print(f"\nTo enable GPU inference, update config.yaml:")
    print(f"  detector:")
    print(f"    checkpoint_ema: \"{out}\"")
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="Export RF-DETR seg.pth to ONNX")
    ap.add_argument("--checkpoint", default="model/seg.pth")
    ap.add_argument("--output-dir", default="model")
    ap.add_argument("--shape",  type=int, default=576,
                    help="Square input size divisible by 24 (e.g. 552, 576, 600)")
    ap.add_argument("--opset",  type=int, default=20,
                    help="ONNX opset version (≥20 required for this model)")
    args = ap.parse_args()
    export(args.checkpoint, args.output_dir, args.shape, args.opset)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
