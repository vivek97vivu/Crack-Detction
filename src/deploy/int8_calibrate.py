"""
int8_calibrate.py — TensorRT INT8 Calibrator for RF-DETR Crack Detection

Uses real crack snapshots captured during live deployment as calibration data.
Preprocessing exactly matches the live inference path in detector.py._predict_trt().

Usage (called internally by build_int8_engine.py):
    calibrator = CrackInt8Calibrator(
        image_dir="alerts/snapshot",
        cache_file="model/int8_calibration.cache",
        input_shape=(1, 3, 576, 576),
    )
"""

from __future__ import annotations

import ctypes
import glob
import logging
import os
import sys

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── TensorRT import (system dist-packages) ─────────────────────────────────────
_TRT_SYSTEM_PATH = "/usr/lib/python3.10/dist-packages"
if _TRT_SYSTEM_PATH not in sys.path:
    sys.path.insert(0, _TRT_SYSTEM_PATH)

import tensorrt as trt

# ── CUDA runtime ───────────────────────────────────────────────────────────────
_cudart = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
_H2D    = ctypes.c_int(1)   # cudaMemcpyHostToDevice


def _cuda_malloc(nbytes: int) -> int:
    ptr = ctypes.c_void_p()
    ret = _cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
    if ret != 0:
        raise RuntimeError(f"cudaMalloc({nbytes}) failed with code {ret}")
    return ptr.value


def _cuda_free(ptr: int):
    _cudart.cudaFree(ctypes.c_void_p(ptr))


def _cuda_h2d(dst: int, src: np.ndarray):
    ret = _cudart.cudaMemcpy(
        ctypes.c_void_p(dst),
        src.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(src.nbytes),
        _H2D,
    )
    if ret != 0:
        raise RuntimeError(f"cudaMemcpy H2D failed with code {ret}")


# ── ImageNet normalization constants ───────────────────────────────────────────
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(bgr: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Identical preprocessing to detector.py._predict_trt():
      BGR -> RGB -> resize(576x576) -> float32 /255 -> ImageNet normalize -> NCHW

    Returns:
        np.ndarray of shape (1, 3, h, w), dtype=float32, C-contiguous
    """
    # Strip alpha if present (BGRx from nvvidconv)
    if bgr.ndim == 3 and bgr.shape[2] == 4:
        bgr = bgr[:, :, :3]

    img_rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    img_float   = img_resized.astype(np.float32) / 255.0
    img_norm    = (img_float - _MEAN) / _STD
    # HWC -> NCHW
    inp = np.ascontiguousarray(
        np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...], dtype=np.float32
    )
    return inp


class CrackInt8Calibrator(trt.IInt8EntropyCalibrator2):
    """
    TensorRT INT8 calibrator using real crack snapshots.

    IInt8EntropyCalibrator2 is the recommended calibrator for most vision models.
    It collects activation histograms and computes optimal INT8 thresholds using
    information-theoretic entropy minimization (KL-divergence method).

    Parameters
    ----------
    image_dir   : directory containing JPEG calibration images
    cache_file  : path to write/read calibration cache (saves ~20 min per rebuild)
    input_shape : (N, C, H, W) — must match engine input. N=1 during calibration.
    """

    def __init__(
        self,
        image_dir: str = "alerts/snapshot",
        cache_file: str = "model/int8_calibration.cache",
        input_shape: tuple = (1, 3, 576, 576),
    ):
        super().__init__()

        self._cache_file  = cache_file
        self._input_shape = input_shape
        self._batch_size  = input_shape[0]      # 1 during calibration
        _, _, self._h, self._w = input_shape

        # Collect all images in the calibration directory
        patterns = [
            os.path.join(image_dir, "*.jpg"),
            os.path.join(image_dir, "*.jpeg"),
            os.path.join(image_dir, "*.png"),
        ]
        self._images: list = []
        for pat in patterns:
            self._images.extend(sorted(glob.glob(pat)))

        if len(self._images) == 0:
            raise FileNotFoundError(
                f"No calibration images found in '{image_dir}'.\n"
                "Run the live pipeline first to generate snapshots, or provide "
                "a folder of .jpg / .png crack images."
            )

        logger.info(
            "[Calibrator] %d calibration images found in '%s'",
            len(self._images), image_dir,
        )
        print(
            f"[Calibrator] {len(self._images)} calibration images found in '{image_dir}'"
        )

        self._index   = 0
        self._gpu_ptr = None
        self._nbytes  = int(np.prod(input_shape)) * 4   # float32 = 4 bytes

        # Allocate a persistent GPU scratch buffer (reused across all calibration steps)
        self._gpu_ptr = _cuda_malloc(self._nbytes)

    # ── TensorRT calibrator interface ─────────────────────────────────────────

    def get_batch_size(self) -> int:
        """TRT calls this to know how many samples per calibration step."""
        return self._batch_size   # Always 1 during calibration

    def get_batch(self, names: list):
        """
        Called by TensorRT repeatedly to supply calibration batches.
        Returns list of GPU device pointers (one per input tensor), or None when done.
        """
        if self._index >= len(self._images):
            print()   # newline after \r progress
            return None   # End of calibration dataset

        img_path = self._images[self._index]
        self._index += 1

        bgr = cv2.imread(img_path)
        if bgr is None:
            logger.warning("[Calibrator] Could not read '%s' — skipping", img_path)
            return self.get_batch(names)    # Recurse to next image

        inp = preprocess_image(bgr, self._h, self._w)
        _cuda_h2d(self._gpu_ptr, inp)

        print(
            f"\r[Calibrator] [{self._index}/{len(self._images)}] "
            f"{os.path.basename(img_path):40s}",
            end="",
            flush=True,
        )

        return [ctypes.c_void_p(self._gpu_ptr)]

    def read_calibration_cache(self):
        """Return cached calibration bytes if they exist, else None (triggers full calibration)."""
        if os.path.exists(self._cache_file):
            print(f"[Calibrator] Loading calibration cache: {self._cache_file}")
            with open(self._cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes):
        """Write calibration cache to disk so future builds skip calibration."""
        os.makedirs(os.path.dirname(os.path.abspath(self._cache_file)), exist_ok=True)
        with open(self._cache_file, "wb") as f:
            f.write(cache)
        print(f"\n[Calibrator] Cache saved -> {self._cache_file}  ({len(cache):,} bytes)")

    def __del__(self):
        if self._gpu_ptr is not None:
            try:
                _cuda_free(self._gpu_ptr)
                self._gpu_ptr = None
            except Exception:
                pass
