"""
trt_engine.py — Native TensorRT inference backend for Jetson (JetPack 6.x / TRT 10.x)

Loads a pre-built .engine file and runs GPU inference using:
  - TensorRT Python bindings from system dist-packages
    (/usr/lib/python3.10/dist-packages/tensorrt)
  - libcudart.so via ctypes for CUDA memory management (no pycuda required)

Usage:
    from src.inference.trt_engine import TRTEngineBackend
    engine = TRTEngineBackend("model/rfdetr-seg-medium-fp16.engine")
    outputs = engine.infer({"input": np_array})  # dict[name -> np.ndarray]
    engine.destroy()
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Locate and import TensorRT from system dist-packages ─────────────────────
_TRT_SYSTEM_PATH = "/usr/lib/python3.10/dist-packages"
if _TRT_SYSTEM_PATH not in sys.path:
    sys.path.insert(0, _TRT_SYSTEM_PATH)

try:
    import tensorrt as trt
    _TRT_AVAILABLE = True
    logger.info("TensorRT %s loaded from system dist-packages.", trt.__version__)
except ImportError:
    _TRT_AVAILABLE = False
    logger.warning("TensorRT Python bindings not found. .engine file loading unavailable.")

# ── CUDA runtime via ctypes ───────────────────────────────────────────────────
_CUDART_PATH = "/usr/local/cuda/lib64/libcudart.so"
try:
    _cudart = ctypes.CDLL(_CUDART_PATH)
except OSError:
    _cudart = None
    logger.warning("libcudart.so not found at %s. TRT engine backend unavailable.", _CUDART_PATH)

# cudaMemcpyKind enum values
_H2D = ctypes.c_int(1)  # cudaMemcpyHostToDevice
_D2H = ctypes.c_int(2)  # cudaMemcpyDeviceToHost

# ── dtype helpers ─────────────────────────────────────────────────────────────
_TRT_DTYPE_TO_NP = None  # Populated after TRT import

def _trt_to_np(trt_dtype):
    """Map TensorRT dtype → numpy dtype."""
    _map = {
        trt.float32: np.float32,
        trt.float16: np.float16,
        trt.int32:   np.int32,
        trt.int8:    np.int8,
        trt.bool:    np.bool_,
    }
    return _map.get(trt_dtype, np.float32)


# ── Low-level CUDA helpers ────────────────────────────────────────────────────
def _cuda_malloc(nbytes: int) -> int:
    """Allocate nbytes on the GPU; return device pointer as Python int."""
    ptr = ctypes.c_void_p()
    ret = _cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
    if ret != 0:
        raise RuntimeError(f"cudaMalloc({nbytes}) failed with code {ret}")
    return ptr.value

def _cuda_free(ptr: int):
    _cudart.cudaFree(ctypes.c_void_p(ptr))

def _cuda_h2d(dst_ptr: int, src_arr: np.ndarray):
    """Copy numpy array (host) → GPU buffer (device), synchronous."""
    src = src_arr.ctypes.data_as(ctypes.c_void_p)
    ret = _cudart.cudaMemcpy(
        ctypes.c_void_p(dst_ptr), src, ctypes.c_size_t(src_arr.nbytes), _H2D
    )
    if ret != 0:
        raise RuntimeError(f"cudaMemcpy H2D failed: {ret}")

def _cuda_d2h(dst_arr: np.ndarray, src_ptr: int):
    """Copy GPU buffer (device) → numpy array (host), synchronous."""
    dst = dst_arr.ctypes.data_as(ctypes.c_void_p)
    ret = _cudart.cudaMemcpy(
        dst, ctypes.c_void_p(src_ptr), ctypes.c_size_t(dst_arr.nbytes), _D2H
    )
    if ret != 0:
        raise RuntimeError(f"cudaMemcpy D2H failed: {ret}")

def _cuda_stream_create() -> int:
    stream = ctypes.c_void_p()
    _cudart.cudaStreamCreate(ctypes.byref(stream))
    return stream.value

def _cuda_stream_sync(stream: int):
    _cudart.cudaStreamSynchronize(ctypes.c_void_p(stream))

def _cuda_stream_destroy(stream: int):
    _cudart.cudaStreamDestroy(ctypes.c_void_p(stream))


# ── TRTEngineBackend ──────────────────────────────────────────────────────────
class TRTEngineBackend:
    """
    Loads a TensorRT .engine file and exposes a simple infer() method.

    Compatible with TRT 8.x and 10.x API (auto-detects).
    """

    def __init__(self, engine_path: str):
        if not _TRT_AVAILABLE:
            raise RuntimeError("TensorRT Python bindings unavailable.")
        if _cudart is None:
            raise RuntimeError("libcudart.so unavailable.")
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f"TRT engine not found: {engine_path}")

        self._engine_path = engine_path
        self._trt_logger = trt.Logger(trt.Logger.WARNING)

        # Load serialised engine
        runtime = trt.Runtime(self._trt_logger)
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        self._engine = runtime.deserialize_cuda_engine(engine_data)
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self._context = self._engine.create_execution_context()
        self._stream  = _cuda_stream_create()

        # ── Discover I/O tensors (TRT 10 API) ────────────────────────────
        self._inputs:  List[dict] = []
        self._outputs: List[dict] = []

        n = self._engine.num_io_tensors
        for i in range(n):
            name  = self._engine.get_tensor_name(i)
            shape = tuple(self._engine.get_tensor_shape(name))
            dtype = _trt_to_np(self._engine.get_tensor_dtype(name))
            mode  = self._engine.get_tensor_mode(name)

            # Replace any -1 dynamic dims with 1 temporarily; caller sets shape
            static_shape = tuple(max(d, 1) for d in shape)
            nbytes = int(np.prod(static_shape)) * np.dtype(dtype).itemsize
            gpu_ptr = _cuda_malloc(nbytes)

            info = {
                "name":         name,
                "shape":        shape,
                "static_shape": static_shape,
                "dtype":        dtype,
                "gpu_ptr":      gpu_ptr,
                "nbytes":       nbytes,
            }
            if mode == trt.TensorIOMode.INPUT:
                self._inputs.append(info)
            else:
                self._outputs.append(info)

        logger.info(
            "[TRTEngineBackend] Loaded %s  inputs=%s  outputs=%s",
            os.path.basename(engine_path),
            [t["name"] for t in self._inputs],
            [t["name"] for t in self._outputs],
        )

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def input_names(self) -> List[str]:
        return [t["name"] for t in self._inputs]

    @property
    def output_names(self) -> List[str]:
        return [t["name"] for t in self._outputs]

    def infer(self, feed: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run one forward pass.

        Parameters
        ----------
        feed : dict[input_name -> np.ndarray]  (float32, contiguous)

        Returns
        -------
        dict[output_name -> np.ndarray]
        """
        # ── Upload inputs to GPU ──────────────────────────────────────────
        for inp in self._inputs:
            name = inp["name"]
            arr  = np.ascontiguousarray(feed[name], dtype=inp["dtype"])

            # Handle dynamic batch / spatial dims
            actual_shape = arr.shape
            if actual_shape != inp["static_shape"]:
                nbytes = arr.nbytes
                if nbytes > inp["nbytes"]:
                    # Reallocate if input grew (unlikely for fixed 576x576)
                    _cuda_free(inp["gpu_ptr"])
                    inp["gpu_ptr"] = _cuda_malloc(nbytes)
                    inp["nbytes"]  = nbytes
                inp["static_shape"] = actual_shape
                self._context.set_input_shape(name, actual_shape)

            self._context.set_tensor_address(name, inp["gpu_ptr"])
            _cuda_h2d(inp["gpu_ptr"], arr)

        # ── Set output tensor addresses ───────────────────────────────────
        for out in self._outputs:
            # After set_input_shape, query real output shape
            real_shape = tuple(self._context.get_tensor_shape(out["name"]))
            real_shape = tuple(max(d, 1) for d in real_shape)
            nbytes = int(np.prod(real_shape)) * np.dtype(out["dtype"]).itemsize
            if nbytes > out["nbytes"]:
                _cuda_free(out["gpu_ptr"])
                out["gpu_ptr"] = _cuda_malloc(nbytes)
                out["nbytes"]  = nbytes
            out["real_shape"] = real_shape
            self._context.set_tensor_address(out["name"], out["gpu_ptr"])

        # ── Execute ───────────────────────────────────────────────────────
        ok = self._context.execute_async_v3(self._stream)
        if not ok:
            raise RuntimeError("TRT execute_async_v3 returned False")
        _cuda_stream_sync(self._stream)

        # ── Download outputs to CPU ───────────────────────────────────────
        results: Dict[str, np.ndarray] = {}
        for out in self._outputs:
            shape = out.get("real_shape", out["static_shape"])
            arr   = np.empty(shape, dtype=out["dtype"])
            _cuda_d2h(arr, out["gpu_ptr"])
            results[out["name"]] = arr

        return results

    def destroy(self):
        """Free all GPU buffers and CUDA stream."""
        for t in self._inputs + self._outputs:
            if t.get("gpu_ptr"):
                _cuda_free(t["gpu_ptr"])
                t["gpu_ptr"] = 0
        if self._stream:
            _cuda_stream_destroy(self._stream)
            self._stream = 0

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.destroy()


def is_available() -> bool:
    """Return True if TRT engine loading is possible on this system."""
    return _TRT_AVAILABLE and _cudart is not None
