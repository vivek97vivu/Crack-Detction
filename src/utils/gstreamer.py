from __future__ import annotations

import logging
import subprocess
import os
from functools import lru_cache
from typing import Optional

import cv2

logger = logging.getLogger(__name__)


# ── Decoder capability detection ─────────────────────────────────────

@lru_cache(maxsize=None)
def _gst_plugin_available(plugin_name: str) -> bool:
    """Return True if a GStreamer plugin/element is installed."""
    try:
        inspect_bin = "/usr/bin/gst-inspect-1.0" if os.path.exists("/usr/bin/gst-inspect-1.0") else "gst-inspect-1.0"
        result = subprocess.run(
            [inspect_bin, plugin_name],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _detect_best_decoder(codec: str) -> str:
    """
    Detect the best available GStreamer decoder element for the given codec.

    Returns one of:
      'jetson_hw'       — nvv4l2decoder (Jetson VPU, JP5/JP6 — correct element name)
      'nvidia_gpu'      — nvh265dec / nvh264dec (x86 NVIDIA GPU NVDEC)
      'nvidia_gpu_legacy' — nvdec (legacy x86 NVIDIA GPU)
      'vaapi'           — vaapih265dec / vaapih264dec (Intel/AMD VA-API)
      'software'        — avdec_h265 / avdec_h264 (CPU libav fallback)
    """
    codec = codec.lower()

    # 1. Jetson VPU hardware decoder — nvv4l2decoder is the correct element on
    #    JetPack 5.x and 6.x (JP6.2 / R36.x).  The old nvh265dec / nvh264dec
    #    elements do NOT exist on these platforms.
    if _gst_plugin_available("nvv4l2decoder"):
        logger.info("Decoder selected: Jetson HW VPU (nvv4l2decoder)")
        return "jetson_hw"

    # 2. Modern NVIDIA NVDEC on x86 GPU (gst-plugins-bad nvcodec elements)
    nvdec_elem = f"nvh{codec[1:]}dec" if codec in ("h264", "h265") else "nvh265dec"
    if _gst_plugin_available(nvdec_elem):
        logger.info("Decoder selected: NVIDIA GPU NVDEC (%s)", nvdec_elem)
        return "nvidia_gpu"
    elif _gst_plugin_available("nvdec"):
        logger.info("Decoder selected: NVIDIA GPU NVDEC (legacy nvdec)")
        return "nvidia_gpu_legacy"

    # 3. VA-API (Intel/AMD iGPU, common on x86 Linux)
    vaapi_elem = f"vaapi{codec}dec"
    if _gst_plugin_available(vaapi_elem):
        logger.info("Decoder selected: VA-API (%s)", vaapi_elem)
        return "vaapi"

    # 4. Software fallback — always available if gstreamer-plugins-bad/ugly installed
    logger.info("Decoder selected: software libav (avdec_%s)", codec)
    return "software"


# ── Pipeline templates ────────────────────────────────────────────────
# {url}       → RTSP URL
# {protocols} → tcp | udp
# {latency}   → ms (0 for live)
# {drop}      → true | false

_TEMPLATES: dict[str, dict[str, str]] = {
    # Jetson VPU hardware decoder — nvv4l2decoder is the correct element on
    # JetPack 5.x and 6.x (JP6.2 / R36.x).  Use nvvidconv (HW colorspace
    # converter) instead of videoconvert (CPU) for zero-copy BGR conversion.
    "jetson_hw": {
        "h265": (
            "rtspsrc location=\"{url}\" protocols={protocols} "
            "latency={latency} drop-on-latency={drop} buffer-mode=0 ! "
            "rtph265depay ! h265parse ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true sync=false max-buffers=1"
        ),
        "h264": (
            "rtspsrc location=\"{url}\" protocols={protocols} "
            "latency={latency} drop-on-latency={drop} buffer-mode=0 ! "
            "rtph264depay ! h264parse ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true sync=false max-buffers=1"
        ),
    },
    # Modern NVIDIA NVDEC on x86 GPU (gst-plugins-bad nvcodec elements)
    "nvidia_gpu": {
        "h265": (
            "rtspsrc location=\"{url}\" protocols={protocols} "
            "latency={latency} drop-on-latency={drop} buffer-mode=0 ! "
            "rtph265depay ! h265parse ! nvh265dec ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true sync=false max-buffers=1"
        ),
        "h264": (
            "rtspsrc location=\"{url}\" protocols={protocols} "
            "latency={latency} drop-on-latency={drop} buffer-mode=0 ! "
            "rtph264depay ! h264parse ! nvh264dec ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true sync=false max-buffers=1"
        ),
    },
    # Legacy NVIDIA NVDEC (nvdec element name)
    "nvidia_gpu_legacy": {
        "h265": (
            "rtspsrc location=\"{url}\" protocols={protocols} "
            "latency={latency} drop-on-latency={drop} buffer-mode=0 ! "
            "rtph265depay ! h265parse ! nvdec ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true sync=false max-buffers=1"
        ),
        "h264": (
            "rtspsrc location=\"{url}\" protocols={protocols} "
            "latency={latency} drop-on-latency={drop} buffer-mode=0 ! "
            "rtph264depay ! h264parse ! nvdec ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true sync=false max-buffers=1"
        ),
    },
    # VA-API (Intel/AMD iGPU)
    "vaapi": {
        "h265": (
            "rtspsrc location=\"{url}\" protocols={protocols} "
            "latency={latency} drop-on-latency={drop} ! "
            "rtph265depay ! h265parse ! vaapih265dec ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true sync=false max-buffers=1"
        ),
        "h264": (
            "rtspsrc location=\"{url}\" protocols={protocols} "
            "latency={latency} drop-on-latency={drop} ! "
            "rtph264depay ! h264parse ! vaapih264dec ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true sync=false max-buffers=1"
        ),
    },
    # Software decoder — universal fallback
    "software": {
        "h265": (
            "rtspsrc location=\"{url}\" protocols={protocols} "
            "latency={latency} drop-on-latency={drop} ! "
            "rtph265depay ! h265parse ! avdec_h265 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true sync=false max-buffers=1"
        ),
        "h264": (
            "rtspsrc location=\"{url}\" protocols={protocols} "
            "latency={latency} drop-on-latency={drop} ! "
            "rtph264depay ! h264parse ! avdec_h264 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true sync=false max-buffers=1"
        ),
    },
}

# Decoder fallback order — if preferred decoder fails, try the next
_DECODER_FALLBACK_ORDER = ["jetson_hw", "nvidia_gpu", "vaapi", "software"]


# ── Public API ────────────────────────────────────────────────────────

def build_pipeline_string(
    url:             str,
    codec:           str  = "h265",
    protocols:       str  = "tcp",
    latency:         int  = 0,
    drop_on_latency: bool = True,
    decoder:         Optional[str] = None,   # None = auto-detect
) -> str:
    """
    Build the GStreamer pipeline string.
    """
    codec_key = codec.lower().strip()
    if codec_key not in ("h264", "h265"):
        logger.warning("Unknown codec '%s', defaulting to h265.", codec)
        codec_key = "h265"

    selected = decoder or _detect_best_decoder(codec_key)
    template = _TEMPLATES.get(selected, {}).get(codec_key)
    if template is None:
        logger.warning("No template for decoder=%s codec=%s, using software.", selected, codec_key)
        template = _TEMPLATES["software"][codec_key]

    pipeline = template.format(
        url=url,
        protocols=protocols,
        latency=latency,
        drop=str(drop_on_latency).lower(),
    )
    logger.debug("[pipeline] %s", pipeline)
    return pipeline


def build_gstreamer_capture(camera_config: dict) -> cv2.VideoCapture:
    """
    Open a cv2.VideoCapture using a GStreamer pipeline.
    Tries each decoder tier in order until one succeeds.

    Parameters
    ----------
    camera_config : dict

    Returns
    -------
    cv2.VideoCapture — caller must check .isOpened()
    """
    url = camera_config.get("source")
    codec_key = camera_config.get("codec", "h265").lower().strip()
    latency = camera_config.get("latency", 0)
    protocols = camera_config.get("protocols", "tcp")
    drop_on_latency = camera_config.get("drop_on_latency", True)
    cam_id = camera_config.get("id", "camera")

    # Start from the best available decoder, fall through on failure
    best = _detect_best_decoder(codec_key)
    order = _DECODER_FALLBACK_ORDER[_DECODER_FALLBACK_ORDER.index(best):]

    for decoder_tier in order:
        pipeline = build_pipeline_string(
            url=str(url),
            codec=codec_key,
            protocols=protocols,
            latency=latency,
            drop_on_latency=drop_on_latency,
            decoder=decoder_tier,
        )

        logger.info(
            "[%s] Trying GStreamer capture — decoder=%s codec=%s",
            cam_id, decoder_tier, codec_key,
        )

        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not cap.isOpened():
            logger.warning(
                "[%s] decoder=%s failed to open — trying next tier.",
                cam_id, decoder_tier,
            )
            cap.release()
            continue

        # Verify frames actually flow (isOpened() can return True on broken pipelines)
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.warning(
                "[%s] decoder=%s opened but first frame read failed — trying next tier.",
                cam_id, decoder_tier,
            )
            cap.release()
            continue

        logger.info(
            "[%s] GStreamer capture OK — decoder=%s  %dx%d",
            cam_id, decoder_tier, frame.shape[1], frame.shape[0],
        )
        return cap

    logger.error(
        "[%s] All GStreamer decoder tiers failed for codec=%s.",
        cam_id, codec_key,
    )
    # Return a dead capture — caller checks isOpened()
    return cv2.VideoCapture()
