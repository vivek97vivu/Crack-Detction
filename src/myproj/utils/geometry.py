"""
geometry.py
Extracts physical crack geometry from a binary segmentation mask.

Outputs per crack instance:
  - length_mm      : total skeleton path length
  - width_mm       : mean and max crack width along skeleton
  - skeleton_path  : ordered (x, y) pixel coordinates of centreline
  - orientation_deg: dominant crack angle
  - aspect_ratio   : length / max_width (used as crack validity check)

NOTE: All mm measurements require pixel_per_mm calibration.
      Width/length are SURFACE measurements only — depth is not measurable
      from 2D images and must not be reported as API 570/579 depth.
"""

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from dataclasses import dataclass, field
from typing import Optional
import logging

log = logging.getLogger(__name__)

try:
    from skimage.morphology import skeletonize, thin
    from skimage.measure import label as sk_label, regionprops
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    log.warning("scikit-image not found — skeletonisation unavailable. "
                "Run: pip install scikit-image --break-system-packages")


@dataclass
class CrackGeometry:
    length_mm:       float = 0.0
    width_mean_mm:   float = 0.0
    width_max_mm:    float = 0.0
    aspect_ratio:    float = 0.0
    orientation_deg: float = 0.0
    pixel_area:      int   = 0
    skeleton_path:   list  = field(default_factory=list)   # [(x,y), ...]
    is_valid_crack:  bool  = False


# ---------------------------------------------------------------------------
# Core geometry extraction
# ---------------------------------------------------------------------------

def extract_geometry(
    mask: np.ndarray,
    pixel_per_mm: float = 10.0,
    min_length_px: int  = 20,
    min_area_px:   int  = 50,
    sample_interval: int = 5,
) -> list[CrackGeometry]:
    """
    Args
    ----
    mask          : uint8 binary mask (255=crack, 0=background), H×W
    pixel_per_mm  : camera calibration constant
    min_length_px : skeleton paths shorter than this are discarded (noise)
    min_area_px   : connected components smaller than this are discarded

    Returns
    -------
    List of CrackGeometry — one per connected crack instance
    """
    if not SKIMAGE_AVAILABLE:
        return []

    if mask.max() > 1:
        binary = (mask > 127).astype(np.uint8)
    else:
        binary = mask.astype(np.uint8)

    # Distance transform — every crack pixel gets its distance to background
    dist_map = distance_transform_edt(binary)

    # Label connected components so we handle multiple cracks per frame
    labeled   = sk_label(binary)
    props     = regionprops(labeled)
    results   = []

    for region in props:
        if region.area < min_area_px:
            continue

        # Isolate this instance
        inst_mask = (labeled == region.label).astype(np.uint8)

        # Skeletonise
        skel = skeletonize(inst_mask).astype(np.uint8)
        skel_pixels = np.argwhere(skel)   # (N, 2) in (row, col)

        if len(skel_pixels) < min_length_px:
            continue

        # Order skeleton pixels by nearest-neighbour traversal
        ordered = _order_skeleton(skel_pixels)

        # Sample widths along skeleton
        widths_px = []
        for pt in ordered[::sample_interval]:
            row, col = pt
            w = dist_map[row, col] * 2.0   # diameter at this point
            widths_px.append(w)

        if not widths_px:
            continue

        # Orientation via PCA on skeleton points
        pts   = ordered.astype(np.float32)
        mean  = pts.mean(axis=0)
        pts_c = pts - mean
        _, _, vt = np.linalg.svd(pts_c)
        dominant = vt[0]
        orientation_deg = float(np.degrees(np.arctan2(dominant[0], dominant[1])))

        # Physical measurements
        length_px      = float(len(ordered))
        width_mean_px  = float(np.mean(widths_px))
        width_max_px   = float(np.max(widths_px))

        geom = CrackGeometry(
            length_mm       = length_px    / pixel_per_mm,
            width_mean_mm   = width_mean_px / pixel_per_mm,
            width_max_mm    = width_max_px  / pixel_per_mm,
            aspect_ratio    = length_px / (width_max_px + 1e-6),
            orientation_deg = orientation_deg,
            pixel_area      = int(region.area),
            skeleton_path   = [(int(c), int(r)) for r, c in ordered],
            is_valid_crack  = True,   # passed all filters
        )
        results.append(geom)

    return results


def _order_skeleton(skel_pixels: np.ndarray) -> np.ndarray:
    """
    Greedy nearest-neighbour ordering of skeleton pixel coordinates.
    Returns ordered (N, 2) array.
    """
    if len(skel_pixels) == 0:
        return skel_pixels

    pts    = skel_pixels.tolist()
    start  = pts[0]
    ordered = [start]
    remaining = set(map(tuple, pts[1:]))

    current = np.array(start)
    while remaining:
        rem_arr = np.array(list(remaining))
        dists   = np.linalg.norm(rem_arr - current, axis=1)
        nearest_idx = int(np.argmin(dists))
        nearest = rem_arr[nearest_idx]
        ordered.append(nearest.tolist())
        remaining.discard(tuple(nearest.tolist()))
        current = nearest

    return np.array(ordered)


# ---------------------------------------------------------------------------
# Eccentricity / validity filter
# ---------------------------------------------------------------------------

def is_crack_shape(contour: np.ndarray, min_aspect_ratio: float = 3.0) -> bool:
    """
    Rejects blobs that look circular (rust spots, bolt heads, etc.).
    Uses fitted ellipse eccentricity as a quick pre-filter before
    running the full skeleton pipeline.
    """
    if len(contour) < 5:
        return False
    _, (w, h), _ = cv2.fitEllipse(contour)
    if min(w, h) < 1e-6:
        return False
    aspect = max(w, h) / min(w, h)
    return aspect >= min_aspect_ratio


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def draw_geometry(frame: np.ndarray, geom: CrackGeometry,
                  offset_xy: tuple = (0, 0),
                  color: tuple = (0, 255, 0)) -> np.ndarray:
    """
    Draws skeleton path and width label on frame.
    offset_xy: (x, y) offset if geom comes from a cropped ROI.
    """
    ox, oy = offset_xy
    pts = [(x + ox, y + oy) for x, y in geom.skeleton_path]
    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i + 1], color, 1, cv2.LINE_AA)

    if pts:
        mid = pts[len(pts) // 2]
        label = (f"L:{geom.length_mm:.1f}mm "
                 f"W:{geom.width_mean_mm:.2f}mm")
        cv2.putText(frame, label, (mid[0] + 4, mid[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return frame
