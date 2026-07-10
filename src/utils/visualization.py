"""
utils/visualization.py — Drawing helpers for crack detection pipeline.
"""
import cv2
import numpy as np
from typing import Optional

SEVERITY_COLORS = {
    1: (0, 255, 255),   # Level 1 — yellow
    2: (0, 165, 255),   # Level 2 — orange
    3: (0, 0, 255),     # Level 3 — red
}
SEVERITY_LABELS = {
    1: "L1:MONITOR",
    2: "L2:REPAIR",
    3: "L3:SHUTDOWN",
}


def draw_detections(frame: np.ndarray, detections: list,
                    show_conf: bool = True) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        color = (0, 255, 0) if det.class_name == "crack" else (255, 165, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name}"
        if det.track_id is not None:
            label += f" ID:{det.track_id}"
        if show_conf:
            label += f" {det.confidence:.2f}"
        cv2.putText(out, label, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def draw_mask_overlay(frame: np.ndarray, mask: np.ndarray,
                      bbox_xyxy: np.ndarray,
                      color: tuple = (0, 0, 255),  # default to Red overlay for cracks in BGR
                      alpha: float = 0.4) -> np.ndarray:
    out      = frame.copy()
    x1, y1, x2, y2 = bbox_xyxy
    roi      = out[y1:y2, x1:x2]
    if roi.size == 0 or mask.size == 0:
        return out
    mask_rs  = cv2.resize(mask, (x2 - x1, y2 - y1),
                           interpolation=cv2.INTER_NEAREST)
    overlay  = roi.copy()
    overlay[mask_rs > 127] = color
    out[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, overlay, alpha, 0)
    return out


def draw_severity_badge(frame: np.ndarray, severity_result,
                         bbox_xyxy: np.ndarray) -> np.ndarray:
    out = frame.copy()
    x1, y1, x2, y2 = bbox_xyxy
    lv    = severity_result.level.value
    color = SEVERITY_COLORS.get(lv, (128, 128, 128))
    label = SEVERITY_LABELS.get(lv, f"L{lv}")
    cv2.rectangle(out, (x1, y2 + 1), (x2, y2 + 18), color, -1)
    cv2.putText(out, label, (x1 + 2, y2 + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def draw_pipeline_hud(frame: np.ndarray, gate_prob: Optional[float],
                       n_detections: int, n_cracks: int) -> np.ndarray:
    out = frame.copy()
    lines = []
    if gate_prob is not None:
        lines.append(f"Gate P(crack): {gate_prob:.3f}")
    lines.extend([
        f"Detections   : {n_detections}",
        f"Cracks       : {n_cracks}",
    ])
    for i, line in enumerate(lines):
        cv2.putText(out, line, (8, 20 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out
