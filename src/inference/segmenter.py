import numpy as np
import cv2
from src.utils.geometry import extract_geometry

class SegmenterInference:
    """
    Post-Processing Geometry & Severity Analysis Engine.

    Processes binary masks generated directly by the RF-DETR segmentation model
    to extract sub-millimeter crack length, width, and severity alerts.
    """
    def __init__(self, checkpoint_path=None, device=None, fallback_to_heuristic=True):
        self.fallback_to_heuristic = fallback_to_heuristic

    def predict(self, crop):
        """
        Fallback Otsu thresholding segmentation on an image crop.
        """
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            h = crop.shape[0] if len(crop.shape) > 0 and crop.shape[0] > 0 else 256
            w = crop.shape[1] if len(crop.shape) > 1 and crop.shape[1] > 0 else 256
            return np.zeros((h, w), dtype=np.uint8)

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        return opened

    def process_crop(self, crop, pixel_per_mm, min_length_px, min_area_px, sample_interval, severity_mapper):
        """
        Segments a crop using Otsu thresholding and processes geometry + severity.
        """
        mask = self.predict(crop)
        geom_list, worst_sev = self.process_mask(
            mask, pixel_per_mm, min_length_px, min_area_px, sample_interval, severity_mapper
        )
        return mask, geom_list, worst_sev

    def process_mask(self, mask, pixel_per_mm, min_length_px, min_area_px, sample_interval, severity_mapper):
        """
        Processes geometry and severity classification directly from a binary mask.
        """
        geom_list = extract_geometry(
            mask, 
            pixel_per_mm=pixel_per_mm,
            min_length_px=min_length_px,
            min_area_px=min_area_px,
            sample_interval=sample_interval
        )

        worst_sev = None
        if geom_list:
            severity_results = [
                severity_mapper.classify(g.width_mean_mm, g.length_mm)
                for g in geom_list
            ]
            worst_sev = severity_mapper.worst_level(severity_results)

        return geom_list, worst_sev
