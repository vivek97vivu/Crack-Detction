"""
severity.py
Maps crack geometry to API 570/579 fitness-for-service severity levels.

IMPORTANT: Width and length here are SURFACE measurements from 2D imaging.
API 579 Part 9 requires depth for full FFS assessment.
This module reports surface severity only — document this in all outputs.
"""

from dataclasses import dataclass
from enum import IntEnum
import yaml
import logging

log = logging.getLogger(__name__)


class SeverityLevel(IntEnum):
    CLEAR     = 0   # no crack detected
    LEVEL_1   = 1   # monitor — within acceptable limits
    LEVEL_2   = 2   # schedule repair — approaching limits
    LEVEL_3   = 3   # immediate shutdown — exceeds limits


@dataclass
class SeverityResult:
    level:             SeverityLevel
    action:            str
    reinspection_days: int
    width_mm:          float
    length_mm:         float
    trigger_reason:    str
    surface_only_note: str = (
        "Surface measurement only. "
        "Crack depth not determinable from 2D imaging. "
        "Full API 579 FFS assessment requires NDE depth measurement."
    )


class APISeverityMapper:
    """
    Maps CrackGeometry → SeverityResult using thresholds from config.yaml.

    Thresholds should be tuned per client inspection spec.
    """

    def __init__(self, cfg: dict):
        # Fallback dictionary structure if "severity" key is missing
        sv = cfg.get("severity", {
            "level_1": {
                "max_width_mm": 0.2,
                "max_length_mm": 20.0,
                "action": "Monitor and log during routine checkups",
                "reinspection_days": 180
            },
            "level_2": {
                "max_width_mm": 0.5,
                "max_length_mm": 50.0,
                "action": "Schedule repair/maintenance within 30 days",
                "reinspection_days": 30
            },
            "level_3": {
                "action": "Immediate shutdown or emergency maintenance",
                "reinspection_days": 0
            }
        })
        self.l1 = sv.get("level_1", {})
        self.l2 = sv.get("level_2", {})
        self.l3 = sv.get("level_3", {})

    def classify(self, width_mm: float, length_mm: float) -> SeverityResult:
        """
        Args
        ----
        width_mm   : mean crack width from geometry.py
        length_mm  : crack skeleton length from geometry.py

        Returns
        -------
        SeverityResult with level, action, and reinspection schedule
        """
        # Level 3: immediate shutdown
        max_w_l2 = self.l2.get("max_width_mm", 0.5)
        max_l_l2 = self.l2.get("max_length_mm", 50.0)
        
        if width_mm > max_w_l2 or length_mm > max_l_l2:
            reason = (
                f"Width {width_mm:.3f}mm > {max_w_l2}mm"
                if width_mm > max_w_l2
                else f"Length {length_mm:.1f}mm > {max_l_l2}mm"
            )
            return SeverityResult(
                level             = SeverityLevel.LEVEL_3,
                action            = self.l3.get("action", "Immediate shutdown or emergency maintenance"),
                reinspection_days = self.l3.get("reinspection_days", 0),
                width_mm          = width_mm,
                length_mm         = length_mm,
                trigger_reason    = reason,
            )

        # Level 2: schedule repair
        max_w_l1 = self.l1.get("max_width_mm", 0.2)
        max_l_l1 = self.l1.get("max_length_mm", 20.0)
        
        if width_mm > max_w_l1 or length_mm > max_l_l1:
            reason = (
                f"Width {width_mm:.3f}mm > {max_w_l1}mm"
                if width_mm > max_w_l1
                else f"Length {length_mm:.1f}mm > {max_l_l1}mm"
            )
            return SeverityResult(
                level             = SeverityLevel.LEVEL_2,
                action            = self.l2.get("action", "Schedule repair/maintenance within 30 days"),
                reinspection_days = self.l2.get("reinspection_days", 30),
                width_mm          = width_mm,
                length_mm         = length_mm,
                trigger_reason    = reason,
            )

        # Level 1: within limits — monitor
        return SeverityResult(
            level             = SeverityLevel.LEVEL_1,
            action            = self.l1.get("action", "Monitor and log during routine checkups"),
            reinspection_days = self.l1.get("reinspection_days", 180),
            width_mm          = width_mm,
            length_mm         = length_mm,
            trigger_reason    = "Within level-1 thresholds",
        )

    def batch_classify(self, geom_list: list) -> list[SeverityResult]:
        """Classify a list of CrackGeometry objects."""
        from utils.geometry import CrackGeometry
        results = []
        for geom in geom_list:
            if not isinstance(geom, CrackGeometry) or not geom.is_valid_crack:
                continue
            results.append(self.classify(geom.width_mean_mm, geom.length_mm))
        return results

    @staticmethod
    def worst_level(results: list[SeverityResult]) -> SeverityResult | None:
        """Returns the highest-severity result from a list."""
        if not results:
            return None
        return max(results, key=lambda r: r.level)
