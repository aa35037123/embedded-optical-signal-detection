"""Detection helpers for the embedded optical target tracker."""

from .detector import Detector
from .types import ColorThreshold, DetectionConfig, DetectionResult, MorphologyConfig

__all__ = [
    "ColorThreshold",
    "DetectionConfig",
    "DetectionResult",
    "Detector",
    "MorphologyConfig",
]
