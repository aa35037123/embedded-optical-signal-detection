from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ColorName = Literal["red", "green", "blue", "none"]


@dataclass
class ColorThreshold:
    """HSV threshold configuration for one detected target color."""

    hue_ranges: list[tuple[int, int]] = field(default_factory=lambda: [(0, 12), (168, 180)])
    saturation_min: int = 80
    saturation_max: int = 255
    value_min: int = 60
    value_max: int = 255


@dataclass
class MorphologyConfig:
    open_kernel: int = 3
    close_kernel: int = 5


@dataclass
class DetectionConfig:
    """Global configuration for the CPU-based optical signal detector."""

    width: int = 640
    height: int = 480
    min_area: int = 50
    max_area: int = 5000
    min_confidence: float = 0.0
    min_circularity: float = 0.0
    min_color_contrast: float = 0.0
    color_samples_bgr: dict[str, list[list[int]]] = field(default_factory=dict)
    color_distance_max: float = 30.0
    color_margin: float = 8.0
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)
    color_thresholds: dict[str, ColorThreshold] = field(
        default_factory=lambda: {
            "red": ColorThreshold(hue_ranges=[(0, 12), (168, 180)]),
            "green": ColorThreshold(hue_ranges=[(36, 90)]),
            "blue": ColorThreshold(hue_ranges=[(90, 130)]),
        }
    )


@dataclass
class DetectionResult:
    """Structured result returned by the detector for a candidate blob."""

    valid: bool = False
    detected_color: ColorName = "none"
    centroid_x: float = 0.0
    centroid_y: float = 0.0
    centroid_norm_x: float = 0.0
    centroid_norm_y: float = 0.0
    area: float = 0.0
    radius: float = 0.0
    circularity: float = 0.0
    color_contrast: float = 0.0
    confidence: float = 0.0
    mean_brightness: float = 0.0
    fps: float = 0.0
    width: int = 640
    height: int = 480
