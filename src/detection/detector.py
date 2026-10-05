from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from src.detection.calibration import calibrated_masks, color_features, validate_samples
from src.detection.types import ColorThreshold, DetectionConfig, DetectionResult, MorphologyConfig


def load_detection_config(config_path: str | Path | None = None) -> DetectionConfig:
    """Load detector settings from YAML. Falls back to sensible defaults if absent."""
    default_path = Path(__file__).resolve().parents[2] / "configs" / "detection.yaml"
    config_file = Path(config_path) if config_path is not None else default_path

    if not config_file.exists() and config_path is None:
        return DetectionConfig()

    with config_file.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    colors = raw.get("colors", {})
    threshold_map: dict[str, ColorThreshold] = {}
    for color_name, values in colors.items():
        hue_ranges = values.get("hue_ranges", [])
        threshold_map[color_name] = ColorThreshold(
            hue_ranges=[(int(lo), int(hi)) for lo, hi in hue_ranges],
            saturation_min=int(values.get("saturation_min", 80)),
            saturation_max=int(values.get("saturation_max", 255)),
            value_min=int(values.get("value_min", 60)),
            value_max=int(values.get("value_max", 255)),
        )

    morphology = raw.get("morphology", {})
    return DetectionConfig(
        width=int(raw.get("width", 640)),
        height=int(raw.get("height", 480)),
        min_area=int(raw.get("min_area", 50)),
        max_area=int(raw.get("max_area", 5000)),
        min_confidence=float(raw.get("min_confidence", 0.0)),
        min_circularity=float(raw.get("min_circularity", 0.0)),
        min_color_contrast=float(raw.get("min_color_contrast", 0.0)),
        color_samples_bgr=raw.get("color_samples_bgr", {}),
        preview_color_matrix=raw.get("preview_color_matrix", []),
        color_distance_max=float(raw.get("color_distance_max", 30.0)),
        color_margin=float(raw.get("color_margin", 8.0)),
        morphology=MorphologyConfig(
            open_kernel=int(morphology.get("open_kernel", 3)),
            close_kernel=int(morphology.get("close_kernel", 5)),
        ),
        color_thresholds=threshold_map or DetectionConfig().color_thresholds,
    )


class Detector:
    """CPU-based OpenCV detector for a circular target spot on a phone display."""

    def __init__(self, config: DetectionConfig | str | Path | None = None) -> None:
        if isinstance(config, (str, Path)):
            self.config = load_detection_config(config)
        elif config is None:
            self.config = load_detection_config()
        else:
            self.config = config

        if not 0.0 <= self.config.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1.")

        if not 0 <= self.config.min_circularity <= 1:
            raise ValueError("min_circularity must be between 0 and 1.")
        for name in ('min_color_contrast', 'color_distance_max', 'color_margin'):
            value = getattr(self.config, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative.')
        if self.config.preview_color_matrix:
            matrix = np.asarray(self.config.preview_color_matrix, dtype=float)
            if matrix.shape != (3, 4) or not np.isfinite(matrix).all():
                raise ValueError('preview_color_matrix must be a finite 3x4 matrix.')
        if self.config.color_samples_bgr:
            validate_samples(self.config.color_samples_bgr)

    def build_mask(self, hsv_frame: np.ndarray, color_name: str, *,
                   morphology: bool = True) -> np.ndarray:
        """Build a binary HSV mask for a target color, using configurable thresholds."""
        if hsv_frame.ndim != 3 or hsv_frame.shape[2] != 3:
            raise ValueError("Expected an HSV image with shape (H, W, 3).")

        threshold = self.config.color_thresholds.get(color_name)
        if threshold is None:
            return np.zeros(hsv_frame.shape[:2], dtype=np.uint8)

        hue = hsv_frame[:, :, 0]
        saturation = hsv_frame[:, :, 1]
        value = hsv_frame[:, :, 2]

        mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
        for low_h, high_h in threshold.hue_ranges:
            hue_mask = (hue >= low_h) & (hue <= high_h)
            sat_mask = (saturation >= threshold.saturation_min) & (saturation <= threshold.saturation_max)
            value_mask = (value >= threshold.value_min) & (value <= threshold.value_max)
            mask |= ((hue_mask & sat_mask & value_mask)).astype(np.uint8)

        if mask.size == 0 or not morphology:
            return mask

        return self.clean_mask(mask)

    def clean_mask(self, mask):
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config.morphology.open_kernel, self.config.morphology.open_kernel))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config.morphology.close_kernel, self.config.morphology.close_kernel))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        return mask

    def _measure_candidate(self, contour: np.ndarray, mask: np.ndarray,
                           hsv_frame: np.ndarray, color_name: str,
                           features: np.ndarray | None = None) -> DetectionResult | None:
        area = cv2.contourArea(contour)
        if not self.config.min_area <= area <= self.config.max_area:
            return None
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            return None

        centroid_x = moments["m10"] / moments["m00"]
        centroid_y = moments["m01"] / moments["m00"]
        normalized_x = centroid_x / max(1, hsv_frame.shape[1])
        normalized_y = centroid_y / max(1, hsv_frame.shape[0])
        radius = float(np.sqrt(area / np.pi))

        # Measure only this contour's local pixels, avoiding a full-frame mask per spot.
        x, y, w, h = cv2.boundingRect(contour)
        selected_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(selected_mask, [contour], -1, 255, cv2.FILLED, offset=(-x, -y))
        selected_pixels = (selected_mask > 0) & (mask[y:y+h, x:x+w] > 0)
        hsv_roi = hsv_frame[y:y+h, x:x+w]
        if not selected_pixels.any():
            return None
        mean_brightness = float(hsv_roi[:, :, 2][selected_pixels].mean())
        mean_saturation = float(hsv_roi[:, :, 1][selected_pixels].mean())
        perimeter = cv2.arcLength(contour, True)
        circularity = min(1.0, 4.0 * np.pi * area / max(perimeter ** 2, 1.0))
        # Heuristic quality score, not a calibrated probability.
        confidence = float(circularity * mean_saturation / 255.0 * mean_brightness / 255.0)

        contrast = 0.0
        if features is not None:
            # Compare the filled blob with a nearby ring, ignoring brightness dominance.
            padding = max(3, min(12, int(radius * 0.4)))
            x0, y0 = max(0, x-padding), max(0, y-padding)
            x1, y1 = min(mask.shape[1], x+w+padding), min(mask.shape[0], y+h+padding)
            region = np.zeros((y1-y0, x1-x0), np.uint8)
            cv2.drawContours(region, [contour], -1, 1, cv2.FILLED, offset=(-x0, -y0))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*padding+1, 2*padding+1))
            ring = (cv2.dilate(region, kernel) > 0) & (region == 0)
            local = features[y0:y1, x0:x1]
            if ring.any():
                contrast = float(np.linalg.norm(np.median(local[region > 0], axis=0)
                                                - np.median(local[ring], axis=0)))

        return DetectionResult(
            valid=True,
            circularity=float(circularity),
            color_contrast=contrast,
            confidence=confidence,
            detected_color=color_name,
            centroid_x=float(centroid_x),
            centroid_y=float(centroid_y),
            centroid_norm_x=float(normalized_x),
            centroid_norm_y=float(normalized_y),
            area=float(area),
            radius=radius,
            mean_brightness=mean_brightness,
            width=hsv_frame.shape[1],
            height=hsv_frame.shape[0],
        )

    def rejection_reason(self, candidate):
        if candidate is None:
            return 'degenerate'
        if candidate.circularity < self.config.min_circularity:
            return 'low_circularity'
        if candidate.color_contrast < self.config.min_color_contrast:
            return 'low_contrast'
        if candidate.confidence < self.config.min_confidence:
            return 'low_confidence'
        return 'accepted'

    def frame_masks(self, frame_bgr, hsv_frame):
        if self.config.color_samples_bgr:
            raw = calibrated_masks(frame_bgr, self.config.color_samples_bgr,
                                   self.config.color_distance_max, self.config.color_margin)
        else:
            raw = {color: self.build_mask(hsv_frame, color, morphology=False)
                   for color in ('red', 'green', 'blue')}
        cleaned = {}
        claimed = np.zeros(frame_bgr.shape[:2], dtype=bool)
        for color, mask in raw.items():
            mask = self.clean_mask(mask)
            mask[claimed] = 0
            claimed |= mask > 0
            cleaned[color] = mask
        return raw, cleaned

    def detect_all(self, frame_bgr: np.ndarray, fps: float = 0.0) -> list[DetectionResult]:
        """Return every valid blob, sorted by descending area. No target yields []."""
        if frame_bgr is None:
            raise ValueError("frame_bgr cannot be None.")
        if frame_bgr.dtype != np.uint8 or frame_bgr.size == 0:
            raise ValueError("Expected a nonempty uint8 BGR frame.")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("Expected a BGR frame with shape (H, W, 3).")

        hsv_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        _, masks = self.frame_masks(frame_bgr, hsv_frame)
        return self.detect_from_masks(frame_bgr, hsv_frame, masks, fps)

    def detect_from_masks(self, frame_bgr, hsv_frame, masks, fps=0.0):
        """Shared CPU contour measurement for CPU and CUDA-generated masks."""
        features = color_features(frame_bgr) if self.config.min_color_contrast > 0 else None
        results = []
        for color_name, mask in masks.items():
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                candidate = self._measure_candidate(contour, mask, hsv_frame, color_name, features)
                if self.rejection_reason(candidate) == 'accepted':
                    candidate.fps = fps
                    results.append(candidate)
        return sorted(results, key=lambda result: (-result.area, result.detected_color,
                                                   result.centroid_y, result.centroid_x))

    def detect(self, frame_bgr: np.ndarray, fps: float = 0.0) -> DetectionResult:
        """Compatibility API: return only the largest valid blob."""
        results = self.detect_all(frame_bgr, fps)
        return results[0] if results else DetectionResult(
            width=frame_bgr.shape[1], height=frame_bgr.shape[0], fps=fps)
