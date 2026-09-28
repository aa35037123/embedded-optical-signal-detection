from __future__ import annotations

import cv2
import numpy as np

from src.detection.types import DetectionResult


def draw_detection(frame_bgr: np.ndarray, detection: DetectionResult, fps: float | None = None) -> np.ndarray:
    """Render candidate geometry and status text onto a BGR frame."""
    overlay = frame_bgr.copy()
    actual_fps = detection.fps if fps is None else fps
    cv2.putText(overlay, f"FPS: {actual_fps:.1f}", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if detection.valid is False or detection.detected_color == "none":
        cv2.putText(overlay, "NO TARGET", (16, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return overlay

    _draw_target(overlay, detection)
    return overlay


def draw_detections(frame_bgr: np.ndarray, detections: list[DetectionResult],
                    fps: float = 0.0) -> np.ndarray:
    """Draw all valid targets, copying the frame only once."""
    overlay = frame_bgr.copy()
    targets = [result for result in detections if result.valid]
    for index, target in enumerate(targets, 1):
        _draw_target(overlay, target, index)
    cv2.putText(overlay, f"FPS: {fps:.1f}  Targets: {len(targets)}", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if not targets:
        cv2.putText(overlay, "NO TARGET", (16, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return overlay


def _draw_target(overlay: np.ndarray, detection: DetectionResult, index: int | None = None) -> None:
    color_map = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
    }
    color = color_map.get(detection.detected_color, (255, 255, 255))

    cx = int(round(detection.centroid_x))
    cy = int(round(detection.centroid_y))
    radius = max(4, int(round(detection.radius)))

    cv2.circle(overlay, (cx, cy), radius, color, thickness=2)
    cv2.circle(overlay, (cx, cy), 4, (255, 255, 255), thickness=-1)

    line_1 = f"{detection.detected_color.upper()}"
    if index is not None:
        line_1 = f"#{index} {line_1}"
    line_2 = f"({detection.centroid_x:.1f}, {detection.centroid_y:.1f})"
    line_3 = f"r={detection.radius:.1f}px conf={detection.confidence:.2f}"

    # Keep labels inside the image even when the target is near an edge.
    text_width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
                     for line in (line_1, line_2, line_3))
    text_x = max(0, min(cx + radius + 8, overlay.shape[1] - text_width - 4))
    text_y = max(16, min(cy - 12, overlay.shape[0] - 48))
    for index, line in enumerate((line_1, line_2, line_3)):
        cv2.putText(overlay, line, (text_x, text_y + index * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color if index == 0 else (255, 255, 255), 1)
