"""Regression for missed phone dots and red bezel noise in a real camera capture."""
from pathlib import Path

import cv2
import pytest

from src.detection import Detector

ROOT = Path(__file__).resolve().parents[1]


def test_red_cast_phone_targets_without_bezel_false_positives():
    frame = cv2.imread(str(ROOT / 'tests/fixtures/red_cast_phone.png'))
    assert frame is not None
    detector = Detector(ROOT / 'configs/detection-red-background.yaml')
    results = detector.detect_all(frame)
    expected = {'red': [(168, 62), (255, 300), (479, 383)],
                'green': [(55, 375), (413, 376)]}
    assert len(results) == 5  # No extra detections along the left/bottom phone bezel.
    for color, centers in expected.items():
        detections = sorted((r for r in results if r.detected_color == color), key=lambda r: r.centroid_x)
        assert len(detections) == len(centers)
        for result, (x, y) in zip(detections, centers):
            assert result.centroid_x == pytest.approx(x, abs=3)
            assert result.centroid_y == pytest.approx(y, abs=3)
            assert result.circularity >= detector.config.min_circularity
            assert result.color_contrast >= detector.config.min_color_contrast
