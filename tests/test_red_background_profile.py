"""Regression for missed phone dots and red bezel noise in a real camera capture."""
from pathlib import Path

import cv2
import pytest

from src.detection import Detector

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('filename,expected', [
    ('red_cast_phone.png', {'red': [(168, 62), (255, 300), (479, 383)],
                            'green': [(55, 375), (413, 376)]}),
    ('red_cast_phone_blue.png', {'blue': [(307, 139), (416, 402)],
                                 'green': [(264, 265)]}),
])
def test_red_cast_phone_targets_without_bezel_false_positives(filename, expected):
    frame = cv2.imread(str(ROOT / 'tests/fixtures' / filename))
    assert frame is not None
    detector = Detector(ROOT / 'configs/detection-red-background.yaml')
    results = detector.detect_all(frame)
    assert len(results) == sum(len(centers) for centers in expected.values())  # No extra detections along the left/bottom phone bezel.
    for color, centers in expected.items():
        detections = sorted((r for r in results if r.detected_color == color), key=lambda r: r.centroid_x)
        assert len(detections) == len(centers)
        for result, (x, y) in zip(detections, centers):
            assert result.centroid_x == pytest.approx(x, abs=3)
            assert result.centroid_y == pytest.approx(y, abs=3)
            assert result.circularity >= detector.config.min_circularity
            assert result.color_contrast >= detector.config.min_color_contrast
