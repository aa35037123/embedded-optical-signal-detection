from dataclasses import asdict
import cv2
import numpy as np
import pytest

from src.detection.backends import CPUBackend, create_backend
from src.detection import Detector


def test_cpu_backend_reuses_detector():
    frame = np.zeros((480, 640, 3), np.uint8)
    for x, color in [(100, (0,0,255)), (300, (0,255,0)), (500, (255,0,0))]:
        cv2.circle(frame, (x,240), 20, color, -1)
    result = CPUBackend().process(frame)
    assert [asdict(d) for d in result.detections] == [asdict(d) for d in Detector().detect_all(frame)]
    assert result.backend == 'cpu' and result.processing_ms > 0
    assert result.upload_ms == result.download_ms == 0
    np.testing.assert_array_equal(frame[240,100], [0,0,255])


def test_unknown_backend():
    with pytest.raises(ValueError):
        create_backend('unknown')
