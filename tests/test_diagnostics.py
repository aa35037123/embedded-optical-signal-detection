import json

import cv2
import numpy as np
import pytest

from src.detection import Detector
from src.detection.diagnostics import inspect_frame, probe_pixel
from src.pipeline import parse_args, run


@pytest.mark.parametrize('hue', [0, 12, 168, 179])
def test_red_hue_boundaries(hue):
    hsv = np.zeros((150, 200, 3), np.uint8)
    cv2.circle(hsv, (80, 70), 20, (hue, 255, 200), -1)
    results = Detector().detect_all(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
    assert len(results) == 1 and results[0].detected_color == 'red'


def test_red_rejection_reasons():
    frame = np.zeros((200, 600, 3), np.uint8)
    cv2.circle(frame, (80, 100), 60, (0, 0, 255), -1)  # too large
    cv2.circle(frame, (200, 100), 3, (0, 0, 255), -1)  # too small
    cv2.circle(frame, (300, 100), 20, (0, 0, 255), -1)  # accepted
    cv2.circle(frame, (400, 100), 20, (0, 0, 40), -1)  # too dark
    frame[100, 500] = (0, 0, 255)  # isolated pixel removed by opening
    report, masks = inspect_frame(Detector(), frame)
    assert {c['status'] for c in report['colors']['red']['candidates']} == {'accepted', 'too_small', 'too_large'}
    assert masks['red-threshold'][100, 500] == 255
    assert masks['red-morphology'][100, 500] == 0
    assert 'brightness outside' in probe_pixel(Detector(), frame, 400, 100)
    assert 'passes red HSV' in probe_pixel(Detector(), frame, 300, 100)
    assert 'hue outside' in probe_pixel(Detector(), np.array([[[255, 0, 0]]], np.uint8), 0, 0)


def test_debug_capture_from_pipeline(tmp_path):
    frame = np.zeros((150, 200, 3), np.uint8)
    cv2.circle(frame, (80, 70), 20, (0, 0, 255), -1)
    source = tmp_path / 'source.png'
    cv2.imwrite(str(source), frame)
    output = tmp_path / 'debug'
    assert run(parse_args(['--image', str(source), '--debug-dir', str(output)])) == 1
    np.testing.assert_array_equal(cv2.imread(str(output / 'frame.png')), frame)
    report = json.loads((output / 'report.json').read_text())
    assert report['colors']['red']['candidates'][0]['status'] == 'accepted'
    assert (output / 'red-threshold.png').exists()
    assert (output / 'red-morphology.png').exists()
