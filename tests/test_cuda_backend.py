"""GPU accuracy tests run on an actual CUDA OpenCV installation; never mocked."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.detection.backends import CPUBackend, create_backend
from src.detection.cuda_backend import require_cuda
from src.detection.types import DetectionConfig
from src.network.receiver import parse_args

ROOT = Path(__file__).resolve().parents[1]


def test_cli_cuda():
    assert parse_args(['--backend', 'cuda']).backend == 'cuda'


def test_no_cuda_fails_explicitly(monkeypatch):
    monkeypatch.setattr(cv2.cuda, 'getCudaEnabledDeviceCount', lambda: 0)
    with pytest.raises(RuntimeError, match='No usable'):
        create_backend('cuda')


def test_calibrated_config_not_silently_ignored():
    config = DetectionConfig(color_samples_bgr={
        'red': [[0,0,255]], 'green': [[0,255,0]],
        'blue': [[255,0,0]], 'background': [[0,0,0]]})
    with pytest.raises(ValueError, match='HSV'):
        create_backend('cuda', config)


@pytest.fixture
def gpu():
    try:
        require_cuda()
    except RuntimeError as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize('fixture', [None, 'red_cast_phone.png', 'red_cast_phone_blue.png'])
def test_gpu_cpu_detection_parity(gpu, fixture):
    config = ROOT / 'configs' / ('detection-red-background.yaml' if fixture else 'detection-noir.yaml')
    if fixture:
        frame = cv2.imread(str(ROOT / 'tests/fixtures' / fixture))
    else:
        frame = np.zeros((480,640,3), np.uint8)
        for x, color in [(100,(0,0,255)), (300,(0,255,0)), (500,(255,0,0))]:
            cv2.circle(frame, (x,240), 20, color, -1)
        # Border target and speckle exercise morphology.
        cv2.circle(frame, (0,100), 15, (0,0,255), -1)
        frame[20,20] = (0,0,255)
    cuda = create_backend('cuda', config)
    for current in (frame, cv2.resize(frame, (320,240)), frame):
        expected = CPUBackend(config).process(current).detections
        actual = cuda.process(current)
        assert actual.backend == 'cuda'
        assert actual.upload_ms >= 0 and actual.download_ms >= 0
        order = lambda d: (d.detected_color, d.centroid_x, d.centroid_y)
        assert len(actual.detections) == len(expected)
        for a, b in zip(sorted(actual.detections, key=order), sorted(expected, key=order)):
            assert a.detected_color == b.detected_color
            assert a.centroid_x == pytest.approx(b.centroid_x, abs=2)
            assert a.centroid_y == pytest.approx(b.centroid_y, abs=2)
            assert a.area == pytest.approx(b.area, rel=.08, abs=5)
