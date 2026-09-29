import json

import cv2
import numpy as np
import pytest

from src.detection import Detector, DetectionConfig
from src.detection.calibration import save_calibration, validate_samples
from src.detection.diagnostics import inspect_frame
from src.pipeline import parse_args, run


def shifted_scene():
    # Known source colors rendered by a color-shifted camera: yellow and purple.
    samples = {'red': [[0, 0, 240]], 'green': [[0, 240, 210]],
               'blue': [[85, 0, 65]], 'background': [[90, 25, 155]]}
    rng = np.random.default_rng(14)
    frame = np.clip(np.full((300, 500, 3), samples['background'][0], dtype=np.int16)
                    + rng.integers(-8, 9, (300, 500, 3)), 0, 255).astype(np.uint8)
    for color, center in [('red', (90, 140)), ('green', (240, 140)), ('blue', (390, 140))]:
        cv2.circle(frame, center, 18, samples[color][0], -1)
    return frame, samples


def test_calibrated_shifted_rgb_and_noisy_background(tmp_path):
    frame, samples = shifted_scene()
    path = tmp_path / 'calibration.yaml'
    save_calibration(path, DetectionConfig(), samples)
    detector = Detector(path)
    assert detector.config.min_confidence == 0
    results = detector.detect_all(frame)
    assert sorted(r.detected_color for r in results) == ['blue', 'green', 'red']
    blue = next(r for r in results if r.detected_color == 'blue')
    assert blue.confidence < 0.4  # dim blue remains a valid high-contrast circular target
    assert all(r.circularity > 0.55 and r.color_contrast >= 12 for r in results)
    assert not any(r.detected_color in ('green', 'blue') for r in Detector().detect_all(frame))
    report, masks = inspect_frame(detector, frame)
    assert all(sum(c['status'] == 'accepted' for c in report['colors'][color]['candidates']) == 1
               for color in ('red', 'green', 'blue'))
    assert all(mask[20, 20] == 0 for mask in masks.values())


def test_local_contrast_rejects_threshold_fragment():
    frame = np.full((200, 200, 3), (0, 0, 179), np.uint8)
    cv2.circle(frame, (100, 100), 20, (0, 0, 181), -1)
    config = DetectionConfig(min_confidence=0, min_color_contrast=12)
    config.color_thresholds['red'].value_min = 180
    report, _ = inspect_frame(Detector(config), frame)
    assert report['colors']['red']['candidates'][0]['status'] == 'low_contrast'
    assert Detector(config).detect_all(frame) == []


def test_calibrated_shape_filter(tmp_path):
    frame, samples = shifted_scene()
    cv2.rectangle(frame, (100, 250), (300, 258), samples['red'][0], -1)
    path = tmp_path / 'calibration.yaml'
    save_calibration(path, DetectionConfig(), samples)
    detector = Detector(path)
    report, _ = inspect_frame(detector, frame)
    assert any(c['status'] == 'low_circularity' for c in report['colors']['red']['candidates'])
    assert len(detector.detect_all(frame)) == 3


def test_rejects_indistinguishable_calibration():
    _, samples = shifted_scene()
    samples['green'] = samples['background']
    with pytest.raises(ValueError, match='too similar'):
        validate_samples(samples)


def test_calibrated_pipeline(tmp_path):
    frame, samples = shifted_scene()
    config, source, log = (tmp_path / name for name in ('colors.yaml', 'frame.png', 'result.jsonl'))
    save_calibration(config, DetectionConfig(), samples)
    cv2.imwrite(str(source), frame)
    run(parse_args(['--image', str(source), '--config', str(config), '--jsonl', str(log)]))
    record = json.loads(log.read_text())
    assert record['count'] == 3


def test_calibration_click_workflow(tmp_path, monkeypatch):
    from src.detection.calibration import calibrate_frame
    frame, _ = shifted_scene()
    callback = None
    def register(name, handler):
        nonlocal callback
        callback = handler
    for name in ('namedWindow', 'imshow', 'destroyWindow'):
        monkeypatch.setattr(cv2, name, lambda *args: None)
    monkeypatch.setattr(cv2, 'setMouseCallback', register)
    def simulate_clicks(delay):
        for x, y in [(90, 140), (240, 140), (390, 140), (20, 100), (20, 200), (450, 250)]:
            callback(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
        return ord('s')
    monkeypatch.setattr(cv2, 'waitKey', simulate_clicks)
    path = tmp_path / 'saved.yaml'
    assert calibrate_frame(frame, path, DetectionConfig())
    assert len(Detector(path).detect_all(frame)) == 3
