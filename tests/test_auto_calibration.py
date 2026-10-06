from copy import deepcopy

import cv2
import numpy as np
import pytest

from src.detection import Detector, DetectionConfig
from src.detection.auto_calibration import (
    chart_samples, fit_calibration, save_auto_calibration, corrected_preview,
)
from src.pipeline import parse_args


def tinted_chart():
    ideal = np.zeros((400,600,3), np.uint8)
    for index, bgr in enumerate([(0,0,255),(0,255,0),(255,0,0),(0,0,0),(128,128,128),(255,255,255)]):
        y,x = (index//3)*200,(index%3)*200
        ideal[y:y+200,x:x+200] = bgr
    observed = (ideal.astype(float)*[.6,.7,.5] + [40,10,70]).astype(np.uint8)
    return ideal, observed


def test_chart_sampling_and_correction(tmp_path):
    ideal, frame = tinted_chart()
    samples = chart_samples(frame)
    path = tmp_path / 'auto.yaml'
    assert save_auto_calibration(path, DetectionConfig(), [samples]*10) == ''
    detector = Detector(path)
    assert detector.config.color_samples_bgr['red'] == [[40,10,197]]
    corrected = corrected_preview(frame, detector.config.preview_color_matrix)
    assert np.abs(corrected.astype(float)-ideal).max() < 3
    # Actual colored dots on the same tinted background are classified correctly.
    dots = np.full((480,640,3), (40,10,70), np.uint8)
    for color, x in [('red',100),('green',300),('blue',500)]:
        cv2.circle(dots,(x,240),20,tuple(map(int,samples[color])),-1)
    assert sorted(r.detected_color for r in detector.detect_all(dots)) == ['blue','green','red']


def test_perspective_chart():
    _, frame = tinted_chart()
    corners = np.float32([[40,20],[660,60],[620,460],[20,430]])
    transform = cv2.getPerspectiveTransform(np.float32([[0,0],[600,0],[600,400],[0,400]]),corners)
    camera = cv2.warpPerspective(frame,transform,(700,500))
    expected = chart_samples(frame,[(0,0),(600,0),(600,400),(0,400)])
    assert chart_samples(camera,corners) == expected


def test_unstable_calibration_preserves_previous_file(tmp_path):
    _, frame = tinted_chart()
    row = chart_samples(frame, [(0,0),(600,0),(600,400),(0,400)])
    rows = [deepcopy(row) for _ in range(10)]
    for changed in rows[5:]:
        changed['red'] = [100,100,100]
    path = tmp_path / 'auto.yaml'
    path.write_text('previous calibration')
    with pytest.raises(ValueError,match='changed'):
        save_auto_calibration(path,DetectionConfig(),rows)
    assert path.read_text() == 'previous calibration'


def test_bad_neutral_reference_disables_preview_correction():
    _, frame = tinted_chart()
    row = chart_samples(frame,[(0,0),(600,0),(600,400),(0,400)])
    row['white'] = [40,10,70]
    samples,matrix,warning = fit_calibration([row]*5)
    assert samples and not matrix and warning


def test_auto_calibration_cli_constraints():
    assert parse_args(['--auto-calibrate-colors','/tmp/auto.yaml']).calibration_frames == 20
    with pytest.raises(SystemExit):
        parse_args(['--auto-calibrate-colors','/tmp/auto.yaml','--demo'])


def test_auto_capture_workflow(monkeypatch, tmp_path):
    from src.capture import camera
    from src.detection.auto_calibration import run_auto_calibration
    _, frame = tinted_chart()
    stopped = []
    class FakeCamera:
        def __init__(self, *args): pass
        def start(self): pass
        def read(self): return frame.copy()
        def stop(self): stopped.append(True)
    monkeypatch.setattr(camera, 'CameraAcquisition', FakeCamera)
    for name in ('namedWindow','imshow','destroyAllWindows'):
        monkeypatch.setattr(cv2,name,lambda *args: None)
    monkeypatch.setattr(cv2, 'setMouseCallback',
                        lambda *args: pytest.fail('Automatic calibration must not request clicks'))
    calls = []
    monkeypatch.setattr(cv2, 'waitKey', lambda delay: calls.append(delay) or -1)
    path = tmp_path/'auto.yaml'
    args = parse_args(['--auto-calibrate-colors',str(path),'--calibration-frames','5'])
    assert run_auto_calibration(args) == 5
    assert len(calls) == 35  # 30 settling frames, then five samples.
    assert stopped == [True]
    assert Detector(path).config.preview_color_matrix
