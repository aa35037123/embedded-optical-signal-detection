import json

import cv2
import numpy as np

from src.capture.camera import CameraAcquisition
from src.detection.types import DetectionResult
from src.pipeline import demo_frame, parse_args, run
from src.visualization import draw_detection


def test_image_pipeline(tmp_path):
    source, output, log = (tmp_path / name for name in ('input.png', 'output.png', 'results.jsonl'))
    cv2.imwrite(str(source), demo_frame(0, 640, 480))
    assert run(parse_args(['--image', str(source), '--output', str(output), '--jsonl', str(log)])) == 1
    result = json.loads(log.read_text())
    assert result['detected_color'] == 'red'
    assert result['centroid_x'] == 320
    assert result['confidence'] > 0.8
    assert cv2.imread(str(output)) is not None


def test_video_pipeline_eof(tmp_path):
    source = tmp_path / 'input.avi'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 30, (640, 480))
    assert writer.isOpened()
    for index in (0, 30, 60):
        writer.write(demo_frame(index, 640, 480))
    writer.release()
    log = tmp_path / 'log.jsonl'
    assert run(parse_args(['--video', str(source), '--jsonl', str(log)])) == 3
    assert [json.loads(line)['detected_color'] for line in log.read_text().splitlines()] == ['red', 'green', 'blue']


def test_no_target_overlay_still_has_status():
    frame = np.zeros((480, 640, 3), np.uint8)
    assert draw_detection(frame, DetectionResult(), fps=30).any()
    assert not frame.any()


def test_picamera_rgb888_preserves_bgr():
    camera = CameraAcquisition.__new__(CameraAcquisition)
    frame = np.array([[[0, 0, 255]]], dtype=np.uint8)
    np.testing.assert_array_equal(camera._to_opencv_frame(frame), frame)


def test_multitarget_pipeline_and_empty_frame(tmp_path, capsys):
    from src.detection import Detector
    from src.visualization import draw_detections
    frame = np.zeros((480, 640, 3), np.uint8)
    targets = [(80, (0, 0, 255)), (230, (0, 0, 255)),
               (380, (0, 255, 0)), (530, (255, 0, 0))]
    for x, color in targets:
        cv2.circle(frame, (x, 200), 20, color, -1)
    overlay = draw_detections(frame, Detector().detect_all(frame), fps=30)
    for x, _ in targets:
        assert tuple(overlay[200, x]) == (255, 255, 255)
    for expected_count, image in ((4, frame), (0, np.zeros_like(frame))):
        source, log = tmp_path / 'input.png', tmp_path / 'log.jsonl'
        cv2.imwrite(str(source), image)
        run(parse_args(['--image', str(source), '--jsonl', str(log)]))
        record = json.loads(log.read_text())
        assert record['count'] == expected_count
        assert len(record['detections']) == expected_count
        assert record['valid'] == bool(expected_count)
        assert f'Targets: {expected_count}' in capsys.readouterr().out
        if expected_count:
            assert sorted(r['detected_color'] for r in record['detections']) == ['blue', 'green', 'red', 'red']
