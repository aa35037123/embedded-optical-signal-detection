import numpy as np
import pytest

from src.detection.detector import Detector


def _make_frame(color_bgr: tuple[int, int, int], cx: int, cy: int, radius: int, width: int = 640, height: int = 480) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:height, :width]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    frame[mask] = color_bgr
    return frame


def test_detects_red_blob() -> None:
    detector = Detector()
    frame = _make_frame((0, 0, 255), 200, 180, 25)

    result = detector.detect(frame, fps=30.0)

    assert result.valid is True
    assert result.detected_color == "red"
    assert result.centroid_x == pytest.approx(200, abs=3)
    assert result.centroid_y == pytest.approx(180, abs=3)
    assert result.area > 0
    assert result.radius > 0


def test_detects_blue_blob_with_normalized_coordinates() -> None:
    detector = Detector()
    frame = _make_frame((255, 0, 0), 320, 240, 20)

    result = detector.detect(frame, fps=12.5)

    assert result.valid is True
    assert result.detected_color == "blue"
    assert result.centroid_norm_x == pytest.approx(0.5, abs=0.05)
    assert result.centroid_norm_y == pytest.approx(0.5, abs=0.05)


def test_returns_invalid_when_no_blob_found() -> None:
    detector = Detector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = detector.detect(frame)

    assert result.valid is False
    assert result.detected_color == "none"


@pytest.mark.parametrize('bgr,color', [((0, 255, 0), 'green'), ((0, 0, 255), 'red'), ((255, 0, 0), 'blue')])
def test_colors_and_confidence(bgr, color):
    result = Detector().detect(_make_frame(bgr, 120, 100, 20))
    assert result.detected_color == color
    assert 0.8 < result.confidence <= 1


def test_largest_valid_across_colors():
    frame = _make_frame((0, 255, 0), 100, 100, 20)
    frame += _make_frame((255, 0, 0), 300, 100, 30)
    frame += _make_frame((0, 0, 255), 500, 100, 70)  # too large
    assert Detector().detect(frame).detected_color == 'blue'


def test_brightness_only_uses_selected_blob():
    frame = _make_frame((0, 0, 255), 100, 100, 25)
    frame += _make_frame((0, 0, 80), 300, 100, 15)
    result = Detector().detect(frame)
    assert result.mean_brightness == 255


def test_rejects_dark_white_and_tiny_targets():
    frame = _make_frame((0, 0, 40), 100, 100, 20)
    frame += _make_frame((255, 255, 255), 300, 100, 20)
    frame += _make_frame((0, 255, 0), 500, 100, 2)
    assert not Detector().detect(frame).valid


def test_upper_red_hue():
    import cv2
    hsv = _make_frame((175, 255, 255), 100, 100, 20)
    assert Detector().detect(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)).detected_color == 'red'


def test_missing_explicit_config_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        Detector(tmp_path / 'missing.yaml')


def test_all_blobs_including_repeated_colors():
    frame = _make_frame((0, 0, 255), 100, 120, 25)
    frame += _make_frame((0, 0, 100), 250, 120, 15)
    frame += _make_frame((0, 255, 0), 400, 120, 20)
    frame += _make_frame((255, 0, 0), 540, 120, 18)
    results = Detector().detect_all(frame, fps=24)
    assert len(results) == 4
    by_x = sorted(results, key=lambda result: result.centroid_x)
    assert [r.detected_color for r in by_x] == ['red', 'red', 'green', 'blue']
    assert [r.centroid_x for r in by_x] == pytest.approx([100, 250, 400, 540], abs=1)
    assert [r.mean_brightness for r in by_x] == pytest.approx([255, 100, 255, 255])
    assert all(r.fps == 24 and 0 < r.confidence <= 1 for r in results)
    assert [r.area for r in results] == sorted([r.area for r in results], reverse=True)
    assert Detector().detect(frame).centroid_x == results[0].centroid_x


def test_all_empty_and_filtering():
    detector = Detector()
    assert detector.detect_all(np.zeros((480, 640, 3), np.uint8)) == []
    frame = _make_frame((0, 0, 255), 100, 120, 70)  # oversized
    frame += _make_frame((0, 255, 0), 250, 120, 2)  # undersized
    frame += _make_frame((255, 0, 0), 400, 120, 20)
    results = detector.detect_all(frame)
    assert len(results) == 1 and results[0].detected_color == 'blue'


def test_overlapping_color_ranges_do_not_duplicate_blob():
    import cv2
    frame = cv2.cvtColor(_make_frame((90, 255, 255), 100, 100, 20), cv2.COLOR_HSV2BGR)
    assert len(Detector().detect_all(frame)) == 1
