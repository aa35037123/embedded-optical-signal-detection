from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.network.sources import frames


def args(**kwargs):
    return SimpleNamespace(image=None, video=None, width=64, height=48, fps=30, **kwargs)


def test_camera_source_cleanup(monkeypatch):
    from src.capture import camera
    calls = []
    class FakeCamera:
        def __init__(self, width, height, fps):
            calls.append((width, height, fps))
        def read(self):
            return np.zeros((48, 64, 3), np.uint8)
        def stop(self):
            calls.append('stop')
    monkeypatch.setattr(camera, 'CameraAcquisition', FakeCamera)
    source = frames(args())
    assert next(source).shape == (48, 64, 3)
    source.close()
    assert calls == [(64, 48, 30), 'stop']


def test_camera_capture_error_cleanup(monkeypatch):
    from src.capture import camera
    closed = []
    class FakeCamera:
        def __init__(self, *args): pass
        def read(self): raise RuntimeError('capture failed')
        def stop(self): closed.append(True)
    monkeypatch.setattr(camera, 'CameraAcquisition', FakeCamera)
    with pytest.raises(RuntimeError):
        next(frames(args()))
    assert closed == [True]


def test_video_source_eof(tmp_path):
    path = tmp_path / 'test.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 30, (64,48))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((48,64,3),np.uint8))
    writer.release()
    settings = args()
    settings.video = path
    assert len(list(frames(settings))) == 3
