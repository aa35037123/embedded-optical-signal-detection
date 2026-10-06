import sys
from types import SimpleNamespace

import pytest

from src.capture.camera import CameraAcquisition
from src.pipeline import parse_args
from src.network.sender import parse_args as sender_args


def test_tuning_loaded_before_camera_open(monkeypatch, tmp_path):
    calls = []
    class FakePicamera:
        @staticmethod
        def load_tuning_file(path):
            calls.append(('load', path))
            return {'test': True}
        def __init__(self, **kwargs):
            calls.append(('open', kwargs))
    monkeypatch.setitem(sys.modules, 'picamera2', SimpleNamespace(Picamera2=FakePicamera))
    path = tmp_path / 'noir.json'
    path.write_text('{}')
    CameraAcquisition(tuning_file=path)
    assert calls == [('load', str(path)), ('open', {'tuning': {'test': True}})]
    calls.clear()
    with pytest.raises(FileNotFoundError):
        CameraAcquisition(tuning_file=tmp_path / 'missing.json')
    assert not calls
    CameraAcquisition()
    assert calls == [('open', {})]


def test_camera_only_cli():
    assert str(parse_args(['--tuning-file', '/tmp/noir.json']).tuning_file) == '/tmp/noir.json'
    assert sender_args(['--host', 'localhost', '--tuning-file', '/tmp/noir.json']).tuning_file
    with pytest.raises(SystemExit):
        parse_args(['--demo', '--tuning-file', '/tmp/noir.json'])
    with pytest.raises(SystemExit):
        sender_args(['--host', 'localhost', '--image', 'a.png', '--tuning-file', '/tmp/noir.json'])


def test_sender_passes_tuning(monkeypatch, tmp_path):
    from src.capture import camera
    from src.network.sources import frames
    calls = []
    class FakeCamera:
        def __init__(self, width, height, fps, **kwargs):
            calls.append(kwargs)
        def read(self):
            return 'frame'
        def stop(self):
            calls.append('closed')
    monkeypatch.setattr(camera, 'CameraAcquisition', FakeCamera)
    args = sender_args(['--host', 'localhost', '--tuning-file', str(tmp_path / 'noir.json')])
    source = frames(args)
    assert next(source) == 'frame'
    source.close()
    assert calls == [{'tuning_file': args.tuning_file}, 'closed']
