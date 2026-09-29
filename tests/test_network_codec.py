from dataclasses import replace

import cv2
import numpy as np
import pytest

from src.network.codec import encode_frame, decode_packet
from src.network.protocol import FramePacket, ProtocolError


def sample_packet():
    image = np.zeros((48, 64, 3), np.uint8)
    cv2.circle(image, (30, 20), 10, (0, 0, 255), -1)
    return encode_frame(image, 7, 123, 90)


def test_jpeg_round_trip():
    packet = sample_packet()
    decoded = decode_packet(packet, 456)
    assert decoded.frame.shape == (48, 64, 3)
    assert decoded.frame[20, 30, 2] > 220
    assert packet.header.frame_id == 7 and packet.header.capture_timestamp_ns == 123
    assert packet.header.jpeg_encode_ns > 0 and decoded.decode_ms > 0
    assert decoded.receive_timestamp_ns == 456


def test_dimension_mismatch_rejected_before_decode(monkeypatch):
    packet = sample_packet()
    monkeypatch.setattr(cv2, 'imdecode', lambda *args: pytest.fail('must reject before decode'))
    with pytest.raises(ProtocolError, match='dimensions'):
        decode_packet(FramePacket(replace(packet.header, width=100), packet.jpeg_payload))


def test_invalid_jpeg_and_decode_failure(monkeypatch):
    packet = sample_packet()
    with pytest.raises(ProtocolError):
        decode_packet(FramePacket(replace(packet.header, jpeg_size=4), b'bad!'))
    monkeypatch.setattr(cv2, 'imdecode', lambda *args: None)
    with pytest.raises(ProtocolError, match='decoding'):
        decode_packet(packet)
