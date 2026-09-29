"""JPEG conversion and validation, separate from framing and detection."""
from dataclasses import dataclass
import time

import cv2
import numpy as np

from .protocol import FrameHeader, FramePacket, ProtocolError, validate_header


@dataclass(frozen=True)
class ReceivedFrame:
    packet: FramePacket
    frame: np.ndarray
    receive_timestamp_ns: int
    decode_ms: float


def encode_frame(frame, frame_id, capture_timestamp_ns, quality=85):
    if not 1 <= quality <= 100:
        raise ValueError('JPEG quality must be in [1, 100]')
    if frame is None or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError('Expected a uint8 BGR frame')
    height, width = frame.shape[:2]
    validate_header(FrameHeader(frame_id, capture_timestamp_ns, width, height, 1))
    started = time.monotonic_ns()
    ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError('JPEG encoding failed')
    payload = encoded.tobytes()
    elapsed = time.monotonic_ns() - started
    header = FrameHeader(frame_id, capture_timestamp_ns, width, height, len(payload), elapsed)
    validate_header(header)
    return FramePacket(header, payload)


def jpeg_dimensions(payload):
    """Read bounded baseline/progressive JPEG dimensions before decoder allocation."""
    if not payload.startswith(b'\xff\xd8') or not payload.endswith(b'\xff\xd9'):
        raise ProtocolError('Missing JPEG start/end markers')
    index = 2
    while index < len(payload) - 2:
        if payload[index] != 255:
            raise ProtocolError('Invalid JPEG marker')
        while index < len(payload) and payload[index] == 255:
            index += 1
        if index >= len(payload):
            break
        marker = payload[index]
        index += 1
        if marker in (0xDA, 0xD9):
            break
        if index + 2 > len(payload):
            break
        length = int.from_bytes(payload[index:index+2], 'big')
        if length < 2 or index + length > len(payload):
            raise ProtocolError('Invalid JPEG segment length')
        if marker in (0xC0, 0xC2):
            if length < 8 or payload[index+2] != 8:
                raise ProtocolError('Expected 8-bit JPEG')
            height = int.from_bytes(payload[index+3:index+5], 'big')
            width = int.from_bytes(payload[index+5:index+7], 'big')
            return width, height
        index += length
    raise ProtocolError('Missing supported JPEG dimensions')


def decode_packet(packet, receive_timestamp_ns=None):
    started = time.monotonic_ns()
    timestamp = started if receive_timestamp_ns is None else receive_timestamp_ns
    validate_header(packet.header)
    if len(packet.jpeg_payload) != packet.header.jpeg_size:
        raise ProtocolError('JPEG payload length mismatch')
    if jpeg_dimensions(packet.jpeg_payload) != (packet.header.width, packet.header.height):
        raise ProtocolError('JPEG dimensions do not match header')
    frame = cv2.imdecode(np.frombuffer(packet.jpeg_payload, np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.shape != (packet.header.height, packet.header.width, 3):
        raise ProtocolError('JPEG decoding failed or dimensions do not match')
    return ReceivedFrame(packet, frame, timestamp, (time.monotonic_ns() - started) / 1e6)
