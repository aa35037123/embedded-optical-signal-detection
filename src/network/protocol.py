"""Version 1 JPEG framing, independent of OpenCV, Picamera2, and detection.

Wire order is big-endian (network byte order), with no native padding:
    magic[4], version:u8, flags:u8, header_size:u16,
    frame_id:u64, capture_timestamp_ns:u64, jpeg_encode_ns:u64,
    width:u16, height:u16, jpeg_size:u32, then exactly jpeg_size payload bytes.

A protocol error, timeout, or truncated packet invalidates the connection; callers
must close it instead of attempting to find magic bytes inside a JPEG. Set socket
timeouts in the application. This module never changes socket ownership/options.
"""
from __future__ import annotations

from dataclasses import dataclass
import socket
import struct

MAGIC = b'OSIG'
VERSION = 1
HEADER_STRUCT = struct.Struct('!4sBBHQQQHHI')
HEADER_SIZE = HEADER_STRUCT.size  # 40 bytes
MAX_PAYLOAD_SIZE = 8 * 1024 * 1024
MAX_DIMENSION = 8192
MAX_PIXELS = 16 * 1024 * 1024
UINT64_MAX = (1 << 64) - 1


class ProtocolError(ValueError):
    """A header, packet, or frame sequence violates the stream contract."""


class ConnectionClosed(EOFError):
    """EOF before recv_exact could complete, including EOF at a packet boundary."""

    def __init__(self, expected: int, received: int) -> None:
        self.phase = None
        self.expected = expected
        self.received = received
        super().__init__(f'Connection closed after {received} of {expected} expected bytes')


@dataclass(frozen=True)
class FrameHeader:
    frame_id: int
    capture_timestamp_ns: int
    width: int
    height: int
    jpeg_size: int
    jpeg_encode_ns: int = 0


@dataclass(frozen=True)
class FramePacket:
    header: FrameHeader
    jpeg_payload: bytes


def _integer(name: str, value: int, minimum: int, maximum: int) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ProtocolError(f'{name} must be an integer in [{minimum}, {maximum}]')


def validate_header(header: FrameHeader) -> None:
    """Bound metadata before allocating/receiving the advertised JPEG payload."""
    _integer('frame_id', header.frame_id, 0, UINT64_MAX)
    _integer('capture_timestamp_ns', header.capture_timestamp_ns, 0, UINT64_MAX)
    _integer('jpeg_encode_ns', header.jpeg_encode_ns, 0, UINT64_MAX)
    _integer('width', header.width, 1, MAX_DIMENSION)
    _integer('height', header.height, 1, MAX_DIMENSION)
    if header.width * header.height > MAX_PIXELS:
        raise ProtocolError(f'Image dimensions exceed {MAX_PIXELS} pixels')
    _integer('jpeg_size', header.jpeg_size, 1, MAX_PAYLOAD_SIZE)


def pack_header(header: FrameHeader) -> bytes:
    validate_header(header)
    return HEADER_STRUCT.pack(MAGIC, VERSION, 0, HEADER_SIZE,
                              header.frame_id, header.capture_timestamp_ns,
                              header.jpeg_encode_ns, header.width, header.height,
                              header.jpeg_size)


def unpack_header(data: bytes) -> FrameHeader:
    if len(data) != HEADER_SIZE:
        raise ProtocolError(f'Header must contain exactly {HEADER_SIZE} bytes')
    magic, version, flags, size, frame_id, timestamp, encode_ns, width, height, payload_size = HEADER_STRUCT.unpack(data)
    if magic != MAGIC:
        raise ProtocolError('Invalid protocol magic')
    if version != VERSION:
        raise ProtocolError(f'Unsupported protocol version: {version}')
    if flags != 0:
        raise ProtocolError(f'Unsupported protocol flags: {flags}')
    if size != HEADER_SIZE:
        raise ProtocolError(f'Invalid header size: {size}')
    header = FrameHeader(frame_id=frame_id, capture_timestamp_ns=timestamp,
                         width=width, height=height, jpeg_size=payload_size,
                         jpeg_encode_ns=encode_ns)
    validate_header(header)
    return header


def recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    """Receive exactly num_bytes; tolerate fragmented/coalesced TCP delivery.

    Zero bytes returns b''. EOF raises ConnectionClosed. Socket errors/timeouts
    propagate unchanged. Callers must bound num_bytes before calling this helper.
    """
    if type(num_bytes) is not int or num_bytes < 0:
        raise ValueError('num_bytes must be a nonnegative integer')
    buffer = bytearray(num_bytes)
    received = 0
    while received < num_bytes:
        chunk = sock.recv(min(num_bytes - received, 65536))
        if not chunk:
            raise ConnectionClosed(num_bytes, received)
        buffer[received:received + len(chunk)] = chunk
        received += len(chunk)
    return bytes(buffer)


@dataclass
class FrameSequence:
    """Track wire-order IDs for ONE connection; gaps are counts, not errors.

    First ID may be any uint64. Equal/decreasing IDs are invalid. Start a new
    tracker after reconnect. This measures gaps in received IDs, not frames later
    overwritten by a latest-frame queue (which must be counted separately).
    """
    last_frame_id: int | None = None
    skipped_frames: int = 0

    def observe(self, frame_id: int) -> int:
        _integer('frame_id', frame_id, 0, UINT64_MAX)
        gap = 0
        if self.last_frame_id is not None:
            if frame_id <= self.last_frame_id:
                raise ProtocolError(f'Frame ID {frame_id} is not greater than {self.last_frame_id}')
            gap = frame_id - self.last_frame_id - 1
        self.last_frame_id = frame_id
        self.skipped_frames += gap
        return gap


def send_packet(sock: socket.socket, header: FrameHeader, jpeg_payload: bytes) -> None:
    """Send a validated header and payload using sendall; no giant concatenation.

    The caller must ensure only one writer uses this socket at a time. A send
    failure may leave a partial packet on the stream, so close the connection.
    """
    packed = pack_header(header)
    if len(jpeg_payload) != header.jpeg_size:
        raise ProtocolError('JPEG payload length does not match the header')
    sock.sendall(packed)
    sock.sendall(jpeg_payload)


def receive_packet(sock: socket.socket, sequence: FrameSequence | None = None) -> FramePacket:
    """Read one bounded packet, updating sequence only after the whole payload.

    This is transport validation only. The receiver must additionally validate
    JPEG decoding and actual image dimensions before handing pixels to detection.
    """
    try:
        header = unpack_header(recv_exact(sock, HEADER_SIZE))
    except ConnectionClosed as exc:
        exc.phase = 'header'
        raise
    try:
        payload = recv_exact(sock, header.jpeg_size)
    except ConnectionClosed as exc:
        exc.phase = 'payload'
        raise
    if sequence is not None:
        sequence.observe(header.frame_id)
    return FramePacket(header, payload)
