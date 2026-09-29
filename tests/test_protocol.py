"""Protocol tests use a fragmented byte stream; no camera/GPU/network required."""
from dataclasses import replace
import socket
import struct

import pytest

from src.network.protocol import (
    MAGIC, VERSION, HEADER_SIZE, MAX_PAYLOAD_SIZE, UINT64_MAX,
    ConnectionClosed, FrameHeader, FrameSequence, ProtocolError,
    pack_header, unpack_header, recv_exact, send_packet, receive_packet,
)


class ByteStream:
    """Emulate recv boundaries and record sendall calls independently of the OS."""
    def __init__(self, data=b'', fragment_size=None):
        self.data = data
        self.fragment_size = fragment_size
        self.read_sizes = []
        self.sent = []

    def recv(self, size):
        self.read_sizes.append(size)
        if self.fragment_size is not None:
            size = min(size, self.fragment_size)
        result, self.data = self.data[:size], self.data[size:]
        return result

    def sendall(self, data):
        self.sent.append(data)


def header(**changes):
    return replace(FrameHeader(frame_id=0, capture_timestamp_ns=123456,
                               width=640, height=480, jpeg_size=4, jpeg_encode_ns=1500), **changes)


def wire(frame_id=0, payload=b'data'):
    return pack_header(header(frame_id=frame_id, jpeg_size=len(payload))) + payload


def test_header_round_trip_and_documented_byte_order():
    value = header(frame_id=1, capture_timestamp_ns=2, jpeg_encode_ns=3)
    packed = pack_header(value)
    assert HEADER_SIZE == len(packed) == 40
    assert packed.hex() == ('4f53494701000028'  # OSIG, version, flags, 40-byte header
                            '0000000000000001'  # frame ID
                            '0000000000000002'  # capture timestamp
                            '0000000000000003'  # encode duration
                            '028001e000000004')  # 640, 480, 4 payload bytes
    assert unpack_header(packed) == value


@pytest.mark.parametrize('fragment_size', [1, 2, 7, 39, 40, None])
def test_fragmented_and_coalesced_frames(fragment_size):
    stream = ByteStream(wire(10) + wire(11, b'next-frame'), fragment_size)
    sequence = FrameSequence()
    first = receive_packet(stream, sequence)
    second = receive_packet(stream, sequence)
    assert first.header.frame_id == 10 and first.jpeg_payload == b'data'
    assert second.header.frame_id == 11 and second.jpeg_payload == b'next-frame'
    assert sequence.skipped_frames == 0
    assert stream.data == b''


@pytest.mark.parametrize('offset,replacement,match', [
    (0, b'FAIL', 'magic'),
    (4, b'\x02', 'version'),
    (5, b'\x01', 'flags'),
    (6, b'\x00\x29', 'header size'),
    (32, b'\x00\x00', 'width'),
    (34, b'\x00\x00', 'height'),
    (36, struct.pack('!I', 0), 'jpeg_size'),
    (36, struct.pack('!I', MAX_PAYLOAD_SIZE + 1), 'jpeg_size'),
    (36, struct.pack('!I', (1 << 32) - 1), 'jpeg_size'),
])
def test_invalid_header_rejected_before_payload_read(offset, replacement, match):
    packed = bytearray(pack_header(header()))
    packed[offset:offset + len(replacement)] = replacement
    stream = ByteStream(bytes(packed) + b'data')
    with pytest.raises(ProtocolError, match=match):
        receive_packet(stream)
    assert stream.read_sizes == [HEADER_SIZE]
    assert stream.data == b'data'


@pytest.mark.parametrize('size', [0, 39, 41])
def test_invalid_header_length(size):
    with pytest.raises(ProtocolError, match='exactly'):
        unpack_header(bytes(size))


@pytest.mark.parametrize('changes', [
    {'frame_id': -1}, {'frame_id': UINT64_MAX + 1}, {'frame_id': True},
    {'capture_timestamp_ns': -1}, {'jpeg_encode_ns': -1},
    {'width': 8193}, {'height': 0}, {'width': 8192, 'height': 8192},
    {'jpeg_size': 0}, {'jpeg_size': MAX_PAYLOAD_SIZE + 1}, {'jpeg_size': 4.0},
])
def test_pack_validation(changes):
    with pytest.raises(ProtocolError):
        pack_header(header(**changes))


def test_integer_and_payload_boundaries():
    value = header(frame_id=UINT64_MAX, capture_timestamp_ns=UINT64_MAX,
                   jpeg_encode_ns=UINT64_MAX, jpeg_size=MAX_PAYLOAD_SIZE,
                   width=4096, height=4096)
    assert unpack_header(pack_header(value)) == value


@pytest.mark.parametrize('data,expected,received', [
    (b'', HEADER_SIZE, 0),
    (pack_header(header())[:17], HEADER_SIZE, 17),
    (pack_header(header()), 4, 0),
    (pack_header(header()) + b'ab', 4, 2),
])
def test_connection_closed_during_header_or_payload(data, expected, received):
    sequence = FrameSequence()
    with pytest.raises(ConnectionClosed) as error:
        receive_packet(ByteStream(data, fragment_size=3), sequence)
    assert error.value.expected == expected
    assert error.value.received == received
    assert sequence.last_frame_id is None


def test_frame_sequence_gaps_and_invalid_order():
    sequence = FrameSequence()
    assert sequence.observe(50) == 0  # no assumption about first ID
    assert sequence.observe(51) == 0
    assert sequence.observe(55) == 3
    assert sequence.skipped_frames == 3
    for invalid in (55, 54):
        with pytest.raises(ProtocolError, match='not greater'):
            sequence.observe(invalid)
    assert sequence.last_frame_id == 55 and sequence.skipped_frames == 3
    assert FrameSequence().observe(0) == 0  # reconnect creates a new tracker


def test_receive_sequence_validation():
    sequence = FrameSequence()
    stream = ByteStream(wire(1) + wire(4) + wire(4), fragment_size=5)
    receive_packet(stream, sequence)
    receive_packet(stream, sequence)
    assert sequence.skipped_frames == 2
    with pytest.raises(ProtocolError, match='not greater'):
        receive_packet(stream, sequence)


def test_sendall_and_length_validation():
    stream = ByteStream()
    send_packet(stream, header(), b'data')
    assert stream.sent == [pack_header(header()), b'data']
    stream = ByteStream()
    with pytest.raises(ProtocolError, match='length'):
        send_packet(stream, header(), b'wrong-length')
    assert not stream.sent


@pytest.mark.parametrize('error', [socket.timeout('timeout'), ConnectionResetError('reset')])
def test_socket_errors_propagate(error):
    class FailingSocket:
        def recv(self, size):
            raise error
        def sendall(self, data):
            raise error
    with pytest.raises(type(error)):
        recv_exact(FailingSocket(), 10)
    with pytest.raises(type(error)):
        send_packet(FailingSocket(), header(), b'data')


def test_recv_exact_zero_and_negative_sizes():
    stream = ByteStream(b'untouched')
    assert recv_exact(stream, 0) == b''
    assert stream.read_sizes == []
    with pytest.raises(ValueError):
        recv_exact(stream, -1)
