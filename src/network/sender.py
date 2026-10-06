"""Stream JPEG frames over TCP: python -m src.network.sender --help."""
import argparse
from contextlib import closing
import math
from pathlib import Path
import socket
import sys
import time

import cv2

from .codec import encode_frame
from .protocol import send_packet
from .sources import frames


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, default=5000)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--image', type=Path, help='Repeat a static image instead of using Picamera2')
    source.add_argument('--video', type=Path, help='Stream a video until EOF instead of using Picamera2')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--jpeg-quality', type=int, default=85)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--timeout', type=float, default=5)
    parser.add_argument('--tuning-file', type=Path, help='Pi camera ISP tuning JSON, e.g. imx219_noir.json (full path).')
    args = parser.parse_args(argv)
    if args.tuning_file and (args.image or args.video):
        parser.error('--tuning-file requires the Pi camera source')
    if not (1 <= args.port <= 65535 and 1 <= args.jpeg_quality <= 100 and
            1 <= args.width <= 8192 and 1 <= args.height <= 8192 and
            args.width * args.height <= 16 * 1024 * 1024 and args.max_frames >= 0 and
            math.isfinite(args.fps) and args.fps > 0 and math.isfinite(args.timeout) and args.timeout > 0):
        parser.error('Invalid port, dimensions, FPS, JPEG quality, frame limit, or timeout')
    return args


def run(args):
    sent = 0
    with socket.create_connection((args.host, args.port), timeout=args.timeout) as connection, closing(frames(args)) as source:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while not args.max_frames or sent < args.max_frames:
            started = time.monotonic_ns()
            try:
                frame = next(source)
            except StopIteration:
                break
            captured_ns = time.monotonic_ns()
            packet = encode_frame(frame, sent, captured_ns, args.jpeg_quality)
            send_packet(connection, packet.header, packet.jpeg_payload)
            sent += 1
            if args.max_frames and sent >= args.max_frames:
                break
            time.sleep(max(0, 1 / args.fps - (time.monotonic_ns() - started) / 1e9))
    print(f'Sent {sent} frames', flush=True)
    return sent


def main():
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print('Sender stopped', flush=True)
    except (OSError, ValueError, RuntimeError, ImportError, cv2.error) as exc:
        print(f'Sender failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
