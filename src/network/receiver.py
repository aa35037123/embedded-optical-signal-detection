"""Receive JPEG frames over TCP: python -m src.network.receiver --help."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import socket
import sys
import time

import cv2

from src.detection.backends import create_backend
from src.visualization.network_overlay import draw_network_frame
from src.profiling.metrics import Metrics, CSVLogger, save_summary

from .latest import LatestPacketReceiver
from .codec import decode_packet
from .protocol import ConnectionClosed, FrameSequence, HEADER_SIZE, receive_packet


class FrameReceiver:
    """One connection's framing, validation, and decoding state."""
    def __init__(self, connection):
        self.connection = connection
        self.sequence = FrameSequence()

    def receive_frame(self):
        packet = receive_packet(self.connection, self.sequence)
        received_ns = time.monotonic_ns()
        return decode_packet(packet, received_ns)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bind', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--backend', choices=['cpu'], default='cpu')
    parser.add_argument('--config', type=Path, help='Existing detector YAML configuration')
    parser.add_argument('--output', type=Path, help='Save the final decoded image')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--once', action='store_true', help='Exit after one client disconnects')
    parser.add_argument('--timeout', type=float, default=5)
    parser.add_argument('--csv', type=Path, help='Per-processed-frame CSV, e.g. results/network/cpu.csv')
    parser.add_argument('--summary', type=Path, help='Aggregate JSON (defaults to CSV path with .summary.json)')
    args = parser.parse_args(argv)
    if not (0 <= args.port <= 65535 and args.max_frames >= 0 and
            math.isfinite(args.timeout) and args.timeout > 0):
        parser.error('Invalid port, frame limit, or timeout')
    return args


def run(args):
    backend = create_backend(args.backend, args.config)
    metrics = Metrics()
    logger = CSVLogger(args.csv) if args.csv else None
    count, session_id, last_frame = 0, 0, None
    last_report = 0
    try:
        if args.display:
            cv2.namedWindow('Network camera', cv2.WINDOW_AUTOSIZE)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((args.bind, args.port))
            listener.listen(1)
            listener.settimeout(0.2)
            print(f'Listening on {args.bind}:{listener.getsockname()[1]}', flush=True)
            while True:
                try:
                    connection, address = listener.accept()
                except socket.timeout:
                    if args.display and cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                        return count
                    continue
                session_id += 1
                print(f'Connected: {address}', flush=True)
                with connection:
                    connection.settimeout(args.timeout)
                    reader = LatestPacketReceiver(connection, metrics).start()
                    try:
                        while True:
                            pending = reader.take()
                            if pending is None:
                                if reader.done:
                                    break
                                if args.display and cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                                    return count
                                continue
                            packet, received_ns = pending
                            started_ns = time.monotonic_ns()
                            received = decode_packet(packet, received_ns)
                            processed = backend.process(received.frame)
                            visual_started = time.monotonic_ns()
                            last_frame = draw_network_frame(received.frame, processed, packet.header.frame_id,
                                                            metrics.snapshot()['received_fps'])
                            key = -1
                            if args.display:
                                cv2.imshow('Network camera', last_frame)
                                key = cv2.waitKey(1) & 0xFF
                            finished_ns = time.monotonic_ns()
                            visualization_ms = (finished_ns - visual_started) / 1e6
                            total_ms = (finished_ns - started_ns) / 1e6
                            count += 1
                            metrics.processed(finished_ns, total_ms)
                            if logger:
                                largest = processed.detections[0] if processed.detections else None
                                logger.write({
                                    'session_id': session_id, 'frame_id': packet.header.frame_id,
                                    'capture_timestamp_ns': packet.header.capture_timestamp_ns,
                                    'jpeg_encode_ms': packet.header.jpeg_encode_ns / 1e6,
                                    'jpeg_size_bytes': packet.header.jpeg_size,
                                    'receive_timestamp_ns': received_ns, 'decode_ms': received.decode_ms,
                                    'upload_ms': processed.upload_ms, 'processing_ms': processed.processing_ms,
                                    'download_ms': processed.download_ms, 'visualization_ms': visualization_ms,
                                    'total_workstation_ms': total_ms,
                                    'queue_wait_ms': (started_ns - received_ns) / 1e6,
                                    'local_receive_to_done_ms': (finished_ns - received_ns) / 1e6,
                                    'backend': processed.backend, 'detected': bool(largest),
                                    'predicted_color': largest.detected_color if largest else 'none',
                                    'predicted_x': largest.centroid_x if largest else '',
                                    'predicted_y': largest.centroid_y if largest else '',
                                    'detections_json': json.dumps([asdict(d) for d in processed.detections]),
                                })
                            if finished_ns - last_report >= 1_000_000_000:
                                print(f'Frame {packet.header.frame_id}: backend={processed.backend} '
                                      f'targets={len(processed.detections)} '
                                      f'colors={",".join(r.detected_color for r in processed.detections)}', flush=True)
                                last_report = finished_ns
                            if key in (ord('q'), 27) or (args.max_frames and count >= args.max_frames):
                                return count
                    finally:
                        reader.stop()
                if args.once:
                    return count
    finally:
        if logger:
            logger.close()
        if args.display:
            cv2.destroyAllWindows()
        if args.output and last_frame is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.output), last_frame):
                raise OSError(f'Cannot save {args.output}')
        summary_path = args.summary or (args.csv.with_suffix('.summary.json') if args.csv else None)
        if summary_path:
            save_summary(summary_path, metrics)
        summary = metrics.snapshot()
        print(f'Received {summary["received_frames"]} frames; processed {count}; '
              f'overwritten {summary["overwritten_frames"]}', flush=True)
        print(json.dumps(summary), flush=True)


def main():
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print('Receiver stopped', flush=True)
    except (OSError, ValueError, EOFError, cv2.error) as exc:
        print(f'Receiver failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
