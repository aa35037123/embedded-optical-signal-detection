"""Bounded streaming statistics. Never compare clocks from different hosts."""
from collections import deque
import csv
import json
from pathlib import Path
import threading

import numpy as np


class Metrics:
    def __init__(self, window_size=10000):
        self.lock = threading.Lock()
        self.received_frames = self.processed_frames = self.skipped_ids = self.overwritten_frames = 0
        self.jpeg_bytes = self.wire_bytes = self.first_wire_bytes = 0
        self.first_rx = self.last_rx = self.first_pc = self.last_pc = None
        self.total_ms = 0.0
        self.latencies = deque(maxlen=window_size)

    def received(self, packet, timestamp_ns, skipped=0):
        from src.network.protocol import HEADER_SIZE
        with self.lock:
            self.received_frames += 1
            self.skipped_ids += skipped
            self.jpeg_bytes += packet.header.jpeg_size
            self.wire_bytes += HEADER_SIZE + packet.header.jpeg_size
            if self.first_rx is None:
                self.first_rx = timestamp_ns
                self.first_wire_bytes = self.wire_bytes
            self.last_rx = timestamp_ns

    def overwritten(self):
        with self.lock:
            self.overwritten_frames += 1

    def processed(self, completed_ns, total_ms):
        with self.lock:
            self.processed_frames += 1
            self.total_ms += total_ms
            self.latencies.append(total_ms)
            if self.first_pc is None:
                self.first_pc = completed_ns
            self.last_pc = completed_ns

    def snapshot(self):
        with self.lock:
            rx_seconds = (self.last_rx - self.first_rx) / 1e9 if self.first_rx is not None else 0
            pc_seconds = (self.last_pc - self.first_pc) / 1e9 if self.first_pc is not None else 0
            percentiles = np.percentile(self.latencies, [50,95,99]).tolist() if self.latencies else [0,0,0]
            return {
                'received_frames': self.received_frames, 'processed_frames': self.processed_frames,
                'skipped_frame_ids': self.skipped_ids, 'overwritten_frames': self.overwritten_frames,
                'received_fps': (self.received_frames-1)/rx_seconds if rx_seconds > 0 else 0,
                'processed_fps': (self.processed_frames-1)/pc_seconds if pc_seconds > 0 else 0,
                'mean_processing_ms': self.total_ms/self.processed_frames if self.processed_frames else 0,
                'p50_processing_ms': percentiles[0], 'p95_processing_ms': percentiles[1],
                'p99_processing_ms': percentiles[2], 'percentile_window_frames': len(self.latencies),
                'average_jpeg_bytes': self.jpeg_bytes/self.received_frames if self.received_frames else 0,
                'network_mbps': (self.wire_bytes-self.first_wire_bytes)*8/rx_seconds/1e6 if rx_seconds > 0 else 0,
                'wire_bytes': self.wire_bytes,
            }


CSV_FIELDS = ('completed_timestamp_ns', 'capture_read_ms', 'local_pipeline_ms', 'session_id', 'frame_id', 'capture_timestamp_ns', 'jpeg_encode_ms', 'jpeg_size_bytes',
              'receive_timestamp_ns', 'decode_ms', 'upload_ms', 'processing_ms', 'download_ms',
              'visualization_ms', 'total_workstation_ms', 'queue_wait_ms', 'local_receive_to_done_ms',
              'backend', 'detected', 'predicted_color', 'predicted_x', 'predicted_y', 'detections_json')


class CSVLogger:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open('w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.handle, fieldnames=CSV_FIELDS)
        self.writer.writeheader()

    def write(self, row):
        self.writer.writerow(row)
        self.handle.flush()

    def close(self):
        self.handle.close()


def save_summary(path, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics.snapshot(), indent=2) + '\n', encoding='utf-8')
