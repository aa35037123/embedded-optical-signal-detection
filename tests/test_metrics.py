import csv
import json

import pytest

from src.network.protocol import FrameHeader, FramePacket
from src.network.latest import LatestPacketReceiver
from src.profiling.metrics import Metrics, CSVLogger, save_summary


def packet(frame_id):
    return FramePacket(FrameHeader(frame_id, 99, 640, 480, 100), b'x' * 100)


def test_statistics_and_bounded_percentiles(tmp_path):
    metrics = Metrics(window_size=3)
    for i, ms in enumerate((1,2,3,4)):
        metrics.received(packet(i), (i+1)*1_000_000_000, skipped=1 if i == 2 else 0)
        metrics.processed((i+1)*1_000_000_000, ms)
    summary = metrics.snapshot()
    assert summary['received_fps'] == summary['processed_fps'] == 1
    assert summary['skipped_frame_ids'] == 1
    assert summary['mean_processing_ms'] == 2.5
    assert summary['p50_processing_ms'] == 3
    assert summary['p95_processing_ms'] == pytest.approx(3.9)
    assert summary['p99_processing_ms'] == pytest.approx(3.98)
    assert summary['average_jpeg_bytes'] == 100
    assert summary['network_mbps'] == pytest.approx(140*8/1e6)
    assert summary['percentile_window_frames'] == 3
    path = tmp_path / 'summary.json'
    save_summary(path, metrics)
    assert json.loads(path.read_text())['processed_frames'] == 4


def test_latest_slot_overwrites_without_queue_growth():
    metrics = Metrics()
    reader = LatestPacketReceiver(None, metrics)
    for i in range(100):
        reader.publish(packet(i), i)
    value = reader.take(timeout=0)
    assert value[0].header.frame_id == 99
    assert reader.take(timeout=0) is None
    assert metrics.snapshot()['overwritten_frames'] == 99


def test_csv_log(tmp_path):
    logger = CSVLogger(tmp_path / 'frames.csv')
    logger.write({'frame_id': 3, 'backend': 'cpu', 'processing_ms': 1.25})
    logger.close()
    with (tmp_path / 'frames.csv').open() as handle:
        row = next(csv.DictReader(handle))
    assert row['frame_id'] == '3' and row['processing_ms'] == '1.25'
