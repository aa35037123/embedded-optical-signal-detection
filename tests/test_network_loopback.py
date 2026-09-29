"""Real loopback CLI smoke test (requires permission to open local TCP sockets)."""
import csv
import json
from pathlib import Path
import selectors
import subprocess
import sys

import cv2
import numpy as np


def test_static_image_tcp_cli(tmp_path):
    source, output = tmp_path / 'source.png', tmp_path / 'received.png'
    log = tmp_path / 'metrics.csv'
    frame = np.zeros((48, 64, 3), np.uint8)
    cv2.circle(frame, (30, 20), 10, (0, 255, 0), -1)
    cv2.imwrite(str(source), frame)
    root = Path(__file__).resolve().parents[1]
    receiver = subprocess.Popen([sys.executable, '-m', 'src.network.receiver', '--bind', '127.0.0.1',
                                 '--port', '0', '--once', '--output', str(output), '--csv', str(log)],
                                cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(receiver.stdout, selectors.EVENT_READ)
            assert selector.select(10), 'receiver did not start'
        line = receiver.stdout.readline().strip()
        assert line.startswith('Listening on '), line
        port = line.rsplit(':', 1)[1]
        sender = subprocess.run([sys.executable, '-m', 'src.network.sender', '--host', '127.0.0.1',
                                 '--port', port, '--image', str(source), '--width', '64', '--height', '48',
                                 '--max-frames', '3'], cwd=root, capture_output=True, text=True, timeout=15)
        assert sender.returncode == 0, sender.stderr
        stdout, stderr = receiver.communicate(timeout=10)
        assert receiver.returncode == 0, stderr
        assert 'Received 3 frames' in stdout
        assert 'backend=cpu targets=1 colors=green' in stdout
        result = cv2.imread(str(output))
        assert result.shape == frame.shape
        assert result[20, 30, 1] > 220
        with log.open() as handle:
            rows = list(csv.DictReader(handle))
        assert rows and all(row['backend'] == 'cpu' for row in rows)
        assert all(float(row['total_workstation_ms']) > 0 for row in rows)
        summary = json.loads(log.with_suffix('.summary.json').read_text())
        assert summary['received_frames'] == 3
        assert summary['processed_frames'] + summary['overwritten_frames'] == 3
    finally:
        if receiver.poll() is None:
            receiver.kill()
        receiver.communicate()
