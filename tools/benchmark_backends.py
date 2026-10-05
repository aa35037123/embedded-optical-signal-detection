"""Compare CPU/CUDA on identical decoded pixels, including transfers."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.detection.backends import create_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/detection-noir.yaml'))
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--output', type=Path, default=Path('results/benchmark/backends.json'))
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    frame = cv2.imread(str(args.image))
    if frame is None:
        parser.error('Cannot read --image')
    report = {'opencv': cv2.__version__, 'shape': list(frame.shape),
              'image': str(args.image), 'config': str(args.config), 'backends': {}}
    for name in ('cpu', 'cuda'):
        backend = create_backend(name, args.config)
        for _ in range(10):
            backend.process(frame)
        timings, stages = [], []
        for _ in range(args.iterations):
            started = time.perf_counter()
            result = backend.process(frame)
            timings.append((time.perf_counter()-started)*1000)
            stages.append([result.upload_ms, result.processing_ms, result.download_ms])
        report['backends'][name] = {
            'median_total_ms': float(np.median(timings)),
            'p95_total_ms': float(np.percentile(timings, 95)),
            'mean_upload_processing_download_ms': np.mean(stages, axis=0).tolist(),
            'targets': len(result.detections)}
    report['cpu_over_cuda_speedup'] = (
        report['backends']['cpu']['median_total_ms'] /
        report['backends']['cuda']['median_total_ms'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
