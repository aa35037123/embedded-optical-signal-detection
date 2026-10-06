"""Report OpenCV/CUDA prerequisites; exit nonzero when no CUDA device is usable."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/benchmark/cuda-preflight.json'))
    args = parser.parse_args()
    error = None
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
    except (AttributeError, cv2.error) as exc:
        count = 0
        error = str(exc)
    report = {
        'opencv_version': cv2.__version__,
        'cuda_device_count': count,
        'error': error,
        'opencv_build_information': cv2.getBuildInformation(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'OpenCV: {cv2.__version__}; CUDA devices: {count}')
    print(f'Full build information: {args.output}')
    if count <= 0:
        print('CUDA prerequisite FAILED: no usable OpenCV CUDA device. Check the NVIDIA workstation, driver, and OpenCV CUDA build.')
        return 1
    try:
        from src.detection.backends import create_backend
        frame = np.zeros((128, 128, 3), np.uint8)
        cv2.circle(frame, (64, 64), 15, (0, 0, 255), -1)
        result = create_backend('cuda').process(frame)
        if len(result.detections) != 1 or result.detections[0].detected_color != 'red':
            raise RuntimeError('GPU smoke test did not detect the red target')
    except (RuntimeError, ValueError, AttributeError, cv2.error) as exc:
        report['operation_error'] = str(exc)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(f'CUDA operation prerequisite FAILED: {exc}')
        return 1
    report['operation_smoke_test'] = 'passed'
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('CUDA operations passed. Run GPU parity tests and benchmark next.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
