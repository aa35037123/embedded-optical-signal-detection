"""Report OpenCV/CUDA prerequisites; exit nonzero when no CUDA device is usable."""
import argparse
import json
from pathlib import Path

import cv2


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
    print('CUDA device prerequisite passed. GPU operation/accuracy/benchmark validation is still required.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
