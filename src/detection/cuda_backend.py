"""OpenCV CUDA HSV/morphology with shared CPU contour validation.

Requires CUDA-enabled OpenCV + contrib, not the CPU-only PyPI wheels.
No silent CPU fallback. Calibrated Lab classification is not supported.
"""
import time

import cv2
import numpy as np

from .backends import DetectionBackend, ProcessingResult
from .detector import Detector


def require_cuda():
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
    except (AttributeError, cv2.error) as exc:
        raise RuntimeError('OpenCV CUDA is unavailable; see docs/cuda.md') from exc
    if count <= 0:
        raise RuntimeError('No usable OpenCV CUDA device; see docs/cuda.md')
    required = ('cvtColor', 'inRange', 'bitwise_or', 'createMorphologyFilter', 'Stream_Null')
    missing = [name for name in required if not hasattr(cv2.cuda, name)]
    if missing:
        raise RuntimeError(f'OpenCV CUDA bindings missing: {missing}; see docs/cuda.md')


class CUDABackend(DetectionBackend):
    name = 'cuda'

    def __init__(self, config=None):
        self.detector = Detector(config)
        if self.detector.config.color_samples_bgr:
            raise ValueError('CUDA supports HSV configs only; use detection-noir.yaml or --backend cpu for calibrated colors')
        require_cuda()
        self.input = cv2.cuda_GpuMat()
        self.stream = cv2.cuda.Stream_Null()
        morph = self.detector.config.morphology
        self.filters = [
            cv2.cuda.createMorphologyFilter(
                op, cv2.CV_8UC1,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
            for op, size in [(cv2.MORPH_OPEN, morph.open_kernel),
                             (cv2.MORPH_CLOSE, morph.close_kernel)]
        ]

    def process(self, frame):
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or
                frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0):
            raise ValueError('Expected a nonempty uint8 BGR frame')
        started = time.perf_counter()
        self.input.upload(np.ascontiguousarray(frame))
        self.stream.waitForCompletion()
        uploaded = time.perf_counter()
        hsv_gpu = cv2.cuda.cvtColor(self.input, cv2.COLOR_BGR2HSV)
        gpu_masks = {}
        for color in ('red', 'green', 'blue'):
            threshold = self.detector.config.color_thresholds.get(color)
            if threshold is None or not threshold.hue_ranges:
                continue
            mask = None
            for lo, hi in threshold.hue_ranges:
                part = cv2.cuda.inRange(
                    hsv_gpu, (lo, threshold.saturation_min, threshold.value_min),
                    (hi, threshold.saturation_max, threshold.value_max))
                mask = part if mask is None else cv2.cuda.bitwise_or(mask, part)
            for operation in self.filters:
                mask = operation.apply(mask)
            gpu_masks[color] = mask
        self.stream.waitForCompletion()
        computed = time.perf_counter()
        hsv = hsv_gpu.download()
        masks = {color: mask.download() for color, mask in gpu_masks.items()}
        self.stream.waitForCompletion()
        downloaded = time.perf_counter()
        # Preserve the CPU detector's red/green/blue ownership order.
        claimed = np.zeros(frame.shape[:2], bool)
        for mask in masks.values():
            mask[claimed] = 0
            claimed |= mask > 0
        detections = self.detector.detect_from_masks(frame, hsv, masks)
        finished = time.perf_counter()
        return ProcessingResult(
            detections, self.name,
            ((computed-uploaded) + (finished-downloaded))*1000,
            (uploaded-started)*1000, (downloaded-computed)*1000)
