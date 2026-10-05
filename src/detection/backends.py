"""Processing adapters: detection remains independent of network transport."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time

import numpy as np

from .detector import Detector
from .types import DetectionResult


@dataclass
class ProcessingResult:
    detections: list[DetectionResult]
    backend: str
    processing_ms: float
    upload_ms: float = 0.0
    download_ms: float = 0.0


class DetectionBackend(ABC):
    name: str

    @abstractmethod
    def process(self, frame: np.ndarray) -> ProcessingResult:
        """Process one BGR frame and return detections plus local timings."""


class CPUBackend(DetectionBackend):
    name = 'cpu'

    def __init__(self, config=None):
        self.detector = Detector(config)

    def process(self, frame):
        started = time.monotonic_ns()
        detections = self.detector.detect_all(frame)
        return ProcessingResult(detections, self.name, (time.monotonic_ns() - started) / 1e6)


def create_backend(name, config=None):
    if name == 'cpu':
        return CPUBackend(config)
    if name == 'cuda':
        from .cuda_backend import CUDABackend
        return CUDABackend(config)
    raise ValueError(f'Unknown backend: {name}')
