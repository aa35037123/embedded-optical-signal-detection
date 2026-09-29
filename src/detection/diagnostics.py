"""Optional diagnostics for distinguishing threshold failures from blob rejection."""
from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from .detector import Detector
from .calibration import color_features


def inspect_frame(detector: Detector, frame: np.ndarray):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    report = {'width': frame.shape[1], 'height': frame.shape[0],
              'config': asdict(detector.config), 'colors': {}}
    masks = {}
    raw_masks, cleaned_masks = detector.frame_masks(frame, hsv)
    features = color_features(frame) if detector.config.min_color_contrast > 0 else None
    for color in ('red', 'green', 'blue'):
        raw, cleaned = raw_masks[color], cleaned_masks[color]
        masks[f'{color}-threshold'] = raw * 255
        masks[f'{color}-morphology'] = cleaned * 255
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            candidate = detector._measure_candidate(contour, cleaned, hsv, color, features)
            reason = ('too_small' if area < detector.config.min_area else
                      'too_large' if area > detector.config.max_area else
                      detector.rejection_reason(candidate))
            candidates.append({'area': area, 'bbox': list(cv2.boundingRect(contour)), 'status': reason,
                               'confidence': candidate.confidence if candidate is not None else None,
                               'circularity': candidate.circularity if candidate is not None else None,
                               'color_contrast': candidate.color_contrast if candidate is not None else None})
        report['colors'][color] = {
            'threshold_pixels': int(np.count_nonzero(raw)),
            'morphology_pixels': int(np.count_nonzero(cleaned)),
            'candidates': sorted(candidates, key=lambda item: -item['area']),
        }
    return report, masks


def describe_color(report, color):
    red = report['colors'][color]
    counts = {key: sum(c['status'] == key for c in red['candidates'])
              for key in ('accepted', 'too_small', 'too_large', 'low_confidence', 'low_circularity', 'low_contrast', 'degenerate')}
    return (f"{color.upper()} debug: threshold pixels={red['threshold_pixels']}, "
            f"after morphology={red['morphology_pixels']}, "
            + ', '.join(f'{key}={value}' for key, value in counts.items()))


def describe_red(report):
    return describe_color(report, 'red')


def probe_pixel(detector, frame, x, y):
    """Explain the raw threshold decision at a clicked source pixel."""
    if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
        return 'Pixel is outside the image.'
    pixel = frame[y:y+1, x:x+1]
    h, s, v = map(int, cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0])
    if detector.config.color_samples_bgr:
        raw, _ = detector.frame_masks(pixel, cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV))
        matches = [color for color, mask in raw.items() if mask[0, 0]]
        return f'Pixel ({x}, {y}) BGR={frame[y, x].tolist()} HSV=({h}, {s}, {v}): calibrated match={matches or "background/ambiguous"}'
    red = detector.config.color_thresholds.get('red')
    failures = []
    if red is None:
        failures.append('red is not configured')
    else:
        if not any(lo <= h <= hi for lo, hi in red.hue_ranges):
            failures.append(f'hue outside {red.hue_ranges}')
        if not red.saturation_min <= s <= red.saturation_max:
            failures.append(f'saturation outside [{red.saturation_min}, {red.saturation_max}]')
        if not red.value_min <= v <= red.value_max:
            failures.append(f'brightness outside [{red.value_min}, {red.value_max}]')
    decision = '; '.join(failures) if failures else 'passes red HSV thresholds; check morphology and contour area'
    return f'Pixel ({x}, {y}) BGR={frame[y, x].tolist()} HSV=({h}, {s}, {v}): {decision}'


def save_debug(directory: Path, detector: Detector, frame: np.ndarray):
    directory.mkdir(parents=True, exist_ok=True)
    report, masks = inspect_frame(detector, frame)
    for name, image in {'frame': frame, **masks}.items():
        path = directory / f'{name}.png'
        if not cv2.imwrite(str(path), image):
            raise OSError(f'Cannot save {path}')
    (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
