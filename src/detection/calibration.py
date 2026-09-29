"""Deterministic color matching from labeled camera pixels (no ML)."""
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import yaml

LABELS = ('red', 'green', 'blue', 'background')


def color_features(bgr):
    """OpenCV Lab with reduced luminance weight to tolerate brightness changes."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 0.35
    return lab


def validate_samples(samples):
    if set(samples) != set(LABELS):
        raise ValueError('Color calibration needs red, green, blue, and background samples.')
    for label in LABELS:
        values = np.asarray(samples[label], dtype=float)
        if (values.ndim != 2 or values.shape[1] != 3 or not len(values)
                or not np.isfinite(values).all() or (values < 0).any() or (values > 255).any()):
            raise ValueError(f'{label} samples must be lists of BGR triples in [0, 255].')
    features = {label: color_features(np.array([samples[label]], np.uint8))[0] for label in LABELS}
    for i, label in enumerate(LABELS):
        for other in LABELS[i+1:]:
            separation = np.linalg.norm(features[label][:, None] - features[other][None, :], axis=2).min()
            if separation < 20:
                raise ValueError(f'{label} and {other} samples are too similar. Resample their centers or improve lighting.')


def calibrated_masks(frame, samples, max_distance, margin):
    features = color_features(frame)
    distances = []
    for label in LABELS:
        references = color_features(np.array([samples[label]], dtype=np.uint8))[0]
        distance = np.full(frame.shape[:2], np.inf, dtype=np.float32)
        for reference in references:
            distance = np.minimum(distance, np.linalg.norm(features - reference, axis=2))
        distances.append(distance)
    distances = np.stack(distances, axis=-1)
    winner = distances.argmin(axis=-1)
    ordered = np.sort(distances, axis=-1)
    confident = (ordered[..., 0] <= max_distance) & (ordered[..., 1] - ordered[..., 0] >= margin)
    return {label: ((winner == index) & confident).astype(np.uint8)
            for index, label in enumerate(LABELS[:3])}


def save_calibration(path, config, samples):
    validate_samples(samples)
    data = asdict(config)
    data['colors'] = data.pop('color_thresholds')
    data.update(color_samples_bgr=samples, min_confidence=0.0,
                min_circularity=0.55, min_color_contrast=12.0,
                color_distance_max=30.0, color_margin=8.0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


def calibrate_frame(frame, path, config):
    """Collect known colors on one frozen raw frame; save only on explicit S."""
    steps = ('red', 'green', 'blue', 'background', 'background', 'background')
    clicks = []
    message = ''
    name = 'Color calibration'
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(steps):
            h, w = frame.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                patch = frame[max(0, y-2):min(h, y+3), max(0, x-2):min(w, x+3)]
                clicks.append((x, y, np.median(patch, axis=(0, 1)).astype(int).tolist()))

    cv2.setMouseCallback(name, click)
    try:
        while True:
            display = frame.copy()
            prompt = (f'Click actual {steps[len(clicks)].upper()} ' +
                      ('dot center' if len(clicks) < 3 else 'background patch (no dot)')) if len(clicks) < 6 else 'S: save calibration'
            cv2.rectangle(display, (0, 0), (display.shape[1], 82), (0, 0, 0), -1)
            for i, text in enumerate((prompt, 'U: undo | Q/Esc: cancel', message[:85])):
                cv2.putText(display, text, (8, 22 + 23*i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
            for index, (x, y, _) in enumerate(clicks):
                cv2.drawMarker(display, (x, y), (255,255,255), cv2.MARKER_CROSS, 12, 1)
                cv2.putText(display, steps[index], (x+8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            cv2.imshow(name, display)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                return False
            if key == ord('u') and clicks:
                clicks.pop(); message = ''
            if key == ord('s') and len(clicks) == 6:
                samples = {label: [] for label in LABELS}
                for label, (_, _, sample) in zip(steps, clicks):
                    samples[label].append(sample)
                try:
                    save_calibration(path, config, samples)
                    return True
                except ValueError as exc:
                    message = str(exc)
                    print(message, flush=True)
    finally:
        cv2.destroyWindow(name)
