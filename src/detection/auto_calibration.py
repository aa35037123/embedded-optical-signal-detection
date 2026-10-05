"""Sample a known 3x2 phone chart over multiple frames; no color guessing."""
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from .calibration import validate_samples

# Row-major layout shared with tools/signal_generator/calibration.html.
CHART_LABELS = ('red', 'green', 'blue', 'background', 'gray', 'white')


def chart_samples(frame, corners=None):
    """Sample the full-frame 3x2 chart unless explicit geometry is supplied."""
    if corners is None:
        height, width = frame.shape[:2]
        corners = [(0, 0), (width, 0), (width, height), (0, height)]
    points = np.asarray(corners, np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError('Select four chart corners: top-left, top-right, bottom-right, bottom-left.')
    contour = points.reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour) or cv2.contourArea(contour) < 1000:
        raise ValueError('Chart corners must form a large convex quadrilateral.')
    transform = cv2.getPerspectiveTransform(points, np.float32([[0,0],[600,0],[600,400],[0,400]]))
    chart = cv2.warpPerspective(frame, transform, (600,400))
    samples = {}
    for index, label in enumerate(CHART_LABELS):
        x, y = 100+200*(index%3), 100+200*(index//3)
        patch = chart[y-30:y+30, x-30:x+30]
        samples[label] = np.median(patch, axis=(0,1)).tolist()
    return samples


def fit_calibration(observations):
    if len(observations) < 5:
        raise ValueError('At least five chart frames are required.')
    values = {label: np.array([row[label] for row in observations], dtype=float) for label in CHART_LABELS}
    if any(np.max(np.std(samples, axis=0)) > 12 for samples in values.values()):
        raise ValueError('Colors changed during capture. Keep the phone still and allow exposure to settle, then retry.')
    medians = {label: np.median(samples, axis=0) for label, samples in values.items()}
    samples = {label: np.unique(np.round(np.percentile(values[label], [10,50,90], axis=0)).astype(int), axis=0).tolist()
               for label in ('red','green','blue','background')}
    validate_samples(samples)
    # Black offset plus a 3x3 mapping from observed primaries to ideal BGR.
    black = medians['background']
    observed = np.stack([medians[label]-black for label in ('blue','green','red')], axis=1)
    matrix = []
    warning = ''
    if np.linalg.cond(observed) < 50:
        linear = 255 * np.linalg.inv(observed)
        affine = np.column_stack((linear, -linear @ black))
        # Reject correction that cannot reproduce independent neutral reference patches.
        neutral_errors = [np.max(np.abs(affine @ np.append(medians[label], 1) - expected))
                          for label, expected in [('gray',128),('white',255)]]
        if max(neutral_errors) <= 65:
            matrix = affine.tolist()
        else:
            warning = 'Preview correction omitted: gray/white references disagree with the fitted color mapping.'
    else:
        warning = 'Preview correction omitted: camera primaries cannot be inverted reliably.'
    return samples, matrix, warning


def save_auto_calibration(path, config, observations):
    samples, matrix, warning = fit_calibration(observations)
    data = asdict(config)
    data['colors'] = data.pop('color_thresholds')
    data.update(color_samples_bgr=samples, preview_color_matrix=matrix,
                min_confidence=0.0, min_circularity=0.55, min_color_contrast=12.0,
                color_distance_max=30.0, color_margin=8.0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Only replace a saved calibration after all observations pass validation.
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    temporary.replace(path)
    return warning


def corrected_preview(frame, matrix):
    if not matrix:
        return frame
    corrected = cv2.transform(frame.astype(np.float32), np.asarray(matrix, np.float32))
    return np.clip(corrected, 0, 255).astype(np.uint8)


def run_auto_calibration(args):
    from src.capture.camera import CameraAcquisition
    from .detector import Detector
    config = Detector(args.config).config
    camera = CameraAcquisition(config.width, config.height, args.fps, **({'tuning_file': args.tuning_file} if getattr(args, 'tuning_file', None) else {}))
    window = 'Automatic color calibration'
    # Let the camera settle before collecting the calibration observations.
    warmup_frames = 30
    rows = []
    try:
        camera.start()
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        for index in range(warmup_frames + args.calibration_frames):
            frame = camera.read()
            if index >= warmup_frames:
                rows.append(chart_samples(frame))
                prompt = f'Sampling {len(rows)}/{args.calibration_frames} | Q: cancel'
            else:
                prompt = f'Full-frame chart: settling {index+1}/{warmup_frames} | Q: cancel'
            shown = frame.copy()
            height, width = frame.shape[:2]
            for cell, label in enumerate(CHART_LABELS):
                x = round(width * ((cell % 3) + .5) / 3)
                y = round(height * ((cell // 3) + .5) / 2)
                dx, dy = round(width * .05), round(height * .075)
                cv2.rectangle(shown, (x-dx, y-dy), (x+dx, y+dy), (255,255,255), 1)
                cv2.putText(shown, label, (x-dx, y-dy-5),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
            cv2.putText(shown, prompt, (8,20), cv2.FONT_HERSHEY_SIMPLEX,
                        .4, (255,255,255), 1)
            cv2.imshow(window, shown)
            if cv2.waitKey(30) & 0xFF in (ord('q'),27):
                return 0
        warning = save_auto_calibration(args.auto_calibrate_colors, config, rows)
        print(f'Calibration saved: {args.auto_calibrate_colors}', flush=True)
        if warning:
            print(warning, flush=True)
        return len(rows)
    finally:
        camera.stop()
        cv2.destroyAllWindows()
