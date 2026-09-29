"""Executable optical target detector (HSV or calibrated colors): python -m src.pipeline --help."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.detection import Detector
from src.visualization import draw_detections
from src.detection.types import DetectionResult
from src.detection.diagnostics import inspect_frame, describe_color, probe_pixel, save_debug
from src.detection.calibration import calibrate_frame


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument('--image', type=Path, help='Process a single image.')
    sources.add_argument('--video', type=Path, help='Process a video until EOF.')
    sources.add_argument('--webcam', type=int, help='OpenCV camera index, e.g. 0.')
    sources.add_argument('--demo', action='store_true', help='Generate moving RGB targets without hardware.')
    parser.add_argument('--config', type=Path, help='Detector YAML configuration.')
    parser.add_argument('--display', action='store_true', help='Show preview; q or Escape exits.')
    parser.add_argument('--fps', type=float, default=30, help='Requested camera/demo FPS.')
    parser.add_argument('--max-frames', type=int, default=0, help='Stop after N frames; 0 means unlimited.')
    parser.add_argument('--output', type=Path, help='Save the last annotated frame as an image.')
    parser.add_argument('--jsonl', type=Path, help='Write every detection to a JSON-lines file.')
    parser.add_argument('--debug-dir', type=Path,
                        help='Show RGB rejection diagnostics and save the last raw frame, masks, and report here.')
    parser.add_argument('--calibrate-colors', type=Path,
                        help='Press C to freeze the preview, sample RGB and background, and save a calibrated YAML.')
    args = parser.parse_args(argv)
    if args.calibrate_colors:
        args.display = True
    if not np.isfinite(args.fps) or args.fps <= 0 or args.max_frames < 0:
        parser.error('--fps must be positive and finite; --max-frames must be nonnegative')
    return args


def demo_frame(index, width, height):
    frame = np.zeros((height, width, 3), np.uint8)
    color = ((0, 0, 255), (0, 255, 0), (255, 0, 0))[(index // 30) % 3]
    center = (int(width * (0.5 + 0.25 * np.sin(index / 20))), height // 2)
    cv2.circle(frame, center, 20, color, -1)
    return frame


def run(args):
    detector = Detector(args.config)
    width, height = detector.config.width, detector.config.height
    camera = capture = log = None
    annotated = None
    debug_frame = None
    debug_windows_ready = False
    count, fps = 0, 0.0
    try:
        if args.jsonl:
            log = args.jsonl.open('w', encoding='utf-8')
        if args.image:
            still = cv2.imread(str(args.image))
            if still is None:
                raise ValueError(f'Cannot read image: {args.image}')
        elif args.video is not None or args.webcam is not None:
            capture = cv2.VideoCapture(str(args.video) if args.video is not None else args.webcam)
            if not capture.isOpened():
                raise RuntimeError('Cannot open video/camera source')
            if args.webcam is not None:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                capture.set(cv2.CAP_PROP_FPS, args.fps)
        elif not args.demo:
            from src.capture.camera import CameraAcquisition
            camera = CameraAcquisition(width, height, args.fps)
            camera.start()

        # Create the display window once and make it fullscreen
        if args.display:
            cv2.namedWindow('Spectrum Tracker', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                'Spectrum Tracker',
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN
            )

        window_start = time.perf_counter()
        window_count = 0
        last_report = window_start - 1
        while True:
            started = time.perf_counter()
            if args.image:
                frame = still
            elif args.demo:
                frame = demo_frame(count, width, height)
            elif capture is not None:
                ok, frame = capture.read()
                if not ok:
                    if args.video is not None and count:
                        break
                    raise RuntimeError('Source returned no frame')
            else:
                frame = camera.read()
            results = detector.detect_all(frame)
            if args.debug_dir:
                debug_frame = frame.copy()
            count += 1
            window_count += 1
            now = time.perf_counter()
            if now - window_start >= 1:
                fps = window_count / (now - window_start)
                window_start, window_count = now, 0
            for result in results:
                result.fps = fps
            annotated = draw_detections(frame, results, fps)
            if log:
                # Preserve the original largest-target fields for existing consumers.
                largest = results[0] if results else DetectionResult(
                    width=frame.shape[1], height=frame.shape[0], fps=fps)
                log.write(json.dumps({'frame': count, **asdict(largest),
                                      'count': len(results),
                                      'detections': [asdict(result) for result in results]}) + '\n')
            if args.image or now - last_report >= 1:
                print(f'Targets: {len(results)} | FPS: {fps:.1f}', flush=True)
                for index, result in enumerate(results, 1):
                    print(f'#{index} Color: {result.detected_color.upper()} | '
                          f'Position: ({result.centroid_x:.0f}, {result.centroid_y:.0f}) | '
                          f'Radius: {result.radius:.1f} px | Confidence: {result.confidence:.2f}',
                          flush=True)
                if args.debug_dir:
                    report, masks = inspect_frame(detector, debug_frame)
                    for color in ('red', 'green', 'blue'):
                        print(describe_color(report, color), flush=True)
                    if args.display:
                        for color in ('red', 'green', 'blue'):
                            cv2.imshow(f'{color}: color threshold', masks[f'{color}-threshold'])
                            cv2.imshow(f'{color}: after morphology', masks[f'{color}-morphology'])
                last_report = now
            if args.display:
                if args.calibrate_colors:
                    cv2.putText(annotated, 'C: freeze for color calibration', (16, 78),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
                cv2.imshow('Spectrum Tracker', annotated)
                if args.debug_dir and not debug_windows_ready:
                    def on_click(event, x, y, flags, param):
                        if event == cv2.EVENT_LBUTTONDOWN and debug_frame is not None:
                            print(probe_pixel(detector, debug_frame, x, y), flush=True)
                    cv2.setMouseCallback('Spectrum Tracker', on_click)
                    print('Click the target in Spectrum Tracker to inspect its BGR/HSV values.', flush=True)
                    debug_windows_ready = True
                key = cv2.waitKey(0 if args.image else 1) & 0xFF
                if key == ord('c') and args.calibrate_colors:
                    if calibrate_frame(frame, args.calibrate_colors, detector.config):
                        print(f'Calibration saved. Run: python -m src.pipeline --display --config {args.calibrate_colors}', flush=True)
                        break
                if key in (ord('q'), 27):
                    break
            if args.image or (args.max_frames and count >= args.max_frames):
                break
            if args.demo:
                time.sleep(max(0, 1 / args.fps - (time.perf_counter() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        if camera is not None:
            camera.stop()
        if capture is not None:
            capture.release()
        if log is not None:
            log.close()
        if args.display:
            cv2.destroyAllWindows()
    if args.debug_dir and debug_frame is not None:
        save_debug(args.debug_dir, detector, debug_frame)
        print(f'Debug capture saved to {args.debug_dir}', flush=True)
    if args.output and annotated is not None:
        if not cv2.imwrite(str(args.output), annotated):
            raise RuntimeError(f'Cannot save output: {args.output}')
    return count


def main():
    args = parse_args()
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, cv2.error, yaml.YAMLError) as exc:
        print(f'Pipeline failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
