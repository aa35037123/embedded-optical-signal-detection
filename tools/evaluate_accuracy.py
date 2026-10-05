"""Evaluate exhaustively labeled images, with optional network JPEG simulation."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.detection.backends import create_backend


def match_points(expected, actual, tolerance):
    """Maximum-cardinality same-color matching; nearest candidates visited first."""
    edges = []
    for truth in expected:
        options = []
        for index, pred in enumerate(actual):
            distance = float(np.hypot(truth['x']-pred['x'], truth['y']-pred['y']))
            if truth['color'] == pred['color'] and distance <= tolerance:
                options.append((distance,index))
        edges.append(sorted(options))
    owners = {}
    def assign(truth_index, visited):
        for distance, prediction in edges[truth_index]:
            if prediction in visited:
                continue
            visited.add(prediction)
            if prediction not in owners or assign(owners[prediction], visited):
                owners[prediction] = truth_index
                return True
        return False
    for index in range(len(expected)):
        assign(index, set())
    errors = [float(np.hypot(expected[t]['x']-actual[p]['x'], expected[t]['y']-actual[p]['y']))
              for p,t in owners.items()]
    return len(owners), len(actual)-len(owners), len(expected)-len(owners), errors


def scores(tp, fp, fn):
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp/(tp+fp) if tp+fp else None,
            'recall': tp/(tp+fn) if tp+fn else None,
            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--backend', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--jpeg-quality', type=int, help='Simulate one encode/decode as in transport')
    parser.add_argument('--tolerance', type=float, default=10, help='Maximum matching distance in pixels')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error('--tolerance must be positive and finite')
    if args.jpeg_quality is not None and not 1 <= args.jpeg_quality <= 100:
        parser.error('--jpeg-quality must be 1..100')
    labels = json.loads(args.labels.read_text())
    if not labels:
        parser.error('Label manifest is empty')
    backend = create_backend(args.backend, args.config)
    counts = {color: np.zeros(3,dtype=int) for color in ('red','green','blue')}
    errors, rows = [], []
    for item in labels:
        frame = cv2.imread(str(args.labels.parent / item['image']))
        if frame is None:
            raise ValueError(f"Cannot read {item['image']}")
        truth = item['targets']
        for target in truth:
            if (target['color'] not in counts or not np.isfinite([target['x'],target['y']]).all()
                    or not 0 <= target['x'] < frame.shape[1] or not 0 <= target['y'] < frame.shape[0]):
                raise ValueError('Invalid label color or coordinates')
        if args.jpeg_quality is not None:
            ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY,args.jpeg_quality])
            if not ok:
                raise RuntimeError('JPEG encoding failed')
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        predictions = [{'color': d.detected_color, 'x': d.centroid_x, 'y': d.centroid_y}
                       for d in backend.process(frame).detections]
        tp,fp,fn,distances = match_points(truth,predictions,args.tolerance)
        errors.extend(distances)
        rows.append({'image': item['image'], **scores(tp,fp,fn), 'predictions': predictions})
        for color in counts:
            a,b,c,_ = match_points([t for t in truth if t['color']==color],
                                    [p for p in predictions if p['color']==color],args.tolerance)
            counts[color] += (a,b,c)
    total = np.sum(list(counts.values()),axis=0).tolist()
    report = {'backend': args.backend, 'config': str(args.config), 'jpeg_quality': args.jpeg_quality,
              'tolerance_px': args.tolerance, 'frames': len(rows), **scores(*total),
              'false_positives_per_frame': total[1]/len(rows),
              'per_color': {k:scores(*v.tolist()) for k,v in counts.items()},
              'mean_matched_position_error_px': float(np.mean(errors)) if errors else None,
              'p95_matched_position_error_px': float(np.percentile(errors,95)) if errors else None,
              'images': rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='images'},indent=2))


if __name__ == '__main__':
    main()
