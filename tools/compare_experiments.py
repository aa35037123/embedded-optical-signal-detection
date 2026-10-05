"""Summarize per-frame CSVs. One method=path argument per independent trial."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def summarize(path, warmup=30):
    with Path(path).open() as handle:
        rows = list(csv.DictReader(handle))
    if len({r['session_id'] for r in rows}) > 1:
        raise ValueError('Use one connection/session per trial')
    rows = rows[warmup:]
    if len(rows) < 2:
        raise ValueError('Need at least two measured frames after warmup')
    stamps = np.array([int(r['completed_timestamp_ns']) for r in rows], dtype=np.int64)
    if np.any(np.diff(stamps) <= 0):
        raise ValueError('Completion timestamps must increase; regenerate old CSV logs')
    duration = (int(stamps[-1])-int(stamps[0]))/1e9
    result = {'csv': str(path), 'backend': rows[0].get('backend'), 'frames': len(rows), 'measurement_span_s': duration,
              'average_fps': (len(rows)-1)/duration,
              'p95_frame_interval_ms': float(np.percentile(np.diff(stamps)/1e6,95))}
    for field in ('processing_ms','upload_ms','download_ms','capture_read_ms',
                  'visualization_ms','local_pipeline_ms','decode_ms',
                  'total_workstation_ms','queue_wait_ms','local_receive_to_done_ms',
                  'jpeg_encode_ms','jpeg_size_bytes'):
        values = [float(r[field]) for r in rows if r.get(field)]
        if values:
            result[field] = {'mean': float(np.mean(values)),
                             'p50': float(np.median(values)),
                             'p95': float(np.percentile(values,95)),
                             'p99': float(np.percentile(values,99))}
    totals = [sum(float(r.get(k) or 0) for k in ('processing_ms','upload_ms','download_ms')) for r in rows]
    result['detector_including_transfers_ms'] = {'mean': float(np.mean(totals)),
                                               'p95': float(np.percentile(totals,95))}
    if rows[0].get('receive_timestamp_ns'):
        ids = [int(r['frame_id']) for r in rows]
        if any(b <= a for a,b in zip(ids, ids[1:])):
            raise ValueError('Frame IDs must increase')
        span = ids[-1]-ids[0]+1
        result['unprocessed_id_fraction_in_span'] = 1-len(ids)/span
        result['note'] = 'ID gaps include unprocessed frames; this is not TCP packet loss.'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='pi=path.csv, pc-cpu=path.csv or pc-cuda=path.csv; repeat for trials')
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--output', type=Path, default=Path('results/experiments/comparison.json'))
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error('--warmup cannot be negative')
    groups = {}
    for item in args.run:
        method, path = item.split('=', 1)
        if method not in ('pi', 'pc-cpu', 'pc-cuda'):
            parser.error('Method must be pi, pc-cpu or pc-cuda')
        trial = summarize(path, args.warmup)
        expected = 'cuda' if method == 'pc-cuda' else 'cpu'
        if trial['backend'] != expected:
            parser.error(f'{method} requires a {expected} CSV')
        if (method == 'pi') != ('capture_read_ms' in trial):
            parser.error('Pi trials require local pipeline logs; PC trials require receiver logs')
        groups.setdefault(method, []).append(trial)
    report = {}
    print('| Method | Trials | Mean FPS | Mean detector ms incl. transfers |')
    print('| --- | ---: | ---: | ---: |')
    for method, trials in groups.items():
        fps = [r['average_fps'] for r in trials]
        latency = [r['detector_including_transfers_ms']['mean'] for r in trials]
        report[method] = {'trials': trials, 'mean_trial_fps': float(np.mean(fps)),
                          'sample_sd_trial_fps': float(np.std(fps,ddof=1)) if len(fps)>1 else None,
                          'mean_trial_detector_ms': float(np.mean(latency)),
                          'sample_sd_trial_detector_ms': float(np.std(latency,ddof=1)) if len(latency)>1 else None}
        print(f'| {method} | {len(trials)} | {np.mean(fps):.2f} | {np.mean(latency):.3f} |')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
