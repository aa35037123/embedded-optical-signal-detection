import csv

import pytest

from tools.compare_experiments import summarize
from tools.evaluate_accuracy import match_points, scores
from src.pipeline import parse_args, run


def test_local_timing_csv(tmp_path):
    path = tmp_path/'pi.csv'
    run(parse_args(['--demo','--max-frames','3','--fps','1000','--csv',str(path)]))
    result = summarize(path,warmup=0)
    assert result['frames'] == 3 and result['average_fps'] > 0
    assert result['capture_read_ms']['mean'] >= 0
    assert result['detector_including_transfers_ms']['mean'] == result['processing_ms']['mean']
    assert result['local_pipeline_ms']['mean'] >= result['processing_ms']['mean']


def test_fps_uses_elapsed_time_and_network_gaps(tmp_path):
    path = tmp_path/'pc.csv'
    fields = ['session_id','frame_id','completed_timestamp_ns','receive_timestamp_ns',
              'processing_ms','upload_ms','download_ms']
    with path.open('w') as handle:
        writer = csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for frame_id, stamp in [(0,1),(1,2),(3,4),(4,5)]:
            writer.writerow(dict(zip(fields,[1,frame_id,stamp*10**9,1,10,2,3])))
    result = summarize(path,warmup=1)
    assert result['average_fps'] == pytest.approx(2/3)
    assert result['unprocessed_id_fraction_in_span'] == .25
    assert result['detector_including_transfers_ms']['mean'] == 15
    with pytest.raises(ValueError):
        summarize(path,warmup=3)


def test_one_to_one_matching_and_wrong_color():
    truth = [{'color':'red','x':0,'y':0},{'color':'red','x':4,'y':0}]
    # First prediction is available to both truths; second only to first.
    pred = [{'color':'red','x':2,'y':0},{'color':'red','x':-2,'y':0}]
    assert match_points(truth,pred,3)[:3] == (2,0,0)
    pred.append({'color':'green','x':0,'y':0})
    assert match_points(truth,pred,3)[:3] == (2,1,0)
    assert match_points(truth,[{'color':'blue','x':0,'y':0}],3)[:3] == (0,1,2)
    assert scores(0,0,0)['precision'] is None
