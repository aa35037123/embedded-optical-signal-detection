# Three-method performance and accuracy experiment

Compare (1) Pi-only CPU, (2) Pi → PC CPU, (3) Pi → PC CUDA.
Actual hardware measurements must be collected on those machines. Desktop/demo
smoke tests are not Pi performance results.

## Metrics and their meaning

| Metric | Purpose |
| --- | --- |
| Mean detector time, including upload/download | Comparable computation cost for all three methods |
| Average completed FPS | Actual throughput: (N-1)/(last completion - first completion) |
| p95/p99 processing and frame interval | Tail delays and visible stuttering |
| Pi capture/read time | Camera wait plus frame acquisition |
| PC decode, GPU upload/download, visualization | Find bottlenecks |
| Queue wait and receive-to-done | Staleness after a frame arrives at the PC |
| Unprocessed frame-ID fraction | Frames skipped by the latest-frame receiver, not TCP packet loss |
| JPEG size, encode time; network summary bandwidth | Transport cost |
| Precision, recall, F1 by color | False points and missed points |
| Matched centroid error in pixels | Position accuracy |

Precision = TP/(TP+FP): how many detected points are real and correctly colored.
Recall = TP/(TP+FN): how many real points were found with the correct color.
Both matter: a detector that outputs nothing has few false alarms but fails its
purpose. F1 balances them. For this localization task, centroid error matters too.
Do not use the detector's heuristic confidence as an accuracy measurement.

A match requires the same color and a center within a fixed tolerance (default
10 pixels). One-to-one maximum-cardinality matching prevents duplicate detections
from inflating TP. A wrong-color detection counts as an FP and an FN. Position
error is measured on matched points only, so always report recall alongside it.
The matcher prioritizes nearby candidates but does not globally minimize distance.

## Controls

1. Verify the GPU first using docs/cuda.md. Stop on preflight/parity failure.
2. Use the same Git revision, detector YAML, camera tuning, lighting, screen
   brightness, focus, distance, target count and target sizes. Record config
   copies, hardware/OS, OpenCV build, GPU driver, and resolution with each trial.
3. Start at 640×480, 30 FPS, JPEG quality 85. Disable --display, --debug-dir and
   --jsonl for baseline trials. All methods still generate their normal overlay.
   Run a separate display-enabled experiment if operator preview is important.
4. Warm up 30 processed frames, then aim for about 60 seconds or more of data.
   The commands below capture/send 1830 frames. A slow Pi can take longer; network
   receivers may process fewer due to dropping. Report actual frames and duration.
5. Run at least three trials of each method, rotating their order to reduce
   thermal/time bias. Use unique output paths. Note Pi temperature/throttling
   (vcgencmd measure_temp and vcgencmd get_throttled) before and after trials.
6. Use a static controlled scene for speed trials. For dynamics, use a repeatable
   recorded clip. Hold target density constant: contour cost depends on it.

Do not subtract Pi capture timestamps from PC timestamps: they are different
monotonic clocks. Current logs do NOT measure true camera-to-display latency.
Measure that separately with an external high-speed recording of a visible
stimulus and the receiver display, or implement clock synchronization with a
quantified error bound.

## Trial commands

Run from the project root. Stop other camera processes. Replace trial1 with
trial2/trial3 and repeat. Save logs even for runs that perform poorly.

### 1. Pi only (on Pi)

```bash
python -m src.pipeline \
  --tuning-file /usr/share/libcamera/ipa/rpi/vc4/imx219_noir.json \
  --config configs/detection-noir.yaml --max-frames 1830 \
  --csv results/experiments/pi-trial1.csv
```

### 2. Pi → PC CPU

Start the PC receiver first:

```bash
python -m src.network.receiver --backend cpu \
  --config configs/detection-noir.yaml --once \
  --csv results/experiments/pc-cpu-trial1.csv
```

On Pi:

```bash
python -m src.network.sender --host YOUR_PC_IP \
  --width 640 --height 480 --fps 30 --jpeg-quality 85 --max-frames 1830 \
  --tuning-file /usr/share/libcamera/ipa/rpi/vc4/imx219_noir.json
```

### 3. Pi → PC CUDA

Activate the PC CUDA environment, then:

```bash
python -m src.network.receiver --backend cuda \
  --config configs/detection-noir.yaml --once \
  --csv results/experiments/pc-cuda-trial1.csv
```

Run exactly the same Pi sender command as Method 2. On Pi 5 use pisp instead of
vc4 in the tuning path. For fair PC CPU/GPU comparisons, use the same CUDA-enabled
OpenCV environment for both receiver backends (avoid comparing different builds).

### Compare logs

Copy Pi CSVs into the PC results folder, then:

```bash
python tools/compare_experiments.py --warmup 30 \
  --run pi=results/experiments/pi-trial1.csv \
  --run pc-cpu=results/experiments/pc-cpu-trial1.csv \
  --run pc-cuda=results/experiments/pc-cuda-trial1.csv \
  --output results/experiments/comparison.json
```

Repeat --run for each additional trial (the same method name is allowed).
The tool prints a compact table and saves detailed per-trial percentiles, frame
counts, timings and across-trial FPS sample standard deviation. Trials are
weighted equally. Missing stage measurements are omitted, not treated as zero.
Warmup exclusion applies to these CSV calculations. Receiver .summary.json
statistics cover the full run, including warmup, and include received FPS,
bandwidth and overwrite count. Its historical mean_processing_ms field describes
whole workstation work, NOT the detector alone; use the comparison tool instead.
Pi local_pipeline_ms includes capture/read waiting; PC total_workstation_ms
starts after reception. These are useful stage totals but not equal boundaries.
Local visualization_ms includes diagnostics/console work after detection. CSV
writes are outside frame stage timings but included in observed FPS intervals.

## Accuracy experiment

Collect 100–300 representative unannotated frames across brightness, distance,
angles, target counts and colors, including no-target scenes and red backgrounds.
Use separate development and held-out test images. Tune thresholds only on the
development set; freeze the config before evaluating the held-out test set.
Label EVERY target in camera image coordinates, including missed targets.
Phone webpage coordinates cannot be used directly without geometric mapping.

Create a JSON manifest next to the images, for example labels.json:

```json
[
  {"image":"frame001.png", "targets":[
    {"color":"red", "x":120, "y":200},
    {"color":"blue", "x":320, "y":150}
  ]},
  {"image":"empty.png", "targets":[]}
]
```

Run on the Pi for the local raw-input result:

```bash
python tools/evaluate_accuracy.py --labels dataset/labels.json \
  --config configs/detection-noir.yaml --backend cpu \
  --output results/experiments/accuracy-pi.json
```

On the PC, simulate the exact one-generation JPEG quality used in transport:

```bash
python tools/evaluate_accuracy.py --labels dataset/labels.json \
  --config configs/detection-noir.yaml --backend cpu --jpeg-quality 85 \
  --output results/experiments/accuracy-pc-cpu.json
python tools/evaluate_accuracy.py --labels dataset/labels.json \
  --config configs/detection-noir.yaml --backend cuda --jpeg-quality 85 \
  --output results/experiments/accuracy-pc-cuda.json
```

Use lossless source images at the sender's resolution: this evaluator does not
resize. PC JPEG simulation tests compression sensitivity; encoder build differences
can differ from real Pi JPEG bytes. For exact transport equivalence, evaluate
saved decoded receiver images against matching annotations without re-encoding.
Null precision/recall denotes an undefined denominator, not a perfect score.
Static-image accuracy excludes dropped frames and temporal detection delays;
report the streaming skip fraction separately. If signals blink, also measure
event recall and detection delay using labeled sequences as a follow-up.

## Additional comparisons

- Repeat at 1280×720 (change Pi detector dimensions and sender dimensions equally)
  to see whether GPU processing becomes worthwhile for larger images.
- Test JPEG qualities 70/85/95 to quantify bandwidth versus detection accuracy.
- Compare 1, 5, 20 points to expose CPU contour-processing costs.
- Benchmark identical decoded pixels without camera/network pacing using
  tools/benchmark_backends.py. This isolates computation; it is not streaming FPS.
- Measure wall power externally if energy efficiency is a goal. GPU utilization
  alone is not evidence of useful speedup.

Choose a method based on recall/precision and latency requirements, then compare
speed and hardware cost. A 30 FPS camera can cap all three methods at 30 FPS even
when their computation times are very different.
