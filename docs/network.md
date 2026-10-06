# TCP optical detection: phases B–E

The sender performs capture and JPEG encoding only. The receiver decodes frames,
runs the existing detector through `CPUBackend`, draws detections, and optionally
logs real measurements. All detector YAMLs, including camera calibration, are
accepted by the CPU backend; CUDA currently supports HSV configs only. The local `src.pipeline` command remains available.

```text
Phone optical signal
  → Raspberry Pi Camera → Picamera2 BGR frames
  → JPEG encoder → TCP (40-byte protocol header + JPEG)
  → Linux workstation JPEG decoder
  → CPUBackend → existing Detector.detect_all()
  → detections + visualization + profiling
```

The receiver also implements a hybrid CUDA backend: see [Method 3](cuda.md)
for its separate OpenCV build, hardware-validation gates, and CPU/CUDA benchmark.
This guide covers the CPU path. Merely having a GPU does not enable acceleration.

## Install and start

On the workstation, use the project virtual environment with
`pip install -r requirements.txt`. Picamera2 is not needed on the workstation.
Find its LAN address with `hostname -I` or `ip -4 addr`; use the address reachable
from the Pi, not `127.0.0.1` and not a Docker-only address.

Start the workstation receiver first:

```bash
python -m src.network.receiver --bind 0.0.0.0 --port 5000 \
  --backend cpu --config configs/detection-red-background.yaml --display \
  --csv results/network/cpu.csv
```

On the Pi, use a virtual environment with system packages so Picamera2 is available:

```bash
python -m src.network.sender --host <WORKSTATION_LAN_IP> --port 5000 \
  --width 640 --height 480 --fps 30 --jpeg-quality 85
```

Allow inbound TCP port 5000 on the workstation firewall from the Pi's address.
If using UFW, a rule can be scoped as `sudo ufw allow from <PI_IP> to any port 5000 proto tcp`.
Both machines need a reachable network path; client isolation on Wi-Fi may block it.
The connection carries unencrypted, unauthenticated video and is intended for a
trusted LAN. Binding to `127.0.0.1` limits testing to the same machine.

Press `q`/Escape in the receiver preview or Ctrl+C in either terminal to exit.
Use `--display` only with a working graphical session; omit it over headless SSH.
The sender releases the camera on disconnect, socket timeout, capture failure,
or interruption. Restart it to reconnect. The receiver accepts sequential clients
unless `--once` is used; a malformed stream or unexpected socket error stops the
receiver with a nonzero exit code and diagnostic message. Restart after fixing it.

## Phase B: static JPEG over TCP

Use any existing image file. Start these in separate terminals on one machine:

```bash
python -m src.network.receiver --bind 127.0.0.1 --port 5000 --once \
  --output results/network/static-received.png --csv results/network/static.csv
python -m src.network.sender --host 127.0.0.1 --port 5000 \
  --image <IMAGE_PATH> --max-frames 30
```

The sender resizes image/video inputs to `--width` × `--height` and paces them
using `--fps`. Image mode repeats the image; `--video <VIDEO_PATH>` stops at EOF.
Image/video capture timestamps mark local acquisition of the frame for sending,
not the original recording time. JPEG is lossy, so compare decoded colors and
geometry rather than requiring byte-identical pixels.

```bash
python -m pytest tests/test_protocol.py tests/test_network_codec.py tests/test_network_loopback.py -q
```

The loopback test uses actual TCP and checks detection and CSV logging as well.
It needs permission to open local sockets.

## Phase C: live camera

Run the receiver as above, then stream a bounded live-camera test:

```bash
python -m src.network.sender --host <WORKSTATION_LAN_IP> --max-frames 60
```

Source cleanup and video EOF tests:

```bash
python -m pytest tests/test_network_sources.py -q
```

Five live IMX219 frames at 640×480 were successfully captured, encoded, transmitted
over localhost TCP, and decoded during development. This does not establish
connectivity to your remote workstation.

## Phase D: CPU detector reuse

`src/detection/backends.py` defines `DetectionBackend.process(frame)` and a
`CPUBackend` adapter that calls `Detector.detect_all()` directly. No detector
algorithm or thresholds were changed for networking. The result includes all
spots, the actual backend name, and measured CPU processing time. CPU upload and
download times are zero because no GPU transfer occurs.

```bash
python -m pytest tests/test_backends.py tests/test_network_loopback.py -q
```

Keep the same YAML that works locally. If JPEG changes small edge pixels enough
to affect detection, compare the source with decoded output or increase JPEG
quality; don't assume networking should select a different color calibration.

## Phase E: latest frame, display, and timings

A dedicated TCP reader validates headers and continuously replaces one pending
packet slot. The main thread takes the newest packet, validates/decompresses its
JPEG, detects targets, and visualizes it. There is no unbounded application queue.
This separates receiver throughput from detector throughput. JPEG decoding happens
only for processed packets. Headers, payload bounds, and sequence order are checked
for every packet; JPEG content/dimensions are checked before processing a frame.

- `skipped_frame_ids`: gaps in wire-order IDs within a connection.
- `overwritten_frames`: fully received packets replaced before processing.
- A newly connected client has a fresh frame-ID tracker and a new CSV `session_id`.
- On normal sender EOF, the last pending packet is processed before closing.
- Stopping the receiver early may discard a pending/in-flight packet; that is not
  counted as an overwrite, so received need not equal processed + overwritten then.
- TCP/kernel buffers can still add latency during network congestion. A one-slot
  receiver bounds application backlog, not network transit time. The sender does
  not accumulate an application queue and aborts a stalled send after `--timeout`.

The preview shows colors, pixel and normalized coordinates, frame ID, received
FPS, backend, and detection processing time. Headless runs still construct the
annotated frame, so visualization time includes overlay rendering even without a
window. An optional `--output` saves the final annotated frame.

CSV examples:

```bash
python -m src.network.receiver --backend cpu --display \
  --csv results/network/cpu.csv
python -m src.network.receiver --backend cpu --once \
  --csv results/benchmark/cpu.csv
```

CSV files overwrite the selected path. A JSON summary is saved next to the CSV
with `.summary.json`, or at `--summary <PATH>`. No rows/results are fabricated.
The CSV includes:

- Sender clock: capture timestamp, JPEG encode duration, encoded size.
- Receiver clock: receive completion timestamp, decode, processing, upload,
  download, visualization, total workstation time, queue wait, and local
  receive-to-completion time.
- Backend, largest predicted spot for simple CSV consumers, and `detections_json`
  containing every detected spot.

`total_workstation_ms` measures decode through overlay/display completion. It
excludes queue waiting and CSV/file output. `processing_ms` measures the backend's
work, while `visualization_ms` includes GUI event handling when enabled. Summary
latency means/percentiles use `total_workstation_ms`. The mean covers all processed
frames; p50/p95/p99 cover at most the latest 10,000 frames, making memory bounded.

Received/processed FPS use inter-completion intervals; one frame produces zero
FPS until a second sample arrives. Network throughput counts protocol header plus
JPEG bytes between first and last receive completions, excluding the first packet
from that interval. It excludes TCP/IP overhead. Average JPEG size uses all fully
received packets. Reconnect idle gaps remain part of aggregate FPS intervals.

**Clock limitation:** Pi `monotonic_ns()` and workstation `monotonic_ns()` have
independent origins. `receiver_time - capture_time` is not one-way network latency.
Only local intervals are reported; cross-machine latency requires synchronized
clocks and an explicit clock-error bound. The capture timestamp is after the
application's camera read, not a hardware exposure timestamp.

Validation:

```bash
python -m pytest tests/test_metrics.py tests/test_network_loopback.py -q
```

Actual development measurements from a short 30-frame live Pi-to-localhost run are
in `results/network/phase-e-live-loopback-cpu.csv` and its summary. That run received
30 frames, processed 19, and overwrote 11 with zero ID gaps. It is a functional
backpressure check on the Pi, **not** a workstation benchmark or CPU/CUDA comparison.
The overlay was also visually checked using synthetic RGB targets. Physical GUI
and remote-LAN behavior still need checking on the workstation.

## Troubleshooting

- Connection refused: start the receiver first and check its bind address/port.
- Timeout: check LAN addresses, firewall, Wi-Fi isolation, and that frames are
  arriving within `--timeout` (default 5 seconds).
- Camera busy: stop another Picamera2/rpicam process before starting the sender.
- No display: omit `--display` and inspect CSV or `--output` from a desktop.
- Invalid JPEG/header: close/restart the connection; do not try to resynchronize
  inside JPEG bytes. Both endpoints must use the documented protocol version.
- Increasing overwrite count: processing is slower than acquisition. This is the
  intentional latest-frame policy; compare RX FPS with processed FPS.
- Wrong colors: use the same detector calibration as the working local pipeline.
  Picamera2 RGB888 already returns OpenCV BGR byte order; do not swap twice.

The shared protocol layout and limits remain documented in the main README.

## Phase F prerequisite check

Run on the NVIDIA workstation using the same Python environment as the receiver:

```bash
python tools/check_cuda.py
```

This calls `cv2.cuda.getCudaEnabledDeviceCount()` and records the complete OpenCV
build information in `results/benchmark/cuda-preflight.json`. A nonpositive count
fails validation; GPU processing must not be claimed in that environment. A
positive count is followed by an actual GPU red-target smoke test. Full accuracy
and speed validation still require the parity tests and benchmark in [the CUDA
guide](cuda.md). Stop and share the report if the check fails.
