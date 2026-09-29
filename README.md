# Real-Time Embedded Optical Signal Detection

A CPU-only OpenCV MVP for detecting all valid red, green, or blue optical
blobs. No ML is used. Camera capture, detection, and visualization are separate.
A browser-based signal generator provides a controllable target on a phone.

```text
BGR frame → HSV → red/green/blue masks + brightness threshold
          → morphology → contours → area and confidence filters → all valid blobs
          → centroid, color, radius, confidence → overlay + JSONL
```

## Project goal

The project currently focuses on these stages:

- capture frames from the Raspberry Pi camera
- detect multiple red, green, and blue circular targets in real time
- calculate centroid, normalized position, area, radius, and brightness
- render a small deterministic overlay for debugging and validation
- keep detection and visualization separate for future experimentation

The first milestone focused on acquisition:

- initialize the Raspberry Pi camera with Picamera2
- capture frames at 640x480 resolution
- target 30 FPS
- convert frames into OpenCV-compatible BGR format
- print measured FPS once per second
- optionally display the live stream with `--display`
- release resources cleanly on `Ctrl+C`

## Folder structure

```text
embedded-optical-signal-detection/
├── README.md
├── requirements.txt
├── configs/
│   └── detection.yaml
├── src/
│   ├── pipeline.py
│   ├── capture/
│   │   ├── __init__.py
│   │   └── camera.py
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   └── types.py
│   └── visualization/
│       ├── __init__.py
│       └── overlay.py
├── tests/
│   └── test_detector.py
└── tools/
    └── signal_generator/
        └── index.html
```

## Setup

On Raspberry Pi OS, install the camera bindings through the system package manager,
then expose them to the virtual environment:

```bash
sudo apt install python3-picamera2
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On a desktop, image/video/demo modes do not require Picamera2:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the spectrum tracker MVP

Run all commands from the project root. The default source is the Raspberry Pi
camera at the configured resolution (640×480 by default), requesting 30 FPS:

```bash
python -m src.pipeline --display
```

Without a desktop/over SSH, omit `--display`. Press Ctrl+C to stop; in the live
preview, `q` or Escape also exits. A single-image preview waits for a key.

```bash
python -m src.pipeline --jsonl detections.jsonl
python -m src.pipeline --demo --display
python -m src.pipeline --demo --max-frames 90 --jsonl demo.jsonl --output demo.png
python -m src.pipeline --image test.jpg --output annotated.jpg
python -m src.pipeline --video recording.mp4 --display
python -m src.pipeline --webcam 0 --display
python -m src.pipeline --config configs/detection.yaml --fps 30 --display
```

`--output` saves the final annotated image; `--jsonl` writes one record per frame
with `count` and a `detections` array containing all valid blobs. The original
top-level detection fields still describe the largest blob for compatibility. These paths overwrite existing files. `--max-frames N` bounds a run;
video files stop at EOF. Console status is printed once per second, or once for a
still image:

```text
Targets: 2 | FPS: 29.4
#1 Color: RED | Position: (318, 221) | Radius: 14.0 px | Confidence: 0.89
#2 Color: BLUE | Position: (120, 180) | Radius: 10.0 px | Confidence: 0.86
```

Values above illustrate the format. FPS is measured pipeline throughput, starts at
zero until the first one-second measurement, and is not a hardware performance
guarantee. Video files are processed as fast as possible. Demo mode is paced by
`--fps` and cycles through moving red, green, and blue targets.

The preview labels every valid blob with its color, position, equivalent-area
radius, and confidence, plus frame FPS and target count. Labels are numbered by
area within each frame; these numbers are not persistent tracking IDs.
Missing targets show `NO TARGET`; JSONL has `count: 0`, `detections: []`, and
`valid: false` in the compatibility fields.

Use `Detector.detect_all(frame)` to obtain the list of targets, ordered by
descending area. The original `Detector.detect(frame)` API still returns only
the largest valid blob. Temporal association and spectral/wavelength estimation
are outside this MVP.

### Threshold tuning

Edit the YAML file passed with `--config` (otherwise `configs/detection.yaml`):

- `colors.*.hue_ranges`: OpenCV hue ranges (0–179); red wraps around the hue boundary.
- `saturation_min`: reject white/gray pixels.
- `value_min`: HSV brightness threshold (0–255); increase to reject dim backgrounds.
- `min_area` / `max_area`: valid contour area in pixels squared. Adjust for target
  distance and input resolution; image/video inputs retain their original size.
- `min_confidence`: minimum blob quality score (default 0, range 0–1). Set to
  0 to disable. Increase to reject weak/irregular background patches; decrease
  if real targets are dim, desaturated, or distorted by perspective.
- `morphology.open_kernel` / `close_kernel`: positive kernel sizes to remove specks
  and close small gaps (defaults 3 and 5).

All contours that pass the area and confidence filters are returned, including multiple blobs
of the same color. Overlapping color masks use red, then green, then blue
priority so the same pixels cannot produce duplicate detections. Touching blobs
of the same color can merge into one contour; keep targets separated.
Radius is `sqrt(area / pi)`. Confidence is a heuristic in [0, 1]:
`circularity × mean saturation / 255 × mean brightness / 255`, measured only on
each blob. It is not a calibrated detection probability. Blobs below
`min_confidence` are rejected and reported as `low_confidence` in debug output.
Bright, round colored background objects can still be detected;
start with a dark background and tune thresholds using the phone target.

Picamera2's `RGB888` arrays already contain BGR bytes, so capture passes them
straight to OpenCV; see the [official Picamera2 manual](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf).
External RGB arrays must be converted with `cv2.COLOR_RGB2BGR` before calling
`Detector.detect()`.

### Debug a missing red target

Keep a red target stationary on a black background, then run:

```bash
python -m src.pipeline --display --debug-dir captures/red-debug
```

The extra windows show the red HSV mask and the mask after morphology. Click the
red target in the main preview to print its original BGR/HSV pixel values and
which red thresholds it fails. Console diagnostics report matched pixel counts
and accepted, too-small, and too-large contours. On exit (`q` or Ctrl+C), the
last unannotated frame, all color masks, and `report.json` are saved in the chosen
directory, overwriting previous debug captures there. Debugging adds processing
overhead. Headless capture is also supported:

```bash
python -m src.pipeline --max-frames 90 --debug-dir captures/red-debug
```

Interpret the red diagnostics before changing `configs/detection.yaml`:

- No red pixels at the target: inspect clicked hue, saturation, and brightness.
  Defaults accept hue 0–12 or 168–179, saturation >= 80, and brightness >= 60.
- Present in the threshold mask but gone after morphology: the target is too
  thin/small for the opening kernel. Enlarge the displayed target or tune the kernel.
- `too_small`: contour area is below 50 camera pixels squared.
- `too_large`: contour area exceeds 5000 camera pixels squared. Inspect the mask
  for merged background regions before increasing this limit.

A red-looking preview alone does not establish which filter failed. The saved
raw frame can be replayed with `--image captures/red-debug/frame.png` while tuning.

### Camera-specific color calibration (pink background, shifted green/blue)

A uniform confidence cutoff can remove dim blue targets while keeping bright
background fragments. Confidence is diagnostic-only by default again. For a
camera whose green targets look yellow or blue targets look purple, use camera
samples instead of broadening HSV ranges:

1. Pause the generator on a scene containing at least one known red, green, and
   blue dot. Keep the camera, phone, and lighting fixed; hide generator controls.
2. Start the calibration preview:

   ```bash
   python -m src.pipeline --calibrate-colors configs/camera-colors.yaml
   ```

3. Let exposure settle, focus the preview window, then press **C** to freeze it.
   Click the **actual red**, **actual green**, and **actual blue** dot centers in
   that order. Label by the generator's source color, even if the camera makes it
   look yellow or purple. Then click three representative background patches
   (no dots), including the phone background and surrounding clutter.
4. Press **S** to save the YAML and exit. **U** undoes the last click;
   **Q/Escape** cancels calibration. If sampled classes are too similar, resample
   or improve lighting. Saving overwrites the specified YAML file.
5. Run the detector with the saved configuration:

   ```bash
   python -m src.pipeline --display --config configs/camera-colors.yaml --debug-dir captures/calibrated
   ```

Calibration uses deterministic distances to labeled color samples in OpenCV Lab
with reduced luminance weight; no ML model is trained. A pixel must be within
`color_distance_max` (30) of a target sample and closer to that class than any
other target/background class by `color_margin` (8). Calibrated matching replaces
HSV thresholds for all three colors. Unmatched and ambiguous pixels are excluded.

The saved profile also enables independent `min_circularity: 0.55` and
`min_color_contrast: 12` filters, comparing each blob with a nearby surrounding
ring. Contrast uses the same weighted Lab units, not a calibrated Delta E metric.
`min_confidence: 0` avoids penalizing a dim but distinct blue target merely for
being dim. All are starting parameters and may need adjustment for perspective,
target size, or lighting. Debug windows/reports now cover **all three colors**
and report `low_circularity` and `low_contrast` rejections.

Recalibrate after changing illumination, camera white balance, or screen settings.
Calibration cannot recover colors that the camera renders indistinguishably or
exclude background objects identical in color and shape to targets. Synthetic
color-shift/noise tests pass; live camera calibration is required before judging
accuracy. `--image` also supports calibration from a saved raw camera frame.

### Red background contamination
*This is most precise solution for pi and IMX219 camera module*
If most of the phone is white in the red debug mask, red targets are connected
to the background instead of forming separate contours. Try the stricter profile:

```bash
python -m src.pipeline --display --config configs/detection-red-background.yaml --debug-dir captures/red-debug
```

This raises only red `value_min` from 60 to 180 and `saturation_min` from 80 to
140. These are starting values, not calibration from your camera. Keep the phone
background black and targets bright. The desired red mask has isolated white
spots against a black phone background. Click both a target and its nearby
background in the main preview to compare HSV values. Set `value_min` above the
background V and below the target V; restart after editing the YAML. If their
values overlap, global brightness thresholding cannot reliably separate them.
Stricter thresholds may reject dim or desaturated targets. Do not increase
`max_area` to accept the entire merged phone region.

### Validation

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover the three colors, red hue wraparound, largest valid blob selection,
brightness rejection, selected-blob confidence, image/video execution, camera
channel order, and status rendering with no target. Camera hardware and GUI
preview require validation on the Raspberry Pi.

The original acquisition-only command remains available:

```bash
python -m src.capture.camera --display
```

## Optical Signal Generator

A browser-based optical signal generator is provided in:

```text
tools/signal_generator/index.html
```

It can display a configurable colored target that can be observed by the Raspberry Pi camera.

### Random multi-target scenes

Select **Random scenes**, enter the inclusive target count range **n–m** (0–200)
and the dwell time in seconds (0.1–3600), then press **套用並換幕** to apply.
Press **開始自動切換** to cycle scenes and the same button to pause. **下一幕**
advances immediately and restarts the dwell timer. Defaults are 1–5 targets and
3 seconds per scene; automatic switching starts only when requested.

Each scene randomly places non-overlapping circles, using mixed red/green/blue
colors or the selected fixed color. Brightness, radius, and background remain
controlled by their sliders/settings. Set n = m for a fixed count, or n = m = 0
for a blank scene. If the maximum count cannot fit, reduce m or the radius.
Resizing creates a new arrangement; returning from a background tab restarts the
full dwell time. Browser timers are approximate, not precision timing hardware.

Use **Experiment Mode** to hide the controls so they do not cover the targets;
click the screen or press Escape to return. **Manual target** retains the original
single-target position controls. Ground truth lists each target's color and
center in browser CSS pixels. The camera pipeline reports all valid blobs displayed by the page, subject to
the configured color, brightness, and area thresholds.

### Start the web server

From the project root:

```bash
cd tools/signal_generator
python3 -m http.server 8000 --bind 0.0.0.0
```

The HTTP server will serve `index.html` on port `8000`.

### Find the Raspberry Pi IP address

On the Raspberry Pi, run:

```bash
hostname -I
```

For example:

```text
192.168.1.193
```

### Open the signal generator on a phone

Make sure the phone and Raspberry Pi are connected to the same local network/Wi-Fi.

Then open a browser on the phone and navigate to:

```text
http://<RASPBERRY_PI_IP>:8000
```

For example:

```text
http://192.168.1.193:8000
```

Do not use `localhost:8000` on the phone. `localhost` would refer to the phone itself rather than the Raspberry Pi.

The resulting setup is:

```text
Phone
  │
  │ Wi-Fi / HTTP
  ▼
Raspberry Pi
  │
  ├── HTTP server :8000
  │     └── tools/signal_generator/index.html
  │
  └── Camera acquisition
        └── src/capture/camera.py
```

The phone can then be positioned in front of the Raspberry Pi camera and used as the optical signal source.

### Stop the web server

Press:

```text
Ctrl+C
```

in the terminal running the HTTP server.

## Notes on design

- The acquisition logic is encapsulated in `CameraAcquisition` so later stages can swap in a network sender or queue without changing the camera contract.
- The camera is configured for RGB888 because that matches the expected image format for simple OpenCV pipelines and avoids unnecessary conversion steps.
- FPS is computed using `time.perf_counter()` instead of wall-clock assumptions, which gives a more accurate measurement of the actual capture rate.
- Cleanup is intentionally explicit so the camera and OpenCV windows are released even when the user interrupts the program with `Ctrl+C`.
- The optical signal generator is kept separate from the acquisition pipeline so the phone can act as an independent ground-truth signal source.

## Future milestones

Future milestones will introduce:

- automatic ground-truth collection
- communication between the signal generator and detection pipeline
- experiment logging and evaluation
- networking to a GPU workstation
- optional GPU/CUDA acceleration
