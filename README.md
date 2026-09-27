# Real-Time Embedded Optical Signal Detection

This repository contains the initial milestone for a Raspberry Pi-based camera acquisition pipeline. The goal of this stage is to prove that the Raspberry Pi camera can capture RGB frames at a stable rate and expose them in an OpenCV-friendly format without yet introducing detection logic, networking, or GPU processing.

The repository also includes a browser-based optical signal generator that can be opened on a phone and used as a controllable visual target for the Raspberry Pi camera.

## Project goal

The first milestone focuses on acquisition:

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
├── src/
│   └── capture/
│       ├── __init__.py
│       └── camera.py
└── tools/
    └── signal_generator/
        └── index.html
```

## Setup

From the project root, create a virtual environment with access to the Raspberry Pi system packages:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the acquisition loop

Run with an OpenCV preview window:

```bash
python3 src/capture/camera.py --display
```

Run without the OpenCV window:

```bash
python3 src/capture/camera.py
```

## Optical Signal Generator

A browser-based optical signal generator is provided in:

```text
tools/signal_generator/index.html
```

It can display a configurable colored target that can be observed by the Raspberry Pi camera.

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

- optical signal detection
- automatic ground-truth collection
- communication between the signal generator and detection pipeline
- experiment logging and evaluation
- networking to a GPU workstation
- optional GPU/CUDA acceleration