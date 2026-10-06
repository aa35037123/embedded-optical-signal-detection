# Method 3: Pi → NVIDIA PC with OpenCV CUDA

This backend is implemented but not hardware-validated in the development
environment, which has no usable CUDA device. Skipped GPU tests are not passes.

## Design

Use CUDA-enabled OpenCV first: its existing operations match this detector.
Custom CUDA/CuPy kernels could fuse operations later but add accuracy and
maintenance work. Benchmark before investing in them.

| Stage | Device |
| --- | --- |
| Camera NoIR tuning, capture, JPEG encoding | Pi |
| TCP reception and JPEG decoding | PC CPU |
| BGR upload | CPU → GPU |
| HSV conversion, HSV masks, red hue-range union, opening/closing | PC GPU |
| HSV and cleaned mask download | GPU → CPU |
| Color ownership, contours, size/shape/contrast/confidence filtering | PC CPU |
| Display and CSV | PC CPU |

The shared Detector.detect_from_masks() preserves the existing contour filters.
GPU HSV rounding and border morphology can differ; test accuracy below.
CUDA accepts HSV configs including configs/detection-noir.yaml. It explicitly
rejects color_samples_bgr calibration configs. There is no silent CPU fallback.

## Phase 1: Build on the NVIDIA PC (Linux)

Keep the Pi environment unchanged. Check the NVIDIA driver and CUDA toolkit:

```bash
nvidia-smi
nvcc --version
```

If either fails, stop and fix installation for your GPU and OS using the
[NVIDIA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
cuDNN is not required for this detector. Compiler/toolkit compatibility matters.

Standard opencv-python and opencv-contrib-python wheels are CPU-only;
installing the toolkit alongside them does not enable CUDA. See the
[official OpenCV Python instructions](https://github.com/opencv/opencv-python).

Create a separate environment from the project root:

```bash
python3 -m venv .venv-cuda
source .venv-cuda/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cuda.txt
```

Do not install requirements.txt here: it installs the CPU OpenCV wheel.
If this interpreter already has a CUDA-enabled OpenCV build, go to Phase 2.
Otherwise, here is a Debian/Ubuntu source-build example. Use matching OpenCV and
contrib tags compatible with your toolkit/compiler; 4.13.0 is shown here.

```bash
sudo apt install build-essential cmake ninja-build git python3-dev \
  libgtk-3-dev libjpeg-dev libpng-dev libtiff-dev libavcodec-dev \
  libavformat-dev libswscale-dev

mkdir -p /tmp/optical-opencv-build
git clone --branch 4.13.0 --depth 1 https://github.com/opencv/opencv.git /tmp/optical-opencv-build/opencv
git clone --branch 4.13.0 --depth 1 https://github.com/opencv/opencv_contrib.git /tmp/optical-opencv-build/opencv_contrib

cmake -S /tmp/optical-opencv-build/opencv -B /tmp/optical-opencv-build/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$VIRTUAL_ENV" \
  -DOPENCV_EXTRA_MODULES_PATH=/tmp/optical-opencv-build/opencv_contrib/modules \
  -DWITH_CUDA=ON -DWITH_CUDNN=OFF -DOPENCV_DNN_CUDA=OFF \
  -DBUILD_LIST=core,imgproc,imgcodecs,highgui,videoio,python3,cudev,cudaarithm,cudaimgproc,cudafilters \
  -DBUILD_opencv_python3=ON \
  -DPYTHON3_EXECUTABLE="$(command -v python)" \
  -DPYTHON3_PACKAGES_PATH="$(python -c 'import sysconfig; print(sysconfig.get_path("platlib"))')" \
  -DPYTHON3_NUMPY_INCLUDE_DIRS="$(python -c 'import numpy; print(numpy.get_include())')" \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF

cmake --build /tmp/optical-opencv-build/build --parallel 4
cmake --install /tmp/optical-opencv-build/build
```

Before building, check CMake's summary: CUDA must be YES, CUDA modules enabled,
and Python must point into .venv-cuda. If required by your toolkit, add
-DCUDA_ARCH_BIN=<your GPU compute capability>; do not copy a different GPU's
architecture. Avoid fast-math flags during accuracy validation. Installation is
inside the virtual environment and does not require sudo.

## Phase 2: Validate before streaming

With .venv-cuda active, from the project root:

```bash
python -c 'import cv2; print(cv2.__file__); print(cv2.cuda.getCudaEnabledDeviceCount())'
python tools/check_cuda.py
python -m pytest tests/test_cuda_backend.py -v
```

Preflight must pass a real GPU red-target operation, not merely count devices.
All GPU parity cases must run and pass, not skip. They compare CPU/GPU target
count, color, position and area using synthetic targets and saved camera frames,
including border targets and changing frame sizes.

**If anything fails or skips, stop and share the error and
results/benchmark/cuda-preflight.json before continuing.**

## Phase 3: Launch

Start the receiver on the **PC**:

```bash
source .venv-cuda/bin/activate
python -m src.network.receiver --bind 0.0.0.0 --port 5000 \
  --backend cuda --display --config configs/detection-noir.yaml \
  --csv results/network/cuda.csv
```

Then start the sender on the **Pi**, closing other camera processes first:

```bash
source .venv/bin/activate
python -m src.network.sender --host YOUR_PC_IP --port 5000 \
  --width 640 --height 480 --fps 30 --jpeg-quality 85 \
  --tuning-file /usr/share/libcamera/ipa/rpi/vc4/imx219_noir.json
```

Use pisp instead of vc4 on Pi 5. Tuning belongs on the Pi; detector YAML belongs
on the PC. Edit colors.red on the PC and restart its receiver to adjust red.
The receiver reports backend=cuda. Allow TCP port 5000 from the Pi as in Method 2.

## Phase 4: Benchmark

Use an unannotated saved frame from the corrected camera:

```bash
python tools/benchmark_backends.py --image captures/noir-debug/frame.png \
  --config configs/detection-noir.yaml
```

Both backends warm up, then process the same decoded pixels 200 times. Total
latency includes GPU transfers and CPU contour work, excluding network, JPEG and
display. cpu_over_cuda_speedup above 1 means CUDA was faster for this input.
Validate live detection too; this single-image benchmark is not an accuracy test.

For streaming comparison repeat with --backend cpu and --csv results/network/cpu.csv,
using the same scene, resolution, JPEG quality and display setting.

- upload_ms / download_ms: synchronized transfers.
- processing_ms: synchronized GPU operations plus CPU contour/contrast work,
  excluding transfers; on CPU this is the whole detector call.
- total_workstation_ms: JPEG decode + backend + visualization/display.
- local_receive_to_done_ms: additionally includes local queue waiting.

At 640×480, launch/transfer costs and remaining CPU work may outweigh GPU savings.
Both methods can also hit the sender's 30 FPS limit. Compare latency and timings,
not just displayed FPS. Latest-frame dropping can select different frame IDs in
network runs; use the offline benchmark to compare identical pixels.

## Files

- src/detection/cuda_backend.py: GPU operations and synchronized timings.
- src/detection/backends.py: CPU/CUDA selection.
- src/detection/detector.py: shared contour measurement and rejection.
- src/network/receiver.py: backend flag; transport is unchanged.
- tools/check_cuda.py: device, binding and operation checks.
- tests/test_cuda_backend.py: hardware parity tests.
- tools/benchmark_backends.py: identical-input performance comparison.

References: [CUDA morphology](https://docs.opencv.org/4.13.0/dc/d66/group__cudafilters.html),
[stream synchronization](https://docs.opencv.org/4.13.0/d9/df3/classcv_1_1cuda_1_1Stream.html).
