#!/usr/bin/env python3
"""Minimal camera acquisition for the Raspberry Pi milestone.

The goal of this module is to keep acquisition concerns separate from later
networking or signal-processing work. The camera setup, frame conversion, and
measurement logic are intentionally isolated in a small class so future code can
replace the local loop with a queue or socket sender without changing the camera
contract.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np


class CameraAcquisition:
    """Thin wrapper around Picamera2 for a simple, testable capture loop.

    The class exposes just enough behavior for milestone 1: initialize the camera,
    capture RGB frames, measure real FPS, and optionally display frames. The
    acquisition loop is separated from the display/cleanup code so it stays easy to
    extend later for buffering or streaming to a workstation.
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.width = width
        self.height = height
        self.target_fps = fps
        from picamera2 import Picamera2

        self.camera = Picamera2()
        self._running = False

    def configure(self) -> None:
        """Set a fixed RGB configuration for the camera.

        Using RGB888 keeps the frame layout simple for downstream OpenCV processing
        and any later serialization over the network. The 640x480 size keeps the
        workload light enough for the Pi during early-stage development while still
        matching typical embedded acquisition constraints.
        """
        config = self.camera.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameRate": self.target_fps},
        )
        self.camera.configure(config)

    def start(self) -> None:
        """Start the camera and wait for the first frame to settle."""
        self.configure()
        self.camera.start()
        self._running = True
        time.sleep(0.2)

    def _to_opencv_frame(self, frame: np.ndarray) -> np.ndarray:
        """RGB888 in Picamera2 already provides BGR bytes for OpenCV."""
        return np.ascontiguousarray(frame)

    def read(self) -> np.ndarray:
        """Capture one BGR frame, starting acquisition if needed."""
        if not self._running:
            self.start()
        return self._to_opencv_frame(self.camera.capture_array())

    def capture_loop(self, display: bool = False) -> None:
        """Run the main acquisition loop until Ctrl+C is pressed."""
        if not self._running:
            self.start()

        window_start = time.perf_counter()
        frames_in_window = 0

        try:
            while True:
                loop_start = time.perf_counter()
                raw_frame = self.camera.capture_array()
                frame = self._to_opencv_frame(raw_frame)
                frames_in_window += 1

                if display:
                    cv2.imshow("Pi Camera", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                elapsed = time.perf_counter() - window_start
                if elapsed >= 1.0:
                    actual_fps = frames_in_window / elapsed
                    print(f"Actual FPS: {actual_fps:.2f}")
                    window_start = time.perf_counter()
                    frames_in_window = 0

                # A small sleep keeps the loop near the desired 30 FPS without
                # starving the CPU. The actual result is measured independently, so
                # we can tell if the camera is truly keeping up.
                sleep_for = (1.0 / self.target_fps) - (time.perf_counter() - loop_start)
                if sleep_for > 0:
                    time.sleep(sleep_for)

        except KeyboardInterrupt:
            print("\nStopping camera acquisition.")
        finally:
            self.stop()

    def stop(self) -> None:
        """Cleanly release both OpenCV windows and the camera device."""
        try:
            if cv2.getWindowProperty("Pi Camera", cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyAllWindows()
        except cv2.error:
            # The window may not exist if the user did not request display output.
            pass

        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass
            try:
                self.camera.close()
            except Exception:
                pass
        self._running = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture RGB frames from the Raspberry Pi camera.")
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display each captured frame in an OpenCV window.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Frame width in pixels (default: 640).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Frame height in pixels (default: 480).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target camera FPS (default: 30).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Starting acquisition at {args.width}x{args.height} @ {args.fps} FPS")
    acquisition = CameraAcquisition(width=args.width, height=args.height, fps=args.fps)

    try:
        acquisition.capture_loop(display=args.display)
    except Exception as exc:  # pragma: no cover - safety net for runtime issues.
        print(f"Camera acquisition failed: {exc}")
        acquisition.stop()
        raise


if __name__ == "__main__":
    main()
