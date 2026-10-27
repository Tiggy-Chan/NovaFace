"""Asynchronous multi-threaded live camera capture engine.

Decouples camera frame polling from neural inference pipeline,
utilizing zero-latency double buffering and AVFoundation on macOS.
"""

import platform
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import cv2
import numpy as np


@dataclass
class CameraConfig:
    """Hardware webcam or stream capture parameters."""

    device_id: Union[int, str] = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    backend: Optional[int] = None


class AsyncCameraCapture:
    """Threaded camera capture reader preventing pipeline stalling."""

    def __init__(self, config: Optional[CameraConfig] = None):
        """Initialize asynchronous camera grabber.

        Args:
            config: Camera configuration options.
        """
        self.config = config or CameraConfig()

        dev = self.config.device_id
        if isinstance(dev, str) and dev.isdigit():
            dev = int(dev)

        # Select backend: AVFoundation on macOS for local hardware
        backend = self.config.backend
        if isinstance(dev, int):
            if backend is None and platform.system() == "Darwin":
                backend = cv2.CAP_AVFOUNDATION
            self.cap = cv2.VideoCapture(dev, backend) if backend is not None else cv2.VideoCapture(dev)
        else:
            # URL / Stream (e.g. RTSP or HTTP IP camera)
            self.cap = cv2.VideoCapture(str(dev))

        # Set capture properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._new_frame_event = threading.Event()

    def start(self) -> "AsyncCameraCapture":
        """Start background camera capture thread."""
        if self._running:
            return self

        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open camera device ID {self.config.device_id}")

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraThread")
        self._thread.start()
        return self

    def _capture_loop(self) -> None:
        """Internal capture loop running in dedicated thread."""
        while self._running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            ts = time.perf_counter()
            with self._lock:
                self._latest_frame = frame
                self._latest_timestamp = ts

            self._new_frame_event.set()

    def read(self, timeout: float = 1.0) -> Tuple[bool, Optional[np.ndarray], float]:
        """Retrieve most recent frame without queuing latency.

        Args:
            timeout: Maximum wait time for a new frame.

        Returns:
            Tuple of (success_bool, frame_bgr, timestamp).
        """
        if not self._running:
            return False, None, 0.0

        if not self._new_frame_event.wait(timeout=timeout):
            with self._lock:
                return (self._latest_frame is not None), self._latest_frame, self._latest_timestamp

        with self._lock:
            self._new_frame_event.clear()
            frame = self._latest_frame.copy() if self._latest_frame is not None else None
            ts = self._latest_timestamp

        return (frame is not None), frame, ts

    def stop(self) -> None:
        """Stop background capture and release video hardware."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()

    def __enter__(self) -> "AsyncCameraCapture":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
