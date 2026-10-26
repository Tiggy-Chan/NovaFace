"""Temporal smoothing algorithms for real-time video and facial landmark tracking.

Implements the One-Euro Filter (Casiez et al., CHI 2012) designed specifically
to eliminate high-frequency jitter while preserving low latency during rapid movement.
"""

import math
import time
from typing import Optional, Union
import numpy as np


class LowPassFilter:
    """First-order low-pass filter with exponential smoothing."""

    def __init__(self, alpha: float = 0.5, init_val: Optional[np.ndarray] = None):
        self._alpha = alpha
        self._s: Optional[np.ndarray] = None
        if init_val is not None:
            self._s = np.copy(init_val)

    def filter(self, value: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
        if alpha is not None:
            self._alpha = alpha
        if self._s is None:
            self._s = np.copy(value)
        else:
            self._s = self._alpha * value + (1.0 - self._alpha) * self._s
        return self._s

    def last_value(self) -> Optional[np.ndarray]:
        return self._s

    def reset(self, val: Optional[np.ndarray] = None) -> None:
        self._s = np.copy(val) if val is not None else None


class OneEuroFilter:
    """One-Euro filter for multidimensional signals (e.g. facial landmarks, bboxes).

    Adapts the cutoff frequency dynamically based on signal velocity:
    - At low speeds, cutoff frequency is low (f_min) -> suppresses static micro-jitter.
    - At high speeds, cutoff frequency increases proportionally to velocity (beta) -> eliminates phase lag.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ):
        """Initialize One-Euro Filter.

        Args:
            min_cutoff: Minimum cutoff frequency in Hz (lower = smoother when stationary).
            beta: Speed coefficient (higher = faster response during quick movement).
            d_cutoff: Cutoff frequency for the derivative filter in Hz.
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self._x_filter = LowPassFilter()
        self._dx_filter = LowPassFilter()
        self._last_time: Optional[float] = None

    @staticmethod
    def _alpha(rate: float, cutoff: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate
        return 1.0 / (1.0 + tau / te)

    def filter(
        self,
        value: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> np.ndarray:
        """Filter incoming measurement array.

        Args:
            value: np.ndarray of arbitrary shape (e.g. (5, 2) or (4,)).
            timestamp: Time of arrival in seconds. If None, uses current system time.

        Returns:
            Smoothed array of identical shape and dtype.
        """
        val = np.asarray(value, dtype=np.float32)
        now = timestamp if timestamp is not None else time.perf_counter()

        if self._last_time is None or self._x_filter.last_value() is None:
            self._last_time = now
            self._x_filter.reset(val)
            self._dx_filter.reset(np.zeros_like(val))
            return val

        dt = now - self._last_time
        self._last_time = now

        # Protect against non-monotonic or zero dt
        if dt <= 1e-6:
            return self._x_filter.last_value() or val

        rate = 1.0 / dt

        # Estimate derivative (velocity)
        prev_x = self._x_filter.last_value()
        dx = (val - prev_x) * rate
        dx_hat = self._dx_filter.filter(dx, self._alpha(rate, self.d_cutoff))

        # Dynamic cutoff frequency based on velocity magnitude
        speed = np.abs(dx_hat)
        cutoff = self.min_cutoff + self.beta * speed

        # Filter signal
        alpha_val = self._alpha(rate, cutoff)
        return self._x_filter.filter(val, alpha_val)

    def reset(self) -> None:
        """Reset internal filter state."""
        self._x_filter.reset()
        self._dx_filter.reset()
        self._last_time = None


class LandmarkSmoother:
    """Stateful temporal smoother for face tracking landmarks and bounding box."""

    def __init__(
        self,
        min_cutoff: float = 0.8,
        beta: float = 0.015,
        jump_threshold: float = 80.0,
    ):
        """Initialize LandmarkSmoother.

        Args:
            min_cutoff: Minimum cutoff frequency for OneEuroFilter.
            beta: Velocity tuning factor.
            jump_threshold: Euclidean distance threshold (in pixels) beyond which
                a tracking jump / identity change is suspected, prompting filter reset.
        """
        self.jump_threshold = float(jump_threshold)
        self._filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self._prev_landmarks: Optional[np.ndarray] = None

    def update(self, landmarks: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        """Process incoming 5-point landmarks and return jitter-free smoothed landmarks."""
        pts = np.asarray(landmarks, dtype=np.float32)

        # Detect discontinuity (e.g. tracking jumped to a different face or position)
        if self._prev_landmarks is not None:
            shift = np.linalg.norm(pts - self._prev_landmarks, axis=-1).mean()
            if shift > self.jump_threshold:
                self._filter.reset()

        smoothed = self._filter.filter(pts, timestamp=timestamp)
        self._prev_landmarks = smoothed.copy()
        return smoothed

    def reset(self) -> None:
        self._filter.reset()
        self._prev_landmarks = None
