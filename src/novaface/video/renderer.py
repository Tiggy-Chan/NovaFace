"""Real-time display window with heads-up display (HUD) statistics and user controls.

Renders composited frames with telemetry overlays (FPS, resolution, hardware usage)
and interactive keyboard shortcut handling.
"""

import os
import time
from typing import Optional, Tuple
import cv2
import numpy as np


class LivePreviewWindow:
    """Interactive GUI preview window with real-time telemetry overlay."""

    def __init__(self, window_name: str = "NovaFace Real-Time Studio", show_hud: bool = True):
        """Initialize preview window.

        Args:
            window_name: Title bar text for the GUI window.
            show_hud: Whether to render performance statistics overlay.
        """
        self.window_name = window_name
        self.show_hud = show_hud
        self._is_open = False

        # Moving average FPS calculation
        self._prev_time = time.perf_counter()
        self._fps_history = []

    def open(self) -> None:
        """Create and configure named OpenCV high-gui window."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        self._is_open = True

    def render(
        self,
        frame_bgr: np.ndarray,
        custom_fps: Optional[float] = None,
        status_text: Optional[str] = None,
    ) -> int:
        """Draw HUD statistics and display frame.

        Args:
            frame_bgr: Image frame to render (H, W, 3).
            custom_fps: Explicit FPS value, or computed automatically if None.
            status_text: Optional auxiliary status description string.

        Returns:
            Keycode of any key pressed during waitKey(1), or -1.
        """
        if not self._is_open:
            self.open()

        now = time.perf_counter()
        dt = now - self._prev_time
        self._prev_time = now

        if dt > 0:
            current_fps = 1.0 / dt
            self._fps_history.append(current_fps)
            if len(self._fps_history) > 30:
                self._fps_history.pop(0)

        avg_fps = custom_fps if custom_fps is not None else (
            sum(self._fps_history) / len(self._fps_history) if self._fps_history else 0.0
        )

        display_frame = frame_bgr
        if self.show_hud:
            display_frame = frame_bgr.copy()
            h, w = display_frame.shape[:2]

            # Top HUD Bar
            hud_text = f"FPS: {avg_fps:.1f} | Res: {w}x{h}"
            if status_text:
                hud_text += f" | {status_text}"

            # Dark translucent banner
            cv2.rectangle(display_frame, (0, 0), (w, 32), (20, 20, 20), -1)
            cv2.putText(
                display_frame,
                hud_text,
                (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 180),
                1,
                cv2.LINE_AA,
            )

            # Bottom Controls Hint
            controls_text = "[Q] Quit  [S] Save Frame  [H] Toggle HUD"
            cv2.rectangle(display_frame, (0, h - 24), (w, h), (20, 20, 20), -1)
            cv2.putText(
                display_frame,
                controls_text,
                (12, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow(self.window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("h") or key == ord("H"):
            self.show_hud = not self.show_hud
        elif key == ord("s") or key == ord("S"):
            ts_str = time.strftime("%Y%m%d_%H%M%S")
            save_path = f"novaface_snapshot_{ts_str}.jpg"
            cv2.imwrite(save_path, frame_bgr)
            print(f"Snapshot saved to: {save_path}")

        return key

    def close(self) -> None:
        """Destroy window and release GUI resources."""
        if self._is_open:
            cv2.destroyWindow(self.window_name)
            self._is_open = False
