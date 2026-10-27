"""End-to-End Face Swapping Pipeline.

Coordinates detection, feature extraction, neural latent identity replacement,
temporal tracking, color grading, and frame compositing.
"""

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union
import cv2
import numpy as np

from novaface.core.blender import FrameBlender
from novaface.inference.engine import DeviceConfig
from novaface.inference.face_detector import DetectedFace, SCRFDFaceDetector
from novaface.inference.face_embedder import ArcFaceEmbedder
from novaface.inference.face_swapper import InswapperModel
from novaface.inference.model_manager import ModelManager, ModelPaths
from novaface.pipeline.tracker import TemporalFaceTracker, TrackedFace


@dataclass
class PipelineConfig:
    """Configuration options for FaceSwapPipeline execution."""

    detector_score_threshold: float = 0.5
    nms_threshold: float = 0.4
    enable_color_transfer: bool = False
    color_blend_ratio: float = 0.85
    temporal_smoothing: bool = True
    target_face_index: int = 0  # 0: largest face, -1: all faces
    device_config: Optional[DeviceConfig] = None


class FaceSwapPipeline:
    """High-performance orchestrator for image and video face swapping."""

    def __init__(
        self,
        model_paths: Optional[ModelPaths] = None,
        config: Optional[PipelineConfig] = None,
    ):
        """Initialize pipeline with neural network models and compositors.

        Args:
            model_paths: Pre-resolved ModelPaths or None to auto-discover.
            config: Pipeline configuration preferences.
        """
        self.config = config or PipelineConfig()
        self.paths = model_paths or ModelManager.resolve_all_models()

        # Initialize neural modules
        self.detector = SCRFDFaceDetector(
            self.paths.detector_path,
            device_config=self.config.device_config,
        )
        self.embedder = ArcFaceEmbedder(
            self.paths.embedder_path,
            device_config=self.config.device_config,
        )
        self.swapper = InswapperModel(
            self.paths.swapper_path,
            device_config=self.config.device_config,
        )

        # Compositor & Tracker
        self.blender = FrameBlender(
            enable_color_transfer=self.config.enable_color_transfer,
            color_blend_ratio=self.config.color_blend_ratio,
        )
        self.tracker = TemporalFaceTracker()

        # State
        self.source_embedding: Optional[np.ndarray] = None
        self.source_image: Optional[np.ndarray] = None

    def set_source_face(self, source: Union[str, np.ndarray]) -> np.ndarray:
        """Load source identity image, detect face, and extract identity embedding vector.

        Args:
            source: File path (str) or loaded BGR image (np.ndarray).

        Returns:
            Extracted 512-dimensional normalized embedding vector.
        """
        if isinstance(source, str):
            if not os.path.isfile(source):
                raise FileNotFoundError(f"Source face image not found at: {source}")
            img = cv2.imread(source)
            if img is None:
                raise ValueError(f"Failed to decode image from: {source}")
        else:
            img = source

        faces = self.detector.detect(img, score_threshold=self.config.detector_score_threshold)
        if not faces:
            raise ValueError("No human face detected in the provided source image.")

        # Pick the most prominent face (highest confidence / largest area)
        primary_face = max(faces, key=lambda f: f.score * f.area)

        # Extract 512-D identity embedding
        embedding = self.embedder.extract_embedding(img, primary_face.landmarks)
        self.source_embedding = embedding
        self.source_image = img.copy()

        return embedding

    def swap_image(self, target: Union[str, np.ndarray]) -> np.ndarray:
        """Perform face swapping on a single still image.

        Args:
            target: Target image file path or BGR numpy array.

        Returns:
            Rendered composite BGR numpy array.
        """
        if self.source_embedding is None:
            raise RuntimeError("Source face identity has not been set. Call set_source_face() first.")

        if isinstance(target, str):
            if not os.path.isfile(target):
                raise FileNotFoundError(f"Target image not found at: {target}")
            frame = cv2.imread(target)
            if frame is None:
                raise ValueError(f"Failed to decode image from: {target}")
        else:
            frame = target.copy()

        faces = self.detector.detect(
            frame,
            score_threshold=self.config.detector_score_threshold,
            nms_threshold=self.config.nms_threshold,
        )

        if not faces:
            return frame

        # Select which faces to swap
        if self.config.target_face_index == -1:
            targets_to_swap = faces
        else:
            # Sort by area descending
            sorted_faces = sorted(faces, key=lambda f: f.area, reverse=True)
            idx = min(self.config.target_face_index, len(sorted_faces) - 1)
            targets_to_swap = [sorted_faces[idx]]

        output_frame = frame
        for face in targets_to_swap:
            crop_patch, affine_m = self.swapper.crop_target_face(output_frame, face.landmarks)
            swapped_patch = self.swapper.swap_face(crop_patch, self.source_embedding)
            output_frame = self.blender.composite_face(
                target_frame=output_frame,
                cropped_target_patch=crop_patch,
                swapped_face_patch=swapped_patch,
                affine_matrix=affine_m,
            )

        return output_frame

    def swap_frame(self, frame_bgr: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        """Process a continuous video stream frame with temporal landmark stabilization.

        Args:
            frame_bgr: Input video frame (H, W, 3) in uint8 BGR.
            timestamp: Presentation timestamp in seconds.

        Returns:
            Processed video frame in uint8 BGR.
        """
        if self.source_embedding is None:
            return frame_bgr

        # 1. Detect faces
        detections = self.detector.detect(
            frame_bgr,
            score_threshold=self.config.detector_score_threshold,
            nms_threshold=self.config.nms_threshold,
        )

        # 2. Temporal tracking and landmark smoothing
        if self.config.temporal_smoothing:
            tracked_faces = self.tracker.update(detections, timestamp=timestamp)
            faces_to_process = [
                (t.landmarks if self.config.temporal_smoothing else t.raw_landmarks)
                for t in tracked_faces
            ]
        else:
            faces_to_process = [d.landmarks for d in detections]

        if not faces_to_process:
            return frame_bgr

        # 3. Select target face
        if self.config.target_face_index == -1:
            active_landmarks = faces_to_process
        else:
            active_landmarks = [faces_to_process[0]]

        # 4. Swap and composite
        output_frame = frame_bgr
        for lms in active_landmarks:
            crop_patch, affine_m = self.swapper.crop_target_face(output_frame, lms)
            swapped_patch = self.swapper.swap_face(crop_patch, self.source_embedding)
            output_frame = self.blender.composite_face(
                target_frame=output_frame,
                cropped_target_patch=crop_patch,
                swapped_face_patch=swapped_patch,
                affine_matrix=affine_m,
            )

        return output_frame

    def swap_video(
        self,
        input_video_path: str,
        output_video_path: str,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> str:
        """Process an entire video file, performing face swapping frame by frame and preserving audio.

        Args:
            input_video_path: Path to input .mp4/.mkv/.mov video file.
            output_video_path: Destination path for rendered video.
            progress_callback: Optional callback func(current_frame, total_frames, current_fps).

        Returns:
            Absolute path to the final output video.
        """
        if self.source_embedding is None:
            raise RuntimeError("Source face identity has not been set.")

        if not os.path.isfile(input_video_path):
            raise FileNotFoundError(f"Input video not found at: {input_video_path}")

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open input video: {input_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        temp_dir = os.path.dirname(os.path.abspath(output_video_path))
        os.makedirs(temp_dir, exist_ok=True)
        temp_video_path = os.path.join(temp_dir, f"_temp_swap_{os.path.basename(output_video_path)}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

        self.tracker.reset()
        frame_idx = 0
        start_time = time.perf_counter()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                ts = frame_idx / fps
                swapped = self.swap_frame(frame, timestamp=ts)
                writer.write(swapped)

                frame_idx += 1
                if progress_callback and frame_idx % 5 == 0:
                    elapsed = max(1e-5, time.perf_counter() - start_time)
                    cur_fps = frame_idx / elapsed
                    progress_callback(frame_idx, total_frames, cur_fps)
        finally:
            cap.release()
            writer.release()

        # Remux with audio via ffmpeg if available
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-i", temp_video_path,
                "-i", input_video_path,
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-shortest",
                output_video_path,
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0 and os.path.exists(output_video_path):
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
                return os.path.abspath(output_video_path)

        # Fallback if ffmpeg is unavailable or fails
        if os.path.exists(temp_video_path):
            if os.path.exists(output_video_path):
                os.remove(output_video_path)
            shutil.move(temp_video_path, output_video_path)

        return os.path.abspath(output_video_path)
