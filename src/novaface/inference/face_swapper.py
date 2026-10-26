"""Neural face swapping engine wrapper.

Executes latent identity substitution using Inswapper architecture,
combining target spatial texture with source biometric identity representations.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from novaface.core.math_utils import compute_similarity_transform, get_canonical_landmarks
from novaface.inference.engine import DeviceConfig, OnnxSession


class InswapperModel:
    """Inswapper-128 neural face replacement model."""

    def __init__(self, model_path: str, device_config: Optional[DeviceConfig] = None):
        """Initialize Inswapper neural engine.

        Args:
            model_path: Path to inswapper_128_fp16.onnx or compatible checkpoint.
            device_config: Hardware acceleration preferences.
        """
        self.session = OnnxSession(model_path, config=device_config)
        self.canonical_5pts_128 = get_canonical_landmarks(128)

        # Detect model input names
        self.target_name = "target" if "target" in self.session.input_names else self.session.input_names[0]
        self.source_name = "source" if "source" in self.session.input_names else self.session.input_names[1]

    def crop_target_face(
        self,
        frame_bgr: np.ndarray,
        landmarks_5pts: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align and crop 128x128 face patch from full target frame.

        Args:
            frame_bgr: Full image / video frame (H, W, 3).
            landmarks_5pts: Target 5 facial landmarks shape (5, 2).

        Returns:
            Tuple of (cropped_patch_128, affine_matrix_2x3).
        """
        affine_matrix = compute_similarity_transform(landmarks_5pts, self.canonical_5pts_128)
        cropped_patch = cv2.warpAffine(
            frame_bgr,
            affine_matrix,
            (128, 128),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return cropped_patch, affine_matrix

    def swap_face(
        self,
        cropped_target_128: np.ndarray,
        source_embedding_512: np.ndarray,
    ) -> np.ndarray:
        """Replace identity of target 128x128 face patch with source identity embedding.

        Args:
            cropped_target_128: Aligned target face patch (128, 128, 3) in uint8 BGR.
            source_embedding_512: Normalized source embedding vector of shape (1, 512).

        Returns:
            Swapped face patch of shape (128, 128, 3) in uint8 BGR.
        """
        # Normalize target image to [0.0, 1.0] and shape (1, 3, 128, 128)
        blob_target = (cropped_target_128.astype(np.float32) / 255.0)
        blob_target = np.transpose(blob_target, (2, 0, 1))[np.newaxis, :, :, :]

        feed_dict = {
            self.target_name: blob_target,
            self.source_name: source_embedding_512.astype(np.float32),
        }

        output_tensor = self.session.run(input_feed=feed_dict)[0]

        # Output shape: (1, 3, 128, 128) in range [0.0, 1.0]
        swapped_bgr = np.transpose(output_tensor[0], (1, 2, 0))
        swapped_u8 = np.clip(swapped_bgr * 255.0, 0.0, 255.0).astype(np.uint8)
        return swapped_u8
