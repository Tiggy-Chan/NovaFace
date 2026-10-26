"""Deep facial identity feature extraction and representation embedding.

Extracts a 512-dimensional normalized metric embedding from an aligned 112x112 face crop
using ArcFace / ResNet50 neural representations.
"""

from typing import Optional
import cv2
import numpy as np

from novaface.core.math_utils import compute_similarity_transform, get_canonical_landmarks
from novaface.inference.engine import DeviceConfig, OnnxSession


class ArcFaceEmbedder:
    """ArcFace 512-D identity representation extractor."""

    def __init__(self, model_path: str, device_config: Optional[DeviceConfig] = None):
        """Initialize ArcFace embedder.

        Args:
            model_path: Path to w600k_r50.onnx or compatible ArcFace ONNX model.
            device_config: Hardware acceleration config.
        """
        self.session = OnnxSession(model_path, config=device_config)
        self.canonical_5pts_112 = get_canonical_landmarks(112)

    def extract_embedding(self, image_bgr: np.ndarray, landmarks_5pts: np.ndarray) -> np.ndarray:
        """Align face to canonical 112x112 coordinates and compute normalized embedding vector.

        Args:
            image_bgr: Source image containing the face (H, W, 3).
            landmarks_5pts: 5 facial landmarks shape (5, 2) in image pixel coordinates.

        Returns:
            np.ndarray of shape (1, 512), normalized float32 feature vector (L2 norm = 1.0).
        """
        # 1. Similarity alignment to 112x112
        affine_matrix = compute_similarity_transform(landmarks_5pts, self.canonical_5pts_112)
        aligned_crop = cv2.warpAffine(
            image_bgr,
            affine_matrix,
            (112, 112),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # 2. Normalize to [-1.0, 1.0]: (pixel - 127.5) / 127.5
        blob = (aligned_crop.astype(np.float32) - 127.5) / 127.5
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, :, :, :]

        # 3. Forward pass
        input_name = self.session.input_names[0]
        embedding = self.session.run(input_feed={input_name: blob})[0]

        # 4. L2 Normalization
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-12)
        normalized_embedding = (embedding / norm).astype(np.float32)

        return normalized_embedding
