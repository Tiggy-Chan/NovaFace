"""Mathematical utilities and geometrical transformations for facial alignment.

Implements the Umeyama (Procrustes) similarity transform and affine mapping
directly from linear algebra first principles without external dependencies.
"""

from typing import Tuple
import numpy as np


# Standard canonical 5-point facial reference coordinates for 112x112 space:
# [left_eye, right_eye, nose, left_mouth_corner, right_mouth_corner]
CANONICAL_FACE_5PTS_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# Standard canonical 5-point facial reference coordinates for 128x128 space
# (scaled and centered for 128x128 models such as Inswapper):
CANONICAL_FACE_5PTS_128 = np.array(
    [
        [43.765257, 59.081486],
        [84.036343, 58.858743],
        [64.028800, 81.984686],
        [47.484914, 105.560571],
        [80.834171, 105.376114],
    ],
    dtype=np.float32,
)


def get_canonical_landmarks(size: int = 128) -> np.ndarray:
    """Retrieve canonical 5-point landmark template scaled to a given square dimension.

    Args:
        size: Target image square dimension (e.g. 112, 128, 256, 512).

    Returns:
        np.ndarray of shape (5, 2) representing the reference coordinates.
    """
    if size == 112:
        return CANONICAL_FACE_5PTS_112.copy()
    if size == 128:
        return CANONICAL_FACE_5PTS_128.copy()
    scale = size / 112.0
    return CANONICAL_FACE_5PTS_112 * scale


def compute_similarity_transform(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    estimate_scale: bool = True,
) -> np.ndarray:
    """Compute the 2x3 affine similarity transformation matrix via Umeyama SVD analysis.

    Finds optimal scale (s), orthogonal rotation (R), and translation (t) minimizing:
        sum_i || s * R * src_i + t - dst_i ||^2

    Args:
        src_points: Array of shape (N, 2), source landmark coordinates.
        dst_points: Array of shape (N, 2), target reference coordinates.
        estimate_scale: Whether to optimize scale factor (similarity) or rigid (rotation+trans).

    Returns:
        A 2x3 affine transformation matrix (np.ndarray of shape (2, 3), float32).
    """
    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)

    if src.ndim != 2 or src.shape[1] != 2 or src.shape != dst.shape:
        raise ValueError("Source and destination points must have shape (N, 2) with matching lengths.")

    num_pts = src.shape[0]
    if num_pts < 2:
        raise ValueError("At least 2 point correspondences required to solve similarity transform.")

    # Centroids
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)

    # Center coordinates
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    # Variances
    src_var = np.var(src, axis=0).sum()
    if src_var < 1e-12:
        raise ValueError("Source points have zero or near-zero variance; degenerate configuration.")

    # Cross-covariance matrix H = (dst_centered^T * src_centered) / num_pts
    covariance = np.dot(dst_centered.T, src_centered) / num_pts

    # Singular Value Decomposition
    u, singular_vals, vh = np.linalg.svd(covariance)
    v = vh.T

    # Reflection detection & correction matrix D
    d = np.eye(2, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        d[1, 1] = -1.0

    # Rotation matrix R = U * D * V^T
    rotation = np.dot(np.dot(u, d), vh)

    # Scale calculation
    if estimate_scale:
        scale = (1.0 / src_var) * np.trace(np.dot(np.diag(singular_vals), d))
    else:
        scale = 1.0

    # Translation vector t = dst_mean - s * R * src_mean
    translation = dst_mean - scale * np.dot(rotation, src_mean)

    # Construct standard 2x3 affine matrix: [s*R | t]
    matrix = np.zeros((2, 3), dtype=np.float32)
    matrix[:2, :2] = (scale * rotation).astype(np.float32)
    matrix[:2, 2] = translation.astype(np.float32)

    return matrix


def invert_affine_transform(affine_matrix: np.ndarray) -> np.ndarray:
    """Compute the analytical inverse of a 2x3 affine matrix.

    Given [A | b], where y = A*x + b, the inverse mapping is x = A^(-1)*(y - b) = [A^(-1) | -A^(-1)*b].

    Args:
        affine_matrix: np.ndarray of shape (2, 3).

    Returns:
        Inverse affine matrix of shape (2, 3), float32.
    """
    matrix = np.asarray(affine_matrix, dtype=np.float64)
    linear_part = matrix[:2, :2]
    translation_part = matrix[:2, 2]

    det = np.linalg.det(linear_part)
    if abs(det) < 1e-12:
        raise ValueError("Affine linear block is singular and cannot be inverted.")

    linear_inv = np.linalg.inv(linear_part)
    translation_inv = -np.dot(linear_inv, translation_part)

    inv_matrix = np.zeros((2, 3), dtype=np.float32)
    inv_matrix[:2, :2] = linear_inv.astype(np.float32)
    inv_matrix[:2, 2] = translation_inv.astype(np.float32)

    return inv_matrix


def transform_points(points: np.ndarray, affine_matrix: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine transformation matrix to an array of 2D coordinates.

    Args:
        points: np.ndarray of shape (N, 2).
        affine_matrix: np.ndarray of shape (2, 3).

    Returns:
        Transformed points of shape (N, 2).
    """
    pts = np.asarray(points, dtype=np.float32)
    mat = np.asarray(affine_matrix, dtype=np.float32)

    if pts.ndim == 1:
        pts = pts.reshape(1, 2)

    # [x, y, 1] dot [A | b]^T
    homogeneous = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
    transformed = np.dot(homogeneous, mat.T)
    return transformed
