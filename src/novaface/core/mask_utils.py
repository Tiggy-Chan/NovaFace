"""Facial mask synthesis and feathering utilities for boundary-free blending.

Generates smooth Alpha masks based on facial geometry, convex hulls, and
distance transform gradients to eliminate hard seams during face substitution.
"""

from typing import Optional, Tuple
import cv2
import numpy as np


def create_elliptical_mask(
    size: Tuple[int, int] = (128, 128),
    center_offset: Tuple[float, float] = (0.0, -0.05),
    radius_ratio: Tuple[float, float] = (0.38, 0.46),
    blur_sigma: float = 12.0,
) -> np.ndarray:
    """Generate a smooth elliptical soft mask centered on the face.

    Args:
        size: (width, height) of the square or rectangular face crop.
        center_offset: Normalized offset (dx, dy) from image center (-1 to 1).
        radius_ratio: Ratio of ellipse radii (rx, ry) relative to crop dimensions.
        blur_sigma: Gaussian blur sigma for boundary feathering.

    Returns:
        np.ndarray of shape (height, width), float32 in range [0.0, 1.0].
    """
    w, h = size
    mask = np.zeros((h, w), dtype=np.float32)

    cx = int(w * (0.5 + center_offset[0]))
    cy = int(h * (0.5 + center_offset[1]))
    rx = int(w * radius_ratio[0])
    ry = int(h * radius_ratio[1])

    cv2.ellipse(
        mask,
        center=(cx, cy),
        axes=(rx, ry),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=1.0,
        thickness=-1,
    )

    if blur_sigma > 0:
        ksize = int(blur_sigma * 4) | 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), blur_sigma)
        mask = np.clip(mask, 0.0, 1.0)

    return mask.astype(np.float32)


def create_convex_hull_mask(
    landmarks: np.ndarray,
    size: Tuple[int, int] = (128, 128),
    erosion_iter: int = 1,
    blur_sigma: float = 8.0,
) -> np.ndarray:
    """Generate a soft mask by computing the convex hull of facial landmarks.

    Args:
        landmarks: Landmark coordinates of shape (N, 2).
        size: (width, height) of the target mask canvas.
        erosion_iter: Number of morphological erosion iterations to shrink edge.
        blur_sigma: Gaussian blur radius for edge softness.

    Returns:
        np.ndarray of shape (height, width), float32 in range [0.0, 1.0].
    """
    w, h = size
    mask = np.zeros((h, w), dtype=np.uint8)

    pts = np.asarray(landmarks, dtype=np.int32).reshape((-1, 1, 2))
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)

    if erosion_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.erode(mask, kernel, iterations=erosion_iter)

    mask_f = mask.astype(np.float32) / 255.0

    if blur_sigma > 0:
        ksize = int(blur_sigma * 4) | 1
        mask_f = cv2.GaussianBlur(mask_f, (ksize, ksize), blur_sigma)
        mask_f = np.clip(mask_f, 0.0, 1.0)

    return mask_f.astype(np.float32)


def create_distance_transform_mask(
    binary_mask: np.ndarray,
    feather_distance: float = 12.0,
) -> np.ndarray:
    """Create continuous artifact-free feathering using Euclidean Distance Transform.

    Args:
        binary_mask: 2D binary mask array (uint8 or bool) where face is 1/255.
        feather_distance: Distance in pixels across which transition from 0 to 1 occurs.

    Returns:
        np.ndarray of shape (H, W), float32 with smooth gradient [0.0, 1.0].
    """
    bin_u8 = (binary_mask > 0.5).astype(np.uint8) * 255
    if np.all(bin_u8 == 0) or np.all(bin_u8 == 255):
        return bin_u8.astype(np.float32) / 255.0

    # Distance to closest background pixel
    dist_inside = cv2.distanceTransform(bin_u8, cv2.DIST_L2, 5)

    if feather_distance <= 0:
        return (dist_inside > 0).astype(np.float32)

    feathered = np.clip(dist_inside / feather_distance, 0.0, 1.0)
    # Apply smoothstep polynomial: 3x^2 - 2x^3 for S-curve feathering
    smooth = feathered * feathered * (3.0 - 2.0 * feathered)
    return smooth.astype(np.float32)
