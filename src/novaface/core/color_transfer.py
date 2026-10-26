"""Color grading and illumination matching algorithms for photorealistic face integration.

Implements Reinhard color transfer in perceptual CIE-Lab color space to harmonize
skin tone, lighting temperature, and shadows between target frame and generated face.
"""

from typing import Optional
import cv2
import numpy as np


def reinhard_color_transfer(
    source_bgr: np.ndarray,
    target_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    blend_ratio: float = 0.85,
) -> np.ndarray:
    """Transfer color characteristics of target frame onto the generated source face.

    Transforms both images to CIE-Lab color space, aligns statistical distribution
    (mean and standard deviation) per channel, and converts back to BGR.

    Args:
        source_bgr: Swapped face patch (H, W, 3) in uint8 BGR format.
        target_bgr: Original cropped target face patch (H, W, 3) in uint8 BGR format.
        mask: Optional mask (H, W) in float32 or uint8 restricting color statistics.
        blend_ratio: Interpolation factor between original source and color-matched source (0.0 to 1.0).

    Returns:
        Color-matched image (H, W, 3) in uint8 BGR.
    """
    if source_bgr.shape != target_bgr.shape:
        raise ValueError("Source and target images must share identical spatial dimensions.")

    src_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Calculate statistics optionally masked
    if mask is not None:
        mask_bool = (mask > 0.2)
        if np.any(mask_bool):
            src_pixels = src_lab[mask_bool]
            tgt_pixels = tgt_lab[mask_bool]
        else:
            src_pixels = src_lab.reshape((-1, 3))
            tgt_pixels = tgt_lab.reshape((-1, 3))
    else:
        src_pixels = src_lab.reshape((-1, 3))
        tgt_pixels = tgt_lab.reshape((-1, 3))

    src_mean = np.mean(src_pixels, axis=0)
    src_std = np.std(src_pixels, axis=0) + 1e-6

    tgt_mean = np.mean(tgt_pixels, axis=0)
    tgt_std = np.std(tgt_pixels, axis=0) + 1e-6

    # Normalize source to zero-mean, unit-variance, scale by target std, add target mean
    matched_lab = (src_lab - src_mean) * (tgt_std / src_std) + tgt_mean
    matched_lab = np.clip(matched_lab, 0.0, 255.0).astype(np.uint8)

    matched_bgr = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)

    if blend_ratio >= 1.0:
        return matched_bgr
    if blend_ratio <= 0.0:
        return source_bgr

    # Linear interpolation between raw source and color-adjusted output
    output = cv2.addWeighted(matched_bgr, blend_ratio, source_bgr, 1.0 - blend_ratio, 0.0)
    return output
