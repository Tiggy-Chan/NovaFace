"""High-performance frame blender and spatial compositor.

Warps normalized swapped face patches back to the original full-frame coordinate space
and seamlessly composites them using distance-smoothed alpha blending or Poisson cloning.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from novaface.core.color_transfer import reinhard_color_transfer
from novaface.core.mask_utils import create_elliptical_mask, create_distance_transform_mask


class FrameBlender:
    """Seamless face compositor with color matching and inverse affine projection."""

    def __init__(
        self,
        enable_color_transfer: bool = True,
        color_blend_ratio: float = 0.85,
        feather_distance: float = 12.0,
        use_poisson_fallback: bool = False,
    ):
        """Initialize FrameBlender.

        Args:
            enable_color_transfer: Whether to perform Reinhard Lab color calibration.
            color_blend_ratio: Strength of color matching (0.0 to 1.0).
            feather_distance: Width of edge feathering in pixels.
            use_poisson_fallback: Whether to use cv2.seamlessClone instead of alpha blend.
        """
        self.enable_color_transfer = enable_color_transfer
        self.color_blend_ratio = color_blend_ratio
        self.feather_distance = feather_distance
        self.use_poisson_fallback = use_poisson_fallback

        # Cache standard canonical mask for 128x128
        self._cached_mask_128 = create_elliptical_mask(size=(128, 128), blur_sigma=8.0)

    def get_default_mask(self, size: Tuple[int, int]) -> np.ndarray:
        """Retrieve or construct default soft feathering mask for given patch size."""
        if size == (128, 128):
            return self._cached_mask_128
        return create_elliptical_mask(size=size, blur_sigma=size[0] * (8.0 / 128.0))

    def composite_face(
        self,
        target_frame: np.ndarray,
        cropped_target_patch: np.ndarray,
        swapped_face_patch: np.ndarray,
        affine_matrix: np.ndarray,
        custom_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Composite swapped face patch back onto the full target frame.

        Args:
            target_frame: Original full frame image (H, W, 3) in uint8 BGR.
            cropped_target_patch: Cropped original face before swap (patch_h, patch_w, 3).
            swapped_face_patch: Output of face swapper model (patch_h, patch_w, 3).
            affine_matrix: 2x3 affine matrix used to map full frame -> cropped patch.
            custom_mask: Optional custom float32 mask in [0, 1] for the patch.

        Returns:
            Composite image (H, W, 3) in uint8 BGR.
        """
        patch_h, patch_w = swapped_face_patch.shape[:2]
        frame_h, frame_w = target_frame.shape[:2]

        # 1. Color Transfer
        patch_to_blend = swapped_face_patch
        if self.enable_color_transfer:
            patch_to_blend = reinhard_color_transfer(
                source_bgr=swapped_face_patch,
                target_bgr=cropped_target_patch,
                blend_ratio=self.color_blend_ratio,
            )

        # 2. Prepare Mask
        if custom_mask is not None:
            mask = custom_mask
        else:
            mask = self.get_default_mask((patch_w, patch_h))

        # 3. Warp swapped patch and mask back to target full frame space
        # affine_matrix maps full_frame -> patch.
        # cv2.warpAffine with WARP_INVERSE_MAP maps patch -> full_frame.
        warped_patch = cv2.warpAffine(
            patch_to_blend,
            affine_matrix,
            (frame_w, frame_h),
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        warped_mask = cv2.warpAffine(
            mask,
            affine_matrix,
            (frame_w, frame_h),
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

        # 4. Optional Poisson Seamless Clone
        if self.use_poisson_fallback:
            mask_u8 = (warped_mask > 0.3).astype(np.uint8) * 255
            # Find bounding rect of the mask
            x, y, bw, bh = cv2.boundingRect(mask_u8)
            if bw > 10 and bh > 10:
                center = (x + bw // 2, y + bh // 2)
                try:
                    return cv2.seamlessClone(
                        warped_patch, target_frame, mask_u8, center, cv2.NORMAL_CLONE
                    )
                except cv2.error:
                    pass  # Fallback to direct alpha blend

        # 5. Fast Alpha Blending via broadcasting
        # Expanding mask to 3 channels (H, W, 1)
        alpha = warped_mask[:, :, np.newaxis]
        target_f = target_frame.astype(np.float32)
        warped_f = warped_patch.astype(np.float32)

        composite_f = target_f * (1.0 - alpha) + warped_f * alpha
        return np.clip(composite_f, 0.0, 255.0).astype(np.uint8)
