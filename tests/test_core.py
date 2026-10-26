"""Unit tests for masking, color transfer, and spatial blending."""

import cv2
import numpy as np

from novaface.core.blender import FrameBlender
from novaface.core.color_transfer import reinhard_color_transfer
from novaface.core.mask_utils import (
    create_distance_transform_mask,
    create_elliptical_mask,
)
from novaface.core.math_utils import compute_similarity_transform, get_canonical_landmarks


def test_elliptical_mask():
    mask = create_elliptical_mask(size=(128, 128), blur_sigma=4.0)
    assert mask.shape == (128, 128)
    assert mask.dtype == np.float32
    assert 0.0 <= mask.min() <= mask.max() <= 1.0
    # Center should be 1.0
    assert mask[64, 64] > 0.95
    # Corner should be 0.0
    assert mask[0, 0] < 0.05


def test_distance_transform_mask():
    binary = np.zeros((100, 100), dtype=np.uint8)
    binary[25:75, 25:75] = 255
    feathered = create_distance_transform_mask(binary, feather_distance=10.0)

    assert feathered.shape == (100, 100)
    assert feathered[0, 0] == 0.0
    assert feathered[50, 50] == 1.0
    # Gradient exists along boundary
    assert 0.0 < feathered[28, 50] < 1.0


def test_reinhard_color_transfer():
    # Warm yellow source patch
    src = np.full((64, 64, 3), (50, 200, 240), dtype=np.uint8)
    # Cold blue target patch
    tgt = np.full((64, 64, 3), (240, 100, 30), dtype=np.uint8)

    transferred = reinhard_color_transfer(src, tgt, blend_ratio=1.0)
    assert transferred.shape == (64, 64, 3)
    # Mean of transferred should shift significantly towards target
    assert transferred[:, :, 0].mean() > 180  # Blue channel increased
    assert transferred[:, :, 2].mean() < 100  # Red channel decreased


def test_frame_blender():
    target_frame = np.full((200, 200, 3), 50, dtype=np.uint8)
    target_patch = np.full((128, 128, 3), 50, dtype=np.uint8)
    swapped_patch = np.full((128, 128, 3), 200, dtype=np.uint8)

    # 5 points centered in target frame
    src_pts = np.array(
        [[70, 70], [130, 70], [100, 100], [80, 130], [120, 130]],
        dtype=np.float32,
    )
    affine_m = compute_similarity_transform(src_pts, get_canonical_landmarks(128))

    blender = FrameBlender(enable_color_transfer=False)
    composite = blender.composite_face(
        target_frame=target_frame,
        cropped_target_patch=target_patch,
        swapped_face_patch=swapped_patch,
        affine_matrix=affine_m,
    )

    assert composite.shape == (200, 200, 3)
    # Center should be close to swapped patch (high values)
    assert composite[100, 100, 0] > 150
    # Outer corners should be intact target values (50)
    assert composite[10, 10, 0] == 50
