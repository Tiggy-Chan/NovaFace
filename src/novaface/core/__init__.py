"""NovaFace Core mathematical, temporal, masking, and spatial blending modules."""

from novaface.core.math_utils import (
    compute_similarity_transform,
    invert_affine_transform,
    transform_points,
    get_canonical_landmarks,
    CANONICAL_FACE_5PTS_112,
    CANONICAL_FACE_5PTS_128,
)
from novaface.core.smoother import OneEuroFilter, LandmarkSmoother, LowPassFilter
from novaface.core.mask_utils import (
    create_elliptical_mask,
    create_convex_hull_mask,
    create_distance_transform_mask,
)
from novaface.core.color_transfer import reinhard_color_transfer
from novaface.core.blender import FrameBlender

__all__ = [
    "compute_similarity_transform",
    "invert_affine_transform",
    "transform_points",
    "get_canonical_landmarks",
    "CANONICAL_FACE_5PTS_112",
    "CANONICAL_FACE_5PTS_128",
    "OneEuroFilter",
    "LandmarkSmoother",
    "LowPassFilter",
    "create_elliptical_mask",
    "create_convex_hull_mask",
    "create_distance_transform_mask",
    "reinhard_color_transfer",
    "FrameBlender",
]
