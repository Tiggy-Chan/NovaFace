"""Unit tests for mathematical transformations and geometrical alignment."""

import numpy as np
import pytest

from novaface.core.math_utils import (
    compute_similarity_transform,
    get_canonical_landmarks,
    invert_affine_transform,
    transform_points,
)


def test_canonical_landmarks():
    lms112 = get_canonical_landmarks(112)
    assert lms112.shape == (5, 2)
    assert np.all(lms112 > 0)
    assert np.all(lms112 < 112)

    lms128 = get_canonical_landmarks(128)
    assert lms128.shape == (5, 2)
    assert np.all(lms128 > 0)
    assert np.all(lms128 < 128)


def test_similarity_transform_and_inverse():
    # 5 standard points
    src = np.array(
        [[30.0, 40.0], [80.0, 45.0], [55.0, 70.0], [35.0, 95.0], [75.0, 92.0]],
        dtype=np.float32,
    )
    dst = get_canonical_landmarks(128)

    # Compute M
    m = compute_similarity_transform(src, dst)
    assert m.shape == (2, 3)

    # Compute M_inv
    m_inv = invert_affine_transform(m)
    assert m_inv.shape == (2, 3)

    # Roundtrip mapping
    transformed = transform_points(src, m)
    reconstructed = transform_points(transformed, m_inv)

    max_err = np.max(np.abs(src - reconstructed))
    assert max_err < 1e-4


def test_degenerate_points():
    # Less than 2 points
    with pytest.raises(ValueError):
        compute_similarity_transform(np.zeros((1, 2)), np.zeros((1, 2)))

    # Zero variance points
    with pytest.raises(ValueError):
        compute_similarity_transform(np.full((5, 2), 10.0), get_canonical_landmarks(128))
