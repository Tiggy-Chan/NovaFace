"""Unit tests for temporal smoothing and jitter elimination."""

import numpy as np

from novaface.core.smoother import LandmarkSmoother, LowPassFilter, OneEuroFilter


def test_low_pass_filter():
    lpf = LowPassFilter(alpha=0.5)
    v1 = np.array([10.0, 10.0], dtype=np.float32)
    v2 = np.array([20.0, 20.0], dtype=np.float32)

    res1 = lpf.filter(v1)
    np.testing.assert_allclose(res1, v1)

    res2 = lpf.filter(v2)
    # 0.5 * 20 + 0.5 * 10 = 15
    np.testing.assert_allclose(res2, np.array([15.0, 15.0], dtype=np.float32))


def test_one_euro_filter_jitter_reduction():
    filter_ = OneEuroFilter(min_cutoff=1.0, beta=0.01)
    true_signal = np.array([50.0, 50.0], dtype=np.float32)

    noisy_series = []
    filtered_series = []

    for t in range(50):
        noise = np.random.normal(0, 2.0, (2,)).astype(np.float32)
        measurement = true_signal + noise
        out = filter_.filter(measurement, timestamp=t * 0.033)

        noisy_series.append(measurement)
        filtered_series.append(out)

    # Variance of filtered stationary signal should be substantially lower than noise
    std_noisy = np.std(np.array(noisy_series)[10:], axis=0).mean()
    std_filtered = np.std(np.array(filtered_series)[10:], axis=0).mean()

    assert std_filtered < std_noisy * 0.6


def test_landmark_smoother_discontinuity_reset():
    smoother = LandmarkSmoother(jump_threshold=50.0)
    pts1 = np.full((5, 2), 100.0, dtype=np.float32)
    # Huge jump to simulate different face
    pts2 = np.full((5, 2), 500.0, dtype=np.float32)

    smoother.update(pts1, timestamp=0.0)
    # Sudden jump beyond threshold resets filter to avoid ghosting
    res2 = smoother.update(pts2, timestamp=0.033)
    # Should instantly latch onto pts2
    diff = np.mean(np.abs(res2 - pts2))
    assert diff < 1.0
