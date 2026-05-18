"""Tests for DP mechanisms."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpdb.mechanisms import laplace_mechanism, laplace_mechanism_array


class TestLaplaceMechanism:
    def test_mean_centered(self):
        """Noisy results should be centered around the true value."""
        np.random.seed(42)
        true_val = 1000.0
        results = [laplace_mechanism(true_val, 1.0, 1.0) for _ in range(10000)]
        assert abs(np.mean(results) - true_val) < 5.0

    def test_scale_increases_with_sensitivity(self):
        """Higher sensitivity -> more noise (larger variance)."""
        np.random.seed(42)
        low_sens = [laplace_mechanism(100.0, 1.0, 1.0) for _ in range(5000)]
        high_sens = [laplace_mechanism(100.0, 100.0, 1.0) for _ in range(5000)]
        assert np.std(high_sens) > np.std(low_sens) * 10

    def test_scale_decreases_with_epsilon(self):
        """Higher epsilon -> less noise."""
        np.random.seed(42)
        low_eps = [laplace_mechanism(100.0, 1.0, 0.1) for _ in range(5000)]
        high_eps = [laplace_mechanism(100.0, 1.0, 10.0) for _ in range(5000)]
        assert np.std(low_eps) > np.std(high_eps) * 5

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError):
            laplace_mechanism(100.0, 1.0, 0.0)

    def test_array_mechanism(self):
        np.random.seed(42)
        true_vals = np.array([100.0, 200.0, 300.0])
        noisy = laplace_mechanism_array(true_vals, 1.0, 1.0)
        assert noisy.shape == true_vals.shape
        assert not np.array_equal(noisy, true_vals)
