"""Differential privacy noise mechanisms."""

import numpy as np


def laplace_mechanism(true_value: float, sensitivity: float, epsilon: float) -> float:
    """Add Laplace noise calibrated to sensitivity/epsilon.

    Parameters
    ----------
    true_value : float
        The true (non-private) query result.
    sensitivity : float
        Global sensitivity of the query.
    epsilon : float
        Privacy parameter for this query.

    Returns
    -------
    float
        Noisy result satisfying epsilon-differential privacy.
    """
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive")
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0.0, scale=scale)
    return true_value + noise


def laplace_mechanism_array(
    true_values: np.ndarray, sensitivity: float, epsilon: float
) -> np.ndarray:
    """Vectorized Laplace mechanism for GROUP BY results."""
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive")
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0.0, scale=scale, size=true_values.shape)
    return true_values + noise
