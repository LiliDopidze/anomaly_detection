"""Small mathematical primitives, with explicit degenerate-window behaviour."""

import numpy as np


def coefficient_of_variation(values: np.ndarray) -> float:
    """Population CoV; meaningful here on linear power, never on dBm."""
    mean = abs(float(np.mean(values)))
    return float(np.std(values) / mean) if mean > 1e-15 else np.nan


def lag_one(values: np.ndarray) -> float:
    centred = values - np.mean(values)
    denominator = float(centred @ centred)
    return (
        float(centred[:-1] @ centred[1:] / denominator) if denominator > 1e-15 else 0.0
    )


def entropy(values: np.ndarray, bins: np.ndarray) -> float:
    counts, _ = np.histogram(values, bins=bins)
    probabilities = counts[counts > 0] / len(values)
    return float(-np.sum(probabilities * np.log(probabilities)))


def negative_cusum(
    values: np.ndarray, means: np.ndarray, allowance: float
) -> np.ndarray:
    result = np.full(len(values), np.nan)
    state = 0.0
    for i, (value, mean) in enumerate(zip(values, means)):
        if not np.isfinite(value) or not np.isfinite(mean):
            state = 0.0
        else:
            state = max(0.0, state + mean - value - allowance)
            result[i] = state
    return result
