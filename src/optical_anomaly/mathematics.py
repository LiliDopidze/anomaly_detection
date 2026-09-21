"""Mathematical primitives; source IDs resolve in METHOD.md, Sources."""

import numpy as np


def coefficient_of_variation(values: np.ndarray) -> float:
    """Population CoV; meaningful here on linear power, never on dBm."""
    # Source: [cov]. Population SD is our window-summary convention.
    if len(values) == 0 or not np.isfinite(values).all():
        return np.nan
    mean = abs(float(np.mean(values)))
    # Assumption: numerical denominator guard, not an optical limit.
    return float(np.std(values) / mean) if mean > 1e-15 else np.nan


def lag_one(values: np.ndarray) -> float:
    """Source: [acf]; constant windows use zero as an explicit convention."""
    if len(values) < 2 or not np.isfinite(values).all():
        return np.nan
    centred = values - np.mean(values)
    denominator = float(centred @ centred)
    # Assumption: near-zero variance uses the documented constant-window convention.
    return (
        float(centred[:-1] @ centred[1:] / denominator) if denominator > 1e-15 else 0.0
    )


def entropy(values: np.ndarray, bins: np.ndarray) -> float:
    """Source: [shannon]; histogram plug-in estimate in natural-log units."""
    if len(values) == 0 or not np.isfinite(values).all():
        return np.nan
    counts, _ = np.histogram(values, bins=bins)
    probabilities = counts[counts > 0] / len(values)
    return float(-np.sum(probabilities * np.log(probabilities)))


def negative_cusum(
    values: np.ndarray, means: np.ndarray, allowance: float
) -> np.ndarray:
    """Sources: [cusum], [page]; missing observations reset accumulated state."""
    if values.ndim != 1 or means.shape != values.shape:
        raise ValueError("CUSUM values and targets must be equal-length vectors")
    if not np.isfinite(allowance) or allowance < 0:
        raise ValueError("CUSUM allowance must be finite and nonnegative")
    result = np.full(len(values), np.nan)
    state = 0.0
    for i, (value, mean) in enumerate(zip(values, means)):
        if not np.isfinite(value) or not np.isfinite(mean):
            state = 0.0
        else:
            state = max(0.0, state + mean - value - allowance)
            result[i] = state
    return result
