"""Checks for mathematics."""

import numpy as np
import pytest
from optical_anomaly.mathematics import (
    coefficient_of_variation,
    lag_one,
    entropy,
    negative_cusum,
)


def test_mathematical_definitions():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert coefficient_of_variation(x) == pytest.approx(np.sqrt(1.25) / 2.5)
    assert coefficient_of_variation(100 * x) == pytest.approx(
        coefficient_of_variation(x)
    )
    assert lag_one(x) == pytest.approx(0.25)
    assert lag_one(np.ones(4)) == 0
    assert entropy(np.array([0, 0, 1, 1]), np.array([-0.5, 0.5, 1.5])) == pytest.approx(
        np.log(2)
    )
    np.testing.assert_allclose(
        negative_cusum(np.array([0, -1, -2]), np.zeros(3), 0.25), [0, 0.75, 2.5]
    )


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "values, expected",
    [
        (np.array([]), (np.nan, np.nan, np.nan)),
        (np.full(4, np.nan), (np.nan, np.nan, np.nan)),
        (np.array([1.0, np.nan]), (np.nan, np.nan, np.nan)),
        (np.array([1.0, np.inf]), (np.nan, np.nan, np.nan)),
        (np.zeros(4), (np.nan, 0.0, 0.0)),
        (np.ones(4), (0.0, 0.0, 0.0)),
    ],
    ids=["empty", "all_missing", "partly_missing", "infinite", "zero", "constant"],
)
def test_degenerate_windows(values, expected):
    actual = (
        coefficient_of_variation(values),
        lag_one(values),
        entropy(values, np.array([-np.inf, 0.5, np.inf])),
    )
    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("allowance", [-0.1, np.nan, np.inf])
def test_cusum_rejects_invalid_allowance(allowance):
    with pytest.raises(ValueError, match="allowance"):
        negative_cusum(np.zeros(3), np.zeros(3), allowance)


def test_cusum_rejects_mismatched_targets_and_resets_at_missing():
    with pytest.raises(ValueError, match="equal-length"):
        negative_cusum(np.zeros(3), np.zeros(2), 0.25)
    values = np.array([-1.0, -1.0, np.nan, -1.0])
    np.testing.assert_allclose(
        negative_cusum(values, np.zeros(4), 0.25),
        [0.75, 1.5, np.nan, 0.75],
        equal_nan=True,
    )
