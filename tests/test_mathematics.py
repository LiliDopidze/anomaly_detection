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
