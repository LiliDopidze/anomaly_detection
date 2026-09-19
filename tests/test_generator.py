"""Checks for generator."""

from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from optical_anomaly.generator import GeneratorConfig, generate, stationary_noise


def test_stationarity_and_separate_truth():
    noise = stationary_noise(100000, 0.2, 0.8, np.random.default_rng(42))
    assert noise.var() == pytest.approx(0.04, rel=0.04)
    assert np.corrcoef(noise[:-1], noise[1:])[0, 1] == pytest.approx(0.8, abs=0.01)
    config = GeneratorConfig(entities=3, days=8)
    native, truth = generate(config)
    again, labels = generate(config)
    pd.testing.assert_frame_equal(native, again)
    pd.testing.assert_frame_equal(truth, labels)
    assert not set(truth.columns).intersection(native.columns)
    assert set(truth.fault_type) == {"random_walk", "exponential", "variance_shift"}
    assert (
        truth.impact_time.dropna() >= truth.loc[truth.impact_time.notna(), "onset_time"]
    ).all()
    _, missing = generate(replace(config, missing_probability=0.3))
    pd.testing.assert_series_equal(truth.impact_time, missing.impact_time)
