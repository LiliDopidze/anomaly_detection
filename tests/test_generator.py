"""Statistical properties, reproducibility and separation of physical truth."""

from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from optical_anomaly.generator import generate


def test_stationary_variance_and_correlation(stationary_sample):
    assert stationary_sample.var() == pytest.approx(0.04, rel=0.04)
    assert np.corrcoef(stationary_sample[:-1], stationary_sample[1:])[
        0, 1
    ] == pytest.approx(0.8, abs=0.01)


def test_reproducible_generation(generator_config, generated_data):
    native, truth = generated_data
    # This second generation is intentional: sharing the same object would not
    # test that independent executions reproduce the same fixture.
    again, labels = generate(generator_config)
    pd.testing.assert_frame_equal(native, again)
    pd.testing.assert_frame_equal(truth, labels)


def test_separate_truth(generated_data):
    native, truth = generated_data
    assert not set(truth.columns).intersection(native.columns)
    assert set(truth.fault_type) == {"random_walk", "exponential", "variance_shift"}
    assert (
        truth.impact_time.dropna() >= truth.loc[truth.impact_time.notna(), "onset_time"]
    ).all()


def test_collection_does_not_change_impact(generator_config, generated_data):
    _, truth = generated_data
    _, missing = generate(replace(generator_config, missing_probability=0.3))
    pd.testing.assert_series_equal(truth.impact_time, missing.impact_time)
