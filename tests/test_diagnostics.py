"""Diagnostics preserve gaps and keep final fault details out of reports."""

import numpy as np
import pandas as pd
from optical_anomaly.diagnostics import (
    missingness_report,
    healthy_statistics,
    development_faults,
)


def test_missing_run_lengths():
    native = pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=6, freq="5min"),
            "device": "A",
            "rx_dbm": [1, np.nan, np.nan, 2, np.nan, 3],
        }
    )
    row = missingness_report(native).iloc[0]
    assert row.observed == 3
    assert row.longest_missing_intervals == 2
    assert row.missing_fraction == 0.5


def test_healthy_statistics_and_final_exclusion(generated_data, generator_config):
    native, truth = generated_data
    start = native.time.min()
    healthy_end = start + pd.Timedelta(days=generator_config.days * 0.4)
    end = start + pd.Timedelta(days=generator_config.days * 0.75)
    stats = healthy_statistics(native.loc[native.time < healthy_end], generator_config)
    assert len(stats) == generator_config.entities
    expected = generator_config.noise_db**2 + generator_config.sensor_noise_db**2
    np.testing.assert_allclose(stats.expected_variance, expected)
    faults = development_faults(truth, native, end, generator_config.interval_minutes)
    assert (faults.onset_time < end).all()
    assert (faults.end_time <= end).all()
    assert len(faults) < len(truth)


def test_no_faults_are_a_valid_diagnostic_case(generated_data, generator_config):
    native, truth = generated_data
    faults = development_faults(
        truth.iloc[:0], native, native.time.max(), generator_config.interval_minutes
    )
    assert faults.empty
    assert "warning_opportunity_proxy" in faults
