"""Diagnostics preserve gaps and keep final fault details out of reports."""

import numpy as np
import pandas as pd
import pytest
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


def test_dependence_reports_preserve_time_gaps_and_constant_channels():
    from optical_anomaly.diagnostics import dependence_reports

    times = pd.date_range("2025-01-01", periods=120, freq="h", tz="UTC")
    rng = np.random.default_rng(7)
    x = rng.normal(size=len(times))
    x[::7] = np.nan
    frame = pd.DataFrame({"timestamp": times, "a": x, "b": 2 * x, "constant": 3.0})
    data = frame.melt("timestamp", var_name="metric_name", value_name="value")
    data["entity_id"] = "A"
    correlation, acf = dependence_reports(data, 60)
    ab = correlation.loc[correlation.left.eq("a") & correlation.right.eq("b")]
    np.testing.assert_allclose(ab.pearson, 1)
    assert ab.paired_observations.eq(np.isfinite(x).sum()).all()
    constant = correlation.loc[correlation.right.eq("constant")]
    assert constant.pearson.isna().all()
    lag1 = acf.loc[
        acf.view.eq("raw") & acf.metric_name.eq("a") & acf.lag_intervals.eq(1)
    ].iloc[0]
    expected = pd.Series(x)
    assert lag1.autocorrelation == pytest.approx(expected.autocorr(1))
    assert (
        lag1.paired_observations == (expected.notna() & expected.shift().notna()).sum()
    )
