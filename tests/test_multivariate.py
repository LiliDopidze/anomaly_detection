"""Causal nested telemetry features and the data/feature boundary."""

import numpy as np
import pandas as pd
import pytest
from optical_anomaly.features import FeatureEngineer, FEATURES
from optical_anomaly.multivariate import (
    MultivariateFeatures,
    FEATURE_SETS,
    columns_for,
    measurement_channels,
)
from optical_anomaly.workflow import adapt_entity
from optical_anomaly.diagnostics import canonical_statistics


@pytest.fixture(scope="module")
def canonical(generated_data, generator_config):
    native, _ = generated_data
    entity = native.device.iloc[0]
    return adapt_entity(
        native.loc[native.device.eq(entity)], generator_config.interval_minutes
    )


def baseline_for(canonical):
    step = canonical.timestamp.drop_duplicates().sort_values().diff().min()
    return FeatureEngineer(interval_minutes=int(step.total_seconds() / 60))


def test_nested_feature_sets_keep_original_baseline(canonical):
    end = canonical.timestamp.min() + pd.Timedelta(days=2)
    training = canonical.loc[canonical.timestamp < end]
    baseline = baseline_for(canonical).fit(training).transform(canonical)
    model = MultivariateFeatures(baseline_for(canonical)).fit(training)
    result = model.transform(canonical)
    pd.testing.assert_frame_equal(result[FEATURES], baseline[FEATURES])
    assert result[columns_for("temperature")].notna().all(axis=1).any()
    previous = set()
    for stage, count in zip(FEATURE_SETS, [12, 18, 30, 46, 52]):
        columns = columns_for(stage)
        assert len(columns) == count
        assert previous.issubset(columns)
        assert set(columns).issubset(result)
        previous = set(columns)


def test_future_values_cannot_change_earlier_features(canonical):
    start = canonical.timestamp.min()
    training = canonical.loc[canonical.timestamp < start + pd.Timedelta(days=2)]
    engineer = MultivariateFeatures(baseline_for(canonical)).fit(training)
    original = engineer.transform(canonical)
    cut = start + pd.Timedelta(days=3)
    modified = canonical.copy()
    modified.loc[modified.timestamp >= cut, "value"] *= 1.5
    changed = engineer.transform(modified)
    pd.testing.assert_frame_equal(
        original.loc[original.timestamp < cut], changed.loc[changed.timestamp < cut]
    )


def test_unknown_extra_measurement_does_not_invent_features(canonical):
    start = canonical.timestamp.min()
    model = MultivariateFeatures(baseline_for(canonical)).fit(
        canonical.loc[canonical.timestamp < start + pd.Timedelta(days=2)]
    )
    missing = canonical.loc[canonical.metric_name.ne("ont_temperature_c")]
    features = model.transform(missing)
    assert features.ont_temperature_level.isna().all()
    assert features[columns_for("fec")].notna().all(axis=1).any()
    with pytest.raises(ValueError, match="Missing channels"):
        MultivariateFeatures(baseline_for(canonical)).fit(missing)


def test_relationships_and_zero_denominator(canonical):
    channels = measurement_channels(canonical)
    wide = canonical.pivot(
        index=["entity_id", "timestamp"], columns="metric_name", values="value"
    ).sort_index()
    np.testing.assert_allclose(
        channels.downstream_loss,
        wide.olt_tx_power_dbm - wide.rx_power_dbm,
        equal_nan=True,
    )
    zero = canonical.copy()
    zero.loc[zero.metric_name.eq("downstream_fec_total_codewords"), "value"] = 0
    assert measurement_channels(zero).downstream_fec_corrected.isna().all()


def test_daily_eda_detects_known_cycle_without_imputation():
    times = pd.date_range("2025-01-01", periods=40 * 24, freq="h", tz="UTC")
    values = 5 + 2 * np.sin(2 * np.pi * np.arange(len(times)) / 24)
    values[::17] = np.nan
    data = pd.DataFrame(
        {
            "timestamp": times,
            "entity_id": "A",
            "metric_name": "temperature",
            "value": values,
        }
    )
    result = canonical_statistics(data, 60).iloc[0]
    assert result.daily_amplitude == pytest.approx(2)
    assert result.daily_holdout_r2 == pytest.approx(1)
    assert result.missing_fraction > 0


@pytest.mark.parametrize("break_kind", ["missing", "timestamp_gap"])
def test_added_summaries_reset_and_rewarm(break_kind):
    values = pd.Series(np.arange(70, dtype=float))
    times = pd.Series(pd.date_range("2025-01-01", periods=70, freq="5min"))
    first = 30
    if break_kind == "missing":
        values.iloc[30] = np.nan
        first = 31
    else:
        times.iloc[30:] += pd.Timedelta(hours=1)
    model = MultivariateFeatures(FeatureEngineer())
    result = model._summaries(values, times, "signal")
    assert result.signal_level.iloc[first : first + 11].isna().all()
    assert result.signal_slope.iloc[first : first + 12].isna().all()
    assert result.signal_level.iloc[first + 11] == pytest.approx(first + 5.5)
    assert result.signal_variability.iloc[first + 11] == pytest.approx(
        np.sqrt((12**2 - 1) / 12)
    )
    assert result.signal_slope.iloc[first + 12] == pytest.approx(12.0)


def test_long_optical_windows_and_regression_have_physical_time_units():
    model = MultivariateFeatures(FeatureEngineer())
    times = pd.Series(pd.date_range("2025-01-01", periods=100, freq="5min"))
    values = pd.Series(2 + 3 * np.arange(100) / 12)
    result = model._summaries(values, times, "upstream_rx")
    assert result.upstream_rx_long_level.iloc[:71].isna().all()
    assert result.upstream_rx_long_level.iloc[71] == pytest.approx(
        values.iloc[:72].mean()
    )
    assert result.upstream_rx_regression_slope.iloc[11] == pytest.approx(3)
    expected = values.iloc[60:72].mean() - values.iloc[:72].mean()
    assert result.upstream_rx_short_minus_long.iloc[71] == pytest.approx(expected)
    values.iloc[80] = np.nan
    broken = model._summaries(values, times, "upstream_rx")
    assert broken.upstream_rx_long_level.iloc[80:].isna().all()


def test_fec_error_frequency_uses_uncentred_counts_and_preserves_unknowns():
    times = pd.Series(pd.date_range("2025-01-01", periods=30, freq="5min"))
    measured = pd.Series(np.tile([0.0, 0.0, 1.0], 10))
    model = MultivariateFeatures(FeatureEngineer())
    result = model._summaries(measured - 5, times, "downstream_fec_corrected", measured)
    assert result.downstream_fec_corrected_error_interval_fraction.iloc[
        11
    ] == pytest.approx(1 / 3)
    measured.iloc[15] = np.nan
    result = model._summaries(measured - 5, times, "downstream_fec_corrected", measured)
    assert (
        result.downstream_fec_corrected_error_interval_fraction.iloc[15:27].isna().all()
    )
