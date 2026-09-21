"""Checks for causality."""

import numpy as np
import pandas as pd
import pytest
from optical_anomaly.adapter import TelemetryAdapter
from optical_anomaly.validation import DataValidator
from optical_anomaly.features import FeatureEngineer, FEATURES


def test_adapter_units_and_past_only_resampling():
    native = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2025-01-01T00:00Z", "2025-01-01T00:02Z", "2025-01-01T00:11Z"]
            ),
            "device": "A",
            "rx": [0.01, 0.001, 0.1],
        }
    )
    adapter = TelemetryAdapter(
        timestamp_column="time",
        entity_column="device",
        metrics={"rx": "rx_power_dbm"},
        units={"rx_power_dbm": "mW"},
        kinds={"rx_power_dbm": "gauge"},
    )
    mapped = adapter.transform(native)
    np.testing.assert_allclose(mapped.value, [-20, -30, -10])
    result = DataValidator().transform(mapped)
    assert result.value.iloc[0] == -20  # The 00:02 observation is unavailable at 00:00.
    assert result.value.iloc[1] == -30
    assert pd.isna(result.value.iloc[2])
    with pytest.raises(ValueError, match="Duplicate"):
        adapter.transform(pd.concat([native, native.iloc[:1]]))


def test_feature_causality_offset_invariance_and_missing_warmup(clean_telemetry):
    data = clean_telemetry
    start = data.timestamp.min()
    end = start + pd.Timedelta(days=2)
    engineer = FeatureEngineer().fit(data.loc[data.timestamp < end])
    original = engineer.transform(data)
    modified = data.copy()
    cut = start + pd.Timedelta(hours=60)
    modified.loc[modified.timestamp >= cut, "value"] -= 5
    changed = engineer.transform(modified)
    pd.testing.assert_frame_equal(
        original.loc[original.timestamp < cut], changed.loc[changed.timestamp < cut]
    )
    offset = data.copy()
    offset.loc[offset.metric_name.eq("rx_power_dbm"), "value"] += 8
    shifted = (
        FeatureEngineer().fit(offset.loc[offset.timestamp < end]).transform(offset)
    )
    np.testing.assert_allclose(
        original[FEATURES], shifted[FEATURES], atol=1e-9, equal_nan=True
    )
    missing_time = start + pd.Timedelta(hours=54)
    rx_index = data.index[data.timestamp.eq(missing_time)]
    missing = data.copy()
    missing.loc[rx_index, "value"] = np.nan
    result = engineer.transform(missing)
    recovery_time = missing_time + pd.Timedelta(
        minutes=(engineer.window + 1) * engineer.interval_minutes
    )
    warming_up = result.timestamp.between(missing_time, recovery_time, inclusive="left")
    assert result.loc[warming_up, "cusum"].isna().all()
    assert result.loc[result.timestamp.eq(recovery_time), "cusum"].notna().all()
    assert result.loc[result.timestamp.eq(recovery_time), FEATURES].notna().all().all()
    first_window_end = missing_time + pd.Timedelta(
        minutes=engineer.window * engineer.interval_minutes
    )
    actual = result.loc[result.timestamp.eq(first_window_end)].iloc[0]
    assert actual.long_level == pytest.approx(actual.level)


def test_unknown_entities_abstain(clean_telemetry):
    data = clean_telemetry
    fit_end = data.timestamp.min() + pd.Timedelta(days=2)
    engineer = FeatureEngineer().fit(data.loc[data.timestamp < fit_end])
    unknown = data.assign(entity_id="new")
    assert engineer.transform(unknown)[FEATURES].isna().all().all()


def test_daily_reference_needs_multiple_cycles(clean_telemetry):
    end = clean_telemetry.timestamp.min() + pd.Timedelta(hours=12)
    short = clean_telemetry.loc[clean_telemetry.timestamp < end]
    with pytest.raises(ValueError, match="at least two days"):
        FeatureEngineer().fit(short)
    # A nonseasonal baseline has no daily-cycle identification requirement.
    assert FeatureEngineer(seasonal=False).fit(short).references


@pytest.mark.parametrize("scenario", ["step", "ramp"])
def test_cusum_keeps_training_target_during_sustained_degradation(scenario):
    times = pd.date_range("2025-01-01", periods=348, freq="5min", tz="UTC")
    onset = times[0] + pd.Timedelta(days=1)
    data = pd.DataFrame(
        {
            "timestamp": times,
            "entity_id": "A",
            "metric_name": "rx_power_dbm",
            "value": -20.0,
        }
    )
    faulty = data.timestamp.ge(onset)
    if scenario == "step":
        data.loc[faulty, "value"] -= 0.05  # One frozen reference-scale unit.
        expected = 60 * 0.75
    else:
        data.loc[faulty, "value"] -= 0.01 * np.arange(1, 61)
        expected = 351.05  # Sum of 0.2*j - 0.25 for j=2,...,60.
    engineer = FeatureEngineer(seasonal=False).fit(data.loc[~faulty])
    result = engineer.transform(data)
    assert result.cusum.iloc[-1] == pytest.approx(expected)

    gap = onset + pd.Timedelta(hours=1)
    interrupted = data.copy()
    interrupted.loc[data.timestamp.eq(gap), "value"] = np.nan
    after_gap = engineer.transform(interrupted)
    recovery = gap + pd.Timedelta(
        minutes=(engineer.window + 1) * engineer.interval_minutes
    )
    warmup = after_gap.timestamp.between(gap, recovery, inclusive="left")
    assert after_gap.loc[warmup, "cusum"].isna().all()
    actual = after_gap.loc[after_gap.timestamp.eq(recovery), "cusum"].iloc[0]
    current = data.loc[data.timestamp.eq(recovery), "value"].iloc[0]
    assert actual == pytest.approx((-20 - current) / 0.05 - engineer.allowance)


def test_features_reset_on_nonstandard_spacing(clean_telemetry):
    training_end = clean_telemetry.timestamp.min() + pd.Timedelta(days=2)
    engineer = FeatureEngineer().fit(
        clean_telemetry.loc[clean_telemetry.timestamp < training_end]
    )
    data = clean_telemetry.copy()
    cut = training_end + pd.Timedelta(hours=6)
    data.loc[data.timestamp >= cut, "timestamp"] += pd.Timedelta(minutes=2)
    result = engineer.transform(data)
    first = cut + pd.Timedelta(minutes=2)
    last_warmup = first + pd.Timedelta(
        minutes=(engineer.window - 1) * engineer.interval_minutes
    )
    warming = result.timestamp.between(first, last_warmup, inclusive="left")
    assert result.loc[warming, "level"].isna().all()
    assert result.loc[result.timestamp.eq(last_warmup), "level"].notna().all()
