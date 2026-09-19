"""Checks for causality."""

import numpy as np
import pandas as pd
import pytest
from optical_anomaly.generator import GeneratorConfig, generate
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
        metrics={"rx": "rx_power_dbm"}, units={"rx_power_dbm": "mW"}
    )
    mapped = adapter.transform(native)
    np.testing.assert_allclose(mapped.value, [-20, -30, -10])
    result = DataValidator().transform(mapped)
    assert result.value.iloc[0] == -20  # The 00:02 observation is unavailable at 00:00.
    assert result.value.iloc[1] == -30
    assert pd.isna(result.value.iloc[2])
    with pytest.raises(ValueError, match="Duplicate"):
        adapter.transform(pd.concat([native, native.iloc[:1]]))


def test_feature_causality_offset_invariance_and_missing_warmup():
    native, _ = generate(GeneratorConfig(entities=1, days=8, missing_probability=0))
    data = DataValidator().transform(TelemetryAdapter().transform(native))
    end = native.time.iloc[1000]
    engineer = FeatureEngineer().fit(data.loc[data.timestamp < end])
    original = engineer.transform(data)
    modified = data.copy()
    cut = native.time.iloc[1200]
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
    rx_index = data.index[data.metric_name.eq("rx_power_dbm")][1100]
    missing = data.copy()
    missing.loc[rx_index, "value"] = np.nan
    result = engineer.transform(missing)
    assert result.cusum.iloc[1100:1113].isna().all()


def test_unknown_entities_abstain():
    native, _ = generate(GeneratorConfig(entities=1, days=8))
    data = DataValidator().transform(TelemetryAdapter().transform(native))
    engineer = FeatureEngineer().fit(data.loc[data.timestamp < native.time.iloc[1000]])
    unknown = data.assign(entity_id="new")
    assert engineer.transform(unknown)[FEATURES].isna().all().all()
