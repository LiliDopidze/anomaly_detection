"""Source contracts are independent of synthetic names and detector needs."""

import numpy as np
import pandas as pd
import pytest
from optical_anomaly.adapter import TelemetryAdapter
from optical_anomaly.sources import synthetic_adapter
from optical_anomaly.validation import require_downstream_rx


def company_adapter(kind="gauge"):
    return TelemetryAdapter(
        timestamp_column="sample_time",
        entity_column="serial",
        metrics={"received_mw": "rx_power_dbm"},
        units={"rx_power_dbm": "mW"},
        kinds={"rx_power_dbm": kind},
        timezone="Europe/London",
    )


def test_different_company_same_canonical_measurement():
    native = pd.DataFrame(
        {
            "sample_time": ["2025-07-01 12:00"],
            "serial": ["A"],
            "received_mw": [0.01],
        }
    )
    expected = pd.DataFrame(
        {
            "time": pd.to_datetime(["2025-07-01 11:00Z"]),
            "device": ["A"],
            "rx_dbm": [-20.0],
        }
    )
    pd.testing.assert_frame_equal(
        company_adapter().transform(native),
        synthetic_adapter(["rx_dbm"]).transform(expected),
    )


def test_missing_mapped_column_and_cumulative_semantics_fail():
    native = pd.DataFrame(
        {
            "sample_time": ["2025-01-01"],
            "serial": ["A"],
            "received_mw": [0.01],
        }
    )
    with pytest.raises(ValueError, match="Missing mapped columns"):
        company_adapter().transform(native.drop(columns="received_mw"))
    with pytest.raises(ValueError, match="cumulative counters"):
        company_adapter("cumulative_count").transform(native)
    with pytest.raises(ValueError, match="Missing mapped columns"):
        synthetic_adapter().transform(pd.DataFrame({"time": [], "device": []}))


def test_temperature_adapts_but_does_not_satisfy_rx_detector():
    native = pd.DataFrame(
        {"time": ["2025-01-01"], "device": ["A"], "ont_temperature_c": [35.0]}
    )
    canonical = synthetic_adapter(["ont_temperature_c"]).transform(native)
    assert canonical.value.iloc[0] == 35
    with pytest.raises(ValueError, match="Downstream Rx"):
        require_downstream_rx(canonical)


def test_missing_rx_for_one_entity_is_explicit():
    native = pd.DataFrame(
        {"time": ["2025-01-01"] * 2, "device": ["A", "B"], "rx_dbm": [-20, np.nan]}
    )
    canonical = synthetic_adapter(["rx_dbm"]).transform(native)
    with pytest.raises(ValueError, match="B"):
        require_downstream_rx(canonical)
