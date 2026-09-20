"""Physical consistency, scope and unit semantics of expanded telemetry."""

from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from optical_anomaly.generator import generate, make_topology
from optical_anomaly.optics import fec_probabilities, fec_counts, validate_generated
from optical_anomaly.sources import synthetic_adapter
from optical_anomaly.validation import DataValidator


def test_fec_probability_and_count_conservation():
    ber = np.array([0.0, 1e-9, 1e-3, 0.1])
    corrected, uncorrectable = fec_probabilities(ber)
    assert corrected[0] == uncorrectable[0] == 0
    assert corrected[1] == pytest.approx(2040e-9, rel=0.001)
    assert corrected[-1] < corrected[2]
    assert uncorrectable[-1] > uncorrectable[2]
    total = np.full(4, 100000)
    c, u = fec_counts(ber, total, np.random.default_rng(7))
    assert ((c >= 0) & (u >= 0) & (c + u <= total)).all()


def test_topology_and_generated_invariants(generator_config, generated_data):
    data, truth = generated_data
    topology = make_topology(generator_config)
    assert set(topology.columns) == {
        "entity_id",
        "olt_id",
        "pon_port_id",
        "splitter_id",
    }
    assert topology.groupby("splitter_id").pon_port_id.nunique().eq(1).all()
    assert topology.groupby("pon_port_id").olt_id.nunique().eq(1).all()
    assert all(validate_generated(data, truth, topology)["checks"].values())
    assert not set(topology.columns).intersection(data.columns)


def test_shared_port_measurements_and_allocations(generator_config):
    config = replace(generator_config, missing_probability=0)
    data, _ = generate(config)
    powers = data.pivot(index="time", columns="device", values="olt_tx_dbm")
    assert powers.sub(powers.iloc[:, 0], axis=0).abs().max().eq(0).all()
    counts = data.groupby("time").upstream_fec_total_codewords.sum()
    physical_maximum = 1.24416e9 * config.interval_minutes * 60 / (255 * 8)
    assert counts.le(physical_maximum).all()
    assert data.groupby("device").first().upstream_rx_dbm.notna().all()


def test_sensor_noise_does_not_manufacture_errors_or_impact(
    generator_config, generated_data
):
    data, truth = generated_data
    changed, changed_truth = generate(replace(generator_config, sensor_noise_db=0.8))
    pd.testing.assert_series_equal(truth.impact_time, changed_truth.impact_time)
    columns = [c for c in data if "fec_" in c or c in ("ber", "upstream_ber")]
    pd.testing.assert_frame_equal(data[columns], changed[columns])
    assert not data.rx_dbm.equals(changed.rx_dbm)


def test_expanded_adapter_and_count_aggregation():
    times = pd.date_range("2025-01-01", periods=4, freq="5min", tz="UTC")
    field = "downstream_fec_corrected_codewords"
    data = pd.DataFrame(
        {
            "time": times,
            "device": "A",
            "rx_dbm": -20.0,
            "upstream_rx_dbm": -21.0,
            field: [np.nan, 2.0, 3.0, 4.0],
        }
    )
    adapted = synthetic_adapter(data.columns.drop(["time", "device"])).transform(data)
    assert "upstream_rx_power_dbm" in set(adapted.metric_name)
    aggregated = DataValidator("15min").transform(adapted)
    counts = aggregated.loc[aggregated.metric_name.eq(field)].set_index("timestamp")
    assert pd.isna(counts.value.iloc[0])
    assert counts.value.iloc[1] == 9


def test_counter_intervals_do_not_use_future_physical_state():
    from optical_anomaly.optics import error_telemetry

    original = np.full(20, -28.5)
    changed = original.copy()
    changed[10:] -= 3
    baseline = error_telemetry(
        original, original, (-27, -28), 5, 8, np.random.SeedSequence(7)
    )
    altered = error_telemetry(
        changed, original, (-27, -28), 5, 8, np.random.SeedSequence(7)
    )
    for column in baseline:
        if "fec_" in column:
            np.testing.assert_array_equal(baseline[column][:11], altered[column][:11])
    assert not np.array_equal(
        baseline["downstream_fec_corrected_codewords"][11:],
        altered["downstream_fec_corrected_codewords"][11:],
    )
