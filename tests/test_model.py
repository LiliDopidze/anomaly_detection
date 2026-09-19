"""Scientific and operational properties of the compact workflow."""

import json

import numpy as np
import pandas as pd
import pytest

from telco_anomaly.adapter import adapt
from telco_anomaly.model import fit_reference, score, calibrate, warnings
from telco_anomaly.experiment import benchmark, final_evaluation
from telco_anomaly.synthetic import SyntheticConfig, generate_dataset
from telco_anomaly.evaluation import evaluate


def telemetry(n=400):
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "timestamp_utc": pd.date_range(
                "2025-01-01", periods=n, freq="15min", tz="UTC"
            ),
            "ont_id": "A",
            "rx_power_dbm": -20 + rng.normal(0, 0.2, n),
            "olt_rx_power_dbm": -21 + rng.normal(0, 0.2, n),
        }
    )


def test_adapter_units_and_label_independence():
    data = telemetry()
    native = data.rename(columns={"rx_power_dbm": "receive"})
    native["receive"] = 10 ** (native.receive / 10)
    native["truth"] = "fault"
    mapped = adapt(
        native,
        {"rx_power_dbm": "receive"},
        {"rx_power_dbm": "mW", "olt_rx_power_dbm": "dBm"},
    )
    pd.testing.assert_frame_equal(mapped, data, atol=1e-12)
    assert adapt(data.drop(columns="olt_rx_power_dbm")).olt_rx_power_dbm.isna().all()
    with pytest.raises(ValueError):
        adapt(data, units={"rx_power_dbm": "unknown"})
    with pytest.raises(ValueError, match="Duplicate"):
        adapt(pd.concat([data, data.iloc[:1]]))


def test_causality_company_offset_and_abstention():
    data = telemetry()
    model = fit_reference(data.iloc[:150])
    original = score(data, model)
    changed = data.copy()
    changed.loc[250:, "rx_power_dbm"] -= 5
    pd.testing.assert_frame_equal(original.iloc[:250], score(changed, model).iloc[:250])
    shifted = data.copy()
    shifted[["rx_power_dbm", "olt_rx_power_dbm"]] += 8
    other = score(shifted, fit_reference(shifted.iloc[:150]))
    np.testing.assert_allclose(original.score, other.score, atol=1e-12)
    unknown = data.assign(ont_id="new device")
    assert score(unknown, model).score.isna().all()
    missing = data.copy()
    missing.loc[200, ["rx_power_dbm", "olt_rx_power_dbm"]] = np.nan
    assert pd.isna(score(missing, model).score.iloc[200])
    # Serialization preserves the exact scoring calculation.
    pd.testing.assert_frame_equal(original, score(data, json.loads(json.dumps(model))))


def test_gradual_warning_is_not_backdated():
    data = telemetry(600)
    model = fit_reference(data.iloc[:200])
    threshold = calibrate(score(data.iloc[200:350], model).score)
    data.loc[350:, "rx_power_dbm"] -= np.linspace(0, 5, 250)
    alerts = warnings(score(data, model), threshold)
    assert data.timestamp_utc.iloc[350] < alerts.start_ts.iloc[0]
    assert alerts.start_ts.iloc[0] < data.timestamp_utc.iloc[550]
    abrupt = telemetry(600)
    abrupt.loc[350:, "rx_power_dbm"] -= 5
    alerts = warnings(score(abrupt, model), threshold)
    assert alerts.start_ts.min() >= abrupt.timestamp_utc.iloc[350]


def test_warning_gaps_and_valid_recovery():
    times = pd.date_range("2025-01-01", periods=7, freq="15min", tz="UTC")
    data = pd.DataFrame(
        {
            "ont_id": "A",
            "timestamp_utc": times,
            "score": [10, np.nan, 10, 10, np.nan, 0, 0],
        }
    )
    result = warnings(data, 5)
    assert result.start_ts.tolist() == [times[3]]
    assert result.end_ts.tolist() == [times[6]]
    assert result.closed_by.tolist() == ["recovered"]
    long = pd.DataFrame(
        {
            "ont_id": "A",
            "timestamp_utc": pd.date_range(times[0], periods=40, freq="15min"),
            "score": [10, 10] + [np.nan] * 38,
        }
    )
    result = warnings(long, 5)
    assert result.end_ts.iloc[0] == times[1]
    assert result.closed_by.iloc[0] == "gap"


def test_future_boundary_fault_does_not_excuse_false_warning(tmp_path):
    start = pd.Timestamp("2025-01-01", tz="UTC")
    end = start + pd.Timedelta(days=1)
    pd.DataFrame(
        {
            "gt_fault_id": ["late"],
            "gt_fault_type": ["gradual"],
            "onset_ts": [start + pd.Timedelta(hours=12)],
            "repair_ts": [end + pd.Timedelta(days=1)],
            "first_observable_ts": [start + pd.Timedelta(hours=12)],
            "impact_ts": [pd.NaT],
        }
    ).to_csv(tmp_path / "gt_fault_registry.csv", index=False)
    pd.DataFrame({"fault_id": ["late"], "entity_id": ["A"]}).to_csv(
        tmp_path / "fault_entity_intervals.csv", index=False
    )
    pd.DataFrame({"ont_id": ["A"]}).to_csv(tmp_path / "topology.csv", index=False)
    alerts = pd.DataFrame(
        {
            "incident_id": ["wrong", "boundary"],
            "entity_id": ["A", "A"],
            "start_ts": [start, start + pd.Timedelta(hours=13)],
            "end_ts": [end, end],
        }
    )
    metrics, _ = evaluate(tmp_path, alerts, start, end)
    assert metrics["false_alarms"] == 1
    assert metrics["boundary_incidents_excluded"] == 1


def test_benchmark_freeze_and_final_guard(tmp_path):
    config = SyntheticConfig(
        days=12,
        n_onts=4,
        n_olts=1,
        ports_per_olt=1,
        splitters_per_port=1,
        faults_per_ont_year=0,
        faults_per_splitter_year=0,
    )
    dataset = generate_dataset(config, tmp_path / "data")
    run = tmp_path / "run"
    result = benchmark(dataset, run)
    assert set(result.model) == {"ewma", "isolation_forest", "static"}
    assert not (run / "FINAL_TEST_OPENED.json").exists()
    saved = (run / "model.json").read_text()
    (run / "model.json").write_text(saved + " ")
    with pytest.raises(ValueError, match="Model changed"):
        final_evaluation(run)
    (run / "model.json").write_text(saved)
    metrics = final_evaluation(run)  # Disposable test data only.
    assert metrics["faults"] == 0
    assert metrics["false_alarms"] == metrics["incidents"]
    with pytest.raises(FileExistsError):
        final_evaluation(run)


def test_early_recall_counts_misses_and_separates_duplicates(tmp_path):
    start = pd.Timestamp("2025-01-01", tz="UTC")
    hour = pd.Timedelta(hours=1)
    pd.DataFrame(
        {
            "gt_fault_id": ["detected", "missed"],
            "gt_fault_type": ["gradual", "abrupt"],
            "onset_ts": [start + hour, start + 12 * hour],
            "repair_ts": [start + 8 * hour, start + 18 * hour],
            "first_observable_ts": [start + hour, start + 12 * hour],
            "impact_ts": [start + 5 * hour, start + 12 * hour],
        }
    ).to_csv(tmp_path / "gt_fault_registry.csv", index=False)
    pd.DataFrame({"fault_id": ["detected", "missed"], "entity_id": ["A", "A"]}).to_csv(
        tmp_path / "fault_entity_intervals.csv", index=False
    )
    pd.DataFrame({"ont_id": ["A"]}).to_csv(tmp_path / "topology.csv", index=False)
    alerts = pd.DataFrame(
        {
            "incident_id": ["early", "duplicate", "false"],
            "entity_id": ["A"] * 3,
            "start_ts": [start + h * hour for h in (2, 4, 20)],
            "end_ts": [start + h * hour for h in (3, 6, 21)],
        }
    )
    metrics, outcomes = evaluate(tmp_path, alerts, start, start + 24 * hour)
    assert metrics["detected"] == metrics["missed"] == 1
    assert metrics["early_recall"] == 0.5
    assert metrics["false_alarms"] == metrics["duplicate_warnings"] == 1
    assert outcomes.loc[outcomes.fault_id.eq("detected"), "lead_hours"].iloc[0] == 3
