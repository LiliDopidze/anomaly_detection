"""Tests of generator meaning and temporal isolation, not just file existence."""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from telco_anomaly.synthetic import (
    SyntheticConfig,
    ar1,
    corrected_codeword_probability,
    generate_dataset,
    make_topology,
    random_stream,
    sample_faults,
    sha256,
)
from telco_anomaly.synthetic_pipeline import (
    build_features,
    evaluate,
    load_observations,
    make_incidents,
    run_holdout,
    split_boundaries,
)
from telco_anomaly.synthetic_validation import validate_dataset


@pytest.fixture
def small_config():
    return SyntheticConfig(
        days=8,
        n_onts=4,
        n_olts=1,
        ports_per_olt=1,
        splitters_per_port=1,
        faults_per_ont_year=100,
        faults_per_splitter_year=100,
    )


def test_reproducible_generation_and_invariants(tmp_path, small_config):
    first = generate_dataset(small_config, tmp_path / "a")
    second = generate_dataset(small_config, tmp_path / "b")
    manifest = json.loads((first / "manifest.json").read_text())
    for name in manifest["files"]:
        assert sha256(first / name) == sha256(second / name)
    report = validate_dataset(first)
    assert report["checks"].status.eq("pass").all()
    with pytest.raises(FileExistsError):
        generate_dataset(small_config, first)
    observations, manifest = load_observations(first)
    assert observations.timestamp_utc.max() < split_boundaries(manifest)["holdout"][0]
    assert not any("fault" in name for name in observations)


def test_collection_and_sensor_changes_do_not_change_physics(tmp_path, small_config):
    first = generate_dataset(small_config, tmp_path / "a")
    altered = replace(small_config, sensor_noise_db=0.5, poll_loss_probability=0.3)
    second = generate_dataset(altered, tmp_path / "b")
    pd.testing.assert_frame_equal(
        sample_faults(small_config, make_topology(small_config)),
        sample_faults(altered, make_topology(altered)),
    )
    a = pd.read_parquet(first / "reference_dataset.parquet")
    b = pd.read_parquet(second / "reference_dataset.parquet")
    joined = a.merge(b, on=["ont_id", "timestamp_utc"], suffixes=("_a", "_b"))
    for metric in ("fec_count", "throughput_mbps", "temperature_c"):
        np.testing.assert_allclose(
            joined[metric + "_a"], joined[metric + "_b"], equal_nan=True
        )
    a = pd.read_csv(first / "fault_entity_intervals.csv")
    b = pd.read_csv(second / "fault_entity_intervals.csv")
    pd.testing.assert_series_equal(a.impact_ts, b.impact_ts)


def test_ou_variance_and_time_scaled_correlation():
    for dt in (0.25, 1):
        values = ar1(200_000, 4, 2, dt, random_stream(17, str(dt)))
        assert np.var(values) == pytest.approx(4, rel=0.05)
        assert np.corrcoef(values[:-1], values[1:])[0, 1] == pytest.approx(
            np.exp(-dt / 4),
            abs=0.01,
        )


def test_corrected_codewords_have_physical_probability():
    p = corrected_codeword_probability(np.array([0, 1e-9, 0.001, 0.1]))
    assert p[0] == 0
    assert p[1] == pytest.approx(2040e-9, rel=0.001)
    assert np.all((p >= 0) & (p <= 1))
    # At severe corruption, codewords become uncorrectable, not ever increasing.
    assert p[3] < p[2]


def test_features_are_causal_and_do_not_bridge_missing_history():
    n = 400
    data = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range(
                "2025-01-01", periods=n, freq="15min", tz="UTC"
            ),
            "ont_id": "A",
            "rx_power_dbm": -20.0,
            "olt_rx_power_dbm": -21.0,
            "tx_power_dbm": 2.0,
            "bias_current_ma": 20.0,
            "fec_count": 0.0,
        }
    )
    before = build_features(data)
    changed = data.copy()
    changed.loc[200:, "rx_power_dbm"] = -30
    after = build_features(changed)
    pd.testing.assert_frame_equal(before.iloc[:200], after.iloc[:200])
    assert after.loc[200, "rx_power_dbm__z6h"] > 50
    gapped = pd.concat([data.iloc[:100], data.iloc[300:]])
    features = build_features(gapped)
    assert pd.isna(features.loc[100, "rx_power_dbm__z6h"])


def test_incidents_use_actual_decision_time_and_reset_at_gaps():
    times = pd.to_datetime(
        [
            "2025-01-01T00:00Z",
            "2025-01-01T00:15Z",
            "2025-01-01T02:00Z",
            "2025-01-01T02:15Z",
        ]
    )
    scores = pd.DataFrame({"ont_id": "A", "timestamp_utc": times, "robust": 10.0})
    incidents = make_incidents(scores, "robust", 5, 900)
    assert incidents.start_ts.tolist() == [times[1], times[3]]
    assert incidents.end_ts.iloc[0] == times[1]


def test_matching_does_not_lose_overlap_event(tmp_path):
    start, end = pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp(
        "2025-01-03", tz="UTC"
    )
    pd.DataFrame(
        {
            "gt_fault_id": ["long", "short"],
            "gt_fault_type": ["loss"] * 2,
            "onset_ts": [start, start + pd.Timedelta(hours=1)],
            "repair_ts": [end, start + pd.Timedelta(hours=4)],
            "first_observable_ts": [start, start],
            "impact_ts": [pd.NaT] * 2,
        }
    ).to_csv(tmp_path / "gt_fault_registry.csv", index=False)
    pd.DataFrame({"fault_id": ["long", "short"], "entity_id": ["A"] * 2}).to_csv(
        tmp_path / "fault_entity_intervals.csv",
        index=False,
    )
    pd.DataFrame({"ont_id": ["A"]}).to_csv(tmp_path / "topology.csv", index=False)
    incidents = pd.DataFrame(
        {
            "incident_id": ["I1", "I2"],
            "entity_id": ["A"] * 2,
            "start_ts": [start + pd.Timedelta(hours=h) for h in (2, 6)],
            "end_ts": [start + pd.Timedelta(hours=h) for h in (3, 7)],
        }
    )
    metrics, _ = evaluate(tmp_path, incidents, start, end)
    assert metrics["detected"] == 2
    assert metrics["nuisance_incidents"] == 0


def test_holdout_requires_selected_model(tmp_path):
    with pytest.raises(ValueError, match="STOP"):
        run_holdout(tmp_path)


def test_invalid_config_rejected(small_config):
    with pytest.raises(ValueError):
        replace(small_config, n_onts=1000).validate()
    with pytest.raises(ValueError):
        replace(small_config, sample_minutes=17).validate()


@pytest.mark.parametrize("changed", ["model", "threshold", "dataset", "code"])
def test_holdout_rejects_changed_frozen_artifacts(tmp_path, small_config, changed):
    from telco_anomaly import synthetic_pipeline

    dataset = generate_dataset(small_config, tmp_path / "data")
    run = tmp_path / "run"
    run.mkdir()
    (run / "selected_configuration.json").write_text(
        json.dumps({"model": "robust", "threshold": 10})
    )
    # Rejection must happen before any deserialization or final scoring.
    (run / "baselines.joblib").write_bytes(b"not loaded")
    receipt = {
        "dataset": str(dataset),
        "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
        "pipeline_sha256": sha256(synthetic_pipeline.__file__),
        "frozen_files": {
            name: sha256(run / name)
            for name in ("baselines.joblib", "selected_configuration.json")
        },
    }
    targets = {
        "model": run / "baselines.joblib",
        "threshold": run / "selected_configuration.json",
        "dataset": dataset / "reference_dataset.parquet",
    }
    if changed == "code":
        receipt["pipeline_sha256"] = "changed"
    else:
        target = targets[changed]
        target.write_bytes(target.read_bytes() + b"changed")
    (run / "run.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="changed"):
        run_holdout(run)
    assert not (run / "holdout").exists()
