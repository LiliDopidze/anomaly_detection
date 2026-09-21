"""Checks for pipeline."""

import pandas as pd
import joblib
import json
import pytest
import yaml
from optical_anomaly.pipeline import develop, final_evaluation


def test_frozen_end_to_end(tmp_path, monkeypatch):
    # A complete test-owned configuration; no repository-relative reads.
    settings = {
        "generator": {
            "seed": 42,
            "entities": 3,
            "days": 8,
            "interval_minutes": 5,
            "noise_db": 0.08,
            "sensor_noise_db": 0.04,
            "correlation_hours": 0.5,
            "daily_amplitude_db": 0.25,
            "missing_probability": 0.02,
            "impact_threshold_dbm": -27.0,
            "faults_per_entity": 2,
            "upstream_impact_threshold_dbm": -28.0,
            "fault_duration_median_hours": 36.0,
            "onts_per_splitter": 8,
            "splitters_per_port": 2,
            "ports_per_olt": 4,
        },
        "features": {
            "window": 12,
            "smoothing_hours": 1.0,
            "allowance": 0.25,
            "seasonal": True,
        },
        "splits": [0.4, 0.55, 0.75, 1.0],
        "incidents": {
            "high": 0.99,
            "low": 0.8,
            "opening_candidates": [2, 3, 6],
            "closing_candidates": [3, 6],
            "nuisance_budget_per_1000_days": 5.0,
        },
        "output": str(tmp_path / "run"),
    }
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(settings))
    run = develop(path)
    assert not (run / "FINAL_OPENED.json").exists()
    topology = pd.read_parquet(run / "topology.parquet")
    assert len(topology) == 3
    assert (run / "generation_checks.json").exists()
    scores = pd.read_parquet(run / "validation_scores.parquet")
    assert (
        scores[["statistical", "isolation_forest", "combined"]]
        .stack()
        .between(0, 1)
        .all()
    )
    comparison = pd.read_csv(run / "validation_comparison.csv")
    assert set(comparison.feature_set) == {
        "rx_only",
        "both_rx",
        "tx_rx",
        "fec",
        "temperature",
    }
    assert comparison.score_coverage.between(0, 1).all()
    canonical = pd.read_parquet(run / "canonical_development.parquet")
    model = joblib.load(run / "model.joblib")
    assert canonical.timestamp.max() < model["split"].validation_end
    from optical_anomaly.workflow import feature_data
    from optical_anomaly.pipeline import score_detectors

    features, _, _ = feature_data(run, settings, model["engineers"])
    replay = score_detectors(features, model["statistical"], model["forest"])
    replay = replay.loc[model["split"].masks(replay.timestamp)["validation"]]
    pd.testing.assert_frame_equal(replay.reset_index(drop=True), scores)
    assert json.loads((run / "manifest.json").read_text())["feature_set"] in set(
        comparison.feature_set
    )
    assert model["feature_set"] == "temperature"
    assert len(model["forest"].feature_columns) == 52
    assert set(model["forests"]) == set(comparison.feature_set)
    assert not (run / "FINAL_OPENED.json").exists()
    import importlib.util

    if importlib.util.find_spec("shap"):
        from optical_anomaly.explanations import explain_run

        report = explain_run(
            run, sample_size=2, background_size=2, local_size=1, permutations=1
        )
        assert len(pd.read_csv(report / "importance.csv")) == 52
        assert not (run / "FINAL_OPENED.json").exists()
    result = final_evaluation(run)  # Disposable test fixture, not the development run.
    assert result["monitored_entity_days"] > 0
    with pytest.raises(FileExistsError):
        final_evaluation(run)


def test_runner_rejects_contaminated_calibration():
    from optical_anomaly.pipeline import check_calibration_truth

    end = pd.Timestamp("2025-01-02", tz="UTC")
    with pytest.raises(ValueError, match="contains labelled faults"):
        check_calibration_truth(
            pd.DataFrame({"onset_time": [end - pd.Timedelta(seconds=1)]}), end
        )
    check_calibration_truth(pd.DataFrame({"onset_time": [end]}), end)
