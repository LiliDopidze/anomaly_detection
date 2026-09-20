"""Checks for pipeline."""

import pandas as pd
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
    result = final_evaluation(run)  # Disposable test fixture, not the development run.
    assert result["monitored_entity_days"] > 0
    with pytest.raises(FileExistsError):
        final_evaluation(run)
