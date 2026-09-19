"""Checks for pipeline."""

import pandas as pd
import pytest
import yaml
from optical_anomaly.pipeline import develop, final_evaluation


def test_frozen_end_to_end(tmp_path):
    settings = yaml.safe_load(open("configs/config.yaml"))
    settings["generator"].update(entities=3, days=8)
    settings["output"] = str(tmp_path / "run")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(settings))
    run = develop(path)
    assert not (run / "FINAL_OPENED.json").exists()
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
