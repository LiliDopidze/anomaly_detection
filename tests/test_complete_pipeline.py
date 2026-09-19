import json
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import yaml

from telco_anomaly.pipeline import (
    runtime,
    prepare,
    develop,
    run_holdout,
    infer,
    ledger,
    frozen_run,
)
from telco_anomaly.features import regularize, channel_scores
from telco_anomaly.operations import incident_queue


@pytest.fixture
def policy(tmp_path):
    p = runtime(Path(__file__).parents[1] / "configs/pipeline.yml")
    generator = yaml.safe_load(Path(p["generator"]).read_text())
    generator.update(days=14, n_onts=8, n_olts=1, ports_per_olt=1, splitters_per_port=2)
    generator_path = tmp_path / "generator.yml"
    generator_path.write_text(yaml.safe_dump(generator))
    experiment = yaml.safe_load(Path(p["experiment"]).read_text())
    # Deliberately permissive fixture gates exercise the selected/holdout paths.
    # Production configuration remains unchanged.
    experiment.update(
        verification_budget_per_1000_days=1e6,
        development_budget_per_1000_days=1e6,
        minimum_event_recall_lower_bound=0.0,
    )
    experiment_path = tmp_path / "experiment.yml"
    experiment_path.write_text(yaml.safe_dump(experiment))
    p.update(
        generator=str(generator_path),
        experiment=str(experiment_path),
        pack=str(tmp_path / "pack"),
        truth=str(tmp_path / "truth"),
        run=str(tmp_path / "run"),
    )
    return p


def test_complete_workflow_freeze_holdout_and_inference_guards(policy, tmp_path):
    prepare(policy)
    result = develop(policy)
    assert result.qualified.any()
    run = Path(policy["run"])
    choices = run / "label_free_choices.json"
    entries = [
        json.loads(line)
        for line in (Path(policy["truth"]) / "selection.jsonl").read_text().splitlines()
    ]
    assert len(entries) == 1
    assert choices.exists()
    frozen_run(run)
    with pytest.raises(ValueError, match="Inference must follow"):
        infer(run, policy["pack"], tmp_path / "inference")
    # Future inference must work while evaluation truth is unavailable.
    from telco_anomaly.adapter import write_pack, load_mapping

    pack = Path(policy["pack"])
    metadata = json.loads((pack / "manifest.json").read_text())["config"]
    future = pd.read_parquet(pack / "telemetry.parquet")
    future.timestamp_utc += pd.Timedelta(days=31)
    metadata["start"] = str(pd.Timestamp(metadata["start"]) + pd.Timedelta(days=31))
    inventory = pd.read_parquet(pack / "inventory.parquet").drop(
        columns=["valid_from", "valid_to"]
    )
    future_pack = write_pack(
        future,
        inventory,
        tmp_path / "future_pack",
        load_mapping(policy["adapter"]),
        metadata,
    )
    truth = Path(policy["truth"])
    hidden_truth = tmp_path / "unmounted_truth"
    truth.rename(hidden_truth)
    try:
        infer(run, future_pack, tmp_path / "future_inference")
        assert (tmp_path / "future_inference/incident_queue.csv").exists()
    finally:
        hidden_truth.rename(truth)
    first = run_holdout(run)
    assert first["faults"] >= 0
    import shutil

    shutil.rmtree(run / "holdout")
    with pytest.raises(ValueError, match="attempt limit"):
        run_holdout(run)
    selected = run / "selected_configuration.json"
    selected.write_text(selected.read_text() + " ")
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        frozen_run(run)


def test_separate_truth_and_prepare_lineage(policy):
    prepare(policy)
    assert not list(Path(policy["pack"]).rglob("*fault*"))
    prepare(policy)
    path = Path(policy["generator"])
    value = yaml.safe_load(path.read_text())
    value["seed"] += 1
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="differs"):
        prepare(policy)


def test_ledger_limits_failed_or_deleted_runs(tmp_path):
    ledger(tmp_path, "holdout", {"status": "opening"}, maximum=1)
    with pytest.raises(ValueError, match="attempt limit"):
        ledger(tmp_path, "holdout", {}, maximum=1)
    assert not (tmp_path / ".ledger.lock").exists()


def test_effective_topology_and_gap_classification():
    t = pd.Timestamp("2025-01-01", tz="UTC")
    times = pd.date_range(t, periods=5, freq="15min")
    data = pd.DataFrame(
        {
            "timestamp_utc": times[[0, 2, 4]],
            "ont_id": "A",
            "uptime_s": [10000, 11800, 100],
            "reboot_count": [0, 0, 1],
        }
    )
    inventory = pd.DataFrame(
        {
            "ont_id": ["A", "A"],
            "splitter_l2": ["old", "new"],
            "valid_from": [t, t + pd.Timedelta(minutes=45)],
            "valid_to": [t + pd.Timedelta(minutes=45), t + pd.Timedelta(days=1)],
        }
    )
    grid = regularize(data, inventory, t, t + pd.Timedelta(minutes=75), 900)
    assert grid.gap_kind.tolist() == [
        "continuous",
        "unobserved",
        "collection",
        "unobserved",
        "device_restart",
    ]
    assert grid.splitter_l2.tolist() == ["old", "old", "old", "new", "new"]


def test_group_silence_is_not_collector_outage():
    p = runtime(Path(__file__).parents[1] / "configs/pipeline.yml")
    t = pd.Timestamp("2025-01-01", tz="UTC")
    grid = pd.DataFrame(
        {
            "timestamp_utc": [t] * 6,
            "ont_id": list("ABCDEF"),
            "splitter_l2": ["S1"] * 3 + ["S2"] * 3,
            "olt_id": "O",
            "reported": [False] * 3 + [True] * 3,
            "gap_kind": "unobserved",
            "rx_power_dbm": np.nan,
            "rx_sensitivity_dbm": -28,
        }
    )
    features = pd.DataFrame(
        {"rx_power_dbm__delta": [np.nan] * 3 + [0] * 3, "rx_power_dbm__z24h": np.nan}
    )
    base = grid[["timestamp_utc", "ont_id"]].copy()
    scores = channel_scores(grid, features, base, p)
    assert scores.silence.iloc[:3].eq(1).all()
    grid.reported = False
    scores = channel_scores(grid, features, base, p)
    assert scores.silence.eq(0).all()


def test_scope_ambiguity_and_power_events_use_whole_scope():
    p = runtime(Path(__file__).parents[1] / "configs/pipeline.yml")
    t = pd.Timestamp("2025-01-01", tz="UTC")
    inventory = pd.DataFrame(
        {
            "ont_id": list("ABC"),
            "olt_id": "O",
            "pon_port": "P",
            "splitter_l1": "L1",
            "splitter_l2": "L2",
            "valid_from": t,
            "valid_to": t + pd.Timedelta(days=1),
        }
    )
    alerts = pd.DataFrame(
        {
            "entity_id": list("ABC"),
            "start_ts": t + pd.Timedelta(hours=1),
            "end_ts": t + pd.Timedelta(hours=2),
        }
    )
    events = pd.DataFrame(
        {
            "ont_id": list("AB"),
            "timestamp_utc": t + pd.Timedelta(minutes=50),
            "family": "onu_power_loss",
        }
    )
    queue = incident_queue(alerts, inventory, events, p)
    assert len(queue) == 1
    assert queue.ambiguous.iloc[0]
    assert queue.disposition.iloc[0] == "probable_power_loss"
    assert queue.power_event_share.iloc[0] == pytest.approx(2 / 3)
    assert "splitter_l1:L1" in queue.top_scope.iloc[0]


def test_thirty_informative_causal_prefixes():
    from telco_anomaly.synthetic_pipeline import build_features

    rng = np.random.default_rng(9)
    n = 420
    data = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range(
                "2025-01-01", periods=n, freq="15min", tz="UTC"
            ),
            "ont_id": "A",
            "rx_power_dbm": -20 + np.cumsum(rng.normal(0, 0.1, n)),
            "olt_rx_power_dbm": -21 + rng.normal(0, 0.2, n),
            "tx_power_dbm": 2 + rng.normal(0, 0.1, n),
            "bias_current_ma": 20 + rng.normal(0, 0.2, n),
            "fec_count": rng.poisson(3, n),
        }
    )
    data.loc[180:210, "rx_power_dbm"] = np.nan
    full = build_features(data)
    informative = 0
    for cutoff in range(120, 420, 10):
        prefix = build_features(data.iloc[:cutoff])
        pd.testing.assert_frame_equal(prefix, full.iloc[:cutoff])
        informative += prefix.filter(like="__z").iloc[-1].gt(0).any()
    assert informative >= 24
