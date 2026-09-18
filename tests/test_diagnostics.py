from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from telco_anomaly.contract import EVAL_SCHEMAS
from telco_anomaly.diagnostics import (
    _child,
    candidate_frontier,
    classify_faults,
    numeric_profile,
    replay,
    run_diagnostics,
    score_evidence,
)
from telco_anomaly.evaluation import _fault_windows, scoreable_exposure
from telco_anomaly.io import file_sha256, read_json, write_json


def fixture_data(tmp_path):
    start = pd.Timestamp("2025-01-01", tz="UTC")
    scores = pd.DataFrame(
        {
            "event_ts": pd.date_range(start, periods=8, freq="15min"),
            "entity_id": "ONT-1",
            "episode_id": "E1",
            "detector": [0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0],
            "detector__leading_feature": "optical.rx_power",
        }
    )
    path = tmp_path / "development_scores.parquet"
    scores.to_parquet(path, index=False)
    events = pd.DataFrame(
        [
            [
                "F1",
                "damage",
                "entity",
                "ONT-1",
                start,
                start,
                start + pd.Timedelta(hours=1),
                start + pd.Timedelta(hours=2),
                None,
                "fixture",
                "fixture",
            ],
            [
                "F2",
                "damage",
                "entity",
                "ONT-2",
                start,
                start,
                start + pd.Timedelta(hours=1),
                start + pd.Timedelta(hours=2),
                None,
                "fixture",
                "fixture",
            ],
        ],
        columns=EVAL_SCHEMAS["fault_events"],
    )
    intervals = pd.DataFrame(
        [
            [r.fault_id, r.domain_id, r.onset_ts, r.end_ts, "fixture", "fixture"]
            for r in events.itertuples()
        ],
        columns=EVAL_SCHEMAS["fault_entity_intervals"],
    )
    config = dict(
        channels=["detector"],
        thresholds={"detector": 1.0},
        persistence_observations={"detector": 2},
        recovery_observations=1,
        recovery_threshold_fraction=0.8,
        cadence_seconds=900,
        incident_quiet_period_seconds=0,
        prompt_detection_horizon_seconds=172800,
    )
    topology = pd.DataFrame(
        [[e, "splitter", "S1", 1, "physical_topology"] for e in ["ONT-1", "ONT-2"]],
        columns=[
            "entity_id",
            "group_type",
            "group_id",
            "hierarchy_level",
            "group_family",
        ],
    )
    return path, scores, events, intervals, config, topology


def test_score_evidence_resets_missing_gaps_and_excludes_end(tmp_path):
    path, scores, events, intervals, config, _ = fixture_data(tmp_path)
    scores["detector"] = [2.0, np.nan, 2.0, 2.0, 2.0, 0.0, 0.0, 99.0]
    # 45-minute gap interrupts what would otherwise be a 3-observation run.
    scores.loc[3:, "event_ts"] += pd.Timedelta(minutes=30)
    scores.to_parquet(path, index=False)
    events.loc[0, "end_ts"] = scores.event_ts.iloc[-1]
    intervals.loc[0, "end_ts"] = scores.event_ts.iloc[-1]
    evidence, availability = score_evidence(
        path, _fault_windows(events, intervals, None), config
    )
    row = evidence.iloc[0]
    assert row.max_window_local_high_run == 2
    assert row.peak_score == 2  # endpoint's 99 is excluded
    assert row.valid_scores == 6
    assert availability.valid.sum() == 7


def test_missing_entity_is_not_threshold_failure(tmp_path):
    path, _, events, intervals, config, topology = fixture_data(tmp_path)
    alerts, _, _, result = replay(path, config, topology, events, intervals, 1.0)
    windows = _fault_windows(events, intervals, None)
    evidence, _ = score_evidence(path, windows, config)
    findings = classify_faults(
        result["fault_results"], evidence, alerts, windows
    ).set_index("fault_id")
    assert (
        findings.loc["F2", "evidence_category"] == "no_score_rows_in_affected_windows"
    )
    assert (result["case_matches"].match_status == "duplicate").sum() == 1


def test_frontier_does_not_promote_ineligible_candidates():
    d = candidate_frontier(
        pd.DataFrame(
            {
                "prompt_event_recall": [0.2, 0.4, 0.1, np.nan],
                "false_incidents_per_entity_day": [0.1, 0.1, 0.2, 0.01],
                "selection_eligible": [True, False, True, True],
            }
        )
    )
    assert d.prompt_workload_pareto.tolist() == [False, True, False, False]
    assert not d.loc[1, "selection_eligible"]


def test_path_and_partition_guards(tmp_path):
    with pytest.raises(PermissionError):
        _child(tmp_path, "holdout_locked/data.parquet")
    with pytest.raises(ValueError):
        _child(tmp_path, "../escape.parquet")
    results = tmp_path / "results"
    results.mkdir()
    write_json(results / "evaluation_manifest.json", {"partition": "holdout"})
    with pytest.raises(PermissionError):
        run_diagnostics(results, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_numeric_profile_nonfinite_and_constants(tmp_path):
    path = tmp_path / "features.parquet"
    pd.DataFrame(
        {"a": [1.0, np.inf, np.nan], "b": [2.0, 2.0, 2.0], "label": ["x"] * 3}
    ).to_parquet(path)
    d = numeric_profile(path).set_index("feature")
    assert d.loc["a", "finite_rows"] == 1
    assert d.loc["b", "constant"]
    assert "label" not in d.index


def build_full_fixture(tmp_path):
    path, scores, events, intervals, config, topology = fixture_data(tmp_path)
    root = tmp_path / "data"

    def stage(name, run):
        p = root / name / "synthetic_pon" / run
        p.mkdir(parents=True)
        return p

    model = stage("models", "model1")
    feature = stage("features", "features1")
    core_run = stage("core", "core1")
    core = core_run / "SPEC-CORE"
    core.mkdir()
    splits = core_run / "SPLITS"
    splits.mkdir()
    truth = stage("evaluation", "truth1")
    dev = truth / "development"
    dev.mkdir()
    selection = stage("selection", "selection1")
    results = tmp_path / "results"
    results.mkdir()
    scores.to_parquet(model / path.name, index=False)
    feats = scores.drop(columns=["detector__leading_feature"])
    feats.to_parquet(feature / "development_features.parquet", index=False)
    feats.to_parquet(feature / "calibration_features.parquet", index=False)
    topology.to_parquet(core / "topology_memberships.parquet", index=False)
    start = scores.event_ts.min()
    pd.DataFrame(
        {
            "partition": ["development"],
            "start_ts": [start],
            "end_ts": [start + pd.Timedelta(days=1)],
        }
    ).to_parquet(splits / "time_partitions.parquet", index=False)
    write_json(core / "manifest.json", {"fingerprint": "core1"})
    events.to_parquet(dev / "fault_events.parquet", index=False)
    intervals.to_parquet(dev / "fault_entity_intervals.parquet", index=False)
    write_json(
        truth / "truth_manifest.json",
        {
            "model_core_fingerprint": "core1",
            "file_sha256": {
                "development/" + p.name: file_sha256(p) for p in dev.iterdir()
            },
        },
    )
    write_json(
        feature / "feature_manifest.json",
        {
            "core_fingerprint": "core1",
            "time_partitions_sha256": file_sha256(splits / "time_partitions.parquet"),
            "partitions": {
                "development": {"features": "development_features.parquet"},
                "calibration_fit": {"features": "calibration_features.parquet"},
            },
        },
    )
    write_json(
        model / "model_manifest.json",
        {"core_fingerprint": "core1", "score_files": {"development": path.name}},
    )
    write_json(model / "resolved_policy.json", {})
    write_json(selection / "selected_configuration.json", config)
    exposure = scoreable_exposure(path, config["channels"], "entity_day", 900)
    _, _, _, result = replay(path, config, topology, events, intervals, exposure)
    for name, frame in result.items():
        frame.to_parquet(results / f"{name}.parquet", index=False)
    import telco_anomaly.evaluation as evaluation
    import telco_anomaly.selection as selection_module

    files = dict(
        selected_configuration_sha256=selection / "selected_configuration.json",
        model_manifest_sha256=model / "model_manifest.json",
        feature_manifest_sha256=feature / "feature_manifest.json",
        truth_manifest_sha256=truth / "truth_manifest.json",
        resolved_policy_sha256=model / "resolved_policy.json",
        evaluation_module_sha256=Path(evaluation.__file__),
        selection_module_sha256=Path(selection_module.__file__),
    )
    write_json(
        results / "evaluation_manifest.json",
        {
            "partition": "development",
            "core_fingerprint": "core1",
            "model_run_id": "model1",
            "selection_run_id": "selection1",
            **{k: file_sha256(p) for k, p in files.items()},
        },
    )
    telemetry = core / "telemetry"
    telemetry.mkdir()
    scores.assign(metric_id="optical.rx_power", value=1.0, quality_code="valid")[
        ["event_ts", "entity_id", "episode_id", "metric_id", "value", "quality_code"]
    ].to_parquet(telemetry / "part.parquet", index=False)
    return root, results, model


def test_full_replay_trace_and_immutable_output(tmp_path):
    root, results, _ = build_full_fixture(tmp_path)
    output = tmp_path / "diagnostics"
    kwargs = dict(
        data_root=root, core_run="core1", feature_run="features1", truth_run="truth1"
    )
    run_diagnostics(
        results, output, trace_fault="F1", trace_limit=2, deep_data=True, **kwargs
    )
    assert read_json(output / "diagnostic_manifest.json")["replay_verified"]
    trace = pd.read_parquet(output / "trace_manifest.parquet")
    assert trace.truncated.all()
    faults = pd.read_parquet(output / "fault_diagnostics.parquet").set_index("fault_id")
    assert not faults.loc["F2", "detected"]
    incidents = pd.read_parquet(output / "incident_status_counts.parquet").set_index(
        "match_status"
    )
    assert incidents.loc["duplicate", "incidents"] == 1
    with pytest.raises(FileExistsError):
        run_diagnostics(results, output, **kwargs)


def test_tampered_inputs_abort_without_publishing(tmp_path):
    root, results, model = build_full_fixture(tmp_path)
    write_json(model / "resolved_policy.json", {"tampered": True}, overwrite=True)
    with pytest.raises(ValueError, match="Lineage mismatch"):
        run_diagnostics(
            results,
            tmp_path / "out",
            data_root=root,
            core_run="core1",
            feature_run="features1",
            truth_run="truth1",
        )
    assert not (tmp_path / "out").exists()


def test_out_of_partition_scores_abort(tmp_path):
    root, results, model = build_full_fixture(tmp_path)
    path = model / "development_scores.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "event_ts"] += pd.Timedelta(days=10)
    frame.to_parquet(path, index=False)
    with pytest.raises(PermissionError, match="outside development"):
        run_diagnostics(
            results,
            tmp_path / "out",
            data_root=root,
            core_run="core1",
            feature_run="features1",
            truth_run="truth1",
        )
    assert not (tmp_path / "out").exists()


def test_case_trace_and_results_only_mode(tmp_path):
    root, results, _ = build_full_fixture(tmp_path)
    run_diagnostics(results, tmp_path / "summary")
    assert not read_json(tmp_path / "summary/diagnostic_manifest.json")[
        "holdout_opened"
    ]
    assert not (tmp_path / "summary/score_evidence.parquet").exists()
    run_diagnostics(
        results,
        tmp_path / "full",
        data_root=root,
        trace_case="C-000002",
        core_run="core1",
        feature_run="features1",
        truth_run="truth1",
    )
    manifest = pd.read_parquet(tmp_path / "full/trace_manifest.parquet")
    assert manifest.trace_id.eq("C-000002").all()
