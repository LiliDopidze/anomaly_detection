"""Small end-to-end checks for the detector and incident mechanics."""

from __future__ import annotations

import numpy as np
import pandas as pd

import telco_anomaly.detectors as detector_helpers
from telco_anomaly.contract import OPTIONAL_CORE_SCHEMAS
from telco_anomaly.detectors import (
    _deduplicate_scope_alerts,
    append_contextual_isolation_scores,
    alert_grid_from_score_file,
    calibration_thresholds,
    fit_contextual_isolation_forest,
    fit_residual_bundle,
    split_feature_file_by_time,
    fit_topology_reference,
    score_topology_file,
    score_partition_file,
    score_residual_episode,
)
from telco_anomaly.evaluation import ALERT_COLUMNS, form_cases


BASE = pd.Timestamp("2025-01-01", tz="UTC")


def topology_fixture(entities):
    rows = []
    for number, entity in enumerate(entities):
        group = f"pon-{number // 8}"
        rows.extend([
            (entity, "pon_port", group, 1, "physical_topology", BASE, pd.NaT),
            (entity, "splitter_l1", group, 2, "physical_topology", BASE, pd.NaT),
        ])
    return pd.DataFrame(
        rows, columns=OPTIONAL_CORE_SCHEMAS["topology_memberships"]
    )


def test_calibration_feature_split_is_chronological_and_complete(tmp_path):
    source = tmp_path / "calibration.parquet"
    fit = tmp_path / "fit.parquet"
    threshold = tmp_path / "threshold.parquet"
    frame = pd.DataFrame({
        "event_ts": pd.date_range(BASE, periods=100, freq="15min"),
        "entity_id": "ont-1",
        "episode_id": "episode-1",
        "signal": np.arange(100, dtype=float),
    })
    frame.to_parquet(source, index=False)

    result = split_feature_file_by_time(
        source, fit, threshold, fit_fraction=0.70
    )
    fit_frame = pd.read_parquet(fit)
    threshold_frame = pd.read_parquet(threshold)

    assert len(fit_frame) + len(threshold_frame) == len(frame)
    assert fit_frame["event_ts"].max() < threshold_frame["event_ts"].min()
    assert result["fit_rows"] == len(fit_frame)
    assert result["threshold_rows"] == len(threshold_frame)


def test_partition_scoring_uses_the_frozen_cusum_policy(tmp_path, monkeypatch):
    feature_path = tmp_path / "features.parquet"
    destination = tmp_path / "scores.parquet"
    pd.DataFrame({"feature": [1.0]}).to_parquet(feature_path, index=False)
    observed = {}

    def fake_residual_score(bundle, source, target, **settings):
        observed.update(settings)
        pd.DataFrame({"score": [2.0]}).to_parquet(target, index=False)
        pd.DataFrame({"residual": [3.0]}).to_parquet(
            settings["residual_destination"], index=False
        )

    monkeypatch.setattr(detector_helpers, "score_residual_file", fake_residual_score)
    score_partition_file(
        {}, feature_path, destination, tmp_path / "work",
        cadence_seconds=900,
        dispersion_window_seconds=86_400,
        resolved_policy={"cusum_allowance": 0.25, "topology_enabled": False},
    )

    assert observed["cusum_allowance"] == 0.25
    assert pd.read_parquet(destination).loc[0, "score"] == 2.0


def test_entity_references_use_full_calibration_not_the_ml_sample(tmp_path):
    rows = []
    for entity_number in range(50):
        entity = f"ont-{entity_number:02d}"
        for step in range(40):
            rows.append({
                "event_ts": BASE + pd.Timedelta(minutes=15 * step),
                "entity_id": entity,
                "episode_id": f"{entity}::episode-1",
                "signal__level": float(entity_number + step / 10),
            })
    path = tmp_path / "calibration_fit.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)

    bundle = fit_residual_bundle(
        path,
        use_entity_reference=True,
        maximum_training_rows=30,
        fit_multivariate=False,
    )

    assert bundle["training_rows"] == 30
    assert len(bundle["entity_centre"]) == 50
    assert bundle["entity_reference_counts"].min().min() == 40


def test_contextual_isolation_forest_fits_and_scores(tmp_path):
    source = tmp_path / "combined_scores.parquet"
    destination = tmp_path / "contextual_scores.parquet"
    rows = 200
    frame = pd.DataFrame({
        "event_ts": pd.date_range(BASE, periods=rows, freq="15min"),
        "entity_id": [f"ont-{value % 10}" for value in range(rows)],
        "episode_id": [f"episode-{value % 10}" for value in range(rows)],
        "rapid_residual": np.sin(np.arange(rows) / 7),
        "peer_deviation": np.cos(np.arange(rows) / 11),
        "group_common_mode": np.sin(np.arange(rows) / 17),
    })
    frame.to_parquet(source, index=False)

    bundle = fit_contextual_isolation_forest(
        source,
        ["rapid_residual", "peer_deviation", "group_common_mode"],
        maximum_training_rows=150,
        trees=10,
        maximum_samples=64,
    )
    append_contextual_isolation_scores(bundle, source, destination)
    scored = pd.read_parquet(destination)

    assert len(scored) == rows
    assert scored["isolation_forest_contextual"].notna().all()
    assert set(scored["isolation_forest_contextual__leading_feature"]) <= set(
        bundle["feature_columns"]
    )


def test_isolation_forest_keeps_base_and_temporal_variants(tmp_path):
    path = tmp_path / "features.parquet"
    rows = 200
    frame = pd.DataFrame({
        "event_ts": pd.date_range(BASE, periods=rows, freq="15min"),
        "entity_id": "ont-1",
        "episode_id": "episode-1",
        "a__level": np.sin(np.arange(rows) / 7),
        "b__level": np.cos(np.arange(rows) / 11),
        "a__level__lag_1h": np.sin(np.arange(rows) / 5),
    })
    frame.to_parquet(path, index=False)

    bundle = fit_residual_bundle(
        path,
        use_entity_reference=False,
        maximum_training_rows=150,
        isolation_trees=10,
        isolation_max_samples=64,
        fit_multivariate=True,
    )
    scored = score_residual_episode(
        bundle,
        frame,
        cadence_seconds=900,
        dispersion_window_seconds=3600,
        cusum_allowance=0.5,
    )

    assert bundle["isolation_base_features"] == ["a__level", "b__level"]
    assert scored["isolation_forest_base"].notna().all()
    assert scored["isolation_forest_temporal"].notna().all()


def test_thresholds_are_quantiles_of_entity_day_block_maxima(tmp_path):
    scores = pd.DataFrame({
        "event_ts": [
            BASE,
            BASE + pd.Timedelta(hours=1),
            BASE + pd.Timedelta(days=1),
            BASE + pd.Timedelta(days=1, hours=1),
        ],
        "entity_id": "ont-1",
        "rapid_residual": [1.0, 2.0, 3.0, 4.0],
    })
    path = tmp_path / "scores.parquet"
    scores.to_parquet(path, index=False)

    result = calibration_thresholds(
        path,
        [0.5],
        block_column="entity_id",
        block_duration_seconds=86_400,
        minimum_block_rows=1,
        model_ids=["rapid_residual"],
    )

    assert result.loc[0, "threshold"] == 3.0
    assert result.loc[0, "blocks_used"] == 2


def test_scoped_threshold_completeness_counts_times_not_descendants(tmp_path):
    rows = []
    for step in range(6):
        for entity_number in range(8):
            rows.append({
                "event_ts": BASE + pd.Timedelta(minutes=15 * step),
                "entity_id": f"ont-{entity_number}",
                "group_common_mode": float(step),
                "group_common_mode__scope_type": "pon_port",
                "group_common_mode__scope_id": "pon-1",
            })
    path = tmp_path / "group_scores.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)

    with np.testing.assert_raises_regex(ValueError, "No calibration blocks"):
        calibration_thresholds(
            path,
            [0.95],
            block_column="entity_id",
            block_duration_seconds=86_400,
            minimum_block_rows=48,
            model_ids=["group_common_mode"],
        )


def test_peer_and_common_mode_scores_use_calibration_topology(tmp_path):
    entities = [f"ont-{number:02d}" for number in range(16)]
    rows = []
    for step in range(40):
        for number, entity in enumerate(entities):
            rows.append((
                BASE + pd.Timedelta(minutes=15 * step),
                entity,
                f"{entity}::episode-1",
                np.sin(step / 5) + number / 100,
                np.cos(step / 7) + number / 100,
            ))
    residuals = pd.DataFrame(rows, columns=[
        "event_ts", "entity_id", "episode_id", "m1__level", "m2__level",
    ])
    residual_path = tmp_path / "residuals.parquet"
    score_path = tmp_path / "scores.parquet"
    residuals.to_parquet(residual_path, index=False)
    topology = topology_fixture(entities)

    reference = fit_topology_reference(
        residual_path,
        topology,
        ["m1__level", "m2__level"],
        peer_group_type="pon_port",
        group_types=["pon_port", "splitter_l1"],
        min_peers=7,
        minimum_reference_rows=10,
        upper_quantile=0.95,
    )
    score_topology_file(
        residual_path,
        topology,
        reference,
        score_path,
        peer_group_type="pon_port",
        group_types=["pon_port", "splitter_l1"],
        min_peers=7,
    )
    scores = pd.read_parquet(score_path)

    assert scores["peer_deviation"].notna().all()
    assert scores["group_common_mode"].notna().all()
    assert set(scores["peer_valid_peers"]) == {7}


def test_repeated_group_score_becomes_one_physical_alert(tmp_path):
    rows = []
    for entity in ("ont-00", "ont-01"):
        for minute in range(3):
            rows.append((
                BASE + pd.Timedelta(minutes=minute),
                entity,
                f"{entity}::episode-1",
                2.0,
                "m1__level",
                "splitter_l1",
                "pon-0",
                2 / 3,
            ))
    scores = pd.DataFrame(rows, columns=[
        "event_ts", "entity_id", "episode_id", "group_common_mode",
        "group_common_mode__leading_feature",
        "group_common_mode__scope_type", "group_common_mode__scope_id",
        "group_common_mode__affected_fraction",
    ])
    path = tmp_path / "scores.parquet"
    scores.to_parquet(path, index=False)
    thresholds = pd.DataFrame([{
        "model_id": "group_common_mode",
        "threshold_quantile": 0.95,
        "threshold": 1.0,
    }])

    alert_sets = alert_grid_from_score_file(
        path,
        thresholds,
        persistence={"group_common_mode": 1},
        recovery_consecutive=1,
    )
    alerts = alert_sets[("group_common_mode", 0.95)]

    assert len(alerts) == 1
    assert alerts.loc[0, "evidence_scope_type"] == "splitter_l1"
    assert alerts.loc[0, "evidence_scope_id"] == "pon-0"


def test_unrelated_entity_alerts_merge_only_with_shared_scope_evidence():
    topology = topology_fixture(["ont-00", "ont-01"])
    ordinary = pd.DataFrame([
        ("A-1", "rapid_residual", "ont-00", "e-0", BASE, BASE, BASE,
         10.0, 2, "m1", None, None, None),
        ("A-2", "rapid_residual", "ont-01", "e-1", BASE, BASE, BASE,
         9.0, 2, "m1", None, None, None),
    ], columns=ALERT_COLUMNS)

    cases, _ = form_cases(
        ordinary,
        topology,
        gap_seconds=60,
        thresholds={"rapid_residual": 5.0},
    )
    assert len(cases) == 2

    common_mode = pd.DataFrame([
        ("A-3", "group_common_mode", "ont-00", "e-0", BASE, BASE, BASE,
         8.0, 1, "m1", "pon_port", "pon-0", 1.0),
        ("A-4", "group_common_mode", "ont-01", "e-1", BASE, BASE, BASE,
         8.0, 1, "m1", "pon_port", "pon-0", 1.0),
    ], columns=ALERT_COLUMNS)
    alerts = pd.DataFrame(
        [*ordinary.to_dict("records"), *common_mode.to_dict("records")],
        columns=ALERT_COLUMNS,
    )
    cases, _ = form_cases(
        alerts,
        topology,
        gap_seconds=60,
        thresholds={"rapid_residual": 5.0, "group_common_mode": 5.0},
    )
    assert len(cases) == 1


def test_overlapping_copies_of_one_scope_are_consolidated():
    alerts = pd.DataFrame([
        ("A-1", "group_common_mode", "ont-00", "e-0", BASE,
         BASE + pd.Timedelta(seconds=20), BASE + pd.Timedelta(seconds=10),
         8.0, 3, "m1", "pon_port", "pon-0", 0.50),
        ("A-2", "group_common_mode", "ont-01", "e-1",
         BASE + pd.Timedelta(seconds=10), BASE + pd.Timedelta(seconds=30),
         BASE + pd.Timedelta(seconds=20), 9.0, 3, "m1",
         "pon_port", "pon-0", 0.75),
    ], columns=ALERT_COLUMNS)

    result = _deduplicate_scope_alerts(alerts)

    assert len(result) == 1
    assert result.loc[0, "entity_id"] == "ont-01"
    assert result.loc[0, "alert_start"] == BASE
    assert result.loc[0, "alert_end"] == BASE + pd.Timedelta(seconds=30)
    assert result.loc[0, "evidence_affected_fraction"] == 0.75
