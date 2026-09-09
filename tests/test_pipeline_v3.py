"""Small deterministic checks for the v3 contract, topology and evaluation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


CODE = Path(__file__).resolve().parents[1] / "notebooks" / "drive_research"
sys.path.insert(0, str(CODE))

from evaluation_core import ALERT_COLUMNS, evaluate_cases, form_cases
from milestone1_core import (
    CORE_SCHEMAS,
    EVAL_SCHEMAS,
    OPTIONAL_CORE_SCHEMAS,
    PACK_SCHEMAS,
    SPLIT_SCHEMAS,
    build_canonical,
    check_core,
    core_fingerprint,
    save_pack,
)
from milestone1_core import _series_batches
from simple_model_core import (
    alert_grid_from_score_file,
    calibration_thresholds,
    fit_topology_reference,
    score_topology_file,
)


BASE = pd.Timestamp("2025-01-01", tz="UTC")


def topology_fixture(entities):
    rows = []
    for number, entity in enumerate(entities):
        rows.extend([
            (entity, "pon_port", f"pon-{number // 8}", 1, "physical_topology"),
            (entity, "splitter_l1", f"pon-{number // 8}", 2, "physical_topology"),
            (entity, "geo_cluster", f"geo-{number // 8}", pd.NA, "geographic_context"),
        ])
    return pd.DataFrame(
        rows, columns=OPTIONAL_CORE_SCHEMAS["topology_memberships"]
    )


class ContractTests(unittest.TestCase):
    def test_gap_batches_never_split_a_metric_series(self):
        sizes = pd.DataFrame([
            ("episode-a", "m1", 80),
            ("episode-a", "m2", 60),
            ("episode-b", "m1", 40),
        ], columns=["episode_id", "metric_id", "observation_rows"])

        batches = list(_series_batches(sizes, maximum_rows=100))

        self.assertEqual(batches, [
            [("episode-a", "m1")],
            [("episode-a", "m2"), ("episode-b", "m1")],
        ])

    def test_truth_removal_does_not_change_model_visible_core(self):
        entities = [f"entity-{number:02d}" for number in range(8)]
        episodes = [f"episode-{number:02d}" for number in range(8)]
        rows = []
        for entity, episode in zip(entities, episodes):
            for step in range(10):
                rows.append((
                    BASE + pd.Timedelta(minutes=15 * step), entity, episode,
                    "signal", float(step), "measured",
                ))
        telemetry = pd.DataFrame(rows, columns=CORE_SCHEMAS["telemetry"])
        catalogue = pd.DataFrame([
            ("signal", "test_asset", "gauge", "unit", "periodic", 900),
        ], columns=PACK_SCHEMAS["metric_catalogue"])
        registry = pd.DataFrame(
            {"entity_id": entities, "entity_type": "test_asset"}
        )[PACK_SCHEMAS["entity_registry"]]
        observation_episodes = pd.DataFrame({
            "episode_id": episodes,
            "entity_id": entities,
            "episode_basis": "test_fixture",
        })[PACK_SCHEMAS["observation_episodes"]]
        splits = {"time_partitions": pd.DataFrame([
            ("calibration", BASE, BASE + pd.Timedelta(hours=1), "test-v1"),
            ("development", BASE + pd.Timedelta(hours=1), BASE + pd.Timedelta(hours=2), "test-v1"),
            ("holdout", BASE + pd.Timedelta(hours=2), BASE + pd.Timedelta(hours=3), "test-v1"),
        ], columns=SPLIT_SCHEMAS["time_partitions"])}
        fault_events = pd.DataFrame([(
            "fault-1", "test_fault", "pon_port", "pon-0", BASE, BASE,
            BASE + pd.Timedelta(minutes=15), BASE + pd.Timedelta(hours=1),
            pd.NA, "fixture", "fixture-1",
        )], columns=EVAL_SCHEMAS["fault_events"])
        fault_intervals = pd.DataFrame([
            ("fault-1", entity, BASE, BASE + pd.Timedelta(hours=1), "fixture", "fixture-1")
            for entity in entities
        ], columns=EVAL_SCHEMAS["fault_entity_intervals"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / "pack"
            save_pack(
                pack, sector="contract_test", pack_version="1",
                source_info={"source_id": "fixture", "files": []},
                telemetry=[telemetry], catalogue=catalogue, entities=registry,
                episodes=observation_episodes, topology=topology_fixture(entities),
                splits=splits,
                evaluation={
                    "fault_events": fault_events,
                    "fault_entity_intervals": fault_intervals,
                },
            )
            mounted, unmounted = root / "mounted", root / "unmounted"
            build_canonical(pack, mounted, include_evaluation=True)
            build_canonical(pack, unmounted, include_evaluation=False)

            self.assertEqual(
                core_fingerprint(mounted / "SPEC-CORE"),
                core_fingerprint(unmounted / "SPEC-CORE"),
            )
            self.assertTrue((mounted / "SPEC-CORE" / "topology_memberships.parquet").is_file())
            self.assertFalse((unmounted / "SPEC-EVAL").exists())
            self.assertEqual(
                check_core(mounted / "SPEC-CORE")["topology_memberships"],
                len(topology_fixture(entities)),
            )


class StatisticalTests(unittest.TestCase):
    def test_thresholds_use_quantiles_of_block_maxima(self):
        frame = pd.DataFrame({
            "entity_id": ["a"] * 3 + ["b"] * 3 + ["c"] * 3,
            "rapid_residual": [1, 2, 3, 2, 4, 6, 5, 7, 9],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.parquet"
            frame.to_parquet(path, index=False)
            result = calibration_thresholds(
                path, [0.5], block_column="entity_id", minimum_block_rows=1,
                model_ids=["rapid_residual"],
            )
        self.assertEqual(result.iloc[0]["threshold"], 6.0)
        self.assertEqual(
            result.iloc[0]["threshold_method"],
            "empirical_quantile_of_block_maxima",
        )

    def test_time_blocks_are_nested_within_entity(self):
        frame = pd.DataFrame({
            "event_ts": [
                BASE, BASE + pd.Timedelta(hours=1),
                BASE + pd.Timedelta(days=1), BASE + pd.Timedelta(days=1, hours=1),
            ],
            "entity_id": "a",
            "rapid_residual": [1.0, 2.0, 3.0, 4.0],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.parquet"
            frame.to_parquet(path, index=False)
            result = calibration_thresholds(
                path, [0.5], block_column="entity_id",
                block_duration_seconds=86_400, minimum_block_rows=1,
                model_ids=["rapid_residual"],
            )
        self.assertEqual(result.iloc[0]["threshold"], 3.0)
        self.assertEqual(result.iloc[0]["blocks_used"], 2)

    def test_peer_and_group_scores_are_calibrated_without_truth(self):
        entities = [f"entity-{number:02d}" for number in range(16)]
        rows = []
        for step in range(40):
            for number, entity in enumerate(entities):
                rows.append((
                    BASE + pd.Timedelta(minutes=15 * step), entity,
                    f"episode-{number:02d}",
                    np.sin(step / 5) + number / 100,
                    np.cos(step / 7) + number / 100,
                ))
        residuals = pd.DataFrame(rows, columns=[
            "event_ts", "entity_id", "episode_id", "m1__level", "m2__level",
        ])
        topology = topology_fixture(entities)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            residual_path, score_path = root / "residuals.parquet", root / "scores.parquet"
            residuals.to_parquet(residual_path, index=False)
            reference = fit_topology_reference(
                residual_path, topology, ["m1__level", "m2__level"],
                peer_group_type="pon_port", group_types=["pon_port", "splitter_l1"],
                min_peers=7, minimum_reference_rows=10, upper_quantile=0.95,
            )
            score_topology_file(
                residual_path, topology, reference, score_path,
                peer_group_type="pon_port", group_types=["pon_port", "splitter_l1"],
                min_peers=7,
            )
            scores = pd.read_parquet(score_path)

        self.assertFalse(reference.empty)
        self.assertTrue(scores["peer_deviation"].notna().all())
        self.assertTrue(scores["group_common_mode"].notna().all())
        self.assertEqual(set(scores["peer_valid_peers"]), {7})
        self.assertEqual(set(scores["group_available_fraction"]), {1.0})

    def test_equivalent_topology_scopes_are_reported_honestly(self):
        entities = ["entity-00", "entity-01", "entity-02"]
        topology = topology_fixture(entities)
        alerts = pd.DataFrame([
            ("alert-1", "group_common_mode", "entity-00", "episode-00", BASE, BASE, BASE, 10.0, 2, "m1", "splitter_l1", "pon-0", 2 / 3),
            ("alert-2", "group_common_mode", "entity-01", "episode-01", BASE, BASE, BASE, 9.0, 2, "m1", "splitter_l1", "pon-0", 2 / 3),
        ], columns=ALERT_COLUMNS)
        cases, members = form_cases(
            alerts, topology, gap_seconds=60, thresholds={"group_common_mode": 5.0}
        )
        events = pd.DataFrame([(
            "fault-1", "shared_fault", "pon_port", "pon-0", BASE, BASE,
            BASE + pd.Timedelta(minutes=10), BASE + pd.Timedelta(hours=1),
            pd.NA, "fixture", "fixture-1",
        )], columns=EVAL_SCHEMAS["fault_events"])
        intervals = pd.DataFrame([
            ("fault-1", entity, BASE, BASE + pd.Timedelta(hours=1), "fixture", "fixture-1")
            for entity in entities[:2]
        ], columns=EVAL_SCHEMAS["fault_entity_intervals"])
        result = evaluate_cases(
            cases, members, events, intervals,
            exposure_value=10, exposure_unit="entity_day",
            decision_horizon_seconds=3_600, topology_memberships=topology,
        )
        metrics = result["metrics"].set_index("metric")["value"]

        self.assertEqual(metrics["event_recall"], 1.0)
        self.assertEqual(metrics["equivalence_aware_localisation_accuracy"], 1.0)
        self.assertEqual(metrics["joint_detection_and_localisation_recall"], 1.0)
        self.assertFalse(result["localisation_results"].iloc[0]["truth_identifiable"])

    def test_unrelated_entity_alerts_need_common_mode_evidence_to_merge(self):
        topology = topology_fixture(["entity-00", "entity-01"])
        alerts = pd.DataFrame([
            ("alert-1", "rapid_residual", "entity-00", "episode-00", BASE, BASE, BASE, 10.0, 2, "m1", None, None, None),
            ("alert-2", "rapid_residual", "entity-01", "episode-01", BASE, BASE, BASE, 9.0, 2, "m1", None, None, None),
        ], columns=ALERT_COLUMNS)
        cases, _ = form_cases(
            alerts, topology, gap_seconds=60,
            thresholds={"rapid_residual": 5.0},
        )
        self.assertEqual(len(cases), 2)

        common_mode = pd.DataFrame([
            ("alert-3", "group_common_mode", "entity-00", "episode-00",
             BASE + pd.Timedelta(minutes=1), BASE + pd.Timedelta(minutes=1),
             BASE + pd.Timedelta(minutes=1), 8.0, 1, "m1", "splitter_l1", "pon-0", 1.0),
            ("alert-4", "group_common_mode", "entity-01", "episode-01",
             BASE + pd.Timedelta(minutes=1), BASE + pd.Timedelta(minutes=1),
             BASE + pd.Timedelta(minutes=1), 8.0, 1, "m1", "splitter_l1", "pon-0", 1.0),
        ], columns=ALERT_COLUMNS)
        shared = pd.DataFrame(
            [*alerts.to_dict("records"), *common_mode.to_dict("records")],
            columns=ALERT_COLUMNS,
        )
        cases, _ = form_cases(
            shared, topology, gap_seconds=60,
            thresholds={"rapid_residual": 5.0, "group_common_mode": 5.0},
        )
        self.assertEqual(len(cases), 1)

    def test_repeated_group_score_becomes_one_scoped_alert(self):
        rows = []
        for entity in ("entity-00", "entity-01"):
            for minute in range(3):
                rows.append((
                    BASE + pd.Timedelta(minutes=minute), entity,
                    f"episode-{entity}", 2.0, "m1", "splitter_l1", "pon-0", 2 / 3,
                ))
        scores = pd.DataFrame(rows, columns=[
            "event_ts", "entity_id", "episode_id", "group_common_mode",
            "group_common_mode__leading_feature",
            "group_common_mode__scope_type", "group_common_mode__scope_id",
            "group_common_mode__affected_fraction",
        ])
        thresholds = pd.DataFrame([{
            "model_id": "group_common_mode",
            "threshold_quantile": 0.95,
            "threshold": 1.0,
        }])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.parquet"
            scores.to_parquet(path, index=False)
            alert_sets = alert_grid_from_score_file(
                path, thresholds,
                persistence={"group_common_mode": 1},
                recovery_consecutive=1,
            )
        alerts = alert_sets[("group_common_mode", 0.95)]
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts.iloc[0]["evidence_scope_type"], "splitter_l1")
        self.assertEqual(alerts.iloc[0]["evidence_scope_id"], "pon-0")

    def test_overlapping_group_alert_copies_are_consolidated(self):
        from simple_model_core import _deduplicate_scope_alerts

        alerts = pd.DataFrame([
            ("A-1", "group_common_mode", "entity-00", "episode-00",
             BASE, BASE + pd.Timedelta(seconds=20), BASE + pd.Timedelta(seconds=10),
             8.0, 3, "m1", "splitter_l1", "pon-0", 0.5),
            ("A-2", "group_common_mode", "entity-01", "episode-01",
             BASE + pd.Timedelta(seconds=10), BASE + pd.Timedelta(seconds=30),
             BASE + pd.Timedelta(seconds=20), 9.0, 3, "m1",
             "splitter_l1", "pon-0", 0.75),
        ], columns=ALERT_COLUMNS)

        result = _deduplicate_scope_alerts(alerts)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["entity_id"], "entity-01")
        self.assertEqual(result.iloc[0]["alert_start"], BASE)
        self.assertEqual(
            result.iloc[0]["alert_end"], BASE + pd.Timedelta(seconds=30)
        )
        self.assertEqual(result.iloc[0]["evidence_affected_fraction"], 0.75)


if __name__ == "__main__":
    unittest.main()
