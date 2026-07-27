from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from anomaly_detection.telecom import SyntheticGponAdapter
from anomaly_detection.core import (
    ContractViolation,
    assert_event_as_of,
    validate_core_bundle,
)
from anomaly_detection.runtime import (
    DetectorConfig,
    RobustHistoryDetector,
    score_core_directory,
)
from anomaly_detection.workflows import (
    materialise_telecom,
    run_week1_acceptance,
)

from .fixtures import evaluation_frames, native_frames


def write_native_fixture(root: Path) -> None:
    panel, topology, windows, tickets, engineering = native_frames()
    registry, intervals, groups = evaluation_frames()
    panel.to_parquet(root / "reference_dataset.parquet", index=False)
    topology.to_csv(root / "topology.csv", index=False)
    windows.to_csv(root / "entity_service_windows.csv", index=False)
    engineering.to_csv(root / "engineering_events.csv", index=False)
    tickets.to_csv(root / "tickets.csv", index=False)
    evaluation = root / "evaluation"
    evaluation.mkdir()
    registry.to_csv(evaluation / "gt_fault_registry.csv", index=False)
    intervals.to_csv(evaluation / "fault_entity_intervals.csv", index=False)
    groups.to_csv(evaluation / "gt_fault_groups.csv", index=False)


class AdapterIsolationTests(unittest.TestCase):
    def setUp(self):
        self.adapter = SyntheticGponAdapter()
        (
            self.panel,
            self.topology,
            self.service_windows,
            self.tickets,
            self.engineering,
        ) = native_frames()

    def test_core_is_an_explicit_truth_free_projection(self):
        core = self.adapter.adapt_core_frames(
            self.panel,
            self.topology,
            self.service_windows,
            tickets=self.tickets,
            engineering_events=self.engineering,
        )
        self.assertEqual(validate_core_bundle(core), ())
        for table_name, frame in core.items():
            leaked = [column for column in frame.columns if str(column).startswith("gt_")]
            self.assertEqual(leaked, [], table_name)
        self.assertNotIn("reported_symptom", core["telemetry"].columns)
        self.assertEqual(
            set(core["operational_events"]["event_type"]),
            {"planned_maintenance"},
        )
        self.assertNotIn("entity_service_windows", core)
        self.assertNotIn("ticket_id", " ".join(
            " ".join(map(str, frame.columns)) for frame in core.values()
        ))

    def test_quality_and_exposure_sensitive_branches_are_exercised(self):
        telemetry = self.adapter.adapt_telemetry(self.panel, cadence_seconds=3600)
        clipped = telemetry.loc[
            telemetry["metric_id"].eq("telecom.link.fec_count")
            & telemetry["quality_code"].eq("clipped")
        ]
        invalid = telemetry.loc[telemetry["quality_code"].eq("invalid")]
        self.assertGreater(len(clipped), 0, "fixture must exercise clipped values")
        self.assertGreater(len(invalid), 0, "fixture must exercise invalid values")

        fec_exposure = telemetry.loc[
            telemetry["metric_id"].eq("telecom.link.fec_count"), "exposure"
        ]
        self.assertEqual(fec_exposure.nunique(), 1)
        self.assertEqual(float(fec_exposure.iloc[0]), 2_488_000_000.0 * 3600)

        crc = telemetry.loc[
            telemetry["metric_id"].eq("telecom.link.crc_errors")
        ].sort_values(["event_ts", "entity_id"])
        self.assertGreater(crc["exposure"].nunique(), 1)
        first = crc.iloc[0]
        native = self.panel.loc[
            pd.to_datetime(self.panel["timestamp_utc"], utc=True).eq(first["event_ts"])
            & self.panel["ont_id"].astype(str).eq(first["entity_id"])
        ].iloc[0]
        expected = float(native["throughput_mbps"]) * 1_000_000 * 3600 / (1500 * 8)
        self.assertAlmostEqual(float(first["exposure"]), expected)

    def test_truth_timestamp_canary_never_reaches_core_values(self):
        canary = pd.Timestamp("2099-12-31T23:59:59.123456Z")
        tickets = self.tickets.copy()
        tickets["reported_ts"] = canary
        tickets["resolved_ts"] = canary
        core = self.adapter.adapt_core_frames(
            self.panel,
            self.topology,
            self.service_windows,
            tickets=tickets,
            engineering_events=self.engineering,
        )
        for table_name, frame in core.items():
            for column in frame.columns:
                if "ts" not in str(column).lower() and "time" not in str(column).lower():
                    continue
                values = pd.to_datetime(frame[column], utc=True, errors="coerce")
                self.assertFalse(values.eq(canary).any(), f"{table_name}.{column}")

        leaky = {name: frame.copy() for name, frame in core.items()}
        leaky["operational_events"]["event_end"] = pd.to_datetime(
            leaky["operational_events"]["event_end"], utc=True
        )
        leaky["operational_events"].loc[:, "event_end"] = canary
        leaked_values = pd.to_datetime(
            leaky["operational_events"]["event_end"], utc=True, errors="coerce"
        )
        self.assertTrue(
            leaked_values.eq(canary).any(),
            "negative control must prove the value-level harness can observe a leak",
        )

    def test_as_of_guard_rejects_delayed_event_until_known(self):
        event = {
            "event_start": pd.Timestamp("2026-01-01T10:00:00Z"),
            "known_at": pd.Timestamp("2026-01-01T10:05:00Z"),
        }
        with self.assertRaises(ContractViolation):
            assert_event_as_of(event, pd.Timestamp("2026-01-01T10:02:00Z"))
        assert_event_as_of(event, pd.Timestamp("2026-01-01T10:06:00Z"))

    def test_missingness_is_derived_without_truth_reason_file(self):
        core = self.adapter.adapt_core_frames(
            self.panel, self.topology, self.service_windows
        )
        gaps = core["collection_gaps"]
        ont2 = gaps.loc[gaps["entity_id"].eq("ONT-00002")]
        self.assertEqual(len(ont2), 1)
        self.assertEqual(
            pd.Timestamp(ont2.iloc[0]["gap_start"]),
            pd.Timestamp("2026-01-01T03:00:00Z"),
        )
        self.assertNotIn("reason", " ".join(gaps.columns).lower())

    def test_detector_output_is_identical_after_eval_mount_removed(self):
        core = self.adapter.adapt_core_frames(
            self.panel,
            self.topology,
            self.service_windows,
            tickets=self.tickets,
            engineering_events=self.engineering,
        )
        registry, intervals, groups = evaluation_frames()
        evaluation = self.adapter.adapt_eval_frames(
            registry, intervals, groups, tickets=self.tickets
        )
        detector = RobustHistoryDetector(
            DetectorConfig(history_window=4, minimum_history=2, threshold=4.0)
        )
        with tempfile.TemporaryDirectory() as temp_root:
            root = Path(temp_root)
            core_dir = root / "spec-core"
            eval_dir = root / "spec-eval"
            self.adapter.write_bundle(core, core_dir, file_format="csv")
            self.adapter.write_bundle(evaluation, eval_dir, file_format="csv")
            with_eval = score_core_directory(core_dir, detector)
            shutil.rmtree(eval_dir)
            without_eval = score_core_directory(core_dir, detector)
        assert_frame_equal(with_eval, without_eval, check_exact=True)

    def test_eval_mapping_does_not_change_core_frames(self):
        before = self.adapter.adapt_core_frames(
            self.panel,
            self.topology,
            self.service_windows,
            tickets=self.tickets,
            engineering_events=self.engineering,
        )
        registry, intervals, groups = evaluation_frames()
        self.adapter.adapt_eval_frames(registry, intervals, groups, tickets=self.tickets)
        after = self.adapter.adapt_core_frames(
            self.panel,
            self.topology,
            self.service_windows,
            tickets=self.tickets,
            engineering_events=self.engineering,
        )
        for table_name in before:
            assert_frame_equal(before[table_name], after[table_name], check_exact=True)

    def test_redacting_truth_columns_and_tickets_does_not_change_core(self):
        original = self.adapter.adapt_core_frames(
            self.panel,
            self.topology,
            self.service_windows,
            tickets=self.tickets,
            engineering_events=self.engineering,
        )
        redacted_panel = self.panel.drop(
            columns=[column for column in self.panel if column.startswith("gt_")]
        )
        redacted_topology = self.topology.drop(
            columns=[column for column in self.topology if column.startswith("gt_")]
        )
        redacted = self.adapter.adapt_core_frames(
            redacted_panel,
            redacted_topology,
            self.service_windows,
            tickets=None,
            engineering_events=self.engineering,
        )
        for table_name in original:
            assert_frame_equal(
                original[table_name],
                redacted[table_name],
                check_exact=True,
            )

    def test_discovery_accepts_supplied_root_and_data_split(self):
        with tempfile.TemporaryDirectory() as temp_root:
            root = Path(temp_root)
            data = root / "Data"
            data.mkdir()
            for filename in (
                "reference_dataset.parquet",
                "topology.csv",
                "fault_entity_intervals.csv",
                "gt_fault_registry.csv",
            ):
                (data / filename).touch()
            for filename in (
                "entity_service_windows.csv",
                "gt_fault_groups.csv",
            ):
                (root / filename).touch()
            inventory = self.adapter.discover(root)
        self.assertTrue(inventory.core_ready, inventory.notes)
        self.assertTrue(inventory.evaluation_ready, inventory.notes)
        self.assertIn("Data/reference_dataset.parquet", inventory.tables)

    def test_named_workflows_materialise_and_accept_a_sensitive_fixture(self):
        with tempfile.TemporaryDirectory() as temp_root:
            root = Path(temp_root)
            source = root / "native"
            source.mkdir()
            write_native_fixture(source)
            run_root = root / "run"
            materialisation = materialise_telecom(
                source,
                run_root,
                batch_native_rows=5,
                memory_budget_gib=8,
            )
            acceptance = run_week1_acceptance(source, run_root)

        self.assertTrue(materialisation["memory"]["budget_pass"])
        self.assertTrue(acceptance["translator_invariance"]["passed"])
        self.assertTrue(acceptance["value_level_leakage"]["negative_control_detected"])
        self.assertGreater(
            acceptance["materialised_telemetry"]["quality_counts"]["clipped"], 0
        )
        self.assertGreater(
            acceptance["materialised_telemetry"]["quality_counts"]["invalid"], 0
        )


if __name__ == "__main__":
    unittest.main()
