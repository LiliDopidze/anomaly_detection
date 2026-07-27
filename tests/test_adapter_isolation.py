from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from telemetry_adapters import SyntheticGponAdapter
from telemetry_contract import validate_core_bundle
from telemetry_runtime import DetectorConfig, RobustHistoryDetector, score_core_directory

from .fixtures import evaluation_frames, native_frames


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


if __name__ == "__main__":
    unittest.main()
