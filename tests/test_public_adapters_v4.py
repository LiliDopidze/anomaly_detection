"""Focused contract tests for optional public Telecom dataset adapters."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from telco_anomaly.adapters.microsoft_optical import (
    build_microsoft_optical_pack,
    microsoft_mapping_template,
)
from telco_anomaly.adapters.optical_failure import build_optical_failure_pack
from telco_anomaly.adapters.ran_pm import (
    build_ran_pm_pack,
    inspect_ran_pm,
    ran_mapping_template,
)
from telco_anomaly.contract import pack_fingerprint, read_pack


def gauge(native_field, metric_id, unit, *, direction="two_sided", cadence=900):
    return {
        "native_field": native_field,
        "metric_id": metric_id,
        "measurement_kind": "gauge",
        "unit": unit,
        "sampling_mode": "periodic",
        "aggregation_semantics": "instantaneous",
        "expected_cadence_seconds": cadence,
        "direction": direction,
        "transform": "identity",
        "minimum_scale": 0.01,
        "seasonality_candidate": False,
        "peer_eligible": True,
    }


class PublicAdapterTests(unittest.TestCase):
    def test_ran_discovery_does_not_guess_metric_meaning(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            pd.DataFrame({
                "time": ["2026-01-01T00:00:00Z"],
                "cell": ["cell-1"],
                "vendor_counter_17": [42.0],
            }).to_csv(source / "pm.csv", index=False)

            inventory = inspect_ran_pm(source)
            template = ran_mapping_template()

            self.assertIn("vendor_counter_17", inventory["numeric_columns_for_review"])
            self.assertEqual(template["metrics"], [])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                build_ran_pm_pack(source, Path(temporary) / "pack", template)

    def test_ran_build_uses_only_the_reviewed_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            pd.DataFrame({
                "time": pd.date_range("2026-01-01", periods=3, freq="15min", tz="UTC"),
                "cell": ["cell-1", "cell-1", "cell-2"],
                "site": ["site-a", "site-a", "site-b"],
                "traffic_mb": [10.0, 12.0, 8.0],
                "unmapped_note": ["a", "b", "c"],
            }).to_csv(source / "pm.csv", index=False)
            mapping = {
                "file_patterns": ["*.csv"],
                "csv_options": {},
                "timestamp_column": "time",
                "entity_column": "cell",
                "entity_type": "ran_cell",
                "source_system": "fixture_ran_pm",
                "episode_basis": "continuous_fixture",
                "topology": [{
                    "native_field": "site",
                    "group_type": "ran_site",
                    "hierarchy_level": 0,
                    "group_family": "physical_topology",
                }],
                "metrics": [
                    gauge("traffic_mb", "ran.traffic_volume", "MB", direction="contextual")
                ],
            }
            pack = Path(temporary) / "pack"
            manifest = build_ran_pm_pack(source, pack, mapping, batch_rows=2)

            telemetry = pd.concat(
                [pd.read_parquet(path) for path in sorted((pack / "PACK-CORE" / "telemetry").glob("*.parquet"))],
                ignore_index=True,
            )
            self.assertEqual(set(telemetry["metric_id"]), {"ran.traffic_volume"})
            self.assertNotIn("unmapped_note", telemetry.columns)
            self.assertEqual(manifest["evaluation_tables"], [])
            self.assertFalse(manifest["source"]["redistributed_by_project"])
            self.assertEqual(manifest["source"]["licence"], "CC BY 4.0")

    def test_microsoft_pack_cannot_claim_evaluation_truth(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            pd.DataFrame({
                "observed_at": pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC"),
                "channel": ["c1", "c1", "c2", "c2"],
                "path": ["p1", "p1", "p1", "p1"],
                "q": [15.1, 15.0, 14.9, 15.2],
                "tx": [-1.0, -1.1, -0.9, -1.0],
            }).to_csv(source / "optical.csv", index=False)

            mapping = microsoft_mapping_template(file_patterns=("*.csv",))
            mapping.update({
                "timestamp_column": "observed_at",
                "entity_column": "channel",
                "topology": [{
                    "native_field": "path",
                    "group_type": "optical_path",
                    "hierarchy_level": 0,
                    "group_family": "physical_topology",
                }],
                "metrics": [
                    gauge("q", "optical.q_factor", "unitless", direction="low_bad"),
                    gauge("tx", "optical.tx_power", "dBm"),
                ],
            })
            pack = Path(temporary) / "pack"
            manifest = build_microsoft_optical_pack(source, pack, mapping, batch_rows=2)

            self.assertEqual(manifest["evaluation_tables"], [])
            self.assertEqual(
                manifest["source"]["evidence_role"],
                "real_optical_behaviour_and_alert_stability",
            )
            self.assertFalse((pack / "PACK-EVAL").exists())
            self.assertFalse(manifest["source"]["redistributed_by_project"])

    def test_optical_failure_truth_is_physically_isolated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            redacted = root / "redacted"
            source.mkdir()
            redacted.mkdir()
            fixture = pd.DataFrame({
                "Timestamp": [0, 0, 3.5, 3.5, 7.0, 7.0, 10.5, 10.5],
                "Type": ["Devices", "Infrastructure"] * 4,
                "ID": ["SPO1/18/11", "Ampli1"] * 4,
                "BER": [0.0, None, 1e-7, None, 1e-5, None, 0.0, None],
                "OSNR": [22.0, None, 21.8, None, 17.0, None, 22.1, None],
                "InputPower": [None, -19.0, None, -19.1, None, -29.0, None, -19.0],
                "OutputPower": [None, 0.0, None, 0.0, None, -9.9, None, 0.1],
                "Failure": [0, 0, 0, 0, 1, 1, 0, 0],
            })
            fixture.to_csv(source / "HardFailure_dataset.csv", index=False)
            fixture.drop(columns="Failure").to_csv(
                redacted / "HardFailure_dataset.csv", index=False
            )

            labelled_pack = root / "labelled_pack"
            redacted_pack = root / "redacted_pack"
            build_optical_failure_pack(source, labelled_pack, include_evaluation=True, batch_rows=3)
            build_optical_failure_pack(redacted, redacted_pack, include_evaluation=False, batch_rows=3)

            labelled_manifest = read_pack(labelled_pack)
            telemetry = pd.concat(
                [pd.read_parquet(path) for path in sorted((labelled_pack / "PACK-CORE" / "telemetry").glob("*.parquet"))],
                ignore_index=True,
            )
            faults = pd.read_parquet(labelled_pack / "PACK-EVAL" / "fault_events.parquet")
            self.assertNotIn("Failure", telemetry.columns)
            self.assertEqual(len(faults), 1)
            self.assertEqual(faults.loc[0, "fault_type"], "hard_optical_failure")
            self.assertEqual(pack_fingerprint(labelled_pack), pack_fingerprint(redacted_pack))
            self.assertIn("fault_events", labelled_manifest["evaluation_tables"])
            self.assertFalse(labelled_manifest["source"]["redistributed_by_project"])

    def test_optical_failure_rejects_missing_requested_truth(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            pd.DataFrame({
                "Timestamp": [0, 3.5],
                "Type": ["Devices", "Devices"],
                "ID": ["SPO1/18/11", "SPO1/18/11"],
                "BER": [0.0, 0.0],
                "OSNR": [22.0, 22.1],
                "InputPower": [None, None],
                "OutputPower": [None, None],
            }).to_csv(source / "SoftFailure_dataset.csv", index=False)
            with self.assertRaisesRegex(ValueError, "no Failure field"):
                build_optical_failure_pack(
                    source, Path(temporary) / "pack", include_evaluation=True
                )


if __name__ == "__main__":
    unittest.main()
