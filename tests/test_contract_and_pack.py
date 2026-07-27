from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from telemetry_contract import CONTRACT_VERSION, SectorPack
from telemetry_packs.telecom import PACK_VERSION, load_pack

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


class ContractAndPackTests(unittest.TestCase):
    def test_telecom_pack_implements_protocol(self):
        pack = load_pack()
        self.assertIsInstance(pack, SectorPack)
        self.assertEqual(pack.pack_version, PACK_VERSION)
        self.assertIn(CONTRACT_VERSION, pack.compatible_contract_versions)
        self.assertGreaterEqual(len(pack.metric_specs()), 10)
        self.assertEqual(
            set(pack.relation_specs()),
            {
                "contains_pon",
                "contains_splitter_l1",
                "contains_splitter_l2",
                "serves_ont",
                "groups_ont",
            },
        )

    def test_catalogue_carries_preprocessing_semantics(self):
        metrics = load_pack().metric_specs()
        self.assertEqual(
            metrics["telecom.link.fec_count"].censoring_type,
            "right_censored_at_generator_ceiling",
        )
        self.assertEqual(
            metrics["telecom.link.crc_errors"].exposure_semantics,
            "transmitted_bit_opportunities_under_frozen_generator",
        )
        for metric in metrics.values():
            self.assertTrue(metric.sampling_semantics)
            self.assertTrue(metric.aggregation_semantics)
            self.assertTrue(metric.expected_behaviour_profile)

    def test_generic_contract_has_no_telecom_branch(self):
        forbidden = ("telecom", "telco", "gpon", "ont_id", "rx_power")
        for path in (SRC / "telemetry_contract").rglob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            for token in forbidden:
                self.assertNotIn(token, text, f"{path} contains sector token {token}")

    def test_runtime_imports_no_evaluation_package(self):
        for path in (SRC / "telemetry_runtime").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.append(node.module)
            self.assertFalse(
                any(name.startswith("telemetry_eval") for name in imported),
                f"{path} imports evaluation code: {imported}",
            )

    def test_pack_has_no_truth_fields(self):
        for path in (SRC / "telemetry_packs" / "telecom").glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("gt_", json.dumps(payload).lower(), str(path))


if __name__ == "__main__":
    unittest.main()
