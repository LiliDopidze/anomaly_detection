from __future__ import annotations

import unittest

from anomaly_detection.evaluation import validate_eval_bundle
from anomaly_detection.telecom import SyntheticGponAdapter

from .fixtures import evaluation_frames


class GroupedFaultTests(unittest.TestCase):
    def test_regional_group_links_distinct_fault_domains(self):
        registry, intervals, groups = evaluation_frames()
        bundle = SyntheticGponAdapter().adapt_eval_frames(registry, intervals, groups)
        self.assertEqual(validate_eval_bundle(bundle), ())

        events = bundle["gt_fault_events"]
        grouped = events.loc[events["cause_group_id"].eq("STORM-001")]
        self.assertEqual(grouped["fault_event_id"].nunique(), 2)
        self.assertEqual(grouped["fault_domain_type"].nunique(), 2)
        self.assertEqual(set(grouped["fault_domain_type"]), {"l1", "l2"})

        affected = bundle["gt_fault_entity_intervals"]
        self.assertGreaterEqual(
            affected.loc[affected["fault_event_id"].eq("F-00001"), "affected_entity_id"].nunique(),
            2,
        )
        self.assertEqual(
            set(bundle["gt_cause_groups"]["cause_group_id"]), {"STORM-001"}
        )

    def test_group_identifier_never_appears_in_runtime_package(self):
        from pathlib import Path

        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "anomaly_detection"
            / "runtime.py"
        )
        self.assertNotIn(
            "cause_group_id", path.read_text(encoding="utf-8"), str(path)
        )


if __name__ == "__main__":
    unittest.main()
