from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class LegacyFreezeTests(unittest.TestCase):
    def test_frozen_attachment_hashes(self):
        manifest = json.loads(
            (ROOT / "legacy" / "attachment-manifest.json").read_text(encoding="utf-8")
        )
        for asset in manifest["assets"]:
            self.assertEqual(len(asset["sha256"]), 64)
            int(asset["sha256"], 16)
            path = ROOT / asset["path"]
            if path.is_file():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(digest, asset["sha256"], asset["path"])

    def test_baseline_gap_is_explicit_not_silently_reconstructed(self):
        freeze = json.loads(
            (ROOT / "legacy" / "baseline-freeze.json").read_text(encoding="utf-8")
        )
        self.assertFalse(freeze["executable_source_available"])
        self.assertEqual(freeze["status"], "descriptor_frozen_pending_source")
        self.assertIn("synthetic generator", freeze["reason"])

    def test_quarantined_generator_matches_its_frozen_manifest(self):
        root = ROOT / "legacy" / "telemetry_synth_v4"
        manifest = json.loads(
            (root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["status"], "frozen_exact_source")
        for relative_path, expected in manifest["files"].items():
            path = root / relative_path
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                expected,
                relative_path,
            )

    def test_generator_is_not_an_installed_source_package(self):
        self.assertEqual(
            list((ROOT / "src" / "telemetry_synth").glob("*.py")),
            [],
        )


if __name__ == "__main__":
    unittest.main()
