from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks" / "google_drive" / "milestone1_v0_3"


class NotebookQualityTests(unittest.TestCase):
    def test_maintained_notebooks_are_reviewable_and_portable(self):
        notebooks = sorted(NOTEBOOK_ROOT.glob("*.ipynb"))
        self.assertEqual(len(notebooks), 4)
        for path in notebooks:
            document = json.loads(path.read_text(encoding="utf-8"))
            sources = [
                "".join(cell.get("source", []))
                for cell in document["cells"]
                if cell.get("cell_type") == "code"
            ]
            self.assertFalse(any("SOURCE_FILES =" in source for source in sources), path)
            self.assertFalse(any("/Users/" in source for source in sources), path)
            self.assertLess(max(map(len, sources)), 10_000, path)


if __name__ == "__main__":
    unittest.main()
