from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks" / "google_drive" / "milestone1_v0_3"


class NotebookQualityTests(unittest.TestCase):
    def test_maintained_notebooks_are_reviewable_and_portable(self):
        notebooks = sorted(NOTEBOOK_ROOT.glob("*.ipynb"))
        self.assertEqual(
            [path.name for path in notebooks],
            [
                "02_M1_V03_TELECOM_MATERIALISATION.ipynb",
                "03_M1_V03_WEEK1_ACCEPTANCE.ipynb",
                "04_M1_V03_PETROBRAS_3W_CHALLENGE.ipynb",
            ],
        )
        for path in notebooks:
            document = json.loads(path.read_text(encoding="utf-8"))
            sources = [
                "".join(cell.get("source", []))
                for cell in document["cells"]
                if cell.get("cell_type") == "code"
            ]
            self.assertFalse(any("SOURCE_FILES =" in source for source in sources), path)
            self.assertFalse(any("/Users/" in source for source in sources), path)
            self.assertFalse(any("def " in source for source in sources), path)
            self.assertLess(max(map(len, sources)), 3_000, path)

    def test_each_notebook_calls_one_named_workflow(self):
        expected = {
            "02_M1_V03_TELECOM_MATERIALISATION.ipynb": "materialise_telecom(",
            "03_M1_V03_WEEK1_ACCEPTANCE.ipynb": "run_week1_acceptance(",
            "04_M1_V03_PETROBRAS_3W_CHALLENGE.ipynb": "challenge_threew(",
        }
        for filename, call in expected.items():
            document = json.loads((NOTEBOOK_ROOT / filename).read_text(encoding="utf-8"))
            source = "\n".join(
                "".join(cell.get("source", [])) for cell in document["cells"]
            )
            self.assertEqual(source.count(call), 1, filename)


if __name__ == "__main__":
    unittest.main()
