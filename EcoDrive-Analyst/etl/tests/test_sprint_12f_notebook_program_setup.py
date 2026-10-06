from __future__ import annotations

import json
import os
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "notebooks" / "notebook_manifest.json"
NOTEBOOK = ROOT / "notebooks" / "12F_01_epa_ingest_and_grain.ipynb"


class Sprint12FNotebookProgramSetupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    def test_manifest_defines_eleven_ordered_live_coding_slices(self) -> None:
        notebooks = self.manifest["notebooks"]
        self.assertEqual([item["order"] for item in notebooks], list(range(1, 12)))
        self.assertEqual(len({item["id"] for item in notebooks}), 11)
        self.assertTrue(all(item["live_coding_slice"] for item in notebooks))

    def test_mvp_contains_all_planned_notebooks(self) -> None:
        planned_paths = [ROOT / item["path"] for item in self.manifest["notebooks"]]
        self.assertTrue(all(path.exists() for path in planned_paths))

    def test_first_notebook_input_is_portable_and_available(self) -> None:
        first = self.manifest["notebooks"][0]
        self.assertTrue(all(not Path(path).is_absolute() for path in first["inputs"]))
        self.assertTrue(all((ROOT / path).is_file() for path in first["inputs"]))

    def test_first_notebook_executes_and_loads_epa_source(self) -> None:
        namespace: dict[str, object] = {}
        original = Path.cwd()
        try:
            os.chdir(ROOT / "notebooks")
            for cell in self.notebook["cells"]:
                if cell["cell_type"] == "code":
                    exec("".join(cell["source"]), namespace)
        finally:
            os.chdir(original)
        self.assertEqual(namespace["epa"].shape, (30194, 67))
        self.assertEqual(int(namespace["grain_counts"]["source_rows"]), 30194)


if __name__ == "__main__":
    unittest.main()
