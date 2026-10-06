from __future__ import annotations

import contextlib
import io
import json
import os
import sqlite3
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "notebooks" / "notebook_manifest.json"
RUNTIME_HASHES = {
    ROOT / "data" / "db" / "eco_drive.db": "CC27CDF22AE189E39F9F45609CCD23C650E822BB20655C552FF2BD89D97F4262",
    ROOT / "data" / "db" / "eco_drive_qa.db": "0EB4A5EC0F402E6B9E1F7D76372EA30068670B16F994D495394613DAFA2CA569",
}


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


class Sprint12FNotebookMVPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_all_eleven_notebooks_are_valid_mvp_artifacts(self) -> None:
        entries = self.manifest["notebooks"]
        self.assertEqual(len(entries), 11)
        self.assertTrue(all(entry["status"] == "MVP_EXECUTABLE" for entry in entries))
        for entry in entries:
            payload = json.loads((ROOT / entry["path"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["nbformat"], 4)
            self.assertTrue(any(cell["cell_type"] == "code" for cell in payload["cells"]))
        by_id = {entry["id"]: entry for entry in entries}
        self.assertEqual(by_id["12F-02"]["inputs"], ["etl/data/raw/epa_testcar/epa_testcar_2026_raw.xlsx"])
        self.assertEqual(by_id["12F-03"]["inputs"], ["etl/data/processed/sprint_12e3_electric_unit_audit/electric_energy_sanity_dataset.csv"])
        self.assertEqual(by_id["12F-08"]["inputs"], ["etl/data/staging/sprint_12e2_epa_fuelcons/eco_drive_canonical_epa_fuelcons.db"])
        self.assertEqual(by_id["12F-10"]["inputs"], ["etl/data/staging/sprint_12e2_epa_fuelcons/eco_drive_canonical_epa_fuelcons.db"])
        self.assertIn("notebooks/_data/12f10_fuelcons_run_adoption.csv", by_id["12F-10"]["outputs"])
        self.assertIn("notebooks/_data/12f10_fuelcons_run_adoption.csv", by_id["12F-11"]["inputs"])

    def test_all_notebooks_execute_in_sequence(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        original = Path.cwd()
        try:
            os.chdir(ROOT / "notebooks")
            for entry in self.manifest["notebooks"]:
                payload = json.loads((ROOT / entry["path"]).read_text(encoding="utf-8"))
                namespace: dict[str, object] = {}
                with contextlib.redirect_stdout(io.StringIO()):
                    for cell in payload["cells"]:
                        if cell["cell_type"] == "code":
                            exec("".join(cell["source"]), namespace)
        finally:
            os.chdir(original)

    def test_canonical_exports_and_disposable_database_are_valid(self) -> None:
        data_dir = ROOT / "notebooks" / "_data"
        tables = ("program", "vehicle_configuration", "vde", "run", "fuelcons", "fuelcons_run_adoption")
        for table in tables:
            self.assertTrue((data_dir / f"12f10_{table}.csv").is_file())
        with (data_dir / "12f10_fuelcons_run_adoption.csv").open(encoding="utf-8") as handle:
            adoption_csv_rows = sum(1 for _ in handle) - 1
        self.assertGreater(adoption_csv_rows, 0)
        database = data_dir / "12f11_canonical_notebook_demo.db"
        self.assertTrue(database.is_file())
        with sqlite3.connect(database) as connection:
            adoption_db_rows = connection.execute("SELECT COUNT(*) FROM fuelcons_run_adoption").fetchone()[0]
            self.assertEqual(adoption_db_rows, adoption_csv_rows)
            missing_links = connection.execute("""
                SELECT COUNT(*)
                FROM fuelcons_run_adoption a
                LEFT JOIN fuelcons f ON f.id=a.fuelcons_id
                LEFT JOIN run r ON r.run_id=a.run_id
                WHERE f.id IS NULL OR r.run_id IS NULL
            """).fetchone()[0]
            wrong_vde = connection.execute("""
                SELECT COUNT(*)
                FROM fuelcons_run_adoption a
                JOIN fuelcons f ON f.id=a.fuelcons_id
                JOIN run r ON r.run_id=a.run_id
                WHERE a.vde_id<>f.vde_id OR a.vde_id<>r.vde_id
            """).fetchone()[0]
            self.assertEqual(missing_links, 0)
            self.assertEqual(wrong_vde, 0)
            self.assertEqual(connection.execute("PRAGMA quick_check").fetchone()[0], "ok")
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_runtime_databases_remain_byte_identical(self) -> None:
        self.assertEqual({path: sha256(path) for path in RUNTIME_HASHES}, RUNTIME_HASHES)


if __name__ == "__main__":
    unittest.main()
