from __future__ import annotations

import csv
from contextlib import closing
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest

from etl.scripts.sprint_12_final_id_normalization import normalize_candidate_database


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Sprint12FinalIdNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="id_normalization_")
        root = Path(self.temporary.name)
        self.source = root / "source.db"
        self.output = root / "candidate.db"
        self.artifacts = root / "artifacts"
        with closing(sqlite3.connect(self.source)) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.executescript(
                """
                CREATE TABLE vde (
                    id INTEGER PRIMARY KEY,
                    source_name TEXT,
                    source_file_version TEXT,
                    source_record_id TEXT,
                    normalization_version TEXT,
                    vehicle_configuration_id TEXT,
                    make TEXT,
                    model TEXT,
                    year INTEGER,
                    legislation TEXT,
                    category TEXT,
                    cycle_name TEXT,
                    coast_A_N REAL,
                    coast_B_N_per_kph REAL,
                    coast_C_N_per_kph2 REAL,
                    test_mass_kg REAL,
                    vde_id_parent INTEGER REFERENCES vde(id) ON UPDATE CASCADE,
                    notes TEXT,
                    provenance_json TEXT
                );
                CREATE TABLE run (
                    run_id TEXT PRIMARY KEY,
                    vde_id INTEGER NOT NULL REFERENCES vde(id) ON UPDATE CASCADE,
                    source_record_id TEXT,
                    result_value REAL,
                    provenance_json TEXT,
                    UNIQUE(run_id,vde_id)
                );
                CREATE TABLE fuelcons (
                    id INTEGER PRIMARY KEY,
                    vde_id INTEGER NOT NULL REFERENCES vde(id) ON UPDATE CASCADE,
                    source_name TEXT,
                    source_file_version TEXT,
                    source_record_id TEXT,
                    normalization_version TEXT,
                    record_origin TEXT,
                    comparison_basis TEXT,
                    electrification TEXT,
                    fuel_type TEXT,
                    fuel_l_per_100km REAL,
                    reference_fuelcons_id INTEGER REFERENCES fuelcons(id) ON UPDATE CASCADE,
                    provenance_json TEXT,
                    UNIQUE(id,vde_id)
                );
                CREATE TABLE fuelcons_run_adoption (
                    fuelcons_id INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    vde_id INTEGER NOT NULL REFERENCES vde(id) ON UPDATE CASCADE,
                    result_dimension TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    PRIMARY KEY(fuelcons_id,run_id,result_dimension),
                    FOREIGN KEY(fuelcons_id,vde_id) REFERENCES fuelcons(id,vde_id) ON UPDATE CASCADE,
                    FOREIGN KEY(run_id,vde_id) REFERENCES run(run_id,vde_id) ON UPDATE CASCADE
                );
                CREATE TABLE component_resolution (component_resolution_id TEXT PRIMARY KEY);
                CREATE TABLE vde_component_resolution (
                    vde_id INTEGER NOT NULL REFERENCES vde(id) ON UPDATE CASCADE,
                    component_resolution_id TEXT NOT NULL REFERENCES component_resolution(component_resolution_id),
                    boundary TEXT NOT NULL,
                    PRIMARY KEY(vde_id,component_resolution_id,boundary)
                );

                INSERT INTO vde VALUES
                  (-1002,'EPA','HASH','EPA-2','v1','VC-2','BMW','330i',2022,'EPA','Car','EPA NORMAL',11,0.2,0.01,1600,-1001,
                   'EPA model-year carryover from VDE -1001.',
                   '{"carryover_from_vde_id":-1001,"lineage_relation":"EPA_MODEL_YEAR_CARRYOVER"}'),
                  (5038,'LEGACY','HASH','LEGACY-1','v1','VC-L','AUDI','QA',2020,'EPA','Car','EPA NORMAL',13,0.3,0.02,1700,NULL,
                   NULL,'{"population":"LEGACY"}'),
                  (-1001,'EPA','HASH','EPA-1','v1','VC-1','BMW','330i',2021,'EPA','Car','EPA NORMAL',11,0.2,0.01,1600,NULL,
                   NULL,'{"lineage_relation":null}');
                INSERT INTO run VALUES
                  ('RUN-1',-1001,'RUN-SOURCE-1',25.0,'{}'),
                  ('RUN-2',-1002,'RUN-SOURCE-2',25.0,'{"carryover_from_run_id":"RUN-1"}'),
                  ('RUN-L',5038,'RUN-SOURCE-L',30.0,'{}');
                INSERT INTO fuelcons VALUES
                  (5018,-1001,'MODEL','HASH','FC-LEGACY','v1','ML_PREDICTION','MODEL','ICE','Gasoline',7.1,NULL,'{}'),
                  (-4000001,-1002,'EPA','HASH','FC-EPA-2','v1','EPA_RECONSTRUCTED','EPA_LABEL_2_CYCLE','ICE','Gasoline',7.1,5018,
                   '{"carryover_from_fuelcons_id":5018,"lineage_relation":"EPA_MODEL_YEAR_CARRYOVER"}');
                INSERT INTO fuelcons_run_adoption VALUES
                  (5018,'RUN-1',-1001,'ALL',0),
                  (-4000001,'RUN-2',-1002,'ALL',0);
                INSERT INTO component_resolution VALUES ('RES-1');
                INSERT INTO vde_component_resolution VALUES (-1002,'RES-1','TOTAL');
                """
            )
        self.source_hash = sha256(self.source)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run(self):
        return normalize_candidate_database(
            self.source, self.output, self.artifacts, rebuild=True
        )

    def test_contiguous_positive_ids_and_all_foreign_keys_are_remapped(self):
        report = self._run()
        with closing(sqlite3.connect(self.output)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*),MIN(id),MAX(id) FROM vde").fetchone(), (3, 1, 3))
            self.assertEqual(connection.execute("SELECT COUNT(*),MIN(id),MAX(id) FROM fuelcons").fetchone(), (2, 1, 2))
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            self.assertEqual(connection.execute("PRAGMA quick_check").fetchone()[0], "ok")
            for table, column in (
                ("vde", "id"), ("vde", "vde_id_parent"), ("run", "vde_id"),
                ("fuelcons", "id"), ("fuelcons", "vde_id"),
                ("fuelcons", "reference_fuelcons_id"),
                ("fuelcons_run_adoption", "fuelcons_id"),
                ("fuelcons_run_adoption", "vde_id"),
                ("vde_component_resolution", "vde_id"),
            ):
                count = connection.execute(
                    f'SELECT COUNT(*) FROM "{table}" WHERE "{column}" < 0'
                ).fetchone()[0]
                self.assertEqual(count, 0, f"{table}.{column}")
        self.assertEqual(report.foreign_key_issues, 0)
        self.assertTrue(all(value == 0 for value in report.orphan_counts.values()))

    def test_source_hash_schema_counts_engineering_and_topology_are_preserved(self):
        report = self._run()
        self.assertEqual(sha256(self.source), self.source_hash)
        self.assertEqual(report.source_sha256_before, report.source_sha256_after)
        self.assertEqual(report.schema_sha256_before, report.schema_sha256_after)
        self.assertEqual(report.table_counts_before, report.table_counts_after)
        self.assertEqual(report.vde_topology_sha256_before, report.vde_topology_sha256_after)
        self.assertEqual(report.run_topology_sha256_before, report.run_topology_sha256_after)
        self.assertEqual(report.fuelcons_topology_sha256_before, report.fuelcons_topology_sha256_after)
        self.assertEqual(report.adoption_topology_sha256_before, report.adoption_topology_sha256_after)
        self.assertEqual(report.engineering_payload_sha256_before, report.engineering_payload_sha256_after)

    def test_mapping_is_deterministic_and_auditable(self):
        first = self._run()
        first_output_hash = first.output_sha256
        first_vde_csv = (self.artifacts / "VDE_ID_REMAP.csv").read_text(encoding="utf-8")
        first_fuel_csv = (self.artifacts / "FUELCONS_ID_REMAP.csv").read_text(encoding="utf-8")
        second = self._run()
        self.assertEqual(second.output_sha256, first_output_hash)
        self.assertEqual((self.artifacts / "VDE_ID_REMAP.csv").read_text(encoding="utf-8"), first_vde_csv)
        self.assertEqual((self.artifacts / "FUELCONS_ID_REMAP.csv").read_text(encoding="utf-8"), first_fuel_csv)
        with (self.artifacts / "VDE_ID_REMAP.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual({int(row["old_id"]) for row in rows}, {-1002, -1001, 5038})
        self.assertEqual({int(row["new_id"]) for row in rows}, {1, 2, 3})

    def test_text_identities_and_human_lineage_note_are_preserved_semantically(self):
        self._run()
        with closing(sqlite3.connect(self.output)) as connection:
            run_ids = {row[0] for row in connection.execute("SELECT run_id FROM run")}
            self.assertEqual(run_ids, {"RUN-1", "RUN-2", "RUN-L"})
            child_id, parent_id, notes, provenance = connection.execute(
                "SELECT id,vde_id_parent,notes,provenance_json FROM vde WHERE source_record_id='EPA-2'"
            ).fetchone()
            self.assertIn(str(parent_id), notes)
            self.assertNotIn("-1001", notes)
            self.assertIn(f'"carryover_from_vde_id":{parent_id}', provenance)
            fuel_parent, fuel_provenance = connection.execute(
                "SELECT reference_fuelcons_id,provenance_json FROM fuelcons WHERE source_record_id='FC-EPA-2'"
            ).fetchone()
            self.assertIn(f'"carryover_from_fuelcons_id":{fuel_parent}', fuel_provenance)
            self.assertGreater(child_id, 0)

    def test_prod_output_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "pre-PROD"):
            normalize_candidate_database(
                self.source,
                Path(self.temporary.name) / "prod" / "eco_drive.db",
                self.artifacts,
                rebuild=True,
            )


if __name__ == "__main__":
    unittest.main()
