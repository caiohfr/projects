from __future__ import annotations

import gc
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.vde_core import db as db_module
from src.vde_core.data_change_log_repository import fetch_change_log
from src.vde_core.database_management_contract import LOCAL_ADMIN_ACTOR
from src.vde_core.vde_net_total_contract import VdeSemanticStatus
from src.vde_core.vde_net_total_normalization import (
    apply_vde_net_total_normalization,
    preview_vde_net_total_normalization,
)


class VdeNetTotalNormalizationTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "legacy.db"
        self._create_legacy_database()
        with db_module.using_db_path(self.db_path):
            db_module.ensure_db()
        self._seed_rows()

    def tearDown(self):
        gc.collect()
        self._temp_dir.cleanup()

    def test_preview_classifies_every_row_without_mutating(self):
        with db_module.using_db_path(self.db_path):
            preview = preview_vde_net_total_normalization()

        self.assertEqual(preview.total_rows_inspected, 5)
        self.assertEqual(
            preview.counts_by_status[VdeSemanticStatus.LEGACY_TOTAL_IN_NET_FIELD.value], 1
        )
        self.assertEqual(
            preview.counts_by_status[VdeSemanticStatus.AMBIGUOUS_REVIEW.value], 1
        )
        self.assertEqual(
            preview.counts_by_status[VdeSemanticStatus.CANONICAL_TOTAL_ONLY.value], 1
        )
        self.assertEqual(
            preview.counts_by_status[VdeSemanticStatus.CANONICAL_TOTAL_AND_NET.value], 1
        )
        self.assertEqual(preview.counts_by_status[VdeSemanticStatus.INVALID.value], 1)

        with sqlite3.connect(self.db_path) as con:
            row = con.execute(
                "SELECT vde_total_mj_per_km, vde_net_mj_per_km FROM vde_db WHERE id=1"
            ).fetchone()
        self.assertEqual(row, (None, 0.49))  # preview must not mutate

    def test_apply_moves_legacy_net_into_total_and_logs_change(self):
        with db_module.using_db_path(self.db_path):
            preview = preview_vde_net_total_normalization()
            result = apply_vde_net_total_normalization(
                preview, LOCAL_ADMIN_ACTOR, reason="test_normalization"
            )

        self.assertEqual(result.rows_normalized, 1)
        self.assertEqual(result.rows_flagged_for_review, 1)
        self.assertEqual(len(result.change_log_ids), 1)

        with sqlite3.connect(self.db_path) as con:
            legacy_row = con.execute(
                "SELECT vde_total_mj_per_km, vde_net_mj_per_km FROM vde_db WHERE id=1"
            ).fetchone()
            ambiguous_row = con.execute(
                "SELECT vde_total_mj_per_km, vde_net_mj_per_km, review_status FROM vde_db WHERE id=2"
            ).fetchone()
            untouched_total_and_net = con.execute(
                "SELECT vde_total_mj_per_km, vde_net_mj_per_km FROM vde_db WHERE id=4"
            ).fetchone()

        self.assertEqual(legacy_row, (0.49, None))
        self.assertEqual(ambiguous_row, (None, 0.51, "REVIEW_REQUIRED"))
        self.assertEqual(untouched_total_and_net, (0.55, 0.50))

        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            log_row = con.execute(
                "SELECT operation_id FROM data_change_log WHERE id=?",
                (result.change_log_ids[0],),
            ).fetchone()
        with db_module.using_db_path(self.db_path):
            receipt = fetch_change_log(log_row["operation_id"])
        self.assertEqual(receipt["entity_type"], "VDE")
        self.assertEqual(receipt["record_id"], "1")
        self.assertEqual(receipt["before_json"], {"vde_net_mj_per_km": 0.49, "vde_total_mj_per_km": None})
        self.assertEqual(receipt["after_json"], {"vde_net_mj_per_km": None, "vde_total_mj_per_km": 0.49})

    def test_normalization_is_idempotent(self):
        with db_module.using_db_path(self.db_path):
            first_preview = preview_vde_net_total_normalization()
            apply_vde_net_total_normalization(
                first_preview, LOCAL_ADMIN_ACTOR, reason="test_normalization"
            )
            second_preview = preview_vde_net_total_normalization()
            second_result = apply_vde_net_total_normalization(
                second_preview, LOCAL_ADMIN_ACTOR, reason="test_normalization_rerun"
            )

        self.assertEqual(len(second_preview.legacy_total_in_net_changes), 0)
        self.assertEqual(second_result.rows_normalized, 0)
        self.assertEqual(
            second_preview.counts_by_status[VdeSemanticStatus.CANONICAL_TOTAL_ONLY.value], 2
        )

    def _create_legacy_database(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.executescript(
                """
                PRAGMA foreign_keys=ON;
                CREATE TABLE vde_db (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT,
                    legislation TEXT NOT NULL,
                    category TEXT NOT NULL,
                    make TEXT NOT NULL,
                    model TEXT NOT NULL,
                    year INTEGER,
                    mass_kg REAL NOT NULL,
                    coast_A_N REAL,
                    vde_net_mj_per_km REAL,
                    vde_total_mj_per_km REAL
                );
                CREATE TABLE fuelcons_db (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    vde_id INTEGER NOT NULL REFERENCES vde_db(id) ON DELETE CASCADE,
                    electrification TEXT NOT NULL
                );
                CREATE TABLE tire_roadload_db (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT,
                    tire_test_code TEXT NOT NULL UNIQUE,
                    manufacturer TEXT NOT NULL,
                    model TEXT NOT NULL,
                    size_code TEXT,
                    is_active INTEGER DEFAULT 1,
                    standard_family TEXT NOT NULL,
                    rr_n_per_kn REAL NOT NULL
                );
                INSERT INTO vde_db (
                    id, legislation, category, make, model, year, mass_kg, coast_A_N
                ) VALUES (1, 'EPA', 'QA', 'QA', 'LEGACY', 2026, 1500, 118);
                INSERT INTO fuelcons_db (id, vde_id, electrification) VALUES (1, 1, 'ICE');
                INSERT INTO tire_roadload_db (
                    id, tire_test_code, manufacturer, model, size_code, is_active, standard_family, rr_n_per_kn
                ) VALUES (1, 'LEGACY-TIRE', 'QA', 'TIRE', '235/55R19', 1, 'ISO', 8);
                """
            )

    def _seed_rows(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            # id=1: legacy TOTAL stored as NET (created above, defaults to record_origin LEGACY via ensure_db backfill)
            con.execute(
                "UPDATE vde_db SET vde_net_mj_per_km=0.49, vde_total_mj_per_km=NULL, record_origin='LEGACY' WHERE id=1"
            )
            # id=2: ambiguous net-only row from a non-legacy origin
            con.execute(
                """
                INSERT INTO vde_db (
                    legislation, category, make, model, year, mass_kg, coast_A_N,
                    vde_net_mj_per_km, vde_total_mj_per_km, record_origin
                ) VALUES ('EPA','QA','QA','MANUAL',2026,1500,118, 0.51, NULL, 'MANUAL')
                """
            )
            # id=3: TOTAL only, canonical
            con.execute(
                """
                INSERT INTO vde_db (
                    legislation, category, make, model, year, mass_kg, coast_A_N,
                    vde_net_mj_per_km, vde_total_mj_per_km, record_origin
                ) VALUES ('EPA','QA','QA','TOTALONLY',2026,1500,118, NULL, 0.55, 'VDE_SETUP')
                """
            )
            # id=4: TOTAL + NET, canonical
            con.execute(
                """
                INSERT INTO vde_db (
                    legislation, category, make, model, year, mass_kg, coast_A_N,
                    vde_net_mj_per_km, vde_total_mj_per_km, record_origin
                ) VALUES ('EPA','QA','QA','BOTH',2026,1500,118, 0.50, 0.55, 'VDE_SETUP')
                """
            )
            # id=5: neither TOTAL nor NET, invalid
            con.execute(
                """
                INSERT INTO vde_db (
                    legislation, category, make, model, year, mass_kg, coast_A_N,
                    vde_net_mj_per_km, vde_total_mj_per_km, record_origin
                ) VALUES ('EPA','QA','QA','EMPTY',2026,1500,118, NULL, NULL, 'LEGACY')
                """
            )
            con.commit()


if __name__ == "__main__":
    unittest.main()
