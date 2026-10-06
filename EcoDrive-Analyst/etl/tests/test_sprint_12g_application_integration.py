from __future__ import annotations

import gc
import hashlib
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.vde_app.components import pwt_system_scenario
from src.vde_app.comparison_report_viewmodels import (
    BrowseAvailabilityFilters,
    apply_availability_filters,
    compute_browse_summary_counters,
    search_browse_candidates,
)
from src.vde_app.powertrain_system_scenario_viewmodels import current_draft
from src.vde_core import db as db_module
from src.vde_core.comparison_report_service import (
    build_comparison_dataset,
    list_comparison_scenarios_detailed,
)
from src.vde_core.fuel_estimation import FuelEstimateRequest, save_fuel_estimate_result
from src.vde_core.quick_scenario import (
    QuickScenario,
    VehicleQuickOverrides,
    resolve_quick_vehicle_scenario,
)
from src.vde_core.repositories import (
    delete_vde_by_id,
    fetch_fuelcons_by_vde_id,
    fetch_vde_by_id,
)
from src.vde_core.repositories import fuelcons_repository
from src.vde_core.system_scenario import ArchitectureClass
from src.vde_core.vde_request_save import execute_vde_request_save_plan
from src.vde_core.vde_setup_service import fetch_vde_edit_rows
from src.vde_core.vde_workflow_service import save_vde_setup_result


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_SOURCE = (
    ROOT
    / "etl"
    / "data"
    / "staging"
    / "sprint_12f13_vde_materialized"
    / "eco_drive_canonical_vde_materialized_candidate.db"
)
RUNTIME_DBS = (
    ROOT / "data" / "db" / "eco_drive.db",
    ROOT / "data" / "db" / "eco_drive_qa.db",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


class Sprint12GCanonicalApplicationIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_db_path = db_module.current_db_path()
        cls.source_hash_before = _sha256(CANONICAL_SOURCE)
        cls.runtime_hashes_before = {str(path): _sha256(path) for path in RUNTIME_DBS}
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.integration_db = Path(cls.temp_dir.name) / "canonical_integration.db"
        shutil.copy2(CANONICAL_SOURCE, cls.integration_db)
        db_module.configure_db_path(cls.integration_db)
        db_module.ensure_db()

        with sqlite3.connect(cls.integration_db) as connection:
            connection.row_factory = sqlite3.Row
            cls.epa = dict(
                connection.execute(
                    "SELECT v.id AS vde_id,f.id AS fuelcons_id,v.vehicle_configuration_id,v.make "
                    "FROM vde v JOIN fuelcons f ON f.vde_id=v.id "
                    "WHERE v.legislation='EPA' AND v.vde_total_mj_per_km IS NOT NULL "
                    "ORDER BY v.id LIMIT 1"
                ).fetchone()
            )
            cls.wltp = dict(
                connection.execute(
                    "SELECT v.id AS vde_id,f.id AS fuelcons_id,v.vehicle_configuration_id "
                    "FROM vde v LEFT JOIN fuelcons f ON f.vde_id=v.id "
                    "WHERE v.legislation='WLTP' AND v.vde_low_mj_per_km IS NOT NULL "
                    "ORDER BY v.id LIMIT 1"
                ).fetchone()
            )
            cls.multi_vde = dict(
                connection.execute(
                    "SELECT vehicle_configuration_id,MIN(id) AS first_vde_id,MAX(id) AS second_vde_id,COUNT(*) AS n "
                    "FROM vde GROUP BY vehicle_configuration_id HAVING COUNT(*)>1 "
                    "ORDER BY vehicle_configuration_id LIMIT 1"
                ).fetchone()
            )
            cls.multi_fuelcons = dict(
                connection.execute(
                    "SELECT vde_id,MIN(id) AS first_fuelcons_id,MAX(id) AS second_fuelcons_id,COUNT(*) AS n "
                    "FROM fuelcons GROUP BY vde_id HAVING COUNT(*)>1 ORDER BY vde_id LIMIT 1"
                ).fetchone()
            )
            cls.no_fuelcons = dict(
                connection.execute(
                    "SELECT v.id AS vde_id FROM vde v LEFT JOIN fuelcons f ON f.vde_id=v.id "
                    "WHERE f.id IS NULL ORDER BY v.id LIMIT 1"
                ).fetchone()
            )

    @classmethod
    def tearDownClass(cls) -> None:
        db_module.configure_db_path(cls.original_db_path)
        gc.collect()
        cls.temp_dir.cleanup()

    def _clone_payload(self, parent_id: int, marker: str) -> dict:
        parent = fetch_vde_by_id(parent_id)
        fields = (
            "legislation",
            "category",
            "make",
            "model",
            "year",
            "mass_kg",
            "test_mass_kg",
            "test_mass_low_kg",
            "test_mass_high_kg",
            "test_mass_basis",
            "inertia_class",
            "cycle_name",
            "cycle_source",
            "coast_A_N",
            "coast_B_N_per_kph",
            "coast_C_N_per_kph2",
            "vde_total_mj_per_km",
            "vde_net_mj_per_km",
            "record_origin",
        )
        payload = {field: parent.get(field) for field in fields if parent.get(field) is not None}
        payload["vde_id_parent"] = int(parent_id)
        payload["notes"] = marker
        return payload

    def _checks(self) -> tuple[list[tuple], str]:
        with sqlite3.connect(self.integration_db) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            fk = connection.execute("PRAGMA foreign_key_check").fetchall()
            quick = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        return fk, quick

    def test_01_canonical_db_path_can_be_selected_without_changing_production_default(self) -> None:
        self.assertEqual(db_module.current_db_path(), self.integration_db)
        self.assertEqual(db_module.DEFAULT_DB_PATH, Path("data/db/eco_drive.db"))
        self.assertEqual(db_module.VDE_WRITE_TABLE, "vde")
        self.assertEqual(db_module.FUELCONS_WRITE_TABLE, "fuelcons")
        self.assertFalse(db_module.LEGACY_FIXTURE_MODE)

    def test_02_browse_repository_reads_canonical_compatibility_surface(self) -> None:
        rows = list_comparison_scenarios_detailed({"make": self.epa["make"]})
        self.assertTrue(rows)
        self.assertTrue(all("fuelcons_id" in row and "vde_id" in row for row in rows))
        identities = {(row["fuelcons_id"], row["vde_id"]) for row in rows}
        self.assertEqual(len(identities), len(rows))
        search_rows = search_browse_candidates(rows, str(self.epa["fuelcons_id"]))
        self.assertTrue(search_rows)
        self.assertTrue(all(row["fuelcons_id"] == self.epa["fuelcons_id"] for row in search_rows))
        fuel_rows = apply_availability_filters(
            rows, BrowseAvailabilityFilters(has_fuel_economy=True)
        )
        counters = compute_browse_summary_counters(fuel_rows)
        self.assertEqual(counters.matching, len(fuel_rows))
        self.assertLessEqual(counters.with_net, counters.matching)

    def test_03_fetch_by_id_works_for_canonical_vde(self) -> None:
        row = fetch_vde_by_id(self.epa["vde_id"])
        self.assertEqual(row["id"], self.epa["vde_id"])
        self.assertEqual(row["legislation"], "EPA")
        self.assertIsNotNone(row["coast_A_N"])

    def test_04_fuelcons_by_vde_reads_canonical_materialized_results(self) -> None:
        rows = fetch_fuelcons_by_vde_id(self.epa["vde_id"])
        selected = next(row for row in rows if row["id"] == self.epa["fuelcons_id"])
        self.assertIn("fuel_l_per_100km", selected)
        self.assertIn("energy_Wh_per_km", selected)

    def test_05_representative_epa_read_works(self) -> None:
        row = fetch_vde_by_id(self.epa["vde_id"])
        self.assertEqual(row["legislation"], "EPA")
        self.assertIsNotNone(row["test_mass_kg"])
        self.assertIsNotNone(row["vde_total_mj_per_km"])

    def test_06_representative_wltp_read_works(self) -> None:
        row = fetch_vde_by_id(self.wltp["vde_id"])
        self.assertEqual(row["legislation"], "WLTP")
        self.assertTrue(any(row.get(field) is not None for field in ("vde_low_mj_per_km", "vde_mid_mj_per_km", "vde_high_mj_per_km")))

    def test_07_multi_vde_same_configuration_case_works(self) -> None:
        first = fetch_vde_by_id(self.multi_vde["first_vde_id"])
        second = fetch_vde_by_id(self.multi_vde["second_vde_id"])
        self.assertGreater(self.multi_vde["n"], 1)
        self.assertNotEqual(first["id"], second["id"])
        self.assertNotEqual(
            (first["coast_A_N"], first["coast_B_N_per_kph"], first["coast_C_N_per_kph2"]),
            (second["coast_A_N"], second["coast_B_N_per_kph"], second["coast_C_N_per_kph2"]),
        )

    def test_08_multi_fuelcons_case_works(self) -> None:
        rows = fetch_fuelcons_by_vde_id(self.multi_fuelcons["vde_id"])
        self.assertGreaterEqual(len(rows), self.multi_fuelcons["n"])
        self.assertIn(self.multi_fuelcons["first_fuelcons_id"], {row["id"] for row in rows})
        self.assertIn(self.multi_fuelcons["second_fuelcons_id"], {row["id"] for row in rows})

    def test_09_no_fuelcons_case_is_handled(self) -> None:
        self.assertTrue(fetch_vde_by_id(self.no_fuelcons["vde_id"]))
        self.assertEqual(fetch_fuelcons_by_vde_id(self.no_fuelcons["vde_id"]), [])

    def test_10_vde_setup_read_works(self) -> None:
        rows = fetch_vde_edit_rows(limit=25)
        self.assertEqual(len(rows), 25)
        self.assertTrue(all("mass_kg" in row and "coast_A_N" in row for row in rows))

    def test_11_vde_setup_create_update_writes_canonical_tables_in_disposable_db(self) -> None:
        payload = self._clone_payload(self.epa["vde_id"], "12G disposable workflow insert")
        preview = {"ok": True, "save_payload": {"insert_row": payload}, "phase_update_row": {}}
        result = save_vde_setup_result(preview, "insert_new")
        inserted_id = result["vde_id"]
        try:
            with sqlite3.connect(self.integration_db) as connection:
                physical = connection.execute(
                    "SELECT vehicle_configuration_id,vde_id_parent,notes FROM vde WHERE id=?",
                    (inserted_id,),
                ).fetchone()
            self.assertIsNotNone(physical)
            self.assertEqual(physical[0], self.epa["vehicle_configuration_id"])
            self.assertEqual(physical[1], self.epa["vde_id"])
            update_preview = {
                "ok": True,
                "save_payload": {
                    "target_vde_id": inserted_id,
                    "update_row": {"notes": "12G disposable workflow update"},
                },
                "phase_update_row": {},
            }
            save_vde_setup_result(update_preview, "update_existing")
            self.assertEqual(fetch_vde_by_id(inserted_id)["notes"], "12G disposable workflow update")
        finally:
            delete_vde_by_id(inserted_id)

    def test_12_write_read_back_parity_passes(self) -> None:
        payload = self._clone_payload(self.epa["vde_id"], "12G compact transaction insert")
        plan = {
            "operation_id": "sprint12g-disposable",
            "status": "ready",
            "can_execute": True,
            "proposals_to_save": [
                {
                    "proposal_id": "12G-P1",
                    "row_payload": payload,
                    "component_plan": [],
                    "final_name": "12G disposable",
                }
            ],
            "skipped_proposals": [],
            "baseline_update_requests": [],
        }
        saved = execute_vde_request_save_plan(plan)
        self.assertEqual(saved["status"], "success", saved)
        inserted_id = saved["saved_proposals"][0]["vde_row_id"]
        try:
            vde_row = fetch_vde_by_id(inserted_id)
            self.assertEqual(vde_row["vde_id_parent"], self.epa["vde_id"])
            self.assertEqual(vde_row["coast_A_N"], payload["coast_A_N"])
            fuel_result = FuelEstimateRequest(
                vde_id=inserted_id,
                energy_basis="MANUAL_VALUE",
                method="manual_imported",
                vehicle_features={
                    "electrification": "ICE",
                    "source_vde_revision": vde_row.get("updated_at") or vde_row.get("created_at"),
                },
                powertrain_features={"fuel_type": "Gasoline", "eta_pt_est": 0.30},
                manual_inputs={"source": "12G disposable", "fuel_l_100km": 7.25},
            )
            fuel_saved = save_fuel_estimate_result(fuel_result, "insert_new")
            fuel_rows = fetch_fuelcons_by_vde_id(inserted_id)
            self.assertEqual(len(fuel_rows), 1)
            self.assertEqual(fuel_rows[0]["id"], fuel_saved["row_id"])
            self.assertAlmostEqual(fuel_rows[0]["fuel_l_per_100km"], 7.25)
        finally:
            delete_vde_by_id(inserted_id)

    def test_13_quick_scenario_works_from_canonical_baseline(self) -> None:
        source = fetch_vde_by_id(self.epa["vde_id"])
        scenario = QuickScenario(
            source_identity=f"vde:{self.epa['vde_id']}",
            slot=1,
            vehicle_overrides=VehicleQuickOverrides(),
        )
        result = resolve_quick_vehicle_scenario(scenario, source_vde_row=source)
        self.assertTrue(result.is_ready, result.issues)
        self.assertEqual(result.quick_scenario_identity, scenario.identity)
        self.assertEqual(result.resolved_vde_row["id"], self.epa["vde_id"])

    def test_14_comparison_works_from_canonical_records(self) -> None:
        dataset = build_comparison_dataset(
            {"kind": "FUELCONS_SCENARIO", "fuelcons_id": self.epa["fuelcons_id"]},
            [{"kind": "VDE_ONLY", "vde_id": self.wltp["vde_id"]}],
        )
        self.assertEqual(dataset.reference.fuelcons_id, self.epa["fuelcons_id"])
        self.assertEqual(dataset.comparisons[0].vde_id, self.wltp["vde_id"])

    def test_15_powertrain_scenario_baseline_resolves(self) -> None:
        drafts = (current_draft(self.epa["vde_id"], ArchitectureClass.ICE, fuelcons_id=self.epa["fuelcons_id"]),)
        sources, labels = pwt_system_scenario._load_sources(
            self.epa["vde_id"],
            self.epa["fuelcons_id"],
            drafts=drafts,
        )
        self.assertIn(self.epa["vde_id"], sources)
        self.assertEqual(sources[self.epa["vde_id"]].fuelcons_row["id"], self.epa["fuelcons_id"])
        self.assertIn(self.epa["vde_id"], labels)

    def test_16_normal_fuelcons_read_does_not_traverse_run(self) -> None:
        observed_sql: list[str] = []
        original = fuelcons_repository.fetchall

        def recording_fetchall(sql, params=()):
            observed_sql.append(str(sql))
            return original(sql, params)

        with patch.object(fuelcons_repository, "fetchall", side_effect=recording_fetchall):
            rows = fuelcons_repository.fetch_fuelcons_by_vde_id(self.epa["vde_id"])
        self.assertTrue(rows)
        self.assertTrue(observed_sql)
        self.assertTrue(all(" run " not in f" {sql.lower()} " for sql in observed_sql))
        self.assertTrue(all("fuelcons_run_adoption" not in sql.lower() for sql in observed_sql))

    def test_17_fk_and_quick_check_pass_after_writes(self) -> None:
        fk, quick = self._checks()
        self.assertEqual(fk, [])
        self.assertEqual(quick, "ok")

    def test_18_runtime_and_source_candidate_hashes_are_unchanged(self) -> None:
        self.assertEqual(_sha256(CANONICAL_SOURCE), self.source_hash_before)
        self.assertEqual(
            {str(path): _sha256(path) for path in RUNTIME_DBS},
            self.runtime_hashes_before,
        )


if __name__ == "__main__":
    unittest.main()
