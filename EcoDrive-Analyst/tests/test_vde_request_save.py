from __future__ import annotations

from copy import deepcopy
import sqlite3
import tempfile
import unittest

from src.vde_core.vde_request_save import (
    SAVE_MODE_REQUIRE_ALL_VALID,
    SAVE_MODE_SELECTED,
    SAVE_MODE_VALID_ONLY,
    build_vde_request_save_plan,
    execute_vde_request_save_plan,
    generate_auto_proposal_name,
)
from src.vde_core.test_mass import inertia_step_for_mass
from src.vde_core.vde_mass_proposal_resolver import resolve_mass_proposal


def _proposal_result(
    proposal_id: str,
    *,
    display_index: int,
    status: str = "OK",
    total_mj_per_km: float | None = 1.25,
    net_mj_per_km: float | None = 1.10,
    walk_from: str = "Baseline",
    source_column: str | None = None,
    issues: list[dict] | None = None,
    component_actions: list[dict] | None = None,
    domain_results: dict | None = None,
    resolved_snapshot: dict | None = None,
    source_snapshot: dict | None = None,
    abc_total: dict | None = None,
) -> dict:
    return {
        "proposal_id": proposal_id,
        "display_index": display_index,
        "source_column": source_column or f"Requested #{display_index}",
        "walk_from": {"column_id": "baseline", "label": walk_from},
        "source_snapshot": deepcopy(
            source_snapshot
            or {
                "mass_kg": 1600.0,
                "test_mass_kg": 1736.0,
                "CdA": 0.62,
                "cycle_name": "FTP75",
            }
        ),
        "requested_snapshot": {},
        "resolved_snapshot": deepcopy(
            resolved_snapshot
            or {
                "legislation": "EPA",
                "category": "MIDSIZE",
                "make": "FORD",
                "model": "TEST",
                "year": 2026,
                "cycle_name": "FTP75",
                "mass_kg": 1600.0,
                "test_mass_kg": 1736.0,
                "CdA": 0.63,
                "front_tire_id": 7,
                "rear_tire_id": 7,
                "tire_A_final": 48.0,
                "tire_B_final": 0.009,
                "tire_C_final": 0.001,
                "tire_calc_source": "tire_service",
                "resolved_mass_setup": {
                    "mass_kg": 1600.0,
                    "test_mass_kg": 1736.0,
                    "test_mass_basis": "EPA",
                    "inertia_class": 60,
                    "resolved_mass_used_kg": 1736.0,
                    "mass_rule_status": "OK",
                    "mass_rule_notes": "Resolved",
                },
            }
        ),
        "domain_results": deepcopy(
            domain_results
            or {
                "mass": {"proposal_type": "CUSTOM_MASS", "status": status, "source": "Baseline"},
                "tire": {"proposal_type": "TIRE_DB_LOOKUP", "status": "OK", "source": "Baseline"},
            }
        ),
        "abc_total": deepcopy(abc_total or {"A": 121.0, "B": 0.021, "C": 0.0105}),
        "abc_net": {"A": 111.0, "B": 0.018, "C": 0.0095} if net_mj_per_km is not None else None,
        "vde_results": {
            "total": {"mj_per_km": total_mj_per_km} if total_mj_per_km is not None else None,
            "net": {"mj_per_km": net_mj_per_km} if net_mj_per_km is not None else None,
        },
        "status": status,
        "issues": deepcopy(issues or []),
        "component_actions": deepcopy(component_actions or []),
        "preview_summary": {"warnings": []},
    }


def _resolution_result(proposals: list[dict]) -> dict:
    return {
        "baseline": {
            "printed": {
                "selected_baseline_vde_id": 5038,
                "mass_kg": 1600.0,
                "A": 120.0,
                "B": 0.02,
                "C": 0.01,
                "cda_m2": 0.62,
            },
            "correction": {
                "mass_kg": 1650.0,
                "A": 121.0,
            },
            "effective": {
                "selected_baseline_vde_id": 5038,
                "legislation": "EPA",
                "category": "MIDSIZE",
                "make": "FORD",
                "model": "TEST",
                "year": 2026,
                "cycle_name": "FTP75",
                "mass_kg": 1650.0,
                "test_mass_kg": 1736.0,
                "A": 121.0,
                "B": 0.02,
                "C": 0.01,
                "cda_m2": 0.62,
            },
            "corrected_fields": ["A", "mass_kg"],
        },
        "resolved_columns": {
            "baseline": {
                "selected_baseline_vde_id": 5038,
                "legislation": "EPA",
                "category": "MIDSIZE",
                "make": "FORD",
                "model": "TEST",
                "year": 2026,
                "cycle_name": "FTP75",
                "mass_kg": 1650.0,
                "test_mass_kg": 1736.0,
                "initial_abc_total": {"A": 121.0, "B": 0.02, "C": 0.01},
            }
        },
        "proposal_results": deepcopy(proposals),
        "status": "Review",
        "issues": [],
    }


def _request_state() -> dict:
    return {
        "columns": {
            "proposal_req_1": {"direct": {"description": "Mass -45 kg + Tire Lookup"}},
            "proposal_req_2": {"direct": {"description": ""}},
            "proposal_req_3": {"direct": {"description": "Needs review"}},
        },
        "vde_request_source": {
            "filename": "EcoDrive_VDE_Request.xlsx",
            "schema_version": "0.1",
        },
        "vde_request_import": {
            "source": {
                "filename": "EcoDrive_VDE_Request.xlsx",
                "schema_version": "0.1",
            }
        },
    }


def _saved_test_mass_from_resolved_mass(source: dict, proposal_type: str, inputs: dict) -> tuple[dict, dict]:
    resolved = resolve_mass_proposal(source, proposal_type, inputs)
    if resolved["status"] != "OK":
        raise AssertionError(f"Fixture mass resolution failed: {resolved}")

    mass_setup = dict(resolved["resolved_snapshot"])
    snapshot = {
        "legislation": source["legislation"],
        "mass_kg": mass_setup["mass_kg"],
        "test_mass_kg": mass_setup["test_mass_kg"],
        "CdA": 0.62,
        "resolved_mass_setup": mass_setup,
    }
    plan = build_vde_request_save_plan(
        _resolution_result(
            [_proposal_result("proposal_req_1", display_index=1, resolved_snapshot=snapshot)]
        ),
        save_mode=SAVE_MODE_VALID_ONLY,
        request_state=_request_state(),
        current_fingerprint="fp1",
        resolution_fingerprint="fp1",
    )
    return mass_setup, plan["proposals_to_save"][0]["row_payload"]


class VdeRequestSavePlanTests(unittest.TestCase):
    def test_epa_curb_to_twc_persists_canonical_mass_not_physical_test_mass(self):
        source = {"legislation": "EPA", "mass_kg": 1500.0, "test_mass_kg": 1644.0, "inertia_class": 1644.0}
        mass_setup, row = _saved_test_mass_from_resolved_mass(source, "EPA_CURB_TO_TWC", {"mass_kg": 1500.0})

        self.assertEqual(row["test_mass_kg"], mass_setup["vde_calculation_mass_kg"])
        self.assertNotEqual(row["test_mass_kg"], mass_setup["test_mass_kg"])

    def test_epa_curb_change_inside_twc_persists_same_canonical_mass(self):
        source = {"legislation": "EPA", "mass_kg": 1500.0, "test_mass_kg": 1644.0, "inertia_class": 1644.0}
        source_twc = inertia_step_for_mass(source["mass_kg"])["inertia_class_kg"]
        mass_setup, row = _saved_test_mass_from_resolved_mass(source, "EPA_CURB_TO_TWC", {"mass_kg": 1501.0})

        self.assertEqual(mass_setup["vde_calculation_mass_kg"], source_twc)
        self.assertEqual(row["test_mass_kg"], source_twc)

    def test_epa_curb_change_crossing_twc_persists_new_canonical_mass(self):
        source = {"legislation": "EPA", "mass_kg": 1500.0, "test_mass_kg": 1644.0, "inertia_class": 1644.0}
        source_step = inertia_step_for_mass(source["mass_kg"])
        target_curb = float(source_step["upper_bound_inclusive"]) + 1.0
        mass_setup, row = _saved_test_mass_from_resolved_mass(source, "EPA_CURB_TO_TWC", {"mass_kg": target_curb})

        self.assertNotEqual(mass_setup["vde_calculation_mass_kg"], source_step["inertia_class_kg"])
        self.assertEqual(row["test_mass_kg"], mass_setup["vde_calculation_mass_kg"])

    def test_epa_twc_shift_persists_shifted_canonical_mass(self):
        source = {"legislation": "EPA", "mass_kg": 1500.0, "test_mass_kg": 1644.0, "inertia_class": 1644.0}
        mass_setup, row = _saved_test_mass_from_resolved_mass(
            source, "MASS_TWC_SHIFT", {"shift_steps": 1.0, "target_side": "Up"}
        )

        self.assertGreater(mass_setup["vde_calculation_mass_kg"], source["inertia_class"])
        self.assertEqual(row["test_mass_kg"], mass_setup["vde_calculation_mass_kg"])

    def test_wltp_save_plan_keeps_canonical_test_mass(self):
        source = {
            "legislation": "WLTP",
            "mass_kg": 1600.0,
            "test_mass_kg": 1780.0,
            "payload_kg": 180.0,
            "options_kg": 0.0,
            "wltp_category": "M1",
        }
        mass_setup, row = _saved_test_mass_from_resolved_mass(source, "WLTP_MASS_LINE", {"mass_kg": 1580.0})

        self.assertEqual(mass_setup["vde_calculation_mass_kg"], mass_setup["test_mass_kg"])
        self.assertEqual(row["test_mass_kg"], mass_setup["test_mass_kg"])

    def test_save_plan_allows_ok_proposal(self):
        resolution = _resolution_result([_proposal_result("proposal_req_1", display_index=1)])

        plan = build_vde_request_save_plan(
            resolution,
            save_mode=SAVE_MODE_VALID_ONLY,
            request_state=_request_state(),
            current_fingerprint="fp1",
            resolution_fingerprint="fp1",
        )

        self.assertTrue(plan["can_execute"])
        self.assertEqual(len(plan["proposals_to_save"]), 1)

    def test_review_without_confirmation_is_blocked(self):
        proposal = _proposal_result(
            "proposal_req_1",
            display_index=1,
            status="Review",
            issues=[{"severity": "review", "code": "manual_reference_override"}],
        )
        resolution = _resolution_result([proposal])

        plan = build_vde_request_save_plan(
            resolution,
            save_mode=SAVE_MODE_SELECTED,
            selected_proposal_ids=["proposal_req_1"],
            request_state=_request_state(),
            current_fingerprint="fp1",
            resolution_fingerprint="fp1",
        )

        self.assertFalse(plan["can_execute"])
        self.assertTrue(any(item["code"] == "proposal_not_eligible" for item in plan["blocking_issues"]))

    def test_review_with_confirmation_becomes_eligible(self):
        proposal = _proposal_result(
            "proposal_req_1",
            display_index=1,
            status="Review",
            issues=[{"severity": "review", "code": "manual_reference_override"}],
        )
        resolution = _resolution_result([proposal])

        plan = build_vde_request_save_plan(
            resolution,
            save_mode=SAVE_MODE_SELECTED,
            selected_proposal_ids=["proposal_req_1"],
            review_confirmations={"proposal_req_1": True},
            request_state=_request_state(),
            current_fingerprint="fp1",
            resolution_fingerprint="fp1",
        )

        self.assertTrue(plan["can_execute"])
        self.assertTrue(plan["proposals_to_save"][0]["review_confirmed"])

    def test_missing_invalid_and_blocked_are_not_eligible(self):
        proposals = [
            _proposal_result("proposal_req_1", display_index=1, status="Missing"),
            _proposal_result("proposal_req_2", display_index=2, status="Invalid"),
            _proposal_result("proposal_req_3", display_index=3, status="Blocked"),
        ]
        resolution = _resolution_result(proposals)

        plan = build_vde_request_save_plan(
            resolution,
            save_mode=SAVE_MODE_VALID_ONLY,
            request_state=_request_state(),
            current_fingerprint="fp1",
            resolution_fingerprint="fp1",
        )

        self.assertFalse(plan["can_execute"])
        self.assertEqual(plan["status"], "empty")

    def test_missing_total_vde_blocks_even_when_net_is_missing_only(self):
        proposal = _proposal_result("proposal_req_1", display_index=1, total_mj_per_km=None, net_mj_per_km=None)
        resolution = _resolution_result([proposal])

        plan = build_vde_request_save_plan(
            resolution,
            save_mode=SAVE_MODE_SELECTED,
            selected_proposal_ids=["proposal_req_1"],
            request_state=_request_state(),
            current_fingerprint="fp1",
            resolution_fingerprint="fp1",
        )

        self.assertFalse(plan["can_execute"])

    def test_missing_net_does_not_block_when_total_exists(self):
        proposal = _proposal_result("proposal_req_1", display_index=1, total_mj_per_km=1.25, net_mj_per_km=None)
        resolution = _resolution_result([proposal])

        plan = build_vde_request_save_plan(
            resolution,
            save_mode=SAVE_MODE_VALID_ONLY,
            request_state=_request_state(),
            current_fingerprint="fp1",
            resolution_fingerprint="fp1",
        )

        self.assertTrue(plan["can_execute"])

    def test_save_selected_only_uses_explicit_selection(self):
        proposals = [
            _proposal_result("proposal_req_1", display_index=1),
            _proposal_result("proposal_req_2", display_index=2),
        ]
        resolution = _resolution_result(proposals)

        plan = build_vde_request_save_plan(
            resolution,
            save_mode=SAVE_MODE_SELECTED,
            selected_proposal_ids=["proposal_req_2"],
            request_state=_request_state(),
            current_fingerprint="fp1",
            resolution_fingerprint="fp1",
        )

        self.assertEqual([item["proposal_id"] for item in plan["proposals_to_save"]], ["proposal_req_2"])

    def test_require_all_valid_blocks_package(self):
        proposals = [
            _proposal_result("proposal_req_1", display_index=1),
            _proposal_result("proposal_req_2", display_index=2, status="Missing"),
        ]
        resolution = _resolution_result(proposals)

        plan = build_vde_request_save_plan(
            resolution,
            save_mode=SAVE_MODE_REQUIRE_ALL_VALID,
            request_state=_request_state(),
            current_fingerprint="fp1",
            resolution_fingerprint="fp1",
        )

        self.assertFalse(plan["can_execute"])
        self.assertTrue(any(item["code"] == "require_all_valid_blocked" for item in plan["blocking_issues"]))

    def test_parent_stays_on_original_baseline(self):
        resolution = _resolution_result([_proposal_result("proposal_req_1", display_index=1)])

        plan = build_vde_request_save_plan(
            resolution,
            save_mode=SAVE_MODE_VALID_ONLY,
            request_state=_request_state(),
            current_fingerprint="fp1",
            resolution_fingerprint="fp1",
        )

        self.assertEqual(plan["proposals_to_save"][0]["row_payload"]["vde_id_parent"], 5038)

    def test_resolved_snapshot_is_used_for_row_payload(self):
        resolution = _resolution_result(
            [
                _proposal_result(
                    "proposal_req_1",
                    display_index=1,
                    abc_total={"A": 130.0, "B": 0.03, "C": 0.02},
                    resolved_snapshot={"mass_kg": 1700.0, "CdA": 0.66, "resolved_mass_setup": {"test_mass_kg": 1800.0, "test_mass_basis": "CUSTOM"}},
                )
            ]
        )

        plan = build_vde_request_save_plan(
            resolution,
            save_mode=SAVE_MODE_VALID_ONLY,
            request_state=_request_state(),
            current_fingerprint="fp1",
            resolution_fingerprint="fp1",
        )

        row = plan["proposals_to_save"][0]["row_payload"]
        self.assertEqual(row["coast_A_N"], 130.0)
        self.assertEqual(row["mass_kg"], 1700.0)
        self.assertEqual(row["test_mass_kg"], 1800.0)

    def test_baseline_updates_default_to_selected(self):
        resolution = _resolution_result([_proposal_result("proposal_req_1", display_index=1)])

        plan = build_vde_request_save_plan(
            resolution,
            save_mode=SAVE_MODE_VALID_ONLY,
            request_state=_request_state(),
            current_fingerprint="fp1",
            resolution_fingerprint="fp1",
        )

        selected = {item["field_key"] for item in plan["baseline_update_requests"]}
        self.assertEqual(selected, {"A", "mass_kg"})

    def test_baseline_updates_can_be_deselected(self):
        resolution = _resolution_result([_proposal_result("proposal_req_1", display_index=1)])

        plan = build_vde_request_save_plan(
            resolution,
            save_mode=SAVE_MODE_VALID_ONLY,
            request_state=_request_state(),
            baseline_update_choices={"A": False, "mass_kg": True},
            current_fingerprint="fp1",
            resolution_fingerprint="fp1",
        )

        selected = {item["field_key"] for item in plan["baseline_update_requests"]}
        self.assertEqual(selected, {"mass_kg"})

    def test_user_name_and_notes_are_preserved(self):
        resolution = _resolution_result([_proposal_result("proposal_req_1", display_index=1)])

        plan = build_vde_request_save_plan(
            resolution,
            save_mode=SAVE_MODE_VALID_ONLY,
            request_state=_request_state(),
            current_fingerprint="fp1",
            resolution_fingerprint="fp1",
        )

        note_text = plan["proposals_to_save"][0]["note_text"]
        self.assertIn("Mass -45 kg + Tire Lookup", note_text)
        self.assertIn("Request schema 0.1", note_text)

    def test_auto_name_is_generated_when_user_name_missing(self):
        proposal = _proposal_result("proposal_req_2", display_index=2)
        plan = build_vde_request_save_plan(
            _resolution_result([proposal]),
            save_mode=SAVE_MODE_VALID_ONLY,
            request_state=_request_state(),
            current_fingerprint="fp1",
            resolution_fingerprint="fp1",
        )

        self.assertTrue(plan["proposals_to_save"][0]["final_name"])
        self.assertEqual(generate_auto_proposal_name(proposal), "Mass + Tire Lookup")

    def test_preview_stale_and_fingerprint_mismatch_block_save(self):
        resolution = _resolution_result([_proposal_result("proposal_req_1", display_index=1)])

        stale_plan = build_vde_request_save_plan(
            resolution,
            save_mode=SAVE_MODE_VALID_ONLY,
            request_state=_request_state(),
            current_fingerprint="fp1",
            resolution_fingerprint="fp1",
            preview_is_stale=True,
        )
        mismatch_plan = build_vde_request_save_plan(
            resolution,
            save_mode=SAVE_MODE_VALID_ONLY,
            request_state=_request_state(),
            current_fingerprint="fp2",
            resolution_fingerprint="fp1",
        )

        self.assertFalse(stale_plan["can_execute"])
        self.assertFalse(mismatch_plan["can_execute"])

    def test_duplicate_protection_skips_already_saved_proposals(self):
        resolution = _resolution_result([_proposal_result("proposal_req_1", display_index=1)])

        plan = build_vde_request_save_plan(
            resolution,
            save_mode=SAVE_MODE_SELECTED,
            selected_proposal_ids=["proposal_req_1"],
            request_state=_request_state(),
            current_fingerprint="fp1",
            resolution_fingerprint="fp1",
            previous_save_fingerprint="fp1",
            previously_saved_proposal_ids=["proposal_req_1"],
        )

        self.assertFalse(plan["can_execute"])
        self.assertEqual(plan["skipped_proposals"][0]["reason"], "already_saved_for_current_preview")


class VdeRequestSaveExecutionTests(unittest.TestCase):
    def _temp_db_services(self):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        tmp.close()
        con = sqlite3.connect(tmp.name)
        con.execute(
            """
            CREATE TABLE vde_db (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                legislation TEXT,
                category TEXT,
                make TEXT,
                model TEXT,
                year INTEGER,
                notes TEXT,
                mass_kg REAL,
                test_mass_kg REAL,
                test_mass_low_kg REAL,
                test_mass_high_kg REAL,
                test_mass_basis TEXT,
                inertia_class REAL,
                weight_dist_fr_pct REAL,
                payload_kg REAL,
                wltp_category TEXT,
                cda_m2 REAL,
                coast_A_N REAL,
                coast_B_N_per_kph REAL,
                coast_C_N_per_kph2 REAL,
                vde_total_mj_per_km REAL,
                vde_net_mj_per_km REAL,
                cycle_name TEXT,
                cycle_source TEXT,
                vde_id_parent INTEGER,
                baseline_A_N REAL,
                baseline_B_N_per_kph REAL,
                baseline_C_N_per_kph2 REAL,
                baseline_mass_kg REAL,
                front_tire_id INTEGER,
                rear_tire_id INTEGER,
                tire_A_final REAL,
                tire_B_final REAL,
                tire_C_final REAL,
                tire_calc_source TEXT,
                tire_load_mass_basis TEXT,
                tire_improvement_pct REAL,
                rrc_N_per_kN REAL,
                smerf REAL,
                front_pressure_psi REAL,
                rear_pressure_psi REAL,
                trans_A_coef_N REAL,
                trans_B_coef_Npkph REAL,
                trans_C_coef_Npkph2 REAL,
                brake_A_coef_N REAL,
                brake_B_coef_Npkph REAL,
                brake_C_coef_Npkph2 REAL,
                parasitic_A_coef_N REAL,
                parasitic_B_coef_Npkph REAL,
                parasitic_C_coef_Npkph2 REAL,
                gvwr_kg REAL,
                gcwr_kg REAL,
                trailer_mass_kg REAL,
                trailer_code TEXT,
                trailer_roadload_source TEXT,
                trailer_A_coef_N REAL,
                trailer_B_coef_Npkph REAL,
                trailer_C_coef_Npkph2 REAL,
                mass_rule_status TEXT,
                mass_rule_notes TEXT,
                updated_at TEXT
            )
            """
        )
        con.execute(
            "INSERT INTO vde_db (id, make, model, coast_A_N, mass_kg) VALUES (5038, 'FORD', 'TEST', 120.0, 1600.0)"
        )
        con.commit()
        con.close()
        def _table_columns(_table):
            table_con = sqlite3.connect(tmp.name)
            try:
                return [row[1] for row in table_con.execute("PRAGMA table_info(vde_db)").fetchall()]
            finally:
                table_con.close()

        services = {
            "ensure_db": lambda: None,
            "connect_db": lambda: sqlite3.connect(tmp.name),
            "table_columns": _table_columns,
        }
        return tmp.name, services

    def test_execute_save_creates_one_row_per_proposal_and_updates_baseline(self):
        proposals = [
            _proposal_result("proposal_req_1", display_index=1),
            _proposal_result("proposal_req_2", display_index=2, source_column="Requested #2"),
        ]
        plan = build_vde_request_save_plan(
            _resolution_result(proposals),
            save_mode=SAVE_MODE_SELECTED,
            selected_proposal_ids=["proposal_req_1", "proposal_req_2"],
            request_state=_request_state(),
            current_fingerprint="fp1",
            resolution_fingerprint="fp1",
        )
        db_path, services = self._temp_db_services()
        self.addCleanup(lambda: __import__("pathlib").Path(db_path).unlink(missing_ok=True))

        result = execute_vde_request_save_plan(plan, services=services)

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["saved_proposals"]), 2)
        con = sqlite3.connect(db_path)
        count = con.execute("SELECT COUNT(*) FROM vde_db").fetchone()[0]
        updated = con.execute("SELECT coast_A_N, mass_kg FROM vde_db WHERE id=5038").fetchone()
        con.close()
        self.assertEqual(count, 3)
        self.assertEqual(updated, (121.0, 1650.0))

    def test_execute_save_returns_partial_when_component_creation_fails(self):
        proposal = _proposal_result(
            "proposal_req_1",
            display_index=1,
            component_actions=[
                {
                    "action": "eligible_for_new_component",
                    "domain": "transmission",
                    "component_id": None,
                    "component_snapshot": {"new_trans_A": 3.0, "new_trans_B": 0.1, "new_trans_C": 0.01, "loss_pct": 2.0},
                    "requires_confirmation": True,
                    "issues": [],
                }
            ],
        )
        plan = build_vde_request_save_plan(
            _resolution_result([proposal]),
            save_mode=SAVE_MODE_SELECTED,
            selected_proposal_ids=["proposal_req_1"],
            request_state=_request_state(),
            current_fingerprint="fp1",
            resolution_fingerprint="fp1",
            component_creation_confirmations={"proposal_req_1:transmission": True},
        )
        db_path, services = self._temp_db_services()
        self.addCleanup(lambda: __import__("pathlib").Path(db_path).unlink(missing_ok=True))
        services["create_component"] = lambda domain, payload: (_ for _ in ()).throw(ValueError("duplicate component"))

        result = execute_vde_request_save_plan(plan, services=services)

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["saved_proposals"][0]["status"], "saved")
        self.assertEqual(result["component_results"][0]["status"], "component_creation_failed")

    def test_execute_blocked_plan_fails_without_writing(self):
        proposal = _proposal_result("proposal_req_1", display_index=1, total_mj_per_km=None)
        plan = build_vde_request_save_plan(
            _resolution_result([proposal]),
            save_mode=SAVE_MODE_SELECTED,
            selected_proposal_ids=["proposal_req_1"],
            request_state=_request_state(),
            current_fingerprint="fp1",
            resolution_fingerprint="fp1",
        )

        result = execute_vde_request_save_plan(plan, services={"ensure_db": lambda: None})

        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["issues"])


if __name__ == "__main__":
    unittest.main()
