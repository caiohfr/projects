from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest

from src.vde_core import db as db_module
from src.vde_core.component_repositories import ComponentRepository, load_mock_component_repository
from src.vde_core.qa_mock_data import QA_DATA_DIR, seed_qa_database
from src.vde_core.repositories import fetch_vde_by_id
from src.vde_core.vde_request_compact_adapter import build_v22_preview_bundle, compact_baseline_context
from src.vde_core.vde_request_compact_persistence import (
    REQUEST_HISTORY_PROPOSAL_TABLE,
    REQUEST_HISTORY_TABLE,
    build_v22_save_plan,
    load_v22_saved_request,
    save_v22_request,
    saved_component_repositories_from_state,
)
from src.vde_core.vde_request_compact_state import (
    apply_v22_baseline,
    apply_v22_corrections,
    apply_v22_domain_inputs,
    apply_v22_new_test_baseline,
    apply_v22_proposal_metadata,
    apply_v22_proposal_matrix,
    create_v22_state,
)
from src.vde_core.vde_request_report import (
    build_vde_request_report_model,
    generate_vde_request_report_xlsx,
)


class TestVdeRequestCompactPersistence(unittest.TestCase):
    maxDiff = None

    def _temp_db_path(self) -> Path:
        QA_DATA_DIR.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="qa_v22_save_", dir=str(QA_DATA_DIR)))
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        return temp_dir / "eco_drive_v22_save.db"

    def _build_ready_state(self):
        baseline_row = fetch_vde_by_id(900001)
        self.assertIsNotNone(baseline_row)
        baseline_row = deepcopy(baseline_row)
        baseline_row["weight_dist_fr_pct"] = None
        state = apply_v22_baseline(create_v22_state(), baseline_row)
        state = apply_v22_corrections(
            state,
            {
                "cda_m2": 0.61,
                "payload_kg": 0.0,
            },
        )
        state = apply_v22_proposal_matrix(
            state,
            [
                {
                    "proposal_id": "requested_1",
                    "walk_from": "baseline",
                    "mass": "Curb mass -> EPA TWC",
                    "aero": "Absolute CdA",
                    "tire": "Target final RRC",
                    "transmission": "Lookup from DB",
                },
                {
                    "proposal_id": "requested_2",
                    "walk_from": "requested_1",
                    "mass": "TWC shift / target class",
                    "aero": "Inherit",
                    "tire": "Inherit",
                    "transmission": "Inherit",
                    "brake": "Delta ABC",
                },
            ],
        )
        transmission = load_mock_component_repository("transmission").get_by_id("TRANS-MOCK-001")
        self.assertIsNotNone(transmission)
        state = apply_v22_domain_inputs(
            state,
            "mass",
            {
                "requested_1": {"mass_kg": 1310.0},
                "requested_2": {"shift_steps": "-1", "curb_position": "Bottom"},
            },
        )
        state = apply_v22_domain_inputs(
            state,
            "aero",
            {
                "requested_1": {"cda_m2": 0.59},
            },
        )
        state = apply_v22_domain_inputs(
            state,
            "tire",
            {
                "requested_1": {
                    "target_rrc_N_per_kN": 7.6,
                    "front_pressure_psi": 36.0,
                    "rear_pressure_psi": 37.0,
                    "tire_load_mass_basis": "TEST_MASS",
                }
            },
        )
        state = apply_v22_domain_inputs(
            state,
            "transmission",
            {
                "requested_1": {
                    "transmission_component_db_id": "TRANS-MOCK-001",
                    "trans_A_coef_N": transmission["trans_A"],
                    "trans_B_coef_Npkph": transmission["trans_B"],
                    "trans_C_coef_Npkph2": transmission["trans_C"],
                    "transmission_loss_pct": transmission["loss_pct"],
                }
            },
        )
        state = apply_v22_domain_inputs(
            state,
            "brake",
            {
                "requested_2": {
                    "delta_A": 1.0,
                    "delta_B": 0.0,
                    "delta_C": 0.0,
                }
            },
        )
        bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        state["preview"] = {
            "status": "fresh",
            "fingerprint": bundle.get("fingerprint"),
            "result": bundle,
        }
        return state, bundle

    def _build_new_test_ready_state(self):
        state = apply_v22_new_test_baseline(
            create_v22_state(),
            {
                "A": 120.0,
                "B": 0.020,
                "C": 0.0080,
                "test_mass_kg": 1600.0,
                "legislation": "EPA",
                "cycle_name": "FTP75_HWFET",
                "category": "MIDSIZE CARS",
                "notes": "Synthetic new test baseline",
            },
        )
        state = apply_v22_proposal_matrix(
            state,
            [
                {
                    "proposal_id": "requested_1",
                    "walk_from": "baseline",
                    "mass": "Custom test mass",
                },
                {
                    "proposal_id": "requested_2",
                    "walk_from": "requested_1",
                    "mass": "Inherit",
                },
            ],
        )
        state = apply_v22_domain_inputs(
            state,
            "mass",
            {
                "requested_1": {"test_mass_kg": 1650.0},
            },
        )
        bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        state["preview"] = {
            "status": "fresh",
            "fingerprint": bundle.get("fingerprint"),
            "result": bundle,
        }
        return state, bundle

    def _build_tire_lookup_ready_state(self):
        baseline_row = fetch_vde_by_id(900001)
        self.assertIsNotNone(baseline_row)
        state = apply_v22_baseline(create_v22_state(), deepcopy(baseline_row))
        state = apply_v22_proposal_matrix(
            state,
            [
                {
                    "proposal_id": "requested_1",
                    "walk_from": "baseline",
                    "tire": "Tire DB lookup",
                }
            ],
        )
        state = apply_v22_domain_inputs(
            state,
            "tire",
            {
                "requested_1": {
                    "tire_db_id": 920104,
                    "tire_code": "QA-LOAD",
                    "rrc_N_per_kN": 8.8,
                    "front_pressure_psi": 30.0,
                    "rear_pressure_psi": 30.0,
                    "tire_load_mass_basis": "TEST_MASS",
                    "tire_snapshot": {
                        "id": 920104,
                        "tire_test_code": "QA-LOAD",
                        "standard_family": "SAE",
                        "rr_n_per_kn": 8.8,
                        "test_pressure_value": 30.0,
                        "test_load_value": 650.0,
                        "sae_reference_pressure_kpa": 206.8427,
                        "sae_reference_load_n": 6374.3225,
                        "sae_alpha": -0.28,
                        "sae_beta": 1.05,
                        "sae_a": 0.0231280363,
                        "sae_b": 0.000022,
                        "sae_c": 0.00000006,
                    },
                }
            },
        )
        bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        state["preview"] = {
            "status": "fresh",
            "fingerprint": bundle.get("fingerprint"),
            "result": bundle,
        }
        return state, bundle

    def _roundtrip_bundle(self, state: dict):
        repositories = saved_component_repositories_from_state(state)
        return build_v22_preview_bundle(
            state,
            baseline_context=compact_baseline_context(state),
            component_repositories=repositories,
        )

    def _proposal_metrics(self, bundle: dict) -> dict[str, dict]:
        metrics = {}
        for proposal in list(dict(bundle.get("resolution_result") or {}).get("proposal_results") or []):
            proposal_id = str(proposal.get("proposal_id") or "")
            snapshot = dict(proposal.get("resolved_snapshot") or {})
            transmission = dict(snapshot.get("transmission_losses") or {})
            metrics[proposal_id] = {
                "mass_kg": snapshot.get("mass_kg"),
                "inertia_class": snapshot.get("inertia_class"),
                "test_mass_kg": dict(snapshot.get("resolved_mass_setup") or {}).get("test_mass_kg") or snapshot.get("test_mass_kg"),
                "rrc_N_per_kN": snapshot.get("rrc_N_per_kN"),
                "transmission_application_mode": snapshot.get("transmission_application_mode"),
                "trans_abc": dict(transmission.get("abc") or {}),
                "abc_total": dict(proposal.get("abc_total") or {}),
                "abc_net": dict(proposal.get("abc_net") or {}),
                "vde_total": dict(dict(proposal.get("vde_results") or {}).get("total") or {}),
                "vde_net": dict(dict(proposal.get("vde_results") or {}).get("net") or {}),
            }
        return metrics

    def _assert_roundtrip_metrics(self, before_bundle: dict, after_bundle: dict) -> None:
        before = self._proposal_metrics(before_bundle)
        after = self._proposal_metrics(after_bundle)
        self.assertEqual(set(before), set(after))
        for proposal_id in before:
            with self.subTest(proposal_id=proposal_id):
                self.assertAlmostEqual(before[proposal_id]["mass_kg"], after[proposal_id]["mass_kg"], places=9)
                self.assertAlmostEqual(before[proposal_id]["inertia_class"], after[proposal_id]["inertia_class"], places=9)
                self.assertAlmostEqual(before[proposal_id]["test_mass_kg"], after[proposal_id]["test_mass_kg"], places=9)
                self.assertAlmostEqual(before[proposal_id]["rrc_N_per_kN"], after[proposal_id]["rrc_N_per_kN"], places=9)
                self.assertEqual(before[proposal_id]["transmission_application_mode"], after[proposal_id]["transmission_application_mode"])
                self.assertEqual(before[proposal_id]["trans_abc"], after[proposal_id]["trans_abc"])
                self.assertEqual(before[proposal_id]["abc_total"], after[proposal_id]["abc_total"])
                self.assertEqual(before[proposal_id]["abc_net"], after[proposal_id]["abc_net"])
                self.assertEqual(before[proposal_id]["vde_total"], after[proposal_id]["vde_total"])
                self.assertEqual(before[proposal_id]["vde_net"], after[proposal_id]["vde_net"])

    def test_db_preview_plan_and_save_share_final_metadata_payload(self):
        state, bundle = self._build_ready_state()
        state = apply_v22_proposal_metadata(
            state,
            "requested_1",
            {"name": "QA final identity", "make": "QA MAKE", "model": "QA MODEL", "model_year": "2028"},
        )

        plan = build_v22_save_plan(state)
        row = next(item for item in plan["proposals"] if item["proposal_id"] == "requested_1")

        self.assertTrue(plan["proposals"])
        self.assertEqual(row["final_name"], "QA final identity")
        self.assertEqual(row["row_payload"]["make"], "QA MAKE")
        self.assertEqual(row["row_payload"]["model"], "QA MODEL")
        self.assertEqual(row["row_payload"]["year"], "2028")
        self.assertEqual(state["preview"]["fingerprint"], bundle["fingerprint"])

    def test_save_plan_uses_configuration_summary_as_blank_name_fallback(self):
        state, _ = self._build_ready_state()

        plan = build_v22_save_plan(state)
        row = next(item for item in plan["proposals"] if item["proposal_id"] == "requested_1")
        summary = next(item for item in plan["configuration_summaries"] if item["proposal_id"] == "requested_1")

        self.assertTrue(row["final_name"])
        self.assertEqual(row["final_name"], summary["suggested_name"])
        self.assertEqual(plan["final_metadata_by_proposal"]["requested_1"]["name"], summary["suggested_name"])

    def test_save_reload_roundtrip_preserves_results_and_audit(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)

        with db_module.using_db_path(db_path):
            state, before_bundle = self._build_ready_state()
            before_report = build_vde_request_report_model(
                before_bundle["draft"],
                before_bundle["resolution_result"],
            )
            before_export = generate_vde_request_report_xlsx(before_report)

            save_result = save_v22_request(state)
            self.assertEqual(save_result["status"], "success")
            self.assertEqual(len(save_result["saved_proposals"]), 2)

            loaded = load_v22_saved_request(save_result["record_id"])
            self.assertIsNotNone(loaded)
            reloaded_state = loaded["state"]
            after_bundle = self._roundtrip_bundle(reloaded_state)
            after_report = loaded["report_model"]
            after_export = generate_vde_request_report_xlsx(after_report)

        self._assert_roundtrip_metrics(before_bundle, after_bundle)
        self.assertEqual(reloaded_state["baseline"]["printed"]["cda_m2"], state["baseline"]["printed"]["cda_m2"])
        self.assertEqual(reloaded_state["baseline"]["corrections"]["cda_m2"], 0.61)
        self.assertEqual(reloaded_state["baseline"]["corrections"]["payload_kg"], 0.0)
        self.assertEqual(reloaded_state["proposals"][1]["walk_from"], "requested_1")
        self.assertIn("saved_component_repository_snapshots", reloaded_state)
        self.assertFalse(any(str(key).startswith("v22_simple_") for key in reloaded_state.keys()))

        draft_req1_tire = loaded["draft"]["proposals"][0]["domain_requests"]["tire"]
        self.assertEqual(draft_req1_tire["raw_values"]["target_rrc_N_per_kN"], 7.6)
        self.assertEqual(after_bundle["resolution_result"]["proposal_results"][0]["resolved_snapshot"]["rrc_N_per_kN"], 7.6)

        warning_messages = [
            str(item.get("message") or "")
            for item in list(after_bundle["resolution_result"]["proposal_results"][0]["issues"] or [])
        ]
        self.assertIn("Front weight fraction defaulted to 50%.", warning_messages)

        component_actions = loaded["proposal_records"][0]["component_actions"]
        reuse_existing = [item for item in component_actions if item.get("action") == "reuse_existing"]
        self.assertTrue(reuse_existing)
        self.assertTrue(any(item.get("component_id") == "TRANS-MOCK-001" for item in reuse_existing))
        trans_action = next(item for item in reuse_existing if item.get("component_id") == "TRANS-MOCK-001")
        self.assertEqual(trans_action["component_snapshot"]["net_bridge_eligible"], "TRUE")
        self.assertEqual(trans_action["component_snapshot"]["component_type"], "TRANSMISSION")

        self.assertGreater(len(before_export), 0)
        self.assertGreater(len(after_export), 0)
        self.assertGreaterEqual(len(after_report["request_rows"]), len(before_report["request_rows"]))
        self.assertGreaterEqual(len(after_report["validation_rows"]), len(before_report["validation_rows"]))
        self.assertGreaterEqual(len(after_report["component_rows"]), len(before_report["component_rows"]))
        self.assertTrue(
            any("Front weight fraction defaulted to 50%." in str(row.get("Message") or row.get("Issues") or "") for row in after_report["validation_rows"])
        )
        self.assertTrue(any("TRANS-MOCK-001" in str(row) for row in after_report["component_rows"]))
        self.assertTrue(
            any(
                row.get("Component Type") == "TRANSMISSION" and row.get("NET Bridge Eligible") == "TRUE"
                for row in after_report["component_rows"]
            )
        )

    def test_new_test_roundtrip_is_independent_from_vde_baseline_row(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)

        with db_module.using_db_path(db_path):
            state, before_bundle = self._build_new_test_ready_state()
            save_result = save_v22_request(state)
            self.assertEqual(save_result["status"], "success")

            loaded = load_v22_saved_request(save_result["record_id"])
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["state"]["baseline"]["source_type"], "NEW_TEST")
            self.assertEqual(loaded["draft"]["baseline_source_type"], "NEW_TEST")
            self.assertEqual(loaded["draft"]["baseline_source_snapshot"]["baseline_source_type"], "NEW_TEST")

            after_bundle = self._roundtrip_bundle(loaded["state"])

        self._assert_roundtrip_metrics(before_bundle, after_bundle)
        self.assertEqual(loaded["state"]["baseline"]["effective"]["A"], 120.0)
        self.assertEqual(loaded["state"]["baseline"]["effective"]["B"], 0.020)
        self.assertEqual(loaded["state"]["baseline"]["effective"]["C"], 0.0080)
        self.assertEqual(loaded["state"]["baseline"]["effective"]["test_mass_kg"], 1600.0)

    def test_save_accepts_review_only_validation_without_hard_blocks(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)

        with db_module.using_db_path(db_path):
            state, _ = self._build_ready_state()
            state["preview"]["result"]["validation_summary"]["overall_status"] = "Review"
            state["preview"]["result"]["validation_summary"]["review_count"] = 2
            state["preview"]["result"]["validation_summary"]["missing_count"] = 0
            state["preview"]["result"]["validation_summary"]["invalid_count"] = 0
            state["preview"]["result"]["validation_summary"]["blocked_count"] = 0
            state["preview"]["result"]["validation_summary"]["warning_count"] = 2

            save_result = save_v22_request(state)

        self.assertEqual(save_result["status"], "success")
        self.assertEqual(len(save_result["saved_proposals"]), 2)

    def test_save_reload_preserves_transmission_application_mode(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)

        with db_module.using_db_path(db_path):
            state, before_bundle = self._build_ready_state()
            state = apply_v22_domain_inputs(
                state,
                "transmission",
                {
                    "requested_1": {
                        **dict(state["proposals"][0]["inputs"]["transmission"]),
                        "transmission_application_mode": "KEEP_TOTAL_FIXED",
                    }
                },
            )
            before_bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
            state["preview"] = {
                "status": "fresh",
                "fingerprint": before_bundle.get("fingerprint"),
                "result": before_bundle,
            }

            save_result = save_v22_request(state)
            self.assertEqual(save_result["status"], "success")
            loaded = load_v22_saved_request(save_result["record_id"])
            self.assertIsNotNone(loaded)
            after_bundle = self._roundtrip_bundle(loaded["state"])

        self._assert_roundtrip_metrics(before_bundle, after_bundle)
        before_snapshot = before_bundle["resolution_result"]["proposal_results"][0]["resolved_snapshot"]
        after_snapshot = after_bundle["resolution_result"]["proposal_results"][0]["resolved_snapshot"]
        self.assertEqual(loaded["state"]["proposals"][0]["inputs"]["transmission"]["transmission_application_mode"], "KEEP_TOTAL_FIXED")
        self.assertEqual(before_snapshot["transmission_application_mode"], "KEEP_TOTAL_FIXED")
        self.assertEqual(after_snapshot["transmission_application_mode"], "KEEP_TOTAL_FIXED")
        self.assertEqual(before_bundle["resolution_result"]["proposal_results"][0]["abc_total"], after_bundle["resolution_result"]["proposal_results"][0]["abc_total"])

    def test_saved_request_ignores_baseline_db_mutation(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)

        with db_module.using_db_path(db_path):
            state, before_bundle = self._build_ready_state()
            save_result = save_v22_request(state)
            self.assertEqual(save_result["status"], "success")
            with sqlite3.connect(str(db_path)) as con:
                con.execute(
                    "UPDATE vde_db SET mass_kg=?, test_mass_kg=?, inertia_class=?, cda_m2=? WHERE id=?",
                    [2200.0, 2404.0, 2404.0, 0.73, 900001],
                )
                con.commit()

            loaded = load_v22_saved_request(save_result["record_id"])
            after_bundle = self._roundtrip_bundle(loaded["state"])

        self._assert_roundtrip_metrics(before_bundle, after_bundle)

    def test_saved_request_preserves_tire_lookup_snapshot_after_tire_db_mutation(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)

        with db_module.using_db_path(db_path):
            state, before_bundle = self._build_tire_lookup_ready_state()
            save_result = save_v22_request(state)
            self.assertEqual(save_result["status"], "success")

            with sqlite3.connect(str(db_path)) as con:
                con.execute(
                    """
                    UPDATE tire_roadload_db
                    SET rr_n_per_kn=?, sae_alpha=?, sae_beta=?, sae_a=?, sae_b=?, sae_c=?
                    WHERE id=?
                    """,
                    [11.9, -0.55, 1.22, 0.081, 0.00011, 0.00000021, 920104],
                )
                con.commit()

            loaded = load_v22_saved_request(save_result["record_id"])
            self.assertIsNotNone(loaded)
            after_bundle = self._roundtrip_bundle(loaded["state"])

        self._assert_roundtrip_metrics(before_bundle, after_bundle)
        loaded_inputs = loaded["state"]["proposals"][0]["inputs"]["tire"]
        self.assertEqual(loaded_inputs["tire_code"], "QA-LOAD")
        self.assertAlmostEqual(loaded_inputs["tire_snapshot"]["sae_alpha"], -0.28, places=9)
        self.assertAlmostEqual(loaded_inputs["tire_snapshot"]["sae_beta"], 1.05, places=9)
        self.assertAlmostEqual(loaded_inputs["tire_snapshot"]["sae_a"], 0.0231280363, places=10)
        proposal_record = loaded["proposal_records"][0]
        component_snapshot = proposal_record["component_actions"][0]["component_snapshot"]
        self.assertEqual(component_snapshot["tire_code"], "QA-LOAD")
        self.assertEqual(component_snapshot["tire_snapshot"]["tire_test_code"], "QA-LOAD")
        self.assertAlmostEqual(component_snapshot["tire_snapshot"]["sae_alpha"], -0.28, places=9)

    def test_saved_request_uses_component_snapshots_not_mutated_repository(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)

        with db_module.using_db_path(db_path):
            state, before_bundle = self._build_ready_state()
            save_result = save_v22_request(state)
            loaded = load_v22_saved_request(save_result["record_id"])
            reloaded_state = loaded["state"]

            baseline_context = compact_baseline_context(reloaded_state)
            snapshot_bundle = build_v22_preview_bundle(
                reloaded_state,
                baseline_context=baseline_context,
                component_repositories=saved_component_repositories_from_state(reloaded_state),
            )

            repo = load_mock_component_repository("transmission")
            mutated_component = deepcopy(repo.get_by_id("TRANS-MOCK-001"))
            mutated_component["trans_A"] = 99.0
            mutated_component["trans_B"] = 0.099
            mutated_component["trans_C"] = 0.0099
            mutated_component["net_bridge_eligible"] = "FALSE"
            mutated_component["physical_boundary"] = "MUTATED BOUNDARY"
            mutated_repo = ComponentRepository(
                domain="transmission",
                source="mutated_for_test",
                _components=[mutated_component],
                _issues=[],
                _by_id={"TRANS-MOCK-001": deepcopy(mutated_component)},
            )
            mutated_bundle = build_v22_preview_bundle(
                reloaded_state,
                baseline_context=baseline_context,
                component_repositories={"transmission": mutated_repo},
            )

        self._assert_roundtrip_metrics(before_bundle, snapshot_bundle)
        before_trans = self._proposal_metrics(before_bundle)["requested_1"]["trans_abc"]
        mutated_trans = self._proposal_metrics(mutated_bundle)["requested_1"]["trans_abc"]
        self.assertNotEqual(before_trans, mutated_trans)
        snapshot_action = snapshot_bundle["resolution_result"]["proposal_results"][0]["domain_results"]["transmission"]["component_action"]
        mutated_action = mutated_bundle["resolution_result"]["proposal_results"][0]["domain_results"]["transmission"]["component_action"]
        self.assertEqual(snapshot_action["component_snapshot"]["net_bridge_eligible"], "TRUE")
        self.assertNotEqual(snapshot_action["component_snapshot"]["physical_boundary"], "MUTATED BOUNDARY")
        self.assertEqual(mutated_action["component_snapshot"]["net_bridge_eligible"], "FALSE")

    def test_save_is_append_only_and_duplicate_safe(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)

        with db_module.using_db_path(db_path):
            state_a, bundle_a = self._build_ready_state()
            first = save_v22_request(state_a)
            second = save_v22_request(state_a)
            self.assertEqual(first["status"], "success")
            self.assertEqual(second["status"], "success")
            self.assertNotEqual(first["record_id"], second["record_id"])

            state_b = apply_v22_domain_inputs(state_a, "aero", {"requested_1": {"cda_m2": 0.57}})
            bundle_b = build_v22_preview_bundle(state_b, baseline_context=compact_baseline_context(state_b))
            state_b["preview"] = {"status": "fresh", "fingerprint": bundle_b["fingerprint"], "result": bundle_b}
            third = save_v22_request(state_b)
            self.assertEqual(third["status"], "success")

            with sqlite3.connect(str(db_path)) as con:
                request_count = con.execute(f"SELECT COUNT(*) FROM {REQUEST_HISTORY_TABLE}").fetchone()[0]
                proposal_count = con.execute(f"SELECT COUNT(*) FROM {REQUEST_HISTORY_PROPOSAL_TABLE}").fetchone()[0]
                vde_count = con.execute("SELECT COUNT(*) FROM vde_db").fetchone()[0]

        self.assertEqual(request_count, 3)
        self.assertEqual(proposal_count, 6)
        self.assertEqual(vde_count, 14)
        self.assertNotEqual(
            self._proposal_metrics(bundle_a)["requested_1"]["abc_total"],
            self._proposal_metrics(bundle_b)["requested_1"]["abc_total"],
        )

    def test_save_rolls_back_history_and_vde_rows_on_failure(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)

        with db_module.using_db_path(db_path):
            state, _ = self._build_ready_state()
            result = save_v22_request(
                state,
                services={
                    "insert_vde_row": lambda con, row_payload, supported_columns: (_ for _ in ()).throw(
                        ValueError("forced insert failure")
                    )
                },
            )
            self.assertEqual(result["status"], "failed")
            self.assertTrue(result["issues"])

            with sqlite3.connect(str(db_path)) as con:
                request_count = con.execute(f"SELECT COUNT(*) FROM {REQUEST_HISTORY_TABLE}").fetchone()[0]
                proposal_count = con.execute(f"SELECT COUNT(*) FROM {REQUEST_HISTORY_PROPOSAL_TABLE}").fetchone()[0]
                vde_count = con.execute("SELECT COUNT(*) FROM vde_db").fetchone()[0]

        self.assertEqual(request_count, 0)
        self.assertEqual(proposal_count, 0)
        self.assertEqual(vde_count, 8)


if __name__ == "__main__":
    unittest.main()
