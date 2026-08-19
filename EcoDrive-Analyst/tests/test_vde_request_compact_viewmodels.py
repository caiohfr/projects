from __future__ import annotations

from copy import deepcopy
import unittest

from src.vde_app.components.vde_request_compact_viewmodels import (
    build_active_corrections_summary,
    build_baseline_candidate_status_payload,
    build_domain_card_payload,
    build_engineering_comparison_payload,
    build_loaded_baseline_summary_payload,
    build_cycle_power_analysis_payload,
    build_roadload_analysis_payload,
    build_preview_audit_payload,
    build_preview_status_payload,
    build_request_inputs_overview_payload,
    build_scenario_overview_payload,
    build_vde_cycle_comparison_payload,
    build_validation_summary_payload,
    build_v22_branding_payload,
    build_v22_domain_status_payload,
    build_v22_flow_status_payload,
    format_v22_issue_for_display,
    proposal_display_label,
    walk_from_display_label,
)
from src.vde_core.vde_request_compact_adapter import build_v22_preview_bundle, compact_baseline_context
from src.vde_core.vde_request_compact_state import (
    apply_v22_baseline,
    apply_v22_domain_inputs,
    apply_v22_proposal_matrix,
    build_v22_canonical_request_draft,
    create_v22_state,
)


def _baseline_row() -> dict:
    return {
        "id": 4998,
        "make": "AUDI",
        "model": "Q6",
        "year": 2027,
        "legislation": "EPA",
        "cycle_name": "FTP75",
        "notes": "Compact baseline row",
        "mass_kg": 1600.0,
        "test_mass_kg": 1736.0,
        "inertia_class": 1750.0,
        "weight_dist_fr_pct": 55.0,
        "cda_m2": 0.62,
        "A": 120.0,
        "B": 0.02,
        "C": 0.01,
        "rrc_N_per_kN": 8.4,
        "front_pressure_psi": 35.0,
        "rear_pressure_psi": 35.0,
        "brake_A_coef_N": 4.0,
        "brake_B_Npkph": 0.001,
        "brake_C_coef_Npkph2": 0.0001,
    }


class TestVdeRequestCompactViewmodels(unittest.TestCase):
    def _loaded_state(self) -> dict:
        return apply_v22_baseline(create_v22_state(), _baseline_row())

    def _preview_ready_state(self) -> dict:
        state = self._loaded_state()
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Custom test mass", "aero": "Delta CdA"},
                {"proposal_id": "requested_2", "walk_from": "requested_1", "mass": "Inherit", "aero": "Inherit"},
            ],
        )
        state = apply_v22_domain_inputs(state, "mass", {"requested_1": {"test_mass_kg": 1810.0}})
        state = apply_v22_domain_inputs(state, "aero", {"requested_1": {"delta_CdA": -0.01}})
        bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        state["preview"] = {"status": "fresh", "fingerprint": bundle["fingerprint"], "result": bundle}
        return state

    def test_branding_payload_requires_loaded_baseline(self):
        state = create_v22_state()
        state["baseline"]["printed"] = {"make": "AUDI", "legislation": "EPA"}

        payload = build_v22_branding_payload(state)

        self.assertFalse(payload["loaded"])
        self.assertEqual(payload["make"], "")
        self.assertEqual(payload["legislation"], "")

    def test_branding_payload_uses_effective_with_printed_fallback(self):
        state = self._loaded_state()
        state["baseline"]["effective"]["make"] = ""

        payload = build_v22_branding_payload(state)

        self.assertTrue(payload["loaded"])
        self.assertEqual(payload["make"], "AUDI")
        self.assertEqual(payload["legislation"], "EPA")

    def test_proposal_and_walk_from_labels_use_display_indexes(self):
        state = create_v22_state()
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "name": "Alpha", "walk_from": "baseline"},
                {"proposal_id": "requested_2", "walk_from": "requested_1"},
            ],
        )

        self.assertEqual(proposal_display_label(state, state["proposals"][0]), "Requested #1")
        self.assertEqual(walk_from_display_label(state, "baseline"), "Baseline")
        self.assertEqual(walk_from_display_label(state, "requested_1"), "Requested #1")

    def test_flow_without_baseline_marks_pending_steps(self):
        payload = build_v22_flow_status_payload(create_v22_state())
        steps = {item["key"]: item for item in payload["steps"]}

        self.assertEqual(steps["baseline"]["base_status"], "pending")
        self.assertEqual(steps["matrix"]["base_status"], "pending")
        self.assertEqual(steps["inputs"]["base_status"], "pending")
        self.assertEqual(steps["preview"]["base_status"], "pending")
        self.assertEqual(payload["context_strip"][0]["value"], "Not loaded")

    def test_flow_with_loaded_baseline_marks_baseline_complete(self):
        payload = build_v22_flow_status_payload(self._loaded_state())
        steps = {item["key"]: item for item in payload["steps"]}

        self.assertEqual(steps["baseline"]["base_status"], "complete")
        self.assertEqual(steps["baseline"]["summary"], "VDE #4998")

    def test_matrix_complete_when_direct_domain_and_valid_walk_from(self):
        state = self._loaded_state()
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Custom test mass"}])

        payload = build_v22_flow_status_payload(state)
        steps = {item["key"]: item for item in payload["steps"]}

        self.assertEqual(steps["matrix"]["base_status"], "complete")
        self.assertIn("1 direct domains", steps["matrix"]["summary"])

    def test_matrix_review_when_walk_from_invalid(self):
        state = self._loaded_state()
        state["proposals"][1]["walk_from"] = "requested_99"

        payload = build_v22_flow_status_payload(state)
        matrix = next(item for item in payload["steps"] if item["key"] == "matrix")

        self.assertEqual(matrix["base_status"], "review")
        self.assertIn("Requested #2", matrix["detail"])

    def test_inputs_pending_for_direct_domain_not_applied(self):
        state = self._loaded_state()
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Custom test mass"}])

        payload = build_v22_flow_status_payload(state)
        inputs = next(item for item in payload["steps"] if item["key"] == "inputs")

        self.assertEqual(inputs["base_status"], "pending")
        self.assertEqual(inputs["summary"], "0/1 inputs applied")
        self.assertIn("Mass pending apply", inputs["detail"])

    def test_inputs_complete_when_all_direct_domains_ready(self):
        state = self._loaded_state()
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Custom test mass"}])
        state = apply_v22_domain_inputs(state, "mass", {"requested_1": {"test_mass_kg": 1800.0}})

        domain_payload = build_v22_domain_status_payload(state, "mass")
        flow_payload = build_v22_flow_status_payload(state)
        inputs = next(item for item in flow_payload["steps"] if item["key"] == "inputs")

        self.assertEqual(domain_payload["status"], "applied_ready")
        self.assertEqual(domain_payload["revision"], 1)
        self.assertEqual(domain_payload["ready_count"], 1)
        self.assertEqual(inputs["base_status"], "complete")
        self.assertEqual(inputs["summary"], "1/1 inputs applied")

    def test_inputs_review_when_domain_applied_incomplete(self):
        state = self._loaded_state()
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "GVWR loaded mass"}])
        state = apply_v22_domain_inputs(state, "mass", {"requested_1": {}})

        payload = build_v22_flow_status_payload(state)
        inputs = next(item for item in payload["steps"] if item["key"] == "inputs")

        self.assertEqual(inputs["base_status"], "review")
        self.assertIn("Mass incomplete", inputs["detail"])

    def test_inputs_stale_when_domain_stale_after_matrix_change(self):
        state = self._loaded_state()
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Custom test mass"}])
        state = apply_v22_domain_inputs(state, "mass", {"requested_1": {"test_mass_kg": 1800.0}})
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "GVWR loaded mass"}])

        payload = build_v22_flow_status_payload(state)
        inputs = next(item for item in payload["steps"] if item["key"] == "inputs")

        self.assertEqual(inputs["base_status"], "stale")
        self.assertIn("Mass stale", inputs["detail"])

    def test_inputs_exclude_inherit_and_not_used_from_pending_counts(self):
        state = self._loaded_state()
        state = apply_v22_proposal_matrix(
            state,
            [
                {
                    "proposal_id": "requested_1",
                    "walk_from": "baseline",
                    "mass": "Custom test mass",
                    "brake": "Not used",
                    "aero": "Inherit",
                }
            ],
        )
        state = apply_v22_domain_inputs(state, "mass", {"requested_1": {"test_mass_kg": 1810.0}})

        payload = build_v22_flow_status_payload(state)
        inputs = next(item for item in payload["steps"] if item["key"] == "inputs")

        self.assertEqual(inputs["summary"], "1/1 inputs applied")
        self.assertEqual(inputs["base_status"], "complete")

    def test_request_inputs_overview_empty_without_direct_domains(self):
        payload = build_request_inputs_overview_payload(self._loaded_state())

        self.assertEqual(payload["active_domain_count"], 0)
        self.assertEqual(payload["summary"], "0 direct domains | 0 applied")
        self.assertFalse(payload["has_active_domains"])
        self.assertEqual(payload["pending_count"], 0)

    def test_request_inputs_overview_counts_mixed_statuses_and_excludes_inactive_domains(self):
        state = self._loaded_state()
        state = apply_v22_proposal_matrix(
            state,
            [
                {
                    "proposal_id": "requested_1",
                    "walk_from": "baseline",
                    "mass": "Custom test mass",
                    "aero": "Delta CdA",
                    "tire": "Target final RRC",
                    "brake": "Not used",
                },
                {
                    "proposal_id": "requested_2",
                    "walk_from": "requested_1",
                    "mass": "Inherit",
                    "aero": "Inherit",
                    "tire": "Inherit",
                    "brake": "Inherit",
                },
            ],
        )
        state = apply_v22_domain_inputs(state, "mass", {"requested_1": {"test_mass_kg": 1810.0}})
        state = apply_v22_domain_inputs(state, "tire", {"requested_1": {}})
        state = apply_v22_domain_inputs(state, "aero", {"requested_1": {"delta_CdA": -0.01}})
        state = apply_v22_proposal_matrix(
            state,
            [
                {
                    "proposal_id": "requested_1",
                    "walk_from": "baseline",
                    "mass": "Custom test mass",
                    "aero": "Absolute CdA",
                    "tire": "Target final RRC",
                    "brake": "Not used",
                },
                {
                    "proposal_id": "requested_2",
                    "walk_from": "requested_1",
                    "mass": "Inherit",
                    "aero": "Inherit",
                    "tire": "Inherit",
                    "brake": "Inherit",
                },
            ],
        )

        payload = build_request_inputs_overview_payload(state)

        self.assertEqual(payload["active_domain_count"], 3)
        self.assertEqual(payload["ready_count"], 1)
        self.assertEqual(payload["review_count"], 1)
        self.assertEqual(payload["stale_count"], 1)
        self.assertEqual(payload["pending_count"], 0)
        self.assertEqual(payload["summary"], "3 direct domains | 1 applied | 1 incomplete | 1 pending")

    def test_tire_warning_only_stays_ready_and_preview_eligible(self):
        baseline = _baseline_row()
        baseline.pop("weight_dist_fr_pct", None)
        baseline["tire_load_mass_basis"] = "TEST_MASS"
        baseline["tire_A_final"] = 48.0
        baseline["tire_B_final"] = 0.009
        baseline["tire_C_final"] = 0.001
        baseline["trans_A_coef_N"] = 6.0
        baseline["trans_B_coef_Npkph"] = 0.003
        baseline["trans_C_coef_Npkph2"] = 0.001
        state = apply_v22_baseline(create_v22_state(), baseline)
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "tire": "Target final RRC"}])
        state = apply_v22_domain_inputs(
            state,
            "tire",
            {"requested_1": {"target_rrc_N_per_kN": 9.0, "front_pressure_psi": 36.0, "rear_pressure_psi": 36.0}},
        )

        domain_payload = build_v22_domain_status_payload(state, "tire")
        overview = build_request_inputs_overview_payload(state)
        warning_issues = state["domain_input_state"]["tire"]["proposal_statuses"]["requested_1"]["issues"]
        bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        state["preview"] = {"status": "fresh", "fingerprint": bundle["fingerprint"], "result": bundle}
        preview = build_preview_status_payload(state)

        self.assertEqual(domain_payload["status"], "applied_ready")
        self.assertEqual(domain_payload["ready_count"], 1)
        self.assertEqual(domain_payload["incomplete_count"], 0)
        self.assertEqual(overview["ready_count"], 1)
        self.assertEqual(overview["review_count"], 0)
        self.assertEqual(overview["summary"], "1 direct domains | 1 applied")
        self.assertIn("Front weight fraction defaulted to 50%.", warning_issues)
        self.assertEqual(preview["validation_status"], "OK")
        self.assertEqual(preview["save_status"], "Eligible")

    def test_domain_card_payload_includes_walk_from_last_applied_and_corrections(self):
        state = self._loaded_state()
        state["baseline"]["corrections"] = {"cda_m2": 0.61}
        state["baseline"]["effective"]["cda_m2"] = 0.61
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "aero": "Delta CdA"},
                {"proposal_id": "requested_2", "walk_from": "requested_1", "aero": "Inherit"},
            ],
        )
        state = apply_v22_domain_inputs(state, "aero", {"requested_1": {"delta_CdA": -0.01}})

        payload = build_domain_card_payload(state, "aero", "Metric")

        self.assertTrue(payload["is_active"])
        self.assertEqual(payload["status_key"], "ready")
        self.assertEqual(payload["proposal_type_summary"], "Delta CdA")
        self.assertIn("Requested #1 <- Baseline", payload["walk_from_lines"])
        self.assertEqual(payload["proposal_summaries"][1]["mode_label"], "Inherit")
        self.assertTrue(payload["last_applied_at"])
        self.assertEqual(payload["reference_changes"][0]["field_label"], "CdA")
        self.assertEqual(payload["reference_changes"][0]["effective_value"], "0.61")

    def test_domain_card_payload_formats_us_units_without_mutating_state(self):
        state = self._loaded_state()
        state["baseline"]["corrections"] = {"inertia_class": 1928.0}
        state["baseline"]["effective"]["inertia_class"] = 1928.0
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Custom test mass"}])
        original = deepcopy(state)

        payload = build_domain_card_payload(state, "mass", "US customary")

        inertia_row = next(item for item in payload["reference_changes"] if item["field_key"] == "inertia_class")
        self.assertEqual(inertia_row["effective_value"], "4251")
        self.assertEqual(inertia_row["field_label"], "EPA ETW / TWC")
        self.assertEqual(state, original)

    def test_preview_status_payload_reports_not_generated_and_stale_without_mutation(self):
        state = self._loaded_state()
        original = deepcopy(state)

        pending = build_preview_status_payload(state)
        self.assertEqual(pending["preview_status"], "not_run")
        self.assertEqual(pending["preview_label"], "Not generated")
        self.assertFalse(pending["has_bundle"])

        state["preview"]["status"] = "stale"
        stale = build_preview_status_payload(state)
        self.assertEqual(stale["preview_label"], "Stale")
        self.assertIn("changed after this preview", stale["stale_message"])
        self.assertEqual(original["baseline"], state["baseline"])

    def test_scenario_overview_payload_shows_walk_from_and_inheritance(self):
        state = self._preview_ready_state()

        payload = build_scenario_overview_payload(state)

        self.assertTrue(payload["has_bundle"])
        self.assertEqual(payload["scenarios"][0]["label"], "Baseline")
        self.assertEqual(payload["scenarios"][1]["walk_from"], "Baseline")
        self.assertEqual(payload["scenarios"][2]["walk_from"], "Requested #1")
        self.assertIn("Mass", payload["scenarios"][2]["inherited"])
        self.assertIn("Aero", payload["scenarios"][2]["inherited"])

    def test_engineering_comparison_payload_shows_inherited_resolved_values_and_units(self):
        state = self._preview_ready_state()

        metric_payload = build_engineering_comparison_payload(state, "Metric")
        us_payload = build_engineering_comparison_payload(state, "US customary")

        mass_group = next(group for group in metric_payload["groups"] if group["title"] == "Mass")
        curb_row = next(row for row in mass_group["rows"] if row["field_key"] == "mass_kg")
        test_mass_row = next(row for row in mass_group["rows"] if row["field_key"] == "test_mass_kg")
        cda_group = next(group for group in metric_payload["groups"] if group["title"] == "Aero")
        cda_row = next(row for row in cda_group["rows"] if row["field_key"] == "CdA")
        roadload_group = next(group for group in metric_payload["groups"] if group["title"] == "Resulting Roadload")
        total_c_row = next(row for row in roadload_group["rows"] if row["field_key"] == "abc_total_C")
        vde_group = next(group for group in metric_payload["groups"] if group["title"] == "VDE")
        vde_total_row = next(row for row in vde_group["rows"] if row["field_key"] == "vde_total_mj_per_km")

        self.assertEqual(curb_row["display_values"]["requested_1"], "1600")
        self.assertEqual(curb_row["display_values"]["requested_2"], "1600")
        self.assertEqual(test_mass_row["display_values"]["requested_1"], "1810")
        self.assertEqual(test_mass_row["display_values"]["requested_2"], "1810")
        self.assertEqual(cda_row["display_values"]["requested_1"], "0.61")
        self.assertEqual(cda_row["display_values"]["requested_2"], "0.61")
        self.assertNotEqual(total_c_row["display_values"]["baseline"], total_c_row["display_values"]["requested_1"])
        self.assertNotEqual(vde_total_row["display_values"]["requested_1"], "—")

        self.assertIn("Mass", metric_payload["changed_group_titles"])
        self.assertIn("Aero", metric_payload["changed_group_titles"])
        self.assertIn("Resulting Roadload", metric_payload["changed_group_titles"])
        self.assertIn("VDE", metric_payload["changed_group_titles"])

        us_mass_group = next(group for group in us_payload["groups"] if group["title"] == "Mass")
        us_curb_row = next(row for row in us_mass_group["rows"] if row["field_key"] == "mass_kg")
        self.assertEqual(us_curb_row["unit"], "lb")

    def test_scenario_overview_uses_canonical_result_metrics(self):
        state = self._preview_ready_state()

        payload = build_scenario_overview_payload(state)

        baseline = next(item for item in payload["scenarios"] if item["id"] == "baseline")
        requested = next(item for item in payload["scenarios"] if item["id"] == "requested_1")
        baseline_metrics = {item["label"]: item["value"] for item in baseline["metrics"]}
        metrics = {item["label"]: item["value"] for item in requested["metrics"]}
        self.assertEqual(baseline["reference_id"], "VDE #4998")
        self.assertNotEqual(baseline_metrics["Curb mass"], "â€”")
        self.assertNotEqual(baseline_metrics["VDE mass"], "â€”")
        self.assertNotEqual(baseline_metrics["VDE TOTAL"], "â€”")
        self.assertNotEqual(baseline_metrics["VDE NET"], "â€”")
        self.assertIn(" / ", baseline_metrics["ABC TOTAL"])
        self.assertNotEqual(metrics["Curb mass"], "â€”")
        self.assertNotEqual(metrics["VDE mass"], "â€”")
        self.assertTrue(metrics["VDE TOTAL"])
        self.assertNotEqual(metrics["ABC TOTAL"], "")

    def test_vde_cycle_comparison_consumes_canonical_by_phase_results(self):
        state = self._preview_ready_state()
        resolution = state["preview"]["result"]["resolution_result"]
        baseline = resolution["resolved_columns"]["baseline"]
        baseline["vde_total"] = dict(baseline.get("vde_total") or {"mj_per_km": 0.36})
        baseline["vde_net"] = dict(baseline.get("vde_net") or {"mj_per_km": 0.34})
        baseline["vde_total"]["by_phase"] = {"city": 0.41, "hwy": 0.31}
        baseline["vde_net"]["by_phase"] = {"city": 0.39, "hwy": 0.29}
        for result in resolution["proposal_results"]:
            result["vde_results"]["total"] = dict(result["vde_results"].get("total") or {"mj_per_km": 0.37})
            result["vde_results"]["net"] = dict(result["vde_results"].get("net") or {"mj_per_km": 0.35})
            result["vde_results"]["total"]["by_phase"] = {"city": 0.42, "hwy": 0.32}
            result["vde_results"]["net"]["by_phase"] = {"city": 0.40, "hwy": 0.30}

        payload = build_vde_cycle_comparison_payload(state, "Metric")

        self.assertTrue(payload["has_cycle_results"])
        labels = [item["label"] for item in payload["rows"]]
        self.assertIn("FTP-75 TOTAL", labels)
        self.assertIn("HWFET NET", labels)
        self.assertIn("Combined TOTAL", labels)
        ftp_total = next(item for item in payload["rows"] if item["label"] == "FTP-75 TOTAL")
        self.assertEqual(ftp_total["display_values"]["baseline"], "0.4100")

    def test_engineering_comparison_payload_shows_resolved_transmission_values(self):
        state = self._loaded_state()
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "transmission": "Absolute ABC"},
                {"proposal_id": "requested_2", "walk_from": "requested_1", "transmission": "Inherit"},
            ],
        )
        state["preview"] = {
            "status": "fresh",
            "fingerprint": "transmission-preview",
            "result": {
                "fingerprint": "transmission-preview",
                "validation_summary": {"overall_status": "OK"},
                "resolution_result": {
                    "baseline": {"effective": {}},
                    "resolved_columns": {
                        "baseline": {
                            "transmission_losses": {
                                "source": "baseline_snapshot",
                                "A_TRANS": 19.0,
                                "B_TRANS": 0.2,
                                "C_TRANS": 0.0,
                            }
                        }
                    },
                    "proposal_results": [
                        {
                            "proposal_id": "requested_1",
                            "source_column": "Requested #1",
                            "resolved_snapshot": {
                                "transmission_losses": {
                                    "source": "INHERITED",
                                    "status": "available",
                                    "abc": {"A": 12.0, "B": 0.2, "C": 0.0},
                                }
                            },
                            "abc_total": {"A": 120.0, "B": 0.02, "C": 0.01},
                            "abc_net": {"A": 108.0, "B": -0.18, "C": 0.01},
                            "vde_results": {"total": {"mj_per_km": 1.2}, "net": {"mj_per_km": 1.0}},
                            "status": "OK",
                        },
                        {
                            "proposal_id": "requested_2",
                            "source_column": "Requested #2",
                            "resolved_snapshot": {
                                "transmission_losses": {
                                    "source": "INHERITED",
                                    "status": "available",
                                    "abc": {"A": 12.0, "B": 0.2, "C": 0.0},
                                }
                            },
                            "abc_total": {"A": 121.0, "B": 0.03, "C": 0.01},
                            "abc_net": {"A": 109.0, "B": -0.17, "C": 0.01},
                            "vde_results": {"total": {"mj_per_km": 1.25}, "net": {"mj_per_km": 1.05}},
                            "status": "OK",
                        },
                    ],
                    "status": "OK",
                    "issues": [],
                },
            },
        }

        payload = build_engineering_comparison_payload(state, "Metric")

        transmission_group = next(group for group in payload["groups"] if group["title"] == "Transmission")
        row_a = next(row for row in transmission_group["rows"] if row["field_key"] == "trans_A_coef_N")
        row_b = next(row for row in transmission_group["rows"] if row["field_key"] == "trans_B_coef_Npkph")
        row_c = next(row for row in transmission_group["rows"] if row["field_key"] == "trans_C_coef_Npkph2")

        self.assertEqual(row_a["display_values"]["baseline"], "19")
        self.assertEqual(row_a["display_values"]["requested_1"], "12")
        self.assertEqual(row_a["display_values"]["requested_2"], "12")
        self.assertEqual(row_b["display_values"]["requested_1"], "0.2")
        self.assertEqual(row_c["display_values"]["requested_2"], "0")

    def test_engineering_comparison_payload_shows_resolved_component_snapshot_values(self):
        state = self._loaded_state()
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "brake": "Component DB lookup", "axle_hubs": "Component DB lookup"},
                {"proposal_id": "requested_2", "walk_from": "requested_1", "parasitic": "Component DB lookup"},
            ],
        )
        state["preview"] = {
            "status": "fresh",
            "fingerprint": "component-preview",
            "result": {
                "fingerprint": "component-preview",
                "validation_summary": {"overall_status": "OK"},
                "resolution_result": {
                    "baseline": {"effective": dict(state["baseline"]["effective"])},
                    "resolved_columns": {"baseline": {}},
                    "proposal_results": [
                        {
                            "proposal_id": "requested_1",
                            "source_column": "Requested #1",
                            "resolved_snapshot": {
                                "brake_A": 1.5,
                                "brake_B": 0.0005,
                                "brake_C": 0.0001,
                                "residual_torque_total_Nm": 12.0,
                                "axle_hub_A": 2.5,
                                "axle_hub_B": 0.0004,
                                "axle_hub_C": 0.0001,
                            },
                            "abc_total": {"A": 121.0, "B": 0.03, "C": 0.01},
                            "abc_net": {"A": 109.0, "B": -0.17, "C": 0.01},
                            "vde_results": {"total": {"mj_per_km": 1.25}, "net": {"mj_per_km": 1.05}},
                            "status": "OK",
                        },
                        {
                            "proposal_id": "requested_2",
                            "source_column": "Requested #2",
                            "resolved_snapshot": {
                                "parasitic_A": 3.5,
                                "parasitic_B": 0.0006,
                                "parasitic_C": 0.0002,
                            },
                            "abc_total": {"A": 122.0, "B": 0.031, "C": 0.011},
                            "abc_net": {"A": 110.0, "B": -0.169, "C": 0.011},
                            "vde_results": {"total": {"mj_per_km": 1.26}, "net": {"mj_per_km": 1.06}},
                            "status": "OK",
                        },
                    ],
                    "status": "OK",
                    "issues": [],
                },
            },
        }

        payload = build_engineering_comparison_payload(state, "Metric")

        brake_group = next(group for group in payload["groups"] if group["title"] == "Brake")
        axle_group = next(group for group in payload["groups"] if group["title"] == "Axle & Hubs")
        parasitic_group = next(group for group in payload["groups"] if group["title"] == "Parasitics")
        brake_a = next(row for row in brake_group["rows"] if row["field_key"] == "brake_A_coef_N")
        residual_torque = next(row for row in brake_group["rows"] if row["field_key"] == "residual_torque_total_Nm")
        axle_a = next(row for row in axle_group["rows"] if row["field_key"] == "axle_hub_A")
        parasitic_a = next(row for row in parasitic_group["rows"] if row["field_key"] == "parasitic_A_coef_N")

        self.assertEqual(brake_a["display_values"]["baseline"], "4")
        self.assertEqual(brake_a["display_values"]["requested_1"], "1.5")
        self.assertEqual(brake_a["raw_values"]["requested_1"], 1.5)
        self.assertEqual(residual_torque["display_values"]["requested_1"], "12.0")
        self.assertEqual(axle_a["display_values"]["requested_1"], "2.5")
        self.assertEqual(parasitic_a["display_values"]["requested_2"], "3.5")
        self.assertEqual(parasitic_a["raw_values"]["requested_2"], 3.5)

    def test_validation_and_audit_payloads_preserve_bundle_and_format_rows(self):
        state = self._preview_ready_state()
        original_bundle = deepcopy(state["preview"]["result"])

        validation_payload = build_validation_summary_payload(state, "Metric")
        audit_payload = build_preview_audit_payload(state, "Metric")

        self.assertTrue(validation_payload["has_bundle"])
        self.assertEqual(validation_payload["summary"]["proposal_count"], 2)
        self.assertEqual(validation_payload["scenario_sections"][0]["label"], "Requested #1")
        self.assertTrue(any(row["Domain"] == "Mass" for row in validation_payload["scenario_sections"][0]["domain_rows"]))
        self.assertTrue(audit_payload["has_bundle"])
        self.assertTrue(audit_payload["audit_rows"])
        self.assertEqual(state["preview"]["result"], original_bundle)

    def test_preview_status_payload_marks_save_allowed_when_validation_is_ok(self):
        state = self._loaded_state()
        state["preview"] = {
            "status": "fresh",
            "fingerprint": "fp",
            "result": {
                "validation_summary": {
                    "overall_status": "OK",
                    "review_count": 0,
                    "missing_count": 0,
                }
            },
        }
        payload = build_preview_status_payload(state)

        self.assertEqual(payload["validation_status"], "OK")
        self.assertEqual(payload["save_status"], "Eligible")

    def test_preview_status_payload_marks_review_without_hard_blocks_as_save_eligible(self):
        state = self._loaded_state()
        state["preview"] = {
            "status": "fresh",
            "fingerprint": "fp",
            "result": {
                "validation_summary": {
                    "overall_status": "Review",
                    "review_count": 2,
                    "missing_count": 0,
                    "invalid_count": 0,
                    "blocked_count": 0,
                    "warning_count": 2,
                }
            },
        }

        payload = build_preview_status_payload(state)

        self.assertEqual(payload["validation_status"], "Review")
        self.assertEqual(payload["save_status"], "Eligible")

    def test_roadload_analysis_payload_uses_resolved_preview_abc_and_preserves_net(self):
        state = self._loaded_state()
        state["preview"] = {
            "status": "fresh",
            "fingerprint": "fp-roadload",
            "result": {
                "validation_summary": {"overall_status": "Review", "review_count": 2, "missing_count": 0, "invalid_count": 0, "blocked_count": 0},
                "resolution_result": {
                    "baseline": {"effective": dict(_baseline_row())},
                    "resolved_columns": {
                        "baseline": {
                            "initial_abc_total": {"A": 120.0, "B": 0.020, "C": 0.0080},
                            "abc_net": {"A": 110.0, "B": 0.015, "C": 0.0072},
                        }
                    },
                    "proposal_results": [
                        {
                            "proposal_id": "requested_1",
                            "source_column": "Requested #1",
                            "abc_total": {"A": 115.0, "B": 0.018, "C": 0.0075},
                            "abc_net": {"A": 106.5, "B": 0.014, "C": 0.0067},
                            "resolved_snapshot": {
                                "transmission_losses": {"abc": {"A": 99.0, "B": 99.0, "C": 99.0}},
                            },
                        },
                        {
                            "proposal_id": "requested_2",
                            "source_column": "Requested #2",
                            "abc_total": {"A": 110.0, "B": 0.016, "C": 0.0070},
                            "abc_net": {"A": 101.5, "B": 0.012, "C": 0.0062},
                            "resolved_snapshot": {},
                        },
                    ],
                },
            },
        }
        state["proposals"] = [
            {"proposal_id": "requested_1", "display_index": 1, "walk_from": "baseline"},
            {"proposal_id": "requested_2", "display_index": 2, "walk_from": "requested_1"},
        ]

        payload = build_roadload_analysis_payload(state, "Metric", speed_max_kph=140)

        self.assertTrue(payload["has_bundle"])
        self.assertTrue(payload["is_fresh"])
        self.assertEqual(len(payload["series"]), 6)
        baseline_total = next(item for item in payload["series"] if item["legend_label"] == "Baseline TOTAL")
        baseline_net = next(item for item in payload["series"] if item["legend_label"] == "Baseline NET")
        req1_total = next(item for item in payload["series"] if item["legend_label"] == "Requested #1 TOTAL")
        req1_net = next(item for item in payload["series"] if item["legend_label"] == "Requested #1 NET")
        req2_total = next(item for item in payload["series"] if item["legend_label"] == "Requested #2 TOTAL")
        req2_net = next(item for item in payload["series"] if item["legend_label"] == "Requested #2 NET")

        self.assertAlmostEqual(baseline_total["checkpoint_force_map_N"][0], 120.0, places=9)
        self.assertAlmostEqual(baseline_total["checkpoint_force_map_N"][50], 141.0, places=9)
        self.assertAlmostEqual(baseline_total["checkpoint_force_map_N"][100], 202.0, places=9)
        self.assertAlmostEqual(baseline_total["checkpoint_force_map_N"][120], 237.6, places=9)
        self.assertAlmostEqual(baseline_net["checkpoint_force_map_N"][120], 215.48, places=9)
        self.assertAlmostEqual(req1_total["checkpoint_force_map_N"][100], 191.8, places=9)
        self.assertAlmostEqual(req1_net["checkpoint_force_map_N"][100], 174.9, places=9)
        self.assertAlmostEqual(req2_total["checkpoint_force_map_N"][120], 212.72, places=9)
        self.assertAlmostEqual(req2_net["checkpoint_force_map_N"][120], 192.22, places=9)
        self.assertAlmostEqual(req1_net["checkpoint_force_map_N"][0], 106.5, places=9)
        self.assertGreater(baseline_total["checkpoint_force_map_N"][100], req1_total["checkpoint_force_map_N"][100])
        self.assertGreater(req1_total["checkpoint_force_map_N"][100], req2_total["checkpoint_force_map_N"][100])
        self.assertLess(req1_net["checkpoint_force_map_N"][100], req1_total["checkpoint_force_map_N"][100])
        self.assertLess(req2_net["checkpoint_force_map_N"][100], req2_total["checkpoint_force_map_N"][100])

    def test_roadload_analysis_payload_requires_fresh_preview(self):
        state = self._loaded_state()
        state["preview"] = {
            "status": "stale",
            "fingerprint": "fp-stale-roadload",
            "result": {
                "validation_summary": {"overall_status": "Review"},
                "resolution_result": {
                    "baseline": {"effective": dict(_baseline_row())},
                    "resolved_columns": {"baseline": {"initial_abc_total": {"A": 120.0, "B": 0.02, "C": 0.008}}},
                    "proposal_results": [],
                },
            },
        }

        payload = build_roadload_analysis_payload(state, "Metric")

        self.assertTrue(payload["has_bundle"])
        self.assertFalse(payload["is_fresh"])
        self.assertEqual(payload["series"], [])
        self.assertIn("Preview is stale", payload["message"])

    def test_cycle_power_analysis_uses_canonical_physical_cycle_and_resolved_abc(self):
        state = self._preview_ready_state()

        payload = build_cycle_power_analysis_payload(state)

        self.assertTrue(payload["has_bundle"])
        self.assertTrue(payload["is_fresh"])
        self.assertIn("FTP-75", payload["cycle_options"])
        self.assertNotIn("Combined", payload["cycle_options"])
        self.assertTrue(payload["time_s"])
        self.assertTrue(payload["speed_kph"])
        self.assertTrue(payload["series"])
        self.assertFalse(payload["decomposition_available"])

    def test_preview_pending_when_not_generated(self):
        payload = build_v22_flow_status_payload(self._loaded_state())
        preview = next(item for item in payload["steps"] if item["key"] == "preview")

        self.assertEqual(preview["base_status"], "pending")
        self.assertIn("Preview not generated", preview["summary"])

    def test_preview_stale_when_stale(self):
        state = self._loaded_state()
        state["preview"]["status"] = "stale"

        payload = build_v22_flow_status_payload(state)
        preview = next(item for item in payload["steps"] if item["key"] == "preview")

        self.assertEqual(preview["base_status"], "stale")

    def test_preview_review_when_validation_has_review(self):
        state = self._loaded_state()
        state["preview"] = {
            "status": "fresh",
            "fingerprint": "fp",
            "result": {
                "validation_summary": {
                    "overall_status": "Review",
                    "review_count": 2,
                    "missing_count": 0,
                }
            },
        }

        payload = build_v22_flow_status_payload(state)
        preview = next(item for item in payload["steps"] if item["key"] == "preview")

        self.assertEqual(preview["base_status"], "review")
        self.assertIn("Validation Review", preview["summary"])

    def test_preview_complete_when_validation_ok(self):
        state = self._loaded_state()
        state["preview"] = {
            "status": "fresh",
            "fingerprint": "fp",
            "result": {
                "validation_summary": {
                    "overall_status": "OK",
                    "review_count": 0,
                    "missing_count": 0,
                }
            },
        }

        payload = build_v22_flow_status_payload(state)
        preview = next(item for item in payload["steps"] if item["key"] == "preview")

        self.assertEqual(preview["base_status"], "complete")
        self.assertEqual(payload["validation_status"], "complete")

    def test_active_section_only_changes_visual_active_status(self):
        state = self._loaded_state()
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Custom test mass"}])
        base_payload = build_v22_flow_status_payload(state)

        preview_state = deepcopy(state)
        preview_state["active_section"] = "preview"
        preview_payload = build_v22_flow_status_payload(preview_state)

        base_steps = {item["key"]: item for item in base_payload["steps"]}
        preview_steps = {item["key"]: item for item in preview_payload["steps"]}
        self.assertEqual(base_steps["baseline"]["status"], "active")
        self.assertEqual(preview_steps["preview"]["status"], "active")
        for key in base_steps:
            self.assertEqual(base_steps[key]["base_status"], preview_steps[key]["base_status"])

    def test_active_section_does_not_change_draft_fingerprint_revision_or_domain_state(self):
        state = self._loaded_state()
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Custom test mass"}])
        state = apply_v22_domain_inputs(state, "mass", {"requested_1": {"test_mass_kg": 1810.0}})
        state["preview"] = {
            "status": "fresh",
            "fingerprint": "seed",
            "result": build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state)),
        }
        draft_before = build_v22_canonical_request_draft(state)
        bundle_before = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))

        next_state = deepcopy(state)
        next_state["active_section"] = "preview"
        draft_after = build_v22_canonical_request_draft(next_state)
        bundle_after = build_v22_preview_bundle(next_state, baseline_context=compact_baseline_context(next_state))

        self.assertEqual(draft_before, draft_after)
        self.assertEqual(bundle_before["fingerprint"], bundle_after["fingerprint"])
        self.assertEqual(state["domain_input_state"], next_state["domain_input_state"])
        self.assertEqual(state["domain_input_state"]["mass"]["revision"], next_state["domain_input_state"]["mass"]["revision"])

    def test_branding_continues_to_follow_loaded_baseline_not_candidate(self):
        state = self._loaded_state()
        state["baseline"]["selected_vde_id"] = 4998
        state["baseline"]["printed"]["make"] = "VOLVO"
        state["baseline"]["effective"]["make"] = "AUDI"

        payload = build_v22_branding_payload(state)

        self.assertTrue(payload["loaded"])
        self.assertEqual(payload["make"], "AUDI")

    def test_baseline_candidate_status_pending_without_loaded_baseline(self):
        payload = build_baseline_candidate_status_payload(create_v22_state(), 4998, selected_label="VDE #4998 · AUDI Q6")

        self.assertFalse(payload["loaded"])
        self.assertEqual(payload["status"], "Pending")
        self.assertFalse(payload["candidate_differs"])

    def test_baseline_candidate_status_flags_candidate_when_different_from_loaded(self):
        state = self._loaded_state()

        payload = build_baseline_candidate_status_payload(state, 5001, selected_label="VDE #5001 · VOLVO XC40")

        self.assertTrue(payload["loaded"])
        self.assertTrue(payload["candidate_differs"])
        self.assertIn("continues using VDE #4998", payload["warning_message"])

    def test_baseline_candidate_status_equal_candidate_is_not_review(self):
        state = self._loaded_state()

        payload = build_baseline_candidate_status_payload(state, 4998, selected_label="VDE #4998 · AUDI Q6")

        self.assertFalse(payload["candidate_differs"])
        self.assertEqual(payload["loaded_baseline_id"], 4998)

    def test_loaded_baseline_summary_payload_is_empty_without_loaded_baseline(self):
        payload = build_loaded_baseline_summary_payload(create_v22_state(), "Metric")

        self.assertFalse(payload["loaded"])
        self.assertEqual(payload["groups"], [])

    def test_loaded_baseline_summary_payload_uses_metric_groups(self):
        state = self._loaded_state()
        state["baseline"]["effective"]["test_mass_basis"] = "EPA_INERTIA_CLASS"
        state["baseline"]["effective"]["vde_total_mj_per_km"] = 1.234
        state["baseline"]["effective"]["vde_net_mj_per_km"] = 1.111

        payload = build_loaded_baseline_summary_payload(state, "Metric")

        groups = {item["title"]: item["items"] for item in payload["groups"]}
        mass_group = {item["label"]: item["value"] for item in groups["Mass"]}
        roadload_group = {item["label"]: item["value"] for item in groups["Roadload"]}
        vde_group = {item["label"]: item["value"] for item in groups["VDE"]}
        self.assertEqual(mass_group["Curb mass"], "1600")
        self.assertEqual(mass_group["Test mass basis"], "EPA_INERTIA_CLASS")
        self.assertEqual(roadload_group["A"], "120")
        self.assertEqual(vde_group["VDE_TOTAL"], "1.2340")

    def test_loaded_baseline_summary_payload_uses_us_units_without_mutating_state(self):
        state = self._loaded_state()
        state["baseline"]["effective"]["vde_total_mj_per_km"] = 1.234
        original = deepcopy(state)

        payload = build_loaded_baseline_summary_payload(state, "US customary")

        groups = {item["title"]: item["items"] for item in payload["groups"]}
        mass_group = {item["label"]: item["value"] for item in groups["Mass"]}
        roadload_group = {item["label"]: item["value"] for item in groups["Roadload"]}
        self.assertEqual(mass_group["Curb mass"], "3527")
        self.assertEqual(roadload_group["A"], "26.98")
        self.assertEqual(state, original)

    def test_active_corrections_summary_empty_when_no_corrections(self):
        payload = build_active_corrections_summary(self._loaded_state(), "Metric")

        self.assertEqual(payload["count"], 0)
        self.assertEqual(payload["entries"], [])

    def test_active_corrections_summary_reports_one_correction_with_effective_value(self):
        state = self._loaded_state()
        state["baseline"]["corrections"] = {"cda_m2": 0.61}
        state["baseline"]["effective"]["cda_m2"] = 0.61

        payload = build_active_corrections_summary(state, "Metric")

        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["entries"][0]["domain"], "Aero")
        self.assertEqual(payload["entries"][0]["printed_value"], "0.62")
        self.assertEqual(payload["entries"][0]["effective_value"], "0.61")

    def test_active_corrections_summary_reports_multiple_and_preserves_zero(self):
        state = self._loaded_state()
        state["baseline"]["printed"]["A"] = 120.0
        state["baseline"]["corrections"] = {"inertia_class": 1928.0, "A": 0.0}
        state["baseline"]["effective"]["inertia_class"] = 1928.0
        state["baseline"]["effective"]["A"] = 0.0

        payload = build_active_corrections_summary(state, "US customary")

        self.assertEqual(payload["count"], 2)
        self.assertEqual(payload["entries"][0]["field_key"], "inertia_class")
        self.assertEqual(payload["entries"][0]["effective_value"], "4251")
        self.assertEqual(payload["entries"][1]["field_key"], "A")
        self.assertEqual(payload["entries"][1]["effective_value"], "0")

    def test_format_v22_issue_for_display_converts_structured_mass_issue_in_metric(self):
        issue = {
            "field_key": "target_curb_mass_kg",
            "actual": 1814.0,
            "min": 1700.0,
            "max": 1800.0,
            "status": "INVALID",
            "severity": "INVALID",
        }

        message = format_v22_issue_for_display(issue, "Metric")

        self.assertEqual(message, "Curb mass 1814 kg is outside the allowed interval (1700, 1800] kg.")
        self.assertEqual(issue["severity"], "INVALID")

    def test_format_v22_issue_for_display_converts_structured_mass_issue_in_us(self):
        issue = {
            "field_key": "target_curb_mass_kg",
            "actual": 1814.0,
            "min": 1700.0,
            "max": 1800.0,
            "status": "INVALID",
            "severity": "INVALID",
        }

        message = format_v22_issue_for_display(issue, "US customary")

        self.assertEqual(message, "Curb mass 3999 lb is outside the allowed interval (3748, 3968] lb.")
        self.assertEqual(issue["severity"], "INVALID")

    def test_format_v22_issue_for_display_converts_structured_pressure_issue_by_unit_system(self):
        issue = {
            "field_key": "front_pressure_psi",
            "actual": 19.0,
            "min": 20.0,
            "max": 60.0,
            "status": "INVALID",
        }

        metric_message = format_v22_issue_for_display(issue, "Metric")
        us_message = format_v22_issue_for_display(issue, "US customary")

        self.assertEqual(metric_message, "Front pressure 131 kPa is outside the allowed interval (138, 414] kPa.")
        self.assertEqual(us_message, "Front pressure 19 psi is outside the allowed interval (20, 60] psi.")

    def test_format_v22_issue_for_display_keeps_rrc_unit_constant(self):
        issue = {
            "field_key": "target_rrc_N_per_kN",
            "actual": 8.5,
            "expected": 8.2,
            "status": "REVIEW",
        }

        metric_message = format_v22_issue_for_display(issue, "Metric")
        us_message = format_v22_issue_for_display(issue, "US customary")

        self.assertEqual(metric_message, "Target final RRC 8.5 N/kN does not match expected 8.2 N/kN.")
        self.assertEqual(metric_message, us_message)

    def test_format_v22_issue_for_display_falls_back_to_original_text_without_mutation(self):
        issue = {
            "field_key": "target_curb_mass_kg",
            "message": "Curb mass is outside the canonical EPA TWC table.",
            "severity": "INVALID",
        }
        original = deepcopy(issue)

        message = format_v22_issue_for_display(issue, "US customary")

        self.assertEqual(message, "Curb mass is outside the canonical EPA TWC table.")
        self.assertEqual(issue, original)

    def test_format_v22_issue_for_display_preserves_severity_across_unit_toggle(self):
        issue = {
            "field_key": "target_curb_mass_kg",
            "actual": 1814.0,
            "min": 1700.0,
            "max": 1800.0,
            "severity": "INVALID",
            "status": "INVALID",
        }

        metric_message = format_v22_issue_for_display(issue, "Metric")
        us_message = format_v22_issue_for_display(issue, "US customary")

        self.assertNotEqual(metric_message, us_message)
        self.assertEqual(issue["severity"], "INVALID")
        self.assertEqual(issue["status"], "INVALID")


if __name__ == "__main__":
    unittest.main()
