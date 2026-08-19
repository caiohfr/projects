from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import src.vde_app.components.vde_request_compact as vde_request_compact
from src.vde_app.components.vde_request_compact_units import to_canonical_field_value
from src.vde_app.components.vde_request_domain_editors import (
    EPA_INERTIA_CLASSES,
    applicable_fields,
    field_schema,
    friendly_message,
    is_field_editable_with_inputs,
    proposal_application_status,
    required_fields,
    resolve_domain_display,
    sanitize_domain_inputs,
)
from src.vde_app.components.vde_request_lookup import active_domain_has_lookup_requests
from src.vde_app.components.vde_request_lookup import apply_lookup_to_inputs
from src.vde_app.components.vde_request_lookup import component_lookup_rows
from src.vde_app.components.vde_request_lookup import default_lookup_source
from src.vde_app.components.vde_request_lookup import lookup_empty_message
from src.vde_app.components.vde_request_lookup import lookup_source_options
from src.vde_app.components.vde_request_lookup import vde_lookup_rows
from src.vde_core.vde_request_adapter import build_v21_workbook_state_from_request_draft
from src.vde_core.vde_request_compact_adapter import build_v22_preview_bundle, compact_baseline_context
from src.vde_core.vde_request_compact_state import (
    apply_v22_baseline,
    apply_v22_corrections,
    apply_v22_domain_inputs,
    apply_v22_proposal_metadata,
    apply_v22_proposal_matrix,
    build_v22_canonical_request_draft,
    create_v22_state,
    normalize_v22_state,
    proposal_type_labels_by_domain,
    resolve_v22_metadata_contexts,
)
from src.vde_core.vde_request_preview import build_request_comparison_rows
from src.vde_core.vde_request_resolver import resolve_vde_request


def _baseline_row() -> dict:
    return {
        "id": 5038,
        "legislation": "EPA",
        "category": "MIDSIZE",
        "electrification": "HEV",
        "transmission_type": "AT",
        "drive_type": "AWD",
        "fuel_type": "Gasoline",
        "make": "AUDI",
        "model": "TEST08062026",
        "year": 2027,
        "cycle_name": "FTP75",
        "description": "Baseline request row",
        "mass_kg": 1600.0,
        "test_mass_kg": 1736.0,
        "inertia_class": 1750.0,
        "payload_kg": 250.0,
        "options_kg": 20.0,
        "gvwr_kg": 2100.0,
        "gcwr_kg": 3200.0,
        "weight_dist_fr_pct": 55.0,
        "cda_m2": 0.62,
        "frontal_area_m2": 2.2,
        "A": 120.0,
        "B": 0.02,
        "C": 0.01,
        "tire_db_id": 77,
        "tire_code": "MOCK-TIRE",
        "rrc_N_per_kN": 9.4,
        "front_pressure_psi": 35.0,
        "rear_pressure_psi": 35.0,
        "tire_load_mass_basis": "TEST_MASS",
        "tire_A_final": 48.0,
        "tire_B_final": 0.009,
        "tire_C_final": 0.001,
        "trans_A_coef_N": 6.0,
        "trans_B_coef_Npkph": 0.003,
        "trans_C_coef_Npkph2": 0.001,
        "brake_A_coef_N": 4.0,
        "brake_B_Npkph": 0.0008,
        "brake_C_coef_Npkph2": 0.0001,
        "axle_hub_A": 2.0,
        "axle_hub_B": 0.0007,
        "axle_hub_C": 0.0001,
        "parasitic_A_coef_N": 3.0,
        "parasitic_B_Npkph": 0.0005,
        "parasitic_C_coef_Npkph2": 0.0001,
    }


def _tire_browser_rows() -> list[dict]:
    return [
        {
            "lookup_id": "920101",
            "Tire ID": 920101,
            "Tire VDE ID": None,
            "Tire code": "QA-BASE",
            "RRC": 8.0,
            "Reference pressure": 38.0,
            "Test load": 610.0,
            "Mileage": 1000.0,
            "alpha": -0.30,
            "beta": 1.00,
            "a": 0.0405987767,
            "b": 0.00002000,
            "c": 0.0000000500,
            "Status": "measured",
            "Source": "qa_mock_seed",
            "Notes": "TIRE-MOCK-BASE",
        },
        {
            "lookup_id": "920102",
            "Tire ID": 920102,
            "Tire VDE ID": None,
            "Tire code": "QA-ECO",
            "RRC": 7.0,
            "Reference pressure": 35.0,
            "Test load": 610.0,
            "Mileage": 1000.0,
            "alpha": -0.32,
            "beta": 1.00,
            "a": 0.0388106091,
            "b": 0.00001800,
            "c": 0.0000000400,
            "Status": "measured",
            "Source": "qa_mock_seed",
            "Notes": "TIRE-MOCK-ECO",
        },
        {
            "lookup_id": "920103",
            "Tire ID": 920103,
            "Tire VDE ID": None,
            "Tire code": "QA-HIGH-RRC",
            "RRC": 10.0,
            "Reference pressure": 32.0,
            "Test load": 610.0,
            "Mileage": 1000.0,
            "alpha": -0.25,
            "beta": 1.02,
            "a": 0.0299397091,
            "b": 0.00002500,
            "c": 0.0000000700,
            "Status": "measured",
            "Source": "qa_mock_seed",
            "Notes": "TIRE-MOCK-HIGH",
        },
        {
            "lookup_id": "920104",
            "Tire ID": 920104,
            "Tire VDE ID": None,
            "Tire code": "QA-LOAD",
            "RRC": 8.8,
            "Reference pressure": 30.0,
            "Test load": 650.0,
            "Mileage": 1000.0,
            "alpha": -0.28,
            "beta": 1.05,
            "a": 0.0231280363,
            "b": 0.00002200,
            "c": 0.0000000600,
            "Status": "measured",
            "Source": "qa_mock_seed",
            "Notes": "TIRE-MOCK-LOAD",
        },
        {
            "lookup_id": "920106",
            "Tire ID": 920106,
            "Tire VDE ID": None,
            "Tire code": "QA-INCOMPLETE",
            "RRC": 9.0,
            "Reference pressure": None,
            "Test load": 610.0,
            "Mileage": 1000.0,
            "alpha": None,
            "beta": None,
            "a": None,
            "b": None,
            "c": None,
            "Status": "incomplete_reference_inputs",
            "Source": "qa_mock_seed",
            "Notes": "TIRE-MOCK-INCOMPLETE",
        },
        {
            "lookup_id": "920200",
            "Tire ID": 920200,
            "Tire VDE ID": None,
            "Tire code": "NON-QA",
            "RRC": 8.2,
            "Reference pressure": 36.0,
            "Test load": 590.0,
            "Mileage": 0.0,
            "alpha": None,
            "beta": None,
            "a": None,
            "b": None,
            "c": None,
            "Status": "measured",
            "Source": "seed",
            "Notes": "OTHER",
        },
    ]


def _domain_request(domain: str, *, raw_type: str, proposal_type: str, raw_values: dict, details_seed: dict | None = None) -> dict:
    return {
        "domain": domain,
        "raw_proposal_type": raw_type,
        "proposal_type": proposal_type,
        "selection_mode": raw_type,
        "raw_values": deepcopy(raw_values),
        "proposal_details_seed": deepcopy(details_seed or {}),
        "normalized_proposal": {
            "ok": True,
            "domain": domain,
            "template_label": raw_type,
            "proposal_type": proposal_type,
            "selection_mode": raw_type,
            "details": deepcopy(details_seed or {}),
            "has_internal_equivalent": True,
        },
        "has_internal_equivalent": True,
        "issues": [],
    }


def _manual_draft(state: dict, domain: str, *, raw_type: str, proposal_type: str, raw_values: dict, details_seed: dict | None = None) -> dict:
    baseline = dict(state.get("baseline") or {})
    return {
        "schema_version": "0.1",
        "template_version": "0.1",
        "source": {"source_type": "unit_test", "interface": "manual_draft"},
        "baseline_printed": deepcopy(dict(baseline.get("printed") or {})),
        "baseline_corrections": deepcopy(dict(baseline.get("corrections") or {})),
        "effective_baseline": deepcopy(dict(baseline.get("effective") or {})),
        "baseline_correction_disposition": baseline.get("correction_disposition") or "request_only",
        "issues": [],
        "proposals": [
            {
                "proposal_id": "requested_1",
                "display_index": 1,
                "source_column": "Requested #1",
                "name": "",
                "walk_from": {"kind": "baseline", "proposal_id": None, "source_column": "Baseline"},
                "issues": [],
                "domain_requests": {
                    domain: _domain_request(
                        domain,
                        raw_type=raw_type,
                        proposal_type=proposal_type,
                        raw_values=raw_values,
                        details_seed=details_seed,
                    )
                },
            }
        ],
    }


class VdeRequestV22BTests(unittest.TestCase):
    def _state(self) -> dict:
        return apply_v22_baseline(create_v22_state(), _baseline_row())

    def _domain_boundary_trace(self, domain: str, label: str, form_values: dict) -> dict:
        state = apply_v22_proposal_matrix(
            self._state(),
            [{"proposal_id": "requested_1", "walk_from": "baseline", domain: label}],
        )
        proposal = state["proposals"][0]
        domain_payload = dict(proposal["domains"][domain])
        payload = vde_request_compact.build_v22_domain_apply_payload(
            domain,
            list(state.get("proposals") or []),
            {"requested_1": deepcopy(form_values)},
        )
        sanitized = sanitize_domain_inputs(
            domain,
            domain_payload["proposal_type"],
            domain_payload["selection_mode"],
            payload["requested_1"],
        )
        applied = apply_v22_domain_inputs(state, domain, payload)
        draft = build_v22_canonical_request_draft(applied)
        domain_request = draft["proposals"][0]["domain_requests"][domain]
        bundle = build_v22_preview_bundle(applied, baseline_context=compact_baseline_context(applied))
        proposal_result = bundle["resolution_result"]["proposal_results"][0]
        return {
            "form": deepcopy(form_values),
            "payload": payload["requested_1"],
            "sanitized": sanitized,
            "applied": deepcopy(dict(applied["proposals"][0].get("inputs") or {}).get(domain) or {}),
            "resolver_raw": deepcopy(dict(domain_request.get("raw_values") or {})),
            "resolver_details": deepcopy(dict(domain_request.get("proposal_details_seed") or {})),
            "domain_result": deepcopy(dict(proposal_result["domain_results"][domain] or {})),
            "resolved_snapshot": deepcopy(dict(proposal_result.get("resolved_snapshot") or {})),
        }

    def _proposal_context(self, domain: str, state: dict, proposal_id: str) -> dict:
        _, contexts = vde_request_compact._domain_contexts(domain, state, state["baseline"]["effective"])
        return contexts[proposal_id]

    def _current_inputs_with_pending_widgets(
        self,
        domain: str,
        proposal_id: str,
        proposal_type: str,
        selection_mode: str,
        editable_inputs: dict,
        pending_widgets: dict,
    ) -> dict:
        fake_streamlit = SimpleNamespace(session_state=dict(pending_widgets or {}))

        with patch.object(vde_request_compact, "st", fake_streamlit):
            return vde_request_compact._current_widget_inputs(
                domain,
                proposal_id,
                proposal_type,
                selection_mode,
                deepcopy(dict(editable_inputs or {})),
            )

    def test_selection_mode_controls_applicable_fields_and_delta_zero(self):
        self.assertIn("delta_A", applicable_fields("transmission", "UPDATE_TRANS_DRAG_ABC", "Delta ABC"))
        self.assertNotIn("trans_A_coef_N", applicable_fields("transmission", "UPDATE_TRANS_DRAG_ABC", "Delta ABC"))
        self.assertIn("tire_improvement_pct", applicable_fields("tire", "TIRE_IMPROVEMENT_PCT", "Tire improvement %"))
        self.assertIn("target_rrc_N_per_kN", applicable_fields("tire", "TIRE_TARGET_RRC", "Target final RRC"))
        self.assertEqual(applicable_fields("brake", "BRAKE_NOT_USED", "Not used"), [])
        self.assertNotIn("target_mass_kg", applicable_fields("mass", "EPA_STATUS", "EPA status mass"))
        self.assertIn("mass_kg", applicable_fields("mass", "EPA_CURB_TO_TWC", "Curb mass -> EPA TWC"))
        self.assertNotIn("target_curb_mass_kg", applicable_fields("mass", "EPA_CURB_TO_TWC", "Curb mass -> EPA TWC"))
        self.assertEqual(required_fields("mass", "EPA_CURB_TO_TWC", "Curb mass -> EPA TWC"), ["mass_kg"])

        cleaned = sanitize_domain_inputs(
            "transmission",
            "UPDATE_TRANS_DRAG_ABC",
            "Delta ABC",
            {"delta_A": "", "delta_B": 1.25, "delta_C": None, "trans_A_coef_N": 9.0},
        )

        self.assertEqual(cleaned["delta_A"], 0.0)
        self.assertEqual(cleaned["delta_C"], 0.0)
        self.assertNotIn("trans_A_coef_N", cleaned)
        self.assertEqual(
            sanitize_domain_inputs(
                "mass",
                "EPA_CURB_TO_TWC",
                "Curb mass -> EPA TWC",
                {"mass_kg": 1222.0, "target_curb_mass_kg": 1500.0},
            ),
            {"mass_kg": 1222.0},
        )

    def test_initial_domain_input_state_is_not_configured(self):
        state = create_v22_state()

        self.assertEqual(state["domain_input_state"]["mass"]["status"], "not_configured")
        self.assertEqual(state["domain_input_state"]["mass"]["revision"], 0)

    def test_apply_domain_inputs_updates_only_target_domain_and_marks_preview_stale(self):
        state = self._state()
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "aero": "Absolute CdA", "transmission": "Delta ABC"},
                {"proposal_id": "requested_2", "walk_from": "requested_1", "mass": "Inherit"},
            ],
        )
        state["proposals"][0]["inputs"]["transmission"] = {"delta_A": 0.0, "delta_B": 0.0, "delta_C": 0.001}
        state["preview"] = {"status": "fresh", "fingerprint": "abc", "result": {"ok": True}}

        next_state = apply_v22_domain_inputs(state, "aero", {"requested_1": {"cda_m2": 0.7}, "requested_2": {}})

        self.assertEqual(next_state["proposals"][0]["inputs"]["aero"]["cda_m2"], 0.7)
        self.assertEqual(next_state["proposals"][0]["inputs"]["transmission"]["delta_A"], 0.0)
        self.assertEqual(next_state["preview"]["status"], "stale")
        self.assertIsNone(next_state["preview"]["result"])

    def test_apply_sets_applied_ready_for_epa_status_without_free_target(self):
        state = self._state()
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "EPA status mass"}])

        next_state = apply_v22_domain_inputs(state, "mass", {"requested_1": {}})
        domain_state = next_state["domain_input_state"]["mass"]
        proposal_status = domain_state["proposal_statuses"]["requested_1"]

        self.assertEqual(domain_state["status"], "applied_ready")
        self.assertEqual(proposal_status["status"], "applied_ready")
        self.assertGreater(domain_state["revision"], 0)
        self.assertIsNotNone(domain_state["last_applied_at"])

    def test_apply_sets_applied_incomplete_for_missing_gvwr(self):
        state = self._state()
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "GVWR loaded mass"}])

        next_state = apply_v22_domain_inputs(state, "mass", {"requested_1": {}})
        proposal_status = next_state["domain_input_state"]["mass"]["proposal_statuses"]["requested_1"]

        self.assertEqual(next_state["domain_input_state"]["mass"]["status"], "applied_incomplete")
        self.assertEqual(proposal_status["status"], "applied_incomplete")
        self.assertIn("Curb mass is required.", proposal_status["issues"])
        self.assertIn("Payload is required.", proposal_status["issues"])

    def test_matrix_change_marks_domain_stale_after_matrix_change(self):
        state = self._state()
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Custom test mass"}])
        state = apply_v22_domain_inputs(state, "mass", {"requested_1": {"test_mass_kg": 1800.0}})

        next_state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "GVWR loaded mass"}])

        self.assertEqual(next_state["domain_input_state"]["mass"]["status"], "stale_after_matrix_change")

    def test_twc_shift_plus_and_minus_resolve(self):
        baseline = self._state()["baseline"]["effective"]
        plus = resolve_domain_display(
            "mass",
            baseline,
            {"domains": {"mass": {"proposal_type": "MASS_TWC_SHIFT", "selection_mode": "TWC shift / target class"}}, "inputs": {"mass": {"shift_steps": "+1"}}},
        )
        minus = resolve_domain_display(
            "mass",
            baseline,
            {"domains": {"mass": {"proposal_type": "MASS_TWC_SHIFT", "selection_mode": "TWC shift / target class"}}, "inputs": {"mass": {"shift_steps": "-1"}}},
        )

        self.assertEqual(plus["target_mass_kg"], 1875.0)
        self.assertEqual(minus["target_mass_kg"], 1625.0)

    def test_twc_shift_accepts_signed_down_shift_after_applied_state_roundtrip(self):
        baseline = {"mass_kg": 2302.0, "test_mass_kg": 2495.0, "inertia_class": 2495.0}
        resolved = resolve_domain_display(
            "mass",
            baseline,
            {
                "domains": {"mass": {"proposal_type": "MASS_TWC_SHIFT", "selection_mode": "TWC shift / target class"}},
                "inputs": {"mass": {"shift_steps": -2.0}},
            },
        )

        self.assertEqual(resolved["target_mass_kg"], 2268.0)
        self.assertEqual(resolved["test_mass_kg"], 2325.0)

    def test_transmission_coastdown_share_display_uses_source_total_not_transmission_delta(self):
        baseline = {
            "A": 118.0,
            "B": 0.020,
            "C": 0.009,
            "source_abc_total_A": 118.0,
            "source_abc_total_B": 0.020,
            "source_abc_total_C": 0.009,
            "trans_A_coef_N": 8.5,
            "trans_B_coef_Npkph": 0.004,
            "trans_C_coef_Npkph2": 0.0008,
        }
        resolved = resolve_domain_display(
            "transmission",
            baseline,
            {
                "domains": {"transmission": {"proposal_type": "TRANS_LOSS_PCT", "selection_mode": "Transmission coastdown share"}},
                "inputs": {"transmission": {"transmission_loss_pct": 1.9}},
            },
        )

        self.assertAlmostEqual(resolved["trans_A_coef_N"], 2.242)
        self.assertAlmostEqual(resolved["trans_B_coef_Npkph"], 0.00038)
        self.assertAlmostEqual(resolved["trans_C_coef_Npkph2"], 0.000171)
        self.assertEqual(resolved["transmission_percent_basis"], "SOURCE_ABC_TOTAL")
        self.assertEqual(resolved["transmission_rule_version"], "COASTDOWN_SHARE_V1")

    def test_normalize_migrates_legacy_transmission_share_to_v1(self):
        state = self._state()
        state = apply_v22_proposal_matrix(
            state,
            [{"proposal_id": "requested_1", "walk_from": "baseline", "transmission": "Transmission coastdown share"}],
        )
        state["proposals"][0]["inputs"]["transmission"] = {"transmission_loss_pct": 1.9}

        normalized = normalize_v22_state(state)
        inputs = normalized["proposals"][0]["inputs"]["transmission"]

        self.assertEqual(inputs["transmission_loss_pct"], 1.9)
        self.assertEqual(inputs["transmission_application_mode"], "KEEP_TOTAL_FIXED")
        self.assertEqual(inputs["percent_basis"], "SOURCE_ABC_TOTAL")
        self.assertEqual(inputs["rule_version"], "COASTDOWN_SHARE_V1")

    def test_select_target_uses_epa_list_and_calculated_fields_stay_readonly(self):
        schema = field_schema("mass", "MASS_TWC_SHIFT", "TWC shift / target class", "target_mass_kg", inputs={"shift_steps": "Select target"})

        self.assertEqual(schema["widget"], "select")
        self.assertEqual(schema["options"], EPA_INERTIA_CLASSES)
        self.assertFalse(is_field_editable_with_inputs("mass", "GVWR", "payload_kg", "GVWR loaded mass", {"gvwr_kg": 2000.0}))
        self.assertFalse(is_field_editable_with_inputs("mass", "PERFORMANCE_CURB_MASS", "test_mass_kg", "Performance loaded mass", {"mass_kg": 1700.0}))

    def test_friendly_message_hides_raw_python_exception_text(self):
        self.assertNotIn("int() argument", friendly_message("int() argument must be a string"))

    def test_lookup_mapping_copies_only_relevant_domain_fields(self):
        tire_inputs = apply_lookup_to_inputs(
            "tire",
            "VDE DB",
            {
                "_raw": {
                    "id": 41,
                    "tire_db_id": 88,
                    "tire_code": "T-88",
                    "rrc_N_per_kN": 8.7,
                    "front_pressure_psi": 34.0,
                    "rear_pressure_psi": 35.0,
                    "cda_m2": 0.59,
                }
            },
        )
        trans_inputs = apply_lookup_to_inputs(
            "transmission",
            "Component DB",
            {
                "_raw": {
                    "component_id": "TRANS-MOCK-001",
                    "trans_A": 6.5,
                    "trans_B": 0.0031,
                    "trans_C": 0.0011,
                    "loss_pct": 2.5,
                    "notes": "mock row",
                }
            },
        )

        self.assertEqual(tire_inputs["tire_source_vde_id"], 41)
        self.assertNotIn("cda_m2", tire_inputs)
        self.assertEqual(trans_inputs["transmission_component_db_id"], "TRANS-MOCK-001")
        self.assertEqual(trans_inputs["transmission_loss_pct"], 2.5)

    def test_lookup_use_selected_row_only_populates_widgets_without_apply(self):
        state = self._state()
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "transmission": "Lookup from DB"}])
        state = apply_v22_domain_inputs(
            state,
            "transmission",
            {"requested_1": {"transmission_component_db_id": "TRANS-MOCK-001", "trans_A_coef_N": 8.5, "trans_B_coef_Npkph": 0.004, "trans_C_coef_Npkph2": 0.0008}},
        )
        state["preview"] = {"status": "fresh", "fingerprint": "fp", "result": {"ok": True}}
        before_inputs = deepcopy(state["proposals"][0]["inputs"]["transmission"])
        before_domain_state = deepcopy(state["domain_input_state"]["transmission"])
        before_preview = deepcopy(state["preview"])
        session_state = {}

        populated = vde_request_compact._apply_lookup_to_widget_state(
            session_state,
            "transmission",
            "requested_1",
            "TRANS_METADATA_ONLY",
            "Lookup from DB",
            {
                "transmission_component_db_id": "TRANS-MOCK-002",
                "trans_A_coef_N": 13.0,
                "trans_B_coef_Npkph": 0.0065,
                "trans_C_coef_Npkph2": 0.0015,
            },
            unit_system="Metric",
        )

        self.assertEqual(populated["transmission_component_db_id"], "TRANS-MOCK-002")
        self.assertEqual(session_state["v22_simple_transmission__requested_1__transmission_component_db_id"], "TRANS-MOCK-002")
        self.assertEqual(session_state["v22_simple_transmission__requested_1__trans_A_coef_N"], 13.0)
        self.assertEqual(state["proposals"][0]["inputs"]["transmission"], before_inputs)
        self.assertEqual(state["domain_input_state"]["transmission"], before_domain_state)
        self.assertEqual(state["preview"], before_preview)

    @patch("src.vde_app.components.vde_request_lookup.fetch_vde_all_rows")
    def test_vde_lookup_uses_only_active_domain_abc_and_replaces_component_identity(self, fetch_rows):
        fetch_rows.return_value = [
            {
                "id": 9901,
                "make": "QA",
                "model": "DOMAIN",
                "trans_A_coef_N": 8.5,
                "trans_B_coef_Npkph": 0.004,
                "trans_C_coef_Npkph2": 0.0008,
                "brake_A_coef_N": 2.1,
                "brake_B_coef_Npkph": 0.0002,
                "brake_C_coef_Npkph2": 0.00003,
                "axle_hub_A": None,
                "axle_hub_B": None,
                "axle_hub_C": None,
            }
        ]
        vde_lookup_rows.clear()
        brake_row = vde_lookup_rows("brake")[0]
        axle_row = vde_lookup_rows("axle_hubs")[0]
        self.assertEqual((brake_row["A"], brake_row["B"], brake_row["C"]), (2.1, 0.0002, 0.00003))
        self.assertEqual((axle_row["A"], axle_row["B"], axle_row["C"]), (None, None, None))

        inputs = apply_lookup_to_inputs("brake", "VDE DB", brake_row)
        self.assertEqual(inputs["brake_vde_db_id"], 9901)
        self.assertEqual(inputs["brake_component_db_id"], "")
        self.assertEqual(inputs["brake_A_coef_N"], 2.1)

        session_state = {}
        populated = vde_request_compact._apply_lookup_to_widget_state(
            session_state,
            "brake",
            "requested_1",
            "BRAKE_METADATA_ONLY",
            "Lookup from DB",
            inputs,
            unit_system="Metric",
        )
        self.assertEqual(populated["brake_component_db_id"], "")
        self.assertEqual(session_state["v22_simple_brake__requested_1__brake_component_db_id"], "")
        self.assertEqual(session_state["v22_simple_brake__requested_1__brake_vde_db_id"], "9901")

    def test_lookup_values_commit_only_after_apply_domain(self):
        state = self._state()
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "transmission": "Lookup from DB"}])
        session_state = {}
        vde_request_compact._apply_lookup_to_widget_state(
            session_state,
            "transmission",
            "requested_1",
            "TRANS_METADATA_ONLY",
            "Lookup from DB",
            {
                "transmission_component_db_id": "TRANS-MOCK-002",
                "trans_A_coef_N": 13.0,
                "trans_B_coef_Npkph": 0.0065,
                "trans_C_coef_Npkph2": 0.0015,
            },
            unit_system="Metric",
        )
        editable_inputs = {}
        fake_streamlit = SimpleNamespace(session_state=session_state)
        with patch.object(vde_request_compact, "st", fake_streamlit):
            editable_inputs = vde_request_compact._current_widget_inputs(
                "transmission",
                "requested_1",
                "TRANS_METADATA_ONLY",
                "Lookup from DB",
                editable_inputs,
            )
        payload = vde_request_compact.build_v22_domain_apply_payload("transmission", list(state.get("proposals") or []), {"requested_1": editable_inputs})
        applied = apply_v22_domain_inputs(state, "transmission", payload)

        self.assertEqual(applied["proposals"][0]["inputs"]["transmission"]["transmission_component_db_id"], "TRANS-MOCK-002")
        self.assertEqual(applied["domain_input_state"]["transmission"]["revision"], 1)

    def test_brake_lookup_apply_keeps_multi_proposal_rows_isolated(self):
        state = apply_v22_proposal_matrix(
            self._state(),
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "brake": "Lookup from DB"},
                {"proposal_id": "requested_2", "walk_from": "requested_1", "brake": "Lookup from DB"},
            ],
        )
        session_state = {}
        vde_request_compact._apply_lookup_to_widget_state(
            session_state,
            "brake",
            "requested_1",
            "BRAKE_METADATA_ONLY",
            "Lookup from DB",
            {
                "brake_component_db_id": "BRAKE-MOCK-LOW",
                "brake_A_coef_N": 1.5,
                "brake_B_Npkph": 0.0005,
                "brake_C_coef_Npkph2": 0.0001,
            },
            unit_system="Metric",
        )
        vde_request_compact._apply_lookup_to_widget_state(
            session_state,
            "brake",
            "requested_2",
            "BRAKE_METADATA_ONLY",
            "Lookup from DB",
            {
                "brake_component_db_id": "BRAKE-MOCK-HIGH",
                "brake_A_coef_N": 4.5,
                "brake_B_Npkph": 0.0015,
                "brake_C_coef_Npkph2": 0.0003,
            },
            unit_system="Metric",
        )

        req1_context = self._proposal_context("brake", state, "requested_1")
        req2_context = self._proposal_context("brake", state, "requested_2")
        fake_streamlit = SimpleNamespace(session_state=session_state)
        with patch.object(vde_request_compact, "st", fake_streamlit):
            payload = vde_request_compact.build_v22_domain_apply_payload(
                "brake",
                list(state.get("proposals") or []),
                {
                    "requested_1": vde_request_compact._current_widget_inputs(
                        "brake",
                        "requested_1",
                        req1_context["proposal_type"],
                        req1_context["selection_mode"],
                        req1_context["inputs"],
                    ),
                    "requested_2": vde_request_compact._current_widget_inputs(
                        "brake",
                        "requested_2",
                        req2_context["proposal_type"],
                        req2_context["selection_mode"],
                        req2_context["inputs"],
                    ),
                },
            )
        applied = apply_v22_domain_inputs(state, "brake", payload)
        _, contexts = vde_request_compact._domain_contexts("brake", applied, applied["baseline"]["effective"])
        req1_inputs = applied["proposals"][0]["inputs"]["brake"]
        req2_inputs = applied["proposals"][1]["inputs"]["brake"]
        req1_status = applied["domain_input_state"]["brake"]["proposal_statuses"]["requested_1"]
        req2_status = applied["domain_input_state"]["brake"]["proposal_statuses"]["requested_2"]

        self.assertEqual(req1_inputs["brake_component_db_id"], "BRAKE-MOCK-LOW")
        self.assertEqual(req1_inputs["brake_A_coef_N"], 1.5)
        self.assertEqual(req1_inputs["brake_B_Npkph"], 0.0005)
        self.assertEqual(req1_inputs["brake_C_coef_Npkph2"], 0.0001)
        self.assertEqual(req2_inputs["brake_component_db_id"], "BRAKE-MOCK-HIGH")
        self.assertEqual(req2_inputs["brake_A_coef_N"], 4.5)
        self.assertEqual(req2_inputs["brake_B_Npkph"], 0.0015)
        self.assertEqual(req2_inputs["brake_C_coef_Npkph2"], 0.0003)
        self.assertEqual(contexts["requested_1"]["resolved_display"]["brake_A_coef_N"], 1.5)
        self.assertEqual(contexts["requested_2"]["resolved_display"]["brake_A_coef_N"], 4.5)
        self.assertEqual(req1_status["status"], "applied_ready")
        self.assertEqual(req2_status["status"], "applied_ready")

    def test_brake_lookup_partial_payload_is_incomplete(self):
        state = apply_v22_proposal_matrix(
            self._state(),
            [{"proposal_id": "requested_1", "walk_from": "baseline", "brake": "Lookup from DB"}],
        )

        applied = apply_v22_domain_inputs(
            state,
            "brake",
            {"requested_1": {"brake_component_db_id": "BRAKE-MOCK-HIGH"}},
        )
        proposal_status = applied["domain_input_state"]["brake"]["proposal_statuses"]["requested_1"]

        self.assertEqual(proposal_status["status"], "applied_incomplete")
        self.assertIn("A is required.", proposal_status["issues"])

    def test_lookup_widget_fill_preserves_unrelated_unapplied_inputs(self):
        session_state = {"v22_simple_brake__requested_1__wheel_radius_m": 0.37}

        vde_request_compact._apply_lookup_to_widget_state(
            session_state,
            "brake",
            "requested_1",
            "BRAKE_DRAG_CHANGE",
            "Residual torque",
            {
                "brake_component_db_id": "BRAKE-MOCK-002",
                "residual_torque_front_Nm": 18.0,
                "residual_torque_rear_Nm": 16.0,
            },
            unit_system="Metric",
        )

        self.assertEqual(session_state["v22_simple_brake__requested_1__wheel_radius_m"], 0.37)
        self.assertEqual(session_state["v22_simple_brake__requested_1__residual_torque_front_Nm"], 18.0)
        self.assertEqual(session_state["v22_simple_brake__requested_1__residual_torque_rear_Nm"], 16.0)

    def test_tire_and_axle_lookup_widget_fill_regressions(self):
        tire_session_state = {}
        axle_session_state = {}

        tire_populated = vde_request_compact._apply_lookup_to_widget_state(
            tire_session_state,
            "tire",
            "requested_1",
            "TIRE_DB_LOOKUP",
            "Tire DB lookup",
            {"tire_db_id": 77, "tire_code": "TIRE-QA-010", "rrc_N_per_kN": 8.4, "front_pressure_psi": 35.0, "rear_pressure_psi": 35.0},
            unit_system="Metric",
            pressure_unit="psi",
        )
        axle_populated = vde_request_compact._apply_lookup_to_widget_state(
            axle_session_state,
            "axle_hubs",
            "requested_1",
            "AXLE_HUB_METADATA_ONLY",
            "Lookup from DB",
            {"axle_hubs_component_db_id": "AXLE-MOCK-001", "axle_hub_A": 2.5, "axle_hub_B": 0.0004, "axle_hub_C": 0.0001},
            unit_system="Metric",
        )

        self.assertEqual(tire_populated["tire_db_id"], 77)
        self.assertEqual(tire_session_state["v22_simple_tire__requested_1__front_pressure_psi"], 35.0)
        self.assertEqual(axle_populated["axle_hubs_component_db_id"], "AXLE-MOCK-001")
        self.assertEqual(axle_session_state["v22_simple_axle_hubs__requested_1__axle_hub_A"], 2.5)

    def test_lookup_unapplied_value_does_not_change_walk_from_until_apply(self):
        state = self._state()
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "transmission": "Lookup from DB"},
                {"proposal_id": "requested_2", "walk_from": "requested_1", "transmission": "Inherit"},
            ],
        )
        state = apply_v22_domain_inputs(
            state,
            "transmission",
            {"requested_1": {"transmission_component_db_id": "TRANS-MOCK-001", "trans_A_coef_N": 8.5, "trans_B_coef_Npkph": 0.004, "trans_C_coef_Npkph2": 0.0008}},
        )
        pending = {
            "v22_simple_transmission__requested_1__transmission_component_db_id": "TRANS-MOCK-002",
            "v22_simple_transmission__requested_1__trans_A_coef_N": 13.0,
            "v22_simple_transmission__requested_1__trans_B_coef_Npkph": 0.0065,
            "v22_simple_transmission__requested_1__trans_C_coef_Npkph2": 0.0015,
        }

        req1_context = self._proposal_context("transmission", state, "requested_1")
        before_context = self._proposal_context("transmission", state, "requested_2")
        pending_inputs = self._current_inputs_with_pending_widgets(
            "transmission",
            "requested_1",
            req1_context["proposal_type"],
            req1_context["selection_mode"],
            req1_context["inputs"],
            pending,
        )
        applied = apply_v22_domain_inputs(
            state,
            "transmission",
            {"requested_1": {"transmission_component_db_id": "TRANS-MOCK-002", "trans_A_coef_N": 13.0, "trans_B_coef_Npkph": 0.0065, "trans_C_coef_Npkph2": 0.0015}},
        )
        _, applied_contexts = vde_request_compact._domain_contexts("transmission", applied, applied["baseline"]["effective"])

        self.assertEqual(pending_inputs["trans_A_coef_N"], 13.0)
        self.assertEqual(before_context["resolved_display"]["trans_A_coef_N"], 8.5)
        self.assertEqual(applied_contexts["requested_2"]["resolved_display"]["trans_A_coef_N"], 13.0)

    def test_component_walk_from_uses_resolved_source_for_delta_abc(self):
        state = apply_v22_proposal_matrix(
            self._state(),
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "transmission": "Absolute ABC"},
                {"proposal_id": "requested_2", "walk_from": "requested_1", "transmission": "Delta ABC"},
            ],
        )
        state = apply_v22_domain_inputs(
            state,
            "transmission",
            {
                "requested_1": {"trans_A_coef_N": 4.0, "trans_B_coef_Npkph": 0.0, "trans_C_coef_Npkph2": 0.0},
                "requested_2": {"delta_A": 1.5, "delta_B": 0.0, "delta_C": 0.0},
            },
        )
        _, contexts = vde_request_compact._domain_contexts("transmission", state, state["baseline"]["effective"])

        bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        proposal_results = list(bundle["resolution_result"]["proposal_results"] or [])
        req1 = proposal_results[0]
        req2 = proposal_results[1]

        self.assertEqual(req1["resolved_snapshot"]["transmission_losses"]["abc"], {"A": 4.0, "B": 0.0, "C": 0.0})
        self.assertEqual(req2["domain_results"]["transmission"]["source"], "requested_1")
        self.assertEqual(contexts["requested_2"]["source_display"]["trans_A_coef_N"], 4.0)
        self.assertEqual(contexts["requested_2"]["resolved_display"]["trans_A_coef_N"], 5.5)
        self.assertEqual(
            req2["domain_results"]["transmission"]["resolved_values"],
            {
                "A": 1.5,
                "B": 0.0,
                "C": 0.0,
                "transmission_application_mode": "APPLY_DELTA_TO_TOTAL",
                "transmission_mode": "Vehicle change - TOTAL updated",
            },
        )

    def test_brake_delta_sanitize_keeps_canonical_delta_inputs_only(self):
        cleaned = sanitize_domain_inputs(
            "brake",
            "BRAKE_DRAG_CHANGE",
            "Delta ABC",
            {"delta_A": -1.0, "delta_B": 0.0, "delta_C": 0.0, "change_mode": "Absolute ABC", "method": "Residual torque"},
        )

        self.assertEqual(cleaned, {"delta_A": -1.0, "delta_B": 0.0, "delta_C": 0.0})

    def test_component_modes_adapt_to_legacy_details_once_in_resolver_adapter(self):
        state = apply_v22_proposal_matrix(
            self._state(),
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "transmission": "Delta ABC", "axle_hubs": "Absolute ABC"},
                {"proposal_id": "requested_2", "walk_from": "baseline", "brake": "Residual torque", "parasitic": "Delta ABC"},
            ],
        )
        state = apply_v22_domain_inputs(state, "transmission", {"requested_1": {"delta_A": 1.0, "delta_B": 0.0, "delta_C": 0.0}})
        state = apply_v22_domain_inputs(state, "brake", {"requested_2": {"residual_torque_total_Nm": 20.0, "wheel_radius_m": 0.4}})
        state = apply_v22_domain_inputs(state, "axle_hubs", {"requested_1": {"axle_hub_A": 4.0, "axle_hub_B": 0.0004, "axle_hub_C": 0.0001}})
        state = apply_v22_domain_inputs(state, "parasitic", {"requested_2": {"delta_A": 2.0, "delta_B": 0.0, "delta_C": 0.0}})

        draft = build_v22_canonical_request_draft(state)
        req1 = draft["proposals"][0]["domain_requests"]["transmission"]["proposal_details_seed"]
        req2 = draft["proposals"][1]["domain_requests"]["brake"]["proposal_details_seed"]
        req3 = draft["proposals"][0]["domain_requests"]["axle_hubs"]["proposal_details_seed"]
        req4 = draft["proposals"][1]["domain_requests"]["parasitic"]["proposal_details_seed"]

        self.assertNotIn("change_mode", req1)
        self.assertNotIn("method", req2)
        self.assertNotIn("change_mode", req2)
        self.assertNotIn("change_mode", req3)
        self.assertNotIn("change_mode", req4)

        workbook = build_v21_workbook_state_from_request_draft(draft)
        mapped_req1 = workbook["proposals"]["requested_1"]["transmission"]["details"]
        mapped_req2 = workbook["proposals"]["requested_2"]["brake"]["details"]
        mapped_req3 = workbook["proposals"]["requested_1"]["axle_hubs"]["details"]
        mapped_req4 = workbook["proposals"]["requested_2"]["parasitic"]["details"]

        self.assertEqual(mapped_req1["change_mode"], "Delta ABC")
        self.assertEqual(mapped_req2["method"], "Residual torque")
        self.assertNotIn("change_mode", mapped_req2)
        self.assertEqual(mapped_req3["change_mode"], "Absolute ABC")
        self.assertEqual(mapped_req4["change_mode"], "Delta ABC")

    def test_brake_mode_switch_drops_incompatible_absolute_fields(self):
        state = apply_v22_proposal_matrix(self._state(), [{"proposal_id": "requested_1", "walk_from": "baseline", "brake": "Absolute ABC"}])
        state = apply_v22_domain_inputs(state, "brake", {"requested_1": {"brake_A_coef_N": 6.0, "brake_B_Npkph": 0.001, "brake_C_coef_Npkph2": 0.0001}})
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "brake": "Delta ABC"}])
        payload = vde_request_compact.build_v22_domain_apply_payload(
            "brake",
            list(state.get("proposals") or []),
            {"requested_1": {"brake_A_coef_N": 9.0, "delta_A": 1.0, "delta_B": 0.0, "delta_C": 0.0}},
        )

        self.assertEqual(payload["requested_1"], {"delta_A": 1.0, "delta_B": 0.0, "delta_C": 0.0})

    def test_build_mass_apply_payload_uses_form_values_not_widget_cache(self):
        state = apply_v22_baseline(
            create_v22_state(),
            {
                **_baseline_row(),
                "mass_kg": 1500.0,
                "test_mass_kg": 1644.0,
                "inertia_class": 1644.0,
            },
        )
        state = apply_v22_proposal_matrix(
            state,
            [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Curb mass -> EPA TWC"}],
        )
        fake_streamlit = SimpleNamespace(session_state={"v22_simple_mass__requested_1__mass_kg": None})

        with patch.object(vde_request_compact, "st", fake_streamlit):
            payload = vde_request_compact.build_v22_domain_apply_payload(
                "mass",
                list(state.get("proposals") or []),
                {"requested_1": {"mass_kg": 1222.0}},
            )

        self.assertEqual(payload["requested_1"], {"mass_kg": 1222.0})

    def test_current_widget_inputs_materializes_curb_mass_widget_as_mass_kg(self):
        state = apply_v22_baseline(
            create_v22_state(),
            {
                **_baseline_row(),
                "mass_kg": 1500.0,
                "test_mass_kg": 1644.0,
                "inertia_class": 1644.0,
            },
        )
        state = apply_v22_proposal_matrix(
            state,
            [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Curb mass -> EPA TWC"}],
        )
        context = self._proposal_context("mass", state, "requested_1")
        current = self._current_inputs_with_pending_widgets(
            "mass",
            "requested_1",
            context["proposal_type"],
            context["selection_mode"],
            {},
            {"v22_simple_mass__requested_1__mass_kg": 1200.0},
        )

        self.assertEqual(current["mass_kg"], 1200.0)
        self.assertNotIn("target_curb_mass_kg", current)

    def test_request_input_boundaries_preserve_values_to_resolver(self):
        mass = self._domain_boundary_trace("mass", "Curb mass -> EPA TWC", {"mass_kg": 1340.0})
        self.assertEqual(mass["form"]["mass_kg"], 1340.0)
        self.assertEqual(mass["payload"]["mass_kg"], 1340.0)
        self.assertEqual(mass["sanitized"]["mass_kg"], 1340.0)
        self.assertEqual(mass["applied"]["mass_kg"], 1340.0)
        self.assertEqual(mass["resolver_details"]["mass_kg"], 1340.0)
        self.assertEqual(mass["domain_result"]["status"], "OK")
        self.assertNotIn("Curb mass is required", str(mass["domain_result"]))
        self.assertEqual(mass["resolved_snapshot"]["mass_kg"], 1340.0)

        aero = self._domain_boundary_trace("aero", "Absolute CdA", {"cda_m2": 0.67})
        self.assertEqual(aero["form"]["cda_m2"], 0.67)
        self.assertEqual(aero["payload"]["cda_m2"], 0.67)
        self.assertEqual(aero["sanitized"]["cda_m2"], 0.67)
        self.assertEqual(aero["applied"]["cda_m2"], 0.67)
        self.assertEqual(aero["resolver_details"]["new_CdA"], 0.67)
        self.assertEqual(aero["domain_result"]["status"], "OK")
        self.assertNotIn("New CdA is required", str(aero["domain_result"]))
        self.assertEqual(aero["resolved_snapshot"]["CdA"], 0.67)

        tire = self._domain_boundary_trace("tire", "Target final RRC", {"target_rrc_N_per_kN": 7.0})
        self.assertEqual(tire["form"]["target_rrc_N_per_kN"], 7.0)
        self.assertEqual(tire["payload"]["target_rrc_N_per_kN"], 7.0)
        self.assertEqual(tire["sanitized"]["target_rrc_N_per_kN"], 7.0)
        self.assertEqual(tire["applied"]["target_rrc_N_per_kN"], 7.0)
        self.assertEqual(tire["resolver_details"]["target_rrc_N_per_kN"], 7.0)
        self.assertEqual(tire["resolved_snapshot"]["rrc_N_per_kN"], 7.0)

        transmission = self._domain_boundary_trace(
            "transmission",
            "Absolute ABC",
            {"trans_A_coef_N": 4.0, "trans_B_coef_Npkph": 0.0, "trans_C_coef_Npkph2": 0.0},
        )
        self.assertEqual(transmission["payload"]["trans_B_coef_Npkph"], 0.0)
        self.assertEqual(transmission["sanitized"]["trans_C_coef_Npkph2"], 0.0)
        self.assertEqual(transmission["applied"]["trans_A_coef_N"], 4.0)
        self.assertEqual(
            transmission["resolver_details"],
            {
                "new_trans_A": 4.0,
                "new_trans_B": 0.0,
                "new_trans_C": 0.0,
                "transmission_application_mode": "APPLY_DELTA_TO_TOTAL",
            },
        )
        self.assertEqual(transmission["domain_result"]["status"], "OK")
        self.assertNotIn("A is required", str(transmission["domain_result"]))
        self.assertEqual(transmission["resolved_snapshot"]["transmission_losses"]["abc"], {"A": 4.0, "B": 0.0, "C": 0.0})

        brake = self._domain_boundary_trace(
            "brake",
            "Absolute ABC",
            {"brake_A_coef_N": 2.0, "brake_B_Npkph": 0.0, "brake_C_coef_Npkph2": 0.0},
        )
        self.assertEqual(brake["payload"]["brake_B_Npkph"], 0.0)
        self.assertEqual(brake["sanitized"]["brake_C_coef_Npkph2"], 0.0)
        self.assertEqual(brake["applied"]["brake_A_coef_N"], 2.0)
        self.assertEqual(brake["resolver_details"], {"brake_A": 2.0, "brake_B": 0.0, "brake_C": 0.0})
        self.assertEqual(brake["domain_result"]["status"], "OK")
        self.assertNotIn("A is required", str(brake["domain_result"]))
        self.assertEqual(brake["resolved_snapshot"]["brake_A"], 2.0)

    def test_mass_unapplied_curb_edit_does_not_change_resolved_display_or_walk_from(self):
        state = apply_v22_baseline(
            create_v22_state(),
            {
                **_baseline_row(),
                "mass_kg": 1848.0,
                "test_mass_kg": 1848.0,
                "inertia_class": 1928.0,
            },
        )
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Curb mass -> EPA TWC"},
                {"proposal_id": "requested_2", "walk_from": "requested_1", "mass": "TWC shift / target class"},
            ],
        )
        state = apply_v22_domain_inputs(
            state,
            "mass",
            {
                "requested_1": {"mass_kg": 1500.0},
                "requested_2": {"shift_steps": "+1", "curb_position": "Top"},
            },
        )
        pending = {"v22_simple_mass__requested_1__mass_kg": 1700.0}

        req1_context = self._proposal_context("mass", state, "requested_1")
        req2_context = self._proposal_context("mass", state, "requested_2")
        pending_inputs = self._current_inputs_with_pending_widgets(
            "mass",
            "requested_1",
            req1_context["proposal_type"],
            req1_context["selection_mode"],
            req1_context["inputs"],
            pending,
        )

        self.assertEqual(pending_inputs["mass_kg"], 1700.0)
        self.assertEqual(req1_context["resolved_display"]["inertia_class"], 1644.0)
        self.assertEqual(req2_context["resolved_display"]["inertia_class"], 1701.0)

        applied = apply_v22_domain_inputs(
            state,
            "mass",
            {
                "requested_1": {"mass_kg": 1700.0},
                "requested_2": {"shift_steps": "+1", "curb_position": "Top"},
            },
        )
        _, applied_contexts = vde_request_compact._domain_contexts("mass", applied, applied["baseline"]["effective"])

        self.assertNotEqual(applied_contexts["requested_1"]["resolved_display"]["inertia_class"], req1_context["resolved_display"]["inertia_class"])
        self.assertNotEqual(applied_contexts["requested_2"]["resolved_display"]["inertia_class"], req2_context["resolved_display"]["inertia_class"])

    def test_aero_unapplied_cda_edit_does_not_change_resolved_display_until_apply(self):
        state = apply_v22_proposal_matrix(self._state(), [{"proposal_id": "requested_1", "walk_from": "baseline", "aero": "Absolute CdA"}])
        state = apply_v22_domain_inputs(state, "aero", {"requested_1": {"cda_m2": 0.7}})
        context = self._proposal_context("aero", state, "requested_1")
        pending_inputs = self._current_inputs_with_pending_widgets(
            "aero",
            "requested_1",
            context["proposal_type"],
            context["selection_mode"],
            context["inputs"],
            {"v22_simple_aero__requested_1__cda_m2": 0.9},
        )

        self.assertEqual(pending_inputs["cda_m2"], 0.9)
        applied = apply_v22_domain_inputs(state, "aero", {"requested_1": {"cda_m2": 0.9}})
        _, applied_contexts = vde_request_compact._domain_contexts("aero", applied, applied["baseline"]["effective"])
        self.assertNotEqual(applied_contexts["requested_1"]["resolved_display"]["delta_CdA"], context["resolved_display"]["delta_CdA"])

    def test_tire_unapplied_rrc_edit_does_not_change_resolved_display_until_apply(self):
        state = apply_v22_proposal_matrix(self._state(), [{"proposal_id": "requested_1", "walk_from": "baseline", "tire": "Target final RRC"}])
        state = apply_v22_domain_inputs(state, "tire", {"requested_1": {"target_rrc_N_per_kN": 8.5}})
        context = self._proposal_context("tire", state, "requested_1")
        pending_inputs = self._current_inputs_with_pending_widgets(
            "tire",
            "requested_1",
            context["proposal_type"],
            context["selection_mode"],
            context["inputs"],
            {"v22_simple_tire__requested_1__target_rrc_N_per_kN": 7.5},
        )

        self.assertEqual(pending_inputs["target_rrc_N_per_kN"], 7.5)
        applied = apply_v22_domain_inputs(state, "tire", {"requested_1": {"target_rrc_N_per_kN": 7.5}})
        _, applied_contexts = vde_request_compact._domain_contexts("tire", applied, applied["baseline"]["effective"])
        self.assertNotEqual(applied_contexts["requested_1"]["resolved_display"]["rrc_N_per_kN"], context["resolved_display"]["rrc_N_per_kN"])

    def test_tire_front_fraction_defaulted_is_ready_not_incomplete(self):
        status = proposal_application_status(
            "tire",
            "TIRE_TARGET_RRC",
            "Target final RRC",
            {"target_rrc_N_per_kN": 9.0, "front_pressure_psi": 36.0, "rear_pressure_psi": 36.0},
            {
                "rrc_N_per_kN": 9.0,
                "tire_rule_status": "OK",
                "tire_rule_notes": "Front weight fraction defaulted to 50%.",
                "tire_rule_issues": [
                    {
                        "severity": "warning",
                        "code": "front_fraction_defaulted",
                        "message": "Front weight fraction defaulted to 50%.",
                    }
                ],
            },
        )

        self.assertEqual(status["status"], "applied_ready")
        self.assertEqual(status["message"], "Applied — Ready")
        self.assertIn("Front weight fraction defaulted to 50%.", status["issues"] )

    def test_tire_apply_counts_warning_only_resolution_as_ready(self):
        baseline = _baseline_row()
        baseline.pop("weight_dist_fr_pct", None)
        state = apply_v22_baseline(create_v22_state(), baseline)
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "tire": "Target final RRC"}])

        state = apply_v22_domain_inputs(
            state,
            "tire",
            {"requested_1": {"target_rrc_N_per_kN": 9.0, "front_pressure_psi": 36.0, "rear_pressure_psi": 36.0}},
        )
        domain_state = state["domain_input_state"]["tire"]
        proposal_status = domain_state["proposal_statuses"]["requested_1"]

        self.assertEqual(domain_state["status"], "applied_ready")
        self.assertEqual(proposal_status["status"], "applied_ready")
        self.assertIn("1 ready, 0 incomplete", str(domain_state.get("last_apply_message") or ""))
        self.assertIn("Front weight fraction defaulted to 50%.", proposal_status["issues"])

    def test_tire_apply_keeps_missing_target_rrc_incomplete(self):
        state = apply_v22_proposal_matrix(self._state(), [{"proposal_id": "requested_1", "walk_from": "baseline", "tire": "Target final RRC"}])

        state = apply_v22_domain_inputs(
            state,
            "tire",
            {"requested_1": {"front_pressure_psi": 36.0, "rear_pressure_psi": 36.0}},
        )
        domain_state = state["domain_input_state"]["tire"]
        proposal_status = domain_state["proposal_statuses"]["requested_1"]

        self.assertEqual(domain_state["status"], "applied_incomplete")
        self.assertEqual(proposal_status["status"], "applied_incomplete")
        self.assertIn("0 ready, 1 incomplete", str(domain_state.get("last_apply_message") or ""))

    def test_mass_weight_distribution_propagates_into_snapshot_and_tire_audit(self):
        state = apply_v22_proposal_matrix(
            self._state(),
            [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Curb mass -> EPA TWC", "tire": "Target final RRC"}],
        )
        state = apply_v22_domain_inputs(
            state,
            "mass",
            {"requested_1": {"mass_kg": 1500.0, "weight_dist_fr_pct": 60.0}},
        )
        state = apply_v22_domain_inputs(
            state,
            "tire",
            {"requested_1": {"target_rrc_N_per_kN": 9.0, "front_pressure_psi": 36.0, "rear_pressure_psi": 36.0}},
        )

        draft = build_v22_canonical_request_draft(state)
        bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        proposal = bundle["resolution_result"]["proposal_results"][0]
        mass_result = proposal["domain_results"]["mass"]
        tire_result = proposal["domain_results"]["tire"]

        self.assertEqual(draft["proposals"][0]["domain_requests"]["mass"]["raw_values"]["weight_dist_fr_pct"], 60.0)
        self.assertEqual(proposal["resolved_snapshot"]["weight_dist_fr_pct"], 60.0)
        self.assertEqual(mass_result["resolved_values"]["weight_dist_fr_pct"], 60.0)
        self.assertEqual(tire_result["resolved_values"]["front_weight_distribution_pct"], 60.0)
        self.assertEqual(tire_result["resolved_values"]["rear_weight_distribution_pct"], 40.0)
        self.assertEqual(tire_result["resolved_values"]["front_weight_fraction"], 0.6)


    def test_component_unapplied_delta_edit_does_not_change_resolved_display_until_apply(self):
        state = apply_v22_proposal_matrix(self._state(), [{"proposal_id": "requested_1", "walk_from": "baseline", "transmission": "Delta ABC"}])
        state = apply_v22_domain_inputs(state, "transmission", {"requested_1": {"delta_A": 1.0, "delta_B": 0.0, "delta_C": 0.0}})
        context = self._proposal_context("transmission", state, "requested_1")
        pending_inputs = self._current_inputs_with_pending_widgets(
            "transmission",
            "requested_1",
            context["proposal_type"],
            context["selection_mode"],
            context["inputs"],
            {"v22_simple_transmission__requested_1__delta_A": 9.0},
        )

        self.assertEqual(pending_inputs["delta_A"], 9.0)
        applied = apply_v22_domain_inputs(state, "transmission", {"requested_1": {"delta_A": 9.0, "delta_B": 0.0, "delta_C": 0.0}})
        _, applied_contexts = vde_request_compact._domain_contexts("transmission", applied, applied["baseline"]["effective"])
        self.assertNotEqual(applied_contexts["requested_1"]["resolved_display"]["new_trans_A"], context["resolved_display"]["new_trans_A"])

    def test_unapplied_widget_edit_does_not_mutate_domain_status_or_preview(self):
        state = apply_v22_proposal_matrix(self._state(), [{"proposal_id": "requested_1", "walk_from": "baseline", "aero": "Absolute CdA"}])
        state = apply_v22_domain_inputs(state, "aero", {"requested_1": {"cda_m2": 0.7}})
        state["preview"] = {"status": "fresh", "fingerprint": "fp", "result": {"ok": True}}
        before_domain_state = deepcopy(state["domain_input_state"]["aero"])
        before_preview = deepcopy(state["preview"])
        context = self._proposal_context("aero", state, "requested_1")

        pending_inputs = self._current_inputs_with_pending_widgets(
            "aero",
            "requested_1",
            context["proposal_type"],
            context["selection_mode"],
            context["inputs"],
            {"v22_simple_aero__requested_1__cda_m2": 0.9},
        )

        self.assertEqual(pending_inputs["cda_m2"], 0.9)
        self.assertEqual(state["domain_input_state"]["aero"], before_domain_state)
        self.assertEqual(state["preview"], before_preview)

    def test_display_readonly_rows_do_not_mutate_comparison_rows(self):
        rows = [
            {
                "Scenario": "Baseline",
                "Mass [kg]": "1814",
                "ABC_TOTAL A [N]": "120",
                "VDE_TOTAL [MJ/km]": "1.25",
            }
        ]
        original = deepcopy(rows)

        display_rows = vde_request_compact._display_readonly_rows(
            rows,
            vde_request_compact._COMPARISON_COLUMN_FIELDS,
            "US customary",
        )

        self.assertEqual(rows, original)
        self.assertEqual(display_rows[0]["Mass [lb]"], "3999")
        self.assertEqual(display_rows[0]["ABC_TOTAL A [lbf]"], "26.98")
        self.assertEqual(display_rows[0]["VDE_TOTAL [Wh/mi]"], "558.8")

    def test_preview_audit_rows_do_not_mutate_resolution_result(self):
        resolution_result = {
            "proposal_results": [
                {
                    "proposal_id": "requested_1",
                    "source_column": "Requested #1",
                    "walk_from": {"label": "Baseline"},
                    "domain_results": {
                        "mass": {
                            "status": "OK",
                            "source": "Baseline",
                            "requested_values": {"mass_kg": 1480.0},
                            "resolved_values": {"inertia_class": 1588.0, "target_twc_interval": "(1423, 1480] kg"},
                            "issues": [],
                        }
                    },
                }
            ]
        }
        original = deepcopy(resolution_result)

        audit_rows = vde_request_compact._preview_audit_rows(resolution_result, "US customary")

        self.assertEqual(resolution_result, original)
        self.assertIn("mass_kg=3263 lb", audit_rows[0]["Requested"])
        self.assertIn("inertia_class=3501 lb", audit_rows[0]["Resolved"])
        self.assertIn("target_twc_interval=(3137, 3263] lb", audit_rows[0]["Resolved"])

    def test_apply_payload_uses_editable_inputs_instead_of_stale_stored_inputs(self):
        state = self._state()
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Performance loaded mass"}])
        proposals = list(state.get("proposals") or [])
        payload = vde_request_compact.build_v22_domain_apply_payload(
            "mass",
            proposals,
            {"requested_1": {"mass_kg": 1600.0}, "requested_2": {}},
        )

        self.assertEqual(payload["requested_1"], {"mass_kg": 1600.0, "preset": "Curb +100 kg"})
        self.assertEqual(payload["requested_2"], {})

    def test_prime_widget_value_does_not_overwrite_existing_key(self):
        fake_streamlit = SimpleNamespace(session_state={"v22_simple_mass__requested_1__mass_kg": 1600.0})
        with patch.object(vde_request_compact, "st", fake_streamlit):
            vde_request_compact._prime_widget_value("v22_simple_mass__requested_1__mass_kg", 1423.0)

        self.assertEqual(fake_streamlit.session_state["v22_simple_mass__requested_1__mass_kg"], 1600.0)

    def test_prime_widget_value_initializes_missing_key(self):
        fake_streamlit = SimpleNamespace(session_state={})
        with patch.object(vde_request_compact, "st", fake_streamlit):
            vde_request_compact._prime_widget_value("v22_simple_mass__requested_1__mass_kg", 1423.0)

        self.assertEqual(fake_streamlit.session_state["v22_simple_mass__requested_1__mass_kg"], 1423.0)

    def test_clear_domain_widget_state_removes_only_incompatible_keys(self):
        session_state = {
            "v22_simple_mass__requested_1__mass_kg": 1600.0,
            "v22_simple_mass__requested_1__gvwr_kg": 2200.0,
            "v22_simple_aero__requested_1__cda_m2": 0.61,
            "v22_simple_mass__requested_2__mass_kg": 1700.0,
        }

        vde_request_compact.clear_v22_domain_widget_state(
            session_state,
            "mass",
            "requested_1",
            allowed_field_keys={"gvwr_kg"},
        )

        self.assertNotIn("v22_simple_mass__requested_1__mass_kg", session_state)
        self.assertIn("v22_simple_mass__requested_1__gvwr_kg", session_state)
        self.assertIn("v22_simple_aero__requested_1__cda_m2", session_state)
        self.assertIn("v22_simple_mass__requested_2__mass_kg", session_state)

    def test_matrix_change_clears_incompatible_widget_keys(self):
        state = self._state()
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Performance loaded mass"}])
        next_state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "GVWR loaded mass"}])
        session_state = {
            "v22_simple_mass__requested_1__mass_kg": 1600.0,
            "v22_simple_mass__requested_1__options_kg": 10.0,
            "v22_simple_mass__requested_1__payload_kg": 100.0,
            "v22_simple_mass__requested_1__gvwr_kg": 2200.0,
            "v22_simple_aero__requested_1__cda_m2": 0.62,
        }

        vde_request_compact._clear_widget_state_after_matrix_change(state, next_state, session_state)

        self.assertNotIn("v22_simple_mass__requested_1__options_kg", session_state)
        self.assertIn("v22_simple_mass__requested_1__mass_kg", session_state)
        self.assertIn("v22_simple_mass__requested_1__payload_kg", session_state)
        self.assertIn("v22_simple_mass__requested_1__gvwr_kg", session_state)
        self.assertIn("v22_simple_aero__requested_1__cda_m2", session_state)

    def test_clear_unit_sensitive_widget_state_keeps_non_physical_and_canonical_selects(self):
        session_state = {
            "v22_simple_mass__requested_1__mass_kg": 4000.0,
            "v22_simple_mass__requested_1__target_mass_kg": 1814.0,
            "v22_simple_mass__requested_1__shift_steps": "-1",
            "v22_simple_tire__requested_1__front_pressure_psi": 241.3,
            "v22_simple_brake__requested_1__wheel_radius_m": 0.35,
            "v22_correction__mass__mass_kg": 3999.2,
            "v22_correction__mass__test_mass_basis": "EPA_INERTIA_CLASS",
        }

        vde_request_compact.clear_v22_unit_sensitive_widget_state(session_state)

        self.assertNotIn("v22_simple_mass__requested_1__mass_kg", session_state)
        self.assertIn("v22_simple_tire__requested_1__front_pressure_psi", session_state)
        self.assertNotIn("v22_correction__mass__mass_kg", session_state)
        self.assertIn("v22_simple_mass__requested_1__target_mass_kg", session_state)
        self.assertIn("v22_simple_mass__requested_1__shift_steps", session_state)
        self.assertIn("v22_simple_brake__requested_1__wheel_radius_m", session_state)
        self.assertIn("v22_correction__mass__test_mass_basis", session_state)

    def test_widget_state_value_converts_display_mass_and_pressure_back_to_canonical(self):
        self.assertAlmostEqual(
            vde_request_compact._widget_state_value(4000.0, "number", "mass_kg", "US customary"),
            1814.36948,
            places=5,
        )
        self.assertAlmostEqual(
            vde_request_compact._widget_state_value(241.31650526095, "number", "front_pressure_psi", "Metric"),
            35.0,
            places=6,
        )
        self.assertAlmostEqual(
            vde_request_compact._widget_state_value(2.4131650526095, "number", "front_pressure_psi", "Metric", "bar"),
            35.0,
            places=6,
        )

    def test_metric_and_us_curb_inputs_produce_equivalent_canonical_and_preview(self):
        metric_state = apply_v22_proposal_matrix(
            self._state(),
            [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Curb mass → EPA TWC"}],
        )
        us_state = deepcopy(metric_state)

        metric_state = apply_v22_domain_inputs(metric_state, "mass", {"requested_1": {"mass_kg": 1814.36948}})
        us_state = apply_v22_domain_inputs(
            us_state,
            "mass",
            {"requested_1": {"mass_kg": to_canonical_field_value("mass_kg", 4000.0, "US customary")}},
        )

        metric_draft = build_v22_canonical_request_draft(metric_state)
        us_draft = build_v22_canonical_request_draft(us_state)
        metric_request = metric_draft["proposals"][0]["domain_requests"]["mass"]
        us_request = us_draft["proposals"][0]["domain_requests"]["mass"]
        self.assertAlmostEqual(metric_request["raw_values"]["mass_kg"], us_request["raw_values"]["mass_kg"], places=5)

        metric_bundle = build_v22_preview_bundle(metric_state, baseline_context=compact_baseline_context(metric_state))
        us_bundle = build_v22_preview_bundle(us_state, baseline_context=compact_baseline_context(us_state))
        metric_result = metric_bundle["resolution_result"]["proposal_results"][0]
        us_result = us_bundle["resolution_result"]["proposal_results"][0]

        self.assertAlmostEqual(metric_result["resolved_snapshot"]["mass_kg"], us_result["resolved_snapshot"]["mass_kg"], places=5)
        self.assertEqual(metric_result["resolved_snapshot"]["inertia_class"], us_result["resolved_snapshot"]["inertia_class"])
        self.assertAlmostEqual(metric_result["resolved_snapshot"]["test_mass_kg"], us_result["resolved_snapshot"]["test_mass_kg"], places=5)
        for key in ("A", "B", "C"):
            self.assertAlmostEqual(metric_result["abc_total"][key], us_result["abc_total"][key], places=9)
            self.assertAlmostEqual(metric_result["abc_net"][key], us_result["abc_net"][key], places=9)
        self.assertAlmostEqual(
            metric_result["vde_results"]["total"]["mj_per_km"],
            us_result["vde_results"]["total"]["mj_per_km"],
            places=9,
        )
        self.assertAlmostEqual(
            metric_result["vde_results"]["net"]["mj_per_km"],
            us_result["vde_results"]["net"]["mj_per_km"],
            places=9,
        )

    def test_mass_apply_reaches_state_and_canonical_draft(self):
        state = self._state()
        state["baseline"]["effective"]["mass_kg"] = 1423.0
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Performance loaded mass"}])
        payload = vde_request_compact.build_v22_domain_apply_payload(
            "mass",
            list(state.get("proposals") or []),
            {"requested_1": {"mass_kg": 1600.0}, "requested_2": {}},
        )
        state = apply_v22_domain_inputs(state, "mass", payload)
        draft = build_v22_canonical_request_draft(state)
        mass_request = draft["proposals"][0]["domain_requests"]["mass"]

        self.assertEqual(state["proposals"][0]["inputs"]["mass"]["mass_kg"], 1600.0)
        self.assertEqual(mass_request["raw_values"]["mass_kg"], 1600.0)
        self.assertEqual(mass_request["proposal_details_seed"]["curb_mass_kg"], 1600.0)

    def test_epa_status_uses_corrected_inertia_in_preview(self):
        state = apply_v22_corrections(self._state(), {"inertia_class": 1928.0})
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "EPA status mass"}])
        state = apply_v22_domain_inputs(state, "mass", {"requested_1": {}})
        bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        resolved_snapshot = bundle["resolution_result"]["proposal_results"][0]["resolved_snapshot"]

        self.assertEqual(resolved_snapshot["test_mass_kg"], 1736.0)
        self.assertEqual(dict(resolved_snapshot.get("resolved_mass_setup") or {}).get("resolved_mass_used_kg"), 1928.0)

    def test_epa_curb_to_twc_apply_reaches_preview_with_exact_curb(self):
        state = apply_v22_baseline(
            create_v22_state(),
            {
                **_baseline_row(),
                "mass_kg": 1423.0,
                "test_mass_kg": 1531.0,
                "inertia_class": 1531.0,
            },
        )
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Curb mass → EPA TWC"}])
        state = apply_v22_domain_inputs(state, "mass", {"requested_1": {"mass_kg": 1480.0}})
        draft = build_v22_canonical_request_draft(state)
        bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        resolved_snapshot = bundle["resolution_result"]["proposal_results"][0]["resolved_snapshot"]

        self.assertEqual(state["proposals"][0]["inputs"]["mass"]["mass_kg"], 1480.0)
        self.assertEqual(draft["proposals"][0]["domain_requests"]["mass"]["raw_values"]["mass_kg"], 1480.0)
        self.assertEqual(draft["proposals"][0]["domain_requests"]["mass"]["proposal_details_seed"]["mass_kg"], 1480.0)
        self.assertEqual(state["domain_input_state"]["mass"]["proposal_statuses"]["requested_1"]["status"], "applied_ready")
        self.assertEqual(resolved_snapshot["current_curb_mass_kg"], 1423.0)
        self.assertEqual(resolved_snapshot["target_curb_mass_kg"], 1480.0)
        self.assertEqual(resolved_snapshot["mass_kg"], 1480.0)
        self.assertEqual(resolved_snapshot["inertia_class"], 1588.0)
        self.assertEqual(resolved_snapshot["test_mass_kg"], 1616.0)
        self.assertEqual(resolved_snapshot["test_mass_basis"], "PHYSICAL_TEST_MASS")
        self.assertEqual(resolved_snapshot["target_twc_interval"], "(1423, 1480] kg")

    def test_epa_curb_to_twc_walk_from_propagates_exact_curb_and_resolved_twc(self):
        state = apply_v22_baseline(
            create_v22_state(),
            {
                **_baseline_row(),
                "mass_kg": 1423.0,
                "test_mass_kg": 1531.0,
                "inertia_class": 1531.0,
            },
        )
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Curb mass → EPA TWC"},
                {"proposal_id": "requested_2", "walk_from": "requested_1", "mass": "Inherit"},
            ],
        )
        state = apply_v22_domain_inputs(state, "mass", {"requested_1": {"mass_kg": 1480.0}})
        bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        req1 = bundle["resolution_result"]["proposal_results"][0]["resolved_snapshot"]
        req2 = bundle["resolution_result"]["proposal_results"][1]["resolved_snapshot"]

        self.assertEqual(req1["mass_kg"], 1480.0)
        self.assertEqual(req1["inertia_class"], 1588.0)
        self.assertEqual(req2["mass_kg"], 1480.0)
        self.assertEqual(req2["inertia_class"], 1588.0)
        self.assertEqual(req2["test_mass_kg"], 1616.0)

    def test_twc_shift_from_curb_to_twc_walk_from_uses_curb_position_without_changing_target_twc(self):
        resolved_by_position = {}
        for position in ("Top", "Mid", "Bottom"):
            state = apply_v22_baseline(
                create_v22_state(),
                {
                    **_baseline_row(),
                    "mass_kg": 1848.0,
                    "test_mass_kg": 1848.0,
                    "inertia_class": 1928.0,
                },
            )
            state = apply_v22_proposal_matrix(
                state,
                [
                    {"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Curb mass -> EPA TWC"},
                    {"proposal_id": "requested_2", "walk_from": "requested_1", "mass": "TWC shift / target class"},
                ],
            )
            state = apply_v22_domain_inputs(
                state,
                "mass",
                {
                    "requested_1": {"mass_kg": 1500.0},
                    "requested_2": {"shift_steps": "+1", "curb_position": position},
                },
            )
            bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
            resolved_by_position[position] = bundle["resolution_result"]["proposal_results"][1]["resolved_snapshot"]

        self.assertEqual(resolved_by_position["Top"]["inertia_class"], 1701.0)
        self.assertEqual(resolved_by_position["Mid"]["inertia_class"], 1701.0)
        self.assertEqual(resolved_by_position["Bottom"]["inertia_class"], 1701.0)
        self.assertEqual(resolved_by_position["Top"]["mass_kg"], 1593.0)
        self.assertEqual(resolved_by_position["Mid"]["mass_kg"], 1565.0)
        self.assertEqual(resolved_by_position["Bottom"]["mass_kg"], 1537.0)

    def test_ascii_arrow_curb_to_twc_label_still_updates_walk_from_shift_from_resolved_class(self):
        state = apply_v22_baseline(
            create_v22_state(),
            {
                **_baseline_row(),
                "mass_kg": 1848.0,
                "test_mass_kg": 1848.0,
                "inertia_class": 1928.0,
            },
        )
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Curb mass -> EPA TWC"},
                {"proposal_id": "requested_2", "walk_from": "requested_1", "mass": "TWC shift / target class"},
            ],
        )
        state = apply_v22_domain_inputs(
            state,
            "mass",
            {
                "requested_1": {"mass_kg": 1500.0},
                "requested_2": {"shift_steps": -1.0},
            },
        )
        draft = build_v22_canonical_request_draft(state)
        bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        req1 = bundle["resolution_result"]["proposal_results"][0]["resolved_snapshot"]
        req2 = bundle["resolution_result"]["proposal_results"][1]["resolved_snapshot"]

        self.assertEqual(state["proposals"][0]["domains"]["mass"]["proposal_type"], "EPA_CURB_TO_TWC")
        self.assertEqual(state["proposals"][0]["domains"]["mass"]["selection_mode"], "Curb mass → EPA TWC")
        self.assertEqual(req1["mass_kg"], 1500.0)
        self.assertEqual(req1["inertia_class"], 1644.0)
        self.assertEqual(draft["proposals"][1]["domain_requests"]["mass"]["proposal_details_seed"]["target_mass_kg"], 1588.0)
        self.assertEqual(req2["inertia_class"], 1588.0)
        self.assertEqual(req2["test_mass_kg"], 1616.0)

    def test_curb_to_twc_apply_propagates_mass_kg_into_walk_from_shift_chain(self):
        state = apply_v22_baseline(
            create_v22_state(),
            {
                **_baseline_row(),
                "mass_kg": 1500.0,
                "test_mass_kg": 1644.0,
                "inertia_class": 1644.0,
            },
        )
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Curb mass -> EPA TWC"},
                {"proposal_id": "requested_2", "walk_from": "requested_1", "mass": "TWC shift / target class"},
            ],
        )
        payload = vde_request_compact.build_v22_domain_apply_payload(
            "mass",
            list(state.get("proposals") or []),
            {
                "requested_1": {"mass_kg": 1222.0},
                "requested_2": {"shift_steps": "-1", "curb_position": "Top"},
            },
        )

        state = apply_v22_domain_inputs(state, "mass", payload)
        rerun_state = normalize_v22_state(state)
        draft = build_v22_canonical_request_draft(rerun_state)
        bundle = build_v22_preview_bundle(rerun_state, baseline_context=compact_baseline_context(rerun_state))
        req1 = bundle["resolution_result"]["proposal_results"][0]["resolved_snapshot"]
        req2 = bundle["resolution_result"]["proposal_results"][1]["resolved_snapshot"]
        req1_status = rerun_state["domain_input_state"]["mass"]["proposal_statuses"]["requested_1"]

        self.assertEqual(rerun_state["proposals"][0]["inputs"]["mass"], {"mass_kg": 1222.0})
        self.assertEqual(draft["proposals"][0]["domain_requests"]["mass"]["raw_values"], {"mass_kg": 1222.0})
        self.assertEqual(draft["proposals"][0]["domain_requests"]["mass"]["proposal_details_seed"]["mass_kg"], 1222.0)
        self.assertNotIn("target_curb_mass_kg", draft["proposals"][0]["domain_requests"]["mass"]["proposal_details_seed"])
        self.assertEqual(req1_status["status"], "applied_ready")
        self.assertNotIn("Curb mass is required", req1_status["message"])
        self.assertEqual(req1["mass_kg"], 1222.0)
        self.assertEqual(req1["inertia_class"], 1361.0)
        self.assertNotEqual(req1["inertia_class"], 1644.0)
        self.assertEqual(draft["proposals"][1]["domain_requests"]["mass"]["proposal_details_seed"]["target_mass_kg"], 1304.0)
        self.assertEqual(req2["target_mass_kg"], 1304.0)
        self.assertEqual(req2["inertia_class"], 1304.0)
        self.assertEqual(req2["mass_kg"], 1196.0)
        self.assertEqual(req2["test_mass_kg"], 1332.0)

    def test_gvwr_apply_reaches_state(self):
        state = self._state()
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "GVWR loaded mass"}])
        payload = vde_request_compact.build_v22_domain_apply_payload(
            "mass",
            list(state.get("proposals") or []),
            {"requested_1": {"gvwr_kg": 2200.0}, "requested_2": {}},
        )
        state = apply_v22_domain_inputs(state, "mass", payload)

        self.assertEqual(state["proposals"][0]["inputs"]["mass"]["gvwr_kg"], 2200.0)

    def test_gvwr_preview_uses_resolved_mass_and_payload(self):
        state = apply_v22_corrections(self._state(), {"mass_kg": 1500.0})
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "GVWR loaded mass"}])
        state = apply_v22_domain_inputs(state, "mass", {"requested_1": {"gvwr_kg": 2400.0}})
        bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        resolved_snapshot = bundle["resolution_result"]["proposal_results"][0]["resolved_snapshot"]
        mass_setup = dict(resolved_snapshot.get("resolved_mass_setup") or {})

        self.assertEqual(resolved_snapshot["vde_calculation_mass_kg"], 2400.0)
        self.assertEqual(mass_setup.get("resolved_mass_used_kg"), 2400.0)
        self.assertEqual(mass_setup.get("payload_kg"), 900.0)

    def test_aero_delta_apply_reaches_state_and_preview(self):
        state = self._state()
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "aero": "Delta CdA"}])
        payload = vde_request_compact.build_v22_domain_apply_payload(
            "aero",
            list(state.get("proposals") or []),
            {"requested_1": {"delta_CdA": -0.02}, "requested_2": {}},
        )
        state = apply_v22_domain_inputs(state, "aero", payload)
        bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))

        self.assertEqual(state["proposals"][0]["inputs"]["aero"]["delta_CdA"], -0.02)
        self.assertAlmostEqual(bundle["resolution_result"]["proposal_results"][0]["resolved_snapshot"]["CdA"], 0.60, places=6)

    def test_transmission_delta_apply_reaches_state_and_preview(self):
        state = self._state()
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "transmission": "Delta ABC"}])
        payload = vde_request_compact.build_v22_domain_apply_payload(
            "transmission",
            list(state.get("proposals") or []),
            {"requested_1": {"delta_A": -5.0, "delta_B": 0.0, "delta_C": 0.0}, "requested_2": {}},
        )
        state = apply_v22_domain_inputs(state, "transmission", payload)
        bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        comparison_rows = build_request_comparison_rows(bundle["resolution_result"])

        self.assertEqual(state["proposals"][0]["inputs"]["transmission"]["delta_A"], -5.0)
        self.assertEqual(float(comparison_rows[1]["ABC_TOTAL A [N]"]), 115.0)

    def test_twc_shift_apply_reaches_state_and_preview(self):
        state = self._state()
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "TWC shift / target class"}])
        payload = vde_request_compact.build_v22_domain_apply_payload(
            "mass",
            list(state.get("proposals") or []),
            {"requested_1": {"shift_steps": "+1"}, "requested_2": {}},
        )
        state = apply_v22_domain_inputs(state, "mass", payload)
        draft = build_v22_canonical_request_draft(state)
        bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        resolved_snapshot = bundle["resolution_result"]["proposal_results"][0]["resolved_snapshot"]

        self.assertEqual(state["proposals"][0]["inputs"]["mass"]["shift_steps"], 1.0)
        self.assertEqual(state["proposals"][0]["inputs"]["mass"]["curb_position"], "Top")
        self.assertEqual(draft["proposals"][0]["domain_requests"]["mass"]["proposal_details_seed"]["target_mass_kg"], 1875.0)
        self.assertEqual(resolved_snapshot["inertia_class"], 1875.0)
        self.assertEqual(resolved_snapshot["mass_intention"], "EPA_PLUS_1_TWC")

    def test_twc_shift_minus_two_stays_down_through_walk_from_and_preview(self):
        state = apply_v22_baseline(
            create_v22_state(),
            {
                **_baseline_row(),
                "mass_kg": 2000.0,
                "test_mass_kg": 2100.0,
                "inertia_class": 2495.0,
            },
        )
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "mass": "EPA status mass"},
                {"proposal_id": "requested_2", "walk_from": "requested_1", "mass": "TWC shift / target class"},
            ],
        )
        payload = vde_request_compact.build_v22_domain_apply_payload(
            "mass",
            list(state.get("proposals") or []),
            {"requested_1": {}, "requested_2": {"shift_steps": "-2"}},
        )
        state = apply_v22_domain_inputs(state, "mass", payload)
        draft = build_v22_canonical_request_draft(state)
        bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        req2_snapshot = bundle["resolution_result"]["proposal_results"][1]["resolved_snapshot"]

        self.assertEqual(state["proposals"][1]["inputs"]["mass"]["shift_steps"], -2.0)
        self.assertNotIn("target_side", state["proposals"][1]["inputs"]["mass"])
        self.assertNotIn("target_side", draft["proposals"][1]["domain_requests"]["mass"]["proposal_details_seed"])
        self.assertEqual(draft["proposals"][1]["domain_requests"]["mass"]["proposal_details_seed"]["target_mass_kg"], 2268.0)
        self.assertEqual(req2_snapshot["inertia_class"], 2268.0)
        self.assertEqual(req2_snapshot["test_mass_kg"], 2325.0)

    def test_twc_shift_uses_next_epa_class_when_reference_is_exact(self):
        state = apply_v22_corrections(self._state(), {"inertia_class": 1928.0})
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "TWC shift / target class"}])
        state = apply_v22_domain_inputs(state, "mass", {"requested_1": {"shift_steps": "+1"}})
        bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        resolved_snapshot = bundle["resolution_result"]["proposal_results"][0]["resolved_snapshot"]

        self.assertEqual(resolved_snapshot["test_mass_kg"], 2098.0)
        self.assertEqual(dict(resolved_snapshot.get("resolved_mass_setup") or {}).get("resolved_mass_used_kg"), 2041.0)

    def test_twc_shift_down_updates_resolved_curb_mass_with_target_class(self):
        state = apply_v22_baseline(
            create_v22_state(),
            {
                **_baseline_row(),
                "mass_kg": 2416.0,
                "test_mass_kg": 2495.0,
                "inertia_class": 2495.0,
            },
        )
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "TWC shift / target class"}])
        state = apply_v22_domain_inputs(state, "mass", {"requested_1": {"shift_steps": "-2"}})

        bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        resolved_snapshot = bundle["resolution_result"]["proposal_results"][0]["resolved_snapshot"]

        self.assertEqual(resolved_snapshot["inertia_class"], 2268.0)
        self.assertEqual(resolved_snapshot["test_mass_kg"], 2325.0)
        self.assertEqual(resolved_snapshot["mass_kg"], 2189.0)

    def test_performance_curb_presets_drive_preview_mass(self):
        state = self._state()
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Performance loaded mass"}])
        state = apply_v22_domain_inputs(state, "mass", {"requested_1": {"mass_kg": 1500.0, "preset": "Curb +100 kg"}})
        bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        resolved_snapshot = bundle["resolution_result"]["proposal_results"][0]["resolved_snapshot"]
        self.assertEqual(dict(resolved_snapshot.get("resolved_mass_setup") or {}).get("resolved_mass_used_kg"), 1600.0)

        state = apply_v22_domain_inputs(state, "mass", {"requested_1": {"mass_kg": 1500.0, "preset": "Curb +300 lb"}})
        bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        resolved_snapshot = bundle["resolution_result"]["proposal_results"][0]["resolved_snapshot"]
        self.assertAlmostEqual(dict(resolved_snapshot.get("resolved_mass_setup") or {}).get("resolved_mass_used_kg"), 1636.1, places=6)

    def test_tire_target_rrc_preview_stays_technical_ok(self):
        state = self._state()
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "tire": "Target final RRC"}])
        state = apply_v22_domain_inputs(state, "tire", {"requested_1": {"target_rrc_N_per_kN": 8.0, "front_pressure_psi": 32.0, "rear_pressure_psi": 32.0}})
        bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        tire_result = bundle["resolution_result"]["proposal_results"][0]["domain_results"]["tire"]

        self.assertEqual(tire_result["status"], "OK")
        self.assertAlmostEqual(bundle["resolution_result"]["proposal_results"][0]["resolved_snapshot"]["rrc_N_per_kN"], 8.0)
        self.assertNotIn("technical formatting error", str(tire_result))

    def test_tire_canonical_draft_stays_target_and_improvement_not_legacy_aliases(self):
        state = self._state()
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "tire": "Target final RRC"},
                {"proposal_id": "requested_2", "walk_from": "baseline", "tire": "Tire improvement %"},
            ],
        )
        state = apply_v22_domain_inputs(
            state,
            "tire",
            {
                "requested_1": {"target_rrc_N_per_kN": 8.0, "front_pressure_psi": 32.0, "rear_pressure_psi": 32.0},
                "requested_2": {"tire_improvement_pct": 6.0, "front_pressure_psi": 34.0, "rear_pressure_psi": 34.0},
            },
        )

        draft = build_v22_canonical_request_draft(state)
        req1 = draft["proposals"][0]["domain_requests"]["tire"]["proposal_details_seed"]
        req2 = draft["proposals"][1]["domain_requests"]["tire"]["proposal_details_seed"]

        self.assertEqual(req1["target_rrc_N_per_kN"], 8.0)
        self.assertEqual(req1["front_pressure_psi"], 32.0)
        self.assertNotIn("delta_RRC_optional", req1)
        self.assertNotIn("psi_front", req1)
        self.assertEqual(req2["tire_improvement_pct"], 6.0)
        self.assertNotIn("improvement_pct", req2)

        workbook = build_v21_workbook_state_from_request_draft(draft)
        mapped_req1 = workbook["proposals"]["requested_1"]["tire"]["details"]
        mapped_req2 = workbook["proposals"]["requested_2"]["tire"]["details"]

        self.assertEqual(mapped_req1["delta_RRC_optional"], 8.0)
        self.assertEqual(mapped_req1["psi_front"], 32.0)
        self.assertEqual(mapped_req2["improvement_pct"], 6.0)

    def test_tire_mode_switch_drops_incompatible_target_and_improvement_fields(self):
        state = apply_v22_proposal_matrix(self._state(), [{"proposal_id": "requested_1", "walk_from": "baseline", "tire": "Target final RRC"}])
        state = apply_v22_domain_inputs(state, "tire", {"requested_1": {"target_rrc_N_per_kN": 8.5, "front_pressure_psi": 32.0, "rear_pressure_psi": 32.0}})
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "tire": "Tire improvement %"}])
        improvement_payload = vde_request_compact.build_v22_domain_apply_payload(
            "tire",
            list(state.get("proposals") or []),
            {"requested_1": {"target_rrc_N_per_kN": 8.9, "tire_improvement_pct": 5.0, "front_pressure_psi": 34.0, "rear_pressure_psi": 34.0}},
        )

        self.assertEqual(improvement_payload["requested_1"], {"tire_improvement_pct": 5.0, "front_pressure_psi": 34.0, "rear_pressure_psi": 34.0})

        state = apply_v22_domain_inputs(state, "tire", improvement_payload)
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "tire": "Target final RRC"}])
        target_payload = vde_request_compact.build_v22_domain_apply_payload(
            "tire",
            list(state.get("proposals") or []),
            {"requested_1": {"target_rrc_N_per_kN": 7.9, "tire_improvement_pct": 4.0, "front_pressure_psi": 33.0, "rear_pressure_psi": 33.0}},
        )

        self.assertEqual(target_payload["requested_1"], {"target_rrc_N_per_kN": 7.9, "front_pressure_psi": 33.0, "rear_pressure_psi": 33.0})

    def test_tire_type_labels_expose_target_rrc_and_hide_smerf(self):
        labels = proposal_type_labels_by_domain()["tire"]

        self.assertIn("Target final RRC", labels)
        self.assertIn("Tire DB lookup", labels)
        self.assertNotIn("SMERF", " | ".join(labels))

    def test_aero_labels_hide_not_used_from_active_ui(self):
        labels = proposal_type_labels_by_domain()["aero"]

        self.assertEqual(labels, ["Inherit", "Absolute CdA", "Delta CdA"])

    def test_tire_not_used_does_not_enable_lookup_flow(self):
        state = apply_v22_proposal_matrix(self._state(), [{"proposal_id": "requested_1", "walk_from": "baseline", "tire": "Not used"}])

        self.assertFalse(active_domain_has_lookup_requests(state, "tire"))

    def test_tire_lookup_source_labels_and_default_are_tire_specific(self):
        self.assertEqual(lookup_source_options("tire"), ["Tire Database", "Existing VDE"])
        self.assertEqual(default_lookup_source("tire"), "Tire Database")
        self.assertEqual(lookup_source_options("transmission"), ["Component DB", "VDE DB"])

    def test_tire_lookup_empty_messages_are_explicit(self):
        self.assertEqual(lookup_empty_message("tire", "Tire Database", "", []), "No Tire Database records available.")
        self.assertEqual(lookup_empty_message("tire", "Tire Database", "QA-ECO", []), "No matching Tire Database records.")
        self.assertEqual(lookup_empty_message("transmission", "Component DB", "MISS", []), "No matching records.")

    def test_tire_component_lookup_rows_expose_sae_columns(self):
        sample_rows = [
            {
                "id": 920101,
                "tire_test_code": "QA-BASE",
                "rr_n_per_kn": 8.0,
                "test_pressure_value": 38.0,
                "pressure_unit": "psi",
                "test_load_value": 610.0,
                "load_unit": "kg",
                "test_mileage_km": 1000.0,
                "sae_alpha": -0.30,
                "sae_beta": 1.00,
                "sae_a": 0.0405987767,
                "sae_b": 0.00002000,
                "sae_c": 0.0000000500,
                "notes": "Synthetic QA data",
                "rr_source": "qa_mock_seed",
            }
        ]

        component_lookup_rows.clear()
        with patch("src.vde_app.components.vde_request_lookup.search_tire_roadload", return_value=sample_rows):
            rows = component_lookup_rows("tire", "", limit=None)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Tire code"], "QA-BASE")
        self.assertAlmostEqual(float(rows[0]["alpha"]), -0.30, places=9)
        self.assertAlmostEqual(float(rows[0]["beta"]), 1.00, places=9)
        self.assertAlmostEqual(float(rows[0]["a"]), 0.0405987767, places=10)
        self.assertAlmostEqual(float(rows[0]["b"]), 0.00002000, places=10)
        self.assertAlmostEqual(float(rows[0]["c"]), 0.0000000500, places=10)

    def test_tire_browser_filters_code_rrc_pressure_load_and_combination(self):
        rows = _tire_browser_rows()

        by_code = vde_request_compact._tire_browser_filter_rows(rows, code_query="QA-ECO")
        by_rrc = vde_request_compact._tire_browser_filter_rows(rows, rrc_min=6.9, rrc_max=7.1)
        by_pressure = vde_request_compact._tire_browser_filter_rows(rows, pressure_min=34.5, pressure_max=35.5)
        by_load = vde_request_compact._tire_browser_filter_rows(rows, load_min=649.0, load_max=651.0)
        combined = vde_request_compact._tire_browser_filter_rows(
            rows,
            code_query="QA",
            rrc_min=8.7,
            rrc_max=8.9,
            pressure_min=29.5,
            pressure_max=30.5,
            load_min=649.0,
            load_max=651.0,
        )

        self.assertEqual([row["Tire code"] for row in by_code], ["QA-ECO"])
        self.assertEqual([row["Tire code"] for row in by_rrc], ["QA-ECO"])
        self.assertEqual([row["Tire code"] for row in by_pressure], ["QA-ECO"])
        self.assertEqual([row["Tire code"] for row in by_load], ["QA-LOAD"])
        self.assertEqual([row["Tire code"] for row in combined], ["QA-LOAD"])

    def test_tire_browser_keeps_incomplete_and_supports_mileage_filters(self):
        rows = _tire_browser_rows()

        incomplete = vde_request_compact._tire_browser_filter_rows(rows, code_query="QA-INCOMPLETE")
        zero_mileage = vde_request_compact._tire_browser_filter_rows(rows, mileage_mode="0 km")
        positive_mileage = vde_request_compact._tire_browser_filter_rows(rows, mileage_mode=">0 km")

        self.assertEqual([row["Tire code"] for row in incomplete], ["QA-INCOMPLETE"])
        self.assertEqual([row["Tire code"] for row in zero_mileage], ["NON-QA"])
        self.assertNotIn("NON-QA", [row["Tire code"] for row in positive_mileage])

    def test_brake_mode_switch_to_not_used_clears_incompatible_inputs(self):
        state = apply_v22_proposal_matrix(self._state(), [{"proposal_id": "requested_1", "walk_from": "baseline", "brake": "Delta ABC"}])
        state = apply_v22_domain_inputs(state, "brake", {"requested_1": {"delta_A": 1.0, "delta_B": 0.0, "delta_C": 0.0}})
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "brake": "Not used"}])

        payload = vde_request_compact.build_v22_domain_apply_payload(
            "brake",
            list(state.get("proposals") or []),
            {"requested_1": {"delta_A": 2.0, "brake_A_coef_N": 5.0}},
        )

        self.assertEqual(payload["requested_1"], {})

    def test_metadata_inherits_from_baseline_and_previous_proposal(self):
        state = self._state()
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline"},
                {"proposal_id": "requested_2", "walk_from": "requested_1"},
            ],
        )
        state = apply_v22_proposal_metadata(
            state,
            "requested_1",
            {
                "name": "Scenario 1",
                "make": "BMW",
                "model": "iX",
                "model_year": 2027,
                "description": "Scenario description",
                "legislation": "WLTP",
                "cycle_name": "WLTC",
            },
        )
        state = apply_v22_proposal_metadata(state, "requested_2", {"model_year": 2028})

        contexts = resolve_v22_metadata_contexts(state)
        req1 = contexts["requested_1"]["effective_metadata"]
        req2 = contexts["requested_2"]["effective_metadata"]

        self.assertEqual(req1["make"], "BMW")
        self.assertEqual(req1["model"], "iX")
        self.assertEqual(req1["model_year"], 2027)
        self.assertEqual(req1["legislation"], "EPA")
        self.assertEqual(req1["cycle_name"], "FTP75")
        self.assertEqual(req2["make"], "BMW")
        self.assertEqual(req2["model"], "iX")
        self.assertEqual(req2["model_year"], 2028)
        self.assertEqual(req2["legislation"], "EPA")
        self.assertEqual(req2["cycle_name"], "FTP75")

    def test_preview_bundle_applies_effective_metadata_chain(self):
        state = self._state()
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline"},
                {"proposal_id": "requested_2", "walk_from": "requested_1"},
            ],
        )
        state = apply_v22_proposal_metadata(state, "requested_1", {"make": "BMW", "model": "iX", "model_year": 2027})
        state = apply_v22_proposal_metadata(state, "requested_2", {"model_year": 2028})

        bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        req1 = bundle["resolution_result"]["proposal_results"][0]["resolved_snapshot"]
        req2 = bundle["resolution_result"]["proposal_results"][1]["resolved_snapshot"]

        self.assertEqual(req1["make"], "BMW")
        self.assertEqual(req1["model"], "iX")
        self.assertEqual(req1["year"], 2027)
        self.assertEqual(req2["make"], "BMW")
        self.assertEqual(req2["model"], "iX")
        self.assertEqual(req2["year"], 2028)
        self.assertEqual(req2["legislation"], "EPA")
        self.assertEqual(req2["cycle_name"], "FTP75")

    def test_second_apply_replaces_previous_value_and_preview_stays_stale(self):
        state = self._state()
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Performance loaded mass"}])
        state = apply_v22_domain_inputs(state, "mass", {"requested_1": {"mass_kg": 1600.0}})
        state = apply_v22_domain_inputs(state, "mass", {"requested_1": {"mass_kg": 1650.0}})

        self.assertEqual(state["proposals"][0]["inputs"]["mass"]["mass_kg"], 1650.0)
        self.assertEqual(state["preview"]["status"], "stale")

    def test_multiple_proposals_keep_same_field_isolated(self):
        state = self._state()
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Performance loaded mass"},
                {"proposal_id": "requested_2", "walk_from": "requested_1", "mass": "Performance loaded mass"},
            ],
        )
        payload = vde_request_compact.build_v22_domain_apply_payload(
            "mass",
            list(state.get("proposals") or []),
            {"requested_1": {"mass_kg": 1600.0}, "requested_2": {"mass_kg": 1700.0}},
        )
        state = apply_v22_domain_inputs(state, "mass", payload)

        self.assertEqual(state["proposals"][0]["inputs"]["mass"]["mass_kg"], 1600.0)
        self.assertEqual(state["proposals"][1]["inputs"]["mass"]["mass_kg"], 1700.0)

    def test_preview_bundle_builds_fingerprint_and_comparison_rows(self):
        state = self._state()
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Custom test mass"},
                {"proposal_id": "requested_2", "walk_from": "requested_1", "mass": "Inherit"},
            ],
        )
        state = apply_v22_domain_inputs(state, "mass", {"requested_1": {"test_mass_kg": 1800.0}})

        bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))

        self.assertTrue(bundle["fingerprint"])
        self.assertGreaterEqual(len(bundle["comparison_rows"]), 2)
        self.assertEqual(bundle["validation_summary"]["proposal_count"], 2)

    def test_preview_fingerprint_is_stable_for_same_state_and_ignores_active_section(self):
        state = self._state()
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Custom test mass"}])
        state = apply_v22_domain_inputs(state, "mass", {"requested_1": {"test_mass_kg": 1810.0}})

        baseline_context = compact_baseline_context(state)
        draft_before = build_v22_canonical_request_draft(state)
        bundle_before = build_v22_preview_bundle(state, baseline_context=baseline_context)
        next_state = deepcopy(state)
        next_state["active_section"] = "preview"
        draft_after = build_v22_canonical_request_draft(next_state)
        bundle_after = build_v22_preview_bundle(next_state, baseline_context=compact_baseline_context(next_state))

        self.assertEqual(draft_before, draft_after)
        self.assertEqual(bundle_before["fingerprint"], bundle_after["fingerprint"])

    def test_walk_from_uses_effectively_resolved_origin_snapshot(self):
        state = self._state()
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Custom test mass"},
                {"proposal_id": "requested_2", "walk_from": "requested_1", "mass": "Inherit"},
            ],
        )
        state = apply_v22_domain_inputs(state, "mass", {"requested_1": {"test_mass_kg": 1810.0}})

        bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        req1 = bundle["resolution_result"]["proposal_results"][0]["resolved_snapshot"]
        req2 = bundle["resolution_result"]["proposal_results"][1]["resolved_snapshot"]

        self.assertEqual(req1["test_mass_kg"], 1810.0)
        self.assertEqual(req2["test_mass_kg"], 1810.0)

    def test_aero_walk_from_uses_applied_previous_absolute_cda(self):
        state = self._state()
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "aero": "Absolute CdA"},
                {"proposal_id": "requested_2", "walk_from": "requested_1", "aero": "Absolute CdA"},
            ],
        )
        state = apply_v22_domain_inputs(
            state,
            "aero",
            {"requested_1": {"cda_m2": 0.67}, "requested_2": {"cda_m2": 0.69}},
        )

        bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        req1 = bundle["resolution_result"]["proposal_results"][0]
        req2 = bundle["resolution_result"]["proposal_results"][1]

        self.assertEqual(req1["resolved_snapshot"]["CdA"], 0.67)
        self.assertEqual(req2["domain_results"]["aero"]["source"], "requested_1")
        self.assertEqual(req2["domain_results"]["aero"]["resolved_values"]["CdA"], 0.69)
        self.assertEqual(req2["resolved_snapshot"]["CdA"], 0.69)

    def test_v22_preview_matches_manual_canonical_draft_for_mass(self):
        state = self._state()
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Custom test mass"}])
        state = apply_v22_domain_inputs(state, "mass", {"requested_1": {"test_mass_kg": 1810.0}})

        v22_bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        manual_state = build_v21_workbook_state_from_request_draft(
            _manual_draft(state, "mass", raw_type="Custom test mass", proposal_type="CUSTOM_MASS", raw_values={"test_mass_kg": 1810.0}),
            {"rows": []},
        )
        manual_result = resolve_vde_request(manual_state, baseline_context=compact_baseline_context(state))

        self.assertEqual(
            v22_bundle["resolution_result"]["proposal_results"][0]["resolved_snapshot"]["test_mass_kg"],
            manual_result["proposal_results"][0]["resolved_snapshot"]["test_mass_kg"],
        )

    def test_v22_preview_matches_manual_canonical_draft_for_aero(self):
        state = self._state()
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "aero": "Absolute CdA"}])
        state = apply_v22_domain_inputs(state, "aero", {"requested_1": {"cda_m2": 0.7}})

        v22_bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        manual_state = build_v21_workbook_state_from_request_draft(
            _manual_draft(state, "aero", raw_type="Absolute CdA", proposal_type="AERO_ABSOLUTE_CDA", raw_values={"cda_m2": 0.7}),
            {"rows": []},
        )
        manual_result = resolve_vde_request(manual_state, baseline_context=compact_baseline_context(state))

        self.assertEqual(
            v22_bundle["resolution_result"]["proposal_results"][0]["resolved_snapshot"]["CdA"],
            manual_result["proposal_results"][0]["resolved_snapshot"]["CdA"],
        )

    def test_v22_preview_matches_manual_canonical_draft_for_transmission(self):
        state = self._state()
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "transmission": "Delta ABC"}])
        state = apply_v22_domain_inputs(state, "transmission", {"requested_1": {"delta_A": 1.5, "delta_B": 0.0, "delta_C": 0.0}})

        v22_bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        manual_draft = _manual_draft(
            state,
            "transmission",
            raw_type="Delta ABC",
            proposal_type="UPDATE_TRANS_DRAG_ABC",
            raw_values={"trans_A_coef_N": 1.5, "trans_B_coef_Npkph": 0.0, "trans_C_coef_Npkph2": 0.0},
            details_seed={"change_mode": "Delta ABC"},
        )
        manual_state = build_v21_workbook_state_from_request_draft(manual_draft, {"rows": []})
        manual_result = resolve_vde_request(manual_state, baseline_context=compact_baseline_context(state))
        v22_rows = build_request_comparison_rows(v22_bundle["resolution_result"])
        manual_rows = build_request_comparison_rows(manual_result)

        self.assertEqual(
            v22_bundle["resolution_result"]["proposal_results"][0]["domain_results"]["transmission"]["status"],
            manual_result["proposal_results"][0]["domain_results"]["transmission"]["status"],
        )
        self.assertEqual(v22_rows[1]["ABC_TOTAL A [N]"], manual_rows[1]["ABC_TOTAL A [N]"])

    def test_brake_delta_apply_reaches_state_and_preview(self):
        state = self._state()
        state = apply_v22_proposal_matrix(state, [{"proposal_id": "requested_1", "walk_from": "baseline", "brake": "Delta ABC"}])
        payload = vde_request_compact.build_v22_domain_apply_payload(
            "brake",
            list(state.get("proposals") or []),
            {"requested_1": {"delta_A": -1.0, "delta_B": 0.0, "delta_C": 0.0}, "requested_2": {}},
        )
        state = apply_v22_domain_inputs(state, "brake", payload)
        bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        comparison_rows = build_request_comparison_rows(bundle["resolution_result"])

        self.assertEqual(state["proposals"][0]["inputs"]["brake"]["delta_A"], -1.0)
        self.assertEqual(
            bundle["resolution_result"]["proposal_results"][0]["domain_results"]["brake"]["status"],
            "OK",
        )
        self.assertEqual(float(comparison_rows[1]["ABC_TOTAL A [N]"]), 119.0)


if __name__ == "__main__":
    unittest.main()
