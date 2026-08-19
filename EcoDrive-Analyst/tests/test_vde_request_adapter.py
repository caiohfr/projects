from __future__ import annotations

from copy import deepcopy
import json
import unittest

from src.vde_core.vde_request_adapter import (
    apply_v21_request_import,
    build_v21_request_import_summary,
    build_v21_workbook_state_from_request_draft,
)


def _base_draft() -> dict:
    return {
        "schema_version": "0.1",
        "template_version": "0.1",
        "source": {
            "filename": "ppe_request.xlsx",
            "imported_at": "2026-07-10T12:00:00Z",
        },
        "baseline_printed": {
            "selected_baseline_vde_id": 5038,
            "mass_kg": 1800,
            "cda_m2": 0.62,
        },
        "baseline_corrections": {
            "mass_kg": 1850,
            "cda_m2": 0.64,
            "year": 2027,
            "notes": "Imported baseline request",
        },
        "effective_baseline": {
            "selected_baseline_vde_id": 5038,
            "legislation": "EPA",
            "make": "AUDI",
            "model": "TEST08062026",
            "year": 2027,
            "cycle_name": "FTP75",
            "notes": "Imported baseline request",
            "mass_kg": 1850,
            "cda_m2": 0.64,
            "A": 145.17,
            "B": 0.09357,
            "C": 0.040838,
        },
        "proposals": [],
        "issues": [],
        "original_request": {},
    }


def _domain_request(
    domain: str,
    *,
    raw_type=None,
    proposal_type=None,
    selection_mode=None,
    details_seed=None,
    raw_values=None,
    issues=None,
    has_internal_equivalent=True,
) -> dict:
    normalized = None
    if proposal_type is not None:
        normalized = {
            "ok": True,
            "domain": domain,
            "template_label": raw_type,
            "proposal_type": proposal_type,
            "selection_mode": selection_mode or raw_type or proposal_type,
            "details": deepcopy(details_seed or {}),
            "has_internal_equivalent": has_internal_equivalent,
        }
    return {
        "domain": domain,
        "raw_proposal_type": raw_type,
        "normalized_proposal": normalized,
        "raw_values": deepcopy(raw_values or {}),
        "normalized_values": deepcopy(raw_values or {}),
        "aliases": {},
        "issues": deepcopy(issues or []),
        "proposal_type": proposal_type,
        "selection_mode": selection_mode or raw_type,
        "has_internal_equivalent": has_internal_equivalent,
        "proposal_details_seed": deepcopy(details_seed or {}),
    }


def _proposal(
    proposal_id: str,
    *,
    display_index: int,
    source_index: int,
    walk_from: dict | None = None,
    name: str | None = None,
    domain_requests: dict | None = None,
    issues: list[dict] | None = None,
) -> dict:
    return {
        "proposal_id": proposal_id,
        "display_index": display_index,
        "source_column": f"Requested #{source_index}",
        "source_index": source_index,
        "name": name,
        "walk_from": deepcopy(walk_from or {"kind": "baseline", "proposal_id": None, "source_column": "Baseline"}),
        "raw_values": {"notes": name} if name else {},
        "normalized_values": {"notes": name} if name else {},
        "domain_requests": deepcopy(domain_requests or {}),
        "issues": deepcopy(issues or []),
    }


class TestVdeRequestAdapter(unittest.TestCase):
    def test_maps_baseline_printed_and_corrections(self):
        state = build_v21_workbook_state_from_request_draft(_base_draft(), {"rows": []})
        baseline = state["columns"]["baseline"]
        self.assertEqual(state["metadata"]["selected_baseline_vde_id"], 5038)
        self.assertEqual(state["metadata"]["model_year"], 2027)
        self.assertEqual(state["metadata"]["description"], "Imported baseline request")
        self.assertEqual(baseline["line_source"], "Existing VDE DB")
        self.assertEqual(baseline["direct"]["curb_mass_kg"], 1850)
        self.assertEqual(baseline["direct"]["CdA"], 0.64)
        self.assertEqual(baseline["printed_overrides"]["__global__"]["curb_mass_kg"], 1800)
        self.assertEqual(baseline["printed_overrides"]["__global__"]["CdA"], 0.62)

    def test_preserves_effective_baseline_payload(self):
        state = build_v21_workbook_state_from_request_draft(_base_draft(), {"rows": []})
        self.assertEqual(state["vde_request_import"]["effective_baseline"]["mass_kg"], 1850)
        self.assertEqual(state["vde_request_import"]["effective_baseline"]["year"], 2027)

    def test_new_test_baseline_uses_new_test_line_source_without_vde_id(self):
        draft = _base_draft()
        draft["baseline_source_type"] = "NEW_TEST"
        draft["baseline_printed"] = {}
        draft["baseline_corrections"] = {
            "abc_total_source_ui": "From test coastdown",
            "A": 120.0,
            "B": 0.02,
            "C": 0.008,
            "test_mass_kg": 1600.0,
            "test_mass_basis": "EPA_INERTIA_CLASS",
            "inertia_class": 1600.0,
            "legislation": "EPA",
            "cycle_name": "FTP75_HWFET",
        }
        draft["effective_baseline"] = deepcopy(draft["baseline_corrections"])

        state = build_v21_workbook_state_from_request_draft(draft, {"rows": []})

        baseline = state["columns"]["baseline"]
        self.assertEqual(baseline["line_source"], "New test ABC_TOTAL")
        self.assertIsNone(baseline["selected_vde_id"])
        self.assertEqual(baseline["direct"]["ABC_TOTAL_A"], 120.0)
        self.assertEqual(baseline["direct"]["ABC_TOTAL_B"], 0.02)
        self.assertEqual(baseline["direct"]["ABC_TOTAL_C"], 0.008)
        self.assertEqual(baseline["direct"]["test_mass_kg"], 1600.0)

    def test_builds_single_simple_proposal(self):
        draft = _base_draft()
        draft["proposals"] = [
            _proposal(
                "proposal_req_1",
                display_index=1,
                source_index=1,
                name="Mass request",
                domain_requests={
                    "mass": _domain_request(
                        "mass",
                        raw_type="Custom test mass",
                        proposal_type="CUSTOM_MASS",
                        raw_values={"mass_kg": 1735},
                    )
                },
            )
        ]
        state = build_v21_workbook_state_from_request_draft(draft, {"rows": []})
        self.assertEqual(state["scenarios"][1]["key"], "proposal_req_1")
        self.assertEqual(state["scenarios"][1]["label"], "Requested #1")
        self.assertEqual(state["columns"]["proposal_req_1"]["walk_from"], "baseline")
        self.assertEqual(state["proposals"]["proposal_req_1"]["mass"]["proposal_type"], "CUSTOM_MASS")
        self.assertEqual(state["proposals"]["proposal_req_1"]["mass"]["details"]["test_mass_kg"], 1735)

    def test_gap_requested_columns_are_renumbered_but_source_is_preserved(self):
        draft = _base_draft()
        draft["proposals"] = [
            _proposal(
                "proposal_req_1",
                display_index=1,
                source_index=1,
                name="First",
                domain_requests={"aero": _domain_request("aero", raw_type="Delta CdA", proposal_type="AERO_DELTA_CDA", raw_values={"cda_m2": 0.01})},
            ),
            _proposal(
                "proposal_req_5",
                display_index=2,
                source_index=5,
                name="Fifth source",
                domain_requests={"transmission": _domain_request("transmission", raw_type="Delta ABC", proposal_type="UPDATE_TRANS_DRAG_ABC", details_seed={"change_mode": "Delta ABC"}, raw_values={"trans_A_coef_N": 2.5})},
            ),
        ]
        state = build_v21_workbook_state_from_request_draft(draft, {"rows": []})
        self.assertEqual(state["scenarios"][1]["label"], "Requested #1")
        self.assertEqual(state["scenarios"][2]["label"], "Requested #2")
        self.assertEqual(state["vde_request_import"]["columns"]["proposal_req_5"]["source_column"], "Requested #5")
        self.assertEqual(state["vde_request_import"]["columns"]["proposal_req_5"]["display_index"], 2)

    def test_walk_from_previous_proposal_uses_internal_column_id(self):
        draft = _base_draft()
        draft["proposals"] = [
            _proposal(
                "proposal_req_1",
                display_index=1,
                source_index=1,
                domain_requests={"mass": _domain_request("mass", raw_type="Custom test mass", proposal_type="CUSTOM_MASS", raw_values={"mass_kg": 1700})},
            ),
            _proposal(
                "proposal_req_3",
                display_index=2,
                source_index=3,
                walk_from={"kind": "proposal", "proposal_id": "proposal_req_1", "source_column": "Requested #1"},
                domain_requests={"aero": _domain_request("aero", raw_type="Absolute CdA", proposal_type="AERO_ABSOLUTE_CDA", raw_values={"cda_m2": 0.7})},
            ),
        ]
        state = build_v21_workbook_state_from_request_draft(draft, {"rows": []})
        self.assertEqual(state["columns"]["proposal_req_3"]["walk_from"], "proposal_req_1")

    def test_invalid_walk_from_issue_is_preserved(self):
        draft = _base_draft()
        draft["proposals"] = [
            _proposal(
                "proposal_req_2",
                display_index=1,
                source_index=2,
                walk_from={"kind": "proposal", "proposal_id": None, "source_column": "Requested #9"},
                domain_requests={"mass": _domain_request("mass", raw_type="Custom test mass", proposal_type="CUSTOM_MASS", raw_values={"mass_kg": 1700})},
                issues=[{"severity": "error", "code": "missing_walk_from_target", "message": "Walk From Requested #9 does not exist."}],
            )
        ]
        state = build_v21_workbook_state_from_request_draft(draft, {"rows": []})
        summary = state["vde_request_import_summary"]
        self.assertEqual(state["columns"]["proposal_req_2"]["walk_from"], "baseline")
        self.assertTrue(any(item["code"] == "missing_walk_from_target" for item in summary["review_issues"]))
        self.assertEqual(state["vde_request_import"]["columns"]["proposal_req_2"]["walk_from_requested"], "Requested #9")

    def test_lookup_from_db_without_clean_internal_equivalent_is_preserved(self):
        draft = _base_draft()
        draft["proposals"] = [
            _proposal(
                "proposal_req_1",
                display_index=1,
                source_index=1,
                domain_requests={
                    "transmission": _domain_request(
                        "transmission",
                        raw_type="Lookup from DB",
                        proposal_type="TRANS_METADATA_ONLY",
                        raw_values={"transmission_component_db_id": 42},
                        issues=[{"severity": "warning", "code": "proposal_contract_note", "message": "metadata-only import"}],
                        has_internal_equivalent=False,
                    )
                },
            )
        ]
        state = build_v21_workbook_state_from_request_draft(draft, {"rows": []})
        proposal = state["proposals"]["proposal_req_1"]["transmission"]
        self.assertEqual(proposal["proposal_type"], "TRANS_METADATA_ONLY")
        self.assertEqual(proposal["status"], "Review")
        self.assertEqual(state["vde_request_import"]["columns"]["proposal_req_1"]["domains"]["transmission"]["raw_values"]["transmission_component_db_id"], 42)

    def test_not_used_is_preserved_for_component_domains(self):
        draft = _base_draft()
        draft["proposals"] = [
            _proposal(
                "proposal_req_1",
                display_index=1,
                source_index=1,
                domain_requests={
                    "brake": _domain_request(
                        "brake",
                        raw_type="Not used",
                        proposal_type="BRAKE_NOT_USED",
                        has_internal_equivalent=False,
                    )
                },
            )
        ]
        state = build_v21_workbook_state_from_request_draft(draft, {"rows": []})
        self.assertEqual(state["proposals"]["proposal_req_1"]["brake"]["proposal_type"], "BRAKE_NOT_USED")
        self.assertEqual(state["proposals"]["proposal_req_1"]["brake"]["status"], "Review")

    def test_aero_not_used_creates_review_placeholder(self):
        draft = _base_draft()
        draft["proposals"] = [
            _proposal(
                "proposal_req_2",
                display_index=1,
                source_index=2,
                domain_requests={
                    "aero": _domain_request(
                        "aero",
                        raw_type="Not used",
                        proposal_type="INHERIT",
                        selection_mode="NOT_USED",
                        has_internal_equivalent=False,
                        issues=[{"severity": "review", "code": "aero_not_used_review", "message": "Aero not used requires review."}],
                    )
                },
            )
        ]
        state = build_v21_workbook_state_from_request_draft(draft, {"rows": []})
        proposal = state["proposals"]["proposal_req_2"]["aero"]
        self.assertEqual(proposal["proposal_type"], "AERO_NOT_USED")
        self.assertEqual(proposal["status"], "Review")
        self.assertTrue(any("Aero Not used" in note for note in proposal["notes"]))

    def test_unknown_proposal_type_is_preserved_without_silent_fallback(self):
        draft = _base_draft()
        draft["proposals"] = [
            _proposal(
                "proposal_req_4",
                display_index=1,
                source_index=4,
                domain_requests={
                    "transmission": _domain_request(
                        "transmission",
                        raw_type="Weird Mode",
                        proposal_type=None,
                        raw_values={"trans_A_coef_N": 8.0},
                        issues=[{"severity": "review", "code": "unknown_proposal_type", "message": "Unknown template proposal type."}],
                        has_internal_equivalent=False,
                    )
                },
            )
        ]
        state = build_v21_workbook_state_from_request_draft(draft, {"rows": []})
        proposal = state["proposals"]["proposal_req_4"]["transmission"]
        self.assertEqual(proposal["proposal_type"], "TRANS_IMPORTED_REVIEW")
        self.assertEqual(state["vde_request_import"]["columns"]["proposal_req_4"]["domains"]["transmission"]["raw_proposal_type"], "Weird Mode")
        self.assertTrue(any(item["code"] == "unknown_proposal_type" for item in state["vde_request_import_summary"]["review_issues"]))

    def test_summary_groups_issues(self):
        draft = _base_draft()
        draft["issues"] = [{"severity": "warning", "code": "template_version_warning", "message": "Version mismatch"}]
        draft["proposals"] = [
            _proposal(
                "proposal_req_1",
                display_index=1,
                source_index=1,
                issues=[{"severity": "error", "code": "future_walk_from", "message": "Future Walk From"}],
                domain_requests={"mass": _domain_request("mass", raw_type="Custom test mass", proposal_type="CUSTOM_MASS", raw_values={"mass_kg": 1700})},
            )
        ]
        summary = build_v21_request_import_summary(draft)
        self.assertEqual(summary["blocking_count"], 0)
        self.assertEqual(summary["review_count"], 1)
        self.assertEqual(summary["warning_count"], 1)
        self.assertEqual(summary["proposal_count"], 1)

    def test_state_is_not_mutated_when_import_is_invalid(self):
        current_state = {"menu": "Scenario Workbook", "columns": {"baseline": {"direct": {}}, "old": {"direct": {"description": "keep"}}}}
        snapshot = deepcopy(current_state)
        with self.assertRaises(ValueError):
            build_v21_workbook_state_from_request_draft({}, current_state)
        self.assertEqual(current_state, snapshot)

    def test_apply_replaces_previous_draft(self):
        draft = _base_draft()
        draft["proposals"] = [
            _proposal(
                "proposal_req_8",
                display_index=1,
                source_index=8,
                name="Replacement",
                domain_requests={"aero": _domain_request("aero", raw_type="Absolute CdA", proposal_type="AERO_ABSOLUTE_CDA", raw_values={"cda_m2": 0.77})},
            )
        ]
        current_state = {
            "menu": "Preview & Save",
            "rows": [],
            "scenarios": [{"key": "baseline", "label": "Baseline", "role": "baseline"}, {"key": "walked_1", "label": "Walked #1", "role": "walked"}],
            "columns": {"baseline": {"direct": {}}, "walked_1": {"direct": {"description": "old"}}},
            "proposals": {"walked_1": {"mass": {"id": "prop_1", "proposal_type": "CUSTOM_MASS"}}},
        }
        snapshot = deepcopy(current_state)
        next_state = apply_v21_request_import(current_state, draft)
        self.assertEqual(current_state, snapshot)
        self.assertIn("proposal_req_8", next_state["columns"])
        self.assertNotIn("walked_1", next_state["columns"])
        self.assertEqual(next_state["menu"], "Preview & Save")
        self.assertEqual(next_state["proposals"]["proposal_req_8"]["aero"]["proposal_type"], "AERO_ABSOLUTE_CDA")

    def test_result_is_json_serializable(self):
        draft = _base_draft()
        draft["proposals"] = [
            _proposal(
                "proposal_req_1",
                display_index=1,
                source_index=1,
                domain_requests={"aero": _domain_request("aero", raw_type="Delta CdA", proposal_type="AERO_DELTA_CDA", raw_values={"cda_m2": 0.02})},
            )
        ]
        state = build_v21_workbook_state_from_request_draft(draft, {"rows": []})
        json.dumps(state, default=str)


if __name__ == "__main__":
    unittest.main()
