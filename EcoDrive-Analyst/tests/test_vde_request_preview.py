from __future__ import annotations

import unittest

from src.vde_core.vde_request_preview import (
    build_component_action_rows,
    build_proposal_preview_model,
    build_request_audit_rows,
    build_request_comparison_rows,
    build_request_resolution_fingerprint,
    build_validation_summary,
)


def _resolution_result() -> dict:
    return {
        "status": "Review",
        "baseline": {
            "effective": {"mass_kg": 1600.0, "test_mass_kg": 1736.0, "cda_m2": 0.62},
            "corrected_fields": ["mass_kg"],
        },
        "resolved_columns": {
            "baseline": {
                "mass_kg": 1600.0,
                "test_mass_kg": 1736.0,
                "CdA": 0.62,
                "initial_abc_total": {"A": 120.0, "B": 0.02, "C": 0.01},
            }
        },
        "proposal_results": [
            {
                "proposal_id": "proposal_req_1",
                "display_index": 1,
                "source_column": "Requested #1",
                "walk_from": {"column_id": "baseline", "label": "Baseline"},
                "source_snapshot": {"mass_kg": 1600.0},
                "requested_snapshot": {"mass_kg": 1600.0, "CdA": 0.63},
                "resolved_snapshot": {
                    "mass_kg": 1600.0,
                    "test_mass_kg": 1736.0,
                    "CdA": 0.63,
                    "resolved_mass_setup": {"resolved_mass_used_kg": 1736.0},
                },
                "domain_results": {
                    "mass": {"status": "OK", "proposal_type": "INHERIT", "source": "Baseline", "requested_values": {}, "resolved_values": {}},
                    "aero": {"status": "Review", "proposal_type": "AERO_ABSOLUTE_CDA", "source": "Baseline", "requested_values": {"new_CdA": 0.63}, "resolved_values": {"CdA": 0.63}, "issues": [{"code": "manual_reference_override"}]},
                },
                "abc_total": {"A": 120.0, "B": 0.02, "C": 0.0105},
                "abc_net": None,
                "vde_results": {"total": {"mj_per_km": 1.234}, "net": None},
                "status": "Review",
                "issues": [{"severity": "review", "code": "manual_reference_override", "domain": "aero", "field_key": "baseline_CdA", "message": "Manual override used."}],
                "component_actions": [
                    {"domain": "aero", "action": "snapshot_only", "component_id": None, "requires_confirmation": True, "issues": [], "component_snapshot": {"CdA": 0.63}}
                ],
                "preview_summary": {"warnings": ["vde_net_unavailable_transmission_losses_missing"]},
            }
        ],
        "issues": [{"severity": "review", "code": "manual_reference_override"}],
    }


class VdeRequestPreviewTests(unittest.TestCase):
    def test_fingerprint_is_deterministic_and_ignores_timestamps(self):
        workbook_a = {
            "scenarios": [{"key": "baseline", "label": "Baseline", "role": "baseline"}],
            "columns": {"baseline": {"kind": "baseline"}},
            "proposals": {},
            "vde_request_import": {
                "baseline_printed": {"mass_kg": 1600.0},
                "baseline_corrections": {"mass_kg": 1650.0},
                "effective_baseline": {"mass_kg": 1650.0},
                "source": {"filename": "C:/temp/foo.xlsx", "imported_at": "2026-07-11T00:00:00Z"},
            },
        }
        workbook_b = {
            "proposals": {},
            "columns": {"baseline": {"kind": "baseline"}},
            "scenarios": [{"role": "baseline", "label": "Baseline", "key": "baseline"}],
            "vde_request_import": {
                "effective_baseline": {"mass_kg": 1650.0},
                "baseline_corrections": {"mass_kg": 1650.0},
                "baseline_printed": {"mass_kg": 1600.0},
                "source": {"filename": "D:/other/bar.xlsx", "imported_at": "2026-07-12T10:30:00Z"},
            },
        }

        self.assertEqual(
            build_request_resolution_fingerprint(workbook_a, {"cycle_df": "ignored", "mass_kg": 1650.0}),
            build_request_resolution_fingerprint(workbook_b, {"cycle_df": "different", "mass_kg": 1650.0}),
        )

    def test_validation_summary_counts_statuses_and_warnings(self):
        summary = build_validation_summary(_resolution_result())

        self.assertEqual(summary["overall_status"], "Review")
        self.assertEqual(summary["proposal_count"], 1)
        self.assertEqual(summary["review_count"], 1)
        self.assertEqual(summary["warning_count"], 1)

    def test_comparison_rows_include_baseline_and_requested(self):
        rows = build_request_comparison_rows(_resolution_result())

        self.assertEqual(rows[0]["Scenario"], "Baseline")
        self.assertEqual(rows[1]["Scenario"], "Requested #1")
        self.assertEqual(rows[1]["VDE_NET [MJ/km]"], "—")

    def test_component_action_rows_are_flattened(self):
        rows = build_component_action_rows(_resolution_result()["proposal_results"][0])

        self.assertEqual(rows[0]["Domain"], "aero")
        self.assertEqual(rows[0]["Action"], "snapshot_only")

    def test_build_proposal_preview_model_uses_proposal_id(self):
        model = build_proposal_preview_model(_resolution_result()["proposal_results"][0])

        self.assertEqual(model["header"]["proposal_id"], "proposal_req_1")
        self.assertEqual(model["header"]["requested_label"], "Requested #1")
        self.assertEqual(model["header"]["source_column"], "Requested #1")
        self.assertTrue(any(row["Field"] == "VDE_NET [MJ/km]" for row in model["engineering_rows"]))
        self.assertTrue(any(row["Severity"] == "warning" for row in model["validation_rows"]))
        self.assertTrue(any(row["Domain"] == "aero" for row in model["audit_rows"]))

    def test_build_request_audit_rows_flattens_domain_results(self):
        rows = build_request_audit_rows(_resolution_result())

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["Scenario"], "Requested #1")


if __name__ == "__main__":
    unittest.main()
