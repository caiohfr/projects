import unittest

from src.vde_core.vde_workbook_v21 import (
    build_v21_save_plan,
    resolve_v21_domain,
    resolve_v21_reference_triplet,
    resolve_v21_reference_value,
    resolve_v21_workbook,
    rollup_v21_statuses,
    validate_v21_absolute_reference,
)


class TestVdeWorkbookV21(unittest.TestCase):
    def test_rollup_statuses_prioritizes_invalid_then_missing(self):
        self.assertEqual(rollup_v21_statuses(["Inherited", "OK", "Review"]), "Review")
        self.assertEqual(rollup_v21_statuses(["OK", "Missing", "Draft"]), "Missing")
        self.assertEqual(rollup_v21_statuses(["OK", "Invalid", "Review"]), "Invalid")

    def test_resolve_v21_domain_keeps_inherited_problem_status(self):
        inherited = resolve_v21_domain(
            "tire",
            {
                "mode": "direct",
                "id": "prop_1",
                "proposal_type": "TIRE_DB_LOOKUP",
                "label": "TPS Tire",
                "status": "Missing",
            },
            None,
            source_column="walked_1",
            current_column="walked_2",
            type_labels={"TIRE_DB_LOOKUP": "Tire DB lookup"},
        )
        self.assertEqual(inherited["mode"], "inherited")
        self.assertEqual(inherited["status"], "Missing")
        self.assertEqual(inherited["display_status"], "Inherited")
        self.assertEqual(inherited["inherited_from"], "walked_1")

    def test_resolve_v21_workbook_accumulates_effective_proposals_by_walk_chain(self):
        workbook_state = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
                {"key": "walked_2", "label": "Walked #2", "role": "walked"},
            ],
            "columns": {
                "baseline": {"kind": "baseline", "label": "Baseline", "domains": {}},
                "walked_1": {
                    "kind": "walked",
                    "label": "Walked #1",
                    "walk_from": "baseline",
                    "domains": {
                        "tire": {
                            "mode": "direct",
                            "id": "prop_1",
                            "proposal_type": "TIRE_DB_LOOKUP",
                            "label": "TPS Tire",
                            "status": "OK",
                        }
                    },
                },
                "walked_2": {
                    "kind": "walked",
                    "label": "Walked #2",
                    "walk_from": "walked_1",
                    "domains": {
                        "aero": {
                            "mode": "direct",
                            "id": "prop_2",
                            "proposal_type": "AERO_DELTA_CDA",
                            "label": "CdA tweak",
                            "status": "Review",
                        }
                    },
                },
            },
        }

        resolved = resolve_v21_workbook(
            workbook_state,
            domain_keys=["mass", "aero", "tire"],
            type_labels={
                "TIRE_DB_LOOKUP": "Tire DB lookup",
                "AERO_DELTA_CDA": "Aero delta CdA",
            },
        )

        walked_2 = resolved["columns"]["walked_2"]
        self.assertEqual(walked_2["walk_from"], "walked_1")
        self.assertEqual(walked_2["proposal_direct"], "Prop #2 · CdA tweak")
        self.assertEqual(
            walked_2["proposal_effective"],
            "Prop #1 · TPS Tire + Prop #2 · CdA tweak",
        )
        self.assertEqual(walked_2["effective_domains"]["tire"]["mode"], "inherited")
        self.assertEqual(walked_2["effective_domains"]["tire"]["inherited_from"], "walked_1")
        self.assertEqual(walked_2["review_status"], "Review")

    def test_resolve_v21_workbook_rejects_self_or_future_walk_from(self):
        workbook_state = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
                {"key": "walked_2", "label": "Walked #2", "role": "walked"},
            ],
            "columns": {
                "baseline": {"kind": "baseline", "label": "Baseline", "domains": {}},
                "walked_1": {"kind": "walked", "label": "Walked #1", "walk_from": "baseline", "domains": {}},
                "walked_2": {"kind": "walked", "label": "Walked #2", "walk_from": "walked_2", "domains": {}},
            },
        }

        resolved = resolve_v21_workbook(workbook_state, domain_keys=["tire"])
        walked_2 = resolved["columns"]["walked_2"]
        self.assertEqual(walked_2["walk_from"], "walked_1")
        self.assertEqual(walked_2["walk_from_status"], "Invalid")
        self.assertEqual(walked_2["review_status"], "Invalid")
        self.assertIn("Invalid Walk From", walked_2["walk_from_note"])

    def test_build_v21_save_plan_blocks_missing_columns_and_marks_review_confirmation(self):
        resolved_workbook = {
            "column_order": ["baseline", "walked_1", "walked_2"],
            "columns": {
                "baseline": {
                    "label": "Baseline",
                    "direct_domains": {},
                    "effective_domains": {},
                    "review_status": "Inherited",
                    "effective_status": "Inherited",
                },
                "walked_1": {
                    "label": "Walked #1",
                    "direct_domains": {"tire": {"status": "OK"}},
                    "effective_domains": {"tire": {"status": "OK"}},
                    "review_status": "Review",
                    "effective_status": "OK",
                },
                "walked_2": {
                    "label": "Walked #2",
                    "direct_domains": {"aero": {"status": "Missing"}},
                    "effective_domains": {"aero": {"status": "Missing"}},
                    "review_status": "Missing",
                    "effective_status": "Missing",
                },
            },
        }
        previews = {
            "baseline": {"ok": True},
            "walked_1": {"ok": True},
            "walked_2": {"ok": False},
        }

        plan = build_v21_save_plan(resolved_workbook, previews, selected_target="walked_1")

        self.assertFalse(plan["can_save_all"])
        self.assertTrue(plan["has_saveable_rows"])
        self.assertIn("Walked #2", plan["blocked_columns"])
        self.assertIn("Walked #1", plan["review_columns"])
        walked_1 = next(item for item in plan["rows"] if item["column_id"] == "walked_1")
        self.assertEqual(walked_1["action"], "create_new")
        self.assertEqual(walked_1["status"], "Review")
        self.assertTrue(walked_1["requires_confirmation"])
        walked_2 = next(item for item in plan["rows"] if item["column_id"] == "walked_2")
        self.assertEqual(walked_2["action"], "skip")
        self.assertEqual(walked_2["status"], "Missing")

    def test_build_v21_save_plan_flags_baseline_update_request(self):
        resolved_workbook = {
            "column_order": ["baseline", "walked_1"],
            "columns": {
                "baseline": {
                    "label": "Baseline",
                    "direct_domains": {},
                    "effective_domains": {},
                    "review_status": "Inherited",
                    "effective_status": "Inherited",
                },
                "walked_1": {
                    "label": "Walked #1",
                    "direct_domains": {
                        "transmission": {
                            "status": "Review",
                            "details": {"baseline_update_requested": True},
                            "proposal_type": "UPDATE_TRANS_DRAG_ABC",
                            "label": "Absolute trans drag",
                            "badge_text": "Prop #1 - Absolute trans drag",
                        }
                    },
                    "effective_domains": {"transmission": {"status": "Review"}},
                    "review_status": "Review",
                    "effective_status": "Review",
                },
            },
        }
        previews = {"baseline": {"ok": True}, "walked_1": {"ok": True}}

        plan = build_v21_save_plan(
            resolved_workbook,
            previews,
            baseline_is_existing=True,
            baseline_target_id=5038,
            domain_labels={"transmission": "Transmission"},
        )

        baseline = next(item for item in plan["rows"] if item["column_id"] == "baseline")
        self.assertEqual(baseline["action"], "update_existing")
        self.assertEqual(baseline["status"], "Review")
        self.assertTrue(baseline["requires_confirmation"])
        self.assertEqual(baseline["target_vde_id"], 5038)
        self.assertEqual(len(plan["baseline_update_requests"]), 1)

    def test_build_v21_save_plan_uses_saved_targets_for_walked_updates(self):
        resolved_workbook = {
            "column_order": ["baseline", "walked_1"],
            "columns": {
                "baseline": {"label": "Baseline", "direct_domains": {}, "effective_domains": {}, "review_status": "Inherited", "effective_status": "Inherited"},
                "walked_1": {
                    "label": "Walked #1",
                    "direct_domains": {"tire": {"status": "OK"}},
                    "effective_domains": {"tire": {"status": "OK"}},
                    "review_status": "OK",
                    "effective_status": "OK",
                },
            },
        }
        previews = {"baseline": {"ok": True}, "walked_1": {"ok": True}}
        plan = build_v21_save_plan(
            resolved_workbook,
            previews,
            saved_targets={"walked_1": 6123},
        )
        walked_1 = next(item for item in plan["rows"] if item["column_id"] == "walked_1")
        self.assertEqual(walked_1["action"], "update_existing")
        self.assertEqual(walked_1["target_vde_id"], 6123)

    def test_validate_v21_absolute_reference_requires_new_and_baseline_inputs(self):
        result = validate_v21_absolute_reference(
            {"new_trans_A": 1.0},
            new_fields=("new_trans_A", "new_trans_B", "new_trans_C"),
            baseline_fields=("baseline_trans_A", "baseline_trans_B", "baseline_trans_C"),
            has_reference=False,
            reference_source=None,
        )
        self.assertEqual(result["status"], "Missing")
        self.assertEqual(result["missing_fields"], ["new_trans_B", "new_trans_C"])

        result = validate_v21_absolute_reference(
            {"new_trans_A": 1.0, "new_trans_B": 2.0, "new_trans_C": 3.0},
            new_fields=("new_trans_A", "new_trans_B", "new_trans_C"),
            baseline_fields=("baseline_trans_A", "baseline_trans_B", "baseline_trans_C"),
            has_reference=False,
            reference_source=None,
        )
        self.assertEqual(result["status"], "Missing")
        self.assertEqual(
            result["missing_fields"],
            ["baseline_trans_A", "baseline_trans_B", "baseline_trans_C"],
        )

    def test_validate_v21_absolute_reference_marks_manual_override_as_review(self):
        result = validate_v21_absolute_reference(
            {
                "brake_A": 1.0,
                "brake_B": 2.0,
                "brake_C": 3.0,
            },
            new_fields=("brake_A", "brake_B", "brake_C"),
            baseline_fields=("baseline_component_A", "baseline_component_B", "baseline_component_C"),
            has_reference=True,
            reference_source="manual_override",
            baseline_update_requested=True,
        )
        self.assertEqual(result["status"], "Review")
        self.assertTrue(result["warnings"])
        self.assertIn("manual baseline override", result["warnings"][0])
        self.assertIn("Baseline update requested", result["warnings"][1])

    def test_validate_v21_absolute_reference_marks_assume_zero_as_review(self):
        result = validate_v21_absolute_reference(
            {"new_trans_A": 1.0, "new_trans_B": 2.0, "new_trans_C": 3.0},
            new_fields=("new_trans_A", "new_trans_B", "new_trans_C"),
            baseline_fields=("baseline_trans_A", "baseline_trans_B", "baseline_trans_C"),
            has_reference=True,
            reference_source="assume_zero",
        )
        self.assertEqual(result["status"], "Review")
        self.assertIn("assuming zero", result["warnings"][0])

    def test_resolve_v21_reference_value_prefers_manual_override(self):
        result = resolve_v21_reference_value(1.5, manual_value=2.5, assume_zero=True)
        self.assertEqual(result["source"], "manual_override")
        self.assertEqual(result["value"], 2.5)

    def test_resolve_v21_reference_triplet_prefers_manual_then_inherited_then_zero(self):
        manual = resolve_v21_reference_triplet((1.0, 2.0, 3.0), manual_values=(4.0, 5.0, 6.0), assume_zero=True)
        self.assertEqual(manual["source"], "manual_override")
        self.assertEqual(manual["values"], (4.0, 5.0, 6.0))

        inherited = resolve_v21_reference_triplet((1.0, 2.0, 3.0), manual_values=(None, None, None), assume_zero=True)
        self.assertEqual(inherited["source"], "inherited")
        self.assertEqual(inherited["values"], (1.0, 2.0, 3.0))

        zero = resolve_v21_reference_triplet((None, None, None), manual_values=(None, None, None), assume_zero=True)
        self.assertEqual(zero["source"], "assume_zero")
        self.assertEqual(zero["values"], (0.0, 0.0, 0.0))

    def test_resolve_v21_reference_triplet_allows_mixed_manual_and_inherited_values(self):
        mixed = resolve_v21_reference_triplet((1.0, None, 3.0), manual_values=(None, 5.0, None), assume_zero=False)
        self.assertTrue(mixed["has_reference"])
        self.assertEqual(mixed["source"], "manual_override")
        self.assertEqual(mixed["values"], (1.0, 5.0, 3.0))


if __name__ == "__main__":
    unittest.main()
