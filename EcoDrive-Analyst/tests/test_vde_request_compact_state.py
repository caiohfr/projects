from __future__ import annotations

from copy import deepcopy
import unittest

from src.vde_core.vde_request_compact_state import (
    V22_MAX_PROPOSALS,
    add_v22_proposal,
    allowed_walk_from_options,
    apply_v22_baseline,
    apply_v22_corrections,
    apply_v22_domain_inputs,
    apply_v22_new_test_baseline,
    apply_v22_proposal_metadata,
    apply_v22_proposal_matrix,
    build_new_test_canonical_baseline,
    build_v22_canonical_request_draft,
    create_v22_state,
    has_v22_tire_pressure_unit_override,
    normalize_v22_state,
    proposal_type_labels_by_domain,
    remove_v22_proposal,
    resolve_v22_tire_pressure_unit,
    resolve_v22_baseline_mass_review,
    set_v22_tire_pressure_unit_preference,
)


class TestVdeRequestCompactState(unittest.TestCase):
    def test_state_initializes_with_two_requested_proposals(self):
        state = create_v22_state()
        self.assertEqual([p["proposal_id"] for p in state["proposals"]], ["requested_1", "requested_2"])
        self.assertEqual(state["baseline"]["correction_disposition"], "request_only")
        self.assertIsNone(state["ui_preferences"]["tire_pressure_unit"])

    def test_metadata_edit_keeps_engineering_preview_fresh_and_invalidates_save_only(self):
        state = create_v22_state()
        state["preview"] = {"status": "fresh", "fingerprint": "physics-fingerprint", "result": {"ok": True}}
        state["save"] = {"status": "success", "result": {"record_id": 1}}

        updated = apply_v22_proposal_metadata(
            state,
            "requested_1",
            {"name": "Low RR tire proposal", "make": "QA MAKE", "model": "QA MODEL", "model_year": "2028"},
        )

        self.assertEqual(updated["preview"]["status"], "fresh")
        self.assertEqual(updated["preview"]["fingerprint"], "physics-fingerprint")
        self.assertEqual(updated["save"]["status"], "pending")
        self.assertEqual(updated["proposals"][0]["metadata_overrides"]["make"], "QA MAKE")

    def test_tire_pressure_unit_defaults_follow_global_until_overridden(self):
        state = create_v22_state()

        self.assertEqual(resolve_v22_tire_pressure_unit(state, "Metric"), "kPa")
        self.assertEqual(resolve_v22_tire_pressure_unit(state, "US customary"), "psi")
        self.assertFalse(has_v22_tire_pressure_unit_override(state))

        state = set_v22_tire_pressure_unit_preference(state, "bar")
        self.assertEqual(resolve_v22_tire_pressure_unit(state, "Metric"), "bar")
        self.assertEqual(resolve_v22_tire_pressure_unit(state, "US customary"), "bar")
        self.assertTrue(has_v22_tire_pressure_unit_override(state))

    def test_add_proposal_preserves_existing_ids_and_limits_to_30(self):
        state = create_v22_state()
        state = add_v22_proposal(state)
        self.assertEqual([p["proposal_id"] for p in state["proposals"][:2]], ["requested_1", "requested_2"])
        self.assertEqual(state["proposals"][2]["proposal_id"], "requested_3")
        for _ in range(40):
            state = add_v22_proposal(state)
        self.assertEqual(len(state["proposals"]), V22_MAX_PROPOSALS)

    def test_remove_proposal_does_not_reuse_id_and_renumbers_display_indexes(self):
        state = create_v22_state()
        state = add_v22_proposal(state)
        state = remove_v22_proposal(state, "requested_2")
        self.assertEqual([p["proposal_id"] for p in state["proposals"]], ["requested_1", "requested_3"])
        self.assertEqual([p["display_index"] for p in state["proposals"]], [1, 2])
        state = add_v22_proposal(state)
        self.assertEqual(state["proposals"][-1]["proposal_id"], "requested_4")

    def test_walk_from_accepts_only_baseline_and_previous_proposals(self):
        state = add_v22_proposal(create_v22_state())
        self.assertEqual(allowed_walk_from_options(state, "requested_1"), ["baseline"])
        self.assertEqual(allowed_walk_from_options(state, "requested_2"), ["baseline", "requested_1"])
        self.assertEqual(allowed_walk_from_options(state, "requested_3"), ["baseline", "requested_1", "requested_2"])

    def test_broken_walk_from_dependency_is_preserved_and_marked_in_draft(self):
        state = add_v22_proposal(create_v22_state())
        state["proposals"][2]["walk_from"] = "requested_2"
        state = remove_v22_proposal(state, "requested_2")
        survivor = state["proposals"][1]
        self.assertEqual(survivor["proposal_id"], "requested_3")
        self.assertEqual(survivor["walk_from"], "requested_2")
        self.assertNotIn("requested_2", allowed_walk_from_options(state, "requested_3"))
        draft = build_v22_canonical_request_draft(state)
        self.assertEqual(draft["proposals"][1]["issues"][0]["code"], "invalid_walk_from")

    def test_blank_correction_inherits_printed_and_zero_replaces_printed(self):
        state = apply_v22_baseline(create_v22_state(), {"id": 5, "mass_kg": 1500.0, "coast_A_N": 10.0})
        state = apply_v22_corrections(state, {"mass_kg": "", "A": 0})
        self.assertEqual(state["baseline"]["effective"]["mass_kg"], 1500.0)
        self.assertEqual(state["baseline"]["effective"]["A"], 0)
        self.assertNotIn("mass_kg", state["baseline"]["corrections"])

    def test_normal_correction_replaces_printed_without_mutating_original_row(self):
        row = {"id": 8, "mass_kg": 1500.0, "coast_A_N": 10.0}
        original = deepcopy(row)
        state = apply_v22_baseline(create_v22_state(), row)
        state = apply_v22_corrections(state, {"mass_kg": 1515.0})
        self.assertEqual(state["baseline"]["effective"]["mass_kg"], 1515.0)
        self.assertEqual(row, original)

    def test_correction_disposition_default_and_saved_option(self):
        state = normalize_v22_state({})
        self.assertEqual(state["baseline"]["correction_disposition"], "request_only")
        state["baseline"]["correction_disposition"] = "save_as_new_baseline"
        state = normalize_v22_state(state)
        self.assertEqual(state["baseline"]["correction_disposition"], "save_as_new_baseline")

    def test_proposal_matrix_preserves_ids_clears_incompatible_inputs_and_stales_preview(self):
        state = create_v22_state()
        state["preview"] = {"status": "draft_built", "fingerprint": "abc", "result": {"ok": True}}
        state["proposals"][0]["domains"]["aero"] = {"proposal_type": "AERO_ABSOLUTE_CDA", "selection_mode": "Absolute CdA"}
        state["proposals"][0]["inputs"]["aero"] = {"cda_m2": 0.7}
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "name": "One", "walk_from": "baseline", "aero": "Delta CdA"},
                {"proposal_id": "requested_2", "name": "Two", "walk_from": "requested_1", "mass": "Custom test mass"},
            ],
        )
        self.assertEqual([p["proposal_id"] for p in state["proposals"]], ["requested_1", "requested_2"])
        self.assertNotIn("aero", state["proposals"][0]["inputs"])
        self.assertEqual(state["preview"]["status"], "stale")
        self.assertEqual(state["save"]["status"], "pending")

    def test_preview_stales_after_correction(self):
        state = apply_v22_baseline(create_v22_state(), {"id": 9, "mass_kg": 1400.0})
        state["preview"] = {"status": "draft_built", "fingerprint": "abc", "result": {"ok": True}}
        state = apply_v22_corrections(state, {"mass_kg": 0})
        self.assertEqual(state["preview"]["status"], "stale")

    def test_canonical_draft_preserves_zero_blank_walk_from_and_mass_for_all_proposals(self):
        state = apply_v22_baseline(create_v22_state(), {"id": 11, "mass_kg": 1400.0, "trans_A_coef_N": 4.0})
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Custom test mass", "transmission": "Delta ABC"},
                {"proposal_id": "requested_2", "walk_from": "requested_1", "mass": "Inherit", "aero": "Absolute CdA"},
            ],
        )
        state["proposals"][0]["inputs"]["transmission"] = {"trans_A_coef_N": 0, "trans_B_coef_Npkph": "", "trans_C_coef_Npkph2": 0.001}
        state["proposals"][1]["inputs"]["aero"] = {"cda_m2": ""}
        draft = build_v22_canonical_request_draft(state)
        self.assertEqual(draft["schema_version"], "0.1")
        self.assertEqual(draft["proposals"][0]["domain_requests"]["transmission"]["raw_values"]["trans_A_coef_N"], 0)
        self.assertEqual(draft["proposals"][1]["domain_requests"]["aero"]["raw_values"]["cda_m2"], "")
        self.assertEqual(draft["proposals"][1]["walk_from"]["proposal_id"], "requested_1")
        self.assertEqual(len([p["domain_requests"]["mass"] for p in draft["proposals"]]), 2)

    def test_apply_domain_inputs_preserves_blank_as_absence_and_zero_as_explicit(self):
        state = apply_v22_baseline(create_v22_state(), {"id": 12, "mass_kg": 1400.0, "trans_A_coef_N": 4.0})
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Custom test mass", "transmission": "Delta ABC"},
                {"proposal_id": "requested_2", "walk_from": "requested_1"},
            ],
        )
        state = apply_v22_domain_inputs(
            state,
            "transmission",
            {
                "requested_1": {"delta_A": 0, "delta_B": "", "delta_C": 0.001},
            },
        )
        state = apply_v22_domain_inputs(
            state,
            "mass",
            {
                "requested_1": {"test_mass_kg": ""},
            },
        )
        draft = build_v22_canonical_request_draft(state)

        self.assertEqual(state["proposals"][0]["inputs"]["transmission"]["delta_A"], 0)
        self.assertNotIn("mass", state["proposals"][0]["inputs"])
        self.assertEqual(draft["proposals"][0]["domain_requests"]["transmission"]["raw_values"]["delta_A"], 0)
        self.assertEqual(draft["proposals"][0]["domain_requests"]["mass"]["raw_values"], {})

    def test_apply_domain_inputs_increments_only_target_domain_revision_once(self):
        state = apply_v22_baseline(create_v22_state(), {"id": 13, "mass_kg": 1400.0, "cda_m2": 0.62})
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Custom test mass", "aero": "Absolute CdA"},
                {"proposal_id": "requested_2", "walk_from": "requested_1"},
            ],
        )

        next_state = apply_v22_domain_inputs(state, "mass", {"requested_1": {"test_mass_kg": 1800.0}})

        self.assertEqual(next_state["domain_input_state"]["mass"]["revision"], 1)
        self.assertEqual(next_state["domain_input_state"]["aero"]["revision"], 0)
        self.assertEqual(next_state["domain_input_state"]["tire"]["revision"], 0)

    def test_apply_domain_inputs_canonicalizes_epa_curb_mass_under_mass_kg(self):
        state = apply_v22_baseline(create_v22_state(), {"id": 14, "mass_kg": 1500.0, "inertia_class": 1644.0})
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Curb mass -> EPA TWC"},
                {"proposal_id": "requested_2", "walk_from": "requested_1"},
            ],
        )

        next_state = apply_v22_domain_inputs(
            state,
            "mass",
            {
                "requested_1": {"target_curb_mass_kg": 1310.0, "target_mass_kg": 1644.0},
            },
        )

        self.assertEqual(next_state["proposals"][0]["inputs"]["mass"], {"mass_kg": 1310.0})
        self.assertNotIn("mass", next_state["proposals"][1]["inputs"])

    def test_mass_labels_hide_legacy_epa_status_but_keep_new_visible_taxonomy(self):
        labels = proposal_type_labels_by_domain()["mass"]

        self.assertEqual(
            labels,
            [
                "Inherit",
                "Curb mass → EPA TWC",
                "TWC shift / target class",
                "Performance loaded mass",
                "WLTP mass line",
                "GVWR loaded mass",
                "GCWR / trailer mass",
                "Custom test mass",
            ],
        )

    def test_baseline_mass_review_suggests_twc_without_overwriting_inertia(self):
        state = apply_v22_baseline(
            create_v22_state(),
            {"id": 21, "legislation": "EPA", "mass_kg": 1480.0, "inertia_class": 1758.0},
        )

        review = resolve_v22_baseline_mass_review(state)

        self.assertEqual(review["baseline_mass_review_status"], "Review")
        self.assertEqual(review["baseline_mass_suggested_inertia_class"], 1588.0)
        self.assertEqual(review["baseline_mass_target_twc_interval"], "(1423, 1480] kg")
        self.assertEqual(state["baseline"]["effective"]["inertia_class"], 1758.0)

    def test_new_test_baseline_uses_input_lane_and_preserves_explicit_zero(self):
        state = apply_v22_new_test_baseline(
            create_v22_state(),
            {
                "A": 120.0,
                "B": 0.0,
                "C": 0.008,
                "test_mass_kg": 1600.0,
                "legislation": "EPA",
                "cycle_name": "FTP75_HWFET",
                "notes": "synthetic new test",
            },
        )

        self.assertEqual(state["baseline"]["source_type"], "NEW_TEST")
        self.assertEqual(state["baseline"]["printed"], {})
        self.assertEqual(state["baseline"]["corrections"]["B"], 0.0)
        self.assertEqual(state["baseline"]["effective"]["abc_total_source_ui"], "From test coastdown")
        self.assertEqual(state["baseline"]["effective"]["test_mass_basis"], "EPA_INERTIA_CLASS")
        self.assertEqual(state["baseline"]["effective"]["inertia_class"], 1600.0)
        self.assertIsNone(state["baseline"]["effective"].get("mass_kg"))

        draft = build_v22_canonical_request_draft(state)
        self.assertEqual(draft["baseline_source_type"], "NEW_TEST")
        self.assertEqual(draft["baseline_source_snapshot"]["baseline_source_type"], "NEW_TEST")
        self.assertEqual(draft["baseline_corrections"]["B"], 0.0)

    def test_new_test_wltp_preserves_direct_test_mass_without_epa_mapping(self):
        payload = build_new_test_canonical_baseline(
            {
                "A": 120.0,
                "B": 0.02,
                "C": 0.008,
                "test_mass_kg": 1600.0,
                "legislation": "WLTP",
                "cycle_name": "WLTP_Class3ab",
            }
        )

        self.assertEqual(payload["corrections"]["test_mass_basis"], "PHYSICAL_TEST_MASS")
        self.assertNotIn("inertia_class", payload["corrections"])
        self.assertEqual(payload["source_snapshot"]["baseline_source_type"], "NEW_TEST")


if __name__ == "__main__":
    unittest.main()
