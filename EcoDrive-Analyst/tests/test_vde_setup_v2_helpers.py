import inspect
import unittest

import streamlit as st

from src.vde_app.components.vde_setup import (
    V21_DOMAIN_SCHEMAS,
    V21_REQUEST_DRAFT_REPORT_BYTES_KEY,
    V21_REQUEST_RESOLUTION_HASH_KEY,
    V21_REQUEST_RESOLUTION_STATE_KEY,
    V21_REQUEST_RESOLUTION_STALE_KEY,
    V21_REQUEST_SAVE_RESULT_KEY,
    V21_REQUEST_SAVED_REPORT_BYTES_KEY,
    V21_STATUS,
    VDE_WORKBOOK_V21_DOMAINS,
    VDE_WORKBOOK_V21_MATRIX_SELECTIONS,
    _v2_allowed_walk_from_ids,
    _v2_apply_mass_intention,
    _v2_apply_selected_baseline_row,
    _v2_set_state,
    _v2_field_config,
    _v2_reindex_walked_columns,
    _v21_advanced_fields_for_proposal,
    _v21_active_domain_proposals,
    _v21_apply_proposal_selection_changes,
    _v21_compact_fields_for_proposal,
    _v21_component_delta_from_absolute,
    _v21_clear_request_runtime_state,
    _v21_detail_fields_for_domain,
    _v21_detail_field_editable,
    _v21_detail_widget_key,
    _v21_apply_proposals_to_effective,
    _v21_calculated_detail_raw_value,
    _v21_field_display_state,
    _v21_local_delta_note,
    _v21_detail_fields_for_type,
    _v21_direct_fields_from_proposal,
    _v21_ensure_workbook_state,
    _v21_effective_proposal_label,
    _v21_display_column_label,
    _v21_reference_fields_for_proposal,
    _v21_reference_raw_value,
    _v21_request_flow_status_payloads,
    _v21_set_reference_override_values,
    _v21_get_direct_proposal,
    _v21_has_direct_proposal,
    _v21_proposal_type_label,
    _v21_proposals,
    _v21_proposal_type_for_cell,
    _v21_proposal_select_label,
    _v21_proposal_select_options,
    _v21_preview_rows,
    _v21_request_column_labels,
    _v21_request_preview_rows,
    _v21_render_preview_save,
    _v21_render_request_resolution_preview,
    _v21_render_request_review_save,
    _v21_review_status,
    _v21_rollup_statuses,
    _v21_set_baseline_printed_override_values,
    _v21_summary_text,
    _v21_transition_proposal_details,
    _v21_validate_proposal_details,
    parse_vde_cell_value,
)


class TestVdeSetupV2Helpers(unittest.TestCase):
    def test_v21_preview_save_route_uses_only_canonical_request_flow(self):
        source = inspect.getsource(_v21_render_preview_save)
        self.assertIn("_v21_render_request_resolution_preview", source)
        self.assertIn("_v21_render_request_review_save", source)
        self.assertNotIn("save_vde_setup_result", source)
        self.assertNotIn("Save selected column", source)
        self.assertNotIn("Save All", source)
        self.assertNotIn("Compute Preview All", source)
        self.assertNotIn("_v2_cached_preview", source)
        self.assertNotIn("_v2_preview", source)
        self.assertNotIn("_v21_save_plan_payload", source)

    def test_v21_has_single_validate_preview_and_review_save_flow(self):
        preview_source = inspect.getsource(_v21_render_request_resolution_preview)
        route_source = inspect.getsource(_v21_render_preview_save)
        self.assertEqual(preview_source.count('"Validate & Preview"'), 1)
        self.assertEqual(route_source.count("_v21_render_request_review_save"), 1)
        self.assertIn("execute_vde_request_save_plan", inspect.getsource(_v21_render_request_review_save))
        self.assertNotIn("save_vde_setup_result", inspect.getsource(_v21_render_request_review_save))

    def test_v21_display_labels_use_requested_for_generic_walked_ids(self):
        self.assertEqual(_v21_display_column_label("baseline"), "Baseline")
        self.assertEqual(_v21_display_column_label("baseline", baseline_printed=True), "Baseline / Printed")
        self.assertEqual(_v21_display_column_label("walked_1"), "Requested #1")
        self.assertEqual(_v21_display_column_label("walked_42"), "Requested #42")

    def test_v21_request_flow_status_reports_stale_preview(self):
        state = {
            "metadata": {
                "line_source": "Existing VDE DB",
                "selected_baseline_vde_id": 1001,
            },
            "rows": [{"id": 1001, "make": "OEM", "model": "ABC", "year": 2026}],
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
            ],
            "columns": {
                "baseline": {"line_source": "Existing VDE DB", "selected_vde_id": 1001, "direct": {}},
                "walked_1": {"walk_from": "baseline", "direct": {}},
            },
            "proposals": {"walked_1": {"mass": {"id": "prop_1", "proposal_type": "CUSTOM_MASS"}}},
        }
        statuses = _v21_request_flow_status_payloads(
            state,
            {
                V21_REQUEST_RESOLUTION_STATE_KEY: {"proposal_results": []},
                V21_REQUEST_RESOLUTION_HASH_KEY: "older-preview",
            },
        )
        self.assertEqual(statuses["Baseline"][0], "Loaded")
        self.assertEqual(statuses["Request"][0], "Defined")
        self.assertEqual(statuses["Preview"][0], "Stale")
        self.assertEqual(statuses["Save"][0], "Pending")

    def test_v21_request_runtime_clear_removes_preview_save_and_reports(self):
        st.session_state[V21_REQUEST_RESOLUTION_STATE_KEY] = {"status": "OK"}
        st.session_state[V21_REQUEST_RESOLUTION_HASH_KEY] = "abc"
        st.session_state[V21_REQUEST_RESOLUTION_STALE_KEY] = True
        st.session_state[V21_REQUEST_SAVE_RESULT_KEY] = {"status": "success"}
        st.session_state[V21_REQUEST_DRAFT_REPORT_BYTES_KEY] = b"draft"
        st.session_state[V21_REQUEST_SAVED_REPORT_BYTES_KEY] = b"saved"
        _v21_clear_request_runtime_state(clear_resolution=True)
        for key in (
            V21_REQUEST_RESOLUTION_STATE_KEY,
            V21_REQUEST_RESOLUTION_HASH_KEY,
            V21_REQUEST_RESOLUTION_STALE_KEY,
            V21_REQUEST_SAVE_RESULT_KEY,
            V21_REQUEST_DRAFT_REPORT_BYTES_KEY,
            V21_REQUEST_SAVED_REPORT_BYTES_KEY,
        ):
            self.assertNotIn(key, st.session_state)

    def test_parse_vde_cell_value_accepts_comma_decimal(self):
        parsed = parse_vde_cell_value("1735,5", expected_type="mass")
        self.assertEqual(parsed["parse_status"], "numeric")
        self.assertEqual(parsed["parsed_value"], 1735.5)

    def test_parse_vde_cell_value_keeps_special_tokens(self):
        parsed = parse_vde_cell_value("not used", expected_type="float")
        self.assertEqual(parsed["parse_status"], "token")
        self.assertEqual(parsed["parsed_value"], "not_used")

    def test_apply_mass_intention_perf_curb_100kg(self):
        effective = {"mass_intention": "PERF_CURB_100KG", "curb_mass_kg": 1500.0}
        _v2_apply_mass_intention(effective)
        self.assertEqual(effective["effective_test_mass_kg"], 1600.0)
        self.assertEqual(effective["vde_mass_basis"], "CURB_PLUS_DRIVER")
        self.assertEqual(effective["mass_rule_status"], "OK")

    def test_apply_mass_intention_gcwr_requires_trailer_weight(self):
        effective = {"mass_intention": "GCWR", "GCWR_kg": 3500.0}
        _v2_apply_mass_intention(effective)
        self.assertEqual(effective["mass_rule_status"], "Missing")
        self.assertIn("trailer_weight_kg", effective["mass_rule_notes"])

    def test_apply_mass_intention_epa_status_falls_back_to_curb_plus_300lb(self):
        effective = {"legislation": "EPA", "mass_intention": "EPA_STATUS", "curb_mass_kg": 1800.0}
        _v2_apply_mass_intention(effective)
        self.assertAlmostEqual(effective["effective_test_mass_kg"], 2041.0)
        self.assertEqual(effective["vde_mass_basis"], "EPA_INERTIA_CLASS")
        self.assertEqual(effective["fuelcons_mass_basis"], "TWC")
        self.assertEqual(effective["mass_rule_status"], "Review")

    def test_apply_mass_intention_gcwr_adds_trailer_abc_when_available(self):
        effective = {
            "mass_intention": "GCWR",
            "curb_mass_kg": 2000.0,
            "GVWR_kg": 2600.0,
            "GCWR_kg": 4200.0,
            "trailer_weight_kg": 1500.0,
            "trailer_roadload_source": "Manual ABC",
            "trailer_A": 10.0,
            "trailer_B": 1.5,
            "trailer_C": 0.1,
            "ABC_TOTAL_A": 100.0,
            "ABC_TOTAL_B": 20.0,
            "ABC_TOTAL_C": 2.0,
        }
        _v2_apply_mass_intention(effective)
        self.assertEqual(effective["mass_rule_status"], "Review")
        self.assertEqual(effective["vehicle_mass_at_gcwr"], 2700.0)
        self.assertEqual(effective["trailer_roadload_status"], "OK")
        self.assertEqual(effective["ABC_TOTAL_A"], 110.0)
        self.assertEqual(effective["ABC_TOTAL_B"], 21.5)
        self.assertEqual(effective["ABC_TOTAL_C"], 2.1)

    def test_apply_selected_baseline_row_resets_baseline_direct_state(self):
        state = {
            "metadata": {"line_source": "Existing VDE DB"},
            "columns": {"baseline": {"selected_vde_id": 1, "direct": {"curb_mass_kg": 9999}}},
            "preview_cache": {"baseline": {"ok": True}},
        }
        row = {
            "id": 42,
            "legislation": "EPA",
            "year": 2027,
            "make": "AUDI",
            "model": "TEST",
            "cycle_name": "EPA",
            "proposal": "Baseline proposal",
        }
        updated = _v2_apply_selected_baseline_row(state, row)
        self.assertEqual(updated["metadata"]["selected_baseline_vde_id"], 42)
        self.assertEqual(updated["columns"]["baseline"]["selected_vde_id"], 42)
        self.assertEqual(updated["columns"]["baseline"]["direct"], {})
        self.assertNotIn("baseline", updated["preview_cache"])

    def test_reindex_walked_columns_rewires_walk_from(self):
        scenarios = [
            {"key": "baseline", "label": "Baseline", "role": "baseline"},
            {"key": "walked_1", "label": "Walked #1", "role": "walked"},
            {"key": "walked_3", "label": "Walked #3", "role": "walked"},
        ]
        columns = {
            "baseline": {"direct": {}},
            "walked_1": {"walk_from": "baseline", "direct": {"proposal_direct": "Tire"}},
            "walked_3": {"walk_from": "walked_1", "direct": {"proposal_direct": "Aero"}},
        }
        new_scenarios, new_columns = _v2_reindex_walked_columns(scenarios, columns)
        self.assertEqual([item["key"] for item in new_scenarios], ["baseline", "walked_1", "walked_2"])
        self.assertEqual(new_columns["walked_2"]["walk_from"], "walked_1")

    def test_walk_from_allows_only_prior_columns(self):
        state = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
                {"key": "walked_2", "label": "Walked #2", "role": "walked"},
                {"key": "walked_3", "label": "Walked #3", "role": "walked"},
            ]
        }
        self.assertEqual(_v2_allowed_walk_from_ids("walked_1", state), ["baseline"])
        self.assertEqual(_v2_allowed_walk_from_ids("walked_2", state), ["baseline", "walked_1"])
        self.assertEqual(_v2_allowed_walk_from_ids("walked_3", state), ["baseline", "walked_1", "walked_2"])

    def test_walk_from_field_config_excludes_current_and_future_columns(self):
        state = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
                {"key": "walked_2", "label": "Walked #2", "role": "walked"},
                {"key": "walked_3", "label": "Walked #3", "role": "walked"},
            ]
        }
        config = _v2_field_config(
            {"id": "walk_from", "label": "Walk From", "kind": "text"},
            section_key="Scenario Workbook",
            scenario_key="walked_2",
            state=state,
        )
        self.assertEqual(config["options"], ["", "Baseline", "Walked #1"])

    def test_v21_mass_aero_proposal_maps_custom_mass(self):
        direct = _v21_direct_fields_from_proposal(
            "mass",
            {"proposal_type": "CUSTOM_MASS", "details": {"test_mass_kg": "1735,5"}},
        )
        self.assertEqual(direct["mass_intention"], "CUSTOM")
        self.assertEqual(direct["test_mass_kg"], "1735,5")

    def test_v21_mass_twc_shift_maps_to_epa_plus_1_twc(self):
        direct = _v21_direct_fields_from_proposal(
            "mass",
            {"proposal_type": "MASS_TWC_SHIFT", "details": {"target_mass_kg": "1900"}},
        )
        self.assertEqual(direct["mass_intention"], "EPA_PLUS_1_TWC")
        self.assertEqual(direct["prep_inertia_class"], "1900")

    def test_v21_domains_split_mass_and_aero(self):
        self.assertIn("mass", VDE_WORKBOOK_V21_DOMAINS)
        self.assertIn("aero", VDE_WORKBOOK_V21_DOMAINS)
        self.assertEqual(VDE_WORKBOOK_V21_DOMAINS["mass"]["label"], "Mass proposal")
        self.assertEqual(VDE_WORKBOOK_V21_DOMAINS["aero"]["label"], "Aero proposal")

    def test_v21_domain_schemas_expose_non_inherit_types_and_statuses(self):
        self.assertEqual(
            V21_STATUS,
            ("Inherited", "OK", "Missing", "Review", "Invalid"),
        )
        self.assertNotIn("INHERIT", V21_DOMAIN_SCHEMAS["mass"]["proposal_types"])
        self.assertIn("CUSTOM_MASS", V21_DOMAIN_SCHEMAS["mass"]["proposal_types"])
        self.assertIn("new_CdA", V21_DOMAIN_SCHEMAS["aero"]["detail_fields"])

    def test_v21_apply_proposal_updates_effective_label_and_fields(self):
        effective = {"proposal_effective": "Baseline proposal"}
        state = {
            "proposals": {
                "walked_1": {
                    "tire": {
                        "proposal_type": "TIRE_DB_LOOKUP",
                        "label": "LRR Tire",
                        "details": {"tire_code": "ABC"},
                    }
                }
            }
        }
        _v21_apply_proposals_to_effective(effective, "walked_1", state)
        self.assertEqual(effective["tire_mode"], "TIRE_DB_LOOKUP")
        self.assertEqual(effective["tire_code"], "ABC")
        self.assertEqual(effective["proposal_direct"], "LRR Tire")
        self.assertEqual(effective["proposal_effective"], "Baseline proposal + LRR Tire")

    def test_v21_ensure_workbook_state_populates_domains_per_walked_column(self):
        state = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
                {"key": "walked_2", "label": "Walked #2", "role": "walked"},
            ],
            "columns": {
                "baseline": {"direct": {}},
                "walked_1": {"walk_from": "baseline", "direct": {}},
                "walked_2": {"walk_from": "walked_9", "direct": {}},
            },
            "proposals": {
                "walked_1": {
                    "aero": {
                        "proposal_type": "AERO_DELTA_CDA",
                        "label": "Aero tweak",
                        "details": {"delta_CdA": "0.01"},
                    }
                }
            },
            "rows": [],
            "metadata": {},
        }
        normalized = _v21_ensure_workbook_state(state)
        self.assertEqual(normalized["columns"]["baseline"]["kind"], "baseline")
        self.assertEqual(normalized["columns"]["baseline"]["label"], "Baseline")
        self.assertIsNone(normalized["columns"]["baseline"]["walk_from"])
        self.assertEqual(normalized["columns"]["baseline"]["domains"], {})
        self.assertEqual(normalized["columns"]["walked_1"]["kind"], "walked")
        self.assertEqual(normalized["columns"]["walked_1"]["domains"]["aero"]["mode"], "direct")
        self.assertEqual(normalized["columns"]["walked_1"]["domains"]["aero"]["proposal_type"], "AERO_DELTA_CDA")
        self.assertEqual(normalized["columns"]["walked_1"]["domains"]["aero"]["status"], "OK")
        self.assertEqual(normalized["columns"]["walked_2"]["walk_from"], "walked_1")
        self.assertEqual(normalized["columns"]["walked_2"]["domains"]["tire"]["mode"], "inherited")
        self.assertEqual(normalized["columns"]["walked_2"]["domains"]["tire"]["status"], "Inherited")

    def test_v21_get_direct_proposal_reads_column_domains_even_without_top_level_proposals(self):
        state = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
            ],
            "columns": {
                "baseline": {"direct": {}, "domains": {}},
                "walked_1": {
                    "walk_from": "baseline",
                    "direct": {},
                    "domains": {
                        "aero": {
                            "mode": "direct",
                            "id": "prop_9",
                            "domain": "aero",
                            "proposal_type": "AERO_DELTA_CDA",
                            "label": "CdA tweak",
                            "details": {"delta_CdA": "0.01"},
                            "status": "OK",
                            "notes": [],
                        }
                    },
                },
            },
            "proposals": {},
            "rows": [],
            "metadata": {},
        }
        proposal = _v21_get_direct_proposal("walked_1", "aero", state)
        self.assertEqual(proposal["id"], "prop_9")
        self.assertEqual(proposal["proposal_type"], "AERO_DELTA_CDA")
        self.assertEqual(proposal["label"], "CdA tweak")

    def test_v21_proposals_reads_domain_only_direct_proposals(self):
        state = {
            "columns": {
                "baseline": {"direct": {}, "domains": {}},
                "walked_1": {
                    "walk_from": "baseline",
                    "direct": {},
                    "domains": {
                        "tire": {
                            "mode": "direct",
                            "id": "prop_12",
                            "domain": "tire",
                            "proposal_type": "TIRE_DB_LOOKUP",
                            "label": "TPS tire",
                            "details": {"new_tire_code": "TPS123"},
                            "status": "Review",
                            "notes": ["supplier pending"],
                        }
                    },
                },
            },
            "proposals": {},
            "proposal_seq": 0,
        }
        proposals = _v21_proposals(state)
        self.assertEqual(proposals["walked_1"]["tire"]["id"], "prop_12")
        self.assertEqual(proposals["walked_1"]["tire"]["proposal_type"], "TIRE_DB_LOOKUP")
        self.assertEqual(proposals["walked_1"]["tire"]["details"]["new_tire_code"], "TPS123")
        self.assertEqual(proposals["walked_1"]["tire"]["notes"], ["supplier pending"])
        self.assertEqual(state["proposal_seq"], 12)

    def test_v21_proposals_prefers_domain_state_over_stale_top_level_payload(self):
        state = {
            "columns": {
                "baseline": {"direct": {}, "domains": {}},
                "walked_1": {
                    "walk_from": "baseline",
                    "direct": {},
                    "domains": {
                        "aero": {
                            "mode": "direct",
                            "id": "prop_8",
                            "domain": "aero",
                            "proposal_type": "AERO_ABSOLUTE_CDA",
                            "label": "Final CdA",
                            "details": {"new_CdA": "0.71"},
                            "status": "OK",
                            "notes": [],
                        }
                    },
                },
            },
            "proposals": {
                "walked_1": {
                    "aero": {
                        "id": "prop_3",
                        "proposal_type": "AERO_DELTA_CDA",
                        "label": "Old CdA",
                        "details": {"delta_CdA": "0.03"},
                        "status": "Draft",
                    }
                }
            },
            "proposal_seq": 3,
        }
        proposals = _v21_proposals(state)
        self.assertEqual(proposals["walked_1"]["aero"]["id"], "prop_8")
        self.assertEqual(proposals["walked_1"]["aero"]["proposal_type"], "AERO_ABSOLUTE_CDA")
        self.assertEqual(proposals["walked_1"]["aero"]["details"], {"new_CdA": "0.71"})
        self.assertEqual(state["proposal_seq"], 8)

    def test_v21_rollup_statuses_surfaces_worst_problem_first(self):
        status, detail = _v21_rollup_statuses(
            [
                ("Baseline", "OK", "Loaded"),
                ("Walked #1", "Review", "Needs check"),
                ("Walked #2", "Missing", "Needs data"),
            ]
        )
        self.assertEqual(status, "Missing")
        self.assertIn("Walked #2: Missing", detail)
        self.assertIn("Walked #1: Review", detail)

    def test_v21_preview_rows_include_warning_and_delta_fields(self):
        state = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
            ],
            "columns": {
                "baseline": {"direct": {}},
                "walked_1": {"walk_from": "baseline", "direct": {"proposal_direct": "Tire swap"}},
            },
            "metadata": {"line_source": "New test ABC_TOTAL"},
            "preview_cache": {
                "baseline": {
                    "ok": True,
                    "abc_total": {"A": 100.0, "B": 10.0, "C": 1.0},
                    "vde_net": {"mj_per_km": 1.2},
                    "vde_total": {"mj_per_km": 1.4},
                },
                "walked_1": {
                    "ok": False,
                    "abc_total": {"A": 101.0, "B": 10.5, "C": 1.1},
                    "vde_net": {"mj_per_km": 1.35},
                    "vde_total": {"mj_per_km": 1.55},
                    "warnings": ["Need tire validation"],
                },
            },
        }
        rows = _v21_preview_rows(state, state["preview_cache"])
        warnings_row = next(row for row in rows if row["field"] == "Warnings")
        delta_row = next(row for row in rows if row["field"] == "Delta vs Baseline")
        save_row = next(row for row in rows if row["field"] == "Save status")
        self.assertEqual(warnings_row["Requested #1"], "Need tire validation")
        self.assertIn("MJ/km", delta_row["Requested #1"])
        self.assertEqual(save_row["Baseline / Printed"], "Ready")
        self.assertEqual(save_row["Requested #1"], "Pending")

    def test_v21_request_column_labels_use_ppe_names(self):
        labels = _v21_request_column_labels(
            {
                "scenarios": [
                    {"key": "baseline", "label": "Baseline", "role": "baseline"},
                    {"key": "walked_1", "label": "Walked #1", "role": "walked"},
                    {"key": "walked_2", "label": "Walked #2", "role": "walked"},
                ]
            }
        )
        self.assertEqual(labels["baseline"], "Baseline / Printed")
        self.assertEqual(labels["walked_1"], "Requested #1")
        self.assertEqual(labels["walked_2"], "Requested #2")

    def test_v21_mass_twc_shift_label_is_generated(self):
        label = _v21_effective_proposal_label(
            {
                "proposal_type": "MASS_TWC_SHIFT",
                "label": "",
                "details": {
                    "twc_shift_steps": "+1",
                    "twc_target_side": "Low",
                },
            }
        )
        self.assertEqual(label, "+1 TWC Low")

    def test_v21_proposal_type_label_is_humanized(self):
        self.assertEqual(_v21_proposal_type_label("GVWR"), "GVWR loaded mass")
        self.assertEqual(_v21_proposal_type_label("TIRE_DB_LOOKUP"), "Tire DB lookup")

    def test_v21_has_direct_proposal_reports_checkbox_state(self):
        state = {
            "proposals": {
                "walked_1": {
                    "mass": {
                        "id": "prop_1",
                        "proposal_type": "GVWR",
                        "label": "",
                        "details": {},
                    }
                }
            }
        }
        self.assertTrue(_v21_has_direct_proposal("walked_1", "mass", state))
        self.assertFalse(_v21_has_direct_proposal("walked_1", "tire", state))

    def test_v21_tire_domain_types_are_refined(self):
        self.assertEqual(
            [item for item in VDE_WORKBOOK_V21_DOMAINS["tire"]["types"] if item != "INHERIT"],
            ["TIRE_DB_LOOKUP", "TIRE_SMERF_RRC_CHANGE"],
        )

    def test_v21_detail_fields_are_filtered_by_proposal_type(self):
        fields = _v21_detail_fields_for_type("tire", "TIRE_SMERF_RRC_CHANGE")
        self.assertIn("delta_SMERF_optional", fields)
        self.assertIn("delta_RRC_optional", fields)
        self.assertNotIn("new_tire_code", fields)

    def test_v21_active_domain_proposals_collect_multiple_columns(self):
        state = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
                {"key": "walked_2", "label": "Walked #2", "role": "walked"},
            ],
            "proposals": {
                "walked_1": {"tire": {"proposal_type": "TIRE_DB_LOOKUP", "details": {}}},
                "walked_2": {"tire": {"proposal_type": "TIRE_SMERF_RRC_CHANGE", "details": {}}},
            },
        }
        active = _v21_active_domain_proposals("tire", state)
        self.assertEqual(set(active), {"walked_1", "walked_2"})

    def test_v21_detail_fields_for_domain_unions_active_types(self):
        active = {
            "walked_1": {"proposal_type": "TIRE_DB_LOOKUP", "details": {}},
            "walked_2": {"proposal_type": "TIRE_SMERF_RRC_CHANGE", "details": {}},
        }
        fields = _v21_detail_fields_for_domain("tire", active)
        self.assertEqual(fields[:3], ["proposal_type", "proposal_label", "status"])
        self.assertIn("new_tire_code", fields)
        self.assertIn("delta_SMERF_optional", fields)

    def test_v21_tire_db_compact_fields_hide_baseline_code_when_reference_exists(self):
        fields = _v21_compact_fields_for_proposal(
            "tire",
            "TIRE_DB_LOOKUP",
            {},
            {"baseline_tire_code": "BASE123"},
        )
        self.assertIn("new_tire_code", fields)
        self.assertNotIn("baseline_tire_code", fields)

    def test_v21_tire_db_advanced_fields_keep_baseline_code_when_reference_exists(self):
        fields = _v21_advanced_fields_for_proposal(
            "tire",
            "TIRE_DB_LOOKUP",
            {},
            {"baseline_tire_code": "BASE123"},
        )
        self.assertIn("baseline_tire_code", fields)

    def test_v21_smerf_compact_fields_drop_load_fields_for_direct_delta_rrc(self):
        fields = _v21_compact_fields_for_proposal(
            "tire",
            "TIRE_SMERF_RRC_CHANGE",
            {"delta_RRC_optional": "0.2"},
            {},
        )
        self.assertIn("delta_RRC_optional", fields)
        self.assertNotIn("tire_load_mass_used_kg", fields)

    def test_v21_mass_epa_compact_fields_use_canonical_ids(self):
        fields = _v21_compact_fields_for_proposal("mass", "EPA_STATUS", {}, {})
        self.assertIn("mass_kg", fields)
        self.assertIn("test_mass_kg", fields)
        self.assertIn("test_mass_basis", fields)
        self.assertNotIn("curb_mass_kg", fields)
        self.assertNotIn("effective_test_mass_kg", fields)

    def test_v21_wltp_mass_line_shows_only_selected_line(self):
        fields = _v21_compact_fields_for_proposal(
            "mass",
            "WLTP_MASS_LINE",
            {"line_type": "TMH", "test_mass_high_kg": "1800"},
            {},
        )
        self.assertIn("test_mass_high_kg", fields)
        self.assertNotIn("test_mass_low_kg", fields)

    def test_v21_tire_db_lookup_does_not_require_baseline_code_when_reference_exists(self):
        state = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
            ],
            "columns": {
                "baseline": {"direct": {}},
                "walked_1": {"walk_from": "baseline", "direct": {}},
            },
            "preview_cache": {
                "baseline": {"tire_code": "BASE123"},
            },
        }
        status, warnings, missing, _ = _v21_validate_proposal_details(
            "walked_1",
            "tire",
            "TIRE_DB_LOOKUP",
            {"new_tire_code": "NEW456"},
            state,
        )
        self.assertEqual(status, "OK")
        self.assertEqual(missing, [])
        self.assertEqual(warnings, [])

    def test_v21_negative_tire_improvement_is_review(self):
        state = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
            ],
            "columns": {
                "baseline": {"direct": {}},
                "walked_1": {"walk_from": "baseline", "direct": {}},
            },
            "preview_cache": {
                "baseline": {"tire_code": "BASE123"},
            },
        }
        status, warnings, missing, _ = _v21_validate_proposal_details(
            "walked_1",
            "tire",
            "TIRE_DB_LOOKUP",
            {"new_tire_code": "NEW456", "tire_improvement_pct": "-12"},
            state,
        )
        self.assertEqual(status, "Review")
        self.assertEqual(missing, [])
        self.assertIn("Negative tire improvement increases RR", warnings[0])

    def test_v21_detail_widget_key_uses_stable_domain_scenario_proposal_field_shape(self):
        key = _v21_detail_widget_key("tire", "walked_2", "front_pressure_psi", "prop_7")
        self.assertEqual(key, "v21_detail__tire__walked_2__prop_7__front_pressure_psi")

    def test_v21_legacy_mass_aero_is_normalized_to_aero(self):
        state = {
            "proposals": {
                "walked_1": {
                    "mass_aero": {
                        "proposal_type": "AERO_DELTA_CDA",
                        "label": "Aero tweak",
                        "details": {"delta_CdA": "0.01"},
                    }
                }
            }
        }
        proposals = _v21_proposals(state)
        self.assertIn("aero", proposals["walked_1"])
        self.assertEqual(proposals["walked_1"]["aero"]["domain"], "aero")

    def test_v21_review_status_reports_missing_required_details(self):
        state = {
            "proposals": {
                "walked_1": {
                    "mass_aero": {
                        "proposal_type": "GVWR",
                        "label": "GVWR run",
                        "details": {},
                    }
                }
            }
        }
        self.assertEqual(_v21_review_status("walked_1", state), "Missing")

    def test_v21_proposal_type_for_cell_defaults_to_inherit(self):
        self.assertEqual(_v21_proposal_type_for_cell("baseline", "tire", {}), "baseline")
        self.assertEqual(_v21_proposal_type_for_cell("walked_1", "tire", {"proposals": {}}), "inherit")

    def test_v21_proposal_select_options_start_with_inherited_source(self):
        state = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
            ],
            "columns": {
                "baseline": {"direct": {}},
                "walked_1": {"walk_from": "baseline", "direct": {}},
            },
        }
        options = _v21_proposal_select_options("walked_1", "tire", state)
        self.assertEqual(options[0], "inherit::Baseline")
        self.assertIn("TIRE_DB_LOOKUP", options)

    def test_v21_component_proposal_select_options_use_composite_matrix_entries(self):
        transmission_values = [item["value"] for item in VDE_WORKBOOK_V21_MATRIX_SELECTIONS["transmission"]]
        self.assertEqual(
            transmission_values,
            [
                "UPDATE_TRANS_DRAG_ABC__DELTA_ABC",
                "UPDATE_TRANS_DRAG_ABC__ABSOLUTE_ABC",
                "TRANS_LOSS_PCT",
            ],
        )
        self.assertNotIn("Baseline + new ABC", transmission_values)
        brake_values = [item["value"] for item in VDE_WORKBOOK_V21_MATRIX_SELECTIONS["brake"]]
        self.assertIn("BRAKE_DRAG_CHANGE__ABSOLUTE_ABC", brake_values)
        self.assertIn("BRAKE_DRAG_CHANGE__RESIDUAL_TORQUE", brake_values)
        self.assertEqual(
            [item["value"] for item in VDE_WORKBOOK_V21_MATRIX_SELECTIONS["axle_hubs"]],
            [
                "AXLE_HUB_DRAG_CHANGE__DELTA_ABC",
                "AXLE_HUB_DRAG_CHANGE__ABSOLUTE_ABC",
            ],
        )
        self.assertEqual(
            [item["value"] for item in VDE_WORKBOOK_V21_MATRIX_SELECTIONS["parasitic"]],
            [
                "PARASITIC_LOSS_CHANGE__DELTA_ABC",
                "PARASITIC_LOSS_CHANGE__ABSOLUTE_ABC",
            ],
        )

    def test_v21_proposal_select_label_formats_inherited_choice(self):
        self.assertEqual(_v21_proposal_select_label("inherit::Walked #1"), "↳ Inherited from Walked #1")
        self.assertEqual(_v21_proposal_select_label("TIRE_DB_LOOKUP"), "Tire DB lookup")
        self.assertEqual(
            _v21_proposal_select_label("UPDATE_TRANS_DRAG_ABC__ABSOLUTE_ABC"),
            "Update trans drag ABC - Absolute ABC",
        )

    def test_v21_transition_proposal_details_keeps_only_compatible_fields(self):
        transitioned = _v21_transition_proposal_details(
            "tire",
            "TIRE_DB_LOOKUP",
            "TIRE_SMERF_RRC_CHANGE",
            {
                "new_tire_code": "ABC",
                "front_pressure_psi": "36",
                "rear_pressure_psi": "34",
                "tire_improvement_pct": "8",
                "source": "Supplier",
            },
        )
        self.assertEqual(transitioned["front_pressure_psi"], "36")
        self.assertEqual(transitioned["rear_pressure_psi"], "34")
        self.assertEqual(transitioned["source"], "Supplier")
        self.assertNotIn("new_tire_code", transitioned)
        self.assertNotIn("tire_improvement_pct", transitioned)

    def test_v21_apply_proposal_selection_changes_creates_and_removes_direct_proposal(self):
        state = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
            ],
            "columns": {
                "baseline": {"direct": {}},
                "walked_1": {"walk_from": "baseline", "direct": {}},
            },
            "proposals": {},
            "proposal_seq": 0,
            "preview_cache": {},
        }
        _v2_set_state(state)
        created, updated, removed = _v21_apply_proposal_selection_changes(
            {("walked_1", "tire"): "TIRE_DB_LOOKUP"}
        )
        self.assertEqual((created, updated, removed), (1, 0, 0))
        self.assertEqual(_v21_proposal_type_for_cell("walked_1", "tire"), "TIRE_DB_LOOKUP")
        created_state = _v21_summary_text("walked_1", "tire")
        self.assertIn("Prop #1", created_state)

        created, updated, removed = _v21_apply_proposal_selection_changes(
            {("walked_1", "tire"): "inherit::Baseline"}
        )
        self.assertEqual((created, updated, removed), (0, 0, 1))
        self.assertEqual(_v21_proposal_type_for_cell("walked_1", "tire"), "inherit")

    def test_v21_apply_proposal_selection_changes_seeds_component_mode_details(self):
        state = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
            ],
            "columns": {
                "baseline": {"direct": {}},
                "walked_1": {"walk_from": "baseline", "direct": {}},
            },
            "proposals": {},
            "proposal_seq": 0,
            "preview_cache": {},
        }
        _v2_set_state(state)
        _v21_apply_proposal_selection_changes(
            {
                ("walked_1", "transmission"): "UPDATE_TRANS_DRAG_ABC__ABSOLUTE_ABC",
                ("walked_1", "brake"): "BRAKE_DRAG_CHANGE__RESIDUAL_TORQUE",
            }
        )
        proposals = _v21_proposals()
        self.assertEqual(
            proposals["walked_1"]["transmission"]["details"]["change_mode"],
            "Absolute ABC",
        )
        self.assertEqual(
            proposals["walked_1"]["brake"]["details"]["method"],
            "Residual torque",
        )

    def test_v21_apply_proposal_selection_changes_updates_domain_only_existing_proposal(self):
        state = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
            ],
            "columns": {
                "baseline": {"direct": {}, "domains": {}},
                "walked_1": {
                    "walk_from": "baseline",
                    "direct": {},
                    "domains": {
                        "tire": {
                            "mode": "direct",
                            "id": "prop_9",
                            "domain": "tire",
                            "proposal_type": "TIRE_DB_LOOKUP",
                            "label": "Old tire",
                            "details": {"new_tire_code": "OLD123"},
                            "status": "Draft",
                            "notes": [],
                        }
                    },
                },
            },
            "proposals": {},
            "proposal_seq": 0,
            "preview_cache": {},
            "rows": [],
            "metadata": {},
        }
        _v2_set_state(state)
        created, updated, removed = _v21_apply_proposal_selection_changes(
            {("walked_1", "tire"): "TIRE_SMERF_RRC_CHANGE"}
        )
        self.assertEqual((created, updated, removed), (0, 1, 0))
        proposal = _v21_get_direct_proposal("walked_1", "tire")
        self.assertEqual(proposal["id"], "prop_9")
        self.assertEqual(proposal["proposal_type"], "TIRE_SMERF_RRC_CHANGE")

    def test_v21_apply_proposal_selection_changes_removes_domain_only_existing_proposal(self):
        state = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
            ],
            "columns": {
                "baseline": {"direct": {}, "domains": {}},
                "walked_1": {
                    "walk_from": "baseline",
                    "direct": {},
                    "domains": {
                        "aero": {
                            "mode": "direct",
                            "id": "prop_4",
                            "domain": "aero",
                            "proposal_type": "AERO_DELTA_CDA",
                            "label": "CdA tweak",
                            "details": {"delta_CdA": "0.01"},
                            "status": "OK",
                            "notes": [],
                        }
                    },
                },
            },
            "proposals": {},
            "proposal_seq": 0,
            "preview_cache": {},
            "rows": [],
            "metadata": {},
        }
        _v2_set_state(state)
        created, updated, removed = _v21_apply_proposal_selection_changes(
            {("walked_1", "aero"): "inherit::Baseline"}
        )
        self.assertEqual((created, updated, removed), (0, 0, 1))
        self.assertEqual(_v21_get_direct_proposal("walked_1", "aero"), {})
        self.assertEqual(_v21_proposal_type_for_cell("walked_1", "aero"), "inherit")

    def test_v21_transmission_absolute_requires_manual_baseline_when_no_baseline_component_exists(self):
        state = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
            ],
            "columns": {
                "baseline": {"direct": {}},
                "walked_1": {"walk_from": "baseline", "direct": {}},
            },
            "preview_cache": {
                "baseline": {},
            },
        }
        status, warnings, missing, _ = _v21_validate_proposal_details(
            "walked_1",
            "transmission",
            "UPDATE_TRANS_DRAG_ABC",
            {
                "change_mode": "Absolute ABC",
                "new_trans_A": "10",
                "new_trans_B": "1.2",
                "new_trans_C": "0.03",
            },
            state,
        )
        self.assertEqual(status, "Missing")
        self.assertEqual(missing, ["baseline_trans_A", "baseline_trans_B", "baseline_trans_C"])
        self.assertEqual(warnings, [])

    def test_v21_transmission_absolute_manual_baseline_is_review(self):
        state = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
            ],
            "columns": {
                "baseline": {"direct": {}},
                "walked_1": {"walk_from": "baseline", "direct": {}},
            },
            "preview_cache": {
                "baseline": {},
            },
        }
        status, warnings, missing, _ = _v21_validate_proposal_details(
            "walked_1",
            "transmission",
            "UPDATE_TRANS_DRAG_ABC",
            {
                "change_mode": "Absolute ABC",
                "new_trans_A": "10",
                "new_trans_B": "1.2",
                "new_trans_C": "0.03",
                "baseline_trans_A": "8",
                "baseline_trans_B": "1.0",
                "baseline_trans_C": "0.01",
            },
            state,
        )
        self.assertEqual(status, "Review")
        self.assertEqual(missing, [])
        self.assertTrue(any("manual baseline override" in warning for warning in warnings))

    def test_v21_component_absolute_delta_uses_manual_reference(self):
        state = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
            ],
            "columns": {
                "baseline": {"direct": {}},
                "walked_1": {"walk_from": "baseline", "direct": {}},
            },
            "preview_cache": {
                "baseline": {},
            },
        }
        computed = _v21_component_delta_from_absolute(
            "transmission",
            "walked_1",
            {
                "change_mode": "Absolute ABC",
                "new_trans_A": "12",
                "new_trans_B": "2.5",
                "new_trans_C": "0.08",
                "baseline_component_reference_mode": "Enter manual baseline component ABC, do not update baseline",
                "baseline_trans_A": "10",
                "baseline_trans_B": "2.0",
                "baseline_trans_C": "0.05",
            },
            state,
        )
        self.assertEqual(
            computed,
            {
                "delta_A": 2.0,
                "delta_B": 0.5,
                "delta_C": 0.03,
            },
        )

    def test_v21_walked_2_absolute_cda_uses_walk_from_effective_state(self):
        state = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
                {"key": "walked_2", "label": "Walked #2", "role": "walked"},
            ],
            "columns": {
                "baseline": {"line_source": "New test ABC_TOTAL", "direct": {}},
                "walked_1": {"walk_from": "baseline", "direct": {}},
                "walked_2": {"walk_from": "walked_1", "direct": {}},
            },
            "proposals": {
                "walked_1": {
                    "aero": {
                        "proposal_type": "AERO_ABSOLUTE_CDA",
                        "label": "CdA 0.60",
                        "details": {
                            "new_CdA": "0.60",
                            "baseline_CdA": "0.50",
                        },
                    }
                }
            },
            "preview_cache": {},
        }
        delta = _v21_calculated_detail_raw_value(
            "walked_2",
            "aero",
            "AERO_ABSOLUTE_CDA",
            "delta_CdA",
            {"details": {"new_CdA": "0.70"}},
            state,
        )
        self.assertAlmostEqual(delta, 0.10, places=6)

    def test_v21_walked_2_absolute_cda_uses_baseline_when_walk_from_is_baseline(self):
        state = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_2", "label": "Walked #2", "role": "walked"},
            ],
            "columns": {
                "baseline": {"line_source": "New test ABC_TOTAL", "direct": {"CdA": "0.55"}},
                "walked_2": {"walk_from": "baseline", "direct": {}},
            },
            "preview_cache": {},
        }
        delta = _v21_calculated_detail_raw_value(
            "walked_2",
            "aero",
            "AERO_ABSOLUTE_CDA",
            "delta_CdA",
            {"details": {"new_CdA": "0.70"}},
            state,
        )
        self.assertAlmostEqual(delta, 0.15, places=6)

    def test_v21_local_delta_note_mentions_walk_from_source(self):
        state = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
                {"key": "walked_2", "label": "Walked #2", "role": "walked"},
            ],
            "columns": {
                "baseline": {"direct": {}},
                "walked_1": {"walk_from": "baseline", "direct": {}},
                "walked_2": {"walk_from": "walked_1", "direct": {}},
            },
        }
        self.assertEqual(
            _v21_local_delta_note("walked_2", "aero", "AERO_ABSOLUTE_CDA", "delta_CdA", {"new_CdA": "0.70"}, state),
            "Local delta vs Requested #1.",
        )

    def test_v21_aero_absolute_requires_baseline_when_inherited_reference_is_missing(self):
        state = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
            ],
            "columns": {
                "baseline": {"direct": {}},
                "walked_1": {"walk_from": "baseline", "direct": {}},
            },
            "preview_cache": {
                "baseline": {},
            },
        }
        status, warnings, missing, _ = _v21_validate_proposal_details(
            "walked_1",
            "aero",
            "AERO_ABSOLUTE_CDA",
            {
                "new_CdA": "0.64",
            },
            state,
        )
        self.assertEqual(status, "Missing")
        self.assertEqual(missing, ["baseline_CdA"])
        self.assertEqual(warnings, [])

    def test_v21_aero_absolute_manual_baseline_override_is_review(self):
        state = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
            ],
            "columns": {
                "baseline": {"direct": {}},
                "walked_1": {"walk_from": "baseline", "direct": {}},
            },
            "preview_cache": {
                "baseline": {},
            },
        }
        status, warnings, missing, _ = _v21_validate_proposal_details(
            "walked_1",
            "aero",
            "AERO_ABSOLUTE_CDA",
            {
                "new_CdA": "0.64",
                "baseline_CdA": "0.58",
            },
            state,
        )
        self.assertEqual(status, "Review")
        self.assertEqual(missing, [])
        self.assertTrue(any("manual baseline override" in warning for warning in warnings))

    def test_v21_aero_absolute_manual_baseline_field_becomes_editable_when_inherited_is_missing(self):
        state = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
            ],
            "columns": {
                "baseline": {"direct": {}},
                "walked_1": {"walk_from": "baseline", "direct": {}},
            },
            "preview_cache": {
                "baseline": {},
            },
        }
        display_state = _v21_field_display_state(
            "aero",
            "AERO_ABSOLUTE_CDA",
            "baseline_CdA",
            {"new_CdA": "0.64"},
            {"column_id": "walked_1", "state": state},
        )
        self.assertEqual(display_state, "missing")

    def test_v21_mass_reference_fields_live_in_baseline_reference_column(self):
        fields = _v21_reference_fields_for_proposal("mass", "GVWR", {})
        self.assertIn("mass_kg", fields)
        self.assertFalse(
            _v21_detail_field_editable(
                "mass",
                "GVWR",
                "mass_kg",
                {"gvwr_kg": "2800"},
                {"column_id": "walked_1", "state": {"columns": {}, "scenarios": []}},
            )
        )

    def test_v21_tire_reference_fields_include_inherited_rr_inputs(self):
        fields = _v21_reference_fields_for_proposal("tire", "TIRE_SMERF_RRC_CHANGE", {})
        self.assertIn("baseline_RRC_optional", fields)
        self.assertIn("front_pressure_psi", fields)
        self.assertIn("weight_dist_fr_pct", fields)

    def test_v21_aero_delta_keeps_reference_field_editable_in_baseline_layer(self):
        fields = _v21_reference_fields_for_proposal("aero", "AERO_DELTA_CDA", {})
        self.assertIn("baseline_CdA", fields)

    def test_v21_transmission_delta_keeps_baseline_reference_fields_visible(self):
        details = {"change_mode": "Delta ABC"}
        fields = _v21_reference_fields_for_proposal("transmission", "UPDATE_TRANS_DRAG_ABC", details)
        self.assertIn("baseline_trans_A", fields)
        self.assertIn("baseline_trans_B", fields)
        self.assertIn("baseline_trans_C", fields)
        compact = _v21_compact_fields_for_proposal(
            "transmission",
            "UPDATE_TRANS_DRAG_ABC",
            details,
            {"column_id": "walked_1", "state": {"columns": {}, "scenarios": []}},
        )
        self.assertIn("baseline_trans_A", compact)
        self.assertIn("baseline_trans_B", compact)
        self.assertIn("baseline_trans_C", compact)

    def test_v21_reference_raw_value_prefers_manual_override(self):
        state = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
            ],
            "columns": {
                "baseline": {"selected_vde_id": 1, "baseline_overrides": {}},
                "walked_1": {"walk_from": "baseline", "direct": {}},
            },
            "rows": [
                {"id": 1, "mass_kg": 2100.0},
            ],
            "preview_cache": {},
        }
        state = _v21_set_reference_override_values("walked_1", "mass", {"mass_kg": "2300"}, state)
        self.assertEqual(_v21_reference_raw_value("walked_1", "mass_kg", state), "2300")

    def test_v21_mass_reference_override_sets_review_status(self):
        state = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
            ],
            "columns": {
                "baseline": {"selected_vde_id": 1, "baseline_overrides": {}},
                "walked_1": {"walk_from": "baseline", "direct": {}},
            },
            "rows": [
                {"id": 1, "mass_kg": 2100.0},
            ],
            "preview_cache": {},
        }
        state = _v21_set_reference_override_values("walked_1", "mass", {"mass_kg": "2200"}, state)
        status, warnings, missing, effective = _v21_validate_proposal_details(
            "walked_1",
            "mass",
            "GVWR",
            {"gvwr_kg": "2800"},
            state,
        )
        self.assertEqual(status, "Review")
        self.assertEqual(missing, [])
        self.assertTrue(any("Manual reference override" in warning for warning in warnings))
        self.assertEqual(effective.get("payload_display_kg"), 600.0)

    def test_v21_request_preview_uses_missing_baseline_as_zero_for_absolute_delta(self):
        state = {
            "baseline_override_enabled": False,
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
            ],
            "columns": {
                "baseline": {"direct": {}, "printed_overrides": {}},
                "walked_1": {"walk_from": "baseline", "direct": {}},
            },
            "proposals": {
                "walked_1": {
                    "aero": {
                        "id": "prop_1",
                        "proposal_type": "AERO_ABSOLUTE_CDA",
                        "label": "Aero req",
                        "details": {"new_CdA": "0.64"},
                    }
                }
            },
            "preview_cache": {},
        }
        rows = _v21_request_preview_rows(state)
        row = next(item for item in rows if item["field"] == "New CdA")
        self.assertEqual(row["Baseline / Printed"], "-")
        self.assertEqual(row["Requested #1 input"], "0.64")
        self.assertEqual(row["Requested #1 delta"], "0.64")
        self.assertEqual(row["Requested #1 status"], "Review")

    def test_v21_request_preview_uses_baseline_override_value(self):
        state = {
            "baseline_override_enabled": True,
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
            ],
            "columns": {
                "baseline": {"direct": {}, "printed_overrides": {}},
                "walked_1": {"walk_from": "baseline", "direct": {}},
            },
            "proposals": {
                "walked_1": {
                    "aero": {
                        "id": "prop_1",
                        "proposal_type": "AERO_ABSOLUTE_CDA",
                        "label": "Aero req",
                        "details": {"new_CdA": "0.64"},
                    }
                }
            },
            "preview_cache": {},
        }
        state = _v21_set_baseline_printed_override_values("aero", {"baseline_CdA": "0.60"}, state)
        rows = _v21_request_preview_rows(state)
        row = next(item for item in rows if item["field"] == "New CdA")
        self.assertEqual(row["Baseline / Printed"], "0.60")
        self.assertEqual(row["Baseline Override"], "Yes")
        self.assertEqual(row["Requested #1 delta"], "0.04")

    def test_v21_summary_text_reports_inherited_walk_from(self):
        state = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "walked_1", "label": "Walked #1", "role": "walked"},
            ],
            "columns": {
                "baseline": {"direct": {}},
                "walked_1": {"walk_from": "baseline", "direct": {}},
            },
            "proposals": {},
        }
        self.assertEqual(_v21_summary_text("walked_1", "tire", state), "Inherited from Baseline")

    def test_v21_proposals_assign_id_to_legacy_entries(self):
        state = {
            "proposals": {
                "walked_1": {
                    "tire": {
                        "proposal_type": "TIRE_DB_LOOKUP",
                        "label": "TPS Tire",
                        "details": {},
                    }
                }
            }
        }
        proposals = _v21_proposals(state)
        self.assertEqual(proposals["walked_1"]["tire"]["id"], "prop_1")
        self.assertEqual(proposals["walked_1"]["tire"]["domain"], "tire")
        self.assertEqual(proposals["walked_1"]["tire"]["type"], "TIRE_DB_LOOKUP")
        self.assertEqual(state["proposal_seq"], 1)

    def test_v21_summary_text_reports_prop_badge(self):
        state = {
            "proposals": {
                "walked_2": {
                    "tire": {
                        "id": "prop_3",
                        "domain": "tire",
                        "type": "TIRE_DB_LOOKUP",
                        "proposal_type": "TIRE_DB_LOOKUP",
                        "label": "TPS Tire",
                        "details": {},
                        "status": "Draft",
                    }
                }
            }
        }
        self.assertEqual(_v21_summary_text("walked_2", "tire", state), "Prop #3 · TPS Tire")


if __name__ == "__main__":
    unittest.main()
