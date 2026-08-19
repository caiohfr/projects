from __future__ import annotations

import unittest

from src.vde_core.vde_request_finalization import build_scenario_configuration_summaries, suggested_scenario_name
from src.vde_core.vde_request_compact_state import apply_v22_baseline, apply_v22_proposal_matrix, create_v22_state


class TestVdeRequestFinalization(unittest.TestCase):
    def test_summary_lists_only_direct_changes_and_uses_effective_metadata(self):
        state = apply_v22_baseline(
            create_v22_state(),
            {"id": 1, "make": "QA", "model": "Program X", "year": 2026, "legislation": "EPA", "cycle_name": "FTP75"},
        )
        state = apply_v22_proposal_matrix(
            state,
            [
                {
                    "proposal_id": "requested_1",
                    "walk_from": "baseline",
                    "mass": "Inherit",
                    "aero": "Absolute CdA",
                    "tire": "Tire DB lookup",
                    "brake": "Absolute ABC",
                },
                {"proposal_id": "requested_2", "walk_from": "requested_1", "mass": "Inherit", "aero": "Inherit"},
            ],
        )
        state["proposals"][0]["inputs"]["tire"] = {"tire_code": "TIRE-TKH"}

        summaries = build_scenario_configuration_summaries(state)
        first, second = summaries

        self.assertEqual(first["program_label"], "QA Program X · MY26")
        self.assertEqual(first["based_on"], "Baseline")
        self.assertEqual(first["engineering_summary"], "Aero Proposal + Tire TKH + Brake Proposal")
        self.assertNotIn("Mass", first["engineering_summary"])
        self.assertEqual(second["based_on"], "Requested #1")
        self.assertEqual(second["engineering_summary"], "No direct engineering changes")

    def test_suggested_name_preserves_meaningful_user_name(self):
        summary = {"proposal_label": "Requested #1", "suggested_name": "QA Program X MY26 - Aero Proposal"}

        self.assertEqual(suggested_scenario_name(summary, ""), "QA Program X MY26 - Aero Proposal")
        self.assertEqual(suggested_scenario_name(summary, "Requested #1"), "QA Program X MY26 - Aero Proposal")
        self.assertEqual(suggested_scenario_name(summary, "My deliberate name"), "My deliberate name")
