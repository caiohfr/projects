from __future__ import annotations

import unittest

from src.vde_app.comparison_report_viewmodels import SelectionState
from src.vde_app.home_context import resolve_home_context


VDES = {
    11: {"id": 11, "make": "QA", "model": "Baseline", "year": 2026},
    12: {"id": 12, "make": "QA", "model": "Saved proposal", "year": 2027},
}
FUELCONS = {21: {"id": 21, "vde_id": 11}}


def fetch_vde(record_id: int):
    return VDES.get(record_id)


def fetch_fuelcons(record_id: int):
    return FUELCONS.get(record_id)


class HomeContextTests(unittest.TestCase):
    def test_first_access_is_explicitly_empty(self):
        context = resolve_home_context({}, fetch_vde=fetch_vde, fetch_fuelcons=fetch_fuelcons)

        self.assertFalse(context.has_active_analysis)
        self.assertIsNone(context.baseline)
        self.assertIsNone(context.proposal)
        self.assertIsNone(context.fuelcons)
        self.assertEqual(context.recent, ())
        self.assertEqual(context.warnings, ())

    def test_active_context_uses_recoverable_baseline_proposal_and_fuelcons(self):
        context = resolve_home_context(
            {
                "vde_setup_v22": {
                    "baseline": {"selected_vde_id": 11},
                    "proposals": [
                        {
                            "proposal_id": "requested_1",
                            "name": "Mass study",
                            "domains": {"mass": {"proposal_type": "MASS_ABSOLUTE"}},
                            "inputs": {"mass": {"mass_kg": 1900}},
                        }
                    ],
                    "preview": {"status": "success"},
                    "save": {"status": "success", "result": {"saved_proposals": [{"vde_row_id": 12}]}},
                },
                "pwt_ss:current_baseline": 21,
                "roadload_load_case_vde_id": 11,
                "comparison_selection": SelectionState(reference_fuelcons_id=21),
            },
            fetch_vde=fetch_vde,
            fetch_fuelcons=fetch_fuelcons,
        )

        self.assertTrue(context.has_active_analysis)
        self.assertEqual(context.baseline.vde_id, 11)
        self.assertEqual(context.proposal.title, "1 temporary proposal")
        self.assertIn("not persisted", context.proposal.source)
        self.assertEqual(context.fuelcons.fuelcons_id, 21)
        self.assertTrue(any(item.vde_id == 12 for item in context.recent))
        self.assertTrue(any(item.kind == "load_case" for item in context.recent))

    def test_invalid_session_references_are_not_presented_as_active_data(self):
        context = resolve_home_context(
            {
                "vde_setup_v22": {"baseline": {"selected_vde_id": 999}},
                "pwt_ss:current_baseline": 998,
            },
            fetch_vde=fetch_vde,
            fetch_fuelcons=fetch_fuelcons,
        )

        self.assertFalse(context.has_active_analysis)
        self.assertEqual(len(context.warnings), 2)
        self.assertTrue(all("no longer available" in warning for warning in context.warnings))

    def test_blank_default_proposals_do_not_create_false_recent_analysis(self):
        context = resolve_home_context(
            {
                "vde_setup_v22": {
                    "proposals": [
                        {
                            "proposal_id": "requested_1",
                            "name": "",
                            "domains": {"mass": {"proposal_type": "INHERIT"}},
                            "inputs": {},
                        }
                    ]
                }
            },
            fetch_vde=fetch_vde,
            fetch_fuelcons=fetch_fuelcons,
        )

        self.assertIsNone(context.proposal)
        self.assertEqual(context.recent, ())


if __name__ == "__main__":
    unittest.main()
