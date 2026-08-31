"""Sprint 11D hotfix coverage for System Scenario source materialization."""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pandas as pd

from src.vde_app.components import pwt_system_scenario
from src.vde_app.powertrain_system_scenario_viewmodels import (
    ScenarioSource,
    add_proposal_draft,
    current_draft,
)
from src.vde_core.system_scenario import ArchitectureClass


def _drafts_with_duplicate_sources():
    drafts = (current_draft(1, ArchitectureClass.ICE),)
    drafts = add_proposal_draft(drafts, vde_id=2, architecture=ArchitectureClass.ICE)
    drafts = add_proposal_draft(drafts, vde_id=2, architecture=ArchitectureClass.ICE)
    drafts = add_proposal_draft(drafts, vde_id=1, architecture=ArchitectureClass.ICE)
    return tuple(drafts)


def _drafts_with_four_unique_sources():
    drafts = (current_draft(1, ArchitectureClass.ICE),)
    for vde_id in (2, 3, 4):
        drafts = add_proposal_draft(drafts, vde_id=vde_id, architecture=ArchitectureClass.ICE)
    return tuple(drafts)


class PowertrainSystemScenarioSourceLoadingTests(unittest.TestCase):
    def test_working_set_deduplicates_current_and_proposal_source_ids(self):
        drafts = _drafts_with_duplicate_sources()

        self.assertEqual(pwt_system_scenario._working_set_vde_ids(1, drafts), (1, 2))

    def test_working_set_uses_current_draft_not_an_obsolete_anchor(self):
        current = replace(current_draft(1, ArchitectureClass.ICE), vde_id=2)
        drafts = add_proposal_draft((current,), vde_id=3, architecture=ArchitectureClass.ICE)

        self.assertEqual(pwt_system_scenario._working_set_vde_ids(1, drafts), (2, 3))

    def test_canonical_delta_options_exclude_unsupported_energy_percent_delta(self):
        self.assertNotIn(
            "energy_percent_delta",
            pwt_system_scenario._CANONICAL_TECH_DELTA_EFFECT_BASES,
        )

    @patch("src.vde_app.components.pwt_system_scenario.fetch_fuelcons_row")
    @patch("src.vde_app.components.pwt_system_scenario.fetch_vde_rows_by_ids")
    @patch("src.vde_app.components.pwt_system_scenario.load_baselines_df")
    def test_large_discovery_list_materializes_only_four_active_sources(
        self,
        load_baselines,
        fetch_details,
        fetch_fuelcons_row,
    ):
        drafts = _drafts_with_four_unique_sources()
        load_baselines.return_value = pd.DataFrame(
            [{"id": vde_id, "make": "Synthetic", "model": f"{vde_id}"} for vde_id in range(1, 5001)]
        )
        fetch_details.return_value = pd.DataFrame(
            [
                {
                    "id": vde_id,
                    "vde_total_mj_per_km": 1.6 + vde_id / 10,
                    "vde_net_mj_per_km": 1.4 + vde_id / 10,
                }
                for vde_id in (1, 2, 3, 4)
            ]
        )
        fetch_fuelcons_row.return_value = {
            "id": 101,
            "vde_id": 1,
            "electrification": "ICE",
            "eta_pt_est": 0.30,
        }

        with patch.object(pwt_system_scenario, "ScenarioSource", wraps=ScenarioSource) as source_constructor:
            sources, labels = pwt_system_scenario._load_sources(
                1,
                101,
                drafts=drafts,
            )

        self.assertEqual(len(labels), 5000)
        self.assertEqual(set(sources), {1, 2, 3, 4})
        fetch_details.assert_called_once_with((1, 2, 3, 4))
        fetch_fuelcons_row.assert_called_once_with(101)
        self.assertTrue(all(source.fuelcons_row["id"] == 101 for source in sources.values()))
        self.assertEqual(source_constructor.call_count, 4)

    @patch("src.vde_app.components.pwt_system_scenario.fetch_fuelcons_baselines")
    def test_fuelcons_discovery_keeps_only_lightweight_search_labels(self, fetch_baselines):
        fetch_baselines.return_value = pd.DataFrame(
            [
                {
                    "fuelcons_id": 101,
                    "vde_id": 11,
                    "make": "Synthetic",
                    "model": "A",
                    "year": 2024,
                    "electrification": "ICE",
                }
            ]
        )

        labels = pwt_system_scenario._fuelcons_baseline_labels()

        self.assertEqual(len(labels), 1)
        self.assertIn("FuelCons-101", labels[101])
        self.assertIn("VDE-11", labels[101])

    def test_vde_impact_view_keeps_missing_net_as_not_evaluated(self):
        source = ScenarioSource(
            vde_id=1,
            vde_row={"id": 1, "vde_total_mj_per_km": 1.8, "vde_net_mj_per_km": None},
            fuelcons_row={"id": 101, "vde_id": 1, "electrification": "ICE"},
        )
        draft = current_draft(1, ArchitectureClass.ICE, fuelcons_id=101)
        columns = [MagicMock() for _ in range(3)]
        with patch.object(pwt_system_scenario.st, "columns", return_value=columns):
            pwt_system_scenario._render_vde_impact_only(draft, {1: source}, {})

        self.assertEqual(columns[2].metric.call_args.args[1], "Not evaluated")


if __name__ == "__main__":
    unittest.main()
