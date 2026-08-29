"""Sprint 11D hotfix coverage for System Scenario source materialization."""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

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

    @patch("src.vde_app.components.pwt_system_scenario.fetch_fuelcons_by_vde")
    @patch("src.vde_app.components.pwt_system_scenario.fetch_vde_rows_by_ids")
    @patch("src.vde_app.components.pwt_system_scenario.load_baselines_df")
    def test_large_discovery_list_materializes_only_four_active_sources(
        self,
        load_baselines,
        fetch_details,
        fetch_fuelcons,
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
        fetch_fuelcons.side_effect = lambda vde_id: pd.DataFrame(
            [{"id": vde_id, "vde_id": vde_id, "electrification": "ICE", "eta_pt_est": 0.30}]
        )

        with patch.object(pwt_system_scenario, "ScenarioSource", wraps=ScenarioSource) as source_constructor:
            sources, labels = pwt_system_scenario._load_sources(
                1,
                {"id": 1, "make": "Synthetic", "model": "1", "vde_total_mj_per_km": 1.8},
                drafts=drafts,
            )

        self.assertEqual(len(labels), 5000)
        self.assertEqual(set(sources), {1, 2, 3, 4})
        fetch_details.assert_called_once_with((1, 2, 3, 4))
        self.assertEqual({call.args[0] for call in fetch_fuelcons.call_args_list}, {1, 2, 3, 4})
        self.assertEqual(fetch_fuelcons.call_count, 4)
        self.assertEqual(source_constructor.call_count, 4)


if __name__ == "__main__":
    unittest.main()
