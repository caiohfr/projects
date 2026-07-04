import unittest

from src.vde_app.components.pwt_fuel_energy import _apply_delta_stack_to_baseline, _effective_baseline_method
from src.vde_core.fuel_estimation import FuelEstimateRequest, run_fuel_estimation


class PowertrainScenarioDeltaTests(unittest.TestCase):
    def test_confirmed_baseline_method_wins_over_active_preview(self):
        self.assertEqual(
            _effective_baseline_method("Regression", "ML Prediction"),
            "ML Prediction",
        )
        self.assertEqual(
            _effective_baseline_method("Regression", None),
            "Regression",
        )

    def _baseline_result(self):
        return run_fuel_estimation(
            FuelEstimateRequest(
                vde_id=301,
                energy_basis="VDE_TOTAL",
                method="physics_simple",
                vehicle_features={"electrification": "ICE", "vde_total_mj_per_km": 1.8},
                powertrain_features={"eta_pt_est": 0.3, "fuel_type": "Gasoline", "LHV_MJ_per_L": 32.0},
            )
        )

    def test_registered_only_delta_does_not_change_proposal(self):
        baseline = self._baseline_result()
        proposal = _apply_delta_stack_to_baseline(
            baseline,
            ctx={"energy_value_mj_per_km": 1.8},
            deltas=[
                {
                    "id": 1,
                    "name": "Engine map change",
                    "effect_basis": "map-based effect",
                    "maturity_level": "simulation_ready",
                    "source_type": "imported_map",
                    "confidence": "medium",
                    "quantitative_status": "pending_model",
                }
            ],
        )

        self.assertEqual(proposal["status"], "No quantitative delta")
        self.assertAlmostEqual(proposal["proposal"]["fuel_l_100km"], proposal["baseline"]["fuel_l_100km"], places=6)
        self.assertEqual(len(proposal["applied_deltas"]), 0)
        self.assertEqual(len(proposal["registered_only_deltas"]), 1)

    def test_manual_fuel_delta_changes_proposal_when_applied(self):
        baseline = self._baseline_result()
        proposal = _apply_delta_stack_to_baseline(
            baseline,
            ctx={"energy_value_mj_per_km": 1.8},
            deltas=[
                {
                    "id": 1,
                    "name": "Manual fuel delta",
                    "effect_basis": "fuel delta",
                    "effect_value": -1.0,
                    "maturity_level": "engineering_assumption",
                    "source_type": "manual",
                    "confidence": "medium",
                    "quantitative_status": "applied",
                }
            ],
        )

        self.assertEqual(proposal["status"], "Estimated")
        self.assertAlmostEqual(proposal["proposal"]["fuel_l_100km"], proposal["baseline"]["fuel_l_100km"] - 1.0, places=6)
        self.assertEqual(len(proposal["applied_deltas"]), 1)

    def test_manual_pse_delta_changes_proposal_when_applied(self):
        baseline = self._baseline_result()
        proposal = _apply_delta_stack_to_baseline(
            baseline,
            ctx={"energy_value_mj_per_km": 1.8},
            deltas=[
                {
                    "id": 1,
                    "name": "Manual PSE delta",
                    "effect_basis": "PSE delta",
                    "effect_value": 0.03,
                    "maturity_level": "engineering_assumption",
                    "source_type": "manual",
                    "confidence": "medium",
                    "quantitative_status": "applied",
                }
            ],
        )

        self.assertGreater(proposal["proposal"]["pse"], proposal["baseline"]["pse"])
        self.assertLess(proposal["proposal"]["fuel_l_100km"], proposal["baseline"]["fuel_l_100km"])
        self.assertEqual(len(proposal["applied_deltas"]), 1)

    def test_pse_percent_delta_updates_pse_and_fuel(self):
        baseline = self._baseline_result()
        proposal = _apply_delta_stack_to_baseline(
            baseline,
            ctx={"energy_value_mj_per_km": 1.8},
            deltas=[
                {
                    "id": 1,
                    "name": "Supplier efficiency claim",
                    "effect_basis": "PSE percent delta",
                    "effect_value": 5.0,
                    "maturity_level": "supplier_data",
                    "source_type": "manual",
                    "confidence": "medium",
                    "quantitative_status": "applied",
                }
            ],
        )

        self.assertAlmostEqual(
            proposal["proposal"]["pse"],
            proposal["baseline"]["pse"] * 1.05,
            places=6,
        )
        self.assertLess(proposal["proposal"]["fuel_l_100km"], proposal["baseline"]["fuel_l_100km"])


if __name__ == "__main__":
    unittest.main()
