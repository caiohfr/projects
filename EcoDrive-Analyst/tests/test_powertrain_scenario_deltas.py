import unittest

from src.vde_app.components import pwt_fuel_energy
from src.vde_app.components.pwt_fuel_energy import _apply_delta_stack_to_baseline, _effective_baseline_method
from src.vde_core import technology_delta
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


class PowertrainAndQuickScenarioShareCanonicalCoreTests(unittest.TestCase):
    """Sprint 10D centralization: pwt_fuel_energy.py's own
    _apply_delta_stack_to_baseline (used by the live Powertrain Scenario
    page) and Quick Scenario's Efficiency resolver both delegate to
    src.vde_core.technology_delta.apply_delta_stack_to_baseline -- these
    tests prove that ownership directly (identity + numeric agreement),
    never by re-deriving the expected value with a third implementation.
    """

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

    def _delta(self, **overrides):
        return technology_delta.normalize_technology_delta({"enabled": True, **overrides})

    def test_pwt_fuel_energy_module_holds_the_canonical_function_not_a_copy(self):
        # The module-level name pwt_fuel_energy._canonical_apply_delta_stack_to_baseline
        # must be the exact same function object as the canonical core --
        # not a reimplementation that merely produces matching numbers.
        self.assertIs(
            pwt_fuel_energy._canonical_apply_delta_stack_to_baseline,
            technology_delta.apply_delta_stack_to_baseline,
        )
        self.assertIs(pwt_fuel_energy._normalize_delta_effect_basis, technology_delta.normalize_delta_effect_basis)
        self.assertIs(pwt_fuel_energy._maturity_rank, technology_delta.maturity_rank)
        self.assertIs(pwt_fuel_energy._delta_status_counts, technology_delta.delta_status_counts)
        self.assertIs(pwt_fuel_energy._proposal_confidence_label, technology_delta.proposal_confidence_label)

    def test_single_percent_delta_matches_canonical_core(self):
        baseline = self._baseline_result()
        deltas = [self._delta(effect_basis="PSE percent delta", effect_value=5.0)]
        ui_result = _apply_delta_stack_to_baseline(baseline, ctx={"energy_value_mj_per_km": 1.8}, deltas=deltas)
        canonical_result = technology_delta.apply_delta_stack_to_baseline(
            baseline, ctx={"energy_value_mj_per_km": 1.8}, deltas=deltas
        )
        self.assertAlmostEqual(ui_result["proposal"]["pse"], canonical_result["proposal"]["pse"], places=9)
        self.assertAlmostEqual(
            ui_result["proposal"]["fuel_l_100km"], canonical_result["proposal"]["fuel_l_100km"], places=9
        )

    def test_two_sequential_percent_deltas_compound_and_match_canonical_core(self):
        baseline = self._baseline_result()
        base_pse = baseline.assumptions["pse_summary"]["value"]
        deltas = [
            self._delta(effect_basis="PSE percent delta", effect_value=5.0),
            self._delta(effect_basis="PSE percent delta", effect_value=5.0),
        ]
        ui_result = _apply_delta_stack_to_baseline(baseline, ctx={"energy_value_mj_per_km": 1.8}, deltas=deltas)
        canonical_result = technology_delta.apply_delta_stack_to_baseline(
            baseline, ctx={"energy_value_mj_per_km": 1.8}, deltas=deltas
        )
        self.assertAlmostEqual(ui_result["proposal"]["pse"], canonical_result["proposal"]["pse"], places=9)
        # Compounds (base*1.05*1.05), never sums (base*1.10) -- proven
        # against the canonical core's own output, not a hand-derived number.
        self.assertAlmostEqual(canonical_result["proposal"]["pse"], base_pse * 1.05 * 1.05, places=9)
        self.assertNotAlmostEqual(canonical_result["proposal"]["pse"], base_pse * 1.10, places=6)

    def test_absolute_plus_percent_delta_matches_canonical_core(self):
        baseline = self._baseline_result()
        deltas = [
            self._delta(effect_basis="PSE delta", effect_value=0.02),
            self._delta(effect_basis="PSE percent delta", effect_value=10.0),
        ]
        ui_result = _apply_delta_stack_to_baseline(baseline, ctx={"energy_value_mj_per_km": 1.8}, deltas=deltas)
        canonical_result = technology_delta.apply_delta_stack_to_baseline(
            baseline, ctx={"energy_value_mj_per_km": 1.8}, deltas=deltas
        )
        self.assertAlmostEqual(ui_result["proposal"]["pse"], canonical_result["proposal"]["pse"], places=9)

    def test_multiplier_delta_matches_canonical_core(self):
        baseline = self._baseline_result()
        deltas = [self._delta(effect_basis="PSE multiplier", effect_value=1.05)]
        ui_result = _apply_delta_stack_to_baseline(baseline, ctx={"energy_value_mj_per_km": 1.8}, deltas=deltas)
        canonical_result = technology_delta.apply_delta_stack_to_baseline(
            baseline, ctx={"energy_value_mj_per_km": 1.8}, deltas=deltas
        )
        self.assertAlmostEqual(ui_result["proposal"]["pse"], canonical_result["proposal"]["pse"], places=9)
        self.assertAlmostEqual(
            canonical_result["proposal"]["pse"], baseline.assumptions["pse_summary"]["value"] * 1.05, places=9
        )

    def test_zero_delta_is_neutral_and_matches_canonical_core(self):
        baseline = self._baseline_result()
        deltas = [self._delta(effect_basis="PSE percent delta", effect_value=0.0)]
        ui_result = _apply_delta_stack_to_baseline(baseline, ctx={"energy_value_mj_per_km": 1.8}, deltas=deltas)
        canonical_result = technology_delta.apply_delta_stack_to_baseline(
            baseline, ctx={"energy_value_mj_per_km": 1.8}, deltas=deltas
        )
        self.assertAlmostEqual(ui_result["proposal"]["pse"], canonical_result["proposal"]["pse"], places=9)
        self.assertAlmostEqual(
            canonical_result["proposal"]["pse"], baseline.assumptions["pse_summary"]["value"], places=9
        )

    def test_ordering_behavior_matches_canonical_core(self):
        baseline = self._baseline_result()
        forward = [
            self._delta(effect_basis="PSE delta", effect_value=0.02),
            self._delta(effect_basis="PSE percent delta", effect_value=10.0),
        ]
        reversed_order = [
            self._delta(effect_basis="PSE percent delta", effect_value=10.0),
            self._delta(effect_basis="PSE delta", effect_value=0.02),
        ]
        ui_forward = _apply_delta_stack_to_baseline(baseline, ctx={"energy_value_mj_per_km": 1.8}, deltas=forward)
        ui_reversed = _apply_delta_stack_to_baseline(baseline, ctx={"energy_value_mj_per_km": 1.8}, deltas=reversed_order)
        canonical_forward = technology_delta.apply_delta_stack_to_baseline(
            baseline, ctx={"energy_value_mj_per_km": 1.8}, deltas=forward
        )
        canonical_reversed = technology_delta.apply_delta_stack_to_baseline(
            baseline, ctx={"energy_value_mj_per_km": 1.8}, deltas=reversed_order
        )
        self.assertAlmostEqual(ui_forward["proposal"]["pse"], canonical_forward["proposal"]["pse"], places=9)
        self.assertAlmostEqual(ui_reversed["proposal"]["pse"], canonical_reversed["proposal"]["pse"], places=9)
        self.assertNotAlmostEqual(canonical_forward["proposal"]["pse"], canonical_reversed["proposal"]["pse"], places=6)

    def test_method_label_decoration_is_preserved_after_centralization(self):
        # The one intentional local behavior kept on top of the canonical
        # core: this page's own pretty method label, not the raw method key
        # the canonical extraction stores.
        baseline = self._baseline_result()
        result = _apply_delta_stack_to_baseline(baseline, ctx={"energy_value_mj_per_km": 1.8}, deltas=[])
        self.assertEqual(result["baseline"]["method"], "Assume efficiency")
        self.assertNotEqual(result["baseline"]["method"], "physics_simple")


if __name__ == "__main__":
    unittest.main()
