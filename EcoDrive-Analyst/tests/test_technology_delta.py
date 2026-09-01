"""Sprint 10D: tests for the extracted, Streamlit-free
src/vde_core/technology_delta.py -- verified byte-for-byte parity with the
existing Powertrain Scenario stacking math
(pwt_fuel_energy._apply_delta_stack_to_baseline), plus new multi-delta
stacking coverage that did not exist before (Sec 15: "Add dedicated
multi-delta tests because prior coverage was weak").
"""

import unittest

from src.vde_core.fuel_estimation import FuelEstimateRequest, run_fuel_estimation
from src.vde_core.technology_delta import (
    DELTA_CONFIDENCE_OPTIONS,
    DELTA_MATURITY_OPTIONS,
    TechDeltaAssumption,
    apply_delta_stack_to_baseline,
    delta_status_counts,
    maturity_rank,
    normalize_delta_effect_basis,
    normalize_technology_delta,
    proposal_confidence_label,
    tech_delta_assumption_from_dict,
    tech_delta_assumption_to_dict,
)


class TechDeltaContractOwnershipTests(unittest.TestCase):
    def test_quick_scenario_reexports_the_exact_canonical_class(self):
        from src.vde_core.quick_scenario.contracts import TechDeltaAssumption as QuickAssumption

        self.assertIs(QuickAssumption, TechDeltaAssumption)

    def test_quick_serialization_reexports_the_exact_canonical_parser(self):
        from src.vde_core.quick_scenario.serialization import tech_delta_assumption_from_dict as quick_parser

        self.assertIs(quick_parser, tech_delta_assumption_from_dict)

    def test_typed_assumption_roundtrip_has_one_shared_adapter(self):
        assumption = TechDeltaAssumption("Efficiency", "pse_percent_delta", 1.25)
        self.assertEqual(
            tech_delta_assumption_from_dict(tech_delta_assumption_to_dict(assumption)),
            assumption,
        )


def _baseline_result():
    request = FuelEstimateRequest(
        vde_id=1,
        energy_basis="VDE_TOTAL",
        method="physics_simple",
        vehicle_features={"electrification": "ICE", "vde_total_mj_per_km": 1.8},
        powertrain_features={"eta_pt_est": 0.3, "fuel_type": "Gasoline", "LHV_MJ_per_L": 32.0},
    )
    return run_fuel_estimation(request)


def _delta(**overrides):
    return normalize_technology_delta({"enabled": True, **overrides})


class NormalizeDeltaEffectBasisTests(unittest.TestCase):
    def test_ui_labels_map_to_canonical_keys(self):
        self.assertEqual(normalize_delta_effect_basis("PSE percent delta"), "pse_percent_delta")
        self.assertEqual(normalize_delta_effect_basis("fuel delta"), "fuel_delta")
        self.assertEqual(normalize_delta_effect_basis("map-based effect"), "map_based_effect")

    def test_unknown_label_passes_through_stripped(self):
        self.assertEqual(normalize_delta_effect_basis("  something_custom  "), "something_custom")


class NormalizeTechnologyDeltaTests(unittest.TestCase):
    def test_disabled_delta_is_registered_only(self):
        delta = normalize_technology_delta({"enabled": False, "effect_basis": "fuel delta", "effect_value": -1.0})
        self.assertEqual(delta["quantitative_status"], "disabled")

    def test_map_based_effect_is_pending_model_even_if_value_present(self):
        delta = normalize_technology_delta({"effect_basis": "map-based effect", "effect_value": 1.0})
        self.assertEqual(delta["quantitative_status"], "pending_model")

    def test_missing_effect_value_is_registered_only(self):
        delta = normalize_technology_delta({"effect_basis": "PSE delta"})
        self.assertEqual(delta["quantitative_status"], "registered_only")

    def test_zero_effect_value_is_applied_not_dropped(self):
        delta = normalize_technology_delta({"effect_basis": "PSE delta", "effect_value": 0.0})
        self.assertEqual(delta["quantitative_status"], "applied")
        self.assertEqual(delta["effect_value"], 0.0)


class SingleDeltaParityWithExistingBehaviorTests(unittest.TestCase):
    """Mirrors tests/test_powertrain_scenario_deltas.py's own single-delta
    assertions exactly, confirming the extraction is behaviorally identical.
    """

    def test_registered_only_delta_does_not_change_proposal(self):
        baseline = _baseline_result()
        delta = _delta(effect_basis="map-based effect", effect_value=1.0)
        result = apply_delta_stack_to_baseline(baseline, ctx={"energy_value_mj_per_km": 1.8}, deltas=[delta])
        self.assertEqual(result["status"], "No quantitative delta")
        self.assertEqual(result["proposal"]["fuel_l_100km"], result["baseline"]["fuel_l_100km"])
        self.assertEqual(len(result["applied_deltas"]), 0)
        self.assertEqual(len(result["registered_only_deltas"]), 1)

    def test_manual_fuel_delta_changes_proposal_when_applied(self):
        baseline = _baseline_result()
        delta = _delta(effect_basis="fuel delta", effect_value=-1.0)
        result = apply_delta_stack_to_baseline(baseline, ctx={"energy_value_mj_per_km": 1.8}, deltas=[delta])
        self.assertEqual(result["status"], "Estimated")
        self.assertAlmostEqual(result["proposal"]["fuel_l_100km"], result["baseline"]["fuel_l_100km"] - 1.0)

    def test_manual_pse_delta_changes_proposal_when_applied(self):
        baseline = _baseline_result()
        delta = _delta(effect_basis="PSE delta", effect_value=0.03)
        result = apply_delta_stack_to_baseline(baseline, ctx={"energy_value_mj_per_km": 1.8}, deltas=[delta])
        self.assertGreater(result["proposal"]["pse"], result["baseline"]["pse"])
        self.assertLess(result["proposal"]["fuel_l_100km"], result["baseline"]["fuel_l_100km"])

    def test_pse_percent_delta_updates_pse_and_fuel(self):
        baseline = _baseline_result()
        delta = _delta(effect_basis="PSE percent delta", effect_value=5.0)
        result = apply_delta_stack_to_baseline(baseline, ctx={"energy_value_mj_per_km": 1.8}, deltas=[delta])
        self.assertAlmostEqual(result["proposal"]["pse"], result["baseline"]["pse"] * 1.05, places=6)
        self.assertLess(result["proposal"]["fuel_l_100km"], result["baseline"]["fuel_l_100km"])


class MultiDeltaStackingTests(unittest.TestCase):
    """New coverage (Sec 15/27): prior coverage never exercised more than
    one delta in a stack.
    """

    def test_two_percent_deltas_compound_not_sum(self):
        baseline = _baseline_result()
        base_pse = baseline.assumptions["pse_summary"]["value"]
        deltas = [
            _delta(effect_basis="PSE percent delta", effect_value=5.0),
            _delta(effect_basis="PSE percent delta", effect_value=5.0),
        ]
        result = apply_delta_stack_to_baseline(baseline, ctx={"energy_value_mj_per_km": 1.8}, deltas=deltas)
        expected_compounded = base_pse * 1.05 * 1.05
        expected_summed = base_pse * 1.10
        self.assertAlmostEqual(result["proposal"]["pse"], expected_compounded, places=9)
        self.assertNotAlmostEqual(result["proposal"]["pse"], expected_summed, places=6)
        self.assertEqual(result["delta_counts"]["applied"], 2)

    def test_absolute_then_percent_delta_applies_sequentially(self):
        baseline = _baseline_result()
        base_pse = baseline.assumptions["pse_summary"]["value"]
        deltas = [
            _delta(effect_basis="PSE delta", effect_value=0.02),
            _delta(effect_basis="PSE percent delta", effect_value=10.0),
        ]
        result = apply_delta_stack_to_baseline(baseline, ctx={"energy_value_mj_per_km": 1.8}, deltas=deltas)
        expected = (base_pse + 0.02) * 1.10
        self.assertAlmostEqual(result["proposal"]["pse"], expected, places=9)

    def test_reversed_order_of_absolute_and_percent_gives_different_result(self):
        # Order matters for additive-then-multiplicative vs. the reverse --
        # confirms sequential application, not a commutative shortcut.
        baseline = _baseline_result()
        base_pse = baseline.assumptions["pse_summary"]["value"]
        forward = apply_delta_stack_to_baseline(
            baseline,
            ctx={"energy_value_mj_per_km": 1.8},
            deltas=[_delta(effect_basis="PSE delta", effect_value=0.02), _delta(effect_basis="PSE percent delta", effect_value=10.0)],
        )
        reversed_order = apply_delta_stack_to_baseline(
            baseline,
            ctx={"energy_value_mj_per_km": 1.8},
            deltas=[_delta(effect_basis="PSE percent delta", effect_value=10.0), _delta(effect_basis="PSE delta", effect_value=0.02)],
        )
        expected_forward = (base_pse + 0.02) * 1.10
        expected_reversed = (base_pse * 1.10) + 0.02
        self.assertAlmostEqual(forward["proposal"]["pse"], expected_forward, places=9)
        self.assertAlmostEqual(reversed_order["proposal"]["pse"], expected_reversed, places=9)
        self.assertNotAlmostEqual(forward["proposal"]["pse"], reversed_order["proposal"]["pse"], places=6)

    def test_zero_value_delta_in_a_stack_is_neutral(self):
        baseline = _baseline_result()
        deltas = [
            _delta(effect_basis="PSE percent delta", effect_value=0.0),
            _delta(effect_basis="PSE percent delta", effect_value=5.0),
        ]
        result = apply_delta_stack_to_baseline(baseline, ctx={"energy_value_mj_per_km": 1.8}, deltas=deltas)
        base_pse = baseline.assumptions["pse_summary"]["value"]
        self.assertAlmostEqual(result["proposal"]["pse"], base_pse * 1.05, places=9)

    def test_co2_delta_is_overwritten_by_fuel_reconciliation_when_stacked_with_pse_delta(self):
        """Documents the existing, un-"fixed" quirk (Sec 15): a direct
        co2_delta applied alongside any PSE/fuel-affecting delta is
        overwritten by the unconditional fuel->CO2 recompute that runs
        after the stacking loop, because gco2_km is always re-derived from
        fuel_l_100km once fuel_l_100km is present.
        """

        baseline = _baseline_result()
        deltas = [
            _delta(effect_basis="CO2 delta", effect_value=-1000.0),
            _delta(effect_basis="PSE percent delta", effect_value=5.0),
        ]
        result = apply_delta_stack_to_baseline(baseline, ctx={"energy_value_mj_per_km": 1.8}, deltas=deltas)
        naive_expected_co2 = baseline.gco2_km - 1000.0
        self.assertNotAlmostEqual(result["proposal"]["gco2_km"], naive_expected_co2, places=3)
        recomputed_co2 = (result["proposal"]["fuel_l_100km"] / 100.0) * (
            baseline.gco2_km / (baseline.fuel_l_100km / 100.0)
        )
        self.assertAlmostEqual(result["proposal"]["gco2_km"], recomputed_co2, places=6)


class DeltaStatusCountsAndConfidenceTests(unittest.TestCase):
    def test_delta_status_counts_tallies_each_bucket(self):
        deltas = [
            _delta(effect_basis="PSE delta", effect_value=0.01),
            _delta(effect_basis="map-based effect", effect_value=1.0),
            _delta(enabled=False, effect_basis="fuel delta", effect_value=-1.0),
        ]
        counts = delta_status_counts(deltas)
        self.assertEqual(counts["applied"], 1)
        self.assertEqual(counts["pending_model"], 1)
        self.assertEqual(counts["disabled"], 1)

    def test_maturity_rank_orders_by_declared_list(self):
        self.assertLess(maturity_rank("metadata_only"), maturity_rank("validated_against_test"))
        self.assertEqual(maturity_rank("not_a_real_level"), -1)

    def test_proposal_confidence_drops_to_low_if_any_applied_delta_is_low_confidence(self):
        deltas = [_delta(effect_basis="PSE delta", effect_value=0.01, confidence="low")]
        self.assertEqual(proposal_confidence_label("high", deltas), "low")

    def test_proposal_confidence_stable_high_when_no_registered_or_low_deltas(self):
        deltas = [_delta(effect_basis="PSE delta", effect_value=0.01, confidence="high")]
        self.assertEqual(proposal_confidence_label("high", deltas), "high")


class NoBaselineTests(unittest.TestCase):
    def test_none_baseline_returns_pending_status(self):
        result = apply_delta_stack_to_baseline(None, ctx={}, deltas=[_delta(effect_basis="PSE delta", effect_value=0.01)])
        self.assertEqual(result["status"], "Proposal pending")
        self.assertEqual(result["proposal"], {})


if __name__ == "__main__":
    unittest.main()
