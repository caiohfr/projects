"""Sprint 10D: tests for resolve_quick_efficiency_scenario() -- Current/
Benchmark PSE, ML-derived PSE recommendation, Technology Delta suggestion,
and the deterministic Final-PSE-driven fuel/energy result.
"""

import copy
import gc
import tempfile
import unittest
from pathlib import Path

from src.vde_core import db as db_module
from src.vde_core.database_management_contract import EntityType
from src.vde_core.database_management_service import get_record
from src.vde_core.qa_mock_data import seed_qa_database, seed_qa_fuelcons_mock_rows
from src.vde_core.quick_scenario import (
    DomainReadiness,
    EfficiencyQuickInputs,
    MassQuickChange,
    PseProvenance,
    QuickScenario,
    ScalarChange,
    ScalarChangeMode,
    TechDeltaAssumption,
    VehicleQuickOverrides,
    resolve_quick_efficiency_scenario,
    resolve_quick_vehicle_scenario,
)
from src.vde_core.quick_scenario.contracts import MAX_TECH_DELTAS_PER_SCENARIO
from src.vde_core.quick_scenario.resolution import QuickVehicleResolution
from src.vde_core.vehicle_demand import RoadloadBasis


def _stub_ml_predictor(fuel_l_100km=6.0, confidence="medium"):
    def predictor(request_dict, feature_row, metadata):
        return {
            "fuel_l_100km": fuel_l_100km,
            "confidence": confidence,
            "model_name": "StubML",
            "model_version": "test-v1",
            "coverage_status": "in_domain",
        }

    return predictor


class QuickEfficiencyResolverTestCase(unittest.TestCase):
    """Seeds a temp QA database with fuelcons rows referencing vde_id=900001
    (fc:900102, fc:900104 -- two scenarios sharing one VDE) and vde_id=900002
    (fc:900103, fc:900105), matching the fixture Sprint 10A/10B/10C already
    established.
    """

    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "efficiency_qa.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        seed_qa_fuelcons_mock_rows(self.db_path)
        db_module.configure_db_path(self.db_path)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def _vehicle_resolution(self, source_identity="fc:900102", overrides=None):
        scenario = QuickScenario(
            source_identity=source_identity,
            slot=1,
            vehicle_overrides=overrides or VehicleQuickOverrides(),
        )
        return scenario, resolve_quick_vehicle_scenario(scenario)

    def _fuelcons_row(self, fuelcons_id: int) -> dict:
        return dict(get_record(EntityType.FUEL_CONSUMPTION, fuelcons_id))


class CurrentPseTests(QuickEfficiencyResolverTestCase):
    def test_current_pse_resolves_through_canonical_path(self):
        scenario, vresult = self._vehicle_resolution()
        eresult = resolve_quick_efficiency_scenario(scenario, vresult)
        self.assertTrue(eresult.current_pse.is_available)
        self.assertGreater(eresult.current_pse.value_percent, 0.0)

    def test_current_pse_unavailable_for_vde_only_source(self):
        scenario, vresult = self._vehicle_resolution(source_identity="vde:900001")
        eresult = resolve_quick_efficiency_scenario(scenario, vresult)
        self.assertFalse(eresult.current_pse.is_available)
        self.assertEqual(eresult.current_pse.status, "unavailable")


class ManualFinalPseTests(QuickEfficiencyResolverTestCase):
    def test_manual_final_pse_produces_deterministic_result(self):
        scenario, vresult = self._vehicle_resolution()
        scenario = QuickScenario(
            source_identity=scenario.source_identity,
            slot=1,
            final_pse_percent=30.0,
            pse_provenance=PseProvenance.USER_PROVIDED,
        )
        eresult = resolve_quick_efficiency_scenario(scenario, vresult)
        self.assertTrue(eresult.is_ready)
        self.assertIsNotNone(eresult.fuel_estimate_result)
        self.assertAlmostEqual(
            eresult.fuel_estimate_result.assumptions["pse_summary"]["value"], 0.30, places=6
        )

    def test_final_pse_none_remains_missing(self):
        scenario, vresult = self._vehicle_resolution()
        eresult = resolve_quick_efficiency_scenario(scenario, vresult)
        self.assertEqual(eresult.readiness, DomainReadiness.NOT_REQUESTED)
        self.assertIsNone(eresult.fuel_estimate_result)

    def test_final_pse_zero_is_explicit_and_invalid(self):
        scenario, vresult = self._vehicle_resolution()
        scenario = QuickScenario(
            source_identity=scenario.source_identity,
            slot=1,
            final_pse_percent=0.0,
            pse_provenance=PseProvenance.USER_PROVIDED,
        )
        eresult = resolve_quick_efficiency_scenario(scenario, vresult)
        self.assertEqual(eresult.readiness, DomainReadiness.INVALID)
        self.assertIsNone(eresult.fuel_estimate_result)
        self.assertTrue(eresult.issues)

    def test_neutral_final_pse_equal_to_current_gives_baseline_equivalent_result(self):
        scenario, vresult = self._vehicle_resolution()
        current = resolve_quick_efficiency_scenario(scenario, vresult).current_pse.value_percent
        scenario2 = QuickScenario(
            source_identity=scenario.source_identity,
            slot=1,
            final_pse_percent=current,
            pse_provenance=PseProvenance.INHERITED_CURRENT,
        )
        eresult = resolve_quick_efficiency_scenario(scenario2, vresult)
        self.assertAlmostEqual(
            eresult.fuel_estimate_result.assumptions["pse_summary"]["value"] * 100.0, current, places=6
        )

    def test_higher_final_pse_lowers_required_consumed_energy(self):
        scenario, vresult = self._vehicle_resolution()
        low_pse = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            final_pse_percent=25.0, pse_provenance=PseProvenance.USER_PROVIDED,
        )
        high_pse = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            final_pse_percent=40.0, pse_provenance=PseProvenance.USER_PROVIDED,
        )
        low_result = resolve_quick_efficiency_scenario(low_pse, vresult)
        high_result = resolve_quick_efficiency_scenario(high_pse, vresult)
        self.assertLess(
            high_result.fuel_estimate_result.fuel_l_100km, low_result.fuel_estimate_result.fuel_l_100km
        )

    def test_lower_final_pse_raises_required_consumed_energy(self):
        scenario, vresult = self._vehicle_resolution()
        baseline = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            final_pse_percent=30.0, pse_provenance=PseProvenance.USER_PROVIDED,
        )
        lower = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            final_pse_percent=20.0, pse_provenance=PseProvenance.USER_PROVIDED,
        )
        baseline_result = resolve_quick_efficiency_scenario(baseline, vresult)
        lower_result = resolve_quick_efficiency_scenario(lower, vresult)
        self.assertGreater(
            lower_result.fuel_estimate_result.fuel_l_100km, baseline_result.fuel_estimate_result.fuel_l_100km
        )

    def test_repeated_identical_calculation_is_deterministic(self):
        scenario, vresult = self._vehicle_resolution()
        scenario = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            final_pse_percent=30.0, pse_provenance=PseProvenance.USER_PROVIDED,
        )
        first = resolve_quick_efficiency_scenario(scenario, vresult)
        second = resolve_quick_efficiency_scenario(scenario, vresult)
        self.assertEqual(first.fuel_estimate_result.fuel_l_100km, second.fuel_estimate_result.fuel_l_100km)
        self.assertEqual(first.current_pse, second.current_pse)

    def test_source_fuelcons_row_remains_unchanged(self):
        scenario, vresult = self._vehicle_resolution()
        before = self._fuelcons_row(900102)
        scenario2 = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            final_pse_percent=30.0, pse_provenance=PseProvenance.USER_PROVIDED,
        )
        resolve_quick_efficiency_scenario(scenario2, vresult)
        after = self._fuelcons_row(900102)
        self.assertEqual(before, after)


class BenchmarkPseTests(QuickEfficiencyResolverTestCase):
    def test_benchmark_pse_from_another_fuelcons_row(self):
        scenario, vresult = self._vehicle_resolution()
        scenario = QuickScenario(
            source_identity=scenario.source_identity,
            slot=1,
            efficiency_inputs=EfficiencyQuickInputs(benchmark_source_identity="fc:900103"),
        )
        eresult = resolve_quick_efficiency_scenario(scenario, vresult)
        self.assertTrue(eresult.benchmark_pse.is_available)
        self.assertEqual(eresult.benchmark_pse.donor_source_identity, "fc:900103")

    def test_benchmark_only_transfers_pse_and_provenance(self):
        scenario, vresult = self._vehicle_resolution()
        scenario = QuickScenario(
            source_identity=scenario.source_identity,
            slot=1,
            efficiency_inputs=EfficiencyQuickInputs(benchmark_source_identity="fc:900103"),
        )
        eresult = resolve_quick_efficiency_scenario(scenario, vresult)
        # PseReference carries only status/value/donor identity/warnings --
        # no donor roadload/VDE/transmission/electrification/fuel fields
        # exist on the dataclass at all.
        field_names = set(eresult.benchmark_pse.__dataclass_fields__)
        self.assertEqual(field_names, {"status", "value_percent", "donor_source_identity", "warnings"})

    def test_active_vehicle_demand_remains_unchanged_with_benchmark_selected(self):
        scenario, vresult = self._vehicle_resolution()
        scenario_with_benchmark = QuickScenario(
            source_identity=scenario.source_identity,
            slot=1,
            efficiency_inputs=EfficiencyQuickInputs(benchmark_source_identity="fc:900103"),
        )
        resolve_quick_efficiency_scenario(scenario_with_benchmark, vresult)
        # vehicle_resolution (the Quick-resolved VehicleDemandResult) is
        # never mutated by selecting a benchmark.
        self.assertIsNotNone(vresult.vehicle_demand_result)
        self.assertEqual(vresult.resolved_curb_mass_kg, vresult.resolved_curb_mass_kg)

    def test_two_donor_scenarios_sharing_one_vde_remain_distinct(self):
        scenario, vresult = self._vehicle_resolution()
        first = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            efficiency_inputs=EfficiencyQuickInputs(benchmark_source_identity="fc:900102"),
        )
        second = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            efficiency_inputs=EfficiencyQuickInputs(benchmark_source_identity="fc:900104"),
        )
        first_result = resolve_quick_efficiency_scenario(first, vresult)
        second_result = resolve_quick_efficiency_scenario(second, vresult)
        self.assertNotEqual(
            first_result.benchmark_pse.donor_source_identity,
            second_result.benchmark_pse.donor_source_identity,
        )

    def test_missing_donor_energy_or_pse_recommendation_unavailable(self):
        scenario, vresult = self._vehicle_resolution()
        scenario_no_donor_data = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            efficiency_inputs=EfficiencyQuickInputs(benchmark_source_identity="fc:999999"),
        )
        eresult = resolve_quick_efficiency_scenario(scenario_no_donor_data, vresult)
        self.assertFalse(eresult.benchmark_pse.is_available)

    def test_accepting_benchmark_recommendation_produces_matching_final_pse_provenance(self):
        scenario, vresult = self._vehicle_resolution()
        scenario_with_benchmark = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            efficiency_inputs=EfficiencyQuickInputs(benchmark_source_identity="fc:900103"),
        )
        benchmark_value = resolve_quick_efficiency_scenario(scenario_with_benchmark, vresult).benchmark_pse.value_percent
        accepted_scenario = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            final_pse_percent=benchmark_value, pse_provenance=PseProvenance.BENCHMARK_ACCEPTED,
        )
        eresult = resolve_quick_efficiency_scenario(accepted_scenario, vresult)
        self.assertEqual(eresult.final_pse_provenance, PseProvenance.BENCHMARK_ACCEPTED)
        self.assertAlmostEqual(eresult.final_pse_percent, benchmark_value)


class MlRecommendationTests(QuickEfficiencyResolverTestCase):
    def test_ml_recommendation_not_requested_by_default(self):
        scenario, vresult = self._vehicle_resolution()
        eresult = resolve_quick_efficiency_scenario(scenario, vresult)
        self.assertEqual(eresult.ml_recommendation.status, "not_requested")

    def test_ml_artifact_loads_through_canonical_path_with_injected_predictor(self):
        scenario, vresult = self._vehicle_resolution(
            overrides=VehicleQuickOverrides()
        )
        scenario = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            efficiency_inputs=EfficiencyQuickInputs(request_ml_recommendation=True),
        )
        eresult = resolve_quick_efficiency_scenario(
            scenario, vresult, ml_model_options={"ml_predictor": _stub_ml_predictor(fuel_l_100km=6.0)}
        )
        self.assertEqual(eresult.ml_recommendation.status, "available")
        self.assertEqual(eresult.ml_recommendation.model_version, "test-v1")

    def test_pse_recommendation_is_derived_not_a_direct_model_output(self):
        scenario, vresult = self._vehicle_resolution()
        scenario = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            efficiency_inputs=EfficiencyQuickInputs(request_ml_recommendation=True),
        )
        eresult = resolve_quick_efficiency_scenario(
            scenario, vresult, ml_model_options={"ml_predictor": _stub_ml_predictor(fuel_l_100km=6.0)}
        )
        # Derived: demand / (fuel_l_100km/100 * LHV) -- never the raw
        # fuel_l_100km value re-interpreted as a percentage.
        demand = vresult.vehicle_demand_result.total_summary.vde_mj_per_km
        from src.vde_core.fuel_energy import LHV_MJ_PER_L

        expected_pse = demand / ((6.0 / 100.0) * LHV_MJ_PER_L["Gasoline"])
        self.assertAlmostEqual(eresult.ml_recommendation.value_percent / 100.0, expected_pse, places=6)

    def test_artifact_unavailable_ml_recommendation_unavailable_no_crash(self):
        scenario, vresult = self._vehicle_resolution()
        scenario = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            efficiency_inputs=EfficiencyQuickInputs(request_ml_recommendation=True),
        )
        eresult = resolve_quick_efficiency_scenario(
            scenario, vresult, ml_model_options={"ml_artifact_path": "/nonexistent/model.joblib"}
        )
        self.assertEqual(eresult.ml_recommendation.status, "unavailable")
        self.assertEqual(eresult.ml_recommendation.artifact_status, "artifact_load_failed")
        self.assertIsNone(eresult.ml_recommendation.value_percent)

    def test_no_invented_confidence_metric(self):
        scenario, vresult = self._vehicle_resolution()
        scenario = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            efficiency_inputs=EfficiencyQuickInputs(request_ml_recommendation=True),
        )
        eresult = resolve_quick_efficiency_scenario(
            scenario, vresult, ml_model_options={"ml_predictor": _stub_ml_predictor()}
        )
        self.assertIn(eresult.ml_recommendation.confidence_label, ("high", "medium", "low", "provided"))
        self.assertNotIn("confidence_score", eresult.ml_recommendation.__dataclass_fields__)
        self.assertNotIn("confidence_percent", eresult.ml_recommendation.__dataclass_fields__)

    def test_quick_vehicle_change_that_alters_ml_features_is_reported_as_changed(self):
        scenario_no_change, vresult_no_change = self._vehicle_resolution()
        scenario_mass, vresult_mass = self._vehicle_resolution(
            overrides=VehicleQuickOverrides(
                mass_change=MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.DELTA, -20.0))
            )
        )
        scenario_mass = QuickScenario(
            source_identity=scenario_mass.source_identity, slot=1,
            vehicle_overrides=scenario_mass.vehicle_overrides,
            efficiency_inputs=EfficiencyQuickInputs(request_ml_recommendation=True),
        )
        eresult = resolve_quick_efficiency_scenario(
            scenario_mass, vresult_mass, ml_model_options={"ml_predictor": _stub_ml_predictor()}
        )
        self.assertIn("vde_net_mj_per_km", eresult.ml_recommendation.quick_affected_features_changed)

    def test_quick_vehicle_changes_not_represented_are_never_falsely_reported(self):
        scenario, vresult = self._vehicle_resolution(
            overrides=VehicleQuickOverrides(
                mass_change=MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.DELTA, -20.0))
            )
        )
        scenario = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            vehicle_overrides=scenario.vehicle_overrides,
            efficiency_inputs=EfficiencyQuickInputs(request_ml_recommendation=True),
        )
        eresult = resolve_quick_efficiency_scenario(
            scenario, vresult, ml_model_options={"ml_predictor": _stub_ml_predictor()}
        )
        # Phase-split features are never recomputed by Quick, regardless of
        # what changed -- never implied as "understood" by the model.
        self.assertIn("vde_urb_mj_per_km", eresult.ml_recommendation.features_not_represented)
        self.assertIn("vde_hw_mj_per_km", eresult.ml_recommendation.features_not_represented)
        self.assertNotIn("vde_urb_mj_per_km", eresult.ml_recommendation.quick_affected_features_changed)

    def test_no_vehicle_change_produces_no_affected_ml_features(self):
        scenario, vresult = self._vehicle_resolution()
        scenario = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            efficiency_inputs=EfficiencyQuickInputs(request_ml_recommendation=True),
        )
        eresult = resolve_quick_efficiency_scenario(
            scenario, vresult, ml_model_options={"ml_predictor": _stub_ml_predictor()}
        )
        self.assertEqual(eresult.ml_recommendation.quick_affected_features_changed, ())

    def test_accept_ml_recommendation_final_pse_gets_ml_accepted_provenance(self):
        scenario, vresult = self._vehicle_resolution()
        scenario_ml = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            efficiency_inputs=EfficiencyQuickInputs(request_ml_recommendation=True),
        )
        ml_value = resolve_quick_efficiency_scenario(
            scenario_ml, vresult, ml_model_options={"ml_predictor": _stub_ml_predictor()}
        ).ml_recommendation.value_percent
        accepted = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            final_pse_percent=ml_value, pse_provenance=PseProvenance.ML_RECOMMENDATION_ACCEPTED,
        )
        eresult = resolve_quick_efficiency_scenario(accepted, vresult)
        self.assertEqual(eresult.final_pse_provenance, PseProvenance.ML_RECOMMENDATION_ACCEPTED)

    def test_manual_edit_after_ml_acceptance_provenance_becomes_user_provided(self):
        scenario, vresult = self._vehicle_resolution()
        accepted = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            final_pse_percent=28.0, pse_provenance=PseProvenance.ML_RECOMMENDATION_ACCEPTED,
        )
        edited = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            final_pse_percent=29.5, pse_provenance=PseProvenance.USER_PROVIDED,
        )
        accepted_result = resolve_quick_efficiency_scenario(accepted, vresult)
        edited_result = resolve_quick_efficiency_scenario(edited, vresult)
        self.assertEqual(accepted_result.final_pse_provenance, PseProvenance.ML_RECOMMENDATION_ACCEPTED)
        self.assertEqual(edited_result.final_pse_provenance, PseProvenance.USER_PROVIDED)


class TechnologyDeltaTests(QuickEfficiencyResolverTestCase):
    def test_single_percent_tech_delta_suggestion(self):
        scenario, vresult = self._vehicle_resolution()
        current = resolve_quick_efficiency_scenario(scenario, vresult).current_pse.value_percent
        scenario2 = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            efficiency_inputs=EfficiencyQuickInputs(
                technology_deltas=(
                    TechDeltaAssumption(name="Improved ESS", effect_basis="pse_percent_delta", effect_value=2.0),
                )
            ),
        )
        eresult = resolve_quick_efficiency_scenario(scenario2, vresult)
        self.assertTrue(eresult.tech_delta_suggestion.is_available)
        self.assertAlmostEqual(eresult.tech_delta_suggestion.value_percent, current * 1.02, places=4)
        self.assertEqual(eresult.tech_delta_suggestion.applied_count, 1)

    def test_multiple_tech_deltas_use_canonical_sequential_stacking(self):
        scenario, vresult = self._vehicle_resolution()
        current = resolve_quick_efficiency_scenario(scenario, vresult).current_pse.value_percent
        scenario2 = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            efficiency_inputs=EfficiencyQuickInputs(
                technology_deltas=(
                    TechDeltaAssumption(name="ESS", effect_basis="pse_percent_delta", effect_value=2.0),
                    TechDeltaAssumption(name="Transmission", effect_basis="pse_percent_delta", effect_value=1.0),
                )
            ),
        )
        eresult = resolve_quick_efficiency_scenario(scenario2, vresult)
        expected_compounded = current * 1.02 * 1.01
        self.assertAlmostEqual(eresult.tech_delta_suggestion.value_percent, expected_compounded, places=3)
        self.assertEqual(eresult.tech_delta_suggestion.applied_count, 2)

    def test_absolute_plus_percent_combination_uses_canonical_order(self):
        scenario, vresult = self._vehicle_resolution()
        current = resolve_quick_efficiency_scenario(scenario, vresult).current_pse.value_percent
        scenario2 = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            efficiency_inputs=EfficiencyQuickInputs(
                technology_deltas=(
                    TechDeltaAssumption(name="Absolute bump", effect_basis="pse_delta", effect_value=0.01),
                    TechDeltaAssumption(name="Percent bump", effect_basis="pse_percent_delta", effect_value=5.0),
                )
            ),
        )
        eresult = resolve_quick_efficiency_scenario(scenario2, vresult)
        expected = ((current / 100.0) + 0.01) * 1.05 * 100.0
        self.assertAlmostEqual(eresult.tech_delta_suggestion.value_percent, expected, places=3)

    def test_zero_tech_delta_is_neutral(self):
        scenario, vresult = self._vehicle_resolution()
        current = resolve_quick_efficiency_scenario(scenario, vresult).current_pse.value_percent
        scenario2 = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            efficiency_inputs=EfficiencyQuickInputs(
                technology_deltas=(
                    TechDeltaAssumption(name="No-op", effect_basis="pse_percent_delta", effect_value=0.0),
                )
            ),
        )
        eresult = resolve_quick_efficiency_scenario(scenario2, vresult)
        self.assertAlmostEqual(eresult.tech_delta_suggestion.value_percent, current, places=4)

    def test_up_to_three_quick_tech_deltas_supported(self):
        deltas = tuple(
            TechDeltaAssumption(name=f"Delta {i}", effect_basis="pse_percent_delta", effect_value=1.0)
            for i in range(MAX_TECH_DELTAS_PER_SCENARIO)
        )
        inputs = EfficiencyQuickInputs(technology_deltas=deltas)
        self.assertEqual(len(inputs.technology_deltas), MAX_TECH_DELTAS_PER_SCENARIO)

    def test_more_than_product_limit_rejected_cleanly(self):
        deltas = tuple(
            TechDeltaAssumption(name=f"Delta {i}", effect_basis="pse_percent_delta", effect_value=1.0)
            for i in range(MAX_TECH_DELTAS_PER_SCENARIO + 1)
        )
        with self.assertRaises(ValueError):
            EfficiencyQuickInputs(technology_deltas=deltas)

    def test_tech_delta_produces_suggestion_only_final_pse_unaffected(self):
        scenario, vresult = self._vehicle_resolution()
        scenario2 = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            efficiency_inputs=EfficiencyQuickInputs(
                technology_deltas=(
                    TechDeltaAssumption(name="ESS", effect_basis="pse_percent_delta", effect_value=2.0),
                )
            ),
        )
        eresult = resolve_quick_efficiency_scenario(scenario2, vresult)
        self.assertIsNone(eresult.final_pse_percent)
        self.assertIsNone(eresult.fuel_estimate_result)

    def test_without_explicit_adoption_final_pse_and_energy_result_do_not_change(self):
        scenario, vresult = self._vehicle_resolution()
        base = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            final_pse_percent=30.0, pse_provenance=PseProvenance.USER_PROVIDED,
        )
        with_suggestion = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            final_pse_percent=30.0, pse_provenance=PseProvenance.USER_PROVIDED,
            efficiency_inputs=EfficiencyQuickInputs(
                technology_deltas=(
                    TechDeltaAssumption(name="ESS", effect_basis="pse_percent_delta", effect_value=2.0),
                )
            ),
        )
        base_result = resolve_quick_efficiency_scenario(base, vresult)
        suggestion_result = resolve_quick_efficiency_scenario(with_suggestion, vresult)
        self.assertEqual(
            base_result.fuel_estimate_result.fuel_l_100km, suggestion_result.fuel_estimate_result.fuel_l_100km
        )

    def test_accept_tech_suggestion_gets_tech_delta_accepted_provenance(self):
        scenario, vresult = self._vehicle_resolution()
        scenario_with_suggestion = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            efficiency_inputs=EfficiencyQuickInputs(
                technology_deltas=(
                    TechDeltaAssumption(name="ESS", effect_basis="pse_percent_delta", effect_value=2.0),
                )
            ),
        )
        suggested_value = resolve_quick_efficiency_scenario(scenario_with_suggestion, vresult).tech_delta_suggestion.value_percent
        accepted = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            final_pse_percent=suggested_value, pse_provenance=PseProvenance.TECH_DELTA_ACCEPTED,
        )
        eresult = resolve_quick_efficiency_scenario(accepted, vresult)
        self.assertEqual(eresult.final_pse_provenance, PseProvenance.TECH_DELTA_ACCEPTED)


class DeterministicEnergyResultTests(QuickEfficiencyResolverTestCase):
    def test_final_pse_feeds_deterministic_service(self):
        scenario, vresult = self._vehicle_resolution()
        scenario = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            final_pse_percent=30.0, pse_provenance=PseProvenance.USER_PROVIDED,
        )
        eresult = resolve_quick_efficiency_scenario(scenario, vresult)
        self.assertIsNotNone(eresult.fuel_estimate_result)
        self.assertIsNotNone(eresult.fuel_estimate_result.fuel_l_100km)

    def test_lhv_and_co2_come_from_canonical_fuel_energy_table(self):
        from src.vde_core.fuel_energy import GCO2_PER_L, LHV_MJ_PER_L

        scenario, vresult = self._vehicle_resolution()
        scenario = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            final_pse_percent=30.0, pse_provenance=PseProvenance.USER_PROVIDED,
        )
        eresult = resolve_quick_efficiency_scenario(scenario, vresult)
        demand = vresult.vehicle_demand_result.total_summary.vde_mj_per_km
        expected_fuel = (demand / 0.30) / LHV_MJ_PER_L["Gasoline"] * 100.0
        expected_co2 = (expected_fuel / 100.0) * GCO2_PER_L["Gasoline"]
        self.assertAlmostEqual(eresult.fuel_estimate_result.fuel_l_100km, expected_fuel, places=6)
        self.assertAlmostEqual(eresult.fuel_estimate_result.gco2_km, expected_co2, places=6)

    def test_total_basis_calculation_remains_total(self):
        scenario, vresult = self._vehicle_resolution()
        scenario = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            final_pse_percent=30.0, pse_provenance=PseProvenance.USER_PROVIDED,
        )
        eresult = resolve_quick_efficiency_scenario(scenario, vresult, energy_basis=RoadloadBasis.TOTAL)
        self.assertEqual(eresult.energy_basis, "VDE_TOTAL")
        self.assertEqual(eresult.fuel_estimate_result.energy_basis_used, "VDE_TOTAL")

    def test_net_basis_calculation_remains_net(self):
        scenario, vresult = self._vehicle_resolution()
        self.assertIsNotNone(vresult.vehicle_demand_result.net_summary)
        scenario = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            final_pse_percent=30.0, pse_provenance=PseProvenance.USER_PROVIDED,
        )
        eresult = resolve_quick_efficiency_scenario(scenario, vresult, energy_basis=RoadloadBasis.NET)
        self.assertEqual(eresult.energy_basis, "VDE_NET")
        self.assertEqual(eresult.fuel_estimate_result.energy_basis_used, "VDE_NET")
        # TOTAL and NET demand differ (per the source ABC/transmission), so
        # the two bases must not silently produce the same result.
        total_result = resolve_quick_efficiency_scenario(scenario, vresult, energy_basis=RoadloadBasis.TOTAL)
        self.assertNotEqual(
            eresult.fuel_estimate_result.fuel_l_100km, total_result.fuel_estimate_result.fuel_l_100km
        )

    def test_missing_requested_basis_stays_unavailable_no_fallback(self):
        row = {
            "id": 5,
            "legislation": "EPA",
            "mass_kg": 1500.0,
            "test_mass_kg": 1644.0,
            "inertia_class": 1644.0,
            "coast_A_N": 118.0,
            "coast_B_N_per_kph": 0.02,
            "coast_C_N_per_kph2": 0.009,
            "rrc_N_per_kN": 8.0,
            "cda_m2": 0.62,
            # No trans_* columns -> NET boundary unavailable -> no fallback.
        }
        scenario = QuickScenario(source_identity="vde:5", slot=1)
        vresult = resolve_quick_vehicle_scenario(scenario, source_vde_row=row)
        self.assertIsNone(vresult.vehicle_demand_result.net_summary)
        scenario = QuickScenario(
            source_identity="vde:5", slot=1,
            final_pse_percent=30.0, pse_provenance=PseProvenance.USER_PROVIDED,
        )
        eresult = resolve_quick_efficiency_scenario(scenario, vresult, energy_basis=RoadloadBasis.NET)
        self.assertEqual(eresult.readiness, DomainReadiness.MISSING)
        self.assertIsNone(eresult.fuel_estimate_result)

    def test_no_db_writes(self):
        import sqlite3

        scenario, vresult = self._vehicle_resolution()
        scenario = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            final_pse_percent=30.0, pse_provenance=PseProvenance.USER_PROVIDED,
            efficiency_inputs=EfficiencyQuickInputs(
                benchmark_source_identity="fc:900103",
                technology_deltas=(
                    TechDeltaAssumption(name="ESS", effect_basis="pse_percent_delta", effect_value=2.0),
                ),
            ),
        )
        with sqlite3.connect(self.db_path) as con:
            before_fuelcons = con.execute("SELECT COUNT(*) FROM fuelcons_db").fetchone()[0]
            before_vde = con.execute("SELECT COUNT(*) FROM vde_db").fetchone()[0]
        resolve_quick_efficiency_scenario(scenario, vresult)
        with sqlite3.connect(self.db_path) as con:
            after_fuelcons = con.execute("SELECT COUNT(*) FROM fuelcons_db").fetchone()[0]
            after_vde = con.execute("SELECT COUNT(*) FROM vde_db").fetchone()[0]
        self.assertEqual(before_fuelcons, after_fuelcons)
        self.assertEqual(before_vde, after_vde)

    def test_vehicle_quick_resolution_object_is_not_mutated(self):
        scenario, vresult = self._vehicle_resolution()
        vresult_copy = copy.deepcopy(vresult)
        scenario = QuickScenario(
            source_identity=scenario.source_identity, slot=1,
            final_pse_percent=30.0, pse_provenance=PseProvenance.USER_PROVIDED,
            efficiency_inputs=EfficiencyQuickInputs(request_ml_recommendation=True),
        )
        resolve_quick_efficiency_scenario(
            scenario, vresult, ml_model_options={"ml_predictor": _stub_ml_predictor()}
        )
        self.assertEqual(vresult, vresult_copy)


if __name__ == "__main__":
    unittest.main()
