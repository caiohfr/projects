import inspect
import json
import unittest

from src.vde_core.quick_scenario import (
    DomainReadiness,
    MassQuickChange,
    PseProvenance,
    QuickScenario,
    QuickVehicleReadiness,
    ReferencePressureProvenance,
    ScalarChange,
    ScalarChangeMode,
    TirePressureDelta,
    TireQuickChange,
    TireSource,
    TireTransformMode,
    VehicleQuickOverrides,
    build_quick_scenario_identity,
    quick_scenario_from_dict,
    to_serializable,
)
from src.vde_core.quick_scenario import contracts as quick_scenario_contracts
from src.vde_core.quick_scenario import serialization as quick_scenario_serialization


def _full_scenario() -> QuickScenario:
    return QuickScenario(
        source_identity="fc:42",
        slot=1,
        label="What if 20kg lighter?",
        vehicle_overrides=VehicleQuickOverrides(
            mass_change=MassQuickChange(curb_change=ScalarChange(mode=ScalarChangeMode.DELTA, value=-20.0)),
            cda_change=ScalarChange(mode=ScalarChangeMode.PERCENT, value=-5.0),
            tire_change=TireQuickChange(
                source=TireSource.TIRE_DB,
                transform_mode=TireTransformMode.PRESSURE_DELTA,
                tire_db_id=901,
                pressure_delta=TirePressureDelta(
                    front_delta_psi=2.0,
                    rear_delta_psi=1.5,
                    reference_pressure_psi=32.0,
                    reference_pressure_provenance=ReferencePressureProvenance.USER_PROVIDED,
                ),
            ),
        ),
        vehicle_readiness=QuickVehicleReadiness(
            mass=DomainReadiness.READY,
            aero=DomainReadiness.READY,
            tire=DomainReadiness.READY,
        ),
        final_pse_percent=27.5,
        pse_provenance=PseProvenance.USER_PROVIDED,
        issues=("Aero override review: percent basis assumed.",),
    )


class ScalarChangeResolutionTests(unittest.TestCase):
    def test_absolute_ignores_missing_source(self):
        change = ScalarChange(mode=ScalarChangeMode.ABSOLUTE, value=1500.0)
        self.assertEqual(change.resolve(None), 1500.0)

    def test_absolute_zero_is_explicit_not_blank(self):
        change = ScalarChange(mode=ScalarChangeMode.ABSOLUTE, value=0.0)
        self.assertEqual(change.resolve(1200.0), 0.0)

    def test_delta_requires_source_returns_none_when_missing(self):
        change = ScalarChange(mode=ScalarChangeMode.DELTA, value=-20.0)
        self.assertIsNone(change.resolve(None))

    def test_delta_adds_to_source(self):
        change = ScalarChange(mode=ScalarChangeMode.DELTA, value=-20.0)
        self.assertAlmostEqual(change.resolve(1500.0), 1480.0)

    def test_delta_zero_is_neutral(self):
        change = ScalarChange(mode=ScalarChangeMode.DELTA, value=0.0)
        self.assertEqual(change.resolve(1500.0), 1500.0)

    def test_percent_requires_source_returns_none_when_missing(self):
        change = ScalarChange(mode=ScalarChangeMode.PERCENT, value=10.0)
        self.assertIsNone(change.resolve(None))

    def test_percent_scales_source(self):
        change = ScalarChange(mode=ScalarChangeMode.PERCENT, value=-10.0)
        self.assertAlmostEqual(change.resolve(0.60), 0.54)

    def test_percent_zero_is_neutral(self):
        change = ScalarChange(mode=ScalarChangeMode.PERCENT, value=0.0)
        self.assertEqual(change.resolve(0.60), 0.60)


class VehicleQuickOverridesTests(unittest.TestCase):
    def test_blank_overrides_mean_no_override(self):
        overrides = VehicleQuickOverrides()
        self.assertTrue(overrides.is_empty)
        self.assertIsNone(overrides.mass_change)
        self.assertIsNone(overrides.cda_change)
        self.assertIsNone(overrides.tire_change)

    def test_any_populated_field_is_not_empty(self):
        overrides = VehicleQuickOverrides(
            mass_change=MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.DELTA, 0.0))
        )
        self.assertFalse(overrides.is_empty)

    def test_aero_reference_user_provided_requires_value(self):
        with self.assertRaises(ValueError):
            VehicleQuickOverrides(
                aero_reference_cda_provenance=ReferencePressureProvenance.USER_PROVIDED
            )

    def test_aero_reference_user_provided_with_value_is_valid(self):
        overrides = VehicleQuickOverrides(
            aero_reference_cda_m2=0.62,
            aero_reference_cda_provenance=ReferencePressureProvenance.USER_PROVIDED,
        )
        self.assertEqual(overrides.aero_reference_cda_m2, 0.62)

    def test_aero_reference_source_provenance_does_not_require_a_value(self):
        overrides = VehicleQuickOverrides(
            aero_reference_cda_provenance=ReferencePressureProvenance.SOURCE
        )
        self.assertIsNone(overrides.aero_reference_cda_m2)


class MassQuickChangeValidationTests(unittest.TestCase):
    def test_neither_curb_change_nor_twc_shift_is_rejected(self):
        with self.assertRaises(ValueError):
            MassQuickChange()

    def test_both_curb_change_and_twc_shift_is_rejected(self):
        with self.assertRaises(ValueError):
            MassQuickChange(
                curb_change=ScalarChange(ScalarChangeMode.DELTA, -20.0),
                twc_shift_steps=1.0,
            )

    def test_curb_change_alone_is_valid(self):
        change = MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.ABSOLUTE, 1500.0))
        self.assertIsNone(change.twc_shift_steps)

    def test_twc_shift_steps_alone_is_valid(self):
        change = MassQuickChange(twc_shift_steps=1.0, twc_shift_side="Up")
        self.assertIsNone(change.curb_change)

    def test_twc_shift_steps_zero_is_explicit_not_blank(self):
        change = MassQuickChange(twc_shift_steps=0.0)
        self.assertEqual(change.twc_shift_steps, 0.0)


class TireQuickChangeValidationTests(unittest.TestCase):
    def test_tire_db_requires_tire_db_id(self):
        with self.assertRaises(ValueError):
            TireQuickChange(source=TireSource.TIRE_DB, transform_mode=TireTransformMode.NONE)

    def test_tire_db_rejects_target_rrc(self):
        with self.assertRaises(ValueError):
            TireQuickChange(
                source=TireSource.TIRE_DB,
                tire_db_id=1,
                transform_mode=TireTransformMode.TARGET_RRC,
                target_rrc_n_per_kn=8.0,
            )

    def test_tire_db_rejects_rrc_delta(self):
        with self.assertRaises(ValueError):
            TireQuickChange(
                source=TireSource.TIRE_DB,
                tire_db_id=1,
                transform_mode=TireTransformMode.RRC_DELTA,
                rrc_delta_n_per_kn=-0.5,
            )

    def test_tire_db_allows_improvement_pct(self):
        change = TireQuickChange(
            source=TireSource.TIRE_DB,
            tire_db_id=1,
            transform_mode=TireTransformMode.IMPROVEMENT_PCT,
            improvement_pct=5.0,
        )
        self.assertEqual(change.improvement_pct, 5.0)

    def test_tire_db_allows_pressure_delta(self):
        change = TireQuickChange(
            source=TireSource.TIRE_DB,
            tire_db_id=1,
            transform_mode=TireTransformMode.PRESSURE_DELTA,
            pressure_delta=TirePressureDelta(front_delta_psi=1.0),
        )
        self.assertEqual(change.pressure_delta.front_delta_psi, 1.0)

    def test_current_allows_target_rrc(self):
        change = TireQuickChange(
            source=TireSource.CURRENT,
            transform_mode=TireTransformMode.TARGET_RRC,
            target_rrc_n_per_kn=7.5,
        )
        self.assertEqual(change.target_rrc_n_per_kn, 7.5)

    def test_current_allows_rrc_delta(self):
        change = TireQuickChange(
            source=TireSource.CURRENT,
            transform_mode=TireTransformMode.RRC_DELTA,
            rrc_delta_n_per_kn=-0.3,
        )
        self.assertEqual(change.rrc_delta_n_per_kn, -0.3)

    def test_transform_mode_requires_its_own_field(self):
        with self.assertRaises(ValueError):
            TireQuickChange(source=TireSource.CURRENT, transform_mode=TireTransformMode.TARGET_RRC)


class TirePressureDeltaProvenanceTests(unittest.TestCase):
    def test_user_provided_requires_reference_pressure(self):
        with self.assertRaises(ValueError):
            TirePressureDelta(
                front_delta_psi=2.0,
                reference_pressure_provenance=ReferencePressureProvenance.USER_PROVIDED,
            )

    def test_user_provided_with_reference_pressure_is_valid(self):
        delta = TirePressureDelta(
            front_delta_psi=2.0,
            reference_pressure_psi=32.0,
            reference_pressure_provenance=ReferencePressureProvenance.USER_PROVIDED,
        )
        self.assertEqual(delta.reference_pressure_psi, 32.0)

    def test_source_provenance_does_not_require_a_value(self):
        delta = TirePressureDelta(
            front_delta_psi=2.0,
            reference_pressure_provenance=ReferencePressureProvenance.SOURCE,
        )
        self.assertIsNone(delta.reference_pressure_psi)

    def test_rear_delta_none_means_same_as_front(self):
        delta = TirePressureDelta(front_delta_psi=2.0)
        self.assertIsNone(delta.rear_delta_psi)

    def test_split_front_rear_delta_is_supported(self):
        delta = TirePressureDelta(front_delta_psi=2.0, rear_delta_psi=1.0)
        self.assertEqual((delta.front_delta_psi, delta.rear_delta_psi), (2.0, 1.0))


class QuickVehicleReadinessTests(unittest.TestCase):
    def test_all_not_requested_is_ready(self):
        self.assertTrue(QuickVehicleReadiness().all_ready)

    def test_not_requested_domains_never_block_readiness(self):
        readiness = QuickVehicleReadiness(mass=DomainReadiness.READY)
        self.assertTrue(readiness.all_ready)

    def test_one_missing_requested_domain_blocks_readiness(self):
        readiness = QuickVehicleReadiness(mass=DomainReadiness.READY, tire=DomainReadiness.MISSING)
        self.assertFalse(readiness.all_ready)


class QuickScenarioIdentityTests(unittest.TestCase):
    def test_identity_format(self):
        self.assertEqual(build_quick_scenario_identity("fc:42", 1), "qs:fc:42:1")

    def test_identity_property_matches_helper(self):
        scenario = QuickScenario(source_identity="vde:900001", slot=2)
        self.assertEqual(scenario.identity, build_quick_scenario_identity("vde:900001", 2))

    def test_slot_zero_is_rejected(self):
        with self.assertRaises(ValueError):
            QuickScenario(source_identity="fc:1", slot=0)

    def test_slot_above_max_is_rejected(self):
        with self.assertRaises(ValueError):
            QuickScenario(source_identity="fc:1", slot=4)

    def test_empty_source_identity_is_rejected(self):
        with self.assertRaises(ValueError):
            QuickScenario(source_identity="", slot=1)

    def test_no_quick_to_quick_lineage(self):
        with self.assertRaises(ValueError):
            QuickScenario(source_identity="qs:fc:1:1", slot=1)

    def test_scenario_identity_preserves_full_comparison_identity_not_only_vde_id(self):
        # Sec 3: "Two FuelCons scenarios may legitimately share one VDE and
        # still represent different Comparison scenarios. Do not collapse
        # them." Both of these sources represent distinct FuelCons scenarios
        # (fc:1 / fc:2) that share one underlying vde_id (900001, per the
        # canonical_identity() convention audited in comparison_report_
        # viewmodels.py) -- their Quick Scenarios must stay distinct too.
        scenario_from_fc1 = QuickScenario(source_identity="fc:1", slot=1)
        scenario_from_fc2 = QuickScenario(source_identity="fc:2", slot=1)
        self.assertNotEqual(scenario_from_fc1.identity, scenario_from_fc2.identity)
        self.assertNotEqual(scenario_from_fc1.source_identity, scenario_from_fc2.source_identity)

    def test_same_source_different_slots_have_distinct_identity(self):
        first = QuickScenario(source_identity="fc:1", slot=1)
        second = QuickScenario(source_identity="fc:1", slot=2)
        third = QuickScenario(source_identity="fc:1", slot=3)
        identities = {first.identity, second.identity, third.identity}
        self.assertEqual(len(identities), 3)


class QuickScenarioFinalPseProvenanceTests(unittest.TestCase):
    def test_final_pse_requires_provenance(self):
        with self.assertRaises(ValueError):
            QuickScenario(source_identity="fc:1", slot=1, final_pse_percent=27.5)

    def test_provenance_requires_final_pse(self):
        with self.assertRaises(ValueError):
            QuickScenario(source_identity="fc:1", slot=1, pse_provenance=PseProvenance.USER_PROVIDED)

    def test_zero_final_pse_is_explicit_not_blank(self):
        scenario = QuickScenario(
            source_identity="fc:1",
            slot=1,
            final_pse_percent=0.0,
            pse_provenance=PseProvenance.USER_PROVIDED,
        )
        self.assertEqual(scenario.final_pse_percent, 0.0)

    def test_neither_final_pse_nor_provenance_is_valid(self):
        scenario = QuickScenario(source_identity="fc:1", slot=1)
        self.assertIsNone(scenario.final_pse_percent)
        self.assertIsNone(scenario.pse_provenance)

    def test_all_five_provenance_values_are_distinguishable(self):
        expected = {
            "INHERITED_CURRENT",
            "USER_PROVIDED",
            "BENCHMARK_ACCEPTED",
            "ML_RECOMMENDATION_ACCEPTED",
            "TECH_DELTA_ACCEPTED",
        }
        self.assertEqual({p.value for p in PseProvenance}, expected)
        for provenance in PseProvenance:
            scenario = QuickScenario(
                source_identity="fc:1",
                slot=1,
                final_pse_percent=25.0,
                pse_provenance=provenance,
            )
            self.assertIs(scenario.pse_provenance, provenance)

    def test_user_edit_after_accepted_recommendation_becomes_user_provided(self):
        # Sec 10: if the user manually edits a previously accepted
        # recommendation, provenance must reflect USER_PROVIDED rather than
        # pretending the exact recommendation remained authoritative. This
        # contract is immutable, so "editing" is modeled as constructing a
        # new QuickScenario with the edited value and USER_PROVIDED.
        accepted = QuickScenario(
            source_identity="fc:1",
            slot=1,
            final_pse_percent=28.0,
            pse_provenance=PseProvenance.ML_RECOMMENDATION_ACCEPTED,
        )
        edited = QuickScenario(
            source_identity="fc:1",
            slot=1,
            final_pse_percent=28.4,
            pse_provenance=PseProvenance.USER_PROVIDED,
        )
        self.assertNotEqual(accepted.pse_provenance, edited.pse_provenance)
        self.assertEqual(edited.pse_provenance, PseProvenance.USER_PROVIDED)


class QuickScenarioSerializationTests(unittest.TestCase):
    def test_minimal_scenario_round_trips(self):
        original = QuickScenario(source_identity="vde:900001", slot=3)
        payload = to_serializable(original)
        json_text = json.dumps(payload)
        restored = quick_scenario_from_dict(json.loads(json_text))
        self.assertEqual(restored, original)

    def test_full_scenario_round_trips(self):
        original = _full_scenario()
        payload = to_serializable(original)
        json_text = json.dumps(payload)
        restored = quick_scenario_from_dict(json.loads(json_text))
        self.assertEqual(restored, original)

    def test_serialized_payload_uses_plain_json_types(self):
        payload = to_serializable(_full_scenario())
        self.assertEqual(payload["vehicle_overrides"]["mass_change"]["curb_change"]["mode"], "DELTA")
        self.assertEqual(
            payload["vehicle_overrides"]["tire_change"]["source"],
            "TIRE_DB",
        )
        self.assertEqual(payload["pse_provenance"], "USER_PROVIDED")
        self.assertIsInstance(payload["issues"], list)

    def test_none_overrides_serialize_to_none_not_omitted(self):
        payload = to_serializable(QuickScenario(source_identity="fc:1", slot=1))
        self.assertIn("mass_change", payload["vehicle_overrides"])
        self.assertIsNone(payload["vehicle_overrides"]["mass_change"])


class QuickScenarioNoPersistenceTests(unittest.TestCase):
    def test_no_save_or_persistence_methods_exist(self):
        forbidden = {"save", "persist", "to_db_row", "write", "commit", "insert", "update"}
        present = forbidden & set(dir(QuickScenario))
        self.assertEqual(present, set())

    def test_no_db_row_identity_field_exists(self):
        field_names = set(QuickScenario.__dataclass_fields__)
        forbidden_fields = {"id", "db_id", "row_id", "fuelcons_id", "vde_id"}
        self.assertEqual(field_names & forbidden_fields, set())


class QuickScenarioNoStreamlitDependencyTests(unittest.TestCase):
    def test_contracts_module_does_not_import_streamlit(self):
        source = inspect.getsource(quick_scenario_contracts)
        self.assertNotIn("import streamlit", source.lower())

    def test_serialization_module_does_not_import_streamlit(self):
        source = inspect.getsource(quick_scenario_serialization)
        self.assertNotIn("import streamlit", source.lower())


if __name__ == "__main__":
    unittest.main()
