"""Sprint 10C Quick Tire behavior and canonical resolver-state parity.

Expected physical values are obtained from independent
``resolve_tire_proposal`` calls. The tests never reproduce RRC, pressure,
load, or Tire ABC formulas.
"""

from __future__ import annotations

import copy
import gc
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.vde_core import db as db_module
from src.vde_core.comparison_report_service import resolve_roadload_boundaries
from src.vde_core.qa_mock_data import seed_qa_database
from src.vde_core.quick_scenario import (
    DomainReadiness,
    MassQuickChange,
    QuickScenario,
    ReferencePressureProvenance,
    ScalarChange,
    ScalarChangeMode,
    TirePressureDelta,
    TireQuickChange,
    TireSource,
    TireTransformMode,
    VehicleQuickOverrides,
    resolve_quick_vehicle_scenario,
)
from src.vde_core.vde_mass_proposal_resolver import resolve_mass_proposal
from src.vde_core.tire_roadload_service import get_tire_by_code, get_tire_by_id
from src.vde_core.vde_tire_proposal_resolver import resolve_tire_proposal
from src.vde_core.vehicle_demand.adapters import build_vehicle_demand_request


def _row(**updates) -> dict:
    value = {
        "id": 1,
        "legislation": "EPA",
        "mass_kg": 1500.0,
        "test_mass_kg": 1644.0,
        "inertia_class": 1644.0,
        "tire_load_mass_basis": "TWC",
        "tire_load_mass_used_kg": 1644.0,
        "weight_dist_fr_pct": 55.0,
        "coast_A_N": 118.0,
        "coast_B_N_per_kph": 0.0200,
        "coast_C_N_per_kph2": 0.0090,
        "trans_A_coef_N": 8.5,
        "trans_B_coef_Npkph": 0.0040,
        "trans_C_coef_Npkph2": 0.0008,
        "rrc_N_per_kN": 8.0,
        "tire_A_final": 50.0,
        "tire_B_final": 0.010,
        "tire_C_final": 0.0010,
        "front_pressure_psi": 38.0,
        "rear_pressure_psi": 38.0,
        "cda_m2": 0.620,
    }
    value.update(updates)
    return value


def _iso_tire(**updates) -> dict:
    value = {
        "id": 99,
        "tire_test_code": "ISO-99",
        "standard_family": "ISO",
        "rr_n_per_kn": 8.6,
        "test_pressure_value": 36.0,
        "pressure_unit": "psi",
    }
    value.update(updates)
    return value


def _sae_tire(**updates) -> dict:
    value = {
        "id": 100,
        "tire_test_code": "SAE-100",
        "standard_family": "SAE",
        "rr_n_per_kn": 8.4,
        "sae_alpha": 0.1,
        "sae_beta": 1.0,
        "sae_a": 0.05,
        "sae_b": 0.0002,
        "sae_c": 0.0,
        "sae_reference_load_n": 3000.0,
        "sae_reference_pressure_kpa": 220.0,
    }
    value.update(updates)
    return value


def _scenario(tire_change, *, mass_change=None, cda_change=None, source_identity="vde:1"):
    return QuickScenario(
        source_identity=source_identity,
        slot=1,
        vehicle_overrides=VehicleQuickOverrides(
            mass_change=mass_change,
            tire_change=tire_change,
            cda_change=cda_change,
        ),
    )


def _resolve(tire_change, *, row=None, tire_record=None, mass_change=None, cda_change=None):
    scenario = _scenario(tire_change, mass_change=mass_change, cda_change=cda_change)
    if tire_record is None:
        return resolve_quick_vehicle_scenario(scenario, source_vde_row=row or _row())
    with mock.patch(
        "src.vde_core.quick_scenario.resolver.get_tire_by_id",
        return_value=copy.deepcopy(tire_record),
    ):
        return resolve_quick_vehicle_scenario(scenario, source_vde_row=row or _row())


def _apply_canonical_tire(row: dict, outcome: dict) -> dict:
    expected = dict(row)
    resolved = outcome["resolved_snapshot"]
    expected.update(
        {
            key: resolved.get(key)
            for key in (
                "tire_db_id",
                "tire_code",
                "tire_snapshot",
                "front_pressure_psi",
                "rear_pressure_psi",
                "rrc_N_per_kN",
                "tire_A_final",
                "tire_B_final",
                "tire_C_final",
                "tire_load_mass_basis",
                "tire_load_mass_used_kg",
            )
            if key in resolved
        }
    )
    for field, component in (
        ("coast_A_N", "A"),
        ("coast_B_N_per_kph", "B"),
        ("coast_C_N_per_kph2", "C"),
    ):
        if expected.get(field) is not None and resolved["tire_delta_abc"].get(component) is not None:
            expected[field] += resolved["tire_delta_abc"][component]
    return expected


def _assert_parity(test: unittest.TestCase, actual, expected_outcome: dict, expected_row: dict):
    resolved = expected_outcome["resolved_snapshot"]
    test.assertAlmostEqual(actual.resolved_rrc_n_per_kn, resolved["rrc_N_per_kN"], places=9)
    test.assertAlmostEqual(actual.resolved_tire_a_n, resolved["tire_A_final"], places=9)
    test.assertAlmostEqual(actual.resolved_tire_b_n_per_kph, resolved["tire_B_final"], places=9)
    test.assertAlmostEqual(actual.resolved_tire_c_n_per_kph2, resolved["tire_C_final"], places=9)
    test.assertEqual(actual.tire_load_mass_basis, resolved["tire_load_mass_basis"])
    test.assertAlmostEqual(actual.tire_load_mass_used_kg, resolved["tire_load_mass_used_kg"])
    boundaries = resolve_roadload_boundaries(expected_row)
    test.assertAlmostEqual(actual.abc_total.A_N, boundaries["total"].A, places=9)
    test.assertAlmostEqual(actual.abc_net.A_N, boundaries["net"].A, places=9)


class CurrentTireTests(unittest.TestCase):
    def test_current_none_is_neutral(self):
        result = _resolve(TireQuickChange(TireSource.CURRENT))
        self.assertTrue(result.is_ready)
        self.assertEqual(result.resolved_rrc_n_per_kn, 8.0)
        self.assertEqual(result.abc_total.A_N, 118.0)

    def test_current_target_rrc_parity_with_existing_tire_abc(self):
        row = _row()
        inputs = {"target_rrc_N_per_kN": 7.5}
        expected = resolve_tire_proposal(row, "TIRE_TARGET_RRC", inputs, current_snapshot=row)
        expected_row = _apply_canonical_tire(row, expected)
        actual = _resolve(
            TireQuickChange(TireSource.CURRENT, TireTransformMode.TARGET_RRC, target_rrc_n_per_kn=7.5),
            row=row,
        )
        _assert_parity(self, actual, expected, expected_row)

    def test_current_target_rrc_parity_through_rrc_to_abc_path(self):
        row = _row(tire_A_final=None, tire_B_final=None, tire_C_final=None)
        expected = resolve_tire_proposal(
            row, "TIRE_TARGET_RRC", {"target_rrc_N_per_kN": 7.5}, current_snapshot=row
        )
        actual = _resolve(
            TireQuickChange(TireSource.CURRENT, TireTransformMode.TARGET_RRC, target_rrc_n_per_kn=7.5),
            row=row,
        )
        _assert_parity(self, actual, expected, _apply_canonical_tire(row, expected))

    def test_current_rrc_delta_delegates_result_to_target_resolver(self):
        actual = _resolve(
            TireQuickChange(TireSource.CURRENT, TireTransformMode.RRC_DELTA, rrc_delta_n_per_kn=-0.5)
        )
        expected = resolve_tire_proposal(
            _row(), "TIRE_TARGET_RRC", {"target_rrc_N_per_kN": 7.5}, current_snapshot=_row()
        )
        self.assertAlmostEqual(actual.resolved_rrc_n_per_kn, expected["resolved_rrc_N_per_kN"])
        self.assertAlmostEqual(actual.resolved_tire_a_n, expected["resolved_snapshot"]["tire_A_final"])

    def test_current_improvement_parity(self):
        row = _row()
        expected = resolve_tire_proposal(
            row, "TIRE_IMPROVEMENT_PCT", {"tire_improvement_pct": 5.0}, current_snapshot=row
        )
        actual = _resolve(
            TireQuickChange(TireSource.CURRENT, TireTransformMode.IMPROVEMENT_PCT, improvement_pct=5.0),
            row=row,
        )
        _assert_parity(self, actual, expected, _apply_canonical_tire(row, expected))

    def test_current_pressure_delta_parity(self):
        row = _row()
        inputs = {"front_pressure_psi": 40.0, "rear_pressure_psi": 40.0}
        expected = resolve_tire_proposal(row, "TIRE_TARGET_RRC", inputs, current_snapshot={**row, **inputs})
        actual = _resolve(
            TireQuickChange(
                TireSource.CURRENT,
                TireTransformMode.PRESSURE_DELTA,
                pressure_delta=TirePressureDelta(2.0),
            ),
            row=row,
        )
        _assert_parity(self, actual, expected, _apply_canonical_tire(row, expected))
        self.assertEqual(actual.reference_pressure_provenance, "SOURCE")

    def test_current_split_pressure_delta_is_not_averaged(self):
        result = _resolve(
            TireQuickChange(
                TireSource.CURRENT,
                TireTransformMode.PRESSURE_DELTA,
                pressure_delta=TirePressureDelta(2.0, rear_delta_psi=-1.0),
            )
        )
        self.assertEqual(result.resolved_front_pressure_psi, 40.0)
        self.assertEqual(result.resolved_rear_pressure_psi, 37.0)

    def test_current_sae_pressure_uses_full_canonical_model(self):
        tire = _sae_tire()
        result = _resolve(
            TireQuickChange(
                TireSource.CURRENT,
                TireTransformMode.PRESSURE_DELTA,
                pressure_delta=TirePressureDelta(2.0),
            ),
            row=_row(tire_db_id=100, tire_snapshot=tire),
        )
        self.assertTrue(result.is_ready)
        self.assertEqual(result.tire_abc_method, "SAE_FULL")

    def test_target_rrc_equal_to_current_is_neutral(self):
        result = _resolve(
            TireQuickChange(TireSource.CURRENT, TireTransformMode.TARGET_RRC, target_rrc_n_per_kn=8.0)
        )
        self.assertAlmostEqual(result.abc_total.A_N, 118.0)

    def test_improvement_zero_is_neutral(self):
        result = _resolve(
            TireQuickChange(TireSource.CURRENT, TireTransformMode.IMPROVEMENT_PCT, improvement_pct=0.0)
        )
        self.assertAlmostEqual(result.resolved_rrc_n_per_kn, 8.0)
        self.assertAlmostEqual(result.abc_total.A_N, 118.0)

    def test_pressure_delta_zero_is_neutral(self):
        result = _resolve(
            TireQuickChange(
                TireSource.CURRENT,
                TireTransformMode.PRESSURE_DELTA,
                pressure_delta=TirePressureDelta(0.0),
            )
        )
        self.assertAlmostEqual(result.resolved_rrc_n_per_kn, 8.0)
        self.assertAlmostEqual(result.abc_total.A_N, 118.0)


class TireDatabaseTests(unittest.TestCase):
    def test_tire_db_none_parity(self):
        row, tire = _row(), _iso_tire()
        inputs = {"tire_db_id": tire["id"], "tire_snapshot": tire}
        expected = resolve_tire_proposal(row, "TIRE_DB_LOOKUP", inputs, current_snapshot=row)
        actual = _resolve(TireQuickChange(TireSource.TIRE_DB, tire_db_id=99), row=row, tire_record=tire)
        _assert_parity(self, actual, expected, _apply_canonical_tire(row, expected))
        self.assertEqual(actual.resolved_tire_db_id, 99)

    def test_tire_db_improvement_uses_db_result_as_canonical_source(self):
        row, tire = _row(), _iso_tire()
        first = resolve_tire_proposal(
            row, "TIRE_DB_LOOKUP", {"tire_db_id": 99, "tire_snapshot": tire}, current_snapshot=row
        )
        selected = _apply_canonical_tire(row, first)
        second = resolve_tire_proposal(
            selected, "TIRE_IMPROVEMENT_PCT", {"tire_improvement_pct": 7.0}, current_snapshot=selected
        )
        actual = _resolve(
            TireQuickChange(
                TireSource.TIRE_DB,
                TireTransformMode.IMPROVEMENT_PCT,
                tire_db_id=99,
                improvement_pct=7.0,
            ),
            row=row,
            tire_record=tire,
        )
        _assert_parity(self, actual, second, _apply_canonical_tire(selected, second))
        self.assertAlmostEqual(actual.reference_rrc_n_per_kn, first["resolved_rrc_N_per_kN"])

    def test_tire_db_iso_pressure_delta_parity(self):
        row, tire = _row(), _iso_tire()
        inputs = {
            "tire_db_id": 99,
            "tire_snapshot": tire,
            "front_pressure_psi": 38.0,
            "rear_pressure_psi": 38.0,
        }
        expected = resolve_tire_proposal(row, "TIRE_DB_LOOKUP", inputs, current_snapshot={**row, **inputs})
        actual = _resolve(
            TireQuickChange(
                TireSource.TIRE_DB,
                TireTransformMode.PRESSURE_DELTA,
                tire_db_id=99,
                pressure_delta=TirePressureDelta(2.0),
            ),
            row=row,
            tire_record=tire,
        )
        _assert_parity(self, actual, expected, _apply_canonical_tire(row, expected))

    def test_tire_db_split_pressure_preserved(self):
        result = _resolve(
            TireQuickChange(
                TireSource.TIRE_DB,
                TireTransformMode.PRESSURE_DELTA,
                tire_db_id=99,
                pressure_delta=TirePressureDelta(2.0, rear_delta_psi=-2.0),
            ),
            tire_record=_iso_tire(),
        )
        self.assertEqual((result.resolved_front_pressure_psi, result.resolved_rear_pressure_psi), (38.0, 34.0))

    def test_tire_db_sae_pressure_uses_full_canonical_model(self):
        result = _resolve(
            TireQuickChange(
                TireSource.TIRE_DB,
                TireTransformMode.PRESSURE_DELTA,
                tire_db_id=100,
                pressure_delta=TirePressureDelta(2.0),
            ),
            tire_record=_sae_tire(),
        )
        self.assertTrue(result.is_ready)
        self.assertEqual(result.tire_abc_method, "SAE_FULL")

    def test_tire_db_user_reference_is_not_replaced_by_db_reference(self):
        result = _resolve(
            TireQuickChange(
                TireSource.TIRE_DB,
                TireTransformMode.PRESSURE_DELTA,
                tire_db_id=99,
                pressure_delta=TirePressureDelta(
                    2.0,
                    reference_pressure_psi=32.0,
                    reference_pressure_provenance=ReferencePressureProvenance.USER_PROVIDED,
                ),
            ),
            tire_record=_iso_tire(test_pressure_value=36.0),
        )
        self.assertTrue(result.is_ready)
        self.assertEqual(result.reference_pressure_provenance, "USER_PROVIDED")
        self.assertEqual(result.reference_front_pressure_psi, 32.0)
        self.assertEqual(result.resolved_front_pressure_psi, 34.0)

    def test_missing_tire_db_id_is_explicit_missing(self):
        with mock.patch("src.vde_core.quick_scenario.resolver.get_tire_by_id", return_value={}):
            result = resolve_quick_vehicle_scenario(
                _scenario(TireQuickChange(TireSource.TIRE_DB, tire_db_id=404)), source_vde_row=_row()
            )
        self.assertEqual(result.readiness.tire, DomainReadiness.MISSING)
        self.assertFalse(result.is_ready)

    def test_tire_db_row_is_not_mutated(self):
        tire = _iso_tire()
        original = copy.deepcopy(tire)
        _resolve(TireQuickChange(TireSource.TIRE_DB, tire_db_id=99), tire_record=tire)
        self.assertEqual(tire, original)

    def test_real_seeded_tire_repository_resolution_is_immutable(self):
        temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(temp_dir.name) / "quick_tire.db"
        try:
            seed_qa_database(db_path, overwrite=False)
            with db_module.using_db_path(db_path):
                tire = get_tire_by_code("TIRE-QA-003")
                before = copy.deepcopy(get_tire_by_id(tire["id"]))
                result = resolve_quick_vehicle_scenario(
                    _scenario(TireQuickChange(TireSource.TIRE_DB, tire_db_id=tire["id"])),
                    source_vde_row=_row(),
                )
                after = get_tire_by_id(tire["id"])
            self.assertTrue(result.is_ready)
            self.assertEqual(result.resolved_tire_db_id, tire["id"])
            self.assertEqual(after, before)
        finally:
            gc.collect()
            temp_dir.cleanup()


class MissingAndProvenanceTests(unittest.TestCase):
    def test_missing_reference_pressure_is_explicit_missing(self):
        row = _row(front_pressure_psi=None, rear_pressure_psi=None)
        result = _resolve(
            TireQuickChange(
                TireSource.CURRENT,
                TireTransformMode.PRESSURE_DELTA,
                pressure_delta=TirePressureDelta(2.0),
            ),
            row=row,
        )
        self.assertEqual(result.readiness.tire, DomainReadiness.MISSING)
        self.assertIsNone(result.vehicle_demand_result)

    def test_user_reference_pressure_resolves_and_preserves_provenance(self):
        row = _row(front_pressure_psi=None, rear_pressure_psi=None)
        result = _resolve(
            TireQuickChange(
                TireSource.CURRENT,
                TireTransformMode.PRESSURE_DELTA,
                pressure_delta=TirePressureDelta(
                    2.0,
                    reference_pressure_psi=32.0,
                    reference_pressure_provenance=ReferencePressureProvenance.USER_PROVIDED,
                ),
            ),
            row=row,
        )
        self.assertTrue(result.is_ready)
        self.assertEqual(result.reference_pressure_provenance, "USER_PROVIDED")
        self.assertEqual((result.resolved_front_pressure_psi, result.resolved_rear_pressure_psi), (34.0, 34.0))

    def test_missing_reference_rrc_for_delta_is_explicit_missing(self):
        result = _resolve(
            TireQuickChange(TireSource.CURRENT, TireTransformMode.RRC_DELTA, rrc_delta_n_per_kn=-0.5),
            row=_row(rrc_N_per_kN=None, tire_A_final=None, tire_B_final=None, tire_C_final=None),
        )
        self.assertEqual(result.readiness.tire, DomainReadiness.MISSING)
        self.assertTrue(any("RRC" in issue for issue in result.issues))

    def test_mass_ready_tire_missing_aero_ready_blocks_whole_scenario(self):
        row = _row(front_pressure_psi=None, rear_pressure_psi=None)
        result = _resolve(
            TireQuickChange(
                TireSource.CURRENT,
                TireTransformMode.PRESSURE_DELTA,
                pressure_delta=TirePressureDelta(1.0),
            ),
            row=row,
            mass_change=MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.DELTA, -20.0)),
            cda_change=ScalarChange(ScalarChangeMode.DELTA, -0.01),
        )
        self.assertEqual(result.readiness.mass, DomainReadiness.READY)
        self.assertEqual(result.readiness.tire, DomainReadiness.MISSING)
        self.assertEqual(result.readiness.aero, DomainReadiness.READY)
        self.assertIsNone(result.abc_total)


class CompositionAndIntegrationTests(unittest.TestCase):
    def test_mass_then_target_rrc_uses_new_canonical_tire_load_mass(self):
        row = _row()
        mass_change = MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.ABSOLUTE, 1700.0))
        mass = resolve_mass_proposal(row, "EPA_CURB_TO_TWC", {"mass_kg": 1700.0})["resolved_snapshot"]
        current = {
            **row,
            "mass_kg": mass["curb_mass_kg"],
            "test_mass_kg": mass["vde_calculation_mass_kg"],
            "inertia_class": mass["inertia_class"],
            "tire_load_mass_basis": mass["tire_load_mass_basis"],
            "tire_load_mass_used_kg": mass["tire_load_mass_used_kg"],
        }
        expected = resolve_tire_proposal(
            row, "TIRE_TARGET_RRC", {"target_rrc_N_per_kN": 7.5}, current_snapshot=current
        )
        result = _resolve(
            TireQuickChange(TireSource.CURRENT, TireTransformMode.TARGET_RRC, target_rrc_n_per_kn=7.5),
            row=row,
            mass_change=mass_change,
        )
        self.assertEqual(result.tire_load_mass_used_kg, mass["tire_load_mass_used_kg"])
        self.assertAlmostEqual(result.resolved_tire_a_n, expected["resolved_snapshot"]["tire_A_final"])

    def test_same_tire_change_with_different_resolved_mass_changes_tire_state(self):
        change = TireQuickChange(TireSource.CURRENT, TireTransformMode.TARGET_RRC, target_rrc_n_per_kn=7.5)
        light = _resolve(
            change,
            mass_change=MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.ABSOLUTE, 1400.0)),
        )
        heavy = _resolve(
            change,
            mass_change=MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.ABSOLUTE, 1800.0)),
        )
        self.assertNotEqual(light.tire_load_mass_used_kg, heavy.tire_load_mass_used_kg)
        self.assertNotEqual(light.resolved_tire_a_n, heavy.resolved_tire_a_n)

    def test_mass_tire_aero_combined_is_ready(self):
        result = _resolve(
            TireQuickChange(TireSource.CURRENT, TireTransformMode.IMPROVEMENT_PCT, improvement_pct=5.0),
            mass_change=MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.DELTA, -20.0)),
            cda_change=ScalarChange(ScalarChangeMode.DELTA, -0.02),
        )
        self.assertTrue(result.is_ready)
        self.assertEqual(result.readiness.mass, DomainReadiness.READY)
        self.assertEqual(result.readiness.tire, DomainReadiness.READY)
        self.assertEqual(result.readiness.aero, DomainReadiness.READY)

    def test_source_row_remains_unchanged(self):
        row = _row()
        original = copy.deepcopy(row)
        _resolve(
            TireQuickChange(TireSource.CURRENT, TireTransformMode.IMPROVEMENT_PCT, improvement_pct=5.0),
            row=row,
        )
        self.assertEqual(row, original)

    def test_repeated_resolution_is_deterministic(self):
        change = TireQuickChange(TireSource.CURRENT, TireTransformMode.PRESSURE_DELTA, pressure_delta=TirePressureDelta(2.0))
        self.assertEqual(_resolve(change), _resolve(change))

    def test_distinct_fuelcons_sources_keep_distinct_quick_identity(self):
        change = TireQuickChange(TireSource.CURRENT)
        first = resolve_quick_vehicle_scenario(_scenario(change, source_identity="fc:1"), source_vde_row=_row())
        second = resolve_quick_vehicle_scenario(_scenario(change, source_identity="fc:2"), source_vde_row=_row())
        self.assertNotEqual(first.quick_scenario_identity, second.quick_scenario_identity)

    def test_total_and_net_remain_explicit(self):
        row = _row(trans_A_coef_N=None, trans_B_coef_Npkph=None, trans_C_coef_Npkph2=None)
        result = _resolve(
            TireQuickChange(TireSource.CURRENT, TireTransformMode.TARGET_RRC, target_rrc_n_per_kn=7.5),
            row=row,
        )
        self.assertIsNotNone(result.abc_total)
        self.assertIsNone(result.abc_net)

    def test_vehicle_demand_request_uses_final_tire_resolved_total(self):
        result = _resolve(
            TireQuickChange(TireSource.CURRENT, TireTransformMode.TARGET_RRC, target_rrc_n_per_kn=7.5)
        )
        self.assertAlmostEqual(result.vehicle_demand_request.roadload_total.A_N, result.abc_total.A_N)
        self.assertNotEqual(result.vehicle_demand_request.roadload_total.A_N, _row()["coast_A_N"])

    def test_vehicle_demand_adapter_remains_the_canonical_builder(self):
        row = _row()
        expected_tire = resolve_tire_proposal(
            row, "TIRE_TARGET_RRC", {"target_rrc_N_per_kN": 7.5}, current_snapshot=row
        )
        expected_request = build_vehicle_demand_request(_apply_canonical_tire(row, expected_tire))
        result = _resolve(
            TireQuickChange(TireSource.CURRENT, TireTransformMode.TARGET_RRC, target_rrc_n_per_kn=7.5),
            row=row,
        )
        self.assertEqual(result.vehicle_demand_request, expected_request)

    def test_quick_resolver_spies_canonical_tire_delegation(self):
        change = TireQuickChange(TireSource.CURRENT, TireTransformMode.IMPROVEMENT_PCT, improvement_pct=5.0)
        with mock.patch(
            "src.vde_core.quick_scenario.resolver.resolve_tire_proposal",
            wraps=resolve_tire_proposal,
        ) as spy:
            result = _resolve(change)
        self.assertTrue(result.is_ready)
        spy.assert_called_once()
        self.assertEqual(spy.call_args.args[1], "TIRE_IMPROVEMENT_PCT")


if __name__ == "__main__":
    unittest.main()
