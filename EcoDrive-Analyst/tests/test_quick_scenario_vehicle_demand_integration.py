"""Cross-path parity: Quick Mass/Aero resolution vs. the canonical VDE
Setup mass/aero resolvers and the frozen Vehicle Demand Core, on real QA
fixture data.

Every comparison here is "same canonical function, same effective input" --
never a cross-engine comparison. The legacy ABC-polynomial path
(resolve_cycle_vde_results, EPA-phase-weighted) and the frozen Vehicle
Demand Core (calculate_vehicle_demand, whole-trace) are known, by Sprint 9's
own reconciliation suite (tests/test_vehicle_demand_integration.py), to
disagree for EPA rows unless deliberately re-combined by EPA policy phase
weights (_epa_combined_vde there) -- that reconciliation is Sprint 9's
concern, not 10B's. 10B only needs to prove the Quick resolver calls
through correctly to each canonical function and reproduces exactly what
calling that same function independently would produce.
"""

import gc
import tempfile
import unittest
from pathlib import Path

from src.vde_core import db as db_module
from src.vde_core.comparison_report_service import (
    resolve_cycle_vde_results,
    resolve_roadload_boundaries,
)
from src.vde_core.qa_mock_data import build_vde_seed_rows, seed_qa_database, seed_qa_fuelcons_mock_rows
from src.vde_core.quick_scenario import (
    MassQuickChange,
    QuickScenario,
    ScalarChange,
    ScalarChangeMode,
    VehicleQuickOverrides,
    resolve_quick_vehicle_scenario,
)
from src.vde_core.roadload import cdA_to_C
from src.vde_core.vde_mass_proposal_resolver import resolve_mass_proposal
from src.vde_core.vehicle_demand import calculate_vehicle_demand
from src.vde_core.vehicle_demand.adapters import (
    build_vehicle_demand_request,
    resolve_vehicle_demand_cycle,
)


def _qa_rows() -> dict:
    return {row["source_record_id"]: row for row in build_vde_seed_rows()}


def _wltp_row() -> dict:
    return {
        "id": 990001,
        "record_origin": "IMPORTED_REFERENCE",
        "legislation": "WLTP",
        "category": "QA_WLTP",
        "wltp_category": "M1",
        "make": "QA",
        "model": "WLTP-SYNTH",
        "year": 2026,
        "mass_kg": 1600.0,
        "test_mass_kg": 1780.0,
        "payload_kg": 180.0,
        "options_kg": 0.0,
        "coast_A_N": 115.0,
        "coast_B_N_per_kph": 0.019,
        "coast_C_N_per_kph2": 0.0088,
        "trans_A_coef_N": 8.2,
        "trans_B_coef_Npkph": 0.0039,
        "trans_C_coef_Npkph2": 0.0008,
        "rrc_N_per_kN": 8.0,
        "cda_m2": 0.60,
    }


def _no_override_scenario(source_identity="vde:1") -> QuickScenario:
    return QuickScenario(source_identity=source_identity, slot=1, vehicle_overrides=VehicleQuickOverrides())


class NoChangeParityTests(unittest.TestCase):
    """A Quick Scenario with no effective Vehicle override must reproduce
    the source scenario's own physical state exactly, both through the
    legacy ABC-polynomial path and the frozen Vehicle Demand Core.
    """

    def _assert_reproduces_source(self, row: dict) -> None:
        result = resolve_quick_vehicle_scenario(_no_override_scenario(), source_vde_row=row)
        self.assertTrue(result.is_ready)

        expected_boundaries = resolve_roadload_boundaries(row)
        self.assertAlmostEqual(result.abc_total.A_N, expected_boundaries["total"].A, places=9)
        self.assertAlmostEqual(result.abc_total.B_N_per_kph, expected_boundaries["total"].B, places=9)
        self.assertAlmostEqual(result.abc_total.C_N_per_kph2, expected_boundaries["total"].C, places=9)
        if expected_boundaries["net"].available:
            self.assertAlmostEqual(result.abc_net.A_N, expected_boundaries["net"].A, places=9)
            self.assertAlmostEqual(result.abc_net.B_N_per_kph, expected_boundaries["net"].B, places=9)
            self.assertAlmostEqual(result.abc_net.C_N_per_kph2, expected_boundaries["net"].C, places=9)

        expected_cycle_results = resolve_cycle_vde_results(row)
        self.assertAlmostEqual(
            result.vde_total_mj_per_km, expected_cycle_results["total"].aggregate, places=9
        )
        self.assertAlmostEqual(
            result.vde_net_mj_per_km, expected_cycle_results["net"].aggregate, places=9
        )

        expected_request = build_vehicle_demand_request(row)
        expected_cycle = resolve_vehicle_demand_cycle(row)
        expected_result = calculate_vehicle_demand(expected_request, expected_cycle)
        self.assertEqual(result.vehicle_demand_request, expected_request)
        self.assertAlmostEqual(
            result.vehicle_demand_result.total_summary.vde_mj_per_km,
            expected_result.total_summary.vde_mj_per_km,
            places=9,
        )
        self.assertAlmostEqual(
            result.vehicle_demand_result.net_summary.vde_mj_per_km,
            expected_result.net_summary.vde_mj_per_km,
            places=9,
        )

    def test_epa_qa_001_no_change_reproduces_source(self):
        self._assert_reproduces_source(_qa_rows()["VDE-QA-001"])

    def test_wltp_no_change_reproduces_source(self):
        self._assert_reproduces_source(_wltp_row())


class CurbMassChangeParityTests(unittest.TestCase):
    """The expected synthetic row is built independently, via the exact
    same resolve_mass_proposal() call the Quick resolver itself makes --
    never a hand-guessed expected mass.
    """

    def test_epa_curb_change_parity(self):
        row = _qa_rows()["VDE-QA-001"]
        overrides = VehicleQuickOverrides(
            mass_change=MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.DELTA, -20.0))
        )
        scenario = QuickScenario(source_identity="vde:1", slot=1, vehicle_overrides=overrides)
        result = resolve_quick_vehicle_scenario(scenario, source_vde_row=row)
        self.assertTrue(result.is_ready)

        outcome = resolve_mass_proposal(dict(row), "EPA_CURB_TO_TWC", {"mass_kg": row["mass_kg"] - 20.0})
        expected_row = dict(row)
        expected_row["mass_kg"] = outcome["resolved_snapshot"]["curb_mass_kg"]
        expected_row["test_mass_kg"] = outcome["resolved_snapshot"]["vde_calculation_mass_kg"]

        self.assertEqual(result.resolved_curb_mass_kg, expected_row["mass_kg"])
        self.assertEqual(result.resolved_vde_calculation_mass_kg, expected_row["test_mass_kg"])

        expected_boundaries = resolve_roadload_boundaries(expected_row)
        self.assertAlmostEqual(result.abc_total.A_N, expected_boundaries["total"].A, places=9)

        expected_cycle_results = resolve_cycle_vde_results(expected_row)
        self.assertAlmostEqual(
            result.vde_total_mj_per_km, expected_cycle_results["total"].aggregate, places=9
        )
        self.assertAlmostEqual(
            result.vde_net_mj_per_km, expected_cycle_results["net"].aggregate, places=9
        )

        expected_request = build_vehicle_demand_request(expected_row)
        expected_cycle = resolve_vehicle_demand_cycle(expected_row)
        expected_result = calculate_vehicle_demand(expected_request, expected_cycle)
        self.assertAlmostEqual(
            result.vehicle_demand_result.total_summary.vde_mj_per_km,
            expected_result.total_summary.vde_mj_per_km,
            places=9,
        )


class CdaChangeParityTests(unittest.TestCase):
    """The expected synthetic row is built independently via cdA_to_C()
    directly -- never a hand-written second CdA->C formula.
    """

    def test_epa_cda_change_parity(self):
        row = _qa_rows()["VDE-QA-001"]
        overrides = VehicleQuickOverrides(cda_change=ScalarChange(ScalarChangeMode.DELTA, -0.05))
        scenario = QuickScenario(source_identity="vde:1", slot=1, vehicle_overrides=overrides)
        result = resolve_quick_vehicle_scenario(scenario, source_vde_row=row)
        self.assertTrue(result.is_ready)

        expected_row = dict(row)
        expected_row["cda_m2"] = row["cda_m2"] - 0.05
        expected_row["coast_C_N_per_kph2"] = row["coast_C_N_per_kph2"] + cdA_to_C(-0.05)

        self.assertAlmostEqual(result.resolved_cda_m2, expected_row["cda_m2"], places=9)

        expected_boundaries = resolve_roadload_boundaries(expected_row)
        self.assertAlmostEqual(result.abc_total.C_N_per_kph2, expected_boundaries["total"].C, places=9)
        self.assertAlmostEqual(result.abc_net.C_N_per_kph2, expected_boundaries["net"].C, places=9)

        expected_cycle_results = resolve_cycle_vde_results(expected_row)
        self.assertAlmostEqual(
            result.vde_total_mj_per_km, expected_cycle_results["total"].aggregate, places=9
        )


class SharedVdeDistinctIdentityParityTests(unittest.TestCase):
    """Two real fuelcons_db rows (900102, 900104) sharing one vde_id
    (900001) (Sec 3/4): same resolved physics, distinct Quick Scenario
    identity. Uses a real temporary seeded QA database so the `fc:` source
    identity exercises the actual DB lookup path
    (_fetch_source_vde_row -> get_record(FUEL_CONSUMPTION) ->
    fetch_vde_by_id), not just an injected `source_vde_row`.
    """

    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "quick_scenario_qa.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        seed_qa_fuelcons_mock_rows(self.db_path)
        db_module.configure_db_path(self.db_path)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def test_two_fuelcons_scenarios_sharing_one_vde_resolve_identically_but_stay_distinct(self):
        first = QuickScenario(source_identity="fc:900102", slot=1)
        second = QuickScenario(source_identity="fc:900104", slot=1)
        self.assertNotEqual(first.identity, second.identity)

        result_1 = resolve_quick_vehicle_scenario(first)
        result_2 = resolve_quick_vehicle_scenario(second)

        self.assertTrue(result_1.is_ready)
        self.assertTrue(result_2.is_ready)
        self.assertNotEqual(result_1.quick_scenario_identity, result_2.quick_scenario_identity)
        self.assertEqual(result_1.resolved_curb_mass_kg, result_2.resolved_curb_mass_kg)
        self.assertEqual(result_1.vde_total_mj_per_km, result_2.vde_total_mj_per_km)


if __name__ == "__main__":
    unittest.main()
