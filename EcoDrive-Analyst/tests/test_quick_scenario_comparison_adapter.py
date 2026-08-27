"""Sprint 10E: tests for src.vde_core.quick_scenario.comparison_adapter --
turning a resolved QuickScenario into an ordinary ComparisonItem via the
EXISTING build_vde_comparison_item/build_scenario_comparison_item builders
(never a hand-built ComparisonItem, never a second Comparison engine).
"""

import gc
import tempfile
import unittest
from pathlib import Path

from src.vde_core import db as db_module
from src.vde_core.comparison_report_service import (
    ComparisonDataset,
    ComparisonRole,
    SourceKind,
    build_vde_comparison_item,
)
from src.vde_core.qa_mock_data import seed_qa_database, seed_qa_fuelcons_mock_rows
from src.vde_core.quick_scenario import (
    PseProvenance,
    QuickScenario,
    QuickSlotCalculationState,
    ScalarChange,
    ScalarChangeMode,
    TireQuickChange,
    TireSource,
    TireTransformMode,
    VehicleQuickOverrides,
    build_quick_comparison_item,
    derive_quick_slot_calculation_state,
    fetch_quick_source_rows_once,
    merge_quick_items_into_dataset,
    quick_slot_sentinel_id,
    resolve_quick_slot,
)


class QuickSlotSentinelIdTests(unittest.TestCase):
    def test_negative_and_deterministic(self):
        first = quick_slot_sentinel_id(900001, 1)
        second = quick_slot_sentinel_id(900001, 1)
        self.assertEqual(first, second)
        self.assertLess(first, 0)

    def test_distinct_across_slots_and_sources(self):
        ids = {
            quick_slot_sentinel_id(900001, 1),
            quick_slot_sentinel_id(900001, 2),
            quick_slot_sentinel_id(900001, 3),
            quick_slot_sentinel_id(900002, 1),
        }
        self.assertEqual(len(ids), 4)


class ComparisonAdapterQaDatabaseTests(unittest.TestCase):
    """Seeds the same QA fixture (vde_id=900001, fc:900102/fc:900104) every
    other Quick Scenario test file this session already established.
    """

    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "comparison_adapter_qa.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        seed_qa_fuelcons_mock_rows(self.db_path)
        db_module.configure_db_path(self.db_path)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def _resolve(self, quick_scenario):
        vde_row, fuelcons_row = fetch_quick_source_rows_once(quick_scenario.source_identity)
        return resolve_quick_slot(
            quick_scenario, source_vde_row=vde_row, source_fuelcons_row=fuelcons_row
        )

    def test_vehicle_unresolved_scenario_produces_no_comparison_item(self):
        scenario = QuickScenario(
            source_identity="fc:900102",
            slot=1,
            vehicle_overrides=VehicleQuickOverrides(
                tire_change=TireQuickChange(
                    source=TireSource.TIRE_DB,
                    transform_mode=TireTransformMode.NONE,
                    tire_db_id=999999999,  # no such Tire DB row -> MISSING
                )
            ),
        )
        vehicle_resolution, efficiency_resolution = self._resolve(scenario)
        self.assertFalse(vehicle_resolution.is_ready)
        item = build_quick_comparison_item(scenario, vehicle_resolution, efficiency_resolution)
        self.assertIsNone(item)

    def test_vehicle_ready_efficiency_not_requested_is_vde_only_shaped(self):
        scenario = QuickScenario(
            source_identity="fc:900102",
            slot=1,
            vehicle_overrides=VehicleQuickOverrides(
                cda_change=ScalarChange(mode=ScalarChangeMode.PERCENT, value=0.0)
            ),
        )
        vehicle_resolution, efficiency_resolution = self._resolve(scenario)
        self.assertTrue(vehicle_resolution.is_ready)
        item = build_quick_comparison_item(scenario, vehicle_resolution, efficiency_resolution)
        self.assertIsNotNone(item)
        self.assertEqual(item.source_kind, SourceKind.VDE_ONLY)
        self.assertIsNone(item.fuel_energy)
        self.assertEqual(item.provenance.record_origin, "QUICK_SCENARIO")
        self.assertLess(item.vde_id, 0)

    def test_vehicle_and_efficiency_ready_is_full_scenario_shaped(self):
        scenario = QuickScenario(
            source_identity="fc:900102",
            slot=1,
            vehicle_overrides=VehicleQuickOverrides(
                cda_change=ScalarChange(mode=ScalarChangeMode.PERCENT, value=0.0)
            ),
            final_pse_percent=30.0,
            pse_provenance=PseProvenance.USER_PROVIDED,
        )
        vehicle_resolution, efficiency_resolution = self._resolve(scenario)
        self.assertTrue(vehicle_resolution.is_ready)
        self.assertTrue(efficiency_resolution.is_ready)
        item = build_quick_comparison_item(scenario, vehicle_resolution, efficiency_resolution)
        self.assertIsNotNone(item)
        self.assertEqual(item.source_kind, SourceKind.FUELCONS_SCENARIO)
        self.assertIsNotNone(item.fuel_energy)
        self.assertIsNotNone(item.fuel_energy.get("fuel_l_per_100km"))
        self.assertEqual(item.provenance.record_origin, "QUICK_SCENARIO")
        self.assertLess(item.fuelcons_id, 0)
        self.assertLess(item.vde_id, 0)

    def test_total_and_net_are_both_present_and_independent(self):
        scenario = QuickScenario(
            source_identity="fc:900102",
            slot=1,
            vehicle_overrides=VehicleQuickOverrides(),
        )
        vehicle_resolution, efficiency_resolution = self._resolve(scenario)
        item = build_quick_comparison_item(scenario, vehicle_resolution, efficiency_resolution)
        self.assertIsNotNone(item)
        self.assertTrue(item.roadload["total"].available)
        # NET is TOTAL minus transmission -- distinct fields, never a
        # TOTAL/NET fallback (comparison_report_service.resolve_roadload_boundaries).
        self.assertIn("total", item.vde["aggregate"])
        self.assertIn("net", item.vde["aggregate"])

    def test_partial_slot_failure_does_not_block_sibling_slots(self):
        good_scenario_1 = QuickScenario(
            source_identity="fc:900102",
            slot=1,
            vehicle_overrides=VehicleQuickOverrides(
                cda_change=ScalarChange(mode=ScalarChangeMode.DELTA, value=0.0)
            ),
        )
        bad_scenario_2 = QuickScenario(
            source_identity="fc:900102",
            slot=2,
            vehicle_overrides=VehicleQuickOverrides(
                tire_change=TireQuickChange(
                    source=TireSource.TIRE_DB,
                    transform_mode=TireTransformMode.NONE,
                    tire_db_id=999999999,  # no such Tire DB row
                )
            ),
        )
        good_scenario_3 = QuickScenario(
            source_identity="fc:900102",
            slot=3,
            vehicle_overrides=VehicleQuickOverrides(
                cda_change=ScalarChange(mode=ScalarChangeMode.PERCENT, value=0.0)
            ),
        )
        vde_row, fuelcons_row = fetch_quick_source_rows_once("fc:900102")
        items = []
        for scenario in (good_scenario_1, bad_scenario_2, good_scenario_3):
            vehicle_resolution, efficiency_resolution = resolve_quick_slot(
                scenario, source_vde_row=vde_row, source_fuelcons_row=fuelcons_row
            )
            items.append(
                build_quick_comparison_item(scenario, vehicle_resolution, efficiency_resolution)
            )
        self.assertIsNotNone(items[0])
        self.assertIsNone(items[1])
        self.assertIsNotNone(items[2])

    def test_merge_preserves_reference_and_existing_comparisons(self):
        vde_row, _ = fetch_quick_source_rows_once("vde:900001")
        reference_item = build_vde_comparison_item(900001, role=ComparisonRole.REFERENCE, vde_row=vde_row)
        dataset = ComparisonDataset(reference=reference_item, comparisons=())

        scenario = QuickScenario(
            source_identity="fc:900102",
            slot=1,
            vehicle_overrides=VehicleQuickOverrides(
                cda_change=ScalarChange(mode=ScalarChangeMode.PERCENT, value=0.0)
            ),
        )
        vehicle_resolution, efficiency_resolution = self._resolve(scenario)
        quick_item = build_quick_comparison_item(scenario, vehicle_resolution, efficiency_resolution)

        merged = merge_quick_items_into_dataset(dataset, [quick_item])
        self.assertIs(merged.reference, dataset.reference)
        self.assertEqual(merged.comparisons, (quick_item,))

    def test_merge_with_no_quick_items_returns_same_dataset(self):
        dataset = ComparisonDataset(reference=None, comparisons=())
        merged = merge_quick_items_into_dataset(dataset, [])
        self.assertIs(merged, dataset)


class DeriveQuickSlotCalculationStateTests(unittest.TestCase):
    def _scenario(self, **overrides):
        return QuickScenario(source_identity="fc:1", slot=1, **overrides)

    def test_not_calculated_before_first_run(self):
        state = derive_quick_slot_calculation_state(self._scenario(), None, None, None)
        self.assertEqual(state, QuickSlotCalculationState.NOT_CALCULATED)

    def test_needs_recalculation_when_inputs_diverge(self):
        original = self._scenario(
            vehicle_overrides=VehicleQuickOverrides(
                cda_change=ScalarChange(mode=ScalarChangeMode.PERCENT, value=1.0)
            )
        )
        edited = self._scenario(
            vehicle_overrides=VehicleQuickOverrides(
                cda_change=ScalarChange(mode=ScalarChangeMode.PERCENT, value=2.0)
            )
        )

        class _FakeReadyVehicle:
            is_ready = True

        state = derive_quick_slot_calculation_state(edited, original, _FakeReadyVehicle(), None)
        self.assertEqual(state, QuickSlotCalculationState.NEEDS_RECALCULATION)

    def test_missing_or_invalid_when_vehicle_not_ready(self):
        scenario = self._scenario()

        class _FakeNotReadyVehicle:
            is_ready = False

        state = derive_quick_slot_calculation_state(
            scenario, scenario, _FakeNotReadyVehicle(), None
        )
        self.assertEqual(state, QuickSlotCalculationState.MISSING_OR_INVALID)

    def test_ready_when_inputs_match_and_vehicle_ready_and_no_efficiency_requested(self):
        scenario = self._scenario()

        class _FakeReadyVehicle:
            is_ready = True

        state = derive_quick_slot_calculation_state(scenario, scenario, _FakeReadyVehicle(), None)
        self.assertEqual(state, QuickSlotCalculationState.READY)

    def test_missing_or_invalid_when_efficiency_requested_but_not_ready(self):
        scenario = self._scenario(
            final_pse_percent=25.0, pse_provenance=PseProvenance.USER_PROVIDED
        )

        class _FakeReadyVehicle:
            is_ready = True

        class _FakeNotReadyEfficiency:
            is_ready = False

        state = derive_quick_slot_calculation_state(
            scenario, scenario, _FakeReadyVehicle(), _FakeNotReadyEfficiency()
        )
        self.assertEqual(state, QuickSlotCalculationState.MISSING_OR_INVALID)


if __name__ == "__main__":
    unittest.main()
