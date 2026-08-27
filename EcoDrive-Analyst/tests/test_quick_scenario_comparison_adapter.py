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
        first = quick_slot_sentinel_id("vde:900001", 1)
        second = quick_slot_sentinel_id("vde:900001", 1)
        self.assertEqual(first, second)
        self.assertLess(first, 0)

    def test_distinct_across_slots_and_sources(self):
        ids = {
            quick_slot_sentinel_id("vde:900001", 1),
            quick_slot_sentinel_id("vde:900001", 2),
            quick_slot_sentinel_id("vde:900001", 3),
            quick_slot_sentinel_id("vde:900002", 1),
        }
        self.assertEqual(len(ids), 4)

    def test_distinct_across_fuelcons_sources_sharing_one_vde(self):
        # Sprint 10E closure audit finding: fc:900102 and fc:900104 both link
        # to vde_id=900001 (the QA fixture used since Sprint 10A) -- their
        # own Quick slot 1 must NOT collide into the same sentinel id, or
        # two genuinely distinct Quick Scenarios would be conflated into one
        # ComparisonItem identity.
        self.assertNotEqual(
            quick_slot_sentinel_id("fc:900102", 1), quick_slot_sentinel_id("fc:900104", 1)
        )

    def test_distinct_between_fc_and_vde_kind_for_the_same_numeric_id(self):
        self.assertNotEqual(
            quick_slot_sentinel_id("fc:900001", 1), quick_slot_sentinel_id("vde:900001", 1)
        )

    def test_user_label_does_not_affect_identity(self):
        # Closure audit family B: "user label does not define identity" --
        # the sentinel (and, by construction, QuickScenario.identity too)
        # depends only on source_identity and slot.
        labeled = QuickScenario(source_identity="fc:1", slot=1, label="What if lighter?")
        unlabeled = QuickScenario(source_identity="fc:1", slot=1, label=None)
        differently_labeled = QuickScenario(source_identity="fc:1", slot=1, label="Something else")
        self.assertEqual(labeled.identity, unlabeled.identity)
        self.assertEqual(labeled.identity, differently_labeled.identity)
        self.assertEqual(
            quick_slot_sentinel_id(labeled.source_identity, labeled.slot),
            quick_slot_sentinel_id(unlabeled.source_identity, unlabeled.slot),
        )


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

    def test_two_fuelcons_sources_sharing_one_vde_produce_distinct_quick_items(self):
        # Same requirement family as the existing "two FuelCons scenarios
        # sharing one VDE remain distinct" guarantee for real Comparison
        # items, applied to their Quick derivatives (closure audit finding).
        scenario_a = QuickScenario(
            source_identity="fc:900102",
            slot=1,
            vehicle_overrides=VehicleQuickOverrides(
                cda_change=ScalarChange(mode=ScalarChangeMode.PERCENT, value=0.0)
            ),
        )
        scenario_b = QuickScenario(
            source_identity="fc:900104",
            slot=1,
            vehicle_overrides=VehicleQuickOverrides(
                cda_change=ScalarChange(mode=ScalarChangeMode.PERCENT, value=0.0)
            ),
        )
        vehicle_a, efficiency_a = self._resolve(scenario_a)
        vehicle_b, efficiency_b = self._resolve(scenario_b)
        item_a = build_quick_comparison_item(scenario_a, vehicle_a, efficiency_a)
        item_b = build_quick_comparison_item(scenario_b, vehicle_b, efficiency_b)
        self.assertIsNotNone(item_a)
        self.assertIsNotNone(item_b)
        self.assertNotEqual(item_a.vde_id, item_b.vde_id)

    def test_recalculation_with_changed_inputs_keeps_the_same_identity(self):
        # Closure audit family C/G: recalculating a slot after editing its
        # inputs must REPLACE the same temporary Quick identity, never mint
        # a new one -- otherwise the Comparison Set would accumulate one
        # entry per calculation instead of one per slot.
        first_pass = QuickScenario(
            source_identity="fc:900102",
            slot=1,
            vehicle_overrides=VehicleQuickOverrides(
                cda_change=ScalarChange(mode=ScalarChangeMode.PERCENT, value=0.0)
            ),
        )
        second_pass = QuickScenario(
            source_identity="fc:900102",
            slot=1,
            vehicle_overrides=VehicleQuickOverrides(
                cda_change=ScalarChange(mode=ScalarChangeMode.PERCENT, value=2.0)
            ),
        )
        vehicle_1, efficiency_1 = self._resolve(first_pass)
        vehicle_2, efficiency_2 = self._resolve(second_pass)
        item_1 = build_quick_comparison_item(first_pass, vehicle_1, efficiency_1)
        item_2 = build_quick_comparison_item(second_pass, vehicle_2, efficiency_2)
        self.assertEqual(item_1.vde_id, item_2.vde_id)
        # And the values themselves genuinely differ -- this is a real
        # recalculation, not a no-op comparison.
        self.assertNotEqual(item_1.roadload["total"].C, item_2.roadload["total"].C)

    def test_rebuilding_quick_items_from_a_results_dict_never_duplicates_a_slot(self):
        # Mirrors exactly how the Streamlit tab rebuilds its `quick_items`
        # list every render (fresh from the results-by-slot dict, never an
        # accumulating list) -- proves that pattern cannot produce two
        # ComparisonItems for one recalculated slot.
        scenario = QuickScenario(
            source_identity="fc:900102",
            slot=1,
            vehicle_overrides=VehicleQuickOverrides(
                cda_change=ScalarChange(mode=ScalarChangeMode.PERCENT, value=0.0)
            ),
        )
        results_by_slot = {}
        for _ in range(3):  # simulate clicking "Calculate" three times
            results_by_slot[1] = self._resolve(scenario)

        quick_items = [
            build_quick_comparison_item(scenario, vehicle, efficiency)
            for vehicle, efficiency in results_by_slot.values()
        ]
        self.assertEqual(len(quick_items), 1)

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
