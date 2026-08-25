from __future__ import annotations

import gc
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.vde_app.comparison_report_viewmodels import (
    MAX_COMPARISONS,
    HP_PER_KW,
    PresentationRole,
    PresentationState,
    SelectionState,
    apply_engineering_filters,
    hp_to_kw,
    kw_to_hp,
    TargetState,
    WalkDeltaBase,
    WalkDisplayMode,
    WalkStep,
    WalkViewSpec,
    build_walk_rows,
    canonical_identity,
    default_walk_steps,
    delta_vs_reference_walk_steps,
    sequential_walk_steps,
    evaluate_target_gap,
    get_target,
    set_target,
    add_comparison,
    build_abc_rows,
    build_competitor_delta_rows,
    build_cycle_demand_rows,
    build_cycle_phase_rows,
    build_explore_bar_rows,
    build_explore_line_rows,
    build_explore_scatter_points,
    build_fe_vde_points,
    build_iso_pse_lines,
    compute_adaptive_pse_guides,
    build_lineage_waterfall,
    build_metric_bar_rows,
    build_reference_summary,
    build_roadload_curve_rows,
    build_scenario_header,
    build_scenario_browse_rows,
    build_scenario_options,
    build_scorecard_sections,
    dataset_items,
    dataset_warnings_summary,
    deduplicate_by_vde_id,
    format_value,
    is_current_item,
    is_temporary_net,
    list_available_explore_metrics,
    list_available_lineage_metrics,
    list_explore_dimension_values,
    list_explore_dimensions,
    list_explore_numeric_metrics,
    list_lineage_capable_metrics,
    metric_axis_label,
    presentation_role_for,
    remove_comparison,
    resolve_lineage_context,
    set_current_item,
    set_presentation_role,
    set_reference,
    sync_comparisons_from_widget,
    RowVisibility,
    ScorecardCell,
    ScorecardRow,
    ScorecardSection,
    visible_rows,
)
from src.vde_core import db as db_module
from src.vde_core.comparison_metric_registry import get_metric, list_metrics
from src.vde_core.comparison_report_service import (
    ComparisonDataset,
    ComparisonRole,
    LineageChainStatus,
    build_scenario_comparison_item,
    build_vde_comparison_item,
    list_comparison_scenarios,
    resolve_lineage_chain,
)
from src.vde_core.qa_mock_data import build_vde_seed_rows, seed_qa_database


def _qa_row(vde_id: int) -> dict:
    return dict({r["id"]: r for r in build_vde_seed_rows()}[vde_id])


# -----------------------------------------------------------------------------
# Selection (Sec 52)
# -----------------------------------------------------------------------------


class SelectionStateTests(unittest.TestCase):
    def test_reference_excluded_from_comparisons(self):
        state = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2, 3))
        state, error = add_comparison(state, 1)
        self.assertIsNotNone(error)
        self.assertNotIn(1, state.comparison_fuelcons_ids)

    def test_reference_switch_removes_duplicate_from_comparisons(self):
        state = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2, 3))
        state = set_reference(state, 2)
        self.assertEqual(state.reference_fuelcons_id, 2)
        self.assertEqual(state.comparison_fuelcons_ids, (3,))

    def test_up_to_ten_comparisons_accepted(self):
        state = SelectionState()
        for i in range(1, MAX_COMPARISONS + 1):
            state, error = add_comparison(state, i)
            self.assertIsNone(error)
        self.assertEqual(len(state.comparison_fuelcons_ids), MAX_COMPARISONS)

    def test_eleventh_comparison_rejected_explicitly(self):
        state = SelectionState()
        for i in range(1, MAX_COMPARISONS + 1):
            state, _ = add_comparison(state, i)
        state, error = add_comparison(state, 999)
        self.assertIsNotNone(error)
        self.assertEqual(len(state.comparison_fuelcons_ids), MAX_COMPARISONS)
        self.assertNotIn(999, state.comparison_fuelcons_ids)

    def test_selection_order_preserved(self):
        state = SelectionState()
        for i in (5, 1, 9, 2):
            state, _ = add_comparison(state, i)
        self.assertEqual(state.comparison_fuelcons_ids, (5, 1, 9, 2))

    def test_duplicate_add_is_a_no_op_not_an_error(self):
        state = SelectionState()
        state, _ = add_comparison(state, 5)
        state, error = add_comparison(state, 5)
        self.assertIsNone(error)
        self.assertEqual(state.comparison_fuelcons_ids, (5,))

    def test_remove_comparison(self):
        state = SelectionState(comparison_fuelcons_ids=(1, 2, 3))
        state = remove_comparison(state, 2)
        self.assertEqual(state.comparison_fuelcons_ids, (1, 3))

    def test_sync_from_widget_preserves_original_order_and_drops_removed(self):
        state = SelectionState()
        state, _ = add_comparison(state, 1)
        state, _ = add_comparison(state, 2)
        state, _ = add_comparison(state, 3)
        # widget returns a DIFFERENT order (e.g. options-list order) plus a new id, minus one
        state, errors = sync_comparisons_from_widget(state, [3, 4, 1])
        self.assertEqual(errors, ())
        self.assertEqual(state.comparison_fuelcons_ids, (1, 3, 4))

    def test_sync_from_widget_reports_cap_overflow(self):
        state = SelectionState()
        for i in range(1, MAX_COMPARISONS + 1):
            state, _ = add_comparison(state, i)
        state, errors = sync_comparisons_from_widget(state, list(range(1, MAX_COMPARISONS + 1)) + [999])
        self.assertEqual(len(errors), 1)
        self.assertNotIn(999, state.comparison_fuelcons_ids)


class PresentationRoleTests(unittest.TestCase):
    """Package 8F Increment 2 -- presentation role/Current overlay is purely
    explicit, session-only state, independent of the canonical ComparisonRole
    and of provenance. Nothing here ever inspects an item to infer a role.
    """

    def test_unknown_identity_defaults_to_unspecified(self):
        state = PresentationState()
        self.assertEqual(presentation_role_for(state, "fc:1"), PresentationRole.UNSPECIFIED)

    def test_set_role_is_reflected_for_that_identity_only(self):
        state = PresentationState()
        state = set_presentation_role(state, "fc:1", PresentationRole.PROPOSAL)
        self.assertEqual(presentation_role_for(state, "fc:1"), PresentationRole.PROPOSAL)
        self.assertEqual(presentation_role_for(state, "fc:2"), PresentationRole.UNSPECIFIED)

    def test_set_role_unspecified_clears_the_entry(self):
        state = PresentationState()
        state = set_presentation_role(state, "vde:900001", PresentationRole.BENCHMARK)
        state = set_presentation_role(state, "vde:900001", PresentationRole.UNSPECIFIED)
        self.assertEqual(presentation_role_for(state, "vde:900001"), PresentationRole.UNSPECIFIED)
        self.assertEqual(dict(state.roles), {})

    def test_current_is_independent_of_role_not_mutually_exclusive(self):
        state = PresentationState()
        state = set_presentation_role(state, "fc:1", PresentationRole.PROPOSAL)
        state = set_current_item(state, "fc:1")
        self.assertEqual(presentation_role_for(state, "fc:1"), PresentationRole.PROPOSAL)
        self.assertTrue(is_current_item(state, "fc:1"))

    def test_current_can_be_cleared(self):
        state = PresentationState()
        state = set_current_item(state, "fc:1")
        state = set_current_item(state, None)
        self.assertFalse(is_current_item(state, "fc:1"))
        self.assertIsNone(state.current_item_id)

    def test_only_one_current_item_at_a_time(self):
        state = PresentationState()
        state = set_current_item(state, "fc:1")
        state = set_current_item(state, "fc:2")
        self.assertFalse(is_current_item(state, "fc:1"))
        self.assertTrue(is_current_item(state, "fc:2"))


class TargetTests(unittest.TestCase):
    """Package 8F Increment 3 -- Target is session-only, keyed by metric_key,
    and its BETTER/WORSE reuses the exact same direction rule compare_metric()
    uses (via semantic_for_delta), never a duplicated sign convention.
    """

    def test_target_defaults_to_none_for_unset_metric(self):
        state = TargetState()
        self.assertIsNone(get_target(state, "vde_total"))

    def test_set_target_is_scoped_to_its_own_metric_key(self):
        state = TargetState()
        state = set_target(state, "vde_total", 1.2)
        self.assertEqual(get_target(state, "vde_total"), 1.2)
        self.assertIsNone(get_target(state, "fuel_l_per_100km"))

    def test_switching_metric_never_reinterprets_a_stored_target(self):
        state = TargetState()
        state = set_target(state, "vde_total", 1.2)
        state = set_target(state, "fuel_l_per_100km", 6.0)
        self.assertEqual(get_target(state, "vde_total"), 1.2)
        self.assertEqual(get_target(state, "fuel_l_per_100km"), 6.0)

    def test_clearing_target_with_none_removes_it(self):
        state = TargetState()
        state = set_target(state, "vde_total", 1.2)
        state = set_target(state, "vde_total", None)
        self.assertIsNone(get_target(state, "vde_total"))

    def test_gap_is_none_without_actual_or_target(self):
        self.assertIsNone(evaluate_target_gap("vde_total", None, 1.2))
        self.assertIsNone(evaluate_target_gap("vde_total", 1.3, None))

    def test_lower_is_better_gap_semantics(self):
        # vde_total is LOWER_IS_BETTER: actual below target is BETTER.
        better = evaluate_target_gap("vde_total", 1.1, 1.2)
        worse = evaluate_target_gap("vde_total", 1.3, 1.2)
        same = evaluate_target_gap("vde_total", 1.2, 1.2)
        self.assertEqual(better.absolute_gap, 1.1 - 1.2)
        self.assertEqual(better.semantic, "BETTER")
        self.assertEqual(worse.semantic, "WORSE")
        self.assertEqual(same.semantic, "SAME")

    def test_higher_is_better_gap_semantics(self):
        # fuel_km_per_l is HIGHER_IS_BETTER: actual above target is BETTER.
        better = evaluate_target_gap("fuel_km_per_l", 16.0, 15.0)
        worse = evaluate_target_gap("fuel_km_per_l", 14.0, 15.0)
        self.assertEqual(better.absolute_gap, 1.0)
        self.assertEqual(better.semantic, "BETTER")
        self.assertEqual(worse.semantic, "WORSE")

    def test_percent_gap_guards_against_zero_target(self):
        gap = evaluate_target_gap("vde_total", 1.0, 0.0)
        self.assertIsNone(gap.percent_gap)
        self.assertEqual(gap.absolute_gap, 1.0)

    def test_unknown_metric_key_returns_none(self):
        self.assertIsNone(evaluate_target_gap("not_a_real_metric", 1.0, 1.0))


class VersatileWalkTests(unittest.TestCase):
    """Package 8F Increment 4 -- mandatory semantic cases A-E from the 8F spec.
    Every delta must route through compare_metric() (never recomputed), the
    walk never reads vde_id_parent/DB lineage, and only advances_anchor=True
    steps change what PREVIOUS_WALK_STATE compares against.
    """

    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "walk.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)
        with sqlite3.connect(self.db_path) as con:
            rows = [
                (1, 900001, "ICE", "Gasoline", "HOMOLOGATED", 6.0, 140.0),
                (2, 900002, "ICE", "Gasoline", "ESTIMATED", 5.6, 130.0),
                (3, 900003, "ICE", "Gasoline", "ESTIMATED", 5.4, 125.0),
                (4, 900001, "ICE", "Gasoline", "SCENARIO", 5.9, 138.0),
                (5, 900002, "ICE", "Gasoline", "HOMOLOGATED", 5.8, 135.0),
            ]
            con.executemany(
                "INSERT INTO fuelcons_db (id, vde_id, electrification, fuel_type, record_origin, "
                "fuel_l_per_100km, gco2_per_km) VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            con.commit()
        # Reference, Proposal A, Proposal B, Current, Benchmark
        self.reference = build_scenario_comparison_item(1, role=ComparisonRole.REFERENCE)
        self.proposal_a = build_scenario_comparison_item(2)
        self.proposal_b = build_scenario_comparison_item(3)
        self.current = build_scenario_comparison_item(4)
        self.benchmark = build_scenario_comparison_item(5)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def _dataset(self):
        return ComparisonDataset(
            reference=self.reference,
            comparisons=(self.proposal_a, self.proposal_b, self.current, self.benchmark),
        )

    def _id(self, item):
        return canonical_identity(item)

    def test_a_sequential_development_walk(self):
        dataset = self._dataset()
        spec = WalkViewSpec(
            metric_key="fuel_l_per_100km",
            steps=(
                WalkStep(self._id(self.reference), WalkDisplayMode.ABSOLUTE, advances_anchor=True),
                WalkStep(self._id(self.proposal_a), WalkDisplayMode.DELTA, WalkDeltaBase.PREVIOUS_WALK_STATE, advances_anchor=True),
                WalkStep(self._id(self.proposal_b), WalkDisplayMode.DELTA, WalkDeltaBase.PREVIOUS_WALK_STATE, advances_anchor=True),
                WalkStep(self._id(self.current), WalkDisplayMode.ABSOLUTE, advances_anchor=True),
                WalkStep(self._id(self.benchmark), WalkDisplayMode.ABSOLUTE, advances_anchor=False),
            ),
        )
        result = build_walk_rows(dataset, spec)
        self.assertEqual(result.warnings, ())
        self.assertTrue(result.has_delta_semantics)
        rows = {row.item_id: row for row in result.rows}
        self.assertEqual(rows[self._id(self.reference)].status, "OK")
        # Proposal A delta must be vs Reference (5.6 - 6.0)
        self.assertAlmostEqual(rows[self._id(self.proposal_a)].delta_value, 5.6 - 6.0)
        self.assertEqual(rows[self._id(self.proposal_a)].delta_base_item_id, self._id(self.reference))
        # Proposal B delta must be vs Proposal A (5.4 - 5.6), NOT vs Reference
        self.assertAlmostEqual(rows[self._id(self.proposal_b)].delta_value, 5.4 - 5.6)
        self.assertEqual(rows[self._id(self.proposal_b)].delta_base_item_id, self._id(self.proposal_a))

    def test_b_alternatives_vs_reference_no_accumulation(self):
        dataset = self._dataset()
        spec = WalkViewSpec(
            metric_key="fuel_l_per_100km",
            steps=(
                WalkStep(self._id(self.reference), WalkDisplayMode.ABSOLUTE),
                WalkStep(self._id(self.proposal_a), WalkDisplayMode.DELTA, WalkDeltaBase.REFERENCE, advances_anchor=False),
                WalkStep(self._id(self.proposal_b), WalkDisplayMode.DELTA, WalkDeltaBase.REFERENCE, advances_anchor=False),
                WalkStep(self._id(self.current), WalkDisplayMode.DELTA, WalkDeltaBase.REFERENCE, advances_anchor=False),
            ),
        )
        result = build_walk_rows(dataset, spec)
        self.assertEqual(result.warnings, ())
        rows = {row.item_id: row for row in result.rows}
        # Every delta is independently vs Reference (6.0) -- never chained.
        self.assertAlmostEqual(rows[self._id(self.proposal_a)].delta_value, 5.6 - 6.0)
        self.assertAlmostEqual(rows[self._id(self.proposal_b)].delta_value, 5.4 - 6.0)
        self.assertAlmostEqual(rows[self._id(self.current)].delta_value, 5.9 - 6.0)
        for item_id in (self._id(self.proposal_a), self._id(self.proposal_b), self._id(self.current)):
            self.assertEqual(rows[item_id].delta_base_item_id, self._id(self.reference))

    def test_c_mixed_walk_benchmark_context_only_does_not_become_anchor(self):
        dataset = self._dataset()
        spec = WalkViewSpec(
            metric_key="fuel_l_per_100km",
            steps=(
                WalkStep(self._id(self.reference), WalkDisplayMode.ABSOLUTE, advances_anchor=True),
                WalkStep(self._id(self.proposal_a), WalkDisplayMode.DELTA, WalkDeltaBase.PREVIOUS_WALK_STATE, advances_anchor=True),
                WalkStep(self._id(self.benchmark), WalkDisplayMode.ABSOLUTE, advances_anchor=False),
                WalkStep(self._id(self.proposal_b), WalkDisplayMode.DELTA, WalkDeltaBase.PREVIOUS_WALK_STATE, advances_anchor=True),
                WalkStep(self._id(self.current), WalkDisplayMode.ABSOLUTE, advances_anchor=True),
            ),
        )
        result = build_walk_rows(dataset, spec)
        self.assertEqual(result.warnings, ())
        rows = {row.item_id: row for row in result.rows}
        # CRITICAL: Proposal B must compare to Proposal A, NOT Benchmark.
        self.assertEqual(rows[self._id(self.proposal_b)].delta_base_item_id, self._id(self.proposal_a))
        self.assertAlmostEqual(rows[self._id(self.proposal_b)].delta_value, 5.4 - 5.6)

    def test_d_benchmark_only_no_reference_no_fake_delta_no_exception(self):
        dataset = ComparisonDataset(reference=None, comparisons=(self.proposal_a, self.proposal_b, self.benchmark))
        spec = WalkViewSpec(
            metric_key="fuel_l_per_100km",
            steps=(
                WalkStep(self._id(self.proposal_a), WalkDisplayMode.ABSOLUTE),
                WalkStep(self._id(self.proposal_b), WalkDisplayMode.ABSOLUTE),
                WalkStep(self._id(self.benchmark), WalkDisplayMode.ABSOLUTE),
            ),
        )
        result = build_walk_rows(dataset, spec)
        self.assertEqual(result.warnings, ())
        self.assertFalse(result.has_delta_semantics)
        self.assertTrue(all(row.status == "OK" for row in result.rows))
        self.assertTrue(all(row.delta_value is None for row in result.rows))

    def test_d_benchmark_only_delta_vs_reference_is_invalid_config_not_fabricated(self):
        dataset = ComparisonDataset(reference=None, comparisons=(self.proposal_a, self.proposal_b))
        spec = WalkViewSpec(
            metric_key="fuel_l_per_100km",
            steps=(
                WalkStep(self._id(self.proposal_a), WalkDisplayMode.ABSOLUTE),
                WalkStep(self._id(self.proposal_b), WalkDisplayMode.DELTA, WalkDeltaBase.REFERENCE),
            ),
        )
        result = build_walk_rows(dataset, spec)
        rows = {row.item_id: row for row in result.rows}
        self.assertEqual(rows[self._id(self.proposal_b)].status, "INVALID_CONFIG")
        self.assertIsNone(rows[self._id(self.proposal_b)].delta_value)
        self.assertEqual(len(result.warnings), 1)

    def test_e_explicit_item_delta(self):
        dataset = self._dataset()
        spec = WalkViewSpec(
            metric_key="fuel_l_per_100km",
            steps=(
                WalkStep(self._id(self.proposal_a), WalkDisplayMode.ABSOLUTE),
                WalkStep(
                    self._id(self.proposal_b),
                    WalkDisplayMode.DELTA,
                    WalkDeltaBase.EXPLICIT_ITEM,
                    explicit_item_id=self._id(self.proposal_a),
                ),
            ),
        )
        result = build_walk_rows(dataset, spec)
        self.assertEqual(result.warnings, ())
        rows = {row.item_id: row for row in result.rows}
        self.assertEqual(rows[self._id(self.proposal_b)].delta_base_item_id, self._id(self.proposal_a))
        self.assertAlmostEqual(rows[self._id(self.proposal_b)].delta_value, 5.4 - 5.6)

    def test_previous_walk_state_with_no_prior_anchor_is_invalid_config(self):
        dataset = self._dataset()
        spec = WalkViewSpec(
            metric_key="fuel_l_per_100km",
            steps=(
                WalkStep(self._id(self.reference), WalkDisplayMode.ABSOLUTE, advances_anchor=False),
                WalkStep(self._id(self.proposal_a), WalkDisplayMode.DELTA, WalkDeltaBase.PREVIOUS_WALK_STATE),
            ),
        )
        result = build_walk_rows(dataset, spec)
        rows = {row.item_id: row for row in result.rows}
        self.assertEqual(rows[self._id(self.proposal_a)].status, "INVALID_CONFIG")

    def test_unresolvable_item_id_is_reported_never_fabricated(self):
        dataset = self._dataset()
        spec = WalkViewSpec(
            metric_key="fuel_l_per_100km",
            steps=(WalkStep("fc:999999", WalkDisplayMode.ABSOLUTE),),
        )
        result = build_walk_rows(dataset, spec)
        self.assertEqual(result.rows, ())
        self.assertEqual(len(result.warnings), 1)

    def test_target_gap_is_attached_per_row_when_target_set(self):
        dataset = self._dataset()
        spec = WalkViewSpec(
            metric_key="fuel_l_per_100km",
            steps=(WalkStep(self._id(self.reference), WalkDisplayMode.ABSOLUTE),),
            target_value=5.5,
        )
        result = build_walk_rows(dataset, spec)
        row = result.rows[0]
        self.assertIsNotNone(row.target_gap)
        self.assertAlmostEqual(row.target_gap.absolute_gap, 6.0 - 5.5)

    def test_no_target_means_no_gap(self):
        dataset = self._dataset()
        spec = WalkViewSpec(metric_key="fuel_l_per_100km", steps=(WalkStep(self._id(self.reference), WalkDisplayMode.ABSOLUTE),))
        result = build_walk_rows(dataset, spec)
        self.assertIsNone(result.rows[0].target_gap)

    def test_default_walk_steps_is_all_absolute_advancing_anchor(self):
        dataset = self._dataset()
        steps = default_walk_steps(dataset)
        self.assertEqual(len(steps), 5)
        self.assertTrue(all(step.display_mode is WalkDisplayMode.ABSOLUTE for step in steps))
        self.assertTrue(all(step.advances_anchor for step in steps))
        result = build_walk_rows(dataset, WalkViewSpec(metric_key="fuel_l_per_100km", steps=steps))
        self.assertFalse(result.has_delta_semantics)
        self.assertTrue(all(row.delta_value is None for row in result.rows))

    def test_sequential_walk_preset_chains_deltas(self):
        dataset = self._dataset()
        steps = sequential_walk_steps(dataset)
        result = build_walk_rows(dataset, WalkViewSpec(metric_key="fuel_l_per_100km", steps=steps))
        self.assertEqual(result.warnings, ())
        self.assertTrue(result.has_delta_semantics)
        rows = {row.item_id: row for row in result.rows}
        self.assertEqual(rows[self._id(self.reference)].display_mode, "ABSOLUTE")
        self.assertEqual(rows[self._id(self.proposal_a)].delta_base_item_id, self._id(self.reference))

    def test_delta_vs_reference_preset_never_accumulates(self):
        dataset = self._dataset()
        steps = delta_vs_reference_walk_steps(dataset)
        result = build_walk_rows(dataset, WalkViewSpec(metric_key="fuel_l_per_100km", steps=steps))
        self.assertEqual(result.warnings, ())
        rows = {row.item_id: row for row in result.rows}
        for item_id in (self._id(self.proposal_a), self._id(self.proposal_b), self._id(self.current), self._id(self.benchmark)):
            self.assertEqual(rows[item_id].delta_base_item_id, self._id(self.reference))

    def test_never_reads_vde_id_parent_walk_is_selection_order_only(self):
        # Walk order is exactly the caller's steps order, regardless of any
        # vde_id_parent lineage relationship between the underlying VDEs.
        dataset = self._dataset()
        spec = WalkViewSpec(
            metric_key="fuel_l_per_100km",
            steps=(
                WalkStep(self._id(self.proposal_b), WalkDisplayMode.ABSOLUTE),
                WalkStep(self._id(self.reference), WalkDisplayMode.ABSOLUTE),
            ),
        )
        result = build_walk_rows(dataset, spec)
        self.assertEqual([row.item_id for row in result.rows], [self._id(self.proposal_b), self._id(self.reference)])


class CyclePhaseTests(unittest.TestCase):
    """Package 8F Increment 7 -- true per-phase VDE, reading
    VDEBoundaryResult.by_phase directly. EPA and WLTP items are never merged
    into one chart family, and nothing is zero-filled or guessed.
    """

    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "cycle_phase.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)
        self.epa_item = build_vde_comparison_item(900001, vde_row=_qa_row(900001))
        wltp_row = _qa_row(900002)
        wltp_row["legislation"] = "WLTP"
        self.wltp_item = build_vde_comparison_item(900002, vde_row=wltp_row)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def _dataset(self, comparisons):
        return ComparisonDataset(reference=None, comparisons=tuple(comparisons))

    def test_epa_item_produces_city_highway_family(self):
        dataset = self._dataset([self.epa_item])
        result = build_cycle_phase_rows(dataset, "TOTAL")
        self.assertEqual(len(result["families"]), 1)
        self.assertEqual(result["families"][0]["family"], "EPA")
        groups = {row["group"] for row in result["families"][0]["rows"]}
        self.assertEqual(groups, {"City", "Highway"})

    def test_wltp_item_produces_low_mid_high_xhigh_family(self):
        dataset = self._dataset([self.wltp_item])
        result = build_cycle_phase_rows(dataset, "TOTAL")
        self.assertEqual(len(result["families"]), 1)
        self.assertEqual(result["families"][0]["family"], "WLTP")
        groups = {row["group"] for row in result["families"][0]["rows"]}
        self.assertEqual(groups, {"Low", "Mid", "High", "Extra High"})

    def test_epa_and_wltp_never_merged_into_one_family(self):
        dataset = self._dataset([self.epa_item, self.wltp_item])
        result = build_cycle_phase_rows(dataset, "TOTAL")
        self.assertEqual(len(result["families"]), 2)
        family_names = {block["family"] for block in result["families"]}
        self.assertEqual(family_names, {"EPA", "WLTP"})
        # Each family's rows only contain that family's own items -- no
        # cross-contamination between EPA and WLTP labels within one family.
        epa_block = next(b for b in result["families"] if b["family"] == "EPA")
        wltp_block = next(b for b in result["families"] if b["family"] == "WLTP")
        self.assertTrue(all(row["label"] == self.epa_item.label for row in epa_block["rows"]))
        self.assertTrue(all(row["label"] == self.wltp_item.label for row in wltp_block["rows"]))

    def test_item_with_no_phase_data_is_excluded_not_zero_filled(self):
        no_cycle_item = build_vde_comparison_item(900006, vde_row=_qa_row(900006))
        dataset = self._dataset([no_cycle_item])
        result = build_cycle_phase_rows(dataset, "NET")  # 900006 has no transmission -> NET unavailable
        self.assertEqual(result["families"], [])
        self.assertEqual(len(result["excluded"]), 1)


class EngineeringFilterTests(unittest.TestCase):
    """Package 8F -- Comparison Engineering filters (displacement/power) over
    catalog rows. A range is inactive unless explicitly supplied; when active,
    a row missing the field is excluded (never coerced to 0), and only rows
    genuinely outside the range are dropped.
    """

    def setUp(self):
        self.rows = [
            {"fuelcons_id": 1, "engine_size_l": 2.0, "engine_max_power_kw": 150.0},
            {"fuelcons_id": 2, "engine_size_l": 3.5, "engine_max_power_kw": 300.0},
            {"fuelcons_id": 3, "engine_size_l": None, "engine_max_power_kw": 90.0},
            {"fuelcons_id": 4, "engine_size_l": 1.6, "engine_max_power_kw": None},
        ]

    def test_no_ranges_retains_every_row(self):
        result = apply_engineering_filters(self.rows)
        self.assertEqual(len(result), 4)

    def test_displacement_range_excludes_missing_and_out_of_range(self):
        result = apply_engineering_filters(self.rows, engine_size_l_range=(1.5, 3.0))
        ids = {r["fuelcons_id"] for r in result}
        self.assertEqual(ids, {1, 4})  # 2.0L and 1.6L match; 3.5L too high; None excluded

    def test_power_range_excludes_missing_and_out_of_range(self):
        result = apply_engineering_filters(self.rows, engine_max_power_kw_range=(100.0, 200.0))
        ids = {r["fuelcons_id"] for r in result}
        self.assertEqual(ids, {1})  # 150kW matches; 300/90 out of range; None excluded

    def test_open_ended_range_only_bounds_one_side(self):
        result = apply_engineering_filters(self.rows, engine_max_power_kw_range=(200.0, None))
        ids = {r["fuelcons_id"] for r in result}
        self.assertEqual(ids, {2})

    def test_both_ranges_combine_as_and(self):
        result = apply_engineering_filters(
            self.rows, engine_size_l_range=(0.0, 5.0), engine_max_power_kw_range=(100.0, 400.0)
        )
        ids = {r["fuelcons_id"] for r in result}
        self.assertEqual(ids, {1, 2})

    def test_missing_field_never_treated_as_zero(self):
        # A row with engine_size_l=None must not match a range that includes 0.
        result = apply_engineering_filters(self.rows, engine_size_l_range=(0.0, 5.0))
        self.assertNotIn(3, {r["fuelcons_id"] for r in result})

    def test_hp_kw_roundtrip_uses_canonical_factor(self):
        self.assertAlmostEqual(hp_to_kw(kw_to_hp(100.0)), 100.0, places=6)
        self.assertAlmostEqual(kw_to_hp(1.0), HP_PER_KW, places=6)

    def test_hp_kw_conversion_passes_through_none(self):
        self.assertIsNone(hp_to_kw(None))
        self.assertIsNone(kw_to_hp(None))


class ScenarioOptionTests(unittest.TestCase):
    def test_same_vde_different_fuelcons_scenarios_remain_distinct_options(self):
        rows = [
            {"fuelcons_id": 1, "vde_id": 900001, "make": "QA", "model": "Nominal EPA baseline", "year": 2026, "legislation": "EPA", "electrification": "ICE", "record_origin": "HOMOLOGATED"},
            {"fuelcons_id": 2, "vde_id": 900001, "make": "QA", "model": "Nominal EPA baseline", "year": 2026, "legislation": "EPA", "electrification": "ICE", "record_origin": "ESTIMATED"},
        ]
        options = build_scenario_options(rows)
        self.assertEqual(len(options), 2)
        self.assertNotEqual(options[0].fuelcons_id, options[1].fuelcons_id)
        self.assertEqual(options[0].vde_id, options[1].vde_id)

    def test_label_does_not_lead_with_raw_id(self):
        rows = [{"fuelcons_id": 1, "vde_id": 900001, "make": "QA", "model": "Baseline", "year": 2026, "legislation": "EPA", "electrification": "ICE", "record_origin": "HOMOLOGATED"}]
        option = build_scenario_options(rows)[0]
        self.assertFalse(option.label.startswith("1"))
        self.assertFalse(option.label.startswith("#"))
        self.assertIn("QA Baseline", option.label)

    def test_label_includes_vde_id_and_fuelcons_id_to_disambiguate_identical_vehicle_names(self):
        rows = [
            {"fuelcons_id": 10, "vde_id": 900001, "make": "HYUNDAI", "model": "G80", "year": 2020, "legislation": "EPA", "electrification": "ICE", "record_origin": "HOMOLOGATED"},
            {"fuelcons_id": 11, "vde_id": 900002, "make": "HYUNDAI", "model": "G80", "year": 2020, "legislation": "EPA", "electrification": "ICE", "record_origin": "ESTIMATED"},
        ]
        first, second = build_scenario_options(rows)
        self.assertIn("VDE 900001", first.label)
        self.assertIn("FC 10", first.label)
        self.assertIn("VDE 900002", second.label)
        self.assertIn("FC 11", second.label)
        self.assertNotEqual(first.label, second.label)

    def test_label_falls_back_to_fuelcons_id_only_when_vde_id_missing(self):
        rows = [{"fuelcons_id": 5, "vde_id": None, "make": "QA", "model": "Orphan", "year": 2026, "legislation": "EPA", "electrification": "ICE", "record_origin": "HOMOLOGATED"}]
        option = build_scenario_options(rows)[0]
        self.assertIn("FC 5", option.label)
        self.assertNotIn("VDE", option.label)


class ScenarioBrowseRowsTests(unittest.TestCase):
    def test_maps_scorecard_equivalent_fields_into_display_columns(self):
        rows = [
            {
                "fuelcons_id": 1,
                "vde_id": 900001,
                "make": "QA",
                "model": "Nominal EPA baseline",
                "year": 2026,
                "legislation": "EPA",
                "category": "Car",
                "cycle_name": "EPA_FTP75_HWFET",
                "electrification": "ICE",
                "fuel_type": "Gasoline",
                "engine_type": "ICE",
                "drive_type": "FWD",
                "transmission_type": "AUTOMATIC",
                "transmission_status": "AVAILABLE",
                "mass_kg": 1500.0,
                "test_mass_kg": 1520.0,
                "cda_m2": 0.65,
                "rrc_N_per_kN": 8.5,
                "coast_A_N": 100.0,
                "coast_B_N_per_kph": 1.0,
                "coast_C_N_per_kph2": 0.03,
                "net_A_N": 80.0,
                "net_B_N_per_kph": 0.8,
                "net_C_N_per_kph2": 0.025,
                "vde_total_mj_per_km": 2.5,
                "vde_net_mj_per_km": 2.1,
                "fuel_l_per_100km": 7.5,
                "fuel_km_per_l": 13.3,
                "energy_Wh_per_km": 650.0,
                "gco2_per_km": 175.0,
                "eta_pt_est": 0.32,
                "gear_count": 8,
                "final_drive_ratio": 3.5,
                "record_origin": "HOMOLOGATED",
                "vde_record_origin": "HOMOLOGATED",
                "created_at": "2026-01-01T00:00:00",
            }
        ]

        [row] = build_scenario_browse_rows(rows)

        self.assertEqual(row["Fuelcons ID"], 1)
        self.assertEqual(row["VDE ID"], 900001)
        self.assertEqual(row["Make"], "QA")
        self.assertEqual(row["Mass [kg]"], 1500.0)
        self.assertEqual(row["CdA [m2]"], 0.65)
        self.assertEqual(row["RRC [N/kN]"], 8.5)
        self.assertEqual(row["Trans. status"], "AVAILABLE")
        self.assertEqual(row["ABC TOTAL"], "100/1/0.03")
        self.assertEqual(row["ABC NET"], "80/0.8/0.025")
        self.assertEqual(row["VDE TOTAL [MJ/km]"], 2.5)
        self.assertEqual(row["VDE NET [MJ/km]"], 2.1)
        self.assertEqual(row["Fuel [L/100km]"], 7.5)
        self.assertEqual(row["Scenario origin"], "HOMOLOGATED")
        self.assertEqual(row["VDE origin"], "HOMOLOGATED")

    def test_missing_net_abc_renders_as_dash_never_falls_back_to_total(self):
        rows = [
            {
                "fuelcons_id": 2,
                "vde_id": 900006,
                "coast_A_N": 100.0,
                "coast_B_N_per_kph": 1.0,
                "coast_C_N_per_kph2": 0.03,
                "net_A_N": None,
                "net_B_N_per_kph": None,
                "net_C_N_per_kph2": None,
                "vde_total_mj_per_km": 2.5,
                "vde_net_mj_per_km": None,
                "transmission_status": "MISSING",
            }
        ]

        [row] = build_scenario_browse_rows(rows)

        self.assertEqual(row["ABC TOTAL"], "100/1/0.03")
        self.assertEqual(row["ABC NET"], "-")
        self.assertEqual(row["VDE NET [MJ/km]"], None)
        self.assertNotEqual(row["ABC NET"], row["ABC TOTAL"])

    def test_empty_catalog_yields_empty_rows(self):
        self.assertEqual(build_scenario_browse_rows([]), [])


# -----------------------------------------------------------------------------
# Scorecard construction, deltas, compatibility, provenance, NET missing (Sec 53-59)
# -----------------------------------------------------------------------------


class ScorecardConstructionTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "scorecard.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)
        with sqlite3.connect(self.db_path) as con:
            rows = [
                (1, 900001, "ICE", "Gasoline", "HOMOLOGATED", None, "2026-07-16T00:00:00Z", 6.0, 16.7, None, 140.0, 0.30),
                (2, 900002, "ICE", "Gasoline", "ESTIMATED", "PWT_L0", "2026-07-16T00:00:00Z", 6.5, 15.4, None, 150.0, 0.28),
                (3, 900001, "ICE", "Gasoline", "SCENARIO", None, "2000-01-01T00:00:00Z", 5.8, 17.2, None, 130.0, 0.32),
                (4, 900006, "BEV", "Electric", "ESTIMATED", None, "2026-07-16T00:00:00Z", None, None, 150.0, 0.0, 0.90),
            ]
            con.executemany(
                "INSERT INTO fuelcons_db (id, vde_id, electrification, fuel_type, record_origin, "
                "engine_method, source_vde_revision, fuel_l_per_100km, fuel_km_per_l, energy_Wh_per_km, "
                "gco2_per_km, eta_pt_est) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            con.commit()
        self.reference = build_scenario_comparison_item(1, role=ComparisonRole.REFERENCE)
        self.estimated = build_scenario_comparison_item(2)
        self.stale = build_scenario_comparison_item(3)
        self.bev_no_net = build_scenario_comparison_item(4)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def _dataset(self, comparisons):
        from src.vde_core.comparison_report_service import ComparisonDataset

        return ComparisonDataset(reference=self.reference, comparisons=tuple(comparisons))

    def test_reference_is_first_and_section_order_is_deterministic(self):
        dataset = self._dataset([self.estimated])
        sections = build_scorecard_sections(dataset)
        titles = [s.title for s in sections]
        self.assertEqual(
            titles,
            [
                "Vehicle / Program",
                "Powertrain",
                "Physical Setup",
                "Roadload",
                "Vehicle Demand",
                "Fuel / Energy / Emissions",
                "Efficiency",
                "Data Status / Provenance",
            ],
        )

    def test_row_order_is_deterministic_across_calls(self):
        dataset = self._dataset([self.estimated])
        sections1 = build_scorecard_sections(dataset)
        sections2 = build_scorecard_sections(dataset)
        self.assertEqual([r.metric_key for s in sections1 for r in s.rows], [r.metric_key for s in sections2 for r in s.rows])

    def test_missing_value_renders_dash_not_zero(self):
        dataset = self._dataset([self.bev_no_net])
        sections = build_scorecard_sections(dataset)
        roadload_section = next(s for s in sections if s.title == "Roadload")
        net_row = next(r for r in roadload_section.rows if r.metric_key == "roadload_a_net")
        cell = net_row.comparison_cells[0]
        self.assertFalse(cell.available)
        self.assertEqual(cell.formatted_value, "-")
        self.assertNotEqual(cell.formatted_value, "0")

    def test_lower_is_better_improvement_marks_better(self):
        dataset = self._dataset([self.stale])  # fuel_l_per_100km 5.8 < reference 6.0
        sections = build_scorecard_sections(dataset)
        fuel_section = next(s for s in sections if s.title == "Fuel / Energy / Emissions")
        row = next(r for r in fuel_section.rows if r.metric_key == "fuel_l_per_100km")
        cell = row.comparison_cells[0]
        self.assertEqual(cell.semantic, "BETTER")
        self.assertIsNotNone(cell.formatted_delta)

    def test_higher_is_better_improvement_marks_better(self):
        dataset = self._dataset([self.stale])  # fuel_km_per_l 17.2 > reference 16.7
        sections = build_scorecard_sections(dataset)
        fuel_section = next(s for s in sections if s.title == "Fuel / Energy / Emissions")
        row = next(r for r in fuel_section.rows if r.metric_key == "fuel_km_per_l")
        cell = row.comparison_cells[0]
        self.assertEqual(cell.semantic, "BETTER")

    def test_worse_result_is_marked(self):
        dataset = self._dataset([self.estimated])  # gco2 150 > reference 140
        sections = build_scorecard_sections(dataset)
        fuel_section = next(s for s in sections if s.title == "Fuel / Energy / Emissions")
        row = next(r for r in fuel_section.rows if r.metric_key == "gco2_per_km")
        cell = row.comparison_cells[0]
        self.assertEqual(cell.semantic, "WORSE")

    def test_neutral_metric_has_no_semantic_verdict(self):
        dataset = self._dataset([self.estimated])
        sections = build_scorecard_sections(dataset)
        physical_section = next(s for s in sections if s.title == "Physical Setup")
        row = next(r for r in physical_section.rows if r.metric_key == "mass_kg")
        for cell in row.comparison_cells:
            self.assertIsNone(cell.semantic)

    def test_zero_reference_yields_absolute_delta_but_no_percent(self):
        dataset = self._dataset([self.estimated])
        sections = build_scorecard_sections(dataset)
        energy_section = next(s for s in sections if s.title == "Fuel / Energy / Emissions")
        row = next(r for r in energy_section.rows if r.metric_key == "energy_wh_per_km")
        # reference has no energy_Wh_per_km at all (ICE) -> unavailable, not a crash
        cell = row.comparison_cells[0]
        self.assertFalse(cell.available)

    def test_epa_vs_epa_cycle_metric_is_compatible(self):
        dataset = self._dataset([self.estimated])
        sections = build_scorecard_sections(dataset)
        demand_section = next(s for s in sections if s.title == "Vehicle Demand")
        row = next(r for r in demand_section.rows if r.metric_key == "vde_total")
        self.assertTrue(row.comparison_cells[0].compatible)

    def test_epa_vs_wltp_cycle_metric_is_incompatible_no_misleading_delta(self):
        wltp_row = _qa_row(900002)
        wltp_row["legislation"] = "WLTP"
        wltp_item = build_vde_comparison_item(900002, vde_row=wltp_row)
        dataset = self._dataset([wltp_item])
        sections = build_scorecard_sections(dataset)
        demand_section = next(s for s in sections if s.title == "Vehicle Demand")
        row = next(r for r in demand_section.rows if r.metric_key == "vde_total")
        cell = row.comparison_cells[0]
        self.assertFalse(cell.compatible)
        self.assertIsNone(cell.absolute_delta)
        self.assertIsNone(cell.semantic)
        self.assertIsNotNone(cell.warning)

    def test_physical_metrics_remain_available_across_legislations(self):
        wltp_row = _qa_row(900002)
        wltp_row["legislation"] = "WLTP"
        wltp_item = build_vde_comparison_item(900002, vde_row=wltp_row)
        dataset = self._dataset([wltp_item])
        sections = build_scorecard_sections(dataset)
        physical_section = next(s for s in sections if s.title == "Physical Setup")
        row = next(r for r in physical_section.rows if r.metric_key == "cda_m2")
        self.assertTrue(row.comparison_cells[0].compatible)
        self.assertTrue(row.comparison_cells[0].available)

    def test_mixed_provenance_dataset_renders_all_items(self):
        dataset = self._dataset([self.estimated, self.stale])
        sections = build_scorecard_sections(dataset)
        provenance_section = next(s for s in sections if s.title == "Data Status / Provenance")
        origin_row = next(r for r in provenance_section.rows if r.metric_key == "provenance_record_origin")
        self.assertEqual(origin_row.reference_cell.raw_value, "HOMOLOGATED")
        self.assertEqual([c.raw_value for c in origin_row.comparison_cells], ["ESTIMATED", "SCENARIO"])

    def test_stale_status_is_visible_and_not_filtered(self):
        dataset = self._dataset([self.stale])
        sections = build_scorecard_sections(dataset)
        provenance_section = next(s for s in sections if s.title == "Data Status / Provenance")
        revision_row = next(r for r in provenance_section.rows if r.metric_key == "provenance_revision_status")
        self.assertEqual(revision_row.comparison_cells[0].raw_value, "STALE")
        header = build_scenario_header(self.stale)
        self.assertTrue(header["is_stale"])
        self.assertIn("STALE SOURCE", header["column_title"])

    def test_net_missing_total_present_no_fallback_no_crash(self):
        dataset = self._dataset([self.bev_no_net])
        sections = build_scorecard_sections(dataset)
        demand_section = next(s for s in sections if s.title == "Vehicle Demand")
        total_row = next(r for r in demand_section.rows if r.metric_key == "vde_total")
        net_row = next(r for r in demand_section.rows if r.metric_key == "vde_net")
        self.assertTrue(total_row.comparison_cells[0].available)
        self.assertFalse(net_row.comparison_cells[0].available)
        self.assertEqual(net_row.comparison_cells[0].formatted_value, "-")

    def test_same_vde_multiple_scenarios_produce_distinct_columns(self):
        item_a = build_scenario_comparison_item(1)
        item_b = build_scenario_comparison_item(3)
        dataset = self._dataset([item_a, item_b])
        self.assertEqual(len(dataset.comparisons), 2)
        self.assertEqual(dataset.comparisons[0].vde_id, dataset.comparisons[1].vde_id)
        self.assertNotEqual(dataset.comparisons[0].fuelcons_id, dataset.comparisons[1].fuelcons_id)

    def test_bev_scenario_does_not_coerce_into_fuel_row(self):
        dataset = self._dataset([self.bev_no_net])
        sections = build_scorecard_sections(dataset)
        fuel_section = next(s for s in sections if s.title == "Fuel / Energy / Emissions")
        fuel_row = next(r for r in fuel_section.rows if r.metric_key == "fuel_l_per_100km")
        self.assertFalse(fuel_row.comparison_cells[0].available)

    def test_dataset_warnings_flags_stale_missing_net_and_mixed_legislation(self):
        wltp_row = _qa_row(900002)
        wltp_row["legislation"] = "WLTP"
        wltp_item = build_vde_comparison_item(900002, vde_row=wltp_row)
        dataset = self._dataset([self.stale, self.bev_no_net, wltp_item])
        warnings = dataset_warnings_summary(dataset)
        joined = " ".join(warnings)
        self.assertIn("stale", joined.lower())
        self.assertIn("NET boundary", joined)
        self.assertIn("Mixed legislations", joined)


class RowVisibilityPolicyTests(unittest.TestCase):
    """Sprint 9 post-freeze hotfix -- "unavailable is information" for
    basic/canonical engineering audit rows; the legacy AUTO behavior
    (disappear when nothing is available) is unchanged for everything else.
    """

    def _dataset_missing(self, field_names: list[str], *, reference_less: bool = False):
        row_a = _qa_row(900001)
        row_b = _qa_row(900004)
        for field_name in field_names:
            row_a[field_name] = None
            row_b[field_name] = None
        item_a = build_vde_comparison_item(900001, role=ComparisonRole.COMPARISON, vde_row=row_a)
        item_b = build_vde_comparison_item(900004, role=ComparisonRole.COMPARISON, vde_row=row_b)
        if reference_less:
            return ComparisonDataset(reference=None, comparisons=(item_a, item_b))
        item_a = build_vde_comparison_item(900001, role=ComparisonRole.REFERENCE, vde_row=row_a)
        return ComparisonDataset(reference=item_a, comparisons=(item_b,))

    def test_1_canonical_row_unavailable_for_all_scenarios_still_renders(self):
        dataset = self._dataset_missing(["cda_m2"])
        section = next(s for s in build_scorecard_sections(dataset) if s.title == "Physical Setup")
        cda_row = next(r for r in section.rows if r.metric_key == "cda_m2")
        self.assertIs(cda_row.visibility, RowVisibility.ALWAYS)
        self.assertFalse(cda_row.reference_cell.available)
        visible = visible_rows(section)
        self.assertIn(cda_row, visible)

    def test_2_optional_auto_row_unavailable_for_all_scenarios_keeps_legacy_hidden_behavior(self):
        dataset = self._dataset_missing(["gear_count"])
        section = next(s for s in build_scorecard_sections(dataset) if s.title == "Powertrain")
        gear_row = next(r for r in section.rows if r.metric_key == "gear_count")
        self.assertIs(gear_row.visibility, RowVisibility.AUTO)
        self.assertFalse(gear_row.reference_cell.available)
        visible = visible_rows(section)
        self.assertNotIn(gear_row, visible)

    def test_basic_scorecard_canonical_metrics_are_exactly_the_expected_set(self):
        expected_always_visible = {
            "mass_kg", "cda_m2", "rrc_n_per_kn",
            "roadload_a_total", "roadload_b_total", "roadload_c_total",
            "roadload_a_net", "roadload_b_net", "roadload_c_net",
            "vde_total", "vde_net",
        }
        actual_always_visible = {m.key for m in list_metrics() if m.always_visible}
        self.assertEqual(actual_always_visible, expected_always_visible)

    def test_6_net_roadload_metric_stays_auditable_with_no_total_fallback(self):
        # VDE-QA-006 has no resolved transmission -> NET roadload/VDE unavailable.
        dataset = ComparisonDataset(
            reference=build_vde_comparison_item(900006, role=ComparisonRole.REFERENCE, vde_row=_qa_row(900006)),
            comparisons=(),
        )
        section = next(s for s in build_scorecard_sections(dataset) if s.title == "Roadload")
        net_a_row = next(r for r in section.rows if r.metric_key == "roadload_a_net")
        total_a_row = next(r for r in section.rows if r.metric_key == "roadload_a_total")

        self.assertIn(net_a_row, visible_rows(section))
        self.assertFalse(net_a_row.reference_cell.available)
        self.assertTrue(total_a_row.reference_cell.available)
        self.assertNotEqual(net_a_row.reference_cell.raw_value, total_a_row.reference_cell.raw_value)

    def test_7_reference_less_dataset_uses_the_same_visibility_policy(self):
        dataset = self._dataset_missing(["cda_m2"], reference_less=True)
        section = next(s for s in build_scorecard_sections(dataset) if s.title == "Physical Setup")
        cda_row = next(r for r in section.rows if r.metric_key == "cda_m2")
        self.assertIn(cda_row, visible_rows(section))
        self.assertFalse(cda_row.reference_cell.available)

    def test_8_zero_value_on_an_always_visible_row_is_available_not_missing(self):
        row = _qa_row(900001)
        row["rrc_N_per_kN"] = 0.0
        dataset = ComparisonDataset(
            reference=build_vde_comparison_item(900001, role=ComparisonRole.REFERENCE, vde_row=row), comparisons=()
        )
        section = next(s for s in build_scorecard_sections(dataset) if s.title == "Physical Setup")
        rrc_row = next(r for r in section.rows if r.metric_key == "rrc_n_per_kn")

        self.assertIn(rrc_row, visible_rows(section))
        self.assertTrue(rrc_row.reference_cell.available)
        self.assertEqual(rrc_row.reference_cell.raw_value, 0.0)
        self.assertNotEqual(rrc_row.reference_cell.formatted_value, "-")

    def test_visible_rows_preserves_row_order(self):
        section = ScorecardSection(
            title="X",
            rows=(
                ScorecardRow(
                    metric_key="a",
                    label="A",
                    reference_cell=ScorecardCell(None, "-", None, None, None, None, True, False, False, None),
                    visibility=RowVisibility.ALWAYS,
                ),
                ScorecardRow(
                    metric_key="b",
                    label="B",
                    reference_cell=ScorecardCell(1.0, "1", None, None, None, None, True, True, False, None),
                ),
                ScorecardRow(
                    metric_key="c",
                    label="C",
                    reference_cell=ScorecardCell(None, "-", None, None, None, None, True, False, False, None),
                ),
            ),
        )
        self.assertEqual([r.metric_key for r in visible_rows(section)], ["a", "b"])


class OptionalReferenceViewmodelTests(unittest.TestCase):
    """Package 8F Increment 1 -- every dataset-consuming builder must degrade
    gracefully (ABSOLUTE-only, no fabricated delta/anchor) when
    dataset.reference is None. compare_metric() itself is never called with
    None -- these builders route through a Reference-less branch instead.
    """

    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "optional_reference_vm.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT INTO fuelcons_db (id, vde_id, electrification, fuel_type, record_origin, fuel_l_per_100km) "
                "VALUES (1, 900001, 'ICE', 'Gasoline', 'ESTIMATED', 6.0)"
            )
            con.execute(
                "INSERT INTO fuelcons_db (id, vde_id, electrification, fuel_type, record_origin, fuel_l_per_100km) "
                "VALUES (2, 900002, 'ICE', 'Gasoline', 'ESTIMATED', 6.5)"
            )
            con.commit()
        self.benchmark_a = build_vde_comparison_item(900001, vde_row=_qa_row(900001))
        self.benchmark_b = build_vde_comparison_item(900002, vde_row=_qa_row(900002))
        self.benchmark_c = build_vde_comparison_item(900003, vde_row=_qa_row(900003))
        self.fuel_a = build_scenario_comparison_item(1)
        self.fuel_b = build_scenario_comparison_item(2)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def _dataset(self, comparisons):
        return ComparisonDataset(reference=None, comparisons=tuple(comparisons))

    def test_dataset_items_returns_comparisons_only_when_no_reference(self):
        dataset = self._dataset([self.benchmark_a, self.benchmark_b])
        self.assertEqual(dataset_items(dataset), (self.benchmark_a, self.benchmark_b))

    def test_no_item_holds_reference_role_in_a_benchmark_only_dataset(self):
        dataset = self._dataset([self.benchmark_a, self.benchmark_b, self.benchmark_c])
        self.assertTrue(all(item.role is ComparisonRole.COMPARISON for item in dataset_items(dataset)))

    def test_scorecard_sections_render_absolute_only_no_delta(self):
        dataset = self._dataset([self.benchmark_a, self.benchmark_b, self.benchmark_c])
        sections = build_scorecard_sections(dataset)
        vde_section = next(s for s in sections if s.title == "Vehicle Demand")
        row = next(r for r in vde_section.rows if r.metric_key == "vde_total")
        for cell in (row.reference_cell, *row.comparison_cells):
            self.assertIsNone(cell.absolute_delta)
            self.assertIsNone(cell.semantic)
            self.assertIsNone(cell.formatted_delta)

    def test_metric_bar_rows_are_absolute_only_without_reference(self):
        dataset = self._dataset([self.benchmark_a, self.benchmark_b])
        result = build_metric_bar_rows(dataset, "vde_total")
        self.assertEqual(len(result["rows"]), 2)
        self.assertTrue(all(row.semantic is None for row in result["rows"]))

    def test_competitor_delta_rows_empty_without_reference(self):
        dataset = self._dataset([self.benchmark_a, self.benchmark_b])
        result = build_competitor_delta_rows(dataset, "vde_total")
        self.assertEqual(result["rows"], [])
        self.assertEqual(result["excluded"], [])

    def test_abc_rows_work_without_reference(self):
        dataset = self._dataset([self.benchmark_a, self.benchmark_b])
        result = build_abc_rows(dataset, "TOTAL")
        self.assertEqual(len(result["rows"]), 2)

    def test_dataset_warnings_summary_works_without_reference(self):
        no_net_item = build_vde_comparison_item(900006, vde_row=_qa_row(900006))
        dataset = self._dataset([no_net_item, self.benchmark_a])
        warnings = dataset_warnings_summary(dataset)
        self.assertTrue(any("NET boundary" in w for w in warnings))

    def test_fe_vde_points_volumetric_anchor_without_reference_role(self):
        dataset = self._dataset([self.fuel_a, self.fuel_b])
        result = build_fe_vde_points(dataset, boundary="TOTAL", mode="volumetric")
        self.assertEqual(len(result["points"]), 2)
        self.assertTrue(all(p["role"] != "REFERENCE" for p in result["points"]))


class DashboardRoadloadViewModelTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "dashboard.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)
        with sqlite3.connect(self.db_path) as con:
            rows = [
                (1, 900001, "ICE", "Gasoline", "HOMOLOGATED", 6.0, 16.7, None, 140.0, 0.30),
                (2, 900001, "ICE", "Gasoline", "ESTIMATED", 6.5, 15.4, None, 150.0, 0.28),  # shares VDE with #1
                (3, 900006, "BEV", "Electric", "ESTIMATED", None, None, 150.0, 0.0, 0.90),  # missing transmission -> no NET
            ]
            con.executemany(
                "INSERT INTO fuelcons_db (id, vde_id, electrification, fuel_type, record_origin, "
                "fuel_l_per_100km, fuel_km_per_l, energy_Wh_per_km, gco2_per_km, eta_pt_est) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            con.commit()
        self.reference = build_scenario_comparison_item(1, role=ComparisonRole.REFERENCE)
        self.same_vde = build_scenario_comparison_item(2)
        self.bev_no_net = build_scenario_comparison_item(3)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def _dataset(self, comparisons):
        return ComparisonDataset(reference=self.reference, comparisons=tuple(comparisons))

    def test_reference_summary_has_no_score(self):
        summary = build_reference_summary(self.reference)
        self.assertEqual(summary["vde_total"], 1.24)
        self.assertNotIn("score", summary)

    def test_metric_bar_rows_include_reference_and_comparison(self):
        dataset = self._dataset([self.same_vde])
        result = build_metric_bar_rows(dataset, "vde_total")
        self.assertEqual(len(result["rows"]), 2)
        self.assertEqual(result["excluded"], [])

    def test_metric_bar_rows_net_excludes_missing_with_reason_never_falls_back(self):
        dataset = self._dataset([self.bev_no_net])
        result = build_metric_bar_rows(dataset, "vde_net")
        self.assertEqual(len(result["rows"]), 1)  # only reference has NET
        self.assertEqual(len(result["excluded"]), 1)
        self.assertIn("unavailable", result["excluded"][0]["reason"])

    def test_dedup_by_vde_id_collapses_shared_vde_with_attribution(self):
        dataset = self._dataset([self.same_vde])
        groups = deduplicate_by_vde_id((dataset.reference, *dataset.comparisons))
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0].used_by), 2)
        self.assertEqual(len(dataset.comparisons), 1)  # Scorecard-level distinctness untouched (8B)

    def test_abc_rows_dedup_and_reflect_boundary(self):
        dataset = self._dataset([self.same_vde])
        result = build_abc_rows(dataset, "TOTAL")
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(len(result["rows"][0]["used_by"]), 2)
        self.assertEqual(result["rows"][0]["A"], 118.0)

    def test_roadload_curve_rows_match_plots_module_shape(self):
        dataset = self._dataset([])
        result = build_roadload_curve_rows(dataset, "TOTAL")
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(set(result["rows"][0]) - {"label"}, {"A_N", "B_N_per_kph", "C_N_per_kph2"})

    def test_roadload_curve_rows_net_missing_excluded_no_fallback(self):
        dataset = self._dataset([self.bev_no_net])
        result = build_roadload_curve_rows(dataset, "NET")
        excluded_labels = {e["label"] for e in result["excluded"]}
        self.assertIn(self.bev_no_net.label, excluded_labels)
        self.assertTrue(all("unavailable" in e["reason"] for e in result["excluded"]))

    def test_cycle_demand_rows_total_and_net_are_distinct_series(self):
        from src.vde_core.cycles import use_standard_cycle

        cycle = use_standard_cycle("EPA")
        dataset = self._dataset([])
        result = build_cycle_demand_rows(dataset, cycle, ["TOTAL", "NET"])
        boundaries = {s["boundary"] for s in result["series"]}
        self.assertEqual(boundaries, {"TOTAL", "NET"})
        total_series = next(s for s in result["series"] if s["boundary"] == "TOTAL")
        net_series = next(s for s in result["series"] if s["boundary"] == "NET")
        self.assertNotEqual(total_series["demanded_power_kw"], net_series["demanded_power_kw"])

    def test_cycle_demand_rows_missing_net_excluded_no_fallback(self):
        from src.vde_core.cycles import use_standard_cycle

        cycle = use_standard_cycle("EPA")
        dataset = self._dataset([self.bev_no_net])
        result = build_cycle_demand_rows(dataset, cycle, ["TOTAL", "NET"])
        net_scenario_ids = {s["scenario_id"] for s in result["series"] if s["boundary"] == "NET"}
        self.assertNotIn(str(self.bev_no_net.vde_id), net_scenario_ids)
        self.assertTrue(any(e["reason"].endswith("unavailable") for e in result["excluded"]))


class FeVdePseCompetitorDeltaTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "fevde.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)
        with sqlite3.connect(self.db_path) as con:
            rows = [
                (1, 900001, "ICE", "Gasoline", "HOMOLOGATED", 6.0, None, 140.0),
                (2, 900002, "ICE", "Gasoline", "ESTIMATED", 6.5, None, 150.0),
                (3, 900003, "ICE", "Ethanol", "ESTIMATED", 8.5, None, 130.0),
                (4, 900004, "ICE", "Flex", "ESTIMATED", 7.0, None, 145.0),
                (5, 900006, "BEV", "Electric", "ESTIMATED", None, 150.0, 0.0),
                (6, 900005, "ICE", "Tier 2 Cert Gasoline", "ESTIMATED", 6.8, None, 155.0),
            ]
            con.executemany(
                "INSERT INTO fuelcons_db (id, vde_id, electrification, fuel_type, record_origin, "
                "fuel_l_per_100km, energy_Wh_per_km, gco2_per_km) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            con.commit()
        self.reference = build_scenario_comparison_item(1, role=ComparisonRole.REFERENCE)
        self.gasoline = build_scenario_comparison_item(2)
        self.ethanol = build_scenario_comparison_item(3)
        self.flex = build_scenario_comparison_item(4)
        self.bev = build_scenario_comparison_item(5)
        self.tier2_gasoline = build_scenario_comparison_item(6)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def _dataset(self, comparisons):
        return ComparisonDataset(reference=self.reference, comparisons=tuple(comparisons))

    def test_volumetric_mode_includes_same_fuel_family_only(self):
        dataset = self._dataset([self.gasoline, self.ethanol])
        result = build_fe_vde_points(dataset, boundary="TOTAL", mode="volumetric")
        point_labels = {p["label"] for p in result["points"]}
        self.assertIn(self.gasoline.label, point_labels)
        self.assertNotIn(self.ethanol.label, point_labels)
        self.assertIn(self.ethanol.label, {e["label"] for e in result["excluded"]})

    def test_energy_normalized_mode_includes_ethanol_and_bev_excludes_flex(self):
        dataset = self._dataset([self.ethanol, self.flex, self.bev])
        result = build_fe_vde_points(dataset, boundary="TOTAL", mode="energy_normalized")
        point_labels = {p["label"] for p in result["points"]}
        self.assertIn(self.ethanol.label, point_labels)
        self.assertIn(self.bev.label, point_labels)
        self.assertNotIn(self.flex.label, point_labels)
        self.assertIn(self.flex.label, {e["label"] for e in result["excluded"]})

    def test_electrical_mode_only_includes_bev(self):
        dataset = self._dataset([self.gasoline, self.bev])
        result = build_fe_vde_points(dataset, boundary="TOTAL", mode="electrical")
        point_labels = {p["label"] for p in result["points"]}
        self.assertIn(self.bev.label, point_labels)
        self.assertNotIn(self.gasoline.label, point_labels)

    def test_missing_net_point_excluded_with_reason_no_fallback(self):
        dataset = self._dataset([self.bev])
        result = build_fe_vde_points(dataset, boundary="NET", mode="electrical")
        self.assertIn(self.bev.label, {e["label"] for e in result["excluded"]})
        self.assertEqual(result["points"], [])

    def test_iso_pse_lines_volumetric_empty_for_unmappable_fuel(self):
        self.assertEqual(build_iso_pse_lines(0.2, 1.2, [0.3], mode="volumetric", fuel_type="Flex"), [])
        self.assertEqual(build_iso_pse_lines(0.2, 1.2, [0.3], mode="volumetric", fuel_type=None), [])

    def test_iso_pse_lines_volumetric_present_for_mappable_fuel(self):
        lines = build_iso_pse_lines(0.2, 1.2, [0.3], mode="volumetric", fuel_type="Gasoline")
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["eta"], 0.3)
        self.assertEqual(len(lines[0]["x"]), 40)

    def test_iso_pse_lines_change_with_basis(self):
        vol_lines = build_iso_pse_lines(0.5, 0.5, [0.3], mode="volumetric", fuel_type="Gasoline")
        energy_lines = build_iso_pse_lines(0.5, 0.5, [0.3], mode="energy_normalized")
        electrical_lines = build_iso_pse_lines(0.5, 0.5, [0.3], mode="electrical")
        values = {vol_lines[0]["y"][0], energy_lines[0]["y"][0], electrical_lines[0]["y"][0]}
        self.assertEqual(len(values), 3)

    def test_tier_2_cert_gasoline_scenario_compatible_with_plain_gasoline_reference(self):
        # Package 8F fuel-normalization patch: "Tier 2 Cert Gasoline" must
        # resolve to the same GASOLINE family as the plain "Gasoline"
        # Reference -- compatibility is judged on the RESOLVED family, not
        # raw string equality.
        dataset = self._dataset([self.tier2_gasoline])
        result = build_fe_vde_points(dataset, boundary="TOTAL", mode="volumetric")
        point_labels = {p["label"] for p in result["points"]}
        self.assertIn(self.tier2_gasoline.label, point_labels)
        self.assertEqual(result["excluded"], [])

    def test_tier_2_cert_gasoline_point_carries_assumption_fuel_basis_label(self):
        dataset = self._dataset([self.tier2_gasoline])
        result = build_fe_vde_points(dataset, boundary="TOTAL", mode="volumetric")
        point = next(p for p in result["points"] if p["label"] == self.tier2_gasoline.label)
        self.assertTrue(point["fuel_basis_label"])
        self.assertIn("gasoline", point["fuel_basis_label"].lower())

    def test_assumption_label_surfaced_when_anchor_uses_an_assumed_basis(self):
        # Reference-less dataset so the Tier-2 item itself becomes the
        # volumetric-mode anchor -- the discreet "PSE energy basis" caption
        # only needs to appear when the ANCHOR is resolved via an assumption.
        dataset = ComparisonDataset(reference=None, comparisons=(self.tier2_gasoline,))
        result = build_fe_vde_points(dataset, boundary="TOTAL", mode="volumetric")
        self.assertTrue(result["assumption_label"])
        self.assertIn("gasoline", result["assumption_label"].lower())

    def test_assumption_label_absent_for_plain_gasoline_anchor(self):
        dataset = self._dataset([self.gasoline])
        result = build_fe_vde_points(dataset, boundary="TOTAL", mode="volumetric")
        self.assertIsNone(result["assumption_label"])

    def test_flex_still_excluded_from_volumetric_mode_even_alongside_gasoline(self):
        dataset = self._dataset([self.flex])
        result = build_fe_vde_points(dataset, boundary="TOTAL", mode="volumetric")
        point_labels = {p["label"] for p in result["points"]}
        self.assertNotIn(self.flex.label, point_labels)
        self.assertIn(self.reference.label, point_labels)  # the Gasoline Reference itself is unaffected
        self.assertIn(self.flex.label, {e["label"] for e in result["excluded"]})

    def test_iso_pse_lines_present_for_tier_2_cert_gasoline(self):
        # Equi-PSE guides must render for the approved-assumption fuel label,
        # not just an exact "Gasoline" match.
        lines = build_iso_pse_lines(0.2, 1.2, [0.3], mode="volumetric", fuel_type="Tier 2 Cert Gasoline")
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["eta"], 0.3)

    def test_iso_pse_lines_still_empty_for_flex(self):
        self.assertEqual(build_iso_pse_lines(0.2, 1.2, [0.3], mode="volumetric", fuel_type="Flex"), [])

    def test_competitor_delta_reference_is_zero_no_verdict(self):
        dataset = self._dataset([self.gasoline])
        result = build_competitor_delta_rows(dataset, "vde_total")
        ref_row = result["rows"][0]
        self.assertEqual(ref_row["role"], "REFERENCE")
        self.assertEqual(ref_row["percent_delta"], 0.0)
        self.assertIsNone(ref_row["semantic"])

    def test_competitor_delta_vde_lower_is_better(self):
        cheaper_row = _qa_row(900002)
        cheaper_row["id"] = 900002
        cheaper_row["vde_total_mj_per_km"] = 1.10  # reference (900001) is 1.24
        cheaper_item = build_scenario_comparison_item(2, vde_row=cheaper_row)
        dataset = self._dataset([cheaper_item])
        result = build_competitor_delta_rows(dataset, "vde_total")
        self.assertEqual(result["rows"][1]["semantic"], "BETTER")

    def test_competitor_delta_fuel_economy_higher_is_better(self):
        dataset = self._dataset([self.ethanol])  # any comparison; direction under test is registry-driven
        result = build_competitor_delta_rows(dataset, "fuel_km_per_l")
        # ethanol fuel_km_per_l unavailable (only fuel_l_per_100km seeded) -> excluded, proving no fabrication
        self.assertEqual(result["rows"], [result["rows"][0]])
        self.assertTrue(result["excluded"])

    def test_competitor_delta_mass_is_neutral(self):
        dataset = self._dataset([self.gasoline])
        result = build_competitor_delta_rows(dataset, "mass_kg")
        self.assertIsNone(result["rows"][1]["semantic"])

    def test_competitor_delta_incompatible_metric_is_excluded_not_faked(self):
        wltp_row = _qa_row(900002)
        wltp_row["id"] = 900002
        wltp_row["legislation"] = "WLTP"
        wltp_scenario = build_scenario_comparison_item(2, vde_row=wltp_row)
        dataset = self._dataset([wltp_scenario])
        result = build_competitor_delta_rows(dataset, "vde_total")
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(len(result["excluded"]), 1)

    def test_is_temporary_net_reflects_transmission_source(self):
        row = _qa_row(900006)
        temp = {"source": "MANUAL", "A": 9.0, "B": 0.003, "C": 0.0006}
        temporary_item = build_vde_comparison_item(900006, vde_row=row, temporary_transmission=temp)
        missing_item = build_vde_comparison_item(900006, vde_row=row)
        self.assertTrue(is_temporary_net(temporary_item))
        self.assertFalse(is_temporary_net(missing_item))


class AdaptivePseGuideTests(unittest.TestCase):
    """Sprint 8 micro-polish: equi-PSE guides are sized to what's actually
    plotted instead of a fixed 20/25/30/35 set. energy_normalized mode is
    used for most cases here because its PSE formula is the identity
    pse=x/y, so setting y=1.0 lets a point's PSE be dictated directly by x
    -- no need to route through a full ComparisonDataset for pure numeric
    checks of the guide-sizing algorithm itself.
    """

    def _pse_points(self, pse_values):
        return [{"x": v, "y": 1.0} for v in pse_values]

    def test_guides_adapt_to_a_low_pse_range(self):
        guides = compute_adaptive_pse_guides(
            self._pse_points([0.234, 0.249, 0.261, 0.272]), mode="energy_normalized"
        )
        self.assertEqual(guides, (0.225, 0.25, 0.275))
        self.assertNotEqual(guides, (0.20, 0.25, 0.30, 0.35))  # not the old fixed set

    def test_guides_adapt_to_a_high_pse_range(self):
        guides = compute_adaptive_pse_guides(self._pse_points([0.31, 0.34]), mode="energy_normalized")
        self.assertEqual(guides, (0.30, 0.325, 0.35))

    def test_narrow_single_point_range_produces_surrounding_guides(self):
        guides = compute_adaptive_pse_guides(self._pse_points([0.233]), mode="energy_normalized")
        self.assertGreaterEqual(len(guides), 3)
        self.assertLess(min(guides), 0.233)
        self.assertGreater(max(guides), 0.233)

    def test_guide_count_remains_restrained_across_ranges(self):
        for pse_values in ([0.233], [0.31, 0.34], [0.05, 0.95], [0.234, 0.249, 0.261, 0.272]):
            with self.subTest(pse_values=pse_values):
                guides = compute_adaptive_pse_guides(self._pse_points(pse_values), mode="energy_normalized")
                self.assertGreaterEqual(len(guides), 3)
                self.assertLessEqual(len(guides), 5)

    def test_generated_lines_are_labeled_with_the_adaptive_guide_values(self):
        points = self._pse_points([0.31, 0.34])
        guides = compute_adaptive_pse_guides(points, mode="energy_normalized")
        lines = build_iso_pse_lines(0.2, 1.2, guides, mode="energy_normalized")
        self.assertEqual({line["eta"] for line in lines}, set(guides))

    def test_unresolved_volumetric_fuel_type_produces_no_fabricated_guides(self):
        points = [{"x": 1.0, "y": 6.0}]
        self.assertEqual(compute_adaptive_pse_guides(points, mode="volumetric", fuel_type="Flex"), ())
        self.assertEqual(compute_adaptive_pse_guides(points, mode="volumetric", fuel_type=None), ())

    def test_no_computable_pse_produces_no_fabricated_guides(self):
        # x/y both present but non-positive -- never divide into a fake PSE.
        points = [{"x": 0.0, "y": 6.0}, {"x": 1.0, "y": 0.0}]
        self.assertEqual(compute_adaptive_pse_guides(points, mode="energy_normalized"), ())

    def test_volumetric_mode_still_uses_the_resolved_lhv_basis(self):
        # Tier 2 Cert Gasoline resolves to the canonical Gasoline LHV
        # (32.0 MJ/L) exactly as build_fe_vde_points/build_iso_pse_lines do
        # -- the adaptive sizing reuses that same resolution, not a new one.
        points = [{"x": 1.0, "y": 6.0}]
        guides = compute_adaptive_pse_guides(points, mode="volumetric", fuel_type="Tier 2 Cert Gasoline")
        self.assertTrue(guides)
        gasoline_guides = compute_adaptive_pse_guides(points, mode="volumetric", fuel_type="Gasoline")
        self.assertEqual(guides, gasoline_guides)  # same canonical family/LHV -> identical guides

    def test_electrical_mode_guides_use_the_electrical_pse_formula(self):
        points = [{"x": 1.0, "y": 1000.0}]
        guides = compute_adaptive_pse_guides(points, mode="electrical")
        self.assertTrue(guides)


class FormatValueTests(unittest.TestCase):
    def test_missing_value_is_dash(self):
        self.assertEqual(format_value(None, "mass_kg", "Metric"), "-")

    def test_text_family_passthrough(self):
        self.assertEqual(format_value("EPA", "text", "Metric"), "EPA")

    def test_mass_metric_and_us(self):
        self.assertIn("kg", format_value(1500.0, "mass_kg", "Metric"))
        self.assertIn("lb", format_value(1500.0, "mass_kg", "US customary"))

    def test_fuel_l_per_100km_converts_for_us_customary(self):
        metric_text = format_value(6.5, "l_per_100km", "Metric")
        us_text = format_value(6.5, "l_per_100km", "US customary")
        self.assertIn("L/100km", metric_text)
        self.assertIn("gal/100mi", us_text)

    def test_signed_delta_shows_plus_for_positive(self):
        self.assertTrue(format_value(0.05, "energy_mj_per_km", "Metric", signed=True).startswith("+"))
        self.assertTrue(format_value(-0.05, "energy_mj_per_km", "Metric", signed=True).startswith("-"))


# -----------------------------------------------------------------------------
# Explore Lite -- dimensions, metric selectors (Sec 52, Package 8D)
# -----------------------------------------------------------------------------


class ExploreDimensionAndMetricSelectorTests(unittest.TestCase):
    def test_x_dimensions_include_scenario_vehicle_and_curated_categoricals(self):
        keys = {d.key for d in list_explore_dimensions("x")}
        self.assertEqual(
            keys,
            {"scenario", "vehicle", "make", "model_year", "category", "legislation", "electrification", "fuel_type", "provenance"},
        )

    def test_order_dimension_is_only_model_year(self):
        keys = [d.key for d in list_explore_dimensions("order")]
        self.assertEqual(keys, ["model_year"])

    def test_group_and_filter_dimensions_exclude_scenario_vehicle_make_and_year(self):
        for role in ("group", "filter"):
            keys = {d.key for d in list_explore_dimensions(role)}
            self.assertEqual(keys, {"category", "legislation", "electrification", "fuel_type", "provenance"})

    def test_numeric_metrics_exclude_text_fields_and_use_registry_label(self):
        metrics = list_explore_numeric_metrics("bar")
        self.assertIn("mass_kg", [m.key for m in metrics])
        self.assertNotIn("make", [m.key for m in metrics])  # text metric never leaks into a numeric axis
        mass = next(m for m in metrics if m.key == "mass_kg")
        self.assertEqual(mass.label, "Mass")  # Registry label, not a raw column name

    def test_scatter_only_offers_scatter_compatible_metrics(self):
        scatter_keys = {m.key for m in list_explore_numeric_metrics("scatter")}
        bar_keys = {m.key for m in list_explore_numeric_metrics("bar")}
        self.assertIn("vde_total", scatter_keys)
        self.assertTrue(scatter_keys.issubset(bar_keys))  # every scatter metric is also bar-eligible today

    def test_line_metric_eligibility_matches_bar(self):
        self.assertEqual(
            {m.key for m in list_explore_numeric_metrics("line")}, {m.key for m in list_explore_numeric_metrics("bar")}
        )

    def test_metric_axis_label_includes_unit(self):
        from src.vde_core.comparison_metric_registry import get_metric

        self.assertEqual(metric_axis_label(get_metric("vde_total"), "Metric"), "VDE TOTAL [MJ/km]")


# -----------------------------------------------------------------------------
# Explore Lite -- Bar / Scatter / Line builders, group/filter, duplicate labels
# (Sec 53-56, 63, Package 8D). VDE_ONLY items are used directly where fuel/
# powertrain fields are not the point under test -- no fuelcons DB needed.
# -----------------------------------------------------------------------------


class ExploreVdeOnlyChartTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "explore_vde_only.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)
        self.reference = build_vde_comparison_item(900001, role=ComparisonRole.REFERENCE)
        self.other = build_vde_comparison_item(900002)
        self.no_mass_source = build_vde_comparison_item(900003)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def _dataset(self, comparisons):
        return ComparisonDataset(reference=self.reference, comparisons=tuple(comparisons))

    def test_bar_scenario_x_vde_total_correct_count_and_reference_preserved(self):
        dataset = self._dataset([self.other])
        result = build_explore_bar_rows(dataset, x_dimension_key="scenario", y_metric_key="vde_total")
        self.assertEqual(len(result["rows"]), 2)
        self.assertEqual(result["excluded"], [])
        self.assertIn("REFERENCE", {r.role for r in result["rows"]})

    def test_bar_missing_metric_excluded_with_reason(self):
        row = _qa_row(900002)
        row["vde_net_mj_per_km"] = None
        row["trans_A_coef_N"] = None
        row["trans_B_coef_Npkph"] = None
        row["trans_C_coef_Npkph2"] = None
        no_net_item = build_vde_comparison_item(900002, vde_row=row)
        dataset = self._dataset([no_net_item])
        result = build_explore_bar_rows(dataset, x_dimension_key="scenario", y_metric_key="vde_net")
        self.assertEqual(len(result["rows"]), 1)  # only Reference has NET
        self.assertEqual(len(result["excluded"]), 1)
        self.assertIn("unavailable", result["excluded"][0]["reason"])

    def test_bar_duplicate_scenario_display_titles_remain_distinct(self):
        row_a = _qa_row(900001)
        row_b = _qa_row(900002)
        row_b["make"], row_b["model"], row_b["year"] = row_a["make"], row_a["model"], row_a["year"]
        twin = build_vde_comparison_item(900002, vde_row=row_b)
        dataset = self._dataset([twin])
        result = build_explore_bar_rows(dataset, x_dimension_key="scenario", y_metric_key="vde_total")
        labels = [r.label for r in result["rows"]]
        self.assertEqual(len(labels), len(set(labels)))  # never merged despite identical source label
        self.assertEqual(len(result["rows"]), 2)

    def test_scatter_mass_vs_vde_total_valid_points(self):
        dataset = self._dataset([self.other])
        result = build_explore_scatter_points(dataset, x_metric_key="mass_kg", y_metric_key="vde_total")
        self.assertEqual(len(result["points"]), 2)
        self.assertEqual(result["excluded"], [])

    def test_scatter_missing_x_excluded_with_reason(self):
        dataset = self._dataset([self.other])
        result = build_explore_scatter_points(dataset, x_metric_key="fuel_l_per_100km", y_metric_key="vde_total")
        # VDE_ONLY items never populate fuel_energy -- X is unavailable for both.
        self.assertEqual(result["points"], [])
        self.assertEqual(len(result["excluded"]), 2)

    def test_scatter_missing_y_excluded_with_reason(self):
        dataset = self._dataset([self.other])
        result = build_explore_scatter_points(dataset, x_metric_key="mass_kg", y_metric_key="gco2_per_km")
        self.assertEqual(result["points"], [])
        self.assertEqual(len(result["excluded"]), 2)

    def test_scatter_scenario_identity_preserved_independently_of_label(self):
        row_b = _qa_row(900002)
        row_b["make"], row_b["model"], row_b["year"] = _qa_row(900001)["make"], _qa_row(900001)["model"], _qa_row(900001)["year"]
        twin = build_vde_comparison_item(900002, vde_row=row_b)
        dataset = self._dataset([twin])
        result = build_explore_scatter_points(dataset, x_metric_key="mass_kg", y_metric_key="vde_total")
        identities = {p.identity for p in result["points"]}
        labels = {p.label for p in result["points"]}
        self.assertEqual(len(identities), 2)  # distinct canonical identity...
        self.assertEqual(len(labels), 1)  # ...despite an identical display label

    def test_line_unordered_dimension_is_rejected(self):
        dataset = self._dataset([self.other])
        result = build_explore_line_rows(dataset, x_dimension_key="scenario", y_metric_key="vde_total")
        self.assertEqual(result["rows"], [])
        self.assertIsNotNone(result.get("unavailable_reason"))

    def test_line_model_year_ordering_is_deterministic_not_selection_order(self):
        row_2010 = _qa_row(900002)
        row_2010["year"] = 2010
        row_2015 = _qa_row(900003)
        row_2015["year"] = 2015
        item_2010 = build_vde_comparison_item(900002, vde_row=row_2010)
        item_2015 = build_vde_comparison_item(900003, vde_row=row_2015)
        # Reference year is 2026 (unmodified QA seed). Selection order below is
        # deliberately NOT chronological -- the builder must still sort by year.
        dataset = self._dataset([item_2015, item_2010])
        result = build_explore_line_rows(dataset, x_dimension_key="model_year", y_metric_key="vde_total")
        years = [r.x for r in result["rows"]]
        self.assertEqual(years, sorted(years))
        self.assertEqual(years, [2010, 2015, 2026])


# -----------------------------------------------------------------------------
# Explore Lite -- group/filter, temporary NET (Sec 56-57, Package 8D). These
# need FuelCons-linked items for Electrification/Fuel type/Provenance.
# -----------------------------------------------------------------------------


class ExploreGroupFilterTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "explore_group_filter.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)
        with sqlite3.connect(self.db_path) as con:
            rows = [
                (1, 900001, "ICE", "Gasoline", "HOMOLOGATED", 6.0, 150.0),
                (2, 900002, "PHEV", "Gasoline", "ESTIMATED", 4.0, 90.0),
                (3, 900003, "BEV", "Electric", "SCENARIO", None, 0.0),
                (4, 900006, "ICE", "Gasoline", "ESTIMATED", 7.0, 160.0),
            ]
            con.executemany(
                "INSERT INTO fuelcons_db (id, vde_id, electrification, fuel_type, record_origin, "
                "fuel_l_per_100km, gco2_per_km) VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            con.commit()
        self.reference = build_scenario_comparison_item(1, role=ComparisonRole.REFERENCE)
        self.phev = build_scenario_comparison_item(2)
        self.bev = build_scenario_comparison_item(3)
        self.ice_no_net = build_scenario_comparison_item(4)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def _dataset(self, comparisons):
        return ComparisonDataset(reference=self.reference, comparisons=tuple(comparisons))

    def test_group_by_electrification(self):
        dataset = self._dataset([self.phev, self.bev])
        result = build_explore_bar_rows(
            dataset, x_dimension_key="scenario", y_metric_key="mass_kg", group_dimension_key="electrification"
        )
        groups = {r.group for r in result["rows"]}
        self.assertEqual(groups, {"ICE", "PHEV", "BEV"})

    def test_group_by_provenance(self):
        dataset = self._dataset([self.phev, self.bev])
        result = build_explore_bar_rows(
            dataset, x_dimension_key="scenario", y_metric_key="mass_kg", group_dimension_key="provenance"
        )
        groups = {r.group for r in result["rows"]}
        self.assertEqual(groups, {"HOMOLOGATED", "ESTIMATED", "SCENARIO"})

    def test_filter_one_category_keeps_only_matching_items(self):
        dataset = self._dataset([self.phev, self.bev, self.ice_no_net])
        result = build_explore_bar_rows(
            dataset,
            x_dimension_key="scenario",
            y_metric_key="mass_kg",
            filter_dimension_key="electrification",
            filter_values=["ICE"],
        )
        self.assertEqual(len(result["rows"]), 2)  # reference + ice_no_net
        self.assertEqual({r.role for r in result["rows"]}, {"REFERENCE", "COMPARISON"})

    def test_filter_excludes_reference_when_it_does_not_match(self):
        dataset = self._dataset([self.phev])
        result = build_explore_bar_rows(
            dataset,
            x_dimension_key="scenario",
            y_metric_key="mass_kg",
            filter_dimension_key="electrification",
            filter_values=["PHEV"],
        )
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(result["rows"][0].role, "COMPARISON")
        self.assertIn(self.reference.label, [e["label"] for e in result["excluded"]])

    def test_empty_filter_result_handled_safely(self):
        dataset = self._dataset([self.phev, self.bev])
        result = build_explore_bar_rows(
            dataset,
            x_dimension_key="scenario",
            y_metric_key="mass_kg",
            filter_dimension_key="electrification",
            filter_values=["HEV"],  # no seeded item has this value
        )
        self.assertEqual(result["rows"], [])
        self.assertEqual(len(result["excluded"]), 3)

    def test_metric_unavailable_for_every_item_is_not_offered(self):
        dataset = self._dataset([self.bev])  # only Reference (ICE) + BEV, neither has energy_Wh_per_km set on Reference
        items = (dataset.reference, *dataset.comparisons)
        metrics = list_available_explore_metrics(items, "bar")
        self.assertNotIn("energy_wh_per_km", [m.key for m in metrics])

    def test_temporary_net_is_marked_on_scatter_points(self):
        row = _qa_row(900006)
        temp = {"source": "MANUAL", "A": 9.0, "B": 0.003, "C": 0.0006}
        temp_reference = build_vde_comparison_item(900006, role=ComparisonRole.REFERENCE, vde_row=row, temporary_transmission=temp)
        missing_comparison = build_vde_comparison_item(900001)
        dataset = ComparisonDataset(reference=temp_reference, comparisons=(missing_comparison,))
        result = build_explore_scatter_points(dataset, x_metric_key="mass_kg", y_metric_key="vde_net")
        temp_point = next(p for p in result["points"] if p.role == "REFERENCE")
        self.assertTrue(temp_point.is_temporary_net)


# -----------------------------------------------------------------------------
# Physical VDE Lineage -- context resolution, metric availability, waterfall
# (Sec 58-63, Package 8D Investigation Addendum)
# -----------------------------------------------------------------------------


class LineageContextTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "lineage_context.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)
        with sqlite3.connect(self.db_path) as con:
            con.executemany(
                "INSERT INTO fuelcons_db (id, vde_id, electrification, fuel_type, record_origin) VALUES (?, ?, ?, ?, ?)",
                [
                    (1, 900001, "ICE", "Gasoline", "HOMOLOGATED"),
                    (2, 900001, "ICE", "Gasoline", "ESTIMATED"),  # shares VDE 900001 with #1
                ],
            )
            con.commit()

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def test_bare_vde_selection_is_not_labeled_as_fuelcons_scenario(self):
        item = build_vde_comparison_item(900001)
        context = resolve_lineage_context(item)
        self.assertFalse(context.is_fuelcons_scenario)

    def test_fuelcons_scenario_selection_resolves_linked_vde_and_keeps_scenario_as_context_only(self):
        item = build_scenario_comparison_item(1)
        context = resolve_lineage_context(item)
        self.assertTrue(context.is_fuelcons_scenario)
        self.assertEqual([n.vde_id for n in context.chain.nodes], [900001])
        self.assertIn(item.label, context.originating_label)

    def test_two_fuelcons_scenarios_sharing_one_vde_produce_identical_physical_lineage(self):
        item1 = build_scenario_comparison_item(1)
        item2 = build_scenario_comparison_item(2)
        self.assertNotEqual(item1.fuelcons_id, item2.fuelcons_id)  # distinct scenarios elsewhere
        context1 = resolve_lineage_context(item1)
        context2 = resolve_lineage_context(item2)
        self.assertEqual(
            [n.vde_id for n in context1.chain.nodes], [n.vde_id for n in context2.chain.nodes]
        )  # same physical lineage
        self.assertEqual(context1.chain.status, context2.chain.status)

    def test_list_lineage_capable_metrics_excludes_fuel_energy_co2_pse(self):
        keys = {m.key for m in list_lineage_capable_metrics()}
        self.assertFalse(keys & {"fuel_l_per_100km", "fuel_km_per_l", "energy_wh_per_km", "gco2_per_km", "eta_pt_est"})
        self.assertIn("vde_total", keys)

    def test_list_available_lineage_metrics_requires_availability_at_every_node(self):
        with sqlite3.connect(self.db_path) as con:
            con.execute("UPDATE vde_db SET vde_id_parent=900001 WHERE id=900002")
            # vde_total is a persisted aggregate (canonical_vde_read), not derived
            # from coast_A/B/C at read time -- it must be nulled directly to make
            # the metric itself unavailable at this node.
            con.execute(
                "UPDATE vde_db SET vde_total_mj_per_km=NULL, vde_net_mj_per_km=NULL, "
                "coast_A_N=NULL, coast_B_N_per_kph=NULL, coast_C_N_per_kph2=NULL WHERE id=900002"
            )
            con.commit()
        chain = resolve_lineage_chain(900002)
        available_keys = {m.key for m in list_available_lineage_metrics(chain)}
        self.assertNotIn("vde_total", available_keys)  # unavailable at the 900002 node
        self.assertIn("mass_kg", available_keys)  # unaffected physical metric stays available


class LineageWaterfallTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "lineage_waterfall.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def _set_parent(self, vde_id: int, parent_id) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute("UPDATE vde_db SET vde_id_parent=? WHERE id=?", (parent_id, vde_id))
            con.commit()

    def test_root_only_chain_produces_single_baseline_step(self):
        chain = resolve_lineage_chain(900001)
        waterfall = build_lineage_waterfall(chain, "vde_total")
        self.assertEqual(len(waterfall.steps), 1)
        self.assertIsNone(waterfall.steps[0].delta)
        self.assertTrue(waterfall.complete)

    def test_child_step_is_child_minus_parent_and_lower_is_better_marks_better(self):
        self._set_parent(900002, 900001)  # 900002's vde_total (1.215) < 900001's (1.24)
        chain = resolve_lineage_chain(900002)
        waterfall = build_lineage_waterfall(chain, "vde_total")
        self.assertEqual(len(waterfall.steps), 2)
        child_step = waterfall.steps[1]
        self.assertAlmostEqual(child_step.delta, 1.215 - 1.24, places=6)
        self.assertEqual(child_step.semantic, "BETTER")

    def test_neutral_metric_has_delta_but_no_semantic_verdict(self):
        self._set_parent(900002, 900001)
        chain = resolve_lineage_chain(900002)
        waterfall = build_lineage_waterfall(chain, "mass_kg")
        child_step = waterfall.steps[1]
        self.assertIsNotNone(child_step.delta)
        self.assertIsNone(child_step.semantic)

    def test_missing_metric_mid_chain_is_incomplete_no_fallback(self):
        self._set_parent(900002, 900001)
        self._set_parent(900003, 900002)
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "UPDATE vde_db SET vde_total_mj_per_km=NULL, vde_net_mj_per_km=NULL, "
                "coast_A_N=NULL, coast_B_N_per_kph=NULL, coast_C_N_per_kph2=NULL WHERE id=900002"
            )
            con.commit()
        chain = resolve_lineage_chain(900003)
        waterfall = build_lineage_waterfall(chain, "vde_total")
        self.assertFalse(waterfall.complete)
        self.assertIsNotNone(waterfall.incomplete_reason)
        self.assertEqual(waterfall.steps[-1].status, "UNAVAILABLE")
        self.assertEqual(len(waterfall.steps), 2)  # walk stops at the broken node, never fabricates 900003's step

    def test_incompatible_cycle_blocks_cycle_specific_metric_but_not_physical_metric(self):
        self._set_parent(900002, 900001)
        with sqlite3.connect(self.db_path) as con:
            con.execute("UPDATE vde_db SET legislation='WLTP' WHERE id=900002")
            con.commit()
        chain = resolve_lineage_chain(900002)

        cycle_specific = build_lineage_waterfall(chain, "vde_total")
        self.assertFalse(cycle_specific.complete)
        self.assertEqual(cycle_specific.steps[-1].status, "INCOMPATIBLE")

        physical = build_lineage_waterfall(chain, "mass_kg")
        self.assertTrue(physical.complete)
        self.assertEqual(physical.steps[-1].status, "OK")

    def test_broken_reference_chain_is_reported_incomplete(self):
        self._set_parent(900002, 999999)
        chain = resolve_lineage_chain(900002)
        self.assertEqual(chain.status, LineageChainStatus.BROKEN)
        waterfall = build_lineage_waterfall(chain, "vde_total")
        self.assertFalse(waterfall.complete)
        self.assertEqual(len(waterfall.steps), 1)  # only the resolvable node


if __name__ == "__main__":
    unittest.main()
