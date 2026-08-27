# Sprint 10E - Integrated Quick Scenario UX + Comparison Insertion

## Status

Complete. `src/vde_core/vehicle_demand/` and `vde_request_resolver.py` were
not touched. No new physics/formula was introduced anywhere in this sprint
-- every numeric result a Quick Scenario produces still comes from the
Sprint 10B-10D canonical resolvers (`resolve_quick_vehicle_scenario`,
`resolve_quick_efficiency_scenario`), and every Comparison item a Quick
Scenario produces goes through the SAME `build_vde_comparison_item`/
`build_scenario_comparison_item` builders every real Comparison item goes
through.

## Closure traceability audit

The original Sprint 10E task arrived as a very long (58-section) spec, and
the literal 51 numbered test requirements / 7 lettered smoke cases did not
survive an earlier context compaction. Rather than leave that as an
unresolved caveat, a dedicated closure audit rebuilt traceability directly
from the feature's own requirement families (Source/Identity, Slot Model,
Stale/Recalculation, Vehicle Contract Mapping, Efficiency, Calculation,
Comparison) against the actual implementation and test suite. Two real
defects were found and fixed as part of this audit (not merely gaps in
test coverage):

1. **Sentinel-id collision across FuelCons sources sharing one VDE.**
   `quick_slot_sentinel_id` was keyed only by the resolved `vde_id`, so two
   distinct sources linked to the same VDE (e.g. the QA fixture's
   `fc:900102`/`fc:900104`, both `vde_id=900001`) would compute the
   IDENTICAL sentinel for their own slot 1, silently conflating two
   genuinely distinct Quick Scenarios into one `ComparisonItem` identity.
   Fixed by keying the sentinel off the full `source_identity` (kind +
   record id) instead. Covered by
   `test_distinct_across_fuelcons_sources_sharing_one_vde`,
   `test_distinct_between_fc_and_vde_kind_for_the_same_numeric_id`, and
   `test_two_fuelcons_sources_sharing_one_vde_produce_distinct_quick_items`
   (`tests/test_quick_scenario_comparison_adapter.py`).
2. **Stale/"Needs recalculation" badge lagged one render behind an edit.**
   `render_quick_scenario_tab` computed each slot's
   `QuickSlotCalculationState` from the PRE-render `QuickScenario`, before
   that render's widgets had a chance to read the user's just-changed
   value back into an updated scenario -- so editing a value and rerunning
   still showed the OLD state (e.g. "Ready") for one extra interaction
   instead of immediately flipping to "Needs recalculation". Fixed by
   moving the state computation to AFTER the domain widgets rebuild
   `updated`, using an `st.empty()` placeholder to keep the badge visually
   in the header. Verified with a before/after test:
   `test_editing_after_calculate_marks_stale_then_recalculate_keeps_identity`
   (`tests/test_comparison_quick_scenario_ui.py`) fails against the
   pre-fix code and passes after.

A third hypothesis -- that the "Accept as Final PSE" button's side-channel
session key couldn't actually override an already-instantiated
`number_input`/`checkbox` (a documented Streamlit widget-state precedence
rule) -- was investigated and given a defensive fix (explicitly seeding
`session_state` before those widgets are instantiated), but a dedicated
regression test (`test_accepting_a_benchmark_reference_sets_final_pse_and_provenance`)
was verified to pass identically before and after that change. It is kept
as the more explicit, standard-conforming pattern, but is **not** claimed
as a confirmed bug fix -- the hypothesized failure did not reproduce.

The full requirement-family traceability matrix is in the "Requirement
traceability" section near the end of this document.

## Scope

Wires the existing, Streamlit-free Quick Scenario core (Sprints 10A-10D)
into the live Comparison Report Streamlit page
(`src/vde_app/components/comparison_report.py`). An engineer can now:

- pick one already-resolved Comparison item (Reference or a Compare-with
  item) as a source;
- define up to 3 temporary Mass / Tire / Aero / Efficiency ("Quick")
  variants of it, using only existing contract enums (no free-text physics
  input);
- calculate all of a source's defined slots with one "Calculate Quick
  Scenarios" button (no per-domain Apply buttons);
- see each successfully-resolved slot appear as an ordinary row in the same
  Comparison dataset Program Review / Energy Drivers / Technical Scorecard
  / Explore already render.

Nothing here is persisted: Quick Scenarios exist only in Streamlit
`session_state` for the current browser session and are never written to
`vde_db`/`fuelcons_db`. There is no Save/Promote action in this sprint.

## Pre-flight (Section 3): Benchmark PSE ownership centralization

Before the main UI work, `pwt_fuel_energy.py`'s benchmark-PSE flow
(`_derive_reference_pse`, `_fuel_type_from_reference_row`, `_load_json_blob`,
and the "Another fuelcons_db line" branch of `_reference_candidates_for_type`)
was found still duplicating the Sprint 10D extractions in
`pwt_fuel_energy_service.py` (`derive_reference_pse`,
`resolve_reference_fuel_type`, `load_json_blob`,
`list_benchmark_fuelcons_candidates`) -- a known backlog item flagged at the
end of Sprint 10D. This was resolved as a separate housekeeping commit
(`refactor(powertrain): centralize benchmark PSE references`) before the
main Sprint 10E work began: `pwt_fuel_energy.py` now imports these 4
functions directly instead of holding local copies; the two genuinely
UI-specific `_reference_candidates_for_type` branches ("Same vehicle
fuelcons_db line", "Saved powertrain scenario") were left untouched. Full
suite after that refactor: 1608 tests (1605 baseline + 3 new shared-
ownership tests), the same 2 known pre-existing `vde_request_resolver`
failures, zero new regressions.

## Decisions confirmed with the user before implementation

**Decision 1 -- UI placement.** The Quick Scenario editor is a new
top-level `st.tabs` entry, `"Quick Scenarios"`, alongside Program Review /
Energy Drivers / Technical Scorecard / Explore -- not an expander above the
tabs, not a sub-tab of Explore.

**Decision 2 -- Provenance marking.** A Quick item is marked non-persisted
by reusing the existing free-text `ComparisonProvenance.record_origin`
field with a new sentinel value, `"QUICK_SCENARIO"` (alongside the existing
real values `HOMOLOGATED`/`MEASURED`/`ESTIMATED`/`POWERTRAIN_L0`/`LEGACY`)
-- not by adding a new field to the `ComparisonItem`/`ComparisonProvenance`
dataclasses. This keeps the sprint's blast radius on the widely-used
`ComparisonItem` contract at zero.

## Files changed

- `src/vde_core/quick_scenario/resolution.py` -- added one additive field,
  `QuickVehicleResolution.resolved_vde_row: Mapping[str, Any] | None`,
  populated only when `readiness.all_ready`.
- `src/vde_core/quick_scenario/resolver.py` -- `resolve_quick_vehicle_scenario`
  now passes its existing internal `synthetic_row` into that new field
  instead of discarding it; added `fetch_quick_source_rows(source_identity)`,
  a thin public wrapper around the module's own existing
  `_parse_source_identity`/`_fetch_source_vde_row` logic that also fetches
  the linked `fuelcons_db` row for a `fc:`-kind identity, so a caller
  resolving several sibling slots for one source fetches its rows once.
- `src/vde_core/quick_scenario/contracts.py` -- renamed the module-private
  `_ALLOWED_TIRE_TRANSFORMS_BY_SOURCE` to the public
  `ALLOWED_TIRE_TRANSFORMS_BY_SOURCE` (no other change) so the UI can read
  the same source/transform-mode allow-list `TireQuickChange.__post_init__`
  already validates against, instead of maintaining a second copy.
- `src/vde_core/quick_scenario/comparison_adapter.py` (new) -- the only new
  "physics-adjacent" module this sprint, and it contains no physics:
  - `quick_slot_sentinel_id(source_vde_id, slot)` -- deterministic negative
    id, unique per (source_identity, slot) -- keyed off the full parsed
    `source_identity` (kind + record id), not merely the resolved `vde_id`
    (see "Closure traceability audit" above for why: two distinct
    FuelCons sources sharing one VDE must not collide), used as the Quick
    item's `vde_id`/`fuelcons_id` so it can never collide with a real
    (positive) database id or be conflated with its own real source by
    `deduplicate_by_vde_id`.
  - `resolve_quick_slot(quick_scenario, source_vde_row=, source_fuelcons_row=)`
    -- one-slot orchestration: calls `resolve_quick_vehicle_scenario`, then
    (only if ready) `resolve_quick_efficiency_scenario`. Each slot is
    resolved independently; neither canonical resolver raises for a
    MISSING/INVALID domain, so one bad slot never blocks its siblings.
  - `build_quick_comparison_item(...)` -- returns `None` when the Vehicle
    layer isn't ready (a Vehicle-unresolved Quick Scenario is never
    inserted); otherwise stamps the sentinel id and `record_origin` onto a
    copy of `resolved_vde_row` and calls `build_vde_comparison_item`
    directly (Vehicle-ready, Efficiency-unavailable -- a VDE_ONLY-shaped
    item with `fuel_energy=None`) or, when Efficiency also resolved,
    builds a synthetic `fuelcons_row` (renaming `FuelEstimateResult`'s
    `fuel_l_100km`/`energy_Wh_km`/`gco2_km` to the `fuelcons_db` column
    names `fuel_l_per_100km`/`energy_Wh_per_km`/`gco2_per_km`, with
    `source_vde_revision=None` so revision comparison reports `MISSING`
    rather than a spurious `STALE`) and calls `build_scenario_comparison_item`
    (a full item). Neither builder is reimplemented -- both are the
    existing `comparison_report_service.py` functions.
  - `QuickSlotCalculationState` (`NOT_CALCULATED`/`READY`/
    `NEEDS_RECALCULATION`/`MISSING_OR_INVALID`) and
    `derive_quick_slot_calculation_state(...)` -- a pure, Streamlit-free
    function deriving the 4-state badge from dataclass equality between the
    live and last-calculated `QuickScenario` plus the last resolutions'
    readiness.
  - `merge_quick_items_into_dataset(dataset, quick_items)` -- appends to
    `dataset.comparisons`; `dataset.reference` is never touched (a Quick
    item is always inserted with `ComparisonRole.COMPARISON`).
- `src/vde_app/components/comparison_quick_scenario_tab.py` (new) -- all
  new Streamlit code for this sprint (kept out of the already-1873-line
  `comparison_report.py`, mirroring how `comparison_report_charts.py` is
  already separate). Public entry point:
  `render_quick_scenario_tab(dataset) -> ComparisonDataset`. Session-state
  keys follow the existing `comparison_<concept>` convention
  (`comparison_quick_active_source`, `comparison_quick_scenarios`,
  `comparison_quick_last_calculated`, `comparison_quick_results`), nested
  `dict[source_identity, dict[slot, ...]]`, mirroring the existing
  `_TEMP_TRANSMISSION_KEY` "temporary variant keyed by source id" pattern.
  Widgets map onto contract enums only (`ScalarChangeMode`, `MassQuickChange`,
  `TireSource`/`TireTransformMode` restricted per
  `ALLOWED_TIRE_TRANSFORMS_BY_SOURCE`, `EfficiencyQuickInputs`); the Tire DB
  picker uses the existing `tire_roadload_service.get_available_tires`
  (never a raw query); the Benchmark PSE picker uses the existing
  `list_benchmark_fuelcons_candidates`; the Technology Delta picker uses the
  existing `load_quick_tech_delta_catalog` presets. Efficiency stays
  strictly advisory: Current/Benchmark/ML/Tech-Delta render as read-only
  reference cards with their own "Accept as Final PSE" action (setting
  `final_pse_percent`/the matching `*_ACCEPTED` provenance); manually typing
  a different Final PSE value always resolves to `PseProvenance.USER_PROVIDED`,
  even if it happens to numerically match a previously-accepted value (Sec
  10's "editing an accepted value un-accepts it").
- `src/vde_app/components/comparison_report.py` -- added the
  `"Quick Scenarios"` tab (5th, after Explore) to the existing `st.tabs(...)`
  call; the merged dataset `render_quick_scenario_tab` returns is what the
  other four tab renderers now consume. No other tab function's internals
  changed.
- `src/vde_core/quick_scenario/__init__.py` -- re-exports the new
  `comparison_adapter` functions/enum and `ALLOWED_TIRE_TRANSFORMS_BY_SOURCE`.

## Vehicle-ready / Efficiency-unavailable partial insertion

A Quick Scenario with a resolved Vehicle layer but no adopted Final PSE (or
an unresolved Efficiency layer) is still inserted -- as a `SourceKind.VDE_ONLY`
item with `fuel_energy=None`, exactly like any other VDE-only Comparison
item today. A Quick Scenario whose Vehicle layer never resolved is not
inserted at all (`build_quick_comparison_item` returns `None`) -- this is
Sprint 10C's own "no silent partial calc" rule surfacing one layer up, not
a new rule invented here.

## TOTAL / NET

Never hand-rolled: because Quick items are built by handing a row to the
existing `build_vde_comparison_item`/`build_scenario_comparison_item`
(which themselves call `resolve_roadload_boundaries`/
`resolve_cycle_vde_results`/`resolve_vde_aggregate`), every Quick item's
`roadload`/`vde` fields carry the identical TOTAL/NET structure and
availability semantics every real item already has, with no separate
TOTAL/NET logic added anywhere in this sprint.

## Tests

- `tests/test_quick_scenario_comparison_adapter.py` (20 tests,
  Streamlit-free): sentinel id determinism/uniqueness, including the two
  closure-audit collision regressions (distinct across FuelCons sources
  sharing one VDE, distinct between `fc:`/`vde:` kinds, label does not
  affect identity); Vehicle-unresolved -> no item; Vehicle-ready/
  Efficiency-not-requested -> VDE_ONLY-shaped item; Vehicle-ready/
  Efficiency-ready -> full scenario-shaped item with correctly-renamed
  fuel/energy fields; TOTAL and NET both present; multi-slot
  partial-failure isolation; recalculation keeps the same identity and
  never duplicates a slot; dataset merge preserves Reference and existing
  Comparisons untouched; the 4 `QuickSlotCalculationState` branches.
- `tests/test_comparison_quick_scenario_ui.py` (9 tests, `AppTest` -- this
  codebase's established manual-smoke substitute per Sprint 9 precedent):
  tab renders with no slots; adding a slot renders its editor; a neutral
  (0%) Aero change calculates to a ready slot and its label appears in a
  rendered table; Reset clears the active source's slots; accepting a
  Benchmark PSE reference sets Final PSE/provenance and the visible widget
  (the closure-audit end-to-end check); a Mass + Tire combination together
  resolves and calculates (smoke-matrix Case B); three sibling Quick
  Scenarios get stable, distinct identities and the 4th slot is refused
  (Case F); editing an input after Calculate shows "Needs recalculation"
  and recalculating keeps the same identity (Case G, and the regression
  test for the stale-badge-lag bug below); switching the active editing
  source preserves a previously-calculated Quick Scenario on another
  source.
- Existing quick_scenario suites (192 tests: contracts, resolver, resolver
  parity, vehicle-demand integration, tire resolution, efficiency resolver)
  and existing Comparison Report suites (163 tests: page smoke, service,
  vehicle-demand smoke, 8E smoke matrix) all still pass unchanged.
- Full suite: 1637 tests (1627 post-integration baseline + 10 new from this
  closure audit: the adapter test file grew from 14 to 20 tests, the UI
  smoke file from 5 to 9), the same 2 known pre-existing
  `vde_request_resolver` failures, zero new regressions.

## Backlog / deferred (not addressed in 10E)

- Save/Promote of a Quick Scenario into a persisted `fuelcons_db`/`vde_db`
  row -- explicitly out of scope for this sprint.
- Promoting a Quick item to the Reference role -- v1 only inserts Quick
  items as `ComparisonRole.COMPARISON`.
- Smoke-matrix Cases D (ML recommendation adoption) and E (Tire DB +
  Pressure provenance) were not given dedicated `AppTest` coverage -- both
  are already thoroughly covered at the resolver level (see the
  traceability matrix below) and adding a UI-level duplicate was judged to
  not be worth the added test surface, per the closure audit's explicit
  "traceability, not test-count matching" goal.
- No real interactive (browser-driven) manual smoke test was possible in
  this environment -- see "Manual smoke test result" below for why, and
  for exactly what `AppTest` did and did not substitute for.

## Requirement traceability

Rebuilt directly from the feature's own requirement families (the literal
51/7-item original checklist did not survive an earlier context
compaction -- see "Closure traceability audit" above). Status legend:
**PASS** (direct automated test), **PASS (indirect)** (covered by an
earlier Sprint 10A-10D core test, not written for 10E specifically),
**PASS (inspection)** (verified by direct code reading, not a runnable
test), **MANUAL** (would need a real browser; strong indirect PASS exists
at the resolver level), **FIXED** (a real defect found by this audit and
corrected, now covered).

### A. Source / Identity

| Requirement | Status | Test / evidence |
|---|---|---|
| Explicit Quick Source | PASS (inspection) | `render_quick_scenario_tab`'s source `st.selectbox` |
| Canonical scenario identity | PASS (indirect) | `test_identity_format`, `test_identity_property_matches_helper` |
| Scenario identity != vde_id | PASS (indirect) | `test_scenario_identity_preserves_full_comparison_identity_not_only_vde_id` |
| Two FuelCons scenarios sharing one VDE remain distinct (contract level) | PASS (indirect) | `test_scenario_identity_preserves_full_comparison_identity_not_only_vde_id` |
| Two FuelCons scenarios sharing one VDE remain distinct (Comparison-item level) | **FIXED** | `test_distinct_across_fuelcons_sources_sharing_one_vde`, `test_two_fuelcons_sources_sharing_one_vde_produce_distinct_quick_items` |
| Source immutable | PASS (indirect) | `test_source_fuelcons_row_remains_unchanged`, resolver source-immutability tests |
| Source change cannot leave stale results masquerading under new source | PASS | `test_switching_active_source_preserves_other_sources_quick_scenarios` + nested-dict-by-source_identity architecture |

### B. Slot Model

| Requirement | Status | Test / evidence |
|---|---|---|
| 1 to 3 Quick Scenarios | PASS (indirect) | `test_slot_zero_is_rejected`, `test_slot_above_max_is_rejected` |
| >3 prevented (UI) | PASS | `test_three_sibling_quick_scenarios_get_stable_distinct_identities` (4th `+ Add` is disabled) |
| All slots share the same selected source | PASS (inspection) | slots are only ever constructed with `source_identity=active` |
| Slots are independent siblings | PASS | `test_partial_slot_failure_does_not_block_sibling_slots` |
| No Quick -> Quick lineage | PASS (indirect + UI) | `test_no_quick_to_quick_lineage`; `test_three_sibling_quick_scenarios_get_stable_distinct_identities` |
| User label does not define identity | **PASS (new)** | `test_user_label_does_not_affect_identity` |
| Stable identity per source + slot | PASS | `test_negative_and_deterministic`, `test_distinct_across_slots_and_sources` |
| Reset is safe | PASS | `test_reset_clears_all_quick_scenario_state` |

### C. Stale / Recalculation

| Requirement | Status | Test / evidence |
|---|---|---|
| Editing calculated inputs marks stale | **FIXED** | `test_needs_recalculation_when_inputs_diverge`; `test_editing_after_calculate_marks_stale_then_recalculate_keeps_identity` |
| Old result not presented as matching new inputs | **FIXED** | same as above (badge-lag bug) |
| Recalculation replaces the same temporary identity | PASS (new) | `test_recalculation_with_changed_inputs_keeps_the_same_identity` |
| Recalculation does not append duplicates | PASS (new) | `test_rebuilding_quick_items_from_a_results_dict_never_duplicates_a_slot` |

### D. Vehicle Contract Mapping

| Requirement | Status | Test / evidence |
|---|---|---|
| Mass Absolute / Delta / Percent | PASS (indirect) | `test_quick_scenario_resolver.py` curb-change tests |
| EPA regulatory mass control | PASS (indirect) | `test_twc_shift_up/down`, `test_curb_change_crossing_a_twc_bracket_changes_twc` |
| Aero Absolute / Delta / Percent | PASS (indirect) | `test_absolute_cda_change`, `test_delta_cda_change`, `test_percent_cda_change` |
| Current Tire: Target RRC / RRC Delta / Improvement / Pressure | PASS (indirect) | `test_current_target_rrc_parity_*`, `test_current_rrc_delta_delegates_result_to_target_resolver`, `test_current_improvement_parity`, `test_current_pressure_delta_parity` |
| Tire DB: None / Improvement / Pressure | PASS (indirect) | `test_tire_db_none_parity`, `test_tire_db_improvement_uses_db_result_as_canonical_source`, `test_tire_db_iso/sae_pressure_*` |
| Forbidden Tire combinations rejected | PASS (indirect) | `TireQuickChangeValidationTests` (contracts) |
| Forbidden Tire combinations not exposed in UI | PASS (inspection) | UI selectbox options are filtered through the same `ALLOWED_TIRE_TRANSFORMS_BY_SOURCE` the contract validates against |
| User-provided tire reference-pressure provenance | PASS (indirect) | `test_user_reference_pressure_resolves_and_preserves_provenance` |
| No physical formulas in UI | PASS (inspection) | `comparison_quick_scenario_tab.py` contains no numeric derivation, only contract object construction |
| Mass + Tire together resolve via UI | PASS (new) | `test_mass_and_tire_change_together_resolve_and_calculate` (Case B) |

### E. Efficiency

| Requirement | Status | Test / evidence |
|---|---|---|
| Current PSE display/reference | PASS (indirect) | `test_current_pse_resolves_through_canonical_path` |
| Benchmark candidate selection uses canonical service | PASS (indirect) | `test_benchmark_pse_from_another_fuelcons_row`; `list_benchmark_fuelcons_candidates` tests |
| Benchmark adoption provenance | PASS (new, end-to-end) | `test_accepting_a_benchmark_reference_sets_final_pse_and_provenance` |
| ML recommendation | MANUAL (strong indirect) | `test_ml_artifact_loads_through_canonical_path_with_injected_predictor` |
| ML adoption provenance | MANUAL (strong indirect) | `test_accept_ml_recommendation_final_pse_gets_ml_accepted_provenance` |
| Technology Delta suggestion | PASS (indirect) | multiple tests in `test_quick_scenario_efficiency_resolver.py` |
| Tech adoption provenance | MANUAL (strong indirect) | `test_accept_tech_suggestion_gets_tech_delta_accepted_provenance` |
| Manual Final PSE | PASS (indirect) | `test_manual_final_pse_produces_deterministic_result` |
| Manual edit after adoption becomes USER_PROVIDED | PASS (new, end-to-end) | UI logic in `_render_efficiency_section`, exercised by `test_accepting_a_benchmark_reference_sets_final_pse_and_provenance` |
| PSE zero remains explicit/invalid | PASS (indirect) | `test_final_pse_zero_is_explicit_and_invalid` |
| Unadopted recommendations do not alter deterministic result | PASS (indirect) | `test_without_explicit_adoption_final_pse_and_energy_result_do_not_change` |

### F. Calculation

| Requirement | Status | Test / evidence |
|---|---|---|
| One Calculate action | PASS (inspection) | only `comparison_quick_calculate` triggers a resolver call; grep-confirmed no per-domain Apply buttons |
| Each slot resolves independently | PASS | `test_partial_slot_failure_does_not_block_sibling_slots` |
| One bad slot does not block other valid slots | PASS | same |
| Requested Vehicle domain failure blocks that slot | PASS (indirect) | `test_mass_ready_aero_missing_leaves_whole_scenario_unresolved` |
| Vehicle READY + Efficiency unavailable is a valid Quick item | PASS (new) | `test_vehicle_ready_efficiency_not_requested_is_vde_only_shaped` |
| No DB writes / no persistence | PASS (indirect + inspection) | `test_no_db_writes`; no `save_*`/`insert`/`update` call anywhere in the Quick Scenario or adapter modules |

### G. Comparison

| Requirement | Status | Test / evidence |
|---|---|---|
| Ready Quick result becomes ordinary ComparisonItem | PASS (new) | `test_vehicle_and_efficiency_ready_is_full_scenario_shaped` |
| Existing selected items / Reference preserved | PASS (new) | `test_merge_preserves_reference_and_existing_comparisons` |
| Quick provenance visible through record_origin | PASS (new) | both shaped-item tests assert `record_origin == "QUICK_SCENARIO"` |
| Stable Quick Comparison identity | PASS (new) | sentinel determinism tests |
| Recalculate replaces instead of duplicates | PASS (new) | `test_rebuilding_quick_items_from_a_results_dict_never_duplicates_a_slot` |
| Quick siblings remain distinct | **FIXED** | `test_two_fuelcons_sources_sharing_one_vde_produce_distinct_quick_items`, `test_three_sibling_quick_scenarios_get_stable_distinct_identities` |
| Comparison viewmodels do not recalculate physics | PASS (inspection) | `comparison_report_viewmodels.py`/`comparison_report_charts.py` have zero diff this sprint |
| Program Review / Energy Drivers / Explore accept Quick | PASS (inspection) | none of the three tab renderers filter on `record_origin`/`source_kind` (grep-confirmed); all consume the same merged `dataset` |
| Technical Scorecard accepts Quick | PASS (new, direct) | `test_calculate_a_neutral_change_produces_a_ready_slot_and_inserts_into_scorecard` |
| TOTAL/NET semantics preserved | PASS (new) | `test_total_and_net_are_both_present_and_independent` |

## Manual smoke test result

This environment is a headless CLI sandbox with no browser or display --
a genuine interactive manual smoke test (opening the page, clicking
through it visually) could not be performed, and is not represented as
having been done. `AppTest` (Streamlit's own script-level test harness) was
used instead, exactly as labeled throughout this document and its test
files -- it is **not** presented as a substitute for the manual smoke
cases as originally specified, only as this codebase's established
next-best verification (the same substitution Sprint 9's own closure used).
Of the 7 originally-named smoke cases: **A, B, C, F, G** have direct
`AppTest` coverage (see the file list above); **D** and **E** were judged
adequately covered by existing resolver-level tests and were not
duplicated at the UI level (see Backlog).

## Freeze / handoff statement

`src/vde_core/vehicle_demand/` and `vde_request_resolver.py` are untouched.
No new Comparison engine was introduced -- Quick items flow through the
existing `build_comparison_dataset` output via a pure, additive merge, and
every existing tab (Program Review, Energy Drivers, Technical Scorecard,
Explore) renders them with zero code changes of their own. Do not start
Sprint 10F.
