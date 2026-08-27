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

## Note on the originating spec

The original Sprint 10E task arrived as a very long (58-section) spec in
this session. Partway through Sprint 10E's pre-flight work, the
conversation was summarized (context compaction), and the literal spec
text -- including the 51 numbered automated-test requirements and the 7
lettered manual-smoke cases (A-G) -- did not survive verbatim, only as a
compressed paraphrase. Two placement/design questions the paraphrase left
ambiguous were confirmed directly with the user before implementation (see
Decision 1/2 below); everywhere else, this sprint was implemented from the
feature's own logical requirements as captured in the paraphrase, not from
the lost literal section numbers. This is stated plainly here rather than
presented as a 1:1 match to the original checklist.

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
    id (`-(abs(source_vde_id) * 10 + slot)`), unique per (source, slot),
    used as the Quick item's `vde_id`/`fuelcons_id` so it can never collide
    with a real (positive) database id or be conflated with its own real
    source by `deduplicate_by_vde_id`.
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

- `tests/test_quick_scenario_comparison_adapter.py` (new, 14 tests,
  Streamlit-free): sentinel id determinism/uniqueness; Vehicle-unresolved
  -> no item; Vehicle-ready/Efficiency-not-requested -> VDE_ONLY-shaped
  item; Vehicle-ready/Efficiency-ready -> full scenario-shaped item with
  correctly-renamed fuel/energy fields; TOTAL and NET both present;
  multi-slot partial-failure isolation (one bad Tire request in slot 2 does
  not block slots 1/3); dataset merge preserves Reference and existing
  Comparisons untouched; the 4 `QuickSlotCalculationState` branches.
- `tests/test_comparison_quick_scenario_ui.py` (new, 5 tests, `AppTest` --
  this codebase's established manual-smoke substitute per Sprint 9
  precedent): tab renders with no slots; adding a slot renders its editor;
  a neutral (0%) Aero change calculates to a ready slot and its label
  appears in a rendered table (proving it flowed into the SAME dataset
  every tab already renders, not a parallel display); Reset clears the
  active source's slots; switching the active editing source preserves a
  previously-calculated Quick Scenario on another source.
- Existing quick_scenario suites (192 tests: contracts, resolver, resolver
  parity, vehicle-demand integration, tire resolution, efficiency resolver)
  and existing Comparison Report suites (163 tests: page smoke, service,
  vehicle-demand smoke, 8E smoke matrix) all still pass unchanged.
- Full suite: see the run recorded in this sprint's commit message (below)
  for the exact final count; the same 2 known pre-existing
  `vde_request_resolver` failures, zero new regressions.

## Backlog / deferred (not addressed in 10E)

- Save/Promote of a Quick Scenario into a persisted `fuelcons_db`/`vde_db`
  row -- explicitly out of scope for this sprint.
- Promoting a Quick item to the Reference role -- v1 only inserts Quick
  items as `ComparisonRole.COMPARISON`.
- A literal reconciliation against the original spec's 51 numbered test
  requirements / 7 lettered smoke cases was not possible (see "Note on the
  originating spec" above); if the user has the original spec text
  available, a follow-up pass could diff this sprint's actual coverage
  against it.

## Freeze / handoff statement

`src/vde_core/vehicle_demand/` and `vde_request_resolver.py` are untouched.
No new Comparison engine was introduced -- Quick items flow through the
existing `build_comparison_dataset` output via a pure, additive merge, and
every existing tab (Program Review, Energy Drivers, Technical Scorecard,
Explore) renders them with zero code changes of their own. Do not start
Sprint 10F.
