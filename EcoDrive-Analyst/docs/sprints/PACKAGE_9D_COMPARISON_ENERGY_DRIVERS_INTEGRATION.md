# Sprint 9D - Comparison / Energy Drivers Integration

## Status

Completed. Integrates the frozen Vehicle Demand Core (Sprint 9A-9C) into the
Comparison Report's Energy Drivers tab as a compact "Vehicle Demand
Summary" table plus one optional breakdown chart. Comparison remains a pure
consumer -- no roadload/RRC/CdA/air-density/inertia/tractive/energy-
integration physics was written in this package.

## Baseline

Branch `sprint-9a-vehicle-demand-contracts` (Sprint 9C commits `d35b994a`/
`9f474ac5`). Baseline: 1238 tests, 1236 passing, 2 known pre-existing
failures in `test_vde_request_resolver.py`, untouched by this package.

## Investigation before implementation

Per Sec 6, the existing Energy Drivers tab (`_render_energy_drivers_tab` in
`src/vde_app/components/comparison_report.py`) was read end-to-end before
any change:

- Its storytelling order is Physical Setup -> Roadload force curve ->
  Roadload ABC -> VDE by Cycle/Phase -> Demanded power (opt-in). Each
  section follows the same shape: a pure builder in
  `comparison_report_viewmodels.py`/`comparison_report_charts.py` returns
  already-formatted rows, a thin render function calls
  `_render_exclusions`/`_render_section`/`st.plotly_chart` on them. No
  physics is computed inside any render function.
- `ScorecardCell`/`ScorecardRow`/`ScorecardSection` (+ `_render_section`,
  the shared dataframe/style renderer) are generic presentation containers,
  not inherently coupled to the Metric Registry -- they are simply
  *usually* constructed via `_metric_row`, which is Registry-specific. This
  meant a Vehicle Demand table could reuse `_render_section` unchanged by
  constructing `ScorecardCell`/`ScorecardRow` directly, without adding
  anything to `comparison_metric_registry.py`.
- `ComparisonItem` already carries pre-resolved TOTAL/NET roadload
  (`item.roadload["total"/"net"]`, built by
  `comparison_report_service.resolve_roadload_boundaries`) and mass/RRC/
  CdA/legislation (`item.vehicle`). This is a *more* resolved shape than
  the raw `vde_db` row the Sprint 9C adapter
  (`vehicle_demand.adapters.build_vehicle_demand_request`) expects, and
  reusing it avoids a second DB fetch per scenario.
- `dataset_items`/`build_scenario_header`/`_dedupe_titles` already
  implement "Reference first, optional", role/provenance badges, and
  duplicate-label handling identically to every other section -- nothing
  new was needed for Reference/Proposal/Benchmark/Current presentation
  (Sec 7); `PresentationRole`/`scenario_intent` are a session-only overlay
  the existing header builder already renders, and Vehicle Demand never
  touches or infers them.
- `compare_metric`'s exact delta formula (`absolute_delta = cmp - ref`,
  `percent_delta = ((cmp/ref)-1)*100 if ref != 0`) and its
  `SAME_LEGISLATION_CYCLE` compatibility rule (`reference.vehicle
  ["legislation"] == item.vehicle["legislation"]`) were read and mirrored
  exactly for Vehicle Demand's own delta computation (Sec 30: no new delta
  logic), since these aren't Registry metrics and `compare_metric` itself
  can't be called for them.

## Files created/modified

Created:
- `src/vde_app/comparison_vehicle_demand_viewmodels.py` -- the canonical
  presentation builder (`get_vehicle_demand_result`,
  `build_vehicle_demand_comparison_rows`,
  `build_vehicle_demand_breakdown_rows`).
- `tests/test_comparison_vehicle_demand_viewmodels.py` -- 19 tests.
- `tests/test_comparison_report_vehicle_demand_smoke.py` -- 5 AppTest smoke
  tests (Sec 53, Smoke A-E).
- `docs/sprints/PACKAGE_9D_COMPARISON_ENERGY_DRIVERS_INTEGRATION.md` (this
  file).

Modified:
- `src/vde_app/components/comparison_report.py` -- new
  `_render_vehicle_demand_summary_section`, wired into
  `_render_energy_drivers_tab` after VDE by Cycle/Phase and before Demanded
  power; new imports; one behavior-preserving tweak to `_render_section`
  (see below).
- `src/vde_app/comparison_report_charts.py` -- new
  `build_vehicle_demand_breakdown_chart` (the one new visual, Sec 16-18).
- `src/vde_app/comparison_report_viewmodels.py` -- one new
  `_UNIT_QUANTITY_MAP` entry (`"energy_mj": "energy_mj"`).
- `src/vde_app/units.py` -- one new `QuantitySpec` (`"energy_mj"`) for
  absolute cycle energy, distinct from the existing `energy_per_distance`
  (MJ/km) rate quantity.

**No file in `src/vde_core/vehicle_demand/` (contracts, engine, physics,
adapters) was touched.** No `comparison_report_service.py` change either --
the integration reads its existing, already-public
`resolve_roadload_boundaries`/`ComparisonItem`/`ComparisonDataset` surface
as-is.

### `_render_section` tweak (small, behavior-preserving)

Changed `elif cell.warning:` to `if cell.warning:` in the cell-text
composition loop, so a cell can show a Reference delta AND a short warning
together (needed for a negative-residual "Review" flag to stay visible even
when a delta is also present). Verified behavior-preserving for every
existing caller: a repo-wide check of every `ScorecardCell(...)`
construction confirmed `warning` and `formatted_delta` have always been
mutually exclusive by construction elsewhere in the codebase (`_metric_row`'s
incompatible branch sets `warning` and leaves `formatted_delta` `None`;
every other cell constructor never sets `warning` at all) -- so for every
pre-9D call site, at most one of the two was ever truthy, and the `if`/`elif`
change is a no-op for them. The full pre-existing
`test_comparison_report_viewmodels.py`/`test_comparison_report_page_smoke.py`
/`test_comparison_report_8e_smoke_matrix.py` suites (320 tests) pass
unmodified against this change.

## Comparison integration architecture

```
ComparisonItem (comparison_report_service.py, pre-resolved roadload/mass/RRC/CdA)
        |
comparison_vehicle_demand_viewmodels.py   (vde_app - pure, Comparison -> Vehicle Demand only)
        |  _vehicle_demand_request_from_comparison_item()  ->  VehicleDemandRequest
        |  resolve_vehicle_demand_cycle() (reused from vehicle_demand.adapters)
        |  calculate_vehicle_demand()     (vehicle_demand engine, frozen)
        |
VehicleDemandOutcome (result | short reason)
        |
build_vehicle_demand_comparison_rows() -> ScorecardSection (reuses ScorecardCell/_render_section)
build_vehicle_demand_breakdown_rows()  -> rows for the one optional chart
        |
components/comparison_report.py: _render_vehicle_demand_summary_section()
```

Dependency direction (Sec 5): `comparison_vehicle_demand_viewmodels.py`
imports FROM `comparison_report_service` (only the `ComparisonItem`/
`ComparisonDataset` types) and FROM `vehicle_demand` (contracts + engine +
the Sprint 9C `resolve_vehicle_demand_cycle` adapter). It is never imported
by either. `vehicle_demand/adapters.py` itself was not touched or deepened
further -- see "Known architectural note" below for why a *second*, smaller
adapter was written on the Comparison side instead of extending the Sprint
9C one.

## Vehicle Demand presentation builder/API

```python
outcome = get_vehicle_demand_result(item)              # VehicleDemandOutcome
section = build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.TOTAL, unit_system)  # ScorecardSection
breakdown = build_vehicle_demand_breakdown_rows(dataset, RoadloadBasis.TOTAL)              # chart rows
```

## How VehicleDemandResult is resolved for selected scenarios

`_vehicle_demand_request_from_comparison_item(item)` maps
`item.roadload["total"/"net"]` (already-resolved `RoadloadBoundary`
objects) directly into `RoadloadCoefficients`, and reads
`test_mass_kg`/`rrc_N_per_kN`/`cda_m2`/`legislation` from `item.vehicle`.
This is a **second, small adapter**, distinct from Sprint 9C's
`vehicle_demand.adapters.build_vehicle_demand_request` (which expects a raw
`vde_db` row with different column names/shape) -- reusing the 9C adapter
here would have required either re-fetching the row a second time or
reshaping `item.roadload` back into raw `coast_A_N`-style columns first,
for no benefit. The only piece of `vehicle_demand.adapters` reused is
`resolve_vehicle_demand_cycle`, called directly with `item.vehicle` (it
only ever reads a `"legislation"` key, which `item.vehicle` already
provides). `calculate_vehicle_demand()` itself is the unmodified, frozen
Sprint 9B engine entry point. Any `ValueError` it raises (invalid mass/RRC/
CdA/ambient, malformed cycle) is caught in `get_vehicle_demand_result` and
turned into a short reason -- never re-raised into the page.

No `AmbientState` data source exists yet in Comparison (`vde_db` has no
ambient columns, and Quick Scenario -- the future source of a user-supplied
ambient override -- is explicitly out of scope for 9D, Sec 55). Every
Comparison-sourced `VehicleDemandRequest` therefore carries
`ambient=AmbientState()` (all `None`), so **Known Aero Energy is always
`UNAVAILABLE` today** -- correct, documented behavior, not a bug (see
"Partial-availability behavior" below).

## Final KPI set shown

One table, primary KPIs first (Sec 13), no separate visual tier:

| Row | Unit |
|---|---|
| VDE | MJ/km |
| Roadload Energy | MJ |
| Positive Tractive Energy | MJ |
| Braking Energy Required | MJ |
| Known Rolling Energy | MJ |
| Known Aero Energy | MJ |
| Residual / Unattributed Roadload | MJ |
| Positive Inertial Work | MJ |

Every unit_family/quantity mapping was checked against the actual 9A/9B
field semantics before writing the renderer (Sec 29): `vde_mj_per_km` is a
distance-normalized rate (`energy_mj_per_km`, MJ/km, matching the existing
Registry `vde_total`/`vde_net` convention exactly); the other seven fields
are absolute energy over the whole cycle trace (`energy_mj`, MJ) -- a
quantity that did not previously exist in `units.py` (only rate-based `MJ/
km` existed) and was added as its own `QuantitySpec`, never silently
reusing or converting through the MJ/km one.

## TOTAL/NET UI behavior

Reuses the existing `roadload_basis` radio (`TOTAL`/`NET`/`Both`) verbatim
-- no new selector. For each selected boundary, one table (and, if
non-empty, one breakdown-chart expander) renders, captioned by boundary
name when both are shown. A scenario whose NET is genuinely unavailable
(e.g. no resolved transmission) shows `"-"` with the warning `"NET roadload
is unavailable for this scenario."` in every KPI cell for that
scenario/basis -- verified never to fall back to the TOTAL value (dedicated
regression test, and Smoke D).

## Reference-less behavior

When `dataset.reference is None`, every item's cells are built as absolute
values with `absolute_delta`/`formatted_delta` left `None` -- no baseline is
fabricated. Verified by a dedicated unit test and Smoke B (only Benchmarks
selected).

## Partial-availability behavior

- Missing RRC -> Known Rolling Energy cell shows `"-"` with warning `"RRC
  unavailable"`; VDE/Roadload Energy/Positive Tractive/Braking remain
  available on the same row set.
- Missing CdA (or, today, always -- see above, no ambient source yet) ->
  Known Aero Energy cell shows `"-"` with warning `"CdA/air density
  unavailable"`.
- Neither available -> Residual absorbs the entire authoritative roadload
  (verified: `residual == authoritative` exactly in that case); VDE/
  Roadload/tractive/braking remain valid. No row is ever hidden entirely --
  only the specific unavailable cell degrades (Sec 40).

## Residual presentation

Label: `"Residual / Unattributed Roadload"` (never "Other Losses" --
verified by a dedicated test). A negative value is shown as-is (never
`abs()`'d), with a discreet `" (Review)"` suffix appended directly to the
formatted value text (not only via the `warning` slot) so it stays visible
even when a Reference delta is also shown for that cell -- the specific
reason `_render_section`'s delta/warning coexistence tweak (above) was
needed. Warning text: `"Known contributions exceed authoritative roadload
for part of the cycle; residual is preserved."`

## Braking-energy terminology

Label: `"Braking Energy Required"`. A dedicated test asserts no row label
or warning anywhere in the Vehicle Demand section contains "regen" or
"recovered" -- braking energy is never presented as captured/recoverable
energy, only as the wheel-side mechanical energy the cycle requires to be
removed.

## Provenance presentation

`VehicleDemandRequest.provenance` (built the same way as Sprint 9C's
adapter: `roadload_total`/`roadload_net`/`transmission`/`rrc`/`cda`, using
the frozen `Provenance` enum) is computed but not separately surfaced in
the UI this sprint -- the KPI table's own per-cell `warning` text already
carries the practically useful subset (which contribution is unavailable
and why) without a second provenance panel. Technical Scorecard's optional
extension (Sec 35: "Vehicle Demand Model Version, Roadload Basis, Rolling/
Aero availability, Ambient basis, Residual warning" -- explicitly marked
"if cheap") was **deliberately not implemented** this sprint: the Energy
Drivers table already surfaces the practically important subset of this
information per-cell, and Sec 54's density concern ("can a user understand
this in ~10 minutes?") weighed against adding a second, largely-redundant
provenance surface in a different tab. Flagged here as a real, available,
low-risk candidate for 9E if a concrete need for it in Technical Scorecard
specifically arises.

## New visual added and justification

One (the maximum allowed, Sec 16): `build_vehicle_demand_breakdown_chart` --
a horizontal stacked bar per scenario, `barmode="relative"` (Known Rolling +
Known Aero + Residual/Unattributed), inside a collapsed-by-default
`st.expander` (matching the existing Demanded Power section's opt-in
pattern). Never stacked-to-100% (Sec 17): a component with no available
data for every row is omitted as a series entirely rather than plotted as a
zero segment, and `barmode="relative"` (not `"stack"`) correctly separates
positive and negative segments so a negative Residual visibly extends the
opposite direction rather than being absorbed into a same-direction stack.
The identity `Known Rolling + Known Aero + Residual == Roadload Energy` is
verified exactly (to 1e-6) in a dedicated test using the same real QA
scenario the table itself uses.

## Scenario-level error behavior

`get_vehicle_demand_result` never raises: a `ValueError` from
`calculate_vehicle_demand` (invalid mass/RRC/CdA/ambient, malformed cycle)
is caught and converted to a short reason string. That reason is used
directly as the cell's `warning` -- it is already a short, plain-English,
one-line message from the frozen engine (e.g. `"test_mass_kg must be
positive, got 0.0."`), so no rewriting/parsing was needed; a `"Vehicle
Demand unavailable: "` prefix is added for page context. Verified: no
`"Traceback"`/`"ValueError"` substring ever appears in a rendered warning,
the failing scenario's column stays visible with every other KPI row still
attempting its own computation independently, and every *other* scenario in
the same dataset computes normally.

## Performance behavior with multi-scenario comparison

`resolve_vehicle_demand_outcomes(dataset)` computes each item's
`VehicleDemandOutcome` **exactly once**, regardless of how many boundaries
(TOTAL/NET/Both) are selected or whether the breakdown chart is also shown
-- `_render_vehicle_demand_summary_section` calls it once and threads the
resulting dict into both `build_vehicle_demand_comparison_rows` (via its
`outcomes=` parameter, reused across all 8 KPI rows and every selected
boundary since each outcome already carries both `total_summary` and
`net_summary`) and `build_vehicle_demand_breakdown_rows`. This is a
per-render memo, discarded after the function returns on every rerun
(Sec 10-11): no `st.session_state` cache, no persistent/global cache, no
new caching framework. **Caught during self-review before committing**: an
earlier draft had the table builder and the breakdown builder each resolve
their own outcomes independently, which would have triggered
`calculate_vehicle_demand()` twice per item per boundary; a dedicated
regression test (`test_shared_outcomes_avoid_recomputation_across_both_
builders`, asserting `calculate_vehicle_demand`'s mock call count) now
guards this. No DB lookup happens per timestep (the adapter reads
already-fetched `ComparisonItem` fields only), and no intermediate
serialization occurs anywhere in the render path. The Smoke A/C tests
(3-scenario, TOTAL+NET) complete well within AppTest's existing 90s timeout
budget, consistent with Sprint 9C's own finding that one full regulatory-
cycle calculation is comfortably sub-second.

## Tests added

- `tests/test_comparison_vehicle_demand_viewmodels.py` -- 19 tests:
  Reference+2 comparisons, no-BETTER/WORSE-semantic, Reference-less,
  TOTAL, NET available, NET-missing-no-fallback, Rolling-missing,
  Aero-missing-without-ambient, partial-decomposition-still-comparable,
  negative-residual-preserved-and-flagged, never-"Other Losses",
  zero-braking-energy-is-real-zero, braking-terminology-not-regen,
  no-physics-duplication (monkeypatched `calculate_vehicle_demand`),
  shared-outcomes-avoid-recomputation-across-both-builders (the fix
  described above, with its own regression test), scenario-error-isolation
  (x2), real-QA-scenario end-to-end (x2, including the breakdown-identity
  check).
- `tests/test_comparison_report_vehicle_demand_smoke.py` -- 5 AppTest smoke
  tests (Smoke A-E, see below).

## Smoke cases/results

All 5 required cases (Sec 53) pass, using real QA fixtures
(`qa_mock_data.build_vde_seed_rows`) through the actual Streamlit page via
`streamlit.testing.v1.AppTest` (no browser tool available in this
environment, consistent with every prior Comparison package -- this is
structural smoke via AppTest, not pixel-level visual review):

- **A** (Reference + 2 comparisons): `len(app.exception) == 0`; "Vehicle
  Demand Summary" heading present.
- **B** (Reference-less, Benchmarks only): no crash; heading present;
  absolute-only construction verified precisely in the unit test suite
  above (AppTest's own settling rerun makes exact per-run element counts
  unreliable -- the existing test suite already established
  `assertGreaterEqual` over exact-count assertions for this reason, e.g.
  `test_comparison_report_page_smoke.py`'s `app.dataframe` checks; this
  package follows the same convention).
- **C** (TOTAL/NET switch): both bases render; `"TOTAL"`/`"NET"` captions
  both present when `"Both"` is selected.
- **D** (NET unavailable): rendered dataframe text contains "unavailable"
  for the scenario whose transmission is unresolved; no crash.
- **E** (partial decomposition, RRC and CdA both missing): no crash; VDE
  still renders; "RRC"/"unavailable" language present.

## Full test count/result

- Before this package: 1238 tests, 1236 passing, 2 known pre-existing
  failures.
- After this package: **1262 tests** (1238 + 19 + 5), **1260 passing**, the
  same 2 known pre-existing failures, **zero new failures**. The full
  pre-existing Comparison suites (`test_comparison_report_viewmodels.py`,
  `test_comparison_report_charts.py`, `test_comparison_report_service.py`,
  `test_comparison_metric_registry.py`, `test_comparison_report_page_
  smoke.py`, `test_comparison_report_8e_smoke_matrix.py` -- 320 tests total)
  were run explicitly against the `_render_section` change and pass
  unmodified.

## Known pre-existing failures

Unchanged: `tests/test_vde_request_resolver.py`, 2 failures (component-
snapshot/axle-hubs), not touched by this package.

## Commit(s)

Recorded below once committed, on branch `sprint-9a-vehicle-demand-contracts`
(not renamed, per Sec 1).

## Known architectural note (not a stop condition)

Sprint 9C flagged a one-directional `vehicle_demand.adapters ->
comparison_report_service` dependency as safe-today-but-worth-watching if
9D deepened it. 9D deliberately did **not** deepen that specific edge:
`vehicle_demand/adapters.py` is untouched, and the new Comparison-side
adapter (`comparison_vehicle_demand_viewmodels.py`) instead depends on
`comparison_report_service` (for the `ComparisonItem` type only) and on
`vehicle_demand` (contracts + engine + the one reused `resolve_vehicle_
demand_cycle` function) -- both in the Comparison-depends-on-Vehicle-Demand
direction Sec 5 requires. No new coupling was introduced in the
`vehicle_demand -> comparison` direction; `vehicle_demand/__init__.py`
still does not import `comparison_report_service.py` anywhere, directly or
transitively.

## Whether 9D is safe to freeze and proceed to 9E

Yes, with the deliberate Technical Scorecard omission above flagged as
available (not blocking) 9E scope. No Vehicle Demand physics was written or
modified. No frozen 9A/9B/9C contract was changed. Program Review,
Technical Scorecard's existing content, and Explore were not touched.
Quick Scenario and Powertrain physics were not implemented. Full suite has
zero new regressions.
