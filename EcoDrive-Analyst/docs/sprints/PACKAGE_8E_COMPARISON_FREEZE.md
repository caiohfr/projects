# Package 8E - Comparison QA, UX Cleanup & Sprint Freeze

## Status

Completed. Sprint 8 (8A-8E) is frozen. This package made no new capability,
no physics change, and no schema change - it validated, polished, and froze
the Comparison Report built across 8A-8D.

## What this package is (and isn't)

8E is subtractive/corrective only, per its own primary rule. Before any edit,
the whole product was audited end-to-end (not package-by-package) and the
proposed diff was reported and confirmed before implementation. Five findings
were approved and applied; a 27-scenario end-to-end smoke matrix (Sec 39,
A-AA) was then run and passed clean with no new bugs, so no QA-driven fixes
were needed this round.

## Findings applied

1. **Legacy Comparison bridge caption (Sec 5)** - `pwt_fuel_energy.py`'s
   `render_comparison_report_page` was inspected directly. Its four sub-tabs
   are *not* uniformly superseded by 8A-8D: **Scenario Compare** (a
   single-VDE, up-to-3-scenario comparator with no Reference/delta model) is
   genuinely redundant with the new Scorecard, but **Method Analysis** (ML/
   regression method explanation), **Peers & Outlook** (DB-wide peer
   benchmark + a regulatory-outlook stub that is itself a pre-existing,
   unrelated placeholder), and **Saved Estimates** are Powertrain-Scenario
   capabilities with no Comparison equivalent. The bridge therefore stays,
   but its caption - which implied full eventual retirement - was rewritten
   to say plainly that Scenario Compare is superseded (prefer the Scorecard
   tab) while the other three remain because they belong to Powertrain
   Scenario. Zero changes to `pwt_fuel_energy.py`.
2. **Stale module docstring (Sec 36)** - `comparison_report.py`'s header
   comment said "until 8D absorbs its remaining useful capabilities"; 8D
   shipped and did not (see above). Corrected to state the actual, permanent
   relationship.
3. **Data-source terminology (Sec 7)** - Roadload's source radio said "VDEs
   linked to selected scenarios" / "Select VDEs directly"; Explore's said
   "Selected complete scenarios" / "Selected physical VDEs" for the same two
   concepts. Roadload's option text was aligned to use the same nouns
   ("complete scenarios" / "physical VDEs") Explore already used.
4. **Empty-state wording (Sec 13)** - four tabs had four different phrasings
   for "no reference selected yet" (including one inconsistent `"scenario/VDE"`
   slash-notation). Consolidated into one `_no_reference_message(action,
   allow_direct_vde=...)` helper so the subject/verb pattern is shared and
   only the action clause differs per section.
5. **Duplicated axis-label construction (Sec 35)** - the Dashboard's FE x VDE
   x-axis title was hand-built (`f"VDE {boundary} [{unit_label(...)}]"`)
   instead of reusing `metric_axis_label()` (added in 8D specifically to be
   the single source of truth for "Label [unit]" axis titles). Same output,
   one fewer parallel implementation.

Considered and explicitly **not** changed: repeating the same global dataset
warning (stale count / no-NET count / mixed legislation) across Scorecard,
Dashboard, and Explore - each tab is an independently-landable view of the
same dataset, so this is the "global warning" tier working as designed
(Sec 12), not duplication.

## End-to-end smoke matrix (Sec 39)

All 27 scenarios (A-AA) implemented as isolated-DB `AppTest` runs in
`tests/test_comparison_report_8e_smoke_matrix.py` - reference-only through
Reference+10, same-VDE/duplicate-title identity, provenance mixes, stale
source, TOTAL/NET/temporary-transmission apply-then-clear, EPA/WLTP/mixed
legislation, compatible/incompatible/BEV fuel, direct VDE-only mode, Explore
Bar/Scatter/Line/Group/Filter, and all four Physical VDE Lineage statuses
(ROOT/EXPLICIT/BROKEN/MALFORMED). **27/27 passed, zero unhandled exceptions,
on the first run** - no bugs were found, so no additional fixes were needed
beyond the five findings above.

## Visual QA (Sec 40-41)

No browser tool is available in this execution environment (consistent with
Package 8C's own finding). Structural smoke QA (AppTest, 27+16+other focused
tests) was run; **true pixel/visual evaluation was not performed** and should
not be read as inspected. This remains a manual step for whoever next opens
the app in a browser.

## Final architecture (Sec 46)

```
fuelcons_db / vde_db  (SQLite, read-only from Comparison's perspective)
        |
comparison_report_service.py         (vde_core - canonical, Streamlit-free)
        |
ComparisonItem / ComparisonDataset    (frozen dataclasses, one per scenario/VDE)
        |
comparison_metric_registry.py        (Metric Registry - single KPI/dimension source)
        |
comparison_report_viewmodels.py      (vde_app - pure, no Streamlit import)
        |
comparison_report_charts.py          (pure Plotly figure builders)
        |
components/comparison_report.py      (Streamlit UI - Scorecard/Dashboard/Roadload/Explore)
        |
pages/Comparison_Report.py           (page entry point)
```

Dependency direction is one-way top-to-bottom; nothing in `vde_core` imports
Streamlit or `vde_app`, and the page never queries SQLite directly.

### Reference semantics
One selected item holds `ComparisonRole.REFERENCE`; every metric is compared
*to* it (`compare_metric(reference_item, comparison_item, metric_key)`).
Reference is always shown at its own absolute value with no delta/verdict.

### TOTAL / NET
TOTAL is always the stored coastdown ABC boundary; NET is TOTAL minus the
resolved transmission boundary, only when that boundary is `AVAILABLE` or
`TEMPORARY` (never derived from NET back to TOTAL, never fabricated when
missing).

### Temporary transmission
Session-only (`st.session_state`), passed into `build_comparison_dataset(...,
temporary_transmission_by_vde_id=...)`; never written to `vde_db` or
`component_db`. Every temporary-derived NET result is visibly tagged via
`is_temporary_net(item)`.

### Fuel compatibility
`fuel_energy.py::LHV_MJ_PER_L`/`MJ_TO_Wh` is canonical for Comparison.
Volumetric mode requires an exact fuel-type match to the Reference;
energy-normalized mode converts to a common MJ/km basis for LHV-mappable
fuels (Gasoline/Diesel/Ethanol) or BEV; `Flex`/unknown fuel types are
excluded with a reason, never guessed.

### Metric compatibility
Every Registry metric declares a `ComparisonRule` (`ALWAYS` /
`BASIS_METADATA` / `SAME_LEGISLATION_CYCLE`). Cycle-specific metrics
(VDE TOTAL/NET, roadload ABC) are blocked across a legislation mismatch;
physical metrics (Mass, CdA, RRC) remain valid across it.

### Physical VDE Lineage
The only explicit lineage source in the schema is `vde_db.vde_id_parent`;
`fuelcons_db` has none. `resolve_lineage_chain()` walks it with a visited-id
guard and reports one of four structured statuses - `ROOT` (valid, no
parent), `EXPLICIT` (valid multi-node chain), `BROKEN` (parent id doesn't
resolve), `MALFORMED` (self-parent/cycle/repeated node) - never infers a
relationship from timestamps, labels, or similar values, and never recurses
without bound.

### Stale-source handling
`compare_saved_scenario_revision()` compares a FuelCons scenario's saved
`source_vde_revision` against the linked VDE's current revision; a mismatch
is surfaced as `RevisionStatus.STALE` (visible badge + Scorecard warning),
never silently refreshed or hidden.

## Performance (Sec 31)

Measured `build_comparison_dataset` directly against the QA DB (5 runs each,
this machine):

| Selection | avg | min-max |
|---|---|---|
| Reference only | 29 ms | 25-35 ms |
| Ref + 1 | 45 ms | 42-49 ms |
| Ref + 5 | 111 ms | 106-115 ms |
| Ref + 10 | 194 ms | 186-200 ms |

Linear, ~17 ms/item, under 200 ms at the documented maximum (10 comparisons).
**No optimization was made** - performance is acceptable as-is. Eager cycle
resolution inside `build_*_comparison_item` (documented as a known 8C
observation) remains a candidate for future lazy resolution only if a real
UI-blocking case is ever observed; it is not one today.

## Technical debt register (Sec 47)

Carried forward from 8C, still true, not addressed in 8E (by design):
- **Centralize fuel-property/LHV definitions** - three inconsistent tables
  still exist repo-wide (`fuel_energy.py`, `derivatives.py`, a `plots.py`
  default); Comparison canonicalized on `fuel_energy.py` for its own use
  only.
- **Move roadload curve physics fully into core** - `plots.py` still
  contains an `A + Bv + Cv^2` implementation used by the Roadload force
  curve chart.
- **Lazy cycle-result resolution** - `build_*_comparison_item` always
  computes on-demand cycle/phase VDE regardless of UI need; acceptable today
  (see Performance above), worth revisiting only if a real bottleneck
  appears.
- **FuelCons repository narrow-column API** - `list_comparison_scenarios`/
  `list_vde_catalog` had to be written independently because the pre-8A
  repository functions don't select the columns Comparison needs.

New, found during 8E:
- **`ComparisonItem.lineage` and `ComparisonDataset.warnings` are computed
  but never read by any UI code.** Confirmed by grepping `vde_app/` for both
  - zero hits outside their own definitions/tests. Both remain part of the
  tested public contract (`LineageTests` in `test_comparison_report_service.py`),
  so this is not dead code to remove, just an unused-by-UI data path. A
  future package could surface `item.lineage` (the per-item EXPLICIT/NONE
  pointer) as a Scorecard provenance row, or drop `dataset.warnings` if it's
  confirmed genuinely superseded by `dataset_warnings_summary()`.
- **The legacy "Scenario Compare" sub-tab is fully superseded** but was not
  removed from `pwt_fuel_energy.py` this package (see Findings #1) - doing
  so touches legacy-owned code beyond 8E's "no broad `pwt_fuel_energy.py`
  refactor" boundary. A future, dedicated legacy-cleanup package could remove
  `render_scorecard_panel` and its tab entry once confirmed unused elsewhere.
- **True Powertrain/FuelCons scenario lineage does not yet exist** - restated
  from 8D; `fuelcons_db` has no parent field, so Physical VDE Lineage remains
  the only lineage domain.
- **Browser-level visual regression testing is not automated** - restated
  from 8C; still true, no browser tool available in this environment.

## Regression (Sec 42-44)

Baseline reconfirmed identical to the post-8D state before any 8E edit:
**1012 tests, 1010 pass, 2 known pre-existing failures** in
`test_vde_request_resolver.py` (component-snapshot/axle-hubs, unrelated to
Comparison). After 8E's changes plus the new 27-scenario smoke matrix:
**1039 tests, 1037 pass, the same 2 known pre-existing failures, zero new
failures.**

## Sprint 8 freeze

The Comparison Report now supports exactly the four-tab surface targeted at
the start of 8E (Scorecard, Dashboard, Roadload & VDE, Explore incl. Physical
VDE Lineage) with no remaining package-boundary placeholder. Sprint 8 is
closed; see the 8E closure report for commit references and the recommended
next sprint.
