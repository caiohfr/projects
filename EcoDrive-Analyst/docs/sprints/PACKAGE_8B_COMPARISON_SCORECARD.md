# Package 8B - Comparison Scorecard & Scenario Selection

## Status

Completed. Follows Package 8A (canonical Comparison data layer). First production
UI on top of it.

## Selection model

The Scorecard's selectable unit is a `fuelcons_db` row (a complete scenario), not a
bare `vde_db` row. Two FuelCons rows linked to the same VDE are two distinct
selectable, distinct-column scenarios - never collapsed by `vde_id`.
`src/vde_core/comparison_report_service.py::list_comparison_scenarios()` provides
lightweight catalog metadata (make/model/year/legislation/electrification/
record_origin) for the selector without resolving a full `ComparisonItem` (no
transmission/cycle work) per candidate row.

## Reference semantics

Reference is a role, not a special record - any selected scenario can be Reference
(`src/vde_app/comparison_report_viewmodels.py::SelectionState`). Selecting a scenario
already present as Reference-vs-comparison auto-resolves: switching Reference drops
that id from the comparison list (`set_reference`); adding a scenario already selected
as Reference is rejected with a message (`add_comparison`).

## 10-comparison UI limit

`MAX_COMPARISONS = 10` is a UI-layer constant only - the Package 8A core
(`build_comparison_dataset`) remains unlimited. An 11th selection is rejected with an
explicit `st.warning`, never silently truncated. `sync_comparisons_from_widget`
reconciles Streamlit's multiselect against the existing `SelectionState` while
preserving original *selection* order (not widget/options-list order, which
Streamlit does not guarantee to match click order).

## Scorecard groups

Eight sections, in this order: Vehicle / Program, Powertrain, Physical Setup,
Roadload, Vehicle Demand, Fuel / Energy / Emissions, Efficiency, Data Status /
Provenance. The first seven are driven entirely by
`comparison_metric_registry.list_metrics(group)`; the eighth reads
`ComparisonItem.provenance` directly (provenance fields aren't "better/worse"
metrics, so they're descriptive-only, no delta/semantic).

## Metric Registry usage

`build_scorecard_sections()` never re-implements delta or compatibility logic - every
row's cells come from `comparison_report_service.compare_metric()`. The registry
gained a `"Powertrain"` group (split out of `"Vehicle"`) and five metrics reusing
already-computed `ComparisonItem` fields (`cycle_name`, `fuel_type`, `engine_type`,
`gear_count`, `final_drive_ratio`) - no new physics, no placeholder KPIs.

## Delta semantics

Each comparison cell shows the metric's own formatted value plus, when compatible,
`compare_metric()`'s `absolute_delta`/`percent_delta` formatted via
`_format_delta()`. `compare_metric()` can return `semantic="SAME"` in addition to
`BETTER`/`WORSE`/`None`; the Scorecard collapses `SAME` to no color (only
`BETTER`/`WORSE` get styled), matching the spec's BETTER/WORSE/NEUTRAL UI vocabulary.
A zero Reference value yields an absolute delta with no percent delta (`compare_metric`
already guards the division) - never a crash or an infinite percentage.

## Mixed legislation behavior

Incompatible cross-cycle metrics (e.g. EPA vs WLTP `vde_total`) still show each
item's own legitimate value - only the delta/semantic are suppressed, replaced with a
"Different cycle / incompatible basis" note. Physical metrics (mass, CdA, RRC) stay
fully comparable across legislations regardless (`ComparisonRule.ALWAYS`/
`BASIS_METADATA`).

## Stale-source behavior

`compare_saved_scenario_revision()`'s `current/changed/missing/unknown` states are
reused unmodified; `changed` displays as `STALE SOURCE` in the column header and
Provenance section. Stale scenarios render with full data, never hidden, recalculated,
or downgraded to unavailable.

## Legacy migration strategy

`pages/Comparison_Report.py` now delegates entirely to
`src/vde_app/components/comparison_report.py::render_comparison_report()`, which owns
the new `Scorecard / Dashboard / Roadload & VDE / Explore` tab structure. The
untouched legacy renderer (`pwt_fuel_energy.py::render_comparison_report_page`, its
four tabs Scenario Compare / Method Analysis / Peers & Outlook / Saved Estimates) is
reachable behind an `st.expander("Legacy comparison tools", expanded=False)` at the
bottom of the page - same pattern as the existing `st.expander("Legacy Sections", ...)`
in `vde_setup.py`. It will be retired as 8C (Dashboard charts) and 8D (Roadload & VDE)
absorb its useful capabilities; no refactor of it was made to fit this bridge.

## Known UI limitations

- Filtering the scenario catalog (make/legislation/electrification/provenance) can
  temporarily hide an already-selected scenario from the selector widgets. The
  underlying selection is preserved (not dropped) and still appears in the Scorecard;
  only its visibility in the "Compare with" multiselect is affected until the filter
  is cleared.
- Scorecard column headers encode role/provenance/stale status as a two-line string
  (`"\n"`-joined) rather than rich HTML badges, since `st.dataframe` column headers
  don't support arbitrary markup.
- Cell background coloring (BETTER/WORSE) uses a `pandas.Styler` with translucent
  `rgba()` backgrounds; this was verified via `st.dataframe` rendering without
  exception (`AppTest`), but actual color rendering in light/dark themes was not
  visually inspected - no screenshot/browser tool is available in this execution
  environment. Manual visual QA (10-comparison width, section readability, warning
  density) is recommended before wide rollout.
- The pre-existing Arrow-serialization warning seen in the legacy renderer's own
  scorecard table (`pwt_fuel_energy.py`, column "Reference VDE") is unrelated to this
  package - confirmed present in the untouched legacy code, auto-recovered by
  Streamlit, and out of scope to fix here.

## Recommended split (updated after implementing 8B)

- **8C - Engineering Dashboard**: VDE/ABC bars, roadload curves, FE/Energy/CO2, FE x
  VDE, equi-PSE, competitor delta walks, cycle demand plots.
- **8D - Explore Lite + Lineage**: Bar/Scatter/Line, X/Y/group picking off the Metric
  Registry, explicit lineage waterfalls (only where `lineage_status=EXPLICIT`).
- **8E - QA / UX / Freeze**.
