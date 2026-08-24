# Package 8D - Explore Lite & Explicit Lineage

## Status

Completed. Follows Package 8C (Dashboard/Roadload). Makes the Explore tab
functional with two independent sections: Custom Chart (Registry-driven
generic exploration) and Lineage (explicit parent-child VDE walk).

## Investigation Addendum: Lineage is Physical VDE Lineage only

Before writing any code, the actual schema was inspected. The only explicit
lineage field anywhere in the database is `vde_db.vde_id_parent`, populated
solely by the "From baseline (editable)" Scenario Setup workflow. `fuelcons_db`
has **no parent field of its own**. This means:

- Lineage is always resolved at the VDE level, never fabricated at the
  FuelCons scenario level.
- A FuelCons scenario may still be the *selection* that enters the Lineage
  view, but it does so by resolving its linked `vde_id` - the chain identity
  is the VDE id chain. The UI labels this clearly as **"Physical VDE
  Lineage"**, adding "resolved from FuelCons scenario '<label>'" as context
  only when the selection was a scenario.
- No schema changes, synthetic scenario-parent fields, or inferred FuelCons
  lineage were added. Two FuelCons scenarios sharing one VDE resolve to the
  *same* chain (`resolve_lineage_context` keyed on `vde_id`), while remaining
  distinct scenarios everywhere else in the Comparison Report (Scorecard,
  Dashboard, `dataset.comparisons`) - see
  `LineageContextTests.test_two_fuelcons_scenarios_sharing_one_vde_produce_identical_physical_lineage`.

## Explore Lite structure

Two `st.tabs`: **Custom Chart** and **Lineage** - never merged into one form.
Both consume `ComparisonDataset` (from the Scorecard selection, or read-only
from the Roadload tab's existing "Select VDEs directly" state - Sec 48; no
second selection UI is rendered) and the Metric Registry only. No raw SQLite
column, arbitrary expression, or second hardcoded KPI list is exposed.

## Custom Chart: dimensions, metrics, chart types

- **Numeric axes** come straight from `comparison_metric_registry.list_metrics()`,
  filtered by `unit_family != "text"` and by the metric's `compatible_chart_types`
  (`"bar"`/`"scatter"`; Line reuses the Bar-eligible set - a Line chart is a
  Bar chart with an explicit ordering basis, not a distinct metric class).
  `compatible_chart_types` was extended in this package for the physical/fuel
  metrics Sec 12/16 name explicitly (Mass, CdA, RRC, roadload A/B/C
  TOTAL/NET, Fuel consumption/economy, CO2, PSE) so Scatter can legitimately
  offer them - this is Registry metadata, not a second KPI list.
- **Categorical dimensions** are a small curated table
  (`_EXPLORE_DIMENSIONS` in `comparison_report_viewmodels.py`): Scenario,
  Vehicle, Make, Model Year, Category, Legislation, Electrification, Fuel
  type, Provenance - each tagged with the roles it's valid for
  (`x`/`order`/`group`/`filter`). Group/Filter only ever offer Category,
  Legislation, Electrification, Fuel type, Provenance (Sec 13). The Line
  chart's X selector only ever offers dimensions tagged `order` - today that
  is Model Year alone, so an unordered choice is structurally unavailable
  rather than merely discouraged (Sec 17).
- **Availability** is dataset-aware at two levels: `list_available_explore_metrics`
  drops a metric from the selector entirely when *no* current item has a
  value (Sec 8 "None available"); each chart builder additionally reports
  per-item exclusions with a reason (missing dimension, missing metric,
  filtered out) rather than silently dropping data (Sec 41, 47).
- **Chart builders** (`comparison_report_charts.py`): `build_explore_bar`
  (thin wrapper reusing `build_grouped_bar_figure` - identical row shape),
  `build_explore_scatter` (Reference marked with a star, one trace per group
  value, rich hover incl. provenance/temporary-NET/stale-source, no
  regression lines - Sec 16, 51), `build_explore_line` (plots rows in the
  order the viewmodel already sorted them; never reorders or infers order
  from selection sequence).

## Scenario identity vs display label

Two distinct scenarios/VDEs can legitimately share an identical label
(Package 8C precedent). Chart-preparation dictionary keys use a canonical
identity (`_canonical_identity`: `fc:<fuelcons_id>` or `vde:<vde_id>`), never
the label. Duplicate display labels are disambiguated for presentation only
(`_dedupe_display_labels`, a local equivalent of `components/comparison_report.py`'s
`_dedupe_titles`, kept local so the viewmodel module stays Streamlit-free) -
this was caught by `ExploreVdeOnlyChartTests.test_scatter_scenario_identity_preserved_independently_of_label`,
which failed against an earlier draft that reused the pre-existing
`_scenario_identity()` helper (that helper falls back to label for VDE_ONLY
items, which is correct for its original Roadload-dedup purpose but not for
a guaranteed-unique chart key).

## Physical VDE Lineage: resolver, waterfall, chart

- `resolve_lineage_chain(vde_id)` (`comparison_report_service.py`) walks
  `vde_id_parent` upward, ordered root → ... → selected. A new
  `LineageChainStatus` enum (`ROOT` / `EXPLICIT` / `BROKEN` / `MALFORMED`)
  reports exactly the four conceptual outcomes the addendum specified,
  without expanding the pre-existing `LineageStatus` enum (which describes a
  single item's own parent pointer, a different concern). A visited-id set
  guards self-parent, cycles, and repeated nodes - malformed data produces a
  warning and a truncated chain, never infinite recursion. A stored parent id
  whose VDE row no longer exists stops the walk with `BROKEN` and a
  structured warning; it is never silently skipped in favor of the
  grandparent.
- `build_lineage_waterfall(chain, metric_key)` (`comparison_report_viewmodels.py`)
  computes baseline = root's absolute value, each subsequent step =
  `compare_metric(parent_item, child_item, metric_key)["absolute_delta"]` -
  delta and BETTER/WORSE semantics are never recomputed, only reused from the
  same canonical comparison the Scorecard already relies on. The first
  incompatible (`SAME_LEGISLATION_CYCLE` basis mismatch) or unavailable node
  truncates the walk (`complete=False`, `incomplete_reason` set) - no
  fallback, no fabricated continuation. Physical metrics (`BASIS_METADATA`/
  `ALWAYS` comparison rule) remain usable across a cycle-incompatible
  transition where a cycle-specific metric (`SAME_LEGISLATION_CYCLE`) is
  blocked, exactly mirroring the Scorecard's existing compatibility rule.
- `build_lineage_waterfall_chart` (`comparison_report_charts.py`) is a real
  Plotly `go.Waterfall` (`measure=["absolute", "relative", ..., "total"]`),
  not the competitor delta bar chart repurposed. Only `status == "OK"` steps
  are plotted; a trailing `UNAVAILABLE`/`INCOMPATIBLE` marker step carries no
  numeric value and is surfaced via `incomplete_reason` text instead of a
  misleading bar.
- Lineage node metrics are restricted to what a bare VDE can legitimately
  expose: `list_lineage_capable_metrics()` excludes every Fuel/Energy/CO2/PSE
  metric (`source_requirement` in `FUEL_CONSUMPTION`/`FUEL_ENERGY`/
  `ELECTRICAL_ENERGY`/`CO2`/`PSE`), since lineage chain nodes are always built
  via `build_vde_comparison_item` (VDE_ONLY), which never populates
  `fuel_energy`. `list_available_lineage_metrics(chain)` further requires the
  metric to be available at **every** node in the current chain (Sec 31) - a
  metric missing at even one node is excluded from the selector rather than
  offered and failing mid-walk.

## Known limitation: no HIGHER_IS_BETTER metric is reachable in Lineage today

Every `MetricDirection.HIGHER_IS_BETTER` metric in the Registry
(`fuel_km_per_l`, `eta_pt_est`) has a Fuel/PSE `source_requirement` and is
therefore excluded from `list_lineage_capable_metrics()` by construction (see
above) - a bare VDE node can never populate them. `build_lineage_waterfall`'s
HIGHER_IS_BETTER branch is exercised indirectly (it delegates 100% to
`compare_metric`, already covered by
`ScorecardConstructionTests.test_higher_is_better_improvement_marks_better`)
but cannot be exercised with real Physical VDE Lineage data until a
HIGHER_IS_BETTER metric with a `VEHICLE`/`ROADLOAD_*`/`VDE_*` source
requirement exists in the Registry. This is a Registry content gap, not a
Package 8D logic gap.

## Explicitly not built (Sec 70)

Pie/radar/3D/heatmap/boxplot/histogram builders, saved Explore
configurations, arbitrary formulas/SQL, regression/correlation/ML,
branching lineage tree visualization, physical component decomposition
(waterfall steps are labeled by scenario, never by a guessed physical
cause - Sec 33).

## Recommended next package

**8E - QA / UX / cleanup / freeze**, per Package 8C's original recommendation
and this package's own STOP rule. No opportunistic ML/RAG/export work.
