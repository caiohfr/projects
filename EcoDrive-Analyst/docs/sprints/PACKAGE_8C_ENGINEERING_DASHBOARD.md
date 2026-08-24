# Package 8C - Engineering Dashboard & Roadload Analysis

## Status

Completed. Follows Package 8B (Scorecard). Makes the Dashboard and Roadload & VDE
tabs functional; Explore stays a placeholder for a future package.

## Dashboard structure

Reference summary + dataset warnings → Vehicle Demand (TOTAL/NET/Both) → Fuel /
Energy / Emissions (dataset-aware: a metric's chart is omitted entirely, not shown
empty, when nothing in the dataset has it) → FE × VDE (mode + boundary selectors) →
Reference-relative KPI delta. Every chart is built from `ComparisonDataset` via
`comparison_report_viewmodels.py`; no chart function queries the database or
recomputes physics.

## Roadload & VDE structure

Source radio (linked VDEs from the current Scorecard selection, deduplicated by
`vde_id` / direct VDE selection, 1 Reference + up to 10, no FuelCons required) →
Roadload basis radio (TOTAL default/NET/Both) → temporary-transmission controls
(only shown when NET is requested and unresolved) → ABC bars → roadload force curve
→ cycle/phase VDE → demanded power over cycle (opt-in checkbox, see Known
limitations).

## TOTAL/NET controls

Every chart that can show a boundary exposes TOTAL/NET/Both explicitly (never a
default silent choice beyond the stated TOTAL default). "Both" always renders TOTAL
and NET as independent series (grouped bars, or dash-style-distinguished line
traces) - never merged or averaged. A missing NET is excluded from that specific
chart with a stated reason (`"Roadload NET unavailable"`, `"Cycle NET unavailable"`,
etc.) - never silently substituted with TOTAL.

## Direct VDE mode

`list_vde_catalog()` (new, `comparison_report_service.py`) provides lightweight
bare-VDE metadata for the selector without resolving any `ComparisonItem`. Selection
reuses `SelectionState`/`set_reference`/`sync_comparisons_from_widget` from Package
8B unmodified - those helpers operate on a generic int id, not something
FuelCons-specific. VDE-only items never fabricate fuel/energy/CO2/PSE values; those
fields are simply absent (`fuel_energy=None` and `powertrain` limited to what
`vde_db` itself carries), exactly as 8A already guarantees for `SourceKind.VDE_ONLY`.

## FE × VDE modes

Three modes, each with its own compatibility rule (`build_fe_vde_points`):
- **Volumetric** - only scenarios sharing the Reference's exact `fuel_type` are
  plotted together (different fuel families are excluded with a reason, never
  treated as energy-equivalent).
- **Energy-normalized** - converts `fuel_l_per_100km`/`energy_Wh_per_km` to a common
  MJ/km consumed-energy basis, making the comparison "independent of liters." Only
  fuel types with a defensible LHV mapping (see below) or BEV items participate.
- **Electrical** - BEV-only, x=VDE boundary, y=`energy_Wh_per_km`.

x is always `item.vde["aggregate"][boundary]` (TOTAL or NET, user-selected) -
never a fallback boundary; a missing value excludes that point with a reason.

## PSE-line semantics

`build_iso_pse_lines()` reuses the exact PSE ratio from `powertrain_efficiency.py`
(`demand / consumed`), inverted to solve for y given x and eta, with a **data-driven
x-domain** (padded from the actual plotted points) instead of a hardcoded range. It
returns `[]` - no lines, not fake ones - whenever the mode/fuel_type combination
isn't defensible (e.g. volumetric mode with an unmapped fuel type).

## Fuel compatibility / the three-LHV-table discrepancy

The repository had three inconsistent fuel-energy-density tables
(`fuel_energy.py::LHV_MJ_PER_L`, `derivatives.py::LHV_DEFAULT_MJ_PER_L`, and a
one-off default in `plots.py`), none matching `fuelcons_db.fuel_type`'s documented
vocabulary (`Gasoline/Ethanol/Flex/Diesel/Electric`). Package 8C resolves this by
treating `fuel_energy.py::LHV_MJ_PER_L`/`MJ_TO_Wh` as canonical (it's what actually
backs the stored consumption numbers) and mapping conservatively:
`Gasoline→32.0 MJ/L`, `Diesel→35.8 MJ/L`, `Ethanol→21.2 MJ/L` (via the table's
`E100` entry). `Flex` and any unrecognized fuel type are **not** LHV-mappable and are
excluded from energy-normalized/volumetric comparisons with a stated reason rather
than assigned a guessed blend percentage. This is a data-contract decision, not a
new physics formula - no LHV table was modified.

## Temporary transmission UX

Session-only: `st.session_state["comparison_temporary_transmission_by_vde_id"]`,
passed straight into 8A's `build_comparison_dataset(..., temporary_transmission_by_vde_id=...)`
- never written to the database. Offered per-VDE only when NET is requested and the
stored transmission boundary is unresolved; supports Component DB (
`component_repositories.load_component_repository("transmission")` +
`resolve_temporary_transmission_from_component`, both reused from 8A unmodified) and
Manual ABC (canonical `units.py` force/force_per_speed/force_per_speed_squared
inputs). A `"Clear temporary assumption"` action per VDE removes the entry; NET then
returns to unavailable unless a real stored boundary exists. Every temporary-derived
NET result is visibly marked (`is_temporary_net(item)` checks
`"temporary_transmission_used" in item.warnings`, threaded through by 8A's existing
resolver - no new `ComparisonItem` field was needed).

## Physical trace deduplication

`deduplicate_by_vde_id()` collapses ABC bars, the roadload curve, and cycle/phase
rows to one physical trace per distinct `vde_id`, carrying a `used_by` tuple (e.g.
`"HOMOLOGATED (#1)"`, `"ESTIMATED (#2)"`) for attribution. This applies **only** to
these three physical-chart builders - the Scorecard and Dashboard KPI rows continue
to show one row per selected FuelCons scenario, since fuel/energy consumption
legitimately differs per scenario even when the underlying VDE is shared.

## Known limitations

- **Demanded power over cycle is opt-in** (a checkbox, default off). Every
  `ComparisonItem` build already runs a full on-demand cycle/phase VDE calculation
  internally (Package 8A's `resolve_cycle_vde_results`, called unconditionally for
  every item regardless of UI need - see "8A API weakness" below); the *additional*
  cost `build_cycle_demand_rows` adds on top of that is a second, independent
  `roadload_analysis.build_cycle_power_analysis` call across the full cycle time
  series. Gating that specific call behind an explicit checkbox is the one place
  this package applies real lazy resolution (Sec 50).
- **Direct VDE mode's filters are simpler than the Scorecard's.** The Scorecard
  (Package 8B) preserves an already-selected scenario even when a filter temporarily
  hides it from the dropdown. Direct VDE mode's filter/selection interaction does not
  replicate that same hidden-id preservation logic - a filter change while VDEs are
  selected may require re-selecting a hidden one. This is a deliberate scope
  reduction for a secondary mode, not a data-integrity issue (selection state itself
  is never corrupted, only its visibility in the widget).
- **FE × VDE line-generation math was not visually inspected.** Formulas are unit
  tested (line changes with basis, empty for unmappable fuel types), but actual
  rendered chart appearance (line/marker overlap, hover readability at 10
  comparisons) depends on manual QA - no browser/screenshot tool is available in
  this execution environment.
- Cell background coloring and chart styling continue the restrained, translucent
  `rgba()` convention established in Package 8B - no new styling language introduced.

## 8A/8B API observations found while building the real UI

- `build_vde_comparison_item`/`build_scenario_comparison_item` **always** compute
  `cycle_results` (on-demand phase VDE) during item construction, regardless of
  whether any UI surface needs it. This was already true for Scorecard (8B) and
  Dashboard's VDE/FE×VDE charts benefit from it directly (no extra work), but it
  means there's no way to build a "cheap" `ComparisonItem` that skips cycle
  integration - a future package wanting a lighter physical-only VDE object would
  need a new 8A builder variant, not a parameter on the existing ones.
- `fetch_fuelcons_by_vde_id`/`fetch_fuelcons_rows` (repository layer, pre-8A) still
  don't select `record_origin`/`fuel_type` - confirmed again in 8C since
  `list_vde_catalog` had to be written independently rather than reusing those
  functions, mirroring the same gap Package 8B already found and worked around with
  `list_comparison_scenarios`.

## Recommended split (unchanged from Package 8B's recommendation)

- **8D - Explore Lite + Lineage**: Bar/Scatter/Line, X/Y/group picking off the Metric
  Registry, explicit lineage waterfalls (only where `lineage_status=EXPLICIT`).
- **8E - QA / UX / Freeze**.
