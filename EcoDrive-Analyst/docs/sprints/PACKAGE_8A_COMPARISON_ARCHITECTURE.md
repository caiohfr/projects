# Package 8A - Comparison Architecture & Canonical Dataset

## Status

Completed. Follows Package 7G (VDE TOTAL/NET data contract). This package adds a
canonical, Streamlit-free Comparison data layer that later Scorecard, Dashboard,
FE x VDE, Roadload & VDE, and Explore packages will consume. It does not rebuild
the current Comparison Report UI, which continues to work unchanged.

## Complete Scenario vs VDE-only

Every comparable item is a `ComparisonItem`. `source_kind` is
`FUELCONS_SCENARIO` (built from a `fuelcons_db` row, linked VDE resolved in
batch) or `VDE_ONLY` (built straight from a `vde_db` row). A VDE-only item has
`fuel_energy=None` and a `powertrain` dict limited to what `vde_db` itself
carries (`engine_type`, `transmission_type`, ...) - missing powertrain/fuel
data is represented as `None`, never `0` or a placeholder.

## Reference role

`Reference` is a role (`ComparisonRole.REFERENCE` / `COMPARISON`), not a
database entity - any valid item, scenario or VDE-only, can be built with
`role=REFERENCE`. Building an item never writes to the database; deltas are
computed against whichever item was built as Reference via `compare_metric`.

## TOTAL default / NET resolution

TOTAL roadload is always `coast_A_N/B/C` (canonical since Package 7G) and is
never derived from NET. NET is `TOTAL - transmission ABC`, resolved only when
the transmission boundary is `AVAILABLE` (all three `trans_*_coef_N` columns
non-NULL - missing is not equivalent to zero) or `TEMPORARY` (an
explicitly-supplied assumption). A missing transmission boundary leaves NET
`None`, never a manufactured value from a flat percentage or a transmission
type default.

Persisted aggregates (`vde_total_mj_per_km`/`vde_net_mj_per_km`) are read
exclusively through the Package 7G `canonical_vde_read()` contract. When the
persisted NET is unavailable, `resolve_vde_aggregate()` may fall back to an
on-demand cycle-calculated NET, but always tags the result `net_source`
(`"stored"` or `"on_demand"`) so no value is ever silently swapped for
another.

## Temporary transmission

`resolve_temporary_transmission_from_component(component_id)` looks up a
transmission component's canonical ABC (`component_repositories.get_component`)
without writing anything. A `temporary_transmission_by_vde_id` map can be
passed into `build_comparison_dataset`/the item builders; a resolved value
from it is always tagged `TransmissionStatus.TEMPORARY` and carries the
`temporary_transmission_used` warning - it is used only when the stored
transmission boundary is missing, never to override a resolved one.

## On-demand cycle VDE

`resolve_cycle_vde_results()` never reads the historical
`vde_urb_mj_per_km`/`vde_low_mj_per_km`/etc. columns as canonical TOTAL/NET.
It resolves the standard cycle trace for the row's legislation
(`cycles.use_standard_cycle`, EPA or WLTP today; any other legislation or a
missing/bad cycle file yields `aggregate=None` with a `cycle_trace_unavailable`
warning, never an invented cycle), resolves the TOTAL/NET ABC boundaries, and
delegates the actual energy integration to the existing
`vde_setup_service.compute_vde_preview_from_inputs` (wrapped as
`compute_vde_for_boundary` purely to avoid new code calling the
historically-named `compute_vde_net()` directly - physics is unchanged).

## Stale-source policy

`compare_saved_scenario_revision()` (Package 7-era, unchanged) is reused as-is
and its `current/changed/missing/unknown` states are translated to the
comparison-facing `RevisionStatus.CURRENT/STALE/MISSING/UNKNOWN`. A stale
scenario still builds successfully, keeps every field, and carries a
`fuelcons_source_stale` warning - it is never hidden, recalculated, or
silently overwritten.

## Compatibility rules

Each `MetricDefinition` carries a `comparison_rule`:

- `ALWAYS` - freely comparable (CdA, RRC, fuel/energy/CO2, efficiency).
- `BASIS_METADATA` - always comparable, but flags `basis_mismatch` when
  legislations differ (mass, test mass).
- `SAME_LEGISLATION_CYCLE` - only comparable when both items share the same
  legislation (VDE TOTAL/NET, roadload coefficients).

`compare_metric()` never assigns an overall vehicle score; `semantic`
(BETTER/WORSE/SAME) only exists for `LOWER_IS_BETTER`/`HIGHER_IS_BETTER`
metrics, and is `None` for `NEUTRAL` metrics like mass or raw roadload
coefficients.

## Future tabs this package prepares

Scorecard/Dashboard consume `ComparisonDataset` + `compare_metric()` directly.
FE x VDE can key off `vde.aggregate`/`fuel_energy` with an explicit energy
basis (TOTAL/NET) instead of hard-coding `vde_net_mj_per_km`. Roadload charts
can consume `roadload.total`/`roadload.net` instead of re-deriving ABC from a
VDE row. Explore/lineage waterfalls can use `lineage.lineage_status` to
distinguish an explicit parent-child chain from an unrelated set of
competitors.

## Appendix - current-capability reuse map

| Existing code | Package 8A reuse |
|---|---|
| `vde_net_total_contract.canonical_vde_read` | `resolve_vde_aggregate` (TOTAL/NET persisted read) |
| `vde_setup_service.compute_vde_preview_from_inputs` | `compute_vde_for_boundary` (on-demand cycle energy) |
| `cycles.use_standard_cycle` | cycle resolution in `resolve_cycle_vde_results` |
| `phase_aggregation.epa_city_hwy_from_phase` / `wltp_phases_from_phase` | invoked transitively via `compute_vde_preview_from_inputs`, never re-implemented |
| `pwt_fuel_energy_service.compare_saved_scenario_revision` / `resolve_vde_source_revision` | stale-source detection (`_resolve_revision`) |
| `component_repositories.get_component` | `resolve_temporary_transmission_from_component` |
| `repositories.fetch_vde_by_id` / `fetch_vde_by_ids` | single and batch VDE reads in the item/dataset builders |
| `database_management_service.get_record(EntityType.FUEL_CONSUMPTION, ...)` | single FuelCons row fetch (the dedicated `fuelcons_repository` fetch functions use narrow column whitelists missing `record_origin`/`fuel_type`; `get_record` already does `SELECT *`) |
| `comparison_report_service.build_vehicle_label` (pre-existing, previously unused) | `ComparisonItem.label` |
| `_scenario_scorecard_field_value` (`pwt_fuel_energy.py`, Streamlit-coupled) | not imported; its ~40-key enumeration informed which metrics the registry should cover, nothing else |

## Known repo/spec gaps documented during this package

- `vde_id_parent` lineage is populated only by the compact-proposal save path
  (`vde_request_save.py`); most historical VDE rows correctly report
  `lineage_status=NONE`, not `UNKNOWN`.
- No QA fixtures exist for `fuelcons_db` rows or a WLTP-legislation VDE row;
  Package 8A tests build minimal fixtures directly rather than extending
  `qa_mock_data.py` (kept out of scope).
- `vde_total_simple()`/`_SIMPLE_TRANS_FACTORS` (`vde_calc.py`) remain in the
  codebase as confirmed dead code (zero call sites in the live path); Package
  8A does not call them and does not remove them (out of scope).
