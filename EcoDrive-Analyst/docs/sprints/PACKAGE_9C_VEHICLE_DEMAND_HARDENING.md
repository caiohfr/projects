# Sprint 9C - Real Scenario Validation & Vehicle Demand Hardening

## Status

Completed. Adds the canonical adapter from resolved `vde_db` rows into
`VehicleDemandRequest`, validates the Sprint 9B engine against real QA
scenarios (not just synthetic contract-level QA), and hardens the engine
against invalid/incomplete physical inputs. No Comparison UI, Quick
Scenario, or Powertrain code was touched.

## Baseline

Branch `sprint-9a-vehicle-demand-contracts` (Sprint 9A commits `cae884c0`/
`91913320`; Sprint 9B commits `66f1bc7a`/`1b3dd9cf`). Baseline: 1196 tests,
1194 passing, 2 known pre-existing failures in `test_vde_request_resolver.py`
(component-snapshot/axle-hubs), untouched by this package.

## Investigation before implementation

Per Sec 4, the repo was inspected for the smallest existing canonical
boundary between EcoDrive and `VehicleDemandRequest` before writing any
adapter code:

- **The canonical "resolved VDE row"** is exactly what
  `src.vde_core.repositories.fetch_vde_by_id` already returns: a plain
  `Mapping[str, Any]` of the `vde_db` row. This is also, precisely, the
  input type `comparison_report_service.py`'s own resolvers
  (`resolve_roadload_boundaries`, `resolve_transmission_boundary`,
  `resolve_cycle_vde_results`, `vde_net_total_contract.canonical_vde_read`)
  already take -- confirming it as the real, smallest, already-stable
  boundary (Sec 7's "avoid a dict interface if a better typed structure
  exists" does not apply here: this dict *is* the existing canonical
  representation, not an ad hoc one).
- **NET ABC resolution** already exists and is exactly reused:
  `comparison_report_service.resolve_roadload_boundaries(vde_row,
  temporary_transmission)` returns TOTAL (always the stored coastdown ABC)
  and NET (TOTAL minus the resolved transmission ABC, only when the
  transmission boundary is AVAILABLE/TEMPORARY, never derived from NET back
  to TOTAL) -- this is bit-identical to what Comparison Report itself uses.
- **Cycle resolution** already exists and is exactly reused:
  `comparison_report_service.resolve_cycle_vde_results` calls
  `cycles.use_standard_cycle(legislation)` -- keyed on legislation ONLY,
  never on the row's free-text `cycle_name` column. The adapter's
  `resolve_vehicle_demand_cycle` is a one-line delegation to the same
  function, so cycle identity, never a heuristic string match, decides the
  trace (Sec 9).
- **Test mass fallback**: `comparison_report_service._resolve_mass_for_cycle`
  (private) uses `test_mass_kg` if present, else `mass_kg`. Reimplemented
  locally as a two-line public helper in the adapter rather than importing
  a leading-underscore name from another module -- the adapter is meant to
  be a stable, frozen surface (see "Deviations" below).
- **`ComparisonItem`/`ComparisonDataset`** (the Comparison-layer canonical
  snapshot) was deliberately NOT used as the adapter's input type, even
  though it is a more fully-typed object than a raw row. Adapting from it
  would create a dependency from `vehicle_demand` onto
  `comparison_report_service.py` for a second, larger reason on top of the
  boundary-resolver reuse already needed (see "Known architectural coupling"
  below) -- the raw `vde_row` mapping is the smaller, more fundamental, and
  direction-correct choice (Comparison should eventually depend on Vehicle
  Demand, Sprint 9D; not vice versa for its own richest object).

## Files created/modified

Created:
- `src/vde_core/vehicle_demand/adapters.py` -- `build_vehicle_demand_request`,
  `resolve_vehicle_demand_cycle`.
- `tests/test_vehicle_demand_integration.py` -- 42 tests.
- `docs/sprints/PACKAGE_9C_VEHICLE_DEMAND_HARDENING.md` (this file).

Modified:
- `src/vde_core/vehicle_demand/physics.py` -- added `_require_finite` and
  sign/range validation to `resolve_air_density`, `known_rolling_force_N`,
  `known_aero_force_N` (Sec 18-22, 24). No existing 9B behavior changed for
  valid inputs; all 24 pre-existing engine tests pass unmodified.
- `src/vde_core/vehicle_demand/engine.py` -- `build_vehicle_demand_profile`
  now validates `test_mass_kg > 0` and that mass/roadload coefficients are
  finite before computing (Sec 19, 24), reusing `physics._require_finite`
  rather than a second copy of the same check.

**No file in `contracts.py` or `serialization.py` was changed, and no field,
type, or `__post_init__` rule was added to any 9A contract.** Hardening was
implemented entirely at the engine/physics layer that Sprint 9B already
owns -- see "API freeze review" below for why this was the right layer.

`__init__.py` was deliberately left unchanged: `adapters.py`'s functions are
NOT re-exported from the package `__init__.py` (see "Known architectural
coupling").

## Canonical adapter public API

```python
from src.vde_core.vehicle_demand.adapters import build_vehicle_demand_request, resolve_vehicle_demand_cycle
from src.vde_core.vehicle_demand import calculate_vehicle_demand

request = build_vehicle_demand_request(vde_row, temporary_transmission=None, ambient=None)
cycle_frame = resolve_vehicle_demand_cycle(vde_row)
result = calculate_vehicle_demand(request, cycle_frame)
```

`vde_row` is whatever `fetch_vde_by_id`/`fetch_vde_by_ids` already returns --
the adapter performs no DB access itself (Sec 8). `temporary_transmission`
passes straight through to `resolve_roadload_boundaries`, and `ambient` lets
a caller supply real environmental conditions; both are explicit override
hooks a future Quick Scenario "temporary override resolver" can use without
the adapter needing to know anything about overrides itself (Sec 36 -- this
was reviewed, not implemented). The adapter raises `ValueError` only when
the row has no TOTAL roadload at all (a row that genuinely cannot be
represented); every other absence (NET, RRC, CdA, ambient) flows through as
`None`/"unavailable", never fabricated.

## Cycle-resolution path

`resolve_vehicle_demand_cycle(vde_row)` = `cycles.use_standard_cycle(vde_row
.get("legislation"))`. Verified against both regulatory cycle files actually
in the repo: `data/cycles/FTP75_HWFET.csv` (EPA, 2139 rows, `bag 1`/`bag 2`/
`HWFET` phases, contiguous non-decreasing time 0-2145s) and
`data/cycles/WLTP_Class3ab.csv` (WLTP, 1802 rows, `low`/`mid`/`high`/`xhigh`
phases, contiguous time 0-1801s). Both load and validate cleanly through
`vde_calc.extract_cycle_arrays`.

## Real deterministic QA scenarios used

All from `qa_mock_data.build_vde_seed_rows()` (7 existing, deterministic,
clearly-synthetic EPA-legislation rows) plus one new WLTP-legislation row
defined locally in the test file (no WLTP row exists in the shared QA
fixtures; kept local per Sec 38 rather than added to `qa_mock_data.py`):

- **VDE-QA-001** ("Nominal EPA baseline") -- TOTAL+NET, RRC+CdA present.
  Primary scenario for the end-to-end flow, reconciliation, provenance, and
  most availability-matrix tests.
- **VDE-QA-004** ("Higher mass baseline") -- TOTAL+NET, RRC+CdA present.
  Second, independent EPA reconciliation data point.
- **VDE-QA-006** ("Missing optional fields") -- TOTAL only (NET genuinely
  unavailable: all `trans_*` columns are `None`), RRC+CdA still present.
  The TOTAL-only scenario.
- One locally-defined synthetic WLTP row -- TOTAL+NET, RRC+CdA present.

## TOTAL/NET reconciliation results -- and the central empirical finding

**The QA fixtures' stored `vde_total_mj_per_km`/`vde_net_mj_per_km` columns
are NOT reconcilable with their own row's ABC/mass/cycle, and must not be
used as the reconciliation target.** This was verified directly before
writing any reconciliation test: running VDE-QA-001's own `coast_A_N`/`B`/`C`
+ `test_mass_kg` through the project's own on-demand recompute
(`comparison_report_service.resolve_cycle_vde_results`) on the real
FTP75_HWFET trace gives **0.313 MJ/km**, while the row's *stored*
`vde_total_mj_per_km` is **1.240** -- off by ~4x, for all three EPA rows
checked. This isn't a bug in the engine; it means the QA fixtures' stored
VDE numbers are independently-chosen placeholders (for testing UI sorting/
delta logic elsewhere), never meant to be physically derived from their own
row's ABC. `comparison_report_service.resolve_cycle_vde_results`'s own
docstring already says as much ("Never trusts historical phase columns...
as canonical TOTAL/NET").

The correct reconciliation target is therefore
`resolve_cycle_vde_results()`'s on-demand, physics-consistent recompute --
the same thing Comparison Report itself uses for on-demand values. Reusing
it:

- **EPA (VDE-QA-001, VDE-QA-004, TOTAL and NET)**: `resolve_cycle_vde_results`
  is EPA 55/45 city/highway phase-weighted
  (`phase_aggregation.epa_city_hwy_from_phase`). `VehicleDemandSummary` has
  no by-phase field (frozen in 9A), so the test reconstructs the same
  combination from three separate engine calls -- one per phase segment
  (`bag 1`, `bag 2`, `HWFET`, via `phase_aggregation.split_by_phase`) --
  summing energy/distance for the two city bags and applying the same
  0.55/0.45 weights `epa_city_hwy_from_phase` uses, rather than teaching the
  engine itself EPA policy. Result: **exact match to floating-point
  precision** (differences of `0.0` or `~1e-17`, i.e. machine epsilon) for
  both TOTAL and NET, across both rows.
- **EPA (VDE-QA-006, TOTAL only)**: same per-phase reconstruction reconciles
  exactly; NET correctly stays unavailable end-to-end (adapter produces
  `roadload_net=None`, `calculate_vehicle_demand` produces `net_summary=
  None`, no fallback).
- **WLTP (synthetic row, TOTAL and NET)**: WLTP's phase combination
  (`wltp_phases_from_phase`) is a genuine distance-weighted average across
  contiguous phases, not a fixed policy weight -- verified empirically that
  a single **whole-trace** engine call already reconciles to floating-point
  precision (`0.0`/`0.0` relative difference) with `resolve_cycle_vde_results`,
  no per-phase reconstruction needed.

This is the strongest possible reconciliation result available given the
frozen 9A/9B contracts' scope (no by-phase Summary) -- not a "cannot
reconcile" stop condition (Sec 42 #2): reconciliation works cleanly once
compared against the right target.

## Availability matrix results (Sec 13, Cases A-D)

All four built from VDE-QA-001 with RRC/CdA selectively removed (real
authoritative roadload/mass/cycle kept, only the decomposition inputs under
test vary):

- **A** (RRC+CdA+ambient all available): Rolling `CALCULATED`, Aero
  `CALCULATED`, Residual `CALCULATED`.
- **B** (CdA missing): Rolling `CALCULATED`, Aero `UNAVAILABLE`, VDE and
  residual still valid.
- **C** (RRC missing): Rolling `UNAVAILABLE`, Aero `CALCULATED`, VDE still
  valid.
- **D** (neither): both `UNAVAILABLE`; residual equals the full
  authoritative roadload exactly (nothing subtracted); authoritative
  roadload/tractive demand/VDE all still valid -- decomposition never
  becomes an engine failure.

Ambient matrix (Sec 14): explicit density used as-is (`SOURCE`/whatever
basis supplied); temperature+pressure produce `CALCULATED` density with the
temperature/pressure provenance passed through; no environment at all
leaves Aero `UNAVAILABLE` with authoritative roadload/VDE unaffected. No
regulatory default was added (still frozen from 9B).

## Provenance/warning behavior

`VehicleDemandRequest.provenance` for a QA row with full data: `{"roadload_
total": "SOURCE", "roadload_net": "CALCULATED", "transmission": "AVAILABLE",
"rrc": "SOURCE", "cda": "SOURCE"}`; for VDE-QA-006 (no transmission):
`"transmission": "MISSING"`, `"roadload_net": "UNAVAILABLE"` -- explained,
never silently dropped. `VehicleDemandSummary.warnings` text was verified to
use plain domain language ("Known rolling contribution unavailable: `rrc_
n_per_kn` ... missing", "Known aero contribution unavailable: `cda_m2`
missing", "Residual roadload is negative at one or more timesteps...") --
already correct from Sprint 9B, confirmed end-to-end through the adapter in
this package. Residual is never called "Other Component Losses" anywhere
(Sec 28); verified by a dedicated test asserting the phrase never appears.

## Residual semantics (Sec 27-28) -- real/synthetic positive, zero, negative

- **Positive**: typical case, demonstrated with VDE-QA-001's own cycle/mass/
  authoritative ABC and a reduced RRC/CdA (the *unmodified* QA-001 RRC/CdA
  combination turned out to already produce a negative residual at every
  single timestep -- an incidental finding, see below -- so a smaller,
  still-realistic RRC/CdA pair was used specifically to show the ordinary
  case).
  qa row.
- **Exactly zero**: engineered so `known_rolling_force_N` (constant, from
  RRC) exactly equals a constant authoritative `A` term with `B=C=0` and
  `CdA=0`; residual is `0.0` to 9 decimal places at every timestep.
  and
- **Negative**: over-attributed case (tiny authoritative `A_N=1.0` against a
  realistic RRC), preserved with no clipping, with a warning present.

**Incidental finding**: VDE-QA-001's own stored `rrc_N_per_kN=8.0` and
`cda_m2=0.620` combination, run against its own `coast_A_N/B/C`, actually
produces a *negative* residual at all 2139 timesteps of the FTP75_HWFET
cycle -- consistent with the broader finding above that these QA fixtures'
physical fields were chosen independently of one another, not as a mutually
self-consistent vehicle. This is not a defect: Sprint 9B/9C's own rule is
that a negative residual must be preserved and flagged, which is exactly
what happens.

## Edge cases added (Sec 18-26)

- **Zero values**: RRC=0, CdA=0 verified as real, present zero forces (not
  `None`); zero braking energy re-verified through the adapter path.
- **Invalid mass** (`<= 0`, including exactly `0.0`): `ValueError`, not a
  nonsense negative-inertia result.
- **Invalid RRC/CdA** (`< 0`): `ValueError`; `0` remains valid.
- **Invalid ambient** (temperature at/below absolute zero, pressure `<= 0`,
  explicit density `<= 0`): `ValueError` for all three, whether the ambient
  came from `temperature_C`+`pressure_kPa` or an explicit `air_density_kg_m3`.
- **Non-finite input** (`NaN`/`inf` in mass, a roadload coefficient, RRC,
  ambient temperature): `ValueError` raised before any computation, not
  silently converted to `None` only once it reaches the JSON boundary.
- **Cycle edge cases**: empty, single-point, duplicate-timestamp, and
  non-monotonic cycles are rejected by `build_vehicle_demand_profile` with
  the exact same `ValueError`s `compute_vde_net` raises (both delegate to
  `vde_calc.extract_cycle_arrays` -- no divergent policy, per Sec 23).
  Negative-but-monotonically-increasing timestamps are accepted by both
  (relative spacing is what the physics cares about; no special rejection
  was invented, since `compute_vde_net` never had one either).
- **Zero distance**: `build_vehicle_demand_profile` still succeeds (a valid
  all-zero-speed time series), but `summarize_vehicle_demand` raises the
  identical `"cycle distance must be positive"` `ValueError`
  `compute_vde_net` raises -- distance-rate normalization is exactly where
  this has to be caught, not earlier.
- **Malformed Profile regression guard**: one test asserts a real QA-derived
  `VehicleDemandProfile`'s every optional/required series matches
  `len(time_s)`, confirming the engine's own construction can never violate
  the 9A shape contract it's built on (Sec 26 -- 9A's `__post_init__`
  already enforces this; nothing new was added here).

**Design decision**: validation added in this package always distinguishes
*missing* (a soft `None`/"unavailable", never an exception) from *present
but physically impossible* (a `ValueError`, fail fast) -- this mirrors the
pre-existing precedent in `vde_calc.extract_cycle_arrays`/`compute_vde_net`
(missing columns are handled; NaN/non-finite/non-monotonic values raise).
No new philosophy was invented; the existing one was extended to the new
physical fields 9B introduced.

## JSON round-trip on a real result

`tests.test_vehicle_demand_integration.JsonRoundTripCanonicalResultTests`
takes a full `calculate_vehicle_demand` result from VDE-QA-001 (both TOTAL
and NET, with rolling/aero/residual all populated) through
`to_serializable()` -> `json.dumps()` -> `json.loads()` ->
`vehicle_demand_result_from_dict()` and asserts equality with the original.

## Interactive-performance sanity result

A full `calculate_vehicle_demand` call on the complete 2139-point
FTP75_HWFET trace (both TOTAL and NET) completes in a small fraction of a
second on this machine -- verified against a generous 2-second bound (no
benchmark framework, per Sec 33). No N^2 loop, no per-timestep DB query, and
no per-timestep serialization exist anywhere in the engine or adapter --
all math is vectorized numpy over the whole trace at once.

## Tests added

`tests/test_vehicle_demand_integration.py`, 42 tests across 15 groups:
`CanonicalAdapterTests` (4), `EndToEndCanonicalFlowTests` (1),
`EpaPhaseWeightedReconciliationTests` (3), `WltpReconciliationTests` (1),
`AvailabilityMatrixTests` (4), `ResidualSemanticsTests` (4),
`AmbientAvailabilityMatrixTests` (3), `ProvenanceAndWarningsTests` (3),
`ZeroValueEdgeCaseTests` (3), `InvalidPhysicalInputTests` (6, one
parametrized over 2 mass cases), `NonFiniteInputTests` (4),
`InvalidCycleConsistencyTests` (3), `MalformedProfileRegressionTests` (1),
`JsonRoundTripCanonicalResultTests` (1), `PerformanceSanityTests` (1).

## Full test count/result

- Before this package (Sprint 9B baseline): 1196 tests, 1194 passing, 2
  known pre-existing failures.
- After this package: **1238 tests** (1196 + 42 new), **1236 passing**, the
  same 2 known pre-existing failures (identical test names/tracebacks in
  `test_vde_request_resolver.py`), **zero new failures**.

## Known pre-existing failures

Unchanged: `tests/test_vde_request_resolver.py`, 2 failures (component-
snapshot/axle-hubs), not touched by this package.

## Any 9A/9B contract issue discovered

None requiring a stop-and-report, and no contract field/type/validation was
changed. One architectural note worth carrying into 9D planning (not a
contract issue, a dependency-direction one):

**Known architectural coupling for 9D to watch**:
`vehicle_demand/adapters.py` imports `resolve_roadload_boundaries`,
`resolve_transmission_boundary`, and `build_vehicle_label` directly from
`comparison_report_service.py`, to reuse the exact existing TOTAL/NET/label
logic rather than duplicate it. This is a one-directional dependency
(`vehicle_demand.adapters` -> `comparison_report_service`) that is entirely
safe today, because `comparison_report_service.py` does not import anything
from `vehicle_demand`. If Sprint 9D wires Comparison Report to consume the
Vehicle Demand engine by importing `vehicle_demand` (the package) from
`comparison_report_service.py`, a true circular import would only occur if
that import path eventually reaches `vehicle_demand/adapters.py` -- which it
will not by default, because **`adapters.py` is deliberately not re-exported
from `vehicle_demand/__init__.py`** (only `contracts`/`engine`/
`serialization` are). Importing the core package (`from src.vde_core.
vehicle_demand import calculate_vehicle_demand`) will therefore never pull
in `comparison_report_service.py`. If 9D specifically needs to import
`vehicle_demand.adapters` FROM `comparison_report_service.py` directly, the
simplest fix at that point is a local/deferred import inside the specific
function that needs it; a more permanent fix would be hoisting
`resolve_roadload_boundaries`/`resolve_transmission_boundary` (which are
already comparison-agnostic, operating only on a raw `vde_row` mapping) out
of `comparison_report_service.py` into a shared, lower-level module -- out
of scope for 9C (Sec 41 "no broad legacy cleanup"), flagged here for 9D to
decide with real requirements in hand.

## Commit(s)

`d35b994a` on branch `sprint-9a-vehicle-demand-contracts` - "feat(vehicle-demand):
add Sprint 9C canonical adapter and hardening". Sprint 9C continues on the
same branch as 9A/9B, per this package's own Sec 1 instruction not to
rename the branch.

## Is the Vehicle Demand Core API safe to freeze?

**Yes.** Reviewed against Sec 29's four questions using real scenarios, not
just the frozen contracts in the abstract:

- **Can Comparison consume these without physics knowledge?** Yes -- three
  calls (`build_vehicle_demand_request`, `resolve_vehicle_demand_cycle`,
  `calculate_vehicle_demand`) given only a `vde_row` Comparison already
  fetches.
- **Can Quick Scenario construct a request without bypassing the adapter?**
  Yes -- `temporary_transmission` and `ambient` are explicit override
  parameters on `build_vehicle_demand_request` precisely for this; reviewed,
  not implemented (Sec 36).
  **Can Powertrain consume tractive power later?** Yes --
  `tractive_power_W`/`tractive_force_N` remain plain, unopinionated fields
  with no efficiency/gear/motor/regen assumption anywhere near them.
- **Can the contracts serialize cleanly?** Yes, proven with a real,
  fully-populated QA-derived result in this package, not only synthetic 9A/
  9B examples.

No `EnergyMode`/`RoadloadBasis` value was added or changed. No
`KinematicPhase`, VSP, or aggressiveness classification was introduced. No
persistence, DB write, migration, FastAPI/MCP/RAG code, or Comparison/Quick
Scenario/Powertrain implementation exists anywhere in this package.

## Is 9D safe to implement as a pure consumer?

**Yes**, with the one architectural note above carried forward: 9D should
import `vehicle_demand` (the core package) freely, and should treat
`vehicle_demand.adapters` as available but should re-check the import
direction if it ever needs `comparison_report_service.py` to import
`vehicle_demand.adapters` specifically (rather than the reverse, which is
what exists today). Vehicle Demand Core API: **FROZEN**.
