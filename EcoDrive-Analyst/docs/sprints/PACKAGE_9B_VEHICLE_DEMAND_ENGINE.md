# Sprint 9B - Vehicle Demand Physics Engine

## Status

Completed. Implements the physics engine over the contracts frozen in
Sprint 9A. No Comparison Report change, no Quick Scenario, no Powertrain
physics, no persistence.

## Baseline

Branch origin: `sprint-9a-vehicle-demand-contracts` (commits `cae884c0`,
`91913320`). Baseline: 1172 tests, 1170 passing, 2 known pre-existing
failures in `tests/test_vde_request_resolver.py` (component-snapshot/
axle-hubs), untouched by this package.

## The core risk and how it was resolved

The package's own stated risk is introducing a second interpretation of
VDE. Before writing any physics, `src/vde_core/vde_calc.py::compute_vde_net`
was identified as the actual, single source of truth: every phase-specific
path (`phase_aggregation.epa_city_hwy_from_phase`,
`phase_aggregation.wltp_phases_from_phase`) and the whole-cycle fallback
(`vde_setup_service.compute_vde_preview_from_inputs`) all funnel through it.
Its math is: `v_kph = v*3.6`; `F_road = A + B*v_kph + C*v_kph^2`;
`a = np.gradient(v, t)`; `P = (F_road + mass*a) * v`; VDE (MJ/km) is
`trapezoid(clip(P, 0, None), t) / 1e6`, normalized by `trapezoid(v, t)/1000`
km. That instantaneous `P` is *already* exactly "tractive power" and its
positive-clipped integral is *already* exactly "positive tractive energy" --
Sprint 9B's requested quantities were not a new formula, they were named
components of a calculation that already existed.

Given that, `compute_vde_net` was refactored (non-behavior-changing) to
delegate its per-timestep math to two small, pure, newly extracted
functions -- `extract_cycle_arrays` (the existing column/finiteness/
monotonic-time validation) and `compute_vde_series` (`v_kph`, `F_road_N`,
`a_mps2`, `F_inertia_N`, `F_tractive_N`, `P_W`). `compute_vde_net`'s
external behavior is provably unchanged: all 16 pre-existing tests in
`tests/test_vde_physics_qa_vde_1b.py` pass unmodified against the refactored
code. The Vehicle Demand engine then calls these same two functions to build
its `VehicleDemandProfile` -- it does not, and structurally cannot, drift
into a second VDE, because the road-load/acceleration/tractive-power math it
uses *is* `compute_vde_net`'s math, not a copy of it.

## Files created/modified

Created:
- `src/vde_core/vehicle_demand/physics.py` -- air density resolution, known
  rolling force (ISO MVP formula), known aero force, `EnergyMode`
  classification. The only genuinely new physics in this package.
- `src/vde_core/vehicle_demand/engine.py` -- `build_vehicle_demand_profile`,
  `summarize_vehicle_demand`, `calculate_vehicle_demand`.
- `tests/test_vehicle_demand_engine.py` -- 24 tests (QA-1 through QA-16 plus
  4 direct `EnergyMode` epsilon tests).
- `docs/sprints/PACKAGE_9B_VEHICLE_DEMAND_ENGINE.md` (this file).

Modified:
- `src/vde_core/vde_calc.py` -- extracted `extract_cycle_arrays` and
  `compute_vde_series` from `compute_vde_net` (see above); `compute_vde_net`
  itself now calls them but returns byte-for-byte the same aggregate shape
  it always did.
- `src/vde_core/vehicle_demand/contracts.py` -- two docstring-only edits
  (see "Contract issue discovered" below); no field, type, or validation
  changed.
- `src/vde_core/vehicle_demand/__init__.py` -- exports the three new engine
  functions plus `VEHICLE_DEMAND_ENGINE_VERSION`; internal helpers in
  `physics.py` (e.g. `resolve_air_density`, epsilon constants) are
  deliberately not re-exported at the package level (Sec 35).

No DB schema, no migration, no UI/Streamlit file, no `vde_db`/`fuelcons_db`
change.

## Existing VDE/RRC/cycle functions reused

- `vde_calc.compute_vde_series` / `extract_cycle_arrays` (new, but a direct
  extraction of pre-existing `compute_vde_net` internals -- see above): all
  road-load, acceleration, tractive-force, and cycle-validation math.
- `roadload.tire_model.G_MPS2` (9.80665): reused directly for the known
  rolling force formula rather than redefining gravity locally.
- `roadload.tire_model.calculate_iso_tire_abc_for_single_tire`'s **formula**
  (`A = rr_n_per_kn * load_kN`, `B = C = 0`) was reused conceptually for
  `physics.known_rolling_force_N`, applied at the whole-vehicle level rather
  than called directly -- see "Rolling-energy implementation" below for why.

## Small refactor required to expose canonical physics

Only one: the `compute_vde_series`/`extract_cycle_arrays` extraction from
`compute_vde_net` described above. Nothing else in the existing codebase
needed to change to make this package possible.

## Final public Vehicle Demand API

`src/vde_core/vehicle_demand/__init__.py` now exports, in addition to the
9A contracts:

```python
build_vehicle_demand_profile(request, cycle_frame, roadload_basis) -> VehicleDemandProfile | None
summarize_vehicle_demand(profile, request) -> VehicleDemandSummary
calculate_vehicle_demand(request, cycle_frame) -> VehicleDemandResult
```

`cycle_frame` (a `t`/`v` DataFrame, the same shape `compute_vde_net` already
takes) is passed alongside `request` rather than being embedded in
`VehicleDemandRequest` -- 9A deliberately kept the request to a cycle
*reference* (`cycle_name`/`cycle_source`/`cycle_version`), not a trace, so
the engine's signature mirrors `compute_vde_preview_from_inputs(df_cycle,
leg, *, A, B, C, mass_kg, ...)`'s existing pattern of taking the trace as a
sibling argument. This also makes the engine trivially testable with
synthetic QA traces that don't correspond to any real named cycle.

`build_vehicle_demand_profile` returns `None` (never a fabricated
substitute) when the requested basis's roadload coefficients or
`test_mass_kg` aren't on the request. `calculate_vehicle_demand` raises only
when `TOTAL` -- required by the frozen 9A contract -- cannot be built; a
missing `NET` simply yields `net_summary=None`.

## Air-density strategy and defaults

Hierarchy implemented exactly as specified (Sec 9): explicit
`air_density_kg_m3` > calculated from `temperature_C` + `pressure_kPa`
(`rho = p / (R_air * T)`, `R_air = 287.058 J/(kg*K)`, ISO 2533 standard
atmosphere value) > unavailable. **No regulatory-reference or
canonical-assumption default was implemented.** A repo-wide search (`grep`
across `src/vde_core`, `data/standards`, `docs/architecture`) before writing
this module found zero existing standard-atmosphere constant or documented
regulatory ambient condition anywhere in the project -- inventing one now
would be exactly the kind of unbacked assumption Sec 10 warns against. Sec
10 explicitly pre-approves this outcome ("Aero Known = unavailable is
acceptable, do not block Vehicle Demand for it"), so Aero Known is simply
`UNAVAILABLE` (with a warning) whenever ambient data doesn't resolve a
density, and everything else (roadload, rolling, VDE) still computes
normally. This is a deliberate scope decision, not a stop condition -- flagged
here for visibility, not requiring confirmation before proceeding.

## Rolling-energy implementation/source

`physics.known_rolling_force_N(rrc_n_per_kn, mass_kg)` computes
`F = rrc_n_per_kn * (mass_kg * G_MPS2 / 1000)` -- speed-independent, exactly
mirroring the ISO MVP tire model's `A = rr_n_per_kn * load_kN` (`B=C=0`)
already canonical in `roadload/tire_model.py`. It is *not* a call into
`calculate_iso_tire_abc_for_single_tire` itself: that function operates on a
single tire's load (needs a front/rear axle split via
`front_weight_distribution_pct`), while `VehicleDemandRequest.rrc_n_per_kn`
(frozen in 9A) is one vehicle-level scalar with no axle split available. The
vehicle-level formula used here is the algebraic sum of the per-tire ISO MVP
formula across all four tires when one RRC applies vehicle-wide (axle loads
always sum to the full vehicle weight) -- mathematically identical to that
existing model for the case this contract can actually represent, not a
new/different rolling-resistance theory. Missing `rrc_n_per_kn` or
`test_mass_kg` makes rolling `UNAVAILABLE`, never inferred from `A`.

## Aero-energy implementation

`physics.known_aero_force_N`: `F_aero(t) = 0.5 * rho * CdA * v(t)^2`,
`P_aero(t) = F_aero(t) * v(t)`, integrated with `np.trapezoid` (uncapped, no
clipping -- Sec 24-27). `CdA = 0` is accepted as a valid known zero; a
missing `CdA` or an unresolved `rho` makes the whole series `None`
(`UNAVAILABLE`), never inferred from the authoritative `C` coefficient.

## Residual semantics

`residual_roadload_force_N(t) = authoritative_roadload_force_N(t) -
(known_rolling_force_N(t) if available else 0) - (known_aero_force_N(t) if
available else 0)` -- "residual absorbs whichever known contributions were
actually available" (the simpler of the two options Sec 15 offered), with
`VehicleDemandSummary.provenance["rolling"]`/`["aero"]` recording
`CALCULATED`/`UNAVAILABLE` so a caller can tell which contributions the
residual actually subtracted. Residual is never clipped, never
renormalized, and never redistributed; a negative residual (known
contributions exceeding the authoritative roadload) is preserved as-is and
surfaces a `VehicleDemandSummary.warnings` entry describing the physical
inconsistency (QA-12).

## `EnergyMode` epsilon values and rationale

`SPEED_EPSILON_MPS = 0.05` (0.18 km/h) -- below any real drive-cycle's
slowest moving segment; only absorbs trace/discretization noise at a
nominal stop. `POWER_EPSILON_W = 5.0` -- small relative to any physically
meaningful roadload/inertial power for a passenger vehicle at nonzero speed
(tens of W to tens of kW); only absorbs floating-point/`np.gradient`
discretization noise at true zero-crossings. Both are local constants in
`physics.py`, not a project-wide configuration system. Deceleration is
never auto-classified as `BRAKING`: only the sign of `tractive_power_W`
(which already reflects both road-load and inertial force) decides
`TRACTION`/`COASTING`/`BRAKING`; `IDLE` is decided purely by speed. QA-4
(gentle deceleration under natural-coast magnitude still yields `TRACTION`)
exists specifically to guard this invariant.

## Canonical VDE reconciliation results

Four dedicated tests in `CanonicalVdeReconciliationTests`
(`tests/test_vehicle_demand_engine.py`) compare
`VehicleDemandSummary.vde_mj_per_km`/`positive_tractive_energy_MJ`/
`distance_km` directly against `compute_vde_net()` on the same cycle/ABC/
mass, across a constant-speed cycle, a linear-acceleration cycle, an
81-point multi-phase trapezoidal cycle (accel/cruise/decel/idle), and both
`TOTAL` and `NET` bases. All four match to `assertAlmostEqual(..., places=9)`
-- effectively exact, as expected given the engine literally calls
`compute_vde_series`/`extract_cycle_arrays` rather than reimplementing them.
**Scope note**: this reconciles against whole-cycle `compute_vde_net`, not
the EPA 55/45 city/highway phase-weighted policy value
(`vde_setup_service`'s `total_mj_km`) -- `VehicleDemandSummary` has no
by-phase breakdown in the frozen 9A contract, so phase-weighted
reconciliation is out of scope here and is a natural candidate for a future
package if by-phase Vehicle Demand summaries are ever added.

## Synthetic physical QA cases/results

All 16 packaged QA cases (Sec 37-52) plus 4 `EnergyMode` epsilon boundary
tests, 20 total, all passing:

- QA-1 Constant speed: zero inertial force, `tractive_force ==
  roadload_force`, pure `TRACTION`, zero braking/positive-inertial energy.
- QA-2 Natural coast: built as an *exact* analytical solution of `m*a =
  -F_road` (constant `F_road`, linear `v(t)`), so `np.gradient` recovers the
  exact acceleration and tractive power is exactly ~0 (not merely
  tolerance-close) -- `COASTING` throughout.
- QA-3 Hard deceleration: `BRAKING`, positive `braking_energy_required_MJ`.
- QA-4 Gentle deceleration (gentler than natural coast): stays `TRACTION`,
  proving deceleration != braking.
- QA-5/6/7 Aero directionality: higher CdA / lower temperature / higher
  pressure each independently yield higher known aero energy; roadload
  energy is unaffected by ambient changes.
- QA-8 Rolling directionality: higher RRC yields higher known rolling force
  /energy, matched against the analytical ISO MVP formula.
- QA-9/10 Missing RRC/CdA: rolling/aero `UNAVAILABLE` (not inferred from
  A/C), roadload/tractive/VDE still compute.
- QA-11 Roadload closure: per-timestep `known_rolling + known_aero +
  residual == authoritative`, checked on the Profile arrays, not just the
  Summary.
- QA-12 Negative residual: over-attributed known contributions (tiny
  authoritative roadload, realistic RRC/CdA) preserve a negative residual
  with no clipping and a warning present.
- QA-13 Braking zero: a trace with no negative tractive power yields exactly
  `0.0` (not `None`).
- QA-14/15 TOTAL/NET independence: differing TOTAL/NET coefficients produce
  differing profiles/summaries; a missing NET returns `None`/`net_summary=
  None` with no fallback to TOTAL.
- QA-16 Serialization after physics: a real `calculate_vehicle_demand()`
  result round-trips through `to_serializable()` -> `json.dumps()` ->
  `vehicle_demand_result_from_dict()` and compares equal to the original.

## Tests added

`tests/test_vehicle_demand_engine.py`: 24 tests total (16 QA groups above
map to 20 test methods, plus 4 canonical-VDE-reconciliation tests).

## Full test count/result

- Before this package: 1172 tests, 1170 passing, 2 known pre-existing
  failures (Sprint 9A baseline).
- After this package (including the `vde_calc.py` refactor and the 24 new
  tests): **1196 tests, 1194 passing, the same 2 known pre-existing
  failures** (identical test names/tracebacks in
  `test_vde_request_resolver.py`), **zero new failures**. The 16
  pre-existing `test_vde_physics_qa_vde_1b.py` tests were run in isolation
  immediately after the `vde_calc.py` refactor and pass unmodified,
  confirming the refactor is behavior-preserving before any new code was
  built on top of it.

## Known pre-existing failures

Unchanged from Sprint 9A: `tests/test_vde_request_resolver.py`, 2 failures
(component-snapshot/axle-hubs), not touched by this package.

## Contract issue discovered in 9A

None requiring a stop-and-report. One clarification was made:
`VehicleDemandProfile.residual_roadload_force_N` and
`VehicleDemandSummary.residual_roadload_energy_MJ`'s docstrings (9A said
"every energy field is a non-negative magnitude") did not precisely describe
`residual`, which Sprint 9B's own spec (Sec 27) requires to be signed. The
frozen 9A contract has **no runtime validator** enforcing non-negativity
anywhere in `VehicleDemandSummary` or `VehicleDemandProfile` -- the
"magnitude" language was documentation-only, so nothing needed to change
behaviorally. The two docstrings were updated to state plainly that
`residual_*` is the one exception to that convention. No field, type,
default, or `__post_init__` validation changed.

## Commit(s)

See branch `sprint-9a-vehicle-demand-contracts` (Sprint 9B continues on the
same branch; no new branch was created since 9B is additive on top of the
frozen 9A contracts with no conflicting concerns). Commit hash recorded
below once committed.

## Safe to freeze and proceed to 9C?

Yes. No Comparison UI, Quick Scenario, Powertrain physics, regen/battery
concept, or DB persistence was introduced. `EnergyMode`/`RoadloadBasis`
remain exactly the four/two values frozen in 9A;
`KinematicPhase`/aggressiveness classification/VSP were not added.
TOTAL/NET independence, zero-vs-unavailable, and negative-residual-no-
clipping all hold under real synthetic physics, not just empty contracts.
Canonical VDE reconciliation is exact (to floating-point tolerance) because
the engine structurally cannot diverge from `compute_vde_net`'s math. Ready
for a 9C that wires this engine into a real caller (Comparison, Quick
Scenario, or an API/tool boundary) per whatever Sprint 9C's own scope turns
out to be.
