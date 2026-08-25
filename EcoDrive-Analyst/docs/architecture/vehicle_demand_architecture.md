# Vehicle Demand Architecture

## Status

Frozen (Sprint 9A-9D; this document is the Sprint 9E documentation
checkpoint). Vehicle Demand Core API and its Comparison integration are
both frozen -- see [Sprint 9 Closure](../sprints/SPRINT_9_VEHICLE_DEMAND_CLOSURE.md)
for the full close-out record and per-package detail
([9A](../sprints/PACKAGE_9A_VEHICLE_DEMAND_CONTRACTS.md),
[9B](../sprints/PACKAGE_9B_VEHICLE_DEMAND_ENGINE.md),
[9C](../sprints/PACKAGE_9C_VEHICLE_DEMAND_HARDENING.md),
[9D](../sprints/PACKAGE_9D_COMPARISON_ENERGY_DRIVERS_INTEGRATION.md)).

## What this layer is

Vehicle Demand is the canonical, auditable representation of *what the
vehicle requires at the wheels*, derived from the project's existing
authoritative VDE/roadload physics -- it is not a new/second VDE model.
Powertrain (how that demand is supplied) is an explicitly separate,
not-yet-built concern; Vehicle Demand never assumes efficiency, gears, a
motor/engine, or regen capture.

## Layer diagram

```text
Resolved VDE Scenario / ComparisonItem (existing EcoDrive data)
        |
        v
Vehicle Demand adapter(s)               (src/vde_core/vehicle_demand/adapters.py,
        |                                 src/vde_app/comparison_vehicle_demand_viewmodels.py)
        v
VehicleDemandRequest                    (frozen contract, Sprint 9A)
        |
        v
Vehicle Demand Core (engine.py)         (frozen physics, Sprint 9B; hardened Sprint 9C)
        |
        +--> VehicleDemandProfile        runtime-only, NEVER persisted
        |
        v
VehicleDemandSummary / VehicleDemandResult   canonical downstream representation
        |
        v
Comparison presentation layer            (src/vde_app/comparison_vehicle_demand_viewmodels.py)
        |
        v
Energy Drivers -- Vehicle Demand Summary (Sprint 9D)
```

`VehicleDemandProfile` is the full time-resolved physical result (one
per `RoadloadBasis`). It is a **runtime object** -- built on demand for
whatever cycle trace is handed to `build_vehicle_demand_profile`, never
written to any table, and not part of the JSON-serializable application
boundary's persisted state.

`VehicleDemandSummary`/`VehicleDemandResult` are the **canonical
downstream representation**: cycle-level aggregates (energy in MJ, VDE
rate in MJ/km) that a consumer (Comparison today; a future API/agent
boundary or Quick Scenario tomorrow) actually needs, and the only shape
that is JSON-serializable at the application boundary (see "JSON/API/
MCP-ready boundary" below).

## Packages and modules

```text
src/vde_core/vehicle_demand/
    contracts.py       AmbientState, Provenance, RoadloadBasis, EnergyMode,
                        VehicleDemandRequest, VehicleDemandProfile,
                        VehicleDemandSummary, VehicleDemandResult
    physics.py          air density, known rolling/aero force, EnergyMode
                        classification -- the only genuinely new Sprint 9
                        physics
    engine.py           build_vehicle_demand_profile / summarize_vehicle_
                        demand / calculate_vehicle_demand -- reuses
                        vde_calc.compute_vde_series/extract_cycle_arrays,
                        never reimplements road-load/inertial/tractive math
    adapters.py         build_vehicle_demand_request (raw vde_db row ->
                        VehicleDemandRequest), resolve_vehicle_demand_cycle
    serialization.py    to_serializable() and typed *_from_dict()
                        reconstructors -- the JSON boundary

src/vde_app/comparison_vehicle_demand_viewmodels.py
                        ComparisonItem -> VehicleDemandRequest (a second,
                        smaller adapter than vehicle_demand/adapters.py,
                        because ComparisonItem already carries pre-resolved
                        roadload/mass/RRC/CdA -- see Sprint 9D's own
                        completion doc for why these are deliberately two
                        adapters, not one)
```

Dependency direction is strictly one-way:
`src/vde_core/vehicle_demand/` has no Streamlit dependency and does not
import `comparison_report_service.py`. `comparison_vehicle_demand_
viewmodels.py` (Comparison-side) imports FROM `vehicle_demand` and FROM
`comparison_report_service` (for the `ComparisonItem` type only) -- never
the reverse. `vehicle_demand/adapters.py` itself does import
`comparison_report_service.resolve_roadload_boundaries`/
`resolve_transmission_boundary` to reuse the existing canonical TOTAL/NET
resolution logic rather than duplicate it; this is a one-directional,
currently-safe coupling documented in Sprint 9C/9D's completion docs, not
a cycle (`vehicle_demand/__init__.py` never imports `comparison_report_
service.py`, so importing the core Vehicle Demand package never pulls
Comparison in).

## Physical invariants (frozen)

These hold across every package in Sprint 9 and are the contract any
future consumer (Quick Scenario, Powertrain, an API/agent boundary) must
respect:

- **Authoritative Roadload remains authoritative.** TOTAL is always the
  stored coastdown ABC; NET is TOTAL minus the resolved transmission ABC,
  only when that boundary is resolved. Neither is ever derived from
  components.
- **Known Contribution + Residual = Authoritative Roadload.** This is an
  explicit closed-form identity (per timestep, in `VehicleDemandProfile`,
  and per cycle, in `VehicleDemandSummary`), never a forced-to-100%
  decomposition -- a component with no available model is omitted, not
  zero-filled.
- **Rolling is never inferred from A.** Known Rolling comes only from an
  explicit RRC + effective mass (the ISO MVP formula); if RRC is
  unavailable, Rolling is `UNAVAILABLE`, never guessed from the
  authoritative `A` coefficient.
- **Aero is never inferred from C.** Known Aero comes only from an
  explicit CdA + resolved air density; if either is unavailable, Aero is
  `UNAVAILABLE`, never guessed from the authoritative `C` coefficient.
- **Residual may be negative.** When known contributions exceed the
  authoritative roadload for part of a cycle, the residual is preserved
  as-is -- never `abs()`'d, clipped, or redistributed -- with a warning
  flagging the inconsistency.
- **Deceleration does not imply braking.** `EnergyMode` is classified from
  the sign of `tractive_power_W` alone (which already reflects both
  road-load and inertial force), never from the sign of acceleration.
- **Braking Energy Required is not recovered regen.** It is a wheel-side
  theoretical ceiling -- mechanical energy that must be removed at the
  wheels beyond natural vehicle resistance to follow the cycle -- not a
  capture/recovery model. Regen capture does not exist yet anywhere in
  this codebase.
- **TOTAL and NET never fall back to each other.** A missing NET stays
  `None`/unavailable; it is never silently replaced by TOTAL, and vice
  versa.
- **Zero and missing remain distinct everywhere.** A resolved `0.0` is a
  real value; an unresolved quantity is `None`. Neither is ever coerced
  into the other, at any layer including JSON serialization (`NaN` ->
  `None`, never `NaN` -> `0`).

## AmbientState boundary

Vehicle Demand already supports resolving air density two ways:

1. an explicit `air_density_kg_m3` (used as-is), or
2. calculated from `temperature_C` + `pressure_kPa` via
   `rho = p / (R_air * T)` (`R_air = 287.058 J/(kg*K)`, ISO 2533 standard
   atmosphere).

**What Sprint 9 does *not* do**: it does not apply any environmental/
temperature/pressure correction to the *authoritative* roadload ABC
itself. Ambient state only ever feeds the Known Aero calculation; the
authoritative TOTAL/NET coefficients used everywhere else are exactly the
stored/resolved values, unmodified by ambient conditions. No regulatory-
reference or canonical-assumption ambient default was implemented -- a
repo-wide audit (Sprint 9B) found no existing standard-atmosphere constant
anywhere in the project, so Aero Known simply stays `UNAVAILABLE` when no
ambient data is supplied (which is the case for every Comparison-sourced
request today -- `vde_db` has no ambient columns).

This is a deliberate scope boundary, not an oversight: it is exactly the
gap a future "roadload condition correction" capability (if ever built)
would need an explicit owner and design for, and exactly the boundary
Sprint 10's Quick Scenario can extend into (see "Future interfaces"
below) without touching Vehicle Demand physics itself.

## JSON/API/MCP-ready boundary (architecture readiness only)

The contracts in `src/vde_core/vehicle_demand/contracts.py` are:

- typed (frozen dataclasses + enums, no `dict[str, Any]` at the
  boundary);
- Streamlit-independent (no `import streamlit` anywhere in the package);
- JSON-serializable at the application boundary via
  `serialization.to_serializable()` / the typed `*_from_dict()`
  reconstructors, which correctly handle `Enum`, numpy scalars, and `NaN`
  (-> `None`, distinct from `0`).

Architecturally, this means:

```text
JSON / MCP / API / UI
        |
        v
adapter / validation
        |
        v
typed VehicleDemandRequest
        |
        v
frozen Vehicle Demand engine (deterministic)
        |
        v
typed VehicleDemandResult
        |
        v
adapter / serialization
        |
        v
JSON / UI / DB
```

**No FastAPI application, MCP server, or agent-tool wiring exists.** This
section documents that the contracts do not *block* building one later --
it is not a claim that one has been implemented.

## Known limitations

- Known decomposition covers Rolling + Aero only. No Brake, Transmission,
  Axle, Hub, or Parasitic attribution exists in Vehicle Demand (that data
  may exist elsewhere in the project for other purposes, e.g. Comparison's
  NET-from-transmission resolution, but Vehicle Demand's own Known/
  Residual split does not use it).
- Residual may therefore contain brake, driveline, bearing, parasitic, and
  any other unattributed roadload effects -- it is explicitly labeled
  "Residual / Unattributed Roadload", never a specific named component.
- Vehicle Demand does not model powertrain efficiency, fuel/electrical
  energy, or CO2 -- those remain owned by the existing Fuel/PSE layer.
- Braking Energy Required is wheel-side only; no regen capture/recovery
  model exists.
- `AmbientState` supports Aero-density calculation but not authoritative
  roadload condition correction (see above).
- `VehicleDemandProfile` is not persisted; every profile is computed on
  demand.
- Stored synthetic QA fixture `vde_total_mj_per_km`/`vde_net_mj_per_km`
  values are not guaranteed to be physically derived from the fixture's
  own ABC/mass/cycle -- see "QA persisted-VDE debt" below.
- `KinematicPhase` (STOPPED/LAUNCH/ACCELERATION/CRUISE/DECELERATION), VSP,
  and driving-aggressiveness classification are deferred; `EnergyMode`
  (IDLE/TRACTION/COASTING/BRAKING) is the only classification that exists.

## QA persisted-VDE debt

Sprint 9C found, and verified empirically, that `qa_mock_data.py`'s
synthetic seed rows' persisted `vde_total_mj_per_km`/`vde_net_mj_per_km`
columns are **not** physically derived from that same row's own
`coast_A_N`/`B`/`C`/`test_mass_kg`/cycle -- running VDE-QA-001's own ABC
through the project's real on-demand VDE calculation
(`comparison_report_service.resolve_cycle_vde_results`) gives ~0.31 MJ/km,
while the row's *stored* value is 1.24 MJ/km (~4x off). These fixture
columns were chosen independently, for other testing purposes (e.g.
Comparison UI sorting/delta behavior), not as physically self-consistent
golden values.

**Consequence for any future physics regression test**: never use a QA
fixture's stored `vde_total_mj_per_km`/`vde_net_mj_per_km` as a golden
physical output to reconcile new physics against. The correct golden is
the project's own canonical on-demand VDE calculation
(`comparison_report_service.resolve_cycle_vde_results`, or
`vde_calc.compute_vde_net` directly for a single trace/phase), computed
fresh from the same ABC/mass/cycle under test -- this is what Sprint 9B/9C
Vehicle Demand reconciliation tests already do.

Fixtures were not modified in Sprint 9 (deliberately -- see each
package's own Sec 41/58/59 "no broad cleanup" boundary). This paragraph is
the backlog note for whoever eventually decides fixture correction is
warranted.

## Future interfaces

**Sprint 10 -- Interactive Quick Scenario.** Expected flow:

```text
Existing Scenario
        |
        v
Temporary Overrides           (Mass / CdA / RRC first; Temperature/Pressure
        |                       once an owner for condition correction exists)
        v
VehicleDemandRequest
        |
        v
Frozen Vehicle Demand Core     (unmodified)
        |
        v
VehicleDemandResult
```

Quick Scenario must not create new physics -- it constructs a
`VehicleDemandRequest` from a temporarily-overridden resolved scenario and
consumes the same frozen `calculate_vehicle_demand()` every other
consumer uses. `build_vehicle_demand_request`'s `temporary_transmission`
parameter and `comparison_vehicle_demand_viewmodels`'s `ambient` override
parameter already exist specifically as the override hooks this future
capability needs.

**Sprint 11+ -- Powertrain Scenario L0 / PWT + Comparison integration.**
`VehicleDemandProfile.tractive_power_W`/`tractive_force_N` remain plain,
unopinionated fields with no efficiency/gear/motor/regen assumption
anywhere near them -- they are the intended wheel-side boundary a future
Powertrain layer will consume as its own input, without Vehicle Demand
ever needing to know about engines, motors, or batteries.
