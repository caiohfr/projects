# Sprint 10C — Quick Tire / RRC / Pressure Resolution

## Scope and outcome

Sprint 10C makes Tire the third executable Vehicle Quick domain. A Quick
Scenario now resolves a temporary physical row in this strict order:

`source -> canonical Mass -> canonical Tire -> canonical Aero -> canonical VehicleDemandRequest -> frozen Vehicle Demand Core`

The source VDE row, source Comparison/FuelCons identity, and selected Tire DB
record are copied and never persisted or mutated. There is no Streamlit,
Quick UI, PSE, ML, Technology Delta, Save/Promote, or schema work in this
package.

## Supported source/change matrix

| Tire source | None | Target RRC | RRC Delta | Tire Improvement % | Pressure Delta |
|---|---:|---:|---:|---:|---:|
| Current/resolved Tire | yes | yes | yes | yes | yes |
| Tire DB | yes | no | no | yes | yes |

Exactly one transformation may follow source selection. The contract rejects
Tire DB + Target RRC, Tire DB + RRC Delta, Improvement + Pressure, Target RRC
+ Pressure, and every other attempt to populate fields for multiple
transformations. Zero remains a real value for Target RRC-neutral,
Improvement 0%, and Pressure Delta 0 cases.

Tire DB + Improvement is deliberately two canonical stages: first
`TIRE_DB_LOOKUP`, then `TIRE_IMPROVEMENT_PCT` against the resolved DB tire
state. It is not treated as two transformations because Tire DB is source
selection, not a transformation.

## Canonical ownership and adapters

Quick owns only request translation, stage ordering, readiness, and temporary
row composition. Physical behavior remains in these existing owners:

- `resolve_mass_proposal()` selects EPA/WLTP calculation and tire load mass.
- `resolve_tire_proposal()` owns source RRC, improvement convention, direct
  Target RRC behavior, pressure estimation, load scaling, and Tire ABC.
- `calculate_vehicle_tire_abc()` remains the underlying Tire model owner.
- `get_tire_by_id()` remains the Tire DB repository boundary.
- `tire_reference_pressure_psi()` is a small shared unit-normalization adapter
  exposed from the canonical Tire resolver; it contains no pressure/RRC model.
- `cdA_to_C()` remains Aero's CdA-to-C owner.
- `resolve_roadload_boundaries()`, `build_vehicle_demand_request()`, and
  `calculate_vehicle_demand()` remain the TOTAL/NET and frozen-core owners.

RRC Delta is translated once at the Quick service boundary: the canonical
INHERIT call resolves the reference RRC, the requested delta identifies a
target, and canonical `TIRE_TARGET_RRC` resolves all physical state. No Tire
ABC/load formula is present in Quick.

The canonical Tire resolver returns `tire_delta_abc`; Quick applies that
canonical delta to the authoritative TOTAL row. It never derives NET from
TOTAL or TOTAL from NET. Missing boundaries remain missing and numeric zero
remains zero.

## Pressure behavior and provenance

Pressure requests preserve independent front and rear values; a missing rear
delta means the front delta applies to both axles. Quick does not average the
deltas.

Reference selection is explicit:

1. an explicit `USER_PROVIDED` common reference is used when supplied;
2. Tire DB selection prefers that Tire DB record's characterized reference;
3. Current Tire uses its saved front/rear references, with its Tire DB
   reference as a fallback;
4. otherwise Tire readiness is `MISSING`. No default psi is assumed.

The resolution output records `SOURCE` (including DB-sourced reference) or
`USER_PROVIDED` provenance and exposes reference/requested front and rear
pressures.

Model selection delegates to the existing canonical paths:

- SAE records use canonical DB lookup and the richer `SAE_FULL` model;
- ISO DB records use canonical DB lookup and its approved reference-point
  pressure estimate;
- Current/reference-point records use canonical pressure-only
  `TIRE_TARGET_RRC` with the target intentionally blank;
- a manually supplied reference remains an input to the same canonical path,
  never a Quick-only pressure relationship.

## Mass → Tire dependency

Mass resolves before Tire. The Mass stage copies the canonical
`inertia_class`, `tire_load_mass_basis`, and `tire_load_mass_used_kg` into the
working row. Tire receives that row as `current_snapshot`, so load-dependent
Tire conversion uses the resolved Quick scenario mass rather than the source
vehicle's stale mass. Resolver-level tests independently call the same Mass
and Tire resolvers and compare the resulting load mass and Tire ABC.

## Readiness and parity boundary

Tire outcomes map canonical `Missing` to `DomainReadiness.MISSING` and
canonical `Invalid` to `DomainReadiness.INVALID`; canonical OK/Review physical
outcomes remain ready with their explanatory issues. If any requested Mass,
Tire, or Aero domain is not ready, no partial physical result or Vehicle
Demand result is returned.

Parity is asserted at the resolver physical-state boundary, not by comparing
unrelated downstream VDE engines. Independent canonical calls generate
expected RRC, reference RRC, pressure, Tire A/B/C, load mass, TOTAL ABC, and
NET ABC for:

- Current Target RRC with existing Tire ABC;
- Current Target RRC through canonical RRC-to-ABC;
- Current Improvement and pressure estimate;
- Tire DB None, Improvement, ISO pressure, and SAE pressure;
- Mass + Tire using the newly resolved mass.

Vehicle Demand integration is then checked by independently building the
request from the final canonical Tire-resolved temporary row. No Vehicle
Demand physics or mapping was changed.

## Verification

Pre-flight, before edits:

- full suite: 1,465 tests; 1,463 passed; two known unrelated
  `vde_request_resolver` failures;
- focused Quick/Tire/Vehicle Demand suites: 259 tests; 257 passed; the same
  two known failures.

Sprint 10C adds 34 dedicated Tire resolver/integration tests and five contract
boundary/readiness tests. Final verification after implementation:

- all Quick Scenario suites: 140/140 passed;
- focused Quick/Tire/VDE Setup/Vehicle Demand suites: 298 tests, 296 passed,
  with only the same two known unrelated failures;
- full suite: 1,504 tests in 856.311 seconds, 1,502 passed, with only the same
  two known unrelated failures.

## Deferred

Transformation stacking remains intentionally deferred: Target RRC +
Pressure, Improvement + Pressure, Tire DB + arbitrary Target RRC, and Tire DB
+ arbitrary RRC Delta are unsupported. PSE/ML/Technology Delta, Quick UI,
persistence, Save/Promote, and all other vehicle domains remain outside Sprint
10C.
