# Sprint 9A - Canonical Vehicle Demand Contracts

## Status

Completed. Contracts only - no physics engine, no Comparison Report change,
no UI change. Establishes the typed, Streamlit-free boundary the future
Vehicle Demand Engine (Sprint 9B+) will be built against.

## What this package is (and isn't)

Sprint 9A defines shape, not behavior:

```
Resolved VDE Scenario + Cycle + AmbientState
        -> VehicleDemandRequest
        -> [future Vehicle Demand Engine - NOT built here]
        -> VehicleDemandProfile / VehicleDemandSummary / VehicleDemandResult
```

No roadload/VDE/acceleration/cycle-distance physics was written or changed.
TOTAL and NET remain the pre-existing authoritative values resolved by
`vde_net_total_contract.canonical_vde_read()`; nothing in this package
reconstructs them from components, and nothing lets one silently stand in
for the other.

## Investigation before implementation

Before writing any contract, the repository was inspected for reusable
conventions (Sprint 9A Sec 3):

- **TOTAL/NET authority**: `src/vde_core/vde_net_total_contract.py`
  (Package 7G) is the single existing authority for TOTAL/NET semantics at
  the `vde_db` row level. Reused conceptually (RoadloadBasis mirrors its
  TOTAL/NET vocabulary with no fallback); not imported, since it operates on
  raw DB rows, a different layer.
- **Dataclass/enum convention**: `comparison_report_service.py` (Package 8A)
  is the most recent canonical example - frozen dataclasses, a shared
  `_TextEnum(str, Enum)` base with `__str__`, `Mapping[str, Any]` for
  flexible payloads, `tuple[...] = field(default_factory=tuple)` for
  warnings, and explicit `available`/status fields so "missing" is never
  confused with "zero". This package follows the same shape.
- **Existing A/B/C structs**: `RoadLoadComponent`/`EquivalentABC`
  (`roadload/models.py`) and `RoadloadBoundary`
  (`comparison_report_service.py`) each already define an equivalent A/B/C
  triplet for their own layer. None was imported directly - `RoadloadBoundary`
  lives in the Comparison layer, and importing it into a foundational
  contracts package would create a backwards dependency (Comparison should
  depend on Vehicle Demand later, not vice versa). A small local
  `RoadloadCoefficients` struct was defined instead, using the same N /
  N/kph / N/kph^2 units and naming style as the rest of the codebase.
- **Provenance vocabulary**: no single project-wide provenance framework
  exists. Each domain defines its own small `_TextEnum`
  (`fuel_energy.LhvBasis`, `fuel_energy.FuelConfidence`,
  `comparison_report_service.RevisionStatus`, etc.), which *is* the
  project's convention. A single `Provenance` enum was added to the new
  package (`SOURCE` / `RESOLVED` / `REGULATORY_REFERENCE` /
  `CANONICAL_ASSUMPTION` / `CALCULATED` / `ASSUMED` / `UNAVAILABLE`),
  reused across `AmbientState`'s per-field basis and the
  request/summary-level provenance maps, rather than one enum per field.
- **Ambient/temperature/pressure/density**: confirmed absent from the
  codebase entirely (`grep -i "temperature|pressure|air_density|ambient"`
  across `src/vde_core` found no physical ambient-state concept anywhere) -
  `AmbientState` is wholly new, not a duplicate of something existing.
- **Energy unit convention**: `vde_setup_service.compute_vde_preview_from_inputs`
  and `comparison_metric_registry.py` (`vde_total`/`vde_net` ->
  `energy_mj_per_km`) confirm MJ/km as the dominant, canonical VDE energy
  rate unit project-wide. `VehicleDemandSummary` follows it
  (`vde_mj_per_km`), and cycle-total energy aggregates use MJ (the absolute
  form of the same convention), documented explicitly in the dataclass
  docstring.
- **Package layout precedent**: `src/vde_core/roadload/` (a `models.py` of
  pure dataclasses + an `__init__.py` re-export list, no Streamlit/DB
  imports) was used as the direct structural template for the new
  `src/vde_core/vehicle_demand/` package.

## Files created

- `src/vde_core/vehicle_demand/__init__.py` - re-exports the public contract
  and serialization surface.
- `src/vde_core/vehicle_demand/contracts.py` - `Provenance`, `RoadloadBasis`,
  `EnergyMode`, `AmbientState`, `RoadloadCoefficients`,
  `VehicleDemandRequest`, `VehicleDemandProfile`, `VehicleDemandSummary`,
  `VehicleDemandResult`. Pure dataclasses/enums; zero physics, zero
  Streamlit/DB imports.
- `src/vde_core/vehicle_demand/serialization.py` - `to_serializable()` (a
  generic dataclass/Enum/Mapping/sequence -> JSON-safe recursive converter,
  NaN/numpy-scalar aware) and one typed `*_from_dict()` per contract, for
  the JSON/API/MCP-ready application boundary (Sec 10).
- `tests/test_vehicle_demand_contracts.py` - 23 tests (see below).

No existing file was modified. No DB schema, no migration, no UI/Streamlit
code, no `vde_db`/`fuelcons_db` change.

## Final contract shapes

- **`AmbientState`** - `temperature_C` / `pressure_kPa` / `air_density_kg_m3`
  (all optional) plus a `Provenance` basis per field. No `rho = p/(R*T)`
  correction is implemented; the shape only prepares for it.
- **`RoadloadBasis`** - `TOTAL` / `NET` only. No third value, no fallback.
- **`EnergyMode`** - `IDLE` / `TRACTION` / `COASTING` / `BRAKING` only. No
  `KinematicPhase` enum was added (Sec 4.4 explicitly said this is optional
  and warned against adding thresholds/classification just to justify it;
  none of the 9A deliverables needed it, so it was left for backlog).
- **`VehicleDemandRequest`** - composes scenario identity
  (`source_kind`/`vde_id`/`fuelcons_id`/`label`), cycle identity
  (`cycle_name`/`cycle_source`/`cycle_version` - a name resolvable via
  `cycles.py`, never an embedded trace), `test_mass_kg`, authoritative
  `roadload_total` (required) and `roadload_net` (optional, independent),
  optional `rrc_n_per_kn`/`cda_m2`, an `AmbientState`, a `provenance` map,
  and `model_version` (the upstream physics version, distinct from
  `contract_version`).
- **`VehicleDemandProfile`** - one object per `RoadloadBasis` (never a
  combined TOTAL+NET object, per Sec 6's explicit preference). Time-series
  tuples for time/speed/accel/forces/powers plus per-sample `EnergyMode`;
  `known_rolling_force_N`/`known_aero_force_N`/`residual_roadload_force_N`
  are optional and independently nullable. `__post_init__` rejects any
  provided series whose length doesn't match `time_s` - both for the
  required series and for whichever optional ones are supplied.
- **`VehicleDemandSummary`** - one object per `RoadloadBasis`. Energy
  aggregates in MJ, rate in `vde_mj_per_km`; every energy field is a
  non-negative magnitude per Sec 7 (direction lives in the field name, e.g.
  `braking_energy_required_MJ`, never a negative aggregate).
  `availability`/`warnings`/`provenance` mirror the
  `ComparisonItem`/`ComparisonProvenance` shape from Package 8A.
- **`VehicleDemandResult`** - `total_summary` (required, must be
  `RoadloadBasis.TOTAL`) + `net_summary` (optional, must be
  `RoadloadBasis.NET` when present) + free-form `metadata`. Does not embed a
  `VehicleDemandProfile` - profiles are computed on demand and are never
  persisted (Sec 9), so bundling one in would invite accidental
  persistence/serialization of a time series this contract doesn't own the
  lifecycle of.

## Serialization strategy

`to_serializable()` is one small generic recursive function (dataclass ->
dict of its fields, `Enum` -> `.value`, `Mapping` -> dict,
`tuple`/`list`/`frozenset`/`set` -> list, numpy scalar -> native Python,
`NaN` -> `None`), not a project-wide serialization framework. Reconstruction
uses one hand-written, explicitly typed `*_from_dict()` function per
contract (not a generic reflective deserializer) so each contract's own
enum/optional/required rules stay visible in code rather than inferred.
Every `to_serializable()` output in the test suite is additionally passed
through `json.dumps()` to prove it is actually JSON-safe, not merely
"looks like a dict".

## Deviations from the package spec

None materially. Two small, in-spirit adaptations:

1. `KinematicPhase` was not added at all (Sec 4.4 marked it fully optional
   and warned against manufacturing thresholds just to include it - nothing
   in 9A's other deliverables needed it).
2. `AmbientState`'s basis fields and the request/summary provenance maps
   share one `Provenance` enum rather than each having its own bespoke
   basis enum, because Sec 4.1's suggested value set
   (`SOURCE`/`RESOLVED`/`REGULATORY_REFERENCE`/`CANONICAL_ASSUMPTION`/
   `CALCULATED`) and Sec 12's provenance vocabulary
   (`SOURCE`/`KNOWN`/`RESOLVED`/`CALCULATED`/`ASSUMED`/`UNAVAILABLE`) are
   clearly describing the same concept (Sec 12's own worked example -
   `temperature = REGULATORY_REFERENCE`, `rho = CALCULATED` - uses Sec 4.1's
   vocabulary directly). One small enum was judged truer to "reuse first,
   avoid duplicated provenance" (Sec 16) than two overlapping ones.

## Tests added

`tests/test_vehicle_demand_contracts.py`, 23 tests in 6 groups:

- `RoadloadBasisAndEnergyModeContractTests` (3) - enum value stability and
  JSON-string-valued behavior.
- `VehicleDemandProfileShapeValidationTests` (4) - matching lengths accepted;
  a mismatched required series and a mismatched optional series are both
  rejected; an entirely absent optional series is allowed.
- `TotalNetDistinctnessTests` (5) - TOTAL and NET summaries carry distinct
  values; `VehicleDemandResult` rejects a `total_summary` that isn't TOTAL or
  a `net_summary` that isn't NET; a request/result with no NET stays `None`
  rather than falling back to TOTAL.
- `ZeroVsMissingTests` (4) - `0.0` survives as `0.0` (not `None`) through
  construction and serialization; `None` stays `None` (not `0.0`); `NaN`
  serializes to `None`, never to `0`.
- `SerializationRoundTripTests` (7) - every contract (`AmbientState`,
  `VehicleDemandRequest` with and without NET roadload,
  `VehicleDemandProfile`, `VehicleDemandSummary`, `VehicleDemandResult` with
  and without a NET summary) round-trips through `to_serializable()` ->
  `json.dumps()` -> `*_from_dict()` and compares equal to the original.

## Full test count / result

Ran the complete existing suite (`python -m unittest discover -s tests -p
"test_*.py"`) before and after this package's changes:

- Baseline (pre-9A, this repo state, before `test_vehicle_demand_contracts.py`
  existed): 1149 tests, 1147 pass, 2 known pre-existing failures in
  `test_vde_request_resolver.py` (component-snapshot/axle-hubs, unrelated to
  Vehicle Demand or Comparison).
- After 9A: 1172 tests (1149 + 23 new), 1170 pass, the same 2 known
  pre-existing failures (identical test names/tracebacks), zero new
  failures.

## Known pre-existing failures

`tests/test_vde_request_resolver.py` - 2 failures, component-snapshot/
axle-hubs related, pre-dating this package (also recorded in Package 8E's
closure report). Not touched or investigated further here; out of scope for
a contracts-only package.

## Commit

See the Sprint 9A closing commit on branch `sprint-9a-vehicle-demand-contracts`.

## Safe to freeze and proceed to 9B?

Yes. No physics from 9B was implemented (no ambient correction engine, no
component decomposition, no classification logic beyond the closed
`EnergyMode` enum itself, no persistence). Existing VDE Setup, Comparison
Report, TOTAL/NET, and Save/Reload behavior is unchanged - no existing file
was modified, only new, additive modules were created. The contracts are
ready for a Sprint 9B Vehicle Demand Engine to populate them.
