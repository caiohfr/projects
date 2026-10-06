# Sprint 12 Component Population Persistence v1

The canonical VDE schema is unchanged. Component population uses the existing
historical projection fields and the canonical `component_resolution` /
`vde_component_resolution` audit layer.

## Projection modes

`MACRO_DECOMPOSITION` uses the following explicit semantic mapping:

| Existing VDE field | Persisted meaning |
|---|---|
| `tire_A_final`, `tire_B_final`, `tire_C_final` | `ROLLING_MINOR` aggregate |
| `aero_C_coef_Npkph2` | `AERO.C` |
| `trans_A_coef_N`, `trans_B_coef_Npkph`, `trans_C_coef_Npkph2` | `DRIVETRAIN_AGGREGATE` |
| Brake and Parasitic fields | unresolved (`NULL`) |

The Transmission mapping supersedes the older blanket prohibition against
backfilling `vde.trans_A/B/C`, but only for rows explicitly tagged in
`vde.provenance_json` as `MACRO_DECOMPOSITION` with
`transmission_slot_semantics=DRIVETRAIN_AGGREGATE`. It is not a claim of
gearbox-only loss.

`FINE_SURROGATE_DECOMPOSITION` is reserved for a coherent, frozen and validated
fine projection. Sprint 12 Pass 1C.5 did not pass its fleet-validation gate, so
the v1.0 materialization produces no fine VDE-row projections. Conditional fine
matches remain supporting resolution/audit evidence.

## Protections

- Macro resolutions remain first-class, queryable `component_resolution` rows.
- `EDRIVE_AGGREGATE` is not relabeled as Transmission merely to fit the old
  schema.
- Transmission and Axle are not summed, rescaled, or subtracted from each other.
- Existing non-null component values are treated conservatively as canonical
  when field-level fidelity cannot be proven; the row is skipped and reported.
- `NULL` is unresolved; zero remains an explicit physical zero.
- Authoritative road-load, identity, RUN, FuelCons and vehicle-configuration
  data are outside the writable surface.
- All writes are first exercised on a real temporary DB copy and must be
  idempotent.

## Canonical promotion

The utility never promotes automatically. Promotion requires clean integrity,
protected-data, determinism, idempotency and test gates plus human review of the
manual engineering sample and any schema-projection limitations.
