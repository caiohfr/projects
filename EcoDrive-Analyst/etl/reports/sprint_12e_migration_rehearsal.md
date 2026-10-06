# Sprint 12E — Canonical Migration Rehearsal

## Status: `MIGRATION_REHEARSAL_REVIEW_REQUIRED`

```text
Did migration build?            YES
Legacy VDE mismatches           0
Legacy FuelCons mismatches      1
Relationship/orphan failures    0
Runtime DB changed?             NO
Canonical Program count         8074
Vehicle Configuration count     15207
VDE count                       16629
RUN count                       35370
FuelCons count                  5253
Quarantined records             77
Performance blockers            0
User decisions required         1
```

## Canonical population

| Table | Rows |
|---|---:|
| `program` | 8,074 |
| `vehicle_configuration` | 15,207 |
| `component_db` | 0 |
| `tire_db` | 1 |
| `component_instance` | 843 |
| `component_resolution` | 0 |
| `vde` | 16,629 |
| `run` | 35,370 |
| `fuelcons` | 5,253 |
| `fuelcons_run_adoption` | 5,253 |
| `vde_component_resolution` | 0 |

The rehearsal retains every current legacy snapshot under its original positive integer ID. Refreshed EPA and JRC VDE/FuelCons IDs use deterministic negative ranges, so source additions cannot overwrite legacy history.

## Deviations and quarantines

- **EPA identity:** 77 of 30,194 rows were quarantined because `Represented Test Veh Make` is year-like. No Program, Configuration, VDE or RUN identity was invented for them.
- **Program consolidation:** 5,715 valid EPA fallback Programs became 2,868 Programs using only `SAFE_CONSOLIDATE`. `PROBABLE_CONSOLIDATE` was not applied.
- **EPA FuelCons:** no FuelCons was created per raw EPA row. All valid EPA rows became RUN evidence; adoption is deferred until an approved pacification rule exists.
- **JRC:** direct SI roadload and declared OEM result fields were loaded, while all 249 identities remain source-scoped and unresolved. Tire/component descriptors are unresolved Component Instances; no master with fabricated RRC was created.
- **EEA:** 10,833,597 monitoring rows remain outside runtime SQLite. The source fingerprint and analytical-storage boundary were validated; zero EEA domain rows were loaded.

- **JSON constraint conflict:** FuelCons `id=5018` contains non-standard `NaN` in `assumptions_json`. The canonical field uses strict JSON with `null`; the original text is retained in RUN provenance. This accounts for 1 compatibility mismatch and requires explicit contract approval before integration.

## Compatibility and invariants

Every one of the 101 VDE and 79 FuelCons legacy-facing columns was compared by ID, including NULLs. Parent lineage and FuelCons multiplicity remain exact. Foreign keys, orphans, same-VDE RUN adoption, duplicate legacy VDE states and master/snapshot immutability were checked after loading.

## Reproducibility and performance

The canonical relationship signature is `2965285913C5BC03A41C5FA5873A84C1C0D1B3050F142A19BF4933A81F5AC5DD`. A prior clean rebuild was available: YES; matching signature: YES. Public timestamps are fixed to `2026-09-10T00:00:00Z`.
Seven representative query families were measured over 20 warm runs. Performance blockers: 0. Detailed plans and timings are in `performance_results.csv`.

## Runtime safety

Runtime SHA-256 before: `CC27CDF22AE189E39F9F45609CCD23C650E822BB20655C552FF2BD89D97F4262`  
Runtime SHA-256 after: `CC27CDF22AE189E39F9F45609CCD23C650E822BB20655C552FF2BD89D97F4262`  
Byte-identical: **YES**. The production and QA databases were opened only through SQLite `mode=ro` plus `PRAGMA query_only=ON`. The output-path guard accepts only `.db` files below `etl/data/staging/` and rejects both runtime paths.

## Evidence tiers

- **DIRECTLY_TESTED:** runtime read-only mode; protected output guard; schema build; full-source migration completion; clean-rebuild signature; foreign keys/orphans; duplicate legacy states; VDE→FuelCons and VDE→RUN multiplicity; same-VDE FuelCons↔RUN invariant; optional Component Resolution; historical snapshot immutability; NULL/value compatibility across 180 fields; runtime fingerprints; EEA boundary.
- **INDIRECTLY_COVERED:** CDR field ownership and Sprint 12D physical constraint design.
- **INSPECTION_SUPPORTED:** application query inventory used to select the seven performance cases.
- **GAP:** this is not production cutover evidence; application write adapters, real interactive smoke and rollback operation belong to integration/cutover.

## Reproduction

```powershell
python etl/scripts/sprint_12e_migration_rehearsal.py --rebuild
python -m unittest discover -s etl/tests -p "test_sprint_12e*.py" -v
```

## USER_DECISION_REQUIRED

Approve or reject the explicit normalization of non-standard JSON `NaN` to JSON `null` for legacy FuelCons `id=5018`. Without approval, integration must not proceed.

