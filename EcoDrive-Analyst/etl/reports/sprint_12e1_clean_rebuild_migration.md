# Sprint 12E.1 — Clean Rebuild Migration Rehearsal

## Status: `CLEAN_REBUILD_READY — PROCEED_TO_INTEGRATION`

```text
Clean rebuild completed?                 YES
Runtime DB changed?                      NO

Canonical Program count                  3117
Vehicle Configuration count              10215
VDE count                                11630
RUN count                                30375
FuelCons count                           254

Legacy VDE records:
  reconstructed/retired                  4999
  irreproducible migrated                4
  unresolved                             0

Legacy FuelCons records:
  reconstructed/retired                  4999
  irreproducible migrated                5
  unresolved                             0

Duplicate legacy Program tree retained?  NO

Relationship failures                    0
Quarantined records                      77
Performance blockers                     0
User decisions required                  0
```

## Canonical population

| Table | Rows |
|---|---:|
| `program` | 3,117 |
| `vehicle_configuration` | 10,215 |
| `component_db` | 0 |
| `tire_db` | 1 |
| `component_instance` | 843 |
| `component_resolution` | 0 |
| `vde` | 11,630 |
| `run` | 30,375 |
| `fuelcons` | 254 |
| `fuelcons_run_adoption` | 254 |
| `vde_component_resolution` | 0 |

## Exceptions

- **EPA identity quarantine:** 77 of 30,194 source rows have a year-like represented make and were not loaded.
- **Scenario parent:** 1 migrated scenario has an intentionally unresolved parent VDE state. It is attached to the safely matched EPA Program through a distinct scenario configuration; both possible refreshed parent states are retained in provenance.
- **EPA FuelCons:** 4,999 legacy baseline results are intentionally retired. Current EPA result fields remain RUN evidence because no modern FuelCons pacification rule is approved.
- **JRC:** 249 source-scoped unresolved identities are loaded with supported SI fields and declared OEM results; no JRC↔EPA merge is attempted.
- **EEA:** 10,833,597 monitoring rows remain in analytical source storage and add zero operational Program/VDE rows.
- **Approved JSON correction:** non-standard `NaN` in FuelCons 5018 is written as JSON `null`; original source text remains in RUN provenance.

## Program and legacy disposition

Valid EPA fallback Programs: 5,715; after SAFE-only consolidation: 2,868; JRC Programs: 249; total: 3,117. The previous parallel legacy Program tree is absent.
All 10,007 legacy VDE/FuelCons rows have an explicit disposition. Four scenario VDEs, five scenario/ML FuelCons records, and one local Tire record are retained as irreproducible state.

## Validation and performance

Foreign-key/orphan failures: 0. Seven current query families were measured; blockers: 0. Two clean rebuilds produced matching counts and relationship signature `CF6D28C3561724D641659619D9C192ECA705685FDAD7F240D500BDE63BEBE686`.

## Runtime safety

Runtime SHA-256 before: `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB`  
Runtime SHA-256 after: `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB`  
Byte-identical: **YES**. Both runtime databases were opened with SQLite `mode=ro` and `PRAGMA query_only=ON`. Output is restricted to `.db` files under `etl/data/staging/sprint_12e1_clean_rebuild/`.

## Evidence tiers

- **DIRECTLY_TESTED:** clean schema build; source-only Program count; complete legacy disposition; selective scenario/ML/tire migration; safe lineage; no baseline tree copy; FK/orphans; same-VDE adoption; NULL vs zero; master/snapshot immutability; approved JSON correction; EEA boundary; performance; clean-rebuild determinism; runtime fingerprints.
- **INDIRECTLY_COVERED:** 12C.3 SAFE consolidation evidence and 12D physical constraints.
- **INSPECTION_SUPPORTED:** the four scenario and five later FuelCons records are the only post-import VDE/FuelCons additions in the supplied runtime DB.
- **GAP:** EPA FuelCons pacification, application write-adapter integration, browser smoke and production cutover remain future work.

## Reproduction

```powershell
python etl/scripts/sprint_12e1_clean_rebuild_migration.py --rebuild
python -m unittest discover -s etl/tests -p "test_sprint_12e1*.py" -v
```

## USER_DECISION_REQUIRED

None.

