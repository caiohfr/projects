# Sprint 12C — Canonical Data Contract v1 Materialization & Compatibility Proof

## Recommendation: `READY_FOR_DDL`

The approved nine-entity CDR baseline can reproduce the current EcoDrive VDE/FuelCons application-facing surface without changing pages, physics, resolvers or runtime databases. The recommendation authorizes preparation/review of physical DDL; it does **not** authorize running a production migration.

## Compatibility result

- Legacy VDE: **5,003 rows × 101 fields**, with **0** reconstructed value mismatches.
- Legacy FuelCons: **5,004 rows × 79 fields**, with **0** reconstructed value mismatches.
- Relationship evidence: **2** VDE(s) have multiple FuelCons rows; **0** orphan FuelCons rows.
- Checks: **15** exact, **2** approved additive corrections, **1** deferred, **0** blockers.
- NULL counts were compared for every application-facing field; zero remained a value, never missing.

## Exact v1 field contract

`canonical_field_contract_v1.csv` defines **333 fields** across all nine entities. Each row includes semantic meaning, logical type, nullability, key/FK role, enum candidates, provenance expectations, legacy mapping, public-source mapping and compatibility projection.

The compatibility strategy is deliberately localized:

```text
PROGRAM + VEHICLE_CONFIGURATION + COMPONENTS + VDE + RUN + FUELCONS
                              ↓
                 compatibility projection
                              ↓
                 current vde_db/fuelcons_db shape
```

VDE stays wide and persisted. FuelCons stays the adopted comparison result. RUN lineage is additive and is not traversed by the compatibility read path.

## Staging materialization

| Entity | Legacy records | Public ETL records | Total staged |
|---|---:|---:|---:|
| PROGRAM | 4,957 | 1,026 | 5,983 |
| VEHICLE_CONFIGURATION | 4,996 | 1,540 | 6,536 |
| COMPONENT_DB | 0 | 0 | 0 |
| TIRE_DB | 1 | 0 | 1 |
| COMPONENT_INSTANCE | 0 | 0 | 0 |
| COMPONENT_RESOLUTION | 0 | 0 | 0 |
| VDE | 5,003 | 249 | 5,252 |
| RUN | 5,004 | 249 | 5,253 |
| FUELCONS | 5,004 | 249 | 5,253 |

EPA MY2026 populates provisional Program and Vehicle Configuration records only. EPA VDE/RUN/FuelCons materialization is deferred because this package has no approved imperial-to-SI ETL mapping and the VDE contract cannot substitute missing mass with zero. Raw EPA evidence remains preserved in the Sprint 12A/12B artifacts. JRC fields with explicit SI units populate VDE/RUN/FuelCons, while exact source-row semantics remain `PARTIAL`/`UNRESOLVED`.

## Relationship and semantic checks

| Check | Result | Evidence |
|---|---|---|
| VDE identity/count | `EXACT_EQUIVALENCE` | Read-only vde_db vs compatibility projection. |
| FuelCons identity/count | `EXACT_EQUIVALENCE` | Read-only fuelcons_db vs compatibility projection. |
| All VDE field values | `EXACT_EQUIVALENCE` | Compared 5,003 rows x 101 fields by ID. |
| All FuelCons field values | `EXACT_EQUIVALENCE` | Compared 5,004 rows x 79 fields by ID. |
| VDE-FuelCons linkage | `EXACT_EQUIVALENCE` | Exact ordered (fuelcons.id, vde_id) relationship signature. |
| FuelCons multiplicity per VDE | `EXACT_EQUIVALENCE` | Full per-VDE multiplicity distribution. |
| VDE parent lineage | `EXACT_EQUIVALENCE` | Exact (VDE id, parent id) signature including NULLs. |
| NULL behavior | `EXACT_EQUIVALENCE` | NULL counts compared for 180 application-facing fields; zero values were not treated as missing. |
| mass | `EXACT_EQUIVALENCE` | Exact value+NULL signature for: mass_kg, inertia_class, payload_kg, baseline_mass_kg, delta_mass_kg, tire_load_mass_basis, tire_load_mass_used_kg, test_mass_kg, test_mass_low_kg, test_mass_high_kg, test_mass_basis, trailer_mass_kg, mass_rule_status, mass_rule_notes |
| roadload ABC | `EXACT_EQUIVALENCE` | Exact value+NULL signature for: coast_A_N, coast_B_N_per_kph, coast_C_N_per_kph2, trans_A_coef_N, trans_B_coef_Npkph, trans_C_coef_Npkph2, brake_A_coef_N, brake_B_coef_Npkph, brake_C_coef_Npkph2, parasitic_A_coef_N, parasitic_B_coef_Npkph, parasitic_C_coef_Npkph2, baseline_A_N, baseline_B_N_per_kph, baseline_C_N_per_kph2, tire_A_final, tire_B_final, tire_C_final, trailer_A_coef_N, trailer_B_coef_Npkph, trailer_C_coef_Npkph2 |
| TOTAL/NET | `EXACT_EQUIVALENCE` | Exact value+NULL signature for: vde_net_mj_per_km, vde_total_mj_per_km |
| Urban/Highway/Combined results | `EXACT_EQUIVALENCE` | Exact value+NULL signature for: energy_Wh_per_km, fuel_l_per_100km, gco2_per_km, energy_ftp75_Wh_per_km, energy_hwfet_Wh_per_km, fuel_ftp75_l_per_100km, fuel_hwfet_l_per_100km, gco2_ftp75_per_km, gco2_hwfet_per_km, label_fuel_l_per_100km, label_gco2_per_km |
| labels/filters | `EXACT_EQUIVALENCE` | Exact value+NULL signature for: electrification, fuel_type, label_program, label_version_year, label_vehicle_category, label_cycle_set, label_class, label_offcycle_method, label_offcycle_energy_factor, label_offcycle_fuel_factor, label_fuel_l_per_100km, label_gco2_per_km, label_range_km, energy_basis, record_origin, record_status, review_status |
| RUN adoption lineage | `EXACT_EQUIVALENCE` | Every legacy FuelCons is linked additively to exactly one preserved evidence Run with the same VDE; RUN is not traversed by compatibility projection. |
| Canonical Program/Configuration identity | `APPROVED_CONTRACT_CORRECTION` | CDR-01/CDR-02 approved source-scoped canonical identity; application-facing fields remain unchanged. |
| FuelCons comparison basis | `APPROVED_CONTRACT_CORRECTION` | CDR-04 approved explicit non-restrictive metadata; compatibility projection remains unchanged. |
| Ambiguous EEA/JRC semantics | `DEFERRED` | CDR-06; not a blocker for legacy compatibility or authoritative-source ingestion. |
| v1 required-field materialization | `EXACT_EQUIVALENCE` | All staged records satisfy every non-null field in the exact logical contract. |

## Approved additive differences

The canonical model adds source-scoped Program/Vehicle Configuration identities, explicit `comparison_basis`, and FuelCons→RUN adoption lineage. These additions are outside the legacy compatibility projection and therefore do not change current page-facing values.

## Deferred semantics

EEA `RLFI`, exact JRC row grain, RAG/document enrichment, component causal decomposition and production migrations remain deferred per CDR. Their raw values/provenance can be retained without blocking valid VDE/RUN/FuelCons records.

## Reproduction

```powershell
python etl/scripts/sprint_12c_contract_compatibility.py
python -m unittest discover -s etl/tests -p "test_sprint_12c*.py" -v
```

The script opens `data/db/eco_drive.db` with SQLite URI `mode=ro` and `PRAGMA query_only=ON`. JSONL staging files are written only under `etl/data/staging/sprint_12c_contract_v1/`.

## Outputs

- `etl/data/processed/sprint_12c_contract_compatibility/canonical_field_contract_v1.csv`
- `etl/data/processed/sprint_12c_contract_compatibility/compatibility_checks.csv`
- `etl/data/processed/sprint_12c_contract_compatibility/mismatch_report.csv`
- `etl/data/processed/sprint_12c_contract_compatibility/materialization_counts.csv`
- `etl/data/processed/sprint_12c_contract_compatibility/public_materialization_scope.csv`
- `etl/data/processed/sprint_12c_contract_compatibility/contract_validation.csv`
- `etl/data/processed/sprint_12c_contract_compatibility/compatibility_proof.json`
- `etl/data/staging/sprint_12c_contract_v1/*.jsonl`
- `etl/reports/sprint_12c_contract_compatibility.md`
