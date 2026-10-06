# Sprint 12F.13 — Canonical Cleanup + VDE Result Materialization

## Status: `VDE_MATERIALIZED_CANDIDATE_READY — PROCEED_TO_12G_INTEGRATION`

```text
Test VDE artifacts removed                      4
Test FuelCons removed                           4
Test RUNs removed                               8
Artificial parent rows removed                  4
ML FuelCons preserved/removed                   PRESERVED 1 / REMOVED 0

Real VDE rows remaining                         11626
TOTAL VDE materialized                          11626
TOTAL VDE unresolved                            0
NET VDE materialized                            0
NET VDE unavailable by contract                 11626
Existing-result mismatches                      0

FK violations                                   0
Adoption relationship failures                  0
SQLite quick_check                              ok
Runtime DB changed?                             NO
Deterministic rebuild?                          YES
User decisions required                         0
Focused acceptance tests                        15/15 PASS
```

## Cleanup

The exact canonical VDE IDs `5031`, `5033`, `5034`, and `5038` were verified against make/model, `LEGACY_IRREPRODUCIBLE_STATE`, `DERIVED_SCENARIO`, and legacy source-record ID before deletion. Their direct descendants were removed in FK-safe order. Four test-owned configurations became orphaned and were removed. Their three distinct program parents are real/shared EPA programs, so no Program was deleted. EPA/JRC identity signatures are unchanged.

The separate ML FuelCons `5018` is preserved. Its provenance identifies an ML prediction attached to a real EPA Lexus GS 350 VDE and does not tie it to the four confirmed QA scenarios.

## Vehicle Demand materialization

Every remaining VDE was bulk-read and passed through `build_vehicle_demand_request`, `resolve_vehicle_demand_cycle`, and `calculate_vehicle_demand` from the canonical Vehicle Demand capability (engine `0.1`, contract `0.1`). The engine output populated only existing VDE result columns. EPA FTP-75/HWFET and WLTP phase outputs were obtained by sending the canonical cycle segments through the same engine.

All 11626 real VDE rows had supported mass, TOTAL coastdown ABC, and a canonical EPA/WLTP cycle. TOTAL was newly materialized for all of them. There were no remaining pre-existing real results to overwrite after test cleanup, therefore no parity mismatch. NET remains NULL for all rows because no persisted transmission-loss ABC boundary is resolved; no NET was fabricated.

Breakdown and row-level provenance are in `vde_materialization_results.csv`. Candidate VDE payload metadata records the adapter, engine, versions, resolved cycle, result statuses, and comparison tolerance.

## Sanity and relationships

TOTAL VDE coverage is 11626 rows; min/median/max are 0.309202 / 0.552152 / 2.185562 MJ/km. Zero or negative values: 0; non-finite values: 0.

Pearson correlations are exported as lightweight QA associations with their paired sample sizes. They are not causal or model-validation claims. 11 high statistical outliers (above Q3 + 3×IQR) are listed for engineering review; they were not deleted or altered based on magnitude.

## Integrity and runtime safety

All primary keys are unique, FK check is clean, every remaining FuelCons retains RUN lineage, and adoption rows preserve the same-VDE invariant. All table export counts match the database. Runtime databases and the notebook demo remained byte-identical.

Candidate SHA-256: `D80E1918FE72AF129E1104F3C2C8C0A07037264B2E2A1E01839543F04D5498DF`. Relationship/result signature: `82EBC79F774D0788CAE500A9F88C6AC197EFC073A0CABAC915196DD1C2FF57B8`.
