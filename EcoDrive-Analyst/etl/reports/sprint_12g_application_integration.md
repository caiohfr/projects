# Sprint 12G — Canonical Database Application Integration

## Status: `CANONICAL_APP_INTEGRATION_READY — PROCEED_TO_12H_CUTOVER`

```text
Canonical candidate used                     etl\data\staging\sprint_12f13_vde_materialized\eco_drive_canonical_vde_materialized_candidate.db
Canonical DB opened successfully?            YES

Browse read                                  PASS
VDE Setup read                               PASS
VDE Setup write                              PASS
Comparison                                   PASS
Quick Scenario                               PASS
Powertrain Scenario                          PASS
FuelCons materialized read                   PASS

Repository changes                           1
Service changes                              1
Page changes                                 0
Schema changes                               0

Focused tests                                18/18
AppTest                                      5/5
Manual browser smoke                         NOT_RUN

FK violations after write tests              0
SQLite quick_check                           ok
Runtime DB changed?                          NO
Canonical source candidate changed?          NO

Integration blockers                         0
User decisions required                      0
```

## Integration decision

The canonical candidate is ready for the controlled 12H cutover. Existing read contracts continue through the `vde_db` and `fuelcons_db` compatibility views. Two localized changes close the persistence gap: the database repository resolves legacy write names to the canonical physical `vde`/`fuelcons` tables, and the compact VDE save transaction uses the configured database path and the same canonical adapter. No page, schema, Vehicle Demand physics, Quick contract, or FuelCons calculation changed.

The production default remains `data/db/eco_drive.db`. Integration and write tests select a temporary copied database through the existing `ECO_DRIVE_DB_PATH`/`configure_db_path` mechanism.

## Flow classification and evidence

| Flow | Classification | Evidence tier | Result |
|---|---|---|---|
| Browse | `NO_CHANGE` | `DIRECT_TESTED` | Repository filter, catalog identities, counters and AppTest load passed. |
| VDE Setup read | `NO_CHANGE` | `DIRECT_TESTED` | Canonical snapshots and the page load passed. |
| VDE Setup write | `LOCALIZED_REPOSITORY_CHANGE` | `DIRECT_TESTED` | Create/update/read-back passed against a disposable copy. |
| Comparison | `NO_CHANGE` | `DIRECT_TESTED` | EPA/WLTP selection and comparison dataset passed. |
| Quick Scenario | `NO_CHANGE` | `DIRECT_TESTED` | Canonical baseline and a mass override resolved; no Quick physics changed. |
| Powertrain Scenario | `NO_CHANGE` | `DIRECT_TESTED` | VDE/FuelCons baseline and page load passed. |
| FuelCons read | `NO_CHANGE` | `DIRECT_TESTED` | Materialized values load without querying RUN/adoption. |
| FuelCons write | `LOCALIZED_REPOSITORY_CHANGE` | `DIRECT_TESTED` | Existing save flow writes the physical canonical table and reads back. |
| Manual browser smoke | `BLOCKED_BY_ENVIRONMENT` | `GAP` | Local Streamlit server started, but the browser-control connection was unavailable. Deferred to 12H. |

## Representative canonical cases

| Case | IDs used |
|---|---|
| EPA nominal | VDE `-1011377`; FuelCons `-4000001` |
| EPA alternative / same-configuration multi-VDE | Vehicle Configuration `CFG-EPA-00018CDDE0E1D6277907`; VDE `-1001683` and `-1000256` |
| WLTP phase result | VDE `-2000250`; FuelCons `-3000250` |
| Multi-FuelCons | VDE `-1011369`; FuelCons `-4000008` and `-4000006` |
| No-FuelCons | VDE `-1011373` |
| Quick Scenario | EPA VDE `-1011377`; temporary mass delta exercised in AppTest |
| VDE/FuelCons write | EPA VDE `-1011377` used as parent; child and result created only in test copy, then removed |

IDs are selected deterministically from the current candidate by the focused tests; none are embedded in application logic.

## Read parity

Application-facing contract-shape evidence is `DIRECT_TESTED`. The current runtime and canonical candidate share 0 VDE IDs and 1 FuelCons IDs. VDE overlapping-record value parity is `GAP`: the refreshed canonical IDs do not overlap the current runtime, so no value-equivalence claim is made. Representative FuelCons `5018` matched on 7/8 inspected fields (`DIRECT_TESTED`). Differences are recorded in the machine-readable summary and are not treated as population regressions because the canonical population intentionally contains refreshed/reconstructed data.

The comparison verifies IDs/identity, mass, A/B/C, total/net VDE, FuelCons values, and application-facing fields where present in both projections. Normal FuelCons reads were instrumented and issued no SQL against `run` or `fuelcons_run_adoption`.

## Read performance

Three samples were collected per operation in one process against the canonical candidate; the median is reported.

| Operation | Median | Rows | Assessment |
|---|---:|---:|---|
| Browse initial load | 534.377 ms | 10822 | ACCEPTABLE |
| Browse filtered query | 25.007 ms | 404 | ACCEPTABLE |
| fetch VDE by ID | 3.572 ms | 1 | ACCEPTABLE |
| fetch FuelCons by VDE | 2.642 ms | 1 | ACCEPTABLE |
| Comparison baseline load | 23.760 ms | 2 | ACCEPTABLE |

No obvious integration regression required optimization.

## Write and integrity evidence

All writes ran only against temporary copies. VDE create/update targeted `vde`, inherited the deterministic parent Vehicle Configuration and retained `vde_id_parent`. FuelCons creation targeted `fuelcons`; the existing application label `MANUAL_VALUE` was mapped at the persistence boundary to the frozen canonical `SOURCE_DECLARED` enum. Read-after-write used the normal repositories, cleanup deleted the temporary parent-dependent rows deterministically, `PRAGMA foreign_key_check` returned zero rows, and `PRAGMA quick_check` returned `ok`.

Candidate counts remained: Program 3117; Vehicle Configuration 10211; VDE 11626; RUN 29250; FuelCons 10822; adoption 18323.

## Focused tests — 18/18

1. `test_01_canonical_db_path_can_be_selected_without_changing_production_default`
2. `test_02_browse_repository_reads_canonical_compatibility_surface`
3. `test_03_fetch_by_id_works_for_canonical_vde`
4. `test_04_fuelcons_by_vde_reads_canonical_materialized_results`
5. `test_05_representative_epa_read_works`
6. `test_06_representative_wltp_read_works`
7. `test_07_multi_vde_same_configuration_case_works`
8. `test_08_multi_fuelcons_case_works`
9. `test_09_no_fuelcons_case_is_handled`
10. `test_10_vde_setup_read_works`
11. `test_11_vde_setup_create_update_writes_canonical_tables_in_disposable_db`
12. `test_12_write_read_back_parity_passes`
13. `test_13_quick_scenario_works_from_canonical_baseline`
14. `test_14_comparison_works_from_canonical_records`
15. `test_15_powertrain_scenario_baseline_resolves`
16. `test_16_normal_fuelcons_read_does_not_traverse_run`
17. `test_17_fk_and_quick_check_pass_after_writes`
18. `test_18_runtime_and_source_candidate_hashes_are_unchanged`

## Streamlit AppTest — 5/5

1. `test_apptest_01_browse_loads_canonical_catalog`
2. `test_apptest_02_vde_setup_opens_canonical_rows`
3. `test_apptest_03_comparison_renders_selected_epa_and_wltp`
4. `test_apptest_04_quick_scenario_calculates_from_canonical_source`
5. `test_apptest_05_powertrain_scenario_loads_canonical_baseline`

AppTest is reported separately from manual browser coverage. The five AppTests loaded the canonical copy and exercised Browse, VDE Setup, selected EPA/WLTP Comparison, a Quick mass override, and Powertrain baseline resolution.

## Runtime safety

- Canonical SHA-256 before/after: `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB` / `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB`.
- `data\db\eco_drive.db`: `CC27CDF22AE189E39F9F45609CCD23C650E822BB20655C552FF2BD89D97F4262`
- `data\db\eco_drive_qa.db`: `0EB4A5EC0F402E6B9E1F7D76372EA30068670B16F994D495394613DAFA2CA569`
- Runtime hashes before and after were identical: **YES**.
- Canonical source candidate remained byte-identical: **YES**.
- Default runtime DB was not replaced: **YES**.

## Known data limitation (non-blocking)

The candidate currently has 0 materialized `rrc_N_per_kN` values and 0 materialized `cda_m2` values. Quick Scenario correctly requires an explicit physical reference for RRC-delta/target tire calculations; the test therefore exercised a real mass override and neutral tire mode instead of fabricating tire data. This is a dataset capability limitation, not an application-integration or frozen-contract mismatch.

## Exit status

`CANONICAL_APP_INTEGRATION_READY — PROCEED_TO_12H_CUTOVER`
