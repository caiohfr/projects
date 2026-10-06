# Sprint 12G.1 — Explicit Canonical DB Path + Integration Simplification

## Status: `EXPLICIT_CANONICAL_DB_INTEGRATION_READY — PROCEED_TO_12H`

```text
Schema autodetection removed?             YES
Automatic table routing removed?          YES
Explicit DB path used?                    YES

Read target:
  VDE                                     vde_db
  FuelCons                                fuelcons_db

Write target:
  VDE                                     vde
  FuelCons                                fuelcons

Repository changes                        3
Service changes                           1
Page changes                              0
Schema changes                            0
Physics changes                           0

Focused tests                             18/18
AppTest                                   5/5

FK violations                             0
SQLite quick_check                        ok
Runtime DB changed?                       NO
Canonical source changed?                 NO

Remaining blockers for 12H                0
User decisions required                   0
```

## Decision

Sprint 12G.1 removed the schema-inspection and automatic compatibility-name routing introduced in 12G. The selected database contract is now declared once at the existing path boundary:

```python
configure_db_path(candidate_path, prepared_canonical=True)
```

That call selects the file, sets the explicit VDE/FuelCons physical write targets, and disables legacy bootstrap. Reapplying the same path during an existing Streamlit rerun preserves the declaration. Selecting a different path without the canonical declaration retains the pre-cutover legacy behavior. The default remains `data/db/eco_drive.db`.

No code queries `sqlite_master` or `PRAGMA` to decide which persistence model is connected. The removed symbols are `_sqlite_object_type`, `_is_canonical_connection`, `is_canonical_database`, `_CANONICAL_WRITE_TARGETS`, and `canonical_write_target`.

## Read and write boundaries

| Domain | Read surface | Write target | Evidence |
|---|---|---|---|
| VDE | `vde_db` compatibility view | `vde` physical table | `DIRECT_TESTED` |
| FuelCons | `fuelcons_db` compatibility view | `fuelcons` physical table | `DIRECT_TESTED` |

Read repositories remain unchanged in meaning and continue to query the compatibility views. Persistence helpers use the write-table names established explicitly when the DB path is selected. VDE and FuelCons repository delete/update calls now pass their configured physical target directly; there is no generic connection-time mapper.

The compact VDE request transaction uses the same configured VDE target and current selected path. Derived VDE insertion still inherits `vehicle_configuration_id` and `source_semantic_status` from the deterministic `vde_id_parent`. FuelCons retains the required `MANUAL_VALUE` to canonical `SOURCE_DECLARED` translation at the persistence boundary.

## `ensure_db()` and destructive reset

`ensure_db()` no longer opens or inspects a prepared canonical database to decide whether to bootstrap. The explicit path configuration disables legacy bootstrap before any migration SQL runs.

`truncate_db()` remains a legacy reset helper. Passing `prepared_canonical=True` raises a clear error before the database is opened or modified. No canonical cascade-delete/reset implementation was added.

## Flow evidence

| Claim | Evidence tier | Result |
|---|---|---|
| Explicit canonical path/configuration | `DIRECT_TESTED` | PASS |
| VDE compatibility-view read | `DIRECT_TESTED` | PASS |
| FuelCons compatibility-view/materialized read | `DIRECT_TESTED` | PASS |
| Physical canonical VDE write | `DIRECT_TESTED` | PASS |
| Physical canonical FuelCons write | `DIRECT_TESTED` | PASS |
| Schema autodetection removed | `DIRECT_TESTED` | PASS |
| Automatic runtime table mapper removed | `DIRECT_TESTED` | PASS |
| VDE Setup write/read-back | `DIRECT_TESTED` | PASS |
| Quick Scenario | `DIRECT_TESTED` | PASS |
| Comparison | `DIRECT_TESTED` | PASS |
| Browse | `DIRECT_TESTED` | PASS |
| Powertrain Scenario | `DIRECT_TESTED` | PASS |
| FuelCons read avoids RUN/adoption | `DIRECT_TESTED` | PASS |
| Default/legacy behavior | `DIRECT_TESTED` | PASS |
| Streamlit page behavior | `DIRECT_TESTED` via AppTest | PASS |

No inspected behavior is described as runtime-tested unless it has a corresponding focused test or AppTest.

## Focused tests — 18/18

1. `test_01_explicit_canonical_db_path_can_be_supplied`
2. `test_02_application_reads_vde_db_from_selected_file`
3. `test_03_application_reads_fuelcons_db_from_selected_file`
4. `test_04_canonical_vde_write_targets_physical_vde`
5. `test_05_canonical_fuelcons_write_targets_physical_fuelcons`
6. `test_06_no_schema_autodetection_helper_is_required`
7. `test_07_no_automatic_legacy_to_canonical_router_remains`
8. `test_08_vde_setup_write_read_back_works_on_disposable_copy`
9. `test_09_quick_scenario_uses_explicit_canonical_baseline`
10. `test_10_comparison_uses_explicit_canonical_database`
11. `test_11_browse_uses_explicit_canonical_database`
12. `test_12_powertrain_resolves_from_explicit_canonical_database`
13. `test_13_materialized_fuelcons_read_does_not_traverse_run`
14. `test_14_legacy_default_database_behavior_is_unchanged`
15. `test_15_canonical_source_candidate_hash_is_unchanged`
16. `test_16_runtime_database_hashes_are_unchanged`
17. `test_17_foreign_key_check_passes_after_disposable_writes`
18. `test_18_sqlite_quick_check_is_ok`

All write cases use a temporary copy or a freshly created disposable legacy fixture.

## AppTest — 5/5

1. `test_apptest_01_browse_loads_canonical_catalog`
2. `test_apptest_02_vde_setup_opens_canonical_rows`
3. `test_apptest_03_comparison_renders_selected_epa_and_wltp`
4. `test_apptest_04_quick_scenario_calculates_from_canonical_source`
5. `test_apptest_05_powertrain_scenario_loads_canonical_baseline`

The AppTests explicitly select the copied canonical file before running the existing pages. No Streamlit page or layout was changed.

The combined regression run passed **112/112** tests: 18 focused 12G.1 tests, 18 prior 12G integration tests, 5 AppTests, and 71 existing VDE/FuelCons persistence and calculation tests. A final focused rerun passed **23/23** (18 focused + 5 AppTest).

## Safety evidence

- Canonical candidate: `etl/data/staging/sprint_12f13_vde_materialized/eco_drive_canonical_vde_materialized_candidate.db`
- Canonical SHA-256 before/after: `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB`
- Runtime `data/db/eco_drive.db`: `CC27CDF22AE189E39F9F45609CCD23C650E822BB20655C552FF2BD89D97F4262`
- Runtime `data/db/eco_drive_qa.db`: `0EB4A5EC0F402E6B9E1F7D76372EA30068670B16F994D495394613DAFA2CA569`
- Runtime files changed: **NO**
- Canonical source changed: **NO**
- Foreign-key violations after disposable writes: **0**
- SQLite quick check: **ok**

## Change inventory

- `src/vde_core/db.py`: explicit path/persistence declaration, direct write targets, explicit bootstrap policy, canonical truncate guard; schema inspection and automatic router removed.
- `src/vde_core/repositories/vde_repository.py`: explicit configured VDE delete target.
- `src/vde_core/repositories/fuelcons_repository.py`: explicit configured FuelCons update/delete target.
- `src/vde_core/vde_request_save.py`: compact transaction writes the configured physical VDE table directly.
- Sprint 12G integration/AppTest fixtures: canonical path declaration made explicit.
- Sprint 12G.1 focused test and closure report added.

## Exit status

`EXPLICIT_CANONICAL_DB_INTEGRATION_READY — PROCEED_TO_12H`
