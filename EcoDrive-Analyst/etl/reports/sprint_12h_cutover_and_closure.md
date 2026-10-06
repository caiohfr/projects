# Sprint 12H — Canonical Cutover, Final Regression & Sprint 12 Closure

## FINAL CANONICAL RUNTIME

| Gate | Result |
|---|---:|
| Program | 3,117 |
| Vehicle Configuration | 10,211 |
| VDE | 11,626 |
| RUN | 29,250 |
| FuelCons | 10,822 |
| FuelCons↔RUN adoption | 18,323 |
| Runtime DB path | `data/db/eco_drive.db` |
| Legacy backup path | `data/backups/eco_drive_pre_sprint12_20260915.db` |
| Runtime DB SHA-256 | `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB` |
| Legacy backup SHA-256 | `CC27CDF22AE189E39F9F45609CCD23C650E822BB20655C552FF2BD89D97F4262` |
| Canonical source hash unchanged? | YES |
| Browse | PASS |
| VDE Setup read | PASS |
| VDE Setup real save | PASS — application/service payload path; UI click remains in manual smoke |
| Comparison | PASS |
| Quick Scenario | PASS |
| Powertrain Scenario | PASS |
| FuelCons read | PASS |
| FuelCons write | PASS — repository write/read cycle |
| Full test suite | 1,841/1,841 |
| Focused Sprint 12 tests | 53/53 |
| AppTest | 5/5 |
| Manual browser smoke | NOT_RUN |
| FK violations | 0 |
| SQLite quick_check | `ok` |
| FuelCons adoption invariant failures | 0 |
| Performance regressions | 0 |
| New regressions | 0 |
| Rollback ready? | YES |
| Runtime cutover complete? | YES |
| Remaining blockers | 1 — actual human browser smoke |
| User decisions required | 0 |

**Exit status:** `SPRINT_12_CUTOVER_READY — MANUAL_BROWSER_SMOKE_REQUIRED`

## Cutover record

The previous default runtime was hashed and preserved before replacement. The frozen canonical candidate was copied, not moved, into the established default runtime path. The source and runtime hashes match after cutover, while the legacy backup matches the recorded pre-cutover runtime hash.

| Artifact | Repository-relative path | SHA-256 |
|---|---|---|
| Final canonical runtime | `data/db/eco_drive.db` | `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB` |
| Frozen canonical source | `etl/data/staging/sprint_12f13_vde_materialized/eco_drive_canonical_vde_materialized_candidate.db` | `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB` |
| Legacy runtime backup | `data/backups/eco_drive_pre_sprint12_20260915.db` | `CC27CDF22AE189E39F9F45609CCD23C650E822BB20655C552FF2BD89D97F4262` |
| QA reference DB (safety check) | `data/db/eco_drive_qa.db` | `0EB4A5EC0F402E6B9E1F7D76372EA30068670B16F994D495394613DAFA2CA569` |

The final default path explicitly selects the prepared canonical persistence contract. No schema inspection or automatic read/write routing is used. Prepared canonical startup skips legacy table creation and migration. Reads retain the `vde_db` and `fuelcons_db` compatibility surfaces; writes target physical `vde` and `fuelcons`. Legacy truncate is explicitly rejected for a prepared canonical database.

## Final architecture

```text
Raw regulatory/source data
        ↓
ETL / deterministic normalization
        ↓
Canonical SQLite database
        ├── Program
        ├── Vehicle Configuration
        ├── VDE
        ├── RUN
        ├── FuelCons
        └── lineage/helper structures
        ↓
Compatibility views for existing reads
        ↓
Repositories/services
        ↓
EcoDrive application
```

```text
VDE                   = persisted resolved analysis state
RUN                   = evidence/execution
FuelCons              = persisted consumption/energy/CO2 result
temporary sensitivity = not persisted by default
```

## Save and runtime safety evidence

The disposable VDE save test uses a real canonical baseline and follows the same application construction path used by VDE Setup: request context, preview calculation, review payload, save plan, transactional insert, and compatibility-view read-back. It changes one supported mass input and verifies a complete payload (more than 20 fields), `vehicle_configuration_id`, `vde_id_parent`, resolved mass and road-load fields, VDE TOTAL/NET service values, physical-table persistence, compatibility-view readability, and FK integrity. The disposable database is then discarded. The Streamlit Preview/Save button clicks themselves could not be automated in this environment and remain part of the manual gate.

The FuelCons smoke creates a row through the active application repository in a disposable canonical copy, proves that the write lands in physical `fuelcons`, reads it through `fuelcons_db`, verifies the referenced VDE and preserved result, runs the FK check, and discards the copy. Normal reads do not traverse RUN.

The actual default runtime path was exercised without a staging override. Default Browse, one VDE fetch, one FuelCons fetch, canonical compatibility views, and suppression of legacy bootstrap all passed.

## Representative canonical cases

| Case | Deterministic evidence |
|---|---|
| EPA nominal with FuelCons | VDE `-1011377`; FuelCons `-4000001` |
| Same configuration / multiple VDE states | Configuration `CFG-EPA-00018CDDE0E1D6277907`; VDEs include `-1001683` and `-1000256` (2 total) |
| WLTP/JRC with FuelCons | VDE `-2000250`; FuelCons `-3000250` |
| Multi-FuelCons | VDE `-1011369`; 3 FuelCons rows, including `-4000008` and `-4000006` |
| No-FuelCons | VDE `-1011373` |
| Derived save | Created only in a disposable DB from a real parent, asserted, then discarded |

The IDs above are evidence fixtures only and are not embedded in production routing or migration logic.

## Regression results

### Complete suite

Final discovery run:

```text
total                 1841
passed                1841
failed                   0
errors                   0
skipped                  0
xfailed                  0
unexpected successes     0
```

There are no failures to classify and no new Sprint 12 regression. Historical project evidence had recorded two pre-existing failures in older suites; neither remains in this final run.

### Focused Sprint 12 run

The combined focused command passed **53/53**. Exact evidence tests:

`etl/tests/test_sprint_12h_cutover.py` — 12 tests:

```text
test_01_default_runtime_is_canonical_without_override
test_02_final_runtime_population_and_views
test_03_default_startup_does_not_run_legacy_bootstrap
test_04_real_vde_setup_preview_save_and_read_back
test_05_fuelcons_write_physical_read_compatibility_view
test_06_core_application_flows_use_canonical_copy
test_07_representative_relationship_cases_exist
test_08_final_integrity_and_adoption_invariant
test_09_canonical_runtime_blocks_legacy_truncate
test_10_legacy_backup_and_rollback_rehearsal
test_11_canonical_source_is_immutable
test_12_runtime_and_qa_safety_hashes
```

`etl/tests/test_sprint_12g1_explicit_db_path.py` — 18 tests:

```text
test_01_explicit_canonical_db_path_can_be_supplied
test_02_application_reads_vde_db_from_selected_file
test_03_application_reads_fuelcons_db_from_selected_file
test_04_canonical_vde_write_targets_physical_vde
test_05_canonical_fuelcons_write_targets_physical_fuelcons
test_06_no_schema_autodetection_helper_is_required
test_07_no_automatic_legacy_to_canonical_router_remains
test_08_vde_setup_write_read_back_works_on_disposable_copy
test_09_quick_scenario_uses_explicit_canonical_baseline
test_10_comparison_uses_explicit_canonical_database
test_11_browse_uses_explicit_canonical_database
test_12_powertrain_resolves_from_explicit_canonical_database
test_13_materialized_fuelcons_read_does_not_traverse_run
test_14_legacy_default_database_behavior_is_unchanged
test_15_canonical_source_candidate_hash_is_unchanged
test_16_runtime_database_hashes_are_unchanged
test_17_foreign_key_check_passes_after_disposable_writes
test_18_sqlite_quick_check_is_ok
```

`etl/tests/test_sprint_12g_application_integration.py` — 18 tests:

```text
test_01_canonical_db_path_can_be_selected_without_changing_production_default
test_02_browse_repository_reads_canonical_compatibility_surface
test_03_fetch_by_id_works_for_canonical_vde
test_04_fuelcons_by_vde_reads_canonical_materialized_results
test_05_representative_epa_read_works
test_06_representative_wltp_read_works
test_07_multi_vde_same_configuration_case_works
test_08_multi_fuelcons_case_works
test_09_no_fuelcons_case_is_handled
test_10_vde_setup_read_works
test_11_vde_setup_create_update_writes_canonical_tables_in_disposable_db
test_12_write_read_back_parity_passes
test_13_quick_scenario_works_from_canonical_baseline
test_14_comparison_works_from_canonical_records
test_15_powertrain_scenario_baseline_resolves
test_16_normal_fuelcons_read_does_not_traverse_run
test_17_fk_and_quick_check_pass_after_writes
test_18_runtime_and_source_candidate_hashes_are_unchanged
```

`etl/tests/test_sprint_12g_application_integration_apptest.py` — 5 tests:

```text
test_apptest_01_browse_loads_canonical_catalog
test_apptest_02_vde_setup_opens_canonical_rows
test_apptest_03_comparison_renders_selected_epa_and_wltp
test_apptest_04_quick_scenario_calculates_from_canonical_source
test_apptest_05_powertrain_scenario_loads_canonical_baseline
```

An additional compatibility regression combining Sprint 12H/12G and mass persistence tests passed **70/70**. AppTest is reported separately as **5/5 PASS**, not as manual-browser evidence.

## Integrity

Validation was read-only against the final runtime:

```text
PRAGMA foreign_key_check                  0 rows
PRAGMA quick_check                        ok
duplicate Program primary keys            0
duplicate Vehicle Configuration PKs       0
duplicate VDE primary keys                0
duplicate RUN primary keys                0
duplicate FuelCons primary keys           0
FuelCons↔RUN same-VDE failures             0
vde_db compatibility-view rows        11,626
fuelcons_db compatibility-view rows    10,822
```

No repair was performed during final validation. Production contains no disposable QA save.

## Performance sanity

Three lightweight samples per operation were taken from the final canonical runtime; values below are medians.

| Operation | Median | Rows/records | Classification |
|---|---:|---:|---|
| Browse initial load | 754.592 ms | 10,822 | ACCEPTABLE |
| Browse filtered query | 67.833 ms | 404 | GOOD |
| Fetch VDE by ID | 5.274 ms | 1 | GOOD |
| FuelCons by VDE | 10.885 ms | 1 | GOOD |
| Comparison load | 64.794 ms | 2 representative records | GOOD |
| VDE Setup load | 8.386 ms | 100 | GOOD |

No obvious regression was observed and no performance change was made.

## Rollback proof

`ROLLBACK_READY = YES`.

The backup exists and its SHA-256 is identical to the recorded pre-cutover runtime hash. A rehearsal used a disposable copy, explicitly selected it as a legacy database, executed legacy startup/read behavior, and restored the original configuration without touching production.

To launch the preserved legacy runtime without overwriting the validated canonical file, from the repository root in PowerShell:

```powershell
$env:ECO_DRIVE_DB_PATH = (Resolve-Path 'data/backups/eco_drive_pre_sprint12_20260915.db')
python -m streamlit run app.py
```

To return to the canonical default in a fresh process:

```powershell
Remove-Item Env:ECO_DRIVE_DB_PATH -ErrorAction SilentlyContinue
python -m streamlit run app.py
```

This rollback path requires no schema detection and no code redesign. Restoring the backup over production is unnecessary for normal rollback and is intentionally not prescribed as the first option.

## Manual browser closure gate

`MANUAL_BROWSER_SMOKE = NOT_RUN`.

The final canonical app server started and returned HTTP 200, but the browser automation runtime was unavailable in this execution environment. No browser pass is claimed. Run this five-minute checklist:

1. From the repository root, clear `ECO_DRIVE_DB_PATH` and run `python -m streamlit run app.py`.
2. In **Browse**, filter/select a real EPA vehicle and confirm its VDE and FuelCons details open without errors.
3. Open **VDE Setup**, load that baseline, change one safe mass input, click **Preview**, and verify the calculated result; do not save to production.
4. In **Comparison**, open representative EPA and WLTP records, then run one safe mass override in **Quick Scenario**.
5. Open **Powertrain Scenario**, then return to Browse and confirm a WLTP vehicle opens; scan all five pages for broken labels, layout, or visible exceptions.

If all five steps pass, the only remaining closure gate is satisfied and the status may be promoted to `SPRINT_12_CLOSED — CANONICAL_RUNTIME_ACTIVE` without another data cutover.

## Evidence classification

| Closure claim | Evidence tier | Basis |
|---|---|---|
| Canonical default startup | DIRECT_TESTED | Actual default path, no override; bootstrap guard asserted |
| Cutover and source immutability | DIRECT_TESTED | Independent SHA-256 checks before/after |
| Browse | DIRECT_TESTED | Repository tests plus Streamlit AppTest |
| VDE Setup read | DIRECT_TESTED | Repository/service tests plus AppTest |
| VDE Setup application/service save | DIRECT_TESTED | Full payload generation/save/read-back in disposable canonical copy |
| VDE Setup Preview/Save browser clicks | GAP | Included in five-minute manual checklist |
| FuelCons read/write | DIRECT_TESTED | Physical write and compatibility read in disposable copy |
| Comparison | DIRECT_TESTED | Representative data test plus AppTest |
| Quick Scenario / Vehicle Demand calculation | DIRECT_TESTED | Canonical baseline with supported mass change; physics unchanged |
| Powertrain Scenario | DIRECT_TESTED | Baseline resolution plus AppTest |
| Full regression | DIRECT_TESTED | 1,841/1,841 clean discovery run |
| Final DB integrity | DIRECT_TESTED | PRAGMA, key, view, and adoption checks |
| Rollback | DIRECT_TESTED | Hash equality and disposable path/config rehearsal |
| Performance sanity | DIRECT_TESTED | Median measurements on final runtime |
| Manual browser smoke | GAP | Browser automation runtime unavailable |

## Remaining known MVP limitations

- EPA BEV FuelCons coverage remains limited.
- Temperature sensitivity, RDE/GLAMYS custom flows, and broader manual/performance productization remain roadmap items.
- NET remains unavailable where the loss-resolution contract cannot support it.
- The prepared canonical component table follows the new evidence-oriented contract and currently has no legacy road-load component rows. Existing MVP component lookup therefore continues to use its established read-only mock catalogue; canonical custom-component CRUD is not claimed by this closure.

None of these limitations corrupts or misrepresents the validated canonical runtime. No new feature, physics model, schema redesign, or automatic database routing was introduced.

