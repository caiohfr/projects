# Sprint 15E execution report

Date: 2026-11-24

- Base SHA: `fdf33edb5a1b4ee36578fbb29201b105fa766795`
- Branch: `sprint-15e-stabilization`
- Implementation SHA: `84bb5dab41fc342aa47c548c876da2119495745b`
- Published branch: `origin/sprint-15e-stabilization`
- PR: creation blocked; `gh` is not installed and the connected GitHub App
  returned `403 Resource not accessible by integration`. Manual creation URL:
  `https://github.com/caiohfr/projects/pull/new/sprint-15e-stabilization`
- Databases used: per-test temporary SQLite files and
  `<TEMP>/ecodrive_sprint15e_g4/eco_drive_qa.db` (synthetic only)
- Architecture decision: `SEQUENTIAL_WITH_PARTIAL_STATUS`
- Release status: `PENDING`

Files changed:

- `src/vde_app/database_management_stage.py`
- `src/vde_app/components/database_management.py`
- `src/vde_app/components/methodology_models.py`
- `tests/test_database_management_stabilization_15e.py`
- `tests/test_methodology_models_ui.py`
- `docs/specs/SPRINT_15E_STABILIZATION_RELEASE_GATE.md`
- `docs/sprints/SPRINT_15E_EXECUTION_REPORT.md`

## Delivered integrity corrections

### Stage identity and cross-database guard

The first staged operation captures the normalized absolute identity of the
actual runtime SQLite path. The same identity is required when adding rows,
building previews, and before every sequential commit operation. A mismatch
performs no receipt lookup and no write. Returning to the original path retains
the stage; explicit confirmed discard clears it without database mutation.

The guard covers manual CRUD, spreadsheet rows, impact operations, and VDE/Fuel
dependency resolution. The process-global `db.DB_PATH` multi-session limitation
is not redesigned here and remains a documented architectural constraint.

### Commit and recovery semantics

The existing persistence services remain transaction owners. Stable stage
operation IDs are propagated into `data_change_log.operation_id`. Confirmed
successes are removed immediately; the first failure plus all later operations
remain pending. A matching receipt reconciles an operation before replay. An
unconfirmed result blocks retry. The UI retains success/failure receipts and
reports spreadsheet confirmed versus pending rows without claiming batch
atomicity.

The pre-fix CMT-05 reproduction injected `sqlite3.OperationalError` on the
second of three operations. The exception escaped the page and left the stage
unreconciled. Post-fix, operation 1 is receipt-confirmed and removed, operations
2 and 3 remain pending, no later operation runs, and the technical traceback is
logged.

## P0 automated evidence

`tests/test_database_management_stabilization_15e.py`:

| ID | Scenario | Result |
|---|---|---|
| STG-01 | First operation binds the stage to DB A | PASS |
| STG-02 | Preview/commit identity rejects DB B | PASS |
| STG-03 | Returning to DB A recovers the unchanged stage | PASS |
| STG-04 | New DB B operation cannot mix with DB A stage | PASS |
| STG-05 | Explicit discard changes neither database | PASS |
| STG-06 | Spreadsheet rows obey the same identity guard | PASS |
| STG-07 | Relative and absolute path forms normalize identically | PASS |
| STG-08 | Mismatch produces zero writes and zero receipt lookup | PASS |
| CMT-01 | Two updates complete with receipts | PASS |
| CMT-02 | Retry does not replay a confirmed create | PASS |
| CMT-03 | Impact success is removed before a later failure | PASS |
| CMT-04 | DB change between operations stops before the next write | PASS |
| CMT-05 | SQLite failure removes only confirmed successes | PASS |
| CMT-06 | Import failure reports confirmed and pending rows | PASS |
| CMT-07 | Discard after partial commit does not undo confirmed write | PASS |
| CMT-08 | Dependency failure stays pending | PASS |

Focused stabilization result: `16 passed`.

An additional receipt-lookup failure guard also passes, bringing the focused
stabilization module to `17 passed`.

Combined Database Management, Methodology, Vehicle Demand integration and Load
Case evidence: `111 passed, 10 subtests passed`.

## Methodology parity

The UI equation now shows positive-power clipping over the cycle interval:

```text
VDE = (1 / d_cycle) integral[0,T] max(0, P_tractive(t)) dt
```

The visible explanation includes SI units and states that negative tractive
power is braking demand without regenerative credit. The regression uses
`compute_vde_series` and `compute_vde_net` on a cycle with negative tractive
power; no UI-side physics implementation was added and no core formula changed.
The four internal source paths rendered by References & Provenance were checked
and exist in the repository.

## Baseline and final full-suite QA

Baseline command:

```text
python -m pytest tests -q --tb=short
```

Baseline result on the verified base: `2024 passed, 14 failed, 7 skipped, 190
subtests passed` in `660.29s`.

Final result: `2042 passed, 14 failed, 7 skipped, 190 subtests passed` in
`492.70s`. The 18 additional passing tests are the 17-test stabilization module
and one Methodology parity regression. The 14 failures are exactly the baseline
set listed below: no failure was added, removed, suppressed, or reclassified as
passing.

### Reproduced baseline failure classification

| Test ID(s) | Classification | Baseline cause | Relationship to 15E |
|---|---|---|---|
| `test_comparison_report_page_smoke.py::ComparisonReportDbPathSelectorTests::test_switch_to_qa_button_does_not_reseed_or_mutate_canonical_qa` | Fixture / environment | Canonical QA path resolves to a DB without `fuelcons_db` | Indirect; reproduced before 15E changes |
| `test_comparison_report_page_smoke.py::ComparisonReportDbPathSelectorTests::test_switch_to_qa_button_points_runtime_db_at_the_canonical_qa_instance` | Fixture / environment | Canonical QA path resolves to a DB without `fuelcons_db` | Indirect; reproduced before 15E changes |
| `test_database_management_ui_7c.py::DatabaseManagementUi7CTests::test_page_renders_all_management_tabs_against_runtime_db` | Fixture / environment | Repository runtime DB is empty/missing canonical `vde_db` | Direct surface, but identical pre-fix fixture failure |
| `test_database_management_vde_net_total_visibility_7g.py::DatabaseManagementVdeNetTotalVisibilityTests::test_page_renders_normalized_vde_record_without_exception` | Fixture / environment | Repository runtime DB is empty/missing canonical `vde_db` | Direct surface, but identical pre-fix fixture failure |
| Seven tests in `test_research_agent_final_micropatch.py` | External fixture gap | Sprint 12 `research_child_tasks.csv` input package is absent | Unrelated |
| Two tests in `test_technical_research_phasea.py` | External fixture gap | Sprint 12 P0 research CSV package is absent | Unrelated |
| `test_vde_setup_v22_helpers.py::TestVdeSetupV22Helpers::test_request_inputs_tire_lookup_uses_ctx_db_path_for_live_repository_reads` | Legacy fixture gap | Referenced legacy DB has no canonical `tire_db` | Unrelated |

The seven `test_research_agent_final_micropatch.py` IDs are:

- `test_golden_set_has_required_cases_and_claim_specific_queries`
- `test_internal_first_resolves_full_scope_and_preserves_partial_scope`
- `test_estimator_unlock_gate_defers_non_actionable_and_keeps_supported_routes`
- `test_known_bad_results_are_rejected_before_fetch`
- `test_source_tier_is_explainable_and_authority_first`
- `test_golden_run_filters_before_fetch_and_cache_replay_is_network_free`
- `test_internal_resolved_and_deferred_tasks_never_call_search`

The two `test_technical_research_phasea.py` IDs are:

- `test_real_p0_batch_is_scoped_split_and_traceable`
- `test_real_evidence_is_complete_read_only_and_conservative`

These failures are not suppressed or relabelled as passing. The final suite must
show the same set or fewer before the patch can claim zero new regressions.

## Visual G4

The real application was started against the disposable synthetic database on
port 8515. The required Computer Use helper failed to connect with:

```text
Computer Use native pipe is unavailable: failed to connect native pipe:
O sistema não pode encontrar o arquivo especificado. (os error 2)
```

The prescribed retry, JavaScript-session reset, reinitialization and final retry
produced the same error. No browser screenshot is claimed and AppTest is not
substituted for real-browser evidence. Coverage at 1440x900 and 1024x768 for all
eight pages is therefore `PENDING`.

Owner browser approval: not yet provided.

## Remaining stop conditions

- G4 evidence for eight pages in both widths is absent.
- Explicit owner approval is absent.
- The 14 fixture/environment failures remain visible pending repository fixture
  provisioning; they are not new 15E regressions.

Sprint 15 remains open until G4 and owner approval are recorded.
