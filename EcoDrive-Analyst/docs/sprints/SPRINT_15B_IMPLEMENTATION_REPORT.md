# Sprint 15B - Implementation Report

Date: 2026-11-23

Baseline: `origin/main@e00357b2`

Branch: `sprint-15b-vde-lab`

## Outcome

Sprint 15B applies the visual workflow refinement to VDE Setup and Roadload & Load Case Lab without changing vehicle-demand physics, database schema, repositories, or persistence behavior. The code is ready for PR review. G3 remains pending because no browser-control surface was available for the required real-browser screenshots and user visual approval.

## Implemented scope

- Added a shared, token-based Sprint 15B presentation layer for workflow headers, persisted-baseline context, result state, and responsive behavior.
- Moved the four-step VDE request flow from the utility sidebar into the main workspace while preserving its existing keys, callbacks, and session state.
- Clarified the VDE workflow hierarchy, proposal-matrix headers, preview-first contract, explicit Save boundary, and persisted-baseline handoff to the Lab.
- Reorganized the Lab controls and mass-state result rows, and made the temporary/no-Save boundary and explicit Calculate action prominent.
- Added persisted VDE context, calculated-result provenance, stale-result guidance, and an empty-database route back to VDE Setup.
- Replaced deprecated `use_container_width` calls in the touched VDE compact renderer with the current `width` API.
- Kept the Home, navigation destinations, physical models, database schema, and authoritative VDE/FuelCons data unchanged.

## Files

UI:

- `pages/VDE_Setup.py`
- `pages/Roadload_Load_Case_Lab.py`
- `src/vde_app/components/sprint15b_style.py`
- `src/vde_app/components/vde_request_compact.py`
- `src/vde_app/components/vde_request_compact_style.py`
- `src/vde_app/components/roadload_load_case_lab.py`

Tests:

- `tests/test_vde_setup_v22_helpers.py`
- `tests/test_roadload_load_case_lab_ui.py`

Documentation:

- `docs/sprints/SPRINT_15B_IMPLEMENTATION_REPORT.md`

No file under `src/vde_core/` was modified.

## Automated verification

The directed suites were run before and after the UX change. The result was unchanged:

| Suite | Baseline | Post-change |
| --- | ---: | ---: |
| `tests.test_vde_setup_v22_helpers` | 52 pass, 1 fail, 1 error | 52 pass, 1 fail, 1 error |
| compact viewmodels/state/persistence | 85 pass | 85 pass |
| Lab UI + vehicle-demand load case | 54 pass | 54 pass |
| Home context + Home 15A | 8 pass | 8 pass |

Total post-change: 199 passed, 1 failed, 1 errored across 201 directed tests.

The two non-passing VDE Setup tests reproduce the pre-existing local-fixture gaps documented by the handoff/PR #19:

- `test_request_inputs_tire_lookup_pending_widgets_apply_and_walk_from_inherit`: the fixture does not expose the expected `Apply Tire` action.
- `test_request_inputs_tire_lookup_uses_ctx_db_path_for_live_repository_reads`: `sqlite3.OperationalError: no such table: tire_db`.

They occur on the unchanged baseline and after the patch. The valid isolated QA database was used for the other directed suites; no personal database was modified.

Additional checks:

- Python compilation of all changed Python modules: PASS.
- `git diff --check`: PASS.
- Home/navigation regression group: PASS (8/8).
- No production-core or schema diff: PASS.

## Numeric parity evidence

The load-case calculation modules are unchanged, and the same core suite passed 54/54 before and after the UX patch. A deterministic synthetic QA drive-cycle fixture (`SYNTHETIC_QA`, custom scenario mass 2,500 kg, no resolved RRC prior) produced:

| Metric | Before UX | After UX | Tolerance | Status |
| --- | ---: | ---: | ---: | --- |
| Baseline calculation mass | 1,600 kg | 1,600 kg | exact | PASS |
| Scenario calculation mass | 2,500 kg | 2,500 kg | exact | PASS |
| Baseline VDE TOTAL | 4.07448 MJ/km | 4.07448 MJ/km | exact | PASS |
| Scenario VDE TOTAL | 6.32448 MJ/km | 6.32448 MJ/km | exact | PASS |
| Baseline positive inertial work | 0.080 MJ | 0.080 MJ | exact | PASS |
| Scenario positive inertial work | 0.125 MJ | 0.125 MJ | exact | PASS |
| Static force curve under mass-only override | identical | identical | exact | PASS |

This confirms that the page changes only alter presentation; the dynamic mass effect and steady-state roadload semantics remain canonical.

## Gate status

- G0 - baseline/integrity: PASS. Isolated worktree based on the recorded `origin/main`; personal checkout and database untouched.
- G1 - VDE Setup: PASS for automated coverage, with the two inherited fixture gaps listed above.
- G2 - Load Case Lab: PASS for automated coverage and numeric parity.
- G3 - real-browser smoke and visual approval: PENDING.

## G3 attempts and remaining evidence

Streamlit was started against an isolated synthetic QA database. Two browser-control paths were attempted:

1. Native browser control: `native pipe is unavailable`.
2. Integrated browser control: `browser access host is unavailable`.

Therefore no screenshots were fabricated or committed. The PR must remain open for review until the VDE Setup and Lab states are inspected at 1440x900 and 1024x768 (or equivalent widths) and approved by the user. Sprint 15B is not considered fully closed until that G3 approval is recorded.
