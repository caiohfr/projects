# Sprint 15A — Home and navigation evidence

Date: 2026-11-17
Implementation baseline: `c249ca08` (`origin/main`)
Working branch: `sprint-15a-home`

## Delivered

- A first-access Home with a prominent **New VDE proposal** action.
- An active-session Home with a distinct **Continue analysis** action and explicit confirmation before discarding temporary VDE Setup state.
- Five working destinations: VDE Setup, Roadload & Load Case Lab, Comparison Report, Powertrain Scenario and Database Management.
- Separate destinations for Methodology & Models and Engineering Tools / Legacy.
- Eight real Streamlit destinations in the global sidebar.
- Active Context and Recent Analysis derived only from recoverable session or persisted records.
- The former scientific Home remains available as Methodology & Models; its scientific copy is intentionally not revised in 15A.

## Displayed data sources

| Home field | Existing source | Validation |
| --- | --- | --- |
| Baseline | `vde_setup_v22.baseline.selected_vde_id` | ID is reloaded through `fetch_vde_by_id` against the active database. |
| Temporary proposal | Configured entries in `vde_setup_v22.proposals`, plus preview status | Session-only and explicitly labelled **not persisted**. Blank default proposals are ignored. |
| FuelCons reference | `pwt_ss:current_baseline`, falling back to the CURRENT draft in `pwt_ss_drafts` | ID is reloaded through `fetch_fuelcons_by_id`; its linked VDE is also resolved when available. |
| Recent saved VDE | `vde_setup_v22.save.result.saved_proposals[].vde_row_id` | Each ID is reloaded before display. |
| Recent Lab context | `roadload_load_case_vde_id` | Persisted VDE is reloaded before display. |
| Recent Comparison context | `comparison_selection` or `comparison_direct_vde_selection` | FuelCons or VDE IDs are reloaded according to the existing comparison mode. |

Missing or stale IDs produce a warning and are not presented as active records. No global per-user history is inferred.

## VDE Setup → Lab decision

The current implemented handoff accepts `roadload_load_case_vde_id`, and the Lab reloads a persisted VDE row. The VDE Setup preview is a richer temporary request result and there is no existing shared handoff contract that proves parity for its resolved components, TOTAL/NET boundary and all proposal inputs.

Decision for 15A: the Home opens the Lab only from a validated persisted baseline. It sets the existing `roadload_load_case_vde_id` and performs no write. Passing an unsaved preview remains a separate contract evolution; no divergent adapter or UI-side physics was added.

## Verification matrix

| Requirement | Evidence |
| --- | --- |
| Empty and active Home | Direct AppTest coverage in `tests/test_home_page_15a.py`. |
| Invalid context | Pure resolver coverage in `tests/test_home_context.py`; stale IDs are omitted and warned. |
| Real destinations | AppTest asserts Streamlit page-link labels; implementation uses `st.page_link` / `st.switch_page`. |
| Explicit discard | Active Home requires a confirmation checkbox before clearing only `vde_setup_v22`. |
| Baseline-only Lab handoff | Direct inspection of VDE Setup and Lab contracts plus guarded Home action. |
| Physics and persistence | No calculation, schema or persistence owner changed. Home calls repositories only for record validation and labels. |
| Browser widths and all-destination smoke | Recorded separately from AppTest; not interchangeable with automated page evidence. |

## Test record

Tests use the repository's synthetic QA database and do not introduce production fixtures.

- `python -m unittest tests.test_home_context tests.test_home_page_15a -v`: **7 passed**.
- Methodology page AppTest: **0 exceptions**.
- Existing directed page set (`test_vde_setup_v22_helpers`, `test_roadload_load_case_lab_ui`, `test_powertrain_system_scenario_ui`, `test_comparison_report_page_smoke`): **150 run; 146 passed; 3 failed; 1 error**.
- The four non-passing cases were reproduced in isolation. They require repository-local canonical fixture databases that are absent or incomplete in this checkout: `tire_db` is absent for two VDE Setup tire lookup cases and `fuelcons_db` is absent from the QA database selected by two Comparison DB-switch cases. No file changed by 15A owns these fixtures or those page implementations.
- `git diff --check`: run at close.

## Browser evidence status

The local Streamlit server started and its health endpoint returned HTTP 200. Both available interactive browser routes then failed before opening a tab: the Windows Computer Use helper reported `native pipe unavailable`, and the integrated browser reported `browser access host is unavailable`. Playwright and Selenium are not installed locally. Consequently, no desktop-wide or narrow-window screenshots are claimed. The responsive rules are present in the clickable v0.3 prototype, but the two-width real-browser gate remains explicitly open for a browser-enabled review environment.
