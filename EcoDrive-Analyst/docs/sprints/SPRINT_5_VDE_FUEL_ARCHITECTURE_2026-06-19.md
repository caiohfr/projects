Sprint 5 - VDE Setup Workflow + Fuel Energy Architecture

Date: 2026-06-19
Last updated: 2026-06-25
Status: implemented in MVP form, pending final validation

Objective

This sprint has two real delivery goals:

1. Consolidate `VDE Setup` as the disciplined roadload workflow.
2. Create the estimation foundation for `Powertrain Scenario` (former `PWT Fuel Energy`), while separating comparison/reporting concerns.

The intent is not to finish every future capability now. The intent is to leave:

- `VDE Setup` operational, traceable, and save-safe
- `Powertrain Scenario` usable as an estimation workflow
- comparison / benchmark / reporting isolated from the estimator path

Current architecture checkpoint

Core delivered:

- `src/vde_core/vde_workflow_service.py`
  - `build_vde_setup_preview(...)`
  - `build_vde_setup_preview_from_ctx(...)`
  - `save_vde_setup_result(...)`
- `tests/test_vde_workflow_service.py` created for the VDE workflow contracts
- `src/vde_core/fuel_estimation.py`
  - common request/result flow for manual, physics, and regression estimation paths
- `tests/test_fuel_estimation.py` created for the Fuel / Powertrain estimation contracts
- `src/vde_core/pwt_fuel_energy_service.py` explicitly resolves `VDE_TOTAL` and `VDE_NET` instead of reconstructing total demand from transmission efficiency
- regression filtering now respects scenario context such as `vde_id` and legislation in the shared data path

UI entrypoints delivered:

- `pages/VDE_Setup.py`
- `src/vde_app/components/vde_setup.py`
- `pages/Powertrain_Scenario.py`
- `src/vde_app/components/pwt_fuel_energy.py`
- `pages/Comparison_Report.py`

Delivered - VDE Setup

Main workflow structure now exists as:

1. `Scenario Setup`
2. `Vehicle Parameters`
3. `Roadload Build-up`
4. `Cycle & Preview`
5. `Results`
6. `Save / Edit`

What is working in the current MVP:

- `Scenario Setup` now owns vehicle metadata plus scenario origin instead of mixing technical deltas into the top of the workflow
- the page uses summary-first navigation so the main workflow is cleaner and less vertically noisy than the previous stacked legacy layout
- `Roadload Build-up` now behaves as the technical roadload workspace instead of a loose collection of legacy panels
- `Roadload Basis` is treated as the superior decision above subordinate technical configuration
- `Mass & Axle Load` is centralized so roadload, tires, preview, and transmission reuse the same resolved vehicle state
- `Transmission` remains separate from normal components while still participating in the same workflow bridge
- `ABC_NET = ABC_TOTAL - ABC_TRANS` is preserved in the workflow layer
- `VDE_TOTAL` is derived from `ABC_TOTAL`
- `VDE_NET` only exists when transmission losses / neutral drag are available
- preview stays non-persistent
- save/update now routes through the workflow service path instead of only through ad hoc page assembly
- `Results` now acts as pre-save review rather than only raw technical output
- UI units can be switched between Metric and US customary without changing internal storage/calc units

Delivered - VDE components

Component build-up is no longer only legacy stacked inputs. It now has a clearer technical surface with basis-aware behavior.

Current component status:

- `Tires`
  - tire DB integration lives inside the tire editor
  - quick-add flow exists
  - scenario-only manual reference exists
  - current/reference vs walked/target logic has been partially structured
  - direct delta, target-style logic, and reference-derived logic now coexist in one component workspace
  - tire size can prefill circumference through the reference dataset path
  - pressure display can switch between `kPa` and `psi` in the UI
  - equivalent `RRC` / `crr1@120` support was introduced for engineering-style tire entry
- `Aero`
  - moved into `Vehicle Parameters`
  - supports inherited/reference behavior and explicit applied change behavior
  - can stage/update baseline reference CdA when intentionally requested
- `Brakes`
  - own component section, separated from generic delta-only treatment
  - baseline-aware update path retained where relevant
- `Parasitics / Hubs / Axle`
  - own component section
  - baseline-aware update path retained where relevant
- `Trailer`
  - placeholder slot remains present as architecture reserve
- `Transmission`
  - handled separately from normal components by design
  - still treated as a component-like contributor in the workflow summary and review

Delivered - VDE results and review

`Results` has been reshaped into a pre-save review layer instead of another editing surface.

Current review layer includes:

- performance summary (`VDE_TOTAL`, `VDE_NET`, cycle, phase outputs, warnings)
- review status
- working scenario summary
- semantic change summary
- reference vs working comparison rows
- staged save payload view
- technical detail expanders

Important rule preserved:

- `Results` does not create a second physical model
- `Results` explains the existing preview and existing save payload
- `Save / Edit` still consumes the same payload path

Delivered - Powertrain Scenario foundation

`PWT Fuel Energy` has been repositioned as `Powertrain Scenario`.

Main page:

- `pages/Powertrain_Scenario.py`

Estimator tabs:

1. `Context & Energy`
2. `Powertrain Inputs`
3. `Estimation Engine`
4. `Results & Save`
5. `Saved Estimates`

What is already working in this estimator-oriented structure:

- compact active VDE source bar kept globally visible across the estimation workflow
- explicit scenario identity/context section
- explicit choice between `VDE_NET` and `VDE_TOTAL`, with `VDE_NET` preferred when available
- local electrification override kept draft-only inside the estimator flow
- no silent mutation of `vde_db` through BEV placeholder behavior
- `Manual / Imported`, `Physics Simple`, and `Regression` share the same common estimation contract path
- `Results & Save` now behaves as the common review/save surface for the estimation methods
- `Saved Estimates` is separated as its own responsibility instead of being buried in the builder

Delivered - Regression inline usability

Regression was not pushed entirely into analysis/reporting.

When `Regression` is selected in `Estimation Engine`, the same workspace now keeps:

- peer filters
- inline scatter preview
- active dataset size
- short Urban / Highway / Combined model summary
- warnings for empty/small data situations
- current scenario estimate

Important architectural decision now reflected in code:

- setup and analysis do not maintain separate regression filter worlds
- the same regression state is reused so the user is not calibrating one model and reviewing another

Delivered - Comparison/report split

Comparison and reporting concerns were separated from the estimator flow.

New reporting page:

- `pages/Comparison_Report.py`

Current comparison/report tabs:

1. `Scenario Compare`
2. `Method Analysis`
3. `Peers & Outlook`

Why this matters:

- `Powertrain Scenario` stays estimation-first
- benchmark, peer context, and method-deeper review stop polluting the save workflow
- the code is already positioned for a future richer comparison/report studio without forcing that complexity into the estimator page

Draft safety and source-change behavior

One important behavior was added for safety:

- when the active VDE source changes in `Powertrain Scenario`, the draft estimation state is reset instead of silently reusing stale scenario context

This directly addresses one of the sprint validation concerns:

- changing the active VDE should not keep an old draft alive invisibly

What is intentionally still legacy or transitional

These items are still intentionally transitional for safety and compatibility:

- final Fuel / Powertrain save path still adapts the resolved estimation result into the current `fuelcons_db` payload contract
- some comparison/report logic is still internally anchored to current VDE-oriented data access even though it is no longer part of the estimator workflow
- `Save / Edit` in `VDE Setup` still contains legacy maintenance behavior that should only be fully retired after DB-path validation is complete
- some VDE component editors are still hybrid, meaning the UI is already structured but the underlying persistence/provenance model is not yet fully normalized per component DB

Validation status

What is already true by implementation:

- preview and save are separated in both `VDE Setup` and `Powertrain Scenario`
- `VDE_TOTAL` / `VDE_NET` semantics were moved into explicit service behavior
- `Powertrain Scenario` no longer behaves like a benchmark-first dashboard
- regression has inline visual feedback and shared active filter state

What is still pending before calling the sprint fully closed:

- run the real automated tests in the local environment
- execute end-to-end DB smoke checks for:
  - baseline VDE flow
  - manual/new VDE flow
  - save new
  - update existing
  - saved Powertrain Scenario persistence
  - refresh behavior when source VDE changes
- manually verify the comparison/report page against saved estimates after more real data exists in `fuelcons_db`

Environment note:

- the current local environment hit Python/runtime issues during recent work, so test files exist and code paths were updated, but full automated validation was not completed inside this checkpoint pass

Practical sprint closure checklist

VDE Setup:

- [x] Core preview/save workflow service exists
- [x] Explicit `TOTAL -> NET` semantics implemented
- [x] Workflow page reorganized around the technical flow
- [x] Mass, components, transmission, cycle, preview, and save are separated by responsibility
- [x] Results works as pre-save review
- [x] UI units are display-only and do not alter internal metric storage/calculation
- [ ] Full DB smoke validation still pending
- [ ] Final cleanup of transitional/legacy helpers still pending

Powertrain Scenario:

- [x] Estimator page reorganized into dedicated tabs
- [x] Active VDE source remains visible in the estimator workflow
- [x] `VDE_NET` vs `VDE_TOTAL` choice is explicit
- [x] Manual / Physics / Regression share a common estimation flow
- [x] Regression has inline filters + scatter + model summary
- [x] Review/save is separated from method configuration
- [x] Comparison/report concerns were moved out to a separate page
- [ ] Saved-estimate real-world validation against broader `fuelcons_db` history still pending
- [ ] Provenance/storage migration for richer reporting remains future work, not sprint-MVP work

Out of scope for this sprint

Still out of scope for the implemented sprint close:

- real ML inference/export pipeline
- physics + ML residual engine
- map-based simulation engine
- operating points / gear simulation / mini-Simulink style tooling
- final component DB normalization for every component family
- full regulatory calculation delivery
- full benchmark studio / PPT / BI reporting layer
- production-polish UI pass

Bottom line

Sprint 5 is effectively in MVP-complete territory.

The remaining work is not to invent the architecture anymore. The remaining work is to validate the real DB flows, shake out transitional edges, and decide what gets promoted from “hybrid but working” into “fully normalized” in the next sprint.
