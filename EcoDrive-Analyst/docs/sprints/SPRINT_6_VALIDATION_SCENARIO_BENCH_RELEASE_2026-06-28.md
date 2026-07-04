Sprint 6 - Validation, Scenario Bench UX, PSE Storytelling, and Release Readiness

Date: 2026-06-28
Status: planned

Objective

Sprint 6 should turn EcoDrive from a technically capable MVP into a demonstrable and trustworthy product flow.

The sprint is intentionally not about opening heavy new capabilities. It is about validating the real workflow, improving the main Powertrain Scenario UX, repositioning the page around cycle-effective powertrain conversion, and hardening the project for demo and release-style use.

Sprint goals:

1. Validate one official happy path end to end.
2. Introduce a first `Scenario Bench` UX layer in `Powertrain Scenario`.
3. Make `Powertrain System Efficiency` / `eta_pt_cycle` explicit in the page storytelling.
4. Add an operational `Estimate Confidence` layer.
5. Make `Comparison Report` usable as a live report v0.
6. Stabilize environment, tests, and release-facing documentation.

Product intent for this sprint

After Sprint 5, the product shape is already clear:

- `VDE Setup` is the physical and traceable roadload workflow
- `Powertrain Scenario` is the estimation block
- `Comparison Report` is the separated benchmark/report direction

Sprint 6 should make that product shape easier to understand in a short walkthrough:

- a user can create or select a VDE scenario
- understand `Vehicle Demand -> Powertrain System Efficiency -> Final Result`
- estimate fuel / energy / CO2 with a consistent review/save flow
- open a report-style view that summarizes the scenario and its confidence context

Powertrain storytelling target

The main narrative for `Powertrain Scenario` in this sprint should be:

1. `Maneuver / Cycle`
2. `Vehicle Demand`
3. `Powertrain System Efficiency`
4. `Final Result`
5. `Confidence`

`VDE` remains important, but it should be presented as the vehicle-demand input rather than the visual protagonist of the page.

Official demo happy path

The official happy path for this sprint is:

1. `VDE Setup`
2. `Powertrain Scenario`
3. `Comparison Report`

In product, UX, and front-end language, `happy path` means the main success flow the user should be able to follow without friction, detours, or failure states.

For this sprint, it is the "demo flow that must work well" before edge cases and fallback states are evaluated.

Expected walkthrough:

1. Select or create a scenario in `VDE Setup`
2. Define mass, roadload basis, components, and transmission
3. Resolve `VDE_TOTAL` and, when available, `VDE_NET`
4. Preview and save
5. Open `Powertrain Scenario`
6. Select the resolved VDE source
7. Read the `Scenario Bench`
8. Understand `Vehicle Demand`
9. Choose the energy basis
10. Stage powertrain inputs
11. Choose one estimation method
12. Estimate or derive `PSE` when possible
13. Generate fuel / energy / CO2
14. Show ML / SHAP / nearest peers only when available
15. Review and save
16. Open `Comparison Report`
17. Show executive summary, provenance, warnings, peer context, and `PSE`

Acceptance target:

- this flow should be demonstrable in about 5 minutes
- the user should not need to explain hidden architecture details while navigating it

Planned delivery tracks

Track A - Happy path validation

Main intent:

- prove the real workflow works with save/update behavior and without hidden state drift

Planned work:

- define one official demo scenario path
- validate `VDE Setup -> save`
- validate `Powertrain Scenario -> Review & Save`
- validate `Comparison Report -> live summary`
- verify state reset behavior when the active VDE source changes
- document known demo limitations explicitly instead of hiding them

Likely code touchpoints:

- `pages/VDE_Setup.py`
- `pages/Powertrain_Scenario.py`
- `pages/Comparison_Report.py`
- `src/vde_app/components/vde_setup.py`
- `src/vde_app/components/pwt_fuel_energy.py`

Track B - Scenario Bench UX

Main intent:

- make `Powertrain Scenario` feel like a lightweight virtual test bench instead of a raw form stack

Planned work:

- add a top-level `Scenario Bench` or equivalent block in `Powertrain Scenario`
- show a simple dyno / roller-bench visual metaphor
- organize the page around:
  - `Maneuver / Cycle`
  - `Vehicle Demand`
  - `Powertrain System Efficiency`
  - `Final Result`
- show key scenario badges:
  - scenario name
  - maneuver / intent
  - drive cycle
  - energy basis
  - `VDE_TOTAL`
  - `VDE_NET`
  - `ABC`
  - `PSE` status
  - estimation method
  - predicted fuel / energy / CO2 when available
  - confidence / status
- add hotspot navigation for:
  - `Driver / Cycle`
  - `Roadload / VDE`
  - `Powertrain Efficiency`
  - `Transmission`
  - `Engine / Fuel`
  - `Electric / Battery`
  - `ML / SHAP / Peers`
  - `Results`
- switch a lower detail panel when the active hotspot changes

Important constraints:

- no 3D
- no heavy animation
- no complex canvas work
- the visual should organize the page, not replace the data

Preferred implementation direction:

- keep this inside the existing `Powertrain Scenario` page structure
- use the bench as lightweight navigation across the estimation workflow
- reuse the current request/result/provenance state instead of creating a parallel state model

Track C - Powertrain System Efficiency

Main intent:

- make `PSE` / `eta_pt_cycle` visible as the conceptual bridge between vehicle demand and final consumption

For this sprint:

- derive `PSE` when the page has both demand and observed/estimated energy consumed
- show `PSE` even when it is only a derived operational metric
- show `PSE pending` or `PSE unavailable` when inputs are insufficient
- model `PSE` as a first-class interpretation layer with explicit mode/source metadata
- keep the interpretation warning visible:
  - `PSE is cycle-effective and should not be interpreted as pure engine efficiency.`

PSE contract language:

- `mode`
  - `assumed`
  - `derived`
  - `predicted`
  - `unavailable`
- `source`
  - `imported_result`
  - `physics_assumption`
  - `physics_result`
  - `regression_fuel_estimate`
  - `ml_fuel_prediction`
  - `ml_pse_prediction`
  - `unavailable`
- `target_type`
  - `fuel_direct`
  - `energy_direct`
  - `pse_direct`
  - `observed_result`
  - `assumption`

Important naming rule:

- current ML artifacts predict final fuel / energy outputs
- when `PSE` is derived from current ML output, label it as `Derived from ML fuel prediction`
- do not label current runtime behavior as `ML-predicted PSE`
- direct `PSE` prediction is a future ML target unless an artifact is trained specifically for `eta_pt_cycle`

Important constraint:

- do not expand the ML pipeline just to train a new `PSE` target in this sprint

Likely code touchpoints:

- `src/vde_core/fuel_estimation.py`
- `src/vde_core/fuel_energy.py`
- optional new helper such as `src/vde_core/powertrain_efficiency.py`
- `src/vde_app/components/pwt_fuel_energy.py`

Track D - Estimate Confidence layer

- `pages/Powertrain_Scenario.py`
- `src/vde_app/components/pwt_fuel_energy.py`
- optional new helper such as `src/vde_app/components/scenario_bench.py`

Main intent:

- expose a human-readable confidence/status layer without pretending it is formal validation

The confidence layer should surface statuses such as:

- `Measured / Imported`
- `Physics Estimate`
- `Regression Estimate`
- `ML Prediction`
- `PSE Available`
- `PSE Unavailable`
- `Low Coverage`
- `Missing Critical Inputs`
- `Out of Domain`
- `Draft Only`
- `SHAP Available`
- `SHAP Unavailable`
- `Peer Group High Quality`
- `Peer Group Medium Quality`
- `Peer Group Low Quality`

The confidence layer should consider:

- selected method
- input completeness
- `VDE_NET` availability
- VDE source origin / state
- `PSE` source and availability
- ML artifact load status
- SHAP availability
- nearest-peer group quality
- peer dispersion
- out-of-domain warnings

Important rule:

- confidence is operational guidance, not formal certification

Preferred implementation direction:

- consolidate confidence/status logic near the shared estimation contract
- keep UI rendering separate from confidence derivation logic
- make the same confidence state reusable in:
  - `Scenario Bench`
  - `Results & Save`
  - `Comparison Report`

Likely code touchpoints:

- `src/vde_core/fuel_estimation.py`
- `src/vde_core/ml_prediction.py`
- `src/vde_core/nearest_peers.py`
- optional new helper such as `src/vde_core/estimate_confidence.py`

Track E - Comparison Report v0

Main intent:

- make `Comparison Report` a clean live report surface without turning it into a full BI product

The report should show the new storytelling:

- `Vehicle Demand`
  - `VDE_TOTAL`
  - `VDE_NET`
  - `ABC`
  - cycle / maneuver
- `Powertrain System Efficiency`
  - `PSE` / `eta_pt_cycle`
  - source / method
  - status / warning
- `Final Result`
  - fuel
  - energy
  - CO2
- `Confidence`
  - method
  - SHAP summary when available
  - nearest peers summary
  - peer median / std / delta / z-score when available
  - warnings
  - limitations
  - source / revision status

Important product decision:

- report v0 is a live view with clear source/status messaging
- versioned export/report packages stay out of scope

Likely code touchpoints:

- `pages/Comparison_Report.py`
- `src/vde_app/components/pwt_fuel_energy.py`
- `src/vde_core/comparison_report_service.py`

Track F - Method hardening

Methods in scope:

- `Manual / Imported`
- `Physics Simple`
- `Regression`
- `ML Prediction`

All of them should:

- consume the same VDE context
- respect the selected energy basis
- update or expose `PSE` when possible
- pass through the same review/save flow
- show warnings consistently
- fail defensively when optional ML/SHAP/peers are unavailable
- never mutate `VDE Setup` state
- never persist physical assumptions automatically because of ML, SHAP, or peers

Method-specific hardening:

- `Regression`
  - preserve filters, scatter/preview, and empirical framing
- `ML Prediction`
  - report artifact status
  - report feature coverage / missing features
  - report coverage / out-of-domain state
  - keep current output framing as fuel / energy / CO2 prediction
  - show `PSE` as derived from ML fuel prediction unless a future `PSE` artifact exists
- `SHAP`
  - show only when supported
  - group into engineering-oriented blocks when possible
  - keep the non-causality warning visible
- `Nearest Peers`
  - show peers, peer metrics, and peer quality
- `Investigation Hints`
  - remain advisory only
  - never be presented as root-cause proof

Likely code touchpoints:

- `src/vde_core/fuel_estimation.py`
- `src/vde_core/ml_prediction.py`
- `src/vde_core/ml_explainability.py`
- `src/vde_core/nearest_peers.py`
- `src/vde_app/components/pwt_fuel_energy.py`

Track F - Environment and release readiness
Track G - Environment and release readiness

Main intent:

- prove the project is installable, runnable, and documentable in a clean environment

Validation targets:

- create a fresh `venv`
- install `requirements.txt`
- install optional `requirements-ml.txt`
- run the app
- run compile validation
- run tests
- validate ML artifact presence / status
- confirm dependencies are documented

Expected commands:

```text
python -m compileall -q pages src tests
python -m unittest discover tests
streamlit run app.py
```

Repository hygiene:

- keep `requirements.txt` in UTF-8
- document `requirements-ml.txt`
- ensure `.gitignore` covers:
  - `.venv/`
  - `__pycache__/`
  - `*.pyc`
  - `*.zip`
  - `*.7z`
  - `.pytest_cache/`
  - heavy local artifacts

Likely code touchpoints:

- `README.md`
- `requirements.txt`
- `requirements-ml.txt`
- `.gitignore`
- validation docs under `docs/`

Execution order

Recommended sequencing for the sprint:

1. Validate the official happy path and capture workflow gaps.
2. Add the `Scenario Bench` shell and hotspot navigation.
3. Make `PSE` explicit in the flow and in the review/save surface.
4. Consolidate the confidence layer and surface it in the bench and review/save flow.
5. Upgrade `Comparison Report` to v0 live-report quality.
6. Harden method failure states and provenance messaging.
7. Run environment, test, and release-readiness checks.
8. Update `README` and sprint/guide docs.

Why this sequence:

- the happy path defines what must stay stable
- the bench UX should organize the workflow that already exists
- confidence/reporting are clearer after the common workflow is validated

Acceptance criteria

Sprint 6 is complete when:

1. An official demo happy path exists and is usable.
2. `Powertrain Scenario` includes a first `Scenario Bench` UX block.
3. The bench tells the story `Vehicle Demand -> PSE -> Final Result`.
4. `VDE` appears as the demand input, not as the dominant visual story.
5. Hotspots switch a lower detail panel.
6. `PSE` appears as an explicit layer in the page.
7. Estimate confidence/status is visible in the workflow.
8. `Comparison Report` v0 shows a useful executive summary.
9. `Manual`, `Physics`, `Regression`, and `ML` all use the same review/save discipline.
10. SHAP and nearest peers fail defensively when unavailable.
11. The app can be installed and run in a clean environment.
12. Main tests pass or remaining limitations are explicitly documented.
13. `README` and related docs reflect the current product flow and explain `PSE` correctly.

Out of scope

The following items remain outside Sprint 6:

- RAG
- MCP integration
- DBSCAN / HDBSCAN clustering
- hidden component priors as a full capability
- real inferred tire / brake / aero / transmission physics completion
- official `INMETRO`, `CAFE`, or `UNECE` regulatory delivery
- real performance simulation such as `0-100`, `60-100`, `80-120`, or `Vmax`
- engine map simulation
- mini-Simulink or ODE longitudinal simulation
- PowerPoint export
- large schema migration
- external technical-data search agents

Suggested sprint-end reporting

At sprint close, the project should be able to answer with:

- files changed
- screenshots or a written demo-flow summary
- tests executed
- known limitations
- items moved to backlog

Bottom line

Sprint 6 should not make EcoDrive broader.

It should make EcoDrive clearer, more stable, and easier to demonstrate with confidence.
