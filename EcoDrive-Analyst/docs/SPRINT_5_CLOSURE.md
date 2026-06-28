# Sprint 5 Closure

Date: 2026-06-28

## Purpose

Sprint 5 closes two real delivery tracks:

1. consolidate `VDE Setup` as the physical roadload workflow
2. establish `Powertrain Scenario` as the disciplined estimation block that consumes resolved VDE energy

The sprint also leaves `Comparison Report` as a separate early reporting space instead of overloading the estimator workflow with benchmark responsibilities.

## Product State After Sprint 5

The current EcoDrive product is organized around three blocks:

### 1. VDE Setup

Role:

- build and review physical roadload scenarios
- manage mass, components, transmission, cycle preview, results, and persistence

Status:

- implemented in MVP form
- physically traceable
- preview and save separated

### 2. Powertrain Scenario

Role:

- estimate fuel, electric energy, and CO2 from a resolved VDE source
- keep energy-basis selection explicit
- support multiple estimation methods with one common result/save contract

Status:

- implemented as the new estimation-first workflow

### 3. Comparison Report

Role:

- initial space for comparison, benchmark, and reporting exploration

Status:

- separated from the estimator path
- still evolving

## VDE Setup Delivered Scope

Current workflow:

1. `Scenario Setup`
2. `Vehicle Parameters`
3. `Roadload Build-up`
4. `Cycle & Preview`
5. `Results`
6. `Save / Edit`

Delivered concepts:

- explicit distinction between `VDE_TOTAL` and `VDE_NET`
- transmission as the bridge from `TOTAL -> NET`
- roadload-basis-aware workflow
- component sections with current/change/applied semantics
- results as pre-save review instead of another editing surface
- display-only unit switching between Metric and US customary

## Powertrain Scenario Delivered Scope

Current workflow:

1. `Context & Energy`
2. `Powertrain Inputs`
3. `Estimation Engine`
4. `Results & Save`
5. `Saved Estimates`

Delivered methods:

- `Manual / Imported`
- `Physics Simple`
- `Regression`
- `ML Prediction`

Planned-only placeholders:

- `Physics + ML Residual`
- `Map-Based Simulation`

Delivered rules:

- the page is an estimation block, not a full benchmark dashboard
- it consumes resolved VDE energy and does not recalculate roadload
- `VDE_NET` is preferred when available, but `VDE_TOTAL` remains selectable
- draft-only overrides do not silently mutate `vde_db`

## ML / SHAP / Nearest Peers Delivered Scope

Current capabilities:

- runtime ML inference when an exported artifact is available
- artifact-aware status reporting
- SHAP-style explainability messaging when available in the current integration
- nearest-peer guidance and peer-group statistics
- investigation hints based on estimation and peer context

Current constraints:

- notebook is not executed as runtime UI logic
- inference depends on exported artifacts and optional dependencies
- peer guidance depends on dataset coverage and consistency

## Comparison Report State

The current comparison area is intentionally not the center of the estimator.

It is the starting point for future:

- scorecards
- benchmark of similar vehicles
- reporting / BI layers
- regulatory labels and comparison surfaces

## Known Limitations

- ML depends on exported artifacts in `models/`.
- SHAP depends on model form and dependency compatibility.
- Nearest Peers depends on dataset coverage and technical consistency.
- Regression remains empirical and should be read with visual/scatter feedback.
- Regulatory labels are still early-stage.
- Performance simulation is still planned.
- RAG / external technical-data agents are still planned.
- hidden component priors are backlog only, not a current capability

## Practical Validation State

What is considered delivered:

- VDE workflow consolidation
- estimation-block restructuring
- separate comparison/report page
- common estimation contracts
- initial ML / SHAP / peer guidance flow

What still remains as final operational validation:

- real DB smoke checks across save/update paths
- broader saved-scenario comparison usage with more production-like data
- selective UX refinement after more user walkthroughs

## Next-Step Direction

Natural next layers after Sprint 5:

- richer `Comparison Report` / benchmark studio
- residual or hybrid ML guidance beyond direct prediction
- stronger provenance and reporting depth
- future performance and regulatory capability build-out

## Related Docs

- [VDE Setup Guide](VDE_SETUP_GUIDE.md)
- [Powertrain Scenario Guide](POWERTRAIN_SCENARIO_GUIDE.md)
- [ML / SHAP / Nearest Peers](ML_SHAP_NEAREST_PEERS.md)
- [Sprint 5 Architecture Checkpoint](sprints/SPRINT_5_VDE_FUEL_ARCHITECTURE_2026-06-19.md)
