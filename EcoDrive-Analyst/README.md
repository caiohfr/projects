# EcoDrive Analyzer

EcoDrive Analyzer is a Streamlit application for roadload engineering, VDE workflow management, powertrain consumption estimation, and early comparison reporting.

The current product is organized around four main blocks:

1. `Database Management`
2. `VDE Setup`
3. `Powertrain Scenario`
4. `Comparison Report`

`Database Management` is the official controlled catalog-administration
workflow. `VDE Setup` is the stable, feature-frozen engineering workflow.
`Comparison Report` remains the next reporting focus; `Powertrain Scenario`
remains a future workstream.

## Current Product Blocks

### Database Management

`Database Management` is the official operational surface for controlled
catalog records:

- VDE baselines and saved VDE records
- fuel-consumption records and their VDE references
- tire records
- transmission, brake, axle & hubs, and parasitic components
- staged create, update, archive, restore, and duplicate changes
- impact review, dependency resolution, and append-only change receipts
- controlled spreadsheet templates with review-before-commit imports

The former direct Tire Database page is preserved under `docs/archive/pages/`
as historical reference code. It is not part of the ordinary application
navigation.

### VDE Setup

`VDE Setup` is the physical and traceable workflow for:

- canonical baseline and effective-baseline corrections
- scenario proposals and Walk From lineage
- Mass, Tire, Aero, Transmission, Brake, Axle & Hubs, and Parasitics
- roadload TOTAL / NET and cycle analysis
- metadata, provenance, engineering comparison, and audit
- append-only Save and historical Reload
- deterministic synthetic QA data

Core idea:

- `VDE_TOTAL` is the demand derived from `ABC_TOTAL`
- `VDE_NET` is available only when transmission losses / neutral drag are resolved
- Historical EcoDrive VDE records stored `VDE_TOTAL` in `vde_net_mj_per_km` (Package 7G
  normalized affected rows). New code must read TOTAL/NET through
  `src/vde_core/vde_net_total_contract.py::canonical_vde_read()` and must never use
  `vde_net_mj_per_km` as a fallback for TOTAL — see
  `docs/sprints/PACKAGE_7G_VDE_NET_TOTAL_CONTRACT.md`.

### Powertrain Scenario

`Powertrain Scenario` consumes a resolved VDE source and estimates:

- fuel consumption
- electric energy
- CO2

Inside that page, the current interpretation flow is:

1. `Vehicle Demand`
2. `Powertrain System Efficiency (PSE)`
3. final fuel / electric result

Supported estimation methods in the current product:

- `Manual / Imported`
- `Physics Simple`
- `Regression`
- `ML Prediction`

Planned but not delivered as runtime engines:

- `Physics + ML Residual`
- `Map-Based Simulation`

Important boundary:

- `Powertrain Scenario` does not recalculate roadload
- it uses an already resolved energy basis such as `VDE_TOTAL` or `VDE_NET`

Practical reading:

- `VDE_TOTAL` or `VDE_NET` tells us how much energy the vehicle demands on the cycle
- `PSE` tells us how effectively the powertrain converts supply energy into that delivered demand
- fuel, electricity, and CO2 are the final outputs of that relationship

### Comparison Report

`Comparison Report` is now its own space for:

- scenario comparison
- method analysis
- peer outlook / benchmark direction

It is intentionally still an MVP surface. It should be read as the first step toward a future report / benchmark studio, not as a finished BI layer.

## Stable Product Status

```text
EcoDrive
|
|-- Database Management   stable / official catalog administration
|-- VDE Setup             stable / feature frozen
|-- Comparison Report     next development focus
`-- Powertrain Scenario   future
```

## Documentation Index

Sprint documentation:

- [Sprint 5 Closure](docs/SPRINT_5_CLOSURE.md)
- [Sprint 6 Plan](docs/sprints/SPRINT_6_VALIDATION_SCENARIO_BENCH_RELEASE_2026-06-28.md)
- [Sprint 7 Database Management Checkpoint](docs/sprints/SPRINT_7_DATABASE_MANAGEMENT.md)
- [VDE Setup Guide](docs/VDE_SETUP_GUIDE.md)
- [VDE Setup v2.2 Final Stable Contract](docs/VDE_SETUP_V22_FINAL_CHECKPOINT.md)
- [Powertrain Scenario Guide](docs/POWERTRAIN_SCENARIO_GUIDE.md)
- [ML / SHAP / Nearest Peers](docs/ML_SHAP_NEAREST_PEERS.md)

Architecture references:

- [Project Structure](docs/architecture/project_structure.md)
- [Roadload Pipeline](docs/architecture/roadload_pipeline.md)
- [UI and Backend Boundary](docs/architecture/ui_backend_boundary.md)
- [Sprint 5 Architecture Checkpoint](docs/sprints/SPRINT_5_VDE_FUEL_ARCHITECTURE_2026-06-19.md)

Notebook notes:

- [Notebooks README](notebooks/README.md)

## Repository Structure

```text
EcoDrive-Analyst/
|-- app.py
|-- data/
|-- docs/
|-- models/
|-- notebooks/
|-- pages/
|-- src/
|-- tests/
|-- requirements.txt
`-- requirements-ml.txt
```

Important runtime pages:

- `pages/VDE_Setup.py`
- `pages/Powertrain_Scenario.py`
- `pages/Comparison_Report.py`
- `pages/Database_Management.py`

## How To Run

Create the local environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional ML dependencies:

```bash
pip install -r requirements-ml.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Testing And Validation

Fast syntax / import pass:

```bash
python -m compileall -q pages src tests
```

Run the test suite:

```bash
python -m unittest discover tests
```

If you want a narrower pass while iterating:

```bash
python -m unittest tests.test_vde_workflow_service tests.test_fuel_estimation
```

## ML Runtime Notes

`ML Prediction` is an inference capability, not a notebook execution mode.

Current expectations:

- the notebook remains an experimental/training source
- runtime inference expects an exported artifact under `models/`
- the current repository already includes a Powertrain Scenario artifact:
  - `models/powertrain_scenario_ml.joblib`
- optional ML dependencies live in `requirements-ml.txt`
- the current ML artifact predicts final fuel / energy outputs
- `PSE` shown in the UI is currently derived from those outputs plus the active demand basis
- direct ML prediction of cycle-effective `PSE` is future work unless a dedicated artifact is trained for that target

Possible ML runtime states:

- artifact found and loaded
- artifact missing
- artifact load failed
- missing features
- partial / out-of-domain coverage

See [ML / SHAP / Nearest Peers](docs/ML_SHAP_NEAREST_PEERS.md) for the detailed explanation.

## Known Limitations

- ML runtime depends on an exported artifact and compatible dependencies.
- SHAP availability depends on model form and explainability compatibility.
- Nearest Peers quality depends on dataset coverage and consistency.
- Regulatory / label benchmarking is still an early scaffold.
- Performance simulation is still planned.
- `Physics + ML Residual` and `Map-Based Simulation` are planned, not production engines.
- Comparison / benchmark reporting is still in an MVP stage.
- Hidden component priors are future backlog, not a delivered causal inference capability.
- current `PSE` is cycle-effective system efficiency, not pure engine efficiency.
- temperature and ambient-pressure roadload conditions are deferred to derived
  scenarios in Comparison Report; they are not persisted by VDE Setup.

## Sprint 5 Status

Sprint 5 delivered the product foundation the project needed:

- `VDE Setup` as a disciplined physical workflow
- `Powertrain Scenario` as an estimation-first page
- `Comparison Report` as a separate reporting direction
- shared estimation contracts for manual, physics, regression, and ML paths
- first-class `PSE` interpretation across estimator, review, and comparison flows
- initial ML explainability and peer-guidance capabilities

See [Sprint 5 Closure](docs/SPRINT_5_CLOSURE.md) for the consolidated close-out.

## Sprint Closure

Current sprint closure:

- `VDE Setup` moved to spreadsheet-first technical input mode.
- `Powertrain Scenario` is consolidated as a guided workflow:
  `Scenario Pairing -> Baseline Estimate -> Technology Delta -> Result & Save`.
- Technical diagnostics were moved behind advanced / technical details by default.
- No schema, VDE formula, or ML training changes were introduced in this closure.

Next steps for Sprint 7:

- `Comparison Report` v0
- baseline vs proposal storytelling
- delta decomposition
- `VehicleScenario` internal contract
- future `Scenario Builder`
