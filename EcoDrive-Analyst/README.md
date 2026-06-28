# EcoDrive Analyzer

EcoDrive Analyzer is a Streamlit application for roadload engineering, VDE workflow management, powertrain consumption estimation, and early comparison reporting.

The current product is organized around three main blocks:

1. `VDE Setup`
2. `Powertrain Scenario`
3. `Comparison Report`

This repository is now in a stronger Sprint 5 state: `VDE Setup` behaves as the physical roadload workflow, `Powertrain Scenario` behaves as an estimation block, and `Comparison Report` is separated as an early benchmark/report space instead of being mixed into the estimator flow.

## Current Product Blocks

### VDE Setup

`VDE Setup` is the physical and traceable workflow for:

- scenario setup and metadata
- vehicle parameters and mass setup
- roadload basis selection
- component build-up
- transmission losses / TOTAL -> NET bridge
- cycle preview
- results as pre-save review
- save / edit

Core idea:

- `VDE_TOTAL` is the demand derived from `ABC_TOTAL`
- `VDE_NET` is available only when transmission losses / neutral drag are resolved

### Powertrain Scenario

`Powertrain Scenario` consumes a resolved VDE source and estimates:

- fuel consumption
- electric energy
- CO2

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

### Comparison Report

`Comparison Report` is now its own space for:

- scenario comparison
- method analysis
- peer outlook / benchmark direction

It is intentionally still an MVP surface. It should be read as the first step toward a future report / benchmark studio, not as a finished BI layer.

## Documentation Index

Sprint 5 documentation:

- [Sprint 5 Closure](docs/SPRINT_5_CLOSURE.md)
- [VDE Setup Guide](docs/VDE_SETUP_GUIDE.md)
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
- `pages/Tire_Database.py`

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

## Sprint 5 Status

Sprint 5 delivered the product foundation the project needed:

- `VDE Setup` as a disciplined physical workflow
- `Powertrain Scenario` as an estimation-first page
- `Comparison Report` as a separate reporting direction
- shared estimation contracts for manual, physics, regression, and ML paths
- initial ML explainability and peer-guidance capabilities

See [Sprint 5 Closure](docs/SPRINT_5_CLOSURE.md) for the consolidated close-out.
