# Notebooks

This folder contains exploratory, ETL, and model-development notebooks.

Production logic must stay in `src/`. Notebooks are for:

- exploration
- diagnostics
- ETL support
- training experiments
- architecture validation before runtime integration

## Current Notebook Roles

- `etl_epa_xlsx_to_sqlite.ipynb`
  - ETL support for EPA-oriented data ingestion into SQLite
- `ML_Regression_VDE.ipynb`
  - experimental modeling and regression work related to VDE, fuel, energy, and CO2 estimation
- `roadload/RoadLoad_Notebook.ipynb`
  - roadload modeling and reference experiments

## ML Notebook Role

`ML_Regression_VDE.ipynb` is an experimental source, not a production runtime.

That means:

- the Streamlit UI does not execute the full notebook
- runtime `ML Prediction` should use an exported artifact
- the artifact is expected under `models/`
- optional ML dependencies are installed with `requirements-ml.txt`

Current artifact path used by the repository:

- `models/powertrain_scenario_ml.joblib`

## Practical Rule

Use notebooks to:

- study data
- build or compare candidate models
- inspect features
- export artifacts

Do not use notebooks to:

- host production UI logic
- replace runtime service contracts
- become the inference path of the application

## Related Documentation

- [Sprint 5 Closure](../docs/SPRINT_5_CLOSURE.md)
- [Powertrain Scenario Guide](../docs/POWERTRAIN_SCENARIO_GUIDE.md)
- [ML / SHAP / Nearest Peers](../docs/ML_SHAP_NEAREST_PEERS.md)

## Naming Convention

Preferred prefixes:

- `etl_*` for ingestion pipelines
- `eda_*` for exploratory analysis
- `ml_*` for modeling experiments
- `diag_*` for diagnostics and explainability
