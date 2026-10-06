# Notebooks

This folder contains exploratory, ETL, and model-development notebooks.

## Sprint 12F Engineering Sequence

The 12F notebooks tell one story: raw regulatory data → source grain →
engineering interpretation → canonical relational data. Production ETL remains
authoritative; notebook transformations stay visible for study and live coding.

| Order | Notebook | Question | Status |
|---:|---|---|---|
| 01 | `12F_01_epa_ingest_and_grain.ipynb` | What is one EPA row? | MVP executable |
| 02 | `12F_02_epa_nonbev_eda.ipynb` | What explains non-BEV fuel/CO2 variation? | MVP executable |
| 03 | `12F_03_epa_bev_energy_eda.ipynb` | Can electric energy be trusted? | MVP executable |
| 04 | `12F_04_wltp_jrc_ingest_and_grain.ipynb` | What does JRC add? | MVP executable |
| 05 | `12F_05_wltp_jrc_nonbev_eda.ipynb` | What patterns exist in JRC non-BEVs? | MVP executable |
| 06 | `12F_06_wltp_jrc_bev_eda.ipynb` | How does the electric JRC population behave? | MVP executable |
| 07 | `12F_07_vehicle_demand_model.ipynb` | How do mass and roadload become VDE? | MVP executable |
| 08 | `12F_08_fuelcons_modelling.ipynb` | How does RUN evidence become FuelCons? | MVP executable |
| 09 | `12F_09_cross_legislation_benchmark.ipynb` | What is safely comparable across programs? | MVP executable |
| 10 | `12F_10_canonical_dataset_build.ipynb` | How do analytical tables become canonical records? | MVP executable |
| 11 | `12F_11_database_ingestion.ipynb` | How is a disposable relational DB loaded safely? | MVP executable |

Machine-readable paths, dependencies, and live-coding slices are in
`notebook_manifest.json`. The MVP notebooks execute independently from
repository-relative inputs; 12F-10/11 additionally demonstrate a small
relational export and disposable database load.

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

The numbered `12F_*` prefix is reserved for the sequential engineering story above.
