# Notebooks

This folder contains analysis and ETL notebooks.  
Keep production logic in `src/` and use notebooks for exploration, ingestion support, and diagnostics.

## Current notebooks
- `etl_epa_xlsx_to_sqlite.ipynb` - EPA ETL pipeline (extract/transform/load into SQLite).
- `ML_Regression_VDE.ipynb` - regression experiments around VDE and consumption.
- `roadload/RoadLoad_Notebook.ipynb` - roadload modeling experiments/reference.

Related docs:
- `docs/notebooks/etl_epa_xlsx_to_sqlite.md`
- `docs/notebooks/etl_epa_xlsx_to_sqlite_narrative.md`

## Naming convention
- Prefer prefixes by purpose:
  - `etl_*` for ingestion pipelines
  - `eda_*` for exploratory analysis
  - `ml_*` for modeling experiments
