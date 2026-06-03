# ETL Notebook - EPA XLSX to SQLite

Notebook: `notebooks/etl_epa_xlsx_to_sqlite.ipynb`  
Purpose: ingest EPA spreadsheets, transform into EcoDrive schema, and load into SQLite tables used by the app.

## Scope
- Source: EPA datasets in XLSX/CSV format.
- Target tables: `vde_db` and related consumption tables (when applicable).
- Output: normalized rows ready for preview/compute/save flows in Streamlit pages.

## Pipeline Stages

### 1) Extract
- Read source files from local path(s).
- Load raw sheets into DataFrames.
- Validate required columns exist before transform.

Expected output:
- `df_raw` (or equivalent), one DataFrame per source/sheet.

### 2) Transform - Standardization
- Normalize column names and types.
- Convert units to project conventions:
  - `A [N]`
  - `B [N/kph]`
  - `C [N/kph^2]`
  - `mass_kg`
- Standardize categorical fields:
  - `legislation`
  - `category`
  - make/model/year labels

Expected output:
- `df_vde` canonical frame aligned with DB schema.

### 3) Transform - Derived Fields
- Apply helper functions from `src/vde_core/services.py` and `src/vde_core/utils.py` as needed.
- Compute/adjust phase and decomposition fields when present.
- Prepare EPA/WLTP compatibility fields used downstream by app pages.

Expected output:
- enriched DataFrame with direct DB mapping and optional diagnostics.

### 4) Load
- Insert rows with repository/db helpers (`insert_vde`, `insert_fuelcons`, etc.).
- Prefer batched inserts and one final commit step when possible.
- Keep idempotency strategy explicit (append vs update vs replace).

Expected output:
- persisted rows in SQLite with traceable source metadata.

### 5) QA / Validation
- Run quick checks:
  - row counts
  - null/required-field checks
  - schema check (`PRAGMA table_info`)
  - sample row preview
- Compare computed fields against expected ranges.

### 6) Safety / Recovery
- Keep backup instructions visible before destructive actions.
- Never run delete/drop cells in routine loads.
- Archive exploratory cells separately from production ETL flow.

## Recommended Notebook Layout
1. Config / paths  
2. Extract  
3. Transform (standardization)  
4. Transform (derived fields)  
5. Load  
6. QA  
7. Optional diagnostics/plots  
8. Dangerous ops (isolated and disabled)

## Operational Notes
- This notebook is an ETL utility, not production runtime code.
- Production app must consume validated DB outputs, not notebook internals.
- If ETL logic stabilizes, migrate critical pieces to Python modules under `src/vde_core/`.

