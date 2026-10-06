# Sprint 12F — Engineering Notebook Program Setup

## Status: `NOTEBOOK_PROGRAM_READY — BEGIN_12F_01`

```text
Planned notebook sequence                     11
Notebook scaffolds created                      1
12F-01 source load                         PASSED
EPA source rows                             30194
EPA source columns                             67
Missing input/path blockers                      0
Production/runtime DB modified?                 NO
```

## Sequence

| Order | Notebook | Dependency | Setup status |
|---:|---|---|---|
| 01 | EPA ingest and grain | EPA raw workbook | Scaffold ready |
| 02 | EPA non-BEV EDA | 12F-01, EPA raw, 12E.2 staging DB | Planned |
| 03 | EPA BEV energy EDA | 12F-01, 12E.3A outputs | Planned |
| 04 | WLTP/JRC ingest and grain | JRC raw workbook | Planned |
| 05 | JRC non-BEV EDA | 12F-04 | Planned |
| 06 | JRC BEV EDA | 12F-04 | Planned |
| 07 | Vehicle Demand model | 12F-02, 12F-05, 12E.2 staging DB | Planned |
| 08 | FuelCons modelling | 12F-07, RUN/FuelCons lineage | Planned |
| 09 | Cross-legislation benchmark | 12F-03, 12F-06, 12F-08 | Planned |
| 10 | Canonical dataset build | Prior ingest/modelling notebooks | Planned |
| 11 | Disposable DB ingestion | 12F-10 outputs, canonical schema | Planned |

Every entry has a 5–15 minute `live_coding_slice` in `notebooks/notebook_manifest.json`.

## Exact available inputs

- `etl/data/raw/epa_testcar/epa_testcar_2026_raw.xlsx`
- `etl/data/raw/wltp_jrc/Data_PV_fleet_2021_EU_PYCSIS.xlsx`
- `etl/data/processed/sprint_12e3_electric_unit_audit/electric_energy_sanity_dataset.csv`
- `etl/config/electric_consumption_overrides.csv`
- `etl/data/staging/sprint_12e2_epa_fuelcons/eco_drive_canonical_epa_fuelcons.db`
- `etl/data/processed/sprint_12e2_epa_fuelcons/run_source_row_lineage.csv`
- `etl/data/processed/sprint_12e2_epa_fuelcons/fuelcons_run_adoption.csv`
- `etl/schema/canonical_schema_v1.sql`

All manifest paths are repository-relative. The five analytical CSVs planned for 12F-10 and the disposable SQLite database planned for 12F-11 are declared future outputs, not missing setup inputs.

## 12F-01 scaffold

`notebooks/12F_01_epa_ingest_and_grain.ipynb` contains short sections for:

1. the source-grain question;
2. imports and portable path discovery;
3. raw workbook loading;
4. shape and model-year counts;
5. relevant identity/test columns;
6. first row/MMY/test/configuration grain counts;
7. real multi-state examples;
8. open questions for the next live-coding step.

The notebook was executed from the `notebooks/` working directory during `test_first_notebook_executes_and_loads_epa_source`; it loaded the current EPA workbook as 30,194 rows × 67 columns. It deliberately does not save output or reproduce the full production grain-closure implementation.

## Output dependencies

- 12F-01 has no persisted output dependency.
- 12F-03 consumes the closed 12E.3A sanity dataset rather than hidden notebook state.
- 12F-10 will create five small staging CSVs under `notebooks/_data/`.
- 12F-11 will consume those CSVs and write only `notebooks/_data/12f11_canonical_notebook_demo.db`.
- No notebook points to either runtime database as a write target.

## Safety and evidence

The setup writes only notebook program files and this report. No production ETL, schema, raw source, application page, physics code, or database was modified. Runtime database fingerprints remained:

- `eco_drive.db`: `CC27CDF22AE189E39F9F45609CCD23C650E822BB20655C552FF2BD89D97F4262`
- `eco_drive_qa.db`: `0EB4A5EC0F402E6B9E1F7D76372EA30068670B16F994D495394613DAFA2CA569`

Direct tests: manifest sequencing/live-coding slices, single-notebook scope, portable existing input, and actual EPA source loading. Notebook contents and program boundaries are inspection-supported.
