# QA Mock Data

This package creates deterministic QA seed data for VDE Setup without touching
production calculations. VDE Setup v2.2 remains the historical implementation
checkpoint for this workflow.

All values are synthetic engineering QA fixtures and must not be treated as production or manufacturer data.

## What gets generated

The current architecture stores `vde_db` and `tire_roadload_db` in the same SQLite database, so QA generation produces a single reproducible file:

- `data/qa/eco_drive_qa.db`

Component fixtures remain in the existing deterministic mock CSV repositories under `data/components/`.

## Generate / regenerate

From the repo root:

```powershell
python scripts/create_qa_mock_dbs.py
python scripts/create_qa_mock_dbs.py --overwrite
```

The script refuses to overwrite paths outside the QA data directory.

## Remove and rebuild

Delete the generated QA database and re-run the script:

```powershell
Remove-Item data/qa/eco_drive_qa.db
python scripts/create_qa_mock_dbs.py
```

## Point the app to the QA database

The app can be pointed to a different SQLite file through `ECO_DRIVE_DB_PATH`.

PowerShell example:

```powershell
$env:ECO_DRIVE_DB_PATH = "data/qa/eco_drive_qa.db"
streamlit run pages/VDE_Setup.py
```

The same environment variable also works for other pages that use `src.vde_core.db`.

## QA baselines

- `VDE-QA-001` / `id=900001` - Nominal EPA baseline
- `VDE-QA-002` / `id=900002` - TWC boundary lower
- `VDE-QA-003` / `id=900003` - TWC boundary upper
- `VDE-QA-004` / `id=900004` - Higher mass baseline
- `VDE-QA-005` / `id=900005` - Lower mass baseline
- `VDE-QA-006` / `id=900006` - Missing optional fields
- `VDE-QA-007` / `id=900007` - Baseline requiring correction

## QA tires

- `TIRE-QA-001` / `id=920001` - Nominal complete
- `TIRE-QA-002` / `id=920002` - Low RRC
- `TIRE-QA-003` / `id=920003` - High RRC
- `TIRE-QA-004` / `id=920004` - RRC only
- `TIRE-QA-005` / `id=920005` - RRC + pressure reference
- `TIRE-QA-006` / `id=920006` - Split axle companion
- `TIRE-QA-007` / `id=920007` - Known test mass
- `TIRE-QA-008` / `id=920008` - Incomplete / Review case
- `TIRE-QA-009` / `id=920009` - Boundary / zero fixture
- `TIRE-QA-010` / `id=920010` - Golden reference

## Golden QA scenario

- Baseline: `VDE-QA-001`
- Tire: `TIRE-QA-010`
- Transmission: `TRANS-MOCK-001`
- Brake: `BRAKE-MOCK-001`
- Axle & Hubs: `AXLE-MOCK-001`
- Parasitics: `PARA-MOCK-001`

## Current limitation

The project does not currently use a separate SQLite component database for v2.2 lookups. Component QA fixtures remain backed by the existing deterministic CSV mock repositories.
