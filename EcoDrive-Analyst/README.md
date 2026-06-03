# EcoDrive Analyzer

EcoDrive Analyzer is a Streamlit application for Vehicle Demanded Energy (VDE) analysis, road-load scenario studies, and fuel or energy correlation workflows.

The project combines:
- physics-based VDE calculation from A/B/C coastdown coefficients and cycle traces;
- a modular roadload pipeline for baseline plus delta scenarios;
- SQLite persistence for VDE snapshots and fuel or energy scenarios;
- regression and comparison tooling for EPA-oriented analysis.

## Current Scope

Main user flows:
- `VDE Setup`: create, preview, edit, and save VDE snapshots;
- `PWT Fuel Energy`: attach fuel or energy scenarios to saved VDE snapshots;
- roadload scenario preview through `RoadLoadRequest -> run_roadload_scenario() -> EquivalentABC`.

Current focus of the codebase:
- modularize UI from core logic;
- keep `vde_core` free from Streamlit;
- reduce page-level SQL and page-level calculation glue;
- prepare the project for future physical component modeling.

## Architecture

The application is organized around a simple boundary:

```text
pages/ -> src/vde_app -> src/vde_core -> SQLite
```

- `pages/` contains Streamlit page orchestration.
- `src/vde_app/` contains reusable UI helpers, page components, and plotting helpers.
- `src/vde_core/` contains calculation, persistence, repository, regression, and service helpers.
- `src/vde_core/roadload/` contains the modular roadload domain pipeline.

The canonical roadload flow is:

```text
RoadLoadRequest
  -> run_roadload_scenario()
  -> EquivalentABC
  -> VDE calculation / preview / save
```

See:
- [Project Structure](docs/architecture/project_structure.md)
- [Roadload Pipeline](docs/architecture/roadload_pipeline.md)
- [UI and Backend Boundary](docs/architecture/ui_backend_boundary.md)

## Project Structure

```text
EcoDrive-Analyst/
|-- app.py
|-- data/
|   |-- db/
|   `-- standards/
|-- docs/
|   |-- architecture/
|   |-- notebooks/
|   |-- sprints/
|   `-- archive/
|-- notebooks/
|   |-- roadload/
|   `-- README.md
|-- pages/
|   |-- Comparison_Report.py
|   |-- PWT_Fuel_Energy.py
|   `-- VDE_Setup.py
|-- src/
|   |-- vde_app/
|   |   |-- __init__.py
|   |   |-- components/
|   |   |   |-- __init__.py
|   |   |   |-- pwt_fuel_energy.py
|   |   |   |-- shared.py
|   |   |   `-- vde_setup.py
|   |   |-- derivatives.py
|   |   |-- plots.py
|   |   |-- state.py
|   |   `-- ...
|   `-- vde_core/
|       |-- cycles.py
|       |-- db.py
|       |-- experimental/
|       |   |-- __init__.py
|       |   |-- tech_effects.py
|       |   `-- vehicle_csv_repo.py
|       |-- pwt_fuel_energy_service.py
|       |-- regression.py
|       |-- repositories/
|       |   |-- __init__.py
|       |   |-- fuelcons_repository.py
|       |   `-- vde_repository.py
|       |-- services.py
|       |-- utils.py
|       |-- vde_setup_service.py
|       `-- roadload/
|           |-- __init__.py
|           |-- adapters.py
|           |-- app_service.py
|           |-- decomposition.py
|           |-- engine.py
|           |-- models.py
|           |-- physics.py
|           |-- physics_legacy.py
|           `-- services.py
`-- tests/
    |-- test_core_services.py
    |-- test_pwt_and_decomposition.py
    |-- test_roadload_engine.py
    `-- test_vde_setup_service.py
```

## Installation

```bash
git clone https://github.com/caiohfr/projects.git
cd projects/EcoDrive-Analyst
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes About the Environment

At the moment, the repository contains tests under `tests/`, but the local virtual environment may need to be recreated if it still points to an old Windows Store Python path.

If that happens, recreate it:

```bash
rmdir /s /q .venv
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Testing

The test suite currently covers:
- roadload engine baseline and delta behavior;
- VDE setup service helpers;
- phase aggregation and mass/inertia helpers;
- PWT fuel-energy service helpers;
- roadload decomposition helpers.

Expected command:

```bash
python -m unittest discover -s tests -v
```

## Data and Notebooks

- working notebooks now live under `notebooks/`;
- notebook-specific narratives and ETL notes live under `docs/notebooks/`;
- archived backup pages live under `docs/archive/pages/`.

## Status

This repository is in an active modularization phase. The current sprint emphasizes hygiene, UI/core separation, roadload consolidation, and test coverage before deeper physical component modeling.
