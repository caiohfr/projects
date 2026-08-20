# Project Structure

## Goal

This document records the intended structure after the hygiene and modular architecture sprint.

The project is split into three practical layers:

```text
pages/ -> src/vde_app -> src/vde_core
```

## Layers

### `pages/`

Streamlit entry points.

Responsibilities:
- page layout;
- user interaction flow;
- calling app components and services;
- rendering outputs.

Should avoid:
- direct physics logic;
- large SQL blocks;
- duplicated helper functions.

### `src/vde_app/`

Reusable UI-facing modules.

Responsibilities:
- shared components;
- page-specific component modules;
- chart helpers;
- session-state defaults and helpers.

Current examples:
- `components/shared.py`
- `components/vde_setup.py`
- `components/pwt_fuel_energy.py`
- `components/database_management.py`
- `plots.py`
- `state.py`

Recent practical result:
- `pages/VDE_Setup.py` now acts mostly as entry-point orchestration;
- reusable VDE sections live in `components/vde_setup.py`;
- shared page state defaults and reset helpers live in `state.py`.

### `src/vde_core/`

Technical backend modules.

Responsibilities:
- VDE and cycle math;
- database helpers;
- repository wrappers for domain-facing queries and writes;
- regression helpers;
- page service helpers;
- roadload orchestration.

Should avoid:
- `streamlit` imports;
- UI rendering concerns.

Support modules now include:
- `cycles.py` for cycle-facing entry points;
- `repositories/` for thin VDE and fuel-consumption persistence wrappers;
- `database_management_service.py`, `database_management_impact_service.py`, and
  `database_management_spreadsheet.py` for controlled catalog changes,
  dependency review, and spreadsheet staging;
- `comparison_report_service.py` for report-facing data loading outside the page;
- `tire_roadload_service.py` for tire-roadload CRUD, preview, and VDE application flow;
- `experimental/` for preserved non-canonical heuristics and CSV experiments;
- `roadload/physics_legacy.py` for older physical experiments kept outside the active pipeline.

## Roadload Package

The roadload package is the domain core for equivalent A/B/C synthesis:

```text
src/vde_core/roadload/
    __init__.py
    models.py
    engine.py
    adapters.py
    app_service.py
    decomposition.py
    physics.py
    physics_legacy.py
    services.py
```

Canonical path:

```text
RoadLoadRequest -> run_roadload_scenario() -> EquivalentABC
```

Notes:
- `engine.py` is pure calculation;
- `adapters.py` translates external inputs;
- `app_service.py` bridges app context to roadload request objects;
- `decomposition.py` provides reporting-friendly helpers;
- `physics.py` is a compatibility entry point for preserved legacy helpers;
- `physics_legacy.py` holds the preserved experimental implementation;
- `services.py` is compatibility-oriented.

## Tests

Tests live in `tests/` and focus on pure logic first:
- roadload engine;
- VDE setup helpers;
- phase aggregation and test mass;
- PWT service helpers;
- decomposition helpers;
- tire roadload helpers and tire model behavior.

Database Management tests cover its contract, staged CRUD path, dependency
impact workflow, controlled spreadsheet imports, and page-level rendering.

## Archive and Notebook Policy

- active notebooks belong in `notebooks/`;
- notebook notes belong in `docs/notebooks/`;
- archived pages and backups belong in `docs/archive/`.
- the direct Tire Database editor is preserved at
  `docs/archive/pages/Tire_Database_legacy.py`; Database Management is the
  active catalog-administration page.
