# UI and Backend Boundary

## Rule

The intended dependency direction is:

```text
pages -> src/vde_app -> src/vde_core
```

Allowed:

```text
vde_app -> vde_core
pages -> vde_app
pages -> vde_core
```

Forbidden:

```text
vde_core -> streamlit
vde_core -> vde_app
```

## What Belongs in `pages/`

- page config;
- section ordering;
- user-triggered actions;
- final rendering;
- success and warning messages.

Pages should behave as orchestration layers, not as calculation modules.

## What Belongs in `src/vde_app/`

- reusable UI components;
- page-specific UI modules;
- chart and display helpers;
- session-state defaults and UI state helpers.

This layer may import Streamlit.

## What Belongs in `src/vde_core/`

- math and physics helpers;
- database helpers;
- repository wrappers used by services;
- request-building and payload-shaping services;
- roadload orchestration;
- regression and data preparation helpers.

This layer should be pure from the UI point of view.

## Current Practical Examples

### VDE Setup

Page flow:

```text
pages/VDE_Setup.py
    -> components/vde_setup.py
    -> vde_setup_service.py
    -> repositories/vde_repository.py
    -> roadload + core services
```

### PWT Fuel Energy

Page flow:

```text
pages/PWT_Fuel_Energy.py
    -> components/pwt_fuel_energy.py
    -> pwt_fuel_energy_service.py
    -> repositories/fuelcons_repository.py
    -> regression / db helpers
```

### Database Management

Page flow:

```text
pages/Database_Management.py
    -> components/database_management.py
    -> database_management_service.py
    -> database_management_impact_service.py
    -> database_management_spreadsheet.py
    -> SQLite catalog tables + append-only data_change_log
```

The direct Tire Database editor is historical code under
`docs/archive/pages/Tire_Database_legacy.py`; it is not an active navigation
path. Tire lookup in VDE Setup remains a read-only consumer of the shared
catalog.

VDE Setup tire flow:

```text
components/vde_setup.py
    -> select/search tire_test_code
    -> tire_roadload_service.py
    -> repositories/tire_roadload_repository.py
    -> roadload/tire_model.py
    -> repositories/vde_tire_repository.py, only after explicit save
```

### Comparison Report

Page flow (current since the Sprint 8 Comparison Report closure -- see
`docs/sprints/PACKAGE_8E_COMPARISON_FREEZE.md` and
`docs/sprints/PACKAGE_8F_PROGRAM_REVIEW_REDESIGN.md`):

```text
pages/Comparison_Report.py
    -> components/comparison_report.py       (Streamlit UI: Program Review /
                                               Energy Drivers / Technical
                                               Scorecard / Explore)
    -> comparison_report_charts.py           (pure Plotly figure builders)
    -> comparison_report_viewmodels.py       (vde_app, pure, no Streamlit import)
    -> comparison_metric_registry.py         (single KPI/dimension source)
    -> comparison_report_service.py          (vde_core, canonical, Streamlit-free)
    -> fuelcons_db / vde_db (SQLite)
```

## Why This Boundary Matters

This separation makes it easier to:
- change UI without touching calculation logic;
- write tests against pure helpers;
- keep roadload and VDE logic reusable for future APIs or new interfaces;
- focus future work on physical modeling instead of page cleanup.
