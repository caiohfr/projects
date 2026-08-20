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

Page flow:

```text
pages/Comparison_Report.py
    -> comparison_report_service.py
    -> cycles / phase aggregation helpers
```

## Why This Boundary Matters

This separation makes it easier to:
- change UI without touching calculation logic;
- write tests against pure helpers;
- keep roadload and VDE logic reusable for future APIs or new interfaces;
- focus future work on physical modeling instead of page cleanup.
