# EcoDrive Analyzer

EcoDrive Analyzer is a Streamlit application for roadload engineering, VDE workflow management, powertrain consumption estimation, and engineering comparison reporting.

The current product is organized around four main blocks:

1. `Database Management`
2. `VDE Setup`
3. `Powertrain Scenario`
4. `Comparison Report`

`Database Management` is the official controlled catalog-administration
workflow. `VDE Setup` is the stable, feature-frozen engineering workflow.
`Comparison Report` is a stable engineering comparison/reporting foundation
delivered in Sprint 8; `Powertrain Scenario` is an existing capability but
still less mature than the VDE/Comparison architecture, and remains a future
development area.

## Current Product Blocks

### Database Management

`Database Management` is the official operational surface for controlled
catalog records:

- VDE baselines and saved VDE records
- fuel-consumption records and their VDE references
- tire records
- transmission, brake, axle & hubs, and parasitic components
- staged create, update, archive, restore, and duplicate changes
- impact review, dependency resolution, and append-only change receipts
- controlled spreadsheet templates with review-before-commit imports

The former direct Tire Database page is preserved under `docs/archive/pages/`
as historical reference code. It is not part of the ordinary application
navigation.

### VDE Setup

`VDE Setup` is the physical and traceable workflow for:

- canonical baseline and effective-baseline corrections
- scenario proposals and Walk From lineage
- Mass, Tire, Aero, Transmission, Brake, Axle & Hubs, and Parasitics
- roadload TOTAL / NET and cycle analysis
- metadata, provenance, engineering comparison, and audit
- append-only Save and historical Reload
- deterministic synthetic QA data

Core idea:

- `VDE_TOTAL` is the demand derived from `ABC_TOTAL`
- `VDE_NET` is available only when transmission losses / neutral drag are resolved
- Historical EcoDrive VDE records stored `VDE_TOTAL` in `vde_net_mj_per_km` (Package 7G
  normalized affected rows). New code must read TOTAL/NET through
  `src/vde_core/vde_net_total_contract.py::canonical_vde_read()` and must never use
  `vde_net_mj_per_km` as a fallback for TOTAL — see
  `docs/sprints/PACKAGE_7G_VDE_NET_TOTAL_CONTRACT.md`.

### Powertrain Scenario

`Powertrain Scenario` consumes a resolved VDE source and estimates:

- fuel consumption
- electric energy
- CO2

Inside that page, the current interpretation flow is:

1. `Vehicle Demand`
2. `Powertrain System Efficiency (PSE)`
3. final fuel / electric result

Supported estimation methods in the current product:

- `Manual / Imported`
- `Physics Simple`
- `Regression`
- `ML Prediction`

Planned but not delivered as runtime engines:

- `Physics + ML Residual`
- `Map-Based Simulation`

Important boundary:

- `Powertrain Scenario` does not recalculate roadload
- it uses an already resolved energy basis such as `VDE_TOTAL` or `VDE_NET`

Practical reading:

- `VDE_TOTAL` or `VDE_NET` tells us how much energy the vehicle demands on the cycle
- `PSE` tells us how effectively the powertrain converts supply energy into that delivered demand
- fuel, electricity, and CO2 are the final outputs of that relationship

### Comparison Report

Stable engineering comparison/reporting foundation delivered in Sprint 8.

Current product structure:

- `Program Review`
- `Energy Drivers`
- `Technical Scorecard`
- `Explore`

Core capabilities now include, at a high level:

- optional Reference
- multi-scenario Comparison Set
- Proposal / Benchmark presentation roles
- Current designation
- Primary KPI and optional Target
- KPI Walk / absolute comparison presentation
- Demand vs Efficiency
- equi-PSE guidance
- Energy & Demand Summary
- physical setup / roadload / ABC evidence
- real EPA/WLTP phase VDE
- demanded power analysis
- Technical Scorecard
- custom Explore analysis
- explicit physical VDE lineage
- strict TOTAL / NET semantics
- provenance / stale-source visibility
- Vehicle Demand Summary in Energy Drivers (Sprint 9)

This is an accepted Sprint 8 product foundation for program / benchmark
review workflows, not a finished final reporting product -- broader BI/
benchmark-studio capabilities remain future work.

### Vehicle Demand

Canonical, reusable wheel-side demand layer delivered in Sprint 9 (CLOSED /
FROZEN). Answers "what does the vehicle require at the wheels, and why is
it different between scenarios?" from the project's existing authoritative
VDE/roadload physics -- it does not implement a second/new VDE model, and
it does not model how that demand is supplied (Powertrain).

```text
Resolved VDE / ComparisonItem
        -> VehicleDemandRequest
        -> Vehicle Demand Core (physics engine, frozen)
        -> VehicleDemandResult
        -> Comparison presentation layer (Energy Drivers -- Vehicle Demand Summary)
```

Core capabilities:

- authoritative Roadload Energy, VDE, Positive Tractive Energy, Braking
  Energy Required (wheel-side, not recovered regen)
- Known Rolling / Known Aero energy, with Residual / Unattributed Roadload
  as an explicit `Known + Residual = Authoritative` identity -- never a
  forced decomposition
- Positive Inertial Work
- strict TOTAL / NET semantics with no fallback (matches Comparison's own
  rule)
- typed, Streamlit-independent, JSON-serializable contracts (architecture
  readiness for a future API/agent boundary -- none is implemented)
- currently surfaced through Comparison Report's Energy Drivers tab; a
  future Quick Scenario (Sprint 10) is expected to reuse the same frozen
  core with temporary overrides, not new physics

See [Vehicle Demand Architecture](docs/architecture/vehicle_demand_architecture.md)
and [Sprint 9 Closure](docs/sprints/SPRINT_9_VEHICLE_DEMAND_CLOSURE.md) for
full detail, physical invariants, and known limitations.

## Stable Product Status

```text
EcoDrive
|
|-- Database Management   stable / official catalog administration
|-- VDE Setup             stable / feature frozen
|-- Comparison Report     stable / accepted Sprint 8 product foundation
|-- Vehicle Demand        stable / CLOSED-FROZEN Sprint 9 canonical layer
`-- Powertrain Scenario   existing capability / future development area
```

## Roadmap

```text
Sprint 7   Database Management                          CLOSED
Sprint 8   Comparison Report Foundation                  CLOSED
Sprint 9   Vehicle Demand Model & Engineering KPIs        CLOSED
Sprint 10  Interactive Quick Scenario
Sprint 11  Powertrain Scenario L0
Sprint 12  PWT + Comparison Integration
           MVP PRODUCT GATE
```

Notes on near-term direction (see
[Sprint 9 Closure](docs/sprints/SPRINT_9_VEHICLE_DEMAND_CLOSURE.md) for the
full handoff):

- **Sprint 10 (Quick Scenario)** applies temporary overrides (Mass, CdA,
  RRC first) to an existing resolved scenario and produces a
  `VehicleDemandRequest` consumed by the same frozen Vehicle Demand Core --
  it must not introduce new physics. Temperature/Pressure ambient overrides
  can enter through the same architecture once an owner for authoritative-
  roadload condition correction is defined; the former standalone "Roadload
  Condition Scenarios" concept is expected to be absorbed here rather than
  remain an independent capability.
- **Database Import** remains an important operational capability but is
  not treated as a blocker for the Product MVP gate above.

This is the current top-level state, not a full redesign of the post-MVP
roadmap.

## Documentation Index

Sprint documentation:

- [Sprint 5 Closure](docs/SPRINT_5_CLOSURE.md)
- [Sprint 6 Plan](docs/sprints/SPRINT_6_VALIDATION_SCENARIO_BENCH_RELEASE_2026-06-28.md)
- [Sprint 7 Database Management Checkpoint](docs/sprints/SPRINT_7_DATABASE_MANAGEMENT.md)
- [Sprint 8 Comparison Report Freeze (Package 8E)](docs/sprints/PACKAGE_8E_COMPARISON_FREEZE.md)
- [Sprint 8F Program Review Redesign](docs/sprints/PACKAGE_8F_PROGRAM_REVIEW_REDESIGN.md)
- [Sprint 9 Closure - Vehicle Demand Model & Engineering KPIs](docs/sprints/SPRINT_9_VEHICLE_DEMAND_CLOSURE.md)
  - [9A Canonical Vehicle Demand Contracts](docs/sprints/PACKAGE_9A_VEHICLE_DEMAND_CONTRACTS.md)
  - [9B Vehicle Demand Physics Engine](docs/sprints/PACKAGE_9B_VEHICLE_DEMAND_ENGINE.md)
  - [9C Real Scenario Validation & Hardening](docs/sprints/PACKAGE_9C_VEHICLE_DEMAND_HARDENING.md)
  - [9D Comparison / Energy Drivers Integration](docs/sprints/PACKAGE_9D_COMPARISON_ENERGY_DRIVERS_INTEGRATION.md)
- [VDE Setup Guide](docs/VDE_SETUP_GUIDE.md)
- [VDE Setup v2.2 Final Stable Contract](docs/VDE_SETUP_V22_FINAL_CHECKPOINT.md)
- [Powertrain Scenario Guide](docs/POWERTRAIN_SCENARIO_GUIDE.md)
- [ML / SHAP / Nearest Peers](docs/ML_SHAP_NEAREST_PEERS.md)

Architecture references:

- [Project Structure](docs/architecture/project_structure.md)
- [Roadload Pipeline](docs/architecture/roadload_pipeline.md)
- [UI and Backend Boundary](docs/architecture/ui_backend_boundary.md)
- [Vehicle Demand Architecture](docs/architecture/vehicle_demand_architecture.md)
- [Sprint 5 Architecture Checkpoint](docs/sprints/SPRINT_5_VDE_FUEL_ARCHITECTURE_2026-06-19.md)

Notebook notes:

- [Notebooks README](notebooks/README.md)

## Repository Structure

```text
EcoDrive-Analyst/
|-- app.py
|-- data/
|-- docs/
|-- models/
|-- notebooks/
|-- pages/
|-- src/
|-- tests/
|-- requirements.txt
`-- requirements-ml.txt
```

Important runtime pages:

- `pages/VDE_Setup.py`
- `pages/Powertrain_Scenario.py`
- `pages/Comparison_Report.py`
- `pages/Database_Management.py`

## How To Run

Create the local environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional ML dependencies:

```bash
pip install -r requirements-ml.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Testing And Validation

Fast syntax / import pass:

```bash
python -m compileall -q pages src tests
```

Run the test suite:

```bash
python -m unittest discover tests
```

If you want a narrower pass while iterating:

```bash
python -m unittest tests.test_vde_workflow_service tests.test_fuel_estimation
```

## ML Runtime Notes

`ML Prediction` is an inference capability, not a notebook execution mode.

Current expectations:

- the notebook remains an experimental/training source
- runtime inference expects an exported artifact under `models/`
- the current repository already includes a Powertrain Scenario artifact:
  - `models/powertrain_scenario_ml.joblib`
- optional ML dependencies live in `requirements-ml.txt`
- the current ML artifact predicts final fuel / energy outputs
- `PSE` shown in the UI is currently derived from those outputs plus the active demand basis
- direct ML prediction of cycle-effective `PSE` is future work unless a dedicated artifact is trained for that target

Possible ML runtime states:

- artifact found and loaded
- artifact missing
- artifact load failed
- missing features
- partial / out-of-domain coverage

See [ML / SHAP / Nearest Peers](docs/ML_SHAP_NEAREST_PEERS.md) for the detailed explanation.

## Known Limitations

- ML runtime depends on an exported artifact and compatible dependencies.
- SHAP availability depends on model form and explainability compatibility.
- Nearest Peers quality depends on dataset coverage and consistency.
- Regulatory / label benchmarking is still an early scaffold.
- Performance simulation is still planned.
- `Physics + ML Residual` and `Map-Based Simulation` are planned, not production engines.
- Comparison Report is a stable Sprint 8 engineering foundation, not a
  finished BI/benchmark-studio product; broader benchmark-authoring and
  peer-analytics capabilities remain future work.
- a legacy Gasoline LHV constant (34.2 MJ/L) still exists in
  `derivatives.py`/`plots.py` alongside the canonical 32.0 MJ/L value
  Comparison/PSE actually use (`fuel_energy.py::LHV_MJ_PER_L`); this is
  known technical debt, not yet harmonized.
- Hidden component priors are future backlog, not a delivered causal inference capability.
- current `PSE` is cycle-effective system efficiency, not pure engine efficiency.
- temperature and ambient-pressure roadload conditions are deferred to derived
  scenarios in Comparison Report; they are not persisted by VDE Setup.
- Vehicle Demand's Known decomposition covers Rolling + Aero only; Residual /
  Unattributed Roadload may contain brake, driveline, bearing, parasitic,
  and other unattributed effects, and is never presented as a specific
  named component.
- Vehicle Demand does not model powertrain efficiency, and Braking Energy
  Required is a wheel-side theoretical figure only -- no regen capture
  model exists yet.
- `AmbientState` supports Aero-density calculation (explicit density, or
  from temperature + pressure) but does not correct the authoritative
  roadload ABC itself; no regulatory-reference ambient default exists.
- `VehicleDemandProfile` is computed on demand and is not persisted.
- synthetic QA fixture `vde_total_mj_per_km`/`vde_net_mj_per_km` values are
  not guaranteed to be physically derived from that fixture's own ABC/mass/
  cycle and must not be treated as physical golden outputs for regression
  -- see [Vehicle Demand Architecture](docs/architecture/vehicle_demand_architecture.md#qa-persisted-vde-debt).
- `KinematicPhase`, VSP, and driving-aggressiveness classification are
  deferred; Vehicle Demand's `EnergyMode` (IDLE/TRACTION/COASTING/BRAKING)
  is the only classification implemented.

## Sprint 5 Status

Sprint 5 delivered the product foundation the project needed:

- `VDE Setup` as a disciplined physical workflow
- `Powertrain Scenario` as an estimation-first page
- `Comparison Report` as a separate reporting direction
- shared estimation contracts for manual, physics, regression, and ML paths
- first-class `PSE` interpretation across estimator, review, and comparison flows
- initial ML explainability and peer-guidance capabilities

See [Sprint 5 Closure](docs/SPRINT_5_CLOSURE.md) for the consolidated close-out.

## Sprint Closure

Current sprint closure:

- `VDE Setup` moved to spreadsheet-first technical input mode.
- `Powertrain Scenario` is consolidated as a guided workflow:
  `Scenario Pairing -> Baseline Estimate -> Technology Delta -> Result & Save`.
- Technical diagnostics were moved behind advanced / technical details by default.
- No schema, VDE formula, or ML training changes were introduced in this closure.

This closure predates Sprint 7, Sprint 8, and Sprint 9, all since delivered:
`Database Management` (Sprint 7), the `Comparison Report` engineering
foundation described above (Sprint 8), and the `Vehicle Demand` canonical
layer described above (Sprint 9) are no longer forward-looking items --
see the Documentation Index for their closure records.
