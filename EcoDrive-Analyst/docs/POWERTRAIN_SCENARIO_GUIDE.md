# Powertrain Scenario Guide

## What The Page Is

`Powertrain Scenario` is the estimation block of EcoDrive.

It is not meant to be a complete dashboard or benchmark center. Its role is to take a resolved VDE source and estimate:

- fuel consumption
- electric energy
- CO2

The page consumes VDE. It does not recalculate roadload.

## Current Workflow

The page is organized as:

1. `Context & Energy`
2. `Powertrain Inputs`
3. `Estimation Engine`
4. `Results & Save`
5. `Saved Estimates`

Each tab has one main responsibility.

## Context & Energy

This tab defines the scenario context:

- active VDE source
- scenario name
- scenario intent
- electrification override
- energy basis
- source revision / compatibility context

### Energy Basis In Simple Language

Energy basis means:

> Which resolved energy value will be used as the starting point for estimating fuel, electric energy, or CO2?

Examples:

- `VDE_TOTAL`
  - use the total resolved VDE demand
- `VDE_NET`
  - use the net demand after transmission losses are separated
- cycle / phase values
  - use phase-specific energy when a method or workflow provides it

Important rule:

- `Powertrain Scenario` uses this basis for estimation
- it does not rebuild `ABC`, roadload, or transmission from scratch

## Powertrain Inputs

This tab stages powertrain assumptions and optional drivetrain metadata.

Typical inputs:

- `fuel_type`
- `LHV`
- `gCO2_per_L`
- `eta_pt_est`
- `bev_eff_drive`
- `grid_gco2_per_kwh`
- `utility_factor`
- `gear_count`
- `final_drive_ratio`
- transmission model metadata when applicable

These inputs support the estimation methods but do not modify the VDE snapshot itself.

## Estimation Engine

This tab selects and configures the method used to estimate the current draft.

### Manual / Imported

Use this when the estimate should come from:

- measured values
- imported values
- official external values

This path is useful when the user wants traceable persistence without recalculating from an internal model.

### Physics Simple

This is the explicit engineering path.

It combines:

- resolved VDE energy basis
- physical assumptions such as efficiency, LHV, and CO2 factors

Use it when a direct and interpretable estimation path is desired.

### Regression

Regression remains an empirical estimation method.

It should be used with visual review of the active dataset and scatter whenever available. It is useful for:

- quick estimation
- empirical benchmark
- scenario sanity checks

But it is not the center of the page. It is one estimation method inside the broader workflow.

### ML Prediction

`ML Prediction` is a runtime inference capability when an exported artifact is available.

It should still produce a result compatible with the common estimation contract used by the other methods.

### Planned Methods

Visible but not delivered as runtime production engines:

- `Physics + ML Residual`
- `Map-Based Simulation`

These are intentionally shown as planned capabilities rather than hidden future work.

## Results & Save

This tab is the review and persistence surface of the page.

Expected responsibilities:

- preview the estimate
- show assumptions
- show warnings
- show provenance
- expose the staged payload in a technical expander
- save or update using the common save flow

Important rules:

- preview does not save
- save should persist the already resolved result, not build a parallel one

## Saved Estimates

This tab is for browsing and managing saved estimation rows.

Its role includes:

- list saved scenarios
- open existing estimates
- update existing estimates
- delete or remove estimates where supported
- show source-changed / refresh-required state

This keeps saved-estimate management separate from the active estimation flow.

## ML Prediction Runtime Behavior

The runtime may report one of several states:

- artifact found and loaded
- artifact missing
- artifact load failed
- missing features
- partial / out-of-domain coverage

That behavior is expected and should be documented, not hidden.

See [ML / SHAP / Nearest Peers](ML_SHAP_NEAREST_PEERS.md) for the detailed model-related explanation.

## Regression Guidance

Regression is still valid, but it should be read carefully:

- use active filters intentionally
- inspect the scatter / visual feedback when available
- treat it as estimation and benchmark support, not as a proof of causality

## Comparison Report Relationship

`Comparison Report` is intentionally outside the estimator page.

That separation exists so:

- `Powertrain Scenario` stays estimation-first
- comparison, BI, and benchmark logic can evolve independently

## Known Limitations

- ML depends on exported artifacts and optional dependencies
- SHAP depends on model compatibility and explainability support
- nearest peers and guidance depend on dataset coverage
- richer reporting and regulatory layers are still evolving
- performance simulation remains planned, not delivered

## Related Docs

- [Sprint 5 Closure](SPRINT_5_CLOSURE.md)
- [VDE Setup Guide](VDE_SETUP_GUIDE.md)
- [ML / SHAP / Nearest Peers](ML_SHAP_NEAREST_PEERS.md)
