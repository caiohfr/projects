# Roadload Pipeline

## Purpose

The roadload pipeline exists to keep A/B/C scenario handling out of Streamlit pages and concentrated in a reusable backend path.

## Canonical Flow

```text
UI input or DB baseline
    -> adapters / app_service
    -> RoadLoadRequest
    -> run_roadload_scenario()
    -> EquivalentABC
    -> VDE preview / compute / save
```

## Main Objects

### `BaselineInput`

Carries baseline A/B/C, mass, metadata, and source.

### `OperatingModifiers`

Carries operating deltas, mainly `delta_mass_kg` for the current scope.

### `ComponentChanges`

Carries scenario deltas by component family such as tire, aero, brakes, and parasitics.

Current practical interpretation:
- deltas are applied to the equivalent roadload total;
- detailed physical decomposition can evolve later without changing the page contract.

### `RoadLoadRequest`

Single request object used by the engine.

### `EquivalentABC`

Final equivalent roadload output used by VDE calculation.

Contains:
- equivalent `A`
- equivalent `B`
- equivalent `C`
- final `mass_kg`
- component table
- warnings

## Engine Responsibilities

`engine.py` is responsible for:
- normalizing request inputs;
- resolving a complete baseline;
- building the internal component set;
- applying supported deltas;
- synthesizing final equivalent A/B/C and mass.

It should not depend on:
- Streamlit;
- SQLite;
- page modules.

## App Bridge

`app_service.py` is the thin bridge from page-like context dictionaries to the canonical roadload request flow.

This lets page services call roadload without manually rebuilding the domain logic each time.

## Decomposition Helpers

`decomposition.py` exists for inspection and reporting:
- show final component table;
- compare equivalent results versus baseline;
- support future UI summaries without crowding the engine.

## Current Limits

Current scope is intentionally conservative:
- component replacement is not treated as a full physical decomposition workflow;
- most scenario handling is delta-oriented;
- future tire-to-tire or aero-to-aero comparisons can build on top of the same request structure.
