# VDE Setup Guide

## What VDE Setup Is

`VDE Setup` is the physical workflow used to build, review, preview, and persist a roadload scenario.

It is not only a coefficient editor. It is the place where the project resolves:

- scenario origin
- vehicle parameters
- roadload basis
- component effects
- transmission bridge
- cycle preview
- staged save payload

## Current Workflow

The page is organized as:

1. `Scenario Setup`
2. `Vehicle Parameters`
3. `Roadload Build-up`
4. `Cycle & Preview`
5. `Results`
6. `Save / Edit`

## Scenario Setup

This section defines:

- vehicle metadata
- scenario origin
- baseline selection when applicable
- inherited snapshot context

The key question here is:

> Where did this scenario come from?

Typical origins:

- from baseline
- new manual / test scenario

This is different from the physical source of `ABC_TOTAL`.

## Vehicle Parameters

This section resolves vehicle-level physical context, including:

- curb mass
- test mass
- inertia / TWC context
- front weight distribution
- resolved calculation mass
- aerodynamic reference / change handling

Mass setup is shared across other calculations so tires, preview, and transmission can reuse a consistent vehicle state.

## Roadload Build-up

This is the technical workspace of the page.

### Roadload Basis

The first decision is the physical basis for `ABC_TOTAL`.

Typical basis choices:

- inherit baseline `ABC_TOTAL`
- use measured / test coastdown `ABC_TOTAL`
- build / synthesize `ABC_TOTAL` from components

This choice changes how downstream component inputs should be interpreted.

### Components

The component workspace handles the main technical contributors:

- Tires
- Aerodynamics
- Brakes
- Parasitics / hubs / axle
- Trailer placeholder
- Transmission, handled separately but still part of the roadload bridge

### Current / Change / Applied

The component pattern should be read as:

- `Current`
  - the inherited or current reference state
- `Change`
  - the explicit engineering change, candidate, delta, or replacement being staged
- `Applied`
  - the resolved effect actually going into the scenario

This is important because not every candidate entry should immediately be interpreted as a physical applied value.

## Tires

The tire workflow is broader than a simple delta field.

The current page supports the following ideas:

- `Current Tire`
- tire change
- manual RR delta
- `Tire Improvement %`
- current vs walked tire comparison
- quick add tire
- Tire DB integration
- engineering-oriented entry paths such as `Custom`, `ISO`, and `SAE` where applicable

The tire area may combine:

- DB-backed reference selection
- scenario-only manual reference
- direct delta behavior
- target / walked comparison behavior

The point is to preserve traceability between:

- what the reference tire is
- what the changed tire is
- what effect is actually applied to the scenario

## VDE_TOTAL vs VDE_NET

This distinction is central.

### VDE_TOTAL

`VDE_TOTAL` is derived from `ABC_TOTAL`.

It represents the full resolved demand on the roadload basis.

### VDE_NET

`VDE_NET` exists only when transmission losses / neutral drag are resolved.

It is based on:

```text
ABC_NET = ABC_TOTAL - ABC_TRANS
```

So the logic is:

- transmission does not invalidate `VDE_TOTAL`
- missing transmission only prevents a valid `VDE_NET`

## Transmission As TOTAL -> NET Bridge

Transmission is intentionally separate from the ordinary component sections because it plays a special role:

- it does not just modify `ABC_TOTAL`
- it creates the bridge from `TOTAL -> NET`

This is why the workflow treats it as component-like, but still distinct.

## Cycle & Preview

This section is for immediate technical feedback.

It is used to:

- preview cycle-resolved demand behavior
- inspect phase outputs when available
- check whether the current scenario state is physically coherent before persistence

Important rule:

- preview does not save

## Results

`Results` should be read as a pre-save review layer, not a dashboard-only surface.

Typical content includes:

- performance summary
- warnings
- working scenario summary
- reference vs working comparison
- staged save payload
- technical details in expanders

Important rule:

- `Results` explains the preview and payload already resolved by the workflow
- it should not create a second independent physical model

## Save / Edit

This is the administrative persistence space of the page.

It contains:

- save as new
- update existing
- delete / deactivate behavior where applicable
- legacy maintenance that still exists for compatibility

The intent is to keep persistence explicit and separated from preview.

## Known Limitations

- some component sections are still hybrid between modern workflow structure and legacy persistence detail
- component provenance is not yet fully normalized for every future component family
- final DB smoke validation is still important before treating every path as hardened

## Related Docs

- [Sprint 5 Closure](SPRINT_5_CLOSURE.md)
- [Powertrain Scenario Guide](POWERTRAIN_SCENARIO_GUIDE.md)
- [ML / SHAP / Nearest Peers](ML_SHAP_NEAREST_PEERS.md)
