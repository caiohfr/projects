# Sprint 11I-C — Powertrain visual polish and result story

## Scope

This pass closes presentation gaps left after Sprint 11I-B. It changes no
Vehicle Demand physics, Fuel Estimate physics, PSE formula, Technology Delta
math, System Scenario contract, schema, persistence, shared Proposal semantics,
or PHEV semantics.

## Frozen terminology

The canonical visible hierarchy is now:

1. persisted FuelCons row: **Source Baseline**;
2. Source Baseline plus optional Current Corrections: **Effective Current**;
3. alternative scenarios inheriting from Effective Current: **Proposal A/B/C**.

Current is not presented as a Proposal. When there are no corrections, the
page explicitly states `Source Baseline = Effective Current`.

## Density changes

Each scenario result card contains one horizontal four-metric surface in the
order Fuel, PSE, CO₂ and Electric. Missing calculated results use a compact
dash instead of four large `Not evaluated` values. Proposal deltas are always
relative to Effective Current, PSE deltas use percentage points, and all delta
colors are neutral.

Scenario and Domain selectors share one row. Domain Proposal selection and
creation share another row. A Proposal domain in `INHERIT` state remains a
compact summary until the user creates or selects a deviation; it no longer
exposes Current Correction controls.

## Result story

Calculated Proposal explanations keep the fixed causal reading order:

1. Vehicle Demand;
2. Powertrain / PSE;
3. Final result.

The existing presentation-only classifier supplies Demand-driven,
Powertrain-driven, Mixed demand + powertrain and No quantitative change.
Interpretation text states what changed without inventing additive attribution.

Configuration-only changes state that physical configuration changed while
the quantitative Energy Balance L0 effect is `NOT REPRESENTED`. Inherited
domains remain absent from the primary impact list. Adopted canonical impacts
and sequential L0 composition remain available, while raw assumptions,
provenance and calculation internals stay under collapsed Technical details.

## Calculation ownership and performance

Both Calculate actions call the same `calculate_drafts` path. Result values
continue to come from `SystemScenarioResult.effective_outputs`. FuelCons
discovery remains lightweight and detailed VDE/FuelCons materialization remains
bounded to the active working set.

## Validation

Focused tests cover terminology, applicability wording, compact missing values,
neutral Fuel/CO₂/PSE deltas, percentage-point formatting, all four driver
stories, configuration-only honesty, inherited-domain omission, adopted-impact
retention, selector state, Calculate parity and source-loading bounds.

The focused and affected suites passed:

- 23 Powertrain System Scenario UI/presentation tests;
- 41 viewmodel and source-loading tests;
- 132 System Scenario contract, resolution, composition and legacy tests;
- 146 Technology Delta, Vehicle Demand, Fuel/PSE and Powertrain Delta tests.

The full repository run collected and ran 1,841 tests in 805.140 seconds. It
finished with 1 failure and 1 error in the pre-existing VDE Request Resolver
component lookup fixtures. Both reproduce in isolation against the unchanged
Sprint 11I-B resolver paths: the active operational SQLite repository does not
contain `AXLE-MOCK-001` or `PARA-MOCK-001`.

Automated browser capture was unavailable because the local browser automation
runtime rejected session initialization. Manual visual smoke remains required
for the canonical result cards and the Demand-driven, Powertrain-driven,
configuration-only, and Configuration/L0 domain-workspace states. This package
is therefore code-closed but not visual-closed until that smoke is confirmed.

## Deferred

Physics, persistence, Comparison integration, shared Proposal redesign, PHEV
semantic redesign, frontend migration and Sprint 12 remain out of scope.
