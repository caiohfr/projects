# Sprint 11I-B — Powertrain visual composition and result drivers

## Scope

This pass restructures the canonical Powertrain System Scenario presentation.
It adds no physics, schema, persistence, Fuel Estimate behavior, Vehicle
Demand behavior, or Technology Delta formula.

## Information hierarchy

Before, the page presented baseline fields, readiness JSON, a matrix, a long
editor, vertically repeated result panels, and a generic explainability table.

After, the canonical story is:

1. compact FuelCons / linked-VDE / readiness baseline cards;
2. compact Current + Proposal result cards near the top;
3. the multi-domain composition matrix;
4. a focused Domain Workspace;
5. demand → powertrain/PSE → final-result explanation;
6. collapsed technical details.

## Baseline and result cards

FuelCons remains the Current system baseline. Linked VDE appears in a distinct
Vehicle Demand card with the active TOTAL/NET basis. Raw canonical assumption
names are replaced on the primary surface by aggregate fuel-path efficiency,
electric-path efficiency, utility factor, and correction count. Raw IDs and
assumptions remain in a collapsed technical expander.

Current and up to three proposals share one reusable compact result-card
viewmodel. Fuel, electric energy, PSE, and CO2 come directly from
`SystemScenarioResult.effective_outputs`; PSE is formatted as percent and its
delta as percentage points.

## Domain Workspace

The selected proposal displays Configuration (what physically changed) beside
L0 Representation (what quantitative effect is adopted). Current corrections
remain collapsed and secondary. Configuration-only state explicitly says
`Not represented`, never a fabricated zero contribution. A second Calculate
action follows the editor.

## Result drivers

The presentation-only classifier uses the resolved Vehicle Demand comparison
and the presence of selected, adopted L0 representations. Its closed
vocabulary is:

- `DEMAND-DRIVEN`;
- `POWERTRAIN-DRIVEN`;
- `MIXED DEMAND + POWERTRAIN`;
- `NO QUANTITATIVE CHANGE`.

Each proposal explanation renders in canonical order: Vehicle Demand impact,
Powertrain/PSE impact, then final Fuel/Energy/CO2. Compact impact rows omit
inherited domains. Adopted and configuration-only changes remain visible.

## L0 composition and audit

When adopted impacts exist, the existing canonical scenario runner is invoked
for each deterministic composition step. This is labeled as sequential L0
scenario composition, not physical subsystem decomposition. Structured
issues, effective assumptions, and provenance live in collapsed Technical
details.

## Performance

FuelCons discovery remains lightweight. Detailed FuelCons materialization is
limited to the selected baseline, and detailed VDE materialization remains
limited to the active Current + three-proposal working set.

## Test evidence

Direct viewmodel tests cover result cards, PSE/fuel deltas, VDE comparison,
all four result-driver classifications, omission of inherited rows, inclusion
of adopted impacts, and configuration-only `Not represented` treatment.
AppTest covers the main workspace, calculation, edits/staleness, and the
post-editor Calculate action. Existing source-loading tests retain the large
discovery/no-N+1 guard.

Final evidence:

- 41 viewmodel/source-loading tests passed;
- 132 System Scenario contract/resolution/composition tests passed;
- all 14 Powertrain System Scenario AppTests passed in controlled batches;
- 146 affected Technology Delta, Vehicle Demand, Fuel/PSE and Powertrain
  Delta tests passed;
- the canonical full suite ran 1,831 tests in 536.208 seconds, retaining the
  repository's documented unrelated baseline of one failure and one error in
  `test_vde_request_resolver`;
- `compileall` and `git diff --check` passed.

## Browser smoke

The local Streamlit server started successfully and answered on
`127.0.0.1:8511`. The mandatory interactive browser pass could not start
because the integrated browser connection rejected the session before
navigation: its required sandbox-policy metadata was unavailable in this
environment. AppTest remains automated UI evidence and is not misreported as
manual visual smoke.

Closure status: **CODE-CLOSED / NOT VISUAL-SMOKE-CLOSED**.

## Deferred

Persistence, new physics, Comparison integration, PHEV semantic changes,
frontend migration, and Sprint 12 remain out of scope.
