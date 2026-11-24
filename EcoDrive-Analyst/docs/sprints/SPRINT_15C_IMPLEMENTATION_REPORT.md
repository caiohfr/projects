# Sprint 15C implementation report

Date: 2026-11-23

Branch: `sprint-15c-comparison-powertrain`

Verified base: `origin/main@ab44532547d228907bd4bdba42a9a33aedb80121`

## Scope delivered

The change is limited to the Streamlit presentation layer for the canonical
Comparison Report and Powertrain System Scenario L0 workspaces.

### Comparison Report

- Added a compact workflow header using the approved EcoDrive navy, blue,
  green, gray and white tokens.
- Made the three-part hierarchy explicit: build the working set, inspect its
  factual context, then review the evidence.
- Preserved optional Reference selection, Comparison-only datasets, filters,
  availability controls and the five canonical tabs.
- Added a factual working-set table sourced only from the existing
  `ComparisonDataset`: role, scenario, cycle, legislation, available TOTAL/NET
  Vehicle Demand basis and provenance.
- Added explicit wording that Vehicle Demand is not FuelCons consumption and
  that unavailable values are not inferred.
- Kept Quick Scenarios explicitly identified as in-memory.

### Powertrain System Scenario L0

- Aligned the existing page styling with the shared tokens in
  `src/vde_app/visual_theme.py`; no parallel product palette was introduced.
- Added explicit page states for persisted FuelCons reference, in-memory
  proposals and absence of implicit save.
- Added a compact state strip for reference selection, Current + proposal
  cardinality and calculated-result availability.
- Made the selected FuelCons ID, linked physical VDE ID, stored energy basis,
  provenance and update value visible next to the selector. Missing metadata
  is displayed as `Unavailable`.
- Preserved Current + maximum three proposals, Add/Remove/Rename, the domain
  workspace, Calculate/Recalculate, result cards, drivers and explainability.

No file under `src/vde_core` was changed. No formula, schema, service contract,
widget key, callback, calculation path or persistence behavior was changed.

## Automated gates

| Gate | Baseline | After | Result |
|---|---:|---:|---|
| Comparison page smoke | 67 run; 65 pass; 2 fail | 67 run; 65 pass; 2 fail | Same two pre-existing fixture failures |
| Comparison service | 61 pass | 61 pass | PASS |
| Comparison related viewmodels/charts/smokes | 314 pass | 314 pass | PASS |
| Powertrain UI/viewmodels/source/deltas/fuel estimation | 96 pass | 96 pass | PASS |

Post-change total: 538 tests executed, 536 passed and 2 reproduced baseline
failures. The two failing tests are:

- `test_switch_to_qa_button_does_not_reseed_or_mutate_canonical_qa`
- `test_switch_to_qa_button_points_runtime_db_at_the_canonical_qa_instance`

Both fail on the verified base and after the patch because the canonical QA
fixture selected by those tests does not contain `fuelcons_db`
(`sqlite3.OperationalError: no such table: fuelcons_db`). No additional failure
was introduced. Two focused AppTest checks for the new Comparison and
Powertrain visual contracts also passed.

`python -m py_compile` passed for all four changed runtime modules, and
`git diff --check` passed.

## Numeric parity snapshot

Fixture: ICE, VDE TOTAL `1.8 MJ/km`, `eta_pt_est=0.30`, gasoline,
`LHV=32.0 MJ/L`; Proposal applies the existing canonical `+5% PSE` delta.

| Output | Current | Proposal |
|---|---:|---:|
| PSE | 0.300000000 | 0.315000000 |
| Fuel | 18.750000000 L/100 km | 17.857142857 L/100 km |
| CO2 | 433.125000000 g/km | 412.500000000 g/km |

The same canonical calculation and delta tests pass before and after the UI
change. Tolerance for parity assertions remains the existing test tolerance
(up to 9 decimal places in the canonical/UI parity tests).

## Gate status

- G0 — PASS: remote refreshed, exact base verified, isolated worktree used,
  personal database and dirty checkout left untouched.
- G1 — PASS (automated): five Comparison tabs, optional Reference, filters,
  physical VDE route, TOTAL/NET context, provenance and in-memory Quick
  Scenarios preserved.
- G2 — PASS (automated): persisted FuelCons selection, linked VDE, Current + 3,
  editors, calculate/recalculate and numeric parity preserved; no save added.
- G3 — PENDING owner browser review: neither available browser-control channel
  could start. Native control reported `Computer Use native pipe is unavailable
  (os error 2)` and integrated control reported `The browser access host is
  unavailable`. No screenshot or visual approval is claimed.

Sprint 15C must remain open for real-browser review at 1440x900 and 1024x768,
including every Comparison tab and the Powertrain Current, Current + 3,
editor and calculated-result states.
