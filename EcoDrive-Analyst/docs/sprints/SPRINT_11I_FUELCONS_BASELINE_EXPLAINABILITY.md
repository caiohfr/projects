# Sprint 11I — FuelCons baseline, VDE impact and result explainability

## Summary

The Powertrain System Scenario workspace now begins from one persisted
`fuelcons_db` row.  The selected row is retained as `fuelcons_id` in the
Current draft and inherited by newly added proposals.  Its linked `vde_id`
remains the Vehicle Demand dependency; it is not a competing baseline.

Discovery uses a deliberately lightweight FuelCons query (`id`, linked VDE,
make, model, year, architecture and fuel type).  The detailed FuelCons row is
loaded only after selection, while detailed VDE snapshots are loaded only for
the Current + at most three proposal working set.  Sprint 11I removes the
System Scenario primary-path heuristic that selected the latest FuelCons row
for a VDE.

## Canonical boundaries

No physical formula, Vehicle Demand calculation, schema, or persisted System
Scenario was added.  `run_fuel_estimation()` remains the owner of Fuel,
electric energy, CO2 and PSE.  The UI only formats its already-calculated PSE
fraction as a percentage and renders comparison/trace records from canonical
scenario calls.

The VDE impact-only section displays the linked persisted TOTAL and NET demand
values.  It is explicitly a demand-side view, not a second solver.

## Explainability

Proposal rows are classified from their actual scenario state:

- **Quantitative impact adopted** — an adopted direct L0 assumption or
  Technology Delta is present;
- **Configuration only** — configuration differs but no quantitative L0
  representation is adopted;
- **Current correction only** — Current contains an explicit source-scoped
  correction;
- **Not represented** — neither a change nor quantitative representation is
  available.

Configuration-only rows never receive invented PSE or Fuel contributions.
For adopted impacts, the sequential trace re-runs the existing canonical
System Scenario resolver after each adopted domain in stable domain order. It
therefore reports scenario-composition states, not a fabricated physical
subsystem decomposition.

## Evidence

Direct tests:

- `PowertrainSystemScenarioSourceLoadingTests.test_large_discovery_list_materializes_only_four_active_sources`
  proves one selected FuelCons row plus only the active VDE working set is
  materialized.
- `PowertrainSystemScenarioSourceLoadingTests.test_fuelcons_discovery_keeps_only_lightweight_search_labels`
  proves FuelCons-first labels carry FuelCons and linked-VDE identities.
- `PowertrainSystemScenarioViewmodelTests.test_fuelcons_identity_is_retained_by_current_and_inherited_proposals`
  proves the chosen FuelCons identity anchors Current and new proposals.
- `PowertrainSystemScenarioViewmodelTests.test_explainability_distinguishes_adopted_configuration_and_correction`
  proves the honest status categories.
- `PowertrainSystemScenarioViewmodelTests.test_sequential_trace_uses_canonical_outputs_only_for_adopted_impacts`
  proves a multi-impact trace consumes canonical outputs and omits
  configuration-only steps.
- `PowertrainSystemScenarioAppTests.test_primary_workspace_is_canonical_without_legacy_renderers`
  and the remaining AppTest class provide automated page-flow coverage.

Indirect / inspection evidence:

- `SystemScenarioResult.effective_outputs` is the unchanged canonical result
  surface.  The UI reads `pse` and formats it as `%`; it does not calculate
  PSE.

Manual smoke:

Not performed. AppTest is automated UI evidence and not a browser/manual
smoke claim.

## Deferred

- Persistence of composed System Scenarios;
- a full physical domain-attribution model;
- grid-CO2 semantic changes and a PHEV generic Technology Delta path;
- Comparison-page changes and any Sprint 11H follow-on scope.
