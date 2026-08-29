# Sprint 11F — Canonical surface cleanup and legacy consolidation

## Summary

Sprint 11F separates the supported product surfaces from retained historical
and compatibility tooling. It moves UI entry points only: core services,
physics, contracts, persistence, and schemas retain their existing owners.
No Sprint 12 work is included.

## UX principles established

`docs/architecture/UX_PRINCIPLES.md` records the permanent principles of a
canonical-first surface, one dominant mental model, focused canonical audit,
explicit computation gates, separate legacy destinations, working-set loading,
and the distinction between AppTest and browser smoke. `AGENTS.md` adds the
concise permanent enforcement rules; the detailed rationale remains in the
architecture document.

## Surface inventory and classification

| Page / block | Classification | Evidence and disposition |
| --- | --- | --- |
| Powertrain title, source anchor, System Scenario matrix, domain editor, Calculate, results | CANONICAL | `pages/Powertrain_Scenario.py` delegates only to `render_system_scenario_workspace()`. |
| System Scenario fidelity, readiness, provenance and `Technical trace` | CANONICAL_AUDIT | `pwt_system_scenario._render_result()` explains the calculated `SystemScenarioResult`. |
| Scenario Pairing / metadata / baseline selection | LEGACY | `render_active_vde_source_bar()` moved to Powertrain Legacy. |
| Baseline method, ML/regression evidence and Technology Proposal UI | LEGACY | `render_powertrain_conversion_workspace()` and `render_technology_proposal_workspace()` moved to Powertrain Legacy. |
| Saved Estimates and historical Powertrain technical footer | LEGACY | Retained under Estimate Management and Powertrain Legacy, respectively. |
| Comparison selection, Program Review, Energy Drivers, Technical Scorecard, Explore, Quick Scenarios | CANONICAL | `comparison_report.render_comparison_report()` remains the current Comparison owner. |
| Historical Scenario Compare, Method Analysis, Peers & Outlook, Saved Estimates | LEGACY | Former `pwt_fuel_energy.render_comparison_report_page()` surface moved to Comparison Legacy. |
| Comparison canonical audit/lineage | CANONICAL_AUDIT | Existing scorecard lineage and current-result views remain with Comparison. |

No immediately related visible block remained `UNKNOWN`: the legacy bridge's
own module header and renderer ownership establish that it is the older
Powertrain-oriented report, while the Sprint 10E Quick Scenario surface is an
active canonical Comparison capability.

## Before and after

Previously, Powertrain appended source pairing, legacy evidence, saved
estimates, and a historical technical footer below the System Scenario
workspace. Comparison appended a `Powertrain Scenario Tools` bridge beneath
its canonical report; because it used a collapsed expander and internal tabs,
its historical renderers could still execute on the canonical path.

The canonical Powertrain page now ends with the System Scenario workspace and
its result-owned audit. The canonical Comparison page now ends with its
canonical report tabs. Neither invokes a retained legacy renderer.

## Legacy destination

`pages/Legacy_Engineering_Tools.py` provides **Legacy & Engineering Tools**,
also linked under **Engineering / Support** in `app.py`. It is visibly labeled
as a legacy/compatibility workspace and routes explicitly to one area:

- **Powertrain Legacy**: scenario pairing, baseline/method work, Technology
  Proposal, plus an opt-in legacy technical footer.
- **Comparison Legacy**: historical Scenario Compare, Method Analysis, Peers
  & Outlook, or Saved Estimates.
- **Estimate / Snapshot Management**: historical FuelCons management.

The destination starts unloaded. Its top-level radio renders only the selected
area. The historical Comparison renderer was changed from tabs to an explicit
radio dispatcher (`render_legacy_comparison_workspace`), so selecting one
legacy comparison workflow does not execute all four.

## Service ownership and performance

The UI relocation imports existing renderers; it does not copy PSE, Technology
Delta, fuel estimation, Vehicle Demand, or Comparison calculation code. The
canonical Powertrain page continues using `resolve_active_vde_source()` and
`render_system_scenario_workspace()`, preserving the Sprint 11D working-set
source loading boundary. Its legacy renderers are no longer called. Likewise,
canonical Comparison no longer imports or calls the historical Powertrain
Comparison bridge.

Heavy legacy Powertrain and Comparison preparation is behind Legacy area
selection; historical Powertrain diagnostics have a second explicit checkbox.
This avoids relying on `st.expander()` or `st.tabs()` as false laziness
boundaries.

## Automated evidence

- `PowertrainSystemScenarioAppTests.test_primary_workspace_is_canonical_without_legacy_renderers` verifies that canonical Powertrain exposes no legacy controls.
- `PowertrainSystemScenarioAppTests.test_canonical_result_keeps_technical_trace_without_legacy_footer` verifies canonical result audit remains available without the old footer.
- `ComparisonReportPageSmokeTests.test_page_opens_with_scenarios_available_no_selection` verifies canonical Comparison no longer exposes the old bridge.
- `LegacyEngineeringToolsAppTests.test_legacy_destination_starts_unloaded_and_labeled` verifies the new destination warning and unloaded default.
- `LegacyEngineeringToolsAppTests.test_powertrain_legacy_is_reachable_only_after_selection` verifies retained Powertrain access.
- `LegacyEngineeringToolsAppTests.test_comparison_legacy_does_not_render_powertrain_legacy` verifies routing isolation between legacy areas.
- `LegacyRenderingDispatchTests.test_legacy_area_dispatches_only_the_selected_renderer` and `LegacyRenderingDispatchTests.test_legacy_comparison_dispatches_only_the_selected_subworkflow` provide direct call-count evidence that unselected legacy renderers do not execute.
- Sprint 11D source-loading tests remain the direct coverage for O(active-working-set) Powertrain materialization.
- Focused Powertrain/Legacy AppTests and dispatch tests: 15 passed. Focused Comparison, Quick
  Scenario, and source-loading tests: 79 passed. `python -m compileall app.py
  pages src tests` and `git diff --check` passed.
- Full regression: `python -m unittest discover tests` ran 1,808 tests in
  497.400 seconds. It reproduced the known baseline failure
  `VdeRequestResolverTests.test_axle_hubs_lookup_snapshot_preserves_boundary_metadata`
  and known baseline error
  `VdeRequestResolverTests.test_component_lookup_provenance_does_not_change_parasitic_math`;
  both are Vehicle Demand/component lookup failures outside this UI-only sprint.

## Manual browser smoke

Manual browser smoke remains **not performed** in this environment. The
available browser integration failed before local navigation in two attempts
(including this sprint), so AppTest evidence is not presented as manual validation.
Required future smoke: canonical Powertrain, canonical Comparison, then all
Legacy areas with responsiveness and non-execution of unselected areas checked.

## Deferred observations and lesson

No redesign of the System Scenario matrix or Comparison scorecard was made.
Future visual evaluation should assess those canonical workflows independently,
now that the appended historical surfaces are gone. The key SDD/UX lesson is
that visible concealment is not execution isolation: ownership boundaries and
explicit selection must exist in the render call graph.
