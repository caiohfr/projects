# Sprint 11D Hotfix — Working-set source loading

## Summary

This hotfix removes the System Scenario page's eager source materialization.
It preserves the full lightweight VDE selector, but limits detailed VDE and
FuelCons retrieval to the Current scenario plus the active Proposal selections.
No physics, System Scenario contract, persistence, Comparison, Technology Delta,
PSE/PHEV behavior, or database schema changed.

## Reproduction and root cause

The local real database contained 5,003 VDE snapshots and approximately 5,004
FuelCons rows. Before this change, `_load_sources()` used `load_baselines_df()`
for discovery and then, for every candidate, called `fetch_vde_row()` and
`fetch_fuelcons_by_vde()`, constructing a `ScenarioSource` each time. One
Streamlit rerun therefore made one listing query plus roughly 5,003 individual
VDE and 5,003 FuelCons retrievals. This was an N+1 scalability defect.

The technical footer was a second amplifier: Streamlit executes code inside a
collapsed expander, so `render_powertrain_technical_footer()` prepared its
regression/ML/readiness diagnostics on the default page path.

## Discovery/materialization boundary

`load_baselines_df()` remains the discovery owner and supplies all selector IDs
and labels. `_working_set_vde_ids()` builds the materialization set from the
active Current anchor and proposal drafts only, deduplicates it, and caps normal
operation at four selections. `_load_sources()` calls the existing
`fetch_vde_rows_by_ids()` bulk owner exactly once for that set, then retrieves
FuelCons and constructs `ScenarioSource` once per unique active ID.

For example, Current=1, Proposal A=2, Proposal B=2, Proposal C=1 produces the
detailed working set `{1, 2}`, not four materializations and not one per
selector candidate.

## Technical diagnostics gate

The page now presents `Load technical diagnostics` inside the existing
`Technical audit and diagnostics` expander. The heavy footer executes only
after that explicit opt-in; a collapsed expander alone no longer executes it.

## Verification

- `PowertrainSystemScenarioSourceLoadingTests.test_working_set_deduplicates_current_and_proposal_source_ids` directly verifies Current=1/A=2/B=2/C=1 resolves to `(1, 2)`.
- `PowertrainSystemScenarioSourceLoadingTests.test_large_discovery_list_materializes_only_four_active_sources` uses 5,000 selector rows and four active IDs, verifying one bulk detail request, four FuelCons lookups, and four `ScenarioSource` constructions.
- `PowertrainSystemScenarioAppTests.test_legacy_source_and_technical_diagnostics_are_reachable_only_by_opt_in` verifies the default page has no `Metadata audit` footer output, then enables diagnostics and verifies it appears.
- Focused direct tests passed: source-loading tests (2) and the named AppTest (1). The broader 181-test System Scenario/service selection was run, along with `python -m compileall pages src tests`.
- Full regression: `python -m unittest discover tests` ran 1,805 tests in 635.273 seconds and retained the historical baseline of one failure and one error: `VdeRequestResolverTests.test_axle_hubs_lookup_snapshot_preserves_boundary_metadata` and `VdeRequestResolverTests.test_component_lookup_provenance_does_not_change_parasitic_math`.

Real-database measurement after the change: the 5,003-label discovery plus a
single active detailed source completed in 1.309 seconds in an isolated local
process. This measures the corrected code path; it is not a human browser
acceptance check.

## Manual recheck status

The real Streamlit app was started for a manual check, but this environment's
browser integration failed before navigation. Therefore browser/manual
acceptance (open time, Add Proposal, source selection, domain selection, and
default-unloaded diagnostics) remains **not performed**, rather than inferred
from AppTest.
