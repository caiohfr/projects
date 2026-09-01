# Sprint 11D Hotfix — UX Contract Alignment

## Status

**UX HOTFIX APPLIED / MANUAL RECHECK REQUIRED.** This is a post-freeze UI
hierarchy correction only. It does not reopen Sprint 11 contracts, physics,
solver, schema, persistence, Comparison, topology, or PHEV scope.

## Confirmed UX acceptance failure

A real manual browser observation supplied for this hotfix found the initial
Powertrain viewport led with `Scenario Pairing`, source-method selection,
demand technical details, editable metadata and ML-oriented fields. The
multi-domain matrix existed further down the page, but was not the product's
primary mental model. This is a **CONFIRMED UX ACCEPTANCE FAILURE**, not a
physics or core defect.

## Root cause and before hierarchy

`pages/Powertrain_Scenario.py` invoked
`render_active_vde_source_bar()` before `render_system_scenario_workspace()`.
That renderer intentionally contains the complete legacy Vehicle Demand plus
baseline-source setup, reference controls, metadata review and technical
details. Its position made a legacy estimate workflow the opening experience.

## After hierarchy

The page now resolves the selected VDE snapshot without rendering the legacy
workbench, then presents:

1. **Powertrain System Scenarios**;
2. Add/Remove Proposal and the real Current/Proposal composition matrix;
3. one scenario/domain editor;
4. Calculate System Scenarios and concise result cards;
5. secondary collapsed workbenches.

The matrix is still generated from `ScenarioDraft` state and shows all eight
domains plus each scenario's status. It is not a static UX placeholder.

## Legacy and diagnostics placement

The original source-pairing renderer is retained under collapsed **Advanced
source / legacy workbench**, enabled only by the explicit `Load source pairing
and metadata workbench` control. Existing Benchmark/ML/Regression/manual and
Technology Delta tools remain reachable under **Advanced evidence and
recommendation workbench**. Legacy saved estimates remain separately opt-in.
Raw audit, metadata and provenance surfaces are under **Technical audit and
diagnostics** rather than the primary composition path.

## Incomplete Current behavior

An active VDE snapshot now anchors the matrix even when its legacy metadata is
sparse. The user can see the Current composition, calculate it, receive the
canonical `NOT READY` status and friendly structured issues, then resolve the
needed domain or choose the advanced source workbench. No metadata table is a
prerequisite for seeing the System Scenario model.

## Automated hierarchy evidence

The Powertrain AppTest suite now includes:

- `test_primary_workspace_is_visible_before_legacy_source_setup` — matrix and
  primary workspace are present while legacy source controls are absent by
  default;
- `test_incomplete_current_keeps_matrix_visible_and_reports_not_ready` — a
  sparse Current VDE retains the matrix and reports canonical `NOT READY`;
- `test_legacy_source_and_metadata_are_reachable_only_by_opt_in` — the source
  pairing and metadata controls remain reachable, but only after opt-in.

Existing AppTests continue to cover Proposal A creation, three-Proposal limit,
single domain editor/VDE selection, architecture applicability, stale state,
calculation result cards, and advanced evidence access.

## Manual recheck still required

The in-app browser connection was unavailable in this environment, so this
document makes no new manual-browser claim. A real recheck should cover:

1. initial viewport hierarchy;
2. Add Proposal;
3. Proposal A / Transmission domain editing;
4. Calculate System Scenarios;
5. stale status after an edit.

## Regression evidence

- hierarchy AppTests and the canonical System Scenario suites pass locally;
- `test_system_scenario*.py`: 131 passing in 20.931 s;
- `test_powertrain_system_scenario_viewmodels.py`: 22 passing in 0.041 s;
- final full suite: 1,803 tests in 1,116.380 s, with the pre-existing one
  failure (`test_axle_hubs_lookup_snapshot_preserves_boundary_metadata`) and
  one error (`test_component_lookup_provenance_does_not_change_parasitic_math`)
  from `test_vde_request_resolver`, and no hotfix regression;
- `compileall` and `git diff --check` pass after the final edit.

## Core boundary audit

The code change is confined to the page, System Scenario UI ordering, legacy
UI source-resolution helper, component export and AppTests. It adds no
physical formula, solver invocation, contract change, database/schema
operation, persistence behavior, Comparison integration or topology logic.
