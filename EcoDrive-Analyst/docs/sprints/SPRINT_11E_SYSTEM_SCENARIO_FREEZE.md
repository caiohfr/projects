# Sprint 11E — System Scenario QA, traceability, and freeze

## 1. Sprint 11 summary and final status

Sprint 11 established a multi-domain Powertrain System Scenario foundation:
immutable Streamlit-free domain contracts, Effective Current and reusable
Domain Proposals, deterministic System Scenario composition, the canonical
Energy Balance L0 adapter, a compact Current plus three-Proposal UI, and
explicit readiness/fidelity/provenance.

11E is a QA and traceability closure. It introduces no product capability,
physical formula, PHEV or Technology Delta semantic, schema, persistence,
Comparison integration, topology graph, or Sprint 12 data work. Subject to
the post-freeze Sprint 11D UX alignment and the manual recheck recorded below,
Sprint 11 is **UX HOTFIX APPLIED / MANUAL RECHECK REQUIRED**.

The Sprint 11D hotfix is limited to hierarchy/orchestration: it puts the
System Scenario matrix before the retained legacy source-pairing workbench.
See `SPRINT_11D_HOTFIX_UX_ALIGNMENT.md` for the confirmed UX acceptance
failure, direct AppTest evidence and manual-recheck scope.

## 2. Baseline and current-branch classification

Pre-change branch state was `sprint-11-system-scenario` at `73e63480`; the
only unrelated worktree entry was the pre-existing, untouched `../.claude/`
directory. The current branch's System Scenario focused baseline passed 129
tests in 9.457 s.

The full current-branch baseline reproduced 1,797 tests in 1,177.623 s with
one failure and one error:

| Classification | Test | Current reproduction | Scope decision |
|---|---|---|---|
| Baseline failure, outside Sprint 11 | `test_axle_hubs_lookup_snapshot_preserves_boundary_metadata` | expected `A=120.5`, actual `120.0` | `test_vde_request_resolver`, no Sprint 11 module is on its execution path; do not alter frozen Vehicle Demand behavior in 11E. |
| Baseline error, outside Sprint 11 | `test_component_lookup_provenance_does_not_change_parasitic_math` | lookup snapshot is `None`, then test subscripts it | `test_vde_request_resolver`, no Sprint 11 module is on its execution path; do not redesign component lookup in 11E. |

These are classified from this branch's actual run, not merely carried forward
from historic closure documents. No Sprint 11 defect was confirmed.

## 3. Final architecture audit

The exercised calculation path is:

```text
legacy/current rows
  -> system_scenario.legacy_adapter
  -> DomainSourceState / EffectiveDomainState
  -> DomainProposal
  -> SystemScenarioDefinition
  -> resolve_system_scenario()
  -> ResolvedSystemScenario
  -> EnergyBalanceL0Adapter
  -> run_fuel_estimation()
  -> SystemScenarioResult
```

`pages/Powertrain_Scenario.py` is a page/composition entry point;
`pwt_system_scenario.py` renders widgets and formats existing result fields;
and `powertrain_system_scenario_viewmodels.py` converts drafts to contracts,
fingerprints them, and calls `run_system_scenario()`. Inspection of imports
and executable arithmetic confirms none contains Vehicle Demand, fuel/energy,
PSE, BEV, PHEV, or Technology Delta formula. The UI has no Comparison import,
write/query persistence operation, migration, or schema statement.

## 4. Requirement traceability — invariants

| Invariant | Tier | Owner and evidence |
|---|---|---|
| INV-11-001 Neutral | DIRECT TEST | `test_neutral_current_matches_independent_canonical_call`; resolver/L0 adapter parity. |
| INV-11-002 Determinism | DIRECT TEST | `test_result_is_deterministic_and_carries_solver_fidelity_and_provenance`; resolver. |
| INV-11-003 Source immutability | DIRECT TEST | `test_correction_does_not_mutate_source_configuration`, `test_vehicle_demand_adapter_never_mutates_the_source_row`. |
| INV-11-004 Proposal independence | DIRECT TEST | `test_editing_proposal_a_does_not_mutate_b`, `test_case_g_proposal_isolation`. |
| INV-11-005 Domain reuse | DIRECT TEST | `test_shared_domain_proposal_is_reused_by_a_and_b`, `test_current_a_b_use_independent_vdes_and_reuse_shared_proposal`. |
| INV-11-006 VDE separation | DIRECT TEST | `test_snapshot_adapter_does_not_recalculate_vehicle_demand`; snapshot adapter. |
| INV-11-007 Fidelity honesty | DIRECT TEST | Battery, Transmission, Engine, and new Electric Drive configuration-only invariance tests. |
| INV-11-008 Higher efficiency | DIRECT TEST | `test_direct_higher_efficiency_reduces_canonical_fuel_input`. |
| INV-11-009 Lower efficiency | INDIRECT CANONICAL COVERAGE | Same controlled low/high canonical comparison proves the inverse direction. |
| INV-11-010 Recommendation separation | DIRECT TEST | `test_unadopted_recommendation_does_not_change_fingerprint_or_result`. |
| INV-11-011 Provenance | DIRECT TEST | `test_adopted_ml_recommendation_flows_with_provenance`; new manual-override provenance test. |
| INV-11-012 DB independence | INSPECTION + DIRECT TEST | adapter boundary and `test_legacy_adapter_isolation_canonical_contract_never_exposes_raw_row_keys`. |

## 5. Requirement traceability — functional requirements

| Requirement | Tier | Owner and exact evidence |
|---|---|---|
| REQ-11-001 | DIRECT TEST | `test_current_plus_one_and_three_proposals_have_stable_bounded_identities`; UI max-three AppTest. |
| REQ-11-002 | DIRECT TEST | `test_each_proposal_selects_its_own_vehicle_demand`. |
| REQ-11-003 | DIRECT TEST | `test_each_scenario_uses_its_own_vehicle_demand`. |
| REQ-11-004 | DIRECT TEST | `test_all_eight_domains_have_a_matching_configuration_type`. |
| REQ-11-005 | DIRECT TEST | architecture applicability tests in `test_system_scenario_domain_resolution.py`. |
| REQ-11-006 | DIRECT TEST | `test_proposal_is_based_on_effective_current_after_correction`. |
| REQ-11-007 | DIRECT TEST | `test_proposal_based_on_another_proposal_is_rejected`. |
| REQ-11-008 | DIRECT TEST | `test_shared_domain_proposal_is_reused_by_a_and_b`. |
| REQ-11-009 | DIRECT TEST | `test_case_b_engine_proposal_no_invented_consumption_benefit`; proposal resolver. |
| REQ-11-010 | DIRECT TEST | configuration-only invariance tests for all four required examples. |
| REQ-11-011 | DIRECT TEST | `test_result_is_deterministic_and_carries_solver_fidelity_and_provenance`. |
| REQ-11-012 | DIRECT TEST | same result test plus `test_fidelity_manifest_roundtrip`. |
| REQ-11-013 | DIRECT TEST | `test_metadata_incomplete_is_distinct_from_ready_solver`. |
| REQ-11-014 | DIRECT TEST | `test_ui_orchestration_delegates_once_per_scenario`; independent canonical parity. |
| REQ-11-015 | INDIRECT CANONICAL COVERAGE | `run_fuel_estimation()` and `powertrain_efficiency` remain owners; UI displays canonical PSE. |
| REQ-11-016 | DIRECT TEST | `test_explicit_l0_effect_uses_canonical_delta_stack`. |
| REQ-11-017 | INSPECTION + AppTest | legacy evidence workbench remains opt-in; `test_legacy_evidence_capability_is_opt_in_not_second_default_workflow`. |
| REQ-11-018 | DIRECT TEST | `test_result_is_deterministic_and_carries_solver_fidelity_and_provenance`. |
| REQ-11-019 | INSPECTION | import/arithmetic audit of the three Sprint 11 UI modules. |
| REQ-11-020 | INSPECTION | commit/file audit: no migration or schema code. |
| REQ-11-021 | DIRECT TEST | source immutability tests in contracts and legacy adapter suites. |
| REQ-11-022 | DIRECT TEST | `test_visible_label_neither_changes_identity_nor_calculation_fingerprint`. |
| REQ-11-023 | DIRECT TEST | `test_editing_proposal_a_does_not_mutate_b`. |
| REQ-11-024 | DIRECT TEST | `test_domain_order_not_slot_or_presentation_order_controls_stack`. |
| REQ-11-025 | INSPECTION + AppTest | legacy saved estimates are explicitly opt-in and unmodified; page AppTest proves the new calculation remains primary. |

## 6. Acceptance-case and system-set traceability

| Case | Tier | Evidence |
|---|---|---|
| A Neutral Current | DIRECT TEST | `test_neutral_current_matches_independent_canonical_call`. |
| B Different VDEs | DIRECT TEST | `test_current_a_b_use_independent_vdes_and_reuse_shared_proposal`. |
| C Shared Engine | DIRECT TEST | same test and `test_shared_domain_proposal_is_reused_by_a_and_b`. |
| D Transmission configuration only | DIRECT TEST | `test_transmission_configuration_only_does_not_change_baseline`. |
| E Explicit L0 effect | DIRECT TEST | `test_explicit_l0_effect_uses_canonical_delta_stack`. |
| F Battery configuration only | DIRECT TEST | `test_battery_capacity_change_is_configuration_only`. |
| G Higher efficiency | DIRECT TEST | `test_direct_higher_efficiency_reduces_canonical_fuel_input`. |
| H Unadopted ML | DIRECT TEST | `test_unadopted_recommendation_does_not_change_fingerprint_or_result`. |
| I Future-only metadata | DIRECT TEST | `test_missing_future_only_engine_torque_does_not_block_l0`. |
| J Proposal isolation | DIRECT TEST | `test_editing_proposal_a_does_not_mutate_b`. |
| K Source correction | DIRECT TEST | `test_case_a_transmission_correction`. |
| L Applicability | DIRECT TEST | BEV/ICE/MHEV/HEV/PHEV applicability tests. |

Working-set uniqueness is directly covered by
`test_current_plus_three_proposals_is_the_maximum`,
`test_duplicate_proposal_index_is_rejected_within_working_set`, and the new
`test_duplicate_scenario_id_is_rejected_within_working_set`. Replacement is
directly covered by `test_recalculation_replaces_same_scenario_identity`.

## 7. Eight-domain audit

| Domain | Effective Current / Proposal | Applicability and sparse behavior | Fidelity/provenance evidence |
|---|---|---|---|
| Vehicle Demand | frozen persisted snapshot adapter; no VDE recalc | required; missing TOTAL/NET is NOT READY | QUANTITATIVE; snapshot/provenance tests. |
| Architecture | Effective Current classification | required, ICE/MHEV/HEV/PHEV/BEV | QUANTITATIVE; applicability tests. |
| Engine | legacy configuration + proposal | BEV N/A; missing torque allowed | fuel identity quantitative, assumptions explicit. |
| Transmission | legacy configuration + proposal | optional | configuration-only unless explicit representation. |
| Electric Drive | explicitly sparse configuration + proposal | BEV/HEV/etc. required by classification | configuration-only or electric-path assumption; no invented motor data. |
| Energy Storage | legacy sparse configuration + proposal | required where electrified | configuration-only unless supported representation. |
| Controls | legacy utility-factor metadata + proposal | optional | supported L0 assumption only. |
| Aux/Thermal | ambient/AC metadata + proposal | optional and safely sparse | configuration-only/not represented. |

## 8. Fidelity and readiness audit

The canonical meanings are retained: `QUANTITATIVE` for selected Vehicle
Demand/Architecture and consumed fuel identity, `EFFECTIVE_ASSUMPTION` for
explicit direct assumptions or active compatible deltas,
`CONFIGURATION_ONLY` for visible but unconsumed configuration, and
`NOT_REPRESENTED` for absent/N/A domains.

Direct numerical invariance is now covered for Battery capacity, Transmission
gear/FDR, Engine displacement/rated power, and Electric Drive motor power.
Readiness is separately proven for missing Engine torque (READY), BEV Engine
N/A, BEV missing electric assumption (NOT READY), ICE missing fuel-path
assumption (NOT READY), and partial per-scenario calculation.

## 9. Technology Delta, PSE, and PHEV audit

`src/vde_core/technology_delta.py` remains the sole stack owner. The resolver
collects proposal deltas in fixed `ALL_DOMAIN_KINDS` order while preserving
each local tuple order; unsupported/disabled/metadata-only entries produce
structured issues or no impact. There is no System Scenario dependency on a
Quick Scenario stacking owner and no duplicate stack implementation.

PSE remains a system result. Current/Benchmark/ML/Regression/manual/Technology
are evidence or assumption-resolution mechanisms, not domains. An unadopted
recommendation changes neither fingerprint nor result. An adopted ML value
records `ML_DERIVED`; the new override regression verifies a replacement
engineering value records `ASSUMED` rather than stale ML provenance.

PHEV continues through `run_fuel_estimation()`. The known inactive-helper
difference remains documented by
`test_phev_co2_preflight_reproduces_legacy_helper_disagreement`; this is a
canonical software-ownership decision, not a physical/product redesign.

## 10. Stale, persistence, Comparison, and rerun audit

Fingerprints exclude labels but include canonical physical input. A physical
edit marks only the affected draft stale, hides old metrics, and Calculate
replaces the result under the same identity. Shared Proposal changes are a
genuine shared dependency; independent A/B/C selections are not.

System Scenario persistence is **DEFERRED**. Domain Proposal persistence is
**DEFERRED**. Legacy saved estimates remain visibly legacy and opt-in; no
ad-hoc JSON was introduced. There is no Sprint 11 Comparison state dependency
or integration. Calculation remains a single explicit button, while evidence
workbench code is opt-in, avoiding unnecessary ML/Regression reruns.

## 11. Bugs and hypotheses investigated in 11E

No Sprint 11 core or physics bug was confirmed. A later real manual browser
observation did confirm the separate UX acceptance failure documented in
`SPRINT_11D_HOTFIX_UX_ALIGNMENT.md`; its UI hierarchy hotfix is applied and
awaits manual recheck.

| Item | Classification | Evidence / disposition |
|---|---|---|
| Electric motor power may alter L0 accidentally | GAP | New invariance regression added; no behavior change required. |
| Duplicate `scenario_id` may pass working-set validation | GAP | New direct rejection regression added; implementation already rejects it. |
| Manual override may retain stale ML provenance | NOT REPRODUCED | New regression proves explicit engineering replacement yields `ASSUMED`. |
| Browser smoke may be possible | ENVIRONMENT BLOCKER | Local Streamlit server starts, but the available in-app browser connection cannot initialize; no manual claim. |
| Two VDE resolver failures | BASELINE, OUTSIDE SPRINT 11 | Reproduced in current full baseline; no Sprint 11 path or permitted scope to change. |

## 12. Manual browser smoke and AppTest status

Manual cases A–H were not executed because the browser integration failed to
initialize after a local Streamlit server was started. The server was stopped.
The exact blocker is the unavailable in-app browser connection in this session,
not an application assertion. Therefore 11E does not claim Current ICE,
different-VDE Transmission, BEV, PHEV, configuration-only Battery, Technology
Delta, Current-plus-three, or stale flow as manual-browser validated.

This is deliberately separate from the passing 7-test Streamlit AppTest suite:
Current calculation, bounded scenarios, stable labels, independent VDE, BEV
N/A/partial result, stale/recalculation, and opt-in legacy evidence behavior.
The post-freeze UX hotfix adds hierarchy-specific AppTests but does not convert
AppTest evidence into a manual smoke claim.

## 13. Regression evidence

- baseline focused System Scenario: 129 passing, 9.457 s;
- new composition/viewmodel coverage: 53 passing, 0.031 s;
- Powertrain AppTests: 7 passing, 150.276 s;
- final affected suite: 482 tests in 112.915 s, with the same one failure and
  one error reproduced from the VDE baseline;
- final full suite: 1,800 tests in 1,125.462 s, again with exactly the same
  one failure and one error and zero new Sprint 11 regressions.

`compileall` and `git diff --check` are also rerun after the final traceability
update.

## 14. SDD pilot retrospective

The formal spec prevented scope drift: the no-physics, no-schema, no
Comparison, no-topology and no-persistence boundaries were repeatedly useful
when auditing UI and save status. The only practical ambiguity was whether
manual browser smoke could be satisfied by AppTest; `AGENTS.md`'s permanent
rule correctly prevented that misclassification. The recommendation
"manual override" wording was resolved without inventing a new architecture:
the existing explicit evidence-source control governs provenance, now covered
by a regression.

`AGENTS.md` remains permanent-rule-only on inspection. The reusable Sprint
implementation skill was useful during earlier packages as a workflow guard,
but closure evidence belongs in sprint docs and tests rather than the skill.
The earlier split between 11A's shell and 11C's real resolver was useful but
made repeated historical test counts redundant; Sprint 12 should use one
closure table with fresh baseline/final commands rather than copying counts
between package documents.

## 15. Deferred backlog and freeze boundary

- Real browser smoke for cases A–H in an environment with working browser
  integration; this is the only freeze gate still open.
- System Scenario and Domain Proposal persistence design.
- Comparison consumption, scorecards, and presentation ordering.
- Persistent proposal library, multi-delta UI editing, topology/maps/SOC and
  higher-fidelity models.
- PHEV product-semantic redesign, if ever approved.

No next product sprint is started by this closure.
