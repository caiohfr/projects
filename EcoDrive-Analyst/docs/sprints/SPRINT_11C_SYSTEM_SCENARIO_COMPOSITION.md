# Sprint 11C — System Scenario composition and Energy Balance L0 adapter

## Scope and outcome

Sprint 11C implements the Streamlit-free calculation seam established by
11A/11B:

```text
SystemScenarioDefinition
    -> deterministic System Scenario resolver
    -> ResolvedSystemScenario + Fidelity Manifest + solver readiness
    -> EnergyBalanceL0Adapter
    -> FuelEstimateRequest -> run_fuel_estimation()
    -> optional canonical Technology Delta stack
    -> SystemScenarioResult
```

This package stops before Sprint 11D. No Powertrain UI, database schema,
persistence, Comparison integration, topology model, new ML/Regression model,
or Vehicle Demand physics was added.

## Pre-flight and ownership cleanup

The pre-11C full-suite baseline was run from commit `5f0f8b6c` plus Claude's
two pre-flight ownership edits. Actual result: **1,737 tests**, **1 failure,
1 error**. Both were the already-recorded `test_vde_request_resolver.py`
failures:

- `test_axle_hubs_lookup_snapshot_preserves_boundary_metadata` — failure;
- `test_component_lookup_provenance_does_not_change_parasitic_math` — error.

`TechDeltaAssumption` and its JSON-boundary parser now live beside the
canonical stack in `src/vde_core/technology_delta.py`. Quick Scenario keeps
backward-compatible re-exports of the exact same class/parser objects; System
Scenario imports the neutral owner directly. The typed-contract-to-stack-dict
adapter is also shared there, eliminating the former Quick-only field copy.

## Composition semantics

- A single scenario resolves its selected domains in the fixed
  `ALL_DOMAIN_KINDS` order, never mapping insertion order or presentation
  order.
- A bounded working-set resolver validates unique scenario IDs and unique
  `(role, proposal_index)` pairs for Current + at most three Proposals.
- Each scenario receives its own fresh `FuelEstimateRequest`; its selected
  frozen `VehicleDemandResult` replaces any stale demand values in the request
  template. Definitions, Domain Proposals, Vehicle Demand results, and request
  templates are not mutated.
- Direct aggregate L0 assumptions use the existing canonical request keys
  (`eta_pt_est`, `bev_eff_drive`, `utility_factor`, grid intensity, LHV, and
  fuel CO2 factor). Conflicting values from different selected domains make the
  scenario explicitly `NOT_READY`; no silent last-write precedence exists.
- Explicit effect-basis assumptions such as `pse_percent_delta` are represented
  as typed Technology Delta assumptions, then evaluated by the existing
  canonical stack. No domain formula was introduced.

## Technology Delta ordering

The cross-domain convention is explicit and deterministic:

1. fixed `ALL_DOMAIN_KINDS` order;
2. within a selected Domain Proposal, explicit L0 effect-basis entries first;
3. then that proposal's `technology_deltas` in their existing tuple order;
4. one call to `apply_delta_stack_to_baseline()` for the complete ordered list.

This is an orchestration convention only. Additive, percentage,
multiplicative, reconciliation, maturity, confidence, and registered-only
behavior remain owned by `technology_delta.py` unchanged.

## L0 adapter and PHEV canonical owner

The spec explicitly defines `FuelEstimateRequest -> run_fuel_estimation()` as
the Sprint 11 L0 boundary. Therefore the adapter treats
`run_fuel_estimation()` as canonical for ICE, MHEV, HEV, PHEV, and BEV parity.
It does not call or reproduce the older DB-reading
`fuel_energy.compute_ice_fuel_from_vde()` path.

This resolves the 11A/11B PHEV ownership ambiguity without changing physics:
System Scenario preserves the current top-level PHEV CO2 behavior of
`run_fuel_estimation()`, including its fuel-side-only top-level `gco2_km`.
The known disagreement with the older helper's fuel-plus-grid CO2 behavior is
documented, not “fixed” locally. PHEV parity is tested against an independent
canonical `run_fuel_estimation()` call.

`SystemScenarioResult.fuel_estimate_result` retains the canonical baseline
object. When selected assumptions contain Technology Deltas,
`technology_delta_result` retains the complete canonical stack audit result.
`effective_outputs` exposes the already-computed final proposal outputs (or
the baseline outputs when no stack exists), so future Comparison consumption
does not require recalculation.

## Readiness and Fidelity Manifest

Solver readiness is distinct from domain data completeness:

- selected TOTAL/NET demand and Architecture classification are required;
- Physics Simple thermal paths require `eta_pt_est`;
- BEV requires `bev_eff_drive`;
- PHEV requires both current canonical path efficiencies;
- future-only configuration fields such as Engine torque do not block L0;
- conflicts in direct aggregate assumptions block rather than silently choose.

Fidelity is evaluated per selected architecture and domain:

- an already-resolved Vehicle Demand and Architecture are quantitative;
- explicit supported aggregate assumptions/deltas are
  `EFFECTIVE_ASSUMPTION`;
- unsupported configuration changes remain `CONFIGURATION_ONLY`;
- absent or architecturally not-applicable domains are `NOT_REPRESENTED`.

Thus a Transmission gear/final-drive change or battery-capacity change alone
is preserved visibly but cannot invent a fuel-consumption benefit.

## Requirement and acceptance traceability

| Requirement / case | Evidence classification | Evidence |
|---|---|---|
| REQ-11-001/002/003, Cases A/B | DIRECT TEST | neutral canonical parity and two independent VDE scenarios |
| REQ-11-004/005, Case L | DIRECT TEST + 11A/11B coverage | eight-domain contracts and BEV applicability/fidelity |
| REQ-11-006/007/008 | DIRECT TEST + 11B coverage | selected proposal composition, Effective Current base, no Proposal-to-Proposal lineage |
| REQ-11-009/010, Cases D/F | DIRECT TEST | Transmission and battery configuration-only parity |
| REQ-11-011/012/013 | DIRECT TEST | real resolver, manifest, READY/NOT_READY and missing torque case |
| REQ-11-014/015, Cases A/E/G | DIRECT TEST | independent canonical L0 parity, canonical delta stack, efficiency direction |
| REQ-11-016/017 | INDIRECT CANONICAL COVERAGE | existing `fuel_estimation`, Quick Scenario ML/Regression/Benchmark tests remain green |
| REQ-11-018 | DIRECT TEST | canonical result, delta audit result, effective outputs, source identities |
| REQ-11-019 | INSPECTION | no UI work in 11C; new modules import no Streamlit |
| REQ-11-020/021 | INSPECTION + DIRECT TEST | no schema/write path; request/source immutability |
| REQ-11-022/023/024 | DIRECT TEST + 11A/11B coverage | stable identity, isolation, fixed domain order, bounded working set |
| REQ-11-025 | INDIRECT CANONICAL COVERAGE | unchanged canonical result/save contracts; focused regression green |
| Case C/J/K | DIRECT TEST + 11A/11B coverage | reusable immutable Domain Proposals and Effective Current correction tests |
| Case H | INDIRECT CANONICAL COVERAGE | existing Quick recommendation-not-adopted tests; 11C only applies selected assumptions |

UI AppTests and manual browser smoke are **not 11C closure claims**; they belong
to 11D/11E. Broad save/Comparison integration is also deferred by the spec.

## Tests

- Focused 11C + affected ownership/regression suites: **254 tests, all
  passing**.
- New `test_system_scenario_composition.py`: **18 tests**, covering canonical
  ICE/PHEV parity, independent VDEs, immutability, readiness, architecture
  applicability, configuration-only behavior, explicit zero, deterministic
  delta order, canonical delta parity, efficiency direction, conflicts, and
  Current + three working-set bounds.
- Technology Delta ownership: three direct identity/round-trip tests prove
  Quick and System consumers share the neutral canonical contract/parser.
- Post-change full suite: **1,758 tests**, **1 failure, 1 error**. The only
  non-green tests are the same two pre-existing `test_vde_request_resolver.py`
  cases recorded in the 1,737-test baseline above; **zero new regressions**.
  The reported wall time was inflated by workstation suspension during the
  run, so it is intentionally not used as a performance measurement.

## Freeze / handoff

Sprint 11C ends at the core resolver/result boundary. Sprint 11D UI has not
started. The pre-existing `.claude/` directory remains unrelated and untouched.
