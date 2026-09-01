# Sprint 11C — System composition, L0 resolution, and canonical result

## Outcome and scope

Sprint 11C closes the Streamlit-free System Scenario calculation boundary:

```text
SystemScenarioDefinition
    -> resolve_system_scenario()
    -> immutable ResolvedSystemScenario
    -> EnergyBalanceL0Adapter
    -> fresh FuelEstimateRequest
    -> fuel_estimation.run_fuel_estimation()
    -> optional technology_delta.apply_delta_stack_to_baseline()
    -> SystemScenarioResult
```

No UI matrix, Comparison integration, persistence, database/schema change,
topology model, new Vehicle Demand physics, PSE reinterpretation, or second
fuel/energy solver was introduced. Sprint 11D was not started.

## Pre-flight findings

### PHEV CO2 ownership

The conflict identified in 11A/11B was reproduced directly with the smallest
common input: VDE 1.8 MJ/km, `eta_pt_est=0.3`, gasoline LHV 32 MJ/L,
`bev_eff_drive=0.9`, utility factor 0.4, and grid intensity 400 g/kWh.

| Existing path | top-level CO2 result |
|---|---:|
| `fuel_estimation.run_fuel_estimation()` | 259.875 g/km |
| `fuel_energy.compute_ice_fuel_from_vde()` | 348.7638888889 g/km |
| Exact difference | 88.8888888889 g/km |

The difference is the utility-weighted electric/grid term. The first path's
top-level PHEV `gco2_km` contains the fuel term; the legacy helper adds the
electric/grid term. `fuel_estimation` phase-output logic also includes the
grid term, so the inconsistency remains documented rather than silently
declared physically correct.

Ownership for System Scenario is nevertheless unambiguous from current API
and lifecycle evidence:

- `run_fuel_estimation()` returns the canonical `FuelEstimateResult`, is used
  by the active Powertrain service/UI and Quick Scenario, and its top-level
  `gco2_km` is what `_build_fuelcons_payload()` persists;
- the existing `test_run_fuel_estimation_physics_simple_phev_combines_both_paths`
  locks the 259.875 g/km top-level contract;
- `compute_ice_fuel_from_vde()` reads an arbitrary legacy DB row, returns an
  untyped dict, has no active caller under `src/`, and is only re-exported by
  `services.py` for compatibility.

Therefore 11C delegates PHEV to `run_fuel_estimation()` and preserves its
existing semantics. It does not choose a new physical truth or alter either
formula. Direct evidence:
`test_phev_co2_preflight_reproduces_legacy_helper_disagreement` and
`test_phev_parity_preserves_run_fuel_estimation_as_owner`.

### Technology Delta owner

The canonical contract and stack now share the neutral owner
`src/vde_core/technology_delta.py`. `TechDeltaAssumption`, parser, serializer,
normalization, and `apply_delta_stack_to_baseline()` are reused by Quick and
System Scenario; Quick exposes exact aliases for compatibility. No second
schema or stacking implementation exists in `system_scenario`.

### Deterministic cross-domain order

The orchestration convention is:

1. fixed `ALL_DOMAIN_KINDS` order;
2. within a selected Domain Proposal, explicit L0 effect-basis assumptions;
3. then that proposal's `technology_deltas` tuple in its existing local order;
4. one call to the canonical stack for the complete active list.

This is ordering, not new stacking math. The existing canonical owner already
defines sequential behavior for all supported bases. Disabled and
metadata-only entries do not enter the quantitative stack. `map_based_effect`
becomes `unsupported_quantitative_representation`; an unknown basis becomes
`incompatible_technology_delta_basis`. Both leave the scenario `NOT_READY`
without combining anything.

## Composition and immutable resolution

The working set accepts Current plus at most three Proposals and validates
unique scenario IDs and unique `(role, proposal_index)` pairs. Every scenario
is resolved independently; no scenario inherits another scenario's VDE,
request, delta stack, or proposal editor state.

Definitions, proposal mappings, resolved-domain mappings, Fidelity mappings,
and effective-assumption mappings are read-only. Resolution stores an
`EnergyBalanceL0RequestSnapshot`; the compatibility request property creates
a fresh `FuelEstimateRequest` for every access/run. A shared Domain Proposal
can therefore be selected by A and B without mutation.

The selected frozen `VehicleDemandResult` is the only Vehicle Demand source.
The adapter clears any template VDE ID, TOTAL, NET, phase outputs, and
Architecture before applying the selected scenario. Missing TOTAL/NET never
falls back to the other basis or to stale template data.

## Architecture compatibility

`domain_applicability_for()` remains the fixed Architecture classification.
An absent/not-selected N/A domain is valid: BEV needs no Engine state for L0.
An explicit Domain Proposal in an N/A slot is a composition conflict and
produces `architecture_domain_incompatible:<architecture>:<domain>:<proposal>`.
The resolver does not invent topology or component connectivity.

## Solver readiness and issues

Domain completeness, Architecture applicability, and L0 readiness remain
separate. For `physics_simple`:

- the selected explicit TOTAL or NET Vehicle Demand value is required;
- Architecture is required;
- ICE/MHEV/HEV fuel paths require positive `eta_pt_est`;
- BEV requires positive `bev_eff_drive`;
- PHEV requires both current canonical path efficiencies;
- future-only metadata such as Engine torque or Battery capacity does not
  block L0;
- conflicting direct assumptions, incompatible selections, and unsupported
  quantitative representations produce structured issues and `NOT_READY`.

Programming/schema corruption may still raise. Ordinary unresolved scenarios
return a `SystemScenarioResult` with readiness, issues/warnings, metadata, and
no calculated fuel result.

## Fidelity Manifest and effective assumptions

Fidelity is based on what actually enters the calculation:

- `QUANTITATIVE`: resolved Vehicle Demand, Architecture classification, and
  selected fuel identity used by the current L0 request;
- `EFFECTIVE_ASSUMPTION`: an adopted direct aggregate assumption or active
  supported Technology Delta;
- `CONFIGURATION_ONLY`: visible configuration that the current L0 does not
  consume quantitatively;
- `NOT_REPRESENTED`: absent or Architecture-N/A domain.

`eta_pt_est` remains an aggregate powertrain/path assumption; it is not
renamed engine efficiency. `bev_eff_drive`, `utility_factor`, LHV, fuel CO2,
and grid intensity keep their existing canonical meanings. Each adopted
direct assumption records key/value, domain, proposal, and `ASSUMED`
provenance.

## L0 adapter and configuration-only invariant

`EnergyBalanceL0Adapter` translates the resolved snapshot into a fresh
`FuelEstimateRequest` and delegates. It contains no MJ/km-to-fuel, electric,
LHV, CO2, PHEV weighting, BEV, PSE, or delta-stacking formula.

Battery capacity, Transmission gear count/final drive, and Engine
displacement/rated power changes produce identical current-L0 results when
no adopted representation changes an input. Engine domain fidelity can still
be `QUANTITATIVE` because its unchanged fuel selection is consumed; that does
not imply displacement or rated power affected the answer.

## Technology Delta composition and provenance

Only normalized status `applied` entries with a basis supported by the
canonical stack enter `ordered_technology_deltas`. For each entry,
`TechnologyDeltaContribution` retains:

- evaluation order;
- contributing domain;
- proposal ID;
- the canonical assumption, including source type, basis, value, maturity,
  confidence, notes, and enabled state;
- quantitative status.

Engine, Transmission, and Controls compatible deltas are tested in fixed
cross-domain order with Transmission local order preserved. The resolved
scenario and final result provenance expose the same audit trail.

## SystemScenarioResult

The result contains scenario identity, selected VDE identity, Architecture,
readiness, Fidelity Manifest, effective assumptions, canonical baseline
`FuelEstimateResult`, optional complete canonical Technology Delta result,
solver identity/version, model method, warnings, and separate configuration,
assumption, delta, and `CALCULATED` result provenance.

`effective_outputs` exposes existing calculated fuel, electric, CO2, PSE, and
consumed-energy metrics and overlays the already-calculated canonical delta
proposal where present. It performs no scorecard or physical recalculation.
The shape is suitable for future Comparison adoption, but 11C does not wire it.

## Parity and determinism evidence

- ICE: `test_neutral_current_matches_independent_canonical_call` compares
  against a separately constructed canonical request.
- BEV: `test_neutral_bev_matches_independent_canonical_call` does the same for
  electric energy and CO2.
- PHEV: ownership reproduction and canonical parity tests described above.
- Current/A/B isolation and shared proposal reuse:
  `test_current_a_b_use_independent_vdes_and_reuse_shared_proposal`.
- repeated result equality including serialized metadata/provenance:
  `test_result_is_deterministic_and_carries_solver_fidelity_and_provenance`.
- efficiency direction: `test_direct_higher_efficiency_reduces_canonical_fuel_input`
  proves both higher-efficiency/lower-input and lower-efficiency/higher-input
  sides using canonical results.

Expected numerical parity values come from independent canonical calls. The
PHEV conflict reproduction intentionally records existing outputs; it does not
reimplement either formula as the System Scenario expected-value owner.

## Test evidence

- Historical baseline entering 11C: 1,737 tests, one failure and one error.
- Fresh post-initial-11C baseline at commit `bdb2f583`: 1,758 tests, the same
  failure/error and zero new regressions.
- System Scenario focused suites: 129 tests, all passing.
- Affected System Scenario, Fuel Estimation, Technology Delta, Quick Scenario,
  Powertrain/PSE service suites: 431 tests, all passing in 94.439 s.
- Final full suite: 1,769 tests in 687.293 s, with one failure and one error;
  both are the same pre-existing VDE request resolver cases observed in the
  baseline, so Sprint 11C introduced zero new regressions.

The two known baseline failures are:

- `test_axle_hubs_lookup_snapshot_preserves_boundary_metadata` — failure;
- `test_component_lookup_provenance_does_not_change_parasitic_math` — error.

## Persistence compatibility and deferred work

No persistence API was changed. The existing save payload consumes canonical
`FuelEstimateResult`; adoption of `SystemScenarioResult` by persistence is
deferred because it would broaden 11C into a save-path redesign.

Deferred to 11D/11E or later: multi-domain UI matrix, presentation/viewmodel
integration, Comparison consumption, persistence integration, new
recommendation UX, higher-fidelity component/topology models, phase/result DB
work, and any product decision to redefine PHEV CO2 semantics.

The pre-existing `.claude/` directory was left untouched.
