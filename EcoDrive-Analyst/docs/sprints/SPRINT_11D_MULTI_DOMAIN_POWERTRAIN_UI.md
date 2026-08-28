# Sprint 11D — Multi-domain Powertrain System Scenario UI

## Outcome and scope

The Powertrain page now presents one canonical multi-domain System Scenario
workspace. Current and up to three independent Proposals are composed in a
compact matrix, edited one domain at a time, and calculated by the Sprint 11C
owner. Sprint 11D adds UI/state orchestration only: no new physics, PSE rule,
PHEV interpretation, Technology Delta math, topology model, Comparison
adapter, persistence model, or database/schema change was introduced.

## Old to new page architecture

The former default page was method-centric: baseline confirmation followed by
Benchmark, ML, Regression, manual estimation, Technology Delta, and legacy
save/result blocks. Those tools remain available in an explicitly opt-in
"Evidence and recommendation workbench" because they still provide valid
engineering evidence. They are no longer a second default calculation flow.

The new default owner is:

```text
active persisted VDE snapshot
    -> Current + zero-to-three independent scenario drafts
    -> compact domain composition matrix
    -> one-domain-at-a-time editor
    -> Calculate System Scenarios
    -> canonical Sprint 11C result per scenario
```

Legacy saved estimates remain separately opt-in. Their existing management
capability was not deleted or silently converted to a System Scenario store.

## Scenario matrix and identity

Matrix rows are Vehicle Demand, Architecture, Engine, Transmission, Electric
Drive, Energy Storage, Controls, Aux/Thermal, and Status. Columns are Current
and active Proposals. The matrix contains selections and status, not scalar
input grids.

Stable canonical identities are `SYS-CURRENT` and `SYS-P1` through `SYS-P3`.
Editable labels are presentation only and do not enter the calculation
fingerprint. Removing a Proposal frees its bounded index; Current cannot be
removed. Each draft owns its VDE, Architecture, and per-domain selections, so
Proposal B never inherits Proposal A.

## Domain editor

The editor addresses one `(scenario identity, domain)` pair at a time. Vehicle
Demand selects an already-persisted VDE snapshot and never exposes Mass, Tire,
Aero, or roadload inputs. Architecture exposes only ICE, MHEV, HEV, PHEV, and
BEV. Other domains select Effective Current or an in-session canonical Domain
Proposal; N/A is available only through canonical Architecture applicability.

Temporary proposals always start from Effective Current, have stable domain
proposal identities, and are reusable across scenario slots without mutation.
Confirmed legacy fields are shown where present. Sparse configuration remains
explicitly sparse. Battery, Transmission, or other configuration changes do
not imply a numerical benefit when Energy Balance L0 has no representation for
them.

## Canonical call chain and ownership

```text
Powertrain_Scenario.py
    -> render_system_scenario_workspace()
    -> ScenarioDraft + ScenarioSource
    -> legacy/current domain adapters
    -> resolve_effective_domain_state()
    -> resolve_domain_proposal() when selected
    -> SystemScenarioDefinition + FuelEstimateRequest template
    -> run_system_scenario()
    -> resolve_system_scenario()
    -> EnergyBalanceL0Adapter
    -> run_fuel_estimation()
    -> canonical Technology Delta owner when represented
    -> SystemScenarioResult
```

The added Vehicle Demand adapter freezes the persisted TOTAL/NET snapshot into
the approved Vehicle Demand result contract. It does not call Vehicle Demand
physics. The UI owns no fuel, energy, PSE, PHEV, BEV, or delta formula. The
Fuel Consumption repository projection was widened only to expose columns
that already existed in the unchanged schema and are consumed by canonical
legacy domain adapters.

## Architecture and applicability UX

Changing Architecture immediately normalizes now-inapplicable selections to
N/A and restores newly-required domains to Effective Current. N/A domains are
absent from the canonical composition. A deliberately retained incompatible
proposal is still passed to the resolver so it becomes a structured
incompatibility rather than a silent correction. No component topology is
inferred.

## Fidelity UX

Every calculated result renders the canonical Fidelity Manifest in a compact
table using Quantitative, Effective assumption, Configuration only, and Not
represented. Proposal editing also states whether a physical configuration
change has an adopted L0 representation. This makes configuration-only
Battery and Transmission changes visible without implying a calculated
impact; sparse Electric Drive configuration and its aggregate electric-path
assumption are presented separately.

## Readiness and issue UX

Scenario cards display canonical READY/NOT READY independently. Missing
non-L0 configuration is summarized as metadata incomplete and does not change
readiness. Resolver issue codes are mapped to concise engineer-facing text;
the raw structured issues remain available only in the technical trace.
Programming/schema exceptions are isolated to their scenario card, while
ordinary unresolved inputs remain canonical NOT READY results. One invalid
Proposal never prevents valid siblings from producing results.

## Recommendation, adoption, and Technology Delta UX

Current observed, Benchmark, ML, Regression, and engineering evidence may be
associated with a recommendation, but it enters deterministic L0 only after
the explicit adoption checkbox. Adopted provenance is carried as observed,
estimated, ML-derived, or assumed through the Domain Proposal and canonical
resolver. An unadopted value is excluded from the physical fingerprint and
result.

The editor can associate one canonical `TechDeltaAssumption` with an in-session
Domain Proposal. The UI collects the source, effect basis, and value, then
delegates ordering/stacking to the existing canonical owner. It performs no
delta arithmetic. The single-entry UI is a deliberately compact 11D surface;
the contract and core continue to support multiple ordered deltas.

## Stale and recalculation behavior

Each result stores a fingerprint of canonical scenario identity, physical
slots, proposal content, source snapshot, and request. Display labels are
excluded. Any physical edit after calculation marks only the affected
scenario `Needs recalculation`; the old metrics are withheld. The next single
Calculate action replaces the result under the same stable scenario identity,
so reruns cannot create duplicate result slots.

## Legacy blocks retained, migrated, and retired

- Retained, opt-in: baseline, Benchmark, ML, Regression, manual recommendation,
  Technology Delta evidence tools, and legacy saved-estimate inspection and
  management.
- Migrated: active VDE discovery and existing service/repository access feed
  ScenarioSource/domain adapters; the canonical System Scenario pipeline owns
  the default result.
- Retired from the default path: the three-step method-centric navigation,
  per-method result ownership, and default legacy save panel.
- Not deleted: existing services, estimators, canonical delta behavior, and
  legacy persistence operations.

This classification avoided the stop condition for deleting an existing
valid capability with no canonical replacement.

## Save and persistence status

No System Scenario save was added. The existing Fuel Consumption row cannot
truthfully represent an eight-domain `SystemScenarioResult`, its Fidelity
Manifest, structured readiness, and proposal composition. Legacy saved
estimates therefore remain explicitly legacy and opt-in. Designing truthful
System Scenario persistence is deferred; no ad-hoc JSON or schema migration
was introduced.

## Automated coverage

The new Streamlit-free suite contains 21 direct tests and the live page has 7
AppTests. Together they cover all Sprint 11D state/UI acceptance cases,
including Current-only and bounded proposals, stable identities, independent
VDEs, proposal isolation/reuse, Architecture applicability, BEV N/A and
NOT READY behavior, sparse electric configuration, configuration-only Battery
and Transmission semantics, canonical Technology Delta representation,
metadata/readiness separation, partial failure, stale/recalculation,
recommendation/adoption provenance, friendly issues, explicit zero, and
canonical delegation.

Key delegation evidence:

- `test_snapshot_adapter_does_not_recalculate_vehicle_demand`;
- `test_ui_orchestration_delegates_once_per_scenario`;
- `test_battery_configuration_only_change_does_not_change_l0`;
- `test_unadopted_recommendation_does_not_change_fingerprint_or_result`;
- `test_adopted_ml_recommendation_flows_with_provenance`;
- existing Sprint 11C canonical ICE, BEV, PHEV, PSE, and Technology Delta
  tests remain the numerical owners; 11D tests do not duplicate formulas.

Fresh test evidence at closure:

- focused new suites: 28 tests, all passing;
- affected System Scenario/Fuel Estimation/Technology Delta/Powertrain suites:
  247 tests, all passing;
- full suite: 1,797 tests in 847.294 s, with the same one failure and one
  error already present in the pre-11D baseline and zero new regressions.

The unchanged baseline occurrences are:

- failure: `test_axle_hubs_lookup_snapshot_preserves_boundary_metadata`;
- error: `test_component_lookup_provenance_does_not_change_parasitic_math`.

## Developer smoke and 11E preparation

AppTest is recorded only as automated UI coverage, never as manual smoke. A
local Streamlit server was started for a separate developer browser pass, but
the in-app browser connection could not initialize in this session; the
server was then stopped. Consequently no manual/browser smoke is claimed for
11D. The UI leaves the requested 11E scenarios constructible: Current ICE,
different-VDE Transmission proposal, BEV, PHEV, configuration-only Battery,
Technology Delta proposal, and Current plus three independent Proposals.

## Deferred to Sprint 11E or later

- final manual/browser acceptance matrix and visual polish;
- SystemScenarioResult persistence design;
- Comparison integration and scorecards;
- persistent Domain Proposal libraries;
- multi-delta editing beyond the compact single-association 11D surface;
- topology/component graphs and higher-fidelity models;
- any PHEV CO2 semantic redesign.

Sprint 11E was not started. The pre-existing `.claude/` content was left
untouched.
