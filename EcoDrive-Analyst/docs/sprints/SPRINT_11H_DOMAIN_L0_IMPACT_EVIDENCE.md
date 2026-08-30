# Sprint 11H — Domain L0 impact, Current corrections and evidence integrity

## Scope and boundary

Sprint 11H connects a System Scenario's domain selections to the existing
Energy Balance L0 request without adding a physical formula, database schema,
persistence, Comparison integration, or Sprint 12 work. The canonical owner of
fuel, electric energy, CO2 and Technology Delta stacking remains unchanged.

## Configuration versus quantitative representation

A Domain Proposal has two independent parts:

- **Configuration** says what physically changed.
- **L0 representation** is an optional, explicitly adopted canonical direct
  assumption or Technology Delta.

Configuration alone remains `CONFIGURATION_ONLY`: it does not infer a fuel,
PSE, efficiency, or energy effect. The UI displays the Effective Current value
next to each editable proposal configuration field, then renders L0
representation separately.

The resolver continues to use the existing direct request keys:
`eta_pt_est`, `bev_eff_drive`, `utility_factor`, and
`grid_gco2_per_kwh`. `eta_pt_est` is retained as the aggregate fuel/system-path
assumption from the canonical `FuelEstimateRequest`; it is not relabelled as
an engine component efficiency.

## Current correction

Current correction is a source-scoped, session-only mapping keyed by `(VDE,
domain)`, distinct from `DomainProposal`. It constructs the existing typed
`DomainCorrection` and produces an `EffectiveDomainState`. The source record
remains immutable. A correction may replace full typed configuration fields or
provide an explicit direct L0 assumption, with a correction/evidence note.

All drafts using the same VDE resolve through that correction. A new Domain
Proposal is built from the resulting Effective Current, while the correction
itself is traceable as `CURRENT_CORRECTION` in L0 assumption provenance.

## System metric bridge

The UI does not calculate PSE or efficiency. Results and the technical trace
are taken from `SystemScenarioResult` and its canonical Energy Balance L0
execution. Supported Technology Delta bases offered by this workspace remain
`pse_percent_delta` and `fuel_percent_delta`; both are passed unchanged to the
canonical Technology Delta owner. `energy_percent_delta` is absent because the
resolver does not support it.

## Architecture treatment

- ICE, MHEV and HEV retain the existing aggregate request semantics.
- BEV uses the existing `bev_eff_drive` direct assumption. A motor-power-only
  change is configuration-only.
- PHEV direct assumptions are restricted in the editor to their explicit
  paths: Engine → `eta_pt_est`, Electric Drive → `bev_eff_drive`, Controls →
  `utility_factor`. Generic Technology Delta association is unavailable for
  PHEV because the present contract does not declare its thermal/electric path.
  No generic delta is silently applied to both paths.

## Evidence and adoption

Manual numeric entry is always an **Engineering assumption** and receives
`ASSUMED` provenance after explicit adoption. The former decorative
ML/Benchmark/Regression source selector was removed. Programmatic attempts to
adopt a manual value while claiming any of those sources raise an error.

There is currently no System Scenario owner that emits an L0-domain
recommendation with a model, benchmark, or regression evidence reference.
Therefore this workspace does not claim ML-derived, benchmark-derived, or
regression-derived L0 impacts. Existing ML and benchmark features remain in
their own surfaces; wiring an actual canonical recommendation contract is
deferred rather than fabricating provenance.

Recommendations remain separate from adopted state: an unadopted manual
recommendation adds no resolved L0 assumption and does not alter the
calculation fingerprint or result.

## Utility factor and grid CO2

The Controls UI labels Utility Factor in percent and validates `0–100`; the
viewmodel converts it once to the `0..1` canonical request fraction. The
technical trace retains the canonical fraction.

`grid_gco2_per_kwh` is a supported request key but no grid default is supplied
by this workspace. Current request summaries display **Not provided** when it
is absent. For BEV, the result surface displays grid CO2 as **Not evaluated**
rather than presenting the canonical numeric zero as known zero emissions. For
PHEV the existing canonical top-level CO2 output is fuel-path-only when grid
is absent; that limitation is disclosed rather than reinterpreting its
numerics. Reworking that result contract is out of scope.

## Shared Domain Proposal characterization

`WorkingSetTests.test_shared_proposal_remains_bound_to_its_original_effective_current`
uses two different Engine Effective Current states and one shared proposal. It
proves the selected proposal retains the original `based_on` object and its
Gasoline configuration when used by the second scenario; it is not rebuilt or
patched over that scenario's Diesel Current. This sprint preserves that
baseline-bound behavior and does not make a persistence or permanent shared
proposal identity decision.

## Verification

Focused coverage includes:

- source immutability, Current correction propagation and proposal inheritance;
- configuration-only invariance already established for Engine, Transmission,
  Energy Storage and Electric Drive;
- direct thermal/electric impacts and canonical-result parity;
- manual-provenance rejection for ML, Benchmark and Regression claims;
- unadopted recommendation inertness;
- Utility Factor fraction semantics;
- supported Technology Delta ordering and basis filtering;
- shared-proposal characterization;
- AppTest reachability of the Current correction and absence of the false
  provenance selector.

Manual browser smoke is not claimed: browser integration is unavailable in
this environment. AppTest is automated UI coverage, not a browser substitute.

## Deferred work

Actual ML/Benchmark/Regression domain-L0 recommendations require a canonical
owner and evidence-reference contract. Grid CO2 interpretation for PHEV and
any generic PHEV Technology Delta path assignment require an explicit future
semantic decision. No such decision was made here.
