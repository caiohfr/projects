# Sprint 10D - Quick PSE / ML Recommendation / Technology Delta

## Status

**Efficiency Quick delivered: Current/Benchmark PSE, ML-derived PSE
recommendation, Technology Delta suggestion, and the Final-PSE-driven
deterministic fuel/energy result. No Vehicle Demand, VDE resolver, or ML
model physics modified. No Comparison UI. No persistence.**

Branch: `sprint-10-interactive-quick-scenarios`, building on Sprint 10A
(`83f93513`), 10B (`fdecba3c`), the EPA mass persistence hotfix
(`f6b183f9`), the Mass/Aero parity closure (`69b85339`), and Sprint 10C
(`451427f8`). Vehicle Quick (Mass -> Tire -> Aero) is complete; 10D adds
the Efficiency side:

```text
VehicleDemandResult (Sprint 10B/10C, unmodified)
        |
        v
Current PSE / Benchmark PSE / ML recommendation / Technology Delta suggestion
        |                                    (all advisory -- Sec 5)
        v
Final PSE  (QuickScenario.final_pse_percent + pse_provenance, Sprint 10A)
        |
        v
existing deterministic fuel/energy calculation (fuel_estimation.run_fuel_estimation)
        |
        v
Temporary Quick Efficiency result (QuickEfficiencyResolution)
```

## Scope

Implemented: Current PSE (source scenario's own achieved PSE), Benchmark
PSE (another FuelCons row's achieved PSE, rebasing nothing else), ML-derived
PSE recommendation with feature-visibility/coverage/confidence metadata,
Technology Delta suggestion (up to 3, existing canonical stacking math,
advisory only), and the Final-PSE-driven deterministic result via the
existing `run_fuel_estimation()` service, all as core/service/contract code
with focused tests -- no Streamlit, no Comparison UI wiring, no
session-state max-3 behavior (deferred to a UI package), no persistence.

Not implemented (explicit non-goals, unchanged): regen/SOC/battery/hybrid
supervisory logic, a new fuel/LHV constant, any Vehicle Demand Core change,
a Technology DB admin UI, ML retraining.

---

## Files changed

Created:
- `src/vde_core/technology_delta.py` -- extracted, Streamlit-free
  Technology Delta vocabulary and stacking math (Decision 1 below).
- `src/vde_core/quick_scenario/efficiency_resolution.py` -- the Efficiency
  Quick *output* contract (`PseReference`, `MlPseRecommendation`,
  `TechDeltaSuggestion`, `QuickEfficiencyResolution`).
- `src/vde_core/quick_scenario/efficiency_resolver.py` --
  `resolve_quick_efficiency_scenario()` and its private helpers.
- `src/vde_core/quick_scenario/tech_delta_catalog.py` --
  `load_quick_tech_delta_catalog()`, mapping `data/quick_tech_deltas.csv`
  onto `TechDeltaAssumption` directly.
- `data/quick_tech_deltas.csv` -- 8 synthetic, non-proprietary planning
  presets across engine/transmission/hybrid-ESS/electrical/auxiliary/
  calibration/thermal/whole-powertrain subsystems.
- `tests/test_technology_delta.py` (20 tests), `tests/test_pwt_fuel_energy_service_pse_reference.py`
  (13 tests), `tests/test_quick_scenario_efficiency_resolver.py` (42 tests),
  `tests/test_quick_tech_delta_catalog.py` (8 tests).
- This document.

Modified:
- `src/vde_core/pwt_fuel_energy_service.py` -- added `derive_reference_pse()`,
  `list_benchmark_fuelcons_candidates()`, `resolve_reference_fuel_type()`,
  `_load_json_blob()` (Streamlit-free extractions from `pwt_fuel_energy.py`,
  Decision 2 below).
- `src/vde_core/quick_scenario/contracts.py` -- added `TechDeltaAssumption`,
  `EfficiencyQuickInputs`, `MAX_TECH_DELTAS_PER_SCENARIO`; added
  `QuickScenario.efficiency_inputs` (sibling of `vehicle_overrides`, never
  nested inside it -- Sec 2).
- `src/vde_core/quick_scenario/serialization.py` -- round-trip support for
  the new contracts.
- `src/vde_core/quick_scenario/__init__.py` -- re-exports.
- `tests/test_quick_scenario_contracts.py` -- 15 new tests (56 -> 71).

**`src/vde_app/components/pwt_fuel_energy.py` was not modified.** Its own
Technology Delta workspace and benchmark-PSE flow continue to use their own
existing code unchanged -- zero behavior risk to the live Powertrain
Scenario page from this sprint.

---

## Decision 1 (Stop Condition 4 resolution): Technology Delta math extraction

Sprint 10A/10D audited `pwt_fuel_energy._apply_delta_stack_to_baseline`
(and its dependencies `_normalize_delta_effect_basis`, `_maturity_rank`,
`_delta_status_counts`, `_proposal_confidence_label`, and the four
`DELTA_*_OPTIONS` constant lists) and confirmed they are logically pure
(operate only on their own arguments) but live exclusively in
`src/vde_app/components/pwt_fuel_energy.py`, which `import streamlit` at
module scope. A Streamlit-free Quick Scenario resolver cannot import that
module regardless of the target functions' own purity -- this is exactly
Sprint 10's Stop Condition 4 ("Technology Delta canonical math cannot be
reused without importing a large Streamlit/UI component into core").

Per the condition's own prescribed remedy ("prefer extracting the smallest
pure canonical calculation to a shared core module while preserving exact
behavior"), `src/vde_core/technology_delta.py` reproduces every function
**verbatim** -- copied and independently re-verified against the live
source (not from memory; two functions initially transcribed from an
audit summary were caught wrong and corrected against direct source
reads before landing: `delta_status_counts`'s default-bucket behavior and
`proposal_confidence_label`'s full body). The one intentional,
behavior-neutral simplification: `baseline["method"]`/`proposal["method"]`
store the raw `FuelEstimateResult.method` string instead of the original's
UI display-label decoration (`_pwt_method_label`, whose only Streamlit
coupling is an `st.session_state` fallback unreachable for any real result)
-- this field is never read by the numeric stacking logic itself, so no
stacking outcome changes.

**Verified byte-for-byte parity**: every existing single-delta test case in
`tests/test_powertrain_scenario_deltas.py` (registered-only, manual fuel
delta, manual PSE delta, PSE-percent delta) is reproduced exactly in
`tests/test_technology_delta.py`'s `SingleDeltaParityWithExistingBehaviorTests`,
confirmed passing against the extracted module before any new behavior was
added. New multi-delta coverage (Sec 15/27, a real gap Sprint 10A had
already flagged) confirms and preserves, rather than "fixes": two percent
deltas compound (`base * 1.05 * 1.05`, not `base * 1.10`); absolute-then-
percent vs. percent-then-absolute give different results (order matters);
and a `co2_delta` stacked alongside any PSE/fuel-affecting delta is
overwritten by the unconditional post-loop fuel-to-CO2 reconciliation --
an existing quirk, documented and locked in as a regression test, not
altered.

`pwt_fuel_energy.py` is untouched and still uses its own original
functions for the live Powertrain Scenario page.

## Decision 2: Current/Benchmark PSE extraction

`pwt_fuel_energy._derive_reference_pse` (a donor `fuelcons_db` row's own
PSE, from its own linked VDE demand and its own recorded consumption) and
`_reference_candidates_for_type(vde_id, "Another fuelcons_db line")` were
both confirmed logically pure (no Streamlit/session-state reads) but likewise
only defined in the Streamlit-importing component file. Extracted verbatim
into `src/vde_core/pwt_fuel_energy_service.py` (the existing Streamlit-free
service module that already owns `resolve_vde_energy_values`/
`build_fuel_estimate_request_from_vde`) as `derive_reference_pse()` and
`list_benchmark_fuelcons_candidates()`, plus their small helper dependencies
`resolve_reference_fuel_type()` (public, since Sprint 10D's own "Current
PSE"/deterministic-calc powertrain-context resolution needs it too, not
just donor rows) and `_load_json_blob()`.

**Current PSE and Benchmark PSE are the exact same computation**, applied
to different rows: Current points `derive_reference_pse()` at the Quick
Scenario's own source `fuelcons_db` row; Benchmark points it at a donor
row selected by `EfficiencyQuickInputs.benchmark_source_identity`. This
satisfies Sec 8/9's "do not implement a second benchmark-PSE formula" and
avoids inventing a third formula for "current" PSE as well -- confirmed by
direct code reuse, not merely numerical coincidence.

`_build_observed_reference_request` (the function that would inject a
donor's PSE into a full rebase request) was audited and found **not**
extractable as-is: it reads 12+ `st.session_state` keys for powertrain
metadata (gear count, transmission model, fuel type overrides, etc.)
unrelated to the PSE injection itself. Sprint 10D does not need this
function -- `efficiency_resolver.py`'s own `_build_fuel_estimate_request()`
builds `powertrain_features` directly from the active scenario's own
`fuelcons_db` row (never a donor's), matching Sec 9's explicit "only
transfer derived/reference PSE... never donor roadload/VDE/transmission/
electrification/fuel type" rule structurally, not just by convention.

---

## PSE semantics

Reused verbatim, no new formula: `powertrain_efficiency.build_powertrain_efficiency_summary()`'s
`pse = demand_mj_per_km / total_consumed_mj_per_km`, invoked exactly once,
inside `fuel_estimation.run_fuel_estimation()`, for every method
(`physics_simple`, `ml_prediction`) this sprint uses. "Current"/"Benchmark"
PSE use the separate but equally canonical `derive_reference_pse()`
computation (Decision 2) since those don't go through a full
`FuelEstimateRequest` at all -- they read a row's own recorded consumption
directly. Both computations agree with each other by construction: 
`derive_reference_pse()`'s `demand / consumed` and `build_powertrain_efficiency_summary()`'s
`demand_mj_per_km / total_consumed_mj_per_km` are the same ratio, just
reached from a stored-consumption row vs. a live `FuelEstimateResult`.

**Final PSE is the sole calculation authority** (Sec 4/5): `QuickScenario.final_pse_percent`/
`pse_provenance` (Sprint 10A, unmodified) are read directly by
`efficiency_resolver.py`'s `_resolve_fuel_estimate_result()`; nothing this
sprint adds ever writes to those fields -- adopting a reference/
recommendation is the caller's job of constructing a new `QuickScenario`
with that value and the matching `*_ACCEPTED` provenance (confirmed by test:
`test_manual_edit_after_ml_acceptance_provenance_becomes_user_provided`
and the equivalent Benchmark/Tech-Delta tests all pass without the resolver
touching `final_pse_percent` itself).

**Zero is explicit, not blank** (Sec 7): `final_pse_percent = 0.0` (or
negative) returns `DomainReadiness.INVALID` with an explicit issue message
before any division is attempted -- confirmed never reaching
`run_fuel_estimation` in that case. No canonical PSE range validator was
found to reuse (the existing `demand/consumed` formula has no upper/lower
bound check at all), so none was invented; only the division-by-zero/
non-positive case is guarded, per Sec 7's explicit instruction not to
invent a competing validation contract.

---

## Benchmark PSE

`EfficiencyQuickInputs.benchmark_source_identity` names a donor scenario by
its **full Comparison identity** (`fc:<fuelcons_id>`, never a bare
`vde_id`), mirroring `QuickScenario.source_identity`'s own rule. The
resolver fetches that donor's `fuelcons_db` row, computes its own PSE via
`derive_reference_pse()`, and exposes only `{status, value_percent,
donor_source_identity, warnings}` on `PseReference` -- confirmed by test
(`test_benchmark_only_transfers_pse_and_provenance`) that the dataclass has
no field capable of carrying donor roadload/VDE/transmission/
electrification/fuel/regen/metadata. The active Quick Vehicle's own demand
and powertrain context are never touched by selecting a benchmark
(`test_active_vehicle_demand_remains_unchanged_with_benchmark_selected`).
Two donor rows sharing one `vde_id` (the QA fixture's `fc:900102`/`fc:900104`,
both `vde_id=900001`) remain distinguishable by their own `fc:` identity
(`test_two_donor_scenarios_sharing_one_vde_remain_distinct`).

---

## ML-derived PSE recommendation

Reuses the canonical inference path exactly: `run_fuel_estimation(method="ml_prediction")`
-> `ml_prediction.predict_fuel_with_ml()` (unmodified) -> the same
`build_powertrain_efficiency_summary()` every other method uses. Confirmed
by test (`test_pse_recommendation_is_derived_not_a_direct_model_output`)
that the recommended PSE equals `demand / ((predicted_fuel_l_100km/100) * LHV)`
exactly -- the model's own output is never reinterpreted as a percentage.
No retraining, no runtime notebook execution, no second PSE model.

`MlPseRecommendation` exposes `artifact_status`/`model_version`/
`coverage_status`/`missing_features` read directly from
`FuelEstimateResult.assumptions` (`integration_status`, `model_version`,
`coverage_status`, `missing_features` -- the exact keys
`predict_fuel_with_ml()` populates, confirmed by direct source read, not
invented names) and `confidence_label` from `FuelEstimateResult.confidence`
(the existing categorical `high`/`medium`/`low`/`provided` vocabulary --
**no numeric confidence score exists on the dataclass**, confirmed by test).

**Feature-visibility honesty** (Sec 11, the single most important ML
constraint from the Sprint 10A audit): `quick_affected_features_changed`
is derived from which Vehicle Quick domain was actually requested, mapped
against the audited data-flow facts --

| Requested domain | ML features it can actually change |
|---|---|
| Mass | `vde_net_mj_per_km` only (mass changes `test_mass_kg`, not ABC) |
| Tire | `coast_A_N`, `coast_B_N_per_kph`, `coast_C_N_per_kph2`, `vde_net_mj_per_km` (the canonical Tire resolver's `tire_delta_abc` can touch any ABC term) |
| Aero (CdA) | `coast_C_N_per_kph2`, `vde_net_mj_per_km` |

`features_not_represented` always includes `vde_urb_mj_per_km`/
`vde_hw_mj_per_km`, regardless of what changed -- `QuickVehicleResolution`
never exposes per-phase VDE (only whole-cycle `vde_total`/`net_mj_per_km`),
so Quick can never honestly claim to have recomputed those two ML features
(confirmed by
`test_quick_vehicle_changes_not_represented_are_never_falsely_reported`).
A scenario with no Vehicle override produces an empty
`quick_affected_features_changed` tuple
(`test_no_vehicle_change_produces_no_affected_ml_features`).

Failure handling (Sec 12): artifact-unavailable, load-failure, and
missing-feature states all resolve to `MlPseRecommendation(status="unavailable", ...)`
with the real `artifact_status` string surfaced -- never a crash, and never
blocking the (already independently valid) Vehicle Quick result or a
manually-supplied Final PSE, confirmed by
`test_artifact_unavailable_ml_recommendation_unavailable_no_crash`.

Tests use an injected stub predictor (`FuelEstimateRequest.model_options["ml_predictor"]`,
the same injection point `run_fuel_estimation`'s own test suite uses) rather
than depending on the real ~16.6 MB artifact's exact trained output --
confirmed separately, once, that the real artifact loads and produces a
plausible result end-to-end during development, but the committed test
suite does not depend on its specific numeric behavior.

---

## Technology Delta

Reuses `technology_delta.apply_delta_stack_to_baseline()` (Decision 1)
against a baseline built from **Current PSE** (never Final PSE, never a
donor's PSE) -- the suggestion is always "what would Current PSE become
with these assumptions," independent of whatever Final PSE the user may
have already set. **Advisory only** (Sec 14): `TechDeltaSuggestion` is a
read-only field on `QuickEfficiencyResolution`; nothing in
`efficiency_resolver.py` ever assigns to `final_pse_percent`. Confirmed by
test: a scenario with an unadopted Tech Delta suggestion present produces
byte-identical `fuel_estimate_result.fuel_l_100km` to the same scenario
with no Tech Delta at all
(`test_without_explicit_adoption_final_pse_and_energy_result_do_not_change`).

**Capped at 3** (`MAX_TECH_DELTAS_PER_SCENARIO`, Sec 15, "a product
complexity limit, not a new physics rule") via
`EfficiencyQuickInputs.__post_init__` -- rejected at construction, before
the resolver ever runs, so a caller cannot accidentally exceed it. Stacking
order is caller-supplied-list order, sequential, exactly as
`apply_delta_stack_to_baseline` implements it (see Decision 1's
compounding-vs-additive table); a 2%+1% pair of `pse_percent_delta`
assumptions compounds to `+3.02%`, not `+3.00%`, confirmed by test.

---

## Tech preset catalog

`data/quick_tech_deltas.csv` (8 synthetic, non-proprietary planning rows,
one per subsystem category from Sec 16) is read by
`load_quick_tech_delta_catalog()` directly into
`{tech_id: TechDeltaAssumption}` -- no second schema, no admin UI, no
database. A missing catalog file returns an empty mapping rather than
raising (the catalog is a convenience, not a requirement -- a caller can
always construct a custom `TechDeltaAssumption` directly, confirmed by
`CustomTechDeltaMapsToCanonicalContractTests`, which also confirms
`effect_value` has no default -- a `TypeError` on construction, per Sec 17's
"no hidden default magnitude").

---

## Deterministic energy integration

`efficiency_resolver._build_fuel_estimate_request()` builds one
`FuelEstimateRequest` per call (Current/Benchmark/ML/Final-PSE calculations
each build their own, varying only `method` and the `eta_pt_est`/
`bev_eff_drive` powertrain override), always reading `fuel_type`/`LHV_MJ_per_L`/
`gCO2_per_L` from the existing `fuel_energy.LHV_MJ_PER_L`/`GCO2_PER_L`
tables (confirmed by test: the deterministic result's `fuel_l_100km`/
`gco2_km` match an independently-computed value using those exact
constants) -- **no new LHV or CO2 constant was introduced anywhere in this
sprint**. `vde_total_mj_per_km`/`vde_net_mj_per_km` fed into every request
come uniformly from the Quick-resolved `VehicleDemandResult.total_summary`/
`net_summary.vde_mj_per_km` (the frozen Sprint 9 core's own output) --
never from `QuickVehicleResolution`'s separate legacy `vde_total_mj_per_km`/
`vde_net_mj_per_km` audit fields (Sprint 10B/10C's own cross-path parity
fields, which exist to prove Vehicle Quick reuse, not to feed further
computation). This keeps every Efficiency Quick number -- deterministic
result, ML feature, Tech Delta baseline -- internally consistent with each
other and with "VehicleDemandResult" as the single authoritative demand
Sec 1's diagram names.

## TOTAL / NET

`energy_basis: RoadloadBasis` (the same enum `vehicle_demand` already
defines -- no third TOTAL/NET vocabulary) selects `VDE_TOTAL`/`VDE_NET` at
the `FuelEstimateRequest` boundary. Confirmed by test that TOTAL and NET
produce different `fuel_l_100km` values for the same scenario (no silent
convergence), and that a scenario whose Quick-resolved NET boundary is
unavailable (no transmission coefficients on the row) returns
`DomainReadiness.MISSING` for a NET-basis Efficiency request rather than
falling back to TOTAL.

## Vehicle Quick / Efficiency Quick independence

Confirmed by test matrix: Current/Benchmark PSE are computed regardless of
whether the Quick Vehicle result succeeded (they only need the source/donor
`fuelcons_db` row); ML recommendation, Tech Delta suggestion, and the
deterministic result all require `vehicle_resolution.vehicle_demand_result`
and report `unavailable`/`MISSING` cleanly, never crash, when it's absent;
supplying a manually-chosen Final PSE produces a `READY` deterministic
result independent of whether ML/Benchmark/Tech-Delta references were ever
computed or available at all.

## Immutability / no DB writes

`test_source_fuelcons_row_remains_unchanged` re-fetches the source
`fuelcons_db` row after resolution and confirms byte-for-byte equality;
`test_no_db_writes` confirms `fuelcons_db`/`vde_db` row counts are
unchanged after a full resolution including a Benchmark selection and a
Technology Delta suggestion; `test_vehicle_quick_resolution_object_is_not_mutated`
confirms the `QuickVehicleResolution` passed in is never altered by the
Efficiency resolver.

---

## Regen (out of scope, confirmed)

No regen capture, recovered-energy, SOC, battery, or hybrid supervisory
concept was introduced. `VehicleDemandResult`'s existing wheel-side braking-
energy figures (frozen Sprint 9 core) are not read, re-labeled, or
otherwise touched by any code in this sprint.

---

## Tests

Pre-change baseline (matches the user-supplied historical count exactly,
confirmed by a fresh run before any 10D edit): **1504 tests, 1502 passing**,
2 known pre-existing failures (`test_component_lookup_provenance_does_not_change_parasitic_math`,
`test_axle_hubs_lookup_snapshot_preserves_boundary_metadata`, both in
`tests/test_vde_request_resolver.py`, unrelated to Quick Scenario, untouched
by any Sprint 10 package).

Focused Sprint 10D suite (`test_quick_scenario_*`, `test_technology_delta`,
`test_pwt_fuel_energy_service_pse_reference`, `test_quick_tech_delta_catalog`,
`test_fuel_estimation`, `test_ml_prediction`, `test_powertrain_scenario_deltas`):
**226 tests, all passing.**

Broader focused suite (adds Mass/Tire/Aero/VDE-save/Vehicle-Demand/
Comparison suites on top of the above): **501 tests**, same 2 known
pre-existing failures, zero new regressions.

Full suite (`python -m unittest discover -s tests`) after 10D: **1597
tests, 1595 passing**, the same 2 known pre-existing failures (confirmed by
the exact `failures=1, errors=1` count matching every other full-suite run
this sprint sequence, plus a separate focused run -- 501 tests -- that
directly showed the same 2 named failures with full tracebacks), **zero new
regressions**. New test files/counts:
`test_quick_scenario_contracts.py` grew to 71 (from 56 in Sprint 10C),
`test_technology_delta.py` (20, new), `test_pwt_fuel_energy_service_pse_reference.py`
(13, new), `test_quick_scenario_efficiency_resolver.py` (42, new),
`test_quick_tech_delta_catalog.py` (8, new).

## Backlog / deferred (not addressed in 10D)

- Quick Scenario UI, Comparison insertion, session-state max-3 enforcement,
  Save/Promote -- all explicitly deferred to a later (10E+) package per
  Sec 29.
- `pwt_fuel_energy.py`'s own Technology Delta/benchmark-PSE code was left
  untouched and still duplicates (not reuses) the newly-extracted
  `vde_core` versions; unifying them (having the Streamlit page import
  from `vde_core` instead of keeping its own copies) is a reasonable
  future cleanup but was out of scope and risk-inappropriate for this
  sprint, which must not touch a live, working UI page.
- The pre-existing CO2-delta-gets-overwritten-by-fuel-reconciliation quirk
  (Decision 1) is preserved, documented, and regression-tested, not fixed
  -- fixing it, if ever desired, is a Powertrain Scenario concern, not a
  Quick Scenario one.

## Freeze / handoff statement

```text
Vehicle Demand Core API                    FROZEN (untouched)
ML artifact / inference contract           UNTOUCHED
pwt_fuel_energy.py (Powertrain Scenario UI) UNTOUCHED
Quick Scenario Vehicle (Mass/Tire/Aero)     DELIVERED (10B/10C)
Quick Scenario Efficiency (PSE/ML/Tech Delta) DELIVERED (10D)
Quick Scenario UI / Comparison / Save       NOT STARTED
```

10E was not started.
