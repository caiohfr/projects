# Sprint 10A - Quick Scenario Contracts & Canonical Reuse Audit

## Status

**AUDIT COMPLETE. Minimal contracts delivered. No Vehicle Demand, VDE,
Powertrain, or ML physics modified.**

Branch: `sprint-10-interactive-quick-scenarios`, forked from the tip of
`sprint-9a-vehicle-demand-contracts` (the repository's actual trunk for this
sprint sequence -- the literal git `main` ref predates Sprint 8/9 and was not
used as the fork point; see "Branch note" below).

Sprint 10's product goal: take one existing resolved Comparison scenario,
apply a small set of temporary engineering assumptions (Vehicle: Mass/Tire/
CdA; Efficiency: PSE), recalculate with the **existing** canonical EcoDrive
engines, and compare the result without saving new DB rows. 10A's job is
narrower: audit where each of those operations is already canonically
implemented, and land the minimum Streamlit-independent contracts a later
package needs -- **no Quick Scenario resolver, no UI, no new physics.**

### Branch note

Git's `main` ref (`828da134`) predates Sprint 8 and Sprint 9 entirely --
neither sprint was ever merged back to `main` via PR; each sprint instead
built on the previous sprint's branch tip (`sprint-7-database-management` ->
`package-7g-vde-net-total-contract` -> `sprint-8-comparison-report` ->
`sprint-9a-vehicle-demand-contracts`). Sprint 9's own closure document
(`docs/sprints/SPRINT_9_VEHICLE_DEMAND_CLOSURE.md`) lives only on that
branch and explicitly hands off to "Sprint 10 - Interactive Quick Scenario"
as the next step. Branching Sprint 10 from literal `main` would have
discarded the frozen Vehicle Demand Core and every canonical resolver this
audit depends on, so `sprint-10-interactive-quick-scenarios` was created
from the `sprint-9a-vehicle-demand-contracts` tip instead, consistent with
the repository's established sprint-branch workflow.

## Baseline tests (before any Sprint 10A change)

Full suite, `python -m unittest discover -s tests`:

```text
1359 tests, 1357 passing, 2 known pre-existing failures.
```

Exactly matches the historical baseline recorded in the Sprint 9 closure
doc. The 2 failures are both in `tests/test_vde_request_resolver.py`,
unrelated to Sprint 10 (component-lookup/axle-hub branches of the *legacy*
VDE Request Resolver -- see Sec 2 below):

- `VdeRequestResolverTests.test_component_lookup_provenance_does_not_change_parasitic_math`
  -- **ERROR**, `TypeError: 'NoneType' object is not subscriptable` at
  `tests/test_vde_request_resolver.py:707`.
- `VdeRequestResolverTests.test_axle_hubs_lookup_snapshot_preserves_boundary_metadata`
  -- **FAIL**, `AssertionError: 120.0 != 120.5` at
  `tests/test_vde_request_resolver.py:718`.

Neither failure was touched by, or is related to, any 10A change.

## Post-change tests

Focused: `tests/test_quick_scenario_contracts.py` -- **48 tests, all
passing** (new file, Sprint 10A's only test-visible change).

Full suite after 10A: **1407 tests, 1405 passing, the same 2 known
pre-existing failures, zero new regressions** (1359 baseline + 48 new Quick
Scenario contract tests).

---

## 1. Capability audit table

`Reusable directly?` = the existing function/module can be called as-is by a
later Quick Scenario resolver package. `Adapter/extraction needed?` = a thin
wrapper or a new parameter is needed, but no existing formula changes.
`Physics change required?` = would require new or duplicated physics (none
found -- see Sec 6).

| Capability | Canonical owner today | Reusable directly? | Adapter/extraction needed? | Physics change required? | Quick Scenario strategy |
|---|---|---|---|---|---|
| Curb -> regulatory mass | `resolve_mass_proposal()`, `src/vde_core/vde_mass_proposal_resolver.py:20` | Yes | No | No | Call with `proposal_type` limited to the Sprint 10 subset (`EPA_CURB_TO_TWC`, `MASS_TWC_SHIFT`, `WLTP_MASS_LINE`, `PERFORMANCE_CURB_MASS`); consume `resolved["vde_calculation_mass_kg"]`/`resolved["tire_load_mass_used_kg"]` (the canonical mass-contract pair from `_populate_canonical_mass_state()`, lines 280-327) |
| EPA TWC (test weight class) | `EPA_CURB_TO_TWC` / `MASS_TWC_SHIFT` branches, `vde_mass_proposal_resolver.py:79-151`, bucket lookup in `test_mass.py:339` (`inertia_step_for_mass`) | Yes | No | No | Same call as above; TWC bracket is `(lower_exclusive, upper_inclusive]` -- a curb-only nudge inside one bracket legitimately leaves TWC unchanged (Sec 4.1's stated invariant, confirmed in code) |
| WLTP mass (TML/TMH) | `compute_wltp_test_masses()`, `src/vde_core/test_mass.py:114`, selected via `WLTP_MASS_LINE` in the mass resolver (`vde_mass_proposal_resolver.py:177-198`) | Yes | No | No | Call `resolve_mass_proposal(proposal_type="WLTP_MASS_LINE", ...)`; do not call `compute_wltp_test_masses` directly (keeps EPA/WLTP dispatch single-sourced) |
| CdA (Aero) | `_resolve_aero()`, `src/vde_core/vde_request_resolver.py:633-674`; the underlying delta->C conversion is `cdA_to_C()`, `src/vde_core/roadload/engine.py:194-198` | Partial -- `_resolve_aero` is a private, request-resolver-internal function | Yes -- extract/call `cdA_to_C()` directly (already public) the same way `_resolve_aero` does, or a thin adapter that mirrors its `AERO_DELTA_CDA`/`AERO_ABSOLUTE_CDA` branches | No | Quick Scenario CdA override resolves target CdA via the generic `ScalarChange` (Sec below), then converts the *delta* to C via `cdA_to_C()` -- never a second `0.5*rho*CdA/3.6**2` implementation |
| Current Tire -> Target RRC | `resolve_tire_proposal(proposal_type="TIRE_TARGET_RRC")`, `src/vde_core/vde_tire_proposal_resolver.py:24,164-183` | Yes | No | No | Call with `source=CURRENT` |
| Current Tire -> RRC Delta | Not a first-class `proposal_type` in the resolver; the delta->target conversion already exists one layer up, `vde_tire_modes.py:38-45` (`delta_RRC_optional` -> `target_rrc_N_per_kN`) | Partial | Yes -- pre-transform `source_rrc + delta` into `target_rrc_N_per_kN` before calling `resolve_tire_proposal(proposal_type="TIRE_TARGET_RRC")`, mirroring the existing `vde_tire_modes.py` legacy-alias pattern | No | Quick Scenario's `ScalarChange(DELTA)` resolves the target RRC value in the contract layer; the resolver call itself still uses `TIRE_TARGET_RRC` |
| Current Tire -> Improvement % | `resolve_tire_proposal(proposal_type="TIRE_IMPROVEMENT_PCT")`, `vde_tire_proposal_resolver.py:151-163` | Yes | No | No | Call directly; sign convention (`resolved = source * (1 - pct/100)`, positive = lower RR) is preserved as a **dedicated field**, never routed through the generic `ScalarChange` percent semantics (Sec 6/20C) |
| Current Tire -> Pressure Delta | `_estimate_pressure_only_rrc()` (heuristic, `vde_tire_proposal_resolver.py:214-256`) or `adjust_rrc_to_operating_condition()` (coefficient-driven, `src/vde_core/roadload/tire_model.py:375-416`), selected automatically by which tire-model data is available | Yes | No | No | Call `resolve_tire_proposal(proposal_type="TIRE_TARGET_RRC")` with pressure inputs and no `target_rrc_N_per_kN` (triggers the pressure-only path, line 170-183); supports independent front/rear |
| Tire DB lookup | `_lookup_tire_record()`, `vde_tire_proposal_resolver.py:439-453`, wrapping `get_tire_by_id`/`get_tire_by_code` in `src/vde_core/tire_roadload_service.py:430-435` | Yes | No | No | Call `resolve_tire_proposal(proposal_type="TIRE_DB_LOOKUP")` with `tire_db_id` |
| Tire DB + Improvement % | Same `TIRE_DB_LOOKUP` branch, improvement applied after lookup (`vde_tire_proposal_resolver.py:113-150`) | Yes | No | No | Same call, `improvement_pct` set |
| Tire DB + Pressure Delta | Same `TIRE_DB_LOOKUP` branch, ISO pressure re-estimate (`vde_tire_proposal_resolver.py:137-150`) | Yes | No | No | Same call, pressure inputs set |
| User-provided reference pressure | **Gap** -- reference pressure provenance (DB/source vs. user-provided) is not a stored field anywhere in the tire resolver; it is only implicit in *which proposal-type branch executed* (see Sec 3 below) | No -- nothing to reuse for the DB/user distinction itself | **Yes** -- smallest viable extension is a caller-side field (this is exactly what `TireQuickChange`/`TirePressureDelta.reference_pressure_provenance` in this package's contracts now carry) that a later resolver-integration package must thread into the tire resolver call (e.g. override `source["front_pressure_psi"]`/`source["rear_pressure_psi"]` before calling `resolve_tire_proposal`, tagged by provenance in the Quick Scenario layer, not inside the resolver itself) | No | Contract-level field added in 10A (`ReferencePressureProvenance`); resolver-side wiring deferred to the package that calls `resolve_tire_proposal` |
| `VehicleDemandRequest` construction | Two existing adapters: `vehicle_demand.adapters.build_vehicle_demand_request()` (raw `vde_db` row, `src/vde_core/vehicle_demand/adapters.py:62`) and `comparison_vehicle_demand_viewmodels._vehicle_demand_request_from_comparison_item()` (already-resolved `ComparisonItem`, `src/vde_app/comparison_vehicle_demand_viewmodels.py:114`) | Yes (either, depending on what a later package has in hand) | Possibly -- neither adapter currently accepts a Mass/CdA/RRC override; `test_mass_kg`/`cda_m2`/`rrc_n_per_kn` are plain scalar fields on `VehicleDemandRequest` itself, so a resolver-integration package can simply construct the request with resolved-override values substituted in, with no contract change | No | Out of scope for 10A (frozen core untouched); documented as the entry point the next package should target: `calculate_vehicle_demand(request, cycle_frame)`, `src/vde_core/vehicle_demand/engine.py:220` |
| Current PSE | `build_powertrain_efficiency_summary()`, `src/vde_core/powertrain_efficiency.py:109-206`, called from `run_fuel_estimation()`, `src/vde_core/fuel_estimation.py:483-490`, for every method | Yes | No | No | Reuse verbatim: `pse = demand_mj_per_km / total_consumed_mj_per_km` (line 162), method-agnostic |
| PSE from another FuelCons row | `_derive_reference_pse()` + `_build_observed_reference_request()`, `src/vde_app/components/pwt_fuel_energy.py:728-755,5496-5527` | Yes | No | No | Reuse: donor's own PSE is computed, then fed as `eta_pt_est`/`bev_eff_drive` through the same `physics_simple` engine against the *active* vehicle's demand -- no separate rebasing formula exists to duplicate |
| Manual Final PSE | **Gap** -- no "Final PSE" field/gate exists yet anywhere in the codebase; closest precedent is the baseline "Confirm baseline" snapshot pattern, `pwt_fuel_energy.py:3859-3874` | No (nothing to call) | **Yes** -- new, but the *pattern* to copy (session-state snapshot with deep-copied request/result, single explicit confirm action) already exists and should be mirrored, not reinvented | No | Delivered in 10A as the `QuickScenario.final_pse_percent` + `pse_provenance` pair (Sec 4 below); UI-level confirm gate is out of scope for 10A |
| ML-derived PSE recommendation | `predict_fuel_with_ml()` (`src/vde_core/ml_prediction.py:606-749`) -> `build_powertrain_efficiency_summary()` (PSE derivation, method-agnostic, same as "Current PSE" row) | Yes | No | No | Reuse `run_fuel_estimation(method="ml_prediction")` verbatim; PSE is **derived**, never a direct model output (confirmed: model predicts `fuel_l_100km`/`energy_Wh_km` only, `ml_prediction.py:186-210,261-264`) |
| Technology Delta -> PSE suggestion | `_apply_delta_stack_to_baseline()`, `pwt_fuel_energy.py:931-1043` | Yes | Presentation-only -- reuse the math, but 10A/Sprint-10 must add an explicit accept-before-Final-PSE gate, since the existing function currently applies live/automatically with no acceptance step (a UI/gating change, not a math change) | No | Reuse `_apply_delta_stack_to_baseline`'s sequential stacking (additive absolute deltas, compounding percent/multiplier deltas) verbatim in a later package; represent its output as a recommendation, gated behind `PseProvenance.TECH_DELTA_ACCEPTED`, in this package's contracts |
| Deterministic PSE -> fuel/energy result | `_physics_simple()`, `src/vde_core/fuel_estimation.py:177-244`, via `run_fuel_estimation()` | Yes | No | No | Reuse verbatim: `fuel_l_100km = (vde_mj_per_km / eta_pt) / lhv * 100.0`; canonical LHV/CO2 tables live only in `src/vde_core/fuel_energy.py` (`LHV_MJ_PER_L`, `GCO2_PER_L`) -- never redeclare |

---

## 2. VDE Mass / Tire / Aero resolvers (Vehicle Quick)

### Mass -- `src/vde_core/vde_mass_proposal_resolver.py`

`resolve_mass_proposal(source_snapshot, proposal_type, inputs) -> dict`
(line 20) is a pure function dispatched by `proposal_type`. Relevant to
Sprint 10: `EPA_CURB_TO_TWC` (line 79), `MASS_TWC_SHIFT`/`EPA_PLUS_1_TWC`
(line 128), `PERFORMANCE_CURB_MASS` (line 153, Absolute/preset-Delta curb
change), `WLTP_MASS_LINE` (line 177, delegates to
`compute_wltp_test_masses()` in `test_mass.py:114`).

**Correction to the task's stated assumption:** `GVWR` (line 199), `GCWR`
(line 223, full trailer-mass math), and `CUSTOM_MASS` (line 249, arbitrary
test mass) **already exist** in this same resolver and are exercised by
existing tests (`tests/test_vde_mass_proposal_resolver.py::test_gvwr_and_gcwr_compute_payload_and_vehicle_mass`,
`::test_gcwr_requires_complete_trailer_curve`). This does not conflict with
Sprint 10's scope -- Quick Scenario simply must not *expose* these
`proposal_type` values (per Sec 4.1's non-goals); the resolver having a
broader surface than Quick Scenario uses is not a physics conflict, just a
correction to avoid an incorrect "must be added later" assumption.

**Canonical mass contract published for downstream consumers**
(`_populate_canonical_mass_state()`, lines 280-327):
`resolved["vde_calculation_mass_kg"]` / `resolved["vde_mass_basis"]` (the
mass VDE/roadload should use) and `resolved["tire_load_mass_used_kg"]` /
`resolved["tire_load_mass_basis"]` (the mass tire/RRC should use). Under
EPA, `vde_calculation_mass_kg` is the TWC, not the physical test mass --
these are two different numbers, both published.

**Confirmed invariant** (Sec 4.1): a curb-mass-only nudge inside one EPA TWC
bracket (`(lower_exclusive, upper_inclusive]`, `test_mass.py:18-29`)
legitimately leaves `inertia_class` (TWC) unchanged, while
`test_mass_kg = curb + 136.0` (`EPA_TEST_MASS_DEFAULT_DELTA_KG`,
`vde_setup_service.py:99`) still moves linearly with curb mass unless an
explicit test-mass override is supplied. Both behaviors are correct and
must not be "corrected" by Quick Scenario, per the task spec.

### Tire -- `src/vde_core/vde_tire_proposal_resolver.py`

`resolve_tire_proposal(source_snapshot, proposal_type, inputs, *,
current_snapshot=None) -> dict` (line 24). Supports (canonicalized via
`canonical_tire_proposal_type`, `vde_tire_modes.py:8-12`):
`TIRE_METADATA_ONLY` (Not-used state), `TIRE_DB_LOOKUP`, `TIRE_IMPROVEMENT_PCT`,
`TIRE_TARGET_RRC` (also serves the pressure-only path when
`target_rrc_N_per_kN` is blank).

- **Improvement % sign convention confirmed**: `resolved_rrc = source_rrc *
  (1 - improvement_pct/100)` (line 157) -- positive improvement = lower RRC,
  test-locked (`test_positive_improvement_pct_lowers_resolved_rrc`).
- **RRC -> Tire ABC**: single-sourced through
  `calculate_vehicle_tire_abc()`, `src/vde_core/roadload/tire_model.py:635-713`
  (ISO/CUSTOM: `A = rr_n_per_kn * load_kN`; SAE: power-law). The tire
  resolver never reimplements this -- it calls the canonical function via
  `_calculate_with_tire_model()` (line 396) or a scalar-RRC stub via
  `_abc_from_rrc()` (line 575).
- **Pressure model**: two legitimate, coexisting mechanisms -- a simplified
  heuristic (`_estimate_pressure_only_rrc()`, line 214, sensitivity 0.30,
  clamped +/-10%) and a coefficient-driven power law
  (`adjust_rrc_to_operating_condition()`, `tire_model.py:375-416`), selected
  automatically by whether the tire record has real SAE reference data. Both
  support independent front/rear pressure. This is not accidental
  duplication -- it is a documented fallback/primary pair -- and Sprint 10
  must reuse whichever the existing proposal-type branch already selects,
  never add a third.
- **Reference pressure provenance gap** (Sec 6, flagged as a genuine gap):
  "source reference pressure" (from the vehicle snapshot,
  `resolved["tire_reference_front_pressure_psi"]`, lines 46-47/68) and
  "tire-DB reference pressure" (from the selected tire record's own test
  pressure, `_tire_reference_pressure_psi()`, lines 484-494) **write into
  the same output field name**, disambiguated only by which `proposal_type`
  branch ran -- there is no explicit stored `provenance` value distinguishing
  them today. Missing source/baseline reference pressure already fails hard
  and explicitly (`status: "Missing"`, lines 232-234); missing DB reference
  pressure already falls back softly with an explicit `Review` issue (lines
  127-135) -- neither silently defaults to a guessed number, which already
  satisfies "no silent pressure default." The smallest extension needed is
  exactly what this package's `TirePressureDelta.reference_pressure_provenance`
  /`reference_pressure_psi` fields now carry at the contract level; wiring
  that into the resolver call itself (so the resolver's *output* also
  carries the same explicit provenance tag) is left to the resolver-
  integration package, since it requires touching resolver call sites, not
  just a new contract.
- **Mass<->Tire coupling** (Sec 8): `resolve_tire_calculation_mass()`,
  `src/vde_core/vde_setup_service.py:226-275`, reads the mass resolver's
  published `tire_load_mass_used_kg`/`tire_load_mass_basis` first, falling
  back to independent re-derivation only if absent. A Quick Scenario mass
  override that writes those two fields is automatically picked up by the
  tire resolver with no separate plumbing.

### Aero (CdA)

No dedicated `vde_aero_proposal_resolver.py` exists. CdA resolution is
inline in `_resolve_aero()`, `src/vde_core/vde_request_resolver.py:633-674`
(`AERO_DELTA_CDA`/`AERO_ABSOLUTE_CDA`), which calls the canonical
`cdA_to_C(delta_cda_m2, rho=1.2)`, `src/vde_core/roadload/engine.py:194-198`
-- the single correct entry point for CdA-affects-C math. `_resolve_aero`
itself is private/request-resolver-internal; Quick Scenario should call
`cdA_to_C()` directly (already public) rather than depend on the private
function, mirroring its exact pattern.

---

## 3. VDE Request Resolver vs. the frozen Vehicle Demand Core -- two separate pipelines

**Critical finding, not a conflict but an important disambiguation**:
`src/vde_core/vde_request_resolver.py` (`resolve_vde_request()`, the sole
public entry point, line 1236) is a **completely separate, older pipeline**
from Sprint 9's frozen `src/vde_core/vehicle_demand/`. It orchestrates
Mass -> Aero -> Tire -> Transmission -> Brake -> Axle Hubs -> Parasitic
(the *actual* fixed order, `vde_request_resolver.py:1311` -- not the
Mass -> Tire -> Aero order implied by the task's conceptual diagram in
Sec 8; the code's own order is authoritative) against a workbook-shaped
`dict`, ultimately delegating to `build_vde_setup_preview()` (from
`vde_workflow_service.py`) for the actual ABC/VDE computation. **It has no
relationship to, and does not import, `src/vde_core/vehicle_demand/`.**

The Sprint 10 entry point Quick Scenario must eventually target is the
**frozen** `calculate_vehicle_demand(request: VehicleDemandRequest,
cycle_frame) -> VehicleDemandResult`, `src/vde_core/vehicle_demand/engine.py:220`
-- fed by a `VehicleDemandRequest` built via either
`vehicle_demand.adapters.build_vehicle_demand_request()` (raw `vde_db` row)
or the `comparison_vehicle_demand_viewmodels` pattern (already-resolved
`ComparisonItem`). `VehicleDemandRequest.test_mass_kg`/`cda_m2`/
`rrc_n_per_kn` are already plain, independently-overridable scalar fields
(`src/vde_core/vehicle_demand/contracts.py:157,166-167`) -- a Quick Scenario
override does not require any contract change to reach the frozen core,
only substituting resolved-override values when constructing the request.

The task spec's Sec 25 stop condition #3 ("Target RRC / Tire DB / pressure
semantics cannot be reproduced through the canonical existing path without
changing physics") does **not** trigger: every Tire/Mass/Aero operation in
scope has a directly reusable canonical function (Sec 1-2 above). Stop
conditions #1/#2/#6/#7/#8 likewise did not trigger -- see Sec 8 below.

---

## 4. Comparison scenario identity (Vehicle + Efficiency shared foundation)

**Confirmed**: `canonical_identity(item: ComparisonItem) -> str`,
`src/vde_app/comparison_report_viewmodels.py:1285-1293`, returns
`f"fc:{item.fuelcons_id}"` when FuelCons-backed, else `f"vde:{item.vde_id}"`.
This is the identity used throughout Scorecard/Dashboard/Walk/Presentation
-- **not** collapsed by `vde_id`. A distinct, narrower function,
`_scenario_identity()` (module-private, line 1278-1282), exists only to
feed `deduplicate_by_vde_id()` (line 1304-1322), which is explicitly scoped
to **physical chart traces only** (two FuelCons scenarios sharing one VDE
draw one overlapping trace) and is documented as never touching
Scorecard/Dashboard scenario-level rows. Two dedicated tests
(`test_dedup_by_vde_id_collapses_shared_vde_with_attribution`,
`test_two_fuelcons_scenarios_sharing_one_vde_produce_identical_physical_lineage`)
confirm scenario-row distinctness survives shared-VDE trace dedup.

`dedupe_titles()` (`comparison_report_viewmodels.py:1880-1895`) is strictly
a display-string disambiguator (appends `" (2)"`, etc.) -- it never touches
`ComparisonItem`/`fuelcons_id`/`vde_id`/`canonical_identity`.

**Consequence for Quick Scenario identity** (Sec 3 of the task spec): a
Quick Scenario has neither a real `fuelcons_id` nor a role as a bare
`vde_id` selection, so it cannot reuse either existing prefix. This
package's `build_quick_scenario_identity(source_identity, slot) ->
f"qs:{source_identity}:{slot}"` uses a distinct third namespace that wraps
the *existing* `canonical_identity()` string as `source_identity`, so a
Quick Scenario's full lineage back to its real source Comparison scenario
(not just its `vde_id`) is always preserved in the identity string itself.
`QuickScenario.__post_init__` rejects a `source_identity` that already
starts with `qs:`, encoding "no Quick -> Quick lineage" as a contract
invariant rather than a UI-only rule.

**DB cardinality** (`src/vde_core/db.py`): `fuelcons_db.vde_id` (line 443,
`NOT NULL REFERENCES vde_db(id) ON DELETE CASCADE`) has no `UNIQUE`
constraint -- confirmed one-VDE-to-many-FuelCons, matching the "Scenario
identity != VDE identity" invariant at the schema level too.

The closest existing precedent for a non-persisted, derived Comparison
variant is the temporary-transmission override already shipped:
`_TEMP_TRANSMISSION_KEY` session state
(`src/vde_app/components/comparison_report.py:136-143,1518`), a plain
`dict[int, dict]` fed into `build_comparison_dataset(...,
temporary_transmission_by_vde_id=...)`
(`src/vde_core/comparison_report_service.py:643-648`), which flows through
unchanged into `build_scenario_comparison_item`/`build_vde_comparison_item`
without ever touching `vde_db`. A later Quick Scenario resolver package
should follow this exact shape (construct a `ComparisonItem` via the
existing factories with an in-memory override, never a new parallel
item-construction path).

---

## 5. Powertrain / PSE / fuel-energy (Efficiency Quick)

- **PSE definition, quoted verbatim** (`src/vde_core/powertrain_efficiency.py:165-167`):
  *"PSE is cycle-effective and should not be interpreted as pure engine
  efficiency."* Formula (line 162): `pse = demand_mj_per_km /
  total_consumed_mj_per_km`, method-agnostic (physics/regression/manual/ML
  all flow through the same `build_powertrain_efficiency_summary()`).
- **No "Final PSE" concept exists today.** The closest precedent is the
  baseline's "Confirm baseline" button (`pwt_fuel_energy.py:3859-3874`),
  which snapshots the active method's request/result into
  `st.session_state["pwt_confirmed_baseline_snapshot"]` -- a single
  explicit confirm-then-lock action. This is the pattern this package's
  `final_pse_percent`/`pse_provenance` pairing on `QuickScenario` is
  designed to be filled by in a later UI package, not a new pattern.
- **Benchmark PSE ("another FuelCons row")**: fully implemented.
  `_derive_reference_pse()` (`pwt_fuel_energy.py:728-755`) computes the
  donor row's own PSE; `_build_observed_reference_request()` (lines
  5496-5527) feeds that value as `eta_pt_est`/`bev_eff_drive` through the
  same `physics_simple` engine against the *active* vehicle's own demand.
  No separate rebasing formula exists to duplicate.
- **Technology Delta**: a plain `dict`, not a dataclass, normalized by
  `_technology_deltas()` (`pwt_fuel_energy.py:863-897`) with fields
  `affected_subsystem`, `source_type`, `maturity_level`, `confidence`,
  `effect_basis`, `effect_value`. **Stacking math, confirmed exactly**
  (`_apply_delta_stack_to_baseline()`, lines 931-1043): a single sequential
  loop mutating one running `proposal` dict -- absolute deltas
  (`pse_delta`, `fuel_delta`, `co2_delta`, `energy_delta`) sum linearly;
  percent/multiplier deltas (`pse_percent_delta`, `pse_multiplier`,
  `efficiency_multiplier`, `fuel_percent_delta`, `co2_percent_delta`)
  compound sequentially (two independent "+5%" deltas yield +10.25%, not
  +10%). This is real, verified-by-code-read behavior with **no existing
  multi-delta regression test** -- a gap worth closing in a later package,
  not 10A. Today, Tech Delta **applies live/automatically** with no
  acceptance gate (unlike "Confirm baseline") -- Sprint 10's requirement
  that Tech Delta act as a recommendation until explicitly accepted has no
  direct precedent and must be newly added at the UI/gating layer in a
  later package; the underlying stacking *math* should still be reused
  verbatim.
- **Canonical LHV/CO2 tables**: `src/vde_core/fuel_energy.py`,
  `LHV_MJ_PER_L`/`GCO2_PER_L` (lines 8-24), explicitly documented as the
  one table that "actually backs the stored consumption numbers" (as
  opposed to two unrelated display-only 34.2 constants elsewhere in
  `src/vde_app/`). Never redeclare.
- **TOTAL/NET**: no silent fallback anywhere in this chain
  (`_resolve_energy_basis()`, `fuel_estimation.py:142-174`; `test_fuel_estimation.py::test_run_fuel_estimation_warns_when_vde_net_is_unavailable`
  locks this in).

---

## 6. ML PSE recommendation

- **Runtime entry point**: `predict_fuel_with_ml()`,
  `src/vde_core/ml_prediction.py:606-749`, invoked through
  `run_fuel_estimation(method="ml_prediction")`
  (`fuel_estimation.py:284-294,477-479`). Artifact:
  `models/powertrain_scenario_ml.joblib`, loaded via `joblib.load` in
  `load_ml_predictor()` (lines 388-452), **no caching** (re-deserializes
  ~16.6 MB on every call).
- **Confirmed: the model never predicts PSE.** `NotebookPowertrainPredictor.__call__`
  (lines 254-282) predicts only `fuel_l_100km`/`energy_Wh_km` (+ urban/
  highway split + CO2). PSE is derived once, downstream, in
  `build_powertrain_efficiency_summary()` -- the exact same method-agnostic
  formula used for every other PSE method (Sec 5 above) -- from
  `demand_mj_per_km` (always from the active VDE basis, never from the
  model) divided by the model's predicted consumption converted to MJ/km.
- **Critical feature-visibility finding**: the model's 15 input features
  (`NotebookPowertrainPredictor.feature_columns`,
  `ml_prediction.py:99-115`) do **not** include `mass_kg` at all. Mass, CdA,
  and RRC can only reach the model indirectly, through 6 "conditional"
  fields (`coast_A_N`, `coast_B_N_per_kph`, `coast_C_N_per_kph2`,
  `vde_net_mj_per_km`, `vde_urb_mj_per_km`, `vde_hw_mj_per_km`) -- **and
  only if** whatever builds the Quick Scenario's ML request recomputes
  those coastdown/VDE values from the edited Mass/CdA/RRC before calling
  the inference path. A later ML-recommendation package must verify this
  wiring field-by-field before ever presenting an ML recommendation as
  informed by a Quick Scenario Mass/CdA/RRC edit -- otherwise the model
  silently reproduces the unedited baseline while the UI implies otherwise.
  This is the single most important constraint for Sec 12 of the task spec.
- **No numeric confidence metric exists** -- only a categorical
  `high`/`medium`/`low`/`provided` label, derived from a stored R2
  threshold and downgraded (never upgraded) by coverage/missing-feature
  count (`_downgrade_ml_confidence()`, lines 551-567). Sprint 10 must not
  invent a numeric confidence score.
- **Explicit opt-in confirmed**: the user must select `"ML prediction"`
  from a method selectbox (`pwt_fuel_energy.py:3658-3677`); nothing applies
  silently. The existing "Confirm baseline" gate is the pattern a Quick
  Scenario "Use ML recommendation" action should mirror.

---

## 7. Vehicle parity plan (for a later package's test suite)

Required future parity assertions (Sec 21 of the task spec), with the exact
functions to call on both sides:

```text
Quick Mass  == equivalent VDE Setup Mass resolution
    Quick side:    resolve_mass_proposal(...)            [vde_mass_proposal_resolver.py:20]
    VDE Setup side: same function, called from the existing VDE Setup UI path
    Compare: vde_calculation_mass_kg, tire_load_mass_used_kg,
             test_mass_kg / resolved_test_mass_kg

Quick Tire  == equivalent VDE Setup Tire resolution
    Quick side:    resolve_tire_proposal(...)             [vde_tire_proposal_resolver.py:24]
    VDE Setup side: same function
    Compare: resolved_rrc_N_per_kN, tire_A_final/B_final/C_final
             (via calculate_vehicle_tire_abc, roadload/tire_model.py:635)

Quick Aero  == equivalent VDE Setup Aero resolution
    Quick side:    cdA_to_C(delta_cda_m2)                 [roadload/engine.py:194]
    VDE Setup side: _resolve_aero(...)                     [vde_request_resolver.py:633]
    Compare: CdA, C_N_per_kph2 (initial_abc_total["C"])

Authoritative ABC / VDE parity (both bases)
    Compare via build_vde_setup_preview(...) [vde_workflow_service.py] on both
    sides -- abc_total, abc_net, vde_total, vde_net -- since this is the
    function both the legacy VDE Request Resolver and (indirectly, via
    resolve_roadload_boundaries in comparison_report_service.py) the
    Comparison/Vehicle Demand path both ultimately depend on for the
    authoritative roadload numbers.
```

Existing test helpers/fixtures identified as reusable for this: the QA
fixture rows in `src/vde_core/qa_mock_data.py` (multiple `fuelcons_db` rows
sharing `vde_id=900001`, useful for both mass/tire/aero parity and identity-
collision regression); `tests/test_vde_mass_proposal_resolver.py`,
`tests/test_vde_tire_proposal_resolver.py`, and
`tests/test_vde_request_resolver.py`'s existing fixture-building helpers as
a template for constructing minimal `source_snapshot`/`current_snapshot`
dicts without touching a real DB. Numerical tolerance: match the project's
existing convention of `assertAlmostEqual` (7 decimal places, unittest
default) used throughout the cited resolver test files.

This plan is **not implemented** in 10A (no Quick resolver exists yet to
compare against) -- it is the design a later package's parity test suite
should follow.

---

## 8. Stop-condition review (Sec 25 of the task spec)

None of the 8 stop conditions triggered:

1. **Canonical VDE Setup resolves an agreed Quick operation differently
   from Sprint semantics** -- not found; every operation in Sec 1's table
   has one canonical implementation, confirmed by direct code read across
   all 5 audit areas.
2. **Quick would require new Mass/Tire/Aero physics** -- not found; every
   Vehicle Quick operation maps to an existing, directly-callable function
   (Sec 1-2).
3. **Target RRC / Tire DB / pressure semantics cannot be reproduced through
   the canonical path without changing physics** -- not found; all
   reproducible via `resolve_tire_proposal()` and its underlying
   `calculate_vehicle_tire_abc()`/pressure-adjustment functions (Sec 2).
4. **PSE meaning differs materially between Comparison, Powertrain
   Scenario, and Quick Scenario** -- not found; PSE's definition
   ("cycle-effective system efficiency") and derivation formula are single-
   sourced in `powertrain_efficiency.py` and reused identically everywhere
   (Sec 5).
5. **Existing Technology Delta semantics would create a materially
   different meaning from "PSE recommendation"** -- not found; the
   underlying stacking math is exactly what Sec 14 of the task spec asked
   to preserve. The only real difference (live-apply vs. accept-gated) is
   an anticipated, spec-acknowledged UI/gating change, not a semantic
   conflict (Sec 5 above, Sec 14 of the task spec explicitly anticipates
   this).
6. **Current ML runtime cannot defensibly derive a PSE recommendation from
   its predicted outputs** -- not found; `build_powertrain_efficiency_summary()`
   already derives PSE from any method's predicted consumption, ML
   included, via the same formula (Sec 6).
7. **Scenario identity cannot be preserved without a significant Comparison
   schema change** -- not found; the existing `canonical_identity()` scheme
   is reused as the `source_identity` payload inside a new, additive `qs:`
   namespace -- no `ComparisonItem`/DB schema change needed, since Quick
   Scenarios are never persisted (Sec 4).
8. **Any DB schema migration appears necessary** -- not found; Quick
   Scenario is fully session-scoped/non-persistent by design.

---

## 9. Contracts delivered in 10A

### Files created

- `src/vde_core/quick_scenario/__init__.py`
- `src/vde_core/quick_scenario/contracts.py`
- `src/vde_core/quick_scenario/serialization.py`
- `tests/test_quick_scenario_contracts.py` (48 tests)
- `docs/sprints/SPRINT_10A_QUICK_SCENARIO_CONTRACT_AUDIT.md` (this document)

No existing file was modified. Package location mirrors
`src/vde_core/vehicle_demand/`'s shape (`contracts.py` + `serialization.py`
+ `__init__.py`) and lives under `src/vde_core/` (Streamlit-free canonical
layer) rather than `src/vde_app/` (Comparison UI) -- neither depends on the
other; `quick_scenario` does not import `vehicle_demand` (the frozen core)
or any Comparison UI module, and nothing outside this package imports it yet
(it has no integration point until a later package wires it in).

### Key contract decisions

- **`ScalarChange`** (`ScalarChangeMode.ABSOLUTE/DELTA/PERCENT` +
  `resolve(source)`): the shared generic scalar-change concept from Sec 20C
  of the task spec, used for Mass and CdA Quick overrides. `resolve()`
  returns `None` (never a guess) for DELTA/PERCENT against a missing
  source, and treats an explicit `value=0.0` as a real change, never as
  "blank" (blank is represented one level up, by the containing
  `Optional[ScalarChange]` field being `None`).
- **Tire Improvement % deliberately does not use `ScalarChange`.** It is a
  dedicated `TireQuickChange.improvement_pct` field, preserving the
  existing EcoDrive lower-RR-positive convention exactly as implemented in
  `resolve_tire_proposal()`, per the task spec's explicit instruction not to
  represent it as a generic percent change.
- **`TireQuickChange`** encodes the Sec 6-7 transformation limit as a
  validated invariant (`__post_init__` checks the `(TireSource,
  TireTransformMode)` combination against
  `_ALLOWED_TIRE_TRANSFORMS_BY_SOURCE`), so `TIRE_DB + TARGET_RRC`/`TIRE_DB
  + RRC_DELTA` are rejected at construction time, not left to a future
  caller to remember.
- **`TirePressureDelta`** carries the DB-vs-user-provided reference-
  pressure distinction identified as a genuine gap in Sec 2 above
  (`ReferencePressureProvenance.SOURCE`/`USER_PROVIDED`), with a validated
  invariant that `USER_PROVIDED` always carries an explicit
  `reference_pressure_psi` (no silent default, per the task spec's frozen
  product decision).
- **`QuickVehicleReadiness`** encodes Sec 18's "no silent partial Vehicle
  override" rule structurally: `all_ready` only considers domains that were
  actually requested (`NOT_REQUESTED` never blocks readiness), so a
  scenario requesting Mass+Aero+Tire remains unresolved until all three
  report `READY`, while a scenario requesting only Mass is unaffected by
  Aero/Tire being untouched.
- **`PseProvenance`** has exactly the 5 values the task spec's Sec 22 test
  requirement names (`INHERITED_CURRENT`, `USER_PROVIDED`,
  `BENCHMARK_ACCEPTED`, `ML_RECOMMENDATION_ACCEPTED`,
  `TECH_DELTA_ACCEPTED`). `QuickScenario.__post_init__` enforces
  `final_pse_percent` and `pse_provenance` are set together or not at all --
  a value with no provenance, or a provenance with no value, is rejected.
- **`QuickScenario.identity`** = `build_quick_scenario_identity(source_identity,
  slot)` = `f"qs:{source_identity}:{slot}"`, with `source_identity` required
  to be a full existing Comparison identity string (not a bare `vde_id`) and
  explicitly rejected if it is itself a `qs:` identity (no Quick -> Quick
  lineage), and `slot` constrained to `1..MAX_QUICK_SCENARIOS_PER_SOURCE`
  (3).
- **No persistence surface**: `QuickScenario` has no `save()`/`to_db_row()`/
  `persist()` method and no field naming a `fuelcons_db`/`vde_db` row it
  owns (`fuelcons_id`/`vde_id`/`id`/`db_id` are all absent from its
  dataclass fields) -- building or holding one cannot, by construction,
  write to either table.

### Serialization / app-boundary behavior

`to_serializable()` recursively converts any contract object (dataclass,
`_TextEnum` member, tuple/frozenset, nested contract) into a plain
JSON-safe structure (mirrors `vehicle_demand/serialization.py`'s
`to_serializable`/`_clean_scalar` pattern exactly, including NaN-to-`None`
normalization). Typed reconstructors (`quick_scenario_from_dict()` and five
supporting `*_from_dict()` functions) invert this losslessly --
`test_full_scenario_round_trips`/`test_minimal_scenario_round_trips` in the
new test file confirm `quick_scenario_from_dict(json.loads(json.dumps(
to_serializable(x)))) == x` for both a fully-populated and a minimal
scenario.

---

## 10. Tests added in 10A

`tests/test_quick_scenario_contracts.py`, 48 tests, all passing, covering:

- `ScalarChange` ABSOLUTE/DELTA/PERCENT resolution, including the
  missing-source-returns-`None` case and zero-as-neutral-change case.
- `VehicleQuickOverrides.is_empty` (blank means no override).
- `TireQuickChange`/`TirePressureDelta` validation for every allowed and
  disallowed `(TireSource, TireTransformMode)` combination from Sec 7, plus
  the reference-pressure-provenance invariant.
- `QuickVehicleReadiness.all_ready` (NOT_REQUESTED never blocks; one
  requested-but-MISSING domain blocks).
- Quick Scenario identity: format, uniqueness across slots for one source,
  rejection of out-of-range slots, rejection of an empty `source_identity`,
  rejection of Quick -> Quick lineage, and -- directly exercising the "two
  FuelCons scenarios sharing one VDE must not collapse" invariant from
  Sec 3/4 above -- two `QuickScenario`s built from distinct
  `fc:`-namespaced sources produce distinct identities even though they
  would (conceptually) share one underlying `vde_id`.
- Final PSE provenance: all 5 required values distinguishable; value and
  provenance required together; zero PSE preserved as explicit; a
  simulated "user edits an accepted recommendation" case demonstrating the
  provenance changes to `USER_PROVIDED`.
- Full and minimal round-trip serialization through `json.dumps`/
  `json.loads`.
- No save/persistence method or DB-row-identity field exists on
  `QuickScenario`.
- Neither `contracts.py` nor `serialization.py` contains an `import
  streamlit` statement.

No physics tests were added for unimplemented physics (per the task spec's
instruction not to fake tests for behavior not yet built) -- the contracts
are validated for shape/invariants only; Sec 7 above documents the parity
tests a later package should add once a Quick resolver exists to test
against.

---

## 11. Deliverable summary

**Repository audit**: complete; canonical owners identified for every
capability in the task spec's Sec 19/20A checklist (Sec 1 table above); two
corrections to the task's stated assumptions found (GVWR/GCWR/CUSTOM_MASS
already exist in the mass resolver, Sec 2; Technology Delta currently
applies live with no accept gate, Sec 5) -- neither is a stop-condition
conflict, both are noted so a later package doesn't rediscover them; one
genuine contract gap found and addressed (reference-pressure provenance,
Sec 2/9).

**Contracts**: `src/vde_core/quick_scenario/{__init__,contracts,serialization}.py`,
Streamlit-free, no dependency on the frozen Vehicle Demand Core or any
Comparison UI module, fully round-trip serializable at the JSON boundary.

**Vehicle parity plan**: documented in Sec 7 above -- exact functions to
call on both the Quick and VDE Setup sides, exact fields to compare, and
which existing fixtures/tolerance convention to reuse. Not implemented (no
resolver exists yet to test).

**PSE**: current definition and derivation formula documented (Sec 5);
manual Final PSE has no existing implementation to reuse but a clear
"Confirm baseline" precedent to mirror; benchmark/ML/Tech-Delta paths are
all directly reusable, PSE-is-always-derived-never-predicted confirmed for
every method including ML.

**ML**: artifact and inference path documented (Sec 6); PSE derivation
method confirmed identical to every other method; feature-visibility limits
for Quick-resolved Mass/CdA/RRC documented field-by-field; no numeric
confidence metric exists and none was invented.

**Technology Delta**: existing schema (plain dict, not a dataclass) and
exact stacking math documented (Sec 5); the tiny catalog concept from the
task spec's Sec 13 can map directly onto the existing `effect_basis`/
`effect_value` fields with no schema change -- deferred to a later package,
since Sec 13 explicitly says not to create the CSV catalog in 10A unless a
contract test needs it (none did).

**Tests**: baseline 1359/1357 (2 known pre-existing, unrelated); 48 new
Quick Scenario contract tests, all passing; full suite post-change
1407/1405, same 2 known pre-existing failures, zero new regressions.

**Git**: committed as 10A on `sprint-10-interactive-quick-scenarios` after
the above test results were confirmed. 10B was not started.
