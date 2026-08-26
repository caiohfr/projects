# Sprint 10B - Quick Mass + Aero Resolution

## Status

**Executable Quick Mass + Aero resolver delivered. No Vehicle Demand, VDE
resolver, Powertrain, or ML physics modified.**

Branch: `sprint-10-interactive-quick-scenarios`, building directly on
Sprint 10A (commit `83f93513`). Sprint 10A audited every canonical
capability and shipped input-only Quick Scenario contracts
(`src/vde_core/quick_scenario/contracts.py`) with no resolver behind them.
10B is the first executable slice: given a `QuickScenario` with Mass and/or
CdA overrides, produce a resolved temporary physical state and feed it
through the **frozen** Sprint 9 Vehicle Demand Core
(`calculate_vehicle_demand()`), using only existing canonical functions.

## Scope

Implemented: Vehicle Curb Mass (Absolute/Delta/Percent), EPA regulatory
mass ("Target TWC" and "TWC Shift"), WLTP canonical mass behavior, Aero
CdA (Absolute/Delta/Percent), Mass+Aero composition, readiness/no-silent-
partial-override semantics, source immutability, scenario-identity
preservation across shared-VDE FuelCons scenarios.

Not implemented (explicit non-goals, unchanged from the base Sprint 10
spec): Tire/RRC, PSE, ML recommendation, Technology Delta, fuel/energy
Quick, any UI, DB persistence/save, regen, ambient/temperature correction,
transmission/brake/parasitic/axle-hub changes. A `QuickScenario` requesting
a Tire change is explicitly rejected (readiness `MISSING`, not silently
ignored) rather than left unhandled.

---

## Architecture

```text
QuickScenario.source_identity ("fc:<id>" | "vde:<id>")
        |
        v
_fetch_source_vde_row()                       [resolver.py, new]
   fc: -> get_record(EntityType.FUEL_CONSUMPTION) -> fetch_vde_by_id(vde_id)
   vde: -> fetch_vde_by_id(id)  directly
        |
        v
_resolve_mass() -----------------> resolve_mass_proposal()      [canonical, unmodified]
_resolve_aero() -----------------> cdA_to_C()                   [canonical, unmodified]
        |
        v
_build_synthetic_row()   (dict copy of source row + Mass/Aero updates)
        |
        +--> resolve_roadload_boundaries()  --> abc_total / abc_net       [canonical, unmodified]
        +--> resolve_cycle_vde_results()    --> vde_total/net_mj_per_km   [canonical, unmodified]
        +--> build_vehicle_demand_request() --> VehicleDemandRequest      [canonical, unmodified]
        +--> resolve_vehicle_demand_cycle() --> cycle_frame               [canonical, unmodified]
        +--> calculate_vehicle_demand()     --> VehicleDemandResult       [FROZEN, unmodified]
        |
        v
QuickVehicleResolution   [resolution.py, new]
```

`src/vde_core/vehicle_demand/` was not modified. `vde_request_resolver.py`
(the legacy workbook pipeline) was never called.

## Files changed

Created:
- `src/vde_core/quick_scenario/resolution.py` -- `QuickVehicleResolution`
  (the resolved-output contract; depends on `vehicle_demand`, unlike
  `contracts.py`).
- `src/vde_core/quick_scenario/resolver.py` -- `resolve_quick_vehicle_scenario()`
  and its private helpers.
- `tests/test_quick_scenario_resolver.py` -- 31 unit tests.
- `tests/test_quick_scenario_vehicle_demand_integration.py` -- 5
  cross-path parity/reconciliation tests.
- This document.

Modified:
- `src/vde_core/quick_scenario/contracts.py` -- added `MassQuickChange`;
  changed `VehicleQuickOverrides.mass_change` type from `ScalarChange` to
  `MassQuickChange`; added `aero_reference_cda_m2`/
  `aero_reference_cda_provenance` fields + a `__post_init__` guard.
- `src/vde_core/quick_scenario/serialization.py` -- added
  `mass_quick_change_from_dict()`; updated
  `vehicle_quick_overrides_from_dict()`.
- `src/vde_core/quick_scenario/__init__.py` -- re-exports the above.
- `tests/test_quick_scenario_contracts.py` -- updated the one fixture using
  the old `mass_change=ScalarChange(...)` shape; added 8 new tests for
  `MassQuickChange` validation and the Aero reference-provenance guard (48
  -> 56 tests).

---

## Decision 1 (fact-derived, not a judgment call): which resolved mass value feeds the frozen core

`resolve_mass_proposal()`'s `resolved_snapshot` carries two different mass
quantities under different keys:

- `test_mass_kg` / `resolved_test_mass_kg`: the resolver's own physical
  test-mass output (for EPA, `curb + EPA_TEST_MASS_DEFAULT_DELTA_KG`, i.e.
  curb+136 -- Loaded Vehicle Weight).
- `vde_calculation_mass_kg`: the canonical mass the resolver itself says
  "VDE/roadload calculations should use" (for EPA, the discretized
  inertia-weight class / TWC; for WLTP, TML/TMH).

The frozen core (`vehicle_demand/adapters.py::_resolve_test_mass_kg`) and
the legacy "golden" reconciliation function
(`comparison_report_service.py::_resolve_mass_for_cycle`) both trust
`vde_row["test_mass_kg"]` **verbatim, with no re-derivation** -- whatever is
stored there is used directly as the mass for VDE physics. Verified against
the QA fixtures: `VDE-QA-001` (`mass_kg=1500.0`, `test_mass_kg=1644.0`) and
`VDE-QA-002`-equivalent (`mass_kg=1480.0`, `test_mass_kg=1588.0`) both store
the **EPA inertia class (TWC)** in `test_mass_kg`, not curb+136 (which would
be 1636.0 / 1616.0 respectively) -- confirmed independently via
`inertia_step_for_mass()` and the fixtures' own `test_mass_basis =
"EPA_INERTIA_CLASS"` column.

**Consequence**: the Quick Mass resolver writes
`resolved_snapshot["vde_calculation_mass_kg"]` into the synthetic row's
`test_mass_kg` key -- never the resolver's own `test_mass_kg`/
`resolved_test_mass_kg` output. Getting this backwards would silently feed
curb+136 into the frozen core for EPA rows instead of the TWC -- a real,
silent, physically-wrong result with no error raised. This was verified
directly against source (not inferred) and independently re-confirmed by a
Plan-review pass before implementation; the resolver's mass-change smoke
test (`inertia_step_for_mass(1480.0)["inertia_class_kg"] == 1588.0`)
reproduces the QA-fixture value exactly.

## Decision 2 (judgment call, recorded for review): Absolute CdA without a source CdA

The canonical `_resolve_aero()`'s `AERO_ABSOLUTE_CDA` branch (in the legacy
`vde_request_resolver.py`, not called by 10B but used as the reference
pattern) always requires a reference CdA to convert an absolute request
into a roadload-C delta -- it never defaults the reference to zero, but it
does accept a caller-supplied manual reference (`baseline_CdA` in its
vocabulary) when source CdA is unavailable, tagged with a `review`-severity
`manual_reference_override` issue.

By contrast, `AERO_DELTA_CDA` does **not** block on a missing source CdA --
it applies the delta unconditionally and only downgrades to a non-blocking
`review` issue. Quick Scenario's DELTA/PERCENT behavior (MISSING when
source CdA is absent) is therefore a **deliberate, spec-mandated
divergence** from that more lenient canonical DELTA behavior, not a match
to it -- and it isn't really an independent Aero rule at all: 10A's
`ScalarChange.resolve()` already returns `None` for DELTA/PERCENT against a
missing source, so the resolver just propagates that `None` to
`DomainReadiness.MISSING`.

For ABSOLUTE, 10B mirrors the canonical resolver's own manual-reference
mechanism, using the same shape 10A already established for Tire's
reference-pressure problem (`TirePressureDelta.reference_pressure_psi`/
`reference_pressure_provenance`): `VehicleQuickOverrides.aero_reference_cda_m2`
+ `aero_reference_cda_provenance: ReferencePressureProvenance`. When source
CdA is available, it's used as the reference (implicit `SOURCE`). When
unavailable, the resolver falls back to `aero_reference_cda_m2` only if
`aero_reference_cda_provenance is USER_PROVIDED`; otherwise the domain is
`MISSING` with an explicit issue -- never a silent zero-reference guess.

## Decision 3: Mass needs two request shapes, not one `ScalarChange`

"Target TWC" (curb-mass-driven, `EPA_CURB_TO_TWC`/`WLTP_MASS_LINE`) and
"TWC Shift" (step-count-driven from the *current* TWC, `MASS_TWC_SHIFT`,
EPA-only) are structurally different inputs -- the latter has no curb-mass
component at all. `MassQuickChange` (mirroring 10A's own `TireQuickChange`
"exactly one sub-mode" pattern) replaces `VehicleQuickOverrides.mass_change`'s
type. `legislation` on the source row picks `EPA_CURB_TO_TWC` vs.
`WLTP_MASS_LINE` for a `curb_change`; `twc_shift_steps` is rejected
(explicit issue, not silently ignored) when `legislation != "EPA"`.

---

## Mass call path

```python
_resolve_mass(source_row, mass_change: MassQuickChange | None)
```

- `curb_change` (EPA): `requested_curb = curb_change.resolve(source_row["mass_kg"])`
  -> `resolve_mass_proposal(dict(source_row), "EPA_CURB_TO_TWC", {"mass_kg": requested_curb})`.
- `curb_change` (WLTP): same, `proposal_type="WLTP_MASS_LINE"`, optional
  `line_type` pass-through.
- `twc_shift_steps` (EPA only): `resolve_mass_proposal(dict(source_row), "MASS_TWC_SHIFT", {"shift_steps": ..., "target_side": ..., "curb_position": ...})`.
- `resolve_mass_proposal`'s `status` maps `"OK"`/`"Review"` -> `READY`,
  `"Missing"`/`"Invalid"` -> `MISSING` (matching the canonical resolver's
  own non-blocking-vs-blocking severity split).
- On success: synthetic-row updates are
  `{"mass_kg": resolved["curb_mass_kg"], "test_mass_kg": resolved["vde_calculation_mass_kg"], "vde_mass_basis": resolved["vde_mass_basis"]}`
  (Decision 1).

**EPA TWC-bracket behavior confirmed by test** (never hand-guessed --
bracket membership is always derived by calling `inertia_step_for_mass()`
at test-write-time): a curb nudge staying inside one TWC bracket leaves
`resolved_vde_calculation_mass_kg` (and hence `test_mass_kg`) unchanged;
crossing a bracket boundary changes it. Both are asserted in
`tests/test_quick_scenario_resolver.py::EpaTwcBoundaryTests`.

## Aero call path

```python
_resolve_aero(source_row, cda_change, aero_reference_cda_m2, aero_reference_cda_provenance)
```

- `target_cda = cda_change.resolve(source_row["cda_m2"])` -- `None` for
  DELTA/PERCENT against a missing source (Decision 2).
- `reference_cda` = source CdA if available, else `aero_reference_cda_m2`
  (only if provenance is `USER_PROVIDED`), else `MISSING`.
- `delta_cda = target_cda - reference_cda`; synthetic-row updates are
  `{"cda_m2": target_cda, "coast_C_N_per_kph2": source_row["coast_C_N_per_kph2"] + cdA_to_C(delta_cda)}`
  -- the exact same two-line composition `_resolve_aero()`'s
  `AERO_DELTA_CDA`/`AERO_ABSOLUTE_CDA` branches use (`coast_A_N`/
  `coast_B_N_per_kph` untouched).
- NET is never touched directly: `resolve_roadload_boundaries()` always
  recomputes NET as TOTAL minus the (unchanged) transmission ABC on the
  synthetic row, so a correct NET falls out automatically.

## VehicleDemandRequest construction path

The synthetic row (a plain `dict`, same shape as a raw `vde_db` row) is
handed directly to the **existing, unmodified** raw-row adapter:
`build_vehicle_demand_request(synthetic_row)` +
`resolve_vehicle_demand_cycle(synthetic_row)` ->
`calculate_vehicle_demand(request, cycle_frame)`. No extraction or
adapter change was needed -- the adapter already accepts a plain row dict
and internally resolves both TOTAL and NET boundaries.

---

## Parity test results

All comparisons are "same canonical function, same effective input" --
never a cross-engine comparison (see the note below on why).

| Case | Test | Result |
|---|---|---|
| No-change, EPA (VDE-QA-001) | `NoChangeParityTests.test_epa_qa_001_no_change_reproduces_source` | PASS |
| No-change, WLTP | `NoChangeParityTests.test_wltp_no_change_reproduces_source` | PASS |
| Curb mass change, EPA | `CurbMassChangeParityTests.test_epa_curb_change_parity` | PASS |
| CdA change, EPA | `CdaChangeParityTests.test_epa_cda_change_parity` | PASS |
| Two FuelCons scenarios sharing one VDE (real seeded DB, `fc:` lookup path) | `SharedVdeDistinctIdentityParityTests` | PASS |

**Note on EPA cross-engine reconciliation**: Sprint 9's own reconciliation
suite (`tests/test_vehicle_demand_integration.py`) already establishes that
the legacy ABC-polynomial path (`resolve_cycle_vde_results`, EPA 55/45
city/highway phase-weighted) and the frozen Vehicle Demand Core
(`calculate_vehicle_demand`, whole-trace) only agree for EPA rows via a
deliberate phase-split-and-recombine step (`_epa_combined_vde` in that
test file) -- a direct `total_summary.vde_mj_per_km` vs.
`resolve_cycle_vde_results(...)["total"].aggregate` comparison does **not**
match for EPA (confirmed directly: 0.2965 vs. 0.3130 MJ/km for VDE-QA-001
with no override applied). This is a pre-existing, Sprint-9-documented
characteristic of the two engines, not a 10B defect, and reconciling it is
Sprint 9's concern. 10B's parity tests therefore never compare one engine's
output against the other's -- each test compares the Quick resolver's call
to a canonical function against an *independent* call to that *same*
function on the *same* effective (synthetic) row, which is exactly what
"prove reuse, don't guess a formula" requires and sidesteps the
EPA-vs-whole-trace divergence entirely. WLTP has no such divergence (its
phase weighting reduces to a plain distance-weighted average that a
whole-trace integral reproduces directly), so the no-change WLTP test
happens to also match cross-engine, incidentally.

**Reuse proof** (`ReuseProofTests`): `resolve_mass_proposal` and `cdA_to_C`
are spied via `unittest.mock.patch(..., wraps=<real function>)` and
asserted called with the expected arguments -- the spy wraps the genuine
implementation, so these tests still exercise real physics rather than a
stand-in, and prove the Quick resolver actually delegates rather than
merely producing matching numbers by coincidence.

---

## Immutability / identity

- `ImmutabilityAndDeterminismTests.test_source_row_is_never_mutated`: a
  deep-copied source row is byte-for-byte unchanged after resolution
  (Mass + Aero combined).
- `ImmutabilityAndDeterminismTests.test_repeated_resolution_is_deterministic`:
  two resolutions of the same scenario against the same row produce
  dataclass-equal results.
- `QuickScenarioIdentitySharedVdeTests` (unit, injected row) and
  `SharedVdeDistinctIdentityParityTests` (integration, real seeded DB):
  two Quick Scenarios built from distinct `fc:<fuelcons_id>` sources that
  share one underlying `vde_id` resolve to identical physics but retain
  distinct `quick_scenario_identity` values -- the "scenario identity !=
  VDE identity" invariant survives into the resolution layer, not just the
  10A contract layer.

---

## Tests

Pre-change focused baseline (Sprint 10A's own recorded state): 1407 tests,
1405 passing, 2 known pre-existing failures
(`test_component_lookup_provenance_does_not_change_parasitic_math`,
`test_axle_hubs_lookup_snapshot_preserves_boundary_metadata`, both in
`tests/test_vde_request_resolver.py`, unrelated to Quick Scenario).

Post-change focused set (`test_quick_scenario_*`, `test_vde_mass_proposal_resolver`,
`test_vde_tire_proposal_resolver`, `test_vde_request_resolver`,
`test_vehicle_demand_contracts/engine/integration`,
`test_comparison_vehicle_demand_viewmodels`, `test_comparison_report_viewmodels`,
`test_comparison_report_service`, `test_test_mass`): **569 tests**, same 2
known pre-existing failures, zero new regressions.

Full suite (`python -m unittest discover -s tests`) after 10B: **1451
tests, 1449 passing**, the same 2 known pre-existing failures, **zero new
regressions** (1407 baseline + 44 net new: +8 in
`test_quick_scenario_contracts.py` [48 -> 56] + 31 new in
`test_quick_scenario_resolver.py` + 5 new in
`test_quick_scenario_vehicle_demand_integration.py`).

---

## Closure hotfix: canonical EPA mass persistence

The resolver-level physical state remains the Quick <-> VDE Setup parity
boundary.  Quick writes its temporary synthetic row with
`vde_calculation_mass_kg`; it does not persist a VDE record and no Quick
Scenario or Vehicle Demand physics changed in this hotfix.

The live VDE Setup v2.2 save boundary had the same two resolver values
available, but `_proposal_row_payload()` wrote
`resolved_mass_setup.test_mass_kg` to `vde_db.test_mass_kg`.  For EPA this
is the physical curb+136 kg value, whereas Comparison and Vehicle Demand
read `vde_db.test_mass_kg` as their canonical calculation mass.  The save
path now writes `resolved_mass_setup.vde_calculation_mass_kg` whenever it is
nonblank, then explicitly falls back to the prior resolved/snapshot value.
The fallback deliberately does not use Python truthiness, so a present zero
is not silently discarded.  WLTP is unchanged because its physical and
canonical values are equal by construction.

`tests/test_vde_request_save.py` now exercises the real save-plan boundary
after resolving each mass proposal: EPA curb-to-TWC, a change within one
TWC, a TWC-boundary crossing, an EPA TWC shift, and WLTP mass-line behavior.
The base EPA regression explicitly proves that the persisted value equals
`vde_calculation_mass_kg` and differs from physical curb+136 kg.

Focused persistence coverage passed: **24/24**
(`python -m unittest -v tests.test_vde_request_save`).  Full discovery
after the hotfix ran **1,465 tests in 936.942 s; 1,463 passed**.  The only
two failures are the documented pre-existing failures in
`tests/test_vde_request_resolver.py`:
`test_component_lookup_provenance_does_not_change_parasitic_math` and
`test_axle_hubs_lookup_snapshot_preserves_boundary_metadata`.  The total is
14 higher than the previously recorded 10B total because the pre-existing,
untracked `test_quick_scenario_resolver_parity.py` was present and discovered
in this worktree; it was not changed by the hotfix.

The local-database audit was read-only and covered the repository's active
and canonical QA databases: `data/db/eco_drive.db`,
`data/db/eco_drive_qa.db`, `data/qa/eco_drive.db`, and
`data/qa/eco_drive_qa.db`.  A row was a strict candidate only when it was
EPA, declared `test_mass_basis = PHYSICAL_TEST_MASS`, stored
`test_mass_kg = mass_kg + 136 kg`, and that value differed from its stored
`inertia_class`.  Result: **0 strict candidates** (0 QA, 0 non-QA); the
production database had 5,002 EPA rows but none with the physical-mass
basis/signature, and the canonical QA database had 7 EPA rows but none.
Older rows without this provenance cannot be attributed to this v2.2 save
defect from stored fields alone, so they were intentionally not guessed or
modified.

The legacy phase-weighted VDE output is still **not expected** to equal the
Sprint 9 frozen Vehicle Demand whole-trace output for EPA without the
separate EPA policy phase recombination described above; that established
cross-engine distinction is not affected by this persistence correction.

## Backlog / deferred (not fixed in 10B)

- Tire/RRC Quick resolution -- deferred to a later package, per scope.
  Requesting one today is explicitly rejected (`MISSING`), not silently
  ignored.
- A pre-existing, unrelated observation surfaced while reading
  `vde_workflow_service.py::_build_row_payload`/`resolve_mass_setup`: the
  legacy VDE Setup **save** pipeline appears to write the resolver's
  physical `test_mass_kg` (curb+136-style) into the persisted
  `vde_db.test_mass_kg` column, whereas the QA fixtures (and the frozen
  core's/Comparison's own read path) treat that column as holding
  `vde_calculation_mass_kg` (TWC). This is a legacy VDE Setup save-path
  question unrelated to Quick Scenario (which never writes to `vde_db`) --
  noted here only so a future package doesn't have to rediscover it, not
  fixed as part of 10B.
- Sprint 9's own EPA phase-weighted reconciliation (`_epa_combined_vde`)
  has no regression test for the *general* multi-phase case beyond
  VDE-QA-001/004/006 -- unrelated to 10B, noted only because it was read
  closely while designing the parity-test strategy above.

## Freeze / handoff statement

```text
Vehicle Demand Core API                    FROZEN (untouched)
Legacy vde_request_resolver.py             UNTOUCHED (never invoked by Quick Scenario)
Quick Scenario Mass + Aero resolution      DELIVERED (10B)
Quick Scenario Tire / PSE / UI / save      NOT STARTED
```

10C was not started. Sprint 10's guiding principle holds:
`resolve_quick_vehicle_scenario()` never implements Mass/Aero/Tire/roadload
physics itself -- every physical answer it produces is delegated to, and
numerically identical to, the same canonical function called directly.
