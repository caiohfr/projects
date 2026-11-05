# Sprint 13 — Roadload & Load Case Lab Closure

## Delivered

- typed load-case contracts and deterministic orchestration in
  `src/vde_core/vehicle_demand/load_case.py`;
- Drive Cycle, Fixed Speed and Speed Sweep modes;
- mass modes for inherit, payload delta, target payload, custom mass, GVWR
  and GCWR;
- trailer mass/ABC, exact grade force, conditional ambient aero delta and
  Tire Roadload-owner pressure delta;
- consistent temporary TOTAL/NET overlays;
- independent Streamlit page and VDE Setup handoff;
- prominent steady-state KPIs and a Scenario Force vs Speed Plotly chart;
- exact Fixed Speed operating-point marker and sweep/table point parity;
- optional resolved-contribution breakdown and humanized status labels;
- Metric/US tire-pressure entry using the existing unit-system state;
- distinct baseline effective, scenario vehicle and scenario system mass
  states, each with its applicable basis;
- physically valid equal fixed-speed curves for mass-only overrides, with a
  compact explanation at 0% grade;
- explicit `APPLIED`/`UNAVAILABLE` tire-pressure correction reporting;
- Drive Cycle dynamic force/power/time visualization sourced directly from
  canonical `VehicleDemandProfile` outputs;
- static ABC chart demoted to an explicitly labeled Roadload Reference Curve
  in Drive Cycle mode;
- shared canonical persisted-mass resolver used by Vehicle Demand,
  Comparison and the Lab, including source-field provenance;
- traceable estimated-RRC fallback with strict source priority, exact
  category/electrification/transmission matching and a default-on user toggle;
- reusable `rolling_rrc_prior.py` core owner that preserves EPA category
  vocabulary and resolves one-speed transmissions to `SS` before lookup;
- linked FuelCons context for explicit electrification/gear discrimination,
  without DB access inside the resolver or a generic blank-to-ICE fallback;
- rolling correction provenance (`value`, `source`, `rule_key`, `status` and
  delta ABC), with visible estimated/unavailable diagnostics;
- GCWR rolling correction based on vehicle tire-load mass while grade and
  inertia retain system mass;
- grade kept outside the public temporary scenario ABC contract;
- no Save action or persistence path.

## Requirement evidence

The direct Sprint 13 tests are:

- `tests/test_vehicle_demand_load_case.py` — physics, composition, parity,
  deterministic behavior, supported/unavailable corrections, TOTAL/NET,
  fixed speed, sweep and zero-write evidence;
- `tests/test_roadload_load_case_lab_ui.py` — three-mode page smoke, absence
  of Save, VDE Setup handoff context, result rendering and unchanged VDE row
  count.

Together they cover the v0.2 matrix LC-001 through LC-036 and UI Refinement
Patch checks. Notable direct
tests include:

- `test_no_overrides_has_canonical_parity`;
- `test_zero_grade_is_exact_noop` and `test_grade_sign`;
- `test_gcwr_resolves_vehicle_mass_and_does_not_add_trailer_twice`;
- `test_complete_trailer_abc_is_added_once`;
- `test_total_net_external_overlay_preserves_gap`;
- `test_supported_tire_owner_supplies_explicit_abc_delta`;
- `test_90_kph_matches_analytical_force_power_and_intensity`;
- `test_negative_grade_is_not_clamped`;
- `test_exact_deterministic_points`;
- `test_calculation_performs_no_db_writes`;
- `test_drive_cycle_calculation_renders_comparison_without_db_write`;
- `test_vde_setup_handoff_preselects_baseline`.
- `test_zero_override_force_curve_matches_baseline`;
- `test_grade_is_constant_force_offset_across_context_curve`;
- `test_trailer_and_grade_curve_contributions_close_additively`;
- `test_selected_curve_marker_matches_reported_wheel_force`;
- `test_us_unit_system_uses_psi_for_tire_pressure_delta`.
- `test_baseline_and_gvwr_scenario_are_distinct_physical_states`;
- `test_mass_alone_does_not_change_flat_steady_demand`;
- `test_applied_zero_tire_delta_is_explicit_not_silent`.
- `test_mass_only_cycle_keeps_static_roadload_but_changes_dynamic_profile`;
- `test_load_case_baseline_has_canonical_vehicle_demand_parity`;
- `test_stored_test_mass_without_special_basis_reports_test_mass_source`;
- `test_curb_mass_is_explicit_final_fallback_with_source`;
- `tests/test_vde_calculation_mass.py` for shared adapter/provenance coverage.
- `RollingPriorTests` for all 12 estimated-RRC fallback QA requirements,
  including source priority, exact/unavailable matching, signs, zero delta,
  toggle disablement, GCWR, immutable baseline ABC and no persistence.
- `tests/test_rolling_rrc_prior.py` for direct exact-table lookup, EPA category
  preservation, one-speed transmission semantics, linked context, missing
  discriminants, legacy labels and baseline/scenario/delta rolling physics.
- `test_estimated_rrc_diagnostic_uses_linked_fuelcons_context` for visible UI
  diagnostics and read-only linked FuelCons integration.

## Verification

Latest focused verification after the corrected reusable RULE_ESTIMATED
resolver extraction:

- direct resolver, Load Case and UI scope: 63 passed and 9 subtests;
- expanded related regression: 280 passed and 31 subtests;
- canonical repository suite, `pytest tests -q`: 2,033 passed and 183
  subtests in 624.61 seconds.

Latest verification after the Estimated RRC Prior Fallback patch:

- direct Sprint 13 core/UI scope: 53 passed;
- expanded Vehicle Demand, Tire Roadload, Comparison and Quick Scenario
  regression: 359 passed and 31 subtests;
- canonical repository suite, `pytest tests -q`: 2,023 passed and 174
  subtests in 940.27 seconds.

Latest focused verification after the mass-state/provenance refinement:

- Sprint 13 core/UI: 39 passed;
- Vehicle Demand, mass and UI regression: 159 passed and 10 subtests;
- canonical repository suite, `pytest tests -q`: 2,004 passed and 174
  subtests in 857.47 seconds.

Latest verification after the Drive Cycle visualization and shared canonical
mass-resolution corrective patch:

- corrective Load Case, UI, Quick Scenario and mass tests: 53 passed;
- expanded Vehicle Demand/Comparison/mass/UI regression: 576 passed and 22
  subtests;
- canonical repository suite, `pytest tests -q`: 2,013 passed and 174
  subtests in 718.79 seconds.

Prior full verification after the visualization refinement:

- Sprint 13 core/UI after the visualization refinement: 36 passed;
- focused Vehicle Demand/mass/UI regression after refinement: 143 passed and
  10 subtests;
- VDE Setup AppTest regression: 54 passed;
- canonical repository suite after refinement, `pytest tests -q`: 2,001
  passed and 174 subtests in 823.71 seconds.

An unscoped `pytest -q` from the repository root cannot collect cleanly
because pre-existing duplicate test module names are present under
`deliverables/`, `etl/` and `tests/`. It stopped with 23 collection errors.
The canonical `tests/` suite itself is fully green as reported above.

## Architecture decisions

- Load-case overlays are new narrow orchestration, not a second Vehicle
  Demand engine.
- A combined towing mass is used for inertia and grade, while vehicle-only
  mass is passed to the Tire Roadload owner.
- Load-dependent rolling resolution is centralized in the core Load Case
  orchestration. The UI displays the result and does not reproduce its physics.
- Exact deterministic priors reuse the existing standards table and project
  normalizers; generic fallback and new persistent defaults remain excluded.
- Known rolling decomposition is deliberately unavailable while towing
  rather than applying the vehicle RRC to combined vehicle/trailer mass.
- Fixed-speed force is taken from a canonical-engine profile. Power and wheel
  energy intensity are exposed by the load-case result, not calculated in UI.
- The legacy runtime SQLite schema may lack canonical Sprint 12 ambient
  columns. In that case scenario density can still be displayed, but the
  ambient demand delta is explicitly unavailable without a baseline density.

## Persistence evidence

The core module imports no VDE/FuelCons write service. Direct mock assertions
verify no insert/update calls, and the UI AppTest verifies the VDE row count is
unchanged after calculation. The page contains no Save action.

## QA limitation

AppTest smoke is complete. A real browser smoke was attempted separately but
the Windows UI-control approval expired before the browser could be operated;
therefore manual visual smoke and screenshots are honestly recorded as not
completed.
