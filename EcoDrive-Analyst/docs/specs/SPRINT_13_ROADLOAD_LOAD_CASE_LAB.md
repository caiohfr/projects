# Sprint 13 — Roadload & Load Case Lab

Status: implemented from the user-supplied v0.2 scope candidate.

## Objective

Provide an in-memory Vehicle Demand workspace for temporary environment,
vehicle-load, trailer/towing, grade, fixed-speed and speed-sweep analysis.
The selected persisted VDE remains immutable and authoritative.

## Product placement

- independent Streamlit page: `pages/Roadload_Load_Case_Lab.py`;
- Vehicle Demand child capability adjacent to VDE Setup;
- VDE Setup handoff preloads the selected persisted `vde_id`;
- direct page entry provides its own persisted-baseline selector.

## Analysis modes

1. `DRIVE_CYCLE` delegates to the canonical Vehicle Demand cycle engine;
2. `FIXED_SPEED` delegates to the canonical engine through a constant-speed
   trace and reports Steady-State Wheel Demand;
3. `SPEED_SWEEP` repeats the same fixed-speed evaluation deterministically.

Fixed-speed and sweep outputs are wheel force, wheel power and wheel energy
intensity. They are not labeled VDE.

## UI and visualization refinement

- primary steady-state wheel-force, power and energy-intensity KPIs appear
  before detailed results;
- a core-resolved Scenario Force vs Speed curve keeps baseline roadload and
  temporary scenario force separately identifiable;
- Fixed Speed uses a contextual curve with an exact selected-point marker;
- Speed Sweep uses the same resolved points for its table and chart;
- an optional force breakdown shows only resolved, non-zero trailer, grade
  and condition contributions;
- tire-pressure inputs follow the existing Metric/US unit selection (kPa or
  psi), while the core contract remains canonical kPa;
- internal provenance/status values are humanized only in presentation.
- baseline effective mass/basis, scenario vehicle mass/basis and scenario
  system mass are displayed as separate physical states.

Drive Cycle uses the canonical cycle-resolved TOTAL profiles as its primary
visualization: tractive force versus time, wheel power versus time and the
cycle speed trace. The static ABC force-versus-speed chart is secondary and
explicitly titled `Roadload Reference Curve`; it does not represent inertial
force. Fixed Speed retains Force vs Speed, while Speed Sweep presents both
Force vs Speed and Wheel Power vs Speed.

## Physics and ownership

- authoritative TOTAL/NET ABC comes from the persisted VDE adapter;
- baseline calculation mass is resolved once through the shared canonical
  Vehicle Demand/Comparison path: explicit `vde_calculation_mass_kg`, valid
  persisted regulatory semantics such as EPA inertia class, stored
  `test_mass_kg`, then `mass_kg` only as an explicit final fallback;
- the result exposes baseline calculation mass, basis and source field plus
  scenario calculation mass, scenario basis and scenario system mass;
- grade uses exact geometry and the canonical gravity constant;
- trailer ABC is added once; trailer mass affects system inertia/grade;
- GCWR already represents combined system mass and is never incremented by
  trailer mass a second time;
- air density uses `vehicle_demand.physics.resolve_air_density`;
- ambient demand correction is an explicit in-memory `delta C` only when
  baseline density, scenario density and CdA are all resolvable;
- tire-pressure correction delegates to the existing Tire Roadload service;
- an active tire-pressure correction is explicitly `APPLIED` or
  `UNAVAILABLE`; an applied zero ABC delta is accompanied by an explanatory
  note;
- unavailable trailer, ambient or tire corrections remain explicit and are
  never replaced with generic empirical models;
- the canonical Vehicle Demand engine remains the sole inertia owner.

### Estimated RRC prior fallback

When vehicle-specific RRC is absent, rolling-load resolution follows one
deterministic hierarchy: persisted VDE RRC, applicable Tire DB resolution,
exact `RULE_ESTIMATED` prior, then `UNAVAILABLE`. The prior is read from the
existing `data/standards/vde_defaults_by_category_trans_elec.csv` table and
requires an exact normalized category + electrification + transmission match.
There is no category cascade or global numeric default.

The reusable owner is `src/vde_core/rolling_rrc_prior.py`. Category lookup
uses only trim, canonical whitespace and case-insensitive equality, preserving
the EPA vocabulary in the CSV. It never maps `Midsize Cars` to an application
family such as `PASSENGER_STANDARD`. Transmission follows the v0.4 structural
rule, including the authoritative `gears == 1 -> SS` override. Electrification
is resolved from explicit VDE context, then already-supplied linked context,
then the narrow v0.4 i4/BEV rule; blank metadata does not fabricate ICE.

The default-on UI toggle can disable only the estimated-prior step. Results
expose the resolved RRC value, source, rule key, status and temporary delta
ABC. Estimated use is visibly warned and never persists or enriches the VDE.
The diagnostic exposes raw/resolved category, electrification, transmission,
gear count, attempted/matched keys, missing discriminants and failure reason.
The delta is calculated between baseline and scenario vehicle tire-load mass;
for GCWR this is `gcwr - trailer_mass`, while system mass remains the owner of
grade and inertia.

The shared physical helper materializes baseline rolling ABC, scenario rolling
ABC and their delta before only the delta is composed into scenario roadload.

The public temporary scenario ABC contract is:

`authoritative baseline ABC + rolling/tire delta + ambient delta + trailer ABC`

Grade and inertia remain outside this ABC. Grade is supplied only to the
effective canonical-engine boundary used for force/profile calculation.

For Fixed Speed and Speed Sweep, acceleration is zero. A mass override does
not change authoritative ABC and therefore does not force the curves apart.
At 0% grade, without a trailer or an active load-dependent rolling
correction, equal baseline and scenario force is the physically correct
result and is explained in the UI.

Resolved rolling/trailer/condition overlays are applied equally to temporary
TOTAL and NET boundaries, preserving the authoritative TOTAL–NET gap.

Grade is applied equally to TOTAL and NET only at calculation time and remains
outside the public temporary ABC.

## Persistence

The Lab has no Save action. Calculation code receives an already-fetched VDE
row and does not call VDE or FuelCons insert/update operations.

## Scope exclusions

Weather APIs, altitude maps, RDE, wind, humidity, J2452 expansion, towing
certification and powertrain capability/thermal/torque modeling remain out of
scope.
