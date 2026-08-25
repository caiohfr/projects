# Sprint 9 - Vehicle Demand Model & Engineering KPIs

## Status

**CLOSED / FROZEN.**

Sprint 9 delivered the canonical Vehicle Demand layer -- contracts,
physics engine, real-scenario hardening, and Comparison Report
integration -- and closes as a stable, reusable capability. This document
is the consolidated freeze artifact; see the individual package docs for
full implementation detail:

- [9A - Canonical Vehicle Demand Contracts](PACKAGE_9A_VEHICLE_DEMAND_CONTRACTS.md)
- [9B - Vehicle Demand Physics Engine](PACKAGE_9B_VEHICLE_DEMAND_ENGINE.md)
- [9C - Real Scenario Validation & Hardening](PACKAGE_9C_VEHICLE_DEMAND_HARDENING.md)
- [9D - Comparison / Energy Drivers Integration](PACKAGE_9D_COMPARISON_ENERGY_DRIVERS_INTEGRATION.md)
- [9E - Polish, Documentation & Freeze](#9e---polish-documentation--freeze) (this document)

See also the technical architecture reference:
[Vehicle Demand Architecture](../architecture/vehicle_demand_architecture.md).

## Scope delivered

```text
Resolved VDE / ComparisonItem
        |
        v
Vehicle Demand adapter(s)
        |
        v
Frozen Vehicle Demand Core (contracts + physics engine)
        |
        v
VehicleDemandResult
        |
        v
Comparison presentation layer
        |
        v
Energy Drivers -- Vehicle Demand Summary
```

- **9A**: `AmbientState`, `Provenance`, `RoadloadBasis`, `EnergyMode`,
  `VehicleDemandRequest`, `VehicleDemandProfile`, `VehicleDemandSummary`,
  `VehicleDemandResult` -- pure, Streamlit-free contracts with a
  JSON-serializable application boundary.
- **9B**: `build_vehicle_demand_profile`/`summarize_vehicle_demand`/
  `calculate_vehicle_demand`, reusing the project's existing canonical VDE
  math (`vde_calc.compute_vde_series`/`extract_cycle_arrays`) rather than
  reimplementing it. New physics limited to air density, known rolling/
  aero force, residual, and `EnergyMode` classification.
- **9C**: `vehicle_demand.adapters.build_vehicle_demand_request` (raw
  `vde_db` row -> `VehicleDemandRequest`), validated against real QA
  scenarios with exact reconciliation against the project's canonical
  on-demand VDE calculation, and hardened against invalid/non-finite
  physical inputs.
- **9D**: `comparison_vehicle_demand_viewmodels.py` (`ComparisonItem` ->
  `VehicleDemandRequest`, a second small adapter), a "Vehicle Demand
  Summary" table and one breakdown chart in the Comparison Report's Energy
  Drivers tab.
- **9E**: UI/wording polish (chart hover/legend, section caption), this
  closure document, the Vehicle Demand architecture reference, roadmap/
  README updates, and final regression + smoke verification.

## Architecture

See [Vehicle Demand Architecture](../architecture/vehicle_demand_architecture.md)
for the full layer diagram, module map, and dependency-direction rules.
Summary: `src/vde_core/vehicle_demand/` (contracts, physics, engine,
serialization, a raw-row adapter) has no Streamlit dependency and does not
import Comparison; `src/vde_app/comparison_vehicle_demand_viewmodels.py`
(Comparison-side) depends on Vehicle Demand, never the reverse.

## Contracts (frozen)

`VehicleDemandRequest`, `VehicleDemandProfile`, `VehicleDemandSummary`,
`VehicleDemandResult`, `RoadloadBasis`, `EnergyMode`, `AmbientState`,
`Provenance` -- unchanged in shape/semantics since Sprint 9A. No field,
type, enum value, or validation rule was added or removed across 9B-9E;
all hardening (9C) was implemented at the engine/physics layer these
contracts are consumed by, never inside the contracts themselves.

## Physics (frozen)

See the "Physical invariants" section of
[Vehicle Demand Architecture](../architecture/vehicle_demand_architecture.md#physical-invariants-frozen)
for the full, authoritative list. Core identity:
`Known Contribution + Residual = Authoritative Roadload`, with Rolling
never inferred from `A`, Aero never inferred from `C`, and TOTAL/NET never
falling back to each other.

## Integration (frozen)

Comparison Report's Energy Drivers tab renders a compact "Vehicle Demand
Summary" table (VDE, Roadload/Known Rolling/Known Aero/Residual energy,
Positive Inertial Work, Positive Tractive Energy, Braking Energy Required)
plus one optional breakdown chart, for the currently-selected Comparison
dataset and TOTAL/NET basis selection. Comparison remains a pure consumer:
no roadload/RRC/CdA/air-density/inertia/tractive/energy physics was ever
written inside `src/vde_app/`.

## Physical invariants

Restated from [Vehicle Demand Architecture](../architecture/vehicle_demand_architecture.md#physical-invariants-frozen)
for a single-glance reference:

1. Authoritative Roadload remains authoritative.
2. Known Contribution + Residual = Authoritative Roadload.
3. Rolling is never inferred from A.
4. Aero is never inferred from C.
5. Residual may be negative (preserved, never `abs()`'d/clipped).
6. Deceleration does not imply braking.
7. Braking Energy Required is not recovered regen.
8. TOTAL and NET never fall back to each other.
9. Zero and missing remain distinct everywhere, including JSON.

## Known limitations

1. Known decomposition initially includes Rolling + Aero only.
2. Residual may contain brake/driveline/bearing/parasitic and other
   unattributed effects -- it is never presented as a specific named
   component.
3. Vehicle Demand does not model powertrain efficiency.
4. Braking Energy Required is wheel-side only.
5. No regen capture model exists yet.
6. `AmbientState` supports Aero-density calculation but not authoritative
   roadload condition correction.
7. `VehicleDemandProfile` is not persisted.
8. Stored synthetic QA `vde_total_mj_per_km`/`vde_net_mj_per_km` values are
   not physical goldens -- see "QA persisted-VDE debt" below.
9. `KinematicPhase` / VSP / driving-aggressiveness classification are
   deferred.

## QA persisted-VDE debt

Restated from Sprint 9C/9D and the architecture doc: QA fixture rows'
stored `vde_total_mj_per_km`/`vde_net_mj_per_km` are demonstrably **not**
derived from that row's own ABC/mass/cycle (verified ~4x off for
VDE-QA-001 against the project's own on-demand recompute). They must never
be treated as physical golden outputs for regression purposes -- the
correct golden is always a fresh on-demand calculation
(`comparison_report_service.resolve_cycle_vde_results` or
`vde_calc.compute_vde_net`) from the same inputs. Fixtures were not
modified in Sprint 9; this is a documented backlog note, not a fix.

## JSON/API/MCP-ready boundary

Documented accurately in
[Vehicle Demand Architecture](../architecture/vehicle_demand_architecture.md#jsonapimcp-ready-boundary-architecture-readiness-only):
the contracts are typed, Streamlit-independent, and JSON-serializable at
the application boundary. **No FastAPI application, MCP server, or agent
tool exists** -- this is architectural readiness only, not a claim of
implementation.

## Test baseline

- Sprint 9 start (post-8E, pre-9A): 1149 tests, 1147 passing, 2 known
  pre-existing failures.
- Sprint 9 close (post-9E): **1268 tests, 1266 passing**, the same 2 known
  pre-existing failures throughout, zero Sprint 9 regressions at any
  package boundary (9A: 1172/1170; 9B: 1196/1194; 9C: 1238/1236;
  9D: 1262/1260; 9E: 1268/1266).
- Known pre-existing failures (unrelated to Sprint 9, not touched by any
  Sprint 9 package): `tests/test_vde_request_resolver.py`, 2 failures
  (component-snapshot/axle-hubs).

## Future interfaces (handoff)

### Next: Sprint 10 - Interactive Quick Scenario

Premise: **do not modify Vehicle Demand physics.** Quick Scenario applies
temporary overrides to an existing resolved scenario and produces a
`VehicleDemandRequest` that flows through the same frozen
`calculate_vehicle_demand()` every other consumer uses.

```text
Existing Scenario
        |
        v
Temporary Overrides
        |
        v
VehicleDemandRequest
        |
        v
Frozen Vehicle Demand Core (unmodified)
        |
        v
VehicleDemandResult
```

Priority override candidates: Mass, CdA, RRC. Ambient conditions
(Temperature, Pressure) can enter through the same architecture once an
owner for authoritative-roadload condition correction is defined -- Sprint
9E does not freeze that UX, only the extension point. The former standalone
"Roadload Condition Scenarios" sprint concept is expected to be absorbed
into this Quick Scenario / derived-scenario architecture rather than
remain an independent capability.

### After that

```text
Sprint 7   Database Management                          CLOSED
Sprint 8   Comparison Report Foundation                  CLOSED
Sprint 9   Vehicle Demand Model & Engineering KPIs        CLOSED
Sprint 10  Interactive Quick Scenario
Sprint 11  Powertrain Scenario L0
Sprint 12  PWT + Comparison Integration
           MVP PRODUCT GATE
```

See the README's Roadmap section for the current top-level state; this
document does not attempt to redesign the full post-MVP roadmap.

## 9E - Polish, Documentation & Freeze

### Files created/modified

Created:
- `docs/architecture/vehicle_demand_architecture.md`
- `docs/sprints/SPRINT_9_VEHICLE_DEMAND_CLOSURE.md` (this file)

Modified:
- `src/vde_app/comparison_report_charts.py` -- explicit `hovertemplate`
  (component name + MJ value) and `legend_title_text="Component"` on the
  Vehicle Demand breakdown chart (Sec 7 hover/legend sanity).
- `src/vde_app/components/comparison_report.py` -- one caption wording
  polish in `_render_vehicle_demand_summary_section` clarifying the
  existing primary-KPIs-first row ordering ("top rows: overall demand...
  lower rows: the roadload explanation behind it") without restructuring
  the table into multiple panels (Sec 5-6: the existing single table
  already communicates well; grouping is conceptual/caption-level, not a
  new visual).
- `README.md` -- new Vehicle Demand product block, updated Stable Product
  Status, new Roadmap section, extended Documentation Index, extended
  Known Limitations.
- `docs/architecture/project_structure.md` -- registered
  `src/vde_core/vehicle_demand/` under the `src/vde_core/` layer
  description.
- `tests/test_comparison_report_charts.py` -- 5 new tests for the
  breakdown chart's barmode/omitted-series/negative-value/hover behavior.
- `tests/test_comparison_report_vehicle_demand_smoke.py` -- enriched
  Smoke A (table density/units/deltas/expander assertions) and added
  Smoke F (scenario-failure isolation as an AppTest smoke case, previously
  only unit-tested).

**No file under `src/vde_core/vehicle_demand/` was touched.** No DB
migration, no new table, no persisted profile.

### UI polish decisions

- **KPI hierarchy**: kept as the single flat table 9D shipped
  (primary-first ordering: VDE, Roadload Energy, Positive Tractive Energy,
  Braking Energy Required, then Known Rolling/Known Aero/Residual/
  Positive Inertial Work), per Sec 5-6's own preference for "the current
  table already communicates better" over multiple panels. The only change
  is a slightly more explicit caption naming the two conceptual groups (top
  = demand outcome, bottom = roadload explanation) so the *ordering*
  itself now has a stated rationale a reader can find in one glance,
  without adding a second visual.
- **Units**: already unambiguous per-cell (`format_value` appends the
  correct unit string -- "0.297 MJ/km" for VDE, "4.704 MJ" for the
  seven absolute-energy rows) since 9D; verified nothing about a bare,
  unlabeled number could read as directly comparable across rows, and this
  matches the existing Scorecard convention of never repeating the unit in
  the row label itself. No change needed.
- **Breakdown chart**: added explicit `hovertemplate` (scenario label +
  component name + MJ value) and a legend title, both small, low-risk
  Plotly-layout-only changes; `barmode="relative"` (not stacked-to-100%,
  not `"stack"`), omitted-not-zero missing components, and the
  Known+Residual=Roadload identity were all re-verified (new dedicated
  chart tests, see below) rather than assumed unchanged.

### Final visible KPI hierarchy

Unchanged from 9D, confirmed adequate:

```text
VDE                                  <- overall demand outcome
Roadload Energy
Positive Tractive Energy
Braking Energy Required

Known Rolling Energy                 <- roadload explanation
Known Aero Energy
Residual / Unattributed Roadload

Positive Inertial Work               <- vehicle dynamics
```

### Breakdown chart final status

`barmode="relative"`; a component absent for every row is omitted as a
series rather than plotted as zero; negative Residual values are
preserved exactly (not clipped); Known + Residual sums exactly to
Roadload Energy (re-verified, Sprint 9D's own dedicated identity test);
hover now explicitly states component name and MJ value; legend now
carries a "Component" title. 5 new dedicated tests
(`VehicleDemandBreakdownChartTests`) lock all of this in as regression
coverage that did not exist before 9E.

### Provenance final approach

Unchanged from 9D: per-cell `warning` text (e.g. "RRC unavailable",
"CdA/air density unavailable", "NET roadload is unavailable for this
scenario") is sufficient for an engineer to discover *why* a given
Vehicle Demand quantity is unavailable, without a second provenance
panel. Technical Scorecard's optional extension (model version/basis/
rolling-aero-ambient status rows) was reviewed again in 9E and
**deliberately still not implemented** -- the marginal audit value did not
clearly outweigh Sec 54/9D's density concern, and no per-cell information
gap was found that would justify it. This remains an available, low-risk,
explicitly-deferred candidate, not a defect.

### TOTAL/NET final verification

Re-run (9C/9D reconciliation and no-fallback tests, plus Smoke C/D): TOTAL
and NET compute independently; NET-unavailable scenarios show `"-"` with
`"NET roadload is unavailable for this scenario."`, never a TOTAL value in
NET's place; a `"Both"` selection renders one table per basis with a clear
`TOTAL`/`NET` caption. No zero-NET-value case was found to collapse into
"unavailable" (zero and missing remain distinct, confirmed by the
Sprint 9C/9D zero-value test suites).

### Reference-less final verification

Confirmed again (Smoke B, plus the existing unit test suite): every
Vehicle Demand row/cell continues to render correctly with `dataset.
reference is None`, showing absolute values with no delta/formatted_delta,
and no exception.

### Partial-decomposition final behavior

Confirmed (Smoke E): a scenario missing RRC and/or CdA shows `"Known
Rolling Energy" -> "-"` / `"Known Aero Energy" -> "-"` with their specific
short reasons, while VDE/Roadload Energy/Positive Tractive/Braking Energy
Required remain fully computed and visible -- the wording never implies
the whole calculation is incomplete, only that a specific explanatory
component is unavailable.

### Negative residual final behavior

Confirmed (existing 9D unit tests, re-run in 9E): shown as-is (never
`abs()`'d), with a discreet `" (Review)"` suffix on the value text itself
(so it survives even alongside a Reference delta) and a short warning
sentence. "Review" is a discreet presentation flag, not a fatal error --
the row, and every other KPI for that scenario, continues to render
normally.

### Scenario failure isolation (final)

Confirmed with a new AppTest smoke case (Smoke F, not present in 9D's
smoke suite, which only unit-tested this): a scenario with physically
invalid data (zero mass) renders `"-"` cells with a short, human-readable
reason, no `"Traceback"`/`"ValueError"` text anywhere on the page, and
every other, valid scenario in the same comparison continues to show real
values.

### Cleanup performed

None beyond the chart/caption polish above -- no unused import, dead
helper, or duplicate presentation code from Sprint 9A-9D was found during
this review that met the "small, local, behavior-preserving" bar for
removal (Sec 42). No broad cleanup was attempted.

### Tests added

- `tests/test_comparison_report_charts.py::VehicleDemandBreakdownChartTests`
  -- 5 tests (empty rows, barmode, omitted-not-zero, negative-value
  preservation, hover/axis unit explicitness).
- `tests/test_comparison_report_vehicle_demand_smoke.py` -- Smoke A
  enriched with table-density/unit/expander assertions; new Smoke F
  (scenario failure isolation), 1 new test.

### Focused test results

`test_vehicle_demand_contracts` + `test_vehicle_demand_engine` +
`test_vehicle_demand_integration` + `test_comparison_vehicle_demand_
viewmodels` + `test_comparison_report_vehicle_demand_smoke` +
`test_comparison_report_charts`: **150 tests, all passing.**

### Smoke A-F results

All 6 AppTest smoke cases pass against real QA fixtures via
`streamlit.testing.v1.AppTest` (no browser tool available in this
environment, consistent with every prior Comparison package):

- **A** (Reference + 2 comparisons): heading present; primary KPI labels
  present; MJ/km and MJ units both present and distinguishable; breakdown
  expander present.
- **B** (Reference-less): no crash; heading present.
- **C** (TOTAL/NET switch): both bases render; captions present for
  `"Both"`.
- **D** (NET unavailable): `"unavailable"` text present, no crash.
- **E** (partial decomposition): no crash; VDE still renders;
  RRC/unavailable language present.
- **F** (scenario failure isolation, new in 9E): valid scenario's MJ/km
  value present; invalid scenario shows `"-"`; no traceback/exception text
  anywhere; a short mass/unavailable-related reason is present.

### Full test count/result

Full suite (`python -m unittest discover -s tests`) after all 9E changes:
**1268 tests, 1266 passing** (1262 post-9D + 6 new: 5 breakdown-chart
tests + 1 AppTest Smoke F), the same 2 known pre-existing failures, **zero
new regressions** (all Sprint 9A-9D regression baselines were already
zero-regression at their own checkpoints, and this package adds only new,
additive tests plus two small presentation-layer edits already covered by
dedicated new tests above).

### Known pre-existing failures

Unchanged: `tests/test_vde_request_resolver.py`, 2 failures (component-
snapshot/axle-hubs). Not touched by Sprint 9 at any point.

### Commit(s)

`600143cb` on branch `sprint-9a-vehicle-demand-contracts` - "docs(vehicle-demand):
polish, document, and freeze Sprint 9". Sprint 9's full commit sequence on
this branch: `cae884c0`/`91913320` (9A), `66f1bc7a`/`1b3dd9cf` (9B),
`d35b994a`/`9f474ac5` (9C), `409902af`/`3f359d85` (9D), `600143cb` (9E).

### Post-freeze auditability hotfix

Found immediately after the 9E freeze: `_render_section`'s pre-existing
row-visibility rule (drop a row entirely when no cell in it is available)
silently hid the "Known Aero Energy" row on every real comparison, because
no Comparison-sourced `VehicleDemandRequest` has ever supplied ambient
data -- Aero is unavailable for every scenario today, not just some. Fixed
by adding a minimal, opt-in per-row `RowVisibility` (`AUTO`, default,
byte-for-byte legacy behavior; `ALWAYS`, for basic/canonical engineering
audit information -- "unavailable is information," never hidden). Applied
to all 8 Vehicle Demand Summary rows and to 11 existing Registry metrics
(Mass, CdA, RRC, A/B/C TOTAL, A/B/C NET, VDE TOTAL, VDE NET) via a new
`MetricDefinition.always_visible` flag. No `src/vde_core/vehicle_demand`
physics changed; no TOTAL/NET fallback introduced. 16 new tests plus one
new AppTest smoke case. Commit `4f0adfae`. Full detail in that commit's
message; this paragraph is the closure-doc pointer to it.

### Freeze statement

```text
Vehicle Demand Core API                    FROZEN
Vehicle Demand Comparison integration      FROZEN
Sprint 9 - Vehicle Demand Model & Engineering KPIs     CLOSED / FROZEN
```

Future changes to `src/vde_core/vehicle_demand/` are permitted only for:
a genuine bug, a physical-contract error, or a strictly-necessary
integration need -- never routine UI/feature work in a downstream
consumer.

Sprint 10 (Interactive Quick Scenario) can start without modifying Vehicle
Demand Core: its entire job is producing a `VehicleDemandRequest` from a
temporarily-overridden resolved scenario and consuming the same frozen
`calculate_vehicle_demand()`.
