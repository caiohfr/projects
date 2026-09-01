# Sprint 11 — Multi-Domain System Scenario Foundation

**Target repository path:** `docs/specs/sprint_11_multi_domain_system_scenario.md`
**Status:** Draft approved for Sprint 11 implementation
**Implementation sequence:** 11A → 11B → 11C → 11D → 11E

---

## 1. Problem / Motivation

The current Powertrain Scenario already contains useful capabilities:

- VDE/source selection;
- baseline/reference powertrain;
- PSE;
- physics-simple / Energy Balance calculations;
- ML;
- Regression;
- Benchmark/reference PSE;
- Technology Delta;
- preview/result;
- existing persistence.

However, scenario ownership is still excessively concentrated in the Streamlit layer and the product is organized mainly around estimation methods rather than around the physical system.

The current model does not explicitly represent the vehicle/powertrain as a reusable composition of physical domains.

Sprint 11 must establish a canonical multi-domain System Scenario architecture that supports:

- Current;
- Proposal A;
- Proposal B;
- Proposal C;

where each scenario may independently combine different states/proposals for:

1. Vehicle Demand;
2. Propulsion Architecture;
3. Engine / Fuel Converter;
4. Transmission / Driveline;
5. Electric Drive;
6. Energy Storage;
7. Energy Management / Controls;
8. Auxiliaries / Thermal.

These domains are parts of an interacting system, not a simple additive stack.

Sprint 11 creates the architectural foundation without implementing high-fidelity simulation.

---

## 2. Objective

Transform the current Powertrain Scenario into a canonical multi-domain capability:

```text
Domain States / Proposals
        ↓
System Scenario Composition
        ↓
Resolved System Scenario
        ↓
Energy Balance L0 Adapter
        ↓
existing canonical fuel estimation
        ↓
System Scenario Result
```

The implementation must provide:

- Streamlit-independent canonical contracts;
- Current / Correction / Effective Current semantics;
- reusable Domain Proposals;
- independent System Scenario composition;
- explicit readiness;
- explicit provenance;
- explicit fidelity coverage;
- reuse of the existing L0 calculation backend;
- a thin UI over the canonical core.

No new physical formula should be introduced merely to populate the new architecture.

---

## 3. Current-State Audit Requirement

Before implementation, inspect the actual current owners in the repository.

At minimum inspect:

- `pages/Powertrain_Scenario.py`
- `src/vde_app/components/pwt_fuel_energy.py`
- `src/vde_core/fuel_estimation.py`
- `src/vde_core/powertrain_efficiency.py`
- `src/vde_core/pwt_fuel_energy_service.py`
- `src/vde_core/technology_delta.py`
- ML services used by Powertrain
- Regression services used by Powertrain
- current FuelCons save path
- Vehicle Demand contracts/adapters
- relevant tests

The repository state is authoritative.

Document:

- current scenario ownership;
- canonical calculation owners;
- baseline/reference owner;
- PSE owner;
- Benchmark owner;
- ML owner;
- Regression owner;
- Technology Delta owner;
- persistence/save owner;
- Vehicle Demand input path.

Do not refactor a module merely because it is large.

---

## 4. Target Architecture

```text
Current DBs / canonical Vehicle Demand
            ↓
        Adapters
            ↓
      Domain Source States
            ↓
 Corrections / Effective Current
            ↓
       Domain Proposals
            ↓
 SystemScenarioDefinition
            ↓
     System Resolver
            ↓
 ResolvedSystemScenario
            ↓
  EnergyBalanceL0Adapter
            ↓
 existing FuelEstimateRequest
            ↓
 existing run_fuel_estimation()
            ↓
  SystemScenarioResult
```

UI dependency direction:

```text
Streamlit
   ↓
canonical contracts
   ↓
resolvers / services
   ↓
canonical result
   ↓
presentation
```

Forbidden:

```text
widget
→ local physical formula
→ result/chart
```

---

## 5. Canonical Domain Taxonomy

Sprint 11 recognizes these domains:

1. **Vehicle Demand**
2. **Propulsion Architecture**
3. **Engine / Fuel Converter**
4. **Transmission / Driveline**
5. **Electric Drive**
6. **Energy Storage**
7. **Energy Management / Controls**
8. **Auxiliaries / Thermal**

This taxonomy is intended to remain stable while domain fidelity increases in later sprints.

Not every domain must quantitatively affect Energy Balance L0.

---

## 6. Propulsion Architecture

Sprint 11 represents propulsion architecture as classification only:

- ICE
- MHEV
- HEV
- PHEV
- BEV

Existing useful topology metadata may be preserved when already available.

Sprint 11 must **not** implement:

- topology graph;
- ports/connections;
- topology editor;
- P0/P1/P2/P3/P4 graph semantics;
- acausal connections;
- graph/system compiler.

Those are future capabilities.

---

## 7. Domain State Semantics

Every physical domain conceptually distinguishes:

```text
SOURCE
    ↓
CURRENT
    ↓
CORRECTION
    ↓
EFFECTIVE CURRENT
    ↓
PROPOSAL
```

### Source

Imported or otherwise authoritative source data.

### Current

The current interpreted domain state.

### Correction

An explicit engineering correction to the Current state.

### Effective Current

The corrected state that becomes the baseline for proposals.

### Proposal

An alternative domain configuration based on Effective Current.

Corrections and proposals must never mutate Source.

---

## 8. Proposal Lineage

Sprint 11 deliberately does **not** implement Proposal → Proposal inheritance.

Every Domain Proposal is based on the domain's Effective Current.

Example:

```text
Transmission Effective Current
├── TRANS-P01
├── TRANS-P02
└── TRANS-P03
```

Not:

```text
Current → P01 → P02 → P03
```

This supersedes older Powertrain guidance that suggested VDE-style `Walk From` inheritance.

System composition and presentation walk are separate concepts.

---

## 9. Domain Proposal vs System Scenario

A Domain Proposal is reusable and domain-specific.

Examples:

- `ENG-P01`
- `TRANS-P02`
- `BAT-P01`

A System Scenario composes selected domain states/proposals.

Example:

```text
SYS-P01

Vehicle Demand   = VDE-P04
Architecture     = MHEV
Engine           = ENG-P01
Transmission     = TRANS-P02
Electric Drive   = EM-P01
Energy Storage   = BAT-P01
Controls         = CTRL-P01
Aux/Thermal      = AUX-CURRENT
```

A Domain Proposal must never be treated as a complete System Scenario.

---

## 10. System Scenario Matrix

Initial UI scope:

- Current
- Proposal A
- Proposal B
- Proposal C

Maximum: Current + 3 proposals.

Conceptual composition:

```text
                       Current      Proposal A      Proposal B      Proposal C
────────────────────────────────────────────────────────────────────────────
Vehicle Demand         VDE-CUR      VDE-P01         VDE-P02         VDE-P03
Architecture           ICE          MHEV            HEV             BEV
Engine                 Current      ENG-P01         ENG-P01         N/A
Transmission           Current      TRANS-P01       TRANS-P02       TRANS-P03
Electric Drive         N/A          EM-P01          EM-P02          EM-P03
Energy Storage         N/A          BAT-P01         BAT-P02         BAT-P03
Controls               Current      CTRL-P01        CTRL-P02        CTRL-P03
Aux/Thermal            Current      Current         AUX-P01         AUX-P02
```

Each column is an independent, complete System Scenario.

Proposal B does not inherit Proposal A.

---

## 11. Vehicle Demand Semantics

Vehicle Demand is one selectable domain/input of each System Scenario.

Each System Scenario has exactly one resolved Vehicle Demand source.

Different System Scenarios may use different VDE / Vehicle Demand results.

Example:

```text
Current     → VDE-CURRENT
Proposal A  → VDE-LIGHTWEIGHT
Proposal B  → VDE-AERO-TIRE
```

Powertrain must not recalculate:

- roadload;
- mass;
- Tire/RRC;
- aero;
- Vehicle Demand physics.

It consumes the canonical Vehicle Demand result/profile.

---

## 12. Configuration vs Model Assumption vs Result

These concepts must remain distinct.

### Configuration

Examples:

- engine displacement;
- engine rated power;
- transmission type;
- final drive;
- motor rated power;
- battery capacity.

### L0 model assumptions

Examples:

- effective fuel-path efficiency;
- effective electric-path efficiency;
- utility factor / electric-share assumption.

### Results

Examples:

- PSE;
- fuel consumption;
- electrical consumption;
- CO₂;
- fuel/electrical input energy.

Never reinterpret:

```text
PSE = Engine efficiency
```

They are not equivalent.

---

## 13. Engine / Fuel Converter Domain

Configuration may include, when available:

- fuel type;
- engine/family identifier;
- displacement;
- rated power;
- rated torque;
- technology descriptors.

At L0, configuration changes alone must not invent a consumption impact.

Example:

```text
2.0 L → 1.6 L
```

does not automatically imply an efficiency improvement.

A quantitative L0 effect may exist only through an explicit supported assumption, Technology Delta, or existing canonical estimation path.

Future maps/operating-point models are out of scope.

---

## 14. Transmission / Driveline Domain

Configuration may include:

- transmission type;
- transmission/model identifier;
- gear count;
- final drive;
- other currently supported canonical metadata.

Changing configuration alone does not imply a quantitative consumption improvement at L0.

An explicit supported effective assumption / Technology Delta may represent a quantitative L0 impact.

Out of scope:

- gear schedule;
- gear-ratio simulation;
- speed/load efficiency maps;
- operating points.

---

## 15. Electric Drive Domain

Configuration may include:

- motor role/type;
- motor count;
- position when known;
- rated/peak power;
- rated/peak torque;
- nominal voltage;
- existing motor/inverter identifiers.

L0 may consume the already-supported effective electric-path efficiency.

Power/torque metadata must not automatically create performance or energy effects if the L0 solver does not model those constraints.

No motor/inverter maps.

---

## 16. Energy Storage Domain

Configuration may include:

- gross capacity;
- usable capacity;
- nominal voltage;
- charge/discharge power limits;
- regen power limits;
- SOC-window metadata.

At Sprint 11 L0, much of this may be:

```text
CONFIGURATION_ONLY
```

Changing battery capacity must not silently alter consumption unless the existing canonical L0 model explicitly supports that dependency.

Out of scope:

- SOC trace;
- electrochemical model;
- thermal battery model.

---

## 17. Energy Management / Controls Domain

Configuration may include existing metadata for:

- hybrid operating strategy;
- utility factor / electric share;
- regen/control metadata;
- start-stop / engine-off metadata;
- calibration/technology assumptions.

L0 may quantitatively represent only assumptions already supported by the canonical model.

Out of scope:

- torque-split simulation;
- SOC supervisory control;
- hybrid controller simulation.

---

## 18. Auxiliaries / Thermal Domain

This domain exists from Sprint 11 so the architecture does not require redesign later.

Current data may be sparse.

Possible initial fidelity states include:

- `CONFIGURATION_ONLY`
- `NOT_REPRESENTED`

A fixed/effective auxiliary load may become quantitative only if an existing canonical model already supports it.

Do not add new thermal physics merely to populate the domain.

---

## 19. Physical Proposal + Fidelity-Specific Representation

A Domain Proposal may contain two separate parts:

```text
Physical / Configuration Proposal
+
optional fidelity-specific quantitative representation
```

Example:

```text
TRANS-P01

Configuration
8AT → 9AT
FDR 3.73 → 3.45

L0 Representation
Effective improvement: +0.8%
Source: Engineering assumption
```

The `+0.8%` is explicit.

It must not be inferred from gear count or final drive.

---

## 20. L0 Assumptions Are System-Level

Aggregate L0 parameters must not be mislabeled as component-specific efficiencies when they represent the complete energy path.

Conceptually:

```text
EnergyBalanceL0Assumptions

fuel_path_effective_efficiency
electric_path_effective_efficiency
utility_factor / electric share
other currently-supported aggregate assumptions
```

These belong to the L0 system-model boundary.

Exact naming should follow the current canonical `FuelEstimateRequest` semantics after repository audit.

Do not create duplicate meanings just to rename fields.

---

## 21. PSE

PSE remains a system-level derived/effective metric.

Current/Benchmark/ML/Regression/Manual/Technology mechanisms are not separate physical domains.

They are assumption/evidence resolution mechanisms or recommendation mechanisms.

Where existing behavior is preserved, these paths must continue to use canonical owners:

- Current;
- Benchmark;
- ML;
- Regression;
- Engineering/manual;
- Technology Delta.

No duplicate PSE formula.

No new ML model.

---

## 22. Technology Delta

`technology_delta.py` remains the canonical owner of existing Technology Delta calculation semantics.

Sprint 11 must not create another delta-stacking implementation.

Domain Proposals may associate a Technology Delta with the physical domain it represents.

Before implementing multi-domain quantitative composition:

- inspect current stacking/order semantics;
- preserve them exactly;
- ensure deterministic explicit ordering;
- do not depend accidentally on dict or UI rendering order.

If multi-domain composition requires a genuinely new physical/math rule, stop and report.

Do not invent the rule.

---

## 23. Fidelity Manifest

Every Resolved System Scenario must explicitly report what the selected solver actually represents.

Minimum semantics:

- `QUANTITATIVE`
- `EFFECTIVE_ASSUMPTION`
- `CONFIGURATION_ONLY`
- `NOT_REPRESENTED`

Example:

```text
Vehicle Demand       QUANTITATIVE
Engine               EFFECTIVE_ASSUMPTION
Transmission         EFFECTIVE_ASSUMPTION
Electric Drive       EFFECTIVE_ASSUMPTION
Energy Storage       CONFIGURATION_ONLY
Controls             EFFECTIVE_ASSUMPTION
Aux/Thermal          NOT_REPRESENTED
```

The manifest answers:

> Did this domain actually influence this result at this fidelity?

---

## 24. Missing Data and Readiness

Missing configuration data does not automatically block a System Scenario.

Readiness is solver-dependent.

Example:

```text
Engine torque = missing
```

may be acceptable for Energy Balance L0 if torque is not required.

A future operating-point solver may require it.

Therefore distinguish:

- domain data completeness;
- solver readiness.

Do not silently impute fields unless an existing canonical estimation path owns that behavior and provenance is preserved.

---

## 25. Canonical Contracts

The implementation must define Streamlit-independent contracts equivalent to:

- Domain State / Effective Domain State;
- Domain Correction;
- Domain Proposal;
- `SystemScenarioDefinition`;
- `ResolvedSystemScenario`;
- `FidelityManifest`;
- `SystemScenarioResult`.

Exact class/helper decomposition is implementation freedom.

### SystemScenarioDefinition

Contains scenario composition/reference intent.

It is not a result.

### ResolvedSystemScenario

Contains the immutable/effective snapshot required by the solver.

The solver must not query Streamlit/session state during calculation.

### SystemScenarioResult

Contains calculation results and audit metadata.

It must be consumable later by Comparison without recalculating Powertrain.

---

## 26. System Scenario Identity

Identity must not be based on visible scenario name.

Current and each Proposal require stable identities.

Visible names may change without changing identity.

Maximum initial UI scope:

```text
Current
Proposal A
Proposal B
Proposal C
```

Do not build arbitrary-N scenario management.

---

## 27. Effective Current

Corrections belong to each domain.

A Proposal starts from that domain's Effective Current.

Example:

```text
Source Transmission
FDR 3.73

Correction
3.73 → 3.70

Effective Current
3.70

TRANS-P01
based on 3.70
```

A correction must not mutate the source record.

---

## 28. L0 Solver Reuse

Sprint 11 does not create a new Energy Balance solver if the existing `fuel_estimation` core already owns the calculation.

Preferred architecture:

```text
ResolvedSystemScenario
        ↓
EnergyBalanceL0Adapter
        ↓
existing FuelEstimateRequest
        ↓
existing run_fuel_estimation()
        ↓
SystemScenarioResult
```

Do not duplicate:

- LHV;
- CO₂ factors;
- fuel-energy math;
- BEV effective-energy math;
- PHEV utility-factor math;
- canonical PSE math;
- Technology Delta math.

---

## 29. Existing Capability Preservation

Sprint 11 is a reorganization/foundation sprint, not a feature-deletion sprint.

Audit and preserve current capabilities where semantically valid:

- current/reference baseline;
- manual assumptions;
- observed/derived PSE;
- benchmark reference;
- ML;
- Regression;
- Technology Delta;
- fuel/energy result;
- provenance;
- existing save behavior where compatible.

If an existing feature conflicts with the new physical contract, stop and report rather than silently preserving invalid behavior.

---

## 30. Database Constraint

Sprint 11 must **not redesign the database**.

Do not:

- create `vehicle_db`;
- create test-set tables;
- create detailed legislation/result tables;
- rebuild `fuelcons_db`;
- remove legislation columns;
- normalize engine/transmission/battery into new physical tables;
- implement the future scorecard/data rebuild.

These decisions belong to Sprint 12.

Current DBs may be read through adapters/repositories.

New domain contracts must not expose raw `fuelcons_db` row layout as their canonical API.

Preferred:

```text
legacy fuelcons row
        ↓
adapter
        ↓
canonical Domain State
```

Sprint 12 can later replace storage/adapters without rewriting the System Scenario contracts.

---

## 31. Persistence Constraint

Do not create new database schema merely to persist Domain Proposals.

Reusable Domain Proposal persistence is not required in Sprint 11.

Existing Powertrain result/save behavior must not regress.

If the current persistence path can safely save an L0 result and preserve required scenario/provenance snapshot without schema redesign, reuse it.

If correct reproducibility requires new schema semantics, stop and report.

Do not introduce hidden new persistence architecture merely to satisfy a checkbox.

---

## 32. Comparison Boundary

Sprint 11 does not redesign Comparison Report.

It only ensures that `SystemScenarioResult` is a clean future input for Comparison.

Do not implement chart-specific Powertrain physics.

Broad System Scenario → Comparison integration belongs to the next sprint.

---

## 33. UX

Reorganize the Powertrain page around System Scenario composition.

Primary mental model:

```text
                     Current   Proposal A   Proposal B   Proposal C
Vehicle Demand
Architecture
Engine
Transmission
Electric Drive
Energy Storage
Controls
Aux / Thermal
```

The matrix represents composition, not all scalar inputs.

Domain details should use a compact secondary editor.

Avoid placing dozens of scalar widgets directly in the matrix.

No VDE-style `Walked #1/#2` language.

No Apply-per-field workflow.

Do not reproduce unnecessary VDE Setup complexity.

---

## 34. Presentation Walk Is Separate

System composition does not define Presentation Walk.

Keep distinct:

1. domain proposal lineage;
2. System Scenario composition;
3. Comparison/presentation ordering.

Sprint 11 must not infer:

```text
Current → Proposal A → Proposal B
```

as physical lineage.

Comparison may later present scenarios in any selected order.

---

## 35. Provenance

Preserve distinctions such as:

- SOURCE / OBSERVED;
- CORRECTED;
- ASSUMED;
- CALCULATED;
- ESTIMATED;
- ML-PREDICTED / ML-DERIVED.

Do not silently collapse them.

Every L0 result should retain enough information to identify:

- Vehicle Demand source;
- System Scenario identity;
- effective assumptions;
- model/solver kind;
- Fidelity Manifest;
- important warnings;
- provenance.

---

## 36. Explicit Zero

Across all new contracts:

```text
0
```

is explicit.

It must not mean:

- missing;
- inherit;
- not requested.

Zero semantics must remain field-specific and explicit.

---

# 37. Invariants

### INV-11-001 — Neutral

Same domain states + same assumptions → same L0 result.

### INV-11-002 — Determinism

Same resolved scenario + same model version → same result.

### INV-11-003 — Source immutability

Corrections/proposals never mutate imported/current source records.

### INV-11-004 — Proposal independence

Proposal A and Proposal B resolve independently.

Proposal B does not inherit Proposal A.

### INV-11-005 — Domain reuse

The same Domain Proposal may be selected by multiple System Scenarios without mutation.

### INV-11-006 — Vehicle Demand separation

Selecting/resolving a System Scenario never recalculates roadload/VDE physics inside Powertrain.

### INV-11-007 — Fidelity honesty

A configuration-only field change cannot change quantitative results unless represented by the selected model.

### INV-11-008 — Efficiency direction

For otherwise identical supported L0 scenarios, higher effective efficiency must reduce required input energy.

### INV-11-009 — Lower efficiency

For otherwise identical supported L0 scenarios, lower effective efficiency must increase required input energy.

### INV-11-010 — Recommendation separation

ML/Benchmark/Technology recommendation not adopted → deterministic final result does not change.

### INV-11-011 — Provenance

Assumed/estimated values must preserve that provenance in downstream results.

### INV-11-012 — DB independence

Canonical System Scenario contracts must not require callers to understand raw `fuelcons_db` schema.

---

# 38. Functional Requirements

### REQ-11-001
The system shall expose Current plus up to three independent System Scenario Proposals.

### REQ-11-002
Each System Scenario shall independently select its Vehicle Demand source.

### REQ-11-003
Two System Scenarios may use different VDE/Vehicle Demand results.

### REQ-11-004
The system shall represent all eight approved domains.

### REQ-11-005
Architecture shall support ICE/MHEV/HEV/PHEV/BEV classification.

### REQ-11-006
Each Domain Proposal shall originate from Effective Current.

### REQ-11-007
Domain Proposal → Domain Proposal inheritance shall not exist in Sprint 11.

### REQ-11-008
Domain Proposals shall be reusable across multiple System Scenarios.

### REQ-11-009
Configuration and L0 quantitative representation shall remain distinct.

### REQ-11-010
Unsupported configuration changes shall remain configuration-only rather than creating invented quantitative impacts.

### REQ-11-011
The resolver shall produce a canonical `ResolvedSystemScenario` before simulation.

### REQ-11-012
`ResolvedSystemScenario` shall include a Fidelity Manifest.

### REQ-11-013
Solver readiness shall be distinguished from general data completeness.

### REQ-11-014
Energy Balance L0 shall reuse the existing canonical calculation path.

### REQ-11-015
PSE shall remain system-level and shall not be renamed/reinterpreted as engine efficiency.

### REQ-11-016
Existing canonical Technology Delta calculation semantics shall be reused.

### REQ-11-017
Existing valid PSE/Benchmark/ML/Regression/manual capabilities shall be reused rather than duplicated.

### REQ-11-018
`SystemScenarioResult` shall preserve model/fidelity/provenance metadata.

### REQ-11-019
The Powertrain UI shall contain no independent physical formula.

### REQ-11-020
Sprint 11 shall not perform database schema redesign.

### REQ-11-021
Existing source DB rows shall remain immutable.

### REQ-11-022
System Scenario identity shall be independent from visible scenario name.

### REQ-11-023
Editing Proposal A shall not mutate Proposal B or Current.

### REQ-11-024
Comparison/presentation order shall not determine System Scenario resolution.

### REQ-11-025
Existing compatible Powertrain save/result behavior shall not regress.

---

# 39. Acceptance Cases

## Case A — Neutral Current

Given a valid Current source and Current Vehicle Demand, resolving Current without corrections/proposals must match existing canonical baseline behavior within the existing numerical tolerance.

## Case B — Two independent VDE proposals

Current uses VDE-001, Proposal A uses VDE-002, Proposal B uses VDE-003.

Each must consume its own Vehicle Demand.

No scenario inherits another scenario's VDE.

## Case C — Shared Engine Proposal

Proposal A and Proposal B both select `ENG-P01`.

Both reference equivalent Engine configuration without mutating `ENG-P01`.

Editing another domain in Proposal A must not mutate Proposal B.

## Case D — Transmission configuration only

Current = 8AT.

`TRANS-P01` = 9AT with changed final drive.

No quantitative L0 assumption is supplied.

Configuration changes must be visible.

Fidelity must mark unsupported quantitative effect appropriately.

The solver must not invent a fuel-consumption improvement.

## Case E — Transmission with explicit L0 effect

`TRANS-P02` carries a supported explicit effective improvement assumption.

The quantitative impact must flow only through the canonical L0/Technology Delta path.

No local Transmission formula may be created.

## Case F — Battery capacity change

`BAT-P01` changes usable capacity.

If Energy Balance L0 does not model battery-size impact:

- Battery proposal resolves;
- Fidelity = `CONFIGURATION_ONLY`;
- no artificial consumption change occurs.

## Case G — Higher effective efficiency

Two otherwise identical L0 scenarios differ only by a supported higher effective efficiency.

Required input energy must decrease.

## Case H — ML recommendation not adopted

ML produces a recommendation but it is not adopted.

Final deterministic result remains unchanged.

## Case I — Missing future-only data

Engine torque is unavailable.

If Energy Balance L0 does not require engine torque, domain metadata may be incomplete while solver remains READY.

## Case J — Proposal isolation

Proposal A and Proposal B are both resolved.

Changing Proposal A's Engine selection must not alter Proposal B.

## Case K — Source correction

A source final-drive value is explicitly corrected.

Source remains unchanged.

Effective Current contains the corrected value.

Transmission proposals start from Effective Current.

## Case L — Architecture applicability

For BEV, Fuel Converter may be N/A and required electric domains are evaluated according to L0 readiness.

Equivalent architecture applicability rules must work for ICE/MHEV/HEV/PHEV without inventing invalid domain requirements.

---

# 40. Test Strategy

Sprint closure evidence must classify each requirement as one of:

- **DIRECT TEST**
- **INDIRECT CANONICAL COVERAGE**
- **INSPECTION**
- **MANUAL SMOKE**
- **GAP**

Do not inflate test counts.

## Contract tests

Cover:

- explicit zero != missing;
- stable identities;
- max-three-proposal constraint;
- domain applicability;
- Effective Current;
- proposal base rules;
- fidelity states;
- serialization/roundtrip where required by project conventions.

## Resolver tests

Cover:

- Current;
- corrections;
- independent proposals;
- different VDE per proposal;
- shared Domain Proposal;
- readiness;
- configuration-only behavior;
- provenance.

## L0 parity tests

Critical requirement.

Equivalent inputs through:

```text
new System Scenario
→ L0 Adapter
```

must match the existing canonical:

```text
FuelEstimateRequest
→ run_fuel_estimation()
```

Do not duplicate formulas in expected test values when canonical parity can be used.

## Technology Delta ownership tests

Verify Sprint 11 uses the canonical Technology Delta owner.

No third stacking implementation.

## UI tests

Use Streamlit `AppTest` where practical for:

- Current + one Proposal;
- Current + three Proposals;
- independent VDE selection;
- Domain Proposal selection;
- configuration-only indication;
- readiness;
- stable scenario identity;
- no Proposal inheritance.

## Full regression

Run the complete suite.

Known failures must be classified from actual baseline evidence, not copied blindly from old docs.

## Manual smoke

Required before Sprint 11 closure.

`AppTest != manual browser smoke`.

---

# 41. Manual Browser Smoke

At minimum verify:

### Smoke A — Current parity
Resolve Current and confirm baseline result/provenance.

### Smoke B — Multi-domain Proposal
Proposal A uses a different VDE plus Engine and Transmission proposals.

Confirm composition and result.

### Smoke C — Independent Proposals
Create A and B with different VDE/domain combinations.

Edit A.

Confirm B does not change.

### Smoke D — Configuration-only
Change Battery capacity or another unsupported L0 configuration.

Confirm UI shows that it is not quantitatively represented.

Confirm result does not falsely change.

### Smoke E — Existing methods
Exercise available Current/Benchmark/ML/Regression/Technology paths enough to prove current capability was not lost.

### Smoke F — Three proposals
Current + A + B + C.

Confirm state remains understandable and stable.

---

# 42. Non-Goals

Sprint 11 must not implement:

- database redesign;
- `vehicle_db`;
- detailed test-set / legislation tables;
- dataset rebuild;
- Comparison Report redesign/integration;
- topology graph;
- physical ports/connections;
- P0/P1/P2/P3/P4 graph engine;
- engine/BSFC maps;
- motor/inverter maps;
- transmission maps;
- gear schedule;
- operating points;
- SOC trace;
- battery electrochemical model;
- battery thermal model;
- hybrid supervisory control;
- torque-split simulation;
- new regen trace model beyond existing canonical behavior;
- new ML model;
- new Regression model;
- optimizer;
- DOE;
- FMU/FMI;
- agent/RAG/MCP;
- arbitrary-N generic scenario framework;
- generic dynamic-domain/plugin framework.

Do not build future infrastructure merely because it may someday be useful.

---

# 43. Sprint 12 Compatibility Note

Sprint 12 is expected to revisit the dataset/data contract.

Current conceptual direction may eventually include:

- vehicle identity separated from VDE/FuelCons;
- FuelCons retained as the primary scorecard/query surface;
- detailed test information separated where useful.

Sprint 11 does not implement any of this.

It only ensures:

```text
DB row
→ adapter
→ canonical Domain State
```

so future storage changes do not rewrite the System Scenario contracts.

---

# 44. SDD Pilot Requirements

Sprint 11 is the first formal SDD pilot.

Required:

1. Create/update root `AGENTS.md` with permanent EcoDrive rules only.
2. Keep Sprint-specific requirements in this spec.
3. Create one reusable sprint implementation skill if the repository does not already have one.
4. Do not create a Sprint-11-specific skill.
5. Review SDD usefulness at Sprint 11 closure.

Permanent instructions belong in `AGENTS.md`.

Sprint-specific requirements belong here.

---

# 45. Agent Autonomy

The coding agent may autonomously:

- inspect repository architecture;
- choose local module/file decomposition;
- define internal helper names;
- extract Streamlit-free services;
- move UI-owned orchestration into canonical service/core ownership;
- add focused tests;
- fix ordinary regressions introduced by Sprint 11;
- update documentation;
- perform small local refactors required to eliminate duplicate ownership.

The agent should not stop for routine code-organization questions.

---

# 46. Stop / Escalation Conditions

Stop and report before proceeding if implementation requires:

1. a new physical formula not already canonical or explicitly defined;
2. reinterpretation of PSE as component efficiency;
3. new Technology Delta stacking math;
4. new Vehicle Demand / roadload physics;
5. database schema migration;
6. dataset rebuild;
7. new persistent Domain Proposal architecture;
8. Comparison redesign;
9. Proposal → Proposal lineage;
10. topology graph/solver;
11. significant deletion of existing valid Powertrain capability;
12. silent provenance semantic change;
13. replacing an existing canonical owner with a competing implementation;
14. a new rule for combining multiple domain L0 impacts that cannot be expressed using existing canonical semantics.

For item 14 especially: do not guess.

Report the exact conflict and evidence.

---

# 47. Definition of Done

## Architecture

- [ ] Canonical multi-domain contracts exist.
- [ ] UI is not scenario owner.
- [ ] Vehicle Demand remains external/canonical.
- [ ] Current + up to three independent Proposals are supported.
- [ ] No Proposal → Proposal inheritance exists.
- [ ] Different VDE per System Scenario works.

## Domains

- [ ] All eight approved domains are represented.
- [ ] Configuration is separate from L0 representation.
- [ ] Effective Current exists.
- [ ] Domain Proposals are reusable.
- [ ] Architecture applicability is handled.
- [ ] Unsupported fields remain honest/configuration-only.

## Fidelity

- [ ] Energy Balance L0 remains the current canonical solver.
- [ ] Fidelity Manifest exists.
- [ ] Domain completeness is distinct from solver readiness.
- [ ] No false quantitative impacts are introduced.

## Existing capability

- [ ] PSE semantics are preserved.
- [ ] Technology Delta canonical ownership is preserved.
- [ ] Current/Benchmark ownership is preserved.
- [ ] ML/Regression/manual paths are preserved where valid.
- [ ] Existing save behavior does not regress.

## Data

- [ ] No DB schema redesign occurs.
- [ ] No Sprint 12 dataset work occurs.
- [ ] Canonical domain contracts are insulated from raw `fuelcons_db` layout.

## Quality

- [ ] Focused tests pass.
- [ ] L0 parity tests pass.
- [ ] UI AppTests exist where practical.
- [ ] Full regression is run.
- [ ] Manual browser smoke is performed.
- [ ] Requirement traceability is documented.
- [ ] Sprint documentation is complete.
- [ ] Closure report is produced.

## Explicitly out of scope

- [ ] No maps.
- [ ] No SOC simulation.
- [ ] No topology graph.
- [ ] No Comparison redesign.
- [ ] No database rebuild.

---

# 48. Implementation Sequence

The Sprint 11 spec is implemented through these packages:

```text
11A — SDD bootstrap + audit + canonical contracts

11B — Current / Correction / Effective Current
      + Domain Proposals

11C — System Scenario composition
      + L0 adapter
      + SystemScenarioResult

11D — Multi-domain Powertrain UI

11E — QA / manual smoke / traceability / freeze
```

Each package must stop at its own scope boundary.

Do not start the following package automatically.

---

# 49. Recommended Coding-Agent Configuration

For Sprint 11A–11C:

- strongest available coding/reasoning model;
- GPT-5.6 Sol if available;
- reasoning: **High**;
- inspect/plan before implementation.

For 11D–11E, use strong coding/reasoning capability with sufficient context to preserve the contracts established earlier.

---

# 50. Frozen Decisions Summary

The following decisions are approved for Sprint 11 and should not be reopened by the coding agent without a concrete conflict:

1. Powertrain/System Scenario is multi-domain.
2. Vehicle Demand is one domain/input of each complete System Scenario.
3. Different System Scenarios may use different VDEs.
4. Current + maximum three Proposals in the initial UI.
5. Domain Proposals start from Effective Current.
6. No Domain Proposal → Domain Proposal inheritance.
7. Domain Proposal is distinct from System Scenario.
8. Presentation Walk is distinct from physical composition.
9. Architecture is ICE/MHEV/HEV/PHEV/BEV classification only in Sprint 11.
10. No topology graph in Sprint 11.
11. Configuration ≠ L0 assumption ≠ result.
12. PSE is system-level, not Engine efficiency.
13. ML/Regression/Benchmark/Technology are resolution/evidence mechanisms, not physical domains.
14. Existing Energy Balance L0 physics must be reused.
15. Unsupported domain changes remain configuration-only rather than producing invented effects.
16. Fidelity coverage must be explicit.
17. Sprint 11 does not redesign the DB.
18. Sprint 12 will own future dataset/data-contract work.
