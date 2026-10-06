# Preliminary Design Review (PDR) - Closure Package

## Sprint 12 - Canonical Data Architecture & ETL

**EcoDrive Analyst**  
PDR baseline frozen for Data Feasibility / Source-to-Contract Analysis and CDR preparation.

> **Codex status:** Treat this document as the approved Sprint 12 PDR baseline. It is a preliminary architecture baseline, not the final physical Data Contract. Do not implement schema/migrations/runtime changes from this document alone. The next phase is Data Feasibility / Source-to-Contract Analysis; architecture changes require evidence and should be recorded as PDR challenges for review before CDR.

| **Document control** | **Value**                                                                      |
|----------------------|--------------------------------------------------------------------------------|
| Status               | PDR CLOSED - candidate baseline frozen                                         |
| Date                 | 09 Sep 2026                                                                    |
| Phase                | Sprint 12 - Canonical Data / ETL                                               |
| Next phase           | Data Feasibility / Source-to-Contract Analysis                                 |
| Next formal review   | Critical Design Review (CDR)                                                   |
| Primary constraint   | Functional-lossless reconstruction versus current VDE_DB + FUELCONS_DB surface |

| **PDR decision:** The conceptual architecture is considered sufficiently stable to stop adding entities and return to evidence. The next phase will attempt to populate and challenge this design with EPA, EEA, JRC and other available sources before the physical data contract is frozen at CDR. |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

# 1. Executive Summary

Sprint 12 started as a database-rebuild discussion and evolved into a source-driven canonical-data effort. The purpose of this PDR was not to define SQL tables in detail, but to establish the target conceptual architecture, the ownership of engineering information, the major system boundaries and the non-regression constraints that the eventual migration must respect.

The design deliberately preserves the operational strengths of the current EcoDrive implementation. VDE remains a wide, resolved Vehicle Demand snapshot; FuelCons remains the final comparison-facing result; existing physics/resolvers are not reopened simply to satisfy normalization. The new architecture adds explicit program/configuration identity, component architecture, reusable analysis resolutions and a generic Run ledger for tests, simulations, estimates and ML outputs.

The PDR exit gate was reviewed and accepted: Component Resolution remains a separate concept; Run is generic enough for multiple evidence types; FuelCons remains the canonical comparison item; Vehicle Configuration grain is acceptable; Component Instance is limited to engineering/simulation-relevant architecture; and incomplete enrichment is explicitly tolerated.

| **Core architectural rule:** The redesign may relocate data, but it shall not silently change what EcoDrive knows. A deterministic canonical reconstruction of the new model must reproduce practically the same engineering surface and semantics currently obtained from VDE_DB + FUELCONS_DB, except for explicitly approved contract corrections. |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

# 2. PDR Objectives and Scope

## 2.1 Objectives

- Preserve all validated EcoDrive roadload and energy capabilities already implemented.

- Establish stable identity above VDE so public-source ETL does not collapse generation, configuration, roadload state and test execution into one row grain.

- Create an engineering architecture layer that can later serve roadload calculations, ML feature generation and future system simulation.

- Keep structured evidence from tests, simulations, deterministic calculations, engineering estimates and ML predictions without forcing each to become a final comparison result.

- Define FuelCons as the pacified, comparison-facing result while preserving how that result was produced.

- Allow incomplete public data to enter the system without fabricating missing component detail.

## 2.2 Explicit non-goals for PDR

- No final SQL DDL, physical indexes, migrations or ORM design.

- No formal SysML implementation, ports, connectors, interface models or topology graph.

- No redesign of canonical VDE mass/tire/roadload physics simply to fit the new schema.

- No exhaustive component ontology or production BOM.

- No final JSON schema, enum list or source-matching algorithm.

- No requirement that every VDE have complete component decomposition.

# 3. Systems Engineering Framing

The workflow is intentionally top-down then bottom-up. The PDR establishes the preliminary target architecture from capabilities and intended use. The next phase returns to source evidence to test whether the architecture can be populated without inventing data. CDR will then freeze the detailed data contract using both the top-down design and bottom-up evidence.

NEEDS / CAPABILITIES  
\|  
v  
PRELIMINARY ARCHITECTURE  
\|  
v  
PDR \<-- CLOSED  
\|  
v  
DATA FEASIBILITY / SOURCE ANALYSIS  
(EPA / EEA / JRC / certification evidence / current DB)  
\|  
v  
CDR  
\|  
v  
DETAILED DATA CONTRACT  
(fields / keys / enums / JSON contracts / migration / adapters)  
\|  
v  
IMPLEMENTATION

# 4. Target Conceptual Architecture

The PDR baseline contains nine primary domain tables. RUN replaces the narrower TEST concept and absorbs structured executions/evidence while keeping the same overall table count.

![PDR target conceptual architecture](assets/pdr_architecture.png)

*Figure 1. PDR target conceptual architecture.*

| **Entity**            | **PDR responsibility**                                                                                    | **Approximate row grain**                                                         |
|-----------------------|-----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| PROGRAM               | Engineering project / product generation.                                                                 | One project/generation context.                                                   |
| VEHICLE_CONFIGURATION | Technical variant inside a Program.                                                                       | One meaningful system/hardware configuration.                                     |
| COMPONENT_DB          | Reusable, extensible component definition and common engineering properties.                              | One reusable component definition.                                                |
| TIRE_DB               | Specialized tire definition/evidence domain retained because tire roadload semantics are richer.          | One tire reference/evidence record per current domain contract.                   |
| COMPONENT_INSTANCE    | Engineering architecture / eBOM-lite usage of a component in a Vehicle Configuration.                     | One component occurrence/role/position in one configuration.                      |
| COMPONENT_RESOLUTION  | Reusable engineering resolution that reduces component evidence/instances into roadload-relevant outputs. | One resolved engineering representation for a defined boundary/method/conditions. |
| VDE                   | Persisted resolved Vehicle Demand snapshot; operational state for current EcoDrive physics.               | One coherent roadload/Vehicle Demand state.                                       |
| RUN                   | Ledger of test, simulation, estimation, calculation or ML execution/evidence.                             | One identifiable execution/evidence record linked to a VDE.                       |
| FUELCONS              | Pacified canonical energy/fuel/CO2/range result used by Comparison.                                       | One comparable result basis for one VDE.                                          |

# 5. Preliminary Entity Contracts

## 5.1 PROGRAM

Question answered: Which engineering project/generation is this?

Example interpretation: a Prius project spanning several model years can remain one Program when the fundamental project/platform generation is the same even if there are facelifts, aero detail changes or version-specific powertrains.

PDR rule: Program identity must not be reduced to commercial model name alone. Program codes from OEM/public evidence are optional provenance, not a prerequisite for the entity to exist.

## 5.2 VEHICLE_CONFIGURATION

Question answered: Which meaningful technical variant of the Program is this?

Configuration changes include relevant engine, transmission, propulsion architecture or other stable hardware choices. Values that vary routinely between VDE states - including MRO, GVWR and GCWR in the current working context - are not used as Vehicle Configuration identity.

| **Boundary:** Vehicle Configuration describes the system architecture. It is not a second VDE table and does not own transient/resolved roadload state. |
|---------------------------------------------------------------------------------------------------------------------------------------------------------|

## 5.3 COMPONENT_DB

Question answered: What reusable engineering component/block is this?

Component DB remains extensible and heterogeneous. ABC and mass are examples of properties that can be shared across many component types, but the table is not restricted to ABC-only records. Stable, scalar engineering properties that are frequently queried, filtered, compared or useful for ML may be first-class columns. Type-specific complex data should use structured custom properties or artifact/model references.

Typical canonical surface (illustrative, not CDR-frozen):  
component_id  
component_type / domain  
manufacturer / model / hardware_reference  
mass_kg  
roadload_A / B / C (nullable)  
selected common scalar fields (nullable by component type)  
custom_properties_json (type-specific small structures)  
data_artifact_ref (maps / large datasets)  
model_artifact_ref (future model / FMU / Simulink / Python reference)  
source / provenance

## 5.4 TIRE_DB

Tires remain specialized rather than being forced immediately into the generic Component DB. The current tire domain carries standards, test methodology, pressure/load conditions, RRC semantics and SAE/ISO-specific behavior that justify preserving a dedicated table during this redesign.

## 5.5 COMPONENT_INSTANCE

Question answered: Where/how is this reusable component used in this Vehicle Configuration?

This is the engineering architecture / eBOM-lite layer and the primary foundation for future simulation reuse. It is intentionally not a full production BOM and does not model screws, seats or unrelated hardware.

Example:  
VEH-001 \| ENGINE \| PRIMARY \| ENG-001  
VEH-001 \| TRANSMISSION \| MAIN \| TRA-014  
VEH-001 \| EMOTOR \| REAR_AXLE \| EM-005  
VEH-001 \| TIRE \| FRONT_LEFT \| TIR-025  
VEH-001 \| TIRE \| FRONT_RIGHT \| TIR-025  
VEH-001 \| TIRE \| REAR_LEFT \| TIR-026  
VEH-001 \| TIRE \| REAR_RIGHT \| TIR-026

PDR intent: this layer should eventually be able to act as a pointer to reusable model blocks or artifacts without forcing VDE to know detailed system topology.

## 5.6 COMPONENT_RESOLUTION

Question answered: How did one or more component definitions/instances become the engineering representation adopted by roadload/VDE?

Component Resolution is kept separate from the architectural BOM. It may represent aggregation, adoption, calibration or reduction of one or more component-level contributions into a canonical roadload boundary. It may be reused by multiple VDEs when its engineering conditions and semantics remain valid. Source lineage may identify the original VDE, run/test or evidence from which the resolution was derived.

Important: the resolution is evidence/traceability, not the hot-path runtime state. The VDE still persists the scalar values actually used.

Example concept:  
Component Instances: Rear-left tire + Rear-right tire  
\|  
v  
Component Resolution: REAR_TIRE / adopted A,B,C  
- method / conditions / provenance  
- JSON lineage to inputs and calculation path  
\|  
v  
VDE snapshot: rear tire A,B,C copied/persisted

## 5.7 VDE

Question answered: What resolved physical/roadload state is EcoDrive using for Vehicle Demand?

The wide VDE snapshot is deliberately preserved. It continues to carry the final quantities required by current physics and analysis: mass state, CdA/aero, tire/RR state, canonical component roadload buckets, authoritative roadload, TOTAL/NET and VDE outputs. The normalized architecture is not allowed to force repeated reconstruction of those values during ordinary UI rendering.

VDE remains operationally recognizable:  
- mass / test mass / calculation mass / inertia-class basis  
- MRO / GVWR / GCWR where applicable to the VDE state  
- CdA / weight distribution / payload / pressures / RRC  
- authoritative roadload / Target / Coast A,B,C  
- Tire A,B,C  
- Transmission A,B,C  
- Brake A,B,C  
- Axle/Hubs A,B,C  
- Parasitic A,B,C  
- TOTAL / NET  
- VDE Urban / Highway / Total / Net  
- lineage / provenance / parent VDE

### VDE persistence semantics (Sprint 12F.14B freeze)

VDE is a persisted resolved Vehicle Demand analysis state associated with a Vehicle Configuration. Multiple VDEs may exist for the same Vehicle Configuration when distinct analysis, test or performance conditions are intentionally retained for later comparison. `vde_id_parent` may express deterministic derivation from a baseline VDE; parentage remains `NULL` when the source or creation flow does not prove it. RUN remains the execution and evidence layer.

A temporary sensitivity, display-only recalculation, unsaved what-if or cycle preview does not create a VDE row by default. Real test-conditioned states, including cold or temperature-conditioned homologation cases, may remain persisted. Future interactive temperature sensitivity is calculated from a selected baseline and is persisted only when the user explicitly saves the resulting condition.

The resolved numeric mass, roadload and component snapshot fields remain authoritative for calculation. The existing mass structure, including test mass, MRO, GVWR and GCWR, is sufficient for the current scope; labels explain basis and provenance but do not replace numeric engine inputs. No additional load-case taxonomy is required for the current scope.

## 5.8 RUN

Question answered: Which identifiable evidence/execution produced a result that may support a VDE or FuelCons?

RUN replaces the narrower TEST table in the conceptual model. It can preserve structured source tests and also support progressive engineering fidelity without creating one table per method.

run_type:  
TEST  
SIMULATION  
ESTIMATION  
CALCULATION  
ML_PREDICTION  
  
fidelity_level:  
L0 / L1 / L2 / L3 / nullable  
  
confidence:  
separate from fidelity (e.g. LOW / MEDIUM / HIGH / nullable)

A Run is append-oriented evidence. An L0 estimate may remain stored after an L1 simulation becomes available. FuelCons may later adopt the L1 result while retaining the earlier L0 Run for historical traceability.

For physical test sources, Run should preserve identifiers, cycle/procedure, set ABC where relevant, conditions and structured phase/detail data without requiring a deep test ontology during Sprint 12.

## 5.9 FUELCONS

Question answered: What result is currently pacified and appropriate for Comparison for this VDE and comparison basis?

PDR decisions already accepted:

- vde_id is mandatory.

- One VDE may have multiple FuelCons rows for different comparison bases.

- reference_fuelcons_id may relate complementary homologation results to a principal homologation reference.

- comparison_basis defines what the row means (e.g. principal certification basis, US06, SC03, engineering L0, etc.).

- Urban, Highway and Combined remain first-class output dimensions when applicable.

- Fuel, electric energy, CO2 and electric range use one unified schema; fields remain present and are NULL when not applicable/not available.

- Zero is reserved for a true physical zero, not used as a placeholder for non-applicable data.

- Final/scorecard values remain columns; the detailed path through Runs, phases and post-processing is structured lineage (JSON).

- Homologated range is preserved as an authoritative/pacified result and is not automatically overwritten by battery/consumption arithmetic.

- comparison_basis and energy_basis are different concepts: methodology/cycle versus energy measurement/calculation boundary.

Unified comparison surface (illustrative):  
fuel_urban_l_per_100km  
fuel_highway_l_per_100km  
fuel_combined_l_per_100km  
  
energy_urban_wh_per_km  
energy_highway_wh_per_km  
energy_combined_wh_per_km  
  
electric_range_km  
  
co2_urban_g_per_km  
co2_highway_g_per_km  
co2_combined_g_per_km  
  
eta_pt_est / bev_eff_drive / utility_factor (when part of the adopted result)  
comparison_basis  
energy_basis  
result_details_json

# 6. Key Architectural Decisions Frozen at PDR

| **Decision**             | **PDR baseline**                                                                                                                                                        |
|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Migration safety         | Canonical reconstruction must be functionally lossless versus the current VDE_DB + FUELCONS_DB surface, except approved corrections.                                    |
| Runtime owner            | VDE remains a persisted resolved snapshot; normalized architecture is not the runtime physics state.                                                                    |
| Architecture vs analysis | Component Instance describes architecture; Component Resolution describes engineering analysis/reduction.                                                               |
| Component extensibility  | Common scalar fields may be first-class; complex type-specific structures use JSON or artifact/model references.                                                        |
| Incomplete enrichment    | Missing components/resolutions do not invalidate authoritative source VDE/Run records.                                                                                  |
| Evidence history         | RUN preserves test/simulation/estimate/calculation/ML evidence; later evidence does not require deleting earlier Runs.                                                  |
| Comparison owner         | FuelCons is the canonical pacified item for comparison/reporting.                                                                                                       |
| Powertrain neutrality    | One FuelCons schema supports ICE/HEV/PHEV/BEV with nullable KPI fields.                                                                                                 |
| Mass                     | Existing mass resolver/core is preserved; MRO/GVWR/GCWR and other varying state values remain VDE-side unless evidence later proves a stable master property is useful. |
| SE scope                 | Architecture is intentionally eBOM-lite; no ports/connectors/topology graph at PDR.                                                                                     |

# 7. Runtime and Integration Principles

The normalized model must not introduce significant latency into the existing VDE Setup. The intended integration pattern separates a fast operational path from a materialization/resolution path.

FAST PATH  
VDE snapshot -> VDE Setup / Comparison / existing physics consumers  
  
RESOLUTION / UPDATE PATH  
Vehicle Configuration  
-> Component Instances  
-> Component DB / Tire DB  
-> Component Resolution / existing resolvers  
-> Apply resolved scalars to VDE snapshot  
  
EVIDENCE / ENERGY PATH  
VDE  
-> Runs (test / estimate / simulation / ML)  
-> FuelCons adoption / post-processing  
-> Comparison / report

Performance risks to avoid at implementation: N+1 component queries, rebuilding complete architecture on every Streamlit rerun, loading large maps/artifacts on Browse, or changing deterministic core functions to know about storage tables directly.

Recommended integration boundary: database/adapters/services materialize existing canonical core requests; deterministic physics remains storage-agnostic.

# 8. Next Phase: Data Feasibility / Source-to-Contract Analysis

PDR closure intentionally stops before detailed schema. The next phase should challenge the architecture with the actual data already collected and audited.

| **Question to test**                                   | **Evidence/source focus**                                                                                         | **Why it matters for CDR**                                       |
|--------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| Can Program be inferred reliably?                      | EPA/EEA/JRC identity fields, model-year patterns, certification identifiers.                                      | Defines Program matching and fallback identity rules.            |
| Can Vehicle Configuration be separated from VDE state? | Engine/transmission/driveline fields, test-number/configuration identifiers, roadload repetition patterns.        | Defines configuration grain and matching keys.                   |
| Which Component Instances are observable?              | Transmission/engine/tire/axle/electric-drive fields, certification evidence, JRC component descriptors.           | Tests whether the eBOM-lite layer is realistic with public data. |
| Which component fields deserve first-class columns?    | Repeated scalar availability across component types and sources.                                                  | Determines Component DB canonical surface versus JSON/artifacts. |
| Which Component Resolutions are defensible?            | Direct roadload evidence, deterministic calculations, isolated component evidence, natural experiment candidates. | Prevents invented component decomposition.                       |
| What Runs can be preserved now?                        | EPA test rows, WLTP/JRC runs, source procedures, later EcoDrive simulations/estimations.                          | Defines Run grain and minimum structured fields.                 |
| Which FuelCons bases can be pacified?                  | Available fuel/electric/CO2/range outputs plus post-processing inputs.                                            | Defines v1 comparison_basis enum and post-processing rules.      |
| Can the legacy surface be reconstructed?               | Current SQLite DBs plus ETL staging data.                                                                         | Primary non-regression gate before implementation.               |

| **Feasibility principle:** The next phase must not force source data into the PDR model. If real data consistently contradicts an entity grain or relationship, the architecture is revised before CDR. |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

# 9. CDR Entry Criteria

Return to CDR only after the source-analysis phase can support the following with concrete examples and measured coverage:

- Program and Vehicle Configuration matching rules are demonstrated on representative source records.

- The initial first-class Component DB field set is justified by observed data, not imagined future completeness.

- Component Resolution cases are demonstrated with explicit evidence/provenance and no hidden closure balancing.

- RUN records can preserve at least the current structured test evidence and a representative estimate/simulation flow.

- FuelCons comparison bases and Urban/Highway/Combined post-processing are demonstrated for representative ICE and electrified cases.

- BEV/PHEV energy basis and range semantics are explicitly supported.

- A draft canonical flat reconstruction is shown against current VDE_DB + FUELCONS_DB records.

- Open fields are classified as first-class scalar, JSON structure, artifact reference or deferred.

# 10. Deferred to CDR or Later

| **Deferred item**                                   | **Reason for deferral**                                                  |
|-----------------------------------------------------|--------------------------------------------------------------------------|
| Physical SQL DDL / indexes / migration scripts      | Requires source feasibility and final grain/key decisions.               |
| Exact field names, FK directions and nullability    | PDR owns semantics; CDR owns physical contract.                          |
| Final comparison_basis / energy_basis enums         | Must be driven by supported datasets and post-processing rules.          |
| Detailed JSON schemas                               | Need concrete source and simulation examples first.                      |
| Component simulation topology / ports / connectors  | No current capability requires graph-level system modeling.              |
| Model registry / FMU / Simulink execution semantics | Keep only forward-compatible artifact/model references at this stage.    |
| Formal configuration/version baseline system        | Not required for Sprint 12; basic provenance/history is sufficient.      |
| Warehouse / feature-store technology choice         | Downstream serving/ML architecture is outside current data-contract PDR. |

# 11. Principal Risks and PDR Mitigations

| **Risk**                                             | **Mitigation frozen at PDR**                                                                               |
|------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| Over-normalization breaks mature EcoDrive workflows. | Preserve VDE as wide snapshot and require legacy-compatible canonical reconstruction.                      |
| Architecture tables become mandatory hot-path joins. | Resolve/materialize only on create/update/audit; ordinary VDE read remains fast.                           |
| Component DB becomes an unqueryable JSON dump.       | Promote stable/common ML/BI scalars to columns; reserve JSON for complex/type-specific structures.         |
| Public sources are incomplete.                       | Incomplete enrichment is valid; authoritative roadload/result records survive without full components.     |
| Estimated/ML data masquerades as measured.           | RUN type, fidelity, confidence and provenance remain separate dimensions.                                  |
| Better later evidence erases engineering history.    | RUN behaves as an evidence ledger; FuelCons adoption may evolve without deleting prior Runs.               |
| Future simulation needs richer architecture.         | Component Instance provides a stable usage layer; topology is deferred until an actual solver requires it. |

# 12. PDR Exit Gate and Closure

| **Exit question**                                                                    | **Decision** |
|--------------------------------------------------------------------------------------|--------------|
| Is Component Resolution a separate concept?                                          | YES          |
| Is RUN generic enough for test/simulation/estimation/calculation/ML evidence?        | YES          |
| Is FuelCons the canonical comparison item?                                           | YES          |
| Is Vehicle Configuration grain acceptable?                                           | YES          |
| Does Component Instance represent only engineering/simulation-relevant architecture? | YES          |
| Can the model tolerate incomplete enrichment?                                        | YES          |

| **PDR closure statement:** The preliminary architecture is frozen as the working baseline for Sprint 12 data-feasibility analysis. No additional domain entities should be introduced during the next phase unless a concrete source/data case demonstrates that the current nine-table model cannot represent the required information without loss or semantic distortion. |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## 12.1 Change-control principle until CDR

During source analysis, findings should be classified as:

- CONFIRMS PDR - source fits the preliminary model.

- FIELD-LEVEL CDR INPUT - affects columns/enums/JSON but not entity architecture.

- PDR CHALLENGE - requires changing grain, ownership or relationship between entities.

- DEFERRED - useful for later simulation/ML but not required for Data Contract v1.

# Appendix A - Preliminary Relationship Summary

| **From**                      | **Relationship**        | **To**                 | **PDR meaning**                                                                 |
|-------------------------------|-------------------------|------------------------|---------------------------------------------------------------------------------|
| PROGRAM                       | 1:N                     | VEHICLE_CONFIGURATION  | A program/generation may have multiple technical variants.                      |
| VEHICLE_CONFIGURATION         | 1:N                     | COMPONENT_INSTANCE     | A configuration contains multiple engineering component occurrences.            |
| COMPONENT_INSTANCE            | N:1                     | COMPONENT_DB / TIRE_DB | Multiple usages may reuse the same component definition.                        |
| VEHICLE_CONFIGURATION         | 1:N                     | VDE                    | A configuration may have multiple roadload/Vehicle Demand states.               |
| COMPONENT_INSTANCE / evidence | N:1 or N:N conceptually | COMPONENT_RESOLUTION   | One resolution may use one or more component inputs/evidence.                   |
| COMPONENT_RESOLUTION          | reusable adoption       | VDE                    | A resolution may be adopted by multiple VDEs when validity conditions match.    |
| VDE                           | 1:N                     | RUN                    | A VDE may have multiple tests/simulations/estimates/calculations/predictions.   |
| VDE                           | 1:N                     | FUELCONS               | A VDE may have multiple comparison bases/results.                               |
| FUELCONS                      | self-reference          | FUELCONS               | Complementary homologation results may point to a principal reference FuelCons. |
| RUN                           | adopted evidence        | FUELCONS               | FuelCons may adopt one or more Runs via structured lineage/post-processing.     |

# Appendix B - Information Ownership Heuristics for CDR

When mapping current fields or new source attributes, use these questions before deciding a table:

| **Question**                                                              | **Likely owner**       |
|---------------------------------------------------------------------------|------------------------|
| Does it identify the project/generation?                                  | PROGRAM                |
| Does it define a stable technical variant?                                | VEHICLE_CONFIGURATION  |
| Does it describe a reusable component independent of where installed?     | COMPONENT_DB / TIRE_DB |
| Does it describe where/how a component is used in the configuration?      | COMPONENT_INSTANCE     |
| Does it depend on a method, boundary, combination or analysis conditions? | COMPONENT_RESOLUTION   |
| Is it the resolved physical state actually used by Vehicle Demand?        | VDE                    |
| Is it an identifiable test/simulation/estimate/calculation/prediction?    | RUN                    |
| Is it the currently adopted/pacified comparison result?                   | FUELCONS               |

Additional data-shape heuristic: scalar engineering quantities frequently used for filtering, BI, scorecards or ML should prefer first-class columns. Complex multidimensional data, solver traces, rich method internals and large maps should use JSON or artifact/model references.

# Appendix C - PDR Design Principles

| **Principle**             | **Meaning**                                                                                                   |
|---------------------------|---------------------------------------------------------------------------------------------------------------|
| Preserve before normalize | Do not sacrifice working VDE/FuelCons behavior to achieve theoretical relational purity.                      |
| Architecture is reusable  | Vehicle configuration and component usage should be meaningful beyond one VDE calculation.                    |
| Analysis is explicit      | Resolved/adopted engineering representations should not be confused with physical component identity.         |
| Evidence is not state     | Tests, estimates, simulations and ML predictions are Runs; the adopted state/result is represented elsewhere. |
| Final KPI is first-class  | Comparison-facing outputs belong in columns even when their derivation is complex.                            |
| Lineage can be structured | JSON is appropriate for calculation paths, constituent Runs, conditions and post-processing details.          |
| Missing is not zero       | NULL represents not applicable/not available; zero represents a true zero.                                    |
| Do not fabricate closure  | Incomplete public component decomposition is tolerated rather than silently balanced or invented.             |
| Core stays deterministic  | Storage architecture should adapt to existing core contracts, not leak table semantics into physics.          |
| CDR follows evidence      | Detailed schema must be justified by available data and measured feasibility.                                 |
