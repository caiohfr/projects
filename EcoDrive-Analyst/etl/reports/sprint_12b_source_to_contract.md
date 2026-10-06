# Sprint 12B — Data Feasibility / Source-to-Contract Analysis

## Decision status

The measured data supports the PDR's nine-entity conceptual architecture. No tenth primary entity is justified and no final SQL schema is implemented. The principal CDR inputs are field-level: fallback Program identity, the exact Vehicle Configuration key, RUN evidence taxonomy, FuelCons comparison bases, and relational reconstruction semantics.

Finding classification: 5 `CONFIRMS_PDR`, 4 `FIELD_LEVEL_CDR_INPUT`, 0 `PDR_CHALLENGE`, and 1 `DEFERRED`.

## Evidence boundary

- Source evidence: Sprint 12A inventory, EPA MY2026 rows, EEA/JRC coverage and current EcoDrive schemas/rows.
- Current SQLite databases were opened with URI `mode=ro` plus `PRAGMA query_only=ON`.
- No raw data, runtime database, `src/`, page, physics, resolver, DDL, migration or production adapter was modified.
- Component-pair evidence is not treated as causal component decomposition.

## Current legacy surface

- `eco_drive.db`: component_db=0, data_change_log=5,003, fuelcons_db=5,004, tire_roadload_db=1, vde_db=5,003, vde_request_history=0, vde_request_history_proposals=0.
- `eco_drive_qa.db`: fuelcons_db=0, tire_roadload_db=0, vde_db=0.

The draft in-memory owner partition reconstructed **10,008/10,008** assessed current rows with exact field/value equality and no collisions. This is a field-level non-regression proof; it does not yet prove foreign-key, adoption, lineage or UI-query equivalence.

## PDR findings

| ID | Classification | Finding | Proposed CDR decision |
|---|---|---|---|
| S12B-01 | `FIELD_LEVEL_CDR_INPUT` | Not reliably as an OEM generation across sources. EPA provides strong source-local commercial/test-family identity, EEA codes remain semantically ambiguous locally, and JRC identity is anonymized. | Allow a source-scoped fallback Program with identity_status/confidence and raw source keys; prohibit cross-year/cross-source merge without explicit evidence. |
| S12B-02 | `CONFIRMS_PDR` | Yes conceptually: 1,291 strict EPA hardware/test-article candidate keys contain 1,092 keys with multiple observed VDE states. | Keep Vehicle Configuration separate from VDE; freeze exact key only after resolving whether Test Vehicle ID/Configuration # identify physical articles, source configurations, or both. |
| S12B-03 | `CONFIRMS_PDR` | Engine, transmission and driveline are partially observable in EPA; JRC also exposes tire, e-motor and battery descriptors. Reusable hardware identity and component position are usually incomplete. | Permit partial Component Instances with nullable component reference/position and direct provenance; never require enrichment for VDE/Run validity. |
| S12B-04 | `FIELD_LEVEL_CDR_INPUT` | Frequently filtered identity and numeric features justify scalars; heterogeneous descriptors fit JSON; maps/PDFs/models fit artifact references. | Use the field_shape_decisions matrix as the CDR candidate surface; do not promote fields solely because they exist in one source. |
| S12B-05 | `CONFIRMS_PDR` | Tier-0 public data supports whole-vehicle roadload, not a component causal split. No EPA/JRC component ABC resolution is defensible from structured data alone. | Component Resolution remains optional and separate; v1 accepts zero public resolutions while preserving authoritative VDE/Run roadload. |
| S12B-06 | `FIELD_LEVEL_CDR_INPUT` | EPA tests and current estimation/calculation/ML flows fit RUN. FuelEconomy and EEA are evidence records but are not accurately named by the current run_type examples. | Keep RUN entity; add evidence_kind or a HOMOLOGATION/MONITORING run type at CDR. Preserve source row/result-detail grain separately. |
| S12B-07 | `FIELD_LEVEL_CDR_INPUT` | EPA label Urban/Highway/Combined, EEA WLTP combined, JRC declared/simulated and current legacy result surfaces are supportable with different basis semantics and nullable dimensions. | Seed comparison_basis candidates from measured source contexts; keep energy_basis independent and preserve units/fuel mode. Do not derive missing combined/phase values yet. |
| S12B-08 | `CONFIRMS_PDR` | At field/value level, yes: 10,008/10,008 current rows across vde_db, fuelcons_db, component_db and tire_roadload_db reconstruct exactly after partitioning by candidate PDR owner. | Accept this as the first non-regression proof, but require relational/adoption-semantic reconstruction tests before CDR exit. |
| S12B-09 | `CONFIRMS_PDR` | No current evidence requires another entity. The monitoring/homologation distinction can be represented as RUN field/enum input without changing the nine-table architecture. | Keep the nine-entity baseline unless future evidence demonstrates semantic loss. |
| S12B-10 | `DEFERRED` | PDF parsing/RAG, EPREL enrichment, ML prediction, component estimation, topology and physical SQL are not required for Data Contract v1 feasibility. | Retain artifact/provenance hooks only; do not implement these capabilities before CDR. |

## Program and Vehicle Configuration grain

Program generation cannot be inferred reliably across sources. EPA has useful source-local commercial/test-family identifiers, EEA identity-code semantics remain unresolved in the supplied local corpus, and JRC identity is anonymized. The defensible v1 fallback is a source-scoped Program identity with explicit low confidence/status; cross-year or cross-source merges require positive evidence.

For EPA MY2026, a strict candidate hardware/test-article key produced **1,291** keys from **3,901** rows. **1,092** keys contain multiple distinct ETW/Target/Set/procedure states. This directly supports separate Vehicle Configuration and VDE concepts, but it does not prove that Test Vehicle ID or Configuration # is a universal hardware key.

## Components and resolutions

EPA directly supports partial engine, transmission and driveline instances; JRC adds partial tire, e-motor and battery descriptions. Part-number-level reusable identity and instance position are generally absent. Component Instance must therefore allow incomplete identity/role data without invalidating VDE or RUN.

No Tier-0 source supports a defensible component roadload split. EPA has authoritative whole-vehicle Target/Set ABC but no tire/RRC/CdA fields; JRC's f0/f1/f2 fields are labelled at whole-vehicle WLTP/real-world level. Component Resolution remains an optional analysis/evidence object, not a required ingestion product.

## RUN and FuelCons

EPA test records fit `RUN(TEST)`. Existing EcoDrive estimation/calculation/ML flows also fit the ledger. FuelEconomy and EEA require a field-level taxonomy decision because a published homologation/monitoring record should not be silently labelled as a physical TEST. JRC declared and simulated values should remain distinct evidence/results.

FuelCons can support EPA label Urban/Highway/Combined, EEA combined WLTP, and JRC declared/simulated result bases using nullable dimensions. `comparison_basis` and `energy_basis` must remain separate; no absent combined or phase value is derived in this phase.

## CDR entry assessment

| Criterion | Status | Evidence |
|---|---|---|
| Program/configuration matching | PARTIAL | Fallback Program and strict EPA configuration candidates demonstrated; final keys unresolved. |
| First-class Component DB fields | READY FOR DECISION | Scalar/JSON/artifact recommendations in `field_shape_decisions.csv`. |
| Component Resolution examples | PARTIAL / EMPTY-BY-DESIGN | No structured public component resolution is defensible; current legacy resolution fields are inventoried. |
| RUN preservation | READY FOR FIELD DECISION | EPA/current flows fit; homologation/monitoring enum decision open. |
| FuelCons bases | READY FOR FIELD DECISION | Candidate basis matrix covers EPA, EEA, JRC and legacy. |
| BEV/PHEV energy and range | PARTIAL | FuelEconomy/JRC/EEA provide evidence; sheet/basis rules require CDR decision. |
| Draft legacy reconstruction | FIELD-LEVEL PASS | Exact field/value reconstruction for every assessed current row. Relational behavior remains open. |
| Open-field shape classification | READY | First-class/JSON/artifact/deferred matrix produced. |

The project is ready for a focused CDR decision workshop, but **not** for final DDL or migration. CDR must close the six decisions in `open_cdr_decisions.csv`, then require relational reconstruction tests before implementation.

## Reproduction

```powershell
python etl/scripts/sprint_12a_audit.py
python etl/scripts/sprint_12b_source_to_contract.py
```

## Machine-readable outputs

- `pdr_findings.csv`
- `source_entity_matrix.csv`
- `legacy_schema_inventory.csv`
- `legacy_field_ownership.csv`
- `legacy_reconstruction_matrix.csv`
- `configuration_candidates.csv`
- `component_observability.csv`
- `field_shape_decisions.csv`
- `run_feasibility.csv`
- `fuelcons_basis_feasibility.csv`
- `open_cdr_decisions.csv`
- `source_to_contract.json`
