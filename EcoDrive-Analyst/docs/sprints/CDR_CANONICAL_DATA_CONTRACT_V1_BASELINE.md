# EcoDrive Sprint 12 â€” CDR Closure / Canonical Data Contract v1 Baseline

## Status

**CDR decision workshop: CLOSED**

This document records the approved CDR decisions derived from the Sprint 12 PDR and Sprint 12B Source-to-Contract analysis.

The architecture remains at nine primary entities:

1. PROGRAM
2. VEHICLE_CONFIGURATION
3. COMPONENT_DB
4. TIRE_DB
5. COMPONENT_INSTANCE
6. COMPONENT_RESOLUTION
7. VDE
8. RUN
9. FUELCONS

No tenth primary entity is introduced. This is a data-contract baseline, not a physical SQL implementation. DDL, migrations, runtime schema replacement, and broad application refactors remain out of scope until the compatibility proof is completed.

## CDR-01 â€” PROGRAM identity

**Decision: APPROVED WITH REFINEMENT**

`PROGRAM` represents an engineering project / vehicle generation and may span multiple model years.

When project/generation identity cannot be established from available evidence, ETL may create a conservative, source-scoped provisional Program identity. Model Year may participate in a fallback identity when needed to avoid unsupported merges, but Model Year is not part of the semantic definition of Program. Cross-year or cross-source consolidation requires positive evidence.

Minimum concepts: canonical `program_id`, commercial identity, generation/program name when known, model-year range when known, identity status/confidence, and source provenance.

## CDR-02 â€” VEHICLE_CONFIGURATION identity and grain

**Decision: APPROVED WITH SOURCE-SCOPED IDENTITY**

`VEHICLE_CONFIGURATION` represents a relatively stable technical variant within a Program and is independent from VDE/test state.

Stable descriptors may include propulsion/engine architecture, transmission, gear count, drive system, axle/final-drive ratio, electrification and relevant component architecture.

ETW/test mass state, Target ABC, Set ABC, procedure and test conditions do not define Vehicle Configuration identity.

EPA Test Vehicle ID, Configuration Number and source-derived signatures are preserved as source identity/provenance and shall not be treated as universal hardware identity without supporting evidence. EcoDrive uses its own canonical `vehicle_configuration_id`.

## CDR-03 â€” RUN evidence model

**Decision: APPROVED**

`RUN` is the canonical ledger for structured technical evidence. It can represent physical tests, simulations, engineering estimations, deterministic calculations, ML predictions and declared/published source results.

Two independent dimensions are retained:

- `run_type`: how the result was produced. Candidate values: `TEST`, `SIMULATION`, `ESTIMATION`, `CALCULATION`, `ML_PREDICTION`, `DECLARED_RESULT`.
- `evidence_kind`: the role/context of the evidence. Candidate values: `ENGINEERING`, `HOMOLOGATION`, `MONITORING`, `BENCHMARK`, `SOURCE_RECORD`.

`fidelity_level` and `confidence` remain independent. Earlier RUN evidence is preserved when better evidence later becomes available.

## CDR-04 â€” FUELCONS comparison semantics

**Decision: APPROVED WITH NON-RESTRICTIVE COMPARISON**

`FUELCONS` is the canonical/pacified result used by Comparison and reporting.

`comparison_basis` describes methodology/context. `energy_basis` remains independent.

Different comparison bases may be compared by the user. The system provides metadata and filters, not hard blocking. Legislation, comparison basis and energy basis remain visible/filterable.

Unavailable dimensions remain `NULL`; `0` means a real physical zero, never missing/not-applicable. Values are not silently derived unless the methodology explicitly defines the derivation.

## CDR-05 â€” Legacy compatibility and application boundary

**Decision: APPROVED â€” BACKWARD-COMPATIBLE APPLICATION CONTRACT**

Core migration principle:

> **Normalize persistence, preserve application contracts.**

The new relational model shall not force broad rewrites of existing pages. Existing application-facing VDE and FuelCons contracts should be preserved wherever practical through a canonical compatibility/read layer.

Target behavior:

```text
NEW CANONICAL MODEL
Program + Vehicle Configuration + Components + VDE + RUN + FuelCons
        â†“
compatibility view / adapter / repository query
        â†“
existing VDE / FuelCons application-facing shape
        â†“
existing pages
```

Expected changes should be localized primarily to persistence, repositories, adapters, canonical read queries/views and migration helpers. Existing pages, physics, resolvers, Quick Scenario, Comparison and VDE Setup should require only minimal and targeted changes unless a new capability explicitly requires more.

VDE remains a persisted wide operational Vehicle Demand snapshot. FuelCons remains a persisted adopted comparison result. RUN shall not become a mandatory runtime traversal step for existing dashboards.

Before physical migration is accepted, the new model must demonstrate relational/query equivalence against the legacy application surface, including at minimum: VDE counts/identity, FuelCons-to-VDE multiplicity, IDs/links, NULL behavior, mass, Target/TOTAL/NET semantics, roadload ABC values, Urban/Highway/Combined results, labels/filter metadata and adoption lineage where introduced.

Any intentional difference must be explicitly approved and documented.

## CDR-06 â€” Unresolved source semantics and RAG boundary

**Decision: APPROVED AS DEFERRED SEMANTICS**

Unknown or ambiguous source semantics are preserved without inventing physical meaning. Current examples include EEA RLFI semantics and exact JRC source-row grain.

Rules:

- preserve raw source values;
- preserve source-row identity;
- preserve provenance;
- mark semantic status as `UNRESOLVED`/`PARTIAL`;
- map only meanings supported by evidence into canonical fields;
- ambiguous fields must not invalidate otherwise valid VDE/RUN/FuelCons ingestion;
- no physical interpretation is inferred from a column name alone.

RAG/document enrichment may later retrieve regulations, manuals, PDFs, technical documentation, papers or OEM evidence to help resolve semantics. RAG output does not automatically become canonical truth. Preferred flow: retrieval â†’ evidence discovery â†’ review/validation â†’ versioned deterministic mapping rule â†’ ETL/canonical data.

## Entity responsibilities

### PROGRAM
Engineering project/generation identity.

### VEHICLE_CONFIGURATION
Stable technical variant within Program.

### COMPONENT_DB
Reusable extensible component definition. Frequently queried stable engineering scalars may be first-class columns; sparse heterogeneous data may use JSON; large maps, curves, PDFs and model files use artifact references.

### TIRE_DB
Specialized reusable tire engineering/evidence catalog.

### COMPONENT_INSTANCE
Lightweight engineering architecture/eBOM occurrence linking a reusable component definition to a Vehicle Configuration. Partial instances are allowed.

### COMPONENT_RESOLUTION
Optional analysis-side resolved representation mapping one or more component instances into canonical VDE/roadload buckets while preserving method, scalar outputs and lineage. Public Tier-0 ingestion may legitimately create zero Component Resolutions.

### VDE
Persisted wide Vehicle Demand snapshot. It keeps actual resolved values used by physics and current workflows. Mass state remains in VDE when it can legitimately vary between VDEs of the same Vehicle Configuration. Existing mass resolver/core physics remain unchanged.

### RUN
Evidence ledger for tests, simulations, estimations, calculations, ML predictions and source-declared results.

### FUELCONS
Pacified/adopted comparison result linked to exactly one VDE. One VDE may have multiple FuelCons records. Urban/Highway/Combined fuel, energy and CO2 remain first-class where applicable; BEV/PHEV electric energy and range remain first-class where applicable.

## Migration guardrails

1. The canonical decomposition must remain functionally/losslessly reconstructable relative to the current `vde_db + fuelcons_db` application surface, except explicitly approved corrections.
2. Incomplete component enrichment must never invalidate authoritative VDE/source records.
3. Whole-vehicle authoritative roadload remains valid even when no component decomposition exists.
4. Static component master edits must not silently rewrite historical VDE snapshots.
5. `NULL` is used for missing/non-applicable data; `0` is a real value.
6. Observed, calculated, estimated, simulated and ML-derived information must remain distinguishable.
7. No new physical effect or inferred causal component split is introduced by ETL.
8. Current physics/core behavior remains canonical unless separately approved.
9. Broad UI refactors are not an acceptable side effect of persistence normalization.
10. RAG/enrichment remains a future evidence-resolution capability, not a hidden production inference dependency.

## CDR closure status

Closed decisions:

- CDR-01 Program fallback identity
- CDR-02 Vehicle Configuration key/grain
- CDR-03 RUN evidence taxonomy
- CDR-04 FuelCons comparison semantics
- CDR-05 legacy relational compatibility
- CDR-06 unresolved source semantics / RAG boundary

Deferred by design: final physical SQL schema, DDL, migrations, production adapters, PDF/RAG implementation, EPREL enrichment, ML enrichment, component estimation, full topology/ports/connectors, and unsupported causal component decomposition.

## Next phase

The next phase is **Data Contract v1 materialization and compatibility proof**, still before production migration.

Recommended sequence:

1. Convert approved CDR semantics into exact field contracts, keys, nullability and enums.
2. Build an in-memory/staging representation of the nine entities.
3. Populate it from current legacy data and selected public ETL sources.
4. Build the canonical compatibility reconstruction.
5. Compare reconstructed VDE/FuelCons application-facing surfaces with current legacy surfaces.
6. Exercise relationship semantics, especially VDE â†” FuelCons and FuelCons â†” adopted RUN lineage.
7. Record intentional contract corrections.
8. Only after compatibility proof passes, propose DDL and migrations.

**Do not modify production databases or broadly refactor application pages before this proof passes.**

