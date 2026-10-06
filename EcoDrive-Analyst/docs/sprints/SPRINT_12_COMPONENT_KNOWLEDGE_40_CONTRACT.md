# Sprint 12 — Component Knowledge 40/60 Contract

## Decision

The Component knowledge model has ten dimensions. Dimensions 1–4 define the
deterministic Sprint 12 baseline; Dimensions 5–10 belong to a future
Research/RAG layer. The percentages describe knowledge-model coverage, not
field completeness, market coverage, or a target row count.

The current local repository does not contain the historical PL raw source,
analysis notebooks, or complete methodology needed to implement the four
deterministic dimensions. The contract is frozen here, but canonical
population is blocked rather than synthesized.

## Non-negotiable rules

- A label is not proof of a physically comparable component.
- Component loss is a curve `F(v) = A + B*v + C*v²` with canonical units N,
  N/kph, and N/kph².
- Force at 50 km/h is derived from canonical ABC; it is not an independent
  measured quantity unless a source explicitly identifies it as such.
- Brake Baseline/as-received and Brake Standard/controlled are distinct and
  must never share a benchmark population.
- Transmission, axle, differential, hub, and bearing records are grouped only
  with compatible physical boundary, position, architecture, method, and test
  condition.
- Negative, zero, and outlier observations are retained and reviewed. Only an
  evidence-backed `INVALID_CONFIRMED` record may be excluded from reference
  statistics.
- Tire laboratory knowledge remains in `tire_db`.
- EPA/JRC whole-vehicle roadload ABC never becomes Component Resolution ABC.
- Synthetic QA fixtures and default-table priors never become canonical truth.
- No reusable hardware identity is created without source evidence.

## Entity ownership

| Entity | Owns | Does not own |
|---|---|---|
| `component_db` | Defensible reusable hardware/catalog identity | Generic labels, vehicle-only applications, loss curves |
| `component_instance` | Observable installed/application context | Fabricated master identity, component loss result |
| `component_resolution` | Evidence-backed physical loss result, boundary, conditions and method | Whole-vehicle roadload split, unsupported fuzzy attribution |
| `vde_component_resolution` | Explicit, defensible VDE adoption | Similarity-based attachment |
| `tire_db` | Tire-specific laboratory/engineering knowledge | Generic Component catalog rows |

## Ten dimensions

| # | Dimension | Layer | Current status | Acceptance meaning |
|---:|---|---|---|---|
| 1 | Component identity / taxonomy | NOW | `PARTIAL_NOW` | Domain and role exist for application instances; reusable hardware remains unresolved. |
| 2 | Application / physical context | NOW | `PARTIAL_NOW` | JRC application context exists; PL boundary/configuration-subtraction evidence is missing. |
| 3 | Engineering loss behavior | NOW | `NOT_SUPPORTED_BY_CURRENT_SOURCE` | No real local component ABC/force-curve source exists. |
| 4 | Methodology / provenance / quality | NOW | `PARTIAL_NOW` | Existing instance provenance and a high-level interface exist; the referenced upstream method/source package is absent. |
| 5 | Supplier/manufacturer/part-number enrichment | LATER | `DEFERRED_RAG` | Candidate evidence only until deterministic validation and approval. |
| 6 | Detailed specifications/maps/datasheets | LATER | `DEFERRED_RAG` | Source-backed future enrichment. |
| 7 | Document evidence/chunks/quotations/links | LATER | `DEFERRED_RAG` | Retrieval evidence remains separate from canonical data. |
| 8 | Cross-source matching/aliases/equivalence | LATER | `DEFERRED_RAG` | No fuzzy identity resolution in this package. |
| 9 | Enriched compatibility/topology/applicability | LATER | `DEFERRED_RAG` | Requires reviewed evidence and deterministic rules. |
| 10 | Retrieval/extraction/AI confidence metadata | LATER | `DEFERRED_RAG` | RAG/agent implementation is out of scope. |

No dimension is counted as fully implemented while its required evidence is
missing. Current strict completion is therefore `0/4`, with three dimensions
partially represented by existing canonical application context.

## Evidence gate for future population

A numeric row may enter `component_resolution` only when the source package
establishes all applicable items below:

1. stable source record identity and source file/version;
2. component family/subtype and physical boundary;
3. vehicle/application context and driveline architecture;
4. component position or an explicit combined boundary;
5. configuration-from/configuration-to subtraction method;
6. test condition and observed/derived status;
7. raw A/B/C or force-at-speed values with units and speed basis;
8. deterministic unit conversion with raw values preserved in provenance;
9. review/confidence status and known limitations;
10. exact adoption evidence before any VDE relationship is created.

Absent any required evidence, the row remains an external candidate or is
blocked; the evidence standard is not relaxed to reach a row-count target.

## Current blocker

The following referenced upstream material is not local:

- the historical PL raw Excel/CSV source;
- the PL analysis notebooks that produce validated comparable populations;
- `Parasitic_Loss_Testing_Methodology(1).md`;
- source metadata establishing units, configuration subtraction, physical
  boundary, architecture, position, conditions, and source record identity.

Until those artifacts are supplied, Component population, reference
populations, outlier analysis, and the official rule-based notebook remain
blocked.
