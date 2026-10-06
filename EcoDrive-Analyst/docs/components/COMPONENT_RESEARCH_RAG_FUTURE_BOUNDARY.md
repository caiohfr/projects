# Component Research/RAG — Future Boundary

## Purpose

Future Research/RAG may discover and structure Component evidence, but it must
never write canonical truth directly.

The only permitted future flow is:

```text
technical source
  -> retrieval/search
  -> evidence chunk with exact source location
  -> structured component candidate
  -> deterministic validation and unit normalization
  -> human engineering approval
  -> canonical component_db / component_instance / component_resolution
```

## Candidate contract

A future candidate should carry, when supported:

- source document identity, version, URL/path, page/table/section and access
  date;
- exact evidence excerpt or structured cell reference;
- proposed family, subtype, position, role and physical boundary;
- application/architecture/compatibility evidence;
- hardware/manufacturer/part identifiers exactly as sourced;
- raw numeric value, raw unit, speed/basis and observed/derived status;
- proposed normalized value plus deterministic conversion version;
- retrieval/extraction confidence separated from engineering confidence;
- conflicts, aliases, missing fields and required human decisions.

The candidate is evidence for review, not a canonical component.

## Deterministic validation boundary

Before approval, non-AI validation must enforce:

- allowed taxonomy and units;
- dimensional conversion;
- physical-boundary completeness;
- Brake Baseline versus Brake Standard separation;
- position and driveline-architecture comparability;
- stable source identity and duplicate handling;
- negative/zero preservation and explicit outlier flags;
- no whole-vehicle roadload-to-component causal inference;
- no VDE adoption without exact evidence.

Human approval records the accepted candidate, rejected alternatives, reason,
reviewer, and time. Canonical provenance links back to the approved evidence.

## Explicitly not implemented in Sprint 12

- embeddings or vector databases;
- document chunking pipelines;
- LLM extraction;
- web-search automation;
- autonomous research agents;
- source-ranking or entity-resolution models;
- direct candidate-to-canonical writes.

These remain future work after the deterministic Component baseline has a
real, reviewed gold set.
