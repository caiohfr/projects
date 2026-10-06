# Technical Research Architecture

## Boundary

```text
EcoDrive/public client fields
        |
        v
adapters/ecodrive.py (read-only)
        |
        v
TechnicalResearchRequest
        |
        v
LangGraph orchestration
        |
        +--> deterministic policies/contracts
        +--> local evidence retrieval
        +--> provider-neutral search/fetch/extraction
        +--> optional evidence-store ingestion
        |
        v
TechnicalResearchResult + claim-level evidence
        |
        v
human review / separate future import process
```

There is no dependency from this package into `src/vde_core` or
`src/vde_app`, and there is no database persistence adapter.

## Graph

```text
START -> validate_request -> retrieve_local_knowledge
  local hit -> extract_claims -> match_application -> verify_and_resolve
  no hit    -> build_search_queries -> external_search
            -> classify_sources -> source_policy_filter
            -> fetch_selected_sources (including constrained technical attachments)
            -> extract_claims -> match_application -> verify_and_resolve
  insufficient + rounds remain -> refine_queries -> external_search
  otherwise -> evaluate_ingestion -> ingest_knowledge -> finalize -> END
```

`validate_request` also records a deterministic request self-consistency audit.
Application matching ignores only the structured fields identified as internally
conflicting; it preserves both request values and the conflict record.

The graph uses a typed `TechnicalResearchState`. Trace entries store node names,
counts, decisions, and stop reasons, not chain-of-thought.

## Responsibilities

Agentic/provider boundary:

- query planning profile;
- configured external search;
- configured structured claim extraction.

Deterministic boundary:

- request allowlist and validation;
- source policy;
- application matching;
- conflict resolution;
- identity confidence;
- hashing, deduplication, chunking, cache, and ingestion;
- search and source limits;
- EcoDrive review transformation.
- transmission hardware-versus-marketing semantics;
- independent-application hardware-group validation;
- final-attempt benchmark consolidation.

`research_batch()` preserves the original tuple API. The additive
`research_batch_with_summary()` API reports batch status and reuse without
changing execution or canonical boundaries.

## Knowledge layer

`SQLiteEvidenceStore` owns its own database and never opens EcoDrive canonical
files. It stores source identity, content hash, metadata, chunks, and cached
results. Retrieval is deterministic lexical retrieval in v0.1. `Vectorizer` is
an optional interface; `NullVectorizer` is the default, so vector retrieval is
currently partial rather than falsely claimed as complete.

## Security

- only public allowlisted vehicle/application fields reach providers;
- fetched source content is treated as untrusted data;
- fetch blocks non-HTTP schemes and non-global IP addresses;
- response size, timeout, MIME type, and redirects are controlled;
- canonical writes are structurally absent.
