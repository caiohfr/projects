# Technical Research Capability v0.1 — Implementation Result

Date: 2026-09-18

## A. Architecture

The capability is isolated under `capabilities/technical_research` and exposes:

- `research_technical_component(request) -> TechnicalResearchResult`
- `research_batch(requests) -> tuple[TechnicalResearchResult, ...]`

The package contains ordinary Python domain contracts and deterministic policy
modules, provider-neutral tool boundaries, an isolated evidence store, a
transmission research profile, an EcoDrive read-only adapter, and a LangGraph
workflow. It does not import or write EcoDrive persistence repositories.

Graph path:

1. `validate_request`
2. `retrieve_local_knowledge`
3. local hit: `extract_claims`; otherwise `build_search_queries`
4. `external_search`
5. `source_policy_filter`
6. `fetch_selected_sources`
7. `extract_claims`
8. `verify_and_resolve`
9. bounded `refine_queries` loop or `evaluate_ingestion`
10. `ingest_knowledge`
11. `finalize`

LangGraph owns orchestration and bounded routing. Contracts, matching, source
policy, evidence policy, conflict handling, scoring, validation, hashing,
deduplication, chunking, cache, and ingestion eligibility remain deterministic
Python services. LangChain is restricted to provider/model adapters and
structured-output extraction.

## B. Search provider

No live search provider or model credential was present in the repository or
environment. The implementation therefore uses a provider-neutral
`SearchProvider` protocol with:

- `NullSearchProvider` as the safe default;
- `FixtureSearchProvider` for deterministic tests;
- `LangChainSearchProviderAdapter` for an injected LangChain Runnable/tool.

No provider-specific environment variable is hardcoded. Required variables are
defined by the provider selected by the caller. Without an injected provider,
the graph stops with `INSUFFICIENT_EVIDENCE` instead of inventing identities.

## C. Knowledge / RAG

No reusable project-wide chunking/vectorization stack was found. A minimal,
isolated evidence-store boundary was added:

- SQLite evidence/cache store outside canonical truth;
- canonical URL and content hashing;
- duplicate detection before chunking/vectorization;
- deterministic chunking;
- `Vectorizer` protocol and `NullVectorizer` default;
- local retriever protocol and retrieval-first graph routing;
- explicit ingestion decisions: `INGEST`, `METADATA_ONLY`, `EPHEMERAL`,
  `REJECT`.

Vector retrieval is therefore PARTIAL: integration boundaries and tests exist,
but no production embedding model/index was configured.

## D. Source and conflict policy

Implemented tiers:

- `TIER_1_PRIMARY`
- `TIER_2_STRONG_SECONDARY`
- `TIER_3_DISCOVERY_ONLY`
- `TIER_4_WEAK`
- `UNCLASSIFIED`

Unsafe, inaccessible, irrelevant, duplicate, or policy-rejected documents are
not vectorized. Higher source tier never overrides an application mismatch.
Unresolved high-quality disagreements preserve both claim sets and return
`CONFLICTING_EVIDENCE`; the model is not allowed to silently choose a winner.

## E. BMW transmission benchmark

The deterministic EcoDrive adapter selected 15 BMW configurations across 9
previous `STRICT_CANDIDATE` groups, including repeated and ambiguous groups.
The offline preflight completed the full result/audit artifact pipeline.

Because no live search/model provider and no populated local evidence corpus
were available, all 15 requests correctly stopped as
`INSUFFICIENT_EVIDENCE`/`UNRESOLVED`:

- DIRECT: 0
- STRONG: 0
- WEAK: 0
- UNRESOLVED: 15
- groups confirmed by external evidence: 0
- groups split by external evidence: 0

This is not considered a completed external-research benchmark. The artifacts
are deliberately explicit about the offline preflight status.

## F. Tests and regression impact

Focused capability tests:

- 14/14 passed in 0.947 s;
- compilation check passed;
- source policy, application matching, conflicts, non-hallucination, exact
  evidence retention, deduplication, vectorization rejection, retrieval-first,
  batch reuse, cache, limits, LangChain structured output, BMW sampling, and
  canonical DB immutability are covered.

Main application regression:

- 1847/1847 passed.

Additional historical ETL suite:

- 264 collected; 254 passed, 4 failed, 6 errored;
- five errors were caused by missing optional test dependencies (`openpyxl` or
  `matplotlib`);
- one error was a historical Sprint 12C schema-contract assertion;
- four failures were stale hash/status snapshot assertions against the current
  Sprint 12H database/artifact state;
- none of those failures executes or references the new technical-research
  package.

Database hashes before and after implementation, benchmark, and tests:

- staging candidate: `1E3B251F39B21A835EF9F0B4F6291A495636A8C425CEF7051754D1D750840ACF`
- production: `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB`
- QA: `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB`

All three hashes remained unchanged.

## G. Status

```ini
TECHNICAL_RESEARCH_CAPABILITY_CREATED = YES
LANGGRAPH_WORKFLOW_READY = YES
LANGCHAIN_INTEGRATION_READY = YES

EXTERNAL_SEARCH_LIVE = NO
LOCAL_RETRIEVAL_READY = YES
KNOWLEDGE_INGESTION_HOOK_READY = YES
VECTOR_RETRIEVAL_INTEGRATED = PARTIAL

SOURCE_POLICY_READY = YES
CONFLICT_POLICY_READY = YES
CANONICAL_WRITE_DISABLED = YES

BMW_TRANSMISSION_BENCHMARK_COMPLETE = NO
DIRECT_OR_STRONG_IDENTITIES = 0
UNRESOLVED_IDENTITIES = 15

READY_TO_RE-RUN_TRANSMISSION_SIGNAL_EXPERIMENT = NO
PRODUCTION_DB_CHANGED = NO
```

The experiment must not be re-run as an identity-enriched analysis until a
live provider or an adequately populated local evidence corpus produces
reviewable DIRECT/STRONG results.
