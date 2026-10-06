# Technical Research Capability v0.1 — Current Reconciliation

## Purpose

This audit reconciles the original v0.1 capability package with the current
v0.2.1 implementation. The historical v0.1 result remains unchanged as a record
of its original offline benchmark; this document reports current capability.

## Requirement reconciliation

- The isolated `capabilities.technical_research` package remains structurally
  unable to write EcoDrive canonical databases.
- Typed request, source, claim, conflict, candidate, result, batch-summary, and
  LangGraph state contracts are present.
- The bounded LangGraph workflow includes local retrieval, external search,
  source classification and policy, attachment-aware fetching, claim extraction,
  application matching, conflict resolution, ingestion evaluation, and finalize.
- Default limits remain 3 search rounds, 5 queries per round, 12 fetched sources,
  and 5 high-quality sources used.
- The default runtime is offline. Live execution is explicit and uses DDGS plus
  OpenAI structured extraction with Terra/medium defaults.
- The independent SQLite evidence/cache store, deterministic lexical retrieval,
  vectorization boundary, source deduplication, cache provenance, and
  `force_refresh` behavior remain present.
- `research_batch()` remains backward compatible. The additive
  `research_batch_with_summary()` contract now supplies the batch summary
  required by v0.1 without changing result semantics.
- Transmission remains the first domain profile. Canonical inputs are read-only,
  and research output remains a review candidate rather than canonical truth.

## Current benchmark evidence

The current live evidence is the v0.2.1 bounded 15-configuration BMW benchmark,
not the historical fixture-only v0.1 run. It completed 230 audited search
attempts across the main pass and two targeted retries, fetched 38 technical
sources, and produced 5 `WEAK` and 10 `UNRESOLVED` identities. It produced no
`DIRECT` or `STRONG` hardware identity, so the transmission grouping signal
experiment remains not ready to rerun. Full details and artifact interpretation
are in `TECHNICAL_RESEARCH_CAPABILITY_V0_2_1_RESULT.md`.

## Verification

The additive batch-summary test proves duplicate normalized requests are reused,
distinct applications remain distinct, all results are counted, and canonical
writes remain disabled.

Focused capability suite:

```text
python -m pytest capabilities/technical_research/tests -q
40 passed, 7 subtests passed
```

Canonical database SHA256 values remained equal to the pre-execution baseline:

- `data/db/eco_drive.db`:
  `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB`
- `data/db/eco_drive_qa.db`:
  `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB`
- `data/db/staging/eco_drive_canonical_candidate.db`:
  `1E3B251F39B21A835EF9F0B4F6291A495636A8C425CEF7051754D1D750840ACF`

```ini
TECHNICAL_RESEARCH_CAPABILITY_CREATED = YES
LANGGRAPH_WORKFLOW_READY = YES
LANGCHAIN_INTEGRATION_READY = YES
EXTERNAL_SEARCH_LIVE = YES
LOCAL_RETRIEVAL_READY = YES
KNOWLEDGE_INGESTION_HOOK_READY = YES
VECTOR_RETRIEVAL_INTEGRATED = PARTIAL
SOURCE_POLICY_READY = YES
CONFLICT_POLICY_READY = YES
TRANSMISSION_PROFILE_READY = YES
BATCH_SUMMARY_READY = YES
BMW_TRANSMISSION_BENCHMARK_COMPLETE = YES
DIRECT_OR_STRONG_IDENTITIES = 0
UNRESOLVED_IDENTITIES = 10
READY_TO_RE_RUN_TRANSMISSION_SIGNAL_EXPERIMENT = NO
PRODUCTION_DB_CHANGED = NO
```
