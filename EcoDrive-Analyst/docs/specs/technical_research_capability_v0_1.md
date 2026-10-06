# Technical Research Capability v0.1

## Purpose

`capabilities.technical_research` is an isolated, evidence-first research package.
Its public single-request API is:

```python
research_technical_component(request) -> TechnicalResearchResult
```

Batch callers may use `research_batch(requests)` for the original tuple result
or `research_batch_with_summary(requests)` for the same results plus typed
status, confidence, source, conflict, cache, and request-reuse counts. None of
these functions writes to an EcoDrive database. Results are review candidates,
never canonical truth.

## MVP scope

- typed request, source, claim, conflict, candidate, and result contracts;
- a LangGraph workflow with bounded search rounds and explicit state;
- provider-neutral search, fetch, extraction, retrieval, and vector interfaces;
- deterministic source policy, application matching, conflict handling, and
  confidence classification;
- an independent SQLite evidence/cache store;
- retrieval before external search;
- transmission as the first domain profile;
- read-only EcoDrive benchmark adapter and CSV artifacts.

## Limits

Defaults are three search rounds, five queries per round, twelve fetched
sources, and five high-quality sources used. Reaching a limit returns an
explicit insufficient result rather than continuing.

## Status semantics

`SUPPORTED`, `PARTIALLY_SUPPORTED`, `CONFLICTING_EVIDENCE`,
`INSUFFICIENT_EVIDENCE`, `NOT_FOUND`, and `ERROR` are the only result statuses.
Identity confidence is independently reported as `DIRECT`, `STRONG`, `WEAK`,
or `UNRESOLVED`.

## Non-goals

No canonical write, automatic approval, transmission-drag estimation, crawler,
knowledge graph, agent swarm, or production promotion exists in v0.1.

## Current provider status

The safe default runtime remains offline and deterministic. The later v0.2.1
implementation also provides an explicit opt-in `live_runtime()` using DDGS for
search and the OpenAI API for structured extraction. It requires
`OPENAI_API_KEY`; model and effort default to `gpt-5.6-terra` and `medium` and
may be overridden with `TECHNICAL_RESEARCH_MODEL` and
`TECHNICAL_RESEARCH_REASONING_EFFORT`. Credentials are read from the process
environment and are not persisted in evidence, cache, or benchmark artifacts.
