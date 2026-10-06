from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
import json
from typing import Iterable
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from ..contracts import (
    IdentityConfidence,
    ResearchStatus,
    TechnicalResearchBatchResult,
    TechnicalResearchBatchSummary,
    TechnicalResearchRequest,
    TechnicalResearchResult,
)
from ..knowledge import request_fingerprint
from .nodes import build_nodes
from .routing import route_after_local, route_after_verification
from .runtime import ResearchRuntime, default_runtime
from .state import TechnicalResearchState


def build_research_graph(runtime: ResearchRuntime):
    nodes = build_nodes(runtime)
    builder = StateGraph(TechnicalResearchState)
    for name, node in nodes.items():
        builder.add_node(name, node)
    builder.add_edge(START, "validate_request")
    builder.add_edge("validate_request", "retrieve_local_knowledge")
    builder.add_conditional_edges(
        "retrieve_local_knowledge",
        route_after_local,
        {
            "extract_claims": "extract_claims",
            "build_search_queries": "build_search_queries",
        },
    )
    builder.add_edge("build_search_queries", "external_search")
    builder.add_edge("external_search", "classify_sources")
    builder.add_edge("classify_sources", "source_policy_filter")
    builder.add_edge("source_policy_filter", "fetch_selected_sources")
    builder.add_edge("fetch_selected_sources", "extract_claims")
    builder.add_edge("extract_claims", "match_application")
    builder.add_edge("match_application", "verify_and_resolve")
    builder.add_conditional_edges(
        "verify_and_resolve",
        lambda state: route_after_verification(state, runtime),
        {
            "refine_queries": "refine_queries",
            "evaluate_ingestion": "evaluate_ingestion",
        },
    )
    builder.add_edge("refine_queries", "build_search_queries")
    builder.add_edge("evaluate_ingestion", "ingest_knowledge")
    builder.add_edge("ingest_knowledge", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


def research_technical_component(
    request: TechnicalResearchRequest,
    *,
    runtime: ResearchRuntime | None = None,
) -> TechnicalResearchResult:
    selected_runtime = runtime or default_runtime()
    key = _request_key(request)
    if selected_runtime.cache_store is not None and not request.force_refresh:
        cached_payload = selected_runtime.cache_store.cache_get(key)
        if cached_payload is not None:
            cached = TechnicalResearchResult.from_dict(cached_payload)
            return replace(
                cached,
                request_id=request.request_id,
                provenance={**cached.provenance, "cache_hit": True},
                trace=(*cached.trace, {"node": "result_cache", "status": "HIT"}),
            )
    graph = build_research_graph(selected_runtime)
    final_state = graph.invoke(
        {
            "request": request,
            "run_id": f"trun-{uuid4().hex}",
            "trace": [],
            "extracted_claims": [],
        }
    )
    result = final_state["result"]
    if selected_runtime.cache_store is not None:
        selected_runtime.cache_store.cache_put(key, result.to_dict())
    return result


def research_batch(
    requests: Iterable[TechnicalResearchRequest],
    *,
    runtime: ResearchRuntime | None = None,
) -> tuple[TechnicalResearchResult, ...]:
    selected_runtime = runtime or default_runtime()
    memo: dict[str, TechnicalResearchResult] = {}
    results: list[TechnicalResearchResult] = []
    for request in requests:
        key = _request_key(request)
        if key in memo and not request.force_refresh:
            cached = replace(memo[key], request_id=request.request_id)
            results.append(cached)
            continue
        result = research_technical_component(request, runtime=selected_runtime)
        memo[key] = result
        results.append(result)
    return tuple(results)


def summarize_research_batch(
    results: Iterable[TechnicalResearchResult],
    *,
    unique_request_count: int | None = None,
) -> TechnicalResearchBatchSummary:
    batch = tuple(results)
    statuses = Counter(result.status.value for result in batch)
    confidences = Counter(result.candidate.confidence.value for result in batch)
    domains = Counter(result.domain for result in batch)
    source_ids = {
        claim.source_id
        for result in batch
        for claim in result.evidence.claims
    }
    return TechnicalResearchBatchSummary(
        request_count=len(batch),
        unique_request_count=(
            len(batch) if unique_request_count is None else unique_request_count
        ),
        reused_request_count=(
            0 if unique_request_count is None else len(batch) - unique_request_count
        ),
        status_counts={
            status.value: statuses[status.value] for status in ResearchStatus
        },
        confidence_counts={
            confidence.value: confidences[confidence.value]
            for confidence in IdentityConfidence
        },
        domain_counts=dict(sorted(domains.items())),
        unique_source_count=len(source_ids),
        conflict_count=sum(len(result.evidence.conflicts) for result in batch),
        cache_hit_count=sum(bool(result.provenance.get("cache_hit")) for result in batch),
    )


def research_batch_with_summary(
    requests: Iterable[TechnicalResearchRequest],
    *,
    runtime: ResearchRuntime | None = None,
) -> TechnicalResearchBatchResult:
    request_batch = tuple(requests)
    results = research_batch(request_batch, runtime=runtime)
    return TechnicalResearchBatchResult(
        results=results,
        summary=summarize_research_batch(
            results,
            unique_request_count=len({_request_key(request) for request in request_batch}),
        ),
    )


def _request_key(request: TechnicalResearchRequest) -> str:
    normalized = json.dumps(
        {
            "contract_version": "technical_research_v0_2_2",
            "domain": request.domain.upper(),
            "known_fields": dict(sorted(request.known_fields.items())),
            "target_fields": request.target_fields,
            "limits": asdict(request.limits),
        },
        sort_keys=True,
        default=str,
    )
    return request_fingerprint(normalized)
