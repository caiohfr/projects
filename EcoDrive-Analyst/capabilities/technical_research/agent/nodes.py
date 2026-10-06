from __future__ import annotations

from dataclasses import asdict, replace
import math
from typing import Any, Callable

from ..contracts import (
    EvidenceBundle,
    IdentityConfidence,
    ResearchStatus,
    SourceDecision,
    TechnicalCandidate,
    TechnicalResearchResult,
)
from ..core import (
    audit_research_request,
    evaluate_evidence,
    match_application as deterministic_match,
    normalize_transmission_identity_claim,
    source_with_classification,
    validate_request,
)
from ..tools import discover_technical_attachments
from .runtime import ResearchRuntime
from .state import TechnicalResearchState


Node = Callable[[TechnicalResearchState], dict[str, Any]]


def _trace(state: TechnicalResearchState, node: str, **details: Any) -> list[dict[str, Any]]:
    return [*state.get("trace", []), {"node": node, **details}]


def _unique_sources(sources: list[Any]) -> list[Any]:
    by_id: dict[str, Any] = {}
    for source in sources:
        by_id.setdefault(source.source_id, source)
    return list(by_id.values())


def _unique_documents(documents: list[Any]) -> list[Any]:
    by_id: dict[str, Any] = {}
    for document in documents:
        by_id.setdefault(document.source.source_id, document)
    return list(by_id.values())


def _source_priority(source: Any) -> tuple[int, int, int, str]:
    document_type = str(
        source.metadata.get("classified_document_type", source.document_type)
    ).upper()
    is_attachment = bool(source.metadata.get("is_technical_attachment"))
    is_technical_document = is_attachment or "PDF" in document_type or "TECHNICAL_DOCUMENT" in document_type
    is_structured_technical = "TECHNICAL_PAGE" in document_type or "DATABASE" in document_type
    is_article_landing = "/article/detail/" in source.url.lower()
    format_rank = (
        0 if is_technical_document
        else 1 if is_structured_technical
        else 2 if is_article_landing
        else 3
    )
    tier_rank = 0 if source.tier.value == "TIER_1_PRIMARY" else 1
    return (
        tier_rank,
        format_rank,
        int(source.metadata.get("search_rank", 999)),
        source.source_id,
    )


def build_nodes(runtime: ResearchRuntime) -> dict[str, Node]:
    def validate_request(state: TechnicalResearchState) -> dict[str, Any]:
        request = state["request"]
        validate_request_contract(request)
        request_audit = audit_research_request(request)
        return {
            "research_profile": runtime.profile.domain,
            "request_identity_audit": request_audit,
            "search_round": 0,
            "trace": _trace(
                state,
                "validate_request",
                status="VALID",
                request_consistency=request_audit.status.value,
                request_conflicts=len(request_audit.conflicts),
            ),
        }

    def retrieve_local_knowledge(state: TechnicalResearchState) -> dict[str, Any]:
        request = state["request"]
        hits = list(
            runtime.local_retriever.retrieve(
                request, limit=request.limits.max_high_quality_sources_used
            )
        )
        return {
            "local_retrieval_hits": hits,
            "fetched_documents": hits,
            "all_fetched_documents": hits,
            "trace": _trace(state, "retrieve_local_knowledge", hits=len(hits)),
        }

    def build_search_queries(state: TechnicalResearchState) -> dict[str, Any]:
        round_number = max(1, state.get("search_round", 0))
        request = state["request"]
        queries = list(runtime.profile.build_queries(request, round_number=round_number))[
            : request.limits.max_search_queries_per_round
        ]
        return {
            "search_round": round_number,
            "search_queries": queries,
            "trace": _trace(state, "build_search_queries", round=round_number, queries=queries),
        }

    def external_search(state: TechnicalResearchState) -> dict[str, Any]:
        request = state["request"]
        discovered: list[Any] = []
        all_discovered = list(state.get("all_discovered_sources", []))
        seen = {source.source_id for source in all_discovered}
        failures: list[dict[str, str]] = []
        discovery_cap = max(
            request.limits.max_sources_fetched * 3,
            request.limits.max_high_quality_sources_used * 5,
        )
        if runtime.search_provider.available:
            for query in state.get("search_queries", []):
                remaining = discovery_cap - len(discovered)
                if remaining <= 0:
                    break
                try:
                    results = runtime.search_provider.search(query, limit=min(5, remaining))
                except Exception as exc:
                    failures.append({"query": query, "error": type(exc).__name__})
                    continue
                for source in results:
                    if source.source_id not in seen:
                        seen.add(source.source_id)
                        discovered.append(
                            replace(
                                source,
                                metadata={
                                    **source.metadata,
                                    "search_round": state.get("search_round", 1),
                                },
                            )
                        )
        all_discovered = _unique_sources([*all_discovered, *discovered])
        return {
            "discovered_sources": discovered,
            "all_discovered_sources": all_discovered,
            "trace": _trace(
                state,
                "external_search",
                provider_available=runtime.search_provider.available,
                discovered=len(discovered),
                failures=failures,
            ),
        }

    def source_policy_filter(state: TechnicalResearchState) -> dict[str, Any]:
        decisions = [runtime.source_policy.evaluate(source) for source in state.get("classified_sources", [])]
        accepted = [
            decision.source
            for decision in decisions
            if decision.decision == SourceDecision.ACCEPT
        ]
        accepted.sort(key=_source_priority)
        accepted = accepted[: state["request"].limits.max_sources_fetched * 2]
        rejected = [decision for decision in decisions if decision.decision != SourceDecision.ACCEPT]
        all_accepted = _unique_sources([*state.get("all_accepted_sources", []), *accepted])
        rejected_by_id = {
            decision.source.source_id: decision
            for decision in [*state.get("all_rejected_sources", []), *rejected]
        }
        return {
            "accepted_sources": accepted,
            "rejected_sources": rejected,
            "all_accepted_sources": all_accepted,
            "all_rejected_sources": list(rejected_by_id.values()),
            "trace": _trace(
                state,
                "source_policy_filter",
                accepted=len(accepted),
                rejected=len(rejected),
            ),
        }

    def classify_sources(state: TechnicalResearchState) -> dict[str, Any]:
        classified = [source_with_classification(source) for source in state.get("discovered_sources", [])]
        all_classified = _unique_sources([*state.get("all_classified_sources", []), *classified])
        counts: dict[str, int] = {}
        for source in classified:
            label = str(source.metadata.get("source_classification", "UNCLASSIFIED"))
            counts[label] = counts.get(label, 0) + 1
        return {
            "classified_sources": classified,
            "all_classified_sources": all_classified,
            "trace": _trace(state, "classify_sources", counts=counts),
        }

    def fetch_selected_sources(state: TechnicalResearchState) -> dict[str, Any]:
        documents = list(state.get("fetched_documents", []))
        all_documents = list(state.get("all_fetched_documents", []))
        failures: list[dict[str, str]] = []
        fetched_ids = {document.source.source_id for document in all_documents}
        queue = list(state.get("accepted_sources", []))
        queued_ids = {source.source_id for source in queue}
        classified = list(state.get("classified_sources", []))
        accepted = list(state.get("accepted_sources", []))
        rejected = list(state.get("rejected_sources", []))
        all_discovered = list(state.get("all_discovered_sources", []))
        all_classified = list(state.get("all_classified_sources", []))
        all_accepted = list(state.get("all_accepted_sources", []))
        all_rejected = list(state.get("all_rejected_sources", []))
        attachment_count = 0
        round_fetches = 0
        remaining_high_quality = max(
            0,
            state["request"].limits.max_high_quality_sources_used - len(all_documents),
        )
        rounds_remaining = max(
            1,
            state["request"].limits.max_search_rounds - state.get("search_round", 1) + 1,
        )
        round_fetch_limit = math.ceil(remaining_high_quality / rounds_remaining)
        index = 0
        while index < len(queue):
            source = queue[index]
            index += 1
            if source.source_id in fetched_ids:
                continue
            if round_fetches >= round_fetch_limit:
                break
            if len(all_documents) >= state["request"].limits.max_high_quality_sources_used:
                break
            if len(all_documents) >= state["request"].limits.max_sources_fetched:
                break
            try:
                document = runtime.document_fetcher.fetch(source)
                documents.append(document)
                all_documents.append(document)
                fetched_ids.add(source.source_id)
                round_fetches += 1
                attachments = discover_technical_attachments(document)
                for attachment in attachments:
                    if attachment.source_id in {item.source_id for item in all_discovered}:
                        continue
                    attachment = replace(
                        attachment,
                        metadata={
                            **attachment.metadata,
                            "search_round": state.get("search_round", 1),
                            "discovered_from_source_id": source.source_id,
                        },
                    )
                    classified_attachment = source_with_classification(attachment)
                    decision = runtime.source_policy.evaluate(classified_attachment)
                    all_discovered.append(attachment)
                    classified.append(classified_attachment)
                    all_classified.append(classified_attachment)
                    attachment_count += 1
                    if decision.decision == SourceDecision.ACCEPT:
                        accepted.append(classified_attachment)
                        all_accepted.append(classified_attachment)
                        if classified_attachment.source_id not in queued_ids:
                            queue.insert(index, classified_attachment)
                            queued_ids.add(classified_attachment.source_id)
                    else:
                        rejected.append(decision)
                        all_rejected.append(decision)
            except Exception as exc:  # provider boundary is recorded, not hidden
                failures.append({"source_id": source.source_id, "error": type(exc).__name__})
        return {
            "fetched_documents": documents,
            "all_fetched_documents": _unique_documents(all_documents),
            "discovered_sources": _unique_sources([*state.get("discovered_sources", []), *all_discovered]),
            "classified_sources": _unique_sources(classified),
            "accepted_sources": sorted(_unique_sources(accepted), key=_source_priority),
            "rejected_sources": list({item.source.source_id: item for item in rejected}.values()),
            "all_discovered_sources": _unique_sources(all_discovered),
            "all_classified_sources": _unique_sources(all_classified),
            "all_accepted_sources": _unique_sources(all_accepted),
            "all_rejected_sources": list({item.source.source_id: item for item in all_rejected}.values()),
            "trace": _trace(
                state,
                "fetch_selected_sources",
                fetched=round_fetches,
                fetched_total=len(_unique_documents(all_documents)),
                attachments_discovered=attachment_count,
                failures=failures,
            ),
        }

    def extract_claims(state: TechnicalResearchState) -> dict[str, Any]:
        existing = list(state.get("extracted_claims", []))
        extracted = tuple(
            normalize_transmission_identity_claim(claim)
            for claim in runtime.claim_extractor.extract(
                state["request"], state.get("fetched_documents", [])
            )
        )
        seen = {
            (claim.source_id, claim.field, str(claim.normalized_value))
            for claim in existing
        }
        for claim in extracted:
            key = (claim.source_id, claim.field, str(claim.normalized_value))
            if key not in seen:
                existing.append(claim)
                seen.add(key)
        return {
            "extracted_claims": existing,
            "trace": _trace(state, "extract_claims", claims=len(existing)),
        }

    def verify_and_resolve(state: TechnicalResearchState) -> dict[str, Any]:
        resolution = evaluate_evidence(
            state.get("extracted_claims", []),
            runtime.profile.identity_field,
            state["request"].target_fields,
        )
        return {
            "final_candidate": resolution.candidate,
            "candidate_hypotheses": [resolution.candidate],
            "conflicts": list(resolution.bundle.conflicts),
            "status": resolution.status.value,
            "confidence": resolution.candidate.confidence.value,
            "trace": _trace(
                state,
                "verify_and_resolve",
                status=resolution.status.value,
                confidence=resolution.candidate.confidence.value,
                conflicts=len(resolution.bundle.conflicts),
            ),
        }

    def match_application(state: TechnicalResearchState) -> dict[str, Any]:
        matched = []
        audit = []
        for claim in state.get("extracted_claims", []):
            decision = deterministic_match(state["request"].known_fields, claim.application_context)
            matched.append(replace(claim, application_match=decision.match))
            audit.append(
                {
                    "source_id": claim.source_id,
                    "field": claim.field,
                    "application_match": decision.match.value,
                    "compared_fields": ";".join(decision.compared_fields),
                    "missing_fields": ";".join(decision.missing_fields),
                    "mismatches": dict(decision.mismatches),
                    "reason": decision.reason,
                    "trusted_request_fields": ";".join(
                        decision.trusted_request_fields
                    ),
                    "conflicting_request_fields": ";".join(
                        decision.conflicting_request_fields
                    ),
                }
            )
        return {
            "extracted_claims": matched,
            "application_match_audit": audit,
            "trace": _trace(state, "match_application", claims=len(matched)),
        }

    def refine_queries(state: TechnicalResearchState) -> dict[str, Any]:
        next_round = state.get("search_round", 0) + 1
        return {
            "search_round": next_round,
            "fetched_documents": [],
            "discovered_sources": [],
            "classified_sources": [],
            "accepted_sources": [],
            "trace": _trace(state, "refine_queries", round=next_round),
        }

    def evaluate_ingestion(state: TechnicalResearchState) -> dict[str, Any]:
        documents = list(state.get("all_fetched_documents", state.get("fetched_documents", [])))
        return {
            "ingestable_sources": documents,
            "trace": _trace(state, "evaluate_ingestion", candidates=len(documents)),
        }

    def ingest_knowledge(state: TechnicalResearchState) -> dict[str, Any]:
        audit: list[dict[str, Any]] = []
        if runtime.ingestion_service is not None:
            for document in state.get("ingestable_sources", []):
                decision = runtime.ingestion_service.ingest(document)
                audit.append(asdict(decision) | {"status": decision.status.value})
        return {
            "ingestion_audit": audit,
            "trace": _trace(state, "ingest_knowledge", actions=len(audit)),
        }

    def finalize(state: TechnicalResearchState) -> dict[str, Any]:
        candidate = state.get(
            "final_candidate",
            TechnicalCandidate(None, {}, IdentityConfidence.UNRESOLVED),
        )
        status = ResearchStatus(state.get("status", ResearchStatus.INSUFFICIENT_EVIDENCE.value))
        if not runtime.search_provider.available and not state.get("local_retrieval_hits"):
            stop_reason = "NO_SEARCH_PROVIDER_AND_NO_LOCAL_EVIDENCE"
        elif state.get("search_round", 0) >= state["request"].limits.max_search_rounds:
            stop_reason = "SEARCH_LIMIT_REACHED"
        elif status == ResearchStatus.SUPPORTED:
            stop_reason = "SUFFICIENT_EVIDENCE"
        elif status == ResearchStatus.CONFLICTING_EVIDENCE:
            stop_reason = "UNRESOLVED_CONFLICT"
        else:
            stop_reason = "INSUFFICIENT_EVIDENCE"
        trace = _trace(state, "finalize", stop_reason=stop_reason)
        all_accepted_sources = state.get("all_accepted_sources", state.get("accepted_sources", []))
        all_rejected_sources = state.get("all_rejected_sources", state.get("rejected_sources", []))
        all_classified_sources = state.get("all_classified_sources", state.get("classified_sources", []))
        all_fetched_documents = state.get("all_fetched_documents", state.get("fetched_documents", []))
        accepted_ids = {source.source_id for source in all_accepted_sources}
        fetched_ids = {document.source.source_id for document in all_fetched_documents}
        rejected_by_id = {
            decision.source.source_id: decision
            for decision in all_rejected_sources
        }
        source_audit = []
        for source in all_classified_sources:
            rejected_decision = rejected_by_id.get(source.source_id)
            source_audit.append(
                {
                    "source_id": source.source_id,
                    "url": source.url,
                    "title": source.title,
                    "publisher": source.publisher,
                    "source_classification": source.metadata.get("source_classification", "UNCLASSIFIED"),
                    "publisher_type": source.metadata.get("publisher_type", "UNKNOWN"),
                    "document_type": source.metadata.get("classified_document_type", source.document_type),
                    "classification_reason": source.metadata.get("classification_reason", ""),
                    "policy_decision": (
                        "ACCEPT" if source.source_id in accepted_ids
                        else rejected_decision.decision.value if rejected_decision else "UNKNOWN"
                    ),
                    "policy_reason": rejected_decision.reason if rejected_decision else "HIGH_QUALITY_SOURCE",
                    "fetched": source.source_id in fetched_ids,
                    "search_round": source.metadata.get("search_round"),
                    "landing_source_id": source.metadata.get("landing_source_id", ""),
                    "discovered_from_source_id": source.metadata.get(
                        "discovered_from_source_id",
                        source.metadata.get("landing_source_id", ""),
                    ),
                }
            )
        result = TechnicalResearchResult(
            run_id=state["run_id"],
            request_id=state["request"].request_id,
            status=status,
            domain=state["request"].domain.upper(),
            candidate=candidate,
            evidence=EvidenceBundle(
                tuple(state.get("extracted_claims", [])),
                tuple(state.get("conflicts", [])),
            ),
            provenance={
                "capability": "technical_research_v0_2_2",
                "canonical_write": "DISABLED",
                "search_rounds": state.get("search_round", 0),
                "request_identity_audit": asdict(state["request_identity_audit"]),
            },
            research_summary=(
                f"{status.value}; identity confidence {candidate.confidence.value}."
            ),
            source_summary={
                "local_hits": len(state.get("local_retrieval_hits", [])),
                "discovered": len(state.get("all_discovered_sources", state.get("discovered_sources", []))),
                "accepted": len(all_accepted_sources),
                "rejected": len(all_rejected_sources),
                "fetched": len(all_fetched_documents),
                "technical_attachments_fetched": sum(
                    bool(document.source.metadata.get("is_technical_attachment"))
                    for document in all_fetched_documents
                ),
                "bmwtechinfo_fetched": sum(
                    "bmwtechinfo.bmwgroup.com" in document.source.url.lower()
                    for document in all_fetched_documents
                ),
                "application_match_audit": state.get("application_match_audit", []),
                "request_identity_audit": asdict(state["request_identity_audit"]),
                "sources": source_audit,
            },
            ingestion_summary={"actions": state.get("ingestion_audit", [])},
            trace=tuple(trace),
            stop_reason=stop_reason,
        )
        return {"result": result, "stop_reason": stop_reason, "trace": trace}

    return {
        "validate_request": validate_request,
        "retrieve_local_knowledge": retrieve_local_knowledge,
        "build_search_queries": build_search_queries,
        "external_search": external_search,
        "classify_sources": classify_sources,
        "source_policy_filter": source_policy_filter,
        "fetch_selected_sources": fetch_selected_sources,
        "extract_claims": extract_claims,
        "match_application": match_application,
        "verify_and_resolve": verify_and_resolve,
        "refine_queries": refine_queries,
        "evaluate_ingestion": evaluate_ingestion,
        "ingest_knowledge": ingest_knowledge,
        "finalize": finalize,
    }


def validate_request_contract(request: Any) -> None:
    validate_request(request)
