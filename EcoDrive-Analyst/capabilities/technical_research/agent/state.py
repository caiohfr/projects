from __future__ import annotations

from typing import Any, TypedDict

from ..contracts import (
    EvidenceClaim,
    EvidenceConflict,
    FetchedDocument,
    SourcePolicyDecision,
    SourceRecord,
    TechnicalCandidate,
    RequestIdentityAudit,
    TechnicalResearchRequest,
    TechnicalResearchResult,
)


class TechnicalResearchState(TypedDict, total=False):
    request: TechnicalResearchRequest
    request_identity_audit: RequestIdentityAudit
    run_id: str
    research_profile: str
    local_retrieval_hits: list[FetchedDocument]
    search_queries: list[str]
    discovered_sources: list[SourceRecord]
    all_discovered_sources: list[SourceRecord]
    classified_sources: list[SourceRecord]
    all_classified_sources: list[SourceRecord]
    accepted_sources: list[SourceRecord]
    all_accepted_sources: list[SourceRecord]
    rejected_sources: list[SourcePolicyDecision]
    all_rejected_sources: list[SourcePolicyDecision]
    fetched_documents: list[FetchedDocument]
    all_fetched_documents: list[FetchedDocument]
    extracted_claims: list[EvidenceClaim]
    application_match_audit: list[dict[str, Any]]
    candidate_hypotheses: list[TechnicalCandidate]
    conflicts: list[EvidenceConflict]
    final_candidate: TechnicalCandidate
    status: str
    confidence: str
    search_round: int
    stop_reason: str
    ingestable_sources: list[FetchedDocument]
    ingestion_audit: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    result: TechnicalResearchResult
