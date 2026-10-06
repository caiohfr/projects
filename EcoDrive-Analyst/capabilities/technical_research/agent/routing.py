from __future__ import annotations

from ..contracts import ApplicationMatch, ResearchStatus, SourceTier
from .runtime import ResearchRuntime
from .state import TechnicalResearchState


def route_after_local(state: TechnicalResearchState) -> str:
    return "extract_claims" if state.get("local_retrieval_hits") else "build_search_queries"


def route_after_verification(state: TechnicalResearchState, runtime: ResearchRuntime) -> str:
    status = state.get("status")
    if status == ResearchStatus.CONFLICTING_EVIDENCE.value:
        return "evaluate_ingestion"
    request = state["request"]
    if status == ResearchStatus.SUPPORTED.value:
        if state.get("local_retrieval_hits") and state.get("search_round", 0) == 0:
            return "evaluate_ingestion"
        authoritative_sources = {
            claim.source_id
            for claim in state.get("extracted_claims", [])
            if claim.source_tier in {
                SourceTier.TIER_1_PRIMARY,
                SourceTier.TIER_2_STRONG_SECONDARY,
            }
            and claim.application_match in {
                ApplicationMatch.EXACT,
                ApplicationMatch.STRONG,
            }
        }
        if len(authoritative_sources) >= 2:
            return "evaluate_ingestion"
    if runtime.search_provider.available and state.get("search_round", 0) < request.limits.max_search_rounds:
        return "refine_queries"
    return "evaluate_ingestion"
