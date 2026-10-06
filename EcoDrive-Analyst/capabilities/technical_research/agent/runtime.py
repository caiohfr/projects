from __future__ import annotations

from dataclasses import dataclass
import os

from ..core import SourcePolicy
from ..knowledge import KnowledgeIngestionService, SQLiteEvidenceStore
from ..profiles import ResearchProfile, TransmissionResearchProfile
from ..tools import (
    ClaimExtractor,
    DDGSSearchProvider,
    DocumentFetcher,
    LangChainStructuredClaimExtractor,
    LocalRetriever,
    MetadataClaimExtractor,
    NullLocalRetriever,
    NullSearchProvider,
    SafeDocumentFetcher,
    SearchProvider,
)


@dataclass
class ResearchRuntime:
    profile: ResearchProfile
    search_provider: SearchProvider
    document_fetcher: DocumentFetcher
    claim_extractor: ClaimExtractor
    local_retriever: LocalRetriever
    source_policy: SourcePolicy
    ingestion_service: KnowledgeIngestionService | None = None
    cache_store: SQLiteEvidenceStore | None = None


def default_runtime() -> ResearchRuntime:
    return ResearchRuntime(
        profile=TransmissionResearchProfile(),
        search_provider=NullSearchProvider(),
        document_fetcher=SafeDocumentFetcher(),
        claim_extractor=MetadataClaimExtractor(),
        local_retriever=NullLocalRetriever(),
        source_policy=SourcePolicy(),
        ingestion_service=None,
        cache_store=None,
    )


def live_runtime(
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
    ingestion_service: KnowledgeIngestionService | None = None,
) -> ResearchRuntime:
    """Construct the opt-in live boundary without persisting API credentials."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not available to the process")
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError("langchain-openai is required for live extraction") from exc
    selected_model = model or os.environ.get("TECHNICAL_RESEARCH_MODEL", "gpt-5.6-terra")
    effort = reasoning_effort or os.environ.get("TECHNICAL_RESEARCH_REASONING_EFFORT", "medium")
    chat_model = ChatOpenAI(
        model=selected_model,
        reasoning_effort=effort,
        use_responses_api=True,
        max_completion_tokens=2_500,
        timeout=90,
        max_retries=2,
    )
    search = DDGSSearchProvider()
    if not search.available:
        raise RuntimeError("ddgs is required for live discovery")
    return ResearchRuntime(
        profile=TransmissionResearchProfile(),
        search_provider=search,
        document_fetcher=SafeDocumentFetcher(timeout_seconds=25.0),
        claim_extractor=LangChainStructuredClaimExtractor(chat_model),
        local_retriever=NullLocalRetriever(),
        source_policy=SourcePolicy(),
        ingestion_service=ingestion_service,
        cache_store=None,
    )
