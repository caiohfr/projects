"""Isolated, evidence-first technical research capability."""

from .agent import (
    ResearchRuntime,
    live_runtime,
    research_batch,
    research_batch_with_summary,
    research_technical_component,
    summarize_research_batch,
)
from .contracts import (
    TechnicalResearchBatchResult,
    TechnicalResearchBatchSummary,
    TechnicalResearchRequest,
    TechnicalResearchResult,
)
from .core import audit_research_request, evaluate_hardware_group

__all__ = [
    "ResearchRuntime",
    "audit_research_request",
    "evaluate_hardware_group",
    "live_runtime",
    "TechnicalResearchRequest",
    "TechnicalResearchResult",
    "TechnicalResearchBatchResult",
    "TechnicalResearchBatchSummary",
    "research_batch",
    "research_batch_with_summary",
    "research_technical_component",
    "summarize_research_batch",
]
