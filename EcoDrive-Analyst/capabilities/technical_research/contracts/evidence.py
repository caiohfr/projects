from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from dataclasses import field
from typing import Any, Mapping

from .source import SourceTier


class ApplicationMatch(str, Enum):
    EXACT = "EXACT"
    STRONG = "STRONG"
    PARTIAL = "PARTIAL"
    MISMATCH = "MISMATCH"
    UNKNOWN = "UNKNOWN"
    # v0.1 aliases retained for cached payload/source compatibility.
    DIRECT = "EXACT"
    AMBIGUOUS = "UNKNOWN"


class ExtractionMethod(str, Enum):
    STRUCTURED = "STRUCTURED"
    MODEL_EXTRACTED = "MODEL_EXTRACTED"


class ConflictResolution(str, Enum):
    RESOLVED_BY_PRIMARY_SOURCE = "RESOLVED_BY_PRIMARY_SOURCE"
    RESOLVED_BY_APPLICATION_MATCH = "RESOLVED_BY_APPLICATION_MATCH"
    UNRESOLVED_CONFLICT = "UNRESOLVED_CONFLICT"


@dataclass(frozen=True)
class EvidenceClaim:
    field: str
    value: Any
    normalized_value: Any
    source_id: str
    source_tier: SourceTier
    evidence_location: str
    evidence_text: str
    extraction_method: ExtractionMethod
    extraction_confidence: float
    application_match: ApplicationMatch
    source_url: str = ""
    publisher: str = ""
    document_title: str = ""
    source_classification: str = "UNCLASSIFIED"
    retrieved_at: str = ""
    application_context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.extraction_confidence <= 1.0:
            raise ValueError("extraction_confidence must be between 0 and 1")
        if not self.evidence_location or not self.evidence_text:
            raise ValueError("claim-level evidence location and text are required")


@dataclass(frozen=True)
class EvidenceConflict:
    field: str
    values: tuple[Any, ...]
    supporting_sources: tuple[str, ...]
    source_tiers: tuple[SourceTier, ...]
    application_differences: tuple[str, ...]
    likely_explanation: str | None
    resolution: ConflictResolution
    resolution_reason: str = ""
    conflict_kind: str = "ATTRIBUTE"


@dataclass(frozen=True)
class EvidenceBundle:
    claims: tuple[EvidenceClaim, ...] = ()
    conflicts: tuple[EvidenceConflict, ...] = ()
