from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .evidence import ApplicationMatch
from .source import SourceTier


class IdentityConfidence(str, Enum):
    DIRECT = "DIRECT"
    STRONG = "STRONG"
    WEAK = "WEAK"
    UNRESOLVED = "UNRESOLVED"


class AttributeStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONFLICTING = "CONFLICTING"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class FieldEvidenceSummary:
    value: Any = None
    normalized_value: Any = None
    support_status: AttributeStatus = AttributeStatus.UNKNOWN
    source_ids: tuple[str, ...] = ()
    strongest_source_tier: SourceTier | None = None
    application_match: ApplicationMatch = ApplicationMatch.UNKNOWN
    conflict_status: str = "NONE"


@dataclass(frozen=True)
class TechnicalCandidate:
    identity: str | None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    confidence: IdentityConfidence = IdentityConfidence.UNRESOLVED
    supporting_claim_ids: tuple[str, ...] = ()
    attribute_status: Mapping[str, AttributeStatus] = field(default_factory=dict)
    field_support: Mapping[str, FieldEvidenceSummary] = field(default_factory=dict)
