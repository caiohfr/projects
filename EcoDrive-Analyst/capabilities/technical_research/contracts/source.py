from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


class SourceTier(str, Enum):
    TIER_1_PRIMARY = "TIER_1_PRIMARY"
    TIER_2_STRONG_SECONDARY = "TIER_2_STRONG_SECONDARY"
    TIER_3_DISCOVERY_ONLY = "TIER_3_DISCOVERY_ONLY"
    TIER_4_WEAK = "TIER_4_WEAK"
    UNCLASSIFIED = "UNCLASSIFIED"


class SourceClassificationTier(str, Enum):
    PRIMARY_TECHNICAL = "PRIMARY_TECHNICAL"
    STRONG_TECHNICAL = "STRONG_TECHNICAL"
    DISCOVERY_ONLY = "DISCOVERY_ONLY"
    WEAK = "WEAK"
    UNCLASSIFIED = "UNCLASSIFIED"


@dataclass(frozen=True)
class SourceClassification:
    source_tier: SourceClassificationTier
    policy_tier: SourceTier
    publisher_type: str
    document_type: str
    classification_reason: str


class SourceDecision(str, Enum):
    ACCEPT = "ACCEPT"
    DISCOVERY_ONLY = "DISCOVERY_ONLY"
    REJECT = "REJECT"


class IngestionStatus(str, Enum):
    INGEST = "INGEST"
    METADATA_ONLY = "METADATA_ONLY"
    EPHEMERAL = "EPHEMERAL"
    REJECT = "REJECT"


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    url: str
    title: str
    publisher: str = ""
    document_type: str = "web_page"
    publication_date: str | None = None
    revision: str | None = None
    tier: SourceTier = SourceTier.UNCLASSIFIED
    domain_tags: tuple[str, ...] = ()
    application_tags: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourcePolicyDecision:
    source: SourceRecord
    decision: SourceDecision
    reason: str


@dataclass(frozen=True)
class FetchedDocument:
    source: SourceRecord
    content: str
    content_type: str = "text/plain"
    content_hash: str = ""
    retrieved_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)
