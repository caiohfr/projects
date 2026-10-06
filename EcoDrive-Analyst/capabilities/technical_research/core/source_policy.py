from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlsplit

from ..contracts import (
    SourceDecision,
    SourcePolicyDecision,
    SourceRecord,
    SourceTier,
)


@dataclass(frozen=True)
class SourcePolicy:
    blocked_hosts: frozenset[str] = field(
        default_factory=lambda: frozenset({"localhost", "127.0.0.1", "::1"})
    )

    def evaluate(self, source: SourceRecord) -> SourcePolicyDecision:
        parsed = urlsplit(source.url)
        host = (parsed.hostname or "").lower()
        if parsed.scheme not in {"http", "https", "fixture", "evidence"}:
            return SourcePolicyDecision(source, SourceDecision.REJECT, "UNSAFE_SCHEME")
        if not host and parsed.scheme in {"http", "https"}:
            return SourcePolicyDecision(source, SourceDecision.REJECT, "MISSING_HOST")
        if host in self.blocked_hosts:
            return SourcePolicyDecision(source, SourceDecision.REJECT, "BLOCKED_HOST")
        if not source.title.strip():
            return SourcePolicyDecision(source, SourceDecision.REJECT, "MISSING_TITLE")
        if source.tier in {SourceTier.TIER_1_PRIMARY, SourceTier.TIER_2_STRONG_SECONDARY}:
            return SourcePolicyDecision(source, SourceDecision.ACCEPT, "HIGH_QUALITY_SOURCE")
        if source.tier in {SourceTier.TIER_3_DISCOVERY_ONLY, SourceTier.TIER_4_WEAK}:
            return SourcePolicyDecision(source, SourceDecision.DISCOVERY_ONLY, "DISCOVERY_ONLY_TIER")
        return SourcePolicyDecision(source, SourceDecision.REJECT, "UNCLASSIFIED_SOURCE")

