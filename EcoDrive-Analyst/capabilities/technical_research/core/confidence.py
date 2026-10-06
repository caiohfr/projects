from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from ..contracts import ApplicationMatch, EvidenceClaim, IdentityConfidence, SourceTier


def identity_confidence(
    claims: Iterable[EvidenceClaim], identity_field: str
) -> IdentityConfidence:
    identity_claims = [
        claim
        for claim in claims
        if claim.field == identity_field and claim.application_match != ApplicationMatch.MISMATCH
    ]
    if not identity_claims:
        return IdentityConfidence.UNRESOLVED
    by_value: dict[str, list[EvidenceClaim]] = defaultdict(list)
    for claim in identity_claims:
        by_value[str(claim.normalized_value)].append(claim)
    if len(by_value) != 1:
        return IdentityConfidence.UNRESOLVED
    supporting = next(iter(by_value.values()))
    if any(
        claim.source_tier == SourceTier.TIER_1_PRIMARY
        and claim.application_match == ApplicationMatch.EXACT
        for claim in supporting
    ):
        return IdentityConfidence.DIRECT
    high_quality_sources = {
        claim.source_id
        for claim in supporting
        if claim.source_tier in {SourceTier.TIER_1_PRIMARY, SourceTier.TIER_2_STRONG_SECONDARY}
        and claim.application_match in {
            ApplicationMatch.EXACT,
            ApplicationMatch.STRONG,
            ApplicationMatch.PARTIAL,
        }
    }
    if len(high_quality_sources) >= 2:
        return IdentityConfidence.STRONG
    return IdentityConfidence.WEAK
