from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from ..contracts import (
    AttributeStatus,
    ApplicationMatch,
    ConflictResolution,
    EvidenceBundle,
    EvidenceClaim,
    FieldEvidenceSummary,
    IdentityConfidence,
    ResearchStatus,
    TechnicalCandidate,
    SourceTier,
)
from .confidence import identity_confidence
from .conflict_resolution import resolve_claim_conflicts


@dataclass(frozen=True)
class EvidenceResolution:
    bundle: EvidenceBundle
    candidate: TechnicalCandidate
    status: ResearchStatus


def evaluate_evidence(
    claims: Iterable[EvidenceClaim],
    identity_field: str,
    target_fields: Iterable[str] = (),
) -> EvidenceResolution:
    all_claims = tuple(claims)
    selected, raw_conflicts = resolve_claim_conflicts(all_claims)
    conflicts = tuple(
        replace(
            conflict,
            conflict_kind="IDENTITY" if conflict.field == identity_field else "ATTRIBUTE",
        )
        for conflict in raw_conflicts
    )
    unresolved_fields = {
        conflict.field
        for conflict in conflicts
        if conflict.resolution == ConflictResolution.UNRESOLVED_CONFLICT
    }
    identity_unresolved = identity_field in unresolved_fields
    confidence = identity_confidence(selected, identity_field)
    identity_claims = [claim for claim in selected if claim.field == identity_field]
    identity = identity_claims[0].value if identity_claims and not identity_unresolved else None
    attributes = {
        claim.field: claim.value
        for claim in selected
        if not any(
            conflict.field == claim.field
            and conflict.resolution == ConflictResolution.UNRESOLVED_CONFLICT
            for conflict in conflicts
        )
    }
    all_fields = {claim.field for claim in all_claims}.union(target_fields)
    attribute_status = {
        field: (
            AttributeStatus.CONFLICTING
            if field in unresolved_fields
            else AttributeStatus.SUPPORTED
            if any(
                claim.field == field
                and claim.application_match in {ApplicationMatch.EXACT, ApplicationMatch.STRONG}
                for claim in selected
            )
            else AttributeStatus.PARTIALLY_SUPPORTED
            if any(claim.field == field for claim in selected)
            else AttributeStatus.UNKNOWN
        )
        for field in all_fields
    }
    tier_rank = {
        SourceTier.TIER_1_PRIMARY: 5,
        SourceTier.TIER_2_STRONG_SECONDARY: 4,
        SourceTier.TIER_3_DISCOVERY_ONLY: 3,
        SourceTier.TIER_4_WEAK: 2,
        SourceTier.UNCLASSIFIED: 1,
    }
    match_rank = {
        ApplicationMatch.EXACT: 5,
        ApplicationMatch.STRONG: 4,
        ApplicationMatch.PARTIAL: 3,
        ApplicationMatch.UNKNOWN: 2,
        ApplicationMatch.MISMATCH: 1,
    }
    field_support: dict[str, FieldEvidenceSummary] = {}
    for field in sorted(all_fields):
        field_claims = [claim for claim in selected if claim.field == field]
        field_conflicts = [conflict for conflict in conflicts if conflict.field == field]
        unresolved = any(
            conflict.resolution == ConflictResolution.UNRESOLVED_CONFLICT
            for conflict in field_conflicts
        )
        representative = (
            max(
                field_claims,
                key=lambda claim: (
                    match_rank[claim.application_match],
                    tier_rank[claim.source_tier],
                    claim.extraction_confidence,
                ),
            )
            if field_claims and not unresolved
            else None
        )
        strongest_tier = (
            max(
                (claim.source_tier for claim in field_claims),
                key=tier_rank.__getitem__,
            )
            if field_claims
            else None
        )
        strongest_match = (
            max(
                (claim.application_match for claim in field_claims),
                key=match_rank.__getitem__,
            )
            if field_claims
            else ApplicationMatch.UNKNOWN
        )
        field_support[field] = FieldEvidenceSummary(
            value=representative.value if representative else None,
            normalized_value=representative.normalized_value if representative else None,
            support_status=attribute_status[field],
            source_ids=tuple(sorted({claim.source_id for claim in field_claims})),
            strongest_source_tier=strongest_tier,
            application_match=strongest_match,
            conflict_status=(
                "UNRESOLVED"
                if unresolved
                else "RESOLVED"
                if field_conflicts
                else "NONE"
            ),
        )
    candidate = TechnicalCandidate(
        identity=identity,
        attributes=attributes,
        confidence=IdentityConfidence.UNRESOLVED if identity_unresolved else confidence,
        supporting_claim_ids=tuple(
            f"{claim.source_id}:{claim.field}" for claim in selected
        ),
        attribute_status=attribute_status,
        field_support=field_support,
    )
    if identity_unresolved:
        status = ResearchStatus.CONFLICTING_EVIDENCE
    elif confidence in {IdentityConfidence.DIRECT, IdentityConfidence.STRONG}:
        status = ResearchStatus.SUPPORTED
    elif selected:
        status = ResearchStatus.PARTIALLY_SUPPORTED
    else:
        status = ResearchStatus.INSUFFICIENT_EVIDENCE
    return EvidenceResolution(EvidenceBundle(all_claims, conflicts), candidate, status)
