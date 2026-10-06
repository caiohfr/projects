from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from ..contracts import (
    ApplicationMatch,
    ConflictResolution,
    EvidenceClaim,
    EvidenceConflict,
    SourceTier,
)


_TIER_RANK = {
    SourceTier.TIER_1_PRIMARY: 4,
    SourceTier.TIER_2_STRONG_SECONDARY: 3,
    SourceTier.TIER_3_DISCOVERY_ONLY: 2,
    SourceTier.TIER_4_WEAK: 1,
    SourceTier.UNCLASSIFIED: 0,
}
_MATCH_RANK = {
    ApplicationMatch.EXACT: 4,
    ApplicationMatch.STRONG: 3,
    ApplicationMatch.PARTIAL: 2,
    ApplicationMatch.UNKNOWN: 1,
    ApplicationMatch.MISMATCH: 0,
}


def resolve_claim_conflicts(
    claims: Iterable[EvidenceClaim],
) -> tuple[tuple[EvidenceClaim, ...], tuple[EvidenceConflict, ...]]:
    usable = [claim for claim in claims if claim.application_match != ApplicationMatch.MISMATCH]
    by_field: dict[str, list[EvidenceClaim]] = defaultdict(list)
    for claim in usable:
        by_field[claim.field].append(claim)

    selected: list[EvidenceClaim] = []
    conflicts: list[EvidenceConflict] = []
    for field, field_claims in sorted(by_field.items()):
        by_value: dict[str, list[EvidenceClaim]] = defaultdict(list)
        for claim in field_claims:
            by_value[str(claim.normalized_value)].append(claim)
        if len(by_value) == 1:
            selected.extend(field_claims)
            continue

        ranked = sorted(
            field_claims,
            key=lambda item: (
                _MATCH_RANK[item.application_match],
                _TIER_RANK[item.source_tier],
                item.extraction_confidence,
            ),
            reverse=True,
        )
        best = ranked[0]
        tied_best_values = {
            str(item.normalized_value)
            for item in ranked
            if _MATCH_RANK[item.application_match] == _MATCH_RANK[best.application_match]
            and _TIER_RANK[item.source_tier] == _TIER_RANK[best.source_tier]
        }
        if len(tied_best_values) > 1 and _TIER_RANK[best.source_tier] >= 3:
            resolution = ConflictResolution.UNRESOLVED_CONFLICT
        elif any(
            _MATCH_RANK[best.application_match] > _MATCH_RANK[item.application_match]
            for item in ranked[1:]
            if str(item.normalized_value) != str(best.normalized_value)
        ):
            resolution = ConflictResolution.RESOLVED_BY_APPLICATION_MATCH
            selected.extend(by_value[str(best.normalized_value)])
        elif best.source_tier == SourceTier.TIER_1_PRIMARY:
            resolution = ConflictResolution.RESOLVED_BY_PRIMARY_SOURCE
            selected.extend(by_value[str(best.normalized_value)])
        else:
            resolution = ConflictResolution.UNRESOLVED_CONFLICT

        conflicts.append(
            EvidenceConflict(
                field=field,
                values=tuple(item.value for item in ranked),
                supporting_sources=tuple(item.source_id for item in ranked),
                source_tiers=tuple(item.source_tier for item in ranked),
                application_differences=tuple(item.application_match.value for item in ranked),
                likely_explanation=None,
                resolution=resolution,
                resolution_reason=(
                    "TOP_RANKED_APPLICATION_MATCH"
                    if resolution == ConflictResolution.RESOLVED_BY_APPLICATION_MATCH
                    else "PRIMARY_SOURCE_OUTRANKED_CONFLICT"
                    if resolution == ConflictResolution.RESOLVED_BY_PRIMARY_SOURCE
                    else "EQUALLY_STRONG_COMPETING_VALUES"
                ),
            )
        )
    return tuple(selected), tuple(conflicts)
