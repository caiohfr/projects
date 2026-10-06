from __future__ import annotations

import unittest

from capabilities.technical_research.contracts import (
    ApplicationMatch,
    ConflictResolution,
    EvidenceClaim,
    ExtractionMethod,
    ResearchStatus,
    SourceTier,
)
from capabilities.technical_research.core import evaluate_evidence


def _claim(
    value: str,
    source_id: str,
    tier: SourceTier,
    match: ApplicationMatch = ApplicationMatch.DIRECT,
) -> EvidenceClaim:
    return EvidenceClaim(
        field="transmission_designation",
        value=value,
        normalized_value=value.upper(),
        source_id=source_id,
        source_tier=tier,
        evidence_location="page 12, table 3",
        evidence_text=f"Transmission {value}",
        extraction_method=ExtractionMethod.STRUCTURED,
        extraction_confidence=0.99,
        application_match=match,
    )


class EvidencePolicyTests(unittest.TestCase):
    def test_tier_one_beats_weaker_conflict_when_application_match_is_equal(self):
        result = evaluate_evidence(
            [
                _claim("GA8HP50Z", "oem", SourceTier.TIER_1_PRIMARY),
                _claim("GA8HP45Z", "forum", SourceTier.TIER_4_WEAK),
            ],
            "transmission_designation",
        )
        self.assertEqual(result.candidate.identity, "GA8HP50Z")
        self.assertEqual(
            result.bundle.conflicts[0].resolution,
            ConflictResolution.RESOLVED_BY_PRIMARY_SOURCE,
        )

    def test_higher_tier_does_not_override_mismatched_application(self):
        result = evaluate_evidence(
            [
                _claim(
                    "WRONG-PRIMARY",
                    "oem-wrong-year",
                    SourceTier.TIER_1_PRIMARY,
                    ApplicationMatch.MISMATCH,
                ),
                _claim("RIGHT-APPLICATION", "sae", SourceTier.TIER_2_STRONG_SECONDARY),
                _claim("RIGHT-APPLICATION", "catalog", SourceTier.TIER_2_STRONG_SECONDARY),
            ],
            "transmission_designation",
        )
        self.assertEqual(result.candidate.identity, "RIGHT-APPLICATION")
        self.assertEqual(result.status, ResearchStatus.SUPPORTED)

    def test_conflicting_high_quality_sources_remain_conflicting(self):
        result = evaluate_evidence(
            [
                _claim("GA8HP50Z", "oem-a", SourceTier.TIER_1_PRIMARY),
                _claim("GA8HP51Z", "oem-b", SourceTier.TIER_1_PRIMARY),
            ],
            "transmission_designation",
        )
        self.assertEqual(result.status, ResearchStatus.CONFLICTING_EVIDENCE)
        self.assertIsNone(result.candidate.identity)
        self.assertEqual(
            result.bundle.conflicts[0].resolution,
            ConflictResolution.UNRESOLVED_CONFLICT,
        )

    def test_insufficient_evidence_does_not_hallucinate_identity(self):
        result = evaluate_evidence([], "transmission_designation")
        self.assertEqual(result.status, ResearchStatus.INSUFFICIENT_EVIDENCE)
        self.assertIsNone(result.candidate.identity)
        self.assertEqual(result.candidate.attributes, {})

    def test_claim_retains_exact_location_and_excerpt(self):
        claim = _claim("GA8HP50Z", "oem", SourceTier.TIER_1_PRIMARY)
        result = evaluate_evidence([claim], "transmission_designation")
        self.assertEqual(result.bundle.claims[0].evidence_location, "page 12, table 3")
        self.assertEqual(result.bundle.claims[0].evidence_text, "Transmission GA8HP50Z")


if __name__ == "__main__":
    unittest.main()

