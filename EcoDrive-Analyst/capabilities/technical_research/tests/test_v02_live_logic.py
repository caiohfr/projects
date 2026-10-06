from __future__ import annotations

import unittest

from capabilities.technical_research.contracts import (
    ApplicationMatch,
    AttributeStatus,
    EvidenceClaim,
    ExtractionMethod,
    FetchedDocument,
    SourceClassificationTier,
    SourceRecord,
    SourceTier,
    TechnicalResearchRequest,
)
from capabilities.technical_research.core import (
    classify_source,
    evaluate_evidence,
    match_application,
    source_with_classification,
)
from capabilities.technical_research.tools import LangChainStructuredClaimExtractor


class SourceClassificationTests(unittest.TestCase):
    def _classify(self, url: str, title: str = "Technical specifications"):
        return classify_source(SourceRecord("s", url, title))

    def test_authority_families_are_deterministic(self):
        cases = (
            ("https://www.press.bmwgroup.com/a/specifications.pdf", SourceClassificationTier.PRIMARY_TECHNICAL),
            ("https://www.zf.com/products/transmission-datasheet.pdf", SourceClassificationTier.STRONG_TECHNICAL),
            ("https://www.nhtsa.gov/database/vehicle", SourceClassificationTier.PRIMARY_TECHNICAL),
            ("https://www.sae.org/publications/technical-paper", SourceClassificationTier.STRONG_TECHNICAL),
            ("https://www.realoem.com/bmw/catalog", SourceClassificationTier.DISCOVERY_ONLY),
            ("https://www.bimmerpost.com/forums/thread", SourceClassificationTier.WEAK),
            ("https://unknown.example/page", SourceClassificationTier.UNCLASSIFIED),
        )
        for url, expected in cases:
            with self.subTest(url=url):
                self.assertEqual(self._classify(url).source_tier, expected)

    def test_oem_marketing_page_is_not_blindly_primary(self):
        result = self._classify("https://www.bmwgroup.com/en/car.html", "The new BMW")
        self.assertEqual(result.source_tier, SourceClassificationTier.DISCOVERY_ONLY)

    def test_provider_tier_is_replaced_by_classifier(self):
        source = SourceRecord(
            "s", "https://unknown.example/page", "Unknown", tier=SourceTier.TIER_1_PRIMARY
        )
        self.assertEqual(source_with_classification(source).tier, SourceTier.UNCLASSIFIED)


class ApplicationMatchingTests(unittest.TestCase):
    request = {
        "make": "BMW",
        "model": "330i",
        "model_year": 2021,
        "drive_type": "RWD",
        "engine": "B48",
        "trim": "Sport",
    }

    def test_exact(self):
        self.assertEqual(match_application(self.request, self.request).match, ApplicationMatch.EXACT)

    def test_missing_noncritical_is_strong(self):
        context = {key: value for key, value in self.request.items() if key != "trim"}
        self.assertEqual(match_application(self.request, context).match, ApplicationMatch.STRONG)

    def test_different_drive_is_mismatch(self):
        context = {**self.request, "drive_type": "xDrive"}
        self.assertEqual(match_application(self.request, context).match, ApplicationMatch.MISMATCH)

    def test_different_engine_is_mismatch(self):
        context = {**self.request, "engine": "B58"}
        self.assertEqual(match_application(self.request, context).match, ApplicationMatch.MISMATCH)

    def test_broad_model_family_is_partial(self):
        context = {"make": "BMW", "model": "330i"}
        self.assertEqual(match_application(self.request, context).match, ApplicationMatch.PARTIAL)

    def test_series_document_is_partial_for_specific_model(self):
        context = {"make": "BMW", "model": "3 Series", "model_year": 2021}
        self.assertEqual(match_application(self.request, context).match, ApplicationMatch.PARTIAL)

    def test_unknown_year_is_partial_not_exact(self):
        context = {"make": "BMW", "model": "330i", "drive_type": "RWD", "engine": "B48"}
        self.assertEqual(match_application(self.request, context).match, ApplicationMatch.PARTIAL)


def _claim(field: str, value: str, source: str) -> EvidenceClaim:
    return EvidenceClaim(
        field=field,
        value=value,
        normalized_value=value.upper(),
        source_id=source,
        source_tier=SourceTier.TIER_1_PRIMARY,
        evidence_location="page 1",
        evidence_text=f"{field}: {value}",
        extraction_method=ExtractionMethod.STRUCTURED,
        extraction_confidence=0.99,
        application_match=ApplicationMatch.EXACT,
    )


class FieldConflictTests(unittest.TestCase):
    def test_secondary_conflict_does_not_erase_identity(self):
        result = evaluate_evidence(
            [
                _claim("transmission_designation", "GA8HP50Z", "oem"),
                _claim("tire_size", "225/45R18", "oem-a"),
                _claim("tire_size", "255/40R18", "oem-b"),
            ],
            "transmission_designation",
        )
        self.assertEqual(result.candidate.identity, "GA8HP50Z")
        self.assertEqual(result.candidate.attribute_status["tire_size"], AttributeStatus.CONFLICTING)
        self.assertNotIn("tire_size", result.candidate.attributes)


class _Runner:
    def invoke(self, prompt):
        return {
            "source_application_context": {"make": "BMW", "model": "330i", "model_year": 2021},
            "claims": [
                {
                    "field": "transmission_designation",
                    "value": "GA8HP50Z",
                    "evidence_location": "page 2",
                    "evidence_text": "GA8HP50Z",
                    "confidence": 0.9,
                    "application_match": "EXACT",
                }
            ],
        }


class _Model:
    def with_structured_output(self, schema):
        return _Runner()


class ExtractionAuthorityTests(unittest.TestCase):
    def test_model_application_match_is_ignored(self):
        source = SourceRecord(
            "s",
            "https://unknown.example/page",
            "Page",
            tier=SourceTier.UNCLASSIFIED,
            metadata={"source_classification": "UNCLASSIFIED"},
        )
        request = TechnicalResearchRequest(
            domain="TRANSMISSION",
            known_fields={"make": "BMW", "model": "330i", "model_year": 2021},
        )
        claim = LangChainStructuredClaimExtractor(_Model()).extract(
            request, [FetchedDocument(source=source, content="GA8HP50Z")]
        )[0]
        self.assertEqual(claim.application_match, ApplicationMatch.UNKNOWN)
        self.assertEqual(claim.source_tier, SourceTier.UNCLASSIFIED)


if __name__ == "__main__":
    unittest.main()
