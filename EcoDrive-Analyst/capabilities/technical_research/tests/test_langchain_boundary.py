from __future__ import annotations

import unittest

from capabilities.technical_research.contracts import FetchedDocument, SourceRecord, SourceTier, TechnicalResearchRequest
from capabilities.technical_research.tools import LangChainStructuredClaimExtractor


class _StructuredRunner:
    def invoke(self, prompt):
        return {
            "claims": [
                {
                    "field": "transmission_designation",
                    "value": "GA8HP50Z",
                    "normalized_value": "GA8HP50Z",
                    "evidence_location": "page 9",
                    "evidence_text": "Transmission GA8HP50Z",
                    "confidence": 0.98,
                    "application_match": "DIRECT",
                }
            ]
        }


class _FakeLangChainChatModel:
    def with_structured_output(self, schema):
        self.schema = schema
        return _StructuredRunner()


class LangChainBoundaryTests(unittest.TestCase):
    def test_structured_output_adapter_preserves_claim_evidence(self):
        source = SourceRecord(
            source_id="oem",
            url="fixture://oem",
            title="OEM manual",
            publisher="BMW",
            tier=SourceTier.TIER_1_PRIMARY,
        )
        document = FetchedDocument(source=source, content="Transmission GA8HP50Z")
        request = TechnicalResearchRequest(
            domain="TRANSMISSION",
            known_fields={"make": "BMW", "model": "330i", "model_year": 2023},
        )
        claims = LangChainStructuredClaimExtractor(_FakeLangChainChatModel()).extract(request, [document])
        self.assertEqual(claims[0].value, "GA8HP50Z")
        self.assertEqual(claims[0].evidence_location, "page 9")
        self.assertEqual(claims[0].source_id, "oem")


if __name__ == "__main__":
    unittest.main()
