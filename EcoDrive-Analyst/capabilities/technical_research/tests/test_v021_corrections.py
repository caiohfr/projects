from __future__ import annotations

import hashlib
import sys
import types
import unittest
from unittest.mock import patch

from capabilities.technical_research import research_technical_component
from capabilities.technical_research.agent import ResearchRuntime
from capabilities.technical_research.agent.nodes import _source_priority
from capabilities.technical_research.contracts import (
    ApplicationMatch,
    FetchedDocument,
    ResearchLimits,
    SourceRecord,
    TechnicalResearchRequest,
)
from capabilities.technical_research.core import SourcePolicy, match_application
from capabilities.technical_research.core import source_with_classification
from capabilities.technical_research.profiles import (
    TRANSMISSION_TARGET_FIELDS,
    TransmissionResearchProfile,
)
from capabilities.technical_research.tools import (
    FixtureSearchProvider,
    DDGSSearchProvider,
    LangChainStructuredClaimExtractor,
    MetadataClaimExtractor,
    NullLocalRetriever,
)


class _AttachmentFetcher:
    def __init__(self, landing_id: str, attachment_url: str):
        self.landing_id = landing_id
        self.attachment_url = attachment_url
        self.fetched_ids: list[str] = []

    def fetch(self, source: SourceRecord) -> FetchedDocument:
        self.fetched_ids.append(source.source_id)
        if source.source_id == self.landing_id:
            return FetchedDocument(
                source=source,
                content="BMW 330i technical data landing page",
                content_type="text/html",
                metadata={
                    "outbound_links": [
                        {
                            "url": self.attachment_url,
                            "title": "Technical specifications PDF",
                            "application_tags": {
                                "make": "BMW",
                                "model": "330i",
                                "model_year": 2023,
                                "drive_type": "RWD",
                                "transmission_type": "Steptronic automatic transmission",
                            },
                            "metadata": {
                                "claims": [
                                    {
                                        "field": "transmission_designation",
                                        "value": "GA8HP50Z",
                                        "evidence_location": "page 4 table 2",
                                        "evidence_text": "Transmission GA8HP50Z",
                                    },
                                    {
                                        "field": "gear_ratios",
                                        "value": "5.000;3.200;2.143;1.720;1.314;1.000;0.822;0.640",
                                        "evidence_location": "page 4 table 2",
                                        "evidence_text": "Gear ratios 5.000 ... 0.640",
                                    },
                                    {
                                        "field": "final_drive_ratio",
                                        "value": "2.813",
                                        "evidence_location": "page 4 table 2",
                                        "evidence_text": "Final drive 2.813",
                                    },
                                ]
                            },
                        }
                    ]
                },
            )
        return FetchedDocument(source=source, content="application-specific technical table")


class _StructuredRunner:
    def __init__(self, payload):
        self.payload = payload

    def invoke(self, prompt):
        return self.payload


class _StructuredModel:
    def __init__(self, payload):
        self.payload = payload

    def with_structured_output(self, schema, **kwargs):
        return _StructuredRunner(self.payload)


class ApplicationNormalizationRegressionTests(unittest.TestCase):
    def test_epa_semi_automatic_and_oem_steptronic_are_same_physical_family(self):
        expected = {
            "make": "BMW",
            "model": "330i",
            "transmission_type": "Semi-Automatic",
        }
        observed = {
            "make": "BMW",
            "model": "330i",
            "transmission_type": "8-speed Steptronic automatic transmission",
        }
        self.assertNotEqual(match_application(expected, observed).match, ApplicationMatch.MISMATCH)

    def test_i4_wheel_suffix_retains_model_family_match(self):
        result = match_application(
            {"make": "BMW", "model": "i4"},
            {"make": "BMW", "model": "i4 xDrive40 Gran Coupe (18'' Wheels)"},
        )
        self.assertEqual(result.match, ApplicationMatch.PARTIAL)
        self.assertNotIn("model", result.mismatches)

    def test_rwd_and_model_embedded_xdrive_remain_mismatch(self):
        result = match_application(
            {"make": "BMW", "model": "i4", "drive_type": "RWD"},
            {"make": "BMW", "model": "i4 xDrive40 Gran Coupe"},
        )
        self.assertEqual(result.match, ApplicationMatch.MISMATCH)
        self.assertIn("drive_type", result.mismatches)

    def test_distinct_powertrain_designations_remain_mismatch(self):
        result = match_application(
            {"make": "BMW", "model": "330i"},
            {"make": "BMW", "model": "330e"},
        )
        self.assertEqual(result.match, ApplicationMatch.MISMATCH)

    def test_bev_automatic_with_one_gear_matches_single_speed_source(self):
        result = match_application(
            {
                "make": "BMW",
                "model": "i4 xDrive40",
                "transmission_type": "Automatic",
                "gears": "1",
            },
            {
                "make": "BMW",
                "model": "i4 xDrive40 Gran Coupe",
                "transmission_type": "Automatic transmission, single-speed with fixed ratio",
                "gears": "single-speed",
            },
        )
        self.assertNotEqual(result.match, ApplicationMatch.MISMATCH)

    def test_edrive_and_xdrive_variants_remain_distinct(self):
        result = match_application(
            {"make": "BMW", "model": "i4 eDrive 35 Gran Coupe"},
            {"make": "BMW", "model": "i4 xDrive40 Gran Coupe"},
        )
        self.assertEqual(result.match, ApplicationMatch.MISMATCH)


class RetrievalCorrectionRegressionTests(unittest.TestCase):
    def test_empty_structured_response_is_isolated_to_its_document(self):
        extractor = LangChainStructuredClaimExtractor(_StructuredModel(None))
        request = TechnicalResearchRequest(
            domain="TRANSMISSION",
            known_fields={"make": "BMW", "model": "330i"},
        )
        document = FetchedDocument(
            source=SourceRecord("empty", "https://www.press.bmwgroup.com/test", "Technical data"),
            content="technical source content",
        )
        self.assertEqual(extractor.extract(request, [document]), ())
        self.assertEqual(extractor.audit[0]["status"], "ERROR")
        self.assertIn("EMPTY_STRUCTURED_RESPONSE", extractor.audit[0]["error"])

    def test_claim_without_location_or_evidence_text_is_skipped(self):
        extractor = LangChainStructuredClaimExtractor(
            _StructuredModel(
                {
                    "source_application_context": {"make": "BMW", "model": "330i"},
                    "claims": [
                        {
                            "field": "transmission_designation",
                            "value": "GA8HP50Z",
                            "evidence_location": "",
                            "evidence_text": "",
                            "confidence": 0.9,
                        }
                    ],
                }
            )
        )
        request = TechnicalResearchRequest(
            domain="TRANSMISSION",
            known_fields={"make": "BMW", "model": "330i"},
        )
        document = FetchedDocument(
            source=SourceRecord("blank", "https://www.press.bmwgroup.com/test", "Technical data"),
            content="technical source content",
        )
        self.assertEqual(extractor.extract(request, [document]), ())
        self.assertEqual(extractor.audit[0]["invalid_claims"], 1)

    def test_failed_live_search_is_audited(self):
        class _FailingDDGS:
            def __init__(self, **kwargs):
                pass

            def text(self, query, max_results):
                raise TimeoutError("fixture timeout")

        provider = DDGSSearchProvider()
        with patch.dict(sys.modules, {"ddgs": types.SimpleNamespace(DDGS=_FailingDDGS)}):
            with self.assertRaises(TimeoutError):
                provider.search("fixture query", limit=5)
        self.assertEqual(provider.audit[0]["status"], "ERROR")
        self.assertEqual(provider.audit[0]["error"], "TimeoutError")

    def test_technical_pdf_precedes_article_landing_within_same_tier(self):
        landing = source_with_classification(
            SourceRecord(
                "landing",
                "https://www.press.bmwgroup.com/global/article/detail/T1/technical-data",
                "Technical data article",
            )
        )
        attachment = source_with_classification(
            SourceRecord(
                "attachment",
                "https://www.press.bmwgroup.com/global/article/attachment/T1/data.pdf",
                "Technical data PDF",
                document_type="pdf",
                metadata={"is_technical_attachment": True},
            )
        )
        self.assertLess(_source_priority(attachment), _source_priority(landing))

    def test_round_two_places_bmwtechinfo_inside_five_query_budget(self):
        request = TechnicalResearchRequest(
            domain="TRANSMISSION",
            known_fields={"make": "BMW", "model": "330i", "model_year": 2023},
        )
        queries = TransmissionResearchProfile().build_queries(request, round_number=2)
        self.assertLessEqual(len(queries), 5)
        self.assertTrue(any("bmwtechinfo.bmwgroup.com" in query for query in queries[:5]))

    def test_article_discovers_and_fetches_technical_attachment_with_table_claims(self):
        landing = SourceRecord(
            source_id="landing",
            url="https://www.press.bmwgroup.com/global/article/detail/T0001/technical-data",
            title="BMW 330i technical data",
        )
        attachment_url = "https://www.press.bmwgroup.com/global/article/attachment/T0001/specifications.pdf"
        attachment_id = "src-" + hashlib.sha256(attachment_url.encode("utf-8")).hexdigest()[:16]
        fetcher = _AttachmentFetcher(landing.source_id, attachment_url)
        runtime = ResearchRuntime(
            profile=TransmissionResearchProfile(),
            search_provider=FixtureSearchProvider({"site:press.bmwgroup.com": (landing,)}),
            document_fetcher=fetcher,
            claim_extractor=MetadataClaimExtractor(),
            local_retriever=NullLocalRetriever(),
            source_policy=SourcePolicy(),
        )
        request = TechnicalResearchRequest(
            domain="TRANSMISSION",
            known_fields={
                "make": "BMW",
                "model": "330i",
                "model_year": 2023,
                "drive_type": "RWD",
                "transmission_type": "Semi-Automatic",
            },
            target_fields=TRANSMISSION_TARGET_FIELDS,
            limits=ResearchLimits(
                max_search_rounds=1,
                max_search_queries_per_round=1,
                max_sources_fetched=4,
                max_high_quality_sources_used=4,
            ),
        )
        result = research_technical_component(request, runtime=runtime)

        self.assertIn(attachment_id, fetcher.fetched_ids)
        self.assertEqual(result.candidate.identity, "GA8HP50Z")
        self.assertEqual(result.candidate.attributes["final_drive_ratio"], "2.813")
        self.assertIn("5.000", result.candidate.attributes["gear_ratios"])
        attachment_source = next(
            row for row in result.source_summary["sources"] if row["source_id"] == attachment_id
        )
        self.assertEqual(attachment_source["landing_source_id"], landing.source_id)
        self.assertTrue(attachment_source["fetched"])

    def test_transmission_contract_uses_explicit_component_supplier_fields(self):
        self.assertIn("transmission_manufacturer", TRANSMISSION_TARGET_FIELDS)
        self.assertIn("transmission_supplier", TRANSMISSION_TARGET_FIELDS)
        self.assertNotIn("manufacturer", TRANSMISSION_TARGET_FIELDS)
        self.assertNotIn("supplier", TRANSMISSION_TARGET_FIELDS)


if __name__ == "__main__":
    unittest.main()
