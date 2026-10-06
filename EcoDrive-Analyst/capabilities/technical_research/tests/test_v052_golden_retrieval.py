from __future__ import annotations

import csv
from pathlib import Path
import unittest

from capabilities.technical_research.contracts import (
    ApplicationMatch,
    EvidenceClaim,
    ExtractionMethod,
    FetchedDocument,
    SourceRecord,
    SourceTier,
)
from capabilities.technical_research.cross_oem_v05 import file_sha256
from capabilities.technical_research.golden_retrieval_v052 import (
    GOLDEN_CASE_CONFIG,
    adjacent_year_is_compatible,
    coverage_sufficient,
    gap_queries,
    normalize_raw_transmission_description,
    normalize_search_identity,
    primary_queries,
    source_richness,
)
from capabilities.technical_research.tools.attachment_discovery import discover_technical_attachments
from capabilities.technical_research.tools.content_extract import extract_document_text
from scripts.run_golden_retrieval_benchmark_v052 import (
    _gear_ratios_compatible,
    _deterministic_transmission_description,
    _marketing_gears_compatible,
    _match_claim,
    select_source,
)


ROOT = Path(__file__).resolve().parents[3]
SAMPLE = ROOT / "artifacts/components_research/v05_1/CROSS_OEM_SAMPLE_V051.csv"
FROZEN_BMW = ROOT / "artifacts/components/component_research_v041a/BMW_COMPONENT_RESEARCH_BENCHMARK_V041A.csv"
FROZEN_BMW_SHA256 = "C4B31774BE2AFB6EB543446FB8CA69CAECCF6E199275BCB143E1B6037919D0E3"


class GoldenRetrievalV052Tests(unittest.TestCase):
    def test_search_identity_removes_noise_without_mutating_request(self) -> None:
        row = {
            "model_year": "2026",
            "make": "Mercedes-Benz",
            "model": "CLA 250+ with EQ Technology (18'' Wheels)",
        }
        original = dict(row)
        identity = normalize_search_identity(row)
        self.assertEqual("2026 Mercedes CLA 250+", identity)
        self.assertEqual(original, row)
        self.assertIn("18'' Wheels", row["model"])
        self.assertNotIn("Wheels", identity)

    def test_primary_queries_precede_field_specific_gap_queries(self) -> None:
        row = {"make": "KIA"}
        primary = primary_queries(row, "2026 Kia Carnival Hybrid")
        gaps = gap_queries(
            row,
            "2026 Kia Carnival Hybrid",
            {"transmission_architecture", "gear_ratios", "physical_final_drive"},
        )
        self.assertTrue(all(query.round_number == 1 for query in primary))
        self.assertTrue(all(query.round_number == 2 for query in gaps))
        self.assertEqual("VEHICLE_SPECS", primary[0].theme)
        self.assertFalse(any("gear ratios" in query.query for query in gaps))
        self.assertTrue(any("drag coefficient" in query.query for query in gaps))

    def test_completed_fields_do_not_trigger_redundant_gap_search(self) -> None:
        found = {
            "transmission_architecture", "transmission_family", "gear_ratios",
            "physical_final_drive", "reduction_front", "reduction_rear", "cd",
            "frontal_area_m2", "tire_general",
        }
        self.assertEqual((), gap_queries({}, "2026 Example Vehicle", found))

    def test_rich_official_source_outscores_weak_single_field_source(self) -> None:
        rich = source_richness(
            {"transmission_architecture", "gear_ratios", "physical_final_drive", "tire_general"},
            official=True,
            technical_document=True,
        )
        weak = source_richness({"tire_general"}, official=False, technical_document=False)
        self.assertGreater(rich, weak)

    def test_architecture_normalization_order_is_semantically_safe(self) -> None:
        self.assertEqual("DCT", normalize_raw_transmission_description("8-speed wet DCT dual clutch"))
        self.assertEqual("CVT", normalize_raw_transmission_description("continuously variable transmission"))
        self.assertEqual("AUTOMATED_MANUAL", normalize_raw_transmission_description("automated manual"))
        self.assertNotEqual("DCT", normalize_raw_transmission_description("automated manual"))
        self.assertEqual("OTHER", normalize_raw_transmission_description("semi-automatic"))
        self.assertNotEqual("TORQUE_CONVERTER_AUTOMATIC", normalize_raw_transmission_description("semi-automatic"))
        self.assertEqual("MULTI_SPEED_EV", normalize_raw_transmission_description("two-speed electric drive", electrification="BEV"))
        self.assertEqual("SINGLE_SPEED_EV", normalize_raw_transmission_description("fixed ratio EV drive", electrification="BEV"))

    def test_rich_source_coverage_enables_early_stop(self) -> None:
        self.assertTrue(
            coverage_sufficient(
                {"transmission_architecture", "gear_ratios", "tire_general"},
                richest_source_field_count=3,
            )
        )
        self.assertFalse(coverage_sufficient({"transmission_architecture"}, richest_source_field_count=1))

    def test_adjacent_model_year_is_approx_only_when_application_matches(self) -> None:
        self.assertTrue(
            adjacent_year_is_compatible(
                2026,
                {"make": "Toyota", "model": "4Runner", "model_year": 2025},
                requested_make="TOYOTA",
                requested_model="4Runner",
            )
        )
        self.assertFalse(
            adjacent_year_is_compatible(
                2026,
                {"make": "Toyota", "model": "Land Cruiser", "model_year": 2025},
                requested_make="TOYOTA",
                requested_model="4Runner",
            )
        )
        self.assertFalse(
            adjacent_year_is_compatible(
                2026,
                {"make": "Toyota", "model": "4Runner", "model_year": 2023},
                requested_make="TOYOTA",
                requested_model="4Runner",
            )
        )

    def test_golden_set_uses_exact_existing_sample_rows_without_substitution(self) -> None:
        expected = {
            "V05-005", "V05-010", "V05-015", "V05-023", "V05-027",
            "V05-035", "V05-038", "V05-042", "V05-046", "V05-050",
        }
        self.assertEqual(expected, {item["sample_id"] for item in GOLDEN_CASE_CONFIG})
        self.assertEqual(10, len(GOLDEN_CASE_CONFIG))
        with SAMPLE.open("r", encoding="utf-8-sig", newline="") as handle:
            actual = {row["sample_id"] for row in csv.DictReader(handle)}
        self.assertTrue(expected.issubset(actual))

    def test_frozen_bmw_benchmark_hash_is_unchanged(self) -> None:
        self.assertTrue(FROZEN_BMW.exists())
        self.assertEqual(FROZEN_BMW_SHA256, file_sha256(FROZEN_BMW))

    def test_benchmark_runner_contains_no_sqlite_write_or_access(self) -> None:
        runner = (ROOT / "scripts/run_golden_retrieval_benchmark_v052.py").read_text(encoding="utf-8")
        lowered = runner.lower()
        self.assertNotIn("import sqlite3", lowered)
        for token in ("insert into", "update ", "delete from", "drop table", "alter table", "create table"):
            self.assertNotIn(token, lowered)

    def test_customer_service_is_not_followed_but_supported_technical_downloads_are(self) -> None:
        source = SourceRecord(
            source_id="landing",
            url="https://www.kia.com/model/specifications",
            title="Specifications",
        )
        document = FetchedDocument(
            source=source,
            content="specifications",
            metadata={
                "outbound_links": (
                    {"url": "https://www.kia.com/customer-service", "title": "Customer Service"},
                    {"url": "https://www.kia.com/download/specifications.xlsx", "title": "Specification sheet"},
                    {"url": "https://www.kia.com/download/technical-specifications.pdf", "title": "Technical specifications"},
                )
            },
        )
        attachments = discover_technical_attachments(document)
        self.assertEqual(2, len(attachments))
        self.assertEqual({"xlsx", "pdf"}, {item.document_type for item in attachments})

    def test_embedded_oem_specification_data_is_retained_for_extraction(self) -> None:
        padding = b"x" * 600_000
        html = b"""<html><body><h1>Vehicle</h1><script type='application/json'>""" + padding + b"""
        {"specifications":{"transmission":"8-speed DCT","final drive":"3.20"}}
        </script><script>console.log('unrelated')</script></body></html>"""
        text = extract_document_text(html, "text/html")
        self.assertIn("Vehicle", text)
        self.assertIn("8-speed DCT", text)
        self.assertNotIn("unrelated", text)

    def test_plus_variant_does_not_select_non_plus_model_page(self) -> None:
        sample = {"make": "Mercedes-Benz", "model": "CLA 250+", "model_alias": "CLA 250+"}
        wrong = SourceRecord(
            source_id="wrong",
            url="https://www.mbusa.com/en/vehicles/model/cla/coupe/cla250c",
            title="2026 Mercedes CLA 250 Coupe",
        )
        selected, _ = select_source((wrong,), sample, set())
        self.assertIsNone(selected)

    def test_explicit_electrification_mismatch_remains_rejected(self) -> None:
        sample = {
            "sample_id": "test", "make": "Mercedes-Benz", "model": "CLA 250+ with EQ Technology",
            "model_alias": "CLA 250+", "model_year": "2026", "electrification": "BEV",
            "drive_type": "RWD", "transmission_type": "Automatic", "gears": "2",
        }
        claim = EvidenceClaim(
            field="transmission_marketing_description", value="8-speed automatic",
            normalized_value="8 SPEED AUTOMATIC", source_id="source",
            source_tier=SourceTier.TIER_1_PRIMARY, evidence_location="specs",
            evidence_text="8-speed automatic", extraction_method=ExtractionMethod.MODEL_EXTRACTED,
            extraction_confidence=1.0, application_match=ApplicationMatch.UNKNOWN,
            source_url="https://www.mbusa.com/en/vehicles/model/cla/coupe/cla250c",
            document_title="2026 Mercedes CLA 250 Coupe",
            application_context={"make": "Mercedes-Benz", "model": "CLA 250", "model_year": 2026, "electrification": "ICE"},
        )
        matched, _, _ = _match_claim(claim, sample)
        self.assertEqual(ApplicationMatch.MISMATCH, matched.application_match)

    def test_source_model_and_trim_are_reconciled_without_losing_phev_identity(self) -> None:
        sample = {
            "sample_id": "test", "make": "Lincoln", "model": "Aviator PHEV",
            "model_alias": "Aviator Grand Touring", "model_year": "2023", "electrification": "PHEV",
            "drive_type": "AWD", "transmission_type": "Semi-Automatic", "gears": "10",
        }
        claim = EvidenceClaim(
            field="transmission_marketing_description", value="10-speed automatic transmission",
            normalized_value="10 SPEED AUTOMATIC", source_id="source",
            source_tier=SourceTier.TIER_2_STRONG_SECONDARY, evidence_location="powertrain",
            evidence_text="Grand Touring uses the same 10-speed transmission",
            extraction_method=ExtractionMethod.MODEL_EXTRACTED, extraction_confidence=1.0,
            application_match=ApplicationMatch.UNKNOWN,
            source_url="https://www.caranddriver.com/lincoln/aviator-2023",
            document_title="2023 Lincoln Aviator Review, Pricing, and Specs",
            application_context={"make": "Lincoln", "model": "Aviator", "model_year": "2023", "trim/variant": "Grand Touring", "electrification": "plug-in hybrid"},
        )
        matched, _, _ = _match_claim(claim, sample)
        self.assertIn(matched.application_match, {ApplicationMatch.EXACT, ApplicationMatch.STRONG, ApplicationMatch.PARTIAL})

    def test_same_plus_model_adjacent_year_is_retained_as_approximate(self) -> None:
        sample = {
            "sample_id": "test", "make": "Mercedes-Benz", "model": "CLA 250+ with EQ Technology",
            "model_alias": "CLA 250+", "model_year": "2026", "electrification": "BEV",
            "drive_type": "RWD", "transmission_type": "Automatic", "gears": "2",
        }
        claim = EvidenceClaim(
            field="transmission_marketing_description", value="two-speed electric drive transmission",
            normalized_value="TWO SPEED ELECTRIC DRIVE", source_id="source",
            source_tier=SourceTier.TIER_1_PRIMARY, evidence_location="specs",
            evidence_text="two-speed electric drive transmission",
            extraction_method=ExtractionMethod.MODEL_EXTRACTED, extraction_confidence=1.0,
            application_match=ApplicationMatch.UNKNOWN,
            source_url="https://www.mbusa.com/en/vehicles/model/cla/sedan/cla250e",
            document_title="2027 CLA 250+ Sedan | Mercedes-Benz USA",
            application_context={"make": "Mercedes-Benz", "model": "CLA 250+ Sedan", "model_year": 2027, "electrification": "battery electric"},
        )
        matched, _, adjacent = _match_claim(claim, sample)
        self.assertEqual(ApplicationMatch.PARTIAL, matched.application_match)
        self.assertTrue(adjacent)

    def test_incomplete_or_wrong_application_gear_sets_are_rejected(self) -> None:
        self.assertTrue(
            _gear_ratios_compatible(
                "{'1st': '4.1', '2nd': '2.5', '3rd': '1.8', '4th': '1.4', '5th': '1.2', '6th': '1.0'}",
                6,
            )
        )
        self.assertFalse(_gear_ratios_compatible("{'6th': '1.0', '7th': '.8', '8th': '.6'}", 8))
        self.assertFalse(_gear_ratios_compatible("{'1st': '4.8', '2nd': '2.9', '3rd': '1.8', '4th': '1.4', '5th': '1.2', '6th': '1.0', '7th': '.8', '8th': '.6'}", 6))
        self.assertTrue(_marketing_gears_compatible("6-speed automatic transmission", 6))
        self.assertFalse(_marketing_gears_compatible("8-speed automatic transmission", 6))

    def test_deterministic_source_text_architecture_uses_raw_description(self) -> None:
        ford = _deterministic_transmission_description(
            "Power reaches the wheels through a 7-speed dual-clutch automatic transmission.",
            expected_gears=7,
            electrification="ICE",
        )
        self.assertIsNotNone(ford)
        self.assertEqual("DCT", ford[1])
        ev = _deterministic_transmission_description(
            "The electric drive uses a two-speed transmission for acceleration and cruising.",
            expected_gears=2,
            electrification="BEV",
        )
        self.assertIsNotNone(ev)
        self.assertEqual("MULTI_SPEED_EV", ev[1])


if __name__ == "__main__":
    unittest.main()
