from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile
import unittest
from unittest.mock import patch

from capabilities.technical_research.contracts import FetchedDocument, SourceRecord
from capabilities.technical_research.cross_oem_v05 import file_sha256
from capabilities.technical_research.golden_retrieval_v052a import (
    architecture_from_raw_wording,
    claim_drop_errors,
    choose_extraction_model,
    distinct_useful_field_count,
    golden_architecture_error_count,
    official_first_order_is_valid,
    official_first_queries,
    select_final_evidence,
)
from capabilities.technical_research.tools.attachment_discovery import (
    discover_technical_attachments,
)
from capabilities.technical_research.tools.content_extract import extract_document_text
from scripts import run_golden_retrieval_benchmark_v052a as runner


ROOT = Path(__file__).resolve().parents[3]
FROZEN_BMW = ROOT / "artifacts/components/component_research_v041a/BMW_COMPONENT_RESEARCH_BENCHMARK_V041A.csv"
FROZEN_BMW_SHA256 = "C4B31774BE2AFB6EB543446FB8CA69CAECCF6E199275BCB143E1B6037919D0E3"


def _ford_sample() -> dict[str, str]:
    _, samples = runner.build_golden_set(runner.read_csv(runner.DEFAULT_SAMPLE))
    return next(row for row in samples if row["golden_case_id"] == "G052-01")


class _SearchProvider:
    def __init__(self, result_batches: list[tuple[SourceRecord, ...]]):
        self.result_batches = result_batches
        self.calls = 0

    def search(self, query: str, *, limit: int = 6) -> tuple[SourceRecord, ...]:
        del query, limit
        index = min(self.calls, len(self.result_batches) - 1)
        self.calls += 1
        return self.result_batches[index]


class _Runtime:
    def __init__(self, result_batches: list[tuple[SourceRecord, ...]]):
        self.search_provider = _SearchProvider(result_batches)


class GoldenRetrievalV052aTests(unittest.TestCase):
    def test_official_queries_are_first_and_secondary_cannot_early_stop(self) -> None:
        sample = _ford_sample()
        queries = official_first_queries(sample, sample["normalized_search_identity"])
        self.assertTrue(official_first_order_is_valid(queries))
        self.assertEqual(
            ["OFFICIAL_TECHNICAL", "OFFICIAL_POWERTRAIN"],
            [item.theme for item in queries[:2]],
        )

        good = SourceRecord(
            source_id="good",
            url="https://www.ford.com/performance/gt/specifications",
            title="2022 Ford GT technical specifications",
            metadata={"search_rank": 1},
        )
        runtime = _Runtime([(good,), ()])

        def extract(_runtime, _request, source, _sample):
            document = FetchedDocument(source=source, content="7-speed DCT; tire 325/30R20")
            return {
                "document": document,
                "accepted": [
                    {"field": "transmission_marketing_description", "value": "7-speed DCT", "application_match": "EXACT", "source_tier": "TIER_1_PRIMARY", "source_url": source.url},
                    {"field": "tire_general", "value": "325/30R20", "application_match": "EXACT", "source_tier": "TIER_1_PRIMARY", "source_url": source.url},
                ],
                "fields": {"transmission_marketing_description", "tire_general"},
                "reason": "", "match": "EXACT", "errors": 0,
                "conflict": False, "rejected_incompatible": 0,
            }

        with patch.object(runner, "_extract_source", side_effect=extract):
            result, query_rows, *_ = runner.run_case(runtime, sample)

        self.assertEqual(2, runtime.search_provider.calls)
        self.assertEqual(2, result["official_attempt_count"])
        self.assertFalse(any(row["theme"].startswith("SECONDARY") for row in query_rows))

    def test_same_query_fetch_failure_uses_next_result_before_new_query(self) -> None:
        sample = _ford_sample()
        failed = SourceRecord(
            source_id="failed", url="https://www.ford.com/gt/technical-failed",
            title="2022 Ford GT technical specifications", metadata={"search_rank": 1},
        )
        good = SourceRecord(
            source_id="good", url="https://www.ford.com/gt/technical-good",
            title="2022 Ford GT technical specifications", metadata={"search_rank": 2},
        )
        runtime = _Runtime([(failed, good), ()])

        def extract(_runtime, _request, source, _sample):
            if source.source_id == "failed":
                return {"document": None, "accepted": [], "fields": set(), "reason": "FETCH_ERROR:Timeout", "match": "NOT_EVALUATED", "errors": 1, "conflict": False, "rejected_incompatible": 0}
            document = FetchedDocument(source=source, content="7-speed DCT; tire 325/30R20")
            accepted = [
                {"field": "transmission_marketing_description", "value": "7-speed DCT", "application_match": "EXACT", "source_tier": "TIER_1_PRIMARY", "source_url": source.url},
                {"field": "tire_general", "value": "325/30R20", "application_match": "EXACT", "source_tier": "TIER_1_PRIMARY", "source_url": source.url},
            ]
            return {"document": document, "accepted": accepted, "fields": {item["field"] for item in accepted}, "reason": "", "match": "EXACT", "errors": 0, "conflict": False, "rejected_incompatible": 0}

        with patch.object(runner, "_extract_source", side_effect=extract):
            _, query_rows, _, _, fallback, _ = runner.run_case(runtime, sample)

        used = [row for row in query_rows if row["fallback_result_used"] == "YES"]
        self.assertTrue(fallback)
        self.assertEqual(1, len(used))
        self.assertEqual("2", str(used[0]["result_rank"]))
        self.assertEqual("YES", used[0]["new_query_avoided"])

    def test_relevant_xlsx_attachment_is_followed_with_lineage(self) -> None:
        source = SourceRecord(
            source_id="kia", url="https://www.kiamedia.com/us/en/models/carnival-hev/2026/specifications",
            title="Carnival HEV Specifications",
        )
        landing = FetchedDocument(
            source=source, content="Specifications", metadata={
                "outbound_links": (
                    {"url": "https://www.kiamedia.com/us/en/download/specifications/xlsx/23310", "title": "Download specifications"},
                    {"url": "https://www.kiamedia.com/us/en/customer-service", "title": "Customer service"},
                )
            },
        )
        attachments = discover_technical_attachments(landing)
        self.assertEqual(1, len(attachments))
        self.assertEqual("xlsx", attachments[0].document_type)
        row = runner._source_row(
            {"golden_case_id": "G052-10", "make": "Kia"}, attachments[0], {"document": FetchedDocument(source=attachments[0], content="6-speed"), "fields": {"gear_ratios"}, "accepted": []},
            parent_url=source.url, reason_followed="DIRECT_RELEVANT_TECHNICAL_ATTACHMENT",
        )
        self.assertEqual(source.url, row["parent_url"])
        self.assertEqual(attachments[0].url, row["child_url"])
        self.assertEqual("xlsx", row["child_type"])

    def test_xlsx_cell_values_are_extracted_without_rewriting_values(self) -> None:
        shared = b'<?xml version="1.0"?><sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><si><t>Final Drive Ratio</t></si></sst>'
        sheet = b'<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1"><v>3.648</v></c></row></sheetData></worksheet>'
        payload = BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("xl/sharedStrings.xml", shared)
            archive.writestr("xl/worksheets/sheet1.xml", sheet)
        text = extract_document_text(
            payload.getvalue(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        self.assertIn("Final Drive Ratio\t3.648", text)

    def test_accepted_physical_final_drive_survives_final_selection(self) -> None:
        evidence = [{
            "field": "physical_final_drive", "value": "3.580",
            "application_match": "EXACT", "source_tier": "TIER_1_PRIMARY",
            "source_url": "https://www.toyota.com/official-spec",
        }]
        selected = select_final_evidence(evidence)
        self.assertEqual("3.580", selected.values["physical_final_drive"])
        self.assertEqual(0, claim_drop_errors(evidence, selected))

    def test_rejected_claim_has_an_explicit_reason(self) -> None:
        source = SourceRecord(source_id="bad", url="https://example.com/wrong", title="Wrong application")
        row = runner._source_row(
            {"golden_case_id": "G052-X", "make": "Example"}, source,
            {"document": FetchedDocument(source=source, content="wrong"), "fields": set(), "accepted": [], "reason": "INCOMPATIBLE_APPLICATION", "match": "MISMATCH"},
        )
        self.assertEqual("REJECTED", row["accepted_rejected"])
        self.assertEqual("INCOMPATIBLE_APPLICATION", row["rejection_reason"])

    def test_architecture_is_deterministic_and_semantic_counts_are_distinct(self) -> None:
        self.assertEqual("DCT", architecture_from_raw_wording("8-speed wet dual-clutch transmission"))
        self.assertEqual("AUTOMATED_MANUAL", architecture_from_raw_wording("6-speed automated manual transmission"))
        self.assertIn(architecture_from_raw_wording("semi-automatic transmission"), {"OTHER", "UNKNOWN"})
        self.assertNotEqual("TORQUE_CONVERTER_AUTOMATIC", architecture_from_raw_wording("semi-automatic transmission"))
        self.assertEqual("MULTI_SPEED_EV", architecture_from_raw_wording("two-speed electric drive", electrification="BEV"))
        values = {
            "transmission_code": "8DCT",
            "transmission_marketing_description": "8-speed wet DCT",
            "transmission_architecture": "DCT",
            "physical_final_drive": "3.58",
        }
        self.assertEqual(3, distinct_useful_field_count(values))
        self.assertEqual(0, golden_architecture_error_count([
            {"golden_case_id": "G052-07", "transmission_architecture": "MULTI_SPEED_EV"},
        ]))
        self.assertEqual(1, golden_architecture_error_count([
            {"golden_case_id": "G052-07", "transmission_architecture": "MANUAL"},
        ]))

    def test_extraction_model_selection_requires_material_sol_gain(self) -> None:
        base = {
            "gpt-5.6-terra": {"field_recall": 0.75, "field_precision": 0.90, "correct_fields": 15},
            "gpt-5.6-sol": {"field_recall": 0.79, "field_precision": 0.92, "correct_fields": 16},
        }
        self.assertEqual("gpt-5.6-terra", choose_extraction_model(base))
        base["gpt-5.6-sol"] = {"field_recall": 0.86, "field_precision": 0.90, "correct_fields": 18}
        self.assertEqual("gpt-5.6-sol", choose_extraction_model(base))

    def test_frozen_bmw_hash_and_database_write_safety(self) -> None:
        self.assertEqual(FROZEN_BMW_SHA256, file_sha256(FROZEN_BMW))
        lowered = (ROOT / "scripts/run_golden_retrieval_benchmark_v052a.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("import sqlite3", lowered)
        for token in ("insert into", "update ", "delete from", "drop table", "alter table", "create table"):
            self.assertNotIn(token, lowered)


if __name__ == "__main__":
    unittest.main()
