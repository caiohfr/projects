from __future__ import annotations

import csv
import json
from pathlib import Path

from capabilities.technical_research.contracts import FetchedDocument, SourceRecord
from capabilities.technical_research.final_micropatch import (
    GOLDEN_TASK_SPECS,
    build_query,
    estimator_unlock,
    resolve_internal,
    run_golden_validation,
    source_relevance,
    source_tier,
)


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs/phasea1_pkg/EcoDrive_Sprint12_Technical_Research_Agent_PhaseA1_Reconciliation_LiveAgent_v1.0/inputs"


def _inputs() -> tuple[list[dict[str, str]], list[dict]]:
    with (INPUTS / "research_child_tasks.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        tasks = list(csv.DictReader(handle))
    with (INPUTS / "research_evidence_staging.jsonl").open("r", encoding="utf-8") as handle:
        evidence = [json.loads(line) for line in handle if line.strip()]
    return tasks, evidence


class FixtureProvider:
    provider_name = "FIXTURE"

    def __init__(self, results: dict[str, list[SourceRecord]]):
        self.results = results
        self.calls = 0

    def search(self, query: str, *, limit: int):
        self.calls += 1
        return tuple(self.results.get(query, ())[:limit])


class FixtureFetcher:
    def __init__(self, documents: dict[str, str]):
        self.documents = documents
        self.calls = 0

    def fetch(self, source: SourceRecord) -> FetchedDocument:
        self.calls += 1
        text = self.documents[source.source_id]
        return FetchedDocument(source=source, content=text, content_hash=f"hash-{source.source_id}")


class ForbiddenProvider:
    provider_name = "FIXTURE"

    def search(self, query: str, *, limit: int):
        raise AssertionError("cache replay attempted network search")


class ForbiddenFetcher:
    def fetch(self, source: SourceRecord):
        raise AssertionError("cache replay attempted network fetch")


def _source(source_id: str, url: str, title: str, snippet: str = "") -> SourceRecord:
    return SourceRecord(
        source_id=source_id, url=url, title=title,
        publisher=url.split("/")[2], metadata={"snippet": snippet},
    )


def _fixture_runtime(tasks: list[dict[str, str]]):
    by_id = {row["child_task_id"]: row for row in tasks}
    hyundai = by_id["RT-C519340F31320050"]
    mazda = by_id["RT-02BB5A52D4C596CB"]
    lexus = by_id["RT-F967A8DCBB29A70C"]
    h_query = build_query(hyundai, "ARCHITECTURE")[1]
    m_query = build_query(mazda, "TRANSMISSION_AXLE_BOUNDARY")[1]
    l_query = build_query(lexus, "TRANSMISSION_IDENTITY")[1]
    bad_h = _source("bad-h", "https://www.schneier.com/news/idg-se/", "News – Schneier on Security")
    good_h = _source(
        "good-h", "https://www.hyundai.com/santa-fe-hybrid-specifications", "Santa Fe Hybrid Technical Specifications",
        "Hyundai Santa Fe Hybrid powertrain transmission 6AT specification",
    )
    bad_m = _source("bad-m", "https://summerplaceal.com/floor-plans/", "Summer Place Apartment Floor Plans")
    good_m = _source(
        "good-m", "https://www.mazda.com/service/1pytgaaa-transaxle", "Mazda3 1PYTGAAA Transaxle Service Manual",
        "Mazda technical final drive differential transmission service manual",
    )
    bad_l = _source("bad-l", "https://example.com/summer", "Summer Vacation Ideas")
    good_l = _source(
        "good-l", "https://www.lexus.com/rx-350/specifications", "Lexus RX 350 Technical Specifications",
        "Lexus RX 350 8-speed automatic transmission specification",
    )
    provider = FixtureProvider({
        h_query: [bad_h, good_h], m_query: [bad_m, good_m], l_query: [bad_l, good_l],
    })
    fetcher = FixtureFetcher({
        "good-h": "Hyundai Santa Fe Hybrid uses a transmission-mounted electric device TMED and Smartstream 6AT.",
        "good-m": "Mazda3 1PYTGAAA transaxle final drive and differential service information.",
        "good-l": "The Lexus RX 350 is equipped with an 8-speed automatic transmission.",
    })
    return provider, fetcher


def test_golden_set_has_required_cases_and_claim_specific_queries() -> None:
    tasks, _ = _inputs()
    by_id = {row["child_task_id"]: row for row in tasks}
    assert len(GOLDEN_TASK_SPECS) == 6
    queries = {
        claim: build_query(by_id[task_id], claim)
        for _, task_id, claim in GOLDEN_TASK_SPECS
    }
    assert queries["ARCHITECTURE"][0] == "ARCHITECTURE_V1"
    assert "powertrain transmission architecture" in queries["ARCHITECTURE"][1]
    assert queries["TRANSMISSION_IDENTITY"][0] == "TRANSMISSION_IDENTITY_V1"
    assert queries["TRANSMISSION_AXLE_BOUNDARY"][0] == "TRANSMISSION_AXLE_BOUNDARY_V1"
    assert "final drive differential service manual" in queries["TRANSMISSION_AXLE_BOUNDARY"][1]


def test_internal_first_resolves_full_scope_and_preserves_partial_scope() -> None:
    tasks, evidence = _inputs()
    by_id = {row["child_task_id"]: row for row in tasks}
    highlander = resolve_internal(by_id["RT-58EFA7A290C2E764"], "ARCHITECTURE", evidence)
    hyundai = resolve_internal(by_id["RT-C519340F31320050"], "ARCHITECTURE", evidence)
    assert highlander["internal_status"] == "INTERNAL_RESOLVED"
    assert hyundai["internal_status"] == "INTERNAL_PARTIAL"
    assert highlander["internal_attempted"] is True


def test_estimator_unlock_gate_defers_non_actionable_and_keeps_supported_routes() -> None:
    tasks, _ = _inputs()
    by_id = {row["child_task_id"]: row for row in tasks}
    assert estimator_unlock(by_id["RT-58F6FAFBF5D647E5"], "ARCHITECTURE")[:2] == (
        "NONE", "DEFER_NO_CURRENT_MODEL_VALUE"
    )
    assert estimator_unlock(by_id["RT-C519340F31320050"], "ARCHITECTURE")[:2] == (
        "HIGH", "RESEARCH_NOW"
    )
    assert estimator_unlock(by_id["RT-F967A8DCBB29A70C"], "TRANSMISSION_IDENTITY")[:2] == (
        "HIGH", "RESEARCH_NOW"
    )


def test_known_bad_results_are_rejected_before_fetch() -> None:
    tasks, _ = _inputs()
    by_id = {row["child_task_id"]: row for row in tasks}
    schneier = _source("s", "https://www.schneier.com/news/idg-se/", "News – Schneier on Security")
    apartment = _source("a", "https://summerplaceal.com/floor-plans/", "Apartment Floor Plans")
    h = source_relevance(schneier, by_id["RT-C519340F31320050"], "ARCHITECTURE")
    m = source_relevance(apartment, by_id["RT-02BB5A52D4C596CB"], "TRANSMISSION_AXLE_BOUNDARY")
    assert h["relevance_status"] == "REJECT"
    assert m["relevance_status"] == "REJECT"
    assert h["rejection_reason"] == "REJECT_NON_AUTOMOTIVE_CONTEXT"
    assert m["rejection_reason"] == "REJECT_NON_AUTOMOTIVE_CONTEXT"


def test_source_tier_is_explainable_and_authority_first() -> None:
    tasks, _ = _inputs()
    task = next(row for row in tasks if row["child_task_id"] == "RT-C519340F31320050")
    official = _source(
        "o", "https://www.hyundai.com/santa-fe-hybrid-specifications", "Santa Fe Hybrid Technical Specifications",
        "Hyundai transmission architecture technical specification",
    )
    relevance = source_relevance(official, task, "ARCHITECTURE")
    assert relevance["relevance_status"] == "ELIGIBLE"
    assert source_tier(official, relevance) == "TIER_1"


def test_golden_run_filters_before_fetch_and_cache_replay_is_network_free(tmp_path: Path) -> None:
    tasks, evidence = _inputs()
    provider, fetcher = _fixture_runtime(tasks)
    first = run_golden_validation(
        tasks=tasks, prior_evidence=evidence, output_dir=tmp_path / "first", cache_dir=tmp_path / "cache",
        cache_only=False, provider=provider, fetcher=fetcher,
    )
    replay = run_golden_validation(
        tasks=tasks, prior_evidence=evidence, output_dir=tmp_path / "replay", cache_dir=tmp_path / "cache",
        cache_only=True, provider=ForbiddenProvider(), fetcher=ForbiddenFetcher(),
    )
    assert provider.calls == 3
    assert fetcher.calls == 3
    assert first["pre_fetch_rejected"] == 3
    assert first["network_source_fetches"] == 3
    assert first["accepted_evidence"] == 3
    assert replay["search_request_count"] == 0
    assert replay["network_source_fetches"] == 0
    assert first["structured_result_sha256"] == replay["structured_result_sha256"]
    staged = (tmp_path / "first/research_agent_evidence_staging.jsonl").read_text(encoding="utf-8")
    assert "resolved_A" not in staged
    assert "coast_A" not in staged


def test_internal_resolved_and_deferred_tasks_never_call_search(tmp_path: Path) -> None:
    tasks, evidence = _inputs()
    provider, fetcher = _fixture_runtime(tasks)
    summary = run_golden_validation(
        tasks=tasks, prior_evidence=evidence, output_dir=tmp_path / "run", cache_dir=tmp_path / "cache",
        cache_only=False, provider=provider, fetcher=fetcher,
    )
    rows = list(csv.DictReader((tmp_path / "run/research_agent_golden_results.csv").open("r", encoding="utf-8-sig")))
    highlander = next(row for row in rows if row["golden_task_id"] == "GOLDEN-TOYOTA-HIGHLANDER")
    avalon = next(row for row in rows if row["golden_task_id"] == "GOLDEN-TOYOTA-AVALON")
    assert highlander["live_search_skipped_reason"] == "INTERNAL_CLAIM_ALREADY_RESOLVED"
    assert highlander["search_result_count"] == "0"
    assert avalon["final_status"] == "DEFER_NO_CURRENT_MODEL_VALUE"
    assert avalon["search_result_count"] == "0"
    assert summary["network_skipped_tasks"] == 3
