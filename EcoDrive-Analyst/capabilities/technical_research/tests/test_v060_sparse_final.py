from __future__ import annotations

from pathlib import Path
import tempfile

from capabilities.technical_research.cross_oem_v05 import file_sha256
from capabilities.technical_research.golden_retrieval_v052 import OEM_SOURCE_HINTS
from scripts.run_component_research_sparse_v060 import (
    DEFAULT_BASELINE,
    FROZEN_BMW,
    FROZEN_BMW_SHA256,
    materially_enriched,
    read_csv,
    resolve_evidence_precedence,
    run,
)


ROOT = Path(__file__).resolve().parents[3]


def test_actual_v051_population_targets_only_36_sparse_rows() -> None:
    rows = read_csv(DEFAULT_BASELINE)
    targeted = [row for row in rows if row["review_flag"] == "REVIEW_SPARSE"]
    assert len(rows) == 50
    assert len(targeted) == 36
    assert not any(row["make"].upper() == "BMW" for row in targeted)


def test_exact_evidence_is_not_displaced_by_weaker_approximate_claim() -> None:
    rows = resolve_evidence_precedence([
        {"sample_id": "A", "field": "physical_final_drive", "value": "3.58", "application_match": "EXACT", "source_url": "official"},
        {"sample_id": "A", "field": "physical_final_drive", "value": "3.73", "application_match": "PARTIAL", "source_url": "secondary"},
    ])
    assert len(rows) == 1
    assert rows[0]["value"] == "3.58"


def test_material_enrichment_requires_added_or_stronger_value() -> None:
    assert materially_enriched(
        {"cd": "", "cd_provenance": "UNKNOWN"},
        {"cd": "0.28", "cd_provenance": "RESEARCHED_EXACT"},
    )
    assert not materially_enriched(
        {"cd": "0.28", "cd_provenance": "RESEARCHED_EXACT"},
        {"cd": "0.28", "cd_provenance": "RESEARCHED_APPROX"},
    )


def test_supported_gm_oems_have_official_domain_hints() -> None:
    for make in ("BUICK", "CHEVROLET", "GMC", "CADILLAC"):
        assert OEM_SOURCE_HINTS[make]


def test_dry_run_preserves_50_rows_and_reports_incomplete() -> None:
    with tempfile.TemporaryDirectory() as directory:
        report = run(output=Path(directory), live=False, resume=False)
        rows = read_csv(Path(directory) / "COMPONENT_RESEARCH_CROSS_OEM_FINAL_V060.csv")
    assert len(rows) == 50
    assert report["targeted"] == 36
    assert report["rerun"] == 0
    assert not report["complete"]
    assert not report["golden_source_injection_used"]


def test_runner_explicitly_disables_golden_source_injection_and_sql_writes() -> None:
    source = (ROOT / "scripts/run_component_research_sparse_v060.py").read_text(encoding="utf-8")
    assert "known_sources={}, evidence_sink=accepted" in source
    lowered = source.lower()
    assert "import sqlite3" not in lowered
    for token in ("insert into", "update ", "delete from", "drop table", "alter table", "create table"):
        assert token not in lowered


def test_bmw_frozen_benchmark_is_unchanged() -> None:
    assert file_sha256(FROZEN_BMW) == FROZEN_BMW_SHA256
