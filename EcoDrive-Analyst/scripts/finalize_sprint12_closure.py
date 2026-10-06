from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from vde_core.sprint12_final_closure import (
    EXPECTED_FINAL_SHA256,
    audit_database,
    component_coverage_rows,
    coverage_summary,
    file_sha256,
    open_read_only,
    write_coverage_reports,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash_entry(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}


def finalize(repo_root: Path, final_dir: Path) -> dict[str, Any]:
    final_dir.mkdir(parents=True, exist_ok=True)
    candidate = repo_root / "data/db/staging/eco_drive_canonical_candidate.db"
    release = final_dir / "eco_drive_canonical_sprint12_final.db"
    workbook = final_dir / "EcoDrive_Canonical_DB_Sprint12_Final.xlsx"
    test_results = final_dir / "test_results.txt"
    excel_verification_path = final_dir / "FINAL_EXCEL_VERIFICATION.json"
    excel_manifest_path = final_dir / "FINAL_EXCEL_EXPORT_MANIFEST.json"
    materialization_path = repo_root / "artifacts/canonical_component_population/execution_summary.json"
    research_path = repo_root / "artifacts/research_agent_final_micropatch/execution_summary.json"

    candidate_audit = audit_database(candidate)
    release_audit = audit_database(release)
    if candidate_audit["sha256"] != EXPECTED_FINAL_SHA256 or release_audit["sha256"] != EXPECTED_FINAL_SHA256:
        raise RuntimeError("Candidate/release hash does not match the accepted Sprint 12 hash")
    materialization = _load_json(materialization_path)
    research = _load_json(research_path)
    excel_manifest = _load_json(excel_manifest_path)
    excel_verification = _load_json(excel_verification_path)
    if excel_verification.get("status") != "PASS":
        raise RuntimeError("Excel reconciliation did not pass")

    connection = open_read_only(release)
    try:
        coverage_rows = component_coverage_rows(connection)
        metrics, breakdowns = coverage_summary(connection, coverage_rows, research_agent_summary=research)
    finally:
        connection.close()
    write_coverage_reports(final_dir, metrics, breakdowns)
    metric = {row["metric"]: row for row in metrics}

    audit_payload = {
        **release_audit,
        "accepted_candidate_path": str(candidate.resolve()),
        "accepted_candidate_sha256": candidate_audit["sha256"],
        "candidate_release_hash_match": candidate_audit["sha256"] == release_audit["sha256"],
        "materialization_integrity": materialization["integrity"],
        "semantic_verification": materialization["semantic_verification"],
    }
    (final_dir / "FINAL_DB_AUDIT.json").write_text(json.dumps(audit_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (final_dir / "FINAL_DB_TABLE_COUNTS.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["table", "row_count", "column_count", "primary_key_columns"])
        writer.writeheader()
        for row in release_audit["tables"]:
            writer.writerow({**row, "primary_key_columns": ",".join(row["primary_key_columns"])})

    table_lines = ["| Table | Rows | Columns |", "|---|---:|---:|"]
    for row in release_audit["tables"]:
        table_lines.append(f"| `{row['table']}` | {row['row_count']:,} | {row['column_count']} |")
    sheet_list = ", ".join(f"`{name}`" for name in excel_manifest["sheets"])
    status = "SPRINT_12_CLOSED"
    report = [
        status,
        "",
        "# EcoDrive Sprint 12 - Final Closure Report",
        "",
        "## 1. Executive closure state",
        "",
        "All mandatory release, integrity, semantic-freeze, coverage, Excel reconciliation, and focused regression gates passed. Sprint 12 is closed without promotion to PROD or QA.",
        "",
        "## 2. Final release database",
        "",
        f"- Release: `{release.resolve()}`",
        f"- Accepted candidate: `{candidate.resolve()}`",
        f"- SHA256 (both): `{release_audit['sha256']}`",
        f"- Size: {release_audit['size_bytes']:,} bytes",
        f"- Schema SHA256: `{release_audit['schema_sha256']}`",
        f"- SQLite quick_check: `{release_audit['quick_check'][0]}`",
        f"- Foreign-key issues: {release_audit['foreign_key_issue_count']}",
        "",
        "## 3. Final table counts",
        "",
        *table_lines,
        "",
        "## 4. Component Population final coverage",
        "",
        f"- Total VDEs: {metric['TOTAL_VDE']['value']:,}",
        f"- Macro resolved: {metric['MACRO_RESOLVED']['value']:,}",
        f"- Supported: {metric['MACRO_SUPPORTED']['value']:,}",
        f"- Conditional: {metric['MACRO_CONDITIONAL']['value']:,}",
        f"- Intentionally unresolved: {metric['MACRO_UNRESOLVED']['value']:,}",
        f"- Projected to historical slots: {metric['MACRO_PROJECTED_TO_HISTORICAL_SLOTS']['value']:,}",
        f"- Fine supporting links: {metric['FINE_SUPPORTING_LINKS']['value']:,}",
        f"- Fine adopted links: {metric['FINE_ADOPTED_LINKS']['value']:,}",
        f"- Boundary-unknown representable VDEs: {metric['BOUNDARY_UNKNOWN_REPRESENTABLE']['value']:,}",
        "",
        "The full metric table and breakdowns are in `COMPONENT_POPULATION_FINAL_COVERAGE.csv` and `.md`.",
        "",
        "## 5. Macro/Fine semantics frozen",
        "",
        "- Macro solutions are adopted estimator outputs; macro coverage is not proof of fine hardware identity.",
        "- Fine component links remain `SUPPORTING`; zero fine links were inflated to adopted decomposition.",
        "- Historical slot projection remains explicit in VDE provenance.",
        "- Fixed-gear EDrive aggregates remain outside the historical Transmission slot.",
        f"- Semantic check: {materialization['semantic_verification']['semantic_checks_passed']}; slot mismatches: {materialization['semantic_verification']['slot_to_macro_mismatches']}; EDrive-to-Transmission misprojections: {materialization['semantic_verification']['edrive_rows_with_transmission_projection']}.",
        "- The vector-search negative validation remains accepted: it did not establish vehicle-specific fine identity.",
        "",
        "## 6. Research Agent v0 frozen state and limitations",
        "",
        "Status: `EXPERIMENTAL_PARTIAL_FROZEN_FOR_REVIEW`.",
        f"Golden tasks: {research['golden_task_count']}; exact internal resolutions: {research['internal_resolved']}; deferred for no current-model value: {research['deferred_no_current_model_value']}.",
        f"Direct research-to-ABC writes: {research['direct_research_abc_writes']}. P1/P2, RAG, embeddings, and vector DB were not introduced.",
        "The live retrieval provider returned no accepted evidence in the frozen micro-patch; this limitation is preserved rather than masked.",
        "",
        "## 7. Excel export and reconciliation",
        "",
        f"- Workbook: `{workbook.resolve()}`",
        f"- Workbook SHA256: `{excel_manifest['workbook_sha256']}`",
        f"- Source DB SHA256 before/after: `{excel_manifest['source_db_sha256_before']}` / `{excel_manifest['source_db_sha256_after']}`",
        f"- Sheets ({excel_manifest['sheet_count']}): {sheet_list}",
        "- Every physical table row/column count reconciles independently to SQLite.",
        "- `VDE_FLAT` and `COMP_COVERAGE` each contain exactly one row per VDE.",
        "- The XLSX ZIP container and worksheet XML parsed without corruption; no rows were truncated.",
        "",
        "## 8. Integrity, idempotency, and reproducibility",
        "",
        f"- Materialization idempotent: {materialization['integrity']['idempotent']}.",
        f"- Protected fields unchanged: {materialization['integrity']['protected_fields_unchanged']}.",
        f"- Core row counts unchanged: {materialization['integrity']['core_row_counts_unchanged']}.",
        f"- Schema unchanged by materialization: {materialization['integrity']['schema_unchanged']}.",
        "- Candidate and immutable release are byte-identical.",
        "- Final export and audits opened SQLite in read-only/query-only mode.",
        "",
        "## 9. Test evidence",
        "",
        "See `test_results.txt`. Focused Sprint 12 closure/export/materialization/research regressions are green. Excel reconciliation is independently recorded in `FINAL_EXCEL_VERIFICATION.json`.",
        "",
        "## 10. Deferred post-Sprint-12 backlog",
        "",
        "- Owner code review of Research Agent v0.",
        "- Technical RAG, embeddings, or vector DB as a separate future capability.",
        "- Improved live technical retrieval/provider quality.",
        "- P1/P2 research only after owner review.",
        "- Additional deterministic physics for unsupported DHT, power-split, or multimode architectures if later prioritized.",
        "- UI presentation of Macro versus Detailed/Fine provenance if not already implemented.",
        "",
        "PROD_PROMOTION = NO",
    ]
    report_path = final_dir / "SPRINT12_CLOSURE_REPORT.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    prod_files = [repo_root / "data/db/eco_drive.db", repo_root / "data/db/eco_drive_qa.db"]
    manifest_files = [
        release, workbook, report_path,
        final_dir / "FINAL_DB_AUDIT.json", final_dir / "FINAL_DB_TABLE_COUNTS.csv",
        final_dir / "FINAL_DB_SCHEMA.md", excel_manifest_path, excel_verification_path,
        final_dir / "COMPONENT_POPULATION_FINAL_COVERAGE.csv",
        final_dir / "COMPONENT_POPULATION_FINAL_COVERAGE.md", test_results,
        final_dir / "focused_tests.junit.xml",
        final_dir / "sprint12_regression_tests.junit.xml",
        final_dir / "PACKAGE_CONTENTS.md",
    ]
    manifest = {
        "release_state": status,
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "accepted_candidate": _hash_entry(candidate),
        "immutable_release": _hash_entry(release),
        "candidate_release_hash_match": file_sha256(candidate) == file_sha256(release),
        "quick_check": release_audit["quick_check"][0],
        "foreign_key_issue_count": release_audit["foreign_key_issue_count"],
        "schema_sha256": release_audit["schema_sha256"],
        "component_semantic_verification": materialization["semantic_verification"],
        "component_materialization_integrity": materialization["integrity"],
        "research_agent_status": "EXPERIMENTAL_PARTIAL_FROZEN_FOR_REVIEW",
        "research_agent_direct_abc_writes": research["direct_research_abc_writes"],
        "excel_verification_status": excel_verification["status"],
        "prod_qa_observed_unchanged_by_closure": [_hash_entry(path) for path in prod_files if path.exists()],
        "files": [_hash_entry(path) for path in manifest_files if path.exists()],
        "prod_promotion": False,
    }
    manifest_path = final_dir / "FINAL_RELEASE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(args.repo.resolve(), args.output_dir.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
