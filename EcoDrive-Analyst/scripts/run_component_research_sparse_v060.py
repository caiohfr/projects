"""Run the final bounded v0.6 research pass over v0.5.1 sparse rows only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capabilities.technical_research import live_runtime  # noqa: E402
from capabilities.technical_research.cross_oem_v05 import (  # noqa: E402
    PROVENANCE_RANK, file_sha256, present,
)
from capabilities.technical_research.cross_oem_v051 import (  # noqa: E402
    CrossOemPragmaticSourcePolicy, enrich_application_v051,
)
from capabilities.technical_research.golden_retrieval_v052 import (  # noqa: E402
    normalize_search_identity,
)
from scripts.run_component_research_enrichment_v051 import (  # noqa: E402
    DB_PATHS, FROZEN_BMW, FROZEN_BMW_SHA256,
)
from scripts.run_golden_retrieval_benchmark_v052a import run_case  # noqa: E402


BASELINE_DIR = ROOT / "artifacts/components_research/v05_1"
DEFAULT_BASELINE = BASELINE_DIR / "COMPONENT_RESEARCH_ENRICHMENT_V051.csv"
DEFAULT_BASELINE_PROVENANCE = BASELINE_DIR / "COMPONENT_VALUE_PROVENANCE_V051.csv"
DEFAULT_GOLDEN_DIR = ROOT / "artifacts/components_research/v05_2a"
V052_GOLDEN_DIR = ROOT / "artifacts/components_research/v05_2"
DEFAULT_OUTPUT = ROOT / "artifacts/components_research/v06"

TECHNICAL_FIELDS = (
    "transmission_code", "transmission_family", "transmission_marketing_description",
    "transmission_supplier", "transmission_architecture", "gear_ratios",
    "physical_final_drive", "reduction_front", "reduction_rear", "cd",
    "frontal_area_m2", "cda_m2", "tire_front", "tire_rear", "tire_general",
)
NUMERIC_FIELDS = frozenset({
    "physical_final_drive", "reduction_front", "reduction_rear", "cd",
    "frontal_area_m2", "cda_m2",
})
IDENTITY_FIELDS = (
    "sample_id", "vehicle_configuration_id", "vde_id", "model_year", "make", "model",
    "normalized_search_identity", "category", "drive_type", "electrification",
    "transmission_type", "gears", "source_final_drive", "source_final_drive_provenance",
    "final_drive_semantics", "final_drive_semantics_provenance", "nv_ratio",
)
FINAL_FIELDS = (
    *IDENTITY_FIELDS,
    *TECHNICAL_FIELDS,
    *(f"{field}_provenance" for field in TECHNICAL_FIELDS),
    *(f"{field}_source" for field in TECHNICAL_FIELDS),
    *(f"{field}_review_level" for field in TECHNICAL_FIELDS),
    "review_flag", "review_notes", "search_stop_reason", "research_runtime_status",
    "searches", "research_calls", "fetched_sources", "accepted_useful_sources",
    "official_useful_sources", "secondary_useful_sources", "fetch_failures",
    "fallback_recoveries", "technical_attachments_followed", "runtime_errors",
)
EVIDENCE_FIELDS = (
    "sample_id", "application", "field", "value", "application_match", "source_tier",
    "source_classification", "source_url", "source_title", "evidence_note",
    "evidence_origin", "accepted_rejected", "selected_rejected_reason",
)
PROVENANCE_FIELDS = (
    "sample_id", "application", "field", "final_value", "provenance", "source_url",
    "exact_approx_calculated_status", "selected_rejected_reason",
)
QUERY_FIELDS = (
    "sample_id", "application", "theme", "query", "query_number",
    "official_secondary_intent", "why_query_was_issued", "result_rank", "source_url",
    "fetch_status", "fetch_failure", "fallback_result_used", "new_query_avoided",
    "result_selected", "results_returned", "search_status", "search_error",
)
SOURCE_FIELDS = (
    "sample_id", "application", "result_rank", "source_url", "source_title",
    "source_tier", "official", "fetch_status", "attachment_parent", "child_type",
    "useful_fields_found", "accepted_rejected", "rejection_reason", "application_match",
)
CASE_METRIC_FIELDS = (
    "sample_id", "searches", "research_calls", "fetched_sources",
    "accepted_useful_sources", "official_useful_sources", "secondary_useful_sources",
    "fetch_failures", "fallback_recoveries", "technical_attachments_followed",
    "materially_enriched", "runtime_errors", "search_stop_reason",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def db_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in DB_PATHS if path.exists()}


def application_label(row: Mapping[str, Any]) -> str:
    return f"{row.get('model_year', '')} {row.get('make', '')} {row.get('model', '')}".strip()


def evidence_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("sample_id", "")), str(row.get("field", "")),
        str(row.get("value", "")), str(row.get("source_url", "")),
        str(row.get("application_match", "")), str(row.get("accepted_rejected", "")),
    )


def dedupe(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for raw in rows:
        row = dict(raw)
        key = evidence_key(row)
        if key not in seen:
            seen.add(key)
            output.append(row)
    return output


def baseline_selected_evidence(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        provenance = str(row.get("provenance") or "")
        field = str(row.get("field") or "")
        value = row.get("selected_value")
        if field not in TECHNICAL_FIELDS or provenance not in {"RESEARCHED_EXACT", "RESEARCHED_APPROX"} or not present(value):
            continue
        output.append({
            "sample_id": row.get("sample_id", ""), "field": field, "value": value,
            "application_match": "EXACT" if provenance == "RESEARCHED_EXACT" else "PARTIAL",
            "source_tier": "PRESERVED_V051", "source_classification": "PRESERVED_RESEARCH",
            "source_url": row.get("source_url", ""), "source_title": "",
            "evidence_note": row.get("note", ""), "evidence_origin": "V051_SELECTED_PRESERVED",
            "accepted_rejected": "ACCEPTED", "selected_rejected_reason": "PRESERVED_BASELINE_SELECTION",
        })
    return output


def golden_selected_evidence(golden_dir: Path) -> list[dict[str, Any]]:
    results = read_csv(golden_dir / "GOLDEN_RETRIEVAL_RESULTS_V052A.csv")
    sources = read_csv(golden_dir / "GOLDEN_RETRIEVAL_SOURCE_AUDIT_V052A.csv")
    source_by_case_field: dict[tuple[str, str], dict[str, str]] = {}
    for source in sources:
        for field in str(source.get("selected_fields") or "").split(";"):
            if field:
                source_by_case_field[(str(source.get("case")), field)] = source
    output: list[dict[str, Any]] = []
    for result in results:
        sample_id = str(result.get("sample_id") or "")
        case = str(result.get("golden_case_id") or "")
        for field in TECHNICAL_FIELDS:
            value = result.get(field)
            if not present(value):
                continue
            source = source_by_case_field.get((case, field), {})
            match_text = str(source.get("application_match") or "")
            match = "EXACT" if "EXACT" in match_text else "PARTIAL"
            output.append({
                "sample_id": sample_id, "field": field, "value": value,
                "application_match": match,
                "source_tier": "TIER_1_PRIMARY" if source.get("source_class") == "OFFICIAL" else "TIER_2_STRONG_SECONDARY",
                "source_classification": source.get("source_class", "PRESERVED_V052A"),
                "source_url": source.get("source_url", ""), "source_title": source.get("source_title", ""),
                "evidence_note": "Accepted final v0.5.2a evidence preserved without source injection.",
                "evidence_origin": "V052A_SELECTED_PRESERVED", "accepted_rejected": "ACCEPTED",
                "selected_rejected_reason": "PRESERVED_ACCEPTED_V052A",
            })
    return output


def v052_selected_evidence(golden_dir: Path) -> list[dict[str, Any]]:
    results = read_csv(golden_dir / "GOLDEN_RETRIEVAL_RESULTS_V052.csv")
    sources = read_csv(golden_dir / "GOLDEN_RETRIEVAL_SOURCE_AUDIT_V052.csv")
    output: list[dict[str, Any]] = []
    for result in results:
        case = str(result.get("golden_case_id") or "")
        case_sources = [row for row in sources if row.get("case") == case]
        for field in TECHNICAL_FIELDS:
            value = result.get(field)
            if not present(value):
                continue
            source = next((
                row for row in case_sources
                if field in str(row.get("fields_extracted") or "").split(";")
            ), {})
            match_text = str(source.get("application_match") or "")
            match = "EXACT" if "EXACT" in match_text or result.get("strongest_provenance") == "RESEARCHED_EXACT" else "PARTIAL"
            output.append({
                "sample_id": result.get("sample_id", ""), "field": field, "value": value,
                "application_match": match,
                "source_tier": "TIER_1_PRIMARY" if source.get("source_class") == "OFFICIAL" else "TIER_2_STRONG_SECONDARY",
                "source_classification": source.get("source_class", "PRESERVED_V052"),
                "source_url": source.get("source_url", ""), "source_title": source.get("source_title", ""),
                "evidence_note": "Accepted v0.5.2 evidence preserved during final consolidation.",
                "evidence_origin": "V052_SELECTED_PRESERVED", "accepted_rejected": "ACCEPTED",
                "selected_rejected_reason": "PRESERVED_ACCEPTED_V052",
            })
    return output


def resolve_evidence_precedence(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Prevent weaker approximate claims from displacing exact accepted evidence."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in dedupe(rows):
        if raw.get("accepted_rejected", "ACCEPTED") != "REJECTED" and present(raw.get("value")):
            grouped[str(raw.get("field"))].append(raw)
    selected: list[dict[str, Any]] = []
    for claims in grouped.values():
        exact = [row for row in claims if str(row.get("application_match", "")).upper() == "EXACT"]
        selected.extend(exact or claims)
    return selected


def materially_enriched(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    for field in TECHNICAL_FIELDS:
        old_value, new_value = before.get(field), after.get(field)
        old_prov = str(before.get(f"{field}_provenance") or "UNKNOWN")
        new_prov = str(after.get(f"{field}_provenance") or "UNKNOWN")
        if not present(old_value) and present(new_value):
            return True
        if present(new_value) and PROVENANCE_RANK.get(new_prov, 0) > PROVENANCE_RANK.get(old_prov, 0):
            return True
    return False


def review_notes(row: Mapping[str, Any]) -> str:
    if row.get("review_flag") == "REVIEW_CONFLICT":
        return "Material value or application-identity conflict remains; inspect evidence audit."
    if row.get("review_flag") == "REVIEW_APPROX":
        return "Useful coverage includes defensible approximate/family/adjacent-MY evidence."
    if row.get("review_flag") == "REVIEW_SPARSE":
        missing = [field for field in TECHNICAL_FIELDS if not present(row.get(field))]
        return "Bounded research exhausted; unresolved for Estimation Methodology v1: " + ", ".join(missing)
    return "Useful researched coverage with no unresolved material conflict."


def conflict_review_note(sample: Mapping[str, Any], evidence: Iterable[Mapping[str, Any]]) -> str:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in evidence:
        if present(row.get("value")):
            grouped[str(row.get("field"))].append(row)
    conflicts: list[str] = []
    for field, claims in grouped.items():
        values = {str(row.get("value")).strip() for row in claims}
        if len(values) < 2:
            continue
        detail = "; ".join(
            f"{row.get('value')} [{row.get('source_title') or row.get('source_url') or 'source identity unavailable'}]"
            for row in claims
        )
        conflicts.append(f"{field}: {detail}")
    return (
        f"Raw sample identity: {application_label(sample)}. Conflicting source identity/evidence: "
        + (" | ".join(conflicts) if conflicts else "material application identity conflict; see evidence audit")
        + ". Recommended resolution: human application/source-applicability review before promotion."
    )


def selected_source_by_field(audit: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    return {
        str(row.get("field")): str(row.get("source_url") or "")
        for row in audit if present(row.get("selected_value"))
    }


def prepare_final_row(
    enriched: Mapping[str, Any], *, sample: Mapping[str, Any], audit: Iterable[Mapping[str, Any]],
    result: Mapping[str, Any], calls: int, source_rows: Sequence[Mapping[str, Any]],
    query_rows: Sequence[Mapping[str, Any]], material: bool,
) -> dict[str, Any]:
    sources = selected_source_by_field(audit)
    final = dict(enriched)
    final["normalized_search_identity"] = sample["normalized_search_identity"]
    for field in TECHNICAL_FIELDS:
        provenance = str(final.get(f"{field}_provenance") or "UNKNOWN")
        final[f"{field}_source"] = sources.get(field, "")
        final[f"{field}_review_level"] = provenance
    official_useful = sum(row.get("source_class") == "OFFICIAL" and present(row.get("selected_fields")) for row in source_rows)
    secondary_useful = sum(row.get("source_class") == "SECONDARY" and present(row.get("selected_fields")) for row in source_rows)
    final.update({
        "review_notes": review_notes(final),
        "search_stop_reason": result.get("search_stop_reason", ""),
        "research_runtime_status": "COMPLETE",
        "searches": result.get("searches_count", 0), "research_calls": calls,
        "fetched_sources": result.get("fetched_sources_count", 0),
        "accepted_useful_sources": result.get("accepted_sources_count", 0),
        "official_useful_sources": official_useful, "secondary_useful_sources": secondary_useful,
        "fetch_failures": sum(row.get("fetch_status") == "FAILED" for row in source_rows),
        "fallback_recoveries": sum(row.get("fallback_result_used") == "YES" and row.get("new_query_avoided") == "YES" for row in query_rows),
        "technical_attachments_followed": sum(bool(row.get("parent_url")) and row.get("fetch_status") == "SUCCESS" for row in source_rows),
        "runtime_errors": result.get("runtime_errors", 0),
        "_materially_enriched": "YES" if material else "NO",
    })
    return final


def map_query_rows(sample: Mapping[str, Any], rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        output.append({
            "sample_id": sample["sample_id"], "application": application_label(sample),
            "theme": row.get("theme", ""), "query": row.get("query", ""),
            "query_number": row.get("query_number", ""),
            "official_secondary_intent": "OFFICIAL" if row.get("official_attempted") == "YES" else "SECONDARY_OR_GAP",
            "why_query_was_issued": row.get("reason", ""), "result_rank": row.get("result_rank", ""),
            "source_url": row.get("source_url", ""), "fetch_status": row.get("fetch_status", ""),
            "fetch_failure": row.get("fetch_failure", ""),
            "fallback_result_used": row.get("fallback_result_used", ""),
            "new_query_avoided": row.get("new_query_avoided", ""),
            "result_selected": row.get("result_selected", ""),
            "results_returned": row.get("results_returned", ""),
            "search_status": row.get("search_status", ""), "search_error": row.get("search_error", ""),
        })
    return output


def map_source_rows(sample: Mapping[str, Any], rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        output.append({
            "sample_id": sample["sample_id"], "application": application_label(sample),
            "result_rank": row.get("result_rank", ""), "source_url": row.get("source_url", ""),
            "source_title": row.get("source_title", ""),
            "source_tier": "TIER_1_PRIMARY" if row.get("source_class") == "OFFICIAL" else "TIER_2_STRONG_SECONDARY",
            "official": "YES" if row.get("source_class") == "OFFICIAL" else "NO",
            "fetch_status": row.get("fetch_status", ""), "attachment_parent": row.get("parent_url", ""),
            "child_type": row.get("child_type", ""), "useful_fields_found": row.get("fields_extracted", ""),
            "accepted_rejected": row.get("accepted_rejected", ""),
            "rejection_reason": row.get("rejection_reason", ""),
            "application_match": row.get("application_match", ""),
        })
    return output


def live_evidence_rows(sample: Mapping[str, Any], accepted: Iterable[Mapping[str, Any]], sources: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in accepted:
        output.append({
            "sample_id": sample["sample_id"], "application": application_label(sample),
            "field": row.get("field", ""), "value": row.get("value", ""),
            "application_match": row.get("application_match", ""), "source_tier": row.get("source_tier", ""),
            "source_classification": row.get("source_classification", ""),
            "source_url": row.get("source_url", ""), "source_title": row.get("source_title", ""),
            "evidence_note": row.get("evidence_note", ""), "evidence_origin": "V060_LIVE",
            "accepted_rejected": "ACCEPTED", "selected_rejected_reason": "ACCEPTED_BY_V052A_FROZEN_WORKFLOW",
        })
    for row in sources:
        if row.get("accepted_rejected") != "REJECTED":
            continue
        output.append({
            "sample_id": sample["sample_id"], "application": application_label(sample),
            "field": "", "value": "", "application_match": row.get("application_match", ""),
            "source_tier": "TIER_1_PRIMARY" if row.get("source_class") == "OFFICIAL" else "TIER_2_STRONG_SECONDARY",
            "source_classification": row.get("source_class", ""),
            "source_url": row.get("source_url", ""), "source_title": row.get("source_title", ""),
            "evidence_note": "", "evidence_origin": "V060_LIVE",
            "accepted_rejected": "REJECTED", "selected_rejected_reason": row.get("rejection_reason", "NO_USEFUL_ACCEPTED_CLAIMS"),
        })
    return output


def checkpoint_paths(output: Path) -> dict[str, Path]:
    checkpoint = output / "checkpoints"
    return {
        "results": checkpoint / "target_results.csv",
        "evidence": checkpoint / "live_evidence.csv",
        "queries": checkpoint / "query_audit.csv",
        "sources": checkpoint / "source_audit.csv",
        "metrics": checkpoint / "case_metrics.csv",
        "selection": checkpoint / "selected_evidence.csv",
    }


def write_checkpoints(output: Path, results: list[dict[str, Any]], evidence: list[dict[str, Any]], queries: list[dict[str, Any]], sources: list[dict[str, Any]], metrics: list[dict[str, Any]], selections: list[dict[str, Any]]) -> None:
    paths = checkpoint_paths(output)
    write_csv(paths["results"], results, FINAL_FIELDS)
    write_csv(paths["evidence"], evidence, EVIDENCE_FIELDS)
    write_csv(paths["queries"], queries, QUERY_FIELDS)
    write_csv(paths["sources"], sources, SOURCE_FIELDS)
    write_csv(paths["metrics"], metrics, CASE_METRIC_FIELDS)
    selection_fields = (
        "sample_id", "field", "value", "application_match", "source_tier",
        "source_classification", "source_url", "source_title", "evidence_note",
        "evidence_origin", "accepted_rejected", "selected_rejected_reason",
    )
    write_csv(paths["selection"], selections, selection_fields)


def coverage_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "transmission_code": sum(present(row.get("transmission_code")) for row in rows),
        "transmission_family": sum(present(row.get("transmission_family")) for row in rows),
        "transmission_supplier": sum(present(row.get("transmission_supplier")) for row in rows),
        "transmission_architecture": sum(present(row.get("transmission_architecture")) for row in rows),
        "gear_ratios": sum(present(row.get("gear_ratios")) for row in rows),
        "physical_final_drive": sum(present(row.get("physical_final_drive")) for row in rows),
        "ev_reduction": sum(present(row.get("reduction_front")) or present(row.get("reduction_rear")) for row in rows),
        "cd": sum(present(row.get("cd")) for row in rows),
        "frontal_area_m2": sum(present(row.get("frontal_area_m2")) for row in rows),
        "cda_m2": sum(present(row.get("cda_m2")) for row in rows),
        "tires": sum(any(present(row.get(field)) for field in ("tire_front", "tire_rear", "tire_general")) for row in rows),
    }


def provenance_matrix(rows: Sequence[Mapping[str, Any]]) -> dict[str, Counter[str]]:
    matrix: dict[str, Counter[str]] = {}
    for field in TECHNICAL_FIELDS:
        counter: Counter[str] = Counter()
        for row in rows:
            provenance = str(row.get(f"{field}_provenance") or "UNKNOWN")
            counter[provenance] += 1
        matrix[field] = counter
    return matrix


def final_provenance_rows(rows: Sequence[Mapping[str, Any]], targeted_ids: set[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        targeted = str(row.get("sample_id")) in targeted_ids
        for field in TECHNICAL_FIELDS:
            provenance = str(row.get(f"{field}_provenance") or "UNKNOWN")
            output.append({
                "sample_id": row.get("sample_id", ""), "application": application_label(row),
                "field": field, "final_value": row.get(field, ""), "provenance": provenance,
                "source_url": row.get(f"{field}_source", ""),
                "exact_approx_calculated_status": provenance,
                "selected_rejected_reason": "SELECTED_BY_PRECEDENCE" if targeted else "PRESERVED_V051_NON_TARGET",
            })
    return output


def make_summary(report: Mapping[str, Any], before_rows: Sequence[Mapping[str, Any]], after_rows: Sequence[Mapping[str, Any]]) -> str:
    before_reviews = Counter(row.get("review_flag") for row in before_rows)
    after_reviews = Counter(row.get("review_flag") for row in after_rows)
    before_coverage = coverage_counts(before_rows)
    after_coverage = coverage_counts(after_rows)
    matrix = provenance_matrix(after_rows)
    lines = [
        "# Components Research v0.6 — Final Cross-OEM Sparse Enrichment", "",
        "## Population and review status", "",
        f"- Baseline: **{len(before_rows)}** applications; targeted sparse rows: **{report['targeted']}**; actually rerun: **{report['rerun']}**.",
        "", "| Review flag | v0.5.1 | v0.6 |", "|---|---:|---:|",
    ]
    for status in ("REVIEW_OK", "REVIEW_APPROX", "REVIEW_CONFLICT", "REVIEW_SPARSE"):
        lines.append(f"| {status} | {before_reviews[status]} | {after_reviews[status]} |")
    lines.extend(["", "## Field coverage", "", "| Field | v0.5.1 | v0.6 | Observed | Exact | Approx | Calculated | Rule | Unknown |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for field in TECHNICAL_FIELDS:
        counter = matrix[field]
        lines.append(
            f"| {field} | {sum(present(row.get(field)) for row in before_rows)} | {sum(present(row.get(field)) for row in after_rows)} | "
            f"{counter['OBSERVED']} | {counter['RESEARCHED_EXACT']} | {counter['RESEARCHED_APPROX']} | {counter['CALCULATED']} | {counter['RULE_ESTIMATED']} | {counter['UNKNOWN']} |"
        )
    lines.extend([
        "", "## Operational metrics", "",
        f"- Searches: **{report['searches']}**; mean/median per rerun application: **{report['mean_searches']:.2f} / {report['median_searches']:.1f}**.",
        f"- Fetched/useful/official/secondary sources: **{report['fetched']} / {report['accepted']} / {report['official']} / {report['secondary']}**.",
        f"- Attachments/fetch failures/fallback recoveries: **{report['attachments']} / {report['fetch_failures']} / {report['fallback_recoveries']}**.",
        f"- Model extraction calls/escalations: **{report['calls']} / 0**.",
        f"- Materially enriched applications: **{report['material']}**; fields added: **{report['fields_added']}**; fields added per 100 searches: **{report['fields_per_100']:.2f}**.",
        "", "## Safety and closure", "",
        f"- Golden source injection used: **NO**; benchmark-only source fixtures disabled at runtime.",
        f"- RULE_ESTIMATED technical values before/after: **{report['rule_estimated_before']} / {report['rule_estimated_after']}**; no new rule estimate introduced.",
        f"- BMW frozen benchmark unchanged: **{'YES' if report['bmw_ok'] else 'NO'}**.",
        f"- Canonical DB hashes unchanged: **{'YES' if report['db_ok'] else 'NO'}**.",
        f"- Remaining sparse applications are explicitly handed to Estimation Methodology v1.",
        "", "## Required status block", "", "```ini",
        status_block(report), "```", "",
    ])
    return "\n".join(lines)


def status_block(report: Mapping[str, Any]) -> str:
    return "\n".join([
        "COMPONENT_RESEARCH_VERSION = 0.6", "",
        f"CROSS_OEM_FINAL_RESEARCH_RUN_COMPLETE = {'YES' if report['complete'] else 'NO'}", "",
        "TOTAL_CROSS_OEM_APPLICATIONS = 50",
        f"APPLICATIONS_TARGETED_FOR_RERUN = {report['targeted']}",
        f"APPLICATIONS_ACTUALLY_RERUN = {report['rerun']}",
        f"APPLICATIONS_MATERIALLY_ENRICHED = {report['material']}", "",
        f"REVIEW_OK_FINAL = {report['reviews']['REVIEW_OK']}",
        f"REVIEW_APPROX_FINAL = {report['reviews']['REVIEW_APPROX']}",
        f"REVIEW_CONFLICT_FINAL = {report['reviews']['REVIEW_CONFLICT']}",
        f"REVIEW_SPARSE_FINAL = {report['reviews']['REVIEW_SPARSE']}", "",
        f"TRANSMISSION_CODE_COVERAGE = {report['coverage']['transmission_code']}/50",
        f"TRANSMISSION_FAMILY_COVERAGE = {report['coverage']['transmission_family']}/50",
        f"TRANSMISSION_SUPPLIER_COVERAGE = {report['coverage']['transmission_supplier']}/50",
        f"TRANSMISSION_ARCHITECTURE_COVERAGE = {report['coverage']['transmission_architecture']}/50",
        f"GEAR_RATIO_SET_COVERAGE = {report['coverage']['gear_ratios']}/50",
        f"PHYSICAL_FDR_COVERAGE = {report['coverage']['physical_final_drive']}/50",
        f"EV_REDUCTION_COVERAGE = {report['coverage']['ev_reduction']}/50",
        f"CD_COVERAGE = {report['coverage']['cd']}/50",
        f"FRONTAL_AREA_COVERAGE = {report['coverage']['frontal_area_m2']}/50",
        f"CDA_COVERAGE = {report['coverage']['cda_m2']}/50",
        f"TIRE_COVERAGE = {report['coverage']['tires']}/50", "",
        f"TOTAL_SEARCHES = {report['searches']}",
        f"MEDIAN_SEARCHES_PER_RERUN_APPLICATION = {report['median_searches']:.1f}",
        f"TOTAL_FETCHED_SOURCES = {report['fetched']}",
        f"USEFUL_ACCEPTED_SOURCES = {report['accepted']}",
        f"OFFICIAL_USEFUL_SOURCES = {report['official']}", "",
        "GOLDEN_SOURCE_INJECTION_USED = NO",
        "SEARCH_PLANNER_CHANGED = NO", "",
        "DEFAULT_EXTRACTION_MODEL = gpt-5.6-terra/medium",
        "STRONGER_MODEL_ESCALATIONS = 0", "",
        f"BMW_FROZEN_BENCHMARK_UNCHANGED = {'YES' if report['bmw_ok'] else 'NO'}", "",
        "CANONICAL_WRITE_DISABLED = YES",
        f"PRODUCTION_DB_CHANGED = {'NO' if report['db_ok'] else 'YES'}", "",
        f"BROAD_EXTERNAL_COMPONENT_RESEARCH_FOR_SPRINT12 = {'CLOSED' if report['complete'] else 'OPEN'}",
        f"READY_FOR_ESTIMATION_METHODOLOGY_V1 = {'YES' if report['complete'] else 'NO'}",
    ])


def run(*, output: Path, live: bool, resume: bool) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    baseline = read_csv(DEFAULT_BASELINE)
    baseline_provenance = read_csv(DEFAULT_BASELINE_PROVENANCE)
    if len(baseline) != 50 or len({row.get("sample_id") for row in baseline}) != 50:
        raise ValueError("v0.5.1 baseline must contain exactly 50 unique applications")
    targeted = [row for row in baseline if row.get("review_flag") == "REVIEW_SPARSE"]
    if len(targeted) != 36:
        raise ValueError(f"Expected 36 actual REVIEW_SPARSE rows, found {len(targeted)}")
    if any(str(row.get("make", "")).upper() == "BMW" for row in targeted):
        raise AssertionError("BMW must not enter the v0.6 target population")

    hashes_before = db_hashes()
    bmw_before = file_sha256(FROZEN_BMW)
    baseline_evidence = baseline_selected_evidence(baseline_provenance)
    v052_evidence = v052_selected_evidence(V052_GOLDEN_DIR)
    golden_evidence = golden_selected_evidence(DEFAULT_GOLDEN_DIR)
    preserved_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in (*baseline_evidence, *v052_evidence, *golden_evidence):
        preserved_by_sample[str(row.get("sample_id"))].append(row)

    paths = checkpoint_paths(output)
    target_results = read_csv(paths["results"]) if resume else []
    live_evidence = read_csv(paths["evidence"]) if resume else []
    query_audit = read_csv(paths["queries"]) if resume else []
    source_audit = read_csv(paths["sources"]) if resume else []
    case_metrics = read_csv(paths["metrics"]) if resume else []
    selected_evidence = read_csv(paths["selection"]) if resume else []
    completed = {str(row.get("sample_id")) for row in target_results}

    runtime = None
    if live and len(completed) < len(targeted):
        runtime = live_runtime(model="gpt-5.6-terra", reasoning_effort="medium")
        runtime.source_policy = CrossOemPragmaticSourcePolicy()
        runtime.claim_extractor.max_document_chars = 24_000

    for baseline_row in targeted:
        sample_id = str(baseline_row["sample_id"])
        if sample_id in completed:
            continue
        if runtime is None:
            break
        sample = dict(baseline_row)
        sample.update({
            "golden_case_id": sample_id,
            "model_alias": sample.get("model", ""),
            "official_hint": "", "official_model_alias": "",
            "normalized_search_identity": normalize_search_identity(sample),
        })
        accepted: list[dict[str, Any]] = []
        extraction_start = len(runtime.claim_extractor.audit)
        try:
            result, raw_queries, raw_sources, _, _, _ = run_case(
                runtime, sample, known_sources={}, evidence_sink=accepted,
            )
            calls = len(runtime.claim_extractor.audit) - extraction_start
            if any(str(row.get("result_rank")) == "0" for row in raw_queries):
                raise AssertionError("Synthetic rank-0 source detected in v0.6")
            mapped_queries = map_query_rows(sample, raw_queries)
            mapped_sources = map_source_rows(sample, raw_sources)
            accepted_rows = live_evidence_rows(sample, accepted, raw_sources)
            combined = resolve_evidence_precedence([
                *preserved_by_sample.get(sample_id, ()),
                *accepted_rows,
            ])
            enriched, provenance = enrich_application_v051(sample, combined)
            material = materially_enriched(baseline_row, enriched)
            final = prepare_final_row(
                enriched, sample=sample, audit=provenance, result=result, calls=calls,
                source_rows=raw_sources, query_rows=raw_queries, material=material,
            )
            target_results.append(final)
            live_evidence.extend(accepted_rows)
            query_audit.extend(mapped_queries)
            source_audit.extend(mapped_sources)
            selected_evidence.extend({"sample_id": sample_id, **row} for row in combined)
            case_metrics.append({
                "sample_id": sample_id, "searches": result.get("searches_count", 0),
                "research_calls": calls, "fetched_sources": result.get("fetched_sources_count", 0),
                "accepted_useful_sources": result.get("accepted_sources_count", 0),
                "official_useful_sources": final["official_useful_sources"],
                "secondary_useful_sources": final["secondary_useful_sources"],
                "fetch_failures": final["fetch_failures"],
                "fallback_recoveries": final["fallback_recoveries"],
                "technical_attachments_followed": final["technical_attachments_followed"],
                "materially_enriched": "YES" if material else "NO",
                "runtime_errors": result.get("runtime_errors", 0),
                "search_stop_reason": result.get("search_stop_reason", ""),
            })
        except Exception as exc:
            # A failed application remains resumable and is not marked complete.
            write_checkpoints(output, target_results, live_evidence, query_audit, source_audit, case_metrics, selected_evidence)
            print(f"V060_ERROR sample={sample_id} error={type(exc).__name__}:{str(exc)[:200]}", flush=True)
            raise
        completed.add(sample_id)
        write_checkpoints(output, target_results, live_evidence, query_audit, source_audit, case_metrics, selected_evidence)
        print(
            f"V060_PROGRESS completed={len(completed)}/{len(targeted)} sample={sample_id} "
            f"review={final['review_flag']} fields={sum(present(final.get(field)) for field in TECHNICAL_FIELDS)} "
            f"searches={result['searches_count']} fetched={result['fetched_sources_count']} material={material}",
            flush=True,
        )

    final_by_sample = {str(row["sample_id"]): dict(row) for row in target_results}
    live_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in live_evidence:
        if row.get("accepted_rejected") != "REJECTED":
            live_by_sample[str(row.get("sample_id"))].append(row)
    final_rows: list[dict[str, Any]] = []
    for baseline_row in baseline:
        sample_id = str(baseline_row["sample_id"])
        if sample_id in final_by_sample:
            # Recompute selection from all preserved v0.5.1/v0.5.2/v0.5.2a
            # evidence plus the checkpointed live claims. This consolidation
            # is local and does not repeat research calls on resume.
            combined = resolve_evidence_precedence([
                *preserved_by_sample.get(sample_id, ()),
                *live_by_sample.get(sample_id, ()),
            ])
            enriched, audit = enrich_application_v051(baseline_row, combined)
            consolidated = dict(final_by_sample[sample_id])
            consolidated.update(enriched)
            consolidated["normalized_search_identity"] = normalize_search_identity(baseline_row)
            sources = selected_source_by_field(audit)
            for field in TECHNICAL_FIELDS:
                provenance = str(consolidated.get(f"{field}_provenance") or "UNKNOWN")
                consolidated[f"{field}_source"] = sources.get(field, "")
                consolidated[f"{field}_review_level"] = provenance
            consolidated["review_notes"] = (
                conflict_review_note(baseline_row, combined)
                if consolidated.get("review_flag") == "REVIEW_CONFLICT"
                else review_notes(consolidated)
            )
            final_rows.append(consolidated)
            continue
        preserved = dict(baseline_row)
        preserved["normalized_search_identity"] = normalize_search_identity(preserved)
        for field in TECHNICAL_FIELDS:
            source = next((
                str(item.get("source_url") or "")
                for item in baseline_provenance
                if item.get("sample_id") == sample_id and item.get("field") == field
            ), "")
            provenance = str(preserved.get(f"{field}_provenance") or "UNKNOWN")
            preserved[f"{field}_source"] = source
            preserved[f"{field}_review_level"] = provenance
        preserved.update({
            "review_notes": "Preserved v0.5.1 non-target row; no concrete defect justified rerun.",
            "search_stop_reason": "PRESERVED_NON_TARGET", "research_runtime_status": "NOT_RERUN",
            "searches": 0, "research_calls": 0, "fetched_sources": 0,
            "accepted_useful_sources": 0, "official_useful_sources": 0,
            "secondary_useful_sources": 0, "fetch_failures": 0, "fallback_recoveries": 0,
            "technical_attachments_followed": 0, "runtime_errors": 0,
        })
        final_rows.append(preserved)
    final_rows.sort(key=lambda row: str(row.get("sample_id")))

    # Final artifacts are emitted even for a dry/incomplete resume so status is explicit.
    selected_all = dedupe([*baseline_evidence, *v052_evidence, *golden_evidence, *live_evidence])
    evidence_output = []
    for row in selected_all:
        sample_id = str(row.get("sample_id") or "")
        baseline_match = next((item for item in baseline if item.get("sample_id") == sample_id), {})
        evidence_output.append({
            "sample_id": sample_id, "application": row.get("application") or application_label(baseline_match),
            **row,
        })
    provenance_output = final_provenance_rows(final_rows, {row["sample_id"] for row in targeted})
    write_csv(output / "COMPONENT_RESEARCH_CROSS_OEM_FINAL_V060.csv", final_rows, FINAL_FIELDS)
    write_csv(output / "COMPONENT_RESEARCH_EVIDENCE_V060.csv", evidence_output, EVIDENCE_FIELDS)
    write_csv(output / "COMPONENT_VALUE_PROVENANCE_V060.csv", provenance_output, PROVENANCE_FIELDS)
    write_csv(output / "COMPONENT_RESEARCH_QUERY_AUDIT_V060.csv", query_audit, QUERY_FIELDS)
    write_csv(output / "COMPONENT_RESEARCH_SOURCE_AUDIT_V060.csv", source_audit, SOURCE_FIELDS)

    hashes_after = db_hashes()
    bmw_after = file_sha256(FROZEN_BMW)
    metric_rows = case_metrics
    searches = sum(int(row.get("searches") or 0) for row in metric_rows)
    search_counts = [int(row.get("searches") or 0) for row in metric_rows]
    before_present = sum(present(row.get(field)) for row in baseline for field in TECHNICAL_FIELDS)
    after_present = sum(present(row.get(field)) for row in final_rows for field in TECHNICAL_FIELDS)
    before_rule = sum(row.get(f"{field}_provenance") == "RULE_ESTIMATED" for row in baseline for field in TECHNICAL_FIELDS)
    after_rule = sum(row.get(f"{field}_provenance") == "RULE_ESTIMATED" for row in final_rows for field in TECHNICAL_FIELDS)
    reviews = Counter(row.get("review_flag") for row in final_rows)
    complete = (
        len(final_rows) == 50 and len(completed) == len(targeted)
        and not any(str(row.get("result_rank")) == "0" for row in query_audit)
        and hashes_before == hashes_after
        and bmw_before == bmw_after == FROZEN_BMW_SHA256
        and after_present > before_present
        and after_rule <= before_rule
    )
    report: dict[str, Any] = {
        "complete": complete, "targeted": len(targeted), "rerun": len(completed),
        "material": sum(row.get("materially_enriched") == "YES" for row in metric_rows),
        "reviews": {
            status: reviews[status]
            for status in ("REVIEW_OK", "REVIEW_APPROX", "REVIEW_CONFLICT", "REVIEW_SPARSE")
        },
        "coverage": coverage_counts(final_rows),
        "searches": searches,
        "mean_searches": statistics.mean(search_counts) if search_counts else 0.0,
        "median_searches": statistics.median(search_counts) if search_counts else 0.0,
        "fetched": sum(int(row.get("fetched_sources") or 0) for row in metric_rows),
        "accepted": sum(int(row.get("accepted_useful_sources") or 0) for row in metric_rows),
        "official": sum(int(row.get("official_useful_sources") or 0) for row in metric_rows),
        "secondary": sum(int(row.get("secondary_useful_sources") or 0) for row in metric_rows),
        "attachments": sum(int(row.get("technical_attachments_followed") or 0) for row in metric_rows),
        "fetch_failures": sum(int(row.get("fetch_failures") or 0) for row in metric_rows),
        "fallback_recoveries": sum(int(row.get("fallback_recoveries") or 0) for row in metric_rows),
        "calls": sum(int(row.get("research_calls") or 0) for row in metric_rows),
        "fields_added": max(0, after_present - before_present),
        "fields_per_100": (max(0, after_present - before_present) * 100 / searches) if searches else 0.0,
        "golden_source_injection_used": False, "search_planner_changed": False,
        "rule_estimated_before": before_rule, "rule_estimated_after": after_rule,
        "stronger_model_escalations": 0, "model": "gpt-5.6-terra", "reasoning_effort": "medium",
        "bmw_ok": bmw_before == bmw_after == FROZEN_BMW_SHA256,
        "db_ok": hashes_before == hashes_after,
        "db_hashes_before": hashes_before, "db_hashes_after": hashes_after,
        "bmw_sha256_before": bmw_before, "bmw_sha256_after": bmw_after,
    }
    (output / "RUN_METRICS_V060.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (output / "COMPONENT_RESEARCH_CROSS_OEM_V060_SUMMARY.md").write_text(
        make_summary(report, baseline, final_rows), encoding="utf-8",
    )
    print(f"V060_COMPLETE status={'YES' if complete else 'NO'} rerun={len(completed)}/{len(targeted)} material={report['material']} searches={searches}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(output=args.output, live=args.live, resume=not args.no_resume)
    print(status_block(report), flush=True)
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
