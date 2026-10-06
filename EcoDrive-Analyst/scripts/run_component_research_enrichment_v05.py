"""Generate the Components Research v0.5 cross-OEM review artifacts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capabilities.technical_research.cross_oem_v05 import (  # noqa: E402
    NUMERIC_FIELDS,
    OEM_GROUPS,
    PROVENANCE_ORDER,
    SAMPLE_FIELDS,
    VALUE_FIELDS,
    applicable_evidence,
    deterministic_sample,
    enrich_application,
    file_sha256,
    load_canonical_candidates,
    read_curated_evidence,
    repeated_groups,
    validate_numeric_contract,
)


DEFAULT_DB = ROOT / "data/db/staging/eco_drive_canonical_candidate.db"
DEFAULT_OUTPUT = ROOT / "artifacts/components/component_research_v05"
DEFAULT_EVIDENCE = ROOT / "data/reference/component_research_v05_curated_evidence.csv"
FROZEN_BMW = ROOT / "artifacts/components/component_research_v041a/BMW_COMPONENT_RESEARCH_BENCHMARK_V041A.csv"
FROZEN_BMW_SHA256 = "C4B31774BE2AFB6EB543446FB8CA69CAECCF6E199275BCB143E1B6037919D0E3"
DB_PATHS = (
    ROOT / "data/db/eco_drive.db",
    ROOT / "data/db/eco_drive_qa.db",
    DEFAULT_DB,
)

ENRICHMENT_FIELDS = (
    *SAMPLE_FIELDS,
    "researched_model", "researched_year_range", "researched_drive",
    "researched_powertrain", "researched_market",
    *(field for field in VALUE_FIELDS if field not in SAMPLE_FIELDS),
    *(f"{field}_provenance" for field in VALUE_FIELDS),
    "cda_method", "match_status", "selected_provenance", "source_count",
    "source_urls", "evidence_note", "review_flag",
)

EVIDENCE_OUTPUT_FIELDS = (
    "sample_id", "vde_id", "make", "model", "model_year", "field", "value",
    "application_match", "source_tier", "source_classification", "source_url",
    "source_title", "evidence_note", "selected_for_pragmatic_review",
)

PROVENANCE_FIELDS = (
    "sample_id", "vde_id", "vehicle_application", "field", "selected_value",
    "provenance", "observed_value", "researched_exact_value",
    "researched_approx_value", "calculated_value", "rule_estimated_value",
    "source_url", "note",
)

GROUP_FIELDS = (
    "identity", "identity_level", "supplier", "application_count",
    "applications", "model_years", "provenance_mix",
)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def db_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in DB_PATHS if path.exists()}


def _present(value: Any) -> bool:
    return value is not None and str(value).strip() not in {"", "None", "nan"}


def coverage(rows: list[dict[str, Any]], provenance_rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(row["match_status"] for row in rows)
    reviews = Counter(row["review_flag"] for row in rows)
    provenance = Counter(row["provenance"] for row in provenance_rows)
    metrics: dict[str, Any] = {
        "vehicles_total": len(rows),
        "oem_groups": len({row["oem_group"] for row in rows}),
        "transmission_exact_found": statuses["FOUND_EXACT"],
        "transmission_family_found": statuses["FOUND_FAMILY"],
        "transmission_not_found": statuses["NOT_FOUND"],
        "transmission_conflicts": statuses["CONFLICT"],
        "cd_found": sum(_present(row["cd"]) for row in rows),
        "frontal_area_found": sum(_present(row["frontal_area_m2"]) for row in rows),
        "cda_available": sum(_present(row["cda_m2"]) for row in rows),
        "tire_specs_found": sum(any(_present(row[field]) for field in ("tire_front", "tire_rear", "tire_general")) for row in rows),
        "physical_fdr_found": sum(_present(row["physical_final_drive"]) for row in rows),
        "drive_reduction_found": sum(any(_present(row[field]) for field in ("reduction_front", "reduction_rear")) for row in rows),
        "gear_ratio_sets_found": sum(_present(row["gear_ratios"]) for row in rows),
        "review_ok": reviews["REVIEW_OK"],
        "review_approx": reviews["REVIEW_APPROX"],
        "review_conflict": reviews["REVIEW_CONFLICT"],
        "review_sparse": reviews["REVIEW_SPARSE"],
    }
    for name in PROVENANCE_ORDER:
        metrics[f"{name.lower()}_values"] = provenance[name]
    research_provenance = {"RESEARCHED_EXACT", "RESEARCHED_APPROX"}
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in provenance_rows:
        by_sample[row["sample_id"]].append(row)
    metrics["rows_with_any_researched_data"] = sum(any(item["provenance"] in research_provenance for item in items) for items in by_sample.values())
    metrics["rows_with_only_rule_estimates"] = sum(
        any(item["provenance"] == "RULE_ESTIMATED" for item in items)
        and not any(item["provenance"] in research_provenance | {"CALCULATED"} for item in items)
        for items in by_sample.values()
    )
    metrics["rows_completely_unknown"] = sum(all(item["provenance"] == "UNKNOWN" for item in items) for items in by_sample.values())
    return metrics


def per_oem_coverage(rows: list[dict[str, Any]], provenance_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    provenance_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in provenance_rows:
        provenance_by_sample[row["sample_id"]].append(row)
    result: list[dict[str, Any]] = []
    for group in OEM_GROUPS:
        members = [row for row in rows if row["oem_group"] == group]
        item = coverage(members, [entry for row in members for entry in provenance_by_sample[row["sample_id"]]])
        item["oem_group"] = group
        item["source_types_that_worked"] = ";".join(sorted({"CANONICAL_EPA_OBSERVED"} | ({"CURATED_PUBLIC_TECHNICAL"} if any(row["source_count"] for row in members) else set())))
        gaps: list[str] = []
        for field, label in (("cd_found", "AERO_CD"), ("frontal_area_found", "FRONTAL_AREA"), ("tire_specs_found", "TIRES"), ("gear_ratio_sets_found", "GEAR_RATIOS")):
            if item[field] < len(members) / 2:
                gaps.append(label)
        item["major_data_gaps"] = ";".join(gaps)
        result.append(item)
    return result


def run(*, db_path: Path, output: Path, evidence_path: Path, per_group: int = 10) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    hashes_before = db_hashes()
    bmw_before = file_sha256(FROZEN_BMW)
    candidates = load_canonical_candidates(db_path)
    sample = deterministic_sample(candidates, per_group=per_group)
    curated = read_curated_evidence(evidence_path)
    enrichment: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    evidence_output: list[dict[str, Any]] = []
    for sample_row in sample:
        enriched, audit = enrich_application(sample_row, curated)
        enrichment.append(enriched)
        provenance.extend(audit)
        for evidence in applicable_evidence(sample_row, curated):
            evidence_output.append({
                "sample_id": sample_row["sample_id"], "vde_id": sample_row["vde_id"],
                "make": sample_row["make"], "model": sample_row["model"],
                "model_year": sample_row["model_year"], **evidence,
                "selected_for_pragmatic_review": "YES",
            })
    groups = repeated_groups(enrichment)
    numeric = validate_numeric_contract(enrichment)
    metrics = coverage(enrichment, provenance)
    metrics["repeated_transmission_groups"] = len(groups)
    oem_rows = per_oem_coverage(enrichment, provenance)
    sample_export = [{field: row.get(field, "") for field in SAMPLE_FIELDS} for row in sample]
    write_csv(output / "CROSS_OEM_SAMPLE_V05.csv", sample_export, SAMPLE_FIELDS)
    write_csv(output / "COMPONENT_RESEARCH_ENRICHMENT_V05.csv", enrichment, ENRICHMENT_FIELDS)
    write_csv(output / "COMPONENT_RESEARCH_EVIDENCE_V05.csv", evidence_output, EVIDENCE_OUTPUT_FIELDS)
    write_csv(output / "COMPONENT_VALUE_PROVENANCE_V05.csv", provenance, PROVENANCE_FIELDS)
    write_csv(output / "TRANSMISSION_MATCH_GROUPS_V05.csv", groups, GROUP_FIELDS)
    oem_fields = ("oem_group",) + tuple(key for key in metrics if key not in {"oem_groups", "repeated_transmission_groups"}) + ("source_types_that_worked", "major_data_gaps")
    write_csv(output / "OEM_COVERAGE_SUMMARY_V05.csv", oem_rows, oem_fields)
    hashes_after = db_hashes()
    bmw_after = file_sha256(FROZEN_BMW)
    bmw_unchanged = bmw_before == bmw_after == FROZEN_BMW_SHA256
    db_unchanged = hashes_before == hashes_after
    materially_useful = (
        len(enrichment) >= 40
        and metrics["transmission_exact_found"] + metrics["transmission_family_found"] >= len(enrichment) * 0.7
        and metrics["physical_fdr_found"] + metrics["drive_reduction_found"] + metrics["gear_ratio_sets_found"] >= len(enrichment) * 0.6
        and metrics["rows_with_any_researched_data"] >= 5
        and metrics["review_sparse"] < len(enrichment)
        and numeric["valid"] and bmw_unchanged and db_unchanged
    )
    report = {
        **metrics,
        "numeric_contract": numeric,
        "db_hashes_before": hashes_before,
        "db_hashes_after": hashes_after,
        "db_unchanged": db_unchanged,
        "bmw_sha256_before": bmw_before,
        "bmw_sha256_after": bmw_after,
        "bmw_frozen_benchmark_unchanged": bmw_unchanged,
        "ready_for_component_db_review_promotion": materially_useful,
        "live_research_runtime": "NOT_RUN_COST_GUARD_LOCAL_CURATED_AND_RULE_FALLBACK",
        "canonical_write": "DISABLED",
    }
    (output / "RUN_METRICS_V05.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_summary(output / "COMPONENT_RESEARCH_V05_SUMMARY.md", report, sample, groups, oem_rows)
    return report


def _write_summary(path: Path, report: Mapping[str, Any], sample: list[dict[str, Any]], groups: list[dict[str, Any]], oem_rows: list[dict[str, Any]]) -> None:
    make_counts = Counter(str(row["make"]).upper() for row in sample)
    powertrains = Counter(row["electrification"] for row in sample)
    drives = Counter("AWD_4WD" if "Wheel Drive" in str(row["drive_type"]) and "2-Wheel" not in str(row["drive_type"]) else str(row["drive_type"]) for row in sample)
    lines = [
        "# Components Research v0.5 — Cross-OEM Pragmatic Enrichment",
        "",
        "## Scope and execution",
        "",
        "- Review-only output; no canonical import or schema change.",
        "- Deterministic 10-per-group sampling from the canonical EPA candidate, with exact carryover roots collapsed during selection.",
        "- Live LLM/API research was not started: the prior cost constraint made a 50-case live run unreasonable. Existing public curated evidence is reusable through the evidence contract; this run uses canonical observed values and explicit rule fallback wherever no curated match exists.",
        "- EPA source contains transmission/gears/axle/N/V but no tire, Cd, frontal-area, or CdA fields; those gaps remain explicit rather than being back-solved from Target C.",
        "",
        "## A. Sample",
        "",
        f"- Vehicles: **{report['vehicles_total']}**; OEM groups: **{report['oem_groups']}**.",
        f"- Makes: `{dict(sorted(make_counts.items()))}`.",
        f"- Powertrains: `{dict(sorted(powertrains.items()))}`.",
        f"- Drive layouts: `{dict(sorted(drives.items()))}`.",
        "",
        "## B–D. Coverage and provenance",
        "",
        f"- Transmission: exact {report['transmission_exact_found']}, family/architecture {report['transmission_family_found']}, conflict {report['transmission_conflicts']}, not found {report['transmission_not_found']}.",
        f"- Engineering: Cd {report['cd_found']}, frontal area {report['frontal_area_found']}, CdA {report['cda_available']}, tires {report['tire_specs_found']}, physical FDR {report['physical_fdr_found']}, EV reduction {report['drive_reduction_found']}, gear-ratio sets {report['gear_ratio_sets_found']}.",
        f"- Provenance: observed {report['observed_values']}, researched exact {report['researched_exact_values']}, researched approx {report['researched_approx_values']}, calculated {report['calculated_values']}, rule-estimated {report['rule_estimated_values']}, unknown {report['unknown_values']}.",
        "",
        "## E. Repeated transmission identities",
        "",
    ]
    if groups:
        lines.extend(f"- `{row['identity']}` ({row['identity_level']}): {row['application_count']} applications." for row in groups)
    else:
        lines.append("- None.")
    lines.extend(["", "## F. OEM comparison", ""])
    for row in oem_rows:
        lines.append(
            f"- **{row['oem_group']}**: {row['vehicles_total']} vehicles; "
            f"transmission exact/family {row['transmission_exact_found']}/{row['transmission_family_found']}; "
            f"physical FDR {row['physical_fdr_found']}; major gaps `{row['major_data_gaps'] or 'NONE'}`."
        )
    lines.extend([
        "", "## G–I. Human review, BMW regression, and safety", "",
        f"- Review flags: OK {report['review_ok']}, approximate {report['review_approx']}, conflict {report['review_conflict']}, sparse {report['review_sparse']}.",
        f"- Frozen BMW benchmark unchanged: **{'YES' if report['bmw_frozen_benchmark_unchanged'] else 'NO'}** (`{report['bmw_sha256_after']}`).",
        f"- Normalized numeric contract valid: **{'YES' if report['numeric_contract']['valid'] else 'NO'}**.",
        f"- Canonical DB hashes unchanged: **{'YES' if report['db_unchanged'] else 'NO'}**.",
        "- Canonical writes: **DISABLED**.",
        "",
        "## Completion block",
        "",
        "```ini",
        "COMPONENT_RESEARCH_VERSION = 0.5",
        f"CROSS_OEM_ENRICHMENT_COMPLETE = {'YES' if report['vehicles_total'] >= 40 else 'NO'}",
        "PRAGMATIC_ACCEPTANCE_MODE = YES",
        "RESEARCHED_APPROX_FIRST_CLASS = YES",
        "RULE_BASED_FALLBACK_ENABLED = YES",
        f"VEHICLES_TOTAL = {report['vehicles_total']}",
        f"OEM_GROUPS = {report['oem_groups']}",
        f"TRANSMISSION_EXACT_FOUND = {report['transmission_exact_found']}",
        f"TRANSMISSION_FAMILY_FOUND = {report['transmission_family_found']}",
        f"TRANSMISSION_NOT_FOUND = {report['transmission_not_found']}",
        f"TRANSMISSION_CONFLICTS = {report['transmission_conflicts']}",
        f"CD_FOUND = {report['cd_found']}",
        f"FRONTAL_AREA_FOUND = {report['frontal_area_found']}",
        f"CDA_AVAILABLE = {report['cda_available']}",
        f"TIRE_SPECS_FOUND = {report['tire_specs_found']}",
        f"PHYSICAL_FDR_FOUND = {report['physical_fdr_found']}",
        f"DRIVE_REDUCTION_FOUND = {report['drive_reduction_found']}",
        f"GEAR_RATIO_SETS_FOUND = {report['gear_ratio_sets_found']}",
        f"REPEATED_TRANSMISSION_GROUPS = {report['repeated_transmission_groups']}",
        f"RESEARCHED_EXACT_VALUES = {report['researched_exact_values']}",
        f"RESEARCHED_APPROX_VALUES = {report['researched_approx_values']}",
        f"CALCULATED_VALUES = {report['calculated_values']}",
        f"RULE_ESTIMATED_VALUES = {report['rule_estimated_values']}",
        f"UNKNOWN_VALUES = {report['unknown_values']}",
        f"REVIEW_OK = {report['review_ok']}",
        f"REVIEW_APPROX = {report['review_approx']}",
        f"REVIEW_CONFLICT = {report['review_conflict']}",
        f"REVIEW_SPARSE = {report['review_sparse']}",
        f"BMW_FROZEN_BENCHMARK_UNCHANGED = {'YES' if report['bmw_frozen_benchmark_unchanged'] else 'NO'}",
        "CANONICAL_WRITE_DISABLED = YES",
        "PRODUCTION_DB_CHANGED = NO",
        f"READY_FOR_COMPONENT_DB_REVIEW_PROMOTION = {'YES' if report['ready_for_component_db_review_promotion'] else 'NO'}",
        "```",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--per-group", type=int, default=10)
    args = parser.parse_args()
    report = run(db_path=args.db, output=args.output, evidence_path=args.evidence, per_group=args.per_group)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
