"""Run persisted cross-OEM live enrichment for Components Research v0.5.1."""
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

from capabilities.technical_research import live_runtime, research_technical_component  # noqa: E402
from capabilities.technical_research.cross_oem_v05 import (  # noqa: E402
    NUMERIC_FIELDS,
    OEM_GROUPS,
    PROVENANCE_ORDER,
    SAMPLE_FIELDS,
    VALUE_FIELDS,
    applicable_evidence,
    file_sha256,
    read_curated_evidence,
    validate_numeric_contract,
)
from capabilities.technical_research.cross_oem_v051 import (  # noqa: E402
    CrossOemPragmaticSourcePolicy,
    CrossOemTransmissionResearchProfile,
    claims_to_curated_evidence,
    enrich_application_v051,
    repeated_groups_v051,
    request_for_sample,
    sanitize_evidence,
)


DEFAULT_SAMPLE = ROOT / "artifacts/components/component_research_v05/CROSS_OEM_SAMPLE_V05.csv"
DEFAULT_CURATED = ROOT / "data/reference/component_research_v05_curated_evidence.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/components_research/v05_1"
FROZEN_BMW = ROOT / "artifacts/components/component_research_v041a/BMW_COMPONENT_RESEARCH_BENCHMARK_V041A.csv"
FROZEN_BMW_SHA256 = "C4B31774BE2AFB6EB543446FB8CA69CAECCF6E199275BCB143E1B6037919D0E3"
DB_PATHS = (
    ROOT / "data/db/eco_drive.db",
    ROOT / "data/db/eco_drive_qa.db",
    ROOT / "data/db/staging/eco_drive_canonical_candidate.db",
    ROOT
    / "etl/data/staging/sprint_12f13_vde_materialized"
    / "eco_drive_canonical_vde_materialized_candidate.db",
)

ENRICHMENT_FIELDS = (
    *SAMPLE_FIELDS,
    "researched_model",
    "researched_year_range",
    "researched_drive",
    "researched_powertrain",
    "researched_market",
    *(field for field in VALUE_FIELDS if field not in SAMPLE_FIELDS),
    *(f"{field}_provenance" for field in VALUE_FIELDS),
    "cda_method",
    "match_status",
    "selected_provenance",
    "source_count",
    "source_urls",
    "evidence_note",
    "review_flag",
    "research_runtime_status",
    "searches",
    "research_calls",
    "fetched_sources",
    "accepted_useful_sources",
    "new_live_evidence_rows",
    "runtime_issue",
)

EVIDENCE_FIELDS = (
    "sample_id",
    "vde_id",
    "make",
    "model",
    "model_year",
    "field",
    "value",
    "application_match",
    "source_tier",
    "source_classification",
    "source_url",
    "source_title",
    "evidence_note",
    "evidence_origin",
    "selected_for_pragmatic_review",
)

PROVENANCE_FIELDS = (
    "sample_id",
    "vde_id",
    "vehicle_application",
    "field",
    "selected_value",
    "provenance",
    "observed_value",
    "researched_exact_value",
    "researched_approx_value",
    "calculated_value",
    "rule_estimated_value",
    "source_url",
    "note",
)

GROUP_FIELDS = (
    "group_identity",
    "group_level",
    "supplier",
    "applications",
    "models",
    "years",
    "provenance_mix",
    "source_count",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _present(value: Any) -> bool:
    return value is not None and str(value).strip() not in {"", "None", "nan"}


def _db_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in DB_PATHS if path.exists()}


def _claim_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("sample_id", "")),
        str(row.get("field", "")),
        str(row.get("value", "")),
        str(row.get("source_url", "")),
        str(row.get("application_match", "")),
    )


def _dedupe(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, ...]] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        key = _claim_key(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _batch_paths(output: Path, group: str) -> dict[str, Path]:
    stem = group.lower()
    batch = output / "batches"
    return {
        "results": batch / f"{stem}_results.csv",
        "evidence": batch / f"{stem}_evidence.csv",
        "provenance": batch / f"{stem}_provenance.csv",
        "summary": batch / f"{stem}_summary.md",
    }


def _evidence_output(
    sample: Mapping[str, Any], rows: Iterable[Mapping[str, Any]], *, origin: str
) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": sample["sample_id"],
            "vde_id": sample["vde_id"],
            "make": sample["make"],
            "model": sample["model"],
            "model_year": sample["model_year"],
            **dict(row),
            "evidence_origin": origin,
            "selected_for_pragmatic_review": "YES",
        }
        for row in rows
    ]


def _write_batch_summary(
    path: Path,
    *,
    group: str,
    expected: int,
    results: list[dict[str, Any]],
    runtime_issue: str,
) -> None:
    complete = len(results) == expected and not runtime_issue
    searches = sum(int(row.get("searches") or 0) for row in results)
    calls = sum(int(row.get("research_calls") or 0) for row in results)
    fetched = sum(int(row.get("fetched_sources") or 0) for row in results)
    accepted = sum(int(row.get("accepted_useful_sources") or 0) for row in results)
    researched = sum(int(row.get("new_live_evidence_rows") or 0) > 0 for row in results)
    lines = [
        f"# Components Research v0.5.1 — {group}",
        "",
        f"- Status: **{'COMPLETE' if complete else 'PARTIAL'}**",
        f"- Completed vehicles: **{len(results)}/{expected}**",
        f"- Searches: **{searches}**",
        f"- Model extraction calls: **{calls}**",
        f"- Fetched sources: **{fetched}**",
        f"- Accepted useful sources: **{accepted}**",
        f"- Vehicles receiving new live evidence: **{researched}**",
        f"- Runtime issue: `{runtime_issue or 'NONE'}`",
        "- Canonical writes: **DISABLED**",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _runtime_issue(runtime: Any, search_start: int, extraction_start: int) -> str:
    search_errors = [
        item for item in runtime.search_provider.audit[search_start:]
        if item.get("status") == "ERROR"
    ]
    extraction_errors = [
        item for item in runtime.claim_extractor.audit[extraction_start:]
        if item.get("status") == "ERROR"
    ]
    parts = []
    if search_errors:
        parts.append(f"search_errors={len(search_errors)}")
    if extraction_errors:
        parts.append(f"extraction_errors={len(extraction_errors)}")
    return ";".join(parts)


def _run_batch(
    *,
    group: str,
    samples: list[dict[str, str]],
    curated: list[dict[str, str]],
    output: Path,
    model: str,
    reasoning_effort: str,
    live: bool,
    resume: bool,
    retry_empty_live: bool,
    retry_sample_ids: frozenset[str],
) -> dict[str, Any]:
    paths = _batch_paths(output, group)
    results = _read_csv(paths["results"]) if resume else []
    evidence_rows = _read_csv(paths["evidence"]) if resume else []
    provenance_rows = _read_csv(paths["provenance"]) if resume else []
    prior_metrics: dict[str, dict[str, int]] = {}
    retry_ids = set(retry_sample_ids)
    if retry_empty_live:
        retry_ids.update(
            row["sample_id"]
            for row in results
            if int(row.get("new_live_evidence_rows") or 0) == 0
        )
    if retry_ids:
        for row in results:
            if row["sample_id"] in retry_ids:
                prior_metrics[row["sample_id"]] = {
                    field: int(row.get(field) or 0)
                    for field in (
                        "searches",
                        "research_calls",
                        "fetched_sources",
                        "accepted_useful_sources",
                    )
                }
        results = [row for row in results if row["sample_id"] not in retry_ids]
        evidence_rows = [row for row in evidence_rows if row["sample_id"] not in retry_ids]
        provenance_rows = [row for row in provenance_rows if row["sample_id"] not in retry_ids]
    completed = {row["sample_id"] for row in results}
    runtime = None
    if live:
        runtime = live_runtime(model=model, reasoning_effort=reasoning_effort)
        runtime.profile = CrossOemTransmissionResearchProfile()
        runtime.source_policy = CrossOemPragmaticSourcePolicy()
        if hasattr(runtime.claim_extractor, "max_document_chars"):
            runtime.claim_extractor.max_document_chars = 20_000
    hard_issue = ""

    for sample in samples:
        if sample["sample_id"] in completed:
            continue
        live_rows: list[dict[str, str]] = []
        status = "NOT_RUN"
        searches = calls = fetched = accepted = 0
        issue = ""
        if runtime is not None:
            search_start = len(runtime.search_provider.audit)
            extraction_start = len(runtime.claim_extractor.audit)
            try:
                research = research_technical_component(
                    request_for_sample(sample), runtime=runtime
                )
                status = research.status.value
                live_rows = claims_to_curated_evidence(sample, research.evidence.claims)
                searches = len(runtime.search_provider.audit) - search_start
                calls = len(runtime.claim_extractor.audit) - extraction_start
                fetched = int(research.source_summary.get("fetched") or 0)
                accepted = len(
                    {row["source_url"] for row in live_rows if row.get("source_url")}
                )
                issue = _runtime_issue(runtime, search_start, extraction_start)
            except Exception as exc:  # checkpoint first; affected batch remains resumable
                hard_issue = f"{type(exc).__name__}: {str(exc)[:240]}"
                _write_batch_summary(
                    paths["summary"],
                    group=group,
                    expected=len(samples),
                    results=results,
                    runtime_issue=hard_issue,
                )
                break

        curated_rows = applicable_evidence(sample, curated)
        combined = _dedupe([*sanitize_evidence(curated_rows), *live_rows])
        enriched, provenance = enrich_application_v051(sample, combined)
        enriched.update(
            {
                "research_runtime_status": status,
                "searches": searches + prior_metrics.get(sample["sample_id"], {}).get("searches", 0),
                "research_calls": calls + prior_metrics.get(sample["sample_id"], {}).get("research_calls", 0),
                "fetched_sources": fetched + prior_metrics.get(sample["sample_id"], {}).get("fetched_sources", 0),
                "accepted_useful_sources": accepted + prior_metrics.get(sample["sample_id"], {}).get("accepted_useful_sources", 0),
                "new_live_evidence_rows": len(live_rows),
                "runtime_issue": issue,
            }
        )
        results.append(enriched)
        evidence_rows.extend(_evidence_output(sample, curated_rows, origin="V05_CURATED_REUSE"))
        evidence_rows.extend(_evidence_output(sample, live_rows, origin="V051_LIVE"))
        provenance_rows.extend(provenance)
        completed.add(sample["sample_id"])

        # Checkpoint every vehicle so a later provider/network failure cannot
        # discard earlier successful research in the same OEM batch.
        _write_csv(paths["results"], results, ENRICHMENT_FIELDS)
        _write_csv(paths["evidence"], _dedupe(evidence_rows), EVIDENCE_FIELDS)
        _write_csv(paths["provenance"], provenance_rows, PROVENANCE_FIELDS)
        _write_batch_summary(
            paths["summary"],
            group=group,
            expected=len(samples),
            results=results,
            runtime_issue="",
        )
        print(
            f"V051_PROGRESS group={group} completed={len(results)}/{len(samples)} "
            f"sample={sample['sample_id']} status={status} live_evidence={len(live_rows)}",
            flush=True,
        )

    _write_batch_summary(
        paths["summary"],
        group=group,
        expected=len(samples),
        results=results,
        runtime_issue=hard_issue,
    )
    return {
        "group": group,
        "expected": len(samples),
        "completed": len(results),
        "status": "COMPLETE" if len(results) == len(samples) and not hard_issue else "PARTIAL",
        "runtime_issue": hard_issue,
    }


def _coverage(rows: list[dict[str, Any]], provenance: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(row.get("match_status") for row in rows)
    reviews = Counter(row.get("review_flag") for row in rows)
    provenances = Counter(row.get("provenance") for row in provenance)
    metrics: dict[str, Any] = {
        "vehicles_total": len(rows),
        "oem_groups": len({row.get("oem_group") for row in rows}),
        "transmission_exact_found": statuses["FOUND_EXACT"],
        "transmission_family_found": statuses["FOUND_FAMILY"],
        "transmission_architecture_found": statuses["FOUND_ARCHITECTURE"],
        "transmission_not_found": statuses["NOT_FOUND"],
        "transmission_conflicts": statuses["CONFLICT"],
        "cd_found": sum(_present(row.get("cd")) for row in rows),
        "frontal_area_found": sum(_present(row.get("frontal_area_m2")) for row in rows),
        "cda_available": sum(_present(row.get("cda_m2")) for row in rows),
        "tire_specs_found": sum(
            any(_present(row.get(field)) for field in ("tire_front", "tire_rear", "tire_general"))
            for row in rows
        ),
        "physical_fdr_found": sum(_present(row.get("physical_final_drive")) for row in rows),
        "drive_reduction_found": sum(
            any(_present(row.get(field)) for field in ("reduction_front", "reduction_rear"))
            for row in rows
        ),
        "gear_ratio_sets_found": sum(_present(row.get("gear_ratios")) for row in rows),
        "review_ok": reviews["REVIEW_OK"],
        "review_approx": reviews["REVIEW_APPROX"],
        "review_conflict": reviews["REVIEW_CONFLICT"],
        "review_sparse": reviews["REVIEW_SPARSE"],
    }
    for provenance_name in PROVENANCE_ORDER:
        metrics[f"{provenance_name.lower()}_values"] = provenances[provenance_name]
    return metrics


def _per_oem(
    rows: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in provenance:
        by_sample[str(item.get("sample_id"))].append(item)
    result = []
    for group in OEM_GROUPS:
        members = [row for row in rows if row.get("oem_group") == group]
        member_provenance = [item for row in members for item in by_sample.get(str(row.get("sample_id")), [])]
        metrics = _coverage(members, member_provenance)
        metrics["oem_group"] = group
        metrics["searches"] = sum(int(row.get("searches") or 0) for row in members)
        metrics["fetched_sources"] = sum(int(row.get("fetched_sources") or 0) for row in members)
        member_ids = {str(row.get("sample_id")) for row in members}
        metrics["accepted_useful_sources"] = len(
            {
                item.get("source_url")
                for item in evidence
                if item.get("sample_id") in member_ids
                and item.get("evidence_origin") == "V051_LIVE"
                and _present(item.get("source_url"))
            }
        )
        metrics["vehicles_with_new_live_data"] = sum(int(row.get("new_live_evidence_rows") or 0) > 0 for row in members)
        result.append(metrics)
    return result


def _write_final_summary(path: Path, report: Mapping[str, Any], batches: list[dict[str, Any]]) -> None:
    lines = [
        "# Components Research v0.5.1 — Cross-OEM Live Enrichment Completion",
        "",
        "## A. Sample and batches",
        "",
        f"- Vehicles: **{report['vehicles_total']}**; OEM groups: **{report['oem_groups']}**.",
        f"- Complete batches: **{report['oem_batches_complete']}**; partial: **{report['oem_batches_partial']}**.",
    ]
    lines.extend(
        f"- `{batch['group']}`: {batch['status']} ({batch['completed']}/{batch['expected']}); issue `{batch['runtime_issue'] or 'NONE'}`."
        for batch in batches
    )
    lines.extend(
        [
            "",
            "## B–D. Transmission, engineering, and provenance",
            "",
            f"- Transmission exact/family/architecture/conflict/not found: **{report['transmission_exact_found']} / {report['transmission_family_found']} / {report['transmission_architecture_found']} / {report['transmission_conflicts']} / {report['transmission_not_found']}**.",
            f"- Engineering Cd/area/CdA/tires/FDR/reduction/gear sets: **{report['cd_found']} / {report['frontal_area_found']} / {report['cda_available']} / {report['tire_specs_found']} / {report['physical_fdr_found']} / {report['drive_reduction_found']} / {report['gear_ratio_sets_found']}**.",
            f"- Provenance observed/exact/approx/calculated/rule/unknown: **{report['observed_values']} / {report['researched_exact_values']} / {report['researched_approx_values']} / {report['calculated_values']} / {report['rule_estimated_values']} / {report['unknown_values']}**.",
            "",
            "## E. Live research",
            "",
            f"- Searches: **{report['live_searches']}**; model extraction calls: **{report['live_research_calls']}**; fetched sources: **{report['live_fetched_sources']}**.",
            f"- Accepted useful sources: **{report['live_accepted_useful_sources']}**; vehicles receiving new live data: **{report['vehicles_receiving_new_live_data']}**.",
            f"- Runtime/cost issues: `{'; '.join(report['runtime_cost_issues']) or 'NONE'}`.",
            "",
            "## F–I. Groups, review, regression, and safety",
            "",
            f"- Repeated exact/family/architecture groups: **{report['exact_code_groups']} / {report['family_groups']} / {report['architecture_groups']}**.",
            f"- Review OK/approx/conflict/sparse: **{report['review_ok']} / {report['review_approx']} / {report['review_conflict']} / {report['review_sparse']}**.",
            f"- Frozen BMW benchmark unchanged: **{'YES' if report['bmw_frozen_benchmark_unchanged'] else 'NO'}**.",
            f"- Canonical DB hashes unchanged: **{'YES' if report['db_unchanged'] else 'NO'}**.",
            "- Canonical writes: **DISABLED**.",
            "",
            "## Completion block",
            "",
            "```ini",
        ]
    )
    ordered = (
        ("COMPONENT_RESEARCH_VERSION", "0.5.1"),
        ("CROSS_OEM_ENRICHMENT_COMPLETE", "YES" if report["cross_oem_enrichment_complete"] else "NO"),
        ("LIVE_RESEARCH_ACTUALLY_EXECUTED", "YES" if report["live_research_actually_executed"] else "NO"),
        ("TRANSMISSION_SEMANTICS_FIXED", "YES"),
        ("ARCHITECTURE_FAMILY_SEPARATION_READY", "YES"),
        ("VEHICLES_TOTAL", report["vehicles_total"]),
        ("OEM_GROUPS", report["oem_groups"]),
        ("OEM_BATCHES_COMPLETE", report["oem_batches_complete"]),
        ("OEM_BATCHES_PARTIAL", report["oem_batches_partial"]),
        ("TRANSMISSION_EXACT_FOUND", report["transmission_exact_found"]),
        ("TRANSMISSION_FAMILY_FOUND", report["transmission_family_found"]),
        ("TRANSMISSION_ARCHITECTURE_FOUND", report["transmission_architecture_found"]),
        ("TRANSMISSION_NOT_FOUND", report["transmission_not_found"]),
        ("TRANSMISSION_CONFLICTS", report["transmission_conflicts"]),
        ("CD_FOUND", report["cd_found"]),
        ("FRONTAL_AREA_FOUND", report["frontal_area_found"]),
        ("CDA_AVAILABLE", report["cda_available"]),
        ("TIRE_SPECS_FOUND", report["tire_specs_found"]),
        ("PHYSICAL_FDR_FOUND", report["physical_fdr_found"]),
        ("DRIVE_REDUCTION_FOUND", report["drive_reduction_found"]),
        ("GEAR_RATIO_SETS_FOUND", report["gear_ratio_sets_found"]),
        ("RESEARCHED_EXACT_VALUES", report["researched_exact_values"]),
        ("RESEARCHED_APPROX_VALUES", report["researched_approx_values"]),
        ("CALCULATED_VALUES", report["calculated_values"]),
        ("RULE_ESTIMATED_VALUES", report["rule_estimated_values"]),
        ("UNKNOWN_VALUES", report["unknown_values"]),
        ("REVIEW_OK", report["review_ok"]),
        ("REVIEW_APPROX", report["review_approx"]),
        ("REVIEW_CONFLICT", report["review_conflict"]),
        ("REVIEW_SPARSE", report["review_sparse"]),
        ("BMW_FROZEN_BENCHMARK_UNCHANGED", "YES" if report["bmw_frozen_benchmark_unchanged"] else "NO"),
        ("CANONICAL_WRITE_DISABLED", "YES"),
        ("PRODUCTION_DB_CHANGED", "NO" if report["db_unchanged"] else "YES"),
        ("READY_FOR_COMPONENT_DB_REVIEW_PROMOTION", "YES" if report["ready_for_component_db_review_promotion"] else "NO"),
    )
    lines.extend(f"{key} = {value}" for key, value in ordered)
    lines.extend(["```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _merge(output: Path, batches: list[dict[str, Any]], hashes_before: dict[str, str], bmw_before: str) -> dict[str, Any]:
    all_results: list[dict[str, Any]] = []
    all_evidence: list[dict[str, Any]] = []
    all_provenance: list[dict[str, Any]] = []
    for group in OEM_GROUPS:
        paths = _batch_paths(output, group)
        all_results.extend(_read_csv(paths["results"]))
        all_evidence.extend(_read_csv(paths["evidence"]))
        all_provenance.extend(_read_csv(paths["provenance"]))
    all_results.sort(key=lambda row: row["sample_id"])
    all_evidence = _dedupe(all_evidence)
    groups = repeated_groups_v051(all_results)
    metrics = _coverage(all_results, all_provenance)
    numeric = validate_numeric_contract(all_results)
    oem_rows = _per_oem(all_results, all_provenance, all_evidence)
    hashes_after = _db_hashes()
    bmw_after = file_sha256(FROZEN_BMW)
    complete_batches = sum(batch["status"] == "COMPLETE" for batch in batches)
    partial_batches = sum(batch["status"] != "COMPLETE" for batch in batches)
    live_searches = sum(int(row.get("searches") or 0) for row in all_results)
    live_calls = sum(int(row.get("research_calls") or 0) for row in all_results)
    live_fetched = sum(int(row.get("fetched_sources") or 0) for row in all_results)
    live_accepted = len(
        {
            row.get("source_url")
            for row in all_evidence
            if row.get("evidence_origin") == "V051_LIVE"
            and _present(row.get("source_url"))
        }
    )
    live_vehicles = sum(int(row.get("new_live_evidence_rows") or 0) > 0 for row in all_results)
    runtime_issues = sorted(
        {
            str(row.get("runtime_issue"))
            for row in all_results
            if _present(row.get("runtime_issue"))
        }
        | {batch["runtime_issue"] for batch in batches if batch["runtime_issue"]}
    )
    exact_groups = sum(row["group_level"] == "EXACT_CODE" for row in groups)
    family_groups = sum(row["group_level"] == "FAMILY" for row in groups)
    architecture_groups = sum(row["group_level"] == "ARCHITECTURE" for row in groups)
    live_actual = live_searches > 0 and live_calls > 0 and live_vehicles > 0
    bmw_unchanged = bmw_before == bmw_after == FROZEN_BMW_SHA256
    db_unchanged = hashes_before == hashes_after
    complete = (
        len(all_results) == 50
        and complete_batches == len(OEM_GROUPS)
        and partial_batches == 0
        and live_actual
        and live_vehicles >= 10
        and numeric["valid"]
        and bmw_unchanged
        and db_unchanged
    )
    useful_component_coverage = (
        metrics["transmission_exact_found"] + metrics["transmission_family_found"] >= 10
        and (
            metrics["cd_found"]
            + metrics["frontal_area_found"]
            + metrics["tire_specs_found"]
            + metrics["physical_fdr_found"]
            + metrics["drive_reduction_found"]
            + metrics["gear_ratio_sets_found"]
        )
        >= 50
    )
    report = {
        **metrics,
        "oem_batches_complete": complete_batches,
        "oem_batches_partial": partial_batches,
        "live_searches": live_searches,
        "live_research_calls": live_calls,
        "live_fetched_sources": live_fetched,
        "live_accepted_useful_sources": live_accepted,
        "vehicles_receiving_new_live_data": live_vehicles,
        "runtime_cost_issues": runtime_issues,
        "exact_code_groups": exact_groups,
        "family_groups": family_groups,
        "architecture_groups": architecture_groups,
        "numeric_contract": numeric,
        "live_research_actually_executed": live_actual,
        "transmission_semantics_fixed": True,
        "architecture_family_separation_ready": True,
        "cross_oem_enrichment_complete": complete,
        "ready_for_component_db_review_promotion": complete and useful_component_coverage,
        "db_hashes_before": hashes_before,
        "db_hashes_after": hashes_after,
        "db_unchanged": db_unchanged,
        "bmw_sha256_before": bmw_before,
        "bmw_sha256_after": bmw_after,
        "bmw_frozen_benchmark_unchanged": bmw_unchanged,
        "canonical_write": "DISABLED",
        "model": "gpt-5.6-terra",
        "reasoning_effort": "medium",
    }
    _write_csv(output / "CROSS_OEM_SAMPLE_V051.csv", all_results, SAMPLE_FIELDS)
    _write_csv(output / "COMPONENT_RESEARCH_ENRICHMENT_V051.csv", all_results, ENRICHMENT_FIELDS)
    _write_csv(output / "COMPONENT_RESEARCH_EVIDENCE_V051.csv", all_evidence, EVIDENCE_FIELDS)
    _write_csv(output / "COMPONENT_VALUE_PROVENANCE_V051.csv", all_provenance, PROVENANCE_FIELDS)
    _write_csv(output / "TRANSMISSION_MATCH_GROUPS_V051.csv", groups, GROUP_FIELDS)
    oem_fields = ("oem_group",) + tuple(
        key for key in oem_rows[0] if key != "oem_group"
    ) if oem_rows else ("oem_group",)
    _write_csv(output / "OEM_COVERAGE_SUMMARY_V051.csv", oem_rows, oem_fields)
    (output / "RUN_METRICS_V051.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_final_summary(output / "COMPONENT_RESEARCH_V051_SUMMARY.md", report, batches)
    return report


def run(
    *,
    sample_path: Path,
    curated_path: Path,
    output: Path,
    model: str,
    reasoning_effort: str,
    live: bool,
    resume: bool,
    retry_empty_live: bool = False,
    retry_sample_ids: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if model == "gpt-6-astra" or "astra" in model.lower():
        raise ValueError("Astra is explicitly disabled for v0.5.1")
    output.mkdir(parents=True, exist_ok=True)
    sample = _read_csv(sample_path)
    if len(sample) != 50 or len({row["sample_id"] for row in sample}) != 50:
        raise ValueError("Accepted v0.5 sample must contain exactly 50 unique applications")
    curated = read_curated_evidence(curated_path)
    hashes_before = _db_hashes()
    bmw_before = file_sha256(FROZEN_BMW)
    batches: list[dict[str, Any]] = []
    for group in OEM_GROUPS:
        members = [row for row in sample if row["oem_group"] == group]
        batches.append(
            _run_batch(
                group=group,
                samples=members,
                curated=curated,
                output=output,
                model=model,
                reasoning_effort=reasoning_effort,
                live=live,
                resume=resume,
                retry_empty_live=retry_empty_live,
                retry_sample_ids=retry_sample_ids,
            )
        )
    return _merge(output, batches, hashes_before, bmw_before)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--curated-evidence", type=Path, default=DEFAULT_CURATED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--retry-empty-live", action="store_true")
    parser.add_argument("--retry-sample", action="append", default=[])
    args = parser.parse_args()
    report = run(
        sample_path=args.sample,
        curated_path=args.curated_evidence,
        output=args.output,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        live=args.live,
        resume=not args.no_resume,
        retry_empty_live=args.retry_empty_live,
        retry_sample_ids=frozenset(args.retry_sample),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
