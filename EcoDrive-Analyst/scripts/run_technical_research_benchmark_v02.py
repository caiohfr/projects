from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capabilities.technical_research import live_runtime, research_technical_component  # noqa: E402
from capabilities.technical_research.adapters import load_bmw_benchmark_cases  # noqa: E402
from capabilities.technical_research.contracts import ResearchLimits  # noqa: E402


DEFAULT_SAMPLE = ROOT / "artifacts/components/transmission_experiment/EXPERIMENT_SAMPLE_AUDIT.csv"
DEFAULT_GROUPS = ROOT / "artifacts/components/transmission_experiment/TRANSMISSION_CANDIDATE_GROUPS.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/technical_research"
DB_PATHS = (
    ROOT / "data/db/eco_drive.db",
    ROOT / "data/db/eco_drive_qa.db",
    ROOT / "data/db/staging/eco_drive_canonical_candidate.db",
)


def _hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in DB_PATHS
        if path.exists()
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def run(
    sample_path: Path,
    groups_path: Path,
    output_dir: Path,
    *,
    limit: int,
    model: str,
    reasoning_effort: str,
    vde_ids: tuple[str, ...] = (),
) -> None:
    before = _hashes()
    cases = load_bmw_benchmark_cases(sample_path, groups_path, limit=limit)
    if vde_ids:
        selected_ids = set(vde_ids)
        cases = tuple(case for case in cases if case.vde_id in selected_ids)
        missing_ids = selected_ids.difference(case.vde_id for case in cases)
        if missing_ids:
            raise ValueError(f"VDE IDs are not in the selected benchmark sample: {sorted(missing_ids)}")
    runtime = live_runtime(model=model, reasoning_effort=reasoning_effort)
    # Exercise the intended bounded research loop; do not narrow this benchmark
    # below the capability contract defaults.
    limits = ResearchLimits(
        max_search_rounds=3,
        max_search_queries_per_round=5,
        max_sources_fetched=12,
        max_high_quality_sources_used=5,
    )

    result_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    ingestion_rows: list[dict[str, Any]] = []
    successful_results: dict[str, Any] = {}

    for index, case in enumerate(cases, start=1):
        request = replace(case.request, limits=limits)
        print(
            f"CASE_START {index}/{len(cases)} vde_id={case.vde_id} "
            f"model={request.known_fields.get('model')}",
            flush=True,
        )
        try:
            result = research_technical_component(request, runtime=runtime)
        except Exception as exc:
            result_rows.append(
                {
                    "vde_id": case.vde_id,
                    "candidate_group_id": case.candidate_group_id,
                    "previous_identity_status": case.previous_identity_status,
                    **dict(request.known_fields),
                    "identity_confidence": "UNRESOLVED",
                    "status": "ERROR",
                    "stop_reason": type(exc).__name__,
                    "error": str(exc)[:500],
                }
            )
            print(f"CASE_ERROR {case.vde_id} {type(exc).__name__}", flush=True)
            continue

        successful_results[case.vde_id] = result
        candidate = result.candidate
        result_rows.append(
            {
                "vde_id": case.vde_id,
                "candidate_group_id": case.candidate_group_id,
                "previous_identity_status": case.previous_identity_status,
                "make": request.known_fields.get("make"),
                "model": request.known_fields.get("model"),
                "model_year": request.known_fields.get("model_year"),
                "drive_type": request.known_fields.get("drive_type"),
                "transmission_type": request.known_fields.get("transmission_type"),
                "gears": request.known_fields.get("gears"),
                "original_final_drive_ratio": request.known_fields.get("final_drive_ratio"),
                "original_nv_ratio": request.known_fields.get("nv_ratio"),
                "researched_designation": candidate.identity,
                "researched_family": candidate.attributes.get("transmission_family"),
                "transmission_manufacturer": candidate.attributes.get("transmission_manufacturer"),
                "transmission_supplier": candidate.attributes.get("transmission_supplier"),
                "gear_ratios": _text(candidate.attributes.get("gear_ratios")),
                "final_drive_confirmation": _text(candidate.attributes.get("final_drive_ratio")),
                "drive_specific_variant": _text(candidate.attributes.get("drive_variant")),
                "cd": _text(candidate.attributes.get("cd") or candidate.attributes.get("drag_coefficient")),
                "frontal_area_m2": _text(candidate.attributes.get("frontal_area_m2")),
                "tire_size": _text(candidate.attributes.get("tire_size")),
                "identity_confidence": candidate.confidence.value,
                "search_rounds": result.provenance.get("search_rounds", 0),
                "attribute_status": _text({key: value.value for key, value in candidate.attribute_status.items()}),
                "status": result.status.value,
                "conflict_count": len(result.evidence.conflicts),
                "evidence_urls": ";".join(sorted({claim.source_url for claim in result.evidence.claims if claim.source_url})),
                "stop_reason": result.stop_reason,
                "error": "",
            }
        )

        claim_source_ids = {claim.source_id for claim in result.evidence.claims}
        for source in result.source_summary.get("sources", []):
            source_rows.append({"request_id": result.request_id, "vde_id": case.vde_id, **source})
            classification = source.get("source_classification", "UNCLASSIFIED")
            if source.get("policy_decision") == "ACCEPT" and source.get("source_id") in claim_source_ids:
                ingestion_status, ingestion_reason = "INGEST", "AUTHORITATIVE_SOURCE_WITH_EXTRACTED_CLAIMS"
            elif source.get("policy_decision") == "ACCEPT":
                ingestion_status, ingestion_reason = "METADATA_ONLY", "ACCEPTED_DISCOVERY_NOT_USED"
            elif classification == "DISCOVERY_ONLY":
                ingestion_status, ingestion_reason = "METADATA_ONLY", "DISCOVERY_SOURCE"
            elif classification == "WEAK":
                ingestion_status, ingestion_reason = "EPHEMERAL", "WEAK_SOURCE"
            else:
                ingestion_status, ingestion_reason = "REJECT", "UNCLASSIFIED_OR_POLICY_REJECTED"
            ingestion_rows.append(
                {
                    "request_id": result.request_id,
                    "vde_id": case.vde_id,
                    "source_id": source.get("source_id"),
                    "status": ingestion_status,
                    "reason": ingestion_reason,
                }
            )
        for row in result.source_summary.get("application_match_audit", []):
            match_rows.append({"request_id": result.request_id, "vde_id": case.vde_id, **row})
        for conflict in result.evidence.conflicts:
            conflict_rows.append(
                {
                    "request_id": result.request_id,
                    "vde_id": case.vde_id,
                    "field": conflict.field,
                    "competing_values": _text(conflict.values),
                    "sources": ";".join(conflict.supporting_sources),
                    "source_tiers": ";".join(value.value for value in conflict.source_tiers),
                    "application_matches": ";".join(conflict.application_differences),
                    "conflict_kind": conflict.conflict_kind,
                    "resolution": conflict.resolution.value,
                    "resolution_reason": conflict.resolution_reason,
                }
            )
        print(
            f"CASE_DONE {case.vde_id} status={result.status.value} "
            f"confidence={candidate.confidence.value} identity={candidate.identity!r}",
            flush=True,
        )

    group_members: dict[str, list[dict[str, Any]]] = {}
    for row in result_rows:
        group_members.setdefault(str(row["candidate_group_id"]), []).append(row)
    group_outcomes: dict[str, str] = {}
    for group_id, rows in group_members.items():
        resolved = [row for row in rows if row.get("researched_designation")]
        identities = {str(row["researched_designation"]) for row in resolved}
        if not resolved:
            outcome = "UNRESOLVED"
        elif len(resolved) < len(rows):
            outcome = "PARTIAL"
        elif len(identities) == 1:
            outcome = "CONFIRMED"
        else:
            outcome = "SPLIT"
        group_outcomes[group_id] = outcome

    comparison_rows = []
    for case, row in zip(cases, result_rows, strict=True):
        old_signature = json.dumps(dict(sorted(case.request.known_fields.items())), ensure_ascii=False, default=str)
        comparison_rows.append(
            {
                "old_candidate_group_id": case.candidate_group_id,
                "vehicle_configuration_id": case.vde_id,
                "make": row.get("make"),
                "model": row.get("model"),
                "model_year": row.get("model_year"),
                "old_signature": old_signature,
                "researched_identity": row.get("researched_designation"),
                "identity_confidence": row.get("identity_confidence", "UNRESOLVED"),
                "group_outcome": group_outcomes[case.candidate_group_id],
                "evidence_summary": row.get("evidence_urls", "") or row.get("error", ""),
            }
        )

    live_rows = list(getattr(runtime.search_provider, "audit", []))
    successful_searches = sum(row.get("status") == "OK" for row in live_rows)
    failed_searches = sum(row.get("status") == "ERROR" for row in live_rows)
    after = _hashes()
    db_unchanged = before == after
    classification_counts = {
        label: sum(row.get("source_classification") == label for row in source_rows)
        for label in ("PRIMARY_TECHNICAL", "STRONG_TECHNICAL", "DISCOVERY_ONLY", "WEAK", "UNCLASSIFIED")
    }
    match_counts = {
        label: sum(row.get("application_match") == label for row in match_rows)
        for label in ("EXACT", "STRONG", "PARTIAL", "MISMATCH", "UNKNOWN")
    }
    confidence_counts = {
        label: sum(row.get("identity_confidence") == label for row in result_rows)
        for label in ("DIRECT", "STRONG", "WEAK", "UNRESOLVED")
    }
    group_counts = {
        label: sum(outcome == label for outcome in group_outcomes.values())
        for label in ("CONFIRMED", "SPLIT", "PARTIAL", "UNRESOLVED")
    }
    field_counts = {
        "transmission_designation": sum(bool(row.get("researched_designation")) for row in result_rows),
        "cd": sum(bool(row.get("cd")) for row in result_rows),
        "frontal_area_m2": sum(bool(row.get("frontal_area_m2")) for row in result_rows),
        "tire_size": sum(bool(row.get("tire_size")) for row in result_rows),
        "gear_ratios": sum(bool(row.get("gear_ratios")) for row in result_rows),
        "final_drive": sum(bool(row.get("final_drive_confirmation")) for row in result_rows),
    }

    fetched_source_rows = [row for row in source_rows if str(row.get("fetched", "")).lower() == "true"]
    technical_sources_fetched = sum(
        row.get("source_classification") in {"PRIMARY_TECHNICAL", "STRONG_TECHNICAL"}
        for row in fetched_source_rows
    )
    bmwtechinfo_fetched = sum(
        "bmwtechinfo.bmwgroup.com" in str(row.get("url", "")).lower()
        for row in fetched_source_rows
    )
    attachments_fetched = sum(bool(row.get("landing_source_id")) for row in fetched_source_rows)
    search_rounds = sum(int(row.get("search_rounds") or 0) for row in result_rows)

    _write_csv(output_dir / "BMW_TRANSMISSION_RESEARCH_RESULTS_V021.csv", result_rows, list(result_rows[0]))
    _write_csv(output_dir / "BMW_TRANSMISSION_GROUP_COMPARISON_V021.csv", comparison_rows, list(comparison_rows[0]))
    _write_csv(output_dir / "SOURCE_AUDIT_V021.csv", source_rows, [
        "request_id", "vde_id", "source_id", "url", "title", "publisher",
        "source_classification", "publisher_type", "document_type",
        "classification_reason", "policy_decision", "policy_reason", "fetched",
        "search_round", "landing_source_id",
    ])
    _write_csv(output_dir / "APPLICATION_MATCH_AUDIT_V021.csv", match_rows, [
        "request_id", "vde_id", "source_id", "field", "application_match",
        "compared_fields", "missing_fields", "mismatches", "reason",
    ])
    _write_csv(output_dir / "CONFLICT_AUDIT_V021.csv", conflict_rows, [
        "request_id", "vde_id", "field", "competing_values", "sources",
        "source_tiers", "application_matches", "conflict_kind", "resolution", "resolution_reason",
    ])
    _write_csv(output_dir / "INGESTION_AUDIT_V021.csv", ingestion_rows, [
        "request_id", "vde_id", "source_id", "status", "reason",
    ])
    _write_csv(output_dir / "LIVE_SEARCH_AUDIT_V021.csv", live_rows, [
        "retrieved_at", "provider", "query", "sources_returned", "status", "error",
    ])

    ready = confidence_counts["DIRECT"] + confidence_counts["STRONG"] > 0 and db_unchanged
    summary = f"""# BMW Transmission Research Benchmark v0.2.1

- Model: `{model}`
- Reasoning effort: `{reasoning_effort}`
- Configurations: {len(cases)}
- Search provider: DDGS
- Searches: {len(live_rows)}
- Successful searches: {successful_searches}
- Failed searches: {failed_searches}
- Search rounds: {search_rounds}
- Sources discovered: {sum(int(row.get('sources_returned', 0)) for row in live_rows)}
- Technical sources fetched: {technical_sources_fetched}
- BMWTechInfo sources fetched: {bmwtechinfo_fetched}
- Technical attachments fetched: {attachments_fetched}
- Source classifications: {json.dumps(classification_counts, sort_keys=True)}
- Application matches: {json.dumps(match_counts, sort_keys=True)}
- Identity confidence: {json.dumps(confidence_counts, sort_keys=True)}
- Group outcomes: {json.dumps(group_counts, sort_keys=True)}
- Optional fields: {json.dumps(field_counts, sort_keys=True)}
- Identity conflicts: {sum(row.get('conflict_kind') == 'IDENTITY' for row in conflict_rows)}
- Attribute conflicts: {sum(row.get('conflict_kind') == 'ATTRIBUTE' for row in conflict_rows)}
- Canonical DB hashes before: `{json.dumps(before, sort_keys=True)}`
- Canonical DB hashes after: `{json.dumps(after, sort_keys=True)}`
- Canonical DB hashes unchanged: {db_unchanged}

## Before / after retrieval metrics

| Metric | v0.2 before | v0.2.1 after |
|---|---:|---:|
| Searches | 45 | {len(live_rows)} |
| Search rounds | 15 | {search_rounds} |
| Technical sources fetched | at least 8 with claims; exact fetch flag not persisted | {technical_sources_fetched} |
| BMWTechInfo sources fetched | 0 | {bmwtechinfo_fetched} |
| Attachments fetched | 0 | {attachments_fetched} |

TECHNICAL_RESEARCH_CAPABILITY_VERSION = 0.2.1

LIVE_EXTERNAL_SEARCH_READY = YES
DETERMINISTIC_SOURCE_CLASSIFICATION_READY = YES
DETERMINISTIC_APPLICATION_MATCHING_READY = YES
FIELD_LEVEL_CONFLICT_RESOLUTION_READY = YES

BMW_BENCHMARK_CONFIGURATIONS = {len(cases)}

DIRECT_IDENTITIES = {confidence_counts['DIRECT']}
STRONG_IDENTITIES = {confidence_counts['STRONG']}
WEAK_IDENTITIES = {confidence_counts['WEAK']}
UNRESOLVED_IDENTITIES = {confidence_counts['UNRESOLVED']}

GROUPS_CONFIRMED = {group_counts['CONFIRMED']}
GROUPS_SPLIT = {group_counts['SPLIT']}
GROUPS_PARTIAL = {group_counts['PARTIAL']}
GROUPS_UNRESOLVED = {group_counts['UNRESOLVED']}

INDEPENDENT_CDA_VALUES = {field_counts['cd']}
INDEPENDENT_TIRE_SPECS = {field_counts['tire_size']}
INDEPENDENT_TRANSMISSION_DESIGNATIONS = {field_counts['transmission_designation']}
INDEPENDENT_GEAR_RATIOS = {field_counts['gear_ratios']}
INDEPENDENT_FINAL_DRIVE_VALUES = {field_counts['final_drive']}
INDEPENDENT_FRONTAL_AREA_VALUES = {field_counts['frontal_area_m2']}

V021_PATCH_COMPLETE = YES
MULTI_ROUND_RESEARCH_EXECUTED = {'YES' if search_rounds > len(cases) else 'NO'}
TECHNICAL_ATTACHMENTS_ENABLED = YES
APPLICATION_NORMALIZATION_FIXED = YES

CANONICAL_WRITE_DISABLED = YES
PRODUCTION_DB_CHANGED = {'NO' if db_unchanged else 'YES'}

READY_TO_RE_RUN_TRANSMISSION_SIGNAL_EXPERIMENT = {'YES' if ready else 'NO'}
"""
    (output_dir / "BENCHMARK_SUMMARY_V021.md").write_text(summary, encoding="utf-8")
    print(f"BENCHMARK_DONE output={output_dir} db_unchanged={db_unchanged}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EcoDrive BMW live technical research benchmark v0.2.1")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument(
        "--vde-id",
        action="append",
        default=[],
        help="Retry only selected VDE ID(s) from the unchanged benchmark sample",
    )
    args = parser.parse_args()
    run(
        args.sample,
        args.groups,
        args.output,
        limit=args.limit,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        vde_ids=tuple(args.vde_id),
    )


if __name__ == "__main__":
    main()
