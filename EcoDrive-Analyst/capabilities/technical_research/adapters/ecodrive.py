from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..contracts import TechnicalResearchRequest, TechnicalResearchResult
from ..core import normalize_identity_value, parse_vehicle_application_identity
from ..profiles import TRANSMISSION_TARGET_FIELDS


@dataclass(frozen=True)
class EcoDriveBenchmarkCase:
    vde_id: str
    candidate_group_id: str
    previous_identity_status: str
    request: TechnicalResearchRequest
    old_group_signature: str = ""
    independent_application_id: str = ""
    source_identity: str = ""


def load_bmw_benchmark_cases(
    sample_audit_path: str | Path,
    candidate_groups_path: str | Path,
    *,
    limit: int = 15,
) -> tuple[EcoDriveBenchmarkCase, ...]:
    """Select a deterministic mixed BMW sample from existing public artifacts."""
    if limit < 10 or limit > 20:
        raise ValueError("BMW benchmark limit must be between 10 and 20")
    with Path(candidate_groups_path).open("r", encoding="utf-8-sig", newline="") as handle:
        group_rows = {row["candidate_group_id"]: row for row in csv.DictReader(handle)}
    with Path(sample_audit_path).open("r", encoding="utf-8-sig", newline="") as handle:
        roots = [
            row
            for row in csv.DictReader(handle)
            if row.get("make", "").strip().upper() == "BMW"
            and row.get("carryover_status") == "ROOT"
        ]
    by_group: dict[str, list[dict[str, str]]] = {}
    for row in roots:
        by_group.setdefault(row["candidate_group_id"], []).append(row)
    for rows in by_group.values():
        rows.sort(key=lambda row: (int(row["model_year"]), row["model"], int(row["vde_id"])))

    family_groups = sorted(
        (item for item in by_group.items() if item[1][0]["identity_status"] == "FAMILY_ONLY"),
        key=lambda item: (-len(item[1]), item[0]),
    )
    strict_groups = sorted(
        (item for item in by_group.items() if item[1][0]["identity_status"] == "STRICT_CANDIDATE"),
        key=lambda item: (-len(item[1]), item[0]),
    )
    selected: list[dict[str, str]] = []
    # Repeated candidates expose possible splits; family-only cases expose ambiguity.
    if strict_groups:
        selected.extend(strict_groups[0][1][:3])
    if family_groups:
        selected.extend(family_groups[0][1][:2])
    for _, rows in strict_groups[1:8]:
        selected.append(rows[0])
    for _, rows in family_groups[1:4]:
        selected.append(rows[0])
    seen_ids = {row["vde_id"] for row in selected}
    for _, rows in strict_groups:
        for row in rows:
            if len(selected) >= limit:
                break
            if row["vde_id"] not in seen_ids:
                selected.append(row)
                seen_ids.add(row["vde_id"])
        if len(selected) >= limit:
            break

    cases: list[EcoDriveBenchmarkCase] = []
    for row in selected[:limit]:
        group = group_rows.get(row["candidate_group_id"], {})
        known_fields: dict[str, Any] = {
            "make": row["make"],
            "model": row["model"],
            "model_year": int(row["model_year"]),
        }
        field_map = {
            "transmission_type": "transmission_type",
            "gears": "gears",
            "drive_type": "drive_type",
            "final_drive_ratio": "axle_ratio_summary",
            "nv_ratio": "nv_ratio_summary",
            "electrification": "electrification",
        }
        for request_field, artifact_field in field_map.items():
            value = group.get(artifact_field)
            if value not in (None, ""):
                known_fields[request_field] = value
        request = TechnicalResearchRequest(
            domain="TRANSMISSION",
            known_fields=known_fields,
            target_fields=TRANSMISSION_TARGET_FIELDS,
            request_id=f"bmw-benchmark-{row['vde_id']}",
        )
        parsed_model = parse_vehicle_application_identity(row["model"])
        independent_application_id = "|".join(
            (
                normalize_identity_value(row["make"]),
                str(row["model_year"]),
                parsed_model.model_family,
                parsed_model.designation,
                parsed_model.body_style,
                parsed_model.drive_variant,
                parsed_model.powertrain_variant,
                normalize_identity_value(known_fields.get("drive_type")),
                normalize_identity_value(known_fields.get("electrification")),
            )
        )
        cases.append(
            EcoDriveBenchmarkCase(
                vde_id=row["vde_id"],
                candidate_group_id=row["candidate_group_id"],
                previous_identity_status=row["identity_status"],
                request=request,
                old_group_signature=group.get("grouping_signature", ""),
                independent_application_id=independent_application_id,
                source_identity=(
                    f"vde_id={row['vde_id']}|root_vde_id={row.get('root_vde_id', '')}"
                    f"|test_numbers={row.get('test_numbers', '')}"
                ),
            )
        )
    return tuple(cases)


def result_for_ecodrive_review(
    case: EcoDriveBenchmarkCase, result: TechnicalResearchResult
) -> dict[str, Any]:
    """Review-only transformation; deliberately has no persistence operation."""
    return {
        "vde_id": case.vde_id,
        "candidate_group_id": case.candidate_group_id,
        "previous_identity_status": case.previous_identity_status,
        "make": case.request.known_fields.get("make"),
        "model": case.request.known_fields.get("model"),
        "model_year": case.request.known_fields.get("model_year"),
        "transmission_hardware_designation": result.candidate.identity,
        "researched_family": result.candidate.attributes.get("transmission_family"),
        "transmission_marketing_description": result.candidate.attributes.get(
            "transmission_marketing_description"
        ),
        "final_drive_confirmation": result.candidate.attributes.get("final_drive_ratio"),
        "gear_ratio_evidence": result.candidate.attributes.get("gear_ratios"),
        "identity_confidence": result.candidate.confidence.value,
        "status": result.status.value,
        "conflict_count": len(result.evidence.conflicts),
        "evidence_source_ids": ";".join(
            sorted({claim.source_id for claim in result.evidence.claims})
        ),
        "stop_reason": result.stop_reason,
    }
