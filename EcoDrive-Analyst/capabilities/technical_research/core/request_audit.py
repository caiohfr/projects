from __future__ import annotations

from typing import Any, Mapping

from ..contracts import (
    RequestConsistencyStatus,
    RequestFieldConflict,
    RequestIdentityAudit,
    TechnicalResearchRequest,
)
from .matching import normalize_identity_value, parse_vehicle_application_identity


_REQUIRED_IDENTITY_FIELDS = ("make", "model", "model_year")
_UNKNOWN_VALUES = {"", "UNKNOWN", "UNSPECIFIED", "N A", "NA", "NONE", "NULL"}


def audit_research_request(
    request: TechnicalResearchRequest | Mapping[str, Any],
) -> RequestIdentityAudit:
    fields = request.known_fields if isinstance(request, TechnicalResearchRequest) else request
    missing = tuple(
        field
        for field in _REQUIRED_IDENTITY_FIELDS
        if not _is_meaningful(fields.get(field))
    )
    conflicts: list[RequestFieldConflict] = []
    identity = parse_vehicle_application_identity(fields.get("model"))
    drive = _drive_family(fields.get("drive_type"))
    model_drive = _model_drive_family(identity.drive_variant)
    if drive and model_drive and drive != model_drive:
        conflicts.append(
            RequestFieldConflict(
                field_a="model",
                value_a=fields.get("model"),
                field_b="drive_type",
                value_b=fields.get("drive_type"),
                conflict_type="MODEL_DRIVE_CONTRADICTION",
            )
        )

    electrification = normalize_identity_value(fields.get("electrification"))
    model_electrification = _model_electrification(identity.designation)
    if (
        _is_meaningful(electrification)
        and model_electrification
        and not _electrification_agrees(electrification, model_electrification)
    ):
        conflicts.append(
            RequestFieldConflict(
                field_a="model",
                value_a=fields.get("model"),
                field_b="electrification",
                value_b=fields.get("electrification"),
                conflict_type="MODEL_ELECTRIFICATION_CONTRADICTION",
            )
        )

    # The explicit structured field is suspect; the model text remains preserved
    # and usable as independent request identity evidence.
    conflicting_fields = tuple(sorted({conflict.field_b for conflict in conflicts}))
    trusted_fields = tuple(
        sorted(
            field
            for field, value in fields.items()
            if _is_meaningful(value) and field not in conflicting_fields
        )
    )
    if conflicts:
        status = RequestConsistencyStatus.CONFLICTING_SOURCE_FIELDS
    elif missing:
        status = RequestConsistencyStatus.INCOMPLETE
    else:
        status = RequestConsistencyStatus.CONSISTENT
    return RequestIdentityAudit(
        status=status,
        conflicts=tuple(conflicts),
        missing_fields=missing,
        trusted_fields=trusted_fields,
        conflicting_fields=conflicting_fields,
    )


def _drive_family(value: Any) -> str:
    normalized = normalize_identity_value(value)
    if normalized in {"AWD", "ALL WHEEL DRIVE", "4WD", "4 WHEEL DRIVE", "XDRIVE"}:
        return "AWD"
    if normalized in {
        "RWD",
        "REAR WHEEL DRIVE",
        "2 WHEEL DRIVE REAR",
        "2WD REAR",
        "EDRIVE",
        "SDRIVE",
    }:
        return "RWD"
    if normalized in {"FWD", "FRONT WHEEL DRIVE", "2 WHEEL DRIVE FRONT", "2WD FRONT"}:
        return "FWD"
    return ""


def _is_meaningful(value: Any) -> bool:
    return normalize_identity_value(value) not in _UNKNOWN_VALUES


def _model_drive_family(value: Any) -> str:
    normalized = normalize_identity_value(value).replace(" ", "")
    if normalized.startswith("XDRIVE"):
        return "AWD"
    if normalized.startswith(("EDRIVE", "SDRIVE")):
        return "RWD"
    return ""


def _model_electrification(designation: str) -> str:
    normalized = normalize_identity_value(designation).replace(" ", "")
    if normalized.startswith("I") and len(normalized) == 2 and normalized[1].isdigit():
        return "BEV"
    if normalized.endswith("E") and normalized[:-1].isdigit():
        return "PHEV"
    return ""


def _electrification_agrees(value: str, expected: str) -> bool:
    aliases = {
        "BEV": {"BEV", "EV", "ELECTRIC", "BATTERY ELECTRIC"},
        "PHEV": {"PHEV", "PLUG IN HYBRID", "PLUG IN HYBRID ELECTRIC"},
    }
    return value in aliases[expected]
