from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from ..contracts import ApplicationMatch


@dataclass(frozen=True)
class ApplicationMatchResult:
    match: ApplicationMatch
    compared_fields: tuple[str, ...]
    missing_fields: tuple[str, ...]
    mismatches: Mapping[str, tuple[Any, Any]]
    reason: str
    trusted_request_fields: tuple[str, ...] = ()
    conflicting_request_fields: tuple[str, ...] = ()


_FIELDS = (
    "make", "model", "model_year", "trim", "variant", "engine",
    "electrification", "drive_type", "transmission_type", "gears", "market",
)
_CRITICAL = {"make", "model", "model_year", "engine", "electrification", "drive_type"}
_UNKNOWN_VALUES = {"", "UNKNOWN", "UNSPECIFIED", "N A", "NA", "NONE", "NULL"}

_BODY_STYLES = (
    "GRAN COUPE",
    "GRAN TURISMO",
    "SPORTS ACTIVITY COUPE",
    "SPORTS ACTIVITY VEHICLE",
    "CONVERTIBLE",
    "HATCHBACK",
    "ROADSTER",
    "TOURING",
    "WAGON",
    "COUPE",
    "SEDAN",
)


@dataclass(frozen=True)
class VehicleApplicationIdentity:
    raw: str
    model_family: str
    designation: str
    body_style: str
    drive_variant: str
    powertrain_variant: str
    wheel_tire_variant: str
    performance_variant: str = ""
    market: str = ""

    @property
    def model_designation(self) -> str:
        return self.designation


def normalize_identity_value(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[^A-Z0-9]+", " ", str(value).upper()).strip()


def _is_known(value: Any) -> bool:
    return normalize_identity_value(value) not in _UNKNOWN_VALUES


def _normalize_field(field: str, value: Any) -> str:
    normalized = normalize_identity_value(value)
    if field == "drive_type":
        if normalized in {"RWD", "REAR WHEEL DRIVE", "2 WHEEL DRIVE REAR", "2WD REAR"}:
            return "RWD"
        if normalized in {"REAR WHEELS", "REAR WHEEL"}:
            return "RWD"
        if normalized in {"AWD", "ALL WHEEL DRIVE", "XDRIVE", "4 WHEEL DRIVE", "4WD"}:
            return "AWD"
        if normalized in {"FWD", "FRONT WHEEL DRIVE", "2 WHEEL DRIVE FRONT", "2WD FRONT"}:
            return "FWD"
    if field == "gears":
        word_counts = {
            "SINGLE": "1", "ONE": "1", "SIX": "6", "SEVEN": "7",
            "EIGHT": "8", "NINE": "9", "TEN": "10",
        }
        numeric = re.search(r"\b(\d{1,2})\s*(?:SPEED|GEAR)?\b", normalized)
        if numeric:
            return numeric.group(1)
        for word, count in word_counts.items():
            if re.search(rf"\b{word}\s+(?:SPEED|GEAR)", normalized):
                return count
    if field == "transmission_type":
        return normalize_transmission_type(value)
    return normalized


def normalize_transmission_type(value: Any) -> str:
    """Map source vocabulary to physical transmission families without rewriting raw evidence."""
    normalized = normalize_identity_value(value)
    if not normalized:
        return "UNKNOWN"
    if any(marker in normalized for marker in ("DUAL CLUTCH", "DOUBLE CLUTCH", " DCT", "DCT ")) or normalized == "DCT":
        return "DCT"
    if "CVT" in normalized or "CONTINUOUSLY VARIABLE" in normalized:
        return "CVT"
    if any(marker in normalized for marker in ("SINGLE SPEED", "1 SPEED", "ONE SPEED")):
        return "SINGLE_SPEED_EV"
    if normalized in {"MANUAL", "MANUAL TRANSMISSION", "MT"} or re.fullmatch(r"\d+ SPEED MANUAL", normalized):
        return "MANUAL"
    if any(
        marker in normalized
        for marker in (
            "SEMI AUTOMATIC",
            "AUTOMATIC",
            "STEPTRONIC",
            "SPORT AUTO",
            "SPORT AUTOMATIC",
            "TORQUE CONVERTER",
        )
    ) or normalized in {"AT", "AUTO"}:
        return "TORQUE_CONVERTER_AUTOMATIC"
    return "OTHER"


def parse_vehicle_application_identity(value: Any) -> VehicleApplicationIdentity:
    """Separate BMW model-family identity from body, drive, powertrain, and wheel qualifiers."""
    raw = normalize_identity_value(value)
    wheel_matches = re.findall(
        r"(?:\b\d{2,3}\s*(?:INCH|IN|WHEEL|WHEELS)\b|\b\d{3}\s*\d{2}\s*R\s*\d{2}\b)",
        raw,
    )
    without_wheels = raw
    for match in wheel_matches:
        without_wheels = without_wheels.replace(match, " ")
    without_wheels = re.sub(r"\bWHEELS?\b", " ", without_wheels)

    body_style = next((style for style in _BODY_STYLES if style in without_wheels), "")
    without_body = without_wheels
    for style in _BODY_STYLES:
        without_body = without_body.replace(style, " ")

    drive_match = re.search(
        r"\b((?:X|S|E)DRIVE(?:\s*\d+)?)\b", without_body
    )
    drive_token = re.sub(r"\s+", "", drive_match.group(1)) if drive_match else ""
    drive_variant = re.sub(r"\d+$", "", drive_token)
    core = re.sub(r"\b(?:X|S|E)DRIVE(?:\s*\d+)?\b", " ", without_body)
    core = re.sub(r"\s+", " ", core).strip()

    designation_match = re.search(r"\b(M?\d{3}[A-Z]*|I\d)\b", core)
    designation = designation_match.group(1) if designation_match else core
    if re.fullmatch(r"I\d", designation):
        model_family = designation
    elif re.fullmatch(r"M?([1-8])\d{2}[A-Z]*", designation):
        model_family = re.sub(r"^M", "", designation)[0] + " SERIES"
    elif re.fullmatch(r"[1-8] SERIES", core):
        model_family = core
        designation = ""
    else:
        model_family = core

    powertrain_variant = drive_token if re.search(r"\d", drive_token) else ""
    if designation:
        suffix_match = re.search(r"(?:\d{3}|I\d)([A-Z]+)$", designation)
        if suffix_match and not powertrain_variant:
            powertrain_variant = suffix_match.group(1)
    performance_variant = ";".join(
        marker
        for marker in ("M", "COMPETITION", "SPORT")
        if re.search(rf"\b{marker}\b", raw)
    )
    return VehicleApplicationIdentity(
        raw=raw,
        model_family=model_family,
        designation=designation,
        body_style=body_style,
        drive_variant=drive_variant,
        powertrain_variant=powertrain_variant,
        wheel_tire_variant=";".join(wheel_matches),
        performance_variant=performance_variant,
    )


def _model_relation(expected: Any, observed: Any) -> str:
    expected_id = parse_vehicle_application_identity(expected)
    observed_id = parse_vehicle_application_identity(observed)
    if expected_id.raw == observed_id.raw:
        return "EXACT"
    if expected_id.model_family != observed_id.model_family:
        return "MISMATCH"
    if (
        expected_id.designation
        and observed_id.designation
        and expected_id.designation != observed_id.designation
    ):
        return "MISMATCH"
    if (
        expected_id.drive_variant
        and observed_id.drive_variant
        and expected_id.drive_variant != observed_id.drive_variant
    ):
        return "MISMATCH"
    if (
        expected_id.body_style
        and observed_id.body_style
        and expected_id.body_style != observed_id.body_style
    ):
        return "MISMATCH"
    if (
        expected_id.powertrain_variant
        and observed_id.powertrain_variant
        and expected_id.powertrain_variant != observed_id.powertrain_variant
    ):
        return "MISMATCH"
    if expected_id.model_family == observed_id.model_family:
        return "FAMILY"
    return "MISMATCH"


def _source_value(field: str, context: Mapping[str, Any]) -> Any:
    observed = context.get(field)
    if _is_known(observed):
        return observed
    if field == "drive_type" and _is_known(context.get("model")):
        model_identity = parse_vehicle_application_identity(context["model"])
        if model_identity.drive_variant.startswith("XDRIVE"):
            return "AWD"
    return observed


def _normalized_transmission_for_context(value: Any, context: Mapping[str, Any]) -> str:
    family = normalize_transmission_type(value)
    if family == "TORQUE_CONVERTER_AUTOMATIC" and _normalize_field("gears", context.get("gears")) == "1":
        return "SINGLE_SPEED_EV"
    return family


def _year_matches(expected: Any, context: Mapping[str, Any]) -> bool | None:
    if expected in (None, ""):
        return None
    observed = context.get("model_year")
    if observed not in (None, ""):
        try:
            return int(float(str(expected))) == int(float(str(observed)))
        except ValueError:
            return normalize_identity_value(expected) == normalize_identity_value(observed)
    start = context.get("model_year_start", context.get("year_start"))
    end = context.get("model_year_end", context.get("year_end"))
    if start in (None, "") and end in (None, ""):
        return None
    try:
        year = int(float(str(expected)))
        lower = int(float(str(start))) if start not in (None, "") else year
        upper = int(float(str(end))) if end not in (None, "") else year
    except ValueError:
        return False
    return lower <= year <= upper


def match_application(
    known_fields: Mapping[str, Any], source_context: Mapping[str, Any]
) -> ApplicationMatchResult:
    """Match explicit application facts; absence lowers strength, disagreement rejects."""
    # Local import avoids coupling the request-audit contract to parsing internals.
    from .request_audit import audit_research_request

    request_audit = audit_research_request(known_fields)
    trusted_fields = request_audit.trusted_fields
    conflicting_fields = request_audit.conflicting_fields
    if not source_context:
        return ApplicationMatchResult(
            ApplicationMatch.UNKNOWN,
            (),
            (),
            {},
            "NO_SOURCE_APPLICATION_CONTEXT",
            trusted_fields,
            conflicting_fields,
        )
    compared: list[str] = []
    missing: list[str] = []
    mismatches: dict[str, tuple[Any, Any]] = {}
    broad_model_family = False
    requested = [
        field
        for field in _FIELDS
        if _is_known(known_fields.get(field)) and field not in conflicting_fields
    ]
    for field in requested:
        expected = known_fields.get(field)
        if field == "model_year":
            agrees = _year_matches(expected, source_context)
            observed = source_context.get(
                "model_year",
                (source_context.get("model_year_start"), source_context.get("model_year_end")),
            )
            if agrees is None:
                missing.append(field)
            elif agrees:
                compared.append(field)
            else:
                mismatches[field] = (expected, observed)
            continue
        observed = _source_value(field, source_context)
        if not _is_known(observed):
            missing.append(field)
            continue
        compared.append(field)
        if field == "model":
            relation = _model_relation(expected, observed)
            if relation == "FAMILY":
                broad_model_family = True
                missing.append("model_specificity")
                continue
            if relation == "MISMATCH":
                mismatches[field] = (expected, observed)
                continue
        elif field == "transmission_type":
            expected_family = _normalized_transmission_for_context(expected, known_fields)
            observed_family = _normalized_transmission_for_context(observed, source_context)
            if expected_family != observed_family:
                mismatches[field] = (expected, observed)
        elif _normalize_field(field, expected) != _normalize_field(field, observed):
            mismatches[field] = (expected, observed)
    if mismatches:
        return ApplicationMatchResult(
            ApplicationMatch.MISMATCH, tuple(compared), tuple(missing), mismatches,
            "EXPLICIT_MATERIAL_FIELD_MISMATCH",
            trusted_fields,
            conflicting_fields,
        )
    if not compared or "model" not in compared:
        return ApplicationMatchResult(
            ApplicationMatch.UNKNOWN, tuple(compared), tuple(missing), {},
            "INSUFFICIENT_APPLICATION_IDENTITY",
            trusted_fields,
            conflicting_fields,
        )
    missing_critical = _CRITICAL.intersection(missing)
    core = {"make", "model", "model_year"}.intersection(requested)
    if broad_model_family:
        match = ApplicationMatch.PARTIAL
        reason = "MODEL_FAMILY_RELEVANT_EXACT_APPLICATION_NOT_PROVEN"
    elif core.issubset(compared) and not missing_critical:
        if not missing:
            match = ApplicationMatch.EXACT
            reason = "ALL_REQUESTED_APPLICATION_FIELDS_EXPLICITLY_AGREE"
        else:
            match = ApplicationMatch.STRONG
            reason = "CORE_AND_CRITICAL_FIELDS_AGREE_NONCRITICAL_FIELD_ABSENT"
    elif {"make", "model"}.issubset(compared):
        match = ApplicationMatch.PARTIAL
        reason = "MODEL_FAMILY_RELEVANT_EXACT_APPLICATION_NOT_PROVEN"
    else:
        match = ApplicationMatch.UNKNOWN
        reason = "INSUFFICIENT_APPLICATION_IDENTITY"
    if conflicting_fields:
        reason = f"{reason};REQUEST_FIELDS_INTERNALLY_CONFLICTING"
    return ApplicationMatchResult(
        match,
        tuple(compared),
        tuple(missing),
        {},
        reason,
        trusted_fields,
        conflicting_fields,
    )


def application_match(
    known_fields: Mapping[str, Any], application_tags: Mapping[str, Any]
) -> ApplicationMatch:
    return match_application(known_fields, application_tags).match
