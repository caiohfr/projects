"""Deterministic semantic cleanup helpers for Components Research v0.4.1."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
import re
from typing import Any, Iterable, Mapping


class V041Provenance(str, Enum):
    OBSERVED = "OBSERVED"
    RESEARCHED_EXACT = "RESEARCHED_EXACT"
    RESEARCHED_APPROX = "RESEARCHED_APPROX"
    CALCULATED = "CALCULATED"
    RULE_ESTIMATED = "RULE_ESTIMATED"
    UNKNOWN = "UNKNOWN"


RESEARCHED_PROVENANCE = {
    V041Provenance.RESEARCHED_EXACT.value,
    V041Provenance.RESEARCHED_APPROX.value,
}
ROMAN_FORWARD_ORDER = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X")


def present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip() not in {"", "None", "nan"}


def numeric_engineering_value(value: Any) -> float | None:
    """Return the first engineering number, stripping common ratio/unit text."""
    if not present(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", str(value))
    return float(match.group(0).replace(",", ".")) if match else None


def provenance_from_application_match(application_match: str) -> str:
    if str(application_match).upper() == "EXACT":
        return V041Provenance.RESEARCHED_EXACT.value
    if str(application_match).upper() in {"STRONG", "PARTIAL"}:
        return V041Provenance.RESEARCHED_APPROX.value
    return V041Provenance.UNKNOWN.value


@dataclass(frozen=True)
class V041ValueSelection:
    selected_value: Any
    provenance: str
    observed_value: Any = None
    researched_exact_value: Any = None
    researched_approx_value: Any = None
    calculated_value: Any = None
    rule_estimated_value: Any = None


def select_v041_value(
    *, observed: Any = None, researched_exact: Any = None,
    researched_approx: Any = None, calculated: Any = None,
    rule_estimated: Any = None,
) -> V041ValueSelection:
    """Select by v0.4.1 precedence while retaining every lower-level value."""
    candidates = (
        (observed, V041Provenance.OBSERVED.value),
        (researched_exact, V041Provenance.RESEARCHED_EXACT.value),
        (researched_approx, V041Provenance.RESEARCHED_APPROX.value),
        (calculated, V041Provenance.CALCULATED.value),
        (rule_estimated, V041Provenance.RULE_ESTIMATED.value),
    )
    selected, provenance = None, V041Provenance.UNKNOWN.value
    for value, candidate_provenance in candidates:
        if present(value):
            selected, provenance = value, candidate_provenance
            break
    return V041ValueSelection(
        selected, provenance, observed, researched_exact, researched_approx,
        calculated, rule_estimated,
    )


def transmission_match_status(
    *, exact_code: Any = None, family: Any = None,
    marketing_description: Any = None, conflict: bool = False,
) -> str:
    if conflict:
        return "CONFLICT"
    if present(exact_code):
        return "FOUND_EXACT"
    if present(family) or derive_transmission_architecture(marketing_description):
        return "FOUND_FAMILY"
    return "NOT_FOUND"


def researched_provenance_for_evidence(
    evidence_rows: Iterable[Mapping[str, Any]], fields: Iterable[str]
) -> str:
    accepted = {str(field) for field in fields}
    ranks = {
        V041Provenance.UNKNOWN.value: 0,
        V041Provenance.RESEARCHED_APPROX.value: 1,
        V041Provenance.RESEARCHED_EXACT.value: 2,
    }
    result = V041Provenance.UNKNOWN.value
    for row in evidence_rows:
        if row.get("field") not in accepted:
            continue
        if row.get("selected_for_pragmatic_review", "YES") == "NO":
            continue
        candidate = provenance_from_application_match(str(row.get("application_match", "")))
        if ranks[candidate] > ranks[result]:
            result = candidate
    return result


def v041_provenance(
    old_provenance: str,
    evidence_rows: Iterable[Mapping[str, Any]],
    evidence_fields: Iterable[str],
) -> str:
    if old_provenance != "RESEARCHED":
        return old_provenance if old_provenance in {item.value for item in V041Provenance} else V041Provenance.UNKNOWN.value
    researched = researched_provenance_for_evidence(evidence_rows, evidence_fields)
    return researched if researched != V041Provenance.UNKNOWN.value else V041Provenance.RESEARCHED_APPROX.value


def derive_transmission_architecture(marketing_description: Any) -> str:
    normalized = re.sub(r"[^A-Z0-9]+", "_", str(marketing_description or "").upper()).strip("_")
    if "SINGLE_SPEED" in normalized and "FIXED_RATIO" in normalized:
        return "SINGLE_SPEED_EV"
    if "M_STEPTRONIC" in normalized and "DRIVELOGIC" in normalized:
        return "M_STEPTRONIC_WITH_DRIVELOGIC"
    return ""


def normalize_gear_ratios(value: Any) -> str:
    """Normalize conventional forward gear ratios to ordered numeric CSV text."""
    if not present(value):
        return ""
    text = str(value).strip()
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = {}
        numbers = [numeric_engineering_value(parsed.get(key)) for key in ROMAN_FORWARD_ORDER if key in parsed]
        return ";".join(format_number(number) for number in numbers if number is not None)
    labeled = re.findall(r"(?:^|;)\s*(VIII|VII|VI|IV|V|III|II|IX|X|I)\s*[:=]?\s*([-+]?\d+(?:[.,]\d+)?)", text, flags=re.IGNORECASE)
    if labeled:
        by_label = {label.upper(): numeric_engineering_value(number) for label, number in labeled}
        return ";".join(format_number(by_label[key]) for key in ROMAN_FORWARD_ORDER if by_label.get(key) is not None)
    parts = [part.strip() for part in text.split(";") if part.strip()]
    values = [numeric_engineering_value(part) for part in parts if not re.match(r"^R\b", part, flags=re.IGNORECASE)]
    return ";".join(format_number(number) for number in values if number is not None)


def format_number(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.12g}"


@dataclass(frozen=True)
class DrivelineSemantics:
    final_drive: float | None
    source_final_drive: float | None
    final_drive_semantics: str
    reduction_front: float | None = None
    reduction_rear: float | None = None

    @property
    def physical_fdr_found(self) -> bool:
        return self.final_drive_semantics == "PHYSICAL_FINAL_DRIVE" and self.final_drive is not None

    @property
    def drive_reduction_found(self) -> bool:
        return self.reduction_front is not None or self.reduction_rear is not None


def driveline_semantics(
    *,
    source_final_drive: Any,
    gear_ratios: Any,
    architecture: str,
    drive_type: Any,
) -> DrivelineSemantics:
    source = numeric_engineering_value(source_final_drive)
    if architecture == "SINGLE_SPEED_EV":
        ratio_text = str(gear_ratios or "")
        front_match = re.search(r"front\s*([-+]?\d+(?:[.,]\d+)?)", ratio_text, flags=re.IGNORECASE)
        rear_match = re.search(r"rear\s*([-+]?\d+(?:[.,]\d+)?)", ratio_text, flags=re.IGNORECASE)
        front = numeric_engineering_value(front_match.group(1)) if front_match else None
        rear = numeric_engineering_value(rear_match.group(1)) if rear_match else None
        if front is None and rear is None:
            single = numeric_engineering_value(ratio_text)
            if single is not None:
                if "FRONT" in str(drive_type).upper():
                    front = single
                else:
                    rear = single
        return DrivelineSemantics(
            final_drive=None,
            source_final_drive=source,
            final_drive_semantics="SOURCE_PLACEHOLDER" if source is not None else "FIXED_DRIVE_REDUCTION",
            reduction_front=front,
            reduction_rear=rear,
        )
    if source is not None:
        return DrivelineSemantics(source, source, "PHYSICAL_FINAL_DRIVE")
    return DrivelineSemantics(None, None, "UNKNOWN")


def calculated_cda(cd: Any, frontal_area_m2: Any) -> float | None:
    cd_value = numeric_engineering_value(cd)
    area_value = numeric_engineering_value(frontal_area_m2)
    if cd_value is None or area_value is None or cd_value <= 0 or area_value <= 0:
        return None
    return float(f"{cd_value * area_value:.12g}")
