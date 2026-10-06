"""Pragmatic, review-only component enrichment for Technical Research v0.4.

This layer deliberately does not modify the canonical identity contract.  It
selects useful external facts with visible uncertainty and preserves rule
fallbacks separately from researched values.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from .contracts import ApplicationMatch, EvidenceClaim, SourceTier
from .core import contains_hardware_designation, normalize_identity_value


class PragmaticStatus(str, Enum):
    FOUND_EXACT = "FOUND_EXACT"
    FOUND_FAMILY = "FOUND_FAMILY"
    CONFLICT = "CONFLICT"
    NOT_FOUND = "NOT_FOUND"


class ValueProvenance(str, Enum):
    OBSERVED = "OBSERVED"
    RESEARCHED = "RESEARCHED"
    CALCULATED = "CALCULATED"
    RULE_ESTIMATED = "RULE_ESTIMATED"
    UNKNOWN = "UNKNOWN"


PROVENANCE_RANK = {
    ValueProvenance.UNKNOWN: 0,
    ValueProvenance.RULE_ESTIMATED: 1,
    ValueProvenance.CALCULATED: 2,
    ValueProvenance.RESEARCHED: 3,
    ValueProvenance.OBSERVED: 4,
}
USABLE_APPLICATION_MATCHES = {
    ApplicationMatch.EXACT,
    ApplicationMatch.STRONG,
    ApplicationMatch.PARTIAL,
}
USABLE_SOURCE_TIERS = {
    SourceTier.TIER_1_PRIMARY,
    SourceTier.TIER_2_STRONG_SECONDARY,
    SourceTier.TIER_3_DISCOVERY_ONLY,
}
FIELD_ALIASES = {
    "transmission_code": ("transmission_hardware_designation",),
    "transmission_family": ("transmission_family",),
    "supplier": ("transmission_supplier", "transmission_manufacturer"),
    "marketing_description": ("transmission_marketing_description",),
    "gears": ("gears",),
    "gear_ratios": ("gear_ratios",),
    "final_drive": ("final_drive_ratio",),
    "cd": ("drag_coefficient_cd",),
    "frontal_area_m2": ("frontal_area_m2",),
    "cda_m2": ("drag_area_cda_m2", "cda_m2"),
    "tire_front": ("tire_size_front",),
    "tire_rear": ("tire_size_rear",),
    "tire_general": ("tire_size_general",),
}


@dataclass(frozen=True)
class ResearchedSelection:
    value: Any = None
    claims: tuple[EvidenceClaim, ...] = ()
    conflict: bool = False
    competing_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValueSelection:
    selected_value: Any = None
    provenance: ValueProvenance = ValueProvenance.UNKNOWN
    researched_value: Any = None
    calculated_value: Any = None
    rule_estimated_value: Any = None
    source: str = ""
    note: str = ""


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip() != ""


def _normalized_claim_value(claim: EvidenceClaim) -> str:
    value = claim.normalized_value if _present(claim.normalized_value) else claim.value
    return normalize_identity_value(value)


def usable_claims(claims: Iterable[EvidenceClaim], fields: Sequence[str]) -> tuple[EvidenceClaim, ...]:
    return tuple(
        claim
        for claim in claims
        if claim.field in fields
        and claim.application_match in USABLE_APPLICATION_MATCHES
        and claim.source_tier in USABLE_SOURCE_TIERS
        and claim.extraction_confidence >= 0.5
    )


def select_researched_value(claims: Iterable[EvidenceClaim], fields: Sequence[str]) -> ResearchedSelection:
    candidates = usable_claims(claims, fields)
    by_value: dict[str, list[EvidenceClaim]] = {}
    for claim in candidates:
        by_value.setdefault(_normalized_claim_value(claim), []).append(claim)
    by_value.pop("", None)
    if not by_value:
        return ResearchedSelection()
    if len(by_value) > 1:
        return ResearchedSelection(
            claims=candidates,
            conflict=True,
            competing_values=tuple(sorted(str(group[0].value) for group in by_value.values())),
        )
    group = next(iter(by_value.values()))
    match_rank = {ApplicationMatch.EXACT: 4, ApplicationMatch.STRONG: 3, ApplicationMatch.PARTIAL: 2}
    tier_rank = {SourceTier.TIER_1_PRIMARY: 3, SourceTier.TIER_2_STRONG_SECONDARY: 2, SourceTier.TIER_3_DISCOVERY_ONLY: 1}
    representative = max(
        group,
        key=lambda claim: (
            match_rank[claim.application_match],
            tier_rank[claim.source_tier],
            claim.extraction_confidence,
        ),
    )
    return ResearchedSelection(representative.value, tuple(group), False, ())


def normalize_transmission_code(value: Any) -> str:
    text = normalize_identity_value(value).replace(" ", "")
    patterns = (
        r"GA\d+[A-Z]{2}\d+[A-Z]?",
        r"\d{1,2}HP\d{2,3}[A-Z]?",
        r"CVT\d+[A-Z]?",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return text if contains_hardware_designation(text) else ""


def transmission_family_from_code(value: Any) -> str:
    code = normalize_transmission_code(value)
    if not code:
        return ""
    match = re.search(r"(\d{1,2}HP\d{2,3})", code)
    if match:
        return match.group(1)
    return re.sub(r"[A-Z]$", "", code)


def pragmatic_transmission_status(claims: Iterable[EvidenceClaim]) -> tuple[PragmaticStatus, ResearchedSelection, ResearchedSelection]:
    claims = tuple(claims)
    code = select_researched_value(claims, FIELD_ALIASES["transmission_code"])
    family = select_researched_value(claims, FIELD_ALIASES["transmission_family"])
    if code.conflict or family.conflict:
        return PragmaticStatus.CONFLICT, code, family
    if _present(code.value) and normalize_transmission_code(code.value):
        return PragmaticStatus.FOUND_EXACT, code, family
    if _present(family.value):
        return PragmaticStatus.FOUND_FAMILY, code, family
    return PragmaticStatus.NOT_FOUND, code, family


def numeric_value(value: Any) -> float | None:
    if not _present(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", str(value))
    return float(match.group().replace(",", ".")) if match else None


def calculated_cda(cd: Any, frontal_area_m2: Any) -> float | None:
    cd_value, area_value = numeric_value(cd), numeric_value(frontal_area_m2)
    if cd_value is None or area_value is None:
        return None
    if cd_value <= 0 or area_value <= 0:
        return None
    return cd_value * area_value


def select_value(
    *,
    observed: Any = None,
    researched: ResearchedSelection | None = None,
    calculated: Any = None,
    rule_estimated: Any = None,
    calculation_note: str = "",
) -> ValueSelection:
    researched = researched or ResearchedSelection()
    source = ";".join(sorted({claim.source_url for claim in researched.claims if claim.source_url}))
    if _present(observed):
        return ValueSelection(observed, ValueProvenance.OBSERVED, researched.value, calculated, rule_estimated, source, "Observed structured application value retained by precedence.")
    if researched.conflict:
        return ValueSelection(None, ValueProvenance.UNKNOWN, ";".join(researched.competing_values), calculated, rule_estimated, source, "Credible researched values conflict; no value selected.")
    if _present(researched.value):
        return ValueSelection(researched.value, ValueProvenance.RESEARCHED, researched.value, calculated, rule_estimated, source, "External technical evidence selected.")
    if _present(calculated):
        return ValueSelection(calculated, ValueProvenance.CALCULATED, None, calculated, rule_estimated, "", calculation_note)
    if _present(rule_estimated):
        return ValueSelection(rule_estimated, ValueProvenance.RULE_ESTIMATED, None, None, rule_estimated, "", "Legacy public engineering rule fallback; not measured evidence.")
    return ValueSelection()


def flat_text(value: Any) -> str:
    if not _present(value):
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def evidence_note(claims: Iterable[EvidenceClaim]) -> str:
    notes = []
    for claim in claims:
        if claim.application_match == ApplicationMatch.MISMATCH:
            continue
        note = re.sub(r"\s+", " ", claim.evidence_text).strip()
        if note:
            notes.append(f"{claim.field}: {note[:240]}")
    return " | ".join(dict.fromkeys(notes))


def source_columns(claims: Iterable[EvidenceClaim]) -> dict[str, str | int]:
    useful = [claim for claim in claims if claim.application_match != ApplicationMatch.MISMATCH]
    return {
        "source_count": len({claim.source_id for claim in useful}),
        "source_urls": ";".join(dict.fromkeys(claim.source_url for claim in useful if claim.source_url)),
        "source_titles": ";".join(dict.fromkeys(claim.document_title for claim in useful if claim.document_title)),
        "source_types": ";".join(dict.fromkeys(claim.source_classification for claim in useful if claim.source_classification)),
        "evidence_note": evidence_note(useful),
    }


def transmission_match_identity(code: Any, family: Any, marketing: Any) -> tuple[str, str]:
    normalized_code = normalize_transmission_code(code)
    if normalized_code:
        return normalized_code, "SAME_EXACT_TRANSMISSION"
    normalized_family = normalize_identity_value(family)
    if normalized_family:
        return normalized_family, "SAME_FAMILY_OR_VARIANT"
    normalized_marketing = normalize_identity_value(marketing)
    if normalized_marketing:
        return normalized_marketing, "DESCRIPTION_MATCH_ONLY"
    return "UNKNOWN", "UNKNOWN"


def selections_for_application(
    known_fields: Mapping[str, Any],
    claims: Iterable[EvidenceClaim],
    rule_estimates: Mapping[str, Any] | None = None,
) -> tuple[dict[str, ValueSelection], PragmaticStatus]:
    claims = tuple(claims)
    rules = dict(rule_estimates or {})
    status, code, family = pragmatic_transmission_status(claims)
    researched = {name: select_researched_value(claims, aliases) for name, aliases in FIELD_ALIASES.items()}
    if not _present(researched["transmission_family"].value) and _present(code.value):
        derived_family = transmission_family_from_code(code.value)
        if derived_family:
            researched["transmission_family"] = ResearchedSelection(derived_family, code.claims)
    cda_calculated = calculated_cda(researched["cd"].value, researched["frontal_area_m2"].value)
    values = {
        "make": select_value(observed=known_fields.get("make")),
        "model": select_value(observed=known_fields.get("model")),
        "model_year": select_value(observed=known_fields.get("model_year")),
        "trim_variant": select_value(observed=known_fields.get("trim") or known_fields.get("variant")),
        "drive_type": select_value(observed=known_fields.get("drive_type")),
        "engine_powertrain": select_value(observed=known_fields.get("engine") or known_fields.get("electrification")),
        "transmission_code": select_value(researched=code),
        "transmission_family": select_value(researched=researched["transmission_family"], rule_estimated=rules.get("transmission_family")),
        "supplier": select_value(researched=researched["supplier"]),
        "marketing_description": select_value(researched=researched["marketing_description"]),
        "gears": select_value(observed=known_fields.get("gears"), researched=researched["gears"]),
        "gear_ratios": select_value(researched=researched["gear_ratios"]),
        "final_drive": select_value(observed=known_fields.get("final_drive_ratio"), researched=researched["final_drive"]),
        "cd": select_value(researched=researched["cd"]),
        "frontal_area_m2": select_value(researched=researched["frontal_area_m2"]),
        "cda_m2": select_value(researched=researched["cda_m2"], calculated=cda_calculated, rule_estimated=rules.get("cda_m2"), calculation_note="Cd × frontal_area_m2; CDA_METHOD=DIRECT_PRODUCT."),
        "tire_front": select_value(researched=researched["tire_front"]),
        "tire_rear": select_value(researched=researched["tire_rear"]),
        "tire_general": select_value(researched=researched["tire_general"]),
    }
    return values, status
