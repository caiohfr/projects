"""Focused v0.5.1 transmission semantics and live-evidence adapters.

This module deliberately builds on the accepted v0.5 sample and enrichment
contract.  It does not own canonical storage and never writes to SQLite.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import re
from typing import Any, Iterable, Mapping, Sequence

from .contracts import (
    ApplicationMatch,
    EvidenceClaim,
    ResearchStatus,
    SourceDecision,
    SourcePolicyDecision,
    SourceTier,
    TechnicalResearchRequest,
)
from .core import SourcePolicy
from .core.validation import public_request_fields
from .cross_oem_v05 import (
    enrich_application,
    grouping_identity,
    present,
    transmission_architecture,
)


ARCHITECTURE_VALUES = frozenset(
    {
        "TORQUE_CONVERTER_AUTOMATIC",
        "DCT",
        "CVT",
        "MANUAL",
        "AUTOMATED_MANUAL",
        "SINGLE_SPEED_EV",
        "MULTI_SPEED_EV",
        "OTHER",
        "UNKNOWN",
    }
)

LIVE_TARGET_FIELDS = (
    "transmission_hardware_designation",
    "transmission_family",
    "transmission_manufacturer",
    "transmission_supplier",
    "transmission_marketing_description",
    "transmission_type_normalized",
    "gears",
    "gear_ratios",
    "final_drive_ratio",
    "reduction_front",
    "reduction_rear",
    "drag_coefficient_cd",
    "frontal_area_m2",
    "drag_area_cda_m2",
    "tire_size_front",
    "tire_size_rear",
    "tire_size_general",
)

CLAIM_FIELD_MAP = {
    "transmission_hardware_designation": "transmission_code",
    "transmission_family": "transmission_family",
    "transmission_manufacturer": "transmission_supplier",
    "transmission_supplier": "transmission_supplier",
    "transmission_marketing_description": "transmission_marketing_description",
    "transmission_type_normalized": "transmission_architecture",
    "gear_ratios": "gear_ratios",
    "final_drive_ratio": "physical_final_drive",
    "reduction_front": "reduction_front",
    "reduction_rear": "reduction_rear",
    "drag_coefficient_cd": "cd",
    "frontal_area_m2": "frontal_area_m2",
    "drag_area_cda_m2": "cda_m2",
    "tire_size_front": "tire_front",
    "tire_size_rear": "tire_rear",
    "tire_size_general": "tire_general",
}


def normalize_architecture(value: Any) -> str:
    """Normalize architecture with specific patterns before broad substrings."""
    text = re.sub(r"[\s_-]+", " ", str(value or "").strip().upper())
    if not text:
        return "UNKNOWN"
    if "DUAL CLUTCH" in text or re.search(r"\bDCT\b", text):
        return "DCT"
    if "CONTINUOUSLY VARIABLE" in text or re.search(r"\bCVT\b", text):
        return "CVT"
    if "AUTOMATED MANUAL" in text or re.search(r"\bAMT\b", text):
        return "AUTOMATED_MANUAL"
    if any(marker in text for marker in ("SEMI AUTOMATIC", "SEQUENTIAL")):
        return "OTHER"
    if "SINGLE SPEED" in text:
        return "SINGLE_SPEED_EV"
    if "MULTI SPEED" in text and any(marker in text for marker in ("EV", "ELECTRIC")):
        return "MULTI_SPEED_EV"
    if "MANUAL" in text:
        return "MANUAL"
    if "AUTOMATIC" in text:
        return "TORQUE_CONVERTER_AUTOMATIC"
    canonical = text.replace(" ", "_")
    return canonical if canonical in ARCHITECTURE_VALUES else "OTHER"


def is_architecture_label(value: Any) -> bool:
    text = re.sub(r"[\s-]+", "_", str(value or "").strip().upper())
    return text in ARCHITECTURE_VALUES


@dataclass(frozen=True)
class CrossOemTransmissionResearchProfile:
    domain: str = "TRANSMISSION"
    identity_field: str = "transmission_hardware_designation"

    def build_queries(self, request: TechnicalResearchRequest, *, round_number: int) -> Sequence[str]:
        fields = public_request_fields(request)
        vehicle = " ".join(
            str(fields.get(name, ""))
            for name in ("model_year", "make", "model", "trim")
            if fields.get(name) not in (None, "")
        )
        make_model = " ".join(
            str(fields.get(name, ""))
            for name in ("make", "model")
            if fields.get(name) not in (None, "")
        )
        if round_number <= 1:
            queries = (
                f'"{vehicle}" transmission code gear ratios final drive',
                f'"{vehicle}" technical specifications transmission PDF',
                f'"{vehicle}" drag coefficient frontal area tire size',
                f'"{make_model}" transmission family supplier technical',
            )
        else:
            queries = (
                f'"{vehicle}" specifications PDF gearbox',
                f'"{make_model}" service manual transmission designation',
                f'"{vehicle}" Cd frontal area technical data',
                f'"{vehicle}" tire size final drive ratio',
            )
        return tuple(dict.fromkeys(query for query in queries if query.strip()))

    def evidence_is_sufficient(self, status: str) -> bool:
        return status in {
            ResearchStatus.SUPPORTED.value,
            ResearchStatus.CONFLICTING_EVIDENCE.value,
        }


class CrossOemPragmaticSourcePolicy(SourcePolicy):
    """Accept explicit claims from official OEM HTML, not only technical PDFs."""

    def evaluate(self, source: Any) -> SourcePolicyDecision:
        decision = super().evaluate(source)
        if (
            decision.decision == SourceDecision.DISCOVERY_ONLY
            and source.metadata.get("publisher_type") == "OEM"
        ):
            upgraded = replace(
                source,
                tier=SourceTier.TIER_1_PRIMARY,
                metadata={
                    **source.metadata,
                    "source_classification": "PRIMARY_OFFICIAL_OEM",
                    "classification_reason": "V051_OFFICIAL_OEM_PAGE_PRAGMATIC_ACCEPTANCE",
                },
            )
            return SourcePolicyDecision(
                upgraded,
                SourceDecision.ACCEPT,
                "V051_OFFICIAL_OEM_PAGE",
            )
        return decision


def request_for_sample(row: Mapping[str, Any], *, force_refresh: bool = True) -> TechnicalResearchRequest:
    from .contracts import ResearchLimits

    known = {
        "make": row.get("make"),
        "model": row.get("model"),
        "model_year": row.get("model_year"),
        "electrification": row.get("electrification"),
        "drive_type": row.get("drive_type"),
        "transmission_type": row.get("transmission_type"),
        "gears": row.get("gears"),
        "final_drive_ratio": row.get("source_final_drive"),
        "nv_ratio": row.get("nv_ratio"),
        "category": row.get("category"),
    }
    return TechnicalResearchRequest(
        domain="TRANSMISSION",
        known_fields={key: value for key, value in known.items() if present(value)},
        target_fields=LIVE_TARGET_FIELDS,
        request_id=f"v051-{row['sample_id']}",
        force_refresh=force_refresh,
        limits=ResearchLimits(
            max_search_rounds=1,
            max_search_queries_per_round=4,
            max_sources_fetched=4,
            max_high_quality_sources_used=2,
        ),
    )


def _normalized_application_text(value: Any) -> str:
    text = re.sub(r"\b(?:AWD|FWD|RWD|4WD|4X4|4MATIC|2WD)\b", "", str(value or "").upper())
    return re.sub(r"[^A-Z0-9]+", " ", text).strip()


def _year_compatible(expected: Any, context: Mapping[str, Any]) -> bool:
    try:
        year = int(float(str(expected)))
        if present(context.get("model_year")):
            return year == int(float(str(context["model_year"])))
        lower = int(float(str(context.get("model_year_start", context.get("year_start", year)))))
        upper = int(float(str(context.get("model_year_end", context.get("year_end", year)))))
        return lower <= year <= upper
    except (TypeError, ValueError):
        return False


def _compatible_broader_claim(row: Mapping[str, Any], claim: EvidenceClaim) -> bool:
    """Allow only field-safe same-model evidence after a broad variant mismatch."""
    context = claim.application_context
    if not context:
        return False
    context_make = _normalized_application_text(context.get("make"))
    row_make = _normalized_application_text(row.get("make"))
    if not context_make or not row_make or not (
        context_make in row_make or row_make in context_make
    ):
        return False
    context_model = _normalized_application_text(context.get("model"))
    row_model = _normalized_application_text(row.get("model"))
    for make_token in set(context_make.split()) | set(row_make.split()):
        context_model = re.sub(rf"\b{re.escape(make_token)}\b", "", context_model).strip()
        row_model = re.sub(rf"\b{re.escape(make_token)}\b", "", row_model).strip()
    if not context_model or not row_model or context_model != row_model:
        return False
    if not _year_compatible(row.get("model_year"), context):
        return False
    source_electrification = str(context.get("electrification") or "").upper()
    if source_electrification in {"NOT STATED", "UNKNOWN", "UNSPECIFIED", "N/A", "NA", "NONE"}:
        source_electrification = ""
    if source_electrification and source_electrification != str(row.get("electrification") or "").upper():
        return False
    # Drive mismatch is compatible for architecture/aero identity, but not for
    # axle ratios, reductions, or fitment-sensitive tire claims.
    safe_fields = {
        "transmission_hardware_designation",
        "transmission_family",
        "transmission_manufacturer",
        "transmission_supplier",
        "transmission_marketing_description",
        "transmission_type_normalized",
        "drag_coefficient_cd",
        "frontal_area_m2",
        "drag_area_cda_m2",
    }
    if claim.field not in safe_fields and claim.field != "gear_ratios":
        return False
    if claim.field.startswith("transmission_") or claim.field == "gear_ratios":
        raw_types = context.get("transmission_type")
        values = raw_types if isinstance(raw_types, (list, tuple)) else (raw_types,)
        expected = transmission_architecture(
            row.get("transmission_type"), row.get("gears"), row.get("electrification")
        )
        if values and any(present(value) for value in values):
            source_architectures = {
                normalize_architecture(value) for value in values if present(value)
            }
            if expected not in source_architectures:
                return False
        if claim.field == "transmission_type_normalized":
            if normalize_architecture(claim.value) != expected:
                return False
        if claim.field == "transmission_marketing_description":
            claim_architecture = normalize_architecture(claim.value)
            if claim_architecture not in {"OTHER", "UNKNOWN", expected}:
                return False
    return True


def _evidence_match(row: Mapping[str, Any], claim: EvidenceClaim) -> str | None:
    if claim.application_match == ApplicationMatch.EXACT:
        return "EXACT"
    if claim.application_match in {ApplicationMatch.STRONG, ApplicationMatch.PARTIAL}:
        return "PARTIAL"
    if claim.application_match == ApplicationMatch.MISMATCH and _compatible_broader_claim(row, claim):
        return "PARTIAL"
    return None


def _single_numeric(value: Any) -> str | None:
    if isinstance(value, bool) or isinstance(value, (dict, list, tuple)):
        return None
    matches = re.findall(r"[-+]?\d+(?:[.,]\d+)?", str(value))
    if len(matches) != 1:
        return None
    return matches[0].replace(",", ".")


def claims_to_curated_evidence(
    row: Mapping[str, Any], claims: Iterable[EvidenceClaim]
) -> list[dict[str, str]]:
    """Convert accepted live claims into the stable v0.5 evidence contract."""
    converted: list[dict[str, str]] = []
    for claim in claims:
        field = CLAIM_FIELD_MAP.get(claim.field)
        match = _evidence_match(row, claim)
        if field is None or match is None or not present(claim.value):
            continue
        value = claim.value
        note = claim.evidence_text.strip()
        if field == "transmission_family" and is_architecture_label(value):
            field = "transmission_architecture"
            note = f"Architecture-only source wording; not treated as family. {note}"
        if field == "transmission_architecture":
            value = normalize_architecture(value)
        if field in {
            "physical_final_drive",
            "reduction_front",
            "reduction_rear",
            "cd",
            "frontal_area_m2",
            "cda_m2",
        }:
            value = _single_numeric(value)
            if value is None:
                continue
        converted.append(
            {
                "sample_id": str(row["sample_id"]),
                "make": str(row["make"]),
                "model_pattern": "",
                "year_min": str(row["model_year"]),
                "year_max": str(row["model_year"]),
                "field": field,
                "value": str(value),
                "application_match": match,
                "source_tier": claim.source_tier.value,
                "source_classification": claim.source_classification,
                "source_url": claim.source_url,
                "source_title": claim.document_title,
                "evidence_note": note,
            }
        )
    return converted


def sanitize_evidence(evidence: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    """Prevent architecture labels from masquerading as transmission families."""
    sanitized: list[dict[str, str]] = []
    for source in evidence:
        item = dict(source)
        if item.get("field") == "transmission_family" and is_architecture_label(item.get("value")):
            item["field"] = "transmission_architecture"
            item["evidence_note"] = (
                "Architecture-only label excluded from family coverage. "
                + item.get("evidence_note", "")
            ).strip()
        if item.get("field") == "transmission_architecture":
            item["value"] = normalize_architecture(item.get("value"))
        sanitized.append(item)
    return sanitized


def enrich_application_v051(
    row: Mapping[str, Any], evidence: Iterable[Mapping[str, str]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return enrich_application(row, sanitize_evidence(evidence))


def repeated_groups_v051(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        identity, level = grouping_identity(row)
        if identity != "UNKNOWN":
            grouped[(identity, level)].append(row)
    result: list[dict[str, Any]] = []
    for (identity, level), members in sorted(grouped.items()):
        if len(members) < 2:
            continue
        urls = {
            url
            for row in members
            for url in str(row.get("source_urls", "")).split(";")
            if url
        }
        result.append(
            {
                "group_identity": identity,
                "group_level": level,
                "supplier": ";".join(
                    sorted(
                        {
                            str(row.get("transmission_supplier"))
                            for row in members
                            if present(row.get("transmission_supplier"))
                        }
                    )
                ),
                "applications": len(members),
                "models": ";".join(
                    sorted({f"{row['make']} {row['model']}" for row in members})
                ),
                "years": ";".join(
                    map(str, sorted({int(row["model_year"]) for row in members}))
                ),
                "provenance_mix": ";".join(
                    sorted(
                        {
                            str(row.get(f"transmission_{name}_provenance", ""))
                            for row in members
                            for name in ("code", "family", "architecture")
                            if row.get(f"transmission_{name}_provenance")
                        }
                    )
                ),
                "source_count": len(urls),
            }
        )
    return result
