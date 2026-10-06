"""Deterministic search planning helpers for the v0.5.2 golden benchmark."""
from __future__ import annotations

from dataclasses import dataclass
import re
from statistics import median
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

from .cross_oem_v05 import present, transmission_architecture
from .cross_oem_v051 import ARCHITECTURE_VALUES, normalize_architecture


GOLDEN_CASE_CONFIG: tuple[dict[str, Any], ...] = (
    {"golden_case_id": "G052-01", "sample_id": "V05-005", "model_alias": "GT", "official_hint": "media.ford.com", "expected_useful_fields": "transmission_family;transmission_architecture;gear_ratios;physical_final_drive", "diagnostic_notes": "Ford GT DCT technical identity and ratios/FDR."},
    {"golden_case_id": "G052-02", "sample_id": "V05-010", "model_alias": "Aviator Grand Touring", "official_hint": "lincoln.com", "gap_hint": "media.ford.com", "expected_useful_fields": "transmission_architecture;physical_final_drive;tire_general", "diagnostic_notes": "PHEV source row; search official Grand Touring name."},
    {"golden_case_id": "G052-03", "sample_id": "V05-015", "model_alias": "CT4-V", "official_hint": "cadillac.com", "expected_useful_fields": "transmission_architecture;transmission_family;tire_general", "diagnostic_notes": "Manual source identity may conflict with CT4-V versus Blackwing naming."},
    {"golden_case_id": "G052-04", "sample_id": "V05-023", "model_alias": "GX 550", "expected_useful_fields": "transmission_architecture;transmission_marketing_description;physical_final_drive", "diagnostic_notes": "Official 10-speed Direct Shift evidence expected."},
    {"golden_case_id": "G052-05", "sample_id": "V05-027", "model_alias": "4Runner", "expected_useful_fields": "transmission_architecture;physical_final_drive;tire_general", "diagnostic_notes": "Official 8-speed and drivetrain specifications; adjacent MY permitted."},
    {"golden_case_id": "G052-06", "sample_id": "V05-035", "model_alias": "AMG CLA 35", "official_hint": "mbusa.com", "expected_useful_fields": "transmission_architecture;cd;tire_general", "diagnostic_notes": "Official DCT plus aero/tire specifications."},
    {"golden_case_id": "G052-07", "sample_id": "V05-038", "model_alias": "CLA 250+", "official_hint": "media.mercedes-benz.com", "expected_useful_fields": "transmission_architecture;reduction_rear;cd", "diagnostic_notes": "EV gearbox/reduction and aero; wheel suffix omitted only from search phrase."},
    {"golden_case_id": "G052-08", "sample_id": "V05-042", "model_alias": "G70", "expected_useful_fields": "transmission_architecture;transmission_marketing_description;tire_general", "diagnostic_notes": "Follow official technical/specification link when exposed."},
    {"golden_case_id": "G052-09", "sample_id": "V05-046", "model_alias": "Elantra N", "expected_useful_fields": "transmission_architecture;transmission_marketing_description;tire_general", "diagnostic_notes": "Application-specific DCT/manual variant must remain explicit."},
    {"golden_case_id": "G052-10", "sample_id": "V05-050", "model_alias": "Carnival Hybrid", "official_model_alias": "Carnival HEV", "expected_useful_fields": "transmission_architecture;gear_ratios;physical_final_drive;tire_general", "diagnostic_notes": "Rich official technical page expected."},
)

OEM_SOURCE_HINTS: dict[str, tuple[str, ...]] = {
    "FORD": ("fromtheroad.ford.com", "media.ford.com", "ford.com", "fordservicecontent.com"),
    "LINCOLN": ("media.lincoln.com", "lincoln.com", "media.ford.com"),
    "CADILLAC": ("media.gm.com", "cadillac.com"),
    "BUICK": ("media.gm.com", "buick.com"),
    "CHEVROLET": ("media.gm.com", "chevrolet.com"),
    "GMC": ("media.gm.com", "gmc.com"),
    "LEXUS": ("pressroom.lexus.com", "lexus.com", "lexus.ca"),
    "TOYOTA": ("pressroom.toyota.com", "toyota.com"),
    "MERCEDES-BENZ": ("mbusa.com", "media.mbusa.com", "media.mercedes-benz.com", "group-media.mercedes-benz.com"),
    "GENESIS": ("genesis.com", "newsroom.genesis.com"),
    "HYUNDAI": ("hyundaiusa.com", "hyundainews.com", "hyundai.com"),
    "KIA": ("kiamedia.com", "kia.com"),
}

TRUSTED_SECONDARY_HOSTS = frozenset(
    {
        "caranddriver.com",
        "edmunds.com",
        "automobile-catalog.com",
        "auto-data.net",
        "conceptcarz.com",
        "fueleconomy.gov",
    }
)

USEFUL_FIELDS = frozenset(
    {
        "transmission_code",
        "transmission_family",
        "transmission_architecture",
        "transmission_supplier",
        "transmission_marketing_description",
        "gear_ratios",
        "physical_final_drive",
        "reduction_front",
        "reduction_rear",
        "cd",
        "frontal_area_m2",
        "cda_m2",
        "tire_front",
        "tire_rear",
        "tire_general",
    }
)

SUPPORT_FIELDS = frozenset(
    {
        "gear_ratios",
        "physical_final_drive",
        "reduction_front",
        "reduction_rear",
        "cd",
        "frontal_area_m2",
        "cda_m2",
        "tire_front",
        "tire_rear",
        "tire_general",
    }
)


@dataclass(frozen=True)
class PlannedQuery:
    theme: str
    query: str
    round_number: int
    reason: str


def normalize_search_identity(row: Mapping[str, Any], *, model_alias: str | None = None) -> str:
    """Build a concise phrase without mutating the raw application identity."""
    year = str(row.get("model_year") or "").strip()
    make = re.sub(r"\s+", " ", str(row.get("make") or "").strip()).title()
    if make.upper() == "MERCEDES-BENZ":
        make = "Mercedes"
    model = model_alias or str(row.get("model") or "")
    model = re.sub(r"\([^)]*(?:WHEEL|R\d{2}|\d{2}[\"']{1,2})[^)]*\)", "", model, flags=re.IGNORECASE)
    model = re.sub(r"\bWITH\s+EQ\s+TECHNOLOGY\b", "", model, flags=re.IGNORECASE)
    model = re.sub(r"\s+", " ", model).strip(" -_,")
    return " ".join(part for part in (year, make, model) if part)


def primary_queries(row: Mapping[str, Any], identity: str) -> tuple[PlannedQuery, ...]:
    make = str(row.get("make") or "").upper()
    hint = str(row.get("official_hint") or "") or OEM_SOURCE_HINTS.get(make, ("",))[0]
    official_identity = normalize_search_identity(row, model_alias=str(row.get("official_model_alias") or "")) if row.get("official_model_alias") else identity
    official = f'site:{hint} "{official_identity}" transmission' if hint else f'"{official_identity}" transmission'
    return (
        PlannedQuery("VEHICLE_SPECS", f'"{identity}" specifications', 1, "PRIMARY_SPECIFICATION_DISCOVERY"),
        PlannedQuery("POWERTRAIN", official, 1, "OFFICIAL_TRANSMISSION_DISCOVERY"),
    )


def missing_high_value_fields(found_fields: Iterable[str]) -> frozenset[str]:
    found = set(found_fields)
    targets = {
        "transmission_architecture",
        "transmission_family",
        "gear_ratios",
        "physical_final_drive",
        "reduction_front",
        "reduction_rear",
        "cd",
        "frontal_area_m2",
        "tire_general",
    }
    return frozenset(targets - found)


def gap_queries(
    row: Mapping[str, Any], identity: str, found_fields: Iterable[str]
) -> tuple[PlannedQuery, ...]:
    missing = missing_high_value_fields(found_fields)
    result: list[PlannedQuery] = []
    transmission_missing = missing & {
        "transmission_architecture",
        "transmission_family",
        "gear_ratios",
        "physical_final_drive",
        "reduction_front",
        "reduction_rear",
    }
    if transmission_missing:
        if "transmission_architecture" in transmission_missing or "transmission_family" in transmission_missing:
            phrase = "transmission code"
        elif "gear_ratios" in transmission_missing:
            phrase = "gear ratios"
        else:
            phrase = "final drive reduction"
        query_identity = identity.replace("+", " Plus")
        if str(row.get("electrification") or "").upper() == "PHEV":
            phrase = "PHEV transmission final drive"
        hint = str(row.get("gap_hint") or "")
        query = f'site:{hint} "{query_identity}" {phrase}' if hint else f'"{query_identity}" {phrase}'
        result.append(PlannedQuery("POWERTRAIN", query, 2, "MISSING_POWERTRAIN_FIELDS:" + ";".join(sorted(transmission_missing))))
    specs_missing = missing & {"cd", "frontal_area_m2", "tire_general"}
    if specs_missing:
        phrase = " ".join(
            label
            for field, label in (
                ("cd", "drag coefficient"),
                ("frontal_area_m2", "frontal area"),
                ("tire_general", "tire size"),
            )
            if field in specs_missing
        )
        query_identity = identity.replace("+", " Plus")
        result.append(PlannedQuery("VEHICLE_SPECS", f'"{query_identity}" {phrase}', 2, "MISSING_VEHICLE_SPEC_FIELDS:" + ";".join(sorted(specs_missing))))
    return tuple(result)


def source_is_official(url: str, make: Any) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return any(host == hint or host.endswith("." + hint) for hint in OEM_SOURCE_HINTS.get(str(make or "").upper(), ()))


def source_is_trusted_secondary(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return any(host == item or host.endswith("." + item) for item in TRUSTED_SECONDARY_HOSTS)


def source_discovery_priority(source: Any, *, make: Any) -> tuple[int, int, int, str]:
    combined = f"{source.title} {source.url} {source.document_type}".lower()
    official = source_is_official(source.url, make)
    technical = any(token in combined for token in ("technical", "specification", "specifications", ".pdf", "spec sheet", "data sheet"))
    rank = int(source.metadata.get("search_rank", 999))
    return (0 if official else 1, 0 if technical else 1, rank, source.url)


def source_richness(
    fields: Iterable[str], *, official: bool, technical_document: bool
) -> int:
    unique = set(fields) & USEFUL_FIELDS
    weights = {
        "transmission_code": 4,
        "transmission_family": 3,
        "transmission_architecture": 2,
        "transmission_supplier": 1,
        "transmission_marketing_description": 1,
        "gear_ratios": 3,
        "physical_final_drive": 2,
        "reduction_front": 2,
        "reduction_rear": 2,
        "cd": 2,
        "frontal_area_m2": 2,
        "cda_m2": 2,
        "tire_front": 1,
        "tire_rear": 1,
        "tire_general": 1,
    }
    return sum(weights[field] for field in unique) + (3 if official else 0) + (2 if technical_document else 0)


def coverage_sufficient(found_fields: Iterable[str], *, richest_source_field_count: int = 0) -> bool:
    fields = set(found_fields)
    has_transmission = bool(fields & {"transmission_code", "transmission_family", "transmission_architecture"})
    has_support = bool(fields & SUPPORT_FIELDS)
    return (has_transmission and has_support) or richest_source_field_count >= 3


def normalize_raw_transmission_description(value: Any, *, electrification: Any = "ICE") -> str:
    text = str(value or "")
    if str(electrification or "").upper() in {"BEV", "FCEV"}:
        upper = text.upper().replace("-", " ")
        if re.search(r"\b(?:2|TWO)\s+SPEED\b", upper):
            return "MULTI_SPEED_EV"
        if "FIXED RATIO" in upper:
            return "SINGLE_SPEED_EV"
        normalized = normalize_architecture(text)
        if normalized in {"SINGLE_SPEED_EV", "MULTI_SPEED_EV"}:
            return normalized
    return normalize_architecture(text)


def adjacent_year_is_compatible(
    requested_year: Any,
    source_context: Mapping[str, Any],
    *,
    requested_make: Any,
    requested_model: Any,
) -> bool:
    def norm(value: Any) -> str:
        return re.sub(r"[^A-Z0-9]+", " ", str(value or "").upper()).strip()

    if norm(source_context.get("make")) != norm(requested_make):
        return False
    if norm(source_context.get("model")) != norm(requested_model):
        return False
    try:
        source_year = int(float(str(source_context.get("model_year"))))
        return abs(source_year - int(float(str(requested_year)))) == 1
    except (TypeError, ValueError):
        return False


def architecture_error_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        present(row.get("transmission_architecture"))
        and str(row.get("transmission_architecture")) not in ARCHITECTURE_VALUES
        for row in rows
    )


def median_searches(rows: Sequence[Mapping[str, Any]]) -> float:
    return float(median([int(row.get("searches_count") or 0) for row in rows])) if rows else 0.0
