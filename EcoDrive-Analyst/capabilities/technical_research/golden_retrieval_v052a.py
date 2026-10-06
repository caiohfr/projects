"""Final deterministic planner and evidence plumbing for v0.5.2a."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping, Sequence

from .cross_oem_v05 import present
from .golden_retrieval_v052 import OEM_SOURCE_HINTS, PlannedQuery, normalize_search_identity


RESULT_VALUE_FIELDS = (
    "transmission_code", "transmission_family", "transmission_architecture",
    "transmission_supplier", "transmission_marketing_description", "gear_ratios",
    "physical_final_drive", "reduction_front", "reduction_rear", "cd",
    "frontal_area_m2", "cda_m2", "tire_front", "tire_rear", "tire_general",
)

PROVENANCE_RANK = {"EXACT": 3, "STRONG": 2, "PARTIAL": 1}
TIER_RANK = {"TIER_1_PRIMARY": 3, "TIER_2_STRONG_SECONDARY": 2, "TIER_3_DISCOVERY_ONLY": 1}


@dataclass(frozen=True)
class SelectedEvidence:
    values: Mapping[str, Any]
    selected_keys: frozenset[tuple[str, str, str]]
    conflicts: frozenset[str]


def official_first_queries(row: Mapping[str, Any], identity: str) -> tuple[PlannedQuery, ...]:
    make = str(row.get("make") or "").upper()
    hints = OEM_SOURCE_HINTS.get(make, ())
    primary_hint = str(row.get("official_hint") or "") or (hints[0] if hints else "")
    official_alias = str(row.get("official_model_alias") or "")
    official_identity = normalize_search_identity(row, model_alias=official_alias) if official_alias else identity
    site = f"site:{primary_hint} " if primary_hint else ""
    return (
        PlannedQuery("OFFICIAL_TECHNICAL", f'{site}"{official_identity}" technical specifications', 1, "OFFICIAL_TECHNICAL_DISCOVERY"),
        PlannedQuery("OFFICIAL_POWERTRAIN", f'{site}"{official_identity}" transmission specifications', 1, "OFFICIAL_POWERTRAIN_DISCOVERY"),
        PlannedQuery("SECONDARY_TECHNICAL", f'"{identity}" technical specifications transmission', 2, "TRUSTED_SECONDARY_AFTER_OFFICIAL"),
        PlannedQuery("GAP", f'"{identity.replace("+", " Plus")}" gear ratios final drive drag coefficient tire size', 2, "HIGH_VALUE_GAPS_AFTER_PRIMARY_EXTRACTION"),
    )


def is_official_phase(query: PlannedQuery) -> bool:
    return query.theme.startswith("OFFICIAL_")


def official_first_order_is_valid(queries: Sequence[PlannedQuery]) -> bool:
    themes = [item.theme for item in queries]
    secondary_index = next((index for index, theme in enumerate(themes) if theme.startswith("SECONDARY") or theme == "GAP"), len(themes))
    return themes[:2] == ["OFFICIAL_TECHNICAL", "OFFICIAL_POWERTRAIN"] and all(theme.startswith("OFFICIAL_") for theme in themes[:secondary_index])


def evidence_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row.get("field", "")), str(row.get("value", "")), str(row.get("source_url", ""))


def select_final_evidence(rows: Iterable[Mapping[str, Any]]) -> SelectedEvidence:
    """Select researched values directly so accepted claims cannot be hidden by observed fallbacks."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in rows:
        field = str(raw.get("field") or "")
        if field not in RESULT_VALUE_FIELDS or not present(raw.get("value")):
            continue
        grouped.setdefault(field, []).append(dict(raw))
    values: dict[str, Any] = {field: "" for field in RESULT_VALUE_FIELDS}
    selected: set[tuple[str, str, str]] = set()
    conflicts: set[str] = set()
    for field, candidates in grouped.items():
        candidates.sort(
            key=lambda row: (
                -PROVENANCE_RANK.get(str(row.get("application_match", "")).upper(), 0),
                -TIER_RANK.get(str(row.get("source_tier", "")), 0),
                str(row.get("source_url", "")),
            )
        )
        best = candidates[0]
        best_rank = (
            PROVENANCE_RANK.get(str(best.get("application_match", "")).upper(), 0),
            TIER_RANK.get(str(best.get("source_tier", "")), 0),
        )
        peer_values = {
            str(row.get("value", "")).strip().casefold()
            for row in candidates
            if (
                PROVENANCE_RANK.get(str(row.get("application_match", "")).upper(), 0),
                TIER_RANK.get(str(row.get("source_tier", "")), 0),
            ) == best_rank
        }
        if len(peer_values) > 1:
            conflicts.add(field)
            continue
        values[field] = best.get("value", "")
        selected.add(evidence_key(best))
    return SelectedEvidence(values=values, selected_keys=frozenset(selected), conflicts=frozenset(conflicts))


def distinct_useful_field_count(values: Mapping[str, Any]) -> int:
    groups = (
        ("transmission_code", "transmission_family", "transmission_marketing_description"),
        ("transmission_supplier",),
        ("transmission_architecture",),
        ("gear_ratios",),
        ("physical_final_drive", "reduction_front", "reduction_rear"),
        ("cd",),
        ("frontal_area_m2",),
        ("cda_m2",),
        ("tire_front", "tire_rear", "tire_general"),
    )
    return sum(any(present(values.get(field)) for field in group) for group in groups)


def claim_drop_errors(rows: Iterable[Mapping[str, Any]], selected: SelectedEvidence) -> int:
    by_field = {key[0] for key in selected.selected_keys}
    accepted_fields = {
        str(row.get("field")) for row in rows
        if str(row.get("field")) in RESULT_VALUE_FIELDS and present(row.get("value"))
    }
    return len(accepted_fields - by_field - set(selected.conflicts))


def architecture_from_raw_wording(value: Any, *, electrification: Any = "ICE") -> str:
    from .golden_retrieval_v052 import normalize_raw_transmission_description

    return normalize_raw_transmission_description(value, electrification=electrification)


def normalize_for_match(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def expected_pattern_matches(value: Any, pattern: str) -> bool:
    return bool(re.search(pattern, str(value or ""), re.IGNORECASE))


def choose_extraction_model(metrics: Mapping[str, Mapping[str, Any]]) -> str:
    terra = metrics["gpt-5.6-terra"]
    sol = metrics["gpt-5.6-sol"]
    recall_gain = float(sol["field_recall"]) - float(terra["field_recall"])
    precision_not_worse = float(sol["field_precision"]) + 0.01 >= float(terra["field_precision"])
    high_value_gain = int(sol["correct_fields"]) - int(terra["correct_fields"])
    return "gpt-5.6-sol" if recall_gain >= 0.05 and precision_not_worse and high_value_gain >= 2 else "gpt-5.6-terra"


def golden_architecture_error_count(rows: Iterable[Mapping[str, Any]]) -> int:
    """Count gross architecture errors for applications with known signatures."""
    expected = {
        "G052-01": "DCT",
        "G052-06": "DCT",
        "G052-07": "MULTI_SPEED_EV",
        "G052-09": "DCT",
    }
    errors = 0
    for row in rows:
        wanted = expected.get(str(row.get("golden_case_id")))
        if wanted and str(row.get("transmission_architecture") or "") != wanted:
            errors += 1
    return errors
