"""Cross-OEM pragmatic enrichment helpers for Components Research v0.5.

The module is deliberately review-only.  SQLite is opened with ``mode=ro``
and an authorizer that rejects every mutating operation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

from .semantic_cleanup_v041 import calculated_cda, numeric_engineering_value, present


PROVENANCE_ORDER = (
    "OBSERVED",
    "RESEARCHED_EXACT",
    "RESEARCHED_APPROX",
    "CALCULATED",
    "RULE_ESTIMATED",
    "UNKNOWN",
)
PROVENANCE_RANK = {name: len(PROVENANCE_ORDER) - index for index, name in enumerate(PROVENANCE_ORDER)}

OEM_GROUPS: dict[str, frozenset[str]] = {
    "FORD_LINCOLN": frozenset({"FORD", "LINCOLN"}),
    "GM": frozenset({"CHEVROLET", "CADILLAC", "BUICK", "GMC"}),
    "TOYOTA_LEXUS": frozenset({"TOYOTA", "LEXUS"}),
    "MERCEDES_BENZ": frozenset({"MERCEDES-BENZ", "MERCEDES BENZ", "MERCEDES"}),
    "HYUNDAI_KIA_GENESIS": frozenset({"HYUNDAI", "KIA", "GENESIS"}),
}

SAMPLE_FIELDS = (
    "sample_id", "vehicle_configuration_id", "vde_id", "make", "model",
    "model_year", "category", "drive_type", "transmission_type", "gears",
    "source_final_drive", "nv_ratio", "electrification", "carryover_root_id",
    "selection_reason", "oem_group",
)

VALUE_FIELDS = (
    "transmission_code", "transmission_family", "transmission_architecture",
    "transmission_supplier", "transmission_marketing_description", "gears",
    "gear_ratios", "source_final_drive", "physical_final_drive",
    "final_drive_semantics", "reduction_front", "reduction_rear", "cd",
    "frontal_area_m2", "cda_m2", "tire_front", "tire_rear", "tire_general",
)

NUMERIC_FIELDS = (
    "cd", "frontal_area_m2", "cda_m2", "physical_final_drive",
    "source_final_drive", "reduction_front", "reduction_rear",
)


@dataclass(frozen=True)
class SelectedValue:
    value: Any = None
    provenance: str = "UNKNOWN"
    observed_value: Any = None
    researched_exact_value: Any = None
    researched_approx_value: Any = None
    calculated_value: Any = None
    rule_estimated_value: Any = None
    source_url: str = ""
    evidence_note: str = ""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _read_only_authorizer(action: int, *_args: Any) -> int:
    denied_names = (
        "SQLITE_INSERT", "SQLITE_UPDATE", "SQLITE_DELETE", "SQLITE_CREATE_INDEX",
        "SQLITE_CREATE_TABLE", "SQLITE_CREATE_TEMP_INDEX", "SQLITE_CREATE_TEMP_TABLE",
        "SQLITE_CREATE_TEMP_TRIGGER", "SQLITE_CREATE_TEMP_VIEW", "SQLITE_CREATE_TRIGGER",
        "SQLITE_CREATE_VIEW", "SQLITE_DROP_INDEX", "SQLITE_DROP_TABLE",
        "SQLITE_DROP_TEMP_INDEX", "SQLITE_DROP_TEMP_TABLE", "SQLITE_DROP_TEMP_TRIGGER",
        "SQLITE_DROP_TEMP_VIEW", "SQLITE_DROP_TRIGGER", "SQLITE_DROP_VIEW",
        "SQLITE_ALTER_TABLE", "SQLITE_REINDEX", "SQLITE_ANALYZE", "SQLITE_ATTACH",
        "SQLITE_DETACH",
    )
    denied = {getattr(sqlite3, name) for name in denied_names if hasattr(sqlite3, name)}
    return sqlite3.SQLITE_DENY if action in denied else sqlite3.SQLITE_OK


def open_read_only(path: Path) -> sqlite3.Connection:
    resolved = path.resolve(strict=True)
    connection = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.set_authorizer(_read_only_authorizer)
    return connection


def select_value(
    *, observed: Any = None, researched_exact: Any = None,
    researched_approx: Any = None, calculated: Any = None,
    rule_estimated: Any = None, source_url: str = "", evidence_note: str = "",
) -> SelectedValue:
    candidates = (
        (observed, "OBSERVED"),
        (researched_exact, "RESEARCHED_EXACT"),
        (researched_approx, "RESEARCHED_APPROX"),
        (calculated, "CALCULATED"),
        (rule_estimated, "RULE_ESTIMATED"),
    )
    selected, provenance = None, "UNKNOWN"
    for value, candidate_provenance in candidates:
        if present(value):
            selected, provenance = value, candidate_provenance
            break
    return SelectedValue(
        value=selected,
        provenance=provenance,
        observed_value=observed,
        researched_exact_value=researched_exact,
        researched_approx_value=researched_approx,
        calculated_value=calculated,
        rule_estimated_value=rule_estimated,
        source_url=source_url,
        evidence_note=evidence_note,
    )


def oem_group(make: Any) -> str:
    normalized = str(make or "").strip().upper()
    for group, makes in OEM_GROUPS.items():
        if normalized in makes:
            return group
    return ""


def drive_bucket(value: Any) -> str:
    text = str(value or "").upper()
    if "ALL WHEEL" in text or "4-WHEEL" in text or "4 WHEEL" in text:
        return "AWD_4WD"
    if "FRONT" in text:
        return "FWD"
    if "REAR" in text:
        return "RWD"
    return "UNKNOWN"


def infer_electrification(reported: Any, propulsion: Any, engine_type: Any, model: Any = None) -> str:
    values = " ".join(str(value or "") for value in (reported, propulsion, engine_type, model)).upper()
    model_text = str(model or "").upper()
    if "PHEV" in values or "PLUG-IN" in values or "PLUG IN" in values or re.search(r"\b\d{3}H\+", model_text):
        return "PHEV"
    if "HYDROGEN" in values or re.search(r"\bFCEV\b", values) or model_text.strip() == "NEXO":
        return "FCEV"
    if "BEV" in values or "BATTERY ELECTRIC" in values or "ELECTRICITY" in values:
        return "BEV"
    if "HYBRID" in values or re.search(r"\bHEV\b", values) or re.search(r"\b\d{3}H\b", model_text):
        return "HEV"
    return "ICE"


def transmission_architecture(transmission_type: Any, gears: Any, electrification: Any) -> str:
    text = str(transmission_type or "").upper()
    gear_count = numeric_engineering_value(gears)
    if str(electrification).upper() in {"BEV", "FCEV"}:
        if "MULTI" in text or (gear_count is not None and gear_count > 1.0):
            return "MULTI_SPEED_EV"
        return "SINGLE_SPEED_EV"
    # Specific architectures must be checked before broad words such as
    # ``AUTOMATIC`` and ``MANUAL``.  EPA source labels are preserved elsewhere;
    # this helper only emits the normalized physical architecture.
    if "DUAL CLUTCH" in text or re.search(r"\bDCT\b", text):
        return "DCT"
    if "CONTINUOUS" in text or "CVT" in text:
        return "CVT"
    if "AUTOMATED MANUAL" in text:
        return "AUTOMATED_MANUAL"
    if any(marker in text for marker in ("SEMI-AUTOMATIC", "SEMI AUTOMATIC", "SEQUENTIAL")):
        return "OTHER"
    if "MANUAL" in text:
        return "MANUAL"
    if "SINGLE SPEED" in text or "SINGLE-SPEED" in text:
        return "SINGLE_SPEED_EV"
    if "MULTI SPEED" in text or "MULTI-SPEED" in text:
        return "MULTI_SPEED_EV"
    if "AUTOMATIC" in text:
        return "TORQUE_CONVERTER_AUTOMATIC"
    if "AUTOMATED" in text:
        return "OTHER"
    if text:
        return "OTHER"
    return "UNKNOWN"


def match_status(*, code: Any = None, family: Any = None, architecture: Any = None, conflict: bool = False) -> str:
    if conflict:
        return "CONFLICT"
    if present(code):
        return "FOUND_EXACT"
    if present(family):
        return "FOUND_FAMILY"
    if present(architecture) and str(architecture) != "UNKNOWN":
        return "FOUND_ARCHITECTURE"
    return "NOT_FOUND"


def grouping_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    for field, level in (
        ("transmission_code", "EXACT_CODE"),
        ("transmission_family", "FAMILY"),
        ("transmission_architecture", "ARCHITECTURE"),
    ):
        if present(row.get(field)) and str(row.get(field)) != "UNKNOWN":
            return str(row[field]), level
    return "UNKNOWN", "UNKNOWN"


def _model_key(value: Any) -> str:
    text = re.sub(r"\([^)]*\)", "", str(value or "").upper())
    text = re.sub(r"\b(?:AWD|FWD|RWD|4WD)\b", "", text)
    return re.sub(r"[^A-Z0-9]+", " ", text).strip()


def _carryover_roots(parent_by_id: Mapping[int, int | None]) -> dict[int, int]:
    roots: dict[int, int] = {}
    for identifier in parent_by_id:
        current, seen = identifier, set()
        while parent_by_id.get(current) is not None and current not in seen:
            seen.add(current)
            current = int(parent_by_id[current])
        roots[identifier] = current
    return roots


def load_canonical_candidates(db_path: Path) -> list[dict[str, Any]]:
    with open_read_only(db_path) as connection:
        rows = connection.execute(
            """
            SELECT v.id AS vde_id, v.vehicle_configuration_id, v.make, v.model,
                   v.year AS model_year, v.category,
                   COALESCE(vc.drive_system, v.drive_type) AS drive_type,
                   COALESCE(vc.transmission_type, v.transmission_type) AS transmission_type,
                   COALESCE(vc.gear_count,
                     (SELECT MIN(f.gear_count) FROM fuelcons f
                      WHERE f.vde_id=v.id AND f.record_status='ACTIVE'
                        AND f.review_status='CURRENT')) AS gears,
                   vc.final_drive_ratio AS source_final_drive, vc.nv_ratio,
                   vc.propulsion_architecture, vc.engine_type,
                   v.vde_id_parent,
                   (SELECT MIN(f.electrification) FROM fuelcons f
                    WHERE f.vde_id=v.id AND f.record_status='ACTIVE'
                      AND f.review_status='CURRENT') AS reported_electrification,
                   v.tire_size, v.cda_m2
              FROM vde v
              JOIN vehicle_configuration vc
                ON vc.vehicle_configuration_id=v.vehicle_configuration_id
             WHERE v.legislation='EPA'
               AND v.record_status='ACTIVE' AND v.review_status='CURRENT'
               AND vc.record_status='ACTIVE' AND vc.review_status='CURRENT'
             ORDER BY UPPER(v.make), UPPER(v.model), v.year, v.id
            """
        ).fetchall()
    parent_by_id = {int(row["vde_id"]): row["vde_id_parent"] for row in rows}
    roots = _carryover_roots(parent_by_id)
    result: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row["oem_group"] = oem_group(row["make"])
        row["electrification"] = infer_electrification(
            row.pop("reported_electrification"), row.pop("propulsion_architecture"),
            row.pop("engine_type"), row["model"],
        )
        row["carryover_root_id"] = roots[int(row["vde_id"])]
        result.append(row)
    return result


def deterministic_sample(candidates: Sequence[Mapping[str, Any]], per_group: int = 10) -> list[dict[str, Any]]:
    """Select a stable, diverse sample while collapsing exact carryover chains."""
    selected: list[dict[str, Any]] = []
    for group in OEM_GROUPS:
        pool = [dict(row) for row in candidates if row.get("oem_group") == group and str(row.get("make", "")).upper() != "BMW"]
        # One representative for an application signature, preferring the newest
        # member of a carryover chain and then the smallest stable VDE id.
        by_signature: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in pool:
            signature = (
                str(row["make"]).upper(), _model_key(row["model"]),
                drive_bucket(row["drive_type"]), row["electrification"],
                str(row["transmission_type"] or "").upper(), row["gears"],
                row["carryover_root_id"],
            )
            incumbent = by_signature.get(signature)
            if incumbent is None or (row["model_year"], -int(row["vde_id"])) > (incumbent["model_year"], -int(incumbent["vde_id"])):
                by_signature[signature] = row
        remaining = sorted(
            by_signature.values(),
            key=lambda row: (
                str(row["make"]).upper(), _model_key(row["model"]),
                -int(row["model_year"] or 0), int(row["vde_id"]),
            ),
        )
        group_selected: list[dict[str, Any]] = []
        seen: dict[str, set[Any]] = defaultdict(set)
        while remaining and len(group_selected) < per_group:
            def score(row: Mapping[str, Any]) -> tuple[int, int, int, int, int, int]:
                architecture = transmission_architecture(row["transmission_type"], row["gears"], row["electrification"])
                return (
                    int(_model_key(row["model"]) not in seen["model"]) * 20,
                    int(str(row["make"]).upper() not in seen["make"]) * 8,
                    int(row["electrification"] not in seen["electrification"]) * 7,
                    int(drive_bucket(row["drive_type"]) not in seen["drive"]) * 5,
                    int((architecture, row["gears"]) not in seen["transmission"]) * 4,
                    int(row["category"] not in seen["category"]) * 2,
                )
            best = max(remaining, key=lambda row: score(row))
            remaining.remove(best)
            reasons: list[str] = []
            model_key = _model_key(best["model"])
            architecture = transmission_architecture(best["transmission_type"], best["gears"], best["electrification"])
            for label, value in (
                ("model", model_key), ("make", str(best["make"]).upper()),
                ("powertrain", best["electrification"]), ("drive", drive_bucket(best["drive_type"])),
                ("transmission", (architecture, best["gears"])), ("category", best["category"]),
            ):
                key = "electrification" if label == "powertrain" else label
                if value not in seen[key]:
                    reasons.append(f"new_{label}")
                seen[key].add(value)
            best["selection_reason"] = ";".join(reasons) or "deterministic_diversity_fill"
            group_selected.append(best)
        if len(group_selected) < per_group:
            raise ValueError(f"OEM group {group} has only {len(group_selected)} usable applications")
        selected.extend(group_selected)
    selected.sort(key=lambda row: (list(OEM_GROUPS).index(str(row["oem_group"])), str(row["make"]).upper(), _model_key(row["model"]), int(row["vde_id"])))
    for index, row in enumerate(selected, start=1):
        row["sample_id"] = f"V05-{index:03d}"
    return selected


def read_curated_evidence(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def applicable_evidence(row: Mapping[str, Any], evidence: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for item in evidence:
        if item.get("sample_id") and item["sample_id"] != row["sample_id"]:
            continue
        if item.get("make") and item["make"].strip().upper() != str(row["make"]).strip().upper():
            continue
        if item.get("model_pattern") and not re.search(item["model_pattern"], str(row["model"]), re.IGNORECASE):
            continue
        if item.get("year_min") and int(row["model_year"]) < int(item["year_min"]):
            continue
        if item.get("year_max") and int(row["model_year"]) > int(item["year_max"]):
            continue
        result.append(dict(item))
    return result


def evidence_selection(rows: Sequence[Mapping[str, str]], field: str) -> tuple[Any, Any, str, str, bool]:
    matches = [row for row in rows if row.get("field") == field and present(row.get("value"))]
    if not matches:
        return None, None, "", "", False
    normalized = {str(row["value"]).strip().casefold() for row in matches}
    if len(normalized) > 1:
        return None, None, ";".join(dict.fromkeys(row.get("source_url", "") for row in matches)), "Credible curated values conflict.", True
    exact = next((row for row in matches if row.get("application_match", "").upper() == "EXACT"), None)
    chosen = exact or matches[0]
    exact_value = chosen["value"] if exact else None
    approx_value = chosen["value"] if not exact else None
    return exact_value, approx_value, chosen.get("source_url", ""), chosen.get("evidence_note", ""), False


def numeric_or_none(value: Any) -> float | None:
    numeric = numeric_engineering_value(value)
    return float(numeric) if numeric is not None else None


def enrich_application(row: Mapping[str, Any], evidence: Sequence[Mapping[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    app_evidence = applicable_evidence(row, evidence)
    architecture = transmission_architecture(row["transmission_type"], row["gears"], row["electrification"])
    rules = {"transmission_architecture": architecture if architecture != "UNKNOWN" else None}
    observed = {
        "gears": row.get("gears"),
        "source_final_drive": row.get("source_final_drive"),
    }
    selections: dict[str, SelectedValue] = {}
    conflicts: set[str] = set()
    evidence_names = {
        "transmission_code", "transmission_family", "transmission_architecture",
        "transmission_supplier", "transmission_marketing_description", "gear_ratios",
        "physical_final_drive", "reduction_front", "reduction_rear", "cd", "frontal_area_m2", "cda_m2",
        "tire_front", "tire_rear", "tire_general",
    }
    for field in evidence_names:
        exact, approx, source, note, conflict = evidence_selection(app_evidence, field)
        if conflict:
            conflicts.add(field)
        if field in NUMERIC_FIELDS or field in {"reduction_front", "reduction_rear", "cd", "frontal_area_m2", "cda_m2"}:
            exact = numeric_or_none(exact)
            approx = numeric_or_none(approx)
        selections[field] = select_value(
            researched_exact=exact, researched_approx=approx,
            rule_estimated=rules.get(field), source_url=source, evidence_note=note,
        )
    selections["gears"] = select_value(observed=observed["gears"])
    selections["source_final_drive"] = select_value(observed=observed["source_final_drive"])
    is_ev = architecture in {"SINGLE_SPEED_EV", "MULTI_SPEED_EV"}
    physical_fdr = None if is_ev else numeric_or_none(row.get("source_final_drive"))
    semantics = "SOURCE_PLACEHOLDER" if is_ev and present(row.get("source_final_drive")) else ("PHYSICAL_FINAL_DRIVE" if physical_fdr is not None else "UNKNOWN")
    researched_fdr = selections.get("physical_final_drive", SelectedValue())
    selections["physical_final_drive"] = select_value(
        observed=physical_fdr,
        researched_exact=researched_fdr.researched_exact_value,
        researched_approx=researched_fdr.researched_approx_value,
        source_url=researched_fdr.source_url,
        evidence_note=researched_fdr.evidence_note,
    )
    if present(selections["physical_final_drive"].value) and semantics == "UNKNOWN":
        semantics = "RESEARCHED_PHYSICAL_FINAL_DRIVE"
    selections["final_drive_semantics"] = select_value(observed=semantics)
    cd = selections["cd"].value
    area = selections["frontal_area_m2"].value
    direct_cda = selections["cda_m2"]
    computed_cda = calculated_cda(cd, area)
    if not present(direct_cda.value) and computed_cda is not None:
        selections["cda_m2"] = select_value(calculated=computed_cda, evidence_note="Cd x frontal_area_m2 (DIRECT_PRODUCT).")
    cda_method = "DIRECT_SOURCE" if direct_cda.provenance in {"RESEARCHED_EXACT", "RESEARCHED_APPROX"} else ("DIRECT_PRODUCT" if computed_cda is not None else "UNKNOWN")
    status = match_status(
        code=selections["transmission_code"].value,
        family=selections["transmission_family"].value,
        architecture=selections["transmission_architecture"].value,
        conflict=bool(conflicts & {"transmission_code", "transmission_family", "transmission_architecture"}),
    )
    application = {field: row.get(field, "") for field in SAMPLE_FIELDS}
    application.update({
        "researched_model": row["model"],
        "researched_year_range": str(row["model_year"]),
        "researched_drive": row["drive_type"],
        "researched_powertrain": row["electrification"],
        "researched_market": "US EPA",
    })
    for field in VALUE_FIELDS:
        selection = selections.get(field, SelectedValue())
        application[field] = selection.value if present(selection.value) else ""
        application[f"{field}_provenance"] = selection.provenance
    application["cda_method"] = cda_method
    application["match_status"] = status
    application["selected_provenance"] = max(
        (selection.provenance for selection in selections.values()),
        key=lambda name: PROVENANCE_RANK[name],
        default="UNKNOWN",
    )
    application["source_count"] = len({item.get("source_url") for item in app_evidence if item.get("source_url")})
    application["source_urls"] = ";".join(dict.fromkeys(item.get("source_url", "") for item in app_evidence if item.get("source_url")))
    application["evidence_note"] = " | ".join(dict.fromkeys(item.get("evidence_note", "") for item in app_evidence if item.get("evidence_note")))
    provenance_counts = Counter(selection.provenance for selection in selections.values())
    if status == "CONFLICT":
        review = "REVIEW_CONFLICT"
    elif provenance_counts["RESEARCHED_APPROX"]:
        review = "REVIEW_APPROX"
    elif provenance_counts["RESEARCHED_EXACT"] or provenance_counts["CALCULATED"]:
        review = "REVIEW_OK"
    else:
        review = "REVIEW_SPARSE"
    application["review_flag"] = review
    audit: list[dict[str, Any]] = []
    for field in VALUE_FIELDS:
        selection = selections.get(field, SelectedValue())
        audit.append({
            "sample_id": row["sample_id"], "vde_id": row["vde_id"],
            "vehicle_application": f"{row['model_year']}|{row['make']}|{row['model']}",
            "field": field, "selected_value": selection.value if present(selection.value) else "",
            "provenance": selection.provenance,
            "observed_value": selection.observed_value if present(selection.observed_value) else "",
            "researched_exact_value": selection.researched_exact_value if present(selection.researched_exact_value) else "",
            "researched_approx_value": selection.researched_approx_value if present(selection.researched_approx_value) else "",
            "calculated_value": selection.calculated_value if present(selection.calculated_value) else "",
            "rule_estimated_value": selection.rule_estimated_value if present(selection.rule_estimated_value) else "",
            "source_url": selection.source_url,
            "note": selection.evidence_note,
        })
    return application, audit


def repeated_groups(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        identity, level = grouping_identity(row)
        if identity != "UNKNOWN":
            grouped[(identity, level)].append(row)
    result: list[dict[str, Any]] = []
    for (identity, level), members in sorted(grouped.items()):
        if len(members) < 2:
            continue
        result.append({
            "identity": identity, "identity_level": level,
            "supplier": ";".join(sorted({str(row.get("transmission_supplier")) for row in members if present(row.get("transmission_supplier"))})),
            "application_count": len(members),
            "applications": ";".join(f"{row['model_year']} {row['make']} {row['model']}" for row in members),
            "model_years": ";".join(map(str, sorted({int(row["model_year"]) for row in members}))),
            "provenance_mix": ";".join(sorted({str(row.get("transmission_code_provenance") or row.get("transmission_family_provenance") or row.get("transmission_architecture_provenance")) for row in members})),
        })
    return result


def validate_numeric_contract(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, dict[str, int]] = {}
    for field in NUMERIC_FIELDS:
        values = [row.get(field) for row in rows if present(row.get(field))]
        parseable = 0
        for value in values:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(parsed):
                parseable += 1
        result[field] = {"non_null": len(values), "parseable": parseable}
    return {
        "fields": result,
        "valid": all(item["non_null"] == item["parseable"] for item in result.values()),
    }
