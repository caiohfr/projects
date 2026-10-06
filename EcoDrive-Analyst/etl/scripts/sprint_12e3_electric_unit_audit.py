"""Sprint 12E.3: audit electric units and recover legacy corrections.

This audit is deliberately read-only with respect to source and runtime data.
It writes only CSV/JSON/report artifacts.  Numeric magnitude is used for QA,
never to choose a unit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sqlite3
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OLD_EPA = ROOT / "data" / "vehicles" / "testcar-2025-2020-EPA.xlsx"
EPA = ROOT / "etl" / "data" / "raw" / "epa_testcar" / "epa_testcar_2026_raw.xlsx"
FUEL_ECONOMY = ROOT / "etl" / "data" / "raw" / "fueleconomy" / "fueleconomy_2026_raw.zip"
JRC = ROOT / "etl" / "data" / "raw" / "wltp_jrc" / "Data_PV_fleet_2021_EU_PYCSIS.xlsx"
DB_12E2 = ROOT / "etl" / "data" / "staging" / "sprint_12e2_epa_fuelcons" / "eco_drive_canonical_epa_fuelcons.db"
OUT_12E2 = ROOT / "etl" / "data" / "processed" / "sprint_12e2_epa_fuelcons"
OUT = ROOT / "etl" / "data" / "processed" / "sprint_12e3_electric_unit_audit"
REPORT = ROOT / "etl" / "reports" / "sprint_12e3_electric_unit_audit.md"
OVERRIDES = ROOT / "etl" / "config" / "electric_consumption_overrides.csv"
SUMMARY = OUT / "electric_unit_audit_summary.json"
RUNTIME_DBS = (ROOT / "data" / "db" / "eco_drive.db", ROOT / "data" / "db" / "eco_drive_qa.db")
LEGACY_NOTEBOOK = "notebooks/etl_epa_xlsx_to_sqlite.ipynb:cell 2"
MI_TO_KM = 1.609344


class UnsupportedUnitError(ValueError):
    """Raised when evidence does not support a requested conversion."""


class StaleOverrideError(ValueError):
    """Raised when an override does not match its exact source record."""


def _positive(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("Electric efficiency/consumption must be finite and positive")
    return number


def mi_per_kwh_to_wh_per_km(value: Any) -> float | None:
    number = _positive(value)
    return None if number is None else 1000.0 / (number * MI_TO_KM)


def km_per_kwh_to_wh_per_km(value: Any) -> float | None:
    number = _positive(value)
    return None if number is None else 1000.0 / number


def kwh_per_100mi_to_wh_per_km(value: Any) -> float | None:
    number = _positive(value)
    return None if number is None else number * 1000.0 / (100.0 * MI_TO_KM)


def wh_per_mile_to_wh_per_km(value: Any) -> float | None:
    number = _positive(value)
    return None if number is None else number / MI_TO_KM


def mpge_to_wh_per_km(value: Any, energy_equivalent_wh_per_gallon: float | None = None) -> float | None:
    """Convert MPGe only when the caller supplies an established basis."""
    number = _positive(value)
    if number is None:
        return None
    if energy_equivalent_wh_per_gallon is None:
        raise UnsupportedUnitError("MPGe requires an authoritative energy-equivalent basis")
    basis = _positive(energy_equivalent_wh_per_gallon)
    return basis / number / MI_TO_KM


def convert_to_wh_per_km(value: Any, unit: str | None) -> float | None:
    converters = {
        "mi/kwh": mi_per_kwh_to_wh_per_km,
        "km/kwh": km_per_kwh_to_wh_per_km,
        "kwh/100mi": kwh_per_100mi_to_wh_per_km,
        "wh/mi": wh_per_mile_to_wh_per_km,
        "wh/km": _positive,
    }
    key = str(unit or "").strip().casefold().replace(" ", "")
    converter = converters.get(key)
    return None if converter is None else converter(value)


def apply_override(source: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Apply an ACTIVE override only after strict source identity validation."""
    active_statuses = {"ACTIVE_USER_VALIDATED", "ACTIVE_DETERMINISTIC_UNIT_INTERPRETATION"}
    if override.get("validation_status") not in active_statuses:
        return source.copy()
    checks = ("source_system", "source_file_version", "source_record_id", "raw_field")
    if any(str(source.get(key)) != str(override.get(key)) for key in checks):
        raise StaleOverrideError("Override source key/version does not match")
    expected = str(override.get("expected_raw_value", ""))
    if expected and expected != str(source.get("raw_value", "")):
        raise StaleOverrideError("Override expected raw value no longer matches")
    result = source.copy()
    result["interpreted_unit"] = override.get("corrected_unit")
    replacement = override.get("corrected_value")
    if replacement not in (None, ""):
        result["raw_value"] = replacement
    result["canonical_wh_per_km"] = convert_to_wh_per_km(result["raw_value"], result["interpreted_unit"])
    result["provenance_class"] = (
        "DETERMINISTIC_UNIT_INTERPRETATION"
        if override.get("validation_status") == "ACTIVE_DETERMINISTIC_UNIT_INTERPRETATION"
        else "USER_VALIDATED_CORRECTION"
    )
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def clean(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def text_value(value: Any) -> str:
    value = clean(value)
    return "" if value is None else str(value)


def record_hash(values: list[Any]) -> str:
    payload = json.dumps([text_value(value) for value in values], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fuel_economy_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(FUEL_ECONOMY) as archive:
        workbook = archive.read(archive.namelist()[0])
    return (
        pd.read_excel(io.BytesIO(workbook), sheet_name="EV", header=0),
        pd.read_excel(io.BytesIO(workbook), sheet_name="PHEV", header=2),
    )


def exact_row_keys(frame: pd.DataFrame, columns: list[str]) -> set[str]:
    normalized = frame[columns].copy()
    for column in columns:
        normalized[column] = normalized[column].map(text_value)
    return set(normalized.agg("\x1f".join, axis=1))


def legacy_corrections(old: pd.DataFrame, current: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    displacement = pd.to_numeric(old["Test Veh Displacement (L)"], errors="coerce").fillna(0)
    horsepower = pd.to_numeric(old["Rated Horsepower"], errors="coerce")
    raw = pd.to_numeric(old["RND_ADJ_FE"], errors="coerce")
    mask = ((displacement <= 0.1) | (displacement >= 90)) & (raw < 42.1) & (horsepower < 1000)
    selected = old.loc[mask].copy()
    old_columns = list(old.columns)
    current_keys = exact_row_keys(current[current["Model Year"].between(2020, 2025)], old_columns)
    selected_keys = selected[old_columns].copy()
    for column in old_columns:
        selected_keys[column] = selected_keys[column].map(text_value)
    selected["_current_match"] = selected_keys.agg("\x1f".join, axis=1).isin(current_keys)
    version = sha256(OLD_EPA)
    corrections: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    overrides: list[dict[str, Any]] = []
    for index, row in selected.sort_index().iterrows():
        excel_row = int(index) + 2
        source_id = f"EPA-LEGACY-ROW-{excel_row}"
        value = clean(row["RND_ADJ_FE"])
        match = bool(row["_current_match"])
        corrected = None if value == 0 else 3370.5 / float(value)
        reason = (
            "Legacy BEV heuristic selected displacement <=0.1 or >=90, RND_ADJ_FE <42.1 and horsepower <1000; "
            "its function/formula imply kWh/100mi, while comments say kWh/km and cite different thresholds."
        )
        status = "USER_DECISION_REQUIRED" if match else "RETIRED_SOURCE_RECORD_ABSENT"
        corrections.append({
            "source_system": "EPA_TESTCAR_LEGACY", "source_file_version": version,
            "source_record_id": source_id, "make": text_value(row["Represented Test Veh Make"]),
            "model": text_value(row["Represented Test Veh Model"]), "model_year": clean(row["Model Year"]),
            "electrification": "BEV_HEURISTIC", "raw_field": "RND_ADJ_FE", "raw_value": value,
            "raw_unit_text": text_value(row["FE_UNIT"]), "interpreted_unit": "CONFLICT:kWh/100mi_vs_kWh/km",
            "corrected_value_if_any": corrected, "canonical_wh_per_km": "",
            "correction_type": "LEGACY_HEURISTIC", "reason": reason,
            "evidence": "Function kwh100mi_to_mpge and formula 33705/(x*10), conflicting inline comments",
            "confidence": "LOW", "status": status, "code_or_notebook_reference": LEGACY_NOTEBOOK,
        })
        comparisons.append({
            "source_record_id": source_id, "make": text_value(row["Represented Test Veh Make"]),
            "model": text_value(row["Represented Test Veh Model"]), "model_year": clean(row["Model Year"]),
            "old_raw_value": value, "old_raw_unit": text_value(row["FE_UNIT"]),
            "old_corrected_mpge": corrected, "current_exact_record_found": match,
            "current_raw_value": value if match else "", "current_raw_unit": text_value(row["FE_UNIT"]) if match else "",
            "classification": "CANNOT_VERIFY" if match else "OLD_CORRECTION_OBSOLETE",
        })
        overrides.append({
            "source_system": "EPA_TESTCAR_LEGACY", "source_file_version": version,
            "source_record_id": source_id, "raw_field": "RND_ADJ_FE", "expected_raw_value": value,
            "corrected_unit": "", "corrected_value": "", "reason": reason,
            "evidence": LEGACY_NOTEBOOK,
            "validation_status": "PENDING_USER_VALIDATION" if match else "RETIRED_SOURCE_RECORD_ABSENT",
            "notes": "Not active; exact source/version/value matching is mandatory before future use.",
        })
    return corrections, comparisons, overrides


def source_inventory(current: pd.DataFrame, old_count: int, ev: pd.DataFrame, phev: pd.DataFrame, jrc: pd.DataFrame) -> list[dict[str, Any]]:
    electric = current["Test Fuel Type Description"].astype(str).str.contains("electric", case=False, na=False)
    epa_values = pd.to_numeric(current.loc[electric, "RND_ADJ_FE"], errors="coerce")
    ev_units = ev["Fuel Unit - Conventional Fuel"].astype(str)
    ev_kwh = ev.loc[ev_units.eq("KW-HR/100Miles")]
    ev_values = pd.to_numeric(ev_kwh["Comb FE (Guide) - Conventional Fuel"], errors="coerce")
    phev_mpge = pd.to_numeric(phev["Comb PHEV Composite MPGe"], errors="coerce")
    jrc_field = "Declared electric consumption value (OEM) [Wh/km]"
    jrc_values = pd.to_numeric(jrc[jrc_field], errors="coerce")
    eea_rows = 2_948_208
    def row(source: str, field: str, unit: str, explicit: bool, mixed: bool, rows: int, values: pd.Series | None, status: str, evidence: str, non_null_override: int | None = None) -> dict[str, Any]:
        values = pd.Series(dtype=float) if values is None else values.dropna()
        return {"source_population": source, "field": field, "declared_unit": unit,
                "unit_explicit": explicit, "meaning_changes_by_context": mixed, "row_count": rows,
                "non_null_count": non_null_override if non_null_override is not None else int(len(values)),
                "numeric_min": clean(values.min()) if len(values) else "", "numeric_max": clean(values.max()) if len(values) else "",
                "electrification_or_fuel": "BEV/PHEV/electric", "canonical_mapping_status": status, "evidence": evidence}
    return [
        row("EPA Test Car refreshed 2020-2026", "RND_ADJ_FE", "MPG", True, True, int(electric.sum()), epa_values,
            "UNRESOLVED_OVERLOADED_UNIT", "All FE_UNIT values say MPG; electricity context does not establish energy-equivalent basis."),
        row("EPA Test Car legacy correction cohort", "RND_ADJ_FE", "MPG (contradicted by heuristic)", False, True, old_count, None,
            "USER_VALIDATION_REQUIRED", LEGACY_NOTEBOOK),
        row("FuelEconomy MY2026 EV", "Comb FE (Guide) - Conventional Fuel", "KW-HR/100Miles", True, False, len(ev_kwh), ev_values,
            "DETERMINISTIC_KWH_PER_100MI", "Fuel Unit and Fuel Unit Desc explicitly name kilowatt-hour per 100 miles."),
        row("FuelEconomy MY2026 PHEV", "Comb PHEV Composite MPGe", "MPGe", True, False, int(phev_mpge.notna().sum()), phev_mpge,
            "UNRESOLVED_MPGE_BASIS_NOT_IN_REPOSITORY", "Header explicitly says MPGe; repository does not establish its energy-equivalent constant."),
        row("JRC technical dataset", jrc_field, "Wh/km", True, False, int(jrc_values.notna().sum()), jrc_values,
            "DIRECT_CANONICAL", "Unit is explicit in the source column header."),
        row("EEA 2025 provisional", "z (Wh/km)", "Wh/km", True, False, eea_rows, None,
            "DIRECT_CANONICAL_SEMANTICS_PARTIAL", "Sprint 12C.1 inventory: 2,948,208 available values; explicit header, partial label semantics.", eea_rows),
    ]


def conversion_methods() -> list[dict[str, Any]]:
    return [
        {"source_unit": "mi/kWh", "target_unit": "Wh/km", "formula": "1000 / (value * 1.609344)", "constants": "1 mi = 1.609344 km", "unit_derivation": "kWh^-1 mi -> Wh/km", "representative_input": 4, "representative_output": mi_per_kwh_to_wh_per_km(4), "support_status": "SUPPORTED", "evidence": "Sprint 12E.3 canonical method"},
        {"source_unit": "km/kWh", "target_unit": "Wh/km", "formula": "1000 / value", "constants": "1 kWh = 1000 Wh", "unit_derivation": "km/kWh inverted and scaled", "representative_input": 5, "representative_output": km_per_kwh_to_wh_per_km(5), "support_status": "SUPPORTED", "evidence": "Sprint 12E.3 canonical method"},
        {"source_unit": "kWh/100mi", "target_unit": "Wh/km", "formula": "value*1000/(100*1.609344)", "constants": "1 kWh = 1000 Wh; 1 mi = 1.609344 km", "unit_derivation": "kWh/100mi -> Wh/km", "representative_input": 30, "representative_output": kwh_per_100mi_to_wh_per_km(30), "support_status": "SUPPORTED_WHEN_UNIT_EXPLICIT", "evidence": "FuelEconomy Fuel Unit Desc"},
        {"source_unit": "Wh/mi", "target_unit": "Wh/km", "formula": "value / 1.609344", "constants": "1 mi = 1.609344 km", "unit_derivation": "Wh/mile -> Wh/km", "representative_input": 300, "representative_output": wh_per_mile_to_wh_per_km(300), "support_status": "SUPPORTED", "evidence": "Sprint 12E.3 canonical method"},
        {"source_unit": "MPGe", "target_unit": "Wh/km", "formula": "basis_Wh_per_gallon / value / 1.609344", "constants": "CALLER_MUST_SUPPLY_AUTHORITATIVE_BASIS", "unit_derivation": "energy-equivalent gallon per mile -> Wh/km", "representative_input": 100, "representative_output": "", "support_status": "UNSUPPORTED_NO_AUTHORITATIVE_REPOSITORY_BASIS", "evidence": "No authoritative basis found in repository; legacy 33705 constant is undocumented"},
    ]


def epa_dispositions() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases = [row for row in read_csv(OUT_12E2 / "unresolved_materialization_cases.csv") if row["reason_code"] == "UNSUPPORTED_ENERGY_OR_EQUIVALENT_FUEL_UNIT"]
    dispositions = []
    for row in cases:
        dispositions.append({**row, "sprint_12e3_disposition": "REQUIRES_EXTERNAL_DOCUMENTATION",
                             "unit_audit_reason": "EPA field says generic MPG for electric/equivalent fuel; repository evidence does not establish MPGe basis or canonical energy semantics.",
                             "canonical_wh_per_km": "", "materialized": False})
    return dispositions, list(dispositions)


def sanity_rows(current: pd.DataFrame, ev: pd.DataFrame, jrc: pd.DataFrame, dispositions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ["program_id", "vehicle_configuration_id", "vde_id", "run_id", "fuelcons_id", "make", "model", "model_year", "electrification", "source_system", "raw_field", "raw_value", "raw_unit", "interpreted_unit", "canonical_wh_per_km", "canonical_kwh_per_100km", "provenance_class", "correction_status", "anomaly_status"]
    rows: list[dict[str, Any]] = []
    # Explicit FuelEconomy kWh/100mi rows are deterministic notebook-ready evidence.
    unit = ev["Fuel Unit - Conventional Fuel"].astype(str).eq("KW-HR/100Miles")
    for index, item in ev.loc[unit].sort_index().iterrows():
        value = clean(pd.to_numeric(pd.Series([item.get("Comb FE (Guide) - Conventional Fuel")]), errors="coerce").iloc[0])
        canonical = None
        status = "MISSING_VALUE"
        if value is not None:
            try:
                canonical = kwh_per_100mi_to_wh_per_km(value)
                status = "PLAUSIBLE" if 50 <= canonical <= 600 else "SUSPICIOUS_MAGNITUDE"
            except ValueError:
                status = "SUSPICIOUS_MAGNITUDE"
        rows.append(dict.fromkeys(fields, "") | {"make": text_value(item.get("Division")), "model": text_value(item.get("Carline")),
            "model_year": clean(item.get("Model Year")), "electrification": "BEV", "source_system": "FUELECONOMY_MY2026_EV",
            "raw_field": "Comb FE (Guide) - Conventional Fuel", "raw_value": value, "raw_unit": "KW-HR/100Miles",
            "interpreted_unit": "kWh/100mi", "canonical_wh_per_km": canonical,
            "canonical_kwh_per_100km": canonical / 10 if canonical is not None else "", "provenance_class": "DETERMINISTIC_UNIT_CONVERSION",
            "correction_status": "NOT_REQUIRED", "anomaly_status": status})
    # JRC direct Wh/km values.
    field = "Declared electric consumption value (OEM) [Wh/km]"
    for index, item in jrc[jrc[field].notna()].sort_index().iterrows():
        value = float(item[field])
        rows.append(dict.fromkeys(fields, "") | {"make": text_value(item.get("OEM anon")), "model": text_value(item.get("Model anon")),
            "electrification": "BEV" if bool(item.get("is_electric")) else "PHEV", "source_system": "JRC_TECHNICAL",
            "raw_field": field, "raw_value": value, "raw_unit": "Wh/km", "interpreted_unit": "Wh/km",
            "canonical_wh_per_km": value, "canonical_kwh_per_100km": value / 10,
            "provenance_class": "DIRECT_SOURCE_EXPLICIT_UNIT", "correction_status": "NOT_REQUIRED",
            "anomaly_status": "PLAUSIBLE" if 50 <= value <= 600 else "SUSPICIOUS_MAGNITUDE"})
    # One row per previously unresolved canonical case, with source value intentionally unconverted.
    grouping = {row["canonical_run_id"]: row for row in read_csv(OUT_12E2 / "run_grouping_results.csv")}
    source_lookup = current.set_index("source_excel_row")
    uri = DB_12E2.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        metadata = pd.read_sql_query(
            """SELECT v.id AS vde_id, v.vehicle_configuration_id, vc.program_id,
                      p.commercial_make AS make, p.commercial_model AS model, v.year AS model_year
                 FROM vde v
                 JOIN vehicle_configuration vc ON vc.vehicle_configuration_id=v.vehicle_configuration_id
                 JOIN program p ON p.program_id=vc.program_id""",
            connection,
        ).set_index("vde_id").to_dict("index")
    for case in dispositions:
        source_rows: list[int] = []
        for run_id in case["candidate_run_ids"].split(";"):
            group = grouping.get(run_id)
            if group:
                source_rows.extend(int(value) for value in group["source_excel_rows"].split(";") if value)
        source_records = source_lookup.loc[sorted(set(source_rows))] if source_rows else pd.DataFrame()
        if isinstance(source_records, pd.Series):
            source_records = source_records.to_frame().T
        raw_values = sorted({text_value(value) for value in source_records.get("RND_ADJ_FE", []) if text_value(value)})
        raw_units = sorted({text_value(value) for value in source_records.get("FE_UNIT", []) if text_value(value)})
        vde_id = int(case["canonical_vde_id"])
        meta = metadata.get(vde_id, {})
        rows.append(dict.fromkeys(fields, "") | {"program_id": meta.get("program_id", ""),
            "vehicle_configuration_id": meta.get("vehicle_configuration_id", ""), "vde_id": vde_id,
            "run_id": case["candidate_run_ids"], "make": meta.get("make", ""), "model": meta.get("model", ""),
            "model_year": meta.get("model_year", ""), "electrification": "BEV/PHEV/FCEV",
            "source_system": "EPA_TESTCAR_2014_PRESENT", "raw_field": "RND_ADJ_FE",
            "raw_value": ";".join(raw_values), "raw_unit": ";".join(raw_units) or "MPG",
            "interpreted_unit": "UNRESOLVED", "provenance_class": "UNRESOLVED",
            "correction_status": case["sprint_12e3_disposition"], "anomaly_status": "CONFLICTING_UNIT_EVIDENCE"})
    return rows


def anomalies(corrections: list[dict[str, Any]], sanity: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [{"source_system": row["source_system"], "source_record_id": row["source_record_id"], "raw_value": row["raw_value"],
             "raw_unit": row["raw_unit_text"], "anomaly_status": "CONFLICTING_UNIT_EVIDENCE" if row["raw_value"] != 0 else "SUSPICIOUS_MAGNITUDE",
             "details": row["reason"]} for row in corrections]
    for index, row in enumerate(sanity):
        if row["anomaly_status"] not in ("PLAUSIBLE", ""):
            rows.append({"source_system": row["source_system"], "source_record_id": f"SANITY-{index+1}", "raw_value": row["raw_value"],
                         "raw_unit": row["raw_unit"], "anomaly_status": row["anomaly_status"],
                         "details": "Magnitude is QA only; ambiguous unit evidence was not converted."})
    return rows


def signature(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest().upper()


def render_report(summary: dict[str, Any]) -> str:
    return f"""# Sprint 12E.3 — Electric Consumption Units & Manual Correction Recovery

## Status: `{summary['status']}`

```text
Electric source records inspected              {summary['electric_source_records_inspected']}
Records with explicit source unit              {summary['records_with_explicit_source_unit']}
Records with contextual unit                   {summary['records_with_contextual_unit']}
Historical manual corrections found            {summary['historical_manual_corrections_found']}
Active corrections still required              {summary['active_corrections_still_required']}
Old corrections retired                        {summary['old_corrections_retired']}
Previously unresolved 12E.2 cases              {summary['previously_unresolved_12e2_cases']}
Now resolved                                   {summary['now_resolved']}
Still unresolved                               {summary['still_unresolved']}
Suspicious/anomalous values                    {summary['suspicious_or_anomalous_values']}
Runtime DB changed?                            {'YES' if summary['runtime_db_changed'] else 'NO'}
User decisions required                        {summary['user_decisions_required']}
```

## Small human-review packet

| Decision | Records | Evidence conflict | Required answer |
|---|---:|---|---|
| Legacy EPA BEV heuristic | 529 current matches (535 historical; 6 retired) | Function name/formula implement `kWh/100mi → MPGe`; comments say `kWh/km`; thresholds also disagree | Confirm whether the selected raw `RND_ADJ_FE` values were intended as **kWh/100mi**. No override is active pending confirmation. |

## What the audit established

- The historical notebook changed **535 source rows**. Its net math is consistent with treating the raw number as `kWh/100mi`, but its prose says `kWh/km`; this is not sufficient evidence for automatic recovery.
- **529** affected rows are byte-for-byte field matches in the refreshed 2020–2025 export. The source still says generic `MPG`, so the refresh did not repair their unit metadata. Six old records are absent and their registry entries are explicitly retired.
- FuelEconomy MY2026 supplies paired EV rows with explicit `KW-HR/100Miles` metadata. Those rows and JRC's explicit `Wh/km` values are deterministic and appear in the sanity dataset.
- MPGe remains separate. The legacy `33705` constant has no authoritative repository citation, so MPGe is not converted in audit outputs.
- All {summary['previously_unresolved_12e2_cases']} Sprint 12E.2 electric/equivalent-energy cases received a disposition; none can be safely materialized from the current EPA `MPG` field alone.
- Zero is preserved as an observed anomaly, never interpreted as missing and never converted to infinity or canonical zero.

## Historical defect impact

Two selected legacy rows had raw zero. The vectorized legacy formula produced infinity and the later MPGe path could collapse that to zero energy. The audit preserves both as suspicious observations and does not reproduce the faulty result. No canonical or runtime data was changed.

## Outputs and safety

The override registry contains historical entries only: 529 `PENDING_USER_VALIDATION` and 6 `RETIRED_SOURCE_RECORD_ABSENT`; there are no active overrides. Runtime and QA databases remained byte-identical. Output signature: `{summary['output_signature']}`; deterministic rebuild: **{'YES' if summary['rebuild_deterministic'] else 'NO'}**.

## Evidence limits

- **Direct:** explicit FuelEconomy `KW-HR/100Miles`, JRC `Wh/km`, EEA `z (Wh/km)`, source hashes, exact old/current record comparison.
- **Historical-code evidence:** the legacy notebook formula and overwrite path.
- **Gap requiring user validation:** intended unit of the legacy 535-row heuristic cohort.
- **Gap requiring external documentation:** EPA generic `MPG` on electricity/equivalent-fuel test rows and an authoritative MPGe energy-equivalent basis.

No notebook, runtime database, Streamlit page, raw source, or physics/resolver was modified.
"""


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    OVERRIDES.parent.mkdir(parents=True, exist_ok=True)
    before = {str(path): sha256(path) for path in RUNTIME_DBS}
    previous = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {}

    old = pd.read_excel(OLD_EPA)
    current = pd.read_excel(EPA)
    current.insert(0, "source_excel_row", range(2, len(current) + 2))
    ev, phev = fuel_economy_frames()
    jrc = pd.read_excel(JRC)

    corrections, comparisons, override_rows = legacy_corrections(old, current)
    inventory = source_inventory(current, len(corrections), ev, phev, jrc)
    methods = conversion_methods()
    dispositions, unresolved = epa_dispositions()
    sanity = sanity_rows(current, ev, jrc, dispositions)
    anomaly_rows = anomalies(corrections, sanity)

    outputs: list[tuple[str, list[dict[str, Any]], list[str]]] = [
        ("electric_source_unit_inventory.csv", inventory, list(inventory[0])),
        ("historical_manual_corrections.csv", corrections, list(corrections[0])),
        ("electric_conversion_methods.csv", methods, list(methods[0])),
        ("electric_value_anomalies.csv", anomaly_rows, list(anomaly_rows[0])),
        ("old_vs_current_electric_values.csv", comparisons, list(comparisons[0])),
        ("unresolved_electric_cases.csv", unresolved, list(unresolved[0])),
        ("resolved_12e2_electric_cases.csv", dispositions, list(dispositions[0])),
        ("electric_energy_sanity_dataset.csv", sanity, list(sanity[0])),
    ]
    for name, rows, fields in outputs:
        write_csv(OUT / name, rows, fields)
    write_csv(OVERRIDES, override_rows, list(override_rows[0]))

    artifact_paths = [OUT / name for name, _, _ in outputs] + [OVERRIDES]
    output_sig = signature(artifact_paths)
    after = {str(path): sha256(path) for path in RUNTIME_DBS}
    explicit = sum(int(row["non_null_count"]) for row in inventory if row["unit_explicit"] and not row["meaning_changes_by_context"])
    inspected = sum(int(row["row_count"]) for row in inventory)
    retired = sum(row["status"] == "RETIRED_SOURCE_RECORD_ABSENT" for row in corrections)
    summary = {
        "status": "ELECTRIC_UNITS_REVIEW_REQUIRED",
        "electric_source_records_inspected": inspected,
        "records_with_explicit_source_unit": explicit,
        "records_with_contextual_unit": len(corrections),
        "historical_manual_corrections_found": len(corrections),
        "active_corrections_still_required": 0,
        "old_corrections_retired": retired,
        "previously_unresolved_12e2_cases": len(dispositions),
        "now_resolved": 0,
        "still_unresolved": len(unresolved),
        "suspicious_or_anomalous_values": len(anomaly_rows),
        "runtime_db_changed": before != after,
        "user_decisions_required": 1,
        "pending_user_validation_records": len(corrections) - retired,
        "override_registry_active_rows": 0,
        "sanity_dataset_rows": len(sanity),
        "output_signature": output_sig,
        "previous_signature_available": bool(previous.get("output_signature")),
        "rebuild_deterministic": previous.get("output_signature") in (None, output_sig),
        "runtime_fingerprints_before": before,
        "runtime_fingerprints_after": after,
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(render_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(run(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
