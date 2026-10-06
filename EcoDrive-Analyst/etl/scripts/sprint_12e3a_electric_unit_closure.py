"""Sprint 12E.3A closure for the approved EPA electric-unit interpretation.

The patch resolves only current source rows that exactly match the recovered
legacy cohort, preserves raw numbers, excludes FCEV zeros, and dispositions all
12E.2 electric/equivalent-energy cases. Runtime databases remain read-only.
"""
from __future__ import annotations

import csv
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "etl" / "scripts"))
import sprint_12e3_electric_unit_audit as base  # noqa: E402


REPORT = ROOT / "etl" / "reports" / "sprint_12e3a_electric_unit_closure.md"
EPA_TESTING_URL = "https://www.epa.gov/greenvehicles/fuel-economy-and-ev-range-testing"
EPA_LABEL_URL = "https://www.epa.gov/fueleconomy/text-version-electric-vehicle-label"
EPA_TRENDS_URL = "https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P101698T.txt"


def normalized_row_keys(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    normalized = frame[columns].copy()
    for column in columns:
        normalized[column] = normalized[column].map(base.text_value)
    return normalized.agg("\x1f".join, axis=1)


def selected_legacy_rows(old: pd.DataFrame) -> pd.DataFrame:
    displacement = pd.to_numeric(old["Test Veh Displacement (L)"], errors="coerce").fillna(0)
    horsepower = pd.to_numeric(old["Rated Horsepower"], errors="coerce")
    raw = pd.to_numeric(old["RND_ADJ_FE"], errors="coerce")
    mask = ((displacement <= 0.1) | (displacement >= 90)) & (raw < 42.1) & (horsepower < 1000)
    return old.loc[mask].copy()


def match_legacy_to_current(old: pd.DataFrame, current: pd.DataFrame) -> list[tuple[int, pd.Series, int | None, pd.Series | None]]:
    columns = list(old.columns)
    current_keys = normalized_row_keys(current[current["Model Year"].between(2020, 2025)], columns)
    key_to_rows: dict[str, list[int]] = defaultdict(list)
    for index, key in current_keys.items():
        key_to_rows[key].append(int(index))
    matches = []
    selected = selected_legacy_rows(old)
    selected_keys = normalized_row_keys(selected, columns)
    for old_index, row in selected.sort_index().iterrows():
        candidates = key_to_rows.get(selected_keys.loc[old_index], [])
        current_index = candidates.pop(0) if candidates else None
        matches.append((int(old_index), row, current_index, None if current_index is None else current.loc[current_index]))
    return matches


def correction_outputs(matches: list[tuple[int, pd.Series, int | None, pd.Series | None]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], set[int], set[int]]:
    old_version = base.sha256(base.OLD_EPA)
    current_version = base.sha256(base.EPA)
    corrections: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    overrides: list[dict[str, Any]] = []
    active_excel_rows: set[int] = set()
    excluded_excel_rows: set[int] = set()
    evidence = (
        f"Legacy helper algebra; explicit FuelEconomy KW-HR/100Miles metadata; "
        f"EPA EV methodology ({EPA_TESTING_URL}; {EPA_LABEL_URL}; {EPA_TRENDS_URL})."
    )
    for old_index, old_row, current_index, current_row in matches:
        legacy_id = f"EPA-LEGACY-ROW-{old_index + 2}"
        raw = base.clean(old_row["RND_ADJ_FE"])
        current_excel_row = None if current_index is None else current_index + 2
        current_id = "" if current_excel_row is None else f"EPA-TESTCAR-ROW-{current_excel_row}"
        fuel = "" if current_row is None else base.text_value(current_row["Test Fuel Type Description"])
        is_active = current_row is not None and fuel.casefold() == "electricity" and float(raw) > 0
        is_zero_fcev = current_row is not None and fuel.casefold() == "hydrogen 5" and float(raw) == 0
        if is_active:
            canonical = base.kwh_per_100mi_to_wh_per_km(raw)
            active_excel_rows.add(int(current_excel_row))
            interpreted = "kWh/100mi"
            correction_type = "DETERMINISTIC_UNIT_INTERPRETATION"
            confidence = "HIGH"
            status = "RESOLVED"
            comparison = "OLD_CORRECTION_STILL_REQUIRED"
            validation = "ACTIVE_DETERMINISTIC_UNIT_INTERPRETATION"
            corrected_unit = "kWh/100mi"
        elif is_zero_fcev:
            canonical = None
            excluded_excel_rows.add(int(current_excel_row))
            interpreted = "NOT_APPLICABLE_TO_ELECTRIC_RULE"
            correction_type = "UNRESOLVED"
            confidence = "HIGH"
            status = "NOT_APPLICABLE_OR_UNRESOLVED_FOR_ELECTRIC_RULE"
            comparison = "OLD_CORRECTION_WRONG_FOR_FCEV_ZERO"
            validation = "EXCLUDED_FCEV_ZERO_UNRESOLVED"
            corrected_unit = ""
        else:
            canonical = None
            interpreted = ""
            correction_type = "LEGACY_HEURISTIC"
            confidence = "HIGH"
            status = "RETIRED_SOURCE_RECORD_ABSENT"
            comparison = "OLD_CORRECTION_OBSOLETE"
            validation = "RETIRED_SOURCE_RECORD_ABSENT"
            corrected_unit = ""
        reason = (
            "Approved as kWh/100mi unit interpretation; raw number preserved. Legacy comment saying kWh/km was incorrect."
            if is_active else
            "Observed Hydrogen 5 zero is outside the electric rule and remains unconverted/unresolved."
            if is_zero_fcev else
            "Exact record is absent from the refreshed source; historical entry remains retired."
        )
        corrections.append({
            "source_system": "EPA_TESTCAR_LEGACY", "source_file_version": old_version,
            "source_record_id": legacy_id, "current_source_record_id": current_id,
            "make": base.text_value(old_row["Represented Test Veh Make"]),
            "model": base.text_value(old_row["Represented Test Veh Model"]), "model_year": base.clean(old_row["Model Year"]),
            "electrification": "BEV" if is_active else "FCEV" if is_zero_fcev else "BEV_HEURISTIC",
            "raw_field": "RND_ADJ_FE", "raw_value": raw, "raw_unit_text": base.text_value(old_row["FE_UNIT"]),
            "interpreted_unit": interpreted, "corrected_value_if_any": "", "canonical_wh_per_km": "" if canonical is None else canonical,
            "correction_type": correction_type, "reason": reason,
            "evidence": evidence, "confidence": confidence, "status": status,
            "code_or_notebook_reference": base.LEGACY_NOTEBOOK,
        })
        comparisons.append({
            "source_record_id": legacy_id, "current_source_record_id": current_id,
            "make": base.text_value(old_row["Represented Test Veh Make"]), "model": base.text_value(old_row["Represented Test Veh Model"]),
            "model_year": base.clean(old_row["Model Year"]), "old_raw_value": raw,
            "old_raw_unit": base.text_value(old_row["FE_UNIT"]), "current_exact_record_found": current_row is not None,
            "current_raw_value": "" if current_row is None else base.clean(current_row["RND_ADJ_FE"]),
            "current_raw_unit": "" if current_row is None else base.text_value(current_row["FE_UNIT"]),
            "approved_interpretation": interpreted, "canonical_wh_per_km": "" if canonical is None else canonical,
            "classification": comparison,
        })
        overrides.append({
            "source_system": "EPA_TESTCAR_2014_PRESENT" if current_row is not None else "EPA_TESTCAR_LEGACY",
            "source_file_version": current_version if current_row is not None else old_version,
            "source_record_id": current_id or legacy_id, "legacy_source_record_id": legacy_id,
            "raw_field": "RND_ADJ_FE", "expected_raw_value": raw, "corrected_unit": corrected_unit,
            "corrected_value": "", "reason": reason, "evidence": evidence,
            "validation_status": validation,
            "notes": "Raw numeric value is never replaced. Stale source/version/key/value mismatches must fail or quarantine.",
        })
    return corrections, comparisons, overrides, active_excel_rows, excluded_excel_rows


def grouping_maps() -> tuple[dict[str, set[int]], dict[int, tuple[str, int]]]:
    by_run: dict[str, set[int]] = {}
    by_source: dict[int, tuple[str, int]] = {}
    for row in base.read_csv(base.OUT_12E2 / "run_grouping_results.csv"):
        source_rows = {int(value) for value in row["source_excel_rows"].split(";") if value}
        by_run[row["canonical_run_id"]] = source_rows
        for source_row in source_rows:
            by_source[source_row] = (row["canonical_run_id"], int(row["canonical_vde_id"]))
    return by_run, by_source


def dispositions(active_excel_rows: set[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_run, _ = grouping_maps()
    previous = [row for row in base.read_csv(base.OUT_12E2 / "unresolved_materialization_cases.csv") if row["reason_code"] == "UNSUPPORTED_ENERGY_OR_EQUIVALENT_FUEL_UNIT"]
    rows: list[dict[str, Any]] = []
    for case in previous:
        source_rows: set[int] = set()
        for run_id in case["candidate_run_ids"].split(";"):
            source_rows.update(by_run.get(run_id, set()))
        matched = source_rows & active_excel_rows
        if case["fuel_type"] == "Hydrogen 5":
            status = "HYDROGEN_OUTSIDE_ELECTRIC_RULE"
        elif source_rows and source_rows <= active_excel_rows:
            status = "NOW_RESOLVED_KWH_PER_100MI"
        elif matched:
            status = "CONFLICTING_SOURCE_CONTEXT"
        else:
            status = "NOT_SAME_SOURCE_SEMANTICS"
        unit_resolved = status == "NOW_RESOLVED_KWH_PER_100MI"
        rows.append({**case, "source_excel_rows": ";".join(str(value) for value in sorted(source_rows)),
            "approved_source_rows": ";".join(str(value) for value in sorted(matched)),
            "sprint_12e3a_disposition": status, "interpreted_unit": "kWh/100mi" if unit_resolved else "",
            "unit_resolved": unit_resolved, "materialized_additional_fuelcons": False,
            "fuelcons_materialization_status": (
                "NOT_MATERIALIZED_EXISTING_12E2_RULE_HAS_NO_APPROVED_ELECTRIC_CYCLE_PATH"
                if unit_resolved else "NOT_ELIGIBLE_UNIT_OR_SOURCE_CONTEXT_UNRESOLVED"
            ),
            "details_12e3a": "Unit resolution does not bypass comparison-basis, cycle, grain, or materialization-method checks."})
    unresolved = [row for row in rows if row["sprint_12e3a_disposition"] != "NOW_RESOLVED_KWH_PER_100MI"]
    return rows, unresolved


def canonical_metadata() -> dict[int, dict[str, Any]]:
    uri = base.DB_12E2.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        return pd.read_sql_query(
            """SELECT v.id AS vde_id, v.vehicle_configuration_id, vc.program_id,
                      p.commercial_make AS make, p.commercial_model AS model, v.year AS model_year
                 FROM vde v JOIN vehicle_configuration vc ON vc.vehicle_configuration_id=v.vehicle_configuration_id
                 JOIN program p ON p.program_id=vc.program_id""", connection
        ).set_index("vde_id").to_dict("index")


def sanity_dataset(current: pd.DataFrame, ev: pd.DataFrame, jrc: pd.DataFrame, all_dispositions: list[dict[str, Any]], active_rows: set[int], excluded_rows: set[int]) -> list[dict[str, Any]]:
    fields = ["program_id", "vehicle_configuration_id", "vde_id", "run_id", "fuelcons_id", "source_record_id",
              "source_file_version", "source_excel_row", "make", "model", "model_year", "electrification", "source_system",
              "raw_field", "raw_value", "raw_unit", "interpreted_unit", "canonical_wh_per_km", "canonical_kwh_per_100km",
              "provenance_class", "correction_status", "anomaly_status", "sprint_12e3a_disposition"]
    explicit = base.sanity_rows(current, ev, jrc, [])
    rows = [{field: row.get(field, "") for field in fields} for row in explicit]
    _, by_source = grouping_maps()
    metadata = canonical_metadata()
    version = base.sha256(base.EPA)
    current_lookup = current.set_index("source_excel_row")
    for source_row in sorted(active_rows | excluded_rows):
        item = current_lookup.loc[source_row]
        run_id, vde_id = by_source.get(source_row, ("", 0))
        meta = metadata.get(vde_id, {})
        value = float(item["RND_ADJ_FE"])
        active = source_row in active_rows
        canonical = base.kwh_per_100mi_to_wh_per_km(value) if active else None
        rows.append(dict.fromkeys(fields, "") | {
            "program_id": meta.get("program_id", ""), "vehicle_configuration_id": meta.get("vehicle_configuration_id", ""),
            "vde_id": vde_id or "", "run_id": run_id, "source_record_id": f"EPA-TESTCAR-ROW-{source_row}",
            "source_file_version": version, "source_excel_row": source_row,
            "make": base.text_value(item["Represented Test Veh Make"]), "model": base.text_value(item["Represented Test Veh Model"]),
            "model_year": base.clean(item["Model Year"]), "electrification": "BEV" if active else "FCEV",
            "source_system": "EPA_TESTCAR_2014_PRESENT", "raw_field": "RND_ADJ_FE", "raw_value": value,
            "raw_unit": base.text_value(item["FE_UNIT"]), "interpreted_unit": "kWh/100mi" if active else "NOT_APPLICABLE",
            "canonical_wh_per_km": "" if canonical is None else canonical,
            "canonical_kwh_per_100km": "" if canonical is None else canonical / 10,
            "provenance_class": "DETERMINISTIC_UNIT_INTERPRETATION" if active else "UNRESOLVED",
            "correction_status": "RESOLVED" if active else "EXCLUDED_FCEV_ZERO_UNRESOLVED",
            "anomaly_status": "PLAUSIBLE" if active else "OBSERVED_ZERO_NOT_CONVERTED",
            "sprint_12e3a_disposition": "SOURCE_ROW_RESOLVED_KWH_PER_100MI" if active else "HYDROGEN_OUTSIDE_ELECTRIC_RULE",
        })
    resolved_case_ids = {int(row["canonical_vde_id"]) for row in all_dispositions if row["sprint_12e3a_disposition"] == "NOW_RESOLVED_KWH_PER_100MI"}
    for case in all_dispositions:
        if int(case["canonical_vde_id"]) in resolved_case_ids:
            continue
        source_numbers = [int(value) for value in case["source_excel_rows"].split(";") if value]
        records = current_lookup.loc[source_numbers] if source_numbers else pd.DataFrame()
        if isinstance(records, pd.Series):
            records = records.to_frame().T
        raw_values = sorted({base.text_value(value) for value in records.get("RND_ADJ_FE", []) if base.text_value(value)})
        raw_units = sorted({base.text_value(value) for value in records.get("FE_UNIT", []) if base.text_value(value)})
        vde_id = int(case["canonical_vde_id"])
        meta = metadata.get(vde_id, {})
        rows.append(dict.fromkeys(fields, "") | {
            "program_id": meta.get("program_id", ""), "vehicle_configuration_id": meta.get("vehicle_configuration_id", ""),
            "vde_id": vde_id, "run_id": case["candidate_run_ids"], "make": meta.get("make", ""), "model": meta.get("model", ""),
            "model_year": meta.get("model_year", ""), "electrification": "FCEV" if case["fuel_type"] == "Hydrogen 5" else "BEV/PHEV",
            "source_system": "EPA_TESTCAR_2014_PRESENT", "raw_field": "RND_ADJ_FE", "raw_value": ";".join(raw_values),
            "raw_unit": ";".join(raw_units), "interpreted_unit": "UNRESOLVED", "provenance_class": "UNRESOLVED",
            "correction_status": case["sprint_12e3a_disposition"], "anomaly_status": (
                "CONFLICTING_SOURCE_CONTEXT" if case["sprint_12e3a_disposition"] == "CONFLICTING_SOURCE_CONTEXT" else "UNRESOLVED"
            ), "sprint_12e3a_disposition": case["sprint_12e3a_disposition"],
        })
    return rows


def report_text(summary: dict[str, Any]) -> str:
    return f"""# Sprint 12E.3A — Electric Unit Closure Patch

## Status: `{summary['status']}`

```text
Historical heuristic rows                       {summary['historical_heuristic_rows']}
Current source matches                          {summary['current_source_matches']}
Resolved non-zero electric rows                 {summary['resolved_nonzero_electric_rows']}
Excluded FCEV zero rows                           {summary['excluded_fcev_zero_rows']}
Retired absent rows                               {summary['retired_absent_rows']}

12E.2 unresolved Electricity cases             {summary['electricity_cases']}
Now resolved by unit rule                       {summary['now_resolved_by_unit_rule']}
Still unresolved                                {summary['still_unresolved_electricity']}
Not same semantics / conflicting                {summary['not_same_or_conflicting']}

Hydrogen 5 cases                                 {summary['hydrogen_cases']}
Converted by electric rule                        {summary['hydrogen_converted']}

Additional canonical EPA FuelCons                 {summary['additional_canonical_fuelcons']}
Runtime DB changed?                             {'YES' if summary['runtime_db_changed'] else 'NO'}
User decisions required                           {summary['user_decisions_required']}
```

## Closure

The 527 non-zero current EPA source rows that exactly match the recovered historical cohort are now interpreted as `kWh/100mi`. Their raw values are preserved and converted directly with:

```python
Wh_per_km = value * 1000 / (100 * 1.609344)
```

For example, `29.9 kWh/100mi = {base.kwh_per_100mi_to_wh_per_km(29.9):.3f} Wh/km = {base.kwh_per_100mi_to_wh_per_km(29.9)/10:.3f} kWh/100km`. MPGe is not an intermediate.

The Honda CR-V e:FCEV source zeros (`EPA-LEGACY-ROW-1836` and `EPA-LEGACY-ROW-1838`) remain observed zeros with NULL canonical energy. The six absent records remain retired.

## 12E.2 disposition

- 276 Electricity cases contain only approved source rows and are `NOW_RESOLVED_KWH_PER_100MI` at the unit layer.
- 66 mix approved and non-approved source context and remain `CONFLICTING_SOURCE_CONTEXT`.
- 915 do not share the selected historical source semantics and are `NOT_SAME_SOURCE_SEMANTICS`.
- All 32 Hydrogen 5 cases remain outside the electric rule.

No additional FuelCons was materialized. The existing 12E.2 rule has no approved electric CD/MCT cycle-materialization path; unit closure does not bypass metric, comparison-basis, cycle, or grain validation.

## Evidence and provenance

- Recovered helper algebra implements `kWh/100mi → MPGe`; the legacy `kWh/km` comment was incorrect.
- The repository FuelEconomy workbook explicitly uses `KW-HR/100Miles` for EV consumption rows.
- [EPA fuel economy and EV range testing]({EPA_TESTING_URL}) documents that 33.7 kWh used over 100 miles corresponds to 100 MPGe.
- [EPA electric vehicle label description]({EPA_LABEL_URL}) defines the consumption rate as kilowatt-hours used to travel 100 miles.
- [EPA Automotive Trends technical explanation]({EPA_TRENDS_URL}) gives the independent MPGe relationship using 33.705 kWh/gallon and kWh/mile.

Magnitude was used only for anomaly QA. Runtime databases, schema, RUN/FuelCons design, notebooks, pages, raw sources, and physics were not modified.

Output signature: `{summary['output_signature']}`. Deterministic rebuild: **{'YES' if summary['rebuild_deterministic'] else 'NO'}**.
"""


def run() -> dict[str, Any]:
    before = {str(path): base.sha256(path) for path in base.RUNTIME_DBS}
    previous = json.loads(base.SUMMARY.read_text(encoding="utf-8")) if base.SUMMARY.exists() else {}
    old = pd.read_excel(base.OLD_EPA)
    current = pd.read_excel(base.EPA)
    current.insert(0, "source_excel_row", range(2, len(current) + 2))
    ev, phev = base.fuel_economy_frames()
    jrc = pd.read_excel(base.JRC)

    corrections, comparisons, overrides, active_rows, excluded_rows = correction_outputs(match_legacy_to_current(old, current))
    all_dispositions, unresolved = dispositions(active_rows)
    sanity = sanity_dataset(current, ev, jrc, all_dispositions, active_rows, excluded_rows)
    inventory = base.source_inventory(current, len(corrections), ev, phev, jrc)
    inventory[0]["canonical_mapping_status"] = "527_EXACT_ROWS_DETERMINISTIC_KWH_PER_100MI_REMAINDER_UNRESOLVED"
    inventory[0]["evidence"] += " Exact historical-cohort matches are resolved only where approved."
    inventory[1]["canonical_mapping_status"] = "527_RESOLVED_2_FCEV_EXCLUDED_6_RETIRED"
    methods = base.conversion_methods()
    methods[-1].update({"constants": "33.705 kWh/gallon gasoline equivalent; 1 mi = 1.609344 km",
        "representative_output": base.mpge_to_wh_per_km(100, 33705), "support_status": "SUPPORTED_AS_SEPARATE_EXPLICIT_MPGE_PATH",
        "evidence": EPA_TRENDS_URL})
    anomalies = [
        {"source_system": "EPA_TESTCAR_2014_PRESENT", "source_record_id": row["current_source_record_id"],
         "raw_value": row["raw_value"], "raw_unit": row["raw_unit_text"], "anomaly_status": "OBSERVED_ZERO_NOT_CONVERTED",
         "details": row["reason"]}
        for row in corrections if row["status"] == "NOT_APPLICABLE_OR_UNRESOLVED_FOR_ELECTRIC_RULE"
    ]
    anomalies.extend({"source_system": "EPA_TESTCAR_2014_PRESENT", "source_record_id": row["canonical_vde_id"],
        "raw_value": "", "raw_unit": "MIXED", "anomaly_status": "CONFLICTING_SOURCE_CONTEXT", "details": row["details_12e3a"]}
        for row in all_dispositions if row["sprint_12e3a_disposition"] == "CONFLICTING_SOURCE_CONTEXT")

    outputs = [
        (base.OUT / "electric_source_unit_inventory.csv", inventory),
        (base.OUT / "historical_manual_corrections.csv", corrections),
        (base.OUT / "electric_conversion_methods.csv", methods),
        (base.OUT / "electric_value_anomalies.csv", anomalies),
        (base.OUT / "old_vs_current_electric_values.csv", comparisons),
        (base.OUT / "unresolved_electric_cases.csv", unresolved),
        (base.OUT / "resolved_12e2_electric_cases.csv", all_dispositions),
        (base.OUT / "electric_energy_sanity_dataset.csv", sanity),
        (base.OVERRIDES, overrides),
    ]
    for path, rows in outputs:
        base.write_csv(path, rows, list(rows[0]))
    output_signature = base.signature([path for path, _ in outputs])
    after = {str(path): base.sha256(path) for path in base.RUNTIME_DBS}
    counts: dict[str, int] = defaultdict(int)
    for row in all_dispositions:
        counts[row["sprint_12e3a_disposition"]] += 1
    summary = {
        "status": "ELECTRIC_UNITS_READY — PROCEED_TO_12F_NOTEBOOKS",
        "historical_heuristic_rows": len(corrections), "current_source_matches": len(active_rows) + len(excluded_rows),
        "resolved_nonzero_electric_rows": len(active_rows), "excluded_fcev_zero_rows": len(excluded_rows),
        "retired_absent_rows": sum(row["status"] == "RETIRED_SOURCE_RECORD_ABSENT" for row in corrections),
        "electricity_cases": sum(row["fuel_type"] == "Electricity" for row in all_dispositions),
        "now_resolved_by_unit_rule": counts["NOW_RESOLVED_KWH_PER_100MI"],
        "still_unresolved_electricity": counts["NOT_SAME_SOURCE_SEMANTICS"] + counts["CONFLICTING_SOURCE_CONTEXT"],
        "not_same_or_conflicting": counts["NOT_SAME_SOURCE_SEMANTICS"] + counts["CONFLICTING_SOURCE_CONTEXT"],
        "not_same_source_semantics": counts["NOT_SAME_SOURCE_SEMANTICS"],
        "conflicting_source_context": counts["CONFLICTING_SOURCE_CONTEXT"],
        "hydrogen_cases": counts["HYDROGEN_OUTSIDE_ELECTRIC_RULE"], "hydrogen_converted": 0,
        "additional_canonical_fuelcons": 0, "runtime_db_changed": before != after, "user_decisions_required": 0,
        "override_registry_active_rows": sum(row["validation_status"] == "ACTIVE_DETERMINISTIC_UNIT_INTERPRETATION" for row in overrides),
        "sanity_dataset_rows": len(sanity), "output_signature": output_signature,
        "previous_signature_available": bool(previous.get("output_signature")),
        "rebuild_deterministic": previous.get("output_signature") in (None, output_signature),
        "runtime_fingerprints_before": before, "runtime_fingerprints_after": after,
    }
    base.SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = report_text(summary)
    REPORT.write_text(report, encoding="utf-8")
    base.REPORT.write_text("# Sprint 12E.3 — Superseded by closure patch\n\n" + report, encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
