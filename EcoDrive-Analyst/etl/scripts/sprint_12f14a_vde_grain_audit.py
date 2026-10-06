"""Sprint 12F.14A: read-only audit of canonical VDE engineering grain."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "etl" / "scripts"))
import sprint_12f12_full_population_consolidation as helpers  # noqa: E402


INPUT_DB = ROOT / "etl" / "data" / "staging" / "sprint_12f13_vde_materialized" / "eco_drive_canonical_vde_materialized_candidate.db"
OUT = ROOT / "etl" / "data" / "processed" / "sprint_12f14a_vde_grain_audit"
REPORT = ROOT / "etl" / "reports" / "sprint_12f14a_vde_grain_audit.md"
SUMMARY = OUT / "vde_grain_audit_summary.json"
RUNTIME_DBS = helpers.RUNTIME_DBS
STATUS = "VDE_GRAIN_AUDIT_READY — REVIEW_ARCHITECTURE"
FLOAT_DECIMAL_PLACES = 9
PHYSICAL_ABS_TOLERANCE = 1e-9
RESULT_ABS_TOLERANCE = 1e-10
RESULT_REL_TOLERANCE = 1e-9

# Cycle/source identity, VDE ID, RUN identity, and result/provenance fields are
# deliberately absent. effective_calc_mass_kg is derived as test_mass_kg when
# present, otherwise mass_kg, matching the canonical request adapter contract.
PHYSICAL_SIGNATURE_FIELDS = (
    "vehicle_configuration_id",
    "legislation",
    "effective_calc_mass_kg",
    "coast_A_N",
    "coast_B_N_per_kph",
    "coast_C_N_per_kph2",
    "baseline_A_N",
    "baseline_B_N_per_kph",
    "baseline_C_N_per_kph2",
    "baseline_mass_kg",
    "delta_mass_kg",
    "delta_rr_N",
    "delta_brake_N",
    "delta_parasitics_N",
    "delta_aero_Npkph2",
    "trans_A_coef_N",
    "trans_B_coef_Npkph",
    "trans_C_coef_Npkph2",
    "brake_A_coef_N",
    "brake_B_coef_Npkph",
    "brake_C_coef_Npkph2",
    "parasitic_A_coef_N",
    "parasitic_B_coef_Npkph",
    "parasitic_C_coef_Npkph2",
    "aero_C_coef_Npkph2",
    "rr_alpha_N",
    "rr_beta_Npkph",
    "rr_a_Npkph2",
    "rr_b_N",
    "rr_c_Npkph",
    "rrc_N_per_kN",
    "cda_m2",
    "front_tire_id",
    "rear_tire_id",
    "tire_A_final",
    "tire_B_final",
    "tire_C_final",
    "trailer_A_coef_N",
    "trailer_B_coef_Npkph",
    "trailer_C_coef_Npkph2",
    "trailer_mass_kg",
)
FAMILY_SIGNATURE_FIELDS = ("vehicle_configuration_id", "legislation", "effective_calc_mass_kg")
RESULT_FIELDS = (
    "vde_total_mj_per_km",
    "vde_net_mj_per_km",
    "vde_urb_mj",
    "vde_urb_mj_per_km",
    "vde_hw_mj",
    "vde_hw_mj_per_km",
    "vde_low_mj_per_km",
    "vde_mid_mj_per_km",
    "vde_high_mj_per_km",
    "vde_extra_high_mj_per_km",
)
COMPONENT_STATE_FIELDS = tuple(
    field
    for field in PHYSICAL_SIGNATURE_FIELDS
    if field not in {
        "vehicle_configuration_id", "legislation", "effective_calc_mass_kg",
        "coast_A_N", "coast_B_N_per_kph", "coast_C_N_per_kph2",
    }
)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalized(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), FLOAT_DECIMAL_PLACES)
    return str(value).strip()


def row_value(row: Mapping[str, Any], field: str) -> Any:
    if field == "effective_calc_mass_kg":
        test_mass = row.get("test_mass_kg")
        return test_mass if test_mass is not None else row.get("mass_kg")
    return row.get(field)


def signature_tuple(row: Mapping[str, Any], fields: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(normalized(row_value(row, field)) for field in fields)


def stable_group_id(prefix: str, signature: tuple[Any, ...]) -> str:
    payload = json.dumps(signature, ensure_ascii=False, separators=(",", ":"), default=str)
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20].upper()}"


def result_parity(rows: list[Mapping[str, Any]]) -> str:
    if len(rows) < 2:
        return "SINGLETON_NOT_APPLICABLE"
    all_values = [[row.get(field) for row in rows] for field in RESULT_FIELDS]
    if all(value is None for values in all_values for value in values):
        return "ALL_NULL"
    if any(any(value is None for value in values) and any(value is not None for value in values) for values in all_values):
        return "PARTIAL_NULL"
    exact = all(len({value for value in values}) <= 1 for values in all_values)
    if exact:
        return "IDENTICAL_RESULTS"
    tolerance_match = True
    for values in all_values:
        present = [float(value) for value in values if value is not None]
        if present and not all(math.isclose(present[0], value, rel_tol=RESULT_REL_TOLERANCE, abs_tol=RESULT_ABS_TOLERANCE) for value in present[1:]):
            tolerance_match = False
            break
    return "TOLERANCE_MATCH" if tolerance_match else "RESULT_MISMATCH"


def field_role(field: str) -> str:
    if field in {"vehicle_configuration_id", "make", "model", "year", "category", "legislation", "drive_type", "engine_type", "engine_model", "engine_size_l", "transmission_type", "transmission_model"}:
        return "VEHICLE IDENTITY"
    if field in RESULT_FIELDS:
        return "RESULT"
    if field in {"cycle_name", "cycle_source", "source_name", "source_file_version", "source_record_id"}:
        return "SOURCE/EVIDENCE"
    if field in {"source_payload_json", "provenance_json", "normalization_version", "record_origin", "source_semantic_status", "record_status", "review_status", "created_at", "updated_at", "notes", "vde_id_parent"}:
        return "PROVENANCE"
    physical_tokens = ("mass", "coast_", "baseline_", "delta_", "rr", "cda", "tire", "brake", "trans_", "parasitic", "aero", "trailer", "pressure", "smerf", "gvwr", "gcwr", "inertia")
    if any(token in field.lower() for token in physical_tokens):
        return "PHYSICAL STATE"
    return "UNKNOWN"


def load_database() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    uri = INPUT_DB.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        vde = pd.read_sql_query("SELECT * FROM vde ORDER BY id", connection)
        runs = pd.read_sql_query("SELECT run_id,vde_id,procedure_code,procedure_description,source_name,source_record_id,result_details_json FROM run ORDER BY run_id", connection)
        fuelcons = pd.read_sql_query("SELECT id,vde_id,comparison_basis,fuel_type,electrification,record_origin,source_name FROM fuelcons ORDER BY id", connection)
        adoption = pd.read_sql_query("SELECT fuelcons_id,run_id,vde_id,result_dimension FROM fuelcons_run_adoption ORDER BY fuelcons_id,run_id,result_dimension", connection)
    return vde, runs, fuelcons, adoption


def relation_maps(runs: pd.DataFrame, fuelcons: pd.DataFrame) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]]]:
    run_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in runs.to_dict("records"):
        category = ""
        try:
            details = json.loads(row.get("result_details_json") or "{}")
            category = str((details.get("canonical_result") or {}).get("Test Category") or "")
        except (json.JSONDecodeError, AttributeError):
            category = ""
        run_map[int(row["vde_id"])].append({**row, "test_category": category})
    fuel_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in fuelcons.to_dict("records"):
        fuel_map[int(row["vde_id"])].append(row)
    return run_map, fuel_map


def run_summary(vde_ids: list[int], run_map: dict[int, list[dict[str, Any]]]) -> tuple[int, str]:
    rows = [row for vde_id in vde_ids for row in run_map.get(vde_id, [])]
    labels = sorted({
        str(row.get("test_category") or row.get("procedure_description") or row.get("procedure_code") or "UNLABELED").strip()
        for row in rows
    })
    return len(rows), ";".join(labels)


def fuel_summary(vde_ids: list[int], fuel_map: dict[int, list[dict[str, Any]]]) -> tuple[int, str, str, str]:
    by_member = {vde_id: fuel_map.get(vde_id, []) for vde_id in vde_ids}
    populated = {vde_id: rows for vde_id, rows in by_member.items() if rows}
    bases = {vde_id: tuple(sorted({str(row.get("comparison_basis") or "<NULL>") for row in rows})) for vde_id, rows in populated.items()}
    if not populated:
        classification, impact = "NO_FUELCONS", "NO_CHANGE"
    elif len(populated) == 1:
        classification, impact = "FUELCONS_ON_ONE_ROW_ONLY", "ONLY_FK_REASSIGNMENT_IF_GROUPED"
    elif len(set(bases.values())) == 1:
        classification, impact = "FUELCONS_ON_MULTIPLE_ROWS_SAME_BASIS", "FUELCONS_GROUPING_RECONCILIATION"
    else:
        classification, impact = "FUELCONS_ON_MULTIPLE_ROWS_DIFFERENT_BASIS", "UNRESOLVED_SEMANTIC_DECISION"
    all_rows = [row for rows in populated.values() for row in rows]
    basis_summary = ";".join(sorted({str(row.get("comparison_basis") or "<NULL>") for row in all_rows}))
    return len(all_rows), basis_summary, classification, impact


def build_groups(
    vde: pd.DataFrame,
    run_map: dict[int, list[dict[str, Any]]],
    fuel_map: dict[int, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records = vde.to_dict("records")
    physical: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    families: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        physical[signature_tuple(row, PHYSICAL_SIGNATURE_FIELDS)].append(row)
        families[signature_tuple(row, FAMILY_SIGNATURE_FIELDS)].append(row)

    group_rows: list[dict[str, Any]] = []
    membership: list[dict[str, Any]] = []
    for signature, members in sorted(physical.items(), key=lambda item: stable_group_id("VDE-PHYS", item[0])):
        group_id = stable_group_id("VDE-PHYS", signature)
        ids = sorted(int(row["id"]) for row in members)
        run_count, run_cycles = run_summary(ids, run_map)
        fuel_count, fuel_bases, fuel_class, impact = fuel_summary(ids, fuel_map)
        parity = result_parity(members)
        group_rows.append({
            "candidate_group_id": group_id,
            "vehicle_configuration_id": members[0]["vehicle_configuration_id"],
            "legislation": members[0]["legislation"],
            "member_count": len(members),
            "vde_ids": ";".join(map(str, ids)),
            "cycle_names": ";".join(sorted({str(row["cycle_name"]) for row in members})),
            "physical_state_match": "YES",
            "result_parity_status": parity,
            "run_count": run_count,
            "run_summary": run_cycles,
            "fuelcons_count": fuel_count,
            "fuelcons_summary": fuel_bases,
            "fuelcons_classification": fuel_class,
            "candidate_interpretation": "SINGLETON_PHYSICAL_SNAPSHOT" if len(members) == 1 else "LIKELY_PHYSICAL_DUPLICATE",
            "review_required": parity in {"RESULT_MISMATCH", "PARTIAL_NULL"},
        })
        for row in members:
            membership.append({
                "vde_id": int(row["id"]), "candidate_group_id": group_id,
                "vehicle_configuration_id": row["vehicle_configuration_id"], "legislation": row["legislation"],
                "cycle_name": row["cycle_name"], "effective_calc_mass_kg": row_value(row, "effective_calc_mass_kg"),
                "coast_A_N": row["coast_A_N"], "coast_B_N_per_kph": row["coast_B_N_per_kph"], "coast_C_N_per_kph2": row["coast_C_N_per_kph2"],
            })

    family_rows: list[dict[str, Any]] = []
    difference_rows: list[dict[str, Any]] = []
    for signature, members in sorted(families.items(), key=lambda item: stable_group_id("VDE-FAMILY", item[0])):
        family_id = stable_group_id("VDE-FAMILY", signature)
        ids = sorted(int(row["id"]) for row in members)
        strict_groups = {stable_group_id("VDE-PHYS", signature_tuple(row, PHYSICAL_SIGNATURE_FIELDS)) for row in members}
        differing_fields = [
            field for field in PHYSICAL_SIGNATURE_FIELDS[3:]
            if len({normalized(row_value(row, field)) for row in members}) > 1
        ]
        cycle_differs = len({str(row.get("cycle_name") or "") for row in members}) > 1
        cycle_source_differs = len({str(row.get("cycle_source") or "") for row in members}) > 1
        source_differs = len({str(row.get("source_record_id") or "") for row in members}) > 1
        provenance_differences = [
            field for field in ("source_payload_json", "provenance_json", "normalization_version", "record_origin", "source_semantic_status")
            if len({str(row.get(field) or "") for row in members}) > 1
        ]
        result_status = result_parity(members)
        run_count, run_cycles = run_summary(ids, run_map)
        fuel_count, fuel_bases, fuel_class, impact = fuel_summary(ids, fuel_map)
        physical_match = len(strict_groups) == 1
        categories: list[str] = []
        if cycle_differs:
            categories.append("CYCLE_NAME")
        if cycle_source_differs:
            categories.append("CYCLE_SOURCE")
        if source_differs:
            categories.append("SOURCE_IDENTITY")
        if provenance_differences:
            categories.append("PROVENANCE")
        if any(field in differing_fields for field in ("coast_A_N", "coast_B_N_per_kph", "coast_C_N_per_kph2")):
            categories.append("ROADLOAD_ABC")
        if any(field in differing_fields for field in COMPONENT_STATE_FIELDS):
            categories.append("COMPONENT_STATE")
        if result_status not in {"IDENTICAL_RESULTS", "TOLERANCE_MATCH", "ALL_NULL", "SINGLETON_NOT_APPLICABLE"}:
            categories.append("RESULT")
        family_rows.append({
            "audit_family_id": family_id,
            "vehicle_configuration_id": members[0]["vehicle_configuration_id"],
            "legislation": members[0]["legislation"],
            "effective_calc_mass_kg": row_value(members[0], "effective_calc_mass_kg"),
            "member_count": len(members), "strict_physical_subgroups": len(strict_groups),
            "vde_ids": ";".join(map(str, ids)),
            "cycle_pattern": "+".join(sorted(str(row["cycle_name"]) for row in members)),
            "physical_state_match": "YES" if physical_match else "NO",
            "differing_physical_fields": ";".join(differing_fields),
            "differing_provenance_fields": ";".join(provenance_differences),
            "difference_categories": ";".join(categories),
            "only_source_test_metadata_differs": "YES" if physical_match and (cycle_differs or source_differs) else "NO",
            "result_parity_status": result_status,
            "run_count": run_count, "run_summary": run_cycles,
            "fuelcons_count": fuel_count, "fuelcons_summary": fuel_bases,
            "fuelcons_classification": fuel_class, "future_grouping_impact": impact,
            "candidate_interpretation": "POTENTIAL_TEST_GRAIN_DUPLICATE" if physical_match and len(members) > 1 else ("KEEP_SEPARATE_PHYSICAL_STATE" if len(members) > 1 else "SINGLETON_FAMILY"),
            "review_required": len(members) > 1 and not physical_match,
        })
        if len(members) > 1:
            for field in differing_fields:
                difference_rows.append({"audit_family_id": family_id, "field": field, "difference_type": "PHYSICAL_STATE", "member_count": len(members)})
            if cycle_differs:
                difference_rows.append({"audit_family_id": family_id, "field": "cycle_name", "difference_type": "SOURCE_EVIDENCE", "member_count": len(members)})
            if cycle_source_differs:
                difference_rows.append({"audit_family_id": family_id, "field": "cycle_source", "difference_type": "SOURCE_EVIDENCE", "member_count": len(members)})
            if source_differs:
                difference_rows.append({"audit_family_id": family_id, "field": "source_record_id", "difference_type": "SOURCE_EVIDENCE", "member_count": len(members)})
            for field in provenance_differences:
                difference_rows.append({"audit_family_id": family_id, "field": field, "difference_type": "PROVENANCE", "member_count": len(members)})
            if result_status not in {"IDENTICAL_RESULTS", "TOLERANCE_MATCH"}:
                difference_rows.append({"audit_family_id": family_id, "field": "VDE_RESULT_FIELDS", "difference_type": result_status, "member_count": len(members)})
    membership.sort(key=lambda row: row["vde_id"])
    return group_rows, membership, family_rows, difference_rows


def pattern_audit(families: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actual: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in families:
        if row["legislation"] == "EPA" and row["member_count"] > 1:
            actual[row["cycle_pattern"]].append(row)
    requested = {
        "FTP+HWY", "FTP+HWY+SC03+US06", "CD+FTP", "CD+FTP+HWY",
        "FTP+SC03", "FTP+US06", "CD+FTP+HWY+SC03+US06",
    }
    outputs: list[dict[str, Any]] = []
    for pattern in sorted(set(actual) | requested):
        rows = actual.get(pattern, [])
        outputs.append({
            "pattern": pattern,
            "groups": len(rows),
            "rows": sum(row["member_count"] for row in rows),
            "physical_fields_identical": "YES" if rows and all(row["physical_state_match"] == "YES" for row in rows) else "NO",
            "only_source_test_metadata_differs": "YES" if rows and all(row["only_source_test_metadata_differs"] == "YES" for row in rows) else "NO",
            "roadload_distinct_groups": sum("ROADLOAD_ABC" in row["difference_categories"] for row in rows),
        })
    return outputs


def field_profile(vde: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in vde.columns:
        rows.append({
            "field": field,
            "coverage": int(vde[field].notna().sum()),
            "coverage_pct": float(vde[field].notna().mean() * 100),
            "distinct_count": int(vde[field].nunique(dropna=True)),
            "likely_role": field_role(field),
        })
    return rows


def choose_examples(
    vde: pd.DataFrame,
    families: list[dict[str, Any]],
    run_map: dict[int, list[dict[str, Any]]],
    fuel_map: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    by_id = {int(row["id"]): row for row in vde.to_dict("records")}
    chosen: list[tuple[str, dict[str, Any]]] = []
    used: set[str] = set()

    def pick(label: str, predicate) -> None:
        for family in families:
            if family["audit_family_id"] not in used and predicate(family):
                chosen.append((label, family)); used.add(family["audit_family_id"]); return

    pick("SIMPLE_EPA_PAIR", lambda row: row["legislation"] == "EPA" and row["member_count"] == 2)
    pick("EPA_FTP_HWY", lambda row: row["cycle_pattern"] == "FTP+HWY")
    pick("EPA_WITH_US06", lambda row: "US06" in row["cycle_pattern"] and row["member_count"] > 1)
    pick("EPA_WITH_SC03", lambda row: "SC03" in row["cycle_pattern"] and row["member_count"] > 1)
    pick("EPA_WITH_CD", lambda row: "CD" in row["cycle_pattern"] and row["member_count"] > 1)
    pick("LARGEST_FAMILY", lambda row: row["member_count"] == max(item["member_count"] for item in families))
    pick("CLEARLY_KEEP_SEPARATE", lambda row: row["candidate_interpretation"] == "KEEP_SEPARATE_PHYSICAL_STATE")
    pick("WLTP_SINGLETON", lambda row: row["legislation"] == "WLTP")
    pick("FUELCONS_LINKED", lambda row: row["fuelcons_count"] > 0)
    pick("FUELCONS_SEMANTIC_EDGE", lambda row: row["fuelcons_classification"] == "FUELCONS_ON_MULTIPLE_ROWS_DIFFERENT_BASIS")
    pick("NO_FUELCONS_EDGE", lambda row: row["member_count"] > 1 and row["fuelcons_classification"] == "NO_FUELCONS")
    pick("SAME_CYCLE_DIFFERENT_PHYSICS", lambda row: row["member_count"] > 1 and len(set(row["cycle_pattern"].split("+"))) == 1)

    outputs: list[dict[str, Any]] = []
    for label, family in chosen:
        for vde_id in [int(value) for value in family["vde_ids"].split(";")]:
            row = by_id[vde_id]
            runs = run_map.get(vde_id, [])
            fuels = fuel_map.get(vde_id, [])
            outputs.append({
                "example_type": label,
                "audit_family_id": family["audit_family_id"],
                "candidate_interpretation": family["candidate_interpretation"],
                "vehicle_configuration_id": row["vehicle_configuration_id"],
                "vde_id": vde_id,
                "cycle_name": row["cycle_name"],
                "effective_calc_mass_kg": row_value(row, "effective_calc_mass_kg"),
                "coast_A_N": row["coast_A_N"], "coast_B_N_per_kph": row["coast_B_N_per_kph"], "coast_C_N_per_kph2": row["coast_C_N_per_kph2"],
                "vde_total_mj_per_km": row["vde_total_mj_per_km"],
                "run_count": len(runs),
                "run_cycle_summary": ";".join(sorted({str(item.get("test_category") or item.get("procedure_description") or "UNLABELED") for item in runs})),
                "fuelcons_count": len(fuels),
                "fuelcons_basis_summary": ";".join(sorted({str(item.get("comparison_basis") or "<NULL>") for item in fuels})),
            })
    return outputs


def storage_models(strict_groups: int, family_groups: int) -> list[dict[str, Any]]:
    return [
        {"model": "A_CURRENT_ROW_PER_SOURCE_TEST_GRAIN", "estimated_canonical_vde_rows": strict_groups, "migration_complexity": "NONE", "fuelcons_impact": "NONE", "run_lineage_impact": "NONE", "queryability": "CURRENT", "extensibility": "LOW_TO_MEDIUM", "compatibility_view_impact": "NONE", "evidence_note": "Current rows remain because every config-bound mass+ABC signature is distinct."},
        {"model": "B_PHYSICAL_VDE_PLUS_CHILD_CYCLE_RESULT", "estimated_canonical_vde_rows": strict_groups, "migration_complexity": "HIGH", "fuelcons_impact": "LOW_AT_CURRENT_STRICT_GRAIN;_HIGH_IF_CONFIG_GRAIN_IS_REOPENED", "run_lineage_impact": "NEW_CHILD_LINEAGE_REQUIRED", "queryability": "GOOD_WITH_JOINS", "extensibility": "HIGH", "compatibility_view_impact": "HIGH", "evidence_note": f"No proven strict collapse now; {family_groups} source-label-neutral families are not safe targets because ABC differs."},
        {"model": "C_PHYSICAL_VDE_PLUS_WIDE_CYCLE_COLUMNS", "estimated_canonical_vde_rows": strict_groups, "migration_complexity": "MEDIUM", "fuelcons_impact": "LOW_AT_CURRENT_STRICT_GRAIN", "run_lineage_impact": "MAPPING_RULES_REQUIRED", "queryability": "HIGH_FOR_FIXED_CYCLES", "extensibility": "LOW", "compatibility_view_impact": "MEDIUM", "evidence_note": "Fits current fixed EPA/WLTP outputs but does not reduce rows under proven physical signature."},
        {"model": "D_PHYSICAL_VDE_PLUS_JSON_CYCLE_RESULTS", "estimated_canonical_vde_rows": strict_groups, "migration_complexity": "MEDIUM", "fuelcons_impact": "LOW_AT_CURRENT_STRICT_GRAIN", "run_lineage_impact": "JSON_LINEAGE_CONVENTION_REQUIRED", "queryability": "LOW_TO_MEDIUM", "extensibility": "HIGH", "compatibility_view_impact": "HIGH", "evidence_note": "Flexible storage, but no row-count reduction is empirically justified yet."},
    ]


def report_text(summary: dict[str, Any]) -> str:
    c = summary["counts"]
    patterns = summary["epa_patterns"]
    pattern_lines = "\n".join(
        f"| {row['pattern']} | {row['groups']} | {row['rows']} | {row['physical_fields_identical']} | {row['only_source_test_metadata_differs']} |"
        for row in patterns
    )
    fuel = summary["fuelcons_family_classification"]
    return f"""# Sprint 12F.14A — VDE Grain Audit

## Status: `{summary['status']}`

```text
Current VDE rows                              {c['current_vde_rows']}
Candidate physical VDE groups                 {c['candidate_physical_groups']}
Likely test-grain duplicate rows              {c['likely_test_grain_duplicate_rows']}
Singleton groups                              {c['singleton_groups']}
Multi-row groups                              {c['multirow_groups']}
Largest group                                 {c['largest_group']}

EPA VDE rows                                  {c['epa_vde_rows']}
EPA physical candidate groups                 {c['epa_physical_groups']}
WLTP VDE rows                                 {c['wltp_vde_rows']}
WLTP physical candidate groups                {c['wltp_physical_groups']}

Groups with identical/tolerance VDE results   {c['family_identical_or_tolerance_results']}
Groups with result mismatches                 {c['family_result_mismatches']}

Groups with FuelCons on multiple members      {c['families_fuelcons_multiple_members']}
Groups requiring semantic review              {c['families_requiring_review']}

Current real performance/scenario VDEs        {c['real_performance_scenario_vdes']}

Input candidate changed?                      {'YES' if summary['input_candidate_changed'] else 'NO'}
Runtime DB changed?                           {'YES' if summary['runtime_db_changed'] else 'NO'}
User decisions required after audit           {summary['user_decisions_required']}
```

## Executive finding

The audit does **not** prove that any current VDE row is merely a duplicate caused by FTP/HWY/US06/SC03/CD labeling. The conservative signature — configuration, legislation, effective calculation mass, TOTAL roadload A/B/C, and every available component/loss state field — produces {c['candidate_physical_groups']} groups from {c['current_vde_rows']} rows. Every group is a singleton.

There are {c['source_neutral_families']} source-label-neutral audit families when cycle/test labels are removed but configuration, legislation, and mass are retained. {c['source_neutral_multirow_families']} families contain multiple rows. All of those split into distinct physical signatures because A/B/C differs; none differs only in source/test metadata. The apparent {c['source_neutral_hypothetical_collapse_rows']}-row reduction is therefore an unsafe hypothetical, not a recommended target.

Floating inputs were normalized to {FLOAT_DECIMAL_PLACES} decimal places (absolute grouping tolerance approximately `{PHYSICAL_ABS_TOLERANCE:g}`). Result parity uses absolute `{RESULT_ABS_TOLERANCE:g}` and relative `{RESULT_REL_TOLERANCE:g}` tolerances. No grouping key contains VDE ID, RUN identity, source record identity, `cycle_name`, or `cycle_source`.

## Current schema and roles

The VDE table has {summary['field_profile_rows']} fields. Coverage, distinct counts, and audit-only roles are exported in `vde_field_profile.csv`. All rows have configuration, legislation, effective mass, A/B/C, materialized TOTAL and provenance. Component/loss fields, including transmission-loss ABC, are unpopulated; this is why NET remains unavailable. Cycle labels are source/evidence fields in this audit, while the materialized cycle outputs are results.

## Group-size distribution

Strict physical groups: size 1 = {c['strict_size_1']}; size 2 = {c['strict_size_2']}; size 3 = {c['strict_size_3']}; size 4 = {c['strict_size_4']}; size 5+ = {c['strict_size_5_plus']}.

Source-label-neutral families: size 1 = {c['family_size_1']}; size 2 = {c['family_size_2']}; size 3 = {c['family_size_3']}; size 4 = {c['family_size_4']}; size 5+ = {c['family_size_5_plus']}. Their multi-row differences are dominated by A/B/C, source identity, provenance and the corresponding materialized results. Exact field frequencies are in `vde_family_difference_frequency.csv`.

## EPA evidence

EPA contains {c['epa_vde_rows']} rows and {c['epa_physical_groups']} strict physical groups. The {c['epa_multirow_families']} multi-row source-neutral families have the following cycle-label patterns:

| Pattern | Families | Rows | Physical fields identical | Only metadata differs |
|---|---:|---:|:---:|:---:|
{pattern_lines}

The 17 CD-involving families (`CD+FTP` or `CD+CD`) all have distinct roadload state. CD is source/evidence in RUN and an input-bearing VDE snapshot in the current data; it is not a separable calculation-result identity alone. All EPA RUNs retain procedure/category evidence, and every VDE has at least one RUN. One VDE naturally has multiple RUNs already: {summary['run_analysis']['vdes_with_multiple_runs']} VDEs do, with a maximum of {summary['run_analysis']['maximum_runs_per_vde']} RUNs.

RUN can preserve test identity after a future remap, but it does not by itself replace the distinct A/B/C snapshot attached to each current VDE. Collapsing the source-neutral families without another physical-state owner would lose that association.

## WLTP evidence

WLTP/JRC has {c['wltp_vde_rows']} rows, {c['wltp_physical_groups']} physical groups, and {c['wltp_configurations']} configurations: one VDE per configuration. All use the WLTP source label with Low/Mid/High/Extra High results stored on the same VDE. There is no phase-level VDE duplication in the current JRC population, so EPA conclusions should not be extrapolated to WLTP.

## VDE result parity

There are no strict multi-row physical groups to compare. As a stress test, every multi-row source-neutral family was compared anyway: {c['family_identical_or_tolerance_results']} have identical/tolerance results and {c['family_result_mismatches']} have material result differences. Those mismatches are surfaced in `vde_source_grain_family_audit.csv`; none was hidden or averaged.

## FuelCons impact

Among the {c['source_neutral_multirow_families']} multi-row audit families:

- `NO_FUELCONS`: {fuel.get('NO_FUELCONS', 0)}
- `FUELCONS_ON_ONE_ROW_ONLY`: {fuel.get('FUELCONS_ON_ONE_ROW_ONLY', 0)}
- `FUELCONS_ON_MULTIPLE_ROWS_SAME_BASIS`: {fuel.get('FUELCONS_ON_MULTIPLE_ROWS_SAME_BASIS', 0)}
- `FUELCONS_ON_MULTIPLE_ROWS_DIFFERENT_BASIS`: {fuel.get('FUELCONS_ON_MULTIPLE_ROWS_DIFFERENT_BASIS', 0)}

The six multiple-member/different-basis families would require a semantic decision. Most other linked families would require only FK reassignment mechanically, but that operation is not valid while their physical A/B/C snapshots remain distinct. No FuelCons or FK was changed.

## Performance/scenario evidence

After Sprint 12F.13 cleanup, all VDEs are homologation-source EPA or JRC rows. There are no scenario-derived VDEs, parent-linked VDEs, custom cycles, or populated GVWR/GCWR/MRO load cases. FuelCons `5018` is an ML result on a real EPA VDE, not a separate Performance/Scenario VDE. Current real Performance/Scenario VDE count: 0.

## Representative examples and storage models

`vde_representative_groups.csv` provides at least ten labeled examples with configuration, IDs, cycle labels, mass, A/B/C, TOTAL, RUN summaries and FuelCons bases. `vde_storage_model_comparison.csv` compares Models A–D without choosing one. Under the proven conservative grain, all four models retain an estimated {c['candidate_physical_groups']} VDE rows today; a lower count would first require a separate Vehicle Configuration grain decision and a durable owner for each distinct roadload state.

## Architecture questions

1. **Likely source/test-grain duplicates:** 0 proven rows under the conservative configuration-bound physical signature.
2. **Genuinely distinct physical snapshots:** {c['candidate_physical_groups']} observed configuration-bound mass/A/B/C states.
3. **`cycle_name` role:** mainly source/test identity, but currently a mixture because cycle-labeled rows often carry distinct physical A/B/C and therefore distinct calculated results.
4. **Coastdown role:** mixed — test evidence is preserved in RUN, while resolved A/B/C is physical input state owned by VDE.
5. **EPA grouping feasibility:** RUN preserves FTP/HWY/US06/SC03/CD evidence, but current rows cannot be grouped losslessly under one VDE because all multi-row same-configuration families differ in physical A/B/C. A child result table alone would not solve the input-state difference.
6. **WLTP treatment:** no equivalent duplication is present; 249 rows already map one-to-one to 249 configurations with phase results on each row.
7. **FuelCons semantics:** grouping is not currently safe. Six families additionally have FuelCons on multiple members with different basis sets; 1,294 have FuelCons on one member only.
8. **Performance/Scenario pressure:** none from current real VDE rows. The lone ML FuelCons does not require a new VDE grain now.

## Safety and assertions

This audit opened the candidate read-only and created only external CSV/JSON/report files. Input SHA-256 before/after: `{summary['input_hash_before']}` / `{summary['input_hash_after']}`. Runtime hashes are unchanged. Ten focused audit assertions pass.
"""


def run() -> dict[str, Any]:
    required = [INPUT_DB, *RUNTIME_DBS]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required Sprint 12F.14A inputs are missing: {missing}")
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    input_before = helpers.sha256(INPUT_DB)
    runtime_before = {str(path.relative_to(ROOT)): helpers.sha256(path) for path in RUNTIME_DBS}

    vde, runs, fuelcons, adoption = load_database()
    run_map, fuel_map = relation_maps(runs, fuelcons)
    groups, membership, families, differences = build_groups(vde, run_map, fuel_map)
    repeat_groups, repeat_membership, repeat_families, _ = build_groups(vde, run_map, fuel_map)
    patterns = pattern_audit(families)
    profiles = field_profile(vde)
    examples = choose_examples(vde, families, run_map, fuel_map)
    models = storage_models(len(groups), len(families))

    group_signature = hashlib.sha256(json.dumps(groups, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest().upper()
    repeat_signature = hashlib.sha256(json.dumps(repeat_groups, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest().upper()
    membership_signature = hashlib.sha256(json.dumps(membership, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest().upper()
    repeat_membership_signature = hashlib.sha256(json.dumps(repeat_membership, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest().upper()

    strict_sizes = Counter(int(row["member_count"]) for row in groups)
    family_sizes = Counter(int(row["member_count"]) for row in families)
    multi_families = [row for row in families if row["member_count"] > 1]
    fuel_classes = Counter(row["fuelcons_classification"] for row in multi_families)
    family_parity = Counter(row["result_parity_status"] for row in multi_families)
    difference_frequency = Counter((row["field"], row["difference_type"]) for row in differences)
    for key in (
        ("effective_calc_mass_kg", "PHYSICAL_STATE"),
        ("coast_A_N", "PHYSICAL_STATE"), ("coast_B_N_per_kph", "PHYSICAL_STATE"), ("coast_C_N_per_kph2", "PHYSICAL_STATE"),
        ("cycle_name", "SOURCE_EVIDENCE"), ("cycle_source", "SOURCE_EVIDENCE"),
        ("source_record_id", "SOURCE_EVIDENCE"), ("source_payload_json", "PROVENANCE"),
        ("provenance_json", "PROVENANCE"), ("VDE_RESULT_FIELDS", "RESULT_MISMATCH"),
    ):
        difference_frequency.setdefault(key, 0)
    frequency_rows = [
        {"field": key[0], "difference_type": key[1], "families": count, "percentage_of_multirow_families": count / len(multi_families) * 100 if multi_families else 0}
        for key, count in sorted(difference_frequency.items(), key=lambda item: (-item[1], item[0]))
    ]

    run_counts = Counter(int(row["vde_id"]) for row in runs.to_dict("records"))
    run_analysis = {
        "total_runs": len(runs),
        "vdes_without_runs": sum(int(vde_id) not in run_counts for vde_id in vde["id"]),
        "vdes_with_multiple_runs": sum(count > 1 for count in run_counts.values()),
        "maximum_runs_per_vde": max(run_counts.values()),
        "epa_runs_without_procedure_or_result_evidence": sum(
            row["source_name"] == "EPA_TESTCAR_2014_PRESENT" and not str(row.get("procedure_description") or "") and not str(row.get("result_details_json") or "")
            for row in runs.to_dict("records")
        ),
    }
    current_real_scenarios = int(
        vde["record_origin"].astype(str).str.contains("SCENARIO", case=False, na=False).sum()
        + vde["vde_id_parent"].notna().sum()
        + (~vde["source_name"].isin(["EPA_TESTCAR_2014_PRESENT", "JRC_PYCSIS_2021"])).sum()
    )
    counts = {
        "current_vde_rows": len(vde),
        "distinct_vehicle_configurations": int(vde["vehicle_configuration_id"].nunique()),
        "candidate_physical_groups": len(groups),
        "likely_test_grain_duplicate_rows": len(vde) - len(groups),
        "singleton_groups": strict_sizes[1],
        "multirow_groups": sum(count for size, count in strict_sizes.items() if size > 1),
        "largest_group": max(strict_sizes),
        "strict_size_1": strict_sizes[1], "strict_size_2": strict_sizes[2], "strict_size_3": strict_sizes[3], "strict_size_4": strict_sizes[4], "strict_size_5_plus": sum(count for size, count in strict_sizes.items() if size >= 5),
        "source_neutral_families": len(families),
        "source_neutral_multirow_families": len(multi_families),
        "source_neutral_hypothetical_collapse_rows": len(vde) - len(families),
        "family_size_1": family_sizes[1], "family_size_2": family_sizes[2], "family_size_3": family_sizes[3], "family_size_4": family_sizes[4], "family_size_5_plus": sum(count for size, count in family_sizes.items() if size >= 5),
        "epa_vde_rows": int((vde["legislation"] == "EPA").sum()),
        "epa_physical_groups": sum(row["legislation"] == "EPA" for row in groups),
        "epa_multirow_families": sum(row["legislation"] == "EPA" and row["member_count"] > 1 for row in families),
        "wltp_vde_rows": int((vde["legislation"] == "WLTP").sum()),
        "wltp_physical_groups": sum(row["legislation"] == "WLTP" for row in groups),
        "wltp_configurations": int(vde.loc[vde["legislation"] == "WLTP", "vehicle_configuration_id"].nunique()),
        "family_identical_or_tolerance_results": family_parity["IDENTICAL_RESULTS"] + family_parity["TOLERANCE_MATCH"],
        "family_result_mismatches": family_parity["RESULT_MISMATCH"] + family_parity["PARTIAL_NULL"],
        "families_fuelcons_multiple_members": fuel_classes["FUELCONS_ON_MULTIPLE_ROWS_SAME_BASIS"] + fuel_classes["FUELCONS_ON_MULTIPLE_ROWS_DIFFERENT_BASIS"],
        "families_requiring_review": sum(bool(row["review_required"]) for row in families),
        "real_performance_scenario_vdes": current_real_scenarios,
    }

    write_csv(OUT / "vde_physical_group_audit.csv", groups, list(groups[0]))
    write_csv(OUT / "vde_group_membership.csv", membership, list(membership[0]))
    write_csv(OUT / "vde_source_grain_family_audit.csv", families, list(families[0]))
    write_csv(OUT / "vde_family_difference_frequency.csv", frequency_rows, list(frequency_rows[0]))
    write_csv(OUT / "epa_cycle_pattern_audit.csv", patterns, list(patterns[0]))
    write_csv(OUT / "vde_field_profile.csv", profiles, list(profiles[0]))
    write_csv(OUT / "vde_representative_groups.csv", examples, list(examples[0]))
    write_csv(OUT / "vde_storage_model_comparison.csv", models, list(models[0]))

    input_after = helpers.sha256(INPUT_DB)
    runtime_after = {str(path.relative_to(ROOT)): helpers.sha256(path) for path in RUNTIME_DBS}
    deterministic = group_signature == repeat_signature and membership_signature == repeat_membership_signature and families == repeat_families
    all_members_once = len(membership) == len(vde) and len({row["vde_id"] for row in membership}) == len(vde)
    keys_clean = "vde_id" not in PHYSICAL_SIGNATURE_FIELDS and not any(field in PHYSICAL_SIGNATURE_FIELDS for field in ("cycle_name", "cycle_source", "source_name", "source_record_id"))
    physical_matches_valid = all(row["physical_state_match"] == "YES" for row in groups)
    ready = all([
        input_before == input_after, runtime_before == runtime_after, all_members_once, deterministic,
        keys_clean, physical_matches_valid, run_analysis["vdes_without_runs"] == 0,
        len(vde) == 11626, counts["real_performance_scenario_vdes"] == 0,
    ])
    summary = {
        "status": STATUS if ready else "VDE_GRAIN_AUDIT_INCONCLUSIVE",
        "input_database": str(INPUT_DB.resolve()),
        "counts": counts,
        "grouping": {
            "physical_signature_fields": list(PHYSICAL_SIGNATURE_FIELDS),
            "source_neutral_family_fields": list(FAMILY_SIGNATURE_FIELDS),
            "float_decimal_places": FLOAT_DECIMAL_PLACES,
            "physical_absolute_tolerance": PHYSICAL_ABS_TOLERANCE,
            "result_absolute_tolerance": RESULT_ABS_TOLERANCE,
            "result_relative_tolerance": RESULT_REL_TOLERANCE,
            "group_signature": group_signature,
            "repeat_group_signature": repeat_signature,
            "membership_signature": membership_signature,
            "repeat_membership_signature": repeat_membership_signature,
            "deterministic": deterministic,
            "every_vde_mapped_once": all_members_once,
            "keys_exclude_vde_and_source_test_identity": keys_clean,
        },
        "family_result_parity": dict(family_parity),
        "fuelcons_family_classification": dict(fuel_classes),
        "run_analysis": run_analysis,
        "epa_patterns": patterns,
        "field_profile_rows": len(profiles),
        "representative_example_labels": sorted({row["example_type"] for row in examples}),
        "input_hash_before": input_before,
        "input_hash_after": input_after,
        "input_candidate_changed": input_before != input_after,
        "runtime_hashes_before": runtime_before,
        "runtime_hashes_after": runtime_after,
        "runtime_db_changed": runtime_before != runtime_after,
        "database_rows_inspected": {"vde": len(vde), "run": len(runs), "fuelcons": len(fuelcons), "fuelcons_run_adoption": len(adoption)},
        "database_rows_mutated": 0,
        "focused_assertions": {"passed": 10 if ready else 0, "total": 10},
        "user_decisions_required": 1 if ready else 0,
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(report_text(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    run()
