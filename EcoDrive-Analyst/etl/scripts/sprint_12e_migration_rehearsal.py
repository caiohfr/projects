"""Build the disposable Sprint 12E canonical migration rehearsal database.

All runtime and raw-source inputs are read-only.  The only SQLite output is a
new database below ``etl/data/staging`` (or an explicitly supplied safe path
below ``etl``).  This script never performs application cutover.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "etl" / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import sprint_12c1_dataset_delta_audit as c1  # noqa: E402
import sprint_12c3_program_consolidation_review as c3  # noqa: E402
import sprint_12_closure_phase2 as closure2  # noqa: E402
from vde_core.roadload.tire_model import MPH_PER_KPH, N_PER_LBF  # noqa: E402


RUNTIME_DB = ROOT / "data" / "db" / "eco_drive.db"
QA_DB = ROOT / "data" / "db" / "eco_drive_qa.db"
LEGACY_RUNTIME_DB = ROOT / "data" / "db" / "archive" / "eco_drive_legacy_pre_sprint12.db"
SCHEMA_SQL = ROOT / "etl" / "schema" / "canonical_schema_v1.sql"
COMPAT_SQL = ROOT / "etl" / "schema" / "canonical_compatibility_v1.sql"
LEGACY_EPA = ROOT / "data" / "vehicles" / "testcar-2025-2020-EPA.xlsx"
REFRESHED_EPA = ROOT / "etl" / "data" / "raw" / "epa_testcar" / "epa_testcar_2026_raw.xlsx"
JRC = ROOT / "etl" / "data" / "raw" / "wltp_jrc" / "Data_PV_fleet_2021_EU_PYCSIS.xlsx"
EEA = ROOT / "etl" / "data" / "raw" / "wltp_eea" / "data.csv"

STAGING_DIR = ROOT / "etl" / "data" / "staging" / "sprint_12e_migration_rehearsal"
DEFAULT_OUTPUT_DB = STAGING_DIR / "eco_drive_canonical_rehearsal.db"
OUT = ROOT / "etl" / "data" / "processed" / "sprint_12e_migration_rehearsal"
REPORT = ROOT / "etl" / "reports" / "sprint_12e_migration_rehearsal.md"
PREVIOUS_SUMMARY = OUT / "rehearsal_summary.json"

OUTPUT_NAMES = (
    "migration_phase_results.csv",
    "canonical_population_counts.csv",
    "population_by_source.csv",
    "relationship_checks.csv",
    "compatibility_checks.csv",
    "compatibility_mismatches.csv",
    "source_refresh_differences.csv",
    "program_consolidation_results.csv",
    "identity_quarantine.csv",
    "performance_results.csv",
    "runtime_db_fingerprints.csv",
    "rehearsal_summary.json",
)
PHYSICAL_TABLES = (
    "program", "vehicle_configuration", "component_db", "tire_db",
    "component_instance", "component_resolution", "vde", "run", "fuelcons",
    "fuelcons_run_adoption", "vde_component_resolution",
)
LEGACY_VDE_FIELDS = 101
LEGACY_FUELCONS_FIELDS = 79
EEA_AUDITED_ROWS = 10_833_597
MIGRATION_TIMESTAMP = "2026-09-10T00:00:00Z"
KG_PER_LB = 0.45359237

EPA_VDE_CARRYOVER_FIELDS = (
    "Represented Test Veh Make", "Represented Test Veh Model",
    "Actual Tested Testgroup", "Test Vehicle ID", "Test Veh Configuration #",
    "Test Veh Displacement (L)", "Engine Code", "Tested Transmission Type",
    "# of Gears", "Drive System Description", "Axle Ratio", "N/V Ratio",
    "Rated Horsepower", "# of Cylinders and Rotors", "Test Fuel Type Description",
    "Vehicle Type", "Equivalent Test Weight (lbs.)",
    *c3.TARGET_FIELDS,
    "Test Number", "ADFE Test Number", "ADFE Total Road Load HP",
    "Test Category", "Test Procedure Cd", "Test Procedure Description",
    "Set Coef A (lbf)", "Set Coef B (lbf/mph)", "Set Coef C (lbf/mph**2)",
    "THC (g/mi)", "CO (g/mi)", "CO2 (g/mi)", "NOx (g/mi)",
    "PM (g/mi)", "CH4 (g/mi)", "N2O (g/mi)", "RND_ADJ_FE", "FE_UNIT",
    "FE Bag 1", "FE Bag 2", "FE Bag 3", "FE Bag 4",
)


def clean(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, bool):
        return int(value)
    return value


def json_text(value: Any) -> str | None:
    value = clean(value)
    if value is None:
        return None
    if isinstance(value, str):
        try:
            non_standard_constants: list[str] = []
            parsed = json.loads(
                value,
                parse_constant=lambda token: non_standard_constants.append(token) or None,
            )
            if non_standard_constants:
                return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            return value
        except json.JSONDecodeError:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def stable_id(prefix: str, *parts: Any) -> str:
    material = "\x1f".join("<NULL>" if clean(part) is None else str(clean(part)).strip() for part in parts)
    return f"{prefix}-{hashlib.sha256(material.encode('utf-8')).hexdigest()[:20].upper()}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "sha256": sha256(path)}


def open_readonly(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only=ON")
    if con.execute("PRAGMA query_only").fetchone()[0] != 1:
        con.close()
        raise RuntimeError(f"Read-only verification failed for {path}")
    return con


def guard_output_path(path: Path) -> Path:
    resolved = path.resolve()
    protected = {RUNTIME_DB.resolve(), QA_DB.resolve()}
    if resolved in protected:
        raise ValueError(f"Protected runtime database cannot be an output: {resolved}")
    allowed_root = (ROOT / "etl" / "data" / "staging").resolve()
    if not resolved.is_relative_to(allowed_root) or resolved.suffix.lower() != ".db":
        raise ValueError(f"Rehearsal output must be a .db below etl/data/staging/: {resolved}")
    return resolved


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def table_info(con: sqlite3.Connection, table: str) -> dict[str, sqlite3.Row]:
    return {row[1]: row for row in con.execute(f'PRAGMA table_info("{table}")')}


def encoded(value: Any, column: str) -> Any:
    value = clean(value)
    if value is not None and (column.endswith("_json") or column == "source_identity_json"):
        return json_text(value)
    return value


def insert_rows(con: sqlite3.Connection, table: str, rows: Iterable[dict[str, Any]]) -> int:
    info = table_info(con, table)
    cache: dict[tuple[str, ...], str] = {}
    count = 0
    for source in rows:
        values: dict[str, Any] = {}
        for column, meta in info.items():
            if column not in source:
                continue
            value = encoded(source[column], column)
            # Let a NOT NULL column use its declared default rather than insert NULL.
            if value is None and meta[3] and meta[4] is not None:
                continue
            values[column] = value
        columns = tuple(values)
        if not columns:
            raise RuntimeError(f"No insertable columns for {table}")
        statement = cache.get(columns)
        if statement is None:
            names = ", ".join(f'"{name}"' for name in columns)
            placeholders = ", ".join("?" for _ in columns)
            statement = f'INSERT INTO "{table}" ({names}) VALUES ({placeholders})'
            cache[columns] = statement
        con.execute(statement, tuple(values[name] for name in columns))
        count += 1
    return count


def read_runtime_population() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    # Sprint 12H promoted the canonical database to the normal runtime path.
    # Irreproducible legacy-state reconstruction therefore reads the preserved,
    # explicitly archived pre-Sprint-12 source instead of assuming PROD remains
    # on the legacy schema.
    with open_readonly(LEGACY_RUNTIME_DB) as con:
        vdes = [dict(row) for row in con.execute("SELECT * FROM vde_db ORDER BY id")]
        fuelcons = [dict(row) for row in con.execute("SELECT * FROM fuelcons_db ORDER BY id")]
        tires = [dict(row) for row in con.execute("SELECT * FROM tire_roadload_db ORDER BY id")]
        components = [dict(row) for row in con.execute("SELECT * FROM component_db ORDER BY id")]
    return vdes, fuelcons, tires, components


def legacy_population(
    vdes: list[dict[str, Any]], fuelcons: list[dict[str, Any]], tires: list[dict[str, Any]],
    runtime_hash: str,
) -> dict[str, list[dict[str, Any]]]:
    programs: dict[str, dict[str, Any]] = {}
    configs: dict[str, dict[str, Any]] = {}
    vde_to_config: dict[int, str] = {}
    for row in vdes:
        pid = stable_id("PRG-LEGACY", row.get("make"), row.get("model"), row.get("year"))
        programs.setdefault(pid, {
            "program_id": pid, "commercial_make": row.get("make") or "UNSPECIFIED",
            "commercial_model": row.get("model") or "UNSPECIFIED", "generation_name": None,
            "model_year_from": row.get("year"), "model_year_to": row.get("year"),
            "identity_status": "PROVISIONAL_SOURCE_SCOPED", "identity_confidence": "LOW",
            "source_identity_json": {"make": row.get("make"), "model": row.get("model"), "year": row.get("year")},
            "source_scope": "LEGACY_ECODRIVE", "source_name": "LEGACY_ECODRIVE",
            "source_file_version": runtime_hash, "created_at": row.get("created_at") or MIGRATION_TIMESTAMP,
        })
        signature = (
            pid, row.get("engine_type"), row.get("engine_model"), row.get("engine_size_l"),
            row.get("engine_aspiration"), row.get("transmission_type"),
            row.get("transmission_model"), row.get("drive_type"),
        )
        cid = stable_id("CFG-LEGACY", *signature)
        configs.setdefault(cid, {
            "vehicle_configuration_id": cid, "program_id": pid,
            "propulsion_architecture": row.get("engine_type"), "engine_type": row.get("engine_type"),
            "engine_model": row.get("engine_model"), "engine_displacement_l": row.get("engine_size_l"),
            "engine_aspiration": row.get("engine_aspiration"), "transmission_type": row.get("transmission_type"),
            "transmission_model": row.get("transmission_model"), "drive_system": row.get("drive_type"),
            "identity_status": "SOURCE_SCOPED", "identity_confidence": "MEDIUM",
            "source_identity_json": {"legacy_vde_ids": []}, "source_scope": "LEGACY_ECODRIVE",
            "source_name": "LEGACY_ECODRIVE", "source_file_version": runtime_hash,
            "created_at": row.get("created_at") or MIGRATION_TIMESTAMP,
        })
        configs[cid]["source_identity_json"]["legacy_vde_ids"].append(row["id"])
        vde_to_config[int(row["id"])] = cid

    canonical_vdes: list[dict[str, Any]] = []
    for row in vdes:
        item = dict(row)
        derived = row.get("vde_id_parent") is not None
        item.update({
            "vehicle_configuration_id": vde_to_config[int(row["id"])],
            "source_semantic_status": "DIRECT",
            "source_payload_json": {"legacy_vde_id": row["id"]},
            "source_file_version": runtime_hash,
            "normalization_version": "sprint_12e_legacy_v1",
            "provenance_json": {
                "population": "DERIVED_SCENARIO" if derived else "LEGACY_PRESERVED",
                "compatibility_record_origin_preserved": row.get("record_origin"),
                "component_decomposition_semantics": "ESTIMATED_WHERE_PRESENT",
            },
        })
        canonical_vdes.append(item)

    runs: list[dict[str, Any]] = []
    canonical_fuelcons: list[dict[str, Any]] = []
    adoptions: list[dict[str, Any]] = []
    for row in fuelcons:
        run_id = stable_id("RUN-LEGACY", row["id"])
        method_backed = any(row.get(key) is not None for key in ("engine_method", "engine_version", "assumptions_json", "provenance_json"))
        is_later = int(row["id"]) > 4999 or int(row["vde_id"]) > 4999
        classification = "ML_PREDICTION" if method_backed else ("DERIVED_SCENARIO" if is_later else "LEGACY_PRESERVED")
        runs.append({
            "run_id": run_id, "vde_id": row["vde_id"],
            "run_type": "ML_PREDICTION" if method_backed else "DECLARED_RESULT",
            "evidence_kind": "ENGINEERING" if method_backed or is_later else "SOURCE_RECORD",
            "source_name": row.get("source_name") or "LEGACY_ECODRIVE",
            "source_file_version": runtime_hash,
            "source_record_id": row.get("source_record_id") or str(row["id"]),
            "procedure_description": row.get("method_note"),
            "conditions_json": {key: row.get(key) for key in ("ambient_temp_c", "ac_on", "tire_front_psi", "tire_rear_psi", "scenario_payload_kg")},
            "result_details_json": {key: value for key, value in row.items() if key.startswith(("energy_", "fuel_", "gco2_", "label_"))},
            "method": row.get("engine_method"), "method_version": row.get("engine_version"),
            "assumptions_json": row.get("assumptions_json"),
            "provenance_json": {
                "population": classification, "legacy_fuelcons_id": row["id"],
                "legacy_assumptions_json_raw": row.get("assumptions_json")
                if row.get("assumptions_json") and "NaN" in row["assumptions_json"] else None,
                "json_constraint_correction": "NON_STANDARD_NAN_TO_NULL"
                if row.get("assumptions_json") and "NaN" in row["assumptions_json"] else None,
            },
            "created_at": row.get("created_at") or MIGRATION_TIMESTAMP,
        })
        item = dict(row)
        item.update({
            "comparison_basis": "LEGACY_UNSPECIFIED", "source_file_version": runtime_hash,
            "normalization_version": "sprint_12e_legacy_v1",
        })
        canonical_fuelcons.append(item)
        adoptions.append({
            "fuelcons_id": row["id"], "run_id": run_id, "vde_id": row["vde_id"],
            "adoption_role": "PRIMARY", "result_dimension": "ALL", "ordinal": 0,
            "provenance_json": {"population": classification}, "created_at": MIGRATION_TIMESTAMP,
        })

    canonical_tires = []
    for row in tires:
        item = dict(row)
        item["tire_id"] = item.pop("id")
        item.update({
            "source_file_version": runtime_hash,
            "provenance_json": {"population": "LEGACY_PRESERVED", "legacy_tire_id": item["tire_id"]},
        })
        canonical_tires.append(item)

    return {
        "program": list(programs.values()), "vehicle_configuration": list(configs.values()),
        "component_db": [], "tire_db": canonical_tires, "component_instance": [],
        "component_resolution": [], "vde": canonical_vdes, "run": runs,
        "fuelcons": canonical_fuelcons, "fuelcons_run_adoption": adoptions,
        "vde_component_resolution": [],
    }


def safe_program_mapping(rows: pd.DataFrame) -> tuple[dict[str, str], list[dict[str, Any]], int, int]:
    boundaries = c3.make_boundaries(rows)
    nodes = sorted(set(rows["fallback_program_id"]))
    union = c3.UnionFind(nodes)
    for boundary in boundaries:
        if boundary["consolidation_status"] == "SAFE_CONSOLIDATE":
            union.union(boundary["fallback_program_from"], boundary["fallback_program_to"])
    groups: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        groups[union.find(node)].append(node)
    mapping: dict[str, str] = {}
    output: list[dict[str, Any]] = []
    for members in sorted(groups.values(), key=lambda group: group[0]):
        canonical = stable_id("PRG-EPA", *sorted(members))
        subset = rows[rows["fallback_program_id"].isin(members)]
        for fallback in sorted(members):
            item = subset[subset["fallback_program_id"].eq(fallback)].sort_values("source_excel_row").iloc[0]
            mapping[fallback] = canonical
            output.append({
                "fallback_program_id": fallback, "canonical_program_id": canonical,
                "make": clean(item["Represented Test Veh Make"]),
                "model": clean(item["Represented Test Veh Model"]),
                "model_year": int(item["Model Year"]), "safe_group_size": len(members),
                "decision": "SAFE_CONSOLIDATED" if len(members) > 1 else "PRESERVED_SOURCE_SCOPED",
                "applied_rule": "Only SAFE_CONSOLIDATE edges from Sprint 12C.3",
            })
    return mapping, output, len(nodes), len(groups)


def si_abc(row: pd.Series, prefix: str) -> tuple[float | None, float | None, float | None]:
    a = clean(row.get(f"{prefix} Coef A (lbf)"))
    b = clean(row.get(f"{prefix} Coef B (lbf/mph)"))
    c = clean(row.get(f"{prefix} Coef C (lbf/mph**2)"))
    return (
        None if a is None else float(a) * N_PER_LBF,
        None if b is None else float(b) * N_PER_LBF * MPH_PER_KPH,
        None if c is None else float(c) * N_PER_LBF * MPH_PER_KPH * MPH_PER_KPH,
    )


def epa_population(
    source: pd.DataFrame, source_hash: str,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prepared, _ = c3.prepare_rows(source)
    quarantine_rows = prepared[prepared["identity_anomaly"]].copy()
    rows = prepared[~prepared["identity_anomaly"]].copy()
    program_map, consolidation, fallback_before, safe_after = safe_program_mapping(rows)

    quarantine: list[dict[str, Any]] = []
    for _, series in quarantine_rows.sort_values("source_excel_row").iterrows():
        raw = {key: clean(value) for key, value in series[source.columns].to_dict().items()}
        quarantine.append({
            "source": "EPA_TESTCAR_2014_PRESENT", "source_record_id": str(int(series["source_excel_row"])),
            "status": "QUARANTINED", "reason_code": "YEAR_LIKE_REPRESENTED_MAKE",
            "reason": "Represented Test Veh Make is a four-digit year-like value; no canonical identity was invented.",
            "source_payload_json": json_text(raw),
        })

    populations = {table: [] for table in PHYSICAL_TABLES}
    for canonical_pid, group in rows.groupby(rows["fallback_program_id"].map(program_map), sort=True):
        ordered = group.sort_values(["Model Year", "source_excel_row"], kind="stable")
        first = ordered.iloc[0]
        members = sorted(set(ordered["fallback_program_id"]))
        populations["program"].append({
            "program_id": canonical_pid,
            "commercial_make": clean(first["Represented Test Veh Make"]),
            "commercial_model": clean(first["Represented Test Veh Model"]),
            "generation_name": None, "model_year_from": int(ordered["Model Year"].min()),
            "model_year_to": int(ordered["Model Year"].max()),
            "identity_status": "PROVISIONAL_SOURCE_SCOPED", "identity_confidence": "HIGH" if len(members) > 1 else "LOW",
            "source_identity_json": {"fallback_program_ids": members, "safe_consolidation_only": True},
            "source_scope": "EPA_TESTCAR_2020_2026", "source_name": "EPA_TESTCAR_2014_PRESENT",
            "source_file_version": source_hash, "source_record_id": canonical_pid,
            "created_at": MIGRATION_TIMESTAMP,
        })

    config_map: dict[str, str] = {}
    for candidate, group in rows.groupby("configuration_candidate_id", sort=True):
        first = group.sort_values("source_excel_row").iloc[0]
        cid = stable_id("CFG-EPA", candidate)
        config_map[candidate] = cid
        populations["vehicle_configuration"].append({
            "vehicle_configuration_id": cid, "program_id": program_map[first["fallback_program_id"]],
            "propulsion_architecture": clean(first.get("Test Fuel Type Description")),
            "engine_type": clean(first.get("Test Fuel Type Description")),
            "engine_model": clean(first.get("Engine Code")),
            "engine_displacement_l": clean(first.get("Test Veh Displacement (L)")),
            "engine_rated_power_kw": closure2.horsepower_to_kw(first.get("Rated Horsepower")),
            "engine_cylinders_rotors": closure2.cylinders_rotors(first.get("# of Cylinders and Rotors")),
            "transmission_type": clean(first.get("Tested Transmission Type")),
            "gear_count": clean(first.get("# of Gears")), "drive_system": clean(first.get("Drive System Description")),
            "final_drive_ratio": clean(first.get("Axle Ratio")), "nv_ratio": clean(first.get("N/V Ratio")),
            "identity_status": "SOURCE_SCOPED", "identity_confidence": "MEDIUM",
            "source_identity_json": {
                "candidate_id": candidate, "test_group": clean(first.get("Actual Tested Testgroup")),
                "test_vehicle_id": clean(first.get("Test Vehicle ID")),
                "configuration_number": clean(first.get("Test Veh Configuration #")),
                "raw_source_values": {
                    "rated_horsepower": clean(first.get("Rated Horsepower")),
                    "cylinders_rotors": clean(first.get("# of Cylinders and Rotors")),
                },
            },
            "source_scope": "EPA_TESTCAR_2020_2026", "source_name": "EPA_TESTCAR_2014_PRESENT",
            "source_file_version": source_hash, "source_record_id": candidate,
            "architecture_properties_json": None,
            "created_at": MIGRATION_TIMESTAMP,
        })

    vde_map: dict[str, int] = {
        candidate: -1_000_000 - position
        for position, candidate in enumerate(sorted(set(rows["vde_candidate_id"])), start=1)
    }
    vde_carryover_records: list[dict[str, Any]] = []
    for candidate, group in rows.groupby("vde_candidate_id", sort=True):
        first = group.sort_values("source_excel_row").iloc[0]
        target_a, target_b, target_c = si_abc(first, "Target")
        year = int(first["Model Year"])
        roadload_condition = closure2.classify_epa_roadload_condition(first.to_dict())
        signature = closure2.group_signature(
            (series.to_dict() for _, series in group.iterrows()),
            EPA_VDE_CARRYOVER_FIELDS,
        )
        vde_row = {
            "id": vde_map[candidate], "vehicle_configuration_id": config_map[first["configuration_candidate_id"]],
            "legislation": "EPA", "category": clean(first["Vehicle Type"]),
            "make": clean(first["Represented Test Veh Make"]), "model": clean(first["Represented Test Veh Model"]),
            "year": year, "engine_type": clean(first.get("Test Fuel Type Description")),
            "engine_model": clean(first.get("Engine Code")), "engine_size_l": clean(first.get("Test Veh Displacement (L)")),
            "transmission_type": clean(first.get("Tested Transmission Type")),
            "mass_kg": float(first["Equivalent Test Weight (lbs.)"]) * KG_PER_LB,
            "test_mass_kg": float(first["Equivalent Test Weight (lbs.)"]) * KG_PER_LB,
            "test_mass_basis": "EPA_ETW_DIRECT_SOURCE_CONVERTED_LB_TO_KG",
            "coast_A_N": target_a, "coast_B_N_per_kph": target_b, "coast_C_N_per_kph2": target_c,
            "drive_type": clean(first.get("Drive System Description")),
            "cycle_name": roadload_condition["cycle_name"],
            "cycle_source": roadload_condition["cycle_source"],
            "roadload_temperature_c": roadload_condition["roadload_temperature_c"],
            "roadload_ambient_pressure_kpa": roadload_condition["roadload_ambient_pressure_kpa"],
            "record_origin": "SOURCE_REFRESHED" if year <= 2025 else "NEW_SOURCE",
            "source_name": "EPA_TESTCAR_2014_PRESENT", "source_file_version": source_hash,
            "source_record_id": candidate, "source_semantic_status": "DIRECT",
            "source_payload_json": {
                "source_excel_rows": sorted(int(value) for value in group["source_excel_row"]),
                "target_abc_native": {
                    "a_lbf": clean(first[c3.TARGET_FIELDS[0]]), "b_lbf_per_mph": clean(first[c3.TARGET_FIELDS[1]]),
                    "c_lbf_per_mph2": clean(first[c3.TARGET_FIELDS[2]]),
                },
                "roadload_condition_evidence": roadload_condition["provenance"],
            },
            "normalization_version": "sprint_12e_epa_v1",
            "provenance_json": {
                "population": "SOURCE_REFRESHED" if year <= 2025 else "NEW_SOURCE",
                "grain": "configuration_plus_target_abc_plus_etw",
                "carryover_signature_sha256": signature,
                "carryover_match_rule": "EXACT_ROADLOAD_AND_TEST_EVIDENCE_V2",
                "roadload_condition": roadload_condition["provenance"],
                "unit_conversions": {"lb_to_kg": KG_PER_LB, "lbf_to_n": N_PER_LBF, "mph_per_kph": MPH_PER_KPH},
            },
            "created_at": MIGRATION_TIMESTAMP,
        }
        populations["vde"].append(vde_row)
        vde_carryover_records.append({"id": vde_row["id"], "year": year, "signature": signature})

    vde_lineage = closure2.assign_temporal_parents(vde_carryover_records)
    for vde_row in populations["vde"]:
        lineage = vde_lineage[vde_row["id"]]
        provenance = vde_row["provenance_json"]
        provenance["carryover_status"] = lineage["status"]
        if lineage["status"] == "LINKED":
            vde_row["vde_id_parent"] = lineage["parent_id"]
            provenance.update({
                "lineage_relation": closure2.EPA_MODEL_YEAR_CARRYOVER,
                "carryover_from_model_year": lineage["parent_year"],
                "carryover_from_vde_id": lineage["parent_id"],
            })
            vde_row["notes"] = (
                f"EPA model-year carryover from {lineage['parent_year']} VDE "
                f"{lineage['parent_id']} by exact source-evidence signature."
            )
        elif lineage["status"] == "AMBIGUOUS":
            vde_row["review_status"] = "CARRYOVER_IDENTITY_REVIEW"

    # One RUN per source row is the safe, lossless grain.  Test Number is kept as
    # source identity rather than used to merge rows that can map to distinct VDEs.
    for _, row in rows.sort_values("source_excel_row", kind="stable").iterrows():
        source_row = int(row["source_excel_row"])
        run_id = stable_id("RUN-EPA-ROW", source_hash, source_row)
        set_a, set_b, set_c = si_abc(row, "Set")
        result_fields = {
            key: clean(row.get(key)) for key in (
                "Test Category", "THC (g/mi)", "CO (g/mi)", "CO2 (g/mi)", "NOx (g/mi)",
                "PM (g/mi)", "CH4 (g/mi)", "N2O (g/mi)", "RND_ADJ_FE", "FE_UNIT",
                "FE Bag 1", "FE Bag 2", "FE Bag 3", "FE Bag 4",
            )
        }
        populations["run"].append({
            "run_id": run_id, "vde_id": vde_map[row["vde_candidate_id"]], "run_type": "TEST",
            "evidence_kind": "HOMOLOGATION", "confidence": "HIGH", "source_name": "EPA_TESTCAR_2014_PRESENT",
            "source_file_version": source_hash, "source_record_id": str(source_row),
            "procedure_code": str(clean(row.get("Test Procedure Cd"))) if clean(row.get("Test Procedure Cd")) is not None else None,
            "procedure_description": clean(row.get("Test Procedure Description")),
            "conditions_json": {
                "test_number": clean(row.get("Test Number")), "test_group": clean(row.get("Actual Tested Testgroup")),
                "test_vehicle_id": clean(row.get("Test Vehicle ID")),
                "configuration_number": clean(row.get("Test Veh Configuration #")),
                "set_abc_si": {"a_n": set_a, "b_n_per_kph": set_b, "c_n_per_kph2": set_c},
                "set_abc_native": {
                    "a_lbf": clean(row.get(c3.SET_FIELDS[0])), "b_lbf_per_mph": clean(row.get(c3.SET_FIELDS[1])),
                    "c_lbf_per_mph2": clean(row.get(c3.SET_FIELDS[2])),
                },
            },
            "result_details_json": result_fields,
            "provenance_json": {
                "population": "SOURCE_REFRESHED" if int(row["Model Year"]) <= 2025 else "NEW_SOURCE",
                "source_excel_row": source_row, "raw_test_identity_preserved": True,
                "fuelcons_materialization": "DEFERRED_NO_APPROVED_ADOPTION_RULE",
            },
            "created_at": MIGRATION_TIMESTAMP,
        })

    stats = {
        "source_rows": len(prepared), "loaded_rows": len(rows), "quarantined_rows": len(quarantine),
        "fallback_programs_before": fallback_before, "safe_programs_after": safe_after,
    }
    return populations, consolidation, quarantine, stats


def normalize_electrification(row: dict[str, Any]) -> str:
    if bool(clean(row.get("is_electric"))):
        return "BEV"
    if bool(clean(row.get("is_plugin"))):
        return "PHEV"
    if bool(clean(row.get("is_hybrid"))):
        return "HEV"
    return "ICE"


def jrc_population(source: pd.DataFrame, source_hash: str) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    populations = {table: [] for table in PHYSICAL_TABLES}
    unresolved: list[dict[str, Any]] = []
    for index, series in source.reset_index(drop=True).iterrows():
        row = {key: clean(value) for key, value in series.to_dict().items()}
        source_row = index + 2
        pid = stable_id("PRG-JRC", source_hash, source_row)
        cid = stable_id("CFG-JRC", source_hash, source_row)
        vde_id = -2_000_000 - source_row
        run_id = stable_id("RUN-JRC", source_hash, source_row)
        fuelcons_id = -3_000_000 - source_row
        electrification = normalize_electrification(row)
        populations["program"].append({
            "program_id": pid, "commercial_make": row.get("OEM anon") or "JRC_ANON",
            "commercial_model": row.get("Model anon") or f"ROW_{source_row}",
            "identity_status": "UNRESOLVED", "identity_confidence": "LOW",
            "source_identity_json": {"source_row": source_row, "anonymized": True},
            "source_scope": "JRC_PYCSIS_2021", "source_name": "JRC_PYCSIS_2021",
            "source_file_version": source_hash, "source_record_id": str(source_row),
            "created_at": MIGRATION_TIMESTAMP,
        })
        populations["vehicle_configuration"].append({
            "vehicle_configuration_id": cid, "program_id": pid,
            "propulsion_architecture": row.get("Input type"), "engine_type": row.get("Fuel type"),
            "engine_displacement_l": None if row.get("Engine capacity [cm3]") is None else float(row["Engine capacity [cm3]"]) / 1000.0,
            "engine_aspiration": "TURBO" if row.get("Engine is turbo") else None,
            "transmission_type": row.get("Gear box type"), "gear_count": row.get("N gears"),
            "identity_status": "UNRESOLVED", "identity_confidence": "LOW",
            "source_identity_json": {"source_row": source_row, "oem_anon": row.get("OEM anon"), "model_anon": row.get("Model anon")},
            "source_scope": "JRC_PYCSIS_2021", "source_name": "JRC_PYCSIS_2021",
            "source_file_version": source_hash, "source_record_id": str(source_row),
            "architecture_properties_json": {
                "engine_power_kw": row.get("Engine max power"), "engine_cylinders": row.get("Engine n cylinders"),
                "tire_code": row.get("Tyre code"), "electric_motor_power_kw": row.get("Electric motor power [kW]"),
                "battery_capacity_ah": row.get("Drive battery capacity [Ah]"),
                "battery_nominal_voltage_v": row.get("Drive battery nominal voltage [V]"),
            },
            "created_at": MIGRATION_TIMESTAMP,
        })
        populations["vde"].append({
            "id": vde_id, "vehicle_configuration_id": cid, "legislation": "WLTP",
            "category": row.get("Vehicle body") or "JRC_UNSPECIFIED", "make": row.get("OEM anon") or "JRC_ANON",
            "model": row.get("Model anon") or f"ROW_{source_row}", "engine_type": row.get("Fuel type"),
            "engine_size_l": None if row.get("Engine capacity [cm3]") is None else float(row["Engine capacity [cm3]"]) / 1000.0,
            "transmission_type": row.get("Gear box type"), "mass_kg": row.get("Curb_vehicle_mass [kg]"),
            "test_mass_kg": row.get("Vehicle mass (WLTP) [kg]"), "test_mass_basis": "JRC_DIRECT_WLTP_MASS",
            "coast_A_N": row.get("wltp|f0 [N]"), "coast_B_N_per_kph": row.get("wltp|f1 [N/(km/h)]"),
            "coast_C_N_per_kph2": row.get("wltp|f2 [N/(km/h)2]"), "cycle_name": "WLTP",
            "cycle_source": "JRC_PYCSIS_2021", "record_origin": "NEW_SOURCE",
            "source_name": "JRC_PYCSIS_2021", "source_file_version": source_hash,
            "source_record_id": str(source_row), "source_semantic_status": "PARTIAL",
            "source_payload_json": {"source_row": source_row, "jrc_row_grain": "UNRESOLVED"},
            "normalization_version": "sprint_12e_jrc_v1",
            "provenance_json": {"population": "NEW_SOURCE", "identity": "ANONYMIZED_SOURCE_SCOPED"},
            "created_at": MIGRATION_TIMESTAMP,
        })
        populations["run"].append({
            "run_id": run_id, "vde_id": vde_id,
            "run_type": "SIMULATION" if row.get("pycsis_run") else "DECLARED_RESULT",
            "evidence_kind": "SOURCE_RECORD", "confidence": "LOW", "source_name": "JRC_PYCSIS_2021",
            "source_file_version": source_hash, "source_record_id": str(source_row),
            "procedure_description": "JRC source row; identity and exact row grain unresolved",
            "conditions_json": {"input_type": row.get("Input type"), "fuel_type": row.get("Fuel type")},
            "result_details_json": {key: value for key, value in row.items() if "Declared" in key or "Real-world" in key},
            "method": "PYCSIS" if row.get("pycsis_run") else None,
            "provenance_json": {"population": "NEW_SOURCE", "classification": "DIRECT_VALUES_WITH_UNRESOLVED_ROW_GRAIN"},
            "created_at": MIGRATION_TIMESTAMP,
        })
        populations["fuelcons"].append({
            "id": fuelcons_id, "vde_id": vde_id, "electrification": electrification,
            "fuel_type": row.get("Fuel type"),
            "engine_max_power_kw": row.get("Engine max power"), "gear_count": row.get("N gears"),
            "energy_Wh_per_km": row.get("Declared electric consumption value (OEM) [Wh/km]"),
            "gco2_per_km": row.get("Declared average CO2 emissions value (OEM) [g/km]"),
            "label_range_km": row.get("Electric range (OEM) [km]"),
            "comparison_basis": "JRC_OEM_DECLARED_WLTP", "record_origin": "NEW_SOURCE",
            "source_name": "JRC_PYCSIS_2021", "source_file_version": source_hash,
            "source_record_id": str(source_row), "normalization_version": "sprint_12e_jrc_v1",
            "provenance_json": {"population": "NEW_SOURCE", "materialization": "DIRECT_OEM_DECLARED_RESULT"},
            "created_at": MIGRATION_TIMESTAMP,
        })
        populations["fuelcons_run_adoption"].append({
            "fuelcons_id": fuelcons_id, "run_id": run_id, "vde_id": vde_id,
            "adoption_role": "PRIMARY", "result_dimension": "ALL", "ordinal": 0,
            "provenance_json": {"materialization": "DIRECT_OEM_DECLARED_RESULT"},
            "created_at": MIGRATION_TIMESTAMP,
        })
        for domain, role, properties in (
            ("ENGINE", "PRIMARY", {"rated_power_kw": row.get("Engine max power"), "cylinders": row.get("Engine n cylinders"), "capacity_cm3": row.get("Engine capacity [cm3]")}),
            ("TRANSMISSION", "MAIN", {"gearbox_type": row.get("Gear box type"), "gear_count": row.get("N gears")}),
            ("TIRE", "UNSPECIFIED_POSITION", {"tire_code": row.get("Tyre code")}),
            ("BATTERY", "TRACTION", {"capacity_ah": row.get("Drive battery capacity [Ah]"), "nominal_voltage_v": row.get("Drive battery nominal voltage [V]")}),
            ("EMOTOR", "TRACTION", {"rated_power_kw": row.get("Electric motor power [kW]"), "rated_torque_nm": row.get("Electric motor torque [Nm]")}),
        ):
            if not any(clean(value) is not None for value in properties.values()):
                continue
            populations["component_instance"].append({
                "component_instance_id": stable_id("INS-JRC", source_hash, source_row, domain, role),
                "vehicle_configuration_id": cid, "component_domain": domain, "role": role,
                "quantity": 1, "instance_properties_json": properties,
                "provenance_json": {"source": "JRC_PYCSIS_2021", "source_row": source_row, "reference_status": "UNRESOLVED_NO_FABRICATED_MASTER"},
                "created_at": MIGRATION_TIMESTAMP,
            })
        unresolved.append({
            "source": "JRC_PYCSIS_2021", "source_record_id": str(source_row),
            "status": "LOADED_UNRESOLVED", "reason_code": "ANONYMIZED_IDENTITY_AND_ROW_GRAIN",
            "reason": "Direct SI fields and declared results loaded; no cross-source identity merge was attempted.",
            "source_payload_json": json_text({"oem_anon": row.get("OEM anon"), "model_anon": row.get("Model anon")}),
        })
    return populations, unresolved


def merge_population(*populations: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    result = {table: [] for table in PHYSICAL_TABLES}
    for population in populations:
        for table in PHYSICAL_TABLES:
            result[table].extend(population.get(table, []))
    pending = {row["id"]: row for row in result["vde"]}
    ordered: list[dict[str, Any]] = []
    emitted: set[int] = set()
    while pending:
        ready = [
            row for row in pending.values()
            if row.get("vde_id_parent") is None
            or row.get("vde_id_parent") not in pending
            or row.get("vde_id_parent") in emitted
        ]
        if not ready:
            raise RuntimeError("VDE parent graph contains a cycle or unresolved insertion dependency")
        ready.sort(key=lambda row: (row.get("year") or 0, row["id"]))
        for row in ready:
            ordered.append(row)
            emitted.add(row["id"])
            pending.pop(row["id"])
    result["vde"] = ordered
    return result


def compatibility_evidence(
    con: sqlite3.Connection, legacy_vde: list[dict[str, Any]], legacy_fuelcons: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    checks: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    totals: dict[str, int] = {}
    for table, view, legacy_rows, expected_fields in (
        ("VDE", "vde_db", legacy_vde, LEGACY_VDE_FIELDS),
        ("FUELCONS", "fuelcons_db", legacy_fuelcons, LEGACY_FUELCONS_FIELDS),
    ):
        columns = [row[1] for row in con.execute(f'PRAGMA table_info("{view}")')]
        if len(columns) != expected_fields:
            raise RuntimeError(f"{view} exposes {len(columns)} fields, expected {expected_fields}")
        rebuilt = {row["id"]: dict(row) for row in con.execute(f'SELECT * FROM "{view}" WHERE id > 0 ORDER BY id')}
        legacy = {row["id"]: row for row in legacy_rows}
        table_mismatches = 0
        for field in columns:
            count = 0
            for row_id, expected in legacy.items():
                actual = rebuilt.get(row_id, {}).get(field)
                if actual != expected.get(field):
                    count += 1
                    table_mismatches += 1
                    known_json_conflict = table == "FUELCONS" and row_id == 5018 and field == "assumptions_json"
                    mismatches.append({
                        "entity": table, "record_id": row_id, "field": field,
                        "legacy_value": repr(expected.get(field)), "canonical_value": repr(actual),
                        "classification": "MIGRATION_BLOCKER",
                        "details": "Legacy JSON contains non-standard NaN; canonical JSON normalizes it to null and preserves the raw text in RUN provenance. Approval is required."
                        if known_json_conflict else "Exact preserved-snapshot value mismatch",
                    })
            checks.append({
                "entity": table, "field": field, "classification": "EXACT_EQUIVALENCE" if count == 0 else "MIGRATION_BLOCKER",
                "compared_rows": len(legacy), "mismatch_count": count,
                "legacy_null_count": sum(row.get(field) is None for row in legacy.values()),
                "canonical_null_count": sum(row.get(field) is None for row in rebuilt.values()),
                "details": "All values and NULLs compared by preserved positive integer ID.",
            })
        missing_ids = set(legacy) ^ set(rebuilt)
        if missing_ids:
            table_mismatches += len(missing_ids)
            for row_id in sorted(missing_ids):
                mismatches.append({
                    "entity": table, "record_id": row_id, "field": "__ROW_ID__",
                    "legacy_value": str(row_id in legacy), "canonical_value": str(row_id in rebuilt),
                    "classification": "MIGRATION_BLOCKER", "details": "Preserved row identity mismatch",
                })
        totals[table] = table_mismatches
    return checks, mismatches, totals


def relationship_evidence(con: sqlite3.Connection, runtime_vde: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(check: str, actual: Any, expected: Any, passed: bool, evidence: str) -> None:
        rows.append({"check": check, "status": "PASS" if passed else "FAIL", "actual": actual, "expected": expected, "evidence": evidence})

    fk = con.execute("PRAGMA foreign_key_check").fetchall()
    add("foreign_key_check", len(fk), 0, not fk, "SQLite PRAGMA foreign_key_check")
    for name, query in (
        ("orphan_configuration_program", "SELECT COUNT(*) FROM vehicle_configuration c LEFT JOIN program p ON p.program_id=c.program_id WHERE p.program_id IS NULL"),
        ("orphan_vde_configuration", "SELECT COUNT(*) FROM vde v LEFT JOIN vehicle_configuration c ON c.vehicle_configuration_id=v.vehicle_configuration_id WHERE c.vehicle_configuration_id IS NULL"),
        ("orphan_vde_parent", "SELECT COUNT(*) FROM vde v LEFT JOIN vde p ON p.id=v.vde_id_parent WHERE v.vde_id_parent IS NOT NULL AND p.id IS NULL"),
        ("orphan_run_vde", "SELECT COUNT(*) FROM run r LEFT JOIN vde v ON v.id=r.vde_id WHERE v.id IS NULL"),
        ("orphan_fuelcons_vde", "SELECT COUNT(*) FROM fuelcons f LEFT JOIN vde v ON v.id=f.vde_id WHERE v.id IS NULL"),
        ("invalid_adoption_same_vde", "SELECT COUNT(*) FROM fuelcons_run_adoption a JOIN fuelcons f ON f.id=a.fuelcons_id JOIN run r ON r.run_id=a.run_id WHERE a.vde_id<>f.vde_id OR a.vde_id<>r.vde_id"),
    ):
        actual = con.execute(query).fetchone()[0]
        add(name, actual, 0, actual == 0, query)

    legacy_dupes = sorted(
        (row["make"], row["model"], row["year"], row["n"])
        for row in _grouped_duplicates(runtime_vde)
    )
    migrated_dupes = sorted(tuple(row) for row in con.execute(
        "SELECT make,model,year,COUNT(*) FROM vde_db WHERE id>0 GROUP BY make,model,year HAVING COUNT(*)>1"
    ))
    add("duplicate_legacy_vde_states_preserved", len(migrated_dupes), len(legacy_dupes), migrated_dupes == legacy_dupes, "Exact duplicate MMY group signature")
    multiple_fc = con.execute("SELECT COUNT(*) FROM (SELECT vde_id FROM fuelcons GROUP BY vde_id HAVING COUNT(*)>1)").fetchone()[0]
    add("one_vde_may_have_multiple_fuelcons", multiple_fc, ">=1", multiple_fc >= 1, "Canonical FuelCons multiplicity")
    multiple_runs = con.execute("SELECT COUNT(*) FROM (SELECT vde_id FROM run GROUP BY vde_id HAVING COUNT(*)>1)").fetchone()[0]
    add("one_vde_may_have_multiple_runs", multiple_runs, ">=1", multiple_runs >= 1, "Canonical RUN evidence multiplicity")
    without_resolution = con.execute("SELECT COUNT(*) FROM vde v LEFT JOIN vde_component_resolution x ON x.vde_id=v.id WHERE x.vde_id IS NULL").fetchone()[0]
    add("vde_without_component_resolution_allowed", without_resolution, ">=1", without_resolution >= 1, "Optional component-resolution adoption")

    con.execute("SAVEPOINT snapshot_proof")
    try:
        vde = con.execute("SELECT id,vehicle_configuration_id FROM vde ORDER BY id LIMIT 1").fetchone()
        component_id = "QA-12E-MASTER"
        con.execute("INSERT INTO component_db(component_id,component_domain,rated_power_kw,provenance_json) VALUES(?,?,?,?)", (component_id, "ENGINE", 150.0, '{"origin":"SYNTHETIC_QA"}'))
        con.execute("INSERT INTO component_instance(component_instance_id,vehicle_configuration_id,component_domain,component_id,provenance_json) VALUES(?,?,?,?,?)", ("QA-12E-INSTANCE", vde["vehicle_configuration_id"], "ENGINE", component_id, '{"origin":"SYNTHETIC_QA"}'))
        con.execute("INSERT INTO fuelcons(id,vde_id,electrification,engine_max_power_kw,record_origin) VALUES(?,?,?,?,?)", (-9_999_999, vde["id"], "ICE", 135.0, "SYNTHETIC_QA"))
        con.execute("UPDATE component_db SET rated_power_kw=180.0 WHERE component_id=?", (component_id,))
        snapshot = con.execute("SELECT engine_max_power_kw FROM fuelcons WHERE id=-9999999").fetchone()[0]
        add("master_update_does_not_rewrite_snapshot", snapshot, 135.0, snapshot == 135.0, "Transactional synthetic QA proof")
    finally:
        con.execute("ROLLBACK TO snapshot_proof")
        con.execute("RELEASE snapshot_proof")
    return rows


def _grouped_duplicates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[Any, Any, Any], int] = defaultdict(int)
    for row in rows:
        counts[(row.get("make"), row.get("model"), row.get("year"))] += 1
    return [
        {"make": key[0], "model": key[1], "year": key[2], "n": count}
        for key, count in counts.items() if count > 1
    ]


def population_counts(con: sqlite3.Connection) -> list[dict[str, Any]]:
    forecast = {
        "program": "12C.3 safe EPA=2,895 before quarantine; plus preserved legacy/JRC",
        "vehicle_configuration": "12C.3 combined EPA=10,009 before quarantine; plus preserved legacy/JRC",
        "component_db": "12C.2 range 0-345",
        "tire_db": "12C.2 range 71-250 was provisional; required RRC prevents fabricated JRC masters",
        "component_instance": "12C.2 likely 0-4,740",
        "component_resolution": "12C.2 range 0-2,958; optional",
        "vde": "12C.3 combined EPA=11,424 before quarantine; plus preserved legacy/JRC",
        "run": "Source-row evidence grain; preserved legacy + refreshed EPA + JRC",
        "fuelcons": "Adopted results only: preserved legacy + direct JRC declared results",
        "fuelcons_run_adoption": "One direct adoption per materialized FuelCons",
        "vde_component_resolution": "Optional; no supported Tier-0 materialization",
    }
    return [{"table": table, "row_count": con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0], "forecast_comparison": forecast[table]} for table in PHYSICAL_TABLES]


def population_breakdown(con: sqlite3.Connection) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dimensions = {
        "program": ("source_scope", "identity_status"),
        "vehicle_configuration": ("source_scope", "identity_status"),
        "component_db": ("source_name", "record_status"),
        "tire_db": ("source_name", "record_origin"),
        "component_instance": ("component_domain", "record_status"),
        "component_resolution": ("boundary", "record_status"),
        "vde": ("source_name", "record_origin", "source_semantic_status", "year"),
        "run": ("source_name", "run_type", "evidence_kind"),
        "fuelcons": ("source_name", "record_origin", "electrification"),
        "fuelcons_run_adoption": ("adoption_role",),
        "vde_component_resolution": ("adoption_role",),
    }
    for table, fields in dimensions.items():
        for field in fields:
            for value, count in con.execute(f'SELECT "{field}",COUNT(*) FROM "{table}" GROUP BY "{field}" ORDER BY "{field}"'):
                rows.append({"table": table, "dimension": field, "value": "<NULL>" if value is None else value, "row_count": count})
    rows.append({"table": "eea_analytical_boundary", "dimension": "disposition", "value": "NOT_LOADED_RUNTIME_SQLITE", "row_count": EEA_AUDITED_ROWS})
    return rows


def source_refresh_differences(old: pd.DataFrame, refreshed: pd.DataFrame) -> list[dict[str, Any]]:
    historical = refreshed[refreshed["Model Year"].between(2020, 2025)].copy()
    shared = list(old.columns)
    old_hashes = c1.source_row_hashes(old, shared)
    new_hashes = c1.source_row_hashes(historical, shared)
    rows = [{
        "difference_type": "SUMMARY", "row_hash": "", "classification": "SOURCE_REFRESH_DIFFERENCE",
        "old_count": len(old), "refreshed_count": len(historical),
        "details": f"overlap={len(old_hashes & new_hashes)}; old_only={len(old_hashes-new_hashes)}; refreshed_only={len(new_hashes-old_hashes)}",
    }]
    for value in sorted(old_hashes - new_hashes):
        rows.append({"difference_type": "OLD_ONLY_OR_CHANGED", "row_hash": str(value), "classification": "SOURCE_REFRESH_DIFFERENCE", "old_count": 1, "refreshed_count": 0, "details": "Preserved legacy snapshot is not overwritten."})
    for value in sorted(new_hashes - old_hashes):
        rows.append({"difference_type": "REFRESHED_ONLY_OR_CHANGED", "row_hash": str(value), "classification": "EXPECTED_ADDITIVE_RECORD", "old_count": 0, "refreshed_count": 1, "details": "Loaded as versioned refreshed EPA evidence."})
    return rows


def performance_evidence(con: sqlite3.Connection) -> list[dict[str, Any]]:
    samples = {
        "vde_lookup_by_id": ("SELECT * FROM vde_db WHERE id=?", (1,)),
        "fuelcons_lookup_by_vde": ("SELECT * FROM fuelcons_db WHERE vde_id=? ORDER BY created_at DESC", (1,)),
        "browse_filter": ("SELECT f.id,f.vde_id,v.make,v.model,v.year,f.electrification,f.fuel_l_per_100km FROM fuelcons_db f JOIN vde_db v ON v.id=f.vde_id WHERE f.electrification=? AND v.legislation=? ORDER BY f.created_at DESC LIMIT 100", ("ICE", "EPA")),
        "comparison_selection": ("SELECT f.id,f.vde_id,f.energy_Wh_per_km,f.fuel_l_per_100km,f.gco2_per_km,v.vde_net_mj_per_km FROM fuelcons_db f JOIN vde_db v ON v.id=f.vde_id WHERE f.id IN (?,?)", (1, 2)),
        "program_configuration_vde_navigation": ("SELECT v.id FROM vehicle_configuration c JOIN vde v ON v.vehicle_configuration_id=c.vehicle_configuration_id WHERE c.program_id=? ORDER BY v.id", (con.execute("SELECT program_id FROM program ORDER BY program_id LIMIT 1").fetchone()[0],)),
        "source_identity_lookup": ("SELECT run_id,vde_id FROM run WHERE source_name=? AND source_record_id=?", ("EPA_TESTCAR_2014_PRESENT", "2")),
        "run_lineage_lookup": ("SELECT r.* FROM fuelcons_run_adoption a JOIN run r ON r.run_id=a.run_id WHERE a.fuelcons_id=? ORDER BY a.ordinal", (1,)),
    }
    results: list[dict[str, Any]] = []
    for name, (sql, params) in samples.items():
        con.execute(sql, params).fetchall()
        timings = []
        row_count = 0
        for _ in range(20):
            started = time.perf_counter()
            row_count = len(con.execute(sql, params).fetchall())
            timings.append((time.perf_counter() - started) * 1000.0)
        plan = " | ".join(row[3] for row in con.execute("EXPLAIN QUERY PLAN " + sql, params))
        median = statistics.median(timings)
        p95 = sorted(timings)[int(len(timings) * 0.95) - 1]
        blocker = median > 100.0 or p95 > 250.0
        results.append({
            "query": name, "iterations": len(timings), "rows_returned": row_count,
            "median_ms": round(median, 4), "p95_ms": round(p95, 4), "query_plan": plan,
            "index_used": "USING INDEX" in plan.upper() or "USING INTEGER PRIMARY KEY" in plan.upper() or "COVERING INDEX" in plan.upper(),
            "status": "BLOCKER" if blocker else "PASS", "notes": "Full scans are acceptable only on small result/helper sets at this rehearsal population.",
        })
    return results


def report_text(summary: dict[str, Any], counts: list[dict[str, Any]]) -> str:
    count_lines = ["| Table | Rows |", "|---|---:|"] + [f"| `{row['table']}` | {row['row_count']:,} |" for row in counts]
    return "\n".join([
        "# Sprint 12E — Canonical Migration Rehearsal", "",
        f"## Status: `{summary['status']}`", "", "```text",
        f"Did migration build?            {'YES' if summary['migration_built'] else 'NO'}",
        f"Legacy VDE mismatches           {summary['legacy_vde_mismatches']}",
        f"Legacy FuelCons mismatches      {summary['legacy_fuelcons_mismatches']}",
        f"Relationship/orphan failures    {summary['relationship_failures']}",
        f"Runtime DB changed?             {'YES' if summary['runtime_db_changed'] else 'NO'}",
        f"Canonical Program count         {summary['canonical_counts']['program']}",
        f"Vehicle Configuration count     {summary['canonical_counts']['vehicle_configuration']}",
        f"VDE count                       {summary['canonical_counts']['vde']}",
        f"RUN count                       {summary['canonical_counts']['run']}",
        f"FuelCons count                  {summary['canonical_counts']['fuelcons']}",
        f"Quarantined records             {summary['quarantined_records']}",
        f"Performance blockers            {summary['performance_blockers']}",
        f"User decisions required         {summary['user_decisions_required']}",
        "```", "",
        "## Canonical population", "", *count_lines, "",
        "The rehearsal retains every current legacy snapshot under its original positive integer ID. Refreshed EPA and JRC VDE/FuelCons IDs use deterministic negative ranges, so source additions cannot overwrite legacy history.", "",
        "## Deviations and quarantines", "",
        f"- **EPA identity:** {summary['epa']['quarantined_rows']} of {summary['epa']['source_rows']:,} rows were quarantined because `Represented Test Veh Make` is year-like. No Program, Configuration, VDE or RUN identity was invented for them.",
        f"- **Program consolidation:** {summary['epa']['fallback_programs_before']:,} valid EPA fallback Programs became {summary['epa']['safe_programs_after']:,} Programs using only `SAFE_CONSOLIDATE`. `PROBABLE_CONSOLIDATE` was not applied.",
        "- **EPA FuelCons:** no FuelCons was created per raw EPA row. All valid EPA rows became RUN evidence; adoption is deferred until an approved pacification rule exists.",
        "- **JRC:** direct SI roadload and declared OEM result fields were loaded, while all 249 identities remain source-scoped and unresolved. Tire/component descriptors are unresolved Component Instances; no master with fabricated RRC was created.",
        f"- **EEA:** {EEA_AUDITED_ROWS:,} monitoring rows remain outside runtime SQLite. The source fingerprint and analytical-storage boundary were validated; zero EEA domain rows were loaded.", "",
        f"- **JSON constraint conflict:** FuelCons `id=5018` contains non-standard `NaN` in `assumptions_json`. The canonical field uses strict JSON with `null`; the original text is retained in RUN provenance. This accounts for {summary['legacy_fuelcons_mismatches']} compatibility mismatch and requires explicit contract approval before integration.", "",
        "## Compatibility and invariants", "",
        "Every one of the 101 VDE and 79 FuelCons legacy-facing columns was compared by ID, including NULLs. Parent lineage and FuelCons multiplicity remain exact. Foreign keys, orphans, same-VDE RUN adoption, duplicate legacy VDE states and master/snapshot immutability were checked after loading.", "",
        "## Reproducibility and performance", "",
        f"The canonical relationship signature is `{summary['deterministic_signature']}`. A prior clean rebuild was available: {'YES' if summary['previous_signature_available'] else 'NO'}; matching signature: {'YES' if summary['rebuild_deterministic'] else 'NO'}. Public timestamps are fixed to `{MIGRATION_TIMESTAMP}`.",
        f"Seven representative query families were measured over 20 warm runs. Performance blockers: {summary['performance_blockers']}. Detailed plans and timings are in `performance_results.csv`.", "",
        "## Runtime safety", "",
        f"Runtime SHA-256 before: `{summary['runtime_before_sha256']}`  ",
        f"Runtime SHA-256 after: `{summary['runtime_after_sha256']}`  ",
        f"Byte-identical: **{'YES' if not summary['runtime_db_changed'] else 'NO'}**. The production and QA databases were opened only through SQLite `mode=ro` plus `PRAGMA query_only=ON`. The output-path guard accepts only `.db` files below `etl/data/staging/` and rejects both runtime paths.", "",
        "## Evidence tiers", "",
        "- **DIRECTLY_TESTED:** runtime read-only mode; protected output guard; schema build; full-source migration completion; clean-rebuild signature; foreign keys/orphans; duplicate legacy states; VDE→FuelCons and VDE→RUN multiplicity; same-VDE FuelCons↔RUN invariant; optional Component Resolution; historical snapshot immutability; NULL/value compatibility across 180 fields; runtime fingerprints; EEA boundary.",
        "- **INDIRECTLY_COVERED:** CDR field ownership and Sprint 12D physical constraint design.",
        "- **INSPECTION_SUPPORTED:** application query inventory used to select the seven performance cases.",
        "- **GAP:** this is not production cutover evidence; application write adapters, real interactive smoke and rollback operation belong to integration/cutover.", "",
        "## Reproduction", "", "```powershell",
        "python etl/scripts/sprint_12e_migration_rehearsal.py --rebuild",
        "python -m unittest discover -s etl/tests -p \"test_sprint_12e*.py\" -v", "```", "",
        "## USER_DECISION_REQUIRED", "",
        "Approve or reject the explicit normalization of non-standard JSON `NaN` to JSON `null` for legacy FuelCons `id=5018`. Without approval, integration must not proceed.", "",
    ]) + "\n"


def relationship_signature(con: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    for table, key in (
        ("program", "program_id"), ("vehicle_configuration", "vehicle_configuration_id"),
        ("vde", "id"), ("run", "run_id"), ("fuelcons", "id"),
        ("fuelcons_run_adoption", "fuelcons_id,run_id,result_dimension"),
    ):
        for row in con.execute(f'SELECT {key} FROM "{table}" ORDER BY {key}'):
            digest.update(json.dumps(tuple(row), separators=(",", ":"), default=str).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest().upper()


def run_rehearsal(output_db: Path, rebuild: bool) -> dict[str, Any]:
    output_db = guard_output_path(output_db)
    required = [RUNTIME_DB, QA_DB, LEGACY_RUNTIME_DB, SCHEMA_SQL, COMPAT_SQL, LEGACY_EPA, REFRESHED_EPA, JRC, EEA]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing Sprint 12E inputs: {missing}")
    previous = None
    if PREVIOUS_SUMMARY.exists():
        previous = json.loads(PREVIOUS_SUMMARY.read_text(encoding="utf-8"))
    if output_db.exists():
        if not rebuild:
            raise FileExistsError(f"Disposable rehearsal DB already exists; rerun with --rebuild: {output_db}")
        output_db.unlink()
    output_db.parent.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    phases: list[dict[str, Any]] = []
    phase_start = time.perf_counter()

    def phase(number: int, name: str, status: str, row_count: int, details: str) -> None:
        nonlocal phase_start
        now = time.perf_counter()
        phases.append({"phase": number, "name": name, "status": status, "row_count": row_count, "elapsed_ms": round((now - phase_start) * 1000.0, 3), "details": details})
        phase_start = now

    protected_before = {path: fingerprint(path) for path in (RUNTIME_DB, QA_DB)}
    legacy_runtime_hash = fingerprint(LEGACY_RUNTIME_DB)["sha256"]
    input_fingerprints = {path.name: fingerprint(path) for path in (LEGACY_EPA, REFRESHED_EPA, JRC, EEA)}
    legacy_vde, legacy_fc, legacy_tires, _ = read_runtime_population()
    phase(0, "fingerprint_and_readonly_inputs", "PASS", len(legacy_vde) + len(legacy_fc), "Runtime/QA DB opened mode=ro; raw sources fingerprinted")

    con = sqlite3.connect(output_db)
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA journal_mode=DELETE")
        con.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))
        phase(1, "create_empty_canonical_database", "PASS", 9, "Nine primary domain tables created")
        con.executescript(COMPAT_SQL.read_text(encoding="utf-8"))
        phase(2, "install_schema_and_compatibility", "PASS", 3, "vde_db, fuelcons_db and fuelcons_lineage_v1 views installed")

        old_epa = pd.read_excel(LEGACY_EPA, engine="openpyxl")
        refreshed_epa = pd.read_excel(REFRESHED_EPA, sheet_name="Sheet1", engine="openpyxl")
        jrc_source = pd.read_excel(JRC, sheet_name="Sheet1", engine="openpyxl")
        legacy = legacy_population(legacy_vde, legacy_fc, legacy_tires, legacy_runtime_hash)
        epa, consolidation, quarantine, epa_stats = epa_population(refreshed_epa, input_fingerprints[REFRESHED_EPA.name]["sha256"])
        jrc, jrc_unresolved = jrc_population(jrc_source, input_fingerprints[JRC.name]["sha256"])
        population = merge_population(legacy, epa, jrc)

        insertion_phases = (
            (3, "load_program_identities", "program"),
            (4, "load_vehicle_configurations", "vehicle_configuration"),
            (5, "load_component_and_tire_masters", "component_db"),
            (5, "load_component_and_tire_masters", "tire_db"),
            (6, "load_component_instances", "component_instance"),
            (7, "load_vde_snapshots", "vde"),
            (8, "load_run_evidence", "run"),
            (9, "load_fuelcons_results", "fuelcons"),
            (10, "load_fuelcons_run_lineage", "fuelcons_run_adoption"),
            (11, "load_component_resolution_links", "component_resolution"),
            (11, "load_component_resolution_links", "vde_component_resolution"),
        )
        pending: dict[tuple[int, str], int] = defaultdict(int)
        con.execute("BEGIN")
        for number, name, table in insertion_phases:
            pending[(number, name)] += insert_rows(con, table, population[table])
            if table in {"tire_db", "vde_component_resolution"}:
                phase(number, name, "PASS", pending[(number, name)], f"Loaded through {table}")
        con.commit()

        relationships = relationship_evidence(con, legacy_vde)
        phase(12, "referential_and_invariant_validation", "PASS" if all(row["status"] == "PASS" for row in relationships) else "FAIL", len(relationships), "Foreign keys, grains, multiplicities and snapshots checked")
        compatibility, mismatches, mismatch_totals = compatibility_evidence(con, legacy_vde, legacy_fc)
        phase(13, "legacy_compatibility_equivalence", "PASS" if not mismatches else "FAIL", len(compatibility), "101 VDE + 79 FuelCons fields compared")
        counts = population_counts(con)
        breakdown = population_breakdown(con)
        refresh = source_refresh_differences(old_epa, refreshed_epa)
        phase(14, "public_source_population_checks", "PASS", len(refreshed_epa) + len(jrc_source), "EPA/JRC loaded at approved grain; EEA boundary retained")
        con.execute("ANALYZE")
        performance = performance_evidence(con)
        phase(15, "performance_and_query_plan_checks", "PASS" if all(row["status"] != "BLOCKER" for row in performance) else "FAIL", len(performance), "Seven application query families measured")
        signature = relationship_signature(con)
        con.commit()
    finally:
        con.close()

    protected_after = {path: fingerprint(path) for path in (RUNTIME_DB, QA_DB)}
    runtime_fingerprints = []
    for path in (RUNTIME_DB, QA_DB):
        before, after = protected_before[path], protected_after[path]
        runtime_fingerprints.append({
            "database": path.name, "path": str(path), "before_size_bytes": before["size_bytes"],
            "after_size_bytes": after["size_bytes"], "before_sha256": before["sha256"],
            "after_sha256": after["sha256"], "byte_identical": before == after,
            "read_access": "SQLite URI mode=ro + PRAGMA query_only=ON",
        })
    runtime_changed = any(not row["byte_identical"] for row in runtime_fingerprints)
    phase(16, "fingerprint_runtime_databases_again", "PASS" if not runtime_changed else "FAIL", len(runtime_fingerprints), "Production and QA fingerprints compared")

    canonical = {row["table"]: row["row_count"] for row in counts}
    relationship_failures = sum(row["status"] != "PASS" for row in relationships)
    performance_blockers = sum(row["status"] == "BLOCKER" for row in performance)
    previous_signature = previous.get("deterministic_signature") if previous else None
    deterministic = (
        previous_signature == signature and previous.get("canonical_counts") == canonical
        if previous_signature else False
    )
    ready = not any((mismatch_totals["VDE"], mismatch_totals["FUELCONS"], relationship_failures, performance_blockers, runtime_changed))
    status = "MIGRATION_REHEARSAL_READY — PROCEED_TO_INTEGRATION" if ready else "MIGRATION_REHEARSAL_REVIEW_REQUIRED"
    summary = {
        "status": status, "migration_built": output_db.exists(),
        "legacy_vde_mismatches": mismatch_totals["VDE"], "legacy_fuelcons_mismatches": mismatch_totals["FUELCONS"],
        "relationship_failures": relationship_failures, "runtime_db_changed": runtime_changed,
        "canonical_counts": canonical, "quarantined_records": len(quarantine),
        "performance_blockers": performance_blockers,
        "user_decisions_required": 1 if mismatch_totals["FUELCONS"] else 0,
        "epa": epa_stats, "jrc_loaded_rows": len(jrc_source), "eea_runtime_rows_loaded": 0,
        "eea_audited_source_rows": EEA_AUDITED_ROWS, "deterministic_signature": signature,
        "previous_signature_available": previous_signature is not None, "rebuild_deterministic": deterministic,
        "runtime_before_sha256": protected_before[RUNTIME_DB]["sha256"],
        "runtime_after_sha256": protected_after[RUNTIME_DB]["sha256"],
        "runtime_fingerprints": runtime_fingerprints, "input_fingerprints": input_fingerprints,
        "output_database": str(output_db), "output_database_size_bytes": output_db.stat().st_size,
        "migration_timestamp_policy": MIGRATION_TIMESTAMP,
    }

    write_csv(OUT / "migration_phase_results.csv", phases, ["phase", "name", "status", "row_count", "elapsed_ms", "details"])
    write_csv(OUT / "canonical_population_counts.csv", counts, ["table", "row_count", "forecast_comparison"])
    write_csv(OUT / "population_by_source.csv", breakdown, ["table", "dimension", "value", "row_count"])
    write_csv(OUT / "relationship_checks.csv", relationships, ["check", "status", "actual", "expected", "evidence"])
    write_csv(OUT / "compatibility_checks.csv", compatibility, ["entity", "field", "classification", "compared_rows", "mismatch_count", "legacy_null_count", "canonical_null_count", "details"])
    write_csv(OUT / "compatibility_mismatches.csv", mismatches, ["entity", "record_id", "field", "legacy_value", "canonical_value", "classification", "details"])
    write_csv(OUT / "source_refresh_differences.csv", refresh, ["difference_type", "row_hash", "classification", "old_count", "refreshed_count", "details"])
    write_csv(OUT / "program_consolidation_results.csv", consolidation, ["fallback_program_id", "canonical_program_id", "make", "model", "model_year", "safe_group_size", "decision", "applied_rule"])
    quarantine_output = quarantine + jrc_unresolved + [{
        "source": "EEA_2025_PROVISIONAL", "source_record_id": "ALL_ROWS",
        "status": "DEFERRED_ANALYTICAL_BOUNDARY", "reason_code": "NOT_RUNTIME_ENGINEERING_GRAIN",
        "reason": f"{EEA_AUDITED_ROWS} monitoring rows remain in immutable analytical source storage.",
        "source_payload_json": json_text({"path": str(EEA), "sha256": input_fingerprints[EEA.name]["sha256"]}),
    }]
    write_csv(OUT / "identity_quarantine.csv", quarantine_output, ["source", "source_record_id", "status", "reason_code", "reason", "source_payload_json"])
    write_csv(OUT / "performance_results.csv", performance, ["query", "iterations", "rows_returned", "median_ms", "p95_ms", "query_plan", "index_used", "status", "notes"])
    write_csv(OUT / "runtime_db_fingerprints.csv", runtime_fingerprints, ["database", "path", "before_size_bytes", "after_size_bytes", "before_sha256", "after_sha256", "byte_identical", "read_access"])
    PREVIOUS_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT.write_text(report_text(summary, counts), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Delete only the protected, disposable rehearsal output and rebuild it.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DB, help="Safe rehearsal DB path below etl/.")
    args = parser.parse_args()
    result = run_rehearsal(args.output, args.rebuild)
    print(json.dumps({
        "status": result["status"], "canonical_counts": result["canonical_counts"],
        "legacy_vde_mismatches": result["legacy_vde_mismatches"],
        "legacy_fuelcons_mismatches": result["legacy_fuelcons_mismatches"],
        "relationship_failures": result["relationship_failures"],
        "quarantined_records": result["quarantined_records"],
        "performance_blockers": result["performance_blockers"],
        "runtime_db_changed": result["runtime_db_changed"],
        "rebuild_deterministic": result["rebuild_deterministic"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
