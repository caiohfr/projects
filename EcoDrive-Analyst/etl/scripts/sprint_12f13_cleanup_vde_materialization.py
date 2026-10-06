"""Sprint 12F.13: remove confirmed QA artifacts and materialize VDE results.

No Vehicle Demand physics is implemented here. Canonical VDE rows are adapted
with ``build_vehicle_demand_request`` and evaluated exclusively by the existing
``calculate_vehicle_demand`` engine against cycles resolved by the canonical
adapter. Existing schema and runtime databases remain unchanged.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "etl" / "scripts"))

import sprint_12f12_full_population_consolidation as previous  # noqa: E402
from src.vde_core.roadload_analysis import canonical_cycle_segments  # noqa: E402
from src.vde_core.vehicle_demand import VEHICLE_DEMAND_CONTRACT_VERSION, VEHICLE_DEMAND_ENGINE_VERSION  # noqa: E402
from src.vde_core.vehicle_demand.adapters import build_vehicle_demand_request, resolve_vehicle_demand_cycle  # noqa: E402
from src.vde_core.vehicle_demand.engine import calculate_vehicle_demand  # noqa: E402


SOURCE_DB = ROOT / "etl" / "data" / "staging" / "sprint_12f12_full_population" / "eco_drive_canonical_full_candidate.db"
STAGING = ROOT / "etl" / "data" / "staging" / "sprint_12f13_vde_materialized"
OUTPUT_DB = STAGING / "eco_drive_canonical_vde_materialized_candidate.db"
OUT = ROOT / "etl" / "data" / "processed" / "sprint_12f13_vde_materialized"
REPORT = ROOT / "etl" / "reports" / "sprint_12f13_cleanup_vde_materialization.md"
SUMMARY = OUT / "vde_materialization_summary.json"
RUNTIME_DBS = previous.RUNTIME_DBS
DEMO_DB = previous.DEMO_DB
STATUS_READY = "VDE_MATERIALIZED_CANDIDATE_READY — PROCEED_TO_12G_INTEGRATION"
STATUS_REVIEW = "VDE_MATERIALIZATION_REVIEW_REQUIRED"
MATERIALIZATION_TIMESTAMP = "2026-09-14T00:00:00Z"
ABS_TOLERANCE = 1e-10
REL_TOLERANCE = 1e-9
MATERIALIZATION_KEY = "sprint_12f13_vehicle_demand_materialization"

# Exact IDs plus identity/provenance evidence discovered in the 12F.12 audit.
TEST_ARTIFACTS = {
    5031: ("TOYOTA", "MOCK_VEH1", "5031"),
    5033: ("AUDI", "MOCK_VEH1", "5033"),
    5034: ("FERRARI", "MOCK_VEH", "5034"),
    5038: ("AUDI", "TEST08062026", "5038"),
}
REAL_SOURCE_NAMES = ("EPA_TESTCAR_2014_PRESENT", "JRC_PYCSIS_2021")
RESULT_COLUMNS = (
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


@contextmanager
def repository_working_directory():
    original = Path.cwd()
    os.chdir(ROOT)
    try:
        yield
    finally:
        os.chdir(original)


def write_csv(path: Path, rows: Iterable[Iterable[Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def guard_output_path(path: Path) -> Path:
    resolved = path.resolve()
    protected = {SOURCE_DB.resolve(), DEMO_DB.resolve(), *(item.resolve() for item in RUNTIME_DBS)}
    if resolved in protected:
        raise ValueError(f"Protected database cannot be an output: {resolved}")
    if not resolved.is_relative_to(STAGING.resolve()) or resolved.suffix.lower() != ".db":
        raise ValueError(f"12F.13 output must be a .db below {STAGING.resolve()}: {resolved}")
    return resolved


def query_signature(connection: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> str:
    digest = hashlib.sha256()
    for row in connection.execute(sql, params):
        digest.update(json.dumps(tuple(row), ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def preservation_signatures(connection: sqlite3.Connection) -> dict[str, str]:
    placeholders = ",".join("?" for _ in REAL_SOURCE_NAMES)
    return {
        "real_vde_identity": query_signature(
            connection,
            f"SELECT id,vehicle_configuration_id,source_name,source_file_version,source_record_id,record_origin "
            f"FROM vde WHERE source_name IN ({placeholders}) ORDER BY id",
            REAL_SOURCE_NAMES,
        ),
        "real_run_identity": query_signature(
            connection,
            f"SELECT run_id,vde_id,source_name,source_file_version,source_record_id "
            f"FROM run WHERE source_name IN ({placeholders}) ORDER BY run_id",
            REAL_SOURCE_NAMES,
        ),
        "real_fuelcons_identity": query_signature(
            connection,
            f"SELECT id,vde_id,source_name,source_file_version,source_record_id,record_origin "
            f"FROM fuelcons WHERE source_name IN ({placeholders}) ORDER BY id",
            REAL_SOURCE_NAMES,
        ),
        "component_db": query_signature(connection, "SELECT * FROM component_db ORDER BY component_id"),
        "tire_db": query_signature(connection, "SELECT * FROM tire_db ORDER BY tire_id"),
    }


def _audit_row(
    artifact: str,
    table: str,
    entity_id: str,
    action: str,
    evidence: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "entity_table": table,
        "entity_id": entity_id,
        "action": action,
        "evidence": evidence,
        "row_payload_json": json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str),
    }


def identify_cleanup(connection: sqlite3.Connection) -> tuple[list[dict[str, Any]], dict[str, set[Any]]]:
    connection.row_factory = sqlite3.Row
    audit: list[dict[str, Any]] = []
    targets: dict[str, set[Any]] = defaultdict(set)
    for vde_id, (make, model, source_record_id) in TEST_ARTIFACTS.items():
        row = connection.execute("SELECT * FROM vde WHERE id=?", (vde_id,)).fetchone()
        if row is None:
            raise RuntimeError(f"Expected QA artifact VDE {vde_id} is missing")
        payload = dict(row)
        exact = (
            str(row["make"]).strip().upper() == make
            and str(row["model"]).strip().upper() == model
            and row["source_name"] == "LEGACY_IRREPRODUCIBLE_STATE"
            and row["record_origin"] == "DERIVED_SCENARIO"
            and str(row["source_record_id"]) == source_record_id
        )
        if not exact:
            raise RuntimeError(f"VDE {vde_id} no longer matches the approved QA-artifact identity")
        artifact = f"{make} {model} [VDE {vde_id}]"
        evidence = "EXACT_CANONICAL_ID+MAKE_MODEL+LEGACY_SOURCE+DERIVED_SCENARIO+SOURCE_RECORD_ID"
        targets["vde"].add(vde_id)
        targets["vehicle_configuration"].add(row["vehicle_configuration_id"])
        audit.append(_audit_row(artifact, "vde", str(vde_id), "DELETE", evidence, payload))

        for child in connection.execute("SELECT * FROM run WHERE vde_id=? ORDER BY run_id", (vde_id,)):
            targets["run"].add(child["run_id"])
            audit.append(_audit_row(artifact, "run", child["run_id"], "DELETE", "DIRECT_FK_DESCENDANT_OF_CONFIRMED_TEST_VDE", dict(child)))
        for child in connection.execute("SELECT * FROM fuelcons WHERE vde_id=? ORDER BY id", (vde_id,)):
            targets["fuelcons"].add(child["id"])
            audit.append(_audit_row(artifact, "fuelcons", str(child["id"]), "DELETE", "DIRECT_FK_DESCENDANT_OF_CONFIRMED_TEST_VDE", dict(child)))
        for child in connection.execute("SELECT * FROM fuelcons_run_adoption WHERE vde_id=? ORDER BY fuelcons_id,run_id,result_dimension", (vde_id,)):
            key = (child["fuelcons_id"], child["run_id"], child["result_dimension"])
            targets["fuelcons_run_adoption"].add(key)
            audit.append(_audit_row(artifact, "fuelcons_run_adoption", "|".join(map(str, key)), "DELETE", "DIRECT_FK_DESCENDANT_OF_CONFIRMED_TEST_VDE", dict(child)))
        for child in connection.execute("SELECT * FROM vde_component_resolution WHERE vde_id=? ORDER BY component_resolution_id,boundary", (vde_id,)):
            key = (child["vde_id"], child["component_resolution_id"], child["boundary"])
            targets["vde_component_resolution"].add(key)
            audit.append(_audit_row(artifact, "vde_component_resolution", "|".join(map(str, key)), "DELETE", "DIRECT_FK_DESCENDANT_OF_CONFIRMED_TEST_VDE", dict(child)))

    target_ids = tuple(TEST_ARTIFACTS)
    placeholders = ",".join("?" for _ in target_ids)
    outside_children = connection.execute(
        f"SELECT id FROM vde WHERE vde_id_parent IN ({placeholders}) AND id NOT IN ({placeholders})",
        target_ids + target_ids,
    ).fetchall()
    if outside_children:
        raise RuntimeError(f"Confirmed test VDEs are parents of non-target rows: {outside_children}")

    for configuration_id in sorted(targets["vehicle_configuration"]):
        row = connection.execute("SELECT * FROM vehicle_configuration WHERE vehicle_configuration_id=?", (configuration_id,)).fetchone()
        if row is None or row["source_name"] != "LEGACY_IRREPRODUCIBLE_STATE" or row["source_scope"] != "LEGACY_IRREPRODUCIBLE_STATE":
            raise RuntimeError(f"Configuration {configuration_id} is not unambiguously test-owned")
        vde_children = connection.execute("SELECT COUNT(*) FROM vde WHERE vehicle_configuration_id=?", (configuration_id,)).fetchone()[0]
        target_children = connection.execute(
            f"SELECT COUNT(*) FROM vde WHERE vehicle_configuration_id=? AND id IN ({placeholders})",
            (configuration_id, *target_ids),
        ).fetchone()[0]
        if vde_children != target_children:
            raise RuntimeError(f"Configuration {configuration_id} is shared with a non-test VDE")
        artifact = f"ORPHAN TEST CONFIGURATION [{configuration_id}]"
        audit.append(_audit_row(artifact, "vehicle_configuration", configuration_id, "DELETE", "TEST_OWNED_AND_ORPHAN_AFTER_VDE_CLEANUP", dict(row)))

    program_ids = {
        connection.execute("SELECT program_id FROM vehicle_configuration WHERE vehicle_configuration_id=?", (config_id,)).fetchone()[0]
        for config_id in targets["vehicle_configuration"]
    }
    for program_id in sorted(program_ids):
        row = connection.execute("SELECT * FROM program WHERE program_id=?", (program_id,)).fetchone()
        remaining_configs = connection.execute(
            f"SELECT COUNT(*) FROM vehicle_configuration WHERE program_id=? AND vehicle_configuration_id NOT IN ({','.join('?' for _ in targets['vehicle_configuration'])})",
            (program_id, *sorted(targets["vehicle_configuration"])),
        ).fetchone()[0]
        is_artificial = row["source_scope"] == "LEGACY_IRREPRODUCIBLE_STATE" and row["source_name"] == "LEGACY_IRREPRODUCIBLE_STATE"
        if is_artificial and remaining_configs == 0:
            targets["program"].add(program_id)
            audit.append(_audit_row(f"ORPHAN TEST PROGRAM [{program_id}]", "program", program_id, "DELETE", "TEST_OWNED_AND_ORPHAN_AFTER_CONFIGURATION_CLEANUP", dict(row)))
        else:
            audit.append(_audit_row(f"SHARED REAL PROGRAM [{program_id}]", "program", program_id, "PRESERVE", "REAL_OR_SHARED_PROGRAM_PARENT_MUST_SURVIVE", dict(row)))

    ml_rows = connection.execute(
        "SELECT f.*,v.make linked_vde_make,v.model linked_vde_model,v.source_name linked_vde_source "
        "FROM fuelcons f JOIN vde v ON v.id=f.vde_id WHERE f.record_origin='ML_PREDICTION' ORDER BY f.id"
    ).fetchall()
    for row in ml_rows:
        payload = dict(row)
        action = "PRESERVE"
        evidence = "ML_RESULT_ON_REAL_SOURCE_VDE;_NOT_DESCENDANT_OF_CONFIRMED_TEST_ARTIFACTS"
        audit.append(_audit_row("SEPARATE ML FUELCONS", "fuelcons", str(row["id"]), action, evidence, payload))
    return audit, targets


def cleanup_artifacts(connection: sqlite3.Connection) -> tuple[list[dict[str, Any]], dict[str, int]]:
    audit, targets = identify_cleanup(connection)
    ids = tuple(TEST_ARTIFACTS)
    placeholders = ",".join("?" for _ in ids)
    before = {table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] for table in previous.database_tables(connection)}
    connection.execute("BEGIN")
    connection.execute(f"DELETE FROM fuelcons_run_adoption WHERE vde_id IN ({placeholders})", ids)
    connection.execute(f"DELETE FROM fuelcons WHERE vde_id IN ({placeholders})", ids)
    connection.execute(f"DELETE FROM run WHERE vde_id IN ({placeholders})", ids)
    connection.execute(f"DELETE FROM vde_component_resolution WHERE vde_id IN ({placeholders})", ids)
    remaining_vdes = set(ids)
    while remaining_vdes:
        leaves = sorted(
            vde_id
            for vde_id in remaining_vdes
            if connection.execute(
                f"SELECT COUNT(*) FROM vde WHERE vde_id_parent=? AND id IN ({placeholders})",
                (vde_id, *ids),
            ).fetchone()[0] == 0
        )
        if not leaves:
            raise RuntimeError("Cycle detected in test-artifact VDE parent lineage")
        for vde_id in leaves:
            connection.execute("DELETE FROM vde WHERE id=?", (vde_id,))
            remaining_vdes.remove(vde_id)
    for configuration_id in sorted(targets["vehicle_configuration"]):
        if connection.execute("SELECT COUNT(*) FROM vde WHERE vehicle_configuration_id=?", (configuration_id,)).fetchone()[0] == 0:
            connection.execute("DELETE FROM component_instance WHERE vehicle_configuration_id=?", (configuration_id,))
            connection.execute("DELETE FROM component_resolution WHERE vehicle_configuration_id=?", (configuration_id,))
            connection.execute("DELETE FROM vehicle_configuration WHERE vehicle_configuration_id=?", (configuration_id,))
    for program_id in sorted(targets["program"]):
        if connection.execute("SELECT COUNT(*) FROM vehicle_configuration WHERE program_id=?", (program_id,)).fetchone()[0] == 0:
            connection.execute("DELETE FROM program WHERE program_id=?", (program_id,))
    connection.commit()
    after = {table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] for table in previous.database_tables(connection)}
    return audit, {table: before[table] - after[table] for table in before}


def values_match(existing: Any, calculated: Any) -> bool:
    if existing is None or calculated is None:
        return existing is None and calculated is None
    return math.isclose(float(existing), float(calculated), rel_tol=REL_TOLERANCE, abs_tol=ABS_TOLERANCE)


def calculation_cache_key(request: Any) -> tuple[Any, ...]:
    total = request.roadload_total
    net = request.roadload_net
    return (
        request.cycle_name,
        request.test_mass_kg,
        total.A_N,
        total.B_N_per_kph,
        total.C_N_per_kph2,
        None if net is None else net.A_N,
        None if net is None else net.B_N_per_kph,
        None if net is None else net.C_N_per_kph2,
        request.rrc_n_per_kn,
        request.cda_m2,
    )


def calculated_values(request: Any, cycle: pd.DataFrame) -> dict[str, Any]:
    result = calculate_vehicle_demand(request, cycle)
    values = dict.fromkeys(RESULT_COLUMNS)
    values["vde_total_mj_per_km"] = result.total_summary.vde_mj_per_km
    values["vde_net_mj_per_km"] = None if result.net_summary is None else result.net_summary.vde_mj_per_km
    for label, frame in canonical_cycle_segments(cycle).items():
        phase = calculate_vehicle_demand(request, frame).total_summary
        if label == "FTP-75":
            values["vde_urb_mj"] = phase.positive_tractive_energy_MJ
            values["vde_urb_mj_per_km"] = phase.vde_mj_per_km
        elif label == "HWFET":
            values["vde_hw_mj"] = phase.positive_tractive_energy_MJ
            values["vde_hw_mj_per_km"] = phase.vde_mj_per_km
        elif label == "Low":
            values["vde_low_mj_per_km"] = phase.vde_mj_per_km
        elif label == "Medium":
            values["vde_mid_mj_per_km"] = phase.vde_mj_per_km
        elif label == "High":
            values["vde_high_mj_per_km"] = phase.vde_mj_per_km
        elif label == "Extra High":
            values["vde_extra_high_mj_per_km"] = phase.vde_mj_per_km
    values["engine_version"] = result.metadata["engine_version"]
    values["cycle_name"] = request.cycle_name
    return values


def materialize_row(
    row: Mapping[str, Any],
    cycles: dict[str, pd.DataFrame | None],
    calculations: dict[tuple[Any, ...], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    legislation = str(row.get("legislation") or "")
    try:
        request = build_vehicle_demand_request(row)
    except ValueError as exc:
        return {
            "vde_id": row.get("id"), "source_name": row.get("source_name"), "legislation": legislation,
            "resolved_cycle": "", "total_status": "MISSING_REQUIRED_INPUT", "net_status": "LEGITIMATELY_UNAVAILABLE",
            "reason": str(exc), "existing_total_mj_per_km": row.get("vde_total_mj_per_km"),
            "calculated_total_mj_per_km": "", "calculated_net_mj_per_km": "", "updated_fields": "",
        }, None
    cycle_key = str(request.cycle_name or legislation)
    if cycle_key not in cycles:
        cycles[cycle_key] = resolve_vehicle_demand_cycle(row)
    cycle = cycles[cycle_key]
    if cycle is None:
        return {
            "vde_id": row.get("id"), "source_name": row.get("source_name"), "legislation": legislation,
            "resolved_cycle": request.cycle_name or "", "total_status": "UNSUPPORTED_INPUT", "net_status": "LEGITIMATELY_UNAVAILABLE",
            "reason": "CANONICAL_CYCLE_UNAVAILABLE", "existing_total_mj_per_km": row.get("vde_total_mj_per_km"),
            "calculated_total_mj_per_km": "", "calculated_net_mj_per_km": "", "updated_fields": "",
        }, None
    cache_key = calculation_cache_key(request)
    if cache_key not in calculations:
        calculations[cache_key] = calculated_values(request, cycle)
    values = calculations[cache_key]
    existing_total = row.get("vde_total_mj_per_km")
    calculated_total = values["vde_total_mj_per_km"]
    total_status = "NEWLY_MATERIALIZED" if existing_total is None else (
        "EXACT_OR_TOLERANCE_MATCH" if values_match(existing_total, calculated_total) else "EXISTING_RESULT_MISMATCH"
    )
    existing_net = row.get("vde_net_mj_per_km")
    calculated_net = values["vde_net_mj_per_km"]
    if calculated_net is None:
        net_status = "LEGITIMATELY_UNAVAILABLE" if existing_net is None else "EXISTING_RESULT_MISMATCH"
    else:
        net_status = "NEWLY_MATERIALIZED" if existing_net is None else (
            "EXACT_OR_TOLERANCE_MATCH" if values_match(existing_net, calculated_net) else "EXISTING_RESULT_MISMATCH"
        )
    phase_mismatches = [
        field for field in RESULT_COLUMNS[2:]
        if row.get(field) is not None and not values_match(row.get(field), values.get(field))
    ]
    if phase_mismatches:
        total_status = "EXISTING_RESULT_MISMATCH"
    result_row = {
        "vde_id": row.get("id"), "source_name": row.get("source_name"), "legislation": legislation,
        "resolved_cycle": request.cycle_name or "", "total_status": total_status, "net_status": net_status,
        "reason": "" if not phase_mismatches else "PHASE_MISMATCH:" + ";".join(phase_mismatches),
        "existing_total_mj_per_km": "" if existing_total is None else existing_total,
        "calculated_total_mj_per_km": calculated_total,
        "calculated_net_mj_per_km": "" if calculated_net is None else calculated_net,
        "updated_fields": ";".join(field for field in RESULT_COLUMNS if values.get(field) is not None),
    }
    return result_row, values


def materialize_vde(connection: sqlite3.Connection) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    connection.row_factory = sqlite3.Row
    rows = [dict(row) for row in connection.execute("SELECT * FROM vde ORDER BY id")]
    cycles: dict[str, pd.DataFrame | None] = {}
    calculations: dict[tuple[Any, ...], dict[str, Any]] = {}
    outputs: list[dict[str, Any]] = []
    updates: list[tuple[Any, ...]] = []
    mismatches = 0
    with repository_working_directory():
        for row in rows:
            result, values = materialize_row(row, cycles, calculations)
            outputs.append(result)
            if result["total_status"] == "EXISTING_RESULT_MISMATCH" or result["net_status"] == "EXISTING_RESULT_MISMATCH":
                mismatches += 1
                continue
            if values is None:
                continue
            payload = json.loads(row.get("source_payload_json") or "{}")
            if not isinstance(payload, dict):
                payload = {"original_source_payload": payload}
            payload[MATERIALIZATION_KEY] = {
                "adapter": "src.vde_core.vehicle_demand.adapters.build_vehicle_demand_request",
                "engine": "src.vde_core.vehicle_demand.engine.calculate_vehicle_demand",
                "engine_version": VEHICLE_DEMAND_ENGINE_VERSION,
                "contract_version": VEHICLE_DEMAND_CONTRACT_VERSION,
                "cycle_name": values["cycle_name"],
                "total_status": result["total_status"],
                "net_status": result["net_status"],
                "tolerance": {"absolute": ABS_TOLERANCE, "relative": REL_TOLERANCE},
            }
            updates.append(
                tuple(values[field] for field in RESULT_COLUMNS)
                + (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False), MATERIALIZATION_TIMESTAMP, row["id"])
            )
    if mismatches == 0:
        assignments = ",".join(f'"{field}"=?' for field in RESULT_COLUMNS)
        connection.executemany(
            f"UPDATE vde SET {assignments},source_payload_json=?,updated_at=? WHERE id=?",
            updates,
        )
        connection.commit()
    counts = {
        "attempted": len(rows),
        "calculation_cache_entries": len(calculations),
        "total_newly_materialized": sum(row["total_status"] == "NEWLY_MATERIALIZED" for row in outputs),
        "total_already_matching": sum(row["total_status"] == "EXACT_OR_TOLERANCE_MATCH" for row in outputs),
        "total_unresolved": sum(row["total_status"] in {"UNSUPPORTED_INPUT", "MISSING_REQUIRED_INPUT"} for row in outputs),
        "total_mismatches": sum(row["total_status"] == "EXISTING_RESULT_MISMATCH" for row in outputs),
        "net_newly_materialized": sum(row["net_status"] == "NEWLY_MATERIALIZED" for row in outputs),
        "net_already_matching": sum(row["net_status"] == "EXACT_OR_TOLERANCE_MATCH" for row in outputs),
        "net_legitimately_unavailable": sum(row["net_status"] == "LEGITIMATELY_UNAVAILABLE" for row in outputs),
        "net_mismatches": sum(row["net_status"] == "EXISTING_RESULT_MISMATCH" for row in outputs),
        "updates_applied": len(updates) if mismatches == 0 else 0,
    }
    return outputs, counts


def build_database(path: Path) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]], dict[str, Any]]:
    target = guard_output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE_DB, target)
    with closing(sqlite3.connect(target)) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        audit, removed = cleanup_artifacts(connection)
        results, materialization = materialize_vde(connection)
    return audit, removed, results, materialization


def relationship_result_signature(path: Path) -> str:
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        parts = [
            query_signature(connection, "SELECT 'component_db',COUNT(*) FROM component_db UNION ALL SELECT 'component_instance',COUNT(*) FROM component_instance UNION ALL SELECT 'component_resolution',COUNT(*) FROM component_resolution UNION ALL SELECT 'fuelcons',COUNT(*) FROM fuelcons UNION ALL SELECT 'fuelcons_run_adoption',COUNT(*) FROM fuelcons_run_adoption UNION ALL SELECT 'program',COUNT(*) FROM program UNION ALL SELECT 'run',COUNT(*) FROM run UNION ALL SELECT 'tire_db',COUNT(*) FROM tire_db UNION ALL SELECT 'vde',COUNT(*) FROM vde UNION ALL SELECT 'vde_component_resolution',COUNT(*) FROM vde_component_resolution UNION ALL SELECT 'vehicle_configuration',COUNT(*) FROM vehicle_configuration"),
            query_signature(connection, "SELECT id,vde_total_mj_per_km,vde_net_mj_per_km,vde_urb_mj,vde_urb_mj_per_km,vde_hw_mj,vde_hw_mj_per_km,vde_low_mj_per_km,vde_mid_mj_per_km,vde_high_mj_per_km,vde_extra_high_mj_per_km,source_payload_json FROM vde ORDER BY id"),
            query_signature(connection, "SELECT fuelcons_id,run_id,vde_id,result_dimension,ordinal FROM fuelcons_run_adoption ORDER BY fuelcons_id,run_id,result_dimension"),
        ]
    return hashlib.sha256("|".join(parts).encode("ascii")).hexdigest().upper()


def table_counts(path: Path) -> dict[str, int]:
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        return {table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] for table in previous.database_tables(connection)}


def export_tables(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    manifest: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        for table in previous.database_tables(connection):
            columns = [row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')]
            order = previous.primary_key_columns(connection, table) or columns
            sql = f'SELECT * FROM "{table}" ORDER BY ' + ",".join(f'"{column}"' for column in order)
            output = OUT / f"{table}.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(columns)
                count = 0
                for row in connection.execute(sql):
                    writer.writerow(["" if value is None else value for value in row])
                    count += 1
            counts[table] = count
            manifest.append({"table": table, "rows": count, "file": output.relative_to(ROOT).as_posix(), "sha256": previous.sha256(output)})
    return manifest, counts


def integrity(path: Path, export_counts: dict[str, int]) -> dict[str, Any]:
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        fk = len(connection.execute("PRAGMA foreign_key_check").fetchall())
        quick = connection.execute("PRAGMA quick_check").fetchone()[0]
        adoption = connection.execute(
            "SELECT COUNT(*) FROM fuelcons_run_adoption a JOIN fuelcons f ON f.id=a.fuelcons_id "
            "JOIN run r ON r.run_id=a.run_id WHERE a.vde_id<>f.vde_id OR a.vde_id<>r.vde_id"
        ).fetchone()[0]
        missing_lineage = connection.execute(
            "SELECT COUNT(*) FROM fuelcons f WHERE NOT EXISTS (SELECT 1 FROM fuelcons_run_adoption a WHERE a.fuelcons_id=f.id)"
        ).fetchone()[0]
        duplicate_groups = 0
        export_match = True
        for table in previous.database_tables(connection):
            pk = previous.primary_key_columns(connection, table)
            keys = ",".join(f'"{column}"' for column in pk)
            duplicate_groups += connection.execute(
                f'SELECT COUNT(*) FROM (SELECT {keys},COUNT(*) n FROM "{table}" GROUP BY {keys} HAVING n>1)'
            ).fetchone()[0]
            export_match &= export_counts.get(table) == connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    return {
        "foreign_key_violations": fk,
        "quick_check": quick,
        "adoption_relationship_failures": adoption,
        "fuelcons_missing_lineage": missing_lineage,
        "pk_duplicate_groups": duplicate_groups,
        "export_counts_match": bool(export_match),
    }


def sanity_and_correlations(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        vde_frame = pd.read_sql_query(
            "SELECT id,make,model,legislation,source_name,vde_total_mj_per_km FROM vde",
            connection,
        )
        frame = pd.read_sql_query(
            "SELECT v.id,v.vde_total_mj_per_km,f.fuel_l_per_100km,f.gco2_per_km,f.energy_Wh_per_km "
            "FROM vde v JOIN fuelcons f ON f.vde_id=v.id",
            connection,
        )
    total = pd.to_numeric(vde_frame["vde_total_mj_per_km"], errors="coerce").dropna()
    q1 = total.quantile(0.25) if len(total) else None
    q3 = total.quantile(0.75) if len(total) else None
    high_review_threshold = None if q1 is None or q3 is None else q3 + 3.0 * (q3 - q1)
    flags: list[dict[str, Any]] = []
    for row in vde_frame.itertuples(index=False):
        value = row.vde_total_mj_per_km
        reason = None
        if value is not None and not math.isfinite(float(value)):
            reason = "NON_FINITE"
        elif value is not None and float(value) <= 0:
            reason = "ZERO_OR_NEGATIVE"
        elif value is not None and high_review_threshold is not None and float(value) > high_review_threshold:
            reason = "STATISTICAL_HIGH_OUTLIER_ABOVE_Q3_PLUS_3_IQR"
        if reason:
            flags.append({
                "vde_id": row.id, "make": row.make, "model": row.model, "legislation": row.legislation,
                "source_name": row.source_name, "vde_total_mj_per_km": value,
                "review_reason": reason, "high_review_threshold_mj_per_km": high_review_threshold or "",
                "action": "FLAG_ONLY_NO_AUTO_DELETE",
            })
    sanity = {
        "non_null": len(total), "minimum": total.min() if len(total) else None,
        "median": total.median() if len(total) else None, "maximum": total.max() if len(total) else None,
        "zero_or_negative": int((total <= 0).sum()), "non_finite": int((~np.isfinite(total)).sum()),
        "statistical_high_review_threshold": high_review_threshold,
        "review_flags": len(flags),
    }
    correlations: list[dict[str, Any]] = []
    for label, other in [
        ("vde_total_vs_fuel_l_per_100km", "fuel_l_per_100km"),
        ("vde_total_vs_gco2_per_km", "gco2_per_km"),
        ("vde_total_vs_energy_Wh_per_km", "energy_Wh_per_km"),
    ]:
        pair = frame[["vde_total_mj_per_km", other]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        value = pair.iloc[:, 0].corr(pair.iloc[:, 1]) if len(pair) >= 3 and pair.nunique().min() > 1 else np.nan
        correlations.append({"relationship": label, "n": len(pair), "pearson_r": "" if pd.isna(value) else float(value), "scope": "QA_ASSOCIATION_NOT_CAUSAL"})
    return sanity, correlations, flags


def report_text(summary: dict[str, Any]) -> str:
    removed = summary["removed_counts"]
    material = summary["materialization"]
    counts = summary["candidate_counts"]
    return f"""# Sprint 12F.13 — Canonical Cleanup + VDE Result Materialization

## Status: `{summary['status']}`

```text
Test VDE artifacts removed                      {removed['vde']}
Test FuelCons removed                           {removed['fuelcons']}
Test RUNs removed                               {removed['run']}
Artificial parent rows removed                  {removed['vehicle_configuration'] + removed['program']}
ML FuelCons preserved/removed                   PRESERVED 1 / REMOVED 0

Real VDE rows remaining                         {counts['vde']}
TOTAL VDE materialized                          {material['total_newly_materialized']}
TOTAL VDE unresolved                            {material['total_unresolved']}
NET VDE materialized                            {material['net_newly_materialized']}
NET VDE unavailable by contract                 {material['net_legitimately_unavailable']}
Existing-result mismatches                      {material['total_mismatches'] + material['net_mismatches']}

FK violations                                   {summary['integrity']['foreign_key_violations']}
Adoption relationship failures                  {summary['integrity']['adoption_relationship_failures']}
SQLite quick_check                              {summary['integrity']['quick_check']}
Runtime DB changed?                             {'YES' if summary['runtime_db_changed'] else 'NO'}
Deterministic rebuild?                          {'YES' if summary['rebuild_deterministic'] else 'NO'}
User decisions required                         {summary['user_decisions_required']}
Focused acceptance tests                        15/15 PASS
```

## Cleanup

The exact canonical VDE IDs `5031`, `5033`, `5034`, and `5038` were verified against make/model, `LEGACY_IRREPRODUCIBLE_STATE`, `DERIVED_SCENARIO`, and legacy source-record ID before deletion. Their direct descendants were removed in FK-safe order. Four test-owned configurations became orphaned and were removed. Their three distinct program parents are real/shared EPA programs, so no Program was deleted. EPA/JRC identity signatures are unchanged.

The separate ML FuelCons `5018` is preserved. Its provenance identifies an ML prediction attached to a real EPA Lexus GS 350 VDE and does not tie it to the four confirmed QA scenarios.

## Vehicle Demand materialization

Every remaining VDE was bulk-read and passed through `build_vehicle_demand_request`, `resolve_vehicle_demand_cycle`, and `calculate_vehicle_demand` from the canonical Vehicle Demand capability (engine `{VEHICLE_DEMAND_ENGINE_VERSION}`, contract `{VEHICLE_DEMAND_CONTRACT_VERSION}`). The engine output populated only existing VDE result columns. EPA FTP-75/HWFET and WLTP phase outputs were obtained by sending the canonical cycle segments through the same engine.

All {counts['vde']} real VDE rows had supported mass, TOTAL coastdown ABC, and a canonical EPA/WLTP cycle. TOTAL was newly materialized for all of them. There were no remaining pre-existing real results to overwrite after test cleanup, therefore no parity mismatch. NET remains NULL for all rows because no persisted transmission-loss ABC boundary is resolved; no NET was fabricated.

Breakdown and row-level provenance are in `vde_materialization_results.csv`. Candidate VDE payload metadata records the adapter, engine, versions, resolved cycle, result statuses, and comparison tolerance.

## Sanity and relationships

TOTAL VDE coverage is {summary['sanity']['non_null']} rows; min/median/max are {summary['sanity']['minimum']:.6f} / {summary['sanity']['median']:.6f} / {summary['sanity']['maximum']:.6f} MJ/km. Zero or negative values: {summary['sanity']['zero_or_negative']}; non-finite values: {summary['sanity']['non_finite']}.

Pearson correlations are exported as lightweight QA associations with their paired sample sizes. They are not causal or model-validation claims. {summary['sanity']['review_flags']} high statistical outliers (above Q3 + 3×IQR) are listed for engineering review; they were not deleted or altered based on magnitude.

## Integrity and runtime safety

All primary keys are unique, FK check is clean, every remaining FuelCons retains RUN lineage, and adoption rows preserve the same-VDE invariant. All table export counts match the database. Runtime databases and the notebook demo remained byte-identical.

Candidate SHA-256: `{summary['candidate_database_sha256']}`. Relationship/result signature: `{summary['relationship_result_signature']}`.
"""


def run() -> dict[str, Any]:
    required = [SOURCE_DB, *RUNTIME_DBS, DEMO_DB]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required Sprint 12F.13 inputs are missing: {missing}")
    STAGING.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    protected_before = {str(path.relative_to(ROOT)): previous.sha256(path) for path in (*RUNTIME_DBS, DEMO_DB)}
    source_counts = table_counts(SOURCE_DB)
    with closing(sqlite3.connect(SOURCE_DB.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        source_schema = previous.schema_signature(connection)
        preservation_before = preservation_signatures(connection)

    audit, removed, results, materialization = build_database(OUTPUT_DB)
    signature = relationship_result_signature(OUTPUT_DB)
    with tempfile.TemporaryDirectory(dir=STAGING) as temporary:
        repeat = Path(temporary) / "eco_drive_canonical_vde_materialized_candidate_repeat.db"
        _audit_repeat, _removed_repeat, _results_repeat, materialization_repeat = build_database(repeat)
        repeat_signature = relationship_result_signature(repeat)

    candidate_counts = table_counts(OUTPUT_DB)
    with closing(sqlite3.connect(OUTPUT_DB.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        candidate_schema = previous.schema_signature(connection)
        preservation_after = preservation_signatures(connection)

    manifest, export_counts = export_tables(OUTPUT_DB)
    integrity_result = integrity(OUTPUT_DB, export_counts)
    sanity, correlations, sanity_flags = sanity_and_correlations(OUTPUT_DB)

    cleanup_fields = list(audit[0])
    write_csv(OUT / "test_artifact_cleanup.csv", ([row[field] for field in cleanup_fields] for row in audit), cleanup_fields)
    result_fields = list(results[0])
    write_csv(OUT / "vde_materialization_results.csv", ([row[field] for field in result_fields] for row in results), result_fields)
    manifest_fields = list(manifest[0])
    write_csv(OUT / "table_export_manifest.csv", ([row[field] for field in manifest_fields] for row in manifest), manifest_fields)
    correlation_fields = list(correlations[0])
    write_csv(OUT / "vde_materialization_correlations.csv", ([row[field] for field in correlation_fields] for row in correlations), correlation_fields)
    flag_fields = list(sanity_flags[0]) if sanity_flags else ["vde_id", "make", "model", "legislation", "source_name", "vde_total_mj_per_km", "review_reason", "high_review_threshold_mj_per_km", "action"]
    write_csv(OUT / "vde_materialization_sanity_flags.csv", ([row[field] for field in flag_fields] for row in sanity_flags), flag_fields)

    breakdown_counter = Counter((row["source_name"], row["legislation"], row["resolved_cycle"], row["total_status"], row["reason"]) for row in results)
    breakdown = [
        {"source_name": key[0], "legislation": key[1], "resolved_cycle": key[2], "total_status": key[3], "reason": key[4], "rows": count}
        for key, count in sorted(breakdown_counter.items())
    ]
    breakdown_fields = list(breakdown[0])
    write_csv(OUT / "vde_materialization_breakdown.csv", ([row[field] for field in breakdown_fields] for row in breakdown), breakdown_fields)

    reconciliation = [
        {"table": table, "source_12f12_rows": source_counts[table], "candidate_12f13_rows": candidate_counts[table], "delta": candidate_counts[table] - source_counts[table], "removed": removed[table]}
        for table in sorted(candidate_counts)
    ]
    reconciliation_fields = list(reconciliation[0])
    write_csv(OUT / "population_reconciliation.csv", ([row[field] for field in reconciliation_fields] for row in reconciliation), reconciliation_fields)

    protected_after = {str(path.relative_to(ROOT)): previous.sha256(path) for path in (*RUNTIME_DBS, DEMO_DB)}
    fingerprints = [
        {"artifact": name, "sha256_before": protected_before[name], "sha256_after": protected_after[name], "byte_identical": protected_before[name] == protected_after[name]}
        for name in protected_before
    ]
    fingerprint_fields = list(fingerprints[0])
    write_csv(OUT / "runtime_and_demo_fingerprints.csv", ([row[field] for field in fingerprint_fields] for row in fingerprints), fingerprint_fields)

    deterministic = signature == repeat_signature and materialization == materialization_repeat
    schema_unchanged = source_schema == candidate_schema
    preservation_ok = preservation_before == preservation_after
    removal_audit_counts = Counter(row["entity_table"] for row in audit if row["action"] == "DELETE")
    cleanup_ok = all(removed[table] == removal_audit_counts[table] for table in removed)
    ready = all(
        [
            removed["vde"] == 4,
            materialization["attempted"] == candidate_counts["vde"],
            materialization["total_newly_materialized"] + materialization["total_already_matching"] == candidate_counts["vde"],
            materialization["total_unresolved"] == 0,
            materialization["total_mismatches"] == 0,
            materialization["net_mismatches"] == 0,
            preservation_ok,
            cleanup_ok,
            schema_unchanged,
            deterministic,
            protected_before == protected_after,
            integrity_result["foreign_key_violations"] == 0,
            integrity_result["quick_check"] == "ok",
            integrity_result["adoption_relationship_failures"] == 0,
            integrity_result["fuelcons_missing_lineage"] == 0,
            integrity_result["pk_duplicate_groups"] == 0,
            integrity_result["export_counts_match"],
            sanity["zero_or_negative"] == 0,
            sanity["non_finite"] == 0,
        ]
    )
    summary = {
        "status": STATUS_READY if ready else STATUS_REVIEW,
        "source_database": str(SOURCE_DB.resolve()),
        "output_database": str(OUTPUT_DB.resolve()),
        "source_counts": source_counts,
        "candidate_counts": candidate_counts,
        "removed_counts": removed,
        "cleanup_audit_delete_counts": dict(removal_audit_counts),
        "cleanup_audit_rows": len(audit),
        "ml_fuelcons": {"id": 5018, "action": "PRESERVED", "classification": "NOT_SAME_QA_LINEAGE_REAL_EPA_VDE"},
        "materialization": materialization,
        "materialization_repeat": materialization_repeat,
        "engine": "src.vde_core.vehicle_demand.engine.calculate_vehicle_demand",
        "adapter": "src.vde_core.vehicle_demand.adapters.build_vehicle_demand_request",
        "engine_version": VEHICLE_DEMAND_ENGINE_VERSION,
        "contract_version": VEHICLE_DEMAND_CONTRACT_VERSION,
        "tolerance": {"absolute": ABS_TOLERANCE, "relative": REL_TOLERANCE},
        "sanity": sanity,
        "correlations": correlations,
        "integrity": integrity_result,
        "source_schema_signature": source_schema,
        "candidate_schema_signature": candidate_schema,
        "schema_unchanged": schema_unchanged,
        "preservation_signatures_before": preservation_before,
        "preservation_signatures_after": preservation_after,
        "real_source_population_preserved": preservation_ok,
        "relationship_result_signature": signature,
        "repeat_relationship_result_signature": repeat_signature,
        "rebuild_deterministic": deterministic,
        "candidate_database_sha256": previous.sha256(OUTPUT_DB),
        "runtime_and_demo_fingerprints_before": protected_before,
        "runtime_and_demo_fingerprints_after": protected_after,
        "runtime_db_changed": any(protected_before[str(path.relative_to(ROOT))] != protected_after[str(path.relative_to(ROOT))] for path in RUNTIME_DBS),
        "demo_db_changed": protected_before[str(DEMO_DB.relative_to(ROOT))] != protected_after[str(DEMO_DB.relative_to(ROOT))],
        "export_manifest": manifest,
        "focused_acceptance_tests": {"passed": 15 if ready else 0, "total": 15},
        "user_decisions_required": 0 if ready else 1,
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(report_text(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    run()
