"""Controlled full-fleet component population and temp-DB materialization.

This module deliberately owns no estimation equations.  Macro resolutions are
produced by :mod:`component_enrichment_pass1`; fine evidence is obtained from
the frozen Pass 1C.5 candidate machinery.  This module only decides how those
results may be projected into the historical VDE component slots.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Iterable

from src.vde_core.component_enrichment_pass1 import (
    ESTIMATOR_VERSION,
    _load_population,
    execute_pass1,
    file_sha256,
    inventory,
    route_architecture,
)
from src.vde_core.component_minor_buildup_pilot import (
    _read_only,
    load_legacy_lookup,
    load_resolved_population,
    load_tire_references,
)
from src.vde_core.component_minor_buildup_refinement import enrich_applicability
from src.vde_core.component_physics_vector_search import load_all_component_references
from src.vde_core.component_reference_discrimination import (
    _build_all_pools,
    _enrich_with_source,
    _load_technical_metadata,
    generate_internal_brake_references,
)
from src.vde_core.component_rollingminor_metadata_completion import resolve_application_class_v14
from src.vde_core.component_vector_search import CandidateVector, load_historical_tire_evidence


METHOD_VERSION = "SPRINT12_COMPONENT_POPULATION_SCALE_MATERIALIZATION_V1.0"
FINE_METHODOLOGY_STATUS = "CONDITIONAL_AUDIT_ONLY_NOT_FLEET_VALIDATED"
FROZEN_PASS1C5_BRAKE_KEYS = {
    ("CROSSOVER", "4WD"),
    ("CROSSOVER", "RWD"),
    ("PASSENGER_LIGHT", "AWD"),
    ("PASSENGER_LIGHT", "RWD"),
    ("PASSENGER_STANDARD", "AWD"),
    ("PASSENGER_STANDARD", "RWD"),
    ("PICKUP_LIGHT_DUTY", "RWD"),
    ("SUV", "AWD"),
    ("SUV", "FWD"),
}

PROJECTION_FIELDS = (
    "tire_A_final", "tire_B_final", "tire_C_final",
    "aero_C_coef_Npkph2",
    "trans_A_coef_N", "trans_B_coef_Npkph", "trans_C_coef_Npkph2",
    "brake_A_coef_N", "brake_B_coef_Npkph", "brake_C_coef_Npkph2",
    "parasitic_A_coef_N", "parasitic_B_coef_Npkph", "parasitic_C_coef_Npkph2",
)
WRITABLE_FIELDS = {
    "tire_A_final", "tire_B_final", "tire_C_final", "aero_C_coef_Npkph2",
    "trans_A_coef_N", "trans_B_coef_Npkph", "trans_C_coef_Npkph2",
    "provenance_json",
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_object(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _csv_bytes(rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> bytes:
    selected = list(fields or (list(rows[0]) if rows else ()))
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=selected, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + handle.getvalue()).encode("utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_csv_bytes(rows, fields))


def _table_counts(connection: sqlite3.Connection) -> dict[str, int]:
    names = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    return {
        table: int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        for table in ("vde", "vehicle_configuration", "run", "fuelcons", "component_resolution", "vde_component_resolution")
        if table in names
    }


def _column_names(connection: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')]


def _protected_columns(connection: sqlite3.Connection) -> list[str]:
    """Protect every VDE column except the explicitly authorized projection surface."""
    ignored = WRITABLE_FIELDS | {"updated_at"}
    return [name for name in _column_names(connection, "vde") if name not in ignored]


def _column_hash(connection: sqlite3.Connection, column: str) -> str:
    digest = sha256()
    for row in connection.execute(f'SELECT id,"{column}" FROM vde ORDER BY id'):
        digest.update(_stable_json([row[0], row[1]]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def protected_snapshot(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    return [
        {"table": "vde", "column": column, "sha256": _column_hash(connection, column)}
        for column in _protected_columns(connection)
    ]


def _macro_resolution_map(connection: sqlite3.Connection) -> dict[int, dict[str, dict[str, Any]]]:
    result: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    sql = """
        SELECT l.vde_id,l.boundary,l.component_resolution_id,r.*
        FROM vde_component_resolution AS l
        JOIN component_resolution AS r
          ON r.component_resolution_id=l.component_resolution_id
        WHERE l.adoption_role='ADOPTED'
          AND r.estimator_version=?
          AND r.record_status='ACTIVE'
        ORDER BY l.vde_id,l.boundary,l.component_resolution_id
    """
    for raw in connection.execute(sql, (ESTIMATOR_VERSION,)):
        row = dict(raw)
        vde_id, boundary = int(row["vde_id"]), str(row["boundary"])
        if boundary in result[vde_id]:
            raise ValueError(f"Multiple adopted macro resolutions for VDE {vde_id}, boundary {boundary}")
        result[vde_id][boundary] = row
    return result


def _fine_evidence(
    db_path: Path,
    legacy_db_path: Path,
    component_catalog_path: Path,
    tire_zip_path: Path,
) -> tuple[dict[int, dict[str, CandidateVector]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Run frozen Pass 1C.5 candidate generation without aggregate promotion."""
    connection = _read_only(db_path)
    legacy = _read_only(legacy_db_path)
    try:
        vehicles = load_resolved_population(connection, load_legacy_lookup(legacy))
        _load_technical_metadata(connection, vehicles)
        historical_tires = load_historical_tire_evidence(legacy)
        base_references = load_all_component_references(component_catalog_path)
        additions, addition_audit = generate_internal_brake_references(
            legacy, base_references, FROZEN_PASS1C5_BRAKE_KEYS
        )
        actual_keys = {(item.application_class, item.drive) for item in additions}
        if actual_keys != FROZEN_PASS1C5_BRAKE_KEYS:
            raise RuntimeError(
                "Frozen Pass 1C.5 internal Brake reference set changed: "
                f"expected={sorted(FROZEN_PASS1C5_BRAKE_KEYS)} actual={sorted(actual_keys)}"
            )
        addition_ids = {item.component_id for item in additions}
        references = _enrich_with_source([*base_references, *additions], addition_ids)
        tires = load_tire_references(tire_zip_path)
        fine: dict[int, dict[str, CandidateVector]] = {}
        for vehicle in vehicles:
            vehicle.update(resolve_application_class_v14(vehicle))
            if not vehicle.get("eligible_for_validation"):
                continue
            pools = _build_all_pools(vehicle, references, tires, historical_tires)
            selected = {domain: candidates[0] for domain, candidates in pools.items() if candidates}
            if selected:
                fine[int(vehicle["vde_id"])] = selected
        addition_rows = [row for row in addition_audit if row.get("status") == "ADDED_EPHEMERAL_PILOT_REFERENCE"]
        existing = {}
        for row in connection.execute(
            "SELECT component_resolution_id,boundary,provenance_json FROM component_resolution ORDER BY component_resolution_id"
        ):
            provenance = _json_object(row["provenance_json"])
            component_id = provenance.get("component_id") or provenance.get("synthetic_reference_component_id")
            if component_id:
                existing[str(component_id)] = {
                    "component_resolution_id": str(row["component_resolution_id"]),
                    "boundary": str(row["boundary"]),
                }
        for row in addition_rows:
            existing[str(row["component_id"])] = {
                "component_resolution_id": str(row["component_resolution_id"]),
                "boundary": "BRAKE",
                "new_frozen_reference": True,
                **row,
            }
        return fine, addition_rows, existing
    finally:
        connection.close()
        legacy.close()


def _triplet(resolution: dict[str, Any] | None) -> tuple[Any, Any, Any]:
    if not resolution:
        return None, None, None
    return (
        resolution.get("resolved_A_N"),
        resolution.get("resolved_B_N_per_kph"),
        resolution.get("resolved_C_N_per_kph2"),
    )


def _existing_classification(row: dict[str, Any]) -> tuple[str, str | None]:
    populated = [field for field in PROJECTION_FIELDS if row.get(field) is not None]
    if not populated:
        return "EMPTY", None
    # The current schema carries no per-field fidelity marker.  Conservative
    # policy therefore treats any pre-existing value as canonical and skips it.
    return "EXISTING_CANONICAL", "EXISTING_HIGHER_QUALITY_OR_UNCLASSIFIED_VALUE:" + ";".join(populated)


def _projection_decision(
    row: dict[str, Any], macro: dict[str, dict[str, Any]], fine: dict[str, CandidateVector] | None,
) -> dict[str, Any]:
    aero = macro.get("AERO")
    rolling = macro.get("ROLLING_MINOR")
    drivetrain = macro.get("DRIVETRAIN_AGGREGATE")
    edrive = macro.get("EDRIVE_AGGREGATE")
    existing_class, conflict = _existing_classification(row)
    reason_codes: list[str] = []
    if fine:
        reason_codes.append("FINE_EVIDENCE_AUDIT_ONLY_METHOD_NOT_FLEET_VALIDATED")
    if not (aero and rolling and (drivetrain or edrive)):
        reason_codes.append("MACRO_DECOMPOSITION_INCOMPLETE")
    if edrive and not drivetrain:
        reason_codes.append("SCHEMA_PROJECTION_UNAVAILABLE_EDRIVE_AGGREGATE")
    if aero and any(abs(float(value or 0.0)) > 1e-12 for value in _triplet(aero)[:2]):
        reason_codes.append("AERO_AB_NOT_REPRESENTABLE_IN_C_ONLY_SCHEMA")
    if conflict:
        reason_codes.append("EXISTING_HIGHER_QUALITY_VALUE")

    can_project = bool(aero and rolling and drivetrain and not conflict)
    can_project = can_project and not any(code == "AERO_AB_NOT_REPRESENTABLE_IN_C_ONLY_SCHEMA" for code in reason_codes)
    projection_mode = "MACRO_DECOMPOSITION" if can_project else "UNRESOLVED"
    write_action = "PROJECT_MACRO" if can_project else "NO_WRITE"
    aero_abc, rolling_abc, drive_abc = _triplet(aero), _triplet(rolling), _triplet(drivetrain or edrive)
    projected = {
        "tire_A_final": rolling_abc[0] if can_project else None,
        "tire_B_final": rolling_abc[1] if can_project else None,
        "tire_C_final": rolling_abc[2] if can_project else None,
        "aero_C_coef_Npkph2": aero_abc[2] if can_project else None,
        "trans_A_coef_N": drive_abc[0] if can_project else None,
        "trans_B_coef_Npkph": drive_abc[1] if can_project else None,
        "trans_C_coef_Npkph2": drive_abc[2] if can_project else None,
    }
    provenance = {
        "decomposition_mode": projection_mode,
        "method": METHOD_VERSION,
        "macro_estimator_version": ESTIMATOR_VERSION,
        "macro_resolution_ids": {
            boundary: value["component_resolution_id"] for boundary, value in sorted(macro.items())
        },
        "tire_slot_semantics": "ROLLING_MINOR_AGGREGATE" if can_project else None,
        "transmission_slot_semantics": "DRIVETRAIN_AGGREGATE" if can_project else None,
        "aero_slot_semantics": "AERO_C_ONLY" if can_project else None,
        "fine_component_identification": False,
        "fine_methodology_status": FINE_METHODOLOGY_STATUS,
        "fine_evidence_domains": sorted(fine or {}),
        "boundary_warnings": [code for code in reason_codes if "BOUNDARY" in code or "SCHEMA" in code],
        "reason_codes": sorted(set(reason_codes)),
    }
    return {
        "projection_mode": projection_mode,
        "write_action": write_action,
        "existing_value_classification": existing_class,
        "conflict_reason": conflict,
        "reason_codes": ";".join(sorted(set(reason_codes))),
        "provenance_summary": _stable_json(provenance),
        **projected,
    }


def _build_manifest(
    db_path: Path, fine_by_vde: dict[int, dict[str, CandidateVector]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    connection = _read_only(db_path)
    try:
        population, _runs, fuelcons = _load_population(connection, None, None)
        macro = _macro_resolution_map(connection)
        slot_fields = ",".join(f'v."{field}"' for field in PROJECTION_FIELDS)
        existing = {
            int(row["id"]): dict(row)
            for row in connection.execute(f"SELECT v.id,v.provenance_json,{slot_fields} FROM vde AS v ORDER BY v.id")
        }
        manifest: list[dict[str, Any]] = []
        resolution_rows: list[dict[str, Any]] = []
        for source in population:
            vde_id = int(source["vde_id"])
            current = existing[vde_id]
            route = route_architecture(source, fuelcons)
            resolutions = macro.get(vde_id, {})
            fine = fine_by_vde.get(vde_id, {})
            decision = _projection_decision(current, resolutions, fine)
            aero = resolutions.get("AERO")
            rolling = resolutions.get("ROLLING_MINOR")
            drive = resolutions.get("DRIVETRAIN_AGGREGATE") or resolutions.get("EDRIVE_AGGREGATE")
            aero_abc, rolling_abc, drive_abc = _triplet(aero), _triplet(rolling), _triplet(drive)
            application_class = None
            if fine:
                application_class = next((candidate.application_class for candidate in fine.values() if candidate.application_class), None)
            macro_statuses = sorted({str(item.get("estimate_status") or "") for item in resolutions.values()})
            macro_status = ";".join(macro_statuses) if macro_statuses else "UNRESOLVED"
            row = {
                "vde_id": vde_id,
                "vehicle_configuration_id": source.get("vehicle_configuration_id"),
                "make": source.get("make"), "model": source.get("model"), "model_year": source.get("year"),
                "architecture_class": route.architecture,
                "application_class": application_class,
                "drive_type": source.get("drive_system"),
                "authoritative_A": source.get("coast_A_N"),
                "authoritative_B": source.get("coast_B_N_per_kph"),
                "authoritative_C": source.get("coast_C_N_per_kph2"),
                "macro_status": macro_status,
                "macro_model_type": route.model_type,
                "macro_estimator_version": ESTIMATOR_VERSION,
                "aero_A": aero_abc[0], "aero_B": aero_abc[1], "aero_C": aero_abc[2],
                "rolling_minor_A": rolling_abc[0], "rolling_minor_B": rolling_abc[1], "rolling_minor_C": rolling_abc[2],
                "drivetrain_aggregate_boundary": drive.get("boundary") if drive else None,
                "drivetrain_aggregate_A": drive_abc[0], "drivetrain_aggregate_B": drive_abc[1], "drivetrain_aggregate_C": drive_abc[2],
                "macro_fit_nrmse_pct": drive.get("fit_nrmse_pct") if drive else None,
                "macro_condition_number": drive.get("condition_number") if drive else None,
                "macro_sensitivity_rel_pct": drive.get("sensitivity_rel_pct") if drive else None,
                "tire_available": int("TIRE" in fine), "brake_available": int("BRAKE" in fine),
                "hub_available": int("HUB_BEARING" in fine), "transmission_available": int("TRANSMISSION" in fine),
                "axle_available": int("AXLE" in fine),
                "fine_evidence_domains": ";".join(sorted(fine)),
                "boundary_status": "BOUNDARY_UNKNOWN" if any(key in fine for key in ("TRANSMISSION", "AXLE")) else None,
                "fine_reason_codes": "FINE_METHOD_NOT_FLEET_VALIDATED" if fine else "REFERENCE_COVERAGE_GAP",
                "projected_tire_semantics": "ROLLING_MINOR_AGGREGATE" if decision["projection_mode"] == "MACRO_DECOMPOSITION" else None,
                "projected_transmission_semantics": "DRIVETRAIN_AGGREGATE" if decision["projection_mode"] == "MACRO_DECOMPOSITION" else None,
                "projected_aero_semantics": "AERO_C_ONLY" if decision["projection_mode"] == "MACRO_DECOMPOSITION" else None,
                **decision,
            }
            manifest.append(row)
            for boundary, resolution in sorted(resolutions.items()):
                resolution_rows.append({
                    "vde_id": vde_id, "evidence_layer": "MACRO", "boundary": boundary,
                    "component_resolution_id": resolution["component_resolution_id"],
                    "reference_ids": None, "resolved_A_N": resolution.get("resolved_A_N"),
                    "resolved_B_N_per_kph": resolution.get("resolved_B_N_per_kph"),
                    "resolved_C_N_per_kph2": resolution.get("resolved_C_N_per_kph2"),
                    "method": resolution.get("method"), "status": resolution.get("estimate_status"),
                    "adoption_role": "ADOPTED", "boundary_status": "CANONICAL_MACRO_BOUNDARY",
                })
            for boundary, candidate in sorted(fine.items()):
                resolution_rows.append({
                    "vde_id": vde_id, "evidence_layer": "FINE_CONDITIONAL_AUDIT", "boundary": boundary,
                    "component_resolution_id": candidate.vector_id,
                    "reference_ids": ";".join(candidate.reference_ids),
                    "resolved_A_N": candidate.abc[0], "resolved_B_N_per_kph": candidate.abc[1],
                    "resolved_C_N_per_kph2": candidate.abc[2],
                    "method": "FROZEN_PASS1C5_RULE_CANDIDATE", "status": "CONDITIONAL",
                    "adoption_role": "SUPPORTING", "boundary_status": candidate.boundary_assumption,
                })
        return manifest, resolution_rows
    finally:
        connection.close()


def _apply_frozen_fine_references(
    connection: sqlite3.Connection,
    fine_by_vde: dict[int, dict[str, CandidateVector]],
    additions: list[dict[str, Any]],
    reference_map: dict[str, dict[str, Any]],
) -> dict[str, int]:
    inserted_resolutions = inserted_links = 0
    for item in additions:
        resolution_id = str(item["component_resolution_id"])
        if connection.execute(
            "SELECT 1 FROM component_resolution WHERE component_resolution_id=?", (resolution_id,)
        ).fetchone() is None:
            provenance = {
                "source": "INTERNAL_LEGACY_COMPONENT_POPULATION_MEDOID",
                "applicability": "EXACT_APPLICATION_CLASS_DRIVE_RULE",
                "component_id": item["component_id"],
                "population_n": item["population_n"],
                "aggregate_target_used": False,
                "fine_projection_allowed": False,
                "methodology_status": FINE_METHODOLOGY_STATUS,
            }
            connection.execute(
                """INSERT INTO component_resolution
                   (component_resolution_id,boundary,method,provenance_json,
                    resolved_A_N,resolved_B_N_per_kph,resolved_C_N_per_kph2,
                    record_status,review_status,estimate_status,estimator_version)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    resolution_id, "BRAKE", "INTERNAL_POPULATION_MEDOID_REFERENCE_V1",
                    _stable_json(provenance), item["A_N"], item["B_N_per_kph"], item["C_N_per_kph2"],
                    "ACTIVE", "CURRENT", "CONDITIONAL", METHOD_VERSION,
                ),
            )
            inserted_resolutions += 1
    for vde_id, candidates in sorted(fine_by_vde.items()):
        for boundary, candidate in sorted(candidates.items()):
            for ordinal, reference_id in enumerate(candidate.reference_ids):
                mapped = reference_map.get(reference_id)
                if not mapped:
                    continue  # Tire/historical vectors remain artifact-only evidence.
                resolution_id = str(mapped["component_resolution_id"])
                exists = connection.execute(
                    "SELECT 1 FROM vde_component_resolution WHERE vde_id=? AND component_resolution_id=? AND boundary=?",
                    (vde_id, resolution_id, boundary),
                ).fetchone()
                if exists:
                    continue
                connection.execute(
                    """INSERT INTO vde_component_resolution
                       (vde_id,component_resolution_id,boundary,adoption_role,ordinal)
                       VALUES (?,?,?,?,?)""",
                    (vde_id, resolution_id, boundary, "SUPPORTING", ordinal),
                )
                inserted_links += 1
    return {"fine_component_resolutions_inserted": inserted_resolutions, "fine_supporting_links_inserted": inserted_links}


def apply_projection_manifest(
    connection: sqlite3.Connection,
    manifest: list[dict[str, Any]],
    fine_by_vde: dict[int, dict[str, CandidateVector]],
    additions: list[dict[str, Any]],
    reference_map: dict[str, dict[str, Any]],
) -> dict[str, int]:
    changed_vdes = 0
    try:
        connection.execute("BEGIN IMMEDIATE")
        fine_counts = _apply_frozen_fine_references(connection, fine_by_vde, additions, reference_map)
        for row in manifest:
            if row["write_action"] != "PROJECT_MACRO":
                continue
            current = connection.execute(
                """SELECT tire_A_final,tire_B_final,tire_C_final,aero_C_coef_Npkph2,
                          trans_A_coef_N,trans_B_coef_Npkph,trans_C_coef_Npkph2,provenance_json
                   FROM vde WHERE id=?""",
                (row["vde_id"],),
            ).fetchone()
            if current is None:
                raise ValueError(f"Missing VDE {row['vde_id']}")
            provenance = _json_object(current[7])
            provenance["component_population_projection"] = json.loads(row["provenance_summary"])
            values = (
                row["tire_A_final"], row["tire_B_final"], row["tire_C_final"],
                row["aero_C_coef_Npkph2"], row["trans_A_coef_N"],
                row["trans_B_coef_Npkph"], row["trans_C_coef_Npkph2"],
                _stable_json(provenance),
            )
            current_values = tuple(current)
            if current_values == values:
                continue
            if any(value is not None for value in current_values[:7]):
                raise RuntimeError(f"Refusing late overwrite of VDE {row['vde_id']}")
            connection.execute(
                """UPDATE vde SET tire_A_final=?,tire_B_final=?,tire_C_final=?,
                          aero_C_coef_Npkph2=?,trans_A_coef_N=?,trans_B_coef_Npkph=?,
                          trans_C_coef_Npkph2=?,provenance_json=? WHERE id=?""",
                (*values, row["vde_id"]),
            )
            changed_vdes += 1
        quick = connection.execute("PRAGMA quick_check").fetchone()[0]
        fk = len(connection.execute("PRAGMA foreign_key_check").fetchall())
        if quick != "ok" or fk:
            raise RuntimeError(f"Temp DB integrity failure: quick_check={quick}, fk_issues={fk}")
        connection.commit()
        return {"vde_rows_changed": changed_vdes, **fine_counts}
    except Exception:
        connection.rollback()
        raise


def _summary_rows(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(manifest)
    modes = Counter(row["projection_mode"] for row in manifest)
    partial = sum(bool(row["fine_evidence_domains"]) for row in manifest)
    resolved_macro = sum(row["macro_status"] != "UNRESOLVED" for row in manifest)
    macro_supported = sum(row["macro_status"] == "SUPPORTED" for row in manifest)
    macro_conditional = sum(row["macro_status"] == "CONDITIONAL" for row in manifest)
    edrive_retained = sum(row["drivetrain_aggregate_boundary"] == "EDRIVE_AGGREGATE" for row in manifest)
    values = [
        ("TOTAL_VDE", total),
        ("MACRO_RESOLVED", resolved_macro),
        ("MACRO_SUPPORTED", macro_supported),
        ("MACRO_CONDITIONAL", macro_conditional),
        ("MACRO_UNRESOLVED", total - resolved_macro),
        ("MACRO_EDRIVE_RETAINED_NO_HISTORICAL_SLOT", edrive_retained),
        ("MACRO_DECOMPOSITION_PROJECTED", modes["MACRO_DECOMPOSITION"]),
        ("FINE_SURROGATE_DECOMPOSITION_PROJECTED", modes["FINE_SURROGATE_DECOMPOSITION"]),
        ("PARTIAL_FINE_EVIDENCE_AUDIT_ONLY", partial),
        ("UNRESOLVED", modes["UNRESOLVED"]),
        ("CONFLICTS_SKIPPED", sum(bool(row["conflict_reason"]) for row in manifest)),
    ]
    return [{"population": name, "count": count, "pct_total": round(100.0 * count / total, 6) if total else 0.0} for name, count in values]


def _architecture_rows(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(
        (row["architecture_class"], row["macro_model_type"], row["projection_mode"], row["macro_status"])
        for row in manifest
    )
    return [
        {"architecture_class": key[0], "model_type": key[1], "projection_mode": key[2], "macro_status": key[3], "vde_count": count}
        for key, count in sorted(counts.items())
    ]


def _failure_rows(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for row in manifest:
        reasons = [reason for reason in str(row["reason_codes"] or "").split(";") if reason]
        if not reasons and row["projection_mode"] != "MACRO_DECOMPOSITION":
            reasons = ["UNSPECIFIED_UNRESOLVED"]
        counts.update(reasons)
    return [{"reason_code": reason, "vde_count": count} for reason, count in counts.most_common()]


def _manual_sample(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def take(rows: list[dict[str, Any]], label: str, limit: int = 20) -> list[dict[str, Any]]:
        result, seen_arch = [], set()
        for row in sorted(rows, key=lambda item: (str(item["architecture_class"]), int(item["vde_id"]))):
            if row["architecture_class"] not in seen_arch:
                result.append({"audit_group": label, **row})
                seen_arch.add(row["architecture_class"])
            if len(result) == limit:
                return result
        selected = {int(row["vde_id"]) for row in result}
        for row in sorted(rows, key=lambda item: int(item["vde_id"])):
            if int(row["vde_id"]) not in selected:
                result.append({"audit_group": label, **row})
            if len(result) == limit:
                break
        return result
    macro = [row for row in manifest if row["projection_mode"] == "MACRO_DECOMPOSITION"]
    fine = [row for row in manifest if row["projection_mode"] == "FINE_SURROGATE_DECOMPOSITION"]
    partial = [row for row in manifest if row["fine_evidence_domains"] and row["projection_mode"] != "FINE_SURROGATE_DECOMPOSITION"]
    unresolved = [row for row in manifest if row["projection_mode"] == "UNRESOLVED"]
    return take(macro, "MACRO_DECOMPOSITION") + take(fine, "FINE_SURROGATE_DECOMPOSITION") + take(partial, "PARTIAL_FINE_AUDIT_ONLY") + take(unresolved, "UNRESOLVED_EDGE")


def _domain_write_summary(before: sqlite3.Connection, after: sqlite3.Connection) -> list[dict[str, Any]]:
    groups = {
        "TIRE": ("tire_A_final", "tire_B_final", "tire_C_final"),
        "AERO": ("aero_C_coef_Npkph2",),
        "TRANSMISSION": ("trans_A_coef_N", "trans_B_coef_Npkph", "trans_C_coef_Npkph2"),
        "BRAKE": ("brake_A_coef_N", "brake_B_coef_Npkph", "brake_C_coef_Npkph2"),
        "PARASITIC": ("parasitic_A_coef_N", "parasitic_B_coef_Npkph", "parasitic_C_coef_Npkph2"),
    }
    rows = []
    for domain, fields in groups.items():
        clause = " OR ".join(f'"{field}" IS NOT NULL' for field in fields)
        before_count = int(before.execute(f"SELECT COUNT(*) FROM vde WHERE {clause}").fetchone()[0])
        after_count = int(after.execute(f"SELECT COUNT(*) FROM vde WHERE {clause}").fetchone()[0])
        rows.append({
            "domain": domain, "existing_before": before_count,
            "newly_populated": max(0, after_count - before_count),
            "preserved_existing": before_count, "remaining_null": int(after.execute(f"SELECT COUNT(*) FROM vde WHERE NOT ({clause})").fetchone()[0]),
        })
    rows.append({"domain": "HUB_AXLE", "existing_before": 0, "newly_populated": 0, "preserved_existing": 0, "remaining_null": int(after.execute("SELECT COUNT(*) FROM vde").fetchone()[0])})
    return rows


def execute_component_population_materialization(
    source_db_path: Path,
    temp_db_path: Path,
    legacy_db_path: Path,
    component_catalog_path: Path,
    tire_zip_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_db_path = Path(source_db_path).resolve(strict=True)
    legacy_db_path = Path(legacy_db_path).resolve(strict=True)
    component_catalog_path = Path(component_catalog_path).resolve(strict=True)
    tire_zip_path = Path(tire_zip_path).resolve(strict=True)
    temp_db_path, output_dir = Path(temp_db_path).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_db_path.parent.mkdir(parents=True, exist_ok=True)
    if temp_db_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing temp DB: {temp_db_path}")

    source_hash_before = file_sha256(source_db_path)
    source_ro = _read_only(source_db_path)
    try:
        source_quick = source_ro.execute("PRAGMA quick_check").fetchone()[0]
        source_fk = len(source_ro.execute("PRAGMA foreign_key_check").fetchall())
        source_counts = _table_counts(source_ro)
        protected_before = protected_snapshot(source_ro)
    finally:
        source_ro.close()
    if source_quick != "ok" or source_fk:
        raise RuntimeError(f"Source DB preflight failed: quick_check={source_quick}, fk_issues={source_fk}")

    macro_dry = execute_pass1(source_db_path, output_dir / "_macro_dry_run", write=False)
    shutil.copy2(source_db_path, temp_db_path)
    macro_write = execute_pass1(
        temp_db_path, output_dir / "_macro_temp_write", write=True, source_db_path=source_db_path
    )
    fine_by_vde, additions, reference_map = _fine_evidence(
        temp_db_path, legacy_db_path, component_catalog_path, tire_zip_path
    )

    manifest_1, resolution_manifest_1 = _build_manifest(temp_db_path, fine_by_vde)
    manifest_2, resolution_manifest_2 = _build_manifest(temp_db_path, fine_by_vde)
    manifest_bytes_1 = _csv_bytes(manifest_1)
    manifest_bytes_2 = _csv_bytes(manifest_2)
    resolution_bytes_1 = _csv_bytes(resolution_manifest_1)
    resolution_bytes_2 = _csv_bytes(resolution_manifest_2)
    if manifest_bytes_1 != manifest_bytes_2 or resolution_bytes_1 != resolution_bytes_2:
        raise RuntimeError("Full-fleet dry-run manifest is not deterministic")
    manifest_hash = sha256(manifest_bytes_1).hexdigest().upper()
    (output_dir / "full_fleet_component_population_manifest.csv").write_bytes(manifest_bytes_1)
    (output_dir / "full_fleet_component_resolution_manifest.csv").write_bytes(resolution_bytes_1)
    _write_csv(output_dir / "full_fleet_population_summary.csv", _summary_rows(manifest_1))
    _write_csv(output_dir / "full_fleet_population_by_architecture.csv", _architecture_rows(manifest_1))
    _write_csv(output_dir / "full_fleet_failure_reasons.csv", _failure_rows(manifest_1))
    conflicts = [row for row in manifest_1 if row["conflict_reason"]]
    _write_csv(output_dir / "existing_value_conflicts.csv", conflicts, list(manifest_1[0]) if manifest_1 else ())
    _write_csv(output_dir / "manual_audit_sample.csv", _manual_sample(manifest_1), list(manifest_1[0]) if manifest_1 else ())

    before_copy = _read_only(source_db_path)
    temp_write = sqlite3.connect(temp_db_path)
    temp_write.row_factory = sqlite3.Row
    try:
        first_apply = apply_projection_manifest(temp_write, manifest_1, fine_by_vde, additions, reference_map)
        first_hash = file_sha256(temp_db_path)
        second_apply = apply_projection_manifest(temp_write, manifest_1, fine_by_vde, additions, reference_map)
        second_hash = file_sha256(temp_db_path)
        temp_counts = _table_counts(temp_write)
        quick = temp_write.execute("PRAGMA quick_check").fetchone()[0]
        fk_issues = len(temp_write.execute("PRAGMA foreign_key_check").fetchall())
        protected_after = protected_snapshot(temp_write)
        domain_summary = _domain_write_summary(before_copy, temp_write)
    finally:
        before_copy.close()
        temp_write.close()

    protected_after_by_col = {row["column"]: row["sha256"] for row in protected_after}
    protected_rows = [
        {
            "table": row["table"], "column": row["column"],
            "sha256_before": row["sha256"],
            "sha256_after": protected_after_by_col[row["column"]],
            "unchanged": row["sha256"] == protected_after_by_col[row["column"]],
        }
        for row in protected_before
    ]
    _write_csv(output_dir / "protected_vde_fields_before_after.csv", protected_rows)
    _write_csv(output_dir / "temp_db_write_summary.csv", domain_summary)

    source_hash_after = file_sha256(source_db_path)
    source_unchanged = source_hash_before == source_hash_after
    protected_unchanged = all(row["unchanged"] for row in protected_rows)
    row_counts_unchanged = all(
        source_counts.get(table) == temp_counts.get(table)
        for table in ("vde", "vehicle_configuration", "run", "fuelcons")
    )
    idempotent = (
        second_apply["vde_rows_changed"] == 0
        and second_apply["fine_component_resolutions_inserted"] == 0
        and second_apply["fine_supporting_links_inserted"] == 0
        and first_hash == second_hash
    )
    (output_dir / "idempotency_check.txt").write_text(
        "\n".join((
            f"manifest_sha256={manifest_hash}",
            "manifest_repeat_identical=YES",
            f"second_run_vde_changes={second_apply['vde_rows_changed']}",
            f"second_run_resolution_inserts={second_apply['fine_component_resolutions_inserted']}",
            f"second_run_link_inserts={second_apply['fine_supporting_links_inserted']}",
            f"temp_db_hash_stable_on_second_run={'YES' if first_hash == second_hash else 'NO'}",
            f"IDEMPOTENT={'YES' if idempotent else 'NO'}",
        )) + "\n", encoding="utf-8"
    )
    (output_dir / "integrity_check.txt").write_text(
        "\n".join((
            f"source_db={source_db_path}", f"source_sha256_before={source_hash_before}",
            f"source_sha256_after={source_hash_after}", f"source_unchanged={'YES' if source_unchanged else 'NO'}",
            f"temp_db={temp_db_path}", f"temp_db_sha256={second_hash}",
            f"quick_check={quick}", f"foreign_key_issues={fk_issues}",
            f"protected_vde_fields_unchanged={'YES' if protected_unchanged else 'NO'}",
            f"core_row_counts_unchanged={'YES' if row_counts_unchanged else 'NO'}",
            f"source_counts={_stable_json(source_counts)}", f"temp_counts={_stable_json(temp_counts)}",
        )) + "\n", encoding="utf-8"
    )

    counts = {row["population"]: row["count"] for row in _summary_rows(manifest_1)}
    gates_clean = all((source_unchanged, protected_unchanged, row_counts_unchanged, idempotent, quick == "ok", fk_issues == 0))
    recommendation = "READY_WITH_REVIEW_ITEMS" if gates_clean else "NOT_READY_FOR_CANONICAL_MATERIALIZATION"
    summary = {
        "method_version": METHOD_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "recommendation": recommendation,
        "source_db": str(source_db_path), "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after, "source_db_written": False,
        "temp_db": str(temp_db_path), "temp_db_sha256": second_hash,
        "manifest_sha256": manifest_hash, "manifest_deterministic": True,
        "population": counts, "macro_dry_run": macro_dry,
        "macro_temp_write": macro_write, "first_apply": first_apply, "second_apply": second_apply,
        "frozen_pass1c5_brake_references": len(additions),
        "quick_check": quick, "foreign_key_issues": fk_issues,
        "protected_vde_fields_unchanged": protected_unchanged,
        "core_row_counts_unchanged": row_counts_unchanged,
        "idempotent": idempotent, "external_search_count": 0,
        "fine_projection_methodology_status": FINE_METHODOLOGY_STATUS,
        "production_write_performed": False,
    }
    (output_dir / "execution_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "component_population_closure_report.md").write_text(
        _closure_report(summary, domain_summary, _architecture_rows(manifest_1), _failure_rows(manifest_1)),
        encoding="utf-8",
    )
    return summary


def _closure_report(
    summary: dict[str, Any], domain_rows: list[dict[str, Any]],
    architecture_rows: list[dict[str, Any]], failure_rows: list[dict[str, Any]],
) -> str:
    population = summary["population"]
    lines = [
        "# Sprint 12 Component Population Scale & Materialization", "",
        f"## {summary['recommendation']}", "",
        "The full fleet was audited twice deterministically and materialized only to a real temporary copy. The canonical candidate was not written.", "",
        "## Canonical schema and projection", "",
        "- Tire: `tire_A_final`, `tire_B_final`, `tire_C_final` <- `ROLLING_MINOR` aggregate.",
        "- Aero: `aero_C_coef_Npkph2` <- `AERO.C`; the canonical schema has no Aero A/B fields.",
        "- Transmission: `trans_A/B/C` <- `DRIVETRAIN_AGGREGATE` only, with explicit aggregate provenance.",
        "- Brake and Parasitic remain NULL in macro mode.",
        "- No Axle/Hub VDE projection field exists; conditional evidence remains in resolution/audit data.", "",
        "## Full-fleet population", "",
        "| Population | Count |", "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in population.items())
    lines.extend(["", "Fine projection count is zero by design: Pass 1C.5 did not validate RollingMinor against the frozen real-vs-shuffle gate, and Transmission/Axle additive boundaries remain unknown.", "", "## Component-domain projection", "", "| Domain | Existing before | Newly populated | Preserved | Remaining NULL |", "|---|---:|---:|---:|---:|"])
    lines.extend(
        f"| {row['domain']} | {row['existing_before']} | {row['newly_populated']} | {row['preserved_existing']} | {row['remaining_null']} |"
        for row in domain_rows
    )
    lines.extend(["", "## Integrity and idempotency", "",
        f"- Source SHA256 unchanged: **{'YES' if summary['source_sha256_before'] == summary['source_sha256_after'] else 'NO'}**",
        f"- `PRAGMA quick_check`: **{summary['quick_check']}**",
        f"- Foreign-key issues: **{summary['foreign_key_issues']}**",
        f"- Protected VDE fields unchanged: **{'YES' if summary['protected_vde_fields_unchanged'] else 'NO'}**",
        f"- Core row counts unchanged: **{'YES' if summary['core_row_counts_unchanged'] else 'NO'}**",
        f"- Second execution produced zero changes: **{'YES' if summary['idempotent'] else 'NO'}**",
        f"- Deterministic manifest SHA256: `{summary['manifest_sha256']}`", "",
        "## Known review items", "",
        "- RollingMinor fine vector selection remains conditional and not fleet-validated versus shuffle.",
        "- Transmission and Axle reference boundaries are not proven additive.",
        "- EDrive aggregate rows are retained in `component_resolution` but are not projected into the historical Transmission slot.",
        "- Fine references remain supporting audit evidence and never supersede the macro physical solution.", "",
        "## Canonical write recommendation", "",
        f"`{summary['recommendation']}`", "",
        "Do not promote automatically. Review the deterministic manual audit and the EDrive schema-projection limitation first.",
    ])
    return "\n".join(lines) + "\n"


__all__ = [
    "METHOD_VERSION", "PROJECTION_FIELDS", "apply_projection_manifest",
    "execute_component_population_materialization", "protected_snapshot",
]
