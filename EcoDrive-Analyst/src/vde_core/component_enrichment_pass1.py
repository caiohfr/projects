"""Deterministic Sprint 12 component-enrichment Pass 1.

This module applies the frozen Component Estimation Physical Contract v1.0
and Estimator Contract v1.0 to canonical VDE data.  It never changes the
authoritative VDE road-load snapshot and never treats generic synthetic
references as vehicle-specific evidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import csv
import json
import math
from pathlib import Path
import sqlite3
import subprocess
from typing import Any, Iterable

import numpy as np

from src.vde_core.estimation.ecodrive_component_estimator_reference_v1_0 import (
    ABC,
    RHO_AIR,
    ROLLING_K,
    dht_aggregated_target_set_2wd,
    epa_to_si,
    fixed_gear_bound_flags,
    fixed_gear_target_set_2wd,
    moskalik_full_target_set,
    moskalik_target_only_structured,
    project_curve_to_abc,
    reconstruction_metrics,
)


ESTIMATOR_VERSION = "ECODRIVE_COMPONENT_ESTIMATOR_V1.0"
PASS_VERSION = "SPRINT12_COMPONENT_ENRICHMENT_PASS1_V1"
CORE_BOUNDARIES = ("AERO", "ROLLING_MINOR", "DRIVETRAIN_AGGREGATE")
ACCEPTED_STATUSES = {"SUPPORTED", "CONDITIONAL"}
TIER_A_STATUSES = {"SUPPORTED"}
TIER_B_STATUSES = {"CONDITIONAL"}
HARD_QA_MIN_KPH = 24.0
HARD_QA_MAX_KPH = 105.0
FORCE_TOLERANCE_N = -0.1
COVERAGE_FIELDS = [
    "boundary", "eligible_slots", "tier_a_resolved", "tier_b_resolved",
    "unresolved", "coverage_pct", "high_confidence_pct",
]
OUTCOME_FIELDS = [
    "vde_id", "vehicle_configuration_id", "boundary", "eligibility_reason",
    "outcome_status", "coverage_tier", "component_resolution_id", "method",
    "estimator_version", "provenance_class", "source_type", "reused_flag",
    "confidence", "fidelity_level", "fit_nrmse_pct", "condition_number",
    "sensitivity_rel_pct", "reason_codes", "assumptions", "source_run_ids",
    "notes", "architecture", "electrification",
]
UNRESOLVED_GROUP_FIELDS = [
    "architecture", "manufacturer", "model_family", "engine_model",
    "transmission_model", "transmission_type", "gear_count", "drive_system",
    "electrification", "group_id", "model_year_from", "model_year_to",
    "unresolved_boundaries", "reason_codes", "represented_vde_count",
    "represented_vehicle_configuration_count", "existing_evidence_summary",
    "recommended_future_research_target", "priority_score",
    "safe_search_query_descriptor",
]


@dataclass(frozen=True)
class RunEvidence:
    status: str
    set_abc: ABC | None
    run_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class Route:
    architecture: str
    model_type: str
    model_role: str
    electrification: str | None
    evidence_source: str
    reason_codes: tuple[str, ...]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _json_object(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _round_key(values: Iterable[Any]) -> tuple[float | None, ...]:
    return tuple(None if value is None else round(float(value), 12) for value in values)


def _complete_abc(row: dict[str, Any]) -> bool:
    return all(row.get(key) is not None and math.isfinite(float(row[key])) for key in ("coast_A_N", "coast_B_N_per_kph", "coast_C_N_per_kph2"))


def _normalize_drive(value: Any) -> str:
    return {
        "2-WHEEL DRIVE, FRONT": "FWD",
        "2-WHEEL DRIVE, REAR": "RWD",
        "ALL WHEEL DRIVE": "AWD",
        "4-WHEEL DRIVE": "4WD",
        "PART-TIME 4-WHEEL DRIVE": "4WD",
    }.get(str(value or "").strip().upper(), "UNMAPPED")


def _electrification_for(
    row: dict[str, Any],
    fuelcons_electrification: dict[int, set[str]],
) -> tuple[str | None, str]:
    values = {
        str(value).strip().upper()
        for value in fuelcons_electrification.get(int(row["vde_id"]), set())
        if str(value or "").strip()
    }
    if len(values) == 1:
        return next(iter(values)), "fuelcons.electrification"
    if len(values) > 1:
        return None, "conflicting_fuelcons.electrification"
    source_value = str(row.get("propulsion_architecture") or "").strip().upper()
    trusted = {
        "PURE ICE": "ICE",
        "EV": "BEV",
        "ELECTRICITY": "BEV",
        "NOVC-HEV": "HEV",
        "OVC-HEV": "PHEV",
        "HYDROGEN 5": "FCEV",
    }
    if source_value in trusted:
        return trusted[source_value], "vehicle_configuration.propulsion_architecture_exact"
    return None, "unresolved_electrification"


def route_architecture(
    row: dict[str, Any],
    fuelcons_electrification: dict[int, set[str]],
) -> Route:
    electrification, evidence_source = _electrification_for(row, fuelcons_electrification)
    transmission = str(row.get("transmission_type") or "").strip().upper()
    drive = _normalize_drive(row.get("drive_system"))
    try:
        gears = int(row["gear_count"]) if row.get("gear_count") is not None else None
    except (TypeError, ValueError):
        gears = None

    if electrification == "ICE" and transmission and transmission != "OTHER":
        return Route("CONVENTIONAL_MULTI_SPEED", "MOSKALIK_2020", "PRIMARY", electrification, evidence_source, ())
    if electrification in {"HEV", "PHEV"}:
        is_cvt = "CONTINUOUS" in transmission or transmission == "CVT"
        if gears is not None and gears > 1 and not is_cvt:
            return Route(
                "HYBRID_PARALLEL_CONVENTIONAL_TRANS",
                "MOSKALIK_2020",
                "CONDITIONAL_PRIMARY",
                electrification,
                evidence_source,
                ("HYBRID_CONVENTIONAL_TRANSMISSION_METADATA",),
            )
        return Route(
            "UNRESOLVED_HYBRID_TOPOLOGY",
            "NONE_V1",
            "NOT_IDENTIFIABLE",
            electrification,
            evidence_source,
            ("HYBRID_TOPOLOGY_NOT_RESOLVED_BY_CANONICAL_METADATA",),
        )
    if electrification in {"BEV", "FCEV"}:
        if gears == 1 and drive in {"FWD", "RWD"}:
            return Route("EV_FIXED_GEAR", "FIXED_GEAR_EDRIVE_V1", "PRIMARY", electrification, evidence_source, ("Q_ROLL_ASSUMED_0_50",))
        if gears == 1 and drive in {"AWD", "4WD"}:
            return Route(
                "EV_FIXED_GEAR_AWD_UNRESOLVED",
                "NONE_V1",
                "NOT_IDENTIFIABLE",
                electrification,
                evidence_source,
                ("AWD_FIXED_GEAR_BOUNDARY_UNRESOLVED_V1",),
            )
        if gears is not None and gears > 1:
            return Route("EV_MULTI_SPEED", "MOSKALIK_2020", "FALLBACK_EXPERIMENTAL", electrification, evidence_source, ("OUTSIDE_MOSKALIK_PRIMARY_DOMAIN",))
    return Route(
        "UNRESOLVED",
        "NONE_V1",
        "NOT_IDENTIFIABLE",
        electrification,
        evidence_source,
        ("ARCHITECTURE_NOT_RESOLVED_FROM_CANONICAL_FIELDS",),
    )


def _apply_research_route_override(route: Route, normalized_architecture: str | None) -> Route:
    """Apply only estimator routes already supported by this module.

    The override is intentionally narrow: research can resolve identity/topology,
    but it cannot add a new physical model. Unsupported researched architectures
    therefore retain the ordinary deterministic route and stop behavior.
    """
    value = str(normalized_architecture or "").strip().upper()
    if value == "PARALLEL_HYBRID_TMED_PMSM_6AT":
        return Route(
            "HYBRID_PARALLEL_CONVENTIONAL_TRANS",
            "MOSKALIK_2020",
            "CONDITIONAL_PRIMARY",
            route.electrification or "HEV",
            "phasea1_reconciled_external_evidence",
            ("RESEARCH_ARCHITECTURE_EXACT", "HYBRID_CONVENTIONAL_TRANSMISSION_METADATA"),
        )
    return route


def _run_evidence(rows: list[dict[str, Any]]) -> RunEvidence:
    grouped: dict[tuple[float, float, float], list[str]] = defaultdict(list)
    available_run_ids: list[str] = []
    for row in rows:
        run_id = str(row.get("run_id") or "").strip()
        if run_id:
            available_run_ids.append(run_id)
        if str(row.get("run_type") or "").upper() != "TEST":
            continue
        if str(row.get("record_status") or "ACTIVE").upper() != "ACTIVE":
            continue
        if "cold co" in str(row.get("procedure_description") or "").lower():
            continue
        conditions = _json_object(row.get("conditions_json"))
        native = conditions.get("set_abc_native")
        if not isinstance(native, dict):
            continue
        keys = ("Set Coef A (lbf)", "Set Coef B (lbf/mph)", "Set Coef C (lbf/mph**2)")
        if any(native.get(key) is None for key in keys):
            continue
        grouped[_round_key(native[key] for key in keys)].append(run_id)
    if len(grouped) == 1:
        native, run_ids = next(iter(grouped.items()))
        return RunEvidence("EXACT_UNIQUE_TARGET_SET", epa_to_si(*native), tuple(sorted(set(run_ids))), ())
    if len(grouped) > 1:
        return RunEvidence("AMBIGUOUS_SET_ABC", None, tuple(sorted(set(available_run_ids))), ("MULTIPLE_DISTINCT_NON_COLD_SET_ABC",))
    return RunEvidence("TARGET_ONLY", None, tuple(sorted(set(available_run_ids))), ("NO_UNIQUE_MATCHED_SET_ABC",))


def _component_triplets_moskalik(result: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        "AERO": {"A": 0.0, "B": 0.0, "C": float(result["pure_aero_C"])},
        "ROLLING_MINOR": {"A": float(result["R0_N"]), "B": float(result["R1_N_per_kph"]), "C": 0.0},
        "DRIVETRAIN_AGGREGATE": {"A": float(result["T0_N"]), "B": float(result["T1_N_per_kph"]), "C": float(result["T2_N_per_kph2"])},
    }


def _qa_components(target: ABC, components: dict[str, dict[str, float]]) -> dict[str, Any]:
    speed = np.linspace(HARD_QA_MIN_KPH, HARD_QA_MAX_KPH, 200)
    force_by_boundary = {
        boundary: ABC(values["A"], values["B"], values["C"]).force(speed)
        for boundary, values in components.items()
    }
    predicted = sum(force_by_boundary.values())
    metrics = reconstruction_metrics(target.force(speed), predicted)
    hard_flags: list[str] = []
    aero = components.get("AERO", {})
    if float(aero.get("C", 0.0)) <= 0.0:
        hard_flags.append("NON_POSITIVE_CDA_OR_AERO")
    if "ROLLING_MINOR" in force_by_boundary and float(np.min(force_by_boundary["ROLLING_MINOR"])) < FORCE_TOLERANCE_N:
        hard_flags.append("MATERIALLY_NEGATIVE_ROLLING_MINOR_FORCE")
    drivetrain_key = next((key for key in ("DRIVETRAIN_AGGREGATE", "EDRIVE_AGGREGATE", "DHT_AGGREGATE") if key in force_by_boundary), None)
    if drivetrain_key and float(np.min(force_by_boundary[drivetrain_key])) < FORCE_TOLERANCE_N:
        hard_flags.append("MATERIALLY_NEGATIVE_DRIVETRAIN_FORCE")
    if not all(math.isfinite(float(value)) for values in components.values() for value in values.values()):
        hard_flags.append("NON_FINITE_COMPONENT_PARAMETER")
    if float(metrics["NRMSE"]) > 0.05 or float(metrics["max_abs_error_norm"]) > 0.10:
        hard_flags.append("ROADLOAD_RECONSTRUCTION_FAILURE")
    return {
        "target_rmse": float(metrics["RMSE"]),
        "target_nrmse_pct": 100.0 * float(metrics["NRMSE"]),
        "target_max_abs_error_norm_pct": 100.0 * float(metrics["max_abs_error_norm"]),
        "hard_fail_flags": hard_flags,
        "minimum_force_by_boundary": {key: float(np.min(value)) for key, value in force_by_boundary.items()},
    }


def _relative_range(values: list[float], nominal: float) -> float | None:
    if not values or abs(float(nominal)) <= 1e-12:
        return None
    return 100.0 * (max(values) - min(values)) / abs(float(nominal))


def _estimate_moskalik(target: ABC, route: Route, evidence: RunEvidence) -> dict[str, Any]:
    reasons = [*route.reason_codes, *evidence.reason_codes]
    if evidence.set_abc is not None:
        raw = moskalik_full_target_set(target, evidence.set_abc)
        state = "E0_FULL_TARGET_SET_ANALYTIC"
        method = "MOSKALIK_2020_ANALYTIC"
        source_rank = 6
    else:
        raw = moskalik_target_only_structured(target)
        raw["pure_aero_C"] = RHO_AIR * float(raw["CdA_m2"]) / 25.92
        state = "E2_TARGET_ONLY_STRUCTURED"
        method = "BOUNDED_LINEAR_LS_MOSKALIK_TARGET_ONLY"
        source_rank = 3
        reasons.append("STRUCTURAL_FLEET_ASSUMPTIONS_ONLY")
    components = _component_triplets_moskalik(raw)
    qa = _qa_components(target, components)
    status = "REJECTED_MODEL" if qa["hard_fail_flags"] else "SUPPORTED"
    if state == "E2_TARGET_ONLY_STRUCTURED" or route.model_role != "PRIMARY":
        status = "REJECTED_MODEL" if qa["hard_fail_flags"] else "CONDITIONAL"
    reasons.extend(qa["hard_fail_flags"])
    return {
        "status": status,
        "coverage_tier": "A" if status == "SUPPORTED" else ("B" if status == "CONDITIONAL" else "UNRESOLVED"),
        "state": state,
        "method": method,
        "model_type": route.model_type,
        "model_role": route.model_role,
        "components": components,
        "fit_nrmse_pct": qa["target_nrmse_pct"],
        "condition_number": raw.get("scaled_condition_number"),
        "sensitivity_by_boundary": {},
        "reason_codes": sorted(set(reasons)),
        "assumptions": {"rolling_k_per_kph": ROLLING_K, "rho_air_kg_per_m3": RHO_AIR},
        "diagnostics": {**qa, "source_information_rank": source_rank, "CdA_m2": raw.get("CdA_m2")},
    }


def _fixed_gear_components(result: dict[str, Any]) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    speed = np.linspace(0.0, 130.0, 261)
    edrive_force = float(result["K0"]) + float(result["Kb"]) * speed ** (2.0 / 3.0) + float(result["Kc"]) * speed**2
    projection = project_curve_to_abc(speed, edrive_force)
    return (
        {
            "AERO": {"A": 0.0, "B": 0.0, "C": RHO_AIR * float(result["CdA_m2"]) / 25.92},
            "ROLLING_MINOR": {"A": float(result["R0_N"]), "B": float(result["R1_N_per_kph"]), "C": 0.0},
            "EDRIVE_AGGREGATE": {"A": float(projection["A"]), "B": float(projection["B"]), "C": float(projection["C"])},
        },
        projection,
    )


def _estimate_fixed_gear(target: ABC, route: Route, evidence: RunEvidence) -> dict[str, Any]:
    if evidence.set_abc is None:
        return {
            "status": "NOT_IDENTIFIABLE",
            "coverage_tier": "UNRESOLVED",
            "state": "E5_NOT_IDENTIFIABLE",
            "method": "FIXED_GEAR_TARGET_ONLY_GATE",
            "model_type": route.model_type,
            "model_role": route.model_role,
            "components": {},
            "fit_nrmse_pct": None,
            "condition_number": None,
            "sensitivity_by_boundary": {},
            "reason_codes": sorted(set((*route.reason_codes, *evidence.reason_codes, "FIXED_GEAR_TARGET_ONLY_REQUIRES_INDEPENDENT_CDA_AND_ROLLING"))),
            "assumptions": {},
            "diagnostics": {},
        }
    sweeps = {
        q: fixed_gear_target_set_2wd(target, evidence.set_abc, q_roll=q)
        for q in (0.4, 0.5, 0.6)
    }
    nominal = sweeps[0.5]
    components, projection = _fixed_gear_components(nominal)
    qa = _qa_components(target, components)
    active_flags = fixed_gear_bound_flags(nominal)
    hard_flags = list(qa["hard_fail_flags"])
    if "ROLLING_COLLAPSED_TO_ZERO" in active_flags:
        hard_flags.append("BOUND_ACTIVE_PHYSICAL_DEGENERACY_ROLLING")
    if "DRIVETRAIN_COLLAPSED_TO_ZERO" in active_flags:
        hard_flags.append("BOUND_ACTIVE_PHYSICAL_DEGENERACY_DRIVETRAIN")
    cda_values = [float(result["CdA_m2"]) for result in sweeps.values()]
    rrc80_values = [float(result["R0_N"] + 80.0 * result["R1_N_per_kph"]) for result in sweeps.values()]
    edrive80_values = [float(result["K0"] + result["Kb"] * 80.0 ** (2.0 / 3.0) + result["Kc"] * 80.0**2) for result in sweeps.values()]
    sensitivity = {
        "AERO": _relative_range(cda_values, float(nominal["CdA_m2"])),
        "ROLLING_MINOR": _relative_range(rrc80_values, float(nominal["R0_N"] + 80.0 * nominal["R1_N_per_kph"])),
        "EDRIVE_AGGREGATE": _relative_range(edrive80_values, float(nominal["K0"] + nominal["Kb"] * 80.0 ** (2.0 / 3.0) + nominal["Kc"] * 80.0**2)),
    }
    reasons = sorted(set((*route.reason_codes, *evidence.reason_codes, *active_flags, *hard_flags)))
    status = "REJECTED_MODEL" if hard_flags else "CONDITIONAL"
    return {
        "status": status,
        "coverage_tier": "B" if status == "CONDITIONAL" else "UNRESOLVED",
        "state": "E1_TARGET_SET_CONSTRAINED",
        "method": "FIXED_GEAR_EDRIVE_V1_QROLL_050",
        "model_type": route.model_type,
        "model_role": route.model_role,
        "components": components,
        "fit_nrmse_pct": qa["target_nrmse_pct"],
        "condition_number": nominal.get("scaled_condition_number"),
        "sensitivity_by_boundary": sensitivity,
        "reason_codes": reasons,
        "assumptions": {"q_roll_nominal": 0.5, "q_roll_sweep": [0.4, 0.5, 0.6], "rolling_k_per_kph": ROLLING_K, "rho_air_kg_per_m3": RHO_AIR},
        "diagnostics": {**qa, "active_bound_flags": active_flags, "native_parameters": {key: nominal[key] for key in ("K0", "Kb", "Kc")}, "projection": projection},
    }


def _not_identifiable(route: Route, evidence: RunEvidence) -> dict[str, Any]:
    return {
        "status": "NOT_IDENTIFIABLE",
        "coverage_tier": "UNRESOLVED",
        "state": "E5_NOT_IDENTIFIABLE",
        "method": "ARCHITECTURE_IDENTIFIABILITY_GATE_V1",
        "model_type": route.model_type,
        "model_role": route.model_role,
        "components": {},
        "fit_nrmse_pct": None,
        "condition_number": None,
        "sensitivity_by_boundary": {},
        "reason_codes": sorted(set((*route.reason_codes, *evidence.reason_codes))),
        "assumptions": {},
        "diagnostics": {},
    }


def estimate_vde(target: ABC, route: Route, evidence: RunEvidence) -> dict[str, Any]:
    if route.architecture in {"CONVENTIONAL_MULTI_SPEED", "HYBRID_PARALLEL_CONVENTIONAL_TRANS", "EV_MULTI_SPEED"}:
        return _estimate_moskalik(target, route, evidence)
    if route.architecture == "EV_FIXED_GEAR":
        return _estimate_fixed_gear(target, route, evidence)
    return _not_identifiable(route, evidence)


def _table_count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def _db_objects(conn: sqlite3.Connection) -> tuple[tuple[str, str], ...]:
    return tuple(conn.execute("SELECT type,name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name").fetchall())


def _vde_total_signature(conn: sqlite3.Connection) -> str:
    digest = sha256()
    for row in conn.execute("SELECT id,coast_A_N,coast_B_N_per_kph,coast_C_N_per_kph2 FROM vde ORDER BY id"):
        digest.update(_json_dump(list(row)).encode("utf-8"))
    return digest.hexdigest().upper()


def inventory(conn: sqlite3.Connection) -> dict[str, Any]:
    tables = ("vde", "vehicle_configuration", "component_db", "component_instance", "component_resolution", "vde_component_resolution")
    return {
        "quick_check": conn.execute("PRAGMA quick_check").fetchone()[0],
        "foreign_key_issues": len(conn.execute("PRAGMA foreign_key_check").fetchall()),
        "row_counts": {table: _table_count(conn, table) for table in tables},
        "objects": [list(item) for item in _db_objects(conn)],
        "vde_total_signature": _vde_total_signature(conn),
    }


def _existing_slot_map(conn: sqlite3.Connection) -> dict[tuple[int, str], dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT link.vde_id,link.boundary,link.component_resolution_id,
               resolution.method,resolution.estimate_status,resolution.confidence,
               resolution.fidelity_level,resolution.vehicle_configuration_id,
               resolution.provenance_json,resolution.record_status
        FROM vde_component_resolution AS link
        JOIN component_resolution AS resolution
          ON resolution.component_resolution_id=link.component_resolution_id
        WHERE resolution.record_status='ACTIVE'
        ORDER BY link.vde_id,link.boundary,link.ordinal
        """
    ).fetchall()
    result: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        payload = dict(row)
        provenance = _json_object(payload.get("provenance_json"))
        generic_synthetic = bool(provenance.get("synthetic_reference")) and not payload.get("vehicle_configuration_id")
        status = str(payload.get("estimate_status") or "").upper()
        method = str(payload.get("method") or "").upper()
        if generic_synthetic:
            tier = "UNRESOLVED"
        elif status in TIER_A_STATUSES or method in {"SOURCE", "MEASURED", "OBSERVED", "EXACT_CANONICAL_REUSE"}:
            tier = "A"
        elif status in TIER_B_STATUSES or method in {"RULE_ESTIMATED", "APPROVED_CARRYOVER", "MODEL_PROJECTION"}:
            tier = "B"
        else:
            tier = "UNRESOLVED"
        payload["coverage_tier"] = tier
        result[(int(payload["vde_id"]), str(payload["boundary"]))] = payload
    return result


def _load_population(conn: sqlite3.Connection, vde_ids: set[int] | None, limit: int | None) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]], dict[int, set[str]]]:
    sql = """
        SELECT v.id AS vde_id,v.vehicle_configuration_id,v.coast_A_N,
               v.coast_B_N_per_kph,v.coast_C_N_per_kph2,v.make,v.model,v.year,
               v.category,v.legislation,v.source_name,v.source_record_id,
               vc.program_id,vc.drive_system,vc.transmission_type,
               vc.transmission_model,vc.gear_count,vc.final_drive_ratio,
               vc.engine_model,vc.engine_type,vc.propulsion_architecture,
               vc.architecture_properties_json,vc.source_identity_json
        FROM vde AS v
        JOIN vehicle_configuration AS vc
          ON vc.vehicle_configuration_id=v.vehicle_configuration_id
        WHERE v.record_status='ACTIVE'
        ORDER BY v.id
    """
    population = [dict(row) for row in conn.execute(sql)]
    if vde_ids is not None:
        population = [row for row in population if int(row["vde_id"]) in vde_ids]
    if limit is not None:
        population = population[: int(limit)]
    selected = {int(row["vde_id"]) for row in population}
    run_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in conn.execute("SELECT run_id,vde_id,run_type,procedure_description,conditions_json,record_status FROM run ORDER BY vde_id,run_id"):
        if int(row["vde_id"]) in selected:
            run_rows[int(row["vde_id"])].append(dict(row))
    fuelcons: dict[int, set[str]] = defaultdict(set)
    for vde_id, electrification in conn.execute("SELECT vde_id,electrification FROM fuelcons WHERE record_status='ACTIVE' AND vde_id IS NOT NULL"):
        if int(vde_id) in selected and electrification:
            fuelcons[int(vde_id)].add(str(electrification))
    return population, run_rows, fuelcons


def _resolution_id(signature: dict[str, Any], boundary: str) -> str:
    digest = sha256(_json_dump({"signature": signature, "boundary": boundary}).encode("utf-8")).hexdigest().upper()[:20]
    prefix = {"AERO": "AERO", "ROLLING_MINOR": "ROLL", "DRIVETRAIN_AGGREGATE": "DRIVE", "EDRIVE_AGGREGATE": "EDRIVE", "SYSTEM_DECOMPOSITION": "SYSTEM"}[boundary]
    return f"CR-P1-{prefix}-{digest}"


def _coverage_rows(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, Counter] = defaultdict(Counter)
    for row in outcomes:
        grouped[str(row["boundary"])][str(row["coverage_tier"])] += 1
    rows = []
    for boundary in sorted(grouped):
        counts = grouped[boundary]
        eligible = sum(counts.values())
        tier_a = counts["A"]
        tier_b = counts["B"]
        unresolved = eligible - tier_a - tier_b
        rows.append({
            "boundary": boundary,
            "eligible_slots": eligible,
            "tier_a_resolved": tier_a,
            "tier_b_resolved": tier_b,
            "unresolved": unresolved,
            "coverage_pct": round(100.0 * (tier_a + tier_b) / eligible, 6) if eligible else 0.0,
            "high_confidence_pct": round(100.0 * tier_a / eligible, 6) if eligible else 0.0,
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    if not rows:
        return ["_No rows._"]
    rendered = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        values = [str(row.get(field, "")).replace("|", "\\|").replace("\n", " ") for field in fields]
        rendered.append("| " + " | ".join(values) + " |")
    return rendered


def _technical_groups(outcomes: list[dict[str, Any]], population_by_id: dict[int, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unresolved_by_vde: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        if row["coverage_tier"] == "UNRESOLVED":
            unresolved_by_vde[int(row["vde_id"])].append(row)
    grouped: dict[str, dict[str, Any]] = {}
    members: list[dict[str, Any]] = []
    for vde_id, slots in sorted(unresolved_by_vde.items()):
        source = population_by_id[vde_id]
        boundaries = sorted(str(slot["boundary"]) for slot in slots)
        reasons = sorted({reason for slot in slots for reason in str(slot.get("reason_codes") or "").split(";") if reason})
        identity = {
            "architecture": slots[0]["architecture"],
            "manufacturer": source.get("make"),
            "model_family": source.get("model"),
            "engine_model": source.get("engine_model"),
            "transmission_model": source.get("transmission_model"),
            "transmission_type": source.get("transmission_type"),
            "gear_count": source.get("gear_count"),
            "drive_system": source.get("drive_system"),
            "electrification": slots[0].get("electrification"),
            "unresolved_boundaries": boundaries,
            "reason_codes": reasons,
        }
        digest = sha256(_json_dump(identity).encode("utf-8")).hexdigest().upper()[:16]
        group_id = f"P1-UNRES-{digest}"
        group = grouped.setdefault(group_id, {**identity, "group_id": group_id, "years": [], "vde_ids": [], "vehicle_configuration_ids": set()})
        group["years"].append(source.get("year"))
        group["vde_ids"].append(vde_id)
        group["vehicle_configuration_ids"].add(source.get("vehicle_configuration_id"))
        members.append({"group_id": group_id, "vde_id": vde_id, "vehicle_configuration_id": source.get("vehicle_configuration_id")})
    rows = []
    for group in grouped.values():
        years = [int(value) for value in group.pop("years") if value is not None]
        vde_ids = group.pop("vde_ids")
        configuration_ids = group.pop("vehicle_configuration_ids")
        boundary_count = len(group["unresolved_boundaries"])
        represented_vdes = len(vde_ids)
        row = {
            **group,
            "model_year_from": min(years) if years else None,
            "model_year_to": max(years) if years else None,
            "unresolved_boundaries": ";".join(group["unresolved_boundaries"]),
            "reason_codes": ";".join(group["reason_codes"]),
            "represented_vde_count": represented_vdes,
            "represented_vehicle_configuration_count": len(configuration_ids),
            "existing_evidence_summary": "Canonical vehicle/configuration metadata and road-load evidence only; no external evidence used.",
            "recommended_future_research_target": "Resolve exact drivetrain topology and/or independent component evidence for listed boundaries.",
            "priority_score": represented_vdes * boundary_count,
            "safe_search_query_descriptor": " | ".join(str(value) for value in (group.get("manufacturer"), group.get("model_family"), group.get("engine_model"), group.get("transmission_model")) if value),
        }
        rows.append(row)
    rows.sort(key=lambda row: (-int(row["priority_score"]), str(row["group_id"])))
    return rows, members


def _git_metadata() -> dict[str, str | None]:
    result: dict[str, str | None] = {"commit": None, "branch": None}
    try:
        result["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        result["branch"] = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return result


def execute_pass1(
    db_path: Path,
    output_dir: Path,
    *,
    write: bool = False,
    source_db_path: Path | None = None,
    vde_ids: set[int] | None = None,
    research_route_overrides: dict[int, str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    db_path = Path(db_path).resolve(strict=True)
    source_db_path = Path(source_db_path).resolve(strict=True) if source_db_path else db_path
    output_dir = Path(output_dir)
    source_hash_before = file_sha256(source_db_path)
    db_hash_before = file_sha256(db_path)
    uri = f"file:{db_path.as_posix()}?mode={'rw' if write else 'ro'}"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    if not write:
        conn.execute("PRAGMA query_only=ON")
    before_inventory = inventory(conn)
    if before_inventory["quick_check"] != "ok" or before_inventory["foreign_key_issues"]:
        conn.close()
        raise ValueError(f"Pre-run DB integrity failed: {before_inventory}")
    population, run_rows, fuelcons = _load_population(conn, vde_ids, limit)
    existing_slots = _existing_slot_map(conn)
    population_by_id = {int(row["vde_id"]): row for row in population}

    before_outcomes: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    resolutions: dict[str, dict[str, Any]] = {}
    links: dict[tuple[int, str, str], dict[str, Any]] = {}
    solve_cache: dict[str, dict[str, Any]] = {}

    for row in population:
        vde_id = int(row["vde_id"])
        route = route_architecture(row, fuelcons)
        if research_route_overrides:
            route = _apply_research_route_override(route, research_route_overrides.get(vde_id))
        evidence = _run_evidence(run_rows.get(vde_id, []))
        source_boundaries = list(CORE_BOUNDARIES)
        if route.architecture == "EV_FIXED_GEAR":
            source_boundaries[-1] = "EDRIVE_AGGREGATE"
        for boundary in source_boundaries:
            existing = existing_slots.get((vde_id, boundary))
            before_tier = str(existing.get("coverage_tier")) if existing else "UNRESOLVED"
            before_outcomes.append({"vde_id": vde_id, "boundary": boundary, "coverage_tier": before_tier})
        if not _complete_abc(row):
            attempt = {
                "status": "STOP_ROADLOAD_STATE",
                "coverage_tier": "UNRESOLVED",
                "state": "E6_STOP",
                "method": "ROADLOAD_COMPLETENESS_GATE",
                "model_type": route.model_type,
                "model_role": route.model_role,
                "components": {},
                "fit_nrmse_pct": None,
                "condition_number": None,
                "sensitivity_by_boundary": {},
                "reason_codes": ["INCOMPLETE_AUTHORITATIVE_TARGET_ABC"],
                "assumptions": {},
                "diagnostics": {},
            }
        else:
            target = ABC(float(row["coast_A_N"]), float(row["coast_B_N_per_kph"]), float(row["coast_C_N_per_kph2"]))
            cache_key = _json_dump({"target": _round_key((target.A, target.B, target.C)), "set": _round_key((evidence.set_abc.A, evidence.set_abc.B, evidence.set_abc.C)) if evidence.set_abc else None, "route": route.__dict__})
            if cache_key not in solve_cache:
                solve_cache[cache_key] = estimate_vde(target, route, evidence)
            attempt = solve_cache[cache_key]
        attempts.append({
            "vde_id": vde_id,
            "architecture": route.architecture,
            "estimator_state": attempt["state"],
            "model_type": attempt["model_type"],
            "final_status": attempt["status"],
            "reason_codes": ";".join(attempt["reason_codes"]),
        })

        accepted = attempt["status"] in ACCEPTED_STATUSES
        component_boundaries = list(attempt["components"])
        if not component_boundaries:
            component_boundaries = source_boundaries
        base_signature = {
            "pass_version": PASS_VERSION,
            "estimator_version": ESTIMATOR_VERSION,
            "vehicle_configuration_id": row["vehicle_configuration_id"],
            "architecture": route.architecture,
            "state": attempt["state"],
            "model_type": attempt["model_type"],
            "model_role": attempt["model_role"],
            "target_abc": _round_key((row.get("coast_A_N"), row.get("coast_B_N_per_kph"), row.get("coast_C_N_per_kph2"))),
            "set_abc": _round_key((evidence.set_abc.A, evidence.set_abc.B, evidence.set_abc.C)) if evidence.set_abc else None,
            "status": attempt["status"],
        }
        boundary_resolution_ids: dict[str, str | None] = {}
        for boundary in component_boundaries:
            existing = existing_slots.get((vde_id, boundary))
            if existing and existing.get("coverage_tier") in {"A", "B"}:
                boundary_resolution_ids[boundary] = str(existing["component_resolution_id"])
                continue
            if not accepted:
                boundary_resolution_ids[boundary] = None
                continue
            resolution_id = _resolution_id(base_signature, boundary)
            boundary_resolution_ids[boundary] = resolution_id
            values = attempt["components"][boundary]
            resolution = resolutions.setdefault(
                resolution_id,
                {
                    "component_resolution_id": resolution_id,
                    "boundary": boundary,
                    "method": attempt["method"],
                    "confidence": "HIGH" if attempt["status"] == "SUPPORTED" else "MEDIUM",
                    "fidelity_level": "L2" if attempt["status"] == "SUPPORTED" else "L1",
                    "resolved_A_N": values["A"],
                    "resolved_B_N_per_kph": values["B"],
                    "resolved_C_N_per_kph2": values["C"],
                    "conditions_json": _json_dump(attempt["assumptions"]),
                    "input_component_instance_ids_json": "[]",
                    "source_run_ids": set(),
                    "source_vde_ids": set(),
                    "vehicle_configuration_id": row["vehicle_configuration_id"],
                    "estimate_status": attempt["status"],
                    "estimator_version": ESTIMATOR_VERSION,
                    "fit_nrmse_pct": attempt["fit_nrmse_pct"],
                    "condition_number": attempt["condition_number"],
                    "sensitivity_rel_pct": attempt["sensitivity_by_boundary"].get(boundary),
                    "record_status": "ACTIVE",
                    "review_status": "CURRENT",
                    "provenance_base": {
                        "pass_version": PASS_VERSION,
                        "provenance_class": "MODEL_ESTIMATED",
                        "configuration_match_status": "EXACT_CANONICAL_VDE_RUN_LINK" if evidence.set_abc else "EXACT_CANONICAL_VDE_TARGET_ONLY",
                        "architecture_class": route.architecture,
                        "architecture_evidence_source": route.evidence_source,
                        "model_type": attempt["model_type"],
                        "model_role": attempt["model_role"],
                        "estimator_state": attempt["state"],
                        "reason_codes": attempt["reason_codes"],
                        "diagnostics": attempt["diagnostics"],
                        "authoritative_vde_total_unchanged": True,
                    },
                },
            )
            resolution["source_run_ids"].update(evidence.run_ids)
            resolution["source_vde_ids"].add(vde_id)
            links[(vde_id, resolution_id, boundary)] = {
                "vde_id": vde_id,
                "component_resolution_id": resolution_id,
                "boundary": boundary,
                "adoption_role": "ADOPTED",
                "ordinal": source_boundaries.index(boundary) if boundary in source_boundaries else len(source_boundaries),
            }

        for boundary in source_boundaries:
            existing = existing_slots.get((vde_id, boundary))
            if existing and existing.get("coverage_tier") in {"A", "B"}:
                tier = existing["coverage_tier"]
                outcome_status = "EXISTING_COMPATIBLE_RESOLUTION"
                resolution_id = existing["component_resolution_id"]
                method = existing.get("method")
                reused = True
                provenance_class = "EXISTING_CANONICAL"
            elif accepted and boundary in boundary_resolution_ids:
                tier = attempt["coverage_tier"]
                outcome_status = attempt["status"]
                resolution_id = boundary_resolution_ids[boundary]
                method = attempt["method"]
                reused = len(resolutions.get(str(resolution_id), {}).get("source_vde_ids", set())) > 1
                provenance_class = "MODEL_ESTIMATED"
            else:
                tier = "UNRESOLVED"
                outcome_status = attempt["status"]
                resolution_id = None
                method = attempt["method"]
                reused = False
                provenance_class = "UNRESOLVED"
            outcomes.append({
                "vde_id": vde_id,
                "vehicle_configuration_id": row["vehicle_configuration_id"],
                "boundary": boundary,
                "eligibility_reason": "CORE_PHYSICAL_CONTRACT_V1",
                "outcome_status": outcome_status,
                "coverage_tier": tier,
                "component_resolution_id": resolution_id,
                "method": method,
                "estimator_version": ESTIMATOR_VERSION,
                "provenance_class": provenance_class,
                "source_type": evidence.status,
                "reused_flag": reused,
                "confidence": "HIGH" if tier == "A" else ("MEDIUM" if tier == "B" else None),
                "fidelity_level": "L2" if tier == "A" else ("L1" if tier == "B" else None),
                "fit_nrmse_pct": attempt["fit_nrmse_pct"],
                "condition_number": attempt["condition_number"],
                "sensitivity_rel_pct": attempt["sensitivity_by_boundary"].get(boundary),
                "reason_codes": ";".join(attempt["reason_codes"]),
                "assumptions": _json_dump(attempt["assumptions"]),
                "source_run_ids": _json_dump(list(evidence.run_ids)),
                "notes": "Authoritative VDE TOTAL ABC remains unchanged.",
                "architecture": route.architecture,
                "electrification": route.electrification,
            })

    planned_resolution_rows = []
    for resolution in resolutions.values():
        payload = dict(resolution)
        provenance = dict(payload.pop("provenance_base"))
        provenance["source_vde_ids"] = sorted(payload.pop("source_vde_ids"))
        source_run_ids = sorted(payload.pop("source_run_ids"))
        payload["source_run_ids_json"] = _json_dump(source_run_ids)
        payload["provenance_json"] = _json_dump(provenance)
        planned_resolution_rows.append(payload)
    planned_resolution_rows.sort(key=lambda row: row["component_resolution_id"])
    planned_links = sorted(links.values(), key=lambda row: (row["vde_id"], row["boundary"], row["component_resolution_id"]))

    inserted_resolutions = inserted_links = 0
    if write:
        try:
            for payload in planned_resolution_rows:
                existing = conn.execute("SELECT 1 FROM component_resolution WHERE component_resolution_id=?", (payload["component_resolution_id"],)).fetchone()
                if existing:
                    continue
                columns = list(payload)
                conn.execute(
                    f"INSERT INTO component_resolution ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                    [payload[column] for column in columns],
                )
                inserted_resolutions += 1
            for payload in planned_links:
                existing = conn.execute(
                    "SELECT 1 FROM vde_component_resolution WHERE vde_id=? AND component_resolution_id=? AND boundary=?",
                    (payload["vde_id"], payload["component_resolution_id"], payload["boundary"]),
                ).fetchone()
                if existing:
                    continue
                conn.execute(
                    "INSERT INTO vde_component_resolution (vde_id,component_resolution_id,boundary,adoption_role,ordinal) VALUES (?,?,?,?,?)",
                    (payload["vde_id"], payload["component_resolution_id"], payload["boundary"], payload["adoption_role"], payload["ordinal"]),
                )
                inserted_links += 1
            pending = inventory(conn)
            if pending["quick_check"] != "ok" or pending["foreign_key_issues"]:
                raise ValueError(f"Pending-write integrity failed: {pending}")
            if pending["vde_total_signature"] != before_inventory["vde_total_signature"]:
                raise ValueError("Unauthorized historical VDE TOTAL ABC change detected")
            conn.commit()
        except Exception:
            conn.rollback()
            conn.close()
            raise

    after_inventory = inventory(conn)
    conn.close()
    db_hash_after = file_sha256(db_path)
    source_hash_after = file_sha256(source_db_path)
    if source_db_path != db_path and source_hash_after != source_hash_before:
        raise RuntimeError("Source candidate DB changed during temp-copy execution")
    if not write and db_hash_after != db_hash_before:
        raise RuntimeError("Dry-run modified the database")
    if after_inventory["objects"] != before_inventory["objects"]:
        raise RuntimeError("Unexpected schema object change")
    if after_inventory["vde_total_signature"] != before_inventory["vde_total_signature"]:
        raise RuntimeError("Historical VDE TOTAL ABC changed")
    for table in ("vde", "vehicle_configuration", "component_db", "component_instance"):
        if after_inventory["row_counts"][table] != before_inventory["row_counts"][table]:
            raise RuntimeError(f"Unauthorized row-count change in {table}")

    before_coverage = _coverage_rows(before_outcomes)
    after_coverage = _coverage_rows(outcomes)
    _write_csv(output_dir / "coverage_before.csv", before_coverage, COVERAGE_FIELDS)
    _write_csv(output_dir / "coverage_after.csv", after_coverage, COVERAGE_FIELDS)
    _write_csv(output_dir / "vde_enrichment_outcomes.csv", outcomes, OUTCOME_FIELDS)

    estimator_grouped: dict[tuple[str, str, str, str], int] = Counter(
        (row["architecture"], row["estimator_state"], row["model_type"], row["final_status"])
        for row in attempts
    )
    estimator_summary = [
        {"architecture": key[0], "estimator_state": key[1], "model_type": key[2], "final_status": key[3], "row_count": count}
        for key, count in sorted(estimator_grouped.items())
    ]
    _write_csv(
        output_dir / "estimator_summary.csv",
        estimator_summary,
        ["architecture", "estimator_state", "model_type", "final_status", "row_count"],
    )
    reason_counts = Counter(reason for row in attempts for reason in str(row["reason_codes"] or "").split(";") if reason)
    reason_summary = [{"reason_code": reason, "row_count": count} for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))]
    _write_csv(output_dir / "reason_code_summary.csv", reason_summary, ["reason_code", "row_count"])
    groups, members = _technical_groups(outcomes, population_by_id)
    _write_csv(output_dir / "unresolved_unique_groups.csv", groups, UNRESOLVED_GROUP_FIELDS)
    _write_csv(
        output_dir / "unresolved_group_members.csv",
        members,
        ["group_id", "vde_id", "vehicle_configuration_id"],
    )

    architecture_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in outcomes:
        architecture_counts[(str(row["architecture"]), str(row["boundary"]))][str(row["coverage_tier"])] += 1
    architecture_rows = []
    for (architecture, boundary), counts in sorted(architecture_counts.items()):
        eligible = sum(counts.values())
        architecture_rows.append({
            "architecture": architecture,
            "boundary": boundary,
            "eligible_slots": eligible,
            "tier_a_resolved": counts["A"],
            "tier_b_resolved": counts["B"],
            "unresolved": counts["UNRESOLVED"],
            "coverage_pct": round(100.0 * (counts["A"] + counts["B"]) / eligible, 6) if eligible else 0.0,
        })
    _write_csv(
        output_dir / "coverage_by_architecture.csv",
        architecture_rows,
        ["architecture", "boundary", "eligible_slots", "tier_a_resolved", "tier_b_resolved", "unresolved", "coverage_pct"],
    )
    provenance_counts = Counter((str(row["provenance_class"]), str(row["coverage_tier"])) for row in outcomes)
    provenance_rows = [{"provenance_class": key[0], "coverage_tier": key[1], "slot_count": count} for key, count in sorted(provenance_counts.items())]
    _write_csv(output_dir / "provenance_distribution.csv", provenance_rows, ["provenance_class", "coverage_tier", "slot_count"])

    resolved_by_vde: dict[int, Counter] = defaultdict(Counter)
    for row in outcomes:
        resolved_by_vde[int(row["vde_id"])][str(row["coverage_tier"])] += 1
    vde_with_1 = sum(counter["A"] + counter["B"] >= 1 for counter in resolved_by_vde.values())
    vde_with_2 = sum(counter["A"] + counter["B"] >= 2 for counter in resolved_by_vde.values())
    all_core = sum(counter["A"] + counter["B"] == 3 for counter in resolved_by_vde.values())
    total_before = {"eligible": sum(row["eligible_slots"] for row in before_coverage), "A": sum(row["tier_a_resolved"] for row in before_coverage), "B": sum(row["tier_b_resolved"] for row in before_coverage)}
    total_after = {"eligible": sum(row["eligible_slots"] for row in after_coverage), "A": sum(row["tier_a_resolved"] for row in after_coverage), "B": sum(row["tier_b_resolved"] for row in after_coverage)}
    before_pct = 100.0 * (total_before["A"] + total_before["B"]) / total_before["eligible"] if total_before["eligible"] else 0.0
    after_pct = 100.0 * (total_after["A"] + total_after["B"]) / total_after["eligible"] if total_after["eligible"] else 0.0

    git = _git_metadata()
    summary_lines = [
        "# EcoDrive Sprint 12 — Component Enrichment Pass 1", "",
        f"- Repository commit: `{git['commit']}`",
        f"- Branch: `{git['branch']}`",
        f"- Database executed: `{db_path}`",
        f"- Source candidate: `{source_db_path}`",
        f"- Mode: `{'write' if write else 'dry-run'}`",
        f"- Canonical estimator: `{ESTIMATOR_VERSION}`",
        f"- External searches: **0**", "",
        "## Coverage definition", "",
        "Primary unit: `(vde_id, eligible canonical core boundary)`. Every active VDE has AERO, ROLLING_MINOR and an architecture-appropriate aggregate drivetrain slot. Fine Tire/Brake/Axle/Hub slots are not created without independent evidence.", "",
        "## BEFORE", "",
        f"- Canonical VDE count considered: **{len(population):,}**",
        f"- Eligible component slots: **{total_before['eligible']:,}**",
        f"- Tier A: **{total_before['A']:,}**",
        f"- Tier B: **{total_before['B']:,}**",
        f"- Resolved usable coverage: **{before_pct:.3f}%**", "",
        "## AFTER", "",
        f"- Tier A: **{total_after['A']:,}**",
        f"- Tier B: **{total_after['B']:,}**",
        f"- Resolved usable slots: **{total_after['A'] + total_after['B']:,}**",
        f"- Resolved usable coverage: **{after_pct:.3f}%**",
        f"- High-confidence coverage: **{100.0 * total_after['A'] / total_after['eligible'] if total_after['eligible'] else 0.0:.3f}%**",
        f"- Coverage-point gain: **{after_pct - before_pct:.3f}**", "",
        "## VDE-level completeness", "",
        f"- VDEs with >=1 resolved slot: **{vde_with_1:,}**",
        f"- VDEs with >=2 resolved slots: **{vde_with_2:,}**",
        f"- VDEs with all core estimator boundaries resolved: **{all_core:,}**",
        f"- VDEs with all eligible slots resolved: **{all_core:,}**", "",
        "## Backlog", "",
        f"- Unresolved slots: **{total_after['eligible'] - total_after['A'] - total_after['B']:,}**",
        f"- Unresolved unique technical groups: **{len(groups):,}**",
        f"- Projected Pass 2 queries (one per group): **{len(groups):,}**",
        "- No Pass 2 search was executed.", "",
        "## Persistence", "",
        f"- Planned reusable component_resolution rows: **{len(planned_resolution_rows):,}**",
        f"- Planned VDE adoption links: **{len(planned_links):,}**",
        f"- Inserted component_resolution rows this execution: **{inserted_resolutions:,}**",
        f"- Inserted adoption links this execution: **{inserted_links:,}**",
        "- Historical VDE TOTAL ABC: **UNCHANGED**", "",
        "## Coverage by boundary", "",
        *_markdown_table(after_coverage, COVERAGE_FIELDS), "",
        "## Coverage by architecture", "",
        *_markdown_table(
            architecture_rows,
            ["architecture", "boundary", "eligible_slots", "tier_a_resolved", "tier_b_resolved", "unresolved", "coverage_pct"],
        ), "",
        "## Estimator state/status", "",
        *_markdown_table(estimator_summary, ["architecture", "estimator_state", "model_type", "final_status", "row_count"]), "",
        "## Provenance distribution", "",
        *_markdown_table(provenance_rows, ["provenance_class", "coverage_tier", "slot_count"]), "",
        "## Leading reason codes", "",
        *_markdown_table(reason_summary[:20], ["reason_code", "row_count"]), "",
        "The companion CSV files contain the complete machine-readable distributions and row-level audit trail.",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pass1_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    integrity = {
        "source_candidate_path": str(source_db_path),
        "original_db_sha256": source_hash_before,
        "source_db_sha256_after": source_hash_after,
        "execution_db_path": str(db_path),
        "execution_db_sha256_before": db_hash_before,
        "execution_db_sha256_after": db_hash_after,
        "mode": "write" if write else "dry-run",
        "before": before_inventory,
        "after": after_inventory,
        "planned_component_resolution_rows": len(planned_resolution_rows),
        "planned_vde_component_resolution_links": len(planned_links),
        "inserted_component_resolution_rows": inserted_resolutions,
        "inserted_vde_component_resolution_links": inserted_links,
        "external_search_count": 0,
    }
    (output_dir / "db_integrity.txt").write_text(json.dumps(integrity, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "pass_version": PASS_VERSION,
        "estimator_version": ESTIMATOR_VERSION,
        "mode": integrity["mode"],
        "vde_count": len(population),
        "coverage_before": total_before,
        "coverage_after": total_after,
        "coverage_before_pct": before_pct,
        "coverage_after_pct": after_pct,
        "coverage_point_gain": after_pct - before_pct,
        "planned_resolution_rows": len(planned_resolution_rows),
        "planned_links": len(planned_links),
        "inserted_resolution_rows": inserted_resolutions,
        "inserted_links": inserted_links,
        "unresolved_unique_groups": len(groups),
        "db_integrity": integrity,
    }
    (output_dir / "execution_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


__all__ = [
    "ESTIMATOR_VERSION",
    "PASS_VERSION",
    "execute_pass1",
    "file_sha256",
    "inventory",
    "route_architecture",
]
