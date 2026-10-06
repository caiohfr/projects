"""Physics-informed, read-only vector methodology experiment (Pass 1C.3)."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from src.vde_core.component_enrichment_pass1 import ESTIMATOR_VERSION, file_sha256
from src.vde_core.component_minor_buildup_pilot import (
    DEFAULT_ARCHITECTURE_TARGETS,
    QA_SPEEDS_KPH,
    Reference,
    _curve,
    _git_metadata,
    _read_only,
    _token,
    _write_csv,
    deterministic_stratified_sample,
    load_legacy_lookup,
    load_resolved_population,
    load_tire_references,
)
from src.vde_core.component_minor_buildup_refinement import (
    ApplicabilityReference,
    classify_closure,
    enrich_applicability,
)
from src.vde_core.component_prior_matching_vnext import normalize_application_class
from src.vde_core.component_vector_search import (
    CandidateVector,
    _rank_unique,
    _reference_score,
    _reference_vector,
    build_candidate_pools,
    generate_axle_pool,
    load_historical_tire_evidence,
)


METHOD_VERSION = "SPRINT12_PASS1C3_PHYSICS_INFORMED_VECTOR_SEARCH_V1.0"
ROLLING_DOMAINS = ("TIRE", "BRAKE", "HUB_BEARING")
DRIVETRAIN_DOMAINS = ("TRANSMISSION", "AXLE")
NOMINAL_CAPS = {"TIRE": 4, "BRAKE": 3, "HUB_BEARING": 3, "TRANSMISSION": 3, "AXLE": 3}
MATERIAL_ACCEPTANCE_LIFT_PP = 10.0
MATERIAL_RESIDUAL_IMPROVEMENT_PP = 5.0
POOL_STABILITY_ACCEPTANCE_PP = 5.0
POOL_STABILITY_RESIDUAL_PP = 5.0


def load_all_component_references(path: Path) -> list[Reference]:
    rows: list[Reference] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            domain = _token(row.get("domain"))
            if domain not in {"BRAKE", "TRANSMISSION", "AXLE", "HUB_BEARING"}:
                continue
            rows.append(Reference(
                component_id=str(row["component_id"]),
                resolution_id=str(row["component_resolution_id"]),
                domain=domain,
                application_class=_token(row.get("application_class")) or "",
                drive=_token(row.get("drive_architecture")) or "",
                position=_token(row.get("position")),
                abc=(float(row["A_N"]), float(row["B_N_per_kph"]), float(row["C_N_per_kph2"])),
                population_n=int(row["population_n"]) if row.get("population_n") else None,
            ))
    return sorted(rows, key=lambda item: (item.domain, item.component_id))


def load_aggregate_slots(connection, vde_ids: Iterable[int]) -> dict[int, dict[str, dict[str, Any]]]:
    ids = sorted(set(int(value) for value in vde_ids))
    placeholders = ",".join("?" for _ in ids)
    sql = f"""
        SELECT link.vde_id,link.boundary,resolution.component_resolution_id,
               resolution.method,resolution.estimate_status,resolution.resolved_A_N,
               resolution.resolved_B_N_per_kph,resolution.resolved_C_N_per_kph2,
               resolution.provenance_json
        FROM vde_component_resolution AS link
        JOIN component_resolution AS resolution
          ON resolution.component_resolution_id=link.component_resolution_id
        WHERE link.vde_id IN ({placeholders})
          AND link.boundary IN ('ROLLING_MINOR','DRIVETRAIN_AGGREGATE','EDRIVE_AGGREGATE','DHT_AGGREGATE')
          AND resolution.record_status='ACTIVE'
          AND resolution.estimate_status IN ('SUPPORTED','CONDITIONAL')
          AND resolution.estimator_version=?
        ORDER BY link.vde_id,link.boundary
    """
    result: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for raw in connection.execute(sql, (*ids, ESTIMATOR_VERSION)):
        row = dict(raw)
        row["abc"] = (float(row["resolved_A_N"]), float(row["resolved_B_N_per_kph"]), float(row["resolved_C_N_per_kph2"]))
        row["provenance"] = json.loads(row.get("provenance_json") or "{}")
        result[int(row["vde_id"])][str(row["boundary"])] = row
    return result


def classify_boundary(vehicle: dict[str, Any], slots: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rolling = slots.get("ROLLING_MINOR")
    architecture = str(vehicle.get("architecture") or "UNRESOLVED")
    aggregate = None
    boundary_class = "BOUNDARY_UNKNOWN"
    reason = "NO_DEFENSIBLE_AGGREGATE_MAPPING"
    if rolling and architecture in {"CONVENTIONAL_MULTI_SPEED", "HYBRID_PARALLEL_CONVENTIONAL_TRANS", "EV_MULTI_SPEED"}:
        aggregate = slots.get("DRIVETRAIN_AGGREGATE")
        if aggregate and "MOSKALIK" in str(aggregate.get("method") or ""):
            boundary_class = "MOSKALIK_DRIVELINE_COMPATIBLE"
            reason = "FROZEN_MOSKALIK_ROLLING_AND_DRIVETRAIN_BOUNDARIES"
    elif rolling and architecture == "EV_FIXED_GEAR":
        aggregate = slots.get("EDRIVE_AGGREGATE")
        if aggregate and str(aggregate.get("method")) == "FIXED_GEAR_EDRIVE_V1_QROLL_050":
            boundary_class = "FIXED_GEAR_BOUNDARY_COMPATIBLE"
            reason = "FROZEN_FIXED_GEAR_ROLLING_AND_EDRIVE_BOUNDARIES"
    elif rolling and "DHT" in architecture:
        aggregate = slots.get("DHT_AGGREGATE")
        if aggregate:
            boundary_class = "DHT_AGGREGATE_COMPATIBLE"
            reason = "FROZEN_DHT_AGGREGATE_BOUNDARY"
    return {
        "boundary_class": boundary_class,
        "reason_code": reason,
        "rolling_boundary": rolling,
        "drivetrain_boundary": aggregate,
        "drivetrain_boundary_name": aggregate.get("boundary") if aggregate else None,
    }


def resolve_application_class(vehicle: dict[str, Any]) -> dict[str, Any]:
    canonical = normalize_application_class((str(vehicle.get("category") or ""),))
    if canonical.status == "MAPPED" and len(canonical.normalized_options) == 1:
        return {"application_class": canonical.normalized_options[0], "source": "CANONICAL", "status": "MAPPED"}
    if vehicle.get("application_class"):
        return {"application_class": vehicle["application_class"], "source": "LEGACY_RULE", "status": "MAPPED"}
    return {"application_class": None, "source": "UNRESOLVED", "status": canonical.status}


def generate_transmission_pool(
    vehicle: dict[str, Any], references: list[ApplicabilityReference], *, limit: int = 5,
) -> list[CandidateVector]:
    vectors: list[CandidateVector] = []
    for item in references:
        if item.reference.domain != "TRANSMISSION":
            continue
        scored = _reference_score(vehicle, item)
        if scored is None:
            continue
        vector = _reference_vector(item, *scored)
        features = dict(vector.feature_vector)
        features["powertrain_architecture"] = vehicle.get("architecture")
        features["transmission_family"] = "MISSING_IN_SYNTHETIC_REFERENCE"
        vectors.append(CandidateVector(
            **{**vector.__dict__, "feature_vector": features,
               "reason_codes": tuple((*vector.reason_codes, "TRANSMISSION_FAMILY_METADATA_MISSING")),
               "boundary_assumption": "TRANSMISSION_ALLOCATED_WITHIN_DRIVETRAIN_AGGREGATE"}
        ))
    return _rank_unique(vectors, limit)


def build_master_pools(
    vehicle: dict[str, Any], references: list[ApplicabilityReference],
    tire_references: list[dict[str, Any]], historical_tires: list[dict[str, Any]],
) -> dict[str, list[CandidateVector]]:
    base = build_candidate_pools(
        vehicle, references, tire_references, historical_tires,
        caps={"TIRE": 5, "BRAKE": 5, "HUB_BEARING": 5, "AXLE": 5},
    )
    base["TRANSMISSION"] = generate_transmission_pool(vehicle, references, limit=5)
    return base


def truncate_pools(
    pools: dict[str, list[CandidateVector]], caps: dict[str, int],
) -> dict[str, list[CandidateVector]]:
    return {domain: list(values[: caps.get(domain, len(values))]) for domain, values in pools.items()}


def _combination_id(vde_id: int, aggregate: str, vectors: list[CandidateVector]) -> str:
    payload = "|".join((str(vde_id), aggregate, *(vector.vector_id for vector in vectors)))
    return f"P13-{sha256(payload.encode()).hexdigest()[:18].upper()}"


def search_aggregate(
    vehicle: dict[str, Any], aggregate_name: str, aggregate_abc: tuple[float, float, float],
    pools: dict[str, list[CandidateVector]], domains: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any]]:
    complete_pool = all(bool(pools.get(domain)) for domain in domains)
    choices = [pools.get(domain) or [None] for domain in domains]
    accepted: list[dict[str, Any]] = []
    tested = 0
    for selected in product(*choices):
        vectors = [vector for vector in selected if vector is not None]
        if not vectors:
            continue
        tested += 1
        abc_by_domain = {domain: vector.abc for domain, vector in zip(domains, selected) if vector is not None}
        closure = classify_closure(aggregate_abc, abc_by_domain)
        if closure["status"] == "REJECTED_BUILDUP":
            continue
        residuals: list[float] = []
        residual_vector: list[float] = []
        closure_vector: list[float] = []
        for speed in QA_SPEEDS_KPH:
            aggregate_force = _curve(aggregate_abc, speed)
            build_force = sum(_curve(vector.abc, speed) for vector in vectors)
            unresolved = max(0.0, aggregate_force - build_force)
            residual_vector.append(unresolved)
            closure_vector.append(max(0.0, build_force - aggregate_force))
            residuals.append(unresolved / max(abs(aggregate_force), 1e-9) * 100.0)
        vector_map = {domain: (vector.vector_id if vector else None) for domain, vector in zip(domains, selected)}
        row = {
            "vde_id": vehicle["vde_id"], "aggregate": aggregate_name,
            "combination_id": _combination_id(int(vehicle["vde_id"]), aggregate_name, vectors),
            **{f"{domain.lower()}_vector": vector_map.get(domain) for domain in domains},
            "component_count": len(vectors), "build_status": "FULL_BUILDUP" if complete_pool else "PARTIAL_BUILDUP",
            "compatibility_score": sum(vector.compatibility_score for vector in vectors),
            "closure_status": closure["status"], "max_overshoot_pct": closure["max_overshoot_pct"],
            "median_unresolved_residual_pct": median(residuals),
            "p90_unresolved_residual_pct": _percentile(residuals, 0.9),
            "median_explained_pct": 100.0 - median(residuals),
            "other_unresolved_vector_json": json.dumps([round(value, 9) for value in residual_vector], separators=(",", ":")),
            "other_minor_unresolved_vector_json": (
                json.dumps([round(value, 9) for value in residual_vector], separators=(",", ":"))
                if aggregate_name == "ROLLING_MINOR" else None
            ),
            "other_driveline_unresolved_vector_json": (
                json.dumps([round(value, 9) for value in residual_vector], separators=(",", ":"))
                if aggregate_name != "ROLLING_MINOR" else None
            ),
            "closure_error_vector_json": json.dumps([round(value, 9) for value in closure_vector], separators=(",", ":")),
            "method": "SYNTHETIC_VECTOR_MATCHED", "fidelity": "SURROGATE_COMPONENT_DECOMPOSITION",
            "estimate_status": "CONDITIONAL", "scaled": 0,
        }
        accepted.append(row)
    accepted.sort(key=lambda row: (
        -float(row["compatibility_score"]), float(row["median_unresolved_residual_pct"]),
        float(row["max_overshoot_pct"]), row["combination_id"],
    ))
    for rank, row in enumerate(accepted, 1):
        row["rank"] = rank
    diagnostics = {"complete_pool": int(complete_pool), "tested": tested, "accepted": len(accepted)}
    return accepted, (accepted[0] if accepted else None), diagnostics


def shuffle_pools(
    pools_by_vde: dict[int, dict[str, list[CandidateVector]]], domains: tuple[str, ...],
) -> dict[int, dict[str, list[CandidateVector]]]:
    result = {vde_id: {domain: list(values) for domain, values in pools.items()} for vde_id, pools in pools_by_vde.items()}
    for domain in domains:
        groups: dict[int, list[int]] = defaultdict(list)
        for vde_id, pools in pools_by_vde.items():
            groups[len(pools.get(domain, []))].append(vde_id)
        for size, ids in groups.items():
            ids.sort()
            if size == 0 or len(ids) < 2:
                continue
            donor_lists = [list(pools_by_vde[vde_id][domain]) for vde_id in ids]
            for index, vde_id in enumerate(ids):
                result[vde_id][domain] = donor_lists[(index + 1) % len(ids)]
    return result


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _candidate_rows(vde_id: int, pools: dict[str, list[CandidateVector]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain in ("TIRE", "BRAKE", "HUB_BEARING", "TRANSMISSION", "AXLE"):
        for rank, vector in enumerate(pools.get(domain, []), 1):
            matched = [key for key, value in vector.feature_vector.items() if value is True]
            missing = [key for key, value in vector.feature_vector.items() if isinstance(value, str) and ("MISSING" in value or "UNKNOWN" in value)]
            grade = "A" if vector.compatibility_score >= 85 else ("B" if vector.compatibility_score >= 65 else "C")
            rows.append({
                "vde_id": vde_id, "domain": domain, "candidate_rank": rank,
                "candidate_id": vector.vector_id, "reference_ids": ";".join(vector.reference_ids),
                "compatibility_score": vector.compatibility_score, "compatibility_grade": grade,
                "matched_dimensions": ";".join(sorted(matched)), "missing_dimensions": ";".join(sorted(missing)),
                "reason_codes": ";".join(vector.reason_codes), "provenance": vector.provenance,
                "boundary_assumption": vector.boundary_assumption, "A_N": vector.abc[0],
                "B_N_per_kph": vector.abc[1], "C_N_per_kph2": vector.abc[2],
                "closure_used_in_eligibility": 0,
                "within_nominal_pool": int(rank <= NOMINAL_CAPS[domain]),
            })
    return rows


def _run_search_set(
    vehicles: list[dict[str, Any]], boundaries: dict[int, dict[str, Any]],
    pools_by_vde: dict[int, dict[str, list[CandidateVector]]], caps: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rolling_selected: list[dict[str, Any]] = []
    drivetrain_selected: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for vehicle in vehicles:
        vde_id = int(vehicle["vde_id"])
        boundary = boundaries[vde_id]
        if boundary["boundary_class"] == "BOUNDARY_UNKNOWN":
            records.append({"vde_id": vde_id, "aggregate": "ROLLING_MINOR", "eligible": 0, "complete_pool": 0, "selected": None})
            records.append({"vde_id": vde_id, "aggregate": "DRIVETRAIN", "eligible": 0, "complete_pool": 0, "selected": None})
            continue
        pools = truncate_pools(pools_by_vde[vde_id], caps)
        rolling_all, rolling_best, rolling_diag = search_aggregate(
            vehicle, "ROLLING_MINOR", boundary["rolling_boundary"]["abc"], pools, ROLLING_DOMAINS
        )
        drive_slot = boundary["drivetrain_boundary"]
        drivetrain_all, drivetrain_best, drivetrain_diag = search_aggregate(
            vehicle, boundary["drivetrain_boundary_name"], drive_slot["abc"], pools, DRIVETRAIN_DOMAINS
        )
        if rolling_best:
            rolling_selected.append(rolling_best)
        if drivetrain_best:
            drivetrain_selected.append(drivetrain_best)
        records.extend([
            {"vde_id": vde_id, "aggregate": "ROLLING_MINOR", "eligible": 1,
             "complete_pool": rolling_diag["complete_pool"], "selected": rolling_best,
             "accepted_combinations": rolling_all},
            {"vde_id": vde_id, "aggregate": "DRIVETRAIN", "eligible": 1,
             "complete_pool": drivetrain_diag["complete_pool"], "selected": drivetrain_best,
             "accepted_combinations": drivetrain_all},
        ])
    return rolling_selected, drivetrain_selected, records


def aggregate_kpis(records: list[dict[str, Any]], aggregate: str) -> dict[str, Any]:
    subset = [row for row in records if row["aggregate"] == aggregate]
    eligible = [row for row in subset if row["eligible"]]
    complete = [row for row in eligible if row["complete_pool"]]
    accepted_complete = [row["selected"] for row in complete if row.get("selected") and row["selected"]["build_status"] == "FULL_BUILDUP"]
    partial = [row["selected"] for row in eligible if row.get("selected") and row["selected"]["build_status"] == "PARTIAL_BUILDUP"]
    residuals = [float(row["median_unresolved_residual_pct"]) for row in accepted_complete]
    overshoots = [float(row["max_overshoot_pct"]) for row in accepted_complete]
    return {
        "eligible_vdes": len(eligible), "complete_candidate_pools": len(complete),
        "accepted_complete_builds": len(accepted_complete), "partial_builds": len(partial),
        "boundary_unknown": len(subset) - len(eligible),
        "complete_pool_availability_rate_pct": 100.0 * len(complete) / len(eligible) if eligible else 0.0,
        "physical_acceptance_rate_pct": 100.0 * len(accepted_complete) / len(complete) if complete else 0.0,
        "median_explained_pct": 100.0 - median(residuals) if residuals else None,
        "median_unresolved_pct": median(residuals) if residuals else None,
        "p90_unresolved_pct": _percentile(residuals, 0.9),
        "median_max_overshoot_pct": median(overshoots) if overshoots else None,
        "max_accepted_overshoot_pct": max(overshoots, default=None),
    }


def _summary_groups(
    vehicles: list[dict[str, Any]], boundaries: dict[int, dict[str, Any]], records: list[dict[str, Any]], key: str,
) -> list[dict[str, Any]]:
    by_vde = {(row["vde_id"], row["aggregate"]): row for row in records}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for vehicle in vehicles:
        label = str(vehicle.get(key) or "UNRESOLVED")
        groups[label].append(vehicle)
    rows: list[dict[str, Any]] = []
    for label, group in sorted(groups.items()):
        ids = [int(vehicle["vde_id"]) for vehicle in group]
        rolling = [by_vde[(value, "ROLLING_MINOR")] for value in ids]
        drive = [by_vde[(value, "DRIVETRAIN")] for value in ids]
        rows.append({
            key: label, "vde_count": len(ids),
            "boundary_compatible": sum(boundaries[value]["boundary_class"] != "BOUNDARY_UNKNOWN" for value in ids),
            "rolling_complete_pools": sum(row["complete_pool"] for row in rolling),
            "rolling_selected_full": sum(bool(row.get("selected") and row["selected"]["build_status"] == "FULL_BUILDUP") for row in rolling),
            "rolling_selected_partial": sum(bool(row.get("selected") and row["selected"]["build_status"] == "PARTIAL_BUILDUP") for row in rolling),
            "drivetrain_complete_pools": sum(row["complete_pool"] for row in drive),
            "drivetrain_selected_full": sum(bool(row.get("selected") and row["selected"]["build_status"] == "FULL_BUILDUP") for row in drive),
            "drivetrain_selected_partial": sum(bool(row.get("selected") and row["selected"]["build_status"] == "PARTIAL_BUILDUP") for row in drive),
        })
    return rows


def _failure_reasons(
    vehicles: list[dict[str, Any]], boundaries: dict[int, dict[str, Any]], pools: dict[int, dict[str, list[CandidateVector]]],
    records: list[dict[str, Any]],
) -> Counter:
    result: Counter = Counter()
    record_map = {(row["vde_id"], row["aggregate"]): row for row in records}
    for vehicle in vehicles:
        vde_id = int(vehicle["vde_id"])
        if boundaries[vde_id]["boundary_class"] == "BOUNDARY_UNKNOWN":
            result["BOUNDARY_UNKNOWN"] += 1
            continue
        if not vehicle.get("application_class"):
            result["APPLICATION_CLASS_UNRESOLVED"] += 1
        for domain, code in (("TIRE", "NO_ELIGIBLE_TIRE"), ("BRAKE", "NO_ELIGIBLE_BRAKE"),
                             ("HUB_BEARING", "NO_ELIGIBLE_HUB"), ("TRANSMISSION", "NO_ELIGIBLE_TRANSMISSION"),
                             ("AXLE", "NO_ELIGIBLE_AXLE")):
            if not pools[vde_id].get(domain):
                result[code] += 1
        for aggregate in ("ROLLING_MINOR", "DRIVETRAIN"):
            row = record_map[(vde_id, aggregate)]
            if row["complete_pool"] and not row.get("selected"):
                result["OVERSHOOT_GT_5PCT"] += 1
            if row.get("selected") and float(row["selected"]["median_unresolved_residual_pct"]) > 70.0:
                result["HIGH_UNRESOLVED_RESIDUAL"] += 1
    return result


def _decision(
    boundary_compatible: int, real: dict[str, dict[str, Any]], shuffled: dict[str, dict[str, Any]],
    sensitivity: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    effects: dict[str, dict[str, float | None]] = {}
    for aggregate in ("ROLLING_MINOR", "DRIVETRAIN"):
        real_kpi, shuffled_kpi = real[aggregate], shuffled[aggregate]
        residual_real = real_kpi["median_unresolved_pct"]
        residual_shuffled = shuffled_kpi["median_unresolved_pct"]
        effects[aggregate] = {
            "acceptance_lift_pp": real_kpi["physical_acceptance_rate_pct"] - shuffled_kpi["physical_acceptance_rate_pct"],
            "residual_improvement_pp": (
                residual_shuffled - residual_real
                if residual_real is not None and residual_shuffled is not None else None
            ),
        }
    sensitivity_map = {(row["pool_size"], row["aggregate"]): row for row in sensitivity}
    stable = True
    for aggregate in ("ROLLING_MINOR", "DRIVETRAIN"):
        top3 = sensitivity_map[("TOP_3", aggregate)]
        top5 = sensitivity_map[("TOP_5", aggregate)]
        if abs(top5["physical_acceptance_rate_pct"] - top3["physical_acceptance_rate_pct"]) > POOL_STABILITY_ACCEPTANCE_PP:
            stable = False
        if top3["median_unresolved_pct"] is not None and top5["median_unresolved_pct"] is not None:
            if abs(top5["median_unresolved_pct"] - top3["median_unresolved_pct"]) > POOL_STABILITY_RESIDUAL_PP:
                stable = False
    if boundary_compatible < 36:
        decision = "BLOCKED_BY_BOUNDARY"
    else:
        material = [
            values["acceptance_lift_pp"] >= MATERIAL_ACCEPTANCE_LIFT_PP
            and values["residual_improvement_pp"] is not None
            and values["residual_improvement_pp"] >= MATERIAL_RESIDUAL_IMPROVEMENT_PP
            for values in effects.values()
        ]
        positive = [
            values["acceptance_lift_pp"] > 0
            or (values["residual_improvement_pp"] is not None and values["residual_improvement_pp"] > 0)
            for values in effects.values()
        ]
        if all(material) and stable:
            decision = "METHOD_SUPPORTED_FOR_SCALE_PILOT"
        elif any(material) or (all(positive) and stable):
            decision = "METHOD_PROMISING_NEEDS_REFINEMENT"
        else:
            decision = "METHOD_NOT_SUPPORTED"
    return decision, {"effects": effects, "pool_stable_top3_to_top5": stable}


def execute_physics_vector_search(
    db_path: Path, legacy_db_path: Path, component_catalog_path: Path, tire_zip_path: Path, output_dir: Path,
) -> dict[str, Any]:
    db_path = Path(db_path).resolve(strict=True); legacy_db_path = Path(legacy_db_path).resolve(strict=True)
    component_catalog_path = Path(component_catalog_path).resolve(strict=True); tire_zip_path = Path(tire_zip_path).resolve(strict=True)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    hashes_before = {"pass1_db": file_sha256(db_path), "legacy_db": file_sha256(legacy_db_path)}
    connection = _read_only(db_path); legacy = _read_only(legacy_db_path)
    try:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        fk_issues = len(connection.execute("PRAGMA foreign_key_check").fetchall())
        population = load_resolved_population(connection, load_legacy_lookup(legacy))
        sample = deterministic_stratified_sample(population, DEFAULT_ARCHITECTURE_TARGETS)
        slots = load_aggregate_slots(connection, (row["vde_id"] for row in sample))
        historical_tires = load_historical_tire_evidence(legacy)
    finally:
        connection.close(); legacy.close()
    if len(sample) != 180:
        raise RuntimeError(f"Expected frozen 180-VDE sample, found {len(sample)}")
    application_rows: list[dict[str, Any]] = []
    for vehicle in sample:
        resolution = resolve_application_class(vehicle)
        vehicle["application_class"] = resolution["application_class"]
        vehicle["application_class_source"] = resolution["source"]
        application_rows.append({
            "vde_id": vehicle["vde_id"], "canonical_category": vehicle.get("category"),
            "application_class": resolution["application_class"], "application_class_source": resolution["source"],
            "resolution_status": resolution["status"], "drive_layout": vehicle.get("drive_layout"),
            "test_mass_kg": vehicle.get("test_mass_resolved_kg"), "tire_size": vehicle.get("tire_size_resolved"),
            "powertrain_architecture": vehicle.get("architecture"),
        })
    boundaries = {int(vehicle["vde_id"]): classify_boundary(vehicle, slots.get(int(vehicle["vde_id"]), {})) for vehicle in sample}
    boundary_rows = [{
        "vde_id": vehicle["vde_id"], "architecture": vehicle["architecture"],
        "boundary_class": boundaries[int(vehicle["vde_id"])]["boundary_class"],
        "reason_code": boundaries[int(vehicle["vde_id"])]["reason_code"],
        "rolling_method": (boundaries[int(vehicle["vde_id"])]["rolling_boundary"] or {}).get("method"),
        "rolling_status": (boundaries[int(vehicle["vde_id"])]["rolling_boundary"] or {}).get("estimate_status"),
        "drivetrain_boundary": boundaries[int(vehicle["vde_id"])]["drivetrain_boundary_name"],
        "drivetrain_method": (boundaries[int(vehicle["vde_id"])]["drivetrain_boundary"] or {}).get("method"),
        "drivetrain_status": (boundaries[int(vehicle["vde_id"])]["drivetrain_boundary"] or {}).get("estimate_status"),
    } for vehicle in sample]
    references = enrich_applicability(load_all_component_references(component_catalog_path))
    tire_refs = load_tire_references(tire_zip_path)
    pools_by_vde: dict[int, dict[str, list[CandidateVector]]] = {}
    candidate_rows: list[dict[str, Any]] = []
    for vehicle in sample:
        vde_id = int(vehicle["vde_id"])
        pools = build_master_pools(vehicle, references, tire_refs, historical_tires)
        pools_by_vde[vde_id] = pools
        candidate_rows.extend(_candidate_rows(vde_id, pools))
    nominal_pools = {vde_id: truncate_pools(pools, NOMINAL_CAPS) for vde_id, pools in pools_by_vde.items()}
    rolling_selected, drivetrain_selected, real_records = _run_search_set(sample, boundaries, nominal_pools, NOMINAL_CAPS)

    rolling_all = [combination for row in real_records if row["aggregate"] == "ROLLING_MINOR" for combination in row.get("accepted_combinations", [])]
    drivetrain_all = [combination for row in real_records if row["aggregate"] == "DRIVETRAIN" for combination in row.get("accepted_combinations", [])]
    shuffled_pools = shuffle_pools(nominal_pools, ("TIRE", "BRAKE", "HUB_BEARING", "TRANSMISSION", "AXLE"))
    _, _, shuffled_records = _run_search_set(sample, boundaries, shuffled_pools, NOMINAL_CAPS)
    real_kpis = {aggregate: aggregate_kpis(real_records, aggregate) for aggregate in ("ROLLING_MINOR", "DRIVETRAIN")}
    shuffled_kpis = {aggregate: aggregate_kpis(shuffled_records, aggregate) for aggregate in ("ROLLING_MINOR", "DRIVETRAIN")}
    shuffle_rows = []
    for aggregate in ("ROLLING_MINOR", "DRIVETRAIN"):
        real, shuffled = real_kpis[aggregate], shuffled_kpis[aggregate]
        shuffle_rows.append({
            "aggregate": aggregate,
            **{f"real_{key}": value for key, value in real.items()},
            **{f"shuffled_{key}": value for key, value in shuffled.items()},
            "acceptance_lift_pp": real["physical_acceptance_rate_pct"] - shuffled["physical_acceptance_rate_pct"],
            "residual_improvement_pp": (
                shuffled["median_unresolved_pct"] - real["median_unresolved_pct"]
                if real["median_unresolved_pct"] is not None and shuffled["median_unresolved_pct"] is not None else None
            ),
        })
    sensitivity_rows: list[dict[str, Any]] = []
    sensitivity_specs = {
        "TOP_1": {domain: 1 for domain in NOMINAL_CAPS},
        "TOP_2": {domain: 2 for domain in NOMINAL_CAPS},
        "TOP_3": {domain: 3 for domain in NOMINAL_CAPS},
        "NOMINAL": NOMINAL_CAPS,
        "TOP_5": {domain: 5 for domain in NOMINAL_CAPS},
    }
    for label, caps in sensitivity_specs.items():
        _, _, records = _run_search_set(sample, boundaries, pools_by_vde, caps)
        for aggregate in ("ROLLING_MINOR", "DRIVETRAIN"):
            sensitivity_rows.append({"pool_size": label, "aggregate": aggregate, **aggregate_kpis(records, aggregate)})
    architecture_rows = _summary_groups(sample, boundaries, real_records, "architecture")
    architecture_labels = {row["architecture"] for row in architecture_rows}
    empty_architecture_row = {
        "vde_count": 0, "boundary_compatible": 0, "rolling_complete_pools": 0,
        "rolling_selected_full": 0, "rolling_selected_partial": 0,
        "drivetrain_complete_pools": 0, "drivetrain_selected_full": 0,
        "drivetrain_selected_partial": 0,
    }
    if not any("DHT" in label or label == "OTHER" for label in architecture_labels):
        architecture_rows.append({"architecture": "DHT_OTHER", **empty_architecture_row})
    if not any(value["boundary_class"] == "BOUNDARY_UNKNOWN" for value in boundaries.values()):
        architecture_rows.append({"architecture": "BOUNDARY_UNKNOWN", **empty_architecture_row})
    application_class_rows = _summary_groups(sample, boundaries, real_records, "application_class")
    failure_reasons = _failure_reasons(sample, boundaries, nominal_pools, real_records)
    compatible_count = sum(value["boundary_class"] != "BOUNDARY_UNKNOWN" for value in boundaries.values())
    decision, decision_evidence = _decision(compatible_count, real_kpis, shuffled_kpis, sensitivity_rows)

    _write_csv(output_dir / "boundary_classification.csv", boundary_rows, boundary_rows[0].keys())
    _write_csv(output_dir / "application_class_resolution.csv", application_rows, application_rows[0].keys())
    _write_csv(output_dir / "candidate_pools.csv", candidate_rows, candidate_rows[0].keys())
    selected_fields = sorted({key for row in (*rolling_selected, *drivetrain_selected) for key in row})
    _write_csv(output_dir / "rollingminor_selected_combinations.csv", rolling_selected, selected_fields)
    _write_csv(output_dir / "drivetrain_selected_combinations.csv", drivetrain_selected, selected_fields)
    _write_csv(output_dir / "rollingminor_all_accepted_combinations.csv", rolling_all, selected_fields)
    _write_csv(output_dir / "drivetrain_all_accepted_combinations.csv", drivetrain_all, selected_fields)
    _write_csv(output_dir / "shuffle_control_summary.csv", shuffle_rows, shuffle_rows[0].keys())
    _write_csv(output_dir / "pool_size_sensitivity.csv", sensitivity_rows, sensitivity_rows[0].keys())
    _write_csv(output_dir / "architecture_summary.csv", architecture_rows, architecture_rows[0].keys())
    _write_csv(output_dir / "application_class_summary.csv", application_class_rows, application_class_rows[0].keys())
    sample_rows = [{"vde_id": row["vde_id"], "vehicle_configuration_id": row["vehicle_configuration_id"],
                    "architecture": row["architecture"], "application_class": row.get("application_class"),
                    "drive_layout": row.get("drive_layout"), "test_mass_kg": row.get("test_mass_resolved_kg"),
                    "stratum": row.get("stratum"), "parent_rolling_minor_resolution_id": row["rolling_minor_resolution_id"]}
                   for row in sample]
    _write_csv(output_dir / "pilot_sample.csv", sample_rows, sample_rows[0].keys())
    hashes_after = {"pass1_db": file_sha256(db_path), "legacy_db": file_sha256(legacy_db_path)}
    if hashes_before != hashes_after:
        raise RuntimeError("Read-only physics experiment changed an input database")
    summary = {
        "method_version": METHOD_VERSION, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "git": _git_metadata(),
        "sample_size": len(sample),
        "sample_identity_sha256": sha256(",".join(str(row["vde_id"]) for row in sample_rows).encode()).hexdigest().upper(),
        "boundary_distribution": dict(sorted(Counter(row["boundary_class"] for row in boundary_rows).items())),
        "application_source_distribution": dict(sorted(Counter(row["application_class_source"] for row in application_rows).items())),
        "real_kpis": real_kpis, "shuffled_kpis": shuffled_kpis,
        "decision_evidence": decision_evidence, "failure_reasons": dict(failure_reasons.most_common()),
        "candidate_count": len(candidate_rows), "rolling_accepted_combination_count": len(rolling_all),
        "drivetrain_accepted_combination_count": len(drivetrain_all),
        "hashes_before": hashes_before, "hashes_after": hashes_after,
        "quick_check": quick_check, "foreign_key_issues": fk_issues, "database_rows_written": 0,
        "database_objects_created": 0, "external_search_count": 0, "scaled_combination_count": 0,
        "scale_up_executed": False, "decision": decision,
    }
    (output_dir / "execution_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "pass1c3_methodology_report.md").write_text(_report(summary, shuffle_rows, sensitivity_rows), encoding="utf-8")
    return summary


def _fmt(value: Any) -> str:
    return "N/A" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value))


def _report(summary: dict[str, Any], shuffle_rows: list[dict[str, Any]], sensitivity_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Pass 1C.3 Methodology Report", "", "## 1. Dataset", "",
        f"- exact 180-VDE pilot reused: **YES**, identity `{summary['sample_identity_sha256']}`",
        "- DB writes: **0**", "- external search: **0**", "- component scaling: **0**", "",
        "## 2. Boundary coverage", "", "| Boundary class | VDEs | % |", "|---|---:|---:|",
    ]
    for boundary, count in summary["boundary_distribution"].items():
        lines.append(f"| {boundary} | {count} | {100.0 * count / summary['sample_size']:.3f}% |")
    for aggregate, title in (("ROLLING_MINOR", "3. RollingMinor results"), ("DRIVETRAIN", "4. Drivetrain results")):
        real, shuffled = summary["real_kpis"][aggregate], summary["shuffled_kpis"][aggregate]
        lines.extend(["", f"## {title}", "", "| KPI | Real matching | Shuffled control |", "|---|---:|---:|"])
        for label, key in (("Eligible", "eligible_vdes"), ("Complete pools", "complete_candidate_pools"),
                           ("Accepted complete builds", "accepted_complete_builds"),
                           ("Acceptance rate %", "physical_acceptance_rate_pct"),
                           ("Median explained %", "median_explained_pct"),
                           ("Median unresolved %", "median_unresolved_pct"),
                           ("P90 unresolved %", "p90_unresolved_pct"),
                           ("Max accepted overshoot %", "max_accepted_overshoot_pct")):
            lines.append(f"| {label} | {_fmt(real[key])} | {_fmt(shuffled[key])} |")
    lines.extend(["", "## 5. Pool-size sensitivity", "",
                  "| Pool size | Rolling acceptance % | Rolling median residual % | Drivetrain acceptance % | Drivetrain median residual % |",
                  "|---|---:|---:|---:|---:|"])
    by_size = defaultdict(dict)
    for row in sensitivity_rows:
        by_size[row["pool_size"]][row["aggregate"]] = row
    for label in ("TOP_1", "TOP_2", "TOP_3", "NOMINAL", "TOP_5"):
        rolling, drive = by_size[label]["ROLLING_MINOR"], by_size[label]["DRIVETRAIN"]
        lines.append(f"| {label} | {_fmt(rolling['physical_acceptance_rate_pct'])} | {_fmt(rolling['median_unresolved_pct'])} | {_fmt(drive['physical_acceptance_rate_pct'])} | {_fmt(drive['median_unresolved_pct'])} |")
    lines.extend(["", "## 6. Architecture and application stratification", "",
                  "Detailed deterministic strata are exported in `architecture_summary.csv` and `application_class_summary.csv`.", "",
                  "## 7. Failure reasons", ""])
    for reason, count in summary["failure_reasons"].items():
        lines.append(f"- {reason}: **{count}**")
    lines.extend(["", "## 8. Determinism and QA", "",
                  "- deterministic rerun: recorded separately in `determinism_check.txt`",
                  f"- source DB hash unchanged: **YES** (`{summary['hashes_after']['pass1_db']}`)",
                  f"- quick_check / FK issues: **{summary['quick_check']} / {summary['foreign_key_issues']}**",
                  "- Axle in RollingMinor: **0 cases by construction**",
                  "- Brake/Hub in Drivetrain: **0 cases by construction**", "",
                  "## 9. Final decision", "", f"`{summary['decision']}`", "", "## 10. Interpretation", "",
                  "The experiment finds a limited application-matching signal for RollingMinor acceptance, but little separation from shuffle for Drivetrain. Low complete-pool availability is the main metadata blocker. The result is therefore promising only as a surrogate methodology requiring refinement, not evidence of exact hardware identity. RollingMinor and Drivetrain/EDrive aggregates remain authoritative."])
    return "\n".join(lines) + "\n"


__all__ = [
    "DRIVETRAIN_DOMAINS", "NOMINAL_CAPS", "ROLLING_DOMAINS", "aggregate_kpis", "build_master_pools",
    "classify_boundary", "execute_physics_vector_search", "generate_transmission_pool", "search_aggregate",
    "shuffle_pools", "truncate_pools",
]
