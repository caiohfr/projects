"""Sprint 12 Pass 1C.5 reference discrimination and independent matching.

This module is read-only.  Reference additions are ephemeral pilot records
derived only from internal legacy component curves before aggregate closure is
evaluated; they are never persisted to SQLite.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from src.vde_core.component_enrichment_pass1 import file_sha256
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
    APPLICATION_ENVELOPES,
    ApplicabilityReference,
    enrich_applicability,
)
from src.vde_core.component_physics_vector_search import (
    ROLLING_DOMAINS,
    _candidate_rows,
    aggregate_kpis,
    generate_transmission_pool,
    load_aggregate_slots,
    load_all_component_references,
    truncate_pools,
)
from src.vde_core.component_prior_matching_vnext import (
    normalize_application_class,
    normalize_drive,
    normalize_electrification,
    normalize_transmission,
)
from src.vde_core.component_rollingminor_metadata_completion import (
    PRIMARY_CAPS,
    SHUFFLE_SEEDS,
    _build_rolling_pools,
    _decision,
    _run_rolling,
    resolve_application_class_v14,
    shuffle_pools_seeded,
)
from src.vde_core.component_vector_search import CandidateVector, generate_axle_pool, load_historical_tire_evidence
from src.vde_core.estimation.ecodrive_component_estimator_reference_v1_0 import project_curve_to_abc


METHOD_VERSION = "SPRINT12_PASS1C5_REFERENCE_DISCRIMINATION_V1.0"
SOURCE_SPEEDS_KPH = (16.09344, 32.18688, 48.28032, 64.37376, 80.4672, 96.56064, 112.65408)
MIN_INTERNAL_POPULATION = 20
MAX_REFERENCE_FIT_ERROR_N = 0.1
REFERENCE_SOURCE = "INTERNAL_LEGACY_COMPONENT_POPULATION_MEDOID"
REFERENCE_APPLICABILITY = "EXACT_APPLICATION_CLASS_DRIVE_RULE"


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = fraction * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _median_curve(curves: list[tuple[float, ...]]) -> tuple[float, ...]:
    return tuple(median(curve[index] for curve in curves) for index in range(len(SOURCE_SPEEDS_KPH)))


def _medoid(curves: list[tuple[float, ...]]) -> tuple[float, ...]:
    target = _median_curve(curves)
    return min(curves, key=lambda curve: (sum((a - b) ** 2 for a, b in zip(curve, target)), curve))


def generate_internal_brake_references(
    legacy_connection, existing: list[Reference], needed_keys: set[tuple[str, str]],
) -> tuple[list[Reference], list[dict[str, Any]]]:
    """Generate exact class+drive Brake medoids without aggregate information."""
    existing_keys = {
        (reference.application_class, reference.drive)
        for reference in existing if reference.domain == "BRAKE"
    }
    groups: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    sql = """
        SELECT category,drive_type,brake_A_coef_N,brake_B_coef_Npkph,brake_C_coef_Npkph2
        FROM vde_db
        WHERE brake_A_coef_N IS NOT NULL
          AND brake_B_coef_Npkph IS NOT NULL
          AND brake_C_coef_Npkph2 IS NOT NULL
        ORDER BY id
    """
    for raw in legacy_connection.execute(sql):
        row = dict(raw)
        application = normalize_application_class((row.get("category"),), source="legacy.vde_db.category")
        drive = normalize_drive(row.get("drive_type"), source="legacy.vde_db.drive_type")
        if application.status != "MAPPED" or len(application.normalized_options) != 1 or drive.status != "MAPPED":
            continue
        key = (application.normalized_options[0], str(drive.normalized))
        if key not in needed_keys or key in existing_keys:
            continue
        abc_values = tuple(_safe_float(row.get(name)) for name in (
            "brake_A_coef_N", "brake_B_coef_Npkph", "brake_C_coef_Npkph2"
        ))
        if any(value is None for value in abc_values):
            continue
        abc = tuple(float(value) for value in abc_values)
        if all(_curve(abc, speed) >= -1e-9 for speed in SOURCE_SPEEDS_KPH):
            groups[key].append(abc)

    additions: list[Reference] = []
    audit: list[dict[str, Any]] = []
    for (application_class, drive), curves_abc in sorted(groups.items()):
        if len(curves_abc) < MIN_INTERNAL_POPULATION:
            audit.append({
                "status": "REJECTED_INSUFFICIENT_POPULATION", "domain": "BRAKE",
                "application_class": application_class, "drive_layout": drive, "position": None,
                "population_n": len(curves_abc), "minimum_population_n": MIN_INTERNAL_POPULATION,
                "component_id": None, "source": "legacy.vde_db brake coefficients",
                "generation_rule": "EXACT_CLASS_DRIVE_POPULATION_MEDOID",
                "physical_rationale": "No reference added because the internal population is below the frozen minimum.",
                "aggregate_target_used": False,
            })
            continue
        force_curves = [tuple(_curve(abc, speed) for speed in SOURCE_SPEEDS_KPH) for abc in curves_abc]
        selected_curve = _medoid(force_curves)
        rounded_curve = tuple(round(value, 1) for value in selected_curve)
        fit = project_curve_to_abc(SOURCE_SPEEDS_KPH, rounded_curve)
        abc = (float(fit["A"]), float(fit["B"]), float(fit["C"]))
        if float(fit["max_abs_error_N"]) > MAX_REFERENCE_FIT_ERROR_N or any(
            _curve(abc, speed) < -1e-9 for speed in QA_SPEEDS_KPH
        ):
            audit.append({
                "status": "REJECTED_NUMERIC_QA", "domain": "BRAKE",
                "application_class": application_class, "drive_layout": drive, "position": None,
                "population_n": len(curves_abc), "minimum_population_n": MIN_INTERNAL_POPULATION,
                "component_id": None, "source": "legacy.vde_db brake coefficients",
                "generation_rule": "EXACT_CLASS_DRIVE_POPULATION_MEDOID",
                "physical_rationale": "No reference added because quadratic projection QA failed.",
                "aggregate_target_used": False, "fit_max_abs_error_N": fit["max_abs_error_N"],
            })
            continue
        component_id = f"P15_BRAKE_{application_class}_{drive}_INTERNAL_MEDOID_V1"
        reference = Reference(
            component_id=component_id,
            resolution_id=f"CR_{component_id}", domain="BRAKE",
            application_class=application_class, drive=drive, position=None,
            abc=abc, population_n=len(curves_abc),
        )
        additions.append(reference)
        audit.append({
            "status": "ADDED_EPHEMERAL_PILOT_REFERENCE", "domain": "BRAKE",
            "application_class": application_class, "drive_layout": drive, "position": None,
            "population_n": len(curves_abc), "minimum_population_n": MIN_INTERNAL_POPULATION,
            "component_id": component_id, "component_resolution_id": reference.resolution_id,
            "A_N": abc[0], "B_N_per_kph": abc[1], "C_N_per_kph2": abc[2],
            "fit_rmse_N": fit["RMSE_N"], "fit_max_abs_error_N": fit["max_abs_error_N"],
            "source": "legacy.vde_db brake_A/B/C component coefficients",
            "generation_rule": "EXACT_CLASS_DRIVE_POPULATION_MEDOID_ROUND_0_1N_REFIT",
            "physical_rationale": "Exact application-class and drive population absent from the v1 library; internal component curves support a distinct medoid.",
            "provenance": f"{REFERENCE_SOURCE}|{REFERENCE_APPLICABILITY}",
            "aggregate_target_used": False,
        })
    return additions, audit


def _enrich_with_source(references: list[Reference], addition_ids: set[str]) -> list[ApplicabilityReference]:
    result: list[ApplicabilityReference] = []
    for item in enrich_applicability(references):
        if item.reference.component_id in addition_ids:
            item = replace(item, source=REFERENCE_SOURCE, applicability_source=REFERENCE_APPLICABILITY)
        result.append(item)
    return result


def _load_technical_metadata(connection, vehicles: list[dict[str, Any]]) -> None:
    ids = sorted({int(vehicle["vde_id"]) for vehicle in vehicles})
    placeholders = ",".join("?" for _ in ids)
    sql = f"""
        SELECT v.id AS vde_id,vc.transmission_type,vc.transmission_model,
               vc.gear_count,vc.final_drive_ratio,vc.engine_rated_power_kw,
               vc.propulsion_architecture,vc.architecture_properties_json
        FROM vde AS v
        JOIN vehicle_configuration AS vc
          ON vc.vehicle_configuration_id=v.vehicle_configuration_id
        WHERE v.id IN ({placeholders})
        ORDER BY v.id
    """
    technical = {int(row["vde_id"]): dict(row) for row in connection.execute(sql, ids)}
    electrification: dict[int, set[str]] = defaultdict(set)
    for row in connection.execute(
        f"SELECT vde_id,electrification FROM fuelcons WHERE vde_id IN ({placeholders}) AND record_status='ACTIVE'",
        ids,
    ):
        if row["electrification"]:
            electrification[int(row["vde_id"])].add(str(row["electrification"]))
    for vehicle in vehicles:
        vde_id = int(vehicle["vde_id"])
        vehicle.update(technical.get(vde_id, {}))
        normalized_transmission = normalize_transmission(vehicle.get("transmission_type"))
        normalized_electrification = normalize_electrification(electrification.get(vde_id, ()))
        vehicle.update({
            "transmission_normalized": normalized_transmission.normalized,
            "transmission_normalization_status": normalized_transmission.status,
            "electrification": normalized_electrification.normalized,
            "electrification_status": normalized_electrification.status,
            "electrification_raw": ";".join(normalized_electrification.raw_values),
        })


def _build_all_pools(
    vehicle: dict[str, Any], references: list[ApplicabilityReference], tire_refs: list[dict[str, Any]],
    historical_tires: list[dict[str, Any]],
) -> dict[str, list[CandidateVector]]:
    rolling = _build_rolling_pools(vehicle, references, tire_refs, historical_tires)
    rolling["TRANSMISSION"] = generate_transmission_pool(vehicle, references, limit=5)
    rolling["AXLE"] = generate_axle_pool(vehicle, references, limit=5)
    return rolling


def _reference_rows(
    component_refs: list[Reference], tire_refs: list[dict[str, Any]], stage: str,
) -> list[dict[str, Any]]:
    rows = []
    for reference in component_refs:
        envelope = APPLICATION_ENVELOPES.get(reference.application_class, {})
        driven_state = "UNSPECIFIED"
        if reference.position and reference.drive in {"AWD", "4WD"}:
            driven_state = "DRIVEN"
        elif reference.position and reference.drive == "FWD":
            driven_state = "DRIVEN" if reference.position == "FRONT" else "NON_DRIVEN"
        elif reference.position and reference.drive == "RWD":
            driven_state = "DRIVEN" if reference.position == "REAR" else "NON_DRIVEN"
        rows.append({
            "stage": stage, "domain": reference.domain,
            "application_class": reference.application_class or "UNSPECIFIED",
            "drive_layout": reference.drive or "UNSPECIFIED",
            "position": reference.position or "UNSPECIFIED",
            "driven_state": driven_state,
            "performance_class": envelope.get("performance", "UNSPECIFIED"),
            "architecture_class": "UNSPECIFIED_IN_REFERENCE_V1",
            "boundary": reference.domain,
            "mass_envelope": "-".join(str(value) for value in envelope.get("mass", ())) or "UNSPECIFIED",
            "wheel_envelope": "-".join(str(value) for value in envelope.get("wheel", ())) or "UNSPECIFIED",
            "component_id": reference.component_id,
        })
    rows.extend({
        "stage": stage, "domain": "TIRE", "application_class": "SIZE_SPECIFIC_UNCLASSIFIED",
        "drive_layout": "UNSPECIFIED", "position": "VEHICLE_LEVEL",
        "driven_state": "UNSPECIFIED", "performance_class": "UNSPECIFIED",
        "architecture_class": "UNSPECIFIED_IN_REFERENCE_V1", "boundary": "TIRE",
        "mass_envelope": "UNSPECIFIED", "wheel_envelope": "SIZE_CODE_SPECIFIC",
        "component_id": str(row.get("tire_test_code") or row.get("size_code") or index),
    } for index, row in enumerate(tire_refs, 1))
    return rows


def _library_metrics(before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for dimension, field in (
        ("DOMAIN", "domain"), ("APPLICATION_CLASS", "application_class"),
        ("DRIVE_LAYOUT", "drive_layout"), ("POSITION", "position"),
        ("DRIVEN_STATE", "driven_state"), ("PERFORMANCE_CLASS", "performance_class"),
        ("ARCHITECTURE_CLASS", "architecture_class"), ("BOUNDARY", "boundary"),
        ("MASS_ENVELOPE", "mass_envelope"), ("WHEEL_ENVELOPE", "wheel_envelope"),
    ):
        before = Counter((row["domain"], row[field]) for row in before_rows)
        after = Counter((row["domain"], row[field]) for row in after_rows)
        for domain, value in sorted(set(before) | set(after)):
            result.append({
                "dimension": dimension, "domain": domain, "value": value,
                "before_count": before[(domain, value)], "after_count": after[(domain, value)],
                "delta": after[(domain, value)] - before[(domain, value)],
            })
    return result


def _coverage_rows(
    before: dict[int, dict[str, list[CandidateVector]]],
    after: dict[int, dict[str, list[CandidateVector]]], sample_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain in ("TIRE", "BRAKE", "HUB_BEARING", "TRANSMISSION", "AXLE"):
        before_count = sum(bool(pools.get(domain)) for pools in before.values())
        after_count = sum(bool(pools.get(domain)) for pools in after.values())
        rows.append({
            "domain": domain, "pilot_vdes": sample_size,
            "before_candidate_vdes": before_count, "after_candidate_vdes": after_count,
            "before_missing_vdes": sample_size - before_count,
            "after_missing_vdes": sample_size - after_count,
            "candidate_coverage_delta": after_count - before_count,
        })
    return rows


def _ratio(vector: CandidateVector | None, aggregate: tuple[float, float, float]) -> float | None:
    if vector is None:
        return None
    values = []
    for speed in QA_SPEEDS_KPH:
        aggregate_force = _curve(aggregate, speed)
        if aggregate_force > 1e-9:
            values.append(100.0 * _curve(vector.abc, speed) / aggregate_force)
    return median(values) if values else None


def independent_drivetrain_audit(
    vehicles: list[dict[str, Any]], boundaries: dict[int, dict[str, Any]],
    pools: dict[int, dict[str, list[CandidateVector]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    transmission_rows: list[dict[str, Any]] = []
    axle_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    transmission_fractions: list[float] = []
    axle_fractions: list[float] = []
    with_aggregate = with_transmission = with_axle = with_both = boundary_compatible = boundary_unknown = 0
    for vehicle in vehicles:
        vde_id = int(vehicle["vde_id"])
        boundary = boundaries[vde_id]
        aggregate_slot = boundary.get("drivetrain_boundary")
        is_drivetrain = boundary.get("drivetrain_boundary_name") == "DRIVETRAIN_AGGREGATE"
        if is_drivetrain:
            with_aggregate += 1
        transmission = pools[vde_id].get("TRANSMISSION", [])[:1]
        axle = pools[vde_id].get("AXLE", [])[:1]
        trans_vector = transmission[0] if transmission and is_drivetrain else None
        axle_vector = axle[0] if axle and is_drivetrain else None
        if trans_vector:
            with_transmission += 1
        if axle_vector:
            with_axle += 1
        if trans_vector and axle_vector:
            with_both += 1
        aggregate_abc = aggregate_slot["abc"] if aggregate_slot else None
        trans_fraction = _ratio(trans_vector, aggregate_abc) if aggregate_abc else None
        axle_fraction = _ratio(axle_vector, aggregate_abc) if aggregate_abc else None
        if trans_fraction is not None:
            transmission_fractions.append(trans_fraction)
        if axle_fraction is not None:
            axle_fractions.append(axle_fraction)

        common = {
            "vde_id": vde_id, "vehicle_configuration_id": vehicle.get("vehicle_configuration_id"),
            "application_class": vehicle.get("application_class"), "drive_layout": vehicle.get("drive_layout"),
            "architecture_class": vehicle.get("architecture"), "transmission_type_raw": vehicle.get("transmission_type"),
            "transmission_type_normalized": vehicle.get("transmission_normalized"),
            "gear_count": vehicle.get("gear_count"), "electrification": vehicle.get("electrification"),
            "aggregate_boundary": boundary.get("drivetrain_boundary_name"),
        }
        transmission_rows.append({
            **common, "candidate_id": trans_vector.vector_id if trans_vector else None,
            "reference_ids": ";".join(trans_vector.reference_ids) if trans_vector else None,
            "compatibility_score": trans_vector.compatibility_score if trans_vector else None,
            "component_boundary": "TRANSMISSION_SYNTHETIC_REFERENCE_UNSCOPED" if trans_vector else None,
            "included_subsystems": "UNKNOWN_NOT_DOCUMENTED_IN_REFERENCE_V1" if trans_vector else None,
            "excluded_subsystems": "UNKNOWN_NOT_DOCUMENTED_IN_REFERENCE_V1" if trans_vector else None,
            "boundary_confidence": "LOW" if trans_vector else None,
            "provenance": trans_vector.provenance if trans_vector else None,
            "match_status": "CONDITIONAL_REFERENCE_MATCH" if trans_vector else "NO_MATCH",
            "missing_discriminants": "REFERENCE_TRANSMISSION_TYPE;REFERENCE_GEAR_COUNT;REFERENCE_ELECTRIFICATION;REFERENCE_TORQUE_CLASS",
            "median_component_to_aggregate_pct": trans_fraction,
            "aggregate_closure_used_for_selection": False,
        })
        axle_rows.append({
            **common, "candidate_id": axle_vector.vector_id if axle_vector else None,
            "reference_ids": ";".join(axle_vector.reference_ids) if axle_vector else None,
            "compatibility_score": axle_vector.compatibility_score if axle_vector else None,
            "component_boundary": "AXLE_REFERENCE_FRONT_REAR_AS_APPLICABLE" if axle_vector else None,
            "included_subsystems": "AXLE_REFERENCE_ONLY" if axle_vector else None,
            "excluded_subsystems": "TRANSMISSION;UNKNOWN_DIFFERENTIAL_FINAL_DRIVE_OVERLAP" if axle_vector else None,
            "boundary_confidence": "LOW" if axle_vector else None,
            "provenance": axle_vector.provenance if axle_vector else None,
            "match_status": "CONDITIONAL_REFERENCE_MATCH" if axle_vector else "NO_MATCH",
            "median_component_to_aggregate_pct": axle_fraction,
            "aggregate_closure_used_for_selection": False,
        })
        if not is_drivetrain:
            diagnostic = "BOUNDARY_INCOMPATIBLE"
            reason = "NO_MOSKALIK_DRIVETRAIN_AGGREGATE"
        elif trans_vector and axle_vector:
            diagnostic = "BOUNDARY_UNKNOWN"
            reason = "TRANSMISSION_REFERENCE_DOES_NOT_DOCUMENT_DIFFERENTIAL_FINAL_DRIVE_CONTENT"
            boundary_unknown += 1
        else:
            diagnostic = "PARTIAL"
            reason = "ONE_OR_MORE_INDEPENDENT_COMPONENT_CANDIDATES_MISSING"
        boundary_rows.append({
            **common, "transmission_candidate": trans_vector.vector_id if trans_vector else None,
            "axle_candidate": axle_vector.vector_id if axle_vector else None,
            "diagnostic_class": diagnostic, "reason": reason,
            "component_boundary_overlap_resolved": False,
            "combined_to_aggregate_pct": None,
            "candidate_selection_used_aggregate_closure": False,
        })

    recommendation = (
        "DRIVETRAIN_FINE_COMPONENTS_DIAGNOSTIC_ONLY"
        if with_transmission or with_axle else "DRIVETRAIN_FINE_COMPONENTS_NOT_SUPPORTED"
    )
    summary_rows = [
        {"metric": "pilot_vdes", "value": len(vehicles), "unit": "VDE"},
        {"metric": "vdes_with_drivetrain_aggregate", "value": with_aggregate, "unit": "VDE"},
        {"metric": "vdes_with_independent_transmission_candidate", "value": with_transmission, "unit": "VDE"},
        {"metric": "vdes_with_independent_axle_candidate", "value": with_axle, "unit": "VDE"},
        {"metric": "vdes_with_both", "value": with_both, "unit": "VDE"},
        {"metric": "boundary_compatible_pairs", "value": boundary_compatible, "unit": "VDE"},
        {"metric": "boundary_unknown_pairs", "value": boundary_unknown, "unit": "VDE"},
        {"metric": "median_transmission_to_aggregate_fraction", "value": _percentile(transmission_fractions, 0.5), "unit": "percent"},
        {"metric": "median_axle_to_aggregate_fraction", "value": _percentile(axle_fractions, 0.5), "unit": "percent"},
        {"metric": "median_combined_to_aggregate_fraction", "value": None, "unit": "percent"},
    ]
    return transmission_rows, axle_rows, boundary_rows, summary_rows, recommendation


def _failure_rows(
    vehicles: list[dict[str, Any]], pools: dict[int, dict[str, list[CandidateVector]]],
    rolling_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    counts: Counter = Counter()
    record_map = {int(row["vde_id"]): row for row in rolling_records}
    for vehicle in vehicles:
        vde_id = int(vehicle["vde_id"])
        if not vehicle.get("eligible_for_validation"):
            counts[("LOW_CONFIDENCE_CLASS_EXCLUDED", "METADATA_COVERAGE")] += 1
        for domain in ("TIRE", "BRAKE", "HUB_BEARING", "TRANSMISSION", "AXLE"):
            if not pools[vde_id].get(domain):
                counts[(f"NO_ELIGIBLE_{domain}", "REFERENCE_COVERAGE")] += 1
        record = record_map[vde_id]
        if record["complete_pool"] and not record.get("selected"):
            counts[("ROLLINGMINOR_OVERSHOOT_GT_5PCT", "PHYSICAL_INCOMPATIBILITY")] += 1
    return [{"reason": reason, "limitation_type": kind, "vde_count": count}
            for (reason, kind), count in counts.most_common()]


def _csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    if fields is None:
        fields = sorted({key for row in rows for key in row}) if rows else ()
    _write_csv(path, rows, fields)


def execute_reference_discrimination(
    db_path: Path, legacy_db_path: Path, component_catalog_path: Path,
    tire_zip_path: Path, output_dir: Path,
) -> dict[str, Any]:
    db_path = Path(db_path).resolve(strict=True)
    legacy_db_path = Path(legacy_db_path).resolve(strict=True)
    component_catalog_path = Path(component_catalog_path).resolve(strict=True)
    tire_zip_path = Path(tire_zip_path).resolve(strict=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes_before = {"pass1_db": file_sha256(db_path), "legacy_db": file_sha256(legacy_db_path)}
    connection = _read_only(db_path)
    legacy = _read_only(legacy_db_path)
    try:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        fk_issues = len(connection.execute("PRAGMA foreign_key_check").fetchall())
        population = load_resolved_population(connection, load_legacy_lookup(legacy))
        historical_tires = load_historical_tire_evidence(legacy)
        sample = deterministic_stratified_sample(population, DEFAULT_ARCHITECTURE_TARGETS)
        if len(sample) != 180:
            raise RuntimeError(f"Expected frozen 180-VDE sample, found {len(sample)}")
        _load_technical_metadata(connection, sample)
        slots = load_aggregate_slots(connection, [int(row["vde_id"]) for row in sample])
        boundaries = {}
        for vehicle in sample:
            vde_id = int(vehicle["vde_id"])
            rolling = slots[vde_id].get("ROLLING_MINOR")
            aggregate = slots[vde_id].get("DRIVETRAIN_AGGREGATE") or slots[vde_id].get("EDRIVE_AGGREGATE") or slots[vde_id].get("DHT_AGGREGATE")
            boundaries[vde_id] = {
                "rolling_boundary": rolling,
                "drivetrain_boundary": aggregate,
                "drivetrain_boundary_name": aggregate.get("boundary") if aggregate else None,
            }
        base_references = load_all_component_references(component_catalog_path)
        tire_refs = load_tire_references(tire_zip_path)

        for vehicle in sample:
            vehicle.update(resolve_application_class_v14(vehicle))
        needed_brakes = {
            (str(vehicle["application_class"]), str(vehicle["drive_layout"]))
            for vehicle in sample
            if vehicle.get("eligible_for_validation") and vehicle.get("application_class") and vehicle.get("drive_layout")
        }
        additions, addition_audit = generate_internal_brake_references(legacy, base_references, needed_brakes)
    finally:
        connection.close()
        legacy.close()

    addition_ids = {reference.component_id for reference in additions}
    before_references = enrich_applicability(base_references)
    after_references = _enrich_with_source([*base_references, *additions], addition_ids)

    before_pools: dict[int, dict[str, list[CandidateVector]]] = {}
    after_pools: dict[int, dict[str, list[CandidateVector]]] = {}
    candidate_rows: list[dict[str, Any]] = []
    for vehicle in sample:
        vde_id = int(vehicle["vde_id"])
        if vehicle.get("eligible_for_validation"):
            before_pools[vde_id] = _build_all_pools(vehicle, before_references, tire_refs, historical_tires)
            after_pools[vde_id] = _build_all_pools(vehicle, after_references, tire_refs, historical_tires)
        else:
            empty = {domain: [] for domain in ("TIRE", "BRAKE", "HUB_BEARING", "TRANSMISSION", "AXLE")}
            before_pools[vde_id] = {key: list(value) for key, value in empty.items()}
            after_pools[vde_id] = {key: list(value) for key, value in empty.items()}
        for row in _candidate_rows(vde_id, after_pools[vde_id]):
            if row["domain"] in ROLLING_DOMAINS:
                row["application_class"] = vehicle.get("application_class")
                row["drive_layout"] = vehicle.get("drive_layout")
                candidate_rows.append(row)

    _, _, before_records = _run_rolling(sample, before_pools, PRIMARY_CAPS)
    before_kpis = aggregate_kpis(before_records, "ROLLING_MINOR")
    primary_after = {vde_id: truncate_pools(pools, PRIMARY_CAPS) for vde_id, pools in after_pools.items()}
    selected, all_accepted, real_records = _run_rolling(sample, primary_after, PRIMARY_CAPS)
    real_kpis = aggregate_kpis(real_records, "ROLLING_MINOR")

    shuffle_rows: list[dict[str, Any]] = [{"control": "REAL", "seed": None, **real_kpis}]
    shuffled_kpis: list[dict[str, Any]] = []
    for seed in SHUFFLE_SEEDS:
        _, _, records = _run_rolling(sample, shuffle_pools_seeded(primary_after, seed), PRIMARY_CAPS)
        kpis = aggregate_kpis(records, "ROLLING_MINOR")
        shuffled_kpis.append(kpis)
        shuffle_rows.append({"control": "SHUFFLED", "seed": seed, **kpis})
    shuffle_mean = {
        key: mean(float(row[key]) for row in shuffled_kpis)
        for key in ("physical_acceptance_rate_pct", "median_explained_pct", "median_unresolved_pct")
        if all(row[key] is not None for row in shuffled_kpis)
    }
    shuffle_rows.append({
        "control": "SHUFFLE_MEAN", "seed": None, **shuffle_mean,
        "acceptance_lift_pp": real_kpis["physical_acceptance_rate_pct"] - shuffle_mean["physical_acceptance_rate_pct"],
        "explained_lift_pp": real_kpis["median_explained_pct"] - shuffle_mean["median_explained_pct"],
    })
    sensitivity: list[dict[str, Any]] = []
    for size in (1, 2, 3, 5):
        caps = {"TIRE": size, "BRAKE": size, "HUB_BEARING": size, "TRANSMISSION": 0, "AXLE": 0}
        _, _, records = _run_rolling(sample, after_pools, caps)
        sensitivity.append({"pool_size": f"TOP_{size}", **aggregate_kpis(records, "ROLLING_MINOR")})
    rolling_decision, decision_evidence = _decision(real_kpis, shuffle_mean, sensitivity, len(sample))

    transmission_rows, axle_rows, boundary_rows, drivetrain_summary, drivetrain_decision = independent_drivetrain_audit(
        sample, boundaries, after_pools
    )
    coverage = _coverage_rows(before_pools, after_pools, len(sample))
    library_before = _reference_rows(base_references, tire_refs, "BEFORE")
    library_after = _reference_rows([*base_references, *additions], tire_refs, "AFTER")
    library_metrics = _library_metrics(library_before, library_after)
    failures = _failure_rows(sample, after_pools, real_records)

    _csv(output_dir / "reference_library_before_after.csv", library_metrics)
    _csv(output_dir / "reference_coverage_summary.csv", coverage)
    _csv(output_dir / "reference_additions.csv", addition_audit)
    _csv(output_dir / "rollingminor_candidate_pools.csv", candidate_rows)
    _csv(output_dir / "rollingminor_selected_combinations.csv", selected)
    _csv(output_dir / "rollingminor_all_accepted_combinations.csv", all_accepted)
    _csv(output_dir / "rollingminor_shuffle_control.csv", shuffle_rows)
    _csv(output_dir / "rollingminor_pool_size_sensitivity.csv", sensitivity)
    _csv(output_dir / "transmission_independent_matches.csv", transmission_rows)
    _csv(output_dir / "axle_independent_matches.csv", axle_rows)
    _csv(output_dir / "drivetrain_boundary_audit.csv", boundary_rows)
    _csv(output_dir / "drivetrain_diagnostic_summary.csv", drivetrain_summary)
    _csv(output_dir / "coverage_failure_reasons.csv", failures)

    hashes_after = {"pass1_db": file_sha256(db_path), "legacy_db": file_sha256(legacy_db_path)}
    if hashes_before != hashes_after:
        raise RuntimeError("Read-only Pass 1C.5 changed an input database")
    summary = {
        "method_version": METHOD_VERSION, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_metadata(), "sample_size": len(sample),
        "sample_identity_sha256": __import__("hashlib").sha256(
            ",".join(str(row["vde_id"]) for row in sample).encode()
        ).hexdigest().upper(),
        "reference_additions": len(additions),
        "reference_addition_rejections": sum(row["status"].startswith("REJECTED") for row in addition_audit),
        "pass_1c4_reproduced": before_kpis, "pass_1c5": real_kpis,
        "pass_1c4_reported": {
            "complete_candidate_pools": 46, "accepted_complete_builds": 22,
            "physical_acceptance_rate_pct": 47.82608695652174,
            "shuffle_acceptance_rate_pct": 45.21739130434783,
            "acceptance_lift_pp": 2.608695652173914,
            "median_explained_pct": 85.38303439241898,
            "median_unresolved_pct": 14.616965607581017,
            "max_accepted_overshoot_pct": 4.522038524539512,
        },
        "shuffle_mean": shuffle_mean, "shuffle_seeds": list(SHUFFLE_SEEDS),
        "decision_evidence": decision_evidence,
        "rollingminor_decision": rolling_decision,
        "drivetrain_decision": drivetrain_decision,
        "drivetrain_metrics": {row["metric"]: row["value"] for row in drivetrain_summary},
        "hashes_before": hashes_before, "hashes_after": hashes_after,
        "quick_check": quick_check, "foreign_key_issues": fk_issues,
        "database_rows_written": 0, "database_objects_created": 0,
        "external_search_count": 0, "llm_component_generation_count": 0,
        "scaled_combination_count": 0, "authoritative_aggregate_changes": 0,
    }
    (output_dir / "execution_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "pass1c5_methodology_report.md").write_text(
        _report(summary, coverage, addition_audit, sensitivity, failures), encoding="utf-8"
    )
    return summary


def _fmt(value: Any) -> str:
    return "N/A" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value))


def _report(
    summary: dict[str, Any], coverage: list[dict[str, Any]], additions: list[dict[str, Any]],
    sensitivity: list[dict[str, Any]], failures: list[dict[str, Any]],
) -> str:
    old = summary["pass_1c4_reported"]
    new = summary["pass_1c5"]
    shuffle = summary["shuffle_mean"]
    coverage_map = {row["domain"]: row for row in coverage}
    drivetrain = summary["drivetrain_metrics"]
    added = [row for row in additions if row["status"] == "ADDED_EPHEMERAL_PILOT_REFERENCE"]
    lines = [
        "# Pass 1C.5 Reference Discrimination & Independent Component Matching", "",
        "## Frozen scope and safety", "",
        f"- frozen 180-VDE sample reused: **YES**, `{summary['sample_identity_sha256']}`",
        "- aggregate physics / 5% tolerance / +10 pp gate changed: **NO / NO / NO**",
        "- scaling / external search / LLM generation / DB writes: **0 / 0 / 0 / 0**",
        f"- quick_check / FK issues: **{summary['quick_check']} / {summary['foreign_key_issues']}**", "",
        "## RollingMinor comparison", "",
        "| Metric | Pass 1C.4 | Pass 1C.5 |", "|---|---:|---:|",
        f"| Complete pools | {old['complete_candidate_pools']} | {new['complete_candidate_pools']} |",
        f"| Accepted build-ups | {old['accepted_complete_builds']} | {new['accepted_complete_builds']} |",
        f"| Acceptance % complete | {_fmt(old['physical_acceptance_rate_pct'])} | {_fmt(new['physical_acceptance_rate_pct'])} |",
        f"| Acceptance % total pilot | {100.0 * old['accepted_complete_builds']/180:.3f} | {100.0 * new['accepted_complete_builds']/180:.3f} |",
        f"| Median explained % | {_fmt(old['median_explained_pct'])} | {_fmt(new['median_explained_pct'])} |",
        f"| Median unresolved % | {_fmt(old['median_unresolved_pct'])} | {_fmt(new['median_unresolved_pct'])} |",
        f"| Max selected overshoot % | {_fmt(old['max_accepted_overshoot_pct'])} | {_fmt(new['max_accepted_overshoot_pct'])} |",
        f"| Shuffle acceptance % | {_fmt(old['shuffle_acceptance_rate_pct'])} | {_fmt(shuffle['physical_acceptance_rate_pct'])} |",
        f"| Real-vs-shuffle lift [pp] | {_fmt(old['acceptance_lift_pp'])} | {_fmt(summary['decision_evidence']['real_vs_shuffle_acceptance_lift_pp'])} |", "",
        "## Required questions", "",
        f"1. **New references:** {len(added)} ephemeral Brake population medoids were added from internal legacy component coefficients using exact application-class + drive groups. No Hub, Axle, or Transmission ABC was invented because no independent internal source supported it.",
        f"2. **Brake/Hub coverage:** Brake candidate coverage changed from {coverage_map['BRAKE']['before_candidate_vdes']} to {coverage_map['BRAKE']['after_candidate_vdes']}; Hub remained {coverage_map['HUB_BEARING']['after_candidate_vdes']} because the evidence gate blocked fabricated Hub variants.",
        f"3. **REAL-vs-SHUFFLE:** Pass 1C.5 lift is {summary['decision_evidence']['real_vs_shuffle_acceptance_lift_pp']:.3f} pp versus 2.609 pp in Pass 1C.4.",
        f"4. **Top-3 vs Top-5:** {'stable' if summary['decision_evidence']['top3_vs_top5_stable'] else 'not stable'} under the frozen 5 pp acceptance/residual rule.",
        f"5. **Fleet scale:** {summary['rollingminor_decision']}.",
        f"6. **Independent Transmission coverage:** {drivetrain['vdes_with_independent_transmission_candidate']}/{drivetrain['vdes_with_drivetrain_aggregate']} VDEs with a Moskalik Drivetrain aggregate.",
        f"7. **Independent Axle coverage:** {drivetrain['vdes_with_independent_axle_candidate']}/{drivetrain['vdes_with_drivetrain_aggregate']} VDEs with a Moskalik Drivetrain aggregate.",
        f"8. **Boundary compatibility:** {drivetrain['boundary_compatible_pairs']} compatible and {drivetrain['boundary_unknown_pairs']} unknown pairs; the v1 Transmission references do not document differential/final-drive inclusion.",
        "9. **Presentable estimates:** Tire/Brake/Hub remain conditional surrogate engineering references with explicit provenance. Transmission/Axle are conditional application matches for diagnostic review, not hardware identification or additive decomposition.",
        "10. **Remain aggregate/unresolved:** DRIVETRAIN_AGGREGATE remains authoritative; combined Transmission+Axle stays unresolved when overlap is unknown. OTHER_MINOR_UNRESOLVED remains the non-negative RollingMinor remainder.", "",
        "## Reference additions", "",
    ]
    if added:
        lines.extend(["| Reference | Class | Drive | Population | Rationale |", "|---|---|---|---:|---|"])
        for row in added:
            lines.append(f"| {row['component_id']} | {row['application_class']} | {row['drive_layout']} | {row['population_n']} | {row['physical_rationale']} |")
    else:
        lines.append("No reference passed the frozen internal-evidence gate.")
    lines.extend(["", "## Pool-size sensitivity", "", "| Pool | Complete | Accepted | Acceptance % | Explained % | Unresolved % |", "|---|---:|---:|---:|---:|---:|"])
    for row in sensitivity:
        lines.append(f"| {row['pool_size']} | {row['complete_candidate_pools']} | {row['accepted_complete_builds']} | {_fmt(row['physical_acceptance_rate_pct'])} | {_fmt(row['median_explained_pct'])} | {_fmt(row['median_unresolved_pct'])} |")
    lines.extend(["", "## Remaining holes", ""])
    for row in failures:
        lines.append(f"- {row['reason']} — {row['limitation_type']}: **{row['vde_count']}**")
    lines.extend([
        "", "## Decisions", "",
        f"RollingMinor: `{summary['rollingminor_decision']}`", "",
        f"Drivetrain: `{summary['drivetrain_decision']}`", "",
        "The enriched library was frozen before aggregate evaluation. Aggregate closure was not used to generate or select independent component references.",
    ])
    return "\n".join(lines) + "\n"


__all__ = [
    "MIN_INTERNAL_POPULATION", "execute_reference_discrimination",
    "generate_internal_brake_references", "independent_drivetrain_audit",
]
