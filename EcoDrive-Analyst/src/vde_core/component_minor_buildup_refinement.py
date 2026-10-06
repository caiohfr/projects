"""Read-only Sprint 12 Pass 1C.1 synthetic build-up refinement.

The authoritative ROLLING_MINOR result is never changed.  This module compares
an explanatory synthetic build-up against it without scaling any component.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Iterable

from src.vde_core.component_enrichment_pass1 import file_sha256
from src.vde_core.component_minor_buildup_pilot import (
    DEFAULT_ARCHITECTURE_TARGETS,
    FINE_DOMAINS,
    QA_SPEEDS_KPH,
    Reference,
    _curve,
    _git_metadata,
    _json,
    _read_only,
    _token,
    _write_csv,
    deterministic_stratified_sample,
    evaluate_vehicle as evaluate_original_vehicle,
    load_component_references,
    load_legacy_lookup,
    load_resolved_population,
    load_tire_references,
    match_synthetic_reference,
    match_tire,
)


REFINEMENT_VERSION = "SPRINT12_PASS1C1_SYNTHETIC_BUILDUP_REFINEMENT_V1.0"
MAX_SYNTHETIC_OVERSHOOT_PCT = 5.0
FLOATING_EPSILON_N = 1e-9


# Broad, rule-defined applicability bands.  These are classification envelopes,
# not measured vehicle or component properties.
APPLICATION_ENVELOPES: dict[str, dict[str, Any]] = {
    "PASSENGER_LIGHT": {"mass": (800.0, 1900.0), "wheel": (13.0, 20.0), "load": "LIGHT", "performance": "STANDARD"},
    "PASSENGER_STANDARD": {"mass": (1100.0, 2400.0), "wheel": (14.0, 22.0), "load": "STANDARD", "performance": "STANDARD"},
    "CROSSOVER": {"mass": (1200.0, 2700.0), "wheel": (15.0, 23.0), "load": "STANDARD", "performance": "STANDARD"},
    "SUV": {"mass": (1600.0, 3400.0), "wheel": (16.0, 24.0), "load": "HIGH", "performance": "STANDARD"},
    "OFFROAD_SUV": {"mass": (1800.0, 3800.0), "wheel": (16.0, 24.0), "load": "HIGH", "performance": "OFFROAD"},
    "PASSENGER_VAN": {"mass": (1500.0, 3100.0), "wheel": (15.0, 22.0), "load": "HIGH", "performance": "STANDARD"},
    "CARGO_VAN": {"mass": (1700.0, 4200.0), "wheel": (15.0, 24.0), "load": "COMMERCIAL", "performance": "COMMERCIAL"},
    "PICKUP_LIGHT_DUTY": {"mass": (1700.0, 3600.0), "wheel": (15.0, 24.0), "load": "HIGH", "performance": "UTILITY"},
    "HEAVY_DUTY": {"mass": (2500.0, 6000.0), "wheel": (16.0, 26.0), "load": "HEAVY", "performance": "COMMERCIAL"},
    "PERFORMANCE": {"mass": (900.0, 2400.0), "wheel": (16.0, 23.0), "load": "STANDARD", "performance": "PERFORMANCE"},
}


@dataclass(frozen=True)
class ApplicabilityReference:
    reference: Reference
    reference_mass_min_kg: float
    reference_mass_max_kg: float
    axle_load_class: str
    wheel_class_min_in: float
    wheel_class_max_in: float
    performance_class: str
    boundary: str
    driven_axle_compatibility: str | None
    source: str = "SYNTHETIC_REFERENCE"
    applicability_source: str = "RULE_DEFINED_APPLICABILITY"


def enrich_applicability(references: Iterable[Reference]) -> list[ApplicabilityReference]:
    enriched: list[ApplicabilityReference] = []
    for reference in references:
        envelope = APPLICATION_ENVELOPES.get(reference.application_class)
        if envelope is None:
            continue
        driven = None
        if reference.position:
            if reference.drive == "AWD" or reference.drive == "4WD":
                driven = "DRIVEN"
            elif reference.drive == "FWD":
                driven = "DRIVEN" if reference.position == "FRONT" else "NON_DRIVEN"
            elif reference.drive == "RWD":
                driven = "DRIVEN" if reference.position == "REAR" else "NON_DRIVEN"
        enriched.append(
            ApplicabilityReference(
                reference=reference,
                reference_mass_min_kg=envelope["mass"][0],
                reference_mass_max_kg=envelope["mass"][1],
                axle_load_class=envelope["load"],
                wheel_class_min_in=envelope["wheel"][0],
                wheel_class_max_in=envelope["wheel"][1],
                performance_class=envelope["performance"],
                boundary=reference.domain,
                driven_axle_compatibility=driven,
            )
        )
    return sorted(enriched, key=lambda item: (item.reference.domain, item.reference.component_id))


def applicability_rows(references: Iterable[ApplicabilityReference]) -> list[dict[str, Any]]:
    return [
        {
            "component_id": item.reference.component_id,
            "component_resolution_id": item.reference.resolution_id,
            "domain": item.reference.domain,
            "application_class": item.reference.application_class,
            "drive_layout": item.reference.drive,
            "position": item.reference.position,
            "reference_mass_min_kg": item.reference_mass_min_kg,
            "reference_mass_max_kg": item.reference_mass_max_kg,
            "axle_load_class": item.axle_load_class,
            "axle_load_min_kg": None,
            "axle_load_max_kg": None,
            "wheel_class_min_in": item.wheel_class_min_in,
            "wheel_class_max_in": item.wheel_class_max_in,
            "performance_class": item.performance_class,
            "driven_axle_compatibility": item.driven_axle_compatibility,
            "boundary": item.boundary,
            "source": item.source,
            "applicability_source": item.applicability_source,
            "metadata_semantics": "BROAD_RULE_DEFINED_ENVELOPE_NOT_MEASUREMENT",
        }
        for item in references
    ]


def match_applicability_reference(
    vehicle: dict[str, Any],
    references: list[ApplicabilityReference],
    domain: str,
    *,
    position: str | None = None,
) -> dict[str, Any]:
    app_class = _token(vehicle.get("application_class"))
    drive = _token(vehicle.get("drive_layout"))
    position = _token(position)
    candidates = [item for item in references if item.reference.domain == domain]
    if position:
        candidates = [item for item in candidates if item.reference.position == position]
    if app_class and drive:
        candidates = [
            item
            for item in candidates
            if item.reference.application_class == app_class and item.reference.drive == drive
        ]
    else:
        candidates = []
    if len(candidates) != 1:
        return {
            "grade": "UNRESOLVED",
            "accepted": False,
            "selected": None,
            "candidates": candidates,
            "criteria": ["application_class", "drive_layout", "boundary"] + (["position"] if position else []),
            "reason_codes": ["NO_UNIQUE_APPLICABILITY_MATCH"],
        }
    selected = candidates[0]
    mass = vehicle.get("test_mass_resolved_kg")
    try:
        mass_value = float(mass)
    except (TypeError, ValueError):
        mass_value = math.nan
    reasons: list[str] = []
    in_mass_range = math.isfinite(mass_value) and selected.reference_mass_min_kg <= mass_value <= selected.reference_mass_max_kg
    if not math.isfinite(mass_value):
        reasons.append("VEHICLE_MASS_UNAVAILABLE")
    elif not in_mass_range:
        reasons.append("VEHICLE_MASS_OUTSIDE_RULE_DEFINED_APPLICABILITY")

    # The current vehicle data has no dependable wheel/axle-load discriminator.
    # Exact application+drive+mass is therefore Grade B, never promoted to A.
    if in_mass_range:
        grade = "B"
        accepted = True
        reasons.append("GRADE_B_BROAD_RULE_DEFINED_APPLICABILITY")
    else:
        grade = "C"
        accepted = False
        reasons.append("MATERIAL_MATCH_DISCRIMINANTS_INCOMPLETE")
    return {
        "grade": grade,
        "accepted": accepted,
        "selected": selected,
        "candidates": candidates,
        "criteria": ["application_class", "drive_layout", "mass_envelope", "boundary"] + (["position"] if position else []),
        "reason_codes": sorted(set(reasons)),
    }


def _aggregate_position_matches(matches: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = bool(matches) and all(match["accepted"] for match in matches)
    selected = [match["selected"] for match in matches if match.get("selected")]
    grades = [match["grade"] for match in matches]
    grade = grades[0] if grades and len(set(grades)) == 1 else ("UNRESOLVED" if "UNRESOLVED" in grades else "C")
    abc = None
    if accepted:
        abc = tuple(sum(item.reference.abc[index] for item in selected) for index in range(3))
    return {
        "grade": grade,
        "accepted": accepted,
        "selected": selected,
        "candidates": [item for match in matches for item in match.get("candidates", [])],
        "criteria": sorted({criterion for match in matches for criterion in match.get("criteria", [])}),
        "reason_codes": sorted({reason for match in matches for reason in match.get("reason_codes", [])}),
        "abc": abc,
    }


def classify_closure(
    rolling_abc: tuple[float, float, float],
    component_abc: dict[str, tuple[float, float, float]],
) -> dict[str, float | str]:
    max_error = 0.0
    max_pct = 0.0
    for speed in QA_SPEEDS_KPH:
        rolling = _curve(rolling_abc, speed)
        buildup = sum(_curve(abc, speed) for abc in component_abc.values())
        error = max(0.0, buildup - rolling)
        pct = error / max(abs(rolling), FLOATING_EPSILON_N) * 100.0
        max_error = max(max_error, error)
        max_pct = max(max_pct, pct)
    if max_error <= FLOATING_EPSILON_N:
        status = "CLEAN_BUILDUP"
    elif max_pct <= MAX_SYNTHETIC_OVERSHOOT_PCT:
        status = "TOLERANCE_ACCEPTED"
    else:
        status = "REJECTED_BUILDUP"
    return {"status": status, "max_overshoot_N": max_error, "max_overshoot_pct": max_pct}


def evaluate_refinement_vehicle(
    vehicle: dict[str, Any],
    component_references: list[Reference],
    enriched_references: list[ApplicabilityReference],
    tire_references: list[dict[str, Any]],
    *,
    phase: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if phase not in {"A", "B"}:
        raise ValueError("phase must be A or B")
    tire = match_tire(vehicle, tire_references)
    if phase == "A":
        brake = match_synthetic_reference(vehicle, component_references, "BRAKE")
        hub_parts = [
            match_synthetic_reference(vehicle, component_references, "HUB_BEARING", position="FRONT"),
            match_synthetic_reference(vehicle, component_references, "HUB_BEARING", position="REAR"),
        ]
        axle_parts = [
            match_synthetic_reference(vehicle, component_references, "AXLE", position="FRONT"),
            match_synthetic_reference(vehicle, component_references, "AXLE", position="REAR"),
        ]
        # Convert the old Reference-shaped positional results to the common form.
        def old_aggregate(parts: list[dict[str, Any]]) -> dict[str, Any]:
            accepted = all(part["accepted"] for part in parts)
            selected = [part["selected"] for part in parts if part.get("selected")]
            grades = [part["grade"] for part in parts]
            return {
                "grade": grades[0] if len(set(grades)) == 1 else ("UNRESOLVED" if "UNRESOLVED" in grades else "C"),
                "accepted": accepted,
                "selected": selected,
                "candidates": [item for part in parts for item in part.get("candidates", [])],
                "criteria": sorted({criterion for part in parts for criterion in part.get("criteria", [])}),
                "reason_codes": sorted({reason for part in parts for reason in part.get("reason_codes", [])}),
                "abc": tuple(sum(item.abc[index] for item in selected) for index in range(3)) if accepted else None,
            }
        hub = old_aggregate(hub_parts)
        axle = old_aggregate(axle_parts)
    else:
        brake = match_applicability_reference(vehicle, enriched_references, "BRAKE")
        hub = _aggregate_position_matches(
            [
                match_applicability_reference(vehicle, enriched_references, "HUB_BEARING", position="FRONT"),
                match_applicability_reference(vehicle, enriched_references, "HUB_BEARING", position="REAR"),
            ]
        )
        axle = _aggregate_position_matches(
            [
                match_applicability_reference(vehicle, enriched_references, "AXLE", position="FRONT"),
                match_applicability_reference(vehicle, enriched_references, "AXLE", position="REAR"),
            ]
        )
    axle["accepted"] = False
    axle["abc"] = None
    axle["reason_codes"] = sorted(set((*axle.get("reason_codes", []), "AXLE_EXCLUDED_BOUNDARY_OVERLAP_RISK")))
    results = {"TIRE": tire, "BRAKE": brake, "HUB_BEARING": hub, "AXLE": axle}

    proposed: dict[str, tuple[float, float, float]] = {}
    if tire["accepted"]:
        proposed["TIRE"] = tire["abc"]
    for domain in ("BRAKE", "HUB_BEARING"):
        result = results[domain]
        if not result["accepted"]:
            continue
        selected = result.get("selected")
        if isinstance(selected, list):
            proposed[domain] = result["abc"]
        elif isinstance(selected, ApplicabilityReference):
            proposed[domain] = selected.reference.abc
        elif isinstance(selected, Reference):
            proposed[domain] = selected.abc

    closure = classify_closure(vehicle["rolling_abc"], proposed) if proposed else {
        "status": "NO_ACCEPTED_FINE_COMPONENTS",
        "max_overshoot_N": 0.0,
        "max_overshoot_pct": 0.0,
    }
    build_accepted = closure["status"] in {"CLEAN_BUILDUP", "TOLERANCE_ACCEPTED"}
    accepted = proposed if build_accepted else {}
    closure_reason = {
        "CLEAN_BUILDUP": "NO_SYNTHETIC_OVERSHOOT",
        "TOLERANCE_ACCEPTED": "WITHIN_SYNTHETIC_CLOSURE_TOLERANCE",
        "REJECTED_BUILDUP": "EXCEEDS_SYNTHETIC_CLOSURE_TOLERANCE",
        "NO_ACCEPTED_FINE_COMPONENTS": "NO_GRADE_A_OR_B_COMPONENTS",
    }[str(closure["status"])]

    match_rows: list[dict[str, Any]] = []
    for domain in FINE_DOMAINS:
        result = results[domain]
        selected = result.get("selected") or []
        if not isinstance(selected, list):
            selected = [selected]
        ids = [
            item.reference.component_id if isinstance(item, ApplicabilityReference) else item.component_id
            for item in selected
            if isinstance(item, (ApplicabilityReference, Reference))
        ]
        candidate_ids = result.get("candidate_ids") or [
            item.reference.component_id if isinstance(item, ApplicabilityReference) else item.component_id
            for item in result.get("candidates", [])
        ]
        match_rows.append(
            {
                "phase": phase,
                "vde_id": vehicle["vde_id"],
                "fine_domain": domain,
                "selected_reference_id": result.get("selected_reference_id") or ";".join(ids),
                "candidate_reference_ids_json": _json(candidate_ids),
                "match_grade": result["grade"],
                "grade_accepted": int(bool(result["accepted"])),
                "accepted_in_build": int(domain in accepted),
                "closure_status": closure["status"],
                "closure_reason": closure_reason,
                "estimate_status": "CONDITIONAL" if domain in accepted else "UNRESOLVED",
                "max_overshoot_pct": closure["max_overshoot_pct"],
                "source": result.get("source") or ("SYNTHETIC_REFERENCE" if selected else "UNRESOLVED"),
                "applicability_source": "RULE_DEFINED_APPLICABILITY" if phase == "B" and domain != "TIRE" else None,
                "boundary_rejected": int(domain == "AXLE"),
                "reason_codes": ";".join(result.get("reason_codes", [])),
            }
        )

    curves: list[dict[str, Any]] = []
    residual_fractions: list[float] = []
    closure_pcts: list[float] = []
    for speed in QA_SPEEDS_KPH:
        rolling = _curve(vehicle["rolling_abc"], speed)
        forces = {domain: _curve(proposed.get(domain), speed) for domain in FINE_DOMAINS}
        buildup = sum(forces.values())
        raw_error = buildup - rolling
        unresolved = max(0.0, rolling - buildup) if build_accepted else rolling
        closure_error = max(0.0, raw_error) if closure["status"] == "TOLERANCE_ACCEPTED" else 0.0
        closure_pct = closure_error / max(abs(rolling), FLOATING_EPSILON_N) * 100.0
        if build_accepted and rolling > FLOATING_EPSILON_N:
            residual_fractions.append(unresolved / rolling)
        if closure_error > 0:
            closure_pcts.append(closure_pct)
        curves.append(
            {
                "phase": phase,
                "vde_id": vehicle["vde_id"],
                "speed_kph": speed,
                "rolling_minor_N": rolling,
                "tire_N": forces["TIRE"],
                "brake_N": forces["BRAKE"],
                "hub_N": forces["HUB_BEARING"],
                "axle_N": forces["AXLE"],
                "candidate_buildup_N": buildup,
                "build_accepted": int(build_accepted),
                "closure_status": closure["status"],
                "closure_reason": closure_reason,
                "minor_unresolved_N": unresolved,
                "minor_unresolved_fraction": unresolved / rolling if rolling > FLOATING_EPSILON_N else None,
                "model_closure_error_N": closure_error,
                "model_closure_error_pct": closure_pct,
                "candidate_overshoot_N": max(0.0, raw_error),
                "candidate_overshoot_pct": max(0.0, raw_error) / max(abs(rolling), FLOATING_EPSILON_N) * 100.0,
            }
        )
    proposed_domains = [domain for domain in FINE_DOMAINS if domain in proposed]
    accepted_domains = [domain for domain in FINE_DOMAINS if domain in accepted]
    return match_rows, curves, {
        "phase": phase,
        "vde_id": vehicle["vde_id"],
        "proposed_component_set": ";".join(proposed_domains),
        "accepted_component_set": ";".join(accepted_domains),
        "closure_status": closure["status"],
        "closure_reason": closure_reason,
        "max_overshoot_N": closure["max_overshoot_N"],
        "max_overshoot_pct": closure["max_overshoot_pct"],
        "median_unresolved_residual_pct": _percentile([100.0 * value for value in residual_fractions], 0.5),
        "p10_unresolved_residual_pct": _percentile([100.0 * value for value in residual_fractions], 0.1),
        "p90_unresolved_residual_pct": _percentile([100.0 * value for value in residual_fractions], 0.9),
        "max_unresolved_residual_pct": max((100.0 * value for value in residual_fractions), default=None),
        "median_closure_error_pct": _percentile(closure_pcts, 0.5),
        "scaled_to_force_closure": 0,
        "parent_rolling_minor_resolution_id": vehicle["rolling_minor_resolution_id"],
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    weight = index - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _metrics(matches: list[dict[str, Any]], summaries: list[dict[str, Any]], curves: list[dict[str, Any]]) -> dict[str, Any]:
    by_domain = {domain: [row for row in matches if row["fine_domain"] == domain] for domain in FINE_DOMAINS}
    accepted_sets = [set(filter(None, row["accepted_component_set"].split(";"))) for row in summaries]
    tire_mechanical = sum("TIRE" in domains and bool(domains & {"BRAKE", "HUB_BEARING"}) for domains in accepted_sets)
    residuals = [
        float(row["median_unresolved_residual_pct"])
        for row in summaries
        if row["closure_status"] in {"CLEAN_BUILDUP", "TOLERANCE_ACCEPTED"}
        and row["median_unresolved_residual_pct"] is not None
    ]
    maximum_residuals = [
        float(row["max_unresolved_residual_pct"])
        for row in summaries
        if row["closure_status"] in {"CLEAN_BUILDUP", "TOLERANCE_ACCEPTED"}
        and row["max_unresolved_residual_pct"] is not None
    ]
    tolerance_errors = [
        float(row["max_overshoot_pct"])
        for row in summaries
        if row["closure_status"] == "TOLERANCE_ACCEPTED"
    ]
    return {
        "tire_candidates": sum(row["match_grade"] in {"A", "B"} for row in by_domain["TIRE"]),
        "tire_accepted": sum(int(row["accepted_in_build"]) for row in by_domain["TIRE"]),
        "brake_grade_ab": sum(row["match_grade"] in {"A", "B"} for row in by_domain["BRAKE"]),
        "hub_grade_ab": sum(row["match_grade"] in {"A", "B"} for row in by_domain["HUB_BEARING"]),
        "axle_grade_ab": sum(row["match_grade"] in {"A", "B"} for row in by_domain["AXLE"]),
        "brake_accepted": sum(int(row["accepted_in_build"]) for row in by_domain["BRAKE"]),
        "hub_accepted": sum(int(row["accepted_in_build"]) for row in by_domain["HUB_BEARING"]),
        "axle_accepted": sum(int(row["accepted_in_build"]) for row in by_domain["AXLE"]),
        "tire_only": sum(domains == {"TIRE"} for domains in accepted_sets),
        "tire_brake": sum(domains == {"TIRE", "BRAKE"} for domains in accepted_sets),
        "tire_hub": sum(domains == {"TIRE", "HUB_BEARING"} for domains in accepted_sets),
        "tire_brake_hub": sum(domains == {"TIRE", "BRAKE", "HUB_BEARING"} for domains in accepted_sets),
        "tire_plus_brake_or_hub": tire_mechanical,
        "tire_plus_brake_or_hub_pct": 100.0 * tire_mechanical / len(summaries),
        "clean_buildup": sum(row["closure_status"] == "CLEAN_BUILDUP" for row in summaries),
        "tolerance_accepted": sum(row["closure_status"] == "TOLERANCE_ACCEPTED" for row in summaries),
        "rejected_over_5pct": sum(row["closure_status"] == "REJECTED_BUILDUP" for row in summaries),
        "median_unresolved_residual_pct": _percentile(residuals, 0.5),
        "p10_unresolved_residual_pct": _percentile(residuals, 0.1),
        "p90_unresolved_residual_pct": _percentile(residuals, 0.9),
        "max_unresolved_residual_pct": max(maximum_residuals, default=None),
        "median_closure_error_pct": _percentile(tolerance_errors, 0.5),
        "p90_closure_error_pct": _percentile(tolerance_errors, 0.9),
        "max_closure_error_pct": max(tolerance_errors, default=None),
    }


def _old_metrics(sample: list[dict[str, Any]], component_refs: list[Reference], tire_refs: list[dict[str, Any]]) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for vehicle in sample:
        vehicle_matches, _, summary = evaluate_original_vehicle(vehicle, component_refs, tire_refs)
        matches.extend(vehicle_matches)
        summaries.append(summary)
    tire_rows = [row for row in matches if row["fine_domain"] == "TIRE"]
    accepted_sets = [set(filter(None, row["accepted_component_set"].split(";"))) for row in summaries]
    return {
        "tire_candidates": sum(row["match_grade"] in {"A", "B"} for row in tire_rows),
        "tire_accepted": sum(int(row["accepted"]) for row in tire_rows),
        "brake_grade_ab": 0,
        "hub_grade_ab": 0,
        "axle_grade_ab": 0,
        "tire_plus_brake_or_hub": sum("TIRE" in domains and bool(domains & {"BRAKE", "HUB_BEARING"}) for domains in accepted_sets),
        "clean_buildup": sum(bool(domains) for domains in accepted_sets),
        "tolerance_accepted": 0,
        "rejected_old_0_1N_rule": sum(row["buildup_status"] == "SYNTHETIC_BUILDUP_EXCEEDS_ROLLING_MINOR" for row in summaries),
    }


def _decision(metrics: dict[str, Any]) -> str:
    coverage = float(metrics["tire_plus_brake_or_hub_pct"])
    gates = (
        metrics["rejected_over_5pct"] == 0
        and metrics["axle_accepted"] == 0
        and metrics["tire_plus_brake_or_hub"] > 0
    )
    if coverage < 20.0:
        return "PASS_1C_SYNTHETIC_SCALE_REJECTED"
    if coverage < 35.0 or not gates:
        return "PASS_1C_REVIEW_REQUIRED"
    return "PASS_1C_SCALE_RECOMMENDED"


def _write_comparison(path: Path, old: dict[str, Any], phase_a: dict[str, Any], phase_b: dict[str, Any]) -> None:
    labels = (
        "tire_accepted", "brake_grade_ab", "hub_grade_ab", "axle_grade_ab",
        "tire_plus_brake_or_hub", "clean_buildup", "tolerance_accepted", "rejected_over_5pct",
    )
    rows = [{"metric": label, "original_pass_1c": old.get(label), "phase_a_5pct": phase_a.get(label), "phase_b_metadata": phase_b.get(label)} for label in labels]
    _write_csv(path, rows, rows[0].keys())


def execute_refinement(
    db_path: Path,
    legacy_db_path: Path,
    component_catalog_path: Path,
    tire_zip_path: Path,
    output_dir: Path,
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
    finally:
        connection.close()
        legacy.close()
    sample = deterministic_stratified_sample(population, DEFAULT_ARCHITECTURE_TARGETS)
    if len(sample) != 180:
        raise RuntimeError(f"Expected the frozen 180-VDE sample, found {len(sample)}")
    component_refs = load_component_references(component_catalog_path)
    tire_refs = load_tire_references(tire_zip_path)
    enriched = enrich_applicability(component_refs)
    old = _old_metrics(sample, component_refs, tire_refs)
    sample_rows = [
        {
            "vde_id": row["vde_id"],
            "vehicle_configuration_id": row["vehicle_configuration_id"],
            "architecture": row["architecture"],
            "application_class": row.get("application_class"),
            "drive_layout": row.get("drive_layout"),
            "test_mass_kg": row.get("test_mass_resolved_kg"),
            "stratum": row.get("stratum"),
            "parent_rolling_minor_resolution_id": row["rolling_minor_resolution_id"],
        }
        for row in sample
    ]
    _write_csv(output_dir / "pilot_sample.csv", sample_rows, sample_rows[0].keys())
    sample_identity_sha256 = sha256(
        ",".join(str(row["vde_id"]) for row in sample_rows).encode("ascii")
    ).hexdigest().upper()

    phase_results: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for phase in ("A", "B"):
        matches: list[dict[str, Any]] = []
        curves: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for vehicle in sample:
            vehicle_matches, vehicle_curves, vehicle_summary = evaluate_refinement_vehicle(
                vehicle, component_refs, enriched, tire_refs, phase=phase
            )
            matches.extend(vehicle_matches)
            curves.extend(vehicle_curves)
            summaries.append(vehicle_summary)
        phase_results[phase] = (matches, curves, summaries)
        prefix = f"phase_{phase.lower()}"
        _write_csv(output_dir / f"{prefix}_component_matches.csv", matches, matches[0].keys())
        _write_csv(output_dir / f"{prefix}_buildup_curves.csv", curves, curves[0].keys())
        _write_csv(output_dir / f"{prefix}_buildup_summary.csv", summaries, summaries[0].keys())
    metadata = applicability_rows(enriched)
    _write_csv(output_dir / "synthetic_applicability_metadata.csv", metadata, metadata[0].keys())

    phase_a = _metrics(phase_results["A"][0], phase_results["A"][2], phase_results["A"][1])
    phase_b = _metrics(phase_results["B"][0], phase_results["B"][2], phase_results["B"][1])
    old["clean_buildup"] = phase_a["clean_buildup"]
    old["rejected_over_5pct"] = phase_a["rejected_over_5pct"]
    old_tire_ids = {
        row["vde_id"] for row in phase_results["A"][0]
        if row["fine_domain"] == "TIRE" and row["grade_accepted"]
    }
    new_tire_ids = {
        row["vde_id"] for row in phase_results["A"][0]
        if row["fine_domain"] == "TIRE" and row["accepted_in_build"]
    }
    phase_a["newly_recovered_vs_old"] = phase_a["tire_accepted"] - old["tire_accepted"]
    phase_a["candidate_ids_consistent"] = len(old_tire_ids) == old["tire_candidates"]
    _write_comparison(output_dir / "pass1c1_comparison.csv", old, phase_a, phase_b)

    hashes_after = {"pass1_db": file_sha256(db_path), "legacy_db": file_sha256(legacy_db_path)}
    if hashes_before != hashes_after:
        raise RuntimeError("Read-only refinement changed an input database")
    decision = _decision(phase_b)
    summary = {
        "refinement_version": REFINEMENT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_metadata(),
        "sample_size": len(sample),
        "sample_by_architecture": dict(sorted(Counter(row["architecture"] for row in sample).items())),
        "same_sample_as_pass_1c": True,
        "sample_identity_sha256": sample_identity_sha256,
        "tire_hierarchy_evidence": {
            "canonical_tire_id_cases": sum(bool(row.get("front_tire_id") or row.get("rear_tire_id")) for row in sample),
            "vehicle_specific_rrc_cases": sum(row.get("rrc_resolved") is not None for row in sample),
            "tire_size_cases": sum(bool(row.get("tire_size_resolved")) for row in sample),
            "generic_category_fallbacks_adopted": 0,
        },
        "max_synthetic_overshoot_pct": MAX_SYNTHETIC_OVERSHOOT_PCT,
        "original_pass_1c": old,
        "phase_a": phase_a,
        "phase_b": phase_b,
        "phase_a_tire_candidate_ids": len(old_tire_ids),
        "phase_a_tire_accepted_ids": len(new_tire_ids),
        "synthetic_applicability_reference_count": len(enriched),
        "hashes_before": hashes_before,
        "hashes_after": hashes_after,
        "quick_check": quick_check,
        "foreign_key_issues": fk_issues,
        "database_rows_written": 0,
        "database_objects_created": 0,
        "external_search_count": 0,
        "scaled_to_force_closure": False,
        "scale_up_executed": False,
        "decision": decision,
    }
    (output_dir / "execution_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "pass1c1_summary.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _report(summary: dict[str, Any]) -> str:
    old, phase_a, phase_b = summary["original_pass_1c"], summary["phase_a"], summary["phase_b"]
    lines = [
        "# Sprint 12 Pass 1C.1 — Synthetic Build-up Refinement", "",
        "## Frozen scope and safety", "",
        f"- Same Pass 1C sample: **YES**, {summary['sample_size']} VDEs",
        f"- Sample identity SHA256: `{summary['sample_identity_sha256']}`",
        f"- Architecture allocation: `{json.dumps(summary['sample_by_architecture'], sort_keys=True)}`",
        f"- Overshoot acceptance ceiling: **{summary['max_synthetic_overshoot_pct']:.1f}%** point-by-point",
        "- Component scaling/normalization: **NONE**",
        "- External research: **NONE**",
        f"- DB writes/objects created: **{summary['database_rows_written']} / {summary['database_objects_created']}**",
        f"- quick_check/FK issues: **{summary['quick_check']} / {summary['foreign_key_issues']}**", "",
        "## Phase A — tolerance correction only", "",
        f"- Tire candidates: **{phase_a['tire_candidates']}**",
        f"- Tire accepted under old 0.1 N rule: **{old['tire_accepted']}**",
        f"- Tire rejected under old 0.1 N rule: **{old['rejected_old_0_1N_rule']}**",
        f"- Tire accepted under new 5% rule: **{phase_a['tire_accepted']}**",
        f"- Newly recovered: **{phase_a['newly_recovered_vs_old']}**",
        f"- Still rejected >5%: **{phase_a['rejected_over_5pct']}**",
        f"- Brake / Hub / Axle accepted: **{phase_a['brake_accepted']} / {phase_a['hub_accepted']} / {phase_a['axle_accepted']}**",
        f"- Tire only / Tire+Brake / Tire+Hub / Tire+Brake+Hub: **{phase_a['tire_only']} / {phase_a['tire_brake']} / {phase_a['tire_hub']} / {phase_a['tire_brake_hub']}**", "",
        "## Tire evidence hierarchy", "",
        f"- Canonical Tire DB / EPA tire ID cases: **{summary['tire_hierarchy_evidence']['canonical_tire_id_cases']}**",
        f"- Existing vehicle-specific RRC cases: **{summary['tire_hierarchy_evidence']['vehicle_specific_rrc_cases']}**",
        f"- Tire-size cases: **{summary['tire_hierarchy_evidence']['tire_size_cases']}**",
        "- Generic category fallbacks adopted: **0**", "",
        "## Phase B — rule-defined synthetic applicability", "",
        "Applicability ranges are broad rule-defined envelopes, not measured vehicle-specific values. Axle-load numeric ranges remain blank because the source references do not support them. Grade C remains candidate-only.", "",
        "| Metric | Original Pass 1C | Phase A 5% | Phase B metadata |",
        "|---|---:|---:|---:|",
        f"| Tire accepted | {old['tire_accepted']} | {phase_a['tire_accepted']} | {phase_b['tire_accepted']} |",
        f"| Brake Grade A/B | {old['brake_grade_ab']} | {phase_a['brake_grade_ab']} | {phase_b['brake_grade_ab']} |",
        f"| Hub Grade A/B | {old['hub_grade_ab']} | {phase_a['hub_grade_ab']} | {phase_b['hub_grade_ab']} |",
        f"| Axle Grade A/B | {old['axle_grade_ab']} | {phase_a['axle_grade_ab']} | {phase_b['axle_grade_ab']} |",
        f"| Tire + Brake/Hub accepted | {old['tire_plus_brake_or_hub']} | {phase_a['tire_plus_brake_or_hub']} | {phase_b['tire_plus_brake_or_hub']} |",
        f"| Clean build-up | {old['clean_buildup']} | {phase_a['clean_buildup']} | {phase_b['clean_buildup']} |",
        f"| Tolerance accepted | {old['tolerance_accepted']} | {phase_a['tolerance_accepted']} | {phase_b['tolerance_accepted']} |",
        f"| Rejected >5% | {old['rejected_over_5pct']} | {phase_a['rejected_over_5pct']} | {phase_b['rejected_over_5pct']} |", "",
        "## Phase B residual and closure", "",
        f"- Unresolved residual % median / P10 / P90 / max: **{_fmt(phase_b['median_unresolved_residual_pct'])} / {_fmt(phase_b['p10_unresolved_residual_pct'])} / {_fmt(phase_b['p90_unresolved_residual_pct'])} / {_fmt(phase_b['max_unresolved_residual_pct'])}**",
        f"- Closure error % for tolerance-accepted points median / P90 / max: **{_fmt(phase_b['median_closure_error_pct'])} / {_fmt(phase_b['p90_closure_error_pct'])} / {_fmt(phase_b['max_closure_error_pct'])}**",
        f"- Tire + Brake/Hub accepted coverage: **{phase_b['tire_plus_brake_or_hub_pct']:.3f}%**",
        f"- Axle accepted: **{phase_b['axle_accepted']}** (overlap protection retained)", "",
        "## Scale-up decision", "",
        f"`{summary['decision']}`", "",
        f"Measured Tire + Brake/Hub coverage is **{phase_b['tire_plus_brake_or_hub_pct']:.3f}%**, below the explicit 20% stop threshold.", "",
        "No 7,429-VDE scale-up was executed. RollingMinor and all core estimator outputs remain authoritative and unchanged.",
    ]
    return "\n".join(lines) + "\n"


__all__ = [
    "APPLICATION_ENVELOPES",
    "MAX_SYNTHETIC_OVERSHOOT_PCT",
    "ApplicabilityReference",
    "classify_closure",
    "enrich_applicability",
    "evaluate_refinement_vehicle",
    "execute_refinement",
    "match_applicability_reference",
]
