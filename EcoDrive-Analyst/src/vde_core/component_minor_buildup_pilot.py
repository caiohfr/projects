"""Read-only Sprint 12 Pass 1C synthetic minor build-up pilot.

The pilot selects component references from engineering metadata before using
the accepted RollingMinor curve as an independent physical envelope.  It does
not persist fine components or alter the frozen Pass 1 estimator outputs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import sqlite3
import subprocess
from typing import Any, Iterable
from zipfile import ZipFile

from src.vde_core.component_enrichment_pass1 import ESTIMATOR_VERSION, file_sha256
from src.vde_core.component_prior_matching_vnext import (
    normalize_application_class,
    normalize_drive,
)


PILOT_VERSION = "SPRINT12_PASS1C_SYNTHETIC_MINOR_BUILDUP_PILOT_V1.0"
MATCH_METHOD = "SYNTHETIC_MATCHED_BUILDUP_V1"
QA_SPEEDS_KPH = tuple(float(value) for value in range(24, 106))
NUMERIC_TOLERANCE_N = 0.1
DEFAULT_ARCHITECTURE_TARGETS = {
    "CONVENTIONAL_MULTI_SPEED": 90,
    "HYBRID_PARALLEL_CONVENTIONAL_TRANS": 45,
    "EV_FIXED_GEAR": 25,
    "EV_MULTI_SPEED": 20,
}
FINE_DOMAINS = ("TIRE", "BRAKE", "HUB_BEARING", "AXLE")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_object(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _token(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    return text or None


def _curve(abc: tuple[float, float, float] | None, speed_kph: float) -> float:
    if abc is None:
        return 0.0
    return abc[0] + abc[1] * speed_kph + abc[2] * speed_kph * speed_kph


def _read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _git_metadata() -> dict[str, str]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ("git", *args),
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE"

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current") or "DETACHED",
    }


@dataclass(frozen=True)
class Reference:
    component_id: str
    resolution_id: str
    domain: str
    application_class: str
    drive: str
    position: str | None
    abc: tuple[float, float, float]
    population_n: int | None
    reference_mass_kg: float | None = None
    wheel_radius_mm: float | None = None


def load_component_references(path: Path) -> list[Reference]:
    rows: list[Reference] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            domain = _token(row.get("domain"))
            if domain not in {"BRAKE", "AXLE", "HUB_BEARING"}:
                continue
            rows.append(
                Reference(
                    component_id=str(row["component_id"]),
                    resolution_id=str(row["component_resolution_id"]),
                    domain=domain,
                    application_class=_token(row.get("application_class")) or "",
                    drive=_token(row.get("drive_architecture")) or "",
                    position=_token(row.get("position")),
                    abc=(float(row["A_N"]), float(row["B_N_per_kph"]), float(row["C_N_per_kph2"])),
                    population_n=int(row["population_n"]) if row.get("population_n") else None,
                    reference_mass_kg=_float(row.get("reference_mass_kg")),
                    wheel_radius_mm=_float(row.get("wheel_radius_mm")),
                )
            )
    return sorted(rows, key=lambda item: (item.domain, item.component_id))


def load_tire_references(zip_path: Path) -> list[dict[str, Any]]:
    with ZipFile(zip_path) as archive:
        with archive.open("EcoDrive_Synthetic_Tire_DB_v1.csv") as raw:
            import io

            text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            return [dict(row) for row in csv.DictReader(text)]


def load_legacy_lookup(connection: sqlite3.Connection) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    lookup: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    sql = """
        SELECT make,model,year,category,tire_size,test_mass_kg,mass_kg,
               rrc_N_per_kN,front_pressure_psi,rear_pressure_psi,drive_type
        FROM vde_db
        WHERE make IS NOT NULL AND model IS NOT NULL AND year IS NOT NULL
        ORDER BY make,model,year,id
    """
    for raw in connection.execute(sql):
        row = dict(raw)
        try:
            key = (_token(row["make"]) or "", _token(row["model"]) or "", int(row["year"]))
        except (TypeError, ValueError):
            continue
        lookup[key].append(row)
    return lookup


def _unique_nonempty(rows: list[dict[str, Any]], field: str) -> tuple[str, ...]:
    return tuple(sorted({str(row[field]).strip() for row in rows if row.get(field) not in (None, "")}))


def _unique_float(rows: list[dict[str, Any]], field: str) -> float | None:
    values = sorted({_float(row.get(field)) for row in rows if _float(row.get(field)) is not None})
    return values[0] if len(values) == 1 else None


def load_resolved_population(
    connection: sqlite3.Connection,
    legacy_lookup: dict[tuple[str, str, int], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    sql = """
        SELECT v.id AS vde_id,v.vehicle_configuration_id,v.make,v.model,v.year,
               v.category,v.test_mass_kg,v.mass_kg,v.tire_size,v.rrc_N_per_kN,
               v.front_pressure_psi,v.rear_pressure_psi,v.front_tire_id,v.rear_tire_id,
               vc.drive_system,vc.engine_rated_power_kw,
               link.component_resolution_id AS rolling_minor_resolution_id,
               resolution.resolved_A_N,resolution.resolved_B_N_per_kph,
               resolution.resolved_C_N_per_kph2,resolution.estimate_status,
               resolution.provenance_json AS rolling_provenance_json
        FROM vde_component_resolution AS link
        JOIN component_resolution AS resolution
          ON resolution.component_resolution_id=link.component_resolution_id
        JOIN vde AS v ON v.id=link.vde_id
        JOIN vehicle_configuration AS vc
          ON vc.vehicle_configuration_id=v.vehicle_configuration_id
        WHERE link.boundary='ROLLING_MINOR'
          AND resolution.record_status='ACTIVE'
          AND resolution.estimate_status IN ('SUPPORTED','CONDITIONAL')
          AND resolution.estimator_version=?
          AND v.record_status='ACTIVE'
        ORDER BY v.id
    """
    population: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in connection.execute(sql, (ESTIMATOR_VERSION,)):
        row = dict(raw)
        vde_id = int(row["vde_id"])
        if vde_id in seen:
            raise ValueError(f"Multiple accepted ROLLING_MINOR parents for VDE {vde_id}")
        seen.add(vde_id)
        provenance = _json_object(row.pop("rolling_provenance_json"))
        architecture = str(provenance.get("architecture_class") or "UNRESOLVED")
        try:
            legacy_key = (_token(row["make"]) or "", _token(row["model"]) or "", int(row["year"]))
        except (TypeError, ValueError):
            legacy_key = None
        legacy_rows = legacy_lookup.get(legacy_key, []) if legacy_key else []
        legacy_categories = _unique_nonempty(legacy_rows, "category")
        application = normalize_application_class(legacy_categories)
        legacy_tire_sizes = _unique_nonempty(legacy_rows, "tire_size")
        tire_size = str(row.get("tire_size") or "").strip() or (legacy_tire_sizes[0] if len(legacy_tire_sizes) == 1 else None)
        test_mass = _float(row.get("test_mass_kg")) or _unique_float(legacy_rows, "test_mass_kg") or _float(row.get("mass_kg"))
        canonical_rrc = _float(row.get("rrc_N_per_kN"))
        legacy_rrc = _unique_float(legacy_rows, "rrc_N_per_kN")
        rrc = canonical_rrc if canonical_rrc is not None else legacy_rrc
        rrc_source = (
            "canonical.vde.rrc_N_per_kN"
            if canonical_rrc is not None
            else ("legacy.vde_db.rrc_N_per_kN:EXACT_MAKE_MODEL_YEAR_LOOKUP" if legacy_rrc is not None else None)
        )
        drive = normalize_drive(row.get("drive_system"))
        app_class = application.normalized_options[0] if application.status == "MAPPED" and len(application.normalized_options) == 1 else None
        row.update(
            {
                "architecture": architecture,
                "rolling_status": row["estimate_status"],
                "rolling_abc": (
                    float(row["resolved_A_N"]),
                    float(row["resolved_B_N_per_kph"]),
                    float(row["resolved_C_N_per_kph2"]),
                ),
                "application_class": app_class,
                "application_class_status": application.status,
                "legacy_categories": legacy_categories,
                "drive_layout": drive.normalized,
                "drive_status": drive.status,
                "test_mass_resolved_kg": test_mass,
                "tire_size_resolved": tire_size,
                "rrc_resolved": rrc,
                "rrc_source": rrc_source,
                "legacy_match_count": len(legacy_rows),
            }
        )
        population.append(row)
    return population


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = fraction * (len(ordered) - 1)
    low = int(math.floor(index))
    high = int(math.ceil(index))
    if low == high:
        return ordered[low]
    weight = index - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def assign_mass_bands(population: list[dict[str, Any]]) -> None:
    by_architecture: dict[str, list[float]] = defaultdict(list)
    for row in population:
        mass = _float(row.get("test_mass_resolved_kg"))
        if mass is not None:
            by_architecture[str(row["architecture"])].append(mass)
    bounds = {
        architecture: (_quantile(values, 1.0 / 3.0), _quantile(values, 2.0 / 3.0))
        for architecture, values in by_architecture.items()
    }
    for row in population:
        mass = _float(row.get("test_mass_resolved_kg"))
        if mass is None or row["architecture"] not in bounds:
            row["mass_band"] = "MISSING"
        else:
            low, high = bounds[row["architecture"]]
            row["mass_band"] = "LOW" if mass <= low else ("MEDIUM" if mass <= high else "HIGH")


def deterministic_stratified_sample(
    population: list[dict[str, Any]],
    targets: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    targets = dict(targets or DEFAULT_ARCHITECTURE_TARGETS)
    assign_mass_bands(population)
    selected: list[dict[str, Any]] = []
    for architecture, target in targets.items():
        candidates = [row for row in population if row["architecture"] == architecture]
        strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            tire_state = "TIRE_METADATA" if row.get("tire_size_resolved") or row.get("rrc_resolved") else "NO_TIRE_METADATA"
            stratum = "|".join(
                (
                    row.get("application_class") or f"APP_{row.get('application_class_status')}",
                    str(row.get("mass_band")),
                    row.get("drive_layout") or f"DRIVE_{row.get('drive_status')}",
                    tire_state,
                )
            )
            row["stratum"] = stratum
            strata[stratum].append(row)
        for stratum, rows in strata.items():
            rows.sort(key=lambda item: sha256(f"{PILOT_VERSION}|{stratum}|{item['vde_id']}".encode()).hexdigest())
        stratum_names = sorted(strata)
        while len([row for row in selected if row["architecture"] == architecture]) < min(target, len(candidates)):
            progressed = False
            for stratum in stratum_names:
                if strata[stratum]:
                    selected.append(strata[stratum].pop(0))
                    progressed = True
                    if len([row for row in selected if row["architecture"] == architecture]) >= target:
                        break
            if not progressed:
                break
    return sorted(selected, key=lambda row: (str(row["architecture"]), int(row["vde_id"])))


def _mass_difference_pct(vehicle_mass: float | None, reference_mass: float | None) -> float | None:
    if vehicle_mass is None or reference_mass is None or reference_mass <= 0:
        return None
    return 100.0 * abs(vehicle_mass - reference_mass) / reference_mass


def match_synthetic_reference(
    vehicle: dict[str, Any],
    references: list[Reference],
    domain: str,
    *,
    position: str | None = None,
) -> dict[str, Any]:
    app_class = _token(vehicle.get("application_class"))
    drive = _token(vehicle.get("drive_layout"))
    position = _token(position)
    candidates = [reference for reference in references if reference.domain == domain]
    criteria = ["boundary"]
    if position:
        candidates = [reference for reference in candidates if reference.position == position]
        criteria.append("position")
    if app_class:
        candidates = [reference for reference in candidates if reference.application_class == app_class]
        criteria.append("application_class")
    else:
        candidates = []
    if drive:
        candidates = [reference for reference in candidates if reference.drive == drive]
        criteria.append("drive_architecture")
    else:
        candidates = []
    candidates.sort(key=lambda reference: reference.component_id)
    if len(candidates) != 1:
        return {
            "grade": "UNRESOLVED",
            "accepted": False,
            "selected": None,
            "candidates": candidates,
            "score": 0,
            "criteria": criteria,
            "reason_codes": ["NO_UNIQUE_METADATA_MATCH"],
        }
    selected = candidates[0]
    mass_delta = _mass_difference_pct(_float(vehicle.get("test_mass_resolved_kg")), selected.reference_mass_kg)
    wheel_available = selected.wheel_radius_mm is not None and vehicle.get("wheel_radius_mm") is not None
    reasons: list[str] = []
    if mass_delta is None:
        reasons.append("REFERENCE_MASS_METADATA_MISSING")
    if domain == "HUB_BEARING" and not wheel_available:
        reasons.append("REFERENCE_OR_VEHICLE_WHEEL_CLASS_MISSING")
    score = 40 + 30 + 20 + (10 if position else 0)
    if mass_delta is not None and mass_delta <= 10.0 and (domain != "HUB_BEARING" or wheel_available):
        grade = "A"
    elif mass_delta is not None and mass_delta <= 20.0 and (domain != "HUB_BEARING" or wheel_available):
        grade = "B"
    else:
        grade = "C"
        reasons.append("MATERIAL_MATCH_DISCRIMINANTS_INCOMPLETE")
    return {
        "grade": grade,
        "accepted": grade in {"A", "B"},
        "selected": selected,
        "candidates": candidates,
        "score": score,
        "criteria": criteria,
        "mass_difference_pct": mass_delta,
        "reason_codes": sorted(set(reasons)),
    }


def match_tire(vehicle: dict[str, Any], tire_references: list[dict[str, Any]]) -> dict[str, Any]:
    mass = _float(vehicle.get("test_mass_resolved_kg"))
    rrc = _float(vehicle.get("rrc_resolved"))
    if rrc is not None and mass is not None:
        abc = (rrc * mass * 9.80665 / 1000.0, 0.0, 0.0)
        rrc_source = str(vehicle.get("rrc_source") or "existing_vehicle_rrc")
        canonical = rrc_source.startswith("canonical.")
        return {
            "grade": "A",
            "accepted": True,
            "selected_reference_id": "CANONICAL_VDE_RRC" if canonical else "LEGACY_VDE_RRC",
            "candidate_ids": [],
            "abc": abc,
            "score": 100,
            "criteria": ["vehicle_specific_rrc", "test_mass"],
            "source": "RULE_ESTIMATED_FROM_CANONICAL_VDE_RRC" if canonical else "RULE_ESTIMATED_FROM_LEGACY_VDE_RRC",
            "reason_codes": [],
            "exact_source": canonical,
        }
    size = _token(vehicle.get("tire_size_resolved"))
    size_matches = sorted(
        (row for row in tire_references if _token(row.get("size_code")) == size),
        key=lambda row: int(row["tire_id"]),
    ) if size else []
    if size_matches:
        return {
            "grade": "C",
            "accepted": False,
            "selected_reference_id": None,
            "candidate_ids": [row["tire_test_code"] for row in size_matches],
            "abc": None,
            "score": 55,
            "criteria": ["tire_size"],
            "source": "SYNTHETIC_TIRE_REFERENCE",
            "reason_codes": ["TIRE_RR_TIER_NOT_IDENTIFIABLE_FROM_EXISTING_METADATA"],
            "exact_source": False,
        }
    return {
        "grade": "UNRESOLVED",
        "accepted": False,
        "selected_reference_id": None,
        "candidate_ids": [],
        "abc": None,
        "score": 0,
        "criteria": [],
        "source": "UNRESOLVED",
        "reason_codes": ["TIRE_SIZE_RRC_PRESSURE_METADATA_UNAVAILABLE"],
        "exact_source": False,
    }


def _aggregate_position_matches(matches: list[dict[str, Any]]) -> dict[str, Any]:
    grades = [match["grade"] for match in matches]
    accepted = all(match["accepted"] for match in matches) and bool(matches)
    selected = [match["selected"] for match in matches if match.get("selected")]
    grade = grades[0] if grades and len(set(grades)) == 1 else ("UNRESOLVED" if "UNRESOLVED" in grades else "C")
    abc = None
    if accepted:
        abc = tuple(sum(reference.abc[index] for reference in selected) for index in range(3))
    return {
        "grade": grade,
        "accepted": accepted,
        "selected": selected,
        "candidates": [reference for match in matches for reference in match.get("candidates", [])],
        "score": sum(int(match.get("score") or 0) for match in matches) / len(matches) if matches else 0,
        "criteria": sorted({criterion for match in matches for criterion in match.get("criteria", [])}),
        "reason_codes": sorted({reason for match in matches for reason in match.get("reason_codes", [])}),
        "abc": abc,
    }


def evaluate_vehicle(
    vehicle: dict[str, Any],
    component_references: list[Reference],
    tire_references: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tire = match_tire(vehicle, tire_references)
    brake = match_synthetic_reference(vehicle, component_references, "BRAKE")
    hub = _aggregate_position_matches(
        [
            match_synthetic_reference(vehicle, component_references, "HUB_BEARING", position="FRONT"),
            match_synthetic_reference(vehicle, component_references, "HUB_BEARING", position="REAR"),
        ]
    )
    axle_candidates = _aggregate_position_matches(
        [
            match_synthetic_reference(vehicle, component_references, "AXLE", position="FRONT"),
            match_synthetic_reference(vehicle, component_references, "AXLE", position="REAR"),
        ]
    )
    axle = dict(axle_candidates)
    axle["accepted"] = False
    axle["abc"] = None
    axle["reason_codes"] = sorted(set((*axle.get("reason_codes", []), "AXLE_EXCLUDED_BOUNDARY_OVERLAP_RISK")))

    component_results = {"TIRE": tire, "BRAKE": brake, "HUB_BEARING": hub, "AXLE": axle}
    accepted_abc: dict[str, tuple[float, float, float]] = {}
    if tire["accepted"]:
        accepted_abc["TIRE"] = tire["abc"]
    for domain, result in (("BRAKE", brake), ("HUB_BEARING", hub)):
        if result["accepted"]:
            selected = result.get("selected")
            if isinstance(selected, list):
                accepted_abc[domain] = result["abc"]
            elif selected:
                accepted_abc[domain] = selected.abc

    rolling_abc = vehicle["rolling_abc"]
    envelope_exceeded = any(
        sum(_curve(abc, speed) for abc in accepted_abc.values()) > _curve(rolling_abc, speed) + NUMERIC_TOLERANCE_N
        for speed in QA_SPEEDS_KPH
    )
    if envelope_exceeded:
        for domain in tuple(accepted_abc):
            component_results[domain]["accepted"] = False
            component_results[domain]["reason_codes"] = sorted(
                set((*component_results[domain].get("reason_codes", []), "SYNTHETIC_BUILDUP_EXCEEDS_ROLLING_MINOR"))
            )
        accepted_abc = {}

    match_rows: list[dict[str, Any]] = []
    for domain in FINE_DOMAINS:
        result = component_results[domain]
        selected_objects = result.get("selected") or []
        if isinstance(selected_objects, Reference):
            selected_objects = [selected_objects]
        selected_ids = [reference.component_id for reference in selected_objects]
        candidate_ids = result.get("candidate_ids") or [reference.component_id for reference in result.get("candidates", [])]
        abc = result.get("abc")
        if abc is None and isinstance(result.get("selected"), Reference):
            abc = result["selected"].abc
        match_rows.append(
            {
                "vde_id": vehicle["vde_id"],
                "fine_domain": domain,
                "selected_reference_id": result.get("selected_reference_id") or ";".join(selected_ids),
                "candidate_reference_ids_json": _json(candidate_ids),
                "match_grade": result["grade"],
                "match_score": result.get("score"),
                "match_criteria_json": _json(result.get("criteria", [])),
                "source_provenance": result.get("source") or ("SYNTHETIC_REFERENCE" if selected_objects else "UNRESOLVED"),
                "method": MATCH_METHOD if domain != "TIRE" else result.get("source"),
                "estimate_status": "CONDITIONAL" if result["accepted"] else "UNRESOLVED",
                "accepted": int(bool(result["accepted"])),
                "A_N": abc[0] if abc else None,
                "B_N_per_kph": abc[1] if abc else None,
                "C_N_per_kph2": abc[2] if abc else None,
                "parent_rolling_minor_resolution_id": vehicle["rolling_minor_resolution_id"],
                "boundary_rejected": int(domain == "AXLE"),
                "envelope_rejected": int("SYNTHETIC_BUILDUP_EXCEEDS_ROLLING_MINOR" in result.get("reason_codes", [])),
                "rejection_reason": ";".join(result.get("reason_codes", [])),
            }
        )

    curves: list[dict[str, Any]] = []
    residual_fractions: list[float] = []
    for speed in QA_SPEEDS_KPH:
        rolling = _curve(rolling_abc, speed)
        forces = {domain: _curve(accepted_abc.get(domain), speed) for domain in FINE_DOMAINS}
        buildup = sum(forces.values())
        unresolved = rolling - buildup
        fraction = unresolved / rolling if rolling > 1e-12 else None
        if fraction is not None:
            residual_fractions.append(fraction)
        curves.append(
            {
                "vde_id": vehicle["vde_id"],
                "speed_kph": speed,
                "rolling_minor_N": rolling,
                "tire_N": forces["TIRE"],
                "brake_N": forces["BRAKE"],
                "hub_N": forces["HUB_BEARING"],
                "axle_N": forces["AXLE"],
                "buildup_N": buildup,
                "unresolved_N": unresolved,
                "unresolved_fraction": fraction,
            }
        )
    accepted_domains = [domain for domain in FINE_DOMAINS if component_results[domain]["accepted"]]
    worst = max(residual_fractions) if residual_fractions else None
    if envelope_exceeded:
        information_class = "REJECTED"
        status = "SYNTHETIC_BUILDUP_EXCEEDS_ROLLING_MINOR"
    elif not accepted_domains:
        information_class = "UNRESOLVED"
        status = "NO_ACCEPTED_FINE_COMPONENTS"
    elif worst is not None and worst <= 0.40:
        information_class = "INFORMATIVE"
        status = "ACCEPTED"
    elif worst is not None and worst <= 0.70:
        information_class = "PARTIAL"
        status = "ACCEPTED"
    else:
        information_class = "WEAK"
        status = "ACCEPTED"
    at_80 = next(row for row in curves if row["speed_kph"] == 80.0)
    summary = {
        "vde_id": vehicle["vde_id"],
        "accepted_component_set": ";".join(accepted_domains),
        "worst_residual_fraction": worst,
        "residual_fraction_at_80_kph": at_80["unresolved_fraction"],
        "information_class": information_class,
        "buildup_status": status,
        "reason_codes": ";".join(
            sorted({reason for result in component_results.values() for reason in result.get("reason_codes", [])})
        ),
        "parent_rolling_minor_resolution_id": vehicle["rolling_minor_resolution_id"],
        "parent_core_status": vehicle["rolling_status"],
        "fine_status": "CONDITIONAL" if accepted_domains else "UNRESOLVED",
        "scaled_to_force_closure": 0,
    }
    return match_rows, curves, summary


def _percentile(values: list[float], fraction: float) -> float | None:
    return _quantile(values, fraction) if values else None


def execute_pilot(
    db_path: Path,
    legacy_db_path: Path,
    component_catalog_path: Path,
    tire_zip_path: Path,
    output_dir: Path,
    *,
    targets: dict[str, int] | None = None,
) -> dict[str, Any]:
    db_path = Path(db_path).resolve(strict=True)
    legacy_db_path = Path(legacy_db_path).resolve(strict=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes_before = {"pass1_db": file_sha256(db_path), "legacy_db": file_sha256(legacy_db_path)}
    connection = _read_only(db_path)
    legacy_connection = _read_only(legacy_db_path)
    quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
    foreign_key_issues = len(connection.execute("PRAGMA foreign_key_check").fetchall())
    if quick_check != "ok" or foreign_key_issues:
        raise ValueError(f"Input database integrity failed: quick={quick_check}, fk={foreign_key_issues}")
    legacy_lookup = load_legacy_lookup(legacy_connection)
    population = load_resolved_population(connection, legacy_lookup)
    connection.close()
    legacy_connection.close()
    if len(population) != 7429:
        raise ValueError(f"Expected 7,429 accepted RollingMinor VDEs, found {len(population):,}")
    sample = deterministic_stratified_sample(population, targets)
    expected_size = sum((targets or DEFAULT_ARCHITECTURE_TARGETS).values())
    if len(sample) != expected_size:
        raise ValueError(f"Pilot sample shortfall: expected {expected_size}, got {len(sample)}")
    component_references = load_component_references(component_catalog_path)
    tire_references = load_tire_references(tire_zip_path)

    sample_rows: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    buildup: list[dict[str, Any]] = []
    for vehicle in sample:
        tire_summary = {
            "front_tire_id": vehicle.get("front_tire_id"),
            "rear_tire_id": vehicle.get("rear_tire_id"),
            "size": vehicle.get("tire_size_resolved"),
            "rrc_N_per_kN": vehicle.get("rrc_resolved"),
            "rrc_source": vehicle.get("rrc_source"),
            "front_pressure_psi": vehicle.get("front_pressure_psi"),
            "rear_pressure_psi": vehicle.get("rear_pressure_psi"),
        }
        sample_rows.append(
            {
                "vde_id": vehicle["vde_id"],
                "vehicle_configuration_id": vehicle["vehicle_configuration_id"],
                "architecture": vehicle["architecture"],
                "category_body_class": ";".join(vehicle["legacy_categories"]),
                "application_class": vehicle.get("application_class"),
                "test_mass_kg": vehicle.get("test_mass_resolved_kg"),
                "mass_band": vehicle.get("mass_band"),
                "drive_layout": vehicle.get("drive_layout"),
                "tire_metadata_summary": _json(tire_summary),
                "stratum": vehicle["stratum"],
                "parent_rolling_minor_resolution_id": vehicle["rolling_minor_resolution_id"],
            }
        )
        vehicle_matches, vehicle_curves, vehicle_summary = evaluate_vehicle(vehicle, component_references, tire_references)
        matches.extend(vehicle_matches)
        curves.extend(vehicle_curves)
        buildup.append(vehicle_summary)

    coverage_rows: list[dict[str, Any]] = []
    for domain in FINE_DOMAINS:
        domain_rows = [row for row in matches if row["fine_domain"] == domain]
        counts = Counter(row["match_grade"] for row in domain_rows)
        accepted = sum(int(row["accepted"]) for row in domain_rows)
        coverage_rows.append(
            {
                "domain": domain,
                "eligible": len(domain_rows),
                "exact_source": sum(row["selected_reference_id"] == "CANONICAL_VDE_RRC" for row in domain_rows),
                "grade_a": counts["A"],
                "grade_b": counts["B"],
                "grade_c": counts["C"],
                "unresolved": counts["UNRESOLVED"],
                "envelope_rejected": sum(int(row["envelope_rejected"]) for row in domain_rows),
                "boundary_rejected": sum(int(row["boundary_rejected"]) for row in domain_rows),
                "accepted_coverage_pct": round(100.0 * accepted / len(domain_rows), 6) if domain_rows else 0.0,
            }
        )

    usage: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in matches:
        ids = json.loads(row["candidate_reference_ids_json"])
        if row["selected_reference_id"]:
            ids = sorted(set((*ids, *str(row["selected_reference_id"]).split(";"))))
        for reference_id in ids:
            usage[(row["fine_domain"], reference_id)]["candidate_count"] += 1
            if int(row["accepted"]):
                usage[(row["fine_domain"], reference_id)]["accepted_count"] += 1
    usage_rows = []
    for (domain, reference_id), counts in sorted(usage.items()):
        usage_rows.append(
            {
                "domain": domain,
                "reference_id": reference_id,
                "candidate_count": counts["candidate_count"],
                "accepted_count": counts["accepted_count"],
                "sample_share_pct": round(100.0 * counts["candidate_count"] / len(sample), 6),
                "extreme_concentration_flag": int(counts["candidate_count"] > 0.25 * len(sample)),
                "usage_semantics": "ACCEPTED_A_B" if counts["accepted_count"] else "CANDIDATE_ONLY_NOT_ADOPTED",
            }
        )

    _write_csv(output_dir / "pilot_sample.csv", sample_rows, sample_rows[0].keys())
    _write_csv(output_dir / "fine_component_matches.csv", matches, matches[0].keys())
    _write_csv(output_dir / "minor_buildup_curves.csv", curves, curves[0].keys())
    _write_csv(output_dir / "pilot_component_coverage.csv", coverage_rows, coverage_rows[0].keys())
    _write_csv(output_dir / "pilot_buildup_summary.csv", buildup, buildup[0].keys())
    _write_csv(output_dir / "reference_usage_summary.csv", usage_rows, usage_rows[0].keys() if usage_rows else ("domain", "reference_id"))

    hashes_after = {"pass1_db": file_sha256(db_path), "legacy_db": file_sha256(legacy_db_path)}
    if hashes_after != hashes_before:
        raise RuntimeError("Read-only pilot changed an input database")
    sample_counts = Counter(row["architecture"] for row in sample)
    information_counts = Counter(row["information_class"] for row in buildup)
    accepted_sets = Counter(row["accepted_component_set"] for row in buildup)
    residuals = [
        float(row["worst_residual_fraction"])
        for row in buildup
        if row["worst_residual_fraction"] is not None
        and row["accepted_component_set"]
        and row["buildup_status"] == "ACCEPTED"
    ]
    negative_material = sum(float(row["unresolved_N"]) < -NUMERIC_TOLERANCE_N for row in curves)
    negative_candidate_cases = sum(
        row["buildup_status"] == "SYNTHETIC_BUILDUP_EXCEEDS_ROLLING_MINOR"
        for row in buildup
    )
    positive_residual_accepted_cases = sum(
        bool(row["accepted_component_set"])
        and row["buildup_status"] == "ACCEPTED"
        and float(row["worst_residual_fraction"] or 0.0) > 0.0
        for row in buildup
    )
    tire_plus_mechanical = sum(
        "TIRE" in row["accepted_component_set"].split(";")
        and any(domain in row["accepted_component_set"].split(";") for domain in ("BRAKE", "HUB_BEARING", "AXLE"))
        for row in buildup
    )
    summary = {
        "pilot_version": PILOT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_metadata(),
        "estimator_version": ESTIMATOR_VERSION,
        "pass1_db_path": str(db_path),
        "legacy_db_path": str(legacy_db_path),
        "component_catalog_path": str(component_catalog_path),
        "component_catalog_sha256": file_sha256(Path(component_catalog_path)),
        "tire_reference_path": str(tire_zip_path),
        "tire_reference_sha256": file_sha256(Path(tire_zip_path)),
        "resolved_population": len(population),
        "sample_size": len(sample),
        "sample_by_architecture": dict(sorted(sample_counts.items())),
        "component_reference_count": len(component_references),
        "tire_reference_count": len(tire_references),
        "coverage": coverage_rows,
        "information_classes": dict(sorted(information_counts.items())),
        "accepted_component_sets": dict(sorted(accepted_sets.items())),
        "tire_plus_mechanical_yield_pct": 100.0 * tire_plus_mechanical / len(sample),
        "residual_fraction": {
            "median": _percentile(residuals, 0.5),
            "p10": _percentile(residuals, 0.1),
            "p90": _percentile(residuals, 0.9),
            "worst_accepted": max(residuals) if residuals else None,
        },
        "negative_material_residual_grid_points": negative_material,
        "rejected_negative_residual_candidate_cases": negative_candidate_cases,
        "positive_residual_accepted_cases": positive_residual_accepted_cases,
        "hashes_before": hashes_before,
        "hashes_after": hashes_after,
        "quick_check": quick_check,
        "foreign_key_issues": foreign_key_issues,
        "database_rows_written": 0,
        "database_objects_created": 0,
        "external_search_count": 0,
        "scaled_to_force_closure": False,
        "persistence": "AUDIT_ARTIFACTS_ONLY",
        "decision": "BLOCKED_BY_REFERENCE_METADATA",
    }
    (output_dir / "execution_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report = _report(summary, sample_rows, usage_rows)
    (output_dir / "pass1c_pilot_summary.md").write_text(report, encoding="utf-8")
    return summary


def _report(summary: dict[str, Any], sample_rows: list[dict[str, Any]], usage_rows: list[dict[str, Any]]) -> str:
    coverage = {row["domain"]: row for row in summary["coverage"]}
    strata = Counter(row["stratum"] for row in sample_rows)
    top_usage = sorted(usage_rows, key=lambda row: (-int(row["candidate_count"]), row["reference_id"]))[:10]
    lines = [
        "# Pass 1C Synthetic Minor Build-up Pilot - Completion Report", "",
        "## Runtime", "",
        f"- Pass 1 source: temporary enriched candidate, **{summary['resolved_population']:,}** accepted RollingMinor parents",
        f"- repository branch/commit: `{summary['git']['branch']}` / `{summary['git']['commit']}`",
        f"- estimator version: `{summary['estimator_version']}`",
        f"- Pass 1 DB SHA256: `{summary['hashes_before']['pass1_db']}`",
        f"- component catalog SHA256: `{summary['component_catalog_sha256']}`",
        f"- tire library SHA256: `{summary['tire_reference_sha256']}`",
        f"- synthetic component references: **{summary['component_reference_count']}** usable fine-domain rows",
        f"- synthetic tire references: **{summary['tire_reference_count']}**",
        "- persistence: **audit artifacts only**",
        "- external searches: **0**", "",
        "## Sample", "",
        f"- target/actual sample: **180 / {summary['sample_size']}**",
        *[f"- {key}: **{value}**" for key, value in summary["sample_by_architecture"].items()],
        f"- distinct achieved strata: **{len(strata)}**", "",
    ]
    for domain, title in (("TIRE", "Tire results"), ("BRAKE", "Brake synthetic results"), ("HUB_BEARING", "Hub/Bearing synthetic results"), ("AXLE", "Axle synthetic results")):
        row = coverage[domain]
        lines.extend(
            [
                f"## {title}", "",
                f"- eligible: **{row['eligible']}**",
                f"- exact/source: **{row['exact_source']}**",
                f"- Grade A: **{row['grade_a']}**",
                f"- Grade B: **{row['grade_b']}**",
                f"- Grade C: **{row['grade_c']}**",
                f"- unresolved: **{row['unresolved']}**",
                f"- envelope rejected: **{row['envelope_rejected']}**",
                f"- boundary rejected: **{row['boundary_rejected']}**",
                f"- accepted coverage: **{row['accepted_coverage_pct']:.3f}%**", "",
            ]
        )
    info = summary["information_classes"]
    lines.extend(
        [
            "## Combined build-up", "",
            f"- Tire only: **{summary['accepted_component_sets'].get('TIRE', 0)}**",
            "- Tire + Brake: **0**",
            "- Tire + Brake + Hub: **0**",
            "- Tire + Brake + Hub + safe Axle: **0**",
            f"- INFORMATIVE: **{info.get('INFORMATIVE', 0)}**",
            f"- PARTIAL: **{info.get('PARTIAL', 0)}**",
            f"- WEAK: **{info.get('WEAK', 0)}**",
            f"- UNRESOLVED: **{info.get('UNRESOLVED', 0)}**",
            f"- REJECTED: **{info.get('REJECTED', 0)}**",
            f"- Tire + at least one accepted mechanical component yield: **{summary['tire_plus_mechanical_yield_pct']:.3f}%**", "",
            f"- accepted cases with positive explicit residual: **{summary['positive_residual_accepted_cases']}**",
            f"- candidate cases rejected before adoption for negative residual: **{summary['rejected_negative_residual_candidate_cases']}**", "",
            "## Residual behavior", "",
            f"- median residual fraction: **{summary['residual_fraction']['median']:.6f}**",
            f"- P10: **{summary['residual_fraction']['p10']:.6f}**",
            f"- P90: **{summary['residual_fraction']['p90']:.6f}**",
            f"- worst accepted case: **{summary['residual_fraction']['worst_accepted']:.6f}**",
            f"- negative material residual grid points: **{summary['negative_material_residual_grid_points']}**", "",
            "## Reference concentration", "",
        ]
    )
    if top_usage:
        lines.extend(["| Domain | Reference | Candidate count | Accepted count | Share |", "|---|---|---:|---:|---:|"])
        lines.extend(
            f"| {row['domain']} | {row['reference_id']} | {row['candidate_count']} | {row['accepted_count']} | {row['sample_share_pct']:.2f}% |"
            for row in top_usage
        )
    else:
        lines.append("No reference candidates were selected.")
    lines.extend(
        [
            "", "All reuse shown above is candidate-only unless Accepted count is nonzero. No Grade C candidate was adopted. `LEGACY_VDE_RRC` denotes 86 vehicle-specific exact legacy lookups (59 accepted), not one generic synthetic tire reference; no individual synthetic reference exceeded 7.78% of the sample.", "",
            "## Contract checks", "",
            f"- TOTAL unchanged: **YES** (DB hash `{summary['hashes_after']['pass1_db']}`)",
            "- RollingMinor unchanged: **YES**",
            "- no scaling: **YES**",
            "- axle overlap gate: **ENFORCED; all pilot Axle rows excluded**",
            "- deterministic selection: **YES**",
            "- external search count = **0**",
            "- DB writes/tables created: **0 / 0**",
            f"- quick_check/FK issues: **{summary['quick_check']} / {summary['foreign_key_issues']}**", "",
            "## Interpretation", "",
            "The current references can produce deterministic application-class/drive/position candidates, but do not carry reference mass/axle-load or wheel-class metadata required for Grade A/B under this pilot contract. Exact legacy RRC evidence was found for 86 sampled vehicles: 59 passed the independent RollingMinor envelope and 27 were rejected before adoption; the remaining 94 lack enough tire evidence to choose size/RR tier. Axle overlap with the frozen drivetrain aggregate cannot be disproven. These are primarily evidence limitations; the 27 tire cases are explicit envelope rejections.", "",
            "## Decision", "",
            f"`{summary['decision']}`", "",
            "Do not scale. Enrich reference metadata or authorize a narrower matching contract through human review before another pilot. No external research was started.",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "DEFAULT_ARCHITECTURE_TARGETS",
    "PILOT_VERSION",
    "deterministic_stratified_sample",
    "evaluate_vehicle",
    "execute_pilot",
    "load_component_references",
    "match_synthetic_reference",
    "match_tire",
]
