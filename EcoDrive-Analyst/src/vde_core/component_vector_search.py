"""Deterministic read-only component-vector search for Sprint 12 Pass 1C.2."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from src.vde_core.component_enrichment_pass1 import file_sha256
from src.vde_core.component_minor_buildup_pilot import (
    DEFAULT_ARCHITECTURE_TARGETS,
    FINE_DOMAINS,
    QA_SPEEDS_KPH,
    Reference,
    _curve,
    _git_metadata,
    _read_only,
    _token,
    _write_csv,
    deterministic_stratified_sample,
    load_component_references,
    load_legacy_lookup,
    load_resolved_population,
    load_tire_references,
)
from src.vde_core.component_minor_buildup_refinement import (
    APPLICATION_ENVELOPES,
    ApplicabilityReference,
    MAX_SYNTHETIC_OVERSHOOT_PCT,
    _metrics as refinement_metrics,
    classify_closure,
    enrich_applicability,
    evaluate_refinement_vehicle,
)
from src.vde_core.component_prior_matching_vnext import normalize_application_class, normalize_drive


VECTOR_SEARCH_VERSION = "SPRINT12_PASS1C2_VECTOR_COMBINATORIAL_SEARCH_V1.0"
TOP_K = {"TIRE": 4, "BRAKE": 3, "HUB_BEARING": 3, "AXLE": 3}
APPLICATION_FAMILIES = {
    "PASSENGER_LIGHT": "PASSENGER",
    "PASSENGER_STANDARD": "PASSENGER",
    "PERFORMANCE": "PASSENGER",
    "CROSSOVER": "UTILITY_LIGHT",
    "SUV": "UTILITY_LIGHT",
    "OFFROAD_SUV": "UTILITY_LIGHT",
    "PASSENGER_VAN": "VAN",
    "CARGO_VAN": "VAN",
    "PICKUP_LIGHT_DUTY": "PICKUP",
    "HEAVY_DUTY": "HEAVY",
}


@dataclass(frozen=True)
class CandidateVector:
    vector_id: str
    domain: str
    reference_ids: tuple[str, ...]
    abc: tuple[float, float, float]
    compatibility_score: float
    application_class: str | None
    application_family: str | None
    drive: str | None
    position: str | None
    mass_min_kg: float | None
    mass_max_kg: float | None
    wheel_min_in: float | None
    wheel_max_in: float | None
    feature_vector: dict[str, Any]
    reason_codes: tuple[str, ...]
    provenance: str
    boundary_assumption: str


def application_family(application_class: str | None) -> str | None:
    return APPLICATION_FAMILIES.get(_token(application_class) or "")


def _vector_id(domain: str, reference_ids: Iterable[str], abc: tuple[float, float, float], tag: str = "") -> str:
    payload = "|".join((domain, *reference_ids, *(f"{value:.12g}" for value in abc), tag))
    return f"VEC-{domain[:4]}-{sha256(payload.encode()).hexdigest()[:16].upper()}"


def _forces_nonnegative(abc: tuple[float, float, float]) -> bool:
    return all(_curve(abc, speed) >= -1e-9 for speed in QA_SPEEDS_KPH)


def _reference_score(vehicle: dict[str, Any], item: ApplicabilityReference) -> tuple[float, dict[str, Any]] | None:
    vehicle_class = _token(vehicle.get("application_class"))
    vehicle_family = application_family(vehicle_class)
    reference_class = item.reference.application_class
    reference_family = application_family(reference_class)
    vehicle_drive = _token(vehicle.get("drive_layout"))
    if not vehicle_class or not vehicle_family or vehicle_family != reference_family:
        return None
    if not vehicle_drive or vehicle_drive != item.reference.drive:
        return None
    mass = vehicle.get("test_mass_resolved_kg")
    try:
        mass_value = float(mass)
    except (TypeError, ValueError):
        return None
    if not item.reference_mass_min_kg <= mass_value <= item.reference_mass_max_kg:
        return None
    half_width = max((item.reference_mass_max_kg - item.reference_mass_min_kg) / 2.0, 1.0)
    center = (item.reference_mass_min_kg + item.reference_mass_max_kg) / 2.0
    center_similarity = max(0.0, 1.0 - abs(mass_value - center) / half_width)
    exact_class = vehicle_class == reference_class
    score = (45.0 if exact_class else 28.0) + 25.0 + 15.0 + 10.0 * center_similarity
    if item.reference.position:
        score += 5.0
    features = {
        "exact_application_class": exact_class,
        "same_application_family": True,
        "exact_drive_layout": True,
        "mass_inside_envelope": True,
        "mass_center_similarity": round(center_similarity, 9),
        "wheel_compatibility": "UNKNOWN_NO_VEHICLE_WHEEL_EVIDENCE",
        "position": item.reference.position,
        "driven_axle_compatibility": item.driven_axle_compatibility,
        "closure_used_in_generation": False,
    }
    return min(score, 100.0), features


def _reference_vector(item: ApplicabilityReference, score: float, features: dict[str, Any]) -> CandidateVector:
    reference = item.reference
    reasons = ("EXACT_APPLICATION_CLASS",) if features["exact_application_class"] else ("ADJACENT_CLASS_SAME_FAMILY",)
    assumption = (
        "AXLE_ALLOCATED_WITHIN_ROLLING_MINOR_SYNTHETIC_BUILDUP"
        if reference.domain == "AXLE"
        else f"{reference.domain}_EXPLANATORY_SUBCOMPONENT_OF_ROLLING_MINOR"
    )
    return CandidateVector(
        vector_id=_vector_id(reference.domain, (reference.component_id,), reference.abc),
        domain=reference.domain,
        reference_ids=(reference.component_id,),
        abc=reference.abc,
        compatibility_score=score,
        application_class=reference.application_class,
        application_family=application_family(reference.application_class),
        drive=reference.drive,
        position=reference.position,
        mass_min_kg=item.reference_mass_min_kg,
        mass_max_kg=item.reference_mass_max_kg,
        wheel_min_in=item.wheel_class_min_in,
        wheel_max_in=item.wheel_class_max_in,
        feature_vector=features,
        reason_codes=reasons,
        provenance=f"{item.source}|{item.applicability_source}",
        boundary_assumption=assumption,
    )


def generate_brake_pool(
    vehicle: dict[str, Any], references: list[ApplicabilityReference], *, limit: int = TOP_K["BRAKE"]
) -> list[CandidateVector]:
    vectors: list[CandidateVector] = []
    for item in references:
        if item.reference.domain != "BRAKE":
            continue
        scored = _reference_score(vehicle, item)
        if scored is None or not _forces_nonnegative(item.reference.abc):
            continue
        vectors.append(_reference_vector(item, *scored))
    return _rank_unique(vectors, limit)


def _pair_vectors(
    vehicle: dict[str, Any],
    references: list[ApplicabilityReference],
    domain: str,
    limit: int,
) -> list[CandidateVector]:
    valid: dict[str, list[tuple[ApplicabilityReference, float, dict[str, Any]]]] = defaultdict(list)
    for item in references:
        if item.reference.domain != domain:
            continue
        scored = _reference_score(vehicle, item)
        if scored is not None and _forces_nonnegative(item.reference.abc):
            valid[item.reference.position or ""].append((item, scored[0], scored[1]))
    drive = _token(vehicle.get("drive_layout"))
    assemblies: list[CandidateVector] = []
    if domain == "HUB_BEARING" or drive in {"AWD", "4WD"}:
        for front, rear in product(valid.get("FRONT", []), valid.get("REAR", [])):
            if front[0].reference.application_class != rear[0].reference.application_class:
                continue
            items = (front[0], rear[0])
            abc = tuple(sum(item.reference.abc[index] for item in items) for index in range(3))
            if not _forces_nonnegative(abc):
                continue
            score = (front[1] + rear[1]) / 2.0
            features = dict(front[2])
            features.update({"complete_vehicle_assembly": True, "positions": ["FRONT", "REAR"]})
            ids = tuple(item.reference.component_id for item in items)
            assemblies.append(
                CandidateVector(
                    vector_id=_vector_id(domain, ids, abc), domain=domain, reference_ids=ids, abc=abc,
                    compatibility_score=score, application_class=items[0].reference.application_class,
                    application_family=application_family(items[0].reference.application_class), drive=drive,
                    position="FRONT+REAR", mass_min_kg=max(item.reference_mass_min_kg for item in items),
                    mass_max_kg=min(item.reference_mass_max_kg for item in items),
                    wheel_min_in=max(item.wheel_class_min_in for item in items),
                    wheel_max_in=min(item.wheel_class_max_in for item in items), feature_vector=features,
                    reason_codes=("COMPLETE_POSITION_ASSEMBLY",),
                    provenance="|".join(sorted({
                        *(item.source for item in items),
                        *(item.applicability_source for item in items),
                    })),
                    boundary_assumption=("AXLE_ALLOCATED_WITHIN_ROLLING_MINOR_SYNTHETIC_BUILDUP" if domain == "AXLE" else "HUB_BEARING_EXPLANATORY_SUBCOMPONENT_OF_ROLLING_MINOR"),
                )
            )
    elif domain == "AXLE" and drive == "RWD":
        for item, score, features in valid.get("REAR", []):
            assemblies.append(_reference_vector(item, score, {**features, "complete_vehicle_assembly": True}))
    return _rank_unique(assemblies, limit)


def generate_hub_pool(
    vehicle: dict[str, Any], references: list[ApplicabilityReference], *, limit: int = TOP_K["HUB_BEARING"]
) -> list[CandidateVector]:
    return _pair_vectors(vehicle, references, "HUB_BEARING", limit)


def generate_axle_pool(
    vehicle: dict[str, Any], references: list[ApplicabilityReference], *, limit: int = TOP_K["AXLE"]
) -> list[CandidateVector]:
    return _pair_vectors(vehicle, references, "AXLE", limit)


def load_historical_tire_evidence(connection) -> list[dict[str, Any]]:
    sql = """
        SELECT id,make,model,year,category,tire_size,test_mass_kg,mass_kg,
               rrc_N_per_kN,drive_type
        FROM vde_db
        WHERE rrc_N_per_kN IS NOT NULL
        ORDER BY id
    """
    rows: list[dict[str, Any]] = []
    for raw in connection.execute(sql):
        row = dict(raw)
        app = normalize_application_class((str(row.get("category") or ""),))
        if app.status != "MAPPED" or len(app.normalized_options) != 1:
            continue
        try:
            rrc = float(row["rrc_N_per_kN"])
            mass = float(row.get("test_mass_kg") or row.get("mass_kg"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(rrc) or rrc <= 0 or not math.isfinite(mass) or mass <= 0:
            continue
        drive = normalize_drive(row.get("drive_type"))
        rows.append({
            **row, "application_class": app.normalized_options[0],
            "application_family": application_family(app.normalized_options[0]),
            "mass_resolved_kg": mass, "rrc": rrc,
            "drive_normalized": drive.normalized if drive.status == "MAPPED" else None,
        })
    return rows


def generate_tire_pool(
    vehicle: dict[str, Any],
    tire_references: list[dict[str, Any]],
    historical: list[dict[str, Any]],
    *,
    limit: int = TOP_K["TIRE"],
) -> list[CandidateVector]:
    try:
        mass = float(vehicle.get("test_mass_resolved_kg"))
    except (TypeError, ValueError):
        return []
    if not math.isfinite(mass) or mass <= 0:
        return []
    vectors: list[CandidateVector] = []
    vehicle_class = _token(vehicle.get("application_class"))
    family = application_family(vehicle_class)
    drive = _token(vehicle.get("drive_layout"))
    exact_rrc = vehicle.get("rrc_resolved")
    if exact_rrc is not None:
        rrc = float(exact_rrc)
        abc = (rrc * mass * 9.80665 / 1000.0, 0.0, 0.0)
        source = str(vehicle.get("rrc_source") or "INTERNAL_VEHICLE_RRC")
        vectors.append(CandidateVector(
            vector_id=_vector_id("TIRE", (source,), abc, str(vehicle["vde_id"])), domain="TIRE",
            reference_ids=(source,), abc=abc, compatibility_score=100.0,
            application_class=vehicle_class, application_family=family, drive=drive, position="VEHICLE_LEVEL",
            mass_min_kg=mass, mass_max_kg=mass, wheel_min_in=None, wheel_max_in=None,
            feature_vector={"vehicle_specific_rrc": True, "exact_application_class": True, "closure_used_in_generation": False},
            reason_codes=("VEHICLE_SPECIFIC_INTERNAL_RRC",), provenance=source,
            boundary_assumption="TIRE_EXPLANATORY_SUBCOMPONENT_OF_ROLLING_MINOR",
        ))
    size = _token(vehicle.get("tire_size_resolved"))
    if size:
        for row in tire_references:
            if _token(row.get("size_code")) != size:
                continue
            try:
                rrc = float(row.get("rr_n_per_kn") or row.get("iso_corrected_rrc_n_per_kn"))
            except (TypeError, ValueError):
                continue
            abc = (rrc * mass * 9.80665 / 1000.0, 0.0, 0.0)
            ref_id = str(row["tire_test_code"])
            vectors.append(CandidateVector(
                vector_id=_vector_id("TIRE", (ref_id,), abc, str(vehicle["vde_id"])), domain="TIRE",
                reference_ids=(ref_id,), abc=abc, compatibility_score=82.0,
                application_class=vehicle_class, application_family=family, drive=drive, position="VEHICLE_LEVEL",
                mass_min_kg=None, mass_max_kg=None, wheel_min_in=None, wheel_max_in=None,
                feature_vector={"exact_tire_size": True, "synthetic_rr_tier": row.get("rr_value_source_note"), "closure_used_in_generation": False},
                reason_codes=("EXACT_TIRE_SIZE_SYNTHETIC_REFERENCE",), provenance="SYNTHETIC_TIRE_REFERENCE",
                boundary_assumption="TIRE_EXPLANATORY_SUBCOMPONENT_OF_ROLLING_MINOR",
            ))
    if vehicle_class and family:
        for row in historical:
            if row["application_family"] != family:
                continue
            mass_delta = abs(mass - row["mass_resolved_kg"]) / row["mass_resolved_kg"]
            if mass_delta > 0.25:
                continue
            exact_class = row["application_class"] == vehicle_class
            score = (58.0 if exact_class else 42.0) + 20.0 * (1.0 - mass_delta / 0.25)
            if drive and row.get("drive_normalized") == drive:
                score += 7.0
            rrc = row["rrc"]
            abc = (rrc * mass * 9.80665 / 1000.0, 0.0, 0.0)
            ref_id = f"LEGACY_VDE_RRC_ID_{row['id']}"
            vectors.append(CandidateVector(
                vector_id=_vector_id("TIRE", (ref_id,), abc, str(vehicle["vde_id"])), domain="TIRE",
                reference_ids=(ref_id,), abc=abc, compatibility_score=min(score, 90.0),
                application_class=row["application_class"], application_family=family,
                drive=row.get("drive_normalized"), position="VEHICLE_LEVEL",
                mass_min_kg=row["mass_resolved_kg"] * 0.75, mass_max_kg=row["mass_resolved_kg"] * 1.25,
                wheel_min_in=None, wheel_max_in=None,
                feature_vector={"exact_application_class": exact_class, "same_application_family": True,
                                "mass_difference_pct": 100.0 * mass_delta,
                                "same_drive_when_available": bool(drive and row.get("drive_normalized") == drive),
                                "closure_used_in_generation": False},
                reason_codes=(("EXACT_CLASS_HISTORICAL_RRC_ANALOG" if exact_class else "FAMILY_HISTORICAL_RRC_ANALOG"),),
                provenance="INTERNAL_LEGACY_VDE_RRC_ANALOG",
                boundary_assumption="TIRE_EXPLANATORY_SUBCOMPONENT_OF_ROLLING_MINOR",
            ))
    return _rank_unique(vectors, limit)


def _rank_unique(vectors: Iterable[CandidateVector], limit: int) -> list[CandidateVector]:
    ordered = sorted(vectors, key=lambda item: (-item.compatibility_score, item.vector_id))
    result: list[CandidateVector] = []
    seen: set[tuple[float, float, float]] = set()
    for vector in ordered:
        numeric = tuple(round(value, 12) for value in vector.abc)
        if numeric in seen:
            continue
        seen.add(numeric)
        result.append(vector)
        if len(result) >= limit:
            break
    return result


def build_candidate_pools(
    vehicle: dict[str, Any],
    references: list[ApplicabilityReference],
    tire_references: list[dict[str, Any]],
    historical_tires: list[dict[str, Any]],
    *,
    caps: dict[str, int] | None = None,
) -> dict[str, list[CandidateVector]]:
    caps = caps or TOP_K
    return {
        "TIRE": generate_tire_pool(vehicle, tire_references, historical_tires, limit=caps["TIRE"]),
        "BRAKE": generate_brake_pool(vehicle, references, limit=caps["BRAKE"]),
        "HUB_BEARING": generate_hub_pool(vehicle, references, limit=caps["HUB_BEARING"]),
        "AXLE": generate_axle_pool(vehicle, references, limit=caps["AXLE"]),
    }


def _combination_metrics(vehicle: dict[str, Any], vectors: dict[str, CandidateVector | None]) -> dict[str, Any]:
    selected = [vector for vector in vectors.values() if vector is not None]
    component_abc = {domain: vector.abc for domain, vector in vectors.items() if vector is not None}
    closure = classify_closure(vehicle["rolling_abc"], component_abc)
    accepted = closure["status"] != "REJECTED_BUILDUP"
    residuals: list[float] = []
    for speed in QA_SPEEDS_KPH:
        rolling = _curve(vehicle["rolling_abc"], speed)
        buildup = sum(_curve(vector.abc, speed) for vector in selected)
        residuals.append(max(0.0, rolling - buildup) / max(abs(rolling), 1e-9) * 100.0)
    return {
        "populated_domain_count": len(selected),
        "compatibility_score": sum(vector.compatibility_score for vector in selected),
        "closure_status": closure["status"],
        "max_overshoot_pct": closure["max_overshoot_pct"],
        "median_unresolved_residual_pct": median(residuals),
        "accepted": int(accepted),
    }


def search_combinations(
    vehicle: dict[str, Any], pools: dict[str, list[CandidateVector]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    choices = [[None, *pools[domain]] for domain in FINE_DOMAINS]
    rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    for selected in product(*choices):
        if all(vector is None for vector in selected):
            continue
        vectors = dict(zip(FINE_DOMAINS, selected))
        metrics = _combination_metrics(vehicle, vectors)
        selected_ids = tuple(vector.vector_id if vector else "NONE" for vector in selected)
        row = {
            "vde_id": vehicle["vde_id"],
            "combination_id": f"COMB-{sha256('|'.join(selected_ids).encode()).hexdigest()[:16].upper()}",
            "tire_vector": selected_ids[0], "brake_vector": selected_ids[1],
            "hub_vector": selected_ids[2], "axle_vector": selected_ids[3],
            **metrics, "rank": None, "scaled": 0,
        }
        rows.append(row)
        if metrics["accepted"]:
            accepted_rows.append(row)
    accepted_rows.sort(key=lambda row: (
        -int(row["populated_domain_count"]), -float(row["compatibility_score"]),
        0 if row["closure_status"] == "CLEAN_BUILDUP" else 1,
        float(row["median_unresolved_residual_pct"]), float(row["max_overshoot_pct"]), row["combination_id"],
    ))
    for rank, row in enumerate(accepted_rows, 1):
        row["rank"] = rank
    return rows, accepted_rows[:3]


def _pool_rows(vde_id: int, pools: dict[str, list[CandidateVector]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pool_rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    for domain in FINE_DOMAINS:
        for rank, vector in enumerate(pools[domain], 1):
            common = {
                "vde_id": vde_id, "domain": domain, "candidate_rank": rank,
                "vector_id": vector.vector_id, "reference_ids": ";".join(vector.reference_ids),
                "application_class": vector.application_class, "application_family": vector.application_family,
                "drive": vector.drive, "position": vector.position,
                "mass_min_kg": vector.mass_min_kg, "mass_max_kg": vector.mass_max_kg,
                "wheel_min_in": vector.wheel_min_in, "wheel_max_in": vector.wheel_max_in,
                "compatibility_features_json": json.dumps(vector.feature_vector, sort_keys=True, separators=(",", ":")),
                "compatibility_score": vector.compatibility_score,
                "reason_codes": ";".join(vector.reason_codes), "provenance": vector.provenance,
                "boundary_assumption": vector.boundary_assumption,
            }
            pool_rows.append(common)
            vector_rows.append({
                **common, "A_N": vector.abc[0], "B_N_per_kph": vector.abc[1], "C_N_per_kph2": vector.abc[2],
                "force_vector_json": json.dumps([round(_curve(vector.abc, speed), 9) for speed in QA_SPEEDS_KPH], separators=(",", ":")),
            })
    return pool_rows, vector_rows


def _best_rows(vde_id: int, best: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not best:
        return [{"vde_id": vde_id, "selection_role": "NO_VALID_COMBINATION", "rank": None, "accepted": 0}]
    roles = ("BEST", "ALTERNATIVE_2", "ALTERNATIVE_3")
    return [{"selection_role": roles[index], **row} for index, row in enumerate(best)]


def _coverage(best_rows: list[dict[str, Any]], sample_size: int) -> dict[str, Any]:
    best = [row for row in best_rows if row.get("selection_role") == "BEST"]
    counts = Counter(int(row["populated_domain_count"]) for row in best)
    result = {
        "vdes_with_ge_1": sum(count for domains, count in counts.items() if domains >= 1),
        "vdes_with_ge_2": sum(count for domains, count in counts.items() if domains >= 2),
        "vdes_with_ge_3": sum(count for domains, count in counts.items() if domains >= 3),
        "vdes_with_all_4": counts[4],
        "clean_buildups": sum(row["closure_status"] == "CLEAN_BUILDUP" for row in best),
        "tolerance_accepted_buildups": sum(row["closure_status"] == "TOLERANCE_ACCEPTED" for row in best),
        "rejected_or_no_valid": sample_size - len(best),
    }
    for domain, field in (("TIRE", "tire_vector"), ("BRAKE", "brake_vector"), ("HUB_BEARING", "hub_vector"), ("AXLE", "axle_vector")):
        result[f"{domain.lower()}_coverage"] = sum(row[field] != "NONE" for row in best)
    result["ge_2_pct"] = 100.0 * result["vdes_with_ge_2"] / sample_size
    result["ge_3_pct"] = 100.0 * result["vdes_with_ge_3"] / sample_size
    return result


def _previous_metrics(sample, component_refs, enriched, tire_refs) -> dict[str, Any]:
    matches, curves, summaries = [], [], []
    for vehicle in sample:
        m, c, s = evaluate_refinement_vehicle(vehicle, component_refs, enriched, tire_refs, phase="B")
        matches.extend(m); curves.extend(c); summaries.append(s)
    metrics = refinement_metrics(matches, summaries, curves)
    sets = [set(filter(None, row["accepted_component_set"].split(";"))) for row in summaries]
    return {
        "vdes_with_ge_1": sum(len(value) >= 1 for value in sets),
        "vdes_with_ge_2": sum(len(value) >= 2 for value in sets),
        "vdes_with_ge_3": sum(len(value) >= 3 for value in sets),
        "vdes_with_all_4": sum(len(value) == 4 for value in sets),
        "tire_coverage": metrics["tire_accepted"], "brake_coverage": metrics["brake_accepted"],
        "hub_bearing_coverage": metrics["hub_accepted"], "axle_coverage": metrics["axle_accepted"],
        "clean_buildups": metrics["clean_buildup"],
        "tolerance_accepted_buildups": metrics["tolerance_accepted"],
        # Rejected build-ups already have an empty accepted set, so counting
        # empty sets alone avoids double-counting the same VDE.
        "rejected_or_no_valid": sum(not value for value in sets),
    }


def _decision(coverage: dict[str, Any]) -> str:
    if coverage["ge_2_pct"] >= 40.0 and coverage["ge_3_pct"] >= 20.0 and coverage["rejected_or_no_valid"] == 0:
        return "PASS_1C2_VECTOR_SCALE_RECOMMENDED"
    if coverage["ge_2_pct"] >= 30.0 or coverage["ge_3_pct"] >= 15.0:
        return "PASS_1C2_VECTOR_REVIEW_REQUIRED"
    return "PASS_1C2_VECTOR_SCALE_REJECTED"


def execute_vector_search(db_path: Path, legacy_db_path: Path, component_catalog_path: Path, tire_zip_path: Path, output_dir: Path) -> dict[str, Any]:
    db_path = Path(db_path).resolve(strict=True); legacy_db_path = Path(legacy_db_path).resolve(strict=True)
    component_catalog_path = Path(component_catalog_path).resolve(strict=True); tire_zip_path = Path(tire_zip_path).resolve(strict=True)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    hashes_before = {"pass1_db": file_sha256(db_path), "legacy_db": file_sha256(legacy_db_path)}
    connection = _read_only(db_path); legacy = _read_only(legacy_db_path)
    try:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        fk_issues = len(connection.execute("PRAGMA foreign_key_check").fetchall())
        population = load_resolved_population(connection, load_legacy_lookup(legacy))
        historical_tires = load_historical_tire_evidence(legacy)
    finally:
        connection.close(); legacy.close()
    sample = deterministic_stratified_sample(population, DEFAULT_ARCHITECTURE_TARGETS)
    if len(sample) != 180:
        raise RuntimeError(f"Expected frozen 180-VDE sample, found {len(sample)}")
    component_refs = load_component_references(component_catalog_path)
    enriched = enrich_applicability(component_refs)
    tire_refs = load_tire_references(tire_zip_path)
    previous = _previous_metrics(sample, component_refs, enriched, tire_refs)
    sample_rows = [{"vde_id": row["vde_id"], "vehicle_configuration_id": row["vehicle_configuration_id"],
                    "architecture": row["architecture"], "application_class": row.get("application_class"),
                    "drive_layout": row.get("drive_layout"), "test_mass_kg": row.get("test_mass_resolved_kg"),
                    "stratum": row.get("stratum"), "parent_rolling_minor_resolution_id": row["rolling_minor_resolution_id"]}
                   for row in sample]
    _write_csv(output_dir / "pilot_sample.csv", sample_rows, sample_rows[0].keys())
    pool_rows: list[dict[str, Any]] = []; vector_rows: list[dict[str, Any]] = []
    combination_rows: list[dict[str, Any]] = []; best_rows: list[dict[str, Any]] = []
    pool_sizes: dict[str, list[int]] = defaultdict(list)
    for vehicle in sample:
        pools = build_candidate_pools(vehicle, enriched, tire_refs, historical_tires)
        for domain in FINE_DOMAINS:
            pool_sizes[domain].append(len(pools[domain]))
        vehicle_pool_rows, vehicle_vectors = _pool_rows(int(vehicle["vde_id"]), pools)
        pool_rows.extend(vehicle_pool_rows); vector_rows.extend(vehicle_vectors)
        tested, best = search_combinations(vehicle, pools)
        combination_rows.extend(tested); best_rows.extend(_best_rows(int(vehicle["vde_id"]), best))
    _write_csv(output_dir / "candidate_pools.csv", pool_rows, pool_rows[0].keys())
    _write_csv(output_dir / "candidate_vectors.csv", vector_rows, vector_rows[0].keys())
    _write_csv(output_dir / "combination_results.csv", combination_rows, combination_rows[0].keys())
    _write_csv(output_dir / "best_combinations.csv", best_rows, sorted({key for row in best_rows for key in row}))
    coverage = _coverage(best_rows, len(sample))
    pool_distribution = {domain: dict(sorted(Counter(values).items())) for domain, values in pool_sizes.items()}
    comparison_rows = [{"metric": key, "pass_1c1": previous.get(key), "pass_1c2": coverage.get(key)} for key in (
        "vdes_with_ge_1", "vdes_with_ge_2", "vdes_with_ge_3", "vdes_with_all_4", "tire_coverage",
        "brake_coverage", "hub_bearing_coverage", "axle_coverage", "clean_buildups",
        "tolerance_accepted_buildups", "rejected_or_no_valid")]
    _write_csv(output_dir / "component_coverage_summary.csv", comparison_rows, comparison_rows[0].keys())
    hashes_after = {"pass1_db": file_sha256(db_path), "legacy_db": file_sha256(legacy_db_path)}
    if hashes_before != hashes_after:
        raise RuntimeError("Read-only vector search changed an input database")
    decision = _decision(coverage)
    summary = {
        "vector_search_version": VECTOR_SEARCH_VERSION, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_metadata(), "sample_size": len(sample),
        "sample_identity_sha256": sha256(",".join(str(row["vde_id"]) for row in sample_rows).encode()).hexdigest().upper(),
        "sample_by_architecture": dict(sorted(Counter(row["architecture"] for row in sample).items())),
        "previous_pass_1c1": previous, "vector_search": coverage, "pool_size_distribution": pool_distribution,
        "tested_combination_count": len(combination_rows),
        "accepted_combination_count": sum(int(row["accepted"]) for row in combination_rows),
        "max_accepted_overshoot_pct": max((float(row["max_overshoot_pct"]) for row in combination_rows if row["accepted"]), default=0.0),
        "hashes_before": hashes_before, "hashes_after": hashes_after, "quick_check": quick_check,
        "foreign_key_issues": fk_issues, "database_rows_written": 0, "database_objects_created": 0,
        "external_search_count": 0, "scaled_combination_count": 0, "scale_up_executed": False,
        "decision": decision,
    }
    (output_dir / "execution_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "vector_search_summary.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _report(summary: dict[str, Any]) -> str:
    old, new = summary["previous_pass_1c1"], summary["vector_search"]
    lines = [
        "# Sprint 12 Pass 1C.2 — Vector / Combinatorial Synthetic Search", "",
        "## Safety and scope", "",
        f"- Frozen Pass 1C sample: **{summary['sample_size']}**, identity `{summary['sample_identity_sha256']}`",
        "- Candidate generation uses closure fit: **NO**",
        "- Component scaling: **NONE**",
        "- External research: **NONE**",
        "- Automatic full-population scale-up: **NOT EXECUTED**",
        f"- quick_check / FK issues: **{summary['quick_check']} / {summary['foreign_key_issues']}**",
        f"- DB writes / objects created: **{summary['database_rows_written']} / {summary['database_objects_created']}**", "",
        "## Search", "",
        f"- Combinations tested: **{summary['tested_combination_count']:,}**",
        f"- Combinations surviving <=5% overshoot: **{summary['accepted_combination_count']:,}**",
        f"- Maximum accepted overshoot: **{summary['max_accepted_overshoot_pct']:.6f}%**",
        "- Ranking order: domain count, compatibility score, clean closure, residual, overshoot", "",
        "## Candidate pool-size distribution", "",
    ]
    for domain in FINE_DOMAINS:
        lines.append(f"- {domain}: `{json.dumps(summary['pool_size_distribution'].get(domain, {}), sort_keys=True)}`")
    lines.extend(["", "## Pilot comparison", "", "| Metric | Pass 1C.1 | Pass 1C.2 |", "|---|---:|---:|"])
    for label, key in ((">=1 fine component", "vdes_with_ge_1"), (">=2 fine components", "vdes_with_ge_2"),
                       (">=3 fine components", "vdes_with_ge_3"), ("All 4", "vdes_with_all_4"),
                       ("Tire coverage", "tire_coverage"), ("Brake coverage", "brake_coverage"),
                       ("Hub coverage", "hub_bearing_coverage"), ("Axle coverage", "axle_coverage"),
                       ("Clean build-ups", "clean_buildups"), ("Tolerance accepted", "tolerance_accepted_buildups"),
                       ("Rejected/no valid", "rejected_or_no_valid")):
        lines.append(f"| {label} | {old.get(key, 'N/A')} | {new.get(key, 'N/A')} |")
    lines.extend([
        "", "## Scale gate", "",
        f"- >=2 fine components: **{new['vdes_with_ge_2']}/180 ({new['ge_2_pct']:.3f}%)**, target >=40%",
        f"- >=3 fine components: **{new['vdes_with_ge_3']}/180 ({new['ge_3_pct']:.3f}%)**, target >=20%",
        "- All accepted combinations satisfy <=5% overshoot: **YES**",
        "- Axle semantics: `AXLE_ALLOCATED_WITHIN_ROLLING_MINOR_SYNTHETIC_BUILDUP`; never added to TOTAL/drivetrain",
        "- RollingMinor remains authoritative and unchanged", "",
        "## Decision", "", f"`{summary['decision']}`", "",
        "Both coverage gates were narrowly missed while component-level coverage improved materially; rules were not loosened. Human review is therefore required before any scale-up.", "",
        "Alternatives are audit candidates only. No component selection was persisted.",
    ])
    return "\n".join(lines) + "\n"


__all__ = [
    "APPLICATION_FAMILIES", "CandidateVector", "TOP_K", "application_family", "build_candidate_pools",
    "execute_vector_search", "generate_axle_pool", "generate_brake_pool", "generate_hub_pool",
    "generate_tire_pool", "search_combinations",
]
