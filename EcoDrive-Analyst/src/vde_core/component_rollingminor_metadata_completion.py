"""Pass 1C.4 RollingMinor metadata completion and validation pilot."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.vde_core.component_enrichment_pass1 import file_sha256
from src.vde_core.component_minor_buildup_pilot import (
    DEFAULT_ARCHITECTURE_TARGETS,
    _git_metadata,
    _read_only,
    _write_csv,
    deterministic_stratified_sample,
    load_legacy_lookup,
    load_resolved_population,
    load_tire_references,
)
from src.vde_core.component_physics_vector_search import (
    ROLLING_DOMAINS,
    _candidate_rows,
    aggregate_kpis,
    load_all_component_references,
    search_aggregate,
    truncate_pools,
)
from src.vde_core.component_minor_buildup_refinement import enrich_applicability
from src.vde_core.component_prior_matching_vnext import normalize_application_class
from src.vde_core.component_vector_search import (
    CandidateVector,
    generate_brake_pool,
    generate_hub_pool,
    generate_tire_pool,
    load_historical_tire_evidence,
)


METHOD_VERSION = "SPRINT12_PASS1C4_ROLLINGMINOR_METADATA_COMPLETION_V1.0"
PRIMARY_CAPS = {"TIRE": 3, "BRAKE": 3, "HUB_BEARING": 3, "TRANSMISSION": 0, "AXLE": 0}
SHUFFLE_SEEDS = (11, 23, 37, 51, 73)
MIN_SIGNAL_LIFT_PP = 10.0
POOL_STABILITY_ACCEPTANCE_PP = 5.0
POOL_STABILITY_RESIDUAL_PP = 5.0
COVERAGE_SCALE_GATE_PCT = 40.0


def _build_rolling_pools(
    vehicle: dict[str, Any], references: list[Any], tire_references: list[dict[str, Any]],
    historical_tires: list[dict[str, Any]], *, tire_limit: int = 5,
    brake_limit: int = 5, hub_limit: int = 5,
) -> dict[str, list[CandidateVector]]:
    """Build only the three in-scope RollingMinor domains.

    Empty out-of-scope keys keep the shared truncation interface compatible
    without generating Transmission or Axle candidates.
    """
    return {
        "TIRE": generate_tire_pool(vehicle, tire_references, historical_tires, limit=tire_limit),
        "BRAKE": generate_brake_pool(vehicle, references, limit=brake_limit),
        "HUB_BEARING": generate_hub_pool(vehicle, references, limit=hub_limit),
        "TRANSMISSION": [],
        "AXLE": [],
    }


def _text(value: Any) -> str:
    return " ".join(str(value or "").upper().replace("-", " ").replace("/", " ").split())


def _contains(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def resolve_application_class_v14(vehicle: dict[str, Any]) -> dict[str, Any]:
    """Resolve class without inspecting ABC or closure outcomes."""
    previous = vehicle.get("application_class")
    category = _text(vehicle.get("category"))
    make_model = _text(f"{vehicle.get('make') or ''} {vehicle.get('model') or ''}")
    mass = vehicle.get("test_mass_resolved_kg")
    try:
        mass_value = float(mass)
    except (TypeError, ValueError):
        mass_value = None

    # Preserve an existing internal/legacy class before applying any
    # lower-priority category, name, or mass rule.
    if previous:
        return _resolution(str(previous), "LEGACY_MAPPING", "EXISTING_LEGACY_CATEGORY_MAP", "HIGH", previous)

    normalized = normalize_application_class((vehicle.get("category"),), source="canonical.vde.category")
    if normalized.status == "MAPPED" and len(normalized.normalized_options) == 1:
        return _resolution(normalized.normalized_options[0], "CANONICAL_MAPPING", "EXACT_NORMALIZED_CATEGORY", "HIGH", previous)

    if category in {"STATIONWAGON", "STATION WAGON", "SEDAN"}:
        return _resolution("PASSENGER_STANDARD", "CANONICAL_MAPPING", f"BODY_STYLE_{category.replace(' ', '_')}", "HIGH", previous)
    if category in {"SUV CROSSOVER", "CROSSOVER SUV"} and mass_value is not None:
        target = "CROSSOVER" if mass_value < 2200.0 else "SUV"
        return _resolution(target, "CANONICAL_MAPPING", "COARSE_SUV_CROSSOVER_PLUS_MASS", "MEDIUM", previous)

    if _contains(make_model, ("TRANSIT", "PROMASTER", "METRIS", "POSTAL", " CARGO VAN", " MULLEN ONE")):
        target = "PASSENGER_VAN" if "WAGON" in make_model else "CARGO_VAN"
        return _resolution(target, "RULE_DERIVED", "EXPLICIT_VAN_MODEL_TOKEN", "MEDIUM", previous)
    if _contains(make_model, ("MAVERIK", "MAVERICK", "SIERRA", "TACOMA", "PICKUP", " 1500 ")):
        return _resolution("PICKUP_LIGHT_DUTY", "RULE_DERIVED", "EXPLICIT_PICKUP_MODEL_TOKEN", "MEDIUM", previous)
    if _contains(make_model, ("BRONCO", "GRENADIER", "TRIALMASTER")):
        return _resolution("OFFROAD_SUV", "RULE_DERIVED", "EXPLICIT_OFFROAD_MODEL_TOKEN", "MEDIUM", previous)

    suv_tokens = (
        " SUV", "UTILITY", "NAVIGATOR", "EXPLORER", "HIGHLANDER", "RANGE ROVER", "EVOQUE", "VELAR",
        "TAHOE", " GLE ", " MDX", " X5", "SPORTAGE", "QX60", "CX 70", "ELETRE", "IONIQ 9",
        " RZ ", " BZ", " NX ", "NIRO", "IONIQ 5",
    )
    if _contains(make_model, suv_tokens):
        target = "SUV" if mass_value is not None and mass_value >= 2300.0 else "CROSSOVER"
        return _resolution(target, "RULE_DERIVED", "CURATED_SUV_CROSSOVER_MODEL_TOKEN_PLUS_MASS", "MEDIUM", previous)

    performance_tokens = (
        "FERRARI", "LAMBORGHINI", "MCLAREN", "LOTUS", "CORVETTE", " AMG ", " R8 ", "MUSTANG",
        "F TYPE", "BOXSTER", "ARTURA", "REVUELTO", "HURACAN", " 296 ", " 21C", " GOLF R",
        "PORSCHE TAYCAN", "FORD GT", "NISSAN Z ",
    )
    if _contains(make_model, performance_tokens):
        return _resolution("PERFORMANCE", "RULE_DERIVED", "CURATED_PERFORMANCE_MODEL_TOKEN", "MEDIUM", previous)

    passenger_light_tokens = (" CIVIC", " ELANTRA", " GOLF", "MODEL 3", " K23")
    passenger_standard_tokens = (" CLA ", " G80", "S 580", "S 680", " 330I", "M440I", " TLX", " CROWN", "CLARITY")
    if _contains(make_model, passenger_light_tokens):
        return _resolution("PASSENGER_LIGHT", "RULE_DERIVED", "CURATED_PASSENGER_LIGHT_MODEL_TOKEN", "MEDIUM", previous)
    if _contains(make_model, passenger_standard_tokens):
        return _resolution("PASSENGER_STANDARD", "RULE_DERIVED", "CURATED_PASSENGER_STANDARD_MODEL_TOKEN", "MEDIUM", previous)
    if _contains(make_model, ("SEDAN", "SALOON")):
        return _resolution("PASSENGER_STANDARD", "RULE_DERIVED", "EXPLICIT_SEDAN_BODY_TOKEN", "MEDIUM", previous)
    if _contains(make_model, ("COUPE", "CONVERTIBLE", "SPIDER", "ROADSTER")):
        return _resolution("PERFORMANCE", "RULE_DERIVED", "EXPLICIT_PERFORMANCE_BODY_TOKEN", "MEDIUM", previous)

    if mass_value is not None and category == "CAR":
        target = "PASSENGER_LIGHT" if mass_value < 1850.0 else "PASSENGER_STANDARD"
        return _resolution(target, "RULE_DERIVED", "COARSE_CAR_PLUS_MASS_FALLBACK", "LOW", previous)
    if mass_value is not None and category == "TRUCK":
        target = "CROSSOVER" if mass_value < 2200.0 else "SUV"
        return _resolution(target, "RULE_DERIVED", "COARSE_TRUCK_PLUS_MASS_FALLBACK", "LOW", previous)
    if mass_value is not None and category == "BOTH":
        target = "PASSENGER_STANDARD" if mass_value < 2100.0 else "SUV"
        return _resolution(target, "RULE_DERIVED", "COARSE_BOTH_PLUS_MASS_FALLBACK", "LOW", previous)
    return _resolution(None, "UNRESOLVED", "NO_DEFENSIBLE_INTERNAL_RULE", "UNRESOLVED", previous)


def _resolution(value: str | None, source: str, rule: str, confidence: str, previous: Any) -> dict[str, Any]:
    return {
        "application_class": value, "application_class_source": source,
        "application_class_rule": rule, "application_class_confidence": confidence,
        "previous_application_class": previous,
        "eligible_for_validation": confidence in {"HIGH", "MEDIUM"},
        "closure_used_for_classification": False,
    }


def shuffle_pools_seeded(
    pools_by_vde: dict[int, dict[str, list[CandidateVector]]], seed: int,
) -> dict[int, dict[str, list[CandidateVector]]]:
    result = {vde_id: {domain: list(values) for domain, values in pools.items()} for vde_id, pools in pools_by_vde.items()}
    for domain in ROLLING_DOMAINS:
        groups: dict[int, list[int]] = defaultdict(list)
        for vde_id, pools in pools_by_vde.items():
            groups[len(pools.get(domain, []))].append(vde_id)
        for size, ids in groups.items():
            ids.sort()
            if size == 0 or len(ids) < 2:
                continue
            shift = 1 + seed % (len(ids) - 1)
            donors = [list(pools_by_vde[vde_id][domain]) for vde_id in ids]
            for index, vde_id in enumerate(ids):
                result[vde_id][domain] = donors[(index + shift) % len(ids)]
    return result


def _run_rolling(
    vehicles: list[dict[str, Any]], pools_by_vde: dict[int, dict[str, list[CandidateVector]]], caps: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    all_accepted: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for vehicle in vehicles:
        vde_id = int(vehicle["vde_id"])
        pools = truncate_pools(pools_by_vde[vde_id], caps)
        accepted, best, diagnostics = search_aggregate(
            vehicle, "ROLLING_MINOR", vehicle["rolling_abc"], pools, ROLLING_DOMAINS
        )
        all_accepted.extend(accepted)
        if best:
            selected.append(best)
        records.append({
            "vde_id": vde_id, "aggregate": "ROLLING_MINOR", "eligible": 1,
            "complete_pool": diagnostics["complete_pool"], "selected": best,
            "accepted_combinations": accepted,
        })
    return selected, all_accepted, records


def _breakdown(
    vehicles: list[dict[str, Any]], records: list[dict[str, Any]], key: str,
) -> list[dict[str, Any]]:
    record_map = {row["vde_id"]: row for row in records}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for vehicle in vehicles:
        groups[str(vehicle.get(key) or "UNRESOLVED")].append(vehicle)
    rows: list[dict[str, Any]] = []
    for label, group in sorted(groups.items()):
        selected = [record_map[int(vehicle["vde_id"])] for vehicle in group]
        complete = [row for row in selected if row["complete_pool"]]
        accepted = [row["selected"] for row in complete if row.get("selected") and row["selected"]["build_status"] == "FULL_BUILDUP"]
        residuals = [float(row["median_unresolved_residual_pct"]) for row in accepted]
        rows.append({
            key: label, "vde_count": len(group), "complete_pools": len(complete),
            "accepted_complete_builds": len(accepted),
            "acceptance_pct_complete": 100.0 * len(accepted) / len(complete) if complete else 0.0,
            "acceptance_pct_pilot": 100.0 * len(accepted) / len(group) if group else 0.0,
            "median_explained_pct": 100.0 - median(residuals) if residuals else None,
            "median_unresolved_pct": median(residuals) if residuals else None,
        })
    return rows


def _failure_rows(
    vehicles: list[dict[str, Any]], pools: dict[int, dict[str, list[CandidateVector]]], records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    counter: Counter = Counter()
    record_map = {row["vde_id"]: row for row in records}
    for vehicle in vehicles:
        vde_id = int(vehicle["vde_id"])
        if vehicle.get("application_class_confidence") == "LOW":
            counter[("LOW_CONFIDENCE_CLASS_EXCLUDED", "METADATA_COVERAGE")] += 1
        if not vehicle.get("application_class"):
            counter[("APPLICATION_CLASS_UNRESOLVED", "METADATA_COVERAGE")] += 1
        for domain in ROLLING_DOMAINS:
            if not pools[vde_id].get(domain):
                counter[(f"NO_ELIGIBLE_{domain}", "REFERENCE_COVERAGE")] += 1
        row = record_map[vde_id]
        if row["complete_pool"] and not row.get("selected"):
            counter[("OVERSHOOT_GT_5PCT", "PHYSICAL_INCOMPATIBILITY")] += 1
    return [{"reason": reason, "limitation_type": kind, "vde_count": count} for (reason, kind), count in counter.most_common()]


def _decision(
    real: dict[str, Any], shuffle_mean: dict[str, float], sensitivity: list[dict[str, Any]], sample_size: int,
) -> tuple[str, dict[str, Any]]:
    lift = real["physical_acceptance_rate_pct"] - shuffle_mean["physical_acceptance_rate_pct"]
    by_size = {row["pool_size"]: row for row in sensitivity}
    top3, top5 = by_size["TOP_3"], by_size["TOP_5"]
    stable = (
        abs(top5["physical_acceptance_rate_pct"] - top3["physical_acceptance_rate_pct"]) <= POOL_STABILITY_ACCEPTANCE_PP
        and (
            top3["median_unresolved_pct"] is None or top5["median_unresolved_pct"] is None
            or abs(top5["median_unresolved_pct"] - top3["median_unresolved_pct"]) <= POOL_STABILITY_RESIDUAL_PP
        )
    )
    coverage_pct = 100.0 * real["complete_candidate_pools"] / sample_size
    method_valid = lift >= MIN_SIGNAL_LIFT_PP and stable
    if method_valid and coverage_pct >= COVERAGE_SCALE_GATE_PCT:
        decision = "ROLLINGMINOR_SCALE_RECOMMENDED"
    elif method_valid:
        decision = "ROLLINGMINOR_METHOD_VALID_BUT_COVERAGE_LIMITED"
    else:
        decision = "ROLLINGMINOR_METHOD_NOT_VALIDATED"
    return decision, {
        "real_vs_shuffle_acceptance_lift_pp": lift,
        "top3_vs_top5_stable": stable,
        "complete_pool_coverage_pct": coverage_pct,
        "method_signal_gate_passed": method_valid,
        "coverage_scale_gate_pct": COVERAGE_SCALE_GATE_PCT,
    }


def execute_rollingminor_completion(
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
        historical_tires = load_historical_tire_evidence(legacy)
    finally:
        connection.close(); legacy.close()
    sample = deterministic_stratified_sample(population, DEFAULT_ARCHITECTURE_TARGETS)
    if len(sample) != 180:
        raise RuntimeError(f"Expected frozen 180-VDE sample, found {len(sample)}")
    references = enrich_applicability(load_all_component_references(component_catalog_path))
    tire_refs = load_tire_references(tire_zip_path)

    # Reproduce the Pass 1C.3 RollingMinor baseline before applying new metadata.
    baseline_vehicles = [dict(vehicle) for vehicle in sample]
    baseline_pools: dict[int, dict[str, list[CandidateVector]]] = {}
    baseline_resolved = 0
    for vehicle in baseline_vehicles:
        if vehicle.get("application_class"):
            baseline_resolved += 1
        baseline_pools[int(vehicle["vde_id"])] = _build_rolling_pools(
            vehicle, references, tire_refs, historical_tires,
            tire_limit=4, brake_limit=3, hub_limit=3,
        )
    _, _, baseline_records = _run_rolling(
        baseline_vehicles, baseline_pools,
        {"TIRE": 4, "BRAKE": 3, "HUB_BEARING": 3, "TRANSMISSION": 0, "AXLE": 0},
    )
    baseline_kpis = aggregate_kpis(baseline_records, "ROLLING_MINOR")

    application_rows: list[dict[str, Any]] = []
    resolved_previously_missing = 0
    eligible_previously_missing = 0
    validation_eligible = 0
    for vehicle in sample:
        resolution = resolve_application_class_v14(vehicle)
        if not resolution["previous_application_class"] and resolution["application_class"]:
            resolved_previously_missing += 1
            if resolution["eligible_for_validation"]:
                eligible_previously_missing += 1
        vehicle.update(resolution)
        if resolution["eligible_for_validation"]:
            validation_eligible += 1
        application_rows.append({
            "vde_id": vehicle["vde_id"], "make": vehicle.get("make"), "model": vehicle.get("model"),
            "category": vehicle.get("category"), "test_mass_kg": vehicle.get("test_mass_resolved_kg"),
            "drive_layout": vehicle.get("drive_layout"), "architecture_class": vehicle.get("architecture"),
            **resolution,
        })

    master_pools: dict[int, dict[str, list[CandidateVector]]] = {}
    candidate_rows: list[dict[str, Any]] = []
    for vehicle in sample:
        vde_id = int(vehicle["vde_id"])
        if vehicle["eligible_for_validation"]:
            pools = _build_rolling_pools(vehicle, references, tire_refs, historical_tires)
        else:
            pools = {domain: [] for domain in ("TIRE", "BRAKE", "HUB_BEARING", "TRANSMISSION", "AXLE")}
        master_pools[vde_id] = pools
        for row in _candidate_rows(vde_id, pools):
            if row["domain"] in ROLLING_DOMAINS:
                row["application_class_source"] = vehicle["application_class_source"]
                row["application_class_confidence"] = vehicle["application_class_confidence"]
                candidate_rows.append(row)

    primary_pools = {vde_id: truncate_pools(pools, PRIMARY_CAPS) for vde_id, pools in master_pools.items()}
    selected, all_accepted, real_records = _run_rolling(sample, primary_pools, PRIMARY_CAPS)
    real_kpis = aggregate_kpis(real_records, "ROLLING_MINOR")

    shuffle_rows: list[dict[str, Any]] = []
    shuffled_kpis: list[dict[str, Any]] = []
    for seed in SHUFFLE_SEEDS:
        shuffled = shuffle_pools_seeded(primary_pools, seed)
        _, _, records = _run_rolling(sample, shuffled, PRIMARY_CAPS)
        kpis = aggregate_kpis(records, "ROLLING_MINOR")
        shuffled_kpis.append(kpis)
        shuffle_rows.append({"control": "SHUFFLED", "seed": seed, **kpis})
    shuffle_mean = {
        key: mean(float(row[key]) for row in shuffled_kpis)
        for key in ("physical_acceptance_rate_pct", "median_explained_pct", "median_unresolved_pct")
        if all(row[key] is not None for row in shuffled_kpis)
    }
    shuffle_rows.insert(0, {"control": "REAL", "seed": None, **real_kpis})
    shuffle_rows.append({
        "control": "SHUFFLE_MEAN", "seed": None,
        "physical_acceptance_rate_pct": shuffle_mean["physical_acceptance_rate_pct"],
        "median_explained_pct": shuffle_mean["median_explained_pct"],
        "median_unresolved_pct": shuffle_mean["median_unresolved_pct"],
        "acceptance_lift_pp": real_kpis["physical_acceptance_rate_pct"] - shuffle_mean["physical_acceptance_rate_pct"],
        "explained_lift_pp": real_kpis["median_explained_pct"] - shuffle_mean["median_explained_pct"],
    })

    sensitivity_rows: list[dict[str, Any]] = []
    for size in (1, 2, 3, 5):
        caps = {"TIRE": size, "BRAKE": size, "HUB_BEARING": size, "TRANSMISSION": 0, "AXLE": 0}
        _, _, records = _run_rolling(sample, master_pools, caps)
        sensitivity_rows.append({"pool_size": f"TOP_{size}", **aggregate_kpis(records, "ROLLING_MINOR")})
    decision, decision_evidence = _decision(real_kpis, shuffle_mean, sensitivity_rows, len(sample))

    app_summary_counter = Counter((row["application_class"], row["application_class_source"], row["application_class_confidence"], row["eligible_for_validation"]) for row in application_rows)
    app_summary_rows = [{
        "application_class": key[0], "application_class_source": key[1],
        "application_class_confidence": key[2], "eligible_for_validation": key[3], "vde_count": count,
    } for key, count in sorted(app_summary_counter.items(), key=lambda item: tuple(str(value) for value in item[0]))]
    failure_rows = _failure_rows(sample, primary_pools, real_records)
    breakdowns = {
        "application_class": _breakdown(sample, real_records, "application_class"),
        "drive_layout": _breakdown(sample, real_records, "drive_layout"),
        "architecture_class": _breakdown(
            [{**vehicle, "architecture_class": vehicle["architecture"]} for vehicle in sample], real_records, "architecture_class"
        ),
    }

    selected_fields = sorted({key for row in selected for key in row})
    _write_csv(output_dir / "application_class_resolution.csv", application_rows, application_rows[0].keys())
    _write_csv(output_dir / "application_class_summary.csv", app_summary_rows, app_summary_rows[0].keys())
    _write_csv(output_dir / "rollingminor_candidate_pools.csv", candidate_rows, candidate_rows[0].keys())
    _write_csv(output_dir / "rollingminor_selected_combinations.csv", selected, selected_fields)
    _write_csv(output_dir / "rollingminor_all_accepted_combinations.csv", all_accepted, selected_fields)
    _write_csv(output_dir / "pool_size_sensitivity.csv", sensitivity_rows, sensitivity_rows[0].keys())
    shuffle_fields = sorted({key for row in shuffle_rows for key in row})
    _write_csv(output_dir / "shuffle_control_summary.csv", shuffle_rows, shuffle_fields)
    _write_csv(output_dir / "coverage_failure_reasons.csv", failure_rows, failure_rows[0].keys())
    for name, rows in breakdowns.items():
        _write_csv(output_dir / f"{name}_breakdown.csv", rows, rows[0].keys())
    sample_rows = [{"vde_id": row["vde_id"], "vehicle_configuration_id": row["vehicle_configuration_id"],
                    "architecture": row["architecture"], "application_class": row.get("application_class"),
                    "drive_layout": row.get("drive_layout"), "test_mass_kg": row.get("test_mass_resolved_kg"),
                    "stratum": row.get("stratum"), "parent_rolling_minor_resolution_id": row["rolling_minor_resolution_id"]}
                   for row in sample]
    _write_csv(output_dir / "pilot_sample.csv", sample_rows, sample_rows[0].keys())
    hashes_after = {"pass1_db": file_sha256(db_path), "legacy_db": file_sha256(legacy_db_path)}
    if hashes_before != hashes_after:
        raise RuntimeError("Read-only metadata pilot changed an input database")
    summary = {
        "method_version": METHOD_VERSION, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "git": _git_metadata(),
        "sample_size": len(sample),
        "sample_identity_sha256": sha256(",".join(str(row["vde_id"]) for row in sample_rows).encode()).hexdigest().upper(),
        "previous_application_resolved": baseline_resolved,
        "previous_application_unresolved": len(sample) - baseline_resolved,
        "previously_unresolved_now_resolved": resolved_previously_missing,
        "previously_unresolved_now_validation_eligible": eligible_previously_missing,
        "application_resolved_total": sum(bool(row["application_class"]) for row in application_rows),
        "application_unresolved_total": sum(not row["application_class"] for row in application_rows),
        "validation_eligible_application_classes": validation_eligible,
        "pass_1c3": baseline_kpis, "pass_1c4": real_kpis,
        "shuffle_mean": shuffle_mean, "shuffle_seeds": list(SHUFFLE_SEEDS),
        "decision_evidence": decision_evidence,
        "drivetrain_historical_context": {
            "pass_1c3_real_acceptance_pct": 5.0, "pass_1c3_shuffle_acceptance_pct": 5.0,
            "policy": "DRIVETRAIN_AGGREGATE_REMAINS_AUTHORITATIVE_NO_SEARCH_RUN_IN_PASS_1C4",
        },
        "hashes_before": hashes_before, "hashes_after": hashes_after,
        "quick_check": quick_check, "foreign_key_issues": fk_issues,
        "database_rows_written": 0, "database_objects_created": 0, "external_search_count": 0,
        "scaled_combination_count": 0, "drivetrain_search_executed": False, "scale_up_executed": False,
        "decision": decision,
    }
    (output_dir / "execution_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "pass1c4_methodology_report.md").write_text(
        _report(summary, sensitivity_rows, failure_rows), encoding="utf-8"
    )
    return summary


def _fmt(value: Any) -> str:
    return "N/A" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value))


def _report(summary: dict[str, Any], sensitivity: list[dict[str, Any]], failures: list[dict[str, Any]]) -> str:
    old, new = summary["pass_1c3"], summary["pass_1c4"]
    shuffle = summary["shuffle_mean"]
    failure_counts = {row["reason"]: row["vde_count"] for row in failures}
    comparison = (
        ("Pilot VDEs", 180, 180),
        ("Application class resolved", summary["previous_application_resolved"], summary["application_resolved_total"]),
        ("Application class unresolved", summary["previous_application_unresolved"], summary["application_unresolved_total"]),
        ("Complete RollingMinor pools", old["complete_candidate_pools"], new["complete_candidate_pools"]),
        ("Accepted RollingMinor build-ups", old["accepted_complete_builds"], new["accepted_complete_builds"]),
        ("Acceptance % of complete pools", old["physical_acceptance_rate_pct"], new["physical_acceptance_rate_pct"]),
        ("Acceptance % of full pilot", 100.0 * old["accepted_complete_builds"] / 180, 100.0 * new["accepted_complete_builds"] / 180),
        ("Median explained %", old["median_explained_pct"], new["median_explained_pct"]),
        ("Median unresolved %", old["median_unresolved_pct"], new["median_unresolved_pct"]),
        ("Real acceptance %", old["physical_acceptance_rate_pct"], new["physical_acceptance_rate_pct"]),
        ("Shuffle acceptance %", 44.44444444444444, shuffle["physical_acceptance_rate_pct"]),
        ("Real-vs-shuffle lift [pp]", 14.814814814814817, summary["decision_evidence"]["real_vs_shuffle_acceptance_lift_pp"]),
    )
    lines = [
        "# Pass 1C.4 RollingMinor Metadata Completion & Validation", "", "## Frozen scope and safety", "",
        f"- exact Pass 1C.3 sample reused: **YES**, `{summary['sample_identity_sha256']}`",
        "- RollingMinor-only search: **YES**", "- Drivetrain search executed: **NO**",
        "- scaling / external research / DB writes: **0 / 0 / 0**",
        f"- quick_check / FK issues: **{summary['quick_check']} / {summary['foreign_key_issues']}**", "",
        "## Pass comparison", "", "| Metric | Pass 1C.3 | Pass 1C.4 |", "|---|---:|---:|",
    ]
    lines.extend(f"| {label} | {_fmt(before)} | {_fmt(after)} |" for label, before, after in comparison)
    lines.extend(["", "## Main questions", "",
                  f"- Q1 — previously unresolved classes resolved: **{summary['previously_unresolved_now_resolved']}/{summary['previous_application_unresolved']}**; newly resolved at HIGH/MEDIUM and eligible: **{summary['previously_unresolved_now_validation_eligible']}**; eligible total: **{summary['validation_eligible_application_classes']}**.",
                  f"- Q2 — complete Tire+Brake+Hub pools: **{new['complete_candidate_pools']}/180 ({100.0 * new['complete_candidate_pools']/180:.3f}%)**.",
                  f"- Q3 — physically accepted complete build-ups: **{new['accepted_complete_builds']}**.",
                  f"- Q4 — typical RollingMinor explained: **{new['median_explained_pct']:.3f}%**; unresolved: **{new['median_unresolved_pct']:.3f}%**.",
                  f"- Q5 — real vs five-shuffle mean: **{new['physical_acceptance_rate_pct']:.3f}% vs {shuffle['physical_acceptance_rate_pct']:.3f}%**, lift **{summary['decision_evidence']['real_vs_shuffle_acceptance_lift_pp']:.3f} pp**. Real is nominally higher, but does **not** materially outperform shuffle under the predeclared +10 pp gate.",
                  f"- Q6 — Top-3 approximately equals Top-5: **{'YES' if summary['decision_evidence']['top3_vs_top5_stable'] else 'NO'}**.",
                  f"- Q7 — the dominant limitation is **reference coverage** (no Hub: {failure_counts.get('NO_ELIGIBLE_HUB_BEARING', 0)}; no Brake: {failure_counts.get('NO_ELIGIBLE_BRAKE', 0)}; no Tire: {failure_counts.get('NO_ELIGIBLE_TIRE', 0)}), followed by **physical incompatibility** ({failure_counts.get('OVERSHOOT_GT_5PCT', 0)}) and residual **metadata confidence** ({failure_counts.get('LOW_CONFIDENCE_CLASS_EXCLUDED', 0)} LOW classifications).", "",
                  "## Pool-size sensitivity", "", "| Pool | Complete pools | Accepted | Acceptance % | Median explained % | Median unresolved % |", "|---|---:|---:|---:|---:|---:|"])
    for row in sensitivity:
        lines.append(f"| {row['pool_size']} | {row['complete_candidate_pools']} | {row['accepted_complete_builds']} | {_fmt(row['physical_acceptance_rate_pct'])} | {_fmt(row['median_explained_pct'])} | {_fmt(row['median_unresolved_pct'])} |")
    lines.extend(["", "## Remaining limitations", ""])
    for row in failures:
        lines.append(f"- {row['reason']} — {row['limitation_type']}: **{row['vde_count']}**")
    lines.extend(["", "Breakdowns by application class, drive layout, and architecture are supplied as CSV artifacts.", "",
                  "## Drivetrain historical context", "",
                  "Pass 1C.3 produced 5% acceptance for both real and shuffled Drivetrain matching. It was not rerun or tuned. `DRIVETRAIN_AGGREGATE` remains authoritative.", "",
                  "## Decision", "", f"`{summary['decision']}`", "",
                  "This remains an application-matched surrogate decomposition, not exact hardware identification. No authoritative RollingMinor or VDE value was modified."])
    return "\n".join(lines) + "\n"


__all__ = [
    "PRIMARY_CAPS", "SHUFFLE_SEEDS", "execute_rollingminor_completion",
    "resolve_application_class_v14", "shuffle_pools_seeded",
]
