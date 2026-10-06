"""Canonical candidate materialization and deterministic research-gap handoff."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Iterable

from src.vde_core import db as db_module
from src.vde_core.component_enrichment_pass1 import ESTIMATOR_VERSION, execute_pass1, file_sha256
from src.vde_core.component_population_materialization import (
    METHOD_VERSION as SCALE_METHOD_VERSION,
    _build_manifest,
    _csv_bytes,
    _fine_evidence,
    _stable_json,
    _table_counts,
    _write_csv,
    apply_projection_manifest,
    protected_snapshot,
)


METHOD_VERSION = "SPRINT12_CANONICAL_MATERIALIZATION_RESEARCH_HANDOFF_V1.0"
EXPECTED_SOURCE_SHA256 = "E0E83A41A0E36141A30498C5FEAB2374D4C1ED64C948F296520B802C302ABE14"
EXPECTED_MANIFEST_SHA256 = "1A1501BDEA5447C231E256001F08779002AC4DED7C98C2AF3800A7B199FBD36A"
FINAL_COMPLETE = "CANONICAL_COMPONENT_POPULATION_COMPLETE"
FINAL_WITH_GAPS = "CANONICAL_COMPONENT_POPULATION_COMPLETE_WITH_KNOWN_GAPS"
STOP_CHANGED = "STOP_CANONICAL_STATE_CHANGED"
FAILED = "CANONICAL_MATERIALIZATION_FAILED_VALIDATION"


def canonical_candidate_path(root: Path) -> Path:
    """Resolve the repository-owned STAGING path, never the runtime default."""
    configured = Path(db_module.STAGING_DB_PATH)
    return configured if configured.is_absolute() else Path(root) / configured


def _ro(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _schema_signature(connection: sqlite3.Connection) -> str:
    rows = [tuple(row) for row in connection.execute(
        "SELECT type,name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
    )]
    return sha256(_stable_json(rows).encode("utf-8")).hexdigest().upper()


def _snapshot(path: Path) -> dict[str, Any]:
    connection = _ro(path)
    try:
        return {
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
            "quick_check": connection.execute("PRAGMA quick_check").fetchone()[0],
            "foreign_key_issues": len(connection.execute("PRAGMA foreign_key_check").fetchall()),
            "row_counts": _table_counts(connection),
            "schema_sha256": _schema_signature(connection),
            "protected_vde_fields": protected_snapshot(connection),
        }
    finally:
        connection.close()


def _protected_equal(before: dict[str, Any], after: dict[str, Any]) -> bool:
    left = {row["column"]: row["sha256"] for row in before["protected_vde_fields"]}
    right = {row["column"]: row["sha256"] for row in after["protected_vde_fields"]}
    return left == right


def _core_counts_equal(before: dict[str, Any], after: dict[str, Any]) -> bool:
    return all(
        before["row_counts"].get(table) == after["row_counts"].get(table)
        for table in ("vde", "vehicle_configuration", "run", "fuelcons")
    )


def _semantic_verification(path: Path) -> dict[str, Any]:
    connection = _ro(path)
    try:
        resolutions: dict[int, dict[str, sqlite3.Row]] = defaultdict(dict)
        for row in connection.execute(
            """SELECT l.vde_id,l.boundary,r.* FROM vde_component_resolution l
               JOIN component_resolution r ON r.component_resolution_id=l.component_resolution_id
               WHERE l.adoption_role='ADOPTED' AND r.estimator_version=?""",
            (ESTIMATOR_VERSION,),
        ):
            resolutions[int(row["vde_id"])][str(row["boundary"])] = row
        projected = mismatches = bad_provenance = 0
        for row in connection.execute(
            """SELECT id,tire_A_final,tire_B_final,tire_C_final,aero_C_coef_Npkph2,
                      trans_A_coef_N,trans_B_coef_Npkph,trans_C_coef_Npkph2,provenance_json
               FROM vde WHERE tire_A_final IS NOT NULL ORDER BY id"""
        ):
            projected += 1
            macro = resolutions[int(row["id"])]
            expected = (
                macro["ROLLING_MINOR"]["resolved_A_N"], macro["ROLLING_MINOR"]["resolved_B_N_per_kph"],
                macro["ROLLING_MINOR"]["resolved_C_N_per_kph2"], macro["AERO"]["resolved_C_N_per_kph2"],
                macro["DRIVETRAIN_AGGREGATE"]["resolved_A_N"], macro["DRIVETRAIN_AGGREGATE"]["resolved_B_N_per_kph"],
                macro["DRIVETRAIN_AGGREGATE"]["resolved_C_N_per_kph2"],
            )
            actual = tuple(row[name] for name in (
                "tire_A_final", "tire_B_final", "tire_C_final", "aero_C_coef_Npkph2",
                "trans_A_coef_N", "trans_B_coef_Npkph", "trans_C_coef_Npkph2",
            ))
            mismatches += actual != expected
            try:
                provenance = json.loads(row["provenance_json"])["component_population_projection"]
                valid = (
                    provenance["decomposition_mode"] == "MACRO_DECOMPOSITION"
                    and provenance["tire_slot_semantics"] == "ROLLING_MINOR_AGGREGATE"
                    and provenance["transmission_slot_semantics"] == "DRIVETRAIN_AGGREGATE"
                    and provenance["fine_component_identification"] is False
                )
            except (TypeError, KeyError, json.JSONDecodeError):
                valid = False
            bad_provenance += not valid
        result = {
            "projected_macro_vdes": projected,
            "slot_to_macro_mismatches": mismatches,
            "bad_or_missing_projection_provenance": bad_provenance,
            "unauthorized_fine_vde_projections": connection.execute(
                """SELECT COUNT(*) FROM vde
                   WHERE brake_A_coef_N IS NOT NULL OR brake_B_coef_Npkph IS NOT NULL OR brake_C_coef_Npkph2 IS NOT NULL
                      OR parasitic_A_coef_N IS NOT NULL OR parasitic_B_coef_Npkph IS NOT NULL OR parasitic_C_coef_Npkph2 IS NOT NULL"""
            ).fetchone()[0],
            "fine_adopted_links": connection.execute(
                """SELECT COUNT(*) FROM vde_component_resolution
                   WHERE adoption_role='ADOPTED' AND boundary IN ('TIRE','BRAKE','HUB_BEARING','TRANSMISSION','AXLE')"""
            ).fetchone()[0],
            "fine_supporting_links": connection.execute(
                """SELECT COUNT(*) FROM vde_component_resolution
                   WHERE adoption_role='SUPPORTING' AND boundary IN ('TIRE','BRAKE','HUB_BEARING','TRANSMISSION','AXLE')"""
            ).fetchone()[0],
            "edrive_macro_resolutions": connection.execute(
                """SELECT COUNT(DISTINCT l.vde_id) FROM vde_component_resolution l
                   JOIN component_resolution r ON r.component_resolution_id=l.component_resolution_id
                   WHERE l.boundary='EDRIVE_AGGREGATE' AND l.adoption_role='ADOPTED' AND r.estimator_version=?""",
                (ESTIMATOR_VERSION,),
            ).fetchone()[0],
            "edrive_rows_with_transmission_projection": connection.execute(
                """SELECT COUNT(DISTINCT v.id) FROM vde v
                   JOIN vde_component_resolution l ON l.vde_id=v.id AND l.boundary='EDRIVE_AGGREGATE'
                   WHERE v.trans_A_coef_N IS NOT NULL OR v.trans_B_coef_Npkph IS NOT NULL OR v.trans_C_coef_Npkph2 IS NOT NULL"""
            ).fetchone()[0],
        }
        result["semantic_checks_passed"] = all((
            projected > 0, mismatches == 0, bad_provenance == 0,
            result["unauthorized_fine_vde_projections"] == 0,
            result["fine_adopted_links"] == 0,
            result["edrive_rows_with_transmission_projection"] == 0,
        ))
        return result
    finally:
        connection.close()


def _normalized_token(value: Any) -> str:
    return " ".join(str(value or "").upper().split())


def _metadata_by_vde(path: Path) -> dict[int, dict[str, Any]]:
    connection = _ro(path)
    try:
        sql = """
            SELECT v.id AS vde_id,v.vehicle_configuration_id,vc.program_id,
                   v.make,v.model,v.year AS model_year,v.category,v.test_mass_kg,
                   v.provenance_json AS current_provenance,
                   vc.drive_system AS drive_type,vc.transmission_type,vc.transmission_model,
                   vc.gear_count AS gears,vc.engine_model AS engine_code,
                   vc.propulsion_architecture,vc.architecture_properties_json
            FROM vde v JOIN vehicle_configuration vc
              ON vc.vehicle_configuration_id=v.vehicle_configuration_id
            WHERE v.record_status='ACTIVE' ORDER BY v.id
        """
        return {int(row["vde_id"]): dict(row) for row in connection.execute(sql)}
    finally:
        connection.close()


def _gap_specs(row: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """Return (family, domain, reason, missing evidence) without inference."""
    result: list[tuple[str, str, str, str]] = []
    architecture = str(row.get("architecture_class") or "UNRESOLVED")
    drive = _normalized_token(row.get("drive_type"))
    macro_unresolved = str(row.get("macro_status")) == "UNRESOLVED"
    if macro_unresolved:
        result.extend((
            ("MACRO_UNRESOLVED", "SYSTEM", "MACRO_MODEL_NOT_RESOLVED", "Independent road-load state / architecture evidence"),
            ("ROADLOAD_STATE_UNRESOLVED", "SYSTEM", "MACRO_INPUT_EVIDENCE_INSUFFICIENT", "Independent set/target road-load evidence"),
        ))
    if "UNRESOLVED" in architecture:
        result.append(("ARCHITECTURE_UNRESOLVED", "POWERTRAIN", "ARCHITECTURE_ROUTING_UNRESOLVED", "Powertrain topology and transmission architecture"))
    if not row.get("application_class"):
        result.append(("APPLICATION_METADATA_GAP", "VEHICLE", "APPLICATION_CLASS_UNAVAILABLE", "Detailed source vehicle category/application class"))
    # Fine gaps are research-preparation only and never cause VDE projection.
    for flag, family, domain, evidence in (
        ("tire_available", "TIRE_EVIDENCE_GAP", "TIRE", "Applicable tire specification and RRC evidence"),
        ("brake_available", "BRAKE_EVIDENCE_GAP", "BRAKE", "Reusable brake drag reference with physical boundary"),
        ("hub_available", "HUB_BEARING_EVIDENCE_GAP", "HUB_BEARING", "Front/rear hub-bearing drag reference"),
    ):
        if not int(row.get(flag) or 0):
            result.append((family, domain, "REFERENCE_COVERAGE_GAP", evidence))
    fixed_edrive = architecture == "EV_FIXED_GEAR"
    if fixed_edrive:
        result.append(("EDRIVE_FINE_BOUNDARY_GAP", "EDRIVE", "EDRIVE_AGGREGATE_NOT_FINE_IDENTIFIED", "Motor/reducer/final-drive physical boundary evidence"))
    else:
        if not int(row.get("transmission_available") or 0):
            result.append(("TRANSMISSION_EVIDENCE_GAP", "TRANSMISSION", "REFERENCE_COVERAGE_GAP", "Transmission hardware/family and loss boundary"))
        if drive in {"RWD", "AWD", "4WD", "ALL WHEEL DRIVE", "4-WHEEL DRIVE"} and not int(row.get("axle_available") or 0):
            result.append(("AXLE_EVIDENCE_GAP", "AXLE", "REFERENCE_COVERAGE_GAP", "Axle/differential hardware and loss boundary"))
    if row.get("boundary_status") == "BOUNDARY_UNKNOWN":
        result.append(("TRANSMISSION_AXLE_BOUNDARY_UNKNOWN", "DRIVETRAIN", "BOUNDARY_UNKNOWN", "Included/excluded final-drive, differential, PTU, axle and hub subsystems"))
    # Preserve order while removing duplicate family/domain pairs.
    seen, unique = set(), []
    for item in result:
        key = item[:2]
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def build_gap_inventory(manifest: list[dict[str, Any]], metadata: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in manifest:
        meta = metadata[int(item["vde_id"])]
        merged = {**item, **meta}
        for family, domain, reason, missing in _gap_specs(merged):
            rows.append({
                "vde_id": merged["vde_id"], "vehicle_configuration_id": merged["vehicle_configuration_id"],
                "program_id": merged.get("program_id"), "make": merged.get("make"), "model": merged.get("model"),
                "model_year": merged.get("model_year"), "category": merged.get("category"),
                "application_class": merged.get("application_class"), "architecture_class": merged.get("architecture_class"),
                "drive_type": merged.get("drive_type"), "transmission_type": merged.get("transmission_type"),
                "gears": merged.get("gears"), "engine_code": merged.get("engine_code"),
                "transmission_code": merged.get("transmission_model"), "test_mass_kg": merged.get("test_mass_kg"),
                "macro_status": merged.get("macro_status"), "macro_model": merged.get("macro_model_type"),
                "fine_evidence_domains": merged.get("fine_evidence_domains"),
                "gap_family": family, "gap_domain": domain, "gap_reason_code": reason,
                "missing_evidence": missing, "current_provenance": merged.get("current_provenance"),
                "current_confidence": merged.get("macro_status"), "boundary_status": merged.get("boundary_status"),
                "propulsion_architecture": merged.get("propulsion_architecture"),
            })
    return sorted(rows, key=lambda row: (str(row["gap_family"]), str(row["gap_domain"]), int(row["vde_id"])))


def _group_identity(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    common = {
        "gap_family": row["gap_family"], "domain": row["gap_domain"],
        "architecture": _normalized_token(row.get("architecture_class")),
        "drive": _normalized_token(row.get("drive_type")),
    }
    transmission_code = _normalized_token(row.get("transmission_code"))
    engine_code = _normalized_token(row.get("engine_code"))
    make, model = _normalized_token(row.get("make")), _normalized_token(row.get("model"))
    transmission_type = _normalized_token(row.get("transmission_type"))
    if transmission_code and row["gap_domain"] in {"TRANSMISSION", "AXLE", "DRIVETRAIN", "EDRIVE"}:
        identity = {**common, "tier": 1, "make": make, "hardware_code": transmission_code}
    elif engine_code and row["gap_domain"] in {"POWERTRAIN", "SYSTEM", "EDRIVE"}:
        identity = {**common, "tier": 2, "make": make, "hardware_code": engine_code}
    elif make and model:
        identity = {**common, "tier": 3, "make": make, "model_family": model, "transmission_family": transmission_type}
    elif make:
        identity = {**common, "tier": 4, "make": make, "transmission_family": transmission_type}
    else:
        identity = {**common, "tier": 5, "application_class": _normalized_token(row.get("application_class"))}
    return _stable_json(identity), identity


def _research_goal(family: str) -> str:
    goals = {
        "MACRO_UNRESOLVED": "Resolve the independent evidence required for deterministic macro routing/estimation.",
        "ARCHITECTURE_UNRESOLVED": "Resolve drivetrain architecture and transmission topology from reliable technical evidence.",
        "ROADLOAD_STATE_UNRESOLVED": "Find independent target/set road-load state evidence without deriving it from closure.",
        "APPLICATION_METADATA_GAP": "Recover the detailed source vehicle category/application classification.",
        "TIRE_EVIDENCE_GAP": "Find applicable tire specification and rolling-resistance evidence.",
        "BRAKE_EVIDENCE_GAP": "Find reusable brake drag evidence with a documented physical boundary.",
        "HUB_BEARING_EVIDENCE_GAP": "Find front/rear hub-bearing drag evidence for the technical family.",
        "TRANSMISSION_EVIDENCE_GAP": "Identify transmission hardware/family and its supported loss boundary.",
        "AXLE_EVIDENCE_GAP": "Identify axle/differential family and its supported loss boundary.",
        "TRANSMISSION_AXLE_BOUNDARY_UNKNOWN": "Resolve whether transmission evidence includes final drive, differential, PTU, axles or hubs.",
        "EDRIVE_FINE_BOUNDARY_GAP": "Resolve motor/reducer/final-drive identities and included subsystem boundaries.",
    }
    return goals[family]


def group_gap_inventory(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identities: dict[str, dict[str, Any]] = {}
    for row in rows:
        key, identity = _group_identity(row)
        grouped[key].append(row)
        identities[key] = identity
    result = []
    impact_weight = {
        "MACRO_UNRESOLVED": 5, "ARCHITECTURE_UNRESOLVED": 5, "ROADLOAD_STATE_UNRESOLVED": 4,
        "TRANSMISSION_AXLE_BOUNDARY_UNKNOWN": 4, "EDRIVE_FINE_BOUNDARY_GAP": 4,
        "APPLICATION_METADATA_GAP": 3, "TRANSMISSION_EVIDENCE_GAP": 3,
        "AXLE_EVIDENCE_GAP": 3, "TIRE_EVIDENCE_GAP": 2,
        "BRAKE_EVIDENCE_GAP": 2, "HUB_BEARING_EVIDENCE_GAP": 2,
    }
    for key, members in sorted(grouped.items()):
        identity = identities[key]
        vdes = sorted({int(row["vde_id"]) for row in members})
        vcs = sorted({str(row["vehicle_configuration_id"]) for row in members})
        years = sorted({int(row["model_year"]) for row in members if row.get("model_year") not in (None, "")})
        family, domain = str(members[0]["gap_family"]), str(members[0]["gap_domain"])
        score = len(vdes) * impact_weight[family] * (6 - int(identity["tier"]))
        priority = "P0" if score >= 500 else "P1" if score >= 100 else "P2" if score >= 20 else "P3"
        reuse = "HIGH" if len(vdes) >= 20 else "MEDIUM" if len(vdes) >= 5 else "LOW"
        digest = sha256(key.encode("utf-8")).hexdigest().upper()[:16]
        def single(field: str) -> Any:
            values = sorted({_normalized_token(row.get(field)) for row in members if _normalized_token(row.get(field))})
            return values[0] if len(values) == 1 else ("MULTIPLE" if values else None)
        hardware_codes = sorted({
            _normalized_token(row.get(field)) for row in members for field in ("transmission_code", "engine_code")
            if _normalized_token(row.get(field))
        })
        result.append({
            "research_group_id": f"RG-{domain[:5]}-{digest}", "gap_family": family, "domain": domain,
            "identity_tier": identity["tier"], "vde_count": len(vdes), "vehicle_configuration_count": len(vcs),
            "make": single("make"), "model_family": single("model"),
            "model_year_min": min(years) if years else None, "model_year_max": max(years) if years else None,
            "application_class": single("application_class"), "architecture_class": single("architecture_class"),
            "drive_type": single("drive_type"), "transmission_type": single("transmission_type"),
            "gears": single("gears"), "known_hardware_codes": ";".join(hardware_codes),
            "front_rear_position": "FRONT_AND_REAR_REQUIRED" if domain == "HUB_BEARING" else None,
            "current_evidence_status": single("current_confidence"),
            "current_boundary_status": single("boundary_status"),
            "missing_evidence": single("missing_evidence"), "research_goal": _research_goal(family),
            "reuse_potential": reuse, "priority": priority, "priority_score": score,
            "priority_reason": f"{len(vdes)} VDE(s) x impact {impact_weight[family]} x identity leverage {6-int(identity['tier'])}",
            "suggested_source_types": "OEM_TECHNICAL;REGULATORY_CERTIFICATION;SAE_TECHNICAL;SUPPLIER_TECHNICAL;STRUCTURED_TECHNICAL_DB",
            "stop_condition": "EXACT_OR_STRONG_REUSABLE_EVIDENCE_FOUND;OTHERWISE_CREDIBLE_SOURCE_FAMILIES_EXHAUSTED_OR_NOT_FOUND",
            "representative_vde_ids": ";".join(str(value) for value in vdes[:10]),
        })
    return sorted(result, key=lambda row: (str(row["priority"]), -int(row["priority_score"]), str(row["research_group_id"])))


def _gap_summary(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str], dict[str, set[Any]]] = defaultdict(lambda: {"groups": set(), "vdes": set()})
    for group in groups:
        key = (str(group["gap_family"]), str(group["domain"]), str(group["priority"]))
        counts[key]["groups"].add(group["research_group_id"])
        # Representative IDs do not cover all VDEs; sum group vde_count because groups are disjoint within family/domain.
    result = []
    for family, domain in sorted({(str(row["gap_family"]), str(row["domain"])) for row in groups}):
        selected = [row for row in groups if row["gap_family"] == family and row["domain"] == domain]
        result.append({
            "gap_family": family, "domain": domain, "research_group_count": len(selected),
            "vde_gap_count": sum(int(row["vde_count"]) for row in selected),
            "p0_groups": sum(row["priority"] == "P0" for row in selected),
            "p1_groups": sum(row["priority"] == "P1" for row in selected),
            "p2_groups": sum(row["priority"] == "P2" for row in selected),
            "p3_groups": sum(row["priority"] == "P3" for row in selected),
        })
    return result


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _gap_report(inventory: list[dict[str, Any]], groups: list[dict[str, Any]]) -> str:
    families = Counter(row["gap_family"] for row in inventory)
    lines = [
        "# Research Gap Inventory", "",
        "This is a deterministic handoff inventory. External searches, LLM classification and component assignment changes: **0**.", "",
        f"- VDE-gap rows: **{len(inventory):,}**",
        f"- Unique reusable research groups: **{len(groups):,}**",
        f"- Unique VDEs represented: **{len({int(row['vde_id']) for row in inventory}):,}**", "",
        "## Gap families", "", "| Family | VDE-gap rows |", "|---|---:|",
    ]
    lines.extend(f"| {family} | {count} |" for family, count in sorted(families.items()))
    lines.extend(["", "## Top 20 groups by leverage", "", "| Priority | Group | Family | Domain | VDEs | Identity tier | Goal |", "|---|---|---|---|---:|---:|---|"])
    for row in groups[:20]:
        lines.append(f"| {row['priority']} | {row['research_group_id']} | {row['gap_family']} | {row['domain']} | {row['vde_count']} | {row['identity_tier']} | {row['research_goal']} |")
    lines.extend(["", "Every group permits `NOT_FOUND`/`UNRESOLVED` through its explicit stop condition. Research outputs must stage evidence before deterministic promotion.", ""])
    return "\n".join(lines)


def execute_canonical_materialization_and_handoff(
    root: Path,
    output_dir: Path,
    legacy_db_path: Path,
    component_catalog_path: Path,
    tire_zip_path: Path,
    *,
    expected_source_sha256: str = EXPECTED_SOURCE_SHA256,
    expected_manifest_sha256: str = EXPECTED_MANIFEST_SHA256,
) -> dict[str, Any]:
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = canonical_candidate_path(root).resolve(strict=True)
    pre = _snapshot(candidate)
    if pre["sha256"] != expected_source_sha256:
        result = {
            "method_version": METHOD_VERSION, "final_recommendation": STOP_CHANGED,
            "candidate_path": str(candidate), "expected_sha256": expected_source_sha256,
            "actual_sha256": pre["sha256"], "database_written": False,
        }
        (output_dir / "execution_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    if pre["quick_check"] != "ok" or pre["foreign_key_issues"]:
        raise RuntimeError(f"Candidate integrity preflight failed: {pre}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = output_dir / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"eco_drive_canonical_candidate.pre_component_population.{timestamp}.db"
    if backup.exists():
        raise FileExistsError(f"Refusing to overwrite backup: {backup}")
    shutil.copy2(candidate, backup)
    backup_hash = file_sha256(backup)
    if backup_hash != pre["sha256"]:
        raise RuntimeError("Byte-for-byte backup hash mismatch")
    backup_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "original_path": str(candidate),
        "backup_path": str(backup), "original_sha256": pre["sha256"],
        "backup_sha256": backup_hash, "rollback": "Replace candidate with this backup after closing SQLite connections.",
    }
    (output_dir / "canonical_backup_manifest.json").write_text(json.dumps(backup_manifest, indent=2), encoding="utf-8")

    try:
        macro_first = execute_pass1(candidate, output_dir / "_canonical_macro_first", write=True)
        fine_by_vde, additions, reference_map = _fine_evidence(
            candidate, legacy_db_path, component_catalog_path, tire_zip_path
        )
        manifest, resolution_manifest = _build_manifest(candidate, fine_by_vde)
        manifest_hash = sha256(_csv_bytes(manifest)).hexdigest().upper()
        if manifest_hash != expected_manifest_sha256:
            raise RuntimeError(f"Validated manifest changed: expected={expected_manifest_sha256}, actual={manifest_hash}")
        conflicts = [row for row in manifest if row["conflict_reason"]]
        connection = sqlite3.connect(candidate)
        connection.row_factory = sqlite3.Row
        try:
            first_apply = apply_projection_manifest(connection, manifest, fine_by_vde, additions, reference_map)
        finally:
            connection.close()
        first_hash = file_sha256(candidate)
        semantic = _semantic_verification(candidate)
        post_first = _snapshot(candidate)

        macro_second = execute_pass1(candidate, output_dir / "_canonical_macro_second", write=True)
        connection = sqlite3.connect(candidate)
        connection.row_factory = sqlite3.Row
        try:
            second_apply = apply_projection_manifest(connection, manifest, fine_by_vde, additions, reference_map)
        finally:
            connection.close()
        second_hash = file_sha256(candidate)
        post = _snapshot(candidate)
        idempotent = all((
            macro_second["inserted_resolution_rows"] == 0, macro_second["inserted_links"] == 0,
            second_apply["vde_rows_changed"] == 0,
            second_apply["fine_component_resolutions_inserted"] == 0,
            second_apply["fine_supporting_links_inserted"] == 0,
            first_hash == second_hash,
        ))
        protected_unchanged = _protected_equal(pre, post)
        core_counts_unchanged = _core_counts_equal(pre, post)
        schema_unchanged = pre["schema_sha256"] == post["schema_sha256"]
        gates = all((
            post["quick_check"] == "ok", post["foreign_key_issues"] == 0,
            protected_unchanged, core_counts_unchanged, schema_unchanged,
            semantic["semantic_checks_passed"], idempotent, not conflicts,
        ))
        if not gates:
            raise RuntimeError("Canonical post-write validation gate failed")

        metadata = _metadata_by_vde(candidate)
        gap_inventory = build_gap_inventory(manifest, metadata)
        groups = group_gap_inventory(gap_inventory)
        gap_summary = _gap_summary(groups)
        _write_csv(output_dir / "research_gap_vde_inventory.csv", gap_inventory)
        _write_csv(output_dir / "research_gap_groups.csv", groups)
        _write_csv(output_dir / "research_gap_group_summary.csv", gap_summary)
        _write_csv(output_dir / "research_gap_priority_queue.csv", groups)
        _write_jsonl(output_dir / "research_agent_handoff.jsonl", groups)
        (output_dir / "research_gap_report.md").write_text(_gap_report(gap_inventory, groups), encoding="utf-8")
        _write_csv(output_dir / "canonical_conflicts.csv", conflicts, list(manifest[0]) if manifest else ())
        write_summary = [
            {"metric": "macro_component_resolution_inserted", "value": macro_first["inserted_resolution_rows"]},
            {"metric": "macro_links_inserted", "value": macro_first["inserted_links"]},
            {"metric": "vde_rows_projected", "value": first_apply["vde_rows_changed"]},
            {"metric": "fine_reference_resolutions_inserted", "value": first_apply["fine_component_resolutions_inserted"]},
            {"metric": "fine_supporting_links_inserted", "value": first_apply["fine_supporting_links_inserted"]},
            {"metric": "conflicts_skipped", "value": len(conflicts)},
            {"metric": "edrive_retained_without_transmission_projection", "value": semantic["edrive_macro_resolutions"]},
        ]
        _write_csv(output_dir / "canonical_write_summary.csv", write_summary)
        before_after = {"candidate_path": str(candidate), "before": pre, "after_first": post_first, "after_second": post}
        (output_dir / "canonical_before_after_hashes.json").write_text(json.dumps(before_after, indent=2), encoding="utf-8")
        (output_dir / "canonical_semantic_verification.json").write_text(json.dumps(semantic, indent=2), encoding="utf-8")
        (output_dir / "canonical_integrity_check.txt").write_text(
            f"quick_check={post['quick_check']}\nforeign_key_issues={post['foreign_key_issues']}\n"
            f"protected_fields_unchanged={'YES' if protected_unchanged else 'NO'}\n"
            f"core_row_counts_unchanged={'YES' if core_counts_unchanged else 'NO'}\n"
            f"schema_unchanged={'YES' if schema_unchanged else 'NO'}\n", encoding="utf-8"
        )
        (output_dir / "canonical_idempotency_check.txt").write_text(
            f"second_macro_resolution_inserts={macro_second['inserted_resolution_rows']}\n"
            f"second_macro_link_inserts={macro_second['inserted_links']}\n"
            f"second_vde_changes={second_apply['vde_rows_changed']}\n"
            f"second_fine_resolution_inserts={second_apply['fine_component_resolutions_inserted']}\n"
            f"second_fine_link_inserts={second_apply['fine_supporting_links_inserted']}\n"
            f"second_hash_unchanged={'YES' if first_hash == second_hash else 'NO'}\n"
            f"IDEMPOTENT={'YES' if idempotent else 'NO'}\n", encoding="utf-8"
        )
        final = FINAL_WITH_GAPS if groups else FINAL_COMPLETE
        result = {
            "method_version": METHOD_VERSION, "scale_method_version": SCALE_METHOD_VERSION,
            "final_recommendation": final, "candidate_path": str(candidate),
            "candidate_sha256_before": pre["sha256"], "candidate_sha256_after": post["sha256"],
            "backup_path": str(backup), "backup_sha256": backup_hash,
            "validated_manifest_sha256": manifest_hash, "macro_first": macro_first,
            "first_apply": first_apply, "macro_second": macro_second, "second_apply": second_apply,
            "semantic_verification": semantic, "integrity": {
                "quick_check": post["quick_check"], "foreign_key_issues": post["foreign_key_issues"],
                "protected_fields_unchanged": protected_unchanged,
                "core_row_counts_unchanged": core_counts_unchanged, "schema_unchanged": schema_unchanged,
                "idempotent": idempotent,
            },
            "research_gap_vde_rows": len(gap_inventory), "research_gap_groups": len(groups),
            "research_gap_unique_vdes": len({int(row['vde_id']) for row in gap_inventory}),
            "gap_families": dict(sorted(Counter(row["gap_family"] for row in gap_inventory).items())),
            "external_search_count": 0, "llm_call_count": 0, "production_db_written": False,
            "canonical_candidate_written": True,
        }
        (output_dir / "execution_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        (output_dir / "canonical_materialization_report.md").write_text(
            _canonical_report(result, write_summary, groups), encoding="utf-8"
        )
        return result
    except Exception as exc:
        # The contract requires a recoverable canonical write.  Restore exactly
        # the validated source when any post-backup gate fails.
        shutil.copy2(backup, candidate)
        if file_sha256(candidate) != pre["sha256"]:
            raise RuntimeError("Canonical rollback failed") from exc
        result = {
            "method_version": METHOD_VERSION, "final_recommendation": FAILED,
            "candidate_path": str(candidate), "candidate_restored_from_backup": True,
            "candidate_sha256": file_sha256(candidate), "backup_path": str(backup),
            "error": f"{type(exc).__name__}: {exc}", "production_db_written": False,
        }
        (output_dir / "execution_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        raise


def _canonical_report(summary: dict[str, Any], write_rows: list[dict[str, Any]], groups: list[dict[str, Any]]) -> str:
    semantic, integrity = summary["semantic_verification"], summary["integrity"]
    lines = [
        "# Canonical Component Population & Research Gap Handoff", "",
        f"## {summary['final_recommendation']}", "",
        "## Canonical state and backup", "",
        f"- Candidate: `{summary['candidate_path']}`",
        f"- SHA256 before: `{summary['candidate_sha256_before']}`",
        f"- SHA256 after: `{summary['candidate_sha256_after']}`",
        f"- Backup: `{summary['backup_path']}`",
        f"- Backup SHA256: `{summary['backup_sha256']}`", "",
        "## Materialization", "", "| Metric | Value |", "|---|---:|",
    ]
    lines.extend(f"| {row['metric']} | {row['value']} |" for row in write_rows)
    lines.extend(["", "## Verification", "",
        f"- quick_check: **{integrity['quick_check']}**",
        f"- foreign key issues: **{integrity['foreign_key_issues']}**",
        f"- protected fields unchanged: **{integrity['protected_fields_unchanged']}**",
        f"- core row counts unchanged: **{integrity['core_row_counts_unchanged']}**",
        f"- schema unchanged: **{integrity['schema_unchanged']}**",
        f"- slot-to-macro mismatches: **{semantic['slot_to_macro_mismatches']}**",
        f"- unauthorized fine projections: **{semantic['unauthorized_fine_vde_projections']}**",
        f"- idempotent second run: **{integrity['idempotent']}**", "",
        "## Research-gap handoff", "",
        f"- VDE-gap rows: **{summary['research_gap_vde_rows']:,}**",
        f"- Unique reusable research groups: **{summary['research_gap_groups']:,}**",
        f"- Unique VDEs represented: **{summary['research_gap_unique_vdes']:,}**",
        "- External searches / LLM calls: **0 / 0**", "",
        "## Top 20 groups", "", "| Priority | Group | Family | VDEs | Goal |", "|---|---|---|---:|---|",
    ])
    for row in groups[:20]:
        lines.append(f"| {row['priority']} | {row['research_group_id']} | {row['gap_family']} | {row['vde_count']} | {row['research_goal']} |")
    lines.extend(["", "## Next step", "", "Run the Technical Research Agent against `research_agent_handoff.jsonl`, writing results to staging evidence only. Do not directly overwrite canonical VDE component values.", ""])
    return "\n".join(lines)


__all__ = [
    "EXPECTED_MANIFEST_SHA256", "EXPECTED_SOURCE_SHA256", "build_gap_inventory",
    "canonical_candidate_path", "execute_canonical_materialization_and_handoff",
    "group_gap_inventory",
]
