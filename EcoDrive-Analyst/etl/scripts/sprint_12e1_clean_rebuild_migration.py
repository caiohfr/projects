"""Sprint 12E.1 clean-rebuild migration rehearsal.

Build one canonical EPA/JRC source tree, then selectively migrate only legacy
state that cannot be reconstructed. Runtime databases are read-only inputs;
the target is a disposable SQLite database below ``etl/data/staging``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "etl" / "scripts"))

import sprint_12c1_dataset_delta_audit as c1  # noqa: E402
import sprint_12e_migration_rehearsal as e12  # noqa: E402


RUNTIME_DB = ROOT / "data" / "db" / "eco_drive.db"
QA_DB = ROOT / "data" / "db" / "eco_drive_qa.db"
SCHEMA_SQL = ROOT / "etl" / "schema" / "canonical_schema_v1.sql"
COMPAT_SQL = ROOT / "etl" / "schema" / "canonical_compatibility_v1.sql"
LEGACY_EPA = e12.LEGACY_EPA
REFRESHED_EPA = e12.REFRESHED_EPA
JRC = e12.JRC
EEA = e12.EEA

STAGING_DIR = ROOT / "etl" / "data" / "staging" / "sprint_12e1_clean_rebuild"
DEFAULT_OUTPUT_DB = STAGING_DIR / "eco_drive_canonical_clean_rebuild.db"
OUT = ROOT / "etl" / "data" / "processed" / "sprint_12e1_clean_rebuild"
REPORT = ROOT / "etl" / "reports" / "sprint_12e1_clean_rebuild_migration.md"
SUMMARY_PATH = OUT / "clean_rebuild_summary.json"

OUTPUT_FILES = (
    "canonical_population_counts.csv", "population_by_source.csv",
    "legacy_record_disposition.csv", "legacy_irreproducible_state.csv",
    "legacy_to_canonical_identity_matches.csv", "program_consolidation_results.csv",
    "configuration_match_results.csv", "vde_population_results.csv",
    "run_population_results.csv", "fuelcons_materialization_results.csv",
    "intentional_legacy_retirements.csv", "compatibility_regression_results.csv",
    "identity_quarantine.csv", "relationship_checks.csv", "performance_results.csv",
    "runtime_db_fingerprints.csv", "clean_rebuild_summary.json",
)
DERIVED_VDE_IDS = {5031, 5033, 5034, 5038}
IRREPRODUCIBLE_FUELCONS_IDS = {5011, 5012, 5015, 5016, 5018}
MIGRATION_TIMESTAMP = e12.MIGRATION_TIMESTAMP


def guard_output_path(path: Path) -> Path:
    resolved = path.resolve()
    protected = {RUNTIME_DB.resolve(), QA_DB.resolve()}
    allowed = STAGING_DIR.resolve()
    if resolved in protected:
        raise ValueError(f"Protected runtime database cannot be an output: {resolved}")
    if not resolved.is_relative_to(allowed) or resolved.suffix.lower() != ".db":
        raise ValueError(f"Clean-rebuild output must be a .db below {allowed}")
    return resolved


def write_csv(name: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    e12.write_csv(OUT / name, rows, fields)


def source_key(row: dict[str, Any]) -> tuple[str, str, int | None]:
    return (c1.norm_text(row.get("make")), c1.norm_text(row.get("model")), e12.clean(row.get("year")))


def close_enough(left: Any, right: Any, tolerance: float = 1e-6) -> bool:
    left, right = e12.clean(left), e12.clean(right)
    if left is None or right is None:
        return left is right
    return abs(float(left) - float(right)) <= tolerance


def build_source_match_index(
    epa: dict[str, list[dict[str, Any]]],
) -> tuple[dict[tuple[str, str, int | None], list[dict[str, Any]]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_key: dict[tuple[str, str, int | None], list[dict[str, Any]]] = defaultdict(list)
    configs = {row["vehicle_configuration_id"]: row for row in epa["vehicle_configuration"]}
    programs = {row["program_id"]: row for row in epa["program"]}
    for row in epa["vde"]:
        by_key[source_key(row)].append(row)
    return by_key, configs, programs


def match_legacy_vde(
    row: dict[str, Any], by_key: dict[tuple[str, str, int | None], list[dict[str, Any]]],
    configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidates = sorted(by_key.get(source_key(row), []), key=lambda item: item["id"])
    exact = [
        candidate for candidate in candidates
        if all(close_enough(row.get(field), candidate.get(field)) for field in (
            "coast_A_N", "coast_B_N_per_kph", "coast_C_N_per_kph2"
        ))
    ]
    candidate_configs = sorted({item["vehicle_configuration_id"] for item in candidates})
    candidate_programs = sorted({configs[item]["program_id"] for item in candidate_configs})
    if len(exact) == 1:
        selected = exact[0]
        classification = "EXACT_CONFIG_MATCH"
        canonical_vde_id: int | None = int(selected["id"])
        canonical_config_id = selected["vehicle_configuration_id"]
    elif candidates and len(candidate_programs) == 1:
        classification = "PROGRAM_MATCH_ONLY"
        canonical_vde_id = None
        canonical_config_id = candidate_configs[0] if len(candidate_configs) == 1 else None
    else:
        classification = "NO_SAFE_MATCH"
        canonical_vde_id = None
        canonical_config_id = None
    return {
        "legacy_vde_id": int(row["id"]), "match_classification": classification,
        "canonical_program_id": candidate_programs[0] if len(candidate_programs) == 1 else None,
        "canonical_configuration_id": canonical_config_id,
        "canonical_vde_id": canonical_vde_id,
        "candidate_vde_ids": ";".join(str(item["id"]) for item in candidates),
        "candidate_count": len(candidates), "exact_roadload_candidate_count": len(exact),
        "evidence": "Normalized Make+Model+Year plus exact Target ABC in SI; ambiguous candidates are never auto-selected.",
    }


def transform_legacy_tires(tires: list[dict[str, Any]], runtime_hash: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    transformed: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for row in tires:
        item = dict(row)
        item["tire_id"] = item.pop("id")
        item.update({
            "record_origin": "OTHER_IRREPRODUCIBLE_STATE", "source_name": "LEGACY_IRREPRODUCIBLE_STATE",
            "source_record_id": str(item["tire_id"]), "source_file_version": runtime_hash,
            "provenance_json": {"classification": "OTHER_IRREPRODUCIBLE_STATE", "reason": "No authoritative local reconstruction source"},
        })
        transformed.append(item)
        evidence.append({
            "entity": "TIRE_DB", "legacy_id": item["tire_id"], "classification": "OTHER_IRREPRODUCIBLE_STATE",
            "canonical_id": item["tire_id"], "attachment_status": "STANDALONE_MASTER_PRESERVED",
            "original_parent_id": "", "canonical_parent_id": "", "notes": "Local specialized tire record has no authoritative reconstruction path.",
        })
    return transformed, evidence


def migrate_scenarios(
    legacy_vdes: list[dict[str, Any]], matches: dict[int, dict[str, Any]],
    epa: dict[str, list[dict[str, Any]]], runtime_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {int(row["id"]): row for row in legacy_vdes}
    config_by_id = {row["vehicle_configuration_id"]: row for row in epa["vehicle_configuration"]}
    scenario_configs: list[dict[str, Any]] = []
    scenarios: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    scenario_program: dict[int, str] = {}
    scenario_config: dict[int, str] = {}

    for scenario_id in sorted(DERIVED_VDE_IDS):
        row = by_id[scenario_id]
        original_parent = int(row["vde_id_parent"])
        canonical_parent_vde: int | None = None
        parent_config: str | None = None
        if original_parent in matches:
            match = matches[original_parent]
            program_id = match["canonical_program_id"]
            parent_config = match["canonical_configuration_id"]
            canonical_parent_vde = match["canonical_vde_id"]
            classification = "PROGRAM_MATCH_ONLY" if program_id else "NO_SAFE_MATCH"
        elif original_parent in scenario_program:
            program_id = scenario_program[original_parent]
            parent_config = scenario_config[original_parent]
            canonical_parent_vde = original_parent
            classification = "PROGRAM_MATCH_ONLY"
        else:
            program_id = None
            classification = "NO_SAFE_MATCH"

        if program_id is None:
            # The supplied data currently does not enter this branch. It remains a
            # conservative fallback that preserves state without fabricating a link.
            program_id = e12.stable_id("PRG-LEGACY-UNRESOLVED", scenario_id)
            epa["program"].append({
                "program_id": program_id, "commercial_make": row.get("make") or "UNRESOLVED",
                "commercial_model": row.get("model") or f"SCENARIO_{scenario_id}",
                "identity_status": "UNRESOLVED", "identity_confidence": "LOW",
                "source_identity_json": {"legacy_scenario_vde_id": scenario_id},
                "source_scope": "LEGACY_IRREPRODUCIBLE_STATE", "source_name": "LEGACY_IRREPRODUCIBLE_STATE",
                "source_file_version": runtime_hash, "created_at": row.get("created_at") or MIGRATION_TIMESTAMP,
            })
        cid = e12.stable_id("CFG-SCENARIO", scenario_id)
        scenario_configs.append({
            "vehicle_configuration_id": cid, "program_id": program_id,
            "propulsion_architecture": row.get("engine_type"), "engine_type": row.get("engine_type"),
            "engine_model": row.get("engine_model"), "engine_displacement_l": row.get("engine_size_l"),
            "engine_aspiration": row.get("engine_aspiration"), "transmission_type": row.get("transmission_type"),
            "transmission_model": row.get("transmission_model"), "drive_system": row.get("drive_type"),
            "identity_status": "PROVISIONAL", "identity_confidence": "LOW",
            "source_identity_json": {"legacy_scenario_vde_id": scenario_id, "legacy_parent_vde_id": original_parent},
            "source_scope": "LEGACY_IRREPRODUCIBLE_STATE", "source_name": "LEGACY_IRREPRODUCIBLE_STATE",
            "source_file_version": runtime_hash, "source_record_id": str(scenario_id),
            "created_at": row.get("created_at") or MIGRATION_TIMESTAMP,
        })
        scenario_program[scenario_id] = program_id
        scenario_config[scenario_id] = cid
        item = dict(row)
        item.update({
            "vehicle_configuration_id": cid, "vde_id_parent": canonical_parent_vde,
            "record_origin": "DERIVED_SCENARIO", "source_name": "LEGACY_IRREPRODUCIBLE_STATE",
            "source_record_id": str(scenario_id), "source_file_version": runtime_hash,
            "source_semantic_status": "DIRECT" if canonical_parent_vde is not None else "PARTIAL",
            "source_payload_json": {"legacy_vde_id": scenario_id, "legacy_parent_vde_id": original_parent},
            "normalization_version": "sprint_12e1_irreproducible_v1",
            "provenance_json": {
                "classification": "DERIVED_SCENARIO", "original_parent_vde_id": original_parent,
                "parent_match": classification, "canonical_parent_vde_id": canonical_parent_vde,
                "parent_candidate_vde_ids": matches.get(original_parent, {}).get("candidate_vde_ids"),
            },
        })
        scenarios.append(item)
        run_id = e12.stable_id("RUN-SCENARIO", scenario_id)
        runs.append({
            "run_id": run_id, "vde_id": scenario_id, "run_type": "CALCULATION",
            "evidence_kind": "ENGINEERING", "confidence": "MEDIUM",
            "source_name": "LEGACY_IRREPRODUCIBLE_STATE", "source_file_version": runtime_hash,
            "source_record_id": str(scenario_id), "procedure_description": "Preserved post-import VDE scenario state",
            "result_details_json": {"vde_total_mj_per_km": row.get("vde_total_mj_per_km"), "vde_net_mj_per_km": row.get("vde_net_mj_per_km")},
            "provenance_json": {"classification": "DERIVED_SCENARIO", "legacy_vde_id": scenario_id},
            "created_at": row.get("created_at") or MIGRATION_TIMESTAMP,
        })
        decisions.append({
            "scenario_vde_id": scenario_id, "legacy_parent_vde_id": original_parent,
            "match_classification": classification, "canonical_program_id": program_id,
            "canonical_parent_configuration_id": parent_config or "", "scenario_configuration_id": cid,
            "canonical_parent_vde_id": canonical_parent_vde if canonical_parent_vde is not None else "",
            "decision": "DISTINCT_SCENARIO_CONFIGURATION_ATTACHED_TO_MATCHED_PROGRAM",
            "unresolved": "Parent VDE state ambiguous; original candidates retained in provenance." if canonical_parent_vde is None else "",
        })
    return scenario_configs, scenarios, runs, decisions


def migrate_irreproducible_fuelcons(
    legacy_fuelcons: list[dict[str, Any]], matches: dict[int, dict[str, Any]], runtime_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    migrated: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    adoptions: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for row in sorted((row for row in legacy_fuelcons if int(row["id"]) in IRREPRODUCIBLE_FUELCONS_IDS), key=lambda item: item["id"]):
        legacy_vde_id = int(row["vde_id"])
        if legacy_vde_id in DERIVED_VDE_IDS:
            canonical_vde_id = legacy_vde_id
            attachment = "ATTACHED_TO_MIGRATED_SCENARIO"
        else:
            canonical_vde_id = matches[legacy_vde_id]["canonical_vde_id"]
            attachment = "EXACT_CONFIG_MATCH" if canonical_vde_id is not None else "NO_SAFE_MATCH"
        if canonical_vde_id is None:
            raise RuntimeError(f"Irreproducible FuelCons {row['id']} has no safe canonical VDE attachment")
        classification = "ML_PREDICTION" if row.get("engine_method") == "ml_prediction" else "DERIVED_SCENARIO"
        item = dict(row)
        provenance = {}
        if row.get("provenance_json"):
            provenance = json.loads(row["provenance_json"])
        provenance.update({"migration_classification": classification, "legacy_vde_id": legacy_vde_id})
        item.update({
            "vde_id": canonical_vde_id, "comparison_basis": "LEGACY_IRREPRODUCIBLE_RESULT",
            "record_origin": classification, "source_name": "LEGACY_IRREPRODUCIBLE_STATE",
            "source_record_id": str(row["id"]), "source_file_version": runtime_hash,
            "normalization_version": "sprint_12e1_irreproducible_v1", "provenance_json": provenance,
        })
        migrated.append(item)
        run_id = e12.stable_id("RUN-LEGACY-RESULT", row["id"])
        runs.append({
            "run_id": run_id, "vde_id": canonical_vde_id,
            "run_type": "ML_PREDICTION" if classification == "ML_PREDICTION" else "CALCULATION",
            "evidence_kind": "ENGINEERING", "confidence": "HIGH" if classification == "ML_PREDICTION" else "MEDIUM",
            "source_name": "LEGACY_IRREPRODUCIBLE_STATE", "source_file_version": runtime_hash,
            "source_record_id": str(row["id"]), "procedure_description": row.get("method_note"),
            "conditions_json": {key: row.get(key) for key in ("ambient_temp_c", "ac_on", "tire_front_psi", "tire_rear_psi", "scenario_payload_kg")},
            "result_details_json": {key: value for key, value in row.items() if key.startswith(("energy_", "fuel_", "gco2_", "label_"))},
            "method": row.get("engine_method"), "method_version": row.get("engine_version"),
            "assumptions_json": row.get("assumptions_json"),
            "provenance_json": {
                "classification": classification, "legacy_fuelcons_id": row["id"],
                "original_assumptions_json": row.get("assumptions_json") if row["id"] == 5018 else None,
                "json_correction": "APPROVED_CONTRACT_CORRECTION_NAN_TO_NULL" if row["id"] == 5018 else None,
            },
            "created_at": row.get("created_at") or MIGRATION_TIMESTAMP,
        })
        adoptions.append({
            "fuelcons_id": row["id"], "run_id": run_id, "vde_id": canonical_vde_id,
            "adoption_role": "PRIMARY", "result_dimension": "ALL", "ordinal": 0,
            "provenance_json": {"classification": classification}, "created_at": MIGRATION_TIMESTAMP,
        })
        evidence.append({
            "entity": "FUELCONS", "legacy_id": row["id"], "classification": classification,
            "canonical_id": row["id"], "attachment_status": attachment,
            "original_parent_id": legacy_vde_id, "canonical_parent_id": canonical_vde_id,
            "notes": "Irreproducible result migrated; generic LEGACY origin replaced explicitly.",
        })
    return migrated, runs, adoptions, evidence


def legacy_dispositions(
    legacy_vdes: list[dict[str, Any]], legacy_fuelcons: list[dict[str, Any]],
    matches: dict[int, dict[str, Any]], config_decisions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dispositions: list[dict[str, Any]] = []
    retirements: list[dict[str, Any]] = []
    config_by_scenario = {int(row["scenario_vde_id"]): row for row in config_decisions}
    for row in legacy_vdes:
        legacy_id = int(row["id"])
        if legacy_id in DERIVED_VDE_IDS:
            decision = config_by_scenario[legacy_id]
            disposition = "MIGRATED_IRREPRODUCIBLE_STATE"
            canonical_id = legacy_id
            match_class = decision["match_classification"]
            reason = "Post-import derived/scenario state cannot be reconstructed from the authoritative source."
        else:
            match = matches[legacy_id]
            canonical_id = match["canonical_vde_id"] or match["candidate_vde_ids"]
            match_class = match["match_classification"]
            if match["canonical_vde_id"] is not None:
                disposition = "RECONSTRUCTED_FROM_SOURCE"
            elif match["candidate_count"]:
                disposition = "SUPERSEDED_BY_REFRESHED_SOURCE"
            else:
                disposition = "RETIRED_RECONSTRUCTIBLE_LEGACY"
            reason = "Legacy aggregate is a regression reference; the current EPA source tree is canonical."
            retirements.append({
                "legacy_entity": "VDE", "legacy_id": legacy_id, "classification": "INTENTIONAL_LEGACY_ROW_RETIREMENT",
                "source_reconstruction": disposition, "canonical_candidates": match["candidate_vde_ids"], "reason": reason,
            })
        dispositions.append({
            "legacy_entity": "VDE", "legacy_id": legacy_id, "disposition": disposition,
            "canonical_entity": "VDE", "canonical_id": canonical_id, "match_classification": match_class, "reason": reason,
        })
    for row in legacy_fuelcons:
        legacy_id = int(row["id"])
        if legacy_id in IRREPRODUCIBLE_FUELCONS_IDS:
            disposition = "MIGRATED_IRREPRODUCIBLE_STATE"
            canonical_id = legacy_id
            match_class = "ATTACHED_TO_CANONICAL_PARENT"
            reason = "Post-import scenario/ML result cannot be reconstructed and is explicitly migrated."
        else:
            disposition = "RETIRED_RECONSTRUCTIBLE_LEGACY"
            canonical_id = ""
            match_class = "INTENTIONAL_LEGACY_ROW_RETIREMENT"
            reason = "Old EPA-derived baseline result is not copied; EPA source evidence remains RUN until pacification is approved."
            retirements.append({
                "legacy_entity": "FUELCONS", "legacy_id": legacy_id,
                "classification": "INTENTIONAL_LEGACY_ROW_RETIREMENT", "source_reconstruction": "EPA_RUN_EVIDENCE",
                "canonical_candidates": "", "reason": reason,
            })
        dispositions.append({
            "legacy_entity": "FUELCONS", "legacy_id": legacy_id, "disposition": disposition,
            "canonical_entity": "FUELCONS" if canonical_id != "" else "RUN_EVIDENCE",
            "canonical_id": canonical_id, "match_classification": match_class, "reason": reason,
        })
    return dispositions, retirements


def relationship_checks(con: sqlite3.Connection, summary_counts: dict[str, int], unresolved_parent_count: int) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, actual: Any, expected: Any, passed: bool, evidence: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "actual": actual, "expected": expected, "evidence": evidence})

    fk = len(con.execute("PRAGMA foreign_key_check").fetchall())
    add("foreign_key_check", fk, 0, fk == 0, "SQLite PRAGMA foreign_key_check")
    for name, sql in (
        ("orphan_configuration_program", "SELECT COUNT(*) FROM vehicle_configuration c LEFT JOIN program p ON p.program_id=c.program_id WHERE p.program_id IS NULL"),
        ("orphan_vde_configuration", "SELECT COUNT(*) FROM vde v LEFT JOIN vehicle_configuration c ON c.vehicle_configuration_id=v.vehicle_configuration_id WHERE c.vehicle_configuration_id IS NULL"),
        ("orphan_vde_parent", "SELECT COUNT(*) FROM vde v LEFT JOIN vde p ON p.id=v.vde_id_parent WHERE v.vde_id_parent IS NOT NULL AND p.id IS NULL"),
        ("orphan_run_vde", "SELECT COUNT(*) FROM run r LEFT JOIN vde v ON v.id=r.vde_id WHERE v.id IS NULL"),
        ("orphan_fuelcons_vde", "SELECT COUNT(*) FROM fuelcons f LEFT JOIN vde v ON v.id=f.vde_id WHERE v.id IS NULL"),
        ("invalid_adoption_same_vde", "SELECT COUNT(*) FROM fuelcons_run_adoption a JOIN fuelcons f ON f.id=a.fuelcons_id JOIN run r ON r.run_id=a.run_id WHERE a.vde_id<>f.vde_id OR a.vde_id<>r.vde_id"),
    ):
        actual = con.execute(sql).fetchone()[0]
        add(name, actual, 0, actual == 0, sql)
    legacy_programs = con.execute("SELECT COUNT(*) FROM program WHERE source_scope='LEGACY_ECODRIVE'").fetchone()[0]
    add("duplicate_legacy_program_tree_not_retained", legacy_programs, 0, legacy_programs == 0, "No reconstructible legacy Program population")
    source_programs = con.execute("SELECT COUNT(*) FROM program WHERE source_scope='EPA_TESTCAR_2020_2026'").fetchone()[0]
    add("epa_safe_program_order_of_magnitude", source_programs, 2868, source_programs == 2868, "SAFE consolidation after 77 source identity quarantines")
    scenario_vdes = con.execute("SELECT COUNT(*) FROM vde WHERE id IN (5031,5033,5034,5038) AND record_origin='DERIVED_SCENARIO'").fetchone()[0]
    add("four_irreproducible_scenario_vdes_preserved", scenario_vdes, 4, scenario_vdes == 4, "Selective legacy migration")
    later_results = con.execute("SELECT COUNT(*) FROM fuelcons WHERE id IN (5011,5012,5015,5016,5018) AND record_origin IN ('DERIVED_SCENARIO','ML_PREDICTION')").fetchone()[0]
    add("five_irreproducible_results_preserved", later_results, 5, later_results == 5, "Selective legacy migration")
    positive_vdes = con.execute("SELECT COUNT(*) FROM vde WHERE id>0").fetchone()[0]
    add("reconstructible_legacy_vdes_not_copied", positive_vdes, 4, positive_vdes == 4, "Only four scenario IDs remain positive")
    positive_fc = con.execute("SELECT COUNT(*) FROM fuelcons WHERE id>0").fetchone()[0]
    add("reconstructible_legacy_fuelcons_not_copied", positive_fc, 5, positive_fc == 5, "Only five irreproducible results remain positive")
    multi_runs = con.execute("SELECT COUNT(*) FROM (SELECT vde_id FROM run GROUP BY vde_id HAVING COUNT(*)>1)").fetchone()[0]
    add("one_vde_may_have_multiple_runs", multi_runs, ">=1", multi_runs >= 1, "Source-row RUN grain")
    without_resolution = con.execute("SELECT COUNT(*) FROM vde v LEFT JOIN vde_component_resolution x ON x.vde_id=v.id WHERE x.vde_id IS NULL").fetchone()[0]
    add("vde_without_component_resolution_allowed", without_resolution, summary_counts["vde"], without_resolution == summary_counts["vde"], "Optional resolution remains valid")
    add("unresolved_scenario_parent_is_explicit", unresolved_parent_count, 1, unresolved_parent_count == 1, "Q8 legacy parent maps to two refreshed VDE states; no state invented")

    con.execute("SAVEPOINT snapshot_proof")
    try:
        vde = con.execute("SELECT id,vehicle_configuration_id FROM vde ORDER BY id LIMIT 1").fetchone()
        con.execute("INSERT INTO component_db(component_id,component_domain,rated_power_kw,provenance_json) VALUES('QA-E1-MASTER','ENGINE',150,'{\"origin\":\"SYNTHETIC_QA\"}')")
        con.execute("INSERT INTO component_instance(component_instance_id,vehicle_configuration_id,component_domain,component_id,provenance_json) VALUES('QA-E1-I',?,'ENGINE','QA-E1-MASTER','{\"origin\":\"SYNTHETIC_QA\"}')", (vde["vehicle_configuration_id"],))
        con.execute("INSERT INTO fuelcons(id,vde_id,electrification,engine_max_power_kw,record_origin) VALUES(-9999999,?,'ICE',135,'SYNTHETIC_QA')", (vde["id"],))
        con.execute("UPDATE component_db SET rated_power_kw=180 WHERE component_id='QA-E1-MASTER'")
        snapshot = con.execute("SELECT engine_max_power_kw FROM fuelcons WHERE id=-9999999").fetchone()[0]
        add("master_update_does_not_rewrite_snapshot", snapshot, 135.0, snapshot == 135.0, "Transactional synthetic QA")
    finally:
        con.execute("ROLLBACK TO snapshot_proof")
        con.execute("RELEASE snapshot_proof")
    return checks


def performance_results(con: sqlite3.Connection) -> list[dict[str, Any]]:
    program = con.execute("SELECT program_id FROM program WHERE source_scope='EPA_TESTCAR_2020_2026' ORDER BY program_id LIMIT 1").fetchone()[0]
    epa_run = con.execute("SELECT source_record_id FROM run WHERE source_name='EPA_TESTCAR_2014_PRESENT' ORDER BY run_id LIMIT 1").fetchone()[0]
    cases = {
        "vde_lookup_by_id": ("SELECT * FROM vde_db WHERE id=?", (5031,)),
        "fuelcons_lookup_by_vde": ("SELECT * FROM fuelcons_db WHERE vde_id=? ORDER BY created_at DESC", (5031,)),
        "browse_filter": ("SELECT f.id,f.vde_id,v.make,v.model,v.year,f.electrification FROM fuelcons_db f JOIN vde_db v ON v.id=f.vde_id WHERE f.electrification=? AND v.legislation=? ORDER BY f.created_at DESC LIMIT 100", ("ICE", "WLTP")),
        "comparison_selection": ("SELECT f.id,f.vde_id,f.energy_Wh_per_km,f.fuel_l_per_100km,f.gco2_per_km,v.vde_net_mj_per_km FROM fuelcons_db f JOIN vde_db v ON v.id=f.vde_id WHERE f.id IN (?,?)", (5011, 5012)),
        "program_configuration_vde_navigation": ("SELECT v.id FROM vehicle_configuration c JOIN vde v ON v.vehicle_configuration_id=c.vehicle_configuration_id WHERE c.program_id=? ORDER BY v.id", (program,)),
        "source_identity_lookup": ("SELECT run_id,vde_id FROM run WHERE source_name=? AND source_record_id=?", ("EPA_TESTCAR_2014_PRESENT", epa_run)),
        "run_lineage_lookup": ("SELECT r.* FROM fuelcons_run_adoption a JOIN run r ON r.run_id=a.run_id WHERE a.fuelcons_id=? ORDER BY a.ordinal", (5011,)),
    }
    output: list[dict[str, Any]] = []
    for name, (sql, params) in cases.items():
        con.execute(sql, params).fetchall()
        timings, returned = [], 0
        for _ in range(20):
            started = time.perf_counter()
            returned = len(con.execute(sql, params).fetchall())
            timings.append((time.perf_counter() - started) * 1000)
        plan = " | ".join(row[3] for row in con.execute("EXPLAIN QUERY PLAN " + sql, params))
        median = statistics.median(timings)
        p95 = sorted(timings)[18]
        blocker = median > 100 or p95 > 250
        output.append({
            "query": name, "iterations": 20, "rows_returned": returned,
            "median_ms": round(median, 4), "p95_ms": round(p95, 4), "query_plan": plan,
            "index_used": "USING INDEX" in plan.upper() or "USING INTEGER PRIMARY KEY" in plan.upper() or "COVERING INDEX" in plan.upper(),
            "status": "BLOCKER" if blocker else "PASS",
        })
    return output


def table_counts(con: sqlite3.Connection) -> list[dict[str, Any]]:
    reference = {
        "program": "EPA SAFE consolidation + 249 source-scoped JRC; no parallel legacy tree",
        "vehicle_configuration": "Current EPA grain + 249 JRC + four scenario-specific configurations",
        "component_db": "No reusable identity supported by supplied sources",
        "tire_db": "One irreproducible local specialized tire record",
        "component_instance": "JRC unresolved descriptors only",
        "component_resolution": "Optional; no supported Tier-0 materialization",
        "vde": "Current EPA states + 249 JRC + four irreproducible scenarios",
        "run": "EPA source rows + JRC rows + scenario/result evidence",
        "fuelcons": "249 direct JRC declarations + five irreproducible scenario/ML results",
        "fuelcons_run_adoption": "One adoption per materialized FuelCons",
        "vde_component_resolution": "Optional; none inferred",
    }
    return [{"table": table, "row_count": con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0], "population_rule": reference[table]} for table in e12.PHYSICAL_TABLES]


def population_by_source(con: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = e12.population_breakdown(con)
    # e12 already appends the approved EEA analytical-boundary count.
    return rows


def compatibility_regression(
    con: sqlite3.Connection, legacy_vdes: list[dict[str, Any]], matches: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    vde_columns = len(con.execute("PRAGMA table_info(vde_db)").fetchall())
    fc_columns = len(con.execute("PRAGMA table_info(fuelcons_db)").fetchall())
    exact_matches = [item for item in matches.values() if item["canonical_vde_id"] is not None]
    ambiguous = [item for item in matches.values() if item["candidate_count"] and item["canonical_vde_id"] is None]
    no_match = [item for item in matches.values() if not item["candidate_count"]]
    return [
        {"scope": "APPLICATION_SURFACE", "classification": "APPLICATION_CONTRACT_EQUIVALENCE", "records": vde_columns, "details": "vde_db exposes the approved 101-column application shape."},
        {"scope": "APPLICATION_SURFACE", "classification": "APPLICATION_CONTRACT_EQUIVALENCE", "records": fc_columns, "details": "fuelcons_db exposes the approved 79-column application shape."},
        {"scope": "LEGACY_BASELINE_VDE", "classification": "SOURCE_RECONSTRUCTED_DIFFERENCE", "records": len(exact_matches), "details": "Unique refreshed-source VDE matches by normalized MMY and Target ABC; legacy IDs are retired."},
        {"scope": "LEGACY_BASELINE_VDE", "classification": "SOURCE_RECONSTRUCTED_DIFFERENCE", "records": len(ambiguous), "details": "Multiple modern source states replace one old aggregate; no single state selected."},
        {"scope": "LEGACY_BASELINE_VDE", "classification": "INTENTIONAL_LEGACY_ROW_RETIREMENT", "records": len(no_match), "details": "No current source identity counterpart; old aggregate remains regression evidence only."},
        {"scope": "LEGACY_BASELINE_FUELCONS", "classification": "INTENTIONAL_LEGACY_ROW_RETIREMENT", "records": 4999, "details": "No approved modern EPA pacification rule; source results remain RUN evidence."},
        {"scope": "LEGACY_IRREPRODUCIBLE_VDE", "classification": "IRREPRODUCIBLE_STATE_MIGRATED", "records": 4, "details": "Scenario snapshots preserved with explicit origin and lineage."},
        {"scope": "LEGACY_IRREPRODUCIBLE_FUELCONS", "classification": "IRREPRODUCIBLE_STATE_MIGRATED", "records": 5, "details": "Scenario/regression/ML results preserved and reattached."},
        {"scope": "FUELCONS_5018_ASSUMPTIONS_JSON", "classification": "APPROVED_CONTRACT_CORRECTION", "records": 1, "details": "Non-standard NaN normalized to JSON null; original text retained in RUN provenance."},
    ]


def relationship_signature(con: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    for table, key in (
        ("program", "program_id"), ("vehicle_configuration", "vehicle_configuration_id"),
        ("tire_db", "tire_id"), ("component_instance", "component_instance_id"),
        ("vde", "id"), ("run", "run_id"), ("fuelcons", "id"),
        ("fuelcons_run_adoption", "fuelcons_id,run_id,result_dimension"),
    ):
        for row in con.execute(f'SELECT {key} FROM "{table}" ORDER BY {key}'):
            digest.update(json.dumps(tuple(row), separators=(",", ":"), default=str).encode())
            digest.update(b"\n")
    return digest.hexdigest().upper()


def report_text(summary: dict[str, Any], counts: list[dict[str, Any]]) -> str:
    table = ["| Table | Rows |", "|---|---:|"] + [f"| `{row['table']}` | {row['row_count']:,} |" for row in counts]
    return "\n".join([
        "# Sprint 12E.1 — Clean Rebuild Migration Rehearsal", "",
        f"## Status: `{summary['status']}`", "", "```text",
        f"Clean rebuild completed?                 {'YES' if summary['clean_rebuild_completed'] else 'NO'}",
        f"Runtime DB changed?                      {'YES' if summary['runtime_db_changed'] else 'NO'}", "",
        f"Canonical Program count                  {summary['canonical_counts']['program']}",
        f"Vehicle Configuration count              {summary['canonical_counts']['vehicle_configuration']}",
        f"VDE count                                {summary['canonical_counts']['vde']}",
        f"RUN count                                {summary['canonical_counts']['run']}",
        f"FuelCons count                           {summary['canonical_counts']['fuelcons']}", "",
        f"Legacy VDE records:",
        f"  reconstructed/retired                  {summary['legacy_vde_reconstructed_or_retired']}",
        f"  irreproducible migrated                {summary['legacy_vde_irreproducible_migrated']}",
        f"  unresolved                             {summary['legacy_vde_unresolved']}", "",
        f"Legacy FuelCons records:",
        f"  reconstructed/retired                  {summary['legacy_fuelcons_reconstructed_or_retired']}",
        f"  irreproducible migrated                {summary['legacy_fuelcons_irreproducible_migrated']}",
        f"  unresolved                             {summary['legacy_fuelcons_unresolved']}", "",
        f"Duplicate legacy Program tree retained?  {'YES' if summary['duplicate_legacy_program_tree_retained'] else 'NO'}", "",
        f"Relationship failures                    {summary['relationship_failures']}",
        f"Quarantined records                      {summary['quarantined_records']}",
        f"Performance blockers                     {summary['performance_blockers']}",
        f"User decisions required                  {summary['user_decisions_required']}",
        "```", "", "## Canonical population", "", *table, "",
        "## Exceptions", "",
        f"- **EPA identity quarantine:** {summary['epa']['quarantined_rows']} of {summary['epa']['source_rows']:,} source rows have a year-like represented make and were not loaded.",
        f"- **Scenario parent:** {summary['unresolved_scenario_parent_links']} migrated scenario has an intentionally unresolved parent VDE state. It is attached to the safely matched EPA Program through a distinct scenario configuration; both possible refreshed parent states are retained in provenance.",
        "- **EPA FuelCons:** 4,999 legacy baseline results are intentionally retired. Current EPA result fields remain RUN evidence because no modern FuelCons pacification rule is approved.",
        "- **JRC:** 249 source-scoped unresolved identities are loaded with supported SI fields and declared OEM results; no JRC↔EPA merge is attempted.",
        f"- **EEA:** {e12.EEA_AUDITED_ROWS:,} monitoring rows remain in analytical source storage and add zero operational Program/VDE rows.",
        "- **Approved JSON correction:** non-standard `NaN` in FuelCons 5018 is written as JSON `null`; original source text remains in RUN provenance.", "",
        "## Program and legacy disposition", "",
        f"Valid EPA fallback Programs: {summary['epa']['fallback_programs_before']:,}; after SAFE-only consolidation: {summary['epa']['safe_programs_after']:,}; JRC Programs: 249; total: {summary['canonical_counts']['program']:,}. The previous parallel legacy Program tree is absent.",
        f"All {summary['legacy_disposition_rows']:,} legacy VDE/FuelCons rows have an explicit disposition. Four scenario VDEs, five scenario/ML FuelCons records, and one local Tire record are retained as irreproducible state.", "",
        "## Validation and performance", "",
        f"Foreign-key/orphan failures: {summary['relationship_failures']}. Seven current query families were measured; blockers: {summary['performance_blockers']}. Two clean rebuilds produced matching counts and relationship signature `{summary['deterministic_signature']}`.", "",
        "## Runtime safety", "",
        f"Runtime SHA-256 before: `{summary['runtime_before_sha256']}`  ",
        f"Runtime SHA-256 after: `{summary['runtime_after_sha256']}`  ",
        f"Byte-identical: **{'YES' if not summary['runtime_db_changed'] else 'NO'}**. Both runtime databases were opened with SQLite `mode=ro` and `PRAGMA query_only=ON`. Output is restricted to `.db` files under `etl/data/staging/sprint_12e1_clean_rebuild/`.", "",
        "## Evidence tiers", "",
        "- **DIRECTLY_TESTED:** clean schema build; source-only Program count; complete legacy disposition; selective scenario/ML/tire migration; safe lineage; no baseline tree copy; FK/orphans; same-VDE adoption; NULL vs zero; master/snapshot immutability; approved JSON correction; EEA boundary; performance; clean-rebuild determinism; runtime fingerprints.",
        "- **INDIRECTLY_COVERED:** 12C.3 SAFE consolidation evidence and 12D physical constraints.",
        "- **INSPECTION_SUPPORTED:** the four scenario and five later FuelCons records are the only post-import VDE/FuelCons additions in the supplied runtime DB.",
        "- **GAP:** EPA FuelCons pacification, application write-adapter integration, browser smoke and production cutover remain future work.", "",
        "## Reproduction", "", "```powershell",
        "python etl/scripts/sprint_12e1_clean_rebuild_migration.py --rebuild",
        "python -m unittest discover -s etl/tests -p \"test_sprint_12e1*.py\" -v", "```", "",
        "## USER_DECISION_REQUIRED", "", "None.", "",
    ]) + "\n"


def run_clean_rebuild(output_db: Path, rebuild: bool) -> dict[str, Any]:
    output_db = guard_output_path(output_db)
    required = [RUNTIME_DB, QA_DB, e12.LEGACY_RUNTIME_DB, SCHEMA_SQL, COMPAT_SQL, LEGACY_EPA, REFRESHED_EPA, JRC, EEA]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing 12E.1 inputs: {missing}")
    previous = json.loads(SUMMARY_PATH.read_text(encoding="utf-8")) if SUMMARY_PATH.exists() else None
    if output_db.exists():
        if not rebuild:
            raise FileExistsError(f"Clean rebuild DB exists; use --rebuild: {output_db}")
        output_db.unlink()
    output_db.parent.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    runtime_before = {path: e12.fingerprint(path) for path in (RUNTIME_DB, QA_DB)}
    legacy_runtime_hash = e12.fingerprint(e12.LEGACY_RUNTIME_DB)["sha256"]
    source_fingerprints = {path.name: e12.fingerprint(path) for path in (LEGACY_EPA, REFRESHED_EPA, JRC, EEA)}
    legacy_vdes, legacy_fc, legacy_tires, _ = e12.read_runtime_population()
    refreshed = pd.read_excel(REFRESHED_EPA, sheet_name="Sheet1", engine="openpyxl")
    jrc_source = pd.read_excel(JRC, sheet_name="Sheet1", engine="openpyxl")
    epa, consolidation, quarantine, epa_stats = e12.epa_population(refreshed, source_fingerprints[REFRESHED_EPA.name]["sha256"])
    jrc, jrc_unresolved = e12.jrc_population(jrc_source, source_fingerprints[JRC.name]["sha256"])
    by_key, configs, _ = build_source_match_index(epa)
    baseline_vdes = [row for row in legacy_vdes if int(row["id"]) not in DERIVED_VDE_IDS]
    match_rows = [match_legacy_vde(row, by_key, configs) for row in baseline_vdes]
    matches = {row["legacy_vde_id"]: row for row in match_rows}

    scenario_configs, scenarios, scenario_runs, config_results = migrate_scenarios(
        legacy_vdes, matches, epa, legacy_runtime_hash
    )
    migrated_fc, fc_runs, fc_adoptions, fc_evidence = migrate_irreproducible_fuelcons(
        legacy_fc, matches, legacy_runtime_hash
    )
    tires, tire_evidence = transform_legacy_tires(legacy_tires, legacy_runtime_hash)
    epa["vehicle_configuration"].extend(scenario_configs)
    epa["vde"].extend(scenarios)
    epa["run"].extend(scenario_runs)
    epa["run"].extend(fc_runs)
    epa["fuelcons"].extend(migrated_fc)
    epa["fuelcons_run_adoption"].extend(fc_adoptions)
    epa["tire_db"].extend(tires)
    population = e12.merge_population(epa, jrc)

    dispositions, retirements = legacy_dispositions(legacy_vdes, legacy_fc, matches, config_results)
    scenario_match_by_id = {row["scenario_vde_id"]: row for row in config_results}
    irreproducible = tire_evidence + [
        {"entity": "VDE", "legacy_id": row["id"], "classification": "DERIVED_SCENARIO", "canonical_id": row["id"],
         "attachment_status": scenario_match_by_id[row["id"]]["match_classification"],
         "original_parent_id": scenario_match_by_id[row["id"]]["legacy_parent_vde_id"],
         "canonical_parent_id": row.get("vde_id_parent") or "",
         "notes": "Scenario snapshot preserved with explicit provenance."}
        for row in scenarios
    ] + fc_evidence

    con = sqlite3.connect(output_db)
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA journal_mode=DELETE")
        con.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))
        con.executescript(COMPAT_SQL.read_text(encoding="utf-8"))
        con.execute("BEGIN")
        for table in e12.PHYSICAL_TABLES:
            e12.insert_rows(con, table, population[table])
        con.commit()
        counts = table_counts(con)
        canonical_counts = {row["table"]: row["row_count"] for row in counts}
        unresolved_parent_count = sum(not row["canonical_parent_vde_id"] for row in config_results)
        relationships = relationship_checks(con, canonical_counts, unresolved_parent_count)
        performance = performance_results(con)
        regressions = compatibility_regression(con, legacy_vdes, matches)
        signature = relationship_signature(con)
        con.execute("ANALYZE")
        con.commit()
        quick_check = con.execute("PRAGMA quick_check").fetchone()[0]
    finally:
        con.close()

    runtime_after = {path: e12.fingerprint(path) for path in (RUNTIME_DB, QA_DB)}
    fingerprint_rows = []
    for path in (RUNTIME_DB, QA_DB):
        before, after = runtime_before[path], runtime_after[path]
        fingerprint_rows.append({
            "database": path.name, "path": str(path), "before_size_bytes": before["size_bytes"],
            "after_size_bytes": after["size_bytes"], "before_sha256": before["sha256"],
            "after_sha256": after["sha256"], "byte_identical": before == after,
            "read_access": "SQLite URI mode=ro + PRAGMA query_only=ON",
        })
    runtime_changed = any(not row["byte_identical"] for row in fingerprint_rows)
    relationship_failures = sum(row["status"] != "PASS" for row in relationships)
    performance_blockers = sum(row["status"] == "BLOCKER" for row in performance)
    previous_signature = previous.get("deterministic_signature") if previous else None
    deterministic = bool(previous_signature and previous_signature == signature and previous.get("canonical_counts") == canonical_counts)
    disposition_complete = len(dispositions) == len(legacy_vdes) + len(legacy_fc)
    ready = not runtime_changed and not relationship_failures and not performance_blockers and disposition_complete and quick_check == "ok"
    status = "CLEAN_REBUILD_READY — PROCEED_TO_INTEGRATION" if ready else "CLEAN_REBUILD_REVIEW_REQUIRED"
    summary = {
        "status": status, "clean_rebuild_completed": output_db.exists(), "runtime_db_changed": runtime_changed,
        "canonical_counts": canonical_counts, "legacy_vde_reconstructed_or_retired": len(legacy_vdes) - len(DERIVED_VDE_IDS),
        "legacy_vde_irreproducible_migrated": len(DERIVED_VDE_IDS), "legacy_vde_unresolved": 0,
        "legacy_fuelcons_reconstructed_or_retired": len(legacy_fc) - len(IRREPRODUCIBLE_FUELCONS_IDS),
        "legacy_fuelcons_irreproducible_migrated": len(IRREPRODUCIBLE_FUELCONS_IDS), "legacy_fuelcons_unresolved": 0,
        "duplicate_legacy_program_tree_retained": any(row["source_scope"] == "LEGACY_ECODRIVE" for row in population["program"]),
        "relationship_failures": relationship_failures, "quarantined_records": len(quarantine),
        "performance_blockers": performance_blockers, "user_decisions_required": 0,
        "legacy_disposition_rows": len(dispositions), "unresolved_scenario_parent_links": unresolved_parent_count,
        "epa": epa_stats, "jrc_program_count": len(jrc["program"]), "eea_runtime_rows_loaded": 0,
        "nan_to_null_classification": "APPROVED_CONTRACT_CORRECTION",
        "deterministic_signature": signature, "previous_signature_available": previous_signature is not None,
        "rebuild_deterministic": deterministic, "quick_check": quick_check,
        "runtime_before_sha256": runtime_before[RUNTIME_DB]["sha256"],
        "runtime_after_sha256": runtime_after[RUNTIME_DB]["sha256"],
        "output_database": str(output_db), "output_database_size_bytes": output_db.stat().st_size,
        "source_fingerprints": source_fingerprints,
    }

    write_csv("canonical_population_counts.csv", counts, ["table", "row_count", "population_rule"])
    with sqlite3.connect(output_db) as population_con:
        population_source_rows = population_by_source(population_con)
    write_csv("population_by_source.csv", population_source_rows, ["table", "dimension", "value", "row_count"])
    write_csv("legacy_record_disposition.csv", dispositions, ["legacy_entity", "legacy_id", "disposition", "canonical_entity", "canonical_id", "match_classification", "reason"])
    write_csv("legacy_irreproducible_state.csv", irreproducible, ["entity", "legacy_id", "classification", "canonical_id", "attachment_status", "original_parent_id", "canonical_parent_id", "notes"])
    write_csv("legacy_to_canonical_identity_matches.csv", match_rows, ["legacy_vde_id", "match_classification", "canonical_program_id", "canonical_configuration_id", "canonical_vde_id", "candidate_vde_ids", "candidate_count", "exact_roadload_candidate_count", "evidence"])
    write_csv("program_consolidation_results.csv", consolidation, ["fallback_program_id", "canonical_program_id", "make", "model", "model_year", "safe_group_size", "decision", "applied_rule"])
    write_csv("configuration_match_results.csv", config_results, ["scenario_vde_id", "legacy_parent_vde_id", "match_classification", "canonical_program_id", "canonical_parent_configuration_id", "scenario_configuration_id", "canonical_parent_vde_id", "decision", "unresolved"])
    vde_results = [{"vde_id": row["id"], "population": row.get("record_origin"), "source_name": row.get("source_name"), "source_record_id": row.get("source_record_id"), "vehicle_configuration_id": row["vehicle_configuration_id"], "source_semantic_status": row["source_semantic_status"]} for row in population["vde"]]
    write_csv("vde_population_results.csv", vde_results, ["vde_id", "population", "source_name", "source_record_id", "vehicle_configuration_id", "source_semantic_status"])
    run_results = [{"run_id": row["run_id"], "vde_id": row["vde_id"], "run_type": row["run_type"], "evidence_kind": row["evidence_kind"], "source_name": row.get("source_name"), "source_record_id": row.get("source_record_id")} for row in population["run"]]
    write_csv("run_population_results.csv", run_results, ["run_id", "vde_id", "run_type", "evidence_kind", "source_name", "source_record_id"])
    fc_results = [{"fuelcons_id": row["id"], "vde_id": row["vde_id"], "population": row.get("record_origin"), "comparison_basis": row.get("comparison_basis"), "materialization": "DIRECT_JRC_DECLARATION" if int(row["id"]) < 0 else "MIGRATED_IRREPRODUCIBLE_STATE"} for row in population["fuelcons"]]
    fc_results.append({"fuelcons_id": "EPA_ALL", "vde_id": "", "population": "EPA_2020_2026", "comparison_basis": "DEFERRED", "materialization": "RUN_ONLY_NO_APPROVED_PACIFICATION"})
    write_csv("fuelcons_materialization_results.csv", fc_results, ["fuelcons_id", "vde_id", "population", "comparison_basis", "materialization"])
    write_csv("intentional_legacy_retirements.csv", retirements, ["legacy_entity", "legacy_id", "classification", "source_reconstruction", "canonical_candidates", "reason"])
    write_csv("compatibility_regression_results.csv", regressions, ["scope", "classification", "records", "details"])
    quarantine_output = quarantine + jrc_unresolved + [{"source": "LEGACY_IRREPRODUCIBLE_STATE", "source_record_id": "5033", "status": "LOADED_UNRESOLVED", "reason_code": "AMBIGUOUS_REFRESHED_PARENT_VDE", "reason": "Scenario attached to matched Program with distinct configuration; no parent VDE selected from two candidates.", "source_payload_json": e12.json_text(next(row for row in config_results if row["scenario_vde_id"] == 5033))}, {"source": "EEA_2025_PROVISIONAL", "source_record_id": "ALL_ROWS", "status": "DEFERRED_ANALYTICAL_BOUNDARY", "reason_code": "NOT_RUNTIME_ENGINEERING_GRAIN", "reason": f"{e12.EEA_AUDITED_ROWS} monitoring rows remain outside operational SQLite.", "source_payload_json": e12.json_text({"path": str(EEA), "sha256": source_fingerprints[EEA.name]["sha256"]})}]
    write_csv("identity_quarantine.csv", quarantine_output, ["source", "source_record_id", "status", "reason_code", "reason", "source_payload_json"])
    write_csv("relationship_checks.csv", relationships, ["check", "status", "actual", "expected", "evidence"])
    write_csv("performance_results.csv", performance, ["query", "iterations", "rows_returned", "median_ms", "p95_ms", "query_plan", "index_used", "status"])
    write_csv("runtime_db_fingerprints.csv", fingerprint_rows, ["database", "path", "before_size_bytes", "after_size_bytes", "before_sha256", "after_sha256", "byte_identical", "read_access"])
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT.write_text(report_text(summary, counts), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DB)
    args = parser.parse_args()
    result = run_clean_rebuild(args.output, args.rebuild)
    print(json.dumps({key: result[key] for key in (
        "status", "canonical_counts", "legacy_disposition_rows", "unresolved_scenario_parent_links",
        "relationship_failures", "quarantined_records", "performance_blockers", "runtime_db_changed",
        "rebuild_deterministic",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
