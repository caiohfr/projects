#!/usr/bin/env python3
"""Run the normalized rule-only component-prior matcher without DB writes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_rule_only_component_matching import (
    _objects,
    _open_read_only,
    load_reference_catalog,
    sha256_file,
)
from src.vde_core.component_prior_matching import ReferencePrior
from src.vde_core.component_prior_matching_vnext import (
    MATCH_BOUNDARIES,
    MATCH_LEVELS,
    ApplicationClass,
    NormalizedTechnicalIdentity,
    PriorDecision,
    match_component_priors_vnext,
    normalize_application_class,
    normalize_drive,
    normalize_electrification,
    normalize_transmission,
)


DEFAULT_LEGACY_DB = REPO_ROOT / "data" / "db" / "archive" / "eco_drive_legacy_pre_sprint12.db"


def _json_dump(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
    )


def _identity_token(value: Any) -> str | None:
    """Case/outer-whitespace normalization only; never fuzzy normalization."""
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None


def _open_legacy_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def load_legacy_category_lookup(
    connection: sqlite3.Connection,
) -> dict[tuple[str, str, int], tuple[str, ...]]:
    columns = {row[1] for row in connection.execute("PRAGMA table_info(vde_db)")}
    if not {"make", "model", "year", "category"}.issubset(columns):
        raise ValueError("Legacy vde_db lacks make/model/year/category")
    buckets: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for row in connection.execute(
        "SELECT make,model,year,category FROM vde_db "
        "WHERE make IS NOT NULL AND model IS NOT NULL "
        "AND year IS NOT NULL AND category IS NOT NULL"
    ):
        make = _identity_token(row["make"])
        model = _identity_token(row["model"])
        category = _identity_token(row["category"])
        try:
            year = int(row["year"])
        except (TypeError, ValueError):
            continue
        if make and model and category:
            buckets[(make, model, year)].add(category)
    return {key: tuple(sorted(values)) for key, values in sorted(buckets.items())}


def _program_years(program: dict[str, Any]) -> tuple[int, ...]:
    try:
        start = int(program["model_year_from"])
        end = int(program.get("model_year_to") or start)
    except (TypeError, ValueError):
        return ()
    if end < start or end - start > 10:
        return ()
    return tuple(range(start, end + 1))


def resolve_legacy_application_class(
    program: dict[str, Any],
    lookup: dict[tuple[str, str, int], tuple[str, ...]],
) -> tuple[ApplicationClass, dict[str, Any]]:
    make = _identity_token(program.get("commercial_make"))
    model = _identity_token(program.get("commercial_model"))
    years = _program_years(program)
    raw_categories: set[str] = set()
    matched_keys: list[dict[str, Any]] = []
    if make and model:
        for year in years:
            values = lookup.get((make, model, year), ())
            if values:
                raw_categories.update(values)
                matched_keys.append({"make": make, "model": model, "model_year": year})
    application = normalize_application_class(
        raw_categories,
        source="legacy.vde_db.category:EXACT_MAKE_MODEL_MODEL_YEAR_LOOKUP",
    )
    audit = {
        "lookup_mode": "CASE_INSENSITIVE_TRIMMED_EXACT_EQUALITY_ONLY",
        "lookup_make_raw": program.get("commercial_make"),
        "lookup_model_raw": program.get("commercial_model"),
        "lookup_years": list(years),
        "matched_keys": matched_keys,
        "legacy_categories": list(application.raw_values),
        "status": application.status,
    }
    return application, audit


def _require_candidate_schema(connection: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    required = {
        "program",
        "vehicle_configuration",
        "vde",
        "fuelcons",
        "component_db",
        "component_resolution",
    }
    missing = required - tables
    if missing:
        raise ValueError(f"Canonical DB is missing required tables: {sorted(missing)}")


def load_population(
    connection: sqlite3.Connection,
    legacy_lookup: dict[tuple[str, str, int], tuple[str, ...]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    programs = {
        str(row["program_id"]): dict(row)
        for row in connection.execute("SELECT * FROM program ORDER BY program_id")
    }
    configurations = {
        str(row["vehicle_configuration_id"]): dict(row)
        for row in connection.execute(
            "SELECT * FROM vehicle_configuration ORDER BY vehicle_configuration_id"
        )
    }
    vdes_by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    vdes: list[dict[str, Any]] = []
    for row in connection.execute("SELECT * FROM vde ORDER BY id"):
        item = dict(row)
        vdes.append(item)
        vdes_by_config[str(item["vehicle_configuration_id"])].append(item)

    electrification_by_config: dict[str, set[str]] = defaultdict(set)
    for row in connection.execute(
        """
        SELECT v.vehicle_configuration_id,f.electrification
        FROM fuelcons f JOIN vde v ON v.id=f.vde_id
        WHERE f.electrification IS NOT NULL
        ORDER BY v.vehicle_configuration_id,f.id
        """
    ):
        raw = str(row["electrification"]).strip()
        if raw:
            electrification_by_config[str(row["vehicle_configuration_id"])].add(raw)

    population: list[dict[str, Any]] = []
    for configuration_id, configuration in configurations.items():
        program = programs.get(str(configuration.get("program_id")), {})
        linked_vdes = vdes_by_config.get(configuration_id, [])
        application, application_audit = resolve_legacy_application_class(program, legacy_lookup)
        identity = NormalizedTechnicalIdentity(
            vehicle_configuration_id=configuration_id,
            drive=normalize_drive(configuration.get("drive_system")),
            transmission=normalize_transmission(configuration.get("transmission_type")),
            electrification=normalize_electrification(
                electrification_by_config.get(configuration_id, set())
            ),
            application_class=application,
        )
        population.append(
            {
                "identity": identity,
                "configuration": configuration,
                "program": program,
                "vdes": linked_vdes,
                "current_vde_categories": tuple(
                    sorted(
                        {
                            str(row["category"]).strip()
                            for row in linked_vdes
                            if row.get("category") is not None and str(row["category"]).strip()
                        }
                    )
                ),
                "application_lookup_audit": application_audit,
            }
        )
    population.sort(key=lambda item: item["identity"].vehicle_configuration_id)
    return population, vdes


def _decision_row(configuration_id: str, decision: PriorDecision) -> dict[str, Any]:
    return {
        "vehicle_configuration_id": configuration_id,
        "domain": decision.domain,
        "boundary": decision.boundary,
        "component_slot": decision.component_slot,
        "match_level": decision.match_level,
        "candidate_component_ids_json": _json_dump(list(decision.candidate_component_ids)),
        "candidate_resolution_ids_json": _json_dump(list(decision.candidate_resolution_ids)),
        "rule_id": decision.rule_id,
        "rule_version": decision.rule_version,
        "matched_on_json": _json_dump(list(decision.matched_on)),
        "missing_discriminants_json": _json_dump(list(decision.missing_discriminants)),
        "reason": decision.reason,
    }


def evaluate_population(
    population: list[dict[str, Any]],
    catalog: list[ReferencePrior],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, tuple[PriorDecision, ...]]]:
    config_rows: list[dict[str, Any]] = []
    vde_rows: list[dict[str, Any]] = []
    decisions_by_config: dict[str, tuple[PriorDecision, ...]] = {}
    for item in population:
        identity: NormalizedTechnicalIdentity = item["identity"]
        decisions = tuple(match_component_priors_vnext(identity, catalog))
        decisions_by_config[identity.vehicle_configuration_id] = decisions
        for decision in decisions:
            config_rows.append(_decision_row(identity.vehicle_configuration_id, decision))
        for vde in item["vdes"]:
            for decision in decisions:
                row = _decision_row(identity.vehicle_configuration_id, decision)
                row["vde_id"] = vde["id"]
                vde_rows.append(row)
    return config_rows, vde_rows, decisions_by_config


def normalization_rows(population: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in population:
        identity: NormalizedTechnicalIdentity = item["identity"]
        program = item["program"]
        rows.append(
            {
                "vehicle_configuration_id": identity.vehicle_configuration_id,
                "commercial_make": program.get("commercial_make"),
                "commercial_model": program.get("commercial_model"),
                "model_year_from": program.get("model_year_from"),
                "model_year_to": program.get("model_year_to"),
                "drive_raw_json": _json_dump(list(identity.drive.raw_values)),
                "drive_normalized": identity.drive.normalized,
                "drive_status": identity.drive.status,
                "drive_source": identity.drive.source,
                "transmission_raw_json": _json_dump(list(identity.transmission.raw_values)),
                "transmission_normalized": identity.transmission.normalized,
                "transmission_status": identity.transmission.status,
                "transmission_source": identity.transmission.source,
                "electrification_raw_json": _json_dump(list(identity.electrification.raw_values)),
                "electrification_normalized": identity.electrification.normalized,
                "electrification_status": identity.electrification.status,
                "electrification_source": identity.electrification.source,
                "legacy_category_raw_json": _json_dump(list(identity.application_class.raw_values)),
                "application_class_options_json": _json_dump(
                    list(identity.application_class.normalized_options)
                ),
                "application_class_status": identity.application_class.status,
                "application_class_source": identity.application_class.source,
                "current_vde_categories_json": _json_dump(list(item["current_vde_categories"])),
                "legacy_lookup_audit_json": _json_dump(item["application_lookup_audit"]),
            }
        )
    return rows


def build_unresolved_queue(
    population: list[dict[str, Any]],
    decisions_by_config: dict[str, tuple[PriorDecision, ...]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in population:
        identity: NormalizedTechnicalIdentity = item["identity"]
        decisions = decisions_by_config[identity.vehicle_configuration_id]
        unresolved = tuple(
            decision
            for decision in decisions
            if decision.match_level not in {"EXACT_RULE_MATCH", "STRONG_RULE_MATCH"}
        )
        if not unresolved:
            continue
        key_payload = {"vehicle_configuration_id": identity.vehicle_configuration_id}
        technical_key = hashlib.sha256(_json_dump(key_payload).encode("utf-8")).hexdigest()[:20].upper()
        program = item["program"]
        rows.append(
            {
                "technical_identity_key": technical_key,
                "vehicle_configuration_id": identity.vehicle_configuration_id,
                "commercial_make": program.get("commercial_make"),
                "commercial_model": program.get("commercial_model"),
                "model_year_from": program.get("model_year_from"),
                "model_year_to": program.get("model_year_to"),
                "unresolved_decisions_json": _json_dump(
                    [
                        {
                            "boundary": decision.boundary,
                            "slot": decision.component_slot,
                            "level": decision.match_level,
                            "rule_id": decision.rule_id,
                        }
                        for decision in unresolved
                    ]
                ),
                "candidate_component_ids_json": _json_dump(
                    sorted(
                        {
                            component_id
                            for decision in unresolved
                            for component_id in decision.candidate_component_ids
                        }
                    )
                ),
                "affected_vde_count": len(item["vdes"]),
            }
        )
    return rows


def build_summary(
    population: list[dict[str, Any]],
    vdes: list[dict[str, Any]],
    catalog: list[ReferencePrior],
    decisions_by_config: dict[str, tuple[PriorDecision, ...]],
    unresolved_queue: list[dict[str, Any]],
) -> dict[str, Any]:
    by_domain: dict[str, Any] = {}
    for domain in MATCH_BOUNDARIES:
        config_counts: Counter[str] = Counter()
        vde_counts: Counter[str] = Counter()
        for item in population:
            config_id = item["identity"].vehicle_configuration_id
            for decision in decisions_by_config[config_id]:
                if decision.domain == domain:
                    config_counts[decision.match_level] += 1
                    vde_counts[decision.match_level] += len(item["vdes"])
        by_domain[domain] = {
            "configuration_slot_decisions": {
                level: config_counts[level] for level in MATCH_LEVELS
            },
            "vde_slot_projections": {level: vde_counts[level] for level in MATCH_LEVELS},
        }

    overall_config: Counter[str] = Counter()
    overall_vde: Counter[str] = Counter()
    affected_configs: set[str] = set()
    affected_vdes: set[int] = set()
    strong_configs: set[str] = set()
    strong_vdes: set[int] = set()
    for item in population:
        config_id = item["identity"].vehicle_configuration_id
        decisions = decisions_by_config[config_id]
        vde_ids = {int(row["id"]) for row in item["vdes"]}
        for decision in decisions:
            overall_config[decision.match_level] += 1
            overall_vde[decision.match_level] += len(item["vdes"])
        if any(decision.match_level != "NO_MATCH" for decision in decisions):
            affected_configs.add(config_id)
            affected_vdes.update(vde_ids)
        if any(decision.match_level == "STRONG_RULE_MATCH" for decision in decisions):
            strong_configs.add(config_id)
            strong_vdes.update(vde_ids)

    normalization = {
        "drive": Counter(item["identity"].drive.status for item in population),
        "transmission": Counter(item["identity"].transmission.status for item in population),
        "electrification": Counter(item["identity"].electrification.status for item in population),
        "application_class": Counter(
            item["identity"].application_class.status for item in population
        ),
    }
    return {
        "population": {
            "vdes": len(vdes),
            "vehicle_configurations": len(population),
            "synthetic_references": len(catalog),
        },
        "v1_comparison": {
            "v1_matched_vde_domain_projections": 0,
            "v1_result": "0 matches",
            "vnext_configuration_slot_decisions": {
                level: overall_config[level] for level in MATCH_LEVELS
            },
            "vnext_vde_slot_projections": {
                level: overall_vde[level] for level in MATCH_LEVELS
            },
        },
        "by_domain": by_domain,
        "affected": {
            "vdes_with_any_rule_signal": len(affected_vdes),
            "vehicle_configurations_with_any_rule_signal": len(affected_configs),
            "vdes_with_strong_match": len(strong_vdes),
            "vehicle_configurations_with_strong_match": len(strong_configs),
            "unresolved_technical_identities": len(unresolved_queue),
        },
        "normalization_status_by_configuration": {
            field: dict(sorted(counts.items())) for field, counts in normalization.items()
        },
        "queue_deduplication": {
            "key_basis": "vehicle_configuration_id",
            "domain_is_not_part_of_key": True,
            "rows": len(unresolved_queue),
        },
    }


def _csv_text(rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> str:
    fieldnames = list(fields or (list(rows[0]) if rows else []))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _coverage_report(summary: dict[str, Any]) -> str:
    vnext = summary["v1_comparison"]["vnext_vde_slot_projections"]
    lines = [
        "# Rule-only component prior matcher vNext",
        "",
        "Read-only deterministic run. No web, LLM, ML, commercial-name component matching, "
        "ABC-based selection, external enrichment, or database writes were used.",
        "",
        "## v1.0 comparison",
        "",
        "- v1.0: **0 matches**",
        f"- vNext STRONG: **{vnext['STRONG_RULE_MATCH']:,}** VDE/slot projections",
        f"- vNext CATEGORY: **{vnext['CATEGORY_RULE_MATCH']:,}** VDE/slot projections",
        f"- vNext AMBIGUOUS: **{vnext['AMBIGUOUS_RULE_MATCH']:,}** VDE/slot projections",
        f"- vNext NO_MATCH: **{vnext['NO_MATCH']:,}** VDE/slot projections",
        "- vNext EXACT: **0** by design; synthetic references do not prove exact hardware identity.",
        "",
        "AXLE and HUB_BEARING each have FRONT and REAR slot projections; BRAKE and "
        "TRANSMISSION each have one projection per VDE.",
        "",
        "## By boundary",
        "",
        "| Boundary | Strong | Category | Ambiguous | No match |",
        "|---|---:|---:|---:|---:|",
    ]
    for domain in MATCH_BOUNDARIES:
        counts = summary["by_domain"][domain]["vde_slot_projections"]
        lines.append(
            f"| {domain} | {counts['STRONG_RULE_MATCH']:,} | "
            f"{counts['CATEGORY_RULE_MATCH']:,} | {counts['AMBIGUOUS_RULE_MATCH']:,} | "
            f"{counts['NO_MATCH']:,} |"
        )
    affected = summary["affected"]
    lines.extend(
        [
            "",
            "## Population impact",
            "",
            f"- VDEs with any rule signal: **{affected['vdes_with_any_rule_signal']:,}**",
            f"- Unique configurations with any rule signal: **{affected['vehicle_configurations_with_any_rule_signal']:,}**",
            f"- VDEs with at least one strong match: **{affected['vdes_with_strong_match']:,}**",
            f"- Unique configurations with at least one strong match: **{affected['vehicle_configurations_with_strong_match']:,}**",
            f"- Unresolved technical identities: **{affected['unresolved_technical_identities']:,}**",
            "",
            "## Legacy application-class source",
            "",
            "The canonical database has no populated detailed class (`fuelcons.label_class` is empty). "
            "The run therefore reads the already-stored `legacy.vde_db.category` at runtime. "
            "Lookup uses only case-insensitive, trimmed exact equality on program make, model, and "
            "model year. It performs no fuzzy/token/model-family inference. The legacy ETL's own "
            "historical provenance is not upgraded; each lookup and miss is exported for review.",
            "",
            "## Queue identity",
            "",
            "The unresolved queue has one row per canonical `vehicle_configuration_id`, with all "
            "unresolved boundaries/slots aggregated. Domain is not part of the deduplication key.",
            "",
        ]
    )
    return "\n".join(lines)


def build_artifacts(
    *,
    normalization: list[dict[str, Any]],
    config_rows: list[dict[str, Any]],
    vde_rows: list[dict[str, Any]],
    unresolved_queue: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, bytes]:
    texts = {
        "normalization_audit.csv": _csv_text(normalization),
        "config_prior_matches.csv": _csv_text(config_rows),
        "vde_prior_matches.csv": _csv_text(vde_rows),
        "unresolved_identity_queue.csv": _csv_text(unresolved_queue),
        "reference_catalog.csv": _csv_text(inventory),
        "coverage_summary.json": _json_dump(summary, indent=2) + "\n",
        "coverage_report.md": _coverage_report(summary),
    }
    return {name: text.encode("utf-8") for name, text in texts.items()}


def run(db_path: Path, legacy_db_path: Path, output_dir: Path) -> dict[str, Any]:
    db_path = db_path.resolve(strict=True)
    legacy_db_path = legacy_db_path.resolve(strict=True)
    db_hash_before = sha256_file(db_path)
    legacy_hash_before = sha256_file(legacy_db_path)

    candidate = _open_read_only(db_path)
    legacy = _open_legacy_read_only(legacy_db_path)
    try:
        _require_candidate_schema(candidate)
        quick_check = candidate.execute("PRAGMA quick_check").fetchone()[0]
        foreign_key_issues = len(candidate.execute("PRAGMA foreign_key_check").fetchall())
        if quick_check != "ok" or foreign_key_issues:
            raise ValueError(
                f"Source DB integrity failed: quick_check={quick_check}, fk_issues={foreign_key_issues}"
            )
        candidate_objects_before = _objects(candidate)
        legacy_objects_before = _objects(legacy)
        legacy_lookup = load_legacy_category_lookup(legacy)
        catalog, inventory = load_reference_catalog(candidate)
        population, vdes = load_population(candidate, legacy_lookup)
        config_rows, vde_rows, decisions = evaluate_population(population, catalog)
        normalized = normalization_rows(population)
        unresolved = build_unresolved_queue(population, decisions)
        candidate_objects_after = _objects(candidate)
        legacy_objects_after = _objects(legacy)
    finally:
        candidate.close()
        legacy.close()

    db_hash_after = sha256_file(db_path)
    legacy_hash_after = sha256_file(legacy_db_path)
    if db_hash_before != db_hash_after or legacy_hash_before != legacy_hash_after:
        raise RuntimeError("A source DB hash changed during the read-only run")
    if candidate_objects_before != candidate_objects_after:
        raise RuntimeError("Canonical SQLite objects changed during the read-only run")
    if legacy_objects_before != legacy_objects_after:
        raise RuntimeError("Legacy SQLite objects changed during the read-only run")

    summary = build_summary(population, vdes, catalog, decisions, unresolved)
    summary["safety"] = {
        "canonical_db_path": str(db_path),
        "canonical_db_sha256_before": db_hash_before,
        "canonical_db_sha256_after": db_hash_after,
        "canonical_db_hash_unchanged": True,
        "legacy_db_path": str(legacy_db_path),
        "legacy_db_sha256_before": legacy_hash_before,
        "legacy_db_sha256_after": legacy_hash_after,
        "legacy_db_hash_unchanged": True,
        "quick_check": quick_check,
        "foreign_key_check_issue_count": foreign_key_issues,
        "no_tables_created": True,
        "no_rows_written": True,
        "external_search_used": False,
        "llm_used_for_classification": False,
        "ml_used": False,
        "agent_v2_started": False,
    }
    first = build_artifacts(
        normalization=normalized,
        config_rows=config_rows,
        vde_rows=vde_rows,
        unresolved_queue=unresolved,
        inventory=inventory,
        summary=summary,
    )
    second = build_artifacts(
        normalization=normalized,
        config_rows=config_rows,
        vde_rows=vde_rows,
        unresolved_queue=unresolved,
        inventory=inventory,
        summary=summary,
    )
    if first != second:
        raise RuntimeError("Deterministic artifact rebuild produced different bytes")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in first.items():
        (output_dir / name).write_bytes(payload)
    return {
        "status": "PASS",
        "output_dir": str(output_dir.resolve()),
        "canonical_db_sha256_before": db_hash_before,
        "canonical_db_sha256_after": db_hash_after,
        "legacy_db_sha256_before": legacy_hash_before,
        "legacy_db_sha256_after": legacy_hash_after,
        "quick_check": quick_check,
        "foreign_key_check_issue_count": foreign_key_issues,
        "configuration_match_rows": len(config_rows),
        "vde_match_rows": len(vde_rows),
        "unresolved_identity_rows": len(unresolved),
        "artifact_sha256": {
            name: hashlib.sha256(payload).hexdigest().upper()
            for name, payload in sorted(first.items())
        },
        "coverage": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--legacy-db", default=DEFAULT_LEGACY_DB, type=Path)
    parser.add_argument(
        "--output-dir", default=Path("artifacts/rule_only_vnext"), type=Path
    )
    args = parser.parse_args()
    try:
        result = run(args.db, args.legacy_db, args.output_dir)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2
    print(_json_dump(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
