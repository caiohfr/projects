#!/usr/bin/env python3
"""Measure deterministic synthetic-component prior coverage without DB writes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
import sqlite3
import sys
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vde_core.component_prior_matching import (
    MATCH_DOMAINS,
    MATCH_LEVELS,
    PriorMatchResult,
    ReferencePrior,
    TechnicalIdentity,
    match_component_priors,
    normalize_drive_system,
)


SOURCE_NAME = "SYNTHETIC_REFERENCE"
PROFILE_FIELDS = {
    "vehicle_configuration": (
        "drive_system",
        "propulsion_architecture",
        "transmission_type",
        "transmission_model",
        "gear_count",
        "final_drive_ratio",
        "engine_type",
        "engine_aspiration",
        "engine_displacement_l",
        "engine_rated_power_kw",
        "architecture_properties_json",
    ),
    "vde": ("category", "drive_type", "transmission_type", "transmission_model"),
    "program": ("commercial_make", "commercial_model", "model_year_from", "model_year_to"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _json_load(value: Any) -> dict[str, Any]:
    if value is None or str(value).strip() == "":
        return {}
    try:
        loaded = json.loads(str(value))
    except (TypeError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _open_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _objects(connection: sqlite3.Connection) -> list[dict[str, str]]:
    return [
        {"type": row[0], "name": row[1], "sql": row[2] or ""}
        for row in connection.execute(
            """
            SELECT type,name,sql FROM sqlite_master
            WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'
            ORDER BY type,name
            """
        )
    ]


def _require_schema(connection: sqlite3.Connection) -> None:
    existing = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    required = {"program", "vehicle_configuration", "vde", "component_db", "component_resolution"}
    missing = required - existing
    if missing:
        raise ValueError(f"Canonical DB is missing required tables: {sorted(missing)}")


def load_reference_catalog(
    connection: sqlite3.Connection,
) -> tuple[list[ReferencePrior], list[dict[str, Any]]]:
    components = connection.execute(
        """
        SELECT component_id,component_domain,model,hardware_reference,
               custom_properties_json,provenance_json,source_name
        FROM component_db WHERE source_name=? ORDER BY component_id
        """,
        (SOURCE_NAME,),
    ).fetchall()
    resolutions = connection.execute(
        """
        SELECT component_resolution_id,boundary,method,resolved_A_N,
               resolved_B_N_per_kph,resolved_C_N_per_kph2,
               conditions_json,provenance_json
        FROM component_resolution ORDER BY component_resolution_id
        """
    ).fetchall()
    resolutions_by_component: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in resolutions:
        provenance = _json_load(row["provenance_json"])
        if provenance.get("synthetic_reference") is True:
            component_id = str(provenance.get("synthetic_reference_component_id") or "")
            resolutions_by_component[component_id].append(row)

    catalog: list[ReferencePrior] = []
    inventory: list[dict[str, Any]] = []
    for component in components:
        component_id = str(component["component_id"])
        linked = resolutions_by_component.get(component_id, [])
        if len(linked) != 1:
            raise ValueError(
                f"Synthetic component {component_id} has {len(linked)} linked resolutions; expected 1"
            )
        resolution = linked[0]
        properties = _json_load(component["custom_properties_json"])
        metadata = {
            "custom_properties": properties,
            "component_provenance": _json_load(component["provenance_json"]),
            "resolution_conditions": _json_load(resolution["conditions_json"]),
            "resolution_provenance": _json_load(resolution["provenance_json"]),
        }
        reference = ReferencePrior(
            component_id=component_id,
            component_resolution_id=str(resolution["component_resolution_id"]),
            component_domain=str(component["component_domain"]),
            boundary=str(resolution["boundary"]),
            model=component["model"],
            hardware_reference=component["hardware_reference"],
            application_class=properties.get("application_class"),
            drive_architecture=properties.get("drive_architecture"),
            position=properties.get("position"),
            metadata=metadata,
        )
        catalog.append(reference)
        inventory.append(
            {
                "component_id": reference.component_id,
                "component_resolution_id": reference.component_resolution_id,
                "component_domain": reference.component_domain,
                "boundary": reference.boundary,
                "model": reference.model,
                "hardware_reference": reference.hardware_reference,
                "method": resolution["method"],
                "resolved_A_N": resolution["resolved_A_N"],
                "resolved_B_N_per_kph": resolution["resolved_B_N_per_kph"],
                "resolved_C_N_per_kph2": resolution["resolved_C_N_per_kph2"],
                "reference_metadata_json": _json_dump(metadata),
            }
        )
    catalog.sort(key=lambda row: (row.boundary, row.component_id))
    inventory.sort(key=lambda row: (str(row["boundary"]), str(row["component_id"])))
    return catalog, inventory


def load_population(
    connection: sqlite3.Connection,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
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
    vdes_by_configuration: dict[str, list[dict[str, Any]]] = defaultdict(list)
    vdes = []
    for row in connection.execute("SELECT * FROM vde ORDER BY id"):
        item = dict(row)
        vdes.append(item)
        vdes_by_configuration[str(item["vehicle_configuration_id"])].append(item)

    identities: list[dict[str, Any]] = []
    for configuration_id, configuration in configurations.items():
        program = programs.get(str(configuration.get("program_id")), {})
        linked_vdes = vdes_by_configuration.get(configuration_id, [])
        categories = sorted(
            {str(row.get("category")).strip() for row in linked_vdes if row.get("category")}
        )
        vde_drives = sorted(
            {str(row.get("drive_type")).strip() for row in linked_vdes if row.get("drive_type")}
        )
        years = sorted(int(row["year"]) for row in linked_vdes if row.get("year") is not None)
        category = categories[0] if len(categories) == 1 else None
        drive_system = configuration.get("drive_system")
        if not drive_system and len(vde_drives) == 1:
            drive_system = vde_drives[0]
        identity = TechnicalIdentity(
            vehicle_configuration_id=configuration_id,
            drive_system=drive_system,
            category=category,
            propulsion_architecture=configuration.get("propulsion_architecture"),
            transmission_type=configuration.get("transmission_type"),
            transmission_model=configuration.get("transmission_model"),
            gear_count=configuration.get("gear_count"),
        )
        identities.append(
            {
                "identity": identity,
                "program": program,
                "configuration": configuration,
                "vdes": linked_vdes,
                "categories": categories,
                "years": years,
            }
        )
    identities.sort(key=lambda row: row["identity"].vehicle_configuration_id)
    raw = {
        "program": list(programs.values()),
        "vehicle_configuration": list(configurations.values()),
        "vde": vdes,
    }
    return identities, vdes, raw


def profile_discriminants(raw: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for table, fields in PROFILE_FIELDS.items():
        rows = raw[table]
        available = set(rows[0]) if rows else set()
        for field in fields:
            if field not in available:
                output.append(
                    {
                        "table": table,
                        "field": field,
                        "row_count": len(rows),
                        "non_null_count": 0,
                        "distinct_count": 0,
                        "top_values_json": "[]",
                        "unexpected_values_json": "[]",
                        "notes": "Column absent from actual schema",
                    }
                )
                continue
            values = [row.get(field) for row in rows]
            present = [value for value in values if value is not None and str(value).strip()]
            counts = Counter(str(value).strip() for value in present)
            top = [{"value": value, "count": count} for value, count in counts.most_common(20)]
            unexpected: list[str] = []
            notes = ""
            if field in {"drive_system", "drive_type"}:
                unexpected = sorted(value for value in counts if normalize_drive_system(value) is None)
            if field == "transmission_model" and not present:
                notes = "No transmission hardware/family identity is populated"
            if field == "category":
                notes = "Values are profiled as stored; no vehicle-class taxonomy is inferred"
            output.append(
                {
                    "table": table,
                    "field": field,
                    "row_count": len(rows),
                    "non_null_count": len(present),
                    "distinct_count": len(counts),
                    "top_values_json": _json_dump(top),
                    "unexpected_values_json": _json_dump(unexpected),
                    "notes": notes,
                }
            )
    return output


def _match_row(configuration_id: str, result: PriorMatchResult) -> dict[str, Any]:
    return {
        "vehicle_configuration_id": configuration_id,
        "domain": result.domain,
        "boundary": result.boundary,
        "match_level": result.match_level,
        "component_id": result.component_id,
        "component_resolution_id": result.component_resolution_id,
        "rule_id": result.rule_id,
        "rule_version": result.rule_version,
        "matched_on_json": _json_dump(list(result.matched_on)),
        "missing_discriminants_json": _json_dump(list(result.missing_discriminants)),
        "reason": result.reason,
    }


def evaluate_population(
    identities: list[dict[str, Any]],
    catalog: list[ReferencePrior],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, PriorMatchResult]]]:
    config_rows: list[dict[str, Any]] = []
    vde_rows: list[dict[str, Any]] = []
    result_map: dict[str, dict[str, PriorMatchResult]] = {}
    for row in identities:
        identity: TechnicalIdentity = row["identity"]
        results = match_component_priors(identity, catalog)
        result_map[identity.vehicle_configuration_id] = {result.domain: result for result in results}
        for result in results:
            config_rows.append(_match_row(identity.vehicle_configuration_id, result))
        for vde in row["vdes"]:
            for result in results:
                vde_rows.append(
                    {
                        "vde_id": vde["id"],
                        "vehicle_configuration_id": identity.vehicle_configuration_id,
                        "domain": result.domain,
                        "boundary": result.boundary,
                        "match_level": result.match_level,
                        "component_id": result.component_id,
                        "component_resolution_id": result.component_resolution_id,
                        "rule_id": result.rule_id,
                        "reason": result.reason,
                    }
                )
    return config_rows, vde_rows, result_map


def _technical_key_payload(row: dict[str, Any], result: PriorMatchResult) -> dict[str, Any]:
    identity: TechnicalIdentity = row["identity"]
    program = row["program"]
    years = row["years"]
    return {
        "domain": result.domain,
        "make": program.get("commercial_make"),
        "model": program.get("commercial_model"),
        "model_year_from": min(years) if years else program.get("model_year_from"),
        "model_year_to": max(years) if years else program.get("model_year_to"),
        "drive_system": identity.drive_system,
        "propulsion_architecture": identity.propulsion_architecture,
        "transmission_type": identity.transmission_type,
        "transmission_model": identity.transmission_model,
        "gear_count": identity.gear_count,
        "category": identity.category,
        "missing_discriminants": list(result.missing_discriminants),
    }


def build_work_queues(
    identities: list[dict[str, Any]],
    result_map: dict[str, dict[str, PriorMatchResult]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    buckets: dict[str, dict[str, dict[str, Any]]] = {
        "NO_MATCH": {},
        "CATEGORY_RULE_MATCH": {},
    }
    for row in identities:
        identity: TechnicalIdentity = row["identity"]
        for result in result_map[identity.vehicle_configuration_id].values():
            if result.match_level not in buckets:
                continue
            payload = _technical_key_payload(row, result)
            canonical = _json_dump(payload)
            key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20].upper()
            target = buckets[result.match_level].get(key)
            if target is None:
                year_from = payload["model_year_from"]
                year_to = payload["model_year_to"]
                year_range = (
                    str(year_from)
                    if year_from == year_to
                    else f"{year_from or ''}-{year_to or ''}"
                )
                target = {
                    "technical_identity_key": key,
                    "vehicle_configuration_id": identity.vehicle_configuration_id,
                    "domain": result.domain,
                    "make": payload["make"],
                    "model": payload["model"],
                    "model_year_range": year_range,
                    "drive_system": identity.drive_system,
                    "propulsion_architecture": identity.propulsion_architecture,
                    "transmission_type": identity.transmission_type,
                    "transmission_model": identity.transmission_model,
                    "gear_count": identity.gear_count,
                    "category": identity.category,
                    "missing_discriminants_json": _json_dump(list(result.missing_discriminants)),
                    "affected_vde_count": 0,
                }
                buckets[result.match_level][key] = target
            target["affected_vde_count"] += len(row["vdes"])
    no_match = [buckets["NO_MATCH"][key] for key in sorted(buckets["NO_MATCH"])]
    category = [
        buckets["CATEGORY_RULE_MATCH"][key]
        for key in sorted(buckets["CATEGORY_RULE_MATCH"])
    ]
    return no_match, category


def _coverage_summary(
    identities: list[dict[str, Any]],
    vdes: list[dict[str, Any]],
    result_map: dict[str, dict[str, PriorMatchResult]],
    *,
    reference_count: int,
    resolution_count: int,
    profile: list[dict[str, Any]],
    no_match_queue: list[dict[str, Any]],
    category_queue: list[dict[str, Any]],
) -> dict[str, Any]:
    by_domain: dict[str, Any] = {}
    for domain in MATCH_DOMAINS:
        config_counts = Counter(
            results[domain].match_level for results in result_map.values()
        )
        vde_counts: Counter[str] = Counter()
        for row in identities:
            identity: TechnicalIdentity = row["identity"]
            vde_counts[result_map[identity.vehicle_configuration_id][domain].match_level] += len(row["vdes"])
        by_domain[domain] = {
            "vehicle_configurations": {level: config_counts[level] for level in MATCH_LEVELS},
            "vdes": {level: vde_counts[level] for level in MATCH_LEVELS},
        }

    overall = Counter()
    cohorts: dict[str, dict[str, Counter[str]]] = {
        "drive_system": defaultdict(Counter),
        "propulsion_architecture": defaultdict(Counter),
        "transmission_type": defaultdict(Counter),
    }
    for row in identities:
        identity: TechnicalIdentity = row["identity"]
        decisions = result_map[identity.vehicle_configuration_id]
        matched = sum(decisions[domain].match_level != "NO_MATCH" for domain in MATCH_DOMAINS)
        bucket = "all_domains_matched" if matched == len(MATCH_DOMAINS) else "partially_matched" if matched else "no_domains_matched"
        count = len(row["vdes"])
        overall[bucket] += count
        for field in cohorts:
            value = getattr(identity, field) or "(missing)"
            cohorts[field][str(value)][bucket] += count

    cohort_output: dict[str, Any] = {}
    for field, values in cohorts.items():
        cohort_output[field] = {
            value: {
                "vdes": sum(counts.values()),
                "all_domains_matched": counts["all_domains_matched"],
                "partially_matched": counts["partially_matched"],
                "no_domains_matched": counts["no_domains_matched"],
            }
            for value, counts in sorted(values.items())
        }

    missing_counts = Counter()
    for results in result_map.values():
        for result in results.values():
            missing_counts.update(result.missing_discriminants)
    return {
        "population": {
            "vdes": len(vdes),
            "vehicle_configurations": len(identities),
            "synthetic_components": reference_count,
            "synthetic_resolutions": resolution_count,
        },
        "by_domain": by_domain,
        "overall": {
            "all_requested_domains_matched": overall["all_domains_matched"],
            "partially_matched": overall["partially_matched"],
            "no_domains_matched": overall["no_domains_matched"],
        },
        "cohort_coverage": cohort_output,
        "missing_discriminants": dict(sorted(missing_counts.items())),
        "discriminant_profile_fields": len(profile),
        "work_queue": {
            "category_unique_identities": len(category_queue),
            "no_match_unique_identities": len(no_match_queue),
            "unmatched_vde_domain_rows": sum(
                data["vdes"]["NO_MATCH"] for data in by_domain.values()
            ),
        },
    }


def _csv_text(rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> str:
    if fields is None:
        fields = list(rows[0]) if rows else []
    field_list = list(fields)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=field_list, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _reference_summary(inventory: list[dict[str, Any]]) -> str:
    counts = Counter(str(row["boundary"]) for row in inventory)
    lines = [
        "# Synthetic reference catalog summary",
        "",
        f"References discovered from the canonical DB: **{len(inventory)}**.",
        "",
        "| Boundary | References |",
        "|---|---:|",
    ]
    lines.extend(f"| {boundary} | {counts[boundary]} |" for boundary in sorted(counts))
    lines.extend(
        [
            "",
            "The matching taxonomy is read from `custom_properties_json`: "
            "`application_class`, `drive_architecture`, and (where applicable) `position`.",
            "Reference ABC values are inventoried but never used by the matcher.",
            "",
        ]
    )
    return "\n".join(lines)


def _coverage_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Rule-only component prior coverage",
        "",
        "This is a zero-LLM, zero-search, read-only coverage baseline. It selects "
        "reference priors only; it does not decompose or alter VDE road load.",
        "",
        "## Population",
        "",
        f"- VDEs: **{summary['population']['vdes']:,}**",
        f"- Vehicle configurations: **{summary['population']['vehicle_configurations']:,}**",
        f"- Synthetic references/resolutions: **{summary['population']['synthetic_components']} / {summary['population']['synthetic_resolutions']}**",
        "",
        "## Coverage by domain (VDE projection)",
        "",
        "| Domain | Exact | Strong | Category | No match |",
        "|---|---:|---:|---:|---:|",
    ]
    for domain in MATCH_DOMAINS:
        counts = summary["by_domain"][domain]["vdes"]
        lines.append(
            f"| {domain} | {counts['EXACT_RULE_MATCH']:,} | {counts['STRONG_RULE_MATCH']:,} | "
            f"{counts['CATEGORY_RULE_MATCH']:,} | {counts['NO_MATCH']:,} |"
        )
    overall = summary["overall"]
    lines.extend(
        [
            "",
            "## Overall VDE coverage",
            "",
            f"- All requested domains matched: **{overall['all_requested_domains_matched']:,}**",
            f"- Partially matched: **{overall['partially_matched']:,}**",
            f"- No domains matched: **{overall['no_domains_matched']:,}**",
            "",
            "## Conservative finding",
            "",
            "The synthetic catalog has an explicit application-class taxonomy, but the current "
            "canonical configuration layer does not. Broad stored labels such as `Car`, `Truck`, "
            "and `Both` are not silently translated into synthetic classes. AXLE and HUB_BEARING "
            "also require an explicit component position. These cases therefore remain NO_MATCH.",
            "",
            "## Future enrichment queue",
            "",
            f"- Unmatched VDE × domain rows: **{summary['work_queue']['unmatched_vde_domain_rows']:,}**",
            f"- Category-only unique identities: **{summary['work_queue']['category_unique_identities']:,}**",
            f"- No-match unique identities: **{summary['work_queue']['no_match_unique_identities']:,}**",
            "",
            "## Most common missing discriminants",
            "",
            "| Discriminant | Configuration × domain decisions |",
            "|---|---:|",
            *(
                f"| {field} | {count:,} |"
                for field, count in sorted(
                    summary["missing_discriminants"].items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "",
            "The next search-agent pilot should target the deduplicated no-match identities and "
            "retrieve an explicit application class plus front/rear position where required. "
            "No web enrichment was performed by this run.",
            "",
        ]
    )
    return "\n".join(lines)


def build_artifacts(
    *,
    inventory: list[dict[str, Any]],
    config_rows: list[dict[str, Any]],
    vde_rows: list[dict[str, Any]],
    no_match_queue: list[dict[str, Any]],
    category_queue: list[dict[str, Any]],
    profile: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, bytes]:
    work_fields = (
        "technical_identity_key", "vehicle_configuration_id", "domain", "make", "model",
        "model_year_range", "drive_system", "propulsion_architecture", "transmission_type",
        "transmission_model", "gear_count", "category", "missing_discriminants_json",
        "affected_vde_count",
    )
    texts = {
        "reference_catalog.csv": _csv_text(inventory),
        "reference_catalog.json": json.dumps(inventory, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        "reference_catalog_summary.md": _reference_summary(inventory),
        "canonical_discriminant_profile.csv": _csv_text(profile),
        "config_prior_matches.csv": _csv_text(config_rows),
        "vde_prior_matches.csv": _csv_text(vde_rows),
        "no_match_work_queue.csv": _csv_text(no_match_queue, work_fields),
        "category_match_work_queue.csv": _csv_text(category_queue, work_fields),
        "coverage_summary.json": json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        "coverage_report.md": _coverage_report(summary),
    }
    return {name: text.encode("utf-8") for name, text in texts.items()}


def run(db_path: Path, output_dir: Path) -> dict[str, Any]:
    db_path = db_path.resolve(strict=True)
    hash_before = sha256_file(db_path)
    connection = _open_read_only(db_path)
    try:
        _require_schema(connection)
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        foreign_key_issues = len(connection.execute("PRAGMA foreign_key_check").fetchall())
        if quick_check != "ok" or foreign_key_issues:
            raise ValueError(
                f"Source DB integrity failed: quick_check={quick_check}, fk_issues={foreign_key_issues}"
            )
        objects_before = _objects(connection)
        catalog, inventory = load_reference_catalog(connection)
        identities, vdes, raw = load_population(connection)
        profile = profile_discriminants(raw)
        config_rows, vde_rows, result_map = evaluate_population(identities, catalog)
        no_match_queue, category_queue = build_work_queues(identities, result_map)
        objects_after = _objects(connection)
    finally:
        connection.close()

    hash_after = sha256_file(db_path)
    if hash_before != hash_after:
        raise RuntimeError("Canonical DB hash changed during a read-only matching run")
    if objects_before != objects_after:
        raise RuntimeError("SQLite object inventory changed during a read-only matching run")

    summary = _coverage_summary(
        identities,
        vdes,
        result_map,
        reference_count=len(catalog),
        resolution_count=len(inventory),
        profile=profile,
        no_match_queue=no_match_queue,
        category_queue=category_queue,
    )
    summary["run"] = {
        "db_path": str(db_path),
        "db_sha256_before": hash_before,
        "db_sha256_after": hash_after,
        "db_modified": False,
        "quick_check": quick_check,
        "foreign_key_check_issue_count": foreign_key_issues,
        "sqlite_object_count": len(objects_before),
        "sqlite_objects_unchanged": True,
        "external_search_used": False,
        "llm_per_vde": False,
    }
    first = build_artifacts(
        inventory=inventory,
        config_rows=config_rows,
        vde_rows=vde_rows,
        no_match_queue=no_match_queue,
        category_queue=category_queue,
        profile=profile,
        summary=summary,
    )
    second = build_artifacts(
        inventory=inventory,
        config_rows=config_rows,
        vde_rows=vde_rows,
        no_match_queue=no_match_queue,
        category_queue=category_queue,
        profile=profile,
        summary=summary,
    )
    if first != second:
        raise RuntimeError("Repeated deterministic artifact build produced different bytes")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in first.items():
        (output_dir / name).write_bytes(payload)
    artifact_hashes = {
        name: hashlib.sha256(payload).hexdigest().upper()
        for name, payload in sorted(first.items())
    }
    return {
        "status": "PASS",
        "output_dir": str(output_dir.resolve()),
        "db_sha256_before": hash_before,
        "db_sha256_after": hash_after,
        "db_modified": False,
        "quick_check": quick_check,
        "foreign_key_check_issue_count": foreign_key_issues,
        "reference_count": len(catalog),
        "configuration_match_rows": len(config_rows),
        "vde_match_rows": len(vde_rows),
        "no_match_work_queue_rows": len(no_match_queue),
        "category_work_queue_rows": len(category_queue),
        "deterministic_artifacts": True,
        "artifact_sha256": artifact_hashes,
        "coverage": summary["by_domain"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("artifacts/rule_only"), type=Path)
    args = parser.parse_args()
    try:
        report = run(args.db, args.output_dir)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
