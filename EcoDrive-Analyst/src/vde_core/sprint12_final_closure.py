from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPECTED_FINAL_SHA256 = "8AFAC44888388452E9EBF85F8162B2BFA233DFCC74AC3C924339F235A0EA0330"
MACRO_ESTIMATOR_VERSION = "ECODRIVE_COMPONENT_ESTIMATOR_V1.0"
FINE_BOUNDARIES = {"TIRE", "BRAKE", "HUB_BEARING", "HUB", "TRANSMISSION", "AXLE", "AXLE_HUBS"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def open_read_only(path: Path) -> sqlite3.Connection:
    resolved = Path(path).resolve(strict=True)
    connection = sqlite3.connect(f"file:{resolved.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def schema_sha256(connection: sqlite3.Connection) -> str:
    rows = connection.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
    ).fetchall()
    payload = json.dumps([list(row) for row in rows], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def audit_database(path: Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    before = file_sha256(path)
    connection = open_read_only(path)
    try:
        quick = [row[0] for row in connection.execute("PRAGMA quick_check")]
        fk_issues = [list(row) for row in connection.execute("PRAGMA foreign_key_check")]
        objects = [
            dict(row)
            for row in connection.execute(
                "SELECT name,type FROM sqlite_master WHERE type IN ('table','view') "
                "AND name NOT LIKE 'sqlite_%' ORDER BY type,name"
            )
        ]
        tables: list[dict[str, Any]] = []
        for obj in objects:
            if obj["type"] != "table":
                continue
            name = str(obj["name"])
            info = list(connection.execute(f"PRAGMA table_info({_quote(name)})"))
            pk = [str(row[1]) for row in sorted(info, key=lambda row: row[5] or 9999) if row[5]]
            tables.append({
                "table": name,
                "row_count": int(connection.execute(f"SELECT COUNT(*) FROM {_quote(name)}").fetchone()[0]),
                "column_count": len(info),
                "primary_key_columns": pk,
            })
        audit = {
            "db_path": str(path), "sha256": before, "size_bytes": path.stat().st_size,
            "quick_check": quick, "foreign_key_issue_count": len(fk_issues),
            "foreign_key_issues": fk_issues, "schema_sha256": schema_sha256(connection),
            "objects": objects, "tables": tables,
        }
    finally:
        connection.close()
    after = file_sha256(path)
    if after != before:
        raise RuntimeError("Read-only final DB audit changed the database hash")
    audit["sha256_after_audit"] = after
    return audit


def component_coverage_rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        WITH macro AS (
          SELECT l.vde_id,
                 COUNT(*) AS macro_link_count,
                 SUM(CASE WHEN r.estimate_status='SUPPORTED' THEN 1 ELSE 0 END) AS supported_count,
                 SUM(CASE WHEN r.estimate_status='CONDITIONAL' THEN 1 ELSE 0 END) AS conditional_count,
                 GROUP_CONCAT(DISTINCT r.estimator_version) AS estimator_versions,
                 GROUP_CONCAT(DISTINCT r.method) AS methods,
                 GROUP_CONCAT(DISTINCT json_extract(r.provenance_json,'$.architecture_class')) AS architecture_routes,
                 MAX(CASE WHEN l.boundary='AERO' THEN r.component_resolution_id END) AS aero_resolution_id,
                 MAX(CASE WHEN l.boundary='ROLLING_MINOR' THEN r.component_resolution_id END) AS rolling_minor_resolution_id,
                 MAX(CASE WHEN l.boundary='DRIVETRAIN_AGGREGATE' THEN r.component_resolution_id END) AS drivetrain_resolution_id,
                 MAX(CASE WHEN l.boundary='EDRIVE_AGGREGATE' THEN r.component_resolution_id END) AS edrive_resolution_id
          FROM vde_component_resolution l
          JOIN component_resolution r ON r.component_resolution_id=l.component_resolution_id
          WHERE l.adoption_role='ADOPTED' AND r.estimator_version=?
          GROUP BY l.vde_id
        ), fine AS (
          SELECT l.vde_id,
                 SUM(CASE WHEN l.adoption_role='SUPPORTING' THEN 1 ELSE 0 END) AS supporting_links,
                 SUM(CASE WHEN l.adoption_role='ADOPTED' AND l.boundary IN ('TIRE','BRAKE','HUB_BEARING','HUB','TRANSMISSION','AXLE','AXLE_HUBS') THEN 1 ELSE 0 END) AS fine_adopted_links,
                 SUM(CASE WHEN l.adoption_role='SUPPORTING' AND l.boundary='TIRE' THEN 1 ELSE 0 END) AS tire_supporting,
                 SUM(CASE WHEN l.adoption_role='SUPPORTING' AND l.boundary='BRAKE' THEN 1 ELSE 0 END) AS brake_supporting,
                 SUM(CASE WHEN l.adoption_role='SUPPORTING' AND l.boundary IN ('HUB_BEARING','HUB') THEN 1 ELSE 0 END) AS hub_supporting,
                 SUM(CASE WHEN l.adoption_role='SUPPORTING' AND l.boundary='TRANSMISSION' THEN 1 ELSE 0 END) AS transmission_supporting,
                 SUM(CASE WHEN l.adoption_role='SUPPORTING' AND l.boundary IN ('AXLE','AXLE_HUBS') THEN 1 ELSE 0 END) AS axle_supporting,
                 GROUP_CONCAT(DISTINCT CASE WHEN l.adoption_role='SUPPORTING' THEN l.boundary END) AS fine_supporting_domains,
                 GROUP_CONCAT(DISTINCT CASE WHEN l.adoption_role='SUPPORTING' THEN r.method END) AS fine_methods
          FROM vde_component_resolution l
          JOIN component_resolution r ON r.component_resolution_id=l.component_resolution_id
          GROUP BY l.vde_id
        )
        SELECT v.id AS vde_id,v.vehicle_configuration_id,v.year AS model_year,v.make,v.model,
               COALESCE(m.architecture_routes,
                        json_extract(v.provenance_json,'$.component_population_projection.architecture_class')) AS architecture_route,
               CASE WHEN m.vde_id IS NULL THEN 'UNRESOLVED'
                    WHEN m.supported_count>0 AND m.conditional_count=0 THEN 'SUPPORTED'
                    WHEN m.conditional_count>0 AND m.supported_count=0 THEN 'CONDITIONAL'
                    ELSE 'MIXED' END AS macro_status,
               m.estimator_versions AS macro_estimator_version,m.methods AS macro_method,
               m.aero_resolution_id,m.rolling_minor_resolution_id,
               m.drivetrain_resolution_id,m.edrive_resolution_id,
               CASE WHEN json_extract(v.provenance_json,'$.component_population_projection.decomposition_mode')='MACRO_DECOMPOSITION' THEN 1 ELSE 0 END AS projected_to_historical_slots,
               json_extract(v.provenance_json,'$.component_population_projection.tire_slot_semantics') AS tire_slot_semantics,
               json_extract(v.provenance_json,'$.component_population_projection.transmission_slot_semantics') AS transmission_slot_semantics,
               json_extract(v.provenance_json,'$.component_population_projection.aero_slot_semantics') AS aero_slot_semantics,
               COALESCE(f.supporting_links,0) AS fine_supporting_links,
               COALESCE(f.tire_supporting,0) AS tire_supporting,
               COALESCE(f.brake_supporting,0) AS brake_supporting,
               COALESCE(f.hub_supporting,0) AS hub_supporting,
               COALESCE(f.transmission_supporting,0) AS transmission_supporting,
               COALESCE(f.axle_supporting,0) AS axle_supporting,
               COALESCE(f.fine_adopted_links,0) AS fine_adopted_links,
               f.fine_supporting_domains,f.fine_methods,
               CASE WHEN COALESCE(f.transmission_supporting,0)>0 OR COALESCE(f.axle_supporting,0)>0 THEN 1 ELSE 0 END AS boundary_unknown,
               CASE WHEN m.vde_id IS NULL THEN 'NO_ADOPTED_MACRO_RESOLUTION' END AS unresolved_reason,
               json_extract(v.provenance_json,'$.component_population_projection.reason_codes') AS projection_reason_codes,
               json_extract(v.provenance_json,'$.component_population_projection') AS projection_provenance_json
        FROM vde v
        LEFT JOIN macro m ON m.vde_id=v.id
        LEFT JOIN fine f ON f.vde_id=v.id
        ORDER BY v.id
        """,
        (MACRO_ESTIMATOR_VERSION,),
    ).fetchall()
    return [dict(row) for row in rows]


def coverage_summary(
    connection: sqlite3.Connection,
    rows: list[dict[str, Any]],
    *,
    research_agent_summary: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    total = len(rows)
    status = Counter(str(row["macro_status"]) for row in rows)
    metrics: list[dict[str, Any]] = []

    def add(metric: str, value: int, source: str, note: str, *, is_vde_count: bool = True) -> None:
        metrics.append({
            "metric": metric, "value": int(value),
            "pct_total_vde": round(100.0 * int(value) / total, 6) if total and is_vde_count else None,
            "source": source, "interpretation": note,
        })

    add("TOTAL_VDE", total, "final DB", "Authoritative VDE population")
    add("MACRO_RESOLVED", total - status["UNRESOLVED"], "final DB links", "At least one complete adopted macro solution")
    add("MACRO_SUPPORTED", status["SUPPORTED"], "final DB links", "Existing estimator status SUPPORTED")
    add("MACRO_CONDITIONAL", status["CONDITIONAL"], "final DB links", "Existing estimator status CONDITIONAL")
    add("MACRO_UNRESOLVED", status["UNRESOLVED"], "final DB links", "No adopted macro solution; intentionally unresolved")
    add("MACRO_PROJECTED_TO_HISTORICAL_SLOTS", sum(int(row["projected_to_historical_slots"]) for row in rows), "vde.provenance_json", "Macro decomposition projected with explicit slot semantics")
    add("EDRIVE_AGGREGATE_RETAINED_WITHOUT_TRANSMISSION_PROJECTION", sum(bool(row["edrive_resolution_id"]) and not int(row["projected_to_historical_slots"]) for row in rows), "final DB links + VDE provenance", "EDrive aggregate retained outside historical Transmission slot")
    fine_supporting_links = int(connection.execute("SELECT COUNT(*) FROM vde_component_resolution WHERE adoption_role='SUPPORTING'").fetchone()[0])
    fine_adopted_links = int(connection.execute(
        "SELECT COUNT(*) FROM vde_component_resolution WHERE adoption_role='ADOPTED' "
        "AND boundary IN ('TIRE','BRAKE','HUB_BEARING','HUB','TRANSMISSION','AXLE','AXLE_HUBS')"
    ).fetchone()[0])
    add("FINE_SUPPORTING_LINKS", fine_supporting_links, "final DB links", "Supporting/reference evidence; not adopted physical decomposition", is_vde_count=False)
    add("FINE_ADOPTED_LINKS", fine_adopted_links, "final DB links", "Independently adopted fine links", is_vde_count=False)
    for domain, field in (
        ("TIRE", "tire_supporting"), ("BRAKE", "brake_supporting"),
        ("HUB_BEARING", "hub_supporting"), ("TRANSMISSION", "transmission_supporting"),
        ("AXLE", "axle_supporting"),
    ):
        add(f"VDE_WITH_{domain}_SUPPORTING_EVIDENCE", sum(int(row[field]) > 0 for row in rows), "final DB links", f"VDEs with {domain} supporting evidence")
    add("BOUNDARY_UNKNOWN_REPRESENTABLE", sum(int(row["boundary_unknown"]) for row in rows), "final DB supporting links", "Transmission/Axle supporting evidence remains boundary-uncertain")
    deferred = int((research_agent_summary or {}).get("deferred_no_current_model_value", 0))
    add("DEFERRED_RESEARCH_NO_CURRENT_MODEL_VALUE_GOLDEN_TASKS", deferred, "Research Agent freeze manifest", "Golden-task count; not a fleet/VDE materialization", is_vde_count=False)

    architecture = Counter((str(row["architecture_route"] or "UNRESOLVED"), str(row["macro_status"])) for row in rows)
    by_architecture = [
        {"architecture_route": key[0], "macro_status": key[1], "vde_count": value}
        for key, value in sorted(architecture.items())
    ]
    by_year = Counter(
        ((str(row["model_year"]) if row["model_year"] is not None else "UNKNOWN"), str(row["macro_status"]))
        for row in rows
    )
    year_rows = [
        {"model_year": key[0], "macro_status": key[1], "vde_count": value}
        for key, value in sorted(by_year.items())
    ]
    domain_rows = []
    for domain, field in (
        ("TIRE", "tire_supporting"), ("BRAKE", "brake_supporting"),
        ("HUB_BEARING", "hub_supporting"), ("TRANSMISSION", "transmission_supporting"),
        ("AXLE", "axle_supporting"),
    ):
        domain_rows.append({
            "component_domain": domain,
            "supporting_link_count": sum(int(row[field]) for row in rows),
            "unique_vde_count": sum(int(row[field]) > 0 for row in rows),
        })
    method_counts = Counter()
    for row in rows:
        method_counts[(str(row["macro_method"] or ""), str(row["macro_estimator_version"] or ""), str(row["macro_status"]))] += 1
    method_rows = [
        {"method": key[0], "estimator_version": key[1], "macro_status": key[2], "vde_count": value}
        for key, value in sorted(method_counts.items())
    ]
    return metrics, {
        "architecture": by_architecture, "model_year": year_rows,
        "component_domain": domain_rows, "method": method_rows,
    }


def write_coverage_reports(
    output_dir: Path,
    metrics: list[dict[str, Any]],
    breakdowns: dict[str, list[dict[str, Any]]],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "COMPONENT_POPULATION_FINAL_COVERAGE.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value", "pct_total_vde", "source", "interpretation"])
        writer.writeheader()
        writer.writerows(metrics)
    lines = [
        "# Component Population — Final Sprint 12 Coverage", "",
        "## Headline metrics", "",
        "| Metric | Value | % total VDE | Source | Interpretation |",
        "|---|---:|---:|---|---|",
    ]
    for row in metrics:
        pct = "n/a" if row["pct_total_vde"] is None else f"{row['pct_total_vde']:.3f}%"
        lines.append(f"| {row['metric']} | {row['value']:,} | {pct} | {row['source']} | {row['interpretation']} |")
    for name, rows in breakdowns.items():
        lines.extend(["", f"## Breakdown — {name}", ""])
        if not rows:
            lines.append("No rows.")
            continue
        fields = list(rows[0])
        lines.append("| " + " | ".join(fields) + " |")
        lines.append("|" + "|".join("---:" if field.endswith(("count", "year")) else "---" for field in fields) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
    lines.extend([
        "", "## Frozen interpretation", "",
        "- Macro coverage is not fine hardware identification.",
        "- Fine supporting/reference evidence is not an adopted physical decomposition.",
        "- Vector search was not validated for vehicle-specific fine identity.",
        "- Research Agent v0 is experimental, partial, and frozen for owner review.",
        "- Unresolved, boundary-unknown, and deferred states are intentional scientific outputs; no gaps were silently filled.",
    ])
    (output_dir / "COMPONENT_POPULATION_FINAL_COVERAGE.md").write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "EXPECTED_FINAL_SHA256", "MACRO_ESTIMATOR_VERSION", "audit_database",
    "component_coverage_rows", "coverage_summary", "file_sha256", "open_read_only",
    "schema_sha256", "write_coverage_reports",
]
