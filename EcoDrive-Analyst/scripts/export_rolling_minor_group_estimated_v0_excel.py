from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from itertools import chain
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from typing import Iterable, Iterator, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export_db_review_excel import _XlsxWriter, _quote_identifier, open_database_read_only


METHOD = "GROUP_ESTIMATED_ROLLING_MINOR_V0"
EXPECTED_SOURCE_SHA256 = "8AFAC44888388452E9EBF85F8162B2BFA233DFCC74AC3C924339F235A0EA0330"
EXPECTED_TABLES = {
    "rolling_minor_group_prior_v0": 9,
    "rolling_minor_group_estimate_v0": 7_429,
}


VDE_REVIEW_SQL = """
WITH brake_support AS (
  SELECT l.vde_id,COUNT(*) AS link_count
  FROM vde_component_resolution l
  JOIN component_resolution cr USING(component_resolution_id)
  WHERE cr.boundary='BRAKE' AND l.adoption_role='SUPPORTING'
  GROUP BY l.vde_id
), hub_support AS (
  SELECT l.vde_id,COUNT(*) AS link_count
  FROM vde_component_resolution l
  JOIN component_resolution cr USING(component_resolution_id)
  WHERE cr.boundary='HUB_BEARING' AND l.adoption_role='SUPPORTING'
  GROUP BY l.vde_id
), axle_support AS (
  SELECT l.vde_id,COUNT(*) AS link_count
  FROM vde_component_resolution l
  JOIN component_resolution cr USING(component_resolution_id)
  WHERE cr.boundary='AXLE' AND l.adoption_role='SUPPORTING'
  GROUP BY l.vde_id
)
SELECT e.vde_id,v.year,v.make,v.model,v.category,v.drive_type,e.drive_group,
       e.rolling_minor_resolution_id,
       cr.resolved_A_N AS rolling_minor_A_N,
       cr.resolved_B_N_per_kph AS rolling_minor_B_N_per_kph,
       cr.resolved_C_N_per_kph2 AS rolling_minor_C_N_per_kph2,
       cr.resolved_A_N + cr.resolved_B_N_per_kph*80.4672 + cr.resolved_C_N_per_kph2*80.4672*80.4672 AS rolling_minor_force_50_N,
       e.brake_scope,e.brake_reference_family_n,e.brake_confidence,e.brake_share_50,
       e.brake_A_N,e.brake_B_N_per_kph,e.brake_C_N_per_kph2,
       e.hub_scope,e.hub_reference_family_n,e.hub_confidence,e.hub_share_50,
       e.hub_A_N,e.hub_B_N_per_kph,e.hub_C_N_per_kph2,
       e.tire_other_residual_share_50,e.tire_other_residual_A_N,
       e.tire_other_residual_B_N_per_kph,e.tire_other_residual_C_N_per_kph2,
       e.residual_min_force_10_70mph_N,e.closure_max_abs_error_10_70mph_N,
       CASE WHEN COALESCE(bs.link_count,0)>0 THEN 1 ELSE 0 END AS has_brake_supporting_reference,
       CASE WHEN COALESCE(hs.link_count,0)>0 THEN 1 ELSE 0 END AS has_hub_supporting_reference,
       COALESCE(ax.link_count,0) AS axle_supporting_link_count,
       e.method,e.metric_basis
FROM rolling_minor_group_estimate_v0 e
JOIN vde v ON v.id=e.vde_id
JOIN component_resolution cr ON cr.component_resolution_id=e.rolling_minor_resolution_id
LEFT JOIN brake_support bs ON bs.vde_id=e.vde_id
LEFT JOIN hub_support hs ON hs.vde_id=e.vde_id
LEFT JOIN axle_support ax ON ax.vde_id=e.vde_id
ORDER BY e.vde_id
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _query_rows(connection: sqlite3.Connection, sql: str) -> Iterator[Sequence[object]]:
    cursor = connection.execute(sql)
    yield tuple(item[0] for item in cursor.description or ())
    yield from cursor


def _table_rows(connection: sqlite3.Connection, table: str, order_by: str) -> Iterator[Sequence[object]]:
    cursor = connection.execute(f'SELECT * FROM {_quote_identifier(table)} ORDER BY {order_by}')
    yield tuple(item[0] for item in cursor.description or ())
    yield from cursor


def _db_inventory(connection: sqlite3.Connection) -> Iterator[Sequence[object]]:
    yield ("object_name", "object_type", "row_count", "column_count", "primary_key_columns", "notes")
    objects = connection.execute(
        "SELECT name,type FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY type,name"
    ).fetchall()
    for obj in objects:
        name, kind = str(obj[0]), str(obj[1])
        columns = list(connection.execute(f"PRAGMA table_info({_quote_identifier(name)})"))
        count = int(connection.execute(f"SELECT COUNT(*) FROM {_quote_identifier(name)}").fetchone()[0])
        pk = ",".join(str(row[1]) for row in sorted(columns, key=lambda row: row[5] or 9999) if row[5])
        note = "Group-estimated v0 materialization" if name.startswith("rolling_minor_group_") else "Original Sprint 12 object; unchanged"
        yield (name, kind, count, len(columns), pk, note)


def _schema_rows(connection: sqlite3.Connection) -> Iterator[Sequence[object]]:
    yield ("object", "object_type", "ordinal", "column", "sqlite_type", "not_null", "default_value", "primary_key_order")
    objects = connection.execute(
        "SELECT name,type FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY type,name"
    ).fetchall()
    for obj in objects:
        for column in connection.execute(f"PRAGMA table_info({_quote_identifier(str(obj[0]))})"):
            yield (obj[0], obj[1], int(column[0]) + 1, column[1], column[2], int(column[3]), column[4], int(column[5]))


def _coverage_rows(connection: sqlite3.Connection) -> list[tuple[object, ...]]:
    total = int(connection.execute("SELECT COUNT(*) FROM vde").fetchone()[0])
    estimates = int(connection.execute("SELECT COUNT(*) FROM rolling_minor_group_estimate_v0").fetchone()[0])
    def supporting(boundary: str) -> int:
        return int(connection.execute(
            "SELECT COUNT(DISTINCT l.vde_id) FROM vde_component_resolution l "
            "JOIN component_resolution cr USING(component_resolution_id) "
            "WHERE cr.boundary=? AND l.adoption_role='SUPPORTING'", (boundary,)
        ).fetchone()[0])
    values = (
        ("TOTAL_VDE", total, "Authoritative VDE population"),
        ("ROLLING_MINOR_ADOPTED", estimates, "Authoritative macro base for v0 estimates"),
        ("BRAKE_GROUP_ESTIMATED_ABC", estimates, "Proxy ABC materialized from Force_50 share"),
        ("HUB_BEARING_GROUP_ESTIMATED_ABC", estimates, "Proxy ABC materialized from Force_50 share"),
        ("TIRE_OTHER_MINOR_RESIDUAL_ABC", estimates, "Exact algebraic residual; not measured tire"),
        ("AXLE_SUPPORTING_REFERENCE_VDE", supporting("AXLE"), "Reference only; boundary-uncertain"),
        ("TRANSMISSION_SUPPORTING_REFERENCE_VDE", supporting("TRANSMISSION"), "Reference only; boundary-uncertain"),
        ("MACRO_UNRESOLVED", total - estimates, "Intentionally left without fine estimates"),
    )
    return [("metric", "vde_count", "pct_total_vde", "interpretation"), *[(name, count, count / total, note) for name, count, note in values]]


def _share_summary(connection: sqlite3.Connection) -> list[tuple[object, ...]]:
    rows = {
        (str(row[0]), str(row[1])): float(row[2])
        for row in connection.execute("SELECT component,drive_group,p50_share FROM rolling_minor_group_prior_v0")
    }
    output: list[tuple[object, ...]] = [("drive_group", "brake_p50_share", "hub_p50_share", "tire_other_residual_p50_share", "hub_prior_scope")]
    for drive in ("FWD", "RWD", "AWD", "4WD", "UNKNOWN"):
        brake_key = ("BRAKE", drive if ("BRAKE", drive) in rows else "GLOBAL")
        hub_key = ("HUB_BEARING", drive if ("HUB_BEARING", drive) in rows else "GLOBAL")
        brake, hub = rows[brake_key], rows[hub_key]
        output.append((drive, brake, hub, 1.0 - brake - hub, "DRIVE_TYPE" if hub_key[1] != "GLOBAL" else "GLOBAL_FALLBACK"))
    return output


def _checks(connection: sqlite3.Connection, qa: dict[str, object], csv_dir: Path) -> list[tuple[object, ...]]:
    def csv_count(name: str) -> int:
        with (csv_dir / name).open(encoding="utf-8-sig", newline="") as handle:
            return max(sum(1 for _ in csv.reader(handle)) - 1, 0)
    estimate_count = int(connection.execute("SELECT COUNT(*) FROM rolling_minor_group_estimate_v0").fetchone()[0])
    prior_count = int(connection.execute("SELECT COUNT(*) FROM rolling_minor_group_prior_v0").fetchone()[0])
    checks = [
        ("SQLite quick_check", connection.execute("PRAGMA quick_check").fetchone()[0], "ok"),
        ("Foreign-key issue count", len(connection.execute("PRAGMA foreign_key_check").fetchall()), 0),
        ("Original table count diffs", len(qa.get("original_table_count_diffs", {})), 0),
        ("Estimate DB rows", estimate_count, EXPECTED_TABLES["rolling_minor_group_estimate_v0"]),
        ("Estimate CSV rows", csv_count("ROLLING_MINOR_GROUP_ESTIMATED_V0.csv"), estimate_count),
        ("Prior DB rows", prior_count, EXPECTED_TABLES["rolling_minor_group_prior_v0"]),
        ("Prior CSV rows", csv_count("ROLLING_MINOR_GROUP_PRIORS_V0.csv"), prior_count),
        ("Non-positive residual cases", int(connection.execute("SELECT COUNT(*) FROM rolling_minor_group_estimate_v0 WHERE residual_min_force_10_70mph_N<=0").fetchone()[0]), 0),
        ("Bad method rows", int(connection.execute("SELECT COUNT(*) FROM rolling_minor_group_estimate_v0 WHERE method<>?", (METHOD,)).fetchone()[0]), 0),
        ("Fine semantic label", "GROUP_ESTIMATED_PROXY", "GROUP_ESTIMATED_PROXY"),
        ("Tire residual semantic label", "TIRE_OTHER_MINOR_RESIDUAL", "TIRE_OTHER_MINOR_RESIDUAL"),
    ]
    return [("check", "actual", "expected", "status"), *[(name, actual, expected, "PASS" if actual == expected else "FAIL") for name, actual, expected in checks]]


def export(db_path: Path, package_dir: Path, output: Path, manifest_path: Path) -> dict[str, object]:
    db_path = db_path.resolve(strict=True)
    package_dir = package_dir.resolve(strict=True)
    output = output.resolve()
    qa = json.loads((package_dir / "FINAL_QA_V0.json").read_text(encoding="utf-8"))
    source_hash_before = sha256(db_path)
    connection = open_database_read_only(db_path)
    try:
        quick = connection.execute("PRAGMA quick_check").fetchone()[0]
        fk_issues = len(connection.execute("PRAGMA foreign_key_check").fetchall())
        counts = {table: int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]) for table in EXPECTED_TABLES}
        if quick != "ok" or fk_issues != 0 or counts != EXPECTED_TABLES:
            raise RuntimeError(f"DB gate failed: quick={quick}, fk={fk_issues}, counts={counts}")
        summary = [
            ("item", "value", "interpretation"),
            ("Closure status", "CLOSED", "Group-estimated v0 artifact status"),
            ("Method", METHOD, "Frozen deterministic rule"),
            ("Metric basis", "FORCE_50_MPH", "Group prior selection basis"),
            ("Output DB SHA256", source_hash_before, "Workbook source database"),
            ("Original Sprint 12 DB SHA256", str(qa["source_db_sha256"]).upper(), "Accepted immutable source"),
            ("Total VDE", int(qa["source_total_vde"]), "Canonical population"),
            ("Estimate rows", int(qa["estimate_rows"]), "One group-estimated row per resolved RollingMinor VDE"),
            ("Macro unresolved", int(qa["source_total_vde"]) - int(qa["estimate_rows"]), "Intentionally untouched"),
            ("Prior rows", int(qa["prior_rows"]), "Drive-type plus global fallbacks"),
            ("Brake global fallback VDE", int(qa["brake_global_fallback_vdes"]), "Unknown drive only"),
            ("Hub global fallback VDE", int(qa["hub_global_fallback_vdes"]), "RWD plus unknown drive"),
            ("Max closure error N", float(qa["max_closure_abs_error_N_10_70mph"]), "10-70 mph numerical closure"),
            ("Non-positive residual cases", int(qa["negative_or_zero_residual_cases_10_70mph"]), "Must remain zero"),
            ("SQLite quick_check", quick, "Integrity gate"),
            ("Foreign-key issues", fk_issues, "Integrity gate"),
        ]
        readme = [
            ("item", "value"),
            ("Workbook", "EcoDrive RollingMinor Group Estimated v0 - Final Review"),
            ("Generated UTC", datetime.now(timezone.utc).replace(microsecond=0).isoformat()),
            ("Canonical source", str(db_path)),
            ("Source SHA256", source_hash_before),
            ("Purpose", "Human review of the non-invasive Group Estimated RollingMinor v0 materialization."),
            ("Authority", "SQLite remains authoritative; this XLSX is a review/export artifact."),
            ("Brake/Hub semantics", "Group-estimated proxy ABCs, not independently identified hardware curves."),
            ("Tire semantics", "TIRE_OTHER_MINOR_RESIDUAL is an exact residual, not measured tire."),
            ("Axle/Transmission", "Supporting/reference-only; not used as additive decomposition."),
            ("NULL representation", "SQLite NULL exports as a blank cell."),
            ("Scope", "All 7,429 estimate rows are included; no sampling or truncation."),
        ]
        with tempfile.TemporaryDirectory(prefix="ecodrive_rm_group_xlsx_") as temp:
            writer = _XlsxWriter(Path(temp))
            writer.add_sheet(position=1, name="00_README", rows=readme, filter_header_row=None, column_styles={2: 3})
            writer.add_sheet(position=2, name="01_CLOSURE_SUMMARY", rows=summary, filter_header_row=1, column_styles={3: 3})
            writer.add_sheet(position=3, name="02_DB_INVENTORY", rows=_db_inventory(connection), column_styles={6: 3})
            writer.add_sheet(position=4, name="03_SCHEMA", rows=_schema_rows(connection))
            writer.add_sheet(position=5, name="04_SHARE_SUMMARY", rows=_share_summary(connection), column_styles={2: 2, 3: 2, 4: 2})
            writer.add_sheet(position=6, name="05_COVERAGE", rows=_coverage_rows(connection), column_styles={3: 2, 4: 3})
            writer.add_sheet(position=7, name="06_PRIORS", rows=_table_rows(connection, "rolling_minor_group_prior_v0", "component,drive_group"), column_styles={5: 2, 6: 2, 7: 2})
            writer.add_sheet(position=8, name="07_ESTIMATES", rows=_table_rows(connection, "rolling_minor_group_estimate_v0", "vde_id"), column_styles={7: 2, 14: 2, 18: 2})
            writer.add_sheet(position=9, name="08_VDE_REVIEW", rows=_query_rows(connection, VDE_REVIEW_SQL), column_styles={16: 2, 23: 2, 27: 2})
            writer.add_sheet(position=10, name="09_CHECKS", rows=_checks(connection, qa, package_dir))
            sheet_names = writer.save(output)
    finally:
        connection.close()
    source_hash_after = sha256(db_path)
    if source_hash_before != source_hash_after:
        raise RuntimeError("Read-only Excel export changed the source DB")
    manifest = {
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "workbook": str(output),
        "workbook_sha256": sha256(output),
        "source_db": str(db_path),
        "source_db_sha256_before": source_hash_before,
        "source_db_sha256_after": source_hash_after,
        "source_db_unchanged": True,
        "sheets": list(sheet_names),
        "estimate_rows": counts["rolling_minor_group_estimate_v0"],
        "prior_rows": counts["rolling_minor_group_prior_v0"],
        "vde_review_rows": counts["rolling_minor_group_estimate_v0"],
        "truncated": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.db, args.package_dir, args.output, args.manifest), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
