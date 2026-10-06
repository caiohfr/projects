from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
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
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.export_db_review_excel import _XlsxWriter, _quote_identifier
from vde_core.sprint12_final_closure import (
    EXPECTED_FINAL_SHA256,
    audit_database,
    component_coverage_rows,
    file_sha256,
    open_read_only,
)


TABLE_SHEETS = {
    "program": "PROGRAM",
    "vehicle_configuration": "VEHICLE_CONFIG",
    "vde": "VDE",
    "fuelcons": "FUELCONS",
    "run": "RUN",
    "component_db": "COMPONENT_DB",
    "component_resolution": "COMP_RESOLUTION",
    "vde_component_resolution": "VDE_COMP_LINK",
    "component_instance": "COMP_INSTANCE",
    "fuelcons_run_adoption": "FUEL_RUN_ADOPT",
    "tire_db": "TIRE_DB",
    "vde_request_history": "VDE_REQ_HISTORY",
    "vde_request_history_proposals": "VDE_REQ_PROPOSALS",
}


VDE_FLAT_SQL = """
WITH ranked_fuelcons AS (
  SELECT f.*,
         ROW_NUMBER() OVER (
           PARTITION BY f.vde_id
           ORDER BY CASE WHEN f.record_status='ACTIVE' THEN 0 ELSE 1 END, f.id
         ) AS rn
  FROM fuelcons f
)
SELECT
  v.id AS vde_id,
  v.vehicle_configuration_id,
  vc.program_id,
  p.commercial_make,
  p.commercial_model,
  v.year AS model_year,
  v.legislation,
  v.cycle_name,
  v.cycle_source,
  v.category,
  v.make AS source_make,
  v.model AS source_model,
  v.coast_A_N,
  v.coast_B_N_per_kph,
  v.coast_C_N_per_kph2,
  v.test_mass_kg,
  v.mass_kg,
  vc.drive_system,
  vc.transmission_type,
  vc.transmission_model,
  vc.gear_count,
  vc.final_drive_ratio,
  vc.propulsion_architecture,
  rf.electrification,
  rf.fuel_type,
  rf.fuel_l_per_100km,
  rf.energy_Wh_per_km,
  rf.gco2_per_km,
  rf.id AS representative_fuelcons_id,
  v.aero_C_coef_Npkph2,
  v.tire_A_final,
  v.tire_B_final,
  v.tire_C_final,
  v.trans_A_coef_N,
  v.trans_B_coef_Npkph,
  v.trans_C_coef_Npkph2,
  v.brake_A_coef_N,
  v.brake_B_coef_Npkph,
  v.brake_C_coef_Npkph2,
  v.vde_total_mj_per_km,
  v.vde_net_mj_per_km,
  v.vde_urb_mj_per_km,
  v.vde_hw_mj_per_km,
  v.vde_high_mj_per_km,
  v.record_origin,
  v.record_status,
  v.review_status,
  v.source_name,
  v.source_record_id,
  vc.architecture_properties_json,
  vc.source_identity_json AS vehicle_configuration_source_identity_json,
  v.provenance_json AS vde_provenance_json
FROM vde v
LEFT JOIN vehicle_configuration vc
  ON vc.vehicle_configuration_id=v.vehicle_configuration_id
LEFT JOIN program p ON p.program_id=vc.program_id
LEFT JOIN ranked_fuelcons rf ON rf.vde_id=v.id AND rf.rn=1
ORDER BY v.id
"""


def _query_rows(connection: sqlite3.Connection, sql: str, parameters: Sequence[object] = ()) -> Iterator[Sequence[object]]:
    cursor = connection.execute(sql, parameters)
    headers = tuple(description[0] for description in cursor.description or ())
    yield headers
    yield from cursor


def _table_rows(connection: sqlite3.Connection, table: str, pk_columns: Sequence[str]) -> Iterator[Sequence[object]]:
    info = list(connection.execute(f"PRAGMA table_info({_quote_identifier(table)})"))
    headers = tuple(str(row[1]) for row in info)
    yield headers
    order = ",".join(_quote_identifier(column) for column in pk_columns)
    sql = f"SELECT * FROM {_quote_identifier(table)}"
    if order:
        sql += f" ORDER BY {order}"
    yield from connection.execute(sql)


def _text_column_styles(connection: sqlite3.Connection, table: str) -> dict[int, int]:
    styles: dict[int, int] = {}
    for index, column in enumerate(connection.execute(f"PRAGMA table_info({_quote_identifier(table)})"), start=1):
        name = str(column[1]).lower()
        if name.endswith("_json") or any(token in name for token in ("notes", "description", "payload", "provenance", "assumptions", "conditions", "details")):
            styles[index] = 3
    return styles


def _readme_rows(audit: dict[str, object]) -> Iterable[Sequence[object]]:
    return (
        ("Item", "Value"),
        ("Workbook", "EcoDrive Canonical DB - Sprint 12 Final"),
        ("Purpose", "Read-only engineering review export of the immutable Sprint 12 canonical release."),
        ("Canonical source", audit["db_path"]),
        ("Canonical authority", "SQLite remains the canonical artifact; this workbook is a read-only review mirror."),
        ("Source SHA256", audit["sha256"]),
        ("Expected release SHA256", EXPECTED_FINAL_SHA256),
        ("Generated UTC", datetime.now(timezone.utc).replace(microsecond=0).isoformat()),
        ("SQLite quick_check", ", ".join(audit["quick_check"])),
        ("SQLite foreign_key_check issues", audit["foreign_key_issue_count"]),
        ("Null representation", "Blank cell means SQLite NULL; strings and numeric zero retain their original meaning."),
        ("Raw data", "All physical tables are exported completely and without value normalization."),
        ("VDE_FLAT", "One row per VDE; representative FuelCons is selected deterministically for review only."),
        ("COMP_COVERAGE", "One row per VDE. Fine links are supporting evidence unless explicitly marked adopted."),
        ("Scientific boundary", "Macro coverage is not fine hardware identification; unresolved states remain explicit."),
    )


def _summary_rows(connection: sqlite3.Connection, audit: dict[str, object]) -> Iterable[Sequence[object]]:
    yield ("object_name", "object_type", "row_count", "column_count", "primary_key_columns", "worksheet", "notes")
    for item in audit["tables"]:
        table = str(item["table"])
        yield (
            table,
            "table",
            item["row_count"],
            item["column_count"],
            ", ".join(item["primary_key_columns"]),
            TABLE_SHEETS.get(table, table[:31].upper()),
            "Complete raw export",
        )
    for obj in audit["objects"]:
        if obj["type"] != "view":
            continue
        name = str(obj["name"])
        columns = list(connection.execute(f"PRAGMA table_info({_quote_identifier(name)})"))
        count = int(connection.execute(f"SELECT COUNT(*) FROM {_quote_identifier(name)}").fetchone()[0])
        yield (name, "view", count, len(columns), "", "", "Schema object documented; physical tables are the raw export authority")
    coverage = component_coverage_rows(connection)
    status_counts: dict[str, int] = {}
    for row in coverage:
        status = str(row["macro_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    headline = (
        ("TOTAL_VDE", len(coverage)),
        ("MACRO_RESOLVED", len(coverage) - status_counts.get("UNRESOLVED", 0)),
        ("MACRO_SUPPORTED", status_counts.get("SUPPORTED", 0)),
        ("MACRO_CONDITIONAL", status_counts.get("CONDITIONAL", 0)),
        ("MACRO_UNRESOLVED", status_counts.get("UNRESOLVED", 0)),
        ("MACRO_PROJECTED", sum(int(row["projected_to_historical_slots"]) for row in coverage)),
        ("FINE_SUPPORTING_LINKS", sum(int(row["fine_supporting_links"]) for row in coverage)),
        ("FINE_ADOPTED_LINKS", sum(int(row["fine_adopted_links"]) for row in coverage)),
    )
    for name, value in headline:
        yield (name, "component_population_kpi", value, "", "", "COMP_COVERAGE", "Recomputed from final release DB")


def _dictionary_rows(connection: sqlite3.Connection, objects: Sequence[dict[str, object]]) -> Iterable[Sequence[object]]:
    yield ("object", "object_type", "column_order", "column_name", "sqlite_type", "not_null", "default_value", "primary_key_order")
    for obj in sorted(objects, key=lambda item: (str(item["type"]), str(item["name"]))):
        for column in connection.execute(f"PRAGMA table_info({_quote_identifier(str(obj['name']))})"):
            yield (obj["name"], obj["type"], int(column[0]) + 1, column[1], column[2], int(column[3]), column[4], int(column[5]))


def _foreign_key_rows(connection: sqlite3.Connection, tables: Sequence[dict[str, object]]) -> Iterable[Sequence[object]]:
    yield ("child_table", "fk_id", "sequence", "child_column", "parent_table", "parent_column", "on_update", "on_delete", "match")
    for item in sorted(tables, key=lambda row: str(row["table"])):
        table = str(item["table"])
        for fk in connection.execute(f"PRAGMA foreign_key_list({_quote_identifier(table)})"):
            yield (table, fk[0], fk[1], fk[3], fk[2], fk[4], fk[5], fk[6], fk[7])


def export_workbook(db_path: Path, output_path: Path, manifest_path: Path) -> dict[str, object]:
    db_path = Path(db_path).resolve(strict=True)
    output_path = Path(output_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    audit = audit_database(db_path)
    if audit["sha256"] != EXPECTED_FINAL_SHA256:
        raise RuntimeError(f"Unexpected final DB SHA256: {audit['sha256']}")
    if audit["quick_check"] != ["ok"] or audit["foreign_key_issue_count"] != 0:
        raise RuntimeError("Final DB integrity gate failed")

    before = file_sha256(db_path)
    connection = open_read_only(db_path)
    raw_manifest: list[dict[str, object]] = []
    try:
        table_by_name = {str(item["table"]): item for item in audit["tables"]}
        missing = sorted(set(TABLE_SHEETS) - set(table_by_name))
        if missing:
            raise RuntimeError(f"Expected final tables are absent: {missing}")
        with tempfile.TemporaryDirectory(prefix="ecodrive_sprint12_excel_") as temp:
            writer = _XlsxWriter(Path(temp))
            position = 1
            writer.add_sheet(position=position, name="00_README", rows=_readme_rows(audit), filter_header_row=None)
            position += 1
            writer.add_sheet(position=position, name="01_DB_SUMMARY", rows=_summary_rows(connection, audit))
            position += 1
            writer.add_sheet(position=position, name="02_DATA_DICTIONARY", rows=_dictionary_rows(connection, audit["objects"]))
            position += 1
            writer.add_sheet(position=position, name="03_FOREIGN_KEYS", rows=_foreign_key_rows(connection, audit["tables"]))
            position += 1

            for table, sheet in TABLE_SHEETS.items():
                item = table_by_name[table]
                writer.add_sheet(
                    position=position,
                    name=sheet,
                    rows=_table_rows(connection, table, item["primary_key_columns"]),
                    column_styles=_text_column_styles(connection, table),
                )
                raw_manifest.append({
                    "table": table,
                    "sheet": sheet,
                    "row_count": item["row_count"],
                    "column_count": item["column_count"],
                })
                position += 1

            vde_flat_count = int(connection.execute(f"SELECT COUNT(*) FROM ({VDE_FLAT_SQL.rstrip().rstrip(';')})").fetchone()[0])
            writer.add_sheet(position=position, name="VDE_FLAT", rows=_query_rows(connection, VDE_FLAT_SQL))
            position += 1
            coverage = component_coverage_rows(connection)
            coverage_headers = tuple(coverage[0]) if coverage else ("vde_id",)
            coverage_data = (tuple(row.get(header) for header in coverage_headers) for row in coverage)
            writer.add_sheet(position=position, name="COMP_COVERAGE", rows=chain((coverage_headers,), coverage_data))
            sheet_names = writer.save(output_path)
    finally:
        connection.close()

    after = file_sha256(db_path)
    if before != after:
        raise RuntimeError("Workbook export modified the read-only source DB")
    manifest: dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_db": str(db_path),
        "source_db_sha256_before": before,
        "source_db_sha256_after": after,
        "workbook": str(output_path),
        "workbook_sha256": file_sha256(output_path),
        "sheets": list(sheet_names),
        "sheet_count": len(sheet_names),
        "raw_tables": raw_manifest,
        "derived_sheets": {
            "VDE_FLAT": {"row_count": vde_flat_count},
            "COMP_COVERAGE": {"row_count": len(coverage)},
        },
        "truncated": False,
        "source_db_unchanged": before == after,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the immutable Sprint 12 canonical DB to a complete review workbook.")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = export_workbook(args.db, args.output, args.manifest)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
