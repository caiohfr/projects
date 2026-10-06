"""Final pre-PROD normalization of canonical VDE and FuelCons surrogate IDs.

This stage copies the latest canonical staging database, deterministically
renumbers only ``vde.id`` and ``fuelcons.id`` to contiguous positive ranges,
and relies on the declared ``ON UPDATE CASCADE`` foreign keys to update every
physical reference.  Stable TEXT identities and engineering values are never
regenerated or rewritten.

The operation is intentionally staging-only.  It must not be used after PROD
promotion: existing production IDs are immutable and new rows must instead be
allocated above the current maximum.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DB = (
    ROOT
    / "etl"
    / "data"
    / "staging"
    / "sprint_12f13_vde_materialized"
    / "eco_drive_canonical_vde_materialized_candidate.db"
)
DEFAULT_OUTPUT_DB = ROOT / "data" / "db" / "staging" / "eco_drive_canonical_candidate.db"
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "id_remap"
PROTECTED_DATABASES = {
    (ROOT / "data" / "db" / "eco_drive.db").resolve(),
    (ROOT / "data" / "db" / "eco_drive_qa.db").resolve(),
}
TEMP_ID_BASE = -8_000_000_000_000_000_000


@dataclass(frozen=True)
class IdNormalizationReport:
    source_database: str
    output_database: str
    source_sha256_before: str
    source_sha256_after: str
    output_sha256: str
    schema_sha256_before: str
    schema_sha256_after: str
    table_counts_before: dict[str, int]
    table_counts_after: dict[str, int]
    vde_old_min: int
    vde_old_max: int
    vde_new_min: int
    vde_new_max: int
    vde_count: int
    fuelcons_old_min: int
    fuelcons_old_max: int
    fuelcons_new_min: int
    fuelcons_new_max: int
    fuelcons_count: int
    vde_reference_columns: tuple[str, ...]
    fuelcons_reference_columns: tuple[str, ...]
    vde_topology_sha256_before: str
    vde_topology_sha256_after: str
    run_topology_sha256_before: str
    run_topology_sha256_after: str
    fuelcons_topology_sha256_before: str
    fuelcons_topology_sha256_after: str
    adoption_topology_sha256_before: str
    adoption_topology_sha256_after: str
    engineering_payload_sha256_before: str
    engineering_payload_sha256_after: str
    negative_identifier_counts: dict[str, int]
    quick_check: str
    foreign_key_issues: int
    orphan_counts: dict[str, int]
    vde_remap_csv: str
    fuelcons_remap_csv: str
    report_json: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _json_digest(rows: Iterable[Sequence[Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(tuple(row), ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest().upper()


def _schema_sha256(connection: sqlite3.Connection) -> str:
    return _json_digest(
        connection.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
        )
    )


def _table_counts(connection: sqlite3.Connection) -> dict[str, int]:
    tables = [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    return {
        table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        for table in tables
    }


def _guard_paths(source: Path, output: Path) -> tuple[Path, Path]:
    source = source.resolve(strict=True)
    output = output.resolve()
    if source == output:
        raise ValueError("Source and output databases must be different files.")
    if output in PROTECTED_DATABASES:
        raise ValueError(f"Runtime/QA database is protected: {output}")
    if output.suffix.lower() != ".db":
        raise ValueError(f"Output must be a SQLite .db file: {output}")
    if any(part.casefold() in {"prod", "production"} for part in output.parts):
        raise ValueError("ID normalization is pre-PROD only; a PROD path is forbidden.")
    return source, output


def _stable_text(value: Any) -> str:
    if value is None:
        return "<NULL>"
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value)


def _vde_mapping(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    connection.row_factory = sqlite3.Row
    rows = [dict(row) for row in connection.execute(
        "SELECT id,source_name,source_file_version,source_record_id,normalization_version,"
        "vehicle_configuration_id,make,model,year,legislation,category,cycle_name,"
        "coast_A_N,coast_B_N_per_kph,coast_C_N_per_kph2,test_mass_kg "
        "FROM vde"
    )]
    stable_fields = (
        "source_name", "source_file_version", "source_record_id", "normalization_version",
        "vehicle_configuration_id", "make", "model", "year", "legislation", "category",
        "cycle_name", "coast_A_N", "coast_B_N_per_kph", "coast_C_N_per_kph2", "test_mass_kg",
    )
    rows.sort(key=lambda row: tuple(_stable_text(row[field]) for field in stable_fields) + (int(row["id"]),))
    return [
        {
            "old_id": int(row["id"]),
            "new_id": new_id,
            "source_name": row["source_name"],
            "source_file_version": row["source_file_version"],
            "source_record_id": row["source_record_id"],
            "vehicle_configuration_id": row["vehicle_configuration_id"],
            "make": row["make"],
            "model": row["model"],
            "year": row["year"],
        }
        for new_id, row in enumerate(rows, start=1)
    ]


def _fuelcons_mapping(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    connection.row_factory = sqlite3.Row
    rows = [dict(row) for row in connection.execute(
        "SELECT f.id,f.vde_id,f.source_name,f.source_file_version,f.source_record_id,"
        "f.normalization_version,f.record_origin,f.comparison_basis,f.electrification,f.fuel_type,"
        "v.source_name AS vde_source_name,v.source_record_id AS vde_source_record_id,"
        "v.vehicle_configuration_id,v.make,v.model,v.year "
        "FROM fuelcons f JOIN vde v ON v.id=f.vde_id"
    )]
    stable_fields = (
        "source_name", "source_file_version", "source_record_id", "normalization_version",
        "record_origin", "comparison_basis", "electrification", "fuel_type",
        "vde_source_name", "vde_source_record_id", "vehicle_configuration_id", "year",
    )
    rows.sort(key=lambda row: tuple(_stable_text(row[field]) for field in stable_fields) + (int(row["id"]),))
    return [
        {
            "old_id": int(row["id"]),
            "new_id": new_id,
            "source_name": row["source_name"],
            "source_file_version": row["source_file_version"],
            "source_record_id": row["source_record_id"],
            "record_origin": row["record_origin"],
            "comparison_basis": row["comparison_basis"],
            "vde_id": row["vde_id"],
            "vde_source_record_id": row["vde_source_record_id"],
            "make": row["make"],
            "model": row["model"],
            "year": row["year"],
        }
        for new_id, row in enumerate(rows, start=1)
    ]


def _foreign_key_references(
    connection: sqlite3.Connection, parent_table: str, parent_column: str
) -> tuple[str, ...]:
    references: set[str] = set()
    tables = [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    for table in tables:
        for row in connection.execute(f'PRAGMA foreign_key_list("{table}")'):
            if row[2] == parent_table and row[4] == parent_column:
                if str(row[5]).upper() != "CASCADE":
                    raise RuntimeError(
                        f"Cannot safely remap {parent_table}.{parent_column}: "
                        f"{table}.{row[3]} uses ON UPDATE {row[5]}"
                    )
                references.add(f"{table}.{row[3]}")
    return tuple(sorted(references))


def _remap_primary_key(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    mapping: Mapping[int, int],
) -> None:
    if not mapping:
        return
    temp_ids = {old_id: TEMP_ID_BASE - new_id for old_id, new_id in mapping.items()}
    existing = {
        int(row[0])
        for row in connection.execute(f'SELECT "{column}" FROM "{table}"')
    }
    collisions = existing & set(temp_ids.values())
    if collisions:
        raise RuntimeError(f"Temporary ID collision in {table}: {sorted(collisions)[:3]}")
    connection.executemany(
        f'UPDATE "{table}" SET "{column}"=? WHERE "{column}"=?',
        ((temp_ids[old_id], old_id) for old_id in mapping),
    )
    connection.executemany(
        f'UPDATE "{table}" SET "{column}"=? WHERE "{column}"=?',
        ((new_id, temp_ids[old_id]) for old_id, new_id in mapping.items()),
    )


def _remap_json_key(value: str | None, key: str, mapping: Mapping[int, int]) -> str | None:
    if value is None:
        return None
    payload = json.loads(value)
    changed = False

    def visit(item: Any) -> None:
        nonlocal changed
        if isinstance(item, dict):
            for item_key, item_value in item.items():
                if item_key == key and isinstance(item_value, int) and item_value in mapping:
                    item[item_key] = mapping[item_value]
                    changed = True
                else:
                    visit(item_value)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(payload)
    if not changed:
        return value
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _update_json_references(
    connection: sqlite3.Connection,
    table: str,
    primary_key: str,
    json_column: str,
    reference_key: str,
    mapping: Mapping[int, int],
) -> int:
    updates: list[tuple[str, Any]] = []
    for row_id, value in connection.execute(
        f'SELECT "{primary_key}","{json_column}" FROM "{table}" '
        f'WHERE "{json_column}" IS NOT NULL AND "{json_column}" LIKE ?',
        (f'%"{reference_key}"%',),
    ):
        remapped = _remap_json_key(value, reference_key, mapping)
        if remapped != value:
            updates.append((remapped, row_id))
    connection.executemany(
        f'UPDATE "{table}" SET "{json_column}"=? WHERE "{primary_key}"=?', updates
    )
    return len(updates)


def _update_vde_notes(
    connection: sqlite3.Connection,
    note_updates: Sequence[tuple[int, int, int]],
) -> int:
    updates: list[tuple[str, int]] = []
    for child_new_id, parent_old_id, parent_new_id in note_updates:
        row = connection.execute("SELECT notes FROM vde WHERE id=?", (child_new_id,)).fetchone()
        if row is None or row[0] is None:
            continue
        notes = str(row[0])
        remapped = re.sub(
            rf"(?<!\d){re.escape(str(parent_old_id))}(?!\d)", str(parent_new_id), notes
        )
        if remapped != notes:
            updates.append((remapped, child_new_id))
    connection.executemany("UPDATE vde SET notes=? WHERE id=?", updates)
    return len(updates)


def _write_mapping(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["old_id", "new_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _identity_maps(connection: sqlite3.Connection) -> tuple[dict[int, str], dict[int, str]]:
    vde = {
        int(row[0]): "|".join(_stable_text(value) for value in row[1:])
        for row in connection.execute(
            "SELECT id,source_name,source_file_version,source_record_id,vehicle_configuration_id,"
            "make,model,year,legislation,category FROM vde"
        )
    }
    fuelcons = {
        int(row[0]): "|".join(_stable_text(value) for value in row[1:])
        for row in connection.execute(
            "SELECT id,source_name,source_file_version,source_record_id,record_origin,"
            "comparison_basis,electrification,fuel_type FROM fuelcons"
        )
    }
    return vde, fuelcons


def _topology_signatures(connection: sqlite3.Connection) -> dict[str, str]:
    vde_identity, fuelcons_identity = _identity_maps(connection)
    vde_rows = sorted(
        (vde_identity[int(child)], None if parent is None else vde_identity[int(parent)])
        for child, parent in connection.execute("SELECT id,vde_id_parent FROM vde")
    )
    run_rows = sorted(
        (
            row[0],
            vde_identity[int(row[1])],
            json.loads(row[2]).get("carryover_from_run_id") if row[2] else None,
        )
        for row in connection.execute("SELECT run_id,vde_id,provenance_json FROM run")
    )
    fuelcons_rows = []
    for row in connection.execute(
        "SELECT id,vde_id,reference_fuelcons_id,provenance_json FROM fuelcons"
    ):
        provenance_parent = (
            json.loads(row[3]).get("carryover_from_fuelcons_id") if row[3] else None
        )
        fuelcons_rows.append(
            (
                fuelcons_identity[int(row[0])],
                vde_identity[int(row[1])],
                None if row[2] is None else fuelcons_identity[int(row[2])],
                None
                if provenance_parent is None
                else fuelcons_identity[int(provenance_parent)],
            )
        )
    fuelcons_rows.sort()
    adoption_rows = sorted(
        (
            fuelcons_identity[int(row[0])],
            row[1],
            vde_identity[int(row[2])],
            row[3],
            row[4],
        )
        for row in connection.execute(
            "SELECT fuelcons_id,run_id,vde_id,result_dimension,ordinal "
            "FROM fuelcons_run_adoption"
        )
    )
    return {
        "vde": _json_digest(vde_rows),
        "run": _json_digest(run_rows),
        "fuelcons": _json_digest(fuelcons_rows),
        "adoption": _json_digest(adoption_rows),
    }


def _engineering_payload_signature(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    for table, excluded in (
        ("vde", {"id", "vde_id_parent", "notes", "provenance_json"}),
        ("run", {"vde_id", "provenance_json"}),
        ("fuelcons", {"id", "vde_id", "reference_fuelcons_id", "provenance_json"}),
    ):
        columns = [
            row[1]
            for row in connection.execute(f'PRAGMA table_info("{table}")')
            if row[1] not in excluded
        ]
        order_column = {"vde": "source_record_id", "run": "run_id", "fuelcons": "source_record_id"}[table]
        query = (
            "SELECT " + ",".join(f'"{column}"' for column in columns)
            + f' FROM "{table}" ORDER BY "{order_column}"'
        )
        digest.update(table.encode("ascii"))
        digest.update(_json_digest(connection.execute(query)).encode("ascii"))
    return digest.hexdigest().upper()


def _negative_identifier_counts(connection: sqlite3.Connection) -> dict[str, int]:
    result: dict[str, int] = {}
    tables = [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    for table in tables:
        for column in connection.execute(f'PRAGMA table_info("{table}")'):
            name = str(column[1])
            declared = str(column[2] or "").upper()
            if "INT" not in declared or not (name == "id" or name.endswith("_id")):
                continue
            count = connection.execute(
                f'SELECT COUNT(*) FROM "{table}" WHERE "{name}" < 0'
            ).fetchone()[0]
            result[f"{table}.{name}"] = count
    return dict(sorted(result.items()))


def _orphan_counts(connection: sqlite3.Connection) -> dict[str, int]:
    queries = {
        "vde.vde_id_parent": "SELECT COUNT(*) FROM vde c LEFT JOIN vde p ON p.id=c.vde_id_parent WHERE c.vde_id_parent IS NOT NULL AND p.id IS NULL",
        "run.vde_id": "SELECT COUNT(*) FROM run c LEFT JOIN vde p ON p.id=c.vde_id WHERE p.id IS NULL",
        "fuelcons.vde_id": "SELECT COUNT(*) FROM fuelcons c LEFT JOIN vde p ON p.id=c.vde_id WHERE p.id IS NULL",
        "fuelcons.reference_fuelcons_id": "SELECT COUNT(*) FROM fuelcons c LEFT JOIN fuelcons p ON p.id=c.reference_fuelcons_id WHERE c.reference_fuelcons_id IS NOT NULL AND p.id IS NULL",
        "fuelcons_run_adoption.vde_id": "SELECT COUNT(*) FROM fuelcons_run_adoption c LEFT JOIN vde p ON p.id=c.vde_id WHERE p.id IS NULL",
        "fuelcons_run_adoption.fuelcons_id": "SELECT COUNT(*) FROM fuelcons_run_adoption c LEFT JOIN fuelcons p ON p.id=c.fuelcons_id WHERE p.id IS NULL",
        "vde_component_resolution.vde_id": "SELECT COUNT(*) FROM vde_component_resolution c LEFT JOIN vde p ON p.id=c.vde_id WHERE p.id IS NULL",
    }
    return {name: connection.execute(sql).fetchone()[0] for name, sql in queries.items()}


def _range(connection: sqlite3.Connection, table: str) -> tuple[int, int, int]:
    count, minimum, maximum = connection.execute(
        f'SELECT COUNT(*),MIN(id),MAX(id) FROM "{table}"'
    ).fetchone()
    return int(count), int(minimum), int(maximum)


def normalize_candidate_database(
    source_db: Path | str = DEFAULT_SOURCE_DB,
    output_db: Path | str = DEFAULT_OUTPUT_DB,
    artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR,
    *,
    rebuild: bool = False,
) -> IdNormalizationReport:
    source, output = _guard_paths(Path(source_db), Path(output_db))
    artifacts = Path(artifact_dir).resolve()
    if output.exists() and not rebuild:
        raise FileExistsError(f"Output exists; pass rebuild=True/--rebuild: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    source_hash_before = sha256_file(source)
    temporary_output = output.with_name(output.name + ".id_normalization.tmp")
    temporary_output.unlink(missing_ok=True)
    for suffix in ("-journal", "-wal", "-shm"):
        Path(str(temporary_output) + suffix).unlink(missing_ok=True)
    shutil.copy2(source, temporary_output)

    try:
        connection = sqlite3.connect(temporary_output)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            schema_before = _schema_sha256(connection)
            counts_before = _table_counts(connection)
            vde_count, vde_old_min, vde_old_max = _range(connection, "vde")
            fuel_count, fuel_old_min, fuel_old_max = _range(connection, "fuelcons")
            topology_before = _topology_signatures(connection)
            engineering_before = _engineering_payload_signature(connection)
            vde_rows = _vde_mapping(connection)
            vde_map = {row["old_id"]: row["new_id"] for row in vde_rows}
            vde_references = _foreign_key_references(connection, "vde", "id")
            note_updates = [
                (vde_map[int(child)], int(parent), vde_map[int(parent)])
                for child, parent in connection.execute(
                    "SELECT id,vde_id_parent FROM vde WHERE vde_id_parent IS NOT NULL"
                )
            ]

            connection.execute("BEGIN IMMEDIATE")
            connection.execute("PRAGMA defer_foreign_keys=ON")
            # These two child-key indexes are transient migration accelerators.
            # They are removed before commit, leaving the physical schema byte-for-byte
            # equivalent at the sqlite_master SQL level.
            connection.execute(
                "CREATE INDEX __id_norm_fuelcons_reference "
                "ON fuelcons(reference_fuelcons_id)"
            )
            connection.execute(
                "CREATE INDEX __id_norm_adoption_vde "
                "ON fuelcons_run_adoption(vde_id)"
            )
            _remap_primary_key(connection, "vde", "id", vde_map)
            _update_json_references(
                connection, "vde", "id", "provenance_json", "carryover_from_vde_id", vde_map
            )
            _update_vde_notes(connection, note_updates)

            fuel_rows = _fuelcons_mapping(connection)
            fuel_map = {row["old_id"]: row["new_id"] for row in fuel_rows}
            fuel_references = _foreign_key_references(connection, "fuelcons", "id")
            _remap_primary_key(connection, "fuelcons", "id", fuel_map)
            _update_json_references(
                connection,
                "fuelcons",
                "id",
                "provenance_json",
                "carryover_from_fuelcons_id",
                fuel_map,
            )
            connection.execute("DROP INDEX __id_norm_adoption_vde")
            connection.execute("DROP INDEX __id_norm_fuelcons_reference")
            connection.commit()

            schema_after = _schema_sha256(connection)
            counts_after = _table_counts(connection)
            vde_count_after, vde_new_min, vde_new_max = _range(connection, "vde")
            fuel_count_after, fuel_new_min, fuel_new_max = _range(connection, "fuelcons")
            topology_after = _topology_signatures(connection)
            engineering_after = _engineering_payload_signature(connection)
            quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
            foreign_key_issues = len(connection.execute("PRAGMA foreign_key_check").fetchall())
            negative_counts = _negative_identifier_counts(connection)
            orphans = _orphan_counts(connection)

            invariants = {
                "schema unchanged": schema_before == schema_after,
                "row counts unchanged": counts_before == counts_after,
                "VDE count unchanged": vde_count == vde_count_after,
                "FuelCons count unchanged": fuel_count == fuel_count_after,
                "VDE contiguous 1..N": (vde_new_min, vde_new_max) == (1, vde_count_after),
                "FuelCons contiguous 1..N": (fuel_new_min, fuel_new_max) == (1, fuel_count_after),
                "VDE topology unchanged": topology_before["vde"] == topology_after["vde"],
                "Run topology unchanged": topology_before["run"] == topology_after["run"],
                "FuelCons topology unchanged": topology_before["fuelcons"] == topology_after["fuelcons"],
                "adoption topology unchanged": topology_before["adoption"] == topology_after["adoption"],
                "engineering payload unchanged": engineering_before == engineering_after,
                "no negative integer identifiers": not any(negative_counts.values()),
                "no orphans": not any(orphans.values()),
                "quick_check ok": quick_check == "ok",
                "foreign_key_check clean": foreign_key_issues == 0,
            }
            failures = [name for name, passed in invariants.items() if not passed]
            if failures:
                raise RuntimeError("ID normalization validation failed: " + "; ".join(failures))
        finally:
            connection.close()

        if output.exists():
            output.unlink()
        os.replace(temporary_output, output)
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        for suffix in ("-journal", "-wal", "-shm"):
            Path(str(temporary_output) + suffix).unlink(missing_ok=True)
        raise

    source_hash_after = sha256_file(source)
    if source_hash_before != source_hash_after:
        output.unlink(missing_ok=True)
        raise RuntimeError("Source database changed during normalization; output removed.")

    vde_csv = artifacts / "VDE_ID_REMAP.csv"
    fuel_csv = artifacts / "FUELCONS_ID_REMAP.csv"
    report_json = artifacts / "sprint_12_final_id_normalization_summary.json"
    _write_mapping(vde_csv, vde_rows)
    _write_mapping(fuel_csv, fuel_rows)

    report = IdNormalizationReport(
        source_database=str(source),
        output_database=str(output),
        source_sha256_before=source_hash_before,
        source_sha256_after=source_hash_after,
        output_sha256=sha256_file(output),
        schema_sha256_before=schema_before,
        schema_sha256_after=schema_after,
        table_counts_before=counts_before,
        table_counts_after=counts_after,
        vde_old_min=vde_old_min,
        vde_old_max=vde_old_max,
        vde_new_min=vde_new_min,
        vde_new_max=vde_new_max,
        vde_count=vde_count_after,
        fuelcons_old_min=fuel_old_min,
        fuelcons_old_max=fuel_old_max,
        fuelcons_new_min=fuel_new_min,
        fuelcons_new_max=fuel_new_max,
        fuelcons_count=fuel_count_after,
        vde_reference_columns=vde_references,
        fuelcons_reference_columns=fuel_references,
        vde_topology_sha256_before=topology_before["vde"],
        vde_topology_sha256_after=topology_after["vde"],
        run_topology_sha256_before=topology_before["run"],
        run_topology_sha256_after=topology_after["run"],
        fuelcons_topology_sha256_before=topology_before["fuelcons"],
        fuelcons_topology_sha256_after=topology_after["fuelcons"],
        adoption_topology_sha256_before=topology_before["adoption"],
        adoption_topology_sha256_after=topology_after["adoption"],
        engineering_payload_sha256_before=engineering_before,
        engineering_payload_sha256_after=engineering_after,
        negative_identifier_counts=negative_counts,
        quick_check=quick_check,
        foreign_key_issues=foreign_key_issues,
        orphan_counts=orphans,
        vde_remap_csv=str(vde_csv),
        fuelcons_remap_csv=str(fuel_csv),
        report_json=str(report_json),
    )
    report_json.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, default=DEFAULT_SOURCE_DB)
    parser.add_argument("--output-db", type=Path, default=DEFAULT_OUTPUT_DB)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = normalize_candidate_database(
        args.source_db, args.output_db, args.artifact_dir, rebuild=args.rebuild
    )
    print(json.dumps(asdict(report), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
