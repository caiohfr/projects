#!/usr/bin/env python3
"""Import the frozen synthetic component references into a canonical DB.

Dry-run is the default.  The source ABC values and resolution boundaries are
never recalculated or relabelled.  A narrow seed adapter reconciles the source
metadata vocabulary with the frozen canonical CHECK constraints while keeping
every original label in provenance JSON.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import sqlite3
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED_FILES = {
    "component_db_seed.csv",
    "component_resolution_seed.csv",
    "synthetic_reference_catalog.csv",
    "component_resolution_manifest.csv",
    "qa_summary.csv",
    "README.md",
}
EXPECTED_TOTAL = 50
EXPECTED_SOURCE_DOMAIN_COUNTS = {
    "BRAKE": 12,
    "TRANSMISSION": 14,
    "AXLE": 14,
    "HUB_BEARING": 10,
}
EXPECTED_CANONICAL_DOMAIN_COUNTS = {
    "BRAKE": 12,
    "TRANSMISSION": 14,
    "AXLE_HUBS": 24,
}
KPI_COLUMNS = {
    "estimate_status": "TEXT",
    "estimator_version": "TEXT",
    "fit_nrmse_pct": "REAL",
    "condition_number": "REAL",
    "sensitivity_rel_pct": "REAL",
}
FORBIDDEN_EXPORT_HEADERS = {
    "program",
    "sales_code",
    "vin",
    "part_number",
    "vehicle_id",
}
CANONICAL_SOURCE_NAME = "SYNTHETIC_REFERENCE"


def norm(value: Any) -> str:
    return str(value or "").strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_from_zip(zf: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    raw = zf.read(name).decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(raw)))


def table_columns(conn: sqlite3.Connection, table: str) -> dict[str, str]:
    return {row[1]: row[2] for row in conn.execute(f'PRAGMA table_info("{table}")')}


def pk_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    return [row[1] for row in sorted(rows, key=lambda row: row[5]) if row[5]]


def parse_json(value: Any) -> dict[str, Any]:
    if value is None or str(value).strip() == "":
        return {}
    parsed = json.loads(str(value))
    return parsed if isinstance(parsed, dict) else {"source_value": parsed}


def dump_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _unique_nonblank(rows: list[dict[str, Any]], field: str) -> set[str]:
    return {norm(row.get(field)) for row in rows if norm(row.get(field))}


def validate_source(
    component_rows: list[dict[str, str]],
    resolution_rows: list[dict[str, str]],
    catalog_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
    qa_rows: list[dict[str, str]],
) -> None:
    errors: list[str] = []
    for name, rows in (
        ("component_db_seed.csv", component_rows),
        ("component_resolution_seed.csv", resolution_rows),
        ("synthetic_reference_catalog.csv", catalog_rows),
        ("component_resolution_manifest.csv", manifest_rows),
        ("qa_summary.csv", qa_rows),
    ):
        if len(rows) != EXPECTED_TOTAL:
            errors.append(f"{name} rows={len(rows)} expected={EXPECTED_TOTAL}")

    headers: set[str] = set()
    for rows in (component_rows, resolution_rows, catalog_rows, manifest_rows, qa_rows):
        if rows:
            headers.update(str(header).lower() for header in rows[0])
    leaked = sorted(
        header
        for header in headers
        if header in FORBIDDEN_EXPORT_HEADERS or header.startswith("sales_code")
    )
    if leaked:
        errors.append(f"Forbidden identity headers present: {leaked}")

    if any(norm(row.get("manufacturer")) for row in component_rows):
        errors.append("Synthetic component manufacturer values must be blank")

    domain_counts = Counter(norm(row.get("component_domain")).upper() for row in component_rows)
    if dict(domain_counts) != EXPECTED_SOURCE_DOMAIN_COUNTS:
        errors.append(
            f"Source domain counts={dict(domain_counts)} expected={EXPECTED_SOURCE_DOMAIN_COUNTS}"
        )

    component_ids = [norm(row.get("component_id")) for row in component_rows]
    resolution_ids = [norm(row.get("component_resolution_id")) for row in resolution_rows]
    manifest_component_ids = [norm(row.get("component_id")) for row in manifest_rows]
    manifest_resolution_ids = [norm(row.get("component_resolution_id")) for row in manifest_rows]
    catalog_component_ids = [norm(row.get("component_id")) for row in catalog_rows]
    catalog_resolution_ids = [norm(row.get("component_resolution_id")) for row in catalog_rows]
    qa_component_ids = [norm(row.get("component_id")) for row in qa_rows]
    for label, values in (
        ("component seed IDs", component_ids),
        ("resolution seed IDs", resolution_ids),
        ("manifest component IDs", manifest_component_ids),
        ("manifest resolution IDs", manifest_resolution_ids),
        ("catalog component IDs", catalog_component_ids),
        ("catalog resolution IDs", catalog_resolution_ids),
        ("QA component IDs", qa_component_ids),
    ):
        if any(not value for value in values):
            errors.append(f"{label} contain blank values")
        if len(values) != len(set(values)):
            errors.append(f"{label} are not unique")

    if set(component_ids) != set(manifest_component_ids) or set(component_ids) != set(catalog_component_ids) or set(component_ids) != set(qa_component_ids):
        errors.append("Component IDs do not reconcile across seed, manifest, catalog and QA")
    if set(resolution_ids) != set(manifest_resolution_ids) or set(resolution_ids) != set(catalog_resolution_ids):
        errors.append("Resolution IDs do not reconcile across seed, manifest and catalog")

    if _unique_nonblank(resolution_rows, "confidence") != {"REFERENCE"}:
        errors.append("Expected source resolution confidence label REFERENCE")
    if _unique_nonblank(resolution_rows, "fidelity_level") != {"SYNTHETIC_POPULATION"}:
        errors.append("Expected source resolution fidelity label SYNTHETIC_POPULATION")
    if _unique_nonblank(qa_rows, "qa_status") != {"PASS"}:
        errors.append("All source QA rows must have qa_status=PASS")
    if any(norm(row.get("source_identifiers_exported")).upper() not in {"FALSE", "0", "NO"} for row in qa_rows):
        errors.append("QA reports exported source identifiers")

    if errors:
        raise ValueError("; ".join(errors))


def adapt_seed_rows(
    component_rows: list[dict[str, Any]],
    resolution_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
) -> None:
    """Apply only the approved seed-to-canonical vocabulary bridge."""
    resolutions_by_component: dict[str, list[str]] = defaultdict(list)
    components_by_resolution: dict[str, str] = {}
    for row in manifest_rows:
        component_id = norm(row.get("component_id"))
        resolution_id = norm(row.get("component_resolution_id"))
        resolutions_by_component[component_id].append(resolution_id)
        components_by_resolution[resolution_id] = component_id

    for row in component_rows:
        component_id = norm(row.get("component_id"))
        original_domain = norm(row.get("component_domain")).upper()
        provenance = parse_json(row.get("provenance_json"))
        provenance["seed_component_domain_original"] = original_domain
        provenance.setdefault("synthetic_reference", True)
        provenance.setdefault("source_artifact", "EcoDrive_Synthetic_Components_v1.zip")
        row["provenance_json"] = dump_json(provenance)

        if original_domain in {"AXLE", "HUB_BEARING"}:
            row["component_domain"] = "AXLE_HUBS"
        row["source_name"] = CANONICAL_SOURCE_NAME

        properties = parse_json(row.get("custom_properties_json"))
        properties.setdefault("synthetic_reference", True)
        properties["synthetic_reference_resolution_ids"] = resolutions_by_component[component_id]
        row["custom_properties_json"] = dump_json(properties)

    for row in resolution_rows:
        resolution_id = norm(row.get("component_resolution_id"))
        original_confidence = norm(row.get("confidence"))
        original_fidelity = norm(row.get("fidelity_level"))
        provenance = parse_json(row.get("provenance_json"))
        provenance["seed_confidence_original"] = original_confidence
        provenance["seed_fidelity_level_original"] = original_fidelity
        provenance.setdefault(
            "synthetic_reference_component_id", components_by_resolution.get(resolution_id)
        )
        provenance.setdefault("synthetic_reference", True)
        provenance.setdefault("source_artifact", "EcoDrive_Synthetic_Components_v1.zip")
        row["provenance_json"] = dump_json(provenance)
        row["confidence"] = None
        row["fidelity_level"] = None
        if not norm(row.get("vehicle_configuration_id")):
            row["vehicle_configuration_id"] = None


def load_and_adapt_seed(seed_zip: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with zipfile.ZipFile(seed_zip) as zf:
        names = set(zf.namelist())
        missing = EXPECTED_FILES - names
        if missing:
            raise ValueError(f"Seed ZIP missing expected members: {sorted(missing)}")
        component_rows = read_csv_from_zip(zf, "component_db_seed.csv")
        resolution_rows = read_csv_from_zip(zf, "component_resolution_seed.csv")
        catalog_rows = read_csv_from_zip(zf, "synthetic_reference_catalog.csv")
        manifest_rows = read_csv_from_zip(zf, "component_resolution_manifest.csv")
        qa_rows = read_csv_from_zip(zf, "qa_summary.csv")
    validate_source(component_rows, resolution_rows, catalog_rows, manifest_rows, qa_rows)
    adapt_seed_rows(component_rows, resolution_rows, manifest_rows)
    return component_rows, resolution_rows


def ensure_kpi_columns(conn: sqlite3.Connection, *, apply: bool) -> list[str]:
    existing = table_columns(conn, "component_resolution")
    missing = [name for name in KPI_COLUMNS if name not in existing]
    if apply:
        for name in missing:
            conn.execute(
                f'ALTER TABLE component_resolution ADD COLUMN "{name}" {KPI_COLUMNS[name]}'
            )
    return missing


def coerce_value(value: Any, sqlite_type: str) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    declared_type = (sqlite_type or "").upper()
    if "INT" in declared_type:
        return int(float(text))
    if any(token in declared_type for token in ("REAL", "FLOA", "DOUB", "NUM")):
        return float(text)
    return text


def insert_rows(
    conn: sqlite3.Connection,
    table: str,
    rows: list[dict[str, Any]],
    *,
    apply: bool,
) -> dict[str, int]:
    columns = table_columns(conn, table)
    primary_keys = pk_columns(conn, table)
    if not primary_keys:
        raise ValueError(f"No primary key discovered for {table}")
    inserted = skipped = 0
    for row in rows:
        payload = {
            key: coerce_value(value, columns[key])
            for key, value in row.items()
            if key in columns and key not in {"created_at", "updated_at"}
        }
        if any(payload.get(key) in (None, "") for key in primary_keys):
            raise ValueError(f"{table} row is missing primary key values")
        where = " AND ".join(f'"{key}"=?' for key in primary_keys)
        primary_values = [payload[key] for key in primary_keys]
        if conn.execute(f'SELECT 1 FROM "{table}" WHERE {where} LIMIT 1', primary_values).fetchone():
            skipped += 1
            continue
        if apply:
            names = list(payload)
            conn.execute(
                f'INSERT INTO "{table}" ({",".join(f"{name}" for name in names)}) '
                f'VALUES ({",".join("?" for _ in names)})',
                [payload[name] for name in names],
            )
        inserted += 1
    return {"inserted": inserted, "skipped_existing": skipped}


def _count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def run_import(
    db_path: Path,
    seed_zip: Path,
    *,
    apply: bool = False,
    backup: Path | None = None,
) -> dict[str, Any]:
    db_path = Path(db_path)
    seed_zip = Path(seed_zip)
    if not db_path.is_file():
        raise FileNotFoundError(f"DB not found: {db_path}")
    if not seed_zip.is_file():
        raise FileNotFoundError(f"Seed ZIP not found: {seed_zip}")

    component_rows, resolution_rows = load_and_adapt_seed(seed_zip)
    hash_before = sha256(db_path)
    if apply:
        backup_path = Path(backup) if backup else db_path.with_suffix(
            db_path.suffix + ".pre_sprint12_component_seed.bak"
        )
        if backup_path.resolve() == db_path.resolve():
            raise ValueError("Backup path must differ from DB path")
        if not backup_path.exists():
            shutil.copy2(db_path, backup_path)
    else:
        backup_path = None

    uri = f"file:{db_path.resolve().as_posix()}?mode={'rw' if apply else 'ro'}"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        required = {
            "component_db",
            "component_resolution",
            "component_instance",
            "vde_component_resolution",
            "vde",
            "vehicle_configuration",
        }
        existing = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        missing = required - existing
        if missing:
            raise ValueError(f"Canonical DB missing required tables: {sorted(missing)}")
        quick_before = conn.execute("PRAGMA quick_check").fetchone()[0]
        fk_before = len(conn.execute("PRAGMA foreign_key_check").fetchall())
        if quick_before != "ok" or fk_before:
            raise ValueError(
                f"Pre-write integrity failed: quick_check={quick_before}, fk_issues={fk_before}"
            )

        before_counts = {
            table: _count(conn, table)
            for table in ("component_db", "component_resolution", "component_instance")
        }
        missing_kpis = ensure_kpi_columns(conn, apply=apply)
        component_stats = insert_rows(conn, "component_db", component_rows, apply=apply)
        resolution_stats = insert_rows(
            conn, "component_resolution", resolution_rows, apply=apply
        )
        quick_pending = conn.execute("PRAGMA quick_check").fetchone()[0]
        fk_pending = len(conn.execute("PRAGMA foreign_key_check").fetchall())
        if quick_pending != "ok" or fk_pending:
            raise ValueError(
                "Pending-write integrity failed: "
                f"quick_check={quick_pending}, fk_issues={fk_pending}"
            )
        if apply:
            conn.commit()
        else:
            conn.rollback()

        quick_after = conn.execute("PRAGMA quick_check").fetchone()[0]
        fk_after = len(conn.execute("PRAGMA foreign_key_check").fetchall())
        after_counts = {
            table: _count(conn, table)
            for table in ("component_db", "component_resolution", "component_instance")
        }
        if quick_after != "ok" or fk_after:
            raise ValueError(
                f"Post-write integrity failed: quick_check={quick_after}, fk_issues={fk_after}"
            )
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    hash_after = sha256(db_path)
    if not apply and hash_after != hash_before:
        raise RuntimeError("Dry-run modified the source DB")
    return {
        "mode": "apply" if apply else "dry-run",
        "db": str(db_path),
        "backup": str(backup_path) if backup_path else None,
        "sha256_before": hash_before,
        "sha256_after": hash_after,
        "quick_check_before": quick_before,
        "quick_check_after": quick_after,
        "foreign_key_issues_before": fk_before,
        "foreign_key_issues_after": fk_after,
        "kpi_columns_added_or_pending": missing_kpis,
        "component_db": component_stats,
        "component_resolution": resolution_stats,
        "row_counts_before": before_counts,
        "row_counts_after": after_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--seed-zip", required=True, type=Path)
    parser.add_argument("--apply", action="store_true", help="Write changes; default is dry-run")
    parser.add_argument("--backup", type=Path)
    args = parser.parse_args()
    try:
        report = run_import(
            args.db,
            args.seed_zip,
            apply=args.apply,
            backup=args.backup,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
