"""Export a compact, read-only SQLite schema audit to Markdown."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sqlite3
from typing import Any


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def compact(value: Any) -> str:
    if value is None:
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def audit(db_path: Path, output_path: Path) -> None:
    resolved = db_path.resolve(strict=True)
    before = sha256(resolved)
    connection = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only = ON")
    try:
        objects = connection.execute(
            """
            SELECT name, type, sql
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name NOT LIKE 'sqlite_%'
            ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END, name
            """
        ).fetchall()
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        foreign_key_issues = len(connection.execute("PRAGMA foreign_key_check").fetchall())
        sections: list[str] = []
        summary_rows: list[str] = []
        compact_sections: list[str] = []
        relationship_rows: list[str] = []
        for name, object_type, _sql in objects:
            quoted = quote_identifier(name)
            row_count = connection.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]
            columns = connection.execute(f"PRAGMA table_xinfo({quoted})").fetchall()
            primary_key = ", ".join(
                column[1] for column in sorted(columns, key=lambda item: item[5]) if column[5]
            ) or "—"
            summary_rows.append(
                f"| `{name}` | {object_type} | {row_count:,} | {len(columns)} | {primary_key} |"
            )
            compact_sections.extend([
                f"### `{name}` ({object_type})", "",
                ", ".join(f"`{column[1]}`" for column in columns), "",
            ])
            lines = [
                f"### `{name}` ({object_type}; {row_count:,} rows)", "",
                "| # | Field | SQLite type | Required | Default | PK order | Hidden/generated |",
                "|---:|---|---|:---:|---|---:|---:|",
            ]
            for cid, column_name, column_type, not_null, default, pk_order, hidden in columns:
                lines.append(
                    f"| {cid} | `{column_name}` | `{column_type or 'ANY'}` | "
                    f"{'YES' if not_null else 'NO'} | {compact(default)} | {pk_order or '—'} | {hidden} |"
                )
            foreign_keys = connection.execute(f"PRAGMA foreign_key_list({quoted})").fetchall()
            if foreign_keys:
                lines.extend(["", "Foreign keys:", ""])
                for fk in foreign_keys:
                    lines.append(
                        f"- `{name}.{fk[3]}` → `{fk[2]}.{fk[4]}` "
                        f"(`ON UPDATE {fk[5]}`, `ON DELETE {fk[6]}`)"
                    )
                    relationship_rows.append(
                        f"| `{name}.{fk[3]}` | `{fk[2]}.{fk[4]}` | {fk[5]} | {fk[6]} |"
                    )
            sections.extend([*lines, ""])
    finally:
        connection.close()
    after = sha256(resolved)
    if before != after:
        raise RuntimeError("Source DB hash changed during read-only schema audit")

    lines = [
        "# EcoDrive Canonical DB — Compact Schema Audit", "",
        f"- Database: `{resolved}`",
        f"- Size: **{resolved.stat().st_size:,} bytes**",
        f"- SHA256 before/after: `{before}` / `{after}`",
        f"- `PRAGMA quick_check`: **{quick_check}**",
        f"- `PRAGMA foreign_key_check` issues: **{foreign_key_issues}**",
        f"- Objects: **{len(objects)}**", "",
        "## Object inventory", "",
        "| Object | Type | Rows | Columns | Primary key |",
        "|---|---|---:|---:|---|",
        *summary_rows, "",
        "## Compact field index", "",
        *compact_sections,
        "## Declared relationships", "",
    ]
    if relationship_rows:
        lines.extend([
            "| Child | Parent | On update | On delete |",
            "|---|---|---|---|", *relationship_rows,
        ])
    else:
        lines.append("No declared SQLite foreign keys.")
    lines.extend(["", "## Fields by object", "", *sections])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit(args.db, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
