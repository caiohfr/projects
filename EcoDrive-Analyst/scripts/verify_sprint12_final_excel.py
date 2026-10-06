from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import xml.etree.ElementTree as ET
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from vde_core.sprint12_final_closure import file_sha256, open_read_only
from scripts.export_sprint12_final_excel import TABLE_SHEETS


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _sheet_parts(archive: zipfile.ZipFile) -> dict[str, str]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {
        node.attrib["Id"]: node.attrib["Target"]
        for node in rels.findall(f"{{{PACKAGE_REL_NS}}}Relationship")
    }
    parts: dict[str, str] = {}
    for sheet in workbook.findall(f".//{{{MAIN_NS}}}sheet"):
        name = sheet.attrib["name"]
        rel_id = sheet.attrib[f"{{{REL_NS}}}id"]
        target = targets[rel_id].lstrip("/")
        parts[name] = target if target.startswith("xl/") else f"xl/{target}"
    return parts


def _sheet_shape(archive: zipfile.ZipFile, part: str) -> tuple[int, int]:
    row_count = 0
    header_cells = 0
    with archive.open(part) as source:
        for event, element in ET.iterparse(source, events=("end",)):
            if element.tag == f"{{{MAIN_NS}}}row":
                row_count += 1
                if row_count == 1:
                    header_cells = len(element.findall(f"{{{MAIN_NS}}}c"))
                element.clear()
    return max(row_count - 1, 0), header_cells


def verify(db_path: Path, workbook_path: Path) -> dict[str, object]:
    db_path = db_path.resolve(strict=True)
    workbook_path = workbook_path.resolve(strict=True)
    required = {
        "00_README", "01_DB_SUMMARY", "02_DATA_DICTIONARY", "03_FOREIGN_KEYS",
        "VDE_FLAT", "COMP_COVERAGE", *TABLE_SHEETS.values(),
    }
    failures: list[str] = []
    raw_reconciliation: list[dict[str, object]] = []
    with zipfile.ZipFile(workbook_path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            failures.append(f"Corrupt ZIP member: {bad_member}")
        parts = _sheet_parts(archive)
        missing = sorted(required - set(parts))
        if missing:
            failures.append(f"Missing sheets: {missing}")
        readme = archive.read(parts["00_README"]).decode("utf-8") if "00_README" in parts else ""
        db_hash = file_sha256(db_path)
        if db_hash not in readme:
            failures.append("Final DB SHA256 not found in 00_README")

        connection = open_read_only(db_path)
        try:
            for table, sheet in TABLE_SHEETS.items():
                db_rows = int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
                db_columns = len(list(connection.execute(f'PRAGMA table_info("{table}")')))
                if sheet not in parts:
                    continue
                excel_rows, excel_columns = _sheet_shape(archive, parts[sheet])
                ok = db_rows == excel_rows and db_columns == excel_columns
                raw_reconciliation.append({
                    "table": table, "sheet": sheet,
                    "db_rows": db_rows, "excel_rows": excel_rows,
                    "db_columns": db_columns, "excel_columns": excel_columns,
                    "match": ok,
                })
                if not ok:
                    failures.append(f"Raw reconciliation mismatch: {table}/{sheet}")
            vde_count = int(connection.execute("SELECT COUNT(*) FROM vde").fetchone()[0])
            derived: dict[str, dict[str, object]] = {}
            for sheet in ("VDE_FLAT", "COMP_COVERAGE"):
                if sheet not in parts:
                    continue
                rows, columns = _sheet_shape(archive, parts[sheet])
                derived[sheet] = {"rows": rows, "columns": columns, "expected_rows": vde_count, "match": rows == vde_count}
                if rows != vde_count:
                    failures.append(f"Derived row mismatch: {sheet}")
        finally:
            connection.close()

    return {
        "status": "PASS" if not failures else "FAIL",
        "db_sha256": file_sha256(db_path),
        "workbook_sha256": file_sha256(workbook_path),
        "sheet_names": list(parts),
        "raw_reconciliation": raw_reconciliation,
        "derived_reconciliation": derived,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.db, args.workbook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
