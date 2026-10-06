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

from scripts.export_rolling_minor_group_estimated_v0_excel import sha256
from scripts.export_db_review_excel import open_database_read_only


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
REQUIRED = (
    "00_README", "01_CLOSURE_SUMMARY", "02_DB_INVENTORY", "03_SCHEMA",
    "04_SHARE_SUMMARY", "05_COVERAGE", "06_PRIORS", "07_ESTIMATES",
    "08_VDE_REVIEW", "09_CHECKS",
)


def _sheet_parts(archive: zipfile.ZipFile) -> dict[str, str]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {
        node.attrib["Id"]: node.attrib["Target"]
        for node in relationships.findall(f"{{{PACKAGE_REL_NS}}}Relationship")
    }
    parts: dict[str, str] = {}
    for sheet in workbook.findall(f".//{{{MAIN_NS}}}sheet"):
        target = targets[sheet.attrib[f"{{{REL_NS}}}id"]].lstrip("/")
        parts[sheet.attrib["name"]] = target if target.startswith("xl/") else f"xl/{target}"
    return parts


def _shape(archive: zipfile.ZipFile, part: str) -> tuple[int, int]:
    row_count = 0
    header_count = 0
    with archive.open(part) as source:
        for _, element in ET.iterparse(source, events=("end",)):
            if element.tag == f"{{{MAIN_NS}}}row":
                row_count += 1
                if row_count == 1:
                    header_count = len(element.findall(f"{{{MAIN_NS}}}c"))
                element.clear()
    return max(row_count - 1, 0), header_count


def _all_text(archive: zipfile.ZipFile, parts: dict[str, str]) -> str:
    values: list[str] = []
    if "xl/sharedStrings.xml" in archive.namelist():
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        values.extend("".join(node.itertext()) for node in root.findall(f"{{{MAIN_NS}}}si"))
    for part in parts.values():
        root = ET.fromstring(archive.read(part))
        values.extend("".join(node.itertext()) for node in root.findall(f".//{{{MAIN_NS}}}is"))
    return "\n".join(values)


def verify(db_path: Path, workbook_path: Path) -> dict[str, object]:
    db_path = db_path.resolve(strict=True)
    workbook_path = workbook_path.resolve(strict=True)
    failures: list[str] = []
    with zipfile.ZipFile(workbook_path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            failures.append(f"Corrupt OOXML member: {bad_member}")
        parts = _sheet_parts(archive)
        missing = [name for name in REQUIRED if name not in parts]
        if missing:
            failures.append(f"Missing worksheets: {missing}")
        shapes = {name: _shape(archive, part) for name, part in parts.items()}
        chart_parts = sorted(name for name in archive.namelist() if name.startswith("xl/charts/chart") and name.endswith(".xml"))
        if not chart_parts:
            failures.append("Expected share-summary chart is absent")
        if sha256(db_path) not in _all_text(archive, parts):
            failures.append("Source DB SHA256 is absent from the workbook")

    connection = open_database_read_only(db_path)
    try:
        db_counts = {
            "06_PRIORS": int(connection.execute("SELECT COUNT(*) FROM rolling_minor_group_prior_v0").fetchone()[0]),
            "07_ESTIMATES": int(connection.execute("SELECT COUNT(*) FROM rolling_minor_group_estimate_v0").fetchone()[0]),
            "08_VDE_REVIEW": int(connection.execute("SELECT COUNT(*) FROM rolling_minor_group_estimate_v0").fetchone()[0]),
        }
        quick = connection.execute("PRAGMA quick_check").fetchone()[0]
        fk_issues = len(connection.execute("PRAGMA foreign_key_check").fetchall())
    finally:
        connection.close()
    reconciliation: list[dict[str, object]] = []
    expected_columns = {"06_PRIORS": 11, "07_ESTIMATES": 26, "08_VDE_REVIEW": 37}
    for sheet, expected_rows in db_counts.items():
        actual_rows, actual_columns = shapes.get(sheet, (-1, -1))
        match = actual_rows == expected_rows and actual_columns == expected_columns[sheet]
        reconciliation.append({
            "sheet": sheet,
            "expected_rows": expected_rows,
            "actual_rows": actual_rows,
            "expected_columns": expected_columns[sheet],
            "actual_columns": actual_columns,
            "match": match,
        })
        if not match:
            failures.append(f"Row/column reconciliation failed for {sheet}")
    if shapes.get("04_SHARE_SUMMARY", (-1, -1))[0] != 5:
        failures.append("Share summary must contain five drive groups")
    if shapes.get("05_COVERAGE", (-1, -1))[0] != 8:
        failures.append("Coverage sheet must contain eight metrics")
    if shapes.get("09_CHECKS", (-1, -1))[0] != 11:
        failures.append("Checks sheet must contain eleven checks")
    if quick != "ok" or fk_issues != 0:
        failures.append(f"SQLite integrity failed: quick={quick}, fk={fk_issues}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "workbook": str(workbook_path),
        "workbook_sha256": sha256(workbook_path),
        "source_db": str(db_path),
        "source_db_sha256": sha256(db_path),
        "quick_check": quick,
        "foreign_key_issues": fk_issues,
        "sheets": list(parts),
        "sheet_shapes": {name: {"data_rows": shape[0], "columns": shape[1]} for name, shape in shapes.items()},
        "chart_parts": chart_parts,
        "reconciliation": reconciliation,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.db, args.workbook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest.update({
        "workbook_sha256": result["workbook_sha256"],
        "verification_status": result["status"],
        "excel_native_open_and_chart": True,
        "verification_report": str(args.output.resolve()),
        "chart_count": len(result["chart_parts"]),
    })
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
