from __future__ import annotations

import hashlib
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile

from scripts.export_db_review_excel import export_database_review


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sheet_targets(workbook_path: Path) -> dict[str, str]:
    with zipfile.ZipFile(workbook_path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {
        relationship.attrib["Id"]: relationship.attrib["Target"]
        for relationship in relationships.findall(f"{{{PACKAGE_REL_NS}}}Relationship")
    }
    return {
        sheet.attrib["name"]: "xl/" + targets[sheet.attrib[f"{{{REL_NS}}}id"]]
        for sheet in workbook.findall(f".//{{{MAIN_NS}}}sheet")
    }


def _column_index(reference: str) -> int:
    letters = "".join(character for character in reference if character.isalpha())
    result = 0
    for character in letters:
        result = result * 26 + ord(character) - 64
    return result - 1


def _sheet_rows(workbook_path: Path, sheet_name: str) -> list[list[object]]:
    target = _sheet_targets(workbook_path)[sheet_name]
    with zipfile.ZipFile(workbook_path) as archive:
        root = ET.fromstring(archive.read(target))
    rows: list[list[object]] = []
    for row in root.findall(f".//{{{MAIN_NS}}}row"):
        values: list[object] = []
        for cell in row.findall(f"{{{MAIN_NS}}}c"):
            index = _column_index(cell.attrib["r"])
            while len(values) <= index:
                values.append(None)
            if cell.attrib.get("t") == "inlineStr":
                text_nodes = cell.findall(f".//{{{MAIN_NS}}}t")
                value: object = "".join(node.text or "" for node in text_nodes)
            else:
                value_node = cell.find(f"{{{MAIN_NS}}}v")
                if value_node is None:
                    value = None
                else:
                    raw = value_node.text or ""
                    value = float(raw) if any(marker in raw for marker in (".", "e", "E")) else int(raw)
            values[index] = value
        rows.append(values)
    return rows


class ExportDatabaseReviewExcelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = tempfile.TemporaryDirectory(prefix="db_review_test_")
        root = Path(cls._temporary_directory.name)
        cls.root = root
        cls.db_path = root / "fixture.db"
        cls.workbook_path = root / "review.xlsx"
        with closing(sqlite3.connect(cls.db_path)) as connection:
            connection.executescript(
                """
                CREATE TABLE program (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    record_origin TEXT,
                    record_status TEXT
                );
                CREATE TABLE vehicle_configuration (
                    id INTEGER PRIMARY KEY,
                    program_id INTEGER NOT NULL,
                    label TEXT,
                    FOREIGN KEY (program_id) REFERENCES program(id)
                );
                CREATE TABLE vde (
                    id INTEGER PRIMARY KEY,
                    vehicle_configuration_id INTEGER,
                    cycle TEXT,
                    record_origin TEXT,
                    record_status TEXT,
                    FOREIGN KEY (vehicle_configuration_id) REFERENCES vehicle_configuration(id)
                );
                CREATE TABLE run (id INTEGER PRIMARY KEY, vde_id INTEGER);
                CREATE TABLE fuelcons (id INTEGER PRIMARY KEY, run_id INTEGER, value REAL);
                CREATE TABLE helper_audit (code TEXT PRIMARY KEY, note TEXT);
                CREATE VIEW vde_db AS SELECT id, cycle, record_origin, record_status FROM vde;
                INSERT INTO program VALUES (-7, 'Negative program', 'IMPORTED_REFERENCE', 'ACTIVE');
                INSERT INTO program VALUES (2, 'Manual program', 'MANUAL', 'DRAFT');
                INSERT INTO vehicle_configuration VALUES (-11, -7, 'Configuration');
                INSERT INTO vde VALUES (-101, -11, 'FTP', 'IMPORTED_REFERENCE', 'ACTIVE');
                INSERT INTO run VALUES (-201, -101);
                INSERT INTO fuelcons VALUES (-301, -201, 5.25);
                INSERT INTO helper_audit VALUES ('A', NULL);
                """
            )
        cls.hash_before = _sha256(cls.db_path)
        cls.report = export_database_review(cls.db_path, cls.workbook_path)
        cls.hash_after = _sha256(cls.db_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def test_export_does_not_modify_database_hash(self):
        self.assertEqual(self.hash_before, self.hash_after)
        self.assertEqual(self.report.source_sha256_before, self.report.source_sha256_after)

    def test_core_sheets_are_created(self):
        sheets = set(_sheet_targets(self.workbook_path))
        self.assertTrue(
            {
                "00_SUMMARY",
                "01_SCHEMA",
                "02_RELATIONSHIPS",
                "03_ORIGIN_COUNTS",
                "04_NULL_COVERAGE",
                "EPA_CARRYOVER_REVIEW",
                "RUN_IDENTITY_REVIEW",
                "VDE_DUPLICATE_CANDIDATES",
                "FUELCONS_DUPLICATE_CANDIDATES",
                "JSON_SCALAR_REVIEW",
                "program",
                "vehicle_configuration",
                "vde",
                "run",
                "fuelcons",
                "vde_db_view",
            }.issubset(sheets)
        )

    def test_semantic_review_sheets_are_safe_for_older_schemas(self):
        for name in (
            "EPA_CARRYOVER_REVIEW",
            "RUN_IDENTITY_REVIEW",
            "VDE_DUPLICATE_CANDIDATES",
            "FUELCONS_DUPLICATE_CANDIDATES",
            "JSON_SCALAR_REVIEW",
        ):
            rows = _sheet_rows(self.workbook_path, name)
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0])

    def test_excel_row_counts_match_sqlite(self):
        with closing(
            sqlite3.connect(f"file:{self.db_path.as_posix()}?mode=ro", uri=True)
        ) as connection:
            for table in ("program", "vehicle_configuration", "vde", "run", "fuelcons"):
                expected = connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                self.assertEqual(len(_sheet_rows(self.workbook_path, table)) - 1, expected)

    def test_negative_ids_are_preserved_as_numeric_values(self):
        rows = _sheet_rows(self.workbook_path, "vde")
        self.assertEqual(rows[0][0], "id")
        self.assertEqual(rows[1][0], -101)
        self.assertIsInstance(rows[1][0], int)

    def test_schema_origin_and_relationship_sheets_use_actual_metadata(self):
        schema_rows = _sheet_rows(self.workbook_path, "01_SCHEMA")
        self.assertIn(["program", "table", 2, "name", "TEXT", True, None, 0], schema_rows)

        origin_rows = _sheet_rows(self.workbook_path, "03_ORIGIN_COUNTS")
        self.assertIn(["program", "IMPORTED_REFERENCE", "ACTIVE", 1], origin_rows)
        self.assertIn(["program", "MANUAL", "DRAFT", 1], origin_rows)

        relationship_rows = _sheet_rows(self.workbook_path, "02_RELATIONSHIPS")
        self.assertIn(
            ["vehicle_configuration", "program_id", "program", "id", "NO ACTION", "NO ACTION"],
            relationship_rows,
        )

    def test_optional_id_remap_sheets_preserve_numeric_ids(self):
        vde_remap = self.root / "VDE_ID_REMAP.csv"
        fuelcons_remap = self.root / "FUELCONS_ID_REMAP.csv"
        workbook = self.root / "review_with_remap.xlsx"
        vde_remap.write_text(
            "old_id,new_id,source_record_id,make,model,year\n"
            "-1000001,1,EPA-1,BMW,330i,2021\n",
            encoding="utf-8",
        )
        fuelcons_remap.write_text(
            "old_id,new_id,source_record_id,vde_id,year\n"
            "-4000001,1,FC-1,1,2021\n",
            encoding="utf-8",
        )
        export_database_review(
            self.db_path,
            workbook,
            vde_id_remap=vde_remap,
            fuelcons_id_remap=fuelcons_remap,
        )
        sheets = set(_sheet_targets(workbook))
        self.assertTrue({"VDE_ID_REMAP", "FUELCONS_ID_REMAP"} <= sheets)
        self.assertEqual(
            _sheet_rows(workbook, "VDE_ID_REMAP")[1][:2], [-1000001, 1]
        )
        self.assertEqual(
            _sheet_rows(workbook, "FUELCONS_ID_REMAP")[1][:2], [-4000001, 1]
        )


if __name__ == "__main__":
    unittest.main()
