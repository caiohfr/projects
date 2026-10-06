from __future__ import annotations

import argparse
import csv
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from itertools import chain, islice
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Iterable, Iterator, Sequence
from xml.sax.saxutils import escape, quoteattr
import zipfile


EXCEL_MAX_ROWS = 1_048_576
EXCEL_MAX_COLUMNS = 16_384
EXCEL_MAX_TEXT_LENGTH = 32_767
MAX_DATA_ROWS_PER_SHEET = EXCEL_MAX_ROWS - 1

METADATA_SHEETS = (
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
)

ID_REMAP_SHEETS = (
    "VDE_ID_REMAP",
    "FUELCONS_ID_REMAP",
)

CORE_OBJECT_ORDER = (
    "program",
    "vehicle_configuration",
    "vde",
    "run",
    "fuelcons",
    "fuelcons_run_adoption",
    "vde_component_resolution",
    "component_db",
    "tire_db",
    "vde_db",
    "fuelcons_db",
    "vde_db_view",
    "fuelcons_db_view",
)


@dataclass(frozen=True)
class DatabaseObject:
    name: str
    object_type: str
    row_count: int
    columns: tuple[sqlite3.Row, ...]
    sheet_names: tuple[str, ...]


@dataclass(frozen=True)
class ExportReport:
    workbook_path: Path
    sheet_names: tuple[str, ...]
    source_sha256_before: str
    source_sha256_after: str
    row_counts: dict[str, int]
    additional_objects: tuple[str, ...]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sqlite_read_only_authorizer(
    action: int,
    _arg1: str | None,
    _arg2: str | None,
    _database: str | None,
    _trigger: str | None,
) -> int:
    denied_names = (
        "SQLITE_INSERT",
        "SQLITE_UPDATE",
        "SQLITE_DELETE",
        "SQLITE_CREATE_INDEX",
        "SQLITE_CREATE_TABLE",
        "SQLITE_CREATE_TEMP_INDEX",
        "SQLITE_CREATE_TEMP_TABLE",
        "SQLITE_CREATE_TEMP_TRIGGER",
        "SQLITE_CREATE_TEMP_VIEW",
        "SQLITE_CREATE_TRIGGER",
        "SQLITE_CREATE_VIEW",
        "SQLITE_DROP_INDEX",
        "SQLITE_DROP_TABLE",
        "SQLITE_DROP_TEMP_INDEX",
        "SQLITE_DROP_TEMP_TABLE",
        "SQLITE_DROP_TEMP_TRIGGER",
        "SQLITE_DROP_TEMP_VIEW",
        "SQLITE_DROP_TRIGGER",
        "SQLITE_DROP_VIEW",
        "SQLITE_ALTER_TABLE",
        "SQLITE_REINDEX",
        "SQLITE_ANALYZE",
        "SQLITE_ATTACH",
        "SQLITE_DETACH",
    )
    denied = {getattr(sqlite3, name) for name in denied_names if hasattr(sqlite3, name)}
    return sqlite3.SQLITE_DENY if action in denied else sqlite3.SQLITE_OK


def open_database_read_only(path: Path) -> sqlite3.Connection:
    resolved = path.resolve(strict=True)
    connection = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.set_authorizer(_sqlite_read_only_authorizer)
    return connection


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _column_letters(index: int) -> str:
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _xml_text(value: object) -> str:
    text = str(value)
    text = "".join(
        character
        for character in text
        if character in "\t\n\r"
        or 0x20 <= ord(character) <= 0xD7FF
        or 0xE000 <= ord(character) <= 0xFFFD
        or 0x10000 <= ord(character) <= 0x10FFFF
    )
    if len(text) > EXCEL_MAX_TEXT_LENGTH:
        raise ValueError(
            f"Cell text exceeds Excel's {EXCEL_MAX_TEXT_LENGTH}-character limit; "
            "export stopped rather than truncating the source value."
        )
    return escape(text)


def _cell_xml(reference: str, value: object, style: int) -> str:
    style_attribute = f' s="{style}"' if style else ""
    if value is None:
        return f'<c r="{reference}"{style_attribute}/>'
    if isinstance(value, bool):
        return f'<c r="{reference}" t="b"{style_attribute}><v>{int(value)}</v></c>'
    if isinstance(value, int):
        return f'<c r="{reference}" t="n"{style_attribute}><v>{value}</v></c>'
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite SQLite REAL value cannot be represented safely in Excel.")
        return f'<c r="{reference}" t="n"{style_attribute}><v>{repr(value)}</v></c>'
    if isinstance(value, bytes):
        raise ValueError("SQLite BLOB values cannot be exported without changing their representation.")
    return (
        f'<c r="{reference}" t="inlineStr"{style_attribute}>'
        f'<is><t xml:space="preserve">{_xml_text(value)}</t></is></c>'
    )


def _display_length(value: object) -> int:
    if value is None:
        return 0
    return max((len(part) for part in str(value).splitlines()), default=0)


@dataclass(frozen=True)
class _SheetArtifact:
    position: int
    name: str
    path: Path


class _XlsxWriter:
    def __init__(self, temp_directory: Path) -> None:
        self._temp_directory = temp_directory
        self._sheets: list[_SheetArtifact] = []

    def add_sheet(
        self,
        *,
        position: int,
        name: str,
        rows: Iterable[Sequence[object]],
        header_rows: set[int] | None = None,
        freeze_rows: int = 1,
        filter_header_row: int | None = 1,
        column_styles: dict[int, int] | None = None,
    ) -> None:
        if len(name) > 31 or re.search(r"[\\/*?:\[\]]", name):
            raise ValueError(f"Invalid Excel worksheet name: {name!r}")
        if any(sheet.name.casefold() == name.casefold() for sheet in self._sheets):
            raise ValueError(f"Duplicate Excel worksheet name: {name!r}")

        header_rows = header_rows or {1}
        column_styles = column_styles or {}
        body_path = self._temp_directory / f"body_{position:04d}.xml"
        final_path = self._temp_directory / f"sheet_{position:04d}.xml"
        widths: list[int] = []
        row_number = 0
        column_count = 0

        with body_path.open("w", encoding="utf-8", newline="") as body:
            for row_number, row in enumerate(rows, start=1):
                values = tuple(row)
                if len(values) > EXCEL_MAX_COLUMNS:
                    raise ValueError(
                        f"Worksheet {name!r} exceeds Excel's {EXCEL_MAX_COLUMNS}-column limit."
                    )
                column_count = max(column_count, len(values))
                while len(widths) < len(values):
                    widths.append(0)
                attributes = f' r="{row_number}"'
                if row_number in header_rows:
                    attributes += ' ht="36" customHeight="1"'
                body.write(f"<row{attributes}>")
                for column_index, value in enumerate(values, start=1):
                    widths[column_index - 1] = max(
                        widths[column_index - 1], _display_length(value)
                    )
                    style = 1 if row_number in header_rows else column_styles.get(column_index, 0)
                    reference = f"{_column_letters(column_index)}{row_number}"
                    body.write(_cell_xml(reference, value, style))
                body.write("</row>")
                if row_number > EXCEL_MAX_ROWS:
                    raise ValueError(f"Worksheet {name!r} exceeds Excel's row limit.")

        last_column = _column_letters(max(column_count, 1))
        with final_path.open("w", encoding="utf-8", newline="") as sheet:
            sheet.write('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>')
            sheet.write('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">')
            sheet.write("<sheetViews><sheetView workbookViewId=\"0\">")
            if freeze_rows:
                sheet.write(
                    f'<pane ySplit="{freeze_rows}" topLeftCell="A{freeze_rows + 1}" '
                    'activePane="bottomLeft" state="frozen"/>'
                )
            sheet.write("</sheetView></sheetViews>")
            sheet.write('<sheetFormatPr defaultRowHeight="15"/>')
            if widths:
                sheet.write("<cols>")
                for index, width in enumerate(widths, start=1):
                    adjusted = min(max(width + 4, 12), 44)
                    sheet.write(
                        f'<col min="{index}" max="{index}" width="{adjusted}" customWidth="1"/>'
                    )
                sheet.write("</cols>")
            sheet.write("<sheetData>")
            with body_path.open("r", encoding="utf-8") as body:
                for block in iter(lambda: body.read(1024 * 1024), ""):
                    sheet.write(block)
            sheet.write("</sheetData>")
            if filter_header_row is not None and row_number >= filter_header_row:
                sheet.write(
                    f'<autoFilter ref="A{filter_header_row}:{last_column}{max(row_number, filter_header_row)}"/>'
                )
            sheet.write("</worksheet>")
        body_path.unlink()
        self._sheets.append(_SheetArtifact(position=position, name=name, path=final_path))

    def save(self, output_path: Path) -> tuple[str, ...]:
        sheets = sorted(self._sheets, key=lambda sheet: sheet.position)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
        timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

        content_types = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
            '<Default Extension="xml" ContentType="application/xml"/>',
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
            '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
        ]
        for index in range(1, len(sheets) + 1):
            content_types.append(
                f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            )
        content_types.append("</Types>")

        workbook_sheets = "".join(
            f'<sheet name={quoteattr(sheet.name)} sheetId="{index}" r:id="rId{index}"/>'
            for index, sheet in enumerate(sheets, start=1)
        )
        workbook_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f"<sheets>{workbook_sheets}</sheets></workbook>"
        )
        workbook_rels = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
        ]
        for index in range(1, len(sheets) + 1):
            workbook_rels.append(
                f'<Relationship Id="rId{index}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                f'Target="worksheets/sheet{index}.xml"/>'
            )
        workbook_rels.append(
            f'<Relationship Id="rId{len(sheets) + 1}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
            'Target="styles.xml"/></Relationships>'
        )

        styles_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<numFmts count="1"><numFmt numFmtId="164" formatCode="0.00%"/></numFmts>'
            '<fonts count="2"><font><sz val="10"/><name val="Aptos"/></font>'
            '<font><b/><color rgb="FFFFFFFF"/><sz val="10"/><name val="Aptos"/></font></fonts>'
            '<fills count="3"><fill><patternFill patternType="none"/></fill>'
            '<fill><patternFill patternType="gray125"/></fill>'
            '<fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill></fills>'
            '<borders count="2"><border><left/><right/><top/><bottom/><diagonal/></border>'
            '<border><left/><right/><top/><bottom style="thin"><color rgb="FFB4C6D7"/></bottom><diagonal/></border></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="4">'
            '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
            '<xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment wrapText="1" vertical="center"/></xf>'
            '<xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
            '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf>'
            '</cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
            '</styleSheet>'
        )

        root_rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
            '</Relationships>'
        )
        core_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            '<dc:creator>EcoDrive SQLite review exporter</dc:creator>'
            '<dc:title>EcoDrive canonical database review</dc:title>'
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created>'
            '</cp:coreProperties>'
        )
        app_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
            'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
            '<Application>EcoDrive</Application></Properties>'
        )

        try:
            with zipfile.ZipFile(
                temporary_output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
            ) as archive:
                archive.writestr("[Content_Types].xml", "".join(content_types))
                archive.writestr("_rels/.rels", root_rels)
                archive.writestr("docProps/core.xml", core_xml)
                archive.writestr("docProps/app.xml", app_xml)
                archive.writestr("xl/workbook.xml", workbook_xml)
                archive.writestr("xl/_rels/workbook.xml.rels", "".join(workbook_rels))
                archive.writestr("xl/styles.xml", styles_xml)
                for index, sheet in enumerate(sheets, start=1):
                    archive.write(sheet.path, f"xl/worksheets/sheet{index}.xml")
            os.replace(temporary_output, output_path)
        finally:
            temporary_output.unlink(missing_ok=True)
        return tuple(sheet.name for sheet in sheets)


def _allocate_sheet_names(base: str, part_count: int, used: set[str]) -> tuple[str, ...]:
    names: list[str] = []
    clean_base = re.sub(r"[\\/*?:\[\]]", "_", base)
    for part in range(1, part_count + 1):
        suffix = "" if part_count == 1 else f"_{part}"
        candidate = (clean_base[: 31 - len(suffix)] + suffix) or "sheet"
        serial = 2
        while candidate.casefold() in used:
            collision_suffix = f"_{serial}"
            candidate = clean_base[: 31 - len(collision_suffix)] + collision_suffix
            serial += 1
        used.add(candidate.casefold())
        names.append(candidate)
    return tuple(names)


def _data_sheet_base(object_name: str, object_type: str) -> str:
    if object_type == "view" and object_name in {"vde_db", "fuelcons_db"}:
        return f"{object_name}_view"
    return object_name


def _ordered_objects(connection: sqlite3.Connection) -> list[tuple[str, str]]:
    rows = connection.execute(
        "SELECT name, type FROM sqlite_schema "
        "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    by_name = {row["name"]: row["type"] for row in rows}
    ordered_names = [name for name in CORE_OBJECT_ORDER if name in by_name]
    ordered_names.extend(sorted(name for name in by_name if name not in CORE_OBJECT_ORDER))
    return [(name, by_name[name]) for name in ordered_names]


def _table_columns(connection: sqlite3.Connection, object_name: str) -> tuple[sqlite3.Row, ...]:
    return tuple(
        connection.execute(
            'SELECT cid, name, type, "notnull", dflt_value, pk '
            "FROM pragma_table_info(?) ORDER BY cid",
            (object_name,),
        ).fetchall()
    )


def _numeric_id_summary(
    connection: sqlite3.Connection, object_name: str, columns: Sequence[sqlite3.Row]
) -> tuple[str | None, int | float | None, int | float | None]:
    by_name = {column["name"]: column for column in columns}
    candidate = by_name.get("id")
    if candidate is None:
        primary_keys = sorted((column for column in columns if column["pk"]), key=lambda c: c["pk"])
        candidate = primary_keys[0] if len(primary_keys) == 1 else None
    if candidate is None:
        return None, None, None
    declared_type = (candidate["type"] or "").upper()
    if not any(token in declared_type for token in ("INT", "REAL", "FLOA", "DOUB", "NUM")):
        return None, None, None
    column = _quote_identifier(candidate["name"])
    table = _quote_identifier(object_name)
    minimum, maximum = connection.execute(
        f"SELECT MIN({column}), MAX({column}) FROM {table} "
        f"WHERE typeof({column}) IN ('integer', 'real')"
    ).fetchone()
    return candidate["name"], minimum, maximum


def _distinct_count(
    connection: sqlite3.Connection, object_name: str, column_names: set[str], column: str
) -> int | None:
    if column not in column_names:
        return None
    table = _quote_identifier(object_name)
    field = _quote_identifier(column)
    return connection.execute(f"SELECT COUNT(DISTINCT {field}) FROM {table}").fetchone()[0]


def _has_columns(connection: sqlite3.Connection, table: str, columns: set[str]) -> bool:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if not exists:
        return False
    actual = {row["name"] for row in connection.execute("SELECT name FROM pragma_table_info(?)", (table,))}
    return columns <= actual


def _semantic_review_sheets(connection: sqlite3.Connection) -> dict[str, tuple[tuple[str, ...], list[Sequence[object]]]]:
    specs: dict[str, tuple[tuple[str, ...], list[Sequence[object]]]] = {}

    carry_headers = (
        "vde_id", "model_year", "parent_vde_id", "parent_model_year", "make", "model",
        "vehicle_configuration_id", "cycle_name", "cycle_source",
        "roadload_temperature_c", "roadload_ambient_pressure_kpa",
        "test_groups", "test_vehicle_ids", "configuration_numbers", "test_numbers",
        "target_A_N", "target_B_N_per_kph", "target_C_N_per_kph2", "test_mass_kg",
        "lineage_relation", "carryover_status", "carryover_signature_sha256",
        "source_record_id", "notes",
    )
    carry_rows: list[Sequence[object]] = []
    full_carryover_columns = {
        "id", "year", "vde_id_parent", "vehicle_configuration_id", "provenance_json",
        "cycle_name", "cycle_source", "roadload_temperature_c",
        "roadload_ambient_pressure_kpa", "coast_A_N", "coast_B_N_per_kph",
        "coast_C_N_per_kph2", "test_mass_kg",
    }
    if _has_columns(connection, "vde", full_carryover_columns) and _has_columns(
        connection, "run", {"vde_id", "conditions_json"}
    ):
        carry_rows = list(connection.execute(
            """
            SELECT v.id, v.year, v.vde_id_parent, p.year, v.make, v.model,
                   v.vehicle_configuration_id,v.cycle_name,v.cycle_source,
                   v.roadload_temperature_c,v.roadload_ambient_pressure_kpa,
                   (SELECT group_concat(value,' | ') FROM (
                      SELECT DISTINCT json_extract(r.conditions_json,'$.test_group') value
                      FROM run r WHERE r.vde_id=v.id AND value IS NOT NULL ORDER BY value)),
                   (SELECT group_concat(value,' | ') FROM (
                      SELECT DISTINCT json_extract(r.conditions_json,'$.test_vehicle_id') value
                      FROM run r WHERE r.vde_id=v.id AND value IS NOT NULL ORDER BY value)),
                   (SELECT group_concat(value,' | ') FROM (
                      SELECT DISTINCT json_extract(r.conditions_json,'$.configuration_number') value
                      FROM run r WHERE r.vde_id=v.id AND value IS NOT NULL ORDER BY value)),
                   (SELECT group_concat(value,' | ') FROM (
                      SELECT DISTINCT json_extract(r.conditions_json,'$.test_number') value
                      FROM run r WHERE r.vde_id=v.id AND value IS NOT NULL ORDER BY value)),
                   v.coast_A_N,v.coast_B_N_per_kph,v.coast_C_N_per_kph2,v.test_mass_kg,
                   json_extract(v.provenance_json,'$.lineage_relation'),
                   json_extract(v.provenance_json,'$.carryover_status'),
                   json_extract(v.provenance_json,'$.carryover_signature_sha256'),
                   v.source_record_id, v.notes
            FROM vde v LEFT JOIN vde p ON p.id=v.vde_id_parent
            WHERE v.legislation='EPA'
            ORDER BY v.make,v.model,v.vehicle_configuration_id,v.year,v.id
            """
        ))
    specs["EPA_CARRYOVER_REVIEW"] = (carry_headers, carry_rows)

    run_headers = (
        "run_id", "vde_id", "model_year", "make", "model", "test_number", "adfe_test_number", "test_group",
        "test_vehicle_id", "configuration_number", "set_abc_native", "test_category",
        "test_procedure", "conflict_fields", "review_status", "carryover_status",
        "review_reason", "carryover_from_run_id",
    )
    run_rows: list[Sequence[object]] = []
    if _has_columns(connection, "run", {"run_id", "vde_id", "conditions_json", "provenance_json", "review_status"}):
        run_rows = list(connection.execute(
            """
            SELECT r.run_id,r.vde_id,v.year,v.make,v.model,
                   json_extract(r.conditions_json,'$.test_number'),
                   json_extract(r.conditions_json,'$.adfe_test_number'),
                   json_extract(r.conditions_json,'$.test_group'),
                   json_extract(r.conditions_json,'$.test_vehicle_id'),
                   json_extract(r.conditions_json,'$.configuration_number'),
                   json_extract(r.conditions_json,'$.set_abc_native'),
                   COALESCE(json_extract(r.conditions_json,'$.test_category'),
                            json_extract(r.result_details_json,'$.canonical_result.Test Category')),
                   COALESCE(r.procedure_description,r.procedure_code),
                   json_extract(r.conditions_json,'$.conflict_fields_within_test_identity'),
                   r.review_status,
                   json_extract(r.provenance_json,'$.carryover_status'),
                   json_extract(r.provenance_json,'$.run_identity_review_reason'),
                   json_extract(r.provenance_json,'$.carryover_from_run_id')
            FROM run r JOIN vde v ON v.id=r.vde_id
            WHERE r.review_status='RUN_IDENTITY_REVIEW'
            ORDER BY v.year,v.make,v.model,r.run_id
            """
        ))
    specs["RUN_IDENTITY_REVIEW"] = (run_headers, run_rows)

    vde_dup_headers = (
        "signature_count", "carryover_signature_sha256", "vde_id", "model_year", "make", "model",
        "vehicle_configuration_id", "target_A_N", "target_B_N_per_kph", "target_C_N_per_kph2",
        "test_mass_kg", "parent_vde_id", "carryover_status",
    )
    vde_dup_rows: list[Sequence[object]] = []
    if _has_columns(connection, "vde", {"id", "year", "provenance_json", "coast_A_N", "test_mass_kg"}):
        vde_dup_rows = list(connection.execute(
            """
            WITH candidates AS (
              SELECT v.*,
                     json_extract(v.provenance_json,'$.carryover_signature_sha256') signature,
                     json_extract(v.provenance_json,'$.carryover_status') carryover_status
              FROM vde v WHERE v.legislation='EPA'
            ), counted AS (
              SELECT *,COUNT(*) OVER(PARTITION BY signature) signature_count FROM candidates
              WHERE signature IS NOT NULL
            )
            SELECT signature_count,signature,id,year,make,model,vehicle_configuration_id,
                   coast_A_N,coast_B_N_per_kph,coast_C_N_per_kph2,test_mass_kg,vde_id_parent,carryover_status
            FROM counted WHERE signature_count>1
            ORDER BY signature,year,id
            """
        ))
    specs["VDE_DUPLICATE_CANDIDATES"] = (vde_dup_headers, vde_dup_rows)

    fuel_headers = (
        "signature_count", "carryover_signature_sha256", "fuelcons_id", "vde_id", "model_year",
        "comparison_basis", "cycle", "fuel_type", "fuel_l_per_100km", "gco2_per_km",
        "carryover_from_fuelcons_id", "carryover_status",
    )
    fuel_rows: list[Sequence[object]] = []
    if _has_columns(connection, "fuelcons", {"id", "vde_id", "provenance_json", "comparison_basis"}):
        fuel_rows = list(connection.execute(
            """
            WITH candidates AS (
              SELECT f.*,v.year,
                     json_extract(f.provenance_json,'$.carryover_signature_sha256') signature,
                     json_extract(f.provenance_json,'$.cycle') cycle,
                     json_extract(f.provenance_json,'$.carryover_from_fuelcons_id') parent_id,
                     json_extract(f.provenance_json,'$.carryover_status') carryover_status
              FROM fuelcons f JOIN vde v ON v.id=f.vde_id
              WHERE f.record_origin='EPA_RECONSTRUCTED'
            ), counted AS (
              SELECT *,COUNT(*) OVER(PARTITION BY signature) signature_count FROM candidates
              WHERE signature IS NOT NULL
            )
            SELECT signature_count,signature,id,vde_id,year,comparison_basis,cycle,fuel_type,
                   fuel_l_per_100km,gco2_per_km,parent_id,carryover_status
            FROM counted WHERE signature_count>1
            ORDER BY signature,year,id
            """
        ))
    specs["FUELCONS_DUPLICATE_CANDIDATES"] = (fuel_headers, fuel_rows)

    scalar_headers = (
        "vehicle_configuration_id", "program_id", "source_record_id", "raw_rated_horsepower",
        "engine_rated_power_kw", "expected_power_kw", "power_matches", "raw_cylinders_rotors",
        "engine_cylinders_rotors", "cylinders_match", "legacy_json_rated_horsepower",
        "legacy_json_cylinders_rotors",
    )
    scalar_rows: list[Sequence[object]] = []
    if _has_columns(connection, "vehicle_configuration", {
        "vehicle_configuration_id", "source_identity_json", "architecture_properties_json",
        "engine_rated_power_kw", "engine_cylinders_rotors",
    }):
        scalar_rows = list(connection.execute(
            """
            SELECT vehicle_configuration_id,program_id,source_record_id,
                   json_extract(source_identity_json,'$.raw_source_values.rated_horsepower') raw_hp,
                   engine_rated_power_kw,
                   json_extract(source_identity_json,'$.raw_source_values.rated_horsepower')*0.745699872 expected_kw,
                   CASE WHEN json_extract(source_identity_json,'$.raw_source_values.rated_horsepower') IS NULL
                        THEN engine_rated_power_kw IS NULL
                        ELSE abs(engine_rated_power_kw-(json_extract(source_identity_json,'$.raw_source_values.rated_horsepower')*0.745699872))<1e-9 END,
                   json_extract(source_identity_json,'$.raw_source_values.cylinders_rotors') raw_cyl,
                   engine_cylinders_rotors,
                   CASE WHEN json_extract(source_identity_json,'$.raw_source_values.cylinders_rotors') IS NULL
                        THEN engine_cylinders_rotors IS NULL
                        ELSE engine_cylinders_rotors=json_extract(source_identity_json,'$.raw_source_values.cylinders_rotors') END,
                   json_extract(architecture_properties_json,'$.rated_horsepower'),
                   json_extract(architecture_properties_json,'$.cylinders_rotors')
            FROM vehicle_configuration
            ORDER BY vehicle_configuration_id
            """
        ))
    specs["JSON_SCALAR_REVIEW"] = (scalar_headers, scalar_rows)
    return specs


def _csv_review_sheet(path: Path) -> tuple[tuple[str, ...], list[Sequence[object]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Review CSV is empty: {path}")
    headers = tuple(rows[0])
    numeric_columns = {
        index
        for index, name in enumerate(headers)
        if name in {"old_id", "new_id", "vde_id", "year"}
    }
    converted: list[Sequence[object]] = []
    for row in rows[1:]:
        values: list[object] = list(row)
        for index in numeric_columns:
            if index < len(values) and values[index] != "":
                values[index] = int(values[index])
        converted.append(tuple(values))
    return headers, converted


def export_database_review(
    db_path: Path | str,
    output_path: Path | str,
    *,
    vde_id_remap: Path | str | None = None,
    fuelcons_id_remap: Path | str | None = None,
) -> ExportReport:
    source = Path(db_path).resolve(strict=True)
    output = Path(output_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == output:
        raise ValueError("Output workbook must not overwrite the source SQLite database.")

    sha_before = file_sha256(source)
    file_size = source.stat().st_size
    optional_review_sheets: dict[str, tuple[tuple[str, ...], list[Sequence[object]]]] = {}
    if vde_id_remap is not None:
        optional_review_sheets["VDE_ID_REMAP"] = _csv_review_sheet(
            Path(vde_id_remap).resolve(strict=True)
        )
    if fuelcons_id_remap is not None:
        optional_review_sheets["FUELCONS_ID_REMAP"] = _csv_review_sheet(
            Path(fuelcons_id_remap).resolve(strict=True)
        )
    metadata_sheet_names = METADATA_SHEETS + tuple(optional_review_sheets)
    used_sheet_names = {name.casefold() for name in metadata_sheet_names}
    schema_rows: list[Sequence[object]] = []
    relationship_rows: list[Sequence[object]] = []
    origin_rows: list[Sequence[object]] = []
    null_rows: list[Sequence[object]] = []
    summary_object_rows: list[Sequence[object]] = []
    database_objects: list[DatabaseObject] = []
    semantic_sheets: dict[str, tuple[tuple[str, ...], list[Sequence[object]]]] = {}

    with tempfile.TemporaryDirectory(prefix="ecodrive_db_review_") as temp_name:
        writer = _XlsxWriter(Path(temp_name))
        with closing(open_database_read_only(source)) as connection:
            quick_check_rows = connection.execute("PRAGMA quick_check").fetchall()
            quick_check = "; ".join(str(row[0]) for row in quick_check_rows)
            foreign_key_issue_count = sum(
                1 for _row in connection.execute("PRAGMA foreign_key_check")
            )
            semantic_sheets = _semantic_review_sheets(connection)

            for object_name, object_type in _ordered_objects(connection):
                table = _quote_identifier(object_name)
                row_count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                columns = _table_columns(connection, object_name)
                part_count = max(1, math.ceil(row_count / MAX_DATA_ROWS_PER_SHEET))
                sheet_names = _allocate_sheet_names(
                    _data_sheet_base(object_name, object_type), part_count, used_sheet_names
                )
                database_objects.append(
                    DatabaseObject(object_name, object_type, row_count, columns, sheet_names)
                )

            data_position = len(metadata_sheet_names)
            for database_object in database_objects:
                object_name = database_object.name
                object_type = database_object.object_type
                columns = database_object.columns
                column_names = [column["name"] for column in columns]
                column_name_set = set(column_names)
                table = _quote_identifier(object_name)

                primary_keys = ", ".join(
                    column["name"]
                    for column in sorted(columns, key=lambda item: item["pk"] or 999_999)
                    if column["pk"]
                )
                numeric_id_column, minimum_id, maximum_id = _numeric_id_summary(
                    connection, object_name, columns
                )
                origin_count = _distinct_count(
                    connection, object_name, column_name_set, "record_origin"
                )
                status_count = _distinct_count(
                    connection, object_name, column_name_set, "record_status"
                )
                summary_object_rows.append(
                    (
                        object_name,
                        object_type,
                        database_object.row_count,
                        len(columns),
                        primary_keys,
                        numeric_id_column,
                        minimum_id,
                        maximum_id,
                        origin_count,
                        status_count,
                        ", ".join(database_object.sheet_names),
                    )
                )

                for column in columns:
                    schema_rows.append(
                        (
                            object_name,
                            object_type,
                            column["cid"] + 1,
                            column["name"],
                            column["type"],
                            bool(column["notnull"]),
                            column["dflt_value"],
                            column["pk"],
                        )
                    )

                if object_type == "table":
                    for foreign_key in connection.execute(
                        "SELECT id, seq, `table`, `from`, `to`, on_update, on_delete "
                        "FROM pragma_foreign_key_list(?) ORDER BY id, seq",
                        (object_name,),
                    ):
                        relationship_rows.append(
                            (
                                object_name,
                                foreign_key["from"],
                                foreign_key["table"],
                                foreign_key["to"],
                                foreign_key["on_update"],
                                foreign_key["on_delete"],
                            )
                        )

                    if "record_origin" in column_name_set:
                        if "record_status" in column_name_set:
                            origin_query = (
                                f'SELECT "record_origin", "record_status", COUNT(*) '
                                f'FROM {table} GROUP BY "record_origin", "record_status" '
                                f'ORDER BY "record_origin", "record_status"'
                            )
                            for origin, status, count in connection.execute(origin_query):
                                origin_rows.append((object_name, origin, status, count))
                        else:
                            origin_query = (
                                f'SELECT "record_origin", COUNT(*) FROM {table} '
                                f'GROUP BY "record_origin" ORDER BY "record_origin"'
                            )
                            for origin, count in connection.execute(origin_query):
                                origin_rows.append((object_name, origin, None, count))

                    for column in columns:
                        field = _quote_identifier(column["name"])
                        null_count, distinct_count = connection.execute(
                            f"SELECT COUNT(*) FILTER (WHERE {field} IS NULL), "
                            f"COUNT(DISTINCT {field}) FROM {table}"
                        ).fetchone()
                        null_rows.append(
                            (
                                object_name,
                                column["name"],
                                database_object.row_count,
                                null_count,
                                (null_count / database_object.row_count)
                                if database_object.row_count
                                else 0.0,
                                distinct_count,
                            )
                        )

                select_columns = ", ".join(_quote_identifier(name) for name in column_names)
                cursor = connection.execute(f"SELECT {select_columns} FROM {table}")
                remaining = database_object.row_count
                for sheet_name in database_object.sheet_names:
                    part_row_count = min(remaining, MAX_DATA_ROWS_PER_SHEET)
                    part_rows: Iterator[Sequence[object]] = islice(cursor, part_row_count)
                    writer.add_sheet(
                        position=data_position,
                        name=sheet_name,
                        rows=chain((tuple(column_names),), part_rows),
                    )
                    data_position += 1
                    remaining -= part_row_count

        sha_after_read = file_sha256(source)
        if sha_after_read != sha_before:
            raise RuntimeError("Source database hash changed during read-only export; workbook not written.")

        summary_headers = (
            "object_name",
            "object_type",
            "row_count",
            "column_count",
            "primary_key_columns",
            "numeric_id_column",
            "minimum_numeric_id",
            "maximum_numeric_id",
            "distinct_record_origin_count",
            "distinct_record_status_count",
            "excel_sheet_names",
        )
        database_metadata = (
            ("database_metric", "value"),
            ("db_file_path", str(source)),
            ("file_size_bytes", file_size),
            ("sha256_before_export", sha_before),
            ("sha256_after_database_read", sha_after_read),
            ("pragma_quick_check", quick_check),
            ("pragma_foreign_key_check_issue_count", foreign_key_issue_count),
            (None, None),
            summary_headers,
        )
        writer.add_sheet(
            position=0,
            name="00_SUMMARY",
            rows=chain(database_metadata, summary_object_rows),
            header_rows={1, 9},
            freeze_rows=9,
            filter_header_row=9,
        )
        writer.add_sheet(
            position=1,
            name="01_SCHEMA",
            rows=chain(
                (("object", "object_type", "column_order", "column_name", "SQLite_type", "not_null", "default_value", "primary_key"),),
                schema_rows,
            ),
        )
        writer.add_sheet(
            position=2,
            name="02_RELATIONSHIPS",
            rows=chain(
                (("child_table", "child_column", "parent_table", "parent_column", "on_update", "on_delete"),),
                relationship_rows,
            ),
        )
        writer.add_sheet(
            position=3,
            name="03_ORIGIN_COUNTS",
            rows=chain(
                (("entity_table", "record_origin", "record_status", "row_count"),),
                origin_rows,
            ),
        )
        writer.add_sheet(
            position=4,
            name="04_NULL_COVERAGE",
            rows=chain(
                (("table", "column", "row_count", "null_count", "null_percent", "distinct_non_null_count"),),
                null_rows,
            ),
            column_styles={5: 2},
        )
        for position, name in enumerate(METADATA_SHEETS[5:], start=5):
            headers, rows = semantic_sheets[name]
            writer.add_sheet(position=position, name=name, rows=chain((headers,), rows))
        for position, (name, (headers, rows)) in enumerate(
            optional_review_sheets.items(), start=len(METADATA_SHEETS)
        ):
            writer.add_sheet(position=position, name=name, rows=chain((headers,), rows))
        sheet_names = writer.save(output)

    sha_after = file_sha256(source)
    if sha_after != sha_before:
        output.unlink(missing_ok=True)
        raise RuntimeError("Source database hash changed during export; generated workbook was removed.")

    known = set(CORE_OBJECT_ORDER)
    additional_objects = tuple(obj.name for obj in database_objects if obj.name not in known)
    return ExportReport(
        workbook_path=output,
        sheet_names=sheet_names,
        source_sha256_before=sha_before,
        source_sha256_after=sha_after,
        row_counts={obj.name: obj.row_count for obj in database_objects},
        additional_objects=additional_objects,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a SQLite database to a read-only Excel review workbook."
    )
    parser.add_argument("--db", required=True, type=Path, help="Source SQLite database path.")
    parser.add_argument("--output", required=True, type=Path, help="Destination .xlsx path.")
    parser.add_argument("--vde-id-remap", type=Path, help="Optional VDE old-to-new ID QA CSV.")
    parser.add_argument("--fuelcons-id-remap", type=Path, help="Optional FuelCons old-to-new ID QA CSV.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = export_database_review(
        args.db,
        args.output,
        vde_id_remap=args.vde_id_remap,
        fuelcons_id_remap=args.fuelcons_id_remap,
    )
    print(f"Workbook: {report.workbook_path}")
    print(f"Sheets ({len(report.sheet_names)}): {', '.join(report.sheet_names)}")
    print(f"Source SHA256 before: {report.source_sha256_before}")
    print(f"Source SHA256 after:  {report.source_sha256_after}")
    print("Row counts:")
    for object_name, row_count in report.row_counts.items():
        print(f"  {object_name}: {row_count}")
    if report.additional_objects:
        print(f"Additional tables/views: {', '.join(report.additional_objects)}")
    else:
        print("Additional tables/views: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
