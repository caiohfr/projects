from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from src.vde_core.vde_request_parser import (
    parse_vde_request_workbook,
    validate_vde_request_workbook,
)


def _excel_column_name(index: int) -> str:
    value = index + 1
    letters = []
    while value:
        value, remainder = divmod(value - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return "".join(reversed(letters))


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _cell_xml(row_index: int, column_index: int, value) -> str:
    ref = f"{_excel_column_name(column_index)}{row_index + 1}"
    if value is None:
        return ""
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"><v>{"1" if value else "0"}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}"><v>{value}</v></c>'
    return f'<c r="{ref}" t="inlineStr"><is><t>{_xml_escape(str(value))}</t></is></c>'


def _sheet_xml(rows: list[list[object]]) -> str:
    row_xml = []
    for row_index, row in enumerate(rows):
        cells = "".join(_cell_xml(row_index, col_index, value) for col_index, value in enumerate(row) if value is not None)
        row_xml.append(f'<row r="{row_index + 1}">{cells}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(row_xml)}</sheetData>"
        "</worksheet>"
    )


def _write_xlsx(path: Path, sheets: dict[str, list[list[object]]]) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            + "".join(
                f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                for index in range(1, len(sheets) + 1)
            )
            + "</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            "<sheets>"
            + "".join(
                f'<sheet name="{_xml_escape(name)}" sheetId="{index}" r:id="rId{index}"/>'
                for index, name in enumerate(sheets.keys(), start=1)
            )
            + "</sheets></workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(
                f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>'
                for index in range(1, len(sheets) + 1)
            )
            + "</Relationships>",
        )
        for index, rows in enumerate(sheets.values(), start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _sheet_xml(rows))


BASE_FIELD_ROWS = [
    ("Scenario / Context", "Name / Description", "notes", "-", "scenario", "metadata"),
    ("Scenario / Context", "Walk From", "walk_from", "-", "scenario", "routing"),
    ("Proposal Matrix", "Mass proposal type", "mass_proposal_type", "-", "mass", "proposal_type"),
    ("Mass", "Curb / Base Mass", "mass_kg", "kg", "mass", "numeric"),
    ("Proposal Matrix", "Aero proposal type", "aero_proposal_type", "-", "aero", "proposal_type"),
    ("Aero", "CdA", "cda_m2", "m²", "aero", "numeric"),
    ("Proposal Matrix", "Transmission proposal type", "transmission_proposal_type", "-", "transmission", "proposal_type"),
    ("Transmission", "Transmission A", "trans_A_coef_N", "N", "transmission", "numeric"),
]


def _build_workbook_data(
    *,
    requested_headers: list[str] | None = None,
    request_value_map: dict[tuple[str, str], dict[str, object]] | None = None,
    baseline_printed_map: dict[str, object] | None = None,
    baseline_correction_map: dict[str, object] | None = None,
    include_lists: bool = True,
    include_rules: bool = True,
    include_request: bool = True,
    include_field_map: bool = True,
    request_header_override: list[object] | None = None,
    duplicate_field_map_row: tuple[str, str, str, str, str, str] | None = None,
    extra_request_rows: list[list[object]] | None = None,
) -> dict[str, list[list[object]]]:
    requested_headers = requested_headers or ["Requested #1", "Requested #2", "Requested #3"]
    request_value_map = request_value_map or {}
    baseline_printed_map = baseline_printed_map or {}
    baseline_correction_map = baseline_correction_map or {}
    extra_request_rows = extra_request_rows or []

    sheets: dict[str, list[list[object]]] = {}

    if include_request:
        request_rows: list[list[object]] = [
            ["EcoDrive VDE PPE Request Input - FINAL v0.1"],
            [],
            request_header_override
            or ["Section", "Field / Parameter", "Unit", "Baseline / Printed", "Baseline Correction", *requested_headers],
        ]
        for section, label, field_key, unit, _domain, _role in BASE_FIELD_ROWS:
            requested_values = request_value_map.get((section, label), {})
            request_rows.append(
                [
                    section,
                    label,
                    unit,
                    baseline_printed_map.get(field_key),
                    baseline_correction_map.get(field_key),
                    *[requested_values.get(header) for header in requested_headers],
                ]
            )
        request_rows.extend(extra_request_rows)
        sheets["REQUEST"] = request_rows

    if include_field_map:
        field_map_rows: list[list[object]] = [
            ["section", "field_label", "field_key", "unit", "domain", "role", "compact_or_advanced", "validation_list", "notes"],
        ]
        for section, label, field_key, unit, domain, role in BASE_FIELD_ROWS:
            field_map_rows.append([section, label, field_key, unit, domain, role, "compact", None, None])
        if duplicate_field_map_row is not None:
            field_map_rows.append([*duplicate_field_map_row, "compact", None, None])
        sheets["FIELD_MAP"] = field_map_rows

    if include_lists:
        sheets["LISTS"] = [
            ["MassProposalTypes", "AeroProposalTypes", "TransmissionProposalTypes", "WalkFrom"],
            ["Inherit", "Inherit", "Inherit", "Baseline"],
            ["Custom test mass", "Absolute CdA", "Absolute ABC", "Requested #1"],
        ]

    if include_rules:
        sheets["RULES"] = [
            ["Rule", "Frozen behavior"],
            ["Blank cells", "Blank requested cells mean inherit/no request."],
        ]

    return sheets


class TestVdeRequestParser(unittest.TestCase):
    def _with_workbook(self, sheets: dict[str, list[list[object]]], callback):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "request.xlsx"
            _write_xlsx(path, sheets)
            return callback(path)

    def test_valid_template(self):
        sheets = _build_workbook_data(
            request_value_map={
                ("Proposal Matrix", "Mass proposal type"): {"Requested #1": "Custom test mass"},
                ("Mass", "Curb / Base Mass"): {"Requested #1": 1800},
            }
        )

        def _run(path: Path):
            validation = validate_vde_request_workbook(path)
            self.assertTrue(validation["ok"])

        self._with_workbook(sheets, _run)

    def test_missing_request_sheet(self):
        sheets = _build_workbook_data(include_request=False)

        def _run(path: Path):
            validation = validate_vde_request_workbook(path)
            self.assertFalse(validation["ok"])
            self.assertTrue(any(issue["code"] == "missing_request_sheet" for issue in validation["errors"]))

        self._with_workbook(sheets, _run)

    def test_missing_field_map_sheet(self):
        sheets = _build_workbook_data(include_field_map=False)

        def _run(path: Path):
            validation = validate_vde_request_workbook(path)
            self.assertFalse(validation["ok"])
            self.assertTrue(any(issue["code"] == "missing_field_map_sheet" for issue in validation["errors"]))

        self._with_workbook(sheets, _run)

    def test_missing_essential_column(self):
        sheets = _build_workbook_data(
            request_header_override=["Section", "Field / Parameter", "Unit", "Baseline / Printed", "Requested #1"]
        )

        def _run(path: Path):
            validation = validate_vde_request_workbook(path)
            self.assertFalse(validation["ok"])
            self.assertTrue(any(issue["code"] == "missing_request_header" for issue in validation["errors"]))

        self._with_workbook(sheets, _run)

    def test_detects_dynamic_requested_columns_and_ignores_empty_gap(self):
        sheets = _build_workbook_data(
            requested_headers=["Requested #1", "Requested #2", "Requested #3"],
            request_value_map={
                ("Proposal Matrix", "Mass proposal type"): {"Requested #1": "Custom test mass", "Requested #3": "Custom test mass"},
                ("Mass", "Curb / Base Mass"): {"Requested #1": 1750, "Requested #3": 1825},
            },
        )

        def _run(path: Path):
            draft = parse_vde_request_workbook(path)
            self.assertEqual(len(draft["proposals"]), 2)
            self.assertEqual(draft["proposals"][0]["source_column"], "Requested #1")
            self.assertEqual(draft["proposals"][0]["display_index"], 1)
            self.assertEqual(draft["proposals"][1]["source_column"], "Requested #3")
            self.assertEqual(draft["proposals"][1]["display_index"], 2)

        self._with_workbook(sheets, _run)

    def test_preserves_source_column_and_source_index(self):
        sheets = _build_workbook_data(
            requested_headers=["Requested #5"],
            request_value_map={
                ("Proposal Matrix", "Aero proposal type"): {"Requested #5": "Absolute CdA"},
                ("Aero", "CdA"): {"Requested #5": 1.2},
            },
        )

        def _run(path: Path):
            draft = parse_vde_request_workbook(path)
            proposal = draft["proposals"][0]
            self.assertEqual(proposal["source_column"], "Requested #5")
            self.assertEqual(proposal["source_index"], 5)
            self.assertEqual(proposal["display_index"], 1)

        self._with_workbook(sheets, _run)

    def test_zero_is_preserved_as_explicit_value(self):
        sheets = _build_workbook_data(
            request_value_map={
                ("Proposal Matrix", "Aero proposal type"): {"Requested #1": "Delta CdA"},
                ("Aero", "CdA"): {"Requested #1": 0},
            }
        )

        def _run(path: Path):
            draft = parse_vde_request_workbook(path)
            proposal = draft["proposals"][0]
            self.assertIn("cda_m2", proposal["raw_values"])
            self.assertEqual(proposal["raw_values"]["cda_m2"], 0)

        self._with_workbook(sheets, _run)

    def test_baseline_correction_precedes_printed(self):
        sheets = _build_workbook_data(
            baseline_printed_map={"mass_kg": 1800},
            baseline_correction_map={"mass_kg": 1850},
        )

        def _run(path: Path):
            draft = parse_vde_request_workbook(path)
            self.assertEqual(draft["baseline_printed"]["mass_kg"], 1800)
            self.assertEqual(draft["baseline_corrections"]["mass_kg"], 1850)
            self.assertEqual(draft["effective_baseline"]["mass_kg"], 1850)

        self._with_workbook(sheets, _run)

    def test_known_proposal_type_is_normalized(self):
        sheets = _build_workbook_data(
            request_value_map={
                ("Proposal Matrix", "Transmission proposal type"): {"Requested #1": "Absolute ABC"},
                ("Transmission", "Transmission A"): {"Requested #1": 12},
            }
        )

        def _run(path: Path):
            draft = parse_vde_request_workbook(path)
            transmission = draft["proposals"][0]["domain_requests"]["transmission"]
            self.assertEqual(transmission["proposal_type"], "UPDATE_TRANS_DRAG_ABC")
            self.assertEqual(transmission["selection_mode"], "Absolute ABC")
            self.assertNotIn("change_mode", transmission["proposal_details_seed"])

        self._with_workbook(sheets, _run)

    def test_unknown_proposal_type_creates_issue(self):
        sheets = _build_workbook_data(
            request_value_map={
                ("Proposal Matrix", "Transmission proposal type"): {"Requested #1": "Weird Mode"},
                ("Transmission", "Transmission A"): {"Requested #1": 12},
            }
        )

        def _run(path: Path):
            draft = parse_vde_request_workbook(path)
            transmission = draft["proposals"][0]["domain_requests"]["transmission"]
            self.assertTrue(any(issue["code"] == "unknown_proposal_type" for issue in transmission["issues"]))

        self._with_workbook(sheets, _run)

    def test_values_without_proposal_type_create_review_issue(self):
        sheets = _build_workbook_data(
            request_value_map={
                ("Aero", "CdA"): {"Requested #1": 1.25},
            }
        )

        def _run(path: Path):
            draft = parse_vde_request_workbook(path)
            aero = draft["proposals"][0]["domain_requests"]["aero"]
            self.assertTrue(any(issue["code"] == "missing_proposal_type" for issue in aero["issues"]))

        self._with_workbook(sheets, _run)

    def test_walk_from_baseline(self):
        sheets = _build_workbook_data(
            request_value_map={
                ("Scenario / Context", "Walk From"): {"Requested #1": "Baseline"},
                ("Proposal Matrix", "Mass proposal type"): {"Requested #1": "Custom test mass"},
            }
        )

        def _run(path: Path):
            draft = parse_vde_request_workbook(path)
            self.assertEqual(draft["proposals"][0]["walk_from"]["kind"], "baseline")

        self._with_workbook(sheets, _run)

    def test_walk_from_previous_proposal(self):
        sheets = _build_workbook_data(
            requested_headers=["Requested #1", "Requested #2", "Requested #3"],
            request_value_map={
                ("Proposal Matrix", "Mass proposal type"): {"Requested #1": "Custom test mass", "Requested #3": "Custom test mass"},
                ("Scenario / Context", "Walk From"): {"Requested #3": "Requested #1"},
                ("Mass", "Curb / Base Mass"): {"Requested #1": 1700, "Requested #3": 1750},
            },
        )

        def _run(path: Path):
            draft = parse_vde_request_workbook(path)
            proposal = draft["proposals"][1]
            self.assertEqual(proposal["walk_from"]["kind"], "proposal")
            self.assertEqual(proposal["walk_from"]["source_column"], "Requested #1")
            self.assertEqual(proposal["walk_from"]["proposal_id"], "proposal_req_1")

        self._with_workbook(sheets, _run)

    def test_walk_from_future_is_blocking(self):
        sheets = _build_workbook_data(
            requested_headers=["Requested #1", "Requested #2"],
            request_value_map={
                ("Scenario / Context", "Walk From"): {"Requested #1": "Requested #2"},
                ("Proposal Matrix", "Mass proposal type"): {"Requested #1": "Custom test mass", "Requested #2": "Custom test mass"},
            },
        )

        def _run(path: Path):
            draft = parse_vde_request_workbook(path)
            proposal = draft["proposals"][0]
            self.assertTrue(any(issue["code"] == "future_walk_from" for issue in proposal["issues"]))

        self._with_workbook(sheets, _run)

    def test_walk_from_empty_gap_is_blocking(self):
        sheets = _build_workbook_data(
            requested_headers=["Requested #1", "Requested #2", "Requested #3"],
            request_value_map={
                ("Scenario / Context", "Walk From"): {"Requested #3": "Requested #2"},
                ("Proposal Matrix", "Mass proposal type"): {"Requested #1": "Custom test mass", "Requested #3": "Custom test mass"},
            },
        )

        def _run(path: Path):
            draft = parse_vde_request_workbook(path)
            proposal = draft["proposals"][1]
            self.assertTrue(any(issue["code"] == "empty_walk_from_target" for issue in proposal["issues"]))

        self._with_workbook(sheets, _run)

    def test_duplicate_field_map_is_detected(self):
        sheets = _build_workbook_data(
            duplicate_field_map_row=("Mass", "Curb / Base Mass", "mass_kg", "kg", "mass", "numeric")
        )

        def _run(path: Path):
            validation = validate_vde_request_workbook(path)
            self.assertFalse(validation["ok"])
            self.assertTrue(any(issue["code"] == "duplicate_field_key" for issue in validation["errors"]))

        self._with_workbook(sheets, _run)

    def test_request_row_without_field_map_is_detected(self):
        sheets = _build_workbook_data(
            extra_request_rows=[
                ["Mystery", "Unmapped Field", "-", None, None, 10, None, None],
            ]
        )

        def _run(path: Path):
            validation = validate_vde_request_workbook(path)
            self.assertFalse(validation["ok"])
            self.assertTrue(any(issue["code"] == "request_row_not_mapped" for issue in validation["errors"]))

        self._with_workbook(sheets, _run)

    def test_result_is_json_serializable(self):
        sheets = _build_workbook_data(
            request_value_map={
                ("Proposal Matrix", "Mass proposal type"): {"Requested #1": "Custom test mass"},
                ("Mass", "Curb / Base Mass"): {"Requested #1": 1800},
            }
        )

        def _run(path: Path):
            draft = parse_vde_request_workbook(path)
            json.dumps(draft)

        self._with_workbook(sheets, _run)

    def test_real_template_in_repo_validates_when_available(self):
        template_path = Path(__file__).resolve().parents[1] / "docs" / "templates" / "EcoDrive_VDE_PPE_Request_Input_template_v01.xlsx"
        if not template_path.exists():
            self.skipTest("Stable repo template not available.")
        validation = validate_vde_request_workbook(template_path)
        self.assertTrue(validation["ok"])


if __name__ == "__main__":
    unittest.main()
