from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pandas as pd

from src.vde_core.vde_request_adapter import build_v21_workbook_state_from_request_draft
from src.vde_core.vde_request_parser import parse_vde_request_workbook
from src.vde_core.vde_request_preview import build_request_resolution_fingerprint
from src.vde_core.vde_request_report import (
    VDE_REQUEST_REPORT_VERSION,
    build_request_equivalent_draft_from_state,
    build_vde_request_report_filename,
    build_vde_request_report_model,
    generate_vde_request_report_xlsx,
)
from src.vde_core.vde_request_resolver import resolve_vde_request
from src.vde_core.vde_request_save import (
    SAVE_MODE_SELECTED,
    build_vde_request_save_plan,
    execute_vde_request_save_plan,
)


XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _proposal_result(
    proposal_id: str,
    *,
    display_index: int,
    status: str = "OK",
    total_mj_per_km: float | None = 1.25,
    net_mj_per_km: float | None = 1.10,
    walk_from: str = "Baseline",
    source_column: str | None = None,
    issues: list[dict] | None = None,
    component_actions: list[dict] | None = None,
    domain_results: dict | None = None,
    resolved_snapshot: dict | None = None,
    source_snapshot: dict | None = None,
    abc_total: dict | None = None,
) -> dict:
    return {
        "proposal_id": proposal_id,
        "display_index": display_index,
        "source_column": source_column or f"Requested #{display_index}",
        "walk_from": {"column_id": "baseline", "label": walk_from},
        "source_snapshot": deepcopy(
            source_snapshot
            or {
                "mass_kg": 1600.0,
                "test_mass_kg": 1736.0,
                "CdA": 0.62,
                "cycle_name": "FTP75",
            }
        ),
        "requested_snapshot": {},
        "resolved_snapshot": deepcopy(
            resolved_snapshot
            or {
                "legislation": "EPA",
                "category": "MIDSIZE",
                "make": "FORD",
                "model": "TEST",
                "year": 2026,
                "cycle_name": "FTP75",
                "mass_kg": 1600.0,
                "test_mass_kg": 1736.0,
                "CdA": 0.63,
                "front_tire_id": 7,
                "rear_tire_id": 7,
                "tire_A_final": 48.0,
                "tire_B_final": 0.009,
                "tire_C_final": 0.001,
                "tire_calc_source": "tire_service",
                "resolved_mass_setup": {
                    "mass_kg": 1600.0,
                    "test_mass_kg": 1736.0,
                    "test_mass_basis": "EPA",
                    "inertia_class": 60,
                    "resolved_mass_used_kg": 1736.0,
                    "mass_rule_status": "OK",
                    "mass_rule_notes": "Resolved",
                },
            }
        ),
        "domain_results": deepcopy(
            domain_results
            or {
                "mass": {"proposal_type": "CUSTOM_MASS", "status": status, "source": "Baseline"},
                "tire": {"proposal_type": "TIRE_DB_LOOKUP", "status": "OK", "source": "Baseline"},
            }
        ),
        "abc_total": deepcopy(abc_total or {"A": 121.0, "B": 0.021, "C": 0.0105}),
        "abc_net": {"A": 111.0, "B": 0.018, "C": 0.0095} if net_mj_per_km is not None else None,
        "vde_results": {
            "total": {"mj_per_km": total_mj_per_km} if total_mj_per_km is not None else None,
            "net": {"mj_per_km": net_mj_per_km} if net_mj_per_km is not None else None,
        },
        "status": status,
        "issues": deepcopy(issues or []),
        "component_actions": deepcopy(component_actions or []),
        "preview_summary": {"warnings": []},
    }


def _request_draft() -> dict:
    return {
        "schema_version": "0.1",
        "template_version": "0.1",
        "source": {
            "filename": "EcoDrive_VDE_Request.xlsx",
            "source_type": "Excel",
        },
        "baseline_printed": {
            "selected_baseline_vde_id": 5038,
            "mass_kg": 1600.0,
            "cda_m2": 0.62,
        },
        "baseline_corrections": {
            "mass_kg": 1650.0,
        },
        "effective_baseline": {
            "selected_baseline_vde_id": 5038,
            "legislation": "EPA",
            "make": "FORD",
            "model": "TEST",
            "year": 2026,
            "cycle_name": "FTP75",
            "mass_kg": 1650.0,
            "cda_m2": 0.62,
        },
        "proposals": [
            {
                "proposal_id": "proposal_req_1",
                "display_index": 1,
                "source_column": "Requested #1",
                "name": "Mass and Tire change",
                "walk_from": {"kind": "baseline", "proposal_id": None, "source_column": "Baseline"},
                "raw_values": {"notes": "Mass and Tire change", "mass_kg": 1735, "walk_from": "Baseline"},
                "normalized_values": {"notes": "Mass and Tire change", "mass_kg": 1735, "walk_from": "Baseline"},
                "domain_requests": {
                    "mass": {
                        "domain": "mass",
                        "proposal_type": "CUSTOM_MASS",
                        "selection_mode": "Custom test mass",
                        "raw_proposal_type": "Custom test mass",
                        "raw_values": {"mass_kg": 1735},
                        "normalized_values": {"mass_kg": 1735},
                        "issues": [],
                    }
                },
                "issues": [],
            }
        ],
        "issues": [],
        "original_request": {
            "request_rows": [
                {
                    "section": "Scenario / Context",
                    "field_label": "Name / Description",
                    "field_key": "notes",
                    "unit": "-",
                    "baseline_printed": None,
                    "baseline_correction": None,
                    "requested_values": {"Requested #1": "Mass and Tire change"},
                },
                {
                    "section": "Scenario / Context",
                    "field_label": "Walk From",
                    "field_key": "walk_from",
                    "unit": "-",
                    "baseline_printed": None,
                    "baseline_correction": None,
                    "requested_values": {"Requested #1": "Baseline"},
                },
                {
                    "section": "Mass",
                    "field_label": "Curb / Base Mass",
                    "field_key": "mass_kg",
                    "unit": "kg",
                    "baseline_printed": 1600.0,
                    "baseline_correction": 1650.0,
                    "requested_values": {"Requested #1": 1735},
                },
            ],
            "requested_columns": [
                {"source_column": "Requested #1", "source_index": 1, "active": True},
                {"source_column": "Requested #2", "source_index": 2, "active": False},
            ],
        },
    }


def _resolution_result(proposals: list[dict] | None = None) -> dict:
    return {
        "baseline": {
            "printed": {
                "selected_baseline_vde_id": 5038,
                "mass_kg": 1600.0,
                "A": 120.0,
                "B": 0.02,
                "C": 0.01,
                "cda_m2": 0.62,
            },
            "correction": {"mass_kg": 1650.0},
            "effective": {
                "selected_baseline_vde_id": 5038,
                "legislation": "EPA",
                "category": "MIDSIZE",
                "make": "FORD",
                "model": "TEST",
                "year": 2026,
                "cycle_name": "FTP75",
                "mass_kg": 1650.0,
                "test_mass_kg": 1736.0,
                "A": 121.0,
                "B": 0.02,
                "C": 0.01,
                "cda_m2": 0.62,
            },
            "corrected_fields": ["mass_kg"],
        },
        "resolved_columns": {
            "baseline": {
                "selected_baseline_vde_id": 5038,
                "legislation": "EPA",
                "category": "MIDSIZE",
                "make": "FORD",
                "model": "TEST",
                "year": 2026,
                "cycle_name": "FTP75",
                "mass_kg": 1650.0,
                "test_mass_kg": 1736.0,
                "CdA": 0.62,
                "initial_abc_total": {"A": 121.0, "B": 0.02, "C": 0.01},
            }
        },
        "proposal_results": deepcopy(proposals or [_proposal_result("proposal_req_1", display_index=1)]),
        "status": "Review",
        "issues": [{"severity": "review", "code": "manual_reference_override", "message": "Manual override used."}],
    }


def _save_result(status: str = "success") -> dict:
    return {
        "operation_id": "saveop_deadbeef1234",
        "status": status,
        "executed_at": "2026-07-11T12:00:00+00:00",
        "saved_proposals": [{"proposal_id": "proposal_req_1", "vde_row_id": 6101, "status": "saved", "name": "Mass and Tire change"}],
        "skipped_proposals": [{"proposal_id": "proposal_req_2", "reason": "not_selected", "status": "Review"}] if status != "success" else [],
        "baseline_updates": [{"baseline_id": 5038, "updated_fields": ["mass_kg"], "status": "updated"}],
        "component_results": [{"proposal_id": "proposal_req_1", "domain": "transmission", "status": "created", "component_id": "TRANS-USER-ABC123"}],
        "issues": [{"code": "partial_component_failure", "severity": "review", "message": "One component failed."}] if status == "partial" else [],
    }


def _sheet_names_from_bytes(payload: bytes) -> list[str]:
    with ZipFile(io.BytesIO(payload)) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        return [sheet.attrib.get("name") for sheet in workbook.findall(f".//{XML_NS}sheet")]


def _report_sheet_xml(payload: bytes, name: str) -> bytes:
    with ZipFile(io.BytesIO(payload)) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        for sheet in workbook.findall(f".//{XML_NS}sheet"):
            if sheet.attrib.get("name") == name:
                rel_id = sheet.attrib.get(f"{REL_NS}id")
                target = rel_map[rel_id]
                return archive.read(f"xl/{target}")
    raise KeyError(name)


def _sheet_text(payload: bytes, name: str) -> str:
    root = ET.fromstring(_report_sheet_xml(payload, name))
    parts = []
    for cell in root.findall(f".//{XML_NS}c"):
        inline = cell.find(f"{XML_NS}is")
        value = cell.find(f"{XML_NS}v")
        if inline is not None:
            parts.extend(node.text or "" for node in inline.iter(f"{XML_NS}t"))
        elif value is not None and value.text is not None:
            parts.append(value.text)
    return "\n".join(parts)


def _has_autofilter(payload: bytes, name: str) -> bool:
    root = ET.fromstring(_report_sheet_xml(payload, name))
    return root.find(f".//{XML_NS}autoFilter") is not None


def _has_freeze_pane(payload: bytes, name: str) -> bool:
    root = ET.fromstring(_report_sheet_xml(payload, name))
    pane = root.find(f".//{XML_NS}pane")
    return pane is not None and pane.attrib.get("state") == "frozen"


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
    ("Scenario / Context", "Baseline VDE ID", "selected_baseline_vde_id", "-", "scenario", "metadata"),
    ("Proposal Matrix", "Mass proposal type", "mass_proposal_type", "-", "mass", "proposal_type"),
    ("Mass", "Curb / Base Mass", "mass_kg", "kg", "mass", "numeric"),
]


def _build_request_workbook() -> dict[str, list[list[object]]]:
    request_rows = [
        ["EcoDrive VDE PPE Request Input - FINAL v0.1"],
        [],
        ["Section", "Field / Parameter", "Unit", "Baseline / Printed", "Baseline Correction", "Requested #1"],
    ]
    field_map_rows = [
        ["section", "field_label", "field_key", "unit", "domain", "role", "compact_or_advanced", "validation_list", "notes"],
    ]
    request_value_map = {
        ("Scenario / Context", "Name / Description"): "Sprint5B e2e",
        ("Scenario / Context", "Walk From"): "Baseline",
        ("Scenario / Context", "Baseline VDE ID"): None,
        ("Proposal Matrix", "Mass proposal type"): "Custom test mass",
        ("Mass", "Curb / Base Mass"): 1800,
    }
    baseline_printed_map = {"mass_kg": 1600, "selected_baseline_vde_id": 5038}
    baseline_correction_map = {"mass_kg": 1650}
    for section, label, field_key, unit, domain, role in BASE_FIELD_ROWS:
        request_rows.append(
            [
                section,
                label,
                unit,
                baseline_printed_map.get(field_key),
                baseline_correction_map.get(field_key),
                request_value_map.get((section, label)),
            ]
        )
        field_map_rows.append([section, label, field_key, unit, domain, role, "compact", None, None])
    return {
        "REQUEST": request_rows,
        "FIELD_MAP": field_map_rows,
        "LISTS": [["MassProposalTypes"], ["Inherit"], ["Custom test mass"]],
        "RULES": [["Rule"], ["Blank cells mean inherit."]],
    }


def _baseline_context() -> dict:
    return {
        "cycle_df": pd.DataFrame({"t": [0.0, 1.0, 2.0], "v": [0.0, 8.0, 10.0]}),
        "legislation": "EPA",
        "category": "MIDSIZE",
        "make": "FORD",
        "model": "TEST",
        "year": 2026,
        "cycle_name": "FTP75",
        "mass_kg": 1600.0,
        "test_mass_kg": 1736.0,
        "weight_dist_fr_pct": 55.0,
        "CdA": 0.62,
        "A": 120.0,
        "B": 0.02,
        "C": 0.01,
    }


class VdeRequestReportTests(unittest.TestCase):
    def test_builds_draft_report_model(self):
        model = build_vde_request_report_model(_request_draft(), _resolution_result(), None)

        self.assertEqual(model["report_version"], VDE_REQUEST_REPORT_VERSION)
        self.assertEqual(model["report_state"], "Draft")
        self.assertEqual(model["summary_counts"]["Total proposals"], 1)
        self.assertTrue(any(row["Scenario"] == "Baseline" for row in model["summary_rows"]))
        self.assertTrue(any(row["Source Column"] == "Requested #1" for row in model["request_rows"]))

    def test_builds_saved_and_partial_report_model(self):
        saved_model = build_vde_request_report_model(_request_draft(), _resolution_result(), _save_result("success"))
        partial_model = build_vde_request_report_model(_request_draft(), _resolution_result(), _save_result("partial"))

        self.assertEqual(saved_model["report_state"], "Saved")
        self.assertEqual(partial_model["report_state"], "Partial")
        self.assertEqual(saved_model["metadata"]["save_operation_id"], "saveop_deadbeef1234")
        self.assertEqual(saved_model["summary_rows"][1]["VDE DB Row ID"], 6101)

    def test_manual_state_can_be_converted_to_equivalent_draft(self):
        state = build_v21_workbook_state_from_request_draft(_request_draft(), {"rows": []})
        manual = build_request_equivalent_draft_from_state(state)

        self.assertEqual(manual["source"]["source_type"], "UI")
        self.assertEqual(manual["proposals"][0]["proposal_id"], "proposal_req_1")
        self.assertIn("walk_from", manual["proposals"][0]["raw_values"])

    def test_filename_is_safe_and_stateful(self):
        draft_name = build_vde_request_report_filename(build_vde_request_report_model(_request_draft(), _resolution_result()))
        saved_name = build_vde_request_report_filename(build_vde_request_report_model(_request_draft(), _resolution_result(), _save_result()))

        self.assertTrue(draft_name.startswith("EcoDrive_VDE_Request_DRAFT_"))
        self.assertIn("SAVED", saved_name)
        self.assertNotIn(":", saved_name)

    def test_report_model_is_json_serializable_and_preserves_zero(self):
        proposal = _proposal_result(
            "proposal_req_1",
            display_index=1,
            resolved_snapshot={"mass_kg": 0.0, "CdA": 0.0, "resolved_mass_setup": {"test_mass_kg": 0.0}},
            source_snapshot={"mass_kg": 0.0, "CdA": 0.0},
            abc_total={"A": 0.0, "B": 0.0, "C": 0.0},
        )
        draft = _request_draft()
        draft["baseline_printed"]["mass_kg"] = 0.0
        draft["baseline_corrections"]["mass_kg"] = None
        draft["original_request"]["request_rows"][2]["baseline_printed"] = 0.0
        draft["original_request"]["request_rows"][2]["baseline_correction"] = None
        draft["original_request"]["request_rows"][2]["requested_values"]["Requested #1"] = 0.0
        model = build_vde_request_report_model(draft, _resolution_result([proposal]))

        json.dumps(model, default=str)
        request_mass_row = next(row for row in model["request_rows"] if row["Field Key"] == "mass_kg")
        self.assertEqual(request_mass_row["Baseline Printed"], 0.0)
        self.assertEqual(request_mass_row["Requested Original"], 0.0)

    def test_report_model_preserves_requested_tire_target_rrc_audit(self):
        proposal = _proposal_result(
            "proposal_req_1",
            display_index=1,
            domain_results={
                "tire": {
                    "proposal_type": "TIRE_TARGET_RRC",
                    "status": "OK",
                    "source": "Baseline",
                    "requested_values": {
                        "target_rrc_N_per_kN": 9.0,
                        "front_pressure_psi": 36.0,
                        "rear_pressure_psi": 36.0,
                        "tire_load_mass_basis": "TEST_MASS",
                    },
                    "resolved_values": {
                        "resolved_rrc_N_per_kN": 9.0,
                        "adjustment_method": "Direct target RRC",
                    },
                }
            },
            resolved_snapshot={
                "rrc_N_per_kN": 9.0,
                "resolved_mass_setup": {"test_mass_kg": 1736.0},
            },
        )

        model = build_vde_request_report_model(_request_draft(), _resolution_result([proposal]), None)
        tire_domain = model["resolution_result"]["proposal_results"][0]["domain_results"]["tire"]

        self.assertEqual(tire_domain["requested_values"]["target_rrc_N_per_kN"], 9.0)
        self.assertEqual(tire_domain["requested_values"]["front_pressure_psi"], 36.0)
        self.assertEqual(tire_domain["resolved_values"]["resolved_rrc_N_per_kN"], 9.0)

    def test_generates_xlsx_bytes_with_required_tabs(self):
        payload = generate_vde_request_report_xlsx(build_vde_request_report_model(_request_draft(), _resolution_result(), _save_result()))
        names = _sheet_names_from_bytes(payload)

        self.assertEqual(names, ["SUMMARY", "REQUEST", "RESULTS", "COMPONENTS", "VALIDATION"])
        self.assertTrue(_has_freeze_pane(payload, "SUMMARY"))
        self.assertTrue(_has_autofilter(payload, "SUMMARY"))
        self.assertIn("EcoDrive VDE Request Report", _sheet_text(payload, "SUMMARY"))
        self.assertIn("Requested #1", _sheet_text(payload, "RESULTS"))

    def test_can_write_to_path_and_stream(self):
        model = build_vde_request_report_model(_request_draft(), _resolution_result(), _save_result())
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "report.xlsx"
            saved_path = generate_vde_request_report_xlsx(model, output_path)
            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.exists())
            buffer = io.BytesIO()
            returned = generate_vde_request_report_xlsx(model, buffer)
            self.assertIs(returned, buffer)
            self.assertGreater(len(buffer.getvalue()), 0)

    def test_report_contains_invalid_and_unsaved_proposals(self):
        proposal = _proposal_result("proposal_req_1", display_index=1, status="Invalid", total_mj_per_km=None, net_mj_per_km=None)
        model = build_vde_request_report_model(_request_draft(), _resolution_result([proposal]), None)

        self.assertEqual(model["summary_rows"][1]["Saved?"], "No")
        self.assertEqual(model["summary_rows"][1]["Status"], "Invalid")

    def test_no_absolute_paths_are_embedded(self):
        draft = _request_draft()
        draft["source"]["filename"] = r"C:\Users\CaioHenriqueFerreira\secret\request.xlsx"
        model = build_vde_request_report_model(draft, _resolution_result(), _save_result())
        payload = generate_vde_request_report_xlsx(model)

        self.assertNotIn(r"C:\Users\CaioHenriqueFerreira\secret", _sheet_text(payload, "SUMMARY"))
        self.assertIn("request.xlsx", _sheet_text(payload, "SUMMARY"))

    def test_end_to_end_request_to_save_to_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            request_path = Path(temp_dir) / "request.xlsx"
            _write_xlsx(request_path, _build_request_workbook())
            draft = parse_vde_request_workbook(request_path)
            state = build_v21_workbook_state_from_request_draft(draft, {"rows": []})
            resolution = resolve_vde_request(state, _baseline_context())
            fingerprint = build_request_resolution_fingerprint(state, _baseline_context())
            plan = build_vde_request_save_plan(
                resolution,
                save_mode=SAVE_MODE_SELECTED,
                selected_proposal_ids=["proposal_req_1"],
                review_confirmations={"proposal_req_1": True},
                request_state=state,
                current_fingerprint=fingerprint,
                resolution_fingerprint=fingerprint,
            )
            db_path = Path(temp_dir) / "vde_temp.db"
            con = sqlite3.connect(db_path)
            con.execute(
                """
                CREATE TABLE vde_db (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    legislation TEXT,
                    category TEXT,
                    make TEXT,
                    model TEXT,
                    year INTEGER,
                    notes TEXT,
                    mass_kg REAL,
                    test_mass_kg REAL,
                    test_mass_low_kg REAL,
                    test_mass_high_kg REAL,
                    test_mass_basis TEXT,
                    inertia_class REAL,
                    weight_dist_fr_pct REAL,
                    payload_kg REAL,
                    wltp_category TEXT,
                    cda_m2 REAL,
                    coast_A_N REAL,
                    coast_B_N_per_kph REAL,
                    coast_C_N_per_kph2 REAL,
                    vde_total_mj_per_km REAL,
                    vde_net_mj_per_km REAL,
                    cycle_name TEXT,
                    cycle_source TEXT,
                    vde_id_parent INTEGER,
                    baseline_A_N REAL,
                    baseline_B_N_per_kph REAL,
                    baseline_C_N_per_kph2 REAL,
                    baseline_mass_kg REAL,
                    front_tire_id INTEGER,
                    rear_tire_id INTEGER,
                    tire_A_final REAL,
                    tire_B_final REAL,
                    tire_C_final REAL,
                    tire_calc_source TEXT,
                    tire_load_mass_basis TEXT,
                    tire_improvement_pct REAL,
                    rrc_N_per_kN REAL,
                    smerf REAL,
                    front_pressure_psi REAL,
                    rear_pressure_psi REAL,
                    trans_A_coef_N REAL,
                    trans_B_coef_Npkph REAL,
                    trans_C_coef_Npkph2 REAL,
                    brake_A_coef_N REAL,
                    brake_B_coef_Npkph REAL,
                    brake_C_coef_Npkph2 REAL,
                    parasitic_A_coef_N REAL,
                    parasitic_B_coef_Npkph REAL,
                    parasitic_C_coef_Npkph2 REAL,
                    gvwr_kg REAL,
                    gcwr_kg REAL,
                    trailer_mass_kg REAL,
                    trailer_code TEXT,
                    trailer_roadload_source TEXT,
                    trailer_A_coef_N REAL,
                    trailer_B_coef_Npkph REAL,
                    trailer_C_coef_Npkph2 REAL,
                    mass_rule_status TEXT,
                    mass_rule_notes TEXT,
                    updated_at TEXT
                )
                """
            )
            con.execute("INSERT INTO vde_db (id, make, model, coast_A_N, mass_kg) VALUES (5038, 'FORD', 'TEST', 120.0, 1600.0)")
            con.commit()
            con.close()

            def _table_columns(_table):
                probe = sqlite3.connect(db_path)
                try:
                    return [row[1] for row in probe.execute("PRAGMA table_info(vde_db)").fetchall()]
                finally:
                    probe.close()

            save_result = execute_vde_request_save_plan(
                plan,
                services={
                    "ensure_db": lambda: None,
                    "connect_db": lambda: sqlite3.connect(db_path),
                    "table_columns": _table_columns,
                },
            )
            report_model = build_vde_request_report_model(draft, resolution, save_result)
            payload = generate_vde_request_report_xlsx(report_model)

            self.assertEqual(save_result["status"], "success")
            self.assertEqual(save_result["saved_proposals"][0]["proposal_id"], "proposal_req_1")
            self.assertIn("Requested #1", _sheet_text(payload, "SUMMARY"))
            self.assertIn(str(save_result["saved_proposals"][0]["vde_row_id"]), _sheet_text(payload, "SUMMARY"))


if __name__ == "__main__":
    unittest.main()
