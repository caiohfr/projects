from __future__ import annotations

import io
from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from src.vde_core.vde_request_parser import parse_vde_request_workbook
from src.vde_core.vde_request_template import (
    build_prefilled_ppe_template,
    build_prefilled_ppe_template_filename,
    compare_printed_snapshot,
    extract_referenced_baseline_id,
    resolve_imported_baseline_status,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "docs" / "templates" / "EcoDrive_VDE_PPE_Request_Input_template_v01.xlsx"
XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _baseline_snapshot() -> dict:
    return {
        "selected_baseline_vde_id": 5038,
        "line_source": "Existing VDE DB",
        "make": "AUDI",
        "model": "TEST08062026",
        "year": 2027,
        "legislation": "EPA",
        "category": "SUBCOMPACT CARS",
        "electrification": "ICE",
        "transmission_type": "AT",
        "drive_type": "4WD",
        "fuel_type": "GASOLINE",
        "cycle_name": "FTP75",
        "mass_kg": 1735.0,
        "test_mass_kg": 1735.0,
        "test_mass_basis": "PHYSICAL_TEST_MASS",
        "A": 145.17,
        "B": 0.09357,
        "C": 0.040838,
        "cda_m2": 0.64,
        "transmission_loss_pct": 6.5,
        "trans_A_coef_N": 0.0,
    }


def _request_draft() -> dict:
    return {
        "schema_version": "0.1",
        "template_version": "0.1",
        "source": {
            "filename": "ui_request.xlsx",
            "source_type": "UI",
        },
        "baseline_corrections": {
            "mass_kg": 1750.0,
        },
        "proposals": [
            {
                "proposal_id": "proposal_req_1",
                "display_index": 1,
                "source_column": "Requested #1",
                "name": "Mass and Aero update",
                "walk_from": {"kind": "baseline", "proposal_id": None, "source_column": "Baseline"},
                "domain_requests": {
                    "mass": {
                        "domain": "mass",
                        "raw_proposal_type": "Custom test mass",
                        "proposal_type": "CUSTOM_MASS",
                        "selection_mode": "Custom test mass",
                        "raw_values": {"mass_kg": 1800.0},
                        "proposal_details_seed": {"mass_kg": 1800.0},
                    },
                    "aero": {
                        "domain": "aero",
                        "raw_proposal_type": "Delta CdA",
                        "proposal_type": "AERO_DELTA_CDA",
                        "selection_mode": "Delta CdA",
                        "raw_values": {"cda_m2": 0.02},
                        "proposal_details_seed": {"cda_m2": 0.02},
                    },
                    "transmission": {
                        "domain": "transmission",
                        "raw_proposal_type": "Delta ABC",
                        "proposal_type": "UPDATE_TRANS_DRAG_ABC",
                        "selection_mode": "Delta ABC",
                        "raw_values": {"trans_A_coef_N": 0.0},
                        "proposal_details_seed": {"change_mode": "Delta ABC", "trans_A_coef_N": 0.0},
                    },
                },
            }
        ],
        "issues": [],
    }


def _sheet_names(payload: bytes) -> list[str]:
    with ZipFile(io.BytesIO(payload)) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        return [sheet.attrib.get("name") for sheet in workbook.findall(f".//{XML_NS}sheet")]


def _custom_props_text(payload: bytes) -> str:
    with ZipFile(io.BytesIO(payload)) as archive:
        if "docProps/custom.xml" not in archive.namelist():
            return ""
        return archive.read("docProps/custom.xml").decode("utf-8")


class TestVdeRequestTemplate(unittest.TestCase):
    def test_build_prefilled_template_requires_baseline(self):
        with self.assertRaises(ValueError):
            build_prefilled_ppe_template(TEMPLATE_PATH, {}, None)

    def test_build_prefilled_template_preserves_template_and_round_trips(self):
        original_bytes = TEMPLATE_PATH.read_bytes()
        payload = build_prefilled_ppe_template(
            TEMPLATE_PATH,
            _baseline_snapshot(),
            _request_draft(),
            {"baseline_source": "Existing VDE DB", "source_type": "UI"},
        )
        self.assertIsInstance(payload, bytes)
        self.assertGreater(len(payload), 0)
        self.assertEqual(TEMPLATE_PATH.read_bytes(), original_bytes)
        self.assertEqual(_sheet_names(payload), ["REQUEST", "FIELD_MAP", "LISTS", "RULES"])

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "prefilled.xlsx"
            path.write_bytes(payload)
            draft = parse_vde_request_workbook(path)

        self.assertEqual(draft["baseline_printed"]["selected_baseline_vde_id"], 5038)
        self.assertEqual(draft["baseline_printed"]["mass_kg"], 1735.0)
        self.assertEqual(draft["baseline_printed"]["trans_A_coef_N"], 0.0)
        self.assertEqual(draft["baseline_corrections"]["mass_kg"], 1750.0)
        self.assertEqual(draft["proposals"][0]["source_column"], "Requested #1")
        self.assertEqual(draft["proposals"][0]["walk_from"]["source_column"], "Baseline")
        self.assertEqual(draft["proposals"][0]["name"], "Mass and Aero update")
        self.assertEqual(draft["proposals"][0]["domain_requests"]["mass"]["raw_proposal_type"], "Custom test mass")
        self.assertEqual(draft["proposals"][0]["domain_requests"]["mass"]["raw_values"]["mass_kg"], 1800.0)
        self.assertEqual(draft["proposals"][0]["domain_requests"]["transmission"]["raw_values"]["trans_A_coef_N"], 0.0)
        self.assertIsNone(draft["baseline_printed"]["payload_kg"])

    def test_prefilled_template_writes_custom_properties_without_absolute_paths(self):
        payload = build_prefilled_ppe_template(
            TEMPLATE_PATH,
            _baseline_snapshot(),
            _request_draft(),
            {"baseline_source": "Existing VDE DB", "source_type": "Excel"},
        )
        custom_xml = _custom_props_text(payload)
        self.assertIn("baseline_vde_id", custom_xml)
        self.assertIn("Existing VDE DB", custom_xml)
        self.assertIn("request_schema_version", custom_xml)
        self.assertNotIn("C:\\Users\\CaioHenriqueFerreira", custom_xml)

    def test_filename_is_sanitized(self):
        name = build_prefilled_ppe_template_filename(5038, "AUDI / TEST:08062026", "2026-07-13")
        self.assertEqual(name, "EcoDrive_VDE_PPE_Request_5038_AUDI_TEST_08062026_2026-07-13.xlsx")

    def test_extract_referenced_baseline_id(self):
        self.assertEqual(extract_referenced_baseline_id({"baseline_printed": {"selected_baseline_vde_id": 4998}}), 4998)

    def test_compare_printed_snapshot_detects_mismatch(self):
        result = compare_printed_snapshot(
            {"selected_baseline_vde_id": 5038, "mass_kg": 1700.0, "trans_A_coef_N": 0.0},
            {"selected_baseline_vde_id": 5038, "mass_kg": 1735.0, "trans_A_coef_N": 0.0},
        )
        self.assertEqual(result["status"], "Review")
        self.assertEqual(result["divergent_fields"][0]["field_key"], "mass_kg")

    def test_compare_printed_snapshot_accepts_identical_values(self):
        result = compare_printed_snapshot(
            {"selected_baseline_vde_id": 5038, "mass_kg": 1735.0, "trans_A_coef_N": 0.0},
            {"selected_baseline_vde_id": 5038, "mass_kg": 1735, "trans_A_coef_N": 0},
        )
        self.assertEqual(result["status"], "OK")
        self.assertTrue(result["ok"])

    def test_resolve_imported_baseline_status(self):
        self.assertEqual(resolve_imported_baseline_status(None, 5038, {"id": 5038})["status"], "ready_to_load")
        self.assertEqual(resolve_imported_baseline_status(5038, 5038, {"id": 5038})["status"], "matched_current")
        self.assertEqual(resolve_imported_baseline_status(5120, 5038, {"id": 5038})["status"], "mismatch")
        self.assertEqual(resolve_imported_baseline_status(None, 5038, None)["status"], "unresolved")
