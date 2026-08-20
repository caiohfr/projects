from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import io

from src.vde_core import db as db_module
from src.vde_core.database_management_contract import ChangeCommand, EntityType
from src.vde_core.database_management_service import apply_change, browse_records, preview_change
from src.vde_core.database_management_spreadsheet import (
    TEMPLATE_VERSION,
    _build_xlsx,
    generate_controlled_template,
    preview_spreadsheet_import,
    spreadsheet_template_contract,
    stage_commands_from_import,
)
from src.vde_core.qa_mock_data import seed_qa_database
from src.vde_core.vde_request_parser import _read_xlsx_workbook


class DatabaseManagementSpreadsheet7ETests(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temporary_directory.name) / "database_management_7e.db"
        seed_qa_database(self.db_path, overwrite=False)

    def tearDown(self):
        gc.collect()
        self._temporary_directory.cleanup()

    def test_generates_four_controlled_templates_with_expected_sheets(self):
        expected = {
            EntityType.VDE: ("VDE template.xlsx", ["Instructions", "VDE"]),
            EntityType.FUEL_CONSUMPTION: ("Fuel Consumption template.xlsx", ["Instructions", "Fuel Consumption"]),
            EntityType.TIRE: ("Tire template.xlsx", ["Instructions", "Tires"]),
            EntityType.COMPONENT: (
                "Components template.xlsx",
                ["Instructions", "Transmission", "Brake", "Axle & Hubs", "Parasitic"],
            ),
        }
        for entity, (expected_filename, expected_sheets) in expected.items():
            filename, content = generate_controlled_template(entity)
            target = Path(self._temporary_directory.name) / filename
            target.write_bytes(content)
            workbook = _read_xlsx_workbook(target)
            self.assertEqual(filename, expected_filename)
            self.assertEqual(workbook["sheet_names"], expected_sheets)
            self.assertEqual(workbook["sheets"]["Instructions"][1][1], TEMPLATE_VERSION)
            contract = spreadsheet_template_contract(entity)
            self.assertEqual(workbook["sheets"][contract.sheets[0].name][0], list(contract.columns))

    def test_update_preserves_explicit_zero_and_blank_means_no_change(self):
        with db_module.using_db_path(self.db_path):
            current = db_module.fetchone("SELECT * FROM tire_roadload_db ORDER BY id LIMIT 1")
            rows = {
                "Tires": [
                    {
                        "internal_id": current["id"],
                        "record_origin": current["record_origin"],
                        "manufacturer": None,
                        "rr_n_per_kn": 0,
                    }
                ]
            }
            _, content = generate_controlled_template(EntityType.TIRE, rows_by_sheet=rows)
            preview = preview_spreadsheet_import(content, EntityType.TIRE)

        self.assertEqual(preview.counts, {"inserted": 0, "updated": 1, "skipped": 0, "invalid": 0})
        imported = preview.rows[0]
        self.assertEqual(imported.payload["rr_n_per_kn"], 0.0)
        self.assertNotIn("manufacturer", imported.payload)

    def test_insert_with_required_blank_is_invalid(self):
        row = {
            "record_origin": "IMPORTED",
            "source_name": "lab",
            "source_record_id": "BLANK-1",
            "tire_test_code": "BLANK-1",
            "manufacturer": "Lab",
            "model": None,
            "standard_family": "ISO",
            "rr_n_per_kn": 7.5,
        }
        _, content = generate_controlled_template(EntityType.TIRE, rows_by_sheet={"Tires": [row]})
        with db_module.using_db_path(self.db_path):
            preview = preview_spreadsheet_import(content, EntityType.TIRE)

        self.assertEqual(preview.counts["invalid"], 1)
        self.assertIn("model is required", " ".join(issue.message for issue in preview.rows[0].issues))

    def test_duplicate_source_identity_in_upload_is_invalid(self):
        base = {
            "record_origin": "IMPORTED",
            "source_name": "supplier",
            "source_record_id": "DUP-7E",
            "tire_test_code": "DUP-7E-A",
            "manufacturer": "Lab",
            "model": "A",
            "standard_family": "ISO",
            "rr_n_per_kn": 8.0,
        }
        second = {**base, "tire_test_code": "DUP-7E-B", "model": "B"}
        _, content = generate_controlled_template(EntityType.TIRE, rows_by_sheet={"Tires": [base, second]})
        with db_module.using_db_path(self.db_path):
            preview = preview_spreadsheet_import(content, EntityType.TIRE)

        self.assertEqual(preview.counts["invalid"], 2)
        self.assertTrue(all(any(issue.code == "duplicate_upload_identity" for issue in row.issues) for row in preview.rows))

    def test_external_identity_import_is_idempotent_and_does_not_delete_absent_rows(self):
        row = {
            "record_origin": "IMPORTED",
            "source_name": "supplier-7e",
            "source_record_id": "TRANS-77",
            "component_code": "TRANS-7E-77",
            "component_name": "Imported transmission",
            "equivalent_A_N": 0,
            "equivalent_B_N_per_kph": 0,
            "equivalent_C_N_per_kph2": 0,
            "loss_pct": 0,
            "notes": "Controlled spreadsheet import",
        }
        _, content = generate_controlled_template(EntityType.COMPONENT, rows_by_sheet={"Transmission": [row]})
        with db_module.using_db_path(self.db_path):
            before_count = len(browse_records("COMPONENT", component_domain="transmission", include_archived=True, limit=1000))
            first = preview_spreadsheet_import(content, EntityType.COMPONENT)
            self.assertEqual(first.counts["inserted"], 1)
            self._apply_ready_rows(first)
            after_count = len(browse_records("COMPONENT", component_domain="transmission", include_archived=True, limit=1000))
            second = preview_spreadsheet_import(content, EntityType.COMPONENT)

        self.assertEqual(after_count, before_count + 1)
        self.assertEqual(second.counts, {"inserted": 0, "updated": 0, "skipped": 1, "invalid": 0})
        self.assertEqual(second.rows[0].match_method, "SOURCE_IDENTITY")
        self.assertEqual(second.rows[0].status, "SKIPPED")

    def test_internal_id_targets_update_and_invalid_internal_id_does_not_fall_back(self):
        with db_module.using_db_path(self.db_path):
            current = db_module.fetchone("SELECT * FROM vde_db ORDER BY id LIMIT 1")
            valid_rows = {
                "VDE": [
                    {
                        "internal_id": current["id"],
                        "record_origin": current["record_origin"],
                        "notes": "Imported note 7E",
                    }
                ]
            }
            _, valid_content = generate_controlled_template(EntityType.VDE, rows_by_sheet=valid_rows)
            valid = preview_spreadsheet_import(valid_content, EntityType.VDE)
            invalid_rows = {
                "VDE": [
                    {
                        "internal_id": 999999999,
                        "record_origin": "IMPORTED_REFERENCE",
                        "source_name": current.get("source_name"),
                        "source_record_id": current.get("source_record_id"),
                        "legislation": "EPA",
                        "category": "QA",
                        "make": "QA",
                        "model": "Invalid ID",
                        "mass_kg": 1500,
                    }
                ]
            }
            _, invalid_content = generate_controlled_template(EntityType.VDE, rows_by_sheet=invalid_rows)
            invalid = preview_spreadsheet_import(invalid_content, EntityType.VDE)

        self.assertEqual(valid.rows[0].action, "UPDATE")
        self.assertEqual(valid.rows[0].match_method, "INTERNAL_ID")
        self.assertEqual(valid.rows[0].status, "READY")
        self.assertEqual(invalid.rows[0].status, "INVALID")
        self.assertTrue(any(issue.code == "internal_id_not_found" for issue in invalid.rows[0].issues))

    def test_unknown_columns_are_reported_and_require_confirmation(self):
        contract = spreadsheet_template_contract(EntityType.TIRE)
        headers = [*contract.columns, "mystery_result"]
        values = [None] * len(headers)
        values[headers.index("record_origin")] = "IMPORTED"
        values[headers.index("tire_test_code")] = "UNKNOWN-COL-7E"
        values[headers.index("manufacturer")] = "Lab"
        values[headers.index("model")] = "Reference"
        values[headers.index("standard_family")] = "ISO"
        values[headers.index("rr_n_per_kn")] = 8.0
        values[-1] = 123
        content = _build_xlsx(
            [
                ("Instructions", [["Template version", TEMPLATE_VERSION]]),
                ("Tires", [headers, values]),
            ]
        )
        with db_module.using_db_path(self.db_path):
            preview = preview_spreadsheet_import(content, EntityType.TIRE)
            with self.assertRaises(ValueError):
                stage_commands_from_import(preview)
            commands = stage_commands_from_import(preview, confirm_unknown_columns=True)

        self.assertEqual(preview.unknown_columns, ("mystery_result",))
        self.assertEqual(len(commands), 1)
        self.assertNotIn("mystery_result", commands[0]["payload"])

    def test_formula_cells_are_rejected_as_database_source_values(self):
        _, content = generate_controlled_template(EntityType.VDE)
        source = io.BytesIO(content)
        target = io.BytesIO()
        with ZipFile(source) as input_archive, ZipFile(target, "w", compression=ZIP_DEFLATED) as output_archive:
            for name in input_archive.namelist():
                payload = input_archive.read(name)
                if name == "xl/worksheets/sheet2.xml":
                    payload = payload.replace(b"</sheetData>", b'<row r="2"><c r="S2"><f>1+1</f><v>2</v></c></row></sheetData>')
                output_archive.writestr(name, payload)
        with db_module.using_db_path(self.db_path):
            preview = preview_spreadsheet_import(target.getvalue(), EntityType.VDE)

        self.assertFalse(preview.can_stage)
        self.assertTrue(any(issue.code == "spreadsheet_formulas_not_supported" for issue in preview.issues))

    def _apply_ready_rows(self, import_preview):
        for command in stage_commands_from_import(import_preview):
            reason = "Controlled spreadsheet import"
            preview = preview_change(
                ChangeCommand(
                    entity_type=command["entity_type"],
                    action=command["action"],
                    record_id=command.get("record_id"),
                    record_origin=command.get("record_origin"),
                    current_record=command.get("current_record") or {},
                    payload=command.get("payload") or {},
                    reason=reason,
                )
            )
            apply_change(preview, reason=reason)


if __name__ == "__main__":
    unittest.main()
