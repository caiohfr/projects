from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path

from src.vde_core import db as db_module
from src.vde_core.data_change_log_repository import fetch_change_log
from src.vde_core.database_management_contract import ChangeCommand
from src.vde_core.database_management_service import (
    apply_change,
    browse_records,
    duplicate_payload_for,
    get_record,
    preview_change,
    simple_dependencies,
)
from src.vde_core.qa_mock_data import seed_qa_database


class DatabaseManagement7CTests(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temporary_directory.name) / "database_management_7c.db"
        seed_qa_database(self.db_path, overwrite=False)

    def tearDown(self):
        gc.collect()
        self._temporary_directory.cleanup()

    def test_browse_filters_component_domain_and_hides_archived_rows(self):
        with db_module.using_db_path(self.db_path):
            rows = browse_records("COMPONENT", component_domain="brake", query="MOCK")
            self.assertTrue(rows)
            self.assertTrue(all(row["domain"] == "brake" for row in rows))
            record = rows[0]
            preview = preview_change(
                ChangeCommand(
                    entity_type="COMPONENT",
                    action="ARCHIVE",
                    record_id=record["id"],
                    record_origin=record["record_origin"],
                    current_record=record,
                    reason="Retire obsolete fixture",
                )
            )
            apply_change(preview, reason="Retire obsolete fixture")
            active_rows = browse_records("COMPONENT", component_domain="brake")
            all_rows = browse_records("COMPONENT", component_domain="brake", include_archived=True)

        self.assertNotIn(record["id"], {row["id"] for row in active_rows})
        self.assertIn(record["id"], {row["id"] for row in all_rows})
        with self.assertRaises(ValueError):
            browse_records("COMPONENT")

    def test_vde_metadata_update_is_staged_until_apply_and_logs_receipt(self):
        with db_module.using_db_path(self.db_path):
            current = get_record("VDE", 900001)
            preview = preview_change(
                ChangeCommand(
                    entity_type="VDE",
                    action="UPDATE",
                    record_id=current["id"],
                    record_origin=current["record_origin"],
                    current_record=current,
                    payload={"make": "QA Corrected"},
                    reason="Correct imported source label",
                )
            )
            self.assertEqual(get_record("VDE", 900001)["make"], current["make"])
            result = apply_change(preview, reason="Correct imported source label")
            changed = get_record("VDE", 900001)
            receipt = fetch_change_log(result.operation_id)

        self.assertTrue(result.committed)
        self.assertIsNotNone(result.change_log_id)
        self.assertEqual(changed["make"], "QA Corrected")
        self.assertEqual(receipt["before_json"]["make"], current["make"])
        self.assertEqual(receipt["after_json"]["make"], "QA Corrected")

    def test_tire_create_duplicate_archive_restore_use_existing_normalization(self):
        with db_module.using_db_path(self.db_path):
            create_preview = preview_change(
                ChangeCommand(
                    entity_type="TIRE",
                    action="CREATE",
                    record_origin="MANUAL",
                    payload={
                        "tire_test_code": "LAB-7C-TIRE",
                        "manufacturer": "Lab",
                        "model": "Reference",
                        "standard_family": "ISO",
                        "rr_n_per_kn": 7.8,
                        "notes": "Created through staged management",
                    },
                )
            )
            created = apply_change(create_preview)
            tire = get_record("TIRE", created.affected_record_ids[0])
            duplicate_preview = preview_change(
                ChangeCommand(
                    entity_type="TIRE",
                    action="DUPLICATE",
                    record_id=tire["id"],
                    record_origin=tire["record_origin"],
                    current_record=tire,
                    payload=duplicate_payload_for("TIRE", tire),
                )
            )
            duplicated = apply_change(duplicate_preview)
            archive_preview = preview_change(
                ChangeCommand(
                    entity_type="TIRE",
                    action="ARCHIVE",
                    record_id=tire["id"],
                    record_origin=tire["record_origin"],
                    current_record=tire,
                    reason="Archive test row",
                )
            )
            apply_change(archive_preview, reason="Archive test row")
            restore_preview = preview_change(
                ChangeCommand(
                    entity_type="TIRE",
                    action="RESTORE",
                    record_id=tire["id"],
                    record_origin=tire["record_origin"],
                    current_record=get_record("TIRE", tire["id"]),
                    reason="Restore test row",
                )
            )
            apply_change(restore_preview, reason="Restore test row")
            restored = get_record("TIRE", tire["id"])

        self.assertEqual(tire["standard_family"], "ISO")
        self.assertEqual(restored["is_active"], 1)
        self.assertNotEqual(duplicated.affected_record_ids[0], created.affected_record_ids[0])

    def test_component_create_update_duplicate_and_change_log_stay_in_one_contract(self):
        with db_module.using_db_path(self.db_path):
            create_preview = preview_change(
                ChangeCommand(
                    entity_type="COMPONENT",
                    action="CREATE",
                    record_origin="MANUAL",
                    payload={
                        "domain": "transmission",
                        "component_code": "TRANS-7C-LAB",
                        "component_name": "Transmission lab observation",
                        "source_name": "lab_report",
                        "notes": "Staged component row",
                        "equivalent_A_N": 9.0,
                        "equivalent_B_N_per_kph": 0.004,
                        "equivalent_C_N_per_kph2": 0.0008,
                        "loss_pct": 0.0,
                    },
                )
            )
            created = apply_change(create_preview)
            component = get_record("COMPONENT", created.affected_record_ids[0], component_domain="transmission")
            update_preview = preview_change(
                ChangeCommand(
                    entity_type="COMPONENT",
                    action="UPDATE",
                    record_id=component["id"],
                    record_origin=component["record_origin"],
                    current_record=component,
                    payload={"equivalent_A_N": 0.0},
                    reason="Correct measured coefficient",
                )
            )
            updated = apply_change(update_preview, reason="Correct measured coefficient")
            duplicate_preview = preview_change(
                ChangeCommand(
                    entity_type="COMPONENT",
                    action="DUPLICATE",
                    record_id=component["id"],
                    record_origin=component["record_origin"],
                    current_record=component,
                    payload=duplicate_payload_for("COMPONENT", component),
                )
            )
            duplicate = apply_change(duplicate_preview)
            corrected = get_record("COMPONENT", updated.affected_record_ids[0], component_domain="transmission")
            copied = get_record("COMPONENT", duplicate.affected_record_ids[0], component_domain="transmission")
            receipt = fetch_change_log(updated.operation_id)

        self.assertEqual(corrected["equivalent_A_N"], 0.0)
        self.assertNotEqual(copied["component_code"], corrected["component_code"])
        self.assertTrue(receipt)

    def test_delete_vde_is_blocked_when_fuel_is_related(self):
        with db_module.using_db_path(self.db_path):
            fuel_preview = preview_change(
                ChangeCommand(
                    entity_type="FUEL_CONSUMPTION",
                    action="CREATE",
                    record_origin="MEASURED",
                    payload={"vde_id": 900001, "electrification": "ICE", "method_note": "QA dependency"},
                )
            )
            apply_change(fuel_preview)
            vde = get_record("VDE", 900001)
            delete_preview = preview_change(
                ChangeCommand(
                    entity_type="VDE",
                    action="DELETE",
                    record_id=vde["id"],
                    record_origin=vde["record_origin"],
                    current_record=vde,
                    reason="Attempt delete with dependency",
                )
            )
            dependencies = simple_dependencies("VDE", vde["id"])
            with self.assertRaises(ValueError):
                apply_change(delete_preview, reason="Attempt delete with dependency")
            remaining = get_record("VDE", 900001)

        self.assertEqual(len(dependencies), 1)
        self.assertTrue(remaining)


if __name__ == "__main__":
    unittest.main()
