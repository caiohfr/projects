from __future__ import annotations

import gc
import sqlite3
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from src.vde_core import db as db_module
from src.vde_core.component_repositories import (
    archive_component,
    create_component,
    duplicate_component,
    find_component_by_source_identity,
    get_component,
    load_component_repository,
    load_mock_component_repository,
    lookup_component,
    restore_component,
    update_component,
)
from src.vde_core.qa_mock_data import seed_qa_database
from src.vde_core.repositories import fetch_vde_by_id
from src.vde_core.vde_request_compact_adapter import build_v22_preview_bundle, compact_baseline_context
from src.vde_core.vde_request_compact_state import (
    apply_v22_baseline,
    apply_v22_domain_inputs,
    apply_v22_proposal_matrix,
    create_v22_state,
)
from src.vde_app.components.vde_request_lookup import component_lookup_rows


class ComponentSqliteRepository7BTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "eco_drive_qa_components.db"
        seed_qa_database(self.db_path, overwrite=False)

    def tearDown(self):
        gc.collect()
        self._temp_dir.cleanup()

    def test_qa_seed_populates_component_table_and_domain_filters(self):
        with db_module.using_db_path(self.db_path):
            domains = {
                domain: load_component_repository(domain)
                for domain in ("transmission", "brake", "axle_hubs", "parasitic")
            }

        for domain, repository in domains.items():
            with self.subTest(domain=domain):
                self.assertEqual(repository.source, "sqlite_component_db")
                self.assertGreaterEqual(len(repository.list_components()), 5)
                self.assertTrue(all(row["domain"] == domain for row in repository.list_components()))
        self.assertEqual(domains["transmission"].get_by_id("TRANS-MOCK-001")["trans_A"], 12.0)
        self.assertEqual(domains["transmission"].get_by_id("TRANS-MOCK-001")["net_bridge_eligible"], "TRUE")
        with sqlite3.connect(self.db_path) as con:
            columns = {row[1] for row in con.execute("PRAGMA table_info(component_db)")}
        self.assertTrue(
            {
                "id",
                "record_origin",
                "record_status",
                "domain",
                "component_code",
                "source_name",
                "source_record_id",
                "equivalent_A_N",
                "equivalent_B_N_per_kph",
                "equivalent_C_N_per_kph2",
            }
            <= columns
        )

    def test_active_lookup_uses_sqlite_adapter_contract(self):
        with db_module.using_db_path(self.db_path):
            result = lookup_component("brake", "BRAKE-MOCK-001")
            browser_rows = component_lookup_rows("brake", "BRAKE-MOCK-001")

        self.assertTrue(result["found"])
        self.assertEqual(result["source"], "sqlite_component_db")
        self.assertEqual(result["component"]["component_id"], "BRAKE-MOCK-001")
        self.assertEqual(result["component"]["brake_A"], 4.0)
        self.assertEqual(result["component"]["record_origin"], "QA")
        self.assertEqual(result["component"]["record_status"], "ACTIVE")
        self.assertEqual(len(browser_rows), 1)
        self.assertEqual(browser_rows[0]["lookup_id"], "BRAKE-MOCK-001")
        self.assertEqual(browser_rows[0]["A"], 4.0)

    def test_create_update_and_source_identity_preserve_explicit_zero(self):
        with db_module.using_db_path(self.db_path):
            created = create_component(
                "transmission",
                {
                    "component_code": "TRANS-LAB-001",
                    "component_name": "Lab transmission",
                    "source_name": "lab_import",
                    "source_record_id": "ROW-77",
                    "source_reference": "LAB-REPORT-77",
                    "trans_A": 9.0,
                    "trans_B": 0.004,
                    "trans_C": 0.0008,
                    "loss_pct": 0.0,
                },
            )
            updated = update_component(
                "transmission",
                created["id"],
                {"trans_A": 0.0, "component_name": "Corrected lab transmission"},
            )
            identity = find_component_by_source_identity("transmission", "lab_import", "ROW-77")

        self.assertEqual(created["record_origin"], "MANUAL")
        self.assertEqual(updated["trans_A"], 0.0)
        self.assertEqual(updated["loss_pct"], 0.0)
        self.assertEqual(updated["component_name"], "Corrected lab transmission")
        self.assertEqual(identity["id"], created["id"])

    def test_archive_restore_and_duplicate_keep_internal_identity_separate(self):
        with db_module.using_db_path(self.db_path):
            archived = archive_component("axle_hubs", "AXLE-MOCK-001")
            hidden = get_component("axle_hubs", "AXLE-MOCK-001")
            restored = restore_component("axle_hubs", archived["id"])
            duplicate = duplicate_component(
                "axle_hubs",
                restored["id"],
                component_code="AXLE-MOCK-001-COPY-QA",
                overrides={"component_name": "Independent copied observation"},
            )

        self.assertEqual(archived["record_status"], "ARCHIVED")
        self.assertIsNone(hidden)
        self.assertEqual(restored["record_status"], "ACTIVE")
        self.assertNotEqual(duplicate["id"], restored["id"])
        self.assertEqual(duplicate["component_id"], "AXLE-MOCK-001-COPY-QA")
        self.assertEqual(duplicate["axle_hubs_A"], restored["axle_hubs_A"])

    def test_search_is_scoped_to_domain_and_active_rows(self):
        with db_module.using_db_path(self.db_path):
            archive_component("parasitic", "PARA-MOCK-LOW")
            active = load_component_repository("parasitic")
            all_rows = load_component_repository("parasitic", include_archived=True)

        self.assertNotIn("PARA-MOCK-LOW", {row["component_id"] for row in active.search("low")})
        self.assertIn("PARA-MOCK-LOW", {row["component_id"] for row in all_rows.search("low")})
        self.assertTrue(all(row["domain"] == "parasitic" for row in all_rows.list_components()))

    def test_golden_component_lookup_is_numerically_equal_before_and_after_cutover(self):
        with db_module.using_db_path(self.db_path):
            baseline = fetch_vde_by_id(900001)
            state = apply_v22_baseline(create_v22_state(), baseline)
            state = apply_v22_proposal_matrix(
                state,
                [
                    {
                        "proposal_id": "requested_1",
                        "walk_from": "baseline",
                        "transmission": "Lookup from DB",
                    }
                ],
            )
            state = apply_v22_domain_inputs(
                state,
                "transmission",
                {"requested_1": {"transmission_component_db_id": "TRANS-MOCK-001"}},
            )
            sqlite_bundle = build_v22_preview_bundle(
                state,
                baseline_context=compact_baseline_context(state),
            )
            fixture_repositories = {
                domain: load_mock_component_repository(domain)
                for domain in ("transmission", "brake", "axle_hubs", "parasitic")
            }
            fixture_bundle = build_v22_preview_bundle(
                deepcopy(state),
                baseline_context=compact_baseline_context(state),
                component_repositories=fixture_repositories,
            )

        sqlite_result = sqlite_bundle["resolution_result"]["proposal_results"][0]
        fixture_result = fixture_bundle["resolution_result"]["proposal_results"][0]
        self.assertEqual(sqlite_result["requested_snapshot"], fixture_result["requested_snapshot"])
        self.assertEqual(sqlite_result["preview_summary"], fixture_result["preview_summary"])
        sqlite_snapshot = sqlite_result["domain_results"]["transmission"]["component_action"]["component_snapshot"]
        fixture_snapshot = fixture_result["domain_results"]["transmission"]["component_action"]["component_snapshot"]
        resolver_fields = {
            "component_id",
            "component_name",
            "trans_A",
            "trans_B",
            "trans_C",
            "loss_pct",
            "component_type",
            "component_position",
            "driveline_architecture",
            "physical_boundary",
            "configuration_from",
            "configuration_to",
            "test_condition_type",
            "test_method",
            "hardware_reference",
            "source_reference",
            "net_bridge_eligible",
        }
        self.assertEqual(
            {field: sqlite_snapshot.get(field) for field in resolver_fields},
            {field: fixture_snapshot.get(field) for field in resolver_fields},
        )


if __name__ == "__main__":
    unittest.main()
