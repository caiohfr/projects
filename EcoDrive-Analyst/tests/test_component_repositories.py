from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.vde_core.component_repositories import (
    COMPONENT_PROVENANCE_FIELDS,
    ComponentRepository,
    create_component,
    load_mock_component_repository,
    lookup_component,
)


class ComponentRepositoriesTests(unittest.TestCase):
    def test_load_mock_repository_reads_csv(self):
        repo = load_mock_component_repository("transmission")

        self.assertIsInstance(repo, ComponentRepository)
        self.assertGreaterEqual(len(repo.list_components()), 2)
        self.assertEqual(repo.domain, "transmission")

    def test_component_mock_repositories_include_base_low_high_variants(self):
        load_mock_component_repository.cache_clear()
        expected = {
            "transmission": {"TRANS-MOCK-BASE", "TRANS-MOCK-LOW", "TRANS-MOCK-HIGH"},
            "brake": {"BRAKE-MOCK-BASE", "BRAKE-MOCK-LOW", "BRAKE-MOCK-HIGH"},
            "axle_hubs": {"AXLE-MOCK-BASE", "AXLE-MOCK-LOW", "AXLE-MOCK-HIGH"},
            "parasitic": {"PARA-MOCK-BASE", "PARA-MOCK-LOW", "PARA-MOCK-HIGH"},
        }
        for domain, ids in expected.items():
            with self.subTest(domain=domain):
                repo = load_mock_component_repository(domain)
                component_ids = {item["component_id"] for item in repo.list_components()}
                self.assertTrue(ids.issubset(component_ids))

    def test_lookup_component_found(self):
        result = lookup_component("brake", "BRAKE-MOCK-001")

        self.assertTrue(result["found"])
        self.assertEqual(result["component"]["component_id"], "BRAKE-MOCK-001")
        self.assertEqual(result["source"], "mock_csv")

    def test_old_csv_without_provenance_still_loads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "brake_components_mock.csv"
            with target.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "component_id",
                        "component_name",
                        "status",
                        "source",
                        "notes",
                        "brake_A",
                        "brake_B",
                        "brake_C",
                        "residual_torque_front_nm",
                        "residual_torque_rear_nm",
                        "wheel_radius_m",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "component_id": "BRAKE-OLD-1",
                        "component_name": "Old brake",
                        "status": "legacy",
                        "source": "legacy_csv",
                        "notes": "No provenance columns",
                        "brake_A": 4.0,
                        "brake_B": 0.001,
                        "brake_C": 0.0001,
                        "residual_torque_front_nm": 8.0,
                        "residual_torque_rear_nm": 8.0,
                        "wheel_radius_m": 0.34,
                    }
                )
            load_mock_component_repository.cache_clear()
            with patch("src.vde_core.component_repositories._DATA_DIR", root):
                component = load_mock_component_repository("brake").get_by_id("BRAKE-OLD-1")
            load_mock_component_repository.cache_clear()

        self.assertEqual(component["brake_A"], 4.0)
        for field_name in COMPONENT_PROVENANCE_FIELDS:
            self.assertIn(field_name, component)
            self.assertEqual(component[field_name], "")

    def test_mock_provenance_distinguishes_brake_conditions(self):
        repo = load_mock_component_repository("brake")

        self.assertEqual(repo.get_by_id("BRAKE-MOCK-BASE")["component_type"], "BRAKE_BASELINE_AS_RECEIVED")
        self.assertEqual(repo.get_by_id("BRAKE-MOCK-001")["component_type"], "BRAKE_STANDARD")
        self.assertEqual(repo.get_by_id("BRAKE-MOCK-001")["test_condition_type"], "STANDARD")

    def test_domain_provenance_survives_lookup(self):
        cases = [
            ("transmission", "TRANS-MOCK-001", "TRANSMISSION", "NOT_APPLICABLE", "TRUE"),
            ("axle_hubs", "AXLE-MOCK-001", "AXLE", "FRONT", "UNKNOWN"),
            ("parasitic", "PARA-MOCK-001", "OTHER_RESIDUAL_COMPONENT_LOSSES", "UNKNOWN", "UNKNOWN"),
        ]
        for domain, component_id, component_type, position, net_bridge in cases:
            with self.subTest(domain=domain):
                result = lookup_component(domain, component_id)
                component = result["component"]

                self.assertTrue(result["found"])
                self.assertEqual(component["component_type"], component_type)
                self.assertEqual(component["component_position"], position)
                self.assertEqual(component["net_bridge_eligible"], net_bridge)

    def test_lookup_component_missing_preserves_requested_id(self):
        result = lookup_component("parasitic", "PARA-UNKNOWN")

        self.assertFalse(result["found"])
        self.assertEqual(result["component_id"], "PARA-UNKNOWN")
        self.assertEqual(result["issues"][0]["severity"], "missing")

    def test_duplicate_ids_are_detected(self):
        repo = ComponentRepository(
            domain="transmission",
            source="unit_test",
            _components=[
                {"component_id": "DUP", "component_name": "One", "status": "mock", "source": "test", "notes": "-", "trans_A": 1.0, "trans_B": 0.1, "trans_C": 0.01, "loss_pct": 1.0},
                {"component_id": "DUP", "component_name": "Two", "status": "mock", "source": "test", "notes": "-", "trans_A": 2.0, "trans_B": 0.2, "trans_C": 0.02, "loss_pct": 2.0},
            ],
            _issues=[
                {
                    "code": "duplicate_component_id",
                    "severity": "error",
                    "domain": "transmission",
                    "component_id": "DUP",
                    "field_key": "component_id",
                    "message": "Duplicate component_id 'DUP' detected in unit_test.",
                }
            ],
            _by_id={"DUP": {"component_id": "DUP"}},
        )

        self.assertTrue(any(item["code"] == "duplicate_component_id" for item in repo.issues))

    def test_validate_component_detects_invalid_technical_data(self):
        repo = load_mock_component_repository("axle_hubs")
        issues = repo.validate_component(
            {
                "component_id": "BAD-1",
                "component_name": "Bad",
                "status": "mock",
                "source": "test",
                "notes": "bad",
                "axle_hubs_A": "oops",
                "axle_hubs_B": 0.1,
                "axle_hubs_C": 0.01,
            }
        )

        self.assertTrue(any(item["code"] == "invalid_numeric_field" for item in issues))

    @patch("src.vde_core.component_repositories.get_tire_by_code")
    @patch("src.vde_core.component_repositories.get_tire_by_id")
    def test_tire_lookup_uses_existing_service_adapter(self, mock_get_tire_by_id, mock_get_tire_by_code):
        mock_get_tire_by_id.return_value = None
        mock_get_tire_by_code.return_value = {"id": 77, "tire_test_code": "MOCK-TIRE", "rr_n_per_kn": 9.4}

        result = lookup_component("tire", "MOCK-TIRE")

        self.assertTrue(result["found"])
        self.assertEqual(result["source"], "tire_service")
        mock_get_tire_by_code.assert_called_once_with("MOCK-TIRE")

    def test_create_component_writes_csv_atomically(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "transmission_components_mock.csv"
            with target.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["component_id", "component_name", "status", "source", "notes", "trans_A", "trans_B", "trans_C", "loss_pct"])
                writer.writeheader()
                writer.writerow(
                    {
                        "component_id": "TRANS-MOCK-BASE",
                        "component_name": "Base",
                        "status": "mock",
                        "source": "seed",
                        "notes": "-",
                        "trans_A": 1.0,
                        "trans_B": 0.1,
                        "trans_C": 0.01,
                        "loss_pct": 2.0,
                    }
                )
            with patch("src.vde_core.component_repositories._DATA_DIR", root):
                created = create_component(
                    "transmission",
                    {
                        "component_name": "Created",
                        "trans_A": 3.0,
                        "trans_B": 0.3,
                        "trans_C": 0.03,
                        "loss_pct": 4.0,
                    },
                )

                with target.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[-1]["component_id"], created["component_id"])

    def test_create_component_blocks_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "parasitic_components_mock.csv"
            with target.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["component_id", "component_name", "status", "source", "notes", "parasitic_A", "parasitic_B", "parasitic_C"])
                writer.writeheader()
                writer.writerow(
                    {
                        "component_id": "PARA-USER-DUP",
                        "component_name": "Base",
                        "status": "mock",
                        "source": "seed",
                        "notes": "-",
                        "parasitic_A": 1.0,
                        "parasitic_B": 0.1,
                        "parasitic_C": 0.01,
                    }
                )
            with patch("src.vde_core.component_repositories._DATA_DIR", root):
                with self.assertRaises(ValueError):
                    create_component(
                        "parasitic",
                        {
                            "component_id": "PARA-USER-DUP",
                            "component_name": "Duplicate",
                            "parasitic_A": 3.0,
                            "parasitic_B": 0.3,
                            "parasitic_C": 0.03,
                        },
                    )


if __name__ == "__main__":
    unittest.main()
