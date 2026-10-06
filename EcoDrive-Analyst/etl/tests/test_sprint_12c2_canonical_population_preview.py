from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "etl" / "scripts"))

import sprint_12c2_canonical_population_preview as preview  # noqa: E402


class Sprint12C2CanonicalPopulationPreviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db_hash_before = hashlib.sha256(preview.DB_PATH.read_bytes()).hexdigest()
        cls.payload = preview.main()
        cls.db_hash_after = hashlib.sha256(preview.DB_PATH.read_bytes()).hexdigest()

    def test_database_access_is_read_only_and_byte_identical(self) -> None:
        self.assertEqual(self.db_hash_before, self.db_hash_after)
        self.assertTrue(self.payload["database_byte_identical"])
        self.assertIn("mode=ro", self.payload["database_access"])

    def test_epa_grouping_counts_are_deterministic(self) -> None:
        summary = self.payload["epa2026_summary"]
        self.assertEqual(summary["source_rows"], 3_901)
        self.assertEqual(summary["raw_mmy_groups"], 784)
        self.assertEqual(summary["program_candidates"], 777)
        self.assertEqual(summary["configuration_candidates"], 1_291)
        self.assertEqual(summary["vde_candidates"], 1_485)
        self.assertEqual(summary["run_candidates"], 3_557)
        self.assertEqual(summary["configurations_with_multiple_vdes"], 194)
        self.assertEqual(summary["runs_with_multiple_vdes"], 1)

    def test_all_legacy_duplicate_vde_states_are_preserved(self) -> None:
        collisions = [
            row for row in self.payload["collision_report"]
            if row["collision_type"] == "LEGACY_DUPLICATED_MMY"
        ]
        self.assertEqual(len(collisions), 46)
        self.assertEqual(sum(row["classification"] == "KEEP_SEPARATE" for row in collisions), 44)
        self.assertEqual(sum(row["classification"] == "UNRESOLVED" for row in collisions), 2)

    def test_null_is_not_coerced_to_zero(self) -> None:
        self.assertIsNone(preview.clean(None))
        self.assertEqual(preview.clean(0), 0)
        self.assertNotEqual(preview.key_value(None), preview.key_value(0))

    def test_example_selection_is_stable_and_covers_required_shapes(self) -> None:
        examples = self.payload["population_examples"]
        self.assertGreaterEqual(len(examples), 20)
        self.assertEqual([row["example_id"] for row in examples], [f"EX-{i:02d}" for i in range(1, len(examples) + 1)])
        self.assertEqual(examples[0]["source_key"], "PREVIEW-VDE-59AA1262BFAECE67")
        required = {
            "ONE_MMY_ONE_CONFIG_ONE_VDE", "MMY_MULTIPLE_CONFIGS", "CONFIG_MULTIPLE_VDE_STATES",
            "MULTIPLE_SOURCE_ROWS_ONE_RUN", "MULTIPLE_RUNS_ONE_VDE", "LEGACY_DUPLICATE_MMY",
            "HISTORICAL_REFRESH", "DERIVED_SCENARIO_VDE", "ML_FUELCONS", "JRC_SOURCE_SCOPED",
            "EEA_MONITORING",
        }
        self.assertTrue(required.issubset({row["category"] for row in examples}))

    def test_outputs_are_scoped_to_etl_only(self) -> None:
        etl_root = (ROOT / "etl").resolve()
        for path in preview.OUTPUT_PATHS:
            self.assertTrue(path.resolve().is_relative_to(etl_root), path)

    def test_forecast_covers_all_nine_entities_and_separates_eea(self) -> None:
        forecast = {row["entity"]: row for row in self.payload["entity_count_forecast"]}
        self.assertEqual(set(forecast), {
            "PROGRAM", "VEHICLE_CONFIGURATION", "COMPONENT_DB", "TIRE_DB",
            "COMPONENT_INSTANCE", "COMPONENT_RESOLUTION", "VDE", "RUN", "FUELCONS",
        })
        self.assertEqual(forecast["VDE"]["eea_reporting_additive_range"], "0")
        self.assertIn("10,833,597", forecast["RUN"]["eea_reporting_additive_range"])


if __name__ == "__main__":
    unittest.main()
