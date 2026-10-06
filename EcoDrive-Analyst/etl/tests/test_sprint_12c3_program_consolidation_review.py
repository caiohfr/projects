from __future__ import annotations

import hashlib
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "etl" / "scripts"))

import sprint_12c3_program_consolidation_review as review  # noqa: E402


class Sprint12C3ProgramConsolidationReviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db_hash_before = hashlib.sha256(review.DB_PATH.read_bytes()).hexdigest()
        cls.payload = review.main()
        cls.db_hash_after = hashlib.sha256(review.DB_PATH.read_bytes()).hexdigest()

    def test_database_access_is_read_only_and_byte_identical(self) -> None:
        self.assertEqual(self.db_hash_before, self.db_hash_after)
        self.assertTrue(self.payload["database_byte_identical"])
        self.assertIn("mode=ro", self.payload["database_access"])
        self.assertEqual(self.payload["runtime_counts_read_only"], {"vde_db": 5_003, "fuelcons_db": 5_004})

    def test_deterministic_normalization_grouping_and_population_forecast(self) -> None:
        summary = self.payload["summary"]
        self.assertEqual(summary["source_rows"], 30_194)
        self.assertEqual(summary["raw_mmy_fallbacks"], 5_798)
        self.assertEqual(summary["normalized_mmy_fallbacks"], 5_742)
        self.assertEqual(summary["normalized_make_model_families"], 1_860)
        self.assertEqual(summary["cross_year_families"], 1_372)
        self.assertEqual(summary["year_like_make_rows_2026"], 19)
        combined = next(
            row for row in self.payload["program_population_forecast"]
            if row["population"] == "Combined EPA 2020-2026"
        )
        self.assertEqual(combined["safe_consolidation_count"], 2_895)
        self.assertEqual(combined["likely_semantic_program_range"], "2,214-2,895")
        self.assertEqual(combined["aggressive_lexical_lower_bound"], 1_860)

    def test_no_merge_is_caused_by_model_year_adjacency_alone(self) -> None:
        merged = [
            row for row in self.payload["program_boundary_candidates"]
            if row["consolidation_status"] in {"SAFE_CONSOLIDATE", "PROBABLE_CONSOLIDATE"}
        ]
        self.assertTrue(merged)
        self.assertTrue(all(row["shared_stable_arch_signatures"] > 0 for row in merged))
        safe = [row for row in merged if row["consolidation_status"] == "SAFE_CONSOLIDATE"]
        self.assertTrue(all(row["missing_model_years_between"] == 0 for row in safe))
        self.assertTrue(all(row["shared_source_vehicle_identities"] > 0 for row in safe))

    def test_boundary_and_consolidation_vocabularies_are_closed(self) -> None:
        boundary_values = {row["boundary_status"] for row in self.payload["program_boundary_candidates"]}
        self.assertTrue(boundary_values.issubset({
            "CONFIRMED_BOUNDARY", "STRONG_BOUNDARY_CANDIDATE",
            "WEAK_BOUNDARY_CANDIDATE", "NO_BOUNDARY_EVIDENCE", "UNRESOLVED",
        }))
        self.assertEqual(Counter(row["boundary_status"] for row in self.payload["program_boundary_candidates"]), {
            "NO_BOUNDARY_EVIDENCE": 3_513,
            "STRONG_BOUNDARY_CANDIDATE": 53,
            "WEAK_BOUNDARY_CANDIDATE": 278,
            "UNRESOLVED": 38,
        })
        consolidation_values = {
            row["consolidation_status"] for row in self.payload["program_consolidation_candidates"]
        }
        self.assertTrue(consolidation_values.issubset({
            "SAFE_CONSOLIDATE", "PROBABLE_CONSOLIDATE", "KEEP_SEPARATE", "UNRESOLVED",
        }))

    def test_program_parent_consolidation_preserves_all_child_populations(self) -> None:
        impact = self.payload["relationship_impact"]
        self.assertEqual(impact["result"], "PASS_NO_CHILD_ROW_LOSS")
        self.assertTrue(impact["fallback_programs_assigned_once"])
        self.assertTrue(impact["fallback_program_identity_set_preserved"])
        self.assertTrue(impact["all_rows_have_proposed_program_parent"])
        self.assertTrue(impact["configuration_identity_set_preserved"])
        self.assertTrue(impact["vde_identity_set_preserved"])
        self.assertTrue(impact["run_identity_set_preserved"])
        for name in ("source_rows", "vehicle_configurations", "vde_states", "run_candidates", "fuelcons_candidate_floor"):
            self.assertEqual(impact[f"{name}_before"], impact[f"{name}_after_parent_remap"])

    def test_null_is_not_coerced_to_zero(self) -> None:
        self.assertFalse(review.present(None))
        self.assertTrue(review.present(0))
        self.assertNotEqual(review.value_key(None), review.value_key(0))
        self.assertIsNone(review.row_signature(__import__("pandas").Series({"a": None}), ("a",), minimum_present=1))

    def test_example_selection_is_stable_and_covers_required_cases(self) -> None:
        examples = self.payload["program_examples"]
        self.assertGreaterEqual(len(examples), 25)
        self.assertEqual([row["example_id"] for row in examples], [f"EX-{index:02d}" for index in range(1, len(examples) + 1)])
        required = {
            "STABLE_ARCHITECTURE_2020_2026",
            "OBVIOUS_TECHNICAL_GENERATION_BREAK",
            "MULTIPLE_CONFIGURATIONS_ONE_LIKELY_PROGRAM",
            "COMMERCIAL_RENAME_OR_TRIM_NOMENCLATURE",
            "SAME_COMMERCIAL_NAME_DISTINCT_GENERATIONS",
            "ONE_YEAR_ONLY_MODEL",
            "MODEL_YEAR_GAP",
            "POWERTRAIN_FAMILY_BEV",
            "POWERTRAIN_FAMILY_ICE",
            "POWERTRAIN_FAMILY_HEV_OR_PHEV_NAME_SIGNAL",
            "INSUFFICIENT_SOURCE_DATA",
        }
        self.assertTrue(required.issubset({row["category"] for row in examples}))
        self.assertEqual(examples[0]["make"], "ACURA")
        self.assertEqual(examples[0]["model"], "RDX AWD")

    def test_forecast_separates_jrc_and_eea_from_epa(self) -> None:
        forecast = {row["population"]: row for row in self.payload["program_population_forecast"]}
        self.assertEqual(forecast["JRC source-scoped"]["safe_consolidation_count"], 249)
        self.assertEqual(forecast["JRC source-scoped"]["engineering_program_population"], "SOURCE_SCOPED_UNRESOLVED")
        self.assertEqual(forecast["EEA 2025 monitoring"]["fallback_upper_bound"], 0)
        self.assertEqual(forecast["EEA 2025 monitoring"]["engineering_program_population"], "NO")

    def test_outputs_are_scoped_to_etl(self) -> None:
        etl_root = (ROOT / "etl").resolve()
        for path in review.OUTPUT_PATHS:
            self.assertTrue(path.resolve().is_relative_to(etl_root), path)


if __name__ == "__main__":
    unittest.main()
