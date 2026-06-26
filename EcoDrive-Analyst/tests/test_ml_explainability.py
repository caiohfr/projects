import unittest

from src.vde_core.ml_explainability import compute_ml_explanation


class MlExplainabilityTests(unittest.TestCase):
    def test_compute_ml_explanation_returns_not_available_when_empty(self):
        result = compute_ml_explanation({})
        self.assertEqual(result["status"], "not_available")
        self.assertEqual(result["grouped_blocks"], [])

    def test_compute_ml_explanation_groups_by_engineering_blocks(self):
        result = compute_ml_explanation(
            {
                "vde_net_mj_per_km": 0.42,
                "coast_A_N": 0.18,
                "engine_size_l": -0.11,
                "gear_count": 0.07,
                "make": 0.03,
            }
        )
        self.assertEqual(result["status"], "available")
        blocks = {row["engineering_block"]: row for row in result["grouped_blocks"]}
        self.assertIn("Roadload / VDE", blocks)
        self.assertIn("Powertrain", blocks)
        self.assertIn("Transmission", blocks)
        self.assertIn("Brand / Model Residual", blocks)


if __name__ == "__main__":
    unittest.main()

