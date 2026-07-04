import unittest

from src.vde_core.fuel_estimation import FuelEstimateRequest
from src.vde_core.ml_prediction import (
    _downgrade_ml_confidence,
    build_ml_features,
    get_ml_notebook_summary,
)


class MlPredictionFeatureTests(unittest.TestCase):
    def test_build_ml_features_prefers_powertrain_overrides(self):
        request = FuelEstimateRequest(
            vde_id=5038,
            vehicle_features={
                "category": "SUBCOMPACT CARS",
                "make": "AUDI",
                "year": 2027,
                "engine_size_l": None,
                "transmission_type": "AT",
                "drive_type": None,
                "electrification": "ICE",
                "gear_count": None,
                "final_drive_ratio": None,
                "coast_A_N": 145.16,
                "coast_B_N_per_kph": 0.09,
                "coast_C_N_per_kph2": 0.03,
                "vde_net_mj_per_km": 0.45,
                "phase_outputs": {
                    "vde_urb_mj_per_km": 0.47,
                    "vde_hw_mj_per_km": 0.42,
                },
            },
            powertrain_features={
                "engine_size_l": 1.984,
                "transmission_type": "OT",
                "drive_type": "FWD",
                "gear_count": 7,
                "final_drive_ratio": 4.17,
            },
        )

        features = build_ml_features(request)["features"]

        self.assertEqual(features["engine_size_l"], 1.984)
        self.assertEqual(features["transmission_type"], "OT")
        self.assertEqual(features["drive_type"], "FWD")
        self.assertEqual(features["gear_count"], 7)
        self.assertEqual(features["final_drive_ratio"], 4.17)

    def test_ml_notebook_summary_mentions_runtime_imputation(self):
        summary = get_ml_notebook_summary()
        self.assertIn("SimpleImputer", summary["preprocessing"]["pipeline"])

    def test_downgrade_ml_confidence_for_partial_domain_or_missing_features(self):
        self.assertEqual(
            _downgrade_ml_confidence("high", coverage_status="partial_domain", missing_features=[]),
            "medium",
        )
        self.assertEqual(
            _downgrade_ml_confidence("high", coverage_status="in_domain", missing_features=["gear_count"]),
            "medium",
        )
        self.assertEqual(
            _downgrade_ml_confidence("high", coverage_status="out_of_domain", missing_features=[]),
            "low",
        )


if __name__ == "__main__":
    unittest.main()
