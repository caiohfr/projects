import unittest
import copy

import pandas as pd

from src.vde_core.nearest_peers import (
    build_peer_analysis_for_request,
    build_target_peer_row,
    classify_peer_group_quality,
    find_nearest_peers,
    generate_investigation_hints,
    summarize_peer_comparison,
)


class NearestPeersTests(unittest.TestCase):
    def _candidate_df(self):
        return pd.DataFrame(
            [
                {
                    "vde_id": 100,
                    "make": "AUDI",
                    "model": "A1",
                    "year": 2025,
                    "category": "B",
                    "electrification": "ICE",
                    "fuel_type": "Gasoline",
                    "transmission_type": "AT",
                    "drive_type": "FWD",
                    "mass_kg": 1450,
                    "engine_size_l": 1.5,
                    "engine_max_power_kw": 110,
                    "gear_count": 6,
                    "final_drive_ratio": 3.9,
                    "coast_A_N": 120,
                    "coast_B_N_per_kph": 0.03,
                    "coast_C_N_per_kph2": 0.0002,
                    "vde_total_mj_per_km": 1.92,
                    "vde_net_mj_per_km": 1.70,
                    "fuel_l_per_100km": 6.4,
                    "gco2_per_km": 148.0,
                    "fuel_ftp75_l_per_100km": 7.0,
                    "fuel_hwfet_l_per_100km": 5.8,
                },
                {
                    "vde_id": 101,
                    "make": "AUDI",
                    "model": "A3",
                    "year": 2024,
                    "category": "B",
                    "electrification": "ICE",
                    "fuel_type": "Gasoline",
                    "transmission_type": "AT",
                    "drive_type": "FWD",
                    "mass_kg": 1480,
                    "engine_size_l": 1.6,
                    "engine_max_power_kw": 115,
                    "gear_count": 6,
                    "final_drive_ratio": 4.0,
                    "coast_A_N": 122,
                    "coast_B_N_per_kph": 0.031,
                    "coast_C_N_per_kph2": 0.0002,
                    "vde_total_mj_per_km": 1.95,
                    "vde_net_mj_per_km": 1.73,
                    "fuel_l_per_100km": 6.7,
                    "gco2_per_km": 152.0,
                    "fuel_ftp75_l_per_100km": 7.2,
                    "fuel_hwfet_l_per_100km": 6.0,
                },
                {
                    "vde_id": 102,
                    "make": "VW",
                    "model": "GOLF",
                    "year": 2025,
                    "category": "B",
                    "electrification": "ICE",
                    "fuel_type": "Gasoline",
                    "transmission_type": "AT",
                    "drive_type": "FWD",
                    "mass_kg": 1500,
                    "engine_size_l": 1.4,
                    "engine_max_power_kw": 108,
                    "gear_count": 6,
                    "final_drive_ratio": 3.8,
                    "coast_A_N": 118,
                    "coast_B_N_per_kph": 0.029,
                    "coast_C_N_per_kph2": 0.0002,
                    "vde_total_mj_per_km": 1.88,
                    "vde_net_mj_per_km": 1.68,
                    "fuel_l_per_100km": 6.1,
                    "gco2_per_km": 143.0,
                    "fuel_ftp75_l_per_100km": 6.8,
                    "fuel_hwfet_l_per_100km": 5.6,
                },
                {
                    "vde_id": 103,
                    "make": "FORD",
                    "model": "FOCUS",
                    "year": 2023,
                    "category": "C",
                    "electrification": "ICE",
                    "fuel_type": "Gasoline",
                    "transmission_type": "MT",
                    "drive_type": "FWD",
                    "mass_kg": 1600,
                    "engine_size_l": 2.0,
                    "engine_max_power_kw": 130,
                    "gear_count": 6,
                    "final_drive_ratio": 4.2,
                    "coast_A_N": 135,
                    "coast_B_N_per_kph": 0.035,
                    "coast_C_N_per_kph2": 0.0003,
                    "vde_total_mj_per_km": 2.10,
                    "vde_net_mj_per_km": 1.85,
                    "fuel_l_per_100km": 8.0,
                    "gco2_per_km": 181.0,
                    "fuel_ftp75_l_per_100km": 8.3,
                    "fuel_hwfet_l_per_100km": 7.9,
                },
            ]
        )

    def test_find_nearest_peers_returns_ranked_candidates(self):
        target = {
            "make": "AUDI",
            "category": "B",
            "electrification": "ICE",
            "fuel_type": "Gasoline",
            "transmission_type": "AT",
            "drive_type": "FWD",
            "mass_kg": 1470,
            "engine_size_l": 1.5,
            "engine_max_power_kw": 112,
            "gear_count": 6,
            "final_drive_ratio": 3.95,
            "coast_A_N": 121,
            "coast_B_N_per_kph": 0.0305,
            "coast_C_N_per_kph2": 0.0002,
            "vde_total_mj_per_km": 1.93,
            "vde_net_mj_per_km": 1.71,
        }
        result = find_nearest_peers(target, self._candidate_df(), n=3)
        self.assertEqual(len(result["peers"]), 3)
        self.assertEqual(result["peers"][0]["vde_id"], 100)
        self.assertIn("mass_kg", result["feature_coverage"])

    def test_peer_summary_computes_dispersion_and_zscore(self):
        target = {
            "fuel_l_per_100km": 7.4,
            "gco2_per_km": 168.0,
            "vde_total_mj_per_km": 2.0,
            "vde_net_mj_per_km": 1.74,
        }
        peers = find_nearest_peers(target, self._candidate_df(), n=4)["peers"]
        summary = summarize_peer_comparison(target, peers)
        fuel_row = next(row for row in summary["metrics"] if row["metric"] == "fuel_l_per_100km")
        self.assertAlmostEqual(fuel_row["median"], 6.55, places=2)
        self.assertIsNotNone(fuel_row["std_dev"])
        self.assertIsNotNone(fuel_row["z_score"])

    def test_classify_peer_group_quality_handles_small_group(self):
        quality = classify_peer_group_quality(3, [0.1, 0.2])
        self.assertEqual(quality["label"], "Low confidence")

    def test_generate_investigation_hints_flags_fuel_and_transmission(self):
        target = {
            "fuel_l_per_100km": 7.9,
            "vde_total_mj_per_km": 2.05,
            "vde_net_mj_per_km": 1.80,
            "fuel_ftp75_l_per_100km": 7.3,
            "fuel_hwfet_l_per_100km": 7.8,
        }
        peers = find_nearest_peers(target, self._candidate_df(), n=4)["peers"]
        summary = summarize_peer_comparison(target, peers)
        hints = generate_investigation_hints(target, summary)
        hint_text = " ".join(item["hint"] for item in hints)
        self.assertIn("Fuel is worse than similar peers.", hint_text)
        self.assertIn("TOTAL is significantly above NET.", hint_text)

    def test_build_target_peer_row_maps_request_and_outputs(self):
        class DummyRequest:
            vde_id = 55

            def __init__(self):
                self.vehicle_features = {
                    "category": "B",
                    "make": "AUDI",
                    "model": "A1",
                    "year": 2025,
                    "engine_size_l": 1.5,
                    "transmission_type": "AT",
                    "drive_type": "FWD",
                    "electrification": "ICE",
                    "gear_count": 6,
                    "final_drive_ratio": 3.9,
                    "coast_A_N": 120,
                    "coast_B_N_per_kph": 0.03,
                    "coast_C_N_per_kph2": 0.0002,
                    "vde_total_mj_per_km": 1.92,
                    "vde_net_mj_per_km": 1.70,
                    "phase_outputs": {
                        "vde_urb_mj_per_km": 2.2,
                        "vde_hw_mj_per_km": 1.6,
                    },
                }
                self.powertrain_features = {"fuel_type": "Gasoline", "gear_count": 6}

        target = build_target_peer_row(
            DummyRequest(),
            outputs={"fuel_l_100km": 6.5, "gco2_km": 150.0},
        )
        self.assertEqual(target["vde_id"], 55)
        self.assertEqual(target["fuel_type"], "Gasoline")
        self.assertAlmostEqual(target["fuel_l_per_100km"], 6.5)
        self.assertAlmostEqual(target["vde_hw_mj_per_km"], 1.6)

    def test_build_peer_analysis_for_request_returns_summary_and_hints(self):
        class DummyRequest:
            vde_id = 55

            def __init__(self):
                self.vehicle_features = {
                    "legislation": "EPA",
                    "category": "B",
                    "make": "AUDI",
                    "model": "A1",
                    "year": 2025,
                    "engine_size_l": 1.5,
                    "transmission_type": "AT",
                    "drive_type": "FWD",
                    "electrification": "ICE",
                    "gear_count": 6,
                    "final_drive_ratio": 3.9,
                    "coast_A_N": 120,
                    "coast_B_N_per_kph": 0.03,
                    "coast_C_N_per_kph2": 0.0002,
                    "vde_total_mj_per_km": 1.92,
                    "vde_net_mj_per_km": 1.70,
                }
                self.powertrain_features = {"fuel_type": "Gasoline", "gear_count": 6}

        with unittest.mock.patch(
            "src.vde_core.nearest_peers.load_peer_candidates",
            return_value=self._candidate_df(),
        ):
            analysis = build_peer_analysis_for_request(
                DummyRequest(),
                outputs={"fuel_l_100km": 7.9, "gco2_km": 170.0},
                n=4,
            )
        self.assertEqual(analysis["summary"]["peer_count"], 4)
        self.assertIn("label", analysis["quality"])
        self.assertIsInstance(analysis["hints"], list)

    def test_build_peer_analysis_for_request_does_not_mutate_request_or_outputs(self):
        class DummyRequest:
            vde_id = 77

            def __init__(self):
                self.vehicle_features = {
                    "legislation": "EPA",
                    "category": "B",
                    "make": "AUDI",
                    "model": "A1",
                    "year": 2025,
                    "engine_size_l": 1.5,
                    "transmission_type": "AT",
                    "drive_type": "FWD",
                    "electrification": "ICE",
                    "gear_count": 6,
                    "final_drive_ratio": 3.9,
                    "coast_A_N": 120,
                    "coast_B_N_per_kph": 0.03,
                    "coast_C_N_per_kph2": 0.0002,
                    "vde_total_mj_per_km": 1.92,
                    "vde_net_mj_per_km": 1.70,
                }
                self.powertrain_features = {"fuel_type": "Gasoline", "gear_count": 6}

        request = DummyRequest()
        outputs = {"fuel_l_100km": 7.1, "gco2_km": 162.0}
        request_before = copy.deepcopy(request.vehicle_features), copy.deepcopy(request.powertrain_features)
        outputs_before = copy.deepcopy(outputs)

        with unittest.mock.patch(
            "src.vde_core.nearest_peers.load_peer_candidates",
            return_value=self._candidate_df(),
        ):
            analysis = build_peer_analysis_for_request(request, outputs=outputs, n=4)

        self.assertEqual(request.vehicle_features, request_before[0])
        self.assertEqual(request.powertrain_features, request_before[1])
        self.assertEqual(outputs, outputs_before)
        self.assertIsInstance(analysis["hints"], list)


if __name__ == "__main__":
    unittest.main()
