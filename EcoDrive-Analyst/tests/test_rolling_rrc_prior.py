from __future__ import annotations

import unittest

from src.vde_core.rolling_rrc_prior import (
    DEFAULT_RRC_PRIOR_PATH,
    calculate_rolling_rrc_mass_delta,
    resolve_rolling_rrc_prior,
    resolve_transmission_rule_key,
)
from src.vde_core.services import load_vde_defaults


class RollingRrcPriorResolverTests(unittest.TestCase):
    def test_exact_table_match_uses_one_speed_override_and_csv_value(self):
        defaults = load_vde_defaults(DEFAULT_RRC_PRIOR_PATH)
        stored = defaults[
            defaults["category"].eq("Midsize Cars")
            & defaults["electrification"].eq("BEV")
            & defaults["transmission_type"].eq("SS")
        ].iloc[0]

        result = resolve_rolling_rrc_prior(
            {
                "category": "  Midsize   Cars ",
                "electrification": "BEV",
                "gears": 1,
                "transmission_type": "AUTOMATIC",
            }
        )

        self.assertEqual(result.lookup_category, "Midsize Cars")
        self.assertEqual(result.resolved_transmission_key, "SS")
        self.assertEqual(result.attempted_rule_key, "Midsize Cars|BEV|SS")
        self.assertEqual(result.matched_rule_key, "Midsize Cars|BEV|SS")
        self.assertEqual(result.rrc_n_per_kn, float(stored["rrc_N_per_kN"]))
        self.assertEqual(result.status, "APPLIED")

    def test_epa_category_vocabulary_is_not_rewritten_to_application_family(self):
        result = resolve_rolling_rrc_prior(
            {
                "category": "Midsize Cars",
                "electrification": "BEV",
                "gears": 1,
                "transmission_type": "automatic",
            }
        )
        self.assertEqual(result.lookup_category, "Midsize Cars")
        self.assertNotIn("PASSENGER_STANDARD", result.attempted_rule_key)

    def test_unknown_category_is_unavailable_without_global_fallback(self):
        result = resolve_rolling_rrc_prior(
            {
                "category": "Unknown",
                "electrification": "BEV",
                "gears": 1,
                "transmission_type": "automatic",
            }
        )
        self.assertEqual(result.status, "UNAVAILABLE")
        self.assertEqual(result.attempted_rule_key, "Unknown|BEV|SS")
        self.assertIsNone(result.matched_rule_key)
        self.assertIsNone(result.rrc_n_per_kn)
        self.assertIn("generic", result.reason.lower())

    def test_application_family_category_reports_incompatible_vocabulary(self):
        result = resolve_rolling_rrc_prior(
            {
                "category": "PASSENGER_STANDARD",
                "electrification": "BEV",
                "gears": 1,
                "transmission_type": "automatic",
            }
        )
        self.assertEqual(result.attempted_rule_key, "PASSENGER_STANDARD|BEV|SS")
        self.assertIsNone(result.matched_rule_key)
        self.assertEqual(result.status, "UNAVAILABLE")
        self.assertIn("incompatible", result.reason.lower())

    def test_linked_context_supplies_electrification_and_gears(self):
        result = resolve_rolling_rrc_prior(
            {"category": "Midsize Cars", "transmission_type": "AUTOMATIC"},
            linked_context={"electrification": "BEV", "gear_count": 1},
        )
        self.assertEqual(result.resolved_electrification, "BEV")
        self.assertEqual(result.electrification_source, "linked_context.electrification")
        self.assertEqual(result.gears, 1)
        self.assertEqual(result.resolved_transmission_key, "SS")
        self.assertEqual(result.status, "APPLIED")

    def test_blank_electrification_does_not_invent_ice(self):
        result = resolve_rolling_rrc_prior(
            {"category": "Midsize Cars", "transmission_type": "automatic", "model": "Unclassified"}
        )
        self.assertEqual(result.status, "UNAVAILABLE")
        self.assertIsNone(result.resolved_electrification)
        self.assertIn("electrification", result.missing_discriminants)

    def test_v04_i4_model_fallback_is_preserved(self):
        result = resolve_rolling_rrc_prior(
            {
                "category": "Midsize Cars",
                "model": "i4 eDrive40",
                "transmission_type": "AUTOMATIC",
                "gears": 1,
            }
        )
        self.assertEqual(result.resolved_electrification, "BEV")
        self.assertEqual(result.electrification_source, "V0.4_MODEL_I4_FALLBACK")
        self.assertEqual(result.status, "APPLIED")

    def test_transmission_rule_supports_canonical_and_legacy_labels(self):
        cases = {
            "manual": "MT",
            "continuously variable": "CVT",
            "DCT": "AMT",
            "automated manual": "AMT",
            "semi-automatic": "AMT",
            "TORQUE_CONVERTER_AUTOMATIC": "AT",
            "automatic": "AT",
            "SINGLE_SPEED_EV": "SS",
            "other": "OT",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(resolve_transmission_rule_key(raw), expected)
        self.assertEqual(resolve_transmission_rule_key("automatic", gears=1), "SS")

    def test_shared_physics_materializes_baseline_scenario_and_delta(self):
        result = calculate_rolling_rrc_mass_delta(8.0, 1600.0, 2000.0, 0.12)
        self.assertGreater(result.scenario_A_N, result.baseline_A_N)
        self.assertAlmostEqual(result.A_N, result.scenario_A_N - result.baseline_A_N, places=12)
        self.assertAlmostEqual(
            result.B_N_per_kph,
            result.scenario_B_N_per_kph - result.baseline_B_N_per_kph,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
