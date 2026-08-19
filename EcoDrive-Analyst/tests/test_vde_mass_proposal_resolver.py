import unittest

from src.vde_core.vde_mass_proposal_resolver import curb_mass_for_twc_position, resolve_mass_proposal
from src.vde_core.vde_request_detail_mapping import detail_key_for_domain_field


class TestVdeMassProposalResolver(unittest.TestCase):

    def test_canonical_mass_contract_separates_vde_and_tire_for_regulatory_and_combined_mass(self):
        epa = resolve_mass_proposal(
            {"mass_kg": 1500.0, "legislation": "EPA"},
            "EPA_CURB_TO_TWC",
            {"mass_kg": 1310.0},
        )["resolved_snapshot"]
        self.assertEqual(epa["vde_mass_basis"], "EPA_TWC")
        self.assertEqual(epa["vde_calculation_mass_kg"], epa["inertia_class"])
        self.assertEqual(epa["tire_load_mass_basis"], "TWC")
        self.assertEqual(epa["tire_load_mass_used_kg"], epa["inertia_class"])

        gcwr = resolve_mass_proposal(
            {"mass_kg": 1500.0},
            "GCWR",
            {"gcwr_kg": 2500.0, "trailer_mass_kg": 500.0, "trailer_A": 1.0, "trailer_B": 0.1, "trailer_C": 0.01},
        )["resolved_snapshot"]
        self.assertEqual(gcwr["vde_calculation_mass_kg"], 2500.0)
        self.assertEqual(gcwr["tire_load_mass_used_kg"], 2000.0)
        self.assertEqual(gcwr["payload_kg"], 500.0)

    def test_gcwr_requires_complete_trailer_curve(self):
        result = resolve_mass_proposal(
            {"mass_kg": 1500.0}, "GCWR", {"gcwr_kg": 2500.0, "trailer_mass_kg": 500.0, "trailer_A": 1.0}
        )
        self.assertEqual(result["status"], "Missing")
        self.assertEqual(result["resolved_snapshot"]["trailer_roadload_status"], "Missing")
    def test_epa_status_uses_effective_inertia_class(self):
        result = resolve_mass_proposal({"mass_kg": 1848.0, "inertia_class": 1928.0}, "EPA_STATUS", {})

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["resolved_snapshot"]["mass_kg"], 1848.0)
        self.assertIsNone(result["resolved_snapshot"]["test_mass_kg"])
        self.assertIsNone(result["resolved_snapshot"]["test_mass_basis"])

    def test_epa_curb_to_twc_preserves_exact_curb_and_resolves_class(self):
        result = resolve_mass_proposal({"mass_kg": 1423.0}, "EPA_CURB_TO_TWC", {"mass_kg": 1480.0})

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["resolved_snapshot"]["mass_kg"], 1480.0)
        self.assertEqual(result["resolved_snapshot"]["target_curb_mass_kg"], 1480.0)
        self.assertEqual(result["resolved_snapshot"]["inertia_class"], 1588.0)
        self.assertEqual(result["resolved_snapshot"]["test_mass_kg"], 1616.0)
        self.assertEqual(result["resolved_snapshot"]["test_mass_basis"], "PHYSICAL_TEST_MASS")
        self.assertEqual(result["resolved_snapshot"]["target_twc_interval"], "(1423, 1480] kg")
        self.assertEqual(result["resolved_snapshot"]["current_curb_mass_kg"], 1423.0)

    def test_epa_curb_to_twc_prioritizes_canonical_mass_kg_over_legacy_aliases(self):
        result = resolve_mass_proposal(
            {"mass_kg": 1500.0, "inertia_class": 1644.0},
            "EPA_CURB_TO_TWC",
            {"mass_kg": 1222.0, "target_curb_mass_kg": 1480.0, "curb_mass_kg": None},
        )

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["resolved_snapshot"]["mass_kg"], 1222.0)
        self.assertEqual(result["resolved_snapshot"]["inertia_class"], 1361.0)
        self.assertNotEqual(result["resolved_snapshot"]["inertia_class"], 1644.0)

    def test_epa_curb_to_twc_accepts_legacy_target_when_canonical_mass_is_absent(self):
        result = resolve_mass_proposal(
            {"mass_kg": 1500.0, "inertia_class": 1644.0},
            "EPA_CURB_TO_TWC",
            {"curb_mass_kg": None, "target_curb_mass_kg": 1222.0},
        )

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["resolved_snapshot"]["mass_kg"], 1222.0)
        self.assertEqual(result["resolved_snapshot"]["inertia_class"], 1361.0)

    def test_epa_curb_to_twc_accepts_explicit_test_mass_override(self):
        result = resolve_mass_proposal(
            {"mass_kg": 1423.0, "test_mass_kg": 1531.0},
            "EPA_CURB_TO_TWC",
            {"target_curb_mass_kg": 1480.0, "test_mass_kg": 1700.0},
        )

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["resolved_snapshot"]["mass_kg"], 1480.0)
        self.assertEqual(result["resolved_snapshot"]["inertia_class"], 1588.0)
        self.assertEqual(result["resolved_snapshot"]["test_mass_kg"], 1700.0)
        self.assertEqual(result["resolved_snapshot"]["test_mass_basis"], "PHYSICAL_TEST_MASS")

    def test_epa_curb_to_twc_lower_bound_is_exclusive(self):
        result = resolve_mass_proposal({}, "EPA_CURB_TO_TWC", {"mass_kg": 1423.0})

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["resolved_snapshot"]["inertia_class"], 1531.0)
        self.assertEqual(result["resolved_snapshot"]["target_twc_interval"], "(1366, 1423] kg")

    def test_epa_curb_to_twc_nonpositive_mass_is_invalid(self):
        result = resolve_mass_proposal({}, "EPA_CURB_TO_TWC", {"mass_kg": 0.0})

        self.assertEqual(result["status"], "Invalid")
        self.assertEqual(result["resolved_snapshot"]["mass_rule_status"], "Invalid")

    def test_epa_curb_to_twc_missing_target_does_not_inherit_source_as_requested_value(self):
        result = resolve_mass_proposal({"mass_kg": 1848.0}, "EPA_CURB_TO_TWC", {})

        self.assertEqual(result["status"], "Missing")
        self.assertIsNone(result["resolved_snapshot"]["target_curb_mass_kg"])
        self.assertEqual(result["resolved_snapshot"]["current_curb_mass_kg"], 1848.0)

    def test_twc_shift_plus_one_uses_reference_class(self):
        result = resolve_mass_proposal({"inertia_class": 1928.0}, "MASS_TWC_SHIFT", {"shift_steps": 1.0, "target_side": "Up"})

        self.assertEqual(result["resolved_snapshot"]["target_mass_kg"], 2041.0)
        self.assertEqual(result["resolved_snapshot"]["test_mass_kg"], 2098.0)
        self.assertEqual(result["resolved_snapshot"]["mass_kg"], 1962.0)

    def test_twc_shift_down_aligns_curb_mass_to_target_class(self):
        result = resolve_mass_proposal(
            {"mass_kg": 2416.0, "test_mass_kg": 2495.0, "inertia_class": 2495.0},
            "MASS_TWC_SHIFT",
            {"shift_steps": -2.0},
        )

        self.assertEqual(result["resolved_snapshot"]["target_mass_kg"], 2268.0)
        self.assertEqual(result["resolved_snapshot"]["test_mass_kg"], 2325.0)
        self.assertEqual(result["resolved_snapshot"]["mass_kg"], 2189.0)

    def test_twc_shift_curb_position_changes_curb_not_target_class(self):
        top = resolve_mass_proposal({"inertia_class": 1588.0}, "MASS_TWC_SHIFT", {"shift_steps": +1.0, "curb_position": "Top"})
        mid = resolve_mass_proposal({"inertia_class": 1588.0}, "MASS_TWC_SHIFT", {"shift_steps": +1.0, "curb_position": "Mid"})
        bottom = resolve_mass_proposal({"inertia_class": 1588.0}, "MASS_TWC_SHIFT", {"shift_steps": +1.0, "curb_position": "Bottom"})

        self.assertEqual(top["resolved_snapshot"]["target_mass_kg"], 1644.0)
        self.assertEqual(mid["resolved_snapshot"]["target_mass_kg"], 1644.0)
        self.assertEqual(bottom["resolved_snapshot"]["target_mass_kg"], 1644.0)
        self.assertEqual(top["resolved_snapshot"]["mass_kg"], 1536.0)
        self.assertEqual(mid["resolved_snapshot"]["mass_kg"], 1508.5)
        self.assertEqual(bottom["resolved_snapshot"]["mass_kg"], 1481.0)

    def test_twc_shift_defaults_curb_position_to_top(self):
        result = resolve_mass_proposal({"inertia_class": 1588.0}, "MASS_TWC_SHIFT", {"shift_steps": +1.0})

        self.assertEqual(result["resolved_snapshot"]["curb_position"], "TOP")
        self.assertEqual(result["resolved_snapshot"]["mass_kg"], 1536.0)

    def test_curb_mass_for_twc_position_is_pure_and_canonical(self):
        self.assertEqual(curb_mass_for_twc_position(1644.0, "Top"), 1536.0)
        self.assertEqual(curb_mass_for_twc_position(1644.0, "Mid"), 1508.5)
        self.assertEqual(curb_mass_for_twc_position(1644.0, "Bottom"), 1481.0)

    def test_performance_curb_supports_presets_and_custom_delta(self):
        plus_100 = resolve_mass_proposal({}, "PERFORMANCE_CURB_MASS", {"mass_kg": 1500.0, "preset": "Curb +100 kg"})
        plus_300_lb = resolve_mass_proposal({}, "PERFORMANCE_CURB_MASS", {"mass_kg": 1500.0, "preset": "Curb +300 lb"})
        custom = resolve_mass_proposal({}, "PERFORMANCE_CURB_MASS", {"mass_kg": 1500.0, "preset": "Custom delta", "custom_delta_kg": 75.0})

        self.assertEqual(plus_100["resolved_snapshot"]["test_mass_kg"], 1600.0)
        self.assertAlmostEqual(plus_300_lb["resolved_snapshot"]["test_mass_kg"], 1636.1, places=6)
        self.assertEqual(custom["resolved_snapshot"]["test_mass_kg"], 1575.0)

    def test_gvwr_and_gcwr_compute_payload_and_vehicle_mass(self):
        gvwr = resolve_mass_proposal({"mass_kg": 1500.0}, "GVWR", {"gvwr_kg": 2400.0})
        gcwr = resolve_mass_proposal({"mass_kg": 1500.0, "gvwr_kg": 2400.0}, "GCWR", {"gcwr_kg": 3200.0, "trailer_mass_kg": 800.0, "trailer_A": 10.0, "trailer_B": 0.1, "trailer_C": 0.01})

        self.assertEqual(gvwr["resolved_snapshot"]["test_mass_kg"], 2400.0)
        self.assertEqual(gvwr["resolved_snapshot"]["payload_kg"], 900.0)
        self.assertEqual(gcwr["resolved_snapshot"]["vehicle_mass_at_gcwr"], 2400.0)
        self.assertEqual(gcwr["resolved_snapshot"]["trailer_roadload_status"], "OK")

    def test_shared_detail_mapping_covers_mass_fields(self):
        self.assertEqual(detail_key_for_domain_field("mass", "EPA_STATUS", "mass_kg"), "curb_mass_kg")
        self.assertEqual(detail_key_for_domain_field("mass", "EPA_CURB_TO_TWC", "mass_kg"), "mass_kg")
        self.assertEqual(detail_key_for_domain_field("mass", "MASS_TWC_SHIFT", "target_mass_kg"), "target_mass_kg")
        self.assertEqual(detail_key_for_domain_field("mass", "GVWR", "gvwr_kg"), "GVWR_kg")
        self.assertEqual(detail_key_for_domain_field("mass", "GCWR", "trailer_mass_kg"), "trailer_weight_kg")
        self.assertEqual(detail_key_for_domain_field("mass", "WLTP_MASS_LINE", "options_kg"), "optional_weight_kg")


if __name__ == "__main__":
    unittest.main()
