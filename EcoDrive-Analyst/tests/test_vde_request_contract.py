import math
import unittest

from src.vde_core.vde_request_contract import (
    FIELD_KEY_ALIASES,
    TEMPLATE_PROPOSAL_MAP,
    VISIBLE_TEMPLATE_PROPOSAL_LABELS,
    VDE_REQUEST_SCHEMA_VERSION,
    is_blank,
    normalize_domain,
    normalize_template_proposal_type,
    resolve_effective_baseline,
)


class TestVdeRequestContract(unittest.TestCase):
    def test_schema_version(self):
        self.assertEqual(VDE_REQUEST_SCHEMA_VERSION, "0.1")

    def test_is_blank_true_for_none_empty_and_nan(self):
        self.assertTrue(is_blank(None))
        self.assertTrue(is_blank(""))
        self.assertTrue(is_blank("   "))
        self.assertTrue(is_blank(math.nan))

    def test_is_blank_false_for_zero_and_booleans(self):
        self.assertFalse(is_blank(0))
        self.assertFalse(is_blank(0.0))
        self.assertFalse(is_blank("0"))
        self.assertFalse(is_blank(False))
        self.assertFalse(is_blank(True))

    def test_resolve_effective_baseline_prefers_correction(self):
        self.assertEqual(resolve_effective_baseline(10, 12), 12)
        self.assertEqual(resolve_effective_baseline("printed", "corrected"), "corrected")

    def test_resolve_effective_baseline_uses_printed_when_correction_blank(self):
        self.assertEqual(resolve_effective_baseline(10, None), 10)
        self.assertEqual(resolve_effective_baseline("printed", ""), "printed")

    def test_normalize_domain_handles_spacing_case_and_known_aliases(self):
        self.assertEqual(normalize_domain(" Axle & Hubs "), "axle_hubs")
        self.assertEqual(normalize_domain("PARASITIC LOSSES"), "parasitic")
        self.assertEqual(normalize_domain("Scenario / Context"), "scenario")
        self.assertEqual(normalize_domain("mass_aero"), "mass")
        self.assertEqual(normalize_domain(""), "")

    def test_normalize_template_proposal_type_supports_all_declared_template_entries(self):
        for domain_key, mapping in TEMPLATE_PROPOSAL_MAP.items():
            for template_label, expected in mapping.items():
                with self.subTest(domain=domain_key, template_label=template_label):
                    normalized = normalize_template_proposal_type(domain_key, template_label)
                    self.assertTrue(normalized["ok"])
                    self.assertEqual(normalized["domain"], domain_key)
                    self.assertEqual(normalized["template_label"], expected["template_label"])
                    self.assertEqual(normalized["proposal_type"], expected["proposal_type"])

    def test_normalize_template_proposal_type_tolerates_case_spacing_and_old_aliases(self):
        self.assertEqual(
            normalize_template_proposal_type("TRANSMISSION", "  absolute abc  ")["proposal_type"],
            "UPDATE_TRANS_DRAG_ABC",
        )
        self.assertEqual(
            normalize_template_proposal_type("mass", "epa+1 twc")["proposal_type"],
            "MASS_TWC_SHIFT",
        )
        self.assertEqual(
            normalize_template_proposal_type("aero", "absolute cd area")["proposal_type"],
            "AERO_ABSOLUTE_CDA",
        )
        self.assertEqual(
            normalize_template_proposal_type("mass", "performance curb mass")["template_label"],
            "Performance loaded mass",
        )
        self.assertEqual(
            normalize_template_proposal_type("mass", "epa status mass")["proposal_type"],
            "EPA_STATUS",
        )
        self.assertEqual(
            normalize_template_proposal_type("mass", "epa_curb_to_twc")["proposal_type"],
            "EPA_CURB_TO_TWC",
        )

    def test_blank_template_proposal_type_resolves_to_inherit(self):
        normalized = normalize_template_proposal_type("mass", "")
        self.assertTrue(normalized["ok"])
        self.assertEqual(normalized["proposal_type"], "INHERIT")
        self.assertEqual(normalized["mode"], "inherited")

    def test_absolute_and_delta_abc_generate_distinct_internal_submodes(self):
        transmission_absolute = normalize_template_proposal_type("transmission", "Absolute ABC")
        transmission_delta = normalize_template_proposal_type("transmission", "Delta ABC")
        self.assertEqual(transmission_absolute["proposal_type"], "UPDATE_TRANS_DRAG_ABC")
        self.assertEqual(transmission_delta["proposal_type"], "UPDATE_TRANS_DRAG_ABC")
        self.assertEqual(transmission_absolute["template_label"], "Absolute ABC")
        self.assertEqual(transmission_delta["template_label"], "Delta ABC")
        self.assertNotIn("change_mode", transmission_absolute.get("details", {}))
        self.assertNotIn("change_mode", transmission_delta.get("details", {}))

        brake_absolute = normalize_template_proposal_type("brake", "Absolute ABC")
        brake_delta = normalize_template_proposal_type("brake", "Delta ABC")
        self.assertEqual(brake_absolute["template_label"], "Absolute ABC")
        self.assertEqual(brake_delta["template_label"], "Delta ABC")
        self.assertNotIn("change_mode", brake_absolute.get("details", {}))
        self.assertNotIn("change_mode", brake_delta.get("details", {}))
        self.assertNotIn("method", brake_absolute.get("details", {}))

    def test_lookup_from_db_returns_structured_mapping(self):
        tire_lookup = normalize_template_proposal_type("tire", "Lookup from DB")
        self.assertTrue(tire_lookup["ok"])
        self.assertEqual(tire_lookup["proposal_type"], "TIRE_DB_LOOKUP")
        self.assertEqual(tire_lookup["template_label"], "Tire DB lookup")

        trans_lookup = normalize_template_proposal_type("transmission", "Lookup from DB")
        self.assertTrue(trans_lookup["ok"])
        self.assertEqual(trans_lookup["proposal_type"], "TRANS_METADATA_ONLY")
        self.assertFalse(trans_lookup["has_internal_equivalent"])

    def test_tire_legacy_aliases_map_to_target_final_rrc(self):
        manual_rrc = normalize_template_proposal_type("tire", "Manual RRC")
        legacy_smerf = normalize_template_proposal_type("tire", "TIRE_SMERF_RRC_CHANGE")

        self.assertTrue(manual_rrc["ok"])
        self.assertEqual(manual_rrc["proposal_type"], "TIRE_TARGET_RRC")
        self.assertEqual(manual_rrc["template_label"], "Target final RRC")
        self.assertTrue(legacy_smerf["ok"])
        self.assertEqual(legacy_smerf["proposal_type"], "TIRE_TARGET_RRC")
        self.assertEqual(legacy_smerf["template_label"], "Target final RRC")

    def test_not_used_returns_structured_mapping(self):
        normalized = normalize_template_proposal_type("brake", "Not used")
        self.assertTrue(normalized["ok"])
        self.assertEqual(normalized["proposal_type"], "BRAKE_NOT_USED")
        self.assertTrue(normalized["has_internal_equivalent"])

    def test_unknown_value_returns_structured_error_without_silent_fallback(self):
        normalized = normalize_template_proposal_type("transmission", "Some new thing")
        self.assertFalse(normalized["ok"])
        self.assertEqual(normalized["error"], "unknown_proposal_type")

    def test_unknown_domain_returns_structured_error(self):
        normalized = normalize_template_proposal_type("mystery", "Inherit")
        self.assertFalse(normalized["ok"])
        self.assertEqual(normalized["error"], "unknown_domain")

    def test_mass_ascii_arrow_label_normalizes_to_epa_curb_to_twc(self):
        normalized = normalize_template_proposal_type("mass", "Curb mass -> EPA TWC")
        self.assertTrue(normalized["ok"])
        self.assertEqual(normalized["proposal_type"], "EPA_CURB_TO_TWC")
        self.assertEqual(normalized["template_label"], "Curb mass → EPA TWC")

    def test_field_key_aliases_include_component_db_examples(self):
        self.assertIn("transmission_component_db_id", FIELD_KEY_ALIASES)
        self.assertIn("brake_component_db_id", FIELD_KEY_ALIASES)
        self.assertIn("axle_hubs_component_db_id", FIELD_KEY_ALIASES)
        self.assertIn("target_curb_mass_kg", FIELD_KEY_ALIASES)

    def test_visible_mass_labels_hide_legacy_epa_status_option(self):
        labels = list(VISIBLE_TEMPLATE_PROPOSAL_LABELS["mass"])

        self.assertIn("Curb mass → EPA TWC", labels)
        self.assertIn("TWC shift / target class", labels)
        self.assertIn("Performance loaded mass", labels)
        self.assertNotIn("Use current EPA ETW / TWC", labels)


if __name__ == "__main__":
    unittest.main()
