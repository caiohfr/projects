from __future__ import annotations

from copy import deepcopy
import unittest

from src.vde_app.components.vde_request_compact_units import (
    display_format_for_field,
    display_precision_for_field,
    display_unit_for_field,
    display_step_for_field,
    display_value_for_field,
    field_uses_display_units,
    format_display_value_for_field,
    format_select_option_for_field,
    format_value_map_for_display,
    quantity_kind_for_field,
    to_canonical_field_value,
)


class TestVdeRequestCompactUnits(unittest.TestCase):
    def test_quantity_mapping_covers_explicit_mass_fields(self):
        self.assertEqual(quantity_kind_for_field("mass_kg"), "mass")
        self.assertEqual(quantity_kind_for_field("current_curb_mass_kg"), "mass")
        self.assertEqual(quantity_kind_for_field("target_curb_mass_kg"), "mass")
        self.assertEqual(quantity_kind_for_field("target_class_min_kg"), "mass")
        self.assertEqual(quantity_kind_for_field("test_mass_kg"), "mass")
        self.assertEqual(quantity_kind_for_field("inertia_class"), "mass")

    def test_quantity_mapping_covers_force_pressure_and_rrc_fields(self):
        self.assertEqual(quantity_kind_for_field("A"), "force")
        self.assertEqual(quantity_kind_for_field("trans_A_coef_N"), "force")
        self.assertEqual(quantity_kind_for_field("B"), "force_per_speed")
        self.assertEqual(quantity_kind_for_field("C"), "force_per_speed_squared")
        self.assertEqual(quantity_kind_for_field("front_pressure_psi"), "pressure")
        self.assertEqual(quantity_kind_for_field("rrc_N_per_kN"), "rrc")

    def test_quantity_mapping_returns_none_for_non_physical_or_unknown_fields(self):
        self.assertIsNone(quantity_kind_for_field("twc_curb_position"))
        self.assertIsNone(quantity_kind_for_field("test_mass_basis"))
        self.assertIsNone(quantity_kind_for_field("mass_rule_status"))
        self.assertIsNone(quantity_kind_for_field("component_db_id"))

    def test_controlled_prefix_aliases_map_only_known_bases(self):
        self.assertEqual(quantity_kind_for_field("resolved_test_mass_kg"), "mass")
        self.assertEqual(quantity_kind_for_field("delta_A"), "force")
        self.assertEqual(quantity_kind_for_field("target_rrc_N_per_kN"), "rrc")
        self.assertIsNone(quantity_kind_for_field("baseline_mass_rule_status"))

    def test_mass_and_pressure_display_conversion(self):
        self.assertAlmostEqual(display_value_for_field("mass_kg", 1814.0, "US customary"), 3999.1854360359, places=6)
        self.assertEqual(display_value_for_field("mass_kg", 0.0, "US customary"), 0.0)
        self.assertAlmostEqual(display_value_for_field("front_pressure_psi", 35.0, "Metric"), 241.31650526095, places=6)
        self.assertEqual(display_value_for_field("front_pressure_psi", 0.0, "Metric"), 0.0)
        self.assertAlmostEqual(display_value_for_field("front_pressure_psi", 35.0, "Metric", pressure_unit="bar"), 2.4131650526095, places=9)

    def test_mass_force_b_c_cda_and_pressure_roundtrip_to_canonical(self):
        self.assertAlmostEqual(to_canonical_field_value("mass_kg", display_value_for_field("mass_kg", 1814.0, "US customary"), "US customary"), 1814.0, places=6)
        self.assertAlmostEqual(to_canonical_field_value("A", display_value_for_field("A", 100.0, "US customary"), "US customary"), 100.0, places=6)
        self.assertAlmostEqual(to_canonical_field_value("B", display_value_for_field("B", 0.02, "US customary"), "US customary"), 0.02, places=9)
        self.assertAlmostEqual(to_canonical_field_value("C", display_value_for_field("C", 0.01, "US customary"), "US customary"), 0.01, places=9)
        self.assertAlmostEqual(to_canonical_field_value("cda_m2", display_value_for_field("cda_m2", 0.62, "US customary"), "US customary"), 0.62, places=6)
        self.assertAlmostEqual(to_canonical_field_value("front_pressure_psi", display_value_for_field("front_pressure_psi", 35.0, "Metric"), "Metric"), 35.0, places=6)
        self.assertAlmostEqual(to_canonical_field_value("front_pressure_psi", display_value_for_field("front_pressure_psi", 35.0, "Metric", pressure_unit="bar"), "Metric", pressure_unit="bar"), 35.0, places=6)

    def test_force_and_rrc_display_conversion(self):
        self.assertAlmostEqual(display_value_for_field("A", 120.0, "US customary"), 26.9770732644, places=6)
        self.assertAlmostEqual(display_value_for_field("B", 0.02, "US customary"), 0.00723589849928, places=9)
        self.assertAlmostEqual(display_value_for_field("C", 0.01, "US customary"), 0.0058225249172, places=9)
        self.assertEqual(display_value_for_field("rrc_N_per_kN", 7.8318, "Metric"), 7.8318)
        self.assertEqual(display_value_for_field("rrc_N_per_kN", 7.8318, "US customary"), 7.8318)

    def test_rrc_zero_none_and_unknown_field_are_preserved(self):
        self.assertEqual(to_canonical_field_value("rrc_N_per_kN", 7.8318, "US customary"), 7.8318)
        self.assertEqual(to_canonical_field_value("mass_kg", 0.0, "US customary"), 0.0)
        self.assertIsNone(to_canonical_field_value("mass_kg", None, "Metric"))
        payload = {"value": 123.4}
        self.assertEqual(to_canonical_field_value("unknown_field", payload["value"], "US customary"), 123.4)
        self.assertEqual(payload, {"value": 123.4})

    def test_formatting_preserves_blank_and_zero(self):
        self.assertEqual(format_display_value_for_field("mass_kg", None, "Metric"), "\u2014")
        self.assertEqual(format_display_value_for_field("mass_kg", 0.0, "US customary"), "0")
        self.assertEqual(format_display_value_for_field("front_pressure_psi", 0.0, "Metric"), "0")

    def test_display_units_keep_rrc_canonical_and_convert_pressure(self):
        self.assertEqual(display_unit_for_field("rrc_N_per_kN", "Metric"), "N/kN")
        self.assertEqual(display_unit_for_field("rrc_N_per_kN", "US customary"), "N/kN")
        self.assertEqual(display_unit_for_field("front_pressure_psi", "Metric"), "kPa")
        self.assertEqual(display_unit_for_field("front_pressure_psi", "US customary"), "psi")
        self.assertEqual(display_unit_for_field("front_pressure_psi", "Metric", pressure_unit="bar"), "bar")

    def test_interval_formatting_converts_when_supported(self):
        self.assertEqual(
            format_display_value_for_field("target_twc_interval", "(1423, 1480] kg", "Metric"),
            "(1423, 1480] kg",
        )
        self.assertEqual(
            format_display_value_for_field("target_twc_interval", "(1423, 1480] kg", "US customary"),
            "(3137, 3263] lb",
        )

    def test_twc_select_option_label_changes_but_value_stays_canonical(self):
        canonical_option = 2041.0
        self.assertEqual(canonical_option, 2041.0)
        self.assertEqual(format_select_option_for_field("target_mass_kg", canonical_option, "Metric"), "2041 kg")
        self.assertEqual(format_select_option_for_field("target_mass_kg", canonical_option, "US customary"), "4500 lb")

    def test_display_step_and_unit_sensitive_flags_follow_quantity_rules(self):
        self.assertTrue(field_uses_display_units("mass_kg"))
        self.assertTrue(field_uses_display_units("front_pressure_psi"))
        self.assertFalse(field_uses_display_units("rrc_N_per_kN"))
        self.assertFalse(field_uses_display_units("wheel_radius_m"))
        self.assertEqual(display_step_for_field("mass_kg", 1.0, "Metric"), 1.0)
        self.assertEqual(display_step_for_field("mass_kg", 1.0, "US customary"), 1.0)
        self.assertEqual(display_step_for_field("custom_delta_kg", 1.0, "Metric"), 0.1)
        self.assertEqual(display_step_for_field("front_pressure_psi", 0.5, "Metric"), 1.0)
        self.assertEqual(display_step_for_field("front_pressure_psi", 0.5, "US customary"), 0.5)
        self.assertEqual(display_step_for_field("front_pressure_psi", 0.5, "Metric", pressure_unit="bar"), 0.05)

    def test_display_precision_and_format_follow_visual_unit_contract(self):
        self.assertEqual(display_precision_for_field("mass_kg", "Metric"), 0)
        self.assertEqual(display_precision_for_field("mass_kg", "US customary"), 0)
        self.assertEqual(display_precision_for_field("custom_delta_kg", "Metric"), 1)
        self.assertEqual(display_precision_for_field("front_pressure_psi", "Metric"), 0)
        self.assertEqual(display_precision_for_field("front_pressure_psi", "US customary"), 1)
        self.assertEqual(display_precision_for_field("front_pressure_psi", "Metric", pressure_unit="bar"), 2)
        self.assertEqual(display_precision_for_field("cda_m2", "Metric"), 3)
        self.assertEqual(display_precision_for_field("B", "Metric"), 4)
        self.assertEqual(display_precision_for_field("C", "Metric"), 6)
        self.assertEqual(display_precision_for_field("rrc_N_per_kN", "Metric"), 3)
        self.assertEqual(display_format_for_field("mass_kg", "%.1f", "Metric"), "%.0f")
        self.assertEqual(display_format_for_field("front_pressure_psi", "%.1f", "Metric"), "%.0f")
        self.assertEqual(display_format_for_field("front_pressure_psi", "%.1f", "US customary"), "%.1f")
        self.assertEqual(display_format_for_field("front_pressure_psi", "%.1f", "Metric", pressure_unit="bar"), "%.2f")
        self.assertEqual(display_format_for_field("cda_m2", "%.4f", "Metric"), "%.3f")
        self.assertEqual(display_format_for_field("rrc_N_per_kN", "%.2f", "Metric"), "%.3f")

    def test_pressure_local_override_roundtrip_keeps_same_canonical_payload(self):
        canonical_from_psi = to_canonical_field_value("front_pressure_psi", 39.0, "US customary", pressure_unit="psi")
        canonical_from_kpa = to_canonical_field_value("front_pressure_psi", 268.895, "Metric", pressure_unit="kPa")
        canonical_from_bar = to_canonical_field_value("front_pressure_psi", 2.68895, "Metric", pressure_unit="bar")

        self.assertAlmostEqual(canonical_from_psi, canonical_from_kpa, places=3)
        self.assertAlmostEqual(canonical_from_psi, canonical_from_bar, places=3)

    def test_pressure_local_override_formatting_and_map_render_use_selected_unit(self):
        rendered = format_display_value_for_field("front_pressure_psi", 38.0, "Metric", pressure_unit="bar")
        payload = format_value_map_for_display({"front_pressure_psi": 38.0, "rear_pressure_psi": 38.0}, "Metric", pressure_unit="bar")

        self.assertEqual(rendered, "2.62")
        self.assertIn("front_pressure_psi=2.62 bar", payload)
        self.assertIn("rear_pressure_psi=2.62 bar", payload)

    def test_render_toggle_roundtrip_does_not_introduce_canonical_drift(self):
        canonical = 1814.25
        us_display = display_value_for_field("mass_kg", canonical, "US customary")
        metric_display = display_value_for_field("mass_kg", to_canonical_field_value("mass_kg", us_display, "US customary"), "Metric")

        self.assertAlmostEqual(to_canonical_field_value("mass_kg", us_display, "US customary"), canonical, places=9)
        self.assertAlmostEqual(metric_display, canonical, places=9)

    def test_format_value_map_for_display_does_not_mutate_input(self):
        payload = {
            "target_curb_mass_kg": 1480.0,
            "inertia_class": 1588.0,
            "target_twc_interval": "(1423, 1480] kg",
            "mass_rule_status": "OK",
        }
        original = deepcopy(payload)

        rendered = format_value_map_for_display(payload, "US customary")

        self.assertEqual(payload, original)
        self.assertIn("target_curb_mass_kg=3263 lb", rendered)
        self.assertIn("inertia_class=3501 lb", rendered)
        self.assertIn("target_twc_interval=(3137, 3263] lb", rendered)
        self.assertIn("mass_rule_status=OK", rendered)


if __name__ == "__main__":
    unittest.main()
