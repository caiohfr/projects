import unittest
from unittest.mock import patch

from src.vde_app.components.vde_request_metadata_options import (
    METADATA_CUSTOM_MAKE_OPTION,
    METADATA_INHERIT_OPTION,
    metadata_category_options,
    metadata_choice_options,
    metadata_field_spec,
    metadata_make_options,
    metadata_override_value,
)


class VdeRequestMetadataOptionsTests(unittest.TestCase):
    def test_metadata_category_options_switch_by_legislation(self):
        self.assertIn("SMALL SUVS", metadata_category_options("EPA"))
        self.assertEqual(metadata_category_options("WLTP"), ["CLASS 1 (<850 KG)", "CLASS 2 (850-1220 KG)", "CLASS 3 (>1220 KG)"])

    def test_metadata_choice_options_reuse_legacy_select_lists(self):
        self.assertEqual(metadata_choice_options("electrification", legislation="EPA"), ["ICE", "HEV", "PHEV", "BEV"])
        self.assertEqual(metadata_choice_options("transmission_type", legislation="EPA"), ["AT", "AMT", "CVT", "MT", "OT"])
        self.assertEqual(metadata_choice_options("drive_type", legislation="EPA"), ["FWD", "RWD", "AWD", "4WD"])
        self.assertIn("HYDROGEN", metadata_choice_options("fuel_type", legislation="EPA"))

    def test_metadata_choice_options_append_current_value_when_unknown(self):
        self.assertIn("MHEV", metadata_choice_options("electrification", legislation="EPA", current_value="MHEV"))

    @patch("src.vde_app.components.vde_request_metadata_options.db_list_makes", return_value=["Volvo", "Audi"])
    def test_metadata_make_options_merge_db_defaults_and_custom(self, mock_db_list_makes):
        options = metadata_make_options(legislation="EPA", category="SMALL SUVS", current_value="Rivian")

        self.assertIn("VOLVO", options)
        self.assertIn("AUDI", options)
        self.assertIn("RIVIAN", options)
        self.assertEqual(options[-1], METADATA_CUSTOM_MAKE_OPTION)
        mock_db_list_makes.assert_called_once_with("EPA", "SMALL SUVS")

    def test_metadata_field_spec_marks_make_as_select_with_custom(self):
        spec = metadata_field_spec("make", legislation="EPA", category="SMALL SUVS", current_value="VOLVO")

        self.assertEqual(spec["widget"], "select")
        self.assertTrue(spec["allow_custom"])
        self.assertEqual(spec["options"][0], METADATA_INHERIT_OPTION)

    def test_metadata_override_value_normalizes_choice_fields_and_inherit(self):
        self.assertEqual(metadata_override_value("fuel_type", "gasoline"), "GASOLINE")
        self.assertEqual(metadata_override_value("make", "volvo"), "VOLVO")
        self.assertEqual(metadata_override_value("category", METADATA_INHERIT_OPTION), "")
        self.assertEqual(metadata_override_value("make", METADATA_CUSTOM_MAKE_OPTION, custom_value="Lucid"), "LUCID")


if __name__ == "__main__":
    unittest.main()
