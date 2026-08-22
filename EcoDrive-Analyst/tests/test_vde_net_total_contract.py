import unittest

from src.vde_core.vde_net_total_contract import (
    VdeSemanticStatus,
    canonical_vde_read,
    classify_vde_row,
)


class VdeNetTotalContractTests(unittest.TestCase):
    # Case 1 -- TOTAL only.
    def test_total_only_row_has_no_fake_net(self):
        row = {
            "vde_total_mj_per_km": 0.55,
            "vde_net_mj_per_km": None,
            "record_origin": "VDE_SETUP",
        }
        self.assertEqual(classify_vde_row(row), VdeSemanticStatus.CANONICAL_TOTAL_ONLY)
        canonical = canonical_vde_read(row)
        self.assertEqual(canonical.total_mj_per_km, 0.55)
        self.assertIsNone(canonical.net_mj_per_km)
        self.assertFalse(canonical.net_available)

    # Case 2 -- TOTAL + legitimate NET.
    def test_total_and_net_row_preserves_both_values(self):
        row = {
            "vde_total_mj_per_km": 0.55,
            "vde_net_mj_per_km": 0.50,
            "record_origin": "VDE_SETUP",
        }
        self.assertEqual(classify_vde_row(row), VdeSemanticStatus.CANONICAL_TOTAL_AND_NET)
        canonical = canonical_vde_read(row)
        self.assertEqual(canonical.total_mj_per_km, 0.55)
        self.assertEqual(canonical.net_mj_per_km, 0.50)
        self.assertTrue(canonical.net_available)

    # Case 3 -- legacy value stored in NET.
    def test_legacy_origin_net_only_row_is_classified_as_legacy_swap(self):
        row = {
            "vde_total_mj_per_km": None,
            "vde_net_mj_per_km": 0.49,
            "record_origin": "LEGACY",
        }
        self.assertEqual(
            classify_vde_row(row), VdeSemanticStatus.LEGACY_TOTAL_IN_NET_FIELD
        )
        canonical = canonical_vde_read(row)
        self.assertIsNone(canonical.total_mj_per_km)
        self.assertIsNone(canonical.net_mj_per_km)
        self.assertFalse(canonical.net_available)

    def test_legacy_classification_is_independent_of_transmission_coefficients(self):
        # trans coefficients being present is not evidence the stored value
        # was ever derived by subtracting them -- see module docstring.
        row = {
            "vde_total_mj_per_km": None,
            "vde_net_mj_per_km": 0.49,
            "record_origin": "LEGACY",
            "trans_A_coef_N": 12.0,
            "trans_B_coef_Npkph": 0.01,
            "trans_C_coef_Npkph2": 0.001,
        }
        self.assertEqual(
            classify_vde_row(row), VdeSemanticStatus.LEGACY_TOTAL_IN_NET_FIELD
        )

    # Case 4 -- ambiguous historical row: no destructive correction.
    def test_non_legacy_origin_net_only_row_is_ambiguous_not_guessed(self):
        for origin in ("MANUAL", "IMPORTED_REFERENCE", "VDE_SETUP"):
            with self.subTest(origin=origin):
                row = {
                    "vde_total_mj_per_km": None,
                    "vde_net_mj_per_km": 0.49,
                    "record_origin": origin,
                }
                self.assertEqual(
                    classify_vde_row(row), VdeSemanticStatus.AMBIGUOUS_REVIEW
                )
                canonical = canonical_vde_read(row)
                self.assertIsNone(canonical.total_mj_per_km)
                self.assertIsNone(canonical.net_mj_per_km)
                self.assertFalse(canonical.net_available)

    def test_row_with_neither_value_is_invalid(self):
        row = {"vde_total_mj_per_km": None, "vde_net_mj_per_km": None, "record_origin": "LEGACY"}
        self.assertEqual(classify_vde_row(row), VdeSemanticStatus.INVALID)


if __name__ == "__main__":
    unittest.main()
