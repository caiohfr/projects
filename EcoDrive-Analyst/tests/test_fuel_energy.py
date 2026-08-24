from __future__ import annotations

import unittest

from src.vde_core.fuel_energy import (
    LHV_MJ_PER_L,
    FuelConfidence,
    LhvBasis,
    resolve_fuel_energy_basis,
)


class FuelEnergyBasisResolutionTests(unittest.TestCase):
    """Package 8F -- "never SILENTLY guess" replaces "never guess": an
    approved, deterministic, traceable assumption is preferred over hiding
    a result outright, but nothing here fabricates an LHV for a genuinely
    unknown or blended fuel.
    """

    def test_tier_2_cert_gasoline_resolves_to_gasoline_family_and_is_available(self):
        basis = resolve_fuel_energy_basis("Tier 2 Cert Gasoline")
        self.assertEqual(basis.canonical_fuel_family, "GASOLINE")
        self.assertEqual(basis.fuel_spec, "TIER_2_CERT_GASOLINE")
        self.assertTrue(basis.available)
        self.assertEqual(basis.lhv_mj_per_l, LHV_MJ_PER_L["Gasoline"])  # reuses canonical value, never a new one
        self.assertEqual(basis.lhv_basis, LhvBasis.CANONICAL_ASSUMPTION)
        self.assertEqual(basis.confidence, FuelConfidence.ASSUMED)
        self.assertIn("gasoline", basis.basis_label.lower())

    def test_tier_3_cert_gasoline_also_resolves_to_gasoline_family(self):
        basis = resolve_fuel_energy_basis("Tier 3 Cert Gasoline")
        self.assertEqual(basis.canonical_fuel_family, "GASOLINE")
        self.assertEqual(basis.fuel_spec, "TIER_3_CERT_GASOLINE")
        self.assertTrue(basis.available)
        self.assertEqual(basis.lhv_mj_per_l, LHV_MJ_PER_L["Gasoline"])

    def test_ordinary_gasoline_label_resolves_as_exact_spec_reference(self):
        basis = resolve_fuel_energy_basis("Gasoline")
        self.assertEqual(basis.canonical_fuel_family, "GASOLINE")
        self.assertTrue(basis.available)
        self.assertEqual(basis.lhv_mj_per_l, LHV_MJ_PER_L["Gasoline"])
        self.assertEqual(basis.lhv_basis, LhvBasis.SPEC_REFERENCE)
        self.assertEqual(basis.confidence, FuelConfidence.HIGH)

    def test_uppercase_gasoline_from_the_other_native_vocabulary_resolves_identically(self):
        # fuelcons_db.fuel_type can legitimately be populated from either of
        # two live, case-differing dropdowns (pwt_fuel_energy.FUEL_TYPE_OPTIONS
        # title-case, vs vde_request_metadata_options._CHOICE_OPTIONS
        # uppercase) -- a case-sensitive lookup previously made the second one
        # invisible to PSE/equi-PSE even though the family is unambiguous.
        basis = resolve_fuel_energy_basis("GASOLINE")
        self.assertEqual(basis.canonical_fuel_family, "GASOLINE")
        self.assertTrue(basis.available)
        self.assertEqual(basis.lhv_mj_per_l, LHV_MJ_PER_L["Gasoline"])
        self.assertEqual(basis.lhv_basis, LhvBasis.SPEC_REFERENCE)

    def test_diesel_and_ethanol_remain_their_own_distinct_families(self):
        diesel = resolve_fuel_energy_basis("DIESEL")
        ethanol = resolve_fuel_energy_basis("ETHANOL")
        self.assertEqual(diesel.canonical_fuel_family, "DIESEL")
        self.assertEqual(diesel.lhv_mj_per_l, LHV_MJ_PER_L["Diesel"])
        self.assertEqual(ethanol.canonical_fuel_family, "ETHANOL")
        self.assertEqual(ethanol.fuel_spec, "E100")
        self.assertEqual(ethanol.lhv_mj_per_l, LHV_MJ_PER_L["E100"])
        self.assertNotEqual(ethanol.canonical_fuel_family, "GASOLINE")

    def test_explicit_lhv_overrides_canonical_assumption(self):
        basis = resolve_fuel_energy_basis("Tier 2 Cert Gasoline", explicit_lhv_mj_per_l=31.4)
        self.assertEqual(basis.lhv_mj_per_l, 31.4)
        self.assertNotEqual(basis.lhv_mj_per_l, LHV_MJ_PER_L["Gasoline"])
        self.assertEqual(basis.lhv_basis, LhvBasis.EXPLICIT)
        self.assertEqual(basis.confidence, FuelConfidence.EXPLICIT)

    def test_unknown_fuel_label_returns_unavailable_never_fabricated(self):
        for label in ("Some Unrecognized Label", "Premium Unleaded XYZ", ""):
            with self.subTest(label=label):
                basis = resolve_fuel_energy_basis(label)
                self.assertFalse(basis.available)
                self.assertIsNone(basis.lhv_mj_per_l)
                self.assertEqual(basis.lhv_basis, LhvBasis.UNKNOWN)

    def test_none_label_returns_unavailable(self):
        basis = resolve_fuel_energy_basis(None)
        self.assertFalse(basis.available)
        self.assertIsNone(basis.lhv_mj_per_l)

    def test_flex_never_silently_becomes_gasoline(self):
        basis = resolve_fuel_energy_basis("Flex")
        self.assertFalse(basis.available)
        self.assertIsNone(basis.canonical_fuel_family)
        self.assertIsNone(basis.lhv_mj_per_l)
        self.assertEqual(basis.lhv_basis, LhvBasis.UNKNOWN)

    def test_electric_and_unsupported_non_liquid_fuels_are_unavailable(self):
        for label in ("Electric", "CNG", "LPG", "Hydrogen"):
            with self.subTest(label=label):
                basis = resolve_fuel_energy_basis(label)
                self.assertFalse(basis.available)

    def test_assumed_basis_is_exposed_in_returned_provenance(self):
        basis = resolve_fuel_energy_basis("Tier 2 Cert Gasoline")
        self.assertEqual(basis.raw_fuel_label, "Tier 2 Cert Gasoline")  # original label preserved, never overwritten
        self.assertEqual(basis.canonical_fuel_family, "GASOLINE")
        self.assertEqual(basis.fuel_spec, "TIER_2_CERT_GASOLINE")
        self.assertEqual(basis.lhv_basis, LhvBasis.CANONICAL_ASSUMPTION)
        self.assertEqual(basis.confidence, FuelConfidence.ASSUMED)
        self.assertTrue(basis.basis_label)  # human-readable string always present

    def test_reference_lhv_values_are_read_from_the_single_canonical_table(self):
        # Guards against ever reintroducing a second, conflicting Gasoline
        # LHV constant (derivatives.py=34.2 and plots.py's default=34.2 are
        # known, pre-existing, display-only duplicates -- out of scope here).
        self.assertEqual(LHV_MJ_PER_L["Gasoline"], 32.0)
        self.assertEqual(resolve_fuel_energy_basis("Gasoline").lhv_mj_per_l, LHV_MJ_PER_L["Gasoline"])
        self.assertEqual(resolve_fuel_energy_basis("Tier 2 Cert Gasoline").lhv_mj_per_l, LHV_MJ_PER_L["Gasoline"])


if __name__ == "__main__":
    unittest.main()
