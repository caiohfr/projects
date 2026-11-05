import unittest

from src.vde_core.vde_calculation_mass import resolve_vde_calculation_mass
from src.vde_core.vehicle_demand.adapters import build_vehicle_demand_request


def _row(**overrides):
    row = {
        "id": 1,
        "make": "QA",
        "model": "MASS",
        "year": 2026,
        "legislation": "EPA",
        "mass_kg": 1531.0,
        "test_mass_kg": 1701.0,
        "test_mass_basis": "PHYSICAL_TEST_MASS",
        "inertia_class": 1814.0,
        "coast_A_N": 100.0,
        "coast_B_N_per_kph": 1.0,
        "coast_C_N_per_kph2": 0.01,
    }
    row.update(overrides)
    return row


class CanonicalVdeCalculationMassTests(unittest.TestCase):
    def test_explicit_canonical_mass_has_highest_precedence(self):
        resolved = resolve_vde_calculation_mass(
            _row(vde_calculation_mass_kg=1900.0, vde_mass_basis="CERTIFIED_VDE_MASS")
        )
        self.assertEqual((resolved.mass_kg, resolved.basis, resolved.source_field), (1900.0, "CERTIFIED_VDE_MASS", "vde_calculation_mass_kg"))

    def test_stored_test_mass_is_shared_with_vehicle_demand_adapter(self):
        row = _row()
        resolved = resolve_vde_calculation_mass(row)
        request = build_vehicle_demand_request(row)
        self.assertEqual((resolved.mass_kg, resolved.source_field), (1701.0, "test_mass_kg"))
        self.assertEqual(request.test_mass_kg, resolved.mass_kg)
        self.assertEqual(request.provenance["mass_source_field"], "test_mass_kg")

    def test_epa_inertia_basis_uses_inertia_class(self):
        row = _row(test_mass_basis="EPA_INERTIA_CLASS")
        resolved = resolve_vde_calculation_mass(row)
        request = build_vehicle_demand_request(row)
        self.assertEqual((resolved.mass_kg, resolved.source_field), (1814.0, "inertia_class"))
        self.assertEqual(request.test_mass_kg, 1814.0)

    def test_epa_does_not_override_valid_stored_test_mass_without_inertia_semantics(self):
        resolved = resolve_vde_calculation_mass(
            _row(test_mass_basis="EPA_ETW_DIRECT_SOURCE_CONVERTED_LB_TO_KG")
        )
        self.assertEqual((resolved.mass_kg, resolved.source_field), (1701.0, "test_mass_kg"))

    def test_curb_is_explicit_final_fallback(self):
        resolved = resolve_vde_calculation_mass(
            _row(legislation="WLTP", test_mass_kg=None, test_mass_basis=None, inertia_class=None)
        )
        self.assertEqual((resolved.mass_kg, resolved.basis, resolved.source_field), (1531.0, "CURB_FALLBACK", "mass_kg"))


if __name__ == "__main__":
    unittest.main()
