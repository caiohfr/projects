import math
import unittest

from src.vde_core.roadload import (
    G_MPS2,
    KPA_PER_PSI,
    MPH_PER_KPH,
    N_PER_LBF,
    apply_tire_improvement,
    build_tire_component,
    calculate_axle_loads,
    calculate_axle_tire_abc_from_single,
    calculate_iso_tire_abc_for_single_tire,
    calculate_sae_smerf_rr_n_per_kn,
    calculate_sae_tire_abc_for_single_tire,
    calculate_single_tire_loads,
    calculate_vehicle_tire_abc,
    combine_front_rear_tire_abc,
)


class TireModelTests(unittest.TestCase):
    def test_calculate_axle_loads_uses_front_distribution_pct(self):
        loads = calculate_axle_loads(mass_kg=1000.0, front_weight_distribution_pct=60.0)

        expected_total = 1000.0 * G_MPS2
        self.assertTrue(math.isclose(loads["total_load_n"], expected_total, rel_tol=1e-9))
        self.assertTrue(math.isclose(loads["front_axle_load_n"], expected_total * 0.60, rel_tol=1e-9))
        self.assertTrue(math.isclose(loads["rear_axle_load_n"], expected_total * 0.40, rel_tol=1e-9))
        self.assertAlmostEqual(loads["rear_weight_distribution_pct"], 40.0)

    def test_calculate_single_tire_loads_divides_each_axle_by_two(self):
        loads = calculate_single_tire_loads(mass_kg=1000.0, front_weight_distribution_pct=60.0)

        self.assertTrue(
            math.isclose(loads["front_single_tire_load_n"], loads["front_axle_load_n"] / 2.0, rel_tol=1e-9)
        )
        self.assertTrue(
            math.isclose(loads["rear_single_tire_load_n"], loads["rear_axle_load_n"] / 2.0, rel_tol=1e-9)
        )
        self.assertTrue(
            math.isclose(loads["front_single_tire_load_kn"], loads["front_single_tire_load_n"] / 1000.0, rel_tol=1e-9)
        )

    def test_calculate_sae_tire_abc_for_single_tire_applies_scale_factor(self):
        tire = {
            "standard_family": "SAE",
            "sae_a": 0.01,
            "sae_b": 0.001,
            "sae_c": 0.0001,
            "sae_alpha": 1.0,
            "sae_beta": 0.0,
        }

        result = calculate_sae_tire_abc_for_single_tire(
            tire=tire,
            single_tire_load_n=3000.0,
            pressure_kpa=220.0,
        )

        expected_scale = 220.0
        self.assertAlmostEqual(result["scale_factor"], expected_scale)
        self.assertAlmostEqual(result["A"], expected_scale * 0.01)
        self.assertAlmostEqual(result["B"], expected_scale * 0.001)
        self.assertAlmostEqual(result["C"], expected_scale * 0.0001)

    def test_calculate_sae_tire_abc_converts_imperial_coefficients_to_internal_units(self):
        tire = {
            "standard_family": "SAE",
            "sae_a": 0.019,
            "sae_b": 0.00029,
            "sae_c": -2.67e-7,
            "sae_alpha": -0.673903152,
            "sae_beta": 1.1234,
            "sae_pressure_unit": "psi",
            "sae_load_unit": "lbf",
            "sae_speed_unit": "mph",
            "sae_force_unit": "lbf",
        }

        result = calculate_sae_tire_abc_for_single_tire(
            tire=tire,
            single_tire_load_n=475.0 * N_PER_LBF,
            pressure_kpa=35.0 * KPA_PER_PSI,
        )

        expected_scale = (35.0 ** tire["sae_alpha"]) * (475.0 ** tire["sae_beta"])
        self.assertAlmostEqual(result["pressure_model_value"], 35.0)
        self.assertAlmostEqual(result["load_model_value"], 475.0)
        self.assertAlmostEqual(result["A"], expected_scale * tire["sae_a"] * N_PER_LBF)
        self.assertAlmostEqual(result["B"], expected_scale * tire["sae_b"] * N_PER_LBF * MPH_PER_KPH)
        self.assertAlmostEqual(result["C"], expected_scale * tire["sae_c"] * N_PER_LBF * MPH_PER_KPH * MPH_PER_KPH)

    def test_calculate_sae_smerf_rr_n_per_kn_uses_weighted_city_highway_force(self):
        inputs = {
            "alpha": -0.673903152,
            "beta": 1.1234,
            "a": 0.019,
            "b": 0.00029,
            "c": -2.67e-7,
            "pressure_kpa": 35.0,
            "load_n": 475.0,
        }

        result = calculate_sae_smerf_rr_n_per_kn(**inputs)

        scale = (inputs["pressure_kpa"] ** inputs["alpha"]) * (inputs["load_n"] ** inputs["beta"])
        a_val = scale * inputs["a"]
        b_val = scale * inputs["b"]
        c_val = scale * inputs["c"]
        f_city = a_val + b_val * 34.04267 + c_val * 1818.112
        f_hwy = a_val + b_val * 77.67619 + c_val * 6297.445
        expected_rr = ((0.55 * f_city) + (0.45 * f_hwy)) * 1000.0 / inputs["load_n"]

        self.assertAlmostEqual(result["rr_n_per_kn"], expected_rr)
        self.assertAlmostEqual(result["smerf"], expected_rr)

    def test_calculate_iso_tire_abc_for_single_tire_maps_rr_to_constant_a(self):
        tire = {
            "standard_family": "ISO",
            "rr_n_per_kn": 10.5,
        }

        result = calculate_iso_tire_abc_for_single_tire(tire=tire, single_tire_load_n=3000.0)

        self.assertAlmostEqual(result["A"], 10.5 * 3.0)
        self.assertAlmostEqual(result["B"], 0.0)
        self.assertAlmostEqual(result["C"], 0.0)
        self.assertAlmostEqual(result["single_tire_load_kn"], 3.0)

    def test_apply_tire_improvement_handles_positive_and_negative_values(self):
        base = {"A": 100.0, "B": 1.0, "C": 0.1}

        improved = apply_tire_improvement(base, 5.0)
        worsened = apply_tire_improvement(base, -5.0)

        self.assertAlmostEqual(improved["A"], 95.0)
        self.assertAlmostEqual(improved["B"], 0.95)
        self.assertAlmostEqual(improved["C"], 0.095)
        self.assertAlmostEqual(worsened["A"], 105.0)
        self.assertAlmostEqual(worsened["B"], 1.05)
        self.assertAlmostEqual(worsened["C"], 0.105)

    def test_calculate_vehicle_tire_abc_same_iso_tire_yields_different_front_and_rear_by_load(self):
        tire = {
            "standard_family": "ISO",
            "rr_n_per_kn": 10.0,
        }
        result = calculate_vehicle_tire_abc(
            front_tire=tire,
            rear_tire=tire,
            inputs={
                "mass_kg": 1000.0,
                "front_weight_distribution_pct": 60.0,
                "tire_improvement_pct": 5.0,
            },
        )

        self.assertGreater(result["front"]["axle_abc"]["A"], result["rear"]["axle_abc"]["A"])
        self.assertAlmostEqual(result["front"]["axle_abc"]["B"], 0.0)
        self.assertAlmostEqual(result["rear"]["axle_abc"]["C"], 0.0)
        self.assertLess(result["total_final_abc"]["A"], result["total_base_abc"]["A"])
        self.assertAlmostEqual(result["tire_load_mass_used_kg"], 1000.0)
        self.assertAlmostEqual(result["applied_rr_n_per_kn"], 10.0)

    def test_calculate_vehicle_tire_abc_supports_mixed_sae_and_iso(self):
        front_tire = {
            "standard_family": "SAE",
            "sae_a": 0.01,
            "sae_b": 0.001,
            "sae_c": 0.0001,
            "sae_alpha": 1.0,
            "sae_beta": 0.0,
            "rr_n_per_kn": 9.5,
        }
        rear_tire = {
            "standard_family": "ISO",
            "rr_n_per_kn": 8.0,
        }

        result = calculate_vehicle_tire_abc(
            front_tire=front_tire,
            rear_tire=rear_tire,
            inputs={
                "mass_kg": 1200.0,
                "front_weight_distribution_pct": 55.0,
                "front_pressure_kpa": 210.0,
                "rear_pressure_kpa": 220.0,
                "tire_improvement_pct": 0.0,
            },
        )

        self.assertGreater(result["front"]["single_tire_abc"]["B"], 0.0)
        self.assertAlmostEqual(result["rear"]["single_tire_abc"]["B"], 0.0)
        self.assertAlmostEqual(
            result["total_base_abc"]["A"],
            result["front"]["axle_abc"]["A"] + result["rear"]["axle_abc"]["A"],
        )

    def test_helpers_can_build_component_from_combined_abc(self):
        front = {"A": 10.0, "B": 1.0, "C": 0.1}
        rear = {"A": 8.0, "B": 0.8, "C": 0.08}
        total = combine_front_rear_tire_abc(front, rear)
        component = build_tire_component("tire", total, meta={"kind": "preview"})

        self.assertEqual(component.name, "tire")
        self.assertAlmostEqual(component.A, 18.0)
        self.assertAlmostEqual(component.B, 1.8)
        self.assertAlmostEqual(component.C, 0.18)
        self.assertEqual(component.meta["kind"], "preview")

    def test_calculate_axle_tire_abc_from_single_multiplies_by_tire_count(self):
        single = {"A": 5.0, "B": 0.5, "C": 0.05}
        axle = calculate_axle_tire_abc_from_single(single, tire_count=2)

        self.assertAlmostEqual(axle["A"], 10.0)
        self.assertAlmostEqual(axle["B"], 1.0)
        self.assertAlmostEqual(axle["C"], 0.1)


if __name__ == "__main__":
    unittest.main()
