import math
import unittest
from unittest.mock import patch

from src.vde_core.tire_roadload_service import (
    PSI_TO_KPA,
    apply_tire_result_to_roadload_request,
    build_tire_component_from_result,
    get_tire_by_code,
    preview_tire_roadload_for_vde,
    preview_tire_roadload_from_row,
    resolve_tire_load_mass,
    save_tire_roadload_to_vde,
    summarize_tire_rr,
)


class TireRoadloadServiceTests(unittest.TestCase):
    @patch("src.vde_core.tire_roadload_service.get_tire_roadload_by_code")
    def test_get_tire_by_code_uses_tire_test_code_lookup(self, mock_get_tire_roadload_by_code):
        mock_get_tire_roadload_by_code.return_value = {
            "id": 7,
            "tire_test_code": "EPA-225-50R17-A",
            "rr_n_per_kn": 9.5,
        }

        tire = get_tire_by_code("EPA-225-50R17-A")

        self.assertEqual(tire["id"], 7)
        self.assertEqual(tire["tire_test_code"], "EPA-225-50R17-A")
        mock_get_tire_roadload_by_code.assert_called_once_with("EPA-225-50R17-A")

    def test_summarize_tire_rr_uses_iso_corrected_rrc_when_available(self):
        summary = summarize_tire_rr(
            {
                "calculation_mode": "ISO_28580",
                "iso_rrc_n_per_kn": 9.8,
                "iso_corrected_rrc_n_per_kn": 9.2,
            }
        )

        self.assertEqual(summary["standard_family"], "ISO")
        self.assertAlmostEqual(summary["rr_n_per_kn"], 9.2)
        self.assertEqual(summary["rr_quality"], "measured_or_corrected_iso")

    def test_summarize_tire_rr_preserves_reference_smerf_values_for_sae_cases(self):
        reference_cases = [
            {
                "size_code": "175/65R14",
                "sae_alpha": -0.673903152,
                "sae_beta": 1.1234,
                "sae_a": 0.0919,
                "sae_b": 0.00029,
                "sae_c": -2.67e-7,
                "test_load_value": 475.0,
                "test_pressure_value": 35.0,
                "smerf": 6.80,
            },
            {
                "size_code": "175/65R14",
                "sae_alpha": -0.628482945,
                "sae_beta": 1.0875,
                "sae_a": 0.1020,
                "sae_b": 0.00026,
                "sae_c": -2.69e-7,
                "test_load_value": 475.0,
                "test_pressure_value": 35.0,
                "smerf": 7.08,
            },
            {
                "size_code": "205/50R17",
                "sae_alpha": -0.289177642,
                "sae_beta": 0.9498,
                "sae_a": 0.0406,
                "sae_b": 0.00014,
                "sae_c": -1.70e-7,
                "test_load_value": 580.0,
                "test_pressure_value": 36.0,
                "smerf": 6.28,
            },
            {
                "size_code": "205/50R17",
                "sae_alpha": -0.399206157,
                "sae_beta": 1.0122,
                "sae_a": 0.0419,
                "sae_b": 0.00015,
                "sae_c": -2.03e-7,
                "test_load_value": 580.0,
                "test_pressure_value": 36.0,
                "smerf": 5.92,
            },
        ]

        for case in reference_cases:
            with self.subTest(size_code=case["size_code"], smerf=case["smerf"]):
                summary = summarize_tire_rr(
                    {
                        "calculation_mode": "SAE_J2452",
                        "pressure_unit": "psi",
                        "load_unit": "kg",
                        **case,
                    }
                )

                self.assertEqual(summary["standard_family"], "SAE")
                self.assertEqual(summary["rr_method"], "SAE_J2452_SMERF_EPA_55_45")
                self.assertEqual(summary["rr_quality"], "reference_smerf_force_input")
                self.assertAlmostEqual(summary["smerf"], case["smerf"], places=2)
                self.assertAlmostEqual(
                    summary["rr_n_per_kn"],
                    case["smerf"] * 1000.0 / (case["test_load_value"] * 9.80665),
                )

    def test_summarize_tire_rr_preserves_distinct_reference_smerf_and_rrc_values_for_sae_cases(self):
        reference_pairs = [
            {"config": "Config 1", "test_weight_lbf": 5250.0, "smerf": 6.88, "rr_n_per_kn": 7.13},
            {"config": "Config 2", "test_weight_lbf": 5250.0, "smerf": 6.81, "rr_n_per_kn": 7.04},
            {"config": "Config 3", "test_weight_lbf": 5250.0, "smerf": 6.85, "rr_n_per_kn": 7.08},
            {"config": "Config 4", "test_weight_lbf": 5500.0, "smerf": 6.88, "rr_n_per_kn": 7.14},
            {"config": "Config 5", "test_weight_lbf": 5500.0, "smerf": 6.81, "rr_n_per_kn": 7.04},
            {"config": "Config 6", "test_weight_lbf": 5500.0, "smerf": 6.85, "rr_n_per_kn": 7.09},
            {"config": "Config 7", "test_weight_lbf": 6100.0, "smerf": 6.88, "rr_n_per_kn": 7.18},
            {"config": "Config 8", "test_weight_lbf": 6100.0, "smerf": 6.81, "rr_n_per_kn": 7.08},
            {"config": "Config 9", "test_weight_lbf": 6100.0, "smerf": 6.85, "rr_n_per_kn": 7.13},
        ]

        for row in reference_pairs:
            with self.subTest(config=row["config"]):
                summary = summarize_tire_rr(
                    {
                        "calculation_mode": "SAE_J2452",
                        "test_source": "VDE/PSE reference",
                        "test_load_value": row["test_weight_lbf"],
                        "load_unit": "lb",
                        "smerf": row["smerf"],
                        "rr_n_per_kn": row["rr_n_per_kn"],
                    }
                )

                self.assertEqual(summary["standard_family"], "SAE")
                self.assertEqual(summary["rr_method"], "SAE_J2452_SMERF_EPA_55_45")
                self.assertEqual(summary["rr_quality"], "reference_rr_and_smerf_input")
                self.assertAlmostEqual(summary["smerf"], row["smerf"], places=2)
                self.assertAlmostEqual(summary["rr_n_per_kn"], row["rr_n_per_kn"], places=2)

    def test_summarize_tire_rr_returns_missing_inputs_for_incomplete_sae_payload(self):
        summary = summarize_tire_rr(
            {
                "calculation_mode": "SAE_J2452",
                "sae_a": 0.01,
                "sae_b": 0.001,
                "sae_c": 0.0001,
            }
        )

        self.assertEqual(summary["standard_family"], "SAE")
        self.assertEqual(summary["rr_quality"], "missing_sae_inputs")
        self.assertIsNone(summary["smerf"])

    def test_resolve_tire_load_mass_prefers_test_mass_kg_for_test_mass(self):
        resolved = resolve_tire_load_mass(
            {"mass_kg": 1400.0, "test_mass_kg": 1550.0, "inertia_class": 1701.0},
            "TEST_MASS",
        )

        self.assertEqual(resolved["basis"], "TEST_MASS")
        self.assertAlmostEqual(resolved["mass_kg"], 1550.0)
        self.assertEqual(resolved["source_field"], "test_mass_kg")
        self.assertFalse(resolved["used_inertia_class"])
        self.assertFalse(resolved["test_mass_defaulted"])

    def test_resolve_tire_load_mass_defaults_epa_test_mass_from_curb_weight(self):
        resolved = resolve_tire_load_mass(
            {"legislation": "EPA", "mass_kg": 1400.0, "inertia_class": 1701.0},
            "TEST_MASS",
        )

        self.assertEqual(resolved["basis"], "TEST_MASS")
        self.assertAlmostEqual(resolved["mass_kg"], 1536.0)
        self.assertEqual(resolved["source_field"], "default_test_mass_kg_epa")
        self.assertFalse(resolved["used_inertia_class"])
        self.assertTrue(resolved["test_mass_defaulted"])
        self.assertEqual(resolved["test_mass_default_rule"], "EPA_CURB_PLUS_136KG")

    def test_resolve_tire_load_mass_twc_falls_back_to_inertia_class(self):
        resolved = resolve_tire_load_mass(
            {"mass_kg": 1550.0, "inertia_class": 1701.0},
            "TWC",
        )

        self.assertEqual(resolved["basis"], "TWC")
        self.assertAlmostEqual(resolved["mass_kg"], 1701.0)
        self.assertEqual(resolved["source_field"], "inertia_class")
        self.assertTrue(resolved["used_fallback"])
        self.assertTrue(resolved["used_inertia_class"])

    @patch("src.vde_core.tire_roadload_service.get_tire_roadload_by_id")
    @patch("src.vde_core.tire_roadload_service.get_vde_tire_application")
    @patch("src.vde_core.tire_roadload_service.fetch_vde_by_id")
    def test_preview_tire_roadload_for_vde_reuses_front_tire_when_same_tire_enabled(
        self,
        mock_fetch_vde_by_id,
        mock_get_vde_tire_application,
        mock_get_tire_roadload_by_id,
    ):
        mock_fetch_vde_by_id.return_value = {
            "id": 42,
            "legislation": "EPA",
            "mass_kg": 1500.0,
            "test_mass_kg": 1600.0,
            "inertia_class": 1644.0,
            "weight_dist_fr_pct": 60.0,
            "front_pressure_psi": 32.0,
            "rear_pressure_psi": 34.0,
            "tire_A_final": 10.0,
            "tire_B_final": 0.0,
            "tire_C_final": 0.0,
        }
        mock_get_vde_tire_application.return_value = {}
        mock_get_tire_roadload_by_id.return_value = {
            "id": 7,
            "standard_family": "SAE",
            "sae_a": 0.01,
            "sae_b": 0.001,
            "sae_c": 0.0001,
            "sae_alpha": 1.0,
            "sae_beta": 0.0,
            "rr_n_per_kn": 9.5,
        }

        preview = preview_tire_roadload_for_vde(
            42,
            {
                "front_tire_id": 7,
                "same_tire_front_rear": True,
                "tire_improvement_pct": 5.0,
                "tire_load_mass_basis": "TEST_MASS",
            },
        )

        self.assertEqual(preview["application"]["rear_tire_id"], 7)
        self.assertEqual(preview["application"]["rear_tire_source"], "same_tire_front_rear")
        self.assertAlmostEqual(preview["mass_resolution"]["mass_kg"], 1600.0)
        self.assertTrue(
            math.isclose(
                preview["calculation"]["front"]["single_tire_abc"]["pressure_kpa"],
                32.0 * PSI_TO_KPA,
                rel_tol=1e-9,
            )
        )
        self.assertLess(
            preview["calculation"]["total_final_abc"]["A"],
            preview["calculation"]["total_base_abc"]["A"],
        )
        self.assertEqual(preview["component_dict"]["name"], "tire")
        self.assertEqual(
            preview["save_payload"]["tire_calc_notes"],
            "basis=TEST_MASS; mass_source=test_mass_kg; mass_used_fallback=False; uses_inertia_class_mass=False; "
            "test_mass_defaulted=False; test_mass_default_rule=None; "
            "rear_tire_source=same_tire_front_rear; front_pressure_source=saved_front_pressure_psi; "
            "rear_pressure_source=saved_rear_pressure_psi; front_weight_distribution_pct=60.0; "
            "weight_dist_defaulted=False; front_standard=SAE; rear_standard=SAE",
        )
        self.assertIn("mass_source=test_mass_kg", preview["save_payload"]["tire_calc_notes"])
        self.assertIn("mass_used_fallback=False", preview["save_payload"]["tire_calc_notes"])
        self.assertIn("uses_inertia_class_mass=False", preview["save_payload"]["tire_calc_notes"])
        self.assertIn("rear_tire_source=same_tire_front_rear", preview["save_payload"]["tire_calc_notes"])
        self.assertAlmostEqual(preview["save_payload"]["rrc_N_per_kN"], 9.5)

    @patch("src.vde_core.tire_roadload_service.get_tire_roadload_by_id")
    @patch("src.vde_core.tire_roadload_service.get_vde_tire_application")
    @patch("src.vde_core.tire_roadload_service.fetch_vde_by_id")
    def test_preview_tire_roadload_for_vde_uses_payload_mass_overrides(
        self,
        mock_fetch_vde_by_id,
        mock_get_vde_tire_application,
        mock_get_tire_roadload_by_id,
    ):
        mock_fetch_vde_by_id.return_value = {
            "id": 42,
            "legislation": "EPA",
            "mass_kg": 1400.0,
            "test_mass_kg": 1536.0,
            "inertia_class": 1644.0,
            "weight_dist_fr_pct": 60.0,
            "front_pressure_psi": 32.0,
            "rear_pressure_psi": 34.0,
        }
        mock_get_vde_tire_application.return_value = {}
        mock_get_tire_roadload_by_id.return_value = {"id": 7, "standard_family": "ISO", "rr_n_per_kn": 9.5}

        preview = preview_tire_roadload_for_vde(
            42,
            {
                "front_tire_id": 7,
                "rear_tire_id": 7,
                "tire_load_mass_basis": "TWC",
                "mass_kg": 1550.0,
                "inertia_class": 1644.0,
                "twc_kg": 1644.0,
            },
        )

        self.assertEqual(preview["mass_resolution"]["basis"], "TWC")
        self.assertAlmostEqual(preview["mass_resolution"]["mass_kg"], 1701.0)
        self.assertEqual(preview["mass_resolution"]["source_field"], "twc_kg")

    @patch("src.vde_core.tire_roadload_service.get_tire_roadload_by_id")
    @patch("src.vde_core.tire_roadload_service.get_vde_tire_application")
    @patch("src.vde_core.tire_roadload_service.fetch_vde_by_id")
    def test_preview_tire_roadload_for_vde_tracks_pressure_source_from_kpa_payload(
        self,
        mock_fetch_vde_by_id,
        mock_get_vde_tire_application,
        mock_get_tire_roadload_by_id,
    ):
        mock_fetch_vde_by_id.return_value = {
            "id": 42,
            "mass_kg": 1500.0,
            "weight_dist_fr_pct": 60.0,
        }
        mock_get_vde_tire_application.return_value = {}
        mock_get_tire_roadload_by_id.return_value = {
            "id": 7,
            "standard_family": "SAE",
            "sae_a": 0.01,
            "sae_b": 0.001,
            "sae_c": 0.0001,
            "sae_alpha": 1.0,
            "sae_beta": 0.0,
            "rr_n_per_kn": 9.5,
        }

        preview = preview_tire_roadload_for_vde(
            42,
            {
                "front_tire_id": 7,
                "rear_tire_id": 7,
                "front_pressure_kpa": 220.0,
                "rear_pressure_kpa": 230.0,
            },
        )

        self.assertEqual(preview["application"]["front_pressure_source"], "front_pressure_kpa")
        self.assertEqual(preview["application"]["rear_pressure_source"], "rear_pressure_kpa")
        self.assertEqual(preview["application"]["rear_tire_source"], "rear_tire_id")
        self.assertIn("front_pressure_source=front_pressure_kpa", preview["save_payload"]["tire_calc_notes"])
        self.assertIn("rear_pressure_source=rear_pressure_kpa", preview["save_payload"]["tire_calc_notes"])
        self.assertIn("rear_tire_source=rear_tire_id", preview["save_payload"]["tire_calc_notes"])

    @patch("src.vde_core.tire_roadload_service.get_tire_roadload_by_id")
    @patch("src.vde_core.tire_roadload_service.get_vde_tire_application")
    @patch("src.vde_core.tire_roadload_service.fetch_vde_by_id")
    def test_preview_tire_roadload_for_vde_defaults_front_weight_distribution_when_missing(
        self,
        mock_fetch_vde_by_id,
        mock_get_vde_tire_application,
        mock_get_tire_roadload_by_id,
    ):
        mock_fetch_vde_by_id.return_value = {
            "id": 42,
            "mass_kg": 1500.0,
            "front_pressure_psi": 32.0,
            "rear_pressure_psi": 34.0,
        }
        mock_get_vde_tire_application.return_value = {}
        mock_get_tire_roadload_by_id.return_value = {"id": 7, "standard_family": "ISO", "rr_n_per_kn": 9.5}

        preview = preview_tire_roadload_for_vde(
            42,
            {
                "front_tire_id": 7,
                "rear_tire_id": 7,
            },
        )

        self.assertAlmostEqual(preview["application"]["front_weight_distribution_pct"], 50.0)
        self.assertTrue(preview["application"]["front_weight_distribution_pct_defaulted"])
        self.assertIn("front_weight_distribution_pct=50.0", preview["save_payload"]["tire_calc_notes"])
        self.assertIn("weight_dist_defaulted=True", preview["save_payload"]["tire_calc_notes"])

    @patch("src.vde_core.tire_roadload_service.get_tire_roadload_by_id")
    @patch("src.vde_core.tire_roadload_service.get_vde_tire_application")
    @patch("src.vde_core.tire_roadload_service.fetch_vde_by_id")
    def test_preview_tire_roadload_for_vde_tracks_saved_pressure_source(
        self,
        mock_fetch_vde_by_id,
        mock_get_vde_tire_application,
        mock_get_tire_roadload_by_id,
    ):
        mock_fetch_vde_by_id.return_value = {
            "id": 42,
            "mass_kg": 1500.0,
            "weight_dist_fr_pct": 60.0,
            "front_pressure_psi": 31.0,
            "rear_pressure_psi": 33.0,
        }
        mock_get_vde_tire_application.return_value = {}
        mock_get_tire_roadload_by_id.return_value = {"id": 7, "standard_family": "ISO", "rr_n_per_kn": 9.5}

        preview = preview_tire_roadload_for_vde(
            42,
            {
                "front_tire_id": 7,
                "rear_tire_id": 7,
            },
        )

        self.assertEqual(preview["application"]["front_pressure_source"], "saved_front_pressure_psi")
        self.assertEqual(preview["application"]["rear_pressure_source"], "saved_rear_pressure_psi")
        self.assertIn("front_pressure_source=saved_front_pressure_psi", preview["save_payload"]["tire_calc_notes"])
        self.assertIn("rear_pressure_source=saved_rear_pressure_psi", preview["save_payload"]["tire_calc_notes"])

    @patch("src.vde_core.tire_roadload_service.get_tire_roadload_by_id")
    @patch("src.vde_core.tire_roadload_service.get_vde_tire_application")
    @patch("src.vde_core.tire_roadload_service.fetch_vde_by_id")
    def test_preview_tire_roadload_for_vde_tracks_epa_test_mass_default(
        self,
        mock_fetch_vde_by_id,
        mock_get_vde_tire_application,
        mock_get_tire_roadload_by_id,
    ):
        mock_fetch_vde_by_id.return_value = {
            "id": 42,
            "legislation": "EPA",
            "mass_kg": 1500.0,
            "weight_dist_fr_pct": 60.0,
            "front_pressure_psi": 31.0,
            "rear_pressure_psi": 33.0,
        }
        mock_get_vde_tire_application.return_value = {}
        mock_get_tire_roadload_by_id.return_value = {"id": 7, "standard_family": "ISO", "rr_n_per_kn": 9.5}

        preview = preview_tire_roadload_for_vde(
            42,
            {
                "front_tire_id": 7,
                "rear_tire_id": 7,
                "tire_load_mass_basis": "TEST_MASS",
            },
        )

        self.assertAlmostEqual(preview["mass_resolution"]["mass_kg"], 1636.0)
        self.assertEqual(preview["mass_resolution"]["source_field"], "default_test_mass_kg_epa")
        self.assertTrue(preview["mass_resolution"]["test_mass_defaulted"])
        self.assertIn("test_mass_defaulted=True", preview["save_payload"]["tire_calc_notes"])
        self.assertIn("test_mass_default_rule=EPA_CURB_PLUS_136KG", preview["save_payload"]["tire_calc_notes"])

    @patch("src.vde_core.tire_roadload_service.get_tire_roadload_by_id")
    @patch("src.vde_core.tire_roadload_service.get_vde_tire_application")
    @patch("src.vde_core.tire_roadload_service.fetch_vde_by_id")
    def test_preview_tire_roadload_for_vde_tracks_mass_fallback_source(
        self,
        mock_fetch_vde_by_id,
        mock_get_vde_tire_application,
        mock_get_tire_roadload_by_id,
    ):
        mock_fetch_vde_by_id.return_value = {
            "id": 42,
            "mass_kg": 1500.0,
            "etw_kg": 1625.0,
            "weight_dist_fr_pct": 60.0,
            "front_pressure_psi": 31.0,
            "rear_pressure_psi": 33.0,
        }
        mock_get_vde_tire_application.return_value = {}
        mock_get_tire_roadload_by_id.return_value = {"id": 7, "standard_family": "ISO", "rr_n_per_kn": 9.5}

        preview = preview_tire_roadload_for_vde(
            42,
            {
                "front_tire_id": 7,
                "rear_tire_id": 7,
                "tire_load_mass_basis": "TWC",
            },
        )

        self.assertEqual(preview["mass_resolution"]["source_field"], "etw_kg")
        self.assertTrue(preview["mass_resolution"]["used_fallback"])
        self.assertFalse(preview["mass_resolution"]["used_inertia_class"])
        self.assertIn("mass_source=etw_kg", preview["save_payload"]["tire_calc_notes"])

    @patch("src.vde_core.tire_roadload_service.get_tire_roadload_by_id")
    def test_preview_tire_roadload_from_row_supports_unsaved_context(self, mock_get_tire_roadload_by_id):
        mock_get_tire_roadload_by_id.return_value = {"id": 7, "standard_family": "ISO", "rr_n_per_kn": 9.5}

        preview = preview_tire_roadload_from_row(
            {
                "legislation": "EPA",
                "mass_kg": 1500.0,
                "test_mass_kg": 1636.0,
                "weight_dist_fr_pct": 60.0,
                "front_pressure_psi": 31.0,
                "rear_pressure_psi": 33.0,
            },
            {
                "front_tire_id": 7,
                "rear_tire_id": 7,
                "tire_load_mass_basis": "TEST_MASS",
            },
        )

        self.assertIsNone(preview["vde_id"])
        self.assertEqual(preview["application"]["front_tire_id"], 7)
        self.assertAlmostEqual(preview["mass_resolution"]["mass_kg"], 1636.0)
        self.assertEqual(preview["component_dict"]["name"], "tire")
        self.assertIn("mass_used_fallback=True", preview["save_payload"]["tire_calc_notes"])
        self.assertIn("uses_inertia_class_mass=False", preview["save_payload"]["tire_calc_notes"])

    def test_resolve_tire_load_mass_test_mass_falls_back_to_mass_kg_when_no_default_exists(self):
        resolved = resolve_tire_load_mass(
            {"legislation": "WLTP", "mass_kg": 1400.0, "inertia_class": 1701.0},
            "TEST_MASS",
        )

        self.assertEqual(resolved["source_field"], "mass_kg")
        self.assertTrue(resolved["used_fallback"])
        self.assertFalse(resolved["used_inertia_class"])
        self.assertFalse(resolved["test_mass_defaulted"])

    @patch("src.vde_core.tire_roadload_service.get_tire_roadload_by_id")
    @patch("src.vde_core.tire_roadload_service.get_vde_tire_application")
    @patch("src.vde_core.tire_roadload_service.fetch_vde_by_id")
    def test_preview_tire_roadload_for_vde_rejects_missing_front_tire(
        self,
        mock_fetch_vde_by_id,
        mock_get_vde_tire_application,
        mock_get_tire_roadload_by_id,
    ):
        mock_fetch_vde_by_id.return_value = {
            "id": 42,
            "mass_kg": 1500.0,
            "weight_dist_fr_pct": 60.0,
            "front_pressure_psi": 32.0,
            "rear_pressure_psi": 34.0,
        }
        mock_get_vde_tire_application.return_value = {}
        mock_get_tire_roadload_by_id.side_effect = [{}, {"id": 8, "standard_family": "ISO", "rr_n_per_kn": 8.9}]

        with self.assertRaisesRegex(ValueError, "Front tire not found: id=7"):
            preview_tire_roadload_for_vde(
                42,
                {
                    "front_tire_id": 7,
                    "rear_tire_id": 8,
                },
            )

    @patch("src.vde_core.tire_roadload_service.get_tire_roadload_by_id")
    @patch("src.vde_core.tire_roadload_service.get_vde_tire_application")
    @patch("src.vde_core.tire_roadload_service.fetch_vde_by_id")
    def test_preview_tire_roadload_for_vde_requires_sae_pressures(
        self,
        mock_fetch_vde_by_id,
        mock_get_vde_tire_application,
        mock_get_tire_roadload_by_id,
    ):
        mock_fetch_vde_by_id.return_value = {
            "id": 42,
            "mass_kg": 1500.0,
            "weight_dist_fr_pct": 60.0,
            "front_pressure_psi": None,
            "rear_pressure_psi": None,
        }
        mock_get_vde_tire_application.return_value = {}
        mock_get_tire_roadload_by_id.side_effect = [
            {"id": 7, "standard_family": "SAE", "sae_a": 0.01, "sae_b": 0.001, "sae_c": 0.0001},
            {"id": 8, "standard_family": "SAE", "sae_a": 0.01, "sae_b": 0.001, "sae_c": 0.0001},
        ]

        with self.assertRaisesRegex(ValueError, "front_pressure_psi is required for SAE front tire preview"):
            preview_tire_roadload_for_vde(
                42,
                {
                    "front_tire_id": 7,
                    "rear_tire_id": 8,
                },
            )

    @patch("src.vde_core.tire_roadload_service.update_vde_tire_application")
    def test_save_tire_roadload_to_vde_persists_preview_payload(self, mock_update_vde_tire_application):
        payload = {
            "front_tire_id": 11,
            "rear_tire_id": 12,
            "front_pressure_psi": 34.0,
            "rear_pressure_psi": 36.0,
            "weight_dist_fr_pct": 58.0,
            "tire_improvement_pct": -2.0,
            "tire_load_mass_basis": "TWC",
            "tire_load_mass_used_kg": 1701.0,
            "tire_A_final": 120.0,
            "tire_B_final": 0.2,
            "tire_C_final": 0.01,
            "rrc_N_per_kN": None,
            "tire_calc_source": "tire_roadload_db:SAE/ISO",
            "tire_calc_notes": "basis=TWC",
        }

        returned = save_tire_roadload_to_vde(99, {"save_payload": payload})

        self.assertEqual(returned["front_tire_id"], 11)
        mock_update_vde_tire_application.assert_called_once_with(99, payload)

    @patch("src.vde_core.tire_roadload_service.update_vde_tire_application")
    def test_save_tire_roadload_to_vde_rejects_incomplete_payload(self, mock_update_vde_tire_application):
        with self.assertRaisesRegex(ValueError, "save_payload is missing required tire fields"):
            save_tire_roadload_to_vde(
                99,
                {
                    "save_payload": {
                        "front_tire_id": 11,
                        "rear_tire_id": 12,
                        "tire_A_final": 120.0,
                    }
                },
            )

        mock_update_vde_tire_application.assert_not_called()

    @patch("src.vde_core.tire_roadload_service.update_vde_tire_application")
    def test_save_tire_roadload_to_vde_rejects_empty_payload(self, mock_update_vde_tire_application):
        with self.assertRaisesRegex(
            ValueError,
            "save_tire_roadload_to_vde requires a preview-like calculation_result with save_payload",
        ):
            save_tire_roadload_to_vde(99, {"save_payload": {}})

        mock_update_vde_tire_application.assert_not_called()

    @patch("src.vde_core.tire_roadload_service.update_vde_tire_application")
    def test_save_tire_roadload_to_vde_rejects_blank_required_fields(self, mock_update_vde_tire_application):
        payload = {
            "front_tire_id": 11,
            "rear_tire_id": 12,
            "front_pressure_psi": "",
            "rear_pressure_psi": 36.0,
            "weight_dist_fr_pct": 58.0,
            "tire_improvement_pct": -2.0,
            "tire_load_mass_basis": "TWC",
            "tire_load_mass_used_kg": 1701.0,
            "tire_A_final": 120.0,
            "tire_B_final": 0.2,
            "tire_C_final": 0.01,
            "tire_calc_source": "tire_roadload_db:SAE/ISO",
            "tire_calc_notes": "basis=TWC",
        }

        with self.assertRaisesRegex(ValueError, "save_payload is missing required tire fields: front_pressure_psi"):
            save_tire_roadload_to_vde(99, {"save_payload": payload})

        mock_update_vde_tire_application.assert_not_called()

    @patch("src.vde_core.tire_roadload_service.get_vde_tire_application")
    def test_apply_tire_result_to_roadload_request_builds_component_from_saved_fields(
        self,
        mock_get_vde_tire_application,
    ):
        mock_get_vde_tire_application.return_value = {
            "front_tire_id": 1,
            "rear_tire_id": 2,
            "tire_load_mass_basis": "TEST_MASS",
            "tire_load_mass_used_kg": 1500.0,
            "tire_A_final": 100.0,
            "tire_B_final": 0.2,
            "tire_C_final": 0.01,
            "tire_calc_source": "tire_roadload_db:SAE/SAE",
        }

        component = apply_tire_result_to_roadload_request(55)

        self.assertIsNotNone(component)
        self.assertEqual(component.name, "tire")
        self.assertAlmostEqual(component.A, 100.0)
        self.assertEqual(component.meta["front_tire_id"], 1)

    @patch("src.vde_core.tire_roadload_service.get_vde_tire_application")
    def test_apply_tire_result_to_roadload_request_returns_none_when_saved_abc_is_incomplete(
        self,
        mock_get_vde_tire_application,
    ):
        mock_get_vde_tire_application.return_value = {
            "front_tire_id": 1,
            "rear_tire_id": 2,
            "tire_A_final": 100.0,
            "tire_B_final": None,
            "tire_C_final": 0.01,
        }

        component = apply_tire_result_to_roadload_request(55)

        self.assertIsNone(component)

    def test_build_tire_component_from_result_uses_preview_metadata(self):
        component = build_tire_component_from_result(
            {
                "application": {
                    "front_tire_id": 7,
                    "rear_tire_id": 8,
                },
                "mass_resolution": {
                    "basis": "TEST_MASS",
                },
                "calculation": {
                    "total_final_abc": {
                        "A": 10.0,
                        "B": 0.1,
                        "C": 0.01,
                    }
                },
                "save_payload": {
                    "tire_calc_source": "tire_roadload_db:ISO/ISO",
                },
            }
        )

        self.assertEqual(component.name, "tire")
        self.assertAlmostEqual(component.A, 10.0)
        self.assertEqual(component.source, "tire_roadload_db:ISO/ISO")
        self.assertEqual(component.meta["front_tire_id"], 7)
        self.assertEqual(component.meta["basis"], "TEST_MASS")


if __name__ == "__main__":
    unittest.main()
