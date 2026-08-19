from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.vde_core import db as db_module
from src.vde_core.qa_mock_data import QA_DATA_DIR, seed_qa_database
from src.vde_core.roadload import calculate_vehicle_tire_abc
from src.vde_core.tire_roadload_service import get_tire_by_code
from src.vde_core.vde_tire_proposal_resolver import resolve_tire_proposal


def _source_snapshot() -> dict:
    return {
        "legislation": "EPA",
        "mass_kg": 2516.0,
        "test_mass_kg": 2516.0,
        "tire_load_mass_basis": "TEST_MASS",
        "weight_dist_fr_pct": 55.0,
        "rrc_N_per_kN": 7.8318,
        "front_pressure_psi": 38.0,
        "rear_pressure_psi": 38.0,
        "tire_A_final": 50.0,
        "tire_B_final": 0.01,
        "tire_C_final": 0.001,
    }


class VdeTireProposalResolverTests(unittest.TestCase):
    def _temp_db_path(self) -> Path:
        QA_DATA_DIR.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="qa_tire_delta_", dir=str(QA_DATA_DIR)))
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        return temp_dir / "qa_seed.db"

    def _measured_total_source_snapshot(self, **overrides) -> dict:
        source = {
            "legislation": "EPA",
            "mass_kg": 1600.0,
            "test_mass_kg": 1600.0,
            "tire_load_mass_basis": "TEST_MASS",
            "weight_dist_fr_pct": 55.0,
            "front_pressure_psi": 38.0,
            "rear_pressure_psi": 38.0,
            "A": 118.0,
            "B": 0.019,
            "C": 0.0098,
        }
        source.update(overrides)
        return source

    def test_direct_target_rrc_uses_exact_value_without_pressure_adjustment(self):
        source = _source_snapshot()

        result = resolve_tire_proposal(
            source,
            "TIRE_TARGET_RRC",
            {"target_rrc_N_per_kN": 8.5, "front_pressure_psi": 32.0, "rear_pressure_psi": 32.0},
            current_snapshot=source,
        )

        resolved = result["resolved_snapshot"]
        self.assertEqual(result["status"], "OK")
        self.assertAlmostEqual(resolved["rrc_N_per_kN"], 8.5, places=9)
        self.assertAlmostEqual(resolved["tire_adjusted_rrc_N_per_kN"], 8.5, places=9)
        self.assertAlmostEqual(resolved["tire_delta_rrc_N_per_kN"], 8.5 - 7.8318, places=9)
        self.assertEqual(resolved["tire_adjustment_method"], "Direct target RRC")
        self.assertGreater(resolved["tire_resolved_abc"]["A"], resolved["tire_source_abc"]["A"])

    def test_direct_target_rrc_still_resolves_when_reference_pressure_is_missing(self):
        source = _source_snapshot()
        source["front_pressure_psi"] = None
        source["rear_pressure_psi"] = None

        result = resolve_tire_proposal(
            source,
            "TIRE_TARGET_RRC",
            {"target_rrc_N_per_kN": 8.5},
            current_snapshot=source,
        )

        resolved = result["resolved_snapshot"]
        self.assertEqual(result["status"], "OK")
        self.assertAlmostEqual(resolved["rrc_N_per_kN"], 8.5, places=9)
        self.assertAlmostEqual(resolved["tire_adjusted_rrc_N_per_kN"], 8.5, places=9)
        self.assertEqual(resolved["tire_adjustment_method"], "Direct target RRC")

    def test_positive_improvement_pct_lowers_resolved_rrc(self):
        source = _source_snapshot()

        result = resolve_tire_proposal(
            source,
            "TIRE_IMPROVEMENT_PCT",
            {"tire_improvement_pct": 5.0, "front_pressure_psi": 38.0, "rear_pressure_psi": 38.0},
            current_snapshot=source,
        )

        resolved = result["resolved_snapshot"]
        self.assertEqual(result["status"], "OK")
        self.assertLess(resolved["rrc_N_per_kN"], source["rrc_N_per_kN"])
        self.assertEqual(resolved["tire_adjustment_method"], "Tire improvement %")

    def test_pressure_only_increase_rrc_when_pressure_reduced(self):
        source = _source_snapshot()

        result = resolve_tire_proposal(
            source,
            "TIRE_TARGET_RRC",
            {"front_pressure_psi": 35.0, "rear_pressure_psi": 36.0},
            current_snapshot=source,
        )

        resolved = result["resolved_snapshot"]
        self.assertEqual(result["status"], "Review")
        self.assertGreater(resolved["rrc_N_per_kN"], source["rrc_N_per_kN"])
        self.assertGreater(resolved["tire_delta_rrc_N_per_kN"], 0.0)
        self.assertGreater(resolved["tire_resolved_abc"]["A"], resolved["tire_source_abc"]["A"])
        self.assertEqual(resolved["tire_adjustment_method"], "Pressure estimate")

    def test_pressure_only_reduction_rrc_when_pressure_increased(self):
        source = _source_snapshot()

        result = resolve_tire_proposal(
            source,
            "TIRE_TARGET_RRC",
            {"front_pressure_psi": 42.0, "rear_pressure_psi": 42.0},
            current_snapshot=source,
        )

        resolved = result["resolved_snapshot"]
        self.assertEqual(result["status"], "Review")
        self.assertLess(resolved["rrc_N_per_kN"], source["rrc_N_per_kN"])
        self.assertLess(resolved["tire_delta_rrc_N_per_kN"], 0.0)

    def test_same_pressure_preserves_explicit_zero_delta(self):
        source = _source_snapshot()

        result = resolve_tire_proposal(
            source,
            "TIRE_TARGET_RRC",
            {"front_pressure_psi": 38.0, "rear_pressure_psi": 38.0},
            current_snapshot=source,
        )

        resolved = result["resolved_snapshot"]
        self.assertEqual(result["status"], "Review")
        self.assertAlmostEqual(resolved["rrc_N_per_kN"], source["rrc_N_per_kN"], places=9)
        self.assertEqual(resolved["tire_delta_rrc_N_per_kN"], 0.0)

    def test_front_fraction_defaults_to_half_when_missing(self):
        source = _source_snapshot()
        source.pop("weight_dist_fr_pct")

        result = resolve_tire_proposal(
            source,
            "TIRE_TARGET_RRC",
            {"front_pressure_psi": 35.0, "rear_pressure_psi": 36.0},
            current_snapshot=source,
        )

        resolved = result["resolved_snapshot"]
        self.assertEqual(result["status"], "Review")
        self.assertEqual(resolved["tire_front_weight_fraction"], 0.5)
        self.assertIn("front_fraction_defaulted", {issue["code"] for issue in result["issues"]})

    def test_pressure_only_rrc_is_clamped_to_plus_minus_ten_percent(self):
        source = _source_snapshot()

        raised = resolve_tire_proposal(
            source,
            "TIRE_TARGET_RRC",
            {"front_pressure_psi": 20.0, "rear_pressure_psi": 20.0},
            current_snapshot=source,
        )
        lowered = resolve_tire_proposal(
            source,
            "TIRE_TARGET_RRC",
            {"front_pressure_psi": 60.0, "rear_pressure_psi": 60.0},
            current_snapshot=source,
        )

        self.assertAlmostEqual(raised["resolved_snapshot"]["rrc_N_per_kN"], source["rrc_N_per_kN"] * 1.1, places=9)
        self.assertAlmostEqual(lowered["resolved_snapshot"]["rrc_N_per_kN"], source["rrc_N_per_kN"] * 0.9, places=9)

    def test_invalid_requested_pressure_returns_missing(self):
        source = _source_snapshot()

        result = resolve_tire_proposal(
            source,
            "TIRE_TARGET_RRC",
            {"front_pressure_psi": 18.0, "rear_pressure_psi": 32.0},
            current_snapshot=source,
        )

        self.assertEqual(result["status"], "Missing")
        self.assertIn("requested_pressure_invalid", {issue["code"] for issue in result["issues"]})

    def test_missing_reference_pressure_returns_missing(self):
        source = _source_snapshot()
        source["front_pressure_psi"] = None

        result = resolve_tire_proposal(
            source,
            "TIRE_TARGET_RRC",
            {"front_pressure_psi": 35.0, "rear_pressure_psi": 35.0},
            current_snapshot=source,
        )

        self.assertEqual(result["status"], "Missing")
        self.assertIn("reference_pressure_missing", {issue["code"] for issue in result["issues"]})

    def test_tire_abc_scales_from_source_triplet_when_available(self):
        source = _source_snapshot()

        result = resolve_tire_proposal(
            source,
            "TIRE_TARGET_RRC",
            {"target_rrc_N_per_kN": 8.5, "front_pressure_psi": 32.0, "rear_pressure_psi": 32.0},
            current_snapshot=source,
        )

        resolved = result["resolved_snapshot"]
        ratio = 8.5 / source["rrc_N_per_kN"]
        self.assertAlmostEqual(resolved["tire_resolved_abc"]["A"], source["tire_A_final"] * ratio, places=9)
        self.assertAlmostEqual(resolved["tire_resolved_abc"]["B"], source["tire_B_final"] * ratio, places=9)
        self.assertAlmostEqual(resolved["tire_resolved_abc"]["C"], source["tire_C_final"] * ratio, places=9)

    def test_rrc_conversion_uses_proposal_mass_when_source_abc_is_unavailable(self):
        source = _source_snapshot()
        source.pop("tire_A_final")
        source.pop("tire_B_final")
        source.pop("tire_C_final")
        proposal_snapshot = dict(source)
        proposal_snapshot["mass_kg"] = 3000.0
        proposal_snapshot["test_mass_kg"] = 3000.0

        result = resolve_tire_proposal(
            source,
            "TIRE_TARGET_RRC",
            {"target_rrc_N_per_kN": 8.5, "front_pressure_psi": 32.0, "rear_pressure_psi": 32.0},
            current_snapshot=proposal_snapshot,
        )

        resolved = result["resolved_snapshot"]
        expected = calculate_vehicle_tire_abc(
            front_tire={"standard_family": "CUSTOM", "rr_n_per_kn": 8.5},
            rear_tire={"standard_family": "CUSTOM", "rr_n_per_kn": 8.5},
            inputs={
                "mass_kg": 3000.0,
                "front_weight_distribution_pct": 55.0,
                "front_pressure_kpa": None,
                "rear_pressure_kpa": None,
                "tire_improvement_pct": 0.0,
            },
        )

        self.assertAlmostEqual(resolved["tire_resolved_abc"]["A"], expected["total_final_abc"]["A"], places=9)
        self.assertAlmostEqual(resolved["tire_resolved_abc"]["B"], expected["total_final_abc"]["B"], places=9)
        self.assertAlmostEqual(resolved["tire_resolved_abc"]["C"], expected["total_final_abc"]["C"], places=9)

    def test_mass_and_pressure_changes_use_both_effective_values(self):
        source = _source_snapshot()
        proposal_snapshot = dict(source)
        proposal_snapshot["test_mass_kg"] = 3000.0
        proposal_snapshot["tire_load_mass_basis"] = "TEST_MASS"

        result = resolve_tire_proposal(
            source,
            "TIRE_TARGET_RRC",
            {"front_pressure_psi": 35.0, "rear_pressure_psi": 36.0},
            current_snapshot=proposal_snapshot,
        )

        resolved = result["resolved_snapshot"]
        expected_factor = (resolved["rrc_N_per_kN"] * 3000.0) / (source["rrc_N_per_kN"] * source["test_mass_kg"])
        self.assertEqual(result["status"], "Review")
        self.assertGreater(resolved["rrc_N_per_kN"], source["rrc_N_per_kN"])
        self.assertEqual(resolved["tire_load_mass_used_kg"], 3000.0)
        self.assertAlmostEqual(resolved["tire_resolved_abc"]["A"], source["tire_A_final"] * expected_factor, places=9)

    def test_invalid_test_mass_state_becomes_review_missing_instead_of_exception(self):
        source = _source_snapshot()
        proposal_snapshot = dict(source)
        proposal_snapshot["test_mass_kg"] = 2100.0
        proposal_snapshot["test_mass_basis"] = "CUSTOM"

        result = resolve_tire_proposal(
            source,
            "TIRE_TARGET_RRC",
            {"target_rrc_N_per_kN": 8.5, "front_pressure_psi": 32.0, "rear_pressure_psi": 32.0},
            current_snapshot=proposal_snapshot,
        )

        self.assertEqual(result["status"], "Missing")
        self.assertIn("tire_mass_invalid", {issue["code"] for issue in result["issues"]})
        self.assertIn("tire_mass_missing", {issue["code"] for issue in result["issues"]})
        self.assertIsNone(result["resolved_snapshot"]["tire_resolved_abc"])

    def test_legacy_alias_is_accepted_without_smerf_ui_name(self):
        source = _source_snapshot()

        result = resolve_tire_proposal(
            source,
            "TIRE_SMERF_RRC_CHANGE",
            {"target_rrc_N_per_kN": 8.5, "front_pressure_psi": 32.0, "rear_pressure_psi": 32.0},
            current_snapshot=source,
        )

        self.assertEqual(result["status"], "OK")
        self.assertAlmostEqual(result["resolved_snapshot"]["rrc_N_per_kN"], 8.5, places=9)
        self.assertEqual(result["resolved_snapshot"]["tire_adjustment_method"], "Direct target RRC")

    @patch("src.vde_core.vde_tire_proposal_resolver.get_tire_by_id")
    def test_lookup_without_full_sae_scales_source_triplet_by_rrc_and_load(self, mock_get_tire_by_id):
        source = _source_snapshot()
        proposal_snapshot = dict(source)
        proposal_snapshot["test_mass_kg"] = 3000.0
        proposal_snapshot["tire_load_mass_basis"] = "TEST_MASS"
        mock_get_tire_by_id.return_value = {"id": 99, "rr_n_per_kn": 8.6}

        result = resolve_tire_proposal(
            source,
            "TIRE_DB_LOOKUP",
            {"tire_db_id": 99, "front_pressure_psi": 36.0, "rear_pressure_psi": 36.0},
            current_snapshot=proposal_snapshot,
        )

        resolved = result["resolved_snapshot"]
        expected_factor = (8.6 * 3000.0) / (source["rrc_N_per_kN"] * source["test_mass_kg"])
        self.assertEqual(result["status"], "OK")
        self.assertEqual(resolved["tire_abc_method"], "RRC_LOAD_SCALING")
        self.assertAlmostEqual(resolved["tire_resolved_abc"]["A"], source["tire_A_final"] * expected_factor, places=9)
        self.assertAlmostEqual(resolved["tire_resolved_abc"]["B"], source["tire_B_final"] * expected_factor, places=9)
        self.assertAlmostEqual(resolved["tire_resolved_abc"]["C"], source["tire_C_final"] * expected_factor, places=9)

    @patch("src.vde_core.vde_tire_proposal_resolver.get_tire_by_id")
    def test_iso_lookup_estimates_rrc_and_abc_from_selected_tire_reference_pressure(self, mock_get_tire_by_id):
        source = _source_snapshot()
        mock_get_tire_by_id.return_value = {
            "id": 99,
            "standard_family": "ISO",
            "rr_n_per_kn": 9.0,
            "test_pressure_value": 30.0,
            "pressure_unit": "psi",
        }

        at_reference = resolve_tire_proposal(
            source,
            "TIRE_DB_LOOKUP",
            {"tire_db_id": 99, "front_pressure_psi": 30.0, "rear_pressure_psi": 30.0},
            current_snapshot=source,
        )
        lower_pressure = resolve_tire_proposal(
            source,
            "TIRE_DB_LOOKUP",
            {"tire_db_id": 99, "front_pressure_psi": 25.0, "rear_pressure_psi": 25.0},
            current_snapshot=source,
        )
        higher_pressure = resolve_tire_proposal(
            source,
            "TIRE_DB_LOOKUP",
            {"tire_db_id": 99, "front_pressure_psi": 38.0, "rear_pressure_psi": 38.0},
            current_snapshot=source,
        )

        reference = at_reference["resolved_snapshot"]
        lower = lower_pressure["resolved_snapshot"]
        higher = higher_pressure["resolved_snapshot"]
        self.assertEqual(at_reference["status"], "Review")
        self.assertEqual(reference["tire_adjustment_method"], "ISO pressure estimate")
        self.assertAlmostEqual(reference["rrc_N_per_kN"], 9.0, places=9)
        self.assertGreater(lower["rrc_N_per_kN"], reference["rrc_N_per_kN"])
        self.assertLess(higher["rrc_N_per_kN"], reference["rrc_N_per_kN"])
        self.assertGreater(lower["tire_resolved_abc"]["A"], reference["tire_resolved_abc"]["A"])
        self.assertLess(higher["tire_resolved_abc"]["A"], reference["tire_resolved_abc"]["A"])

    @patch("src.vde_core.vde_tire_proposal_resolver.get_tire_by_id")
    def test_iso_lookup_without_reference_pressure_keeps_lookup_rrc_and_requires_review(self, mock_get_tire_by_id):
        source = _source_snapshot()
        mock_get_tire_by_id.return_value = {"id": 99, "standard_family": "ISO", "rr_n_per_kn": 9.0}

        result = resolve_tire_proposal(
            source,
            "TIRE_DB_LOOKUP",
            {"tire_db_id": 99, "front_pressure_psi": 30.0, "rear_pressure_psi": 30.0},
            current_snapshot=source,
        )

        resolved = result["resolved_snapshot"]
        self.assertEqual(result["status"], "Review")
        self.assertAlmostEqual(resolved["rrc_N_per_kN"], 9.0, places=9)
        self.assertIn("reference pressure", resolved["tire_rule_notes"])
        self.assertEqual(resolved["tire_adjustment_method"], "DB lookup RRC (ISO reference pressure unavailable)")

    def test_inherit_recalculates_tire_abc_when_test_mass_changes(self):
        source = _source_snapshot()
        current = dict(source)
        current["test_mass_kg"] = 3000.0
        current["tire_load_mass_basis"] = "TEST_MASS"

        result = resolve_tire_proposal(source, "INHERIT", {}, current_snapshot=current)

        resolved = result["resolved_snapshot"]
        self.assertEqual(result["status"], "OK")
        self.assertEqual(resolved["tire_abc_method"], "RRC_LOAD_SCALING")
        self.assertEqual(resolved["tire_load_mass_used_kg"], 3000.0)
        self.assertNotEqual(resolved["tire_resolved_abc"]["A"], source["tire_A_final"])

    def test_twc_basis_prefers_explicit_inertia_class_over_curb_inference(self):
        source = _source_snapshot()
        current = dict(source)
        current["mass_kg"] = 1550.0
        current["inertia_class"] = 1644.0
        current["tire_load_mass_basis"] = "TWC"

        result = resolve_tire_proposal(source, "INHERIT", {}, current_snapshot=current)

        resolved = result["resolved_snapshot"]
        self.assertEqual(resolved["tire_load_mass_used_kg"], 1644.0)

    @patch("src.vde_core.vde_tire_proposal_resolver.get_tire_by_id")
    def test_tire_db_lookup_with_complete_sae_uses_new_tire_coefficients(self, mock_get_tire_by_id):
        source = _source_snapshot()
        source["tire_db_id"] = 10
        source["tire_code"] = "OLD"
        source["front_pressure_psi"] = 38.0
        source["rear_pressure_psi"] = 38.0

        mock_get_tire_by_id.side_effect = [
            {
                "id": 10,
                "standard_family": "SAE",
                "rr_n_per_kn": 7.8,
                "sae_alpha": 0.1,
                "sae_beta": 1.0,
                "sae_a": 0.02,
                "sae_b": 0.0001,
                "sae_c": 0.0,
                "sae_reference_load_n": 3000.0,
                "sae_reference_pressure_kpa": 220.0,
            },
            {
                "id": 99,
                "standard_family": "SAE",
                "rr_n_per_kn": 8.6,
                "sae_alpha": 0.1,
                "sae_beta": 1.0,
                "sae_a": 0.05,
                "sae_b": 0.0002,
                "sae_c": 0.0,
                "sae_reference_load_n": 3000.0,
                "sae_reference_pressure_kpa": 220.0,
            },
        ]

        result = resolve_tire_proposal(
            source,
            "TIRE_DB_LOOKUP",
            {"tire_db_id": 99, "front_pressure_psi": 36.0, "rear_pressure_psi": 36.0},
            current_snapshot=source,
        )

        resolved = result["resolved_snapshot"]
        self.assertEqual(result["status"], "OK")
        self.assertEqual(resolved["tire_abc_method"], "SAE_FULL")
        self.assertGreater(resolved["rrc_N_per_kN"], 8.0)
        self.assertGreater(resolved["tire_resolved_abc"]["A"], source["tire_A_final"])

    def test_qa_neutral_lookup_ignores_measured_total_identity_and_returns_zero_delta(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)

        with db_module.using_db_path(db_path):
            qa_base = get_tire_by_code("QA-BASE")
            qa_neutral = get_tire_by_code("QA-NEUTRAL")
            source = self._measured_total_source_snapshot(
                rrc_N_per_kN=float(qa_base["rr_n_per_kn"]),
                tire_db_id=qa_base["id"],
                tire_code=qa_base["tire_test_code"],
                front_tire_id=qa_base["id"],
                rear_tire_id=qa_base["id"],
            )
            result = resolve_tire_proposal(
                source,
                "TIRE_DB_LOOKUP",
                {
                    "tire_db_id": qa_neutral["id"],
                    "tire_code": qa_neutral["tire_test_code"],
                    "rrc_N_per_kN": float(qa_neutral["rr_n_per_kn"]),
                    "front_pressure_psi": 38.0,
                    "rear_pressure_psi": 38.0,
                    "tire_load_mass_basis": "TEST_MASS",
                },
                current_snapshot=dict(source),
            )

        resolved = result["resolved_snapshot"]
        self.assertEqual(result["status"], "OK")
        self.assertNotAlmostEqual(resolved["tire_source_abc"]["A"], 118.0)
        for key in ("A", "B", "C"):
            self.assertAlmostEqual(resolved["tire_source_abc"][key], resolved["tire_resolved_abc"][key], places=9)
            self.assertAlmostEqual(resolved["tire_delta_abc"][key], 0.0, places=9)

    def test_same_reference_rrc_with_different_sae_coefficients_produces_nonzero_tire_delta(self):
        # Equal reference RRC does not imply equivalent SAE roadload curves.
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)

        with db_module.using_db_path(db_path):
            qa_base = get_tire_by_code("QA-BASE")
            qa_same_rrc = get_tire_by_code("QA-SAME-RRC-DIFF-SAE")
            source = self._measured_total_source_snapshot(
                rrc_N_per_kN=float(qa_base["rr_n_per_kn"]),
                tire_db_id=qa_base["id"],
                tire_code=qa_base["tire_test_code"],
                front_tire_id=qa_base["id"],
                rear_tire_id=qa_base["id"],
            )
            result = resolve_tire_proposal(
                source,
                "TIRE_DB_LOOKUP",
                {
                    "tire_db_id": qa_same_rrc["id"],
                    "tire_code": qa_same_rrc["tire_test_code"],
                    "rrc_N_per_kN": float(qa_same_rrc["rr_n_per_kn"]),
                    "front_pressure_psi": 38.0,
                    "rear_pressure_psi": 38.0,
                    "tire_load_mass_basis": "TEST_MASS",
                },
                current_snapshot=dict(source),
            )

        resolved = result["resolved_snapshot"]
        self.assertEqual(result["status"], "OK")
        self.assertAlmostEqual(resolved["tire_source_rrc_N_per_kN"], 8.0, places=6)
        self.assertAlmostEqual(float(qa_same_rrc["rr_n_per_kn"]), 8.0, places=6)
        delta = resolved["tire_delta_abc"]
        self.assertTrue(any(abs(float(delta[key])) > 1e-9 for key in ("A", "B", "C")))

    def test_target_rrc_neutral_uses_tire_reference_not_measured_total(self):
        source = self._measured_total_source_snapshot(rrc_N_per_kN=8.0)

        result = resolve_tire_proposal(
            source,
            "TIRE_TARGET_RRC",
            {"target_rrc_N_per_kN": 8.0, "front_pressure_psi": 38.0, "rear_pressure_psi": 38.0},
            current_snapshot=dict(source),
        )

        resolved = result["resolved_snapshot"]
        self.assertEqual(result["status"], "OK")
        self.assertNotAlmostEqual(resolved["tire_source_abc"]["A"], 118.0)
        for key in ("A", "B", "C"):
            self.assertAlmostEqual(resolved["tire_source_abc"][key], resolved["tire_resolved_abc"][key], places=9)
            self.assertAlmostEqual(resolved["tire_delta_abc"][key], 0.0, places=9)

    def test_target_rrc_direction_from_measured_total_baseline_is_delta_only(self):
        source = self._measured_total_source_snapshot(rrc_N_per_kN=8.0)

        lower = resolve_tire_proposal(
            source,
            "TIRE_TARGET_RRC",
            {"target_rrc_N_per_kN": 7.5, "front_pressure_psi": 38.0, "rear_pressure_psi": 38.0},
            current_snapshot=dict(source),
        )
        higher = resolve_tire_proposal(
            source,
            "TIRE_TARGET_RRC",
            {"target_rrc_N_per_kN": 9.0, "front_pressure_psi": 38.0, "rear_pressure_psi": 38.0},
            current_snapshot=dict(source),
        )

        self.assertLess(lower["resolved_snapshot"]["tire_delta_abc"]["A"], 0.0)
        self.assertGreater(higher["resolved_snapshot"]["tire_delta_abc"]["A"], 0.0)

    def test_missing_source_tire_reference_does_not_silently_assume_zero(self):
        source = self._measured_total_source_snapshot()

        result = resolve_tire_proposal(
            source,
            "TIRE_TARGET_RRC",
            {"target_rrc_N_per_kN": 8.5, "front_pressure_psi": 38.0, "rear_pressure_psi": 38.0},
            current_snapshot=dict(source),
        )

        self.assertEqual(result["status"], "Missing")
        self.assertIn("tire_source_reference_missing", {issue["code"] for issue in result["issues"]})
        self.assertIsNone(result["resolved_snapshot"]["tire_resolved_abc"])
        self.assertIsNone(result["resolved_snapshot"]["tire_delta_abc"])


if __name__ == "__main__":
    unittest.main()
