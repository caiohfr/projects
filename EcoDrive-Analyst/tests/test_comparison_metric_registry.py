from __future__ import annotations

import gc
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.vde_core import db as db_module
from src.vde_core.comparison_metric_registry import ComparisonRule, MetricDirection, get_metric, list_metrics
from src.vde_core.comparison_report_service import (
    ComparisonRole,
    SourceKind,
    build_scenario_comparison_item,
    build_vde_comparison_item,
    compare_metric,
)
from src.vde_core.qa_mock_data import build_vde_seed_rows, seed_qa_database


def _qa_row(vde_id: int) -> dict:
    return dict({r["id"]: r for r in build_vde_seed_rows()}[vde_id])


class MetricRegistryContentTests(unittest.TestCase):
    def test_registry_is_deliberately_small_but_covers_all_groups(self):
        groups = {m.group for m in list_metrics()}
        self.assertEqual(
            groups,
            {"Vehicle", "Physical setup", "Roadload", "Vehicle demand", "Fuel / Energy / CO2", "Efficiency"},
        )
        self.assertLess(len(list_metrics()), 40)

    def test_direction_examples_from_spec(self):
        self.assertEqual(get_metric("vde_total").direction, MetricDirection.LOWER_IS_BETTER)
        self.assertEqual(get_metric("gco2_per_km").direction, MetricDirection.LOWER_IS_BETTER)
        self.assertEqual(get_metric("fuel_l_per_100km").direction, MetricDirection.LOWER_IS_BETTER)
        self.assertEqual(get_metric("fuel_km_per_l").direction, MetricDirection.HIGHER_IS_BETTER)
        self.assertEqual(get_metric("mass_kg").direction, MetricDirection.NEUTRAL)

    def test_unknown_metric_returns_none(self):
        self.assertIsNone(get_metric("not_a_real_metric"))


class CompareMetricTests(unittest.TestCase):
    def setUp(self):
        self.reference = build_vde_comparison_item(900001, role=ComparisonRole.REFERENCE, vde_row=_qa_row(900001))
        cheaper = _qa_row(900001)
        cheaper["id"] = 900001
        cheaper["vde_total_mj_per_km"] = 1.10  # lower than reference's 1.24
        self.better_vde = build_vde_comparison_item(900001, vde_row=cheaper)
        dirtier = _qa_row(900001)
        dirtier["vde_total_mj_per_km"] = 1.50
        self.worse_vde = build_vde_comparison_item(900001, vde_row=dirtier)

    def test_lower_is_better_metric_marks_better(self):
        result = compare_metric(self.reference, self.better_vde, "vde_total")
        self.assertTrue(result["compatible"])
        self.assertTrue(result["available"])
        self.assertLess(result["absolute_delta"], 0)
        self.assertEqual(result["semantic"], "BETTER")

    def test_lower_is_better_metric_marks_worse(self):
        result = compare_metric(self.reference, self.worse_vde, "vde_total")
        self.assertGreater(result["absolute_delta"], 0)
        self.assertEqual(result["semantic"], "WORSE")

    def test_mass_delta_has_no_automatic_verdict(self):
        heavier = _qa_row(900001)
        heavier["mass_kg"] = 1600.0
        heavier_item = build_vde_comparison_item(900001, vde_row=heavier)
        result = compare_metric(self.reference, heavier_item, "mass_kg")
        self.assertTrue(result["available"])
        self.assertGreater(result["absolute_delta"], 0)
        self.assertIsNone(result["semantic"])

    def test_epa_vs_epa_vde_total_is_compatible(self):
        result = compare_metric(self.reference, self.better_vde, "vde_total")
        self.assertTrue(result["compatible"])

    def test_epa_vs_wltp_vde_total_is_incompatible(self):
        wltp_row = _qa_row(900001)
        wltp_row["legislation"] = "WLTP"
        wltp_item = build_vde_comparison_item(900001, vde_row=wltp_row)
        result = compare_metric(self.reference, wltp_item, "vde_total")
        self.assertFalse(result["compatible"])

    def test_mass_epa_vs_wltp_is_compatible_with_basis_flag(self):
        wltp_row = _qa_row(900001)
        wltp_row["legislation"] = "WLTP"
        wltp_item = build_vde_comparison_item(900001, vde_row=wltp_row)
        result = compare_metric(self.reference, wltp_item, "mass_kg")
        self.assertTrue(result["compatible"])
        self.assertTrue(result["basis_mismatch"])

    def test_cda_always_compatible_across_legislations(self):
        wltp_row = _qa_row(900001)
        wltp_row["legislation"] = "WLTP"
        wltp_item = build_vde_comparison_item(900001, vde_row=wltp_row)
        result = compare_metric(self.reference, wltp_item, "cda_m2")
        self.assertTrue(result["compatible"])
        self.assertFalse(result["basis_mismatch"])

    def test_missing_value_is_unavailable_not_zero(self):
        no_net_row = _qa_row(900006)
        item = build_vde_comparison_item(900006, vde_row=no_net_row)
        result = compare_metric(self.reference, item, "roadload_a_net")
        self.assertFalse(result["available"])
        self.assertIsNone(result["absolute_delta"])

    def test_unknown_metric_key_is_not_compatible(self):
        result = compare_metric(self.reference, self.better_vde, "not_a_real_metric")
        self.assertFalse(result["compatible"])
        self.assertFalse(result["available"])

    def test_text_metric_has_no_numeric_delta(self):
        result = compare_metric(self.reference, self.better_vde, "make")
        self.assertTrue(result["available"])
        self.assertIsNone(result["absolute_delta"])
        self.assertIsNone(result["semantic"])


class CompareMetricFuelConsTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "compare_metric_fuelcons.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT INTO fuelcons_db (id, vde_id, electrification, record_origin, gco2_per_km, eta_pt_est) "
                "VALUES (1, 900001, 'ICE', 'ESTIMATED', 140.0, 0.30)"
            )
            con.execute(
                "INSERT INTO fuelcons_db (id, vde_id, electrification, record_origin, gco2_per_km, eta_pt_est) "
                "VALUES (2, 900001, 'ICE', 'ESTIMATED', 180.0, 0.25)"
            )
            con.commit()

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def test_co2_higher_value_is_worse(self):
        reference = build_scenario_comparison_item(1, role=ComparisonRole.REFERENCE)
        dirtier = build_scenario_comparison_item(2)
        result = compare_metric(reference, dirtier, "gco2_per_km")
        self.assertTrue(result["available"])
        self.assertGreater(result["absolute_delta"], 0)
        self.assertEqual(result["semantic"], "WORSE")

    def test_efficiency_higher_value_is_better(self):
        reference = build_scenario_comparison_item(2, role=ComparisonRole.REFERENCE)
        more_efficient = build_scenario_comparison_item(1)
        result = compare_metric(reference, more_efficient, "eta_pt_est")
        self.assertTrue(result["available"])
        self.assertGreater(result["absolute_delta"], 0)
        self.assertEqual(result["semantic"], "BETTER")


if __name__ == "__main__":
    unittest.main()
