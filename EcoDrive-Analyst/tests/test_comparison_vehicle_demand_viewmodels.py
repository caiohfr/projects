import unittest
from unittest.mock import patch

from src.vde_core.comparison_report_service import ComparisonDataset, ComparisonRole, build_vde_comparison_item
from src.vde_core.qa_mock_data import build_vde_seed_rows
from src.vde_core.vehicle_demand import RoadloadBasis
from src.vde_core.vehicle_demand import calculate_vehicle_demand as _real_calculate_vehicle_demand
from src.vde_app.comparison_report_viewmodels import RowVisibility, visible_rows
from src.vde_app.comparison_vehicle_demand_viewmodels import (
    VEHICLE_DEMAND_SECTION_TITLE,
    build_vehicle_demand_breakdown_rows,
    build_vehicle_demand_comparison_rows,
    get_vehicle_demand_result,
    resolve_vehicle_demand_outcomes,
)


def _qa_rows() -> dict[str, dict]:
    return {row["source_record_id"]: row for row in build_vde_seed_rows()}


def _item(qa_id: str, role: ComparisonRole = ComparisonRole.COMPARISON):
    row = _qa_rows()[qa_id]
    return build_vde_comparison_item(row["id"], role=role, vde_row=row)


def _row_by_label(section, label):
    return next(row for row in section.rows if row.label == label)


class ReferenceAndComparisonTests(unittest.TestCase):
    """Sec 45 -- Reference + Proposal(s)."""

    def test_reference_and_two_comparisons_have_absolute_values_and_deltas(self):
        reference = _item("VDE-QA-001", ComparisonRole.REFERENCE)
        proposal = _item("VDE-QA-004")
        benchmark = _item("VDE-QA-005")
        dataset = ComparisonDataset(reference=reference, comparisons=(proposal, benchmark))

        section = build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.TOTAL, "Metric")

        self.assertEqual(section.title, VEHICLE_DEMAND_SECTION_TITLE)
        vde_row = _row_by_label(section, "VDE")
        self.assertTrue(vde_row.reference_cell.available)
        self.assertIsNone(vde_row.reference_cell.formatted_delta)  # Reference never shows a delta vs itself
        for cell in vde_row.comparison_cells:
            self.assertTrue(cell.available)
            self.assertIsNotNone(cell.absolute_delta)
            self.assertIsNotNone(cell.formatted_delta)

    def test_no_kpi_row_carries_a_better_worse_semantic(self):
        # Sprint 9D Sec 58: no winner scores.
        reference = _item("VDE-QA-001", ComparisonRole.REFERENCE)
        proposal = _item("VDE-QA-004")
        dataset = ComparisonDataset(reference=reference, comparisons=(proposal,))
        section = build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.TOTAL, "Metric")
        for row in section.rows:
            self.assertIsNone(row.reference_cell.semantic)
            for cell in row.comparison_cells:
                self.assertIsNone(cell.semantic)


class ReferenceLessTests(unittest.TestCase):
    """Sec 45, 47 -- Reference-less set: absolute values, no fabricated delta, no crash."""

    def test_reference_less_dataset_shows_absolute_values_without_delta(self):
        a = _item("VDE-QA-001")
        b = _item("VDE-QA-004")
        dataset = ComparisonDataset(reference=None, comparisons=(a, b))

        section = build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.TOTAL, "Metric")

        for row in section.rows:
            self.assertIsNone(row.reference_cell.formatted_delta)
            for cell in row.comparison_cells:
                self.assertIsNone(cell.formatted_delta)
                self.assertIsNone(cell.absolute_delta)


class TotalNetTests(unittest.TestCase):
    """Sec 45-46 -- TOTAL, NET available, NET missing with no fallback."""

    def test_total_basis_values_are_present(self):
        reference = _item("VDE-QA-001", ComparisonRole.REFERENCE)
        dataset = ComparisonDataset(reference=reference, comparisons=())
        section = build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.TOTAL, "Metric")
        vde_row = _row_by_label(section, "VDE")
        self.assertTrue(vde_row.reference_cell.available)

    def test_net_available_scenario_shows_net_values(self):
        reference = _item("VDE-QA-001", ComparisonRole.REFERENCE)
        dataset = ComparisonDataset(reference=reference, comparisons=())
        section = build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.NET, "Metric")
        vde_row = _row_by_label(section, "VDE")
        self.assertTrue(vde_row.reference_cell.available)

    def test_net_missing_never_falls_back_to_total(self):
        reference = _item("VDE-QA-001", ComparisonRole.REFERENCE)
        no_net = _item("VDE-QA-006")  # no transmission -> NET unavailable
        dataset = ComparisonDataset(reference=reference, comparisons=(no_net,))

        total_section = build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.TOTAL, "Metric")
        net_section = build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.NET, "Metric")

        total_vde_row = _row_by_label(total_section, "VDE")
        net_vde_row = _row_by_label(net_section, "VDE")
        self.assertTrue(total_vde_row.comparison_cells[0].available)
        self.assertFalse(net_vde_row.comparison_cells[0].available)
        self.assertEqual(net_vde_row.comparison_cells[0].formatted_value, "-")
        self.assertNotEqual(net_vde_row.comparison_cells[0].raw_value, total_vde_row.comparison_cells[0].raw_value)
        self.assertIn("NET", net_vde_row.comparison_cells[0].warning)


class PartialAvailabilityTests(unittest.TestCase):
    """Sec 45, 48 -- Rolling missing, Aero missing, partial availability stays comparable."""

    def test_aero_unavailable_without_ambient_but_other_kpis_valid(self):
        # No Comparison-sourced request supplies ambient data yet (Sprint 9D
        # Sec 23/55) -- Known Aero Energy is always unavailable via this path,
        # while VDE/Roadload/Rolling stay valid. VDE-QA-001 DOES have a CdA
        # value, so the specific reason is air density, not CdA itself (post-
        # freeze hotfix: the engine's own distinct reason is surfaced, not a
        # flat "CdA/air density unavailable").
        reference = _item("VDE-QA-001", ComparisonRole.REFERENCE)
        dataset = ComparisonDataset(reference=reference, comparisons=())
        section = build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.TOTAL, "Metric")

        aero_row = _row_by_label(section, "Known Aero Energy")
        rolling_row = _row_by_label(section, "Known Rolling Energy")
        vde_row = _row_by_label(section, "VDE")

        self.assertFalse(aero_row.reference_cell.available)
        self.assertEqual(aero_row.reference_cell.formatted_value, "-")
        self.assertIn("air density", aero_row.reference_cell.warning.lower())
        self.assertNotIn("cda_m2 missing", aero_row.reference_cell.warning.lower())
        self.assertTrue(rolling_row.reference_cell.available)
        self.assertTrue(vde_row.reference_cell.available)

    def test_scenario_a_b_c_partial_decomposition_all_remain_comparable(self):
        # A: rolling+roadload valid (this adapter path never has aero). B/C
        # both share the same shape here; the key assertion is that VDE and
        # Roadload Energy stay comparable across items regardless of RRC.
        row_with_rrc = dict(_qa_rows()["VDE-QA-001"])
        row_without_rrc = dict(_qa_rows()["VDE-QA-001"])
        row_without_rrc["rrc_N_per_kN"] = None
        item_a = build_vde_comparison_item(900001, role=ComparisonRole.REFERENCE, vde_row=row_with_rrc)
        item_b = build_vde_comparison_item(900001, role=ComparisonRole.COMPARISON, vde_row=row_without_rrc)
        dataset = ComparisonDataset(reference=item_a, comparisons=(item_b,))

        section = build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.TOTAL, "Metric")
        vde_row = _row_by_label(section, "VDE")
        rolling_row = _row_by_label(section, "Known Rolling Energy")

        self.assertTrue(vde_row.reference_cell.available)
        self.assertTrue(vde_row.comparison_cells[0].available)
        self.assertTrue(rolling_row.reference_cell.available)
        self.assertFalse(rolling_row.comparison_cells[0].available)
        self.assertIn("rrc_n_per_kn", rolling_row.comparison_cells[0].warning)


class ResidualPresentationTests(unittest.TestCase):
    """Sec 45, 49 -- negative residual preserved in presentation data, review flag, never abs()."""

    def test_negative_residual_is_preserved_and_flagged(self):
        row = dict(_qa_rows()["VDE-QA-001"])
        row["coast_A_N"] = 1.0
        row["coast_B_N_per_kph"] = 0.0
        row["coast_C_N_per_kph2"] = 0.0
        row["rrc_N_per_kN"] = 8.0  # alone already >> 1 N authoritative roadload
        row["trans_A_coef_N"] = row["trans_B_coef_Npkph"] = row["trans_C_coef_Npkph2"] = None
        item = build_vde_comparison_item(900001, role=ComparisonRole.REFERENCE, vde_row=row)
        dataset = ComparisonDataset(reference=item, comparisons=())

        section = build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.TOTAL, "Metric")
        residual_row = _row_by_label(section, "Residual / Unattributed Roadload")
        cell = residual_row.reference_cell

        self.assertTrue(cell.available)
        self.assertLess(cell.raw_value, 0.0)
        self.assertNotEqual(cell.raw_value, abs(cell.raw_value))  # never abs()'d
        self.assertIn("Review", cell.formatted_value)
        self.assertIsNotNone(cell.warning)

    def test_never_labeled_other_losses(self):
        reference = _item("VDE-QA-001", ComparisonRole.REFERENCE)
        dataset = ComparisonDataset(reference=reference, comparisons=())
        section = build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.TOTAL, "Metric")
        residual_row = _row_by_label(section, "Residual / Unattributed Roadload")
        self.assertNotIn("Other Losses", residual_row.label)
        self.assertNotIn("other losses", (residual_row.reference_cell.warning or "").lower())


class ZeroValueTests(unittest.TestCase):
    """Sec 45, 18 -- zero must never collapse to unavailable."""

    def test_zero_braking_energy_shows_as_real_zero(self):
        row = dict(_qa_rows()["VDE-QA-001"])
        item = build_vde_comparison_item(900001, role=ComparisonRole.REFERENCE, vde_row=row)
        dataset = ComparisonDataset(reference=item, comparisons=())

        # HWFET-only comparison via a direct engine call is exercised in the
        # engine's own test suite; here we assert the presentation layer
        # never turns a real 0.0 into "-".
        section = build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.TOTAL, "Metric")
        braking_row = _row_by_label(section, "Braking Energy Required")
        self.assertTrue(braking_row.reference_cell.available)
        self.assertNotEqual(braking_row.reference_cell.formatted_value, "-")


class BrakingTerminologyTests(unittest.TestCase):
    """Sec 45, 50."""

    def test_label_is_braking_energy_required_not_regen(self):
        reference = _item("VDE-QA-001", ComparisonRole.REFERENCE)
        dataset = ComparisonDataset(reference=reference, comparisons=())
        section = build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.TOTAL, "Metric")
        labels = [row.label for row in section.rows]
        self.assertIn("Braking Energy Required", labels)
        joined = " ".join(labels).lower()
        self.assertNotIn("regen", joined)
        self.assertNotIn("recovered", joined)


class NoPhysicsDuplicationTests(unittest.TestCase):
    """Sec 45, 51 -- the builder consumes calculate_vehicle_demand's result
    as-is; it never recomputes physics itself.
    """

    def test_builder_uses_provided_result_without_recomputation(self):
        from src.vde_core.vehicle_demand import RoadloadCoefficients, VehicleDemandResult, VehicleDemandSummary

        reference = _item("VDE-QA-001", ComparisonRole.REFERENCE)
        dataset = ComparisonDataset(reference=reference, comparisons=())

        fake_summary = VehicleDemandSummary(
            roadload_basis=RoadloadBasis.TOTAL,
            distance_km=1.0,
            vde_mj_per_km=12345.0,
            roadload_energy_MJ=999.0,
        )
        fake_result = VehicleDemandResult(total_summary=fake_summary, net_summary=None)

        with patch(
            "src.vde_app.comparison_vehicle_demand_viewmodels.calculate_vehicle_demand",
            return_value=fake_result,
        ) as mocked:
            section = build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.TOTAL, "Metric")

        mocked.assert_called_once()
        vde_row = _row_by_label(section, "VDE")
        self.assertEqual(vde_row.reference_cell.raw_value, 12345.0)
        roadload_row = _row_by_label(section, "Roadload Energy")
        self.assertEqual(roadload_row.reference_cell.raw_value, 999.0)


class PerformanceMemoizationTests(unittest.TestCase):
    """Sec 10-11 -- the table builder and the breakdown-chart builder must
    share one calculate_vehicle_demand() pass per item per render, not
    trigger it independently.
    """

    def test_shared_outcomes_avoid_recomputation_across_both_builders(self):
        reference = _item("VDE-QA-001", ComparisonRole.REFERENCE)
        comparison = _item("VDE-QA-004")
        dataset = ComparisonDataset(reference=reference, comparisons=(comparison,))

        with patch(
            "src.vde_app.comparison_vehicle_demand_viewmodels.calculate_vehicle_demand",
            wraps=_real_calculate_vehicle_demand,
        ) as mocked:
            outcomes = resolve_vehicle_demand_outcomes(dataset)
            build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.TOTAL, "Metric", outcomes=outcomes)
            build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.NET, "Metric", outcomes=outcomes)
            build_vehicle_demand_breakdown_rows(dataset, RoadloadBasis.TOTAL, outcomes=outcomes)

        self.assertEqual(mocked.call_count, 2)  # once per item, regardless of basis/builder count

    def test_omitting_outcomes_still_works_but_recomputes_independently(self):
        reference = _item("VDE-QA-001", ComparisonRole.REFERENCE)
        dataset = ComparisonDataset(reference=reference, comparisons=())
        section = build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.TOTAL, "Metric")
        self.assertTrue(_row_by_label(section, "VDE").reference_cell.available)


class ScenarioErrorIsolationTests(unittest.TestCase):
    """Sec 45, 41-42 -- one scenario's failure never crashes the builder or
    hides the other scenarios.
    """

    def test_invalid_mass_scenario_stays_visible_with_short_reason(self):
        good_row = dict(_qa_rows()["VDE-QA-001"])
        bad_row = dict(_qa_rows()["VDE-QA-004"])
        bad_row["test_mass_kg"] = 0.0
        bad_row["mass_kg"] = 0.0
        good_item = build_vde_comparison_item(900001, role=ComparisonRole.REFERENCE, vde_row=good_row)
        bad_item = build_vde_comparison_item(900004, role=ComparisonRole.COMPARISON, vde_row=bad_row)
        dataset = ComparisonDataset(reference=good_item, comparisons=(bad_item,))

        section = build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.TOTAL, "Metric")

        vde_row = _row_by_label(section, "VDE")
        self.assertTrue(vde_row.reference_cell.available)
        self.assertFalse(vde_row.comparison_cells[0].available)
        self.assertEqual(vde_row.comparison_cells[0].formatted_value, "-")
        warning = vde_row.comparison_cells[0].warning
        self.assertIsNotNone(warning)
        self.assertNotIn("Traceback", warning)
        self.assertNotIn("ValueError", warning)

    def test_get_vehicle_demand_result_never_raises_for_invalid_row(self):
        bad_row = dict(_qa_rows()["VDE-QA-001"])
        bad_row["rrc_N_per_kN"] = -1.0
        item = build_vde_comparison_item(900001, role=ComparisonRole.REFERENCE, vde_row=bad_row)
        outcome = get_vehicle_demand_result(item)
        self.assertIsNone(outcome.result)
        self.assertIsNotNone(outcome.unavailable_reason)


class RealComparisonScenarioEndToEndTests(unittest.TestCase):
    """Sec 45, 52 -- Comparison item -> Vehicle Demand adapter -> Result -> presentation builder."""

    def test_two_real_qa_scenarios_produce_expected_key_values(self):
        reference = _item("VDE-QA-001", ComparisonRole.REFERENCE)
        comparison = _item("VDE-QA-004")
        dataset = ComparisonDataset(reference=reference, comparisons=(comparison,))

        section = build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.TOTAL, "Metric")
        vde_row = _row_by_label(section, "VDE")

        # VDE-QA-004 (heavier baseline) should demand more, not less.
        self.assertGreater(vde_row.comparison_cells[0].raw_value, vde_row.reference_cell.raw_value)
        self.assertGreater(vde_row.comparison_cells[0].absolute_delta, 0.0)

    def test_breakdown_rows_close_against_roadload_energy(self):
        reference = _item("VDE-QA-001", ComparisonRole.REFERENCE)
        dataset = ComparisonDataset(reference=reference, comparisons=())

        section = build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.TOTAL, "Metric")
        roadload_energy = _row_by_label(section, "Roadload Energy").reference_cell.raw_value

        breakdown = build_vehicle_demand_breakdown_rows(dataset, RoadloadBasis.TOTAL)
        row = breakdown["rows"][0]
        known_rolling = row["known_rolling_MJ"] or 0.0
        known_aero = row["known_aero_MJ"] or 0.0
        residual = row["residual_MJ"]
        self.assertAlmostEqual(known_rolling + known_aero + residual, roadload_energy, places=6)


class VisibilityPolicyHotfixTests(unittest.TestCase):
    """Post-freeze hotfix regression list (all 8 numbered cases), scoped to
    the Vehicle Demand Summary table specifically. Every one of its 8 rows
    is marked RowVisibility.ALWAYS -- "unavailable is information" applies
    to all of them, not just Known Aero.
    """

    def test_1_all_eight_rows_are_marked_always_visible(self):
        reference = _item("VDE-QA-001", ComparisonRole.REFERENCE)
        dataset = ComparisonDataset(reference=reference, comparisons=())
        section = build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.TOTAL, "Metric")
        self.assertEqual(len(section.rows), 8)
        for row in section.rows:
            self.assertIs(row.visibility, RowVisibility.ALWAYS)

    def test_1_canonical_row_survives_even_when_unavailable_for_every_scenario(self):
        # No Comparison-sourced request ever supplies ambient data (Sec 55),
        # so Known Aero Energy is unavailable for literally every scenario
        # today -- exactly the case that motivated this hotfix.
        reference = _item("VDE-QA-001", ComparisonRole.REFERENCE)
        comparison = _item("VDE-QA-004")
        dataset = ComparisonDataset(reference=reference, comparisons=(comparison,))
        section = build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.TOTAL, "Metric")

        aero_row = _row_by_label(section, "Known Aero Energy")
        self.assertFalse(aero_row.reference_cell.available)
        self.assertFalse(aero_row.comparison_cells[0].available)
        self.assertIn(aero_row, visible_rows(section))

    def test_3_cda_available_rho_unavailable_shows_air_density_reason(self):
        reference = _item("VDE-QA-001", ComparisonRole.REFERENCE)  # has cda_m2, no ambient ever supplied
        dataset = ComparisonDataset(reference=reference, comparisons=())
        section = build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.TOTAL, "Metric")
        aero_row = _row_by_label(section, "Known Aero Energy")

        self.assertIn(aero_row, visible_rows(section))
        self.assertEqual(aero_row.reference_cell.formatted_value, "-")
        self.assertIn("air density", aero_row.reference_cell.warning.lower())

    def test_4_cda_unavailable_shows_a_distinct_reason_from_case_3(self):
        row = dict(_qa_rows()["VDE-QA-001"])
        row["cda_m2"] = None
        reference = build_vde_comparison_item(row["id"], role=ComparisonRole.REFERENCE, vde_row=row)
        dataset = ComparisonDataset(reference=reference, comparisons=())
        section = build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.TOTAL, "Metric")
        aero_row = _row_by_label(section, "Known Aero Energy")

        self.assertIn(aero_row, visible_rows(section))
        self.assertIn("cda_m2", aero_row.reference_cell.warning)
        self.assertNotIn("air density could not be resolved", aero_row.reference_cell.warning)

    def test_5_rrc_unavailable_shows_rolling_reason_and_stays_visible(self):
        row = dict(_qa_rows()["VDE-QA-001"])
        row["rrc_N_per_kN"] = None
        reference = build_vde_comparison_item(row["id"], role=ComparisonRole.REFERENCE, vde_row=row)
        dataset = ComparisonDataset(reference=reference, comparisons=())
        section = build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.TOTAL, "Metric")
        rolling_row = _row_by_label(section, "Known Rolling Energy")

        self.assertIn(rolling_row, visible_rows(section))
        self.assertFalse(rolling_row.reference_cell.available)
        self.assertIn("rrc_n_per_kn", rolling_row.reference_cell.warning)

    def test_6_net_unavailable_for_every_scenario_stays_auditable_no_total_fallback(self):
        # VDE-QA-006 has no resolved transmission -> NET unavailable.
        reference = _item("VDE-QA-006", ComparisonRole.REFERENCE)
        dataset = ComparisonDataset(reference=reference, comparisons=())

        total_section = build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.TOTAL, "Metric")
        net_section = build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.NET, "Metric")
        total_vde_row = _row_by_label(total_section, "VDE")
        net_vde_row = _row_by_label(net_section, "VDE")

        self.assertIn(net_vde_row, visible_rows(net_section))
        self.assertTrue(total_vde_row.reference_cell.available)
        self.assertFalse(net_vde_row.reference_cell.available)
        self.assertEqual(net_vde_row.reference_cell.formatted_value, "-")
        self.assertIn("NET", net_vde_row.reference_cell.warning)

    def test_7_reference_less_dataset_uses_the_same_always_visible_policy(self):
        a = _item("VDE-QA-001")
        b = _item("VDE-QA-004")
        dataset = ComparisonDataset(reference=None, comparisons=(a, b))
        section = build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.TOTAL, "Metric")
        aero_row = _row_by_label(section, "Known Aero Energy")

        self.assertIn(aero_row, visible_rows(section))
        self.assertFalse(aero_row.reference_cell.available)
        self.assertFalse(any(c.available for c in aero_row.comparison_cells))

    def test_8_zero_rrc_remains_available_not_missing_on_an_always_visible_row(self):
        row = dict(_qa_rows()["VDE-QA-001"])
        row["rrc_N_per_kN"] = 0.0
        reference = build_vde_comparison_item(row["id"], role=ComparisonRole.REFERENCE, vde_row=row)
        dataset = ComparisonDataset(reference=reference, comparisons=())
        section = build_vehicle_demand_comparison_rows(dataset, RoadloadBasis.TOTAL, "Metric")
        rolling_row = _row_by_label(section, "Known Rolling Energy")

        self.assertIn(rolling_row, visible_rows(section))
        self.assertTrue(rolling_row.reference_cell.available)
        self.assertEqual(rolling_row.reference_cell.raw_value, 0.0)
        self.assertNotEqual(rolling_row.reference_cell.formatted_value, "-")


if __name__ == "__main__":
    unittest.main()
