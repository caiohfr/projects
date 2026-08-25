import json
import time
import unittest

from src.vde_core.comparison_report_service import resolve_cycle_vde_results
from src.vde_core.cycles import use_standard_cycle
from src.vde_core.phase_aggregation import split_by_phase
from src.vde_core.qa_mock_data import build_vde_seed_rows
from src.vde_core.vde_calc import compute_vde_net
from src.vde_core.vehicle_demand import (
    AmbientState,
    Provenance,
    RoadloadBasis,
    build_vehicle_demand_profile,
    calculate_vehicle_demand,
    summarize_vehicle_demand,
    to_serializable,
    vehicle_demand_result_from_dict,
)
from src.vde_core.vehicle_demand.adapters import build_vehicle_demand_request, resolve_vehicle_demand_cycle

# EPA regulatory city/highway combination weights -- same constants
# phase_aggregation.epa_city_hwy_from_phase uses internally. Reconciliation
# tests below rebuild that exact combination from the engine's own per-phase
# outputs, to prove parity without importing a private helper.
_EPA_CITY_WEIGHT = 0.55
_EPA_HWY_WEIGHT = 0.45


def _qa_rows() -> dict[str, dict]:
    return {row["source_record_id"]: row for row in build_vde_seed_rows()}


def _wltp_qa_row() -> dict:
    """A locally-defined, clearly-synthetic WLTP-legislation row in the same
    shape/style as qa_mock_data._base_vde_row. No existing WLTP QA VDE row
    exists in qa_mock_data.py (all 7 seeded rows are EPA) -- kept local to
    this test file rather than added to the shared QA seed module, per
    Sprint 9C Sec 38's "dedicated test file" preference.
    """
    return {
        "id": 990001,
        "record_origin": "IMPORTED_REFERENCE",
        "legislation": "WLTP",
        "category": "QA_WLTP",
        "make": "QA",
        "model": "WLTP-SYNTH",
        "year": 2026,
        "mass_kg": 1600.0,
        "test_mass_kg": 1780.0,
        "coast_A_N": 115.0,
        "coast_B_N_per_kph": 0.019,
        "coast_C_N_per_kph2": 0.0088,
        "trans_A_coef_N": 8.2,
        "trans_B_coef_Npkph": 0.0039,
        "trans_C_coef_Npkph2": 0.0008,
        "rrc_N_per_kN": 8.0,
        "cda_m2": 0.60,
    }


class CanonicalAdapterTests(unittest.TestCase):
    """Test #1 -- the adapter builds a complete request without downstream
    ever hand-assembling one.
    """

    def test_adapter_reads_total_net_mass_rrc_cda_cycle_from_qa_row(self):
        row = _qa_rows()["VDE-QA-001"]
        request = build_vehicle_demand_request(row)

        self.assertEqual(request.vde_id, row["id"])
        self.assertEqual(request.roadload_total.A_N, row["coast_A_N"])
        self.assertIsNotNone(request.roadload_net)
        self.assertEqual(request.test_mass_kg, row["test_mass_kg"])
        self.assertEqual(request.rrc_n_per_kn, row["rrc_N_per_kN"])
        self.assertEqual(request.cda_m2, row["cda_m2"])
        self.assertEqual(request.cycle_name, "FTP75_HWFET")
        self.assertEqual(request.cycle_source, "STANDARD")

    def test_adapter_rejects_row_with_no_total_roadload(self):
        row = dict(_qa_rows()["VDE-QA-001"])
        row["coast_A_N"] = None
        with self.assertRaises(ValueError):
            build_vehicle_demand_request(row)

    def test_adapter_performs_no_db_access(self):
        """DB independence (Sec 8): the adapter takes an already-fetched
        mapping and does no lookups of its own -- proven simply by the fact
        that a plain dict (not a live DB row) works and produces a label
        derived purely from the mapping's own fields.
        """
        row = dict(_qa_rows()["VDE-QA-001"])
        request = build_vehicle_demand_request(row)
        self.assertEqual(request.label, f"QA {row['model']} {row['year']}")

    def test_cycle_resolution_ignores_free_text_cycle_name_column(self):
        row = dict(_qa_rows()["VDE-QA-001"])
        row["cycle_name"] = "some made up label that is not a real file"
        cycle = resolve_vehicle_demand_cycle(row)
        self.assertIsNotNone(cycle)
        self.assertEqual(len(cycle), 2139)


class EndToEndCanonicalFlowTests(unittest.TestCase):
    """Test #2 -- QA saved/resolved VDE -> adapter -> request -> engine -> result."""

    def test_qa_scenario_flows_end_to_end_without_manual_request_assembly(self):
        row = _qa_rows()["VDE-QA-001"]
        request = build_vehicle_demand_request(row)
        cycle = resolve_vehicle_demand_cycle(row)

        result = calculate_vehicle_demand(request, cycle)

        self.assertEqual(result.total_summary.roadload_basis, RoadloadBasis.TOTAL)
        self.assertIsNotNone(result.net_summary)
        self.assertGreater(result.total_summary.vde_mj_per_km, 0.0)
        self.assertGreater(result.total_summary.distance_km, 0.0)


class EpaPhaseWeightedReconciliationTests(unittest.TestCase):
    """Tests #3, #4 -- real QA EPA scenarios reconcile with the project's
    existing canonical on-demand VDE (comparison_report_service.
    resolve_cycle_vde_results, which is itself EPA 55/45 city/highway
    phase-weighted -- see Sprint 9C completion report for why the stored
    vde_total_mj_per_km/vde_net_mj_per_km QA fixture columns are NOT a valid
    reconciliation target: they were empirically found to be unrelated
    placeholder numbers, not derived from the row's own ABC/mass/cycle).

    VehicleDemandSummary has no by-phase field (frozen in 9A), so this test
    reconstructs the EPA combination from three separate engine calls (one
    per phase segment), exactly mirroring phase_aggregation.
    epa_city_hwy_from_phase's own combination formula, rather than teaching
    the engine itself about EPA policy weighting.
    """

    def _epa_combined_vde(self, request, groups, basis):
        bag1 = summarize_vehicle_demand(build_vehicle_demand_profile(request, groups["bag_1"], basis), request)
        bag2 = summarize_vehicle_demand(build_vehicle_demand_profile(request, groups["bag_2"], basis), request)
        hwfet = summarize_vehicle_demand(build_vehicle_demand_profile(request, groups["hwfet"], basis), request)

        urban_energy_MJ = bag1.positive_tractive_energy_MJ + bag2.positive_tractive_energy_MJ
        urban_distance_km = bag1.distance_km + bag2.distance_km
        urban_mj_per_km = urban_energy_MJ / urban_distance_km
        highway_mj_per_km = hwfet.vde_mj_per_km
        return _EPA_CITY_WEIGHT * urban_mj_per_km + _EPA_HWY_WEIGHT * highway_mj_per_km

    def test_total_reconciles_for_qa_001_and_qa_004(self):
        rows = _qa_rows()
        for qa_id in ("VDE-QA-001", "VDE-QA-004"):
            with self.subTest(qa_id=qa_id):
                row = rows[qa_id]
                request = build_vehicle_demand_request(row)
                cycle = resolve_vehicle_demand_cycle(row)
                groups = split_by_phase(cycle)
                canonical = resolve_cycle_vde_results(row)

                combined = self._epa_combined_vde(request, groups, RoadloadBasis.TOTAL)
                self.assertAlmostEqual(combined, canonical["total"].aggregate, places=9)

    def test_net_reconciles_for_qa_001_and_qa_004(self):
        rows = _qa_rows()
        for qa_id in ("VDE-QA-001", "VDE-QA-004"):
            with self.subTest(qa_id=qa_id):
                row = rows[qa_id]
                request = build_vehicle_demand_request(row)
                self.assertIsNotNone(request.roadload_net)
                cycle = resolve_vehicle_demand_cycle(row)
                groups = split_by_phase(cycle)
                canonical = resolve_cycle_vde_results(row)

                combined = self._epa_combined_vde(request, groups, RoadloadBasis.NET)
                self.assertAlmostEqual(combined, canonical["net"].aggregate, places=9)

    def test_total_only_scenario_reconciles_and_net_stays_unavailable(self):
        row = _qa_rows()["VDE-QA-006"]
        request = build_vehicle_demand_request(row)
        self.assertIsNone(request.roadload_net)

        cycle = resolve_vehicle_demand_cycle(row)
        groups = split_by_phase(cycle)
        canonical = resolve_cycle_vde_results(row)

        combined = self._epa_combined_vde(request, groups, RoadloadBasis.TOTAL)
        self.assertAlmostEqual(combined, canonical["total"].aggregate, places=9)

        result = calculate_vehicle_demand(request, cycle)
        self.assertIsNotNone(result.total_summary)
        self.assertIsNone(result.net_summary)


class WltpReconciliationTests(unittest.TestCase):
    """Sec 9 -- at least one WLTP-type physical cycle. WLTP's phase
    combination (phase_aggregation.wltp_phases_from_phase) is a genuine
    distance-weighted average across contiguous phases, which a single
    whole-trace integral reproduces directly (verified empirically to
    floating-point precision before writing this test) -- no per-phase
    reconstruction needed here, unlike EPA's fixed 55/45 policy weights.
    """

    def test_whole_trace_reconciles_with_wltp_phase_weighted_canonical_result(self):
        row = _wltp_qa_row()
        request = build_vehicle_demand_request(row)
        cycle = resolve_vehicle_demand_cycle(row)
        self.assertEqual(row["legislation"], "WLTP")

        result = calculate_vehicle_demand(request, cycle)
        canonical = resolve_cycle_vde_results(row)

        self.assertAlmostEqual(result.total_summary.vde_mj_per_km, canonical["total"].aggregate, places=9)
        self.assertAlmostEqual(result.net_summary.vde_mj_per_km, canonical["net"].aggregate, places=9)


class AvailabilityMatrixTests(unittest.TestCase):
    """Test #5-8, Sprint 9C Sec 13 Cases A-D. Built from VDE-QA-001 with RRC/
    CdA selectively removed, so the authoritative roadload/mass/cycle stay a
    real QA scenario and only the decomposition inputs under test vary.
    """

    def _request_and_cycle(self, *, drop_rrc: bool, drop_cda: bool):
        row = dict(_qa_rows()["VDE-QA-001"])
        if drop_rrc:
            row["rrc_N_per_kN"] = None
        if drop_cda:
            row["cda_m2"] = None
        request = build_vehicle_demand_request(row, ambient=AmbientState(air_density_kg_m3=1.2, density_basis=Provenance.SOURCE))
        cycle = resolve_vehicle_demand_cycle(row)
        return request, cycle

    def test_case_a_rrc_and_cda_available_yields_rolling_aero_and_residual(self):
        request, cycle = self._request_and_cycle(drop_rrc=False, drop_cda=False)
        result = calculate_vehicle_demand(request, cycle)
        summary = result.total_summary
        self.assertIsNotNone(summary.known_rolling_energy_MJ)
        self.assertIsNotNone(summary.known_aero_energy_MJ)
        self.assertEqual(summary.provenance["rolling"], "CALCULATED")
        self.assertEqual(summary.provenance["aero"], "CALCULATED")
        self.assertEqual(summary.provenance["residual"], "CALCULATED")

    def test_case_b_cda_missing_leaves_aero_unavailable_but_valid_result(self):
        request, cycle = self._request_and_cycle(drop_rrc=False, drop_cda=True)
        result = calculate_vehicle_demand(request, cycle)
        summary = result.total_summary
        self.assertIsNotNone(summary.known_rolling_energy_MJ)
        self.assertIsNone(summary.known_aero_energy_MJ)
        self.assertIsNotNone(summary.vde_mj_per_km)
        self.assertIsNotNone(summary.residual_roadload_energy_MJ)

    def test_case_c_rrc_missing_leaves_rolling_unavailable_but_valid_result(self):
        request, cycle = self._request_and_cycle(drop_rrc=True, drop_cda=False)
        result = calculate_vehicle_demand(request, cycle)
        summary = result.total_summary
        self.assertIsNone(summary.known_rolling_energy_MJ)
        self.assertIsNotNone(summary.known_aero_energy_MJ)
        self.assertIsNotNone(summary.vde_mj_per_km)

    def test_case_d_neither_available_roadload_and_vde_still_valid(self):
        request, cycle = self._request_and_cycle(drop_rrc=True, drop_cda=True)
        result = calculate_vehicle_demand(request, cycle)
        summary = result.total_summary
        self.assertIsNone(summary.known_rolling_energy_MJ)
        self.assertIsNone(summary.known_aero_energy_MJ)
        self.assertIsNotNone(summary.roadload_energy_MJ)
        self.assertIsNotNone(summary.vde_mj_per_km)
        # Residual absorbs the entire authoritative roadload when nothing is known.
        profile = build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)
        for residual, authoritative in zip(profile.residual_roadload_force_N, profile.authoritative_roadload_force_N):
            self.assertAlmostEqual(residual, authoritative, places=9)


class ResidualSemanticsTests(unittest.TestCase):
    """Test #12, Sprint 9C Sec 27-28 -- residual must be able to represent
    positive, exactly-zero, and negative outcomes, all real values rather
    than special cases. Terminology check: nothing in this package ever
    labels residual "Other Component Losses" (Sec 28) -- it stays
    Residual/Unattributed Roadload.
    """

    def test_typical_qa_scenario_has_positive_residual(self):
        # VDE-QA-001's own RRC (8.0) and CdA (0.62) turn out to already exceed
        # its coast_A_N/coast_C_N_per_kph2 at points across the cycle
        # (empirically found while writing this test -- the QA fixture's ABC/
        # RRC/CdA were chosen independently for other purposes and are not
        # guaranteed to be mutually consistent). Both are lowered here,
        # keeping every other QA-001 value untouched, specifically to
        # demonstrate the ordinary "known contributions comfortably fit under
        # authoritative roadload" case across the whole cycle -- the negative
        # case is covered by the over-attributed test below.
        row = dict(_qa_rows()["VDE-QA-001"])
        row["rrc_N_per_kN"] = 3.0
        row["cda_m2"] = 0.2
        request = build_vehicle_demand_request(row, ambient=AmbientState(air_density_kg_m3=1.2, density_basis=Provenance.SOURCE))
        cycle = resolve_vehicle_demand_cycle(row)
        profile = build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)

        self.assertTrue(all(force > 0 for force in profile.residual_roadload_force_N))
        summary = summarize_vehicle_demand(profile, request)
        self.assertGreater(summary.residual_roadload_energy_MJ, 0.0)

    def test_engineered_scenario_has_exactly_zero_residual(self):
        row = dict(_qa_rows()["VDE-QA-001"])
        rolling_rrc = 8.0
        mass_kg = row["test_mass_kg"]
        from src.vde_core.roadload.tire_model import G_MPS2

        rolling_force_N = rolling_rrc * mass_kg * G_MPS2 / 1000.0
        row["coast_A_N"] = rolling_force_N
        row["coast_B_N_per_kph"] = 0.0
        row["coast_C_N_per_kph2"] = 0.0
        row["rrc_N_per_kN"] = rolling_rrc
        row["cda_m2"] = 0.0  # aero contributes exactly zero, still "known"
        row["trans_A_coef_N"] = row["trans_B_coef_Npkph"] = row["trans_C_coef_Npkph2"] = None

        request = build_vehicle_demand_request(row, ambient=AmbientState(air_density_kg_m3=1.2))
        cycle = resolve_vehicle_demand_cycle(row)
        profile = build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)

        for residual in profile.residual_roadload_force_N:
            self.assertAlmostEqual(residual, 0.0, places=9)
        summary = summarize_vehicle_demand(profile, request)
        self.assertAlmostEqual(summary.residual_roadload_energy_MJ, 0.0, places=9)

    def test_over_attributed_qa_derived_scenario_preserves_negative_residual(self):
        row = dict(_qa_rows()["VDE-QA-001"])
        row["coast_A_N"] = 1.0
        row["coast_B_N_per_kph"] = 0.0
        row["coast_C_N_per_kph2"] = 0.0
        row["rrc_N_per_kN"] = 8.0  # alone already >> 1 N authoritative roadload
        row["trans_A_coef_N"] = row["trans_B_coef_Npkph"] = row["trans_C_coef_Npkph2"] = None

        request = build_vehicle_demand_request(row, ambient=AmbientState(air_density_kg_m3=1.2))
        cycle = resolve_vehicle_demand_cycle(row)
        profile = build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)
        summary = summarize_vehicle_demand(profile, request)

        self.assertTrue(all(force < 0 for force in profile.residual_roadload_force_N))
        self.assertLess(summary.residual_roadload_energy_MJ, 0.0)
        self.assertTrue(any("negative" in warning.lower() for warning in summary.warnings))

    def test_residual_terminology_never_implies_a_named_component(self):
        row = _qa_rows()["VDE-QA-001"]
        request = build_vehicle_demand_request(row, ambient=AmbientState(air_density_kg_m3=1.2))
        cycle = resolve_vehicle_demand_cycle(row)
        summary = summarize_vehicle_demand(build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL), request)

        joined = " ".join(summary.warnings).lower()
        self.assertNotIn("other component losses", joined)


class AmbientAvailabilityMatrixTests(unittest.TestCase):
    """Test #9-11, Sprint 9C Sec 14."""

    def test_explicit_air_density_is_used_as_is(self):
        row = _qa_rows()["VDE-QA-001"]
        request = build_vehicle_demand_request(row, ambient=AmbientState(air_density_kg_m3=1.25, density_basis=Provenance.SOURCE))
        cycle = resolve_vehicle_demand_cycle(row)
        summary = summarize_vehicle_demand(build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL), request)
        self.assertEqual(summary.provenance["aero"], "CALCULATED")
        self.assertEqual(summary.provenance["air_density"], "SOURCE")

    def test_temperature_and_pressure_yield_calculated_density(self):
        row = _qa_rows()["VDE-QA-001"]
        request = build_vehicle_demand_request(
            row,
            ambient=AmbientState(temperature_C=20.0, pressure_kPa=101.325, temperature_basis=Provenance.REGULATORY_REFERENCE, pressure_basis=Provenance.REGULATORY_REFERENCE),
        )
        cycle = resolve_vehicle_demand_cycle(row)
        summary = summarize_vehicle_demand(build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL), request)
        self.assertEqual(summary.provenance["aero"], "CALCULATED")
        self.assertEqual(summary.provenance["air_density"], "CALCULATED")
        self.assertEqual(summary.provenance["temperature"], "REGULATORY_REFERENCE")

    def test_no_environment_leaves_aero_unavailable_but_authoritative_vde_valid(self):
        row = _qa_rows()["VDE-QA-001"]
        request = build_vehicle_demand_request(row)  # no ambient override -> AmbientState()
        cycle = resolve_vehicle_demand_cycle(row)
        summary = summarize_vehicle_demand(build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL), request)
        self.assertEqual(summary.provenance["aero"], "UNAVAILABLE")
        self.assertIsNone(summary.known_aero_energy_MJ)
        self.assertIsNotNone(summary.vde_mj_per_km)
        self.assertIsNotNone(summary.roadload_energy_MJ)


class ProvenanceAndWarningsTests(unittest.TestCase):
    """Test #10 (provenance), Sprint 9C Sec 16-17."""

    def test_request_provenance_explains_each_source(self):
        row = _qa_rows()["VDE-QA-001"]
        request = build_vehicle_demand_request(row)
        self.assertEqual(request.provenance["roadload_total"], "SOURCE")
        self.assertEqual(request.provenance["roadload_net"], "CALCULATED")
        self.assertEqual(request.provenance["transmission"], "AVAILABLE")
        self.assertEqual(request.provenance["rrc"], "SOURCE")
        self.assertEqual(request.provenance["cda"], "SOURCE")

    def test_missing_transmission_is_explained_as_missing_not_silently_dropped(self):
        row = _qa_rows()["VDE-QA-006"]
        request = build_vehicle_demand_request(row)
        self.assertEqual(request.provenance["transmission"], "MISSING")
        self.assertEqual(request.provenance["roadload_net"], "UNAVAILABLE")

    def test_warnings_are_human_readable_domain_language(self):
        row = dict(_qa_rows()["VDE-QA-001"])
        row["rrc_N_per_kN"] = None
        row["cda_m2"] = None
        request = build_vehicle_demand_request(row)
        cycle = resolve_vehicle_demand_cycle(row)
        summary = summarize_vehicle_demand(build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL), request)

        joined = " ".join(summary.warnings).lower()
        self.assertIn("rolling", joined)
        self.assertIn("rrc", joined)
        self.assertIn("aero", joined)
        self.assertIn("cda", joined)


class ZeroValueEdgeCaseTests(unittest.TestCase):
    """Test #13, Sprint 9C Sec 18 -- zero must never collapse to missing."""

    def test_zero_rrc_is_a_valid_known_zero_rolling_force(self):
        row = dict(_qa_rows()["VDE-QA-001"])
        row["rrc_N_per_kN"] = 0.0
        request = build_vehicle_demand_request(row)
        cycle = resolve_vehicle_demand_cycle(row)
        profile = build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)
        self.assertIsNotNone(profile.known_rolling_force_N)
        self.assertTrue(all(force == 0.0 for force in profile.known_rolling_force_N))

    def test_zero_cda_is_a_valid_known_zero_aero_force(self):
        row = dict(_qa_rows()["VDE-QA-001"])
        row["cda_m2"] = 0.0
        request = build_vehicle_demand_request(row, ambient=AmbientState(air_density_kg_m3=1.2))
        cycle = resolve_vehicle_demand_cycle(row)
        profile = build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)
        self.assertIsNotNone(profile.known_aero_force_N)
        self.assertTrue(all(force == 0.0 for force in profile.known_aero_force_N))

    def test_zero_braking_energy_remains_a_real_zero_not_none(self):
        row = _qa_rows()["VDE-QA-001"]
        request = build_vehicle_demand_request(row)
        cycle = use_standard_cycle("EPA")
        groups = split_by_phase(cycle)
        # HWFET alone at cruise-like speeds has no meaningfully braking segment for this ABC;
        # this asserts the field is a real, present 0.0, not that it is always exactly 0.
        summary = summarize_vehicle_demand(build_vehicle_demand_profile(request, groups["bag_1"], RoadloadBasis.TOTAL), request)
        self.assertIsInstance(summary.braking_energy_required_MJ, float)
        self.assertGreaterEqual(summary.braking_energy_required_MJ, 0.0)


class InvalidPhysicalInputTests(unittest.TestCase):
    """Tests #14-17, Sprint 9C Sec 19-22 -- present-but-impossible values
    raise ValueError; they must never silently produce a nonsense result.
    """

    def _request_and_cycle(self, row_overrides=None, ambient=None):
        row = dict(_qa_rows()["VDE-QA-001"])
        if row_overrides:
            row.update(row_overrides)
        request = build_vehicle_demand_request(row, ambient=ambient)
        cycle = resolve_vehicle_demand_cycle(row)
        return request, cycle

    def test_non_positive_mass_is_rejected(self):
        for bad_mass in (0.0, -100.0):
            with self.subTest(mass=bad_mass):
                request, cycle = self._request_and_cycle({"test_mass_kg": bad_mass, "mass_kg": bad_mass})
                with self.assertRaises(ValueError):
                    build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)

    def test_negative_rrc_is_rejected(self):
        request, cycle = self._request_and_cycle({"rrc_N_per_kN": -1.0})
        with self.assertRaises(ValueError):
            build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)

    def test_negative_cda_is_rejected(self):
        request, cycle = self._request_and_cycle(ambient=AmbientState(air_density_kg_m3=1.2), row_overrides={"cda_m2": -0.1})
        with self.assertRaises(ValueError):
            build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)

    def test_ambient_temperature_at_or_below_absolute_zero_is_rejected(self):
        request, cycle = self._request_and_cycle(ambient=AmbientState(temperature_C=-273.15, pressure_kPa=101.325))
        with self.assertRaises(ValueError):
            build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)

    def test_ambient_non_positive_pressure_is_rejected(self):
        request, cycle = self._request_and_cycle(ambient=AmbientState(temperature_C=20.0, pressure_kPa=0.0))
        with self.assertRaises(ValueError):
            build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)

    def test_ambient_non_positive_explicit_density_is_rejected(self):
        request, cycle = self._request_and_cycle(ambient=AmbientState(air_density_kg_m3=-1.0))
        with self.assertRaises(ValueError):
            build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)


class NonFiniteInputTests(unittest.TestCase):
    """Test #19, Sprint 9C Sec 24 -- NaN/inf in fundamental physical fields
    must raise before reaching the serialization boundary, not be silently
    converted to None there.
    """

    def _request_and_cycle(self, row_overrides=None, ambient=None):
        row = dict(_qa_rows()["VDE-QA-001"])
        if row_overrides:
            row.update(row_overrides)
        request = build_vehicle_demand_request(row, ambient=ambient)
        cycle = resolve_vehicle_demand_cycle(row)
        return request, cycle

    def test_nan_mass_is_rejected(self):
        request, cycle = self._request_and_cycle({"test_mass_kg": float("nan"), "mass_kg": float("nan")})
        with self.assertRaises(ValueError):
            build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)

    def test_inf_roadload_coefficient_is_rejected(self):
        request, cycle = self._request_and_cycle({"coast_A_N": float("inf")})
        with self.assertRaises(ValueError):
            build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)

    def test_nan_rrc_is_rejected(self):
        request, cycle = self._request_and_cycle({"rrc_N_per_kN": float("nan")})
        with self.assertRaises(ValueError):
            build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)

    def test_nan_ambient_temperature_is_rejected(self):
        request, cycle = self._request_and_cycle(ambient=AmbientState(temperature_C=float("nan"), pressure_kPa=101.325))
        with self.assertRaises(ValueError):
            build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)


class InvalidCycleConsistencyTests(unittest.TestCase):
    """Test #18, Sprint 9C Sec 23 -- the Vehicle Demand engine must reject
    the exact same malformed cycles compute_vde_net already rejects, via the
    same shared vde_calc.extract_cycle_arrays validation -- no divergent
    policy between the two.
    """

    def _request(self):
        row = _qa_rows()["VDE-QA-001"]
        return build_vehicle_demand_request(row)

    def _abc(self, request):
        return (request.roadload_total.A_N, request.roadload_total.B_N_per_kph, request.roadload_total.C_N_per_kph2)

    def test_empty_and_malformed_cycles_are_rejected_identically(self):
        import pandas as pd

        request = self._request()
        A, B, C = self._abc(request)
        mass = request.test_mass_kg

        malformed_cycles = {
            "empty": pd.DataFrame({"t": [], "v": []}),
            "single_point": pd.DataFrame({"t": [0.0], "v": [10.0]}),
            "duplicate_timestamps": pd.DataFrame({"t": [0.0, 1.0, 1.0, 2.0], "v": [0.0, 1.0, 2.0, 3.0]}),
            "non_monotonic": pd.DataFrame({"t": [0.0, 2.0, 1.0], "v": [0.0, 1.0, 2.0]}),
        }
        for name, cycle in malformed_cycles.items():
            with self.subTest(cycle=name):
                with self.assertRaises(ValueError):
                    compute_vde_net(cycle, A, B, C, mass)
                with self.assertRaises(ValueError):
                    build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)

    def test_negative_but_monotonic_timestamps_are_accepted_by_both(self):
        import pandas as pd

        request = self._request()
        A, B, C = self._abc(request)
        mass = request.test_mass_kg
        cycle = pd.DataFrame({"t": [-5.0, -4.0, -3.0, -2.0], "v": [10.0, 10.0, 10.0, 10.0]})

        canonical = compute_vde_net(cycle, A, B, C, mass)
        profile = build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)
        summary = summarize_vehicle_demand(profile, request)

        self.assertAlmostEqual(summary.vde_mj_per_km, canonical["MJ_km"], places=9)

    def test_zero_distance_cycle_raises_on_summary_not_on_profile(self):
        import pandas as pd

        request = self._request()
        cycle = pd.DataFrame({"t": [0.0, 1.0, 2.0], "v": [0.0, 0.0, 0.0]})

        profile = build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)
        self.assertIsNotNone(profile)
        with self.assertRaises(ValueError):
            summarize_vehicle_demand(profile, request)


class MalformedProfileRegressionTests(unittest.TestCase):
    """Test Sprint 9C Sec 26 -- the engine itself must never construct a
    Profile whose optional series length disagrees with time_s (9A's
    __post_init__ would already reject it; this is a regression guard on
    the engine's own construction, using a real QA scenario).
    """

    def test_real_profile_satisfies_9a_shape_contract(self):
        row = _qa_rows()["VDE-QA-001"]
        request = build_vehicle_demand_request(row, ambient=AmbientState(air_density_kg_m3=1.2))
        cycle = resolve_vehicle_demand_cycle(row)
        profile = build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)

        n = len(profile.time_s)
        for series in (
            profile.speed_mps,
            profile.accel_mps2,
            profile.authoritative_roadload_force_N,
            profile.inertial_force_N,
            profile.tractive_force_N,
            profile.authoritative_roadload_power_W,
            profile.inertial_power_W,
            profile.tractive_power_W,
            profile.energy_mode,
            profile.known_rolling_force_N,
            profile.known_aero_force_N,
            profile.residual_roadload_force_N,
        ):
            self.assertEqual(len(series), n)


class JsonRoundTripCanonicalResultTests(unittest.TestCase):
    """Test #20, Sprint 9C Sec 32."""

    def test_real_qa_scenario_result_round_trips_through_json(self):
        row = _qa_rows()["VDE-QA-001"]
        request = build_vehicle_demand_request(row, ambient=AmbientState(air_density_kg_m3=1.2, density_basis=Provenance.SOURCE))
        cycle = resolve_vehicle_demand_cycle(row)
        result = calculate_vehicle_demand(request, cycle)

        serialized = to_serializable(result)
        json_text = json.dumps(serialized)
        restored = vehicle_demand_result_from_dict(json.loads(json_text))

        self.assertEqual(restored, result)
        self.assertEqual(restored.total_summary.vde_mj_per_km, result.total_summary.vde_mj_per_km)


class PerformanceSanityTests(unittest.TestCase):
    """Sprint 9C Sec 33 -- no benchmark framework, just a sanity bound."""

    def test_full_regulatory_cycle_calculation_is_comfortably_interactive(self):
        row = _qa_rows()["VDE-QA-001"]
        request = build_vehicle_demand_request(row, ambient=AmbientState(air_density_kg_m3=1.2))
        cycle = resolve_vehicle_demand_cycle(row)
        self.assertGreater(len(cycle), 2000)

        started = time.perf_counter()
        calculate_vehicle_demand(request, cycle)
        elapsed_s = time.perf_counter() - started

        self.assertLess(elapsed_s, 2.0)


if __name__ == "__main__":
    unittest.main()
