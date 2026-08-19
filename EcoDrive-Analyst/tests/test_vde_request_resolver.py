from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.vde_core import db as db_module
from src.vde_core.component_repositories import ComponentRepository
from src.vde_core.qa_mock_data import QA_DATA_DIR, seed_qa_database
from src.vde_core.repositories.vde_repository import fetch_vde_by_id
from src.vde_core.roadload_analysis import roadload_force_N
from src.vde_core.vde_request_compact_adapter import compact_baseline_context
from src.vde_core.vde_request_compact_state import apply_v22_baseline, create_v22_state
from src.vde_core.vde_request_resolver import resolve_vde_request


def _cycle_df():
    return pd.DataFrame({"t": [0.0, 1.0, 2.0], "v": [0.0, 8.0, 10.0]})


def _baseline_context() -> dict:
    return {
        "cycle_df": _cycle_df(),
        "legislation": "EPA",
        "category": "MIDSIZE",
        "mass_kg": 1600.0,
        "test_mass_kg": 1736.0,
        "weight_dist_fr_pct": 55.0,
        "CdA": 0.62,
        "rrc_N_per_kN": 7.8318,
        "front_pressure_psi": 38.0,
        "rear_pressure_psi": 38.0,
        "tire_A_final": 50.0,
        "tire_B_final": 0.01,
        "tire_C_final": 0.001,
        "front_tire_id": 7,
        "rear_tire_id": 7,
        "trans_A_coef_N": 10.0,
        "trans_B_coef_Npkph": 0.005,
        "trans_C_coef_Npkph2": 0.001,
        "brake_A": 4.0,
        "brake_B": 0.0008,
        "brake_C": 0.0001,
        "axle_hub_A": 2.0,
        "axle_hub_B": 0.0004,
        "axle_hub_C": 0.0001,
        "parasitic_A": 3.0,
        "parasitic_B": 0.0005,
        "parasitic_C": 0.0001,
    }


def _baseline_context_without_front_weight() -> dict:
    payload = _baseline_context()
    payload.pop("weight_dist_fr_pct", None)
    return payload


def _workbook_with_domains(direct_domains: dict, *, walk_from: str = "baseline", proposal_id: str = "proposal_req_1", proposal_label: str = "Requested #1") -> dict:
    return {
        "scenarios": [
            {"key": "baseline", "label": "Baseline", "role": "baseline"},
            {"key": proposal_id, "label": proposal_label, "role": "walked"},
        ],
        "columns": {
            "baseline": {"kind": "baseline"},
            proposal_id: {"kind": "walked", "walk_from": walk_from},
        },
        "proposals": {
            proposal_id: deepcopy(direct_domains),
        },
        "vde_request_import": {
            "baseline_printed": {
                "selected_baseline_vde_id": 5038,
                "mass_kg": 1600.0,
                "A": 120.0,
                "B": 0.02,
                "C": 0.01,
                "cda_m2": 0.62,
            },
            "baseline_corrections": {},
            "effective_baseline": {
                "selected_baseline_vde_id": 5038,
                "legislation": "EPA",
                "category": "MIDSIZE",
                "make": "FORD",
                "model": "TEST",
                "year": 2026,
                "cycle_name": "FTP75",
                "mass_kg": 1600.0,
                "test_mass_kg": 1736.0,
                "A": 120.0,
                "B": 0.02,
                "C": 0.01,
                "cda_m2": 0.62,
            },
            "columns": {
                proposal_id: {
                    "proposal_id": proposal_id,
                    "display_index": 1,
                    "source_column": proposal_label,
                }
            },
        },
    }


def _proposal(domain: str, proposal_type: str, details: dict | None = None, *, label: str | None = None) -> dict:
    return {
        "id": "prop_1",
        "domain": domain,
        "proposal_type": proposal_type,
        "label": label or proposal_type,
        "details": deepcopy(details or {}),
        "status": "Draft",
    }


def _transmission_repository() -> ComponentRepository:
    component = {
        "domain": "transmission",
        "component_id": "TRANS_TARGET",
        "component_name": "TRANS_TARGET",
        "status": "synthetic",
        "source": "unit_test",
        "notes": "synthetic",
        "trans_A": 8.0,
        "trans_B": 0.004,
        "trans_C": 0.0008,
        "loss_pct": 0.0,
        "component_type": "TRANSMISSION",
        "component_position": "NOT_APPLICABLE",
        "driveline_architecture": "FWD",
        "physical_boundary": "Synthetic transmission input-to-output boundary",
        "configuration_from": "baseline",
        "configuration_to": "target",
        "test_condition_type": "CONTROLLED",
        "test_method": "PL_DYNO",
        "hardware_reference": "SYNTH_TRANS_TARGET",
        "source_reference": "UNIT_TEST_SYNTHETIC",
        "net_bridge_eligible": "TRUE",
    }
    return ComponentRepository(
        domain="transmission",
        source="unit_test",
        _components=[component],
        _issues=[],
        _by_id={"TRANS_TARGET": component},
    )


def _sae_tire_record() -> dict:
    return {
        "id": 101,
        "code": "TIRE_SOURCE",
        "standard_family": "SAE",
        "rr_n_per_kn": 8.0,
        "sae_alpha": 1.0,
        "sae_beta": 1.0,
        "sae_a": 1.0,
        "sae_b": 0.1,
        "sae_c": 0.01,
        "sae_reference_load_n": 4000.0,
        "sae_reference_pressure_kpa": 250.0,
    }


class VdeRequestResolverTests(unittest.TestCase):
    def test_valid_baseline_materializes_canonical_total_and_net_results(self):
        result = resolve_vde_request(_workbook_with_domains({}), _baseline_context())

        baseline = result["baseline_result"]
        resolved = result["resolved_columns"]["baseline"]
        self.assertEqual(baseline["abc_total"], {"A": 120.0, "B": 0.02, "C": 0.01})
        self.assertIsNotNone(baseline["abc_net"])
        self.assertIsNotNone(baseline["vde_results"]["total"])
        self.assertIsNotNone(baseline["vde_results"]["net"])
        self.assertEqual(resolved["vde_total"], baseline["vde_results"]["total"])
        self.assertEqual(resolved["vde_net"], baseline["vde_results"]["net"])

    def _temp_db_path(self) -> Path:
        QA_DATA_DIR.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="qa_request_resolver_", dir=str(QA_DATA_DIR)))
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        return temp_dir / "qa_seed.db"

    def _qa_baseline_context(self) -> dict:
        return {
            "cycle_df": _cycle_df(),
            "legislation": "EPA",
            "category": "MIDSIZE",
            "mass_kg": 1600.0,
            "test_mass_kg": 1600.0,
            "weight_dist_fr_pct": 55.0,
            "CdA": 0.62,
            "rrc_N_per_kN": 8.0,
            "front_pressure_psi": 38.0,
            "rear_pressure_psi": 38.0,
            "front_tire_id": 920101,
            "rear_tire_id": 920101,
            "tire_db_id": 920101,
            "tire_code": "QA-BASE",
            "trans_A_coef_N": 10.0,
            "trans_B_coef_Npkph": 0.005,
            "trans_C_coef_Npkph2": 0.001,
        }

    def test_inherit_proposal_keeps_source_snapshot(self):
        workbook = _workbook_with_domains({})

        result = resolve_vde_request(workbook, _baseline_context())

        proposal = result["proposal_results"][0]
        self.assertEqual(proposal["domain_results"]["mass"]["status"], "OK")
        self.assertEqual(proposal["domain_results"]["aero"]["status"], "OK")
        self.assertAlmostEqual(proposal["resolved_snapshot"]["initial_abc_total"]["A"], 120.0)

    def test_absolute_aero_uses_walk_from_reference(self):
        workbook = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "proposal_req_1", "label": "Requested #1", "role": "walked"},
                {"key": "proposal_req_2", "label": "Requested #2", "role": "walked"},
            ],
            "columns": {
                "baseline": {"kind": "baseline"},
                "proposal_req_1": {"kind": "walked", "walk_from": "baseline"},
                "proposal_req_2": {"kind": "walked", "walk_from": "proposal_req_1"},
            },
            "proposals": {
                "proposal_req_1": {"aero": _proposal("aero", "AERO_ABSOLUTE_CDA", {"new_CdA": 0.70})},
                "proposal_req_2": {"aero": _proposal("aero", "AERO_ABSOLUTE_CDA", {"new_CdA": 0.75})},
            },
            "vde_request_import": {
                "baseline_printed": {"cda_m2": 0.62, "A": 120.0, "B": 0.02, "C": 0.01},
                "baseline_corrections": {},
                "effective_baseline": {
                    "legislation": "EPA",
                    "category": "MIDSIZE",
                    "mass_kg": 1600.0,
                    "test_mass_kg": 1736.0,
                    "A": 120.0,
                    "B": 0.02,
                    "C": 0.01,
                    "cda_m2": 0.62,
                },
                "columns": {
                    "proposal_req_1": {"proposal_id": "proposal_req_1", "display_index": 1, "source_column": "Requested #1"},
                    "proposal_req_2": {"proposal_id": "proposal_req_2", "display_index": 2, "source_column": "Requested #2"},
                },
            },
        }

        result = resolve_vde_request(workbook, _baseline_context())

        self.assertEqual(result["proposal_results"][0]["domain_results"]["aero"]["status"], "OK")
        self.assertEqual(result["proposal_results"][1]["domain_results"]["aero"]["status"], "OK")
        self.assertAlmostEqual(result["proposal_results"][0]["resolved_snapshot"]["CdA"], 0.70)
        self.assertAlmostEqual(result["proposal_results"][1]["resolved_snapshot"]["CdA"], 0.75)

    def test_absolute_aero_without_reference_is_missing(self):
        workbook = _workbook_with_domains({"aero": _proposal("aero", "AERO_ABSOLUTE_CDA", {"new_CdA": 0.75})})
        baseline = _baseline_context()
        baseline["CdA"] = None
        workbook["vde_request_import"]["baseline_printed"]["cda_m2"] = None
        workbook["vde_request_import"]["effective_baseline"]["cda_m2"] = None

        result = resolve_vde_request(workbook, baseline)

        self.assertEqual(result["proposal_results"][0]["domain_results"]["aero"]["status"], "Missing")

    def test_baseline_correction_is_preserved(self):
        workbook = _workbook_with_domains({})
        workbook["vde_request_import"]["baseline_corrections"] = {"mass_kg": 1650.0}
        workbook["vde_request_import"]["effective_baseline"]["mass_kg"] = 1650.0

        result = resolve_vde_request(workbook, _baseline_context())

        self.assertEqual(result["baseline"]["corrected_fields"], ["mass_kg"])
        self.assertEqual(result["baseline"]["effective"]["mass_kg"], 1650.0)

    def test_walk_from_invalid_blocks_only_dependent_proposal(self):
        workbook = _workbook_with_domains({"mass": _proposal("mass", "CUSTOM_MASS", {"test_mass_kg": 1800.0})}, walk_from="proposal_req_9")

        result = resolve_vde_request(workbook, _baseline_context())

        self.assertEqual(result["proposal_results"][0]["status"], "Blocked")

    def test_delta_brake_updates_total(self):
        workbook = _workbook_with_domains({"brake": _proposal("brake", "BRAKE_DRAG_CHANGE", {"change_mode": "Delta ABC", "delta_A": 2.0, "delta_B": 0.0, "delta_C": 0.0})})

        result = resolve_vde_request(workbook, _baseline_context())

        self.assertEqual(result["proposal_results"][0]["domain_results"]["brake"]["status"], "OK")
        self.assertGreater(result["proposal_results"][0]["resolved_snapshot"]["abc_total"]["A"], 120.0)

    def test_brake_residual_torque_accepts_total_without_legacy_method(self):
        proposal = _proposal("brake", "BRAKE_DRAG_CHANGE", {"residual_torque_total_Nm": 20.0, "wheel_radius_m": 0.5})
        proposal["selection_mode"] = "Residual torque"
        workbook = _workbook_with_domains({"brake": proposal})

        result = resolve_vde_request(workbook, _baseline_context())

        self.assertEqual(result["proposal_results"][0]["domain_results"]["brake"]["status"], "OK")
        self.assertAlmostEqual(result["proposal_results"][0]["resolved_snapshot"]["brake_A"], 44.0)

    def test_brake_residual_torque_accepts_canonical_front_rear_without_legacy_method(self):
        proposal = _proposal(
            "brake",
            "BRAKE_DRAG_CHANGE",
            {"residual_torque_front_Nm": 12.0, "residual_torque_rear_Nm": 8.0, "wheel_radius_m": 0.5},
        )
        proposal["selection_mode"] = "Residual torque"
        workbook = _workbook_with_domains({"brake": proposal})

        result = resolve_vde_request(workbook, _baseline_context())

        self.assertEqual(result["proposal_results"][0]["domain_results"]["brake"]["status"], "OK")
        self.assertAlmostEqual(result["proposal_results"][0]["resolved_snapshot"]["brake_A"], 44.0)

    def test_brake_residual_torque_without_torque_is_missing(self):
        proposal = _proposal("brake", "BRAKE_DRAG_CHANGE", {"wheel_radius_m": 0.5})
        proposal["selection_mode"] = "Residual torque"
        workbook = _workbook_with_domains({"brake": proposal})

        result = resolve_vde_request(workbook, _baseline_context())

        self.assertEqual(result["proposal_results"][0]["domain_results"]["brake"]["status"], "Missing")

    def test_lookup_component_not_found_is_missing(self):
        workbook = _workbook_with_domains({"transmission": _proposal("transmission", "TRANS_METADATA_ONLY", {"transmission_component_db_id": "UNKNOWN"})})

        result = resolve_vde_request(workbook, _baseline_context())

        self.assertEqual(result["proposal_results"][0]["domain_results"]["transmission"]["status"], "Missing")

    def test_tire_target_rrc_updates_snapshot_and_vde(self):
        workbook = _workbook_with_domains(
            {"tire": _proposal("tire", "TIRE_TARGET_RRC", {"target_rrc_N_per_kN": 8.5, "psi_front": 32.0, "psi_rear": 32.0})}
        )

        result = resolve_vde_request(workbook, _baseline_context())

        proposal = result["proposal_results"][0]
        self.assertEqual(proposal["domain_results"]["tire"]["status"], "OK")
        self.assertAlmostEqual(proposal["resolved_snapshot"]["rrc_N_per_kN"], 8.5)
        self.assertEqual(proposal["domain_results"]["tire"]["resolved_values"]["adjustment_method"], "Direct target RRC")
        self.assertGreater(proposal["resolved_snapshot"]["abc_total"]["A"], 120.0)
        self.assertIsNotNone(proposal["vde_results"]["total"])

    def test_mass_change_with_tire_inherit_recalculates_tire_component(self):
        workbook = _workbook_with_domains(
            {"mass": _proposal("mass", "PERFORMANCE_CURB_MASS", {"mass_kg": 1800.0, "preset": "Curb +100 kg"})}
        )

        result = resolve_vde_request(workbook, _baseline_context())

        proposal = result["proposal_results"][0]
        tire_result = proposal["domain_results"]["tire"]
        self.assertEqual(tire_result["status"], "OK")
        self.assertEqual(tire_result["resolved_values"]["tire_abc_method"], "RRC_LOAD_SCALING")
        self.assertEqual(tire_result["resolved_values"]["tire_load_mass_basis"], "TWC")
        self.assertNotEqual(proposal["resolved_snapshot"]["tire_A_final"], 50.0)

    def test_tire_sae_full_uses_resolved_effective_pressure(self):
        workbook = _workbook_with_domains(
            {
                "tire": _proposal(
                    "tire",
                    "TIRE_DB_LOOKUP",
                    {"tire_db_id": 101, "rrc_N_per_kN": 8.0, "front_pressure_psi": 39.0, "rear_pressure_psi": 39.0},
                )
            }
        )
        baseline = _baseline_context()
        baseline["front_pressure_psi"] = 36.0
        baseline["rear_pressure_psi"] = 36.0

        with patch("src.vde_core.vde_tire_proposal_resolver.get_tire_by_id", return_value=_sae_tire_record()), patch(
            "src.vde_core.vde_tire_proposal_resolver.calculate_vehicle_tire_abc",
            return_value={"applied_rr_n_per_kn": 8.0, "total_final_abc": {"A": 60.0, "B": 0.02, "C": 0.002}},
        ) as calculate:
            result = resolve_vde_request(workbook, baseline)

        proposal = result["proposal_results"][0]
        tire_inputs = calculate.call_args.kwargs["inputs"]
        self.assertAlmostEqual(tire_inputs["front_pressure_kpa"], 39.0 * 6.89475729)
        self.assertAlmostEqual(tire_inputs["rear_pressure_kpa"], 39.0 * 6.89475729)
        self.assertAlmostEqual(proposal["resolved_snapshot"]["front_pressure_psi"], 39.0)
        self.assertEqual(proposal["domain_results"]["tire"]["resolved_values"]["tire_abc_method"], "SAE_FULL")

    def test_tire_pressure_only_change_recalculates_rrc_and_tire_abc(self):
        workbook = _workbook_with_domains(
            {"tire": _proposal("tire", "TIRE_TARGET_RRC", {"front_pressure_psi": 39.0, "rear_pressure_psi": 39.0})}
        )
        baseline = _baseline_context()
        baseline["front_pressure_psi"] = 36.0
        baseline["rear_pressure_psi"] = 36.0

        result = resolve_vde_request(workbook, baseline)["proposal_results"][0]

        self.assertAlmostEqual(result["resolved_snapshot"]["front_pressure_psi"], 39.0)
        self.assertNotAlmostEqual(result["resolved_snapshot"]["rrc_N_per_kN"], baseline["rrc_N_per_kN"])
        self.assertNotAlmostEqual(result["resolved_snapshot"]["tire_A_final"], baseline["tire_A_final"])

    def test_tire_mass_and_pressure_change_use_resolved_load_and_pressure_together(self):
        workbook = _workbook_with_domains(
            {
                "mass": _proposal("mass", "PERFORMANCE_CURB_MASS", {"mass_kg": 1800.0, "preset": "Curb +100 kg"}),
                "tire": _proposal("tire", "TIRE_TARGET_RRC", {"front_pressure_psi": 39.0, "rear_pressure_psi": 39.0}),
            }
        )
        baseline = _baseline_context()
        baseline["front_pressure_psi"] = 36.0
        baseline["rear_pressure_psi"] = 36.0

        result = resolve_vde_request(workbook, baseline)["proposal_results"][0]

        self.assertAlmostEqual(result["resolved_snapshot"]["front_pressure_psi"], 39.0)
        self.assertNotAlmostEqual(result["resolved_snapshot"]["tire_load_mass_used_kg"], baseline["test_mass_kg"])
        self.assertNotAlmostEqual(result["resolved_snapshot"]["tire_A_final"], baseline["tire_A_final"])

    def test_walk_from_inherits_resolved_tire_pressure(self):
        workbook = {
            "scenarios": [
                {"key": "baseline", "label": "BASELINE_SYNTHETIC", "role": "baseline"},
                {"key": "proposal_req_1", "label": "REQUESTED_1", "role": "walked"},
                {"key": "proposal_req_2", "label": "REQUESTED_2", "role": "walked"},
            ],
            "columns": {
                "baseline": {"kind": "baseline"},
                "proposal_req_1": {"kind": "walked", "walk_from": "baseline"},
                "proposal_req_2": {"kind": "walked", "walk_from": "proposal_req_1"},
            },
            "proposals": {
                "proposal_req_1": {"tire": _proposal("tire", "TIRE_TARGET_RRC", {"front_pressure_psi": 39.0, "rear_pressure_psi": 39.0})},
                "proposal_req_2": {},
            },
            "vde_request_import": {
                "baseline_printed": {"mass_kg": 1600.0, "A": 120.0, "B": 0.02, "C": 0.01, "cda_m2": 0.62},
                "baseline_corrections": {},
                "effective_baseline": {"legislation": "EPA", "category": "SYNTHETIC", "mass_kg": 1600.0, "test_mass_kg": 1736.0, "A": 120.0, "B": 0.02, "C": 0.01, "cda_m2": 0.62},
                "columns": {
                    "proposal_req_1": {"proposal_id": "proposal_req_1", "display_index": 1, "source_column": "REQUESTED_1"},
                    "proposal_req_2": {"proposal_id": "proposal_req_2", "display_index": 2, "source_column": "REQUESTED_2"},
                },
            },
        }
        baseline = _baseline_context()
        baseline["front_pressure_psi"] = 36.0
        baseline["rear_pressure_psi"] = 36.0

        result = resolve_vde_request(workbook, baseline)

        req1 = result["proposal_results"][0]["resolved_snapshot"]
        req2 = result["proposal_results"][1]["resolved_snapshot"]
        self.assertAlmostEqual(req1["front_pressure_psi"], 39.0)
        self.assertAlmostEqual(req2["front_pressure_psi"], 39.0)
        self.assertAlmostEqual(req2["rear_pressure_psi"], 39.0)

    def test_epa_curb_to_twc_uses_target_curb_absolutely_even_with_corrected_baseline(self):
        workbook = _workbook_with_domains(
            {"mass": _proposal("mass", "EPA_CURB_TO_TWC", {"target_curb_mass_kg": 1850.0})}
        )
        workbook["vde_request_import"]["baseline_corrections"] = {"mass_kg": 2010.0}
        workbook["vde_request_import"]["effective_baseline"]["mass_kg"] = 2010.0

        result = resolve_vde_request(workbook, _baseline_context())

        proposal = result["proposal_results"][0]
        resolved_snapshot = dict(proposal["resolved_snapshot"] or {})
        mass_result = dict(proposal["domain_results"]["mass"]["resolved_values"] or {})

        self.assertEqual(resolved_snapshot["mass_kg"], 1850.0)
        self.assertEqual(resolved_snapshot["current_curb_mass_kg"], 2010.0)
        self.assertEqual(resolved_snapshot["target_curb_mass_kg"], 1850.0)
        self.assertEqual(mass_result["current_curb_mass_kg"], 2010.0)
        self.assertEqual(mass_result["target_curb_mass_kg"], 1850.0)

    def test_epa_curb_to_twc_second_request_replaces_first_target_instead_of_adding(self):
        workbook = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "proposal_req_1", "label": "Requested #1", "role": "walked"},
                {"key": "proposal_req_2", "label": "Requested #2", "role": "walked"},
            ],
            "columns": {
                "baseline": {"kind": "baseline"},
                "proposal_req_1": {"kind": "walked", "walk_from": "baseline"},
                "proposal_req_2": {"kind": "walked", "walk_from": "proposal_req_1"},
            },
            "proposals": {
                "proposal_req_1": {"mass": _proposal("mass", "EPA_CURB_TO_TWC", {"target_curb_mass_kg": 1850.0})},
                "proposal_req_2": {"mass": _proposal("mass", "EPA_CURB_TO_TWC", {"target_curb_mass_kg": 1700.0})},
            },
            "vde_request_import": {
                "baseline_printed": {"mass_kg": 2000.0, "test_mass_kg": 2136.0, "inertia_class": 2155.0, "A": 120.0, "B": 0.02, "C": 0.01, "cda_m2": 0.62},
                "baseline_corrections": {},
                "effective_baseline": {"legislation": "EPA", "category": "MIDSIZE", "mass_kg": 2000.0, "test_mass_kg": 2136.0, "inertia_class": 2155.0, "A": 120.0, "B": 0.02, "C": 0.01, "cda_m2": 0.62},
                "columns": {
                    "proposal_req_1": {"proposal_id": "proposal_req_1", "display_index": 1, "source_column": "Requested #1"},
                    "proposal_req_2": {"proposal_id": "proposal_req_2", "display_index": 2, "source_column": "Requested #2"},
                },
            },
        }

        result = resolve_vde_request(workbook, _baseline_context())

        req1 = dict(result["proposal_results"][0]["resolved_snapshot"] or {})
        req2 = dict(result["proposal_results"][1]["resolved_snapshot"] or {})

        self.assertEqual(req1["mass_kg"], 1850.0)
        self.assertEqual(req1["target_curb_mass_kg"], 1850.0)
        self.assertEqual(req2["current_curb_mass_kg"], 1850.0)
        self.assertEqual(req2["mass_kg"], 1700.0)
        self.assertEqual(req2["target_curb_mass_kg"], 1700.0)
        self.assertNotEqual(req2["mass_kg"], 3550.0)

    def test_not_used_keeps_aero_in_review(self):
        workbook = _workbook_with_domains({"aero": _proposal("aero", "AERO_NOT_USED", {})})

        result = resolve_vde_request(workbook, _baseline_context())

        self.assertEqual(result["proposal_results"][0]["domain_results"]["aero"]["status"], "Review")

    def test_transmission_total_and_net_semantics_are_preserved(self):
        workbook = _workbook_with_domains({"transmission": _proposal("transmission", "UPDATE_TRANS_DRAG_ABC", {"change_mode": "Absolute ABC", "new_trans_A": 8.0, "new_trans_B": 0.004, "new_trans_C": 0.001})})

        result = resolve_vde_request(workbook, _baseline_context())

        proposal = result["proposal_results"][0]
        self.assertIsNotNone(proposal["abc_total"])
        self.assertIsNotNone(proposal["abc_net"])
        self.assertGreater(proposal["vde_results"]["total"]["mj_per_km"], proposal["vde_results"]["net"]["mj_per_km"])
        self.assertEqual(proposal["resolved_snapshot"]["transmission_application_mode"], "APPLY_DELTA_TO_TOTAL")

    def test_transmission_delta_and_absolute_are_physically_equivalent(self):
        delta_workbook = _workbook_with_domains(
            {
                "transmission": _proposal(
                    "transmission",
                    "UPDATE_TRANS_DRAG_ABC",
                    {"change_mode": "Delta ABC", "delta_A": -2.0, "delta_B": -0.001, "delta_C": -0.0002},
                )
            }
        )
        absolute_workbook = _workbook_with_domains(
            {
                "transmission": _proposal(
                    "transmission",
                    "UPDATE_TRANS_DRAG_ABC",
                    {"change_mode": "Absolute ABC", "new_trans_A": 8.0, "new_trans_B": 0.004, "new_trans_C": 0.0008},
                )
            }
        )

        delta_result = resolve_vde_request(delta_workbook, _baseline_context())["proposal_results"][0]
        absolute_result = resolve_vde_request(absolute_workbook, _baseline_context())["proposal_results"][0]

        for key, expected_total, expected_net, expected_trans in (
            ("A", 118.0, 110.0, 8.0),
            ("B", 0.019, 0.015, 0.004),
            ("C", 0.0098, 0.009, 0.0008),
        ):
            self.assertAlmostEqual(delta_result["abc_total"][key], expected_total)
            self.assertAlmostEqual(absolute_result["abc_total"][key], expected_total)
            self.assertAlmostEqual(delta_result["abc_net"][key], expected_net)
            self.assertAlmostEqual(absolute_result["abc_net"][key], expected_net)
        self.assertAlmostEqual(delta_result["requested_snapshot"]["transmission_losses"]["A_TRANS"], 8.0)
        self.assertAlmostEqual(absolute_result["requested_snapshot"]["transmission_losses"]["A_TRANS"], 8.0)
        self.assertEqual(delta_result["resolved_snapshot"]["transmission_application_mode"], "APPLY_DELTA_TO_TOTAL")
        self.assertEqual(absolute_result["resolved_snapshot"]["transmission_application_mode"], "APPLY_DELTA_TO_TOTAL")

    def test_transmission_keep_total_fixed_preserves_total_and_recalculates_net(self):
        workbook = _workbook_with_domains(
            {
                "transmission": _proposal(
                    "transmission",
                    "UPDATE_TRANS_DRAG_ABC",
                    {
                        "change_mode": "Absolute ABC",
                        "new_trans_A": 10.0,
                        "new_trans_B": 0.005,
                        "new_trans_C": 0.0008,
                        "transmission_application_mode": "KEEP_TOTAL_FIXED",
                    },
                )
            }
        )

        result = resolve_vde_request(workbook, _baseline_context())["proposal_results"][0]

        self.assertAlmostEqual(result["abc_total"]["A"], 120.0)
        self.assertAlmostEqual(result["abc_total"]["B"], 0.020)
        self.assertAlmostEqual(result["abc_total"]["C"], 0.010)
        self.assertAlmostEqual(result["abc_net"]["A"], 110.0)
        self.assertAlmostEqual(result["abc_net"]["B"], 0.015)
        self.assertAlmostEqual(result["abc_net"]["C"], 0.0092)
        self.assertEqual(result["resolved_snapshot"]["transmission_application_mode"], "KEEP_TOTAL_FIXED")
        self.assertEqual(
            result["domain_results"]["transmission"]["resolved_values"]["transmission_mode"],
            "Fixed measured TOTAL - NET recalculated",
        )

    def test_transmission_neutral_change_matches_both_modes(self):
        keep_fixed_workbook = _workbook_with_domains(
            {
                "transmission": _proposal(
                    "transmission",
                    "UPDATE_TRANS_DRAG_ABC",
                    {
                        "change_mode": "Absolute ABC",
                        "new_trans_A": 10.0,
                        "new_trans_B": 0.005,
                        "new_trans_C": 0.001,
                        "transmission_application_mode": "KEEP_TOTAL_FIXED",
                    },
                )
            }
        )
        default_workbook = _workbook_with_domains(
            {
                "transmission": _proposal(
                    "transmission",
                    "UPDATE_TRANS_DRAG_ABC",
                    {
                        "change_mode": "Absolute ABC",
                        "new_trans_A": 10.0,
                        "new_trans_B": 0.005,
                        "new_trans_C": 0.001,
                    },
                )
            }
        )

        keep_fixed = resolve_vde_request(keep_fixed_workbook, _baseline_context())["proposal_results"][0]
        default_mode = resolve_vde_request(default_workbook, _baseline_context())["proposal_results"][0]

        self.assertEqual(keep_fixed["abc_total"], default_mode["abc_total"])
        self.assertEqual(keep_fixed["abc_net"], default_mode["abc_net"])

    def test_transmission_lookup_applies_component_delta_to_total(self):
        workbook = _workbook_with_domains(
            {"transmission": _proposal("transmission", "TRANS_METADATA_ONLY", {"transmission_component_db_id": "TRANS_TARGET"})}
        )

        result = resolve_vde_request(
            workbook,
            _baseline_context(),
            component_repositories={"transmission": _transmission_repository()},
        )["proposal_results"][0]

        self.assertAlmostEqual(result["abc_total"]["A"], 118.0)
        self.assertAlmostEqual(result["abc_total"]["B"], 0.019)
        self.assertAlmostEqual(result["abc_total"]["C"], 0.0098)
        self.assertAlmostEqual(result["abc_net"]["A"], 110.0)
        self.assertAlmostEqual(result["requested_snapshot"]["transmission_losses"]["A_TRANS"], 8.0)
        action = result["domain_results"]["transmission"]["component_action"]
        self.assertEqual(action["component_snapshot"]["net_bridge_eligible"], "TRUE")
        self.assertEqual(
            result["domain_results"]["transmission"]["resolved_values"]["component_provenance"]["physical_boundary"],
            "Synthetic transmission input-to-output boundary",
        )
        self.assertEqual(result["resolved_snapshot"]["transmission_application_mode"], "APPLY_DELTA_TO_TOTAL")

    def test_brake_vde_lookup_uses_selected_snapshot_without_component_db_identity(self):
        workbook = _workbook_with_domains(
            {
                "brake": _proposal(
                    "brake",
                    "BRAKE_METADATA_ONLY",
                    {
                        "brake_vde_db_id": 9901,
                        "brake_A_coef_N": 2.0,
                        "brake_B_Npkph": 0.0003,
                        "brake_C_coef_Npkph2": 0.00004,
                    },
                )
            }
        )

        result = resolve_vde_request(workbook, _baseline_context())["proposal_results"][0]

        self.assertAlmostEqual(result["resolved_snapshot"]["brake_A"], 2.0)
        self.assertAlmostEqual(result["abc_total"]["A"], 118.0)
        action = result["domain_results"]["brake"]["component_action"]
        self.assertEqual(action["action"], "reuse_vde_snapshot")
        self.assertEqual(action["vde_id"], 9901)

    def test_component_lookup_provenance_does_not_change_parasitic_math(self):
        workbook = _workbook_with_domains(
            {"parasitic": _proposal("parasitic", "PARASITIC_METADATA_ONLY", {"parasitic_component_db_id": "PARA-MOCK-001"})}
        )

        result = resolve_vde_request(workbook, _baseline_context())["proposal_results"][0]
        action = result["domain_results"]["parasitic"]["component_action"]

        self.assertAlmostEqual(result["abc_total"]["A"], 120.0)
        self.assertAlmostEqual(result["abc_total"]["B"], 0.02)
        self.assertAlmostEqual(result["abc_total"]["C"], 0.01)
        self.assertEqual(action["component_snapshot"]["component_type"], "OTHER_RESIDUAL_COMPONENT_LOSSES")
        self.assertIn("excluding explicit transmission brake axle and hub", action["component_snapshot"]["physical_boundary"])

    def test_axle_hubs_lookup_snapshot_preserves_boundary_metadata(self):
        workbook = _workbook_with_domains(
            {"axle_hubs": _proposal("axle_hubs", "AXLE_HUB_METADATA_ONLY", {"axle_hubs_component_db_id": "AXLE-MOCK-001"})}
        )

        result = resolve_vde_request(workbook, _baseline_context())["proposal_results"][0]
        action = result["domain_results"]["axle_hubs"]["component_action"]

        self.assertAlmostEqual(result["abc_total"]["A"], 120.5)
        self.assertEqual(action["component_snapshot"]["component_type"], "AXLE")
        self.assertEqual(action["component_snapshot"]["component_position"], "FRONT")

    def test_transmission_loss_percent_uses_single_total_and_net_pipeline(self):
        workbook = _workbook_with_domains(
            {"transmission": _proposal("transmission", "TRANS_LOSS_PCT", {"loss_pct": -20.0})}
        )

        result = resolve_vde_request(workbook, _baseline_context())["proposal_results"][0]

        self.assertAlmostEqual(result["abc_total"]["A"], 118.0)
        self.assertAlmostEqual(result["abc_net"]["A"], 110.0)
        self.assertAlmostEqual(result["requested_snapshot"]["transmission_losses"]["A_TRANS"], 8.0)
        self.assertEqual(result["resolved_snapshot"]["transmission_application_mode"], "APPLY_DELTA_TO_TOTAL")

    def test_transmission_coastdown_share_v1_keeps_total_fixed_and_uses_walk_from_total(self):
        workbook = _workbook_with_domains(
            {"transmission": _proposal("transmission", "TRANS_LOSS_PCT", {"loss_pct": 10.0, "rule_version": "COASTDOWN_SHARE_V1"})}
        )

        result = resolve_vde_request(workbook, _baseline_context())["proposal_results"][0]

        self.assertEqual(result["domain_results"]["transmission"]["status"], "OK")
        self.assertAlmostEqual(result["abc_total"]["A"], 120.0)
        self.assertAlmostEqual(result["requested_snapshot"]["transmission_losses"]["A_TRANS"], 12.0)
        self.assertAlmostEqual(result["abc_net"]["A"], 108.0)
        self.assertEqual(result["resolved_snapshot"]["transmission_application_mode"], "KEEP_TOTAL_FIXED")

    def test_transmission_inherit_walk_from_preserves_effective_losses_for_net(self):
        workbook = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "proposal_req_1", "label": "Requested #1", "role": "walked"},
                {"key": "proposal_req_2", "label": "Requested #2", "role": "walked"},
            ],
            "columns": {
                "baseline": {"kind": "baseline"},
                "proposal_req_1": {"kind": "walked", "walk_from": "baseline"},
                "proposal_req_2": {"kind": "walked", "walk_from": "proposal_req_1"},
            },
            "proposals": {
                "proposal_req_1": {
                    "transmission": _proposal(
                        "transmission",
                        "UPDATE_TRANS_DRAG_ABC",
                        {
                            "change_mode": "Absolute ABC",
                            "new_trans_A": 12.0,
                            "new_trans_B": 0.2,
                            "new_trans_C": 0.0,
                            "transmission_application_mode": "KEEP_TOTAL_FIXED",
                        },
                    )
                },
                "proposal_req_2": {
                    "transmission": _proposal("transmission", "INHERIT", {})
                },
            },
            "vde_request_import": {
                "baseline_printed": {"selected_baseline_vde_id": 5038, "mass_kg": 1600.0, "A": 120.0, "B": 0.02, "C": 0.01, "cda_m2": 0.62},
                "baseline_corrections": {},
                "effective_baseline": {
                    "selected_baseline_vde_id": 5038,
                    "legislation": "EPA",
                    "category": "MIDSIZE",
                    "make": "FORD",
                    "model": "TEST",
                    "year": 2026,
                    "cycle_name": "FTP75",
                    "mass_kg": 1600.0,
                    "test_mass_kg": 1736.0,
                    "A": 120.0,
                    "B": 0.02,
                    "C": 0.01,
                    "cda_m2": 0.62,
                },
                "columns": {
                    "proposal_req_1": {"proposal_id": "proposal_req_1", "display_index": 1, "source_column": "Requested #1"},
                    "proposal_req_2": {"proposal_id": "proposal_req_2", "display_index": 2, "source_column": "Requested #2"},
                },
            },
        }

        result = resolve_vde_request(workbook, _baseline_context())
        proposal_1 = result["proposal_results"][0]
        proposal_2 = result["proposal_results"][1]

        self.assertEqual(proposal_2["source_snapshot"]["transmission_losses"]["abc"]["A"], 12.0)
        self.assertAlmostEqual(proposal_1["resolved_snapshot"]["trans_A_coef_N"], 12.0)
        self.assertAlmostEqual(proposal_2["resolved_snapshot"]["trans_A_coef_N"], 12.0)
        self.assertAlmostEqual(proposal_2["resolved_snapshot"]["trans_B_coef_Npkph"], 0.2)
        self.assertEqual(proposal_1["resolved_snapshot"]["transmission_application_mode"], "KEEP_TOTAL_FIXED")
        self.assertEqual(proposal_2["source_snapshot"]["transmission_application_mode"], "KEEP_TOTAL_FIXED")
        self.assertEqual(proposal_2["resolved_snapshot"]["transmission_application_mode"], "KEEP_TOTAL_FIXED")
        self.assertAlmostEqual(proposal_2["abc_net"]["A"], proposal_2["abc_total"]["A"] - 12.0)
        self.assertAlmostEqual(proposal_2["abc_net"]["B"], proposal_2["abc_total"]["B"] - 0.2)
        self.assertAlmostEqual(proposal_2["abc_net"]["C"], proposal_2["abc_total"]["C"])
        self.assertNotEqual(
            proposal_2["vde_results"]["total"]["mj_per_km"],
            proposal_2["vde_results"]["net"]["mj_per_km"],
        )
        self.assertEqual(proposal_1["abc_total"], proposal_2["abc_total"])

    def test_tire_front_weight_default_is_warning_only_and_does_not_block_target_rrc(self):
        workbook = _workbook_with_domains(
            {"tire": _proposal("tire", "TIRE_TARGET_RRC", {"target_rrc_N_per_kN": 9.0})}
        )

        result = resolve_vde_request(workbook, _baseline_context_without_front_weight())
        proposal = result["proposal_results"][0]
        tire_domain = proposal["domain_results"]["tire"]

        self.assertEqual(proposal["status"], "OK")
        self.assertEqual(tire_domain["status"], "OK")
        self.assertEqual(proposal["resolved_snapshot"]["tire_front_weight_fraction"], 0.5)
        self.assertIn("weight_distribution_missing_default_50pct", proposal["preview_summary"]["warnings"])
        self.assertIn(
            "warning",
            {str(issue.get("severity") or "").lower() for issue in list(proposal.get("issues") or [])},
        )
        self.assertEqual(proposal["abc_net"]["A"], proposal["abc_total"]["A"] - proposal["resolved_snapshot"]["trans_A_coef_N"])

    def test_tire_target_rrc_requested_values_are_preserved_for_audit(self):
        workbook = _workbook_with_domains(
            {"tire": _proposal("tire", "TIRE_TARGET_RRC", {"target_rrc_N_per_kN": 9.0, "front_pressure_psi": 36.0, "rear_pressure_psi": 35.0, "tire_load_mass_basis": "TEST_MASS"})}
        )

        result = resolve_vde_request(workbook, _baseline_context())
        tire_domain = result["proposal_results"][0]["domain_results"]["tire"]

        self.assertEqual(tire_domain["requested_values"]["target_rrc_N_per_kN"], 9.0)
        self.assertEqual(tire_domain["requested_values"]["front_pressure_psi"], 36.0)
        self.assertEqual(tire_domain["requested_values"]["rear_pressure_psi"], 35.0)
        self.assertEqual(tire_domain["requested_values"]["tire_load_mass_basis"], "TEST_MASS")
        self.assertEqual(tire_domain["resolved_values"]["resolved_rrc_N_per_kN"], 9.0)

    def test_simple_component_delta_and_absolute_rules_are_preserved(self):
        cases = [
            ("brake", "BRAKE_DRAG_CHANGE", {"change_mode": "Delta ABC", "delta_A": 2.0, "delta_B": 0.0, "delta_C": 0.0}, 122.0),
            ("brake", "BRAKE_DRAG_CHANGE", {"change_mode": "Absolute ABC", "brake_A": 6.0, "brake_B": 0.0008, "brake_C": 0.0001}, 122.0),
            ("axle_hubs", "AXLE_HUB_DRAG_CHANGE", {"change_mode": "Delta ABC", "delta_A": 2.0, "delta_B": 0.0, "delta_C": 0.0}, 122.0),
            ("axle_hubs", "AXLE_HUB_DRAG_CHANGE", {"change_mode": "Absolute ABC", "axle_hub_A": 4.0, "axle_hub_B": 0.0004, "axle_hub_C": 0.0001}, 122.0),
            ("parasitic", "PARASITIC_LOSS_CHANGE", {"change_mode": "Delta ABC", "delta_A": 2.0, "delta_B": 0.0, "delta_C": 0.0}, 122.0),
            ("parasitic", "PARASITIC_LOSS_CHANGE", {"change_mode": "Absolute ABC", "parasitic_A": 5.0, "parasitic_B": 0.0005, "parasitic_C": 0.0001}, 122.0),
        ]
        for domain, proposal_type, details, expected_total_a in cases:
            with self.subTest(domain=domain, details=details):
                workbook = _workbook_with_domains({domain: _proposal(domain, proposal_type, details)})

                result = resolve_vde_request(workbook, _baseline_context())["proposal_results"][0]

                self.assertAlmostEqual(result["abc_total"]["A"], expected_total_a)

    def test_walk_from_uses_selected_resolved_source_for_later_component_delta(self):
        workbook = {
            "scenarios": [
                {"key": "baseline", "label": "BASELINE_SYNTHETIC", "role": "baseline"},
                {"key": "proposal_req_1", "label": "REQUESTED_1", "role": "walked"},
                {"key": "proposal_req_2", "label": "REQUESTED_2", "role": "walked"},
            ],
            "columns": {
                "baseline": {"kind": "baseline"},
                "proposal_req_1": {"kind": "walked", "walk_from": "baseline"},
                "proposal_req_2": {"kind": "walked", "walk_from": "proposal_req_1"},
            },
            "proposals": {
                "proposal_req_1": {"brake": _proposal("brake", "BRAKE_DRAG_CHANGE", {"change_mode": "Delta ABC", "delta_A": 2.0, "delta_B": 0.0, "delta_C": 0.0})},
                "proposal_req_2": {"parasitic": _proposal("parasitic", "PARASITIC_LOSS_CHANGE", {"change_mode": "Delta ABC", "delta_A": 3.0, "delta_B": 0.0, "delta_C": 0.0})},
            },
            "vde_request_import": {
                "baseline_printed": {"mass_kg": 1600.0, "A": 120.0, "B": 0.02, "C": 0.01, "cda_m2": 0.62},
                "baseline_corrections": {},
                "effective_baseline": {"legislation": "EPA", "category": "SYNTHETIC", "mass_kg": 1600.0, "test_mass_kg": 1736.0, "A": 120.0, "B": 0.02, "C": 0.01, "cda_m2": 0.62},
                "columns": {
                    "proposal_req_1": {"proposal_id": "proposal_req_1", "display_index": 1, "source_column": "REQUESTED_1"},
                    "proposal_req_2": {"proposal_id": "proposal_req_2", "display_index": 2, "source_column": "REQUESTED_2"},
                },
            },
        }
        baseline_walk = deepcopy(workbook)
        baseline_walk["columns"]["proposal_req_2"]["walk_from"] = "baseline"

        chained_result = resolve_vde_request(workbook, _baseline_context())["proposal_results"][1]
        baseline_result = resolve_vde_request(baseline_walk, _baseline_context())["proposal_results"][1]

        self.assertAlmostEqual(chained_result["abc_total"]["A"], 125.0)
        self.assertAlmostEqual(baseline_result["abc_total"]["A"], 123.0)

    def test_inherit_components_do_not_create_total_delta(self):
        result = resolve_vde_request(_workbook_with_domains({}), _baseline_context())["proposal_results"][0]

        self.assertAlmostEqual(result["abc_total"]["A"], 120.0)
        self.assertAlmostEqual(result["abc_net"]["A"], 110.0)

    def test_transmission_not_used_zeroes_component_and_keeps_net_consistent(self):
        workbook = _workbook_with_domains({"transmission": _proposal("transmission", "TRANS_LOSS_NOT_AVAILABLE", {})})

        result = resolve_vde_request(workbook, _baseline_context())

        proposal = result["proposal_results"][0]
        self.assertEqual(proposal["domain_results"]["transmission"]["status"], "OK")
        self.assertAlmostEqual(proposal["abc_total"]["A"], 110.0)
        self.assertAlmostEqual(proposal["abc_net"]["A"], 110.0)
        self.assertAlmostEqual(proposal["requested_snapshot"]["transmission_losses"]["A_TRANS"], 0.0)
        self.assertIsNotNone(proposal["vde_results"]["net"])

    def test_component_not_used_zeroes_component_triplets(self):
        for domain_key, proposal_type, snapshot_key in (
            ("brake", "BRAKE_NOT_USED", "brake_A"),
            ("axle_hubs", "AXLE_HUB_NOT_USED", "axle_hub_A"),
            ("parasitic", "PARASITIC_NOT_USED", "parasitic_A"),
        ):
            with self.subTest(domain=domain_key):
                workbook = _workbook_with_domains({domain_key: _proposal(domain_key, proposal_type, {})})

                result = resolve_vde_request(workbook, _baseline_context())

                proposal = result["proposal_results"][0]
                self.assertEqual(proposal["domain_results"][domain_key]["status"], "OK")
                self.assertEqual(proposal["resolved_snapshot"][snapshot_key], 0.0)

    def test_tire_not_used_zeroes_tire_component(self):
        workbook = _workbook_with_domains({"tire": _proposal("tire", "TIRE_METADATA_ONLY", {})})

        result = resolve_vde_request(workbook, _baseline_context())

        proposal = result["proposal_results"][0]
        self.assertEqual(proposal["domain_results"]["tire"]["status"], "OK")
        self.assertEqual(proposal["resolved_snapshot"]["tire_A_final"], 0.0)
        self.assertAlmostEqual(proposal["abc_total"]["A"], 70.0)

    def test_qa_neutral_tire_lookup_preserves_measured_total_and_curve_points(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)
        workbook = _workbook_with_domains(
            {
                "tire": _proposal(
                    "tire",
                    "TIRE_DB_LOOKUP",
                    {
                        "tire_db_id": 920105,
                        "tire_code": "QA-NEUTRAL",
                        "rrc_N_per_kN": 8.0,
                        "front_pressure_psi": 38.0,
                        "rear_pressure_psi": 38.0,
                        "tire_load_mass_basis": "TEST_MASS",
                    },
                )
            }
        )
        workbook["vde_request_import"]["baseline_printed"].update({"A": 118.0, "B": 0.019, "C": 0.0098, "mass_kg": 1600.0})
        workbook["vde_request_import"]["effective_baseline"].update({"A": 118.0, "B": 0.019, "C": 0.0098, "mass_kg": 1600.0, "test_mass_kg": 1600.0})

        with db_module.using_db_path(db_path):
            proposal = resolve_vde_request(workbook, self._qa_baseline_context())["proposal_results"][0]

        source_total = proposal["source_snapshot"]["initial_abc_total"]
        delta_tire = proposal["domain_results"]["tire"]["resolved_values"]["delta_tire_ABC"]
        result_total = proposal["abc_total"]
        self.assertEqual(proposal["domain_results"]["tire"]["status"], "OK")
        for key in ("A", "B", "C"):
            self.assertAlmostEqual(delta_tire[key], 0.0, places=9)
            self.assertAlmostEqual(result_total[key], source_total[key], places=9)
            self.assertAlmostEqual(result_total[key], source_total[key] + delta_tire[key], places=9)
        for speed_kph in (0.0, 50.0, 100.0):
            self.assertAlmostEqual(
                roadload_force_N(source_total["A"], source_total["B"], source_total["C"], speed_kph),
                roadload_force_N(result_total["A"], result_total["B"], result_total["C"], speed_kph),
                places=9,
            )
        self.assertAlmostEqual(proposal["abc_net"]["A"], proposal["abc_total"]["A"] - proposal["resolved_snapshot"]["trans_A_coef_N"])
        self.assertAlmostEqual(proposal["abc_net"]["B"], proposal["abc_total"]["B"] - proposal["resolved_snapshot"]["trans_B_coef_Npkph"])
        self.assertAlmostEqual(proposal["abc_net"]["C"], proposal["abc_total"]["C"] - proposal["resolved_snapshot"]["trans_C_coef_Npkph2"])

    def test_real_seeded_baseline_shape_keeps_qa_neutral_lookup_delta_neutral(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)

        with db_module.using_db_path(db_path):
            baseline_row = fetch_vde_by_id(900001)
            state = apply_v22_baseline(create_v22_state(), baseline_row)
            baseline = dict(state["baseline"])
            workbook = {
                "scenarios": [
                    {"key": "baseline", "label": "Baseline", "role": "baseline"},
                    {"key": "proposal_req_1", "label": "Requested #1", "role": "walked"},
                ],
                "columns": {
                    "baseline": {"kind": "baseline"},
                    "proposal_req_1": {"kind": "walked", "walk_from": "baseline"},
                },
                "proposals": {
                    "proposal_req_1": {
                        "tire": _proposal(
                            "tire",
                            "TIRE_DB_LOOKUP",
                            {
                                "tire_db_id": 920105,
                                "tire_code": "QA-NEUTRAL",
                                "rrc_N_per_kN": 8.0,
                                "front_pressure_psi": 38.0,
                                "rear_pressure_psi": 38.0,
                                "tire_load_mass_basis": "TEST_MASS",
                            },
                        )
                    }
                },
                "vde_request_import": {
                    "baseline_printed": deepcopy(dict(baseline.get("printed") or {})),
                    "baseline_corrections": deepcopy(dict(baseline.get("corrections") or {})),
                    "effective_baseline": deepcopy(dict(baseline.get("effective") or {})),
                    "columns": {
                        "proposal_req_1": {
                            "proposal_id": "proposal_req_1",
                            "display_index": 1,
                            "source_column": "Requested #1",
                        }
                    },
                },
            }
            result = resolve_vde_request(workbook, compact_baseline_context(state))

        proposal = result["proposal_results"][0]
        tire_values = proposal["domain_results"]["tire"]["resolved_values"]
        source_total = proposal["source_snapshot"]["initial_abc_total"]
        result_total = proposal["abc_total"]
        delta_tire = tire_values["delta_tire_ABC"]

        self.assertEqual(proposal["domain_results"]["tire"]["status"], "OK")
        self.assertEqual(proposal["source_snapshot"]["front_tire_id"], 920101)
        self.assertIsNone(proposal["source_snapshot"].get("tire_A_final"))
        self.assertIsNone(proposal["source_snapshot"].get("tire_B_final"))
        self.assertIsNone(proposal["source_snapshot"].get("tire_C_final"))
        for key in ("A", "B", "C"):
            self.assertAlmostEqual(delta_tire[key], 0.0, places=9)
            self.assertAlmostEqual(result_total[key], source_total[key], places=9)

    def test_same_reference_rrc_sae_lookup_changes_measured_total_by_tire_delta(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)
        workbook = _workbook_with_domains(
            {
                "tire": _proposal(
                    "tire",
                    "TIRE_DB_LOOKUP",
                    {
                        "tire_db_id": 920109,
                        "tire_code": "QA-SAME-RRC-DIFF-SAE",
                        "rrc_N_per_kN": 8.0,
                        "front_pressure_psi": 38.0,
                        "rear_pressure_psi": 38.0,
                        "tire_load_mass_basis": "TEST_MASS",
                    },
                )
            }
        )
        workbook["vde_request_import"]["baseline_printed"].update({"A": 118.0, "B": 0.019, "C": 0.0098, "mass_kg": 1600.0})
        workbook["vde_request_import"]["effective_baseline"].update({"A": 118.0, "B": 0.019, "C": 0.0098, "mass_kg": 1600.0, "test_mass_kg": 1600.0})

        with db_module.using_db_path(db_path):
            proposal = resolve_vde_request(workbook, self._qa_baseline_context())["proposal_results"][0]

        source_total = proposal["source_snapshot"]["initial_abc_total"]
        result_total = proposal["abc_total"]
        delta_tire = proposal["domain_results"]["tire"]["resolved_values"]["delta_tire_ABC"]
        self.assertEqual(proposal["domain_results"]["tire"]["status"], "OK")
        self.assertAlmostEqual(proposal["resolved_snapshot"]["tire_source_rrc_N_per_kN"], 8.0, places=6)
        self.assertAlmostEqual(proposal["resolved_snapshot"]["tire_target_rrc_N_per_kN"], 8.0, places=6)
        self.assertTrue(any(abs(float(delta_tire[key])) > 1e-9 for key in ("A", "B", "C")))
        for key in ("A", "B", "C"):
            self.assertAlmostEqual(result_total[key], source_total[key] + delta_tire[key], places=9)
        source_curve = [roadload_force_N(source_total["A"], source_total["B"], source_total["C"], speed) for speed in (0.0, 50.0, 80.0, 120.0)]
        result_curve = [roadload_force_N(result_total["A"], result_total["B"], result_total["C"], speed) for speed in (0.0, 50.0, 80.0, 120.0)]
        self.assertNotEqual(source_curve, result_curve)

    def test_walk_from_inherit_does_not_reapply_tire_delta(self):
        workbook = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "proposal_req_1", "label": "Requested #1", "role": "walked"},
                {"key": "proposal_req_2", "label": "Requested #2", "role": "walked"},
            ],
            "columns": {
                "baseline": {"kind": "baseline"},
                "proposal_req_1": {"kind": "walked", "walk_from": "baseline"},
                "proposal_req_2": {"kind": "walked", "walk_from": "proposal_req_1"},
            },
            "proposals": {
                "proposal_req_1": {"tire": _proposal("tire", "TIRE_TARGET_RRC", {"target_rrc_N_per_kN": 7.5, "front_pressure_psi": 38.0, "rear_pressure_psi": 38.0})},
                "proposal_req_2": {},
            },
            "vde_request_import": {
                "baseline_printed": {"selected_baseline_vde_id": 900001, "mass_kg": 1600.0, "A": 118.0, "B": 0.019, "C": 0.0098, "cda_m2": 0.62},
                "baseline_corrections": {},
                "effective_baseline": {"selected_baseline_vde_id": 900001, "legislation": "EPA", "category": "MIDSIZE", "mass_kg": 1600.0, "test_mass_kg": 1600.0, "A": 118.0, "B": 0.019, "C": 0.0098, "cda_m2": 0.62},
                "columns": {
                    "proposal_req_1": {"proposal_id": "proposal_req_1", "display_index": 1, "source_column": "Requested #1"},
                    "proposal_req_2": {"proposal_id": "proposal_req_2", "display_index": 2, "source_column": "Requested #2"},
                },
            },
        }

        result = resolve_vde_request(workbook, self._qa_baseline_context())

        req1 = result["proposal_results"][0]
        req2 = result["proposal_results"][1]
        for key in ("A", "B", "C"):
            self.assertAlmostEqual(req2["abc_total"][key], req1["abc_total"][key], places=9)
            self.assertAlmostEqual(req2["domain_results"]["tire"]["resolved_values"]["delta_tire_ABC"][key], 0.0, places=9)

    def test_walk_from_mass_change_recalculates_incremental_tire_delta(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)
        workbook = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "proposal_req_1", "label": "Requested #1", "role": "walked"},
                {"key": "proposal_req_2", "label": "Requested #2", "role": "walked"},
            ],
            "columns": {
                "baseline": {"kind": "baseline"},
                "proposal_req_1": {"kind": "walked", "walk_from": "baseline"},
                "proposal_req_2": {"kind": "walked", "walk_from": "proposal_req_1"},
            },
            "proposals": {
                "proposal_req_1": {
                    "tire": _proposal(
                        "tire",
                        "TIRE_DB_LOOKUP",
                        {
                            "tire_db_id": 920104,
                            "tire_code": "QA-LOAD",
                            "rrc_N_per_kN": 8.8,
                            "front_pressure_psi": 30.0,
                            "rear_pressure_psi": 30.0,
                            "tire_load_mass_basis": "TEST_MASS",
                        },
                    )
                },
                "proposal_req_2": {
                    "mass": _proposal("mass", "PERFORMANCE_CURB_MASS", {"mass_kg": 1800.0, "preset": "Curb +100 kg"})
                },
            },
            "vde_request_import": {
                "baseline_printed": {"selected_baseline_vde_id": 900001, "mass_kg": 1600.0, "A": 118.0, "B": 0.019, "C": 0.0098, "cda_m2": 0.62},
                "baseline_corrections": {},
                "effective_baseline": {"selected_baseline_vde_id": 900001, "legislation": "EPA", "category": "MIDSIZE", "mass_kg": 1600.0, "test_mass_kg": 1600.0, "A": 118.0, "B": 0.019, "C": 0.0098, "cda_m2": 0.62},
                "columns": {
                    "proposal_req_1": {"proposal_id": "proposal_req_1", "display_index": 1, "source_column": "Requested #1"},
                    "proposal_req_2": {"proposal_id": "proposal_req_2", "display_index": 2, "source_column": "Requested #2"},
                },
            },
        }

        with db_module.using_db_path(db_path):
            result = resolve_vde_request(workbook, self._qa_baseline_context())

        req1 = result["proposal_results"][0]
        req2 = result["proposal_results"][1]
        req1_tire = req1["resolved_snapshot"]
        req2_tire = req2["resolved_snapshot"]
        for key, tire_key in (("A", "tire_A_final"), ("B", "tire_B_final"), ("C", "tire_C_final")):
            expected_delta = req2_tire[tire_key] - req1_tire[tire_key]
            actual_delta = req2["domain_results"]["tire"]["resolved_values"]["delta_tire_ABC"][key]
            self.assertAlmostEqual(actual_delta, expected_delta, places=9)
            self.assertAlmostEqual(req2["abc_total"][key], req1["abc_total"][key] + actual_delta, places=9)
        self.assertGreater(req2["resolved_snapshot"]["tire_load_mass_used_kg"], req1["resolved_snapshot"]["tire_load_mass_used_kg"])

    def test_walk_from_inherit_preserves_removed_component(self):
        workbook = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "proposal_req_1", "label": "Requested #1", "role": "walked"},
                {"key": "proposal_req_2", "label": "Requested #2", "role": "walked"},
            ],
            "columns": {
                "baseline": {"kind": "baseline"},
                "proposal_req_1": {"kind": "walked", "walk_from": "baseline"},
                "proposal_req_2": {"kind": "walked", "walk_from": "proposal_req_1"},
            },
            "proposals": {
                "proposal_req_1": {"brake": _proposal("brake", "BRAKE_NOT_USED", {})},
                "proposal_req_2": {},
            },
            "vde_request_import": {
                "baseline_printed": {"selected_baseline_vde_id": 5038, "mass_kg": 1600.0, "A": 120.0, "B": 0.02, "C": 0.01, "cda_m2": 0.62},
                "baseline_corrections": {},
                "effective_baseline": {"selected_baseline_vde_id": 5038, "legislation": "EPA", "category": "MIDSIZE", "make": "FORD", "model": "TEST", "year": 2026, "cycle_name": "FTP75", "mass_kg": 1600.0, "test_mass_kg": 1736.0, "A": 120.0, "B": 0.02, "C": 0.01, "cda_m2": 0.62},
                "columns": {
                    "proposal_req_1": {"proposal_id": "proposal_req_1", "display_index": 1, "source_column": "Requested #1"},
                    "proposal_req_2": {"proposal_id": "proposal_req_2", "display_index": 2, "source_column": "Requested #2"},
                },
            },
        }

        result = resolve_vde_request(workbook, _baseline_context())

        req1 = result["proposal_results"][0]["resolved_snapshot"]
        req2 = result["proposal_results"][1]["resolved_snapshot"]
        self.assertEqual(req1["brake_A"], 0.0)
        self.assertEqual(req2["brake_A"], 0.0)

    def test_component_actions_cover_manual_complete_and_snapshot_only(self):
        workbook = _workbook_with_domains(
            {
                "brake": _proposal("brake", "BRAKE_DRAG_CHANGE", {"change_mode": "Absolute ABC", "brake_A": 5.0, "brake_B": 0.001, "brake_C": 0.0001}),
                "parasitic": _proposal("parasitic", "PARASITIC_LOSS_CHANGE", {"change_mode": "Absolute ABC", "parasitic_A": 4.0}),
            }
        )

        result = resolve_vde_request(workbook, _baseline_context())

        actions = {item["domain"]: item["action"] for item in result["proposal_results"][0]["component_actions"]}
        self.assertEqual(actions["brake"], "eligible_for_new_component")
        self.assertEqual(actions["parasitic"], "snapshot_only")

    def test_chain_with_three_proposals_is_serializable_and_preserves_ids(self):
        workbook = {
            "scenarios": [
                {"key": "baseline", "label": "Baseline", "role": "baseline"},
                {"key": "proposal_req_1", "label": "Requested #1", "role": "walked"},
                {"key": "proposal_req_2", "label": "Requested #2", "role": "walked"},
                {"key": "proposal_req_3", "label": "Requested #3", "role": "walked"},
            ],
            "columns": {
                "baseline": {"kind": "baseline"},
                "proposal_req_1": {"kind": "walked", "walk_from": "baseline"},
                "proposal_req_2": {"kind": "walked", "walk_from": "proposal_req_1"},
                "proposal_req_3": {"kind": "walked", "walk_from": "proposal_req_2"},
            },
            "proposals": {
                "proposal_req_1": {"mass": _proposal("mass", "CUSTOM_MASS", {"test_mass_kg": 1800.0})},
                "proposal_req_2": {"aero": _proposal("aero", "AERO_DELTA_CDA", {"delta_CdA": 0.01})},
                "proposal_req_3": {"brake": _proposal("brake", "BRAKE_DRAG_CHANGE", {"change_mode": "Delta ABC", "delta_A": 1.0, "delta_B": 0.0, "delta_C": 0.0})},
            },
            "vde_request_import": {
                "baseline_printed": {"A": 120.0, "B": 0.02, "C": 0.01, "mass_kg": 1600.0, "cda_m2": 0.62},
                "baseline_corrections": {},
                "effective_baseline": {"legislation": "EPA", "category": "MIDSIZE", "mass_kg": 1600.0, "test_mass_kg": 1736.0, "A": 120.0, "B": 0.02, "C": 0.01, "cda_m2": 0.62},
                "columns": {
                    "proposal_req_1": {"proposal_id": "proposal_req_1", "display_index": 1, "source_column": "Requested #1"},
                    "proposal_req_2": {"proposal_id": "proposal_req_2", "display_index": 2, "source_column": "Requested #2"},
                    "proposal_req_3": {"proposal_id": "proposal_req_3", "display_index": 3, "source_column": "Requested #3"},
                },
            },
        }

        result = resolve_vde_request(workbook, _baseline_context())

        self.assertEqual(len(result["proposal_results"]), 3)
        self.assertEqual(result["proposal_results"][2]["proposal_id"], "proposal_req_3")
        json.dumps(result, default=str)


if __name__ == "__main__":
    unittest.main()
