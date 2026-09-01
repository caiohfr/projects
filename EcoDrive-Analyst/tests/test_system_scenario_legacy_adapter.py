"""Sprint 11A: tests for src.vde_core.system_scenario.legacy_adapter --
proving legacy vde_db/fuelcons_db rows can populate canonical Domain States
without the canonical contracts depending on raw row layout (INV-11-012).
Uses the same QA fixture (vde_id=900001, fc:900102/900104) established
across Sprints 10A-10E.
"""

from __future__ import annotations

import gc
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.vde_core import db as db_module
from src.vde_core.database_management_contract import EntityType
from src.vde_core.database_management_service import get_record
from src.vde_core.qa_mock_data import seed_qa_database, seed_qa_fuelcons_mock_rows
from src.vde_core.repositories import fetch_vde_by_id
from src.vde_core.system_scenario import (
    ArchitectureClass,
    ArchitectureConfiguration,
    DomainKind,
    EngineConfiguration,
    ProvenanceKind,
    TransmissionConfiguration,
    VehicleDemandConfiguration,
    architecture_domain_state_from_legacy_vde_row,
    aux_thermal_domain_state_from_legacy_row,
    controls_domain_state_from_legacy_row,
    electric_drive_domain_state_sparse,
    energy_storage_domain_state_from_legacy_row,
    engine_domain_state_from_legacy_row,
    transmission_domain_state_from_legacy_row,
    vehicle_demand_domain_state_from_legacy_vde_row,
    vehicle_demand_domain_state_from_result,
)
from src.vde_core.vehicle_demand import RoadloadBasis, VehicleDemandResult, VehicleDemandSummary


class LegacyAdapterQaDatabaseTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "system_scenario_adapter_qa.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        seed_qa_fuelcons_mock_rows(self.db_path)
        db_module.configure_db_path(self.db_path)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def _vde_row(self):
        return fetch_vde_by_id(900001)

    def _fuelcons_row(self, fuelcons_id=900102):
        return dict(get_record(EntityType.FUEL_CONSUMPTION, fuelcons_id))

    def test_vehicle_demand_adapter_produces_a_real_quantitative_result_from_a_legacy_row(self):
        state = vehicle_demand_domain_state_from_legacy_vde_row(self._vde_row(), source_identity="vde:900001")
        self.assertEqual(state.domain, DomainKind.VEHICLE_DEMAND)
        self.assertIsInstance(state.configuration, VehicleDemandConfiguration)
        result = state.configuration.vehicle_demand_result
        self.assertIsNotNone(result)
        self.assertIsInstance(result, VehicleDemandResult)
        self.assertIsNotNone(result.total_summary.vde_mj_per_km)
        self.assertGreater(result.total_summary.vde_mj_per_km, 0.0)
        self.assertEqual(state.configuration.source_identity, "vde:900001")
        self.assertEqual(state.provenance, ProvenanceKind.CALCULATED)

    def test_vehicle_demand_adapter_never_mutates_the_source_row(self):
        row = self._vde_row()
        original = dict(row)
        vehicle_demand_domain_state_from_legacy_vde_row(row, source_identity="vde:900001")
        self.assertEqual(row, original)

    def test_vehicle_demand_adapter_reuses_the_same_frozen_core_functions_as_quick_scenario(self):
        # Reuse proof: same call chain as quick_scenario/resolver.py -- not a
        # parallel/second Vehicle Demand computation path.
        from src.vde_core.quick_scenario import resolver as quick_scenario_resolver
        from src.vde_core.system_scenario import legacy_adapter as system_scenario_adapter

        self.assertIs(
            system_scenario_adapter.build_vehicle_demand_request,
            quick_scenario_resolver.build_vehicle_demand_request,
        )
        self.assertIs(
            system_scenario_adapter.calculate_vehicle_demand,
            quick_scenario_resolver.calculate_vehicle_demand,
        )

    def test_architecture_adapter_reads_electrification_from_fuelcons_row_not_vde_row(self):
        vde_row = self._vde_row()
        fuelcons_row = self._fuelcons_row(900102)
        self.assertEqual(fuelcons_row.get("electrification"), "ICE")
        state = architecture_domain_state_from_legacy_vde_row(vde_row, fuelcons_row)
        self.assertIsInstance(state.configuration, ArchitectureConfiguration)
        self.assertEqual(state.configuration.architecture_class, ArchitectureClass.ICE)
        self.assertEqual(state.provenance, ProvenanceKind.SOURCE_OBSERVED)

    def test_architecture_adapter_without_fuelcons_row_is_explicit_not_available_not_a_guess(self):
        vde_row = self._vde_row()
        state = architecture_domain_state_from_legacy_vde_row(vde_row, None)
        self.assertIsNone(state.configuration.architecture_class)
        self.assertEqual(state.provenance, ProvenanceKind.NOT_AVAILABLE)

    def test_engine_adapter_populates_fuel_type_and_rated_power_from_fuelcons_row(self):
        vde_row = self._vde_row()
        fuelcons_row = self._fuelcons_row(900102)
        state = engine_domain_state_from_legacy_row(vde_row, fuelcons_row)
        self.assertIsInstance(state.configuration, EngineConfiguration)
        self.assertEqual(state.configuration.fuel_type, fuelcons_row.get("fuel_type"))
        self.assertEqual(state.configuration.rated_power_kw, fuelcons_row.get("engine_max_power_kw"))

    def test_transmission_adapter_populates_type_from_vde_row_and_gear_fdr_from_fuelcons_row(self):
        vde_row = self._vde_row()
        fuelcons_row = self._fuelcons_row(900102)
        state = transmission_domain_state_from_legacy_row(vde_row, fuelcons_row)
        self.assertIsInstance(state.configuration, TransmissionConfiguration)
        self.assertEqual(state.configuration.transmission_type, vde_row.get("transmission_type"))
        self.assertEqual(state.configuration.gear_count, fuelcons_row.get("gear_count"))
        self.assertEqual(state.configuration.final_drive_ratio, fuelcons_row.get("final_drive_ratio"))

    def test_legacy_adapter_isolation_canonical_contract_never_exposes_raw_row_keys(self):
        # INV-11-012: canonical contracts must not require callers to
        # understand raw fuelcons_db schema. The resulting DomainSourceState
        # exposes only its own typed fields -- never a raw dict passthrough.
        vde_row = self._vde_row()
        fuelcons_row = self._fuelcons_row(900102)
        state = engine_domain_state_from_legacy_row(vde_row, fuelcons_row)
        exposed_fields = set(state.configuration.__dataclass_fields__)
        raw_row_only_keys = {"id", "vde_id", "record_origin", "assumptions_json", "provenance_json"}
        self.assertEqual(exposed_fields & raw_row_only_keys, set())


class LegacyAdapterNoDbTests(unittest.TestCase):
    def test_vehicle_demand_domain_state_from_result_never_touches_the_database(self):
        result = VehicleDemandResult(
            total_summary=VehicleDemandSummary(roadload_basis=RoadloadBasis.TOTAL, vde_mj_per_km=1.2),
            net_summary=None,
        )
        state = vehicle_demand_domain_state_from_result(result, source_identity="fc:1")
        self.assertIs(state.configuration.vehicle_demand_result, result)
        self.assertEqual(state.configuration.source_identity, "fc:1")

    def test_architecture_adapter_maps_all_five_legacy_electrification_values(self):
        for legacy_value, expected in (
            ("ICE", ArchitectureClass.ICE),
            ("MHEV", ArchitectureClass.MHEV),
            ("HEV", ArchitectureClass.HEV),
            ("PHEV", ArchitectureClass.PHEV),
            ("BEV", ArchitectureClass.BEV),
        ):
            state = architecture_domain_state_from_legacy_vde_row({}, {"electrification": legacy_value})
            self.assertEqual(state.configuration.architecture_class, expected)

    def test_architecture_adapter_unrecognized_value_is_none_not_a_guess(self):
        state = architecture_domain_state_from_legacy_vde_row({}, {"electrification": "SOMETHING_NEW"})
        self.assertIsNone(state.configuration.architecture_class)
        self.assertIn("SOMETHING_NEW", state.configuration.topology_notes)


class Sprint11BExpandedAdapterTests(unittest.TestCase):
    """Sprint 11B: expanded Engine/Transmission fields plus the new Energy
    Storage / Controls / Aux-Thermal / (sparse) Electric Drive adapters.
    The base QA fixture seeds every one of these new columns as NULL (a
    direct PRAGMA table_info + SELECT check confirmed this during
    development), so this class augments the seeded fuelcons row with real,
    non-null values via a direct SQL UPDATE in setUp -- the same
    established pattern `test_comparison_report_page_smoke.py` already uses
    for scenario-specific fixture data -- to prove genuine data flow-
    through, not just correct null-handling (which the QA-database class
    above already covers).
    """

    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "system_scenario_expanded_adapter_qa.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        seed_qa_fuelcons_mock_rows(self.db_path)
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "UPDATE fuelcons_db SET engine_max_torque_nm = 350.0, "
                "battery_capacity_kwh = 1.5, battery_usable_kwh = 1.2, "
                "bms_discharge_limit_kw = 25.0, bms_regen_limit_kw = 15.0, "
                "bms_note = 'QA test note', utility_factor_pct = 65.0, "
                "ambient_temp_c = 23.0, ac_on = 1 "
                "WHERE id = 900102"
            )
            con.execute("UPDATE vde_db SET engine_model = 'QA-ENGINE-01', engine_type = 'I4', "
                        "engine_aspiration = 'Turbo', transmission_model = 'QA-TRANS-9AT' WHERE id = 900001")
            con.commit()
        db_module.configure_db_path(self.db_path)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def _rows(self):
        return fetch_vde_by_id(900001), dict(get_record(EntityType.FUEL_CONSUMPTION, 900102))

    def test_engine_adapter_populates_expanded_fields(self):
        vde_row, fc_row = self._rows()
        state = engine_domain_state_from_legacy_row(vde_row, fc_row)
        self.assertEqual(state.configuration.engine_family_id, "QA-ENGINE-01")
        self.assertEqual(state.configuration.rated_torque_nm, 350.0)
        self.assertIn("I4", state.configuration.technology_descriptors)
        self.assertIn("Turbo", state.configuration.technology_descriptors)

    def test_transmission_adapter_populates_model_id(self):
        vde_row, fc_row = self._rows()
        state = transmission_domain_state_from_legacy_row(vde_row, fc_row)
        self.assertEqual(state.configuration.transmission_model_id, "QA-TRANS-9AT")

    def test_energy_storage_adapter_sparse_when_no_row_supplied(self):
        state = energy_storage_domain_state_from_legacy_row(None)
        self.assertIsNone(state.configuration.gross_capacity_kwh)
        self.assertIsNone(state.configuration.usable_capacity_kwh)
        self.assertIsNone(state.configuration.discharge_power_limit_kw)
        self.assertIsNone(state.configuration.regen_power_limit_kw)

    def test_energy_storage_adapter_populates_real_values(self):
        _vde_row, fc_row = self._rows()
        state = energy_storage_domain_state_from_legacy_row(fc_row)
        self.assertEqual(state.configuration.gross_capacity_kwh, 1.5)
        self.assertEqual(state.configuration.usable_capacity_kwh, 1.2)
        self.assertEqual(state.configuration.discharge_power_limit_kw, 25.0)
        self.assertEqual(state.configuration.regen_power_limit_kw, 15.0)
        self.assertIsNone(state.configuration.charge_power_limit_kw)  # no confirmed column -- stays missing
        self.assertEqual(state.notes, "QA test note")
        self.assertEqual(state.provenance, ProvenanceKind.SOURCE_OBSERVED)

    def test_controls_adapter_populates_utility_factor(self):
        _vde_row, fc_row = self._rows()
        state = controls_domain_state_from_legacy_row(fc_row)
        self.assertEqual(state.configuration.utility_factor_pct, 65.0)
        self.assertEqual(state.provenance, ProvenanceKind.SOURCE_OBSERVED)
        self.assertIsNone(state.configuration.hybrid_operating_strategy)  # no confirmed column

    def test_controls_adapter_without_utility_factor_is_not_available(self):
        state = controls_domain_state_from_legacy_row({})
        self.assertIsNone(state.configuration.utility_factor_pct)
        self.assertEqual(state.provenance, ProvenanceKind.NOT_AVAILABLE)

    def test_aux_thermal_adapter_populates_real_values(self):
        _vde_row, fc_row = self._rows()
        state = aux_thermal_domain_state_from_legacy_row(fc_row)
        self.assertEqual(state.configuration.ambient_temp_c, 23.0)
        self.assertTrue(state.configuration.ac_on)
        self.assertEqual(state.provenance, ProvenanceKind.SOURCE_OBSERVED)

    def test_aux_thermal_adapter_sparse_when_no_row_supplied(self):
        state = aux_thermal_domain_state_from_legacy_row(None)
        self.assertIsNone(state.configuration.ambient_temp_c)
        self.assertIsNone(state.configuration.ac_on)

    def test_electric_drive_sparse_adapter_is_valid_and_explicit(self):
        state = electric_drive_domain_state_sparse()
        self.assertEqual(state.domain, DomainKind.ELECTRIC_DRIVE)
        self.assertIsNone(state.configuration.motor_role)
        self.assertIsNone(state.configuration.rated_power_kw)
        self.assertEqual(state.provenance, ProvenanceKind.NOT_AVAILABLE)


if __name__ == "__main__":
    unittest.main()
