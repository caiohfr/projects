"""Sprint 11C System Scenario composition and canonical L0 parity tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.vde_core.fuel_energy import compute_ice_fuel_from_vde
from src.vde_core.fuel_estimation import FuelEstimateRequest, run_fuel_estimation
from src.vde_core.system_scenario import (
    ArchitectureClass,
    ArchitectureConfiguration,
    ControlsConfiguration,
    DomainKind,
    DomainProposalIdentity,
    DomainSourceState,
    ElectricDriveConfiguration,
    EnergyStorageConfiguration,
    EngineConfiguration,
    FidelityLevel,
    SolverReadiness,
    SystemScenarioDefinition,
    SystemScenarioIdentity,
    SystemScenarioRole,
    TransmissionConfiguration,
    resolve_domain_proposal,
    resolve_effective_domain_state,
    resolve_system_scenario,
    resolve_system_scenarios,
    run_system_scenario,
    run_system_scenarios,
    to_serializable,
    vehicle_demand_domain_state_from_result,
)
from src.vde_core.technology_delta import (
    TechDeltaAssumption,
    apply_delta_stack_to_baseline,
    normalize_technology_delta,
    tech_delta_assumption_to_dict,
)
from src.vde_core.vehicle_demand import RoadloadBasis, VehicleDemandResult, VehicleDemandSummary


def _effective(domain, configuration):
    return resolve_effective_domain_state(DomainSourceState(domain=domain, configuration=configuration))


def _vde_state(total=1.8, net=1.6, identity="vde:1"):
    result = VehicleDemandResult(
        total_summary=VehicleDemandSummary(
            roadload_basis=RoadloadBasis.TOTAL,
            vde_mj_per_km=total,
            cycle_name="WLTP",
        ),
        net_summary=(
            VehicleDemandSummary(
                roadload_basis=RoadloadBasis.NET,
                vde_mj_per_km=net,
                cycle_name="WLTP",
            )
            if net is not None
            else None
        ),
    )
    return resolve_effective_domain_state(
        vehicle_demand_domain_state_from_result(result, source_identity=identity)
    )


def _architecture_state(architecture=ArchitectureClass.ICE):
    return _effective(
        DomainKind.ARCHITECTURE,
        ArchitectureConfiguration(architecture_class=architecture),
    )


def _engine_state(*, torque=None, fuel_type="Gasoline"):
    return _effective(
        DomainKind.ENGINE_FUEL_CONVERTER,
        EngineConfiguration(fuel_type=fuel_type, rated_torque_nm=torque),
    )


def _identity(name="SYS-CURRENT", *, proposal=False):
    return SystemScenarioIdentity(
        scenario_id=name,
        role=SystemScenarioRole.PROPOSAL if proposal else SystemScenarioRole.CURRENT,
        proposal_index=1 if proposal else None,
    )


def _definition(*, vde=None, architecture=None, engine=None, extras=None, name="SYS-CURRENT", proposal=False):
    slots = {
        DomainKind.VEHICLE_DEMAND: vde or _vde_state(),
        DomainKind.ARCHITECTURE: architecture or _architecture_state(),
        DomainKind.ENGINE_FUEL_CONVERTER: engine or _engine_state(),
    }
    slots.update(extras or {})
    return SystemScenarioDefinition(identity=_identity(name, proposal=proposal), slots=slots)


def _ice_template(eta=0.3, *, energy_basis="VDE_TOTAL"):
    return FuelEstimateRequest(
        vde_id=999,
        energy_basis=energy_basis,
        method="physics_simple",
        vehicle_features={
            "electrification": "BEV",  # deliberately stale: scenario must replace it
            "vde_total_mj_per_km": 999.0,
            "vde_net_mj_per_km": 999.0,
            "phase_outputs": {"vde_urb_mj_per_km": 999.0},
        },
        powertrain_features={
            "eta_pt_est": eta,
            "fuel_type": "Gasoline",
            "LHV_MJ_per_L": 32.0,
            "gCO2_per_L": 2310.0,
        },
    )


class L0CanonicalParityTests(unittest.TestCase):
    def test_neutral_current_matches_independent_canonical_call(self):
        outcome = run_system_scenario(_definition(), request_template=_ice_template())
        expected = run_fuel_estimation(
            FuelEstimateRequest(
                energy_basis="VDE_TOTAL",
                method="physics_simple",
                vehicle_features={
                    "electrification": "ICE",
                    "vde_total_mj_per_km": 1.8,
                    "vde_net_mj_per_km": 1.6,
                },
                powertrain_features={
                    "eta_pt_est": 0.3,
                    "fuel_type": "Gasoline",
                    "LHV_MJ_per_L": 32.0,
                    "gCO2_per_L": 2310.0,
                },
            )
        )
        self.assertEqual(outcome.resolved_scenario.solver_readiness, SolverReadiness.READY)
        self.assertEqual(outcome.fuel_estimate_result.fuel_l_100km, expected.fuel_l_100km)
        self.assertEqual(outcome.fuel_estimate_result.energy_Wh_km, expected.energy_Wh_km)
        self.assertEqual(outcome.fuel_estimate_result.gco2_km, expected.gco2_km)
        self.assertIsNone(outcome.technology_delta_result)

    def test_each_scenario_uses_its_own_vehicle_demand(self):
        first = run_system_scenario(
            _definition(vde=_vde_state(total=1.2, identity="vde:2"), name="SYS-A", proposal=True),
            request_template=_ice_template(),
        )
        second = run_system_scenario(
            _definition(vde=_vde_state(total=2.4, identity="vde:3"), name="SYS-B", proposal=True),
            request_template=_ice_template(),
        )
        self.assertEqual(first.fuel_estimate_result.request.vehicle_features["vde_total_mj_per_km"], 1.2)
        self.assertEqual(second.fuel_estimate_result.request.vehicle_features["vde_total_mj_per_km"], 2.4)
        self.assertEqual(first.fuel_estimate_result.request.vde_id, 2)
        self.assertEqual(second.fuel_estimate_result.request.vde_id, 3)
        self.assertNotIn("phase_outputs", first.fuel_estimate_result.request.vehicle_features)
        self.assertNotEqual(first.fuel_estimate_result.fuel_l_100km, second.fuel_estimate_result.fuel_l_100km)

    def test_request_template_and_source_contracts_are_not_mutated(self):
        definition = _definition()
        template = _ice_template()
        original_vehicle_features = dict(template.vehicle_features)
        original_powertrain_features = dict(template.powertrain_features)
        source_result = definition.vehicle_demand_selection.configuration.vehicle_demand_result
        run_system_scenario(definition, request_template=template)
        self.assertEqual(template.vehicle_features, original_vehicle_features)
        self.assertEqual(template.powertrain_features, original_powertrain_features)
        self.assertIs(definition.vehicle_demand_selection.configuration.vehicle_demand_result, source_result)

    def test_phev_parity_preserves_run_fuel_estimation_as_owner(self):
        template = FuelEstimateRequest(
            vehicle_features={"vde_total_mj_per_km": 99.0},
            powertrain_features={
                "eta_pt_est": 0.3,
                "bev_eff_drive": 0.9,
                "utility_factor": 0.4,
                "fuel_type": "Gasoline",
                "LHV_MJ_per_L": 32.0,
                "gCO2_per_L": 2310.0,
                "grid_gco2_per_kwh": 400.0,
            },
        )
        definition = _definition(architecture=_architecture_state(ArchitectureClass.PHEV))
        actual = run_system_scenario(definition, request_template=template).fuel_estimate_result
        expected = run_fuel_estimation(actual.request)
        self.assertEqual(actual.fuel_l_100km, expected.fuel_l_100km)
        self.assertEqual(actual.energy_Wh_km, expected.energy_Wh_km)
        self.assertEqual(actual.gco2_km, expected.gco2_km)

    def test_phev_co2_preflight_reproduces_legacy_helper_disagreement(self):
        canonical_request = FuelEstimateRequest(
            energy_basis="VDE_TOTAL",
            method="physics_simple",
            vehicle_features={"electrification": "PHEV", "vde_total_mj_per_km": 1.8},
            powertrain_features={
                "fuel_type": "Gasoline",
                "eta_pt_est": 0.3,
                "LHV_MJ_per_L": 32.0,
                "bev_eff_drive": 0.9,
                "utility_factor": 0.4,
                "grid_gco2_per_kwh": 400.0,
            },
        )
        canonical = run_fuel_estimation(canonical_request)
        with patch(
            "src.vde_core.fuel_energy._get_vde_row",
            return_value={"vde_net_mj_per_km": 1.8, "legislation": "WLTP"},
        ):
            legacy = compute_ice_fuel_from_vde(
                1,
                "Gasoline",
                0.3,
                lhv_mj_per_l=32.0,
                electrification="PHEV",
                uf_phev=0.4,
                driveline_eff=0.9,
                grid_gco2_per_kwh=400.0,
            )

        self.assertAlmostEqual(canonical.gco2_km, 259.875, places=6)
        self.assertAlmostEqual(legacy["gco2_per_km"], 348.7638888889, places=6)
        self.assertAlmostEqual(legacy["gco2_per_km"] - canonical.gco2_km, 88.8888888889, places=6)

    def test_neutral_bev_matches_independent_canonical_call(self):
        template = FuelEstimateRequest(
            powertrain_features={"bev_eff_drive": 0.9, "grid_gco2_per_kwh": 400.0}
        )
        outcome = run_system_scenario(
            _definition(architecture=_architecture_state(ArchitectureClass.BEV)),
            request_template=template,
        )
        expected = run_fuel_estimation(
            FuelEstimateRequest(
                vehicle_features={
                    "electrification": "BEV",
                    "vde_total_mj_per_km": 1.8,
                    "vde_net_mj_per_km": 1.6,
                },
                powertrain_features={"bev_eff_drive": 0.9, "grid_gco2_per_kwh": 400.0},
            )
        )
        self.assertEqual(outcome.fuel_estimate_result.energy_Wh_km, expected.energy_Wh_km)
        self.assertEqual(outcome.fuel_estimate_result.gco2_km, expected.gco2_km)
        self.assertIsNone(outcome.fuel_estimate_result.fuel_l_100km)

    def test_manual_imported_method_and_provenance_pass_through_canonical_owner(self):
        template = FuelEstimateRequest(
            method="manual_imported",
            manual_inputs={
                "source": "engineering_sheet",
                "fuel_l_100km": 7.2,
                "gco2_km": 155.0,
            },
        )
        actual = run_system_scenario(_definition(), request_template=template).fuel_estimate_result
        expected = run_fuel_estimation(actual.request)
        self.assertEqual(actual.method, "manual_imported")
        self.assertEqual(actual.fuel_l_100km, expected.fuel_l_100km)
        self.assertEqual(actual.gco2_km, expected.gco2_km)
        self.assertEqual(actual.assumptions["source"], "engineering_sheet")

    def test_engine_fuel_change_drops_stale_template_factors_and_uses_canonical_defaults(self):
        definition = _definition(engine=_engine_state(fuel_type="Diesel"), proposal=True)
        actual = run_system_scenario(definition, request_template=_ice_template()).fuel_estimate_result
        self.assertEqual(actual.request.powertrain_features["fuel_type"], "Diesel")
        self.assertNotIn("LHV_MJ_per_L", actual.request.powertrain_features)
        self.assertNotIn("gCO2_per_L", actual.request.powertrain_features)
        expected = run_fuel_estimation(actual.request)
        self.assertEqual(actual.fuel_l_100km, expected.fuel_l_100km)
        self.assertEqual(actual.gco2_km, expected.gco2_km)


class FidelityAndReadinessTests(unittest.TestCase):
    def test_missing_future_only_engine_torque_does_not_block_l0(self):
        resolved = resolve_system_scenario(
            _definition(engine=_engine_state(torque=None)), request_template=_ice_template()
        )
        self.assertEqual(resolved.solver_readiness, SolverReadiness.READY)
        self.assertNotIn("rated_torque_nm_missing", resolved.issues)

    def test_missing_required_effective_efficiency_blocks_physics_simple(self):
        resolved = resolve_system_scenario(_definition(), request_template=FuelEstimateRequest())
        self.assertEqual(resolved.solver_readiness, SolverReadiness.NOT_READY)
        self.assertIn("eta_pt_est_missing", resolved.issues)
        result = run_system_scenario(_definition(), request_template=FuelEstimateRequest())
        self.assertIsNone(result.fuel_estimate_result)

    def test_missing_vehicle_demand_never_falls_back_to_template_values(self):
        definition = SystemScenarioDefinition(
            identity=_identity(),
            slots={
                DomainKind.ARCHITECTURE: _architecture_state(),
                DomainKind.ENGINE_FUEL_CONVERTER: _engine_state(),
            },
        )
        resolved = resolve_system_scenario(definition, request_template=_ice_template())
        self.assertEqual(resolved.solver_readiness, SolverReadiness.NOT_READY)
        self.assertIn("vde_total_mj_per_km_missing", resolved.issues)
        self.assertIsNone(resolved.fuel_estimate_request.vehicle_features["vde_total_mj_per_km"])

    def test_net_basis_never_falls_back_to_total_or_stale_template_net(self):
        resolved = resolve_system_scenario(
            _definition(vde=_vde_state(net=None)),
            request_template=_ice_template(energy_basis="VDE_NET"),
        )
        self.assertEqual(resolved.solver_readiness, SolverReadiness.NOT_READY)
        self.assertIn("vde_net_mj_per_km_missing", resolved.issues)
        self.assertIsNone(resolved.fuel_estimate_request.vehicle_features["vde_net_mj_per_km"])

    def test_bev_engine_is_not_represented_and_electric_efficiency_controls_readiness(self):
        definition = _definition(architecture=_architecture_state(ArchitectureClass.BEV))
        resolved = resolve_system_scenario(
            definition,
            request_template=FuelEstimateRequest(powertrain_features={"bev_eff_drive": 0.9}),
        )
        self.assertEqual(resolved.solver_readiness, SolverReadiness.READY)
        self.assertEqual(
            resolved.fidelity_manifest.fidelity_for(DomainKind.ENGINE_FUEL_CONVERTER),
            FidelityLevel.NOT_REPRESENTED,
        )

    def test_bev_explicit_engine_proposal_is_structured_architecture_incompatibility(self):
        current = _engine_state()
        engine_proposal = resolve_domain_proposal(
            DomainProposalIdentity(DomainKind.ENGINE_FUEL_CONVERTER, "ENG-BEV-P01"),
            current,
            requested_changes={"fuel_type": "Diesel"},
        )
        resolved = resolve_system_scenario(
            _definition(
                architecture=_architecture_state(ArchitectureClass.BEV),
                engine=engine_proposal,
                proposal=True,
            ),
            request_template=FuelEstimateRequest(powertrain_features={"bev_eff_drive": 0.9}),
        )
        self.assertEqual(resolved.solver_readiness, SolverReadiness.NOT_READY)
        self.assertIn(
            "architecture_domain_incompatible:BEV:ENGINE_FUEL_CONVERTER:ENG-BEV-P01",
            resolved.issues,
        )

    def test_transmission_configuration_only_does_not_change_baseline(self):
        transmission = _effective(
            DomainKind.TRANSMISSION_DRIVELINE,
            TransmissionConfiguration(transmission_type="8AT", gear_count=8, final_drive_ratio=3.73),
        )
        proposal = resolve_domain_proposal(
            DomainProposalIdentity(DomainKind.TRANSMISSION_DRIVELINE, "TRANS-P01"),
            transmission,
            requested_changes={"transmission_type": "9AT", "gear_count": 9, "final_drive_ratio": 3.45},
        )
        actual = run_system_scenario(
            _definition(extras={DomainKind.TRANSMISSION_DRIVELINE: proposal}, proposal=True),
            request_template=_ice_template(),
        )
        neutral = run_system_scenario(_definition(), request_template=_ice_template())
        self.assertEqual(actual.fuel_estimate_result.fuel_l_100km, neutral.fuel_estimate_result.fuel_l_100km)
        self.assertEqual(
            actual.resolved_scenario.fidelity_manifest.fidelity_for(DomainKind.TRANSMISSION_DRIVELINE),
            FidelityLevel.CONFIGURATION_ONLY,
        )

    def test_battery_capacity_change_is_configuration_only(self):
        current = _effective(
            DomainKind.ENERGY_STORAGE,
            EnergyStorageConfiguration(usable_capacity_kwh=1.0),
        )
        proposal = resolve_domain_proposal(
            DomainProposalIdentity(DomainKind.ENERGY_STORAGE, "BAT-P01"),
            current,
            requested_changes={"usable_capacity_kwh": 1.5},
        )
        actual = run_system_scenario(
            _definition(
                architecture=_architecture_state(ArchitectureClass.HEV),
                extras={DomainKind.ENERGY_STORAGE: proposal},
                proposal=True,
            ),
            request_template=_ice_template(),
        )
        neutral = run_system_scenario(
            _definition(architecture=_architecture_state(ArchitectureClass.HEV)),
            request_template=_ice_template(),
        )
        self.assertEqual(actual.fuel_estimate_result.fuel_l_100km, neutral.fuel_estimate_result.fuel_l_100km)
        self.assertEqual(
            actual.resolved_scenario.fidelity_manifest.fidelity_for(DomainKind.ENERGY_STORAGE),
            FidelityLevel.CONFIGURATION_ONLY,
        )

    def test_engine_size_and_power_changes_are_configuration_only_for_current_l0(self):
        current = _effective(
            DomainKind.ENGINE_FUEL_CONVERTER,
            EngineConfiguration(fuel_type="Gasoline", displacement_l=1.5, rated_power_kw=100.0),
        )
        proposal = resolve_domain_proposal(
            DomainProposalIdentity(DomainKind.ENGINE_FUEL_CONVERTER, "ENG-CONFIG-P01"),
            current,
            requested_changes={"displacement_l": 2.0, "rated_power_kw": 150.0},
        )
        actual = run_system_scenario(
            _definition(engine=proposal, proposal=True), request_template=_ice_template()
        )
        neutral = run_system_scenario(_definition(), request_template=_ice_template())
        self.assertEqual(actual.fuel_estimate_result.fuel_l_100km, neutral.fuel_estimate_result.fuel_l_100km)
        self.assertEqual(
            actual.fidelity_manifest.fidelity_for(DomainKind.ENGINE_FUEL_CONVERTER),
            FidelityLevel.QUANTITATIVE,
        )

    def test_electric_motor_power_change_is_configuration_only_for_current_l0(self):
        current = _effective(
            DomainKind.ELECTRIC_DRIVE,
            ElectricDriveConfiguration(rated_power_kw=100.0, peak_power_kw=120.0),
        )
        proposal = resolve_domain_proposal(
            DomainProposalIdentity(DomainKind.ELECTRIC_DRIVE, "EM-CONFIG-P01"),
            current,
            requested_changes={"rated_power_kw": 150.0, "peak_power_kw": 180.0},
        )
        template = FuelEstimateRequest(
            method="physics_simple",
            powertrain_features={"bev_eff_drive": 0.9},
        )
        actual = run_system_scenario(
            _definition(
                architecture=_architecture_state(ArchitectureClass.BEV),
                extras={DomainKind.ELECTRIC_DRIVE: proposal},
                proposal=True,
            ),
            request_template=template,
        )
        neutral = run_system_scenario(
            _definition(architecture=_architecture_state(ArchitectureClass.BEV)),
            request_template=template,
        )
        self.assertEqual(actual.fuel_estimate_result.energy_Wh_km, neutral.fuel_estimate_result.energy_Wh_km)
        self.assertEqual(
            actual.fidelity_manifest.fidelity_for(DomainKind.ELECTRIC_DRIVE),
            FidelityLevel.CONFIGURATION_ONLY,
        )

    def test_explicit_zero_is_present_and_effectively_represented(self):
        transmission = _effective(DomainKind.TRANSMISSION_DRIVELINE, TransmissionConfiguration())
        proposal = resolve_domain_proposal(
            DomainProposalIdentity(DomainKind.TRANSMISSION_DRIVELINE, "TRANS-ZERO"),
            transmission,
            l0_effective_assumption={"pse_percent_delta": 0.0},
        )
        resolved = resolve_system_scenario(
            _definition(extras={DomainKind.TRANSMISSION_DRIVELINE: proposal}, proposal=True),
            request_template=_ice_template(),
        )
        self.assertEqual(resolved.ordered_technology_deltas[0].effect_value, 0.0)
        self.assertEqual(
            resolved.fidelity_manifest.fidelity_for(DomainKind.TRANSMISSION_DRIVELINE),
            FidelityLevel.EFFECTIVE_ASSUMPTION,
        )


class DeterministicTechnologyDeltaTests(unittest.TestCase):
    def _proposal(self, domain, configuration, proposal_id, deltas=(), assumptions=None):
        current = _effective(domain, configuration)
        return resolve_domain_proposal(
            DomainProposalIdentity(domain, proposal_id),
            current,
            technology_deltas=deltas,
            l0_effective_assumption=assumptions or {},
        )

    def test_domain_order_not_slot_or_presentation_order_controls_stack(self):
        engine_delta = TechDeltaAssumption("Engine absolute", "pse_delta", 0.01)
        transmission_deltas = (
            TechDeltaAssumption("Transmission percent", "pse_percent_delta", 5.0),
            TechDeltaAssumption("Transmission fuel", "fuel_percent_delta", -1.0),
        )
        engine = self._proposal(
            DomainKind.ENGINE_FUEL_CONVERTER,
            EngineConfiguration(fuel_type="Gasoline"),
            "ENG-P01",
            (engine_delta,),
        )
        transmission = self._proposal(
            DomainKind.TRANSMISSION_DRIVELINE,
            TransmissionConfiguration(),
            "TRANS-P01",
            transmission_deltas,
        )
        controls = self._proposal(
            DomainKind.ENERGY_MANAGEMENT_CONTROLS,
            ControlsConfiguration(),
            "CTRL-P01",
            (TechDeltaAssumption("Controls CO2", "co2_percent_delta", -0.5),),
        )
        reverse_slots = {
            DomainKind.ENERGY_MANAGEMENT_CONTROLS: controls,
            DomainKind.TRANSMISSION_DRIVELINE: transmission,
            DomainKind.ENGINE_FUEL_CONVERTER: engine,
            DomainKind.ARCHITECTURE: _architecture_state(),
            DomainKind.VEHICLE_DEMAND: _vde_state(),
        }
        resolved = resolve_system_scenario(
            SystemScenarioDefinition(identity=_identity("SYS-A", proposal=True), slots=reverse_slots),
            request_template=_ice_template(),
        )
        self.assertEqual(
            [delta.name for delta in resolved.ordered_technology_deltas],
            ["Engine absolute", "Transmission percent", "Transmission fuel", "Controls CO2"],
        )
        self.assertEqual(
            [
                (item.evaluation_order, item.domain, item.proposal_id, item.assumption.name)
                for item in resolved.technology_delta_contributions
            ],
            [
                (1, DomainKind.ENGINE_FUEL_CONVERTER, "ENG-P01", "Engine absolute"),
                (2, DomainKind.TRANSMISSION_DRIVELINE, "TRANS-P01", "Transmission percent"),
                (3, DomainKind.TRANSMISSION_DRIVELINE, "TRANS-P01", "Transmission fuel"),
                (4, DomainKind.ENERGY_MANAGEMENT_CONTROLS, "CTRL-P01", "Controls CO2"),
            ],
        )

    def test_explicit_l0_effect_uses_canonical_delta_stack(self):
        transmission = self._proposal(
            DomainKind.TRANSMISSION_DRIVELINE,
            TransmissionConfiguration(),
            "TRANS-P02",
            assumptions={"pse_percent_delta": 0.8},
        )
        outcome = run_system_scenario(
            _definition(extras={DomainKind.TRANSMISSION_DRIVELINE: transmission}, proposal=True),
            request_template=_ice_template(),
        )
        baseline = run_fuel_estimation(outcome.fuel_estimate_result.request)
        normalized = [
            normalize_technology_delta(tech_delta_assumption_to_dict(delta), index=index + 1)
            for index, delta in enumerate(outcome.resolved_scenario.ordered_technology_deltas)
        ]
        expected = apply_delta_stack_to_baseline(
            baseline,
            ctx={"energy_value_mj_per_km": 1.8},
            deltas=normalized,
        )
        self.assertEqual(outcome.technology_delta_result["proposal"], expected["proposal"])
        for key, value in expected["proposal"].items():
            self.assertEqual(outcome.effective_outputs[key], value)

    def test_only_active_supported_deltas_enter_stack(self):
        proposal = self._proposal(
            DomainKind.TRANSMISSION_DRIVELINE,
            TransmissionConfiguration(),
            "TRANS-PENDING",
            (
                TechDeltaAssumption("Disabled", "fuel_delta", -1.0, enabled=False),
                TechDeltaAssumption("Needs map", "map-based effect", 1.0),
            ),
        )
        resolved = resolve_system_scenario(
            _definition(extras={DomainKind.TRANSMISSION_DRIVELINE: proposal}, proposal=True),
            request_template=_ice_template(),
        )
        self.assertEqual(resolved.ordered_technology_deltas, ())
        self.assertEqual(resolved.technology_delta_contributions, ())
        self.assertEqual(resolved.solver_readiness, SolverReadiness.NOT_READY)
        self.assertIn(
            "unsupported_quantitative_representation:TRANSMISSION_DRIVELINE:TRANS-PENDING:map_based_effect",
            resolved.issues,
        )

    def test_unknown_delta_basis_is_unresolved_without_invented_math(self):
        proposal = self._proposal(
            DomainKind.TRANSMISSION_DRIVELINE,
            TransmissionConfiguration(),
            "TRANS-UNKNOWN",
            (TechDeltaAssumption("Unknown", "unapproved_basis", 1.0),),
        )
        resolved = resolve_system_scenario(
            _definition(extras={DomainKind.TRANSMISSION_DRIVELINE: proposal}, proposal=True),
            request_template=_ice_template(),
        )
        self.assertEqual(resolved.ordered_technology_deltas, ())
        self.assertEqual(resolved.solver_readiness, SolverReadiness.NOT_READY)
        self.assertIn(
            "incompatible_technology_delta_basis:TRANSMISSION_DRIVELINE:TRANS-UNKNOWN:unapproved_basis",
            resolved.issues,
        )

    def test_direct_higher_efficiency_reduces_canonical_fuel_input(self):
        low = self._proposal(
            DomainKind.ENGINE_FUEL_CONVERTER,
            EngineConfiguration(fuel_type="Gasoline"),
            "ENG-LOW",
            assumptions={"eta_pt_est": 0.25},
        )
        high = self._proposal(
            DomainKind.ENGINE_FUEL_CONVERTER,
            EngineConfiguration(fuel_type="Gasoline"),
            "ENG-HIGH",
            assumptions={"eta_pt_est": 0.40},
        )
        template = _ice_template(eta=0.3)
        low_result = run_system_scenario(_definition(engine=low, proposal=True), request_template=template)
        high_result = run_system_scenario(_definition(engine=high, proposal=True), request_template=template)
        self.assertGreater(low_result.fuel_estimate_result.fuel_l_100km, high_result.fuel_estimate_result.fuel_l_100km)

    def test_conflicting_direct_assumptions_fail_loudly(self):
        engine = self._proposal(
            DomainKind.ENGINE_FUEL_CONVERTER,
            EngineConfiguration(fuel_type="Gasoline"),
            "ENG-P01",
            assumptions={"eta_pt_est": 0.3},
        )
        transmission = self._proposal(
            DomainKind.TRANSMISSION_DRIVELINE,
            TransmissionConfiguration(),
            "TRANS-P01",
            assumptions={"eta_pt_est": 0.4},
        )
        resolved = resolve_system_scenario(
            _definition(engine=engine, extras={DomainKind.TRANSMISSION_DRIVELINE: transmission}, proposal=True),
            request_template=_ice_template(),
        )
        self.assertEqual(resolved.solver_readiness, SolverReadiness.NOT_READY)
        self.assertIn("conflicting_l0_assumption:eta_pt_est", resolved.issues)


class WorkingSetTests(unittest.TestCase):
    def test_current_plus_three_proposals_is_the_maximum(self):
        definitions = [_definition()]
        definitions.extend(
            _definition(name=f"SYS-P{index}", proposal=True)
            for index in range(1, 4)
        )
        # Give each Proposal its own stable index while keeping the helper compact.
        definitions[2] = SystemScenarioDefinition(
            identity=SystemScenarioIdentity("SYS-P2", SystemScenarioRole.PROPOSAL, 2),
            slots=definitions[2].slots,
        )
        definitions[3] = SystemScenarioDefinition(
            identity=SystemScenarioIdentity("SYS-P3", SystemScenarioRole.PROPOSAL, 3),
            slots=definitions[3].slots,
        )
        templates = {definition.identity.scenario_id: _ice_template() for definition in definitions}
        self.assertEqual(len(resolve_system_scenarios(definitions, request_templates=templates)), 4)

        extra = SystemScenarioDefinition(
            identity=SystemScenarioIdentity("SYS-P4", SystemScenarioRole.PROPOSAL, 3),
            slots=_definition().slots,
        )
        with self.assertRaises(ValueError):
            resolve_system_scenarios([*definitions, extra], request_templates=templates)

    def test_duplicate_proposal_index_is_rejected_within_working_set(self):
        first = _definition(name="SYS-A", proposal=True)
        second = _definition(name="SYS-B", proposal=True)
        with self.assertRaisesRegex(ValueError, "roles/proposal indexes"):
            resolve_system_scenarios([first, second])

    def test_duplicate_scenario_id_is_rejected_within_working_set(self):
        current = _definition(name="SYS-DUP")
        proposal = _definition(name="SYS-DUP", proposal=True)
        with self.assertRaisesRegex(ValueError, "scenario_id values"):
            resolve_system_scenarios([current, proposal])

    def test_resolution_snapshot_and_request_view_are_immutable_and_isolated(self):
        resolved = resolve_system_scenario(_definition(), request_template=_ice_template())
        with self.assertRaises(TypeError):
            resolved.resolved_domains[DomainKind.ARCHITECTURE] = _architecture_state(ArchitectureClass.BEV)
        first_request = resolved.fuel_estimate_request
        first_request.vehicle_features["vde_total_mj_per_km"] = 99.0
        self.assertEqual(resolved.fuel_estimate_request.vehicle_features["vde_total_mj_per_km"], 1.8)

    def test_current_a_b_use_independent_vdes_and_reuse_shared_proposal(self):
        shared_engine = resolve_domain_proposal(
            DomainProposalIdentity(DomainKind.ENGINE_FUEL_CONVERTER, "ENG-SHARED"),
            _engine_state(),
            l0_effective_assumption={"eta_pt_est": 0.35},
        )
        current = _definition(vde=_vde_state(total=1.0, identity="vde:1"))
        proposal_a = SystemScenarioDefinition(
            identity=SystemScenarioIdentity("SYS-A", SystemScenarioRole.PROPOSAL, 1),
            slots=_definition(vde=_vde_state(total=2.0, identity="vde:2"), engine=shared_engine).slots,
        )
        proposal_b = SystemScenarioDefinition(
            identity=SystemScenarioIdentity("SYS-B", SystemScenarioRole.PROPOSAL, 2),
            slots=_definition(vde=_vde_state(total=3.0, identity="vde:3"), engine=shared_engine).slots,
        )
        outcomes = run_system_scenarios(
            (current, proposal_a, proposal_b),
            request_templates={item.identity.scenario_id: _ice_template() for item in (current, proposal_a, proposal_b)},
        )
        self.assertEqual(
            [item.selected_vehicle_demand_identity for item in outcomes],
            ["vde:1", "vde:2", "vde:3"],
        )
        self.assertIs(proposal_a.slots[DomainKind.ENGINE_FUEL_CONVERTER], shared_engine)
        self.assertIs(proposal_b.slots[DomainKind.ENGINE_FUEL_CONVERTER], shared_engine)
        self.assertEqual(shared_engine.l0_effective_assumption["eta_pt_est"], 0.35)
        with self.assertRaises(TypeError):
            shared_engine.l0_effective_assumption["eta_pt_est"] = 0.9
        self.assertEqual(outcomes[0].fuel_estimate_result.request.powertrain_features["eta_pt_est"], 0.3)
        self.assertEqual(outcomes[1].fuel_estimate_result.request.powertrain_features["eta_pt_est"], 0.35)
        self.assertEqual(outcomes[2].fuel_estimate_result.request.powertrain_features["eta_pt_est"], 0.35)

    def test_shared_proposal_remains_bound_to_its_original_effective_current(self):
        current_a = _effective(
            DomainKind.ENGINE_FUEL_CONVERTER,
            EngineConfiguration(fuel_type="Gasoline", rated_torque_nm=140.0),
        )
        current_b = _effective(
            DomainKind.ENGINE_FUEL_CONVERTER,
            EngineConfiguration(fuel_type="Diesel", rated_torque_nm=260.0),
        )
        shared_engine = resolve_domain_proposal(
            DomainProposalIdentity(DomainKind.ENGINE_FUEL_CONVERTER, "ENG-SHARED"),
            current_a,
            requested_changes={"rated_torque_nm": 160.0},
        )
        proposal_a = SystemScenarioDefinition(
            identity=SystemScenarioIdentity("SYS-A", SystemScenarioRole.PROPOSAL, 1),
            slots=_definition(engine=shared_engine).slots,
        )
        proposal_b = SystemScenarioDefinition(
            identity=SystemScenarioIdentity("SYS-B", SystemScenarioRole.PROPOSAL, 2),
            slots=_definition(engine=shared_engine).slots,
        )

        resolved = resolve_system_scenarios(
            (proposal_a, proposal_b),
            request_templates={
                proposal_a.identity.scenario_id: _ice_template(),
                proposal_b.identity.scenario_id: _ice_template(),
            },
        )

        self.assertNotEqual(current_a.configuration, current_b.configuration)
        self.assertIs(shared_engine.based_on, current_a)
        self.assertIs(proposal_a.slots[DomainKind.ENGINE_FUEL_CONVERTER], shared_engine)
        self.assertIs(proposal_b.slots[DomainKind.ENGINE_FUEL_CONVERTER], shared_engine)
        self.assertIs(
            resolved[1].resolved_domains[DomainKind.ENGINE_FUEL_CONVERTER].based_on,
            current_a,
        )
        self.assertIsNot(
            resolved[1].resolved_domains[DomainKind.ENGINE_FUEL_CONVERTER].based_on,
            current_b,
        )
        self.assertEqual(
            resolved[1]
            .resolved_domains[DomainKind.ENGINE_FUEL_CONVERTER]
            .configuration
            .fuel_type,
            "Gasoline",
        )

    def test_result_is_deterministic_and_carries_solver_fidelity_and_provenance(self):
        transmission = resolve_domain_proposal(
            DomainProposalIdentity(DomainKind.TRANSMISSION_DRIVELINE, "TRANS-P01"),
            _effective(DomainKind.TRANSMISSION_DRIVELINE, TransmissionConfiguration()),
            l0_effective_assumption={"pse_percent_delta": 0.0},
        )
        definition = _definition(
            extras={DomainKind.TRANSMISSION_DRIVELINE: transmission}, proposal=True
        )
        first = run_system_scenario(definition, request_template=_ice_template())
        second = run_system_scenario(definition, request_template=_ice_template())
        self.assertEqual(to_serializable(first), to_serializable(second))
        self.assertEqual(first.selected_vehicle_demand_identity, "vde:1")
        self.assertEqual(first.architecture_class, ArchitectureClass.ICE)
        self.assertEqual(first.readiness, SolverReadiness.READY)
        self.assertEqual(first.fidelity_manifest, first.resolved_scenario.fidelity_manifest)
        self.assertIn("fuel_estimation.run_fuel_estimation", first.solver_identity)
        self.assertEqual(first.model_identity, "physics_simple")
        self.assertEqual(first.provenance["calculated_result"], "CALCULATED")
        self.assertEqual(first.effective_outputs["pse"], first.technology_delta_result["proposal"]["pse"])


if __name__ == "__main__":
    unittest.main()
