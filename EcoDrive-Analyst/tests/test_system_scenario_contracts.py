"""Sprint 11A: tests for the Streamlit-free System Scenario canonical
contracts (src.vde_core.system_scenario.contracts) -- data shape and
structural invariants only, no physics, no DB, no Streamlit.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from src.vde_core.system_scenario import (
    ALL_DOMAIN_KINDS,
    MAX_SYSTEM_SCENARIO_PROPOSALS,
    ArchitectureClass,
    ArchitectureConfiguration,
    DomainCorrection,
    DomainKind,
    DomainProposal,
    DomainProposalIdentity,
    DomainSourceState,
    EffectiveDomainState,
    EngineConfiguration,
    FidelityLevel,
    FidelityManifest,
    ProvenanceKind,
    SystemScenarioDefinition,
    SystemScenarioIdentity,
    SystemScenarioRole,
    TransmissionConfiguration,
    VehicleDemandConfiguration,
    domain_typically_applicable,
    resolve_effective_domain_state,
    resolve_system_scenario_shell,
    to_serializable,
)
from src.vde_core.system_scenario.serialization import (
    domain_proposal_from_dict,
    domain_source_state_from_dict,
    effective_domain_state_from_dict,
    fidelity_manifest_from_dict,
    system_scenario_identity_from_dict,
)


def _trans_source(final_drive_ratio=3.73, gear_count=8) -> DomainSourceState:
    return DomainSourceState(
        domain=DomainKind.TRANSMISSION_DRIVELINE,
        configuration=TransmissionConfiguration(final_drive_ratio=final_drive_ratio, gear_count=gear_count),
    )


def _trans_effective(**kwargs) -> EffectiveDomainState:
    return resolve_effective_domain_state(_trans_source(**kwargs))


class ExplicitZeroNotMissingTests(unittest.TestCase):
    def test_zero_final_drive_ratio_is_preserved_not_treated_as_missing(self):
        state = DomainSourceState(
            domain=DomainKind.TRANSMISSION_DRIVELINE,
            configuration=TransmissionConfiguration(final_drive_ratio=0.0, gear_count=1),
        )
        self.assertEqual(state.configuration.final_drive_ratio, 0.0)
        self.assertIsNotNone(state.configuration.final_drive_ratio)

    def test_zero_utility_factor_pct_is_preserved_not_missing(self):
        from src.vde_core.system_scenario import ControlsConfiguration

        config = ControlsConfiguration(utility_factor_pct=0.0)
        self.assertEqual(config.utility_factor_pct, 0.0)
        self.assertIsNotNone(config.utility_factor_pct)

    def test_none_field_is_genuinely_missing_distinct_from_zero(self):
        config = TransmissionConfiguration(final_drive_ratio=None)
        self.assertIsNone(config.final_drive_ratio)
        zero_config = TransmissionConfiguration(final_drive_ratio=0.0)
        self.assertNotEqual(config.final_drive_ratio, zero_config.final_drive_ratio)

    def test_zero_l0_effective_assumption_value_is_explicit(self):
        effective = _trans_effective()
        proposal = DomainProposal(
            identity=DomainProposalIdentity(domain=DomainKind.TRANSMISSION_DRIVELINE, proposal_id="TRANS-NEUTRAL"),
            domain=DomainKind.TRANSMISSION_DRIVELINE,
            configuration=effective.configuration,
            based_on=effective,
            l0_effective_assumption={"pse_percent_delta": 0.0},
        )
        self.assertIn("pse_percent_delta", proposal.l0_effective_assumption)
        self.assertEqual(proposal.l0_effective_assumption["pse_percent_delta"], 0.0)


class StableIdentityTests(unittest.TestCase):
    def test_system_scenario_identity_is_independent_of_label(self):
        identity = SystemScenarioIdentity(scenario_id="SYS-CURRENT", role=SystemScenarioRole.CURRENT)
        scenario_a = SystemScenarioDefinition(identity=identity, slots={}, label="Baseline")
        scenario_b = SystemScenarioDefinition(identity=identity, slots={}, label="Renamed Baseline")
        self.assertEqual(scenario_a.identity, scenario_b.identity)

    def test_domain_proposal_identity_is_independent_of_label(self):
        effective = _trans_effective()
        identity = DomainProposalIdentity(domain=DomainKind.TRANSMISSION_DRIVELINE, proposal_id="TRANS-P01")
        proposal_a = DomainProposal(
            identity=identity, domain=DomainKind.TRANSMISSION_DRIVELINE,
            configuration=effective.configuration, based_on=effective, label="9AT Swap",
        )
        proposal_b = DomainProposal(
            identity=identity, domain=DomainKind.TRANSMISSION_DRIVELINE,
            configuration=effective.configuration, based_on=effective, label="Renamed",
        )
        self.assertEqual(proposal_a.identity, proposal_b.identity)

    def test_identity_stable_across_repeated_construction(self):
        first = SystemScenarioIdentity(scenario_id="SYS-P01", role=SystemScenarioRole.PROPOSAL, proposal_index=1)
        second = SystemScenarioIdentity(scenario_id="SYS-P01", role=SystemScenarioRole.PROPOSAL, proposal_index=1)
        self.assertEqual(first, second)


class CurrentPlusMaxThreeProposalTests(unittest.TestCase):
    def test_current_requires_no_proposal_index(self):
        identity = SystemScenarioIdentity(scenario_id="SYS-CURRENT", role=SystemScenarioRole.CURRENT)
        self.assertIsNone(identity.proposal_index)

    def test_current_rejects_a_proposal_index(self):
        with self.assertRaises(ValueError):
            SystemScenarioIdentity(scenario_id="SYS-CURRENT", role=SystemScenarioRole.CURRENT, proposal_index=1)

    def test_proposal_indices_1_to_3_are_valid(self):
        for index in range(1, MAX_SYSTEM_SCENARIO_PROPOSALS + 1):
            identity = SystemScenarioIdentity(
                scenario_id=f"SYS-P{index}", role=SystemScenarioRole.PROPOSAL, proposal_index=index
            )
            self.assertEqual(identity.proposal_index, index)

    def test_proposal_index_zero_is_rejected(self):
        with self.assertRaises(ValueError):
            SystemScenarioIdentity(scenario_id="SYS-P0", role=SystemScenarioRole.PROPOSAL, proposal_index=0)

    def test_proposal_index_above_max_is_rejected(self):
        with self.assertRaises(ValueError):
            SystemScenarioIdentity(
                scenario_id="SYS-P4", role=SystemScenarioRole.PROPOSAL, proposal_index=MAX_SYSTEM_SCENARIO_PROPOSALS + 1
            )

    def test_proposal_requires_an_index(self):
        with self.assertRaises(ValueError):
            SystemScenarioIdentity(scenario_id="SYS-P?", role=SystemScenarioRole.PROPOSAL, proposal_index=None)


class IndependentProposalIdentityTests(unittest.TestCase):
    def test_two_proposals_for_the_same_domain_have_distinct_identities(self):
        effective = _trans_effective()
        proposal_a = DomainProposal(
            identity=DomainProposalIdentity(domain=DomainKind.TRANSMISSION_DRIVELINE, proposal_id="TRANS-P01"),
            domain=DomainKind.TRANSMISSION_DRIVELINE, configuration=effective.configuration, based_on=effective,
        )
        proposal_b = DomainProposal(
            identity=DomainProposalIdentity(domain=DomainKind.TRANSMISSION_DRIVELINE, proposal_id="TRANS-P02"),
            domain=DomainKind.TRANSMISSION_DRIVELINE, configuration=effective.configuration, based_on=effective,
        )
        self.assertNotEqual(proposal_a.identity, proposal_b.identity)

    def test_proposal_identity_domain_must_match_proposal_domain(self):
        effective = _trans_effective()
        with self.assertRaises(ValueError):
            DomainProposal(
                identity=DomainProposalIdentity(domain=DomainKind.ENGINE_FUEL_CONVERTER, proposal_id="ENG-P01"),
                domain=DomainKind.TRANSMISSION_DRIVELINE,
                configuration=effective.configuration,
                based_on=effective,
            )


class ProposalBasedOnEffectiveCurrentTests(unittest.TestCase):
    def test_proposal_is_based_on_effective_current_after_correction(self):
        source = _trans_source(final_drive_ratio=3.73)
        correction = DomainCorrection(
            domain=DomainKind.TRANSMISSION_DRIVELINE,
            configuration=replace(source.configuration, final_drive_ratio=3.70),
        )
        effective = resolve_effective_domain_state(source, correction)
        proposal = DomainProposal(
            identity=DomainProposalIdentity(domain=DomainKind.TRANSMISSION_DRIVELINE, proposal_id="TRANS-P01"),
            domain=DomainKind.TRANSMISSION_DRIVELINE,
            configuration=replace(effective.configuration, gear_count=9),
            based_on=effective,
        )
        self.assertIs(proposal.based_on, effective)
        self.assertEqual(proposal.based_on.configuration.final_drive_ratio, 3.70)

    def test_proposal_based_on_domain_must_match_proposal_domain(self):
        wrong_domain_effective = resolve_effective_domain_state(
            DomainSourceState(domain=DomainKind.ENGINE_FUEL_CONVERTER, configuration=EngineConfiguration())
        )
        with self.assertRaises(ValueError):
            DomainProposal(
                identity=DomainProposalIdentity(domain=DomainKind.TRANSMISSION_DRIVELINE, proposal_id="TRANS-P01"),
                domain=DomainKind.TRANSMISSION_DRIVELINE,
                configuration=TransmissionConfiguration(),
                based_on=wrong_domain_effective,
            )


class ProposalToProposalRejectedTests(unittest.TestCase):
    def test_proposal_based_on_another_proposal_is_rejected(self):
        effective = _trans_effective()
        first = DomainProposal(
            identity=DomainProposalIdentity(domain=DomainKind.TRANSMISSION_DRIVELINE, proposal_id="TRANS-P01"),
            domain=DomainKind.TRANSMISSION_DRIVELINE, configuration=effective.configuration, based_on=effective,
        )
        with self.assertRaises(TypeError):
            DomainProposal(
                identity=DomainProposalIdentity(domain=DomainKind.TRANSMISSION_DRIVELINE, proposal_id="TRANS-P02"),
                domain=DomainKind.TRANSMISSION_DRIVELINE,
                configuration=first.configuration,
                based_on=first,  # a DomainProposal, not an EffectiveDomainState
            )

    def test_proposal_based_on_a_non_effective_state_object_is_rejected(self):
        with self.assertRaises(TypeError):
            DomainProposal(
                identity=DomainProposalIdentity(domain=DomainKind.TRANSMISSION_DRIVELINE, proposal_id="TRANS-P01"),
                domain=DomainKind.TRANSMISSION_DRIVELINE,
                configuration=TransmissionConfiguration(),
                based_on=_trans_source(),  # a DomainSourceState, not EffectiveDomainState
            )


class DifferentVehicleDemandPerSystemScenarioTests(unittest.TestCase):
    def test_two_system_scenarios_may_reference_different_vde_sources(self):
        vd_a = resolve_effective_domain_state(
            DomainSourceState(
                domain=DomainKind.VEHICLE_DEMAND,
                configuration=VehicleDemandConfiguration(source_identity="vde:900001"),
            )
        )
        vd_b = resolve_effective_domain_state(
            DomainSourceState(
                domain=DomainKind.VEHICLE_DEMAND,
                configuration=VehicleDemandConfiguration(source_identity="vde:900002"),
            )
        )
        current = SystemScenarioDefinition(
            identity=SystemScenarioIdentity(scenario_id="SYS-CURRENT", role=SystemScenarioRole.CURRENT),
            slots={DomainKind.VEHICLE_DEMAND: vd_a},
        )
        proposal_a = SystemScenarioDefinition(
            identity=SystemScenarioIdentity(scenario_id="SYS-A", role=SystemScenarioRole.PROPOSAL, proposal_index=1),
            slots={DomainKind.VEHICLE_DEMAND: vd_b},
        )
        self.assertNotEqual(
            current.vehicle_demand_selection.configuration.source_identity,
            proposal_a.vehicle_demand_selection.configuration.source_identity,
        )


class SharedDomainProposalReuseTests(unittest.TestCase):
    def test_same_domain_proposal_referenced_by_two_system_scenarios_is_unmutated(self):
        effective = resolve_effective_domain_state(
            DomainSourceState(domain=DomainKind.ENGINE_FUEL_CONVERTER, configuration=EngineConfiguration(fuel_type="Gasoline"))
        )
        shared_engine_proposal = DomainProposal(
            identity=DomainProposalIdentity(domain=DomainKind.ENGINE_FUEL_CONVERTER, proposal_id="ENG-P01"),
            domain=DomainKind.ENGINE_FUEL_CONVERTER,
            configuration=replace(effective.configuration, rated_power_kw=120.0),
            based_on=effective,
        )
        scenario_a = SystemScenarioDefinition(
            identity=SystemScenarioIdentity(scenario_id="SYS-A", role=SystemScenarioRole.PROPOSAL, proposal_index=1),
            slots={DomainKind.ENGINE_FUEL_CONVERTER: shared_engine_proposal},
        )
        scenario_b = SystemScenarioDefinition(
            identity=SystemScenarioIdentity(scenario_id="SYS-B", role=SystemScenarioRole.PROPOSAL, proposal_index=2),
            slots={DomainKind.ENGINE_FUEL_CONVERTER: shared_engine_proposal},
        )
        self.assertIs(scenario_a.slots[DomainKind.ENGINE_FUEL_CONVERTER], scenario_b.slots[DomainKind.ENGINE_FUEL_CONVERTER])
        self.assertEqual(shared_engine_proposal.configuration.rated_power_kw, 120.0)

        # Editing scenario A's OTHER domain must not touch the shared proposal.
        trans_effective = _trans_effective()
        scenario_a_edited = SystemScenarioDefinition(
            identity=scenario_a.identity,
            slots={DomainKind.ENGINE_FUEL_CONVERTER: shared_engine_proposal, DomainKind.TRANSMISSION_DRIVELINE: trans_effective},
        )
        self.assertEqual(shared_engine_proposal.configuration.rated_power_kw, 120.0)
        self.assertNotIn(DomainKind.TRANSMISSION_DRIVELINE, scenario_b.slots)


class AllEightDomainsRepresentableTests(unittest.TestCase):
    def test_all_eight_domains_have_a_matching_configuration_type(self):
        from src.vde_core.system_scenario import configuration_type_for

        self.assertEqual(len(ALL_DOMAIN_KINDS), 8)
        for domain in ALL_DOMAIN_KINDS:
            config_type = configuration_type_for(domain)
            self.assertTrue(hasattr(config_type, "__dataclass_fields__"))

    def test_a_system_scenario_definition_can_populate_all_eight_domains(self):
        from src.vde_core.system_scenario import (
            AuxThermalConfiguration,
            ControlsConfiguration,
            ElectricDriveConfiguration,
            EnergyStorageConfiguration,
        )

        slots = {
            DomainKind.VEHICLE_DEMAND: resolve_effective_domain_state(
                DomainSourceState(domain=DomainKind.VEHICLE_DEMAND, configuration=VehicleDemandConfiguration())
            ),
            DomainKind.ARCHITECTURE: resolve_effective_domain_state(
                DomainSourceState(
                    domain=DomainKind.ARCHITECTURE,
                    configuration=ArchitectureConfiguration(architecture_class=ArchitectureClass.HEV),
                )
            ),
            DomainKind.ENGINE_FUEL_CONVERTER: resolve_effective_domain_state(
                DomainSourceState(domain=DomainKind.ENGINE_FUEL_CONVERTER, configuration=EngineConfiguration())
            ),
            DomainKind.TRANSMISSION_DRIVELINE: _trans_effective(),
            DomainKind.ELECTRIC_DRIVE: resolve_effective_domain_state(
                DomainSourceState(domain=DomainKind.ELECTRIC_DRIVE, configuration=ElectricDriveConfiguration())
            ),
            DomainKind.ENERGY_STORAGE: resolve_effective_domain_state(
                DomainSourceState(domain=DomainKind.ENERGY_STORAGE, configuration=EnergyStorageConfiguration())
            ),
            DomainKind.ENERGY_MANAGEMENT_CONTROLS: resolve_effective_domain_state(
                DomainSourceState(domain=DomainKind.ENERGY_MANAGEMENT_CONTROLS, configuration=ControlsConfiguration())
            ),
            DomainKind.AUX_THERMAL: resolve_effective_domain_state(
                DomainSourceState(domain=DomainKind.AUX_THERMAL, configuration=AuxThermalConfiguration())
            ),
        }
        definition = SystemScenarioDefinition(
            identity=SystemScenarioIdentity(scenario_id="SYS-CURRENT", role=SystemScenarioRole.CURRENT),
            slots=slots,
        )
        self.assertEqual(set(definition.slots.keys()), set(ALL_DOMAIN_KINDS))

    def test_mismatched_configuration_type_for_a_domain_is_rejected(self):
        with self.assertRaises(TypeError):
            DomainSourceState(domain=DomainKind.ENGINE_FUEL_CONVERTER, configuration=TransmissionConfiguration())


class ArchitectureClassificationTests(unittest.TestCase):
    def test_all_five_architecture_classes_are_valid(self):
        for value in ("ICE", "MHEV", "HEV", "PHEV", "BEV"):
            self.assertEqual(ArchitectureClass(value).value, value)

    def test_bev_typically_excludes_engine_fuel_converter(self):
        self.assertFalse(domain_typically_applicable(ArchitectureClass.BEV, DomainKind.ENGINE_FUEL_CONVERTER))

    def test_ice_typically_excludes_electric_drive_and_energy_storage(self):
        self.assertFalse(domain_typically_applicable(ArchitectureClass.ICE, DomainKind.ELECTRIC_DRIVE))
        self.assertFalse(domain_typically_applicable(ArchitectureClass.ICE, DomainKind.ENERGY_STORAGE))

    def test_bev_still_typically_applicable_for_electric_drive(self):
        self.assertTrue(domain_typically_applicable(ArchitectureClass.BEV, DomainKind.ELECTRIC_DRIVE))

    def test_hev_has_no_typically_inapplicable_domain(self):
        for domain in ALL_DOMAIN_KINDS:
            self.assertTrue(domain_typically_applicable(ArchitectureClass.HEV, domain))


class FidelityStateTests(unittest.TestCase):
    def test_all_four_fidelity_levels_are_valid(self):
        for value in ("QUANTITATIVE", "EFFECTIVE_ASSUMPTION", "CONFIGURATION_ONLY", "NOT_REPRESENTED"):
            self.assertEqual(FidelityLevel(value).value, value)

    def test_unpopulated_domain_defaults_to_not_represented(self):
        manifest = FidelityManifest(per_domain={DomainKind.ARCHITECTURE: FidelityLevel.CONFIGURATION_ONLY})
        self.assertEqual(manifest.fidelity_for(DomainKind.ENGINE_FUEL_CONVERTER), FidelityLevel.NOT_REPRESENTED)

    def test_resolve_system_scenario_shell_marks_vehicle_demand_quantitative_when_result_present(self):
        from src.vde_core.vehicle_demand import (
            RoadloadBasis,
            VehicleDemandResult,
            VehicleDemandSummary,
        )

        vd_result = VehicleDemandResult(
            total_summary=VehicleDemandSummary(roadload_basis=RoadloadBasis.TOTAL, vde_mj_per_km=1.5),
            net_summary=None,
        )
        vd_effective = resolve_effective_domain_state(
            DomainSourceState(
                domain=DomainKind.VEHICLE_DEMAND,
                configuration=VehicleDemandConfiguration(vehicle_demand_result=vd_result),
            )
        )
        definition = SystemScenarioDefinition(
            identity=SystemScenarioIdentity(scenario_id="SYS-CURRENT", role=SystemScenarioRole.CURRENT),
            slots={DomainKind.VEHICLE_DEMAND: vd_effective},
        )
        resolved = resolve_system_scenario_shell(definition)
        self.assertEqual(resolved.fidelity_manifest.fidelity_for(DomainKind.VEHICLE_DEMAND), FidelityLevel.QUANTITATIVE)

    def test_resolve_system_scenario_shell_marks_vehicle_demand_not_represented_without_result(self):
        vd_effective = resolve_effective_domain_state(
            DomainSourceState(domain=DomainKind.VEHICLE_DEMAND, configuration=VehicleDemandConfiguration())
        )
        definition = SystemScenarioDefinition(
            identity=SystemScenarioIdentity(scenario_id="SYS-CURRENT", role=SystemScenarioRole.CURRENT),
            slots={DomainKind.VEHICLE_DEMAND: vd_effective},
        )
        resolved = resolve_system_scenario_shell(definition)
        self.assertEqual(resolved.fidelity_manifest.fidelity_for(DomainKind.VEHICLE_DEMAND), FidelityLevel.NOT_REPRESENTED)


class ConfigurationOnlyVsQuantitativeTests(unittest.TestCase):
    def test_populated_non_vehicle_demand_domain_is_configuration_only_not_quantitative(self):
        definition = SystemScenarioDefinition(
            identity=SystemScenarioIdentity(scenario_id="SYS-CURRENT", role=SystemScenarioRole.CURRENT),
            slots={DomainKind.TRANSMISSION_DRIVELINE: _trans_effective()},
        )
        resolved = resolve_system_scenario_shell(definition)
        self.assertEqual(
            resolved.fidelity_manifest.fidelity_for(DomainKind.TRANSMISSION_DRIVELINE), FidelityLevel.CONFIGURATION_ONLY
        )
        self.assertNotEqual(
            resolved.fidelity_manifest.fidelity_for(DomainKind.TRANSMISSION_DRIVELINE), FidelityLevel.QUANTITATIVE
        )

    def test_manifest_reports_no_quantitative_claim_when_everything_is_configuration_only(self):
        manifest = FidelityManifest(
            per_domain={
                DomainKind.TRANSMISSION_DRIVELINE: FidelityLevel.CONFIGURATION_ONLY,
                DomainKind.ENGINE_FUEL_CONVERTER: FidelityLevel.CONFIGURATION_ONLY,
            }
        )
        self.assertTrue(manifest.is_configuration_only_everywhere_quantitative_is_absent)

    def test_manifest_reports_a_quantitative_claim_when_at_least_one_domain_is_quantitative(self):
        manifest = FidelityManifest(
            per_domain={
                DomainKind.VEHICLE_DEMAND: FidelityLevel.QUANTITATIVE,
                DomainKind.TRANSMISSION_DRIVELINE: FidelityLevel.CONFIGURATION_ONLY,
            }
        )
        self.assertFalse(manifest.is_configuration_only_everywhere_quantitative_is_absent)


class SourceImmutabilityTests(unittest.TestCase):
    def test_correction_does_not_mutate_source_configuration(self):
        source = _trans_source(final_drive_ratio=3.73)
        correction = DomainCorrection(
            domain=DomainKind.TRANSMISSION_DRIVELINE,
            configuration=replace(source.configuration, final_drive_ratio=3.70),
        )
        resolve_effective_domain_state(source, correction)
        self.assertEqual(source.configuration.final_drive_ratio, 3.73)

    def test_source_dataclass_is_frozen(self):
        source = _trans_source()
        with self.assertRaises(Exception):
            source.configuration = TransmissionConfiguration()

    def test_proposal_construction_does_not_mutate_effective_or_source(self):
        source = _trans_source(final_drive_ratio=3.73)
        effective = resolve_effective_domain_state(source)
        DomainProposal(
            identity=DomainProposalIdentity(domain=DomainKind.TRANSMISSION_DRIVELINE, proposal_id="TRANS-P01"),
            domain=DomainKind.TRANSMISSION_DRIVELINE,
            configuration=replace(effective.configuration, final_drive_ratio=3.20),
            based_on=effective,
        )
        self.assertEqual(effective.configuration.final_drive_ratio, 3.73)
        self.assertEqual(source.configuration.final_drive_ratio, 3.73)


class CorrectionProducesEffectiveCurrentTests(unittest.TestCase):
    def test_no_correction_effective_equals_source_configuration(self):
        source = _trans_source(final_drive_ratio=3.73)
        effective = resolve_effective_domain_state(source)
        self.assertEqual(effective.configuration, source.configuration)
        self.assertIsNone(effective.correction)
        self.assertEqual(effective.provenance, ProvenanceKind.SOURCE_OBSERVED)

    def test_with_correction_effective_reflects_corrected_value(self):
        source = _trans_source(final_drive_ratio=3.73)
        correction = DomainCorrection(
            domain=DomainKind.TRANSMISSION_DRIVELINE,
            configuration=replace(source.configuration, final_drive_ratio=3.70),
            reason="engineering correction",
        )
        effective = resolve_effective_domain_state(source, correction)
        self.assertEqual(effective.configuration.final_drive_ratio, 3.70)
        self.assertIs(effective.correction, correction)
        self.assertEqual(effective.provenance, ProvenanceKind.CORRECTED)

    def test_correction_domain_mismatch_is_rejected(self):
        source = _trans_source()
        mismatched_correction = DomainCorrection(
            domain=DomainKind.ENGINE_FUEL_CONVERTER, configuration=EngineConfiguration()
        )
        # A domain-mismatched correction is necessarily also a
        # configuration-type mismatch (each domain has its own configuration
        # type) -- EffectiveDomainState's configuration-type check fires
        # first and raises TypeError; either way, the mismatched correction
        # is rejected, which is the actual invariant under test.
        with self.assertRaises((TypeError, ValueError)):
            resolve_effective_domain_state(source, mismatched_correction)


class SerializationRoundtripTests(unittest.TestCase):
    def test_domain_source_state_roundtrip(self):
        source = _trans_source(final_drive_ratio=3.73, gear_count=8)
        data = to_serializable(source)
        restored = domain_source_state_from_dict(data)
        self.assertEqual(restored, source)

    def test_effective_domain_state_roundtrip_with_correction(self):
        source = _trans_source(final_drive_ratio=3.73)
        correction = DomainCorrection(
            domain=DomainKind.TRANSMISSION_DRIVELINE,
            configuration=replace(source.configuration, final_drive_ratio=3.70),
            reason="corrected",
        )
        effective = resolve_effective_domain_state(source, correction)
        data = to_serializable(effective)
        restored = effective_domain_state_from_dict(data)
        self.assertEqual(restored, effective)

    def test_domain_proposal_roundtrip(self):
        effective = _trans_effective()
        proposal = DomainProposal(
            identity=DomainProposalIdentity(domain=DomainKind.TRANSMISSION_DRIVELINE, proposal_id="TRANS-P01"),
            domain=DomainKind.TRANSMISSION_DRIVELINE,
            configuration=replace(effective.configuration, gear_count=9),
            based_on=effective,
            label="9AT Swap",
            l0_effective_assumption={"pse_percent_delta": 0.8},
        )
        data = to_serializable(proposal)
        restored = domain_proposal_from_dict(data)
        self.assertEqual(restored, proposal)

    def test_fidelity_manifest_roundtrip(self):
        manifest = FidelityManifest(
            per_domain={
                DomainKind.VEHICLE_DEMAND: FidelityLevel.QUANTITATIVE,
                DomainKind.TRANSMISSION_DRIVELINE: FidelityLevel.CONFIGURATION_ONLY,
            }
        )
        data = to_serializable(manifest)
        restored = fidelity_manifest_from_dict(data)
        self.assertEqual(restored, manifest)

    def test_system_scenario_identity_roundtrip(self):
        identity = SystemScenarioIdentity(scenario_id="SYS-P01", role=SystemScenarioRole.PROPOSAL, proposal_index=1)
        data = to_serializable(identity)
        restored = system_scenario_identity_from_dict(data)
        self.assertEqual(restored, identity)

    def test_serialization_never_confuses_none_with_zero(self):
        config = TransmissionConfiguration(final_drive_ratio=0.0, gear_count=None)
        data = to_serializable(config)
        self.assertEqual(data["final_drive_ratio"], 0.0)
        self.assertIsNone(data["gear_count"])


if __name__ == "__main__":
    unittest.main()
