"""Sprint 11B: tests for src.vde_core.system_scenario.domain_resolution
(resolve_domain_proposal / changed_fields) and the DomainApplicability
classification in contracts.py. Streamlit-free, no DB. Also covers the
Sprint 11B spec's lettered Acceptance Cases A-J.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from src.vde_core.quick_scenario.contracts import TechDeltaAssumption
from src.vde_core.system_scenario import (
    ArchitectureClass,
    ArchitectureConfiguration,
    ControlsConfiguration,
    DomainApplicability,
    DomainKind,
    DomainProposal,
    DomainProposalIdentity,
    DomainSourceState,
    EffectiveDomainState,
    EnergyStorageConfiguration,
    EngineConfiguration,
    FidelityLevel,
    ProvenanceKind,
    TransmissionConfiguration,
    changed_fields,
    domain_applicability_for,
    resolve_domain_proposal,
    resolve_effective_domain_state,
    resolve_system_scenario_shell,
)
from src.vde_core.system_scenario.contracts import (
    SystemScenarioDefinition,
    SystemScenarioIdentity,
    SystemScenarioRole,
)


def _trans_effective(final_drive_ratio=3.73, gear_count=8, transmission_type="8AT") -> EffectiveDomainState:
    source = DomainSourceState(
        domain=DomainKind.TRANSMISSION_DRIVELINE,
        configuration=TransmissionConfiguration(
            final_drive_ratio=final_drive_ratio, gear_count=gear_count, transmission_type=transmission_type
        ),
    )
    return resolve_effective_domain_state(source)


def _engine_effective(displacement_l=2.0, rated_power_kw=200.0) -> EffectiveDomainState:
    source = DomainSourceState(
        domain=DomainKind.ENGINE_FUEL_CONVERTER,
        configuration=EngineConfiguration(displacement_l=displacement_l, rated_power_kw=rated_power_kw),
    )
    return resolve_effective_domain_state(source)


class ResolveDomainProposalTests(unittest.TestCase):
    def test_unrequested_fields_inherit_from_effective_current(self):
        effective = _trans_effective(final_drive_ratio=3.73, gear_count=8)
        proposal = resolve_domain_proposal(
            DomainProposalIdentity(domain=DomainKind.TRANSMISSION_DRIVELINE, proposal_id="TRANS-P01"),
            effective,
            requested_changes={"gear_count": 9},
        )
        self.assertEqual(proposal.configuration.gear_count, 9)
        self.assertEqual(proposal.configuration.final_drive_ratio, 3.73)  # inherited, not requested
        self.assertEqual(proposal.configuration.transmission_type, "8AT")  # inherited

    def test_explicit_override_resolves_correctly(self):
        effective = _engine_effective()
        proposal = resolve_domain_proposal(
            DomainProposalIdentity(domain=DomainKind.ENGINE_FUEL_CONVERTER, proposal_id="ENG-P01"),
            effective,
            requested_changes={"rated_power_kw": 180.0},
        )
        self.assertEqual(proposal.configuration.rated_power_kw, 180.0)

    def test_explicit_zero_in_requested_changes_is_preserved(self):
        effective = _trans_effective(final_drive_ratio=3.73)
        proposal = resolve_domain_proposal(
            DomainProposalIdentity(domain=DomainKind.TRANSMISSION_DRIVELINE, proposal_id="TRANS-P01"),
            effective,
            requested_changes={"final_drive_ratio": 0.0},
        )
        self.assertEqual(proposal.configuration.final_drive_ratio, 0.0)
        self.assertIsNotNone(proposal.configuration.final_drive_ratio)

    def test_unrequested_missing_field_stays_missing(self):
        source = DomainSourceState(domain=DomainKind.ENGINE_FUEL_CONVERTER, configuration=EngineConfiguration(fuel_type="Gasoline"))
        effective = resolve_effective_domain_state(source)
        proposal = resolve_domain_proposal(
            DomainProposalIdentity(domain=DomainKind.ENGINE_FUEL_CONVERTER, proposal_id="ENG-P01"),
            effective,
            requested_changes={"rated_power_kw": 180.0},
        )
        self.assertIsNone(proposal.configuration.rated_torque_nm)  # never touched, was already None

    def test_invalid_requested_field_name_raises(self):
        effective = _trans_effective()
        with self.assertRaises(TypeError):
            resolve_domain_proposal(
                DomainProposalIdentity(domain=DomainKind.TRANSMISSION_DRIVELINE, proposal_id="TRANS-P01"),
                effective,
                requested_changes={"rated_power_kw": 180.0},  # not a TransmissionConfiguration field
            )

    def test_domain_mismatch_between_identity_and_based_on_is_rejected(self):
        effective = _trans_effective()
        # A domain-mismatched based_on is necessarily also a configuration-
        # type mismatch (the mismatched domain's configuration type differs
        # from what based_on actually carries) -- DomainProposal's
        # configuration-type check fires first and raises TypeError; either
        # way the mismatch is rejected, which is the actual invariant here.
        with self.assertRaises((TypeError, ValueError)):
            resolve_domain_proposal(
                DomainProposalIdentity(domain=DomainKind.ENGINE_FUEL_CONVERTER, proposal_id="ENG-P01"),
                effective,
            )

    def test_requested_changes_are_stored_as_provenance(self):
        effective = _trans_effective()
        proposal = resolve_domain_proposal(
            DomainProposalIdentity(domain=DomainKind.TRANSMISSION_DRIVELINE, proposal_id="TRANS-P01"),
            effective,
            requested_changes={"gear_count": 9},
        )
        self.assertEqual(proposal.requested_changes, {"gear_count": 9})

    def test_technology_deltas_and_l0_assumption_pass_through_unmodified(self):
        effective = _trans_effective()
        delta = TechDeltaAssumption(name="9-speed efficiency gain", effect_basis="pse_percent_delta", effect_value=0.8)
        proposal = resolve_domain_proposal(
            DomainProposalIdentity(domain=DomainKind.TRANSMISSION_DRIVELINE, proposal_id="TRANS-P01"),
            effective,
            requested_changes={"gear_count": 9},
            l0_effective_assumption={"pse_percent_delta": 0.8},
            technology_deltas=(delta,),
        )
        self.assertEqual(proposal.l0_effective_assumption, {"pse_percent_delta": 0.8})
        self.assertEqual(proposal.technology_deltas, (delta,))
        self.assertEqual(proposal.technology_deltas[0].affected_subsystem, "whole powertrain")

    def test_multiple_technology_deltas_are_preserved_unstacked_in_local_order(self):
        # Sec 20/21: association only -- no combined/stacked value is ever
        # computed here, and local order within the tuple is preserved
        # exactly as given (a plain tuple, never a set/dict).
        effective = _trans_effective()
        delta_a = TechDeltaAssumption(name="Delta A", effect_basis="pse_percent_delta", effect_value=0.5)
        delta_b = TechDeltaAssumption(name="Delta B", effect_basis="pse_delta", effect_value=0.01)
        proposal = resolve_domain_proposal(
            DomainProposalIdentity(domain=DomainKind.TRANSMISSION_DRIVELINE, proposal_id="TRANS-P01"),
            effective,
            technology_deltas=(delta_a, delta_b),
        )
        self.assertEqual(proposal.technology_deltas, (delta_a, delta_b))
        self.assertEqual(proposal.technology_deltas[0].name, "Delta A")
        self.assertEqual(proposal.technology_deltas[1].name, "Delta B")
        # No stacked/combined field exists anywhere on the proposal -- the
        # only quantitative representation is whatever the caller supplied
        # in l0_effective_assumption, verbatim.
        self.assertEqual(proposal.l0_effective_assumption, {})

    def test_only_11c_l0_adapter_imports_the_canonical_delta_stack(self):
        # The 11A/11B contract/domain services still do not stack. Sprint
        # 11C's dedicated adapter imports the exact canonical owner rather
        # than a local copy.
        import src.vde_core.system_scenario.contracts as contracts_module
        import src.vde_core.system_scenario.domain_resolution as domain_resolution_module
        import src.vde_core.system_scenario.legacy_adapter as legacy_adapter_module
        import src.vde_core.system_scenario.l0_adapter as l0_adapter_module
        from src.vde_core.technology_delta import apply_delta_stack_to_baseline

        for module in (contracts_module, domain_resolution_module, legacy_adapter_module):
            self.assertFalse(hasattr(module, "apply_delta_stack_to_baseline"))
        self.assertIs(l0_adapter_module.apply_delta_stack_to_baseline, apply_delta_stack_to_baseline)


class ChangedFieldsTests(unittest.TestCase):
    def test_changed_fields_reports_only_actual_differences(self):
        effective = _trans_effective(final_drive_ratio=3.70, gear_count=8)
        proposal = resolve_domain_proposal(
            DomainProposalIdentity(domain=DomainKind.TRANSMISSION_DRIVELINE, proposal_id="TRANS-P01"),
            effective,
            requested_changes={"gear_count": 9, "final_drive_ratio": 3.45},
        )
        diffs = changed_fields(proposal)
        self.assertEqual(diffs, {"gear_count": (8, 9), "final_drive_ratio": (3.70, 3.45)})

    def test_changed_fields_empty_when_proposal_matches_effective_current(self):
        effective = _trans_effective()
        proposal = resolve_domain_proposal(
            DomainProposalIdentity(domain=DomainKind.TRANSMISSION_DRIVELINE, proposal_id="TRANS-NEUTRAL"),
            effective,
        )
        self.assertEqual(changed_fields(proposal), {})

    def test_changed_fields_works_for_a_manually_constructed_proposal_too(self):
        # Robustness: changed_fields diffs configurations directly, so it
        # stays correct even bypassing resolve_domain_proposal.
        effective = _trans_effective(gear_count=8)
        proposal = DomainProposal(
            identity=DomainProposalIdentity(domain=DomainKind.TRANSMISSION_DRIVELINE, proposal_id="TRANS-P02"),
            domain=DomainKind.TRANSMISSION_DRIVELINE,
            configuration=replace(effective.configuration, gear_count=10),
            based_on=effective,
        )
        self.assertEqual(changed_fields(proposal), {"gear_count": (8, 10)})


class DomainApplicabilityTests(unittest.TestCase):
    def test_bev_engine_is_not_applicable(self):
        self.assertEqual(
            domain_applicability_for(ArchitectureClass.BEV, DomainKind.ENGINE_FUEL_CONVERTER),
            DomainApplicability.NOT_APPLICABLE,
        )

    def test_bev_electric_drive_and_energy_storage_are_required(self):
        self.assertEqual(
            domain_applicability_for(ArchitectureClass.BEV, DomainKind.ELECTRIC_DRIVE), DomainApplicability.REQUIRED
        )
        self.assertEqual(
            domain_applicability_for(ArchitectureClass.BEV, DomainKind.ENERGY_STORAGE), DomainApplicability.REQUIRED
        )

    def test_ice_engine_is_required(self):
        self.assertEqual(
            domain_applicability_for(ArchitectureClass.ICE, DomainKind.ENGINE_FUEL_CONVERTER),
            DomainApplicability.REQUIRED,
        )

    def test_ice_electric_drive_and_energy_storage_are_not_applicable(self):
        self.assertEqual(
            domain_applicability_for(ArchitectureClass.ICE, DomainKind.ELECTRIC_DRIVE), DomainApplicability.NOT_APPLICABLE
        )
        self.assertEqual(
            domain_applicability_for(ArchitectureClass.ICE, DomainKind.ENERGY_STORAGE), DomainApplicability.NOT_APPLICABLE
        )

    def test_mhev_hev_phev_require_both_thermal_and_electric_domains(self):
        for architecture in (ArchitectureClass.MHEV, ArchitectureClass.HEV, ArchitectureClass.PHEV):
            self.assertEqual(
                domain_applicability_for(architecture, DomainKind.ENGINE_FUEL_CONVERTER), DomainApplicability.REQUIRED
            )
            self.assertEqual(
                domain_applicability_for(architecture, DomainKind.ELECTRIC_DRIVE), DomainApplicability.REQUIRED
            )
            self.assertEqual(
                domain_applicability_for(architecture, DomainKind.ENERGY_STORAGE), DomainApplicability.REQUIRED
            )

    def test_controls_and_aux_thermal_remain_optional_not_not_applicable_everywhere(self):
        for architecture in ArchitectureClass:
            self.assertEqual(
                domain_applicability_for(architecture, DomainKind.ENERGY_MANAGEMENT_CONTROLS),
                DomainApplicability.OPTIONAL,
            )
            self.assertEqual(
                domain_applicability_for(architecture, DomainKind.AUX_THERMAL), DomainApplicability.OPTIONAL
            )

    def test_vehicle_demand_and_architecture_are_always_required(self):
        for architecture in ArchitectureClass:
            self.assertEqual(
                domain_applicability_for(architecture, DomainKind.VEHICLE_DEMAND), DomainApplicability.REQUIRED
            )
            self.assertEqual(
                domain_applicability_for(architecture, DomainKind.ARCHITECTURE), DomainApplicability.REQUIRED
            )

    def test_applicability_is_purely_advisory_missing_engine_data_does_not_raise_for_bev(self):
        # Case E: missing Engine data does not create an error, even though
        # Engine is NOT_APPLICABLE for BEV -- applicability never gates
        # construction.
        empty_engine = resolve_effective_domain_state(
            DomainSourceState(domain=DomainKind.ENGINE_FUEL_CONVERTER, configuration=EngineConfiguration())
        )
        self.assertIsNone(empty_engine.configuration.rated_power_kw)
        self.assertEqual(
            domain_applicability_for(ArchitectureClass.BEV, DomainKind.ENGINE_FUEL_CONVERTER),
            DomainApplicability.NOT_APPLICABLE,
        )


class AcceptanceCaseTests(unittest.TestCase):
    """Sprint 11B spec Sec 31, Cases A-J."""

    def test_case_a_transmission_correction(self):
        from src.vde_core.system_scenario import DomainCorrection

        source = DomainSourceState(
            domain=DomainKind.TRANSMISSION_DRIVELINE, configuration=TransmissionConfiguration(final_drive_ratio=3.73)
        )
        correction = DomainCorrection(
            domain=DomainKind.TRANSMISSION_DRIVELINE, configuration=replace(source.configuration, final_drive_ratio=3.70)
        )
        effective = resolve_effective_domain_state(source, correction)
        self.assertEqual(source.configuration.final_drive_ratio, 3.73)
        self.assertEqual(effective.configuration.final_drive_ratio, 3.70)
        self.assertEqual(effective.provenance, ProvenanceKind.CORRECTED)

    def test_explicit_zero_via_a_domain_correction_survives_resolution(self):
        # Item 4 of Sec 32's required test list, distinct from explicit
        # zero via a Proposal's requested_changes: a Correction setting a
        # field to 0 must resolve to Effective Current = 0, not missing.
        from src.vde_core.system_scenario import DomainCorrection

        source = DomainSourceState(
            domain=DomainKind.TRANSMISSION_DRIVELINE, configuration=TransmissionConfiguration(final_drive_ratio=3.73)
        )
        correction = DomainCorrection(
            domain=DomainKind.TRANSMISSION_DRIVELINE,
            configuration=replace(source.configuration, final_drive_ratio=0.0),
        )
        effective = resolve_effective_domain_state(source, correction)
        self.assertEqual(effective.configuration.final_drive_ratio, 0.0)
        self.assertIsNotNone(effective.configuration.final_drive_ratio)

    def test_case_b_engine_proposal_no_invented_consumption_benefit(self):
        effective = _engine_effective(displacement_l=2.0, rated_power_kw=200.0)
        proposal = resolve_domain_proposal(
            DomainProposalIdentity(domain=DomainKind.ENGINE_FUEL_CONVERTER, proposal_id="ENG-P01"),
            effective,
            requested_changes={"displacement_l": 1.6, "rated_power_kw": 180.0},
        )
        self.assertEqual(proposal.configuration.displacement_l, 1.6)
        self.assertEqual(proposal.configuration.rated_power_kw, 180.0)
        self.assertEqual(effective.configuration.displacement_l, 2.0)  # Effective Current unchanged
        self.assertEqual(proposal.l0_effective_assumption, {})  # absent unless explicitly provided

        definition = SystemScenarioDefinition(
            identity=SystemScenarioIdentity(scenario_id="SYS-A", role=SystemScenarioRole.PROPOSAL, proposal_index=1),
            slots={DomainKind.ENGINE_FUEL_CONVERTER: proposal},
        )
        resolved = resolve_system_scenario_shell(definition)
        self.assertEqual(
            resolved.fidelity_manifest.fidelity_for(DomainKind.ENGINE_FUEL_CONVERTER), FidelityLevel.CONFIGURATION_ONLY
        )

    def test_case_c_transmission_proposal_with_explicit_l0_assumption_no_stacking(self):
        effective = _trans_effective(final_drive_ratio=3.70, gear_count=8, transmission_type="8AT")
        delta = TechDeltaAssumption(
            name="9AT efficiency gain", effect_basis="pse_percent_delta", effect_value=0.8, source_type="engineering_assumption"
        )
        proposal = resolve_domain_proposal(
            DomainProposalIdentity(domain=DomainKind.TRANSMISSION_DRIVELINE, proposal_id="TRANS-P01"),
            effective,
            requested_changes={"transmission_type": "9AT", "final_drive_ratio": 3.45},
            l0_effective_assumption={"pse_percent_delta": 0.8},
            technology_deltas=(delta,),
        )
        self.assertEqual(proposal.configuration.transmission_type, "9AT")
        self.assertEqual(proposal.configuration.final_drive_ratio, 3.45)
        self.assertEqual(proposal.l0_effective_assumption, {"pse_percent_delta": 0.8})
        self.assertEqual(proposal.technology_deltas[0].source_type, "engineering_assumption")
        # No stacking/system result performed -- the proposal carries only
        # the association, nothing computes a combined/stacked value here.
        self.assertFalse(hasattr(proposal, "stacked_result"))

    def test_case_d_battery_capacity_configuration_only_unless_supplied(self):
        source = DomainSourceState(
            domain=DomainKind.ENERGY_STORAGE, configuration=EnergyStorageConfiguration(usable_capacity_kwh=1.0)
        )
        effective = resolve_effective_domain_state(source)
        proposal = resolve_domain_proposal(
            DomainProposalIdentity(domain=DomainKind.ENERGY_STORAGE, proposal_id="BAT-P01"),
            effective,
            requested_changes={"usable_capacity_kwh": 1.5},
        )
        self.assertEqual(proposal.configuration.usable_capacity_kwh, 1.5)
        self.assertEqual(proposal.l0_effective_assumption, {})
        definition = SystemScenarioDefinition(
            identity=SystemScenarioIdentity(scenario_id="SYS-A", role=SystemScenarioRole.PROPOSAL, proposal_index=1),
            slots={DomainKind.ENERGY_STORAGE: proposal},
        )
        resolved = resolve_system_scenario_shell(definition)
        self.assertEqual(
            resolved.fidelity_manifest.fidelity_for(DomainKind.ENERGY_STORAGE), FidelityLevel.CONFIGURATION_ONLY
        )

    def test_case_e_bev_applicability(self):
        self.assertEqual(
            domain_applicability_for(ArchitectureClass.BEV, DomainKind.ENGINE_FUEL_CONVERTER),
            DomainApplicability.NOT_APPLICABLE,
        )
        self.assertEqual(
            domain_applicability_for(ArchitectureClass.BEV, DomainKind.ELECTRIC_DRIVE), DomainApplicability.REQUIRED
        )
        self.assertEqual(
            domain_applicability_for(ArchitectureClass.BEV, DomainKind.ENERGY_STORAGE), DomainApplicability.REQUIRED
        )
        # missing Engine data does not raise
        empty_engine_source = DomainSourceState(domain=DomainKind.ENGINE_FUEL_CONVERTER, configuration=EngineConfiguration())
        resolve_effective_domain_state(empty_engine_source)  # must not raise

    def test_case_f_ice_applicability(self):
        self.assertEqual(
            domain_applicability_for(ArchitectureClass.ICE, DomainKind.ENGINE_FUEL_CONVERTER),
            DomainApplicability.REQUIRED,
        )
        self.assertEqual(
            domain_applicability_for(ArchitectureClass.ICE, DomainKind.ELECTRIC_DRIVE), DomainApplicability.NOT_APPLICABLE
        )
        self.assertEqual(
            domain_applicability_for(ArchitectureClass.ICE, DomainKind.ENERGY_STORAGE), DomainApplicability.NOT_APPLICABLE
        )

    def test_case_g_proposal_isolation(self):
        effective = _engine_effective(rated_power_kw=200.0)
        eng_p01 = resolve_domain_proposal(
            DomainProposalIdentity(domain=DomainKind.ENGINE_FUEL_CONVERTER, proposal_id="ENG-P01"),
            effective,
            requested_changes={"rated_power_kw": 180.0},
        )
        eng_p02 = resolve_domain_proposal(
            DomainProposalIdentity(domain=DomainKind.ENGINE_FUEL_CONVERTER, proposal_id="ENG-P02"),
            effective,
            requested_changes={"rated_power_kw": 220.0},
        )
        # "Editing/resolving" ENG-P01 further (constructing yet another
        # proposal from the same Effective Current) must not alter ENG-P02
        # or Effective Current.
        resolve_domain_proposal(
            DomainProposalIdentity(domain=DomainKind.ENGINE_FUEL_CONVERTER, proposal_id="ENG-P01-EDIT"),
            effective,
            requested_changes={"rated_power_kw": 150.0},
        )
        self.assertEqual(eng_p01.configuration.rated_power_kw, 180.0)
        self.assertEqual(eng_p02.configuration.rated_power_kw, 220.0)
        self.assertEqual(effective.configuration.rated_power_kw, 200.0)

    def test_case_h_proposal_reuse_no_mutation_or_copy(self):
        effective = _engine_effective()
        eng_p01 = resolve_domain_proposal(
            DomainProposalIdentity(domain=DomainKind.ENGINE_FUEL_CONVERTER, proposal_id="ENG-P01"),
            effective,
            requested_changes={"rated_power_kw": 180.0},
        )
        definition_a = SystemScenarioDefinition(
            identity=SystemScenarioIdentity(scenario_id="SYS-A", role=SystemScenarioRole.PROPOSAL, proposal_index=1),
            slots={DomainKind.ENGINE_FUEL_CONVERTER: eng_p01},
        )
        definition_b = SystemScenarioDefinition(
            identity=SystemScenarioIdentity(scenario_id="SYS-B", role=SystemScenarioRole.PROPOSAL, proposal_index=2),
            slots={DomainKind.ENGINE_FUEL_CONVERTER: eng_p01},
        )
        self.assertIs(definition_a.slots[DomainKind.ENGINE_FUEL_CONVERTER], definition_b.slots[DomainKind.ENGINE_FUEL_CONVERTER])
        self.assertIs(definition_a.slots[DomainKind.ENGINE_FUEL_CONVERTER], eng_p01)

    def test_case_i_explicit_zero_survives_resolution(self):
        source = DomainSourceState(
            domain=DomainKind.ENERGY_MANAGEMENT_CONTROLS,
            configuration=ControlsConfiguration(utility_factor_pct=50.0),
        )
        effective = resolve_effective_domain_state(source)
        proposal = resolve_domain_proposal(
            DomainProposalIdentity(domain=DomainKind.ENERGY_MANAGEMENT_CONTROLS, proposal_id="CTRL-P01"),
            effective,
            requested_changes={"utility_factor_pct": 0.0},
        )
        self.assertEqual(proposal.configuration.utility_factor_pct, 0.0)
        self.assertIsNotNone(proposal.configuration.utility_factor_pct)

    def test_case_j_sparse_domain_no_fabricated_defaults_no_exception(self):
        from src.vde_core.system_scenario import AuxThermalConfiguration

        source = DomainSourceState(domain=DomainKind.AUX_THERMAL, configuration=AuxThermalConfiguration())
        effective = resolve_effective_domain_state(source)  # must not raise
        self.assertIsNone(effective.configuration.ambient_temp_c)
        self.assertIsNone(effective.configuration.ac_on)
        self.assertIsNone(effective.configuration.notes)
        definition = SystemScenarioDefinition(
            identity=SystemScenarioIdentity(scenario_id="SYS-CURRENT", role=SystemScenarioRole.CURRENT),
            slots={DomainKind.AUX_THERMAL: effective},
        )
        resolved = resolve_system_scenario_shell(definition)  # must not raise
        self.assertEqual(resolved.fidelity_manifest.fidelity_for(DomainKind.AUX_THERMAL), FidelityLevel.CONFIGURATION_ONLY)


if __name__ == "__main__":
    unittest.main()
