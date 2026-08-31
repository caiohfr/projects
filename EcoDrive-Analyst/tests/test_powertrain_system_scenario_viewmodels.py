"""Sprint 11D direct state/orchestration coverage (no Streamlit runtime)."""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

from src.vde_app.powertrain_system_scenario_viewmodels import (
    CURRENT_SELECTION,
    NOT_APPLICABLE_SELECTION,
    ScenarioSource,
    add_proposal_draft,
    build_definition,
    calculate_drafts,
    calculation_fingerprint,
    correction_key,
    current_correction_from_editor,
    current_draft,
    effective_states_for_source,
    explainability_rows,
    friendly_issue,
    is_stale,
    metadata_incomplete_fields,
    proposal_from_editor,
    replace_draft,
    result_deltas,
    sequential_impact_trace,
    update_selection,
)
from src.vde_core.system_scenario import (
    ArchitectureClass,
    DomainKind,
    FidelityLevel,
    ProvenanceKind,
    SolverReadiness,
)
from src.vde_core.technology_delta import TechDeltaAssumption


def _source(vde_id: int = 1, *, total: float | None = 1.8, net: float | None = 1.6, **fuel):
    fuel_row = {
        "id": 100 + vde_id,
        "vde_id": vde_id,
        "electrification": "ICE",
        "fuel_type": "Gasoline",
        "eta_pt_est": 0.3,
        "energy_basis": "VDE_TOTAL",
        **fuel,
    }
    return ScenarioSource(
        vde_id=vde_id,
        vde_row={
            "id": vde_id,
            "vde_total_mj_per_km": total,
            "vde_net_mj_per_km": net,
            "cycle_name": "WLTC",
            "legislation": "WLTP",
            "engine_size_l": 2.0,
            "transmission_type": "AT",
            "gear_count": 8,
            "final_drive_ratio": 3.7,
        },
        fuelcons_row=fuel_row,
    )


class PowertrainSystemScenarioViewmodelTests(unittest.TestCase):
    def setUp(self):
        self.sources = {1: _source(1), 2: _source(2, total=2.0), 3: _source(3, total=2.2)}
        self.current = current_draft(1, ArchitectureClass.ICE)

    def test_current_only_delegates_and_is_ready(self):
        calculations = calculate_drafts((self.current,), sources=self.sources, proposals={})
        result = calculations["SYS-CURRENT"].result
        self.assertIsNotNone(result)
        self.assertIs(result.readiness, SolverReadiness.READY)
        self.assertEqual(result.selected_vehicle_demand_identity, "vde:1")

    def test_current_plus_one_and_three_proposals_have_stable_bounded_identities(self):
        drafts = add_proposal_draft((self.current,), vde_id=1, architecture=ArchitectureClass.ICE)
        self.assertEqual([item.identity.scenario_id for item in drafts], ["SYS-CURRENT", "SYS-P1"])
        drafts = add_proposal_draft(drafts, vde_id=1, architecture=ArchitectureClass.ICE)
        drafts = add_proposal_draft(drafts, vde_id=1, architecture=ArchitectureClass.ICE)
        self.assertEqual(len(drafts), 4)
        with self.assertRaisesRegex(ValueError, "at most 3"):
            add_proposal_draft(drafts, vde_id=1, architecture=ArchitectureClass.ICE)

    def test_visible_label_neither_changes_identity_nor_calculation_fingerprint(self):
        renamed = replace(self.current, label="My Current")
        self.assertEqual(renamed.identity, self.current.identity)
        self.assertEqual(
            calculation_fingerprint(self.current, sources=self.sources, proposals={}),
            calculation_fingerprint(renamed, sources=self.sources, proposals={}),
        )

    def test_each_proposal_selects_its_own_vehicle_demand(self):
        drafts = add_proposal_draft((self.current,), vde_id=2, architecture=ArchitectureClass.ICE)
        drafts = add_proposal_draft(drafts, vde_id=3, architecture=ArchitectureClass.ICE)
        calculations = calculate_drafts(drafts, sources=self.sources, proposals={})
        identities = {
            scenario_id: calculation.result.selected_vehicle_demand_identity
            for scenario_id, calculation in calculations.items()
        }
        self.assertEqual(identities, {"SYS-CURRENT": "vde:1", "SYS-P1": "vde:2", "SYS-P2": "vde:3"})

    def test_editing_proposal_a_does_not_mutate_b(self):
        drafts = add_proposal_draft((self.current,), vde_id=2, architecture=ArchitectureClass.ICE)
        drafts = add_proposal_draft(drafts, vde_id=3, architecture=ArchitectureClass.ICE)
        before_b = drafts[2]
        updated_a = update_selection(drafts[1], DomainKind.TRANSMISSION_DRIVELINE, "TRANS-P01")
        drafts = replace_draft(drafts, updated_a)
        self.assertIs(drafts[2], before_b)
        self.assertEqual(drafts[2].selection_for(DomainKind.TRANSMISSION_DRIVELINE), CURRENT_SELECTION)

    def test_shared_domain_proposal_is_reused_by_a_and_b(self):
        base = effective_states_for_source(self.sources[1])[DomainKind.ENGINE_FUEL_CONVERTER]
        proposal = proposal_from_editor(
            proposal_id="ENG-P01",
            domain=DomainKind.ENGINE_FUEL_CONVERTER,
            based_on=base,
            label="Shared engine",
            requested_changes={"rated_power_kw": 150.0},
        )
        drafts = add_proposal_draft((self.current,), vde_id=2, architecture=ArchitectureClass.ICE)
        drafts = add_proposal_draft(drafts, vde_id=3, architecture=ArchitectureClass.ICE)
        drafts = replace_draft(drafts, update_selection(drafts[1], proposal.domain, "ENG-P01"))
        drafts = replace_draft(drafts, update_selection(drafts[2], proposal.domain, "ENG-P01"))
        first, _ = build_definition(drafts[1], sources=self.sources, proposals={"ENG-P01": proposal})
        second, _ = build_definition(drafts[2], sources=self.sources, proposals={"ENG-P01": proposal})
        self.assertIs(first.slots[proposal.domain], proposal)
        self.assertIs(second.slots[proposal.domain], proposal)

    def test_bev_engine_na_and_missing_electric_assumption(self):
        bev_source = _source(4, electrification="BEV", eta_pt_est=None, fuel_type=None)
        draft = current_draft(4, ArchitectureClass.BEV)
        draft = update_selection(draft, DomainKind.ENGINE_FUEL_CONVERTER, NOT_APPLICABLE_SELECTION)
        calculations = calculate_drafts((draft,), sources={4: bev_source}, proposals={})
        result = calculations["SYS-CURRENT"].result
        self.assertNotIn(DomainKind.ENGINE_FUEL_CONVERTER, result.resolved_scenario.resolved_domains)
        self.assertIs(result.readiness, SolverReadiness.NOT_READY)
        self.assertIn("bev_eff_drive_missing", result.resolved_scenario.issues)

    def test_sparse_electric_drive_configuration_does_not_crash(self):
        hybrid_source = _source(4, electrification="HEV")
        hybrid = current_draft(4, ArchitectureClass.HEV)
        definition, _ = build_definition(hybrid, sources={4: hybrid_source}, proposals={})
        config = definition.slots[DomainKind.ELECTRIC_DRIVE].configuration
        self.assertTrue(all(getattr(config, field.name) is None for field in __import__("dataclasses").fields(config)))

    def test_battery_configuration_only_change_does_not_change_l0(self):
        hybrid_source = _source(4, electrification="HEV")
        hybrid_sources = {4: hybrid_source}
        hybrid = current_draft(4, ArchitectureClass.HEV)
        base = effective_states_for_source(hybrid_source)[DomainKind.ENERGY_STORAGE]
        proposal = proposal_from_editor(
            proposal_id="BAT-P01",
            domain=DomainKind.ENERGY_STORAGE,
            based_on=base,
            label="2 kWh",
            requested_changes={"usable_capacity_kwh": 2.0},
        )
        changed = update_selection(hybrid, DomainKind.ENERGY_STORAGE, "BAT-P01")
        current_result = calculate_drafts((hybrid,), sources=hybrid_sources, proposals={})["SYS-CURRENT"].result
        changed_result = calculate_drafts((changed,), sources=hybrid_sources, proposals={"BAT-P01": proposal})["SYS-CURRENT"].result
        self.assertEqual(current_result.effective_outputs, changed_result.effective_outputs)
        self.assertIs(
            changed_result.fidelity_manifest.fidelity_for(DomainKind.ENERGY_STORAGE),
            FidelityLevel.CONFIGURATION_ONLY,
        )

    def test_transmission_configuration_only_is_visible_but_not_quantitative(self):
        base = effective_states_for_source(self.sources[1])[DomainKind.TRANSMISSION_DRIVELINE]
        proposal = proposal_from_editor(
            proposal_id="TRANS-P01",
            domain=DomainKind.TRANSMISSION_DRIVELINE,
            based_on=base,
            label="9AT",
            requested_changes={"gear_count": 9, "final_drive_ratio": 3.45},
        )
        draft = update_selection(self.current, proposal.domain, proposal.identity.proposal_id)
        result = calculate_drafts((draft,), sources=self.sources, proposals={proposal.identity.proposal_id: proposal})["SYS-CURRENT"].result
        self.assertEqual(result.resolved_scenario.resolved_domains[proposal.domain].configuration.gear_count, 9)
        self.assertIs(result.fidelity_manifest.fidelity_for(proposal.domain), FidelityLevel.CONFIGURATION_ONLY)

    def test_explicit_technology_delta_is_effective_assumption(self):
        base = effective_states_for_source(self.sources[1])[DomainKind.TRANSMISSION_DRIVELINE]
        proposal = proposal_from_editor(
            proposal_id="TRANS-P02",
            domain=DomainKind.TRANSMISSION_DRIVELINE,
            based_on=base,
            label="Quantitative transmission",
            technology_deltas=(
                TechDeltaAssumption("Transmission delta", "pse_percent_delta", 0.8, affected_subsystem="transmission"),
            ),
        )
        draft = update_selection(self.current, proposal.domain, proposal.identity.proposal_id)
        result = calculate_drafts((draft,), sources=self.sources, proposals={proposal.identity.proposal_id: proposal})["SYS-CURRENT"].result
        self.assertIs(result.fidelity_manifest.fidelity_for(proposal.domain), FidelityLevel.EFFECTIVE_ASSUMPTION)
        self.assertEqual(result.provenance["technology_deltas"][0]["proposal_id"], "TRANS-P02")

    def test_metadata_incomplete_is_distinct_from_ready_solver(self):
        definition, _ = build_definition(self.current, sources=self.sources, proposals={})
        result = calculate_drafts((self.current,), sources=self.sources, proposals={})["SYS-CURRENT"].result
        missing = metadata_incomplete_fields(definition)
        self.assertIs(result.readiness, SolverReadiness.READY)
        self.assertIn("rated_torque_nm", missing[DomainKind.ENGINE_FUEL_CONVERTER])

    def test_one_not_ready_proposal_does_not_block_ready_current(self):
        bev_source = _source(4, electrification="BEV", eta_pt_est=None, fuel_type=None)
        bev = add_proposal_draft((self.current,), vde_id=4, architecture=ArchitectureClass.BEV)[1]
        calculations = calculate_drafts(
            (self.current, bev), sources={**self.sources, 4: bev_source}, proposals={}
        )
        self.assertIs(calculations["SYS-CURRENT"].readiness, SolverReadiness.READY)
        self.assertIs(calculations["SYS-P1"].readiness, SolverReadiness.NOT_READY)

    def test_calculated_scenario_becomes_stale_after_physical_edit(self):
        calculation = calculate_drafts((self.current,), sources=self.sources, proposals={})["SYS-CURRENT"]
        changed = replace(self.current, vde_id=2)
        self.assertTrue(is_stale(changed, calculation, sources=self.sources, proposals={}))

    def test_recalculation_replaces_same_scenario_identity(self):
        first = calculate_drafts((self.current,), sources=self.sources, proposals={})["SYS-CURRENT"]
        changed = replace(self.current, vde_id=2)
        second = calculate_drafts((changed,), sources=self.sources, proposals={})["SYS-CURRENT"]
        self.assertEqual(first.scenario_id, second.scenario_id)
        self.assertNotEqual(first.fingerprint, second.fingerprint)

    def test_unadopted_recommendation_does_not_change_fingerprint_or_result(self):
        base = effective_states_for_source(self.sources[1])[DomainKind.ENGINE_FUEL_CONVERTER]
        first = proposal_from_editor(
            proposal_id="ENG-P01", domain=base.domain, based_on=base, label="Evidence", recommendation_key="eta_pt_est", recommendation_value=0.4, recommendation_source="ML", adopted=False
        )
        second = proposal_from_editor(
            proposal_id="ENG-P01", domain=base.domain, based_on=base, label="Evidence", recommendation_key="eta_pt_est", recommendation_value=0.5, recommendation_source="Benchmark", adopted=False
        )
        draft = update_selection(self.current, base.domain, "ENG-P01")
        fp1 = calculation_fingerprint(draft, sources=self.sources, proposals={"ENG-P01": first})
        fp2 = calculation_fingerprint(draft, sources=self.sources, proposals={"ENG-P01": second})
        self.assertEqual(fp1, fp2)

    def test_manual_value_cannot_claim_ml_benchmark_or_regression_provenance(self):
        base = effective_states_for_source(self.sources[1])[DomainKind.ENGINE_FUEL_CONVERTER]
        for source in ("ML", "Benchmark", "Regression"):
            with self.assertRaisesRegex(ValueError, "canonical evidence result"):
                proposal_from_editor(
                    proposal_id="ENG-P01",
                    domain=base.domain,
                    based_on=base,
                    label=f"{source} claim",
                    recommendation_key="eta_pt_est",
                    recommendation_value=0.4,
                    recommendation_source=source,
                    adopted=True,
                )

    def test_manual_adoption_uses_assumed_provenance(self):
        base = effective_states_for_source(self.sources[1])[DomainKind.ENGINE_FUEL_CONVERTER]
        proposal = proposal_from_editor(
            proposal_id="ENG-P01",
            domain=base.domain,
            based_on=base,
            label="Engineer assumption",
            recommendation_key="eta_pt_est",
            recommendation_value=0.5,
            recommendation_source="Engineering assumption",
            adopted=True,
        )
        draft = update_selection(self.current, base.domain, "ENG-P01")
        result = calculate_drafts(
            (draft,), sources=self.sources, proposals={"ENG-P01": proposal}
        )["SYS-CURRENT"].result
        self.assertEqual(
            result.provenance["l0_assumptions"][0]["provenance"],
            ProvenanceKind.ASSUMED.value,
        )

    def test_current_correction_is_distinct_from_proposal_and_propagates_to_inheritance(self):
        incomplete = _source(4, eta_pt_est=None)
        source = {4: incomplete}
        current = current_draft(4, ArchitectureClass.ICE)
        engine = effective_states_for_source(incomplete)[DomainKind.ENGINE_FUEL_CONVERTER]
        correction = current_correction_from_editor(
            based_on=engine,
            l0_assumption_key="eta_pt_est",
            l0_assumption_value=0.248,
            reason="Engineering correction",
        )
        corrections = {correction_key(4, engine.domain): correction}
        corrected = effective_states_for_source(incomplete, corrections=corrections)[engine.domain]
        proposal = proposal_from_editor(
            proposal_id="ENG-P01",
            domain=engine.domain,
            based_on=corrected,
            label="Configuration only",
            requested_changes={"displacement_l": 1.6},
        )
        changed = update_selection(current, engine.domain, proposal.identity.proposal_id)

        self.assertEqual(engine.l0_effective_assumption, {})
        self.assertEqual(corrected.l0_effective_assumption["eta_pt_est"], 0.248)
        self.assertIs(proposal.based_on, corrected)
        result = calculate_drafts(
            (changed,),
            sources=source,
            proposals={proposal.identity.proposal_id: proposal},
            corrections=corrections,
        )["SYS-CURRENT"].result
        self.assertIs(result.readiness, SolverReadiness.READY)
        self.assertEqual(result.effective_assumptions["eta_pt_est"], 0.248)
        self.assertEqual(result.provenance["l0_assumptions"][0]["proposal_id"], "CURRENT_CORRECTION")

    def test_utility_factor_correction_is_a_canonical_fraction(self):
        source = _source(
            4,
            electrification="PHEV",
            eta_pt_est=0.3,
            bev_eff_drive=0.85,
            utility_factor=None,
        )
        current = current_draft(4, ArchitectureClass.PHEV)
        controls = effective_states_for_source(source)[DomainKind.ENERGY_MANAGEMENT_CONTROLS]
        correction = current_correction_from_editor(
            based_on=controls,
            l0_assumption_key="utility_factor",
            l0_assumption_value=0.5,
        )
        result = calculate_drafts(
            (current,),
            sources={4: source},
            proposals={},
            corrections={correction_key(4, controls.domain): correction},
        )["SYS-CURRENT"].result
        self.assertEqual(result.effective_assumptions["utility_factor"], 0.5)

    def test_friendly_issue_hides_raw_resolver_code(self):
        message = friendly_issue("bev_eff_drive_missing")
        self.assertNotIn("bev_eff_drive_missing", message)
        self.assertIn("electric-path", message)

    def test_snapshot_adapter_does_not_recalculate_vehicle_demand(self):
        with patch("src.vde_core.system_scenario.legacy_adapter.calculate_vehicle_demand") as calculate:
            effective_states_for_source(self.sources[1])
        calculate.assert_not_called()

    def test_ui_orchestration_delegates_once_per_scenario(self):
        drafts = add_proposal_draft((self.current,), vde_id=2, architecture=ArchitectureClass.ICE)
        with patch(
            "src.vde_app.powertrain_system_scenario_viewmodels.run_system_scenario",
            wraps=__import__("src.vde_core.system_scenario", fromlist=["run_system_scenario"]).run_system_scenario,
        ) as run:
            calculate_drafts(drafts, sources=self.sources, proposals={})
        self.assertEqual(run.call_count, 2)

    def test_explicit_zero_configuration_is_not_missing(self):
        base = effective_states_for_source(self.sources[1])[DomainKind.ENERGY_STORAGE]
        proposal = proposal_from_editor(
            proposal_id="BAT-P00",
            domain=base.domain,
            based_on=base,
            label="Explicit zero",
            requested_changes={"usable_capacity_kwh": 0.0},
        )
        self.assertEqual(proposal.configuration.usable_capacity_kwh, 0.0)
        self.assertIn("usable_capacity_kwh", proposal.requested_changes)

    def test_fuelcons_identity_is_retained_by_current_and_inherited_proposals(self):
        current = current_draft(1, ArchitectureClass.ICE, fuelcons_id=101)
        proposal = add_proposal_draft(
            (current,),
            vde_id=1,
            architecture=ArchitectureClass.ICE,
            fuelcons_id=current.fuelcons_id,
        )[1]
        self.assertEqual(current.fuelcons_id, 101)
        self.assertEqual(proposal.fuelcons_id, 101)

    def test_explainability_distinguishes_adopted_configuration_and_correction(self):
        engine = effective_states_for_source(self.sources[1])[DomainKind.ENGINE_FUEL_CONVERTER]
        transmission = effective_states_for_source(self.sources[1])[DomainKind.TRANSMISSION_DRIVELINE]
        adopted = proposal_from_editor(
            proposal_id="ENG-P01",
            domain=engine.domain,
            based_on=engine,
            label="Adopted",
            recommendation_key="eta_pt_est",
            recommendation_value=0.32,
            adopted=True,
        )
        configuration_only = proposal_from_editor(
            proposal_id="TRANS-P01",
            domain=transmission.domain,
            based_on=transmission,
            label="Configuration",
            requested_changes={"gear_count": 9},
        )
        correction = current_correction_from_editor(
            based_on=effective_states_for_source(self.sources[1])[DomainKind.AUX_THERMAL],
            requested_changes={"ambient_temp_c": 20.0},
        )
        draft = update_selection(self.current, engine.domain, "ENG-P01")
        draft = update_selection(draft, transmission.domain, "TRANS-P01")
        rows = explainability_rows(
            draft,
            sources=self.sources,
            proposals={"ENG-P01": adopted, "TRANS-P01": configuration_only},
            corrections={correction_key(1, correction.domain): correction},
        )
        statuses = {row["domain"]: row["status"] for row in rows}
        self.assertEqual(statuses[engine.domain.value], "Quantitative impact adopted")
        self.assertEqual(statuses[transmission.domain.value], "Configuration only")
        self.assertEqual(statuses[correction.domain.value], "Current correction only")

    def test_sequential_trace_uses_canonical_outputs_only_for_adopted_impacts(self):
        engine = effective_states_for_source(self.sources[1])[DomainKind.ENGINE_FUEL_CONVERTER]
        transmission = effective_states_for_source(self.sources[1])[DomainKind.TRANSMISSION_DRIVELINE]
        first = proposal_from_editor(
            proposal_id="ENG-P01", domain=engine.domain, based_on=engine, label="Engine",
            recommendation_key="eta_pt_est", recommendation_value=0.32, adopted=True,
        )
        second = proposal_from_editor(
            proposal_id="TRANS-P01", domain=transmission.domain, based_on=transmission, label="Transmission",
            technology_deltas=(TechDeltaAssumption("Transmission", "pse_percent_delta", 1.0),),
        )
        draft = update_selection(self.current, engine.domain, "ENG-P01")
        draft = update_selection(draft, transmission.domain, "TRANS-P01")
        trace = sequential_impact_trace(
            draft,
            sources=self.sources,
            proposals={"ENG-P01": first, "TRANS-P01": second},
        )
        self.assertEqual(len(trace), 3)
        self.assertIn("pse", trace[-1]["outputs"])
        current = calculate_drafts((self.current,), sources=self.sources, proposals={})["SYS-CURRENT"].result
        final = calculate_drafts((draft,), sources=self.sources, proposals={"ENG-P01": first, "TRANS-P01": second})["SYS-CURRENT"].result
        self.assertIn("pse", result_deltas(current, final))


if __name__ == "__main__":
    unittest.main()
