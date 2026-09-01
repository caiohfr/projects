"""Sprint 11D compact multi-domain System Scenario renderer."""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from src.vde_app.powertrain_system_scenario_viewmodels import (
    CURRENT_SELECTION,
    EDITABLE_DOMAINS,
    NOT_APPLICABLE_SELECTION,
    ScenarioCalculation,
    ScenarioDraft,
    ScenarioSource,
    add_proposal_draft,
    build_definition,
    calculate_drafts,
    correction_key,
    current_correction_from_editor,
    current_draft,
    effective_states_for_source,
    compact_impact_rows,
    explainability_rows,
    friendly_issue,
    is_stale,
    metadata_incomplete_fields,
    proposal_from_editor,
    remove_proposal_draft,
    replace_draft,
    result_deltas,
    result_card_viewmodel,
    result_driver,
    sequential_impact_trace,
    update_selection,
    vehicle_demand_comparison,
)
from src.vde_core.pwt_fuel_energy_service import (
    fetch_fuelcons_baselines,
    fetch_fuelcons_row,
    fetch_vde_rows_by_ids,
)
from src.vde_core.system_scenario import (
    ArchitectureClass,
    DomainApplicability,
    DomainCorrection,
    DomainKind,
    DomainProposal,
    FidelityLevel,
    SolverReadiness,
    domain_applicability_for,
    resolve_system_scenario,
)
from src.vde_core.technology_delta import TechDeltaAssumption
from src.vde_core.vde_setup_service import load_baselines_df


_DRAFTS_KEY = "pwt_ss_drafts"
_PROPOSALS_KEY = "pwt_ss_domain_proposals"
_RESULTS_KEY = "pwt_ss_calculations"
_CORRECTIONS_KEY = "pwt_ss_current_corrections"
_BASELINE_SELECTOR_KEY = "pwt_ss:current_baseline"

_DOMAIN_LABELS: Mapping[DomainKind, str] = {
    DomainKind.VEHICLE_DEMAND: "Vehicle Demand",
    DomainKind.ARCHITECTURE: "Architecture",
    DomainKind.ENGINE_FUEL_CONVERTER: "Engine",
    DomainKind.TRANSMISSION_DRIVELINE: "Transmission",
    DomainKind.ELECTRIC_DRIVE: "Electric Drive",
    DomainKind.ENERGY_STORAGE: "Energy Storage",
    DomainKind.ENERGY_MANAGEMENT_CONTROLS: "Controls",
    DomainKind.AUX_THERMAL: "Aux / Thermal",
}

_PROPOSAL_PREFIX: Mapping[DomainKind, str] = {
    DomainKind.ENGINE_FUEL_CONVERTER: "ENG",
    DomainKind.TRANSMISSION_DRIVELINE: "TRANS",
    DomainKind.ELECTRIC_DRIVE: "EM",
    DomainKind.ENERGY_STORAGE: "BAT",
    DomainKind.ENERGY_MANAGEMENT_CONTROLS: "CTRL",
    DomainKind.AUX_THERMAL: "AUX",
}

_CONFIG_FIELDS: Mapping[DomainKind, tuple[str, ...]] = {
    DomainKind.ENGINE_FUEL_CONVERTER: (
        "fuel_type",
        "engine_family_id",
        "displacement_l",
        "rated_power_kw",
        "rated_torque_nm",
    ),
    DomainKind.TRANSMISSION_DRIVELINE: (
        "transmission_type",
        "transmission_model_id",
        "gear_count",
        "final_drive_ratio",
    ),
    DomainKind.ELECTRIC_DRIVE: (
        "motor_role",
        "motor_count",
        "motor_position",
        "rated_power_kw",
        "peak_power_kw",
        "nominal_voltage_v",
    ),
    DomainKind.ENERGY_STORAGE: (
        "gross_capacity_kwh",
        "usable_capacity_kwh",
        "nominal_voltage_v",
        "charge_power_limit_kw",
        "discharge_power_limit_kw",
        "regen_power_limit_kw",
    ),
    DomainKind.ENERGY_MANAGEMENT_CONTROLS: (
        "hybrid_operating_strategy",
        "utility_factor_pct",
        "start_stop_enabled",
        "calibration_notes",
    ),
    DomainKind.AUX_THERMAL: ("ambient_temp_c", "ac_on", "notes"),
}

_ASSUMPTION_OPTIONS: Mapping[DomainKind, tuple[tuple[str, str], ...]] = {
    DomainKind.ENGINE_FUEL_CONVERTER: (
        ("eta_pt_est", "Aggregate fuel-path efficiency"),
    ),
    DomainKind.TRANSMISSION_DRIVELINE: (
        ("eta_pt_est", "Aggregate fuel-path efficiency"),
    ),
    DomainKind.ELECTRIC_DRIVE: (
        ("bev_eff_drive", "Effective electric-path assumption"),
    ),
    DomainKind.ENERGY_MANAGEMENT_CONTROLS: (
        ("utility_factor", "PHEV utility factor"),
    ),
}

_CANONICAL_TECH_DELTA_EFFECT_BASES = (
    "pse_percent_delta",
    "fuel_percent_delta",
)

_FIDELITY_LABELS = {
    FidelityLevel.QUANTITATIVE: "Quantitative",
    FidelityLevel.EFFECTIVE_ASSUMPTION: "Effective assumption",
    FidelityLevel.CONFIGURATION_ONLY: "Configuration only",
    FidelityLevel.NOT_REPRESENTED: "Not represented",
}


def _assumption_options_for(
    domain: DomainKind,
    architecture: ArchitectureClass,
) -> tuple[tuple[str, str], ...]:
    """Return only direct assumptions with an unambiguous L0 path."""

    if architecture is ArchitectureClass.PHEV:
        allowed = {
            DomainKind.ENGINE_FUEL_CONVERTER: {"eta_pt_est"},
            DomainKind.ELECTRIC_DRIVE: {"bev_eff_drive"},
            DomainKind.ENERGY_MANAGEMENT_CONTROLS: {"utility_factor"},
        }.get(domain, set())
        return tuple(item for item in _ASSUMPTION_OPTIONS.get(domain, ()) if item[0] in allowed)
    return _ASSUMPTION_OPTIONS.get(domain, ())


def _assumption_preview(domain: DomainKind, key: str, value: float) -> str:
    label = dict(_ASSUMPTION_OPTIONS.get(domain, ())).get(key, "Effective assumption")
    if key in {"eta_pt_est", "bev_eff_drive", "utility_factor"}:
        return f"{label}: {value * 100:.2f}%"
    return f"{label}: {value:g}"


def _technology_delta_preview(effect_basis: str, effect_value: float) -> str:
    label = {
        "pse_percent_delta": "PSE delta",
        "fuel_percent_delta": "Fuel consumption delta",
    }.get(effect_basis, "Canonical Technology Delta")
    return f"{label}: {effect_value:+g}%"


def _working_set_vde_ids(
    current_vde_id: int,
    drafts: tuple[ScenarioDraft, ...],
) -> tuple[int, ...]:
    """Return detailed source ids for the active System Scenario working set.

    Discovery remains intentionally separate: the VDE selector needs every
    lightweight label, while resolver sources are materialized only for the
    four scenarios that can be active in this workspace.
    """

    working_ids = {int(draft.vde_id) for draft in drafts}
    if not working_ids:
        working_ids.add(int(current_vde_id))
    return tuple(sorted(working_ids))


def _fuelcons_baseline_labels() -> dict[int, str]:
    """Discover persisted baselines without materializing their source objects."""

    frame = fetch_fuelcons_baselines()
    if frame is None or frame.empty:
        return {}
    labels: dict[int, str] = {}
    for row in frame.to_dict("records"):
        fuelcons_id = row.get("fuelcons_id")
        vde_id = row.get("vde_id")
        if fuelcons_id is None or vde_id is None:
            continue
        vehicle = f"{row.get('make') or ''} {row.get('model') or ''}".strip() or "Snapshot"
        year = row.get("year")
        year_text = str(int(year)) if pd.notna(year) else ""
        architecture = row.get("electrification") or "Architecture unavailable"
        labels[int(fuelcons_id)] = (
            f"FuelCons-{int(fuelcons_id)} · VDE-{int(vde_id)} · {vehicle} {year_text} · {architecture}"
        )
    return labels


def _load_sources(
    current_vde_id: int,
    fuelcons_id: int | None,
    *,
    drafts: tuple[ScenarioDraft, ...] = (),
) -> tuple[dict[int, ScenarioSource], dict[int, str]]:
    """Materialize the selected FuelCons baseline and active VDE snapshots only."""

    frame = load_baselines_df()
    rows: list[dict[str, Any]] = []
    if frame is not None and not frame.empty:
        rows.extend(frame.to_dict("records"))
    labels: dict[int, str] = {}
    for row in rows:
        if row.get("id") is None:
            continue
        vde_id = int(row["id"])
        vehicle = f"{row.get('make') or ''} {row.get('model') or ''}".strip()
        labels[vde_id] = f"VDE-{vde_id} · {vehicle or 'Snapshot'}"

    for row in rows:
        if row.get("id") is None:
            continue
        year = row.get("year")
        if pd.notna(year):
            vde_id = int(row["id"])
            labels[vde_id] = f"{labels[vde_id]} · {int(year)}"

    working_ids = _working_set_vde_ids(current_vde_id, drafts)
    detail_frame = fetch_vde_rows_by_ids(working_ids)
    detail_rows = (
        detail_frame.to_dict("records")
        if detail_frame is not None and not detail_frame.empty
        else []
    )
    details_by_id = {
        int(row["id"]): dict(row)
        for row in detail_rows
        if row.get("id") is not None
    }
    fuelcons_row = fetch_fuelcons_row(fuelcons_id) if fuelcons_id is not None else {}
    sources = {
        vde_id: ScenarioSource(
            vde_id,
            details_by_id.get(vde_id, {"id": vde_id}),
            fuelcons_row,
        )
        for vde_id in working_ids
    }
    return sources, labels


def _architecture_for(source: ScenarioSource) -> ArchitectureClass:
    raw = str(source.fuelcons_row.get("electrification") or "ICE").upper()
    return ArchitectureClass(raw) if raw in {item.value for item in ArchitectureClass} else ArchitectureClass.ICE


def _ensure_state(current_vde_id: int, sources: Mapping[int, ScenarioSource]) -> None:
    if _DRAFTS_KEY not in st.session_state:
        source = sources.get(current_vde_id, ScenarioSource(current_vde_id, {"id": current_vde_id}))
        fuelcons_id = source.fuelcons_row.get("id")
        st.session_state[_DRAFTS_KEY] = (
            current_draft(
                current_vde_id,
                _architecture_for(source),
                fuelcons_id=int(fuelcons_id) if fuelcons_id is not None else None,
            ),
        )
        st.session_state[_PROPOSALS_KEY] = {}
        st.session_state[_RESULTS_KEY] = {}
        st.session_state[_CORRECTIONS_KEY] = {}
        st.session_state[_CORRECTIONS_KEY] = {}
    st.session_state.setdefault(_CORRECTIONS_KEY, {})


def _drafts() -> tuple[ScenarioDraft, ...]:
    return tuple(st.session_state.get(_DRAFTS_KEY) or ())


def _proposals() -> dict[str, DomainProposal]:
    return dict(st.session_state.get(_PROPOSALS_KEY) or {})


def _calculations() -> dict[str, ScenarioCalculation]:
    return dict(st.session_state.get(_RESULTS_KEY) or {})


def _corrections() -> dict[tuple[int, DomainKind], DomainCorrection]:
    return dict(st.session_state.get(_CORRECTIONS_KEY) or {})


def _current_draft(drafts: tuple[ScenarioDraft, ...]) -> ScenarioDraft | None:
    return next(
        (
            draft
            for draft in drafts
            if draft.identity.role.value == "CURRENT"
        ),
        None,
    )


def _render_current_baseline_selector(
    drafts: tuple[ScenarioDraft, ...],
    baseline_labels: Mapping[int, str],
) -> tuple[int | None, int | None, bool]:
    """Choose one persisted FuelCons row before scenario composition."""

    st.subheader("Source Baseline")
    st.caption(
        "Select a persisted FuelCons record to establish Current. Search by "
        "FuelCons ID, VDE ID, make, model, year, or architecture."
    )

    available_ids = list(baseline_labels)
    if not available_ids:
        st.info("No persisted FuelCons rows are available. Create a FuelCons row linked to a VDE before composing a System Scenario.")
        return None, None, False

    current = _current_draft(drafts)
    current_fuelcons_id = current.fuelcons_id if current is not None else None
    index = available_ids.index(current_fuelcons_id) if current_fuelcons_id in available_ids else 0
    selected_fuelcons_id = st.selectbox(
        "Source Baseline",
        available_ids,
        index=index,
        format_func=lambda item: baseline_labels[item],
        placeholder="Search persisted FuelCons records",
        key=_BASELINE_SELECTOR_KEY,
    )

    selected_row = fetch_fuelcons_row(int(selected_fuelcons_id))
    selected_vde_id = selected_row.get("vde_id")
    if selected_vde_id is None:
        st.error("The selected FuelCons baseline is no longer linked to a VDE.")
        return None, None, False
    if current is None or int(selected_fuelcons_id) == current.fuelcons_id:
        st.caption(f"Selected: {baseline_labels[int(selected_fuelcons_id)]}")
        return int(selected_fuelcons_id), int(selected_vde_id), False

    st.warning(
        "Changing the Source Baseline resets domain proposals and calculated "
        "results. Scenario identities remain stable, but every proposal returns "
        "to Inherit from the new Effective Current."
    )
    confirmed = st.button(
        "Apply baseline change and reset scenarios",
        key="pwt_ss:confirm_baseline_change",
        type="primary",
    )
    return int(selected_fuelcons_id), int(selected_vde_id), bool(confirmed)


def _reset_drafts_for_baseline(
    drafts: tuple[ScenarioDraft, ...],
    *,
    vde_id: int,
    fuelcons_id: int,
    architecture: ArchitectureClass,
) -> tuple[ScenarioDraft, ...]:
    inherited_selections = {domain: CURRENT_SELECTION for domain in EDITABLE_DOMAINS}
    return tuple(
        replace(
            draft,
            vde_id=vde_id,
            fuelcons_id=fuelcons_id,
            architecture=architecture,
            selections=inherited_selections,
        )
        for draft in drafts
    )


def _current_readiness(
    draft: ScenarioDraft,
    sources: Mapping[int, ScenarioSource],
    proposals: Mapping[str, DomainProposal],
    corrections: Mapping[tuple[int, DomainKind], DomainCorrection],
):
    definition, request = build_definition(
        draft,
        sources=sources,
        proposals=proposals,
        corrections=corrections,
    )
    return resolve_system_scenario(definition, request_template=request)


def _render_current_readiness(
    draft: ScenarioDraft,
    sources: Mapping[int, ScenarioSource],
    proposals: Mapping[str, DomainProposal],
    corrections: Mapping[tuple[int, DomainKind], DomainCorrection],
) -> None:
    st.markdown("### L0 Input Readiness")
    resolved = _current_readiness(draft, sources, proposals, corrections)
    source = sources.get(draft.vde_id)
    if source is None or len(source.vde_row) <= 1:
        st.error("Selected Current baseline could not be materialized. Choose another VDE snapshot.")
        return

    observed_architecture = str(source.fuelcons_row.get("electrification") or "").upper()
    if observed_architecture not in {item.value for item in ArchitectureClass}:
        st.warning("Architecture: Assumed ICE — confirm or change it in the Architecture domain editor.")
    else:
        st.caption(f"Architecture: {draft.architecture.value}")

    vehicle_demand = effective_states_for_source(
        source,
        corrections=corrections,
    )[DomainKind.VEHICLE_DEMAND]
    result = vehicle_demand.configuration.vehicle_demand_result
    total = result.total_summary.vde_mj_per_km if result is not None else None
    if total is None:
        st.caption("Vehicle Demand: Not defined")
    else:
        st.caption(f"Vehicle Demand: {total:.4f} MJ/km TOTAL")

    powertrain = resolved.l0_request_snapshot.powertrain_features
    l0_summary = {
        "fuel_type": powertrain.get("fuel_type") or "Not provided",
        "eta_pt_est": powertrain.get("eta_pt_est", "Not provided"),
        "bev_eff_drive": powertrain.get("bev_eff_drive", "Not provided"),
        "utility_factor": powertrain.get("utility_factor", "Not provided"),
        "grid_gco2_per_kwh": powertrain.get("grid_gco2_per_kwh", "Not provided"),
    }
    st.caption("Canonical L0 assumptions consumed by the current request")
    st.json(l0_summary, expanded=False)
    if l0_summary["grid_gco2_per_kwh"] == "Not provided":
        st.caption("Grid carbon factor: Not provided. A zero grid CO2 result is not displayed as a known zero here.")

    if resolved.solver_readiness is SolverReadiness.READY:
        st.success("L0 readiness: READY")
    else:
        st.warning("L0 readiness: NOT READY")
        for issue in resolved.issues:
            st.caption(f"• {friendly_issue(issue)}")


def _display_value(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return "Not provided"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _assumption_availability(
    value: float | None,
    *,
    applicable: bool,
    digits: int,
) -> str:
    if not applicable:
        return "Not applicable"
    if value is None:
        return "Not provided"
    return f"{value * 100:.{digits}f}%"


def _render_baseline_summary(
    draft: ScenarioDraft,
    source: ScenarioSource | None,
    sources: Mapping[int, ScenarioSource],
    proposals: Mapping[str, DomainProposal],
    corrections: Mapping[tuple[int, DomainKind], DomainCorrection],
) -> None:
    st.markdown("### Effective Current")
    if source is None:
        st.error("The selected FuelCons baseline could not be materialized.")
        return
    row = source.fuelcons_row
    resolved = _current_readiness(draft, sources, proposals, corrections)
    demand_state = effective_states_for_source(source, corrections=corrections)[DomainKind.VEHICLE_DEMAND]
    demand_result = demand_state.configuration.vehicle_demand_result
    basis = "NET" if str(row.get("energy_basis") or "").upper() == "VDE_NET" else "TOTAL"
    demand_summary = (
        demand_result.net_summary
        if demand_result is not None and basis == "NET"
        else demand_result.total_summary if demand_result is not None else None
    )
    demand = demand_summary.vde_mj_per_km if demand_summary is not None else None
    vehicle = f"{source.vde_row.get('make') or ''} {source.vde_row.get('model') or ''}".strip()

    baseline_card, demand_card, readiness_card = st.columns(3)
    with baseline_card.container(border=True):
        st.caption("FUELCONS")
        st.metric("FuelCons", f"FC-{row.get('id', draft.fuelcons_id or '—')}")
        st.write(vehicle or "Vehicle not identified")
        st.caption(f"{row.get('fuel_type') or 'Fuel not provided'} · {draft.architecture.value}")
    with demand_card.container(border=True):
        st.caption("VEHICLE DEMAND")
        st.metric("Linked VDE", f"VDE-{source.vde_id}")
        st.write("Not evaluated" if demand is None else f"{demand:.4f} MJ/km")
        st.caption(f"{basis} basis")
    with readiness_card.container(border=True):
        st.caption("L0 READINESS")
        readiness = resolved.solver_readiness.value
        st.metric("Canonical solver", readiness)
        st.write(draft.architecture.value)
        st.caption("Assumptions ready" if readiness == "READY" else "Input attention required")

    powertrain = resolved.l0_request_snapshot.powertrain_features
    fuel_path_applicable = draft.architecture is not ArchitectureClass.BEV
    electric_path_applicable = draft.architecture in {
        ArchitectureClass.PHEV,
        ArchitectureClass.BEV,
    }
    utility_factor_applicable = draft.architecture is ArchitectureClass.PHEV
    assumption_values = (
        (
            "Aggregate fuel-path efficiency",
            _assumption_availability(
                powertrain.get("eta_pt_est"),
                applicable=fuel_path_applicable,
                digits=2,
            ),
        ),
        (
            "Electric-path efficiency",
            _assumption_availability(
                powertrain.get("bev_eff_drive"),
                applicable=electric_path_applicable,
                digits=2,
            ),
        ),
        (
            "Utility factor",
            _assumption_availability(
                powertrain.get("utility_factor"),
                applicable=utility_factor_applicable,
                digits=1,
            ),
        ),
    )
    correction_count = sum(1 for key in corrections if key[0] == draft.vde_id)
    assumptions = st.columns(4)
    for column, (label, value) in zip(
        assumptions,
        (*assumption_values, ("Current corrections", str(correction_count))),
    ):
        with column.container(border=True):
            st.caption(label)
            st.write(value)
    if correction_count:
        st.caption(
            f"Effective Current includes {correction_count} correction"
            f"{'s' if correction_count != 1 else ''} over the Source Baseline."
        )
    else:
        st.caption("Source Baseline = Effective Current · no Current Corrections applied.")
    if resolved.solver_readiness is SolverReadiness.NOT_READY:
        with st.expander("Readiness issues", expanded=False):
            for issue in resolved.issues:
                st.warning(friendly_issue(issue))
    with st.expander("Technical baseline details", expanded=False):
        st.write(
            {
                "fuelcons_id": row.get("id"),
                "linked_vde_id": source.vde_id,
                "energy_basis": basis,
                "canonical_l0_assumptions": dict(powertrain),
            }
        )


def _render_vde_impact_only(
    draft: ScenarioDraft,
    sources: Mapping[int, ScenarioSource],
    corrections: Mapping[tuple[int, DomainKind], DomainCorrection],
) -> None:
    st.markdown("### VDE impact only")
    st.caption("Demand-side reference only. It is not a second powertrain solver.")
    source = sources.get(draft.vde_id)
    if source is None:
        st.info("Linked Vehicle Demand is unavailable.")
        return
    state = effective_states_for_source(source, corrections=corrections)[DomainKind.VEHICLE_DEMAND]
    result = state.configuration.vehicle_demand_result
    if result is None:
        st.warning("Vehicle Demand: Not evaluated.")
        return
    columns = st.columns(3)
    columns[0].metric("Linked VDE", f"VDE-{draft.vde_id}")
    columns[1].metric("TOTAL demand [MJ/km]", _display_value(result.total_summary.vde_mj_per_km))
    net_summary = result.net_summary
    net_demand = net_summary.vde_mj_per_km if net_summary is not None else None
    columns[2].metric(
        "NET demand [MJ/km]",
        "Not evaluated" if net_demand is None else _display_value(net_demand),
    )


def _scenario_status(
    draft: ScenarioDraft,
    calculation: ScenarioCalculation | None,
    sources: Mapping[int, ScenarioSource],
    proposals: Mapping[str, DomainProposal],
    corrections: Mapping[tuple[int, DomainKind], DomainCorrection],
) -> str:
    if calculation is None:
        return "Not calculated"
    if is_stale(
        draft,
        calculation,
        sources=sources,
        proposals=proposals,
        corrections=corrections,
    ):
        return "Needs recalculation"
    if calculation.programming_error:
        return "Cannot calculate L0"
    return "READY" if calculation.readiness is SolverReadiness.READY else "NOT READY"


def _configuration_summary(
    domain: DomainKind,
    source: ScenarioSource | None,
    corrections: Mapping[tuple[int, DomainKind], DomainCorrection],
) -> str:
    if source is None or len(source.vde_row) <= 1:
        return "Unavailable"

    state = effective_states_for_source(source, corrections=corrections)[domain]
    config = state.configuration
    if domain is DomainKind.VEHICLE_DEMAND:
        result = config.vehicle_demand_result
        total = result.total_summary.vde_mj_per_km if result is not None else None
        return f"VDE-{source.vde_id} · {total:.4f} MJ/km" if total is not None else f"VDE-{source.vde_id} · Incomplete"
    if domain is DomainKind.ARCHITECTURE:
        return config.architecture_class.value if config.architecture_class is not None else "Not defined"
    if domain is DomainKind.ENGINE_FUEL_CONVERTER:
        values = []
        if config.displacement_l is not None:
            values.append(f"{config.displacement_l:g} L")
        if config.rated_power_kw is not None:
            values.append(f"{config.rated_power_kw:g} kW")
        return " · ".join(values) if values else "Incomplete"
    if domain is DomainKind.TRANSMISSION_DRIVELINE:
        values = [config.transmission_type] if config.transmission_type else []
        if config.gear_count is not None:
            values.append(f"{config.gear_count} spd")
        if config.final_drive_ratio is not None:
            values.append(f"FDR {config.final_drive_ratio:g}")
        return " · ".join(values) if values else "Incomplete"
    if domain is DomainKind.ELECTRIC_DRIVE:
        return "N/A" if state.provenance.value == "NOT_AVAILABLE" else "Incomplete"
    if domain is DomainKind.ENERGY_STORAGE:
        value = config.usable_capacity_kwh
        if value is None:
            value = config.gross_capacity_kwh
        return f"{value:g} kWh" if value is not None else "N/A"
    if domain is DomainKind.ENERGY_MANAGEMENT_CONTROLS:
        return "Incomplete" if config.utility_factor_pct is None else f"UF {config.utility_factor_pct:g}%"
    if domain is DomainKind.AUX_THERMAL:
        return "Incomplete" if config.ambient_temp_c is None else f"{config.ambient_temp_c:g} °C"
    return "Incomplete"


def _selection_text(
    draft: ScenarioDraft,
    domain: DomainKind,
    sources: Mapping[int, ScenarioSource],
    proposals: Mapping[str, DomainProposal],
    corrections: Mapping[tuple[int, DomainKind], DomainCorrection],
) -> str:
    if draft.identity.role.value == "CURRENT":
        summary = _configuration_summary(domain, sources.get(draft.vde_id), corrections)
        corrected = correction_key(draft.vde_id, domain) in corrections
        return f"{summary} · CORRECTED" if corrected else summary
    if domain is DomainKind.VEHICLE_DEMAND:
        return f"VDE-{draft.vde_id}"
    if domain is DomainKind.ARCHITECTURE:
        return draft.architecture.value
    selection = draft.selection_for(domain)
    if selection == CURRENT_SELECTION:
        return "INHERIT"
    if selection == NOT_APPLICABLE_SELECTION:
        return "N/A"
    proposal = proposals.get(selection)
    if proposal is None:
        return f"{selection} · NOT REPRESENTED"
    if proposal.l0_effective_assumption or proposal.technology_deltas:
        status = "QUANTITATIVE"
    elif proposal.requested_changes:
        status = "CONFIG ONLY"
    else:
        status = "NOT REPRESENTED"
    return f"{proposal.label or selection} · {status}"


def _render_matrix(
    drafts: tuple[ScenarioDraft, ...],
    calculations: Mapping[str, ScenarioCalculation],
    sources: Mapping[int, ScenarioSource],
    proposals: Mapping[str, DomainProposal],
    corrections: Mapping[tuple[int, DomainKind], DomainCorrection],
) -> None:
    matrix: dict[str, list[str]] = {"Domain": [_DOMAIN_LABELS[domain] for domain in DomainKind] + ["Status"]}
    for draft in drafts:
        matrix[draft.label] = [
            _selection_text(draft, domain, sources, proposals, corrections) for domain in DomainKind
        ] + [
            _scenario_status(
                draft,
                calculations.get(draft.identity.scenario_id),
                sources,
                proposals,
                corrections,
            )
        ]
    st.dataframe(pd.DataFrame(matrix), hide_index=True, width="stretch")


def _optional_value_widget(*, key: str, label: str, value: Any) -> Any:
    if isinstance(value, bool):
        return st.checkbox(label.replace("_", " ").title(), value=value, key=key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return st.number_input(label.replace("_", " ").title(), value=value, key=key)
    text = st.text_input(
        label.replace("_", " ").title(),
        value="" if value is None else str(value),
        key=key,
    )
    return text if text != "" else None


def _coerce_config_value(current: Any, raw: Any, annotation: Any) -> Any:
    if raw is None:
        return None
    annotation_text = str(annotation)
    if isinstance(current, bool) or "bool" in annotation_text:
        if isinstance(raw, str):
            normalized = raw.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off"}:
                return False
            raise ValueError(f"Invalid boolean value: {raw!r}")
        return bool(raw)
    if isinstance(current, int) or "int" in annotation_text:
        return int(raw)
    if isinstance(current, float) or "float" in annotation_text:
        return float(raw)
    return raw


def _next_proposal_id(domain: DomainKind, proposals: Mapping[str, DomainProposal]) -> str:
    prefix = _PROPOSAL_PREFIX[domain]
    used = {proposal.identity.proposal_id for proposal in proposals.values() if proposal.domain is domain}
    index = 1
    while f"{prefix}-P{index:02d}" in used:
        index += 1
    return f"{prefix}-P{index:02d}"


def _sanitize_architecture_selections(draft: ScenarioDraft) -> ScenarioDraft:
    selections = dict(draft.selections)
    for domain in EDITABLE_DOMAINS:
        applicability = domain_applicability_for(draft.architecture, domain)
        if applicability is DomainApplicability.NOT_APPLICABLE:
            selections[domain] = NOT_APPLICABLE_SELECTION
        elif selections.get(domain) == NOT_APPLICABLE_SELECTION:
            selections[domain] = CURRENT_SELECTION
    return replace(draft, selections=selections)


def _render_scenario_identity_editor(drafts: tuple[ScenarioDraft, ...]) -> tuple[ScenarioDraft, ...]:
    updated = drafts
    for draft in drafts:
        key = f"pwt_ss:{draft.identity.scenario_id}:label"
        label = st.text_input(
            f"{draft.identity.scenario_id} label",
            value=draft.label,
            key=key,
            label_visibility="collapsed",
        )
        if label != draft.label:
            updated = replace_draft(updated, replace(draft, label=label or draft.label))
    return updated


def _render_domain_editor(
    drafts: tuple[ScenarioDraft, ...],
    sources: Mapping[int, ScenarioSource],
    source_labels: Mapping[int, str],
    proposals: dict[str, DomainProposal],
    corrections: dict[tuple[int, DomainKind], DomainCorrection],
) -> tuple[
    tuple[ScenarioDraft, ...],
    dict[str, DomainProposal],
    dict[tuple[int, DomainKind], DomainCorrection],
]:
    scenario_ids = [draft.identity.scenario_id for draft in drafts]
    scenario_context, domain_context = st.columns(2)
    scenario_id = scenario_context.selectbox(
        "Scenario",
        scenario_ids,
        format_func=lambda item: next(d.label for d in drafts if d.identity.scenario_id == item),
        key="pwt_ss:editor:scenario",
    )
    domain = domain_context.selectbox(
        "Domain",
        list(DomainKind),
        format_func=lambda item: _DOMAIN_LABELS[item],
        key="pwt_ss:editor:domain",
    )
    draft = next(item for item in drafts if item.identity.scenario_id == scenario_id)
    key_base = f"pwt_ss:{draft.identity.scenario_id}:{domain.value}"
    context_label = (
        "EFFECTIVE CURRENT"
        if draft.identity.role.value == "CURRENT"
        else draft.label.upper()
    )
    st.markdown(f"#### {context_label} · {_DOMAIN_LABELS[domain].upper()}")

    if domain is DomainKind.VEHICLE_DEMAND:
        available_ids = list(source_labels)
        selected_id = st.selectbox(
            "Vehicle Demand snapshot",
            available_ids,
            index=available_ids.index(draft.vde_id) if draft.vde_id in available_ids else 0,
            format_func=lambda item: source_labels[item],
            key=f"{key_base}:vde_id",
        )
        if selected_id != draft.vde_id:
            drafts = replace_draft(drafts, replace(draft, vde_id=int(selected_id)))
            st.session_state[_DRAFTS_KEY] = drafts
            st.rerun()
        st.caption("Uses the persisted canonical VDE snapshot. No Mass, Tire, Aero or roadload calculation occurs here.")
        return drafts, proposals, corrections

    if domain is DomainKind.ARCHITECTURE:
        architecture = st.selectbox(
            "Architecture classification",
            list(ArchitectureClass),
            index=list(ArchitectureClass).index(draft.architecture),
            format_func=lambda item: item.value,
            key=f"{key_base}:architecture",
        )
        if architecture is not draft.architecture:
            updated = _sanitize_architecture_selections(replace(draft, architecture=architecture))
            drafts = replace_draft(drafts, updated)
        st.caption("Classification only: ICE, MHEV, HEV, PHEV or BEV. No topology graph is inferred.")
        return drafts, proposals, corrections

    applicability = domain_applicability_for(draft.architecture, domain)
    st.caption(f"Architecture applicability: {applicability.value.replace('_', ' ').title()}")
    domain_proposals = [
        proposal for proposal in proposals.values() if proposal.domain is domain
    ]
    options = [CURRENT_SELECTION, *[proposal.identity.proposal_id for proposal in domain_proposals]]
    if draft.identity.role.value == "CURRENT":
        options = [CURRENT_SELECTION]
    elif applicability is DomainApplicability.NOT_APPLICABLE:
        options = [NOT_APPLICABLE_SELECTION]
    current_selection = draft.selection_for(domain)
    if current_selection not in options:
        current_selection = options[0]
    proposal_context, create_context = st.columns([3, 1])
    selection = proposal_context.selectbox(
        "Selection",
        options,
        index=options.index(current_selection),
        format_func=lambda item: {
            CURRENT_SELECTION: "Effective Current",
            NOT_APPLICABLE_SELECTION: "N/A",
        }.get(item, item),
        key=f"{key_base}:selection",
    )
    if selection != draft.selection_for(domain):
        updated = update_selection(draft, domain, selection)
        drafts = replace_draft(drafts, updated)
        draft = updated

    if applicability is DomainApplicability.NOT_APPLICABLE:
        st.info(f"{_DOMAIN_LABELS[domain]} is not applicable to {draft.architecture.value}.")
        return drafts, proposals, corrections

    source = sources.get(draft.vde_id)
    if source is None or len(source.vde_row) <= 1:
        st.error(
            "The selected Vehicle Demand source is unavailable for this "
            "scenario. Select a materialized VDE baseline before editing "
            "internal domains."
        )
        return drafts, proposals, corrections
    based_on = effective_states_for_source(source, corrections=corrections)[domain]
    can_create = draft.identity.role.value == "PROPOSAL"
    if create_context.button(
        "+ Create Proposal",
        key=f"{key_base}:create",
        disabled=not can_create,
        width="stretch",
    ):
        proposal_id = _next_proposal_id(domain, proposals)
        proposal = proposal_from_editor(
            proposal_id=proposal_id,
            domain=domain,
            based_on=based_on,
            label=proposal_id,
        )
        proposals[proposal_id] = proposal
        drafts = replace_draft(
            drafts,
            update_selection(draft, domain, proposal_id),
        )
        st.session_state[_PROPOSALS_KEY] = proposals
        st.session_state[_DRAFTS_KEY] = drafts
        st.rerun()

    if selection == CURRENT_SELECTION:
        if draft.identity.role.value == "PROPOSAL":
            with st.container(border=True):
                st.caption("INHERIT")
                st.write(
                    f"{draft.label} uses Effective Current for "
                    f"{_DOMAIN_LABELS[domain]}. Create or select a Domain Proposal "
                    "to define a deviation."
                )
            return drafts, proposals, corrections
        config = based_on.configuration
        values = {
            name: getattr(config, name)
            for name in _CONFIG_FIELDS.get(domain, ())
        }
        st.markdown("#### Effective Current")
        visible_values = [(name, value) for name, value in values.items() if value is not None]
        if visible_values:
            value_columns = st.columns(min(3, len(visible_values)))
            for index, (name, value) in enumerate(visible_values):
                value_columns[index % len(value_columns)].metric(
                    name.replace("_", " ").title(),
                    value,
                )
        if all(value is None for value in values.values()) and values:
            st.info("Configuration unavailable / sparse. Existing L0 assumptions are displayed separately from physical configuration.")
        correction = corrections.get(correction_key(source.vde_id, domain))
        with st.expander("Current correction", expanded=correction is not None):
            st.caption(
                "Correct source/current information without creating a Domain Proposal. "
                "The correction becomes this source's Effective Current."
            )
            correction_changes: dict[str, Any] = {}
            config_type = type(config)
            annotations = {field.name: field.type for field in dataclasses.fields(config_type)}
            for field_name in _CONFIG_FIELDS.get(domain, ()):
                source_value = getattr(based_on.source.configuration, field_name)
                current_value = getattr(config, field_name)
                raw = _optional_value_widget(
                    key=f"{key_base}:correction:{field_name}",
                    label=f"Corrected {field_name}",
                    value=current_value,
                )
                value = _coerce_config_value(current_value, raw, annotations[field_name])
                if value != source_value:
                    correction_changes[field_name] = value

            correction_options = _assumption_options_for(domain, draft.architecture)
            correction_assumption_key: str | None = None
            correction_assumption_value: float | None = None
            if correction_options:
                correction_keys = ["(none)", *[item[0] for item in correction_options]]
                existing_key = next(
                    (
                        key
                        for key in (correction.l0_effective_assumption if correction else {})
                        if key in correction_keys
                    ),
                    "(none)",
                )
                correction_assumption_key = st.selectbox(
                    "Correct canonical L0 assumption",
                    correction_keys,
                    index=correction_keys.index(existing_key),
                    format_func=lambda item: (
                        "No L0 correction"
                        if item == "(none)"
                        else dict(correction_options)[item]
                    ),
                    key=f"{key_base}:correction:assumption_key",
                )
                if correction_assumption_key == "(none)":
                    correction_assumption_key = None
                else:
                    prior = (
                        correction.l0_effective_assumption.get(correction_assumption_key)
                        if correction is not None
                        else None
                    )
                    if correction_assumption_key == "utility_factor":
                        correction_assumption_value = st.number_input(
                            "Utility factor (%)",
                            min_value=0.0,
                            max_value=100.0,
                            value=float((prior if prior is not None else 0.0) * 100.0),
                            key=f"{key_base}:correction:assumption_value",
                        ) / 100.0
                    else:
                        correction_assumption_value = st.number_input(
                            "Corrected canonical value",
                            value=float(prior if prior is not None else 0.0),
                            format="%.6f",
                            key=f"{key_base}:correction:assumption_value",
                        )
            correction_reason = st.text_input(
                "Correction evidence/reference note",
                value=correction.reason if correction is not None else "",
                key=f"{key_base}:correction:reason",
            )
            if st.button("Apply Current correction", key=f"{key_base}:correction:apply"):
                key = correction_key(source.vde_id, domain)
                if correction_changes or correction_assumption_key:
                    corrections[key] = current_correction_from_editor(
                        based_on=based_on,
                        requested_changes=correction_changes,
                        l0_assumption_key=correction_assumption_key,
                        l0_assumption_value=correction_assumption_value,
                        reason=correction_reason,
                    )
                else:
                    corrections.pop(key, None)
                st.session_state[_CORRECTIONS_KEY] = corrections
                st.session_state[_RESULTS_KEY] = {}
                st.rerun()
        st.caption("No Domain Proposal selected; calculation uses Effective Current.")
        return drafts, proposals, corrections

    proposal = proposals[selection]
    configuration_context, l0_context = st.columns(2)
    with configuration_context.container(border=True):
        st.markdown("##### A · Configuration")
        st.caption("What physically changed?")
        changed_fields = [
            name
            for name in _CONFIG_FIELDS.get(domain, ())
            if getattr(proposal.configuration, name) != getattr(based_on.configuration, name)
        ]
        if changed_fields:
            for field_name in changed_fields:
                before = getattr(based_on.configuration, field_name)
                after = getattr(proposal.configuration, field_name)
                st.write(f"{field_name.replace('_', ' ').title()}: {before} → {after}")
        else:
            st.caption("No physical configuration change.")
    with l0_context.container(border=True):
        st.markdown("##### B · L0 Representation")
        st.caption("What quantitative effect are we representing?")
        if proposal.l0_effective_assumption:
            for key, value in proposal.l0_effective_assumption.items():
                st.write(f"{_assumption_preview(domain, key, value)} · ADOPTED")
        elif proposal.technology_deltas:
            for delta in proposal.technology_deltas:
                st.write(
                    f"{_technology_delta_preview(delta.effect_basis, delta.effect_value)} "
                    "· ADOPTED"
                )
        elif proposal.requested_changes:
            st.write("CONFIGURATION ONLY")
            st.caption("Quantitative effect at Energy Balance L0: NOT REPRESENTED")
        else:
            st.caption("No quantitative L0 impact adopted.")
    st.caption(f"Edit {_DOMAIN_LABELS[domain]} proposal `{selection}` below.")
    st.markdown("##### Configuration inputs")
    label = st.text_input(
        "Proposal label",
        value=proposal.label or selection,
        key=f"pwt_ss:proposal:{selection}:label",
    )
    requested_changes: dict[str, Any] = {}
    config_type = type(based_on.configuration)
    annotations = {field.name: field.type for field in dataclasses.fields(config_type)}
    columns = st.columns(2)
    for index, field_name in enumerate(_CONFIG_FIELDS.get(domain, ())):
        current_value = getattr(based_on.configuration, field_name)
        proposed_value = getattr(proposal.configuration, field_name)
        with columns[index % 2]:
            st.caption(f"Current: {current_value if current_value is not None else 'Not provided'}")
            raw = _optional_value_widget(
                key=f"pwt_ss:proposal:{selection}:{field_name}",
                label=field_name,
                value=proposed_value,
            )
        value = _coerce_config_value(current_value, raw, annotations[field_name])
        if value != current_value:
            requested_changes[field_name] = value

    assumption_options = _assumption_options_for(domain, draft.architecture)
    adopted = False
    assumption_key: str | None = None
    recommendation_value: float | None = None
    evidence_reference = ""
    if assumption_options:
        st.markdown("##### L0 representation")
        st.caption(
            "A manual value is an Engineering assumption. Benchmark, ML and Regression "
            "recommendations are unavailable here until a canonical evidence owner supplies one."
        )
        assumption_keys = [item[0] for item in assumption_options]
        existing_assumption_key = next(
            (key for key in proposal.l0_effective_assumption if key in assumption_keys),
            assumption_keys[0],
        )
        assumption_key = st.selectbox(
            "Canonical L0 assumption",
            assumption_keys,
            index=assumption_keys.index(existing_assumption_key),
            format_func=lambda item: dict(assumption_options)[item],
            key=f"pwt_ss:proposal:{selection}:assumption_key",
        )
        prior = proposal.l0_effective_assumption.get(assumption_key)
        if assumption_key == "utility_factor":
            recommendation_value = st.number_input(
                "Utility factor (%)",
                min_value=0.0,
                max_value=100.0,
                value=float((prior if prior is not None else 0.0) * 100.0),
                key=f"pwt_ss:proposal:{selection}:assumption_value",
            ) / 100.0
        else:
            recommendation_value = st.number_input(
                "Manual engineering-assumption value",
                value=float(prior if prior is not None else 0.0),
                format="%.6f",
                key=f"pwt_ss:proposal:{selection}:assumption_value",
            )
        evidence_reference = st.text_input(
            "Engineering evidence/reference note",
            key=f"pwt_ss:proposal:{selection}:evidence_reference",
        )
        adopted = st.checkbox(
            "Adopt this value into the deterministic L0 scenario",
            value=assumption_key in proposal.l0_effective_assumption,
            key=f"pwt_ss:proposal:{selection}:adopt",
        )
        if not adopted:
            st.caption("Recommendation only — deterministic result is unchanged until explicit adoption.")

    st.markdown("#### Technology Delta representation")
    prior_delta = proposal.technology_deltas[0] if proposal.technology_deltas else None
    use_delta = False
    if draft.architecture is ArchitectureClass.PHEV:
        st.info(
            "Generic Technology Delta is unavailable for PHEV because the current contract "
            "does not assign it to one thermal or electric path."
        )
    else:
        use_delta = st.checkbox(
            "Associate an active canonical Technology Delta",
            value=prior_delta is not None,
            key=f"pwt_ss:proposal:{selection}:delta_enabled",
        )
    deltas: tuple[TechDeltaAssumption, ...] = ()
    if use_delta:
        delta_basis = st.selectbox(
            "Effect basis",
            _CANONICAL_TECH_DELTA_EFFECT_BASES,
            index=(
                _CANONICAL_TECH_DELTA_EFFECT_BASES.index(prior_delta.effect_basis)
                if prior_delta and prior_delta.effect_basis in _CANONICAL_TECH_DELTA_EFFECT_BASES
                else 0
            ),
            key=f"pwt_ss:proposal:{selection}:delta_basis",
        )
        delta_value = st.number_input(
            "Effect value",
            value=float(prior_delta.effect_value if prior_delta else 0.0),
            key=f"pwt_ss:proposal:{selection}:delta_value",
        )
        delta_source = st.selectbox(
            "Delta source",
            [
                "engineering_assumption",
                "supplier_data",
                "imported_map",
                "simulation_result",
                "test_data",
            ],
            key=f"pwt_ss:proposal:{selection}:delta_source",
        )
        deltas = (
            TechDeltaAssumption(
                name=f"{selection} L0 representation",
                effect_basis=delta_basis,
                effect_value=float(delta_value),
                affected_subsystem=_DOMAIN_LABELS[domain],
                source_type=delta_source,
                confidence="engineering",
            ),
        )
        st.caption("Stacking is performed by the canonical Technology Delta owner, never by this UI.")

    rebuilt = proposal_from_editor(
        proposal_id=selection,
        domain=domain,
        based_on=based_on,
        label=label,
        requested_changes=requested_changes,
        recommendation_key=assumption_key,
        recommendation_value=recommendation_value,
        evidence_reference=evidence_reference,
        adopted=adopted,
        technology_deltas=deltas,
    )
    proposals[selection] = rebuilt
    changed = bool(requested_changes)
    represented = bool(rebuilt.l0_effective_assumption or rebuilt.technology_deltas)
    if represented:
        st.success("L0 quantitative representation: explicitly adopted.")
    elif changed:
        st.info("Configuration changed · L0 quantitative impact: Not represented.")
    else:
        st.caption("No configuration or quantitative change from Effective Current.")
    return drafts, proposals, corrections


def _render_result(
    draft: ScenarioDraft,
    calculation: ScenarioCalculation | None,
    current_calculation: ScenarioCalculation | None,
    sources: Mapping[int, ScenarioSource],
    proposals: Mapping[str, DomainProposal],
    corrections: Mapping[tuple[int, DomainKind], DomainCorrection],
) -> None:
    status = _scenario_status(draft, calculation, sources, proposals, corrections)
    with st.container(border=True):
        st.markdown(f"#### {draft.label}")
        st.caption(f"`{draft.identity.scenario_id}` · {status} · Energy Balance L0")
        if calculation is None:
            st.info("Not calculated yet.")
            return
        if status == "Needs recalculation":
            st.warning("Needs recalculation — visible inputs changed after the last result.")
            return
        if calculation.programming_error:
            st.error("This scenario could not be calculated because its draft is invalid.")
            with st.expander("Technical error", expanded=False):
                st.code(calculation.programming_error)
            return
        result = calculation.result
        if result is None:
            return
        metrics = result.effective_outputs
        current_result = current_calculation.result if current_calculation is not None else None
        deltas = result_deltas(current_result, result)
        cols = st.columns(4)
        cols[0].metric("Vehicle Demand", f"VDE-{draft.vde_id}")
        cols[1].metric("Architecture", draft.architecture.value)
        fuel = metrics.get("fuel_l_100km")
        energy = metrics.get("energy_Wh_km")
        fuel_delta = f"{deltas['fuel_l_100km']:+.3f}" if "fuel_l_100km" in deltas else None
        energy_delta = f"{deltas['energy_Wh_km']:+.2f}" if "energy_Wh_km" in deltas else None
        cols[2].metric("Fuel [L/100km]", "Not evaluated" if fuel is None else f"{fuel:.3f}", None if draft.identity.role.value == "CURRENT" else fuel_delta)
        cols[3].metric("Electric [Wh/km]", "Not evaluated" if energy is None else f"{energy:.2f}", None if draft.identity.role.value == "CURRENT" else energy_delta)
        grid_missing = (
            result.resolved_scenario.l0_request_snapshot.powertrain_features.get("grid_gco2_per_kwh")
            is None
        )
        co2_value = metrics.get("gco2_km")
        co2_display = "-" if co2_value is None else f"{co2_value:.2f}"
        if grid_missing and draft.architecture is ArchitectureClass.BEV:
            co2_display = "Not evaluated"
        cols2 = st.columns(3)
        co2_delta = f"{deltas['gco2_km']:+.2f}" if "gco2_km" in deltas else None
        pse = metrics.get("pse")
        pse_delta = f"{deltas['pse'] * 100:+.2f} pp" if "pse" in deltas else None
        cols2[0].metric("CO₂ [g/km]", co2_display, None if draft.identity.role.value == "CURRENT" else co2_delta)
        cols2[1].metric("PSE [%]", "Not evaluated" if pse is None else f"{pse * 100:.2f}%", None if draft.identity.role.value == "CURRENT" else pse_delta)
        cols2[2].metric("Solver", result.solver_identity or "-")
        if grid_missing and draft.architecture is ArchitectureClass.PHEV:
            st.caption("Grid carbon factor is not provided; displayed CO₂ is the canonical fuel-path result only.")

        fidelity_rows = [
            {
                "Domain": _DOMAIN_LABELS[domain],
                "Fidelity": _FIDELITY_LABELS[result.fidelity_manifest.fidelity_for(domain)],
            }
            for domain in DomainKind
        ]
        st.dataframe(pd.DataFrame(fidelity_rows), hide_index=True, width="stretch")
        definition = result.resolved_scenario
        metadata_missing = metadata_incomplete_fields(
            build_definition(
                draft,
                sources=sources,
                proposals=proposals,
                corrections=corrections,
            )[0]
        )
        if metadata_missing:
            st.caption(
                "Metadata incomplete (does not necessarily block L0): "
                + "; ".join(
                    f"{_DOMAIN_LABELS[domain]}: {', '.join(fields)}"
                    for domain, fields in metadata_missing.items()
                )
            )
        if result.readiness is SolverReadiness.NOT_READY:
            for issue in definition.issues:
                st.warning(friendly_issue(issue))
        with st.expander("Technical trace", expanded=False):
            st.write(
                {
                    "readiness": result.readiness.value,
                    "structured_issues": definition.issues,
                    "effective_assumptions": dict(result.effective_assumptions),
                    "provenance": dict(result.provenance),
                }
            )


def _format_metric(value: float | None, *, digits: int, suffix: str) -> str:
    return "Not evaluated" if value is None else f"{value:.{digits}f}{suffix}"


def _compact_metric(value: float | None, *, digits: int) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _metric_delta(
    value: float | None,
    *,
    digits: int,
    scale: float = 1.0,
    suffix: str = "",
) -> str | None:
    if value is None:
        return None
    return f"{value * scale:+.{digits}f}{suffix}"


def _render_result_card(
    draft: ScenarioDraft,
    calculation: ScenarioCalculation | None,
    current_calculation: ScenarioCalculation | None,
    sources: Mapping[int, ScenarioSource],
    proposals: Mapping[str, DomainProposal],
    corrections: Mapping[tuple[int, DomainKind], DomainCorrection],
) -> None:
    status = _scenario_status(draft, calculation, sources, proposals, corrections)
    card = result_card_viewmodel(draft, calculation, current=current_calculation)
    with st.container(border=True):
        title = "CURRENT — EFFECTIVE" if draft.identity.role.value == "CURRENT" else card.label.upper()
        st.markdown(f"#### {title}")
        st.caption(status)
        if status == "Needs recalculation":
            st.warning("Needs recalculation — visible inputs changed.")
        fuel_delta = card.deltas.get("fuel_l_100km")
        pse_delta = card.deltas.get("pse")
        co2_delta = card.deltas.get("gco2_km")
        energy_delta = card.deltas.get("energy_Wh_km")
        fuel, pse, co2, electric = st.columns(4)
        fuel.metric(
            "Fuel",
            _compact_metric(card.fuel_l_100km, digits=3),
            _metric_delta(fuel_delta, digits=3),
            delta_color="off",
        )
        fuel.caption("L/100 km")
        pse.metric(
            "PSE",
            "—" if card.pse is None else f"{card.pse * 100:.2f}%",
            _metric_delta(pse_delta, digits=2, scale=100.0, suffix=" pp"),
            delta_color="off",
        )
        co2.metric(
            "CO₂",
            _compact_metric(card.gco2_km, digits=1),
            _metric_delta(co2_delta, digits=1),
            delta_color="off",
        )
        co2.caption("g/km")
        electric.metric(
            "Electric",
            _compact_metric(card.energy_wh_km, digits=1),
            _metric_delta(energy_delta, digits=1),
            delta_color="off",
        )
        electric.caption("Wh/km")


def _render_result_summary(
    drafts: tuple[ScenarioDraft, ...],
    calculations: Mapping[str, ScenarioCalculation],
    sources: Mapping[int, ScenarioSource],
    proposals: Mapping[str, DomainProposal],
    corrections: Mapping[tuple[int, DomainKind], DomainCorrection],
) -> None:
    st.markdown("### Scenario Results")
    st.caption("Proposals compare directly with Effective Current. Deltas use neutral styling.")
    columns = st.columns(len(drafts))
    current_calculation = calculations.get("SYS-CURRENT")
    for column, draft in zip(columns, drafts):
        with column:
            _render_result_card(
                draft,
                calculations.get(draft.identity.scenario_id),
                current_calculation,
                sources,
                proposals,
                corrections,
            )


def _driver_narrative(driver: str) -> str:
    return {
        "DEMAND-DRIVEN": (
            "Fuel consumption changed because Vehicle Demand changed. "
            "No represented powertrain-efficiency improvement contributed."
        ),
        "POWERTRAIN-DRIVEN": (
            "Vehicle Demand remained unchanged. The scenario result changed "
            "through adopted L0 powertrain assumptions."
        ),
        "MIXED DEMAND + POWERTRAIN": (
            "Vehicle Demand and adopted system-level L0 representations both "
            "changed. No additive causal allocation is inferred."
        ),
        "NO QUANTITATIVE CHANGE": (
            "Vehicle Demand remained unchanged, no adopted powertrain impact is "
            "active, and the final result remained unchanged."
        ),
    }[driver]


def _driver_label(driver: str) -> str:
    return {
        "DEMAND-DRIVEN": "Demand-driven",
        "POWERTRAIN-DRIVEN": "Powertrain-driven",
        "MIXED DEMAND + POWERTRAIN": "Mixed demand + powertrain",
        "NO QUANTITATIVE CHANGE": "No quantitative change",
    }[driver]


def _render_result_drivers(
    drafts: tuple[ScenarioDraft, ...],
    calculations: Mapping[str, ScenarioCalculation],
    sources: Mapping[int, ScenarioSource],
    proposals: Mapping[str, DomainProposal],
    corrections: Mapping[tuple[int, DomainKind], DomainCorrection],
) -> None:
    st.markdown("### Why did it change?")
    current = _current_draft(drafts)
    current_calculation = calculations.get("SYS-CURRENT")
    if current is None:
        return
    proposal_drafts = [draft for draft in drafts if draft.identity.role.value == "PROPOSAL"]
    if not proposal_drafts:
        st.info("Add a proposal to expose its demand → powertrain → final-result story.")
        return
    for draft in proposal_drafts:
        calculation = calculations.get(draft.identity.scenario_id)
        with st.container(border=True):
            st.markdown(f"#### Why did {draft.label} change?")
            if (
                calculation is None
                or calculation.result is None
                or current_calculation is None
                or _scenario_status(draft, calculation, sources, proposals, corrections) != "READY"
            ):
                st.info("Calculate the working set to resolve this result story.")
                continue
            driver = result_driver(
                current,
                draft,
                sources=sources,
                proposals=proposals,
            )
            st.caption(f"RESULT DRIVER · {_driver_label(driver).upper()}")

            demand = vehicle_demand_comparison(current, draft, sources=sources)
            current_card = result_card_viewmodel(current, current_calculation)
            proposal_card = result_card_viewmodel(draft, calculation, current=current_calculation)
            demand_column, powertrain_column, final_column = st.columns(3)
            with demand_column:
                st.markdown("##### 1 · Vehicle Demand")
                st.caption("CHANGED" if demand.changed else "UNCHANGED")
                st.write(
                    f"{_format_metric(demand.current_mj_per_km, digits=4, suffix=' MJ/km')} → "
                    f"{_format_metric(demand.proposal_mj_per_km, digits=4, suffix=' MJ/km')}"
                )
                if demand.delta_percent is not None:
                    st.metric(
                        "Demand delta",
                        f"{demand.delta_percent:+.2f}%",
                        delta_color="off",
                    )
                if not demand.changed:
                    st.write("Vehicle Demand remained unchanged.")
                elif (
                    demand.current_mj_per_km is not None
                    and demand.proposal_mj_per_km is not None
                    and demand.proposal_mj_per_km < demand.current_mj_per_km
                ):
                    st.write("Vehicle Demand decreased.")
                else:
                    st.write("Vehicle Demand increased.")
                st.caption(f"Canonical {demand.basis} Vehicle Demand")
            with powertrain_column:
                st.markdown("##### 2 · Powertrain / PSE")
                pse_delta = proposal_card.deltas.get("pse")
                impact_rows = compact_impact_rows(draft, proposals=proposals)
                adopted = [row for row in impact_rows if row["status"] == "ADOPTED"]
                st.caption("CHANGED" if adopted else "UNCHANGED")
                st.write(
                    f"{_format_metric(None if current_card.pse is None else current_card.pse * 100, digits=2, suffix='%')} → "
                    f"{_format_metric(None if proposal_card.pse is None else proposal_card.pse * 100, digits=2, suffix='%')}"
                )
                st.metric(
                    "PSE delta",
                    "—" if pse_delta is None else f"{pse_delta * 100:+.2f} pp",
                    delta_color="off",
                )
                if not adopted:
                    st.write("No adopted L0 powertrain impact.")
                    if pse_delta == 0:
                        st.write("PSE remained unchanged.")
                elif pse_delta == 0:
                    st.write("Adopted impacts are active; net PSE remained unchanged.")
            with final_column:
                st.markdown("##### 3 · Final result")
                st.write(
                    f"Fuel: {_format_metric(current_card.fuel_l_100km, digits=3, suffix='')} → "
                    f"{_format_metric(proposal_card.fuel_l_100km, digits=3, suffix='')} L/100 km"
                )
                fuel_delta = proposal_card.deltas.get("fuel_l_100km")
                if fuel_delta is not None:
                    st.caption(f"Δ {fuel_delta:+.3f} L/100 km")
                st.write(
                    f"Energy: {_format_metric(current_card.energy_wh_km, digits=2, suffix='')} → "
                    f"{_format_metric(proposal_card.energy_wh_km, digits=2, suffix='')} Wh/km"
                )
                st.write(
                    f"CO₂: {_format_metric(current_card.gco2_km, digits=2, suffix='')} → "
                    f"{_format_metric(proposal_card.gco2_km, digits=2, suffix='')} g/km"
                )
                co2_delta = proposal_card.deltas.get("gco2_km")
                if co2_delta is not None:
                    st.caption(f"Δ {co2_delta:+.1f} g/km")

            st.markdown("##### Interpretation")
            st.write(_driver_narrative(driver))

            impact_rows = compact_impact_rows(draft, proposals=proposals)
            if impact_rows:
                st.markdown("##### Adopted L0 impacts")
                if any(row["status"] == "CONFIG ONLY" for row in impact_rows):
                    st.info(
                        "CONFIGURATION ONLY · Physical configuration changed. "
                        "Quantitative effect at Energy Balance L0: NOT REPRESENTED."
                    )
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Domain": _DOMAIN_LABELS[DomainKind(row["domain"])],
                                "L0 representation": row["representation"],
                                "Evidence": row["evidence"],
                                "Status": row["status"],
                            }
                            for row in impact_rows
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                )
            trace = sequential_impact_trace(
                draft,
                sources=sources,
                proposals=proposals,
                corrections=corrections,
            )
            if len(trace) > 1:
                with st.expander("L0 scenario composition", expanded=False):
                    st.caption("Sequential canonical scenario composition; not a subsystem energy decomposition.")
                    for step in trace:
                        pse = step["outputs"].get("pse")
                        st.write(
                            f"{step['label']}: "
                            f"{_format_metric(None if pse is None else pse * 100, digits=2, suffix='%')}"
                        )
            with st.expander("Technical details", expanded=False):
                st.write(
                    {
                        "scenario_id": draft.identity.scenario_id,
                        "structured_issues": calculation.result.resolved_scenario.issues,
                        "effective_assumptions": dict(calculation.result.effective_assumptions),
                        "provenance": dict(calculation.result.provenance),
                    }
                )


def _render_explainability(
    drafts: tuple[ScenarioDraft, ...],
    calculations: Mapping[str, ScenarioCalculation],
    sources: Mapping[int, ScenarioSource],
    proposals: Mapping[str, DomainProposal],
    corrections: Mapping[tuple[int, DomainKind], DomainCorrection],
) -> None:
    st.markdown("### Why did it change?")
    st.caption("This is a trace of adopted L0 scenario composition, not a physical subsystem decomposition.")
    proposal_drafts = [draft for draft in drafts if draft.identity.role.value == "PROPOSAL"]
    if not proposal_drafts:
        st.info("Add a proposal to compare adopted L0 impacts with Effective Current.")
        return
    for draft in proposal_drafts:
        calculation = calculations.get(draft.identity.scenario_id)
        if calculation is None or calculation.result is None or _scenario_status(
            draft, calculation, sources, proposals, corrections
        ) != "READY":
            st.caption(f"{draft.label}: calculate a current result to show its trace.")
            continue
        st.markdown(f"#### {draft.label}")
        rows = explainability_rows(
            draft,
            sources=sources,
            proposals=proposals,
            corrections=corrections,
        )
        display_rows = [
            {
                "Domain": _DOMAIN_LABELS[DomainKind(row["domain"])],
                "Config change": row["config_change"],
                "Quantitative representation": row["representation"],
                "Provenance": row["provenance"],
                "Status": row["status"],
                "Effect on PSE": "Shown in sequential trace" if row["status"] == "Quantitative impact adopted" else "—",
                "Effect on Fuel": "Shown in sequential trace" if row["status"] == "Quantitative impact adopted" else "—",
            }
            for row in rows
        ]
        st.dataframe(pd.DataFrame(display_rows), hide_index=True, width="stretch")
        trace = sequential_impact_trace(
            draft,
            sources=sources,
            proposals=proposals,
            corrections=corrections,
        )
        trace_rows = []
        for step in trace:
            outputs = step["outputs"]
            pse = outputs.get("pse")
            fuel = outputs.get("fuel_l_100km")
            trace_rows.append(
                {
                    "Composition step": step["label"],
                    "PSE": "Not evaluated" if pse is None else f"{pse * 100:.2f}%",
                    "Fuel [L/100km]": "Not evaluated" if fuel is None else f"{fuel:.3f}",
                }
            )
        st.dataframe(pd.DataFrame(trace_rows), hide_index=True, width="stretch")


def render_system_scenario_workspace() -> None:
    """Render one Current + max-three Proposal workspace."""

    drafts = _drafts()
    baseline_labels = _fuelcons_baseline_labels()
    fuelcons_id, current_vde_id, baseline_changed = _render_current_baseline_selector(
        drafts,
        baseline_labels,
    )
    if fuelcons_id is None or current_vde_id is None:
        return

    sources, source_labels = _load_sources(current_vde_id, fuelcons_id, drafts=drafts)
    if baseline_changed:
        source = sources.get(
            current_vde_id,
            ScenarioSource(current_vde_id, {"id": current_vde_id}),
        )
        reset_drafts = _reset_drafts_for_baseline(
            drafts,
            vde_id=current_vde_id,
            fuelcons_id=fuelcons_id,
            architecture=_architecture_for(source),
        )
        st.session_state[_DRAFTS_KEY] = reset_drafts
        st.session_state[_PROPOSALS_KEY] = {}
        st.session_state[_RESULTS_KEY] = {}
        st.rerun()

    _ensure_state(current_vde_id, sources)
    drafts = _drafts()
    proposals = _proposals()
    corrections = _corrections()
    calculations = _calculations()
    current = _current_draft(drafts)
    if current is None:
        st.error("Current scenario state is unavailable. Refresh the workspace.")
        return

    st.session_state["current_vde_id"] = current.vde_id
    _render_baseline_summary(
        current,
        sources.get(current.vde_id),
        sources,
        proposals,
        corrections,
    )
    _render_result_summary(drafts, calculations, sources, proposals, corrections)

    st.subheader("Multi-domain System Scenarios")
    st.caption(
        "Each column is an independent complete scenario. The matrix shows composition; edit one domain at a time below."
    )

    action1, action2 = st.columns(2)
    if action1.button(
        "Add Proposal",
        disabled=len(drafts) >= 4,
        key="pwt_ss:add_proposal",
        width="stretch",
    ):
        drafts = add_proposal_draft(
            drafts,
            vde_id=current.vde_id,
            architecture=current.architecture,
            fuelcons_id=current.fuelcons_id,
        )
        st.session_state[_DRAFTS_KEY] = drafts
        st.rerun()
    removable = [draft for draft in drafts if draft.identity.role.value == "PROPOSAL"]
    remove_id = action2.selectbox(
        "Remove",
        [draft.identity.scenario_id for draft in removable] or ["None"],
        format_func=lambda item: next((d.label for d in removable if d.identity.scenario_id == item), item),
        key="pwt_ss:remove_select",
        label_visibility="collapsed",
        disabled=not removable,
    )
    if removable and action2.button("Remove Proposal", key="pwt_ss:remove", width="stretch"):
        drafts = remove_proposal_draft(drafts, remove_id)
        calculations.pop(remove_id, None)
        st.session_state[_DRAFTS_KEY] = drafts
        st.session_state[_RESULTS_KEY] = calculations
        st.rerun()
    drafts = _render_scenario_identity_editor(drafts)
    st.session_state[_DRAFTS_KEY] = drafts
    _render_matrix(drafts, calculations, sources, proposals, corrections)

    if st.button(
        "Calculate System Scenarios",
        key="pwt_ss:calculate",
        type="primary",
    ):
        calculations = dict(
            calculate_drafts(
                drafts,
                sources=sources,
                proposals=proposals,
                corrections=corrections,
            )
        )
        st.session_state[_RESULTS_KEY] = calculations
        st.rerun()

    st.markdown("### Domain Workspace")
    st.caption("Choose one scenario and domain, then separate physical configuration from its optional L0 representation.")
    drafts, proposals, corrections = _render_domain_editor(
        drafts,
        sources,
        source_labels,
        proposals,
        corrections,
    )
    st.session_state[_DRAFTS_KEY] = drafts
    st.session_state[_PROPOSALS_KEY] = proposals
    st.session_state[_CORRECTIONS_KEY] = corrections

    if st.button(
        "Calculate after editing",
        key="pwt_ss:calculate_after_editor",
        type="primary",
    ):
        calculations = dict(
            calculate_drafts(
                drafts,
                sources=sources,
                proposals=proposals,
                corrections=corrections,
            )
        )
        st.session_state[_RESULTS_KEY] = calculations
        st.rerun()

    _render_result_drivers(drafts, calculations, sources, proposals, corrections)


__all__ = ["render_system_scenario_workspace"]
