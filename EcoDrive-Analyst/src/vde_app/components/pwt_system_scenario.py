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
    friendly_issue,
    is_stale,
    metadata_incomplete_fields,
    proposal_from_editor,
    remove_proposal_draft,
    replace_draft,
    update_selection,
)
from src.vde_core.pwt_fuel_energy_service import (
    derive_reference_pse,
    fetch_fuelcons_by_vde,
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


def _latest_fuel_row(vde_id: int) -> dict[str, Any]:
    rows = fetch_fuelcons_by_vde(vde_id)
    if rows is None or rows.empty:
        return {}
    if "id" in rows.columns:
        rows = rows.sort_values("id", ascending=False)
    row = rows.iloc[0].to_dict()
    architecture = str(row.get("electrification") or "ICE").upper()
    assumption_key = "bev_eff_drive" if architecture == "BEV" else "eta_pt_est"
    if row.get(assumption_key) is None:
        observed = derive_reference_pse(row)
        if observed.get("status") == "available":
            row[assumption_key] = observed.get("value")
    return row


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


def _discover_source_labels() -> dict[int, str]:
    frame = load_baselines_df()
    if frame is None or frame.empty:
        return {}

    labels: dict[int, str] = {}
    for row in frame.to_dict("records"):
        if row.get("id") is None:
            continue
        vde_id = int(row["id"])
        vehicle = f"{row.get('make') or ''} {row.get('model') or ''}".strip()
        year = row.get("year")
        year_text = str(int(year)) if pd.notna(year) else ""
        labels[vde_id] = f"VDE-{vde_id} - {vehicle or 'Snapshot'} {year_text}".strip()
    return labels


def _load_sources(
    current_vde_id: int,
    *,
    drafts: tuple[ScenarioDraft, ...] = (),
) -> tuple[dict[int, ScenarioSource], dict[int, str]]:
    """Load selector labels broadly and resolver sources for the working set."""

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
    sources = {
        vde_id: ScenarioSource(
            vde_id,
            details_by_id.get(vde_id, {"id": vde_id}),
            _latest_fuel_row(vde_id),
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
        st.session_state[_DRAFTS_KEY] = (current_draft(current_vde_id, _architecture_for(source)),)
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


def _legacy_current_vde_id(source_labels: Mapping[int, str]) -> int | None:
    """Read an older page's selected VDE only as an initial visible default."""

    legacy_label = str(st.session_state.get("pwt_active_vde_source") or "")
    if not legacy_label.startswith("#"):
        return None
    raw_id = legacy_label.split(" ", 1)[0].removeprefix("#")
    try:
        vde_id = int(raw_id)
    except ValueError:
        return None
    return vde_id if vde_id in source_labels else None


def _baseline_label(vde_id: int, source_labels: Mapping[int, str]) -> str:
    return source_labels.get(vde_id, f"VDE-{vde_id} · Snapshot")


def _render_current_baseline_selector(
    drafts: tuple[ScenarioDraft, ...],
    source_labels: Mapping[int, str],
) -> tuple[int | None, bool]:
    """Choose the Current source before any scenario composition is rendered."""

    st.subheader("Current Baseline")
    st.caption("Search VDE ID, make, model, or year. Detailed data loads only for active scenarios.")

    available_ids = list(source_labels)
    if not available_ids:
        st.info("No VDE_DB snapshots are available. Create one on VDE Setup to compose a System Scenario.")
        return None, False

    current = _current_draft(drafts)
    current_vde_id = current.vde_id if current is not None else _legacy_current_vde_id(source_labels)
    index = available_ids.index(current_vde_id) if current_vde_id in available_ids else None
    selected_vde_id = st.selectbox(
        "Current baseline",
        available_ids,
        index=index,
        format_func=lambda item: _baseline_label(item, source_labels),
        placeholder="Search a VDE baseline",
        key=_BASELINE_SELECTOR_KEY,
    )

    if selected_vde_id is None:
        st.info("Select a Current baseline to begin a System Scenario.")
        return None, False

    if current is None:
        st.caption(f"Selected: {_baseline_label(selected_vde_id, source_labels)}")
        return int(selected_vde_id), False

    if int(selected_vde_id) == current.vde_id:
        st.caption(f"Selected: {_baseline_label(current.vde_id, source_labels)}")
        return current.vde_id, False

    st.warning(
        "Changing the Current baseline resets domain proposals and calculated "
        "results. Scenario identities remain stable, but every proposal returns "
        "to Inherit from the new Effective Current."
    )
    confirmed = st.button(
        "Apply baseline change and reset scenarios",
        key="pwt_ss:confirm_baseline_change",
        type="primary",
    )
    return (int(selected_vde_id), bool(confirmed))


def _reset_drafts_for_baseline(
    drafts: tuple[ScenarioDraft, ...],
    *,
    vde_id: int,
    architecture: ArchitectureClass,
) -> tuple[ScenarioDraft, ...]:
    inherited_selections = {domain: CURRENT_SELECTION for domain in EDITABLE_DOMAINS}
    return tuple(
        replace(
            draft,
            vde_id=vde_id,
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
    corrections: Mapping[tuple[int, DomainKind], DomainCorrection],
) -> str:
    if draft.identity.role.value == "CURRENT":
        return _configuration_summary(domain, sources.get(draft.vde_id), corrections)
    if domain is DomainKind.VEHICLE_DEMAND:
        return f"VDE-{draft.vde_id}"
    if domain is DomainKind.ARCHITECTURE:
        return draft.architecture.value
    selection = draft.selection_for(domain)
    if selection == CURRENT_SELECTION:
        return "INHERIT"
    if selection == NOT_APPLICABLE_SELECTION:
        return "N/A"
    return selection


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
            _selection_text(draft, domain, sources, corrections) for domain in DomainKind
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
    scenario_id = st.selectbox(
        "Scenario",
        scenario_ids,
        format_func=lambda item: next(d.label for d in drafts if d.identity.scenario_id == item),
        key="pwt_ss:editor:scenario",
    )
    domain = st.selectbox(
        "Domain",
        list(DomainKind),
        format_func=lambda item: _DOMAIN_LABELS[item],
        key="pwt_ss:editor:domain",
    )
    draft = next(item for item in drafts if item.identity.scenario_id == scenario_id)
    key_base = f"pwt_ss:{draft.identity.scenario_id}:{domain.value}"

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
    if applicability is DomainApplicability.NOT_APPLICABLE:
        options = [NOT_APPLICABLE_SELECTION]
    current_selection = draft.selection_for(domain)
    if current_selection not in options:
        current_selection = options[0]
    selection = st.selectbox(
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
    if st.button("Create Domain Proposal", key=f"{key_base}:create"):
        proposal_id = _next_proposal_id(domain, proposals)
        proposal = proposal_from_editor(
            proposal_id=proposal_id,
            domain=domain,
            based_on=based_on,
            label=proposal_id,
        )
        proposals[proposal_id] = proposal
        st.session_state[_PROPOSALS_KEY] = proposals
        st.rerun()

    if selection == CURRENT_SELECTION:
        config = based_on.configuration
        values = {
            name: getattr(config, name)
            for name in _CONFIG_FIELDS.get(domain, ())
        }
        st.markdown("#### Effective Current")
        st.write(values)
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
    st.markdown("##### Configuration")
    st.markdown(f"#### {_DOMAIN_LABELS[domain]} proposal · `{selection}`")
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
        cols = st.columns(4)
        cols[0].metric("Vehicle Demand", f"VDE-{draft.vde_id}")
        cols[1].metric("Architecture", draft.architecture.value)
        cols[2].metric("Fuel [L/100km]", "-" if metrics.get("fuel_l_100km") is None else f"{metrics['fuel_l_100km']:.3f}")
        cols[3].metric("Electric [Wh/km]", "-" if metrics.get("energy_Wh_km") is None else f"{metrics['energy_Wh_km']:.2f}")
        grid_missing = (
            result.resolved_scenario.l0_request_snapshot.powertrain_features.get("grid_gco2_per_kwh")
            is None
        )
        co2_value = metrics.get("gco2_km")
        co2_display = "-" if co2_value is None else f"{co2_value:.2f}"
        if grid_missing and draft.architecture is ArchitectureClass.BEV:
            co2_display = "Not evaluated"
        cols2 = st.columns(3)
        cols2[0].metric("CO₂ [g/km]", co2_display)
        cols2[1].metric("PSE", "-" if metrics.get("pse_value") is None else f"{metrics['pse_value']:.4f}")
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


def render_system_scenario_workspace() -> None:
    """Render one Current + max-three Proposal workspace."""

    drafts = _drafts()
    discovery_labels = _discover_source_labels()
    current_vde_id, baseline_changed = _render_current_baseline_selector(
        drafts,
        discovery_labels,
    )
    if current_vde_id is None:
        return

    sources, source_labels = _load_sources(current_vde_id, drafts=drafts)
    if baseline_changed:
        source = sources.get(
            current_vde_id,
            ScenarioSource(current_vde_id, {"id": current_vde_id}),
        )
        reset_drafts = _reset_drafts_for_baseline(
            drafts,
            vde_id=current_vde_id,
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
    _render_current_readiness(current, sources, proposals, corrections)

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
        width="stretch",
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

    st.markdown("### Domain editor")
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

    st.markdown("### Scenario results")
    for draft in drafts:
        _render_result(
            draft,
            calculations.get(draft.identity.scenario_id),
            sources,
            proposals,
            corrections,
        )


__all__ = ["render_system_scenario_workspace"]
