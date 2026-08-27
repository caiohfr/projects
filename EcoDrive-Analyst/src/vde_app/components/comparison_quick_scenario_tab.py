# src/vde_app/components/comparison_quick_scenario_tab.py
# -----------------------------------------------------------------------------
# Sprint 10E - the Streamlit "Quick Scenarios" tab for the Comparison Report
# page. Every widget here maps onto an EXISTING Quick Scenario contract
# field/enum (src.vde_core.quick_scenario.contracts) -- no free-text physics
# input, no independent Mass/Tire/Aero/PSE math. All calculation happens by
# calling src.vde_core.quick_scenario.comparison_adapter.resolve_quick_slot/
# build_quick_comparison_item, which themselves only ever call the existing
# canonical resolvers (resolve_quick_vehicle_scenario/resolve_quick_efficiency_scenario).
# This file's only responsibilities are: render widgets, hold session-state,
# and merge the resulting ComparisonItems into the existing dataset via
# merge_quick_items_into_dataset -- no parallel chart/analysis subsystem.
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import streamlit as st

from src.vde_app.comparison_report_viewmodels import canonical_identity
from src.vde_core.comparison_report_service import ComparisonDataset
from src.vde_core.pwt_fuel_energy_service import list_benchmark_fuelcons_candidates
from src.vde_core.quick_scenario import (
    ALLOWED_TIRE_TRANSFORMS_BY_SOURCE,
    MAX_QUICK_SCENARIOS_PER_SOURCE,
    EfficiencyQuickInputs,
    MassQuickChange,
    PseProvenance,
    QuickScenario,
    QuickSlotCalculationState,
    ReferencePressureProvenance,
    ScalarChange,
    ScalarChangeMode,
    TechDeltaAssumption,
    TirePressureDelta,
    TireQuickChange,
    TireSource,
    TireTransformMode,
    VehicleQuickOverrides,
    build_quick_comparison_item,
    derive_quick_slot_calculation_state,
    fetch_quick_source_rows_once,
    load_quick_tech_delta_catalog,
    merge_quick_items_into_dataset,
    resolve_quick_slot,
)
from src.vde_core.tire_roadload_service import get_available_tires

_ACTIVE_SOURCE_KEY = "comparison_quick_active_source"
_SCENARIOS_KEY = "comparison_quick_scenarios"
_LAST_CALCULATED_KEY = "comparison_quick_last_calculated"
_RESULTS_KEY = "comparison_quick_results"

_STATE_LABELS = {
    QuickSlotCalculationState.NOT_CALCULATED: "Not calculated",
    QuickSlotCalculationState.READY: "Ready",
    QuickSlotCalculationState.NEEDS_RECALCULATION: "Needs recalculation",
    QuickSlotCalculationState.MISSING_OR_INVALID: "Missing/Invalid",
}

_ACCEPT_PROVENANCE_BY_SUFFIX = {
    "benchmark": PseProvenance.BENCHMARK_ACCEPTED,
    "ml": PseProvenance.ML_RECOMMENDATION_ACCEPTED,
    "techdelta": PseProvenance.TECH_DELTA_ACCEPTED,
}


def _all_dataset_items(dataset: ComparisonDataset) -> list[Any]:
    items = list(dataset.comparisons)
    if dataset.reference is not None:
        items = [dataset.reference] + items
    return items


def _source_options(dataset: ComparisonDataset) -> dict[str, Any]:
    return {canonical_identity(item): item for item in _all_dataset_items(dataset)}


def render_quick_scenario_tab(dataset: ComparisonDataset) -> ComparisonDataset:
    """Renders the Quick Scenarios tab body and returns `dataset` with any
    calculated, Vehicle-ready Quick items merged into `comparisons` -- the
    caller passes the RETURNED dataset into the other four tabs so they see
    Quick items exactly like any other Comparison row.
    """

    st.caption(
        "Define up to 3 temporary Mass / Tire / Aero / Efficiency variants of an "
        "existing scenario above. Nothing here is saved to the database -- Quick "
        "Scenarios exist only for this browser session."
    )

    options = _source_options(dataset)
    if not options:
        st.info("Select a Reference or Compare-with scenario above to build Quick Scenarios from it.")
        return dataset

    identities = list(options.keys())
    active = st.session_state.get(_ACTIVE_SOURCE_KEY)
    if active not in options:
        active = identities[0]
    active = st.selectbox(
        "Build Quick Scenarios from",
        options=identities,
        index=identities.index(active),
        format_func=lambda ident: f"{options[ident].label}  ({ident})",
        key="comparison_quick_source_select",
    )
    st.session_state[_ACTIVE_SOURCE_KEY] = active
    active_source_vde_id = options[active].vde_id

    scenarios_by_source: dict[str, dict[int, QuickScenario]] = st.session_state.setdefault(_SCENARIOS_KEY, {})
    slots = scenarios_by_source.setdefault(active, {})
    last_calculated_by_source: dict[str, dict[int, QuickScenario]] = st.session_state.setdefault(
        _LAST_CALCULATED_KEY, {}
    )
    results_by_source: dict[str, dict[int, tuple]] = st.session_state.setdefault(_RESULTS_KEY, {})
    last_calculated = last_calculated_by_source.setdefault(active, {})
    results = results_by_source.setdefault(active, {})

    add_col, reset_col, _spacer = st.columns([1, 1, 3])
    with add_col:
        can_add = len(slots) < MAX_QUICK_SCENARIOS_PER_SOURCE
        if st.button("+ Add Quick Scenario", disabled=not can_add, key="comparison_quick_add_slot"):
            next_slot = next(s for s in range(1, MAX_QUICK_SCENARIOS_PER_SOURCE + 1) if s not in slots)
            slots[next_slot] = QuickScenario(source_identity=active, slot=next_slot)
            st.rerun()
    with reset_col:
        if st.button("Reset Quick Scenarios", key="comparison_quick_reset"):
            st.session_state[_SCENARIOS_KEY] = {}
            st.session_state[_LAST_CALCULATED_KEY] = {}
            st.session_state[_RESULTS_KEY] = {}
            st.rerun()

    if not slots:
        st.info('No Quick Scenarios yet for this source. Click "+ Add Quick Scenario" to start.')
        return dataset

    removed_slot: int | None = None
    for slot in sorted(slots):
        vehicle_resolution, efficiency_resolution = results.get(slot, (None, None))
        updated, remove = _render_slot_editor(
            active,
            active_source_vde_id,
            slot,
            slots[slot],
            last_calculated.get(slot),
            vehicle_resolution,
            efficiency_resolution,
        )
        if remove:
            removed_slot = slot
        else:
            slots[slot] = updated

    if removed_slot is not None:
        del slots[removed_slot]
        results.pop(removed_slot, None)
        last_calculated.pop(removed_slot, None)
        st.rerun()

    if st.button("Calculate Quick Scenarios", type="primary", key="comparison_quick_calculate"):
        vde_row, fuelcons_row = fetch_quick_source_rows_once(active)
        for slot, scenario in list(slots.items()):
            vehicle_resolution, efficiency_resolution = resolve_quick_slot(
                scenario, source_vde_row=vde_row, source_fuelcons_row=fuelcons_row
            )
            results[slot] = (vehicle_resolution, efficiency_resolution)
            last_calculated[slot] = scenario
        st.rerun()

    quick_items = []
    for source_identity, source_slots in results_by_source.items():
        source_scenarios = scenarios_by_source.get(source_identity, {})
        for slot, (vehicle_resolution, efficiency_resolution) in source_slots.items():
            scenario = source_scenarios.get(slot)
            if scenario is None or vehicle_resolution is None:
                continue
            item = build_quick_comparison_item(scenario, vehicle_resolution, efficiency_resolution)
            if item is not None:
                quick_items.append(item)

    return merge_quick_items_into_dataset(dataset, quick_items)


def _render_slot_editor(
    source_identity: str,
    source_vde_id: int | None,
    slot: int,
    scenario: QuickScenario,
    last_calculated_scenario: QuickScenario | None,
    vehicle_resolution,
    efficiency_resolution,
) -> tuple[QuickScenario, bool]:
    """Renders one slot's editor. Returns (possibly-updated scenario, remove_requested).

    The calculation-state badge is computed from the FRESHLY-rebuilt
    `updated` scenario (after all domain widgets below have been read),
    not from `scenario` as passed in -- computing it from the pre-render
    scenario would show a state that is one render behind whatever the
    user just typed (found by the Sprint 10E closure audit: editing a
    value and rerunning showed the OLD "Ready" badge instead of "Needs
    recalculation" until a second, unrelated interaction). A placeholder
    keeps the badge visually in the header row despite being filled in
    after the domain sections render.
    """

    with st.container(border=True):
        header_col, state_col, remove_col = st.columns([3, 2, 1])
        with header_col:
            label = st.text_input(
                "Label (optional)",
                value=scenario.label or "",
                key=f"comparison_quick_label_{source_identity}_{slot}",
            )
        with state_col:
            state_placeholder = st.empty()
            for issue in (vehicle_resolution.issues if vehicle_resolution is not None else ()):
                st.caption(f"Note: {issue}")
        with remove_col:
            remove = st.button("Remove", key=f"comparison_quick_remove_{source_identity}_{slot}")

        mass_change = _render_mass_section(source_identity, slot, scenario.vehicle_overrides.mass_change)
        cda_change, aero_ref_cda, aero_ref_prov = _render_aero_section(
            source_identity,
            slot,
            scenario.vehicle_overrides.cda_change,
            scenario.vehicle_overrides.aero_reference_cda_m2,
            scenario.vehicle_overrides.aero_reference_cda_provenance,
        )
        tire_change = _render_tire_section(source_identity, slot, scenario.vehicle_overrides.tire_change)

        vehicle_overrides = VehicleQuickOverrides(
            mass_change=mass_change,
            cda_change=cda_change,
            aero_reference_cda_m2=aero_ref_cda,
            aero_reference_cda_provenance=aero_ref_prov,
            tire_change=tire_change,
        )

        efficiency_inputs, final_pse_percent, pse_provenance = _render_efficiency_section(
            source_identity, source_vde_id, slot, scenario, efficiency_resolution
        )

        updated = QuickScenario(
            source_identity=source_identity,
            slot=slot,
            label=label or None,
            vehicle_overrides=vehicle_overrides,
            efficiency_inputs=efficiency_inputs,
            final_pse_percent=final_pse_percent,
            pse_provenance=pse_provenance,
        )
        state = derive_quick_slot_calculation_state(
            updated, last_calculated_scenario, vehicle_resolution, efficiency_resolution
        )
        state_placeholder.caption(f"Slot {slot} -- {_STATE_LABELS[state]}")
        return updated, remove


def _render_mass_section(source_identity: str, slot: int, current: MassQuickChange | None) -> MassQuickChange | None:
    st.markdown("**Mass**")
    key_prefix = f"comparison_quick_mass_{source_identity}_{slot}"
    mode_options = ["No change", "Target curb-to-TWC / WLTP mass line", "TWC Shift (EPA only)"]
    default_index = 0
    if current is not None and current.curb_change is not None:
        default_index = 1
    elif current is not None and current.twc_shift_steps is not None:
        default_index = 2
    mode = st.radio(
        "Mass change",
        mode_options,
        index=default_index,
        horizontal=True,
        key=f"{key_prefix}_mode",
        label_visibility="collapsed",
    )
    if mode == mode_options[0]:
        return None
    if mode == mode_options[1]:
        scalar_mode = st.selectbox(
            "Change type",
            list(ScalarChangeMode),
            format_func=lambda m: m.value,
            key=f"{key_prefix}_scalar_mode",
        )
        value = st.number_input("Value (kg, or %)", value=0.0, key=f"{key_prefix}_scalar_value")
        return MassQuickChange(curb_change=ScalarChange(mode=scalar_mode, value=float(value)))
    steps = st.number_input("TWC shift steps", value=1.0, step=1.0, key=f"{key_prefix}_shift_steps")
    side = st.selectbox("Shift side", ["Up", "Down"], key=f"{key_prefix}_shift_side")
    return MassQuickChange(twc_shift_steps=float(steps), twc_shift_side=side)


def _render_aero_section(
    source_identity: str,
    slot: int,
    current_change: ScalarChange | None,
    current_ref_cda: float | None,
    current_ref_prov: ReferencePressureProvenance | None,
) -> tuple[ScalarChange | None, float | None, ReferencePressureProvenance | None]:
    st.markdown("**Aero (CdA)**")
    key_prefix = f"comparison_quick_aero_{source_identity}_{slot}"
    enabled = st.checkbox("Change CdA", value=current_change is not None, key=f"{key_prefix}_enabled")
    if not enabled:
        return None, None, None

    scalar_mode = st.selectbox(
        "Change type",
        list(ScalarChangeMode),
        index=list(ScalarChangeMode).index(current_change.mode) if current_change else 0,
        format_func=lambda m: m.value,
        key=f"{key_prefix}_mode",
    )
    value = st.number_input(
        "Value (m^2, or %)", value=float(current_change.value) if current_change else 0.0, key=f"{key_prefix}_value"
    )
    change = ScalarChange(mode=scalar_mode, value=float(value))

    ref_cda: float | None = None
    ref_prov: ReferencePressureProvenance | None = None
    if scalar_mode is ScalarChangeMode.ABSOLUTE:
        provide_manual = st.checkbox(
            "Provide manual reference CdA (only used if source CdA is unavailable)",
            value=current_ref_prov is ReferencePressureProvenance.USER_PROVIDED,
            key=f"{key_prefix}_manual_ref_enabled",
        )
        if provide_manual:
            ref_cda = st.number_input(
                "Reference CdA (m^2)",
                value=float(current_ref_cda) if current_ref_cda is not None else 0.0,
                key=f"{key_prefix}_manual_ref_value",
            )
            ref_prov = ReferencePressureProvenance.USER_PROVIDED
    return change, ref_cda, ref_prov


def _render_tire_section(
    source_identity: str, slot: int, current: TireQuickChange | None
) -> TireQuickChange | None:
    st.markdown("**Tire**")
    key_prefix = f"comparison_quick_tire_{source_identity}_{slot}"
    source_options = list(TireSource)
    default_source_index = source_options.index(current.source) if current else 0
    source = st.radio(
        "Tire source",
        source_options,
        index=default_source_index,
        format_func=lambda s: s.value,
        horizontal=True,
        key=f"{key_prefix}_source",
    )

    tire_db_id: int | None = None
    if source is TireSource.TIRE_DB:
        available_tires = get_available_tires({}) or []
        options_by_id = {int(t["id"]): t for t in available_tires if t.get("id") is not None}
        if not options_by_id:
            st.warning("No Tire DB rows are available.")
        else:
            ids = list(options_by_id.keys())
            default_id = current.tire_db_id if current and current.tire_db_id in ids else ids[0]
            tire_db_id = st.selectbox(
                "Tire DB row",
                ids,
                index=ids.index(default_id),
                format_func=lambda tid: str(options_by_id[tid].get("tire_code") or options_by_id[tid].get("code") or tid),
                key=f"{key_prefix}_tire_db_id",
            )

    allowed_modes = sorted(ALLOWED_TIRE_TRANSFORMS_BY_SOURCE[source], key=lambda m: m.value)
    default_mode_index = (
        allowed_modes.index(current.transform_mode) if current and current.transform_mode in allowed_modes else 0
    )
    mode = st.selectbox(
        "Transformation",
        allowed_modes,
        index=default_mode_index,
        format_func=lambda m: m.value,
        key=f"{key_prefix}_transform_mode",
    )

    kwargs: dict[str, Any] = {}
    if mode is TireTransformMode.TARGET_RRC:
        kwargs["target_rrc_n_per_kn"] = st.number_input(
            "Target RRC (N/kN)",
            value=float(current.target_rrc_n_per_kn) if current and current.target_rrc_n_per_kn else 0.0,
            key=f"{key_prefix}_target_rrc",
        )
    elif mode is TireTransformMode.RRC_DELTA:
        kwargs["rrc_delta_n_per_kn"] = st.number_input(
            "RRC Delta (N/kN)",
            value=float(current.rrc_delta_n_per_kn) if current and current.rrc_delta_n_per_kn else 0.0,
            key=f"{key_prefix}_rrc_delta",
        )
    elif mode is TireTransformMode.IMPROVEMENT_PCT:
        kwargs["improvement_pct"] = st.number_input(
            "Improvement (%, positive = lower RR)",
            value=float(current.improvement_pct) if current and current.improvement_pct else 0.0,
            key=f"{key_prefix}_improvement_pct",
        )
    elif mode is TireTransformMode.PRESSURE_DELTA:
        front_delta = st.number_input(
            "Front pressure delta (psi)",
            value=float(current.pressure_delta.front_delta_psi) if current and current.pressure_delta else 0.0,
            key=f"{key_prefix}_pressure_front_delta",
        )
        kwargs["pressure_delta"] = TirePressureDelta(front_delta_psi=float(front_delta))

    return TireQuickChange(source=source, transform_mode=mode, tire_db_id=tire_db_id, **kwargs)


def _render_efficiency_section(
    source_identity: str,
    source_vde_id: int | None,
    slot: int,
    scenario: QuickScenario,
    efficiency_resolution,
) -> tuple[EfficiencyQuickInputs, float | None, PseProvenance | None]:
    st.markdown("**Efficiency (advisory references; Final PSE is the sole calculation authority)**")
    key_prefix = f"comparison_quick_pse_{source_identity}_{slot}"

    benchmark_col, ml_col, techdelta_col = st.columns(3)

    with benchmark_col:
        st.caption("Benchmark PSE")
        candidates = list_benchmark_fuelcons_candidates(source_vde_id) if source_vde_id else []
        candidates_by_identity = {f"fc:{c['id']}": c for c in candidates if c.get("id") is not None}
        benchmark_options = ["(none)"] + list(candidates_by_identity.keys())
        current_benchmark = scenario.efficiency_inputs.benchmark_source_identity
        default_benchmark_index = (
            benchmark_options.index(current_benchmark) if current_benchmark in benchmark_options else 0
        )
        benchmark_choice = st.selectbox(
            "Donor scenario",
            benchmark_options,
            index=default_benchmark_index,
            key=f"{key_prefix}_benchmark_select",
            label_visibility="collapsed",
        )
        benchmark_source_identity = None if benchmark_choice == "(none)" else benchmark_choice
        if efficiency_resolution is not None and efficiency_resolution.benchmark_pse is not None:
            _render_reference_value(efficiency_resolution.benchmark_pse, key_prefix, "benchmark")

    with ml_col:
        st.caption("ML recommendation")
        request_ml = st.checkbox(
            "Request ML recommendation",
            value=scenario.efficiency_inputs.request_ml_recommendation,
            key=f"{key_prefix}_ml_enabled",
        )
        if efficiency_resolution is not None and efficiency_resolution.ml_recommendation is not None:
            _render_reference_value(efficiency_resolution.ml_recommendation, key_prefix, "ml")

    with techdelta_col:
        st.caption("Technology Delta (up to 3 presets)")
        catalog = load_quick_tech_delta_catalog()
        preset_ids = list(catalog.keys())
        current_names = {d.name for d in scenario.efficiency_inputs.technology_deltas}
        default_preset_ids = [pid for pid in preset_ids if catalog[pid].name in current_names]
        selected_preset_ids = st.multiselect(
            "Presets",
            preset_ids,
            default=default_preset_ids,
            format_func=lambda pid: catalog[pid].name,
            key=f"{key_prefix}_techdelta_presets",
            label_visibility="collapsed",
        )[:3]
        technology_deltas = tuple(catalog[pid] for pid in selected_preset_ids)
        if efficiency_resolution is not None and efficiency_resolution.tech_delta_suggestion is not None:
            _render_reference_value(efficiency_resolution.tech_delta_suggestion, key_prefix, "techdelta")

    efficiency_inputs = EfficiencyQuickInputs(
        benchmark_source_identity=benchmark_source_identity,
        request_ml_recommendation=request_ml,
        technology_deltas=technology_deltas,
    )

    if efficiency_resolution is not None and efficiency_resolution.current_pse is not None:
        st.caption(
            f"Current PSE: {efficiency_resolution.current_pse.value_percent:.2f}%"
            if efficiency_resolution.current_pse.is_available
            else "Current PSE: unavailable"
        )

    current_value = scenario.final_pse_percent
    accepted_value = st.session_state.pop(f"{key_prefix}_accept_value", None)
    accepted_provenance = st.session_state.pop(f"{key_prefix}_accept_provenance", None)

    enabled_key = f"{key_prefix}_final_enabled"
    value_key = f"{key_prefix}_final_value"
    if accepted_value is not None:
        current_value = accepted_value
        # A keyed widget only honors value=/index= the FIRST time that key
        # is ever instantiated; on every later rerun Streamlit keeps
        # whatever is already in session_state for that key regardless of
        # what value= is passed. Setting session_state directly BEFORE
        # instantiating the widgets below is the only way an Accept click
        # (a value arriving from a different widget's callback) can
        # actually override an already-existing number_input/checkbox --
        # found by the Sprint 10E closure audit, since without this the
        # Accept button silently failed to update Final PSE after the
        # widgets' first render.
        st.session_state[enabled_key] = True
        st.session_state[value_key] = float(accepted_value)

    manual_enabled = st.checkbox(
        "Set Final PSE for this Quick Scenario",
        value=current_value is not None,
        key=enabled_key,
    )
    if not manual_enabled:
        return efficiency_inputs, None, None

    new_value = st.number_input(
        "Final PSE (%)",
        value=float(current_value) if current_value is not None else 0.0,
        key=value_key,
    )
    if accepted_provenance is not None and accepted_value is not None and float(new_value) == float(accepted_value):
        provenance = accepted_provenance
    elif current_value is not None and float(new_value) == current_value and scenario.pse_provenance is not None:
        provenance = scenario.pse_provenance
    else:
        # Sec 10: a manually-typed/edited value is always USER_PROVIDED, even
        # if it happens to numerically match a value that was once accepted.
        provenance = PseProvenance.USER_PROVIDED
    return efficiency_inputs, float(new_value), provenance


def _render_reference_value(reference, key_prefix: str, suffix: str) -> None:
    if not getattr(reference, "is_available", False):
        st.caption("unavailable")
        return
    st.caption(f"{reference.value_percent:.2f}%")
    if st.button("Accept as Final PSE", key=f"{key_prefix}_accept_{suffix}"):
        st.session_state[f"{key_prefix}_accept_value"] = reference.value_percent
        st.session_state[f"{key_prefix}_accept_provenance"] = _ACCEPT_PROVENANCE_BY_SUFFIX[suffix]
        st.rerun()


__all__ = ["render_quick_scenario_tab"]
