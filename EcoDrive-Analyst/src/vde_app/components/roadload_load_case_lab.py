"""Streamlit presentation for the temporary Roadload & Load Case Lab."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.vde_app.units import KPA_PER_PSI, UNIT_SYSTEM_METRIC, normalize_unit_system
from src.vde_core.repositories.fuelcons_repository import fetch_fuelcons_by_vde_id
from src.vde_core.repositories.vde_repository import fetch_vde_all_rows, fetch_vde_by_id
from src.vde_core.vehicle_demand.load_case import (
    FixedSpeedSpec,
    LoadCaseAnalysisMode,
    LoadCaseEnvironment,
    LoadCaseMass,
    LoadCaseMassMode,
    LoadCaseRoute,
    LoadCaseTrailer,
    RoadloadLoadCaseRequest,
    SpeedSweepSpec,
    calculate_roadload_load_case,
)


LAB_VDE_SESSION_KEY = "roadload_load_case_vde_id"
LAB_RESULT_SESSION_KEY = "roadload_load_case_result"

_MODE_LABELS = {
    "Drive Cycle": LoadCaseAnalysisMode.DRIVE_CYCLE,
    "Fixed Speed": LoadCaseAnalysisMode.FIXED_SPEED,
    "Speed Sweep": LoadCaseAnalysisMode.SPEED_SWEEP,
}

_MASS_LABELS = {
    "Inherit baseline": LoadCaseMassMode.INHERIT,
    "Payload delta": LoadCaseMassMode.PAYLOAD_DELTA,
    "Target payload": LoadCaseMassMode.TARGET_PAYLOAD,
    "Custom vehicle mass": LoadCaseMassMode.CUSTOM,
    "GVWR": LoadCaseMassMode.GVWR,
    "GCWR / towing": LoadCaseMassMode.GCWR,
}

_STATUS_LABELS = {
    "NOT_USED": "Not used",
    "NOT_REQUESTED": "Not requested",
    "CALCULATED": "Calculated",
    "APPLIED": "Applied",
    "UNAVAILABLE": "Unavailable",
    "SOURCE": "Source",
    "INHERITED": "Inherited",
    "INHERITED_BASELINE": "Inherited baseline",
    "RESOLVED": "Resolved",
    "REVIEW": "Review",
    "COMBINED_GCWR": "Combined GCWR",
    "PAYLOAD_DELTA": "Payload delta",
    "TARGET_PAYLOAD": "Target payload",
    "CUSTOM": "Custom vehicle mass",
    "GVWR": "GVWR",
    "GCWR": "GCWR",
    "EPA_INERTIA_CLASS": "EPA inertia class",
    "PHYSICAL_TEST_MASS": "Physical test mass",
}


def _row_label(row: dict) -> str:
    identity = " ".join(
        str(row.get(key) or "").strip() for key in ("make", "model", "year") if row.get(key) not in (None, "")
    ).strip()
    return f"VDE-{int(row['id']):03d} · {identity or 'Unnamed VDE'}"


def _optional_number(container, label: str, *, key: str, value: float | None = None, **kwargs):
    enabled = container.checkbox(f"Override {label}", key=f"{key}__enabled", value=False)
    if not enabled:
        return None
    return container.number_input(label, key=key, value=value, **kwargs)


def _humanize(value) -> str:
    raw = str(value or "").strip()
    return _STATUS_LABELS.get(raw, raw.replace("_", " ").strip().title() or "Unavailable")


def _pressure_unit() -> str:
    return "kPa" if normalize_unit_system(None) == UNIT_SYSTEM_METRIC else "psi"


def _pressure_widget_key(base_key: str) -> str:
    return f"{base_key}__{_pressure_unit().lower()}"


def _pressure_to_kpa(value) -> float | None:
    if value is None:
        return None
    return float(value) if _pressure_unit() == "kPa" else float(value) * KPA_PER_PSI


def _optional_pressure(container, label: str, *, key: str, default_kpa: float):
    unit = _pressure_unit()
    display_default = default_kpa if unit == "kPa" else default_kpa / KPA_PER_PSI
    enabled = container.checkbox(f"Override {label}", key=f"{key}__enabled", value=False)
    if not enabled:
        return None
    return container.number_input(
        f"{label} [{unit}]",
        key=_pressure_widget_key(key),
        value=float(display_default),
        step=1.0 if unit == "kPa" else 0.5,
    )


def _build_request(selected_vde_id: int):
    mode_label = st.session_state.get("load_case_analysis_mode", "Drive Cycle")
    analysis_mode = _MODE_LABELS[mode_label]
    mass_label = st.session_state.get("load_case_mass_mode", "Inherit baseline")
    mass_mode = _MASS_LABELS[mass_label]

    fixed_speed = None
    speed_sweep = None
    if analysis_mode is LoadCaseAnalysisMode.FIXED_SPEED:
        fixed_speed = FixedSpeedSpec(float(st.session_state["load_case_fixed_speed_kph"]))
    elif analysis_mode is LoadCaseAnalysisMode.SPEED_SWEEP:
        speed_sweep = SpeedSweepSpec(
            float(st.session_state["load_case_sweep_start_kph"]),
            float(st.session_state["load_case_sweep_end_kph"]),
            float(st.session_state["load_case_sweep_step_kph"]),
        )

    environment = LoadCaseEnvironment(
        temperature_C=st.session_state.get("load_case_temperature_C") if st.session_state.get("load_case_temperature_C__enabled") else None,
        pressure_kPa=(
            _pressure_to_kpa(st.session_state.get(_pressure_widget_key("load_case_pressure")))
            if st.session_state.get("load_case_pressure__enabled")
            else None
        ),
        air_density_kg_m3=(
            st.session_state.get("load_case_air_density_kg_m3")
            if st.session_state.get("load_case_air_density_kg_m3__enabled")
            else None
        ),
        tire_pressure_delta_kPa=(
            _pressure_to_kpa(st.session_state.get(_pressure_widget_key("load_case_tire_pressure_delta")))
            if st.session_state.get("load_case_tire_pressure_delta__enabled")
            else None
        ),
    )

    mass = LoadCaseMass(
        mode=mass_mode,
        payload_delta_kg=st.session_state.get("load_case_payload_delta_kg") if mass_mode is LoadCaseMassMode.PAYLOAD_DELTA else None,
        target_payload_kg=st.session_state.get("load_case_target_payload_kg") if mass_mode is LoadCaseMassMode.TARGET_PAYLOAD else None,
        custom_vehicle_mass_kg=st.session_state.get("load_case_custom_mass_kg") if mass_mode is LoadCaseMassMode.CUSTOM else None,
        gvwr_kg=st.session_state.get("load_case_gvwr_kg") if mass_mode is LoadCaseMassMode.GVWR else None,
        gcwr_kg=st.session_state.get("load_case_gcwr_kg") if mass_mode is LoadCaseMassMode.GCWR else None,
        use_estimated_rrc_prior=bool(st.session_state.get("load_case_use_estimated_rrc_prior", True)),
    )

    trailer_enabled = bool(st.session_state.get("load_case_trailer_enabled"))
    trailer_abc_enabled = bool(st.session_state.get("load_case_trailer_abc_enabled"))
    trailer = LoadCaseTrailer(
        enabled=trailer_enabled,
        mass_kg=st.session_state.get("load_case_trailer_mass_kg") if trailer_enabled else None,
        A_N=st.session_state.get("load_case_trailer_A_N") if trailer_enabled and trailer_abc_enabled else None,
        B_N_per_kph=st.session_state.get("load_case_trailer_B_N_per_kph") if trailer_enabled and trailer_abc_enabled else None,
        C_N_per_kph2=st.session_state.get("load_case_trailer_C_N_per_kph2") if trailer_enabled and trailer_abc_enabled else None,
    )

    return RoadloadLoadCaseRequest(
        baseline_vde_id=selected_vde_id,
        analysis_mode=analysis_mode,
        environment=environment,
        mass=mass,
        trailer=trailer,
        route=LoadCaseRoute(float(st.session_state.get("load_case_grade_pct", 0.0))),
        fixed_speed=fixed_speed,
        speed_sweep=speed_sweep,
    )


def _render_controls() -> bool:
    with st.container():
        st.subheader("Analysis")
        analysis_cols = st.columns(2)
        analysis_cols[0].selectbox("Analysis mode", list(_MODE_LABELS), key="load_case_analysis_mode")
        selected_mode = _MODE_LABELS[st.session_state.get("load_case_analysis_mode", "Drive Cycle")]
        if selected_mode is LoadCaseAnalysisMode.DRIVE_CYCLE:
            analysis_cols[1].text_input(
                "Cycle",
                value="Canonical cycle from selected VDE",
                disabled=True,
                key="load_case_cycle_display",
            )
        elif selected_mode is LoadCaseAnalysisMode.FIXED_SPEED:
            analysis_cols[1].number_input(
                "Fixed speed [km/h]", min_value=0.0, value=90.0, step=5.0, key="load_case_fixed_speed_kph"
            )
        else:
            sweep_cols = analysis_cols[1].columns(3)
            sweep_cols[0].number_input("Start [km/h]", min_value=0.0, value=40.0, step=5.0, key="load_case_sweep_start_kph")
            sweep_cols[1].number_input("End [km/h]", min_value=0.0, value=120.0, step=5.0, key="load_case_sweep_end_kph")
            sweep_cols[2].number_input("Increment [km/h]", min_value=0.0, value=20.0, step=5.0, key="load_case_sweep_step_kph")

        env_col, mass_col = st.columns(2)
        with env_col:
            st.subheader("Environment")
            _optional_number(st, "temperature [°C]", key="load_case_temperature_C", value=20.0, step=1.0)
            _optional_pressure(st, "ambient pressure", key="load_case_pressure", default_kpa=101.325)
            _optional_number(st, "explicit air density [kg/m³]", key="load_case_air_density_kg_m3", min_value=0.0, step=0.01)
            _optional_pressure(st, "tire-pressure delta", key="load_case_tire_pressure_delta", default_kpa=0.0)

        with mass_col:
            st.subheader("Vehicle Load")
            st.selectbox("Mass basis", list(_MASS_LABELS), key="load_case_mass_mode")
            st.checkbox(
                "Use estimated RRC when vehicle-specific RRC is unavailable",
                value=True,
                key="load_case_use_estimated_rrc_prior",
            )
            mass_mode = _MASS_LABELS[st.session_state.get("load_case_mass_mode", "Inherit baseline")]
            if mass_mode is LoadCaseMassMode.PAYLOAD_DELTA:
                st.number_input("Payload delta [kg]", value=0.0, step=10.0, key="load_case_payload_delta_kg")
            elif mass_mode is LoadCaseMassMode.TARGET_PAYLOAD:
                st.number_input("Target payload [kg]", min_value=0.0, value=300.0, step=10.0, key="load_case_target_payload_kg")
            elif mass_mode is LoadCaseMassMode.CUSTOM:
                st.number_input("Custom vehicle mass [kg]", min_value=0.0, value=1800.0, step=10.0, key="load_case_custom_mass_kg")
            elif mass_mode is LoadCaseMassMode.GVWR:
                st.number_input("GVWR [kg]", min_value=0.0, value=2200.0, step=10.0, key="load_case_gvwr_kg")
            elif mass_mode is LoadCaseMassMode.GCWR:
                st.number_input("GCWR [kg]", min_value=0.0, value=3200.0, step=10.0, key="load_case_gcwr_kg")

        trailer_col, route_col = st.columns(2)
        with trailer_col:
            st.subheader("Trailer / Towing")
            trailer_enabled = st.checkbox("Enable trailer", key="load_case_trailer_enabled")
            if trailer_enabled:
                st.number_input("Trailer mass [kg]", min_value=0.0, value=800.0, step=10.0, key="load_case_trailer_mass_kg")
                abc_enabled = st.checkbox("Complete trailer ABC available", key="load_case_trailer_abc_enabled")
                if abc_enabled:
                    abc_cols = st.columns(3)
                    abc_cols[0].number_input("A [N]", value=0.0, step=1.0, key="load_case_trailer_A_N")
                    abc_cols[1].number_input("B [N/kph]", value=0.0, step=0.01, key="load_case_trailer_B_N_per_kph")
                    abc_cols[2].number_input("C [N/kph²]", value=0.0, step=0.0001, format="%.6f", key="load_case_trailer_C_N_per_kph2")
                else:
                    st.caption("Without complete ABC, trailer roadload is reported as unavailable; mass still affects inertia and grade.")
        with route_col:
            st.subheader("Route")
            st.number_input("Grade [%]", value=0.0, step=0.5, key="load_case_grade_pct")

        st.caption("Temporary calculation only. No VDE or FuelCons row will be inserted or updated.")
        return st.button("Calculate load case", type="primary", key="calculate_roadload_load_case")


def _render_drive_cycle_result(result) -> None:
    st.subheader("Drive Cycle · Baseline vs Temporary Scenario")
    rows = []
    for item in result.comparisons:
        rows.append(
            {
                "Boundary": item.roadload_basis.value,
                "Baseline VDE [MJ/km]": item.baseline_mj_per_km,
                "Scenario VDE [MJ/km]": item.scenario_mj_per_km,
                "Delta [MJ/km]": item.delta_mj_per_km,
                "Delta [%]": item.delta_pct,
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def build_force_curve_figure(result, *, show_breakdown: bool = False) -> go.Figure:
    points = result.force_curve_points
    speeds = [point.speed_kph for point in points]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=speeds,
            y=[point.baseline_force_N for point in points],
            mode="lines",
            name="Baseline",
            line={"color": "#64748b", "width": 3},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=speeds,
            y=[point.scenario_force_N for point in points],
            mode="lines+markers" if result.request.analysis_mode is LoadCaseAnalysisMode.SPEED_SWEEP else "lines",
            name="Scenario",
            line={"color": "#0f766e", "width": 3},
        )
    )
    if show_breakdown:
        breakdown = (
            ("Rolling load delta", [point.rolling_force_N for point in points], "#16a34a"),
            ("Trailer", [point.trailer_force_N for point in points], "#f59e0b"),
            ("Grade", [point.grade_force_N for point in points], "#7c3aed"),
            ("Condition delta", [point.condition_force_N for point in points], "#0284c7"),
        )
        for name, values, color in breakdown:
            if any(abs(value) > 1e-12 for value in values):
                figure.add_trace(
                    go.Scatter(
                        x=speeds,
                        y=values,
                        mode="lines",
                        name=name,
                        line={"color": color, "width": 2, "dash": "dot"},
                    )
                )
    selected = [point for point in points if point.selected_operating_point]
    if selected:
        point = selected[0]
        figure.add_trace(
            go.Scatter(
                x=[point.speed_kph],
                y=[point.scenario_force_N],
                mode="markers",
                name=f"Selected operating point: {point.speed_kph:g} km/h",
                marker={"size": 12, "color": "#dc2626", "symbol": "diamond"},
            )
        )
    figure.update_layout(
        xaxis_title="Speed [km/h]",
        yaxis_title="Force [N]",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        margin={"l": 20, "r": 20, "t": 35, "b": 20},
    )
    return figure


def build_drive_cycle_profile_figure(result) -> go.Figure:
    """Plot canonical cycle-resolved outputs; no UI-side force physics."""
    baseline = result.baseline_total_profile
    scenario = result.scenario_total_profile
    if baseline is None or scenario is None:
        raise ValueError("Drive Cycle profiles are unavailable.")
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Tractive Force", "Wheel Power", "Cycle Speed"),
    )
    for profile, name, color in (
        (baseline, "Baseline", "#64748b"),
        (scenario, "Scenario", "#0f766e"),
    ):
        figure.add_trace(
            go.Scatter(
                x=profile.time_s,
                y=profile.tractive_force_N,
                mode="lines",
                name=name,
                legendgroup=name,
                line={"color": color, "width": 2},
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=profile.time_s,
                y=[value / 1000.0 for value in profile.tractive_power_W],
                mode="lines",
                name=name,
                legendgroup=name,
                showlegend=False,
                line={"color": color, "width": 2},
            ),
            row=2,
            col=1,
        )
    figure.add_trace(
        go.Scatter(
            x=baseline.time_s,
            y=[value * 3.6 for value in baseline.speed_mps],
            mode="lines",
            name="Cycle speed",
            line={"color": "#2563eb", "width": 1.5},
        ),
        row=3,
        col=1,
    )
    figure.update_yaxes(title_text="Force [N]", row=1, col=1)
    figure.update_yaxes(title_text="Power [kW]", row=2, col=1)
    figure.update_yaxes(title_text="Speed [km/h]", row=3, col=1)
    figure.update_xaxes(title_text="Time [s]", row=3, col=1)
    figure.update_layout(
        title="Drive Cycle Dynamic Demand",
        hovermode="x unified",
        height=720,
        legend={"orientation": "h", "y": 1.08, "x": 0},
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )
    return figure


def build_wheel_power_curve_figure(result) -> go.Figure:
    points = result.steady_state_points
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[point.speed_kph for point in points],
            y=[point.total.baseline_roadload_force_N * (point.speed_kph / 3.6) / 1000.0 for point in points],
            mode="lines+markers",
            name="Baseline",
            line={"color": "#64748b", "width": 3},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[point.speed_kph for point in points],
            y=[point.total.wheel_power_kW for point in points],
            mode="lines+markers",
            name="Scenario",
            line={"color": "#0f766e", "width": 3},
        )
    )
    figure.update_layout(
        xaxis_title="Speed [km/h]",
        yaxis_title="Wheel Power [kW]",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        margin={"l": 20, "r": 20, "t": 35, "b": 20},
    )
    return figure


def _render_force_curve(result, *, title: str = "Force vs Speed · Baseline vs Scenario", key_suffix: str = "primary") -> None:
    st.subheader(title)
    show_breakdown = st.checkbox("Show force breakdown", key="load_case_show_force_breakdown")
    st.plotly_chart(
        build_force_curve_figure(result, show_breakdown=show_breakdown),
        width="stretch",
        key=f"roadload_load_case_force_curve_{key_suffix}",
    )


def _steady_rows(result) -> list[dict]:
    rows = []
    for point in result.steady_state_points:
        for boundary in (point.total, point.net):
            if boundary is None:
                continue
            rows.append(
                {
                    "Speed [km/h]": point.speed_kph,
                    "Boundary": boundary.roadload_basis.value,
                    "Baseline roadload [N]": boundary.baseline_roadload_force_N,
                    "Rolling delta [N]": boundary.rolling_force_N,
                    "Trailer [N]": boundary.trailer_force_N,
                    "Grade [N]": boundary.grade_force_N,
                    "Condition [N]": boundary.condition_force_N,
                    "Wheel force [N]": boundary.wheel_force_N,
                    "Wheel power [kW]": boundary.wheel_power_kW,
                    "Wheel energy intensity [MJ/km]": boundary.wheel_energy_intensity_MJ_per_km,
                    "State": boundary.energy_mode.value,
                }
            )
    return rows


def _render_steady_state_result(result) -> None:
    rows = _steady_rows(result)
    frame = pd.DataFrame(rows)
    st.dataframe(frame, hide_index=True, width="stretch")


def _render_primary_kpis(result) -> None:
    if not result.steady_state_points:
        return
    point = result.steady_state_points[0] if result.request.analysis_mode is LoadCaseAnalysisMode.FIXED_SPEED else result.steady_state_points[-1]
    demand = point.total
    st.subheader("Steady-State Wheel Demand")
    st.caption(f"{point.speed_kph:g} km/h | {_humanize(result.mass_state.mass_basis)}")
    kpis = st.columns(3)
    kpis[0].metric("Wheel Force", f"{demand.wheel_force_N:,.0f} N")
    kpis[1].metric("Wheel Power", f"{demand.wheel_power_kW:,.1f} kW")
    kpis[2].metric("Energy Intensity", f"{demand.wheel_energy_intensity_MJ_per_km:,.3f} MJ/km")


def _render_result(result) -> None:
    _render_primary_kpis(result)
    summary_cols = st.columns(5)
    summary_cols[0].metric("Baseline calculation mass", f"{result.baseline_calculation_mass_kg:,.0f} kg")
    summary_cols[1].metric("Baseline basis", _humanize(result.baseline_mass_basis))
    summary_cols[2].metric("Source", result.baseline_mass_source_field)
    summary_cols[3].metric("Scenario calculation mass", f"{result.scenario_calculation_mass_kg:,.0f} kg")
    summary_cols[4].metric("Scenario basis", _humanize(result.scenario_mass_basis))
    system_col = st.columns(1)[0]
    system_col.metric("System mass", f"{result.scenario_system_mass_kg:,.0f} kg")

    context_cols = st.columns(4)
    context_cols[0].metric("Grade", f"{result.request.route.grade_pct:g} %")
    context_cols[1].metric(
        "Trailer",
        f"{result.trailer_state.mass_kg:,.0f} kg" if result.trailer_state.enabled else "Not used",
        help=f"Roadload status: {_humanize(result.trailer_state.status)}",
    )
    context_cols[2].metric(
        "Tire-pressure correction",
        _humanize(result.provenance["tire_pressure_delta"]),
    )
    rolling_display = {
        "RULE_ESTIMATED": "Estimated",
        "VDE_SPECIFIC": "Applied · VDE RRC",
        "TIRE_DB": "Applied · Tire DB",
    }.get(result.rolling_rrc_source, _humanize(result.rolling_correction_status))
    context_cols[3].metric("Rolling correction", rolling_display)

    with st.expander(
        "Rolling correction details",
        expanded=result.rolling_rrc_source == "RULE_ESTIMATED",
    ):
        rolling_cols = st.columns(4)
        rolling_cols[0].metric(
            "RRC",
            "Unavailable" if result.rolling_rrc_value is None else f"{result.rolling_rrc_value:.3f} N/kN",
        )
        rolling_cols[1].metric("Source", result.rolling_rrc_source)
        rolling_cols[2].metric("Match", result.rolling_state.match or "Unavailable")
        rolling_cols[3].metric("Confidence", result.rolling_state.confidence or "Unavailable")
        if result.rolling_delta_abc is not None:
            baseline_rolling = result.rolling_state.baseline_rolling_abc
            scenario_rolling = result.rolling_state.scenario_rolling_abc
            if baseline_rolling is not None and scenario_rolling is not None:
                st.caption(
                    "Resolved rolling states · "
                    f"Baseline ABC=({baseline_rolling.A_N:.3f}, {baseline_rolling.B_N_per_kph:.6f}, "
                    f"{baseline_rolling.C_N_per_kph2:.6f}) · "
                    f"Scenario ABC=({scenario_rolling.A_N:.3f}, {scenario_rolling.B_N_per_kph:.6f}, "
                    f"{scenario_rolling.C_N_per_kph2:.6f})"
                )
            st.caption(
                "Temporary rolling ΔABC: "
                f"A={result.rolling_delta_abc.A_N:.3f} N, "
                f"B={result.rolling_delta_abc.B_N_per_kph:.6f} N/kph, "
                f"C={result.rolling_delta_abc.C_N_per_kph2:.6f} N/kph² · "
                f"Rule key: {result.rolling_rrc_rule_key or '-'}"
            )
        diagnostic = result.rolling_state.diagnostic
        if diagnostic is not None:
            diagnostic_rows = [
                ("Raw category", diagnostic.raw_category),
                ("Lookup category", diagnostic.lookup_category),
                ("Raw electrification", diagnostic.raw_electrification),
                ("Resolved electrification", diagnostic.resolved_electrification),
                ("Electrification source", diagnostic.electrification_source),
                ("Raw transmission", diagnostic.raw_transmission),
                ("Gears", diagnostic.gears),
                ("Resolved transmission", diagnostic.resolved_transmission_key),
                ("Attempted key", diagnostic.attempted_rule_key),
                ("Matched key", diagnostic.matched_rule_key),
                (
                    "RRC prior",
                    None if diagnostic.rrc_n_per_kn is None else f"{diagnostic.rrc_n_per_kn:.3f} N/kN",
                ),
                ("Status", f"{diagnostic.source} / {diagnostic.status}"),
                ("Missing discriminants", ", ".join(diagnostic.missing_discriminants)),
                ("Reason", diagnostic.reason),
            ]
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Field": label, "Value": value if value not in (None, "") else "—"}
                        for label, value in diagnostic_rows
                    ]
                ),
                hide_index=True,
                width="stretch",
            )

    for note in result.explanatory_notes:
        st.info(note)

    if result.request.analysis_mode is LoadCaseAnalysisMode.DRIVE_CYCLE:
        st.plotly_chart(
            build_drive_cycle_profile_figure(result),
            width="stretch",
            key="roadload_load_case_cycle_profile",
        )
        _render_drive_cycle_result(result)
        with st.expander("Roadload Reference Curve", expanded=False):
            st.caption(
                "Static A + Bv + Cv² reference only. It does not represent inertial force from cycle acceleration."
            )
            _render_force_curve(result, title="Roadload Reference Curve", key_suffix="reference")
    else:
        _render_force_curve(result)
        if result.request.analysis_mode is LoadCaseAnalysisMode.SPEED_SWEEP:
            st.subheader("Wheel Power vs Speed · Baseline vs Scenario")
            st.plotly_chart(
                build_wheel_power_curve_figure(result),
                width="stretch",
                key="roadload_load_case_power_curve",
            )
        _render_steady_state_result(result)

    density_cols = st.columns(2)
    density_cols[0].metric(
        "Baseline air density",
        "Unavailable" if result.baseline_air_density_kg_m3 is None else f"{result.baseline_air_density_kg_m3:.4f} kg/m³",
    )
    density_cols[1].metric(
        "Scenario air density",
        "Unavailable" if result.scenario_air_density_kg_m3 is None else f"{result.scenario_air_density_kg_m3:.4f} kg/m³",
    )

    if result.warnings:
        with st.expander("Warnings / unavailable corrections", expanded=True):
            for warning in result.warnings:
                st.warning(warning)
    with st.expander("Provenance", expanded=False):
        st.dataframe(
            pd.DataFrame([{"Term": _humanize(key), "Status / basis": _humanize(value)} for key, value in result.provenance.items()]),
            hide_index=True,
            width="stretch",
        )


def render_roadload_load_case_lab() -> None:
    rows = fetch_vde_all_rows()
    if not rows:
        st.warning("No persisted VDE baseline is available in the active database.")
        return
    by_id = {int(row["id"]): row for row in rows if row.get("id") is not None}
    labels = {_row_label(row): int(row["id"]) for row in rows if row.get("id") is not None}
    requested_id = st.session_state.get(LAB_VDE_SESSION_KEY)
    try:
        requested_id = int(requested_id) if requested_id is not None else None
    except (TypeError, ValueError):
        requested_id = None
    default_id = requested_id if requested_id in by_id else next(iter(by_id))
    default_label = next(label for label, vde_id in labels.items() if vde_id == default_id)

    selected_label = st.selectbox(
        "Persisted VDE baseline",
        list(labels),
        index=list(labels).index(default_label),
        key="load_case_baseline_selector",
    )
    selected_id = labels[selected_label]
    st.session_state[LAB_VDE_SESSION_KEY] = selected_id
    selected_row = fetch_vde_by_id(selected_id)

    st.caption(
        f"Authoritative persisted baseline: VDE-{selected_id:03d}. The source row remains immutable; all overrides are in memory."
    )
    submitted = _render_controls()
    if submitted:
        try:
            result = calculate_roadload_load_case(
                selected_row,
                _build_request(selected_id),
                rolling_prior_context=tuple(fetch_fuelcons_by_vde_id(selected_id)),
            )
        except (TypeError, ValueError) as exc:
            st.session_state.pop(LAB_RESULT_SESSION_KEY, None)
            st.error(str(exc))
        else:
            st.session_state[LAB_RESULT_SESSION_KEY] = result

    result = st.session_state.get(LAB_RESULT_SESSION_KEY)
    if result is not None and result.request.baseline_vde_id == selected_id:
        st.divider()
        _render_result(result)


__all__ = [
    "LAB_VDE_SESSION_KEY",
    "LAB_RESULT_SESSION_KEY",
    "build_force_curve_figure",
    "render_roadload_load_case_lab",
]
