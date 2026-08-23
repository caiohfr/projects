# src/vde_app/comparison_report_charts.py
# -----------------------------------------------------------------------------
# Package 8C - pure Plotly figure builders for the Comparison Report Dashboard
# and Roadload & VDE tabs. No database access, no Streamlit, no session-state.
# Takes already-filtered/labeled rows from comparison_report_viewmodels.py and
# returns a plotly.graph_objects.Figure. No roadload/VDE physics here.
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Sequence

import plotly.graph_objects as go

_SEMANTIC_COLOR = {
    "BETTER": "rgba(34,197,94,0.75)",
    "WORSE": "rgba(239,68,68,0.75)",
}
_NEUTRAL_COLOR = "rgba(107,114,128,0.6)"
_BOUNDARY_DASH = {"TOTAL": "solid", "NET": "dash"}


def build_grouped_bar_figure(
    rows: Sequence[dict[str, Any]],
    *,
    y_title: str,
    x_title: str = "Scenario",
    color_by_semantic: bool = False,
) -> go.Figure:
    """rows: [{"label", "value", "group" (optional), "semantic" (optional)}].

    When any row carries a "group" (e.g. TOTAL/NET), bars are grouped by it
    (grouped bar chart, boundary as a distinct series -- never merged/averaged).
    Otherwise, when color_by_semantic=True, bar color reflects BETTER/WORSE
    from the Metric Registry direction (competitor delta chart) -- color is a
    presentation concern only, never computed here.
    """
    fig = go.Figure()
    if not rows:
        fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
        return fig

    has_group = any(row.get("group") for row in rows)
    if has_group:
        groups: list[str] = []
        for row in rows:
            group = row.get("group") or "value"
            if group not in groups:
                groups.append(group)
        for group in groups:
            group_rows = [row for row in rows if (row.get("group") or "value") == group]
            fig.add_bar(
                name=group,
                x=[row["label"] for row in group_rows],
                y=[row["value"] for row in group_rows],
            )
        fig.update_layout(barmode="group")
    else:
        marker_color = None
        if color_by_semantic:
            marker_color = [_SEMANTIC_COLOR.get(row.get("semantic"), _NEUTRAL_COLOR) for row in rows]
        fig.add_bar(
            x=[row["label"] for row in rows],
            y=[row["value"] for row in rows],
            marker_color=marker_color,
        )
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
    return fig


def build_fe_vde_figure(
    points: Sequence[dict[str, Any]],
    lines: Sequence[dict[str, Any]],
    *,
    x_title: str,
    y_title: str,
) -> go.Figure:
    """points: from build_fe_vde_points()["points"]. lines: from build_iso_pse_lines()."""
    fig = go.Figure()
    for line in lines:
        fig.add_scatter(
            x=line["x"],
            y=line["y"],
            mode="lines",
            name=f"eta={line['eta']:.2f}",
            line=dict(width=1, dash="dot"),
            hoverinfo="name",
        )
    if points:
        hover_text = []
        for point in points:
            parts = [point["label"], f"Provenance: {point.get('provenance') or 'UNKNOWN'}"]
            if point.get("is_temporary_net"):
                parts.append("NET · TEMPORARY")
            if point.get("revision_status") == "STALE":
                parts.append("STALE SOURCE")
            hover_text.append("<br>".join(parts))
        fig.add_scatter(
            x=[point["x"] for point in points],
            y=[point["y"] for point in points],
            mode="markers",
            marker=dict(
                symbol=["star" if point["role"] == "REFERENCE" else "circle" for point in points],
                size=[14 if point["role"] == "REFERENCE" else 10 for point in points],
            ),
            hovertext=hover_text,
            hoverinfo="text",
            name="Scenarios",
        )
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
    return fig


def build_cycle_demand_figure(series: Sequence[dict[str, Any]], time_s: Sequence[float]) -> go.Figure:
    """series: from build_cycle_demand_rows()["series"] (roadload_analysis output,
    unmodified). TOTAL/NET distinguished by line-dash, matching the existing
    convention in vde_request_compact.py's cycle-power figure.
    """
    fig = go.Figure()
    for entry in series:
        fig.add_scatter(
            x=time_s,
            y=entry["demanded_power_kw"],
            mode="lines",
            name=f"{entry['scenario_label']} · {entry['boundary']}",
            line=dict(dash=_BOUNDARY_DASH.get(entry["boundary"], "solid")),
        )
    fig.update_layout(xaxis_title="Time [s]", yaxis_title="Demanded power [kW]")
    return fig


__all__ = [
    "build_grouped_bar_figure",
    "build_fe_vde_figure",
    "build_cycle_demand_figure",
]
