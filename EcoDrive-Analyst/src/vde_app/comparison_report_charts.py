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


_PSE_GUIDE_COLOR = "rgba(107,114,128,0.55)"
_PSE_GUIDE_LABEL_COLOR = "rgba(75,85,99,0.9)"


def build_fe_vde_figure(
    points: Sequence[dict[str, Any]],
    lines: Sequence[dict[str, Any]],
    *,
    x_title: str,
    y_title: str,
) -> go.Figure:
    """points: from build_fe_vde_points()["points"]. lines: from build_iso_pse_lines()
    -- always plotted, when present, as thin muted guides with a direct
    end-of-line label (Package 8F polish: restrained, visually secondary to
    the scenario markers, no legend entries competing with "Scenarios").
    """
    fig = go.Figure()
    for line in lines:
        fig.add_scatter(
            x=line["x"],
            y=line["y"],
            mode="lines",
            name=f"eta={line['eta']:.2f}",
            line=dict(width=1, dash="dot", color=_PSE_GUIDE_COLOR),
            hoverinfo="name",
            showlegend=False,
        )
        if line["x"] and line["y"]:
            fig.add_annotation(
                x=line["x"][-1],
                y=line["y"][-1],
                text=f"η={line['eta']:.2f}",
                showarrow=False,
                xanchor="left",
                font=dict(size=10, color=_PSE_GUIDE_LABEL_COLOR),
            )
    if points:
        hover_text = []
        for point in points:
            parts = [point["label"], f"Provenance: {point.get('provenance') or 'UNKNOWN'}"]
            if point.get("is_temporary_net"):
                parts.append("NET · TEMPORARY")
            if point.get("revision_status") == "STALE":
                parts.append("STALE SOURCE")
            if point.get("fuel_basis_label"):
                parts.append(f"Fuel basis: {point['fuel_basis_label']}")
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


def build_explore_bar(rows: Sequence[Any], *, x_title: str, y_title: str) -> go.Figure:
    """Explore Custom Chart - Bar (Package 8D). Row shape (label/value/optional
    group) is identical to build_grouped_bar_figure's, so it is reused rather
    than duplicated -- see comparison_report_viewmodels.build_explore_bar_rows.
    """
    dict_rows = [{"label": row.label, "value": row.value, "group": row.group} for row in rows]
    return build_grouped_bar_figure(dict_rows, y_title=y_title, x_title=x_title)


def _explore_scatter_hover(point: Any) -> str:
    parts = [point.label, f"Provenance: {point.provenance or 'UNKNOWN'}"]
    if point.is_temporary_net:
        parts.append("NET · TEMPORARY")
    if point.revision_status == "STALE":
        parts.append("STALE SOURCE")
    return "<br>".join(parts)


def build_explore_scatter(points: Sequence[Any], *, x_title: str, y_title: str) -> go.Figure:
    """Explore Custom Chart - Scatter (Package 8D). No regression/trend lines
    -- exploration only (Sec 16, 51). Reference marked with a star, matching
    build_fe_vde_figure's existing convention; grouped by dimension when the
    caller supplies one (one trace per group value, never averaged).
    """
    fig = go.Figure()
    if not points:
        fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
        return fig

    groups: list[str] = []
    for point in points:
        group = point.group or "Scenarios"
        if group not in groups:
            groups.append(group)

    for group in groups:
        group_points = [p for p in points if (p.group or "Scenarios") == group]
        fig.add_scatter(
            x=[p.x for p in group_points],
            y=[p.y for p in group_points],
            mode="markers",
            name=group,
            marker=dict(
                symbol=["star" if p.role == "REFERENCE" else "circle" for p in group_points],
                size=[14 if p.role == "REFERENCE" else 10 for p in group_points],
            ),
            hovertext=[_explore_scatter_hover(p) for p in group_points],
            hoverinfo="text",
        )
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
    return fig


def build_explore_line(rows: Sequence[Any], *, x_title: str, y_title: str) -> go.Figure:
    """Explore Custom Chart - Line (Package 8D). Rows must already be ordered
    by an explicit basis (comparison_report_viewmodels.build_explore_line_rows
    sorts by the chosen ordered dimension) -- this builder never reorders or
    infers order from selection sequence (Sec 17).
    """
    fig = go.Figure()
    if not rows:
        fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
        return fig

    groups: list[str] = []
    for row in rows:
        group = row.group or "Value"
        if group not in groups:
            groups.append(group)

    for group in groups:
        group_rows = [r for r in rows if (r.group or "Value") == group]
        fig.add_scatter(
            x=[r.x for r in group_rows], y=[r.y for r in group_rows], mode="lines+markers", name=group
        )
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
    return fig


def build_lineage_waterfall_chart(steps: Sequence[Any], *, y_title: str) -> go.Figure:
    """Physical VDE Lineage waterfall (Package 8D Sec 32, 37). `steps` is
    LineageWaterfallResult.steps: baseline is an absolute value, each
    following OK step is already child-parent (never recomputed here). Only
    OK steps are plotted -- a trailing UNAVAILABLE/INCOMPATIBLE marker step
    carries no numeric value and is surfaced by the caller via
    LineageWaterfallResult.incomplete_reason instead of a misleading bar.
    """
    ok_steps = [step for step in steps if step.status == "OK"]
    fig = go.Figure()
    if not ok_steps:
        fig.update_layout(yaxis_title=y_title)
        return fig

    measures = ["absolute"] + ["relative"] * (len(ok_steps) - 1)
    x_labels = [step.label for step in ok_steps]
    y_values = [step.value if step.delta is None else step.delta for step in ok_steps]
    text = [step.formatted_value if step.delta is None else (step.formatted_delta or "") for step in ok_steps]

    if len(ok_steps) > 1:
        measures.append("total")
        x_labels.append(f"{ok_steps[-1].label} (final)")
        y_values.append(ok_steps[-1].value)
        text.append(ok_steps[-1].formatted_value)

    fig.add_trace(
        go.Waterfall(
            x=x_labels,
            y=y_values,
            measure=measures,
            text=text,
            textposition="outside",
            connector=dict(line=dict(color="rgba(107,114,128,0.5)")),
        )
    )
    fig.update_layout(yaxis_title=y_title, showlegend=False)
    return fig


def build_walk_chart(rows: Sequence[Any], *, y_title: str, target_value: float | None = None) -> go.Figure:
    """Versatile KPI Walk hero chart (Package 8F Increment 5). ABSOLUTE rows
    render a full bar from 0; DELTA rows render a floating bar from their own
    resolved delta base's absolute value (base = row.absolute_value -
    row.delta_value, algebraically exact -- WalkRow never carries a second
    "base value" field to keep it in sync with) up to the item's absolute
    value. This is a plain go.Bar with an explicit per-bar `base`, never
    go.Waterfall -- a WalkStep's delta base is explicit presentation intent
    (PREVIOUS_WALK_STATE / REFERENCE / EXPLICIT_ITEM), not a running
    cumulative total, so Waterfall's automatic running-total semantics would
    misrender a REFERENCE- or EXPLICIT_ITEM-based delta. Only status=="OK"
    rows are plotted -- UNAVAILABLE/INCOMPATIBLE/INVALID_CONFIG rows are the
    caller's responsibility to surface as text (mirrors
    build_lineage_waterfall_chart's own OK-only rule), and this builder is
    intentionally separate from that one -- the two domains (explicit DB
    lineage vs. presentation-intent walk) are never coupled or repurposed
    into each other.
    """
    fig = go.Figure()
    ok_rows = [row for row in rows if row.status == "OK"]
    if ok_rows:
        bases: list[float] = []
        heights: list[float] = []
        colors: list[str] = []
        texts: list[str] = []
        labels: list[str] = []
        hover: list[str] = []
        for row in ok_rows:
            if row.display_mode == "ABSOLUTE":
                bases.append(0.0)
                heights.append(row.absolute_value)
                texts.append(row.formatted_absolute_value)
            else:
                bases.append(row.absolute_value - row.delta_value)
                heights.append(row.delta_value)
                texts.append(row.formatted_delta or "")
            colors.append(_SEMANTIC_COLOR.get(row.semantic, _NEUTRAL_COLOR))
            labels.append(row.label)
            hover_parts = [row.label, f"Absolute: {row.formatted_absolute_value}"]
            if row.display_mode == "DELTA" and row.formatted_delta:
                hover_parts.append(f"Δ vs {row.delta_base_label}: {row.formatted_delta}")
            if row.target_gap is not None:
                hover_parts.append(f"Target gap: {row.target_gap.absolute_gap:+.3g}")
            if row.provenance:
                hover_parts.append(f"Provenance: {row.provenance}")
            hover.append("<br>".join(hover_parts))
        fig.add_bar(
            x=labels,
            y=heights,
            base=bases,
            marker_color=colors,
            text=texts,
            textposition="outside",
            hovertext=hover,
            hoverinfo="text",
        )
    if target_value is not None:
        fig.add_hline(y=target_value, line_dash="dot", line_color="rgba(107,114,128,0.9)", annotation_text="Target")
    fig.update_layout(yaxis_title=y_title, showlegend=False)
    return fig


__all__ = [
    "build_grouped_bar_figure",
    "build_fe_vde_figure",
    "build_cycle_demand_figure",
    "build_explore_bar",
    "build_explore_scatter",
    "build_explore_line",
    "build_lineage_waterfall_chart",
    "build_walk_chart",
]
