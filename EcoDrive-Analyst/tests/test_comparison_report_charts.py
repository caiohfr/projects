from __future__ import annotations

import unittest

import plotly.graph_objects as go

from src.vde_app.comparison_report_charts import (
    build_cycle_demand_figure,
    build_explore_bar,
    build_explore_line,
    build_explore_scatter,
    build_fe_vde_figure,
    build_grouped_bar_figure,
    build_lineage_waterfall_chart,
)
from src.vde_app.comparison_report_viewmodels import ExploreBarRow, ExploreLineRow, ExploreScatterPoint, LineageStep


class GroupedBarFigureTests(unittest.TestCase):
    def test_empty_rows_returns_empty_figure_without_crash(self):
        fig = build_grouped_bar_figure([], y_title="VDE [MJ/km]")
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 0)

    def test_ungrouped_rows_produce_one_trace(self):
        rows = [{"label": "Reference", "value": 1.24}, {"label": "Comparison", "value": 1.10}]
        fig = build_grouped_bar_figure(rows, y_title="VDE [MJ/km]")
        self.assertEqual(len(fig.data), 1)
        self.assertEqual(list(fig.data[0].x), ["Reference", "Comparison"])
        self.assertEqual(list(fig.data[0].y), [1.24, 1.10])

    def test_grouped_rows_total_net_produce_independent_series(self):
        rows = [
            {"label": "Reference", "value": 1.24, "group": "TOTAL"},
            {"label": "Reference", "value": 1.18, "group": "NET"},
            {"label": "Comparison", "value": 1.10, "group": "TOTAL"},
        ]
        fig = build_grouped_bar_figure(rows, y_title="VDE [MJ/km]")
        self.assertEqual(len(fig.data), 2)
        names = {trace.name for trace in fig.data}
        self.assertEqual(names, {"TOTAL", "NET"})
        total_trace = next(t for t in fig.data if t.name == "TOTAL")
        net_trace = next(t for t in fig.data if t.name == "NET")
        self.assertEqual(list(total_trace.y), [1.24, 1.10])
        self.assertEqual(list(net_trace.y), [1.18])  # NET missing for Comparison is absent, never fabricated

    def test_color_by_semantic_maps_better_worse(self):
        rows = [
            {"label": "A", "value": -4.0, "semantic": "BETTER"},
            {"label": "B", "value": 7.0, "semantic": "WORSE"},
            {"label": "C", "value": 0.0, "semantic": None},
        ]
        fig = build_grouped_bar_figure(rows, y_title="% delta", color_by_semantic=True)
        colors = list(fig.data[0].marker.color)
        self.assertNotEqual(colors[0], colors[1])
        self.assertNotEqual(colors[0], colors[2])


class FeVdeFigureTests(unittest.TestCase):
    def test_lines_and_points_produce_expected_trace_count(self):
        points = [
            {"label": "Reference", "role": "REFERENCE", "x": 1.24, "y": 6.0, "provenance": "HOMOLOGATED", "is_temporary_net": False, "revision_status": None},
            {"label": "Comparison", "role": "COMPARISON", "x": 1.10, "y": 5.5, "provenance": "ESTIMATED", "is_temporary_net": True, "revision_status": "STALE"},
        ]
        lines = [{"eta": 0.3, "x": [0.2, 1.2], "y": [0.6, 3.6]}, {"eta": 0.35, "x": [0.2, 1.2], "y": [0.5, 3.1]}]
        fig = build_fe_vde_figure(points, lines, x_title="VDE TOTAL [MJ/km]", y_title="Fuel [L/100km]")
        self.assertEqual(len(fig.data), 3)  # 2 lines + 1 marker trace
        marker_trace = fig.data[-1]
        self.assertEqual(len(marker_trace.x), 2)
        self.assertIn("NET · TEMPORARY", marker_trace.hovertext[1])
        self.assertIn("STALE SOURCE", marker_trace.hovertext[1])

    def test_no_points_still_returns_figure_with_lines_only(self):
        lines = [{"eta": 0.3, "x": [0.2, 1.2], "y": [0.6, 3.6]}]
        fig = build_fe_vde_figure([], lines, x_title="x", y_title="y")
        self.assertEqual(len(fig.data), 1)

    def test_no_lines_no_points_returns_empty_figure(self):
        fig = build_fe_vde_figure([], [], x_title="x", y_title="y")
        self.assertEqual(len(fig.data), 0)


class CycleDemandFigureTests(unittest.TestCase):
    def test_total_and_net_series_get_distinct_line_dash(self):
        series = [
            {"scenario_label": "Reference", "boundary": "TOTAL", "demanded_power_kw": [1.0, 2.0]},
            {"scenario_label": "Reference", "boundary": "NET", "demanded_power_kw": [0.8, 1.6]},
        ]
        fig = build_cycle_demand_figure(series, [0.0, 1.0])
        dashes = {trace.name: trace.line.dash for trace in fig.data}
        self.assertEqual(dashes["Reference · TOTAL"], "solid")
        self.assertEqual(dashes["Reference · NET"], "dash")

    def test_missing_net_series_simply_absent_not_fabricated(self):
        series = [{"scenario_label": "Reference", "boundary": "TOTAL", "demanded_power_kw": [1.0, 2.0]}]
        fig = build_cycle_demand_figure(series, [0.0, 1.0])
        self.assertEqual(len(fig.data), 1)


class ExploreBarChartTests(unittest.TestCase):
    def test_empty_rows_returns_empty_figure_without_crash(self):
        fig = build_explore_bar([], x_title="Scenario", y_title="VDE [MJ/km]")
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 0)

    def test_ungrouped_rows_produce_one_trace(self):
        rows = [
            ExploreBarRow(identity="fc:1", label="Reference", value=1.24, formatted_value="1.24", role="REFERENCE"),
            ExploreBarRow(identity="fc:2", label="Comparison", value=1.10, formatted_value="1.10", role="COMPARISON"),
        ]
        fig = build_explore_bar(rows, x_title="Scenario", y_title="VDE [MJ/km]")
        self.assertEqual(len(fig.data), 1)
        self.assertEqual(list(fig.data[0].x), ["Reference", "Comparison"])

    def test_grouped_rows_produce_one_trace_per_group(self):
        rows = [
            ExploreBarRow(identity="fc:1", label="A", value=1.0, formatted_value="1.0", role="REFERENCE", group="ICE"),
            ExploreBarRow(identity="fc:2", label="B", value=2.0, formatted_value="2.0", role="COMPARISON", group="BEV"),
        ]
        fig = build_explore_bar(rows, x_title="Scenario", y_title="VDE [MJ/km]")
        self.assertEqual({t.name for t in fig.data}, {"ICE", "BEV"})


class ExploreScatterChartTests(unittest.TestCase):
    def test_empty_points_returns_empty_figure_without_crash(self):
        fig = build_explore_scatter([], x_title="Mass [kg]", y_title="VDE [MJ/km]")
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 0)

    def test_reference_marked_with_star_symbol(self):
        points = [
            ExploreScatterPoint(
                identity="fc:1", label="Reference", role="REFERENCE", x=1500.0, y=1.24,
                provenance="HOMOLOGATED", is_temporary_net=False, revision_status=None,
            ),
            ExploreScatterPoint(
                identity="fc:2", label="Comparison", role="COMPARISON", x=1480.0, y=1.21,
                provenance="ESTIMATED", is_temporary_net=False, revision_status=None,
            ),
        ]
        fig = build_explore_scatter(points, x_title="Mass [kg]", y_title="VDE [MJ/km]")
        self.assertEqual(len(fig.data), 1)  # no group -> single trace
        symbols = list(fig.data[0].marker.symbol)
        self.assertEqual(symbols, ["star", "circle"])

    def test_grouped_points_produce_one_trace_per_group(self):
        points = [
            ExploreScatterPoint(
                identity="fc:1", label="A", role="REFERENCE", x=1.0, y=1.0,
                provenance=None, is_temporary_net=False, revision_status=None, group="ICE",
            ),
            ExploreScatterPoint(
                identity="fc:2", label="B", role="COMPARISON", x=2.0, y=2.0,
                provenance=None, is_temporary_net=False, revision_status=None, group="BEV",
            ),
        ]
        fig = build_explore_scatter(points, x_title="x", y_title="y")
        self.assertEqual({t.name for t in fig.data}, {"ICE", "BEV"})

    def test_temporary_net_and_stale_source_appear_in_hover(self):
        points = [
            ExploreScatterPoint(
                identity="fc:1", label="A", role="COMPARISON", x=1.0, y=1.0,
                provenance="ESTIMATED", is_temporary_net=True, revision_status="STALE",
            )
        ]
        fig = build_explore_scatter(points, x_title="x", y_title="y")
        hover = fig.data[0].hovertext[0]
        self.assertIn("NET · TEMPORARY", hover)
        self.assertIn("STALE SOURCE", hover)


class ExploreLineChartTests(unittest.TestCase):
    def test_empty_rows_returns_empty_figure_without_crash(self):
        fig = build_explore_line([], x_title="Model Year", y_title="VDE [MJ/km]")
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 0)

    def test_rows_plotted_in_given_order_never_reordered(self):
        rows = [
            ExploreLineRow(identity="vde:1", x=2010, x_label="2010", y=1.30, formatted_y="1.30", role="COMPARISON"),
            ExploreLineRow(identity="vde:2", x=2020, x_label="2020", y=1.10, formatted_y="1.10", role="REFERENCE"),
        ]
        fig = build_explore_line(rows, x_title="Model Year", y_title="VDE [MJ/km]")
        self.assertEqual(len(fig.data), 1)
        self.assertEqual(list(fig.data[0].x), [2010, 2020])
        self.assertEqual(fig.data[0].mode, "lines+markers")

    def test_grouped_rows_produce_one_trace_per_group(self):
        rows = [
            ExploreLineRow(identity="vde:1", x=2010, x_label="2010", y=1.0, formatted_y="1.0", role="REFERENCE", group="EPA"),
            ExploreLineRow(identity="vde:2", x=2020, x_label="2020", y=1.1, formatted_y="1.1", role="COMPARISON", group="WLTP"),
        ]
        fig = build_explore_line(rows, x_title="Model Year", y_title="y")
        self.assertEqual({t.name for t in fig.data}, {"EPA", "WLTP"})


class LineageWaterfallChartTests(unittest.TestCase):
    def _step(self, vde_id, label, parent_vde_id, value, delta, semantic, status="OK"):
        return LineageStep(
            vde_id=vde_id, label=label, parent_vde_id=parent_vde_id, provenance="ESTIMATED",
            value=value, formatted_value="-" if value is None else f"{value:.3f}",
            delta=delta, formatted_delta=None if delta is None else f"{delta:+.3f}",
            semantic=semantic, status=status,
        )

    def test_no_ok_steps_returns_empty_figure_without_crash(self):
        steps = [self._step(1, "Root", None, None, None, None, status="UNAVAILABLE")]
        fig = build_lineage_waterfall_chart(steps, y_title="VDE TOTAL [MJ/km]")
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 0)

    def test_single_baseline_step_is_absolute_only_no_total_bar(self):
        steps = [self._step(1, "Root", None, 1.24, None, None)]
        fig = build_lineage_waterfall_chart(steps, y_title="VDE TOTAL [MJ/km]")
        self.assertEqual(len(fig.data), 1)
        waterfall_trace = fig.data[0]
        self.assertEqual(list(waterfall_trace.measure), ["absolute"])

    def test_multi_step_chain_appends_total_bar(self):
        steps = [
            self._step(1, "Root", None, 1.24, None, None),
            self._step(2, "Child", 1, 1.21, -0.03, "BETTER"),
        ]
        fig = build_lineage_waterfall_chart(steps, y_title="VDE TOTAL [MJ/km]")
        waterfall_trace = fig.data[0]
        self.assertEqual(list(waterfall_trace.measure), ["absolute", "relative", "total"])
        self.assertAlmostEqual(waterfall_trace.y[-1], 1.21, places=6)

    def test_trailing_incomplete_step_excluded_from_plotted_bars(self):
        steps = [
            self._step(1, "Root", None, 1.24, None, None),
            self._step(2, "Child", 1, None, None, None, status="UNAVAILABLE"),
        ]
        fig = build_lineage_waterfall_chart(steps, y_title="VDE TOTAL [MJ/km]")
        waterfall_trace = fig.data[0]
        self.assertEqual(len(waterfall_trace.x), 1)  # only the baseline is plotted


if __name__ == "__main__":
    unittest.main()
