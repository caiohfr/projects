from __future__ import annotations

import unittest

import plotly.graph_objects as go

from src.vde_app.comparison_report_charts import (
    build_cycle_demand_figure,
    build_fe_vde_figure,
    build_grouped_bar_figure,
)


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


if __name__ == "__main__":
    unittest.main()
