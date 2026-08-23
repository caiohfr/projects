# src/vde_core/comparison_metric_registry.py
# -----------------------------------------------------------------------------
# Package 8A - a deliberately small canonical metric catalog for the
# Comparison data layer. Not a generic BI framework: just enough per metric
# to support direction (better/worse), compatibility, and chart hints for
# later Scorecard/Dashboard packages.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MetricDirection(str, Enum):
    LOWER_IS_BETTER = "LOWER_IS_BETTER"
    HIGHER_IS_BETTER = "HIGHER_IS_BETTER"
    NEUTRAL = "NEUTRAL"
    CONTEXT_DEPENDENT = "CONTEXT_DEPENDENT"

    def __str__(self) -> str:
        return self.value


class ComparisonRule(str, Enum):
    ALWAYS = "ALWAYS"
    BASIS_METADATA = "BASIS_METADATA"
    SAME_LEGISLATION_CYCLE = "SAME_LEGISLATION_CYCLE"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class MetricDefinition:
    key: str
    label: str
    group: str
    unit_family: str
    direction: MetricDirection
    source_requirement: str
    compatible_chart_types: tuple[str, ...]
    comparison_rule: ComparisonRule


_METRICS: tuple[MetricDefinition, ...] = (
    # --- Vehicle -------------------------------------------------------
    MetricDefinition("make", "Make", "Vehicle", "text", MetricDirection.NEUTRAL, "VEHICLE", ("table",), ComparisonRule.ALWAYS),
    MetricDefinition("model", "Model", "Vehicle", "text", MetricDirection.NEUTRAL, "VEHICLE", ("table",), ComparisonRule.ALWAYS),
    MetricDefinition("year", "Year", "Vehicle", "count", MetricDirection.NEUTRAL, "VEHICLE", ("table",), ComparisonRule.ALWAYS),
    MetricDefinition("category", "Category", "Vehicle", "text", MetricDirection.NEUTRAL, "VEHICLE", ("table",), ComparisonRule.ALWAYS),
    MetricDefinition("legislation", "Legislation", "Vehicle", "text", MetricDirection.NEUTRAL, "VEHICLE", ("table",), ComparisonRule.ALWAYS),
    MetricDefinition("electrification", "Electrification", "Vehicle", "text", MetricDirection.NEUTRAL, "VEHICLE", ("table",), ComparisonRule.ALWAYS),
    MetricDefinition("drive_type", "Drive type", "Vehicle", "text", MetricDirection.NEUTRAL, "VEHICLE", ("table",), ComparisonRule.ALWAYS),
    MetricDefinition("transmission_type", "Transmission", "Vehicle", "text", MetricDirection.NEUTRAL, "VEHICLE", ("table",), ComparisonRule.ALWAYS),
    # --- Physical setup --------------------------------------------------
    MetricDefinition("mass_kg", "Mass", "Physical setup", "mass_kg", MetricDirection.NEUTRAL, "VEHICLE", ("table", "bar"), ComparisonRule.BASIS_METADATA),
    MetricDefinition("test_mass_kg", "Test mass", "Physical setup", "mass_kg", MetricDirection.NEUTRAL, "VEHICLE", ("table", "bar"), ComparisonRule.BASIS_METADATA),
    MetricDefinition("cda_m2", "CdA", "Physical setup", "area_m2", MetricDirection.LOWER_IS_BETTER, "VEHICLE", ("table", "bar"), ComparisonRule.ALWAYS),
    MetricDefinition("rrc_n_per_kn", "RRC", "Physical setup", "rrc_n_per_kn", MetricDirection.LOWER_IS_BETTER, "VEHICLE", ("table", "bar"), ComparisonRule.ALWAYS),
    # --- Roadload (raw coefficients, no automatic verdict) --------------
    MetricDefinition("roadload_a_total", "A (TOTAL)", "Roadload", "force_n", MetricDirection.NEUTRAL, "ROADLOAD_TOTAL", ("table",), ComparisonRule.SAME_LEGISLATION_CYCLE),
    MetricDefinition("roadload_b_total", "B (TOTAL)", "Roadload", "force_n_per_kph", MetricDirection.NEUTRAL, "ROADLOAD_TOTAL", ("table",), ComparisonRule.SAME_LEGISLATION_CYCLE),
    MetricDefinition("roadload_c_total", "C (TOTAL)", "Roadload", "force_n_per_kph2", MetricDirection.NEUTRAL, "ROADLOAD_TOTAL", ("table",), ComparisonRule.SAME_LEGISLATION_CYCLE),
    MetricDefinition("roadload_a_net", "A (NET)", "Roadload", "force_n", MetricDirection.NEUTRAL, "ROADLOAD_NET", ("table",), ComparisonRule.SAME_LEGISLATION_CYCLE),
    MetricDefinition("roadload_b_net", "B (NET)", "Roadload", "force_n_per_kph", MetricDirection.NEUTRAL, "ROADLOAD_NET", ("table",), ComparisonRule.SAME_LEGISLATION_CYCLE),
    MetricDefinition("roadload_c_net", "C (NET)", "Roadload", "force_n_per_kph2", MetricDirection.NEUTRAL, "ROADLOAD_NET", ("table",), ComparisonRule.SAME_LEGISLATION_CYCLE),
    # --- Vehicle demand ---------------------------------------------------
    MetricDefinition("vde_total", "VDE TOTAL", "Vehicle demand", "energy_mj_per_km", MetricDirection.LOWER_IS_BETTER, "VDE_TOTAL", ("table", "bar", "scatter"), ComparisonRule.SAME_LEGISLATION_CYCLE),
    MetricDefinition("vde_net", "VDE NET", "Vehicle demand", "energy_mj_per_km", MetricDirection.LOWER_IS_BETTER, "VDE_NET", ("table", "bar", "scatter"), ComparisonRule.SAME_LEGISLATION_CYCLE),
    # --- Fuel / Energy / CO2 (existing FuelCons outputs only) ------------
    MetricDefinition("fuel_l_per_100km", "Fuel consumption", "Fuel / Energy / CO2", "l_per_100km", MetricDirection.LOWER_IS_BETTER, "FUEL_CONSUMPTION", ("table", "bar"), ComparisonRule.ALWAYS),
    MetricDefinition("fuel_km_per_l", "Fuel economy", "Fuel / Energy / CO2", "km_per_l", MetricDirection.HIGHER_IS_BETTER, "FUEL_CONSUMPTION", ("table", "bar"), ComparisonRule.ALWAYS),
    MetricDefinition("energy_wh_per_km", "Energy consumption", "Fuel / Energy / CO2", "wh_per_km", MetricDirection.LOWER_IS_BETTER, "FUEL_ENERGY", ("table", "bar", "scatter"), ComparisonRule.ALWAYS),
    MetricDefinition("gco2_per_km", "CO2", "Fuel / Energy / CO2", "gco2_per_km", MetricDirection.LOWER_IS_BETTER, "CO2", ("table", "bar"), ComparisonRule.ALWAYS),
    # --- Efficiency (existing PSE / effective efficiency metrics only) --
    MetricDefinition("eta_pt_est", "Estimated powertrain efficiency", "Efficiency", "ratio", MetricDirection.HIGHER_IS_BETTER, "PSE", ("table", "bar"), ComparisonRule.ALWAYS),
)

_METRICS_BY_KEY: dict[str, MetricDefinition] = {m.key: m for m in _METRICS}


def get_metric(key: str) -> MetricDefinition | None:
    return _METRICS_BY_KEY.get(key)


def list_metrics(group: str | None = None) -> tuple[MetricDefinition, ...]:
    if group is None:
        return _METRICS
    return tuple(m for m in _METRICS if m.group == group)


__all__ = [
    "MetricDirection",
    "ComparisonRule",
    "MetricDefinition",
    "get_metric",
    "list_metrics",
]
