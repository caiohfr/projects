# src/vde_core/technology_delta.py
# -----------------------------------------------------------------------------
# Sprint 10D - canonical, Streamlit-free Technology Delta vocabulary and
# stacking math.
#
# This module is an EXTRACTION, not a new schema: `apply_delta_stack_to_baseline`,
# `normalize_delta_effect_basis`, `maturity_rank`, `delta_status_counts`, and
# `proposal_confidence_label` reproduce, field-for-field and branch-for-branch,
# the existing Powertrain Scenario logic in
# `src/vde_app/components/pwt_fuel_energy.py` (`_apply_delta_stack_to_baseline`
# et al., lines ~863-1099 as audited in Sprint 10A/10D). That module is not
# imported here or modified by this module -- it `import streamlit` at module
# scope, so a Streamlit-free Quick Scenario resolver cannot import from it
# directly (Sprint 10 Stop Condition 4: "Technology Delta canonical math
# cannot be reused without importing a large Streamlit/UI component into
# core... prefer extracting the smallest pure canonical calculation to a
# shared core module while preserving exact behavior"). This module is that
# extraction. `pwt_fuel_energy.py` is left untouched; its own Technology
# Delta workspace continues to use its own copy of this logic unchanged, so
# existing Powertrain Scenario behavior carries zero risk from this change.
#
# The one intentional, behavior-neutral simplification: the original
# `_apply_delta_stack_to_baseline` decorates `baseline["method"]`/
# `proposal["method"]` with a display label via `_pwt_method_label`, whose
# only Streamlit coupling is an unreachable-in-practice `st.session_state`
# fallback (only hit when `baseline_result.method` is falsy, which never
# happens for a real result). That field is never read by the numeric
# stacking logic itself, so this module stores the raw `method` string
# instead of a decorated label -- every numeric branch, every stacking rule,
# and the CO2-recompute-clobbers-CO2-delta quirk documented below are
# reproduced exactly as-is, not "fixed".
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Mapping

from src.vde_core.fuel_energy import GCO2_PER_L, LHV_MJ_PER_L, MJ_TO_Wh
from src.vde_core.vde_setup_service import to_float

DELTA_SUBSYSTEM_OPTIONS = [
    "engine",
    "transmission",
    "hybrid/ESS",
    "electrical/alternator",
    "auxiliary loads",
    "calibration",
    "thermal",
    "fuel system",
    "whole powertrain",
]

DELTA_SOURCE_TYPE_OPTIONS = [
    "engineering_assumption",
    "supplier_data",
    "imported_map",
    "simulation_result",
    "map_analysis",
    "test_data",
    "manual",
    "metadata_only",
]

DELTA_MATURITY_OPTIONS = [
    "metadata_only",
    "engineering_assumption",
    "supplier_data",
    "imported_map",
    "simulation_ready",
    "simulation_result",
    "correlated_model",
    "validated_against_test",
]

DELTA_EFFECT_BASIS_OPTIONS = [
    "Fuel consumption percent delta",
    "PSE percent delta",
    "CO2 percent delta",
    "Efficiency multiplier",
    "Metadata-only / registered-only",
]

DELTA_EFFECT_BASIS_ADVANCED_OPTIONS = [
    "PSE delta",
    "PSE multiplier",
    "fuel delta",
    "CO2 delta",
    "energy delta",
    "map-based effect",
]

DELTA_CONFIDENCE_OPTIONS = ["unknown", "low", "medium", "high"]

_EFFECT_BASIS_LABEL_TO_KEY = {
    "Fuel consumption percent delta": "fuel_percent_delta",
    "PSE percent delta": "pse_percent_delta",
    "CO2 percent delta": "co2_percent_delta",
    "Efficiency multiplier": "efficiency_multiplier",
    "Metadata-only / registered-only": "metadata_only",
    "fuel delta": "fuel_delta",
    "fuel percent delta": "fuel_percent_delta",
    "PSE delta": "pse_delta",
    "PSE multiplier": "pse_multiplier",
    "CO2 delta": "co2_delta",
    "energy delta": "energy_delta",
    "efficiency multiplier": "efficiency_multiplier",
    "metadata only": "metadata_only",
    "map-based effect": "map_based_effect",
}


def normalize_delta_effect_basis(effect_basis: str | None) -> str:
    """Verbatim extraction of `pwt_fuel_energy._normalize_delta_effect_basis`."""

    text = str(effect_basis or "").strip()
    return _EFFECT_BASIS_LABEL_TO_KEY.get(text, text)


def maturity_rank(level: str | None) -> int:
    """Verbatim extraction of `pwt_fuel_energy._maturity_rank`."""

    try:
        return DELTA_MATURITY_OPTIONS.index(str(level or "").strip())
    except ValueError:
        return -1


def normalize_technology_delta(raw: Mapping[str, Any], *, index: int = 1) -> dict[str, Any]:
    """Verbatim extraction of the per-delta normalization body of
    `pwt_fuel_energy._technology_deltas` (its session-state sourcing and
    live-form-preview behavior are UI concerns and are not reproduced here
    -- callers supply the raw delta dicts directly).
    """

    row = dict(raw or {})
    row.setdefault("id", index)
    row.setdefault("name", f"Delta {index}")
    row.setdefault("affected_subsystem", "whole powertrain")
    row.setdefault("source_type", "manual")
    row.setdefault("maturity_level", "engineering_assumption")
    row.setdefault("effect_basis", "metadata only")
    row.setdefault("confidence", "unknown")
    row.setdefault("enabled", True)
    row.setdefault("notes", "")
    row.setdefault("reference_description", "")

    effect_value = to_float(row.get("effect_value"))
    row["effect_value"] = effect_value
    effect_basis = normalize_delta_effect_basis(row.get("effect_basis") or "metadata_only")
    row["effect_basis"] = effect_basis

    if not bool(row.get("enabled")):
        row["quantitative_status"] = "disabled"
    elif effect_basis in {"map_based_effect"}:
        row["quantitative_status"] = "pending_model"
    elif effect_basis in {"metadata_only"} or str(row.get("source_type") or "") == "metadata_only":
        row["quantitative_status"] = "registered_only"
    elif effect_value is None:
        row["quantitative_status"] = "registered_only"
    else:
        row["quantitative_status"] = "applied"
    return row


def delta_status_counts(deltas: list[Mapping[str, Any]]) -> dict[str, int]:
    """Verbatim extraction of `pwt_fuel_energy._delta_status_counts`."""

    summary = {"applied": 0, "registered_only": 0, "pending_model": 0, "disabled": 0}
    for delta in deltas:
        status = str(delta.get("quantitative_status") or "registered_only")
        if status in summary:
            summary[status] += 1
    return summary


def proposal_confidence_label(baseline_confidence: str | None, deltas: list[Mapping[str, Any]]) -> str:
    """Verbatim extraction of `pwt_fuel_energy._proposal_confidence_label`."""

    level = str(baseline_confidence or "low").strip().lower()
    if any(
        str(delta.get("confidence") or "").lower() == "low"
        for delta in deltas
        if delta.get("quantitative_status") == "applied"
    ):
        return "low"
    if any(str(delta.get("quantitative_status") or "") in {"registered_only", "pending_model"} for delta in deltas):
        return "medium" if level == "high" else level or "medium"
    return level or "low"


def apply_delta_stack_to_baseline(
    baseline_result: Any,
    *,
    ctx: dict[str, Any],
    deltas: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verbatim extraction of `pwt_fuel_energy._apply_delta_stack_to_baseline`.

    `baseline_result` is a `FuelEstimateResult` (or any object exposing the
    same `.assumptions`/`.fuel_l_100km`/`.energy_Wh_km`/`.gco2_km`/`.method`/
    `.confidence`/`.request` attributes). `ctx` must carry
    `"energy_value_mj_per_km"` (the active demand, MJ/km, for the basis the
    result was computed on). Every stacking rule -- additive vs.
    multiplicative/compounding per `effect_basis`, the post-loop PSE/fuel/
    CO2/energy reconciliation, and the resulting CO2-delta-gets-recomputed-
    away-by-the-fuel-reconciliation-step quirk -- is reproduced exactly as
    audited; none of it is "corrected" here.
    """

    if baseline_result is None:
        return {
            "status": "Proposal pending",
            "baseline": {},
            "proposal": {},
            "applied_deltas": [],
            "registered_only_deltas": list(deltas),
            "confidence": "low",
            "warnings": ["baseline_pending"],
            "delta_counts": delta_status_counts(deltas),
        }

    assumptions = dict((baseline_result.assumptions or {}) or {})
    pse_summary = dict(assumptions.get("pse_summary") or {})
    demand_mj_per_km = to_float(ctx.get("energy_value_mj_per_km"))
    baseline = {
        "pse": to_float(pse_summary.get("value")),
        "fuel_l_100km": to_float(baseline_result.fuel_l_100km),
        "energy_Wh_km": to_float(baseline_result.energy_Wh_km),
        "gco2_km": to_float(baseline_result.gco2_km),
        "method": str(baseline_result.method or "-"),
        "confidence": str(baseline_result.confidence or "-"),
    }
    proposal = dict(baseline)
    applied_deltas: list[dict[str, Any]] = []
    registered_only: list[dict[str, Any]] = []
    warnings: list[str] = []
    fuel_type = str(baseline_result.request.powertrain_features.get("fuel_type") or "Gasoline")
    lhv = float(
        baseline_result.request.powertrain_features.get("LHV_MJ_per_L")
        or LHV_MJ_PER_L.get(fuel_type, LHV_MJ_PER_L["Gasoline"])
    )
    gco2_per_l = float(
        baseline_result.request.powertrain_features.get("gCO2_per_L")
        or GCO2_PER_L.get(fuel_type, GCO2_PER_L["Gasoline"])
    )

    for delta in deltas:
        status = str(delta.get("quantitative_status") or "registered_only")
        if status != "applied":
            registered_only.append(delta)
            continue
        effect_basis = normalize_delta_effect_basis(delta.get("effect_basis") or "")
        value = to_float(delta.get("effect_value"))
        if value is None:
            registered_only.append(delta)
            continue
        if effect_basis == "pse_delta" and proposal.get("pse") is not None:
            proposal["pse"] = float(proposal["pse"]) + float(value)
        elif effect_basis == "pse_percent_delta" and proposal.get("pse") is not None:
            proposal["pse"] = float(proposal["pse"]) * (1.0 + float(value) / 100.0)
        elif effect_basis in {"pse_multiplier", "efficiency_multiplier"} and proposal.get("pse") is not None:
            proposal["pse"] = float(proposal["pse"]) * float(value)
        elif effect_basis == "fuel_delta" and proposal.get("fuel_l_100km") is not None:
            proposal["fuel_l_100km"] = float(proposal["fuel_l_100km"]) + float(value)
        elif effect_basis == "fuel_percent_delta" and proposal.get("fuel_l_100km") is not None:
            proposal["fuel_l_100km"] = float(proposal["fuel_l_100km"]) * (1.0 + float(value) / 100.0)
        elif effect_basis == "co2_delta" and proposal.get("gco2_km") is not None:
            proposal["gco2_km"] = float(proposal["gco2_km"]) + float(value)
        elif effect_basis == "co2_percent_delta" and proposal.get("gco2_km") is not None:
            proposal["gco2_km"] = float(proposal["gco2_km"]) * (1.0 + float(value) / 100.0)
        elif effect_basis == "energy_delta" and proposal.get("energy_Wh_km") is not None:
            proposal["energy_Wh_km"] = float(proposal["energy_Wh_km"]) + float(value)
        else:
            delta = dict(delta)
            delta["quantitative_status"] = "registered_only"
            registered_only.append(delta)
            continue
        applied_deltas.append(delta)

    if proposal.get("pse") is not None and demand_mj_per_km is not None and proposal["pse"] > 0:
        if baseline_result.request.vehicle_features.get("electrification") == "BEV":
            proposal["energy_Wh_km"] = demand_mj_per_km / proposal["pse"] * MJ_TO_Wh
        elif proposal.get("fuel_l_100km") is None or any(
            normalize_delta_effect_basis(delta.get("effect_basis") or "")
            in {"pse_delta", "pse_multiplier", "efficiency_multiplier", "pse_percent_delta"}
            for delta in applied_deltas
        ):
            proposal["fuel_l_100km"] = (demand_mj_per_km / proposal["pse"]) / lhv * 100.0
        if proposal.get("fuel_l_100km") is not None:
            proposal["gco2_km"] = (proposal["fuel_l_100km"] / 100.0) * gco2_per_l

    if proposal.get("fuel_l_100km") is not None and demand_mj_per_km is not None and lhv > 0:
        consumed_mj = (proposal["fuel_l_100km"] / 100.0) * lhv
        if consumed_mj > 0:
            proposal["pse"] = demand_mj_per_km / consumed_mj
        proposal["gco2_km"] = (proposal["fuel_l_100km"] / 100.0) * gco2_per_l

    if (
        proposal.get("energy_Wh_km") is not None
        and demand_mj_per_km is not None
        and proposal["energy_Wh_km"] > 0
        and baseline_result.request.vehicle_features.get("electrification") == "BEV"
    ):
        proposal["pse"] = demand_mj_per_km / (proposal["energy_Wh_km"] / MJ_TO_Wh)

    counts = delta_status_counts(deltas)
    if not applied_deltas and registered_only:
        status = "No quantitative delta"
        warnings.append("registered_only_deltas")
    elif applied_deltas:
        status = "Estimated"
    else:
        status = "Proposal pending"

    highest_maturity = "-"
    if deltas:
        highest_maturity = max(
            (str(delta.get("maturity_level") or "-") for delta in deltas), key=maturity_rank
        )
    return {
        "status": status,
        "baseline": baseline,
        "proposal": proposal,
        "applied_deltas": applied_deltas,
        "registered_only_deltas": registered_only,
        "confidence": proposal_confidence_label(baseline_result.confidence, deltas),
        "warnings": warnings,
        "delta_counts": counts,
        "highest_maturity": highest_maturity,
    }


__all__ = [
    "DELTA_SUBSYSTEM_OPTIONS",
    "DELTA_SOURCE_TYPE_OPTIONS",
    "DELTA_MATURITY_OPTIONS",
    "DELTA_EFFECT_BASIS_OPTIONS",
    "DELTA_EFFECT_BASIS_ADVANCED_OPTIONS",
    "DELTA_CONFIDENCE_OPTIONS",
    "normalize_delta_effect_basis",
    "maturity_rank",
    "normalize_technology_delta",
    "delta_status_counts",
    "proposal_confidence_label",
    "apply_delta_stack_to_baseline",
]
