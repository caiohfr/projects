"""
Application-facing helpers for RoadLoad integration.

This module keeps lightweight orchestration logic out of Streamlit pages while
still relying on the canonical RoadLoad adapter + engine pipeline.

This is the preferred bridge from UI-like context dicts into the new RoadLoad
package. For historical reference utilities, see ``services.py`` and
``physics.py`` in the same package.
"""

import math

from .adapters import build_request_from_manual_inputs
from .engine import run_roadload_scenario


def _to_float(value, default=None):
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        out = float(value)
        if math.isnan(out):
            return default
        return out
    except Exception:
        return default


def resolve_equiv_from_ctx(ctx: dict):
    """
    Build a RoadLoadRequest from a UI-like context dict and return EquivalentABC.
    """
    delta_rr = _to_float(ctx.get("delta_rr_N"), 0.0)
    frac120 = _to_float(ctx.get("crr1_frac_at_120kph"), 0.0)
    delta_rr_B = delta_rr * (frac120 / 120.0) if frac120 else 0.0

    req = build_request_from_manual_inputs(
        A=_to_float(ctx.get("A"), 0.0),
        B=_to_float(ctx.get("B"), 0.0),
        C=_to_float(ctx.get("C"), 0.0),
        mass_kg=_to_float(ctx.get("mass_kg"), 1500.0),
        legislation=ctx.get("legislation"),
        category=ctx.get("category"),
        source="vde_setup_ctx",
        delta_mass_kg=_to_float(ctx.get("delta_mass_kg"), 0.0),
        tire_improve_pct=_to_float(ctx.get("tire_improve_pct"), 0.0),
        tire_delta_A=delta_rr,
        tire_delta_B=delta_rr_B,
        delta_cda_m2=_to_float(ctx.get("delta_aero_cdA"), 0.0),
        brake_delta_A=_to_float(ctx.get("delta_brake_N"), 0.0),
        parasitic_delta_A=_to_float(ctx.get("delta_parasitics_N"), 0.0),
    )
    return run_roadload_scenario(req)
