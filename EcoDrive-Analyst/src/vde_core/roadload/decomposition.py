"""
Helpers for decomposing current RoadLoad Engine outputs.

These helpers are intentionally small and pure so the package can grow a more
detailed physical component view later without coupling that work to Streamlit
pages or database code.
"""

from __future__ import annotations


def _safe_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def decompose_equivalent_abc(equiv) -> dict:
    """
    Normalize an EquivalentABC-like object into a simple decomposition payload.
    """
    component_table = getattr(equiv, "component_table", None) or []
    return {
        "A": _safe_float(getattr(equiv, "A", None)),
        "B": _safe_float(getattr(equiv, "B", None)),
        "C": _safe_float(getattr(equiv, "C", None)),
        "mass_kg": _safe_float(getattr(equiv, "mass_kg", None)),
        "components": list(component_table),
        "warnings": list(getattr(equiv, "warnings", None) or []),
    }


def component_delta_vs_baseline(baseline, equiv) -> dict:
    """
    Compare a baseline-like object with an EquivalentABC-like object.
    """
    baseline_A = _safe_float(getattr(baseline, "A", None))
    baseline_B = _safe_float(getattr(baseline, "B", None))
    baseline_C = _safe_float(getattr(baseline, "C", None))
    baseline_mass = _safe_float(getattr(baseline, "mass_kg", None))

    equiv_A = _safe_float(getattr(equiv, "A", None))
    equiv_B = _safe_float(getattr(equiv, "B", None))
    equiv_C = _safe_float(getattr(equiv, "C", None))
    equiv_mass = _safe_float(getattr(equiv, "mass_kg", None))

    return {
        "delta_A": equiv_A - baseline_A,
        "delta_B": equiv_B - baseline_B,
        "delta_C": equiv_C - baseline_C,
        "delta_mass_kg": equiv_mass - baseline_mass,
    }
