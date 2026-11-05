"""Canonical persisted-VDE calculation-mass resolution.

This module is deliberately small and repository-free so every Vehicle
Demand consumer can resolve the same stored physical state without UI logic
or Proposal Matrix re-execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ResolvedCalculationMass:
    mass_kg: float | None
    basis: str | None
    source_field: str | None
    epa_mass_is_resolved_twc: bool = False


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _positive_float(value: Any) -> float | None:
    result = _float_or_none(value)
    return result if result is not None and math.isfinite(result) and result > 0 else None


def _text(value: Any) -> str | None:
    result = str(value or "").strip().upper()
    return result or None


def _uses_regulatory_inertia(row: Mapping[str, Any]) -> bool:
    basis_tokens = {
        _text(row.get("vde_mass_basis")),
        _text(row.get("test_mass_basis")),
    }
    if any(token and ("INERTIA" in token or token in {"TWC", "EPA_TWC"}) for token in basis_tokens):
        return True
    intention = _text(row.get("mass_intention"))
    return bool(intention and intention.startswith("EPA_"))


def resolve_vde_calculation_mass(row: Mapping[str, Any]) -> ResolvedCalculationMass:
    """Resolve the mass actually used by Vehicle Demand, with provenance.

    Precedence follows persisted semantics rather than legislation alone:
    explicit canonical VDE mass; an explicitly selected regulatory inertia
    representation; stored test/calculation mass; then, only when no stored
    test mass exists, an EPA inertia class; curb mass as final fallback.
    """
    legislation = _text(row.get("legislation"))
    canonical_raw = row.get("vde_calculation_mass_kg")
    canonical = _float_or_none(canonical_raw)
    if canonical_raw not in (None, ""):
        return ResolvedCalculationMass(
            canonical,
            _text(row.get("vde_mass_basis")) or _text(row.get("test_mass_basis")) or "CANONICAL_VDE_MASS",
            "vde_calculation_mass_kg",
            legislation == "EPA",
        )

    stored_test_raw = row.get("test_mass_kg")
    stored_test_mass = _float_or_none(stored_test_raw)
    if stored_test_raw not in (None, "") and (
        stored_test_mass is None or not math.isfinite(stored_test_mass) or stored_test_mass <= 0
    ):
        return ResolvedCalculationMass(
            stored_test_mass,
            _text(row.get("test_mass_basis")) or "STORED_TEST_MASS",
            "test_mass_kg",
            legislation == "EPA",
        )

    inertia = _positive_float(row.get("inertia_class"))
    if inertia is not None and _uses_regulatory_inertia(row):
        return ResolvedCalculationMass(
            inertia,
            _text(row.get("test_mass_basis")) or _text(row.get("vde_mass_basis")) or "EPA_INERTIA_CLASS",
            "inertia_class",
            legislation == "EPA",
        )

    if stored_test_mass is not None:
        return ResolvedCalculationMass(
            stored_test_mass,
            _text(row.get("test_mass_basis")) or "STORED_TEST_MASS",
            "test_mass_kg",
            legislation == "EPA",
        )

    # A persisted EPA inertia class is a valid regulatory representation when
    # no more explicit calculation/test mass exists. It does not supersede an
    # independently stored test mass merely because legislation is EPA.
    if legislation == "EPA" and inertia is not None:
        return ResolvedCalculationMass(
            inertia,
            _text(row.get("test_mass_basis")) or "EPA_INERTIA_CLASS",
            "inertia_class",
            True,
        )

    curb_raw = row.get("mass_kg")
    curb = _float_or_none(curb_raw)
    if curb_raw not in (None, ""):
        return ResolvedCalculationMass(curb, "CURB_FALLBACK", "mass_kg", False)
    return ResolvedCalculationMass(None, None, None, False)


__all__ = ["ResolvedCalculationMass", "resolve_vde_calculation_mass"]
