"""Deterministic, diagnostic RRC prior resolution for temporary analysis.

This module extracts the established Component Research v0.4 rule semantics
into a reusable core owner. It performs no database access and deliberately
preserves the EPA category vocabulary used by the defaults CSV.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.vde_core.roadload.tire_model import G_MPS2
from src.vde_core.services import load_vde_defaults


DEFAULT_RRC_PRIOR_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "standards"
    / "vde_defaults_by_category_trans_elec.csv"
)
GENERIC_CATEGORIES = frozenset(
    {"CAR", "CARS", "SUV", "TRUCK", "VEHICLE", "UNKNOWN", "UNSPECIFIED", "OTHER"}
)
_EXPLICIT_ELECTRIFICATION = frozenset({"ICE", "HEV", "PHEV", "BEV", "MHEV"})


@dataclass(frozen=True)
class RollingRrcPriorResolution:
    raw_category: str | None
    lookup_category: str | None
    raw_electrification: str | None
    resolved_electrification: str | None
    electrification_source: str
    raw_transmission: str | None
    gears: int | None
    resolved_transmission_key: str
    attempted_rule_key: str
    matched_rule_key: str | None
    rrc_n_per_kn: float | None
    crr1_frac_at_120kph: float | None
    status: str
    source: str
    missing_discriminants: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class RollingRrcMassDelta:
    baseline_A_N: float
    baseline_B_N_per_kph: float
    scenario_A_N: float
    scenario_B_N_per_kph: float
    A_N: float
    B_N_per_kph: float
    C_N_per_kph2: float = 0.0


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _canonical_whitespace(value: Any) -> str | None:
    text = _clean(value)
    return re.sub(r"\s+", " ", text) if text else None


def _first_present(context: Mapping[str, Any], *keys: str) -> tuple[str | None, str | None]:
    for key in keys:
        value = _clean(context.get(key))
        if value is not None:
            return value, key
    return None, None


def _linked_contexts(
    context: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None,
) -> tuple[Mapping[str, Any], ...]:
    if context is None:
        return ()
    if isinstance(context, Mapping):
        return (context,)
    return tuple(item for item in context if isinstance(item, Mapping))


def _normalize_electrification(value: Any, *, engine_type: bool = False) -> str | None:
    raw = _canonical_whitespace(value)
    if raw is None:
        return None
    normalized = raw.upper().replace("-", " ")
    if normalized in _EXPLICIT_ELECTRIFICATION:
        return normalized
    if normalized == "NONE":
        return "ICE"
    if engine_type and normalized in {"SI", "CI", "ICE"}:
        return "ICE"
    return None


def _resolve_electrification(
    vde_context: Mapping[str, Any],
    linked_context: tuple[Mapping[str, Any], ...],
) -> tuple[str | None, str | None, str, bool]:
    raw, key = _first_present(vde_context, "electrification", "powertrain_electrification")
    if raw is not None:
        resolved = _normalize_electrification(raw)
        if resolved is not None:
            return raw, resolved, f"vde.{key}", False

    engine_raw, engine_key = _first_present(vde_context, "engine_type")
    if engine_raw is not None:
        resolved = _normalize_electrification(engine_raw, engine_type=True)
        if resolved is not None:
            return raw or engine_raw, resolved, f"vde.{engine_key}", False

    linked_raw = [
        raw_value
        for item in linked_context
        for raw_value, _key in [_first_present(item, "electrification", "powertrain_electrification")]
        if raw_value is not None
    ]
    normalized_linked = [_normalize_electrification(value) for value in linked_raw]
    linked_resolved = {value for value in normalized_linked if value is not None}
    if linked_raw and None not in normalized_linked and len(linked_resolved) == 1:
        return raw or " | ".join(dict.fromkeys(linked_raw)), next(iter(linked_resolved)), "linked_context.electrification", False
    if linked_raw:
        return raw or " | ".join(dict.fromkeys(linked_raw)), None, "linked_context.electrification:CONFLICT", True

    model = _canonical_whitespace(vde_context.get("model")) or ""
    if model.upper().startswith("I4"):
        return raw or engine_raw, "BEV", "V0.4_MODEL_I4_FALLBACK", False
    return raw or engine_raw, None, "UNAVAILABLE", False


def _parse_gears(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
        return None
    return int(numeric)


def _resolve_gears(
    vde_context: Mapping[str, Any],
    linked_context: tuple[Mapping[str, Any], ...],
) -> tuple[int | None, bool]:
    direct_raw, _key = _first_present(vde_context, "gears", "gear_count", "number_of_gears", "num_gears")
    if direct_raw is not None:
        return _parse_gears(direct_raw), False
    values = {
        parsed
        for item in linked_context
        for raw, _key in [_first_present(item, "gears", "gear_count", "number_of_gears", "num_gears")]
        for parsed in [_parse_gears(raw)]
        if parsed is not None
    }
    if len(values) == 1:
        return next(iter(values)), False
    return None, len(values) > 1


def resolve_transmission_rule_key(raw_transmission: Any, gears: Any = None) -> str:
    """Preserve the v0.4 transmission rule and supported legacy labels."""
    if _parse_gears(gears) == 1:
        return "SS"
    raw = _canonical_whitespace(raw_transmission)
    if raw is None:
        return "OT"
    normalized = re.sub(r"[_\-/]+", " ", raw.upper())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if normalized in {"SINGLE SPEED EV", "SINGLE SPEED", "SS"} or "ONE SPEED" in normalized:
        return "SS"
    if normalized in {"DCT", "AMT"} or "DUAL CLUTCH" in normalized or "DOUBLE CLUTCH" in normalized:
        return "AMT"
    if "AUTOMATED MANUAL" in normalized or "SEMI AUTOMATIC" in normalized:
        return "AMT"
    if normalized in {"CVT"} or "CONTINUOUSLY VARIABLE" in normalized:
        return "CVT"
    if normalized in {"MANUAL", "MANUAL TRANSMISSION", "MT"} or re.fullmatch(
        r"\d+ SPEED MANUAL", normalized
    ):
        return "MT"
    if normalized in {"TORQUE CONVERTER AUTOMATIC", "AT", "AUTO", "AUTOMATIC"} or any(
        marker in normalized
        for marker in ("TORQUE CONVERTER", "STEPTRONIC", "SPORT AUTO", "SPORT AUTOMATIC")
    ):
        return "AT"
    return "OT"


def _attempted_key(category: str | None, electrification: str | None, transmission: str) -> str:
    return "|".join((category or "<missing>", electrification or "<missing>", transmission))


def resolve_rolling_rrc_prior(
    vde_context: Mapping[str, Any],
    *,
    linked_context: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    defaults_path: str | Path = DEFAULT_RRC_PRIOR_PATH,
) -> RollingRrcPriorResolution:
    """Resolve only an exact EPA-category/electrification/transmission prior."""
    raw_category = _clean(vde_context.get("category"))
    lookup_category = _canonical_whitespace(raw_category)
    linked = _linked_contexts(linked_context)
    raw_electrification, electrification, electrification_source, electrification_conflict = (
        _resolve_electrification(vde_context, linked)
    )
    raw_transmission, _transmission_key = _first_present(vde_context, "transmission_type", "transmission")
    transmission_conflict = False
    if raw_transmission is None:
        linked_transmissions = [
            value
            for item in linked
            for value, _key in [_first_present(item, "transmission_type", "transmission")]
            if value is not None
        ]
        distinct_transmissions = {value.casefold() for value in linked_transmissions}
        if len(distinct_transmissions) == 1:
            raw_transmission = linked_transmissions[0]
        elif len(distinct_transmissions) > 1:
            transmission_conflict = True
    gears, gears_conflict = _resolve_gears(vde_context, linked)
    transmission = resolve_transmission_rule_key(raw_transmission, gears)
    attempted = _attempted_key(lookup_category, electrification, transmission)

    missing = []
    if lookup_category is None:
        missing.append("category")
    if electrification is None:
        missing.append("electrification")
    if gears_conflict:
        missing.append("gears")
    if transmission_conflict:
        missing.append("transmission")
    if electrification_conflict and "electrification" not in missing:
        missing.append("electrification")

    common = dict(
        raw_category=raw_category,
        lookup_category=lookup_category,
        raw_electrification=raw_electrification,
        resolved_electrification=electrification,
        electrification_source=electrification_source,
        raw_transmission=raw_transmission,
        gears=gears,
        resolved_transmission_key=transmission,
        attempted_rule_key=attempted,
        source="RULE_ESTIMATED",
    )
    if missing:
        return RollingRrcPriorResolution(
            **common,
            matched_rule_key=None,
            rrc_n_per_kn=None,
            crr1_frac_at_120kph=None,
            status="UNAVAILABLE",
            missing_discriminants=tuple(missing),
            reason=f"Missing or conflicting discriminants: {', '.join(missing)}.",
        )

    category_key = lookup_category.upper()
    if category_key in GENERIC_CATEGORIES:
        return RollingRrcPriorResolution(
            **common,
            matched_rule_key=None,
            rrc_n_per_kn=None,
            crr1_frac_at_120kph=None,
            status="UNAVAILABLE",
            missing_discriminants=(),
            reason="Category is generic and is not eligible for a numeric RRC prior.",
        )

    defaults = load_vde_defaults(defaults_path)
    category_series = defaults["category"].astype(str).map(_canonical_whitespace)
    exact = defaults[
        category_series.str.casefold().eq(lookup_category.casefold())
        & defaults["electrification"].astype(str).str.strip().str.casefold().eq(electrification.casefold())
        & defaults["transmission_type"].astype(str).str.strip().str.casefold().eq(transmission.casefold())
    ]
    if len(exact.index) != 1:
        categories = {str(value).casefold() for value in category_series if value is not None}
        reason = (
            "Category vocabulary is incompatible with defaults table."
            if lookup_category.casefold() not in categories
            else "No exact category/electrification/transmission row exists in defaults table."
        )
        return RollingRrcPriorResolution(
            **common,
            matched_rule_key=None,
            rrc_n_per_kn=None,
            crr1_frac_at_120kph=None,
            status="UNAVAILABLE",
            missing_discriminants=(),
            reason=reason,
        )

    row = exact.iloc[0]
    rrc = float(row["rrc_N_per_kN"])
    crr1 = float(row["crr1_frac_at_120kph"])
    if not math.isfinite(rrc) or rrc < 0 or not math.isfinite(crr1):
        return RollingRrcPriorResolution(
            **common,
            matched_rule_key=None,
            rrc_n_per_kn=None,
            crr1_frac_at_120kph=None,
            status="UNAVAILABLE",
            missing_discriminants=(),
            reason="Exact defaults row contains an invalid RRC or speed correction.",
        )
    matched_key = "|".join(
        (
            str(row["category"]).strip(),
            str(row["electrification"]).strip(),
            str(row["transmission_type"]).strip(),
        )
    )
    return RollingRrcPriorResolution(
        **common,
        matched_rule_key=matched_key,
        rrc_n_per_kn=rrc,
        crr1_frac_at_120kph=crr1,
        status="APPLIED",
        missing_discriminants=(),
        reason="Exact category/electrification/transmission prior matched.",
    )


def calculate_rolling_rrc_mass_delta(
    rrc_n_per_kn: float,
    baseline_vehicle_mass_kg: float,
    scenario_vehicle_mass_kg: float,
    crr1_frac_at_120kph: float = 0.0,
) -> RollingRrcMassDelta:
    """Calculate only the rolling ABC delta owned by an RRC mass change."""
    rrc = float(rrc_n_per_kn)
    baseline_mass = float(baseline_vehicle_mass_kg)
    scenario_mass = float(scenario_vehicle_mass_kg)
    crr1 = float(crr1_frac_at_120kph)
    if not all(math.isfinite(value) for value in (rrc, baseline_mass, scenario_mass, crr1)):
        raise ValueError("RRC rolling-delta inputs must be finite.")
    if rrc < 0 or baseline_mass <= 0 or scenario_mass <= 0:
        raise ValueError("RRC must be non-negative and rolling masses must be positive.")
    baseline_A = rrc * G_MPS2 * baseline_mass / 1000.0
    scenario_A = rrc * G_MPS2 * scenario_mass / 1000.0
    baseline_B = baseline_A * crr1 / 120.0
    scenario_B = scenario_A * crr1 / 120.0
    return RollingRrcMassDelta(
        baseline_A_N=baseline_A,
        baseline_B_N_per_kph=baseline_B,
        scenario_A_N=scenario_A,
        scenario_B_N_per_kph=scenario_B,
        A_N=scenario_A - baseline_A,
        B_N_per_kph=scenario_B - baseline_B,
    )


__all__ = [
    "DEFAULT_RRC_PRIOR_PATH",
    "RollingRrcMassDelta",
    "RollingRrcPriorResolution",
    "calculate_rolling_rrc_mass_delta",
    "resolve_rolling_rrc_prior",
    "resolve_transmission_rule_key",
]
