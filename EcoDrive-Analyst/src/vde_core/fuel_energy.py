from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum


LHV_MJ_PER_L = {
    "Gasoline": 32.0,
    "E10": 31.2,
    "E22": 30.0,
    "E100": 21.2,
    "Diesel": 35.8,
    "Other": 32.0,
}

GCO2_PER_L = {
    "Gasoline": 2310.0,
    "E10": 2270.0,
    "E22": 2200.0,
    "E100": 0.0,
    "Diesel": 2640.0,
    "Other": 2310.0,
}

MJ_TO_Wh = 277.7777777778


# -----------------------------------------------------------------------------
# Fuel-energy-basis resolution (raw label -> canonical family -> LHV)
#
# Repo audit before this patch found three inconsistent Gasoline LHV values:
# this module's LHV_MJ_PER_L (32.0, canonical -- it is what actually backs
# the stored consumption numbers), src/vde_app/derivatives.py (34.2,
# display-only) and src/vde_app/plots.py's _add_eta_lines_ice default (34.2,
# matches derivatives.py, not this module). This resolver reuses ONLY
# LHV_MJ_PER_L above -- it does not introduce a second, conflicting Gasoline
# constant, and it does not touch derivatives.py/plots.py.
#
# Rule: never SILENTLY guess an LHV, but an approved, deterministic,
# traceable assumption is preferable to hiding a result outright. A raw fuel
# label is resolved through, in order: (1) an explicit scenario-supplied LHV
# override, (2) an exact (case/whitespace-normalized) match to this module's
# own canonical vocabulary, (3) a small explicit registry of known
# certification/regional labels that map deterministically to one of this
# module's canonical families, (4) unknown -- returns available=False, never
# a fabricated value. A recognized family's LHV always comes from
# LHV_MJ_PER_L above, never a second table.
# -----------------------------------------------------------------------------


class LhvBasis(str, Enum):
    EXPLICIT = "EXPLICIT"
    SPEC_REFERENCE = "SPEC_REFERENCE"
    CANONICAL_ASSUMPTION = "CANONICAL_ASSUMPTION"
    REGIONAL_ASSUMPTION = "REGIONAL_ASSUMPTION"
    UNKNOWN = "UNKNOWN"

    def __str__(self) -> str:
        return self.value


class FuelConfidence(str, Enum):
    EXPLICIT = "EXPLICIT"
    HIGH = "HIGH"
    ASSUMED = "ASSUMED"
    UNKNOWN = "UNKNOWN"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class FuelEnergyBasis:
    raw_fuel_label: str | None
    canonical_fuel_family: str | None  # e.g. "GASOLINE" / "DIESEL" / "ETHANOL"; None when unresolved
    fuel_spec: str | None  # e.g. "UNSPECIFIED" / "TIER_2_CERT_GASOLINE" / "E100"; None when unresolved
    lhv_mj_per_l: float | None
    lhv_basis: LhvBasis
    confidence: FuelConfidence
    basis_label: str  # short, human-readable provenance string
    available: bool  # True iff lhv_mj_per_l is usable for a calculation


# Canonical family -> the LHV_MJ_PER_L key that actually backs it. Every
# value below is looked up from LHV_MJ_PER_L at call time -- never
# hardcoded here a second time.
_FAMILY_TO_LHV_KEY = {
    "GASOLINE": "Gasoline",
    "DIESEL": "Diesel",
    "ETHANOL": "E100",
}

_FAMILY_ASSUMPTION_LABEL = {
    "GASOLINE": "Assumed North America gasoline LHV (canonical fuel_energy.py value)",
    "DIESEL": "Assumed canonical diesel LHV",
    "ETHANOL": "Assumed canonical E100 ethanol LHV",
}

# Normalized (uppercase, trimmed) raw label -> (canonical_fuel_family, fuel_spec, basis).
# Exact matches to this repo's own native fuel_type vocabulary (both the
# title-case FUEL_TYPE_OPTIONS used by the fuel-energy calculator and the
# UPPERCASE _CHOICE_OPTIONS used by the VDE-request metadata workflow --
# these two existing, live vocabularies differ only in case, and a
# case-sensitive lookup previously made the second one invisible to PSE/
# equi-PSE) resolve as SPEC_REFERENCE. Certification-style labels resolve as
# CANONICAL_ASSUMPTION -- deterministic and traceable, never a silent guess.
# Flex/Electric/CNG/LPG/Hydrogen/blends are deliberately absent: they must
# never inherit a gasoline (or any other) LHV.
_FUEL_LABEL_REGISTRY: dict[str, tuple[str, str, LhvBasis]] = {
    "GASOLINE": ("GASOLINE", "UNSPECIFIED", LhvBasis.SPEC_REFERENCE),
    "DIESEL": ("DIESEL", "UNSPECIFIED", LhvBasis.SPEC_REFERENCE),
    "ETHANOL": ("ETHANOL", "E100", LhvBasis.SPEC_REFERENCE),
    "E100": ("ETHANOL", "E100", LhvBasis.SPEC_REFERENCE),
    "TIER 2 CERT GASOLINE": ("GASOLINE", "TIER_2_CERT_GASOLINE", LhvBasis.CANONICAL_ASSUMPTION),
    "TIER 2 CERTIFICATION GASOLINE": ("GASOLINE", "TIER_2_CERT_GASOLINE", LhvBasis.CANONICAL_ASSUMPTION),
    "TIER 3 CERT GASOLINE": ("GASOLINE", "TIER_3_CERT_GASOLINE", LhvBasis.CANONICAL_ASSUMPTION),
    "TIER 3 CERTIFICATION GASOLINE": ("GASOLINE", "TIER_3_CERT_GASOLINE", LhvBasis.CANONICAL_ASSUMPTION),
}


def resolve_fuel_energy_basis(
    raw_fuel_label: str | None, *, explicit_lhv_mj_per_l: float | None = None
) -> FuelEnergyBasis:
    """The single controlled resolver for "what LHV should this raw fuel
    label use, and why". Never returns a fabricated value: an unrecognized
    label (Flex, an unmapped blend, CNG/LPG/Hydrogen, empty/None) comes back
    with available=False -- callers must treat that exactly like today's
    "not LHV-mappable" case, not as an error.
    """
    if explicit_lhv_mj_per_l is not None:
        return FuelEnergyBasis(
            raw_fuel_label=raw_fuel_label,
            canonical_fuel_family=None,
            fuel_spec=None,
            lhv_mj_per_l=float(explicit_lhv_mj_per_l),
            lhv_basis=LhvBasis.EXPLICIT,
            confidence=FuelConfidence.EXPLICIT,
            basis_label="Explicit scenario-provided LHV",
            available=True,
        )

    normalized = str(raw_fuel_label or "").strip().upper()
    entry = _FUEL_LABEL_REGISTRY.get(normalized)
    if entry is None:
        return FuelEnergyBasis(
            raw_fuel_label=raw_fuel_label,
            canonical_fuel_family=None,
            fuel_spec=None,
            lhv_mj_per_l=None,
            lhv_basis=LhvBasis.UNKNOWN,
            confidence=FuelConfidence.UNKNOWN,
            basis_label="Unknown fuel label -- no LHV assumption available",
            available=False,
        )

    family, spec, basis = entry
    lhv_key = _FAMILY_TO_LHV_KEY.get(family)
    lhv = LHV_MJ_PER_L.get(lhv_key) if lhv_key else None
    if lhv is None:
        return FuelEnergyBasis(
            raw_fuel_label=raw_fuel_label,
            canonical_fuel_family=family,
            fuel_spec=spec,
            lhv_mj_per_l=None,
            lhv_basis=LhvBasis.UNKNOWN,
            confidence=FuelConfidence.UNKNOWN,
            basis_label="Recognized fuel family has no canonical LHV entry",
            available=False,
        )

    if basis is LhvBasis.SPEC_REFERENCE:
        confidence = FuelConfidence.HIGH
        basis_label = f"Exact match to canonical {family.title()} LHV"
    else:
        confidence = FuelConfidence.ASSUMED
        basis_label = _FAMILY_ASSUMPTION_LABEL.get(family, f"Assumed canonical {family.title()} LHV")

    return FuelEnergyBasis(
        raw_fuel_label=raw_fuel_label,
        canonical_fuel_family=family,
        fuel_spec=spec,
        lhv_mj_per_l=lhv,
        lhv_basis=basis,
        confidence=confidence,
        basis_label=basis_label,
        available=True,
    )


def _get_vde_row(vde_id: int) -> dict | None:
    from src.vde_core.db import fetchone as _fetchone

    return _fetchone("SELECT * FROM vde_db WHERE id=?;", (vde_id,))


def compute_ice_fuel_from_vde(
    vde_id: int,
    fuel_type: str,
    eta_pt: float,
    lhv_mj_per_l: float | None = None,
    electrification: str = "ICE",
    uf_phev: float | None = None,
    driveline_eff: float | None = None,
    grid_gco2_per_kwh: float | None = None,
) -> dict:
    row = _get_vde_row(vde_id)
    assert row and ("vde_net_mj_per_km" in row), "VDE row sem vde_net_mj_per_km"
    vde_mj_per_km = float(row["vde_net_mj_per_km"])

    lhv = float(lhv_mj_per_l) if lhv_mj_per_l else float(LHV_MJ_PER_L.get(fuel_type, 32.0))
    gco2_per_l = float(GCO2_PER_L.get(fuel_type, 2310.0))

    mj_pk_ice = vde_mj_per_km / max(eta_pt, 1e-6)
    L_per_km_ice = mj_pk_ice / max(lhv, 1e-6)
    L_per_100km_ice = 100.0 * L_per_km_ice
    km_per_L_ice = 100.0 / max(L_per_100km_ice, 1e-9)
    gco2_per_km_ice = L_per_km_ice * gco2_per_l

    L_per_100 = L_per_100km_ice
    km_per_L = km_per_L_ice
    gco2_km = gco2_per_km_ice
    Wh_per_km = None

    if str(electrification).upper() == "PHEV" and uf_phev is not None:
        uf = max(0.0, min(1.0, float(uf_phev)))
        if driveline_eff and grid_gco2_per_kwh is not None:
            energy_Wh_per_km_elec = (vde_mj_per_km / max(driveline_eff, 1e-6)) * MJ_TO_Wh
            gco2_km_elec = (energy_Wh_per_km_elec / 1000.0) * float(grid_gco2_per_kwh)
        else:
            energy_Wh_per_km_elec = 0.0
            gco2_km_elec = 0.0

        L_per_km_blend = (1.0 - uf) * L_per_km_ice
        L_per_100 = 100.0 * L_per_km_blend
        km_per_L = 100.0 / max(L_per_100, 1e-9) if L_per_100 > 0 else None
        gco2_km = (1.0 - uf) * gco2_per_km_ice + uf * gco2_km_elec
        Wh_per_km = uf * energy_Wh_per_km_elec

    assumptions = {
        "fuel_type": fuel_type,
        "eta_pt": eta_pt,
        "lhv_mj_per_l": lhv,
        "gco2_per_l": gco2_per_l,
        "electrification": electrification,
        "uf_phev": uf_phev,
        "driveline_eff": driveline_eff,
        "grid_gco2_per_kwh": grid_gco2_per_kwh,
        "vde_net_mj_per_km": vde_mj_per_km,
    }

    return {
        "cycle": row.get("legislation", "auto"),
        "fuel_l_per_100km": L_per_100,
        "fuel_km_per_l": km_per_L,
        "energy_Wh_per_km": Wh_per_km,
        "gco2_per_km": gco2_km,
        "assumptions_json": json.dumps(assumptions),
    }


def compute_bev_from_vde(
    vde_id: int,
    driveline_eff: float,
    grid_gco2_per_kwh: float = 0.0,
) -> dict:
    row = _get_vde_row(vde_id)
    assert row and ("vde_net_mj_per_km" in row), "VDE row sem vde_net_mj_per_km"
    vde_mj_per_km = float(row["vde_net_mj_per_km"])

    Wh_per_km = (vde_mj_per_km / max(driveline_eff, 1e-6)) * MJ_TO_Wh
    gco2_km = (Wh_per_km / 1000.0) * float(grid_gco2_per_kwh)

    assumptions = {
        "driveline_eff": driveline_eff,
        "grid_gco2_per_kwh": grid_gco2_per_kwh,
        "vde_net_mj_per_km": vde_mj_per_km,
    }

    return {
        "cycle": row.get("legislation", "auto"),
        "energy_Wh_per_km": Wh_per_km,
        "gco2_per_km": gco2_km,
        "assumptions_json": json.dumps(assumptions),
    }


__all__ = [
    "GCO2_PER_L",
    "LHV_MJ_PER_L",
    "LhvBasis",
    "FuelConfidence",
    "FuelEnergyBasis",
    "resolve_fuel_energy_basis",
    "compute_bev_from_vde",
    "compute_ice_fuel_from_vde",
]
