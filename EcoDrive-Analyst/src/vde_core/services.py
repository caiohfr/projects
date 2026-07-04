"""
Compatibility-oriented core facade for EcoDrive-Analyst.

This module historically concentrated most backend helpers. The implementation
is now split into smaller modules, while ``services.py`` remains available as a
stable import surface for the rest of the application.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.vde_core.conversion import mjkm_to_whkm, whkm_to_mjkm
from src.vde_core.cycles import (
    cycle_summary,
    default_cycle_for_legislation,
    load_cycle_csv,
    use_standard_cycle,
)
from src.vde_core.fuel_energy import (
    GCO2_PER_L,
    LHV_MJ_PER_L,
    compute_bev_from_vde,
    compute_ice_fuel_from_vde,
)
from src.vde_core.phase_aggregation import epa_city_hwy_from_phase, split_by_phase, wltp_phases_from_phase
from src.vde_core.test_mass import (
    DEFAULT_DRIVER_MASS_KG,
    WLTP_ADDITIONAL_MASS_KG,
    WLTP_BASE_INCREMENT_KG,
    WLTP_LOAD_FACTORS,
    WltpTestMassResult,
    autoresolve_test_mass,
    compute_mro_from_stda,
    compute_wltp_test_mass,
    compute_wltp_test_masses,
    derive_laden_mass_kg,
    get_wltp_light_duty_scope_warning,
    inertia_class_from_mass,
    normalize_wltp_category,
    resolve_test_mass_kg,
    to_float_or_none,
)
from src.vde_core.vde_calc import (
    apply_coastdown_deltas,
    compute_vde_net,
    compute_vde_net_mj_per_km,
    vde_total_simple,
)


G = 9.80665
RHO = 1.2
TOL_A = 5.0
TOL_B = 0.10
TOL_C = 1e-1

DEFAULTS_REQUIRED_COLS = [
    "category",
    "electrification",
    "transmission_type",
    "cdA_default_m2",
    "rrc_N_per_kN",
    "crr1_frac_at_120kph",
]


def load_vde_defaults(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    missing = [col for col in DEFAULTS_REQUIRED_COLS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in defaults CSV: {missing}")
    return df


def estimate_aux_from_coastdown(
    *,
    A_N: float,
    B_N_per_kph: float,
    C_N_per_kph2: float,
    mass_kg: float,
    category: str,
    electrification: str,
    transmission_type: str,
    cdA_override_m2: Optional[float] = None,
    defaults_df: Optional[pd.DataFrame] = None,
    defaults_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    if defaults_df is None:
        if not defaults_path:
            raise ValueError("Provide defaults_df or defaults_path.")
        defaults_df = load_vde_defaults(defaults_path)

    df = defaults_df
    match = df[
        (df["category"].astype(str).str.upper() == str(category).upper())
        & (df["electrification"].astype(str).str.upper() == str(electrification).upper())
        & (df["transmission_type"].astype(str).str.upper() == str(transmission_type).upper())
    ]
    if match.empty:
        match = df[df["category"].astype(str).str.upper() == str(category).upper()]
        if match.empty:
            raise ValueError("No defaults found for this (category/electrification/transmission).")

    row = match.iloc[0]
    cdA_default = float(row["cdA_default_m2"])
    rrc_N_per_kN = float(row["rrc_N_per_kN"])
    crr1_frac_120 = float(row["crr1_frac_at_120kph"])

    A = float(A_N)
    B = float(B_N_per_kph)
    C = float(C_N_per_kph2)
    if mass_kg is None or mass_kg <= 0:
        raise ValueError("mass_kg must be > 0")

    cdA = float(cdA_override_m2) if cdA_override_m2 is not None else cdA_default

    load_kN = mass_kg * G / 1000.0
    A_rr = rrc_N_per_kN * load_kN
    B_rr = A_rr * (crr1_frac_120 / 120.0)
    C_rr = 0.0

    C_aero = 0.5 * RHO * cdA * (1 / 3.6) ** 2

    A_par = max(0.0, A - A_rr)
    B_par = max(0.0, B - B_rr)
    C_par = max(0.0, C - C_aero)

    dA = (A_rr + A_par) - A
    dB = (B_rr + B_par) - B
    dC = (C_aero + C_rr + C_par) - C
    check_ok = (abs(dA) <= TOL_A) and (abs(dB) <= TOL_B) and (abs(dC) <= TOL_C)

    return {
        "rr_alpha_N": A_rr,
        "rr_beta_Npkph": B_rr,
        "aero_C_coef_Npkph2": C_aero,
        "parasitic_A_coef_N": A_par,
        "parasitic_B_coef_Npkph": B_par,
        "parasitic_C_coef_Npkph2": C_par,
        "cdA_used_m2": cdA,
        "dA": dA,
        "dB": dB,
        "dC": dC,
        "check_ok": bool(check_ok),
        "rl_source": "measured_decomposed_NET",
    }


__all__ = [
    "DEFAULTS_REQUIRED_COLS",
    "DEFAULT_DRIVER_MASS_KG",
    "G",
    "GCO2_PER_L",
    "LHV_MJ_PER_L",
    "RHO",
    "TOL_A",
    "TOL_B",
    "TOL_C",
    "WLTP_ADDITIONAL_MASS_KG",
    "WLTP_BASE_INCREMENT_KG",
    "WLTP_LOAD_FACTORS",
    "WltpTestMassResult",
    "apply_coastdown_deltas",
    "autoresolve_test_mass",
    "compute_bev_from_vde",
    "compute_ice_fuel_from_vde",
    "compute_mro_from_stda",
    "compute_vde_net",
    "compute_vde_net_mj_per_km",
    "compute_wltp_test_mass",
    "compute_wltp_test_masses",
    "cycle_summary",
    "default_cycle_for_legislation",
    "derive_laden_mass_kg",
    "get_wltp_light_duty_scope_warning",
    "epa_city_hwy_from_phase",
    "estimate_aux_from_coastdown",
    "inertia_class_from_mass",
    "load_cycle_csv",
    "load_vde_defaults",
    "mjkm_to_whkm",
    "normalize_wltp_category",
    "resolve_test_mass_kg",
    "split_by_phase",
    "to_float_or_none",
    "use_standard_cycle",
    "vde_total_simple",
    "whkm_to_mjkm",
    "wltp_phases_from_phase",
]
