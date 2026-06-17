from __future__ import annotations

from copy import deepcopy


DEFAULTS = {
    "roadload_params": {"f0": 100.0, "f1": 0.5, "f2": 0.04, "mass": 1500.0},
    "eta_pt": 0.24,
    "lhv": 32.0,
    "selected_techs": [],
    "cycle_df": None,
}

PWT_SIDEBAR_DEFAULTS = {
    "sb_eta_trans": 0.92,
    "sb_eta_pt": 0.35,
    "sb_fuel_type": "Gasoline",
    "sb_lhv_override": 34.2,
    "sb_uf": 0.50,
    "sb_eta_drive": 0.88,
    "sb_grid": 400.0,
    "pwt_gears": 6,
    "pwt_fdr": 3.91,
}

VDE_SETUP_CTX_DEFAULTS = {
    "legislation": "EPA",
    "category": "",
    "make": "",
    "model": "",
    "year": 2024,
    "notes": "",
    "A": 120.0,
    "B": 0.0,
    "C": 0.012,
    "mass_kg": 1550.0,
    "test_mass_kg": None,
    "test_mass_use_default": True,
    "cd": 0.30,
    "frontal_area_m2": 2.20,
    "cda": 0.66,
    "crr": 0.010,
    "crr1_frac_at_120kph": 0.010,
    "cycle_df": None,
    "cycle_source": "",
    "baseline_id": None,
    "baseline_dict": None,
    "vde_id_parent": None,
    "from_delta": "Deltas",
    "mode": "From baseline (editable)",
}

VDE_SETUP_META_KEYS = (
    "legislation",
    "category",
    "make",
    "model",
    "year",
    "notes",
    "cycle_df",
    "cycle_source",
)

VDE_SETUP_VOLATILE_KEYS = (
    "A",
    "B",
    "C",
    "mass_kg",
    "test_mass_kg",
    "test_mass_use_default",
    "delta_rr_N",
    "delta_brake_N",
    "delta_parasitics_N",
    "delta_aero_Npkph2",
    "delta_aero_cdA",
    "tire_size",
    "tire_circ_m",
    "diameter_mm",
    "rrc_N_per_kN",
    "crr1_frac_at_120kph",
    "front_pressure_psi",
    "rear_pressure_psi",
    "rr_load_kpa",
    "smerf",
    "parasitic_A_coef_N",
    "parasitic_B_Npkph",
    "parasitic_C_coef_Npkph2",
    "brake_A_coef_N",
    "brake_B_Npkph",
    "brake_C_coef_Npkph2",
    "baseline_id",
    "baseline_dict",
    "vde_id_parent",
)


def _clone_default(value):
    if isinstance(value, (dict, list, set)):
        return deepcopy(value)
    return value


def ensure_defaults(ss):
    for key, value in DEFAULTS.items():
        if key not in ss:
            ss[key] = _clone_default(value)


def ensure_pwt_sidebar_defaults(ss):
    for key, value in PWT_SIDEBAR_DEFAULTS.items():
        if key not in ss:
            ss[key] = _clone_default(value)


def ensure_vde_setup_state(ss):
    if "ctx" not in ss:
        ss["ctx"] = {}
    ctx = ss["ctx"]
    for key, value in VDE_SETUP_CTX_DEFAULTS.items():
        if key not in ctx:
            ctx[key] = _clone_default(value)
    ss.setdefault("_last_mode", ctx["mode"])


def reset_vde_setup_state(ss, preserve_meta: bool = True):
    if "ctx" not in ss:
        ss["ctx"] = {}
    ctx = ss["ctx"]
    meta = {key: ctx.get(key) for key in VDE_SETUP_META_KEYS} if preserve_meta else {}

    for key in VDE_SETUP_VOLATILE_KEYS:
        ctx.pop(key, None)

    if preserve_meta:
        for key, value in meta.items():
            ctx[key] = value

    for key in ("A", "B", "C", "mass_kg", "test_mass_kg", "test_mass_use_default", "from_delta"):
        ctx.setdefault(key, _clone_default(VDE_SETUP_CTX_DEFAULTS[key]))
