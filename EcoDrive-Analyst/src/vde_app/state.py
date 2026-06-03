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
