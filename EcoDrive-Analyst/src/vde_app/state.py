DEFAULTS = {
    "roadload_params": {"f0": 100.0, "f1": 0.5, "f2": 0.04, "mass": 1500.0},
    "eta_pt": 0.24,
    "lhv": 32.0,
    "selected_techs": [],
    "cycle_df": None,
}

def ensure_defaults(ss):
    for k, v in DEFAULTS.items():
        if k not in ss:
            ss[k] = v
