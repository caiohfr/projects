"""
Pure helper/service functions for VDE setup page orchestration.

No Streamlit dependency here.
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from src.vde_core.roadload import cdA_to_C, resolve_equiv_from_ctx
from src.vde_core.cycles import default_cycle_for_legislation
from src.vde_core.phase_aggregation import epa_city_hwy_from_phase, wltp_phases_from_phase
from src.vde_core.test_mass import (
    compute_wltp_test_masses,
    inertia_class_from_mass,
    resolve_test_mass_kg as resolve_test_mass_from_basis,
)
from src.vde_core.repositories import (
    count_linked_fuelcons_rows,
    delete_vde_by_id,
    fetch_vde_all_rows,
    fetch_vde_edit_rows as repo_fetch_vde_edit_rows,
    fetch_vde_make_rows,
    insert_vde_row,
    update_vde_by_id,
)
from src.vde_core.vde_calc import compute_vde_net, compute_vde_net_mj_per_km
from src.vde_core.services import estimate_aux_from_coastdown


_VDE_EXTRA_CTX_FIELDS = [
    "engine_type",
    "engine_model",
    "engine_size_l",
    "engine_aspiration",
    "transmission_type",
    "transmission_model",
    "drive_type",
    "inertia_class",
    "test_mass_kg",
    "test_mass_low_kg",
    "test_mass_high_kg",
    "test_mass_basis",
    "cda_m2",
    "weight_dist_fr_pct",
    "payload_kg",
    "mro_kg",
    "options_kg",
    "wltp_category",
    "tire_size",
    "tire_rr_note",
    "smerf",
    "front_tire_id",
    "rear_tire_id",
    "front_pressure_psi",
    "rear_pressure_psi",
    "rrc_N_per_kN",
    "crr1_frac_at_120kph",
    "rr_load_kpa",
    "tire_improvement_pct",
    "tire_load_mass_basis",
    "tire_load_mass_used_kg",
    "tire_A_final",
    "tire_B_final",
    "tire_C_final",
    "tire_calc_source",
    "tire_calc_notes",
    "trans_A_coef_N",
    "trans_B_coef_Npkph",
    "trans_C_coef_Npkph2",
    "brake_A_coef_N",
    "brake_B_coef_Npkph",
    "brake_C_coef_Npkph2",
    "parasitic_A_coef_N",
    "parasitic_B_coef_Npkph",
    "parasitic_C_coef_Npkph2",
    "aero_C_coef_Npkph2",
    "rr_alpha_N",
    "rr_beta_Npkph",
    "rr_a_Npkph2",
    "rr_b_N",
    "rr_c_Npkph",
]

EPA_TEST_MASS_DEFAULT_DELTA_KG = 136.0


def to_float(value, default=None):
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        out = float(value)
        if pd.isna(out):
            return default
        return out
    except Exception:
        return default


def resolve_test_mass_state(ctx: dict) -> dict:
    data = dict(ctx or {})
    base_mass = to_float(data.get("mass_kg"))
    payload_kg = to_float(data.get("payload_kg"))
    options_kg = to_float(data.get("options_kg"), 0.0)
    inertia_class = to_float(data.get("inertia_class"))
    legislation = str(data.get("legislation") or "").strip().upper()
    existing_basis = str(data.get("test_mass_basis") or "").strip().upper() or None
    existing_test_mass = to_float(data.get("test_mass_kg"))

    wltp_result = compute_wltp_test_masses(
        mass_kg=base_mass,
        payload_kg=payload_kg,
        options_kg=options_kg,
        wltp_category=data.get("wltp_category"),
    )

    basis = existing_basis
    manual_test_mass = existing_test_mass if basis in {"CUSTOM", "PHYSICAL_TEST_MASS"} else None

    if basis is None:
        if existing_test_mass is not None:
            basis = "CUSTOM"
            manual_test_mass = existing_test_mass
        elif legislation == "WLTP" and wltp_result.test_mass_high_kg is not None:
            basis = "WLTP_TMH"
        elif legislation == "EPA" and resolve_tire_load_mass_basis(data) == "TWC" and inertia_class is not None:
            basis = "EPA_INERTIA_CLASS"
        elif legislation == "EPA" and base_mass is not None and base_mass > 0:
            basis = "PHYSICAL_TEST_MASS"
            manual_test_mass = base_mass + EPA_TEST_MASS_DEFAULT_DELTA_KG
        elif base_mass is not None and base_mass > 0:
            basis = "CURB_FALLBACK"

    resolved_mass, resolved_basis, warnings = resolve_test_mass_from_basis(
        basis=basis,
        mass_kg=base_mass,
        options_kg=options_kg,
        test_mass_low_kg=wltp_result.test_mass_low_kg,
        test_mass_high_kg=wltp_result.test_mass_high_kg,
        inertia_class=inertia_class,
        manual_test_mass_kg=manual_test_mass,
    )

    if resolved_mass is not None and base_mass is not None and base_mass > 0 and resolved_mass < base_mass:
        raise ValueError("Test mass cannot be lower than curb weight.")

    return {
        "test_mass_kg": resolved_mass,
        "test_mass_basis": resolved_basis,
        "test_mass_low_kg": wltp_result.test_mass_low_kg,
        "test_mass_high_kg": wltp_result.test_mass_high_kg,
        "laden_mass_kg": wltp_result.laden_mass_kg,
        "available_load_low_kg": wltp_result.available_load_low_kg,
        "available_load_high_kg": wltp_result.available_load_high_kg,
        "reference_mass_kg": wltp_result.reference_mass_kg,
        "light_duty_scope_warning": wltp_result.light_duty_scope_warning,
        "wltp_category": wltp_result.wltp_category,
        "warnings": list(wltp_result.warnings) + list(warnings),
    }


def resolve_test_mass_kg(ctx: dict) -> float | None:
    return resolve_test_mass_state(ctx).get("test_mass_kg")


def is_test_mass_defaulted(ctx: dict) -> bool:
    data = dict(ctx or {})
    if to_float(data.get("test_mass_kg")) is not None:
        return False
    return resolve_test_mass_kg(data) is not None


def build_test_mass_hint(ctx: dict) -> str:
    legislation = str((ctx or {}).get("legislation") or "").strip().upper()
    if legislation == "EPA":
        return "default Curb +300 pounds / 136 kg"
    if legislation == "WLTP":
        return "WLTP-like test mass uses base mass, payload, optional equipment, and WLTP category when available"
    return ""


def resolve_tire_load_mass_basis(ctx: dict) -> str:
    legislation = str((ctx or {}).get("legislation") or "").strip().upper()
    basis = str((ctx or {}).get("tire_load_mass_basis") or "").strip().upper()
    if basis in {"TEST_MASS", "TWC"}:
        return basis
    if legislation == "EPA":
        return "TWC"
    return "TEST_MASS"


def resolve_tire_calculation_mass(ctx: dict) -> dict:
    data = dict(ctx or {})
    legislation = str(data.get("legislation") or "").strip().upper()
    basis = resolve_tire_load_mass_basis(data)
    test_mass_state = resolve_test_mass_state(data)

    if basis == "TWC":
        base_mass = to_float(data.get("mass_kg"))
        if legislation == "EPA" and base_mass is not None and base_mass > 0:
            twc_kg = inertia_class_from_mass(base_mass)
            source = "inertia_class_from_mass"
        else:
            twc_kg = to_float(data.get("twc_kg"))
            if twc_kg is None:
                twc_kg = to_float(data.get("etw_kg"))
            if twc_kg is None:
                twc_kg = to_float(data.get("inertia_class"))
            source = "twc_kg" if to_float(data.get("twc_kg")) is not None else (
                "etw_kg" if to_float(data.get("etw_kg")) is not None else "inertia_class"
            )
        return {
            "basis": basis,
            "mass_kg": twc_kg,
            "source": source,
        }

    return {
        "basis": basis,
        "mass_kg": test_mass_state.get("test_mass_kg"),
        "source": test_mass_state.get("test_mass_basis") or "test_mass_kg",
    }


def _with_effective_test_mass(ctx: dict) -> dict:
    out = dict(ctx or {})
    test_mass_state = resolve_test_mass_state(out)
    effective_test_mass = test_mass_state.get("test_mass_kg")
    if test_mass_state.get("test_mass_low_kg") is not None:
        out["test_mass_low_kg"] = test_mass_state.get("test_mass_low_kg")
    if test_mass_state.get("test_mass_high_kg") is not None:
        out["test_mass_high_kg"] = test_mass_state.get("test_mass_high_kg")
    if test_mass_state.get("test_mass_basis") is not None:
        out["test_mass_basis"] = test_mass_state.get("test_mass_basis")
    if effective_test_mass is not None:
        out["test_mass_kg"] = effective_test_mass
        out["mass_kg_effective_for_calc"] = effective_test_mass
        # Roadload engine still expects mass_kg as the active calculation mass.
        out["mass_kg"] = effective_test_mass
    return out


def validate_core(A, B, C, mass_kg):
    errs, warns = [], []
    if A is None or C is None or mass_kg is None:
        errs.append("Fill A, C and curb weight with numeric values.")
        return errs, warns
    if A < 0:
        errs.append("A cannot be negative.")
    # B may be negative (ok)
    if C < 0:
        errs.append("C cannot be negative.")
    if mass_kg <= 0:
        errs.append("Curb weight must be > 0.")
    return errs, warns


def db_list_makes(legislation: str, category: str) -> list[str]:
    rows = fetch_vde_make_rows(legislation, category)
    return [r["make"] for r in rows]


def fetch_vde_rows_full() -> list[dict]:
    return fetch_vde_all_rows()


def fetch_vde_edit_rows(limit: int = 100) -> list[dict]:
    return repo_fetch_vde_edit_rows(limit)


def fetch_linked_fuelcons_count(vde_id: int) -> int:
    return count_linked_fuelcons_rows(vde_id)


def update_vde_snapshot(vde_id: int, payload: dict) -> None:
    update_vde_by_id(vde_id, payload)


def insert_vde_snapshot(payload: dict) -> int:
    return insert_vde_row(payload)


def delete_vde_snapshot(vde_id: int) -> int:
    return delete_vde_by_id(vde_id)


def load_baselines_df():
    rows = fetch_vde_all_rows()
    data = []
    for r in rows:
        data.append(
            {
                "id": r.get("id"),
                "legislation": r.get("legislation", ""),
                "category": r.get("category", ""),
                "make": r.get("make", ""),
                "model": r.get("model", r.get("desc", "")),
                "year": r.get("year", ""),
                "A": to_float(r.get("coast_A_N"), 0.0),
                "B": to_float(r.get("coast_B_N_per_kph"), 0.0),
                "C": to_float(r.get("coast_C_N_per_kph2"), 0.0),
                "mass_kg": to_float(r.get("mass_kg"), 1500.0),
                "test_mass_kg": to_float(r.get("test_mass_kg"), None),
                "inertia_class": to_float(r.get("inertia_class"), None),
                "cd": to_float(r.get("cd"), None),
                "frontal_area_m2": to_float(r.get("frontal_area_m2"), None),
                "crr": to_float(r.get("crr"), None),
                "driveline_eff": to_float(r.get("driveline_eff"), None),
                "notes": r.get("notes", ""),
            }
        )
    return pd.DataFrame(data) if data else pd.DataFrame(
        columns=[
            "id",
            "legislation",
            "category",
            "make",
            "model",
            "year",
            "A",
            "B",
            "C",
            "mass_kg",
            "test_mass_kg",
            "inertia_class",
            "cd",
            "frontal_area_m2",
            "crr",
            "driveline_eff",
            "notes",
        ]
    )


def ensure_baseline_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "A" not in out.columns and "coast_A_N" in out.columns:
        out["A"] = out["coast_A_N"]
    if "B" not in out.columns and "coast_B_N_per_kph" in out.columns:
        out["B"] = out["coast_B_N_per_kph"]
    if "C" not in out.columns and "coast_C_N_per_kph2" in out.columns:
        out["C"] = out["coast_C_N_per_kph2"]
    return out


def baseline_filter_options(df: pd.DataFrame) -> dict:
    leg_values = sorted(df.get("legislation", pd.Series(dtype=str)).dropna().unique().tolist())
    make_values = sorted(df.get("make", pd.Series(dtype=str)).dropna().unique().tolist())
    return {
        "legislation": ["(all)"] + leg_values,
        "make": ["(all)"] + make_values,
    }


def apply_baseline_filters(df: pd.DataFrame, *, legislation: str, make: str, category_contains: str, year_eq: str) -> pd.DataFrame:
    out = df.copy()
    if legislation != "(all)" and "legislation" in out.columns:
        out = out[out["legislation"] == legislation]
    if make != "(all)" and "make" in out.columns:
        out = out[out["make"] == make]
    if str(category_contains).strip() and "category" in out.columns:
        out = out[out["category"].astype(str).str.contains(str(category_contains).strip(), case=False, na=False)]
    if str(year_eq).strip().isdigit() and "year" in out.columns:
        out = out[out["year"] == int(str(year_eq).strip())]
    return out


def build_baseline_state_payload(base: dict, selected_id: int) -> dict:
    return {
        "vde_id_parent": int(selected_id),
        "baseline_dict": {
            "A": base.get("A", base.get("coast_A_N")),
            "B": base.get("B", base.get("coast_B_N_per_kph")),
            "C": base.get("C", base.get("coast_C_N_per_kph2")),
            "mass_kg": base.get("mass_kg"),
            "test_mass_kg": base.get("test_mass_kg"),
            "legislation": base.get("legislation"),
            "category": base.get("category"),
            "tire_size": base.get("tire_size"),
            "rrc_N_per_kN": base.get("rrc_N_per_kN"),
            "crr1_frac_at_120kph": base.get("crr1_frac_at_120kph"),
            "front_pressure_psi": base.get("front_pressure_psi"),
            "rear_pressure_psi": base.get("rear_pressure_psi"),
            "rr_load_kpa": base.get("rr_load_kpa"),
            "smerf": base.get("smerf"),
            "parasitic_A_coef_N": base.get("parasitic_A_coef_N"),
            "parasitic_B_Npkph": base.get("parasitic_B_Npkph"),
            "parasitic_C_coef_Npkph2": base.get("parasitic_C_coef_Npkph2"),
            "brake_A_coef_N": base.get("brake_A_coef_N"),
            "brake_B_Npkph": base.get("brake_B_Npkph"),
            "brake_C_coef_Npkph2": base.get("brake_C_coef_Npkph2"),
            "trans_A_coef_N": base.get("trans_A_coef_N"),
            "trans_B_coef_Npkph": base.get("trans_B_coef_Npkph", base.get("trans_B_Npkph")),
            "trans_C_coef_Npkph2": base.get("trans_C_coef_Npkph2"),
            "electrification": base.get("electrification"),
            "transmission_type": base.get("transmission_type"),
            "cda_m2": base.get("cda_m2"),
        },
    }


def build_delta_mode_ctx_updates(base: dict) -> dict:
    updates = {
        "A": float(base.get("A", base.get("coast_A_N", 0.0)) or 0.0),
        "B": float(base.get("B", base.get("coast_B_N_per_kph", 0.0)) or 0.0),
        "C": float(base.get("C", base.get("coast_C_N_per_kph2", 0.0)) or 0.0),
        "mass_kg": float(base.get("mass_kg", 0.0) or 0.0),
        "test_mass_kg": to_float(base.get("test_mass_kg")),
        "test_mass_low_kg": to_float(base.get("test_mass_low_kg")),
        "test_mass_high_kg": to_float(base.get("test_mass_high_kg")),
        "test_mass_basis": str(base.get("test_mass_basis") or "").strip().upper() or None,
        "payload_kg": to_float(base.get("payload_kg")),
        "options_kg": to_float(base.get("options_kg"), 0.0),
        "wltp_category": base.get("wltp_category"),
    }
    if base.get("crr1_frac_at_120kph") is not None:
        updates["crr1_frac_at_120kph"] = to_float(base.get("crr1_frac_at_120kph"))
    if base.get("rrc_N_per_kN") is not None:
        updates["rrc_N_per_kN"] = to_float(base.get("rrc_N_per_kN"))
    if base.get("tire_size"):
        updates["tire_size"] = str(base.get("tire_size"))
    return updates


def build_live_vde_preview(ctx: dict) -> dict:
    """
    Compute live VDE preview payload from page context.

    Returns a dict with keys:
      - ok: bool
      - error: Optional[str]
      - total_mj_km: Optional[float]
      - phases: dict
      - equiv: Optional[EquivalentABC]
    """
    calc_ctx = _with_effective_test_mass(ctx)
    df_cycle = calc_ctx.get("cycle_df")

    try:
        leg = str(calc_ctx.get("legislation", "")).upper()
        equiv = resolve_equiv_from_ctx(calc_ctx)
        A1, B1, C1, mass_kg1 = equiv.A, equiv.B, equiv.C, equiv.mass_kg
    except Exception as e:
        return {"ok": False, "error": f"Preview not available (inputs): {e}", "total_mj_km": None, "phases": {}, "equiv": None}

    total_mj_km = None
    phases = {}
    try:
        if isinstance(df_cycle, pd.DataFrame) and "phase" in df_cycle.columns:
            if leg == "EPA":
                res = epa_city_hwy_from_phase(df_cycle, A1, B1, C1, mass_kg1) or {}
                city = res.get("urb_MJ_km")
                hwy = res.get("hwy_MJ_km") or res.get("hw_MJ_km") or res.get("hwy_MJ_per_km")
                if city is not None:
                    phases["city"] = float(city)
                if hwy is not None:
                    phases["hwy"] = float(hwy)
                if res.get("net_comb_MJ_km") is not None:
                    total_mj_km = float(res["net_comb_MJ_km"])
                elif ("city" in phases) and ("hwy" in phases):
                    total_mj_km = 0.55 * phases["city"] + 0.45 * phases["hwy"]
            else:
                res = wltp_phases_from_phase(df_cycle, A1, B1, C1, mass_kg1) or {}
                for ki, ko in [
                    ("vde_low_mj_per_km", "low"),
                    ("vde_mid_mj_per_km", "mid"),
                    ("vde_high_mj_per_km", "high"),
                    ("vde_extra_high_mj_per_km", "xhigh"),
                ]:
                    if res.get(ki) is not None:
                        phases[ko] = float(res[ki])
                if res.get("vde_net_mj_per_km") is not None:
                    total_mj_km = float(res["vde_net_mj_per_km"])

        if total_mj_km is None:
            if not isinstance(df_cycle, pd.DataFrame):
                raise ValueError("No cycle loaded.")
            g = df_cycle.copy()
            if "v_mps" not in g.columns:
                if "v" in g.columns:
                    g["v_mps"] = pd.to_numeric(g["v"], errors="coerce")
                else:
                    raise ValueError("Cycle has no 'v' (m/s) or 'v_mps' column.")

            tcol = "t" if "t" in g.columns else ("time_s" if "time_s" in g.columns else None)
            if tcol is None:
                raise ValueError("Cycle has no 't' or 'time_s' column.")

            g[tcol] = pd.to_numeric(g[tcol], errors="coerce")
            g = g.dropna(subset=[tcol, "v_mps"]).sort_values(tcol).reset_index(drop=True)
            g["dt"] = g[tcol].diff().fillna(0.0).clip(lower=0.0)

            r = compute_vde_net(g, A1, B1, C1, mass_kg1)
            total_mj_km = float(r["MJ_km"]) if isinstance(r, dict) else float(r)

        return {"ok": True, "error": None, "total_mj_km": total_mj_km, "phases": phases, "equiv": equiv}
    except Exception as e:
        return {"ok": False, "error": f"Preview not available: {e}", "total_mj_km": None, "phases": {}, "equiv": None}


def build_compute_vde_from_ctx(ctx: dict) -> dict:
    """
    Compute payload used by VDE compute/save flow.

    Returns:
      - ok: bool
      - error: Optional[str]
      - equiv: EquivalentABC | None
      - total_mj_km: float | None
      - by_phase: dict
      - deltas: dict with rr/brake/parasitics/aero_c/mass
    """
    preview = build_live_vde_preview(ctx)
    if not preview.get("ok"):
        return {
            "ok": False,
            "error": preview.get("error", "Compute not available."),
            "equiv": None,
            "total_mj_km": None,
            "by_phase": {},
            "deltas": {},
        }

    d_rr = to_float(ctx.get("delta_rr_N"), 0.0)
    d_br = to_float(ctx.get("delta_brake_N"), 0.0)
    d_par = to_float(ctx.get("delta_parasitics_N"), 0.0)
    d_cda = cdA_to_C(to_float(ctx.get("delta_aero_cdA"), 0.0))
    d_mass = to_float(ctx.get("delta_mass_kg"), 0.0)

    return {
        "ok": True,
        "error": None,
        "equiv": preview.get("equiv"),
        "total_mj_km": float(preview.get("total_mj_km")),
        "by_phase": dict(preview.get("phases", {})),
        "deltas": {
            "delta_rr_N": d_rr,
            "delta_brake_N": d_br,
            "delta_parasitics_N": d_par,
            "delta_aero_Npkph2": d_cda,
            "delta_mass_kg": d_mass,
        },
    }


def build_vde_insert_row(
    ctx: dict,
    *,
    leg: str,
    cat: str,
    make: str,
    model: str,
    year: Optional[int],
    notes: str,
    cycle_name: str,
    cycle_source: str,
    equiv,
    total_mj_km: float,
    by_phase: dict,
    deltas: dict,
    decomp: Optional[dict] = None,
) -> dict:
    test_mass_state = resolve_test_mass_state({**dict(ctx or {}), "legislation": leg})
    row = {
        "legislation": leg,
        "category": cat,
        "make": make,
        "model": model,
        "year": year,
        "notes": notes,
        "mass_kg": to_float(ctx.get("mass_kg"), equiv.mass_kg),
        "test_mass_kg": test_mass_state.get("test_mass_kg"),
        "test_mass_low_kg": test_mass_state.get("test_mass_low_kg"),
        "test_mass_high_kg": test_mass_state.get("test_mass_high_kg"),
        "test_mass_basis": test_mass_state.get("test_mass_basis"),
        "coast_A_N": equiv.A,
        "coast_B_N_per_kph": equiv.B,
        "coast_C_N_per_kph2": equiv.C,
        "cycle_name": cycle_name,
        "cycle_source": cycle_source,
        "vde_net_mj_per_km": total_mj_km,
        "delta_rr_N": deltas.get("delta_rr_N", 0.0),
        "delta_brake_N": deltas.get("delta_brake_N", 0.0),
        "delta_mass_kg": deltas.get("delta_mass_kg", 0.0),
        "delta_parasitics_N": deltas.get("delta_parasitics_N", 0.0),
        "delta_aero_Npkph2": deltas.get("delta_aero_Npkph2", 0.0),
    }

    if "city" in by_phase:
        row["vde_urb_mj_per_km"] = float(by_phase["city"])
    if "hwy" in by_phase:
        row["vde_hw_mj_per_km"] = float(by_phase["hwy"])
    if "low" in by_phase:
        row["vde_low_mj_per_km"] = float(by_phase["low"])
    if "mid" in by_phase:
        row["vde_mid_mj_per_km"] = float(by_phase["mid"])
    if "high" in by_phase:
        row["vde_high_mj_per_km"] = float(by_phase["high"])
    if "xhigh" in by_phase:
        row["vde_extra_high_mj_per_km"] = float(by_phase["xhigh"])

    for key in _VDE_EXTRA_CTX_FIELDS:
        value = ctx.get(key, None)
        if value not in (None, ""):
            row[key] = value

    base = ctx.get("baseline_dict")
    if ctx.get("vde_id_parent") and isinstance(base, dict):
        row.update(
            {
                "vde_id_parent": ctx["vde_id_parent"],
                "baseline_A_N": base.get("A"),
                "baseline_B_N_per_kph": base.get("B"),
                "baseline_C_N_per_kph2": base.get("C"),
                "baseline_mass_kg": base.get("mass_kg"),
            }
        )

    if decomp:
        row.update(
            {
                k: float(v)
                for k, v in {
                    "rr_alpha_N": decomp.get("rr_alpha_N"),
                    "rr_beta_Npkph": decomp.get("rr_beta_Npkph"),
                    "aero_C_coef_Npkph2": decomp.get("aero_C_coef_Npkph2"),
                    "parasitic_A_coef_N": decomp.get("parasitic_A_coef_N"),
                    "parasitic_B_Npkph": decomp.get("parasitic_B_Npkph"),
                    "parasitic_C_coef_Npkph2": decomp.get("parasitic_C_coef_Npkph2"),
                }.items()
                if v is not None
            }
        )

    return {
        k: v
        for k, v in row.items()
        if v is not None
        and (not isinstance(v, str) or v.strip() != "")
        and (not isinstance(v, (int, float)) or math.isfinite(float(v)))
    }


def save_vde_from_ctx(ctx: dict, *, defaults_df=None) -> dict:
    """
    Persist a new VDE snapshot from the current setup context.

    Returns:
      - vde_id: int
      - row: dict
      - calc: dict
      - equiv: EquivalentABC
      - total_mj_km: float
      - by_phase: dict
      - phase_updates: dict
      - decomp: dict | None
    """
    calc = build_compute_vde_from_ctx(ctx)
    if not calc.get("ok"):
        raise ValueError(calc.get("error", "Compute not available."))

    leg = str(ctx.get("legislation", ""))
    cat = ctx.get("category")
    make = ctx.get("make")
    model = ctx.get("model")
    year_raw = ctx.get("year")
    year = int(year_raw) if str(year_raw).isdigit() else None
    notes = ctx.get("notes", "")
    cycle_name = default_cycle_for_legislation(leg)
    cycle_source = ctx.get("cycle_source", f"standard:{leg}")

    equiv = calc["equiv"]
    total_mj_km = float(calc["total_mj_km"])
    by_phase = dict(calc.get("by_phase", {}))
    deltas = dict(calc.get("deltas", {}))

    decomp = None
    if defaults_df is not None:
        try:
            decomp = estimate_aux_from_coastdown(
                A_N=equiv.A,
                B_N_per_kph=equiv.B,
                C_N_per_kph2=equiv.C,
                mass_kg=equiv.mass_kg,
                category=cat,
                electrification=ctx.get("electrification", "ICE"),
                transmission_type=ctx.get("transmission_type", "AT"),
                cdA_override_m2=ctx.get("cda_m2"),
                defaults_df=defaults_df,
            )
        except Exception:
            decomp = None

    row = build_vde_insert_row(
        ctx,
        leg=leg,
        cat=cat,
        make=make,
        model=model,
        year=year,
        notes=notes,
        cycle_name=cycle_name,
        cycle_source=cycle_source,
        equiv=equiv,
        total_mj_km=total_mj_km,
        by_phase=by_phase,
        deltas=deltas,
        decomp=decomp,
    )

    vde_id = insert_vde_snapshot(row)

    df_cycle = ctx.get("cycle_df")
    phase_updates = build_vde_phase_update(
        df_cycle,
        leg,
        A=equiv.A,
        B=equiv.B,
        C=equiv.C,
        mass_kg=equiv.mass_kg,
    )
    if phase_updates:
        update_vde_snapshot(vde_id, phase_updates)

    return {
        "vde_id": int(vde_id),
        "row": row,
        "calc": calc,
        "equiv": equiv,
        "total_mj_km": total_mj_km,
        "by_phase": by_phase,
        "phase_updates": phase_updates,
        "decomp": decomp,
    }


def build_vde_phase_update(df_cycle, leg: str, *, A: float, B: float, C: float, mass_kg: float) -> dict:
    if not (isinstance(df_cycle, pd.DataFrame) and "phase" in df_cycle.columns):
        return {}

    upd = {}
    if str(leg).upper() == "EPA":
        res = epa_city_hwy_from_phase(df_cycle, A, B, C, mass_kg) or {}
        if res.get("urb_MJ") is not None:
            upd["vde_urb_mj"] = float(res["urb_MJ"])
        if res.get("hw_MJ") is not None:
            upd["vde_hw_mj"] = float(res["hw_MJ"])
        if res.get("net_comb_MJ_km") is not None:
            upd["vde_net_mj_per_km"] = float(res["net_comb_MJ_km"])
    else:
        res = wltp_phases_from_phase(df_cycle, A, B, C, mass_kg) or {}
        for key in ("vde_low_mj_per_km", "vde_mid_mj_per_km", "vde_high_mj_per_km", "vde_extra_high_mj_per_km"):
            if res.get(key) is not None:
                upd[key] = float(res[key])
        if res.get("vde_net_mj_per_km") is not None:
            upd["vde_net_mj_per_km"] = float(res["vde_net_mj_per_km"])
    return upd


def build_edit_core_update(
    *,
    A: float,
    B: float,
    C: float,
    mass_kg: float,
    test_mass_kg: float | None,
    make: str,
    model: str,
    year: int,
    notes: str,
) -> dict:
    payload = {
        "coast_A_N": A,
        "coast_B_N_per_kph": B,
        "coast_C_N_per_kph2": C,
        "mass_kg": mass_kg,
        "make": make,
        "model": model,
        "year": int(year),
        "notes": notes,
    }
    if test_mass_kg is not None and test_mass_kg > 0:
        payload["test_mass_kg"] = test_mass_kg
    return payload


def collect_ctx_updates(ctx: Optional[dict], fields: list[str], *, include_none: bool = False) -> dict:
    if not isinstance(ctx, dict):
        return {}
    out = {}
    for key in fields:
        value = ctx.get(key)
        if include_none:
            if value is not None:
                out[key] = value
        else:
            if value not in (None, ""):
                out[key] = value
    return out


def merge_update_payloads(*payloads: Optional[dict]) -> dict:
    merged = {}
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            if value is None:
                continue
            if isinstance(value, str) and value.strip() == "":
                continue
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                continue
            merged[key] = value
    return merged


def build_decomp_update_for_edit(decomp: Optional[dict]) -> dict:
    if not isinstance(decomp, dict):
        return {}
    return {
        k: float(v)
        for k, v in {
            "rr_alpha_N": decomp.get("rr_alpha_N"),
            "rr_beta_Npkph": decomp.get("rr_beta_Npkph"),
            "aero_C_coef_Npkph2": decomp.get("aero_C_coef_Npkph2"),
            "parasitic_A_coef_N": decomp.get("parasitic_A_coef_N"),
            "parasitic_B_coef_Npkph": decomp.get("parasitic_B_Npkph"),
            "parasitic_C_coef_Npkph2": decomp.get("parasitic_C_coef_Npkph2"),
        }.items()
        if v is not None
    }


def compute_vde_preview_from_inputs(df_cycle, leg: str, *, A: float, B: float, C: float, mass_kg: float) -> dict:
    if not (isinstance(df_cycle, pd.DataFrame) and not df_cycle.empty):
        return {"ok": False, "error": "Cycle not available.", "total_mj_km": None, "by_phase": {}}

    total_mj_km = None
    by_phase = {}
    if "phase" in df_cycle.columns:
        if str(leg).upper() == "EPA":
            res = epa_city_hwy_from_phase(df_cycle, A, B, C, mass_kg) or {}
            city = res.get("city_MJ_km") or res.get("urb_MJ_km") or res.get("city_MJ_per_km")
            hwy = res.get("hwy_MJ_km") or res.get("hw_MJ_km") or res.get("hwy_MJ_per_km")
            if city is not None:
                by_phase["city"] = float(city)
            if hwy is not None:
                by_phase["hwy"] = float(hwy)
            if res.get("net_comb_MJ_km") is not None:
                total_mj_km = float(res["net_comb_MJ_km"])
            elif "city" in by_phase and "hwy" in by_phase:
                total_mj_km = 0.55 * by_phase["city"] + 0.45 * by_phase["hwy"]
        else:
            res = wltp_phases_from_phase(df_cycle, A, B, C, mass_kg) or {}
            for key_in, key_out in [
                ("vde_low_mj_per_km", "low"),
                ("vde_mid_mj_per_km", "mid"),
                ("vde_high_mj_per_km", "high"),
                ("vde_extra_high_mj_per_km", "xhigh"),
            ]:
                if res.get(key_in) is not None:
                    by_phase[key_out] = float(res[key_in])
            if res.get("vde_net_mj_per_km") is not None:
                total_mj_km = float(res["vde_net_mj_per_km"])

    if total_mj_km is None:
        r_all = compute_vde_net_mj_per_km(df_cycle, A, B, C, mass_kg)
        total_mj_km = float(r_all["MJ_km"]) if isinstance(r_all, dict) else float(r_all)

    return {"ok": True, "error": None, "total_mj_km": total_mj_km, "by_phase": by_phase}
