# pages/1_Data_&_Setup.py
# -----------------------------------------------------------------------------
# 📥 VDE Snapshot (NET)
# Flow: 1) Context  2) Roadloads  3) Cycle  4) Compute NET  5) Save
# -----------------------------------------------------------------------------

import streamlit as st
import pandas as pd
from pathlib import Path
from src.vde_core.utils import cycle_kpis

# ---- your project dependencies (kept same names) ----------------------------
from src.vde_core.db import ensure_db, fetchall, fetchone, insert_vde, update_vde, delete_row
from src.vde_core.services import (
    default_cycle_for_legislation, load_cycle_csv,
    compute_vde_net_mj_per_km, compute_vde_net,
    epa_city_hwy_from_phase, wltp_phases_from_phase, 
)
from src.vde_app.plots import cycle_chart

# =============================================================================
# Small helpers
# =============================================================================
def _safe_get(row: dict, key: str, default=None):
    return row[key] if (row and key in row and row[key] is not None) else default

def _try_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

# ===========================
# Drive cycle (auto + upload)
# ===========================
def _use_standard_cycle(leg):
    fname = default_cycle_for_legislation(leg)
    try:
        df = load_cycle_csv(fname)
        st.session_state["cycle_df"] = df
        st.session_state["cycle_source"] = f"standard:{leg}"
        st.success(f"Using default **{leg}** cycle: `{fname}.csv`")
        k = cycle_kpis(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duration", f"{k['duration_s']:.0f} s")
        c2.metric("Distance", f"{k['distance_km']:.2f} km")
        c3.metric("Avg Speed", f"{k['v_mean_kmh']:.1f} km/h")
        c4.metric("Samples", f"{k['n_points']}")
        return df
    except Exception as e:
        st.warning(f"Default cycle for **{leg}** not found. {e}")
        st.info("Please upload a custom cycle below.")
        return None

def _load_baselines():
    """Fetch useful fields from vde_db to select a baseline (schema-resilient)."""
    ensure_db()
    rows = fetchall("SELECT * FROM vde_db ORDER BY COALESCE(updated_at,created_at) DESC")
    data = []
    for r in rows:
        data.append({
            "id": _safe_get(r, "id"),
            "legislation": _safe_get(r, "legislation", ""),
            "category": _safe_get(r, "category", ""),
            "make": _safe_get(r, "make", ""),
            "model": _safe_get(r, "model", _safe_get(r, "desc", "")),
            "year": _safe_get(r, "year", ""),
            "A": _try_float(_safe_get(r, "coast_A_N"), 0.0),
            "B": _try_float(_safe_get(r, "coast_B_N_per_kph"), 0.0),
            "C": _try_float(_safe_get(r, "coast_C_N_per_kph2"), 0.0),
            "mass_kg": _try_float(_safe_get(r, "mass_kg"), 0.0)
        })
    return pd.DataFrame(data) if data else pd.DataFrame(columns=["id","legislation","category","make","model","year","A","B","C","mass_kg"])

def _validate_abc_mass(A, B, C, M):
    """
    Simple rules (B can be negative):
      - A >= 0 (error if < 0)
      - B: negative allowed
      - C >= 0 (error if < 0)
      - M > 0 (error if <= 0); warnings for <300 or >3500 kg
    Returns (errors, warns)
    """
    errors, warns = [], []

    if A is None or C is None or M is None:
        errors.append("Fill A, C and Mass with numeric values.")
        return errors, warns

    if A < 0:
        errors.append("A cannot be negative.")
    # B may be < 0 (ok)
    if C < 0:
        errors.append("C cannot be negative.")

    if M <= 0:
        errors.append("Mass must be > 0.")
    elif M < 300:
        warns.append("Very low mass (< 300 kg): confirm unit/value.")
    elif M > 3500:
        warns.append("Very high mass (> 3500 kg): confirm unit/value.")

    # soft sanity checks
    if (A == 0) and (_try_float(B, 0.0) == 0.0) and (C == 0):
        warns.append("A, B and C all zero is unusual. Please confirm.")
    if _try_float(B, 0.0) < -0.5:
        warns.append("B is quite negative (< -0.5 N/kph). If expected, ignore; otherwise, review input.")

    return errors, warns

def _cycle_summary(df_cycle: pd.DataFrame):
    if df_cycle is None or df_cycle.empty:
        return "No cycle loaded.", ""
    # simple KPIs
    t = df_cycle.iloc[:,0].astype(float)
    v = df_cycle.iloc[:,1].astype(float)
    dur = t.iloc[-1] - t.iloc[0]
    dist = (v.sum() * (t.diff().fillna(0))).sum()  # crude integral v*dt (meters)
    dist_km = dist / 1000.0
    vavg = v.mean() * 3.6  # m/s -> km/h
    return f"Duration: {dur:.0f} s • Distance: {dist_km:.2f} km • v̄: {vavg:.1f} km/h", f"{dist_km:.3f}"

# =============================================================================
# Initial state
# =============================================================================
st.set_page_config(page_title="VDE Snapshot (NET)", layout="wide")
ensure_db()

if "snapshot_ctx" not in st.session_state:
    st.session_state["snapshot_ctx"] = {
        "legislation": "WLTP",
        "category": "",
        "make": "",
        "model": "",
        "year": "",
        "notes": "",
        "A": None, "B": None, "C": None, "mass_kg": None,
        "origin": "",      # baseline|from_test|semi_param
        "parent_id": None, # baseline id if any
        "cycle_df": None,
        "cycle_name": "",
        "results": None,   # dict with NET results per phase/cycle
    }

ctx = st.session_state["snapshot_ctx"]

# =============================================================================
# H1 + sticky summary
# =============================================================================
st.title("📥 VDE Snapshot (NET)")

summary = st.container()
with summary:
    st.markdown("**Snapshot Summary (updates as you proceed)**")
    cols = st.columns([1,1,1,1,1])
    cols[0].metric("Legislation", ctx["legislation"] or "-")
    cols[1].metric("Category", ctx["category"] or "-")
    cols[2].metric("Vehicle", f"{ctx['make']} {ctx['model']}".strip() or "-")
    cols[3].metric("Year", str(ctx["year"]) or "-")
    if ctx.get("A") is not None:
        cols = st.columns([1,1,1,1])
        cols[0].metric("A [N]", f"{ctx['A']:.3f}")
        cols[1].metric("B [N/kph]", f"{_try_float(ctx['B'],0.0):.5f}")
        cols[2].metric("C [N/kph²]", f"{ctx['C']:.6f}")
        cols[3].metric("Mass [kg]", f"{ctx['mass_kg']:.1f}")
    if ctx.get("cycle_df") is not None:
        kpi, _ = _cycle_summary(ctx["cycle_df"])
        st.caption(f"Cycle: **{ctx['cycle_name'] or 'default'}** — {kpi}")
    if ctx.get("results"):
        st.success("NET results ready (see Step 4).")

# =============================================================================
# Step 1 — Vehicle context
# =============================================================================
st.subheader("1) Vehicle context")


c1, c2, c3, c4 = st.columns([1,1,1,1])
legislation_options = ["WLTP", "EPA", "ABNT (Brazil)"]
# Map legacy "BR" to "ABNT (Brazil)" for compatibility
if ctx["legislation"] == "BR":
    ctx["legislation"] = "ABNT (Brazil)"
if ctx["legislation"] not in legislation_options:
    ctx["legislation"] = legislation_options[0]
ctx["legislation"] = c1.selectbox("Legislation", ["WLTP","EPA","BR"], index=legislation_options.index(ctx["legislation"]))
# categorias oficiais
epa_classes = [
    "Two Seaters", "Minicompact Cars", "Subcompact Cars", "Compact Cars",
    "Midsize Cars", "Large Cars",
    "Small Station Wagons", "Midsize Station Wagons",
    "Small SUVs", "Standard SUVs",
    "Minivans", "Vans",
    "Small Pickup Trucks", "Standard Pickup Trucks"
]
wltp_classes = [
    "Class 1 (<850 kg)", "Class 2 (850–1220 kg)", "Class 3 (>1220 kg)"
]
category_list = epa_classes if ctx["legislation"] == "EPA" else wltp_classes
category_list_upper = [c.upper() for c in category_list]

if ctx["category"] not in category_list_upper:
    ctx["category"] = category_list_upper[0]
ctx["category"]      = c2.selectbox("Category", category_list_upper, index=category_list_upper.index(ctx["category"]))
ctx["make"]          = c3.text_input("Make", ctx["make"])
ctx["model"]         = c4.text_input("Model/Desc.", ctx["model"])
c5, c6 = st.columns([1,3])
ctx["year"]          = c5.number_input("Year", value=int(_try_float(ctx["year"], 2024)), min_value=1900, max_value=2100, step=1)
ctx["notes"]         = c6.text_input("Notes (optional)", ctx["notes"])

st.markdown("---")

# =============================================================================
# Step 2 — Roadloads & Mass (A/B/C, kg)
# =============================================================================
st.subheader("2) How to obtain roadloads (A/B/C) and mass")
mode = st.radio(
    "Choose a mode",
    ["Enter A/B/C + mass (test)", "Use baseline (DB)", "Simple estimate (semi-param)"],
    index=0, horizontal=True
)

if mode == "Enter A/B/C + mass (test)":
    ctx["origin"] = "from_test"
    cc1, cc2, cc3, cc4 = st.columns(4)
    ctx["A"]       = _try_float(cc1.number_input("A [N]", value=_try_float(ctx["A"], 100.0), step=1.0, format="%.3f"))
    ctx["B"]       = _try_float(cc2.number_input("B [N/kph] (may be < 0)", value=_try_float(ctx["B"], 0.00000), step=0.00001, format="%.5f"))
    ctx["C"]       = _try_float(cc3.number_input("C [N/kph²]", value=_try_float(ctx["C"], 0.010000), step=0.000001, format="%.6f"))
    ctx["mass_kg"] = _try_float(cc4.number_input("Mass [kg]", value=_try_float(ctx["mass_kg"], 1500.0), step=1.0, format="%.1f"))

elif mode == "Use baseline (DB)":
    ctx["origin"] = "baseline"
    df = _load_baselines()
    if df.empty:
        st.info("No snapshots in DB yet. Enter values manually or save one first.")
    else:
        # quick filters
        f1, f2, f3 = st.columns(3)
        f_leg = f1.selectbox("Filter by legislation", ["(all)"] + sorted(df["legislation"].dropna().unique().tolist()))
        f_make = f2.selectbox("Filter by make", ["(all)"] + sorted(df["make"].dropna().unique().tolist()))
        f_cat = f3.text_input("Filter by category (contains)", "")

        dfv = df.copy()
        if f_leg != "(all)": dfv = dfv[dfv["legislation"] == f_leg]
        if f_make != "(all)": dfv = dfv[dfv["make"] == f_make]
        if f_cat.strip(): dfv = dfv[dfv["category"].str.contains(f_cat, case=False, na=False)]

        st.dataframe(dfv[["id","legislation","category","make","model","year","A","B","C","mass_kg"]],
                     use_container_width=True, hide_index=True)
        sel_id = st.selectbox("Choose baseline (id)", ["(none)"] + dfv["id"].astype(str).tolist())
        if sel_id != "(none)":
            base = dfv[dfv["id"].astype(str) == sel_id].iloc[0].to_dict()
            ctx["parent_id"] = int(base["id"])
            ctx["A"], ctx["B"], ctx["C"], ctx["mass_kg"] = base["A"], base["B"], base["C"], base["mass_kg"]
            st.success(f"Using baseline #{ctx['parent_id']}: A={ctx['A']:.3f}, B={ctx['B']:.5f}, C={ctx['C']:.6f}, mass={ctx['mass_kg']:.1f} kg")
        else:
            ctx["parent_id"] = None

else:
    ctx["origin"] = "semi_param"
    st.info("Simple estimate: provide approximate values.")
    cc1, cc2, cc3, cc4 = st.columns(4)
    ctx["A"]       = _try_float(cc1.number_input("A [N] (approx.)", value=_try_float(ctx["A"], 120.0), step=1.0, format="%.3f"))
    ctx["B"]       = _try_float(cc2.number_input("B [N/kph] (may be < 0)", value=_try_float(ctx["B"], 0.00000), step=0.00001, format="%.5f"))
    ctx["C"]       = _try_float(cc3.number_input("C [N/kph²] (approx.)", value=_try_float(ctx["C"], 0.012000), step=0.000001, format="%.6f"))
    ctx["mass_kg"] = _try_float(cc4.number_input("Mass [kg] (approx.)", value=_try_float(ctx["mass_kg"], 1550.0), step=1.0, format="%.1f"))

# Validation (B can be < 0)
errors, warns = _validate_abc_mass(ctx["A"], ctx["B"], ctx["C"], ctx["mass_kg"])
if warns:
    for w in warns:
        st.warning(w)

st.markdown("---")

# =============================================================================
# Step 3 — Drive cycle
# =============================================================================
st.subheader("3) Drive cycle")

cleft, cright = st.columns([1,1])
use_default = cleft.button("Use legislation default cycle")
upload = cright.file_uploader("or upload CSV with columns [t, v] (s, m/s)", type=["csv"], accept_multiple_files=False)

if use_default:
    cycle_name = default_cycle_for_legislation(ctx["legislation"])
    df_cycle = _use_standard_cycle(ctx["legislation"])
    ctx["cycle_df"] = df_cycle
    ctx["cycle_name"] = cycle_name


if upload is not None:
    try:
        df_cycle = load_cycle_csv(upload)
        ctx["cycle_df"] = df_cycle
        ctx["cycle_name"] = upload.name
        st.success(f"Cycle loaded: {upload.name}")
    except Exception as e:
        st.error(f"CSV load error: {e}")

# Cycle KPI
if ctx["cycle_df"] is not None:
    kpi, dist_km = _cycle_summary(ctx["cycle_df"])
    st.caption(kpi)
else:
    errors.append("No cycle loaded. Use default or upload a CSV.")

if ctx["cycle_df"] is not None:
    fig = cycle_chart(ctx["cycle_df"])
    if fig:
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# =============================================================================
# Step 4 — Compute VDE_NET
# =============================================================================
st.subheader("4) Compute VDE_NET (normative)")
disabled_calc = len(errors) > 0

if disabled_calc:
    for e in errors:
        st.error(e)

if st.button("Compute VDE_NET", disabled=disabled_calc):
    try:
        # For EPA and WLTP, use your service helpers (kept names)
        if ctx["legislation"] == "EPA":
            # returns dict city/highway/etc.
            res = epa_city_hwy_from_phase(ctx["cycle_df"], ctx["A"], ctx["B"], ctx["C"], ctx["mass_kg"])
        elif ctx["legislation"] == "WLTP":
            res = wltp_phases_from_phase(ctx["cycle_df"], ctx["A"], ctx["B"], ctx["C"], ctx["mass_kg"])
        else:
            # generic fallback
            vde_per_km = compute_vde_net_mj_per_km(ctx["cycle_df"], ctx["A"], ctx["B"], ctx["C"], ctx["mass_kg"])
            res = {"combined_MJ_per_km": vde_per_km}

        ctx["results"] = res
        st.success("NET computed successfully.")

    except Exception as e:
        st.error(f"Failed to compute VDE_NET: {e}")
        ctx["results"] = None

# Read-only results
if ctx["results"]:
    st.info("Badge: **NET (normative)**")
    st.json(ctx["results"])

st.markdown("---")

# =============================================================================
# Step 5 — Save snapshot
# =============================================================================
st.subheader("5) Save snapshot")

def _build_lineage_note():
    origin = ctx["origin"]
    parent = f"baseline#{ctx['parent_id']}" if origin == "baseline" and ctx["parent_id"] else origin
    cycle_tag = ctx["legislation"]
    if ctx["cycle_name"]:
        cycle_tag += f":{ctx['cycle_name']}"
    return f"[lineage] origin={parent}; cycle={cycle_tag}; net_method=normative_v1"

can_save = (ctx["results"] is not None)
if not can_save:
    st.caption("Compute VDE_NET before saving.")

if st.button("Save snapshot", disabled=not can_save):
    try:
        notes = (ctx["notes"] or "").strip()
        # append lineage into notes
        lineage = _build_lineage_note()
        if notes:
            notes = notes + " | " + lineage
        else:
            notes = lineage

        row = {
            "legislation": ctx["legislation"],
            "category": ctx["category"],
            "make": ctx["make"],
            "model": ctx["model"],
            "year": int(ctx["year"]) if str(ctx["year"]).isdigit() else None,
            "notes": notes,
            # main inputs
            "coast_A_N": ctx["A"],
            "coast_B_N_per_kph": ctx["B"],
            "coast_C_N_per_kph2": ctx["C"],
            "inertia_class": ctx["mass_kg"],  # keeping your current schema
            # NET outputs (store compact; adapt to the fields you already use)
            "vde_net_result_json": ctx["results"],  # comment if your schema doesn’t have this yet
        }

        new_id = insert_vde(row)
        st.success(f"Snapshot saved! id={new_id}")
        st.balloons()

    except Exception as e:
        st.error(f"Error saving snapshot: {e}")

# =============================================================================
# Maintenance (optional) — in expander
# =============================================================================
with st.expander("✏️ Maintenance (edit/delete)"):
    st.caption("Use carefully. For major changes prefer creating a new (derived) snapshot.")
    try:
        df_all = _load_baselines()
        if not df_all.empty:
            st.dataframe(df_all[["id","legislation","category","make","model","year","A","B","C","mass_kg"]],
                         use_container_width=True, hide_index=True)
            c1, c2 = st.columns(2)
            rid = c1.text_input("id to delete/edit", "")
            if c2.button("Delete", type="secondary") and rid.strip().isdigit():
                delete_row("vde_db", int(rid))
                st.warning(f"Snapshot #{rid} deleted.")
    except Exception as e:
        st.error(f"Maintenance unavailable: {e}")
