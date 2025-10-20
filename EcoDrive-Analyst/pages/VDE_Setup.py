# pages/0_Mock_Data.py
# ----------------------------------------------------------------------
# Mock Data / Editor
# - Keep sections style: vehicle_basics(), rr_section(), aero_section(), pwt_section(), cycle_section()
# - Baseline: prefill + allow editing EVERYTHING
# - Define all parameters: full manual entry
# - B may be < 0
# ----------------------------------------------------------------------

import streamlit as st
import pandas as pd
from pathlib import Path
import math

# import your own helpers/db as in your project

from src.vde_core.db import ensure_db, fetchall, fetchone, insert_vde, update_vde, delete_row
from src.vde_app.plots import cycle_chart
from src.vde_core.services import   (default_cycle_for_legislation, load_cycle_csv, use_standard_cycle, cycle_summary,
    compute_vde_net_mj_per_km, compute_vde_net, 
    apply_coastdown_deltas, epa_city_hwy_from_phase, wltp_phases_from_phase,load_vde_defaults, estimate_aux_from_coastdown)
from src.vde_core.utils import cycle_kpis, load_tire_catalog
from src.vde_app.components import vde_by_phase, show_vde_feedback, search_logo, get_legislation_icon

st.set_page_config(page_title="Mock Data / Editor", layout="wide")
ensure_db()

DEFAULTS_PATH = Path(
    r"C:\Users\CaioHenriqueFerreira\Downloads\From Git\projects\EcoDrive-Analyst\data\standards\vde_defaults_by_category_trans_elec.csv"
)
# 1) Garanta que o catálogo está carregado (no topo do script da página)
TIRE_CSV =Path(r"C:\Users\CaioHenriqueFerreira\Downloads\From Git\projects\EcoDrive-Analyst\data\standards\tiresize_fromcode_table.csv")


@st.cache_resource(show_spinner=False)
def get_defaults_df():
    return load_vde_defaults(DEFAULTS_PATH)
# -----------------------------
# State
# -----------------------------
if "ctx" not in st.session_state:
    st.session_state.ctx = {
        "legislation": "EPA",
        "category": "",
        "make": "",
        "model": "",
        "year": 2024,
        "notes": "",
        # core VDE inputs
        "A": 100.0, "B": 0.1, "C": 0.03000, "mass_kg": 1300.0,
        # aero / tires (keep names you actually use in your DB; safe fallbacks here)
        "cd": 0.30, "frontal_area_m2": 2.20,"cda_m2": 0.66, "crr1_frac_at_120kph": 0.010, "crr": 0.010,
        # PWT (optional minimal)
        "driveline_eff": 0.90,
        # cycle
        "cycle_df": None,
        "cycle_source": "",
        # baseline id
        "baseline_id": None,
        # mode
        "mode": "From baseline (editable)"
    }

ctx = st.session_state.ctx

# -----------------------------
# Utilities
# -----------------------------
def to_float(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        out = float(x)
        if pd.isna(out):  # cobre NaN do pandas
            return default
        return out
    except Exception:
        return default

def db_list_makes(legislation: str, category: str) -> list[str]:
    rows = fetchall("""
        SELECT DISTINCT make FROM vde_db
        WHERE legislation=? AND category=?
        ORDER BY make
    """, (legislation, category))
    return [r["make"] for r in rows]

def load_baselines_df():
    rows = fetchall("SELECT * FROM vde_db ORDER BY COALESCE(updated_at, created_at) DESC")
    data = []
    for r in rows:
        data.append({
            "id": r.get("id"),
            "legislation": r.get("legislation", ""),
            "category": r.get("category", ""),
            "make": r.get("make", ""),
            "model": r.get("model", r.get("desc","")),
            "year": r.get("year", ""),
            # prefer your actual column names
            "A": to_float(r.get("coast_A_N"), 0.0),
            "B": to_float(r.get("coast_B_N_per_kph"), 0.0),
            "C": to_float(r.get("coast_C_N_per_kph2"), 0.0),
            "mass_kg": to_float(r.get("inertia_class"), to_float(r.get("mass_kg"), 1500.0)),
            # optional extras if they exist in your DB
            "cd": to_float(r.get("cd"), None),
            "frontal_area_m2": to_float(r.get("frontal_area_m2"), None),
            "crr": to_float(r.get("crr"), None),
            "driveline_eff": to_float(r.get("driveline_eff"), None),
            "notes": r.get("notes",""),
        })
    return pd.DataFrame(data) if data else pd.DataFrame(columns=[
        "id","legislation","category","make","model","year","A","B","C","mass_kg",
        "cd","frontal_area_m2","crr","driveline_eff","notes"
    ])

def validate_core(A, B, C, mass_kg):
    errs, warns = [], []
    if A is None or C is None or mass_kg is None:
        errs.append("Fill A, C and Mass with numeric values.")
        return errs, warns
    if A < 0: errs.append("A cannot be negative.")
    # B may be negative (ok)
    if C < 0: errs.append("C cannot be negative.")
    if mass_kg <= 0: errs.append("Mass must be > 0.")
    return errs, warns


def mode_selector():
    ctx = st.session_state.ctx
    st.subheader("Mode")
    prev_mode = ctx.get("mode", "From baseline (editable)")

    ctx["mode"] = st.radio(
        "Mode",
        ["From baseline (editable)", "Define all parameters (no baseline)", "From test (direct coastdown)"],
        index=["From baseline (editable)", "Define all parameters (no baseline)", "From test (direct coastdown)"].index(
            ctx.get("mode", "From baseline (editable)")
        ),
        horizontal=True,
        key="mode_radio",         # <<< CHAVE ÚNICA AQUI
    )

    # se o modo mudou, limpa estado volátil e re-renderiza
    if ctx["mode"] != prev_mode:
        reset_ctx(preserve_meta=True)
        st.session_state["_last_mode"] = ctx["mode"]
        st.rerun()


def show_live_vde_preview():


    # --- helpers leves (fallback se não houver to_float/apply...) ---
    def _to_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return float(default)

    # se você já tem apply_coastdown_deltas importado, pode remover este
    def _apply_cd_deltas(A, B, C, mass_kg, delta_rr_N, delta_brake_N,delta_mass_kg, delta_parasitics_N, delta_aero_Npkph2, crr1_frac_at_120kph):
        # B extra proporcional ao delta_rr via fração @120 km/h, como você já faz
        dB_rr = (delta_rr_N * (crr1_frac_at_120kph / 120.0)) if crr1_frac_at_120kph else 0.0
        A1 = A + delta_rr_N + delta_brake_N + delta_parasitics_N
        B1 = B + dB_rr
        C1 = C + delta_aero_Npkph2
        mass_kg1 = mass_kg + delta_mass_kg # massa mínima arbitrária
        return A1, B1, C1, mass_kg1

    ctx = st.session_state.get("ctx", {})
    df_cycle = ctx.get("cycle_df")
    



    # --- lê entradas e aplica deltas ---
    try:
        A = _to_float(ctx.get("A")); B = _to_float(ctx.get("B")); C = _to_float(ctx.get("C"))
        m = _to_float(ctx.get("mass_kg"), 1500.0)
        leg = str(ctx.get("legislation", "")).upper()

        delta_rr   = _to_float(ctx.get("delta_rr_N"), 0.0)
        delta_br   = _to_float(ctx.get("delta_brake_N"), 0.0)
        delta_par  = _to_float(ctx.get("delta_parasitics_N"), 0.0)
        delta_cda  = _to_float(ctx.get("delta_aero_cdA"), 0.0) * 0.0472068  # cdA→C (N/kph²)
        delta_mass = _to_float(ctx.get("delta_mass_kg"), 0.0)
        frac120    = _to_float(ctx.get("crr1_frac_at_120kph"), 0.0)

        # use a função do projeto se existir; senão usa o fallback local
        try:
            A1, B1, C1, mass_kg1 = apply_coastdown_deltas(
                A, B, C, m,
                delta_rr_N=delta_rr,
                delta_brake_N=delta_br,
                delta_parasitics_N=delta_par,
                delta_aero_Npkph2=delta_cda,
                delta_mass_kg=delta_mass,
                crr1_frac_at_120kph=frac120
            )
        except Exception:
            A1, B1, C1, mass_kg1 = _apply_cd_deltas(A, B, C, m, delta_rr, delta_br, delta_par, delta_cda, delta_mass, frac120)
    except Exception as e:
        st.warning(f"Preview not available (inputs): {e}")
        return

    # --- cálculo por fase (preferencial) + fallback genérico sem compute_vde_net_mj_per_km ---
    total_mj_km, phases = None, {}

    try:
        if "phase" in df_cycle.columns:
            if leg == "EPA":
                res = epa_city_hwy_from_phase(df_cycle, A1, B1, C1, m) or {}
                city = res.get("urb_MJ_km") 
                hwy  = res.get("hwy_MJ_km")  or res.get("hw_MJ_km")  or res.get("hwy_MJ_per_km")
                if city is not None: phases["city"] = float(city)
                if hwy  is not None: phases["hwy"]  = float(hwy)
                print(phases)
                if res.get("net_comb_MJ_km") is not None:
                    total_mj_km = float(res["net_comb_MJ_km"])
                elif ("city" in phases) and ("hwy" in phases):
                    total_mj_km = 0.55*phases["city"] + 0.45*phases["hwy"]

            else:  # WLTP
                res = wltp_phases_from_phase(df_cycle, A1, B1, C1, m) or {}
                for ki, ko in [
                    ("vde_low_mj_per_km","low"),
                    ("vde_mid_mj_per_km","mid"),
                    ("vde_high_mj_per_km","high"),
                    ("vde_extra_high_mj_per_km","xhigh"),
                ]:
                    if res.get(ki) is not None:
                        phases[ko] = float(res[ki])
                if res.get("vde_net_mj_per_km") is not None:
                    total_mj_km = float(res["vde_net_mj_per_km"])

        # --- fallback sem phase/sem total específico: integra direto com compute_vde_net ---
        if total_mj_km is None:
            g = df_cycle.copy()

            # garantir colunas mínimas para o integrador
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

            r = compute_vde_net(g, A1, B1, C1, m)  # <- integrador base já usado nas rotinas por fase
            total_mj_km = float(r["MJ_km"]) if isinstance(r, dict) else float(r)

        # --- UI ---
        st.info(f"Live preview — VDE_NET: **{total_mj_km:.4f} MJ/km**  ({total_mj_km*277.7778:.1f} Wh/km)")
        if phases:
            order = ["city","hwy","low","mid","high","xhigh"]
            ordered = [k for k in order if k in phases] + [k for k in phases if k not in order]
            cols = st.columns(min(4, len(ordered)))
            for i, k in enumerate(ordered):
                cols[i % len(cols)].metric(k.upper(), f"{phases[k]:.4f} MJ/km")

    except Exception as e:
        st.warning(f"Preview not available: {e}")

def init_state():
    if "ctx" not in st.session_state:
        st.session_state.ctx = {}
    ctx = st.session_state.ctx

    # defaults só se a chave não existe (setdefault)
    ctx.setdefault("legislation", "EPA")
    ctx.setdefault("category", "")
    ctx.setdefault("make", "")
    ctx.setdefault("model", "")
    ctx.setdefault("year", 2024)
    ctx.setdefault("notes", "")

    # core VDE inputs (mantém se usuário já digitou)
    ctx.setdefault("A", 120.0)
    ctx.setdefault("B", 0.00000)        # pode ser < 0, UI já permite
    ctx.setdefault("C", 0.012000)
    ctx.setdefault("mass_kg", 1550.0)

    # aero / pneus — só se você realmente usa
    ctx.setdefault("cd", 0.30)
    ctx.setdefault("frontal_area_m2", 2.20)
    ctx.setdefault("cda", 0.66)
    ctx.setdefault("crr", 0.010)
    ctx.setdefault('crr1_frac_at_120kph', 0.010)
    

    # ciclo / origem
    ctx.setdefault("cycle_df", None)
    ctx.setdefault("cycle_source", "")

    # baseline / modo
    ctx.setdefault("baseline_id", None)
    ctx.setdefault("baseline_dict", None)
    ctx.setdefault("vde_id_parent", None)
    ctx.setdefault("from_delta", "Deltas")
    ctx.setdefault("mode", "From baseline (editable)")
    # usado para detectar mudança de modo
    st.session_state.setdefault("_last_mode", ctx["mode"])


def reset_ctx(preserve_meta: bool = True):
    ctx = st.session_state.get("ctx", {})
    meta = {k: ctx.get(k) for k in ("legislation","category","make","model","year","notes","cycle_df","cycle_source")} if preserve_meta else {}
    st.session_state.ctx = {
        **meta,
        "A": 0.0, "B": 0.0, "C": 0.0, "mass_kg": 1500.0,
        "from_delta": "Deltas",
        "delta_rr_N": 0.0, "delta_brake_N": 0.0, "delta_parasitics_N": 0.0, "delta_aero_cdA": 0.0, "delta_mass_kg": 0.0,
        "vde_id_parent": None, "baseline_dict": None,
        # … (outros campos voláteis que você usa nas sections)
    }

    if preserve_meta:
        for k in ["legislation","category","make","model","year","notes","cycle_df","cycle_source"]:
            meta[k] = ctx.get(k)

    # zera blocos voláteis
    volatile_keys = [
        # core inputs
        "A","B","C","mass_kg",
        # deltas
        "delta_rr_N","delta_brake_N","delta_parasitics_N","delta_aero_Npkph2","delta_aero_cdA",
        # pneus / rr auxiliares
        "tire_size","tire_circ_m","diameter_mm","rrc_N_per_kN","crr1_frac_at_120kph",
        "front_pressure_psi","rear_pressure_psi","rr_load_kpa","smerf",
        # parasitic/brake
        "parasitic_A_coef_N","parasitic_B_Npkph","parasitic_C_coef_Npkph2",
        "brake_A_coef_N","brake_B_Npkph","brake_C_coef_Npkph2",
        # baseline
        "baseline_id","baseline_dict","vde_id_parent",

    ]
    for k in volatile_keys:
        if k in ctx:
            del ctx[k]

    # restaura meta, se pedido
    if preserve_meta:
        for k, v in meta.items():
            ctx[k] = v

    # reponha defaults mínimos que você quer após reset (ex: deltas = 0)
    ctx.setdefault("A", 120.0)
    ctx.setdefault("B", 0.00000)
    ctx.setdefault("C", 0.012000)
    ctx.setdefault("mass_kg", 1550.0)
    ctx.setdefault("from_delta", "Deltas")

def show_if_exists(col, path, *, width=64, caption=None):
    p = Path(path) if path else None
    with col:
        if p and p.exists():
            st.image(str(p), width=width, caption=caption)

# -----------------------------
# Sections
# -----------------------------
def vehicle_basics_sidebar():
    # ============ SIDEBAR: vehicle basics & mode ============ #
    with st.sidebar:
        st.header("Vehicle meta")

        # ---- Legislation ----
        leg_opts = ["WLTP", "EPA", "ABNT (Brazil)"]
        if ctx.get("legislation") not in leg_opts:
            ctx["legislation"] = "WLTP"

        c1, c2 = st.columns(2)
        with c1:
            ctx["legislation"] = st.selectbox(
                "Legislation",
                leg_opts,
                index=leg_opts.index(ctx["legislation"]),
                key="sb_leg"
            )
        # ---- Category (depends on legislation) ----
        epa_classes = [
            "Unknown","Two Seaters","Minicompact Cars","Subcompact Cars","Compact Cars",
            "Midsize Cars","Large Cars","Small Station Wagons","Midsize Station Wagons",
            "Small SUVs","Standard SUVs","Minivans","Vans","Small Pickup Trucks","Standard Pickup Trucks"
        ]
        wltp_classes = ["Class 1 (<850 kg)", "Class 2 (850–1220 kg)", "Class 3 (>1220 kg)"]
        if ctx["legislation"] == "EPA":
            category_list = epa_classes
        else:
            category_list = wltp_classes
        category_list_upper = [c.upper() for c in category_list]

        # default de categoria
        if ctx.get("category") not in category_list_upper:
            ctx["category"] = category_list_upper[0]

        with c2:
            ctx["category"] = st.selectbox(
                "Category",
                category_list_upper,
                index=category_list_upper.index(ctx["category"]),
                key="sb_cat"
            )

        # ---- Make / Model ----
        # tenta ler marcas do DB; se falhar, usa fallback
        default_makes = [
            "Toyota","Honda","Nissan","Mitsubishi","Mazda","Subaru","Hyundai","Kia",
            "Volkswagen","Audi","BMW","Mercedes-Benz","Porsche","Peugeot","Renault","Citroën",
            "Fiat","Alfa Romeo","Volvo","Jaguar","Land Rover","Skoda","Seat","Opel",
            "Ford","Chevrolet","Dodge","Chrysler","Jeep","Ram","Cadillac","Buick","GMC",
            "Lincoln","Tesla","Suzuki","Mini","Smart","Lexus","Infiniti","Acura"
        ]
        default_makes_upper = [m.upper() for m in default_makes]
        try:
            ensure_db()
            makes_db = db_list_makes(ctx["legislation"], ctx["category"])  # sua função
            makes_db = [m.upper() for m in makes_db]
        except Exception:
            makes_db = []

        merged_makes = list(dict.fromkeys(makes_db + [m for m in default_makes_upper if m not in makes_db]))
        if "OTHER (TYPE MANUALLY)" not in merged_makes:
            merged_makes.append("OTHER (TYPE MANUALLY)")

        c3, c4 = st.columns(2)
        with c3:
            make_choice = st.selectbox(
                "Make/Brand",
                merged_makes,
                index=(merged_makes.index(ctx["make"].upper()) if ctx.get("make","").upper() in merged_makes else 0),
                key="sb_make_sel"
            )
            if make_choice == "OTHER (TYPE MANUALLY)":
                ctx["make"] = st.text_input("Enter custom brand", value=ctx.get("make",""), key="sb_make_text").upper()
            else:
                ctx["make"] = make_choice

        with c4:
            ctx["model"] = st.text_input("Model/Desc.", value=ctx.get("model",""), key="sb_model")

        # ---- Year & Notes ----
        c5, c6 = st.columns([1, 2])
        with c5:
            ctx["year"] = st.number_input("Year", 1900, 2100, int(ctx.get("year", 2024)), step=1, key="sb_year")
        with c6:
            ctx["notes"] = st.text_input("Proposal Description", value=ctx.get("notes",""), key="sb_notes")

        # ---- Electrification & Transmission ----
        elec_opts  = ["ICE", "HEV", "PHEV", "BEV"]
        trans_opts = ["AT", "AMT", "CVT", "MT", "OT"]
        c7, c8 = st.columns(2)
        with c7:
            ctx["electrification"] = st.selectbox(
                "Electrification",
                elec_opts,
                index=elec_opts.index(ctx.get("electrification","ICE")),
                key="sb_elec"
            )
        with c8:
            ctx["transmission_type"] = st.selectbox(
                "Transmission",
                trans_opts,
                index=trans_opts.index(ctx.get("transmission_type","AT")),
                key="sb_trans"
            )

        st.markdown("---")
        # ---- Mode (com key para não duplicar) ----
        prev_mode = ctx.get("mode", "From baseline (editable)")
        ctx["mode"] = st.radio(
            "Mode",
            ["From baseline (editable)", "Define all parameters (no baseline)", "From test (direct coastdown)"],
            index=["From baseline (editable)", "Define all parameters (no baseline)", "From test (direct coastdown)"].index(prev_mode),
            key="mode_radio"
        )
        if ctx["mode"] != prev_mode:
            reset_ctx(preserve_meta=True)  # sua função
            st.rerun()


def from_test_section():
    """
    Enter coastdown outputs and test mass directly (as obtained from test).
    Keeps compatibility with your old session keys: st.session_state['abc'] / ['manual_mass'].
    """
    ctx = st.session_state.ctx
    st.subheader("From test — direct coastdown (A/B/C) and mass")

    colA, colB, colC, colM = st.columns(4)
    A = colA.number_input("A [N]", 0.0, 500.0, float(ctx.get("A", 30.0)), 0.1)
    B = colB.number_input("B [N/kph]", -1.0, 5.0, float(ctx.get("B", 0.80)), 0.01)  # B may be < 0
    C = colC.number_input("C [N/kph²]", 0.000, 0.100, float(ctx.get("C", 0.011)), 0.001)
    mass = colM.number_input("Test mass [kg]", 300.0, 3500.0, float(ctx.get("mass_kg", 1500.0)), 5.0)

    # write into ctx (new flow)
    ctx["A"], ctx["B"], ctx["C"], ctx["mass_kg"] = to_float(A), to_float(B), to_float(C), to_float(mass)

    # keep old compatibility keys used by your previous compute/save
    st.session_state["abc"] = {"A": float(A), "B": float(B), "C": float(C)}
    st.session_state["manual_mass"] = to_float(mass)

    st.info("Values stored in ctx and in session_state['abc'] / ['manual_mass'] for compatibility.")

def baseline_picker_and_editor():
    """
    Lists vde_db rows, shows VDE metrics, lets you pick one baseline,
    and then either:
      - apply deltas on top of baseline A/B/C (Deltas), or
      - change parameters via sections (Change Parameters).
    """
    st.subheader("Baseline → Prefill + Edit everything")
    ctx = st.session_state.ctx  # <--- garante o ctx

    # 1) Load raw table
    try:
        rows = fetchall("SELECT * FROM vde_db ORDER BY COALESCE(updated_at, created_at) DESC;")
    except Exception as e:
        st.error(f"Could not read vde_db: {e}")
        return

    if not rows:
        st.info("No snapshots in vde_db yet. Add one via 'Compute & Save' first.")
        return

    df = pd.DataFrame(rows)

    # 2) Ensure A/B/C aliases
    if "A" not in df and "coast_A_N" in df: df["A"] = df["coast_A_N"]
    if "B" not in df and "coast_B_N_per_kph" in df: df["B"] = df["coast_B_N_per_kph"]
    if "C" not in df and "coast_C_N_per_kph2" in df: df["C"] = df["coast_C_N_per_kph2"]

    # 3) Quick filters
    with st.expander("Filters"):
        c1, c2, c3, c4 = st.columns(4)
        leg = c1.selectbox("Legislation", ["(all)"] + sorted(df.get("legislation", pd.Series(dtype=str)).dropna().unique().tolist()))
        make = c2.selectbox("Make", ["(all)"] + sorted(df.get("make", pd.Series(dtype=str)).dropna().unique().tolist()))
        cat_contains = c3.text_input("Category contains", "")
        year_eq = c4.text_input("Year (=)", "")

    dfv = df.copy()
    if leg != "(all)" and "legislation" in dfv: dfv = dfv[dfv["legislation"] == leg]
    if make != "(all)" and "make" in dfv: dfv = dfv[dfv["make"] == make]
    if cat_contains.strip() and "category" in dfv:
        dfv = dfv[dfv["category"].astype(str).str.contains(cat_contains, case=False, na=False)]
    if year_eq.strip().isdigit() and "year" in dfv:
        dfv = dfv[dfv["year"] == int(year_eq)]

    if dfv.empty:
        st.warning("No rows after filters.")
        with st.expander("Show raw columns (debug)"):
            st.write(sorted(df.columns.tolist()))
        return

    # 4) Compact grid with VDE metrics (+ tire info)
    cols_to_show = [
                    # meta
                    "id","created_at","updated_at",
                    "legislation","category","make","model","year","notes",

                    # powertrain
                    "engine_type","engine_model","engine_size_l","engine_aspiration",
                    "transmission_type","transmission_model","drive_type",

                    # massa / aero
                    "mass_kg","inertia_class","cda_m2","weight_dist_fr_pct","payload_kg",
                    "mro_kg","options_kg","wltp_category",

                    # pneus / RR
                    "tire_size","tire_rr_note","smerf","front_pressure_psi","rear_pressure_psi",
                    "rrc_N_per_kN","crr1_frac_at_120kph","rr_load_kpa",

                    # coastdown principais
                    "coast_A_N","coast_B_N_per_kph","coast_C_N_per_kph2",

                    # coef. adicionais (transmissão/freio/aero)
                    "trans_A_coef_N","trans_B_Npkph","trans_C_coef_Npkph2",
                    "brake_A_coef_N","brake_B_Npkph","brake_C_coef_Npkph2",
                    "aero_C_coef_Npkph2",

                    # modelo RR avançado (opcional)
                    "rr_alpha_N","rr_beta_Npkph","rr_a_Npkph2","rr_b_N","rr_c_Npkph",

                    # ciclo
                    "cycle_name","cycle_source",

                    # resultados agregados
                    "vde_urb_mj","vde_hw_mj",
                    "vde_net_mj_per_km","vde_total_mj_per_km",
                    "vde_urb_mj_per_km","vde_hw_mj_per_km",
                    "vde_low_mj_per_km","vde_mid_mj_per_km","vde_high_mj_per_km","vde_extra_high_mj_per_km",

                    # rastreabilidade mínima de baseline
                    "vde_id_parent","baseline_A_N","baseline_B_N_per_kph","baseline_C_N_per_kph2","baseline_mass_kg",

                    # deltas aplicados sobre o baseline
                    "delta_rr_N","delta_brake_N","delta_parasitics_N","delta_aero_Npkph2", "delta_mass_kg",
                ]

    cols_to_show = [c for c in cols_to_show if c in dfv.columns]
    st.dataframe(
        dfv[cols_to_show].sort_values("id", ascending=False),
        use_container_width=True, hide_index=True
    )

    # 5) Picker
    options = dfv.sort_values("id", ascending=False)["id"].astype(int).tolist()
    sel_id = st.selectbox("Pick baseline id", options)
    base = dfv[dfv["id"] == sel_id].iloc[0].to_dict()
    

    # Guardar baseline para o fluxo de deltas/save
    st.session_state.ctx["vde_id_parent"] = int(sel_id)
    st.session_state.ctx["baseline_dict"] = {
        # core coastdown
        "A": base.get("A", base.get("coast_A_N")),
        "B": base.get("B", base.get("coast_B_N_per_kph")),
        "C": base.get("C", base.get("coast_C_N_per_kph2")),
        "mass_kg": base.get("mass_kg", base.get("inertia_class")),

        # contexto mínimo
        "legislation": base.get("legislation"),
        "category":    base.get("category"),

        # pneus / RR
        "tire_size":           base.get("tire_size"),
        "rrc_N_per_kN":        base.get("rrc_N_per_kN"),
        "crr1_frac_at_120kph": base.get("crr1_frac_at_120kph"),
        "front_pressure_psi":  base.get("front_pressure_psi"),
        "rear_pressure_psi":   base.get("rear_pressure_psi"),
        "rr_load_kpa":         base.get("rr_load_kpa"),
        "smerf":               base.get("smerf"),

        # parasitics & brake (se existirem no registro)
        "parasitic_A_coef_N":      base.get("parasitic_A_coef_N"),
        "parasitic_B_Npkph":       base.get("parasitic_B_Npkph"),
        "parasitic_C_coef_Npkph2": base.get("parasitic_C_coef_Npkph2"),
        "brake_A_coef_N":          base.get("brake_A_coef_N"),
        "brake_B_Npkph":           base.get("brake_B_Npkph"),
        "brake_C_coef_Npkph2":     base.get("brake_C_coef_Npkph2"),

        # opcionais p/ serviços de defaults/decompose
        "electrification":   base.get("electrification"),
        "transmission_type": base.get("transmission_type"),
        "cda_m2":            base.get("cda_m2"),
    }



    prev_from_delta = ctx.get("from_delta", "Deltas")
    ctx["from_delta"] = st.radio(
        "How do you want to calculate on baseline?",
        ["Deltas", "Change Parameters"],
        index=["Deltas", "Change Parameters"].index(prev_from_delta),
        horizontal=True,
        key="baseline_flow_radio",   # <<< CHAVE ÚNICA AQUI
)


    if ctx["from_delta"] == "Deltas":
        # Preenche A/B/C/massa com o baseline para que o preview/cálculo use isso + deltas
        ctx["A"] = float(base.get("A", base.get("coast_A_N", 0.0)) or 0.0)
        ctx["B"] = float(base.get("B", base.get("coast_B_N_per_kph", 0.0)) or 0.0)
        ctx["C"] = float(base.get("C", base.get("coast_C_N_per_kph2", 0.0)) or 0.0)
        ctx["mass_kg"] = float(base.get("mass_kg", base.get("inertia_class", 0.0)) or 0.0)
        

        # adições diretas (sem helper), úteis pro ΔB e consistência
        if base.get("crr1_frac_at_120kph") is not None:
            ctx["crr1_frac_at_120kph"] = to_float(base["crr1_frac_at_120kph"])
        if base.get("rrc_N_per_kN") is not None:
            ctx["rrc_N_per_kN"] = to_float(base["rrc_N_per_kN"])
        if base.get("tire_size"):
            ctx["tire_size"] = str(base["tire_size"])

        with st.expander("Δ Deltas from baseline"):
            c1, c2 = st.columns(2)
            ctx["delta_rr_N"]         = c1.number_input("ΔRR (A) [N]", value=float(ctx.get("delta_rr_N", 0.0)), step=0.1)
            ctx["delta_aero_cdA"]  = c2.number_input("ΔAero (CdA) [m2]", value=float(ctx.get("delta_aero_cdA", 0.0)), step=0.001, format="%.3f")
            c3, c4, c5 = st.columns(3)
            ctx["delta_brake_N"]      = c3.number_input("ΔBrake (A) [N]", value=float(ctx.get("delta_brake_N", 0.0)), step=0.1)
            ctx["delta_parasitics_N"] = c4.number_input("ΔParasitics (A) [N]", value=float(ctx.get("delta_parasitics_N", 0.0)), step=0.1)
            ctx["delta_mass_kg"] = c5.number_input("ΔMass [kg]", value=float(ctx.get("delta_mass_kg", 0.0)), step=1.0)
    else:
        # Editar parâmetros (mantém suas sections)
        # Carrega o catálogo de pneus se disponível
        tires_df = None
        try:
            tires_df = load_tire_catalog(TIRE_CSV)  # TIRE_CSV deve apontar para seu CSV 2-colunas
        except Exception:
            tires_df = None  # rr_section deve lidar com tires_df=None

        st.success(f"Editing baseline #{base.get('id', '')} (all fields below are editable).")

        rr_section(
            prefill={
                "rrc_N_per_kN":        base.get("rrc_N_per_kN"),
                "crr1_frac_at_120kph": base.get("crr1_frac_at_120kph"),
                "mass_kg":             base.get("mass_kg", base.get("inertia_class")),
                "tire_size":           base.get("tire_size"),
            },
            tires_df=tires_df
        )
        aero_section(
            prefill={
                "cd":                base.get("cd"),
                "frontal_area_m2":   base.get("frontal_area_m2"),
                "cda_m2":            base.get("cda_m2"),
            }
        )
        parasitic_brake_section(
            prefill={
                "parasitic_A_coef_N":      base.get("parasitic_A_coef_N"),
                "parasitic_B_Npkph":       base.get("parasitic_B_Npkph"),
                "parasitic_C_coef_Npkph2": base.get("parasitic_C_coef_Npkph2"),
                "brake_A_coef_N":          base.get("brake_A_coef_N"),
                "brake_B_Npkph":           base.get("brake_B_Npkph"),
                "brake_C_coef_Npkph2":     base.get("brake_C_coef_Npkph2"),
            }
        )

    # 7) Debug (opcional)
    with st.expander("Baseline snapshot (debug)"):
        key_cols = [
            "id","legislation","category","make","model","year","mass_kg","A","B","C",
            "vde_net_mj_per_km","vde_urb_mj_per_km","vde_hw_mj_per_km",
            "vde_low_mj_per_km","vde_mid_mj_per_km","vde_high_mj_per_km","vde_extra_high_mj_per_km"
        ]
        st.write({k: base.get(k) for k in key_cols if k in base})

def rr_section(prefill=None, tires_df=None):
    """
    RR only (não mexe em A/B/C):
      IN: rrc_N_per_kN [N/kN], crr1_frac_at_120kph [-], mass_kg [kg]
      PLUS: selectbox de pneu (opcional, se tires_df for passado)
      OUT: rr_alpha_N [N], rr_beta_Npkph [N/kph]; salva tire_size no ctx
    """
    ctx = st.session_state.ctx
    st.subheader("Rolling Resistance")

    # --- Tire select (opcional) ---
    if isinstance(tires_df, pd.DataFrame) and not tires_df.empty:
        sizes = tires_df["tire_size"].tolist()
        # valor inicial
        size0 = prefill.get("tire_size") if prefill else ctx.get("tire_size")
        try:
            idx0 = sizes.index(size0) if size0 in sizes else 0
        except Exception:
            idx0 = 0
        sel = st.selectbox("Tire size", sizes, index=idx0)
        ctx["tire_size"] = sel
        # info rápida do pneu
        trow = tires_df.loc[tires_df["tire_size"] == sel].iloc[0].to_dict()
        st.caption(f'Ø {trow["tire_circ_mm"]:.0f} mm ')
        # se quiser guardar no ctx:
        ctx["tire_circ_m"] = float(trow["tire_circ_mm"]) / 1000.0  # m

    # --- Prefill de RR ---
    if prefill:
        rrc0  = to_float(prefund := prefill.get("rrc_N_per_kN"), ctx.get("rrc_N_per_kN", 9.5))
        frac0 = to_float(prefill.get("crr1_frac_at_120kph"), ctx.get("crr1_frac_at_120kph", 0.10))
        m0    = to_float(prefill.get("mass_kg"), ctx.get("mass_kg", ctx.get("inertia_class", 1500.0)))
    else:
        rrc0  = to_float(ctx.get("rrc_N_per_kN"), 9.5)
        frac0 = to_float(ctx.get("crr1_frac_at_120kph"), 0.10)
        m0    = to_float(ctx.get("mass_kg", ctx.get("inertia_class")), 1500.0)

    c1, c2, c3 = st.columns(3)
    ctx["rrc_N_per_kN"]        = c1.number_input("RRC [N/kN]", value=float(rrc0), step=0.1, format="%.2f")
    ctx["crr1_frac_at_120kph"] = c2.number_input("Frac @120 kph", value=float(frac0),
                                                  min_value=0.0, max_value=1.0, step=0.005, format="%.3f")
    ctx["mass_kg"]             = c3.number_input("Mass [kg]", value=float(m0), step=1.0, format="%.1f")

    # --- Cálculo RR ---
    G = 9.80665
    load_kN = (ctx["mass_kg"] * G) / 1000.0 if ctx.get("mass_kg") else 0.0
    A_rr = (ctx["rrc_N_per_kN"] or 0.0) * load_kN
    B_rr = A_rr * ((ctx["crr1_frac_at_120kph"] or 0.0) / 120.0)

    ctx["rr_alpha_N"]    = float(A_rr)
    ctx["rr_beta_Npkph"] = float(B_rr)
    ctx["smerf_est_N"]   = float(A_rr)

    c4, c5, c6 = st.columns(3)
    c4.metric("Load [kN]", f"{load_kN:.2f}")
    c5.metric("A_rr ≈ SMERF [N]", f"{A_rr:.2f}")
    c6.metric("B_rr [N/kph]", f"{B_rr:.4f}")

def aero_section(prefill=None):
    """
    Usa cda_m2 (DB). Exibe C_aero estimado (N/kph²) como referência.
    Não altera o C medido do coastdown.
    """
    ctx = st.session_state.ctx
    st.subheader("Aerodynamics")

    cda0 = to_float(prefill.get("cda_m2"), ctx.get("cda_m2")) if prefill else to_float(ctx.get("cda_m2"), None)
    cda = st.number_input("CdA [m²]", value=float(cda0 or 0.0), step=0.01, format="%.3f")
    ctx["cda_m2"] = to_float(cda)

    RHO = 1.2
    C_aero = 0.5 * RHO * ctx["cda_m2"] * (1/3.6)**2  # N/kph²
    ctx["aero_C_coef_Npkph2"] = C_aero

    st.metric("C_aero (est.) [N/kph²]", f"{C_aero:.6f}")
    st.caption("O coastdown C medido permanece em 'coast_C_N_per_kph2'; isto é referencial.")

def parasitic_brake_section(prefill=None):
    """
    Parasitics + Brake numa única seção (DB fields):
      parasitic_A/B/C, brake_A/B/C  (todas opcionais; default 0)
    """
    ctx = st.session_state.ctx
    st.subheader("Parasitics + Brake")

    if prefill:
        parA0 = to_float(prefill.get("parasitic_A_coef_N"),   ctx.get("parasitic_A_coef_N", 0.0))
        parB0 = to_float(prefill.get("parasitic_B_Npkph"),    ctx.get("parasitic_B_Npkph", 0.0))
        parC0 = to_float(prefill.get("parasitic_C_coef_Npkph2"), ctx.get("parasitic_C_coef_Npkph2", 0.0))
        brA0  = to_float(prefill.get("brake_A_coef_N"),       ctx.get("brake_A_coef_N", 0.0))
        brB0  = to_float(prefill.get("brake_B_Npkph"),        ctx.get("brake_B_Npkph", 0.0))
        brC0  = to_float(prefill.get("brake_C_coef_Npkph2"),  ctx.get("brake_C_coef_Npkph2", 0.0))
    else:
        parA0 = to_float(ctx.get("parasitic_A_coef_N"),   0.0)
        parB0 = to_float(ctx.get("parasitic_B_Npkph"),    0.0)
        parC0 = to_float(ctx.get("parasitic_C_coef_Npkph2"), 0.0)
        brA0  = to_float(ctx.get("brake_A_coef_N"),       0.0)
        brB0  = to_float(ctx.get("brake_B_Npkph"),        0.0)
        brC0  = to_float(ctx.get("brake_C_coef_Npkph2"),  0.0)

    p1, p2, p3 = st.columns(3)
    ctx["parasitic_A_coef_N"]      = p1.number_input("Parasitic A [N]", value=float(parA0), step=0.1, format="%.2f")
    ctx["parasitic_B_Npkph"]       = p2.number_input("Parasitic B [N/kph]", value=float(parB0), step=0.001, format="%.5f")
    ctx["parasitic_C_coef_Npkph2"] = p3.number_input("Parasitic C [N/kph²]", value=float(parC0), step=0.0001, format="%.6f")

    b1, b2, b3 = st.columns(3)
    ctx["brake_A_coef_N"]      = b1.number_input("Brake A [N]", value=float(brA0), step=0.1, format="%.2f")
    ctx["brake_B_Npkph"]       = b2.number_input("Brake B [N/kph]", value=float(brB0), step=0.001, format="%.5f")
    ctx["brake_C_coef_Npkph2"] = b3.number_input("Brake C [N/kph²]", value=float(brC0), step=0.0001, format="%.6f")

    c1, c2 = st.columns(2)
    c1.metric("A_par + A_brake [N]", f"{ctx['parasitic_A_coef_N'] + ctx['brake_A_coef_N']:.2f}")
    c2.metric("B_par + B_brake [N/kph]", f"{ctx['parasitic_B_Npkph'] + ctx['brake_B_Npkph']:.5f}")

def auxiliaries_section():
    """
    Usa A/B/C + mass + (category, electrification, transmission_type) do ctx
    para decompor o coastdown (NET) usando os defaults do CSV.
    """
    ctx = st.session_state.ctx
    st.subheader("Estimate auxiliaries from coastdown (NET)")

    # habilita o botão só se os inputs mínimos existem
    missing = [k for k in ("A","B","C","mass_kg","category") if ctx.get(k) in (None, "")]
    disabled = len(missing) > 0
    if disabled:
        st.caption(f"Fill first: {', '.join(missing)}")

    if st.button("Estimate using defaults CSV", disabled=disabled):
        res = estimate_aux_from_coastdown(
            A_N=ctx["A"],
            B_N_per_kph=ctx["B"],          # B pode ser < 0
            C_N_per_kph2=ctx["C"],
            mass_kg=ctx["mass_kg"],
            category=ctx["category"],
            electrification=ctx.get("electrification", "ICE"),
            transmission_type=ctx.get("transmission_type", "AT"),
            cdA_override_m2=ctx.get("cda_m2"),         # opcional
            defaults_df=get_defaults_df(),             # já cacheado
        )

        # joga no ctx para uso posterior (compute/save, etc.)
        ctx.update({
            "rr_alpha_N": res["rr_alpha_N"],
            "rr_beta_Npkph": res["rr_beta_Npkph"],
            "aero_C_coef_Npkph2": res["aero_C_coef_Npkph2"],
            "parasitic_A_N": res["parasitic_A_coef_N"],
            "parasitic_B_Npkph": res["parasitic_B_coef_Npkph"],
            "parasitic_C_Npkph2": res["parasitic_C_coef_Npkph2"],
            "decomp_check_ok": res["check_ok"],
            "cda_m2": res["cdA_used_m2"],
        })

        # feedback compacto
        c1, c2, c3 = st.columns(3)
        c1.metric("RR α [N]", f"{res['rr_alpha_N']:.2f}")
        c2.metric("RR β [N/kph]", f"{res['rr_beta_Npkph']:.3f}")
        c3.metric("Aero C [N/kph²]", f"{res['aero_C_coef_Npkph2']:.3f}")
        d1, d2, d3 = st.columns(3)
        d1.metric("Parasitic A [N]", f"{res['parasitic_A_coef_N']:.2f}")
        d2.metric("Parasitic B [N/kph]", f"{res['parasitic_B_coef_Npkph']:.3f}")
        d3.metric("Check", "OK" if res["check_ok"] else "Review")

def cycle_section():

    # 1) pegue o estado logo no começo
    ctx = st.session_state.ctx
    st.subheader("Drive cycle")
    # Validation (B can be < 0)
    errors, warns = validate_core(ctx["A"], ctx["B"], ctx["C"], ctx["mass_kg"])
    if warns:
        for w in warns:
            st.warning(w)
    cleft, cright = st.columns([1,1])
    use_default = cleft.button("Use legislation default cycle")
    upload = cright.file_uploader("or upload CSV with columns [t, v] (s, m/s)", type=["csv"], accept_multiple_files=False)

    if use_default:
        cycle_name = default_cycle_for_legislation(ctx["legislation"])
        df_cycle = use_standard_cycle(ctx["legislation"] )
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
        kpi, dist_km = cycle_summary(ctx["cycle_df"])
        st.caption(kpi)
    else:
        errors.append("No cycle loaded. Use default or upload a CSV.")

    if ctx["cycle_df"] is not None:
        fig = cycle_chart(ctx["cycle_df"])
        if fig:
            st.plotly_chart(fig, use_container_width=True)

# ==============================
# Compute & Save (function)
# ==============================

def compute_and_save():
    ctx = st.session_state.ctx
    st.markdown("---")
    st.subheader("Compute VDE and Save to DB")

    # --- validação básica ---
    errs, warns = validate_core(ctx["A"], ctx["B"], ctx["C"], ctx["mass_kg"])
    for w in (warns or []): st.warning(w)
    if ctx.get("cycle_df") is None:
        errs.append("Cycle not loaded. Pick default or upload a CSV.")
    for e in (errs or []): st.error(e)
    disabled_btn = bool(errs)

    # --- meta ---
    leg  = str(ctx["legislation"])
    cat  = ctx["category"]
    make = ctx["make"]; model = ctx["model"]
    year = int(ctx["year"]) if str(ctx["year"]).isdigit() else None
    notes = ctx["notes"]
    cycle_name   = default_cycle_for_legislation(leg)   # nome padrão
    cycle_source = ctx.get("cycle_source", f"standard:{leg}")

    if st.button("Compute VDE_NET and Save", key="btn_compute_save_main", disabled=disabled_btn):
        try:
            df_cycle = ctx["cycle_df"]
            A = float(ctx["A"]); B = float(ctx["B"]); C = float(ctx["C"]); mass_kg = float(ctx["mass_kg"])

            # --- APLICA DELTAS → A1/B1/C1 ---
            d_rr   = to_float(ctx.get("delta_rr_N"), 0.0)
            d_br   = to_float(ctx.get("delta_brake_N"), 0.0)
            d_par  = to_float(ctx.get("delta_parasitics_N"), 0.0)
            d_cda  = to_float(ctx.get("delta_aero_cdA"), 0.0) * 0.0472068  # CdA→C (N/kph²)
            d_mass  = to_float(ctx.get("delta_mass_kg"), 0.0) # CdA→C (N/kph²)
            frac120 = to_float(ctx.get("crr1_frac_at_120kph"), 0.0)
            
            dB_rr = d_rr * (frac120 / 120.0) if frac120 else 0.0
            A1 = A + d_rr + d_br + d_par
            B1 = B + dB_rr
            C1 = C + d_cda
            mass_kg1 = mass_kg + d_mass

            # --- CÁLCULO ESPECÍFICO POR FASE (prioritário) ---
            total_mj_km = None
            by_phase = {}
            if isinstance(df_cycle, pd.DataFrame) and ("phase" in df_cycle.columns):
                if leg.upper() == "EPA":
                    res = epa_city_hwy_from_phase(df_cycle, A1, B1, C1, mass_kg1) or {}
                    # padroniza chaves internas do preview
                    city = res.get("city_MJ_km") or res.get("urb_MJ_km") or res.get("city_MJ_per_km")
                    hwy  = res.get("hwy_MJ_km")  or res.get("hw_MJ_km")  or res.get("hwy_MJ_per_km")
                    if city is not None: by_phase["city"] = float(city)
                    if hwy  is not None: by_phase["hwy"]  = float(hwy)
                    if res.get("net_comb_MJ_km") is not None:
                        total_mj_km = float(res["net_comb_MJ_km"])
                    elif ("city" in by_phase) and ("hwy" in by_phase):
                        total_mj_km = 0.55*by_phase["city"] + 0.45*by_phase["hwy"]

                else:  # WLTP
                    res = wltp_phases_from_phase(df_cycle, A1, B1, C1, mass_kg1) or {}
                    mapping = [
                        ("vde_low_mj_per_km","low"), ("vde_mid_mj_per_km","mid"),
                        ("vde_high_mj_per_km","high"), ("vde_extra_high_mj_per_km","xhigh")
                    ]
                    for ki, ko in mapping:
                        if res.get(ki) is not None:
                            by_phase[ko] = float(res[ki])
                    if res.get("vde_net_mj_per_km") is not None:
                        total_mj_km = float(res["vde_net_mj_per_km"])

            # --- fallback (sem phase/sem total específico) ---
            if total_mj_km is None:
                r_all = compute_vde_net_mj_per_km(df_cycle, A1, B1, C1, mass_kg)
                total_mj_km = float(r_all["MJ_km"]) if isinstance(r_all, dict) else float(r_all)

            # --- feedback imediato ---
            st.info(f"VDE (NET): **{total_mj_km:.4f} MJ/km**  ({total_mj_km*277.7778:.1f} Wh/km)")
            if by_phase:
                order = ["city","hwy","low","mid","high","xhigh"]
                keys = [k for k in order if k in by_phase] + [k for k in by_phase if k not in order]
                cols = st.columns(min(4, len(keys)))
                for i, k in enumerate(keys):
                    label = {"city":"CITY", "hwy":"HWY"}.get(k, k.upper())
                    cols[i % len(cols)].metric(label, f"{float(by_phase[k]):.4f} MJ/km")

            # --- (opcional) decompor auxiliares com A1/B1/C1 ---
            decomp = None
            try:
                defaults_df = load_vde_defaults(DEFAULTS_PATH)
                decomp = estimate_aux_from_coastdown(
                    A_N=A1, B_N_per_kph=B1, C_N_per_kph2=C1, mass_kg=mass_kg,
                    category=cat,
                    electrification=ctx.get("electrification","ICE"),
                    transmission_type=ctx.get("transmission_type","AT"),
                    cdA_override_m2=ctx.get("cda_m2"),
                    defaults_df=defaults_df,
                )
            except Exception:
                pass

            # --- monta row (usa A1/B1/C1 e inclui fases por km se disponíveis) ---
            row = {
                "legislation": leg, "category": cat,
                "make": make, "model": model, "year": year, "notes": notes,
                "mass_kg": mass_kg,
                "coast_A_N": A1, "coast_B_N_per_kph": B1, "coast_C_N_per_kph2": C1,
                "cycle_name": cycle_name, "cycle_source": cycle_source,
                "vde_net_mj_per_km": total_mj_km,
                # deltas aplicados
                "delta_rr_N": d_rr, "delta_brake_N": d_br, "delta_mass_kg": d_mass,
                "delta_parasitics_N": d_par, "delta_aero_Npkph2": d_cda,
            }

            # EPA phases → DB
            if "city" in by_phase: row["vde_urb_mj_per_km"] = float(by_phase["city"])
            if "hwy"  in by_phase: row["vde_hw_mj_per_km"]  = float(by_phase["hwy"])
            # WLTP phases → DB
            if "low"  in by_phase: row["vde_low_mj_per_km"]        = float(by_phase["low"])
            if "mid"  in by_phase: row["vde_mid_mj_per_km"]        = float(by_phase["mid"])
            if "high" in by_phase: row["vde_high_mj_per_km"]       = float(by_phase["high"])
            if "xhigh" in by_phase: row["vde_extra_high_mj_per_km"] = float(by_phase["xhigh"])

            # campos extras do ctx (se existirem)
            for k in [
                "engine_type","engine_model","engine_size_l","engine_aspiration",
                "transmission_type","transmission_model","drive_type",
                "inertia_class","cda_m2","weight_dist_fr_pct","payload_kg",
                "mro_kg","options_kg","wltp_category",
                "tire_size","tire_rr_note","smerf","front_pressure_psi","rear_pressure_psi",
                "rrc_N_per_kN","crr1_frac_at_120kph","rr_load_kpa",
                "trans_A_coef_N","trans_B_coef_Npkph","trans_C_coef_Npkph2",
                "brake_A_coef_N","brake_B_coef_Npkph","brake_C_coef_Npkph2",
                "parasitic_A_coef_N","parasitic_B_coef_Npkph","parasitic_C_coef_Npkph2",
                "aero_C_coef_Npkph2",
                "rr_alpha_N","rr_beta_Npkph","rr_a_Npkph2","rr_b_N","rr_c_Npkph",
            ]:
                v = ctx.get(k, None)
                if v not in (None, ""): row[k] = v

            # baseline mínimo (se veio do picker)
            base = ctx.get("baseline_dict")
            if ctx.get("vde_id_parent") and isinstance(base, dict):
                row.update({
                    "vde_id_parent": ctx["vde_id_parent"],
                    "baseline_A_N": base.get("A"),
                    "baseline_B_N_per_kph": base.get("B"),
                    "baseline_C_N_per_kph2": base.get("C"),
                    "baseline_mass_kg": base.get("mass_kg"),
                })

            # merge da decomposição (se disponível)
            if decomp:
                row.update({k: float(v) for k, v in {
                    "rr_alpha_N": decomp.get("rr_alpha_N"),
                    "rr_beta_Npkph": decomp.get("rr_beta_Npkph"),
                    "aero_C_coef_Npkph2": decomp.get("aero_C_coef_Npkph2"),
                    "parasitic_A_coef_N": decomp.get("parasitic_A_coef_N"),
                    "parasitic_B_Npkph": decomp.get("parasitic_B_Npkph"),
                    "parasitic_C_coef_Npkph2": decomp.get("parasitic_C_coef_Npkph2"),
                }.items() if v is not None})

            # ... depois de montar `row` ...
            row = {k: v for k, v in row.items()
                if v is not None
                and (not isinstance(v, str) or v.strip() != "")
                and (not isinstance(v, (int, float)) or math.isfinite(float(v)))}


            # --- INSERT ---
            vde_id = insert_vde({k: v for k, v in row.items() if v is not None})
            st.session_state["vde_id"] = vde_id

            # --- UPDATE por fase (reusa A1/B1/C1) ---
            if isinstance(df_cycle, pd.DataFrame) and "phase" in df_cycle.columns:
                upd = {}
                if leg.upper() == "EPA":
                    res = epa_city_hwy_from_phase(df_cycle, A1, B1, C1, mass_kg) or {}
                    if res.get("urb_MJ")  is not None: upd["vde_urb_mj"] = float(res["urb_MJ"])
                    if res.get("hw_MJ")   is not None: upd["vde_hw_mj"]  = float(res["hw_MJ"])
                    if res.get("net_comb_MJ_km") is not None:
                        upd["vde_net_mj_per_km"] = float(res["net_comb_MJ_km"])
                else:
                    res = wltp_phases_from_phase(df_cycle, A1, B1, C1, mass_kg) or {}
                    for k in ("vde_low_mj_per_km","vde_mid_mj_per_km","vde_high_mj_per_km","vde_extra_high_mj_per_km"):
                        if res.get(k) is not None: upd[k] = float(res[k])
                    if res.get("vde_net_mj_per_km") is not None:
                        upd["vde_net_mj_per_km"] = float(res["vde_net_mj_per_km"])
                if upd:
                    update_vde(vde_id, upd)

            st.success(f"Saved VDE snapshot (id={vde_id}).")
            # limpa estado volátil e volta “zerado” mantendo meta
            reset_ctx(preserve_meta=True)
            st.rerun()

        except Exception as e:
            st.error(f"Failed to compute/save VDE: {e}")

# ====================================
# Edit / Delete (function)
# ====================================

def edit_or_delete():
    st.markdown("---")
    st.subheader("✏️ Edit / Delete an existing VDE row")

    rows = fetchall("""
        SELECT id, legislation, category, make, model, year,
               coast_A_N, coast_B_N_per_kph, coast_C_N_per_kph2, mass_kg, notes
        FROM vde_db
        ORDER BY id DESC
        LIMIT 100
    """)

    if not rows:
        st.info("No VDE rows saved yet.")
        return

    labels = [
        f'#{r["id"]} — {r["legislation"]} | {r["category"]} | {r["make"]} {r["model"]} ({r.get("year","")})'
        for r in rows
    ]
    idx = st.selectbox("Pick a VDE to edit/delete", list(range(len(labels))), format_func=lambda i: labels[i])
    sel = rows[idx]
    vde_id_edit = sel["id"]
    st.caption(f"Editing VDE id: {vde_id_edit}")

    # optional: linked scenarios count
    try:
        dep = fetchall("SELECT COUNT(*) AS n FROM fuelcons_db WHERE vde_id=?", (vde_id_edit,))
        st.caption(f'Linked scenarios in fuelcons_db: {dep[0]["n"] if dep else 0}')
    except Exception:
        pass

    # --- Edit form (B can be negative) ---
    with st.form(key=f"edit_vde_{vde_id_edit}"):
        c1, c2, c3, c4 = st.columns(4)
        A_edit = c1.number_input("A [N]", 0.0, 5000.0, float(sel["coast_A_N"] or 0.0), 0.1)
        B_edit = c2.number_input("B [N/kph]", -5.0, 5.0, float(sel["coast_B_N_per_kph"] or 0.0), 0.01)
        C_edit = c3.number_input("C [N/kph²]", 0.000000, 1.000000, float(sel["coast_C_N_per_kph2"] or 0.0), 0.000001)
        M_edit = c4.number_input("Mass [kg]", 1.0, 4000.0, float(sel["mass_kg"] or 0.0), 1.0)

        c5, c6, c7 = st.columns(3)
        make_edit  = c5.text_input("Make",  value=sel["make"] or "")
        model_edit = c6.text_input("Model", value=sel["model"] or "")
        year_edit  = c7.number_input("Year", 1990, 2100, int(sel["year"] or 2020))
        notes_edit = st.text_area("Notes", value=sel["notes"] or "")

        save_btn = st.form_submit_button("💾 Save changes")
        if save_btn:
            try:
                
                # 1) Persist core fields
                update_vde(vde_id_edit, {
                    "coast_A_N": A_edit,
                    "coast_B_N_per_kph": B_edit,
                    "coast_C_N_per_kph2": C_edit,
                    "mass_kg": M_edit,
                    "make": make_edit,
                    "model": model_edit,
                    "year": int(year_edit),
                    "notes": notes_edit,
                })

                # 1a) Persist RR fields if rr_section stored them in ctx
                rr_updates = {}
                if "ctx" in st.session_state:
                    for k in [
                        "tire_size","rrc_N_per_kN", "crr1_frac_at_120kph",
                        "front_pressure_psi", "rear_pressure_psi",
                        "rr_load_kpa", "smerf",
                    ]:
                        v = st.session_state.ctx.get(k)
                        if v not in (None, ""):
                            rr_updates[k] = v
                if rr_updates:
                    update_vde(vde_id_edit, rr_updates)

                # 1b) Persist parasitic + brake if present in ctx
                pb_updates = {}
                if "ctx" in st.session_state:
                    for k in [
                        "parasitic_A_coef_N","parasitic_B_coef_Npkph","parasitic_C_coef_Npkph2",
                        "brake_A_coef_N","brake_B_Npkph","brake_C_coef_Npkph2",
                    ]:
                        v = st.session_state.ctx.get(k)
                        if v is not None:
                            pb_updates[k] = v
                if pb_updates:
                    update_vde(vde_id_edit, pb_updates)

                # 2) Optional decomposition (same as compute_and_save)
                try:
                    defaults_df = load_vde_defaults(DEFAULTS_PATH)
                    decomp = estimate_aux_from_coastdown(
                        A_N=A_edit, B_N_per_kph=B_edit, C_N_per_kph2=C_edit, mass_kg=M_edit,
                        category=sel.get("category",""),
                        electrification=sel.get("electrification","ICE"),
                        transmission_type=sel.get("transmission_type","AT"),
                        cdA_override_m2=sel.get("cda_m2"),
                        defaults_df=defaults_df,
                    )
                    update_vde(vde_id_edit, {k: float(v) for k, v in {
                        "rr_alpha_N": decomp.get("rr_alpha_N"),
                        "rr_beta_Npkph": decomp.get("rr_beta_Npkph"),
                        "aero_C_coef_Npkph2": decomp.get("aero_C_coef_Npkph2"),
                        "parasitic_A_coef_N": decomp.get("parasitic_A_coef_N"),
                        "parasitic_B_coef_Npkph": decomp.get("parasitic_B_Npkph"),
                        "parasitic_C_coef_Npkph2": decomp.get("parasitic_C_coef_Npkph2"),
                    }.items() if v is not None})
                except Exception:
                    pass

                # 3) Recompute VDE on a default cycle for row's legislation + show feedback
                leg_row = sel.get("legislation", "EPA")
                try:
                    df_cycle = default_cycle_for_legislation(leg_row)
                except Exception:
                    df_cycle = None

                if isinstance(df_cycle, pd.DataFrame) and not df_cycle.empty:
                    total_mj_km = None
                    by_phase = {}
                    if "phase" in df_cycle.columns:
                        if leg_row == "EPA":
                            res = epa_city_hwy_from_phase(df_cycle, A_edit, B_edit, C_edit, M_edit) or {}
                            if res.get("city_MJ_km") is not None: by_phase["city"] = float(res["city_MJ_km"])
                            if res.get("hwy_MJ_km")  is not None: by_phase["hwy"]  = float(res["hwy_MJ_km"])
                            if res.get("net_comb_MJ_km") is not None:
                                total_mj_km = float(res["net_comb_MJ_km"])
                            elif "city" in by_phase and "hwy" in by_phase:
                                total_mj_km = 0.55*by_phase["city"] + 0.45*by_phase["hwy"]
                        else:
                            res = wltp_phases_from_phase(df_cycle, A_edit, B_edit, C_edit, M_edit) or {}
                            for key_in, key_out in [
                                ("vde_low_mj_per_km","low"),
                                ("vde_mid_mj_per_km","mid"),
                                ("vde_high_mj_per_km","high"),
                                ("vde_extra_high_mj_per_km","xhigh"),
                            ]:
                                if res.get(key_in) is not None:
                                    by_phase[key_out] = float(res[key_in])
                            if res.get("vde_net_mj_per_km") is not None:
                                total_mj_km = float(res["vde_net_mj_per_km"])

                    # fallback se precisar
                    if total_mj_km is None:
                        r_all = compute_vde_net_mj_per_km(df_cycle, A_edit, B_edit, C_edit, M_edit)
                        total_mj_km = float(r_all["MJ_km"]) if isinstance(r_all, dict) else float(r_all)

                    # feedback + persist
                    show_vde_feedback(total_mj_km, by_phase)

                    upd = {"vde_net_mj_per_km": total_mj_km}
                    if "phase" in df_cycle.columns:
                        if leg_row == "EPA":
                            res = epa_city_hwy_from_phase(df_cycle, A_edit, B_edit, C_edit, M_edit) or {}
                            if res.get("urb_MJ") is not None: upd["vde_urb_mj"] = float(res["urb_MJ"])
                            if res.get("hw_MJ")  is not None: upd["vde_hw_mj"]  = float(res["hw_MJ"])
                            if res.get("net_comb_MJ_km") is not None:
                                upd["vde_net_mj_per_km"] = float(res["net_comb_MJ_km"])
                        else:
                            res = wltp_phases_from_phase(df_cycle, A_edit, B_edit, C_edit, M_edit) or {}
                            upd.update({k: float(v) for k, v in res.items() if v is not None})
                    update_vde(vde_id_edit, upd)

                else:
                    st.warning("Row updated, but default cycle could not be loaded; phase VDE not recomputed.")

                st.success("Row updated.")
                reset_ctx(preserve_meta=True)
                st.rerun()


            except Exception as e:
                st.error(f"Failed to update: {e}")

    # --- Delete block ---
    with st.expander("🗑️ Delete this VDE row"):
        st.warning("This action is irreversible. Linked fuelcons_db rows will be deleted (ON DELETE CASCADE).")
        confirm_text = st.text_input("Type DELETE to confirm:")
        delete_disabled = (confirm_text != "DELETE")
        if st.button(f"Delete VDE id={vde_id_edit}", type="secondary", disabled=delete_disabled):
            try:
                delete_row("vde_db", vde_id_edit)
                st.success(f"VDE id={vde_id_edit} deleted.")
                reset_ctx(preserve_meta=True)
                st.rerun()

            except Exception as e:
                st.error(f"Failed to delete: {e}")


# -----------------------------
# MAIN
# -----------------------------

def main():
    # --- page setup ---
    st.set_page_config(page_title="EcoDrive — VDE", layout="wide")
    ensure_db()
    
    init_state()
    
    ctx = st.session_state.ctx
    print(ctx)
    # ============ HEADER ============ #
    h1, i1, i2, i3 = st.columns([1.0, 0.12, 0.12, 0.12])
    with h1:
        st.title("EcoDrive Analyst · VDE")
        st.caption("Quick setup · clean preview · save/edit snapshots")
    st.divider()

    # ============ SIDEBAR: meta & modo ============
    vehicle_basics_sidebar()  # aqui o ctx é atualizado (make/legislation)
    
    # --- ícones automáticos (sem inputs) ---
    logo_path = search_logo(ctx, base_dir="data/images/logos", fallback="_unknown.png") or ""
    leg_icon  = get_legislation_icon(ctx, base_dir="data/images") or ""

    # atribui no ctx (use "=" e não "==")
    ctx["brand_icon"] = logo_path
    ctx["leg_icon"]   = leg_icon

    # mostra nas colunas do header
    show_if_exists(i1, ctx["brand_icon"], width=50, caption=ctx["make"])
    show_if_exists(i2, ctx["leg_icon"],   width=50, caption=ctx["legislation"])

   # ============ BODY ============ #
    # 1) bloco principal por modo (enxuto)
    
    if ctx["mode"] == "From baseline (editable)":

        baseline_picker_and_editor()   # seu picker (inclui Deltas/Change Parameters)
        

    elif ctx["mode"] == "Define all parameters (no baseline)":
        with st.expander("Road load & Mass", expanded=True):
            rr_section(prefill=None)
        with st.expander("Aerodynamics", expanded=False):
            aero_section(prefill=None)
        with st.expander("Parasitic & Brake", expanded=False):
            parasitic_brake_section(prefill=None)

    else:  # "From test (direct coastdown)"
        from_test_section()
        # 2) auxiliares onde já estavam
        auxiliaries_section()

    # 3) ciclo (padrão ou CSV) e preview ao vivo
    cycle_section()

    show_live_vde_preview()   # mantém sua função existente de preview

    # 4) salvar/editar
    compute_and_save()
    edit_or_delete()




#------------------------------
### BACKUP/ OLD CODE ###
#------------------------------

def main2():
    
    st.set_page_config(page_title="Mock Data / Editor", layout="wide")
    ensure_db()
    init_state()
    st.title("🧪 Mock Data / Full-Edit")

    vehicle_basics()
    mode_selector()
    tires_df = load_tire_catalog(TIRE_CSV)

    if ctx["mode"] == "From baseline (editable)":
        baseline_picker_and_editor()


    elif ctx["mode"] == "Define all parameters (no baseline)":
        rr_section(prefill=None, tires_df=tires_df)
        aero_section(prefill=None)
        parasitic_brake_section(prefill=None)


    else:  # "From test (direct coastdown)"
        from_test_section()
        auxiliaries_section()
        
    print(ctx)
    # ⬇️ traga o ciclo ANTES do preview
    cycle_section()
    # ⬇️ e só depois mostre o preview (para todas as modalidades)
    show_live_vde_preview()

    compute_and_save()
    edit_or_delete()

    st.markdown("---")



def pwt_section(prefill=None):
    st.subheader("PWT (minimal)")
    eff0 = to_float(prefill.get("driveline_eff"), ctx["driveline_eff"]) if prefill else ctx["driveline_eff"]
    ctx["driveline_eff"] = st.number_input("Driveline efficiency (0–1)", value=float(eff0), min_value=0.0, max_value=1.0, step=0.01)

def vehicle_basics():
    st.subheader("Vehicle basics")
    c1, c2, c3, c4 = st.columns(4)
    # (1) Your point: it’s fine if you only changed the display text.
    # Use the same internal values, different labels via format if you wish.
    leg_opts = ["WLTP", "EPA", "ABNT (Brazil)"]  # display labels
    # keep ctx value consistent with an item from leg_opts
    if ctx["legislation"] not in leg_opts:
        ctx["legislation"] = "WLTP"
    ctx["legislation"] = c1.selectbox("Legislation", leg_opts, index=leg_opts.index(ctx["legislation"]))

    # categorias oficiais
    epa_classes = [
         "Unknown","Two Seaters", "Minicompact Cars", "Subcompact Cars", "Compact Cars",
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
    ctx["category"]  = c2.selectbox("Category", category_list_upper, index=category_list_upper.index(ctx["category"]))

    # marcas sugeridas (mantidas p/ consistência)
    default_makes = [ "Toyota", "Honda", "Nissan", "Mitsubishi", "Mazda", "Subaru","Hyundai", "Kia", "Volkswagen", "Audi", "BMW", "Mercedes-Benz", "Porsche", "Peugeot","Renault", "Citroën", "Fiat", "Alfa Romeo", "Volvo", "Jaguar", "Land Rover",
                       "Skoda", "Seat", "Opel", "Ford", "Chevrolet", "Dodge", "Chrysler", "Jeep", "Ram", "Cadillac","Buick", "GMC", "Lincoln", "Tesla", "Suzuki", "Mini", "Smart", "Lexus", "Infiniti", "Acura"]
    # juntar marcas do DB + sugeridas (sem duplicar) + opção Other
    ensure_db()
    makes_db = db_list_makes(ctx["legislation"], ctx["category"])
    # Aplica .upper() nas marcas sugeridas
    default_makes_upper = [m.upper() for m in default_makes]
    merged_makes = list(dict.fromkeys(makes_db + [m for m in default_makes_upper if m not in makes_db]))
    if "Other (type manually)" not in merged_makes:
        merged_makes.append("Other (type manually)")
    
    # Select make/brand from merged list, or allow manual entry
    make_choice = c3.selectbox(
        "Make/Brand",
        merged_makes,
        index=merged_makes.index(ctx["make"]) if ctx["make"] in merged_makes else 0
    )
    if make_choice == "Other (type manually)":
        make = c3.text_input("Enter custom brand", value=ctx["make"] if ctx["make"] not in merged_makes else "")
    else:
        make = make_choice
    ctx["make"] = make
    #ctx["make"]     = c3.text_input("Make", ctx["make"])
    ctx["model"]    = c4.text_input("Model/Desc.", ctx["model"])
    c5, c6, c7, c8 = st.columns([1,3,1,1])  
    ctx["year"]  = c5.number_input("Year", value=int(ctx["year"]), min_value=1900, max_value=2100, step=1)
    ctx["notes"] = c6.text_input("Proposal Description", ctx["notes"])

    # NEW: electrification & transmission (chaves no ctx)
    elec_opts = ["ICE", "HEV", "PHEV", "BEV"]
    trans_opts = ["AT", "AMT", "CVT", "MT", "OT"]
    ctx["electrification"]   = c7.selectbox("Electrification", elec_opts, index=elec_opts.index(ctx.get("electrification","ICE")))
    ctx["transmission_type"] = c8.selectbox("Transmission",   trans_opts, index=trans_opts.index(ctx.get("transmission_type","AT")))

if __name__ == "__main__":
    main()