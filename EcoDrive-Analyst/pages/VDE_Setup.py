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

# import your own helpers/db as in your project

from src.vde_core.db import ensure_db
from src.vde_app.plots import cycle_chart
from src.vde_core.cycles import default_cycle_for_legislation, load_cycle_csv, use_standard_cycle, cycle_summary
from src.vde_core.services import load_vde_defaults, estimate_aux_from_coastdown
from src.vde_core.utils import cycle_kpis
from src.vde_app.components.shared import vde_by_phase, search_logo, get_legislation_icon
from src.vde_app.components.vde_setup import (
    render_baseline_picker_and_editor_panel,
    render_vde_edit_delete_panel,
)
from src.vde_core.vde_setup_service import (
    to_float,
    validate_core,
    db_list_makes,
    build_live_vde_preview,
    build_compute_vde_from_ctx,
    build_vde_insert_row,
    build_vde_phase_update,
    insert_vde_snapshot,
    update_vde_snapshot,
)

st.set_page_config(page_title="Mock Data / Editor", layout="wide")
ensure_db()


# 1) Garanta que o catálogo está carregado (no topo do script da página)
ABS_DIR = Path(__file__).resolve().parent          # pasta do arquivo atual
default_path = ABS_DIR.parent / 'data' / 'standards' / 'vde_defaults_by_category_trans_elec.csv' 
DEFAULTS_PATH = Path(default_path)

tire_path = ABS_DIR.parent / 'data' / 'standards' / 'tiresize_fromcode_table.csv' 
TIRE_CSV =Path(tire_path)


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
    ctx = st.session_state.get("ctx", {})
    preview = build_live_vde_preview(ctx)
    if not preview.get("ok"):
        st.warning(preview.get("error", "Preview not available."))
        return

    total_mj_km = float(preview["total_mj_km"])
    phases = preview.get("phases", {})
    equiv = preview.get("equiv")

    st.info(f"Live preview — VDE_NET: **{total_mj_km:.4f} MJ/km**  ({total_mj_km*277.7778:.1f} Wh/km)")
    if equiv is not None:
        with st.expander("RoadLoad breakdown", expanded=False):
            st.dataframe(pd.DataFrame(equiv.component_table), use_container_width=True, hide_index=True)

    if phases:
        order = ["city", "hwy", "low", "mid", "high", "xhigh"]
        ordered = [k for k in order if k in phases] + [k for k in phases if k not in order]
        cols = st.columns(min(4, len(ordered)))
        for i, k in enumerate(ordered):
            cols[i % len(cols)].metric(k.upper(), f"{phases[k]:.4f} MJ/km")

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
        df_cycle = use_standard_cycle(ctx["legislation"])
        if df_cycle is None:
            st.warning(f"Default cycle for {ctx['legislation']} not found.")
            st.info("Upload a custom CSV cycle with columns [t, v].")
        else:
            ctx["cycle_df"] = df_cycle
            ctx["cycle_name"] = cycle_name
            st.success(f"Using default cycle: {cycle_name}.csv")


    if upload is not None:
        try:
            df_cycle = pd.read_csv(upload)
            if not {"t", "v"} <= set(df_cycle.columns):
                raise ValueError("Uploaded CSV must have columns: t, v (v in m/s)")
            df_cycle = df_cycle.copy()
            df_cycle["t"] = pd.to_numeric(df_cycle["t"], errors="raise")
            df_cycle["v"] = pd.to_numeric(df_cycle["v"], errors="raise")
            if "phase" in df_cycle.columns:
                df_cycle["phase"] = df_cycle["phase"].astype(str)
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
            calc = build_compute_vde_from_ctx(ctx)
            if not calc.get("ok"):
                raise ValueError(calc.get("error", "Compute not available."))

            equiv = calc["equiv"]
            A1, B1, C1, mass_kg1 = equiv.A, equiv.B, equiv.C, equiv.mass_kg
            total_mj_km = float(calc["total_mj_km"])
            by_phase = dict(calc.get("by_phase", {}))
            deltas = calc.get("deltas", {})
            d_rr = deltas.get("delta_rr_N", 0.0)
            d_br = deltas.get("delta_brake_N", 0.0)
            d_par = deltas.get("delta_parasitics_N", 0.0)
            d_cda = deltas.get("delta_aero_Npkph2", 0.0)
            d_mass = deltas.get("delta_mass_kg", 0.0)

            # --- feedback imediato ---
            st.info(f"VDE (NET): **{total_mj_km:.4f} MJ/km**  ({total_mj_km*277.7778:.1f} Wh/km)")
            with st.expander("RoadLoad breakdown", expanded=False):
                st.dataframe(pd.DataFrame(equiv.component_table), use_container_width=True, hide_index=True)
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
                    A_N=A1, B_N_per_kph=B1, C_N_per_kph2=C1, mass_kg=mass_kg1,
                    category=cat,
                    electrification=ctx.get("electrification","ICE"),
                    transmission_type=ctx.get("transmission_type","AT"),
                    cdA_override_m2=ctx.get("cda_m2"),
                    defaults_df=defaults_df,
                )
            except Exception:
                pass

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
                deltas={
                    "delta_rr_N": d_rr,
                    "delta_brake_N": d_br,
                    "delta_parasitics_N": d_par,
                    "delta_aero_Npkph2": d_cda,
                    "delta_mass_kg": d_mass,
                },
                decomp=decomp,
            )

            # --- INSERT ---
            vde_id = insert_vde_snapshot(row)
            st.session_state["vde_id"] = vde_id

            upd = build_vde_phase_update(df_cycle, leg, A=A1, B=B1, C=C1, mass_kg=mass_kg1)
            if upd:
                update_vde_snapshot(vde_id, upd)

            st.success(f"Saved VDE snapshot (id={vde_id}).")
            # limpa estado volátil e volta “zerado” mantendo meta
            reset_ctx(preserve_meta=True)
            st.rerun()

        except Exception as e:
            st.error(f"Failed to compute/save VDE: {e}")

# ====================================
# Edit / Delete (function)
# ====================================

# -----------------------------
# MAIN
# -----------------------------

def main():
    # --- page setup ---
    st.set_page_config(page_title="EcoDrive - VDE", layout="wide")
    ensure_db()
    
    init_state()
    
    ctx = st.session_state.ctx
    print(ctx)
    # ============ HEADER ============ #
    h1, i1, i2, i3 = st.columns([1.0, 0.12, 0.12, 0.12])
    with h1:
        st.title("EcoDrive Analyst - VDE")
        st.caption("Quick setup - clean preview - save/edit snapshots")
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

        render_baseline_picker_and_editor_panel(
            tire_csv=TIRE_CSV,
            rr_section=rr_section,
            aero_section=aero_section,
            parasitic_brake_section=parasitic_brake_section,
        )
        

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
    render_vde_edit_delete_panel(defaults_path=DEFAULTS_PATH, reset_ctx=reset_ctx)


if __name__ == "__main__":
    main()

