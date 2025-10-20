import streamlit as st
from .state import ensure_defaults
from src.vde_core.services import  compute_vde_net_mj_per_km
import pandas as pd

# --- DB / services (use os caminhos reais do seu projeto) ---
from src.vde_core.db import fetchall, update_row, delete_row

from src.vde_app.derivatives import load_fuelcons_allowed



from typing import Dict, Any, Tuple, Optional
from src.vde_core.db import fetchall
def sidebar_inputs():
    ensure_defaults(st.session_state)
    st.sidebar.header("Parameters (sidebar)")
    # Return values to be used in pages
    return st.session_state["roadload_params"]


def pressure_input_with_units(key_prefix=""):
    unit = st.radio("Unit", ["kPa","psi"], key=f"{key_prefix}press_unit", horizontal=True)
    base_kpa = float(st.session_state.get(f"{key_prefix}press_kpa", 230.0))
    default_display = base_kpa if unit=="kPa" else base_kpa/6.89475729
    val = st.number_input(f"Pressure [{unit}]", 0.0, 500.0 if unit=="kPa" else 100.0, default_display, step=1.0 if unit=="kPa" else 0.5, key=f"{key_prefix}press_val")
    kpa = val if unit=="kPa" else val*6.89475729
    st.session_state[f"{key_prefix}press_kpa"] = kpa
    st.caption(f"{kpa:.1f} kPa ≈ {kpa/6.89475729:.1f} psi")
    return kpa


# ---------- helpers (place near your other small utils) ----------
def vde_by_phase(df_cycle, leg, A, B, C, mass_kg):
    """Return dict {phase_label: MJ/km} if 'phase' column exists."""
    out = {}
    if not isinstance(df_cycle, pd.DataFrame) or "phase" not in df_cycle.columns:
        return out

    def _norm(p):
        p = str(p).strip().lower()
        if leg == "EPA":
            if "city" in p or "ftp" in p: return "city"
            if "hwy" in p or "hwfet" in p or "highway" in p: return "hwy"
        else:  # WLTP
            if "low" in p: return "low"
            if "mid" in p or "medium" in p: return "mid"
            if "high" in p and "extra" not in p: return "high"
            if "xhigh" in p or "extra" in p: return "xhigh"
        return p

    for ph in df_cycle["phase"].unique():
        sub = df_cycle[df_cycle["phase"] == ph]
        r = compute_vde_net_mj_per_km(sub, A, B, C, mass_kg)
        out[_norm(ph)] = float(r["MJ_km"]) if isinstance(r, dict) else float(r)
    return out


def show_vde_feedback(overall_mj_km, by_phase):
    """Render total and per-phase numbers immediately."""
    st.success(f"VDE_NET (cycle total) ≈ {overall_mj_km:.4f} MJ/km")
    if by_phase:
        pref = ["city","hwy","low","mid","high","xhigh"]
        ordered = [k for k in pref if k in by_phase] + [k for k in by_phase if k not in pref]
        cols = st.columns(min(4, len(ordered)))
        for i, k in enumerate(ordered):
            cols[i % len(cols)].metric(k.upper(), f"{by_phase[k]:.4f} MJ/km")


from pathlib import Path
import unicodedata, re, difflib

def search_logo(ctx: dict, base_dir: str = "data/logos", fallback: str | None = None) -> str | None:
    """
    Retorna o caminho (str) do logo .png dentro de base_dir para o fabricante em ctx["make"].
    Convenção: nome em minúsculas, espaços -> '-', extensão .png (ex.: 'Land Rover' -> 'land-rover.png').

    - Remove acentos e símbolos, normaliza para slug.
    - Tenta arquivo direto, alguns sinônimos comuns e um match aproximado (difflib).
    - Se não achar e 'fallback' for fornecido (ex.: '_unknown.png'), retorna o fallback (se existir).
    - Caso contrário, retorna None.
    """
    make_raw = str((ctx or {}).get("make", "")).strip()
    if not make_raw:
        return None

    def _slugify(s: str) -> str:
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        s = s.lower().replace("&", " and ")
        s = re.sub(r"[^a-z0-9]+", "-", s)      # tudo que não é [a-z0-9] vira '-'
        s = re.sub(r"-{2,}", "-", s).strip("-") # colapsa múltiplos '-' e tira das pontas
        return s

    slug = _slugify(make_raw)
    base = Path(base_dir)

    # 1) tentativa direta
    direct = base / f"{slug}.png"
    if direct.exists():
        return str(direct)

    # 2) sinônimos/variações comuns
    synonyms = {
        "mercedes": "mercedes-benz",
        "landrover": "land-rover",
        "vw": "volkswagen",
        "chevy": "chevrolet",
        "byd-auto": "byd",
        "bayerische-motoren-werke": "bmw",
        "citroen": "citroen",  # garante o caso com/sem acento
    }
    alt = synonyms.get(slug)
    if alt:
        alt_p = base / f"{alt}.png"
        if alt_p.exists():
            return str(alt_p)

    # 3) normalização dos arquivos existentes + heurísticas
    pngs = list(base.glob("*.png"))
    if pngs:
        norm_map = {}
        for fp in pngs:
            norm = _slugify(fp.stem)
            norm_map[norm] = fp
            if norm == slug:
                return str(fp)

        # começa com / contém
        for norm, fp in norm_map.items():
            if norm.startswith(slug) or slug in norm:
                return str(fp)

        # aproximação
        hit = difflib.get_close_matches(slug, list(norm_map.keys()), n=1, cutoff=0.84)
        if hit:
            return str(norm_map[hit[0]])

    # 4) fallback opcional
    if fallback:
        fb = Path(fallback)
        if not fb.is_absolute():
            fb = base / fb
        if fb.exists():
            return str(fb)

    return None


from pathlib import Path

def get_legislation_icon(ctx: dict, base_dir: str = "data/icons") -> str | None:
    """
    Retorna o caminho do ícone (bandeira) conforme ctx["legislation"].
    Mapeamento esperado:
      EPA/USA/US       -> flag_usa.png
      WLTP/EU/UNECE    -> flag_eu.png
      BRA/BR/PROCONVE  -> flag_brazil.png
    """
    leg = str((ctx or {}).get("legislation", "")).strip().lower().replace(" ", "")

    mapping = {
        "epa": "flag_usa.png",
        "usa": "flag_usa.png",
        "us":  "flag_usa.png",

        "wltp":  "flag_eu.png",
        "eu":    "flag_eu.png",
        "unece": "flag_eu.png",

        "bra":      "flag_brazil.png",
        "br":       "flag_brazil.png",
        "proconve": "flag_brazil.png",
        "pbev":     "flag_brazil.png",
        "mover":    "flag_brazil.png",
    }

    # match direto ou por “contains”
    fname = mapping.get(leg)
    if not fname:
        for k, v in mapping.items():
            if k in leg:
                fname = v
                break

    if not fname:
        return None

    p = Path(base_dir) / fname
    return str(p) if p.exists() else None



def filters_bar(vde_id: int, electrification: str, key_ns: str = "fb") -> Dict[str, Any]:
    k = lambda name: f"{key_ns}_{name}"  # helper p/ keys únicas
    st.markdown("### Filters")
    c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1.1, 1.1, 1])

    # options via DISTINCT
    cats = [r["category"] for r in fetchall("SELECT DISTINCT category FROM vde_db WHERE category IS NOT NULL AND category<>'' ORDER BY category;")] or []
    makes = [r["make"] for r in fetchall("SELECT DISTINCT make FROM vde_db WHERE make IS NOT NULL AND make<>'' ORDER BY make;")] or []
    elecs = [r["electrification"] for r in fetchall("SELECT DISTINCT electrification FROM fuelcons_db WHERE electrification IS NOT NULL AND electrification<>'' ORDER BY electrification;")] or ["ICE","MHEV","HEV","PHEV","BEV"]

    with c1:
        view_scope = st.selectbox("View", ["Only this Vehicle id", "All"],
                                  index=1, key=k("fl_scope"))
    with c2:
        elec_choice = st.selectbox(
            "Electrification",
            ["(all)", f"(current: {electrification})"] + [e for e in elecs if e != electrification],
            key=k("fl_elec"),
        )
    with c3:
        cat_choice = st.selectbox("Category", ["(all)"] + cats, key=k("fl_cat"))
    with c4:
        make_choice = st.selectbox("Make", ["(all)"] + makes, key=k("fl_make"))
    with c5:
        p_choice = st.selectbox("Power (hp)", ["(all)", "≤160", "161–270", "271–470", "471–670", ">670"],
                                key=k("fl_pbin"))

    filters: Dict[str, Any] = {}
    if view_scope == "Only this Vehicle id":
        filters["vde_id"] = vde_id

    if elec_choice not in ("(all)", f"(current: {electrification})"):
        filters["electrification"] = elec_choice
    elif elec_choice.startswith("(current:"):
        filters["electrification"] = electrification

    if cat_choice != "(all)":
        filters["category"] = cat_choice
    if make_choice != "(all)":
        filters["make"] = make_choice
    

    # power bin em hp → converte p/ kW na query
    hp_to_kw = lambda hp: float(hp) / 1.34102209
    pmap = {
        "≤150 HP": (None, 150), "151–300 HP": (151, 300),
        "301–500 HP": (301, 500), "501–700 HP": (501, 700), ">700 HP": (701, None)
    }

    if p_choice in pmap:
        lo_hp, hi_hp = pmap[p_choice]
        lo_kw = hp_to_kw(lo_hp) if lo_hp is not None else None
        hi_kw = hp_to_kw(hi_hp) if hi_hp is not None else None
        filters["power_kw_range"] = (lo_kw, hi_kw)

    return filters


# =============================================================================
# Tabela fuelcons — ver por VDE ou ver TODOS
# =============================================================================


def fetch_fuelcons_by_vde(vde_id: int) -> pd.DataFrame:
    q = (
        "SELECT id, created_at, method_note, "
        "fuel_l_per_100km, fuel_km_per_l, energy_Wh_per_km, gco2_per_km, "
        "gear_count, final_drive_ratio "
        "FROM fuelcons_db WHERE vde_id=? ORDER BY created_at DESC"
    )
    rows = fetchall(q, (vde_id,))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_fuelcons_all(filters: Dict[str, Any]) -> pd.DataFrame:
    """Mostra todas as linhas; permite filtros rápidos por category/make/electrification via JOIN com vde_db."""
    base = (
        "SELECT f.id, f.created_at, f.vde_id, f.electrification, f.method_note, "
        "f.fuel_l_per_100km, f.engine_max_power_kw, f.fuel_km_per_l, f.energy_Wh_per_km, f.gco2_per_km, f.gear_count, f.final_drive_ratio, "
        "v.make, v.model, v.year, v.category, v.legislation, v.vde_net_mj_per_km "
        "FROM fuelcons_db f JOIN vde_db v ON v.id = f.vde_id WHERE 1=1"
    )
    params = []
    if filters.get("electrification"):
        base += " AND f.electrification = ?"
        params.append(filters["electrification"])
    if filters.get("category"):
        base += " AND v.category = ?"
        params.append(filters["category"])
    if filters.get("make"):
        base += " AND v.make = ?"
        params.append(filters["make"])
    if filters.get("power_kw_range"):
        lo, hi = filters["power_kw_range"]
        if lo is not None:
            base += " AND f.engine_max_power_kw >= ?"
            params.append(lo)
        if hi is not None:
            base += " AND f.engine_max_power_kw < ?"
            params.append(hi)
    base += " ORDER BY f.created_at DESC"
    rows = fetchall(base, tuple(params)) if params else fetchall(base)
    return pd.DataFrame(rows) if rows else pd.DataFrame()



def render_fuelcons_table(df: pd.DataFrame, editable: bool = False) -> None:
    if df is None or df.empty:
        st.info("No scenarios.")
        return

    # colunas que mostramos na listagem
    show_cols = [c for c in [
        "id", "vde_id", "electrification", "fuel_l_per_100km", "energy_Wh_per_km",
        "fuel_ftp75_l_per_100km", "fuel_hwfet_l_per_100km",
        "energy_ftp75_Wh_per_km", "energy_hwfet_Wh_per_km",
        "method_note", "created_at"
    ] if c in df.columns]

    st.dataframe(df[show_cols].sort_values("id", ascending=False), use_container_width=True)

    if not editable:
        return

    st.markdown("#### Edit / Delete")
    # um cartão por linha, com botões e um mini-formulário de edição
    for _, row in df.sort_values("id", ascending=False).iterrows():
        rid = int(row["id"])
        with st.expander(f"#{rid} · {row.get('electrification','?')} · y={row.get('fuel_l_per_100km') or row.get('energy_Wh_per_km')}", expanded=False):
            c1, c2, c3, c4, c5, c6 = st.columns([1.1, 1.1, 1.1, 1.1, 2, 1.2])

            # campos principais (só os que existirem no DF)
            elec = c1.selectbox(
                "Electrification", ["ICE","MHEV","HEV","PHEV","BEV"],
                index= ["ICE","MHEV","HEV","PHEV","BEV"].index(str(row.get("electrification","ICE"))),
                key=f"fc_elec_{rid}"
            )
            f_comb = c2.number_input("fuel L/100km", value=float(row["fuel_l_per_100km"]) if pd.notna(row.get("fuel_l_per_100km")) else 0.0,
                                     step=0.01, format="%.2f", key=f"fc_fcomb_{rid}")
            e_comb = c3.number_input("energy Wh/km", value=float(row["energy_Wh_per_km"]) if pd.notna(row.get("energy_Wh_per_km")) else 0.0,
                                     step=1.0, format="%.0f", key=f"fc_ecomb_{rid}")
            f_ftp  = c4.number_input("FTP-75 L/100", value=float(row["fuel_ftp75_l_per_100km"]) if pd.notna(row.get("fuel_ftp75_l_per_100km")) else 0.0,
                                     step=0.01, format="%.2f", key=f"fc_fftp_{rid}")
            e_ftp  = c5.number_input("FTP-75 Wh/km", value=float(row["energy_ftp75_Wh_per_km"]) if pd.notna(row.get("energy_ftp75_Wh_per_km")) else 0.0,
                                     step=1.0, format="%.0f", key=f"fc_eftp_{rid}")
            note   = c6.text_input("Note", value=str(row.get("method_note") or ""), key=f"fc_note_{rid}")

            c7, c8, c9 = st.columns([1, 1, 6])
            f_hwy = c7.number_input("HWFET L/100", value=float(row["fuel_hwfet_l_per_100km"]) if pd.notna(row.get("fuel_hwfet_l_per_100km")) else 0.0,
                                    step=0.01, format="%.2f", key=f"fc_fhwy_{rid}")
            e_hwy = c8.number_input("HWFET Wh/km", value=float(row["energy_hwfet_Wh_per_km"]) if pd.notna(row.get("energy_hwfet_Wh_per_km")) else 0.0,
                                    step=1.0, format="%.0f", key=f"fc_ehwy_{rid}")
            st.caption("Preencha apenas os campos que deseja alterar; vazios não sobrescrevem.")

            # ações
            a1, a2, a3 = st.columns([1.1, 1.1, 6])
            if a1.button("✏️ Save", key=f"fc_save_{rid}"):
                # monta payload apenas com valores informados (>0 ou não vazios)
                payload = {}
                if elec: payload["electrification"] = elec
                if f_comb and f_comb > 0: payload["fuel_l_per_100km"] = float(f_comb)
                if e_comb and e_comb > 0: payload["energy_Wh_per_km"] = float(e_comb)
                if f_ftp  and f_ftp  > 0: payload["fuel_ftp75_l_per_100km"] = float(f_ftp)
                if e_ftp  and e_ftp  > 0: payload["energy_ftp75_Wh_per_km"] = float(e_ftp)
                if f_hwy  and f_hwy  > 0: payload["fuel_hwfet_l_per_100km"] = float(f_hwy)
                if e_hwy  and e_hwy  > 0: payload["energy_hwfet_Wh_per_km"] = float(e_hwy)
                if note is not None:      payload["method_note"] = note

                # filtra pelas colunas existentes (dinâmico)
                FUELCONS_ALLOWED = load_fuelcons_allowed()
                allowed = set(FUELCONS_ALLOWED)
                payload = {k: v for k, v in payload.items() if k in allowed}

                if payload:
                    try:
                        update_row("fuelcons_db", rid, payload)
                        st.success("Saved.")
                    except Exception as e:
                        st.error(f"Update failed: {e}")
                else:
                    st.info("Nada para salvar.")

            # delete com confirmação simples
            confirm_key = f"fc_confirm_{rid}"
            if a2.button("🗑️ Delete", key=f"fc_del_{rid}"):
                st.session_state[confirm_key] = True

            if st.session_state.get(confirm_key):
                b1, b2 = st.columns([1, 6])
                b1.warning("Confirmar exclusão?")
                if b1.button("Confirm", key=f"fc_del_ok_{rid}"):
                    try:
                        delete_row("fuelcons_db", rid)
                        st.success("Deleted.")
                        st.session_state.pop(confirm_key, None)
                    except Exception as e:
                        st.error(f"Delete failed: {e}")
                if b2.button("Cancelar", key=f"fc_del_cancel_{rid}"):
                    st.session_state.pop(confirm_key, None)

