# pages/report_vde_simplificado.py
# -*- coding: utf-8 -*-
"""
Relatório VDE (Simplificado)
- Foco: funcional, robusto e mínimo.
- Lê vde_db, aplica filtros básicos, calcula/usa VDE (urb/hwy/comb),
  mostra 1 KPI, ranking e (se houver) scatter VDE vs Label.
"""

import os, sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ===== Config básica da página =====
st.set_page_config(page_title="VDE Report (Simplificado)", page_icon="📊", layout="wide")
st.title("📊 VDE Report (Simplificado)")
st.caption("Versão mínima para comparação objetiva de VDE. Evita cálculos e gráficos complexos.")

# ===== Tentativa de importar suas funções (sem travar a página) =====
try:
    from src.vde_core.services import (
        default_cycle_for_legislation, load_cycle_csv,
        epa_city_hwy_from_phase, wltp_phases_from_phase
    )
    HAS_FUNCS = True
except Exception:
    HAS_FUNCS = False

# ===== Carregar base =====
DB_PATH = st.session_state.get("ctx", {}).get("DB_PATH", "data/db/eco_drive.db")

def read_table(db_path: str, query: str) -> pd.DataFrame:
    if not os.path.exists(db_path):
        st.error(f"DB não encontrado: {db_path}")
        st.stop()
    con = sqlite3.connect(db_path, timeout=30, isolation_level=None)
    try:
        df = pd.read_sql_query(query, con)
    finally:
        con.close()
    return df

df = read_table(DB_PATH, "SELECT * FROM vde_db ORDER BY COALESCE(updated_at, created_at) DESC;")
if df.empty:
    st.warning("vde_db está vazio.")
    st.stop()

# Aliases simples p/ A/B/C
if "A" not in df and "coast_A_N" in df: df["A"] = df["coast_A_N"]
if "B" not in df and "coast_B_N_per_kph" in df: df["B"] = df["coast_B_N_per_kph"]
if "C" not in df and "coast_C_N_per_kph2" in df: df["C"] = df["coast_C_N_per_kph2"]

# Label amigável do veículo
def veh_label(r):
    parts = [str(r.get("make","")).strip(), str(r.get("model","")).strip()]
    y = r.get("year", None)
    if pd.notna(y): parts.append(str(int(y)))
    return " ".join([p for p in parts if p])

df["veh_label"] = df.apply(veh_label, axis=1)

# ===== Filtros enxutos =====
col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1])

with col1:
    leg_opts = sorted([x for x in df["legislation"].dropna().unique() if x])
    leg_sel = st.multiselect("Legislation", leg_opts, default=leg_opts or [])

with col2:
    eng_opts = sorted([x for x in df["engine_type"].dropna().unique() if x])
    eng_sel = st.multiselect("Powertrain", eng_opts, default=[])

with col3:
    make_opts = sorted([x for x in df["make"].dropna().unique() if x])
    make_sel = st.multiselect("Make", make_opts, default=[])

with col4:
    cycle_sel = st.selectbox("Cycle", ["combined", "urb", "hwy"], index=0)

mask = pd.Series(True, index=df.index)
if leg_sel: mask &= df["legislation"].isin(leg_sel)
if eng_sel: mask &= df["engine_type"].isin(eng_sel)
if make_sel: mask &= df["make"].isin(make_sel)
dfv = df[mask].copy()

if dfv.empty:
    st.info("Nenhum veículo após filtros.")
    st.stop()

# Baseline
baseline = st.selectbox("Baseline", dfv["veh_label"].tolist(), index=0)

# ===== Funções mínimas para obter VDE por ciclo =====
def extract_vde_cols(frame: pd.DataFrame):
    """Garante colunas de VDE 'VDE_urb_mj_per_km', 'VDE_hwy_mj_per_km', 'VDE_net_comb_mj_per_km' se existirem."""
    for c in ["VDE_urb_mj_per_km", "VDE_hwy_mj_per_km", "VDE_net_comb_mj_per_km"]:
        if c not in frame.columns:
            frame[c] = np.nan
    return frame

def calc_vde_if_possible(frame: pd.DataFrame, cycle_key: str) -> pd.DataFrame:
    """Tenta calcular VDE via suas funções. Se não der, mantém o que vier do DB e preenche combinado da EPA com 0.55/0.45."""
    out = frame.copy()
    extract_vde_cols(out)

    if not HAS_FUNCS:
        # Apenas tenta calcular combinado EPA por 0.55/0.45 se já houver urb/hwy
        mask_epa = out["legislation"].astype(str).str.upper().eq("EPA")
        need_comb = out["VDE_net_comb_mj_per_km"].isna() & mask_epa
        can = need_comb & out["VDE_urb_mj_per_km"].notna() & out["VDE_hwy_mj_per_km"].notna()
        out.loc[can, "VDE_net_comb_mj_per_km"] = (
            0.55 * out.loc[can, "VDE_urb_mj_per_km"] +
            0.45 * out.loc[can, "VDE_hwy_mj_per_km"]
        )
        return out

    # Com funções disponíveis: tenta por legislação
    for leg, g in out.groupby("legislation"):
        try:
            fname = default_cycle_for_legislation(str(leg))
            cyc = load_cycle_csv(fname)
            # renome mínimo para segurança
            if "v_kph" not in cyc.columns and "speed_kph" in cyc.columns:
                cyc = cyc.rename(columns={"speed_kph":"v_kph"})
            if "time_s" not in cyc.columns and "t_s" in cyc.columns:
                cyc = cyc.rename(columns={"t_s":"time_s"})
        except Exception:
            continue

        for idx, r in g.iterrows():
            A, B, C = r.get("A"), r.get("B"), r.get("C")
            m = r.get("mass_kg")
            if any(pd.isna(x) for x in [A, B, C, m]):
                continue
            try:
                if str(leg).upper() == "EPA":
                    res = epa_city_hwy_from_phase(cyc, A, B, C, m)
                    v_urb = res.get("urb_MJ_km") or res.get("urb_MJ_per_km")
                    v_hwy = res.get("hwy_MJ_km") or res.get("hwy_MJ_per_km")
                    v_comb = res.get("comb_MJ_km") or res.get("comb_MJ_per_km")
                    if pd.notna(v_urb): out.at[idx, "VDE_urb_mj_per_km"] = float(v_urb)
                    if pd.notna(v_hwy): out.at[idx, "VDE_hwy_mj_per_km"] = float(v_hwy)
                    if pd.notna(v_comb):
                        out.at[idx, "VDE_net_comb_mj_per_km"] = float(v_comb)
                else:
                    res = wltp_phases_from_phase(cyc, A, B, C, m)
                    v_urb = res.get("urb_MJ_km") or res.get("urb_MJ_per_km")
                    v_hwy = res.get("hwy_MJ_km") or res.get("hwy_MJ_per_km")
                    v_comb = res.get("comb_MJ_km") or res.get("comb_MJ_per_km") or res.get("VDE_NET")
                    if pd.notna(v_urb): out.at[idx, "VDE_urb_mj_per_km"] = float(v_urb)
                    if pd.notna(v_hwy): out.at[idx, "VDE_hwy_mj_per_km"] = float(v_hwy)
                    if pd.notna(v_comb):
                        out.at[idx, "VDE_net_comb_mj_per_km"] = float(v_comb)
            except Exception:
                continue

    # Se ainda faltar combinado EPA, usa 0.55/0.45
    mask_epa = out["legislation"].astype(str).str.upper().eq("EPA")
    need_comb = out["VDE_net_comb_mj_per_km"].isna() & mask_epa
    can = need_comb & out["VDE_urb_mj_per_km"].notna() & out["VDE_hwy_mj_per_km"].notna()
    out.loc[can, "VDE_net_comb_mj_per_km"] = (
        0.55 * out.loc[can, "VDE_urb_mj_per_km"] +
        0.45 * out.loc[can, "VDE_hwy_mj_per_km"]
    )
    return out

# Normaliza labels se existirem
def normalize_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    # cria padronizados vazios
    for dst, cands in {
        "label_urb_mj_per_km": ["label_urb_mj_per_km","label_city_mj_per_km","label_city_MJ_km"],
        "label_hwy_mj_per_km": ["label_hwy_mj_per_km","label_highway_mj_per_km","label_hwy_MJ_km"],
        "label_comb_mj_per_km": ["label_comb_mj_per_km","label_combined_mj_per_km","label_comb_MJ_km"],
    }.items():
        if dst not in out.columns:
            # procura candidatos
            found = None
            for c in cands:
                if c in out.columns:
                    found = c; break
            out[dst] = out[found] if found else np.nan
    return out

dfv = normalize_labels(calc_vde_if_possible(dfv, cycle_sel))

# Escolhe coluna de VDE / Label conforme ciclo
if cycle_sel == "urb":
    dfv["VDE_view_mj_per_km"] = dfv["VDE_urb_mj_per_km"]
    dfv["Label_view_mj_per_km"] = dfv["label_urb_mj_per_km"]
elif cycle_sel == "hwy":
    dfv["VDE_view_mj_per_km"] = dfv["VDE_hwy_mj_per_km"]
    dfv["Label_view_mj_per_km"] = dfv["label_hwy_mj_per_km"]
else:
    dfv["VDE_view_mj_per_km"] = dfv["VDE_net_comb_mj_per_km"]
    dfv["Label_view_mj_per_km"] = dfv["label_comb_mj_per_km"]

# ===== KPI simples =====
bl_row = dfv[dfv["veh_label"].eq(baseline)].head(1)
bl_vde = float(bl_row.iloc[0]["VDE_view_mj_per_km"]) if not bl_row.empty and pd.notna(bl_row.iloc[0]["VDE_view_mj_per_km"]) else np.nan
k1, k2 = st.columns(2)
with k1:
    st.metric(f"VDE_{cycle_sel.upper()} (Baseline)", f"{bl_vde:.3f} MJ/km" if pd.notna(bl_vde) else "n/d")
with k2:
    st.metric("Veículos no subset", f"{len(dfv)}")

st.markdown("---")

# ===== Gráfico 1 – Ranking (barras) =====
rank = dfv[["veh_label","engine_type","VDE_view_mj_per_km"]].dropna().copy()
if rank.empty:
    st.info("Sem dados de VDE suficientes para ranking.")
else:
    rank = rank.sort_values("VDE_view_mj_per_km", ascending=True)
    fig_rank = px.bar(rank, x="veh_label", y="VDE_view_mj_per_km", color="engine_type",
                      title=f"Ranking – VDE_{cycle_sel.upper()} (MJ/km)")
    fig_rank.update_layout(xaxis_title="", yaxis_title="MJ/km", showlegend=True)
    st.plotly_chart(fig_rank, use_container_width=True)

# ===== Gráfico 2 – Scatter VDE vs Label (opcional) =====
sc = dfv[["veh_label","engine_type","VDE_view_mj_per_km","Label_view_mj_per_km"]].dropna().copy()
if sc.shape[0] >= 3 and sc["Label_view_mj_per_km"].notna().sum() >= 3:
    fig_sc = px.scatter(sc, x="VDE_view_mj_per_km", y="Label_view_mj_per_km",
                        color="engine_type", hover_name="veh_label",
                        title=f"VDE_{cycle_sel.upper()} vs Label (MJ/km)")
    fig_sc.update_layout(xaxis_title="VDE (MJ/km)", yaxis_title="Label (MJ/km)")
    st.plotly_chart(fig_sc, use_container_width=True)
else:
    st.caption("Labels oficiais não disponíveis/suficientes para o scatter (opcional).")

st.markdown("---")

# ===== Tabela + Export =====
show_cols = [
    "veh_label","legislation","engine_type","year",
    "A","B","C","mass_kg",
    "VDE_urb_mj_per_km","VDE_hwy_mj_per_km","VDE_net_comb_mj_per_km",
    "label_urb_mj_per_km","label_hwy_mj_per_km","label_comb_mj_per_km"
]
show_cols = [c for c in show_cols if c in dfv.columns]
st.subheader("Dados (simplificado)")
st.dataframe(dfv[show_cols], use_container_width=True, hide_index=True)

st.download_button(
    "Export CSV",
    data=dfv[show_cols].to_csv(index=False).encode("utf-8"),
    file_name="vde_report_simplificado.csv",
    mime="text/csv"
)

st.caption("Obs.: Se as funções EPA/WLTP não estiverem disponíveis, o app usa apenas os VDEs já salvos no DB e calcula o combinado EPA como 0.55*urb + 0.45*hwy quando possível.")
