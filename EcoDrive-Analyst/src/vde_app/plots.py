import pandas as pd
import plotly.express as px
from src.vde_core.db import fetchall, fetchone, update_vde, insert_fuelcons, ensure_db
import streamlit as st
from typing import Dict, Any, List
def line_power(df: pd.DataFrame):
    fig = px.line(df, x="t", y="P", title="Instantaneous Power")
    return fig

def cycle_chart(df: pd.DataFrame):
    """df deve ter colunas: t (s) e v (m/s)"""
    if not {"t", "v"} <= set(df.columns):
        return None
    dfx = df.copy().dropna(subset=["t", "v"])
    dfx["v_kmh"] =3.6* dfx["v"] # csv ciclos estão em km/h
    fig = px.line(
        dfx, x="t", y="v_kmh",
        labels={"t": "Time [s]", "v_kmh": "Speed [km/h]"},  # <-- labels (minúsculo)
        title="Drive cycle speed profile"
    )
    fig.update_layout(margin=dict(l=0, r=0, t=35, b=0), height=280)
    return fig



def build_scatter_data() -> pd.DataFrame:
    q = """
    SELECT 
        f.vde_id,
        f.fuel_l_per_100km,
        f.fuel_km_per_l,
        f.energy_Wh_per_km,
        f.gco2_per_km,
        f.method_note,
        f.engine_max_power_kw,
        v.vde_net_mj_per_km,
        v.engine_size_l,
        v.transmission_type,
        v.drive_type,
        v.category,
        v.make,
        v.model,
        v.year
    FROM fuelcons_db f
    JOIN vde_db v ON v.id = f.vde_id
    """
    rows = fetchall(q)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "transmission_type" in df.columns:
        df.rename(columns={"transmission_type": "trans_topology"}, inplace=True)
    if "category" in df.columns:
        df.rename(columns={"category": "vehicle_class"}, inplace=True)

    def _bin_power_kw(p):
        # 5 faixas: ≤120, 121–150, 151–200, 201–250, >250
        if p is None or pd.isna(p):
            return None
        p = float(p)
        if p <= 150*0.7457:  return "≤150 HP"
        if p <= 300*0.7457:  return "151–300 HP"
        if p <= 500*0.7457:  return "301–500 HP"
        if p <= 700*0.7457:  return "501–700 HP"
        return ">700 HP"


    def _bin_engine_size_l(s):
        # fallback caso não haja potência
        if s is None or pd.isna(s):
            return None
        s = float(s)
        if s <= 1.0:  return "≤1.0L"
        if s <= 1.6:  return "1.1–1.6L"
        if s <= 2.5:  return "1.7–2.5L"
        if s <= 3.3:  return "2.6–3.3L"
        return ">3.3L"

    def _map_color_group(r):
        # BEV: mantém por vehicle_class/drive_type (como antes)
        if pd.notna(r.get("energy_Wh_per_km")) and pd.isna(r.get("fuel_l_per_100km")):
            return r.get("vehicle_class") or r.get("drive_type") or "BEV"

        # ICE/MxHEV: usa potência se houver, senão engine size, senão trans_topology
        g = _bin_power_kw(r.get("engine_max_power_kw"))
        if g: 
            return g
        g = _bin_engine_size_l(r.get("engine_size_l"))
        if g:
            return g
        return "Unknown"


    df["color_group"] = df.apply(_map_color_group, axis=1)
    df["is_bev"] = df["energy_Wh_per_km"].notna() & df["fuel_l_per_100km"].isna()
    df = df[df["vde_net_mj_per_km"].notna()]
    return df


def show_summary_plots(df: pd.DataFrame):
    st.subheader("Summary — Energy/Consumption vs VDE_NET")
    if df.empty:
        st.info("Sem dados para plot.")
        return
    try:
        import plotly.express as px
        df_bev = df[(df["is_bev"] == True) & df["energy_Wh_per_km"].notna()]
        df_ice = df[(df["is_bev"] == False) & df["fuel_l_per_100km"].notna()]
        if not df_bev.empty:
            st.markdown("**BEV: Energy [Wh/km] vs VDE_NET [MJ/km]**")
            fig1 = px.scatter(
                df_bev, x="vde_net_mj_per_km", y="energy_Wh_per_km", color="color_group",
                hover_data=["make","model","year","vehicle_class"]
            )
            fig1.update_layout(margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("Sem dados BEV válidos.")

        if not df_ice.empty:
            st.markdown("**ICE/MxHEV: Fuel [L/100km] vs VDE_NET [MJ/km]**")
            fig2 = px.scatter(
                df_ice, x="vde_net_mj_per_km", y="fuel_l_per_100km", color="color_group",
                hover_data=["make","model","year","engine_size_l","trans_topology"]
            )
            fig2.update_layout(margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Sem dados ICE/Híbridos válidos.")

    except Exception as e:
        st.warning(f"Plotly indisponível ({e}).")

def build_scatter_from_fuel(df_fuel: pd.DataFrame) -> pd.DataFrame:
    """Monta o DF do scatter a partir do df_fuel que já está filtrado, juntando colunas do VDE quando necessário."""
    if df_fuel is None or df_fuel.empty:
        return pd.DataFrame()
    # Se sua fetch_fuelcons_all já devolve make/model/year/category/vde_net_mj_per_km, ótimo.
    # Caso contrário, faça um join simples:
    need_cols = {"vde_net_mj_per_km","engine_size_l","transmission_type","drive_type","category","make","model","year"}
    if not need_cols.issubset(df_fuel.columns):
        vde_ids = tuple(set(df_fuel["vde_id"].tolist()))
        if len(vde_ids) == 1:
            rows = fetchall("SELECT * FROM vde_db WHERE id=?;", (vde_ids[0],))
        else:
            qmarks = ",".join("?" for _ in vde_ids)
            rows = fetchall(f"SELECT * FROM vde_db WHERE id IN ({qmarks});", vde_ids)
        dv = pd.DataFrame(rows) if rows else pd.DataFrame()
        df = df_fuel.merge(dv, left_on="vde_id", right_on="id", how="left", suffixes=("", "_v"))
    else:
        df = df_fuel.copy()

    # mapeia coloração
    if "transmission_type" in df.columns:
        df.rename(columns={"transmission_type": "trans_topology"}, inplace=True)
    if "category" in df.columns:
        df.rename(columns={"category": "vehicle_class"}, inplace=True)

    def _map_color_group(r):
        # BEV: usa vehicle_class/drive_type
        if pd.notna(r.get("energy_Wh_per_km")) and pd.isna(r.get("fuel_l_per_100km")):
            return r.get("vehicle_class") or r.get("drive_type") or "BEV"
        # ICE/Híbridos: bins por potência hp se houver, senão engine size
        pkw = r.get("engine_max_power_kw")
        if pd.notna(pkw):
            hp = float(pkw) * 1.34102209
            if hp <= 150:  return "≤150 HP"
            if hp <= 300:  return "151–300 HP"
            if hp <= 500:  return "301–500 HP"
            if hp <= 700:  return "501–700 HP"
            return ">700 HP"
        size = r.get("engine_size_l")
        if pd.notna(size):
            s = float(size)
            if s <= 1.0:  return "≤1.0L"
            if s <= 1.7:  return "1.1–1.7L"
            if s <= 2.5:  return "1.8–2.5L"
            if s <= 3.4:  return "2.6–3.4L"
            return ">3.5L"
        return r.get("trans_topology") or "Unknown"

    df["color_group"] = df.apply(_map_color_group, axis=1)
    df["is_bev"] = df["energy_Wh_per_km"].notna() & df["fuel_l_per_100km"].isna()
    df = df[df["vde_net_mj_per_km"].notna()]
    # >>> ADD: flag do VDE atual + color group destacado
    cur_id = st.session_state.get("current_vde_id", None)
    if cur_id is not None and "vde_id" in df.columns:
        df["is_current"] = (df["vde_id"].astype("Int64") == int(cur_id))
        # sobrescreve o grupo de cor para destacar no plot
        df.loc[df["is_current"] == True, "color_group"] = "★ Current VDE"
    else:
        df["is_current"] = False
    # <<< ADD
    return df
def _add_eta_lines_bev(fig, eta_list: List[float]):
    """Wh/km = VDE_NET [MJ/km] * 277.78 / η_drive"""
    import numpy as np
    xs = np.linspace(0.2, 1.2, 40)
    for eta in eta_list:
        ys = xs * 277.7778 / max(eta, 1e-9)
        fig.add_scatter(x=xs, y=ys, mode="lines", name=f"η_drive={eta:.2f}", line=dict(width=1, dash="dot"))


def _add_eta_lines_ice(fig, eta_list: List[float], lhv_mj_per_l: float = 34.2):
    """L/100km = VDE_NET [MJ/km] / (η_pt * LHV) * 100"""
    import numpy as np
    xs = np.linspace(0.2, 1.2, 40)
    for eta in eta_list:
        ys = xs / max(eta, 1e-9) / max(lhv_mj_per_l, 1e-9) * 100.0
        fig.add_scatter(x=xs, y=ys, mode="lines", name=f"η_pt={eta:.2f}", line=dict(width=1, dash="dot"))
def _add_regression_lines(fig, model: Dict[str, Any], electrification: str, y_kind: str):
    """Desenha linha única ou duas linhas (urb/hw)."""
    import numpy as np
    xs = np.linspace(0.2, 1.2, 50)  # faixa típica de VDE_NET; ajuste se quiser

    def _line(a, b):
        return a + b * xs

    if "a" in model and "b" in model:
        ys = _line(model["a"], model["b"])
        fig.add_scatter(x=xs, y=ys, mode="lines", name="Regression", line=dict(dash="dash"))
        return

    if model.get("_is_split"):
        urb, hw, comb = model.get("urb", {}), model.get("hw", {}), model.get("combined", {})
        if urb.get("a") is not None:
            fig.add_scatter(x=xs, y=_line(urb["a"], urb["b"]), mode="lines", name="Reg. Urban", line=dict(dash="dot"))
        if hw.get("a") is not None:
            fig.add_scatter(x=xs, y=_line(hw["a"], hw["b"]), mode="lines", name="Reg. Highway", line=dict(dash="dot"))
        if comb.get("a") is not None:
            fig.add_scatter(x=xs, y=_line(comb["a"], comb["b"]), mode="lines", name="Reg. Combined", line=dict(dash="solid"))


def plot_scatter_with_overlays(df: pd.DataFrame, electrification: str, model: Dict[str, Any] | None, eta_lines: List[float] | None):
    """Mostra scatter filtrado + reta(s) de regressão + linhas iso-eficiência."""
    if df.empty:
        st.info("No data for the selected filters.")
        return
    import plotly.express as px
    import plotly.graph_objects as go

    # separa BEV vs ICE/MxHEV
    df_bev = df[(df.get("is_bev", False) == True) & df["energy_Wh_per_km"].notna()]
    df_ice = df[(df.get("is_bev", False) == False) & df["fuel_l_per_100km"].notna()]
    # >>> ADD: se o contexto/regressão for BEV, mostra só BEV; senão, só combustão
    if str(electrification or "").upper() == "BEV":
        df_ice = df_ice.iloc[0:0]   # esvazia combustão
    else:
        df_bev = df_bev.iloc[0:0]   # esvazia BEV
    # <<< ADD
    figs = []

    if not df_bev.empty:
        fig = px.scatter(
            df_bev, x="vde_net_mj_per_km", y="energy_Wh_per_km",
            color="color_group", hover_data=["make","model","year","vehicle_class"]
        )
        # >>> ADD: trace destacado para o VDE atual (se existir)
        cur = df_bev[df_bev.get("is_current", False) == True]
        if not cur.empty:
            fig.add_scatter(
                x=cur["vde_net_mj_per_km"], y=cur["energy_Wh_per_km"],
                mode="markers", name="★ Current VDE",
                marker=dict(size=12, symbol="star", line=dict(width=1))
            )
        # <<< ADD
        # linhas de regressão (split: urb/hw → duas linhas)
        if model:
            _add_regression_lines(fig, model, electrification, y_kind="bev")
        # iso-η (driveline)
        if eta_lines:
            _add_eta_lines_bev(fig, eta_lines)
        fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), title="BEV: Energy [Wh/km] vs VDE_NET [MJ/km]")
        figs.append(fig)

    if not df_ice.empty:
        fig = px.scatter(
            df_ice, x="vde_net_mj_per_km", y="fuel_l_per_100km",
            color="color_group", hover_data=["make","model","year","engine_size_l","trans_topology"]
        )
        # >>> ADD: trace destacado para o VDE atual (se existir)
        cur = df_ice[df_ice.get("is_current", False) == True]
        if not cur.empty:
            fig.add_scatter(
                x=cur["vde_net_mj_per_km"], y=cur["fuel_l_per_100km"],
                mode="markers", name="★ Current VDE",
                marker=dict(size=12, symbol="star", line=dict(width=1))
            )
        # <<< ADD
        if model:
            _add_regression_lines(fig, model, electrification, y_kind="ice")
        if eta_lines:
            _add_eta_lines_ice(fig, eta_lines)  # usa LHV default/override se quiser
        fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), title="ICE/MxHEV: Fuel [L/100km] vs VDE_NET [MJ/km]")
        figs.append(fig)

    for fig in figs:
        st.plotly_chart(fig, use_container_width=True)

