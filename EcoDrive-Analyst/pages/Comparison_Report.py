# pages/report_vde_simplificado.py
"""
Simplified VDE comparison report.

Purpose:
- read `vde_db`
- apply lightweight filters
- reuse saved VDE values when available
- optionally recompute EPA/WLTP views
- show KPI, ranking, scatter, and export
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.vde_core.comparison_report_service import load_vde_report_frame
from src.vde_core.cycles import default_cycle_for_legislation, load_cycle_csv
from src.vde_core.phase_aggregation import epa_city_hwy_from_phase, wltp_phases_from_phase


st.set_page_config(page_title="VDE Report (Simplified)", page_icon="📊", layout="wide")
st.title("📊 VDE Report (Simplified)")
st.caption("Lean comparison view for VDE snapshots, filters, ranking, and export.")


DB_PATH = st.session_state.get("ctx", {}).get("DB_PATH", "data/db/eco_drive.db")

try:
    df = load_vde_report_frame(DB_PATH)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

if df.empty:
    st.warning("`vde_db` is empty.")
    st.stop()


def extract_vde_cols(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ["VDE_urb_mj_per_km", "VDE_hwy_mj_per_km", "VDE_net_comb_mj_per_km"]:
        if column not in out.columns:
            out[column] = np.nan
    return out


def calc_vde_if_possible(frame: pd.DataFrame) -> pd.DataFrame:
    out = extract_vde_cols(frame)

    for leg, group in out.groupby("legislation"):
        try:
            fname = default_cycle_for_legislation(str(leg))
            cyc = load_cycle_csv(fname)
            if "v_kph" not in cyc.columns and "speed_kph" in cyc.columns:
                cyc = cyc.rename(columns={"speed_kph": "v_kph"})
            if "time_s" not in cyc.columns and "t_s" in cyc.columns:
                cyc = cyc.rename(columns={"t_s": "time_s"})
        except Exception:
            continue

        for idx, row in group.iterrows():
            A, B, C = row.get("A"), row.get("B"), row.get("C")
            test_mass_kg = row.get("test_mass_kg")
            mass_kg = test_mass_kg if pd.notna(test_mass_kg) else row.get("mass_kg")
            if any(pd.isna(value) for value in [A, B, C, mass_kg]):
                continue

            try:
                if str(leg).upper() == "EPA":
                    result = epa_city_hwy_from_phase(cyc, A, B, C, mass_kg)
                    v_urb = result.get("urb_MJ_km") or result.get("urb_MJ_per_km")
                    v_hwy = result.get("hwy_MJ_km") or result.get("hwy_MJ_per_km")
                    v_comb = result.get("comb_MJ_km") or result.get("comb_MJ_per_km")
                else:
                    result = wltp_phases_from_phase(cyc, A, B, C, mass_kg)
                    v_urb = result.get("urb_MJ_km") or result.get("urb_MJ_per_km")
                    v_hwy = result.get("hwy_MJ_km") or result.get("hwy_MJ_per_km")
                    v_comb = (
                        result.get("comb_MJ_km")
                        or result.get("comb_MJ_per_km")
                        or result.get("VDE_NET")
                    )

                if pd.notna(v_urb):
                    out.at[idx, "VDE_urb_mj_per_km"] = float(v_urb)
                if pd.notna(v_hwy):
                    out.at[idx, "VDE_hwy_mj_per_km"] = float(v_hwy)
                if pd.notna(v_comb):
                    out.at[idx, "VDE_net_comb_mj_per_km"] = float(v_comb)
            except Exception:
                continue

    mask_epa = out["legislation"].astype(str).str.upper().eq("EPA")
    need_comb = out["VDE_net_comb_mj_per_km"].isna() & mask_epa
    can_fill = need_comb & out["VDE_urb_mj_per_km"].notna() & out["VDE_hwy_mj_per_km"].notna()
    out.loc[can_fill, "VDE_net_comb_mj_per_km"] = (
        0.55 * out.loc[can_fill, "VDE_urb_mj_per_km"]
        + 0.45 * out.loc[can_fill, "VDE_hwy_mj_per_km"]
    )
    return out


def normalize_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    mapping = {
        "label_urb_mj_per_km": ["label_urb_mj_per_km", "label_city_mj_per_km", "label_city_MJ_km"],
        "label_hwy_mj_per_km": ["label_hwy_mj_per_km", "label_highway_mj_per_km", "label_hwy_MJ_km"],
        "label_comb_mj_per_km": ["label_comb_mj_per_km", "label_combined_mj_per_km", "label_comb_MJ_km"],
    }
    for dst, candidates in mapping.items():
        if dst in out.columns:
            continue
        found = next((column for column in candidates if column in out.columns), None)
        out[dst] = out[found] if found else np.nan
    return out


col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1])

with col1:
    leg_opts = sorted([value for value in df["legislation"].dropna().unique() if value])
    leg_sel = st.multiselect("Legislation", leg_opts, default=leg_opts or [])

with col2:
    eng_opts = sorted([value for value in df["engine_type"].dropna().unique() if value])
    eng_sel = st.multiselect("Powertrain", eng_opts, default=[])

with col3:
    make_opts = sorted([value for value in df["make"].dropna().unique() if value])
    make_sel = st.multiselect("Make", make_opts, default=[])

with col4:
    cycle_sel = st.selectbox("Cycle", ["combined", "urb", "hwy"], index=0)

mask = pd.Series(True, index=df.index)
if leg_sel:
    mask &= df["legislation"].isin(leg_sel)
if eng_sel:
    mask &= df["engine_type"].isin(eng_sel)
if make_sel:
    mask &= df["make"].isin(make_sel)

dfv = df[mask].copy()
if dfv.empty:
    st.info("No vehicles left after filters.")
    st.stop()

baseline = st.selectbox("Baseline", dfv["veh_label"].tolist(), index=0)
dfv = normalize_labels(calc_vde_if_possible(dfv))

if cycle_sel == "urb":
    dfv["VDE_view_mj_per_km"] = dfv["VDE_urb_mj_per_km"]
    dfv["Label_view_mj_per_km"] = dfv["label_urb_mj_per_km"]
elif cycle_sel == "hwy":
    dfv["VDE_view_mj_per_km"] = dfv["VDE_hwy_mj_per_km"]
    dfv["Label_view_mj_per_km"] = dfv["label_hwy_mj_per_km"]
else:
    dfv["VDE_view_mj_per_km"] = dfv["VDE_net_comb_mj_per_km"]
    dfv["Label_view_mj_per_km"] = dfv["label_comb_mj_per_km"]

baseline_row = dfv[dfv["veh_label"].eq(baseline)].head(1)
baseline_vde = (
    float(baseline_row.iloc[0]["VDE_view_mj_per_km"])
    if not baseline_row.empty and pd.notna(baseline_row.iloc[0]["VDE_view_mj_per_km"])
    else np.nan
)

k1, k2 = st.columns(2)
with k1:
    st.metric(f"VDE_{cycle_sel.upper()} (Baseline)", f"{baseline_vde:.3f} MJ/km" if pd.notna(baseline_vde) else "n/a")
with k2:
    st.metric("Vehicles in subset", f"{len(dfv)}")

st.markdown("---")

rank = dfv[["veh_label", "engine_type", "VDE_view_mj_per_km"]].dropna().copy()
if rank.empty:
    st.info("Not enough VDE data for ranking.")
else:
    rank = rank.sort_values("VDE_view_mj_per_km", ascending=True)
    fig_rank = px.bar(
        rank,
        x="veh_label",
        y="VDE_view_mj_per_km",
        color="engine_type",
        title=f"Ranking - VDE_{cycle_sel.upper()} (MJ/km)",
    )
    fig_rank.update_layout(xaxis_title="", yaxis_title="MJ/km", showlegend=True)
    st.plotly_chart(fig_rank, use_container_width=True)

scatter = dfv[["veh_label", "engine_type", "VDE_view_mj_per_km", "Label_view_mj_per_km"]].dropna().copy()
if scatter.shape[0] >= 3 and scatter["Label_view_mj_per_km"].notna().sum() >= 3:
    fig_scatter = px.scatter(
        scatter,
        x="VDE_view_mj_per_km",
        y="Label_view_mj_per_km",
        color="engine_type",
        hover_name="veh_label",
        title=f"VDE_{cycle_sel.upper()} vs Label (MJ/km)",
    )
    fig_scatter.update_layout(xaxis_title="VDE (MJ/km)", yaxis_title="Label (MJ/km)")
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.caption("Official label data is not available in enough quantity for the optional scatter.")

st.markdown("---")

show_cols = [
    "veh_label",
    "legislation",
    "engine_type",
    "year",
    "A",
    "B",
    "C",
    "mass_kg",
    "VDE_urb_mj_per_km",
    "VDE_hwy_mj_per_km",
    "VDE_net_comb_mj_per_km",
    "label_urb_mj_per_km",
    "label_hwy_mj_per_km",
    "label_comb_mj_per_km",
]
show_cols = [column for column in show_cols if column in dfv.columns]

st.subheader("Data")
st.dataframe(dfv[show_cols], use_container_width=True, hide_index=True)

st.download_button(
    "Export CSV",
    data=dfv[show_cols].to_csv(index=False).encode("utf-8"),
    file_name="vde_report_simplificado.csv",
    mime="text/csv",
)

st.caption(
    "If recomputation is not possible for a row, the report keeps the VDE values already saved in the DB "
    "and fills EPA combined as 0.55 * urban + 0.45 * highway whenever possible."
)
