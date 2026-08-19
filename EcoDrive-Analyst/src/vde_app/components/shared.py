from __future__ import annotations

import base64
import difflib
import html
import mimetypes
import re
import unicodedata
from pathlib import Path

import pandas as pd
import streamlit as st

from ..state import ensure_defaults
from src.vde_core.vde_calc import compute_vde_net_mj_per_km


def sidebar_inputs():
    ensure_defaults(st.session_state)
    st.sidebar.header("Parameters (sidebar)")
    return st.session_state["roadload_params"]


def pressure_input_with_units(key_prefix=""):
    unit = st.radio("Unit", ["kPa", "psi"], key=f"{key_prefix}press_unit", horizontal=True)
    base_kpa = float(st.session_state.get(f"{key_prefix}press_kpa", 230.0))
    default_display = base_kpa if unit == "kPa" else base_kpa / 6.89475729
    val = st.number_input(
        f"Pressure [{unit}]",
        0.0,
        500.0 if unit == "kPa" else 100.0,
        default_display,
        step=1.0 if unit == "kPa" else 0.5,
        key=f"{key_prefix}press_val",
    )
    kpa = val if unit == "kPa" else val * 6.89475729
    st.session_state[f"{key_prefix}press_kpa"] = kpa
    st.caption(f"{kpa:.1f} kPa ~= {kpa / 6.89475729:.1f} psi")
    return kpa


def vde_by_phase(df_cycle, leg, A, B, C, mass_kg):
    out = {}
    if not isinstance(df_cycle, pd.DataFrame) or "phase" not in df_cycle.columns:
        return out

    def _norm(phase_name):
        p = str(phase_name).strip().lower()
        if leg == "EPA":
            if "city" in p or "ftp" in p:
                return "city"
            if "hwy" in p or "hwfet" in p or "highway" in p:
                return "hwy"
        else:
            if "low" in p:
                return "low"
            if "mid" in p or "medium" in p:
                return "mid"
            if "high" in p and "extra" not in p:
                return "high"
            if "xhigh" in p or "extra" in p:
                return "xhigh"
        return p

    for phase_name in df_cycle["phase"].unique():
        sub = df_cycle[df_cycle["phase"] == phase_name]
        result = compute_vde_net_mj_per_km(sub, A, B, C, mass_kg)
        out[_norm(phase_name)] = float(result["MJ_km"]) if isinstance(result, dict) else float(result)
    return out


def show_vde_feedback(overall_mj_km, by_phase):
    st.success(f"VDE_NET (cycle total) ~= {overall_mj_km:.4f} MJ/km")
    if by_phase:
        preferred = ["city", "hwy", "low", "mid", "high", "xhigh"]
        ordered = [key for key in preferred if key in by_phase] + [key for key in by_phase if key not in preferred]
        cols = st.columns(min(4, len(ordered)))
        for index, key in enumerate(ordered):
            cols[index % len(cols)].metric(key.upper(), f"{by_phase[key]:.4f} MJ/km")


def search_logo(ctx: dict, base_dir: str = "data/logos", fallback: str | None = None) -> str | None:
    make_raw = str((ctx or {}).get("make", "")).strip()
    if not make_raw:
        return None

    def _slugify(value: str) -> str:
        value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
        value = value.lower().replace("&", " and ")
        value = re.sub(r"[^a-z0-9]+", "-", value)
        value = re.sub(r"-{2,}", "-", value).strip("-")
        return value

    slug = _slugify(make_raw)
    base = Path(base_dir)

    direct = base / f"{slug}.png"
    if direct.exists():
        return str(direct)

    synonyms = {
        "mercedes": "mercedes-benz",
        "landrover": "land-rover",
        "vw": "volkswagen",
        "chevy": "chevrolet",
        "byd-auto": "byd",
        "bayerische-motoren-werke": "bmw",
        "citroen": "citroen",
    }
    alt = synonyms.get(slug)
    if alt:
        alt_path = base / f"{alt}.png"
        if alt_path.exists():
            return str(alt_path)

    pngs = list(base.glob("*.png"))
    if pngs:
        norm_map = {}
        for fp in pngs:
            norm = _slugify(fp.stem)
            norm_map[norm] = fp
            if norm == slug:
                return str(fp)

        for norm, fp in norm_map.items():
            if norm.startswith(slug) or slug in norm:
                return str(fp)

        hit = difflib.get_close_matches(slug, list(norm_map.keys()), n=1, cutoff=0.84)
        if hit:
            return str(norm_map[hit[0]])

    if fallback:
        fb = Path(fallback)
        if not fb.is_absolute():
            fb = base / fb
        if fb.exists():
            return str(fb)

    return None


def get_legislation_icon(ctx: dict, base_dir: str = "data/icons") -> str | None:
    leg = str((ctx or {}).get("legislation", "")).strip().lower().replace(" ", "")
    mapping = {
        "epa": "flag_usa.png",
        "usa": "flag_usa.png",
        "us": "flag_usa.png",
        "wltp": "flag_eu.png",
        "eu": "flag_eu.png",
        "unece": "flag_eu.png",
        "bra": "flag_brazil.png",
        "br": "flag_brazil.png",
        "proconve": "flag_brazil.png",
        "pbev": "flag_brazil.png",
        "mover": "flag_brazil.png",
    }

    fname = mapping.get(leg)
    if not fname:
        for key, value in mapping.items():
            if key in leg:
                fname = value
                break

    if not fname:
        return None

    path = Path(base_dir) / fname
    return str(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def _inline_image_data_uri(path: str) -> str | None:
    image_path = Path(path)
    if not image_path.exists() or not image_path.is_file():
        return None
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if not mime_type:
        mime_type = "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def render_inline_image(path: str | Path | None, *, width: int | None = None, caption: str | None = None) -> bool:
    image_path = Path(path) if path else None
    if not image_path or not image_path.exists():
        return False
    data_uri = _inline_image_data_uri(str(image_path))
    if not data_uri:
        return False
    width_style = f"width:{int(width)}px;max-width:100%;height:auto;" if width else "max-width:100%;height:auto;"
    caption_html = (
        f'<div style="margin-top:0.35rem;font-size:0.8rem;color:#667085;">{html.escape(str(caption))}</div>'
        if caption
        else ""
    )
    st.markdown(
        (
            '<div style="display:flex;flex-direction:column;align-items:center;gap:0;">'
            f'<img src="{data_uri}" style="{width_style}" />'
            f"{caption_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    return True
