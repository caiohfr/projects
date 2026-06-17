from __future__ import annotations

import pandas as pd
import streamlit as st

from src.vde_core import tire_roadload_service as tire_service
from src.vde_core.db import ensure_db
from src.vde_core.loaders import load_tire_size_reference, lookup_tire_size_reference


st.set_page_config(page_title="EcoDrive - Tire Database", layout="wide")
ensure_db()


CALCULATION_MODES = ["SAE_J2452", "ISO_28580", "EU_LABEL_ESTIMATED", "CUSTOM"]
SAE_PRESSURE_UNITS = ["psi", "kPa"]
SAE_LOAD_UNITS = ["kg", "lbf", "N"]
KPA_PER_PSI = 6.89475729
N_PER_LBF = 4.4482216152605
G_MPS2 = 9.80665


def _to_float(value, default=None):
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def _mode_to_family(mode: str) -> str:
    mode = str(mode or "").upper()
    if mode == "SAE_J2452":
        return "SAE"
    if mode == "ISO_28580":
        return "ISO"
    return "CUSTOM"


def _mode_label(mode: str) -> str:
    labels = {
        "SAE_J2452": "SAE J2452",
        "ISO_28580": "ISO 28580",
        "EU_LABEL_ESTIMATED": "EU label estimate",
        "CUSTOM": "Custom/manual",
    }
    return labels.get(str(mode or ""), str(mode or ""))


def _row_mode(row: dict) -> str:
    mode = str((row or {}).get("calculation_mode") or "").upper()
    if mode in CALCULATION_MODES:
        return mode
    family = str((row or {}).get("standard_family") or "").upper()
    if family == "SAE":
        return "SAE_J2452"
    if family == "ISO":
        return "ISO_28580"
    return "CUSTOM"


def _sae_pressure_unit_from_row(row: dict) -> str:
    pressure = str((row or {}).get("pressure_unit") or (row or {}).get("sae_pressure_unit") or "").lower()
    return "kPa" if pressure == "kpa" else "psi"


def _sae_load_unit_from_row(row: dict) -> str:
    load = str((row or {}).get("load_unit") or (row or {}).get("sae_load_unit") or "").lower()
    if load in {"kg", "kgf", "kilogram", "kilograms", "kilogram-force", "kilograms-force"}:
        return "kg"
    if load in {"n", "newton", "newtons"}:
        return "N"
    return "lbf"


def _pressure_from_kpa(kpa, unit: str):
    value = _to_float(kpa)
    if value is None:
        return None
    if str(unit or "").lower() == "psi":
        return value / KPA_PER_PSI
    return value


def _pressure_to_kpa(value, unit: str):
    out = _to_float(value)
    if out is None:
        return None
    if str(unit or "").lower() == "psi":
        return out * KPA_PER_PSI
    return out


def _load_from_n(load_n, unit: str):
    value = _to_float(load_n)
    if value is None:
        return None
    if str(unit or "").lower() == "lbf":
        return value / N_PER_LBF
    if str(unit or "").lower() in {"kg", "kgf"}:
        return value / G_MPS2
    return value


def _load_to_n(value, unit: str):
    out = _to_float(value)
    if out is None:
        return None
    if str(unit or "").lower() == "lbf":
        return out * N_PER_LBF
    if str(unit or "").lower() in {"kg", "kgf"}:
        return out * G_MPS2
    return out


def _tire_label(row: dict) -> str:
    return (
        f"#{row.get('id')} | {row.get('manufacturer', '')} {row.get('model', '')} | "
        f"{row.get('size_code', '')} | {_mode_label(_row_mode(row))} | "
        f"{row.get('tire_test_code', '')}"
    )


@st.cache_data(show_spinner=False)
def _load_size_reference_df():
    return load_tire_size_reference()


def _size_options():
    df = _load_size_reference_df()
    if df.empty or "size_code" not in df.columns:
        return []
    return sorted(df["size_code"].dropna().astype(str).unique().tolist())


def _validate_payload(payload: dict) -> list[str]:
    errors = []
    if not str(payload.get("tire_test_code", "")).strip():
        errors.append("Tire test code is required.")
    if not str(payload.get("manufacturer", "")).strip():
        errors.append("Manufacturer is required.")
    if not str(payload.get("model", "")).strip():
        errors.append("Model is required.")
    if not str(payload.get("test_date", "")).strip():
        errors.append("Test date is required.")
    return errors


def _set_state(prefix: str, key: str, value):
    st.session_state[f"{prefix}_{key}"] = value


def _get_state(prefix: str, key: str, default=None):
    return st.session_state.get(f"{prefix}_{key}", default)


def _seed_form_state(prefix: str, row: dict | None = None):
    data = dict(row or {})
    mode = _row_mode(data) if data else "SAE_J2452"
    pressure_unit = _sae_pressure_unit_from_row(data) if data else "kPa"
    load_unit = _sae_load_unit_from_row(data) if data else "kg"
    pressure_value = data.get("test_pressure_value")
    if pressure_value is None:
        pressure_value = _pressure_from_kpa(data.get("sae_reference_pressure_kpa"), pressure_unit)
    load_value = data.get("test_load_value")
    if load_value is None:
        load_value = _load_from_n(data.get("sae_reference_load_n"), load_unit)

    defaults = {
        "tire_test_code": data.get("tire_test_code", ""),
        "manufacturer": data.get("manufacturer", ""),
        "model": data.get("model", ""),
        "test_date": data.get("test_date", ""),
        "calculation_mode": mode,
        "size_code": data.get("size_code", ""),
        "custom_size_code": "",
        "load_index": data.get("load_index", ""),
        "speed_rating": data.get("speed_rating", ""),
        "effective_circumference_override_mm": data.get("effective_circumference_override_mm", 0.0) or 0.0,
        "test_mileage_km": data.get("test_mileage_km", 0.0) or 0.0,
        "is_broken_in": bool(data.get("is_broken_in", 0)),
        "is_tested_value": bool(data.get("is_tested_value", 0)),
        "is_estimated_value": bool(data.get("is_estimated_value", 0)),
        "is_active": bool(data.get("is_active", 1)),
        "notes": data.get("notes", ""),
        "standard_version": data.get("standard_version", ""),
        "test_method": data.get("test_method", ""),
        "test_source": data.get("test_source", ""),
        "test_temperature_c": data.get("test_temperature_c", 25.0) if data.get("test_temperature_c") is not None else 25.0,
        "reference_temperature_c": data.get("reference_temperature_c", 25.0) if data.get("reference_temperature_c") is not None else 25.0,
        "temperature_correction_applied": bool(data.get("temperature_correction_applied", 0)),
        "conditioning_notes": data.get("conditioning_notes", ""),
        "sae_alpha": data.get("sae_alpha", 0.0) if data.get("sae_alpha") is not None else 0.0,
        "sae_beta": data.get("sae_beta", 0.0) if data.get("sae_beta") is not None else 0.0,
        "sae_a": data.get("sae_a", 0.0) if data.get("sae_a") is not None else 0.0,
        "sae_b": data.get("sae_b", 0.0) if data.get("sae_b") is not None else 0.0,
        "sae_c": data.get("sae_c", 0.0) if data.get("sae_c") is not None else 0.0,
        "pressure_unit": pressure_unit,
        "load_unit": load_unit,
        "test_pressure_value": pressure_value or 260.0,
        "test_load_value": load_value or 0.0,
        "test_speed_value": data.get("test_speed_value", 0.0) if data.get("test_speed_value") is not None else 0.0,
        "iso_test_pressure_kpa": data.get("iso_test_pressure_kpa", 0.0) if data.get("iso_test_pressure_kpa") is not None else 0.0,
        "iso_test_load_n": data.get("iso_test_load_n", 0.0) if data.get("iso_test_load_n") is not None else 0.0,
        "iso_test_speed_kph": data.get("iso_test_speed_kph", 0.0) if data.get("iso_test_speed_kph") is not None else 0.0,
        "iso_rrc_n_per_kn": data.get("iso_rrc_n_per_kn", 0.0) if data.get("iso_rrc_n_per_kn") is not None else 0.0,
        "iso_corrected_rrc_n_per_kn": data.get("iso_corrected_rrc_n_per_kn", 0.0) if data.get("iso_corrected_rrc_n_per_kn") is not None else 0.0,
        "iso_rolling_resistance_force_n": data.get("iso_rolling_resistance_force_n", 0.0) if data.get("iso_rolling_resistance_force_n") is not None else 0.0,
        "iso_condition_notes": data.get("iso_condition_notes", ""),
        "rr_n_per_kn": data.get("rr_n_per_kn", 0.0) if data.get("rr_n_per_kn") is not None else 0.0,
        "rr_source": data.get("rr_source", data.get("rr_value_source_note", "")) or "",
        "rr_quality": data.get("rr_quality", "") or "",
    }

    size_options = _size_options()
    if defaults["size_code"] and defaults["size_code"] not in size_options:
        defaults["custom_size_code"] = defaults["size_code"]
        defaults["size_code"] = ""

    for key, value in defaults.items():
        _set_state(prefix, key, value)


def _build_payload(prefix: str) -> dict:
    mode = _get_state(prefix, "calculation_mode", "SAE_J2452")
    size_code = str(_get_state(prefix, "custom_size_code", "") or _get_state(prefix, "size_code", "") or "").strip()
    pressure_unit = _get_state(prefix, "pressure_unit", "kPa")
    load_unit = _get_state(prefix, "load_unit", "kg")
    pressure_value = _to_float(_get_state(prefix, "test_pressure_value"))
    load_value = _to_float(_get_state(prefix, "test_load_value"))

    return {
        "tire_test_code": str(_get_state(prefix, "tire_test_code", "") or "").strip(),
        "manufacturer": str(_get_state(prefix, "manufacturer", "") or "").strip(),
        "model": str(_get_state(prefix, "model", "") or "").strip(),
        "test_date": str(_get_state(prefix, "test_date", "") or "").strip(),
        "calculation_mode": mode,
        "standard_family": _mode_to_family(mode),
        "size_code": size_code,
        "load_index": str(_get_state(prefix, "load_index", "") or "").strip() or None,
        "speed_rating": str(_get_state(prefix, "speed_rating", "") or "").strip() or None,
        "effective_circumference_override_mm": _to_float(_get_state(prefix, "effective_circumference_override_mm")),
        "test_mileage_km": _to_float(_get_state(prefix, "test_mileage_km")),
        "is_broken_in": 1 if _get_state(prefix, "is_broken_in", False) else 0,
        "is_tested_value": 1 if _get_state(prefix, "is_tested_value", False) else 0,
        "is_estimated_value": 1 if _get_state(prefix, "is_estimated_value", False) else 0,
        "is_active": 1 if _get_state(prefix, "is_active", True) else 0,
        "notes": str(_get_state(prefix, "notes", "") or "").strip() or None,
        "standard_version": str(_get_state(prefix, "standard_version", "") or "").strip() or None,
        "test_method": str(_get_state(prefix, "test_method", "") or "").strip() or None,
        "test_source": str(_get_state(prefix, "test_source", "") or "").strip() or None,
        "test_temperature_c": _to_float(_get_state(prefix, "test_temperature_c")),
        "reference_temperature_c": _to_float(_get_state(prefix, "reference_temperature_c")),
        "temperature_correction_applied": 1 if _get_state(prefix, "temperature_correction_applied", False) else 0,
        "conditioning_notes": str(_get_state(prefix, "conditioning_notes", "") or "").strip() or None,
        "sae_alpha": _to_float(_get_state(prefix, "sae_alpha")),
        "sae_beta": _to_float(_get_state(prefix, "sae_beta")),
        "sae_a": _to_float(_get_state(prefix, "sae_a")),
        "sae_b": _to_float(_get_state(prefix, "sae_b")),
        "sae_c": _to_float(_get_state(prefix, "sae_c")),
        "test_pressure_value": pressure_value,
        "test_load_value": load_value,
        "test_speed_value": _to_float(_get_state(prefix, "test_speed_value")),
        "sae_reference_pressure_kpa": _pressure_to_kpa(pressure_value, pressure_unit),
        "sae_reference_load_n": _load_to_n(load_value, load_unit),
        "sae_pressure_unit": "kPa",
        "sae_load_unit": "N",
        "sae_speed_unit": "kph",
        "sae_force_unit": "N",
        "pressure_unit": pressure_unit,
        "load_unit": load_unit,
        "speed_unit": "kph",
        "force_unit": "N",
        "iso_test_pressure_kpa": _to_float(_get_state(prefix, "iso_test_pressure_kpa")),
        "iso_test_load_n": _to_float(_get_state(prefix, "iso_test_load_n")),
        "iso_test_speed_kph": _to_float(_get_state(prefix, "iso_test_speed_kph")),
        "iso_rrc_n_per_kn": _to_float(_get_state(prefix, "iso_rrc_n_per_kn")),
        "iso_corrected_rrc_n_per_kn": _to_float(_get_state(prefix, "iso_corrected_rrc_n_per_kn")),
        "iso_rolling_resistance_force_n": _to_float(_get_state(prefix, "iso_rolling_resistance_force_n")),
        "iso_condition_notes": str(_get_state(prefix, "iso_condition_notes", "") or "").strip() or None,
        "rr_n_per_kn": _to_float(_get_state(prefix, "rr_n_per_kn")),
        "rr_source": str(_get_state(prefix, "rr_source", "") or "").strip() or None,
        "rr_value_source_note": str(_get_state(prefix, "rr_source", "") or "").strip() or None,
        "rr_quality": str(_get_state(prefix, "rr_quality", "") or "").strip() or None,
    }


def _render_identity(prefix: str):
    st.markdown("**Tire Identity**")
    c1, c2, c3, c4 = st.columns(4)
    c1.text_input("Tire test code *", key=f"{prefix}_tire_test_code")
    c2.text_input("Manufacturer *", key=f"{prefix}_manufacturer")
    c3.text_input("Model *", key=f"{prefix}_model")
    c4.text_input("Test date *", key=f"{prefix}_test_date")

    with st.expander("Optional tire geometry", expanded=False):
        size_options = _size_options()
        s1, s2, s3, s4 = st.columns([1.3, 1.3, 1, 1])
        s1.selectbox("Size code", [""] + size_options, key=f"{prefix}_size_code")
        s2.text_input("Custom size code", key=f"{prefix}_custom_size_code")
        s3.text_input("Load index", key=f"{prefix}_load_index")
        s4.text_input("Speed rating", key=f"{prefix}_speed_rating")

        selected_size = _get_state(prefix, "custom_size_code") or _get_state(prefix, "size_code")
        if selected_size:
            size_ref = lookup_tire_size_reference(selected_size)
            if size_ref:
                r1, r2, r3 = st.columns(3)
                r1.metric("Expected effective circ. [mm]", f"{float(size_ref.get('expected_effective_circumference_mm', 0.0)):.1f}")
                r2.metric("Expected rolling radius [mm]", f"{float(size_ref.get('expected_rolling_radius_mm', 0.0)):.1f}")
                r3.number_input(
                    "Circ. override [mm]",
                    min_value=0.0,
                    step=1.0,
                    format="%.1f",
                    key=f"{prefix}_effective_circumference_override_mm",
                )

    with st.expander("Optional record metadata", expanded=False):
        c5, c6, c7, c8 = st.columns(4)
        c5.number_input("Test mileage [km]", min_value=0.0, step=100.0, format="%.1f", key=f"{prefix}_test_mileage_km")
        c6.checkbox("Broken in", key=f"{prefix}_is_broken_in")
        c7.checkbox("Tested value", key=f"{prefix}_is_tested_value")
        c8.checkbox("Estimated value", key=f"{prefix}_is_estimated_value")
        st.text_area("Notes", key=f"{prefix}_notes")


def _render_standard(prefix: str):
    st.markdown("**Standard / Calculation Mode**")
    c1, c2 = st.columns([2, 1])
    c1.selectbox(
        "Calculation mode",
        CALCULATION_MODES,
        format_func=_mode_label,
        key=f"{prefix}_calculation_mode",
    )
    c2.checkbox("Active", key=f"{prefix}_is_active")

    with st.expander("Optional test metadata", expanded=False):
        m1, m2, m3 = st.columns(3)
        m1.text_input("Standard version", key=f"{prefix}_standard_version")
        m2.text_input("Test method", key=f"{prefix}_test_method")
        m3.text_input("Test source", key=f"{prefix}_test_source")

        t1, t2, t3 = st.columns(3)
        t1.number_input("Test temp. [C]", step=1.0, format="%.1f", key=f"{prefix}_test_temperature_c")
        t2.number_input("Reference temp. [C]", step=1.0, format="%.1f", key=f"{prefix}_reference_temperature_c")
        t3.checkbox("Temperature correction", key=f"{prefix}_temperature_correction_applied")
        st.text_input("Conditioning notes", key=f"{prefix}_conditioning_notes")


def _render_sae(prefix: str):
    st.markdown("**SAE J2452 Fields**")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.number_input("alpha", step=0.01, format="%.6f", key=f"{prefix}_sae_alpha")
    c2.number_input("beta", step=0.01, format="%.6f", key=f"{prefix}_sae_beta")
    c3.number_input("a", step=0.0001, format="%.8f", key=f"{prefix}_sae_a")
    c4.number_input("b", step=0.00001, format="%.9f", key=f"{prefix}_sae_b")
    c5.number_input("c", step=0.0000001, format="%.10f", key=f"{prefix}_sae_c")

    r1, r2, r3, r4 = st.columns([1.5, 0.7, 1.5, 0.7])
    r1.number_input(
        "Reference pressure",
        min_value=0.0,
        step=1.0,
        format="%.3f",
        key=f"{prefix}_test_pressure_value",
    )
    r2.selectbox(
        "Unit",
        SAE_PRESSURE_UNITS,
        key=f"{prefix}_pressure_unit",
    )
    r3.number_input(
        "Reference load",
        min_value=0.0,
        step=1.0,
        format="%.3f",
        key=f"{prefix}_test_load_value",
    )
    r4.selectbox(
        "Unit",
        SAE_LOAD_UNITS,
        key=f"{prefix}_load_unit",
    )


def _render_iso(prefix: str):
    st.markdown("**ISO 28580 Fields**")
    c1, c2, c3 = st.columns(3)
    c1.number_input("ISO RRC [N/kN]", min_value=0.0, step=0.1, format="%.3f", key=f"{prefix}_iso_rrc_n_per_kn")
    c2.number_input("Corrected ISO RRC [N/kN]", min_value=0.0, step=0.1, format="%.3f", key=f"{prefix}_iso_corrected_rrc_n_per_kn")
    c3.number_input("Rolling resistance force [N]", min_value=0.0, step=0.1, format="%.3f", key=f"{prefix}_iso_rolling_resistance_force_n")

    i1, i2, i3 = st.columns(3)
    i1.number_input("Test pressure [kPa]", min_value=0.0, step=1.0, format="%.1f", key=f"{prefix}_iso_test_pressure_kpa")
    i2.number_input("Test load [N]", min_value=0.0, step=10.0, format="%.1f", key=f"{prefix}_iso_test_load_n")
    i3.number_input("Test speed [kph]", min_value=0.0, step=1.0, format="%.1f", key=f"{prefix}_iso_test_speed_kph")
    st.text_input("ISO condition notes", key=f"{prefix}_iso_condition_notes")


def _render_manual_rr(prefix: str):
    st.markdown("**Manual / Estimated RR**")
    c1, c2, c3 = st.columns(3)
    c1.number_input("RR [N/kN]", min_value=0.0, step=0.1, format="%.3f", key=f"{prefix}_rr_n_per_kn")
    c2.text_input("RR source", key=f"{prefix}_rr_source")
    c3.text_input("RR quality", key=f"{prefix}_rr_quality")


def _render_rr_summary(prefix: str) -> dict:
    payload = _build_payload(prefix)
    summary = tire_service.summarize_tire_rr(payload)
    rr = summary.get("rr_n_per_kn")
    smerf = summary.get("smerf")

    st.markdown("**RR Summary**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RR [N/kN]", "-" if rr is None else f"{float(rr):.4f}")
    c2.metric("SMERF [N/kN]", "-" if smerf is None else f"{float(smerf):.4f}")
    c3.metric("Method", str(summary.get("rr_method") or "-"))
    c4.metric("Quality", str(summary.get("rr_quality") or "-"))
    return summary


def _render_tire_editor(prefix: str):
    _render_identity(prefix)
    st.divider()
    _render_standard(prefix)

    mode = str(_get_state(prefix, "calculation_mode", "SAE_J2452")).upper()
    st.divider()
    if mode == "SAE_J2452":
        _render_sae(prefix)
    elif mode == "ISO_28580":
        _render_iso(prefix)
    else:
        _render_manual_rr(prefix)

    st.divider()
    summary = _render_rr_summary(prefix)
    return summary


def _display_table(tires: list[dict]):
    if not tires:
        st.info("No tire records match the current filters.")
        return

    df = pd.DataFrame(tires)
    df["mode"] = [_mode_label(_row_mode(row)) for row in tires]
    preferred_cols = [
        "id",
        "tire_test_code",
        "manufacturer",
        "model",
        "size_code",
        "mode",
        "rr_n_per_kn",
        "smerf",
        "rr_method",
        "rr_quality",
        "is_active",
        "updated_at",
    ]
    show_cols = [col for col in preferred_cols if col in df.columns]
    st.dataframe(df[show_cols].sort_values("id", ascending=False), use_container_width=True, hide_index=True)


def main():
    st.title("EcoDrive Analyst - Tire Database")

    with st.sidebar:
        st.header("Filters")
        manufacturer = st.text_input("Manufacturer", value="")
        model = st.text_input("Model", value="")
        size_code = st.text_input("Size code", value="")
        mode_filter = st.selectbox(
            "Calculation mode",
            ["(all)"] + CALCULATION_MODES,
            format_func=lambda value: "All" if value == "(all)" else _mode_label(value),
        )
        include_inactive = st.checkbox("Include inactive", value=False)

    try:
        tires = tire_service.get_available_tires(
            {
                "manufacturer": manufacturer.strip() or None,
                "model": model.strip() or None,
                "size_code": size_code.strip() or None,
                "include_inactive": include_inactive,
            }
        )
    except Exception as exc:
        st.error(f"Failed to load tire database: {exc}")
        st.stop()

    if mode_filter != "(all)":
        tires = [row for row in tires if _row_mode(row) == mode_filter]

    st.subheader("Current Tires")
    _display_table(tires)

    create_tab, edit_tab = st.tabs(["Create", "Edit"])

    with create_tab:
        if "create_tire_test_code" not in st.session_state:
            _seed_form_state("create", {})

        reset_col, _ = st.columns([1, 5])
        if reset_col.button("Reset create form"):
            _seed_form_state("create", {})
            st.rerun()

        _render_tire_editor("create")
        if st.button("Create tire", type="primary", key="create_tire_btn"):
            payload = _build_payload("create")
            errors = _validate_payload(payload)
            if errors:
                for error in errors:
                    st.error(error)
            else:
                try:
                    new_id = tire_service.create_tire_from_form(payload)
                    st.success(f"Tire created with id={new_id}.")
                    _seed_form_state("create", {})
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to create tire: {exc}")

    with edit_tab:
        if not tires:
            st.info("Create at least one tire to edit.")
            return

        options = [int(row["id"]) for row in tires if row.get("id") is not None]
        selected_id = st.selectbox(
            "Select tire",
            options,
            format_func=lambda tire_id: _tire_label(next(row for row in tires if int(row["id"]) == tire_id)),
        )
        selected_row = tire_service.get_tire_by_id(int(selected_id))
        if st.session_state.get("edit_tire_seed_id") != int(selected_id):
            _seed_form_state("edit", selected_row)
            st.session_state["edit_tire_seed_id"] = int(selected_id)

        _render_tire_editor("edit")
        c1, c2, c3 = st.columns([1, 1, 2])
        if c1.button("Save changes", type="primary", key=f"save_tire_{selected_id}"):
            payload = _build_payload("edit")
            errors = _validate_payload(payload)
            if errors:
                for error in errors:
                    st.error(error)
            else:
                try:
                    tire_service.update_tire_from_form(int(selected_id), payload)
                    st.success(f"Tire id={selected_id} updated.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to update tire: {exc}")

        if c2.button("Deactivate", key=f"deactivate_tire_{selected_id}"):
            try:
                tire_service.deactivate_tire_record(int(selected_id))
                st.success(f"Tire id={selected_id} deactivated.")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to deactivate tire: {exc}")

        confirm = c3.text_input("Type DELETE to delete", key=f"delete_confirm_{selected_id}")
        if c3.button("Delete", disabled=(confirm.strip().upper() != "DELETE"), key=f"delete_tire_{selected_id}"):
            try:
                deleted = tire_service.delete_tire_record(int(selected_id))
                if deleted:
                    st.success(f"Tire id={selected_id} deleted.")
                    st.rerun()
                else:
                    st.warning(f"Tire id={selected_id} was not found.")
            except Exception as exc:
                st.error(f"Failed to delete tire: {exc}")


if __name__ == "__main__":
    main()
