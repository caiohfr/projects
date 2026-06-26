import json

import pandas as pd
import streamlit as st
import numpy as np
from urllib.parse import quote_plus
from pathlib import Path

from src.vde_core.db import ensure_db
from src.vde_core.cycles import default_cycle_for_legislation, load_cycle_csv, use_standard_cycle
from src.vde_core.services import estimate_aux_from_coastdown, load_vde_defaults
from src.vde_core.tire_roadload_service import (
    create_tire_from_form,
    get_available_tires,
    get_tire_by_code,
    get_tire_by_id,
    preview_tire_roadload_from_row,
    save_tire_roadload_to_vde,
    summarize_tire_rr,
)
from src.vde_core.vde_setup_service import (
    build_decomp_update_for_edit,
    build_edit_core_update,
    build_test_mass_hint,
    build_vde_phase_update,
    build_live_vde_preview,
    collect_ctx_updates,
    compute_vde_preview_from_inputs,
    delete_vde_snapshot,
    fetch_linked_fuelcons_count,
    fetch_vde_edit_rows,
    fetch_vde_rows_full,
    ensure_baseline_aliases,
    baseline_filter_options,
    apply_baseline_filters,
    build_baseline_state_payload,
    build_delta_mode_ctx_updates,
    db_list_makes,
    merge_update_payloads,
    resolve_tire_calculation_mass,
    resolve_tire_load_mass_basis,
    resolve_test_mass_kg,
    to_float,
    validate_core,
    update_vde_snapshot,
)
from src.vde_core.vde_workflow_service import (
    build_vde_pre_save_review,
    build_vde_workflow_payload_from_ctx,
    build_vde_setup_preview_from_ctx,
    save_vde_setup_result,
    summarize_component_build_up_from_ctx,
)
from src.vde_app.plots import cycle_chart
from src.vde_app.components.shared import show_vde_feedback
from src.vde_app.units import (
    format_quantity,
    normalize_unit_system,
    quantity_input,
    quantity_metric,
    to_display,
    unit_label,
)
from src.vde_core.loaders import load_tire_size_reference, lookup_tire_size_reference

TIRE_KPA_PER_PSI = 6.89475729317


def _tire_label(row: dict) -> str:
    return (
        f"#{row.get('id')} | "
        f"{row.get('manufacturer', '')} {row.get('model', '')} | "
        f"{row.get('size_code', '')} | "
        f"{str(row.get('standard_family', '')).upper()} | "
        f"{row.get('tire_test_code', '')}"
    )


def _current_unit_system() -> str:
    return normalize_unit_system(st.session_state.get("unit_system"))


@st.cache_data(show_spinner=False)
def _cycle_speed_moments_kph(cycle_name: str) -> tuple[float | None, float | None]:
    try:
        df = load_cycle_csv(cycle_name)
    except Exception:
        return None, None
    if df is None or df.empty or "v" not in df.columns:
        return None, None
    v_vals = pd.to_numeric(df["v"], errors="coerce").dropna()
    if v_vals.empty:
        return None, None
    v_kph = v_vals * 3.6
    return float(v_kph.mean()), float((v_kph ** 2).mean())


def _epa_weighted_speed_moments_kph() -> tuple[float | None, float | None]:
    ftp_avg, ftp_avg2 = _cycle_speed_moments_kph("FTP75")
    hwfet_avg, hwfet_avg2 = _cycle_speed_moments_kph("HWFET")
    if ftp_avg is None or ftp_avg2 is None or hwfet_avg is None or hwfet_avg2 is None:
        return None, None
    return (
        (0.55 * ftp_avg) + (0.45 * hwfet_avg),
        (0.55 * ftp_avg2) + (0.45 * hwfet_avg2),
    )


def _equivalent_rr_from_abc(
    a_force,
    b_force,
    c_force,
    *,
    load_kN,
) -> tuple[float | None, float | None, float | None]:
    epa_v_avg, epa_v2_avg = _epa_weighted_speed_moments_kph()
    if load_kN is None or float(load_kN or 0.0) <= 0.0:
        return None, epa_v_avg, epa_v2_avg
    effective_force = float(to_float(a_force, 0.0) or 0.0)
    if epa_v_avg is not None and epa_v2_avg is not None:
        effective_force = (
            float(to_float(a_force, 0.0) or 0.0)
            + (float(to_float(b_force, 0.0) or 0.0) * epa_v_avg)
            + (float(to_float(c_force, 0.0) or 0.0) * epa_v2_avg)
        )
    return effective_force / float(load_kN), epa_v_avg, epa_v2_avg


def _abc_from_final_rr_target(
    rr_n_per_kn,
    *,
    load_kN,
    crr_frac_120,
) -> tuple[float, float, float, float | None]:
    target_rr = float(to_float(rr_n_per_kn, 0.0) or 0.0)
    target_force = target_rr * float(to_float(load_kN, 0.0) or 0.0)
    crr_frac = float(to_float(crr_frac_120, 0.0) or 0.0)
    epa_v_avg, _ = _epa_weighted_speed_moments_kph()
    if epa_v_avg is not None:
        denom = 1.0 + ((crr_frac * epa_v_avg) / 120.0)
        a_rr = target_force / denom if abs(denom) > 1e-12 else target_force
    else:
        a_rr = target_force
    b_rr = (crr_frac * a_rr) / 120.0 if a_rr else 0.0
    return a_rr, b_rr, 0.0, epa_v_avg


def _string_or_none(value) -> str | None:
    text = str(value or "").strip()
    return text or None


def _truthy_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _mm_display(value) -> str:
    numeric = to_float(value)
    if numeric is None:
        return "Pending"
    if _current_unit_system() == "US customary":
        return f"{numeric / 25.4:.2f} in"
    return f"{numeric:.0f} mm"


def _tire_pressure_display_unit() -> str:
    ctx = st.session_state.ctx
    unit = str(ctx.get("tire_pressure_display_unit") or "psi").strip()
    return unit if unit in {"psi", "kPa"} else "psi"


def _format_tire_pressure(value_psi) -> str:
    numeric = to_float(value_psi)
    if numeric is None:
        return "Pending"
    unit = _tire_pressure_display_unit()
    display_value = numeric if unit == "psi" else numeric * TIRE_KPA_PER_PSI
    return f"{display_value:.1f} {unit}"


def _render_tire_pressure_unit_toggle(*, key: str = "tire_pressure_display_unit_toggle"):
    ctx = st.session_state.ctx
    current = _tire_pressure_display_unit()
    ctx["tire_pressure_display_unit"] = st.radio(
        "Tire pressure unit",
        ["psi", "kPa"],
        horizontal=True,
        index=(0 if current == "psi" else 1),
        key=key,
    )
    return ctx["tire_pressure_display_unit"]


def _render_tire_pressure_input(container, label: str, value_psi, *, key_base: str, default_psi: float = 32.0) -> float:
    canonical_psi = to_float(value_psi, default_psi)
    unit = _tire_pressure_display_unit()
    factor = 1.0 if unit == "psi" else TIRE_KPA_PER_PSI
    display_value = float(canonical_psi or 0.0) * factor
    display_input = container.number_input(
        f"{label} [{unit}]",
        value=float(display_value),
        step=0.5 if unit == "psi" else 5.0,
        format="%.1f",
        key=f"{key_base}_{unit.lower()}",
    )
    return float(display_input) / factor


def _distance_display_km(value) -> str:
    numeric = to_float(value)
    if numeric is None:
        return "Pending"
    if _current_unit_system() == "US customary":
        return f"{numeric * 0.621371192237:.0f} mi"
    return f"{numeric:.0f} km"


def _tire_size_reference_details(size_code: str | None) -> dict:
    size_code = _string_or_none(size_code)
    if not size_code:
        return {}
    try:
        return dict(lookup_tire_size_reference(size_code) or {})
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def _load_tire_size_reference_df():
    return load_tire_size_reference()


def _tire_size_options() -> list[str]:
    try:
        df = _load_tire_size_reference_df()
    except Exception:
        return []
    if df.empty or "size_code" not in df.columns:
        return []
    return sorted(df["size_code"].dropna().astype(str).unique().tolist())


def _render_tire_size_selector(*, field_key: str, label: str = "Tire size") -> str:
    ctx = st.session_state.ctx
    size_options = _tire_size_options()
    current_value = str(ctx.get(field_key) or "").strip()
    select_key = f"{field_key}_select"
    custom_key = f"{field_key}_custom"

    if size_options:
        select_options = [""] + size_options
        selected_option = current_value if current_value in size_options else ""
        custom_default = current_value if current_value and current_value not in size_options else str(ctx.get(custom_key) or "")
        c1, c2 = st.columns([1.3, 1.0])
        chosen = c1.selectbox(
            label,
            select_options,
            index=select_options.index(selected_option),
            key=select_key,
        )
        custom_value = c2.text_input(
            "Custom size code",
            value=custom_default,
            key=custom_key,
        )
        resolved = str(custom_value or chosen or "").strip()
    else:
        resolved = st.text_input(
            label,
            value=current_value,
            key=f"{field_key}_text",
        ).strip()
    ctx[field_key] = resolved
    return resolved


def _prefill_circumference_from_size(*, size_field_key: str, circumference_field_key: str) -> float | None:
    ctx = st.session_state.ctx
    size_code = _string_or_none(ctx.get(size_field_key))
    if not size_code:
        return to_float(ctx.get(circumference_field_key))
    size_ref = _tire_size_reference_details(size_code)
    target_mm = to_float(size_ref.get("expected_effective_circumference_mm"))
    if target_mm is None:
        target_mm = to_float(size_ref.get("unloaded_circumference_mm"))
    if target_mm is None or target_mm <= 0:
        return to_float(ctx.get(circumference_field_key))

    previous_size = _string_or_none(ctx.get(f"{circumference_field_key}_prefill_size"))
    previous_value = to_float(ctx.get(f"{circumference_field_key}_prefill_value"))
    current_value = to_float(ctx.get(circumference_field_key))
    should_prefill = (
        current_value is None
        or abs(float(current_value or 0.0)) <= 0.0
        or (
            size_code != previous_size
            and previous_value is not None
            and current_value is not None
            and abs(float(current_value) - float(previous_value)) <= 1e-6
        )
        or (size_code != previous_size and previous_size is None)
    )
    if should_prefill:
        ctx[circumference_field_key] = float(target_mm)
    ctx[f"{circumference_field_key}_prefill_size"] = size_code
    ctx[f"{circumference_field_key}_prefill_value"] = float(target_mm)
    return to_float(ctx.get(circumference_field_key))


def _tire_circumference_mm(reference: dict | None) -> float | None:
    data = dict(reference or {})
    explicit = to_float(data.get("effective_circumference_override_mm"))
    if explicit is not None and explicit > 0:
        return explicit
    size_ref = _tire_size_reference_details(data.get("size_code"))
    for key in ("expected_effective_circumference_mm", "unloaded_circumference_mm"):
        numeric = to_float(size_ref.get(key))
        if numeric is not None and numeric > 0:
            return numeric
    return None


def _reference_has_nonzero_abc(reference: dict | None) -> bool:
    data = dict(reference or {})
    return any(abs(float(to_float(data.get(key), 0.0) or 0.0)) > 0.0 for key in ("A", "B", "C"))


def _build_tire_reference(
    *,
    source: str,
    abc: dict | None = None,
    front_tire: dict | None = None,
    rear_tire: dict | None = None,
    front_pressure_psi=None,
    rear_pressure_psi=None,
    tire_load_mass_basis=None,
    tire_load_mass_used_kg=None,
    tire_calc_source=None,
    tire_calc_notes=None,
    extra: dict | None = None,
) -> dict:
    ref = {
        "source": source,
        "A": float(to_float((abc or {}).get("A"), 0.0) or 0.0),
        "B": float(to_float((abc or {}).get("B"), 0.0) or 0.0),
        "C": float(to_float((abc or {}).get("C"), 0.0) or 0.0),
        "front_tire": dict(front_tire or {}),
        "rear_tire": dict(rear_tire or {}),
        "front_tire_id": (front_tire or {}).get("id"),
        "rear_tire_id": (rear_tire or {}).get("id"),
        "front_pressure_psi": to_float(front_pressure_psi),
        "rear_pressure_psi": to_float(rear_pressure_psi),
        "tire_load_mass_basis": tire_load_mass_basis,
        "tire_load_mass_used_kg": to_float(tire_load_mass_used_kg),
        "tire_calc_source": tire_calc_source,
        "tire_calc_notes": tire_calc_notes,
    }
    if front_tire:
        ref["tire_test_code"] = _string_or_none(front_tire.get("tire_test_code"))
        ref["manufacturer"] = _string_or_none(front_tire.get("manufacturer"))
        ref["model"] = _string_or_none(front_tire.get("model"))
        ref["size_code"] = _string_or_none(front_tire.get("size_code"))
        ref["load_index"] = _string_or_none(front_tire.get("load_index"))
        ref["speed_rating"] = _string_or_none(front_tire.get("speed_rating"))
        ref["standard_family"] = _string_or_none(front_tire.get("standard_family"))
        ref["rr_n_per_kn"] = to_float(front_tire.get("rr_n_per_kn"))
        ref["smerf"] = to_float(front_tire.get("smerf"))
        ref["test_mileage_km"] = to_float(front_tire.get("test_mileage_km"))
        ref["test_method"] = _string_or_none(front_tire.get("test_method"))
        ref["test_source"] = _string_or_none(front_tire.get("test_source"))
        ref["is_tested_value"] = _truthy_flag(front_tire.get("is_tested_value"))
        ref["effective_circumference_override_mm"] = to_float(front_tire.get("effective_circumference_override_mm"))
    if extra:
        ref.update({k: v for k, v in extra.items() if v not in (None, "")})
    ref["circumference_mm"] = _tire_circumference_mm(ref)
    return ref


def _manual_tire_reference_from_ctx(prefix: str, *, source_label: str) -> dict | None:
    ctx = st.session_state.ctx
    manual_basis = str(ctx.get(f"{prefix}_manual_basis") or "RRC-based reference").strip()
    user_notes = _string_or_none(ctx.get(f"{prefix}_notes"))
    load_kN = _resolved_tire_load_kN(ctx)
    rr_n_per_kn = to_float(ctx.get(f"{prefix}_rr_n_per_kn"))

    if manual_basis == "Direct ABC override":
        abc = {
            "A": to_float(ctx.get(f"{prefix}_A")),
            "B": to_float(ctx.get(f"{prefix}_B")),
            "C": to_float(ctx.get(f"{prefix}_C")),
        }
        if not any(value is not None and abs(float(value)) > 0.0 for value in abc.values()):
            return None
        equivalent_rr, epa_v_avg, epa_v2_avg = _equivalent_rr_from_abc(
            abc.get("A"),
            abc.get("B"),
            abc.get("C"),
            load_kN=load_kN,
        )
        rr_n_per_kn = equivalent_rr
        tire_calc_source = "scenario_manual_reference_abc_override"
        derived_note = None
        if equivalent_rr is not None:
            if epa_v_avg is not None and epa_v2_avg is not None:
                derived_note = (
                    f"Equivalent RRC derived from direct ABC using EPA weighted v_avg={epa_v_avg:.2f} kph, "
                    f"v2_avg={epa_v2_avg:.2f} kph^2 and load={load_kN:.3f} kN"
                )
            else:
                derived_note = f"Equivalent RRC derived from direct ABC using load={load_kN:.3f} kN"
        tire_calc_notes = (
            derived_note if not user_notes else f"{derived_note} | {user_notes}"
        ) if derived_note else user_notes
    else:
        if rr_n_per_kn is None or rr_n_per_kn <= 0.0:
            return None
        crr_frac = float(to_float(ctx.get("crr1_frac_at_120kph"), 0.0) or 0.0)
        a_rr, b_rr, c_rr, epa_v_avg = _abc_from_final_rr_target(
            rr_n_per_kn,
            load_kN=load_kN,
            crr_frac_120=crr_frac,
        )
        abc = {"A": a_rr, "B": b_rr, "C": c_rr}
        tire_calc_source = "scenario_manual_reference_rrc"
        if epa_v_avg is not None:
            derived_note = (
                f"Derived from final RRC target using EPA v_avg={epa_v_avg:.2f} kph, "
                f"load={load_kN:.3f} kN and crr1@120={crr_frac:.5f}"
            )
        else:
            derived_note = (
                f"Derived from manual RRC using load={load_kN:.3f} kN and crr1@120={crr_frac:.5f}"
            )
        tire_calc_notes = derived_note if not user_notes else f"{derived_note} | {user_notes}"

    extra = {
        "tire_test_code": _string_or_none(ctx.get(f"{prefix}_tire_test_code")),
        "manufacturer": _string_or_none(ctx.get(f"{prefix}_manufacturer")),
        "model": _string_or_none(ctx.get(f"{prefix}_model")),
        "size_code": _string_or_none(ctx.get(f"{prefix}_size_code")),
        "standard_family": _string_or_none(ctx.get(f"{prefix}_standard_family")),
        "rr_n_per_kn": rr_n_per_kn,
        "smerf": to_float(ctx.get(f"{prefix}_smerf")),
        "test_mileage_km": to_float(ctx.get(f"{prefix}_test_mileage_km")),
        "test_method": _string_or_none(ctx.get(f"{prefix}_test_method")),
        "test_source": _string_or_none(ctx.get(f"{prefix}_test_source")),
        "is_tested_value": _truthy_flag(ctx.get(f"{prefix}_is_tested_value")),
        "effective_circumference_override_mm": to_float(ctx.get(f"{prefix}_effective_circumference_override_mm")),
        "front_pressure_psi": to_float(ctx.get(f"{prefix}_front_pressure_psi")),
        "rear_pressure_psi": to_float(ctx.get(f"{prefix}_rear_pressure_psi")),
        "tire_calc_source": tire_calc_source,
        "tire_calc_notes": tire_calc_notes,
    }
    return _build_tire_reference(
        source=source_label,
        abc=abc,
        front_pressure_psi=extra.pop("front_pressure_psi"),
        rear_pressure_psi=extra.pop("rear_pressure_psi"),
        tire_calc_source=extra.pop("tire_calc_source"),
        tire_calc_notes=extra.pop("tire_calc_notes"),
        extra=extra,
    )


def _tire_provenance_note(reference: dict | None) -> str:
    data = dict(reference or {})
    parts = []
    for key in ("source", "tire_test_code", "manufacturer", "model", "size_code", "standard_family"):
        value = _string_or_none(data.get(key))
        if value:
            parts.append(f"{key}={value}")
    rr_value = to_float(data.get("rr_n_per_kn"))
    if rr_value is not None:
        parts.append(f"rr_n_per_kn={rr_value:.3f}")
    if data.get("tire_calc_notes"):
        parts.append(str(data.get("tire_calc_notes")))
    return "; ".join(parts)

def _compact_abc(values: dict | None) -> str:
    data = dict(values or {})
    return (
        f"{format_quantity(data.get('A'), 'force', include_unit=False, unavailable='0.00', format_str='%.2f')} / "
        f"{format_quantity(data.get('B'), 'force_per_speed', include_unit=False, unavailable='0.00000', format_str='%.5f')} / "
        f"{format_quantity(data.get('C'), 'force_per_speed_squared', include_unit=False, unavailable='0.000000', format_str='%.6f')}"
    )


def _format_energy_value(value, *, unavailable: str = "Unavailable") -> str:
    return format_quantity(value, "energy_per_distance", unavailable=unavailable)


def _component_build_up_enabled(ctx: dict | None = None) -> bool:
    data = dict(ctx or st.session_state.get("ctx", {}))
    source_ui = str(data.get("abc_total_source_ui") or "").strip()
    if source_ui:
        return source_ui == "Component Build-up"
    payload = build_vde_workflow_payload_from_ctx(data)
    return str(payload.get("initial_abc_total_source") or "").strip().upper() == "COMPONENT_BUILD_UP"


def _roadload_basis_value(ctx: dict | None = None) -> str:
    data = dict(ctx or st.session_state.get("ctx", {}))
    source_ui = str(data.get("abc_total_source_ui") or "").strip()
    if source_ui:
        return source_ui
    mode = str(data.get("mode") or "").strip()
    if mode == "From baseline (editable)":
        return "Baseline ABC"
    return "Component Build-up"


def _component_mode_key(component_label: str) -> str:
    slug = (
        component_label.lower()
        .replace(" / ", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )
    return f"component_mode_{slug}"


def _component_mode_options(component_label: str, basis: str) -> tuple[list[str], str]:
    build_mode = basis == "Component Build-up"
    if build_mode:
        if component_label == "Tires":
            return ["Replace / manual input", "Lookup from DB"], "Replace / manual input"
        return ["Replace / manual input"], "Replace / manual input"
    return ["Keep inherited", "Apply delta"], "Keep inherited"


def _component_mode_default(component_label: str, basis: str, ctx: dict) -> str:
    options, fallback = _component_mode_options(component_label, basis)
    key = _component_mode_key(component_label)
    current = str(ctx.get(key) or "").strip()
    if current in options:
        return current

    if basis != "Component Build-up":
        if component_label == "Tires" and abs(float(to_float(ctx.get("delta_rr_N"), 0.0) or 0.0)) > 0.0:
            return "Apply delta"
        if component_label == "Aerodynamics" and abs(float(to_float(ctx.get("delta_aero_cdA"), 0.0) or 0.0)) > 0.0:
            return "Apply delta"
        if component_label == "Brakes" and abs(float(to_float(ctx.get("delta_brake_N"), 0.0) or 0.0)) > 0.0:
            return "Apply delta"
        if component_label == "Parasitics / Hubs / Axle" and abs(float(to_float(ctx.get("delta_parasitics_N"), 0.0) or 0.0)) > 0.0:
            return "Apply delta"
    elif component_label == "Tires" and str(ctx.get("tire_component_source") or "").strip() == "Tire DB":
        return "Lookup from DB"

    return fallback


def _resolve_component_mode(component_label: str, ctx: dict | None = None) -> str:
    data = dict(ctx or st.session_state.get("ctx", {}))
    basis = _roadload_basis_value(data)
    key = _component_mode_key(component_label)
    resolved = _component_mode_default(component_label, basis, data)
    if ctx is None:
        st.session_state.ctx[key] = resolved
    return resolved


def _component_overview_rows(ctx: dict) -> list[dict]:
    basis = _roadload_basis_value(ctx)
    active = str(ctx.get("component_editor_active") or "Tires")
    tire_mode = _resolve_component_mode("Tires", ctx)
    brake_mode = _resolve_component_mode("Brakes", ctx)
    parasitic_mode = _resolve_component_mode("Parasitics / Hubs / Axle", ctx)
    selected_baseline_row = dict(ctx.get("selected_baseline_row") or {})
    baseline_mode = str(ctx.get("mode") or "") == "From baseline (editable)"
    baseline_tire_reference = _saved_tire_reference_from_row(selected_baseline_row)
    scenario_current_reference = _resolve_tire_reference_from_ctx(
        "tire_current_reference",
        source_label="Scenario Current reference",
        preview_ctx_key="tire_current_reference_preview_result",
    )
    current_tire_reference = scenario_current_reference or baseline_tire_reference
    walked_tire_reference = _resolve_tire_reference_from_ctx(
        "tire_walked_reference",
        source_label="Scenario Walked reference",
        preview_ctx_key="tire_walked_reference_preview_result",
    )
    tire_application = _normalize_tire_scenario_application(ctx.get("tire_scenario_application"))

    if basis == "Component Build-up":
        tire_reference = "Absolute source"
        tire_active_change = "Absolute tire source" if (
            (str(ctx.get("tire_component_source") or "").strip() == "Tire DB" and ctx.get("tire_preview_result"))
            or any(abs(float(to_float(ctx.get(key), 0.0) or 0.0)) > 0.0 for key in ("rr_alpha_N", "rr_beta_Npkph"))
        ) else "Excluded"
        tire_status = "Applied" if tire_active_change == "Absolute tire source" else "Pending"
    else:
        if current_tire_reference:
            if baseline_tire_reference and not scenario_current_reference and not any(
                ctx.get(key) not in (None, "", 0, 0.0) for key in ("tire_current_reference_A", "tire_current_reference_B", "tire_current_reference_C")
            ):
                tire_reference = "Inherited current"
            else:
                tire_reference = "Scenario current"
        elif baseline_mode:
            tire_reference = "Missing"
        else:
            tire_reference = "Optional"

        if tire_mode == "Keep inherited" or tire_application == "Keep inherited":
            tire_active_change = "Inherited"
            tire_status = "OK"
        elif tire_application == "Manual Delta RR":
            manual_delta_rr = float(to_float(ctx.get("tire_manual_delta_rr_n_per_kn"), 0.0) or 0.0)
            tire_active_change = f"Manual delta final RRC ({format_quantity(manual_delta_rr, 'rrc', format_str='%.2f')})"
            tire_status = "Applied" if abs(float(to_float(ctx.get("delta_rr_N"), 0.0) or 0.0)) > 0.0 else "Pending"
        elif tire_application == "Tire Improvement %":
            improvement_pct = float(to_float(ctx.get("tire_improvement_pct"), 0.0) or 0.0)
            tire_active_change = f"Tire Improvement ({improvement_pct:.1f}%)"
            if improvement_pct == 0.0:
                tire_status = "Pending"
            elif current_tire_reference:
                tire_status = "Applied"
            else:
                tire_status = "Pending current"
        else:
            tire_active_change = "Walked comparison"
            if walked_tire_reference and current_tire_reference and abs(float(to_float(ctx.get("delta_rr_N"), 0.0) or 0.0)) > 0.0:
                tire_status = "Applied"
            elif walked_tire_reference and not current_tire_reference:
                tire_status = "Pending current"
            elif not walked_tire_reference:
                tire_status = "Pending walked"
            else:
                tire_status = "Pending"

    rows = [
        {
            "Component": "Tires",
            "Reference": tire_reference,
            "Active change": tire_active_change,
            "Status": tire_status,
            "Action": "Edit below" if active == "Tires" else "Configure below",
        }
    ]

    component_specs = [
        ("Brakes", brake_mode, "delta_brake_N", "Absolute brake input"),
        ("Parasitics / Hubs / Axle", parasitic_mode, "delta_parasitics_N", "Absolute loss input"),
    ]
    for label, mode, delta_key, absolute_text in component_specs:
        if basis == "Component Build-up":
            reference = "Absolute source"
            if label == "Brakes":
                configured = any(abs(float(to_float(ctx.get(key), 0.0) or 0.0)) > 0.0 for key in ("brake_A_coef_N", "brake_B_Npkph", "brake_C_coef_Npkph2"))
            else:
                configured = any(abs(float(to_float(ctx.get(key), 0.0) or 0.0)) > 0.0 for key in ("parasitic_A_coef_N", "parasitic_B_Npkph", "parasitic_C_coef_Npkph2"))
            active_change = "Absolute value" if configured else "Excluded"
            status = "Applied" if configured else "Pending"
        else:
            if label == "Brakes":
                has_reference = any(to_float(selected_baseline_row.get(key)) is not None for key in ("brake_A_coef_N", "brake_B_Npkph", "brake_C_coef_Npkph2")) or not baseline_mode
                calc_mode = str(ctx.get("brake_calculation_mode") or "Inherited")
            else:
                has_reference = any(to_float(selected_baseline_row.get(key)) is not None for key in ("parasitic_A_coef_N", "parasitic_B_Npkph", "parasitic_C_coef_Npkph2")) or not baseline_mode
                calc_mode = str(ctx.get("parasitic_calculation_mode") or "Inherited")
            reference = "Inherited" if has_reference else "Missing"
            if calc_mode == "Use candidate vs reference":
                active_change = "Derived delta"
                status = "Applied"
            elif calc_mode == "Manual delta A" and abs(float(to_float(ctx.get(delta_key), 0.0) or 0.0)) > 0.0:
                active_change = "Manual delta"
                status = "Applied"
            elif mode == "Keep inherited":
                active_change = "Inherited"
                status = "OK"
            else:
                active_change = "Excluded"
                status = "Reference missing" if reference == "Missing" else "Pending"
        rows.append(
            {
                "Component": label,
                "Reference": reference,
                "Active change": active_change,
                "Status": status,
                "Action": "Edit below" if active == label else "Configure below",
            }
        )

    rows.append(
        {
            "Component": "Trailer",
            "Reference": "Placeholder",
            "Active change": "Excluded",
            "Status": "Placeholder",
            "Action": "Reserved",
        }
    )
    return rows


def _render_component_overview_table(ctx: dict):
    rows = _component_overview_rows(ctx)
    icon_map = {
        "Tires": "[T]",
        "Brakes": "[B]",
        "Parasitics / Hubs / Axle": "[P]",
        "Trailer": "[ ]",
    }
    st.caption("Components Overview")
    active = str(ctx.get("component_editor_active") or "Tires")
    for row in rows:
        component = str(row.get("Component") or "")
        icon = icon_map.get(component, "[?]")
        c1, c2, c3, c4 = st.columns([1.25, 1.15, 1.2, 0.9])
        is_active = component == active
        button_label = f"{icon} {component}"
        if c1.button(
            button_label,
            key=f"component_overview_open_{component}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            ctx["component_editor_active"] = component
            st.rerun()
        c2.caption("Reference")
        c2.write(str(row.get("Reference") or "-"))
        c3.caption("Active change")
        c3.write(str(row.get("Active change") or "-"))
        c4.caption("Status")
        c4.write(str(row.get("Status") or "-"))
    st.caption("Reference = current/base state. Active change = what is actually affecting the scenario now.")
    st.caption("Transmission stays outside this grid because it acts on the TOTAL -> NET bridge, but it should be read with the same discipline: clear reference, explicit application, and no silent effect on calculation.")


def _render_tire_editor_block_header(title: str, caption: str):
    st.markdown(f"**{title}**")
    st.caption(caption)


def _render_applied_effect_block(*, title: str, caption: str, inherited_message: str, metrics: list[tuple[str, str]] | None = None):
    with st.container(border=True):
        _render_tire_editor_block_header(title, caption)
        if not metrics:
            st.info(inherited_message)
            return
        cols = st.columns(len(metrics))
        for col, (label, value) in zip(cols, metrics):
            col.metric(label, value)


def _render_scalar_reference_metric(*, label: str, value, quantity: str, status: str):
    st.caption(f"Reference status: `{status}`")
    quantity_metric(st, label, value, quantity, format_str="%.4f")


def _render_abc_reference_metrics(*, title: str, a_value, b_value, c_value, status: str, caption: str | None = None):
    st.caption(f"{title} | status: `{status}`")
    c1, c2, c3 = st.columns(3)
    quantity_metric(c1, "A", a_value, "force", format_str="%.3f")
    quantity_metric(c2, "B", b_value, "force_per_speed", format_str="%.6f")
    quantity_metric(c3, "C", c_value, "force_per_speed_squared", format_str="%.8f")
    if caption:
        st.caption(caption)


def _render_candidate_cda_input(*, key_prefix: str, title: str) -> float | None:
    ctx = st.session_state.ctx
    st.caption(title)
    value = quantity_input(
        st,
        "Candidate CdA",
        to_float(ctx.get(f"{key_prefix}_cda"), 0.0),
        "cda",
        key=f"{key_prefix}_cda_input",
        step_canonical=0.01,
        format_str="%.3f",
    )
    ctx[f"{key_prefix}_cda"] = value
    if abs(float(value)) <= 0.0:
        st.caption("Candidate still pending. Enter a proposed CdA to stage it.")
        return None
    return float(value)


def _render_candidate_abc_inputs(*, key_prefix: str, title: str) -> dict | None:
    ctx = st.session_state.ctx
    st.caption(title)
    c1, c2, c3 = st.columns(3)
    a_val = quantity_input(
        c1,
        "Candidate A",
        to_float(ctx.get(f"{key_prefix}_A"), 0.0),
        "force",
        key=f"{key_prefix}_A_input",
        step_canonical=0.1,
        format_str="%.2f",
    )
    b_val = quantity_input(
        c2,
        "Candidate B",
        to_float(ctx.get(f"{key_prefix}_B"), 0.0),
        "force_per_speed",
        key=f"{key_prefix}_B_input",
        step_canonical=0.001,
        format_str="%.5f",
    )
    c_val = quantity_input(
        c3,
        "Candidate C",
        to_float(ctx.get(f"{key_prefix}_C"), 0.0),
        "force_per_speed_squared",
        key=f"{key_prefix}_C_input",
        step_canonical=0.0001,
        format_str="%.6f",
    )
    ctx[f"{key_prefix}_A"] = a_val
    ctx[f"{key_prefix}_B"] = b_val
    ctx[f"{key_prefix}_C"] = c_val
    if abs(float(a_val)) <= 0.0 and abs(float(b_val)) <= 0.0 and abs(float(c_val)) <= 0.0:
        st.caption("Candidate still pending. Enter at least one non-zero ABC value to stage it.")
        return None
    return {"A": float(a_val), "B": float(b_val), "C": float(c_val)}


def _transmission_component_row(ctx: dict, prefill: dict | None = None) -> dict:
    base = dict(prefill or {})
    source = str(ctx.get("transmission_losses_source") or ("Baseline" if base.get("trans_A_coef_N") is not None else "Missing")).strip().title()
    if source == "Baseline":
        reference = "Inherited"
        calc_now = "Applied to NET"
        status = "Applied"
    elif source == "Manual":
        reference = "Manual"
        calc_now = "Applied to NET"
        status = "Applied" if any(abs(float(to_float(ctx.get(key), 0.0) or 0.0)) > 0.0 for key in ("trans_A_coef_N", "trans_B_coef_Npkph", "trans_C_coef_Npkph2")) else "Pending"
    else:
        reference = "Missing"
        calc_now = "Excluded"
        status = "Pending"
    return {
        "Component": "Transmission / Neutral Drag",
        "Reference": reference,
        "Candidate": "N/A",
        "Calculation now": calc_now,
        "Status": status,
    }


def _safe_workflow_preview(ctx: dict | None = None) -> dict:
    data = dict(ctx or st.session_state.get("ctx", {}))
    try:
        return dict(build_vde_setup_preview_from_ctx(data) or {})
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _line_source_summary(ctx: dict) -> str:
    mode = str(ctx.get("mode") or "").strip()
    if mode == "From baseline (editable)":
        baseline_id = ctx.get("vde_id_parent") or ctx.get("baseline_id")
        return f"Baseline #{baseline_id}" if baseline_id else "Baseline flow"
    if mode == "New line (manual / test)":
        return "New line"
    return "-"


def _total_source_summary(ctx: dict, preview: dict | None = None) -> str:
    data = dict(preview or {})
    line_source = dict(data.get("line_source") or {})
    source_ui = str(ctx.get("abc_total_source_ui") or "").strip()
    mode = str(ctx.get("mode") or "").strip()
    from_delta = str(ctx.get("from_delta") or "").strip()
    if source_ui:
        return source_ui
    if mode == "From baseline (editable)" and from_delta == "Deltas":
        baseline_id = ctx.get("vde_id_parent") or ctx.get("baseline_id")
        return f"Baseline ABC_TOTAL #{baseline_id}" if baseline_id else "Baseline ABC_TOTAL"
    if _component_build_up_enabled(ctx):
        return "Component Build-up"
    return str(line_source.get("mode") or "Manual inputs")


def _inertia_class_table() -> list[dict]:
    csv_path = Path(__file__).resolve().parents[3] / "data" / "standards" / "inertia_classes_by_mass.csv"
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []
    rows = []
    for row in df.to_dict(orient="records"):
        rows.append(
            {
                "mass_min_kg_exclusive": to_float(row.get("mass_min_kg_exclusive")),
                "mass_max_kg_inclusive": to_float(row.get("mass_max_kg_inclusive")),
                "inertia_class_kg": to_float(row.get("inertia_class_kg")),
            }
        )
    return rows


def _inertia_step_for_mass(mass_kg: float | None) -> tuple[int | None, dict | None]:
    if mass_kg is None:
        return None, None
    rows = _inertia_class_table()
    for idx, row in enumerate(rows):
        lo = row.get("mass_min_kg_exclusive")
        hi = row.get("mass_max_kg_inclusive")
        if lo is None and hi is not None and mass_kg <= hi:
            return idx, row
        if hi is None and lo is not None and mass_kg > lo:
            return idx, row
        if lo is not None and hi is not None and mass_kg > lo and mass_kg <= hi:
            return idx, row
    return None, None


def _mass_point_for_inertia_step(step: dict | None, point: str) -> float | None:
    if not step:
        return None
    lo = step.get("mass_min_kg_exclusive")
    hi = step.get("mass_max_kg_inclusive")
    if hi is None:
        return None
    low = float(lo + 1.0) if lo is not None else float(hi)
    top = float(hi)
    mid = float(round((low + top) / 2.0, 1))
    if point == "low":
        return low
    if point == "mid":
        return mid
    if point == "top":
        return top
    return None


def _relative_twc_mass_targets(baseline_mass_kg: float | None) -> list[dict]:
    idx, _ = _inertia_step_for_mass(baseline_mass_kg)
    rows = _inertia_class_table()
    if idx is None or not rows:
        return []
    targets = []
    for offset, prefix in [(-1, "-1"), (0, "Current"), (1, "+1")]:
        target_idx = idx + offset
        if target_idx < 0 or target_idx >= len(rows):
            continue
        step = rows[target_idx]
        twc_value = step.get("inertia_class_kg")
        for point, suffix in [("low", "Low"), ("mid", "Mid"), ("top", "Top")]:
            target_mass = _mass_point_for_inertia_step(step, point)
            if target_mass is None:
                continue
            targets.append(
                {
                    "label": f"{prefix} TWC {suffix}",
                    "mass_kg": float(target_mass),
                    "twc_kg": float(twc_value) if twc_value is not None else None,
                }
            )
    return targets


def _relative_twc_target(baseline_mass_kg: float | None, class_offset: int, point: str) -> dict | None:
    idx, _ = _inertia_step_for_mass(baseline_mass_kg)
    rows = _inertia_class_table()
    if idx is None or not rows:
        return None
    target_idx = idx + int(class_offset or 0)
    if target_idx < 0 or target_idx >= len(rows):
        return None
    step = rows[target_idx]
    target_mass = _mass_point_for_inertia_step(step, point)
    if target_mass is None:
        return None
    prefix = "Current" if class_offset == 0 else f"{class_offset:+d}"
    return {
        "label": f"{prefix} TWC {str(point).title()}",
        "mass_kg": float(target_mass),
        "twc_kg": float(step.get("inertia_class_kg")) if step.get("inertia_class_kg") is not None else None,
        "target_index": target_idx,
    }


def _mass_setup_summary(preview: dict | None = None, ctx: dict | None = None) -> str:
    data = dict(preview or {})
    mass_setup = dict(data.get("mass_setup") or {})
    source_ctx = dict(ctx or st.session_state.get("ctx", {}))
    basis = str(mass_setup.get("mass_basis") or source_ctx.get("tire_load_mass_basis") or "TEST_MASS")
    resolved = to_float(mass_setup.get("resolved_mass_used_kg"), to_float(source_ctx.get("mass_kg")))
    if resolved is None:
        return basis
    return f"{basis} | {format_quantity(resolved, 'mass', include_unit=True, format_str='%.0f')}"


def _abc_unit_triplet_label() -> str:
    return (
        f"{unit_label('force')} / "
        f"{unit_label('force_per_speed')} / "
        f"{unit_label('force_per_speed_squared')}"
    )


def _transmission_summary(preview: dict | None = None, ctx: dict | None = None) -> str:
    data = dict(preview or {})
    transmission = dict(data.get("transmission_losses") or {})
    source_ctx = dict(ctx or st.session_state.get("ctx", {}))
    status = str(transmission.get("status") or "").strip().lower()
    source = str(source_ctx.get("transmission_losses_source") or "Missing").strip()
    if status == "available":
        return f"{source} | NET available"
    return f"{source} | NET pending"


def _summary_status_payload(state: str) -> dict:
    normalized = str(state or "pending").strip().lower()
    if normalized == "ok":
        return {"label": "OK", "class_name": "is-ok", "icon_html": "&#10003;"}
    if normalized == "warn":
        return {"label": "Check", "class_name": "is-warn", "icon_html": "&#33;"}
    return {"label": "Pending", "class_name": "is-pending", "icon_html": "&#9679;"}


def _overview_icon(label: str) -> str:
    icons = {
        "Vehicle Data": "&#128663;",
        "Scenario Origin": "&#9873;",
        "ABC_TOTAL Basis": "&#8776;",
        "Mass": "&#9878;",
        "Transmission": "&#8644;",
        "Cycle": "&#10227;",
        "VDE_TOTAL": "&#9312;",
        "VDE_NET": "&#9313;",
    }
    return icons.get(label, "&#9679;")


def _normalize_tire_scenario_application(value: str | None) -> str:
    current = str(value or "").strip()
    mapping = {
        "Apply manual delta A": "Manual Delta RR",
        "Apply derived comparison": "Walked tire comparison",
        "Apply Tire Improvement %": "Tire Improvement %",
        "Stage candidate only / do not apply": "Keep inherited",
    }
    normalized = mapping.get(current, current)
    if normalized not in {
        "Keep inherited",
        "Manual Delta RR",
        "Tire Improvement %",
        "Walked tire comparison",
    }:
        return "Keep inherited"
    return normalized


def _display_tire_application_method(value: str | None) -> str:
    current = str(value or "").strip()
    mapping = {
        "Manual Delta RR": "Manual delta final RRC",
        "Tire Improvement %": "Tire Improvement %",
        "Walked tire comparison": "Walked tire comparison",
        "Keep inherited": "Inherited",
    }
    return mapping.get(current, current or "Inherited")


def _display_data_role(value: str | None) -> str:
    current = str(value or "").strip()
    mapping = {
        "engineering_target": "Engineering target",
        "scenario_assumption": "Scenario assumption",
    }
    return mapping.get(current, current or "-")


def _tire_reference_brief(reference: dict | None) -> str:
    data = dict(reference or {})
    identity = " ".join(
        part for part in (
            _string_or_none(data.get("manufacturer")),
            _string_or_none(data.get("model")),
        ) if part
    ) or _string_or_none(data.get("tire_test_code")) or "Scenario reference"
    size_code = _string_or_none(data.get("size_code"))
    rr_text = format_quantity(data.get("rr_n_per_kn"), "rrc", unavailable="Pending", format_str="%.3f")
    if size_code:
        return f"{identity} | {size_code} | {rr_text}"
    return f"{identity} | {rr_text}"


def _default_tire_change_method(ctx: dict) -> str:
    explicit = str(ctx.get("tire_change_method") or "").strip()
    if explicit in {"Manual tire adjustment", "Walked tire comparison"}:
        return explicit
    application = _normalize_tire_scenario_application(ctx.get("tire_scenario_application"))
    if application == "Walked tire comparison":
        return "Walked tire comparison"
    return "Manual tire adjustment"


def _default_tire_manual_input_type(ctx: dict) -> str:
    explicit = str(ctx.get("tire_manual_adjustment_input_type") or "").strip()
    if explicit in {"Delta RR", "Delta final RRC"}:
        return "Delta final RRC"
    if explicit in {"Target RRC", "Target final RRC"}:
        return "Target final RRC"
    if explicit == "Tire Improvement %":
        return explicit
    application = _normalize_tire_scenario_application(ctx.get("tire_scenario_application"))
    if application == "Tire Improvement %":
        return "Tire Improvement %"
    return "Delta final RRC"


def _resolved_tire_load_kN(ctx: dict) -> float:
    tire_mass_resolution = resolve_tire_calculation_mass(ctx)
    calc_mass_kg = to_float(tire_mass_resolution.get("mass_kg"))
    if calc_mass_kg is None:
        calc_mass_kg = to_float(ctx.get("mass_kg"), 0.0)
    return (float(calc_mass_kg or 0.0) * 9.80665) / 1000.0


def _resolve_tire_reference_from_ctx(
    prefix: str,
    *,
    source_label: str,
    preview_ctx_key: str | None = None,
) -> dict | None:
    ctx = st.session_state.ctx
    preview_key = preview_ctx_key or f"{prefix}_preview_result"
    preview_result = ctx.get(preview_key)
    if isinstance(preview_result, dict):
        reference = _reference_from_preview_result(preview_result, source_label=source_label)
        if reference:
            return reference
    return _manual_tire_reference_from_ctx(prefix, source_label=source_label)


def _ensure_vehicle_metadata_defaults(ctx: dict):
    leg_opts = ["WLTP", "EPA", "ABNT (Brazil)"]
    if ctx.get("legislation") not in leg_opts:
        ctx["legislation"] = "WLTP"

    epa_classes = [
        "Unknown", "Two Seaters", "Minicompact Cars", "Subcompact Cars", "Compact Cars",
        "Midsize Cars", "Large Cars", "Small Station Wagons", "Midsize Station Wagons",
        "Small SUVs", "Standard SUVs", "Minivans", "Vans", "Small Pickup Trucks", "Standard Pickup Trucks",
    ]
    wltp_classes = ["Class 1 (<850 kg)", "Class 2 (850-1220 kg)", "Class 3 (>1220 kg)"]
    category_list = epa_classes if ctx["legislation"] == "EPA" else wltp_classes
    category_list_upper = [category.upper() for category in category_list]
    if ctx.get("category") not in category_list_upper:
        ctx["category"] = category_list_upper[0]

    if not str(ctx.get("electrification") or "").strip():
        ctx["electrification"] = "ICE"
    if not str(ctx.get("transmission_type") or "").strip():
        ctx["transmission_type"] = "AT"
    if not str(ctx.get("year") or "").strip().isdigit():
        ctx["year"] = 2024


def _metadata_status(ctx: dict) -> dict:
    legislation = str(ctx.get("legislation") or "").strip()
    category = str(ctx.get("category") or "").strip()
    make = str(ctx.get("make") or "").strip()
    model = str(ctx.get("model") or "").strip()
    electrification = str(ctx.get("electrification") or "").strip()
    transmission_type = str(ctx.get("transmission_type") or "").strip()
    year_raw = ctx.get("year")

    missing = []
    if not legislation:
        missing.append("legislation")
    if not category or category.upper() == "UNKNOWN":
        missing.append("category")
    if not make:
        missing.append("make")
    if not model:
        missing.append("model")
    if not str(year_raw).strip().isdigit():
        missing.append("year")
    if not electrification:
        missing.append("electrification")
    if not transmission_type:
        missing.append("transmission")

    year_text = str(int(year_raw)) if str(year_raw).strip().isdigit() else "Year pending"
    headline_parts = [part for part in (make.upper(), model) if part]
    headline = " ".join(headline_parts).strip() or "Vehicle data pending"
    value = f"{headline} | {year_text}"
    detail = f"{legislation} | {category} | {electrification or '-'} | {transmission_type or '-'}"

    if missing:
        missing_text = ", ".join(missing[:3])
        if len(missing) > 3:
            missing_text += ", ..."
        return {
            "status": "pending",
            "value": value,
            "detail": f"{detail} | Missing: {missing_text}",
            "missing": missing,
        }

    return {
        "status": "ok",
        "value": value,
        "detail": detail,
        "missing": [],
    }


def _line_source_status(ctx: dict) -> str:
    mode = str(ctx.get("mode") or "").strip()
    if mode == "From baseline (editable)":
        return "ok" if (ctx.get("vde_id_parent") or ctx.get("baseline_id")) else "pending"
    if mode == "New line (manual / test)":
        return "ok"
    return "pending"


def _total_source_status(ctx: dict) -> str:
    source_ui = str(ctx.get("abc_total_source_ui") or "").strip()
    if source_ui == "Baseline ABC":
        return "ok" if (ctx.get("vde_id_parent") or ctx.get("baseline_id")) else "pending"
    if source_ui == "From test coastdown":
        errs, _ = validate_core(ctx.get("A"), ctx.get("B"), ctx.get("C"), ctx.get("mass_kg"))
        return "ok" if not errs else "pending"
    if source_ui == "Component Build-up":
        return "ok"
    return "pending"


def _mass_setup_status(preview: dict | None = None, ctx: dict | None = None) -> str:
    data = dict(preview or {})
    source_ctx = dict(ctx or st.session_state.get("ctx", {}))
    mass_setup = dict(data.get("mass_setup") or {})
    resolved_mass = to_float(mass_setup.get("resolved_mass_used_kg"), to_float(source_ctx.get("mass_kg")))
    weight_dist = to_float(source_ctx.get("weight_dist_fr_pct"))
    if resolved_mass is None or resolved_mass <= 0:
        return "pending"
    if weight_dist is None or weight_dist < 0 or weight_dist > 100:
        return "pending"
    return "ok"


def _cycle_status(ctx: dict) -> str:
    return "ok" if (ctx.get("cycle_df") is not None or str(ctx.get("cycle_name") or "").strip()) else "pending"


def _energy_status(value) -> str:
    return "ok" if value is not None else "pending"


def _summary_action_href(view: str, technical_view: str | None = None) -> str:
    parts = [f"vde_setup_view_target={quote_plus(view)}"]
    if technical_view:
        parts.append(f"technical_build_up_view_target={quote_plus(technical_view)}")
    return "?" + "&".join(parts)


def apply_summary_navigation_from_query_params():
    params = st.query_params
    target_view = params.get("vde_setup_view_target")
    target_technical = params.get("technical_build_up_view_target")

    if isinstance(target_view, list):
        target_view = target_view[0] if target_view else None
    if isinstance(target_technical, list):
        target_technical = target_technical[0] if target_technical else None

    target_view = str(target_view or "").strip()
    target_technical = str(target_technical or "").strip()
    if target_view == "Cycle & Results":
        target_view = "Results"
    if target_view == "Drive Cycle":
        target_view = "Cycle & Preview"
    if target_view == "Technical Build-up":
        target_view = "Roadload Build-up"
    if not target_view:
        return

    valid_views = {"Scenario Setup", "Vehicle Parameters", "Roadload Build-up", "Cycle & Preview", "Results", "Save / Edit"}
    valid_technical = {"Tires", "Brakes", "Parasitics / Hubs / Axle", "Trailer", "Transmission"}

    ctx = st.session_state.ctx
    if target_view in valid_views:
        ctx["vde_setup_view"] = target_view
        st.session_state["vde_setup_view_selector"] = target_view
    if target_technical in valid_technical:
        ctx["technical_build_up_view"] = target_technical
        st.session_state["technical_build_up_view_selector"] = target_technical

    for key in ("vde_setup_view_target", "technical_build_up_view_target"):
        try:
            del st.query_params[key]
        except Exception:
            pass


def _render_summary_chip(
    label: str,
    value: str,
    *,
    status: str = "pending",
    detail: str | None = None,
    action_view: str | None = None,
    action_technical_view: str | None = None,
):
    status_payload = _summary_status_payload(status)
    overview_icon = _overview_icon(label)
    detail_html = f"<div class='vde-summary-chip-detail'>{detail}</div>" if detail else ""
    open_link = ""
    close_link = ""
    if action_view:
        href = _summary_action_href(action_view, action_technical_view)
        open_link = f'<a class="vde-summary-link" href="{href}" target="_self">'
        close_link = "</a>"
    st.markdown(
        f"""
        {open_link}
        <div class="vde-summary-chip {status_payload['class_name']}">
            <div class="vde-summary-chip-top">
                <strong><span class="vde-summary-status-icon">{overview_icon}</span> {label}</strong>
                <span class="vde-summary-status {status_payload['class_name']}">
                    <span class="vde-summary-status-icon">{status_payload['icon_html']}</span>
                    <span>{status_payload['label']}</span>
                </span>
            </div>
            <span>{value}</span>
            {detail_html}
        </div>
        {close_link}
        """,
        unsafe_allow_html=True,
    )


def _render_component_mode_selector(component_label: str) -> str:
    ctx = st.session_state.ctx
    basis = _roadload_basis_value(ctx)
    options, _ = _component_mode_options(component_label, basis)
    current = _component_mode_default(component_label, basis, ctx)
    key = _component_mode_key(component_label)
    if len(options) == 1:
        ctx[key] = options[0]
        st.caption(f"{component_label} mode: `{options[0]}`")
        return ctx[key]
    ctx[key] = st.radio(
        f"{component_label} mode",
        options,
        horizontal=True,
        index=options.index(current),
        key=f"{key}_radio",
    )
    return ctx[key]


def render_vde_setup_view_selector() -> str:
    ctx = st.session_state.ctx
    options = ["Scenario Setup", "Vehicle Parameters", "Roadload Build-up", "Cycle & Preview", "Results", "Save / Edit"]
    current = str(ctx.get("vde_setup_view") or "Scenario Setup")
    if current == "Cycle & Results":
        current = "Results"
    if current == "Drive Cycle":
        current = "Cycle & Preview"
    if current == "Technical Build-up":
        current = "Roadload Build-up"
    if current not in options:
        current = "Scenario Setup"
    ctx["vde_setup_view"] = st.radio(
        "View",
        options,
        horizontal=True,
        index=options.index(current),
        key="vde_setup_view_selector",
    )
    return ctx["vde_setup_view"]


def render_technical_build_up_view_selector() -> str:
    ctx = st.session_state.ctx
    options = ["Tires", "Brakes", "Parasitics / Hubs / Axle", "Trailer", "Transmission"]
    current = str(ctx.get("technical_build_up_view") or "Tires")
    if current in {"Mass & Axle Load", "Components"}:
        current = "Tires"
    if current not in options:
        current = "Tires"
    ctx["technical_build_up_view"] = st.radio(
        "Roadload section",
        options,
        horizontal=True,
        index=options.index(current),
        key="technical_build_up_view_selector",
    )
    return ctx["technical_build_up_view"]


def render_initial_abc_total_source_section(*, defaults_df_getter=None):
    ctx = st.session_state.ctx
    mode = str(ctx.get("mode") or "From baseline (editable)")
    options = (
        ["Baseline ABC", "From test coastdown", "Component Build-up"]
        if mode == "From baseline (editable)"
        else ["From test coastdown", "Component Build-up"]
    )
    default_value = str(ctx.get("abc_total_source_ui") or "").strip()
    if default_value not in options:
        default_value = options[0] if mode == "From baseline (editable)" else "Component Build-up"

    ctx["abc_total_source_ui"] = st.radio(
        "ABC_TOTAL basis",
        options,
        horizontal=True,
        index=options.index(default_value),
        key="abc_total_source_ui_radio",
        format_func=lambda value: {
            "Baseline ABC": "Inherit baseline ABC_TOTAL",
            "From test coastdown": "Measured/test coastdown ABC_TOTAL",
            "Component Build-up": "Build/synthesize ABC_TOTAL from components",
        }.get(value, value),
    )

    selected = ctx["abc_total_source_ui"]
    if selected == "Baseline ABC":
        ctx["from_delta"] = "Deltas"
        baseline_id = ctx.get("vde_id_parent") or ctx.get("baseline_id")
        if not baseline_id:
            st.warning("Select a baseline scenario in Scenario Setup before using the inherited baseline ABC_TOTAL basis.")
    elif selected == "From test coastdown":
        ctx["from_delta"] = "From test"
        render_from_test_section()
        render_auxiliaries_section(defaults_df_getter=defaults_df_getter)
    else:
        ctx["from_delta"] = "Change Parameters"


def render_mass_setup_section(*, prefill=None):
    ctx = st.session_state.ctx
    base = dict(prefill or {})
    mode = str(ctx.get("mode") or "")
    abc_total_source_ui = str(ctx.get("abc_total_source_ui") or "")
    baseline_inherited_mode = mode == "From baseline (editable)" and abc_total_source_ui == "Baseline ABC"

    mass_default = to_float(base.get("mass_kg"), to_float(ctx.get("mass_kg"), 1550.0))
    test_mass_prefill = to_float(base.get("test_mass_kg"), to_float(ctx.get("test_mass_kg")))
    weight_dist_default = to_float(base.get("weight_dist_fr_pct"), to_float(ctx.get("weight_dist_fr_pct"), 50.0))
    baseline_mass_reference = to_float(base.get("mass_kg"), mass_default)
    existing_delta_mass = float(to_float(ctx.get("delta_mass_kg"), 0.0) or 0.0)
    legislation = str(ctx.get("legislation") or "").strip().upper()
    scenario_mass_for_calcs = to_float(ctx.get("mass_kg"), mass_default)
    force_twc_basis = False

    if baseline_inherited_mode and baseline_mass_reference is not None:
        baseline_test_mass = resolve_test_mass_kg(
            {
                **ctx,
                "mass_kg": baseline_mass_reference,
                "test_mass_kg": to_float(base.get("test_mass_kg")),
            }
        )
        baseline_twc = resolve_tire_calculation_mass(
            {
                **ctx,
                "mass_kg": baseline_mass_reference,
                "test_mass_kg": to_float(base.get("test_mass_kg")),
                "tire_load_mass_basis": "TWC",
                "twc_kg": to_float(base.get("twc_kg")),
                "etw_kg": to_float(base.get("etw_kg")),
                "inertia_class": to_float(base.get("inertia_class")),
            }
        ).get("mass_kg")

        ctx["mass_kg"] = float(baseline_mass_reference)
        base1, base2, base3 = st.columns(3)
        quantity_metric(base1, "Baseline curb mass", baseline_mass_reference, "mass", format_str="%.1f")
        quantity_metric(base2, "Baseline test mass", baseline_test_mass, "mass", format_str="%.1f")
        quantity_metric(base3, "Baseline TWC", baseline_twc, "mass", format_str="%.1f")

        st.caption("Baseline mass snapshot is read-only here. Choose how this scenario changes mass relative to the inherited baseline.")

        mass_change_mode_options = ["Input new Curb / Test Mass", "Delta"]
        selected_mass_change_mode = str(ctx.get("mass_change_mode") or mass_change_mode_options[0])
        if selected_mass_change_mode not in mass_change_mode_options:
            selected_mass_change_mode = mass_change_mode_options[0]
        ctx["mass_change_mode"] = st.radio(
            "Mass scenario treatment",
            mass_change_mode_options,
            horizontal=True,
            index=mass_change_mode_options.index(selected_mass_change_mode),
            key="mass_change_mode_radio",
        )

        if ctx["mass_change_mode"] == "Input new Curb / Test Mass":
            new_curb_mass = quantity_input(
                st,
                "New curb weight",
                to_float(ctx.get("mass_walked_reference_kg"), scenario_mass_for_calcs),
                "mass",
                key="mass_setup_new_curb_weight",
                min_canonical=300.0,
                max_canonical=5000.0,
                step_canonical=1.0,
                format_str="%.1f",
            )
            scenario_mass_for_calcs = float(new_curb_mass)
            ctx["delta_mass_kg"] = float(scenario_mass_for_calcs) - float(baseline_mass_reference)

            tm1, tm2 = st.columns([1.2, 1.2])
            default_new_test_mass = resolve_test_mass_kg(
                {
                    **ctx,
                    "mass_kg": scenario_mass_for_calcs,
                    "test_mass_kg": None,
                }
            )
            use_default = tm1.checkbox("Use default test mass", value=bool(ctx.get("test_mass_use_default", True)), key="mass_setup_inherited_use_default_test_mass")
            ctx["test_mass_use_default"] = use_default
            if use_default:
                ctx["test_mass_kg"] = None
                quantity_metric(tm2, "Scenario test mass", default_new_test_mass, "mass", format_str="%.1f")
            else:
                scenario_test_mass = quantity_input(
                    tm2,
                    "New test mass",
                    max(to_float(ctx.get("test_mass_kg"), default_new_test_mass or scenario_mass_for_calcs), scenario_mass_for_calcs),
                    "mass",
                    key="mass_setup_new_test_mass",
                    min_canonical=float(scenario_mass_for_calcs),
                    max_canonical=5000.0,
                    step_canonical=1.0,
                    format_str="%.1f",
                )
                ctx["test_mass_kg"] = scenario_test_mass
        else:
            delta_mode_options = ["TWC class target", "Manual delta"]
            selected_delta_mode = str(ctx.get("mass_delta_mode") or delta_mode_options[0])
            if selected_delta_mode not in delta_mode_options:
                selected_delta_mode = delta_mode_options[0]
            ctx["mass_delta_mode"] = st.radio(
                "Delta mode",
                delta_mode_options,
                horizontal=True,
                index=delta_mode_options.index(selected_delta_mode),
                key="mass_delta_mode_radio",
            )

            if ctx["mass_delta_mode"] == "TWC class target":
                base_idx, _ = _inertia_step_for_mass(baseline_mass_reference)
                rows = _inertia_class_table()
                if base_idx is not None and rows:
                    min_offset = -base_idx
                    max_offset = len(rows) - 1 - base_idx
                    offset_col, point_col = st.columns([1, 1.2])
                    class_offset = int(
                        offset_col.number_input(
                            "TWC class offset",
                            min_value=int(min_offset),
                            max_value=int(max_offset),
                            value=int(ctx.get("mass_delta_class_offset", 0) or 0),
                            step=1,
                            key="mass_delta_class_offset_input",
                        )
                    )
                    point_options = ["Low", "Mid", "Top"]
                    selected_point = str(ctx.get("mass_delta_class_point") or "Mid").title()
                    if selected_point not in point_options:
                        selected_point = "Mid"
                    point_value = point_col.selectbox(
                        "Target within class",
                        point_options,
                        index=point_options.index(selected_point),
                        key="mass_delta_class_point_select",
                    )
                    ctx["mass_delta_class_offset"] = class_offset
                    ctx["mass_delta_class_point"] = point_value
                    selected_target = _relative_twc_target(baseline_mass_reference, class_offset, point_value.lower())
                    if selected_target:
                        force_twc_basis = True
                        scenario_mass_for_calcs = float(selected_target["mass_kg"])
                        ctx["delta_mass_kg"] = scenario_mass_for_calcs - float(baseline_mass_reference)
                        ctx["test_mass_kg"] = None
                        ctx["test_mass_use_default"] = True
                        ctx["twc_kg"] = selected_target.get("twc_kg")
                        st.caption(f"Resolved target: `{selected_target['label']}`")
                        adopted_test_mass = resolve_test_mass_kg(
                            {
                                **ctx,
                                "mass_kg": scenario_mass_for_calcs,
                                "test_mass_kg": None,
                            }
                        )
                        adopted1, adopted2, adopted3 = st.columns(3)
                        quantity_metric(adopted1, "Target TWC", selected_target.get("twc_kg"), "mass", format_str="%.1f")
                        quantity_metric(adopted2, "Adopted curb mass", scenario_mass_for_calcs, "mass", format_str="%.1f")
                        quantity_metric(adopted3, "Adopted test mass", adopted_test_mass, "mass", format_str="%.1f")
                        st.caption("This delta-class path auto-adopts the selected TWC target and keeps the VDE calculation mass basis on TWC.")
                    else:
                        st.warning("Could not resolve the selected TWC class target.")
                else:
                    st.warning("Could not resolve TWC class targets from the inertia-class table.")
                    scenario_mass_for_calcs = float(baseline_mass_reference + existing_delta_mass)
            else:
                manual_delta_mass = quantity_input(
                    st,
                    "Delta Mass",
                    existing_delta_mass,
                    "mass",
                    key="mass_setup_delta_mass_kg",
                    step_canonical=1.0,
                    format_str="%.1f",
                )
                ctx["delta_mass_kg"] = float(manual_delta_mass)
                scenario_mass_for_calcs = float(baseline_mass_reference) + float(manual_delta_mass)
                ctx["test_mass_kg"] = None
                ctx["test_mass_use_default"] = True

        ctx["mass_walked_reference_kg"] = scenario_mass_for_calcs
    else:
        ctx["mass_kg"] = quantity_input(
            st,
            "Curb weight",
            to_float(mass_default, 1550.0),
            "mass",
            key="mass_setup_curb_weight",
            min_canonical=300.0,
            max_canonical=5000.0,
            step_canonical=1.0,
            format_str="%.1f",
        )
        scenario_mass_for_calcs = float(ctx["mass_kg"])
        ctx["delta_mass_kg"] = float(to_float(ctx.get("delta_mass_kg"), 0.0) or 0.0)

    row1c1, row1c2, row1c3 = st.columns(3)
    row1c1.metric("Legislation", str(ctx.get("legislation") or "-"))
    ctx["weight_dist_fr_pct"] = row1c2.number_input(
        "Front weight distribution [%]",
        min_value=0.0,
        max_value=100.0,
        value=float(weight_dist_default or 50.0),
        step=0.5,
        format="%.1f",
        key="mass_setup_weight_dist",
    )

    current_basis = resolve_tire_load_mass_basis(
        {
            "legislation": ctx.get("legislation"),
            "tire_load_mass_basis": ctx.get("tire_load_mass_basis") or base.get("tire_load_mass_basis"),
        }
    )
    if legislation == "EPA":
        if force_twc_basis:
            ctx["tire_load_mass_basis"] = "TWC"
            row1c3.metric("VDE calculation mass", "TWC")
        else:
            ctx["tire_load_mass_basis"] = row1c3.selectbox(
                "VDE calculation mass",
                ["TWC", "TEST_MASS"],
                index=["TWC", "TEST_MASS"].index(current_basis if current_basis in {"TWC", "TEST_MASS"} else "TWC"),
                key="mass_setup_tire_basis",
            )
    else:
        ctx["tire_load_mass_basis"] = "TEST_MASS"
        row1c3.metric("VDE calculation mass", "TEST_MASS")

    test_mass_default = resolve_test_mass_kg({**ctx, "mass_kg": scenario_mass_for_calcs, "test_mass_kg": None})
    if not baseline_inherited_mode:
        saved_use_default = bool(ctx.get("test_mass_use_default", True))
        if test_mass_prefill is not None and test_mass_default is not None and abs(test_mass_prefill - test_mass_default) > 1e-9:
            saved_use_default = False

        row2c1, row2c2, row2c3 = st.columns([1, 1.4, 1.4])
        use_default = row2c1.checkbox("Use default test mass", value=saved_use_default, key="mass_setup_use_default_test_mass")
        ctx["test_mass_use_default"] = use_default
        if use_default:
            ctx["test_mass_kg"] = None
            quantity_metric(row2c2, "Test mass", test_mass_default, "mass", format_str="%.1f")
        else:
            ctx["test_mass_kg"] = quantity_input(
                row2c2,
                "Test mass",
                max(test_mass_prefill if test_mass_prefill is not None else (test_mass_default or scenario_mass_for_calcs or 0.0), scenario_mass_for_calcs or 0.0),
                "mass",
                key="mass_setup_manual_test_mass",
                min_canonical=float(scenario_mass_for_calcs or 0.0),
                max_canonical=5000.0,
                step_canonical=1.0,
                format_str="%.1f",
            )

        hint = build_test_mass_hint(ctx)
        if hint:
            row2c3.caption(hint)
    else:
        st.caption(build_test_mass_hint(ctx))

    tire_mass_resolution = resolve_tire_calculation_mass({**ctx, "mass_kg": scenario_mass_for_calcs})
    calc_mass_kg = tire_mass_resolution.get("mass_kg")
    if legislation == "EPA" and ctx.get("tire_load_mass_basis") == "TWC":
        ctx["inertia_class"] = calc_mass_kg
        ctx["twc_kg"] = calc_mass_kg

    row3c1, row3c2, row3c3 = st.columns(3)
    quantity_metric(row3c1, "Resolved calc mass", calc_mass_kg, "mass", format_str="%.1f")
    row3c2.metric("Mass basis", str(ctx.get("tire_load_mass_basis") or "TEST_MASS"))
    row3c3.metric("Weight distribution", f"{float(ctx.get('weight_dist_fr_pct') or 50.0):.1f}%")
    st.caption("Mass setup is centralized here so tire, preview, and transmission sections can reuse the same resolved vehicle state.")


def render_executive_summary_panel():
    ctx = st.session_state.ctx
    _ensure_vehicle_metadata_defaults(ctx)
    preview = _safe_workflow_preview(ctx)
    vde_total = dict(preview.get("vde_total") or {})
    vde_net = dict(preview.get("vde_net") or {})
    metadata = _metadata_status(ctx)
    transmission_status = str(dict(preview.get("transmission_losses") or {}).get("status") or "").strip().lower()

    items = [
        {
            "label": "Vehicle Data",
            "value": metadata["value"],
            "detail": metadata["detail"],
            "status": metadata["status"],
            "action_view": "Scenario Setup",
        },
        {
            "label": "Scenario Origin",
            "value": _line_source_summary(ctx),
            "status": _line_source_status(ctx),
            "action_view": "Scenario Setup",
        },
        {
            "label": "ABC_TOTAL Basis",
            "value": _total_source_summary(ctx, preview),
            "status": _total_source_status(ctx),
            "action_view": "Roadload Build-up",
        },
        {
            "label": "Mass",
            "value": _mass_setup_summary(preview, ctx),
            "status": _mass_setup_status(preview, ctx),
            "action_view": "Vehicle Parameters",
        },
        {
            "label": "Transmission",
            "value": _transmission_summary(preview, ctx),
            "status": "ok" if transmission_status == "available" else "pending",
            "action_view": "Roadload Build-up",
            "action_technical_view": "Transmission",
        },
        {
            "label": "Cycle",
            "value": str(ctx.get("cycle_name") or "Standard / pending"),
            "status": _cycle_status(ctx),
            "action_view": "Cycle & Preview",
        },
        {
            "label": "VDE_TOTAL",
            "value": format_quantity(vde_total.get("mj_per_km"), "energy_per_distance", unavailable="Pending", format_str="%.3f"),
            "status": _energy_status(vde_total.get("mj_per_km")),
            "action_view": "Results",
        },
        {
            "label": "VDE_NET",
            "value": format_quantity(vde_net.get("mj_per_km"), "energy_per_distance", unavailable="Pending", format_str="%.3f"),
            "status": _energy_status(vde_net.get("mj_per_km")),
            "action_view": "Results",
        },
    ]

    st.caption("Executive Summary")
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            _render_summary_chip(
                item["label"],
                item["value"],
                status=item.get("status", "pending"),
                detail=item.get("detail"),
                action_view=item.get("action_view"),
                action_technical_view=item.get("action_technical_view"),
            )

    warnings = list(preview.get("warnings") or [])
    if warnings:
        st.warning("Current workflow warnings: " + ", ".join(warnings[:3]))


def _build_tire_context_row(base_row: dict | None = None) -> dict:
    ctx = st.session_state.ctx
    row = dict(base_row or {})
    for key in (
        "id",
        "legislation",
        "mass_kg",
        "test_mass_kg",
        "inertia_class",
        "twc_kg",
        "etw_kg",
        "weight_dist_fr_pct",
        "front_tire_id",
        "rear_tire_id",
        "front_pressure_psi",
        "rear_pressure_psi",
        "tire_improvement_pct",
        "tire_load_mass_basis",
        "tire_A_final",
        "tire_B_final",
        "tire_C_final",
    ):
        value = ctx.get(key)
        if value not in (None, ""):
            row[key] = value
    return row


def _saved_tire_reference_from_row(base_row: dict | None = None) -> dict | None:
    base = dict(base_row or {})
    abc = {
        "A": to_float(base.get("tire_A_final")),
        "B": to_float(base.get("tire_B_final")),
        "C": to_float(base.get("tire_C_final")),
    }
    if not any(value is not None for value in abc.values()):
        return None
    front_tire = {}
    rear_tire = {}
    if base.get("front_tire_id"):
        try:
            front_tire = get_tire_by_id(int(base.get("front_tire_id")))
        except Exception:
            front_tire = {}
    if base.get("rear_tire_id"):
        try:
            rear_tire = get_tire_by_id(int(base.get("rear_tire_id")))
        except Exception:
            rear_tire = {}
    return _build_tire_reference(
        source="Baseline saved tire reference",
        abc=abc,
        front_tire=front_tire,
        rear_tire=rear_tire or front_tire,
        front_pressure_psi=base.get("front_pressure_psi"),
        rear_pressure_psi=base.get("rear_pressure_psi"),
        tire_load_mass_basis=base.get("tire_load_mass_basis"),
        tire_load_mass_used_kg=base.get("tire_load_mass_used_kg"),
        tire_calc_source=base.get("tire_calc_source"),
        tire_calc_notes=base.get("tire_calc_notes"),
        extra={
            "size_code": base.get("tire_size"),
            "rr_n_per_kn": base.get("rrc_N_per_kN"),
        },
    )


def _reference_from_preview_result(preview_result: dict | None, *, source_label: str) -> dict | None:
    if not isinstance(preview_result, dict):
        return None
    payload = dict(preview_result.get("save_payload") or {})
    if not payload:
        return None
    return _build_tire_reference(
        source=source_label,
        abc={
            "A": payload.get("tire_A_final"),
            "B": payload.get("tire_B_final"),
            "C": payload.get("tire_C_final"),
        },
        front_tire=preview_result.get("front_tire"),
        rear_tire=preview_result.get("rear_tire"),
        front_pressure_psi=payload.get("front_pressure_psi"),
        rear_pressure_psi=payload.get("rear_pressure_psi"),
        tire_load_mass_basis=payload.get("tire_load_mass_basis"),
        tire_load_mass_used_kg=payload.get("tire_load_mass_used_kg"),
        tire_calc_source=payload.get("tire_calc_source"),
        tire_calc_notes=payload.get("tire_calc_notes"),
    )


def _render_tire_reference_metrics(reference: dict, *, title: str):
    st.caption(title)
    identity = " | ".join(
        part
        for part in (
            _string_or_none(reference.get("tire_test_code")),
            " ".join(part for part in (_string_or_none(reference.get("manufacturer")), _string_or_none(reference.get("model"))) if part),
            _string_or_none(reference.get("size_code")),
        )
        if part
    ) or "Scenario-only tire reference"
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Identity", identity)
    i2.metric("Standard", _string_or_none(reference.get("standard_family")) or "Scenario-only")
    i3.metric("RRC", format_quantity(reference.get("rr_n_per_kn"), "rrc", format_str="%.3f"))
    i4.metric("Circumference", _mm_display(reference.get("circumference_mm")))

    d1, d2, d3 = st.columns(3)
    d1.metric("Front pressure", _format_tire_pressure(reference.get("front_pressure_psi")))
    d2.metric("Rear pressure", _format_tire_pressure(reference.get("rear_pressure_psi")))
    d3.metric("Notes", "Available" if _string_or_none(reference.get("tire_calc_notes")) else "Optional")

    detail_parts = [str(reference.get("source") or "").strip()]
    if reference.get("tire_load_mass_basis"):
        detail_parts.append(f"basis={reference.get('tire_load_mass_basis')}")
    if reference.get("tire_load_mass_used_kg") is not None:
        detail_parts.append(
            "mass="
            + format_quantity(reference.get("tire_load_mass_used_kg"), "mass", format_str="%.1f")
        )
    if reference.get("test_method"):
        detail_parts.append(f"method={reference.get('test_method')}")
    if reference.get("test_source"):
        detail_parts.append(f"source={reference.get('test_source')}")
    if reference.get("tire_calc_source"):
        detail_parts.append(str(reference.get("tire_calc_source")))
    if detail_parts:
        st.caption(" | ".join(part for part in detail_parts if part))

    with st.expander("Roadload details", expanded=False):
        m1, m2, m3 = st.columns(3)
        quantity_metric(m1, "A", reference.get("A"), "force", format_str="%.3f")
        quantity_metric(m2, "B", reference.get("B"), "force_per_speed", format_str="%.6f")
        quantity_metric(m3, "C", reference.get("C"), "force_per_speed_squared", format_str="%.8f")
        if reference.get("tire_calc_notes"):
            st.caption(str(reference.get("tire_calc_notes")))


def _render_manual_tire_reference_inputs(*, prefix: str, title: str) -> dict | None:
    ctx = st.session_state.ctx
    st.caption(title)
    ctx[f"{prefix}_tire_test_code"] = ""
    ctx[f"{prefix}_manufacturer"] = ""
    ctx[f"{prefix}_model"] = ""

    size_col, note_col = st.columns([1.2, 1.8])
    with size_col:
        ctx[f"{prefix}_size_code"] = _render_tire_size_selector(
            field_key=f"{prefix}_size_code",
            label="Tire size",
        )
    ctx[f"{prefix}_notes"] = note_col.text_area(
        "Notes",
        value=str(ctx.get(f"{prefix}_notes") or ""),
        key=f"{prefix}_notes_input",
        height=68,
    )
    _prefill_circumference_from_size(
        size_field_key=f"{prefix}_size_code",
        circumference_field_key=f"{prefix}_effective_circumference_override_mm",
    )

    basis_options = ["RRC-based reference", "Direct ABC override"]
    current_basis = str(ctx.get(f"{prefix}_manual_basis") or basis_options[0]).strip()
    if current_basis not in basis_options:
        current_basis = basis_options[0]
    ctx[f"{prefix}_manual_basis"] = st.radio(
        "Roadload representation",
        basis_options,
        horizontal=True,
        index=basis_options.index(current_basis),
        key=f"{prefix}_manual_basis_radio",
    )

    n1, n2, n3, n4 = st.columns(4)
    ctx[f"{prefix}_standard_family"] = "CUSTOM"
    n1.caption("Reference family")
    n1.caption("Scenario-only")
    if ctx[f"{prefix}_manual_basis"] == "RRC-based reference":
        ctx[f"{prefix}_rr_n_per_kn"] = quantity_input(
            n2,
            "Final RRC",
            to_float(ctx.get(f"{prefix}_rr_n_per_kn"), 0.0),
            "rrc",
            key=f"{prefix}_rr_n_per_kn_input",
            step_canonical=0.05,
            format_str="%.3f",
        )
    else:
        n2.caption("Equivalent RRC")
        n2.caption("Will be derived from A after ABC is entered.")
    ctx[f"{prefix}_front_pressure_psi"] = _render_tire_pressure_input(
        n3,
        "Front pressure",
        to_float(ctx.get(f"{prefix}_front_pressure_psi"), 32.0),
        key_base=f"{prefix}_front_pressure_input",
    )
    ctx[f"{prefix}_rear_pressure_psi"] = _render_tire_pressure_input(
        n4,
        "Rear pressure",
        to_float(ctx.get(f"{prefix}_rear_pressure_psi"), 32.0),
        key_base=f"{prefix}_rear_pressure_input",
    )

    p1, p2, p3, p4 = st.columns(4)
    default_circumference_mm = _tire_circumference_mm({"size_code": ctx.get(f"{prefix}_size_code")})
    display_circumference = to_float(ctx.get(f"{prefix}_effective_circumference_override_mm"))
    if display_circumference is None:
        display_circumference = default_circumference_mm
    display_circumference = float((display_circumference or 0.0) / (25.4 if _current_unit_system() == "US customary" else 1.0))
    ctx[f"{prefix}_effective_circumference_override_mm"] = p1.number_input(
        f"Circumference [{('mm' if _current_unit_system() == 'Metric' else 'in')}]",
        value=display_circumference,
        step=1.0 if _current_unit_system() == "Metric" else 0.1,
        format="%.1f",
        key=f"{prefix}_effective_circumference_input",
    )
    if _current_unit_system() == "US customary":
        ctx[f"{prefix}_effective_circumference_override_mm"] = float(ctx[f"{prefix}_effective_circumference_override_mm"]) * 25.4
    ctx[f"{prefix}_test_mileage_km"] = 0.0
    ctx[f"{prefix}_test_method"] = ""
    ctx[f"{prefix}_is_tested_value"] = False
    ctx[f"{prefix}_test_source"] = ""
    p2.caption("Scenario-only reference")
    p3.caption("No DB metadata required")
    p4.caption("Notes + tire size are enough here")

    if ctx[f"{prefix}_manual_basis"] == "Direct ABC override":
        st.caption("Use this only when you want to override the reference roadload directly instead of deriving it from RRC.")
        c1, c2, c3 = st.columns(3)
        a_val = quantity_input(c1, "A", to_float(ctx.get(f"{prefix}_A"), 0.0), "force", key=f"{prefix}_A_input", step_canonical=0.1)
        b_val = quantity_input(c2, "B", to_float(ctx.get(f"{prefix}_B"), 0.0), "force_per_speed", key=f"{prefix}_B_input", step_canonical=0.0001, format_str="%.6f")
        c_val = quantity_input(c3, "C", to_float(ctx.get(f"{prefix}_C"), 0.0), "force_per_speed_squared", key=f"{prefix}_C_input", step_canonical=0.000001, format_str="%.8f")
        ctx[f"{prefix}_A"] = a_val
        ctx[f"{prefix}_B"] = b_val
        ctx[f"{prefix}_C"] = c_val
        if abs(float(a_val)) <= 0.0 and abs(float(b_val)) <= 0.0 and abs(float(c_val)) <= 0.0:
            st.caption("Reference still pending. Enter at least one non-zero ABC value.")
            return None
        load_kN = _resolved_tire_load_kN(ctx)
        equivalent_rr, epa_v_avg, epa_v2_avg = _equivalent_rr_from_abc(
            a_val,
            b_val,
            c_val,
            load_kN=load_kN,
        )
        expected_b = ((float(to_float(ctx.get("crr1_frac_at_120kph"), 0.0) or 0.0) * float(a_val)) / 120.0) if abs(float(a_val)) > 0.0 else 0.0
        d1, d2, d3, d4 = st.columns(4)
        quantity_metric(d1, "Equivalent RRC", equivalent_rr, "rrc", format_str="%.3f")
        quantity_metric(d2, "B expected from A", expected_b, "force_per_speed", format_str="%.6f")
        quantity_metric(d3, "B entered", b_val, "force_per_speed", format_str="%.6f")
        d4.caption(
            (
                f"EPA v_avg: {epa_v_avg:.2f} kph | EPA v2_avg: {epa_v2_avg:.2f} kph^2"
                if epa_v_avg is not None and epa_v2_avg is not None
                else "EPA speed moments unavailable"
            )
        )
    else:
        rr_value = to_float(ctx.get(f"{prefix}_rr_n_per_kn"))
        if rr_value is None or rr_value <= 0.0:
            st.caption("Reference still pending. Enter a non-zero RRC value to derive the roadload.")
            return None
        ctx["crr1_frac_at_120kph"] = st.number_input(
            "crr1 @ 120 kph [-]",
            value=float(to_float(ctx.get("crr1_frac_at_120kph"), 0.0) or 0.0),
            step=0.001,
            format="%.5f",
            key=f"{prefix}_crr1_frac_at_120kph_input",
        )
        load_kN = _resolved_tire_load_kN(ctx)
        crr_frac = float(to_float(ctx.get("crr1_frac_at_120kph"), 0.0) or 0.0)
        derived_a, derived_b, derived_c, epa_v_avg = _abc_from_final_rr_target(
            rr_value,
            load_kN=load_kN,
            crr_frac_120=crr_frac,
        )
        ctx[f"{prefix}_A"] = derived_a
        ctx[f"{prefix}_B"] = derived_b
        ctx[f"{prefix}_C"] = derived_c
        st.caption("Reference roadload is derived automatically from final RRC and crr1@120 so the resolved A/B matches the EPA-equivalent target.")
        d1, d2, d3, d4 = st.columns(4)
        quantity_metric(d1, "Derived A", derived_a, "force", format_str="%.3f")
        quantity_metric(d2, "Derived B", derived_b, "force_per_speed", format_str="%.6f")
        quantity_metric(d3, "Derived C", derived_c, "force_per_speed_squared", format_str="%.8f")
        d4.caption(f"Load used: {load_kN:.3f} kN")
        d4.caption(f"crr1@120: {crr_frac:.5f}")
        if epa_v_avg is not None:
            d4.caption(f"EPA v_avg: {epa_v_avg:.2f} kph")

    return _manual_tire_reference_from_ctx(prefix, source_label="Scenario-only manual reference")


def _clear_tire_reference_ctx(prefix: str, *, preview_ctx_key: str | None = None, widget_prefix: str | None = None):
    ctx = st.session_state.ctx
    for suffix in (
        "_manual_basis",
        "_A",
        "_B",
        "_C",
        "_tire_test_code",
        "_manufacturer",
        "_model",
        "_size_code",
        "_standard_family",
        "_rr_n_per_kn",
        "_smerf",
        "_front_pressure_psi",
        "_rear_pressure_psi",
        "_effective_circumference_override_mm",
        "_effective_circumference_override_mm_prefill_size",
        "_effective_circumference_override_mm_prefill_value",
        "_test_mileage_km",
        "_test_method",
        "_test_source",
        "_is_tested_value",
        "_notes",
    ):
        ctx.pop(f"{prefix}{suffix}", None)
    if preview_ctx_key:
        ctx.pop(preview_ctx_key, None)
    if widget_prefix:
        for suffix in (
            "_front_tire_id",
            "_rear_tire_id",
            "_front_pressure_psi",
            "_rear_pressure_psi",
            "_weight_dist_fr_pct",
            "_tire_improvement_pct",
            "_tire_load_mass_basis",
            "_same_tire_front_rear",
            "_quick_add_created_id",
        ):
            ctx.pop(f"{widget_prefix}{suffix}", None)


def _sync_baseline_cda_in_ctx(cda_value: float, *, baseline_target_id: int | None = None):
    ctx = st.session_state.ctx
    if isinstance(ctx.get("selected_baseline_row"), dict):
        ctx["selected_baseline_row"]["cda_m2"] = float(cda_value)
    if isinstance(ctx.get("baseline_dict"), dict):
        ctx["baseline_dict"]["cda_m2"] = float(cda_value)
    if baseline_target_id and int(ctx.get("vde_id_parent") or ctx.get("baseline_id") or 0) == int(baseline_target_id):
        ctx["cda_m2"] = float(cda_value)


def _sync_baseline_abc_reference_in_ctx(
    *,
    a_key: str,
    b_key: str,
    c_key: str,
    a_value: float,
    b_value: float,
    c_value: float,
    baseline_target_id: int | None = None,
):
    ctx = st.session_state.ctx
    if isinstance(ctx.get("selected_baseline_row"), dict):
        ctx["selected_baseline_row"][a_key] = float(a_value)
        ctx["selected_baseline_row"][b_key] = float(b_value)
        ctx["selected_baseline_row"][c_key] = float(c_value)
    if isinstance(ctx.get("baseline_dict"), dict):
        ctx["baseline_dict"][a_key] = float(a_value)
        ctx["baseline_dict"][b_key] = float(b_value)
        ctx["baseline_dict"][c_key] = float(c_value)
    if baseline_target_id and int(ctx.get("vde_id_parent") or ctx.get("baseline_id") or 0) == int(baseline_target_id):
        ctx[a_key] = float(a_value)
        ctx[b_key] = float(b_value)
        ctx[c_key] = float(c_value)


def _characterization_basis_options() -> list[str]:
    return ["ISO 28580", "SAE J2452 / SMERF", "Custom / Manual measured RR"]


def _characterization_basis_to_mode(basis: str) -> str:
    normalized = str(basis or "").strip()
    if normalized == "ISO 28580":
        return "ISO_28580"
    if normalized == "SAE J2452 / SMERF":
        return "SAE_J2452"
    return "CUSTOM"


def _quick_add_tire_payload_from_ctx(form_prefix: str) -> tuple[dict, list[str], dict]:
    ctx = st.session_state.ctx
    basis = str(ctx.get(f"{form_prefix}_characterization_basis") or "Custom / Manual measured RR").strip()
    calculation_mode = _characterization_basis_to_mode(basis)
    family = "ISO" if calculation_mode == "ISO_28580" else "SAE" if calculation_mode == "SAE_J2452" else "CUSTOM"
    payload = {
        "tire_test_code": _string_or_none(ctx.get(f"{form_prefix}_tire_test_code")) or "",
        "manufacturer": _string_or_none(ctx.get(f"{form_prefix}_manufacturer")) or "",
        "model": _string_or_none(ctx.get(f"{form_prefix}_model")) or "",
        "test_date": _string_or_none(ctx.get(f"{form_prefix}_test_date")) or "",
        "calculation_mode": calculation_mode,
        "standard_family": family,
        "size_code": _string_or_none(ctx.get(f"{form_prefix}_size_code")),
        "load_index": _string_or_none(ctx.get(f"{form_prefix}_load_index")),
        "speed_rating": _string_or_none(ctx.get(f"{form_prefix}_speed_rating")),
        "effective_circumference_override_mm": to_float(ctx.get(f"{form_prefix}_effective_circumference_override_mm")),
        "test_method": _string_or_none(ctx.get(f"{form_prefix}_test_method")),
        "test_source": _string_or_none(ctx.get(f"{form_prefix}_test_source")),
        "test_mileage_km": to_float(ctx.get(f"{form_prefix}_test_mileage_km")),
        "is_tested_value": 1 if _truthy_flag(ctx.get(f"{form_prefix}_is_tested_value")) else 0,
        "notes": _string_or_none(ctx.get(f"{form_prefix}_notes")),
        "rr_source": _string_or_none(ctx.get(f"{form_prefix}_test_source")),
        "rr_value_source_note": _string_or_none(ctx.get(f"{form_prefix}_test_source")),
    }

    if calculation_mode == "ISO_28580":
        payload.update(
            {
                "iso_rrc_n_per_kn": to_float(ctx.get(f"{form_prefix}_iso_rrc_n_per_kn")),
                "iso_corrected_rrc_n_per_kn": to_float(ctx.get(f"{form_prefix}_iso_corrected_rrc_n_per_kn")),
                "iso_test_pressure_kpa": to_float(ctx.get(f"{form_prefix}_iso_test_pressure_kpa")),
                "iso_test_load_n": to_float(ctx.get(f"{form_prefix}_iso_test_load_n")),
                "iso_test_speed_kph": to_float(ctx.get(f"{form_prefix}_iso_test_speed_kph")),
                "iso_rolling_resistance_force_n": to_float(ctx.get(f"{form_prefix}_iso_rolling_resistance_force_n")),
                "iso_condition_notes": _string_or_none(ctx.get(f"{form_prefix}_iso_condition_notes")),
            }
        )
    elif calculation_mode == "SAE_J2452":
        payload.update(
            {
                "sae_alpha": to_float(ctx.get(f"{form_prefix}_sae_alpha")),
                "sae_beta": to_float(ctx.get(f"{form_prefix}_sae_beta")),
                "sae_a": to_float(ctx.get(f"{form_prefix}_sae_a")),
                "sae_b": to_float(ctx.get(f"{form_prefix}_sae_b")),
                "sae_c": to_float(ctx.get(f"{form_prefix}_sae_c")),
                "smerf": to_float(ctx.get(f"{form_prefix}_smerf")),
                "test_pressure_value": to_float(ctx.get(f"{form_prefix}_sae_test_pressure_value")),
                "pressure_unit": str(ctx.get(f"{form_prefix}_sae_pressure_unit") or "kPa"),
                "test_load_value": to_float(ctx.get(f"{form_prefix}_sae_test_load_value")),
                "load_unit": str(ctx.get(f"{form_prefix}_sae_load_unit") or "N"),
            }
        )
    else:
        payload.update(
            {
                "rr_n_per_kn": to_float(ctx.get(f"{form_prefix}_rr_n_per_kn")),
                "rr_quality": "manual_input",
                "rr_method": "MANUAL_ESTIMATED",
            }
        )

    summary = summarize_tire_rr(payload)
    rr_valid = to_float(summary.get("rr_n_per_kn")) is not None and float(to_float(summary.get("rr_n_per_kn"), 0.0) or 0.0) > 0.0
    errors = []
    if not payload["tire_test_code"]:
        errors.append("Tire test code is required.")
    if not payload["manufacturer"]:
        errors.append("Manufacturer is required.")
    if not payload["model"]:
        errors.append("Model is required.")
    if not payload["test_date"]:
        errors.append("Test date is required.")
    if calculation_mode == "CUSTOM":
        if payload.get("rr_n_per_kn") is None or payload.get("rr_n_per_kn") <= 0:
            errors.append("Custom / Manual measured RR requires a valid final / equivalent RRC value.")
        if not _string_or_none(payload.get("test_source")):
            errors.append("Custom / Manual measured RR requires source / provenance.")
        if not _string_or_none(payload.get("notes")):
            errors.append("Custom / Manual measured RR requires notes / rationale.")
    elif not rr_valid:
        errors.append("Preview characterization must resolve a valid RR before this tire can be created.")
    return payload, errors, summary


def _render_characterization_preview(summary: dict, *, basis: str):
    rr_value = to_float(summary.get("rr_n_per_kn"))
    smerf_value = to_float(summary.get("smerf"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Basis", basis)
    c2.metric("Resolved final RRC", format_quantity(rr_value, "rrc", format_str="%.3f"))
    c3.metric("SMERF", "-" if smerf_value is None else f"{float(smerf_value):.4f}")
    c4.metric("Method", str(summary.get("rr_method") or "Pending"))
    quality = str(summary.get("rr_quality") or "").strip()
    source = str(summary.get("rr_source") or "").strip()
    if quality or source:
        st.caption(" | ".join(part for part in (f"quality={quality}" if quality else "", f"source={source}" if source else "") if part))


def _render_quick_add_characterization_fields(form_prefix: str):
    ctx = st.session_state.ctx
    basis_options = _characterization_basis_options()
    current_basis = str(ctx.get(f"{form_prefix}_characterization_basis") or basis_options[2]).strip()
    if current_basis not in basis_options:
        current_basis = basis_options[2]
    ctx[f"{form_prefix}_characterization_basis"] = st.radio(
        "Tire characterization basis",
        basis_options,
        horizontal=True,
        index=basis_options.index(current_basis),
        key=f"{form_prefix}_characterization_basis_radio",
    )
    basis = ctx[f"{form_prefix}_characterization_basis"]

    if basis == "ISO 28580":
        st.caption("Use ISO fields to resolve the canonical RR through the existing backend normalization path.")
        i1, i2, i3, i4 = st.columns(4)
        ctx[f"{form_prefix}_iso_rrc_n_per_kn"] = quantity_input(i1, "ISO RRC", to_float(ctx.get(f"{form_prefix}_iso_rrc_n_per_kn"), 0.0), "rrc", key=f"{form_prefix}_iso_rrc_input", step_canonical=0.05, format_str="%.3f")
        ctx[f"{form_prefix}_iso_corrected_rrc_n_per_kn"] = quantity_input(i2, "Corrected ISO RRC", to_float(ctx.get(f"{form_prefix}_iso_corrected_rrc_n_per_kn"), 0.0), "rrc", key=f"{form_prefix}_iso_corrected_rrc_input", step_canonical=0.05, format_str="%.3f")
        ctx[f"{form_prefix}_iso_test_pressure_kpa"] = i3.number_input("ISO pressure [kPa]", value=float(to_float(ctx.get(f"{form_prefix}_iso_test_pressure_kpa"), 0.0) or 0.0), step=5.0, format="%.1f", key=f"{form_prefix}_iso_pressure_input")
        ctx[f"{form_prefix}_iso_test_load_n"] = i4.number_input("ISO load [N]", value=float(to_float(ctx.get(f"{form_prefix}_iso_test_load_n"), 0.0) or 0.0), step=50.0, format="%.1f", key=f"{form_prefix}_iso_load_input")
        i5, i6 = st.columns(2)
        ctx[f"{form_prefix}_iso_test_speed_kph"] = i5.number_input("ISO speed [km/h]", value=float(to_float(ctx.get(f"{form_prefix}_iso_test_speed_kph"), 0.0) or 0.0), step=1.0, format="%.1f", key=f"{form_prefix}_iso_speed_input")
        ctx[f"{form_prefix}_iso_rolling_resistance_force_n"] = i6.number_input("ISO rolling resistance force [N]", value=float(to_float(ctx.get(f"{form_prefix}_iso_rolling_resistance_force_n"), 0.0) or 0.0), step=1.0, format="%.2f", key=f"{form_prefix}_iso_force_input")
        ctx[f"{form_prefix}_iso_condition_notes"] = st.text_input("ISO condition notes", value=str(ctx.get(f"{form_prefix}_iso_condition_notes") or ""), key=f"{form_prefix}_iso_condition_notes_input")
    elif basis == "SAE J2452 / SMERF":
        st.caption("Use official SAE/J2452 characterization inputs to calculate SMERF and reference RRC.")
        s1, s2, s3, s4, s5 = st.columns(5)
        ctx[f"{form_prefix}_sae_alpha"] = s1.number_input("alpha", value=float(to_float(ctx.get(f"{form_prefix}_sae_alpha"), 0.0) or 0.0), step=0.01, format="%.6f", key=f"{form_prefix}_sae_alpha_input")
        ctx[f"{form_prefix}_sae_beta"] = s2.number_input("beta", value=float(to_float(ctx.get(f"{form_prefix}_sae_beta"), 0.0) or 0.0), step=0.01, format="%.6f", key=f"{form_prefix}_sae_beta_input")
        ctx[f"{form_prefix}_sae_a"] = s3.number_input("a", value=float(to_float(ctx.get(f"{form_prefix}_sae_a"), 0.0) or 0.0), step=0.0001, format="%.6f", key=f"{form_prefix}_sae_a_input")
        ctx[f"{form_prefix}_sae_b"] = s4.number_input("b", value=float(to_float(ctx.get(f"{form_prefix}_sae_b"), 0.0) or 0.0), step=0.0001, format="%.6f", key=f"{form_prefix}_sae_b_input")
        ctx[f"{form_prefix}_sae_c"] = s5.number_input("c", value=float(to_float(ctx.get(f"{form_prefix}_sae_c"), 0.0) or 0.0), step=0.0001, format="%.6f", key=f"{form_prefix}_sae_c_input")
        s6, s7, s8, s9 = st.columns(4)
        pressure_units = ["kPa", "psi"]
        load_units = ["N", "kg", "lbf"]
        pressure_unit = str(ctx.get(f"{form_prefix}_sae_pressure_unit") or "kPa")
        load_unit = str(ctx.get(f"{form_prefix}_sae_load_unit") or "N")
        if pressure_unit not in pressure_units:
            pressure_unit = "kPa"
        if load_unit not in load_units:
            load_unit = "N"
        ctx[f"{form_prefix}_sae_pressure_unit"] = s6.selectbox("Pressure unit", pressure_units, index=pressure_units.index(pressure_unit), key=f"{form_prefix}_sae_pressure_unit_input")
        ctx[f"{form_prefix}_sae_test_pressure_value"] = s7.number_input(f"SAE pressure [{ctx[f'{form_prefix}_sae_pressure_unit']}]", value=float(to_float(ctx.get(f"{form_prefix}_sae_test_pressure_value"), 0.0) or 0.0), step=1.0, format="%.2f", key=f"{form_prefix}_sae_pressure_value_input")
        ctx[f"{form_prefix}_sae_load_unit"] = s8.selectbox("Load unit", load_units, index=load_units.index(load_unit), key=f"{form_prefix}_sae_load_unit_input")
        ctx[f"{form_prefix}_sae_test_load_value"] = s9.number_input(f"SAE load [{ctx[f'{form_prefix}_sae_load_unit']}]", value=float(to_float(ctx.get(f"{form_prefix}_sae_test_load_value"), 0.0) or 0.0), step=1.0, format="%.2f", key=f"{form_prefix}_sae_load_value_input")
        ctx[f"{form_prefix}_smerf"] = st.number_input("Optional SMERF override", value=float(to_float(ctx.get(f"{form_prefix}_smerf"), 0.0) or 0.0), step=0.1, format="%.4f", key=f"{form_prefix}_smerf_input")
    else:
        st.caption("Custom data stores a measured or engineering RR directly without claiming SAE / ISO methodology.")
        st.caption("Use this path when you have a final / equivalent RRC from manual measurement, estimate, or external engineering input.")
        ctx[f"{form_prefix}_rr_n_per_kn"] = quantity_input(st, "Final / equivalent RRC", to_float(ctx.get(f"{form_prefix}_rr_n_per_kn"), 0.0), "rrc", key=f"{form_prefix}_rr_input", step_canonical=0.05, format_str="%.3f")
        st.info("For custom data, source / provenance and notes are required before creating a physical Tire DB record.")


def _render_quick_add_tire_reference_builder(
    *,
    widget_prefix: str,
    preview_ctx_key: str,
    base_row: dict | None = None,
) -> dict | None:
    ctx = st.session_state.ctx
    form_prefix = f"{widget_prefix}_quick_add"
    ctx.setdefault(f"{form_prefix}_test_date", pd.Timestamp.today().strftime("%Y-%m-%d"))
    ctx.setdefault(f"{form_prefix}_characterization_basis", "Custom / Manual measured RR")
    st.caption("Create a compact tire record in Tire DB without leaving VDE Setup. After save, it becomes the selected reference here.")

    q1, q2, q3, q4 = st.columns(4)
    ctx[f"{form_prefix}_tire_test_code"] = q1.text_input("Tire test code *", value=str(ctx.get(f"{form_prefix}_tire_test_code") or ""), key=f"{form_prefix}_tire_test_code_input")
    ctx[f"{form_prefix}_manufacturer"] = q2.text_input("Manufacturer *", value=str(ctx.get(f"{form_prefix}_manufacturer") or ""), key=f"{form_prefix}_manufacturer_input")
    ctx[f"{form_prefix}_model"] = q3.text_input("Model *", value=str(ctx.get(f"{form_prefix}_model") or ""), key=f"{form_prefix}_model_input")
    ctx[f"{form_prefix}_test_date"] = q4.text_input("Test date *", value=str(ctx.get(f"{form_prefix}_test_date") or ""), key=f"{form_prefix}_test_date_input")

    q5, q6, q7, q8 = st.columns(4)
    with q5:
        ctx[f"{form_prefix}_size_code"] = _render_tire_size_selector(
            field_key=f"{form_prefix}_size_code",
            label="Tire size",
        )
    _prefill_circumference_from_size(
        size_field_key=f"{form_prefix}_size_code",
        circumference_field_key=f"{form_prefix}_effective_circumference_override_mm",
    )
    ctx[f"{form_prefix}_load_index"] = q6.text_input("Load index", value=str(ctx.get(f"{form_prefix}_load_index") or ""), key=f"{form_prefix}_load_index_input")
    ctx[f"{form_prefix}_speed_rating"] = q7.text_input("Speed rating", value=str(ctx.get(f"{form_prefix}_speed_rating") or ""), key=f"{form_prefix}_speed_rating_input")
    ctx[f"{form_prefix}_test_method"] = q8.text_input("Test type", value=str(ctx.get(f"{form_prefix}_test_method") or ""), key=f"{form_prefix}_test_method_input")

    q9, q10, q11, q12 = st.columns(4)
    ctx[f"{form_prefix}_front_pressure_psi"] = _render_tire_pressure_input(
        q9,
        "Front pressure",
        to_float(ctx.get(f"{form_prefix}_front_pressure_psi"), 32.0),
        key_base=f"{form_prefix}_front_pressure_input",
    )
    ctx[f"{form_prefix}_rear_pressure_psi"] = _render_tire_pressure_input(
        q10,
        "Rear pressure",
        to_float(ctx.get(f"{form_prefix}_rear_pressure_psi"), to_float(ctx.get(f"{form_prefix}_front_pressure_psi"), 32.0)),
        key_base=f"{form_prefix}_rear_pressure_input",
    )
    ctx[f"{form_prefix}_test_mileage_km"] = q11.number_input(
        f"Mileage [{'km' if _current_unit_system() == 'Metric' else 'mi'}]",
        value=float((to_float(ctx.get(f"{form_prefix}_test_mileage_km"), 0.0) or 0.0) * (0.621371192237 if _current_unit_system() == "US customary" else 1.0)),
        step=100.0,
        format="%.0f",
        key=f"{form_prefix}_test_mileage_input",
    )
    if _current_unit_system() == "US customary":
        ctx[f"{form_prefix}_test_mileage_km"] = float(ctx[f"{form_prefix}_test_mileage_km"]) / 0.621371192237
    circumference_default_mm = _tire_circumference_mm({"size_code": ctx.get(f"{form_prefix}_size_code")})
    circumference_value = to_float(ctx.get(f"{form_prefix}_effective_circumference_override_mm"), circumference_default_mm)
    display_circumference = float((circumference_value or 0.0) / (25.4 if _current_unit_system() == "US customary" else 1.0))
    ctx[f"{form_prefix}_effective_circumference_override_mm"] = q12.number_input(
        f"Circumference [{'mm' if _current_unit_system() == 'Metric' else 'in'}]",
        value=display_circumference,
        step=1.0 if _current_unit_system() == "Metric" else 0.1,
        format="%.1f",
        key=f"{form_prefix}_circumference_input",
    )
    if _current_unit_system() == "US customary":
        ctx[f"{form_prefix}_effective_circumference_override_mm"] = float(ctx[f"{form_prefix}_effective_circumference_override_mm"]) * 25.4

    ctx[f"{form_prefix}_is_tested_value"] = st.checkbox("Tested value", value=bool(ctx.get(f"{form_prefix}_is_tested_value", False)), key=f"{form_prefix}_is_tested_value_input")
    ctx[f"{form_prefix}_test_source"] = st.text_input("Source / provenance", value=str(ctx.get(f"{form_prefix}_test_source") or ""), key=f"{form_prefix}_test_source_input")
    ctx[f"{form_prefix}_notes"] = st.text_area("Notes", value=str(ctx.get(f"{form_prefix}_notes") or ""), key=f"{form_prefix}_notes_input", height=80)
    _render_quick_add_characterization_fields(form_prefix)

    preview_summary = ctx.get(f"{form_prefix}_characterization_preview")
    preview_col, create_hint_col = st.columns([1, 4])
    if preview_col.button("Preview characterization", key=f"{form_prefix}_preview_characterization_button"):
        payload, _, summary = _quick_add_tire_payload_from_ctx(form_prefix)
        preview_summary = {"payload": payload, "summary": summary}
        ctx[f"{form_prefix}_characterization_preview"] = preview_summary

    if isinstance(preview_summary, dict):
        summary = dict(preview_summary.get("summary") or {})
        _render_characterization_preview(summary, basis=str(ctx.get(f"{form_prefix}_characterization_basis") or ""))
        rr_preview = to_float(summary.get("rr_n_per_kn"))
        if rr_preview is None or rr_preview <= 0:
            create_hint_col.caption("Characterization preview is still missing a valid RR result.")
            st.warning("Preview characterization did not resolve a valid RR yet. Complete the required characterization inputs first.")
    else:
        create_hint_col.caption("Run Preview characterization before creating a Tire DB record from this compact form.")

    preview_result = ctx.get(preview_ctx_key)
    code = _string_or_none(ctx.get(f"{form_prefix}_tire_test_code"))
    existing = {}
    if code:
        try:
            existing = get_tire_by_code(code) or {}
        except Exception:
            existing = {}

    if existing:
        st.warning(f"Tire test code `{code}` already exists in Tire DB as id={existing.get('id')}.")
        if st.button("Use existing tire", key=f"{form_prefix}_use_existing_button"):
            ctx[f"{widget_prefix}_front_tire_id"] = int(existing["id"])
            ctx[f"{widget_prefix}_rear_tire_id"] = int(existing["id"])
            ctx[f"{widget_prefix}_same_tire_front_rear"] = True
            try:
                preview_result = preview_tire_roadload_from_row(
                    _build_tire_context_row(base_row),
                    {
                        "front_tire_id": int(existing["id"]),
                        "rear_tire_id": int(existing["id"]),
                        "same_tire_front_rear": True,
                        "front_pressure_psi": to_float(ctx.get(f"{form_prefix}_front_pressure_psi"), 32.0),
                        "rear_pressure_psi": to_float(ctx.get(f"{form_prefix}_rear_pressure_psi"), 32.0),
                        "front_weight_distribution_pct": to_float(ctx.get("weight_dist_fr_pct"), 50.0),
                        "tire_improvement_pct": 0.0,
                        "tire_load_mass_basis": resolve_tire_load_mass_basis(ctx),
                        "mass_kg": ctx.get("mass_kg"),
                        "test_mass_kg": ctx.get("test_mass_kg"),
                        "inertia_class": ctx.get("inertia_class"),
                        "twc_kg": ctx.get("twc_kg"),
                        "etw_kg": ctx.get("etw_kg"),
                    },
                )
                ctx[preview_ctx_key] = preview_result
            except Exception as exc:
                st.error(f"Could not preview the existing tire: {exc}")

    preview_rr_ready = False
    if isinstance(preview_summary, dict):
        preview_rr = to_float(((preview_summary.get("summary") or {}).get("rr_n_per_kn")))
        preview_rr_ready = preview_rr is not None and preview_rr > 0

    create_col, _ = st.columns([1, 4])
    if create_col.button("Create and select tire", key=f"{form_prefix}_create_button", disabled=bool(existing) or not preview_rr_ready):
        payload, errors, summary = _quick_add_tire_payload_from_ctx(form_prefix)
        if errors:
            for error in errors:
                st.error(error)
        else:
            try:
                new_id = int(create_tire_from_form(payload))
                ctx[f"{widget_prefix}_quick_add_created_id"] = new_id
                ctx[f"{widget_prefix}_front_tire_id"] = new_id
                ctx[f"{widget_prefix}_rear_tire_id"] = new_id
                ctx[f"{widget_prefix}_same_tire_front_rear"] = True
                preview_result = preview_tire_roadload_from_row(
                    _build_tire_context_row(base_row),
                    {
                        "front_tire_id": new_id,
                        "rear_tire_id": new_id,
                        "same_tire_front_rear": True,
                        "front_pressure_psi": to_float(ctx.get(f"{form_prefix}_front_pressure_psi"), 32.0),
                        "rear_pressure_psi": to_float(ctx.get(f"{form_prefix}_rear_pressure_psi"), 32.0),
                        "front_weight_distribution_pct": to_float(ctx.get("weight_dist_fr_pct"), 50.0),
                        "tire_improvement_pct": 0.0,
                        "tire_load_mass_basis": resolve_tire_load_mass_basis(ctx),
                        "mass_kg": ctx.get("mass_kg"),
                        "test_mass_kg": ctx.get("test_mass_kg"),
                        "inertia_class": ctx.get("inertia_class"),
                        "twc_kg": ctx.get("twc_kg"),
                        "etw_kg": ctx.get("etw_kg"),
                    },
                )
                ctx[preview_ctx_key] = preview_result
                st.success(f"Tire created in Tire DB with id={new_id} and selected here.")
            except Exception as exc:
                st.error(f"Could not create tire: {exc}")

    if not isinstance(preview_result, dict):
        return None
    reference = _reference_from_preview_result(preview_result, source_label="Quick-added Tire DB reference")
    if reference:
        _render_tire_reference_metrics(reference, title="Resolved quick-add reference")
    return preview_result


def _render_tire_reference_selector(
    *,
    title: str,
    helper_text: str,
    mode_key: str,
    prefix: str,
    widget_prefix: str,
    source_label: str,
    base_row: dict | None = None,
    inherited_reference: dict | None = None,
    allow_unset: bool = False,
    save_to_baseline_label: str | None = None,
    baseline_target_id: int | None = None,
) -> dict | None:
    ctx = st.session_state.ctx
    with st.container(border=True):
        _render_tire_editor_block_header(title, helper_text)
        options = []
        if inherited_reference:
            options.append("Inherited baseline")
        if allow_unset:
            options.append("Not set")
        options.extend(["Select from Tire DB", "Quick add tire to DB", "Scenario-only manual reference"])
        default_mode = str(ctx.get(mode_key) or ("Inherited baseline" if inherited_reference else options[0]))
        if default_mode not in options:
            default_mode = "Inherited baseline" if inherited_reference else options[0]
        ctx[mode_key] = st.radio(
            f"{title} source",
            options,
            horizontal=False,
            index=options.index(default_mode),
            key=f"{mode_key}_radio",
        )

        if ctx[mode_key] == "Inherited baseline":
            _clear_tire_reference_ctx(prefix, preview_ctx_key=f"{prefix}_preview_result", widget_prefix=widget_prefix)
            reference = inherited_reference
        elif ctx[mode_key] == "Not set":
            _clear_tire_reference_ctx(prefix, preview_ctx_key=f"{prefix}_preview_result", widget_prefix=widget_prefix)
            reference = None
            st.caption("Reference is staged only when you choose a source below.")
        elif ctx[mode_key] == "Scenario-only manual reference":
            reference = _render_manual_tire_reference_inputs(prefix=prefix, title="Scenario-only tire reference")
        elif ctx[mode_key] == "Quick add tire to DB":
            preview = _render_quick_add_tire_reference_builder(
                widget_prefix=widget_prefix,
                preview_ctx_key=f"{prefix}_preview_result",
                base_row=base_row,
            )
            reference = _reference_from_preview_result(preview, source_label=source_label)
        else:
            preview = _render_tire_db_reference_builder(
                widget_prefix=widget_prefix,
                preview_ctx_key=f"{prefix}_preview_result",
                base_row=base_row,
                save_to_baseline_label=save_to_baseline_label,
                baseline_target_id=baseline_target_id,
            )
            reference = _reference_from_preview_result(preview, source_label=source_label)

        if reference:
            _render_tire_reference_metrics(reference, title="Active reference")
        return reference


def _sync_tire_scenario_snapshot_fields(
    *,
    applied_method: str,
    current_reference: dict | None,
    walked_reference: dict | None,
    equivalent_a,
    equivalent_b,
    equivalent_c,
):
    ctx = st.session_state.ctx
    if applied_method == "Keep inherited" or str(ctx.get("component_mode_tires") or "").strip() == "Keep inherited":
        for key in (
            "front_tire_id",
            "rear_tire_id",
            "tire_load_mass_used_kg",
            "tire_A_final",
            "tire_B_final",
            "tire_C_final",
        ):
            ctx.pop(key, None)
        ctx["tire_rr_note"] = "Keep inherited"
        ctx["tire_calc_source"] = "keep_inherited"
        ctx["tire_calc_notes"] = json.dumps({"applied_method": "Keep inherited"}, ensure_ascii=True)
        return

    for key in (
        "front_tire_id",
        "rear_tire_id",
        "tire_load_mass_basis",
        "tire_load_mass_used_kg",
        "tire_A_final",
        "tire_B_final",
        "tire_C_final",
    ):
        ctx.pop(key, None)

    active_reference = None
    if applied_method == "Walked tire comparison" and walked_reference:
        active_reference = walked_reference
    elif applied_method == "Tire Improvement %" and current_reference:
        active_reference = current_reference
    elif applied_method == "Manual Delta RR" and current_reference:
        active_reference = current_reference

    if active_reference:
        for key in (
            "front_tire_id",
            "rear_tire_id",
            "front_pressure_psi",
            "rear_pressure_psi",
            "tire_load_mass_basis",
            "tire_load_mass_used_kg",
            "tire_calc_source",
        ):
            value = active_reference.get(key)
            if value not in (None, ""):
                ctx[key] = value
        if active_reference.get("size_code"):
            ctx["tire_size"] = active_reference.get("size_code")
        if active_reference.get("rr_n_per_kn") is not None:
            ctx["rrc_N_per_kN"] = active_reference.get("rr_n_per_kn")
        if _reference_has_nonzero_abc(active_reference):
            ctx["tire_A_final"] = active_reference.get("A")
            ctx["tire_B_final"] = active_reference.get("B")
            ctx["tire_C_final"] = active_reference.get("C")

    note_payload = {
        "applied_method": applied_method,
        "change_intent": _string_or_none(ctx.get("tire_manual_change_intent")) or "Scenario-only engineering adjustment",
        "data_role": (
            "engineering_target"
            if str(ctx.get("tire_manual_change_intent") or "").strip() == "Engineering target / supplier request"
            else "scenario_assumption"
        ),
        "evidence_basis": (
            "engineering_estimate"
            if str(ctx.get("tire_manual_change_intent") or "").strip() == "Engineering target / supplier request"
            else "manual_input"
        ),
        "current_reference": _tire_provenance_note(current_reference),
        "walked_reference": _tire_provenance_note(walked_reference),
        "equivalent_delta_abc": {
            "A": equivalent_a,
            "B": equivalent_b,
            "C": equivalent_c,
        },
        "reference_rrc_n_per_kn": to_float((current_reference or {}).get("rr_n_per_kn"), to_float(ctx.get("rrc_N_per_kN"))),
        "target_rrc_n_per_kn": to_float(ctx.get("tire_manual_target_rr_n_per_kn")),
        "manual_delta_rr_n_per_kn": to_float(ctx.get("tire_manual_delta_rr_n_per_kn")),
        "tire_improvement_pct": to_float(ctx.get("tire_improvement_pct")),
        "manual_source": _string_or_none(ctx.get("tire_manual_delta_rr_source")),
        "manual_target_label": _string_or_none(ctx.get("tire_manual_delta_rr_label")),
        "manual_size_code": _string_or_none(ctx.get("tire_manual_delta_rr_size_code")),
        "manual_notes": _string_or_none(ctx.get("tire_manual_delta_rr_notes")),
        "input_basis": _string_or_none(ctx.get("tire_manual_adjustment_input_type")),
    }
    ctx["tire_calc_source"] = (
        "engineering_target"
        if str(ctx.get("tire_manual_change_intent") or "").strip() == "Engineering target / supplier request"
        else
        "tire_walked_comparison" if applied_method == "Walked tire comparison"
        else "tire_improvement_pct" if applied_method == "Tire Improvement %"
        else "manual_delta_rr"
    )
    ctx["tire_calc_notes"] = json.dumps(note_payload, ensure_ascii=True)
    if _string_or_none(note_payload.get("manual_size_code")):
        ctx["tire_size"] = note_payload["manual_size_code"]
    ctx["tire_rr_note"] = applied_method


def _render_tire_db_reference_builder(
    *,
    widget_prefix: str,
    preview_ctx_key: str,
    base_row: dict | None = None,
    save_to_baseline_label: str | None = None,
    baseline_target_id: int | None = None,
) -> dict | None:
    ctx = st.session_state.ctx
    try:
        tires = get_available_tires()
    except Exception as e:
        st.error(f"Could not load tire database: {e}")
        return None

    if not tires:
        st.warning("No active tires found in the tire database.")
        return None

    tire_by_id = {int(r["id"]): r for r in tires if r.get("id") is not None}
    tire_by_code = {
        str(r.get("tire_test_code") or "").strip(): r
        for r in tires
        if str(r.get("tire_test_code") or "").strip()
    }
    tire_codes = sorted(tire_by_code.keys())
    if not tire_codes:
        st.warning("Active tire rows need tire_test_code before they can be used here.")
        return None

    def _code_for_tire_id(tire_id) -> str:
        try:
            row = tire_by_id.get(int(tire_id))
        except Exception:
            row = None
        return str((row or {}).get("tire_test_code") or "").strip()

    base = dict(base_row or {})
    front_default = _code_for_tire_id(ctx.get(f"{widget_prefix}_front_tire_id") or base.get("front_tire_id"))
    rear_default = _code_for_tire_id(ctx.get(f"{widget_prefix}_rear_tire_id") or base.get("rear_tire_id"))
    if front_default not in tire_codes:
        front_default = tire_codes[0]
    if rear_default not in tire_codes:
        rear_default = front_default

    same_default = bool(ctx.get(f"{widget_prefix}_same_tire_front_rear", True) if ctx.get(f"{widget_prefix}_same_tire_front_rear") is not None else (front_default == rear_default))
    legislation = str(base.get("legislation") or ctx.get("legislation") or "").strip().upper()
    basis_default = resolve_tire_load_mass_basis(
        {
            "legislation": legislation,
            "tire_load_mass_basis": ctx.get(f"{widget_prefix}_tire_load_mass_basis") or ctx.get("tire_load_mass_basis") or base.get("tire_load_mass_basis"),
        }
    )

    c1, c2 = st.columns(2)
    front_code = c1.selectbox(
        "Front tire",
        tire_codes,
        index=tire_codes.index(front_default),
        key=f"{widget_prefix}_front_tire_code",
    )
    front_tire_id = int(tire_by_code[front_code]["id"])
    c1.caption(_tire_label(tire_by_code[front_code]))
    same_tire = c2.checkbox(
        "Same tire front/rear",
        value=same_default,
        key=f"{widget_prefix}_same_tire_checkbox",
    )

    rear_tire_id = front_tire_id
    if same_tire:
        rear_code = front_code
        st.caption(f"Rear tire mirrors front tire: {_tire_label(tire_by_id[front_tire_id])}")
    else:
        rear_code = st.selectbox(
            "Rear tire",
            tire_codes,
            index=tire_codes.index(rear_default),
            key=f"{widget_prefix}_rear_tire_code",
        )
        rear_tire_id = int(tire_by_code[rear_code]["id"])
        st.caption(_tire_label(tire_by_code[rear_code]))

    p1, p2, p3, p4 = st.columns(4)
    front_pressure_psi = _render_tire_pressure_input(
        p1,
        "Front pressure",
        to_float(ctx.get(f"{widget_prefix}_front_pressure_psi", ctx.get("front_pressure_psi", base.get("front_pressure_psi", 32.0))), 32.0),
        key_base=f"{widget_prefix}_front_pressure_input",
    )
    rear_pressure_psi = _render_tire_pressure_input(
        p2,
        "Rear pressure",
        to_float(ctx.get(f"{widget_prefix}_rear_pressure_psi", ctx.get("rear_pressure_psi", base.get("rear_pressure_psi", 32.0))), 32.0),
        key_base=f"{widget_prefix}_rear_pressure_input",
    )
    front_weight_distribution_pct = p3.number_input(
        "Front weight distribution [%]",
        value=float(ctx.get(f"{widget_prefix}_weight_dist_fr_pct", ctx.get("weight_dist_fr_pct", base.get("weight_dist_fr_pct", 50.0))) or 50.0),
        min_value=0.0,
        max_value=100.0,
        step=0.5,
        format="%.1f",
        key=f"{widget_prefix}_weight_dist_input",
    )
    tire_improvement_pct = p4.number_input(
        "Tire improvement [%]",
        value=float(ctx.get(f"{widget_prefix}_tire_improvement_pct", ctx.get("tire_improvement_pct", base.get("tire_improvement_pct", 0.0))) or 0.0),
        step=0.5,
        format="%.1f",
        key=f"{widget_prefix}_improvement_input",
    )

    tire_load_mass_basis = str(ctx.get(f"{widget_prefix}_tire_load_mass_basis") or basis_default).strip().upper()
    st.caption(f"Reference mass basis: `{tire_load_mass_basis}`")

    preview_result = None
    if st.button("Preview tire reference", key=f"{widget_prefix}_preview_button"):
        try:
            row_context = _build_tire_context_row(base_row)
            preview_result = preview_tire_roadload_from_row(
                row_context,
                {
                    "front_tire_id": front_tire_id,
                    "rear_tire_id": rear_tire_id,
                    "same_tire_front_rear": same_tire,
                    "front_pressure_psi": front_pressure_psi,
                    "rear_pressure_psi": rear_pressure_psi,
                    "front_weight_distribution_pct": front_weight_distribution_pct,
                    "tire_improvement_pct": tire_improvement_pct,
                    "tire_load_mass_basis": tire_load_mass_basis,
                    "mass_kg": ctx.get("mass_kg"),
                    "test_mass_kg": ctx.get("test_mass_kg"),
                    "inertia_class": ctx.get("inertia_class"),
                    "twc_kg": ctx.get("twc_kg"),
                    "etw_kg": ctx.get("etw_kg"),
                },
            )
            ctx[f"{widget_prefix}_front_tire_id"] = front_tire_id
            ctx[f"{widget_prefix}_rear_tire_id"] = rear_tire_id
            ctx[f"{widget_prefix}_front_pressure_psi"] = front_pressure_psi
            ctx[f"{widget_prefix}_rear_pressure_psi"] = rear_pressure_psi
            ctx[f"{widget_prefix}_weight_dist_fr_pct"] = front_weight_distribution_pct
            ctx[f"{widget_prefix}_tire_improvement_pct"] = tire_improvement_pct
            ctx[f"{widget_prefix}_tire_load_mass_basis"] = tire_load_mass_basis
            ctx[preview_ctx_key] = preview_result
        except Exception as e:
            st.error(f"Failed to preview tire reference: {e}")
            return None

    if preview_result is None:
        cached = ctx.get(preview_ctx_key)
        if isinstance(cached, dict) and cached.get("save_payload"):
            preview_result = cached

    if not preview_result:
        return None

    reference = _reference_from_preview_result(preview_result, source_label="Tire DB reference")
    if reference:
        _render_tire_reference_metrics(reference, title="Resolved Tire DB reference")

    if save_to_baseline_label and baseline_target_id:
        if st.button(save_to_baseline_label, key=f"{widget_prefix}_save_to_baseline_button"):
            try:
                payload = save_tire_roadload_to_vde(int(baseline_target_id), preview_result)
                ctx["tire_saved_payload"] = payload
                st.success(f"Tire reference saved to VDE id={baseline_target_id}.")
            except Exception as e:
                st.error(f"Failed to update baseline tire reference: {e}")

    with st.expander("Reference preview details", expanded=False):
        st.write(
            {
                "application": preview_result.get("application"),
                "mass_resolution": preview_result.get("mass_resolution"),
                "calculation": preview_result.get("calculation"),
                "save_payload": preview_result.get("save_payload"),
            }
        )
    return preview_result


def _render_tire_preview_result(preview_result: dict, *, saved_vde_id: int | None = None):
    ctx = st.session_state.ctx
    calc = preview_result.get("calculation", {})
    loads = calc.get("loads", {})
    total_final = calc.get("total_final_abc", {})
    mass_resolution = preview_result.get("mass_resolution", {})
    delta_vs_saved = preview_result.get("delta_vs_saved", {})

    st.success("Tire component preview ready.")
    m1, m2, m3, m4 = st.columns(4)
    quantity_metric(m1, "Mass used", calc.get("tire_load_mass_used_kg"), "mass", format_str="%.1f")
    quantity_metric(m2, "Front axle load", loads.get("front_axle_load_n"), "force", format_str="%.1f")
    quantity_metric(m3, "Rear axle load", loads.get("rear_axle_load_n"), "force", format_str="%.1f")
    m4.metric("Tire ABC", _compact_abc(total_final))

    include_default = bool(ctx.get("include_tire_component", True))
    ctx["include_tire_component"] = st.checkbox(
        "Include tire component in TOTAL preview",
        value=include_default,
        key="tire_component_include_checkbox",
    )
    st.caption(
        f"Mass basis: {mass_resolution.get('basis')} ({mass_resolution.get('source_field')}). "
        "When enabled, the workflow preview adds this tire component into ABC_TOTAL."
    )

    if delta_vs_saved:
        st.caption(f"Delta vs saved tire ABC: {_compact_abc(delta_vs_saved)}")

    with st.expander("Tire component details", expanded=False):
        st.write(
            {
                "application": preview_result.get("application"),
                "mass_resolution": preview_result.get("mass_resolution"),
                "calculation": preview_result.get("calculation"),
                "save_payload": preview_result.get("save_payload"),
                "component": preview_result.get("component_dict"),
            }
        )

    if saved_vde_id:
        save_col, note_col = st.columns([1, 3])
        if save_col.button("Apply tire data to saved baseline", key=f"btn_tire_save_component_{saved_vde_id}"):
            try:
                payload = save_tire_roadload_to_vde(int(saved_vde_id), preview_result)
                ctx["tire_saved_payload"] = payload
                st.success(f"Tire roadload saved to VDE id={saved_vde_id}.")
            except Exception as e:
                st.error(f"Failed to save tire roadload to VDE: {e}")

        saved_payload = ctx.get("tire_saved_payload")
        if isinstance(saved_payload, dict):
            note_col.caption(
                "Saved tire application: "
                f"front={saved_payload.get('front_tire_id')} | "
                f"rear={saved_payload.get('rear_tire_id')} | "
                f"ABC={_compact_abc({'A': saved_payload.get('tire_A_final'), 'B': saved_payload.get('tire_B_final'), 'C': saved_payload.get('tire_C_final')})}"
            )


def render_tire_component_section(
    *,
    base_row: dict | None = None,
    saved_vde_id: int | None = None,
    tires_df=None,
    source_mode_override: str | None = None,
    show_source_selector: bool = True,
):
    ctx = st.session_state.ctx
    _render_tire_pressure_unit_toggle(key="tire_component_section_pressure_unit_toggle")
    source_options = ["Tire DB", "Manual RR"]
    current_source = str(source_mode_override or ctx.get("tire_component_source") or "Manual RR")
    if current_source not in source_options:
        current_source = "Manual RR"
    if show_source_selector:
        ctx["tire_component_source"] = st.radio(
            "Tire component source",
            source_options,
            horizontal=True,
            index=source_options.index(current_source),
            key="tire_component_source_radio",
        )
    else:
        ctx["tire_component_source"] = current_source
        st.caption(f"Tire source mode: `{current_source}`")

    if ctx["tire_component_source"] == "Manual RR":
        ctx["include_tire_component"] = False
        st.caption("Manual RR stays inside the Tires component as a fallback path when no tire DB component is selected.")
        render_rr_section(prefill=base_row, tires_df=tires_df)
        return

    try:
        tires = get_available_tires()
    except Exception as e:
        st.error(f"Could not load tire database: {e}")
        ctx["include_tire_component"] = False
        return

    if not tires:
        st.warning("No active tires found in the tire database. Use Manual RR for now.")
        ctx["include_tire_component"] = False
        return

    tire_by_id = {int(r["id"]): r for r in tires if r.get("id") is not None}
    tire_by_code = {
        str(r.get("tire_test_code") or "").strip(): r
        for r in tires
        if str(r.get("tire_test_code") or "").strip()
    }
    tire_codes = sorted(tire_by_code.keys())
    if not tire_codes:
        st.warning("Active tire rows need tire_test_code before they can be used in VDE Setup.")
        ctx["include_tire_component"] = False
        return

    def _code_for_tire_id(tire_id) -> str:
        try:
            row = tire_by_id.get(int(tire_id))
        except Exception:
            row = None
        return str((row or {}).get("tire_test_code") or "").strip()

    base = dict(base_row or {})
    front_default = _code_for_tire_id(ctx.get("front_tire_id") or base.get("front_tire_id"))
    rear_default = _code_for_tire_id(ctx.get("rear_tire_id") or base.get("rear_tire_id"))
    if front_default not in tire_codes:
        front_default = tire_codes[0]
    if rear_default not in tire_codes:
        rear_default = front_default

    same_default = bool(ctx.get("same_tire_front_rear", False) or (front_default == rear_default))
    legislation = str(base.get("legislation") or ctx.get("legislation") or "").strip().upper()
    basis_default = resolve_tire_load_mass_basis(
        {
            "legislation": legislation,
            "tire_load_mass_basis": ctx.get("tire_load_mass_basis") or base.get("tire_load_mass_basis"),
        }
    )

    c1, c2 = st.columns(2)
    front_tire_code = c1.selectbox("Front tire", tire_codes, index=tire_codes.index(front_default), key="tire_component_front_code")
    front_tire_id = int(tire_by_code[front_tire_code]["id"])
    c1.caption(_tire_label(tire_by_code[front_tire_code]))
    same_tire = c2.checkbox("Same tire front/rear", value=same_default, key="tire_component_same_checkbox")

    rear_tire_id = front_tire_id
    if same_tire:
        st.caption(f"Rear tire mirrors front tire: {_tire_label(tire_by_id[front_tire_id])}")
        rear_tire_code = front_tire_code
    else:
        rear_tire_code = st.selectbox("Rear tire", tire_codes, index=tire_codes.index(rear_default), key="tire_component_rear_code")
        rear_tire_id = int(tire_by_code[rear_tire_code]["id"])
        st.caption(_tire_label(tire_by_code[rear_tire_code]))

    p1, p2, p3, p4 = st.columns(4)
    front_pressure_psi = _render_tire_pressure_input(
        p1,
        "Front pressure",
        to_float(ctx.get("front_pressure_psi", base.get("front_pressure_psi", 32.0)), 32.0),
        key_base="tire_component_front_psi",
    )
    rear_pressure_psi = _render_tire_pressure_input(
        p2,
        "Rear pressure",
        to_float(ctx.get("rear_pressure_psi", base.get("rear_pressure_psi", 32.0)), 32.0),
        key_base="tire_component_rear_psi",
    )
    front_weight_distribution_pct = p3.number_input(
        "Front weight distribution [%]",
        value=float(ctx.get("weight_dist_fr_pct", base.get("weight_dist_fr_pct", 50.0)) or 50.0),
        min_value=0.0,
        max_value=100.0,
        step=0.5,
        format="%.1f",
        key="tire_component_weight_dist",
    )
    tire_improvement_pct = p4.number_input(
        "Tire improvement [%]",
        value=float(ctx.get("tire_improvement_pct", base.get("tire_improvement_pct", 0.0)) or 0.0),
        step=0.5,
        format="%.1f",
        key="tire_component_improvement",
    )

    tire_load_mass_basis = str(ctx.get("tire_load_mass_basis") or basis_default).strip().upper()
    st.caption(f"VDE calculation mass basis: `{tire_load_mass_basis}`")

    preview_result = None
    if st.button("Preview tire component", key="btn_tire_component_preview"):
        try:
            row_context = _build_tire_context_row(base_row)
            preview_result = preview_tire_roadload_from_row(
                row_context,
                {
                    "front_tire_id": front_tire_id,
                    "rear_tire_id": rear_tire_id,
                    "same_tire_front_rear": same_tire,
                    "front_pressure_psi": front_pressure_psi,
                    "rear_pressure_psi": rear_pressure_psi,
                    "front_weight_distribution_pct": front_weight_distribution_pct,
                    "tire_improvement_pct": tire_improvement_pct,
                    "tire_load_mass_basis": tire_load_mass_basis,
                    "mass_kg": ctx.get("mass_kg"),
                    "test_mass_kg": ctx.get("test_mass_kg"),
                    "inertia_class": ctx.get("inertia_class"),
                    "twc_kg": ctx.get("twc_kg"),
                    "etw_kg": ctx.get("etw_kg"),
                },
            )
            ctx["front_tire_id"] = front_tire_id
            ctx["rear_tire_id"] = rear_tire_id
            ctx["front_tire_test_code"] = front_tire_code
            ctx["rear_tire_test_code"] = rear_tire_code
            ctx["same_tire_front_rear"] = same_tire
            ctx["front_pressure_psi"] = front_pressure_psi
            ctx["rear_pressure_psi"] = rear_pressure_psi
            ctx["weight_dist_fr_pct"] = front_weight_distribution_pct
            ctx["tire_improvement_pct"] = tire_improvement_pct
            ctx["tire_load_mass_basis"] = tire_load_mass_basis
            ctx["tire_preview_result"] = preview_result
            ctx["include_tire_component"] = True
        except Exception as e:
            st.error(f"Failed to preview tire component: {e}")
            ctx["include_tire_component"] = False
            return

    if preview_result is None:
        cached = ctx.get("tire_preview_result")
        if isinstance(cached, dict) and cached.get("save_payload"):
            preview_result = cached

    if preview_result:
        _render_tire_preview_result(preview_result, saved_vde_id=saved_vde_id)
    else:
        ctx["include_tire_component"] = False
        st.info("Choose tires and preview the component to feed ABC_TOTAL from tire_db.")


def render_vehicle_basics_sidebar(*, reset_ctx):
    with st.sidebar:
        st.header("VDE Setup")
        st.caption("Scenario origin, roadload basis, mass setup, and technical configuration now live inside the main workflow.")
        st.radio(
            "Display units",
            ["Metric", "US customary"],
            horizontal=True,
            key="unit_system",
        )


def render_scenario_origin_section(*, reset_ctx):
    ctx = st.session_state.ctx
    mode_options = [
        "From baseline (editable)",
        "New line (manual / test)",
    ]
    labels = {
        "From baseline (editable)": "From baseline",
        "New line (manual / test)": "New manual / test scenario",
    }

    prev_mode = ctx.get("mode", "From baseline (editable)")
    if prev_mode not in mode_options:
        prev_mode = mode_options[0]
    ctx["mode"] = st.radio(
        "Scenario Origin",
        mode_options,
        horizontal=True,
        index=mode_options.index(prev_mode),
        key="scenario_origin_radio",
        format_func=lambda value: labels.get(value, value),
    )
    if ctx["mode"] != prev_mode:
        reset_ctx(preserve_meta=True)
        st.rerun()

    if ctx["mode"] == "From baseline (editable)":
        source_options = [
            "Inherit baseline ABC_TOTAL",
            "New test ABC_TOTAL",
        ]
        option_map = {
            "Inherit baseline ABC_TOTAL": "Baseline ABC",
            "New test ABC_TOTAL": "From test coastdown",
        }
    else:
        source_options = [
            "Insert new test final ABC & define all current/walk components",
            "Calculate final ABC from components",
        ]
        option_map = {
            "Insert new test final ABC & define all current/walk components": "From test coastdown",
            "Calculate final ABC from components": "Component Build-up",
        }

    reverse_map = {value: key for key, value in option_map.items()}
    current_source_ui = str(ctx.get("abc_total_source_ui") or "").strip()
    default_source_option = reverse_map.get(current_source_ui, source_options[0])
    selected_source_option = st.radio(
        "Roadload source path",
        source_options,
        horizontal=True,
        index=source_options.index(default_source_option),
        key="scenario_source_path_radio",
    )
    ctx["abc_total_source_ui"] = option_map[selected_source_option]

    selected = ctx["abc_total_source_ui"]
    if selected == "Baseline ABC":
        ctx["from_delta"] = "Deltas"
        baseline_id = ctx.get("vde_id_parent") or ctx.get("baseline_id")
        if not baseline_id:
            st.warning("Select a baseline scenario before using inherited baseline ABC_TOTAL.")
    elif selected == "From test coastdown":
        ctx["from_delta"] = "From test"
    else:
        ctx["from_delta"] = "Change Parameters"


def render_vehicle_meta_header():
    ctx = st.session_state.ctx
    _ensure_vehicle_metadata_defaults(ctx)

    leg_opts = ["WLTP", "EPA", "ABNT (Brazil)"]

    epa_classes = [
        "Unknown", "Two Seaters", "Minicompact Cars", "Subcompact Cars", "Compact Cars",
        "Midsize Cars", "Large Cars", "Small Station Wagons", "Midsize Station Wagons",
        "Small SUVs", "Standard SUVs", "Minivans", "Vans", "Small Pickup Trucks", "Standard Pickup Trucks",
    ]
    wltp_classes = ["Class 1 (<850 kg)", "Class 2 (850-1220 kg)", "Class 3 (>1220 kg)"]
    category_list = epa_classes if ctx["legislation"] == "EPA" else wltp_classes
    category_list_upper = [category.upper() for category in category_list]

    if ctx.get("category") not in category_list_upper:
        ctx["category"] = category_list_upper[0]

    elec_opts = ["ICE", "HEV", "PHEV", "BEV"]
    trans_opts = ["AT", "AMT", "CVT", "MT", "OT"]

    r1c1, r1c2, r1c3 = st.columns([1.1, 1.6, 1.3])
    ctx["legislation"] = r1c1.selectbox(
        "Legislation",
        leg_opts,
        index=leg_opts.index(ctx["legislation"]),
        key="hdr_leg",
    )
    category_list = epa_classes if ctx["legislation"] == "EPA" else wltp_classes
    category_list_upper = [category.upper() for category in category_list]
    if ctx.get("category") not in category_list_upper:
        ctx["category"] = category_list_upper[0]
    ctx["category"] = r1c2.selectbox(
        "Category",
        category_list_upper,
        index=category_list_upper.index(ctx["category"]),
        key="hdr_cat",
    )

    default_makes = [
        "Toyota", "Honda", "Nissan", "Mitsubishi", "Mazda", "Subaru", "Hyundai", "Kia",
        "Volkswagen", "Audi", "BMW", "Mercedes-Benz", "Porsche", "Peugeot", "Renault", "Citroen",
        "Fiat", "Alfa Romeo", "Volvo", "Jaguar", "Land Rover", "Skoda", "Seat", "Opel",
        "Ford", "Chevrolet", "Dodge", "Chrysler", "Jeep", "Ram", "Cadillac", "Buick", "GMC",
        "Lincoln", "Tesla", "Suzuki", "Mini", "Smart", "Lexus", "Infiniti", "Acura",
    ]
    default_makes_upper = [make.upper() for make in default_makes]
    try:
        ensure_db()
        makes_db = db_list_makes(ctx["legislation"], ctx["category"])
        makes_db = [make.upper() for make in makes_db]
    except Exception:
        makes_db = []

    merged_makes = list(dict.fromkeys(makes_db + [make for make in default_makes_upper if make not in makes_db]))
    if "OTHER (TYPE MANUALLY)" not in merged_makes:
        merged_makes.append("OTHER (TYPE MANUALLY)")

    selected_make = str(ctx.get("make", "")).upper()
    make_choice = r1c3.selectbox(
        "Make/Brand",
        merged_makes,
        index=(merged_makes.index(selected_make) if selected_make in merged_makes else 0),
        key="hdr_make_sel",
    )
    if make_choice == "OTHER (TYPE MANUALLY)":
        ctx["make"] = r1c3.text_input("Enter custom brand", value=ctx.get("make", ""), key="hdr_make_text").upper()
    else:
        ctx["make"] = make_choice

    r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns([1.6, 0.8, 1.0, 1.0, 1.2])
    ctx["model"] = r2c1.text_input("Model/Desc.", value=ctx.get("model", ""), key="hdr_model")
    ctx["year"] = r2c2.number_input("Year", 1900, 2100, int(ctx.get("year", 2024)), step=1, key="hdr_year")
    ctx["electrification"] = r2c3.selectbox(
        "Electrification",
        elec_opts,
        index=elec_opts.index(ctx.get("electrification", "ICE")),
        key="hdr_elec",
    )
    ctx["transmission_type"] = r2c4.selectbox(
        "Transmission",
        trans_opts,
        index=trans_opts.index(ctx.get("transmission_type", "AT")),
        key="hdr_trans",
    )
    ctx["notes"] = r2c5.text_input("Proposal / Scenario", value=ctx.get("notes", ""), key="hdr_notes")

    st.caption("Required for Vehicle Data = OK: legislation, category, make, model, year, electrification, and transmission.")


def _render_tire_component_editor(*, base_row: dict | None = None, saved_vde_id: int | None = None, tires_df=None):
    ctx = st.session_state.ctx
    basis = _roadload_basis_value(ctx)
    if basis != "Component Build-up":
        _render_tire_pressure_unit_toggle(key="tire_component_editor_pressure_unit_toggle")
    if basis == "Component Build-up":
        mode = _render_component_mode_selector("Tires")
        if mode == "Lookup from DB":
            render_tire_component_section(
                base_row=base_row,
                saved_vde_id=saved_vde_id,
                tires_df=tires_df,
                source_mode_override="Tire DB",
                show_source_selector=False,
            )
        else:
            render_tire_component_section(
                base_row=base_row,
                saved_vde_id=saved_vde_id,
                tires_df=tires_df,
                source_mode_override="Manual RR",
                show_source_selector=False,
            )
        return

    base = dict(base_row or {})
    baseline_current = _saved_tire_reference_from_row(base)
    current_reference = _resolve_tire_reference_from_ctx(
        "tire_current_reference",
        source_label="Scenario Current reference",
        preview_ctx_key="tire_current_reference_preview_result",
    ) or baseline_current
    walked_reference = _resolve_tire_reference_from_ctx(
        "tire_walked_reference",
        source_label="Scenario Walked reference",
        preview_ctx_key="tire_walked_reference_preview_result",
    )

    treatment_options = ["Keep inherited", "Apply tire change"]
    current_application = _normalize_tire_scenario_application(ctx.get("tire_scenario_application"))
    current_mode = str(ctx.get("component_mode_tires") or "")
    if current_mode == "Apply delta" or current_application != "Keep inherited":
        treatment_default = "Apply tire change"
    else:
        treatment_default = "Keep inherited"

    with st.container(border=True):
        _render_tire_editor_block_header(
            "Tire Scenario Treatment",
            "Choose whether tires remain inherited from the selected roadload basis or whether this scenario stages and applies a tire change.",
        )
        tire_treatment = st.radio(
            "Treatment",
            treatment_options,
            horizontal=True,
            index=treatment_options.index(treatment_default),
            key="tire_scenario_treatment_radio",
        )
        if tire_treatment == "Keep inherited":
            ctx["component_mode_tires"] = "Keep inherited"
            ctx["tire_scenario_application"] = "Keep inherited"
            ctx["delta_rr_N"] = 0.0
            ctx["tire_improvement_pct"] = 0.0
            ctx["tire_change_method"] = "Manual tire adjustment"
            if baseline_current:
                st.info("Inherited tire reference remains active. No tire adjustment is applied to this scenario.")
                _render_tire_reference_metrics(baseline_current, title="Inherited tire reference")
            else:
                st.info("No explicit tire reference is saved on the baseline, but this scenario is not applying an active tire adjustment.")
            return
        st.caption("A tire change is active for this scenario. Define the reference on the left and the active change on the right.")

    current_col, change_col = st.columns(2)

    with current_col:
        current_reference = _render_tire_reference_selector(
            title="Current Tire (Reference)",
            helper_text="Defines the current/base tire state used as reference for this scenario. Current is only required for Tire Improvement % and Current vs Walked comparison.",
            mode_key="tire_current_reference_mode",
            prefix="tire_current_reference",
            widget_prefix="tire_current_ref",
            source_label="Scenario Current reference",
            base_row=base_row,
            inherited_reference=baseline_current,
            save_to_baseline_label="Update baseline with this Current reference" if (saved_vde_id or base.get("id")) else None,
            baseline_target_id=int(saved_vde_id or base.get("id")) if (saved_vde_id or base.get("id")) else None,
        )

    crr_frac = float(to_float(ctx.get("crr1_frac_at_120kph"), 0.0) or 0.0)
    load_kN = _resolved_tire_load_kN(ctx)
    suggested_delta_a = None
    suggested_delta_b = None
    suggested_delta_c = None
    if current_reference and walked_reference:
        suggested_delta_a = float(to_float(walked_reference.get("A"), 0.0) or 0.0) - float(to_float(current_reference.get("A"), 0.0) or 0.0)
        suggested_delta_b = float(to_float(walked_reference.get("B"), 0.0) or 0.0) - float(to_float(current_reference.get("B"), 0.0) or 0.0)
        suggested_delta_c = float(to_float(walked_reference.get("C"), 0.0) or 0.0) - float(to_float(current_reference.get("C"), 0.0) or 0.0)

    equivalent_a = None
    equivalent_b = None
    equivalent_c = None

    with change_col:
        with st.container(border=True):
            _render_tire_editor_block_header(
                "Tire Change",
                "Choose how this scenario applies a tire change: direct manual adjustment or walked-vs-current comparison.",
            )
            change_method_options = ["Manual tire adjustment", "Walked tire comparison"]
            change_method_default = _default_tire_change_method(ctx)
            ctx["tire_change_method"] = st.radio(
                "Method",
                change_method_options,
                horizontal=False,
                index=change_method_options.index(change_method_default),
                key="tire_change_method_radio",
            )

            if ctx["tire_change_method"] == "Manual tire adjustment":
                change_intent_options = ["Scenario-only engineering adjustment", "Engineering target / supplier request"]
                current_change_intent = str(ctx.get("tire_manual_change_intent") or change_intent_options[0]).strip()
                if current_change_intent not in change_intent_options:
                    current_change_intent = change_intent_options[0]
                ctx["tire_manual_change_intent"] = st.radio(
                    "Change intent",
                    change_intent_options,
                    horizontal=False,
                    index=change_intent_options.index(current_change_intent),
                    key="tire_manual_change_intent_radio",
                )
                input_type_options = ["Delta final RRC", "Target final RRC", "Tire Improvement %"]
                input_type_default = _default_tire_manual_input_type(ctx)
                ctx["tire_manual_adjustment_input_type"] = st.radio(
                    "Input basis",
                    input_type_options,
                    horizontal=True,
                    index=input_type_options.index(input_type_default),
                    key="tire_manual_adjustment_input_type_radio",
                )

                current_rrc = None
                if current_reference and to_float(current_reference.get("rr_n_per_kn")) is not None:
                    current_rrc = float(to_float(current_reference.get("rr_n_per_kn"), 0.0) or 0.0)
                elif to_float(ctx.get("rrc_N_per_kN")) is not None:
                    current_rrc = float(to_float(ctx.get("rrc_N_per_kN"), 0.0) or 0.0)

                if ctx["tire_manual_adjustment_input_type"] == "Delta final RRC":
                    default_delta_rr = to_float(ctx.get("tire_manual_delta_rr_n_per_kn"))
                    if default_delta_rr is None and load_kN > 0:
                        default_delta_rr, _, _ = _equivalent_rr_from_abc(
                            to_float(ctx.get("delta_rr_N"), 0.0),
                            float(to_float(ctx.get("delta_rr_N"), 0.0) or 0.0) * (crr_frac / 120.0),
                            0.0,
                            load_kN=load_kN,
                        )
                    ctx["tire_manual_delta_rr_n_per_kn"] = quantity_input(
                        st,
                        "Delta final RRC",
                        to_float(default_delta_rr, 0.0),
                        "rrc",
                        key="tire_manual_delta_rr_input",
                        step_canonical=0.05,
                        format_str="%.3f",
                    )
                    md1, md2 = st.columns(2)
                    ctx["tire_manual_delta_rr_label"] = md1.text_input(
                        "Intended tire label",
                        value=str(ctx.get("tire_manual_delta_rr_label") or ""),
                        key="tire_manual_delta_rr_label_input",
                    )
                    with md2:
                        ctx["tire_manual_delta_rr_size_code"] = _render_tire_size_selector(
                            field_key="tire_manual_delta_rr_size_code",
                            label="Intended tire size",
                        )
                    md3, md4 = st.columns(2)
                    ctx["tire_manual_delta_rr_source"] = md3.text_input(
                        "Source / provenance",
                        value=str(ctx.get("tire_manual_delta_rr_source") or ""),
                        key="tire_manual_delta_rr_source_input",
                    )
                    ctx["tire_manual_delta_rr_notes"] = md4.text_input(
                        "Notes",
                        value=str(ctx.get("tire_manual_delta_rr_notes") or ""),
                        key="tire_manual_delta_rr_notes_input",
                    )
                    equivalent_a, equivalent_b, equivalent_c, epa_v_avg = _abc_from_final_rr_target(
                        ctx["tire_manual_delta_rr_n_per_kn"],
                        load_kN=load_kN,
                        crr_frac_120=crr_frac,
                    )

                    ctx["component_mode_tires"] = "Apply delta"
                    ctx["tire_scenario_application"] = "Manual Delta RR"
                    ctx["tire_delta_calculation_mode"] = "Manual delta RR"
                    ctx["tire_improvement_pct"] = 0.0
                    ctx["delta_rr_N"] = equivalent_a

                    if ctx["tire_manual_change_intent"] == "Engineering target / supplier request":
                        if current_rrc is None:
                            ctx["component_mode_tires"] = "Keep inherited"
                            ctx["delta_rr_N"] = 0.0
                            st.warning("Engineering target / supplier request needs a Current Tire reference so the requested delta can be anchored to a reference final RRC.")
                        else:
                            target_rrc = current_rrc + float(ctx["tire_manual_delta_rr_n_per_kn"])
                            t1, t2, t3 = st.columns(3)
                            t1.metric("Reference final RRC", format_quantity(current_rrc, "rrc", format_str="%.3f"))
                            t2.metric("Target final RRC", format_quantity(target_rrc, "rrc", format_str="%.3f"))
                            t3.metric("Delta final RRC", format_quantity(ctx["tire_manual_delta_rr_n_per_kn"], "rrc", format_str="%.3f"))
                    else:
                        st.caption(
                            f"Resolved with VDE tire calculation load = {load_kN:.2f} kN. "
                            "The workflow stores the applied effect internally as equivalent delta A/B for the current EPA-aligned final RRC target."
                        )
                        if epa_v_avg is not None:
                            st.caption(f"EPA v_avg used for delta decomposition: {epa_v_avg:.2f} kph")
                elif ctx["tire_manual_adjustment_input_type"] == "Target final RRC":
                    ctx["tire_manual_target_rr_n_per_kn"] = quantity_input(
                        st,
                        "Target final RRC",
                        to_float(ctx.get("tire_manual_target_rr_n_per_kn"), current_rrc or 0.0),
                        "rrc",
                        key="tire_manual_target_rr_input",
                        step_canonical=0.05,
                        format_str="%.3f",
                    )
                    md1, md2 = st.columns(2)
                    ctx["tire_manual_delta_rr_label"] = md1.text_input(
                        "Intended tire label",
                        value=str(ctx.get("tire_manual_delta_rr_label") or ""),
                        key="tire_manual_target_label_input",
                    )
                    with md2:
                        ctx["tire_manual_delta_rr_size_code"] = _render_tire_size_selector(
                            field_key="tire_manual_delta_rr_size_code",
                            label="Intended tire size",
                        )
                    md3, md4 = st.columns(2)
                    ctx["tire_manual_delta_rr_source"] = md3.text_input(
                        "Source / provenance",
                        value=str(ctx.get("tire_manual_delta_rr_source") or ""),
                        key="tire_manual_target_source_input",
                    )
                    ctx["tire_manual_delta_rr_notes"] = md4.text_input(
                        "Notes",
                        value=str(ctx.get("tire_manual_delta_rr_notes") or ""),
                        key="tire_manual_target_notes_input",
                    )
                    ctx["tire_improvement_pct"] = 0.0
                    if current_rrc is None:
                        ctx["component_mode_tires"] = "Keep inherited"
                        ctx["delta_rr_N"] = 0.0
                        st.warning("Target final RRC needs a Current Tire reference so the scenario can derive a delta from the target.")
                    else:
                        target_rrc = float(to_float(ctx.get("tire_manual_target_rr_n_per_kn"), 0.0) or 0.0)
                        delta_rr_n_per_kn = target_rrc - current_rrc
                        ctx["tire_manual_delta_rr_n_per_kn"] = delta_rr_n_per_kn
                        equivalent_a, equivalent_b, equivalent_c, epa_v_avg = _abc_from_final_rr_target(
                            delta_rr_n_per_kn,
                            load_kN=load_kN,
                            crr_frac_120=crr_frac,
                        )
                        ctx["component_mode_tires"] = "Apply delta"
                        ctx["tire_scenario_application"] = "Manual Delta RR"
                        ctx["tire_delta_calculation_mode"] = "Target RRC"
                        ctx["delta_rr_N"] = equivalent_a
                        t1, t2, t3 = st.columns(3)
                        t1.metric("Reference final RRC", format_quantity(current_rrc, "rrc", format_str="%.3f"))
                        t2.metric("Target final RRC", format_quantity(target_rrc, "rrc", format_str="%.3f"))
                        t3.metric("Delta final RRC", format_quantity(delta_rr_n_per_kn, "rrc", format_str="%.3f"))
                        if epa_v_avg is not None:
                            st.caption(f"EPA v_avg used for target-to-delta decomposition: {epa_v_avg:.2f} kph")
                else:
                    ctx["tire_improvement_pct"] = st.number_input(
                        "Tire Improvement [%]",
                        value=float(ctx.get("tire_improvement_pct", 0.0) or 0.0),
                        step=0.5,
                        format="%.1f",
                        key="tire_improvement_pct_apply_input",
                    )
                    ctx["tire_scenario_application"] = "Tire Improvement %"
                    ctx["tire_delta_calculation_mode"] = "Tire Improvement %"
                    if current_reference:
                        factor = 1.0 - (float(ctx["tire_improvement_pct"]) / 100.0)
                        improved_a = float(to_float(current_reference.get("A"), 0.0) or 0.0) * factor
                        improved_b = float(to_float(current_reference.get("B"), 0.0) or 0.0) * factor
                        improved_c = float(to_float(current_reference.get("C"), 0.0) or 0.0) * factor
                        equivalent_a = improved_a - float(to_float(current_reference.get("A"), 0.0) or 0.0)
                        equivalent_b = improved_b - float(to_float(current_reference.get("B"), 0.0) or 0.0)
                        equivalent_c = improved_c - float(to_float(current_reference.get("C"), 0.0) or 0.0)
                        ctx["component_mode_tires"] = "Apply delta"
                        ctx["delta_rr_N"] = equivalent_a
                        if current_rrc is not None:
                            target_rrc = current_rrc * factor
                            t1, t2, t3 = st.columns(3)
                            t1.metric("Reference final RRC", format_quantity(current_rrc, "rrc", format_str="%.3f"))
                            t2.metric("Target final RRC", format_quantity(target_rrc, "rrc", format_str="%.3f"))
                            t3.metric("Delta final RRC", format_quantity(target_rrc - current_rrc, "rrc", format_str="%.3f"))
                        else:
                            st.caption("Equivalent ABC delta is derived from the Current Tire reference for traceability.")
                    else:
                        ctx["component_mode_tires"] = "Keep inherited"
                        ctx["delta_rr_N"] = 0.0
                        st.warning("Current Tire is optional for manual setup, but it is still needed here to derive an equivalent ABC delta from Tire Improvement %.")
            else:
                walked_reference = _render_tire_reference_selector(
                    title="Walked Tire (Candidate)",
                    helper_text="Stage the walked / proposed tire here. Without a Current reference the candidate can still be registered, but no comparison delta is calculated.",
                    mode_key="tire_walked_reference_mode",
                    prefix="tire_walked_reference",
                    widget_prefix="tire_walked_ref",
                    source_label="Walked tire reference",
                    base_row=base_row,
                    allow_unset=True,
                )

                ctx["tire_scenario_application"] = "Walked tire comparison"
                ctx["tire_delta_calculation_mode"] = "Use Current vs Walked delta"
                ctx["tire_improvement_pct"] = 0.0

                if not current_reference:
                    ctx["component_mode_tires"] = "Keep inherited"
                    ctx["delta_rr_N"] = 0.0
                    st.warning(
                        "Current Tire is required for Walked tire comparison.\n"
                        "Without a Current reference, no comparison delta is calculated."
                    )
                elif walked_reference:
                    suggested_delta_a = float(to_float(walked_reference.get("A"), 0.0) or 0.0) - float(to_float(current_reference.get("A"), 0.0) or 0.0)
                    suggested_delta_b = float(to_float(walked_reference.get("B"), 0.0) or 0.0) - float(to_float(current_reference.get("B"), 0.0) or 0.0)
                    suggested_delta_c = float(to_float(walked_reference.get("C"), 0.0) or 0.0) - float(to_float(current_reference.get("C"), 0.0) or 0.0)
                    equivalent_a = suggested_delta_a
                    equivalent_b = suggested_delta_b
                    equivalent_c = suggested_delta_c
                    ctx["component_mode_tires"] = "Apply delta"
                    ctx["delta_rr_N"] = suggested_delta_a
                    st.caption("Walked comparison is active. The scenario uses the resolved Current vs Walked delta.")
                else:
                    ctx["component_mode_tires"] = "Keep inherited"
                    ctx["delta_rr_N"] = 0.0
                    st.info("Stage a Walked Tire to enable the comparison effect on this scenario.")

    with st.container(border=True):
        _render_tire_editor_block_header(
            "Applied to scenario",
            "Read-only summary of the tire effect currently resolved for this scenario.",
        )
        applied_method = _normalize_tire_scenario_application(ctx.get("tire_scenario_application"))
        if applied_method == "Keep inherited" or ctx.get("component_mode_tires") == "Keep inherited":
            if applied_method == "Walked tire comparison" and walked_reference and not current_reference:
                st.info("Method: Walked tire comparison. Effect is pending because Current Tire is still missing.")
            elif applied_method == "Walked tire comparison" and not walked_reference:
                st.info("Method: Walked tire comparison. Effect is pending until a Walked Tire is selected.")
            elif applied_method == "Tire Improvement %" and float(to_float(ctx.get("tire_improvement_pct"), 0.0) or 0.0) > 0.0 and not current_reference:
                st.info("Method: Tire Improvement %. Effect is pending until a Current Tire reference is available for equivalent ABC derivation.")
            else:
                st.info("Method: inherited. No tire adjustment is currently affecting the scenario.")
        else:
            if applied_method == "Manual Delta RR":
                effect_text = format_quantity(ctx.get("tire_manual_delta_rr_n_per_kn"), "rrc", format_str="%.3f")
            elif applied_method == "Tire Improvement %":
                effect_text = f"{float(to_float(ctx.get('tire_improvement_pct'), 0.0) or 0.0):.1f}%"
            else:
                effect_text = "Current vs Walked"
            data_role = "engineering_target" if str(ctx.get("tire_manual_change_intent") or "").strip() == "Engineering target / supplier request" else "scenario_assumption"

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Method", _display_tire_application_method(applied_method))
            c2.metric("Effect applied", effect_text)
            c3.metric("Status", "Applied")
            c4.metric("Role", _display_data_role(data_role))

            reference_rrc = to_float((current_reference or {}).get("rr_n_per_kn"), to_float(ctx.get("rrc_N_per_kN")))
            target_rrc = to_float(ctx.get("tire_manual_target_rr_n_per_kn"))
            summary_target_text = None
            if applied_method == "Walked tire comparison" and walked_reference:
                summary_target_text = _tire_reference_brief(walked_reference)
            elif target_rrc is not None and abs(float(target_rrc or 0.0)) > 0.0:
                summary_target_text = format_quantity(target_rrc, "rrc", unavailable="-", format_str="%.3f")

            top_summary_parts = [
                f"Reference: {_tire_reference_brief(current_reference)}" if current_reference else None,
                f"Target: {summary_target_text}" if summary_target_text else None,
                f"Intent: {str(ctx.get('tire_manual_change_intent') or '-')}",
            ]
            st.caption(" | ".join(part for part in top_summary_parts if part))

            if equivalent_a is not None:
                e1, e2, e3 = st.columns(3)
                e1.metric(
                    "Equivalent delta ABC",
                    _compact_abc({"A": equivalent_a, "B": equivalent_b or 0.0, "C": equivalent_c or 0.0}),
                )
                e2.metric("Reference final RRC", format_quantity(reference_rrc, "rrc", unavailable="-", format_str="%.3f"))
                e3.metric(
                    "Target / compared final RRC",
                    (
                        format_quantity(target_rrc, "rrc", unavailable="-", format_str="%.3f")
                        if applied_method != "Walked tire comparison"
                        else format_quantity((walked_reference or {}).get("rr_n_per_kn"), "rrc", unavailable="-", format_str="%.3f")
                    ),
                )

            with st.expander("Reference and comparison details", expanded=False):
                ref1, ref2 = st.columns(2)
                with ref1:
                    if current_reference:
                        _render_tire_reference_metrics(current_reference, title="Reference used")
                with ref2:
                    if walked_reference and applied_method == "Walked tire comparison":
                        _render_tire_reference_metrics(walked_reference, title="Candidate used")
                    elif applied_method != "Walked tire comparison":
                        st.caption("No comparison candidate is active for this method.")
                st.caption("Equivalent ABC remains available for traceability. The inherited workflow still consumes the resolved tire effect through the existing delta-based calculation path.")
        _sync_tire_scenario_snapshot_fields(
            applied_method=applied_method,
            current_reference=current_reference,
            walked_reference=walked_reference,
            equivalent_a=equivalent_a,
            equivalent_b=equivalent_b,
            equivalent_c=equivalent_c,
        )


def _render_aero_component_editor(*, base_row: dict | None = None):
    ctx = st.session_state.ctx
    basis = _roadload_basis_value(ctx)
    if basis == "Component Build-up":
        with st.container(border=True):
            _render_tire_editor_block_header(
                "Reference Aero",
                "In component build-up, aerodynamics contributes as an absolute source instead of a delta over inherited roadload.",
            )
            st.caption("Reference status: `Absolute source`")
            render_aero_section(prefill=base_row)
        return
    base = dict(base_row or {})
    reference_cda = to_float(ctx.get("aero_reference_cda_override"), to_float(base.get("cda_m2")))
    reference_defined = reference_cda is not None and abs(float(reference_cda)) > 0.0
    baseline_target_id = int(base.get("id") or ctx.get("vde_id_parent") or ctx.get("baseline_id") or 0) or None
    reference_status = "Scenario override" if to_float(ctx.get("aero_reference_cda_override")) is not None else "Inherited"
    candidate_cda = None

    with st.container(border=True):
        _render_tire_editor_block_header(
            "Reference Aero",
            "Defines the current aerodynamic reference used by this scenario when roadload is inherited from baseline or coastdown.",
        )
        if reference_cda is not None:
            _render_scalar_reference_metric(
                label="Reference CdA",
                value=reference_cda,
                quantity="cda",
                status=reference_status,
            )
        else:
            st.warning("Reference status: Missing. No explicit baseline CdA is available for derived comparison.")
        if reference_cda is not None and not reference_defined:
            st.warning("Reference CdA is currently zero. If that is only a missing baseline value, you can correct the baseline row here.")
        if baseline_target_id:
            with st.expander("Set / update baseline reference aero", expanded=not reference_defined):
                baseline_cda_default = to_float(ctx.get("aero_baseline_reference_cda"), reference_cda)
                baseline_cda_value = quantity_input(
                    st,
                    "Baseline reference CdA",
                    to_float(baseline_cda_default, 0.0),
                    "cda",
                    key="aero_baseline_reference_cda_input",
                    step_canonical=0.001,
                    format_str="%.3f",
                )
                ctx["aero_baseline_reference_cda"] = baseline_cda_value
                update_baseline = st.checkbox(
                    "Also update selected baseline row in VDE DB",
                    value=not reference_defined,
                    key="aero_update_baseline_checkbox",
                )
                if st.button("Apply reference aero", key="aero_apply_reference_button"):
                    ctx["cda_m2"] = float(baseline_cda_value)
                    ctx["aero_reference_cda_override"] = float(baseline_cda_value)
                    reference_cda = float(baseline_cda_value)
                    reference_defined = abs(reference_cda) > 0.0
                    if update_baseline:
                        try:
                            update_vde_snapshot(int(baseline_target_id), {"cda_m2": float(baseline_cda_value)})
                            _sync_baseline_cda_in_ctx(float(baseline_cda_value), baseline_target_id=baseline_target_id)
                            ctx["aero_reference_cda_override"] = None
                            st.success(f"Baseline reference aero updated on VDE id={baseline_target_id}.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not update baseline reference aero: {exc}")
                    else:
                        st.success("Reference aero staged only for the current scenario context.")

    with st.container(border=True):
        _render_tire_editor_block_header(
            "Proposed / Applied Aero Change",
            "Define the active aerodynamic change as a manual delta or a proposed-vs-reference comparison.",
        )
        new_cda_label = "New CdA"
        aero_change_options = [
            "Keep inherited",
            new_cda_label,
            "Manual delta CdA",
        ]
        current_candidate_mode = str(ctx.get("aero_candidate_mode") or "Not set")
        current_calc_mode = str(ctx.get("aero_calculation_mode") or "Inherited")
        current_component_mode = str(ctx.get("component_mode_aerodynamics") or "Keep inherited")
        if current_component_mode != "Apply delta" or current_calc_mode == "Inherited":
            current_aero_change = "Keep inherited"
        elif current_candidate_mode == "Manual proposal" or current_calc_mode == "Use candidate vs reference":
            current_aero_change = new_cda_label
        else:
            current_aero_change = "Manual delta CdA"

        selected_aero_change = st.radio(
            "Aero change method",
            aero_change_options,
            horizontal=True,
            index=aero_change_options.index(current_aero_change),
            key="aero_change_mode_radio",
        )

        if selected_aero_change == "Keep inherited":
            ctx["aero_candidate_mode"] = "Not set"
            ctx["aero_calculation_mode"] = "Inherited"
            ctx["component_mode_aerodynamics"] = "Keep inherited"
            ctx["delta_aero_cdA"] = 0.0
            st.info("Calculation status: `Inherited`")
        elif selected_aero_change == new_cda_label:
            ctx["aero_candidate_mode"] = "Manual proposal"
            candidate_cda = _render_candidate_cda_input(
                key_prefix="aero_candidate",
                title="New CdA",
            )
            if candidate_cda is not None and reference_cda is not None:
                suggested_delta_cda = float(candidate_cda) - float(reference_cda)
                ctx["aero_calculation_mode"] = "Use candidate vs reference"
                ctx["component_mode_aerodynamics"] = "Apply delta"
                ctx["delta_aero_cdA"] = suggested_delta_cda
                quantity_metric(st, "Suggested delta CdA", suggested_delta_cda, "cda", format_str="%.3f")
                st.success("Derived change from Proposed Aero vs Reference Aero is active.")
            else:
                ctx["aero_calculation_mode"] = "Use candidate vs reference"
                ctx["component_mode_aerodynamics"] = "Keep inherited"
                ctx["delta_aero_cdA"] = 0.0
                if candidate_cda is None:
                    st.caption("No proposed CdA is staged right now.")
                else:
                    st.warning("Reference Aero is required before a proposed CdA can generate an active delta.")
        else:
            ctx["aero_candidate_mode"] = "Not set"
            ctx["aero_calculation_mode"] = "Manual delta CdA"
            ctx["component_mode_aerodynamics"] = "Apply delta"
            ctx["delta_aero_cdA"] = quantity_input(
                st,
                "Aerodynamics delta CdA",
                to_float(ctx.get("delta_aero_cdA"), 0.0),
                "cda",
                key="aero_delta_cda_input",
                step_canonical=0.001,
                format_str="%.3f",
            )
        st.caption("The inherited/coastdown workflow currently applies aerodynamics through explicit delta CdA only.")
    applied_method = str(ctx.get("aero_calculation_mode") or "Inherited")
    delta_cda = float(to_float(ctx.get("delta_aero_cdA"), 0.0) or 0.0)
    resolved_new_cda = None
    if applied_method == "Use candidate vs reference":
        resolved_new_cda = to_float(ctx.get("aero_candidate_cda"))
    elif reference_cda is not None and ctx.get("component_mode_aerodynamics") == "Apply delta":
        resolved_new_cda = float(reference_cda) + float(delta_cda)

    _render_applied_effect_block(
        title="Applied Aero State",
        caption="Read-only summary of the aerodynamic effect currently resolved for this scenario.",
        inherited_message="Method: inherited. No aerodynamic change is currently affecting the scenario.",
        metrics=[
            ("Method", applied_method),
            ("Effect applied", format_quantity(delta_cda, "cda", format_str="%.3f")),
            ("New CdA", format_quantity(resolved_new_cda, "cda", format_str="%.3f", unavailable="Pending")),
            ("Status", "Applied" if abs(delta_cda) > 0.0 or applied_method == 'Use candidate vs reference' else "Pending"),
        ] if ctx.get("component_mode_aerodynamics") == "Apply delta" else None,
    )


def render_vehicle_aero_section(*, base_row: dict | None = None):
    _render_aero_component_editor(base_row=base_row)


def _render_brake_component_editor(*, base_row: dict | None = None):
    ctx = st.session_state.ctx
    basis = _roadload_basis_value(ctx)
    if basis == "Component Build-up":
        with st.container(border=True):
            _render_tire_editor_block_header(
                "Reference Brake Drag",
                "In component build-up, brake drag contributes as an absolute source instead of an inherited delta.",
            )
            st.caption("Reference status: `Absolute source`")
            render_brake_section(prefill=base_row)
        return
    base = dict(base_row or {})
    ref_a = to_float(ctx.get("brake_reference_A_override"), base.get("brake_A_coef_N"))
    ref_b = to_float(ctx.get("brake_reference_B_override"), base.get("brake_B_Npkph"))
    ref_c = to_float(ctx.get("brake_reference_C_override"), base.get("brake_C_coef_Npkph2"))
    baseline_target_id = int(base.get("id") or ctx.get("vde_id_parent") or ctx.get("baseline_id") or 0) or None
    reference_defined = any(abs(float(to_float(value, 0.0) or 0.0)) > 0.0 for value in (ref_a, ref_b, ref_c))
    reference_status = "Scenario override" if any(ctx.get(key) is not None for key in ("brake_reference_A_override", "brake_reference_B_override", "brake_reference_C_override")) else "Inherited"

    with st.container(border=True):
        _render_tire_editor_block_header(
            "Reference Brake Drag",
            "Defines the current brake drag reference used when roadload is inherited from baseline or coastdown.",
        )
        if any(value is not None for value in (ref_a, ref_b, ref_c)):
            _render_abc_reference_metrics(
                title="Reference brake drag",
                a_value=ref_a,
                b_value=ref_b,
                c_value=ref_c,
                status=reference_status,
                caption="Current inherited workflow only consumes brake delta through A when baseline/coastdown is active.",
            )
        else:
            st.warning("Reference status: Missing. No explicit baseline brake split is available for derived comparison.")
        if baseline_target_id:
            with st.expander("Set / update baseline reference brake drag", expanded=not reference_defined):
                b1, b2, b3 = st.columns(3)
                baseline_ref_a = quantity_input(b1, "Baseline brake A", to_float(ctx.get("brake_baseline_reference_A"), ref_a or 0.0), "force", key="brake_baseline_reference_A_input", step_canonical=0.1, format_str="%.3f")
                baseline_ref_b = quantity_input(b2, "Baseline brake B", to_float(ctx.get("brake_baseline_reference_B"), ref_b or 0.0), "force_per_speed", key="brake_baseline_reference_B_input", step_canonical=0.0001, format_str="%.6f")
                baseline_ref_c = quantity_input(b3, "Baseline brake C", to_float(ctx.get("brake_baseline_reference_C"), ref_c or 0.0), "force_per_speed_squared", key="brake_baseline_reference_C_input", step_canonical=0.000001, format_str="%.8f")
                ctx["brake_baseline_reference_A"] = baseline_ref_a
                ctx["brake_baseline_reference_B"] = baseline_ref_b
                ctx["brake_baseline_reference_C"] = baseline_ref_c
                update_baseline = st.checkbox(
                    "Also update selected baseline row in VDE DB",
                    value=not reference_defined,
                    key="brake_update_baseline_checkbox",
                )
                if st.button("Apply reference brake drag", key="brake_apply_reference_button"):
                    ctx["brake_reference_A_override"] = float(baseline_ref_a)
                    ctx["brake_reference_B_override"] = float(baseline_ref_b)
                    ctx["brake_reference_C_override"] = float(baseline_ref_c)
                    if update_baseline:
                        try:
                            update_vde_snapshot(
                                int(baseline_target_id),
                                {
                                    "brake_A_coef_N": float(baseline_ref_a),
                                    "brake_B_Npkph": float(baseline_ref_b),
                                    "brake_C_coef_Npkph2": float(baseline_ref_c),
                                },
                            )
                            _sync_baseline_abc_reference_in_ctx(
                                a_key="brake_A_coef_N",
                                b_key="brake_B_Npkph",
                                c_key="brake_C_coef_Npkph2",
                                a_value=float(baseline_ref_a),
                                b_value=float(baseline_ref_b),
                                c_value=float(baseline_ref_c),
                                baseline_target_id=baseline_target_id,
                            )
                            ctx["brake_reference_A_override"] = None
                            ctx["brake_reference_B_override"] = None
                            ctx["brake_reference_C_override"] = None
                            st.success(f"Baseline reference brake drag updated on VDE id={baseline_target_id}.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not update baseline reference brake drag: {exc}")
                    else:
                        st.success("Reference brake drag staged only for the current scenario context.")

    with st.container(border=True):
        _render_tire_editor_block_header(
            "Change",
            "Choose whether brake drag stays inherited, is calculated from a new brake-drag proposal, or uses a direct delta A.",
        )
        new_brake_label = "New brake drag"
        brake_change_options = ["Keep inherited", new_brake_label, "Manual delta A"]
        current_component_mode = str(ctx.get("component_mode_brakes") or "Keep inherited")
        current_candidate_mode = str(ctx.get("brake_candidate_mode") or "Not set")
        current_calc_mode = str(ctx.get("brake_calculation_mode") or "Inherited")
        if current_component_mode != "Apply delta" or current_calc_mode == "Inherited":
            current_brake_change = "Keep inherited"
        elif current_candidate_mode == "Manual proposal" or current_calc_mode == "Use candidate vs reference":
            current_brake_change = new_brake_label
        else:
            current_brake_change = "Manual delta A"
        if current_brake_change not in brake_change_options:
            current_brake_change = "Keep inherited"

        selected_brake_change = st.radio(
            "Brake drag change",
            brake_change_options,
            horizontal=True,
            index=brake_change_options.index(current_brake_change),
            key="brake_change_mode_radio",
        )

        if selected_brake_change == "Keep inherited":
            ctx["brake_candidate_mode"] = "Not set"
            ctx["brake_calculation_mode"] = "Inherited"
            ctx["component_mode_brakes"] = "Keep inherited"
            ctx["delta_brake_N"] = 0.0
            st.info("Calculation status: inherited. No brake-drag change is currently applied.")
        elif selected_brake_change == new_brake_label:
            ctx["brake_candidate_mode"] = "Manual proposal"
            candidate = _render_candidate_abc_inputs(
                key_prefix="brake_candidate",
                title="New brake drag ABC",
            )
            ctx["brake_calculation_mode"] = "Use candidate vs reference"
            if candidate and ref_a is not None:
                suggested_delta_a = float(candidate.get("A") or 0.0) - float(ref_a or 0.0)
                ctx["component_mode_brakes"] = "Apply delta"
                ctx["delta_brake_N"] = suggested_delta_a
                quantity_metric(st, "Applied brake delta A", suggested_delta_a, "force", format_str="%.3f")
                st.success("Derived change from new brake drag vs reference is active.")
            elif candidate and ref_a is None:
                ctx["component_mode_brakes"] = "Keep inherited"
                ctx["delta_brake_N"] = 0.0
                st.warning("New brake drag is staged, but derived brake delta needs an explicit Reference Brake Drag. Use manual delta A instead.")
            else:
                ctx["component_mode_brakes"] = "Keep inherited"
                ctx["delta_brake_N"] = 0.0
                st.caption("New brake drag is not fully defined yet.")
        else:
            ctx["brake_candidate_mode"] = "Not set"
            ctx["brake_calculation_mode"] = "Manual delta A"
            ctx["component_mode_brakes"] = "Apply delta"
            ctx["delta_brake_N"] = quantity_input(
                st,
                "Brakes delta A",
                to_float(ctx.get("delta_brake_N"), 0.0),
                "force",
                key="brake_delta_input",
                step_canonical=0.1,
                format_str="%.3f",
            )
        st.caption("Current inherited/coastdown workflow uses brake drag adjustment through A only.")
    delta_brake = float(to_float(ctx.get("delta_brake_N"), 0.0) or 0.0)
    _render_applied_effect_block(
        title="Applied effect",
        caption="Read-only summary of the brake-drag effect currently resolved for this scenario.",
        inherited_message="Method: inherited. No brake-drag change is currently affecting the scenario.",
        metrics=[
            ("Method", str(ctx.get("brake_calculation_mode") or "Inherited")),
            ("Effect applied", format_quantity(delta_brake, "force", format_str="%.3f")),
            ("Status", "Applied" if abs(delta_brake) > 0.0 or str(ctx.get("brake_calculation_mode") or "") == "Use candidate vs reference" else "Pending"),
        ] if ctx.get("component_mode_brakes") == "Apply delta" else None,
    )


def _render_parasitic_component_editor(*, base_row: dict | None = None):
    ctx = st.session_state.ctx
    basis = _roadload_basis_value(ctx)
    if basis == "Component Build-up":
        with st.container(border=True):
            _render_tire_editor_block_header(
                "Reference Losses",
                "In component build-up, parasitic losses contribute as an absolute source instead of an inherited delta.",
            )
            st.caption("Reference status: `Absolute source`")
            render_parasitic_section(prefill=base_row)
        return
    base = dict(base_row or {})
    ref_a = to_float(ctx.get("parasitic_reference_A_override"), base.get("parasitic_A_coef_N"))
    ref_b = to_float(ctx.get("parasitic_reference_B_override"), base.get("parasitic_B_Npkph"))
    ref_c = to_float(ctx.get("parasitic_reference_C_override"), base.get("parasitic_C_coef_Npkph2"))
    baseline_target_id = int(base.get("id") or ctx.get("vde_id_parent") or ctx.get("baseline_id") or 0) or None
    reference_defined = any(abs(float(to_float(value, 0.0) or 0.0)) > 0.0 for value in (ref_a, ref_b, ref_c))
    reference_status = "Scenario override" if any(ctx.get(key) is not None for key in ("parasitic_reference_A_override", "parasitic_reference_B_override", "parasitic_reference_C_override")) else "Inherited"

    with st.container(border=True):
        _render_tire_editor_block_header(
            "Reference Losses",
            "Defines the current parasitic-loss reference used when roadload is inherited from baseline or coastdown.",
        )
        if any(value is not None for value in (ref_a, ref_b, ref_c)):
            _render_abc_reference_metrics(
                title="Reference parasitic losses",
                a_value=ref_a,
                b_value=ref_b,
                c_value=ref_c,
                status=reference_status,
                caption="Current inherited/coastdown workflow only consumes parasitic adjustment through A.",
            )
        else:
            st.warning("Reference status: Missing. No explicit baseline parasitic split is available for derived comparison.")
        if baseline_target_id:
            with st.expander("Set / update baseline reference parasitic losses", expanded=not reference_defined):
                p1, p2, p3 = st.columns(3)
                baseline_ref_a = quantity_input(p1, "Baseline parasitic A", to_float(ctx.get("parasitic_baseline_reference_A"), ref_a or 0.0), "force", key="parasitic_baseline_reference_A_input", step_canonical=0.1, format_str="%.3f")
                baseline_ref_b = quantity_input(p2, "Baseline parasitic B", to_float(ctx.get("parasitic_baseline_reference_B"), ref_b or 0.0), "force_per_speed", key="parasitic_baseline_reference_B_input", step_canonical=0.0001, format_str="%.6f")
                baseline_ref_c = quantity_input(p3, "Baseline parasitic C", to_float(ctx.get("parasitic_baseline_reference_C"), ref_c or 0.0), "force_per_speed_squared", key="parasitic_baseline_reference_C_input", step_canonical=0.000001, format_str="%.8f")
                ctx["parasitic_baseline_reference_A"] = baseline_ref_a
                ctx["parasitic_baseline_reference_B"] = baseline_ref_b
                ctx["parasitic_baseline_reference_C"] = baseline_ref_c
                update_baseline = st.checkbox(
                    "Also update selected baseline row in VDE DB",
                    value=not reference_defined,
                    key="parasitic_update_baseline_checkbox",
                )
                if st.button("Apply reference parasitic losses", key="parasitic_apply_reference_button"):
                    ctx["parasitic_reference_A_override"] = float(baseline_ref_a)
                    ctx["parasitic_reference_B_override"] = float(baseline_ref_b)
                    ctx["parasitic_reference_C_override"] = float(baseline_ref_c)
                    if update_baseline:
                        try:
                            update_vde_snapshot(
                                int(baseline_target_id),
                                {
                                    "parasitic_A_coef_N": float(baseline_ref_a),
                                    "parasitic_B_Npkph": float(baseline_ref_b),
                                    "parasitic_C_coef_Npkph2": float(baseline_ref_c),
                                },
                            )
                            _sync_baseline_abc_reference_in_ctx(
                                a_key="parasitic_A_coef_N",
                                b_key="parasitic_B_Npkph",
                                c_key="parasitic_C_coef_Npkph2",
                                a_value=float(baseline_ref_a),
                                b_value=float(baseline_ref_b),
                                c_value=float(baseline_ref_c),
                                baseline_target_id=baseline_target_id,
                            )
                            ctx["parasitic_reference_A_override"] = None
                            ctx["parasitic_reference_B_override"] = None
                            ctx["parasitic_reference_C_override"] = None
                            st.success(f"Baseline reference parasitic losses updated on VDE id={baseline_target_id}.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not update baseline reference parasitic losses: {exc}")
                    else:
                        st.success("Reference parasitic losses staged only for the current scenario context.")

    with st.container(border=True):
        _render_tire_editor_block_header(
            "Change",
            "Choose whether parasitic losses stay inherited, are calculated from a new proposal, or use a direct delta A.",
        )
        new_parasitic_label = "New parasitic losses"
        parasitic_change_options = ["Keep inherited", new_parasitic_label, "Manual delta A"]
        current_component_mode = str(ctx.get("component_mode_parasitics_hubs_axle") or "Keep inherited")
        current_candidate_mode = str(ctx.get("parasitic_candidate_mode") or "Not set")
        current_calc_mode = str(ctx.get("parasitic_calculation_mode") or "Inherited")
        if current_component_mode != "Apply delta" or current_calc_mode == "Inherited":
            current_parasitic_change = "Keep inherited"
        elif current_candidate_mode == "Manual proposal" or current_calc_mode == "Use candidate vs reference":
            current_parasitic_change = new_parasitic_label
        else:
            current_parasitic_change = "Manual delta A"
        if current_parasitic_change not in parasitic_change_options:
            current_parasitic_change = "Keep inherited"

        selected_parasitic_change = st.radio(
            "Parasitic losses change",
            parasitic_change_options,
            horizontal=True,
            index=parasitic_change_options.index(current_parasitic_change),
            key="parasitic_change_mode_radio",
        )

        if selected_parasitic_change == "Keep inherited":
            ctx["parasitic_candidate_mode"] = "Not set"
            ctx["parasitic_calculation_mode"] = "Inherited"
            ctx["component_mode_parasitics_hubs_axle"] = "Keep inherited"
            ctx["delta_parasitics_N"] = 0.0
            st.info("Calculation status: inherited. No parasitic-loss change is currently applied.")
        elif selected_parasitic_change == new_parasitic_label:
            ctx["parasitic_candidate_mode"] = "Manual proposal"
            candidate = _render_candidate_abc_inputs(
                key_prefix="parasitic_candidate",
                title="New parasitic-loss ABC",
            )
            ctx["parasitic_calculation_mode"] = "Use candidate vs reference"
            if candidate and ref_a is not None:
                suggested_delta_a = float(candidate.get("A") or 0.0) - float(ref_a or 0.0)
                ctx["component_mode_parasitics_hubs_axle"] = "Apply delta"
                ctx["delta_parasitics_N"] = suggested_delta_a
                quantity_metric(st, "Applied parasitic delta A", suggested_delta_a, "force", format_str="%.3f")
                st.success("Derived change from new parasitic losses vs reference is active.")
            elif candidate and ref_a is None:
                ctx["component_mode_parasitics_hubs_axle"] = "Keep inherited"
                ctx["delta_parasitics_N"] = 0.0
                st.warning("New parasitic losses are staged, but derived parasitic delta needs an explicit Reference Losses setup. Use manual delta A instead.")
            else:
                ctx["component_mode_parasitics_hubs_axle"] = "Keep inherited"
                ctx["delta_parasitics_N"] = 0.0
                st.caption("New parasitic losses are not fully defined yet.")
        else:
            ctx["parasitic_candidate_mode"] = "Not set"
            ctx["parasitic_calculation_mode"] = "Manual delta A"
            ctx["component_mode_parasitics_hubs_axle"] = "Apply delta"
            ctx["delta_parasitics_N"] = quantity_input(
                st,
                "Parasitics delta A",
                to_float(ctx.get("delta_parasitics_N"), 0.0),
                "force",
                key="parasitic_delta_input",
                step_canonical=0.1,
                format_str="%.3f",
            )
        st.caption("Current inherited/coastdown workflow uses parasitic adjustment through A only.")
    delta_parasitics = float(to_float(ctx.get("delta_parasitics_N"), 0.0) or 0.0)
    _render_applied_effect_block(
        title="Applied effect",
        caption="Read-only summary of the parasitic-loss effect currently resolved for this scenario.",
        inherited_message="Method: inherited. No parasitic-loss change is currently affecting the scenario.",
        metrics=[
            ("Method", str(ctx.get("parasitic_calculation_mode") or "Inherited")),
            ("Effect applied", format_quantity(delta_parasitics, "force", format_str="%.3f")),
            ("Status", "Applied" if abs(delta_parasitics) > 0.0 or str(ctx.get("parasitic_calculation_mode") or "") == "Use candidate vs reference" else "Pending"),
        ] if ctx.get("component_mode_parasitics_hubs_axle") == "Apply delta" else None,
    )


def render_component_build_up_panel(*, base_row: dict | None = None, saved_vde_id: int | None = None, tires_df=None):
    ctx = st.session_state.ctx
    options = ["Tires", "Brakes", "Parasitics / Hubs / Axle", "Trailer"]
    active = str(ctx.get("component_editor_active") or "Tires")
    if active not in options:
        active = "Tires"
    ctx["component_editor_active"] = active

    _render_component_overview_table(ctx)
    st.divider()

    with st.container(border=True):
        if ctx["component_editor_active"] == "Tires":
            _render_tire_component_editor(base_row=base_row, saved_vde_id=saved_vde_id, tires_df=tires_df)
        elif ctx["component_editor_active"] == "Brakes":
            _render_brake_component_editor(base_row=base_row)
        elif ctx["component_editor_active"] == "Parasitics / Hubs / Axle":
            _render_parasitic_component_editor(base_row=base_row)
        else:
            st.info("Trailer stays as a reserved slot in Sprint 5. The visual slot exists now so future component DBs can follow the same pattern.")

    summary = summarize_component_build_up_from_ctx(ctx)
    if summary.get("enabled"):
        abc_total = dict(summary.get("abc_total") or {})
        ctx["A"] = float(to_float(abc_total.get("A"), 0.0) or 0.0)
        ctx["B"] = float(to_float(abc_total.get("B"), 0.0) or 0.0)
        ctx["C"] = float(to_float(abc_total.get("C"), 0.0) or 0.0)

        st.caption("Current ABC_TOTAL resolved from Component Build-up")
        s1, s2, s3 = st.columns(3)
        quantity_metric(s1, "A_TOTAL", ctx["A"], "force", format_str="%.2f")
        quantity_metric(s2, "B_TOTAL", ctx["B"], "force_per_speed", format_str="%.5f")
        quantity_metric(s3, "C_TOTAL", ctx["C"], "force_per_speed_squared", format_str="%.6f")

        component_rows = [
            {
                "component": component.get("name"),
                "source": component.get("source"),
                "A": float(to_float(component.get("A"), 0.0) or 0.0),
                "B": float(to_float(component.get("B"), 0.0) or 0.0),
                "C": float(to_float(component.get("C"), 0.0) or 0.0),
            }
            for component in summary.get("components") or []
        ]
        if component_rows:
            with st.expander("Resolved components", expanded=False):
                st.dataframe(pd.DataFrame(component_rows), use_container_width=True, hide_index=True)

def render_baseline_picker_and_editor_panel():
    ctx = st.session_state.ctx

    try:
        rows = fetch_vde_rows_full()
    except Exception as e:
        st.error(f"Could not read vde_db: {e}")
        return

    if not rows:
        st.info("No snapshots in vde_db yet. Save a scenario first before trying to inherit from baseline.")
        return

    df = ensure_baseline_aliases(pd.DataFrame(rows))

    filter_opts = baseline_filter_options(df)
    with st.expander("Scenario filters"):
        c1, c2, c3, c4 = st.columns(4)
        leg = c1.selectbox("Legislation", filter_opts["legislation"])
        make = c2.selectbox("Make", filter_opts["make"])
        cat_contains = c3.text_input("Category contains", "")
        year_eq = c4.text_input("Year (=)", "")

    dfv = apply_baseline_filters(
        df,
        legislation=leg,
        make=make,
        category_contains=cat_contains,
        year_eq=year_eq,
    )

    if dfv.empty:
        st.warning("No rows after filters.")
        with st.expander("Show raw columns (debug)"):
            st.write(sorted(df.columns.tolist()))
        return

    cols_to_show = [
        "id", "created_at", "updated_at",
        "legislation", "category", "make", "model", "year", "notes",
        "engine_type", "engine_model", "engine_size_l", "engine_aspiration",
        "transmission_type", "transmission_model", "drive_type",
        "mass_kg", "test_mass_kg", "inertia_class", "cda_m2", "weight_dist_fr_pct", "payload_kg",
        "mro_kg", "options_kg", "wltp_category",
        "tire_size", "tire_rr_note", "smerf", "front_pressure_psi", "rear_pressure_psi",
        "rrc_N_per_kN", "crr1_frac_at_120kph", "rr_load_kpa",
        "coast_A_N", "coast_B_N_per_kph", "coast_C_N_per_kph2",
        "trans_A_coef_N", "trans_B_Npkph", "trans_C_coef_Npkph2",
        "brake_A_coef_N", "brake_B_Npkph", "brake_C_coef_Npkph2",
        "aero_C_coef_Npkph2",
        "rr_alpha_N", "rr_beta_Npkph", "rr_a_Npkph2", "rr_b_N", "rr_c_Npkph",
        "cycle_name", "cycle_source",
        "vde_urb_mj", "vde_hw_mj",
        "vde_net_mj_per_km", "vde_total_mj_per_km",
        "vde_urb_mj_per_km", "vde_hw_mj_per_km",
        "vde_low_mj_per_km", "vde_mid_mj_per_km", "vde_high_mj_per_km", "vde_extra_high_mj_per_km",
        "vde_id_parent", "baseline_A_N", "baseline_B_N_per_kph", "baseline_C_N_per_kph2", "baseline_mass_kg",
        "delta_rr_N", "delta_brake_N", "delta_parasitics_N", "delta_aero_Npkph2", "delta_mass_kg",
    ]
    cols_to_show = [c for c in cols_to_show if c in dfv.columns]
    options_df = dfv.sort_values("id", ascending=False).copy()
    options_df["baseline_label"] = options_df.apply(
        lambda row: (
            f"VDE-{int(row['id']):03d} | "
            f"{str(row.get('make') or '').upper()} "
            f"{str(row.get('model') or '')} | "
            f"{str(row.get('year') or '')} | "
            f"{str(row.get('legislation') or '')}"
        ),
        axis=1,
    )
    option_map = {row["baseline_label"]: int(row["id"]) for _, row in options_df.iterrows()}
    labels = list(option_map.keys())
    default_label = labels[0]
    selected_id = int(ctx.get("vde_id_parent") or ctx.get("baseline_id") or option_map[default_label])
    for label, row_id in option_map.items():
        if row_id == selected_id:
            default_label = label
            break

    c_pick, c_hint = st.columns([2, 1])
    sel_label = c_pick.selectbox("Select baseline scenario", labels, index=labels.index(default_label))
    sel_id = option_map[sel_label]
    c_hint.caption("Baseline scenarios are filtered by the controls above.")

    with st.expander("Matching baseline scenarios", expanded=False):
        display_df = options_df[cols_to_show + ["baseline_label"]] if "baseline_label" not in cols_to_show else options_df[cols_to_show]
        display_df = display_df.copy()
        rename_map = {}
        if "mass_kg" in display_df.columns:
            display_df["mass_kg"] = display_df["mass_kg"].apply(lambda value: to_display(value, "mass"))
            rename_map["mass_kg"] = f"mass [{unit_label('mass')}]"
        if "test_mass_kg" in display_df.columns:
            display_df["test_mass_kg"] = display_df["test_mass_kg"].apply(lambda value: to_display(value, "mass"))
            rename_map["test_mass_kg"] = f"test_mass [{unit_label('mass')}]"
        if "coast_A_N" in display_df.columns:
            display_df["coast_A_N"] = display_df["coast_A_N"].apply(lambda value: to_display(value, "force"))
            rename_map["coast_A_N"] = f"coast_A [{unit_label('force')}]"
        if "coast_B_N_per_kph" in display_df.columns:
            display_df["coast_B_N_per_kph"] = display_df["coast_B_N_per_kph"].apply(lambda value: to_display(value, "force_per_speed"))
            rename_map["coast_B_N_per_kph"] = f"coast_B [{unit_label('force_per_speed')}]"
        if "coast_C_N_per_kph2" in display_df.columns:
            display_df["coast_C_N_per_kph2"] = display_df["coast_C_N_per_kph2"].apply(lambda value: to_display(value, "force_per_speed_squared"))
            rename_map["coast_C_N_per_kph2"] = f"coast_C [{unit_label('force_per_speed_squared')}]"
        if "vde_net_mj_per_km" in display_df.columns:
            display_df["vde_net_mj_per_km"] = display_df["vde_net_mj_per_km"].apply(lambda value: to_display(value, "energy_per_distance"))
            rename_map["vde_net_mj_per_km"] = f"vde_net [{unit_label('energy_per_distance')}]"
        if "vde_total_mj_per_km" in display_df.columns:
            display_df["vde_total_mj_per_km"] = display_df["vde_total_mj_per_km"].apply(lambda value: to_display(value, "energy_per_distance"))
            rename_map["vde_total_mj_per_km"] = f"vde_total [{unit_label('energy_per_distance')}]"
        if rename_map:
            display_df = display_df.rename(columns=rename_map)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    base = dfv[dfv["id"] == sel_id].iloc[0].to_dict()

    st.session_state.ctx.update(build_baseline_state_payload(base, int(sel_id)))
    st.session_state.ctx["selected_baseline_row"] = base

    summary1, summary2, summary3, summary4 = st.columns(4)
    summary1.metric("Baseline ID", f"{int(base.get('id', 0))}")
    summary2.metric(
        f"ABC_TOTAL [{_abc_unit_triplet_label()}]",
        _compact_abc({"A": base.get("A"), "B": base.get("B"), "C": base.get("C")}),
    )
    quantity_metric(summary3, "Mass", base.get("mass_kg"), "mass", format_str="%.0f")
    quantity_metric(summary4, "Test mass", base.get("test_mass_kg"), "mass", format_str="%.0f")

    source_ui = str(ctx.get("abc_total_source_ui") or "Baseline ABC")
    if source_ui not in {"Baseline ABC", "From test coastdown", "Component Build-up"}:
        source_ui = "Baseline ABC"
        ctx["abc_total_source_ui"] = source_ui

    if source_ui == "Baseline ABC":
        ctx["from_delta"] = "Deltas"
        ctx.update(build_delta_mode_ctx_updates(base))
        st.info("Baseline scenario loaded. Roadload Basis in Roadload Build-up now decides whether ABC_TOTAL stays inherited, switches to measured/test coastdown, or is rebuilt from components.")
    elif source_ui == "Component Build-up":
        ctx["from_delta"] = "Change Parameters"
        st.success(f"Editing baseline #{base.get('id', '')} with full parameter override.")
        if str(ctx.get("legislation", "")).strip().upper() == "WLTP":
            st.info(
                "WLTP baseline path currently uses test mass for road/tire calculation. "
                "The code already has hooks for MRO/TPMLM-based WLTP mass resolution, "
                "but vehicle-type/cargo and TPMLM UI inputs are still a placeholder."
            )
        st.caption("Roadload component editors are available in Roadload Build-up.")
    else:
        ctx["from_delta"] = "From test"
        st.info("Baseline row remains selected as scenario context, but ABC_TOTAL will be replaced by measured coastdown inputs in Roadload Build-up.")

    with st.expander("Inherited baseline snapshot", expanded=False):
        key_cols = [
            "id", "legislation", "category", "make", "model", "year", "mass_kg", "test_mass_kg", "A", "B", "C",
            "vde_net_mj_per_km", "vde_urb_mj_per_km", "vde_hw_mj_per_km",
            "vde_low_mj_per_km", "vde_mid_mj_per_km", "vde_high_mj_per_km", "vde_extra_high_mj_per_km",
        ]
        snapshot = {k: base.get(k) for k in key_cols if k in base}
        if "mass_kg" in snapshot:
            snapshot["mass"] = format_quantity(snapshot.pop("mass_kg"), "mass", format_str="%.0f")
        if "test_mass_kg" in snapshot:
            snapshot["test_mass"] = format_quantity(snapshot.pop("test_mass_kg"), "mass", format_str="%.0f")
        if {"A", "B", "C"} <= set(snapshot.keys()):
            snapshot["ABC_TOTAL"] = _compact_abc(snapshot)
        if "vde_net_mj_per_km" in snapshot:
            snapshot["vde_net"] = format_quantity(snapshot.pop("vde_net_mj_per_km"), "energy_per_distance", format_str="%.4f")
        if "vde_urb_mj_per_km" in snapshot:
            snapshot["vde_urb"] = format_quantity(snapshot.pop("vde_urb_mj_per_km"), "energy_per_distance", format_str="%.4f")
        if "vde_hw_mj_per_km" in snapshot:
            snapshot["vde_hw"] = format_quantity(snapshot.pop("vde_hw_mj_per_km"), "energy_per_distance", format_str="%.4f")
        st.write(snapshot)


def render_compute_and_save_panel(*, defaults_df_getter, reset_ctx):
    ctx = st.session_state.ctx
    flash = st.session_state.pop("vde_setup_save_flash", None)
    if isinstance(flash, dict):
        st.success(str(flash.get("message") or "VDE snapshot persisted."))
        total_label = flash.get("vde_total")
        net_label = flash.get("vde_net")
        if total_label:
            st.info(f"Saved VDE_TOTAL: **{total_label}**")
        if net_label:
            st.info(f"Saved VDE_NET: **{net_label}**")

    st.caption(
        "Save uses the Sprint 5 workflow contract and still preserves the legacy snapshot fields needed by the current DB."
    )

    errs, warns = validate_core(ctx["A"], ctx["B"], ctx["C"], ctx["mass_kg"])
    for warning in (warns or []):
        st.warning(warning)
    if ctx.get("cycle_df") is None:
        errs.append("Cycle not loaded. Pick default or upload a CSV.")
    for error in (errs or []):
        st.error(error)
    disabled_actions = bool(errs)
    workflow_preview = None
    target_vde_id = None

    if not disabled_actions:
        try:
            workflow_preview = build_vde_setup_preview_from_ctx(ctx)
        except Exception as e:
            workflow_preview = {"ok": False, "error": f"Workflow preview not available: {e}"}
        if not workflow_preview.get("ok"):
            st.error(workflow_preview.get("error") or "Workflow preview not available.")
            disabled_actions = True
        else:
            save_payload = dict(workflow_preview.get("save_payload") or {})
            target_vde_id = save_payload.get("target_vde_id") or ctx.get("vde_id_parent") or ctx.get("baseline_id")
            preview_line_source = dict(workflow_preview.get("line_source") or {})
            preview_total = dict(workflow_preview.get("vde_total") or {})
            preview_net = dict(workflow_preview.get("vde_net") or {})
            preview_mass = dict(workflow_preview.get("mass_setup") or {})

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Scenario origin", _line_source_summary(ctx))
            s2.metric("Roadload basis", _total_source_summary(ctx, workflow_preview))
            s3.metric("Preview VDE_TOTAL", _format_energy_value(preview_total.get("mj_per_km"), unavailable="-"))
            s4.metric("Preview VDE_NET", _format_energy_value(preview_net.get("mj_per_km")))

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Update target", f"VDE-{int(target_vde_id):03d}" if target_vde_id is not None else "None")
            p2.metric("Cycle", str(ctx.get("cycle_name") or "-"))
            p3.metric("Mass basis", str(preview_mass.get("mass_basis") or "-"))
            quantity_metric(
                p4,
                "Resolved mass",
                preview_mass.get("resolved_mass_used_kg"),
                "mass",
                format_str="%.1f",
            )

            with st.expander("Snapshot / provenance", expanded=False):
                st.write(
                    {
                        "scenario_origin": _line_source_summary(ctx),
                        "roadload_basis": _total_source_summary(ctx, workflow_preview),
                        "update_target_vde_id": target_vde_id,
                        "line_source": preview_line_source,
                        "save_payload": workflow_preview.get("save_payload"),
                    }
                )

    update_confirm = st.checkbox(
        "Confirm update of selected baseline snapshot",
        value=False,
        disabled=disabled_actions or target_vde_id is None,
        key="vde_save_update_confirm",
    )

    action_cols = st.columns(2)
    save_as_new_clicked = action_cols[0].button(
        "Save as new snapshot",
        key="btn_compute_save_new",
        disabled=disabled_actions,
        use_container_width=True,
    )
    update_existing_clicked = action_cols[1].button(
        "Update existing snapshot",
        key="btn_compute_save_update",
        disabled=disabled_actions or target_vde_id is None or not update_confirm,
        use_container_width=True,
    )

    if target_vde_id is None and not disabled_actions:
        st.caption("No baseline target is selected, so only Save as new is available in this workflow.")
    else:
        st.caption("Use Save as new to preserve provenance as a separate row. Use Update existing only when you intentionally want to overwrite the selected baseline snapshot.")

    save_mode = None
    if save_as_new_clicked:
        save_mode = "insert_new"
    elif update_existing_clicked:
        save_mode = "update_existing"

    if save_mode:
        try:
            defaults_df = defaults_df_getter() if callable(defaults_df_getter) else None
            workflow_preview = workflow_preview or build_vde_setup_preview_from_ctx(ctx)
            saved = save_vde_setup_result(
                workflow_preview,
                save_mode,
                ctx=ctx,
                defaults_df=defaults_df,
            )
            vde_id = int(saved["vde_id"])
            row = dict(saved.get("row") or {})
            vde_total_mj_km = to_float(row.get("vde_total_mj_per_km"))
            vde_net_mj_km = to_float(row.get("vde_net_mj_per_km"))
            by_phase = dict((workflow_preview.get("vde_net") or workflow_preview.get("vde_total") or {}).get("by_phase") or {})

            total_label = format_quantity(vde_total_mj_km, "energy_per_distance") if vde_total_mj_km is not None else "-"
            net_label = format_quantity(vde_net_mj_km, "energy_per_distance", unavailable="Unavailable")
            st.info(f"Saved VDE_TOTAL: **{total_label}**")
            st.info(f"Saved VDE_NET: **{net_label}**")
            if by_phase:
                order = ["city", "hwy", "low", "mid", "high", "xhigh"]
                keys = [k for k in order if k in by_phase] + [k for k in by_phase if k not in order]
                cols = st.columns(min(4, len(keys)))
                for i, key in enumerate(keys):
                    label = {"city": "CITY", "hwy": "HWY"}.get(key, key.upper())
                    quantity_metric(cols[i % len(cols)], label, by_phase[key], "energy_per_distance", format_str="%.4f")
            st.session_state["vde_id"] = vde_id

            action_label = "Updated" if save_mode == "update_existing" else "Saved"
            st.session_state["vde_setup_save_flash"] = {
                "message": f"{action_label} VDE snapshot (id={vde_id}).",
                "vde_total": total_label,
                "vde_net": net_label,
            }
            reset_ctx(preserve_meta=True)
            st.rerun()

        except Exception as e:
            st.error(f"Failed to persist VDE snapshot: {e}")


def _render_phase_metrics(title: str, phases: dict | None) -> None:
    phase_dict = dict(phases or {})
    if not phase_dict:
        return

    st.caption(title)
    order = ["city", "hwy", "low", "mid", "high", "xhigh"]
    ordered = [key for key in order if key in phase_dict] + [key for key in phase_dict if key not in order]
    cols = st.columns(min(4, len(ordered)))
    for i, key in enumerate(ordered):
        quantity_metric(cols[i % len(cols)], key.upper(), phase_dict[key], "energy_per_distance", format_str="%.4f")


def _render_cycle_preview_snapshot() -> None:
    ctx = st.session_state.get("ctx", {})
    workflow_preview = _safe_workflow_preview(ctx)
    if not workflow_preview.get("ok"):
        st.info("Phase preview will appear here after the cycle and upstream inputs are valid.")
        return

    vde_total = dict(workflow_preview.get("vde_total") or {})
    vde_net = dict(workflow_preview.get("vde_net") or {})
    total_phases = dict(vde_total.get("by_phase") or {})
    net_phases = dict(vde_net.get("by_phase") or {})
    warnings = list(workflow_preview.get("warnings") or [])

    top1, top2, top3 = st.columns(3)
    top1.metric("Preview VDE_TOTAL", _format_energy_value(vde_total.get("mj_per_km"), unavailable="-"))
    top2.metric("Preview VDE_NET", _format_energy_value(vde_net.get("mj_per_km")))
    top3.metric("Cycle", str(ctx.get("cycle_name") or "-"))

    phase_order = ["city", "hwy", "low", "mid", "high", "xhigh"]
    present_phases = [key for key in phase_order if key in total_phases or key in net_phases]
    present_phases += [key for key in total_phases if key not in present_phases]
    present_phases += [key for key in net_phases if key not in present_phases]

    if present_phases:
        rows = []
        labels = {
            "city": "URB / CITY",
            "hwy": "HWY",
            "low": "LOW",
            "mid": "MID",
            "high": "HIGH",
            "xhigh": "XHIGH",
        }
        for phase in present_phases:
            rows.append(
                {
                    "Phase": labels.get(phase, phase.upper()),
                    "VDE_TOTAL": _format_energy_value(total_phases.get(phase), unavailable="-"),
                    "VDE_NET": _format_energy_value(net_phases.get(phase)),
                }
            )
        st.caption("Phase Preview")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if warnings:
        st.warning("Preview warnings: " + ", ".join(warnings[:3]))


def _review_value_label(value) -> str:
    if isinstance(value, dict):
        if {"A", "B", "C"} & set(value.keys()):
            return _compact_abc(value)
        if "mj_per_km" in value:
            return _format_energy_value(value.get("mj_per_km"))
        return json.dumps(value, default=str)
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "-"
    text = str(value or "").strip()
    return text or "-"


def _dict_to_rows(payload: dict | None) -> list[dict[str, str]]:
    data = dict(payload or {})
    rows = []
    for key, value in data.items():
        rows.append({"Field": str(key), "Value": _review_value_label(value)})
    return rows


def _save_mode_label(target_vde_id) -> str:
    if target_vde_id is None:
        return "Create new scenario"
    return "Create new scenario / Update existing VDE"


def render_vde_results_review_panel():
    ctx = st.session_state.get("ctx", {})
    component_build_up_active = _component_build_up_enabled(ctx)

    try:
        workflow_preview = build_vde_setup_preview_from_ctx(ctx)
    except Exception as exc:
        workflow_preview = {"ok": False, "error": f"Workflow preview not available: {exc}"}

    if not workflow_preview.get("ok"):
        st.warning(workflow_preview.get("error") or "Preview not available.")
        return

    save_payload = workflow_preview.get("save_payload") or {}
    review = build_vde_pre_save_review(ctx, workflow_preview, save_payload)
    performance_total = dict(workflow_preview.get("vde_total") or {})
    performance_net = dict(workflow_preview.get("vde_net") or {})
    warnings = list(workflow_preview.get("warnings") or [])
    working_summary = dict(review.get("working_scenario_summary") or {})
    change_summary = dict(review.get("change_summary") or {})
    reference_snapshot = dict(review.get("reference_snapshot") or {})
    baseline_rows = list(review.get("baseline_vs_working_rows") or [])
    staged_payload = dict(review.get("staged_save_payload") or {})
    phase_update_row = dict(workflow_preview.get("phase_update_row") or {})
    legacy_preview = build_live_vde_preview(ctx)

    with st.container(border=True):
        st.subheader("1. Performance Summary")
        p1, p2, p3 = st.columns(3)
        p1.metric("VDE_TOTAL", _format_energy_value(performance_total.get("mj_per_km"), unavailable="-"))
        p2.metric("VDE_NET", _format_energy_value(performance_net.get("mj_per_km")))
        p3.metric("Cycle", str(ctx.get("cycle_name") or working_summary.get("cycle_name") or "-"))

        phase_rows = []
        total_phases = dict(performance_total.get("by_phase") or {})
        net_phases = dict(performance_net.get("by_phase") or {})
        phase_order = ["city", "hwy", "low", "mid", "high", "xhigh"]
        present_phases = [key for key in phase_order if key in total_phases or key in net_phases]
        present_phases += [key for key in total_phases if key not in present_phases]
        present_phases += [key for key in net_phases if key not in present_phases]
        labels = {"city": "URB / CITY", "hwy": "HWY", "low": "LOW", "mid": "MID", "high": "HIGH", "xhigh": "XHIGH"}
        for phase in present_phases:
            phase_rows.append(
                {
                    "Phase": labels.get(phase, phase.upper()),
                    "VDE_TOTAL": _format_energy_value(total_phases.get(phase), unavailable="-"),
                    "VDE_NET": _format_energy_value(net_phases.get(phase)),
                }
            )
        if phase_rows:
            st.caption("Phase outputs")
            st.dataframe(pd.DataFrame(phase_rows), use_container_width=True, hide_index=True)
        if warnings:
            st.warning("Warnings: " + ", ".join(warnings))

    with st.container(border=True):
        st.subheader("2. Review Status")
        target_vde_id = staged_payload.get("target_vde_id")
        basis_label = (
            "Baseline ABC_TOTAL" if str(working_summary.get("roadload_basis") or "") == "BASELINE"
            else "Component Build-up" if str(working_summary.get("roadload_basis") or "") in {"COMPONENT_BUILD_UP", "COMPONENTS"}
            else "Measured/test coastdown"
        )
        r1, r2, r3 = st.columns(3)
        r1.metric("Reference source", str(reference_snapshot.get("kind") or "-"))
        r2.metric("Roadload basis", basis_label)
        r3.metric("Save mode", _save_mode_label(target_vde_id))
        st.caption("All review data below was resolved from the current draft state using the existing workflow preview and staged save payload.")

    with st.container(border=True):
        st.subheader("3. What Changed")
        change_rows = []
        order = [
            ("Mass", "mass"),
            ("Tires", "tires"),
            ("Aerodynamics", "aero"),
            ("Brakes", "brakes"),
            ("Parasitics", "parasitics"),
            ("Trailer", "trailer"),
            ("Transmission", "transmission"),
            ("Cycle", "cycle"),
        ]
        for label, key in order:
            item = dict(change_summary.get(key) or {})
            change_rows.append(
                {
                    "Area": label,
                    "Working state": item.get("working") or "-",
                    "Status": item.get("state") or "-",
                }
            )
        st.dataframe(pd.DataFrame(change_rows), use_container_width=True, hide_index=True)

    with st.container(border=True):
        st.subheader("4. Reference vs Working Scenario")
        main_rows = []
        quiet_rows = []
        for row in baseline_rows:
            compact = dict(row)
            if str(compact.get("Change") or "").strip().lower() == "not applicable" or "not needed" in str(compact.get("Change") or "").strip().lower():
                quiet_rows.append(compact)
            else:
                main_rows.append(compact)
        if main_rows:
            st.dataframe(pd.DataFrame(main_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No major comparison rows are available yet.")
        if quiet_rows:
            with st.expander(f"Not applicable rows ({len(quiet_rows)})", expanded=False):
                st.dataframe(pd.DataFrame(quiet_rows), use_container_width=True, hide_index=True)

    with st.container(border=True):
        st.subheader("5. Staged Save Payload")
        st.caption("This is the payload that Save / Edit will persist.")
        with st.expander("Staged payload details", expanded=False):
            st.caption("Insert row")
            st.dataframe(pd.DataFrame(_dict_to_rows(staged_payload.get("insert_row"))), use_container_width=True, hide_index=True)
            st.caption("Update row")
            st.dataframe(pd.DataFrame(_dict_to_rows(staged_payload.get("update_row"))), use_container_width=True, hide_index=True)
            target_label = f"VDE-{int(staged_payload['target_vde_id']):03d}" if staged_payload.get("target_vde_id") is not None else "None"
            st.caption(f"Target VDE id: {target_label}")
            if phase_update_row:
                st.caption("Phase update row")
                st.dataframe(pd.DataFrame(_dict_to_rows(phase_update_row)), use_container_width=True, hide_index=True)

    with st.container(border=True):
        st.subheader("6. Technical Details")
        with st.expander("Resolved components", expanded=False):
            components = list(workflow_preview.get("components") or [])
            if components:
                st.dataframe(pd.DataFrame(components), use_container_width=True, hide_index=True)
            else:
                st.caption("No resolved component records are present for this preview.")
        with st.expander("Physical preview details", expanded=False):
            st.write(
                {
                    "line_source": workflow_preview.get("line_source"),
                    "initial_abc_total_base": workflow_preview.get("initial_abc_total_base"),
                    "component_abc_total": workflow_preview.get("component_abc_total"),
                    "abc_total": workflow_preview.get("abc_total"),
                    "transmission_losses": workflow_preview.get("transmission_losses"),
                    "abc_net": workflow_preview.get("abc_net"),
                    "mass_setup": workflow_preview.get("mass_setup"),
                    "vde_total": workflow_preview.get("vde_total"),
                    "vde_net": workflow_preview.get("vde_net"),
                    "warnings": workflow_preview.get("warnings"),
                }
            )
        with st.expander("Existing save payload object", expanded=False):
            st.write(
                {
                    "reference_snapshot": review.get("reference_snapshot"),
                    "working_scenario_summary": review.get("working_scenario_summary"),
                    "change_summary": review.get("change_summary"),
                    "staged_save_payload": staged_payload,
                    "phase_update_row": phase_update_row,
                }
            )
        if legacy_preview.get("ok") and not component_build_up_active:
            with st.expander("Legacy preview (debug)", expanded=False):
                total_mj_km = float(legacy_preview["total_mj_km"])
                phases = legacy_preview.get("phases", {})
                st.caption(f"Legacy NET preview: {format_quantity(total_mj_km, 'energy_per_distance', format_str='%.4f')}")
                _render_phase_metrics("Legacy phase outputs", phases)


def render_live_vde_preview_panel():
    ctx = st.session_state.get("ctx", {})
    component_build_up_active = _component_build_up_enabled(ctx)

    try:
        workflow_preview = build_vde_setup_preview_from_ctx(ctx)
    except Exception as e:
        workflow_preview = {"ok": False, "error": f"Workflow preview not available: {e}"}

    legacy_preview = build_live_vde_preview(ctx)
    if not workflow_preview.get("ok") and not legacy_preview.get("ok"):
        st.warning(workflow_preview.get("error") or legacy_preview.get("error") or "Preview not available.")
        return

    if workflow_preview.get("ok"):
        vde_total = workflow_preview.get("vde_total") or {}
        vde_net = workflow_preview.get("vde_net") or {}
        abc_total = workflow_preview.get("abc_total") or {}
        abc_net = workflow_preview.get("abc_net") or {}
        mass_setup = workflow_preview.get("mass_setup") or {}
        transmission = workflow_preview.get("transmission_losses") or {}
        transmission_abc = transmission.get("abc") or {}
        line_source = workflow_preview.get("line_source") or {}
        warnings = list(workflow_preview.get("warnings") or [])

        top1, top2, top3, top4 = st.columns(4)
        top1.metric("VDE_TOTAL", _format_energy_value(vde_total.get("mj_per_km"), unavailable="-"))
        top2.metric("VDE_NET", _format_energy_value(vde_net.get("mj_per_km")))
        top3.metric("Roadload basis", _total_source_summary(ctx, workflow_preview))
        top4.metric("Transmission", str(transmission.get("status") or "missing").replace("_", " ").title())

        preview_rows = [
            {
                "Stage": "TOTAL",
                "ABC": _compact_abc(abc_total),
                "Energy": _format_energy_value(vde_total.get("mj_per_km"), unavailable="-"),
                "Status": "Available",
            },
            {
                "Stage": "TRANS",
                "ABC": _compact_abc(transmission_abc) if transmission.get("status") == "available" else "Unavailable",
                "Energy": "-",
                "Status": str(transmission.get("status") or "missing").replace("_", " ").title(),
            },
            {
                "Stage": "NET",
                "ABC": _compact_abc(abc_net) if abc_net else "Unavailable",
                "Energy": _format_energy_value(vde_net.get("mj_per_km")),
                "Status": "Available" if vde_net.get("mj_per_km") is not None else "Pending",
            },
        ]
        st.caption("Preview Overview")
        st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Mass basis", str(mass_setup.get("mass_basis") or "-"))
        quantity_metric(d2, "Resolved mass", mass_setup.get("resolved_mass_used_kg"), "mass", format_str="%.1f")
        d3.metric("Scenario origin", _line_source_summary(ctx))
        d4.metric("Cycle", str(ctx.get("cycle_name") or line_source.get("cycle_name") or "-"))

        if warnings:
            st.warning("Workflow warnings: " + ", ".join(warnings))

        with st.expander("Phase outputs", expanded=False):
            _render_phase_metrics("TOTAL phase outputs", vde_total.get("by_phase"))
            _render_phase_metrics("NET phase outputs", vde_net.get("by_phase"))

        with st.expander("Technical preview details", expanded=False):
            st.write(
                {
                    "initial_abc_total_base": workflow_preview.get("initial_abc_total_base"),
                    "component_abc_total": workflow_preview.get("component_abc_total"),
                    "abc_total": workflow_preview.get("abc_total"),
                    "abc_net": workflow_preview.get("abc_net"),
                    "transmission_losses": transmission,
                    "mass_setup": mass_setup,
                    "components": workflow_preview.get("components"),
                    "save_payload": workflow_preview.get("save_payload"),
                }
            )

    if legacy_preview.get("ok") and not workflow_preview.get("ok"):
        total_mj_km = float(legacy_preview["total_mj_km"])
        phases = legacy_preview.get("phases", {})
        equiv = legacy_preview.get("equiv")

        st.info(f"Legacy NET preview: **{format_quantity(total_mj_km, 'energy_per_distance')}**")
        if equiv is not None:
            with st.expander("RoadLoad breakdown", expanded=False):
                st.dataframe(pd.DataFrame(equiv.component_table), use_container_width=True, hide_index=True)
        _render_phase_metrics("Legacy phase outputs", phases)
    elif legacy_preview.get("ok") and workflow_preview.get("ok") and not component_build_up_active:
        with st.expander("Legacy preview (debug)", expanded=False):
            total_mj_km = float(legacy_preview["total_mj_km"])
            phases = legacy_preview.get("phases", {})
            st.caption(f"Legacy NET preview: {format_quantity(total_mj_km, 'energy_per_distance', format_str='%.4f')}")
            _render_phase_metrics("Legacy phase outputs", phases)


def render_cycle_section(*, include_preview_snapshot: bool = True):
    ctx = st.session_state.ctx

    errors, warns = validate_core(ctx["A"], ctx["B"], ctx["C"], ctx["mass_kg"])
    if warns:
        for warning in warns:
            st.warning(warning)

    cleft, cright = st.columns([1, 1])
    use_default = cleft.button("Use legislation default cycle")
    upload = cright.file_uploader(
        "or upload CSV with columns [t, v] (s, m/s)",
        type=["csv"],
        accept_multiple_files=False,
    )

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

    if ctx["cycle_df"] is not None:
        df_cycle = ctx["cycle_df"]
        t_vals = pd.to_numeric(df_cycle["t"], errors="coerce")
        v_vals = pd.to_numeric(df_cycle["v"], errors="coerce")
        duration_s = float(t_vals.iloc[-1] - t_vals.iloc[0]) if len(t_vals) else 0.0
        distance_km = float(np.trapezoid(v_vals, t_vals) / 1000.0)
        avg_speed_kph = float(v_vals.mean() * 3.6) if len(v_vals) else 0.0
        distance_value = distance_km if _current_unit_system() == "Metric" else distance_km * 0.621371192237
        distance_unit = "km" if _current_unit_system() == "Metric" else "mi"
        st.caption(
            f"Duration: {duration_s:.0f} s | Distance: {distance_value:.2f} {distance_unit} | "
            f"v_avg: {format_quantity(avg_speed_kph, 'speed', format_str='%.1f')}"
        )
    else:
        errors.append("No cycle loaded. Use default or upload a CSV.")

    if ctx["cycle_df"] is not None:
        display_cycle_df = ctx["cycle_df"].copy()
        if _current_unit_system() == "US customary":
            display_cycle_df["v"] = pd.to_numeric(display_cycle_df["v"], errors="coerce") * 0.621371192237
        fig = cycle_chart(display_cycle_df, unit_system=_current_unit_system())
        if fig:
            fig.update_yaxes(title_text=f"Speed [{unit_label('speed')}]")
            st.plotly_chart(fig, use_container_width=True)

    if include_preview_snapshot:
        st.divider()
        st.caption("Cycle-linked VDE Preview")
        st.caption("This view keeps cycle selection and phase preview together so URB/CITY and HWY impacts stay visible while you adjust the scenario.")
        _render_cycle_preview_snapshot()


def render_auxiliaries_section(*, defaults_df_getter):
    """
    Uses A/B/C + mass + (category, electrification, transmission_type) from ctx
    to decompose NET coastdown using the defaults CSV.
    """
    ctx = st.session_state.ctx
    st.subheader("Estimate auxiliaries from coastdown (NET)")

    missing = [k for k in ("A", "B", "C", "mass_kg", "category") if ctx.get(k) in (None, "")]
    disabled = len(missing) > 0
    if disabled:
        st.caption(f"Fill first: {', '.join(missing)}")

    if st.button("Estimate using defaults CSV", disabled=disabled):
        defaults_df = defaults_df_getter() if callable(defaults_df_getter) else None
        res = estimate_aux_from_coastdown(
            A_N=ctx["A"],
            B_N_per_kph=ctx["B"],
            C_N_per_kph2=ctx["C"],
            mass_kg=ctx["mass_kg"],
            category=ctx["category"],
            electrification=ctx.get("electrification", "ICE"),
            transmission_type=ctx.get("transmission_type", "AT"),
            cdA_override_m2=ctx.get("cda_m2"),
            defaults_df=defaults_df,
        )

        ctx.update(
            {
                "rr_alpha_N": res["rr_alpha_N"],
                "rr_beta_Npkph": res["rr_beta_Npkph"],
                "aero_C_coef_Npkph2": res["aero_C_coef_Npkph2"],
                "parasitic_A_N": res["parasitic_A_coef_N"],
                "parasitic_B_Npkph": res["parasitic_B_coef_Npkph"],
                "parasitic_C_Npkph2": res["parasitic_C_coef_Npkph2"],
                "decomp_check_ok": res["check_ok"],
                "cda_m2": res["cdA_used_m2"],
            }
        )

        c1, c2, c3 = st.columns(3)
        quantity_metric(c1, "RR alpha", res["rr_alpha_N"], "force", format_str="%.2f")
        quantity_metric(c2, "RR beta", res["rr_beta_Npkph"], "force_per_speed", format_str="%.3f")
        quantity_metric(c3, "Aero C", res["aero_C_coef_Npkph2"], "force_per_speed_squared", format_str="%.3f")
        d1, d2, d3 = st.columns(3)
        quantity_metric(d1, "Parasitic A", res["parasitic_A_coef_N"], "force", format_str="%.2f")
        quantity_metric(d2, "Parasitic B", res["parasitic_B_coef_Npkph"], "force_per_speed", format_str="%.3f")
        d3.metric("Check", "OK" if res["check_ok"] else "Review")


def render_from_test_section():
    """
    Enter coastdown outputs directly.
    Keeps compatibility with legacy session keys as a transition layer.
    """
    ctx = st.session_state.ctx
    st.subheader("From test - direct coastdown (A/B/C)")

    colA, colB, colC = st.columns(3)
    A = quantity_input(colA, "A", to_float(ctx.get("A"), 30.0), "force", key="from_test_A", min_canonical=0.0, max_canonical=500.0, step_canonical=0.1, format_str="%.2f")
    B = quantity_input(colB, "B", to_float(ctx.get("B"), 0.80), "force_per_speed", key="from_test_B", min_canonical=-1.0, max_canonical=5.0, step_canonical=0.01, format_str="%.5f")
    C = quantity_input(colC, "C", to_float(ctx.get("C"), 0.011), "force_per_speed_squared", key="from_test_C", min_canonical=0.0, max_canonical=0.100, step_canonical=0.001, format_str="%.6f")
    ctx["A"], ctx["B"], ctx["C"] = to_float(A), to_float(B), to_float(C)

    st.session_state["abc"] = {"A": float(A), "B": float(B), "C": float(C)}
    st.session_state["manual_mass"] = to_float(ctx.get("mass_kg"))
    st.caption("Mass and test-mass inputs are managed in Vehicle Parameters.")


def render_aero_section(*, prefill=None):
    """
    Uses cda_m2 from DB. Shows estimated aero C [N/kph^2] as reference.
    Does not overwrite measured coastdown C.
    """
    ctx = st.session_state.ctx
    st.subheader("Aerodynamics")

    cda0 = to_float(prefill.get("cda_m2"), ctx.get("cda_m2")) if prefill else to_float(ctx.get("cda_m2"), None)
    cda = quantity_input(st, "CdA", to_float(cda0, 0.0), "cda", key="aero_section_cda", step_canonical=0.01, format_str="%.3f")
    ctx["cda_m2"] = to_float(cda)

    rho = 1.2
    C_aero = 0.5 * rho * ctx["cda_m2"] * (1 / 3.6) ** 2
    ctx["aero_C_coef_Npkph2"] = C_aero

    quantity_metric(st, "C_aero (est.)", C_aero, "force_per_speed_squared", format_str="%.6f")
    st.caption("Measured coastdown C remains in 'coast_C_N_per_kph2'; this is only reference.")


def render_rr_section(*, prefill=None, tires_df=None):
    """
    RR only (does not overwrite A/B/C):
      IN: rrc_N_per_kN [N/kN], crr1_frac_at_120kph [-], mass_kg [kg]
      Optional: tire selectbox when tires_df is provided
      OUT: rr_alpha_N [N], rr_beta_Npkph [N/kph]; stores tire_size in ctx
    """
    ctx = st.session_state.ctx
    st.subheader("Rolling Resistance")

    if isinstance(tires_df, pd.DataFrame) and not tires_df.empty:
        sizes = sorted(tires_df["tire_size"].dropna().astype(str).unique().tolist())
        current_size = str(ctx.get("tire_size") or (prefill.get("tire_size") if prefill else "") or "")
        try:
            idx0 = sizes.index(current_size) if current_size in sizes else 0
        except Exception:
            idx0 = 0
        sel = st.selectbox("Tire size", sizes, index=idx0)
        ctx["tire_size"] = sel
        trow = tires_df.loc[tires_df["tire_size"] == sel].iloc[0].to_dict()
        st.caption(f'Diameter {trow["tire_circ_mm"]:.0f} mm ')
        ctx["tire_circ_m"] = float(trow["tire_circ_mm"]) / 1000.0

    if prefill:
        rrc0 = to_float(prefill.get("rrc_N_per_kN"), to_float(ctx.get("rrc_N_per_kN"), 9.5))
        frac0 = to_float(prefill.get("crr1_frac_at_120kph"), to_float(ctx.get("crr1_frac_at_120kph"), 0.10))
        m0 = to_float(prefill.get("mass_kg"), to_float(ctx.get("mass_kg"), 1500.0))
        weight_dist0 = to_float(prefill.get("weight_dist_fr_pct"), to_float(ctx.get("weight_dist_fr_pct"), 50.0))
    else:
        rrc0 = to_float(ctx.get("rrc_N_per_kN"), 9.5)
        frac0 = to_float(ctx.get("crr1_frac_at_120kph"), 0.10)
        m0 = to_float(ctx.get("mass_kg"), 1500.0)
        weight_dist0 = to_float(ctx.get("weight_dist_fr_pct"), 50.0)

    c1, c2 = st.columns(2)
    ctx["rrc_N_per_kN"] = quantity_input(c1, "Final RRC", to_float(rrc0, 0.0), "rrc", key="rr_section_rrc", step_canonical=0.1, format_str="%.2f")
    ctx["crr1_frac_at_120kph"] = c2.number_input(
        "Linear RR fraction at 120 kph [-]",
        value=float(frac0),
        min_value=0.0,
        max_value=1.0,
        step=0.005,
        format="%.3f",
    )

    legislation = str(ctx.get("legislation", "") or "").strip().upper()
    st.caption(
        "Mass setup is controlled in Vehicle Parameters. "
        f"Using curb mass={format_quantity(ctx.get('mass_kg') or m0 or 0.0, 'mass', format_str='%.1f')}, "
        f"front weight distribution={float(ctx.get('weight_dist_fr_pct') or weight_dist0 or 50.0):.1f}%, "
        f"VDE mass basis={str(ctx.get('tire_load_mass_basis') or resolve_tire_load_mass_basis(ctx))}."
    )

    tire_mass_resolution = resolve_tire_calculation_mass(ctx)
    calc_mass_kg = tire_mass_resolution.get("mass_kg")
    if legislation == "EPA" and ctx.get("tire_load_mass_basis") == "TWC":
        ctx["inertia_class"] = calc_mass_kg
        ctx["twc_kg"] = calc_mass_kg
    G = 9.80665
    load_kN = (ctx["mass_kg"] * G) / 1000.0 if ctx.get("mass_kg") else 0.0
    A_rr, B_rr, _, epa_v_avg = _abc_from_final_rr_target(
        ctx.get("rrc_N_per_kN"),
        load_kN=load_kN,
        crr_frac_120=ctx.get("crr1_frac_at_120kph"),
    )

    ctx["rr_alpha_N"] = A_rr
    ctx["rr_beta_Npkph"] = B_rr

    c4, c5, c6 = st.columns(3)
    c4.metric("Load [kN]", f"{load_kN:.2f}")
    quantity_metric(c5, "A_rr ~ SMERF", A_rr, "force", format_str="%.2f")
    quantity_metric(c6, "Calculation mass", calc_mass_kg, "mass", format_str="%.1f")
    st.caption(f"Current VDE mass basis: {tire_mass_resolution.get('basis')} ({tire_mass_resolution.get('source')})")
    st.caption(f"Resolved from final RRC target: F_rr(v) = {A_rr:.2f} + {B_rr:.5f}*v   [v in kph]")
    if epa_v_avg is not None:
        st.caption(f"EPA v_avg used for decomposition: {epa_v_avg:.2f} kph")


def render_parasitic_section(*, prefill=None):
    ctx = st.session_state.ctx
    st.subheader("Parasitics / Hubs / Axle")

    if prefill:
        parA0 = to_float(prefill.get("parasitic_A_coef_N"), ctx.get("parasitic_A_coef_N", 0.0))
        parB0 = to_float(prefill.get("parasitic_B_Npkph"), ctx.get("parasitic_B_Npkph", 0.0))
        parC0 = to_float(prefill.get("parasitic_C_coef_Npkph2"), ctx.get("parasitic_C_coef_Npkph2", 0.0))
    else:
        parA0 = to_float(ctx.get("parasitic_A_coef_N"), 0.0)
        parB0 = to_float(ctx.get("parasitic_B_Npkph"), 0.0)
        parC0 = to_float(ctx.get("parasitic_C_coef_Npkph2"), 0.0)

    p1, p2, p3 = st.columns(3)
    ctx["parasitic_A_coef_N"] = quantity_input(p1, "Parasitic A", to_float(parA0, 0.0), "force", key="parasitic_A_coef_N", step_canonical=0.1, format_str="%.2f")
    ctx["parasitic_B_Npkph"] = quantity_input(p2, "Parasitic B", to_float(parB0, 0.0), "force_per_speed", key="parasitic_B_Npkph", step_canonical=0.001, format_str="%.5f")
    ctx["parasitic_C_coef_Npkph2"] = quantity_input(p3, "Parasitic C", to_float(parC0, 0.0), "force_per_speed_squared", key="parasitic_C_coef_Npkph2", step_canonical=0.0001, format_str="%.6f")

    c1, c2, c3 = st.columns(3)
    quantity_metric(c1, "Parasitic A", ctx["parasitic_A_coef_N"], "force", format_str="%.2f")
    quantity_metric(c2, "Parasitic B", ctx["parasitic_B_Npkph"], "force_per_speed", format_str="%.5f")
    quantity_metric(c3, "Parasitic C", ctx["parasitic_C_coef_Npkph2"], "force_per_speed_squared", format_str="%.6f")

    st.caption("This slot is meant to absorb hub, axle, bearing and other non-brake parasitic losses.")


def render_brake_section(*, prefill=None):
    ctx = st.session_state.ctx
    st.subheader("Brakes")

    if prefill:
        brA0 = to_float(prefill.get("brake_A_coef_N"), ctx.get("brake_A_coef_N", 0.0))
        brB0 = to_float(prefill.get("brake_B_Npkph"), ctx.get("brake_B_Npkph", 0.0))
        brC0 = to_float(prefill.get("brake_C_coef_Npkph2"), ctx.get("brake_C_coef_Npkph2", 0.0))
    else:
        brA0 = to_float(ctx.get("brake_A_coef_N"), 0.0)
        brB0 = to_float(ctx.get("brake_B_Npkph"), 0.0)
        brC0 = to_float(ctx.get("brake_C_coef_Npkph2"), 0.0)

    b1, b2, b3 = st.columns(3)
    ctx["brake_A_coef_N"] = quantity_input(b1, "Brake A", to_float(brA0, 0.0), "force", key="brake_A_coef_N", step_canonical=0.1, format_str="%.2f")
    ctx["brake_B_Npkph"] = quantity_input(b2, "Brake B", to_float(brB0, 0.0), "force_per_speed", key="brake_B_Npkph", step_canonical=0.001, format_str="%.5f")
    ctx["brake_C_coef_Npkph2"] = quantity_input(b3, "Brake C", to_float(brC0, 0.0), "force_per_speed_squared", key="brake_C_coef_Npkph2", step_canonical=0.0001, format_str="%.6f")

    c1, c2, c3 = st.columns(3)
    quantity_metric(c1, "Brake A", ctx["brake_A_coef_N"], "force", format_str="%.2f")
    quantity_metric(c2, "Brake B", ctx["brake_B_Npkph"], "force_per_speed", format_str="%.5f")
    quantity_metric(c3, "Brake C", ctx["brake_C_coef_Npkph2"], "force_per_speed_squared", format_str="%.6f")

    st.caption("Brake drag remains an explicit TOTAL component so future brake DB rows can plug into the same slot.")


def render_transmission_losses_section(*, prefill=None):
    """
    Explicit TOTAL -> NET bridge.

    Keeps transmission losses visible in the workflow so VDE_NET semantics are
    not hidden inside page-specific assumptions.
    """
    ctx = st.session_state.ctx
    st.subheader("Transmission Losses / Neutral Drag")
    st.caption("Transmission is configured separately from Components because it bridges ABC_TOTAL to ABC_NET, but it behaves like a technical component with explicit reference and explicit application.")

    prefill = dict(prefill or {})
    source_options = ["Missing", "Baseline", "Manual"]
    source_default = str(
        ctx.get("transmission_losses_source")
        or ("Baseline" if prefill.get("trans_A_coef_N") is not None else "Missing")
    ).strip().title()
    if source_default not in source_options:
        source_default = "Missing"

    base_a = to_float(prefill.get("trans_A_coef_N"), 0.0)
    base_b = to_float(prefill.get("trans_B_coef_Npkph", prefill.get("trans_B_Npkph")), 0.0)
    base_c = to_float(prefill.get("trans_C_coef_Npkph2"), 0.0)
    baseline_available = any(abs(float(value or 0.0)) > 0.0 for value in (base_a, base_b, base_c))
    baseline_target_id = int(prefill.get("id") or ctx.get("vde_id_parent") or ctx.get("baseline_id") or 0) or None

    with st.container(border=True):
        _render_tire_editor_block_header(
            "Reference Transmission Losses",
            "Defines the inherited neutral-drag reference available from the selected baseline row.",
        )
        if baseline_available:
            _render_abc_reference_metrics(
                title="Baseline transmission losses",
                a_value=base_a,
                b_value=base_b,
                c_value=base_c,
                status="Loaded",
                caption="This is the current inherited transmission reference before any manual replacement.",
            )
        else:
            st.warning("Reference status: Missing. No transmission losses are stored on the selected baseline row.")
        if baseline_target_id:
            with st.expander("Set / update baseline reference transmission losses", expanded=not baseline_available):
                t1, t2, t3 = st.columns(3)
                baseline_ref_a = quantity_input(
                    t1,
                    "Baseline transmission A",
                    to_float(ctx.get("transmission_baseline_reference_A"), base_a or 0.0),
                    "force",
                    key="transmission_baseline_reference_A_input",
                    step_canonical=0.1,
                    format_str="%.3f",
                )
                baseline_ref_b = quantity_input(
                    t2,
                    "Baseline transmission B",
                    to_float(ctx.get("transmission_baseline_reference_B"), base_b or 0.0),
                    "force_per_speed",
                    key="transmission_baseline_reference_B_input",
                    step_canonical=0.0001,
                    format_str="%.6f",
                )
                baseline_ref_c = quantity_input(
                    t3,
                    "Baseline transmission C",
                    to_float(ctx.get("transmission_baseline_reference_C"), base_c or 0.0),
                    "force_per_speed_squared",
                    key="transmission_baseline_reference_C_input",
                    step_canonical=0.000001,
                    format_str="%.8f",
                )
                ctx["transmission_baseline_reference_A"] = baseline_ref_a
                ctx["transmission_baseline_reference_B"] = baseline_ref_b
                ctx["transmission_baseline_reference_C"] = baseline_ref_c
                update_baseline = st.checkbox(
                    "Also update selected baseline row in VDE DB",
                    value=not baseline_available,
                    key="transmission_update_baseline_checkbox",
                )
                if st.button("Apply reference transmission losses", key="transmission_apply_reference_button"):
                    ctx["trans_A_coef_N"] = float(baseline_ref_a)
                    ctx["trans_B_coef_Npkph"] = float(baseline_ref_b)
                    ctx["trans_C_coef_Npkph2"] = float(baseline_ref_c)
                    ctx["trans_B_Npkph"] = float(baseline_ref_b)
                    if update_baseline:
                        try:
                            update_vde_snapshot(
                                int(baseline_target_id),
                                {
                                    "trans_A_coef_N": float(baseline_ref_a),
                                    "trans_B_Npkph": float(baseline_ref_b),
                                    "trans_C_coef_Npkph2": float(baseline_ref_c),
                                },
                            )
                            _sync_baseline_abc_reference_in_ctx(
                                a_key="trans_A_coef_N",
                                b_key="trans_B_Npkph",
                                c_key="trans_C_coef_Npkph2",
                                a_value=float(baseline_ref_a),
                                b_value=float(baseline_ref_b),
                                c_value=float(baseline_ref_c),
                                baseline_target_id=baseline_target_id,
                            )
                            if baseline_target_id and int(ctx.get("vde_id_parent") or ctx.get("baseline_id") or 0) == int(baseline_target_id):
                                ctx["trans_B_coef_Npkph"] = float(baseline_ref_b)
                            st.success(f"Baseline reference transmission losses updated on VDE id={baseline_target_id}.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not update baseline reference transmission losses: {exc}")
                    else:
                        st.success("Reference transmission losses staged only for the current scenario context.")

    with st.container(border=True):
        _render_tire_editor_block_header(
            "Applied Transmission Change",
            "Choose the transmission losses used to subtract ABC_TRANS from ABC_TOTAL and resolve ABC_NET.",
        )
        source = st.radio(
            "Transmission source",
            source_options,
            horizontal=True,
            index=source_options.index(source_default),
            key="transmission_losses_source_radio",
        )
        ctx["transmission_losses_source"] = source

        if source == "Baseline":
            ctx["trans_A_coef_N"] = base_a
            ctx["trans_B_coef_Npkph"] = base_b
            ctx["trans_C_coef_Npkph2"] = base_c
            if baseline_available:
                st.info("Calculation status: baseline transmission losses are active.")
            else:
                st.warning("Baseline source is selected, but no baseline transmission losses are available. VDE_NET will remain unavailable.")
        elif source == "Missing":
            ctx["trans_A_coef_N"] = 0.0
            ctx["trans_B_coef_Npkph"] = 0.0
            ctx["trans_C_coef_Npkph2"] = 0.0
            st.warning("VDE_TOTAL remains available; VDE_NET will remain unavailable until transmission losses are provided.")
        else:
            a0 = to_float(ctx.get("trans_A_coef_N"), base_a)
            b0 = to_float(ctx.get("trans_B_coef_Npkph"), base_b)
            c0 = to_float(ctx.get("trans_C_coef_Npkph2"), base_c)
            c1, c2, c3 = st.columns(3)
            ctx["trans_A_coef_N"] = quantity_input(c1, "A_TRANS", to_float(a0, 0.0), "force", key="trans_A_coef_N", step_canonical=0.1, format_str="%.2f")
            ctx["trans_B_coef_Npkph"] = quantity_input(c2, "B_TRANS", to_float(b0, 0.0), "force_per_speed", key="trans_B_coef_Npkph", step_canonical=0.001, format_str="%.5f")
            ctx["trans_C_coef_Npkph2"] = quantity_input(c3, "C_TRANS", to_float(c0, 0.0), "force_per_speed_squared", key="trans_C_coef_Npkph2", step_canonical=0.0001, format_str="%.6f")
            st.info("Calculation status: manual transmission losses are active.")

    # Keep the legacy alias in sync with the DB-oriented name used by Sprint 5 services.
    ctx["trans_B_Npkph"] = ctx.get("trans_B_coef_Npkph", 0.0)

    applied_trans = {
        "A": float(to_float(ctx.get("trans_A_coef_N"), 0.0) or 0.0),
        "B": float(to_float(ctx.get("trans_B_coef_Npkph"), 0.0) or 0.0),
        "C": float(to_float(ctx.get("trans_C_coef_Npkph2"), 0.0) or 0.0),
    }

    with st.container(border=True):
        _render_tire_editor_block_header(
            "Instant TOTAL -> NET Preview",
            "Read-only bridge preview using the current ABC_TOTAL and applied transmission losses.",
        )
        preview = _safe_workflow_preview(ctx)
        abc_total = dict(preview.get("abc_total") or {})
        abc_net = dict(preview.get("abc_net") or {})
        vde_net = dict(preview.get("vde_net") or {})
        transmission_ready = source == "Manual" or (source == "Baseline" and baseline_available)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current ABC_TOTAL", _compact_abc(abc_total))
        c2.metric("Applied ABC_TRANS", _compact_abc(applied_trans))
        if transmission_ready and abc_net:
            c3.metric("Resolved ABC_NET", _compact_abc(abc_net))
            c4.metric("VDE_NET", _format_energy_value(vde_net.get("mj_per_km")))
            st.caption("ABC_NET = ABC_TOTAL - ABC_TRANS. VDE_TOTAL is still based on ABC_TOTAL.")
        else:
            c3.metric("Resolved ABC_NET", "Unavailable")
            c4.metric("VDE_NET", "Unavailable")
            st.caption("VDE_TOTAL remains available; NET outputs require transmission losses.")

    st.dataframe(
        pd.DataFrame([_transmission_component_row(ctx, prefill)]),
        use_container_width=True,
        hide_index=True,
    )


def render_vde_edit_delete_panel(*, defaults_path, reset_ctx, defaults_df_getter=None):
    st.markdown("---")
    with st.expander("Existing row maintenance", expanded=False):
        st.subheader("Edit / Delete an existing VDE row")

        rows = fetch_vde_edit_rows(limit=100)
        if not rows:
            st.info("No VDE rows saved yet.")
            return

        labels = [
            f'#{r["id"]} - {r["legislation"]} | {r["category"]} | {r["make"]} {r["model"]} ({r.get("year","")})'
            for r in rows
        ]
        idx = st.selectbox("Pick a VDE to edit/delete", list(range(len(labels))), format_func=lambda i: labels[i])
        sel = rows[idx]
        vde_id_edit = sel["id"]
        st.caption(f"Editing VDE id: {vde_id_edit}")

        try:
            linked_n = fetch_linked_fuelcons_count(vde_id_edit)
            st.caption(f"Linked scenarios in fuelcons_db: {linked_n}")
        except Exception:
            pass

        with st.form(key=f"edit_vde_{vde_id_edit}"):
            c1, c2, c3, c4 = st.columns(4)
            a_edit = quantity_input(c1, "A", to_float(sel["coast_A_N"], 0.0), "force", key=f"edit_vde_a_{vde_id_edit}", min_canonical=0.0, max_canonical=5000.0, step_canonical=0.1, format_str="%.2f")
            b_edit = quantity_input(c2, "B", to_float(sel["coast_B_N_per_kph"], 0.0), "force_per_speed", key=f"edit_vde_b_{vde_id_edit}", min_canonical=-5.0, max_canonical=5.0, step_canonical=0.01, format_str="%.5f")
            c_edit = quantity_input(c3, "C", to_float(sel["coast_C_N_per_kph2"], 0.0), "force_per_speed_squared", key=f"edit_vde_c_{vde_id_edit}", min_canonical=0.0, max_canonical=1.0, step_canonical=0.000001, format_str="%.6f")
            m_edit = quantity_input(c4, "Curb weight", to_float(sel["mass_kg"], 0.0), "mass", key=f"edit_vde_mass_{vde_id_edit}", min_canonical=1.0, max_canonical=4000.0, step_canonical=1.0, format_str="%.1f")

            test_mass_prefill = to_float(sel.get("test_mass_kg"))
            test_mass_default = resolve_test_mass_kg(
                {
                    "legislation": sel.get("legislation"),
                    "mass_kg": m_edit,
                    "test_mass_kg": None,
                }
            )
            use_default_edit = st.checkbox(
                "Use default test mass",
                value=not (test_mass_prefill is not None and test_mass_default is not None and abs(test_mass_prefill - test_mass_default) > 1e-9),
                key=f"edit_vde_use_default_{vde_id_edit}",
            )
            if use_default_edit:
                test_mass_edit = None
                st.caption(f"Test mass: {format_quantity(test_mass_default, 'mass', format_str='%.1f')}")
            else:
                test_mass_edit = quantity_input(
                    st,
                    "Test mass",
                    max(test_mass_prefill if test_mass_prefill is not None else (test_mass_default or m_edit), m_edit),
                    "mass",
                    key=f"edit_vde_manual_test_mass_{vde_id_edit}",
                    min_canonical=float(m_edit),
                    max_canonical=4000.0,
                    step_canonical=1.0,
                    format_str="%.1f",
                )
            hint = build_test_mass_hint({"legislation": sel.get("legislation")})
            if hint:
                st.caption(hint)

            meta1, meta2, meta3, meta4 = st.columns(4)
            meta1.metric("Legislation", str(sel.get("legislation") or "-"))
            category_edit = meta2.text_input("Category", value=str(sel.get("category") or ""))
            electrification_options = ["ICE", "HEV", "PHEV", "BEV"]
            electrification_default = str(sel.get("electrification") or "ICE").upper()
            if electrification_default not in electrification_options:
                electrification_options.append(electrification_default)
            electrification_edit = meta3.selectbox(
                "Electrification",
                electrification_options,
                index=electrification_options.index(electrification_default),
            )
            transmission_options = ["AT", "AMT", "CVT", "MT", "OT"]
            transmission_default = str(sel.get("transmission_type") or "AT").upper()
            if transmission_default not in transmission_options:
                transmission_options.append(transmission_default)
            transmission_type_edit = meta4.selectbox(
                "Transmission",
                transmission_options,
                index=transmission_options.index(transmission_default),
            )

            c5, c6, c7, c8 = st.columns(4)
            make_edit = c5.text_input("Make", value=sel["make"] or "")
            model_edit = c6.text_input("Model", value=sel["model"] or "")
            year_edit = c7.number_input("Year", 1990, 2100, int(sel["year"] or 2020))
            cycle_name_edit = c8.text_input("Cycle name", value=str(sel.get("cycle_name") or default_cycle_for_legislation(sel.get("legislation", "EPA")) or ""))

            extra1, extra2, extra3, extra4 = st.columns(4)
            cda_edit = extra1.number_input("CdA [m^2]", value=float(to_float(sel.get("cda_m2"), 0.0) or 0.0), step=0.001, format="%.4f")
            tire_size_edit = extra2.text_input("Tire size", value=str(sel.get("tire_size") or ""))
            weight_dist_edit = extra3.number_input(
                "Front weight distribution [%]",
                min_value=0.0,
                max_value=100.0,
                value=float(to_float(sel.get("weight_dist_fr_pct"), 50.0) or 50.0),
                step=0.5,
                format="%.1f",
            )
            mass_basis_options = ["TWC", "TEST_MASS"]
            mass_basis_default = str(resolve_tire_load_mass_basis(sel) or "TEST_MASS").upper()
            if mass_basis_default not in mass_basis_options:
                mass_basis_default = "TEST_MASS"
            tire_load_mass_basis_edit = extra4.selectbox(
                "VDE mass basis",
                mass_basis_options,
                index=mass_basis_options.index(mass_basis_default),
            )

            notes_edit = st.text_area("Notes", value=sel["notes"] or "")

            if st.form_submit_button("Save changes"):
                try:
                    core_updates = build_edit_core_update(
                        A=a_edit,
                        B=b_edit,
                        C=c_edit,
                        mass_kg=m_edit,
                        test_mass_kg=to_float(test_mass_edit) or None,
                        make=make_edit,
                        model=model_edit,
                        year=int(year_edit),
                        notes=notes_edit,
                    )
                    extra_edit_updates = {
                        "category": str(category_edit or "").strip(),
                        "electrification": str(electrification_edit or "").strip().upper(),
                        "transmission_type": str(transmission_type_edit or "").strip().upper(),
                        "cycle_name": str(cycle_name_edit or "").strip(),
                        "cda_m2": float(to_float(cda_edit, 0.0) or 0.0),
                        "tire_size": str(tire_size_edit or "").strip(),
                        "weight_dist_fr_pct": float(to_float(weight_dist_edit, 50.0) or 50.0),
                        "tire_load_mass_basis": str(tire_load_mass_basis_edit or "TEST_MASS").strip().upper(),
                    }

                    rr_updates = collect_ctx_updates(
                        st.session_state.get("ctx"),
                        [
                            "tire_size",
                            "rrc_N_per_kN",
                            "crr1_frac_at_120kph",
                            "front_pressure_psi",
                            "rear_pressure_psi",
                            "rr_load_kpa",
                            "smerf",
                        ],
                        include_none=False,
                    )
                    if extra_edit_updates.get("tire_size"):
                        rr_updates["tire_size"] = extra_edit_updates["tire_size"]

                    pb_updates = collect_ctx_updates(
                        st.session_state.get("ctx"),
                        [
                            "parasitic_A_coef_N",
                            "parasitic_B_coef_Npkph",
                            "parasitic_C_coef_Npkph2",
                            "brake_A_coef_N",
                            "brake_B_Npkph",
                            "brake_C_coef_Npkph2",
                        ],
                        include_none=True,
                    )
                    decomp_upd = {}

                    try:
                        defaults_df = (
                            defaults_df_getter()
                            if callable(defaults_df_getter)
                            else load_vde_defaults(defaults_path)
                        )
                        decomp = estimate_aux_from_coastdown(
                            A_N=a_edit,
                            B_N_per_kph=b_edit,
                            C_N_per_kph2=c_edit,
                            mass_kg=m_edit,
                            category=extra_edit_updates.get("category") or sel.get("category", ""),
                            electrification=extra_edit_updates.get("electrification") or sel.get("electrification", "ICE"),
                            transmission_type=extra_edit_updates.get("transmission_type") or sel.get("transmission_type", "AT"),
                            cdA_override_m2=extra_edit_updates.get("cda_m2"),
                            defaults_df=defaults_df,
                        )
                        decomp_upd = build_decomp_update_for_edit(decomp)
                    except Exception:
                        decomp_upd = {}

                    leg_row = sel.get("legislation", "EPA")
                    try:
                        cycle_name = extra_edit_updates.get("cycle_name") or default_cycle_for_legislation(leg_row)
                        df_cycle = load_cycle_csv(cycle_name) if cycle_name else None
                    except Exception:
                        df_cycle = None

                    if isinstance(df_cycle, pd.DataFrame) and not df_cycle.empty:
                        preview = compute_vde_preview_from_inputs(
                            df_cycle,
                            leg_row,
                            A=a_edit,
                            B=b_edit,
                            C=c_edit,
                            mass_kg=m_edit,
                        )
                        if not preview.get("ok"):
                            raise ValueError(preview.get("error", "Failed to recompute VDE."))

                        total_mj_km = float(preview["total_mj_km"])
                        by_phase = dict(preview.get("by_phase", {}))
                        show_vde_feedback(total_mj_km, by_phase)

                        phase_updates = merge_update_payloads(
                            {"vde_net_mj_per_km": total_mj_km},
                            build_vde_phase_update(
                                df_cycle,
                                leg_row,
                                A=a_edit,
                                B=b_edit,
                                C=c_edit,
                                mass_kg=m_edit,
                            ),
                        )
                        final_updates = merge_update_payloads(
                            core_updates,
                            extra_edit_updates,
                            rr_updates,
                            pb_updates,
                            decomp_upd,
                            phase_updates,
                        )
                        if final_updates:
                            update_vde_snapshot(vde_id_edit, final_updates)
                    else:
                        final_updates = merge_update_payloads(
                            core_updates,
                            extra_edit_updates,
                            rr_updates,
                            pb_updates,
                            decomp_upd,
                        )
                        if final_updates:
                            update_vde_snapshot(vde_id_edit, final_updates)
                        st.warning("Row updated, but cycle could not be loaded; phase VDE not recomputed.")

                    st.success("Row updated.")
                    reset_ctx(preserve_meta=True)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to update: {e}")

        with st.container(border=True):
            st.warning("Delete is irreversible. Linked fuelcons_db rows will be deleted as well.")
            confirm_text = st.text_input("Type DELETE to confirm:")
            delete_disabled = (str(confirm_text).strip().upper() != "DELETE")
            if st.button(f"Delete VDE id={vde_id_edit}", type="secondary", disabled=delete_disabled):
                try:
                    deleted = delete_vde_snapshot(vde_id_edit)
                    if deleted > 0:
                        st.success(f"VDE id={vde_id_edit} deleted.")
                        reset_ctx(preserve_meta=True)
                        st.rerun()
                    else:
                        st.warning(f"VDE id={vde_id_edit} was not found (no row deleted).")
                except Exception as e:
                    st.error(f"Failed to delete: {e}")
