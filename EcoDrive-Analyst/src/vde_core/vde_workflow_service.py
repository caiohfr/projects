"""
Additive VDE workflow service for Sprint 5.

This module is intentionally separate from the legacy VDE setup helpers.
It provides a safer migration path:

- preview is pure and does not persist
- TOTAL and NET semantics are explicit
- callers can adopt it incrementally without replacing the old page flow
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from src.vde_core.cycles import default_cycle_for_legislation
from src.vde_core.roadload import cdA_to_C
from src.vde_core.repositories import (
    delete_vde_by_id,
    fetch_vde_by_id,
    insert_vde_row,
    update_vde_by_id,
)
from src.vde_core.services import estimate_aux_from_coastdown
from src.vde_core.test_mass import inertia_class_from_mass
from src.vde_core.tire_roadload_service import build_tire_component_from_result
from src.vde_core.vde_setup_service import (
    build_vde_insert_row,
    build_vde_phase_update,
    compute_vde_preview_from_inputs,
    resolve_test_mass_kg,
    to_float,
)


TOTAL_COMPONENT_ROLES = {"TOTAL_COMPONENT", "RESIDUAL", "UNKNOWN", "RESIDUAL_UNKNOWN"}
NET_SUBTRACTION_ROLE = "NET_SUBTRACTION"
INSERT_MODES = {"insert_new", "save_as_new", "new"}
UPDATE_MODES = {"update_existing", "update"}
DELETE_MODES = {"delete_existing", "delete"}
DEACTIVATE_MODES = {"deactivate_existing", "deactivate"}


def _clean_text(value: Any, default: str | None = None, *, upper: bool = False) -> str | None:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return text.upper() if upper else text


def _mode_tokens(mode: Any) -> str:
    return _clean_text(mode, "", upper=True) or ""


def _is_baseline_mode(mode: Any, baseline_id: Any = None) -> bool:
    tokens = _mode_tokens(mode)
    return bool(
        baseline_id
        or "FROM BASELINE" in tokens
        or tokens in {"BASELINE", "BASELINE EDITABLE"}
    )


def _is_component_build_up_mode(mode: Any) -> bool:
    tokens = _mode_tokens(mode)
    return any(
        marker in tokens
        for marker in (
            "NEW LINE",
            "MANUAL / TEST",
            "DEFINE ALL PARAMETERS",
            "NO BASELINE",
            "NEW MANUAL",
        )
    )


def _uses_component_build_up(data: dict[str, Any], *, baseline_id: Any = None) -> bool:
    source_ui = _clean_text(data.get("abc_total_source_ui"), "", upper=True) or ""
    mode = data.get("mode")
    from_delta = _clean_text(data.get("from_delta"), "", upper=True) or ""

    if source_ui in {
        "COMPONENT BUILD-UP",
        "COMPONENT_BUILD_UP",
        "BUILD/SYNTHESIZE ABC_TOTAL FROM COMPONENTS",
        "BUILD FROM COMPONENTS",
    }:
        return True
    if source_ui in {
        "FROM TEST COASTDOWN",
        "MEASURED/TEST COASTDOWN ABC_TOTAL",
        "MEASURED TEST ABC_TOTAL",
        "NEW TEST ABC",
        "INSERT NEW TEST FINAL ABC",
        "MANUAL",
        "MEASURED",
    }:
        return False
    if source_ui in {
        "BASELINE ABC",
        "BASELINE ABC_TOTAL",
        "INHERIT BASELINE ABC_TOTAL",
    }:
        return False

    if _mode_tokens(mode) and "DEFINE ALL PARAMETERS" in _mode_tokens(mode):
        return True
    return False


def _normalize_abc_dict(payload: Any, *, keys: tuple[str, str, str] = ("A", "B", "C")) -> dict[str, float]:
    data = dict(payload or {})
    return {
        "A": float(to_float(data.get(keys[0]), 0.0) or 0.0),
        "B": float(to_float(data.get(keys[1]), 0.0) or 0.0),
        "C": float(to_float(data.get(keys[2]), 0.0) or 0.0),
    }


def _abc_from_vde_row(row: dict | None) -> dict[str, float]:
    data = dict(row or {})
    return {
        "A": float(to_float(data.get("coast_A_N"), 0.0) or 0.0),
        "B": float(to_float(data.get("coast_B_N_per_kph"), 0.0) or 0.0),
        "C": float(to_float(data.get("coast_C_N_per_kph2"), 0.0) or 0.0),
    }


def _resolve_baseline_row(payload: dict, warnings: list[str]) -> dict:
    baseline_row = payload.get("baseline_row")
    if isinstance(baseline_row, dict):
        row = dict(baseline_row)
    else:
        row = {}

    baseline_id = payload.get("baseline_id")
    if baseline_id is not None and not row:
        fetched = fetch_vde_by_id(int(baseline_id))
        if not fetched:
            raise ValueError(f"Baseline VDE row not found: id={baseline_id}")
        row = dict(fetched)

    if row and row.get("vde_total_mj_per_km") in (None, "") and row.get("vde_net_mj_per_km") not in (None, ""):
        warnings.append("legacy_vde_net_used_as_total_candidate")

    return row


def resolve_mass_setup(payload: dict, baseline_row: dict | None = None) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    merged = dict(baseline_row or {})
    merged.update(dict(payload or {}))

    mass_kg = to_float(merged.get("mass_kg"))
    inertia_class = to_float(merged.get("inertia_class"))
    twc_kg = to_float(merged.get("twc_kg"))
    etw_kg = to_float(merged.get("etw_kg"))
    weight_dist = to_float(merged.get("weight_dist_fr_pct"))
    mass_basis = _clean_text(merged.get("mass_basis") or merged.get("tire_load_mass_basis"), "TEST_MASS", upper=True)
    legislation = _clean_text(merged.get("legislation"), "", upper=True) or ""

    if weight_dist is None:
        warnings.append("weight_distribution_missing_default_50pct")
        weight_dist = 50.0

    test_mass_kg = resolve_test_mass_kg(merged)

    resolved_source = "test_mass_kg"
    resolved_mass_used_kg = test_mass_kg
    if mass_basis == "TWC":
        if legislation == "EPA" and mass_kg is not None and mass_kg > 0:
            resolved_mass_used_kg = inertia_class_from_mass(mass_kg)
            resolved_source = "inertia_class_from_mass"
        else:
            resolved_mass_used_kg = twc_kg or etw_kg or inertia_class
            resolved_source = (
                "twc_kg"
                if twc_kg is not None
                else "etw_kg"
                if etw_kg is not None
                else "inertia_class"
                if inertia_class is not None
                else "missing_twc"
            )
        if resolved_mass_used_kg is None:
            warnings.append("twc_selected_but_inertia_class_missing")
    elif resolved_mass_used_kg is None:
        resolved_mass_used_kg = mass_kg
        resolved_source = "mass_kg_fallback"

    return (
        {
            "mass_kg": mass_kg,
            "test_mass_kg": test_mass_kg,
            "inertia_class": inertia_class,
            "twc_kg": twc_kg,
            "etw_kg": etw_kg,
            "weight_dist_fr_pct": weight_dist,
            "mass_basis": mass_basis,
            "resolved_mass_used_kg": resolved_mass_used_kg,
            "resolved_mass_source": resolved_source,
        },
        warnings,
    )


def _normalize_component(name: str, payload: Any) -> dict[str, Any]:
    if hasattr(payload, "A") and hasattr(payload, "B") and hasattr(payload, "C"):
        role = _clean_text(getattr(payload, "role", None), "TOTAL_COMPONENT", upper=True)
        source = _clean_text(getattr(payload, "source", None), "object")
        meta = getattr(payload, "meta", None) or {}
        abc = {
            "A": float(to_float(getattr(payload, "A", None), 0.0) or 0.0),
            "B": float(to_float(getattr(payload, "B", None), 0.0) or 0.0),
            "C": float(to_float(getattr(payload, "C", None), 0.0) or 0.0),
        }
    else:
        data = dict(payload or {})
        role = _clean_text(data.get("role"), "TOTAL_COMPONENT", upper=True)
        source = _clean_text(data.get("source"), "manual")
        meta = dict(data.get("meta") or {})
        abc = _normalize_abc_dict(data)

    return {
        "name": name,
        "role": role,
        "source": source,
        "A": abc["A"],
        "B": abc["B"],
        "C": abc["C"],
        "meta": meta,
    }


def _normalize_components(payload: dict) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

    raw_components = payload.get("components") or {}
    if isinstance(raw_components, dict):
        for name, item in raw_components.items():
            normalized.append(_normalize_component(str(name), item))
    elif isinstance(raw_components, list):
        for idx, item in enumerate(raw_components):
            name = _clean_text((item or {}).get("name"), f"component_{idx + 1}") if isinstance(item, dict) else f"component_{idx + 1}"
            normalized.append(_normalize_component(str(name), item))

    tire_component = payload.get("tire_component")
    if tire_component is not None:
        normalized.append(_normalize_component("tire", tire_component))

    return normalized


def _sum_components(components: list[dict[str, Any]], *, allowed_roles: set[str]) -> dict[str, float]:
    total = {"A": 0.0, "B": 0.0, "C": 0.0}
    for component in components:
        role = _clean_text(component.get("role"), "", upper=True) or ""
        if role not in allowed_roles:
            continue
        total["A"] += float(to_float(component.get("A"), 0.0) or 0.0)
        total["B"] += float(to_float(component.get("B"), 0.0) or 0.0)
        total["C"] += float(to_float(component.get("C"), 0.0) or 0.0)
    return total


def _subtract_abc(lhs: dict[str, float], rhs: dict[str, float]) -> dict[str, float]:
    return {
        "A": float(lhs["A"] - rhs["A"]),
        "B": float(lhs["B"] - rhs["B"]),
        "C": float(lhs["C"] - rhs["C"]),
    }


def _add_abc(lhs: dict[str, float], rhs: dict[str, float]) -> dict[str, float]:
    return {
        "A": float(lhs["A"] + rhs["A"]),
        "B": float(lhs["B"] + rhs["B"]),
        "C": float(lhs["C"] + rhs["C"]),
    }


def _resolve_initial_abc_total(payload: dict, baseline_row: dict, components: list[dict[str, Any]]) -> tuple[dict[str, float], str]:
    source = _clean_text(payload.get("initial_abc_total_source"), "manual", upper=True) or "MANUAL"
    if source == "BASELINE":
        if not baseline_row:
            raise ValueError("initial_abc_total_source=BASELINE requires a baseline row")
        return _abc_from_vde_row(baseline_row), source
    if source in {"COMPONENT_BUILD_UP", "COMPONENTS"}:
        return _sum_components(components, allowed_roles=TOTAL_COMPONENT_ROLES), source

    initial_abc = payload.get("initial_abc_total")
    if initial_abc is not None:
        return _normalize_abc_dict(initial_abc), source
    return _normalize_abc_dict(payload), source


def _build_components_from_ctx(ctx: dict) -> dict[str, Any]:
    data = dict(ctx or {})
    components: dict[str, Any] = {}
    mode = _clean_text(data.get("mode"), "", upper=True) or ""
    from_delta = _clean_text(data.get("from_delta"), "", upper=True) or ""
    use_component_build_up = _uses_component_build_up(data)
    tire_mode = _clean_text(data.get("component_mode_tires"), "", upper=False) or ""
    aero_mode = _clean_text(data.get("component_mode_aerodynamics"), "", upper=False) or ""
    brake_mode = _clean_text(data.get("component_mode_brakes"), "", upper=False) or ""
    parasitic_mode = _clean_text(data.get("component_mode_parasitics_hubs_axle"), "", upper=False) or ""

    if use_component_build_up:
        tire_source = _clean_text(data.get("tire_component_source"), "MANUAL RR", upper=True) or "MANUAL RR"
        rr_a = float(to_float(data.get("rr_alpha_N"), 0.0) or 0.0)
        rr_b = float(to_float(data.get("rr_beta_Npkph"), 0.0) or 0.0)
        if tire_mode in {"", "REPLACE / MANUAL INPUT"} and tire_source == "MANUAL RR" and (abs(rr_a) > 0.0 or abs(rr_b) > 0.0):
            components["tires_manual_rr"] = {
                "role": "TOTAL_COMPONENT",
                "source": "ctx_manual_rr",
                "A": rr_a,
                "B": rr_b,
                "C": 0.0,
            }

        if tire_mode in {"", "LOOKUP FROM DB"} and data.get("include_tire_component") and data.get("tire_preview_result"):
            components["tire"] = build_tire_component_from_result(data["tire_preview_result"])

        aero_c = float(to_float(data.get("aero_C_coef_Npkph2"), 0.0) or 0.0)
        if aero_mode in {"", "REPLACE / MANUAL INPUT"} and abs(aero_c) > 0.0:
            components["aerodynamics"] = {
                "role": "TOTAL_COMPONENT",
                "source": "ctx_aero_component",
                "A": 0.0,
                "B": 0.0,
                "C": aero_c,
            }

        par_a = float(to_float(data.get("parasitic_A_coef_N"), 0.0) or 0.0)
        par_b = float(to_float(data.get("parasitic_B_Npkph"), 0.0) or 0.0)
        par_c = float(to_float(data.get("parasitic_C_coef_Npkph2"), 0.0) or 0.0)
        if parasitic_mode in {"", "REPLACE / MANUAL INPUT"} and (abs(par_a) > 0.0 or abs(par_b) > 0.0 or abs(par_c) > 0.0):
            components["parasitics"] = {
                "role": "TOTAL_COMPONENT",
                "source": "ctx_parasitics_component",
                "A": par_a,
                "B": par_b,
                "C": par_c,
            }

        brake_a = float(to_float(data.get("brake_A_coef_N"), 0.0) or 0.0)
        brake_b = float(to_float(data.get("brake_B_Npkph"), 0.0) or 0.0)
        brake_c = float(to_float(data.get("brake_C_coef_Npkph2"), 0.0) or 0.0)
        if brake_mode in {"", "REPLACE / MANUAL INPUT"} and (abs(brake_a) > 0.0 or abs(brake_b) > 0.0 or abs(brake_c) > 0.0):
            components["brakes"] = {
                "role": "TOTAL_COMPONENT",
                "source": "ctx_brakes_component",
                "A": brake_a,
                "B": brake_b,
                "C": brake_c,
            }

        return components

    delta_rr_a = float(to_float(data.get("delta_rr_N"), 0.0) or 0.0)
    rr_frac_120 = float(to_float(data.get("crr1_frac_at_120kph"), 0.0) or 0.0)
    if tire_mode in {"", "Apply delta"} and abs(delta_rr_a) > 0.0:
        components["rolling_resistance_delta"] = {
            "role": "TOTAL_COMPONENT",
            "source": "ctx_delta_rr",
            "A": delta_rr_a,
            "B": delta_rr_a * (rr_frac_120 / 120.0),
            "C": 0.0,
        }

    delta_brake_a = float(to_float(data.get("delta_brake_N"), 0.0) or 0.0)
    if brake_mode in {"", "Apply delta"} and abs(delta_brake_a) > 0.0:
        components["brakes_delta"] = {
            "role": "TOTAL_COMPONENT",
            "source": "ctx_delta_brake",
            "A": delta_brake_a,
            "B": 0.0,
            "C": 0.0,
        }

    delta_parasitics_a = float(to_float(data.get("delta_parasitics_N"), 0.0) or 0.0)
    if parasitic_mode in {"", "Apply delta"} and abs(delta_parasitics_a) > 0.0:
        components["parasitics_delta"] = {
            "role": "TOTAL_COMPONENT",
            "source": "ctx_delta_parasitics",
            "A": delta_parasitics_a,
            "B": 0.0,
            "C": 0.0,
        }

    delta_aero_c = float(cdA_to_C(to_float(data.get("delta_aero_cdA"), 0.0) or 0.0))
    if aero_mode in {"", "Apply delta"} and abs(delta_aero_c) > 0.0:
        components["aero_delta"] = {
            "role": "TOTAL_COMPONENT",
            "source": "ctx_delta_aero",
            "A": 0.0,
            "B": 0.0,
            "C": delta_aero_c,
        }
    return components


def build_vde_workflow_payload_from_ctx(ctx: dict) -> dict[str, Any]:
    data = dict(ctx or {})
    baseline_id = data.get("vde_id_parent") or data.get("baseline_id")
    baseline_dict = dict(data.get("baseline_dict") or {})
    from_delta = _clean_text(data.get("from_delta"), "", upper=True) or ""
    mode = _clean_text(data.get("mode"), "", upper=True) or ""
    delta_mass_kg = float(to_float(data.get("delta_mass_kg"), 0.0) or 0.0)
    source_ui = _clean_text(data.get("abc_total_source_ui"), "", upper=True) or ""
    use_component_build_up = _uses_component_build_up(data, baseline_id=baseline_id)
    initial_total_source = (
        "BASELINE"
        if source_ui in {"BASELINE ABC", "BASELINE ABC_TOTAL", "INHERIT BASELINE ABC_TOTAL"}
        or (not source_ui and baseline_id and from_delta == "DELTAS")
        else "COMPONENT_BUILD_UP"
        if use_component_build_up
        else "MANUAL"
    )

    mass_kg = to_float(data.get("mass_kg"))
    if initial_total_source == "BASELINE" and mass_kg is not None:
        mass_kg = float(mass_kg) + delta_mass_kg

    trans_a = data.get("trans_A_coef_N")
    trans_b = data.get("trans_B_coef_Npkph", data.get("trans_B_Npkph"))
    trans_c = data.get("trans_C_coef_Npkph2")

    payload = {
        "baseline_id": baseline_id,
        "line_source_mode": "BASELINE" if _is_baseline_mode(mode, baseline_id) else "NEW",
        "initial_abc_total_source": initial_total_source,
        "initial_abc_total": {
            "A": data.get("A"),
            "B": data.get("B"),
            "C": data.get("C"),
        },
        "mass_kg": mass_kg,
        "test_mass_kg": data.get("test_mass_kg"),
        "inertia_class": data.get("inertia_class"),
        "twc_kg": data.get("twc_kg"),
        "etw_kg": data.get("etw_kg"),
        "mass_basis": data.get("mass_basis") or data.get("tire_load_mass_basis") or "TEST_MASS",
        "weight_dist_fr_pct": data.get("weight_dist_fr_pct"),
        "legislation": data.get("legislation"),
        "category": data.get("category"),
        "make": data.get("make"),
        "model": data.get("model"),
        "year": data.get("year"),
        "notes": data.get("notes"),
        "cycle_df": data.get("cycle_df"),
        "cycle_name": data.get("cycle_name"),
        "cycle_source": data.get("cycle_source"),
        "components": _build_components_from_ctx(data),
        "transmission_losses": {
            "source": "MANUAL"
            if any(
                to_float(data.get(key)) not in (None, 0.0)
                for key in ("trans_A_coef_N", "trans_B_coef_Npkph", "trans_B_Npkph", "trans_C_coef_Npkph2")
            )
            else "MISSING",
            "A_TRANS": trans_a,
            "B_TRANS": trans_b,
            "C_TRANS": trans_c,
        },
    }

    if baseline_dict and not baseline_id:
        payload["baseline_row"] = {
            "coast_A_N": baseline_dict.get("A"),
            "coast_B_N_per_kph": baseline_dict.get("B"),
            "coast_C_N_per_kph2": baseline_dict.get("C"),
            "mass_kg": baseline_dict.get("mass_kg"),
            "test_mass_kg": baseline_dict.get("test_mass_kg"),
            "legislation": baseline_dict.get("legislation"),
            "category": baseline_dict.get("category"),
        }

    return payload


def summarize_component_build_up_from_ctx(ctx: dict) -> dict[str, Any]:
    payload = build_vde_workflow_payload_from_ctx(ctx)
    components = _normalize_components(payload)
    total = _sum_components(components, allowed_roles=TOTAL_COMPONENT_ROLES)
    source = _clean_text(payload.get("initial_abc_total_source"), "MANUAL", upper=True) or "MANUAL"
    return {
        "enabled": source == "COMPONENT_BUILD_UP",
        "source": source,
        "components": components,
        "abc_total": total,
    }


def build_vde_setup_preview_from_ctx(ctx: dict) -> dict[str, Any]:
    preview = build_vde_setup_preview(build_vde_workflow_payload_from_ctx(ctx))
    return prepare_vde_setup_preview_for_save(preview, ctx=ctx)


def _resolve_transmission_losses(payload: dict, baseline_row: dict) -> dict[str, Any]:
    raw = dict(payload.get("transmission_losses") or {})
    source = _clean_text(raw.get("source"), None, upper=True)

    if source == "BASELINE":
        abc = {
            "A": float(to_float(baseline_row.get("trans_A_coef_N"), 0.0) or 0.0),
            "B": float(to_float(baseline_row.get("trans_B_coef_Npkph"), 0.0) or 0.0),
            "C": float(to_float(baseline_row.get("trans_C_coef_Npkph2"), 0.0) or 0.0),
        }
    else:
        abc = {
            "A": float(to_float(raw.get("A_TRANS"), to_float(raw.get("A"), 0.0)) or 0.0),
            "B": float(to_float(raw.get("B_TRANS"), to_float(raw.get("B"), 0.0)) or 0.0),
            "C": float(to_float(raw.get("C_TRANS"), to_float(raw.get("C"), 0.0)) or 0.0),
        }

    has_any = any(abs(value) > 0.0 for value in abc.values())
    if source == "MISSING" or (source is None and not has_any):
        return {
            "source": source or "MISSING",
            "status": "missing",
            "abc": None,
        }

    return {
        "source": source or "MANUAL",
        "status": "available",
        "abc": abc,
    }


def _compute_vde_energy_preview(payload: dict, abc: dict[str, float], mass_kg: float | None) -> dict[str, Any] | None:
    cycle_df = payload.get("cycle_df")
    legislation = _clean_text(payload.get("legislation"), "", upper=True) or ""
    if cycle_df is None or mass_kg is None:
        return None

    result = compute_vde_preview_from_inputs(
        cycle_df,
        legislation,
        A=abc["A"],
        B=abc["B"],
        C=abc["C"],
        mass_kg=mass_kg,
    )
    if not result.get("ok"):
        raise ValueError(result.get("error", "Could not compute VDE preview from inputs."))

    return {
        "mj_per_km": float(result["total_mj_km"]),
        "by_phase": dict(result.get("by_phase", {})),
    }


def _build_row_payload(preview_result: dict) -> dict[str, Any]:
    request = dict(preview_result.get("request") or {})
    mass_setup = dict(preview_result.get("mass_setup") or {})
    total_abc = dict(preview_result.get("abc_total") or {})
    transmission = dict(preview_result.get("transmission_losses") or {})
    total_energy = dict(preview_result.get("vde_total") or {})
    net_energy = preview_result.get("vde_net") or {}

    row = {
        "legislation": request.get("legislation"),
        "category": request.get("category"),
        "make": request.get("make"),
        "model": request.get("model"),
        "year": request.get("year"),
        "notes": request.get("notes"),
        "mass_kg": mass_setup.get("mass_kg"),
        "test_mass_kg": mass_setup.get("test_mass_kg"),
        "inertia_class": mass_setup.get("inertia_class"),
        "weight_dist_fr_pct": mass_setup.get("weight_dist_fr_pct"),
        "coast_A_N": total_abc.get("A"),
        "coast_B_N_per_kph": total_abc.get("B"),
        "coast_C_N_per_kph2": total_abc.get("C"),
        "vde_total_mj_per_km": total_energy.get("mj_per_km"),
        "vde_net_mj_per_km": net_energy.get("mj_per_km"),
        "trans_A_coef_N": (transmission.get("abc") or {}).get("A"),
        "trans_B_coef_Npkph": (transmission.get("abc") or {}).get("B"),
        "trans_C_coef_Npkph2": (transmission.get("abc") or {}).get("C"),
        "cycle_name": request.get("cycle_name"),
        "cycle_source": request.get("cycle_source"),
    }
    return {key: value for key, value in row.items() if value not in (None, "")}


def _build_deltas_from_ctx(ctx: dict | None) -> dict[str, float]:
    data = dict(ctx or {})
    return {
        "delta_rr_N": float(to_float(data.get("delta_rr_N"), 0.0) or 0.0),
        "delta_brake_N": float(to_float(data.get("delta_brake_N"), 0.0) or 0.0),
        "delta_parasitics_N": float(to_float(data.get("delta_parasitics_N"), 0.0) or 0.0),
        "delta_aero_Npkph2": float(cdA_to_C(to_float(data.get("delta_aero_cdA"), 0.0) or 0.0)),
        "delta_mass_kg": float(to_float(data.get("delta_mass_kg"), 0.0) or 0.0),
    }


def _estimate_decomp_for_save(preview_result: dict, ctx: dict | None, defaults_df) -> dict[str, Any] | None:
    if defaults_df is None:
        return None

    preview = dict(preview_result or {})
    request = dict(preview.get("request") or {})
    mass_setup = dict(preview.get("mass_setup") or {})
    abc_total = dict(preview.get("abc_total") or {})
    if not abc_total:
        return None

    data = dict(ctx or {})
    try:
        return estimate_aux_from_coastdown(
            A_N=abc_total.get("A"),
            B_N_per_kph=abc_total.get("B"),
            C_N_per_kph2=abc_total.get("C"),
            mass_kg=mass_setup.get("resolved_mass_used_kg") or mass_setup.get("mass_kg"),
            category=request.get("category"),
            electrification=data.get("electrification", "ICE"),
            transmission_type=data.get("transmission_type", "AT"),
            cdA_override_m2=data.get("cda_m2"),
            defaults_df=defaults_df,
        )
    except Exception:
        return None


def _build_rich_save_row(preview_result: dict, ctx: dict | None, defaults_df=None) -> dict[str, Any]:
    preview = dict(preview_result or {})
    request = dict(preview.get("request") or {})
    data = dict(ctx or {})
    save_ctx = dict(data)
    mass_setup = dict(preview.get("mass_setup") or {})
    abc_total = dict(preview.get("abc_total") or {})
    abc_net = dict(preview.get("abc_net") or {})
    transmission = dict(preview.get("transmission_losses") or {})
    transmission_abc = dict(transmission.get("abc") or {})
    vde_total = dict(preview.get("vde_total") or {})
    vde_net = dict(preview.get("vde_net") or {})

    coast_abc = abc_total or abc_net
    active_energy = vde_net or vde_total
    equiv = SimpleNamespace(
        A=float(to_float(coast_abc.get("A"), 0.0) or 0.0),
        B=float(to_float(coast_abc.get("B"), 0.0) or 0.0),
        C=float(to_float(coast_abc.get("C"), 0.0) or 0.0),
        mass_kg=float(
            to_float(
                mass_setup.get("resolved_mass_used_kg"),
                to_float(mass_setup.get("mass_kg"), 0.0),
            )
            or 0.0
        ),
        component_table=[],
    )

    if request.get("mass_kg") is not None:
        save_ctx["mass_kg"] = request.get("mass_kg")
    if mass_setup.get("test_mass_kg") is not None:
        save_ctx["test_mass_kg"] = mass_setup.get("test_mass_kg")
    if mass_setup.get("inertia_class") is not None:
        save_ctx["inertia_class"] = mass_setup.get("inertia_class")

    row = build_vde_insert_row(
        save_ctx,
        leg=str(request.get("legislation") or ""),
        cat=request.get("category"),
        make=request.get("make"),
        model=request.get("model"),
        year=int(request["year"]) if str(request.get("year", "")).isdigit() else None,
        notes=request.get("notes", ""),
        cycle_name=request.get("cycle_name") or default_cycle_for_legislation(str(request.get("legislation") or "")),
        cycle_source=request.get("cycle_source") or save_ctx.get("cycle_source") or "",
        equiv=equiv,
        total_mj_km=float(to_float(active_energy.get("mj_per_km"), 0.0) or 0.0),
        by_phase=dict(active_energy.get("by_phase") or {}),
        deltas=_build_deltas_from_ctx(save_ctx),
        decomp=_estimate_decomp_for_save(preview, save_ctx, defaults_df),
    )

    if vde_total.get("mj_per_km") is not None:
        row["vde_total_mj_per_km"] = float(vde_total["mj_per_km"])
    if vde_net.get("mj_per_km") is not None:
        row["vde_net_mj_per_km"] = float(vde_net["mj_per_km"])
    else:
        row.pop("vde_net_mj_per_km", None)

    if transmission_abc:
        row["trans_A_coef_N"] = float(to_float(transmission_abc.get("A"), 0.0) or 0.0)
        row["trans_B_coef_Npkph"] = float(to_float(transmission_abc.get("B"), 0.0) or 0.0)
        row["trans_C_coef_Npkph2"] = float(to_float(transmission_abc.get("C"), 0.0) or 0.0)

    return row


def _build_phase_updates_for_save(preview_result: dict) -> dict[str, Any]:
    preview = dict(preview_result or {})
    request = dict(preview.get("request") or {})
    mass_setup = dict(preview.get("mass_setup") or {})
    abc_active = dict(preview.get("abc_net") or preview.get("abc_total") or {})
    resolved_mass = mass_setup.get("resolved_mass_used_kg") or mass_setup.get("mass_kg")
    cycle_df = request.get("cycle_df")

    if not abc_active or resolved_mass is None or cycle_df is None:
        return {}

    phase_updates = build_vde_phase_update(
        cycle_df,
        str(request.get("legislation") or ""),
        A=float(to_float(abc_active.get("A"), 0.0) or 0.0),
        B=float(to_float(abc_active.get("B"), 0.0) or 0.0),
        C=float(to_float(abc_active.get("C"), 0.0) or 0.0),
        mass_kg=float(to_float(resolved_mass, 0.0) or 0.0),
    )
    if preview.get("vde_net") is None:
        phase_updates.pop("vde_net_mj_per_km", None)
    return phase_updates


def prepare_vde_setup_preview_for_save(
    preview_result: dict,
    *,
    ctx: dict | None = None,
    defaults_df=None,
) -> dict[str, Any]:
    preview = dict(preview_result or {})
    save_payload = dict(preview.get("save_payload") or {})
    row = _build_rich_save_row(preview, ctx, defaults_df) if ctx is not None else _build_row_payload(preview)

    save_payload["insert_row"] = dict(row)
    save_payload["update_row"] = dict(row)
    preview["save_payload"] = save_payload
    preview["phase_update_row"] = _build_phase_updates_for_save(preview)
    return preview


def _format_scalar(value: Any, *, places: int = 3) -> str:
    if value in (None, ""):
        return "-"
    numeric = to_float(value)
    if numeric is not None:
        return f"{float(numeric):.{places}f}"
    return str(value)


def _format_abc_value(abc: dict | None, *, places: tuple[int, int, int] = (3, 6, 8)) -> str:
    data = dict(abc or {})
    if not data:
        return "-"
    return " / ".join(
        [
            _format_scalar(data.get("A"), places=places[0]),
            _format_scalar(data.get("B"), places=places[1]),
            _format_scalar(data.get("C"), places=places[2]),
        ]
    )


def _format_energy_value(energy: dict | None) -> str:
    data = dict(energy or {})
    if data.get("mj_per_km") in (None, ""):
        return "Unavailable"
    return f"{float(to_float(data.get('mj_per_km'), 0.0) or 0.0):.4f} MJ/km"


def _format_mass_setup_value(mass_setup: dict | None) -> str:
    data = dict(mass_setup or {})
    if not data:
        return "-"
    parts = []
    basis = _clean_text(data.get("mass_basis"), None, upper=True)
    if basis:
        parts.append(basis)
    resolved = to_float(data.get("resolved_mass_used_kg"))
    if resolved is not None:
        parts.append(f"resolved={resolved:.1f} kg")
    test_mass = to_float(data.get("test_mass_kg"))
    if test_mass is not None:
        parts.append(f"test={test_mass:.1f} kg")
    inertia = to_float(data.get("inertia_class"))
    if inertia is not None:
        parts.append(f"twc={inertia:.1f}")
    return " | ".join(parts) if parts else "-"


def _format_component_names(components: list[dict[str, Any]]) -> str:
    if not components:
        return "-"
    return ", ".join(str(component.get("name") or "component") for component in components)


def _state_allowed(value: str) -> str:
    allowed = {"Inherited", "Overridden", "Derived", "Applied", "Missing", "Not applicable"}
    return value if value in allowed else "Not applicable"


def _source_label_from_initial_source(source: str) -> str:
    normalized = _clean_text(source, "", upper=True) or ""
    if normalized == "BASELINE":
        return "Baseline ABC_TOTAL"
    if normalized in {"COMPONENT_BUILD_UP", "COMPONENTS"}:
        return "Component Build-up"
    return "Measured/test coastdown"


def _reference_snapshot_from_preview(ctx: dict, preview: dict) -> dict[str, Any]:
    request = dict(preview.get("request") or {})
    line_source = dict(preview.get("line_source") or {})
    baseline_row = dict(preview.get("baseline_row") or {})
    initial_source = _clean_text(preview.get("initial_abc_total_source"), "MANUAL", upper=True) or "MANUAL"
    has_baseline = bool(baseline_row) or line_source.get("baseline_id") is not None

    if has_baseline:
        kind = "Baseline Snapshot"
        state = "Inherited"
        abc_value = _abc_from_vde_row(baseline_row) or dict(preview.get("initial_abc_total_base") or {})
        source = "baseline_row"
        mass_kg = baseline_row.get("mass_kg")
        test_mass_kg = baseline_row.get("test_mass_kg")
        cycle_name = baseline_row.get("cycle_name")
    elif initial_source in {"COMPONENT_BUILD_UP", "COMPONENTS"}:
        kind = "Component Build-up Reference"
        state = "Derived"
        abc_value = dict(preview.get("component_abc_total") or preview.get("initial_abc_total_base") or {})
        source = "component_build_up"
        mass_kg = request.get("mass_kg")
        test_mass_kg = request.get("test_mass_kg")
        cycle_name = request.get("cycle_name")
    else:
        kind = "Measured Coastdown Reference"
        state = "Applied"
        abc_value = dict(preview.get("initial_abc_total_base") or {})
        source = "manual_coastdown"
        mass_kg = request.get("mass_kg")
        test_mass_kg = request.get("test_mass_kg")
        cycle_name = request.get("cycle_name")

    return {
        "kind": kind,
        "state": _state_allowed(state),
        "line_source_mode": _clean_text(line_source.get("mode"), "NEW", upper=True),
        "baseline_id": line_source.get("baseline_id"),
        "abc": abc_value,
        "mass_kg": mass_kg,
        "test_mass_kg": test_mass_kg,
        "mass_basis": request.get("mass_basis") or request.get("tire_load_mass_basis"),
        "cycle_name": cycle_name,
        "source": source,
    }


def _working_scenario_summary_from_preview(preview: dict, save_payload: dict) -> dict[str, Any]:
    request = dict(preview.get("request") or {})
    return {
        "line_source": dict(preview.get("line_source") or {}),
        "roadload_basis": _clean_text(preview.get("initial_abc_total_source"), "MANUAL", upper=True),
        "mass_setup": dict(preview.get("mass_setup") or {}),
        "components": list(preview.get("components") or []),
        "transmission_losses": dict(preview.get("transmission_losses") or {}),
        "abc_total": dict(preview.get("abc_total") or {}),
        "abc_net": dict(preview.get("abc_net") or {}) if preview.get("abc_net") is not None else None,
        "vde_total": dict(preview.get("vde_total") or {}) if preview.get("vde_total") is not None else None,
        "vde_net": dict(preview.get("vde_net") or {}) if preview.get("vde_net") is not None else None,
        "cycle_name": request.get("cycle_name"),
        "cycle_source": request.get("cycle_source"),
        "target_vde_id": dict(save_payload or {}).get("target_vde_id"),
        "warnings": list(preview.get("warnings") or []),
    }


def _matching_components(preview: dict, names: set[str]) -> list[dict[str, Any]]:
    matches = []
    for component in list(preview.get("components") or []):
        name = _clean_text(component.get("name"), "", upper=True) or ""
        if name in names:
            matches.append(dict(component))
    return matches


def _aggregate_component_abc(components: list[dict[str, Any]]) -> dict[str, float] | None:
    if not components:
        return None
    return _sum_components(components, allowed_roles={_clean_text(component.get("role"), "TOTAL_COMPONENT", upper=True) or "TOTAL_COMPONENT" for component in components})


def _change_item(state: str, *, reference: str, working: str, change: str, source: str) -> dict[str, str]:
    return {
        "state": _state_allowed(state),
        "reference": reference,
        "working": working,
        "change": change,
        "source": source,
    }


def _compare_mass_change(reference_snapshot: dict, mass_setup: dict, has_baseline: bool) -> dict[str, str]:
    working = _format_mass_setup_value(mass_setup)
    reference_mass_setup = {
        "mass_basis": reference_snapshot.get("mass_basis"),
        "resolved_mass_used_kg": reference_snapshot.get("mass_kg"),
        "test_mass_kg": reference_snapshot.get("test_mass_kg"),
    }
    reference = _format_mass_setup_value(reference_mass_setup)
    if not mass_setup:
        return _change_item("Missing", reference=reference, working="-", change="Mass setup unavailable", source="preview.mass_setup")

    resolved_source = _clean_text(mass_setup.get("resolved_mass_source"), "", upper=False) or ""
    same_reference = (
        to_float(reference_snapshot.get("mass_kg")) == to_float(mass_setup.get("mass_kg"))
        and to_float(reference_snapshot.get("test_mass_kg")) == to_float(mass_setup.get("test_mass_kg"))
    )
    if has_baseline and same_reference:
        state = "Inherited"
        change = "No change"
    elif resolved_source == "inertia_class_from_mass":
        state = "Derived"
        change = f"Resolved via {resolved_source}"
    elif has_baseline:
        state = "Overridden"
        change = "Vehicle mass inputs changed from reference"
    else:
        state = "Applied"
        change = "Vehicle mass inputs defined for this scenario"
    return _change_item(state, reference=reference, working=working, change=change, source=f"mass_setup.{resolved_source or 'resolved'}")


def _compare_component_change(
    *,
    preview: dict,
    component_names: set[str],
    label: str,
    has_baseline: bool,
    roadload_basis: str,
) -> dict[str, str]:
    components = _matching_components(preview, component_names)
    aggregated = _aggregate_component_abc(components)
    working = _format_abc_value(aggregated) if aggregated else "-"
    source = _format_component_names(components) if components else "preview.components"

    if components:
        return _change_item("Applied", reference="Inherited baseline" if has_baseline else "Scenario-defined", working=working, change=f"{label} contributes explicitly to working ABC_TOTAL", source=source)
    if roadload_basis in {"COMPONENT_BUILD_UP", "COMPONENTS"}:
        return _change_item("Missing", reference="Component build-up expected", working="-", change=f"{label} not defined in component build-up", source=source)
    if has_baseline:
        return _change_item("Inherited", reference="Baseline reference", working="Inherited", change="No explicit scenario change", source=source)
    return _change_item("Not applicable", reference="No baseline reference", working="-", change=f"{label} not needed for the selected roadload basis", source=source)


def _build_change_summary(ctx: dict, preview: dict, reference_snapshot: dict) -> dict[str, dict[str, str]]:
    request = dict(preview.get("request") or {})
    mass_setup = dict(preview.get("mass_setup") or {})
    transmission = dict(preview.get("transmission_losses") or {})
    line_source = dict(preview.get("line_source") or {})
    roadload_basis = _clean_text(preview.get("initial_abc_total_source"), "MANUAL", upper=True) or "MANUAL"
    has_baseline = bool(dict(preview.get("baseline_row") or {})) or line_source.get("baseline_id") is not None

    if roadload_basis == "BASELINE":
        basis_state = "Inherited"
        basis_change = "ABC_TOTAL starts from the selected baseline row"
    elif roadload_basis in {"COMPONENT_BUILD_UP", "COMPONENTS"}:
        basis_state = "Derived"
        basis_change = "ABC_TOTAL is synthesized from component contributions"
    else:
        basis_state = "Overridden" if has_baseline else "Applied"
        basis_change = "ABC_TOTAL is provided by measured/test coastdown inputs"

    change_summary = {
        "roadload_basis": _change_item(
            basis_state,
            reference=reference_snapshot["kind"],
            working=_source_label_from_initial_source(roadload_basis),
            change=basis_change,
            source=f"preview.initial_abc_total_source={roadload_basis}",
        ),
        "mass": _compare_mass_change(reference_snapshot, mass_setup, has_baseline),
        "tires": _compare_component_change(
            preview=preview,
            component_names={"TIRE", "TIRES_MANUAL_RR", "ROLLING_RESISTANCE_DELTA"},
            label="Tires",
            has_baseline=has_baseline,
            roadload_basis=roadload_basis,
        ),
        "aero": _compare_component_change(
            preview=preview,
            component_names={"AERODYNAMICS", "AERO_DELTA"},
            label="Aerodynamics",
            has_baseline=has_baseline,
            roadload_basis=roadload_basis,
        ),
        "brakes": _compare_component_change(
            preview=preview,
            component_names={"BRAKES", "BRAKES_DELTA"},
            label="Brakes",
            has_baseline=has_baseline,
            roadload_basis=roadload_basis,
        ),
        "parasitics": _compare_component_change(
            preview=preview,
            component_names={"PARASITICS", "PARASITICS_DELTA"},
            label="Parasitics",
            has_baseline=has_baseline,
            roadload_basis=roadload_basis,
        ),
        "trailer": _compare_component_change(
            preview=preview,
            component_names={"TRAILER"},
            label="Trailer",
            has_baseline=has_baseline,
            roadload_basis=roadload_basis,
        ),
    }

    transmission_abc = dict(transmission.get("abc") or {}) if transmission.get("abc") is not None else None
    baseline_transmission = dict(preview.get("baseline_row") or {})
    baseline_trans_abc = {
        "A": baseline_transmission.get("trans_A_coef_N"),
        "B": baseline_transmission.get("trans_B_coef_Npkph", baseline_transmission.get("trans_B_Npkph")),
        "C": baseline_transmission.get("trans_C_coef_Npkph2"),
    } if has_baseline else None
    transmission_source_ui = _clean_text(ctx.get("transmission_losses_source"), "", upper=False) or ""
    if transmission.get("status") != "available" or transmission_abc is None:
        transmission_item = _change_item(
            "Missing",
            reference=_format_abc_value(baseline_trans_abc) if baseline_trans_abc else "No reference",
            working="Unavailable",
            change="VDE_NET cannot be resolved without transmission losses",
            source=f"preview.transmission_losses.{transmission.get('status')}",
        )
    else:
        if has_baseline and transmission_source_ui.lower() == "baseline":
            state = "Inherited"
            change = "Using baseline transmission losses"
        elif has_baseline:
            state = "Overridden"
            change = "Transmission losses differ from baseline reference"
        else:
            state = "Applied"
            change = "Scenario transmission losses provided explicitly"
        transmission_item = _change_item(
            state,
            reference=_format_abc_value(baseline_trans_abc) if baseline_trans_abc else "No baseline transmission",
            working=_format_abc_value(transmission_abc),
            change=change,
            source=f"preview.transmission_losses.{transmission.get('source') or transmission_source_ui or 'MANUAL'}",
        )
    change_summary["transmission"] = transmission_item

    cycle_name = _clean_text(request.get("cycle_name"), None)
    baseline_cycle = _clean_text((preview.get("baseline_row") or {}).get("cycle_name"), None)
    if cycle_name:
        if has_baseline and baseline_cycle and baseline_cycle == cycle_name:
            cycle_state = "Inherited"
            cycle_change = "Cycle matches the baseline snapshot"
        elif has_baseline and baseline_cycle and baseline_cycle != cycle_name:
            cycle_state = "Overridden"
            cycle_change = "Cycle changed relative to baseline"
        else:
            cycle_state = "Applied"
            cycle_change = "Cycle selected for this scenario"
        cycle_item = _change_item(
            cycle_state,
            reference=baseline_cycle or "No baseline cycle",
            working=cycle_name,
            change=cycle_change,
            source=_clean_text(request.get("cycle_source"), "request.cycle_name") or "request.cycle_name",
        )
    else:
        cycle_item = _change_item(
            "Missing",
            reference=baseline_cycle or "No baseline cycle",
            working="Missing",
            change="Cycle selection is still missing",
            source="request.cycle_name",
        )
    change_summary["cycle"] = cycle_item
    return change_summary


def _delta_text(lhs: dict | None, rhs: dict | None) -> str:
    left = dict(lhs or {})
    right = dict(rhs or {})
    if not left or not right:
        return "-"
    delta = _subtract_abc(left, right)
    return _format_abc_value(delta)


def _build_baseline_vs_working_rows(
    preview: dict,
    reference_snapshot: dict,
    working_scenario_summary: dict,
    change_summary: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    request = dict(preview.get("request") or {})
    mass_setup = dict(preview.get("mass_setup") or {})
    abc_total = dict(preview.get("abc_total") or {})
    abc_net = dict(preview.get("abc_net") or {}) if preview.get("abc_net") is not None else None
    vde_total = dict(preview.get("vde_total") or {}) if preview.get("vde_total") is not None else None
    vde_net = dict(preview.get("vde_net") or {}) if preview.get("vde_net") is not None else None
    transmission = dict(preview.get("transmission_losses") or {})

    rows.append(
        {
            "Field": "Roadload basis",
            "Reference": reference_snapshot["kind"],
            "Working scenario": _source_label_from_initial_source(working_scenario_summary.get("roadload_basis") or ""),
            "Change": change_summary["roadload_basis"]["change"],
            "Source": change_summary["roadload_basis"]["source"],
        }
    )
    rows.append(
        {
            "Field": "ABC_TOTAL",
            "Reference": _format_abc_value(reference_snapshot.get("abc")),
            "Working scenario": _format_abc_value(abc_total),
            "Change": _delta_text(abc_total, reference_snapshot.get("abc")),
            "Source": "preview.abc_total",
        }
    )
    rows.append(
        {
            "Field": "Mass / test mass / TWC",
            "Reference": _format_mass_setup_value(
                {
                    "mass_basis": reference_snapshot.get("mass_basis"),
                    "resolved_mass_used_kg": reference_snapshot.get("mass_kg"),
                    "test_mass_kg": reference_snapshot.get("test_mass_kg"),
                }
            ),
            "Working scenario": _format_mass_setup_value(mass_setup),
            "Change": change_summary["mass"]["change"],
            "Source": change_summary["mass"]["source"],
        }
    )

    for field_name, key in (
        ("Tires", "tires"),
        ("Aerodynamics", "aero"),
        ("Brakes", "brakes"),
        ("Parasitics", "parasitics"),
        ("Trailer", "trailer"),
        ("Transmission losses", "transmission"),
        ("Cycle", "cycle"),
    ):
        item = change_summary[key]
        rows.append(
            {
                "Field": field_name,
                "Reference": item["reference"],
                "Working scenario": item["working"],
                "Change": item["change"],
                "Source": item["source"],
            }
        )

    rows.append(
        {
            "Field": "ABC_NET",
            "Reference": _format_abc_value(reference_snapshot.get("abc")) if reference_snapshot.get("kind") == "Baseline Snapshot" else "Not applicable",
            "Working scenario": _format_abc_value(abc_net),
            "Change": _delta_text(abc_net, reference_snapshot.get("abc")) if abc_net else "Unavailable",
            "Source": f"preview.transmission_losses.{transmission.get('status')}",
        }
    )
    rows.append(
        {
            "Field": "VDE_TOTAL",
            "Reference": "Not stored in reference layer",
            "Working scenario": _format_energy_value(vde_total),
            "Change": "Preview result",
            "Source": "preview.vde_total",
        }
    )
    rows.append(
        {
            "Field": "VDE_NET",
            "Reference": "Not stored in reference layer",
            "Working scenario": _format_energy_value(vde_net),
            "Change": "Preview result" if vde_net else "Unavailable",
            "Source": "preview.vde_net",
        }
    )

    if request.get("cycle_name") and not any(row["Field"] == "Cycle" for row in rows):
        rows.append(
            {
                "Field": "Cycle",
                "Reference": reference_snapshot.get("cycle_name") or "No reference cycle",
                "Working scenario": str(request.get("cycle_name")),
                "Change": change_summary["cycle"]["change"],
                "Source": change_summary["cycle"]["source"],
            }
        )
    return rows


def build_vde_pre_save_review(
    ctx: dict,
    workflow_preview: dict,
    save_payload: dict | None = None,
) -> dict[str, Any]:
    preview = dict(workflow_preview or {})
    if not preview.get("ok"):
        raise ValueError("Pre-save review requires a valid workflow preview.")

    staged_save_payload = save_payload if save_payload is not None else preview.get("save_payload")
    if not isinstance(staged_save_payload, dict):
        raise ValueError("Pre-save review requires an existing save_payload produced by the workflow preview.")

    reference_snapshot = _reference_snapshot_from_preview(dict(ctx or {}), preview)
    working_scenario_summary = _working_scenario_summary_from_preview(preview, staged_save_payload)
    change_summary = _build_change_summary(dict(ctx or {}), preview, reference_snapshot)
    baseline_vs_working_rows = _build_baseline_vs_working_rows(
        preview,
        reference_snapshot,
        working_scenario_summary,
        change_summary,
    )

    return {
        "reference_snapshot": reference_snapshot,
        "working_scenario_summary": working_scenario_summary,
        "change_summary": change_summary,
        "baseline_vs_working_rows": baseline_vs_working_rows,
        "staged_save_payload": staged_save_payload,
    }


def build_vde_setup_preview(payload: dict) -> dict[str, Any]:
    warnings: list[str] = []
    request = dict(payload or {})
    baseline_row = _resolve_baseline_row(request, warnings)
    mass_setup, mass_warnings = resolve_mass_setup(request, baseline_row=baseline_row)
    warnings.extend(mass_warnings)

    components = _normalize_components(request)
    initial_abc_total, total_source = _resolve_initial_abc_total(request, baseline_row, components)
    component_abc_total = _sum_components(components, allowed_roles=TOTAL_COMPONENT_ROLES)
    abc_total = (
        dict(component_abc_total)
        if total_source in {"COMPONENT_BUILD_UP", "COMPONENTS"}
        else _add_abc(initial_abc_total, component_abc_total)
    )
    transmission = _resolve_transmission_losses(request, baseline_row)

    baseline_abc = _abc_from_vde_row(baseline_row) if baseline_row else None
    resolved_mass = mass_setup.get("resolved_mass_used_kg")
    vde_total = _compute_vde_energy_preview(request, abc_total, resolved_mass)

    abc_net = None
    vde_net = None
    if transmission.get("status") == "available" and transmission.get("abc") is not None:
        abc_net = _subtract_abc(abc_total, transmission["abc"])
        vde_net = _compute_vde_energy_preview(request, abc_net, resolved_mass)
    else:
        warnings.append("vde_net_unavailable_transmission_losses_missing")

    preview = {
        "ok": True,
        "request": request,
        "line_source": {
            "mode": _clean_text(request.get("line_source_mode"), "NEW", upper=True),
            "baseline_id": request.get("baseline_id") or baseline_row.get("id"),
        },
        "baseline_row": baseline_row,
        "initial_abc_total_source": total_source,
        "initial_abc_total_base": initial_abc_total,
        "component_abc_total": component_abc_total,
        "mass_setup": mass_setup,
        "components": components,
        "abc_total": abc_total,
        "vde_total": vde_total,
        "transmission_losses": transmission,
        "abc_net": abc_net,
        "vde_net": vde_net,
        "delta_vs_baseline": {
            "abc_total": _subtract_abc(abc_total, baseline_abc) if baseline_abc else None,
            "abc_net": _subtract_abc(abc_net, baseline_abc) if (baseline_abc and abc_net) else None,
        },
        "warnings": warnings,
        "preview_saved": False,
    }
    preview["save_payload"] = {
        "target_vde_id": request.get("target_vde_id") or request.get("baseline_id") or baseline_row.get("id"),
        "insert_row": _build_row_payload(preview),
        "update_row": _build_row_payload(preview),
    }
    preview["phase_update_row"] = _build_phase_updates_for_save(preview)
    return preview


def save_vde_setup_result(
    preview_result: dict,
    save_mode: str,
    *,
    ctx: dict | None = None,
    defaults_df=None,
) -> dict[str, Any]:
    preview = prepare_vde_setup_preview_for_save(preview_result, ctx=ctx, defaults_df=defaults_df) if ctx is not None else dict(preview_result or {})
    if not preview.get("ok"):
        raise ValueError("Cannot save an invalid VDE workflow preview result.")

    mode = _clean_text(save_mode, "", upper=False) or ""
    save_payload = dict(preview.get("save_payload") or {})
    target_vde_id = save_payload.get("target_vde_id")
    phase_update_row = dict(preview.get("phase_update_row") or {})

    if mode in INSERT_MODES:
        row = dict(save_payload.get("insert_row") or {})
        inserted_id = int(insert_vde_row(row))
        if phase_update_row:
            update_vde_by_id(inserted_id, phase_update_row)
        return {
            "action": "insert_new",
            "vde_id": inserted_id,
            "row": row,
            "phase_update_row": phase_update_row,
        }

    if mode in UPDATE_MODES:
        if target_vde_id is None:
            raise ValueError("update_existing requires target_vde_id or baseline_id")
        row = dict(save_payload.get("update_row") or {})
        update_vde_by_id(int(target_vde_id), row)
        if phase_update_row:
            update_vde_by_id(int(target_vde_id), phase_update_row)
        return {
            "action": "update_existing",
            "vde_id": int(target_vde_id),
            "row": row,
            "phase_update_row": phase_update_row,
        }

    if mode in DELETE_MODES:
        if target_vde_id is None:
            raise ValueError("delete_existing requires target_vde_id or baseline_id")
        deleted = int(delete_vde_by_id(int(target_vde_id)))
        return {
            "action": "delete_existing",
            "vde_id": int(target_vde_id),
            "deleted_rows": deleted,
        }

    if mode in DEACTIVATE_MODES:
        raise ValueError("deactivate_existing is not supported by the current vde_db schema")

    raise ValueError(f"Unsupported save_mode: {save_mode!r}")


__all__ = [
    "build_vde_pre_save_review",
    "build_vde_setup_preview",
    "build_vde_setup_preview_from_ctx",
    "build_vde_workflow_payload_from_ctx",
    "summarize_component_build_up_from_ctx",
    "prepare_vde_setup_preview_for_save",
    "resolve_mass_setup",
    "save_vde_setup_result",
]
