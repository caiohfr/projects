from __future__ import annotations

from copy import deepcopy


DEFAULTS = {
    "roadload_params": {"f0": 100.0, "f1": 0.5, "f2": 0.04, "mass": 1500.0},
    "eta_pt": 0.24,
    "lhv": 32.0,
    "selected_techs": [],
    "cycle_df": None,
}

PWT_SIDEBAR_DEFAULTS = {
    "sb_eta_trans": 0.92,
    "sb_eta_pt": 0.35,
    "sb_fuel_type": "Gasoline",
    "sb_lhv_override": 34.2,
    "sb_uf": 0.50,
    "sb_eta_drive": 0.88,
    "sb_grid": 400.0,
    "pwt_gears": 6,
    "pwt_fdr": 3.91,
}

VDE_SETUP_CTX_DEFAULTS = {
    "legislation": "EPA",
    "category": "",
    "make": "",
    "model": "",
    "year": 2024,
    "notes": "",
    "A": 120.0,
    "B": 0.0,
    "C": 0.012,
    "mass_kg": 1550.0,
    "test_mass_kg": None,
    "test_mass_use_default": True,
    "weight_dist_fr_pct": 50.0,
    "tire_load_mass_basis": "TEST_MASS",
    "cd": 0.30,
    "frontal_area_m2": 2.20,
    "cda": 0.66,
    "crr": 0.010,
    "crr1_frac_at_120kph": 0.010,
    "tire_component_source": "Manual RR",
    "tire_pressure_display_unit": "psi",
    "include_tire_component": False,
    "component_editor_active": "Tires",
    "tire_application_method": "Direct delta RR",
    "tire_current_reference_mode": "Not set",
    "tire_current_reference_manual_basis": "RRC-based reference",
    "tire_walked_reference_mode": "Not set",
    "tire_walked_reference_manual_basis": "RRC-based reference",
    "tire_scenario_application": "Keep inherited",
    "tire_change_method": "Manual tire adjustment",
    "tire_manual_change_intent": "Scenario-only engineering adjustment",
    "tire_manual_adjustment_input_type": "Delta RR",
    "tire_manual_delta_rr_n_per_kn": 0.0,
    "tire_manual_target_rr_n_per_kn": 0.0,
    "tire_delta_calculation_mode": "Manual delta RR",
    "tire_manual_delta_rr_label": "",
    "tire_manual_delta_rr_size_code": "",
    "tire_manual_delta_rr_source": "",
    "tire_manual_delta_rr_notes": "",
    "aero_baseline_reference_cda": 0.0,
    "aero_reference_cda_override": None,
    "aero_candidate_mode": "Not set",
    "aero_calculation_mode": "Inherited",
    "brake_candidate_mode": "Not set",
    "brake_calculation_mode": "Inherited",
    "parasitic_candidate_mode": "Not set",
    "parasitic_calculation_mode": "Inherited",
    "same_tire_front_rear": True,
    "tire_improvement_pct": 0.0,
    "transmission_losses_source": "Missing",
    "trans_A_coef_N": 0.0,
    "trans_B_coef_Npkph": 0.0,
    "trans_C_coef_Npkph2": 0.0,
    "cycle_df": None,
    "cycle_source": "",
    "baseline_id": None,
    "baseline_dict": None,
    "vde_id_parent": None,
    "abc_total_source_ui": "Baseline ABC",
    "vde_setup_view": "Scenario Setup",
    "technical_build_up_view": "Tires",
    "from_delta": "Deltas",
    "mode": "From baseline (editable)",
}

VDE_SETUP_META_KEYS = (
    "legislation",
    "category",
    "make",
    "model",
    "year",
    "notes",
    "cycle_df",
    "cycle_source",
)

VDE_SETUP_VOLATILE_KEYS = (
    "A",
    "B",
    "C",
    "mass_kg",
    "test_mass_kg",
    "test_mass_use_default",
    "delta_rr_N",
    "delta_brake_N",
    "delta_parasitics_N",
    "delta_aero_Npkph2",
    "delta_aero_cdA",
    "tire_size",
    "tire_circ_m",
    "diameter_mm",
    "rrc_N_per_kN",
    "crr1_frac_at_120kph",
    "front_pressure_psi",
    "rear_pressure_psi",
    "rr_load_kpa",
    "smerf",
    "tire_component_source",
    "tire_pressure_display_unit",
    "include_tire_component",
    "component_editor_active",
    "component_mode_tires",
    "tire_application_method",
    "tire_current_reference_mode",
    "tire_current_reference_manual_basis",
    "tire_current_reference_A",
    "tire_current_reference_B",
    "tire_current_reference_C",
    "tire_current_reference_tire_test_code",
    "tire_current_reference_manufacturer",
    "tire_current_reference_model",
    "tire_current_reference_size_code",
    "tire_current_reference_standard_family",
    "tire_current_reference_rr_n_per_kn",
    "tire_current_reference_smerf",
    "tire_current_reference_front_pressure_psi",
    "tire_current_reference_rear_pressure_psi",
    "tire_current_reference_effective_circumference_override_mm",
    "tire_current_reference_test_mileage_km",
    "tire_current_reference_test_method",
    "tire_current_reference_test_source",
    "tire_current_reference_is_tested_value",
    "tire_current_reference_notes",
    "tire_current_preview_result",
    "tire_current_reference_preview_result",
    "tire_current_ref_front_tire_id",
    "tire_current_ref_rear_tire_id",
    "tire_current_ref_front_pressure_psi",
    "tire_current_ref_rear_pressure_psi",
    "tire_current_ref_weight_dist_fr_pct",
    "tire_current_ref_tire_improvement_pct",
    "tire_current_ref_tire_load_mass_basis",
    "tire_walked_reference_mode",
    "tire_walked_reference_manual_basis",
    "tire_walked_reference_A",
    "tire_walked_reference_B",
    "tire_walked_reference_C",
    "tire_walked_reference_tire_test_code",
    "tire_walked_reference_manufacturer",
    "tire_walked_reference_model",
    "tire_walked_reference_size_code",
    "tire_walked_reference_standard_family",
    "tire_walked_reference_rr_n_per_kn",
    "tire_walked_reference_smerf",
    "tire_walked_reference_front_pressure_psi",
    "tire_walked_reference_rear_pressure_psi",
    "tire_walked_reference_effective_circumference_override_mm",
    "tire_walked_reference_test_mileage_km",
    "tire_walked_reference_test_method",
    "tire_walked_reference_test_source",
    "tire_walked_reference_is_tested_value",
    "tire_walked_reference_notes",
    "tire_walked_preview_result",
    "tire_walked_reference_preview_result",
    "tire_walked_ref_front_tire_id",
    "tire_walked_ref_rear_tire_id",
    "tire_walked_ref_front_pressure_psi",
    "tire_walked_ref_rear_pressure_psi",
    "tire_walked_ref_weight_dist_fr_pct",
    "tire_walked_ref_tire_improvement_pct",
    "tire_walked_ref_tire_load_mass_basis",
    "tire_scenario_application",
    "tire_change_method",
    "tire_manual_change_intent",
    "tire_manual_adjustment_input_type",
    "tire_manual_delta_rr_n_per_kn",
    "tire_manual_target_rr_n_per_kn",
    "tire_manual_delta_rr_label",
    "tire_manual_delta_rr_size_code",
    "tire_manual_delta_rr_source",
    "tire_manual_delta_rr_notes",
    "tire_delta_calculation_mode",
    "aero_baseline_reference_cda",
    "aero_reference_cda_override",
    "aero_candidate_mode",
    "aero_candidate_cda",
    "aero_calculation_mode",
    "brake_candidate_mode",
    "brake_candidate_A",
    "brake_candidate_B",
    "brake_candidate_C",
    "brake_calculation_mode",
    "parasitic_candidate_mode",
    "parasitic_candidate_A",
    "parasitic_candidate_B",
    "parasitic_candidate_C",
    "parasitic_calculation_mode",
    "component_mode_aerodynamics",
    "component_mode_brakes",
    "component_mode_parasitics_hubs_axle",
    "same_tire_front_rear",
    "tire_improvement_pct",
    "parasitic_A_coef_N",
    "parasitic_B_Npkph",
    "parasitic_C_coef_Npkph2",
    "brake_A_coef_N",
    "brake_B_Npkph",
    "brake_C_coef_Npkph2",
    "transmission_losses_source",
    "trans_A_coef_N",
    "trans_B_coef_Npkph",
    "trans_C_coef_Npkph2",
    "baseline_id",
    "baseline_dict",
    "selected_baseline_row",
    "vde_id_parent",
    "abc_total_source_ui",
    "technical_build_up_view",
)


def _clone_default(value):
    if isinstance(value, (dict, list, set)):
        return deepcopy(value)
    return value


def ensure_defaults(ss):
    for key, value in DEFAULTS.items():
        if key not in ss:
            ss[key] = _clone_default(value)


def ensure_pwt_sidebar_defaults(ss):
    for key, value in PWT_SIDEBAR_DEFAULTS.items():
        if key not in ss:
            ss[key] = _clone_default(value)


def ensure_vde_setup_state(ss):
    if "ctx" not in ss:
        ss["ctx"] = {}
    ss.setdefault("unit_system", "Metric")
    ctx = ss["ctx"]
    for key, value in VDE_SETUP_CTX_DEFAULTS.items():
        if key not in ctx:
            ctx[key] = _clone_default(value)
    ss.setdefault("_last_mode", ctx["mode"])


def reset_vde_setup_state(ss, preserve_meta: bool = True):
    if "ctx" not in ss:
        ss["ctx"] = {}
    ctx = ss["ctx"]
    meta = {key: ctx.get(key) for key in VDE_SETUP_META_KEYS} if preserve_meta else {}

    for key in VDE_SETUP_VOLATILE_KEYS:
        ctx.pop(key, None)

    if preserve_meta:
        for key, value in meta.items():
            ctx[key] = value

    for key in ("A", "B", "C", "mass_kg", "test_mass_kg", "test_mass_use_default", "from_delta"):
        ctx.setdefault(key, _clone_default(VDE_SETUP_CTX_DEFAULTS[key]))
