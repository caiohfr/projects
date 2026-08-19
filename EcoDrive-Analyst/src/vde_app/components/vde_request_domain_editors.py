from __future__ import annotations

from copy import deepcopy

from src.vde_core.vde_mass_proposal_resolver import resolve_mass_proposal
from src.vde_core.vde_component_modes import canonical_component_mode
from src.vde_core.vde_not_used_modes import EXPLICIT_NOT_USED_PROPOSAL_TYPES, is_not_used_proposal
from src.vde_core.vde_request_contract import is_blank
from src.vde_core.vde_tire_modes import canonical_tire_proposal_type
from src.vde_core.vde_tire_proposal_resolver import resolve_tire_proposal


EPA_INERTIA_CLASSES = [
    454.0, 510.0, 567.0, 624.0, 680.0, 737.0, 794.0, 850.0, 907.0, 964.0, 1021.0, 1077.0,
    1134.0, 1191.0, 1247.0, 1304.0, 1361.0, 1417.0, 1474.0, 1531.0, 1588.0, 1644.0, 1701.0,
    1758.0, 1814.0, 1928.0, 2041.0, 2155.0, 2268.0, 2381.0, 2495.0, 2722.0, 2948.0, 3175.0,
    3402.0, 3856.0, 4082.0,
]
TWC_SHIFT_OPTIONS = ["-3", "-2", "-1", "+1", "+2", "+3", "Select target"]
CURB_POSITION_OPTIONS = ["Top", "Mid", "Bottom"]
TRANSMISSION_APPLICATION_MODE_OPTIONS = ["APPLY_DELTA_TO_TOTAL", "KEEP_TOTAL_FIXED"]
NOT_USED_PROPOSAL_TYPES = set(EXPLICIT_NOT_USED_PROPOSAL_TYPES)


FIELD_META = {
    "proposal_type": {"label": "Proposal type", "unit": "-", "kind": "text", "widget": "readonly"},
    "mass_kg": {"label": "Curb mass", "unit": "kg", "kind": "number", "widget": "number", "step": 1.0, "format": "%.1f"},
    "current_curb_mass_kg": {"label": "Current / inherited curb mass", "unit": "kg", "kind": "number", "widget": "readonly"},
    "target_curb_mass_kg": {"label": "Curb mass", "unit": "kg", "kind": "number", "widget": "readonly"},
    "test_mass_kg": {"label": "VDE mass", "unit": "kg", "kind": "number", "widget": "number", "step": 1.0, "format": "%.1f"},
    "vde_calculation_mass_kg": {"label": "VDE mass", "unit": "kg", "kind": "number", "widget": "readonly"},
    "vde_mass_basis": {"label": "VDE mass basis", "unit": "-", "kind": "text", "widget": "readonly"},
    "tire_load_mass_used_kg": {"label": "Tire calculation mass", "unit": "kg", "kind": "number", "widget": "readonly"},
    "weight_dist_fr_pct": {"label": "Front weight distribution", "unit": "%", "kind": "number", "widget": "number", "step": 0.1, "format": "%.1f", "min": 0.0, "max": 100.0},
    "test_mass_low_kg": {"label": "TML", "unit": "kg", "kind": "number", "widget": "readonly"},
    "test_mass_high_kg": {"label": "TMH", "unit": "kg", "kind": "number", "widget": "readonly"},
    "test_mass_basis": {"label": "Test mass basis", "unit": "-", "kind": "text", "widget": "readonly"},
    "tire_load_mass_basis": {"label": "Tire calculation mass basis", "unit": "-", "kind": "select", "widget": "select", "options": ["TWC", "TEST_MASS"]},
    "inertia_class": {"label": "EPA ETW / TWC", "unit": "kg", "kind": "number", "widget": "readonly"},
    "target_twc_interval": {"label": "Target TWC interval", "unit": "-", "kind": "text", "widget": "readonly"},
    "shift_steps": {"label": "TWC change", "unit": "-", "kind": "select", "widget": "select", "options": TWC_SHIFT_OPTIONS},
    "curb_position": {"label": "Curb position", "unit": "-", "kind": "select", "widget": "select", "options": CURB_POSITION_OPTIONS},
    "target_mass_kg": {"label": "Target ETW / TWC", "unit": "kg", "kind": "number", "widget": "select", "options": EPA_INERTIA_CLASSES},
    "line_type": {"label": "Line type", "unit": "-", "kind": "select", "widget": "select", "options": ["TML", "TMH"]},
    "preset": {"label": "Preset", "unit": "-", "kind": "select", "widget": "select", "options": ["Curb +100 kg", "Curb +300 lb", "Custom delta"]},
    "custom_delta_kg": {"label": "Custom delta", "unit": "kg", "kind": "number", "widget": "number", "step": 1.0, "format": "%.1f"},
    "options_kg": {"label": "Options mass", "unit": "kg", "kind": "number", "widget": "number", "step": 1.0, "format": "%.1f"},
    "payload_kg": {"label": "Payload", "unit": "kg", "kind": "number", "widget": "number", "step": 1.0, "format": "%.1f"},
    "gvwr_kg": {"label": "GVWR", "unit": "kg", "kind": "number", "widget": "number", "step": 1.0, "format": "%.1f"},
    "gcwr_kg": {"label": "GCWR", "unit": "kg", "kind": "number", "widget": "number", "step": 1.0, "format": "%.1f"},
    "trailer_mass_kg": {"label": "Trailer mass", "unit": "kg", "kind": "number", "widget": "number", "step": 1.0, "format": "%.1f"},
    "vehicle_mass_at_gcwr": {"label": "Vehicle mass at GCWR", "unit": "kg", "kind": "text", "widget": "readonly"},
    "trailer_code": {"label": "Trailer code", "unit": "-", "kind": "text", "widget": "readonly"},
    "trailer_A": {"label": "Trailer A", "unit": "N", "kind": "number", "widget": "number", "step": 0.1, "format": "%.3f"},
    "trailer_B": {"label": "Trailer B", "unit": "N/kph", "kind": "number", "widget": "number", "step": 0.0001, "format": "%.5f"},
    "trailer_C": {"label": "Trailer C", "unit": "N/kph2", "kind": "number", "widget": "number", "step": 0.000001, "format": "%.6f"},
    "trailer_roadload_status": {"label": "Trailer roadload status", "unit": "-", "kind": "text", "widget": "readonly"},
    "mass_rule_status": {"label": "Mass status", "unit": "-", "kind": "text", "widget": "readonly"},
    "mass_rule_notes": {"label": "Mass notes", "unit": "-", "kind": "text", "widget": "readonly"},
    "cda_m2": {"label": "CdA", "unit": "m2", "kind": "number", "widget": "number", "step": 0.0001, "format": "%.4f"},
    "delta_CdA": {"label": "Delta CdA", "unit": "m2", "kind": "number", "widget": "number", "step": 0.0001, "format": "%.4f"},
    "new_CdA": {"label": "New CdA", "unit": "m2", "kind": "text", "widget": "readonly"},
    "baseline_CdA": {"label": "Baseline CdA", "unit": "m2", "kind": "text", "widget": "readonly"},
    "aero_source_vde_id": {"label": "Aero VDE ID", "unit": "-", "kind": "text", "widget": "readonly"},
    "frontal_area_m2": {"label": "Frontal area", "unit": "m2", "kind": "number", "widget": "number", "step": 0.01, "format": "%.2f"},
    "Cd_display": {"label": "Cd display", "unit": "-", "kind": "text", "widget": "readonly"},
    "tire_db_id": {"label": "Tire ID", "unit": "-", "kind": "text", "widget": "lookup"},
    "tire_code": {"label": "Tire code", "unit": "-", "kind": "text", "widget": "readonly"},
    "tire_source_vde_id": {"label": "Tire VDE ID", "unit": "-", "kind": "text", "widget": "readonly"},
    "rrc_N_per_kN": {"label": "Final RRC", "unit": "N/kN", "kind": "number", "widget": "readonly", "step": 0.1, "format": "%.2f"},
    "target_rrc_N_per_kN": {"label": "Target final RRC", "unit": "N/kN", "kind": "number", "widget": "number", "step": 0.1, "format": "%.2f"},
    "front_pressure_psi": {"label": "Front pressure", "unit": "psi", "kind": "number", "widget": "number", "step": 0.5, "format": "%.1f"},
    "rear_pressure_psi": {"label": "Rear pressure", "unit": "psi", "kind": "number", "widget": "number", "step": 0.5, "format": "%.1f"},
    "tire_improvement_pct": {"label": "Tire improvement", "unit": "%", "kind": "number", "widget": "number", "step": 0.1, "format": "%.1f"},
    "tire_review_status": {"label": "Tire status", "unit": "-", "kind": "text", "widget": "readonly"},
    "change_mode": {"label": "Change mode", "unit": "-", "kind": "select", "widget": "select", "options": ["Absolute ABC", "Delta ABC"]},
    "transmission_component_db_id": {"label": "Transmission ID", "unit": "-", "kind": "text", "widget": "lookup"},
    "transmission_vde_db_id": {"label": "Transmission VDE ID", "unit": "-", "kind": "text", "widget": "readonly"},
    "trans_A_coef_N": {"label": "A", "unit": "N", "kind": "number", "widget": "number", "step": 0.1, "format": "%.3f"},
    "trans_B_coef_Npkph": {"label": "B", "unit": "N/kph", "kind": "number", "widget": "number", "step": 0.0001, "format": "%.5f"},
    "trans_C_coef_Npkph2": {"label": "C", "unit": "N/kph2", "kind": "number", "widget": "number", "step": 0.000001, "format": "%.6f"},
    "transmission_application_mode": {
        "label": "Transmission application mode",
        "unit": "-",
        "kind": "select",
        "widget": "select",
        "options": TRANSMISSION_APPLICATION_MODE_OPTIONS,
    },
    "delta_A": {"label": "Delta A", "unit": "N", "kind": "number", "widget": "number", "step": 0.1, "format": "%.3f"},
    "delta_B": {"label": "Delta B", "unit": "N/kph", "kind": "number", "widget": "number", "step": 0.0001, "format": "%.5f"},
    "delta_C": {"label": "Delta C", "unit": "N/kph2", "kind": "number", "widget": "number", "step": 0.000001, "format": "%.6f"},
    "new_trans_A": {"label": "New A", "unit": "N", "kind": "text", "widget": "readonly"},
    "new_trans_B": {"label": "New B", "unit": "N/kph", "kind": "text", "widget": "readonly"},
    "new_trans_C": {"label": "New C", "unit": "N/kph2", "kind": "text", "widget": "readonly"},
    "transmission_loss_pct": {"label": "Transmission coastdown share", "unit": "%", "kind": "number", "widget": "number", "step": 0.1, "format": "%.1f", "min": 0.0, "max": 100.0},
    "percent_basis": {"label": "Percent basis", "unit": "-", "kind": "text", "widget": "text"},
    "rule_version": {"label": "Rule version", "unit": "-", "kind": "text", "widget": "text"},
    "brake_component_db_id": {"label": "Brake ID", "unit": "-", "kind": "text", "widget": "lookup"},
    "brake_vde_db_id": {"label": "Brake VDE ID", "unit": "-", "kind": "text", "widget": "readonly"},
    "brake_A_coef_N": {"label": "A", "unit": "N", "kind": "number", "widget": "number", "step": 0.1, "format": "%.3f"},
    "brake_B_Npkph": {"label": "B", "unit": "N/kph", "kind": "number", "widget": "number", "step": 0.0001, "format": "%.5f"},
    "brake_C_coef_Npkph2": {"label": "C", "unit": "N/kph2", "kind": "number", "widget": "number", "step": 0.000001, "format": "%.6f"},
    "method": {"label": "Method", "unit": "-", "kind": "select", "widget": "select", "options": ["Brake ABC", "Residual torque"]},
    "residual_torque_front_Nm": {"label": "Front torque", "unit": "Nm", "kind": "number", "widget": "number", "step": 0.1, "format": "%.2f"},
    "residual_torque_rear_Nm": {"label": "Rear torque", "unit": "Nm", "kind": "number", "widget": "number", "step": 0.1, "format": "%.2f"},
    "residual_torque_total_Nm": {"label": "Total torque", "unit": "Nm", "kind": "number", "widget": "number", "step": 0.1, "format": "%.2f"},
    "wheel_radius_m": {"label": "Wheel radius", "unit": "m", "kind": "number", "widget": "number", "step": 0.001, "format": "%.3f"},
    "brake_drag_force_N": {"label": "Resolved drag force", "unit": "N", "kind": "text", "widget": "readonly"},
    "axle_hubs_component_db_id": {"label": "Axle/Hubs ID", "unit": "-", "kind": "text", "widget": "lookup"},
    "axle_hubs_vde_db_id": {"label": "Axle/Hubs VDE ID", "unit": "-", "kind": "text", "widget": "readonly"},
    "axle_hub_A": {"label": "A", "unit": "N", "kind": "number", "widget": "number", "step": 0.1, "format": "%.3f"},
    "axle_hub_B": {"label": "B", "unit": "N/kph", "kind": "number", "widget": "number", "step": 0.0001, "format": "%.5f"},
    "axle_hub_C": {"label": "C", "unit": "N/kph2", "kind": "number", "widget": "number", "step": 0.000001, "format": "%.6f"},
    "parasitic_component_db_id": {"label": "Parasitic ID", "unit": "-", "kind": "text", "widget": "lookup"},
    "parasitic_vde_db_id": {"label": "Parasitic VDE ID", "unit": "-", "kind": "text", "widget": "readonly"},
    "parasitic_A_coef_N": {"label": "A", "unit": "N", "kind": "number", "widget": "number", "step": 0.1, "format": "%.3f"},
    "parasitic_B_Npkph": {"label": "B", "unit": "N/kph", "kind": "number", "widget": "number", "step": 0.0001, "format": "%.5f"},
    "parasitic_C_coef_Npkph2": {"label": "C", "unit": "N/kph2", "kind": "number", "widget": "number", "step": 0.000001, "format": "%.6f"},
}

PROPOSAL_FIELDS = {
    ("mass", "EPA_STATUS"): ["inertia_class", "test_mass_kg", "weight_dist_fr_pct", "tire_load_mass_basis"],
    ("mass", "EPA_CURB_TO_TWC"): ["current_curb_mass_kg", "mass_kg", "inertia_class", "target_twc_interval", "test_mass_kg", "test_mass_basis", "weight_dist_fr_pct", "tire_load_mass_basis", "mass_rule_status", "mass_rule_notes"],
    ("mass", "MASS_TWC_SHIFT"): ["shift_steps", "target_mass_kg", "curb_position", "mass_kg", "inertia_class", "target_twc_interval", "test_mass_kg", "test_mass_basis", "weight_dist_fr_pct", "tire_load_mass_basis", "mass_rule_status", "mass_rule_notes"],
    ("mass", "PERFORMANCE_CURB_MASS"): ["mass_kg", "preset", "custom_delta_kg", "test_mass_kg", "test_mass_basis", "weight_dist_fr_pct", "tire_load_mass_basis", "mass_rule_status", "mass_rule_notes"],
    ("mass", "WLTP_MASS_LINE"): ["line_type", "mass_kg", "payload_kg", "options_kg", "test_mass_low_kg", "test_mass_high_kg", "test_mass_kg", "weight_dist_fr_pct", "tire_load_mass_basis"],
    ("mass", "GVWR"): ["mass_kg", "payload_kg", "gvwr_kg", "vde_calculation_mass_kg", "vde_mass_basis", "tire_load_mass_used_kg", "tire_load_mass_basis", "weight_dist_fr_pct"],
    ("mass", "GCWR"): ["mass_kg", "gcwr_kg", "trailer_mass_kg", "vehicle_mass_at_gcwr", "payload_kg", "trailer_A", "trailer_B", "trailer_C", "vde_calculation_mass_kg", "vde_mass_basis", "tire_load_mass_used_kg", "tire_load_mass_basis", "weight_dist_fr_pct"],
    ("mass", "CUSTOM_MASS"): ["test_mass_kg", "test_mass_basis", "weight_dist_fr_pct", "tire_load_mass_basis"],
    ("aero", "AERO_ABSOLUTE_CDA"): ["baseline_CdA", "cda_m2", "delta_CdA"],
    ("aero", "AERO_DELTA_CDA"): ["baseline_CdA", "delta_CdA", "new_CdA"],
    ("aero", "AERO_NOT_USED"): ["baseline_CdA"],
    ("tire", "TIRE_DB_LOOKUP"): ["tire_db_id", "tire_source_vde_id", "tire_code", "rrc_N_per_kN", "front_pressure_psi", "rear_pressure_psi", "tire_load_mass_used_kg", "tire_load_mass_basis", "tire_review_status"],
    ("tire", "TIRE_TARGET_RRC"): ["target_rrc_N_per_kN", "front_pressure_psi", "rear_pressure_psi", "rrc_N_per_kN", "tire_load_mass_used_kg", "tire_load_mass_basis", "tire_review_status"],
    ("tire", "TIRE_IMPROVEMENT_PCT"): ["tire_improvement_pct", "rrc_N_per_kN", "front_pressure_psi", "rear_pressure_psi", "tire_load_mass_used_kg", "tire_load_mass_basis", "tire_review_status"],
    ("tire", "TIRE_SMERF_RRC_CHANGE"): ["target_rrc_N_per_kN", "front_pressure_psi", "rear_pressure_psi", "rrc_N_per_kN", "tire_load_mass_used_kg", "tire_load_mass_basis", "tire_review_status"],
    ("tire", "TIRE_METADATA_ONLY"): ["tire_db_id", "tire_source_vde_id", "tire_code", "rrc_N_per_kN", "front_pressure_psi", "rear_pressure_psi"],
    ("transmission", "TRANS_METADATA_ONLY"): ["transmission_application_mode", "transmission_component_db_id", "transmission_vde_db_id", "trans_A_coef_N", "trans_B_coef_Npkph", "trans_C_coef_Npkph2"],
    ("transmission", "UPDATE_TRANS_DRAG_ABC"): ["transmission_application_mode", "trans_A_coef_N", "trans_B_coef_Npkph", "trans_C_coef_Npkph2", "delta_A", "delta_B", "delta_C", "new_trans_A", "new_trans_B", "new_trans_C"],
    ("transmission", "TRANS_LOSS_PCT"): ["transmission_loss_pct", "trans_A_coef_N", "trans_B_coef_Npkph", "trans_C_coef_Npkph2"],
    ("transmission", "TRANS_LOSS_NOT_AVAILABLE"): ["transmission_application_mode", "trans_A_coef_N", "trans_B_coef_Npkph", "trans_C_coef_Npkph2"],
    ("brake", "BRAKE_METADATA_ONLY"): ["brake_component_db_id", "brake_vde_db_id", "brake_A_coef_N", "brake_B_Npkph", "brake_C_coef_Npkph2"],
    ("brake", "BRAKE_DRAG_CHANGE"): ["brake_A_coef_N", "brake_B_Npkph", "brake_C_coef_Npkph2", "delta_A", "delta_B", "delta_C", "residual_torque_front_Nm", "residual_torque_rear_Nm", "residual_torque_total_Nm", "wheel_radius_m", "brake_drag_force_N"],
    ("brake", "BRAKE_NOT_USED"): ["brake_A_coef_N", "brake_B_Npkph", "brake_C_coef_Npkph2"],
    ("axle_hubs", "AXLE_HUB_METADATA_ONLY"): ["axle_hubs_component_db_id", "axle_hubs_vde_db_id", "axle_hub_A", "axle_hub_B", "axle_hub_C"],
    ("axle_hubs", "AXLE_HUB_DRAG_CHANGE"): ["axle_hub_A", "axle_hub_B", "axle_hub_C", "delta_A", "delta_B", "delta_C"],
    ("axle_hubs", "AXLE_HUB_NOT_USED"): ["axle_hub_A", "axle_hub_B", "axle_hub_C"],
    ("parasitic", "PARASITIC_METADATA_ONLY"): ["parasitic_component_db_id", "parasitic_vde_db_id", "parasitic_A_coef_N", "parasitic_B_Npkph", "parasitic_C_coef_Npkph2"],
    ("parasitic", "PARASITIC_LOSS_CHANGE"): ["parasitic_A_coef_N", "parasitic_B_Npkph", "parasitic_C_coef_Npkph2", "delta_A", "delta_B", "delta_C"],
    ("parasitic", "PARASITIC_NOT_USED"): ["parasitic_A_coef_N", "parasitic_B_Npkph", "parasitic_C_coef_Npkph2"],
}

CALCULATED_FIELDS = {
    "test_mass_basis",
    "mass_rule_status",
    "mass_rule_notes",
    "vehicle_mass_at_gcwr",
    "trailer_roadload_status",
    "test_mass_low_kg",
    "test_mass_high_kg",
    "delta_CdA",
    "new_CdA",
    "Cd_display",
    "new_trans_A",
    "new_trans_B",
    "new_trans_C",
    "brake_drag_force_N",
    "tire_review_status",
    "tire_rule_status",
    "tire_rule_notes",
    "tire_adjustment_method",
    "tire_delta_rrc_N_per_kN",
}

READONLY_FIELDS = {
    "aero_source_vde_id",
    "tire_source_vde_id",
    "transmission_vde_db_id",
    "brake_vde_db_id",
    "axle_hubs_vde_db_id",
    "parasitic_vde_db_id",
}

DELTA_ZERO_FIELDS = {
    "delta_A",
    "delta_B",
    "delta_C",
}

PROPOSAL_READONLY_FIELDS = {
    ("mass", "EPA_STATUS"): {"inertia_class", "test_mass_kg"},
    ("mass", "EPA_CURB_TO_TWC"): {"current_curb_mass_kg", "inertia_class", "target_twc_interval", "test_mass_kg", "test_mass_basis", "mass_rule_status", "mass_rule_notes"},
    ("mass", "MASS_TWC_SHIFT"): {"mass_kg", "inertia_class", "test_mass_kg", "test_mass_basis", "target_twc_interval", "mass_rule_status", "mass_rule_notes"},
    ("mass", "PERFORMANCE_CURB_MASS"): {"test_mass_kg"},
    ("mass", "GVWR"): {"mass_kg", "payload_kg", "test_mass_kg"},
    ("mass", "WLTP_MASS_LINE"): {"test_mass_kg", "test_mass_low_kg", "test_mass_high_kg"},
    ("mass", "GCWR"): {"vehicle_mass_at_gcwr", "trailer_code", "test_mass_kg"},
}


def field_meta(field_key: str) -> dict:
    return deepcopy(FIELD_META.get(field_key, {"label": field_key, "unit": "", "kind": "text"}))


def proposal_fields(domain: str, proposal_type: str) -> list[str]:
    if str(domain or "").strip().lower() == "tire":
        proposal_type = canonical_tire_proposal_type(proposal_type)
    return list(PROPOSAL_FIELDS.get((domain, proposal_type), []))


def applicable_fields(domain: str, proposal_type: str, selection_mode: str | None = None) -> list[str]:
    if proposal_is_not_used(proposal_type, selection_mode, domain=domain):
        return []
    fields = list(proposal_fields(domain, proposal_type))
    mode = str(selection_mode or proposal_type or "").strip().lower()
    if domain == "tire":
        return fields
    if domain == "transmission" and proposal_type == "UPDATE_TRANS_DRAG_ABC":
        if mode == "absolute abc":
            return [field for field in fields if field not in {"delta_A", "delta_B", "delta_C"}]
        return [field for field in fields if field not in {"trans_A_coef_N", "trans_B_coef_Npkph", "trans_C_coef_Npkph2"}]
    if domain == "brake" and proposal_type == "BRAKE_DRAG_CHANGE":
        component_mode = canonical_component_mode(domain, proposal_type, selection_mode)
        if component_mode == "RESIDUAL_TORQUE":
            return [field for field in fields if field in {"residual_torque_front_Nm", "residual_torque_rear_Nm", "residual_torque_total_Nm", "wheel_radius_m", "brake_drag_force_N"}]
        if component_mode == "ABSOLUTE_ABC":
            return [field for field in fields if field not in {"delta_A", "delta_B", "delta_C", "residual_torque_front_Nm", "residual_torque_rear_Nm", "residual_torque_total_Nm", "wheel_radius_m", "brake_drag_force_N"}]
        return [field for field in fields if field not in {"brake_A_coef_N", "brake_B_Npkph", "brake_C_coef_Npkph2", "residual_torque_front_Nm", "residual_torque_rear_Nm", "residual_torque_total_Nm", "wheel_radius_m", "brake_drag_force_N"}]
    if domain == "axle_hubs" and proposal_type == "AXLE_HUB_DRAG_CHANGE":
        if mode == "absolute abc":
            return [field for field in fields if field not in {"delta_A", "delta_B", "delta_C"}]
        return [field for field in fields if field not in {"axle_hub_A", "axle_hub_B", "axle_hub_C"}]
    if domain == "parasitic" and proposal_type == "PARASITIC_LOSS_CHANGE":
        if mode == "absolute abc":
            return [field for field in fields if field not in {"delta_A", "delta_B", "delta_C"}]
        return [field for field in fields if field not in {"parasitic_A_coef_N", "parasitic_B_Npkph", "parasitic_C_coef_Npkph2"}]
    return fields


def sanitize_domain_inputs(domain: str, proposal_type: str, selection_mode: str, inputs: dict | None) -> dict:
    if str(domain or "").strip().lower() == "tire":
        proposal_type = canonical_tire_proposal_type(proposal_type)
    allowed = set(applicable_fields(domain, proposal_type, selection_mode))
    cleaned = {}
    for key, value in dict(inputs or {}).items():
        if key not in allowed:
            continue
        if key in DELTA_ZERO_FIELDS and is_blank(value):
            cleaned[key] = 0.0
            continue
        if is_blank(value):
            continue
        cleaned[key] = value
    if domain == "tire" and proposal_type in {"TIRE_DB_LOOKUP", "TIRE_METADATA_ONLY"}:
        tire_snapshot = dict(inputs or {}).get("tire_snapshot")
        if isinstance(tire_snapshot, dict) and tire_snapshot:
            cleaned["tire_snapshot"] = deepcopy(tire_snapshot)
    if domain == "mass" and proposal_type == "MASS_TWC_SHIFT":
        cleaned = _normalize_mass_twc_shift_inputs(cleaned, dict(inputs or {}))
    if domain == "mass" and proposal_type == "PERFORMANCE_CURB_MASS":
        cleaned.setdefault("preset", str(dict(inputs or {}).get("preset") or "Curb +100 kg"))
    if domain == "mass" and proposal_type == "EPA_CURB_TO_TWC":
        legacy_target = dict(inputs or {}).get("target_curb_mass_kg")
        if is_blank(cleaned.get("mass_kg")) and not is_blank(legacy_target):
            cleaned["mass_kg"] = legacy_target
    if domain == "mass" and proposal_type == "EPA_STATUS":
        cleaned.pop("target_mass_kg", None)
        cleaned.pop("mass_kg", None)
    if domain == "transmission" and proposal_type not in {"", "INHERIT"}:
        if proposal_type == "TRANS_LOSS_PCT":
            cleaned["transmission_application_mode"] = "KEEP_TOTAL_FIXED"
            cleaned["percent_basis"] = "SOURCE_ABC_TOTAL"
            cleaned["rule_version"] = "COASTDOWN_SHARE_V1"
        else:
            cleaned["transmission_application_mode"] = _normalize_transmission_application_mode(
                cleaned.get("transmission_application_mode") or dict(inputs or {}).get("transmission_application_mode")
            )
    return cleaned


def proposal_is_not_used(proposal_type: str, selection_mode: str | None = None, *, domain: str | None = None) -> bool:
    return is_not_used_proposal(domain, proposal_type, selection_mode)


def required_fields(domain: str, proposal_type: str, selection_mode: str | None = None, inputs: dict | None = None) -> list[str]:
    if str(domain or "").strip().lower() == "tire":
        proposal_type = canonical_tire_proposal_type(proposal_type)
    mode = str(selection_mode or proposal_type or "").strip().lower()
    cleaned_inputs = dict(inputs or {})
    mapping = {
        ("mass", "EPA_CURB_TO_TWC"): ["mass_kg"],
        ("mass", "PERFORMANCE_CURB_MASS"): ["mass_kg"],
        ("mass", "GVWR"): ["mass_kg", "payload_kg"],
        ("mass", "WLTP_MASS_LINE"): ["line_type", "mass_kg", "payload_kg"],
        ("mass", "GCWR"): ["mass_kg", "gcwr_kg", "trailer_mass_kg", "trailer_A", "trailer_B", "trailer_C"],
        ("mass", "CUSTOM_MASS"): ["test_mass_kg"],
        ("aero", "AERO_ABSOLUTE_CDA"): ["cda_m2"],
        ("aero", "AERO_DELTA_CDA"): ["delta_CdA"],
        ("tire", "TIRE_DB_LOOKUP"): ["tire_db_id"],
        ("tire", "TIRE_IMPROVEMENT_PCT"): ["tire_improvement_pct"],
        ("tire", "TIRE_TARGET_RRC"): [],
        ("transmission", "TRANS_METADATA_ONLY"): ["transmission_component_db_id", "trans_A_coef_N", "trans_B_coef_Npkph", "trans_C_coef_Npkph2"],
        ("brake", "BRAKE_METADATA_ONLY"): ["brake_component_db_id", "brake_A_coef_N", "brake_B_Npkph", "brake_C_coef_Npkph2"],
        ("axle_hubs", "AXLE_HUB_METADATA_ONLY"): ["axle_hubs_component_db_id", "axle_hub_A", "axle_hub_B", "axle_hub_C"],
        ("parasitic", "PARASITIC_METADATA_ONLY"): ["parasitic_component_db_id", "parasitic_A_coef_N", "parasitic_B_Npkph", "parasitic_C_coef_Npkph2"],
        ("transmission", "TRANS_LOSS_PCT"): ["transmission_loss_pct"],
    }
    required = list(mapping.get((domain, proposal_type), []))
    vde_identity_fields = {
        "transmission": "transmission_vde_db_id",
        "brake": "brake_vde_db_id",
        "axle_hubs": "axle_hubs_vde_db_id",
        "parasitic": "parasitic_vde_db_id",
    }
    if proposal_type.endswith("_METADATA_ONLY") and not is_blank(cleaned_inputs.get(vde_identity_fields.get(domain, ""))):
        required = [field_key for field_key in required if not field_key.endswith("component_db_id")]
    if domain == "mass" and proposal_type == "PERFORMANCE_CURB_MASS" and str(cleaned_inputs.get("preset") or "Curb +100 kg").strip() == "Custom delta":
        required.append("custom_delta_kg")
    if domain == "mass" and proposal_type == "MASS_TWC_SHIFT":
        if _is_select_target_mode(cleaned_inputs.get("shift_steps")):
            return ["target_mass_kg"]
        return ["shift_steps"]
    if domain in {"transmission", "axle_hubs", "parasitic"} and "delta abc" in mode:
        return []
    if domain == "brake" and proposal_type == "BRAKE_DRAG_CHANGE":
        component_mode = canonical_component_mode(domain, proposal_type, selection_mode, cleaned_inputs)
        if component_mode == "RESIDUAL_TORQUE":
            required_torque = []
            if is_blank(cleaned_inputs.get("residual_torque_total_Nm")) and is_blank(cleaned_inputs.get("residual_torque_front_Nm")) and is_blank(cleaned_inputs.get("residual_torque_rear_Nm")):
                required_torque.append("residual_torque_total_Nm")
            return [*required_torque, "wheel_radius_m"]
        if component_mode == "ABSOLUTE_ABC":
            return ["brake_A_coef_N", "brake_B_Npkph", "brake_C_coef_Npkph2"]
        return []
    if domain == "transmission" and proposal_type == "UPDATE_TRANS_DRAG_ABC" and mode == "absolute abc":
        return ["trans_A_coef_N", "trans_B_coef_Npkph", "trans_C_coef_Npkph2"]
    if domain == "axle_hubs" and proposal_type == "AXLE_HUB_DRAG_CHANGE" and mode == "absolute abc":
        return ["axle_hub_A", "axle_hub_B", "axle_hub_C"]
    if domain == "parasitic" and proposal_type == "PARASITIC_LOSS_CHANGE" and mode == "absolute abc":
        return ["parasitic_A_coef_N", "parasitic_B_Npkph", "parasitic_C_coef_Npkph2"]
    return required


def field_schema(
    domain: str,
    proposal_type: str,
    selection_mode: str,
    field_key: str,
    *,
    inputs: dict | None = None,
) -> dict:
    schema = field_meta(field_key)
    schema.setdefault("widget", schema.get("kind", "text"))
    schema["required"] = field_key in required_fields(domain, proposal_type, selection_mode, inputs)
    readonly = field_key in READONLY_FIELDS or field_key in CALCULATED_FIELDS or field_key in PROPOSAL_READONLY_FIELDS.get((domain, proposal_type), set())
    if proposal_is_not_used(proposal_type, selection_mode, domain=domain) or proposal_type == "INHERIT":
        readonly = True
    if domain == "mass" and proposal_type == "CUSTOM_MASS" and field_key == "test_mass_basis":
        readonly = False
        schema["widget"] = "text"
        schema["required"] = False
    if domain == "mass" and proposal_type == "PERFORMANCE_CURB_MASS" and field_key == "preset":
        schema["widget"] = "select"
    if domain == "mass" and proposal_type == "PERFORMANCE_CURB_MASS" and field_key == "custom_delta_kg":
        schema["widget"] = "number" if str(dict(inputs or {}).get("preset") or "Curb +100 kg").strip() == "Custom delta" else "readonly"
    if domain == "mass" and proposal_type == "MASS_TWC_SHIFT" and field_key == "shift_steps":
        schema["widget"] = "select"
        schema["options"] = list(TWC_SHIFT_OPTIONS)
    if domain == "mass" and proposal_type == "MASS_TWC_SHIFT" and field_key == "curb_position":
        schema["widget"] = "select"
        schema["options"] = list(CURB_POSITION_OPTIONS)
    if domain == "mass" and proposal_type == "MASS_TWC_SHIFT" and field_key == "target_mass_kg":
        schema["widget"] = "select" if _is_select_target_mode(dict(inputs or {}).get("shift_steps")) else "readonly"
        schema["options"] = list(EPA_INERTIA_CLASSES)
    if domain == "tire" and field_key == "rrc_N_per_kN":
        schema["widget"] = "readonly"
    if schema.get("widget") == "lookup":
        readonly = True
    if readonly:
        schema["widget"] = "readonly"
    return schema


def proposal_application_status(
    domain: str,
    proposal_type: str,
    selection_mode: str,
    inputs: dict | None,
    resolved_display: dict | None = None,
) -> dict:
    cleaned = sanitize_domain_inputs(domain, proposal_type, selection_mode, inputs)
    resolved = dict(resolved_display or {})
    if proposal_type == "INHERIT":
        return {"status": "inherited", "message": "Inherited", "missing_fields": [], "issues": []}
    if proposal_is_not_used(proposal_type, selection_mode, domain=domain):
        return {"status": "not_used", "message": "Not used", "missing_fields": [], "issues": []}

    missing = [
        field_key
        for field_key in required_fields(domain, proposal_type, selection_mode, cleaned)
        if is_blank(cleaned.get(field_key))
    ]
    if domain == "transmission" and proposal_type == "UPDATE_TRANS_DRAG_ABC" and str(selection_mode or "").strip().lower() == "delta abc":
        if not any(not is_blank(cleaned.get(field_key)) and float(cleaned.get(field_key)) != 0.0 for field_key in ("delta_A", "delta_B", "delta_C")):
            missing.append("delta_A")
    if domain in {"axle_hubs", "parasitic"} and "delta abc" in str(selection_mode or "").strip().lower():
        if not any(not is_blank(cleaned.get(field_key)) and float(cleaned.get(field_key)) != 0.0 for field_key in ("delta_A", "delta_B", "delta_C")):
            missing.append("delta_A")
    if domain == "brake" and proposal_type == "BRAKE_DRAG_CHANGE" and str(selection_mode or "").strip().lower() == "delta abc":
        if not any(not is_blank(cleaned.get(field_key)) and float(cleaned.get(field_key)) != 0.0 for field_key in ("delta_A", "delta_B", "delta_C")):
            missing.append("delta_A")

    friendly_missing = [friendly_field_requirement(domain, proposal_type, selection_mode, field_key) for field_key in list(dict.fromkeys(missing))]
    if missing:
        return {
            "status": "applied_incomplete",
            "message": f"Applied — {friendly_missing[0]}",
            "missing_fields": list(dict.fromkeys(missing)),
            "issues": friendly_missing,
        }

    if str(resolved.get("mass_rule_status") or "").strip().lower() in {"missing", "invalid"}:
        note = friendly_message(str(resolved.get("mass_rule_notes") or "")) or "Mass inputs are incomplete."
        return {"status": "applied_incomplete", "message": f"Applied — {note}", "missing_fields": [], "issues": [note]}

    if domain == "tire" and _has_only_nonblocking_tire_warnings(resolved):
        return {
            "status": "applied_ready",
            "message": "Applied — Ready",
            "missing_fields": [],
            "issues": _friendly_tire_issue_messages(resolved),
        }
    if str(resolved.get("tire_rule_status") or resolved.get("tire_review_status") or "").strip().lower() == "review":
        note = friendly_message(str(resolved.get("tire_rule_notes") or "Tire calculation requires review."))
        return {"status": "applied_incomplete", "message": f"Applied — {note}", "missing_fields": [], "issues": [note]}
    if str(resolved.get("tire_rule_status") or resolved.get("tire_review_status") or "").strip().lower() == "missing":
        note = friendly_message(str(resolved.get("tire_rule_notes") or "Tire inputs are incomplete."))
        return {"status": "applied_incomplete", "message": f"Applied — {note}", "missing_fields": [], "issues": [note]}

    return {"status": "applied_ready", "message": "Applied — Ready", "missing_fields": [], "issues": []}


def friendly_field_requirement(domain: str, proposal_type: str, selection_mode: str, field_key: str) -> str:
    special = {
        ("mass", "EPA_CURB_TO_TWC", "mass_kg"): "Curb mass is required.",
        ("mass", "PERFORMANCE_CURB_MASS", "mass_kg"): "New curb mass is required.",
        ("mass", "PERFORMANCE_CURB_MASS", "custom_delta_kg"): "Custom delta is required.",
        ("mass", "GVWR", "gvwr_kg"): "GVWR is required.",
        ("mass", "CUSTOM_MASS", "test_mass_kg"): "Custom test mass is required.",
        ("mass", "MASS_TWC_SHIFT", "shift_steps"): "TWC change is required.",
        ("mass", "MASS_TWC_SHIFT", "target_mass_kg"): "Target ETW / TWC is required.",
        ("mass", "GCWR", "gcwr_kg"): "GCWR is required.",
        ("mass", "GCWR", "trailer_mass_kg"): "Trailer mass is required.",
        ("aero", "AERO_ABSOLUTE_CDA", "cda_m2"): "New CdA is required.",
        ("aero", "AERO_DELTA_CDA", "delta_CdA"): "Delta CdA is required.",
        ("tire", "TIRE_DB_LOOKUP", "tire_db_id"): "Tire DB row is required.",
        ("tire", "TIRE_IMPROVEMENT_PCT", "tire_improvement_pct"): "Tire improvement % is required.",
    }
    if (domain, proposal_type, field_key) in special:
        return special[(domain, proposal_type, field_key)]
    label = field_meta(field_key).get("label") or field_key.replace("_", " ")
    return f"{label} is required."


def friendly_message(message: str) -> str:
    text = str(message or "").strip()
    mapping = {
        "curb_mass_kg required": "New curb mass is required.",
        "GVWR_kg required": "GVWR is required.",
        "test_mass_kg required": "Custom test mass is required.",
        "GCWR_kg required": "GCWR is required.",
        "trailer_weight_kg required": "Trailer mass is required.",
        "GVWR_kg cannot be lower than curb_mass_kg": "GVWR cannot be lower than curb mass.",
        "vehicle_mass_at_gcwr cannot be lower than curb_mass_kg": "Vehicle mass at GCWR cannot be lower than curb mass.",
        "trailer_weight_kg must be lower than GCWR_kg": "Trailer mass must be lower than GCWR.",
        "EPA ETW / TWC target unavailable": "Target ETW / TWC could not be resolved.",
        "Curb mass is required.": "Curb mass is required.",
        "Curb mass must be a finite number.": "Curb mass must be a finite number.",
        "Curb mass must be greater than zero.": "Curb mass must be greater than zero.",
        "Curb mass is outside the canonical EPA TWC table.": "Curb mass is outside the canonical EPA TWC table.",
        "Baseline/source RRC is required for pressure-only estimate.": "Source RRC is required for pressure-only estimate.",
        "Reference front/rear pressure is required for pressure-only estimate.": "Reference front/rear pressure is required for pressure-only estimate.",
        "Requested front/rear pressure is required when target RRC is blank.": "Requested front/rear pressure is required when target RRC is blank.",
        "Requested pressures must be between 20 and 60 psi.": "Requested pressures must be between 20 and 60 psi.",
        "Reference pressures are outside the supported 20 to 60 psi range.": "Reference pressures are outside the supported 20 to 60 psi range.",
        "Resolved proposal mass is required to convert RRC into Tire ABC.": "Resolved proposal mass is required to convert RRC into Tire ABC.",
    }
    if text in mapping:
        return mapping[text]
    if "int() argument" in text.lower():
        return "Input validation needs review."
    return text


def _tire_rule_issues(resolved: dict | None) -> list[dict]:
    payload = dict(resolved or {})
    issues = payload.get("tire_rule_issues")
    if not isinstance(issues, list):
        return []
    return [dict(item or {}) for item in issues]


def _friendly_tire_issue_messages(resolved: dict | None) -> list[str]:
    messages = []
    for issue in _tire_rule_issues(resolved):
        message = friendly_message(str(issue.get("message") or ""))
        if message and message not in messages:
            messages.append(message)
    if messages:
        return messages
    note = friendly_message(str(dict(resolved or {}).get("tire_rule_notes") or ""))
    return [note] if note else []


def _has_only_nonblocking_tire_warnings(resolved: dict | None) -> bool:
    payload = dict(resolved or {})
    status = str(payload.get("tire_rule_status") or payload.get("tire_review_status") or "").strip().lower()
    issues = _tire_rule_issues(payload)
    if not issues:
        return False
    severities = {str(issue.get("severity") or "").strip().lower() for issue in issues}
    return status == "ok" and severities == {"warning"}

def proposal_status_label(status_payload: dict | None) -> str:
    payload = dict(status_payload or {})
    status = str(payload.get("status") or "not_configured")
    if status == "applied_ready":
        return "Applied — Ready"
    if status == "applied_incomplete":
        return str(payload.get("message") or "Applied — Incomplete")
    if status == "inherited":
        return "Inherited"
    if status == "not_used":
        return "Not used"
    return "Not configured"


def resolve_domain_display(domain: str, baseline: dict, proposal: dict) -> dict:
    domain_payload = dict(dict(proposal.get("domains") or {}).get(domain) or {})
    proposal_type = str(domain_payload.get("proposal_type") or "INHERIT")
    selection_mode = str(domain_payload.get("selection_mode") or proposal_type)
    inputs = sanitize_domain_inputs(domain, proposal_type, selection_mode, dict(dict(proposal.get("inputs") or {}).get(domain) or {}))
    if domain == "mass":
        return _resolve_mass_display(baseline, proposal_type, inputs)
    if domain == "aero":
        return _resolve_aero_display(baseline, proposal_type, inputs)
    if domain == "tire":
        return _resolve_tire_display(baseline, proposal_type, inputs)
    return _resolve_component_display(domain, baseline, proposal_type, selection_mode, inputs)


def rows_for_active_domain(domain: str, state: dict) -> list[str]:
    rows = ["proposal_type"]
    seen = set(rows)
    for proposal in list(state.get("proposals") or []):
        domain_payload = dict(dict(proposal.get("domains") or {}).get(domain) or {})
        proposal_type = str(domain_payload.get("proposal_type") or "INHERIT")
        selection_mode = str(domain_payload.get("selection_mode") or proposal_type)
        for field_key in applicable_fields(domain, proposal_type, selection_mode):
            if field_key not in seen:
                rows.append(field_key)
                seen.add(field_key)
    return rows


def is_field_editable(domain: str, proposal_type: str, field_key: str, selection_mode: str | None = None) -> bool:
    if proposal_type == "INHERIT":
        return False
    if field_key == "proposal_type":
        return False
    if proposal_is_not_used(proposal_type, selection_mode, domain=domain):
        return False
    if field_key in CALCULATED_FIELDS and not (domain == "mass" and proposal_type == "CUSTOM_MASS" and field_key == "test_mass_basis"):
        return False
    if field_key in READONLY_FIELDS:
        return False
    if field_key in PROPOSAL_READONLY_FIELDS.get((domain, proposal_type), set()):
        return False
    if domain == "mass" and proposal_type == "MASS_TWC_SHIFT" and field_key == "target_mass_kg":
        return _is_select_target_mode(dict().get("shift_steps"))
    return field_key in applicable_fields(domain, proposal_type, selection_mode)


def is_field_editable_with_inputs(domain: str, proposal_type: str, field_key: str, selection_mode: str | None = None, inputs: dict | None = None) -> bool:
    if not is_field_editable(domain, proposal_type, field_key, selection_mode):
        return False
    if domain == "mass" and proposal_type == "MASS_TWC_SHIFT" and field_key == "target_mass_kg":
        return _is_select_target_mode(dict(inputs or {}).get("shift_steps"))
    return True


def _resolve_mass_display(baseline: dict, proposal_type: str, inputs: dict) -> dict:
    if proposal_type == "INHERIT":
        return {
            "mass_kg": _to_float(baseline.get("mass_kg")),
            "current_curb_mass_kg": _to_float(baseline.get("mass_kg")),
            "test_mass_kg": _to_float(baseline.get("test_mass_kg")),
            "test_mass_basis": baseline.get("test_mass_basis"),
            "weight_dist_fr_pct": _to_float(baseline.get("weight_dist_fr_pct")),
            "inertia_class": _to_float(baseline.get("inertia_class")),
            "target_twc_interval": baseline.get("target_twc_interval"),
            "payload_kg": _to_float(baseline.get("payload_kg")),
            "options_kg": _to_float(baseline.get("options_kg")),
            "gvwr_kg": _to_float(baseline.get("gvwr_kg")),
            "gcwr_kg": _to_float(baseline.get("gcwr_kg")),
            "trailer_mass_kg": _to_float(baseline.get("trailer_mass_kg")),
        }
    return dict(resolve_mass_proposal(baseline, proposal_type, inputs).get("resolved_snapshot") or {})


def _resolve_aero_display(baseline: dict, proposal_type: str, inputs: dict) -> dict:
    baseline_cda = _to_float(baseline.get("cda_m2"))
    if proposal_type == "INHERIT":
        return {"baseline_CdA": baseline_cda, "cda_m2": baseline_cda}
    if proposal_type == "AERO_NOT_USED":
        return {"baseline_CdA": baseline_cda}
    if proposal_type == "AERO_ABSOLUTE_CDA":
        new_cda = _to_float(inputs.get("cda_m2"))
        delta = None if new_cda is None or baseline_cda is None else new_cda - baseline_cda
        return {
            "baseline_CdA": baseline_cda,
            "cda_m2": new_cda,
            "delta_CdA": delta,
            "Cd_display": None if new_cda is None else new_cda / _to_float(inputs.get("frontal_area_m2") or baseline.get("frontal_area_m2") or 1.0),
            "frontal_area_m2": _to_float(inputs.get("frontal_area_m2") or baseline.get("frontal_area_m2")),
        }
    delta = _to_float(inputs.get("cda_m2") or inputs.get("delta_CdA"))
    new_cda = None if delta is None or baseline_cda is None else baseline_cda + delta
    return {"baseline_CdA": baseline_cda, "delta_CdA": delta, "new_CdA": new_cda}


def _resolve_tire_display(baseline: dict, proposal_type: str, inputs: dict) -> dict:
    if proposal_type == "INHERIT":
        return {
            "tire_db_id": baseline.get("tire_db_id"),
            "tire_code": baseline.get("tire_code"),
            "rrc_N_per_kN": _to_float(baseline.get("rrc_N_per_kN")),
            "target_rrc_N_per_kN": None,
            "front_pressure_psi": _to_float(baseline.get("front_pressure_psi")),
            "rear_pressure_psi": _to_float(baseline.get("rear_pressure_psi")),
            "tire_load_mass_used_kg": _to_float(baseline.get("tire_load_mass_used_kg")),
            "tire_load_mass_basis": baseline.get("tire_load_mass_basis"),
            "tire_review_status": "OK",
            "tire_rule_status": "OK",
            "tire_rule_notes": "Inherited.",
            "tire_rule_issues": [],
        }
    result = resolve_tire_proposal(baseline, proposal_type, inputs, current_snapshot=baseline)
    resolved = dict(result.get("resolved_snapshot") or {})
    resolved["tire_rule_issues"] = deepcopy(list(result.get("issues") or []))
    return resolved

def _resolve_component_display(domain: str, baseline: dict, proposal_type: str, selection_mode: str, inputs: dict) -> dict:
    prefixes = {
        "transmission": ("trans_A_coef_N", "trans_B_coef_Npkph", "trans_C_coef_Npkph2"),
        "brake": ("brake_A_coef_N", "brake_B_Npkph", "brake_C_coef_Npkph2"),
        "axle_hubs": ("axle_hub_A", "axle_hub_B", "axle_hub_C"),
        "parasitic": ("parasitic_A_coef_N", "parasitic_B_Npkph", "parasitic_C_coef_Npkph2"),
    }
    a_key, b_key, c_key = prefixes[domain]
    base_triplet = {
        a_key: _to_float(baseline.get(a_key)),
        b_key: _to_float(baseline.get(b_key)),
        c_key: _to_float(baseline.get(c_key)),
    }
    source_total = {}
    if domain == "transmission":
        base_triplet["transmission_application_mode"] = _normalize_transmission_application_mode(
            inputs.get("transmission_application_mode")
            or baseline.get("transmission_application_mode")
            or dict(baseline.get("transmission_losses") or {}).get("transmission_application_mode")
        )
        source_total = {
            "source_abc_total_A": _to_float(baseline.get("source_abc_total_A")),
            "source_abc_total_B": _to_float(baseline.get("source_abc_total_B")),
            "source_abc_total_C": _to_float(baseline.get("source_abc_total_C")),
        }
    if proposal_type == "INHERIT":
        return {**base_triplet, **source_total}
    resolved = {**deepcopy(base_triplet), **source_total}
    if proposal_type.endswith("METADATA_ONLY"):
        resolved.update({a_key: inputs.get(a_key), b_key: inputs.get(b_key), c_key: inputs.get(c_key)})
        return resolved
    if proposal_type.endswith("NOT_USED"):
        return resolved
    if proposal_type == "TRANS_LOSS_PCT":
        pct = _to_float(inputs.get("transmission_loss_pct"))
        source_values = (
            source_total.get("source_abc_total_A"),
            source_total.get("source_abc_total_B"),
            source_total.get("source_abc_total_C"),
        )
        if pct is not None and all(value is not None for value in source_values):
            resolved[a_key] = source_values[0] * (pct / 100.0)
            resolved[b_key] = source_values[1] * (pct / 100.0)
            resolved[c_key] = source_values[2] * (pct / 100.0)
        else:
            resolved[a_key] = None
            resolved[b_key] = None
            resolved[c_key] = None
        resolved["transmission_percent_basis"] = "SOURCE_ABC_TOTAL"
        resolved["transmission_rule_version"] = "COASTDOWN_SHARE_V1"
        return resolved
    component_mode = canonical_component_mode(domain, proposal_type, selection_mode, inputs)
    if domain == "brake" and component_mode == "RESIDUAL_TORQUE":
        torque_front = _to_float(inputs.get("residual_torque_front_Nm"))
        torque_rear = _to_float(inputs.get("residual_torque_rear_Nm"))
        torque_total = _to_float(inputs.get("residual_torque_total_Nm"))
        wheel_radius = _to_float(inputs.get("wheel_radius_m"))
        torque_sum = torque_total if torque_total is not None else sum(value for value in (torque_front, torque_rear) if value is not None)
        if torque_sum is not None and wheel_radius not in (None, 0.0):
            drag_force = torque_sum / wheel_radius
            resolved["brake_drag_force_N"] = drag_force
            resolved[a_key] = None if base_triplet[a_key] is None else base_triplet[a_key] + drag_force
        return resolved
    if component_mode == "DELTA_ABC":
        delta_a = _to_float(inputs.get("delta_A")) or 0.0
        delta_b = _to_float(inputs.get("delta_B")) or 0.0
        delta_c = _to_float(inputs.get("delta_C")) or 0.0
        resolved[a_key] = None if base_triplet[a_key] is None else base_triplet[a_key] + delta_a
        resolved[b_key] = None if base_triplet[b_key] is None else base_triplet[b_key] + delta_b
        resolved[c_key] = None if base_triplet[c_key] is None else base_triplet[c_key] + delta_c
        if domain == "transmission":
            resolved["new_trans_A"] = resolved[a_key]
            resolved["new_trans_B"] = resolved[b_key]
            resolved["new_trans_C"] = resolved[c_key]
        return resolved
    resolved[a_key] = _to_float(inputs.get(a_key))
    resolved[b_key] = _to_float(inputs.get(b_key))
    resolved[c_key] = _to_float(inputs.get(c_key))
    if domain == "transmission":
        resolved["new_trans_A"] = resolved[a_key]
        resolved["new_trans_B"] = resolved[b_key]
        resolved["new_trans_C"] = resolved[c_key]
    return resolved


def _normalize_transmission_application_mode(value) -> str:
    text = str(value or "").strip().upper()
    if text == "KEEP_TOTAL_FIXED":
        return "KEEP_TOTAL_FIXED"
    return "APPLY_DELTA_TO_TOTAL"


def _to_float(value):
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _normalize_mass_twc_shift_inputs(cleaned: dict, raw_inputs: dict) -> dict:
    result = deepcopy(dict(cleaned or {}))
    result["curb_position"] = _normalize_curb_position(raw_inputs.get("curb_position") or result.get("curb_position"))
    raw_value = raw_inputs.get("shift_steps")
    if _is_select_target_mode(raw_value):
        result.pop("shift_steps", None)
        result.pop("target_side", None)
        return result
    text = str(raw_value or "").strip()
    if text in {"+1", "+2", "+3", "-1", "-2", "-3"}:
        result["shift_steps"] = float(text)
        result.pop("target_side", None)
        result.pop("target_mass_kg", None)
        return result
    numeric = _to_float(raw_value)
    if numeric is not None:
        explicit_side = str(raw_inputs.get("target_side") or result.get("target_side") or "").strip().lower()
        signed_shift = float(numeric)
        if signed_shift > 0 and explicit_side == "down":
            signed_shift = -signed_shift
        result["shift_steps"] = signed_shift
        result.pop("target_side", None)
        result.pop("target_mass_kg", None)
    return result


def _is_select_target_mode(value) -> bool:
    return str(value or "").strip().lower() == "select target"


def _normalize_curb_position(value) -> str:
    text = str(value or "").strip().lower()
    if text == "bottom":
        return "Bottom"
    if text == "mid":
        return "Mid"
    return "Top"
