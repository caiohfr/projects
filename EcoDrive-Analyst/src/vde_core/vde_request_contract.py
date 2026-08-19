from __future__ import annotations

import math
import re
from copy import deepcopy


VDE_REQUEST_SCHEMA_VERSION = "0.1"


def is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(value != value)
    except Exception:
        return False


def resolve_effective_baseline(printed, correction):
    return correction if not is_blank(correction) else printed


def _normalize_text(value) -> str:
    if is_blank(value):
        return ""
    text = str(value).strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[\u2010-\u2015]+", "-", text)
    text = re.sub(r"[_/]+", " ", text)
    text = re.sub(r"[^a-z0-9%+\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


DOMAIN_ALIASES = {
    "scenario": ("scenario", "scenario context", "context"),
    "mass": ("mass", "mass aero", "mass_aero", "massa"),
    "aero": ("aero", "aerodynamic", "aerodynamics"),
    "tire": ("tire", "tires", "tyre", "tyres"),
    "transmission": ("transmission", "trans", "gearbox"),
    "brake": ("brake", "brakes"),
    "axle_hubs": ("axle hubs", "axle and hubs", "axle hub", "axle_hub", "axle hubs proposal"),
    "parasitic": ("parasitic", "parasitics", "parasitic losses", "parasitic loss"),
    "roadload": ("roadload", "abc total", "abc_total"),
    "trailer": ("trailer", "trailer gcwr", "trailer and gcwr", "gcwr trailer"),
}


def normalize_domain(value: str) -> str:
    normalized = _normalize_text(value)
    if not normalized:
        return ""
    for canonical, aliases in DOMAIN_ALIASES.items():
        if normalized == canonical or normalized in aliases:
            return canonical
    return normalized.replace(" ", "_")


def _proposal_entry(
    domain: str,
    template_label: str,
    proposal_type: str | None,
    *,
    details: dict | None = None,
    selection_mode: str | None = None,
    has_internal_equivalent: bool = True,
    notes: str = "",
) -> dict:
    domain_key = normalize_domain(domain)
    proposal_mode = "inherited" if proposal_type == "INHERIT" else "direct"
    return {
        "ok": True,
        "domain": domain_key,
        "template_label": template_label,
        "proposal_type": proposal_type,
        "details": deepcopy(details or {}),
        "selection_mode": selection_mode or template_label,
        "mode": proposal_mode,
        "has_internal_equivalent": has_internal_equivalent,
        "notes": notes,
    }


TEMPLATE_PROPOSAL_MAP = {
    "mass": {
        "Inherit": _proposal_entry("mass", "Inherit", "INHERIT"),
        "Use current EPA ETW / TWC": _proposal_entry("mass", "Use current EPA ETW / TWC", "EPA_STATUS"),
        "Curb mass → EPA TWC": _proposal_entry("mass", "Curb mass → EPA TWC", "EPA_CURB_TO_TWC"),
        "TWC shift / target class": _proposal_entry("mass", "TWC shift / target class", "MASS_TWC_SHIFT"),
        "Performance loaded mass": _proposal_entry("mass", "Performance loaded mass", "PERFORMANCE_CURB_MASS"),
        "WLTP mass line": _proposal_entry("mass", "WLTP mass line", "WLTP_MASS_LINE"),
        "GVWR loaded mass": _proposal_entry("mass", "GVWR loaded mass", "GVWR"),
        "GCWR / trailer mass": _proposal_entry("mass", "GCWR / trailer mass", "GCWR"),
        "Custom test mass": _proposal_entry("mass", "Custom test mass", "CUSTOM_MASS"),
    },
    "aero": {
        "Inherit": _proposal_entry("aero", "Inherit", "INHERIT"),
        "Absolute CdA": _proposal_entry("aero", "Absolute CdA", "AERO_ABSOLUTE_CDA"),
        "Delta CdA": _proposal_entry("aero", "Delta CdA", "AERO_DELTA_CDA"),
        "Not used": _proposal_entry(
            "aero",
            "Not used",
            "AERO_NOT_USED",
            has_internal_equivalent=False,
            notes="Aero Not used remains a legacy review-only state because physical aero exclusion is not consolidated.",
        ),
    },
    "tire": {
        "Inherit": _proposal_entry("tire", "Inherit", "INHERIT"),
        "Tire DB lookup": _proposal_entry("tire", "Tire DB lookup", "TIRE_DB_LOOKUP"),
        "Target final RRC": _proposal_entry(
            "tire",
            "Target final RRC",
            "TIRE_TARGET_RRC",
        ),
        "Tire improvement %": _proposal_entry(
            "tire",
            "Tire improvement %",
            "TIRE_IMPROVEMENT_PCT",
        ),
        "Not used": _proposal_entry(
            "tire",
            "Not used",
            "TIRE_METADATA_ONLY",
        ),
    },
    "transmission": {
        "Inherit": _proposal_entry("transmission", "Inherit", "INHERIT"),
        "Lookup from DB": _proposal_entry(
            "transmission",
            "Lookup from DB",
            "TRANS_METADATA_ONLY",
            has_internal_equivalent=False,
            notes="Legacy metadata-only type exists, but the current v2.1 matrix does not expose DB lookup as a first-class transmission proposal.",
        ),
        "Absolute ABC": _proposal_entry(
            "transmission",
            "Absolute ABC",
            "UPDATE_TRANS_DRAG_ABC",
        ),
        "Delta ABC": _proposal_entry(
            "transmission",
            "Delta ABC",
            "UPDATE_TRANS_DRAG_ABC",
        ),
        "Transmission coastdown share": _proposal_entry("transmission", "Transmission coastdown share", "TRANS_LOSS_PCT"),
        "Not used": _proposal_entry(
            "transmission",
            "Not used",
            "TRANS_LOSS_NOT_AVAILABLE",
        ),
    },
    "brake": {
        "Inherit": _proposal_entry("brake", "Inherit", "INHERIT"),
        "Lookup from DB": _proposal_entry(
            "brake",
            "Lookup from DB",
            "BRAKE_METADATA_ONLY",
            has_internal_equivalent=False,
            notes="Legacy metadata-only type exists, but the current v2.1 matrix does not expose DB lookup as a first-class brake proposal.",
        ),
        "Absolute ABC": _proposal_entry(
            "brake",
            "Absolute ABC",
            "BRAKE_DRAG_CHANGE",
        ),
        "Delta ABC": _proposal_entry(
            "brake",
            "Delta ABC",
            "BRAKE_DRAG_CHANGE",
        ),
        "Residual torque": _proposal_entry(
            "brake",
            "Residual torque",
            "BRAKE_DRAG_CHANGE",
        ),
        "Not used": _proposal_entry(
            "brake",
            "Not used",
            "BRAKE_NOT_USED",
        ),
    },
    "axle_hubs": {
        "Inherit": _proposal_entry("axle_hubs", "Inherit", "INHERIT"),
        "Lookup from DB": _proposal_entry(
            "axle_hubs",
            "Lookup from DB",
            "AXLE_HUB_METADATA_ONLY",
            has_internal_equivalent=False,
            notes="Legacy metadata-only type exists, but the current v2.1 matrix does not expose DB lookup as a first-class axle/hub proposal.",
        ),
        "Absolute ABC": _proposal_entry(
            "axle_hubs",
            "Absolute ABC",
            "AXLE_HUB_DRAG_CHANGE",
        ),
        "Delta ABC": _proposal_entry(
            "axle_hubs",
            "Delta ABC",
            "AXLE_HUB_DRAG_CHANGE",
        ),
        "Not used": _proposal_entry(
            "axle_hubs",
            "Not used",
            "AXLE_HUB_NOT_USED",
        ),
    },
    "parasitic": {
        "Inherit": _proposal_entry("parasitic", "Inherit", "INHERIT"),
        "Lookup from DB": _proposal_entry(
            "parasitic",
            "Lookup from DB",
            "PARASITIC_METADATA_ONLY",
            has_internal_equivalent=False,
            notes="Legacy metadata-only type exists, but the current v2.1 matrix does not expose DB lookup as a first-class parasitic proposal.",
        ),
        "Absolute ABC": _proposal_entry(
            "parasitic",
            "Absolute ABC",
            "PARASITIC_LOSS_CHANGE",
        ),
        "Delta ABC": _proposal_entry(
            "parasitic",
            "Delta ABC",
            "PARASITIC_LOSS_CHANGE",
        ),
        "Not used": _proposal_entry(
            "parasitic",
            "Not used",
            "PARASITIC_NOT_USED",
        ),
    },
}

VISIBLE_TEMPLATE_PROPOSAL_LABELS = {
    "mass": (
        "Inherit",
        "Curb mass → EPA TWC",
        "TWC shift / target class",
        "Performance loaded mass",
        "WLTP mass line",
        "GVWR loaded mass",
        "GCWR / trailer mass",
        "Custom test mass",
    ),
    "aero": (
        "Inherit",
        "Absolute CdA",
        "Delta CdA",
    ),
    "tire": (
        "Inherit",
        "Tire DB lookup",
        "Target final RRC",
        "Tire improvement %",
        "Not used",
    ),
    "transmission": (
        "Inherit",
        "Lookup from DB",
        "Absolute ABC",
        "Delta ABC",
        "Transmission coastdown share",
        "Not used",
    ),
    "brake": (
        "Inherit",
        "Lookup from DB",
        "Absolute ABC",
        "Delta ABC",
        "Residual torque",
        "Not used",
    ),
    "axle_hubs": (
        "Inherit",
        "Lookup from DB",
        "Absolute ABC",
        "Delta ABC",
        "Not used",
    ),
    "parasitic": (
        "Inherit",
        "Lookup from DB",
        "Absolute ABC",
        "Delta ABC",
        "Not used",
    ),
}


TEMPLATE_PROPOSAL_ALIASES = {
    "mass": {
        "inherit": "Inherit",
        "use current epa etw / twc": "Use current EPA ETW / TWC",
        "use current epa etw twc": "Use current EPA ETW / TWC",
        "epa status": "Use current EPA ETW / TWC",
        "epa status mass": "Use current EPA ETW / TWC",
        "curb mass epa twc": "Curb mass → EPA TWC",
        "curb mass - epa twc": "Curb mass → EPA TWC",
        "curb mass to epa twc": "Curb mass → EPA TWC",
        "curb mass -> epa twc": "Curb mass → EPA TWC",
        "epa curb to twc": "Curb mass → EPA TWC",
        "twc shift": "TWC shift / target class",
        "twc shift target class": "TWC shift / target class",
        "epa plus 1 twc": "TWC shift / target class",
        "epa+1 twc": "TWC shift / target class",
        "performance curb mass": "Performance loaded mass",
        "performance loaded mass": "Performance loaded mass",
        "wltp mass line": "WLTP mass line",
        "gvwr loaded mass": "GVWR loaded mass",
        "gvwr": "GVWR loaded mass",
        "gcwr trailer mass": "GCWR / trailer mass",
        "gcwr trailer": "GCWR / trailer mass",
        "gcwr / trailer mass": "GCWR / trailer mass",
        "custom test mass": "Custom test mass",
    },
    "aero": {
        "inherit": "Inherit",
        "absolute cda": "Absolute CdA",
        "absolute cd area": "Absolute CdA",
        "delta cda": "Delta CdA",
        "delta abc": "Delta CdA",
        "not used": "Not used",
        "not used": "Not used",
    },
    "tire": {
        "inherit": "Inherit",
        "tire db lookup": "Tire DB lookup",
        "lookup from db": "Tire DB lookup",
        "db lookup": "Tire DB lookup",
        "target final rrc": "Target final RRC",
        "manual rrc": "Target final RRC",
        "tire smerf rrc change": "Target final RRC",
        "tire target rrc": "Target final RRC",
        "tire smerf rrc change": "Target final RRC",
        "tire improvement %": "Tire improvement %",
        "tire improvement pct": "Tire improvement %",
        "not used": "Not used",
        "not used": "Not used",
    },
    "transmission": {
        "inherit": "Inherit",
        "lookup from db": "Lookup from DB",
        "absolute abc": "Absolute ABC",
        "delta abc": "Delta ABC",
        "transmission loss %": "Transmission coastdown share",
        "transmission loss pct": "Transmission coastdown share",
        "transmission coastdown share": "Transmission coastdown share",
        "not used": "Not used",
        "not used": "Not used",
    },
    "brake": {
        "inherit": "Inherit",
        "lookup from db": "Lookup from DB",
        "absolute abc": "Absolute ABC",
        "delta abc": "Delta ABC",
        "residual torque": "Residual torque",
        "not used": "Not used",
        "not used": "Not used",
    },
    "axle_hubs": {
        "inherit": "Inherit",
        "lookup from db": "Lookup from DB",
        "absolute abc": "Absolute ABC",
        "delta abc": "Delta ABC",
        "not used": "Not used",
        "not used": "Not used",
    },
    "parasitic": {
        "inherit": "Inherit",
        "lookup from db": "Lookup from DB",
        "absolute abc": "Absolute ABC",
        "delta abc": "Delta ABC",
        "not used": "Not used",
        "not used": "Not used",
    },
}


def normalize_template_proposal_type(domain: str, value: str) -> dict:
    domain_key = normalize_domain(domain)
    if domain_key not in TEMPLATE_PROPOSAL_MAP:
        return {
            "ok": False,
            "domain": domain_key,
            "template_value": value,
            "error": "unknown_domain",
            "message": f"Unknown template domain '{domain}'.",
        }

    if is_blank(value):
        return deepcopy(TEMPLATE_PROPOSAL_MAP[domain_key]["Inherit"])

    normalized_value = _normalize_text(value)
    aliases = TEMPLATE_PROPOSAL_ALIASES.get(domain_key, {})
    canonical_label = aliases.get(normalized_value)
    if canonical_label is None:
        return {
            "ok": False,
            "domain": domain_key,
            "template_value": value,
            "error": "unknown_proposal_type",
            "message": f"Unknown template proposal type '{value}' for domain '{domain_key}'.",
        }
    return deepcopy(TEMPLATE_PROPOSAL_MAP[domain_key][canonical_label])


FIELD_KEY_ALIASES = {
    "notes": ("notes", "description"),
    "selected_baseline_vde_id": ("selected_baseline_vde_id", "baseline_vde_id", "baseline_id"),
    "walk_from": ("walk_from",),
    "legislation": ("legislation",),
    "category": ("category",),
    "make": ("make", "manufacturer"),
    "model": ("model", "vehicle_label"),
    "year": ("year", "model_year"),
    "electrification": ("electrification",),
    "transmission_type": ("transmission_type",),
    "drive_type": ("drive_type",),
    "fuel_type": ("fuel_type",),
    "cycle_name": ("cycle_name", "cycle"),
    "cycle_source": ("cycle_source",),
    "mass_proposal_type": ("mass_proposal_type", "proposal_type"),
    "aero_proposal_type": ("aero_proposal_type", "proposal_type"),
    "tire_proposal_type": ("tire_proposal_type", "proposal_type"),
    "transmission_proposal_type": ("transmission_proposal_type", "proposal_type"),
    "brake_proposal_type": ("brake_proposal_type", "proposal_type"),
    "axle_hubs_proposal_type": ("axle_hubs_proposal_type", "proposal_type"),
    "parasitic_proposal_type": ("parasitic_proposal_type", "proposal_type"),
    "abc_total_basis": ("abc_total_basis", "roadload_basis"),
    "A": ("A", "baseline_A_N", "A_coef_N"),
    "B": ("B", "baseline_B_N_per_kph", "B_coef_Npkph"),
    "C": ("C", "baseline_C_N_per_kph2", "C_coef_Npkph2"),
    "abc_total_source_ui": ("abc_total_source_ui", "roadload_source_path", "roadload_source"),
    "mass_kg": ("mass_kg", "curb_mass_kg", "baseline_mass_kg"),
    "target_curb_mass_kg": ("target_curb_mass_kg", "curb_mass_kg", "mass_kg"),
    "current_curb_mass_kg": ("current_curb_mass_kg", "mass_kg", "curb_mass_kg"),
    "inertia_class": ("inertia_class", "prep_inertia_class"),
    "test_mass_basis": ("test_mass_basis", "vde_mass_basis"),
    "test_mass_kg": ("test_mass_kg", "effective_test_mass_kg"),
    "target_twc_interval": ("target_twc_interval",),
    "target_twc_lower_bound_exclusive": ("target_twc_lower_bound_exclusive",),
    "target_twc_upper_bound_inclusive": ("target_twc_upper_bound_inclusive",),
    "payload_kg": ("payload_kg", "payload_display_kg"),
    "weight_dist_fr_pct": ("weight_dist_fr_pct",),
    "wltp_category": ("wltp_category",),
    "tire_load_mass_basis": ("tire_load_mass_basis", "load_basis"),
    "mass_delta_class_offset": ("mass_delta_class_offset", "shift_steps", "twc_shift_steps"),
    "mass_delta_class_point": ("mass_delta_class_point", "target_side", "twc_target_side"),
    "trailer_code": ("trailer_code",),
    "trailer_db_id": ("trailer_db_id",),
    "mass_profile_gcwr_kg": ("mass_profile_gcwr_kg", "gcwr_kg", "GCWR_kg"),
    "mass_profile_trailer_mass_kg": ("mass_profile_trailer_mass_kg", "trailer_mass_kg", "trailer_weight_kg"),
    "trailer_A": ("trailer_A", "trailer_A_coef_N"),
    "trailer_B": ("trailer_B", "trailer_B_coef_Npkph"),
    "trailer_C": ("trailer_C", "trailer_C_coef_Npkph2"),
    "cda_m2": ("cda_m2", "new_CdA", "baseline_CdA", "CdA"),
    "frontal_area_m2": ("frontal_area_m2", "Af_optional"),
    "cda_source": ("cda_source", "source"),
    "tire_code": ("tire_code", "baseline_tire_code", "new_tire_code"),
    "tire_db_id": ("tire_db_id",),
    "tire_size": ("tire_size",),
    "front_pressure_psi": ("front_pressure_psi", "psi_front"),
    "rear_pressure_psi": ("rear_pressure_psi", "psi_rear"),
    "hot_front_pressure_psi": ("hot_front_pressure_psi",),
    "hot_rear_pressure_psi": ("hot_rear_pressure_psi",),
    "rrc_N_per_kN": ("rrc_N_per_kN", "baseline_RRC_optional", "delta_RRC_optional"),
    "target_rrc_N_per_kN": ("target_rrc_N_per_kN", "delta_RRC_optional"),
    "smerf": ("smerf", "baseline_SMERF_optional", "delta_SMERF_optional"),
    "tire_improvement_pct": ("tire_improvement_pct", "improvement_pct"),
    "tire_notes": ("tire_notes", "notes"),
    "transmission_component_db_id": ("transmission_component_db_id", "transmission_db_id", "component_db_id"),
    "transmission_vde_db_id": ("transmission_vde_db_id", "transmission_source_vde_id"),
    "trans_A_coef_N": ("trans_A_coef_N", "new_trans_A", "baseline_trans_A", "delta_A"),
    "trans_B_coef_Npkph": ("trans_B_coef_Npkph", "new_trans_B", "baseline_trans_B", "delta_B"),
    "trans_C_coef_Npkph2": ("trans_C_coef_Npkph2", "new_trans_C", "baseline_trans_C", "delta_C"),
    "transmission_loss_pct": ("transmission_loss_pct", "loss_pct"),
    "transmission_percent_basis": ("transmission_percent_basis", "percent_basis"),
    "brake_component_db_id": ("brake_component_db_id", "brake_db_id", "component_db_id"),
    "brake_vde_db_id": ("brake_vde_db_id", "brake_source_vde_id"),
    "brake_A_coef_N": ("brake_A_coef_N", "brake_A", "baseline_component_A", "delta_A"),
    "brake_B_Npkph": ("brake_B_Npkph", "brake_B", "baseline_component_B", "delta_B"),
    "brake_C_coef_Npkph2": ("brake_C_coef_Npkph2", "brake_C", "baseline_component_C", "delta_C"),
    "residual_torque_total_Nm": ("residual_torque_total_Nm", "brake_drag_force_N"),
    "axle_hubs_component_db_id": ("axle_hubs_component_db_id", "axle_hub_db_id", "component_db_id"),
    "axle_hubs_vde_db_id": ("axle_hubs_vde_db_id", "axle_hub_source_vde_id"),
    "axle_hub_A": ("axle_hub_A", "baseline_component_A", "delta_A"),
    "axle_hub_B": ("axle_hub_B", "baseline_component_B", "delta_B"),
    "axle_hub_C": ("axle_hub_C", "baseline_component_C", "delta_C"),
    "parasitic_component_db_id": ("parasitic_component_db_id", "parasitic_db_id", "component_db_id"),
    "parasitic_vde_db_id": ("parasitic_vde_db_id", "parasitic_source_vde_id"),
    "parasitic_A_coef_N": ("parasitic_A_coef_N", "parasitic_A", "baseline_component_A", "delta_A"),
    "parasitic_B_Npkph": ("parasitic_B_Npkph", "parasitic_B", "baseline_component_B", "delta_B"),
    "parasitic_C_coef_Npkph2": ("parasitic_C_coef_Npkph2", "parasitic_C", "baseline_component_C", "delta_C"),
}
