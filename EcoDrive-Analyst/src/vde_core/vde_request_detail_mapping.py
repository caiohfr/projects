from copy import deepcopy

from src.vde_core.vde_component_modes import canonical_component_mode
from src.vde_core.vde_request_contract import FIELD_KEY_ALIASES
from src.vde_core.vde_tire_modes import canonical_tire_proposal_type


def preferred_detail_alias(field_key: str) -> str:
    aliases = FIELD_KEY_ALIASES.get(field_key)
    if aliases:
        return str(aliases[0])
    return str(field_key)


def detail_key_for_domain_field(domain_key: str, proposal_type: str, field_key: str, seed: dict | None = None) -> str:
    proposal_type = str(proposal_type or "").strip().upper()
    seed = deepcopy(dict(seed or {}))
    domain = str(domain_key or "").strip().lower()

    if domain == "mass":
        if proposal_type == "CUSTOM_MASS" and field_key in {"mass_kg", "test_mass_kg"}:
            return "test_mass_kg"
        if proposal_type == "EPA_STATUS":
            return {
                "mass_kg": "curb_mass_kg",
                "inertia_class": "inertia_class",
                "test_mass_kg": "test_mass_kg",
            }.get(field_key, preferred_detail_alias(field_key))
        if proposal_type == "EPA_CURB_TO_TWC":
            return {
                "target_curb_mass_kg": "target_curb_mass_kg",
                "mass_kg": "mass_kg",
                "inertia_class": "inertia_class",
                "test_mass_kg": "test_mass_kg",
                "test_mass_basis": "test_mass_basis",
            }.get(field_key, preferred_detail_alias(field_key))
        if proposal_type in {"MASS_TWC_SHIFT", "EPA_PLUS_1_TWC"}:
            return {
                "shift_steps": "shift_steps",
                "target_side": "target_side",
                "curb_position": "curb_position",
                "target_mass_kg": "target_mass_kg",
                "test_mass_kg": "target_mass_kg",
                "inertia_class": "reference_mass_kg",
                "mass_kg": "reference_mass_kg",
            }.get(field_key, preferred_detail_alias(field_key))
        if proposal_type == "PERFORMANCE_CURB_MASS":
            return {
                "mass_kg": "curb_mass_kg",
                "preset": "preset",
                "custom_delta_kg": "custom_delta_kg",
                "test_mass_kg": "effective_test_mass_kg",
            }.get(field_key, preferred_detail_alias(field_key))
        if proposal_type == "WLTP_MASS_LINE":
            return {
                "line_type": "line_type",
                "mass_kg": "mass_kg",
                "payload_kg": "payload_kg",
                "options_kg": "optional_weight_kg",
                "test_mass_low_kg": "test_mass_low_kg",
                "test_mass_high_kg": "test_mass_high_kg",
                "test_mass_kg": "effective_test_mass_kg",
            }.get(field_key, preferred_detail_alias(field_key))
        if proposal_type == "GVWR":
            return {
                "gvwr_kg": "GVWR_kg",
                "payload_kg": "payload_kg",
                "test_mass_kg": "test_mass_kg",
            }.get(field_key, preferred_detail_alias(field_key))
        if proposal_type == "GCWR":
            return {
                "gcwr_kg": "GCWR_kg",
                "trailer_mass_kg": "trailer_weight_kg",
                "vehicle_mass_at_gcwr": "vehicle_mass_at_gcwr",
                "trailer_A": "trailer_A",
                "trailer_B": "trailer_B",
                "trailer_C": "trailer_C",
                "test_mass_kg": "test_mass_kg",
            }.get(field_key, preferred_detail_alias(field_key))
        return preferred_detail_alias(field_key)

    if domain == "aero":
        if field_key == "cda_m2":
            return "new_CdA" if proposal_type == "AERO_ABSOLUTE_CDA" else "delta_CdA"
        if field_key == "frontal_area_m2":
            return "Af_optional"
        return preferred_detail_alias(field_key)

    if domain == "tire":
        proposal_type = canonical_tire_proposal_type(proposal_type)
        mapping = {
            "tire_code": "tire_code",
            "front_pressure_psi": "front_pressure_psi",
            "rear_pressure_psi": "rear_pressure_psi",
            "tire_load_mass_basis": "tire_load_mass_basis",
            "tire_improvement_pct": "tire_improvement_pct",
            "rrc_N_per_kN": "rrc_N_per_kN",
            "target_rrc_N_per_kN": "target_rrc_N_per_kN",
            "smerf": "smerf",
        }
        return mapping.get(field_key, preferred_detail_alias(field_key))

    if domain == "transmission":
        if field_key in {"trans_A_coef_N", "trans_B_coef_Npkph", "trans_C_coef_Npkph2"}:
            suffix = {"trans_A_coef_N": "A", "trans_B_coef_Npkph": "B", "trans_C_coef_Npkph2": "C"}[field_key]
            if canonical_component_mode(domain, proposal_type, seed.get("selection_mode"), seed) == "ABSOLUTE_ABC":
                return f"new_trans_{suffix}"
            return f"delta_{suffix}"
        if field_key == "transmission_loss_pct":
            return "loss_pct"
        if field_key == "transmission_percent_basis":
            return "percent_basis"
        return preferred_detail_alias(field_key)

    if domain == "brake":
        if field_key in {"brake_A_coef_N", "brake_B_Npkph", "brake_C_coef_Npkph2"}:
            suffix = {"brake_A_coef_N": "A", "brake_B_Npkph": "B", "brake_C_coef_Npkph2": "C"}[field_key]
            mode = canonical_component_mode(domain, proposal_type, seed.get("selection_mode"), seed)
            if mode == "RESIDUAL_TORQUE":
                return f"brake_{suffix}"
            if mode == "ABSOLUTE_ABC":
                return f"brake_{suffix}"
            return f"delta_{suffix}"
        return preferred_detail_alias(field_key)

    if domain == "axle_hubs":
        if field_key in {"axle_hub_A", "axle_hub_B", "axle_hub_C"}:
            suffix = field_key.rsplit("_", 1)[-1]
            if canonical_component_mode(domain, proposal_type, seed.get("selection_mode"), seed) == "ABSOLUTE_ABC":
                return f"axle_hub_{suffix}"
            return f"delta_{suffix}"
        return preferred_detail_alias(field_key)

    if domain == "parasitic":
        if field_key in {"parasitic_A_coef_N", "parasitic_B_Npkph", "parasitic_C_coef_Npkph2"}:
            suffix = {"parasitic_A_coef_N": "A", "parasitic_B_Npkph": "B", "parasitic_C_coef_Npkph2": "C"}[field_key]
            if canonical_component_mode(domain, proposal_type, seed.get("selection_mode"), seed) == "ABSOLUTE_ABC":
                return f"parasitic_{suffix}"
            return f"delta_{suffix}"
        return preferred_detail_alias(field_key)

    return preferred_detail_alias(field_key)
