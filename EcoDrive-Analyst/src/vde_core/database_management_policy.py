from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.vde_core.database_management_contract import EntityType, normalize_entity_type, normalize_record_origin


class FieldAccess(str, Enum):
    EDITABLE = "EDITABLE"
    ADVANCED_CORRECTION = "ADVANCED_CORRECTION"
    DERIVED = "DERIVED"
    IMMUTABLE = "IMMUTABLE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class EntityFieldPolicy:
    editable: frozenset[str]
    advanced_correction: frozenset[str]
    derived: frozenset[str]
    immutable: frozenset[str]

    @property
    def known_fields(self) -> frozenset[str]:
        return self.editable | self.advanced_correction | self.derived | self.immutable


COMMON_IMMUTABLE = frozenset({"id", "created_at", "updated_at", "record_status"})
COMMON_SOURCE = frozenset({"source_name", "source_record_id", "notes"})

VDE_METADATA = frozenset(
    {
        "legislation",
        "category",
        "make",
        "model",
        "year",
        "notes",
        "source_name",
        "source_record_id",
        "engine_type",
        "engine_model",
        "engine_size_l",
        "engine_aspiration",
        "transmission_type",
        "transmission_model",
        "drive_type",
        "cycle_name",
        "cycle_source",
    }
)
VDE_PHYSICAL = frozenset(
    {
        "mass_kg",
        "test_mass_kg",
        "test_mass_low_kg",
        "test_mass_high_kg",
        "test_mass_basis",
        "inertia_class",
        "cda_m2",
        "weight_dist_fr_pct",
        "payload_kg",
        "gvwr_kg",
        "gcwr_kg",
        "trailer_mass_kg",
        "front_pressure_psi",
        "rear_pressure_psi",
        "coast_A_N",
        "coast_B_N_per_kph",
        "coast_C_N_per_kph2",
        "trans_A_coef_N",
        "trans_B_coef_Npkph",
        "trans_C_coef_Npkph2",
        "brake_A_coef_N",
        "brake_B_coef_Npkph",
        "brake_C_coef_Npkph2",
        "aero_C_coef_Npkph2",
        "rrc_N_per_kN",
        "front_tire_id",
        "rear_tire_id",
        "tire_load_mass_basis",
    }
)
VDE_DERIVED = frozenset(
    {
        "vde_urb_mj",
        "vde_hw_mj",
        "vde_urb_mj_per_km",
        "vde_hw_mj_per_km",
        "vde_net_mj_per_km",
        "vde_total_mj_per_km",
        "vde_low_mj_per_km",
        "vde_mid_mj_per_km",
        "vde_high_mj_per_km",
        "vde_extra_high_mj_per_km",
        "tire_load_mass_used_kg",
        "tire_A_final",
        "tire_B_final",
        "tire_C_final",
        "mass_rule_status",
        "mass_rule_notes",
    }
)

FUEL_METADATA = frozenset(
    {
        "vde_id",
        "electrification",
        "fuel_type",
        "method_note",
        "source_name",
        "source_record_id",
        "label_program",
        "label_version_year",
        "label_vehicle_category",
        "label_cycle_set",
        "label_class",
    }
)
FUEL_INPUTS = frozenset(
    {
        "eta_pt_est",
        "bev_eff_drive",
        "utility_factor_pct",
        "engine_max_power_kw",
        "engine_rpm_max_power",
        "engine_max_torque_nm",
        "engine_rpm_max_torque",
        "gear_count",
        "final_drive_ratio",
        "battery_capacity_kwh",
        "battery_usable_kwh",
        "bms_discharge_limit_kw",
        "bms_regen_limit_kw",
        "bms_note",
        "ambient_temp_c",
        "ac_on",
        "tire_front_psi",
        "tire_rear_psi",
        "scenario_payload_kg",
        "energy_basis",
        "engine_method",
        "engine_version",
        "assumptions_json",
        "provenance_json",
    }
)
FUEL_DERIVED = frozenset(
    {
        "energy_Wh_per_km",
        "fuel_km_per_l",
        "fuel_l_per_100km",
        "gco2_per_km",
        "energy_low_Wh_per_km",
        "energy_mid_Wh_per_km",
        "energy_high_Wh_per_km",
        "energy_xhigh_Wh_per_km",
        "fuel_low_l_per_100km",
        "fuel_mid_l_per_100km",
        "fuel_high_l_per_100km",
        "fuel_xhigh_l_per_100km",
        "gco2_low_per_km",
        "gco2_mid_per_km",
        "gco2_high_per_km",
        "gco2_xhigh_per_km",
        "source_vde_revision",
        "review_status",
    }
)

TIRE_EDITABLE = frozenset(
    {
        "tire_test_code",
        "manufacturer",
        "model",
        "size_code",
        "load_index",
        "speed_rating",
        "notes",
        "source_name",
        "source_record_id",
        "standard_family",
        "standard_version",
        "test_method",
        "test_source",
        "test_date",
        "test_mileage_km",
        "is_tested_value",
        "is_estimated_value",
        "break_in_distance_km",
        "is_broken_in",
        "test_temperature_c",
        "reference_temperature_c",
        "temperature_correction_applied",
        "rr_n_per_kn",
        "rr_value_source_note",
        "sae_a",
        "sae_b",
        "sae_c",
        "sae_alpha",
        "sae_beta",
        "pressure_unit",
        "load_unit",
        "speed_unit",
        "force_unit",
        "iso_rrc_n_per_kn",
        "iso_test_pressure_kpa",
        "iso_test_load_n",
        "iso_test_speed_kph",
        "iso_rolling_resistance_force_n",
        "iso_condition_notes",
        "calculation_mode",
        "smerf",
    }
)
TIRE_DERIVED = frozenset({"iso_corrected_rrc_n_per_kn", "effective_circumference_override_mm"})

COMPONENT_EDITABLE = frozenset(
    {
        "domain",
        "component_code",
        "component_name",
        "source_name",
        "source_record_id",
        "source_reference",
        "hardware_reference",
        "component_type",
        "component_position",
        "driveline_architecture",
        "physical_boundary",
        "configuration_from",
        "configuration_to",
        "test_condition_type",
        "test_method",
        "net_bridge_eligible",
        "equivalent_A_N",
        "equivalent_B_N_per_kph",
        "equivalent_C_N_per_kph2",
        "loss_pct",
        "residual_torque_front_nm",
        "residual_torque_rear_nm",
        "wheel_radius_m",
        "notes",
    }
)


def field_policy_for(entity_type: EntityType | str, record_origin: str | None) -> EntityFieldPolicy:
    entity = normalize_entity_type(entity_type)
    origin = normalize_record_origin(entity, record_origin)
    immutable = COMMON_IMMUTABLE | frozenset({"record_origin"})

    if entity is EntityType.VDE:
        advanced = VDE_PHYSICAL
        derived = VDE_DERIVED
        if origin == "VDE_SETUP":
            derived = derived | VDE_PHYSICAL
            advanced = frozenset()
        elif origin == "IMPORTED_REFERENCE":
            advanced = VDE_PHYSICAL | VDE_DERIVED
            derived = frozenset()
        return EntityFieldPolicy(VDE_METADATA, advanced, derived, immutable)

    if entity is EntityType.FUEL_CONSUMPTION:
        if origin in {"HOMOLOGATED", "MEASURED"}:
            return EntityFieldPolicy(FUEL_METADATA | FUEL_INPUTS, FUEL_DERIVED, frozenset(), immutable)
        if origin in {"ESTIMATED", "POWERTRAIN_L0"}:
            return EntityFieldPolicy(FUEL_METADATA | FUEL_INPUTS, frozenset(), FUEL_DERIVED, immutable)
        return EntityFieldPolicy(FUEL_METADATA | FUEL_INPUTS, FUEL_DERIVED, frozenset(), immutable)

    if entity is EntityType.TIRE:
        return EntityFieldPolicy(
            TIRE_EDITABLE,
            frozenset(),
            TIRE_DERIVED,
            immutable | frozenset({"is_active"}),
        )

    return EntityFieldPolicy(COMPONENT_EDITABLE, frozenset(), frozenset(), immutable)


def field_access_for(entity_type: EntityType | str, record_origin: str | None, field_name: str) -> FieldAccess:
    policy = field_policy_for(entity_type, record_origin)
    field = str(field_name or "").strip()
    if field in policy.editable:
        return FieldAccess.EDITABLE
    if field in policy.advanced_correction:
        return FieldAccess.ADVANCED_CORRECTION
    if field in policy.derived:
        return FieldAccess.DERIVED
    if field in policy.immutable:
        return FieldAccess.IMMUTABLE
    return FieldAccess.UNKNOWN
