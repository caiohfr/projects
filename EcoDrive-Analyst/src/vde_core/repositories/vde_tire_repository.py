from __future__ import annotations

from src.vde_core.db import fetchone, update_vde


_TIRE_VDE_FIELDS = (
    "front_tire_id",
    "rear_tire_id",
    "front_pressure_psi",
    "rear_pressure_psi",
    "weight_dist_fr_pct",
    "tire_improvement_pct",
    "tire_load_mass_basis",
    "tire_load_mass_used_kg",
    "tire_A_final",
    "tire_B_final",
    "tire_C_final",
    "rrc_N_per_kN",
    "tire_calc_source",
    "tire_calc_notes",
)


def get_vde_tire_application(vde_id: int) -> dict:
    row = fetchone(
        "SELECT id, front_tire_id, rear_tire_id, front_pressure_psi, rear_pressure_psi, "
        "weight_dist_fr_pct, tire_improvement_pct, tire_load_mass_basis, "
        "tire_load_mass_used_kg, tire_A_final, tire_B_final, tire_C_final, "
        "rrc_N_per_kN, tire_calc_source, tire_calc_notes, mass_kg, test_mass_kg, inertia_class "
        "FROM vde_db WHERE id=?;",
        (int(vde_id),),
    )
    return row or {}


def update_vde_tire_application(vde_id: int, payload: dict) -> None:
    updates = {k: payload[k] for k in _TIRE_VDE_FIELDS if k in payload}
    if not updates:
        return
    update_vde(int(vde_id), updates)
