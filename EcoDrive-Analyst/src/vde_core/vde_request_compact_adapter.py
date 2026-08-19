from __future__ import annotations

from copy import deepcopy

from src.vde_core.vde_request_adapter import build_v21_workbook_state_from_request_draft
from src.vde_core.vde_request_compact_state import build_v22_canonical_request_draft, normalize_v22_state
from src.vde_core.vde_request_preview import (
    build_component_action_rows,
    build_proposal_preview_model,
    build_request_audit_rows,
    build_request_comparison_rows,
    build_request_resolution_fingerprint,
    build_validation_summary,
)
from src.vde_core.vde_request_resolver import resolve_vde_request


def build_v22_workbook_state(state: dict) -> dict:
    normalized = normalize_v22_state(state)
    draft = build_v22_canonical_request_draft(normalized)
    return build_v21_workbook_state_from_request_draft(draft, {"rows": []})


def build_v22_preview_bundle(state: dict, *, baseline_context: dict | None = None, component_repositories=None) -> dict:
    normalized = normalize_v22_state(state)
    draft = build_v22_canonical_request_draft(normalized)
    workbook_state = build_v21_workbook_state_from_request_draft(draft, {"rows": []})
    fingerprint = build_request_resolution_fingerprint(workbook_state, baseline_context=baseline_context)
    resolution_result = resolve_vde_request(workbook_state, baseline_context=baseline_context, component_repositories=component_repositories)
    return {
        "draft": draft,
        "workbook_state": workbook_state,
        "fingerprint": fingerprint,
        "resolution_result": resolution_result,
        "comparison_rows": build_request_comparison_rows(resolution_result),
        "validation_summary": build_validation_summary(resolution_result),
        "audit_rows": build_request_audit_rows(resolution_result),
        "proposal_models": [build_proposal_preview_model(item) for item in list(resolution_result.get("proposal_results") or [])],
        "component_action_rows": {
            str(item.get("proposal_id") or ""): build_component_action_rows(item)
            for item in list(resolution_result.get("proposal_results") or [])
        },
    }


def compact_baseline_context(state: dict) -> dict:
    normalized = normalize_v22_state(state)
    effective = deepcopy(dict(dict(normalized.get("baseline") or {}).get("effective") or {}))
    return {
        "baseline_source_type": effective.get("baseline_source_type"),
        "selected_baseline_vde_id": effective.get("selected_baseline_vde_id"),
        "legislation": effective.get("legislation"),
        "category": effective.get("category"),
        "electrification": effective.get("electrification"),
        "transmission_type": effective.get("transmission_type"),
        "drive_type": effective.get("drive_type"),
        "fuel_type": effective.get("fuel_type"),
        "make": effective.get("make"),
        "model": effective.get("model"),
        "year": effective.get("year"),
        "cycle_name": effective.get("cycle_name"),
        "mass_kg": effective.get("mass_kg"),
        "test_mass_kg": effective.get("test_mass_kg"),
        "payload_kg": effective.get("payload_kg"),
        "weight_dist_fr_pct": effective.get("weight_dist_fr_pct"),
        "inertia_class": effective.get("inertia_class"),
        "CdA": effective.get("cda_m2"),
        "frontal_area_m2": effective.get("frontal_area_m2"),
        "front_tire_id": effective.get("front_tire_id"),
        "rear_tire_id": effective.get("rear_tire_id"),
        "tire_db_id": effective.get("tire_db_id"),
        "tire_code": effective.get("tire_code"),
        "rrc_N_per_kN": effective.get("rrc_N_per_kN"),
        "tire_load_mass_basis": effective.get("tire_load_mass_basis"),
        "tire_A_final": effective.get("tire_A_final"),
        "tire_B_final": effective.get("tire_B_final"),
        "tire_C_final": effective.get("tire_C_final"),
        "tire_calc_source": effective.get("tire_calc_source"),
        "smerf": effective.get("smerf"),
        "trans_A_coef_N": effective.get("trans_A_coef_N"),
        "trans_B_coef_Npkph": effective.get("trans_B_coef_Npkph"),
        "trans_C_coef_Npkph2": effective.get("trans_C_coef_Npkph2"),
        "brake_A": effective.get("brake_A_coef_N"),
        "brake_B": effective.get("brake_B_Npkph"),
        "brake_C": effective.get("brake_C_coef_Npkph2"),
        "axle_hub_A": effective.get("axle_hub_A"),
        "axle_hub_B": effective.get("axle_hub_B"),
        "axle_hub_C": effective.get("axle_hub_C"),
        "parasitic_A": effective.get("parasitic_A_coef_N"),
        "parasitic_B": effective.get("parasitic_B_Npkph"),
        "parasitic_C": effective.get("parasitic_C_coef_Npkph2"),
        "A": effective.get("A"),
        "B": effective.get("B"),
        "C": effective.get("C"),
    }
