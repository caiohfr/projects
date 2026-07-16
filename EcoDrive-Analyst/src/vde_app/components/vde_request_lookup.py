from __future__ import annotations

from copy import deepcopy

import streamlit as st

from src.vde_core.component_repositories import load_mock_component_repository
from src.vde_core.repositories import fetch_vde_all_rows, fetch_vde_by_id
from src.vde_core.tire_roadload_service import get_tire_by_code, get_tire_by_id, search_tire_roadload
from src.vde_core.vde_request_contract import is_blank


LOOKUP_PROPOSAL_TYPES = {
    "tire": {"TIRE_DB_LOOKUP", "TIRE_METADATA_ONLY"},
    "transmission": {"TRANS_METADATA_ONLY"},
    "brake": {"BRAKE_METADATA_ONLY"},
    "axle_hubs": {"AXLE_HUB_METADATA_ONLY"},
    "parasitic": {"PARASITIC_METADATA_ONLY"},
}


def active_domain_has_lookup_requests(state: dict, domain: str) -> bool:
    for proposal in list(state.get("proposals") or []):
        payload = dict(dict(proposal.get("domains") or {}).get(domain) or {})
        if str(payload.get("proposal_type") or "") in LOOKUP_PROPOSAL_TYPES.get(domain, set()):
            return True
    return False


@st.cache_data(show_spinner=False)
def component_lookup_rows(domain: str, query: str = "") -> list[dict]:
    domain_key = str(domain or "").strip()
    if domain_key == "tire":
        needle = str(query or "").strip().lower()
        rows = search_tire_roadload()
        results = []
        for row in rows:
            item = dict(row or {})
            haystack = " ".join(
                str(item.get(key) or "").lower()
                for key in ("id", "tire_test_code", "manufacturer", "model", "size_code", "notes")
            )
            if needle and needle not in haystack:
                continue
            results.append(
                {
                    "lookup_id": str(item.get("id") or item.get("tire_test_code") or ""),
                    "ID": item.get("id"),
                    "Code / Name": item.get("tire_test_code") or item.get("tire_size") or item.get("description"),
                    "RRC": item.get("rr_n_per_kn") or item.get("iso_rrc_n_per_kn"),
                    "Status": item.get("rr_quality") or item.get("status") or "active",
                    "Source": item.get("rr_source") or item.get("standard_family") or "tire_service",
                    "Description": item.get("tire_line") or item.get("notes") or "",
                    "_raw": item,
                }
            )
        return results

    repo = load_mock_component_repository(domain_key)
    rows = repo.search(query)
    results = []
    for row in rows:
        item = dict(row or {})
        results.append(
            {
                "lookup_id": item.get("component_id"),
                "ID": item.get("component_id"),
                "Code / Name": item.get("component_name"),
                "A": item.get("trans_A") or item.get("brake_A") or item.get("axle_hubs_A") or item.get("parasitic_A"),
                "B": item.get("trans_B") or item.get("brake_B") or item.get("axle_hubs_B") or item.get("parasitic_B"),
                "C": item.get("trans_C") or item.get("brake_C") or item.get("axle_hubs_C") or item.get("parasitic_C"),
                "Status": item.get("status"),
                "Source": item.get("source"),
                "Description": item.get("notes"),
                "_raw": item,
            }
        )
    return results


@st.cache_data(show_spinner=False)
def vde_lookup_rows(domain: str, query: str = "") -> list[dict]:
    needle = str(query or "").strip().lower()
    rows = []
    for row in fetch_vde_all_rows():
        item = dict(row or {})
        haystack = " ".join(
            str(item.get(key) or "").lower()
            for key in ("id", "make", "model", "notes", "year", "tire_code", "trailer_code")
        )
        if needle and needle not in haystack:
            continue
        rows.append(
            {
                "lookup_id": item.get("id"),
                "VDE ID": item.get("id"),
                "Make": item.get("make"),
                "Model": item.get("model"),
                "Year": item.get("year"),
                "Notes": item.get("notes"),
                "CdA": item.get("cda_m2"),
                "Mass": item.get("mass_kg"),
                "RRC": item.get("rrc_N_per_kN"),
                "A": item.get("trans_A_coef_N") or item.get("brake_A_coef_N") or item.get("axle_hub_A") or item.get("parasitic_A_coef_N"),
                "B": item.get("trans_B_coef_Npkph") or item.get("brake_B_coef_Npkph") or item.get("axle_hub_B") or item.get("parasitic_B_coef_Npkph"),
                "C": item.get("trans_C_coef_Npkph2") or item.get("brake_C_coef_Npkph2") or item.get("axle_hub_C") or item.get("parasitic_C_coef_Npkph2"),
                "_raw": item,
            }
        )
    return rows


def apply_lookup_to_inputs(domain: str, source_kind: str, selected_row: dict | None) -> dict:
    item = deepcopy(dict(selected_row or {}))
    raw = deepcopy(dict(item.get("_raw") or item))
    if source_kind == "Component DB":
        return _component_lookup_inputs(domain, raw)
    return _vde_lookup_inputs(domain, raw)


def lookup_row_by_id(domain: str, source_kind: str, lookup_id) -> dict | None:
    rows = component_lookup_rows(domain, "") if source_kind == "Component DB" else vde_lookup_rows(domain, "")
    target = str(lookup_id or "")
    for row in rows:
        if str(dict(row).get("lookup_id") or "") == target:
            return deepcopy(dict(row))
    return None


def load_vde_lookup_source(vde_id: int) -> dict:
    return deepcopy(fetch_vde_by_id(int(vde_id)))


def _component_lookup_inputs(domain: str, raw: dict) -> dict:
    if domain == "tire":
        tire_id = raw.get("id")
        tire_code = raw.get("tire_test_code") or raw.get("tire_code")
        return {
            "tire_db_id": tire_id,
            "tire_code": tire_code,
            "rrc_N_per_kN": raw.get("rr_n_per_kn") or raw.get("iso_rrc_n_per_kn"),
            "front_pressure_psi": _raw_pressure_psi(raw),
            "rear_pressure_psi": _raw_pressure_psi(raw),
        }
    if domain == "transmission":
        return {
            "transmission_component_db_id": raw.get("component_id"),
            "trans_A_coef_N": raw.get("trans_A"),
            "trans_B_coef_Npkph": raw.get("trans_B"),
            "trans_C_coef_Npkph2": raw.get("trans_C"),
            "transmission_loss_pct": raw.get("loss_pct"),
        }
    if domain == "brake":
        return {
            "brake_component_db_id": raw.get("component_id"),
            "brake_A_coef_N": raw.get("brake_A"),
            "brake_B_Npkph": raw.get("brake_B"),
            "brake_C_coef_Npkph2": raw.get("brake_C"),
            "residual_torque_front_Nm": raw.get("residual_torque_front_nm"),
            "residual_torque_rear_Nm": raw.get("residual_torque_rear_nm"),
            "wheel_radius_m": raw.get("wheel_radius_m"),
        }
    if domain == "axle_hubs":
        return {
            "axle_hubs_component_db_id": raw.get("component_id"),
            "axle_hub_A": raw.get("axle_hubs_A"),
            "axle_hub_B": raw.get("axle_hubs_B"),
            "axle_hub_C": raw.get("axle_hubs_C"),
        }
    if domain == "parasitic":
        return {
            "parasitic_component_db_id": raw.get("component_id"),
            "parasitic_A_coef_N": raw.get("parasitic_A"),
            "parasitic_B_Npkph": raw.get("parasitic_B"),
            "parasitic_C_coef_Npkph2": raw.get("parasitic_C"),
        }
    return {}


def _vde_lookup_inputs(domain: str, raw: dict) -> dict:
    if domain == "mass":
        return {
            "mass_kg": raw.get("mass_kg"),
            "test_mass_kg": raw.get("test_mass_kg"),
            "inertia_class": raw.get("inertia_class"),
            "payload_kg": raw.get("payload_kg"),
            "options_kg": raw.get("options_kg"),
            "gvwr_kg": raw.get("gvwr_kg"),
            "gcwr_kg": raw.get("gcwr_kg"),
            "trailer_mass_kg": raw.get("trailer_mass_kg"),
            "trailer_code": raw.get("trailer_code"),
            "trailer_A": raw.get("trailer_A_coef_N") or raw.get("trailer_A"),
            "trailer_B": raw.get("trailer_B_coef_Npkph") or raw.get("trailer_B"),
            "trailer_C": raw.get("trailer_C_coef_Npkph2") or raw.get("trailer_C"),
        }
    if domain == "aero":
        return {
            "cda_m2": raw.get("cda_m2"),
            "frontal_area_m2": raw.get("frontal_area_m2"),
            "aero_source_vde_id": raw.get("id"),
        }
    if domain == "tire":
        return {
            "tire_db_id": raw.get("tire_db_id"),
            "tire_code": raw.get("tire_code"),
            "rrc_N_per_kN": raw.get("rrc_N_per_kN"),
            "front_pressure_psi": raw.get("front_pressure_psi"),
            "rear_pressure_psi": raw.get("rear_pressure_psi"),
            "tire_source_vde_id": raw.get("id"),
        }
    if domain == "transmission":
        return {
            "transmission_vde_db_id": raw.get("id"),
            "trans_A_coef_N": raw.get("trans_A_coef_N"),
            "trans_B_coef_Npkph": raw.get("trans_B_coef_Npkph"),
            "trans_C_coef_Npkph2": raw.get("trans_C_coef_Npkph2"),
        }
    if domain == "brake":
        return {
            "brake_vde_db_id": raw.get("id"),
            "brake_A_coef_N": raw.get("brake_A_coef_N"),
            "brake_B_Npkph": raw.get("brake_B_coef_Npkph") or raw.get("brake_B_Npkph"),
            "brake_C_coef_Npkph2": raw.get("brake_C_coef_Npkph2"),
        }
    if domain == "axle_hubs":
        return {
            "axle_hubs_vde_db_id": raw.get("id"),
            "axle_hub_A": raw.get("axle_hub_A"),
            "axle_hub_B": raw.get("axle_hub_B"),
            "axle_hub_C": raw.get("axle_hub_C"),
        }
    if domain == "parasitic":
        return {
            "parasitic_vde_db_id": raw.get("id"),
            "parasitic_A_coef_N": raw.get("parasitic_A_coef_N"),
            "parasitic_B_Npkph": raw.get("parasitic_B_coef_Npkph") or raw.get("parasitic_B_Npkph"),
            "parasitic_C_coef_Npkph2": raw.get("parasitic_C_coef_Npkph2"),
        }
    return {}


def _raw_pressure_psi(raw: dict):
    for key in ("test_pressure_value", "pressure_psi", "front_pressure_psi"):
        value = raw.get(key)
        if not is_blank(value):
            return value
    return None
