from __future__ import annotations

from copy import deepcopy

import streamlit as st

from src.vde_core.component_repositories import component_repository_signature, load_component_repository
from src.vde_core.db import current_db_path
from src.vde_core.repositories import fetch_vde_all_rows, fetch_vde_by_id
from src.vde_core.tire_roadload_service import get_tire_by_code, get_tire_by_id, search_tire_roadload
from src.vde_core.vde_request_contract import is_blank


LOOKUP_PROPOSAL_TYPES = {
    "tire": {"TIRE_DB_LOOKUP"},
    "transmission": {"TRANS_METADATA_ONLY"},
    "brake": {"BRAKE_METADATA_ONLY"},
    "axle_hubs": {"AXLE_HUB_METADATA_ONLY"},
    "parasitic": {"PARASITIC_METADATA_ONLY"},
}
LOOKUP_SOURCE_LABELS = {
    "default": {"component": "Component DB", "vde": "VDE DB"},
    "tire": {"component": "Tire Database", "vde": "Existing VDE"},
}
TIRE_LOOKUP_BROWSE_LIMIT = 25


def _first_present(item: dict, *keys: str):
    for key in keys:
        value = item.get(key)
        if not is_blank(value):
            return value
    return None


def active_domain_has_lookup_requests(state: dict, domain: str) -> bool:
    for proposal in list(state.get("proposals") or []):
        payload = dict(dict(proposal.get("domains") or {}).get(domain) or {})
        if str(payload.get("proposal_type") or "") in LOOKUP_PROPOSAL_TYPES.get(domain, set()):
            return True
    return False


def lookup_source_options(domain: str) -> list[str]:
    labels = LOOKUP_SOURCE_LABELS["tire"] if str(domain or "").strip() == "tire" else LOOKUP_SOURCE_LABELS["default"]
    return [labels["component"], labels["vde"]]


def default_lookup_source(domain: str) -> str:
    return lookup_source_options(domain)[0]


def is_component_lookup_source(domain: str, source_kind: str) -> bool:
    source = str(source_kind or "").strip()
    labels = lookup_source_options(domain)
    return source in {labels[0], LOOKUP_SOURCE_LABELS["default"]["component"], LOOKUP_SOURCE_LABELS["tire"]["component"]}


def is_vde_lookup_source(domain: str, source_kind: str) -> bool:
    return not is_component_lookup_source(domain, source_kind)


def component_lookup_rows(domain: str, query: str = "", limit: int | None = None) -> list[dict]:
    domain_key = str(domain or "").strip()
    revision = "" if domain_key == "tire" else component_repository_signature(domain_key)
    return _component_lookup_rows_cached(
        domain_key,
        query=query,
        limit=limit,
        db_path_signature=str(current_db_path()),
        repository_signature=revision,
    )


@st.cache_data(show_spinner=False)
def _component_lookup_rows_cached(
    domain: str,
    query: str = "",
    limit: int | None = None,
    db_path_signature: str = "",
    repository_signature: str = "",
) -> list[dict]:
    del db_path_signature, repository_signature
    domain_key = str(domain or "").strip()
    if domain_key == "tire":
        needle = str(query or "").strip().lower()
        rows = search_tire_roadload()
        results = []
        for row in rows:
            item = dict(row or {})
            haystack = " ".join(
                str(item.get(key) or "").lower()
                for key in (
                    "id",
                    "tire_test_code",
                    "manufacturer",
                    "model",
                    "size_code",
                    "notes",
                    "rr_n_per_kn",
                    "iso_rrc_n_per_kn",
                    "test_pressure_value",
                    "test_load_value",
                    "test_mileage_km",
                    "rr_value_source_note",
                )
            )
            if needle and needle not in haystack:
                continue
            results.append(
                {
                    "lookup_id": str(item.get("id") or item.get("tire_test_code") or ""),
                    "Tire ID": item.get("id"),
                    "Tire VDE ID": item.get("source_vde_id") or item.get("vde_id") or item.get("tire_source_vde_id"),
                    "Tire code": item.get("tire_test_code") or item.get("tire_code"),
                    "RRC": item.get("rr_n_per_kn") or item.get("iso_rrc_n_per_kn"),
                    "SMERF": item.get("smerf"),
                    "Reference pressure": _raw_pressure_psi(item),
                    "Test load": _raw_test_load_kg(item),
                    "Mileage": item.get("test_mileage_km"),
                    "alpha": item.get("sae_alpha"),
                    "beta": item.get("sae_beta"),
                    "a": item.get("sae_a"),
                    "b": item.get("sae_b"),
                    "c": item.get("sae_c"),
                    "Status": item.get("rr_quality") or item.get("status") or "active",
                    "Source": item.get("rr_source") or item.get("test_source") or item.get("rr_value_source_note"),
                    "Notes": item.get("notes") or item.get("model") or "",
                    "_raw": item,
                }
            )
        if limit is not None and limit > 0:
            return results[: int(limit)]
        return results

    repo = load_component_repository(domain_key)
    rows = repo.search(query)
    results = []
    for row in rows:
        item = dict(row or {})
        results.append(
            {
                "lookup_id": item.get("component_id"),
                "ID": item.get("component_id"),
                "Code / Name": item.get("component_name"),
                "A": _first_present(item, "trans_A", "brake_A", "axle_hubs_A", "parasitic_A"),
                "B": _first_present(item, "trans_B", "brake_B", "axle_hubs_B", "parasitic_B"),
                "C": _first_present(item, "trans_C", "brake_C", "axle_hubs_C", "parasitic_C"),
                "Status": item.get("status"),
                "Source": item.get("source"),
                "Component type": item.get("component_type"),
                "Position": item.get("component_position"),
                "Test condition": item.get("test_condition_type"),
                "Driveline architecture": item.get("driveline_architecture"),
                "Physical boundary": item.get("physical_boundary"),
                "Configuration from": item.get("configuration_from"),
                "Configuration to": item.get("configuration_to"),
                "Test method": item.get("test_method"),
                "NET bridge eligibility": item.get("net_bridge_eligible"),
                "Description": item.get("notes"),
                "_raw": item,
            }
        )
    return results


def vde_lookup_rows(domain: str, query: str = "", limit: int | None = None) -> list[dict]:
    return _vde_lookup_rows_cached(domain, query=query, limit=limit, db_path_signature=str(current_db_path()))


@st.cache_data(show_spinner=False)
def _vde_lookup_rows_cached(
    domain: str,
    query: str = "",
    limit: int | None = None,
    db_path_signature: str = "",
) -> list[dict]:
    del db_path_signature
    needle = str(query or "").strip().lower()
    rows = []
    for row in fetch_vde_all_rows():
        item = dict(row or {})
        haystack = " ".join(
            str(item.get(key) or "").lower()
            for key in ("id", "make", "model", "notes", "year", "tire_code", "trailer_code", "rrc_N_per_kN", "front_pressure_psi", "rear_pressure_psi")
        )
        if needle and needle not in haystack:
            continue
        if str(domain or "").strip() == "tire":
            rows.append(
                {
                    "lookup_id": item.get("id"),
                    "VDE ID": item.get("id"),
                    "Make": item.get("make"),
                    "Model": item.get("model"),
                    "Tire code": item.get("tire_code"),
                    "RRC": item.get("rrc_N_per_kN"),
                    "SMERF": item.get("smerf"),
                    "Reference pressure": item.get("front_pressure_psi") if not is_blank(item.get("front_pressure_psi")) else item.get("rear_pressure_psi"),
                    "Description": item.get("notes"),
                    "_raw": item,
                }
            )
            continue
        abc = _vde_domain_abc(str(domain or ""), item)
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
                "A": abc["A"],
                "B": abc["B"],
                "C": abc["C"],
                "_raw": item,
            }
        )
    if limit is not None and limit > 0:
        return rows[: int(limit)]
    return rows


component_lookup_rows.clear = _component_lookup_rows_cached.clear  # type: ignore[attr-defined]
vde_lookup_rows.clear = _vde_lookup_rows_cached.clear  # type: ignore[attr-defined]


def apply_lookup_to_inputs(domain: str, source_kind: str, selected_row: dict | None) -> dict:
    item = deepcopy(dict(selected_row or {}))
    raw = deepcopy(dict(item.get("_raw") or item))
    if is_component_lookup_source(domain, source_kind):
        return _component_lookup_inputs(domain, raw)
    return _vde_lookup_inputs(domain, raw)


def lookup_row_by_id(domain: str, source_kind: str, lookup_id) -> dict | None:
    rows = component_lookup_rows(domain, "") if is_component_lookup_source(domain, source_kind) else vde_lookup_rows(domain, "")
    target = str(lookup_id or "")
    for row in rows:
        if str(dict(row).get("lookup_id") or "") == target:
            return deepcopy(dict(row))
    return None


def lookup_empty_message(
    domain: str,
    source_kind: str,
    query: str,
    results: list[dict] | None = None,
    *,
    filters_active: bool = False,
) -> str:
    if list(results or []):
        return ""
    if str(domain or "").strip() == "tire" and is_component_lookup_source(domain, source_kind):
        return "No Tire Database records available." if (not filters_active and is_blank(query)) else "No matching Tire Database records."
    return "No matching records."


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
            "tire_snapshot": deepcopy(raw),
        }
    if domain == "transmission":
        return {
            "transmission_component_db_id": raw.get("component_id"),
            "transmission_vde_db_id": "",
            "trans_A_coef_N": raw.get("trans_A"),
            "trans_B_coef_Npkph": raw.get("trans_B"),
            "trans_C_coef_Npkph2": raw.get("trans_C"),
            "transmission_loss_pct": raw.get("loss_pct"),
        }
    if domain == "brake":
        return {
            "brake_component_db_id": raw.get("component_id"),
            "brake_vde_db_id": "",
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
            "axle_hubs_vde_db_id": "",
            "axle_hub_A": raw.get("axle_hubs_A"),
            "axle_hub_B": raw.get("axle_hubs_B"),
            "axle_hub_C": raw.get("axle_hubs_C"),
        }
    if domain == "parasitic":
        return {
            "parasitic_component_db_id": raw.get("component_id"),
            "parasitic_vde_db_id": "",
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
        abc = _vde_domain_abc(domain, raw)
        return {
            "transmission_vde_db_id": raw.get("id"),
            "transmission_component_db_id": "",
            "trans_A_coef_N": abc["A"],
            "trans_B_coef_Npkph": abc["B"],
            "trans_C_coef_Npkph2": abc["C"],
        }
    if domain == "brake":
        abc = _vde_domain_abc(domain, raw)
        return {
            "brake_vde_db_id": raw.get("id"),
            "brake_component_db_id": "",
            "brake_A_coef_N": abc["A"],
            "brake_B_Npkph": abc["B"],
            "brake_C_coef_Npkph2": abc["C"],
        }
    if domain == "axle_hubs":
        abc = _vde_domain_abc(domain, raw)
        return {
            "axle_hubs_vde_db_id": raw.get("id"),
            "axle_hubs_component_db_id": "",
            "axle_hub_A": abc["A"],
            "axle_hub_B": abc["B"],
            "axle_hub_C": abc["C"],
        }
    if domain == "parasitic":
        abc = _vde_domain_abc(domain, raw)
        return {
            "parasitic_vde_db_id": raw.get("id"),
            "parasitic_component_db_id": "",
            "parasitic_A_coef_N": abc["A"],
            "parasitic_B_Npkph": abc["B"],
            "parasitic_C_coef_Npkph2": abc["C"],
        }
    return {}


def _vde_domain_abc(domain: str, row: dict) -> dict:
    """Read only the component split belonging to the active lookup domain."""
    mappings = {
        "transmission": ("trans_A_coef_N", "trans_B_coef_Npkph", "trans_C_coef_Npkph2"),
        "brake": ("brake_A_coef_N", "brake_B_coef_Npkph", "brake_C_coef_Npkph2"),
        "axle_hubs": ("axle_hub_A", "axle_hub_B", "axle_hub_C"),
        "parasitic": ("parasitic_A_coef_N", "parasitic_B_coef_Npkph", "parasitic_C_coef_Npkph2"),
    }
    keys = mappings.get(str(domain or ""), (None, None, None))
    return {"A": row.get(keys[0]) if keys[0] else None, "B": row.get(keys[1]) if keys[1] else None, "C": row.get(keys[2]) if keys[2] else None}


def _raw_pressure_psi(raw: dict):
    for key in ("test_pressure_value", "pressure_psi", "front_pressure_psi"):
        value = raw.get(key)
        if not is_blank(value):
            return value
    return None


def _raw_test_load_kg(raw: dict):
    for key in ("test_load_value", "test_mass_kg"):
        value = raw.get(key)
        if not is_blank(value):
            return value
    return None
