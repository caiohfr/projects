from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd
import streamlit as st

from src.vde_core.database_management_contract import ChangeCommand, EntityType, ORIGINS_BY_ENTITY
from src.vde_core.database_management_impact_service import (
    apply_change_with_impact,
    apply_vde_dependency_resolution,
    discover_catalog_usage,
    discover_vde_dependencies,
    preview_dependency_impact,
)
from src.vde_core.database_management_policy import FieldAccess, field_access_for
from src.vde_core.database_management_service import (
    apply_change,
    browse_records,
    duplicate_payload_for,
    get_record,
    preview_change,
    simple_dependencies,
)


STAGE_KEY = "database_management_staged_commands"

_ENTITY_LABELS = {
    EntityType.VDE: "VDE",
    EntityType.FUEL_CONSUMPTION: "Fuel Consumption",
    EntityType.TIRE: "Tires",
    EntityType.COMPONENT: "Components",
}
_GRID_FIELDS = {
    EntityType.VDE: ("id", "legislation", "category", "make", "model", "year", "source_name", "record_origin", "record_status", "updated_at"),
    EntityType.FUEL_CONSUMPTION: ("id", "vde_id", "electrification", "fuel_type", "method_note", "review_status", "record_origin", "record_status", "updated_at"),
    EntityType.TIRE: ("id", "tire_test_code", "manufacturer", "model", "size_code", "standard_family", "rr_n_per_kn", "source_name", "record_origin", "is_active", "updated_at"),
    EntityType.COMPONENT: ("id", "component_code", "component_name", "source_name", "equivalent_A_N", "equivalent_B_N_per_kph", "equivalent_C_N_per_kph2", "loss_pct", "record_origin", "record_status", "updated_at"),
}
_DETAIL_FIELDS = {
    EntityType.VDE: (
        "legislation", "category", "make", "model", "year", "notes", "source_name", "source_record_id",
        "engine_type", "engine_model", "transmission_type", "transmission_model", "drive_type",
        "mass_kg", "test_mass_kg", "inertia_class", "cda_m2", "weight_dist_fr_pct", "payload_kg",
        "coast_A_N", "coast_B_N_per_kph", "coast_C_N_per_kph2",
    ),
    EntityType.FUEL_CONSUMPTION: (
        "vde_id", "electrification", "fuel_type", "method_note", "source_name", "source_record_id",
        "eta_pt_est", "bev_eff_drive", "utility_factor_pct", "engine_max_power_kw", "gear_count",
        "final_drive_ratio", "battery_capacity_kwh", "ambient_temp_c", "ac_on", "tire_front_psi", "tire_rear_psi",
        "scenario_payload_kg", "energy_Wh_per_km", "fuel_l_per_100km", "gco2_per_km", "review_status",
    ),
    EntityType.TIRE: (
        "tire_test_code", "manufacturer", "model", "size_code", "load_index", "speed_rating", "notes",
        "source_name", "source_record_id", "standard_family", "standard_version", "test_method", "test_source",
        "test_date", "test_mileage_km", "rr_n_per_kn", "rr_value_source_note", "sae_a", "sae_b", "sae_c",
        "sae_alpha", "sae_beta", "iso_rrc_n_per_kn", "iso_test_pressure_kpa", "iso_test_load_n",
        "iso_test_speed_kph", "calculation_mode", "smerf", "iso_corrected_rrc_n_per_kn",
    ),
    EntityType.COMPONENT: (
        "domain", "component_code", "component_name", "source_name", "source_record_id", "source_reference",
        "hardware_reference", "component_type", "component_position", "driveline_architecture", "physical_boundary",
        "configuration_from", "configuration_to", "test_condition_type", "test_method", "net_bridge_eligible",
        "equivalent_A_N", "equivalent_B_N_per_kph", "equivalent_C_N_per_kph2", "loss_pct",
        "residual_torque_front_nm", "residual_torque_rear_nm", "wheel_radius_m", "notes",
    ),
}
_CREATE_DEFAULTS = {
    EntityType.VDE: {"legislation": "EPA", "category": "", "make": "", "model": "", "mass_kg": None, "notes": ""},
    EntityType.FUEL_CONSUMPTION: {"vde_id": None, "electrification": "ICE", "fuel_type": "", "method_note": ""},
    EntityType.TIRE: {"tire_test_code": "", "manufacturer": "", "model": "", "standard_family": "ISO", "rr_n_per_kn": None, "notes": ""},
    EntityType.COMPONENT: {"component_code": "", "component_name": "", "source_name": "", "equivalent_A_N": None, "equivalent_B_N_per_kph": None, "equivalent_C_N_per_kph2": None, "loss_pct": None, "notes": ""},
}


def render_database_management() -> None:
    st.title("Database Management")
    st.caption("Browse, stage, validate, review, and confirm changes. Nothing is saved automatically.")
    _ensure_stage()
    tabs = st.tabs([_ENTITY_LABELS[entity] for entity in EntityType])
    for tab, entity in zip(tabs, EntityType):
        with tab:
            _render_entity_tab(entity)
    st.divider()
    _render_review_changes()


def _render_entity_tab(entity: EntityType) -> None:
    state_prefix = f"database_management_{entity.value.lower()}"
    filter_cols = st.columns((3, 1, 1))
    with filter_cols[0]:
        query = st.text_input("Search", key=f"{state_prefix}_query", placeholder="Search visible identity and source fields")
    with filter_cols[1]:
        include_archived = st.toggle("Include archived", key=f"{state_prefix}_archived")
    component_domain = None
    with filter_cols[2]:
        if entity is EntityType.COMPONENT:
            component_domain = st.selectbox(
                "Component domain",
                ("transmission", "brake", "axle_hubs", "parasitic"),
                key=f"{state_prefix}_domain",
            )
        else:
            st.caption("Changes stay staged until review.")
    rows = browse_records(entity, query=query, include_archived=include_archived, component_domain=component_domain)
    st.caption(f"{len(rows)} record(s) shown")
    _render_grid(entity, rows, component_domain=component_domain)
    _render_add_form(entity, component_domain=component_domain)
    _render_selected_record(entity, rows, component_domain=component_domain)


def _render_grid(entity: EntityType, rows: list[dict], *, component_domain: str | None) -> None:
    fields = _GRID_FIELDS[entity]
    grid_rows = [{field: row.get(field) for field in fields} for row in rows]
    dataframe = pd.DataFrame(grid_rows, columns=fields)
    editable_columns = _grid_editable_columns(entity, rows, fields)
    edited = st.data_editor(
        dataframe,
        key=f"database_management_grid_{entity.value.lower()}",
        disabled=[field for field in fields if field not in editable_columns],
        hide_index=True,
        width="stretch",
        num_rows="fixed",
    )
    if st.button("Stage grid edits", key=f"database_management_stage_grid_{entity.value.lower()}", disabled=not rows):
        changed = 0
        for original, edited_row in zip(rows, edited.to_dict(orient="records")):
            payload = {
                field: edited_row.get(field)
                for field in editable_columns
                if not _same_value(original.get(field), edited_row.get(field))
            }
            if payload:
                _stage_command(
                    _command_payload(entity, "UPDATE", original, payload=payload, component_domain=component_domain)
                )
                changed += 1
        if changed:
            st.success(f"{changed} row(s) staged for review.")
        else:
            st.info("No editable grid changes detected.")


def _render_add_form(entity: EntityType, *, component_domain: str | None) -> None:
    with st.expander("Add row"):
        with st.form(f"database_management_add_{entity.value.lower()}"):
            origin_options = sorted(ORIGINS_BY_ENTITY[entity])
            origin = st.selectbox("Record origin", origin_options, key=f"database_management_add_origin_{entity.value.lower()}")
            if entity is EntityType.COMPONENT:
                st.caption(f"Domain: {component_domain}")
            values = _render_create_fields(entity)
            if st.form_submit_button("Stage new row"):
                payload = {**values, "record_origin": origin}
                if entity is EntityType.COMPONENT:
                    payload["domain"] = component_domain
                _stage_command(
                    {
                        "entity_type": entity.value,
                        "action": "CREATE",
                        "record_origin": origin,
                        "payload": _drop_blank(payload),
                        "current_record": {},
                        "record_id": None,
                    }
                )
                st.success("New row staged for review.")


def _render_create_fields(entity: EntityType) -> dict:
    defaults = _CREATE_DEFAULTS[entity]
    if entity is EntityType.VDE:
        cols = st.columns(2)
        return {
            "legislation": cols[0].selectbox("Legislation", ("EPA", "WLTP")),
            "category": cols[1].text_input("Category"),
            "make": cols[0].text_input("Make"),
            "model": cols[1].text_input("Model"),
            "mass_kg": cols[0].number_input("Curb mass [kg]", min_value=0.0, value=None),
            "notes": st.text_area("Notes"),
        }
    if entity is EntityType.FUEL_CONSUMPTION:
        cols = st.columns(2)
        return {
            "vde_id": cols[0].number_input("VDE ID", min_value=1, value=None, step=1),
            "electrification": cols[1].selectbox("Electrification", ("ICE", "MHEV", "HEV", "PHEV", "BEV")),
            "fuel_type": cols[0].text_input("Fuel type"),
            "method_note": cols[1].text_input("Method note"),
        }
    if entity is EntityType.TIRE:
        cols = st.columns(2)
        return {
            "tire_test_code": cols[0].text_input("Tire test code"),
            "manufacturer": cols[1].text_input("Manufacturer"),
            "model": cols[0].text_input("Model"),
            "standard_family": cols[1].selectbox("Standard family", ("ISO", "SAE", "CUSTOM")),
            "rr_n_per_kn": cols[0].number_input("RRC [N/kN]", min_value=0.0, value=None),
            "notes": st.text_area("Notes"),
        }
    cols = st.columns(2)
    return {
        "component_code": cols[0].text_input("Component code"),
        "component_name": cols[1].text_input("Component name"),
        "source_name": cols[0].text_input("Source"),
        "equivalent_A_N": cols[1].number_input("Equivalent A [N]", value=None),
        "equivalent_B_N_per_kph": cols[0].number_input("Equivalent B [N/kph]", value=None, format="%.6f"),
        "equivalent_C_N_per_kph2": cols[1].number_input("Equivalent C [N/kph^2]", value=None, format="%.8f"),
        "loss_pct": cols[0].number_input("Loss [%]", value=None),
        "notes": st.text_area("Notes"),
    }


def _render_selected_record(entity: EntityType, rows: list[dict], *, component_domain: str | None) -> None:
    if not rows:
        return
    labels = {str(row["id"]): _record_label(entity, row) for row in rows}
    selected_id = st.selectbox(
        "Selected record",
        list(labels),
        format_func=lambda record_id: labels[record_id],
        key=f"database_management_selected_{entity.value.lower()}",
    )
    record = get_record(entity, selected_id, component_domain=component_domain)
    if not record:
        st.warning("The selected record is no longer available. Refresh the browser.")
        return
    st.subheader("Record details")
    _render_detail_editor(entity, record, component_domain=component_domain)
    _render_record_actions(entity, record, component_domain=component_domain)


def _render_detail_editor(entity: EntityType, record: dict, *, component_domain: str | None) -> None:
    fields = tuple(field for field in _DETAIL_FIELDS[entity] if field in record)
    visible = {field: record.get(field) for field in fields}
    editable = {
        field
        for field in fields
        if field_access_for(entity, record.get("record_origin"), field)
        in {FieldAccess.EDITABLE, FieldAccess.ADVANCED_CORRECTION}
    }
    detail = st.data_editor(
        pd.DataFrame([visible]),
        key=f"database_management_detail_{entity.value.lower()}_{record['id']}",
        disabled=[field for field in fields if field not in editable],
        hide_index=True,
        width="stretch",
        num_rows="fixed",
    )
    if st.button("Stage selected edits", key=f"database_management_stage_detail_{entity.value.lower()}_{record['id']}"):
        edited = detail.to_dict(orient="records")[0]
        payload = {field: edited.get(field) for field in editable if not _same_value(record.get(field), edited.get(field))}
        if payload:
            _stage_command(_command_payload(entity, "UPDATE", record, payload=payload, component_domain=component_domain))
            st.success("Selected changes staged for review.")
        else:
            st.info("No editable changes detected.")
    readonly = [field for field in fields if field not in editable and record.get(field) is not None]
    if readonly:
        with st.expander("Read-only and derived values"):
            st.dataframe(pd.DataFrame([{"Field": field, "Value": record.get(field)} for field in readonly]), hide_index=True, width="stretch")


def _render_record_actions(entity: EntityType, record: dict, *, component_domain: str | None) -> None:
    action_cols = st.columns(4)
    if action_cols[0].button("Duplicate", key=f"database_management_duplicate_{entity.value.lower()}_{record['id']}"):
        _stage_command(
            _command_payload(
                entity,
                "DUPLICATE",
                record,
                payload=duplicate_payload_for(entity, record),
                component_domain=component_domain,
            )
        )
        st.success("Duplicate staged for review.")
    archived = not bool(record.get("is_active")) if entity is EntityType.TIRE else record.get("record_status") == "ARCHIVED"
    action = "RESTORE" if archived else "ARCHIVE"
    if action_cols[1].button("Restore" if archived else "Archive", key=f"database_management_archive_{entity.value.lower()}_{record['id']}"):
        _stage_command(_command_payload(entity, action, record, component_domain=component_domain))
        st.success(f"{action.title().lower()} staged for review.")
    dependencies = simple_dependencies(entity, record["id"])
    catalog_usage = None
    vde_dependencies = None
    if entity in {EntityType.TIRE, EntityType.COMPONENT}:
        catalog_usage = discover_catalog_usage(entity, record["id"], component_domain=component_domain)
        dependencies = (
            tuple(dependencies)
            + tuple(catalog_usage.get("usages") or ())
            + tuple(catalog_usage.get("historical_usages") or ())
            + tuple(catalog_usage.get("review_required") or ())
        )
    elif entity is EntityType.VDE:
        vde_dependencies = discover_vde_dependencies(record["id"])
        dependencies = (
            tuple(dependencies)
            + tuple(vde_dependencies.get("saved_proposals") or ())
            + tuple(vde_dependencies.get("baseline_requests") or ())
        )
    if action_cols[2].button("Delete permanently", key=f"database_management_delete_{entity.value.lower()}_{record['id']}", disabled=bool(dependencies)):
        _stage_command(_command_payload(entity, "DELETE", record, component_domain=component_domain))
        st.success("Delete staged for review.")
    if dependencies:
        action_cols[3].warning(f"Delete blocked: {len(dependencies)} direct dependency(ies).")
    if catalog_usage and dependencies:
        _render_dependency_resolution(entity, record, catalog_usage, component_domain=component_domain)
    if vde_dependencies and dependencies:
        _render_vde_dependency_resolution(record, vde_dependencies)


def _render_review_changes() -> None:
    staged = list(st.session_state.get(STAGE_KEY) or [])
    st.subheader("Review Changes")
    if not staged:
        st.caption("No changes staged.")
        return
    reason = st.text_area("Reason for staged corrections and lifecycle actions", key="database_management_review_reason")
    previews = [_preview_for_stage(command, reason) for command in staged]
    impact_previews = {}
    review_rows = []
    for index, (command, preview) in enumerate(zip(staged, previews)):
        impact = _render_impact_preview(index, command, preview)
        if impact is not None:
            impact_previews[index] = impact
        review_rows.append(
            {
                "Action": preview.action,
                "Entity": preview.entity_type,
                "Record": preview.record_id or "new",
                "Fields changed": ", ".join(diff.field for diff in preview.field_diff) or "-",
                "Ready": impact.can_commit if impact is not None else preview.can_commit,
                "Issues": "; ".join(
                    issue.message
                    for issue in [*preview.validation_issues, *(impact.validation_issues if impact is not None else ())]
                ) or "-",
            }
        )
    st.dataframe(pd.DataFrame(review_rows), hide_index=True, width="stretch")
    can_commit = all(
        preview.can_commit and (index not in impact_previews or impact_previews[index].can_commit)
        for index, preview in enumerate(previews)
    )
    action_cols = st.columns((1, 1, 3))
    if action_cols[0].button("Clear staged changes", key="database_management_clear_stage"):
        st.session_state[STAGE_KEY] = []
        st.rerun()
    if action_cols[1].button("Commit reviewed changes", key="database_management_commit_stage", disabled=not can_commit):
        failures: list[str] = []
        for index, preview in enumerate(previews):
            try:
                impact = impact_previews.get(index)
                command = staged[index]
                if command.get("resolution_action"):
                    apply_vde_dependency_resolution(
                        preview,
                        resolution_action=command["resolution_action"],
                        fuel_row_ids=tuple(command.get("fuel_row_ids") or ()),
                        replacement_vde_id=command.get("replacement_vde_id"),
                        reason=reason,
                    )
                elif impact is None:
                    apply_change(preview, reason=reason)
                else:
                    apply_change_with_impact(
                        preview,
                        impact,
                        reason=reason,
                        replacement_record_id=command.get("replacement_record_id"),
                        component_domain=command.get("component_domain"),
                    )
            except ValueError as exc:
                failures.append(str(exc))
                break
        if failures:
            st.error("No further staged changes were committed: " + failures[0])
        else:
            st.session_state[STAGE_KEY] = []
            st.success("All reviewed changes were committed and logged.")
            st.rerun()
    if not can_commit:
        action_cols[2].caption("Resolve validation issues and enter a reason where required before committing.")


def _preview_for_stage(command: dict, reason: str):
    return preview_change(
        ChangeCommand(
            entity_type=command["entity_type"],
            action=command["action"],
            record_id=command.get("record_id"),
            record_origin=command.get("record_origin"),
            current_record=deepcopy(command.get("current_record") or {}),
            payload=deepcopy(command.get("payload") or {}),
            reason=reason,
            physics_action=command.get("physics_action"),
        )
    )


def _command_payload(entity: EntityType, action: str, record: dict, *, payload: dict | None = None, component_domain: str | None) -> dict:
    return {
        "entity_type": entity.value,
        "action": action,
        "record_id": record.get("id"),
        "record_origin": record.get("record_origin"),
        "current_record": deepcopy(record),
        "payload": _drop_blank(payload or {}),
        "component_domain": component_domain,
    }


def _render_dependency_resolution(
    entity: EntityType,
    record: dict,
    usage: dict,
    *,
    component_domain: str | None,
) -> None:
    usage_count = len(usage.get("usages") or ())
    historical_count = len(usage.get("historical_usages") or ())
    vde_count = len(usage.get("affected_vde_ids") or ())
    review_count = len(usage.get("review_required") or ())
    with st.expander("Dependency resolution"):
        st.caption(
            f"{usage_count} active request usage(s), {historical_count} historical snapshot(s), "
            f"{vde_count} tracked VDE(s), "
            f"{review_count} VDE(s) requiring manual review."
        )
        cols = st.columns((1, 2))
        if cols[0].button(
            "Archive instead",
            key=f"database_management_dependency_archive_{entity.value.lower()}_{record['id']}",
        ):
            _stage_command(_command_payload(entity, "ARCHIVE", record, component_domain=component_domain))
            st.success("Archive staged for review.")

        candidates = [
            item
            for item in browse_records(entity, include_archived=False, component_domain=component_domain)
            if str(item.get("id")) != str(record.get("id"))
        ]
        if candidates:
            labels = {str(item["id"]): _record_label(entity, item) for item in candidates}
            replacement_id = cols[1].selectbox(
                "Compatible replacement",
                tuple(labels),
                format_func=lambda item: labels[item],
                key=f"database_management_dependency_replacement_{entity.value.lower()}_{record['id']}",
            )
            if st.button(
                "Stage reassignment",
                key=f"database_management_dependency_reassign_{entity.value.lower()}_{record['id']}",
            ):
                command = _command_payload(entity, "REASSIGN_RELATIONSHIP", record, component_domain=component_domain)
                command["replacement_record_id"] = replacement_id
                command["physics_action"] = "RECALCULATE_UPDATE"
                _stage_command(command)
                st.success("Reassignment staged for impact review.")
        else:
            cols[1].caption("No compatible active replacement is available.")

        keep_snapshots = st.checkbox(
            "Keep existing VDE snapshots as explicit historical references",
            key=f"database_management_dependency_keep_snapshots_{entity.value.lower()}_{record['id']}",
        )
        if st.button(
            "Stage permanent delete with historical snapshots",
            key=f"database_management_dependency_delete_{entity.value.lower()}_{record['id']}",
            disabled=not keep_snapshots,
        ):
            command = _command_payload(entity, "DELETE", record, component_domain=component_domain)
            command["physics_action"] = "KEEP_EXISTING"
            _stage_command(command)
            st.success("Permanent delete staged with explicit snapshot preservation.")


def _render_vde_dependency_resolution(record: dict, dependencies: dict) -> None:
    fuel_rows = tuple(dependencies.get("fuel_rows") or ())
    saved_proposals = tuple(dependencies.get("saved_proposals") or ())
    baseline_requests = tuple(dependencies.get("baseline_requests") or ())
    record_id = int(record["id"])
    with st.expander("VDE dependency resolution"):
        st.caption(
            f"{len(fuel_rows)} Fuel row(s), {len(saved_proposals)} saved proposal reference(s), "
            f"and {len(baseline_requests)} baseline request reference(s)."
        )
        if fuel_rows:
            st.dataframe(
                pd.DataFrame(fuel_rows)[
                    ["id", "vde_id", "electrification", "fuel_type", "record_origin", "review_status"]
                ],
                hide_index=True,
                width="stretch",
            )

        if st.button("Archive instead", key=f"database_management_vde_dependency_archive_{record_id}"):
            _stage_command(_command_payload(EntityType.VDE, "ARCHIVE", record, component_domain=None))
            st.success("VDE archive staged for review.")

        candidates = [
            item
            for item in browse_records(EntityType.VDE, include_archived=False)
            if int(item["id"]) != record_id
        ]
        eligible_rows = [
            row for row in fuel_rows if str(row.get("record_origin") or "").upper() in {"HOMOLOGATED", "MEASURED"}
        ]
        blocked_rows = [row for row in fuel_rows if row not in eligible_rows]
        if blocked_rows:
            st.warning(
                f"{len(blocked_rows)} calculated, Powertrain, or legacy Fuel row(s) cannot be reassigned without recalculation."
            )
        if candidates and eligible_rows:
            labels = {str(item["id"]): _record_label(EntityType.VDE, item) for item in candidates}
            replacement_id = st.selectbox(
                "Replacement VDE",
                tuple(labels),
                format_func=lambda item: labels[item],
                key=f"database_management_vde_replacement_{record_id}",
            )
            eligible_labels = {
                int(row["id"]): f"Fuel {row['id']} | {row.get('record_origin') or '-'} | {row.get('electrification') or '-'}"
                for row in eligible_rows
            }
            selected_fuel_ids = st.multiselect(
                "Fuel rows to reassign",
                tuple(eligible_labels),
                format_func=lambda item: eligible_labels[item],
                default=tuple(eligible_labels),
                key=f"database_management_vde_fuel_reassign_{record_id}",
            )
            if st.button(
                "Stage Fuel reassignment",
                key=f"database_management_vde_reassign_stage_{record_id}",
                disabled=not selected_fuel_ids,
            ):
                command = _command_payload(EntityType.VDE, "REASSIGN_RELATIONSHIP", record, component_domain=None)
                command.update(
                    {
                        "resolution_action": "REASSIGN_FUEL",
                        "replacement_vde_id": replacement_id,
                        "fuel_row_ids": tuple(selected_fuel_ids),
                    }
                )
                _stage_command(command)
                st.success("Fuel reassignment staged for review.")
        elif fuel_rows:
            st.caption("No directly reassignable Fuel rows or no compatible active VDE is available.")

        all_fuel_ids = tuple(int(row["id"]) for row in fuel_rows)
        confirm_delete = st.checkbox(
            "Delete every linked Fuel row before deleting this VDE",
            key=f"database_management_vde_delete_dependencies_{record_id}",
        )
        if st.button(
            "Stage Fuel and VDE delete",
            key=f"database_management_vde_delete_stage_{record_id}",
            disabled=not confirm_delete,
        ):
            command = _command_payload(EntityType.VDE, "DELETE", record, component_domain=None)
            command.update(
                {
                    "resolution_action": "DELETE_FUEL_AND_VDE",
                    "fuel_row_ids": all_fuel_ids,
                }
            )
            _stage_command(command)
            st.success("Explicit Fuel-first VDE delete staged for review.")

        confirm_admin = st.checkbox(
            "Administrative delete: acknowledge all linked Fuel rows will be explicitly removed",
            key=f"database_management_vde_admin_delete_{record_id}",
        )
        if st.button(
            "Stage administrative delete",
            key=f"database_management_vde_admin_delete_stage_{record_id}",
            disabled=not confirm_admin,
        ):
            command = _command_payload(EntityType.VDE, "DELETE", record, component_domain=None)
            command.update({"resolution_action": "ADMIN_DELETE", "fuel_row_ids": all_fuel_ids})
            _stage_command(command)
            st.success("Administrative VDE delete staged for review.")


def _render_impact_preview(index: int, command: dict, preview) -> object | None:
    entity = EntityType(str(command.get("entity_type") or ""))
    action = str(command.get("action") or "").upper()
    if entity not in {EntityType.TIRE, EntityType.COMPONENT} or action not in {
        "UPDATE",
        "DELETE",
        "REASSIGN_RELATIONSHIP",
    }:
        return None

    choice_labels = {
        "KEEP_EXISTING": "Keep existing VDEs and snapshots",
        "RECALCULATE_UPDATE": "Recalculate and update existing VDE rows",
        "RECALCULATE_NEW": "Recalculate and create new VDE rows",
    }
    choices = ("RECALCULATE_UPDATE", "RECALCULATE_NEW") if action == "REASSIGN_RELATIONSHIP" else tuple(choice_labels)
    default_choice = command.get("physics_action") if command.get("physics_action") in choices else choices[0]
    selected = st.selectbox(
        f"Impact handling - {entity.value.title()} {command.get('record_id')}",
        choices,
        index=choices.index(default_choice),
        format_func=lambda item: choice_labels[item],
        key=f"database_management_impact_choice_{index}_{entity.value}_{command.get('record_id')}",
    )
    impact = preview_dependency_impact(
        preview,
        selected,
        replacement_record_id=command.get("replacement_record_id"),
        component_domain=command.get("component_domain"),
    )
    summary = st.columns(5)
    summary[0].metric("Request usages", len(impact.usages))
    summary[1].metric("Historical snapshots", len(impact.historical_usages))
    summary[2].metric("VDE rows", len(impact.affected_vde_ids))
    summary[3].metric("Fuel rows", len(impact.stale_fuel_rows))
    summary[4].metric("Manual review", len(impact.review_required))
    comparison_rows = _impact_comparison_rows(impact)
    if comparison_rows:
        with st.expander(f"Resolved impact - {entity.value.title()} {command.get('record_id')}"):
            st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, width="stretch")
    if impact.review_required:
        st.warning("Some VDE rows have no canonical saved request and cannot be recalculated automatically.")
    return impact


def _impact_comparison_rows(impact) -> list[dict]:
    rows = []
    for request in impact.request_recalculations:
        for comparison in request.get("comparisons") or ():
            rows.append(
                {
                    "Request history": request.get("request_history_id"),
                    "Proposal": comparison.get("proposal_id"),
                    "Walk From": comparison.get("walk_from"),
                    "Status before": comparison.get("before_status"),
                    "Status after": comparison.get("after_status"),
                    "Mass before": comparison.get("before_mass_kg"),
                    "Mass after": comparison.get("after_mass_kg"),
                    "ABC TOTAL before": _format_abc(comparison.get("before_abc_total")),
                    "ABC TOTAL after": _format_abc(comparison.get("after_abc_total")),
                    "VDE before": _format_vde(comparison.get("before_vde")),
                    "VDE after": _format_vde(comparison.get("after_vde")),
                }
            )
    return rows


def _format_abc(value: dict | None) -> str:
    payload = dict(value or {})
    return " / ".join(_format_number(payload.get(key)) for key in ("A", "B", "C"))


def _format_vde(value: dict | None) -> str:
    payload = dict(value or {})
    total_payload = dict(payload.get("total") or {})
    net_payload = dict(payload.get("net") or {})
    total = _first_defined(total_payload.get("combined_mj_per_km"), total_payload.get("value"))
    net = _first_defined(net_payload.get("combined_mj_per_km"), net_payload.get("value"))
    return f"TOTAL {_format_number(total)} | NET {_format_number(net)}"


def _format_number(value) -> str:
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return "-"


def _first_defined(*values):
    return next((value for value in values if value is not None), None)


def _stage_command(command: dict) -> None:
    staged = list(st.session_state.get(STAGE_KEY) or [])
    identity = (command.get("entity_type"), command.get("action"), command.get("record_id"))
    for index, existing in enumerate(staged):
        if (existing.get("entity_type"), existing.get("action"), existing.get("record_id")) == identity:
            merged = dict(existing)
            merged.update({key: deepcopy(value) for key, value in command.items() if key != "payload"})
            merged["payload"] = {**dict(existing.get("payload") or {}), **dict(command.get("payload") or {})}
            staged[index] = merged
            st.session_state[STAGE_KEY] = staged
            return
    staged.append(deepcopy(command))
    st.session_state[STAGE_KEY] = staged


def _grid_editable_columns(entity: EntityType, rows: list[dict], fields: tuple[str, ...]) -> set[str]:
    editable: set[str] = set()
    for field in fields:
        if rows and all(field_access_for(entity, row.get("record_origin"), field) is FieldAccess.EDITABLE for row in rows):
            editable.add(field)
    return editable


def _record_label(entity: EntityType, row: dict) -> str:
    if entity is EntityType.VDE:
        return f"{row['id']} | {row.get('make') or '-'} {row.get('model') or '-'}"
    if entity is EntityType.FUEL_CONSUMPTION:
        return f"{row['id']} | VDE {row.get('vde_id')} | {row.get('electrification') or '-'}"
    if entity is EntityType.TIRE:
        return f"{row['id']} | {row.get('tire_test_code') or '-'}"
    return f"{row['id']} | {row.get('component_code') or row.get('component_id') or '-'}"


def _drop_blank(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None and value != ""}


def _same_value(first: Any, second: Any) -> bool:
    if pd.isna(first) and pd.isna(second):
        return True
    return first == second


def _ensure_stage() -> None:
    if not isinstance(st.session_state.get(STAGE_KEY), list):
        st.session_state[STAGE_KEY] = []
