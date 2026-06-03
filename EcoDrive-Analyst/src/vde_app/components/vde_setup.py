import pandas as pd
import streamlit as st

from src.vde_core.cycles import default_cycle_for_legislation, load_cycle_csv
from src.vde_core.services import estimate_aux_from_coastdown, load_vde_defaults
from src.vde_core.vde_setup_service import (
    build_decomp_update_for_edit,
    build_edit_core_update,
    build_vde_phase_update,
    collect_ctx_updates,
    compute_vde_preview_from_inputs,
    delete_vde_snapshot,
    fetch_linked_fuelcons_count,
    fetch_vde_edit_rows,
    fetch_vde_rows_full,
    ensure_baseline_aliases,
    baseline_filter_options,
    apply_baseline_filters,
    build_baseline_state_payload,
    build_delta_mode_ctx_updates,
    update_vde_snapshot,
)
from src.vde_core.utils import load_tire_catalog
from src.vde_app.components.shared import show_vde_feedback


def render_baseline_picker_and_editor_panel(
    *,
    tire_csv,
    rr_section,
    aero_section,
    parasitic_brake_section,
):
    st.subheader("Baseline -> Prefill + Edit everything")
    ctx = st.session_state.ctx

    try:
        rows = fetch_vde_rows_full()
    except Exception as e:
        st.error(f"Could not read vde_db: {e}")
        return

    if not rows:
        st.info("No snapshots in vde_db yet. Add one via 'Compute & Save' first.")
        return

    df = ensure_baseline_aliases(pd.DataFrame(rows))

    filter_opts = baseline_filter_options(df)
    with st.expander("Filters"):
        c1, c2, c3, c4 = st.columns(4)
        leg = c1.selectbox("Legislation", filter_opts["legislation"])
        make = c2.selectbox("Make", filter_opts["make"])
        cat_contains = c3.text_input("Category contains", "")
        year_eq = c4.text_input("Year (=)", "")

    dfv = apply_baseline_filters(
        df,
        legislation=leg,
        make=make,
        category_contains=cat_contains,
        year_eq=year_eq,
    )

    if dfv.empty:
        st.warning("No rows after filters.")
        with st.expander("Show raw columns (debug)"):
            st.write(sorted(df.columns.tolist()))
        return

    cols_to_show = [
        "id", "created_at", "updated_at",
        "legislation", "category", "make", "model", "year", "notes",
        "engine_type", "engine_model", "engine_size_l", "engine_aspiration",
        "transmission_type", "transmission_model", "drive_type",
        "mass_kg", "inertia_class", "cda_m2", "weight_dist_fr_pct", "payload_kg",
        "mro_kg", "options_kg", "wltp_category",
        "tire_size", "tire_rr_note", "smerf", "front_pressure_psi", "rear_pressure_psi",
        "rrc_N_per_kN", "crr1_frac_at_120kph", "rr_load_kpa",
        "coast_A_N", "coast_B_N_per_kph", "coast_C_N_per_kph2",
        "trans_A_coef_N", "trans_B_Npkph", "trans_C_coef_Npkph2",
        "brake_A_coef_N", "brake_B_Npkph", "brake_C_coef_Npkph2",
        "aero_C_coef_Npkph2",
        "rr_alpha_N", "rr_beta_Npkph", "rr_a_Npkph2", "rr_b_N", "rr_c_Npkph",
        "cycle_name", "cycle_source",
        "vde_urb_mj", "vde_hw_mj",
        "vde_net_mj_per_km", "vde_total_mj_per_km",
        "vde_urb_mj_per_km", "vde_hw_mj_per_km",
        "vde_low_mj_per_km", "vde_mid_mj_per_km", "vde_high_mj_per_km", "vde_extra_high_mj_per_km",
        "vde_id_parent", "baseline_A_N", "baseline_B_N_per_kph", "baseline_C_N_per_kph2", "baseline_mass_kg",
        "delta_rr_N", "delta_brake_N", "delta_parasitics_N", "delta_aero_Npkph2", "delta_mass_kg",
    ]
    cols_to_show = [c for c in cols_to_show if c in dfv.columns]
    st.dataframe(
        dfv[cols_to_show].sort_values("id", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    options = dfv.sort_values("id", ascending=False)["id"].astype(int).tolist()
    sel_id = st.selectbox("Pick baseline id", options)
    base = dfv[dfv["id"] == sel_id].iloc[0].to_dict()

    st.session_state.ctx.update(build_baseline_state_payload(base, int(sel_id)))

    prev_from_delta = ctx.get("from_delta", "Deltas")
    ctx["from_delta"] = st.radio(
        "How do you want to calculate on baseline?",
        ["Deltas", "Change Parameters"],
        index=["Deltas", "Change Parameters"].index(prev_from_delta),
        horizontal=True,
        key="baseline_flow_radio",
    )

    if ctx["from_delta"] == "Deltas":
        ctx.update(build_delta_mode_ctx_updates(base))
        with st.expander("Delta inputs from baseline"):
            c1, c2 = st.columns(2)
            ctx["delta_rr_N"] = c1.number_input("Delta RR (A) [N]", value=float(ctx.get("delta_rr_N", 0.0)), step=0.1)
            ctx["delta_aero_cdA"] = c2.number_input(
                "Delta Aero (CdA) [m2]",
                value=float(ctx.get("delta_aero_cdA", 0.0)),
                step=0.001,
                format="%.3f",
            )
            c3, c4, c5 = st.columns(3)
            ctx["delta_brake_N"] = c3.number_input("Delta Brake (A) [N]", value=float(ctx.get("delta_brake_N", 0.0)), step=0.1)
            ctx["delta_parasitics_N"] = c4.number_input(
                "Delta Parasitics (A) [N]",
                value=float(ctx.get("delta_parasitics_N", 0.0)),
                step=0.1,
            )
            ctx["delta_mass_kg"] = c5.number_input("Delta Mass [kg]", value=float(ctx.get("delta_mass_kg", 0.0)), step=1.0)
    else:
        tires_df = None
        try:
            tires_df = load_tire_catalog(tire_csv)
        except Exception:
            tires_df = None

        st.success(f"Editing baseline #{base.get('id', '')} (all fields below are editable).")

        rr_section(
            prefill={
                "rrc_N_per_kN": base.get("rrc_N_per_kN"),
                "crr1_frac_at_120kph": base.get("crr1_frac_at_120kph"),
                "mass_kg": base.get("mass_kg", base.get("inertia_class")),
                "tire_size": base.get("tire_size"),
            },
            tires_df=tires_df,
        )
        aero_section(
            prefill={
                "cd": base.get("cd"),
                "frontal_area_m2": base.get("frontal_area_m2"),
                "cda_m2": base.get("cda_m2"),
            }
        )
        parasitic_brake_section(
            prefill={
                "parasitic_A_coef_N": base.get("parasitic_A_coef_N"),
                "parasitic_B_Npkph": base.get("parasitic_B_Npkph"),
                "parasitic_C_coef_Npkph2": base.get("parasitic_C_coef_Npkph2"),
                "brake_A_coef_N": base.get("brake_A_coef_N"),
                "brake_B_Npkph": base.get("brake_B_Npkph"),
                "brake_C_coef_Npkph2": base.get("brake_C_coef_Npkph2"),
            }
        )

    with st.expander("Baseline snapshot (debug)"):
        key_cols = [
            "id", "legislation", "category", "make", "model", "year", "mass_kg", "A", "B", "C",
            "vde_net_mj_per_km", "vde_urb_mj_per_km", "vde_hw_mj_per_km",
            "vde_low_mj_per_km", "vde_mid_mj_per_km", "vde_high_mj_per_km", "vde_extra_high_mj_per_km",
        ]
        st.write({k: base.get(k) for k in key_cols if k in base})


def render_vde_edit_delete_panel(*, defaults_path, reset_ctx):
    st.markdown("---")
    st.subheader("Edit / Delete an existing VDE row")

    rows = fetch_vde_edit_rows(limit=100)
    if not rows:
        st.info("No VDE rows saved yet.")
        return

    labels = [
        f'#{r["id"]} - {r["legislation"]} | {r["category"]} | {r["make"]} {r["model"]} ({r.get("year","")})'
        for r in rows
    ]
    idx = st.selectbox("Pick a VDE to edit/delete", list(range(len(labels))), format_func=lambda i: labels[i])
    sel = rows[idx]
    vde_id_edit = sel["id"]
    st.caption(f"Editing VDE id: {vde_id_edit}")

    try:
        linked_n = fetch_linked_fuelcons_count(vde_id_edit)
        st.caption(f"Linked scenarios in fuelcons_db: {linked_n}")
    except Exception:
        pass

    with st.form(key=f"edit_vde_{vde_id_edit}"):
        c1, c2, c3, c4 = st.columns(4)
        a_edit = c1.number_input("A [N]", 0.0, 5000.0, float(sel["coast_A_N"] or 0.0), 0.1)
        b_edit = c2.number_input("B [N/kph]", -5.0, 5.0, float(sel["coast_B_N_per_kph"] or 0.0), 0.01)
        c_edit = c3.number_input("C [N/kph^2]", 0.000000, 1.000000, float(sel["coast_C_N_per_kph2"] or 0.0), 0.000001)
        m_edit = c4.number_input("Mass [kg]", 1.0, 4000.0, float(sel["mass_kg"] or 0.0), 1.0)

        c5, c6, c7 = st.columns(3)
        make_edit = c5.text_input("Make", value=sel["make"] or "")
        model_edit = c6.text_input("Model", value=sel["model"] or "")
        year_edit = c7.number_input("Year", 1990, 2100, int(sel["year"] or 2020))
        notes_edit = st.text_area("Notes", value=sel["notes"] or "")

        if st.form_submit_button("Save changes"):
            try:
                update_vde_snapshot(
                    vde_id_edit,
                    build_edit_core_update(
                        A=a_edit,
                        B=b_edit,
                        C=c_edit,
                        mass_kg=m_edit,
                        make=make_edit,
                        model=model_edit,
                        year=int(year_edit),
                        notes=notes_edit,
                    ),
                )

                rr_updates = collect_ctx_updates(
                    st.session_state.get("ctx"),
                    [
                        "tire_size",
                        "rrc_N_per_kN",
                        "crr1_frac_at_120kph",
                        "front_pressure_psi",
                        "rear_pressure_psi",
                        "rr_load_kpa",
                        "smerf",
                    ],
                    include_none=False,
                )
                if rr_updates:
                    update_vde_snapshot(vde_id_edit, rr_updates)

                pb_updates = collect_ctx_updates(
                    st.session_state.get("ctx"),
                    [
                        "parasitic_A_coef_N",
                        "parasitic_B_coef_Npkph",
                        "parasitic_C_coef_Npkph2",
                        "brake_A_coef_N",
                        "brake_B_Npkph",
                        "brake_C_coef_Npkph2",
                    ],
                    include_none=True,
                )
                if pb_updates:
                    update_vde_snapshot(vde_id_edit, pb_updates)

                try:
                    defaults_df = load_vde_defaults(defaults_path)
                    decomp = estimate_aux_from_coastdown(
                        A_N=a_edit,
                        B_N_per_kph=b_edit,
                        C_N_per_kph2=c_edit,
                        mass_kg=m_edit,
                        category=sel.get("category", ""),
                        electrification=sel.get("electrification", "ICE"),
                        transmission_type=sel.get("transmission_type", "AT"),
                        cdA_override_m2=sel.get("cda_m2"),
                        defaults_df=defaults_df,
                    )
                    decomp_upd = build_decomp_update_for_edit(decomp)
                    if decomp_upd:
                        update_vde_snapshot(vde_id_edit, decomp_upd)
                except Exception:
                    pass

                leg_row = sel.get("legislation", "EPA")
                try:
                    cycle_name = default_cycle_for_legislation(leg_row)
                    df_cycle = load_cycle_csv(cycle_name) if cycle_name else None
                except Exception:
                    df_cycle = None

                if isinstance(df_cycle, pd.DataFrame) and not df_cycle.empty:
                    preview = compute_vde_preview_from_inputs(
                        df_cycle,
                        leg_row,
                        A=a_edit,
                        B=b_edit,
                        C=c_edit,
                        mass_kg=m_edit,
                    )
                    if not preview.get("ok"):
                        raise ValueError(preview.get("error", "Failed to recompute VDE."))

                    total_mj_km = float(preview["total_mj_km"])
                    by_phase = dict(preview.get("by_phase", {}))
                    show_vde_feedback(total_mj_km, by_phase)

                    upd = {"vde_net_mj_per_km": total_mj_km}
                    upd_phase = build_vde_phase_update(
                        df_cycle,
                        leg_row,
                        A=a_edit,
                        B=b_edit,
                        C=c_edit,
                        mass_kg=m_edit,
                    )
                    if upd_phase:
                        upd.update(upd_phase)
                    update_vde_snapshot(vde_id_edit, upd)
                else:
                    st.warning("Row updated, but default cycle could not be loaded; phase VDE not recomputed.")

                st.success("Row updated.")
                reset_ctx(preserve_meta=True)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to update: {e}")

    with st.expander("Delete this VDE row"):
        st.warning("This action is irreversible. Linked fuelcons_db rows will be deleted (ON DELETE CASCADE).")
        confirm_text = st.text_input("Type DELETE to confirm:")
        delete_disabled = (str(confirm_text).strip().upper() != "DELETE")
        if st.button(f"Delete VDE id={vde_id_edit}", type="secondary", disabled=delete_disabled):
            try:
                deleted = delete_vde_snapshot(vde_id_edit)
                if deleted > 0:
                    st.success(f"VDE id={vde_id_edit} deleted.")
                    reset_ctx(preserve_meta=True)
                    st.rerun()
                else:
                    st.warning(f"VDE id={vde_id_edit} was not found (no row deleted).")
            except Exception as e:
                st.error(f"Failed to delete: {e}")
