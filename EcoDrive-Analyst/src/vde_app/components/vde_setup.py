import pandas as pd
import streamlit as st

from src.vde_core.db import ensure_db
from src.vde_core.cycles import default_cycle_for_legislation, load_cycle_csv, use_standard_cycle, cycle_summary
from src.vde_core.services import estimate_aux_from_coastdown, load_vde_defaults
from src.vde_core.tire_roadload_service import (
    get_available_tires,
    preview_tire_roadload_for_vde,
    save_tire_roadload_to_vde,
)
from src.vde_core.vde_setup_service import (
    build_decomp_update_for_edit,
    build_edit_core_update,
    build_test_mass_hint,
    build_vde_phase_update,
    build_live_vde_preview,
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
    db_list_makes,
    merge_update_payloads,
    resolve_tire_calculation_mass,
    resolve_tire_load_mass_basis,
    resolve_test_mass_kg,
    save_vde_from_ctx,
    to_float,
    validate_core,
    update_vde_snapshot,
)
from src.vde_core.utils import load_tire_catalog
from src.vde_app.plots import cycle_chart
from src.vde_app.components.shared import show_vde_feedback


def _tire_label(row: dict) -> str:
    return (
        f"#{row.get('id')} | "
        f"{row.get('manufacturer', '')} {row.get('model', '')} | "
        f"{row.get('size_code', '')} | "
        f"{str(row.get('standard_family', '')).upper()} | "
        f"{row.get('tire_test_code', '')}"
    )


def render_vehicle_basics_sidebar(*, reset_ctx):
    ctx = st.session_state.ctx
    with st.sidebar:
        st.header("Vehicle meta")

        leg_opts = ["WLTP", "EPA", "ABNT (Brazil)"]
        if ctx.get("legislation") not in leg_opts:
            ctx["legislation"] = "WLTP"

        c1, c2 = st.columns(2)
        with c1:
            ctx["legislation"] = st.selectbox(
                "Legislation",
                leg_opts,
                index=leg_opts.index(ctx["legislation"]),
                key="sb_leg",
            )

        epa_classes = [
            "Unknown", "Two Seaters", "Minicompact Cars", "Subcompact Cars", "Compact Cars",
            "Midsize Cars", "Large Cars", "Small Station Wagons", "Midsize Station Wagons",
            "Small SUVs", "Standard SUVs", "Minivans", "Vans", "Small Pickup Trucks", "Standard Pickup Trucks",
        ]
        wltp_classes = ["Class 1 (<850 kg)", "Class 2 (850-1220 kg)", "Class 3 (>1220 kg)"]
        category_list = epa_classes if ctx["legislation"] == "EPA" else wltp_classes
        category_list_upper = [category.upper() for category in category_list]

        if ctx.get("category") not in category_list_upper:
            ctx["category"] = category_list_upper[0]

        with c2:
            ctx["category"] = st.selectbox(
                "Category",
                category_list_upper,
                index=category_list_upper.index(ctx["category"]),
                key="sb_cat",
            )

        default_makes = [
            "Toyota", "Honda", "Nissan", "Mitsubishi", "Mazda", "Subaru", "Hyundai", "Kia",
            "Volkswagen", "Audi", "BMW", "Mercedes-Benz", "Porsche", "Peugeot", "Renault", "Citroen",
            "Fiat", "Alfa Romeo", "Volvo", "Jaguar", "Land Rover", "Skoda", "Seat", "Opel",
            "Ford", "Chevrolet", "Dodge", "Chrysler", "Jeep", "Ram", "Cadillac", "Buick", "GMC",
            "Lincoln", "Tesla", "Suzuki", "Mini", "Smart", "Lexus", "Infiniti", "Acura",
        ]
        default_makes_upper = [make.upper() for make in default_makes]
        try:
            ensure_db()
            makes_db = db_list_makes(ctx["legislation"], ctx["category"])
            makes_db = [make.upper() for make in makes_db]
        except Exception:
            makes_db = []

        merged_makes = list(dict.fromkeys(makes_db + [make for make in default_makes_upper if make not in makes_db]))
        if "OTHER (TYPE MANUALLY)" not in merged_makes:
            merged_makes.append("OTHER (TYPE MANUALLY)")

        c3, c4 = st.columns(2)
        with c3:
            selected_make = str(ctx.get("make", "")).upper()
            make_choice = st.selectbox(
                "Make/Brand",
                merged_makes,
                index=(merged_makes.index(selected_make) if selected_make in merged_makes else 0),
                key="sb_make_sel",
            )
            if make_choice == "OTHER (TYPE MANUALLY)":
                ctx["make"] = st.text_input("Enter custom brand", value=ctx.get("make", ""), key="sb_make_text").upper()
            else:
                ctx["make"] = make_choice

        with c4:
            ctx["model"] = st.text_input("Model/Desc.", value=ctx.get("model", ""), key="sb_model")

        c5, c6 = st.columns([1, 2])
        with c5:
            ctx["year"] = st.number_input("Year", 1900, 2100, int(ctx.get("year", 2024)), step=1, key="sb_year")
        with c6:
            ctx["notes"] = st.text_input("Proposal Description", value=ctx.get("notes", ""), key="sb_notes")

        elec_opts = ["ICE", "HEV", "PHEV", "BEV"]
        trans_opts = ["AT", "AMT", "CVT", "MT", "OT"]
        c7, c8 = st.columns(2)
        with c7:
            ctx["electrification"] = st.selectbox(
                "Electrification",
                elec_opts,
                index=elec_opts.index(ctx.get("electrification", "ICE")),
                key="sb_elec",
            )
        with c8:
            ctx["transmission_type"] = st.selectbox(
                "Transmission",
                trans_opts,
                index=trans_opts.index(ctx.get("transmission_type", "AT")),
                key="sb_trans",
            )

        st.markdown("---")
        prev_mode = ctx.get("mode", "From baseline (editable)")
        ctx["mode"] = st.radio(
            "Mode",
            ["From baseline (editable)", "Define all parameters (no baseline)", "From test (direct coastdown)"],
            index=["From baseline (editable)", "Define all parameters (no baseline)", "From test (direct coastdown)"].index(prev_mode),
            key="mode_radio",
        )
        if ctx["mode"] != prev_mode:
            reset_ctx(preserve_meta=True)
            st.rerun()


def render_tire_roadload_preview_panel(*, vde_id: int | None, base_row: dict | None = None):
    ctx = st.session_state.ctx
    st.subheader("Tire RoadLoad Preview")

    if not vde_id:
        st.info("Tire roadload preview needs a saved baseline row. Pick a baseline first.")
        return

    try:
        tires = get_available_tires()
    except Exception as e:
        st.error(f"Could not load tire_roadload_db: {e}")
        return

    if not tires:
        st.info("No active tires in tire_roadload_db yet.")
        return

    tire_by_id = {int(r["id"]): r for r in tires if r.get("id") is not None}
    tire_ids = list(tire_by_id.keys())
    base = dict(base_row or {})
    current_saved = {
        "front_tire_id": base.get("front_tire_id"),
        "rear_tire_id": base.get("rear_tire_id"),
        "front_pressure_psi": base.get("front_pressure_psi"),
        "rear_pressure_psi": base.get("rear_pressure_psi"),
        "tire_load_mass_basis": base.get("tire_load_mass_basis"),
        "tire_A_final": base.get("tire_A_final"),
        "tire_B_final": base.get("tire_B_final"),
        "tire_C_final": base.get("tire_C_final"),
        "tire_calc_source": base.get("tire_calc_source"),
        "tire_calc_notes": base.get("tire_calc_notes"),
    }
    has_saved_application = any(
        current_saved.get(key) is not None
        for key in ("front_tire_id", "rear_tire_id", "tire_A_final", "tire_B_final", "tire_C_final")
    )

    if has_saved_application:
        st.caption("Current tire application already saved in this VDE row.")
        s1, s2, s3 = st.columns(3)
        s1.metric("Saved tire A [N]", f"{float(current_saved.get('tire_A_final') or 0.0):.3f}")
        s2.metric("Saved tire B [N/kph]", f"{float(current_saved.get('tire_B_final') or 0.0):.6f}")
        s3.metric("Saved tire C [N/kph^2]", f"{float(current_saved.get('tire_C_final') or 0.0):.8f}")

        with st.expander("Saved tire application", expanded=False):
            saved_front_id = current_saved.get("front_tire_id")
            saved_rear_id = current_saved.get("rear_tire_id")
            st.write(
                {
                    "front_tire": _tire_label(tire_by_id[saved_front_id]) if saved_front_id in tire_by_id else saved_front_id,
                    "rear_tire": _tire_label(tire_by_id[saved_rear_id]) if saved_rear_id in tire_by_id else saved_rear_id,
                    "front_pressure_psi": current_saved.get("front_pressure_psi"),
                    "rear_pressure_psi": current_saved.get("rear_pressure_psi"),
                    "tire_load_mass_basis": current_saved.get("tire_load_mass_basis"),
                    "tire_calc_source": current_saved.get("tire_calc_source"),
                    "tire_calc_notes": current_saved.get("tire_calc_notes"),
                }
            )

        st.divider()

    front_default = ctx.get("front_tire_id") or base.get("front_tire_id")
    rear_default = ctx.get("rear_tire_id") or base.get("rear_tire_id")
    if front_default not in tire_ids:
        front_default = tire_ids[0]
    if rear_default not in tire_ids:
        rear_default = front_default

    same_default = bool(ctx.get("same_tire_front_rear", False) or (front_default == rear_default))
    legislation = str(base.get("legislation") or ctx.get("legislation") or "").strip().upper()
    basis_default = resolve_tire_load_mass_basis(
        {
            "legislation": legislation,
            "tire_load_mass_basis": ctx.get("tire_load_mass_basis") or base.get("tire_load_mass_basis"),
        }
    )

    c1, c2 = st.columns(2)
    front_tire_id = c1.selectbox(
        "Front tire",
        tire_ids,
        index=tire_ids.index(front_default),
        format_func=lambda tid: _tire_label(tire_by_id[tid]),
        key=f"tire_preview_front_{vde_id}",
    )
    same_tire = c2.checkbox(
        "Same tire front/rear",
        value=same_default,
        key=f"tire_preview_same_{vde_id}",
    )

    rear_tire_id = front_tire_id
    if same_tire:
        st.caption(f"Rear tire mirrors front tire: {_tire_label(tire_by_id[front_tire_id])}")
    else:
        rear_tire_id = st.selectbox(
            "Rear tire",
            tire_ids,
            index=tire_ids.index(rear_default),
            format_func=lambda tid: _tire_label(tire_by_id[tid]),
            key=f"tire_preview_rear_{vde_id}",
        )

    p1, p2, p3, p4 = st.columns(4)
    front_pressure_psi = p1.number_input(
        "Front pressure [psi]",
        value=float(ctx.get("front_pressure_psi", base.get("front_pressure_psi", 32.0)) or 32.0),
        step=0.5,
        format="%.1f",
        key=f"tire_preview_front_psi_{vde_id}",
    )
    rear_pressure_psi = p2.number_input(
        "Rear pressure [psi]",
        value=float(ctx.get("rear_pressure_psi", base.get("rear_pressure_psi", 32.0)) or 32.0),
        step=0.5,
        format="%.1f",
        key=f"tire_preview_rear_psi_{vde_id}",
    )
    front_weight_distribution_pct = p3.number_input(
        "Front weight distribution [%]",
        value=float(ctx.get("weight_dist_fr_pct", base.get("weight_dist_fr_pct", 50.0)) or 50.0),
        min_value=0.0,
        max_value=100.0,
        step=0.5,
        format="%.1f",
        key=f"tire_preview_weightdist_{vde_id}",
    )
    tire_improvement_pct = p4.number_input(
        "Tire improvement [%]",
        value=float(ctx.get("tire_improvement_pct", base.get("tire_improvement_pct", 0.0)) or 0.0),
        step=0.5,
        format="%.1f",
        key=f"tire_preview_improve_{vde_id}",
    )

    tire_load_mass_basis = str(ctx.get("tire_load_mass_basis") or basis_default).strip().upper()
    st.markdown("#### Tire calculation mass")
    st.caption(f"Using mass basis from Rolling Resistance block: `{tire_load_mass_basis}`")
    st.caption("Preview is non-persistent for now. It only calculates the tire contribution for the selected baseline.")

    preview_result = None
    if st.button("Preview Tire RoadLoad", key=f"btn_tire_preview_{vde_id}"):
        try:
            preview_result = preview_tire_roadload_for_vde(
                int(vde_id),
                {
                    "front_tire_id": front_tire_id,
                    "rear_tire_id": rear_tire_id,
                    "same_tire_front_rear": same_tire,
                    "front_pressure_psi": front_pressure_psi,
                    "rear_pressure_psi": rear_pressure_psi,
                    "front_weight_distribution_pct": front_weight_distribution_pct,
                    "tire_improvement_pct": tire_improvement_pct,
                    "tire_load_mass_basis": tire_load_mass_basis,
                    "mass_kg": ctx.get("mass_kg"),
                    "test_mass_kg": ctx.get("test_mass_kg"),
                    "inertia_class": ctx.get("inertia_class"),
                    "twc_kg": ctx.get("twc_kg"),
                    "etw_kg": ctx.get("etw_kg"),
                },
            )
            ctx["front_tire_id"] = front_tire_id
            ctx["rear_tire_id"] = rear_tire_id
            ctx["same_tire_front_rear"] = same_tire
            ctx["front_pressure_psi"] = front_pressure_psi
            ctx["rear_pressure_psi"] = rear_pressure_psi
            ctx["weight_dist_fr_pct"] = front_weight_distribution_pct
            ctx["tire_improvement_pct"] = tire_improvement_pct
            ctx["tire_load_mass_basis"] = tire_load_mass_basis
            ctx["tire_preview_result"] = preview_result
        except Exception as e:
            st.error(f"Failed to preview tire roadload: {e}")
            return

    if preview_result is None:
        cached = ctx.get("tire_preview_result")
        if isinstance(cached, dict) and cached.get("vde_id") == int(vde_id):
            preview_result = cached

    if not preview_result:
        return

    calc = preview_result.get("calculation", {})
    loads = calc.get("loads", {})
    total_base = calc.get("total_base_abc", {})
    total_final = calc.get("total_final_abc", {})
    mass_resolution = preview_result.get("mass_resolution", {})
    component_dict = preview_result.get("component_dict", {})
    delta_vs_saved = preview_result.get("delta_vs_saved", {})

    st.success("Tire preview ready.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mass used [kg]", f"{float(calc.get('tire_load_mass_used_kg', 0.0)):.1f}")
    m2.metric("Front axle load [N]", f"{float(loads.get('front_axle_load_n', 0.0)):.1f}")
    m3.metric("Rear axle load [N]", f"{float(loads.get('rear_axle_load_n', 0.0)):.1f}")
    m4.metric("Mass basis", str(mass_resolution.get("basis", "")))

    t1, t2, t3 = st.columns(3)
    t1.metric("Tire A final [N]", f"{float(total_final.get('A', 0.0)):.3f}")
    t2.metric("Tire B final [N/kph]", f"{float(total_final.get('B', 0.0)):.6f}")
    t3.metric("Tire C final [N/kph²]", f"{float(total_final.get('C', 0.0)):.8f}")

    with st.expander("Tire ABC details", expanded=False):
        st.write(
            {
                "front_tire": _tire_label(preview_result.get("front_tire", {})),
                "rear_tire": _tire_label(preview_result.get("rear_tire", {})),
                "mass_resolution": mass_resolution,
                "total_base_abc": total_base,
                "total_final_abc": total_final,
                "component": component_dict,
                "delta_vs_saved": delta_vs_saved,
                "save_payload": preview_result.get("save_payload", {}),
            }
        )

    save_col, note_col = st.columns([1, 3])
    if save_col.button("Apply Tire RoadLoad to this VDE", key=f"btn_tire_save_{vde_id}"):
        try:
            payload = save_tire_roadload_to_vde(int(vde_id), preview_result)
            ctx["tire_saved_payload"] = payload
            st.success(f"Tire roadload saved to VDE id={vde_id}.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to save tire roadload to VDE: {e}")

    saved_payload = ctx.get("tire_saved_payload")
    if isinstance(saved_payload, dict):
        note_col.caption(
            "Saved tire application: "
            f"front={saved_payload.get('front_tire_id')} | "
            f"rear={saved_payload.get('rear_tire_id')} | "
            f"A={float(saved_payload.get('tire_A_final', 0.0)):.3f} | "
            f"B={float(saved_payload.get('tire_B_final', 0.0)):.6f} | "
            f"C={float(saved_payload.get('tire_C_final', 0.0)):.8f}"
        )


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
        "mass_kg", "test_mass_kg", "inertia_class", "cda_m2", "weight_dist_fr_pct", "payload_kg",
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
        if str(ctx.get("legislation", "")).strip().upper() == "WLTP":
            st.info(
                "WLTP baseline path currently uses test mass for road/tire calculation. "
                "The code already has hooks for MRO/TPMLM-based WLTP mass resolution, "
                "but vehicle-type/cargo and TPMLM UI inputs are still a placeholder."
            )

        rr_section(
            prefill={
                "rrc_N_per_kN": base.get("rrc_N_per_kN"),
                "crr1_frac_at_120kph": base.get("crr1_frac_at_120kph"),
                "mass_kg": base.get("mass_kg"),
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

    with st.expander("Tire RoadLoad (preview only)", expanded=False):
        render_tire_roadload_preview_panel(vde_id=int(sel_id), base_row=base)

    with st.expander("Baseline snapshot (debug)"):
        key_cols = [
            "id", "legislation", "category", "make", "model", "year", "mass_kg", "test_mass_kg", "A", "B", "C",
            "vde_net_mj_per_km", "vde_urb_mj_per_km", "vde_hw_mj_per_km",
            "vde_low_mj_per_km", "vde_mid_mj_per_km", "vde_high_mj_per_km", "vde_extra_high_mj_per_km",
        ]
        st.write({k: base.get(k) for k in key_cols if k in base})


def render_compute_and_save_panel(*, defaults_df_getter, reset_ctx):
    ctx = st.session_state.ctx
    st.markdown("---")
    st.subheader("Compute VDE and Save to DB")

    errs, warns = validate_core(ctx["A"], ctx["B"], ctx["C"], ctx["mass_kg"])
    for warning in (warns or []):
        st.warning(warning)
    if ctx.get("cycle_df") is None:
        errs.append("Cycle not loaded. Pick default or upload a CSV.")
    for error in (errs or []):
        st.error(error)
    disabled_btn = bool(errs)

    if st.button("Compute VDE_NET and Save", key="btn_compute_save_main", disabled=disabled_btn):
        try:
            defaults_df = defaults_df_getter() if callable(defaults_df_getter) else None
            saved = save_vde_from_ctx(ctx, defaults_df=defaults_df)
            vde_id = int(saved["vde_id"])
            equiv = saved["equiv"]
            total_mj_km = float(saved["total_mj_km"])
            by_phase = dict(saved.get("by_phase", {}))

            st.info(f"VDE (NET): **{total_mj_km:.4f} MJ/km**  ({total_mj_km*277.7778:.1f} Wh/km)")
            with st.expander("RoadLoad breakdown", expanded=False):
                st.dataframe(pd.DataFrame(equiv.component_table), use_container_width=True, hide_index=True)
            if by_phase:
                order = ["city", "hwy", "low", "mid", "high", "xhigh"]
                keys = [k for k in order if k in by_phase] + [k for k in by_phase if k not in order]
                cols = st.columns(min(4, len(keys)))
                for i, key in enumerate(keys):
                    label = {"city": "CITY", "hwy": "HWY"}.get(key, key.upper())
                    cols[i % len(cols)].metric(label, f"{float(by_phase[key]):.4f} MJ/km")
            st.session_state["vde_id"] = vde_id

            st.success(f"Saved VDE snapshot (id={vde_id}).")
            reset_ctx(preserve_meta=True)
            st.rerun()

        except Exception as e:
            st.error(f"Failed to compute/save VDE: {e}")


def render_live_vde_preview_panel():
    ctx = st.session_state.get("ctx", {})
    preview = build_live_vde_preview(ctx)
    if not preview.get("ok"):
        st.warning(preview.get("error", "Preview not available."))
        return

    total_mj_km = float(preview["total_mj_km"])
    phases = preview.get("phases", {})
    equiv = preview.get("equiv")

    st.info(f"Live preview - VDE_NET: **{total_mj_km:.4f} MJ/km**  ({total_mj_km*277.7778:.1f} Wh/km)")
    if equiv is not None:
        with st.expander("RoadLoad breakdown", expanded=False):
            st.dataframe(pd.DataFrame(equiv.component_table), use_container_width=True, hide_index=True)

    if phases:
        order = ["city", "hwy", "low", "mid", "high", "xhigh"]
        ordered = [k for k in order if k in phases] + [k for k in phases if k not in order]
        cols = st.columns(min(4, len(ordered)))
        for i, key in enumerate(ordered):
            cols[i % len(cols)].metric(key.upper(), f"{phases[key]:.4f} MJ/km")


def render_cycle_section():
    ctx = st.session_state.ctx
    st.subheader("Drive cycle")

    errors, warns = validate_core(ctx["A"], ctx["B"], ctx["C"], ctx["mass_kg"])
    if warns:
        for warning in warns:
            st.warning(warning)

    cleft, cright = st.columns([1, 1])
    use_default = cleft.button("Use legislation default cycle")
    upload = cright.file_uploader(
        "or upload CSV with columns [t, v] (s, m/s)",
        type=["csv"],
        accept_multiple_files=False,
    )

    if use_default:
        cycle_name = default_cycle_for_legislation(ctx["legislation"])
        df_cycle = use_standard_cycle(ctx["legislation"])
        if df_cycle is None:
            st.warning(f"Default cycle for {ctx['legislation']} not found.")
            st.info("Upload a custom CSV cycle with columns [t, v].")
        else:
            ctx["cycle_df"] = df_cycle
            ctx["cycle_name"] = cycle_name
            st.success(f"Using default cycle: {cycle_name}.csv")

    if upload is not None:
        try:
            df_cycle = pd.read_csv(upload)
            if not {"t", "v"} <= set(df_cycle.columns):
                raise ValueError("Uploaded CSV must have columns: t, v (v in m/s)")
            df_cycle = df_cycle.copy()
            df_cycle["t"] = pd.to_numeric(df_cycle["t"], errors="raise")
            df_cycle["v"] = pd.to_numeric(df_cycle["v"], errors="raise")
            if "phase" in df_cycle.columns:
                df_cycle["phase"] = df_cycle["phase"].astype(str)
            ctx["cycle_df"] = df_cycle
            ctx["cycle_name"] = upload.name
            st.success(f"Cycle loaded: {upload.name}")
        except Exception as e:
            st.error(f"CSV load error: {e}")

    if ctx["cycle_df"] is not None:
        kpi, _dist_km = cycle_summary(ctx["cycle_df"])
        st.caption(kpi)
    else:
        errors.append("No cycle loaded. Use default or upload a CSV.")

    if ctx["cycle_df"] is not None:
        fig = cycle_chart(ctx["cycle_df"])
        if fig:
            st.plotly_chart(fig, use_container_width=True)


def render_auxiliaries_section(*, defaults_df_getter):
    """
    Uses A/B/C + mass + (category, electrification, transmission_type) from ctx
    to decompose NET coastdown using the defaults CSV.
    """
    ctx = st.session_state.ctx
    st.subheader("Estimate auxiliaries from coastdown (NET)")

    missing = [k for k in ("A", "B", "C", "mass_kg", "category") if ctx.get(k) in (None, "")]
    disabled = len(missing) > 0
    if disabled:
        st.caption(f"Fill first: {', '.join(missing)}")

    if st.button("Estimate using defaults CSV", disabled=disabled):
        defaults_df = defaults_df_getter() if callable(defaults_df_getter) else None
        res = estimate_aux_from_coastdown(
            A_N=ctx["A"],
            B_N_per_kph=ctx["B"],
            C_N_per_kph2=ctx["C"],
            mass_kg=ctx["mass_kg"],
            category=ctx["category"],
            electrification=ctx.get("electrification", "ICE"),
            transmission_type=ctx.get("transmission_type", "AT"),
            cdA_override_m2=ctx.get("cda_m2"),
            defaults_df=defaults_df,
        )

        ctx.update(
            {
                "rr_alpha_N": res["rr_alpha_N"],
                "rr_beta_Npkph": res["rr_beta_Npkph"],
                "aero_C_coef_Npkph2": res["aero_C_coef_Npkph2"],
                "parasitic_A_N": res["parasitic_A_coef_N"],
                "parasitic_B_Npkph": res["parasitic_B_coef_Npkph"],
                "parasitic_C_Npkph2": res["parasitic_C_coef_Npkph2"],
                "decomp_check_ok": res["check_ok"],
                "cda_m2": res["cdA_used_m2"],
            }
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("RR alpha [N]", f"{res['rr_alpha_N']:.2f}")
        c2.metric("RR beta [N/kph]", f"{res['rr_beta_Npkph']:.3f}")
        c3.metric("Aero C [N/kph^2]", f"{res['aero_C_coef_Npkph2']:.3f}")
        d1, d2, d3 = st.columns(3)
        d1.metric("Parasitic A [N]", f"{res['parasitic_A_coef_N']:.2f}")
        d2.metric("Parasitic B [N/kph]", f"{res['parasitic_B_coef_Npkph']:.3f}")
        d3.metric("Check", "OK" if res["check_ok"] else "Review")


def render_from_test_section():
    """
    Enter coastdown outputs, curb weight, and optional test mass directly.
    Keeps compatibility with legacy session keys as a transition layer.
    """
    ctx = st.session_state.ctx
    st.subheader("From test - direct coastdown (A/B/C), curb weight, and test mass")

    colA, colB, colC, colM = st.columns(4)
    A = colA.number_input("A [N]", 0.0, 500.0, float(ctx.get("A", 30.0)), 0.1)
    B = colB.number_input("B [N/kph]", -1.0, 5.0, float(ctx.get("B", 0.80)), 0.01)
    C = colC.number_input("C [N/kph^2]", 0.000, 0.100, float(ctx.get("C", 0.011)), 0.001)
    mass = colM.number_input("Curb weight [kg]", 300.0, 3500.0, float(ctx.get("mass_kg", 1500.0)), 5.0)

    current_mass = to_float(mass)
    test_mass_default = resolve_test_mass_kg({**ctx, "mass_kg": current_mass, "test_mass_kg": None})
    use_default = st.checkbox("Use default test mass", value=bool(ctx.get("test_mass_use_default", True)), key="from_test_use_default")

    ctx["A"], ctx["B"], ctx["C"], ctx["mass_kg"] = to_float(A), to_float(B), to_float(C), current_mass
    ctx["test_mass_use_default"] = use_default

    if use_default:
        ctx["test_mass_kg"] = None
        st.caption(f"Test mass [kg]: {float(test_mass_default or 0.0):.1f}")
    else:
        manual_test_mass = st.number_input(
            "Test mass [kg]",
            min_value=float(current_mass or 0.0),
            max_value=3500.0,
            value=float(max(to_float(ctx.get("test_mass_kg"), test_mass_default or current_mass or 0.0), current_mass or 0.0)),
            step=5.0,
            format="%.1f",
            key="from_test_manual_test_mass",
        )
        ctx["test_mass_kg"] = to_float(manual_test_mass)

    hint = build_test_mass_hint(ctx)
    if hint:
        st.caption(hint)

    st.session_state["abc"] = {"A": float(A), "B": float(B), "C": float(C)}
    st.session_state["manual_mass"] = to_float(mass)

    st.info("Values stored in ctx and in session_state['abc'] / ['manual_mass'] for compatibility.")


def render_aero_section(*, prefill=None):
    """
    Uses cda_m2 from DB. Shows estimated aero C [N/kph^2] as reference.
    Does not overwrite measured coastdown C.
    """
    ctx = st.session_state.ctx
    st.subheader("Aerodynamics")

    cda0 = to_float(prefill.get("cda_m2"), ctx.get("cda_m2")) if prefill else to_float(ctx.get("cda_m2"), None)
    cda = st.number_input("CdA [m^2]", value=float(cda0 or 0.0), step=0.01, format="%.3f")
    ctx["cda_m2"] = to_float(cda)

    rho = 1.2
    C_aero = 0.5 * rho * ctx["cda_m2"] * (1 / 3.6) ** 2
    ctx["aero_C_coef_Npkph2"] = C_aero

    st.metric("C_aero (est.) [N/kph^2]", f"{C_aero:.6f}")
    st.caption("Measured coastdown C remains in 'coast_C_N_per_kph2'; this is only reference.")


def render_rr_section(*, prefill=None, tires_df=None):
    """
    RR only (does not overwrite A/B/C):
      IN: rrc_N_per_kN [N/kN], crr1_frac_at_120kph [-], mass_kg [kg]
      Optional: tire selectbox when tires_df is provided
      OUT: rr_alpha_N [N], rr_beta_Npkph [N/kph]; stores tire_size in ctx
    """
    ctx = st.session_state.ctx
    st.subheader("Rolling Resistance")

    if isinstance(tires_df, pd.DataFrame) and not tires_df.empty:
        sizes = sorted(tires_df["tire_size"].dropna().astype(str).unique().tolist())
        current_size = str(ctx.get("tire_size") or (prefill.get("tire_size") if prefill else "") or "")
        try:
            idx0 = sizes.index(current_size) if current_size in sizes else 0
        except Exception:
            idx0 = 0
        sel = st.selectbox("Tire size", sizes, index=idx0)
        ctx["tire_size"] = sel
        trow = tires_df.loc[tires_df["tire_size"] == sel].iloc[0].to_dict()
        st.caption(f'Diameter {trow["tire_circ_mm"]:.0f} mm ')
        ctx["tire_circ_m"] = float(trow["tire_circ_mm"]) / 1000.0

    if prefill:
        rrc0 = to_float(prefill.get("rrc_N_per_kN"), to_float(ctx.get("rrc_N_per_kN"), 9.5))
        frac0 = to_float(prefill.get("crr1_frac_at_120kph"), to_float(ctx.get("crr1_frac_at_120kph"), 0.10))
        m0 = to_float(prefill.get("mass_kg"), to_float(ctx.get("mass_kg"), 1500.0))
        weight_dist0 = to_float(prefill.get("weight_dist_fr_pct"), to_float(ctx.get("weight_dist_fr_pct"), 50.0))
    else:
        rrc0 = to_float(ctx.get("rrc_N_per_kN"), 9.5)
        frac0 = to_float(ctx.get("crr1_frac_at_120kph"), 0.10)
        m0 = to_float(ctx.get("mass_kg"), 1500.0)
        weight_dist0 = to_float(ctx.get("weight_dist_fr_pct"), 50.0)

    c1, c2 = st.columns(2)
    ctx["rrc_N_per_kN"] = c1.number_input("RRC [N/kN]", value=float(rrc0), step=0.1, format="%.2f")
    ctx["crr1_frac_at_120kph"] = c2.number_input(
        "Crr1 @120kph [-]",
        value=float(frac0),
        min_value=0.0,
        max_value=1.0,
        step=0.005,
        format="%.3f",
    )

    st.markdown("#### Mass")
    mass1, mass2, mass3 = st.columns(3)
    legislation_display = str(ctx.get("legislation", "") or "-")
    mass1.text_input("Legislation", value=legislation_display, disabled=True, key="rr_legislation_display")
    ctx["mass_kg"] = mass2.number_input("Curb weight [kg]", value=float(m0), step=1.0, format="%.1f")
    ctx["weight_dist_fr_pct"] = mass3.number_input(
        "Front weight distribution [%]",
        min_value=0.0,
        max_value=100.0,
        value=float(weight_dist0 or 50.0),
        step=0.5,
        format="%.1f",
    )

    legislation = str(ctx.get("legislation", "") or "").strip().upper()
    if legislation == "EPA":
        current_basis = resolve_tire_load_mass_basis(ctx)
        ctx["tire_load_mass_basis"] = st.radio(
            "EPA tire calculation mass",
            ["TWC", "TEST_MASS"],
            index=["TWC", "TEST_MASS"].index(current_basis if current_basis in {"TWC", "TEST_MASS"} else "TWC"),
            horizontal=True,
            key="rr_epa_tire_mass_basis",
        )
        st.caption("EPA tire calculation can use TWC or test mass. Default is TWC.")
    else:
        ctx["tire_load_mass_basis"] = "TEST_MASS"
        st.caption("Tire calculation mass basis: TEST_MASS")

    test_mass_prefill = to_float(prefill.get("test_mass_kg"), to_float(ctx.get("test_mass_kg"))) if prefill else to_float(ctx.get("test_mass_kg"))
    test_mass_default = resolve_test_mass_kg({**ctx, "mass_kg": ctx["mass_kg"], "test_mass_kg": None})
    saved_use_default = bool(ctx.get("test_mass_use_default", True))
    if test_mass_prefill is not None and test_mass_default is not None and abs(test_mass_prefill - test_mass_default) > 1e-9:
        saved_use_default = False

    tm1, tm2 = st.columns([1, 2])
    use_default = tm1.checkbox("Use default test mass", value=saved_use_default, key="rr_use_default_test_mass")
    ctx["test_mass_use_default"] = use_default
    if use_default:
        ctx["test_mass_kg"] = None
        tm2.caption(f"Test mass [kg]: {float(test_mass_default or 0.0):.1f}")
    else:
        ctx["test_mass_kg"] = tm2.number_input(
            "Test mass [kg]",
            min_value=float(ctx["mass_kg"] or 0.0),
            max_value=4000.0,
            value=float(max(test_mass_prefill if test_mass_prefill is not None else (test_mass_default or ctx["mass_kg"] or 0.0), ctx["mass_kg"] or 0.0)),
            step=1.0,
            format="%.1f",
            key="rr_manual_test_mass",
        )

    hint = build_test_mass_hint(ctx)
    if hint:
        st.caption(hint)

    tire_mass_resolution = resolve_tire_calculation_mass(ctx)
    calc_mass_kg = tire_mass_resolution.get("mass_kg")
    if legislation == "EPA" and ctx.get("tire_load_mass_basis") == "TWC":
        ctx["inertia_class"] = calc_mass_kg
        ctx["twc_kg"] = calc_mass_kg
    G = 9.80665
    load_kN = (ctx["mass_kg"] * G) / 1000.0 if ctx.get("mass_kg") else 0.0
    A_rr = (ctx["rrc_N_per_kN"] or 0.0) * load_kN
    B_rr = ((ctx["crr1_frac_at_120kph"] or 0.0) * A_rr) / 120.0

    ctx["rr_alpha_N"] = A_rr
    ctx["rr_beta_Npkph"] = B_rr

    c4, c5, c6 = st.columns(3)
    c4.metric("Load [kN]", f"{load_kN:.2f}")
    c5.metric("A_rr ~ SMERF [N]", f"{A_rr:.2f}")
    c6.metric("Calculation mass [kg]", f"{float(calc_mass_kg or 0.0):.1f}")
    st.caption(f"Current tire mass basis: {tire_mass_resolution.get('basis')} ({tire_mass_resolution.get('source')})")
    st.caption(f"Estimated RR model: F_rr(v) = {A_rr:.2f} + {B_rr:.5f}*v   [v in kph]")


def render_parasitic_brake_section(*, prefill=None):
    """
    Parasitics + Brake in one section (DB fields):
      parasitic_A/B/C, brake_A/B/C (all optional; default 0)
    """
    ctx = st.session_state.ctx
    st.subheader("Parasitics + Brake")

    if prefill:
        parA0 = to_float(prefill.get("parasitic_A_coef_N"), ctx.get("parasitic_A_coef_N", 0.0))
        parB0 = to_float(prefill.get("parasitic_B_Npkph"), ctx.get("parasitic_B_Npkph", 0.0))
        parC0 = to_float(prefill.get("parasitic_C_coef_Npkph2"), ctx.get("parasitic_C_coef_Npkph2", 0.0))
        brA0 = to_float(prefill.get("brake_A_coef_N"), ctx.get("brake_A_coef_N", 0.0))
        brB0 = to_float(prefill.get("brake_B_Npkph"), ctx.get("brake_B_Npkph", 0.0))
        brC0 = to_float(prefill.get("brake_C_coef_Npkph2"), ctx.get("brake_C_coef_Npkph2", 0.0))
    else:
        parA0 = to_float(ctx.get("parasitic_A_coef_N"), 0.0)
        parB0 = to_float(ctx.get("parasitic_B_Npkph"), 0.0)
        parC0 = to_float(ctx.get("parasitic_C_coef_Npkph2"), 0.0)
        brA0 = to_float(ctx.get("brake_A_coef_N"), 0.0)
        brB0 = to_float(ctx.get("brake_B_Npkph"), 0.0)
        brC0 = to_float(ctx.get("brake_C_coef_Npkph2"), 0.0)

    p1, p2, p3 = st.columns(3)
    ctx["parasitic_A_coef_N"] = p1.number_input("Parasitic A [N]", value=float(parA0), step=0.1, format="%.2f")
    ctx["parasitic_B_Npkph"] = p2.number_input("Parasitic B [N/kph]", value=float(parB0), step=0.001, format="%.5f")
    ctx["parasitic_C_coef_Npkph2"] = p3.number_input("Parasitic C [N/kph^2]", value=float(parC0), step=0.0001, format="%.6f")

    b1, b2, b3 = st.columns(3)
    ctx["brake_A_coef_N"] = b1.number_input("Brake A [N]", value=float(brA0), step=0.1, format="%.2f")
    ctx["brake_B_Npkph"] = b2.number_input("Brake B [N/kph]", value=float(brB0), step=0.001, format="%.5f")
    ctx["brake_C_coef_Npkph2"] = b3.number_input("Brake C [N/kph^2]", value=float(brC0), step=0.0001, format="%.6f")

    c1, c2 = st.columns(2)
    c1.metric("A_par + A_brake [N]", f"{ctx['parasitic_A_coef_N'] + ctx['brake_A_coef_N']:.2f}")
    c2.metric("B_par + B_brake [N/kph]", f"{ctx['parasitic_B_Npkph'] + ctx['brake_B_Npkph']:.5f}")


def render_vde_edit_delete_panel(*, defaults_path, reset_ctx, defaults_df_getter=None):
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
        m_edit = c4.number_input("Curb weight [kg]", 1.0, 4000.0, float(sel["mass_kg"] or 0.0), 1.0)
        test_mass_prefill = to_float(sel.get("test_mass_kg"))
        test_mass_default = resolve_test_mass_kg(
            {
                "legislation": sel.get("legislation"),
                "mass_kg": m_edit,
                "test_mass_kg": None,
            }
        )
        use_default_edit = st.checkbox(
            "Use default test mass",
            value=not (test_mass_prefill is not None and test_mass_default is not None and abs(test_mass_prefill - test_mass_default) > 1e-9),
            key=f"edit_vde_use_default_{vde_id_edit}",
        )
        if use_default_edit:
            test_mass_edit = None
            st.caption(f"Test mass [kg]: {float(test_mass_default or 0.0):.1f}")
        else:
            test_mass_edit = st.number_input(
                "Test mass [kg]",
                min_value=float(m_edit),
                max_value=4000.0,
                value=float(max(test_mass_prefill if test_mass_prefill is not None else (test_mass_default or m_edit), m_edit)),
                step=1.0,
                format="%.1f",
                key=f"edit_vde_manual_test_mass_{vde_id_edit}",
            )
        hint = build_test_mass_hint({"legislation": sel.get("legislation")})
        if hint:
            st.caption(hint)

        c5, c6, c7 = st.columns(3)
        make_edit = c5.text_input("Make", value=sel["make"] or "")
        model_edit = c6.text_input("Model", value=sel["model"] or "")
        year_edit = c7.number_input("Year", 1990, 2100, int(sel["year"] or 2020))
        notes_edit = st.text_area("Notes", value=sel["notes"] or "")

        if st.form_submit_button("Save changes"):
            try:
                core_updates = build_edit_core_update(
                    A=a_edit,
                    B=b_edit,
                    C=c_edit,
                    mass_kg=m_edit,
                    test_mass_kg=to_float(test_mass_edit) or None,
                    make=make_edit,
                    model=model_edit,
                    year=int(year_edit),
                    notes=notes_edit,
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
                decomp_upd = {}

                try:
                    defaults_df = (
                        defaults_df_getter()
                        if callable(defaults_df_getter)
                        else load_vde_defaults(defaults_path)
                    )
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
                except Exception:
                    decomp_upd = {}

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

                    phase_updates = merge_update_payloads(
                        {"vde_net_mj_per_km": total_mj_km},
                        build_vde_phase_update(
                            df_cycle,
                            leg_row,
                            A=a_edit,
                            B=b_edit,
                            C=c_edit,
                            mass_kg=m_edit,
                        ),
                    )
                    final_updates = merge_update_payloads(
                        core_updates,
                        rr_updates,
                        pb_updates,
                        decomp_upd,
                        phase_updates,
                    )
                    if final_updates:
                        update_vde_snapshot(vde_id_edit, final_updates)
                else:
                    final_updates = merge_update_payloads(
                        core_updates,
                        rr_updates,
                        pb_updates,
                        decomp_upd,
                    )
                    if final_updates:
                        update_vde_snapshot(vde_id_edit, final_updates)
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
