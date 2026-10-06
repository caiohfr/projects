"""Sprint 12C.2 read-only canonical population and grain preview.

This script creates audit artifacts only. It never executes DDL, writes to a
runtime database, edits raw sources, or assigns final canonical identities.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "etl" / "scripts"))
import sprint_12c1_dataset_delta_audit as c1  # noqa: E402


DB_PATH = ROOT / "data" / "db" / "eco_drive.db"
OUT = ROOT / "etl" / "data" / "processed" / "sprint_12c2_canonical_population_preview"
REPORT = ROOT / "etl" / "reports" / "sprint_12c2_canonical_population_preview.md"
TREE_REPORT = ROOT / "etl" / "reports" / "sprint_12c2_population_tree_examples.md"

OUTPUT_PATHS = (
    OUT / "population_funnel.csv",
    OUT / "entity_count_forecast.csv",
    OUT / "epa2026_grouping_preview.csv",
    OUT / "legacy_future_mapping.csv",
    OUT / "population_examples.csv",
    OUT / "collision_report.csv",
    OUT / "historical_refresh_delta.csv",
    OUT / "population_preview.json",
    REPORT,
    TREE_REPORT,
)

CONFIG_FIELDS = (
    "Represented Test Veh Make", "Represented Test Veh Model", "Model Year",
    "Actual Tested Testgroup", "Test Vehicle ID", "Test Veh Configuration #",
    "Test Veh Displacement (L)", "Engine Code", "Tested Transmission Type",
    "# of Gears", "Drive System Description", "Axle Ratio", "N/V Ratio",
)
TARGET_FIELDS = (
    "Target Coef A (lbf)", "Target Coef B (lbf/mph)", "Target Coef C (lbf/mph**2)",
)
SET_FIELDS = (
    "Set Coef A (lbf)", "Set Coef B (lbf/mph)", "Set Coef C (lbf/mph**2)",
)
MMY_FIELDS = ("Represented Test Veh Make", "Represented Test Veh Model", "Model Year")
RUN_FIELDS = ("Test Number", "Actual Tested Testgroup")


def clean(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


def key_value(value: Any, normalize_text: bool = False) -> str:
    value = clean(value)
    if value is None:
        return "<NULL>"
    if normalize_text and isinstance(value, str):
        return c1.norm_text(value) or "<EMPTY>"
    return str(value).strip()


def stable_preview_id(prefix: str, *parts: Any) -> str:
    material = "\x1f".join(key_value(part, normalize_text=isinstance(part, str)) for part in parts)
    return f"PREVIEW-{prefix}-{hashlib.sha256(material.encode('utf-8')).hexdigest()[:16].upper()}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_db() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        con.execute("PRAGMA query_only = ON")
        vde = pd.read_sql_query("SELECT * FROM vde_db ORDER BY id", con)
        fuelcons = pd.read_sql_query("SELECT * FROM fuelcons_db ORDER BY id", con)
    finally:
        con.close()
    return vde, fuelcons


def tuple_from_row(row: pd.Series, fields: Iterable[str], normalize_text: bool = False) -> tuple[str, ...]:
    return tuple(key_value(row.get(field), normalize_text=normalize_text) for field in fields)


def prepare_epa_candidates(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = source.copy().reset_index().rename(columns={"index": "_source_index"})
    rows["source_excel_row"] = rows["_source_index"].astype(int) + 2
    rows["program_candidate_id"] = rows.apply(
        lambda row: stable_preview_id("PROGRAM", "EPA_TESTCAR", *tuple_from_row(row, MMY_FIELDS, True)), axis=1
    )
    rows["configuration_candidate_id"] = rows.apply(
        lambda row: stable_preview_id("VC", "EPA_TESTCAR", *tuple_from_row(row, CONFIG_FIELDS, True)), axis=1
    )
    rows["vde_candidate_id"] = rows.apply(
        lambda row: stable_preview_id(
            "VDE", row["configuration_candidate_id"],
            *tuple_from_row(row, TARGET_FIELDS), key_value(row.get("Equivalent Test Weight (lbs.)")),
        ), axis=1
    )
    rows["run_candidate_id"] = rows.apply(
        lambda row: stable_preview_id("RUN", "EPA_TESTCAR", *tuple_from_row(row, RUN_FIELDS, True)), axis=1
    )

    config_state_counts = rows.groupby("configuration_candidate_id")["vde_candidate_id"].nunique().to_dict()
    mmy_config_counts = rows.groupby("program_candidate_id")["configuration_candidate_id"].nunique().to_dict()
    run_row_counts = rows.groupby("run_candidate_id").size().to_dict()
    preview: list[dict[str, Any]] = []
    for vde_id, group in rows.groupby("vde_candidate_id", sort=True):
        first = group.sort_values("source_excel_row").iloc[0]
        run_ids = sorted(group["run_candidate_id"].unique())
        set_variants = group[list(SET_FIELDS)].drop_duplicates().shape[0]
        preview.append({
            "program_candidate_id": first["program_candidate_id"],
            "configuration_candidate_id": first["configuration_candidate_id"],
            "vde_candidate_id": vde_id,
            "make": clean(first["Represented Test Veh Make"]),
            "model": clean(first["Represented Test Veh Model"]),
            "model_year": clean(first["Model Year"]),
            "test_group": clean(first.get("Actual Tested Testgroup")),
            "test_vehicle_id": clean(first.get("Test Vehicle ID")),
            "configuration_number": clean(first.get("Test Veh Configuration #")),
            "etw_lb": clean(first.get("Equivalent Test Weight (lbs.)")),
            "target_a_lbf": clean(first[TARGET_FIELDS[0]]),
            "target_b_lbf_per_mph": clean(first[TARGET_FIELDS[1]]),
            "target_c_lbf_per_mph2": clean(first[TARGET_FIELDS[2]]),
            "source_rows": len(group),
            "source_excel_rows": ";".join(map(str, sorted(group["source_excel_row"].astype(int).tolist()))),
            "run_candidates": len(run_ids),
            "run_candidate_ids": ";".join(run_ids),
            "set_abc_variants": set_variants,
            "vde_states_in_configuration": config_state_counts[first["configuration_candidate_id"]],
            "configurations_in_program_candidate": mmy_config_counts[first["program_candidate_id"]],
            "max_source_rows_in_one_run": max(run_row_counts[run_id] for run_id in run_ids),
            "grouping_rule": "Same provisional configuration signature + exact Target ABC + exact ETW",
            "status": "PROVISIONAL_GROUP",
            "unresolved": "Test Vehicle ID/Configuration # are source identity only; Set ABC/procedure remain RUN evidence.",
        })
    return rows, pd.DataFrame(preview).sort_values(
        ["make", "model", "model_year", "configuration_candidate_id", "vde_candidate_id"],
        kind="stable",
    ).reset_index(drop=True)


def historical_refresh_delta(old: pd.DataFrame, refreshed: pd.DataFrame) -> list[dict[str, Any]]:
    hist = refreshed[refreshed["Model Year"].between(2020, 2025)].copy()
    shared_columns = list(old.columns)
    old_hashes = c1.source_row_hashes(old, shared_columns)
    new_hashes = c1.source_row_hashes(hist, shared_columns)
    rows = [
        {"slice": "2020-2025", "metric": "source_rows", "old_count": len(old), "refreshed_count": len(hist), "delta": len(hist) - len(old), "status": "REFRESH_CHANGED", "evidence": "Shared model-year slice."},
        {"slice": "2020-2025", "metric": "unique_shared_projection_hashes", "old_count": len(old_hashes), "refreshed_count": len(new_hashes), "delta": len(new_hashes) - len(old_hashes), "status": "REFRESH_CHANGED", "evidence": f"overlap={len(old_hashes & new_hashes)}; old_only={len(old_hashes-new_hashes)}; refreshed_only={len(new_hashes-old_hashes)}"},
        {"slice": "2020-2025", "metric": "raw_mmy_groups", "old_count": old[list(MMY_FIELDS)].drop_duplicates().shape[0], "refreshed_count": hist[list(MMY_FIELDS)].drop_duplicates().shape[0], "delta": hist[list(MMY_FIELDS)].drop_duplicates().shape[0] - old[list(MMY_FIELDS)].drop_duplicates().shape[0], "status": "REFRESH_CHANGED", "evidence": "Exact raw Make+Model+Year tuples."},
    ]
    old_years = old["Model Year"].value_counts().to_dict()
    new_years = hist["Model Year"].value_counts().to_dict()
    for year in range(2020, 2026):
        before, after = int(old_years.get(year, 0)), int(new_years.get(year, 0))
        rows.append({"slice": str(year), "metric": "source_rows", "old_count": before, "refreshed_count": after, "delta": after - before, "status": "UNCHANGED" if before == after else "REFRESH_CHANGED", "evidence": "Model Year row count."})
    return rows


def population_funnel(
    old: pd.DataFrame,
    refreshed_hist_rows: pd.DataFrame,
    refreshed_hist_preview: pd.DataFrame,
    epa_rows: pd.DataFrame,
    epa_preview: pd.DataFrame,
    jrc: pd.DataFrame,
    core_vde: pd.DataFrame,
) -> list[dict[str, Any]]:
    legacy_programs = core_vde[["make", "model", "year"]].drop_duplicates().shape[0]
    legacy_configs = 4_996  # Sprint 12C exact materialization evidence.
    return [
        {"source_population": "Legacy EPA 2020-2025 source", "source_rows": len(old), "run_candidates": "4,999-26,262", "vde_candidates": "4,999", "vehicle_configuration_candidates": str(legacy_configs), "program_candidates": str(legacy_programs), "fuelcons_candidates": "4,999", "grain_rule": "Preserve 4,999 adopted MMY VDE/FuelCons; raw source rows may become RUN evidence.", "confidence": "HIGH for retained VDE/FuelCons; PARTIAL for RUN", "unresolved_issues": "Original source-row/test IDs were not persisted."},
        {"source_population": "EPA refreshed historical 2020-2025", "source_rows": len(refreshed_hist_rows), "run_candidates": f"{refreshed_hist_rows['run_candidate_id'].nunique():,}-{len(refreshed_hist_rows):,}", "vde_candidates": f"{len(refreshed_hist_preview):,}", "vehicle_configuration_candidates": f"{refreshed_hist_rows['configuration_candidate_id'].nunique():,}", "program_candidates": f"{refreshed_hist_rows['program_candidate_id'].nunique():,}", "fuelcons_candidates": f"{len(refreshed_hist_preview):,}-{len(refreshed_hist_rows):,}", "grain_rule": "Preview signature only; refresh replaces/version-controls historical source evidence.", "confidence": "PROVISIONAL", "unresolved_issues": "31 net new rows and shared-column hash changes; reload policy required."},
        {"source_population": "EPA Test Car MY2026", "source_rows": len(epa_rows), "run_candidates": f"{epa_rows['run_candidate_id'].nunique():,}-{len(epa_rows):,}", "vde_candidates": f"{len(epa_preview):,}-{len(epa_rows):,}", "vehicle_configuration_candidates": f"{epa_rows['configuration_candidate_id'].nunique():,}-{len(epa_rows):,}", "program_candidates": f"{epa_rows['program_candidate_id'].nunique():,}-{epa_rows[list(MMY_FIELDS)].drop_duplicates().shape[0]:,}", "fuelcons_candidates": f"{len(epa_preview):,}-{len(epa_rows):,}", "grain_rule": "Program=source-scoped MMY; VC=strict signature; VDE=VC+Target ABC+ETW; RUN=Test Number+Testgroup.", "confidence": "PROVISIONAL", "unresolved_issues": "Source identity anomalies; universal configuration identity and result adoption unresolved."},
        {"source_population": "Current post-import scenario/ML", "source_rows": "N/A", "run_candidates": "5", "vde_candidates": "4", "vehicle_configuration_candidates": "0 additive", "program_candidates": "0 additive", "fuelcons_candidates": "5", "grain_rule": "Derived VDE references parent configuration/program; later result becomes explicit RUN/FuelCons evidence.", "confidence": "HIGH", "unresolved_issues": "Current record_origin incorrectly says LEGACY."},
        {"source_population": "JRC technical dataset", "source_rows": len(jrc), "run_candidates": "249", "vde_candidates": "249", "vehicle_configuration_candidates": "249", "program_candidates": "249", "fuelcons_candidates": "249", "grain_rule": "One source-scoped provisional lineage per unresolved source row; no cross-source merge.", "confidence": "PARTIAL/UNRESOLVED", "unresolved_issues": "Anonymized identity and exact real-vehicle/archetype/simulation grain."},
        {"source_population": "EEA 2025 provisional", "source_rows": "10,833,597", "run_candidates": "UNRESOLVED-10,833,597 reporting", "vde_candidates": "0 engineering", "vehicle_configuration_candidates": "0 engineering", "program_candidates": "0 engineering", "fuelcons_candidates": "UNRESOLVED-10,833,597 reporting", "grain_rule": "Monitoring/declared-result evidence remains separate from engineering VDE/configuration grain.", "confidence": "PARTIAL", "unresolved_issues": "Reporting aggregation/adoption key; RLFI semantics unresolved."},
    ]


def legacy_future_mapping(vde: pd.DataFrame, fuelcons: pd.DataFrame, duplicates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    core = vde[vde["cycle_source"].eq("standard:EPA")]
    post_vde = vde[~vde["cycle_source"].eq("standard:EPA")]
    core_fc = fuelcons[fuelcons["label_program"].eq("EPA")]
    post_fc = fuelcons[~fuelcons["label_program"].eq("EPA")]
    return [
        {"legacy_population": "4,999 historical VDE", "future_entity": "PROGRAM", "future_rows_or_range": str(core[["make", "model", "year"]].drop_duplicates().shape[0]), "mapping_rule": "Source-scoped exact persisted MMY fallback; no cross-year merge without evidence.", "provenance": "LEGACY_EPA_RECONSTRUCTED", "status": "PROVISIONAL"},
        {"legacy_population": "4,999 historical VDE", "future_entity": "VEHICLE_CONFIGURATION", "future_rows_or_range": "4,996", "mapping_rule": "Sprint 12C stable legacy technical signature; target ABC/mass excluded from configuration identity.", "provenance": "LEGACY_EPA_RECONSTRUCTED", "status": "PROVISIONAL"},
        {"legacy_population": "4,999 historical VDE", "future_entity": "VDE", "future_rows_or_range": str(len(core)), "mapping_rule": f"Preserve every historical VDE, including {len(duplicates)} duplicate MMY identity groups.", "provenance": "SOURCE_LIKELY_AGGREGATED", "status": "KEEP_SEPARATE"},
        {"legacy_population": "4,999 historical FuelCons", "future_entity": "FUELCONS", "future_rows_or_range": str(len(core_fc)), "mapping_rule": "One retained adopted comparison result per legacy VDE.", "provenance": "DETERMINISTIC_FROM_SOURCE_TEST_RESULTS", "status": "PRESERVE"},
        {"legacy_population": "Legacy source evidence", "future_entity": "RUN", "future_rows_or_range": "4,999-26,262", "mapping_rule": "Minimum one adopted-result lineage per VDE; upper bound preserves each source row as evidence.", "provenance": "SOURCE_ROW_ID_UNAVAILABLE_IN_DB", "status": "PARTIAL"},
        {"legacy_population": "2,958 component decompositions", "future_entity": "COMPONENT_RESOLUTION", "future_rows_or_range": "0-2,958", "mapping_rule": "Optional estimated/calculated resolution evidence; never measured EPA component ABC.", "provenance": "ESTIMATED_DEFAULT_PRIOR_SPLIT", "status": "REVIEW_REQUIRED"},
        {"legacy_population": "4 derived/scenario VDE", "future_entity": "VDE", "future_rows_or_range": str(len(post_vde)), "mapping_rule": "Retain as descendants of parent configuration/VDE; no new Program required.", "provenance": "SCENARIO_OR_TEST_DERIVATIVE", "status": "PRESERVE"},
        {"legacy_population": "5 later FuelCons", "future_entity": "RUN + FUELCONS", "future_rows_or_range": str(len(post_fc)), "mapping_rule": "Retain scenario/regression/ML evidence and adopted result separately.", "provenance": "SCENARIO/ML; CURRENT LABEL NEEDS CORRECTION", "status": "PRESERVE"},
        {"legacy_population": "Sparse tire record", "future_entity": "TIRE_DB", "future_rows_or_range": "1", "mapping_rule": "Preserve current specialized tire evidence; do not fabricate instances.", "provenance": "LEGACY_TIRE_EVIDENCE", "status": "PRESERVE"},
    ]


def entity_forecast(
    epa_rows: pd.DataFrame, epa_preview: pd.DataFrame, jrc: pd.DataFrame
) -> list[dict[str, Any]]:
    epa_program_min = int(epa_rows["program_candidate_id"].nunique())
    epa_program_max = int(epa_rows[list(MMY_FIELDS)].drop_duplicates().shape[0])
    epa_config_min = int(epa_rows["configuration_candidate_id"].nunique())
    epa_vde_min = len(epa_preview)
    jrc_tires = int(jrc["Tyre code"].nunique(dropna=True))
    jrc_component_upper = int(jrc["Engine max power"].notna().sum()) + int(jrc["Electric motor power [kW]"].notna().sum()) + int(jrc["Drive battery capacity [Ah]"].notna().sum())
    jrc_instances = len(jrc) * 3 + int(jrc["Electric motor power [kW]"].notna().sum()) + int(jrc["Drive battery capacity [Ah]"].notna().sum())
    epa_instances_likely = epa_config_min * 3
    epa_instances_upper = len(epa_rows) * 3
    rows = [
        ("PROGRAM", 4_953, epa_program_min + 249, f"{4_953 + epa_program_min + 249:,}-{4_953 + epa_program_max + 249:,}", 4_953 + epa_program_max + 249, "0", "Legacy source-scoped MMY + EPA2026 preview Programs + one JRC source-scoped Program per unresolved row."),
        ("VEHICLE_CONFIGURATION", 4_996, epa_config_min + 249, f"{4_996 + epa_config_min + 249:,}-{4_996 + len(epa_rows) + 249:,}", 4_996 + len(epa_rows) + 249, "0", "Strict EPA signature is the likely lower grouping; source rows are the safe upper bound."),
        ("COMPONENT_DB", 0, 0, f"0-{jrc_component_upper:,}", jrc_component_upper, "0", "Reusable component identity is absent; JRC descriptor rows are an upper source-scoped candidate bound."),
        ("TIRE_DB", 1, jrc_tires, f"{1 + jrc_tires:,}-{1 + len(jrc):,}", 1 + len(jrc), "0", "70 distinct JRC tire codes are likely provisional definitions; one per row is the unresolved upper bound."),
        ("COMPONENT_INSTANCE", 0, 0, f"0-{epa_instances_likely + jrc_instances:,} likely if partial instances are adopted", epa_instances_upper + jrc_instances, "0", "Optional partial engine/transmission/driveline/tire/electric instances; never required for VDE validity."),
        ("COMPONENT_RESOLUTION", 0, 0, "0-2,958", 2_958, "0", "Legacy estimated decomposition may be retained as optional resolution evidence; public Tier-0 adds none."),
        ("VDE", 5_003, epa_vde_min + 249, f"{5_003 + epa_vde_min + 249:,}-{5_003 + len(epa_rows) + 249:,}", 5_003 + len(epa_rows) + 249, "0", "Legacy states preserved; EPA VC+Target+ETW preview lower bound; one EPA VDE per source row upper bound; JRC source-scoped."),
        ("RUN", 5_004, int(epa_rows["run_candidate_id"].nunique()) + 249, f"{5_004 + int(epa_rows['run_candidate_id'].nunique()) + 249:,}-30,448", 30_448, "0-10,833,597", "Lower uses current result lineage + EPA test candidates + JRC. Upper also preserves refreshed historical source rows. EEA reporting dominates separately."),
        ("FUELCONS", 5_004, epa_vde_min + 249, f"{5_004 + epa_vde_min + 249:,}-{5_004 + len(epa_rows) + 249:,}", 5_004 + len(epa_rows) + 249, "UNRESOLVED-10,833,597", "Engineering adopted-result count depends on EPA result adoption; EEA remains reporting-grain and separate."),
    ]
    return [
        {"entity": entity, "legacy_backed_minimum": legacy, "public_engineering_additive_minimum": additive, "likely_engineering_count_or_range": likely, "engineering_upper_bound": upper, "eea_reporting_additive_range": eea, "reason": reason}
        for entity, legacy, additive, likely, upper, eea, reason in rows
    ]


def collision_report(
    core: pd.DataFrame,
    duplicates: list[dict[str, Any]],
    epa_rows: pd.DataFrame,
    epa_preview: pd.DataFrame,
    refresh: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for duplicate in duplicates:
        rows.append({
            "collision_type": "LEGACY_DUPLICATED_MMY",
            "scope_key": f"{duplicate['make']}|{duplicate['model']}|{duplicate['year']}",
            "affected_rows": duplicate["vde_rows"],
            "classification": "KEEP_SEPARATE" if duplicate["target_abc_variants"] > 1 else "UNRESOLVED",
            "evidence": f"VDE ids={duplicate['vde_ids']}; Target-ABC variants={duplicate['target_abc_variants']}.",
            "required_action": "Preserve both VDE states; resolve configuration/source identity later.",
        })
    suspicious = epa_rows[epa_rows["Represented Test Veh Make"].astype(str).str.fullmatch(r"\d{4}")]
    for value, group in suspicious.groupby(suspicious["Represented Test Veh Make"].astype(str)):
        rows.append({"collision_type": "EPA_SUSPICIOUS_MAKE", "scope_key": value, "affected_rows": len(group), "classification": "SOURCE_DATA_QUALITY_ISSUE", "evidence": f"Vehicle Manufacturer Name values: {';'.join(sorted(map(str, group['Vehicle Manufacturer Name'].dropna().unique())))}", "required_action": "Preserve raw value; validate source mapping before Program identity."})

    identity_fields = ["Test Vehicle ID", "Test Veh Configuration #"]
    descriptor_fields = ["Test Veh Displacement (L)", "Engine Code", "Tested Transmission Type", "# of Gears", "Drive System Description", "Axle Ratio", "N/V Ratio"]
    for key, group in epa_rows.groupby(identity_fields, dropna=False, sort=True):
        descriptor_variants = group[descriptor_fields].drop_duplicates().shape[0]
        if descriptor_variants > 1:
            rows.append({"collision_type": "EPA_SOURCE_ID_DESCRIPTOR_CONFLICT", "scope_key": "|".join(map(str, key if isinstance(key, tuple) else (key,))), "affected_rows": len(group), "classification": "KEEP_SEPARATE", "evidence": f"{descriptor_variants} stable-descriptor signatures under the same Test Vehicle ID/Configuration #.", "required_action": "Do not use source IDs alone as canonical Vehicle Configuration identity."})
    if not any(row["collision_type"] == "EPA_SOURCE_ID_DESCRIPTOR_CONFLICT" for row in rows):
        rows.append({"collision_type": "EPA_SOURCE_ID_DESCRIPTOR_CONFLICT", "scope_key": "ALL_EPA2026", "affected_rows": 0, "classification": "SAFE_GROUP", "evidence": "No Test Vehicle ID/Configuration # group had multiple stable-descriptor signatures under the current preview fields.", "required_action": "Keep the key provisional; absence of an observed conflict does not make it universal identity."})

    for row in epa_preview.drop_duplicates("configuration_candidate_id").itertuples(index=False):
        if row.vde_states_in_configuration > 1:
            rows.append({"collision_type": "EPA_CONFIGURATION_MULTIPLE_VDE_STATES", "scope_key": row.configuration_candidate_id, "affected_rows": row.vde_states_in_configuration, "classification": "KEEP_SEPARATE", "evidence": "Same preview configuration signature has multiple Target-ABC+ETW states.", "required_action": "Keep one Vehicle Configuration with multiple VDE candidates."})
    for row in epa_preview.itertuples(index=False):
        if row.run_candidates > 1:
            rows.append({"collision_type": "EPA_MULTIPLE_RUNS_ONE_VDE", "scope_key": row.vde_candidate_id, "affected_rows": row.run_candidates, "classification": "PROVISIONAL_GROUP", "evidence": "Multiple Test Number+Testgroup candidates share the preview VDE state.", "required_action": "Preserve all RUN evidence; adoption rule remains separate."})

    run_state_counts = epa_rows.groupby("run_candidate_id")["vde_candidate_id"].nunique()
    for run_id, count in run_state_counts[run_state_counts > 1].items():
        rows.append({"collision_type": "EPA_RUN_MULTIPLE_VDE_STATES", "scope_key": run_id, "affected_rows": int(count), "classification": "UNRESOLVED", "evidence": "One Test Number+Testgroup candidate maps to multiple preview VDE states.", "required_action": "Retain source rows; review whether RUN key needs additional result/test context."})

    changed = next(row for row in refresh if row["metric"] == "unique_shared_projection_hashes")
    rows.append({"collision_type": "EPA_HISTORICAL_REFRESH_HASH_DELTA", "scope_key": "2020-2025", "affected_rows": "124 old-only + 155 refreshed-only hashes", "classification": "SOURCE_DATA_QUALITY_ISSUE", "evidence": changed["evidence"], "required_action": "Version the source snapshot and define replace/supersede behavior; do not blind-append."})
    return sorted(rows, key=lambda row: (row["collision_type"], row["scope_key"]))


def example_tree_from_preview(row: pd.Series, label: str) -> str:
    run_ids = str(row["run_candidate_ids"]).split(";") if row["run_candidate_ids"] else []
    branches = "\n".join(f"      {'└─' if i == len(run_ids)-1 else '├─'} {run_id}" for i, run_id in enumerate(run_ids[:4]))
    if len(run_ids) > 4:
        branches += f"\n      └─ … {len(run_ids)-4} more RUN candidates"
    return (
        f"{row['make']} | {row['model']} | {row['model_year']} [{label}]\n"
        f"└─ {row['program_candidate_id']}\n"
        f"   └─ {row['configuration_candidate_id']} ({row['vde_states_in_configuration']} VDE state(s))\n"
        f"      └─ {row['vde_candidate_id']} Target=({row['target_a_lbf']}, {row['target_b_lbf_per_mph']}, {row['target_c_lbf_per_mph2']}) ETW={row['etw_lb']}\n"
        f"{branches}"
    )


def select_examples(
    epa_rows: pd.DataFrame,
    preview: pd.DataFrame,
    duplicates: list[dict[str, Any]],
    extras: list[dict[str, Any]],
    refresh: list[dict[str, Any]],
    jrc: pd.DataFrame,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []

    def add(category: str, key: str, population: str, tree: str, evidence: str) -> None:
        examples.append({"example_id": f"EX-{len(examples)+1:02d}", "category": category, "source_key": key, "population": population, "tree": tree, "evidence": evidence})

    simple = preview[(preview["configurations_in_program_candidate"] == 1) & (preview["vde_states_in_configuration"] == 1) & (preview["run_candidates"] == 1)].head(3)
    multi_config = preview[preview["configurations_in_program_candidate"] > 1].drop_duplicates("program_candidate_id").head(4)
    multi_state = preview[preview["vde_states_in_configuration"] > 1].drop_duplicates("configuration_candidate_id").head(4)
    multi_run = preview[preview["run_candidates"] > 1].head(3)
    multi_source_run_ids = set(epa_rows.groupby("run_candidate_id").size().loc[lambda s: s > 1].index)
    multi_source_run = preview[preview["run_candidate_ids"].map(lambda value: bool(set(str(value).split(";")) & multi_source_run_ids))].head(2)
    for label, frame in (("ONE_MMY_ONE_CONFIG_ONE_VDE", simple), ("MMY_MULTIPLE_CONFIGS", multi_config), ("CONFIG_MULTIPLE_VDE_STATES", multi_state), ("MULTIPLE_RUNS_ONE_VDE", multi_run), ("MULTIPLE_SOURCE_ROWS_ONE_RUN", multi_source_run)):
        for _, row in frame.iterrows():
            add(label, row["vde_candidate_id"], "EPA Test Car MY2026", example_tree_from_preview(row, label), f"source_rows={row['source_rows']}; run_candidates={row['run_candidates']}; configurations_in_program={row['configurations_in_program_candidate']}; vde_states_in_configuration={row['vde_states_in_configuration']}")

    for duplicate in duplicates[:3]:
        tree = f"{duplicate['make']} | {duplicate['model']} | {duplicate['year']}\n└─ persisted MMY identity\n   ├─ VDE {duplicate['vde_ids'].split(';')[0]}\n   └─ VDE {duplicate['vde_ids'].split(';')[1]} ({duplicate['target_abc_variants']} Target-ABC variants)"
        add("LEGACY_DUPLICATE_MMY", duplicate["vde_ids"], "Legacy EPA 2020-2025", tree, duplicate["interpretation"])

    refresh_row = next(row for row in refresh if row["metric"] == "source_rows" and row["slice"] == "2025")
    add("HISTORICAL_REFRESH", "EPA-2025", "EPA refreshed historical", f"EPA MY2025 refresh\n├─ old source rows: {refresh_row['old_count']}\n└─ refreshed source rows: {refresh_row['refreshed_count']} (delta {refresh_row['delta']:+d})", refresh_row["evidence"])

    scenario = next(row for row in extras if row["classification"] == "POST_IMPORT_SCENARIO_OR_TEST_DERIVATIVE")
    add("DERIVED_SCENARIO_VDE", str(scenario["vde_id"]), "Current post-import", f"Parent VDE {scenario['parent_vde_id']}\n└─ derived VDE {scenario['vde_id']} | {scenario['make']} {scenario['model']} {scenario['year']}\n   └─ {scenario['fuelcons_rows']} FuelCons row(s)", scenario["evidence"])
    ml = next(row for row in extras if row["classification"] == "POST_IMPORT_ML_FUELCONS_ON_LEGACY_VDE")
    add("ML_FUELCONS", str(ml["vde_id"]), "Current post-import", f"Legacy VDE {ml['vde_id']}\n└─ RUN(ML_PREDICTION) candidate\n   └─ later FuelCons result", ml["evidence"])

    jr = jrc.sort_values(["OEM anon", "Model anon"], kind="stable").iloc[0]
    add("JRC_SOURCE_SCOPED", "JRC-row", "JRC", f"{clean(jr['OEM anon'])} | {clean(jr['Model anon'])}\n└─ source-scoped Program/Configuration (UNRESOLVED)\n   ├─ VDE WLTP f0/f1/f2\n   ├─ RUN {clean(jr['pycsis_run'])}\n   └─ FuelCons declared/simulated evidence", "Anonymized identity; explicit mass/gearbox/gears/tire and roadload labels.")

    with c1.EEA_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        eea = next(csv.DictReader(handle))
    add("EEA_MONITORING", str(eea.get("ID")), "EEA", f"EEA record {eea.get('ID')} | {eea.get('Mk')} {eea.get('Cn')} | {eea.get('year')}\n└─ RUN(MONITORING/DECLARED_RESULT) candidate\n   └─ FuelCons reporting evidence\n      └─ no engineering VDE/Configuration implied", "Direct reporting fields; RLFI remains unresolved.")
    return examples


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{key: clean(value) for key, value in row.items()} for row in rows])


def md_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join("---" for _ in fields) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")).replace("|", "/").replace("\n", "<br>") for field in fields) + " |")
    return lines


def tree_report(examples: list[dict[str, Any]]) -> str:
    lines = ["# Sprint 12C.2 — Real Population Tree Examples", "", "All identifiers are deterministic preview identifiers only; none were written to a runtime database.", ""]
    for example in examples:
        lines += [f"## {example['example_id']} — {example['category']}", "", f"Population: `{example['population']}` · Source key: `{example['source_key']}`", "", "```text", example["tree"], "```", "", f"Evidence: {example['evidence']}", ""]
    return "\n".join(lines)


def report_text(payload: dict[str, Any]) -> str:
    funnel = payload["population_funnel"]
    forecast = payload["entity_count_forecast"]
    mapping = payload["legacy_future_mapping"]
    examples = payload["population_examples"]
    collision_counts = Counter(row["classification"] for row in payload["collision_report"])
    preview = payload["epa2026_summary"]
    lines = [
        "# Sprint 12C.2 — Canonical Population Preview",
        "",
        f"## Status: `{payload['status']}`",
        "",
        "The future engineering database is likely to remain in the **thousands to low tens of thousands**, not millions. EEA can add millions of RUN/FuelCons reporting records, but it does not imply millions of Programs, Configurations or VDEs.",
        "",
        "This is a read-only, pre-DDL grouping preview. Candidate identifiers are deterministic audit labels only. The runtime database remained byte-identical; no schema, migration, application, physics, resolver or raw source changed.",
        "",
        "## Population funnel",
        "",
        "```text",
        "SOURCE ROWS → RUN candidates → VDE states → VEHICLE_CONFIGURATION → PROGRAM → FUELCONS",
        "```",
        "",
        *md_table(funnel, ["source_population", "source_rows", "run_candidates", "vde_candidates", "vehicle_configuration_candidates", "program_candidates", "fuelcons_candidates", "confidence"]),
        "",
        "Full grain rules and unresolved issues are retained in `population_funnel.csv`.",
        "",
        "## EPA MY2026 preview",
        "",
        f"The 3,901 source rows produce **{preview['program_candidates']:,}–{preview['raw_mmy_groups']:,} Program candidates**, **{preview['configuration_candidates']:,}–3,901 Configuration candidates**, **{preview['vde_candidates']:,}–3,901 VDE candidates**, and **{preview['run_candidates']:,}–3,901 RUN candidates** under the current provisional rules.",
        "",
        "The lower preview groups Program by source-scoped normalized MMY, Configuration by the existing strict technical/source signature, VDE by Configuration+Target ABC+ETW, and RUN by Test Number+Testgroup. None of those keys is promoted to universal canonical identity.",
        "",
        f"Observed topology: **{preview['configurations_with_multiple_vdes']:,}** candidate Configurations have multiple VDE states; **{preview['vdes_with_multiple_runs']:,}** VDE candidates have multiple RUN candidates; **{preview['runs_with_multiple_vdes']:,}** RUN candidates touch multiple preview VDE states and remain unresolved.",
        "",
        "## Nine-entity forecast",
        "",
        *md_table(forecast, ["entity", "legacy_backed_minimum", "public_engineering_additive_minimum", "likely_engineering_count_or_range", "engineering_upper_bound", "eea_reporting_additive_range", "reason"]),
        "",
        "The two growth drivers are different: EPA/JRC expand engineering state/evidence into thousands or low tens of thousands; EEA expands reporting evidence into millions only if row-level monitoring is retained.",
        "",
        "## Legacy-to-future representation",
        "",
        *md_table(mapping, ["legacy_population", "future_entity", "future_rows_or_range", "mapping_rule", "provenance", "status"]),
        "",
        "The 46 duplicated legacy MMY identities remain represented as separate VDE states. The 44 groups with different Target ABC are classified `KEEP_SEPARATE`; the other two remain `UNRESOLVED`, not auto-merged.",
        "",
        "## Grouping examples",
        "",
        f"The audit generated **{len(examples)} deterministic real examples**. A compact selection follows; the full trees are in `etl/reports/sprint_12c2_population_tree_examples.md`.",
        "",
        *md_table(examples[:8], ["example_id", "category", "population", "source_key", "evidence"]),
        "",
        "## Collision and ambiguity summary",
        "",
        *md_table([{"classification": key, "cases": value} for key, value in sorted(collision_counts.items())], ["classification", "cases"]),
        "",
        "`collision_report.csv` includes every legacy duplicated MMY, suspicious EPA Make value, source-ID/descriptor conflict, multi-state configuration, multi-RUN VDE, RUN-to-multiple-VDE collision and historical refresh hash delta.",
        "",
        "## Historical refresh",
        "",
        *md_table(payload["historical_refresh_delta"], ["slice", "metric", "old_count", "refreshed_count", "delta", "status", "evidence"]),
        "",
        "The refreshed file is not a pure 2026 append. MY2025 gains 31 source rows and the 2020–2025 shared projection contains both removed/changed and new/changed row hashes. A versioned supersede/refresh rule is required.",
        "",
        "## Evidence tiers and remaining gaps",
        "",
        "- **Directly tested:** read-only/byte-identical database access; deterministic EPA counts and example selection; all 46 duplicate legacy groups retained; NULL distinct from zero; output scope limited to `etl/` and sprint documentation.",
        "- **Indirectly covered:** exact 12C reconstruction of the current VDE/FuelCons application surface and 12C.1 provenance classification.",
        "- **Inspection-supported:** legacy notebook grouping/decomposition logic and CDR source-scoped identity guardrails.",
        "- **Gap:** final EPA configuration/adoption key, exact JRC grain, EEA aggregation/adoption key and EEA RLFI semantics remain unresolved by design.",
        "",
        "## Recommendation",
        "",
        "Population shape is sufficiently visible to start notebooks and physical DDL design using ranges and explicit unresolved states. DDL implementation/migration remains unauthorized in this sprint. The 12D design must not encode provisional preview keys as universal identities.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        "python etl/scripts/sprint_12c2_canonical_population_preview.py",
        "python -m unittest discover -s etl/tests -p \"test_sprint_12c2*.py\" -v",
        "```",
        "",
        "## Outputs",
        "",
        *[f"- `{str(path.relative_to(ROOT)).replace(chr(92), '/')}`" for path in OUTPUT_PATHS],
    ]
    return "\n".join(lines) + "\n"


def main() -> dict[str, Any]:
    required = [DB_PATH, c1.LEGACY_EPA_PATH, c1.EPA_PATH, c1.JRC_PATH, c1.EEA_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing preview inputs: {missing}")
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    db_hash_before = sha256(DB_PATH)
    vde, fuelcons = read_db()
    old = pd.read_excel(c1.LEGACY_EPA_PATH, engine="openpyxl")
    refreshed = pd.read_excel(c1.EPA_PATH, sheet_name="Sheet1", engine="openpyxl")
    jrc = pd.read_excel(c1.JRC_PATH, sheet_name="Sheet1", engine="openpyxl")
    epa_2026 = refreshed[refreshed["Model Year"].eq(2026)].copy()
    refreshed_hist = refreshed[refreshed["Model Year"].between(2020, 2025)].copy()
    epa_rows, epa_preview = prepare_epa_candidates(epa_2026)
    hist_rows, hist_preview = prepare_epa_candidates(refreshed_hist)
    core = vde[vde["cycle_source"].eq("standard:EPA")].copy()
    duplicates = c1.legacy_duplicate_groups(core)
    extras = c1.extra_records(vde, fuelcons)
    refresh_delta = historical_refresh_delta(old, refreshed)
    funnel = population_funnel(old, hist_rows, hist_preview, epa_rows, epa_preview, jrc, core)
    mapping = legacy_future_mapping(vde, fuelcons, duplicates)
    forecast = entity_forecast(epa_rows, epa_preview, jrc)
    collisions = collision_report(core, duplicates, epa_rows, epa_preview, refresh_delta)
    examples = select_examples(epa_rows, epa_preview, duplicates, extras, refresh_delta, jrc)
    db_hash_after = sha256(DB_PATH)

    if db_hash_before != db_hash_after:
        raise RuntimeError("Runtime database changed during the read-only preview.")
    if len(epa_rows) != 3_901 or epa_rows["configuration_candidate_id"].nunique() != 1_291:
        raise RuntimeError("EPA preview no longer matches the audited source population.")
    if len(duplicates) != 46 or sum(row["target_abc_variants"] > 1 for row in duplicates) != 44:
        raise RuntimeError("Legacy duplicate VDE-state preservation invariant failed.")
    if len(examples) < 20:
        raise RuntimeError(f"Only {len(examples)} stable examples were selected; at least 20 are required.")

    config_state_counts = epa_rows.groupby("configuration_candidate_id")["vde_candidate_id"].nunique()
    run_state_counts = epa_rows.groupby("run_candidate_id")["vde_candidate_id"].nunique()
    summary = {
        "source_rows": len(epa_rows),
        "raw_mmy_groups": epa_2026[list(MMY_FIELDS)].drop_duplicates().shape[0],
        "program_candidates": epa_rows["program_candidate_id"].nunique(),
        "configuration_candidates": epa_rows["configuration_candidate_id"].nunique(),
        "vde_candidates": epa_rows["vde_candidate_id"].nunique(),
        "run_candidates": epa_rows["run_candidate_id"].nunique(),
        "configurations_with_multiple_vdes": int((config_state_counts > 1).sum()),
        "vdes_with_multiple_runs": int((epa_preview["run_candidates"] > 1).sum()),
        "runs_with_multiple_vdes": int((run_state_counts > 1).sum()),
    }
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "POPULATION_SHAPE_CLEAR — PROCEED_TO_NOTEBOOKS_AND_12D",
        "scope": "Read-only population/grain preview; DDL implementation and migration unauthorized.",
        "database_access": "SQLite URI mode=ro + PRAGMA query_only=ON",
        "database_sha256_before": db_hash_before,
        "database_sha256_after": db_hash_after,
        "database_byte_identical": True,
        "epa2026_summary": summary,
        "population_funnel": funnel,
        "entity_count_forecast": forecast,
        "legacy_future_mapping": mapping,
        "population_examples": examples,
        "collision_report": collisions,
        "historical_refresh_delta": refresh_delta,
        "evidence_tiers": {
            "directly_tested": ["read-only DB", "deterministic grouping", "duplicate preservation", "NULL vs zero", "stable examples", "output scope"],
            "indirectly_covered": ["Sprint 12C compatibility", "Sprint 12C.1 provenance audit"],
            "inspection_supported": ["legacy ETL notebook", "CDR source-scoped identity rules"],
            "gaps": ["final EPA identity/adoption", "JRC row grain", "EEA aggregation and RLFI semantics"],
        },
    }

    write_csv(OUT / "population_funnel.csv", funnel)
    write_csv(OUT / "entity_count_forecast.csv", forecast)
    write_csv(OUT / "epa2026_grouping_preview.csv", epa_preview.to_dict("records"))
    write_csv(OUT / "legacy_future_mapping.csv", mapping)
    write_csv(OUT / "population_examples.csv", examples)
    write_csv(OUT / "collision_report.csv", collisions)
    write_csv(OUT / "historical_refresh_delta.csv", refresh_delta)
    (OUT / "population_preview.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=clean), encoding="utf-8")
    REPORT.write_text(report_text(payload), encoding="utf-8")
    TREE_REPORT.write_text(tree_report(examples), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "epa2026_summary": summary,
        "examples": len(examples),
        "collisions": len(collisions),
        "database_byte_identical": True,
        "report": str(REPORT.relative_to(ROOT)),
    }, indent=2, ensure_ascii=False))
    return payload


if __name__ == "__main__":
    main()
