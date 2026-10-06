"""Sprint 12C.1: read-only dataset delta and engineering coverage audit.

The runtime SQLite database and all source files are opened read-only. Outputs
are audit CSV/JSON files plus a Markdown report; no DDL or migration is run.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sqlite3
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "db" / "eco_drive.db"
LEGACY_EPA_PATH = ROOT / "data" / "vehicles" / "testcar-2025-2020-EPA.xlsx"
EPA_PATH = ROOT / "etl" / "data" / "raw" / "epa_testcar" / "epa_testcar_2026_raw.xlsx"
JRC_PATH = ROOT / "etl" / "data" / "raw" / "wltp_jrc" / "Data_PV_fleet_2021_EU_PYCSIS.xlsx"
EEA_PATH = ROOT / "etl" / "data" / "raw" / "wltp_eea" / "data.csv"
FIELD_INVENTORY = ROOT / "etl" / "data" / "processed" / "sprint_12a_audit" / "field_inventory.csv"
GRAIN_12A = ROOT / "etl" / "data" / "processed" / "sprint_12a_audit" / "epa_2026_grain.csv"
NOTEBOOK = ROOT / "notebooks" / "etl_epa_xlsx_to_sqlite.ipynb"
OUT = ROOT / "etl" / "data" / "processed" / "sprint_12c1_dataset_delta_audit"
REPORT = ROOT / "etl" / "reports" / "sprint_12c1_dataset_delta_audit.md"


def clean(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    return value


def present(value: Any) -> bool:
    value = clean(value)
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def norm_text(value: Any) -> str:
    if not present(value):
        return ""
    text = unicodedata.normalize("NFKD", str(clean(value))).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Z0-9]+", " ", text.upper()).strip()


def pct(count: int, total: int) -> float:
    return round(100.0 * count / total, 3) if total else 0.0


def complete_count(df: pd.DataFrame, fields: Iterable[str]) -> int:
    fields = list(fields)
    if not fields or any(field not in df.columns for field in fields):
        return 0
    return int(df[fields].notna().all(axis=1).sum())


def any_count(df: pd.DataFrame, fields: Iterable[str]) -> int:
    fields = [field for field in fields if field in df.columns]
    return int(df[fields].notna().any(axis=1).sum()) if fields else 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def db_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        con.execute("PRAGMA query_only = ON")
        vde = pd.read_sql_query("SELECT * FROM vde_db ORDER BY id", con)
        fuelcons = pd.read_sql_query("SELECT * FROM fuelcons_db ORDER BY id", con)
    finally:
        con.close()
    return vde, fuelcons


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    path = OUT / name
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean(value) for key, value in row.items()})


def notebook_evidence() -> dict[str, Any]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = ["".join(cell.get("source", [])) for cell in notebook.get("cells", [])]
    all_source = "\n".join(sources)
    checks = {
        "groups_by_make_model_year": 'df.groupby("_veh_key", sort=False)' in all_source,
        "vde_target_abc_averaged": 'g["Target Coef A (lbf)"].mean()' in all_source,
        "mass_from_etw_lookup": 'df["mass_kg"] = df["inertia_class"].apply(max_mass_from_inertia_class)' in all_source,
        "transmission_normalized": 'df_vde["transmission_type"].apply(normalize_transmission_db)' in all_source,
        "component_split_is_estimate": '"trans_A_est_N": trans_A_est' in all_source and '"brake_A_est_N": brake_A_est' in all_source,
        "component_priors_from_defaults": 'df_def = pd.read_csv(DEFAULTS_PATH)' in all_source and 'trans_A_prior' in all_source,
        "roadload_energy_calculated": 'epa_city_hwy_from_phase' in all_source,
        "database_insert_loop": 'insert_vde(vde_row)' in all_source and 'insert_fuelcons(fc_row)' in all_source,
    }
    return {
        "path": str(NOTEBOOK.relative_to(ROOT)),
        "checks": checks,
        "all_expected_evidence_present": all(checks.values()),
    }


def field_inventory_lookup() -> dict[tuple[str, str], dict[str, Any]]:
    inventory = pd.read_csv(FIELD_INVENTORY, low_memory=False).fillna("")
    return {
        (str(row["source"]), str(row["original_field_name"])): row.to_dict()
        for _, row in inventory.iterrows()
    }


def inventory_count(lookup: dict[tuple[str, str], dict[str, Any]], source: str, field: str) -> int:
    return int(float(lookup[(source, field)]["non_null_count"]))


def population_summary(
    vde: pd.DataFrame,
    fuelcons: pd.DataFrame,
    legacy_source: pd.DataFrame,
    epa_2026: pd.DataFrame,
    jrc: pd.DataFrame,
) -> list[dict[str, Any]]:
    core = vde[vde["cycle_source"].eq("standard:EPA")]
    extra_vde = vde[~vde["cycle_source"].eq("standard:EPA")]
    original_fc = fuelcons[fuelcons["label_program"].eq("EPA")]
    extra_fc = fuelcons[~fuelcons["label_program"].eq("EPA")]
    legacy_groups = legacy_source.groupby(
        ["Represented Test Veh Make", "Represented Test Veh Model", "Model Year"], dropna=False
    ).ngroups
    return [
        {
            "population": "Legacy EPA source 2020-2025",
            "source_rows": len(legacy_source),
            "test_candidates": "UNRESOLVED",
            "vde_or_reporting_rows": legacy_groups,
            "vehicle_or_mmy_groups": legacy_groups,
            "years": "2020-2025",
            "grain": "Source tests aggregated to Make+Model+Year",
        },
        {
            "population": "Current legacy core in SQLite",
            "source_rows": len(legacy_source),
            "test_candidates": "COLLAPSED",
            "vde_or_reporting_rows": len(core),
            "vehicle_or_mmy_groups": core[["make", "model", "year"]].drop_duplicates().shape[0],
            "years": "2020-2025",
            "grain": "One averaged VDE per retained Make+Model+Year",
        },
        {
            "population": "Current post-import VDE additions",
            "source_rows": "N/A",
            "test_candidates": "N/A",
            "vde_or_reporting_rows": len(extra_vde),
            "vehicle_or_mmy_groups": extra_vde[["make", "model", "year"]].drop_duplicates().shape[0],
            "years": ",".join(map(str, sorted(extra_vde["year"].dropna().astype(int).unique()))),
            "grain": "User/test scenario descendants with vde_id_parent",
        },
        {
            "population": "Current post-import FuelCons additions",
            "source_rows": "N/A",
            "test_candidates": "N/A",
            "vde_or_reporting_rows": len(extra_fc),
            "vehicle_or_mmy_groups": extra_fc["vde_id"].nunique(),
            "years": "N/A",
            "grain": "Scenario/regression/ML result",
        },
        {
            "population": "EPA Test Car MY2026",
            "source_rows": len(epa_2026),
            "test_candidates": epa_2026[["Test Number", "Actual Tested Testgroup"]].drop_duplicates().shape[0],
            "vde_or_reporting_rows": epa_2026[["Target Coef A (lbf)", "Target Coef B (lbf/mph)", "Target Coef C (lbf/mph**2)"]].drop_duplicates().shape[0],
            "vehicle_or_mmy_groups": epa_2026[["Represented Test Veh Make", "Represented Test Veh Model", "Model Year"]].drop_duplicates().shape[0],
            "years": "2026",
            "grain": "Source row/test; canonical VDE grain still requires an explicit rule",
        },
        {
            "population": "JRC technical dataset",
            "source_rows": len(jrc),
            "test_candidates": int(jrc["pycsis_run"].notna().sum()) if "pycsis_run" in jrc else "UNRESOLVED",
            "vde_or_reporting_rows": len(jrc),
            "vehicle_or_mmy_groups": "UNRESOLVED",
            "years": "No explicit model year",
            "grain": "Anonymized vehicle/archetype/simulation row; unresolved",
        },
        {
            "population": "EEA 2025 provisional",
            "source_rows": 10_833_597,
            "test_candidates": "N/A",
            "vde_or_reporting_rows": 10_833_597,
            "vehicle_or_mmy_groups": "REPORTING GRAIN",
            "years": "2025",
            "grain": "Registration/monitoring row; not a VDE/configuration row",
        },
    ]


def add_coverage(
    rows: list[dict[str, Any]], population: str, concept: str, count: int, total: int,
    provenance: str, evidence: str, semantic_status: str = "SUPPORTED",
) -> None:
    rows.append({
        "population": population,
        "concept": concept,
        "available_rows": int(count),
        "population_rows": int(total),
        "coverage_pct": pct(int(count), int(total)),
        "provenance_quality": provenance,
        "semantic_status": semantic_status,
        "evidence": evidence,
    })


def engineering_coverage(
    core: pd.DataFrame,
    core_fc: pd.DataFrame,
    epa: pd.DataFrame,
    jrc: pd.DataFrame,
    inventory: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_legacy, n_epa, n_jrc, n_eea = len(core), len(epa), len(jrc), 10_833_597
    fc_by_vde = core_fc.groupby("vde_id", as_index=True).agg(lambda s: s.notna().any())

    legacy_specs = [
        ("Whole-vehicle/Target ABC", complete_count(core, ["coast_A_N", "coast_B_N_per_kph", "coast_C_N_per_kph2"]), "SOURCE_LIKELY_AGGREGATED", "Legacy notebook averages EPA Target A/B/C by Make+Model+Year."),
        ("Set ABC", 0, "ABSENT", "No Set ABC fields persisted."),
        ("ETW / test mass", complete_count(core, ["mass_kg"]), "DETERMINISTIC_FROM_SOURCE_ETW", "mass_kg was mapped from ETW through a lookup; ETW/inertia_class is source-derived."),
        ("Transmission identity", complete_count(core, ["transmission_type"]), "SOURCE_NORMALIZED_WITH_FALLBACK", "Source transmission text normalized to short codes; unmatched values fall back to OT."),
        ("Gear count", int(fc_by_vde["gear_count"].sum()), "SOURCE_LIKELY_AGGREGATED", "Persisted in FuelCons, not VDE."),
        ("Axle / final-drive ratio", int(fc_by_vde["final_drive_ratio"].sum()), "SOURCE_LIKELY_AGGREGATED", "Persisted in FuelCons, not Vehicle Configuration."),
        ("N/V ratio", 0, "DROPPED_BY_LEGACY_ETL", "Present in EPA source but not persisted."),
        ("Transmission loss ABC", complete_count(core, ["trans_A_coef_N", "trans_B_coef_Npkph", "trans_C_coef_Npkph2"]), "ESTIMATED", "Residual split using default-table priors; not directly observed by EPA."),
        ("Brake ABC", complete_count(core, ["brake_A_coef_N", "brake_B_coef_Npkph", "brake_C_coef_Npkph2"]), "ESTIMATED", "Residual split using default-table priors; not directly observed by EPA."),
        ("Tire RRC", complete_count(core, ["rrc_N_per_kN"]), "BACKCALCULATED_BOUNDED_ESTIMATE", "Back-calculated from Target A and priors, then bounded; not directly observed."),
        ("Tire identity", any_count(core, ["tire_size"]), "SPARSE_UNRESOLVED", "Only a sparse tire_size snapshot; no reliable source identity."),
        ("Cd / CdA", any_count(core, ["cda_m2"]), "SPARSE_OR_SCENARIO", "No direct Cd/CdA in the legacy EPA source; aero_C is not CdA evidence."),
        ("Fuel / CO2", int((fc_by_vde.get("fuel_l_per_100km", False) | fc_by_vde.get("gco2_per_km", False)).sum()), "DETERMINISTIC_FROM_SOURCE_TEST_RESULTS", "Cycle aggregation and unit conversions performed by the legacy notebook."),
        ("BEV energy", int(core_fc.loc[core_fc["electrification"].astype(str).str.upper().eq("BEV") & core_fc["energy_Wh_per_km"].notna(), "vde_id"].nunique()), "DERIVED_FROM_SOURCE_ELECTRIC_TEST_RESULTS", "Counts only BEV VDEs with energy; population-wide denominator is retained for cross-population comparison."),
        ("Range", int(fc_by_vde.get("label_range_km", pd.Series(False, index=fc_by_vde.index)).sum()), "ABSENT", "No persisted range in the original population."),
    ]
    for concept, count, provenance, evidence in legacy_specs:
        add_coverage(rows, "Legacy core (4,999 VDE)", concept, count, n_legacy, provenance, evidence)

    epa_specs = [
        ("Whole-vehicle/Target ABC", ["Target Coef A (lbf)", "Target Coef B (lbf/mph)", "Target Coef C (lbf/mph**2)"], "DIRECT_SOURCE"),
        ("Set ABC", ["Set Coef A (lbf)", "Set Coef B (lbf/mph)", "Set Coef C (lbf/mph**2)"], "DIRECT_SOURCE"),
        ("ETW / test mass", ["Equivalent Test Weight (lbs.)"], "DIRECT_SOURCE"),
        ("Transmission identity", ["Tested Transmission Type"], "DIRECT_SOURCE"),
        ("Gear count", ["# of Gears"], "DIRECT_SOURCE"),
        ("Axle / final-drive ratio", ["Axle Ratio"], "DIRECT_SOURCE"),
        ("N/V ratio", ["N/V Ratio"], "DIRECT_SOURCE"),
        ("Fuel / CO2", ["CO2 (g/mi)"], "DIRECT_SOURCE"),
    ]
    for concept, fields, provenance in epa_specs:
        add_coverage(rows, "EPA Test Car MY2026", concept, complete_count(epa, fields), n_epa, provenance, "; ".join(fields))
    for concept in ("Transmission loss ABC", "Brake ABC", "Tire RRC", "Tire identity", "Cd / CdA", "Range"):
        add_coverage(rows, "EPA Test Car MY2026", concept, 0, n_epa, "ABSENT", "No explicit source field located.")
    bev_mask = epa["Test Fuel Type Description"].astype(str).str.contains("electric", case=False, na=False)
    bev_energy = int((bev_mask & epa["RND_ADJ_FE"].notna()).sum())
    add_coverage(rows, "EPA Test Car MY2026", "BEV energy", bev_energy, n_epa, "DIRECT_SOURCE_WITH_UNIT_CONTEXT", "Electric source rows with RND_ADJ_FE; not a range field.")

    jrc_specs = [
        ("Whole-vehicle/Target ABC", ["wltp|f0 [N]", "wltp|f1 [N/(km/h)]", "wltp|f2 [N/(km/h)2]"], "DIRECT_LABELS_SEMANTICS_PARTIAL"),
        ("ETW / test mass", ["Vehicle mass (WLTP) [kg]"], "DIRECT_SOURCE"),
        ("Transmission identity", ["Gear box type"], "DIRECT_SOURCE"),
        ("Gear count", ["N gears"], "DIRECT_SOURCE"),
        ("Tire identity", ["Tyre code"], "DIRECT_SOURCE_PARTIAL_IDENTITY"),
        ("Fuel / CO2", ["Declared average CO2 emissions value (OEM) [g/km]"], "DIRECT_SOURCE"),
        ("BEV energy", ["Declared electric consumption value (OEM) [Wh/km]"], "DIRECT_SOURCE"),
        ("Range", ["Electric range (OEM) [km]"], "DIRECT_SOURCE"),
    ]
    for concept, fields, provenance in jrc_specs:
        add_coverage(rows, "JRC technical dataset", concept, complete_count(jrc, fields), n_jrc, provenance, "; ".join(fields), "PARTIAL" if "PARTIAL" in provenance else "SUPPORTED")
    for concept in ("Set ABC", "Axle / final-drive ratio", "N/V ratio", "Transmission loss ABC", "Brake ABC", "Tire RRC", "Cd / CdA"):
        add_coverage(rows, "JRC technical dataset", concept, 0, n_jrc, "ABSENT", "No explicit equivalent field located.")

    eea_source = "EEA 2025 provisional passenger-car data"
    eea_specs = [
        ("ETW / test mass", "m (kg)", "DIRECT_LABEL_VALUE_SEMANTICS_PARTIAL"),
        ("Fuel / CO2", "Ewltp (g/km)", "DIRECT_LABEL_VALUE_SEMANTICS_PARTIAL"),
        ("BEV energy", "z (Wh/km)", "DIRECT_LABEL_VALUE_SEMANTICS_PARTIAL"),
        ("Range", "Electric range (km)", "DIRECT_SOURCE"),
    ]
    for concept, field, provenance in eea_specs:
        add_coverage(rows, "EEA 2025 provisional", concept, inventory_count(inventory, eea_source, field), n_eea, provenance, field, "PARTIAL" if "PARTIAL" in provenance else "SUPPORTED")
    for concept in ("Whole-vehicle/Target ABC", "Set ABC", "Transmission identity", "Gear count", "Axle / final-drive ratio", "N/V ratio", "Transmission loss ABC", "Brake ABC", "Tire RRC", "Tire identity", "Cd / CdA"):
        add_coverage(rows, "EEA 2025 provisional", concept, 0, n_eea, "ABSENT_OR_UNRESOLVED", "RLFI is retained as unresolved and is not interpreted as roadload ABC.", "UNRESOLVED" if concept == "Whole-vehicle/Target ABC" else "ABSENT")
    return rows


def source_row_hashes(frame: pd.DataFrame, columns: list[str]) -> set[int]:
    normalized = frame[columns].copy()
    for column in columns:
        normalized[column] = normalized[column].map(lambda value: "<NULL>" if not present(value) else str(clean(value)))
    return set(pd.util.hash_pandas_object(normalized, index=False).astype("uint64").tolist())


def epa_overlap(legacy_source: pd.DataFrame, all_epa: pd.DataFrame, core: pd.DataFrame) -> list[dict[str, Any]]:
    epa_2026 = all_epa[all_epa["Model Year"].eq(2026)].copy()
    new_hist = all_epa[all_epa["Model Year"].between(2020, 2025)].copy()
    shared_columns = list(legacy_source.columns)
    old_hashes = source_row_hashes(legacy_source, shared_columns)
    hist_hashes = source_row_hashes(new_hist, shared_columns)
    legacy_mmy = {(norm_text(r.make), norm_text(r.model), int(r.year)) for r in core.itertuples()}
    epa_mmy = {
        (norm_text(make), norm_text(model), int(year))
        for make, model, year in epa_2026[["Represented Test Veh Make", "Represented Test Veh Model", "Model Year"]].itertuples(index=False, name=None)
    }
    legacy_pairs = {(make, model) for make, model, _ in legacy_mmy}
    epa_pairs = {(make, model) for make, model, _ in epa_mmy}
    return [
        {"comparison": "Legacy core vs EPA2026", "grain": "normalized Make+Model+Year", "left_count": len(legacy_mmy), "right_count": len(epa_mmy), "overlap": len(legacy_mmy & epa_mmy), "right_only": len(epa_mmy - legacy_mmy), "interpretation": "No exact overlap is expected because EPA2026 adds a new model year."},
        {"comparison": "Legacy core vs EPA2026", "grain": "normalized Make+Model, ignoring year", "left_count": len(legacy_pairs), "right_count": len(epa_pairs), "overlap": len(legacy_pairs & epa_pairs), "right_only": len(epa_pairs - legacy_pairs), "interpretation": "Lexical continuity only; this does not prove same Program or Configuration."},
        {"comparison": "Old workbook vs refreshed EPA file (2020-2025)", "grain": "exact shared-column row hash", "left_count": len(old_hashes), "right_count": len(hist_hashes), "overlap": len(old_hashes & hist_hashes), "right_only": len(hist_hashes - old_hashes), "interpretation": "Shared 32-column projection; detects source refresh deltas without assigning cause."},
        {"comparison": "Old workbook vs refreshed EPA file (2020-2025)", "grain": "source row count", "left_count": len(legacy_source), "right_count": len(new_hist), "overlap": "N/A", "right_only": len(new_hist) - len(legacy_source), "interpretation": "The refreshed source has a net +31 historical rows, all in MY2025."},
    ]


def eea_lexical_overlap(legacy_keys: set[tuple[str, str, int]]) -> dict[str, Any]:
    legacy_fingerprint = hashlib.sha256(
        json.dumps(sorted(legacy_keys), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    source_stat = EEA_PATH.stat()
    cache_path = OUT / "eea_lexical_overlap_cache.json"
    fingerprint = {
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "legacy_keys_sha256": legacy_fingerprint,
    }
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fingerprint:
            return cached["result"]
    keys: set[tuple[str, str, int]] = set()
    invalid = 0
    with EEA_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            make, model = norm_text(row.get("Mk")), norm_text(row.get("Cn"))
            try:
                year = int(float(row.get("year") or 0))
            except ValueError:
                invalid += 1
                continue
            if make and model and year:
                keys.add((make, model, year))
            else:
                invalid += 1
    result = {
        "comparison": "Legacy core vs EEA 2025",
        "grain": "normalized lexical Mk+Cn+year",
        "left_count": len(legacy_keys),
        "right_count": len(keys),
        "overlap": len(legacy_keys & keys),
        "right_only": len(keys - legacy_keys),
        "interpretation": f"Lexical overlap only; EEA is reporting-grain, not VDE/configuration-grain. {invalid} row(s) lacked a usable lexical key.",
    }
    cache_path.write_text(
        json.dumps({"fingerprint": fingerprint, "result": result}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def strict_epa_configuration_count(epa: pd.DataFrame) -> int:
    fields = [
        "Represented Test Veh Make", "Represented Test Veh Model", "Model Year",
        "Actual Tested Testgroup", "Test Vehicle ID", "Test Veh Configuration #",
        "Test Veh Displacement (L)", "Engine Code", "Tested Transmission Type",
        "# of Gears", "Drive System Description", "Axle Ratio", "N/V Ratio",
    ]
    return int(epa[fields].drop_duplicates().shape[0])


def usability_rows(core: pd.DataFrame, core_fc: pd.DataFrame, epa: pd.DataFrame, jrc: pd.DataFrame) -> list[dict[str, Any]]:
    fc = core_fc.sort_values("id").drop_duplicates("vde_id", keep="first").set_index("vde_id")
    levels: dict[str, Counter[str]] = {}
    legacy_levels: Counter[str] = Counter()
    for row in core.itertuples(index=False):
        roadload = all(present(getattr(row, field)) for field in ("coast_A_N", "coast_B_N_per_kph", "coast_C_N_per_kph2", "mass_kg"))
        frow = fc.loc[row.id] if row.id in fc.index else None
        hardware = sum([
            present(row.transmission_type), present(row.engine_size_l), present(row.drive_type),
            frow is not None and present(frow.get("gear_count")),
            frow is not None and present(frow.get("final_drive_ratio")), present(row.tire_size),
        ])
        component_partial = any(present(getattr(row, field)) for field in ("trans_A_coef_N", "brake_A_coef_N", "rrc_N_per_kN"))
        level = "L0" if not roadload else "L3" if hardware >= 2 and component_partial else "L2" if hardware >= 2 else "L1"
        legacy_levels[level] += 1
    levels["Legacy core (4,999 VDE)"] = legacy_levels
    levels["EPA Test Car MY2026"] = Counter({"L2": len(epa)})
    levels["JRC technical dataset"] = Counter({"L3": len(jrc)})
    levels["EEA 2025 provisional"] = Counter({"L0": 10_833_597})
    rows: list[dict[str, Any]] = []
    definitions = {
        "L0": "Identity/reporting only; no usable complete whole-vehicle roadload state.",
        "L1": "Complete whole-vehicle roadload plus mass.",
        "L2": "L1 plus at least two useful hardware descriptors.",
        "L3": "L2 plus partial component descriptor/build-up support; provenance may still be estimated.",
        "L4": "Component-rich and scenario-ready with strong multi-domain provenance.",
    }
    for population, counts in levels.items():
        total = sum(counts.values())
        for level in ("L0", "L1", "L2", "L3", "L4"):
            rows.append({"population": population, "level": level, "rows": counts[level], "population_rows": total, "coverage_pct": pct(counts[level], total), "audit_definition": definitions[level]})
    return rows


def legacy_duplicate_groups(core: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (make, model, year), group in core.groupby(["make", "model", "year"], dropna=False):
        if len(group) < 2:
            continue
        abc_variants = group[["coast_A_N", "coast_B_N_per_kph", "coast_C_N_per_kph2"]].drop_duplicates().shape[0]
        rows.append({
            "make": make,
            "model": model,
            "year": int(year),
            "vde_rows": len(group),
            "vde_ids": ";".join(map(str, group["id"].astype(int).tolist())),
            "target_abc_variants": abc_variants,
            "interpretation": "Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved.",
        })
    return sorted(rows, key=lambda row: (str(row["make"]), str(row["model"]), int(row["year"])))


def provenance_rows(core: pd.DataFrame, core_fc: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {"population": "Legacy core", "field_group": "Identity + Target ABC + source test results", "classification": "SOURCE_LIKELY_AGGREGATED", "rows": len(core), "confidence": "HIGH at pipeline level; UNRESOLVED per row", "reason": "Notebook and source workbook align, but source_name/source_record_id and field-level lineage were not persisted."},
        {"population": "Legacy core", "field_group": "mass_kg", "classification": "DETERMINISTIC_CALCULATED", "rows": complete_count(core, ["mass_kg"]), "confidence": "HIGH", "reason": "Notebook maps ETW to a mass-class lookup."},
        {"population": "Legacy core", "field_group": "Transmission + Brake ABC", "classification": "ESTIMATED_DEFAULT_PRIOR_SPLIT", "rows": complete_count(core, ["trans_A_coef_N", "trans_B_coef_Npkph", "brake_A_coef_N", "brake_B_coef_Npkph"]), "confidence": "HIGH", "reason": "Notebook explicitly names *_est fields and splits residuals using default-table priors."},
        {"population": "Legacy core", "field_group": "Tire RRC + RR terms", "classification": "BACKCALCULATED_BOUNDED_ESTIMATE", "rows": complete_count(core, ["rrc_N_per_kN"]), "confidence": "HIGH", "reason": "Notebook back-calculates and bounds RRC/rolling terms; not a measured tire result."},
        {"population": "Legacy core", "field_group": "VDE and FuelCons results", "classification": "DETERMINISTIC_CALCULATED", "rows": int(core_fc["vde_id"].nunique()), "confidence": "HIGH", "reason": "Notebook calculates cycle energy and comparison results from source/assumption inputs."},
        {"population": "EPA Test Car MY2026", "field_group": "Target/Set ABC, ETW, transmission/gears/axle/NV", "classification": "DIRECT_SOURCE", "rows": 3901, "confidence": "HIGH", "reason": "Explicit fields with complete source-row coverage."},
        {"population": "JRC", "field_group": "Mass, gearbox/gears, tire code, declared results", "classification": "DIRECT_SOURCE_PARTIAL_SEMANTICS", "rows": 249, "confidence": "MEDIUM", "reason": "Explicit labels/units; source-row grain and anonymized identity remain unresolved."},
        {"population": "EEA", "field_group": "Monitoring mass/consumption/CO2/range", "classification": "DIRECT_REPORTING_SOURCE_PARTIAL_SEMANTICS", "rows": 10_833_597, "confidence": "MEDIUM", "reason": "Field presence is direct; reporting grain differs and RLFI remains unresolved."},
    ]


def extra_records(vde: pd.DataFrame, fuelcons: pd.DataFrame) -> list[dict[str, Any]]:
    extras = vde[~vde["cycle_source"].eq("standard:EPA")].copy()
    fc_counts = fuelcons.groupby("vde_id").size().to_dict()
    rows = []
    for row in extras.itertuples(index=False):
        rows.append({
            "vde_id": int(row.id), "created_at": row.created_at, "updated_at": row.updated_at,
            "make": row.make, "model": row.model, "year": int(row.year), "legislation": row.legislation,
            "parent_vde_id": int(row.vde_id_parent) if present(row.vde_id_parent) else "",
            "fuelcons_rows": int(fc_counts.get(row.id, 0)), "record_origin": row.record_origin,
            "source_name": row.source_name or "", "classification": "POST_IMPORT_SCENARIO_OR_TEST_DERIVATIVE",
            "evidence": "cycle_source absent; vde_id_parent/delta fields present; mock/test identity or notes present.",
        })
    extra_fc = fuelcons[~fuelcons["label_program"].eq("EPA")]
    for row in extra_fc.itertuples(index=False):
        if row.vde_id not in set(extras["id"]):
            rows.append({
                "vde_id": int(row.vde_id), "created_at": row.created_at, "updated_at": row.updated_at,
                "make": "", "model": "", "year": "", "legislation": "",
                "parent_vde_id": "", "fuelcons_rows": 1, "record_origin": row.record_origin,
                "source_name": row.source_name or "", "classification": "POST_IMPORT_ML_FUELCONS_ON_LEGACY_VDE",
                "evidence": f"FuelCons id={int(row.id)}; engine_method={row.engine_method}; explicit provenance_json present.",
            })
    return rows


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join("---" for _ in fields) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")).replace("|", "/") for field in fields) + " |")
    return lines


def report_text(payload: dict[str, Any]) -> str:
    pop = payload["population_summary"]
    coverage = payload["engineering_coverage"]
    overlaps = payload["identity_overlap"]
    usability = payload["engineering_usability"]
    extras = payload["legacy_extra_records"]
    duplicates = payload["legacy_duplicate_groups"]
    key_concepts = ["Whole-vehicle/Target ABC", "Set ABC", "ETW / test mass", "Transmission identity", "Gear count", "Axle / final-drive ratio", "N/V ratio", "Transmission loss ABC", "Brake ABC", "Tire RRC", "Tire identity", "Cd / CdA", "Fuel / CO2", "BEV energy", "Range"]
    populations = ["Legacy core (4,999 VDE)", "EPA Test Car MY2026", "JRC technical dataset", "EEA 2025 provisional"]
    cov_index = {(row["population"], row["concept"]): row for row in coverage}
    matrix = []
    for concept in key_concepts:
        item = {"Information": concept}
        for population in populations:
            row = cov_index[(population, concept)]
            item[population] = f"{row['coverage_pct']:.1f}% [{row['provenance_quality']}]"
        matrix.append(item)
    lines = [
        "# Sprint 12C.1 — Dataset Delta & Engineering Coverage Audit",
        "",
        "## Decision: `PAUSE_DDL — SOURCE_POPULATION_DECISION_REQUIRED`",
        "",
        "Sprint 12C remains a successful compatibility proof. This audit answers the separate question of whether the new sources add better engineering evidence. The answer is **yes, but as complementary populations—not as a row-count replacement for the legacy 4,999**.",
        "",
        "The runtime database was opened read-only and remained byte-identical. No DDL, migration, application, physics, resolver or raw-source file was changed.",
        "",
        "## Executive findings",
        "",
        "1. The legacy core is exactly **4,999 VDEs** derived from **26,262 EPA 2020–2025 source rows**. The notebook grouped 5,000 raw Make+Model+Year populations and discarded one group with no consumption evidence. Later text normalization collapsed these to 4,953 exact persisted MMY identities, leaving 46 two-row duplicate identity groups that must remain distinct until their VDE state differences are resolved.",
        "2. The current database has **5,003 VDEs** because four later scenario/test descendants were added. All four have `vde_id_parent` and delta fields, yet inherited `record_origin=LEGACY` and empty source identity. The database also has five post-import FuelCons records: four attached to those VDEs and one explicit ML result attached to a legacy VDE.",
        "3. Legacy Transmission/Brake ABC coverage is real as *population*, but not as observation: the notebook explicitly generated `*_est` values by splitting residual roadload with priors from `vde_defaults_by_category_trans_elec.csv`. Those fields must be classified as **ESTIMATED**, never SOURCE/MEASURED.",
        f"4. EPA MY2026 adds **3,901 source rows**, **3,557 Test Number+Testgroup candidates**, **1,291 strict configuration candidates**, **1,380 Target-ABC sets**, and **784 raw Make+Model+Year groups**. It is 100% complete for Target ABC, Set ABC, ETW, transmission, gears, axle ratio and N/V—but has no direct component ABC, tire, RRC or Cd/CdA. **{payload['epa_2026_source_identity_anomaly_rows']} rows** contain a four-digit year-like value in `Represented Test Veh Make`, so source identity must be validated rather than trusted blindly.",
        "5. JRC adds only **249 rows**, but every row has WLTP mass, f0/f1/f2, gearbox/gears and a tire code. That is richer hardware context than EPA, although identity and row grain remain unresolved.",
        "6. EEA contributes monitoring scale and fuel/energy/CO₂/range coverage, not VDE/configuration grain. `RLFI` remains unresolved and is not counted as roadload ABC.",
        "",
        "## Population and grain",
        "",
        *markdown_table(pop, ["population", "source_rows", "test_candidates", "vde_or_reporting_rows", "vehicle_or_mmy_groups", "years", "grain"]),
        "",
        "The refreshed EPA workbook is not only a 2026 append: its 2020–2025 projection has 26,293 rows, a net **+31** versus the old workbook, and 5,014 raw MMY groups versus 5,000. At the shared 32-column projection, some historical rows also differ; therefore a future canonical reload needs a versioned refresh policy rather than a blind append.",
        "",
        "### Duplicate persisted legacy identities",
        "",
        f"There are **{len(duplicates)}** exact Make+Model+Year groups with two VDE rows each. These are not safe deduplication candidates: `{sum(row['target_abc_variants'] > 1 for row in duplicates)}` groups contain multiple Target-ABC states.",
        "",
        *markdown_table(duplicates, ["make", "model", "year", "vde_rows", "vde_ids", "target_abc_variants", "interpretation"]),
        "",
        "## Engineering coverage and provenance",
        "",
        *markdown_table(matrix, ["Information", *populations]),
        "",
        "Percentages are row-population coverage, not proof of equal grain or semantic equivalence. Bracketed labels describe provenance quality. Detailed counts and evidence are in `engineering_coverage.csv`.",
        "",
        "## Overlap",
        "",
        *markdown_table(overlaps, ["comparison", "grain", "left_count", "right_count", "overlap", "right_only", "interpretation"]),
        "",
        "JRC overlap is not asserted because OEM/model are anonymized. EEA overlap is lexical only and must not be promoted to a canonical Program/Configuration match without a source-specific rule.",
        "",
        "## Post-import additions",
        "",
        *markdown_table(extras, ["vde_id", "created_at", "make", "model", "year", "parent_vde_id", "fuelcons_rows", "record_origin", "classification"]),
        "",
        "The first four rows are the additional VDEs; the last line is the fifth later FuelCons result, an ML prediction attached to legacy VDE 4861. They should be retained as scenario/evidence history, but their origin must be corrected during migration from generic `LEGACY` to explicit scenario/test/ML provenance. This audit did not edit them.",
        "",
        "## Engineering usability L0–L4",
        "",
        *markdown_table(usability, ["population", "level", "rows", "population_rows", "coverage_pct", "audit_definition"]),
        "",
        "This is an audit-only metric, not a proposed database taxonomy. Legacy L3 means partial build-up is available, but mainly as estimated/back-calculated evidence; JRC L3 means direct hardware descriptors with unresolved row identity. Neither is L4.",
        "",
        "## What was actually gained",
        "",
        "- **EPA2026:** a new model year, source-row/test lineage, multiple VDE states per commercial vehicle, Set ABC, and complete observed configuration descriptors. It improves provenance and grain even though direct component decomposition is absent.",
        "- **JRC:** direct tire code plus gearbox, gears, mass and WLTP/real-world roadload/result context. It improves descriptor richness, but does not replace identifiable EPA history.",
        "- **EEA:** very large 2025 monitoring coverage for mass, consumption, CO₂, electric energy and range. It supports homologation/monitoring comparisons, not engineering VDE replacement.",
        "- **Legacy ETL:** remains valuable for 2020–2025 historical coverage and deterministic application-ready VDE/FuelCons results. Its component ABC/RRC estimates must be retained with corrected provenance, not mistaken for measured values.",
        "",
        "## Canonical source population decision recommended before 12D",
        "",
        "1. Retain/re-ingest the 4,999 legacy EPA MMY population as historical 2020–2025 coverage, with source file/version and derivation lineage reconstructed.",
        "2. Ingest EPA2026 at source RUN grain; derive Configuration and VDE state only through an approved key. Do not repeat the old Make+Model+Year averaging collapse.",
        "3. Preserve legacy component decomposition as ESTIMATION/CALCULATION evidence, never as direct EPA source data.",
        "4. Keep the four scenario VDE descendants and five later FuelCons records in a separate scenario/ML evidence class.",
        "5. Keep JRC source-scoped and `PARTIAL/UNRESOLVED`; use its direct descriptors without cross-source identity merging.",
        "6. Treat EEA as monitoring/declared-result evidence. Do not create roadload from RLFI until its semantics are approved.",
        "",
        "Once these six population rules are approved, Sprint 12D can design physical DDL against the correct ingestion grains.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        "python etl/scripts/sprint_12c1_dataset_delta_audit.py",
        "```",
        "",
        "The first run streams the local EEA CSV to compute a conservative lexical overlap; all source/database reads are read-only.",
        "",
        "## Outputs",
        "",
        "- `etl/data/processed/sprint_12c1_dataset_delta_audit/population_summary.csv`",
        "- `etl/data/processed/sprint_12c1_dataset_delta_audit/engineering_coverage.csv`",
        "- `etl/data/processed/sprint_12c1_dataset_delta_audit/provenance_quality.csv`",
        "- `etl/data/processed/sprint_12c1_dataset_delta_audit/identity_overlap.csv`",
        "- `etl/data/processed/sprint_12c1_dataset_delta_audit/legacy_extra_records.csv`",
        "- `etl/data/processed/sprint_12c1_dataset_delta_audit/legacy_duplicate_groups.csv`",
        "- `etl/data/processed/sprint_12c1_dataset_delta_audit/engineering_usability.csv`",
        "- `etl/data/processed/sprint_12c1_dataset_delta_audit/audit_results.json`",
        "- `etl/reports/sprint_12c1_dataset_delta_audit.md`",
    ]
    return "\n".join(lines) + "\n"


def main() -> dict[str, Any]:
    required = [DB_PATH, LEGACY_EPA_PATH, EPA_PATH, JRC_PATH, EEA_PATH, FIELD_INVENTORY, NOTEBOOK]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required audit inputs: {missing}")
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    db_hash_before = sha256(DB_PATH)
    vde, fuelcons = db_frames()
    legacy_source = pd.read_excel(LEGACY_EPA_PATH, engine="openpyxl")
    all_epa = pd.read_excel(EPA_PATH, sheet_name="Sheet1", engine="openpyxl")
    epa_2026 = all_epa[all_epa["Model Year"].eq(2026)].copy()
    jrc = pd.read_excel(JRC_PATH, sheet_name="Sheet1", engine="openpyxl")
    core = vde[vde["cycle_source"].eq("standard:EPA")].copy()
    core_fc = fuelcons[fuelcons["label_program"].eq("EPA")].copy()
    inventory = field_inventory_lookup()

    populations = population_summary(vde, fuelcons, legacy_source, epa_2026, jrc)
    coverage = engineering_coverage(core, core_fc, epa_2026, jrc, inventory)
    overlaps = epa_overlap(legacy_source, all_epa, core)
    legacy_keys = {(norm_text(r.make), norm_text(r.model), int(r.year)) for r in core.itertuples()}
    overlaps.append(eea_lexical_overlap(legacy_keys))
    overlaps.append({"comparison": "Legacy core vs JRC", "grain": "identity", "left_count": len(legacy_keys), "right_count": len(jrc), "overlap": "UNRESOLVED", "right_only": "UNRESOLVED", "interpretation": "JRC OEM/model are anonymized; no defensible identity overlap can be asserted."})
    extras = extra_records(vde, fuelcons)
    duplicates = legacy_duplicate_groups(core)
    usability = usability_rows(core, core_fc, epa_2026, jrc)
    provenance = provenance_rows(core, core_fc)
    notebook = notebook_evidence()
    db_hash_after = sha256(DB_PATH)
    if db_hash_before != db_hash_after:
        raise RuntimeError("Runtime database changed during a read-only audit.")
    if len(core) != 4_999 or len(vde) != 5_003 or len(epa_2026) != 3_901 or len(jrc) != 249:
        raise RuntimeError("Observed population counts changed; audit assumptions require review.")
    if not notebook["all_expected_evidence_present"]:
        raise RuntimeError("Legacy notebook provenance evidence is incomplete.")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "decision": "PAUSE_DDL — SOURCE_POPULATION_DECISION_REQUIRED",
        "database_access": "SQLite URI mode=ro + PRAGMA query_only=ON",
        "database_sha256_before": db_hash_before,
        "database_sha256_after": db_hash_after,
        "database_byte_identical": db_hash_before == db_hash_after,
        "strict_epa_2026_configuration_candidates": strict_epa_configuration_count(epa_2026),
        "epa_2026_source_identity_anomaly_rows": int(epa_2026["Represented Test Veh Make"].astype(str).str.fullmatch(r"\d{4}").sum()),
        "population_summary": populations,
        "engineering_coverage": coverage,
        "provenance_quality": provenance,
        "identity_overlap": overlaps,
        "legacy_extra_records": extras,
        "legacy_duplicate_groups": duplicates,
        "engineering_usability": usability,
        "legacy_notebook_evidence": notebook,
    }
    write_csv("population_summary.csv", populations)
    write_csv("engineering_coverage.csv", coverage)
    write_csv("provenance_quality.csv", provenance)
    write_csv("identity_overlap.csv", overlaps)
    write_csv("legacy_extra_records.csv", extras)
    write_csv("legacy_duplicate_groups.csv", duplicates)
    write_csv("engineering_usability.csv", usability)
    (OUT / "audit_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=clean), encoding="utf-8")
    REPORT.write_text(report_text(payload), encoding="utf-8")
    print(json.dumps({
        "decision": payload["decision"],
        "legacy_core_vde": len(core),
        "post_import_vde": len(vde) - len(core),
        "epa_2026_rows": len(epa_2026),
        "epa_2026_configuration_candidates": payload["strict_epa_2026_configuration_candidates"],
        "jrc_rows": len(jrc),
        "database_byte_identical": payload["database_byte_identical"],
        "report": str(REPORT.relative_to(ROOT)),
    }, indent=2, ensure_ascii=False))
    return payload


if __name__ == "__main__":
    main()
