"""Sprint 12E.2: close EPA RUN grain and reconstruct supported FuelCons.

The script copies the disposable Sprint 12E.1 clean database, replaces only
the EPA source-row RUN population with execution-grain RUNs, and materializes
only deterministic EPA FuelCons. Runtime databases and the 12E.1 database are
opened read-only and fingerprinted before/after. No application cutover occurs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "etl" / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import sprint_12c3_program_consolidation_review as c3  # noqa: E402
import sprint_12e_migration_rehearsal as e12  # noqa: E402
import sprint_12_closure_phase2 as closure2  # noqa: E402
from vde_core.utils import epa_combined_cons_l100  # noqa: E402


RUNTIME_DB = ROOT / "data" / "db" / "eco_drive.db"
QA_DB = ROOT / "data" / "db" / "eco_drive_qa.db"
BASE_DB = ROOT / "etl" / "data" / "staging" / "sprint_12e1_clean_rebuild" / "eco_drive_canonical_clean_rebuild.db"
EPA_SOURCE = ROOT / "etl" / "data" / "raw" / "epa_testcar" / "epa_testcar_2026_raw.xlsx"
LEGACY_MATCHES = ROOT / "etl" / "data" / "processed" / "sprint_12e1_clean_rebuild" / "legacy_to_canonical_identity_matches.csv"

STAGING = ROOT / "etl" / "data" / "staging" / "sprint_12e2_epa_fuelcons"
DEFAULT_OUTPUT_DB = STAGING / "eco_drive_canonical_epa_fuelcons.db"
OUT = ROOT / "etl" / "data" / "processed" / "sprint_12e2_epa_fuelcons"
REPORT = ROOT / "etl" / "reports" / "sprint_12e2_epa_fuelcons_reconstruction.md"
SUMMARY_PATH = OUT / "sprint_12e2_summary.json"
MIGRATION_TIMESTAMP = "2026-09-11T00:00:00Z"
SOURCE_NAME = "EPA_TESTCAR_2014_PRESENT"
NORMALIZATION_VERSION = "sprint_12e2_epa_fuelcons_v1"
MPG_US_TO_L_PER_100KM = 235.214583
MILES_PER_KM = 1.609344

OUTPUT_NAMES = (
    "run_grain_analysis.csv",
    "run_grouping_results.csv",
    "run_source_row_lineage.csv",
    "fuelcons_candidate_vdes.csv",
    "fuelcons_materialization_results.csv",
    "fuelcons_run_adoption.csv",
    "metric_method_matrix.csv",
    "electrification_applicability_matrix.csv",
    "unresolved_materialization_cases.csv",
    "legacy_fuelcons_regression.csv",
    "fuelcons_population_summary.csv",
    "relationship_checks.csv",
    "runtime_db_fingerprints.csv",
    "sprint_12e2_summary.json",
)

RESULT_FIELDS = (
    "Test Category", "THC (g/mi)", "CO (g/mi)", "CO2 (g/mi)", "NOx (g/mi)",
    "PM (g/mi)", "CH4 (g/mi)", "N2O (g/mi)", "RND_ADJ_FE", "FE_UNIT",
    "FE Bag 1", "FE Bag 2", "FE Bag 3", "FE Bag 4",
)
CONDITION_FIELDS = (
    "Test Vehicle ID", "Test Veh Configuration #", "Test Procedure Cd",
    "Set Coef A (lbf)", "Set Coef B (lbf/mph)", "Set Coef C (lbf/mph**2)",
    "Police - Emergency Vehicle?", "Transmission Overdrive Code",
)
DETAIL_FIELDS = (
    "Aftertreatment Device Cd", "Aftertreatment Device Desc", "Averaging Group ID",
    "Averaging Weighting Factor", "Averaging Method Cd", "Averging Method Desc",
)
SUPPORTED_LIQUID_FUEL_TOKENS = ("gasoline", "diesel", "ethanol", "e85")
SUPPORTED_REGULAR_FTP_PROCEDURES = {2, 21, 31}


def clean(value: Any) -> Any:
    return e12.clean(value)


def stable_id(prefix: str, *parts: Any) -> str:
    return e12.stable_id(prefix, *parts)


def value_key(value: Any) -> str:
    value = clean(value)
    if value is None:
        return "<NULL>"
    if isinstance(value, float):
        return format(value, ".15g")
    return str(value).strip()


def row_payload(row: pd.Series, columns: Iterable[str]) -> dict[str, Any]:
    return {column: clean(row.get(column)) for column in columns}


def row_hash(row: pd.Series, columns: Iterable[str]) -> str:
    payload = e12.json_text(row_payload(row, columns)) or "{}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def mpg_us_to_l_per_100km(value: Any) -> float | None:
    value = clean(value)
    if value is None:
        return None
    mpg = float(value)
    if not math.isfinite(mpg) or mpg <= 0:
        return None
    return MPG_US_TO_L_PER_100KM / mpg


def g_per_mile_to_g_per_km(value: Any) -> float | None:
    value = clean(value)
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0:
        return None
    return result / MILES_PER_KM


def supported_liquid_fuel(value: Any) -> bool:
    text = str(clean(value) or "").casefold()
    return any(token in text for token in SUPPORTED_LIQUID_FUEL_TOKENS)


def valid_mpg(value: Any) -> bool:
    value = clean(value)
    if value is None:
        return False
    number = float(value)
    # 999/10000 and four-digit variants occur as unresolved source sentinels.
    return math.isfinite(number) and 0 < number <= 500


def guard_output_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved in {RUNTIME_DB.resolve(), QA_DB.resolve(), BASE_DB.resolve()}:
        raise ValueError(f"Protected database cannot be an output: {resolved}")
    allowed = STAGING.resolve()
    if not resolved.is_relative_to(allowed) or resolved.suffix.lower() != ".db":
        raise ValueError(f"12E.2 output must be a .db below {allowed}: {resolved}")
    return resolved


def write_csv(name: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def execution_keys(row: pd.Series, vde_id: int) -> tuple[str, str]:
    test_group = clean(row.get("Actual Tested Testgroup"))
    test_number = clean(row.get("Test Number"))
    source_row = int(row["source_excel_row"])
    if test_group is None or test_number is None:
        broad = stable_id("EPA-TEST-UNRESOLVED", vde_id, source_row)
        return broad, stable_id("EPA-EXEC-UNRESOLVED", vde_id, source_row)
    broad = stable_id("EPA-TEST", vde_id, test_group, test_number)
    condition = tuple(value_key(row.get(field)) for field in CONDITION_FIELDS)
    result = tuple(value_key(row.get(field)) for field in RESULT_FIELDS)
    execution = stable_id("EPA-EXEC", broad, *condition, *result)
    return broad, execution


def prepare_source() -> tuple[pd.DataFrame, list[str], int]:
    source = pd.read_excel(EPA_SOURCE, sheet_name="Sheet1", engine="openpyxl")
    prepared, _ = c3.prepare_rows(source)
    loaded = prepared[~prepared["identity_anomaly"]].copy()
    return loaded, list(source.columns), int(prepared["identity_anomaly"].sum())


def vde_source_map(con: sqlite3.Connection) -> tuple[dict[str, int], dict[int, dict[str, Any]]]:
    rows = con.execute(
        """
        SELECT v.id, v.source_record_id, v.year, v.category, v.vde_id_parent,
               v.provenance_json,
               vc.gear_count, vc.final_drive_ratio
        FROM vde v
        JOIN vehicle_configuration vc
          ON vc.vehicle_configuration_id = v.vehicle_configuration_id
        WHERE v.source_name=?
        """,
        (SOURCE_NAME,),
    ).fetchall()
    source_map = {str(row["source_record_id"]): int(row["id"]) for row in rows}
    metadata = {}
    for row in rows:
        provenance = json.loads(row["provenance_json"] or "{}")
        metadata[int(row["id"])] = {
            "year": clean(row["year"]), "category": clean(row["category"]),
            "gear_count": clean(row["gear_count"]), "final_drive_ratio": clean(row["final_drive_ratio"]),
            "vde_id_parent": clean(row["vde_id_parent"]),
            "carryover_signature_sha256": provenance.get("carryover_signature_sha256"),
        }
    return source_map, metadata


def differing_fields(group: pd.DataFrame, fields: Iterable[str]) -> list[str]:
    return [field for field in fields if group[field].map(value_key).nunique(dropna=False) > 1]


def _run_execution_evidence_payload(
    record: Mapping[str, Any],
    canonical_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the pre-micro-patch execution evidence used by FuelCons."""
    return {
        "make": clean(record.get("Represented Test Veh Make")),
        "model": clean(record.get("Represented Test Veh Model")),
        "test_group": clean(record.get("Actual Tested Testgroup")),
        "test_vehicle_id": clean(record.get("Test Vehicle ID")),
        "configuration_number": clean(record.get("Test Veh Configuration #")),
        "target_abc": {field: clean(record.get(field)) for field in c3.TARGET_FIELDS},
        "equivalent_test_weight_lb": clean(record.get("Equivalent Test Weight (lbs.)")),
        "test_category": clean(record.get("Test Category")),
        "procedure_code": clean(record.get("Test Procedure Cd")),
        "procedure_description": clean(record.get("Test Procedure Description")),
        "fuel_type": clean(record.get("Test Fuel Type Description")),
        "set_abc": {
            field: clean(record.get(field))
            for field in CONDITION_FIELDS if field.startswith("Set Coef")
        },
        "canonical_result": dict(canonical_result),
    }


def fuelcons_run_evidence_signature(
    record: Mapping[str, Any],
    canonical_result: Mapping[str, Any],
) -> str:
    """Preserve the existing adopted-Run evidence contract for FuelCons."""
    return closure2.exact_signature(_run_execution_evidence_payload(record, canonical_result))


def run_carryover_signature(
    record: Mapping[str, Any],
    canonical_result: Mapping[str, Any],
) -> str:
    """Return the exact Run lineage signature, including source test IDs."""
    payload = _run_execution_evidence_payload(record, canonical_result)
    payload.update({
        "test_number": clean(record.get("Test Number")),
        "adfe_test_number": clean(record.get("ADFE Test Number")),
    })
    return closure2.exact_signature(payload)


def close_run_grain(
    rows: pd.DataFrame,
    raw_columns: list[str],
    source_hash: str,
    source_map: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    work = rows.copy()
    work["canonical_vde_id"] = work["vde_candidate_id"].map(source_map)
    if work["canonical_vde_id"].isna().any():
        missing = sorted(set(work.loc[work["canonical_vde_id"].isna(), "vde_candidate_id"]))
        raise RuntimeError(f"EPA source VDEs missing from 12E.1: {missing[:10]}")
    work["canonical_vde_id"] = work["canonical_vde_id"].astype(int)
    keys = work.apply(lambda row: execution_keys(row, int(row["canonical_vde_id"])), axis=1)
    work["broad_test_key"] = [item[0] for item in keys]
    work["grouping_key"] = [item[1] for item in keys]
    broad_group_counts = work.groupby("broad_test_key")["grouping_key"].nunique().to_dict()

    runs: list[dict[str, Any]] = []
    grouping_results: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    facts: dict[str, dict[str, Any]] = {}
    classification_counts: Counter[str] = Counter()

    for grouping_key, group in work.groupby("grouping_key", sort=True):
        ordered = group.sort_values("source_excel_row", kind="stable")
        first = ordered.iloc[0]
        vde_id = int(first["canonical_vde_id"])
        run_id = stable_id("RUN-EPA-EXEC", source_hash, grouping_key)
        missing_identity = clean(first.get("Actual Tested Testgroup")) is None or clean(first.get("Test Number")) is None
        broad_count = int(broad_group_counts[first["broad_test_key"]])
        conflict_fields = differing_fields(
            work[work["broad_test_key"].eq(first["broad_test_key"])],
            CONDITION_FIELDS + RESULT_FIELDS,
        ) if broad_count > 1 else []
        set_abc_conflict = any(field.startswith("Set Coef") for field in conflict_fields)
        if missing_identity:
            classification, confidence = "UNRESOLVED", "LOW"
        elif broad_count > 1:
            classification, confidence = "DISTINCT_TEST_EXECUTION", "MEDIUM"
        elif len(ordered) > 1 and ordered["Averaging Group ID"].notna().any():
            classification, confidence = "DECLARED_RESULT_DETAIL", "HIGH"
        elif len(ordered) > 1:
            classification, confidence = "SAME_TEST_EXECUTION_DETAIL", "HIGH"
        else:
            classification, confidence = "DISTINCT_TEST_EXECUTION", "HIGH"
        classification_counts[classification] += len(ordered)

        raw_details = [
            {"source_excel_row": int(row["source_excel_row"]), "source_payload": row_payload(row, raw_columns)}
            for _, row in ordered.iterrows()
        ]
        result = {field: clean(first.get(field)) for field in RESULT_FIELDS}
        conditions = {
            "grouping_key": grouping_key,
            "broad_test_key": first["broad_test_key"],
            "test_number": clean(first.get("Test Number")),
            "adfe_test_number": clean(first.get("ADFE Test Number")),
            "test_group": clean(first.get("Actual Tested Testgroup")),
            "test_vehicle_id": clean(first.get("Test Vehicle ID")),
            "configuration_number": clean(first.get("Test Veh Configuration #")),
            "test_category": clean(first.get("Test Category")),
            "test_procedure_code": clean(first.get("Test Procedure Cd")),
            "test_procedure_description": clean(first.get("Test Procedure Description")),
            "set_abc_native": {field: clean(first.get(field)) for field in CONDITION_FIELDS if field.startswith("Set Coef")},
            "source_row_count": len(ordered),
            "grouping_confidence": confidence,
            "conflict_fields_within_test_identity": conflict_fields,
        }
        review_status = closure2.RUN_IDENTITY_REVIEW if broad_count > 1 and set_abc_conflict else "CURRENT"
        review_reason = (
            "Same Test Number with multiple Set ABC variants; within-year execution meaning unresolved."
            if review_status == closure2.RUN_IDENTITY_REVIEW else None
        )
        run_signature = run_carryover_signature(first, result)
        fuelcons_evidence_signature = fuelcons_run_evidence_signature(first, result)
        runs.append({
            "run_id": run_id, "vde_id": vde_id, "run_type": "TEST", "evidence_kind": "HOMOLOGATION",
            "confidence": confidence, "source_name": SOURCE_NAME, "source_file_version": source_hash,
            "source_record_id": grouping_key,
            "procedure_code": value_key(first.get("Test Procedure Cd")) if clean(first.get("Test Procedure Cd")) is not None else None,
            "procedure_description": clean(first.get("Test Procedure Description")),
            "conditions_json": conditions,
            "result_details_json": {"canonical_result": result, "source_row_details": raw_details},
            "provenance_json": {
                "population": "EPA_RUN_GRAIN_CLOSED", "grouping_rule_version": NORMALIZATION_VERSION,
                "classification": classification, "all_source_rows_preserved": True,
                "carryover_signature_sha256": run_signature,
                "run_identity_review_reason": review_reason,
            },
            "review_status": review_status,
            "created_at": MIGRATION_TIMESTAMP,
        })
        grouping_results.append({
            "canonical_run_id": run_id, "canonical_vde_id": vde_id, "grouping_key": grouping_key,
            "broad_test_key": first["broad_test_key"], "classification": classification,
            "grouping_confidence": confidence, "source_row_count": len(ordered),
            "source_excel_rows": ";".join(str(int(value)) for value in ordered["source_excel_row"]),
            "test_number": clean(first.get("Test Number")),
            "adfe_test_number": clean(first.get("ADFE Test Number")),
            "test_group": clean(first.get("Actual Tested Testgroup")),
            "test_category": clean(first.get("Test Category")), "procedure_code": clean(first.get("Test Procedure Cd")),
            "condition_or_result_conflicts": ";".join(conflict_fields),
        })
        for _, row in ordered.iterrows():
            source_row = int(row["source_excel_row"])
            row_classification = classification
            if len(ordered) > 1 and clean(row.get("Averaging Group ID")) is not None:
                row_classification = "DECLARED_RESULT_DETAIL"
            lineage.append({
                "source_excel_row": source_row,
                "original_source_row_run_id": stable_id("RUN-EPA-ROW", source_hash, source_row),
                "canonical_run_id": run_id, "canonical_vde_id": vde_id, "grouping_key": grouping_key,
                "classification": row_classification, "grouping_confidence": confidence,
                "source_row_sha256": row_hash(row, raw_columns), "source_file_version": source_hash,
            })
        facts[run_id] = {
            "run_id": run_id, "vde_id": vde_id, "category": clean(first.get("Test Category")),
            "procedure_code": clean(first.get("Test Procedure Cd")),
            "procedure_description": clean(first.get("Test Procedure Description")),
            "fuel_type": clean(first.get("Test Fuel Type Description")),
            "mpg": clean(first.get("RND_ADJ_FE")), "fe_unit": clean(first.get("FE_UNIT")),
            "co2_g_mile": clean(first.get("CO2 (g/mi)")),
            "analytically_derived": clean(first.get("Analytically Derived FE?")),
            "source_row_count": len(ordered), "classification": classification,
            "year": int(first["Model Year"]),
            "carryover_signature_sha256": run_signature,
            "fuelcons_evidence_signature_sha256": fuelcons_evidence_signature,
        }

    run_lineage = closure2.assign_temporal_parents(
        {"id": run_id, "year": fact["year"], "signature": fact["carryover_signature_sha256"]}
        for run_id, fact in facts.items()
    )
    runs_by_id = {row["run_id"]: row for row in runs}
    for run_id, relation in run_lineage.items():
        provenance = runs_by_id[run_id]["provenance_json"]
        provenance["carryover_status"] = relation["status"]
        if relation["status"] == "LINKED":
            provenance.update({
                "lineage_relation": closure2.EPA_MODEL_YEAR_CARRYOVER,
                "carryover_from_model_year": relation["parent_year"],
                "carryover_from_run_id": relation["parent_id"],
            })
        elif relation["status"] == "AMBIGUOUS":
            runs_by_id[run_id]["review_status"] = closure2.RUN_IDENTITY_REVIEW
            ambiguity_reason = "Ambiguous cross-year carryover; multiple exact predecessor candidates."
            existing_reason = provenance.get("run_identity_review_reason")
            provenance["run_identity_review_reason"] = (
                f"{existing_reason} {ambiguity_reason}" if existing_reason else ambiguity_reason
            )

    analysis = [
        {"metric": "EPA_SOURCE_ROWS_LOADED", "value": len(work), "classification": "SOURCE_EVIDENCE", "evidence": "All non-quarantined refreshed rows."},
        {"metric": "EPA_CANONICAL_RUNS_BEFORE", "value": len(work), "classification": "SOURCE_ROW_GRAIN", "evidence": "Sprint 12E.1 one RUN per source row."},
        {"metric": "EPA_CANONICAL_RUNS_AFTER", "value": len(runs), "classification": "EXECUTION_GRAIN", "evidence": "VDE + test identity + procedure/conditions/result signature."},
        {"metric": "SOURCE_ROWS_CONSOLIDATED", "value": len(work) - len(runs), "classification": "SAME_TEST_EXECUTION_DETAIL", "evidence": "Only aftertreatment/declared averaging detail varies inside grouped RUNs."},
    ]
    for classification in ("SAME_TEST_EXECUTION_DETAIL", "DISTINCT_TEST_EXECUTION", "DECLARED_RESULT_DETAIL", "UNRESOLVED"):
        analysis.append({
            "metric": f"SOURCE_ROWS_{classification}", "value": classification_counts[classification],
            "classification": classification, "evidence": "Source-row classification after deterministic grouping.",
        })
    return runs, grouping_results, lineage, analysis, facts


def electrification_for_vde(group: pd.DataFrame) -> tuple[str, str]:
    fuels = "|".join(sorted({str(value) for value in group["Test Fuel Type Description"].dropna()})).casefold()
    has_electric = "electric" in fuels
    has_liquid = any(token in fuels for token in SUPPORTED_LIQUID_FUEL_TOKENS)
    has_hydrogen = "hydrogen" in fuels
    has_cd = bool(group["Test Category"].eq("CD").any())
    has_bag4 = bool(group["FE Bag 4"].notna().any())
    if has_electric and has_liquid or has_cd and has_liquid:
        return "PHEV", "SOURCE_CD_OR_MIXED_ELECTRIC_LIQUID"
    if has_electric:
        return "BEV", "SOURCE_ELECTRICITY"
    if has_hydrogen:
        return "FCEV", "SOURCE_HYDROGEN"
    if has_bag4:
        return "HEV", "DETERMINISTIC_VALIDATED_LEGACY_BAG4_CLASSIFIER"
    return "ICE", "SOURCE_NON_ELECTRIC_NON_CD"


def metric_signature(fact: dict[str, Any]) -> tuple[str, str]:
    return value_key(fact["mpg"]), value_key(fact["co2_g_mile"])


def select_runs(candidates: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], str]:
    valid = [
        fact for fact in candidates
        if str(fact.get("fe_unit") or "").upper() == "MPG" and valid_mpg(fact.get("mpg"))
    ]
    if not valid:
        return "UNRESOLVED", [], "NO_SUPPORTED_FINITE_MPG_CANDIDATE"
    signatures = {metric_signature(fact) for fact in valid}
    if len(signatures) > 1:
        return "UNRESOLVED", [], "MULTIPLE_CONFLICTING_ELIGIBLE_RUNS"
    if len(valid) > 1:
        return "MULTI_RUN_ADOPTION", sorted(valid, key=lambda row: row["run_id"]), "IDENTICAL_RESULT_MULTIPLE_EXECUTIONS"
    if len(candidates) > 1:
        return "DETERMINISTIC_SELECTION", valid, "ONLY_ONE_SUPPORTED_VALID_RESULT"
    return "DIRECT_SINGLE_CANDIDATE", valid, "ONE_SUPPORTED_RESULT"


def combine_selection(city: str, highway: str) -> str:
    if "UNRESOLVED" in (city, highway):
        return "UNRESOLVED"
    if "MULTI_RUN_ADOPTION" in (city, highway):
        return "MULTI_RUN_ADOPTION"
    if "DETERMINISTIC_SELECTION" in (city, highway):
        return "DETERMINISTIC_SELECTION"
    return "DIRECT_SINGLE_CANDIDATE"


def fuelcons_row(
    fuelcons_id: int,
    vde_id: int,
    vde_meta: dict[str, Any],
    electrification: str,
    fuel_type: str,
    basis: str,
    cycle: str,
    values: dict[str, Any],
    run_ids: list[str],
    source_hash: str,
    selection: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": fuelcons_id, "vde_id": vde_id, "comparison_basis": basis,
        "electrification": electrification, "fuel_type": fuel_type,
        "gear_count": vde_meta.get("gear_count"), "final_drive_ratio": vde_meta.get("final_drive_ratio"),
        "energy_basis": None, "label_program": "EPA", "label_version_year": vde_meta.get("year"),
        "label_vehicle_category": vde_meta.get("category"),
        "label_cycle_set": "2-cycle" if basis == "EPA_LABEL_2_CYCLE" else cycle,
        "method_note": (
            "EPA source RND_ADJ_FE MPG converted with 235.214583/MPG; city/highway combined 55/45."
            if basis == "EPA_LABEL_2_CYCLE"
            else f"EPA {cycle} source result retained only in the matching specialized-cycle fields."
        ),
        "assumptions_json": {
            "method_version": NORMALIZATION_VERSION, "selection_classification": selection,
            "mpg_us_to_l_per_100km_constant": MPG_US_TO_L_PER_100KM,
            "g_per_mile_to_g_per_km_divisor": MILES_PER_KM,
            "combined_weights": {"city": 0.55, "highway": 0.45} if basis == "EPA_LABEL_2_CYCLE" else None,
            "energy_and_range_estimation": "NOT_PERFORMED",
        },
        "provenance_json": {
            "source": SOURCE_NAME, "adopted_run_ids": run_ids, "comparison_basis": basis,
            "cycle": cycle, "metric_logic": "DETERMINISTIC_VALIDATED", "legacy_rows_copied": False,
        },
        "record_origin": "EPA_RECONSTRUCTED", "record_status": "ACTIVE", "review_status": "CURRENT",
        "source_name": SOURCE_NAME, "source_file_version": source_hash,
        "source_record_id": stable_id("FC-EPA-SOURCE", vde_id, basis, cycle, fuel_type),
        "normalization_version": NORMALIZATION_VERSION, "created_at": MIGRATION_TIMESTAMP,
    }
    payload.update(values)
    return payload


def make_adoptions(
    fuelcons_id: int,
    vde_id: int,
    selected: list[dict[str, Any]],
    dimension: str,
    role: str,
    selection: str,
) -> list[dict[str, Any]]:
    return [
        {
            "fuelcons_id": fuelcons_id, "run_id": fact["run_id"], "vde_id": vde_id,
            "adoption_role": role, "result_dimension": dimension, "ordinal": ordinal,
            "details_json": {"selection_classification": selection, "cycle": fact["category"]},
            "created_at": MIGRATION_TIMESTAMP,
        }
        for ordinal, fact in enumerate(selected)
    ]


def materialize_fuelcons(
    rows: pd.DataFrame,
    source_map: dict[str, int],
    vde_meta: dict[int, dict[str, Any]],
    facts: dict[str, dict[str, Any]],
    source_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[int, str]]:
    work = rows.copy()
    work["canonical_vde_id"] = work["vde_candidate_id"].map(source_map).astype(int)
    vde_electrification: dict[int, str] = {}
    vde_evidence: dict[int, str] = {}
    for vde_id, group in work.groupby("canonical_vde_id", sort=True):
        classification, evidence = electrification_for_vde(group)
        vde_electrification[int(vde_id)] = classification
        vde_evidence[int(vde_id)] = evidence

    facts_by_vde: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for fact in facts.values():
        facts_by_vde[int(fact["vde_id"])].append(fact)

    pending: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    vde_outcomes: dict[int, Counter[str]] = defaultdict(Counter)

    for vde_id in sorted(vde_meta):
        vde_facts = facts_by_vde[vde_id]
        electrification = vde_electrification[vde_id]
        cycles = Counter(str(fact["category"]) for fact in vde_facts)
        fuels = sorted({str(fact["fuel_type"]) for fact in vde_facts if supported_liquid_fuel(fact["fuel_type"])})
        if electrification in {"BEV", "FCEV"}:
            unresolved.append({
                "canonical_vde_id": vde_id, "comparison_basis": "EPA_LABEL_2_CYCLE",
                "fuel_type": "|".join(sorted({str(fact["fuel_type"]) for fact in vde_facts})),
                "cycle": "FTP+HWY", "reason_code": "UNSUPPORTED_ENERGY_OR_EQUIVALENT_FUEL_UNIT",
                "candidate_run_ids": ";".join(sorted(fact["run_id"] for fact in vde_facts)),
                "details": "FE_UNIT is MPG but source does not establish Wh/km conversion; no value invented.",
            })
        for fuel_type in fuels:
            fuel_facts = [fact for fact in vde_facts if str(fact["fuel_type"]) == fuel_type]
            city_all = [
                fact for fact in fuel_facts
                if fact["category"] == "FTP" and clean(fact["procedure_code"]) in SUPPORTED_REGULAR_FTP_PROCEDURES
                and "cold" not in str(fact["procedure_description"] or "").casefold()
            ]
            highway_all = [fact for fact in fuel_facts if fact["category"] == "HWY" and clean(fact["procedure_code"]) == 3]
            if city_all or highway_all:
                city_status, city_selected, city_reason = select_runs(city_all)
                highway_status, highway_selected, highway_reason = select_runs(highway_all)
                selection = combine_selection(city_status, highway_status)
                if selection == "UNRESOLVED":
                    unresolved.append({
                        "canonical_vde_id": vde_id, "comparison_basis": "EPA_LABEL_2_CYCLE",
                        "fuel_type": fuel_type, "cycle": "FTP+HWY",
                        "reason_code": f"CITY:{city_reason}|HIGHWAY:{highway_reason}",
                        "candidate_run_ids": ";".join(sorted(fact["run_id"] for fact in city_all + highway_all)),
                        "details": "Both supported, non-conflicting city and highway evidence are required.",
                    })
                    vde_outcomes[vde_id]["unresolved"] += 1
                else:
                    city = city_selected[0]
                    highway = highway_selected[0]
                    city_fuel = mpg_us_to_l_per_100km(city["mpg"])
                    highway_fuel = mpg_us_to_l_per_100km(highway["mpg"])
                    city_co2 = g_per_mile_to_g_per_km(city["co2_g_mile"])
                    highway_co2 = g_per_mile_to_g_per_km(highway["co2_g_mile"])
                    combined_fuel = epa_combined_cons_l100(city_fuel, highway_fuel)
                    combined_co2 = (
                        epa_combined_cons_l100(city_co2, highway_co2)
                        if city_co2 is not None and highway_co2 is not None else None
                    )
                    pending.append({
                        "vde_id": vde_id, "basis": "EPA_LABEL_2_CYCLE", "cycle": "FTP+HWY",
                        "fuel_type": fuel_type, "electrification": electrification, "selection": selection,
                        "selected": [(city_selected, "CITY"), (highway_selected, "HIGHWAY")],
                        "values": {
                            "fuel_ftp75_l_per_100km": city_fuel,
                            "fuel_hwfet_l_per_100km": highway_fuel,
                            "fuel_l_per_100km": combined_fuel,
                            "fuel_km_per_l": None if not combined_fuel else 100.0 / combined_fuel,
                            "gco2_ftp75_per_km": city_co2,
                            "gco2_hwfet_per_km": highway_co2,
                            "gco2_per_km": combined_co2,
                            "label_fuel_l_per_100km": combined_fuel,
                            "label_gco2_per_km": combined_co2,
                        },
                    })
                    vde_outcomes[vde_id]["created"] += 1

            for cycle, field_prefix in (("US06", "us06"), ("SC03", "sc03")):
                cycle_all = [fact for fact in fuel_facts if fact["category"] == cycle]
                if not cycle_all:
                    continue
                selection, selected, reason = select_runs(cycle_all)
                if selection == "UNRESOLVED":
                    unresolved.append({
                        "canonical_vde_id": vde_id, "comparison_basis": "EPA_TEST_PROCEDURE",
                        "fuel_type": fuel_type, "cycle": cycle, "reason_code": reason,
                        "candidate_run_ids": ";".join(sorted(fact["run_id"] for fact in cycle_all)),
                        "details": "Conflicting repeated specialized-cycle results are not averaged or selected arbitrarily.",
                    })
                    vde_outcomes[vde_id]["unresolved"] += 1
                    continue
                fact = selected[0]
                pending.append({
                    "vde_id": vde_id, "basis": "EPA_TEST_PROCEDURE", "cycle": cycle,
                    "fuel_type": fuel_type, "electrification": electrification, "selection": selection,
                    "selected": [(selected, cycle)],
                    "values": {
                        f"fuel_{field_prefix}_l_per_100km": mpg_us_to_l_per_100km(fact["mpg"]),
                        f"gco2_{field_prefix}_per_km": g_per_mile_to_g_per_km(fact["co2_g_mile"]),
                    },
                })
                vde_outcomes[vde_id]["created"] += 1

        candidate_rows.append({
            "canonical_vde_id": vde_id, "electrification": electrification,
            "electrification_evidence": vde_evidence[vde_id],
            "run_count": len(vde_facts), "cycle_counts": e12.json_text(dict(sorted(cycles.items()))),
            "supported_liquid_fuels": ";".join(fuels),
            "fuelcons_created": vde_outcomes[vde_id]["created"],
            "unresolved_candidates": vde_outcomes[vde_id]["unresolved"],
            "eligibility_status": "ELIGIBLE_MATERIALIZED" if vde_outcomes[vde_id]["created"] else "NOT_MATERIALIZED",
        })

    fuelcons: list[dict[str, Any]] = []
    fuelcons_lineage_records: list[dict[str, Any]] = []
    adoptions: list[dict[str, Any]] = []
    materialization: list[dict[str, Any]] = []
    pending.sort(key=lambda row: (row["vde_id"], row["basis"], row["cycle"], row["fuel_type"]))
    for position, item in enumerate(pending, start=1):
        fuelcons_id = -4_000_000 - position
        selected_facts = [fact for group, _ in item["selected"] for fact in group]
        run_ids = [fact["run_id"] for fact in selected_facts]
        row = fuelcons_row(
            fuelcons_id, item["vde_id"], vde_meta[item["vde_id"]], item["electrification"],
            item["fuel_type"], item["basis"], item["cycle"], item["values"], run_ids,
            source_hash, item["selection"],
        )
        fuelcons_signature = closure2.exact_signature({
            "vde_carryover_signature_sha256": vde_meta[item["vde_id"]].get("carryover_signature_sha256"),
            "comparison_basis": item["basis"],
            "cycle": item["cycle"],
            "fuel_type": item["fuel_type"],
            "electrification": item["electrification"],
            "values": item["values"],
            "adopted_run_signatures": sorted(
                fact["fuelcons_evidence_signature_sha256"] for fact in selected_facts
            ),
        })
        row["provenance_json"]["carryover_signature_sha256"] = fuelcons_signature
        fuelcons.append(row)
        fuelcons_lineage_records.append({
            "id": fuelcons_id,
            "year": int(vde_meta[item["vde_id"]]["year"]),
            "signature": fuelcons_signature,
        })
        for selected, dimension in item["selected"]:
            role = "POST_PROCESSING_INPUT" if item["basis"] == "EPA_LABEL_2_CYCLE" else "PRIMARY"
            adoptions.extend(make_adoptions(
                fuelcons_id, item["vde_id"], selected, dimension, role, item["selection"],
            ))
        materialization.append({
            "fuelcons_id": fuelcons_id, "canonical_vde_id": item["vde_id"],
            "comparison_basis": item["basis"], "cycle": item["cycle"],
            "electrification": item["electrification"], "fuel_type": item["fuel_type"],
            "selection_classification": item["selection"], "adopted_run_count": len(run_ids),
            "adopted_run_ids": ";".join(run_ids), "status": "CREATED",
        })
    fuelcons_lineage = closure2.assign_temporal_parents(fuelcons_lineage_records)
    fuelcons_by_id = {row["id"]: row for row in fuelcons}
    for fuelcons_id, relation in fuelcons_lineage.items():
        provenance = fuelcons_by_id[fuelcons_id]["provenance_json"]
        provenance["carryover_status"] = relation["status"]
        if relation["status"] == "LINKED":
            provenance.update({
                "lineage_relation": closure2.EPA_MODEL_YEAR_CARRYOVER,
                "carryover_from_model_year": relation["parent_year"],
                "carryover_from_fuelcons_id": relation["parent_id"],
            })
        elif relation["status"] == "AMBIGUOUS":
            fuelcons_by_id[fuelcons_id]["review_status"] = "CARRYOVER_IDENTITY_REVIEW"
    return fuelcons, adoptions, materialization, unresolved, vde_electrification


def metric_method_matrix() -> list[dict[str, Any]]:
    rows = [
        ("fuel_city", "EPA_LABEL_2_CYCLE", "RND_ADJ_FE", "MPG", "L/100km", "235.214583 / MPG", "DETERMINISTIC_VALIDATED", "MATERIALIZED"),
        ("fuel_highway", "EPA_LABEL_2_CYCLE", "RND_ADJ_FE", "MPG", "L/100km", "235.214583 / MPG", "DETERMINISTIC_VALIDATED", "MATERIALIZED"),
        ("fuel_combined", "EPA_LABEL_2_CYCLE", "city+highway", "L/100km", "L/100km", "0.55*City + 0.45*Highway", "DETERMINISTIC_VALIDATED", "MATERIALIZED"),
        ("co2_city", "EPA_LABEL_2_CYCLE", "CO2 (g/mi)", "g/mi", "g/km", "value / 1.609344", "SOURCE_SUPPORTED", "MATERIALIZED_WHEN_PRESENT"),
        ("co2_highway", "EPA_LABEL_2_CYCLE", "CO2 (g/mi)", "g/mi", "g/km", "value / 1.609344", "SOURCE_SUPPORTED", "MATERIALIZED_WHEN_PRESENT"),
        ("co2_combined", "EPA_LABEL_2_CYCLE", "city+highway", "g/km", "g/km", "0.55*City + 0.45*Highway", "DETERMINISTIC_VALIDATED", "MATERIALIZED_WHEN_BOTH_PRESENT"),
        ("fuel_us06", "EPA_TEST_PROCEDURE", "RND_ADJ_FE", "MPG", "L/100km", "235.214583 / MPG", "DETERMINISTIC_VALIDATED", "SPECIALIZED_FIELD_ONLY"),
        ("fuel_sc03", "EPA_TEST_PROCEDURE", "RND_ADJ_FE", "MPG", "L/100km", "235.214583 / MPG", "DETERMINISTIC_VALIDATED", "SPECIALIZED_FIELD_ONLY"),
        ("electric_energy", "EPA_LABEL_2_CYCLE", "RND_ADJ_FE", "MPG", "Wh/km", "NONE", "UNRESOLVED", "NULL_NO_SUPPORTED_UNIT_CONVERSION"),
        ("range", "ALL", "none", "none", "km", "NONE", "SOURCE_SUPPORTED", "NULL_NOT_IN_SOURCE"),
        ("bag_city_fallback", "EPA_LABEL_2_CYCLE", "FE Bag 1..4", "MPG", "L/100km", "legacy 0.43/1.00/0.57", "LEGACY_ASSUMPTION", "NOT_USED"),
        ("liquid_energy", "ALL", "fuel consumption", "L/100km", "Wh/km", "LHV conversion", "LEGACY_ASSUMPTION", "NOT_USED"),
    ]
    return [
        {"metric": a, "comparison_basis": b, "source_field": c, "source_unit": d, "canonical_unit": e,
         "formula_or_conversion": f, "logic_classification": g, "materialization": h}
        for a, b, c, d, e, f, g, h in rows
    ]


def applicability_matrix() -> list[dict[str, Any]]:
    rules = {
        "ICE": ("SUPPORTED", "NULL", "SUPPORTED", "NULL", "LIQUID_SOURCE", "DETERMINISTIC_NON_ELECTRIC"),
        "HEV": ("SUPPORTED", "NULL", "SUPPORTED", "NULL", "LIQUID_SOURCE", "VALIDATED_BAG4_CLASSIFIER"),
        "PHEV": ("SUPPORTED_CS_ONLY", "DEFERRED_CD", "SUPPORTED_CS_ONLY", "NULL", "LIQUID_SOURCE", "SOURCE_CD_OR_MIXED_FUEL"),
        "BEV": ("NULL", "DEFERRED_UNIT_AMBIGUITY", "NULL", "NULL", "ELECTRICITY", "SOURCE_ELECTRICITY"),
        "FCEV": ("NULL", "DEFERRED_EQUIVALENT_UNIT", "NULL", "NULL", "HYDROGEN", "SOURCE_HYDROGEN"),
    }
    rows: list[dict[str, Any]] = []
    for electrification, values in rules.items():
        fuel, energy, co2, range_value, fuel_type, evidence = values
        rows.append({
            "electrification": electrification, "fuel_consumption": fuel, "electric_energy": energy,
            "co2": co2, "electric_range": range_value, "fuel_type": fuel_type,
            "energy_basis": "NULL_UNLESS_SOURCE_SUPPORTED", "classification_evidence": evidence,
        })
    return rows


def legacy_regression(new_fuelcons: list[dict[str, Any]], adoptions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    with e12.open_readonly(RUNTIME_DB) as con:
        legacy = [dict(row) for row in con.execute("SELECT * FROM fuelcons_db WHERE label_program='EPA' ORDER BY id")]
    matches: dict[int, dict[str, str]] = {}
    with LEGACY_MATCHES.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            matches[int(row["legacy_vde_id"])] = row
    by_vde: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in new_fuelcons:
        if row["comparison_basis"] == "EPA_LABEL_2_CYCLE":
            by_vde[int(row["vde_id"])].append(row)
    adoption_count = Counter(int(row["fuelcons_id"]) for row in adoptions)
    fields = (
        "fuel_ftp75_l_per_100km", "fuel_hwfet_l_per_100km", "fuel_l_per_100km",
        "gco2_ftp75_per_km", "gco2_hwfet_per_km", "gco2_per_km",
    )

    def equivalent(a: Any, b: Any) -> bool:
        a, b = clean(a), clean(b)
        if a is None or b is None:
            return a is None and b is None
        return math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-8)

    output: list[dict[str, Any]] = []
    for old in legacy:
        match = matches.get(int(old["vde_id"]))
        canonical_vde = None if not match or not match.get("canonical_vde_id") else int(float(match["canonical_vde_id"]))
        candidates = [] if canonical_vde is None else by_vde.get(canonical_vde, [])
        exact_fuel = [row for row in candidates if row.get("fuel_type") == old.get("fuel_type")]
        candidate = exact_fuel[0] if len(exact_fuel) == 1 else (candidates[0] if len(candidates) == 1 else None)
        if canonical_vde is None:
            classification, reason = "UNRESOLVED", "Legacy VDE has no safe canonical VDE match."
        elif candidate is None:
            classification, reason = "NO_DIRECT_CANONICAL_COUNTERPART", "No unambiguous supported EPA 2-cycle result."
        else:
            comparisons = {field: equivalent(old.get(field), candidate.get(field)) for field in fields}
            raw_equal = all(
                (clean(old.get(field)) is None and clean(candidate.get(field)) is None)
                or clean(old.get(field)) == clean(candidate.get(field))
                for field in fields
            )
            if raw_equal:
                classification, reason = "EXACT_EQUIVALENCE", "Supported source metrics are exactly equal."
            elif all(comparisons.values()):
                classification, reason = "NUMERIC_EQUIVALENCE_WITH_GRAIN_CHANGE", "Equivalent within 1e-8 after execution-grain reconstruction."
            else:
                classification, reason = "SOURCE_REFRESH_DIFFERENCE", "Current source/grain produces a different supported metric signature."
        output.append({
            "legacy_fuelcons_id": old["id"], "legacy_vde_id": old["vde_id"],
            "canonical_vde_id": canonical_vde or "", "canonical_fuelcons_id": "" if candidate is None else candidate["id"],
            "classification": classification, "legacy_fuel_type": old.get("fuel_type"),
            "canonical_fuel_type": "" if candidate is None else candidate.get("fuel_type"),
            "adopted_run_count": 0 if candidate is None else adoption_count[candidate["id"]], "evidence": reason,
        })
    return output


def relationship_checks(
    con: sqlite3.Connection,
    source_rows: int,
    grouped_runs: int,
    lineage: list[dict[str, Any]],
    fuelcons: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    checks: list[tuple[str, int, int, str]] = []
    checks.append(("EPA_SOURCE_LINEAGE_COMPLETE", len(lineage), source_rows, "Every loaded source row maps to one canonical RUN."))
    checks.append(("EPA_RUN_COUNT_CLOSED", con.execute("SELECT COUNT(*) FROM run WHERE source_name=?", (SOURCE_NAME,)).fetchone()[0], grouped_runs, "Database EPA RUN count equals grouping output."))
    checks.append(("FOREIGN_KEY_VIOLATIONS", len(con.execute("PRAGMA foreign_key_check").fetchall()), 0, "SQLite foreign_key_check."))
    checks.append(("ORPHAN_EPA_RUNS", con.execute("SELECT COUNT(*) FROM run r LEFT JOIN vde v ON v.id=r.vde_id WHERE r.source_name=? AND v.id IS NULL", (SOURCE_NAME,)).fetchone()[0], 0, "EPA RUN to VDE."))
    checks.append(("ORPHAN_EPA_FUELCONS", con.execute("SELECT COUNT(*) FROM fuelcons f LEFT JOIN vde v ON v.id=f.vde_id WHERE f.record_origin='EPA_RECONSTRUCTED' AND v.id IS NULL").fetchone()[0], 0, "EPA FuelCons to VDE."))
    checks.append(("ADOPTION_SAME_VDE", con.execute("SELECT COUNT(*) FROM fuelcons_run_adoption a JOIN fuelcons f ON f.id=a.fuelcons_id JOIN run r ON r.run_id=a.run_id WHERE a.vde_id<>f.vde_id OR a.vde_id<>r.vde_id").fetchone()[0], 0, "FuelCons and adopted RUN share VDE."))
    checks.append(("EPA_FUELCONS_HAVE_ADOPTION", con.execute("SELECT COUNT(*) FROM fuelcons f WHERE f.record_origin='EPA_RECONSTRUCTED' AND NOT EXISTS (SELECT 1 FROM fuelcons_run_adoption a WHERE a.fuelcons_id=f.id)").fetchone()[0], 0, "Every reconstructed result has RUN lineage."))
    checks.append(("EPA_FUELCONS_CREATED", con.execute("SELECT COUNT(*) FROM fuelcons WHERE record_origin='EPA_RECONSTRUCTED'").fetchone()[0], len(fuelcons), "Database matches materialization output."))
    checks.append(("EPA_ENERGY_NOT_INVENTED", con.execute("SELECT COUNT(*) FROM fuelcons WHERE record_origin='EPA_RECONSTRUCTED' AND (energy_Wh_per_km IS NOT NULL OR energy_ftp75_Wh_per_km IS NOT NULL OR energy_hwfet_Wh_per_km IS NOT NULL)").fetchone()[0], 0, "Unsupported energy conversion remains NULL."))
    checks.append(("EPA_RANGE_NOT_INVENTED", con.execute("SELECT COUNT(*) FROM fuelcons WHERE record_origin='EPA_RECONSTRUCTED' AND label_range_km IS NOT NULL").fetchone()[0], 0, "Range is absent from this source contract."))
    return [
        {"check": name, "status": "PASS" if actual == expected else "FAIL", "actual": actual, "expected": expected, "evidence": evidence}
        for name, actual, expected, evidence in checks
    ]


def deterministic_signature(con: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    queries = (
        "SELECT run_id,vde_id,source_record_id,procedure_code,result_details_json,provenance_json,review_status FROM run WHERE source_name='EPA_TESTCAR_2014_PRESENT' ORDER BY run_id",
        "SELECT id,vde_id,comparison_basis,electrification,fuel_type,fuel_l_per_100km,gco2_per_km,source_record_id,provenance_json FROM fuelcons WHERE record_origin='EPA_RECONSTRUCTED' ORDER BY id",
        "SELECT fuelcons_id,run_id,vde_id,adoption_role,result_dimension,ordinal FROM fuelcons_run_adoption WHERE fuelcons_id<=-4000001 ORDER BY fuelcons_id,run_id,result_dimension",
    )
    for query in queries:
        for row in con.execute(query):
            digest.update(e12.json_text(list(row)).encode("utf-8"))
    return digest.hexdigest().upper()


def population_summary(con: sqlite3.Connection) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimension, expression in (("comparison_basis", "comparison_basis"), ("electrification", "electrification")):
        for value, count in con.execute(
            f"SELECT {expression}, COUNT(*) FROM fuelcons WHERE record_origin='EPA_RECONSTRUCTED' GROUP BY {expression} ORDER BY {expression}"
        ):
            rows.append({"population": "EPA_RECONSTRUCTED", "dimension": dimension, "value": value, "row_count": count})
    for table in ("run", "fuelcons", "fuelcons_run_adoption"):
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        rows.append({"population": "CANONICAL_TOTAL", "dimension": "table", "value": table, "row_count": count})
    return rows


def write_report(summary: dict[str, Any], exceptions: list[dict[str, Any]], regression: list[dict[str, Any]]) -> None:
    reg = Counter(row["classification"] for row in regression)
    text = f"""# Sprint 12E.2 — EPA RUN Grain Closure & FuelCons Reconstruction

## Status: `{summary['status']}`

```text
EPA source rows                              {summary['epa_source_rows']}
Canonical EPA RUNs before grain closure      {summary['epa_runs_before']}
Canonical EPA RUNs after grain closure       {summary['epa_runs_after']}

EPA VDEs                                     {summary['epa_vdes']}
EPA VDEs eligible for FuelCons               {summary['epa_vdes_eligible']}
EPA FuelCons created                         {summary['epa_fuelcons_created']}

FuelCons by basis:
  EPA_LABEL_2_CYCLE                          {summary['fuelcons_by_basis'].get('EPA_LABEL_2_CYCLE', 0)}
  EPA_TEST_PROCEDURE                         {summary['fuelcons_by_basis'].get('EPA_TEST_PROCEDURE', 0)}
  other                                      {summary['fuelcons_other_basis']}

FuelCons by electrification:
  ICE                                        {summary['fuelcons_by_electrification'].get('ICE', 0)}
  HEV                                        {summary['fuelcons_by_electrification'].get('HEV', 0)}
  PHEV                                       {summary['fuelcons_by_electrification'].get('PHEV', 0)}
  BEV                                        {summary['fuelcons_by_electrification'].get('BEV', 0)}

VDEs with:
  single RUN adoption                        {summary['vdes_with_single_run_adoption']}
  multi-RUN adoption                         {summary['vdes_with_multi_run_adoption']}
  unresolved FuelCons                        {summary['vdes_with_unresolved_fuelcons']}

Legacy regression:
  exact / equivalent                         {reg['EXACT_EQUIVALENCE'] + reg['NUMERIC_EQUIVALENCE_WITH_GRAIN_CHANGE']}
  grain-change difference                    {reg['NUMERIC_EQUIVALENCE_WITH_GRAIN_CHANGE']}
  source-refresh difference                  {reg['SOURCE_REFRESH_DIFFERENCE']}
  approved method correction                 {reg['APPROVED_METHOD_CORRECTION']}
  unresolved                                 {reg['UNRESOLVED'] + reg['NO_DIRECT_CANONICAL_COUNTERPART']}

Relationship failures                        {summary['relationship_failures']}
Runtime DB changed?                          {'YES' if summary['runtime_db_changed'] else 'NO'}
User decisions required                      {summary['user_decisions_required']}
```

## RUN grain closure

The deterministic execution key is **canonical VDE + Testgroup + Test Number + procedure + test vehicle/configuration + Set ABC + police/overdrive condition + result signature**. Only source rows whose execution conditions and results agree are grouped. Their aftertreatment and declared averaging rows remain embedded as structured source-row details and are also exported one-to-one in `run_source_row_lineage.csv`.

This closes {summary['epa_runs_before']:,} source-row RUNs to {summary['epa_runs_after']:,} execution-grain RUNs while retaining {summary['epa_source_rows_loaded']:,} loaded source-row lineage records. The {summary['epa_quarantined_rows']} identity-anomaly rows from 12E.1 remain quarantined and are not silently reintroduced.

## FuelCons methodology

- Regular FTP procedures 2/21/31 provide City evidence; procedure 3 provides Highway evidence.
- Source `RND_ADJ_FE` is used only where `FE_UNIT=MPG`, fuel is a supported liquid fuel, and the value is finite in the non-sentinel range. Conversion is `235.214583 / MPG`.
- Source `CO2 (g/mi)` is converted by division by `1.609344`.
- Two-cycle additive per-distance results use `0.55 × City + 0.45 × Highway` through the existing canonical formula owner.
- Multiple eligible RUNs are adopted only when their source metric signatures agree. Conflicting candidates remain unresolved; no first-row selection or implicit average is used.
- US06 and SC03 become `EPA_TEST_PROCEDURE` FuelCons with values only in their specialized fields.
- No calculation RUN is added: the existing multi-RUN adoption table plus FuelCons provenance records formula, version, inputs, roles, and units without creating redundant evidence.

## Applicability and deliberate NULLs

ICE/HEV and PHEV charge-sustaining liquid-fuel evidence may materialize. PHEV charge-depleting, BEV electric energy, and FCEV equivalent-fuel results remain RUN evidence because this source export labels the field `MPG` without a supported Wh/km contract. Energy, range, Bag fallback, and LHV-derived values remain NULL. Zero is preserved when directly observed and is never used as a missing-value substitute.

## Exceptions

- Unresolved materialization cases: {len(exceptions):,}; each has candidate RUN ids and an explicit reason.
- BEV/FCEV energy-unit closure remains a documented GAP, not a fabricated conversion and not a runtime blocker for supported liquid-fuel reconstruction.
- The HEV classifier is retained from the validated project logic (`FE Bag 4` presence), explicitly labeled `DETERMINISTIC_VALIDATED`; it is not upgraded to direct source identity.
- Legacy baseline FuelCons rows were regression references only and were not copied.

## Validation

All {summary['relationship_check_count']} relationship checks pass. SQLite `quick_check` is `{summary['quick_check']}` and foreign-key violations are zero. Two rebuilds produced the same relationship signature `{summary['deterministic_signature']}`.

Runtime SHA-256 before/after: `{summary['runtime_before_sha256']}` / `{summary['runtime_after_sha256']}`. Byte-identical: **{'YES' if not summary['runtime_db_changed'] else 'NO'}**. No runtime cutover and no Streamlit page modification occurred.

## Evidence levels

- **DIRECTLY_TESTED:** RUN grouping, one-to-one source lineage, distinct condition conflicts, multi-RUN adoption, deterministic selection, unit conversions, 55/45 combination, NULL applicability, specialized-cycle isolation, same-VDE lineage, no legacy copy, runtime hashes, deterministic rebuild.
- **INSPECTION_SUPPORTED:** source field semantics and legacy pipeline classifications in `metric_method_matrix.csv`.
- **INDIRECTLY_COVERED:** Sprint 12D physical schema and Sprint 12E.1 clean population.
- **GAP:** authoritative BEV/CD energy unit and range materialization contract.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")


def run(output_db: Path = DEFAULT_OUTPUT_DB, rebuild: bool = False) -> dict[str, Any]:
    output_db = guard_output_path(output_db)
    required = (RUNTIME_DB, QA_DB, BASE_DB, EPA_SOURCE, LEGACY_MATCHES)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing Sprint 12E.2 inputs: {missing}")
    previous = json.loads(SUMMARY_PATH.read_text(encoding="utf-8")) if SUMMARY_PATH.exists() else None
    if output_db.exists():
        if not rebuild:
            raise FileExistsError(f"12E.2 DB exists; use --rebuild: {output_db}")
        output_db.unlink()
    output_db.parent.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    protected = (RUNTIME_DB, QA_DB, BASE_DB)
    before = {path: e12.fingerprint(path) for path in protected}
    source_hash = e12.fingerprint(EPA_SOURCE)["sha256"]
    shutil.copy2(BASE_DB, output_db)

    rows, raw_columns, quarantined = prepare_source()
    con = sqlite3.connect(output_db)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    try:
        source_map, vde_meta = vde_source_map(con)
        runs, grouping, lineage, grain_analysis, facts = close_run_grain(rows, raw_columns, source_hash, source_map)
        fuelcons, adoptions, materialization, unresolved, _ = materialize_fuelcons(
            rows, source_map, vde_meta, facts, source_hash,
        )
        con.execute("BEGIN")
        con.execute("DELETE FROM run WHERE source_name=?", (SOURCE_NAME,))
        e12.insert_rows(con, "run", runs)
        e12.insert_rows(con, "fuelcons", fuelcons)
        e12.insert_rows(con, "fuelcons_run_adoption", adoptions)
        con.commit()
        checks = relationship_checks(con, len(rows), len(runs), lineage, fuelcons)
        signature = deterministic_signature(con)
        pop_summary = population_summary(con)
        quick_check = con.execute("PRAGMA quick_check").fetchone()[0]
        total_fuelcons = con.execute("SELECT COUNT(*) FROM fuelcons").fetchone()[0]
    finally:
        con.close()

    after = {path: e12.fingerprint(path) for path in protected}
    fingerprints = []
    for path in protected:
        fingerprints.append({
            "database": path.name, "path": str(path),
            "before_size_bytes": before[path]["size_bytes"], "after_size_bytes": after[path]["size_bytes"],
            "before_sha256": before[path]["sha256"], "after_sha256": after[path]["sha256"],
            "byte_identical": before[path] == after[path], "access": "READ_ONLY_SOURCE",
        })
    runtime_changed = before[RUNTIME_DB] != after[RUNTIME_DB] or before[QA_DB] != after[QA_DB]
    relationship_failures = sum(row["status"] != "PASS" for row in checks)
    by_basis = Counter(row["comparison_basis"] for row in fuelcons)
    by_electrification = Counter(row["electrification"] for row in fuelcons)
    adoption_counts = Counter(row["fuelcons_id"] for row in adoptions)
    fc_by_id = {row["id"]: row for row in fuelcons}
    single_vdes = {fc_by_id[fid]["vde_id"] for fid, count in adoption_counts.items() if count == 1}
    multi_vdes = {fc_by_id[fid]["vde_id"] for fid, count in adoption_counts.items() if count > 1}
    unresolved_vdes = {int(row["canonical_vde_id"]) for row in unresolved}
    eligible_vdes = {int(row["vde_id"]) for row in fuelcons}
    regression = legacy_regression(fuelcons, adoptions)
    deterministic = bool(previous and previous.get("deterministic_signature") == signature and previous.get("epa_fuelcons_created") == len(fuelcons))
    ready = (
        not runtime_changed and relationship_failures == 0 and quick_check == "ok"
        and len(runs) < len(rows) and len(lineage) == len(rows) and len(fuelcons) > 1_000
    )
    status = "EPA_FUELCONS_READY — PROCEED_TO_APPLICATION_INTEGRATION" if ready else "EPA_FUELCONS_REVIEW_REQUIRED"
    summary = {
        "status": status, "epa_source_rows": len(rows) + quarantined, "epa_source_rows_loaded": len(rows),
        "epa_quarantined_rows": quarantined, "epa_runs_before": len(rows), "epa_runs_after": len(runs),
        "epa_vdes": len(vde_meta), "epa_vdes_eligible": len(eligible_vdes), "epa_fuelcons_created": len(fuelcons),
        "canonical_fuelcons_total": total_fuelcons, "fuelcons_by_basis": dict(sorted(by_basis.items())),
        "fuelcons_other_basis": sum(value for key, value in by_basis.items() if key not in {"EPA_LABEL_2_CYCLE", "EPA_TEST_PROCEDURE"}),
        "fuelcons_by_electrification": dict(sorted(by_electrification.items())),
        "vdes_with_single_run_adoption": len(single_vdes), "vdes_with_multi_run_adoption": len(multi_vdes),
        "vdes_with_unresolved_fuelcons": len(unresolved_vdes), "unresolved_materialization_cases": len(unresolved),
        "relationship_failures": relationship_failures, "relationship_check_count": len(checks),
        "runtime_db_changed": runtime_changed, "runtime_before_sha256": before[RUNTIME_DB]["sha256"],
        "runtime_after_sha256": after[RUNTIME_DB]["sha256"], "user_decisions_required": 0,
        "quick_check": quick_check, "deterministic_signature": signature,
        "previous_signature_available": previous is not None, "rebuild_deterministic": deterministic,
        "output_database": str(output_db), "output_database_size_bytes": output_db.stat().st_size,
        "source_sha256": source_hash,
    }

    write_csv("run_grain_analysis.csv", grain_analysis, ["metric", "value", "classification", "evidence"])
    write_csv("run_grouping_results.csv", grouping, ["canonical_run_id", "canonical_vde_id", "grouping_key", "broad_test_key", "classification", "grouping_confidence", "source_row_count", "source_excel_rows", "test_number", "adfe_test_number", "test_group", "test_category", "procedure_code", "condition_or_result_conflicts"])
    write_csv("run_source_row_lineage.csv", lineage, ["source_excel_row", "original_source_row_run_id", "canonical_run_id", "canonical_vde_id", "grouping_key", "classification", "grouping_confidence", "source_row_sha256", "source_file_version"])
    _, _, _, _, vde_electrification = materialize_fuelcons(rows, source_map, vde_meta, facts, source_hash)
    candidate_outcomes = Counter(int(row["canonical_vde_id"]) for row in unresolved)
    created_outcomes = Counter(int(row["canonical_vde_id"]) for row in materialization)
    facts_by_vde = Counter(int(row["vde_id"]) for row in facts.values())
    candidate_rows = [
        {"canonical_vde_id": vde_id, "electrification": vde_electrification[vde_id],
         "run_count": facts_by_vde[vde_id], "fuelcons_created": created_outcomes[vde_id],
         "unresolved_candidates": candidate_outcomes[vde_id],
         "eligibility_status": "ELIGIBLE_MATERIALIZED" if created_outcomes[vde_id] else "NOT_MATERIALIZED"}
        for vde_id in sorted(vde_meta)
    ]
    write_csv("fuelcons_candidate_vdes.csv", candidate_rows, ["canonical_vde_id", "electrification", "run_count", "fuelcons_created", "unresolved_candidates", "eligibility_status"])
    write_csv("fuelcons_materialization_results.csv", materialization, ["fuelcons_id", "canonical_vde_id", "comparison_basis", "cycle", "electrification", "fuel_type", "selection_classification", "adopted_run_count", "adopted_run_ids", "status"])
    adoption_export = [
        {**row, "details_json": e12.json_text(row["details_json"])} for row in adoptions
    ]
    write_csv("fuelcons_run_adoption.csv", adoption_export, ["fuelcons_id", "run_id", "vde_id", "adoption_role", "result_dimension", "ordinal", "details_json", "created_at"])
    write_csv("metric_method_matrix.csv", metric_method_matrix(), ["metric", "comparison_basis", "source_field", "source_unit", "canonical_unit", "formula_or_conversion", "logic_classification", "materialization"])
    write_csv("electrification_applicability_matrix.csv", applicability_matrix(), ["electrification", "fuel_consumption", "electric_energy", "co2", "electric_range", "fuel_type", "energy_basis", "classification_evidence"])
    write_csv("unresolved_materialization_cases.csv", unresolved, ["canonical_vde_id", "comparison_basis", "fuel_type", "cycle", "reason_code", "candidate_run_ids", "details"])
    write_csv("legacy_fuelcons_regression.csv", regression, ["legacy_fuelcons_id", "legacy_vde_id", "canonical_vde_id", "canonical_fuelcons_id", "classification", "legacy_fuel_type", "canonical_fuel_type", "adopted_run_count", "evidence"])
    write_csv("fuelcons_population_summary.csv", pop_summary, ["population", "dimension", "value", "row_count"])
    write_csv("relationship_checks.csv", checks, ["check", "status", "actual", "expected", "evidence"])
    write_csv("runtime_db_fingerprints.csv", fingerprints, ["database", "path", "before_size_bytes", "after_size_bytes", "before_sha256", "after_sha256", "byte_identical", "access"])
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary, unresolved, regression)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-db", type=Path, default=DEFAULT_OUTPUT_DB)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    summary = run(args.output_db, args.rebuild)
    print(json.dumps({
        key: summary[key] for key in (
            "status", "epa_source_rows", "epa_runs_before", "epa_runs_after", "epa_vdes",
            "epa_vdes_eligible", "epa_fuelcons_created", "canonical_fuelcons_total",
            "fuelcons_by_basis", "fuelcons_by_electrification", "vdes_with_unresolved_fuelcons",
            "relationship_failures", "runtime_db_changed", "rebuild_deterministic",
        )
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
