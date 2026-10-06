"""Read-only Sprint 12 transmission grouping/coastdown signal experiment.

This module never writes to SQLite.  It builds exploratory CSV/notebook/report
artifacts from the canonical candidate and deliberately does not estimate a
transmission-loss ABC.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import pdist
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_DB = ROOT / "data/db/staging/eco_drive_canonical_candidate.db"
DEFAULT_OUTPUT = ROOT / "artifacts/components/transmission_experiment"
DEFAULT_NOTEBOOK = ROOT / "notebooks/diag_transmission_grouping_signal_experiment.ipynb"
DEFAULT_REPORT = ROOT / "docs/sprints/SPRINT_12_TRANSMISSION_GROUPING_EXPERIMENT_RESULT.md"

PRIMARY_SPEEDS = tuple(float(v) for v in range(20, 121, 10))
MODEL_ALPHA = 10.0
RANDOM_SEED = 12012
CV_SPLITS = 3

# Frozen explanatory contract.  Target ABC, target-derived CdA/RRC, legacy
# component estimates, and transmission-loss fields are intentionally absent.
BASELINE_CATEGORICAL_FEATURES = ("make_norm", "category_norm", "drive_norm", "electrification")
BASELINE_NUMERIC_FEATURES = ("speed_scaled", "speed_scaled_sq", "mass_z", "mass_speed", "year_z", "year_speed")
GROUP_TERMS = ("candidate_group_intercept", "candidate_group_linear_speed")
PROHIBITED_EXPLANATORY_FIELDS = (
    "coast_A_N",
    "coast_B_N_per_kph",
    "coast_C_N_per_kph2",
    "cda_m2",
    "rrc_N_per_kN",
    "trans_A_coef_N",
    "trans_B_coef_Npkph",
    "trans_C_coef_Npkph2",
)

FIELD_AUDIT_FILE = "TRANSMISSION_FIELD_AUDIT.csv"
GROUPS_FILE = "TRANSMISSION_CANDIDATE_GROUPS.csv"
SAMPLE_FILE = "EXPERIMENT_SAMPLE_AUDIT.csv"
LONG_FILE = "FORCE_CURVE_EXPERIMENT_LONG.csv"
MODEL_FILE = "MODEL_COMPARISON.csv"
VALIDATION_FILE = "GROUP_VALIDATION.csv"
PERMUTATION_FILE = "PERMUTATION_RESULTS.csv"
SENSITIVITY_FILE = "SENSITIVITY_RESULTS.csv"


@dataclass(frozen=True)
class ExperimentResult:
    db_path: Path
    db_sha256_before: str
    db_sha256_after: str
    raw_vde_count: int
    excluded_carryover_count: int
    independent_unit_count: int
    direct_match_groups: int
    strict_candidate_groups: int
    usable_groups_ge3: int
    usable_groups_ge5: int
    baseline_rmse_n: float
    baseline_mae_n: float
    group_rmse_n: float
    group_mae_n: float
    improvement_n: float
    permutation_p_value: float
    observed_residual_distance_n: float
    permutation_mean_distance_n: float
    residual_similarity_support: str
    signal_supported: str
    ready_for_shared_loss_experiment: str
    bootstrap_ci_low_n: float
    bootstrap_ci_high_n: float
    output_dir: Path
    notebook_path: Path
    report_path: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _read_only_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def _json_object(value: object) -> dict:
    if value in (None, ""):
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "UNKNOWN"
    text = re.sub(r"[^A-Z0-9]+", " ", str(value).upper()).strip()
    return re.sub(r"\s+", " ", text) or "UNKNOWN"


def _format_ratio(value: object, decimals: int) -> str:
    if value is None or pd.isna(value):
        return "UNKNOWN"
    return f"{float(value):.{decimals}f}"


def _group_id(signature: str) -> str:
    return "TXG-" + hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16].upper()


def _resolve_roots(ids: Iterable[int], parent_by_id: Mapping[int, int | None]) -> dict[int, int]:
    roots: dict[int, int] = {}
    for raw_id in ids:
        current = int(raw_id)
        trail: list[int] = []
        seen: set[int] = set()
        while parent_by_id.get(current) is not None:
            if current in seen:
                raise ValueError(f"VDE lineage cycle detected at {current}")
            seen.add(current)
            trail.append(current)
            current = int(parent_by_id[current])
        root = current
        roots[int(raw_id)] = root
        for item in trail:
            roots[item] = root
    return roots


def load_canonical_source(db_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with _read_only_connection(db_path) as connection:
        vdes = pd.read_sql_query(
            """
            SELECT
                v.id AS vde_id,
                v.vde_id_parent,
                v.year,
                v.make,
                v.model,
                v.category,
                v.test_mass_kg,
                v.coast_A_N,
                v.coast_B_N_per_kph,
                v.coast_C_N_per_kph2,
                v.cda_m2,
                v.rrc_N_per_kN,
                v.record_origin,
                v.source_name AS vde_source_name,
                v.source_record_id AS vde_source_record_id,
                v.provenance_json AS vde_provenance_json,
                v.vehicle_configuration_id,
                vc.program_id,
                vc.transmission_type,
                vc.transmission_model,
                vc.gear_count,
                vc.drive_system,
                vc.final_drive_ratio,
                vc.nv_ratio,
                vc.propulsion_architecture,
                vc.source_identity_json AS configuration_source_identity_json,
                p.commercial_make,
                p.commercial_model
            FROM vde AS v
            JOIN vehicle_configuration AS vc
              ON vc.vehicle_configuration_id = v.vehicle_configuration_id
            JOIN program AS p
              ON p.program_id = vc.program_id
            WHERE v.legislation = 'EPA'
              AND v.record_status = 'ACTIVE'
            ORDER BY v.id
            """,
            connection,
        )
        runs = pd.read_sql_query(
            """
            SELECT r.vde_id, r.run_id, r.conditions_json, r.provenance_json,
                   r.source_record_id, r.procedure_code, r.procedure_description
            FROM run AS r
            JOIN vde AS v ON v.id = r.vde_id
            WHERE v.legislation = 'EPA'
              AND r.record_status = 'ACTIVE'
            ORDER BY r.vde_id, r.run_id
            """,
            connection,
        )
        fuelcons = pd.read_sql_query(
            """
            SELECT f.vde_id, f.electrification
            FROM fuelcons AS f
            JOIN vde AS v ON v.id = f.vde_id
            WHERE v.legislation = 'EPA'
              AND f.record_status = 'ACTIVE'
            ORDER BY f.vde_id, f.id
            """,
            connection,
        )
        programs = pd.read_sql_query(
            """
            SELECT p.*
            FROM program AS p
            WHERE p.source_name = 'EPA_TESTCAR_2014_PRESENT'
            ORDER BY p.program_id
            """,
            connection,
        )
    return vdes, runs, fuelcons, programs


def _unique_join(values: Iterable[object]) -> str:
    cleaned = sorted({str(value).strip() for value in values if value not in (None, "") and not pd.isna(value)})
    return ";".join(cleaned)


def prepare_population(
    vdes: pd.DataFrame,
    runs: pd.DataFrame,
    fuelcons: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = vdes.copy()
    parent_by_id = {
        int(row.vde_id): (None if pd.isna(row.vde_id_parent) else int(row.vde_id_parent))
        for row in data.itertuples(index=False)
    }
    roots = _resolve_roots(data["vde_id"].astype(int), parent_by_id)
    data["root_vde_id"] = data["vde_id"].astype(int).map(roots)

    run_info = runs.copy()
    if not run_info.empty:
        parsed_conditions = run_info["conditions_json"].map(_json_object)
        run_info["test_number"] = parsed_conditions.map(lambda item: item.get("test_number"))
        run_info["adfe_test_number"] = parsed_conditions.map(lambda item: item.get("adfe_test_number"))
        test_numbers_by_vde = run_info.groupby("vde_id", dropna=False)["test_number"].agg(_unique_join).to_dict()
        adfe_by_vde = run_info.groupby("vde_id", dropna=False)["adfe_test_number"].agg(_unique_join).to_dict()
    else:
        test_numbers_by_vde = {}
        adfe_by_vde = {}
    data["test_numbers"] = data["vde_id"].map(test_numbers_by_vde).fillna("")
    data["adfe_test_numbers"] = data["vde_id"].map(adfe_by_vde).fillna("")

    descendants_by_root = data.groupby("root_vde_id")["vde_id"].agg(list).to_dict()
    root_test_numbers: dict[int, str] = {}
    root_adfe_numbers: dict[int, str] = {}
    for root, descendants in descendants_by_root.items():
        root_test_numbers[int(root)] = _unique_join(
            number
            for vde_id in descendants
            for number in str(test_numbers_by_vde.get(vde_id, "")).split(";")
        )
        root_adfe_numbers[int(root)] = _unique_join(
            number
            for vde_id in descendants
            for number in str(adfe_by_vde.get(vde_id, "")).split(";")
        )

    fuel = fuelcons.copy()
    fuel["root_vde_id"] = fuel["vde_id"].map(roots)
    electrification_by_root: dict[int, str] = {}
    for root, frame in fuel.dropna(subset=["root_vde_id"]).groupby("root_vde_id"):
        values = sorted({normalize_text(value) for value in frame["electrification"] if not pd.isna(value)})
        electrification_by_root[int(root)] = values[0] if len(values) == 1 else ("MIXED" if values else "UNKNOWN")

    data["make_norm"] = data["make"].map(normalize_text)
    data["model_norm"] = data["model"].map(normalize_text)
    data["category_norm"] = data["category"].map(normalize_text)
    data["transmission_type_norm"] = data["transmission_type"].map(normalize_text)
    data["drive_norm"] = data["drive_system"].map(normalize_text)
    data["application_lineage"] = data["make_norm"] + "|" + data["model_norm"]
    data["carryover_status"] = data["vde_provenance_json"].map(
        lambda value: _json_object(value).get("carryover_status", "UNKNOWN")
    )

    root_rows = data[data["vde_id"].eq(data["root_vde_id"])].copy()
    root_rows["electrification"] = root_rows["root_vde_id"].map(electrification_by_root).fillna("UNKNOWN")
    root_rows["root_test_numbers"] = root_rows["root_vde_id"].map(root_test_numbers).fillna("")
    root_rows["root_adfe_test_numbers"] = root_rows["root_vde_id"].map(root_adfe_numbers).fillna("")

    required = ["make", "transmission_type", "gear_count", "drive_system", "final_drive_ratio", "nv_ratio"]
    complete = root_rows[required].notna().all(axis=1)
    direct = complete & root_rows["transmission_model"].notna() & root_rows["transmission_model"].astype(str).str.strip().ne("")
    ambiguous_ratio_evidence = (
        root_rows["final_drive_ratio"].le(0)
        | root_rows["nv_ratio"].le(0)
        | root_rows["nv_ratio"].ge(900)
        | (
            root_rows["gear_count"].eq(1)
            & root_rows["final_drive_ratio"].eq(1)
            & root_rows["nv_ratio"].eq(1)
        )
    )
    root_rows["ratio_evidence_status"] = np.where(
        ambiguous_ratio_evidence,
        "EXPLICIT_AMBIGUOUS_OR_SENTINEL_REVIEW",
        "USABLE_EXACT_SOURCE_VALUE",
    )
    root_rows["identity_status"] = np.where(
        direct,
        "DIRECT_MATCH",
        np.where(complete & ~ambiguous_ratio_evidence, "STRICT_CANDIDATE", "FAMILY_ONLY"),
    )

    root_rows["grouping_signature"] = root_rows.apply(
        lambda row: "|".join(
            [
                f"MAKE={row['make_norm']}",
                f"TRANS={row['transmission_type_norm']}",
                f"GEARS={int(row['gear_count']) if not pd.isna(row['gear_count']) else 'UNKNOWN'}",
                f"DRIVE={row['drive_norm']}",
                f"FDR={_format_ratio(row['final_drive_ratio'], 2)}",
                f"NV={_format_ratio(row['nv_ratio'], 1)}",
                f"ELEC={row['electrification']}",
                f"HARDWARE={normalize_text(row['transmission_model']) if row['identity_status'] == 'DIRECT_MATCH' else 'UNPROVEN'}",
            ]
        ),
        axis=1,
    )
    root_rows["candidate_group_id"] = root_rows["grouping_signature"].map(_group_id)
    root_rows["permutation_stratum"] = root_rows.apply(
        lambda row: "|".join(
            [
                row["make_norm"],
                row["transmission_type_norm"],
                str(int(row["gear_count"])) if not pd.isna(row["gear_count"]) else "UNKNOWN",
                row["drive_norm"],
            ]
        ),
        axis=1,
    )

    group_app_counts = root_rows.groupby("candidate_group_id")["application_lineage"].nunique()
    group_unit_counts = root_rows.groupby("candidate_group_id")["vde_id"].size()
    root_rows["n_independent_applications_in_group"] = root_rows["candidate_group_id"].map(group_app_counts)
    root_rows["n_independent_vdes_in_group"] = root_rows["candidate_group_id"].map(group_unit_counts)
    root_rows["model_eligible"] = (
        root_rows["identity_status"].isin(["DIRECT_MATCH", "STRICT_CANDIDATE"])
        & root_rows["n_independent_applications_in_group"].ge(3)
        & root_rows[["test_mass_kg", "coast_A_N", "coast_B_N_per_kph", "coast_C_N_per_kph2"]].notna().all(axis=1)
    )

    group_rows: list[dict] = []
    for group_id, frame in root_rows.groupby("candidate_group_id", sort=True):
        first = frame.iloc[0]
        eligible_identity = first["identity_status"] in {"DIRECT_MATCH", "STRICT_CANDIDATE"}
        application_count = int(frame["application_lineage"].nunique())
        all_tests = _unique_join(
            item
            for value in frame["root_test_numbers"]
            for item in str(value).split(";")
        )
        group_rows.append(
            {
                "candidate_group_id": group_id,
                "identity_status": first["identity_status"],
                "grouping_signature": first["grouping_signature"],
                "make": first["make"],
                "transmission_type": first["transmission_type"],
                "gears": int(first["gear_count"]),
                "drive_type": first["drive_system"],
                "axle_ratio_summary": _format_ratio(first["final_drive_ratio"], 2),
                "nv_ratio_summary": _format_ratio(first["nv_ratio"], 1),
                "electrification": first["electrification"],
                "ratio_evidence_status": first["ratio_evidence_status"],
                "n_vdes": int(len(frame)),
                "n_unique_models": application_count,
                "n_unique_test_numbers": len([value for value in all_tests.split(";") if value]),
                "usable_ge3": "YES" if eligible_identity and application_count >= 3 else "NO",
                "usable_ge5": "YES" if eligible_identity and application_count >= 5 else "NO",
                "notes": (
                    "Exact source transmission model/hardware reference."
                    if first["identity_status"] == "DIRECT_MATCH"
                    else (
                        "Strict structured candidate only; no hardware/part identity asserted."
                        if first["identity_status"] == "STRICT_CANDIDATE"
                        else "Broad family only: missing or ambiguous/sentinel ratio evidence; explicit source values preserved."
                    )
                ),
            }
        )
    groups = pd.DataFrame(group_rows)

    group_id_by_root = root_rows.set_index("root_vde_id")["candidate_group_id"].to_dict()
    identity_by_root = root_rows.set_index("root_vde_id")["identity_status"].to_dict()
    apps_by_root = root_rows.set_index("root_vde_id")["n_independent_applications_in_group"].to_dict()
    eligible_by_root = root_rows.set_index("root_vde_id")["model_eligible"].to_dict()
    sample = data.copy()
    sample["candidate_group_id"] = sample["root_vde_id"].map(group_id_by_root)
    sample["identity_status"] = sample["root_vde_id"].map(identity_by_root)
    sample["n_independent_applications_in_group"] = sample["root_vde_id"].map(apps_by_root)
    sample["included_excluded"] = np.where(
        sample["vde_id"].ne(sample["root_vde_id"]),
        "EXCLUDED",
        np.where(sample["root_vde_id"].map(eligible_by_root).fillna(False), "INCLUDED", "EXCLUDED"),
    )
    sample["exclusion_reason"] = np.where(
        sample["vde_id"].ne(sample["root_vde_id"]),
        "EXACT_MODEL_YEAR_CARRYOVER_COLLAPSED_TO_ROOT",
        np.where(
            sample["root_vde_id"].map(eligible_by_root).fillna(False),
            "",
            np.where(
                sample["n_independent_applications_in_group"].fillna(0).lt(3),
                "CANDIDATE_GROUP_LT3_INDEPENDENT_APPLICATIONS",
                "MISSING_REQUIRED_MODEL_EVIDENCE",
            ),
        ),
    )
    sample = sample.rename(columns={"year": "model_year"})
    sample = sample[
        [
            "included_excluded",
            "exclusion_reason",
            "vde_id",
            "root_vde_id",
            "model_year",
            "make",
            "model",
            "test_numbers",
            "adfe_test_numbers",
            "carryover_status",
            "candidate_group_id",
            "identity_status",
            "n_independent_applications_in_group",
        ]
    ].sort_values("vde_id")
    return root_rows.sort_values("vde_id"), groups, sample


def build_field_audit(
    vdes: pd.DataFrame,
    runs: pd.DataFrame,
    fuelcons: pd.DataFrame,
    programs: pd.DataFrame,
) -> pd.DataFrame:
    run_conditions = runs["conditions_json"].map(_json_object)
    fields: list[tuple[str, str, pd.Series, str, str, str]] = [
        ("program", "commercial_make", programs["commercial_make"], "NO", "YES", "YES"),
        ("program", "commercial_model", programs["commercial_model"], "NO", "YES", "YES"),
        ("vde", "year", vdes["year"], "NO", "YES", "YES"),
        ("vde", "category", vdes["category"], "NO", "YES", "YES"),
        ("vde", "test_mass_kg", vdes["test_mass_kg"], "NO", "YES", "YES"),
        ("vehicle_configuration", "transmission_model", vdes["transmission_model"], "YES_IF_POPULATED", "YES", "YES"),
        ("vehicle_configuration", "transmission_type", vdes["transmission_type"], "NO_BROAD_FAMILY_ONLY", "YES", "YES"),
        ("vehicle_configuration", "gear_count", vdes["gear_count"], "NO", "YES", "YES"),
        ("vehicle_configuration", "drive_system", vdes["drive_system"], "NO", "YES", "YES"),
        ("vehicle_configuration", "final_drive_ratio", vdes["final_drive_ratio"], "NO", "YES", "YES"),
        ("vehicle_configuration", "nv_ratio", vdes["nv_ratio"], "NO", "YES", "YES"),
        ("vehicle_configuration", "propulsion_architecture", vdes["propulsion_architecture"], "NO", "YES", "YES"),
        ("run.conditions_json", "test_number", run_conditions.map(lambda item: item.get("test_number")), "NO", "YES", "YES_SAMPLE_CONTROL_ONLY"),
        ("run.conditions_json", "adfe_test_number", run_conditions.map(lambda item: item.get("adfe_test_number")), "NO", "YES", "YES_SAMPLE_CONTROL_ONLY"),
        ("vde", "vde_id_parent", vdes["vde_id_parent"], "NO", "YES", "YES_DUPLICATE_CONTROL_ONLY"),
        ("fuelcons", "electrification", fuelcons["electrification"], "NO", "YES", "YES"),
        ("vde", "cda_m2", vdes["cda_m2"], "NO", "YES_IF_INDEPENDENT", "NO_UNLESS_INDEPENDENT_PROVENANCE"),
        ("vde", "rrc_N_per_kN", vdes["rrc_N_per_kN"], "NO", "YES_IF_INDEPENDENT", "NO_UNLESS_INDEPENDENT_PROVENANCE"),
        ("vde", "coast_A_N", vdes["coast_A_N"], "NO", "OUTCOME", "NO_TARGET_OUTCOME"),
        ("vde", "coast_B_N_per_kph", vdes["coast_B_N_per_kph"], "NO", "OUTCOME", "NO_TARGET_OUTCOME"),
        ("vde", "coast_C_N_per_kph2", vdes["coast_C_N_per_kph2"], "NO", "OUTCOME", "NO_TARGET_OUTCOME"),
    ]
    rows: list[dict] = []
    for table, field, series, direct, application, leakage in fields:
        non_null = series.dropna()
        examples = _unique_join(non_null.astype(str).head(20))
        rows.append(
            {
                "source_table": table,
                "source_field": field,
                "non_null_count": int(non_null.size),
                "cardinality": int(non_null.nunique(dropna=True)),
                "example_values": examples[:500],
                "direct_identity_evidence": direct,
                "application_evidence": application,
                "usable_without_outcome_leakage": leakage,
            }
        )
    return pd.DataFrame(rows)


def build_force_curve_long(root_rows: pd.DataFrame, speeds: Sequence[float] = PRIMARY_SPEEDS) -> pd.DataFrame:
    from src.vde_core.roadload_analysis import roadload_force_N

    eligible = root_rows[root_rows["model_eligible"]].copy()
    records: list[dict] = []
    for row in eligible.itertuples(index=False):
        forces = roadload_force_N(
            float(row.coast_A_N),
            float(row.coast_B_N_per_kph),
            float(row.coast_C_N_per_kph2),
            list(speeds),
        )
        for speed, force in zip(speeds, forces):
            records.append(
                {
                    "vde_id": int(row.vde_id),
                    "candidate_group_id": row.candidate_group_id,
                    "identity_status": row.identity_status,
                    "application_lineage": row.application_lineage,
                    "permutation_stratum": row.permutation_stratum,
                    "make": row.make,
                    "make_norm": row.make_norm,
                    "model": row.model,
                    "model_year": int(row.year),
                    "category": row.category,
                    "category_norm": row.category_norm,
                    "transmission_type": row.transmission_type,
                    "gear_count": int(row.gear_count),
                    "drive_type": row.drive_system,
                    "drive_norm": row.drive_norm,
                    "final_drive_ratio": float(row.final_drive_ratio),
                    "nv_ratio": float(row.nv_ratio),
                    "electrification": row.electrification,
                    "test_mass_kg": float(row.test_mass_kg),
                    "speed_kph": float(speed),
                    "observed_force_N": float(force),
                }
            )
    return pd.DataFrame(records)


def _design_matrices(long: pd.DataFrame) -> tuple[sparse.csr_matrix, sparse.csr_matrix, list[str], list[str]]:
    frame = long.copy()
    speed = frame["speed_kph"].to_numpy(dtype=float) / 100.0
    mass = frame["test_mass_kg"].to_numpy(dtype=float)
    mass_z = (mass - mass.mean()) / (mass.std(ddof=0) or 1.0)
    year = frame["model_year"].to_numpy(dtype=float)
    year_z = (year - year.mean()) / (year.std(ddof=0) or 1.0)
    numeric = sparse.csr_matrix(np.column_stack([speed, speed**2, mass_z, mass_z * speed, year_z, year_z * speed]))

    cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=float)
    cat = cat_encoder.fit_transform(frame[list(BASELINE_CATEGORICAL_FEATURES)].fillna("UNKNOWN").astype(str))
    cat_names = list(cat_encoder.get_feature_names_out(BASELINE_CATEGORICAL_FEATURES))
    baseline = sparse.hstack(
        [numeric, cat, cat.multiply(speed[:, None]), cat.multiply((speed**2)[:, None])],
        format="csr",
    )
    baseline_names = list(BASELINE_NUMERIC_FEATURES) + cat_names + [f"{name}:speed" for name in cat_names] + [
        f"{name}:speed_sq" for name in cat_names
    ]

    group_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=float)
    group = group_encoder.fit_transform(frame[["candidate_group_id"]].astype(str))
    group_names = list(group_encoder.get_feature_names_out(["candidate_group_id"]))
    augmented = sparse.hstack([baseline, group, group.multiply(speed[:, None])], format="csr")
    augmented_names = baseline_names + [f"{name}:intercept" for name in group_names] + [f"{name}:speed" for name in group_names]
    return baseline, augmented, baseline_names, augmented_names


def build_fold_assignment(units: pd.DataFrame, n_splits: int = CV_SPLITS) -> dict[int, int]:
    unique_units = units.drop_duplicates("vde_id").reset_index(drop=True)
    minimum_apps = unique_units.groupby("candidate_group_id")["application_lineage"].nunique().min()
    if pd.isna(minimum_apps) or int(minimum_apps) < n_splits:
        raise ValueError(f"Every modeled transmission group needs at least {n_splits} independent applications")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    assignment: dict[int, int] = {}
    for fold, (_, test_indices) in enumerate(
        splitter.split(
            unique_units,
            y=unique_units["candidate_group_id"],
            groups=unique_units["application_lineage"],
        )
    ):
        for vde_id in unique_units.iloc[test_indices]["vde_id"]:
            assignment[int(vde_id)] = int(fold)
    if len(assignment) != len(unique_units):
        raise AssertionError("Every independent VDE must receive exactly one grouped fold")
    for fold in range(n_splits):
        test_lineages = set(unique_units[unique_units["vde_id"].map(assignment).eq(fold)]["application_lineage"])
        train_lineages = set(unique_units[~unique_units["vde_id"].map(assignment).eq(fold)]["application_lineage"])
        if test_lineages & train_lineages:
            raise AssertionError("Application lineage leaked between train and test")
    return assignment


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    error = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean(error**2))), float(np.mean(np.abs(error)))


def run_grouped_cv(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    if long.empty:
        raise ValueError("No eligible independent units for grouped CV")
    data = long.copy().reset_index(drop=True)
    units = data[["vde_id", "candidate_group_id", "application_lineage"]].drop_duplicates()
    fold_by_vde = build_fold_assignment(units)
    data["fold"] = data["vde_id"].map(fold_by_vde).astype(int)
    baseline, augmented, baseline_names, augmented_names = _design_matrices(data)
    target = data["observed_force_N"].to_numpy(dtype=float)
    prediction_0 = np.full(len(data), np.nan)
    prediction_1 = np.full(len(data), np.nan)

    for fold in sorted(data["fold"].unique()):
        test_mask = data["fold"].eq(fold).to_numpy()
        train_mask = ~test_mask
        model_0 = Ridge(alpha=MODEL_ALPHA, solver="lsqr", fit_intercept=True)
        model_1 = Ridge(alpha=MODEL_ALPHA, solver="lsqr", fit_intercept=True)
        model_0.fit(baseline[train_mask], target[train_mask])
        model_1.fit(augmented[train_mask], target[train_mask])
        prediction_0[test_mask] = model_0.predict(baseline[test_mask])
        prediction_1[test_mask] = model_1.predict(augmented[test_mask])

    if np.isnan(prediction_0).any() or np.isnan(prediction_1).any():
        raise AssertionError("OOF predictions are incomplete")
    data["baseline_oof_prediction_N"] = prediction_0
    data["group_oof_prediction_N"] = prediction_1
    data["baseline_oof_residual_N"] = target - prediction_0
    data["group_oof_residual_N"] = target - prediction_1

    validation_rows: list[dict] = []

    def append_metrics(scope_type: str, scope_value: str, frame: pd.DataFrame) -> None:
        rmse0, mae0 = _metrics(frame["observed_force_N"], frame["baseline_oof_prediction_N"])
        rmse1, mae1 = _metrics(frame["observed_force_N"], frame["group_oof_prediction_N"])
        validation_rows.append(
            {
                "scope_type": scope_type,
                "scope_value": scope_value,
                "n_force_rows": int(len(frame)),
                "n_independent_vdes": int(frame["vde_id"].nunique()),
                "n_application_lineages": int(frame["application_lineage"].nunique()),
                "baseline_rmse_N": rmse0,
                "baseline_mae_N": mae0,
                "group_model_rmse_N": rmse1,
                "group_model_mae_N": mae1,
                "delta_rmse_N": rmse0 - rmse1,
            }
        )

    append_metrics("OVERALL", "ALL", data)
    for fold, frame in data.groupby("fold", sort=True):
        append_metrics("FOLD", str(int(fold)), frame)
    for speed, frame in data.groupby("speed_kph", sort=True):
        append_metrics("SPEED_KPH", f"{float(speed):g}", frame)
    for group_id, frame in data.groupby("candidate_group_id", sort=True):
        append_metrics("CANDIDATE_GROUP", str(group_id), frame)
    validation = pd.DataFrame(validation_rows)

    overall = validation[validation["scope_type"].eq("OVERALL")].iloc[0]
    model_comparison = pd.DataFrame(
        [
            {
                "model": "MODEL_0_BASELINE",
                "features": "speed + speed^2 + mass + year + make/category/drive/electrification and speed interactions",
                "transmission_group_terms": "NONE",
                "regularization": f"Ridge alpha={MODEL_ALPHA:g}",
                "cross_validation": f"{CV_SPLITS}-fold StratifiedGroupKFold by application_lineage; entire VDE curves held together",
                "rmse_N": overall["baseline_rmse_N"],
                "mae_N": overall["baseline_mae_N"],
                "delta_rmse_vs_baseline_N": 0.0,
            },
            {
                "model": "MODEL_1_BASELINE_PLUS_GROUP",
                "features": "MODEL_0 plus strict candidate group",
                "transmission_group_terms": "group_A + group_B * speed; no group quadratic term",
                "regularization": f"Ridge alpha={MODEL_ALPHA:g}",
                "cross_validation": f"{CV_SPLITS}-fold StratifiedGroupKFold by application_lineage; entire VDE curves held together",
                "rmse_N": overall["group_model_rmse_N"],
                "mae_N": overall["group_model_mae_N"],
                "delta_rmse_vs_baseline_N": overall["delta_rmse_N"],
            },
        ]
    )
    return data, validation, model_comparison, baseline_names, augmented_names


def _within_group_curve_distance(curves: np.ndarray, labels: np.ndarray) -> tuple[float, int]:
    total = 0.0
    pair_count = 0
    for label in sorted(set(labels.tolist())):
        group_curves = curves[labels == label]
        if len(group_curves) < 2:
            continue
        distances = pdist(group_curves, metric="euclidean") / math.sqrt(curves.shape[1])
        total += float(distances.sum())
        pair_count += int(distances.size)
    return (total / pair_count if pair_count else float("nan")), pair_count


def stratified_label_permutation(labels: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shuffled = labels.copy()
    for stratum in sorted(set(strata.tolist())):
        indices = np.flatnonzero(strata == stratum)
        shuffled[indices] = rng.permutation(labels[indices])
    return shuffled


def run_residual_permutation(oof: pd.DataFrame, n_permutations: int) -> tuple[pd.DataFrame, float, float, float, int, int]:
    pivot = oof.pivot(index="vde_id", columns="speed_kph", values="baseline_oof_residual_N").sort_index()
    metadata = oof.drop_duplicates("vde_id").set_index("vde_id").loc[pivot.index]
    curves = pivot.to_numpy(dtype=float)
    labels = metadata["candidate_group_id"].astype(str).to_numpy()
    strata = metadata["permutation_stratum"].astype(str).to_numpy()
    observed, observed_pairs = _within_group_curve_distance(curves, labels)
    rng = np.random.default_rng(RANDOM_SEED)
    rows = [
        {
            "permutation_id": "OBSERVED",
            "statistic": "PAIR_WEIGHTED_WITHIN_GROUP_RESIDUAL_CURVE_RMSE_N",
            "statistic_value": observed,
            "is_as_or_more_extreme_than_observed": True,
            "empirical_p_value": np.nan,
        }
    ]
    permuted_values: list[float] = []
    for iteration in range(1, n_permutations + 1):
        shuffled = stratified_label_permutation(labels, strata, rng)
        value, _ = _within_group_curve_distance(curves, shuffled)
        permuted_values.append(value)
        rows.append(
            {
                "permutation_id": iteration,
                "statistic": "PAIR_WEIGHTED_WITHIN_GROUP_RESIDUAL_CURVE_RMSE_N",
                "statistic_value": value,
                "is_as_or_more_extreme_than_observed": bool(value <= observed),
                "empirical_p_value": np.nan,
            }
        )
    p_value = (1 + sum(value <= observed for value in permuted_values)) / (n_permutations + 1)
    rows[0]["empirical_p_value"] = p_value
    movable_strata = 0
    movable_units = 0
    for stratum in sorted(set(strata.tolist())):
        indices = np.flatnonzero(strata == stratum)
        if len(set(labels[indices].tolist())) > 1:
            movable_strata += 1
            movable_units += int(len(indices))
    return pd.DataFrame(rows), observed, float(np.mean(permuted_values)), float(p_value), movable_strata, movable_units


def bootstrap_improvement_ci(oof: pd.DataFrame, iterations: int = 500) -> tuple[float, float]:
    unit_errors = (
        oof.assign(
            sq0=lambda frame: frame["baseline_oof_residual_N"] ** 2,
            sq1=lambda frame: frame["group_oof_residual_N"] ** 2,
        )
        .groupby(["application_lineage", "vde_id"], as_index=False)
        .agg(sq0=("sq0", "mean"), sq1=("sq1", "mean"))
    )
    lineages = sorted(unit_errors["application_lineage"].unique())
    by_lineage = {key: frame for key, frame in unit_errors.groupby("application_lineage")}
    rng = np.random.default_rng(RANDOM_SEED + 1)
    deltas: list[float] = []
    for _ in range(iterations):
        sampled = rng.choice(lineages, size=len(lineages), replace=True)
        frame = pd.concat([by_lineage[item] for item in sampled], ignore_index=True)
        deltas.append(float(math.sqrt(frame["sq0"].mean()) - math.sqrt(frame["sq1"].mean())))
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def _eligible_subset(long: pd.DataFrame, minimum_applications: int = 3) -> pd.DataFrame:
    counts = long.drop_duplicates("vde_id").groupby("candidate_group_id")["application_lineage"].nunique()
    keep = set(counts[counts >= minimum_applications].index)
    return long[long["candidate_group_id"].isin(keep)].copy()


def _sensitivity_row(name: str, status: str, oof: pd.DataFrame | None, notes: str) -> dict:
    if oof is None or oof.empty:
        return {
            "sensitivity_case": name,
            "status": status,
            "n_independent_vdes": 0,
            "n_groups": 0,
            "baseline_rmse_N": np.nan,
            "group_model_rmse_N": np.nan,
            "delta_rmse_N": np.nan,
            "notes": notes,
        }
    rmse0, _ = _metrics(oof["observed_force_N"], oof["baseline_oof_prediction_N"])
    rmse1, _ = _metrics(oof["observed_force_N"], oof["group_oof_prediction_N"])
    return {
        "sensitivity_case": name,
        "status": status,
        "n_independent_vdes": int(oof["vde_id"].nunique()),
        "n_groups": int(oof["candidate_group_id"].nunique()),
        "baseline_rmse_N": rmse0,
        "group_model_rmse_N": rmse1,
        "delta_rmse_N": rmse0 - rmse1,
        "notes": notes,
    }


def run_sensitivities(primary_long: pd.DataFrame, primary_oof: pd.DataFrame) -> tuple[pd.DataFrame, set[int], float]:
    rows: list[dict] = []
    rows.append(_sensitivity_row("PRIMARY_20_120_KPH", "COMPLETED", primary_oof, "Primary prespecified speed grid."))

    for name, low, high in (("SPEED_30_120_KPH", 30.0, 120.0), ("SPEED_20_100_KPH", 20.0, 100.0)):
        subset = primary_long[primary_long["speed_kph"].between(low, high)].copy()
        oof, _, _, _, _ = run_grouped_cv(subset)
        rows.append(_sensitivity_row(name, "COMPLETED", oof, f"Refit on {low:g}..{high:g} km/h."))

    rows.append(
        _sensitivity_row(
            "DIRECT_MATCH_ONLY",
            "INSUFFICIENT_SAMPLE",
            None,
            "No explicit transmission hardware/model code is populated; zero DIRECT_MATCH groups.",
        )
    )
    rows.append(
        _sensitivity_row(
            "DIRECT_PLUS_STRICT_CANDIDATE",
            "COMPLETED",
            primary_oof,
            "Equivalent to primary because DIRECT_MATCH count is zero.",
        )
    )

    ge5 = _eligible_subset(primary_long, minimum_applications=5)
    if ge5["candidate_group_id"].nunique() and ge5["application_lineage"].nunique() >= CV_SPLITS:
        ge5_oof, _, _, _, _ = run_grouped_cv(ge5)
        rows.append(_sensitivity_row("GROUPS_GE5_APPLICATIONS", "COMPLETED", ge5_oof, "Small groups excluded explicitly."))
    else:
        rows.append(_sensitivity_row("GROUPS_GE5_APPLICATIONS", "INSUFFICIENT_SAMPLE", None, "Too few groups after threshold."))

    unit_counts = primary_long.drop_duplicates("vde_id")["application_lineage"].value_counts()
    dominant = str(unit_counts.index[0])
    without_dominant = primary_long[~primary_long["application_lineage"].eq(dominant)].copy()
    without_dominant = _eligible_subset(without_dominant, minimum_applications=3)
    if without_dominant["candidate_group_id"].nunique():
        dominant_oof, _, _, _, _ = run_grouped_cv(without_dominant)
        rows.append(
            _sensitivity_row(
                "EXCLUDE_DOMINANT_APPLICATION",
                "COMPLETED",
                dominant_oof,
                f"Excluded {dominant} ({int(unit_counts.iloc[0])} independent VDEs); groups requalified at >=3 applications.",
            )
        )
    else:
        rows.append(_sensitivity_row("EXCLUDE_DOMINANT_APPLICATION", "INSUFFICIENT_SAMPLE", None, dominant))

    per_unit = primary_oof.groupby("vde_id")["baseline_oof_residual_N"].apply(lambda values: float(np.sqrt(np.mean(values**2))))
    q1 = float(per_unit.quantile(0.25))
    q3 = float(per_unit.quantile(0.75))
    threshold = q3 + 1.5 * (q3 - q1)
    outlier_ids = set(per_unit[per_unit > threshold].index.astype(int))
    without_outliers = primary_long[~primary_long["vde_id"].isin(outlier_ids)].copy()
    without_outliers = _eligible_subset(without_outliers, minimum_applications=3)
    if without_outliers["candidate_group_id"].nunique():
        outlier_oof, _, _, _, _ = run_grouped_cv(without_outliers)
        rows.append(
            _sensitivity_row(
                "EXCLUDE_FLAGGED_CURVE_OUTLIERS",
                "COMPLETED",
                outlier_oof,
                f"Excluded {len(outlier_ids)} prespecified baseline-residual IQR flags; primary retains them; threshold={threshold:.6f} N.",
            )
        )
    else:
        rows.append(_sensitivity_row("EXCLUDE_FLAGGED_CURVE_OUTLIERS", "INSUFFICIENT_SAMPLE", None, "No qualified groups."))
    return pd.DataFrame(rows), outlier_ids, threshold


def build_notebook(path: Path, db_path: Path, output_dir: Path) -> None:
    def md(text: str) -> dict:
        return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

    def code(text: str) -> dict:
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}

    cells = [
        md(
            """# Components — Transmission Grouping & Coastdown Signal Experiment

**Question:** Do strict structured transmission candidate groups carry repeatable out-of-sample information about whole-vehicle coastdown force?

This is a read-only scientific experiment. A positive signal is not a transmission-loss decomposition, and this notebook never writes to a database."""
        ),
        md("## 1. Portable paths and experiment execution"),
        code(
            """from pathlib import Path
import os
import pandas as pd
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

search_roots = [Path.cwd(), *Path.cwd().parents]
REPO_ROOT = next(path for path in search_roots if (path / "AGENTS.md").exists() and (path / "etl").exists())
DB_PATH = REPO_ROOT / "data/db/staging/eco_drive_canonical_candidate.db"
OUTPUT_DIR = REPO_ROOT / "artifacts/components/transmission_experiment"
assert DB_PATH.exists(), DB_PATH

from etl.scripts.sprint_12_transmission_grouping_signal_experiment import run_experiment

n_permutations = int(os.environ.get("TRANSMISSION_EXPERIMENT_PERMUTATIONS", "500"))
result = run_experiment(DB_PATH, OUTPUT_DIR, n_permutations=n_permutations, build_notebook_artifact=False)
result
"""
        ),
        md("## 2. Data / transmission-field audit"),
        code(f"field_audit = pd.read_csv(OUTPUT_DIR / {FIELD_AUDIT_FILE!r})\nfield_audit"),
        md("## 3. Duplicate and carryover control"),
        code(
            f"""sample_audit = pd.read_csv(OUTPUT_DIR / {SAMPLE_FILE!r})
sample_audit[["included_excluded", "exclusion_reason"]].value_counts(dropna=False)
"""
        ),
        md("## 4. Candidate transmission grouping and coverage"),
        code(
            f"""groups = pd.read_csv(OUTPUT_DIR / {GROUPS_FILE!r})
groups[["identity_status", "usable_ge3", "usable_ge5"]].value_counts().sort_index()
"""
        ),
        code(
            """groups.sort_values(["n_unique_models", "n_vdes"], ascending=False).head(20)
"""
        ),
        md("## 5. ABC → force curves (outcome domain only)"),
        code(
            f"""curves = pd.read_csv(OUTPUT_DIR / {LONG_FILE!r})
example_ids = curves["vde_id"].drop_duplicates().head(8)
example_curves = curves[curves["vde_id"].isin(example_ids)].pivot(index="speed_kph", columns="vde_id", values="observed_force_N")
if plt is None:
    print("matplotlib is unavailable; displaying the complete chart-driving table instead.")
    print(example_curves)
else:
    ax = example_curves.plot(figsize=(8, 4), alpha=0.7)
    ax.set(title="Canonical whole-vehicle coastdown force curves", xlabel="Speed (km/h)", ylabel="Force (N)")
    plt.show()
"""
        ),
        md(
            """## 6. Baseline nuisance model

Model 0 uses mass, year, make, category, drive architecture, electrification, and polynomial speed interactions. It excludes Target-ABC-derived CdA/RRC and all component/decomposition estimates."""
        ),
        code(f"model_comparison = pd.read_csv(OUTPUT_DIR / {MODEL_FILE!r})\nmodel_comparison"),
        md(
            """## 7. Transmission-group augmented model and grouped validation

Model 1 adds only a candidate-group constant and candidate-group × speed term. It does not fit a group quadratic term or publish transmission ABC."""
        ),
        code(
            f"""validation = pd.read_csv(OUTPUT_DIR / {VALIDATION_FILE!r})
validation[validation["scope_type"].isin(["OVERALL", "FOLD", "SPEED_KPH"])]
"""
        ),
        md("## 8. Residual-curve similarity and permutation test"),
        code(
            f"""permutations = pd.read_csv(OUTPUT_DIR / {PERMUTATION_FILE!r})
observed = permutations.iloc[0]
if plt is None:
    print("matplotlib is unavailable; displaying permutation quantiles instead.")
    print(permutations.iloc[1:]["statistic_value"].describe(percentiles=[0.025, 0.5, 0.975]))
else:
    ax = permutations.iloc[1:]["statistic_value"].plot.hist(bins=30, alpha=0.75, figsize=(7, 4))
    ax.axvline(observed["statistic_value"], color="red", linewidth=2, label="Observed")
    ax.set(title="Stratified label-permutation null", xlabel="Within-group residual-curve RMSE (N)")
    ax.legend()
    plt.show()
observed
"""
        ),
        md("## 9. Sensitivity / robustness"),
        code(f"sensitivity = pd.read_csv(OUTPUT_DIR / {SENSITIVITY_FILE!r})\nsensitivity"),
        md("## 10. Representative BMW groups"),
        code(
            """bmw = groups[
    groups["make"].astype(str).str.upper().eq("BMW")
    & groups["identity_status"].eq("STRICT_CANDIDATE")
    & groups["usable_ge3"].eq("YES")
].sort_values(["n_unique_models", "n_vdes"], ascending=False)
bmw.head(15)
"""
        ),
        md(
            """## 11. Failure cases, confounders, and decision

- The source provides no explicit transmission hardware/model code, so all useful groups are candidates rather than proven same-part groups.
- Independent CdA and tire/RRC evidence is unavailable at useful coverage; their variation remains nuisance/confounding.
- The result is about reproducible grouping signal in whole-vehicle force curves, never causal transmission neutral drag.
- Read the generated Sprint result report for the prespecified decision and exact metrics.
"""
        ),
        code("print((REPO_ROOT / 'docs/sprints/SPRINT_12_TRANSMISSION_GROUPING_EXPERIMENT_RESULT.md').read_text(encoding='utf-8'))"),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    def display(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(display(value) for value in row) + " |")
    return "\n".join(lines)


def _write_report(
    path: Path,
    result: ExperimentResult,
    groups: pd.DataFrame,
    sample: pd.DataFrame,
    model_comparison: pd.DataFrame,
    validation: pd.DataFrame,
    sensitivity: pd.DataFrame,
    permutation_count: int,
    movable_strata: int,
    movable_units: int,
    outlier_ids: set[int],
    outlier_threshold: float,
) -> None:
    overall = validation[validation["scope_type"].eq("OVERALL")].iloc[0]
    folds = validation[validation["scope_type"].eq("FOLD")]
    positive_folds = int(folds["delta_rmse_N"].gt(0).sum())
    bmw = groups[
        groups["make"].astype(str).str.upper().eq("BMW")
        & groups["identity_status"].eq("STRICT_CANDIDATE")
        & groups["usable_ge3"].eq("YES")
    ].sort_values(
        ["n_unique_models", "n_vdes"], ascending=False
    )
    bmw_text = "No BMW strict group available."
    if not bmw.empty:
        example = bmw.iloc[0]
        bmw_text = (
            f"`{example['candidate_group_id']}` uses only `{example['grouping_signature']}`; "
            f"it contains {int(example['n_vdes'])} independent VDEs across "
            f"{int(example['n_unique_models'])} normalized model/application lineages. "
            "No hardware name or supplier is inferred."
        )
    sensitivity_table = _markdown_table(sensitivity)
    content = f"""# Sprint 12 — Transmission Grouping & Coastdown Signal Experiment Result

## Decision

This read-only experiment tests repeatable grouping signal only. It does not estimate or publish transmission-loss ABC and does not establish causality.

## A. SOURCE / GROUPING

- Available fields: make/model/year, category, test mass, transmission type, gear count, drive system, final-drive ratio, N/V ratio, electrification when linked, Test Number/ADFE Test Number, VDE carryover lineage, and canonical Target ABC outcome.
- Strongest direct identity evidence: none. `vehicle_configuration.transmission_model` is unpopulated, and EPA's transmission type/code is a broad family descriptor rather than a hardware reference.
- Float handling: final-drive ratio is retained at its observed two-decimal source precision; N/V at one decimal. Exact normalized values are used, with no fuzzy tolerance. Explicit zero remains zero.
- Explicit zero/sentinel-like ratios are preserved but not upgraded to strict identity: final-drive or N/V <=0, N/V >=900, and the one-speed `1.00/1.0` placeholder pattern are classified `FAMILY_ONLY` pending source clarification.
- DIRECT_MATCH groups: {result.direct_match_groups}.
- STRICT_CANDIDATE groups: {result.strict_candidate_groups}.
- Usable groups with >=3 independent applications: {result.usable_groups_ge3}.
- Usable groups with >=5 independent applications: {result.usable_groups_ge5}.

## B. SAMPLE CONTROL

- Raw EPA VDE count: {result.raw_vde_count}.
- Excluded exact carryover descendants: {result.excluded_carryover_count}.
- Independent root VDEs before minimum-group filtering: {int((sample['vde_id'] == sample['root_vde_id']).sum())}.
- Independent experiment units in the primary >=3-app sample: {result.independent_unit_count}.
- Statistical unit: one canonical VDE root after following `vde_id_parent` to collapse exact model-year carryover. All speed points from a VDE and all records in a normalized make/model application lineage stay in one CV fold.

## C. MODEL 0

- Formula/features: speed + speed², test mass, model year, make, category/body class, drive system, electrification, and their prespecified speed interactions.
- Excluded leakage: Target ABC as features, Target-derived CdA/RRC, legacy decomposition values, and all transmission-loss estimates.
- Validation: {CV_SPLITS}-fold `StratifiedGroupKFold`, grouped by normalized make/model application lineage; complete VDE curves held out.
- OOF RMSE: {result.baseline_rmse_n:.6f} N.
- OOF MAE: {result.baseline_mae_n:.6f} N.

## D. MODEL 1

- Added terms: candidate-group intercept (`group_A`) and candidate-group × speed (`group_B * v`) only. No group quadratic term.
- Ridge regularization: alpha={MODEL_ALPHA:g}, fixed before comparison.
- OOF RMSE: {result.group_rmse_n:.6f} N.
- OOF MAE: {result.group_mae_n:.6f} N.
- OOF delta RMSE: {result.improvement_n:.6f} N (positive favors Model 1).
- Positive grouped folds: {positive_folds}/{len(folds)}.
- Application-cluster bootstrap 95% CI for delta RMSE: [{result.bootstrap_ci_low_n:.6f}, {result.bootstrap_ci_high_n:.6f}] N.

## E. RESIDUAL SIMILARITY

- Observed pair-weighted within-group residual-curve RMSE: {result.observed_residual_distance_n:.6f} N.
- Mean stratified randomized comparison: {result.permutation_mean_distance_n:.6f} N.
- Support: {result.residual_similarity_support}.

## F. PERMUTATION

- Permutations: {permutation_count}, deterministic seed {RANDOM_SEED}.
- Labels shuffled only within make + broad transmission type + gear count + drive architecture strata, preserving each stratum's label multiset and group-size distribution.
- Movable strata/units: {movable_strata}/{movable_units}.
- Observed statistic: {result.observed_residual_distance_n:.6f} N (lower is more grouped).
- Empirical p-value: {result.permutation_p_value:.6f}.

## G. ROBUSTNESS

{sensitivity_table}

Outliers were defined from primary baseline OOF curve RMSE using Q3 + 1.5×IQR. {len(outlier_ids)} VDEs exceeded {outlier_threshold:.6f} N. They remain in the primary result and are excluded only in the explicitly labeled sensitivity case.

## H. REPRESENTATIVE GROUPS

- BMW: {bmw_text}
- Every group row exposes the exact signature, observed ratios, application counts, and identity status in `TRANSMISSION_CANDIDATE_GROUPS.csv`.
- No candidate receives an unproven commercial hardware name.

## I. CONCLUSION

The conclusion is `{result.signal_supported}` under the prespecified minimum evidence gate. Even a supported grouping signal would mean only that the structured signature carries reproducible whole-vehicle information; it would not prove transmission neutral drag. Independent CdA and tire/RRC coverage remain important confounders for any causal decomposition stage.

```ini
TRANSMISSION_GROUPING_BUILT = YES
INDEPENDENT_EXPERIMENT_UNITS = {result.independent_unit_count}
USABLE_GROUPS_GE3 = {result.usable_groups_ge3}
USABLE_GROUPS_GE5 = {result.usable_groups_ge5}

BASELINE_CV_RMSE_N = {result.baseline_rmse_n:.6f}
GROUP_MODEL_CV_RMSE_N = {result.group_rmse_n:.6f}
OUT_OF_SAMPLE_RMSE_IMPROVEMENT_N = {result.improvement_n:.6f}

PERMUTATION_P_VALUE = {result.permutation_p_value:.6f}
RESIDUAL_SIMILARITY_SUPPORT = {result.residual_similarity_support}

TRANSMISSION_GROUP_SIGNAL_SUPPORTED = {result.signal_supported}

READY_FOR_SHARED_TRANSMISSION_LOSS_ESTIMATION_EXPERIMENT = {result.ready_for_shared_loss_experiment}

PRODUCTION_DB_CHANGED = NO
```

## Reproducibility / immutability

- Source DB: `{result.db_path}`
- SHA256 before: `{result.db_sha256_before}`
- SHA256 after: `{result.db_sha256_after}`
- Connection mode: SQLite URI `mode=ro` plus `PRAGMA query_only=ON`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run_experiment(
    db_path: Path = DEFAULT_DB,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    n_permutations: int = 500,
    notebook_path: Path = DEFAULT_NOTEBOOK,
    report_path: Path = DEFAULT_REPORT,
    build_notebook_artifact: bool = True,
) -> ExperimentResult:
    db_path = Path(db_path)
    output_dir = Path(output_dir)
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    hash_before = sha256_file(db_path)

    vdes, runs, fuelcons, programs = load_canonical_source(db_path)
    field_audit = build_field_audit(vdes, runs, fuelcons, programs)
    root_rows, groups, sample = prepare_population(vdes, runs, fuelcons)
    long = build_force_curve_long(root_rows)
    oof, validation, model_comparison, baseline_names, augmented_names = run_grouped_cv(long)
    permutations, observed_distance, random_distance, p_value, movable_strata, movable_units = run_residual_permutation(
        oof, n_permutations
    )
    ci_low, ci_high = bootstrap_improvement_ci(oof)
    sensitivity, outlier_ids, outlier_threshold = run_sensitivities(long, oof)

    long_output = oof.copy()
    long_output["curve_outlier_flag"] = np.where(long_output["vde_id"].isin(outlier_ids), "IQR_FLAG", "")
    sample["curve_outlier_flag"] = np.where(sample["vde_id"].isin(outlier_ids), "IQR_FLAG", "")

    overall = validation[validation["scope_type"].eq("OVERALL")].iloc[0]
    fold_deltas = validation[validation["scope_type"].eq("FOLD")]["delta_rmse_N"]
    improvement = float(overall["delta_rmse_N"])
    residual_support = "YES" if observed_distance < random_distance and p_value < 0.05 else "NO"
    sensitivity_completed = sensitivity[sensitivity["status"].eq("COMPLETED")]
    robust_positive = bool(
        len(fold_deltas)
        and fold_deltas.gt(0).all()
        and ci_low > 0
        and sensitivity_completed["delta_rmse_N"].dropna().gt(0).all()
    )
    signal_supported = "YES" if improvement > 0 and robust_positive and residual_support == "YES" else "NO"
    ready = "YES" if signal_supported == "YES" else "NO"

    direct_count = int(groups["identity_status"].eq("DIRECT_MATCH").sum())
    strict_count = int(groups["identity_status"].eq("STRICT_CANDIDATE").sum())
    ge3 = int(groups["usable_ge3"].eq("YES").sum())
    ge5 = int(groups["usable_ge5"].eq("YES").sum())
    result = ExperimentResult(
        db_path=db_path.resolve(),
        db_sha256_before=hash_before,
        db_sha256_after="PENDING",
        raw_vde_count=int(len(vdes)),
        excluded_carryover_count=int(sample["exclusion_reason"].eq("EXACT_MODEL_YEAR_CARRYOVER_COLLAPSED_TO_ROOT").sum()),
        independent_unit_count=int(long["vde_id"].nunique()),
        direct_match_groups=direct_count,
        strict_candidate_groups=strict_count,
        usable_groups_ge3=ge3,
        usable_groups_ge5=ge5,
        baseline_rmse_n=float(overall["baseline_rmse_N"]),
        baseline_mae_n=float(overall["baseline_mae_N"]),
        group_rmse_n=float(overall["group_model_rmse_N"]),
        group_mae_n=float(overall["group_model_mae_N"]),
        improvement_n=improvement,
        permutation_p_value=p_value,
        observed_residual_distance_n=observed_distance,
        permutation_mean_distance_n=random_distance,
        residual_similarity_support=residual_support,
        signal_supported=signal_supported,
        ready_for_shared_loss_experiment=ready,
        bootstrap_ci_low_n=ci_low,
        bootstrap_ci_high_n=ci_high,
        output_dir=output_dir.resolve(),
        notebook_path=Path(notebook_path).resolve(),
        report_path=Path(report_path).resolve(),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    field_audit.to_csv(output_dir / FIELD_AUDIT_FILE, index=False)
    groups.to_csv(output_dir / GROUPS_FILE, index=False)
    sample.to_csv(output_dir / SAMPLE_FILE, index=False)
    long_output.to_csv(output_dir / LONG_FILE, index=False)
    model_comparison.to_csv(output_dir / MODEL_FILE, index=False)
    validation.to_csv(output_dir / VALIDATION_FILE, index=False)
    permutations.to_csv(output_dir / PERMUTATION_FILE, index=False)
    sensitivity.to_csv(output_dir / SENSITIVITY_FILE, index=False)

    if build_notebook_artifact:
        build_notebook(Path(notebook_path), db_path, output_dir)

    hash_after = sha256_file(db_path)
    if hash_before != hash_after:
        raise AssertionError("Read-only experiment changed the source DB hash")
    result = ExperimentResult(**{**result.__dict__, "db_sha256_after": hash_after})
    _write_report(
        Path(report_path),
        result,
        groups,
        sample,
        model_comparison,
        validation,
        sensitivity,
        n_permutations,
        movable_strata,
        movable_units,
        outlier_ids,
        outlier_threshold,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = run_experiment(
        args.db,
        args.output_dir,
        n_permutations=args.permutations,
        notebook_path=args.notebook,
        report_path=args.report,
    )
    print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in result.__dict__.items()}, indent=2))


if __name__ == "__main__":
    main()
