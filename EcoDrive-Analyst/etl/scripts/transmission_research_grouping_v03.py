"""Evidence-weighted BMW transmission grouping experiment (research only).

The module is intentionally read-only with respect to SQLite.  External
technical evidence creates hypotheses; coastdown outcomes only test them.
Research labels produced here are never canonical component identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etl.scripts.sprint_12_transmission_grouping_signal_experiment import (  # noqa: E402
    PRIMARY_SPEEDS,
    load_canonical_source,
    normalize_text,
    prepare_population,
    sha256_file,
)
from src.vde_core.roadload_analysis import roadload_force_N  # noqa: E402

DEFAULT_DB = ROOT / "data/db/staging/eco_drive_canonical_candidate.db"
DEFAULT_EVIDENCE = ROOT / "artifacts/technical_research/BMW_TRANSMISSION_RESEARCH_FINAL_V022.csv"
DEFAULT_ENRICHMENT = ROOT / "artifacts/technical_research/VEHICLE_ENRICHMENT_AUDIT_V022.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/components/transmission_research_grouping_v03"
DEFAULT_REPORT = DEFAULT_OUTPUT / "TRANSMISSION_RESEARCH_GROUPING_V03_REPORT.md"
RANDOM_SEED = 12003
OUTER_SPLITS = 5
INNER_SPLITS = 3
ALPHAS = (0.1, 1.0, 10.0, 100.0)
LAMBDAS = (0.1, 1.0, 10.0, 100.0)
PERMUTATIONS = 1000
BOOTSTRAPS = 1000

WEIGHT_MAPS: dict[str, dict[str, float]] = {
    "conservative": {"CONFIRMED": 1.0, "PROBABLE": 0.5, "PLAUSIBLE": 0.1, "UNRESOLVED": 0.0, "CONFLICTING": 0.0},
    "primary": {"CONFIRMED": 1.0, "PROBABLE": 0.7, "PLAUSIBLE": 0.3, "UNRESOLVED": 0.0, "CONFLICTING": 0.0},
    "permissive": {"CONFIRMED": 1.0, "PROBABLE": 0.85, "PLAUSIBLE": 0.5, "UNRESOLVED": 0.0, "CONFLICTING": 0.0},
}

CONTROLLED_NUMERIC_FEATURES = (
    "speed_scaled", "speed_scaled_sq", "test_mass_kg", "model_year", "gear_count",
    "final_drive_ratio", "nv_ratio", "mass_speed", "year_speed", "gear_speed",
    "fdr_speed", "nv_speed",
)
CONTROLLED_CATEGORICAL_FEATURES = (
    "make_norm", "model_norm", "category_norm", "drive_norm", "electrification",
    "transmission_type_norm",
)
PROHIBITED_GROUP_INPUTS = {
    "coast_A_N", "coast_B_N_per_kph", "coast_C_N_per_kph2", "observed_force_N",
    "cda_m2", "rrc_N_per_kN", "residual", "target",
}


@dataclass(frozen=True)
class PairEvidence:
    relation: str
    positive: tuple[str, ...]
    negative: tuple[str, ...]
    score: float


@dataclass(frozen=True)
class ExperimentSummary:
    independent_applications: int
    model0_rmse: float
    model1_rmse: float
    model2_rmse: float
    permutation_p: float
    signal_supported: bool
    soft_adds_value: bool
    db_hash_before: str
    db_hash_after: str


def _clean(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _source_ids(value: object) -> tuple[str, ...]:
    return tuple(sorted({item for item in _clean(value).split(";") if item}))


def _gear_ratios(value: object) -> tuple[float, ...]:
    """Return forward ratios only, in gear order, from v0.2.2 evidence."""
    text = _clean(value)
    if not text:
        return ()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if isinstance(parsed, dict):
        order = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X")
        result = []
        for key in order:
            if key in parsed:
                match = re.search(r"[-+]?\d+(?:\.\d+)?", str(parsed[key]))
                if match:
                    result.append(float(match.group()))
        return tuple(result)
    if re.fullmatch(r"\s*\d+(?:\.\d+)?\s*(?::\s*1)?\s*", text):
        return (float(re.search(r"\d+(?:\.\d+)?", text).group()),)
    result = []
    for label in ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"):
        match = re.search(rf"(?:^|[;,{{])\s*{label}\s*[: ]\s*[\"']?([-+]?\d+(?:\.\d+)?)", text, re.I)
        if match:
            result.append(float(match.group(1)))
    return tuple(result)


def _ratio_relation(left: Sequence[float], right: Sequence[float]) -> str:
    if not left or not right:
        return "UNKNOWN"
    if len(left) != len(right):
        return "INCOMPATIBLE"
    a, b = np.asarray(left), np.asarray(right)
    relative = np.max(np.abs(a - b) / np.maximum(np.abs(a), np.abs(b)))
    if relative <= 0.02:
        return "NEAR_IDENTICAL"
    if relative > 0.04:
        return "INCOMPATIBLE"
    return "DIFFERENT"


def classify_pair(left: Mapping[str, object], right: Mapping[str, object]) -> PairEvidence:
    """Classify a pair from identity/structure evidence only (never outcomes)."""
    forbidden = PROHIBITED_GROUP_INPUTS.intersection(left) | PROHIBITED_GROUP_INPUTS.intersection(right)
    if forbidden:
        raise ValueError(f"Outcome/derived fields are forbidden in evidence classification: {sorted(forbidden)}")
    positive: list[str] = []
    negative: list[str] = []
    hardware_l = normalize_text(left.get("hardware_designation")) if _clean(left.get("hardware_designation")) else ""
    hardware_r = normalize_text(right.get("hardware_designation")) if _clean(right.get("hardware_designation")) else ""
    family_l = normalize_text(left.get("transmission_family")) if _clean(left.get("transmission_family")) else ""
    family_r = normalize_text(right.get("transmission_family")) if _clean(right.get("transmission_family")) else ""
    ratios_l, ratios_r = _gear_ratios(left.get("gear_ratios")), _gear_ratios(right.get("gear_ratios"))
    ratio_relation = _ratio_relation(ratios_l, ratios_r)
    external_l = bool(_source_ids(left.get("external_source_ids")))
    external_r = bool(_source_ids(right.get("external_source_ids")))

    if hardware_l and hardware_r:
        (positive if hardware_l == hardware_r else negative).append(
            "exact_same_hardware_designation" if hardware_l == hardware_r else "different_explicit_hardware_designation"
        )
    if family_l and family_r:
        (positive if family_l == family_r else negative).append(
            "same_externally_supported_family" if family_l == family_r else "different_physical_transmission_family"
        )
    if ratio_relation == "NEAR_IDENTICAL":
        positive.append("near_identical_explicit_forward_gear_ratio_set")
    elif ratio_relation == "INCOMPATIBLE":
        negative.append("incompatible_explicit_forward_gear_ratio_sets")

    same_trans = left.get("normalized_transmission_type") == right.get("normalized_transmission_type")
    same_gears = str(left.get("gears")) == str(right.get("gears"))
    same_drive = normalize_text(left.get("drive_architecture")) == normalize_text(right.get("drive_architecture"))
    same_model = normalize_text(left.get("model")) == normalize_text(right.get("model"))
    years_close = abs(int(left.get("model_year") or 0) - int(right.get("model_year") or 0)) <= 1
    marketing_l = normalize_text(left.get("marketing_description")) if _clean(left.get("marketing_description")) else ""
    marketing_r = normalize_text(right.get("marketing_description")) if _clean(right.get("marketing_description")) else ""
    if same_trans:
        positive.append("same_normalized_transmission_type")
    if same_gears:
        positive.append("same_gear_count")
    if same_drive:
        positive.append("compatible_drive_architecture")
    if marketing_l and marketing_l == marketing_r:
        positive.append("same_oem_marketing_description")
    if (external_l or external_r) and same_model and years_close and same_trans and same_gears and same_drive:
        positive.append("partial_external_same_application_generation_anchor")

    if "different_explicit_hardware_designation" in negative or "different_physical_transmission_family" in negative or "incompatible_explicit_forward_gear_ratio_sets" in negative:
        return PairEvidence("CONFLICTING", tuple(positive), tuple(negative), 0.0)
    if hardware_l and hardware_r and hardware_l == hardware_r:
        return PairEvidence("CONFIRMED", tuple(positive), (), 10.0)
    if external_l and external_r and (
        (family_l and family_l == family_r) or ratio_relation == "NEAR_IDENTICAL"
    ) and same_trans and same_gears:
        return PairEvidence("PROBABLE", tuple(positive), (), 7.0)
    if (external_l and external_r and marketing_l and marketing_l == marketing_r and same_trans and same_gears) or "partial_external_same_application_generation_anchor" in positive:
        return PairEvidence("PLAUSIBLE", tuple(positive), (), 3.0)
    return PairEvidence("UNRESOLVED", tuple(positive), tuple(negative), 0.0)


def evidence_weight(relation: str, mapping: str = "primary") -> float:
    return float(WEIGHT_MAPS[mapping][relation])


def _evidence_records(root_rows: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    by_id = evidence.drop_duplicates("vde_id", keep="last").set_index("vde_id") if not evidence.empty else pd.DataFrame()
    records = []
    for row in root_rows.itertuples(index=False):
        ext = by_id.loc[int(row.vde_id)] if not by_id.empty and int(row.vde_id) in by_id.index else None
        application_match = _clean(ext.get("strongest_application_match")) if ext is not None else "UNKNOWN"
        has_tire = ext is not None and any(
            _clean(ext.get(field)) for field in ("tire_size_front", "tire_size_rear", "tire_size_general")
        )
        research_use = {
            "gear_ratios": "RESEARCH_USABLE" if ext is not None and _clean(ext.get("gear_ratios")) and application_match == "PARTIAL" else "NOT_USABLE",
            "final_drive_confirmation": "SENSITIVITY_ONLY" if ext is not None and _clean(ext.get("final_drive_confirmation")) and application_match == "PARTIAL" else "NOT_USABLE",
            "cd_frontal_area": "SENSITIVITY_ONLY" if ext is not None and (_clean(ext.get("drag_coefficient_cd")) or _clean(ext.get("frontal_area_m2"))) and application_match == "PARTIAL" else "NOT_USABLE",
            "tire_specification": "SENSITIVITY_ONLY" if has_tire and application_match == "PARTIAL" else "NOT_USABLE",
        }
        records.append({
            "vde_id": int(row.vde_id),
            "vehicle_configuration_id": str(row.vehicle_configuration_id),
            "independent_application_id": f"VDE-{int(row.vde_id)}",
            "model": row.model,
            "model_year": int(row.year),
            "canonical_identity_confidence": _clean(ext.get("identity_confidence")) if ext is not None else "UNRESOLVED",
            "hardware_designation": _clean(ext.get("transmission_hardware_designation")) if ext is not None else "",
            "transmission_family": _clean(ext.get("transmission_family")) if ext is not None else "",
            "marketing_description": _clean(ext.get("transmission_marketing_description")) if ext is not None else "",
            "normalized_transmission_type": _clean(ext.get("normalized_transmission_type")) if ext is not None else normalize_text(row.transmission_type),
            "gears": int(row.gear_count),
            "gear_ratios": _clean(ext.get("gear_ratios")) if ext is not None else "",
            "final_drive_ratio": float(row.final_drive_ratio),
            "nv_ratio": float(row.nv_ratio),
            "drive_architecture": row.drive_system,
            "external_source_ids": _clean(ext.get("source_ids")) if ext is not None else "",
            "external_application_match": application_match,
            "forensic_v022_case": "YES" if ext is not None else "NO",
            "external_drag_coefficient_cd": _clean(ext.get("drag_coefficient_cd")) if ext is not None else "",
            "external_frontal_area_m2": _clean(ext.get("frontal_area_m2")) if ext is not None else "",
            "external_tire_size_front": _clean(ext.get("tire_size_front")) if ext is not None else "",
            "external_tire_size_rear": _clean(ext.get("tire_size_rear")) if ext is not None else "",
            "external_tire_size_general": _clean(ext.get("tire_size_general")) if ext is not None else "",
            "research_use_classification": json.dumps(research_use, sort_keys=True),
            "application_lineage": row.application_lineage,
            "permutation_stratum": row.permutation_stratum,
        })
    return pd.DataFrame(records)


def build_pairwise_evidence(apps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    records = apps.to_dict("records")
    for left, right in combinations(records, 2):
        result = classify_pair(left, right)
        rows.append({
            "application_i": left["independent_application_id"],
            "application_j": right["independent_application_id"],
            "relation": result.relation,
            "evidence_score": result.score,
            "positive_evidence": ";".join(result.positive),
            "negative_evidence": ";".join(result.negative),
            "primary_weight": evidence_weight(result.relation, "primary"),
            "conservative_weight": evidence_weight(result.relation, "conservative"),
            "permissive_weight": evidence_weight(result.relation, "permissive"),
        })
    return pd.DataFrame(rows)


def _relation_lookup(pairwise: pd.DataFrame) -> dict[frozenset[str], str]:
    return {frozenset((r.application_i, r.application_j)): r.relation for r in pairwise.itertuples(index=False)}


def assign_research_groups(apps: pd.DataFrame, pairwise: pd.DataFrame) -> pd.DataFrame:
    """Deterministic complete-link grouping avoids weak transitive bridges."""
    lookup = _relation_lookup(pairwise)
    rank = {"CONFIRMED": 3, "PROBABLE": 2, "PLAUSIBLE": 1, "UNRESOLVED": 0, "CONFLICTING": -1}
    positive_degree = {node: 0 for node in apps["independent_application_id"]}
    for row in pairwise[pairwise["relation"].isin(("CONFIRMED", "PROBABLE", "PLAUSIBLE"))].itertuples(index=False):
        positive_degree[row.application_i] += 1
        positive_degree[row.application_j] += 1
    nodes = sorted(positive_degree, key=lambda n: (-positive_degree[n], n))
    groups: list[list[str]] = []
    for node in nodes:
        placed = False
        for group in groups:
            if all(rank[lookup.get(frozenset((node, other)), "UNRESOLVED")] >= 1 for other in group):
                group.append(node)
                placed = True
                break
        if not placed:
            groups.append([node])
    group_meta = {}
    for number, group in enumerate(groups, 1):
        relations = [lookup[frozenset(pair)] for pair in combinations(group, 2)]
        if not relations:
            confidence = "UNRESOLVED"
        elif all(value == "CONFIRMED" for value in relations):
            confidence = "CONFIRMED"
        elif all(value in {"CONFIRMED", "PROBABLE"} for value in relations):
            confidence = "PROBABLE"
        else:
            confidence = "PLAUSIBLE"
        gid = f"TRG-{number:04d}" if len(group) > 1 else f"UNRESOLVED-{group[0]}"
        for node in group:
            group_meta[node] = (gid, confidence, len(group))
    result = apps.copy()
    result[["research_group_id", "research_group_confidence", "research_group_size"]] = result[
        "independent_application_id"
    ].map(group_meta).apply(pd.Series)
    return result


def _long_population(root_rows: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    meta = groups.set_index("vde_id")
    records = []
    for row in root_rows.itertuples(index=False):
        forces = roadload_force_N(float(row.coast_A_N), float(row.coast_B_N_per_kph), float(row.coast_C_N_per_kph2), list(PRIMARY_SPEEDS))
        group = meta.loc[int(row.vde_id)]
        for speed, force in zip(PRIMARY_SPEEDS, forces):
            scaled = float(speed) / 100.0
            records.append({
                "vde_id": int(row.vde_id), "independent_application_id": f"VDE-{int(row.vde_id)}",
                "application_lineage": row.application_lineage, "research_group_id": group.research_group_id,
                "research_group_confidence": group.research_group_confidence,
                "make_norm": row.make_norm, "model_norm": row.model_norm, "category_norm": row.category_norm,
                "drive_norm": row.drive_norm, "electrification": row.electrification,
                "transmission_type_norm": row.transmission_type_norm, "gear_count": float(row.gear_count),
                "final_drive_ratio": float(row.final_drive_ratio), "nv_ratio": float(row.nv_ratio),
                "test_mass_kg": float(row.test_mass_kg), "model_year": float(row.year),
                "speed_kph": float(speed), "speed_scaled": scaled, "speed_scaled_sq": scaled**2,
                "mass_speed": float(row.test_mass_kg) * scaled, "year_speed": float(row.year) * scaled,
                "gear_speed": float(row.gear_count) * scaled, "fdr_speed": float(row.final_drive_ratio) * scaled,
                "nv_speed": float(row.nv_ratio) * scaled, "observed_force_N": float(force),
            })
    return pd.DataFrame(records)


def build_fold_assignment(units: pd.DataFrame, n_splits: int = OUTER_SPLITS) -> dict[int, int]:
    optional = [column for column in ("independent_application_id", "research_group_id", "research_group_confidence") if column in units]
    unique = units[["vde_id", "application_lineage", *optional]].drop_duplicates().sort_values(["application_lineage", "vde_id"])
    splitter = GroupKFold(n_splits=n_splits)
    assignment = {}
    for fold, (_, test) in enumerate(splitter.split(unique, groups=unique["application_lineage"])):
        for value in unique.iloc[test]["vde_id"]:
            assignment[int(value)] = fold
    # Evidence-aware balancing is outcome-blind.  It only makes independently
    # testable research groups observable across folds; a lineage is never split.
    if {"research_group_id", "research_group_confidence"}.issubset(unique.columns):
        shared = unique[unique.research_group_confidence.isin(("CONFIRMED", "PROBABLE", "PLAUSIBLE"))]
        fold_sizes = {fold: sum(value == fold for value in assignment.values()) for fold in range(n_splits)}
        for _, members in shared.groupby("research_group_id", sort=True):
            lineages = list(dict.fromkeys(members.application_lineage))
            used: set[int] = set()
            for lineage in lineages:
                lineage_ids = unique[unique.application_lineage.eq(lineage)].vde_id.astype(int).tolist()
                current = assignment[lineage_ids[0]]
                if current in used and len(used) < n_splits:
                    choices = [fold for fold in range(n_splits) if fold not in used]
                    target = min(choices, key=lambda fold: (fold_sizes[fold], fold))
                    for vde_id in lineage_ids:
                        assignment[vde_id] = target
                    fold_sizes[current] -= len(lineage_ids)
                    fold_sizes[target] += len(lineage_ids)
                    current = target
                used.add(current)
    for fold in range(n_splits):
        test_lineages = set(unique[unique.vde_id.map(assignment).eq(fold)].application_lineage)
        train_lineages = set(unique[~unique.vde_id.map(assignment).eq(fold)].application_lineage)
        if test_lineages & train_lineages:
            raise AssertionError("Application lineage leakage between train and test")
    return assignment


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))])
    return ColumnTransformer([("numeric", numeric, list(CONTROLLED_NUMERIC_FEATURES)), ("categorical", categorical, list(CONTROLLED_CATEGORICAL_FEATURES))])


def _matrices(train: pd.DataFrame, test: pd.DataFrame, hard_labels: Mapping[str, str] | None = None):
    prep = _preprocessor()
    x_train = prep.fit_transform(train)
    x_test = prep.transform(test)
    interaction_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    cat_train = interaction_encoder.fit_transform(train[list(CONTROLLED_CATEGORICAL_FEATURES)].fillna("UNKNOWN").astype(str))
    cat_test = interaction_encoder.transform(test[list(CONTROLLED_CATEGORICAL_FEATURES)].fillna("UNKNOWN").astype(str))
    x_train = sparse.hstack((x_train, cat_train.multiply(train.speed_scaled.to_numpy()[:, None])), format="csr")
    x_test = sparse.hstack((x_test, cat_test.multiply(test.speed_scaled.to_numpy()[:, None])), format="csr")
    if hard_labels is None:
        return x_train, x_test
    train_apps = train.independent_application_id.astype(str)
    labels = sorted({str(value) for value in hard_labels.values()})
    covers_all_with_one_label = len(labels) == 1 and train_apps.map(hard_labels).notna().all()
    if not labels or covers_all_with_one_label:
        return x_train, x_test
    column = {label: number for number, label in enumerate(labels)}
    def group_matrix(frame: pd.DataFrame) -> sparse.csr_matrix:
        row_indices, column_indices = [], []
        for row_number, app in enumerate(frame.independent_application_id.astype(str)):
            label = hard_labels.get(app)
            if label is not None:
                row_indices.append(row_number)
                column_indices.append(column[str(label)])
        return sparse.csr_matrix(
            (np.ones(len(row_indices)), (row_indices, column_indices)),
            shape=(len(frame), len(labels)),
        )
    g_train, g_test = group_matrix(train), group_matrix(test)
    x_train = sparse.hstack((x_train, g_train, g_train.multiply(train.speed_scaled.to_numpy()[:, None])), format="csr")
    x_test = sparse.hstack((x_test, g_test, g_test.multiply(test.speed_scaled.to_numpy()[:, None])), format="csr")
    return x_train, x_test


def _rmse(actual, predicted) -> float:
    return float(math.sqrt(mean_squared_error(actual, predicted)))


def _tune_alpha(train: pd.DataFrame, hard_labels: Mapping[str, str] | None = None) -> tuple[float, list[dict]]:
    groups = train.application_lineage
    audit, scores = [], {alpha: [] for alpha in ALPHAS}
    for inner_fold, (fit_idx, val_idx) in enumerate(GroupKFold(n_splits=INNER_SPLITS).split(train, groups=groups)):
        fit, val = train.iloc[fit_idx], train.iloc[val_idx]
        if set(fit.application_lineage) & set(val.application_lineage):
            raise AssertionError("Inner tuning leaked final validation lineage")
        x_fit, x_val = _matrices(fit, val, hard_labels)
        for alpha in ALPHAS:
            model = Ridge(alpha=alpha, solver="lsqr").fit(x_fit, fit.observed_force_N)
            score = _rmse(val.observed_force_N, model.predict(x_val))
            scores[alpha].append(score)
            audit.append({"inner_fold": inner_fold, "alpha": alpha, "rmse_N": score, "fit_lineages": fit.application_lineage.nunique(), "validation_lineages": val.application_lineage.nunique(), "lineage_overlap": 0})
    return min(ALPHAS, key=lambda a: (np.mean(scores[a]), a)), audit


def _edge_map(pairwise: pd.DataFrame, weight_column: str, allowed: Iterable[str] = ("CONFIRMED", "PROBABLE", "PLAUSIBLE")) -> dict[str, list[tuple[str, float]]]:
    allowed_set = set(allowed)
    result: dict[str, list[tuple[str, float]]] = {}
    for row in pairwise[pairwise.relation.isin(allowed_set)].itertuples(index=False):
        weight = float(getattr(row, weight_column))
        if weight <= 0:
            continue
        result.setdefault(row.application_i, []).append((row.application_j, weight))
        result.setdefault(row.application_j, []).append((row.application_i, weight))
    return result


def _application_effects(frame: pd.DataFrame, residuals: np.ndarray) -> dict[str, np.ndarray]:
    data = frame[["independent_application_id", "speed_scaled"]].copy()
    data["residual"] = residuals
    effects = {}
    for app, part in data.groupby("independent_application_id"):
        design = np.column_stack((np.ones(len(part)), part.speed_scaled.to_numpy()))
        effects[app] = np.linalg.lstsq(design, part.residual.to_numpy(), rcond=None)[0]
    return effects


def _soft_correction(frame: pd.DataFrame, effects: Mapping[str, np.ndarray], edges: Mapping[str, list[tuple[str, float]]], lam: float) -> np.ndarray:
    correction = np.zeros(len(frame))
    for app, indices in frame.groupby("independent_application_id").groups.items():
        neighbours = [(other, weight) for other, weight in edges.get(app, ()) if other in effects]
        if neighbours:
            total = sum(weight for _, weight in neighbours)
            beta = sum((weight * effects[other] for other, weight in neighbours), np.zeros(2)) / (lam + total)
            correction[np.asarray(list(indices), dtype=int)] = beta[0] + beta[1] * frame.loc[indices, "speed_scaled"].to_numpy()
    return correction


def _fit_baseline(train: pd.DataFrame, test: pd.DataFrame, alpha: float, hard_labels=None):
    x_train, x_test = _matrices(train, test, hard_labels)
    model = Ridge(alpha=alpha, solver="lsqr").fit(x_train, train.observed_force_N)
    return model.predict(x_train), model.predict(x_test)


def _tune_lambda(train: pd.DataFrame, alpha: float, edges: Mapping[str, list[tuple[str, float]]]) -> tuple[float, list[dict]]:
    scores = {lam: [] for lam in LAMBDAS}
    audit = []
    for inner_fold, (fit_idx, val_idx) in enumerate(GroupKFold(n_splits=INNER_SPLITS).split(train, groups=train.application_lineage)):
        fit, val = train.iloc[fit_idx].reset_index(drop=True), train.iloc[val_idx].reset_index(drop=True)
        pred_fit, pred_val = _fit_baseline(fit, val, alpha)
        effects = _application_effects(fit, fit.observed_force_N.to_numpy() - pred_fit)
        for lam in LAMBDAS:
            prediction = pred_val + _soft_correction(val, effects, edges, lam)
            score = _rmse(val.observed_force_N, prediction)
            scores[lam].append(score)
            audit.append({"inner_fold": inner_fold, "lambda": lam, "rmse_N": score, "lineage_overlap": 0})
    return min(LAMBDAS, key=lambda value: (np.mean(scores[value]), value)), audit


def run_models(long: pd.DataFrame, groups: pd.DataFrame, pairwise: pd.DataFrame, weight_map: str = "primary") -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    data = long.reset_index(drop=True).copy()
    folds = build_fold_assignment(data[["vde_id", "application_lineage", "independent_application_id", "research_group_id", "research_group_confidence"]])
    data["fold"] = data.vde_id.map(folds)
    hard = groups[groups.research_group_confidence.isin(("CONFIRMED", "PROBABLE")) & groups.research_group_size.ge(2)].set_index("independent_application_id").research_group_id.to_dict()
    edges = _edge_map(pairwise, f"{weight_map}_weight")
    predictions, audit = [], []
    fold_rows = []
    for fold in range(OUTER_SPLITS):
        train = data[data.fold.ne(fold)].drop(columns="fold").reset_index(drop=True)
        test = data[data.fold.eq(fold)].drop(columns="fold").reset_index(drop=True)
        if set(train.application_lineage) & set(test.application_lineage):
            raise AssertionError("Outer CV application leakage")
        alpha0, audit0 = _tune_alpha(train)
        alpha1, audit1 = _tune_alpha(train, hard)
        lam, audit2 = _tune_lambda(train, alpha0, edges)
        pred0_train, pred0 = _fit_baseline(train, test, alpha0)
        _, pred1 = _fit_baseline(train, test, alpha1, hard)
        effects = _application_effects(train, train.observed_force_N.to_numpy() - pred0_train)
        pred2 = pred0 + _soft_correction(test, effects, edges, lam)
        for kind, rows in (("alpha_model0", audit0), ("alpha_model1", audit1), ("lambda_model2", audit2)):
            audit.extend({"outer_fold": fold, "tuning": kind, **row} for row in rows)
        fold_rows.append({
            "weight_mapping": weight_map, "fold": fold, "train_applications": train.vde_id.nunique(),
            "test_applications": test.vde_id.nunique(), "train_lineages": train.application_lineage.nunique(),
            "test_lineages": test.application_lineage.nunique(), "lineage_overlap": 0,
            "alpha_model0": alpha0, "alpha_model1": alpha1, "lambda_model2": lam,
            "model0_rmse_N": _rmse(test.observed_force_N, pred0), "model1_rmse_N": _rmse(test.observed_force_N, pred1),
            "model2_rmse_N": _rmse(test.observed_force_N, pred2),
        })
        out = test[["vde_id", "independent_application_id", "application_lineage", "speed_kph", "observed_force_N"]].copy()
        out["fold"] = fold
        out["model0_prediction_N"] = pred0
        out["model1_prediction_N"] = pred1
        out["model2_prediction_N"] = pred2
        out["model0_residual_N"] = out.observed_force_N - pred0
        out["model1_residual_N"] = out.observed_force_N - pred1
        out["model2_residual_N"] = out.observed_force_N - pred2
        predictions.append(out)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(fold_rows), audit


def run_soft_sensitivity(
    long: pd.DataFrame,
    pairwise: pd.DataFrame,
    base_predictions: pd.DataFrame,
    base_folds: pd.DataFrame,
    weight_map: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Reuse primary baseline/hard predictions; only the soft weights differ."""
    edges = _edge_map(pairwise, f"{weight_map}_weight")
    predictions, fold_rows, audit = [], [], []
    for base in base_folds.itertuples(index=False):
        fold = int(base.fold)
        test_base = base_predictions[base_predictions.fold.eq(fold)].copy()
        test_apps = set(test_base.independent_application_id)
        train = long[~long.independent_application_id.isin(test_apps)].reset_index(drop=True)
        test = long[long.independent_application_id.isin(test_apps)].reset_index(drop=True)
        alpha0 = float(base.alpha_model0)
        lam, lambda_audit = _tune_lambda(train, alpha0, edges)
        pred0_train, _ = _fit_baseline(train, test, alpha0)
        effects = _application_effects(train, train.observed_force_N.to_numpy() - pred0_train)
        test_base = test_base.sort_values(["independent_application_id", "speed_kph"]).reset_index(drop=True)
        test = test.sort_values(["independent_application_id", "speed_kph"]).reset_index(drop=True)
        test_base["model2_prediction_N"] = test_base.model0_prediction_N.to_numpy() + _soft_correction(test, effects, edges, lam)
        test_base["model2_residual_N"] = test_base.observed_force_N - test_base.model2_prediction_N
        predictions.append(test_base)
        fold_rows.append({
            **base._asdict(), "weight_mapping": weight_map, "lambda_model2": lam,
            "model2_rmse_N": _rmse(test_base.observed_force_N, test_base.model2_prediction_N),
        })
        audit.extend({"outer_fold": fold, "tuning": "lambda_model2", **row} for row in lambda_audit)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(fold_rows), audit


def _metric_rows(predictions: pd.DataFrame, weight_map: str) -> list[dict]:
    rows = []
    for number, column in ((0, "model0_prediction_N"), (1, "model1_prediction_N"), (2, "model2_prediction_N")):
        rows.append({"weight_mapping": weight_map, "model": f"MODEL_{number}", "oof_rmse_N": _rmse(predictions.observed_force_N, predictions[column]), "oof_mae_N": float(mean_absolute_error(predictions.observed_force_N, predictions[column]))})
    return rows


def cluster_bootstrap(predictions: pd.DataFrame, iterations: int = BOOTSTRAPS) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    apps = predictions.independent_application_id.unique()
    rows = []
    for iteration in range(iterations):
        sampled = rng.choice(apps, size=len(apps), replace=True)
        chunks = [predictions[predictions.independent_application_id.eq(app)] for app in sampled]
        frame = pd.concat(chunks, ignore_index=True)
        rmse0 = _rmse(frame.observed_force_N, frame.model0_prediction_N)
        rmse1 = _rmse(frame.observed_force_N, frame.model1_prediction_N)
        rmse2 = _rmse(frame.observed_force_N, frame.model2_prediction_N)
        rows.append({"bootstrap": iteration + 1, "model1_delta_vs_model0_N": rmse0 - rmse1, "model2_delta_vs_model0_N": rmse0 - rmse2, "model2_delta_vs_model1_N": rmse1 - rmse2})
    return pd.DataFrame(rows)


def _fold_soft_context(long: pd.DataFrame, predictions: pd.DataFrame, fold_rows: pd.DataFrame):
    contexts = []
    for row in fold_rows.itertuples(index=False):
        fold = int(row.fold)
        test_predictions = predictions[predictions.fold.eq(fold)].reset_index(drop=True)
        test_apps = set(test_predictions.independent_application_id)
        test = long[long.independent_application_id.isin(test_apps)].reset_index(drop=True).merge(
            test_predictions[
                ["independent_application_id", "speed_kph", "model0_prediction_N"]
            ],
            on=["independent_application_id", "speed_kph"],
            how="left",
            validate="one_to_one",
        )
        train_apps = set(long.independent_application_id) - test_apps
        train = long[long.independent_application_id.isin(train_apps)].reset_index(drop=True)
        pred_train, _ = _fit_baseline(train, test, float(row.alpha_model0))
        effects = _application_effects(train, train.observed_force_N.to_numpy() - pred_train)
        contexts.append((fold, test, effects, float(row.lambda_model2)))
    return contexts


def permutation_test(long: pd.DataFrame, predictions: pd.DataFrame, fold_rows: pd.DataFrame, pairwise: pd.DataFrame, apps: pd.DataFrame, iterations: int = PERMUTATIONS) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 1)
    base_edges = _edge_map(pairwise, "primary_weight")
    contexts = _fold_soft_context(long, predictions, fold_rows)
    strata = apps.groupby("permutation_stratum").independent_application_id.apply(list).to_dict()
    observed_delta = _rmse(predictions.observed_force_N, predictions.model0_prediction_N) - _rmse(predictions.observed_force_N, predictions.model2_prediction_N)
    rows = [{"permutation": 0, "kind": "OBSERVED", "delta_rmse_N": observed_delta}]
    for iteration in range(1, iterations + 1):
        relabel = {}
        for nodes in strata.values():
            shuffled = rng.permutation(nodes)
            relabel.update(zip(nodes, shuffled))
        perm_edges: dict[str, list[tuple[str, float]]] = {}
        for source, neighbours in base_edges.items():
            for target, weight in neighbours:
                perm_edges.setdefault(relabel[source], []).append((relabel[target], weight))
        actual, baseline, soft = [], [], []
        for _, test, effects, lam in contexts:
            actual.extend(test.observed_force_N)
            baseline.extend(test.model0_prediction_N)
            soft.extend(test.model0_prediction_N.to_numpy() + _soft_correction(test, effects, perm_edges, lam))
        rows.append({"permutation": iteration, "kind": "STRATIFIED_NODE_RELABEL", "delta_rmse_N": _rmse(actual, baseline) - _rmse(actual, soft)})
    frame = pd.DataFrame(rows)
    pvalue = (1 + int(frame.loc[frame.permutation.gt(0), "delta_rmse_N"].ge(observed_delta).sum())) / (iterations + 1)
    frame["empirical_p_value"] = pvalue
    frame["scheme"] = "Bijective node relabeling within transmission-type|gear-count|drive strata; graph topology and group sizes preserved."
    return frame


def residual_similarity(predictions: pd.DataFrame, pairwise: pd.DataFrame, apps: pd.DataFrame, iterations: int = PERMUTATIONS) -> pd.DataFrame:
    curves = predictions.pivot(index="independent_application_id", columns="speed_kph", values="model0_residual_N")
    positive = pairwise[pairwise.relation.isin(("CONFIRMED", "PROBABLE", "PLAUSIBLE"))][["application_i", "application_j"]]
    def distance(pairs):
        values = [float(np.sqrt(np.mean((curves.loc[a].to_numpy() - curves.loc[b].to_numpy()) ** 2))) for a, b in pairs if a in curves.index and b in curves.index]
        return float(np.mean(values)) if values else float("nan")
    observed_pairs = list(positive.itertuples(index=False, name=None))
    observed = distance(observed_pairs)
    rng = np.random.default_rng(RANDOM_SEED + 2)
    by_stratum = apps.groupby("permutation_stratum").independent_application_id.apply(list).to_dict()
    app_stratum = apps.set_index("independent_application_id").permutation_stratum.to_dict()
    rows = [{"randomization": 0, "kind": "OBSERVED_EVIDENCE_PAIRS", "mean_residual_curve_rmse_N": observed}]
    for iteration in range(1, iterations + 1):
        random_pairs = []
        for left, _ in observed_pairs:
            pool = [item for item in by_stratum[app_stratum[left]] if item != left]
            if pool:
                random_pairs.append((left, rng.choice(pool)))
        rows.append({"randomization": iteration, "kind": "STRATIFIED_RANDOM_PAIRS", "mean_residual_curve_rmse_N": distance(random_pairs)})
    result = pd.DataFrame(rows)
    result["empirical_p_value"] = (1 + int(result.loc[result.randomization.gt(0), "mean_residual_curve_rmse_N"].le(observed).sum())) / (iterations + 1) if np.isfinite(observed) else 1.0
    return result


def _control_labels(groups: pd.DataFrame, name: str) -> dict[str, str]:
    frame = groups.copy()
    if name == "make_only":
        labels = pd.Series("BMW", index=frame.index)
    elif name == "gear_count_only":
        labels = frame.gears.map(lambda value: f"GEARS-{value}")
    elif name == "transmission_type_only":
        labels = frame.normalized_transmission_type.map(lambda value: f"TRANS-{value}")
    elif name == "fdr_bin_only":
        labels = pd.cut(frame.final_drive_ratio, bins=[-np.inf, 2.5, 3.0, 3.5, 4.0, np.inf], labels=False).map(lambda value: f"FDR-{value}")
    elif name == "random_same_sizes":
        rng = np.random.default_rng(RANDOM_SEED + 3)
        sizes = sorted(frame[frame.research_group_size.gt(1)].groupby("research_group_id").size(), reverse=True)
        nodes = list(rng.permutation(frame.independent_application_id))
        mapping, cursor = {}, 0
        for number, size in enumerate(sizes):
            for node in nodes[cursor:cursor + int(size)]:
                mapping[node] = f"RANDOM-{number}"
            cursor += int(size)
        return mapping
    else:
        raise ValueError(name)
    return dict(zip(frame.independent_application_id, labels))


def _hard_oof(long: pd.DataFrame, labels: Mapping[str, str], train_only_alphas: Mapping[int, float] | None = None) -> tuple[float, float]:
    folds = build_fold_assignment(long[["vde_id", "application_lineage", "independent_application_id", "research_group_id", "research_group_confidence"]])
    actual, predicted = [], []
    for fold in range(OUTER_SPLITS):
        train = long[long.vde_id.map(folds).ne(fold)].reset_index(drop=True)
        test = long[long.vde_id.map(folds).eq(fold)].reset_index(drop=True)
        alpha = float(train_only_alphas[fold]) if train_only_alphas is not None else _tune_alpha(train, labels)[0]
        _, pred = _fit_baseline(train, test, alpha, labels)
        actual.extend(test.observed_force_N)
        predicted.extend(pred)
    return _rmse(actual, predicted), float(mean_absolute_error(actual, predicted))


def negative_controls(long: pd.DataFrame, groups: pd.DataFrame, baseline_rmse: float, fold_rows: pd.DataFrame) -> pd.DataFrame:
    alphas = fold_rows.set_index("fold").alpha_model1.to_dict()
    rows = []
    for name in ("random_same_sizes", "make_only", "gear_count_only", "transmission_type_only", "fdr_bin_only"):
        rmse, mae = _hard_oof(long, _control_labels(groups, name), alphas)
        rows.append({"negative_control": name, "oof_rmse_N": rmse, "oof_mae_N": mae, "delta_vs_controlled_baseline_N": baseline_rmse - rmse})
    return pd.DataFrame(rows)


def _labels_for_relations(groups: pd.DataFrame, pairwise: pd.DataFrame, allowed: set[str]) -> dict[str, str]:
    edges = pairwise[pairwise.relation.isin(allowed)]
    parent = {node: node for node in groups.independent_application_id}
    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    for row in edges.itertuples(index=False):
        a, b = find(row.application_i), find(row.application_j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    components = {}
    for node in parent:
        components.setdefault(find(node), []).append(node)
    return {node: root for root, nodes in components.items() if len(nodes) > 1 for node in nodes}


def ablations(long: pd.DataFrame, groups: pd.DataFrame, pairwise: pd.DataFrame, model_by_weight: Mapping[str, pd.DataFrame], fold_rows: pd.DataFrame) -> pd.DataFrame:
    baseline = _metric_rows(model_by_weight["primary"], "primary")[0]
    rows = [{"ablation": "controlled_baseline_only", "oof_rmse_N": baseline["oof_rmse_N"], "oof_mae_N": baseline["oof_mae_N"]}]
    structured = dict(zip(groups.independent_application_id, groups.normalized_transmission_type.astype(str) + "|" + groups.gears.astype(str) + "|" + groups.drive_architecture.map(normalize_text) + "|" + groups.final_drive_ratio.round(2).astype(str) + "|" + groups.nv_ratio.round(1).astype(str)))
    candidates = {
        "epa_structured_signature_only": structured,
        "external_confirmed_only": _labels_for_relations(groups, pairwise, {"CONFIRMED"}),
        "external_confirmed_plus_probable": _labels_for_relations(groups, pairwise, {"CONFIRMED", "PROBABLE"}),
        "external_confirmed_probable_plausible": _labels_for_relations(groups, pairwise, {"CONFIRMED", "PROBABLE", "PLAUSIBLE"}),
    }
    alphas = fold_rows.set_index("fold").alpha_model1.to_dict()
    for name, labels in candidates.items():
        rmse, mae = _hard_oof(long, labels, alphas)
        rows.append({"ablation": name, "oof_rmse_N": rmse, "oof_mae_N": mae})
    for mapping, predictions in model_by_weight.items():
        row = _metric_rows(predictions, mapping)[2]
        rows.append({"ablation": f"soft_evidence_{mapping}_weights", "oof_rmse_N": row["oof_rmse_N"], "oof_mae_N": row["oof_mae_N"]})
    result = pd.DataFrame(rows)
    result["delta_vs_controlled_baseline_N"] = float(baseline["oof_rmse_N"]) - result.oof_rmse_N
    return result


def _ci(series: pd.Series) -> tuple[float, float]:
    return float(series.quantile(0.025)), float(series.quantile(0.975))


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in frame.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(path: Path, summary: ExperimentSummary, metrics: pd.DataFrame, pairwise: pd.DataFrame, groups: pd.DataFrame, bootstrap: pd.DataFrame, permutation: pd.DataFrame, residual: pd.DataFrame, controls: pd.DataFrame, ablation: pd.DataFrame, db_path: Path) -> None:
    rel = pairwise.relation.value_counts().to_dict()
    gc = groups.drop_duplicates("research_group_id").research_group_confidence.value_counts().to_dict()
    m = metrics.set_index("model")
    ci1, ci2, ci21 = _ci(bootstrap.model1_delta_vs_model0_N), _ci(bootstrap.model2_delta_vs_model0_N), _ci(bootstrap.model2_delta_vs_model1_N)
    residual_p = float(residual.empirical_p_value.iloc[0])
    delta1 = float(m.loc["MODEL_0", "oof_rmse_N"] - m.loc["MODEL_1", "oof_rmse_N"])
    delta2 = float(m.loc["MODEL_0", "oof_rmse_N"] - m.loc["MODEL_2", "oof_rmse_N"])
    delta21 = float(m.loc["MODEL_1", "oof_rmse_N"] - m.loc["MODEL_2", "oof_rmse_N"])
    sensitivity_stable = all(
        ablation.loc[ablation.ablation.eq(f"soft_evidence_{name}_weights"), "delta_vs_controlled_baseline_N"].iloc[0] >= 0
        for name in WEIGHT_MAPS
    )
    negative_controls_passed = delta2 > max(0.0, float(controls.delta_vs_controlled_baseline_N.max()))
    lines = [
        "# Transmission Research Grouping v0.3 — Result", "",
        "Scientific experiment only. Research groups are hypotheses and were not written to canonical tables.", "",
        "## Population and evidence", "",
        f"- Source: `{db_path}` (SQLite read-only/query-only)",
        f"- Independent BMW root VDE/applications: {summary.independent_applications}",
        f"- Application lineages used for grouped CV: {groups.application_lineage.nunique()}",
        f"- Pair relations: {json.dumps(rel, sort_keys=True)}",
        f"- Research groups: {json.dumps(gc, sort_keys=True)}",
        "- Group formation used external/structured identity evidence only; no ABC, force, residual, CdA, or RRC field was accepted by the classifier.",
        "- The 15 v0.2.2 cases are a forensic evidence subset; the statistical population is the full independent BMW root population.", "",
        "## Controlled grouped OOF results", "",
        "| Model | RMSE [N] | MAE [N] |", "|---|---:|---:|",
    ]
    for model in ("MODEL_0", "MODEL_1", "MODEL_2"):
        lines.append(f"| {model} | {m.loc[model, 'oof_rmse_N']:.6f} | {m.loc[model, 'oof_mae_N']:.6f} |")
    lines += ["", f"- Model 1 Δ vs Model 0: {m.loc['MODEL_0','oof_rmse_N']-m.loc['MODEL_1','oof_rmse_N']:.6f} N; cluster-bootstrap 95% CI [{ci1[0]:.6f}, {ci1[1]:.6f}].",
              f"- Model 2 Δ vs Model 0: {m.loc['MODEL_0','oof_rmse_N']-m.loc['MODEL_2','oof_rmse_N']:.6f} N; cluster-bootstrap 95% CI [{ci2[0]:.6f}, {ci2[1]:.6f}].",
              f"- Model 2 Δ vs Model 1: {m.loc['MODEL_1','oof_rmse_N']-m.loc['MODEL_2','oof_rmse_N']:.6f} N; cluster-bootstrap 95% CI [{ci21[0]:.6f}, {ci21[1]:.6f}].", "",
              "## Inference and controls", "",
              f"- Stratified graph permutations completed: {len(permutation)-1}; empirical p={summary.permutation_p:.6f}.",
              f"- Controlled-baseline residual-pair similarity empirical p={residual_p:.6f}.",
              "- Permutation scheme: bijective evidence-graph node relabeling within transmission-type, gear-count, and drive strata; graph topology/group sizes are preserved.",
              "- Hyperparameters were selected by train-only inner grouped CV. Entire model/application lineages stayed within a single outer fold.", "",
              "### Negative controls", "", _markdown_table(controls), "", "### Ablations", "", _markdown_table(ablation), "",
              "## Forensic v0.2.2 subset", "",
              f"All {int(groups.forensic_v022_case.eq('YES').sum())} benchmark rows are flagged in `TRANSMISSION_RESEARCH_GROUPS_V03.csv`. They document canonical confidence, research confidence, evidence links, PARTIAL/MISMATCH applicability, Cd, frontal area, tires, and per-field research-use classification. PARTIAL evidence can form a research hypothesis but never upgrades canonical identity.",
              "The optional intrinsic torque model was skipped: exact application tire rolling radius was not independently established, so no radius was fabricated.", "",
              "## Scientific decision", "",
              f"- Research grouping signal supported: {'YES' if summary.signal_supported else 'NO'}",
              f"- Soft grouping adds value: {'YES' if summary.soft_adds_value else 'NO'}",
              "- Ready for component-loss identification: NO. Predictive grouping, even if present, is not causal transmission-loss identification and aero/tire identifiability remains insufficient.", "",
              "## Safety", "",
              f"- Candidate SHA256 before: `{summary.db_hash_before}`", f"- Candidate SHA256 after: `{summary.db_hash_after}`",
              "- Canonical writes: 0", "- Canonical identity rules changed: NO", "",
              "## Required status block", "", "```text",
              "TRANSMISSION_RESEARCH_GROUPING_VERSION = 0.3", "",
              "CANONICAL_IDENTITY_RULES_CHANGED = NO", "RESEARCH_CONFIDENCE_LAYER_READY = YES",
              "SOFT_EVIDENCE_WEIGHTING_READY = YES", "OUTCOME_LEAKAGE_FOUND = NO", "",
              f"INDEPENDENT_APPLICATIONS = {summary.independent_applications}", "",
              f"CONFIRMED_RESEARCH_GROUPS = {gc.get('CONFIRMED', 0)}",
              f"PROBABLE_RESEARCH_GROUPS = {gc.get('PROBABLE', 0)}",
              f"PLAUSIBLE_RESEARCH_GROUPS = {gc.get('PLAUSIBLE', 0)}",
              "CONFLICTING_RESEARCH_GROUPS = 0",
              f"UNRESOLVED_RESEARCH_GROUPS = {gc.get('UNRESOLVED', 0)}", "",
              f"MODEL0_OOF_RMSE_N = {summary.model0_rmse:.6f}",
              f"MODEL1_OOF_RMSE_N = {summary.model1_rmse:.6f}",
              f"MODEL2_OOF_RMSE_N = {summary.model2_rmse:.6f}", "",
              f"MODEL1_DELTA_VS_MODEL0_N = {delta1:.6f}",
              f"MODEL2_DELTA_VS_MODEL0_N = {delta2:.6f}",
              f"MODEL2_DELTA_VS_MODEL1_N = {delta21:.6f}", "",
              f"PERMUTATIONS_COMPLETED = {len(permutation) - 1}",
              f"PERMUTATION_TEST_INSUFFICIENT = {'YES' if len(permutation) - 1 < 500 else 'NO'}",
              f"PERMUTATION_P_VALUE = {summary.permutation_p:.6f}", "",
              f"RESIDUAL_SIMILARITY_SUPPORT = {'YES' if residual_p <= 0.05 else 'NO'}",
              f"NEGATIVE_CONTROLS_PASSED = {'YES' if negative_controls_passed else 'NO'}",
              f"WEIGHT_SENSITIVITY_STABLE = {'YES' if sensitivity_stable else 'NO'}", "",
              f"RESEARCH_GROUP_SIGNAL_SUPPORTED = {'YES' if summary.signal_supported else 'NO'}",
              f"SOFT_GROUPING_ADDS_VALUE = {'YES' if summary.soft_adds_value else 'NO'}",
              "READY_FOR_COMPONENT_LOSS_IDENTIFICATION = NO", "",
              "CANONICAL_WRITE_DISABLED = YES", "PRODUCTION_DB_CHANGED = NO", "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(db_path: Path = DEFAULT_DB, evidence_path: Path = DEFAULT_EVIDENCE, enrichment_path: Path = DEFAULT_ENRICHMENT, output_dir: Path = DEFAULT_OUTPUT, permutations: int = PERMUTATIONS) -> ExperimentSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    before = sha256_file(db_path)
    vdes, runs, fuelcons, _ = load_canonical_source(db_path)
    roots, _, _ = prepare_population(vdes, runs, fuelcons)
    roots = roots[roots.make_norm.eq("BMW")].copy().reset_index(drop=True)
    required = ["test_mass_kg", "coast_A_N", "coast_B_N_per_kph", "coast_C_N_per_kph2"]
    if not roots[required].notna().all(axis=1).all():
        raise ValueError("BMW population has missing primary outcome fields; exclusions must be explicitly reviewed")
    evidence = pd.read_csv(evidence_path)
    apps = _evidence_records(roots, evidence)
    pairwise = build_pairwise_evidence(apps)
    groups = assign_research_groups(apps, pairwise)
    parent_by_id = dict(zip(vdes.vde_id.astype(int), vdes.vde_id_parent))
    def root_of(value):
        current = int(value)
        seen = set()
        while current in parent_by_id and not pd.isna(parent_by_id[current]):
            if current in seen:
                raise ValueError("carryover cycle")
            seen.add(current); current = int(parent_by_id[current])
        return current
    counts = vdes.vde_id.map(root_of).value_counts().to_dict()
    groups["exact_carryover_collapsed"] = groups.vde_id.map(lambda value: "YES" if counts.get(int(value), 1) > 1 else "NO")
    incident = {}
    for row in pairwise[pairwise.relation.ne("UNRESOLVED")].itertuples(index=False):
        for node in (row.application_i, row.application_j):
            incident.setdefault(node, []).append(row)
    groups["evidence_items"] = groups.independent_application_id.map(lambda node: ";".join(sorted({item for row in incident.get(node, []) for item in str(row.positive_evidence).split(";") if item})))
    groups["evidence_conflicts"] = groups.independent_application_id.map(lambda node: ";".join(sorted({item for row in incident.get(node, []) for item in str(row.negative_evidence).split(";") if item})))
    groups["evidence_weight_primary"] = groups.research_group_confidence.map(WEIGHT_MAPS["primary"]).fillna(0.0)
    groups["evidence_weight_conservative"] = groups.research_group_confidence.map(WEIGHT_MAPS["conservative"]).fillna(0.0)
    groups["evidence_weight_permissive"] = groups.research_group_confidence.map(WEIGHT_MAPS["permissive"]).fillna(0.0)
    groups["notes"] = "Research-only hypothesis; never canonical identity."
    long = _long_population(roots, groups)

    model_by_weight, fold_by_weight, tuning_audit, metric_rows = {}, {}, [], []
    predictions, folds, audit = run_models(long, groups, pairwise, "primary")
    model_by_weight["primary"], fold_by_weight["primary"] = predictions, folds
    tuning_audit.extend({"weight_mapping": "primary", **row} for row in audit)
    metric_rows.extend(_metric_rows(predictions, "primary"))
    for mapping in ("conservative", "permissive"):
        predictions, folds, audit = run_soft_sensitivity(
            long, pairwise, model_by_weight["primary"], fold_by_weight["primary"], mapping
        )
        model_by_weight[mapping], fold_by_weight[mapping] = predictions, folds
        tuning_audit.extend({"weight_mapping": mapping, **row} for row in audit)
        metric_rows.extend(_metric_rows(predictions, mapping))
    metrics = pd.DataFrame(metric_rows)
    primary = model_by_weight["primary"]
    primary_metrics = metrics[metrics.weight_mapping.eq("primary")].set_index("model")
    bootstrap = cluster_bootstrap(primary)
    permutation = permutation_test(long, primary, fold_by_weight["primary"], pairwise, groups, permutations)
    residual = residual_similarity(primary, pairwise, groups, permutations)
    controls = negative_controls(long, groups, float(primary_metrics.loc["MODEL_0", "oof_rmse_N"]), fold_by_weight["primary"])
    ablation = ablations(long, groups, pairwise, model_by_weight, fold_by_weight["primary"])

    delta1 = float(primary_metrics.loc["MODEL_0", "oof_rmse_N"] - primary_metrics.loc["MODEL_1", "oof_rmse_N"])
    delta2 = float(primary_metrics.loc["MODEL_0", "oof_rmse_N"] - primary_metrics.loc["MODEL_2", "oof_rmse_N"])
    delta21 = float(primary_metrics.loc["MODEL_1", "oof_rmse_N"] - primary_metrics.loc["MODEL_2", "oof_rmse_N"])
    pvalue = float(permutation.empirical_p_value.iloc[0])
    ci1, ci2, ci21 = _ci(bootstrap.model1_delta_vs_model0_N), _ci(bootstrap.model2_delta_vs_model0_N), _ci(bootstrap.model2_delta_vs_model1_N)
    fold_consistent = bool((fold_by_weight["primary"].model2_rmse_N < fold_by_weight["primary"].model0_rmse_N).all())
    control_pass = delta2 > max(0.0, float(controls.delta_vs_controlled_baseline_N.max()))
    probable_row = ablation[ablation.ablation.eq("external_confirmed_plus_probable")]
    not_plausible_only = bool(not probable_row.empty and probable_row.delta_vs_controlled_baseline_N.iloc[0] > 0)
    stable = all(_metric_rows(model_by_weight[name], name)[0]["oof_rmse_N"] - _metric_rows(model_by_weight[name], name)[2]["oof_rmse_N"] > 0 for name in WEIGHT_MAPS)
    signal = bool((delta1 > 0 and ci1[0] > 0 or delta2 > 0 and ci2[0] > 0) and pvalue <= 0.05 and control_pass and not_plausible_only and fold_consistent)
    soft_adds = bool(delta21 > 0 and ci21[0] > 0 and stable and pvalue <= 0.05)
    after = sha256_file(db_path)
    if before != after:
        raise AssertionError("Canonical candidate DB hash changed during read-only experiment")
    summary = ExperimentSummary(len(roots), float(primary_metrics.loc["MODEL_0", "oof_rmse_N"]), float(primary_metrics.loc["MODEL_1", "oof_rmse_N"]), float(primary_metrics.loc["MODEL_2", "oof_rmse_N"]), pvalue, signal, soft_adds, before, after)

    pairwise.to_csv(output_dir / "TRANSMISSION_PAIRWISE_EVIDENCE_V03.csv", index=False)
    groups.to_csv(output_dir / "TRANSMISSION_RESEARCH_GROUPS_V03.csv", index=False)
    metrics.to_csv(output_dir / "TRANSMISSION_SOFT_GROUP_MODEL_RESULTS_V03.csv", index=False)
    ablation.to_csv(output_dir / "TRANSMISSION_MODEL_ABLATIONS_V03.csv", index=False)
    permutation.to_csv(output_dir / "TRANSMISSION_PERMUTATION_RESULTS_V03.csv", index=False)
    residual.to_csv(output_dir / "TRANSMISSION_RESIDUAL_SIMILARITY_V03.csv", index=False)
    controls.to_csv(output_dir / "TRANSMISSION_NEGATIVE_CONTROLS_V03.csv", index=False)
    audit_rows = pd.DataFrame(tuning_audit)
    fold_audit = pd.concat(fold_by_weight.values(), ignore_index=True)
    forensic_audit = groups[groups.forensic_v022_case.eq("YES")][
        ["vde_id", "independent_application_id", "canonical_identity_confidence", "research_group_confidence", "external_application_match", "external_source_ids", "gear_ratios", "external_drag_coefficient_cd", "external_frontal_area_m2", "external_tire_size_front", "external_tire_size_rear", "external_tire_size_general", "research_use_classification"]
    ]
    audit_rows = pd.concat(
        [
            audit_rows.assign(audit_type="INNER_TUNING"),
            fold_audit.assign(audit_type="OUTER_FOLD"),
            forensic_audit.assign(audit_type="FORENSIC_V022_EVIDENCE"),
        ],
        ignore_index=True,
        sort=False,
    )
    audit_rows.to_csv(output_dir / "TRANSMISSION_RESEARCH_GROUP_AUDIT_V03.csv", index=False)
    _write_report(output_dir / "TRANSMISSION_RESEARCH_GROUPING_V03_REPORT.md", summary, metrics[metrics.weight_mapping.eq("primary")], pairwise, groups, bootstrap, permutation, residual, controls, ablation, db_path)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--enrichment", type=Path, default=DEFAULT_ENRICHMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_experiment(args.db, args.evidence, args.enrichment, args.output, args.permutations)
    print(json.dumps(summary.__dict__, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
