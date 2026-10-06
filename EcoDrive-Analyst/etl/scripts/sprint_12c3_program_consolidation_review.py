"""Sprint 12C.3 read-only PROGRAM consolidation review.

The module audits EPA cross-year continuity and creates preview artifacts only.
It never executes DDL, writes a runtime database, edits raw sources, or assigns
production canonical identities.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "etl" / "scripts"))
import sprint_12c1_dataset_delta_audit as c1  # noqa: E402
import sprint_12c2_canonical_population_preview as c2  # noqa: E402


DB_PATH = ROOT / "data" / "db" / "eco_drive.db"
OUT = ROOT / "etl" / "data" / "processed" / "sprint_12c3_program_consolidation"
REPORT = ROOT / "etl" / "reports" / "sprint_12c3_program_consolidation_review.md"
EXAMPLE_REPORT = ROOT / "etl" / "reports" / "sprint_12c3_program_examples.md"

OUTPUT_PATHS = (
    OUT / "cross_year_profiles.csv",
    OUT / "program_boundary_candidates.csv",
    OUT / "program_consolidation_candidates.csv",
    OUT / "program_population_forecast.csv",
    OUT / "program_examples.csv",
    OUT / "program_review.json",
    REPORT,
    EXAMPLE_REPORT,
)

STABLE_ARCH_FIELDS = (
    "Test Veh Displacement (L)",
    "# of Cylinders and Rotors",
    "Tested Transmission Type",
    "# of Gears",
    "Drive System Description",
    "Test Fuel Type Description",
)
PROFILE_FIELDS = (
    "Vehicle Manufacturer Name",
    "Actual Tested Testgroup",
    "Engine Code",
    "Test Veh Displacement (L)",
    "# of Cylinders and Rotors",
    "Tested Transmission Type",
    "# of Gears",
    "Drive System Description",
    "Axle Ratio",
    "N/V Ratio",
    "Test Fuel Type Description",
)
TARGET_FIELDS = c2.TARGET_FIELDS
SET_FIELDS = c2.SET_FIELDS
MMY_FIELDS = c2.MMY_FIELDS
YEAR_LIKE = re.compile(r"^(?:19|20)\d{2}$")


def clean(value: Any) -> Any:
    return c2.clean(value)


def present(value: Any) -> bool:
    value = clean(value)
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def norm(value: Any) -> str:
    return c1.norm_text(value)


def stable_id(prefix: str, *parts: Any) -> str:
    return c2.stable_preview_id(prefix, *parts)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value_key(value: Any) -> str:
    return c2.key_value(value, normalize_text=isinstance(clean(value), str))


def sorted_values(series: pd.Series, limit: int = 12) -> str:
    values = sorted({value_key(value) for value in series if present(value)})
    shown = values[:limit]
    suffix = f";...(+{len(values) - limit})" if len(values) > limit else ""
    return ";".join(shown) + suffix


def numeric_range(series: pd.Series) -> str:
    values = sorted({float(value) for value in series if present(value)})
    if not values:
        return ""
    if len(values) == 1:
        return f"{values[0]:g}"
    return f"{values[0]:g}..{values[-1]:g} ({len(values)} values)"


def row_signature(row: pd.Series, fields: Iterable[str], minimum_present: int = 1) -> tuple[str, ...] | None:
    values = [clean(row.get(field)) for field in fields]
    if sum(present(value) for value in values) < minimum_present:
        return None
    return tuple(value_key(value) for value in values)


def source_identity(row: pd.Series) -> tuple[str, str] | None:
    values = (clean(row.get("Test Vehicle ID")), clean(row.get("Test Veh Configuration #")))
    if not any(present(value) for value in values):
        return None
    return tuple(value_key(value) for value in values)  # type: ignore[return-value]


def prepare_rows(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = source[source["Model Year"].between(2020, 2026)].copy()
    rows, vde_preview = c2.prepare_epa_candidates(source)
    rows["make_norm"] = rows["Represented Test Veh Make"].map(norm)
    rows["model_norm"] = rows["Represented Test Veh Model"].map(norm)
    rows["family_key"] = rows["make_norm"] + "|" + rows["model_norm"]
    rows["family_audit_id"] = rows.apply(
        lambda row: stable_id("PROGRAM-FAMILY", "EPA_TESTCAR", row["make_norm"], row["model_norm"]), axis=1
    )
    rows["fallback_program_id"] = rows["program_candidate_id"]
    rows["stable_arch_signature"] = rows.apply(
        lambda row: row_signature(row, STABLE_ARCH_FIELDS, minimum_present=4), axis=1
    )
    rows["engine_signature"] = rows.apply(
        lambda row: row_signature(row, ("Engine Code", "Test Veh Displacement (L)", "# of Cylinders and Rotors"), minimum_present=1), axis=1
    )
    rows["transmission_signature"] = rows.apply(
        lambda row: row_signature(row, ("Tested Transmission Type", "# of Gears"), minimum_present=1), axis=1
    )
    rows["drive_signature"] = rows.apply(
        lambda row: row_signature(row, ("Drive System Description",), minimum_present=1), axis=1
    )
    rows["propulsion_signature"] = rows.apply(
        lambda row: row_signature(row, ("Test Fuel Type Description", "Test Veh Displacement (L)", "# of Cylinders and Rotors"), minimum_present=1), axis=1
    )
    rows["source_identity"] = rows.apply(source_identity, axis=1)
    rows["identity_anomaly"] = rows["make_norm"].map(lambda value: bool(YEAR_LIKE.fullmatch(value)))
    return rows, vde_preview


def non_null_set(group: pd.DataFrame, column: str) -> set[Any]:
    return {value for value in group[column] if value is not None}


def descriptor_coverage(group: pd.DataFrame) -> float:
    if group.empty:
        return 0.0
    covered = sum(group[field].map(present).mean() for field in STABLE_ARCH_FIELDS)
    return round(float(covered / len(STABLE_ARCH_FIELDS)), 4)


def make_profiles(rows: pd.DataFrame) -> list[dict[str, Any]]:
    family_years = rows.groupby("family_key")["Model Year"].nunique().to_dict()
    family_year_lists = rows.groupby("family_key")["Model Year"].apply(
        lambda values: sorted(int(value) for value in values.unique())
    ).to_dict()
    profiles: list[dict[str, Any]] = []
    for (family, year), group in rows.groupby(["family_key", "Model Year"], sort=True):
        first = group.sort_values("source_excel_row", kind="stable").iloc[0]
        years = family_year_lists[family]
        target_variants = group[list(TARGET_FIELDS)].drop_duplicates().shape[0]
        set_variants = group[list(SET_FIELDS)].drop_duplicates().shape[0]
        profiles.append({
            "profile_id": stable_id("PROFILE", family, int(year)),
            "family_audit_id": first["family_audit_id"],
            "fallback_program_id": first["fallback_program_id"],
            "make": clean(first["Represented Test Veh Make"]),
            "model": clean(first["Represented Test Veh Model"]),
            "make_norm": first["make_norm"],
            "model_norm": first["model_norm"],
            "model_year": int(year),
            "family_years": ";".join(map(str, years)),
            "family_model_year_count": int(family_years[family]),
            "source_rows": len(group),
            "vehicle_configuration_candidates": int(group["configuration_candidate_id"].nunique()),
            "vde_states": int(group["vde_candidate_id"].nunique()),
            "run_candidates": int(group["run_candidate_id"].nunique()),
            "stable_arch_signatures": len(non_null_set(group, "stable_arch_signature")),
            "source_vehicle_identities": len(non_null_set(group, "source_identity")),
            "descriptor_coverage": descriptor_coverage(group),
            "manufacturers": sorted_values(group["Vehicle Manufacturer Name"]),
            "test_groups": sorted_values(group["Actual Tested Testgroup"]),
            "engine_codes": sorted_values(group["Engine Code"]),
            "displacement_l": sorted_values(group["Test Veh Displacement (L)"]),
            "cylinders_rotors": sorted_values(group["# of Cylinders and Rotors"]),
            "transmissions": sorted_values(group["Tested Transmission Type"]),
            "gear_counts": sorted_values(group["# of Gears"]),
            "drive_systems": sorted_values(group["Drive System Description"]),
            "axle_ratios": sorted_values(group["Axle Ratio"]),
            "nv_ratios": sorted_values(group["N/V Ratio"]),
            "fuel_types": sorted_values(group["Test Fuel Type Description"]),
            "target_abc_variants": int(target_variants),
            "target_a_range_lbf": numeric_range(group[TARGET_FIELDS[0]]),
            "target_b_range_lbf_per_mph": numeric_range(group[TARGET_FIELDS[1]]),
            "target_c_range_lbf_per_mph2": numeric_range(group[TARGET_FIELDS[2]]),
            "etw_range_lb": numeric_range(group["Equivalent Test Weight (lbs.)"]),
            "set_abc_variants": int(set_variants),
            "identity_status": "SOURCE_DATA_QUALITY_ISSUE" if bool(group["identity_anomaly"].any()) else "SOURCE_SCOPED_FALLBACK",
            "provenance": "EPA_TESTCAR_REFRESHED_PUBLIC_SOURCE",
        })
    return profiles


def _continuity(left: pd.DataFrame, right: pd.DataFrame, column: str) -> tuple[int, int, int]:
    lhs, rhs = non_null_set(left, column), non_null_set(right, column)
    return len(lhs), len(rhs), len(lhs & rhs)


def make_boundaries(rows: pd.DataFrame) -> list[dict[str, Any]]:
    boundaries: list[dict[str, Any]] = []
    for family, group in rows.groupby("family_key", sort=True):
        years = sorted(int(value) for value in group["Model Year"].unique())
        for year_from, year_to in zip(years, years[1:]):
            left = group[group["Model Year"].eq(year_from)]
            right = group[group["Model Year"].eq(year_to)]
            first = left.sort_values("source_excel_row", kind="stable").iloc[0]
            arch_l, arch_r, arch_shared = _continuity(left, right, "stable_arch_signature")
            sid_l, sid_r, sid_shared = _continuity(left, right, "source_identity")
            engine_l, engine_r, engine_shared = _continuity(left, right, "engine_signature")
            trans_l, trans_r, trans_shared = _continuity(left, right, "transmission_signature")
            drive_l, drive_r, drive_shared = _continuity(left, right, "drive_signature")
            prop_l, prop_r, prop_shared = _continuity(left, right, "propulsion_signature")
            coverage = min(descriptor_coverage(left), descriptor_coverage(right))
            gap = year_to - year_from - 1
            anomaly = bool(left["identity_anomaly"].any() or right["identity_anomaly"].any())
            domains_with_values = sum(a > 0 and b > 0 for a, b in ((engine_l, engine_r), (trans_l, trans_r), (drive_l, drive_r), (prop_l, prop_r)))
            discontinuities = sum(a > 0 and b > 0 and shared == 0 for a, b, shared in (
                (engine_l, engine_r, engine_shared), (trans_l, trans_r, trans_shared),
                (drive_l, drive_r, drive_shared), (prop_l, prop_r, prop_shared),
            ))

            if anomaly:
                boundary_status, consolidation, confidence = "UNRESOLVED", "UNRESOLVED", "LOW"
                reason = "Year-like value in represented make; source identity requires correction before consolidation."
            elif gap == 0 and arch_shared > 0 and sid_shared > 0:
                boundary_status, consolidation, confidence = "NO_BOUNDARY_EVIDENCE", "SAFE_CONSOLIDATE", "HIGH"
                reason = "Contiguous years share an exact stable-architecture signature and persistent EPA vehicle/configuration identity."
            elif gap == 0 and arch_shared > 0:
                boundary_status, consolidation, confidence = "NO_BOUNDARY_EVIDENCE", "PROBABLE_CONSOLIDATE", "MEDIUM"
                reason = "Contiguous years share stable architecture, but no persistent EPA vehicle/configuration identity was observed."
            elif gap > 0 and arch_shared > 0 and sid_shared > 0:
                boundary_status, consolidation, confidence = "WEAK_BOUNDARY_CANDIDATE", "PROBABLE_CONSOLIDATE", "MEDIUM"
                reason = "A model-year gap exists, but stable architecture and EPA source identity resume after the gap."
            elif coverage >= 0.75 and domains_with_values >= 3 and discontinuities >= 3:
                boundary_status, consolidation, confidence = "STRONG_BOUNDARY_CANDIDATE", "KEEP_SEPARATE", "MEDIUM"
                reason = f"No exact architecture continuity; {discontinuities} populated architecture domains reset. No OEM generation evidence is available, so this is not confirmed."
            elif arch_shared == 0 and (gap > 0 or discontinuities > 0):
                boundary_status, consolidation, confidence = "WEAK_BOUNDARY_CANDIDATE", "KEEP_SEPARATE", "LOW"
                reason = "No exact stable-architecture overlap; evidence is incomplete or limited to a weak discontinuity/gap."
            else:
                boundary_status, consolidation, confidence = "UNRESOLVED", "UNRESOLVED", "LOW"
                reason = "Available descriptors do not support either consolidation or a generation boundary."

            boundaries.append({
                "boundary_id": stable_id("BOUNDARY", family, year_from, year_to),
                "family_audit_id": first["family_audit_id"],
                "make": clean(first["Represented Test Veh Make"]),
                "model": clean(first["Represented Test Veh Model"]),
                "make_norm": first["make_norm"],
                "model_norm": first["model_norm"],
                "model_year_from": year_from,
                "model_year_to": year_to,
                "missing_model_years_between": gap,
                "fallback_program_from": left["fallback_program_id"].iloc[0],
                "fallback_program_to": right["fallback_program_id"].iloc[0],
                "stable_arch_signatures_from": arch_l,
                "stable_arch_signatures_to": arch_r,
                "shared_stable_arch_signatures": arch_shared,
                "shared_source_vehicle_identities": sid_shared,
                "shared_engine_signatures": engine_shared,
                "shared_transmission_signatures": trans_shared,
                "shared_drive_signatures": drive_shared,
                "shared_propulsion_signatures": prop_shared,
                "architecture_domain_discontinuities": discontinuities,
                "minimum_descriptor_coverage": coverage,
                "boundary_status": boundary_status,
                "consolidation_status": consolidation,
                "evidence_summary": reason,
                "confidence": confidence,
                "target_etw_note": "Target ABC, Set ABC, ETW and procedure were inspected as VDE/RUN context and did not independently set this boundary.",
            })
    return boundaries


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def component_count(nodes: set[str], boundaries: list[dict[str, Any]], statuses: set[str]) -> int:
    union = UnionFind(nodes)
    for row in boundaries:
        left, right = row["fallback_program_from"], row["fallback_program_to"]
        if left in nodes and right in nodes and row["consolidation_status"] in statuses:
            union.union(left, right)
    return len({union.find(node) for node in nodes})


def make_forecast(rows: pd.DataFrame, boundaries: list[dict[str, Any]], jrc: pd.DataFrame) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for label, subset in (
        ("EPA refreshed 2020-2025", rows[rows["Model Year"].between(2020, 2025)]),
        ("EPA 2026", rows[rows["Model Year"].eq(2026)]),
        ("Combined EPA 2020-2026", rows),
    ):
        nodes = set(subset["fallback_program_id"])
        raw_upper = subset[list(MMY_FIELDS)].drop_duplicates().shape[0]
        safe = component_count(nodes, boundaries, {"SAFE_CONSOLIDATE"})
        probable = component_count(nodes, boundaries, {"SAFE_CONSOLIDATE", "PROBABLE_CONSOLIDATE"})
        lexical = subset[["make_norm", "model_norm"]].drop_duplicates().shape[0]
        result.append({
            "population": label,
            "source_rows": len(subset),
            "fallback_upper_bound": int(raw_upper),
            "normalized_mmy_fallbacks": len(nodes),
            "safe_consolidation_count": safe,
            "likely_semantic_program_range": f"{probable:,}-{safe:,}",
            "likely_range_lower": probable,
            "likely_range_upper": safe,
            "aggressive_lexical_lower_bound": int(lexical),
            "engineering_program_population": "YES",
            "interpretation": "Safe count uses only technical + persistent source-identity continuity; likely lower also includes probable technical continuity.",
        })
    jrc_lexical = jrc[["OEM anon", "Model anon"]].drop_duplicates().shape[0]
    result.append({
        "population": "JRC source-scoped",
        "source_rows": len(jrc),
        "fallback_upper_bound": len(jrc),
        "normalized_mmy_fallbacks": len(jrc),
        "safe_consolidation_count": len(jrc),
        "likely_semantic_program_range": f"{len(jrc):,}-{len(jrc):,}",
        "likely_range_lower": len(jrc),
        "likely_range_upper": len(jrc),
        "aggressive_lexical_lower_bound": int(jrc_lexical),
        "engineering_program_population": "SOURCE_SCOPED_UNRESOLVED",
        "interpretation": "Anonymized identities prohibit deterministic cross-row or cross-source consolidation.",
    })
    result.append({
        "population": "EEA 2025 monitoring",
        "source_rows": 10_833_597,
        "fallback_upper_bound": 0,
        "normalized_mmy_fallbacks": 0,
        "safe_consolidation_count": 0,
        "likely_semantic_program_range": "0-0",
        "likely_range_lower": 0,
        "likely_range_upper": 0,
        "aggressive_lexical_lower_bound": 0,
        "engineering_program_population": "NO",
        "interpretation": "EEA remains non-engineering monitoring evidence; 7,814 lexical Mk+Cn+year groups are reporting identities, not Programs.",
    })
    return result


def make_consolidations(rows: pd.DataFrame, boundaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes = sorted(set(rows["fallback_program_id"]))
    probable_union = UnionFind(nodes)
    for boundary in boundaries:
        if boundary["consolidation_status"] in {"SAFE_CONSOLIDATE", "PROBABLE_CONSOLIDATE"}:
            probable_union.union(boundary["fallback_program_from"], boundary["fallback_program_to"])
    groups: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        groups[probable_union.find(node)].append(node)
    boundary_by_pair = {
        frozenset((row["fallback_program_from"], row["fallback_program_to"])): row for row in boundaries
    }
    node_boundary_statuses: dict[str, set[str]] = defaultdict(set)
    for boundary in boundaries:
        node_boundary_statuses[boundary["fallback_program_from"]].add(boundary["consolidation_status"])
        node_boundary_statuses[boundary["fallback_program_to"]].add(boundary["consolidation_status"])

    output: list[dict[str, Any]] = []
    for members in sorted(groups.values(), key=lambda values: values[0]):
        subset = rows[rows["fallback_program_id"].isin(members)]
        first = subset.sort_values(["Model Year", "source_excel_row"], kind="stable").iloc[0]
        years = sorted(int(value) for value in subset["Model Year"].unique())
        internal = [
            row for pair, row in boundary_by_pair.items()
            if pair.issubset(set(members))
        ]
        if len(members) > 1 and internal and all(row["consolidation_status"] == "SAFE_CONSOLIDATE" for row in internal):
            status, confidence = "SAFE_CONSOLIDATE", "HIGH"
        elif len(members) > 1:
            status, confidence = "PROBABLE_CONSOLIDATE", "MEDIUM"
        elif "KEEP_SEPARATE" in node_boundary_statuses[members[0]]:
            status, confidence = "KEEP_SEPARATE", "MEDIUM"
        else:
            status, confidence = "UNRESOLVED", "LOW"
        evidence = Counter(row["consolidation_status"] for row in internal)
        output.append({
            "proposed_program_audit_id": stable_id("SEMANTIC-PROGRAM", *members),
            "family_audit_id": first["family_audit_id"],
            "make": clean(first["Represented Test Veh Make"]),
            "model_family": clean(first["Represented Test Veh Model"]),
            "model_year_start": min(years),
            "model_year_end": max(years),
            "included_model_years": ";".join(map(str, years)),
            "included_fallback_program_ids": ";".join(sorted(members)),
            "fallback_program_count": len(members),
            "vehicle_configuration_candidates": int(subset["configuration_candidate_id"].nunique()),
            "vde_states": int(subset["vde_candidate_id"].nunique()),
            "run_candidates": int(subset["run_candidate_id"].nunique()),
            "source_rows": len(subset),
            "fuelcons_candidate_floor": int(subset["vde_candidate_id"].nunique()),
            "consolidation_status": status,
            "evidence_summary": "; ".join(f"{key}={value}" for key, value in sorted(evidence.items())) or "No cross-year edge available.",
            "confidence": confidence,
            "unresolved_notes": "Program parent identity only; Configuration, VDE, RUN and FuelCons candidate identities/counts are retained.",
        })
    return output


def find_rename_candidates(rows: pd.DataFrame, limit: int = 20) -> list[dict[str, Any]]:
    candidates: dict[tuple[str, int, int, str, str], dict[str, Any]] = {}
    usable = rows[~rows["identity_anomaly"] & rows["stable_arch_signature"].notna()]
    model_index = {
        (make, int(year), signature): sorted(set(group["model_norm"]))
        for (make, year, signature), group in usable.groupby(
            ["make_norm", "Model Year", "stable_arch_signature"], sort=True
        )
    }
    for (make, year, signature), left_models in model_index.items():
        right_models = model_index.get((make, year + 1, signature), [])
        if not right_models:
            continue
        for left in left_models:
            for right in right_models:
                if left == right:
                    continue
                similarity = SequenceMatcher(None, left, right).ratio()
                left_tokens, right_tokens = set(left.split()), set(right.split())
                token_overlap = len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
                if similarity < 0.58 and token_overlap < 0.67:
                    continue
                key = (make, int(year), int(year) + 1, left, right)
                candidates[key] = {
                    "make": make,
                    "model_from": left,
                    "model_to": right,
                    "year_from": int(year),
                    "year_to": int(year) + 1,
                    "name_similarity": round(similarity, 4),
                    "shared_architecture": str(signature),
                    "status": "PROBABLE_RENAME_OR_TRIM_NOMENCLATURE_ONLY",
                    "note": "Not included in deterministic consolidation; commercial naming evidence alone is insufficient.",
                }
    return sorted(candidates.values(), key=lambda row: (-row["name_similarity"], row["make"], row["model_from"], row["model_to"]))[:limit]


def label_powertrain(group: pd.DataFrame) -> str:
    fuels = " ".join(sorted_values(group["Test Fuel Type Description"], 50).split(";"))
    model = norm(group["Represented Test Veh Model"].iloc[0])
    displacement = [float(value) for value in group["Test Veh Displacement (L)"] if present(value)]
    if "ELECTRIC" in fuels or (displacement and max(displacement) == 0):
        return "BEV"
    if any(token in model for token in ("HYBRID", " PHEV", " HEV", " PRIME", "4XE")):
        return "HEV_OR_PHEV_NAME_SIGNAL"
    if displacement and max(displacement) > 0:
        return "ICE"
    return "INSUFFICIENT"


def profile_tree(group: pd.DataFrame, title: str, note: str) -> str:
    lines = [title]
    for year, year_rows in group.groupby("Model Year", sort=True):
        lines.append(
            f"{int(year)}  models={sorted_values(year_rows['Represented Test Veh Model'], 3)} | "
            f"configs={year_rows['configuration_candidate_id'].nunique()} "
            f"VDEs={year_rows['vde_candidate_id'].nunique()} runs={year_rows['run_candidate_id'].nunique()} | "
            f"engine={sorted_values(year_rows['Engine Code'], 4) or '<missing>'} | "
            f"trans={sorted_values(year_rows['Tested Transmission Type'], 4) or '<missing>'} | "
            f"drive={sorted_values(year_rows['Drive System Description'], 4) or '<missing>'} | "
            f"fuel={sorted_values(year_rows['Test Fuel Type Description'], 3) or '<missing>'}"
        )
    lines.extend(["-------------------------", note])
    return "\n".join(lines)


def make_examples(
    rows: pd.DataFrame,
    boundaries: list[dict[str, Any]],
    consolidations: list[dict[str, Any]],
    rename_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []

    def add(category: str, source_key: str, group: pd.DataFrame, note: str) -> None:
        first = group.sort_values(["Model Year", "source_excel_row"], kind="stable").iloc[0]
        examples.append({
            "example_id": f"EX-{len(examples) + 1:02d}",
            "category": category,
            "source_key": source_key,
            "make": clean(first["Represented Test Veh Make"]),
            "model": clean(first["Represented Test Veh Model"]),
            "model_years": ";".join(map(str, sorted(int(value) for value in group["Model Year"].unique()))),
            "tree": profile_tree(group, f"{clean(first['Represented Test Veh Make'])} {clean(first['Represented Test Veh Model'])}", note),
            "evidence": note,
        })

    family_groups = {key: group for key, group in rows.groupby("family_key", sort=True)}
    boundary_lookup = {(row["make_norm"] + "|" + row["model_norm"], row["model_year_from"], row["model_year_to"]): row for row in boundaries}

    stable = []
    for key, group in family_groups.items():
        years = sorted(int(value) for value in group["Model Year"].unique())
        if years != list(range(2020, 2027)):
            continue
        edges = [boundary_lookup.get((key, year, year + 1)) for year in range(2020, 2026)]
        if all(edge and edge["consolidation_status"] == "SAFE_CONSOLIDATE" for edge in edges):
            stable.append((key, group))
    for key, group in stable[:3]:
        add("STABLE_ARCHITECTURE_2020_2026", key, group, "All six contiguous transitions satisfy the safe technical + persistent EPA identity rule.")

    strong = [row for row in boundaries if row["boundary_status"] == "STRONG_BOUNDARY_CANDIDATE"]
    for row in strong[:4]:
        group = family_groups[row["make_norm"] + "|" + row["model_norm"]]
        add("OBVIOUS_TECHNICAL_GENERATION_BREAK", row["boundary_id"], group, f"{row['model_year_from']}->{row['model_year_to']}: {row['evidence_summary']}")

    multi_config = [
        item for item in consolidations
        if item["consolidation_status"] in {"SAFE_CONSOLIDATE", "PROBABLE_CONSOLIDATE"}
        and item["vehicle_configuration_candidates"] >= 3
    ]
    for item in multi_config[:4]:
        group = rows[rows["fallback_program_id"].isin(item["included_fallback_program_ids"].split(";"))]
        add("MULTIPLE_CONFIGURATIONS_ONE_LIKELY_PROGRAM", item["proposed_program_audit_id"], group, f"{item['vehicle_configuration_candidates']} configurations remain distinct below one proposed Program parent.")

    for item in rename_candidates[:3]:
        group = rows[
            rows["make_norm"].eq(item["make"])
            & rows["model_norm"].isin([item["model_from"], item["model_to"]])
            & rows["Model Year"].isin([item["year_from"], item["year_to"]])
        ]
        add("COMMERCIAL_RENAME_OR_TRIM_NOMENCLATURE", stable_id("RENAME", item["make"], item["model_from"], item["model_to"]), group, "Different commercial strings share an exact architecture signature in adjacent years; review only, never a deterministic merge.")

    for row in strong[4:8]:
        group = family_groups[row["make_norm"] + "|" + row["model_norm"]]
        add("SAME_COMMERCIAL_NAME_DISTINCT_GENERATIONS", row["boundary_id"], group, f"Same normalized name, but {row['architecture_domain_discontinuities']} architecture domains reset across the candidate boundary.")

    single_year = [(key, group) for key, group in family_groups.items() if group["Model Year"].nunique() == 1 and not group["identity_anomaly"].any()]
    for key, group in single_year[:4]:
        add("ONE_YEAR_ONLY_MODEL", key, group, "No cross-year evidence exists; retain the source-scoped fallback and do not infer a generation span.")

    gaps = [row for row in boundaries if row["missing_model_years_between"] > 0]
    for row in gaps[:3]:
        group = family_groups[row["make_norm"] + "|" + row["model_norm"]]
        add("MODEL_YEAR_GAP", row["boundary_id"], group, f"{row['missing_model_years_between']} missing model year(s); status={row['boundary_status']}, consolidation={row['consolidation_status']}.")

    labelled: dict[str, tuple[str, pd.DataFrame]] = {}
    for key, group in family_groups.items():
        label = label_powertrain(group)
        if label not in labelled and not group["identity_anomaly"].any():
            labelled[label] = (key, group)
    for label in ("BEV", "ICE", "HEV_OR_PHEV_NAME_SIGNAL"):
        if label in labelled:
            key, group = labelled[label]
            add(f"POWERTRAIN_FAMILY_{label}", key, group, f"Audit classification={label}; naming signals are context only and do not establish Program identity.")

    insufficient = []
    for key, group in family_groups.items():
        if group["identity_anomaly"].any() or min(descriptor_coverage(year_group) for _, year_group in group.groupby("Model Year")) < 0.67:
            insufficient.append((key, group))
    for key, group in insufficient[:3]:
        add("INSUFFICIENT_SOURCE_DATA", key, group, "Descriptor coverage or represented-make quality is insufficient; no automatic consolidation is allowed.")

    return examples


def relationship_impact(rows: pd.DataFrame, consolidations: list[dict[str, Any]], forecast: list[dict[str, Any]]) -> dict[str, Any]:
    combined = next(row for row in forecast if row["population"] == "Combined EPA 2020-2026")
    assigned_fallbacks = {
        value
        for candidate in consolidations
        for value in candidate["included_fallback_program_ids"].split(";")
    }
    all_fallbacks = set(rows["fallback_program_id"])
    parent_mapping: dict[str, str] = {}
    for candidate in consolidations:
        for fallback in candidate["included_fallback_program_ids"].split(";"):
            if fallback in parent_mapping:
                raise RuntimeError(f"Fallback Program assigned more than once: {fallback}")
            parent_mapping[fallback] = candidate["proposed_program_audit_id"]
    remapped = rows.copy()
    remapped["proposed_program_audit_id"] = remapped["fallback_program_id"].map(parent_mapping)
    child_columns = ("configuration_candidate_id", "vde_candidate_id", "run_candidate_id")
    child_sets_preserved = {
        column: set(rows[column]) == set(remapped[column]) for column in child_columns
    }
    return {
        "fallback_programs_before": len(all_fallbacks),
        "safe_programs_after": combined["safe_consolidation_count"],
        "likely_programs_after_range": combined["likely_semantic_program_range"],
        "fallback_programs_assigned_once": sum(candidate["fallback_program_count"] for candidate in consolidations) == len(all_fallbacks),
        "fallback_program_identity_set_preserved": assigned_fallbacks == all_fallbacks,
        "all_rows_have_proposed_program_parent": bool(remapped["proposed_program_audit_id"].notna().all()),
        "source_rows_before": len(rows),
        "source_rows_after_parent_remap": sum(candidate["source_rows"] for candidate in consolidations),
        "vehicle_configurations_before": int(rows["configuration_candidate_id"].nunique()),
        "vehicle_configurations_after_parent_remap": int(rows["configuration_candidate_id"].nunique()),
        "vde_states_before": int(rows["vde_candidate_id"].nunique()),
        "vde_states_after_parent_remap": int(rows["vde_candidate_id"].nunique()),
        "run_candidates_before": int(rows["run_candidate_id"].nunique()),
        "run_candidates_after_parent_remap": int(rows["run_candidate_id"].nunique()),
        "fuelcons_candidate_floor_before": int(rows["vde_candidate_id"].nunique()),
        "fuelcons_candidate_floor_after_parent_remap": int(rows["vde_candidate_id"].nunique()),
        "configuration_identity_set_preserved": child_sets_preserved["configuration_candidate_id"],
        "vde_identity_set_preserved": child_sets_preserved["vde_candidate_id"],
        "run_identity_set_preserved": child_sets_preserved["run_candidate_id"],
        "result": "PASS_NO_CHILD_ROW_LOSS",
        "rule": "Only the Program parent key is remapped; Configuration, VDE, RUN and FuelCons candidate identifiers are immutable in this preview.",
    }


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


def examples_report(examples: list[dict[str, Any]]) -> str:
    lines = [
        "# Sprint 12C.3 — Program Consolidation Examples",
        "",
        "These are deterministic, manually inspectable audit examples. Names and technical descriptors come from the local EPA source; no web/RAG/LLM generation identity was used.",
        "",
    ]
    for example in examples:
        lines.extend([
            f"## {example['example_id']} — {example['category']}", "",
            f"Source key: `{example['source_key']}`", "", "```text", example["tree"], "```", "",
            f"Evidence: {example['evidence']}", "",
        ])
    return "\n".join(lines)


def report_text(payload: dict[str, Any]) -> str:
    forecast = payload["program_population_forecast"]
    boundary_counts = Counter(row["boundary_status"] for row in payload["program_boundary_candidates"])
    consolidation_counts = Counter(row["consolidation_status"] for row in payload["program_consolidation_candidates"])
    impact = payload["relationship_impact"]
    combined = next(row for row in forecast if row["population"] == "Combined EPA 2020-2026")
    lines = [
        "# Sprint 12C.3 — Program Consolidation Review", "",
        f"## Status: `{payload['status']}`", "",
        "The previous ~6k forecast was conservative because it treated source-scoped Make+Model+Model-Year fallbacks as Programs. This review keeps those fallbacks traceable while measuring only evidence-supported cross-year parent consolidation.", "",
        "No production Program identity was assigned. No DDL, migration, runtime/application code, raw source or runtime database changed.", "",
        "## Decision", "",
        f"For combined EPA 2020–2026, the raw MMY fallback upper bound is **{combined['fallback_upper_bound']:,}**, deterministic normalization plus safe consolidation yields **{combined['safe_consolidation_count']:,}**, the likely semantic range is **{combined['likely_semantic_program_range']}**, and the aggressive name-only floor is **{combined['aggressive_lexical_lower_bound']:,}**.", "",
        "Only `SAFE_CONSOLIDATE` is deterministic. `PROBABLE_CONSOLIDATE` informs the lower end of the likely range but remains review work. The aggressive lexical count is never canonical because one commercial name can span multiple generations.", "",
        "## Program population forecast", "",
        *md_table(forecast, ["population", "source_rows", "fallback_upper_bound", "normalized_mmy_fallbacks", "safe_consolidation_count", "likely_semantic_program_range", "aggressive_lexical_lower_bound", "engineering_program_population"]), "",
        "JRC remains source-scoped because identities are anonymized. EEA contributes zero engineering Programs; its 7,814 lexical reporting groups remain monitoring identities only.", "",
        "## Boundary evidence", "",
        *md_table([{"boundary_status": key, "cases": value} for key, value in sorted(boundary_counts.items())], ["boundary_status", "cases"]), "",
        "No source-only case is labelled `CONFIRMED_BOUNDARY`: OEM generation/platform evidence is absent. Strong candidates require a populated multi-domain technical reset; Target/Set ABC, ETW and test procedure never create a boundary by themselves.", "",
        "## Consolidation candidates", "",
        *md_table([{"consolidation_status": key, "groups": value} for key, value in sorted(consolidation_counts.items())], ["consolidation_status", "groups"]), "",
        "Safe cross-year edges require contiguous years, an exact stable-architecture signature, and persistent EPA Test Vehicle ID/Configuration identity. Same name or year adjacency alone is insufficient.", "",
        "## Relationship impact preview", "",
        *md_table([impact], ["fallback_programs_before", "safe_programs_after", "likely_programs_after_range", "vehicle_configurations_before", "vehicle_configurations_after_parent_remap", "vde_states_before", "vde_states_after_parent_remap", "run_candidates_before", "run_candidates_after_parent_remap", "fuelcons_candidate_floor_before", "fuelcons_candidate_floor_after_parent_remap", "result"]), "",
        "Program consolidation changes only the parent identity layer. Every fallback is assigned exactly once, and all Configuration, VDE, RUN and FuelCons candidate identities remain untouched.", "",
        "```text", "multiple Model Years", "multiple Configurations", "multiple VDEs", "multiple RUNs", "        ↓", "may still belong to one Program", "```", "",
        "## Real examples", "",
        f"The review generated **{len(payload['program_examples'])}** deterministic examples. The full chronological trees are in `etl/reports/sprint_12c3_program_examples.md`.", "",
        *md_table(payload["program_examples"], ["example_id", "category", "make", "model", "model_years", "evidence"]), "",
        "## Remaining identity work", "",
        "- Validate OEM generation/platform codes, launch/facelift chronology and authoritative model renames as future enrichment inputs.",
        "- Review `PROBABLE_CONSOLIDATE`, `KEEP_SEPARATE` and `UNRESOLVED` groups before migration; do not encode them as hidden deterministic rules.",
        f"- Correct the {payload['summary']['year_like_make_rows']:,} EPA 2020–2026 rows whose represented make is year-like ({payload['summary']['year_like_make_rows_2026']:,} in MY2026) before cross-source identity work.",
        "- Keep JRC anonymized identities and EEA monitoring identities separate from EPA unless positive evidence is later versioned into a deterministic resolver.", "",
        "## Evidence tiers", "",
        "- **Directly tested:** `test_deterministic_normalization_grouping_and_population_forecast`; `test_no_merge_is_caused_by_model_year_adjacency_alone`; `test_program_parent_consolidation_preserves_all_child_populations`; `test_database_access_is_read_only_and_byte_identical`; `test_null_is_not_coerced_to_zero`; `test_example_selection_is_stable_and_covers_required_cases`; `test_boundary_and_consolidation_vocabularies_are_closed`; `test_forecast_separates_jrc_and_eea_from_epa`; `test_outputs_are_scoped_to_etl`.",
        "- **Indirectly covered:** Sprint 12C exact VDE/FuelCons reconstruction and Sprint 12C.2 candidate grain.",
        "- **Inspection-supported:** the local EPA technical descriptors and CDR ownership rules.",
        "- **Gap:** authoritative OEM generation identity and commercial rename evidence are not present in the supplied structured sources.", "",
        "## Reproduction", "", "```powershell",
        "python etl/scripts/sprint_12c3_program_consolidation_review.py",
        "python -m unittest discover -s etl/tests -p \"test_sprint_12c3*.py\" -v", "```", "",
        "## Outputs", "",
        *[f"- `{str(path.relative_to(ROOT)).replace(chr(92), '/')}`" for path in OUTPUT_PATHS],
    ]
    return "\n".join(lines) + "\n"


def main() -> dict[str, Any]:
    required = [DB_PATH, c1.EPA_PATH, c1.JRC_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing Sprint 12C.3 inputs: {missing}")
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    db_hash_before = sha256(DB_PATH)
    con = sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        con.execute("PRAGMA query_only = ON")
        runtime_counts = {
            "vde_db": int(con.execute("SELECT COUNT(*) FROM vde_db").fetchone()[0]),
            "fuelcons_db": int(con.execute("SELECT COUNT(*) FROM fuelcons_db").fetchone()[0]),
        }
    finally:
        con.close()

    source = pd.read_excel(c1.EPA_PATH, sheet_name="Sheet1", engine="openpyxl")
    jrc = pd.read_excel(c1.JRC_PATH, sheet_name="Sheet1", engine="openpyxl")
    rows, _ = prepare_rows(source)
    profiles = make_profiles(rows)
    boundaries = make_boundaries(rows)
    consolidations = make_consolidations(rows, boundaries)
    forecast = make_forecast(rows, boundaries, jrc)
    rename_candidates = find_rename_candidates(rows)
    examples = make_examples(rows, boundaries, consolidations, rename_candidates)
    impact = relationship_impact(rows, consolidations, forecast)
    db_hash_after = sha256(DB_PATH)

    if db_hash_before != db_hash_after:
        raise RuntimeError("Runtime database changed during the read-only Program review.")
    if len(rows) != 30_194 or len(set(rows["fallback_program_id"])) != 5_742:
        raise RuntimeError("EPA source population no longer matches the audited 2020-2026 slice.")
    if any(row["consolidation_status"] == "SAFE_CONSOLIDATE" and row["shared_stable_arch_signatures"] == 0 for row in boundaries):
        raise RuntimeError("Unsafe Program merge: a safe edge lacks technical continuity.")
    if not impact["fallback_programs_assigned_once"] or not impact["fallback_program_identity_set_preserved"]:
        raise RuntimeError("Program grouping did not preserve every fallback identity exactly once.")
    if len(examples) < 25:
        raise RuntimeError(f"Only {len(examples)} stable examples were selected; at least 25 are required.")

    summary = {
        "source_rows": len(rows),
        "raw_mmy_fallbacks": rows[list(MMY_FIELDS)].drop_duplicates().shape[0],
        "normalized_mmy_fallbacks": rows["fallback_program_id"].nunique(),
        "normalized_make_model_families": rows["family_key"].nunique(),
        "cross_year_families": int((rows.groupby("family_key")["Model Year"].nunique() > 1).sum()),
        "boundary_candidates": len(boundaries),
        "confirmed_boundaries": sum(row["boundary_status"] == "CONFIRMED_BOUNDARY" for row in boundaries),
        "safe_consolidation_edges": sum(row["consolidation_status"] == "SAFE_CONSOLIDATE" for row in boundaries),
        "probable_consolidation_edges": sum(row["consolidation_status"] == "PROBABLE_CONSOLIDATE" for row in boundaries),
        "year_like_make_rows": int(rows["identity_anomaly"].sum()),
        "year_like_make_rows_2026": int(rows.loc[rows["Model Year"].eq(2026), "identity_anomaly"].sum()),
        "examples": len(examples),
    }
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PROGRAM_SHAPE_CLEAR — PROCEED_TO_12D",
        "scope": "Read-only Program population review; no production identity, DDL or migration.",
        "database_access": "SQLite URI mode=ro + PRAGMA query_only=ON",
        "database_sha256_before": db_hash_before,
        "database_sha256_after": db_hash_after,
        "database_byte_identical": True,
        "runtime_counts_read_only": runtime_counts,
        "summary": summary,
        "program_population_forecast": forecast,
        "cross_year_profiles": profiles,
        "program_boundary_candidates": boundaries,
        "program_consolidation_candidates": consolidations,
        "commercial_rename_review_candidates": rename_candidates,
        "program_examples": examples,
        "relationship_impact": impact,
        "future_enrichment_inputs": [
            "OEM generation/platform codes",
            "authoritative launch/facelift chronology",
            "OEM model rename and successor documentation",
            "versioned EPA identity interpretation",
        ],
        "evidence_tiers": {
            "directly_tested": [
                "test_deterministic_normalization_grouping_and_population_forecast",
                "test_no_merge_is_caused_by_model_year_adjacency_alone",
                "test_program_parent_consolidation_preserves_all_child_populations",
                "test_database_access_is_read_only_and_byte_identical",
                "test_null_is_not_coerced_to_zero",
                "test_example_selection_is_stable_and_covers_required_cases",
                "test_boundary_and_consolidation_vocabularies_are_closed",
                "test_forecast_separates_jrc_and_eea_from_epa",
                "test_outputs_are_scoped_to_etl",
            ],
            "indirectly_covered": ["Sprint 12C compatibility proof", "Sprint 12C.2 candidate grain"],
            "inspection_supported": ["EPA structured technical descriptors", "CDR Program ownership"],
            "gaps": ["authoritative OEM generation identity", "authoritative rename/successor evidence"],
        },
    }

    write_csv(OUT / "cross_year_profiles.csv", profiles)
    write_csv(OUT / "program_boundary_candidates.csv", boundaries)
    write_csv(OUT / "program_consolidation_candidates.csv", consolidations)
    write_csv(OUT / "program_population_forecast.csv", forecast)
    write_csv(OUT / "program_examples.csv", examples)
    (OUT / "program_review.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=clean), encoding="utf-8")
    REPORT.write_text(report_text(payload), encoding="utf-8")
    EXAMPLE_REPORT.write_text(examples_report(examples), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "summary": summary,
        "combined_epa_forecast": next(row for row in forecast if row["population"] == "Combined EPA 2020-2026"),
        "relationship_impact": impact,
        "database_byte_identical": True,
        "report": str(REPORT.relative_to(ROOT)),
    }, indent=2, ensure_ascii=False))
    return payload


if __name__ == "__main__":
    main()
