from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator

from src.vde_core import db as db_module


REPO_ROOT = Path(__file__).resolve().parents[2]
QA_DATA_DIR = REPO_ROOT / "data" / "qa"
DEFAULT_QA_DB_PATH = QA_DATA_DIR / "eco_drive_qa.db"
INERTIA_TABLE_PATH = REPO_ROOT / "data" / "standards" / "inertia_classes_by_mass.csv"
SYNTHETIC_QA_WARNING = (
    "All values are synthetic engineering QA fixtures and must not be treated as production or manufacturer data."
)
FIXED_TIMESTAMP = "2026-07-16T00:00:00Z"
QA_KPA_PER_PSI = 6.89475729
QA_N_PER_KG = 9.80665

GOLDEN_QA_SCENARIO = {
    "baseline_id": 900001,
    "baseline_qa_id": "VDE-QA-001",
    "tire_id": 920010,
    "tire_qa_id": "TIRE-QA-010",
    "components": {
        "transmission": "TRANS-MOCK-001",
        "brake": "BRAKE-MOCK-001",
        "axle_hubs": "AXLE-MOCK-001",
        "parasitic": "PARA-MOCK-001",
    },
}

QA_COMPONENT_FIXTURES = {
    "transmission": [
        {"component_id": "TRANS-MOCK-001", "purpose": "Nominal transmission ABC fixture", "component_type": "TRANSMISSION", "net_bridge_eligible": "TRUE"},
        {"component_id": "TRANS-MOCK-002", "purpose": "Alternative transmission ABC fixture", "component_type": "TRANSMISSION", "net_bridge_eligible": "UNKNOWN"},
    ],
    "brake": [
        {"component_id": "BRAKE-MOCK-001", "purpose": "Nominal brake ABC fixture", "component_type": "BRAKE_STANDARD", "test_condition_type": "STANDARD"},
        {"component_id": "BRAKE-MOCK-002", "purpose": "Alternative brake ABC fixture", "component_type": "BRAKE_BASELINE_AS_RECEIVED", "test_condition_type": "AS_RECEIVED"},
    ],
    "axle_hubs": [
        {"component_id": "AXLE-MOCK-001", "purpose": "Nominal axle & hubs ABC fixture", "component_type": "AXLE", "component_position": "FRONT"},
        {"component_id": "AXLE-MOCK-002", "purpose": "Alternative axle & hubs ABC fixture", "component_type": "AXLE_HUB_COMBINED", "component_position": "COMBINED"},
    ],
    "parasitic": [
        {"component_id": "PARA-MOCK-001", "purpose": "Nominal parasitic ABC fixture", "component_type": "OTHER_RESIDUAL_COMPONENT_LOSSES"},
        {"component_id": "PARA-MOCK-002", "purpose": "Alternative parasitic ABC fixture", "component_type": "OTHER_RESIDUAL_COMPONENT_LOSSES"},
    ],
}

COMPLETE_SAE_QA_TIRE_SPECS = (
    {
        "row_id": 920101,
        "qa_id": "QA-BASE",
        "model_name": "TIRE-MOCK-001",
        "rr_n_per_kn": 8.0,
        "pressure_psi": 38.0,
        "load_kg": 610.0,
        "test_mileage_km": 1000.0,
        "test_speed_value": 80.0,
        "sae_alpha": -0.30,
        "sae_beta": 1.00,
        "sae_a": 0.0405987767,
        "sae_b": 0.00002000,
        "sae_c": 0.0000000500,
        "notes": "QA MOCK | TIRE-MOCK-001 | Synthetic QA data | Primary synthetic SAE baseline/reference tire.",
    },
    {
        "row_id": 920102,
        "qa_id": "QA-ECO",
        "model_name": "TIRE-MOCK-002",
        "rr_n_per_kn": 7.0,
        "pressure_psi": 35.0,
        "load_kg": 610.0,
        "test_mileage_km": 1000.0,
        "test_speed_value": 80.0,
        "sae_alpha": -0.32,
        "sae_beta": 1.00,
        "sae_a": 0.0388106091,
        "sae_b": 0.00001800,
        "sae_c": 0.0000000400,
        "notes": "QA MOCK | TIRE-MOCK-002 | Synthetic QA data | Lower-RRC synthetic SAE tire.",
    },
    {
        "row_id": 920103,
        "qa_id": "QA-HIGH-RRC",
        "model_name": "TIRE-MOCK-003",
        "rr_n_per_kn": 10.0,
        "pressure_psi": 32.0,
        "load_kg": 610.0,
        "test_mileage_km": 1000.0,
        "test_speed_value": 80.0,
        "sae_alpha": -0.25,
        "sae_beta": 1.02,
        "sae_a": 0.0299397091,
        "sae_b": 0.00002500,
        "sae_c": 0.0000000700,
        "notes": "QA MOCK | TIRE-MOCK-003 | Synthetic QA data | Higher-RRC synthetic SAE tire.",
    },
    {
        "row_id": 920104,
        "qa_id": "QA-LOAD",
        "model_name": "TIRE-MOCK-004",
        "rr_n_per_kn": 8.8,
        "pressure_psi": 30.0,
        "load_kg": 650.0,
        "test_mileage_km": 1000.0,
        "test_speed_value": 80.0,
        "sae_alpha": -0.28,
        "sae_beta": 1.05,
        "sae_a": 0.0231280363,
        "sae_b": 0.00002200,
        "sae_c": 0.0000000600,
        "notes": "QA MOCK | TIRE-MOCK-004 | Synthetic QA data | Load-sensitivity SAE tire for mass/inherit QA.",
    },
    {
        "row_id": 920105,
        "qa_id": "QA-NEUTRAL",
        "model_name": "TIRE-MOCK-005",
        "rr_n_per_kn": 8.0,
        "pressure_psi": 38.0,
        "load_kg": 610.0,
        "test_mileage_km": 1000.0,
        "test_speed_value": 80.0,
        "sae_alpha": -0.30,
        "sae_beta": 1.00,
        "sae_a": 0.0405987767,
        "sae_b": 0.00002000,
        "sae_c": 0.0000000500,
        "notes": "QA MOCK | TIRE-MOCK-005 | Synthetic QA data | Different identity with QA-BASE-equivalent engineering behavior.",
    },
    {
        "row_id": 920109,
        "qa_id": "QA-SAME-RRC-DIFF-SAE",
        "model_name": "TIRE-MOCK-SAME-RRC-DIFF-SAE",
        "rr_n_per_kn": 8.0,
        "pressure_psi": 38.0,
        "load_kg": 610.0,
        "test_mileage_km": 1000.0,
        "test_speed_value": 80.0,
        "sae_alpha": -0.24,
        "sae_beta": 1.03,
        "sae_a": 0.0220038491,
        "sae_b": 0.00001050,
        "sae_c": 0.0000000950,
        "notes": "QA MOCK | TIRE-MOCK-SAME-RRC-DIFF-SAE | Synthetic QA data | Same reference RRC as QA-BASE with deliberately different SAE speed curve.",
    },
    {
        "row_id": 920107,
        "qa_id": "QA-LOW-PRESSURE",
        "model_name": "TIRE-MOCK-006",
        "rr_n_per_kn": 9.2,
        "pressure_psi": 28.0,
        "load_kg": 610.0,
        "test_mileage_km": 1000.0,
        "test_speed_value": 80.0,
        "sae_alpha": -0.35,
        "sae_beta": 1.00,
        "sae_a": 0.0558223321,
        "sae_b": 0.00002300,
        "sae_c": 0.0000000600,
        "notes": "QA MOCK | TIRE-MOCK-006 | Synthetic QA data | Pressure-sensitivity low-pressure SAE tire.",
    },
    {
        "row_id": 920108,
        "qa_id": "QA-HIGH-PRESSURE",
        "model_name": "TIRE-MOCK-007",
        "rr_n_per_kn": 7.6,
        "pressure_psi": 42.0,
        "load_kg": 610.0,
        "test_mileage_km": 1000.0,
        "test_speed_value": 80.0,
        "sae_alpha": -0.27,
        "sae_beta": 0.99,
        "sae_a": 0.0364973585,
        "sae_b": 0.00001900,
        "sae_c": 0.0000000450,
        "notes": "QA MOCK | TIRE-MOCK-007 | Synthetic QA data | Pressure-sensitivity high-pressure SAE tire.",
    },
)


def load_inertia_table() -> list[dict[str, float | None]]:
    rows: list[dict[str, float | None]] = []
    with INERTIA_TABLE_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "mass_min_kg_exclusive": _to_float(raw.get("mass_min_kg_exclusive")),
                    "mass_max_kg_inclusive": _to_float(raw.get("mass_max_kg_inclusive")),
                    "inertia_class_kg": _required_float(raw.get("inertia_class_kg")),
                }
            )
    return rows


def inertia_row_for_mass(mass_kg: float | int | None) -> dict[str, float | None] | None:
    mass = _to_float(mass_kg)
    if mass is None:
        return None
    for row in load_inertia_table():
        lower = row.get("mass_min_kg_exclusive")
        upper = row.get("mass_max_kg_inclusive")
        if lower is None and upper is not None and mass <= upper:
            return deepcopy(row)
        if lower is not None and upper is None and mass > lower:
            return deepcopy(row)
        if lower is not None and upper is not None and mass > lower and mass <= upper:
            return deepcopy(row)
    return None


def inertia_row_for_class(inertia_class_kg: float | int | None) -> dict[str, float | None] | None:
    target = _to_float(inertia_class_kg)
    if target is None:
        return None
    for row in load_inertia_table():
        if float(row["inertia_class_kg"]) == float(target):
            return deepcopy(row)
    return None


def build_vde_seed_rows() -> list[dict[str, Any]]:
    nominal_mass = 1500.0
    nominal_class = _required_inertia_class_for_mass(nominal_mass)
    lower_boundary_mass = 1480.0
    lower_boundary_class = _required_inertia_class_for_mass(lower_boundary_mass)
    upper_boundary_mass = 1536.0
    upper_boundary_class = _required_inertia_class_for_mass(upper_boundary_mass)
    higher_mass = 1800.0
    higher_class = _required_inertia_class_for_mass(higher_mass)
    lower_mass = 1400.0
    lower_mass_class = _required_inertia_class_for_mass(lower_mass)
    optional_mass = 1570.0
    optional_class = _required_inertia_class_for_mass(optional_mass)
    correction_mass = 1500.0

    return [
        _base_vde_row(
            900001,
            "VDE-QA-001",
            "Nominal EPA baseline",
            mass_kg=nominal_mass,
            inertia_class=nominal_class,
            test_mass_kg=nominal_class,
            rrc_n_per_kn=8.0,
            front_pressure_psi=38.0,
            rear_pressure_psi=38.0,
            cda_m2=0.620,
            coast_a_n=118.0,
            coast_b_n_per_kph=0.0200,
            coast_c_n_per_kph2=0.0090,
            trans_a=8.5,
            trans_b=0.0040,
            trans_c=0.0008,
            brake_a=4.0,
            brake_b=0.0008,
            brake_c=0.0001,
            parasitic_a=3.0,
            parasitic_b=0.0005,
            parasitic_c=0.0001,
            axle_a=2.5,
            axle_b=0.0004,
            axle_c=0.0001,
            vde_total=1.240,
            vde_net=1.180,
            notes="QA MOCK | VDE-QA-001 | Nominal EPA baseline for deterministic request QA.",
            front_tire_id=920101,
            rear_tire_id=920101,
            tire_code="QA-BASE",
            tire_a_final=None,
            tire_b_final=None,
            tire_c_final=None,
        ),
        _base_vde_row(
            900002,
            "VDE-QA-002",
            "TWC boundary lower",
            mass_kg=lower_boundary_mass,
            inertia_class=lower_boundary_class,
            test_mass_kg=lower_boundary_class,
            rrc_n_per_kn=8.0,
            front_pressure_psi=35.0,
            rear_pressure_psi=35.0,
            cda_m2=0.615,
            coast_a_n=116.0,
            coast_b_n_per_kph=0.0190,
            coast_c_n_per_kph2=0.0088,
            trans_a=8.4,
            trans_b=0.0039,
            trans_c=0.0008,
            brake_a=4.0,
            brake_b=0.0008,
            brake_c=0.0001,
            parasitic_a=3.0,
            parasitic_b=0.0005,
            parasitic_c=0.0001,
            axle_a=2.5,
            axle_b=0.0004,
            axle_c=0.0001,
            vde_total=1.215,
            vde_net=1.160,
            notes="QA MOCK | VDE-QA-002 | Curb mass sits at the lower exclusive boundary for the next EPA class.",
            front_tire_id=920001,
            rear_tire_id=920001,
            tire_code="TIRE-QA-001",
        ),
        _base_vde_row(
            900003,
            "VDE-QA-003",
            "TWC boundary upper",
            mass_kg=upper_boundary_mass,
            inertia_class=upper_boundary_class,
            test_mass_kg=upper_boundary_class,
            rrc_n_per_kn=8.5,
            front_pressure_psi=35.0,
            rear_pressure_psi=35.0,
            cda_m2=0.618,
            coast_a_n=119.0,
            coast_b_n_per_kph=0.0205,
            coast_c_n_per_kph2=0.0091,
            trans_a=8.6,
            trans_b=0.0040,
            trans_c=0.0008,
            brake_a=4.1,
            brake_b=0.0008,
            brake_c=0.0001,
            parasitic_a=3.1,
            parasitic_b=0.0005,
            parasitic_c=0.0001,
            axle_a=2.6,
            axle_b=0.0004,
            axle_c=0.0001,
            vde_total=1.248,
            vde_net=1.188,
            notes="QA MOCK | VDE-QA-003 | Curb mass sits exactly at an EPA upper inclusive boundary.",
            front_tire_id=920005,
            rear_tire_id=920005,
            tire_code="TIRE-QA-005",
        ),
        _base_vde_row(
            900004,
            "VDE-QA-004",
            "Higher mass baseline",
            mass_kg=higher_mass,
            inertia_class=higher_class,
            test_mass_kg=higher_class,
            rrc_n_per_kn=9.0,
            front_pressure_psi=32.0,
            rear_pressure_psi=32.0,
            cda_m2=0.655,
            coast_a_n=132.0,
            coast_b_n_per_kph=0.0230,
            coast_c_n_per_kph2=0.0105,
            trans_a=9.0,
            trans_b=0.0044,
            trans_c=0.0009,
            brake_a=4.5,
            brake_b=0.0010,
            brake_c=0.0001,
            parasitic_a=3.4,
            parasitic_b=0.0006,
            parasitic_c=0.0001,
            axle_a=2.8,
            axle_b=0.0005,
            axle_c=0.0001,
            vde_total=1.360,
            vde_net=1.290,
            notes="QA MOCK | VDE-QA-004 | Heavier baseline for negative TWC shift scenarios.",
            front_tire_id=920003,
            rear_tire_id=920003,
            tire_code="TIRE-QA-003",
        ),
        _base_vde_row(
            900005,
            "VDE-QA-005",
            "Lower mass baseline",
            mass_kg=lower_mass,
            inertia_class=lower_mass_class,
            test_mass_kg=lower_mass_class,
            rrc_n_per_kn=7.5,
            front_pressure_psi=38.0,
            rear_pressure_psi=38.0,
            cda_m2=0.590,
            coast_a_n=109.0,
            coast_b_n_per_kph=0.0180,
            coast_c_n_per_kph2=0.0082,
            trans_a=8.0,
            trans_b=0.0038,
            trans_c=0.0007,
            brake_a=3.8,
            brake_b=0.0007,
            brake_c=0.0001,
            parasitic_a=2.9,
            parasitic_b=0.0005,
            parasitic_c=0.0001,
            axle_a=2.4,
            axle_b=0.0004,
            axle_c=0.0001,
            vde_total=1.150,
            vde_net=1.095,
            notes="QA MOCK | VDE-QA-005 | Lighter baseline for positive TWC shift scenarios.",
            front_tire_id=920002,
            rear_tire_id=920002,
            tire_code="TIRE-QA-002",
        ),
        _base_vde_row(
            900006,
            "VDE-QA-006",
            "Missing optional fields",
            mass_kg=optional_mass,
            inertia_class=optional_class,
            test_mass_kg=optional_class,
            rrc_n_per_kn=8.2,
            front_pressure_psi=32.0,
            rear_pressure_psi=35.0,
            cda_m2=0.612,
            coast_a_n=120.0,
            coast_b_n_per_kph=0.0200,
            coast_c_n_per_kph2=0.0092,
            trans_a=None,
            trans_b=None,
            trans_c=None,
            brake_a=None,
            brake_b=None,
            brake_c=None,
            parasitic_a=None,
            parasitic_b=None,
            parasitic_c=None,
            axle_a=None,
            axle_b=None,
            axle_c=None,
            vde_total=1.255,
            vde_net=None,
            notes="QA MOCK | VDE-QA-006 | Optional component fields intentionally missing while baseline remains usable.",
            front_tire_id=920004,
            rear_tire_id=920004,
            tire_code="TIRE-QA-004",
            include_optional_fields=False,
        ),
        _base_vde_row(
            900007,
            "VDE-QA-007",
            "Correction review baseline",
            mass_kg=correction_mass,
            inertia_class=1814.0,
            test_mass_kg=1814.0,
            rrc_n_per_kn=8.8,
            front_pressure_psi=30.0,
            rear_pressure_psi=38.0,
            cda_m2=0.625,
            coast_a_n=121.0,
            coast_b_n_per_kph=0.0205,
            coast_c_n_per_kph2=0.0094,
            trans_a=8.7,
            trans_b=0.0041,
            trans_c=0.0008,
            brake_a=4.2,
            brake_b=0.0009,
            brake_c=0.0001,
            parasitic_a=3.2,
            parasitic_b=0.0005,
            parasitic_c=0.0001,
            axle_a=2.6,
            axle_b=0.0004,
            axle_c=0.0001,
            vde_total=1.270,
            vde_net=1.205,
            notes="QA MOCK | VDE-QA-007 | Mass and inertia_class are intentionally inconsistent for Baseline Correction / Review exercises.",
            front_tire_id=920007,
            rear_tire_id=920007,
            tire_code="TIRE-QA-007",
        ),
    ]


def build_tire_seed_rows() -> list[dict[str, Any]]:
    return [
        *[_complete_sae_qa_tire_row(spec) for spec in COMPLETE_SAE_QA_TIRE_SPECS],
        _base_tire_row(
            920106,
            "QA-INCOMPLETE",
            "TIRE-MOCK-INCOMPLETE",
            rr_n_per_kn=9.0,
            standard_family="CUSTOM",
            pressure_psi=None,
            load_kg=610.0,
            include_reference_fields=False,
            test_mass_kg=610.0,
            test_mileage_km=1000.0,
            rr_quality="incomplete_reference_inputs",
            notes="QA MOCK | TIRE-MOCK-INCOMPLETE | Deterministic incomplete lookup reference for negative QA.",
        ),
        _base_tire_row(
            920001,
            "TIRE-QA-001",
            "Nominal complete",
            rr_n_per_kn=8.0,
            standard_family="ISO",
            pressure_psi=38.0,
            load_kg=580.0,
            notes="QA MOCK | TIRE-QA-001 | Nominal complete tire reference.",
        ),
        _base_tire_row(
            920002,
            "TIRE-QA-002",
            "Low RRC",
            rr_n_per_kn=7.0,
            standard_family="ISO",
            pressure_psi=35.0,
            load_kg=560.0,
            notes="QA MOCK | TIRE-QA-002 | Lower-RRC reference for target/improvement paths.",
        ),
        _base_tire_row(
            920003,
            "TIRE-QA-003",
            "High RRC",
            rr_n_per_kn=10.0,
            standard_family="ISO",
            pressure_psi=32.0,
            load_kg=620.0,
            notes="QA MOCK | TIRE-QA-003 | Higher-RRC reference.",
        ),
        _base_tire_row(
            920004,
            "TIRE-QA-004",
            "RRC only",
            rr_n_per_kn=8.5,
            standard_family="CUSTOM",
            pressure_psi=None,
            load_kg=None,
            include_reference_fields=False,
            notes="QA MOCK | TIRE-QA-004 | Direct target RRC without full pressure reference.",
        ),
        _base_tire_row(
            920005,
            "TIRE-QA-005",
            "RRC with pressure reference",
            rr_n_per_kn=9.0,
            standard_family="ISO",
            pressure_psi=35.0,
            load_kg=600.0,
            notes="QA MOCK | TIRE-QA-005 | Pressure-adjustment reference tire.",
        ),
        _base_tire_row(
            920006,
            "TIRE-QA-006",
            "Split axle companion",
            rr_n_per_kn=8.2,
            standard_family="ISO",
            pressure_psi=32.0,
            load_kg=590.0,
            notes=(
                "QA MOCK | TIRE-QA-006 | Use with VDE-QA-006 split front/rear baseline pressures. "
                "The current tire schema stores one reference pressure only."
            ),
        ),
        _base_tire_row(
            920007,
            "TIRE-QA-007",
            "Known test mass",
            rr_n_per_kn=8.8,
            standard_family="SAE",
            pressure_psi=30.0,
            load_kg=650.0,
            test_mass_kg=650.0,
            notes="QA MOCK | TIRE-QA-007 | Known test load fixture for load-sensitive scenarios.",
        ),
        _base_tire_row(
            920008,
            "TIRE-QA-008",
            "Incomplete review fixture",
            rr_n_per_kn=9.4,
            standard_family="CUSTOM",
            pressure_psi=None,
            load_kg=None,
            include_reference_fields=False,
            rr_quality="incomplete_reference_inputs",
            notes="QA MOCK | TIRE-QA-008 | Intentionally incomplete to trigger current Review/Missing behavior.",
        ),
        _base_tire_row(
            920009,
            "TIRE-QA-009",
            "Boundary zero fixture",
            rr_n_per_kn=9.0,
            standard_family="ISO",
            pressure_psi=30.0,
            load_kg=580.0,
            test_mileage_km=0.0,
            break_in_distance_km=0.0,
            notes="QA MOCK | TIRE-QA-009 | Legal zero-valued optional fields for blank-vs-zero checks.",
        ),
        _base_tire_row(
            920010,
            "TIRE-QA-010",
            "Golden reference",
            rr_n_per_kn=7.8,
            standard_family="SAE",
            pressure_psi=38.0,
            load_kg=610.0,
            test_mass_kg=610.0,
            rr_quality="reference_rr_and_smerf_input",
            smerf=6.9,
            notes="QA MOCK | TIRE-QA-010 | Most complete deterministic golden tire reference.",
            include_sae_coefficients=True,
        ),
    ]


def build_seed_manifest() -> dict[str, Any]:
    return {
        "version": "QA-1",
        "warning": SYNTHETIC_QA_WARNING,
        "database_path_env_var": db_module.DB_PATH_ENV_VAR,
        "default_demo_db_path": str(DEFAULT_QA_DB_PATH),
        "golden_scenario": deepcopy(GOLDEN_QA_SCENARIO),
        "baselines": [
            {
                "qa_id": "VDE-QA-001",
                "vde_id": 900001,
                "purpose": "Nominal EPA baseline",
                "expected_use": "Primary baseline for deterministic request QA.",
                "canonical_inputs": {"mass_kg": 1500.0, "inertia_class": 1644.0, "front_pressure_psi": 38.0, "rear_pressure_psi": 38.0},
                "expected_missing_fields": [],
                "notes": "Golden baseline.",
            },
            {
                "qa_id": "VDE-QA-002",
                "vde_id": 900002,
                "purpose": "Boundary lower exclusive",
                "expected_use": "Verify EPA class selection at the lower exclusive boundary.",
                "canonical_inputs": {"mass_kg": 1480.0},
                "expected_missing_fields": [],
                "notes": "Should remain in the prior class.",
            },
            {
                "qa_id": "VDE-QA-003",
                "vde_id": 900003,
                "purpose": "Boundary upper inclusive",
                "expected_use": "Verify EPA class selection at the upper inclusive boundary.",
                "canonical_inputs": {"mass_kg": 1536.0},
                "expected_missing_fields": [],
                "notes": "Should resolve into the current class.",
            },
            {
                "qa_id": "VDE-QA-004",
                "vde_id": 900004,
                "purpose": "Higher mass baseline",
                "expected_use": "Used for negative TWC shift validation.",
                "canonical_inputs": {"mass_kg": 1800.0},
                "expected_missing_fields": [],
                "notes": "Heavier curb mass.",
            },
            {
                "qa_id": "VDE-QA-005",
                "vde_id": 900005,
                "purpose": "Lower mass baseline",
                "expected_use": "Used for positive TWC shift validation.",
                "canonical_inputs": {"mass_kg": 1400.0},
                "expected_missing_fields": [],
                "notes": "Lighter curb mass.",
            },
            {
                "qa_id": "VDE-QA-006",
                "vde_id": 900006,
                "purpose": "Missing optional fields",
                "expected_use": "Exercise missing optional component fields without blocking load.",
                "canonical_inputs": {"mass_kg": 1570.0, "front_pressure_psi": 32.0, "rear_pressure_psi": 35.0},
                "expected_missing_fields": ["trans_A_coef_N", "brake_A_coef_N", "parasitic_A_coef_N", "axle_hub_A", "vde_net_mj_per_km"],
                "notes": "Useful for split-axle pressure scenarios.",
            },
            {
                "qa_id": "VDE-QA-007",
                "vde_id": 900007,
                "purpose": "Baseline requiring correction",
                "expected_use": "Exercise Baseline Correction and Review without changing resolver semantics.",
                "canonical_inputs": {"mass_kg": 1500.0, "inertia_class": 1814.0},
                "expected_missing_fields": [],
                "notes": "Intentional mass/TWC mismatch.",
            },
        ],
        "tires": [
            {
                "qa_id": "QA-BASE",
                "tire_id": 920101,
                "purpose": "Synthetic SAE baseline/reference lookup",
                "expected_behavior": "Visible immediately in Tire Database browse and suitable for full SAE QA.",
                "canonical_inputs": {"rr_n_per_kn": 8.0, "test_pressure_value": 38.0, "test_load_value": 610.0, "test_speed_value": 80.0},
                "expected_missing_fields": [],
                "notes": "Synthetic QA data.",
            },
            {
                "qa_id": "QA-ECO",
                "tire_id": 920102,
                "purpose": "Synthetic low-RRC SAE lookup",
                "expected_behavior": "Should improve roadload relative to QA-BASE.",
                "canonical_inputs": {"rr_n_per_kn": 7.0, "test_pressure_value": 35.0, "test_load_value": 610.0, "test_speed_value": 80.0},
                "expected_missing_fields": [],
                "notes": "Synthetic QA data.",
            },
            {
                "qa_id": "QA-HIGH-RRC",
                "tire_id": 920103,
                "purpose": "Synthetic high-RRC SAE lookup",
                "expected_behavior": "Should worsen roadload relative to QA-BASE.",
                "canonical_inputs": {"rr_n_per_kn": 10.0, "test_pressure_value": 32.0, "test_load_value": 610.0, "test_speed_value": 80.0},
                "expected_missing_fields": [],
                "notes": "Synthetic QA data.",
            },
            {
                "qa_id": "QA-LOAD",
                "tire_id": 920104,
                "purpose": "Synthetic load-sensitive SAE lookup",
                "expected_behavior": "Exercises load-sensitive tire calculations.",
                "canonical_inputs": {"rr_n_per_kn": 8.8, "test_pressure_value": 30.0, "test_load_value": 650.0, "test_speed_value": 80.0},
                "expected_missing_fields": [],
                "notes": "Synthetic QA data.",
            },
            {
                "qa_id": "QA-NEUTRAL",
                "tire_id": 920105,
                "purpose": "Synthetic neutral-equivalent SAE lookup",
                "expected_behavior": "Provides a distinct lookup identity with neutral nominal values.",
                "canonical_inputs": {"rr_n_per_kn": 8.0, "test_pressure_value": 38.0, "test_load_value": 610.0, "test_speed_value": 80.0},
                "expected_missing_fields": [],
                "notes": "Synthetic QA data.",
            },
            {
                "qa_id": "QA-SAME-RRC-DIFF-SAE",
                "tire_id": 920109,
                "purpose": "Synthetic same-RRC but different-SAE lookup",
                "expected_behavior": "Matches QA-BASE at the reference point but differs across the SAE speed curve.",
                "canonical_inputs": {"rr_n_per_kn": 8.0, "test_pressure_value": 38.0, "test_load_value": 610.0, "test_speed_value": 80.0},
                "expected_missing_fields": [],
                "notes": "Synthetic QA data.",
            },
            {
                "qa_id": "QA-LOW-PRESSURE",
                "tire_id": 920107,
                "purpose": "Synthetic low-pressure SAE lookup",
                "expected_behavior": "Supports pressure sensitivity checks at lower reference pressure.",
                "canonical_inputs": {"rr_n_per_kn": 9.2, "test_pressure_value": 28.0, "test_load_value": 610.0, "test_speed_value": 80.0},
                "expected_missing_fields": [],
                "notes": "Synthetic QA data.",
            },
            {
                "qa_id": "QA-HIGH-PRESSURE",
                "tire_id": 920108,
                "purpose": "Synthetic high-pressure SAE lookup",
                "expected_behavior": "Supports pressure sensitivity checks at higher reference pressure.",
                "canonical_inputs": {"rr_n_per_kn": 7.6, "test_pressure_value": 42.0, "test_load_value": 610.0, "test_speed_value": 80.0},
                "expected_missing_fields": [],
                "notes": "Synthetic QA data.",
            },
            {
                "qa_id": "QA-INCOMPLETE",
                "tire_id": 920106,
                "purpose": "Deterministic incomplete lookup",
                "expected_behavior": "Must remain selectable and surface current validation behavior.",
                "canonical_inputs": {"rr_n_per_kn": 9.0, "test_load_value": 610.0},
                "expected_missing_fields": ["test_pressure_value", "sae_reference_pressure_kpa"],
                "notes": "",
            },
            {
                "qa_id": "TIRE-QA-001",
                "tire_id": 920001,
                "purpose": "Nominal complete",
                "expected_behavior": "Lookup-friendly nominal reference.",
                "canonical_inputs": {"rr_n_per_kn": 8.0, "test_pressure_value": 38.0},
                "expected_missing_fields": [],
                "notes": "Main nominal tire.",
            },
            {
                "qa_id": "TIRE-QA-002",
                "tire_id": 920002,
                "purpose": "Low RRC",
                "expected_behavior": "Lower-RRC reference for target and improvement.",
                "canonical_inputs": {"rr_n_per_kn": 7.0, "test_pressure_value": 35.0},
                "expected_missing_fields": [],
                "notes": "",
            },
            {
                "qa_id": "TIRE-QA-003",
                "tire_id": 920003,
                "purpose": "High RRC",
                "expected_behavior": "Higher-RRC reference.",
                "canonical_inputs": {"rr_n_per_kn": 10.0, "test_pressure_value": 32.0},
                "expected_missing_fields": [],
                "notes": "",
            },
            {
                "qa_id": "TIRE-QA-004",
                "tire_id": 920004,
                "purpose": "RRC only",
                "expected_behavior": "Direct target RRC path remains valid without pressure reference.",
                "canonical_inputs": {"rr_n_per_kn": 8.5},
                "expected_missing_fields": ["test_pressure_value", "sae_reference_pressure_kpa", "test_load_value", "sae_reference_load_n"],
                "notes": "",
            },
            {
                "qa_id": "TIRE-QA-005",
                "tire_id": 920005,
                "purpose": "RRC + pressure reference",
                "expected_behavior": "Pressure-adjustment path has reference pressure available.",
                "canonical_inputs": {"rr_n_per_kn": 9.0, "test_pressure_value": 35.0},
                "expected_missing_fields": [],
                "notes": "",
            },
            {
                "qa_id": "TIRE-QA-006",
                "tire_id": 920006,
                "purpose": "Split axle companion",
                "expected_behavior": "Companion fixture for split-axle baseline pressure scenarios.",
                "canonical_inputs": {"rr_n_per_kn": 8.2, "test_pressure_value": 32.0},
                "expected_missing_fields": [],
                "notes": "Current tire schema stores a single reference pressure only.",
            },
            {
                "qa_id": "TIRE-QA-007",
                "tire_id": 920007,
                "purpose": "Known test mass",
                "expected_behavior": "Supports load-sensitive QA scenarios.",
                "canonical_inputs": {"rr_n_per_kn": 8.8, "test_load_value": 650.0},
                "expected_missing_fields": [],
                "notes": "",
            },
            {
                "qa_id": "TIRE-QA-008",
                "tire_id": 920008,
                "purpose": "Incomplete / Review case",
                "expected_behavior": "Can be selected but should preserve current Review/Missing behavior.",
                "canonical_inputs": {"rr_n_per_kn": 9.4},
                "expected_missing_fields": ["test_pressure_value", "sae_reference_pressure_kpa", "test_load_value", "sae_reference_load_n"],
                "notes": "",
            },
            {
                "qa_id": "TIRE-QA-009",
                "tire_id": 920009,
                "purpose": "Boundary / zero fixture",
                "expected_behavior": "Preserves legal zero-valued optional fields.",
                "canonical_inputs": {"rr_n_per_kn": 9.0, "test_mileage_km": 0.0, "break_in_distance_km": 0.0},
                "expected_missing_fields": [],
                "notes": "",
            },
            {
                "qa_id": "TIRE-QA-010",
                "tire_id": 920010,
                "purpose": "Golden reference",
                "expected_behavior": "Most complete deterministic tire reference for future QA packs.",
                "canonical_inputs": {"rr_n_per_kn": 7.8, "test_pressure_value": 38.0, "test_load_value": 610.0, "smerf": 6.9},
                "expected_missing_fields": [],
                "notes": "Golden tire.",
            },
        ],
        "components": deepcopy(QA_COMPONENT_FIXTURES),
    }


def seed_qa_database(db_path: str | Path | None = None, *, overwrite: bool = False) -> Path:
    target = Path(db_path or DEFAULT_QA_DB_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"Target QA database already exists: {target}")
        if not is_safe_qa_db_path(target):
            raise ValueError(f"Refusing to overwrite a non-QA database path: {target}")

    with db_module.using_db_path(target):
        db_module.ensure_db()
        _replace_seed_rows(target)
    return target


def is_safe_qa_db_path(path: str | Path) -> bool:
    target = Path(path).expanduser()
    try:
        target_resolved = target.resolve()
        qa_root_resolved = QA_DATA_DIR.resolve()
    except FileNotFoundError:
        target_resolved = target.absolute()
        qa_root_resolved = QA_DATA_DIR.absolute()
    return target_resolved == qa_root_resolved or qa_root_resolved in target_resolved.parents


def seeded_database_digest(db_path: str | Path) -> str:
    table_names = ("vde_db", "tire_roadload_db")
    payload: dict[str, list[dict[str, Any]]] = {}
    with sqlite3.connect(str(db_path)) as con:
        con.row_factory = sqlite3.Row
        for table_name in table_names:
            rows = con.execute(f"SELECT * FROM {table_name} ORDER BY id ASC").fetchall()
            payload[table_name] = [dict(row) for row in rows]
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def manifest_json(indent: int = 2) -> str:
    return json.dumps(build_seed_manifest(), indent=indent, sort_keys=True, ensure_ascii=True)


@contextmanager
def temporary_seeded_db(db_path: str | Path) -> Iterator[Path]:
    target = Path(db_path)
    seed_qa_database(target, overwrite=True)
    with db_module.using_db_path(target):
        yield target


def _replace_seed_rows(db_path: Path) -> None:
    with sqlite3.connect(str(db_path), timeout=30) as con:
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("DELETE FROM fuelcons_db")
        con.execute("DELETE FROM vde_db")
        con.execute("DELETE FROM tire_roadload_db")
        con.execute("DELETE FROM sqlite_sequence WHERE name IN ('vde_db', 'fuelcons_db', 'tire_roadload_db')")
        _insert_rows(con, "tire_roadload_db", build_tire_seed_rows())
        _insert_rows(con, "vde_db", build_vde_seed_rows())
        con.commit()


def _insert_rows(con: sqlite3.Connection, table_name: str, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        columns = list(row.keys())
        placeholders = ", ".join(["?"] * len(columns))
        sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
        con.execute(sql, [row[column] for column in columns])


def _base_vde_row(
    row_id: int,
    qa_id: str,
    model_name: str,
    *,
    mass_kg: float,
    inertia_class: float,
    test_mass_kg: float,
    rrc_n_per_kn: float,
    front_pressure_psi: float,
    rear_pressure_psi: float,
    cda_m2: float,
    coast_a_n: float,
    coast_b_n_per_kph: float,
    coast_c_n_per_kph2: float,
    trans_a: float | None,
    trans_b: float | None,
    trans_c: float | None,
    brake_a: float | None,
    brake_b: float | None,
    brake_c: float | None,
    parasitic_a: float | None,
    parasitic_b: float | None,
    parasitic_c: float | None,
    axle_a: float | None,
    axle_b: float | None,
    axle_c: float | None,
    vde_total: float,
    vde_net: float | None,
    notes: str,
    front_tire_id: int,
    rear_tire_id: int,
    tire_code: str,
    tire_a_final: float | None = 5.0,
    tire_b_final: float | None = 0.0008,
    tire_c_final: float | None = 0.0002,
    include_optional_fields: bool = True,
) -> dict[str, Any]:
    row = {
        "id": int(row_id),
        "created_at": FIXED_TIMESTAMP,
        "updated_at": FIXED_TIMESTAMP,
        "legislation": "EPA",
        "category": "QA_EPA",
        "make": "QA",
        "model": model_name,
        "year": 2026,
        "notes": notes,
        "engine_type": "BEV",
        "engine_model": "QA Powertrain",
        "engine_size_l": None,
        "engine_aspiration": "NA",
        "transmission_type": "AT",
        "transmission_model": "QA-TRANS",
        "mass_kg": float(mass_kg),
        "test_mass_kg": float(test_mass_kg),
        "test_mass_basis": "EPA_INERTIA_CLASS",
        "inertia_class": float(inertia_class),
        "cda_m2": float(cda_m2),
        "weight_dist_fr_pct": 55.0,
        "payload_kg": 300.0,
        "tire_size": "QA-235/50R18",
        "tire_rr_note": qa_id,
        "front_pressure_psi": float(front_pressure_psi),
        "rear_pressure_psi": float(rear_pressure_psi),
        "coast_A_N": float(coast_a_n),
        "coast_B_N_per_kph": float(coast_b_n_per_kph),
        "coast_C_N_per_kph2": float(coast_c_n_per_kph2),
        "trans_A_coef_N": trans_a,
        "trans_B_coef_Npkph": trans_b,
        "trans_C_coef_Npkph2": trans_c,
        "brake_A_coef_N": brake_a,
        "brake_B_coef_Npkph": brake_b,
        "brake_C_coef_Npkph2": brake_c,
        "cycle_name": "FTP-75",
        "cycle_source": "qa_mock_seed",
        "vde_net_mj_per_km": vde_net,
        "vde_total_mj_per_km": float(vde_total),
        "drive_type": "FWD",
        "options_kg": 0.0 if include_optional_fields else None,
        "rrc_N_per_kN": float(rrc_n_per_kn),
        "front_tire_id": int(front_tire_id),
        "rear_tire_id": int(rear_tire_id),
        "tire_improvement_pct": None,
        "tire_load_mass_basis": "TEST_MASS",
        "tire_load_mass_used_kg": float(test_mass_kg),
        "tire_A_final": tire_a_final,
        "tire_B_final": tire_b_final,
        "tire_C_final": tire_c_final,
        "tire_calc_source": f"qa_mock_seed:{qa_id}",
        "tire_calc_notes": f"Synthetic QA seed linked to {tire_code}",
        "parasitic_A_coef_N": parasitic_a,
        "parasitic_B_coef_Npkph": parasitic_b,
        "parasitic_C_coef_Npkph2": parasitic_c,
        "smerf": float(rrc_n_per_kn),
    }
    if include_optional_fields:
        row.update(
            {
                "gvwr_kg": float(mass_kg + 500.0),
                "gcwr_kg": float(mass_kg + 900.0),
                "trailer_mass_kg": 0.0,
                "trailer_code": f"{qa_id}-TRAILER",
                "trailer_roadload_source": "qa_mock_seed",
                "trailer_A_coef_N": 0.0,
                "trailer_B_coef_Npkph": 0.0,
                "trailer_C_coef_Npkph2": 0.0,
                "mass_rule_status": "OK",
                "mass_rule_notes": f"{qa_id} synthetic baseline",
                "vde_urb_mj_per_km": round(float(vde_total) * 0.95, 3),
                "vde_hw_mj_per_km": round(float(vde_total) * 1.05, 3),
            }
        )
    return row


def _base_tire_row(
    row_id: int,
    qa_id: str,
    model_name: str,
    *,
    rr_n_per_kn: float,
    standard_family: str,
    pressure_psi: float | None,
    load_kg: float | None,
    notes: str,
    include_reference_fields: bool = True,
    include_sae_coefficients: bool = False,
    rr_quality: str | None = None,
    smerf: float | None = None,
    test_mass_kg: float | None = None,
    test_mileage_km: float | None = 1200.0,
    break_in_distance_km: float | None = 100.0,
    test_speed_value: float | None = None,
    sae_a: float | None = None,
    sae_b: float | None = None,
    sae_c: float | None = None,
    sae_alpha: float | None = None,
    sae_beta: float | None = None,
) -> dict[str, Any]:
    calculation_mode = {
        "ISO": "ISO_28580",
        "SAE": "SAE_J2452",
    }.get(str(standard_family).upper(), "CUSTOM")
    row = {
        "id": int(row_id),
        "created_at": FIXED_TIMESTAMP,
        "updated_at": FIXED_TIMESTAMP,
        "tire_test_code": qa_id,
        "manufacturer": "QA MOCK",
        "model": model_name,
        "size_code": "QA-235/50R18",
        "load_index": "100",
        "speed_rating": "Q",
        "is_active": 1,
        "notes": notes,
        "standard_family": standard_family,
        "standard_version": "QA-0",
        "test_method": calculation_mode,
        "test_source": "qa_mock_seed",
        "test_date": "2026-07-16",
        "test_mileage_km": test_mileage_km,
        "is_tested_value": 1,
        "is_estimated_value": 0,
        "break_in_distance_km": break_in_distance_km,
        "is_broken_in": 1,
        "test_temperature_c": 25.0,
        "reference_temperature_c": 25.0,
        "temperature_correction_applied": 0,
        "rr_n_per_kn": float(rr_n_per_kn),
        "rr_value_source_note": qa_id,
        "pressure_unit": "psi",
        "load_unit": "kg",
        "speed_unit": "kph",
        "force_unit": "N",
        "calculation_mode": calculation_mode,
        "sae_pressure_unit": "kPa",
        "sae_load_unit": "N",
        "sae_speed_unit": "kph",
        "sae_force_unit": "N",
        "rr_method": "QA_MOCK_REFERENCE",
        "rr_source": "qa_mock_seed",
        "rr_quality": rr_quality or ("reference_rr_and_smerf_input" if smerf is not None else "measured_or_corrected_iso"),
        "conditioning_notes": qa_id,
        "smerf": smerf,
    }
    if test_speed_value is not None:
        row["test_speed_value"] = float(test_speed_value)
    if include_reference_fields and pressure_psi is not None:
        row.update(
            {
                "test_pressure_value": float(pressure_psi),
                "sae_reference_pressure_kpa": round(float(pressure_psi) * QA_KPA_PER_PSI, 4),
            }
        )
    if include_reference_fields and load_kg is not None:
        row.update(
            {
                "test_load_value": float(load_kg),
                "sae_reference_load_n": round(float(load_kg) * QA_N_PER_KG, 4),
            }
        )
    if str(standard_family).upper() == "ISO":
        row["iso_rrc_n_per_kn"] = float(rr_n_per_kn)
        if include_reference_fields and pressure_psi is not None:
            row["iso_test_pressure_kpa"] = round(float(pressure_psi) * 6.89476, 4)
        if include_reference_fields and load_kg is not None:
            row["iso_test_load_n"] = round(float(load_kg) * 9.80665, 4)
        row["iso_test_speed_kph"] = 80.0
        if include_reference_fields and load_kg is not None:
            row["iso_rolling_resistance_force_n"] = round(float(rr_n_per_kn) * float(load_kg) * 9.80665 / 1000.0, 4)
        row["iso_corrected_rrc_n_per_kn"] = float(rr_n_per_kn)
        row["iso_condition_notes"] = qa_id
    if include_sae_coefficients:
        row.update(
            {
                "sae_a": 0.0420 if sae_a is None else float(sae_a),
                "sae_b": 0.00015 if sae_b is None else float(sae_b),
                "sae_c": -0.0000002 if sae_c is None else float(sae_c),
                "sae_alpha": -0.4000 if sae_alpha is None else float(sae_alpha),
                "sae_beta": 1.0100 if sae_beta is None else float(sae_beta),
            }
        )
    if test_mass_kg is not None:
        row["test_load_value"] = float(test_mass_kg)
        row["load_unit"] = "kg"
        row["sae_reference_load_n"] = round(float(test_mass_kg) * QA_N_PER_KG, 4)
    return row


def _complete_sae_qa_tire_row(spec: dict[str, Any]) -> dict[str, Any]:
    return _base_tire_row(
        int(spec["row_id"]),
        str(spec["qa_id"]),
        str(spec["model_name"]),
        rr_n_per_kn=float(spec["rr_n_per_kn"]),
        standard_family="SAE",
        pressure_psi=float(spec["pressure_psi"]),
        load_kg=float(spec["load_kg"]),
        test_mass_kg=float(spec["load_kg"]),
        test_mileage_km=float(spec["test_mileage_km"]),
        test_speed_value=float(spec["test_speed_value"]),
        include_sae_coefficients=True,
        sae_alpha=float(spec["sae_alpha"]),
        sae_beta=float(spec["sae_beta"]),
        sae_a=float(spec["sae_a"]),
        sae_b=float(spec["sae_b"]),
        sae_c=float(spec["sae_c"]),
        rr_quality="reference_rr_input",
        notes=str(spec["notes"]),
    )


def _required_inertia_class_for_mass(mass_kg: float) -> float:
    row = inertia_row_for_mass(mass_kg)
    if not row:
        raise ValueError(f"No canonical inertia class found for mass {mass_kg}")
    return float(row["inertia_class_kg"])


def _to_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _required_float(value: Any) -> float:
    result = _to_float(value)
    if result is None:
        raise ValueError(f"Expected a numeric value, got {value!r}")
    return float(result)


__all__ = [
    "COMPLETE_SAE_QA_TIRE_SPECS",
    "DEFAULT_QA_DB_PATH",
    "GOLDEN_QA_SCENARIO",
    "INERTIA_TABLE_PATH",
    "QA_COMPONENT_FIXTURES",
    "QA_DATA_DIR",
    "SYNTHETIC_QA_WARNING",
    "build_seed_manifest",
    "build_tire_seed_rows",
    "build_vde_seed_rows",
    "inertia_row_for_class",
    "inertia_row_for_mass",
    "is_safe_qa_db_path",
    "load_inertia_table",
    "manifest_json",
    "seed_qa_database",
    "seeded_database_digest",
    "temporary_seeded_db",
]
