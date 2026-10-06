from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3

from scripts.run_rule_only_component_matching import run
from src.vde_core.component_prior_matching import (
    MATCH_DOMAINS,
    ReferencePrior,
    TechnicalIdentity,
    match_component_priors,
)


def _reference(
    domain: str,
    *,
    application_class: str = "SUV",
    drive: str = "AWD",
    position: str | None = None,
    suffix: str = "1",
    hardware_reference: str = "SYNTHETIC_REFERENCE",
) -> ReferencePrior:
    return ReferencePrior(
        component_id=f"C_{domain}_{suffix}",
        component_resolution_id=f"R_{domain}_{suffix}",
        component_domain=domain if domain in {"BRAKE", "TRANSMISSION"} else "AXLE_HUBS",
        boundary=domain,
        model=f"{domain} reference",
        hardware_reference=hardware_reference,
        application_class=application_class,
        drive_architecture=drive,
        position=position,
    )


def _catalog() -> list[ReferencePrior]:
    return [
        _reference("BRAKE"),
        _reference("TRANSMISSION"),
        _reference("AXLE", position="FRONT"),
        _reference("HUB_BEARING", position="FRONT"),
    ]


def test_explicit_category_rules_do_not_overclaim_position_specific_references() -> None:
    identity = TechnicalIdentity(
        vehicle_configuration_id="VC1",
        category="SUV",
        drive_system="All Wheel Drive",
    )

    results = {row.domain: row for row in match_component_priors(identity, _catalog())}

    assert results["BRAKE"].match_level == "CATEGORY_RULE_MATCH"
    assert results["TRANSMISSION"].match_level == "CATEGORY_RULE_MATCH"
    assert results["AXLE"].match_level == "NO_MATCH"
    assert results["AXLE"].missing_discriminants == ("component_position",)
    assert results["HUB_BEARING"].match_level == "NO_MATCH"


def test_broad_or_missing_class_is_no_match_instead_of_a_guess() -> None:
    broad = TechnicalIdentity(
        vehicle_configuration_id="VC1", category="Truck", drive_system="4-Wheel Drive"
    )
    missing = replace(broad, category=None)

    for identity in (broad, missing):
        results = match_component_priors(identity, _catalog())
        assert all(row.match_level == "NO_MATCH" for row in results)
        assert all("application_class" in row.missing_discriminants for row in results)


def test_ambiguous_explicit_rule_is_no_match() -> None:
    catalog = _catalog() + [_reference("BRAKE", suffix="2")]
    identity = TechnicalIdentity(
        vehicle_configuration_id="VC1",
        application_class="SUV",
        drive_system="AWD",
    )

    brake = match_component_priors(identity, catalog)[0]

    assert brake.match_level == "NO_MATCH"
    assert brake.rule_id == "BRAKE_AMBIGUOUS_REFERENCE_V1"


def test_axle_and_hub_bearing_remain_separate_and_order_is_deterministic() -> None:
    identity = TechnicalIdentity(
        vehicle_configuration_id="VC1",
        application_class="SUV",
        drive_system="AWD",
        component_position="FRONT",
    )

    forward = match_component_priors(identity, _catalog())
    reverse = match_component_priors(identity, reversed(_catalog()))

    assert forward == reverse
    assert [row.domain for row in forward] == list(MATCH_DOMAINS)
    assert forward[2].component_id == "C_AXLE_1"
    assert forward[3].component_id == "C_HUB_BEARING_1"


def test_exact_requires_a_unique_non_generic_hardware_identifier() -> None:
    catalog = _catalog() + [
        _reference("BRAKE", suffix="HW", hardware_reference="HW-123")
    ]
    identity = TechnicalIdentity(
        vehicle_configuration_id="VC1", hardware_reference="HW-123"
    )

    brake = match_component_priors(identity, catalog)[0]

    assert brake.match_level == "EXACT_RULE_MATCH"
    assert brake.component_id == "C_BRAKE_HW"


def test_matching_has_no_dependency_on_reference_abc_magnitude() -> None:
    identity = TechnicalIdentity(
        vehicle_configuration_id="VC1", category="SUV", drive_system="AWD"
    )
    baseline = match_component_priors(identity, _catalog())
    metadata_only = [replace(ref, metadata={"resolved_A_N": 999999}) for ref in _catalog()]

    assert match_component_priors(identity, metadata_only) == baseline


def _fixture_db(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA foreign_keys=ON;
        CREATE TABLE program (
            program_id TEXT PRIMARY KEY,
            commercial_make TEXT NOT NULL,
            commercial_model TEXT NOT NULL,
            model_year_from INTEGER,
            model_year_to INTEGER
        );
        CREATE TABLE vehicle_configuration (
            vehicle_configuration_id TEXT PRIMARY KEY,
            program_id TEXT NOT NULL REFERENCES program(program_id),
            drive_system TEXT,
            propulsion_architecture TEXT,
            transmission_type TEXT,
            transmission_model TEXT,
            gear_count INTEGER,
            final_drive_ratio REAL,
            engine_type TEXT,
            engine_aspiration TEXT,
            engine_displacement_l REAL,
            engine_rated_power_kw REAL,
            architecture_properties_json TEXT
        );
        CREATE TABLE vde (
            id INTEGER PRIMARY KEY,
            vehicle_configuration_id TEXT NOT NULL REFERENCES vehicle_configuration(vehicle_configuration_id),
            category TEXT,
            drive_type TEXT,
            transmission_type TEXT,
            transmission_model TEXT,
            year INTEGER
        );
        CREATE TABLE component_db (
            component_id TEXT PRIMARY KEY,
            component_domain TEXT NOT NULL,
            model TEXT,
            hardware_reference TEXT,
            custom_properties_json TEXT,
            provenance_json TEXT NOT NULL,
            source_name TEXT
        );
        CREATE TABLE component_resolution (
            component_resolution_id TEXT PRIMARY KEY,
            boundary TEXT NOT NULL,
            method TEXT NOT NULL,
            resolved_A_N REAL,
            resolved_B_N_per_kph REAL,
            resolved_C_N_per_kph2 REAL,
            conditions_json TEXT,
            provenance_json TEXT NOT NULL
        );
        INSERT INTO program VALUES ('P1','Example','Vehicle',2024,2024);
        INSERT INTO vehicle_configuration VALUES (
            'VC1','P1','All Wheel Drive','Electricity','Automatic',NULL,1,
            1.0,'Electricity',NULL,0.001,200.0,NULL
        );
        INSERT INTO vde VALUES (1,'VC1','SUV','All Wheel Drive','Automatic',NULL,2024);
        """
    )
    for reference in _catalog():
        properties = json.dumps(
            {
                "application_class": reference.application_class,
                "drive_architecture": reference.drive_architecture,
                "position": reference.position,
            },
            sort_keys=True,
        )
        connection.execute(
            "INSERT INTO component_db VALUES (?,?,?,?,?,?,?)",
            (
                reference.component_id,
                reference.component_domain,
                reference.model,
                reference.hardware_reference,
                properties,
                json.dumps({"synthetic_reference": True}),
                "SYNTHETIC_REFERENCE",
            ),
        )
        connection.execute(
            "INSERT INTO component_resolution VALUES (?,?,?,?,?,?,?,?)",
            (
                reference.component_resolution_id,
                reference.boundary,
                "SYNTHETIC_POPULATION_REFERENCE_V1",
                1.0,
                2.0,
                3.0,
                "{}",
                json.dumps(
                    {
                        "synthetic_reference": True,
                        "synthetic_reference_component_id": reference.component_id,
                    }
                ),
            ),
        )
    connection.commit()
    connection.close()


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_full_runner_is_read_only_complete_and_repeatable(tmp_path: Path) -> None:
    db = tmp_path / "canonical.db"
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    _fixture_db(db)
    before = _hash(db)

    first = run(db, first_dir)
    second = run(db, second_dir)

    assert _hash(db) == before
    assert first["db_modified"] is False
    assert first["quick_check"] == "ok"
    assert first["foreign_key_check_issue_count"] == 0
    assert first["configuration_match_rows"] == 4
    assert first["vde_match_rows"] == 4
    assert first["artifact_sha256"] == second["artifact_sha256"]
    expected = {
        "reference_catalog.csv",
        "reference_catalog.json",
        "reference_catalog_summary.md",
        "canonical_discriminant_profile.csv",
        "config_prior_matches.csv",
        "vde_prior_matches.csv",
        "no_match_work_queue.csv",
        "category_match_work_queue.csv",
        "coverage_summary.json",
        "coverage_report.md",
    }
    assert {path.name for path in first_dir.iterdir()} == expected
    summary = json.loads((first_dir / "coverage_summary.json").read_text(encoding="utf-8"))
    assert summary["population"]["vdes"] == 1
    assert summary["by_domain"]["BRAKE"]["vdes"]["CATEGORY_RULE_MATCH"] == 1
    assert summary["by_domain"]["AXLE"]["vdes"]["NO_MATCH"] == 1
