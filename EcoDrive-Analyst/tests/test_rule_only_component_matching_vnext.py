from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

from scripts.run_rule_only_component_matching_vnext import run
from src.vde_core.component_prior_matching import ReferencePrior
from src.vde_core.component_prior_matching_vnext import (
    NormalizedTechnicalIdentity,
    match_component_priors_vnext,
    normalize_application_class,
    normalize_drive,
    normalize_electrification,
    normalize_transmission,
)


def _reference(
    boundary: str,
    *,
    application_class: str = "CROSSOVER",
    drive: str = "AWD",
    position: str | None = None,
    suffix: str = "1",
) -> ReferencePrior:
    return ReferencePrior(
        component_id=f"C_{boundary}_{suffix}",
        component_resolution_id=f"R_{boundary}_{suffix}",
        component_domain=boundary if boundary in {"BRAKE", "TRANSMISSION"} else "AXLE_HUBS",
        boundary=boundary,
        hardware_reference="SYNTHETIC_REFERENCE",
        application_class=application_class,
        drive_architecture=drive,
        position=position,
    )


def _catalog() -> list[ReferencePrior]:
    return [
        _reference("BRAKE"),
        _reference("TRANSMISSION"),
        _reference("AXLE", position="FRONT", suffix="F"),
        _reference("AXLE", position="REAR", suffix="R"),
        _reference("HUB_BEARING", position="FRONT", suffix="F"),
        _reference("HUB_BEARING", position="REAR", suffix="R"),
    ]


def _identity(*, category: str = "SMALL SUVS", drive: str = "All Wheel Drive") -> NormalizedTechnicalIdentity:
    return NormalizedTechnicalIdentity(
        vehicle_configuration_id="VC1",
        drive=normalize_drive(drive),
        transmission=normalize_transmission("Automatic"),
        electrification=normalize_electrification(["ICE"]),
        application_class=normalize_application_class([category]),
    )


def test_approved_drive_normalization_preserves_raw_value() -> None:
    expected = {
        "2-Wheel Drive, Front": "FWD",
        "2-Wheel Drive, Rear": "RWD",
        "All Wheel Drive": "AWD",
        "4-Wheel Drive": "4WD",
        "Part-time 4-Wheel Drive": "4WD",
    }
    for raw, normalized in expected.items():
        result = normalize_drive(raw)
        assert result.raw_values == (raw,)
        assert result.normalized == normalized
        assert result.status == "MAPPED"

    assert normalize_drive("Four by four").status == "UNMAPPED"


def test_transmission_uses_only_permitted_exact_legacy_contract() -> None:
    assert normalize_transmission("Automatic").normalized == "AT"
    assert normalize_transmission("Semi-Automatic").normalized == "AMT"
    assert normalize_transmission("Continuously Variable").normalized == "CVT"
    assert normalize_transmission("Manual").normalized == "MT"
    assert normalize_transmission("Other").normalized == "OT"

    # These are not exact permitted-output entries in the legacy contract.
    assert normalize_transmission("CVT").status == "UNMAPPED"
    assert normalize_transmission(
        "Selectable Continuously Variable (e.g. CVT with paddles)"
    ).status == "UNMAPPED"
    assert normalize_transmission("DCT").status == "UNMAPPED"
    assert normalize_transmission("Single Speed").status == "UNMAPPED"


def test_electrification_uses_stored_canonical_values_without_architecture_fallback() -> None:
    bev = normalize_electrification(["BEV"])
    assert bev.normalized == "BEV"
    assert bev.source == "fuelcons.electrification"
    assert normalize_electrification(["ICE", "HEV"]).status == "CONFLICT"
    assert normalize_electrification([]).status == "MISSING"


def test_application_mapping_is_exact_and_keeps_approved_ambiguity() -> None:
    assert normalize_application_class(["SMALL SUVS"]).normalized_options == ("CROSSOVER",)
    vans = normalize_application_class(["VANS"])
    assert vans.status == "AMBIGUOUS"
    assert vans.normalized_options == ("CARGO_VAN", "PASSENGER_VAN")
    two_seaters = normalize_application_class(["TWO SEATERS"])
    assert two_seaters.status == "AMBIGUOUS"
    assert two_seaters.normalized_options == ()
    assert normalize_application_class(["CLASS 3 (>1220 KG)"]).status == "UNMAPPED"


def test_unique_application_drive_match_is_strong_and_slots_remain_distinct() -> None:
    decisions = match_component_priors_vnext(_identity(), _catalog())

    assert len(decisions) == 6
    assert all(decision.match_level == "STRONG_RULE_MATCH" for decision in decisions)
    axle = [decision for decision in decisions if decision.domain == "AXLE"]
    hub = [decision for decision in decisions if decision.domain == "HUB_BEARING"]
    assert [decision.component_slot for decision in axle] == ["FRONT", "REAR"]
    assert [decision.component_slot for decision in hub] == ["FRONT", "REAR"]
    assert axle[0].candidate_component_ids != axle[1].candidate_component_ids


def test_ambiguous_legacy_class_is_never_upgraded_by_catalog_availability() -> None:
    catalog = [
        _reference("BRAKE", application_class="CARGO_VAN", drive="FWD", suffix="C"),
        _reference("BRAKE", application_class="PASSENGER_VAN", drive="FWD", suffix="P"),
    ]
    decisions = match_component_priors_vnext(
        _identity(category="VANS", drive="2-Wheel Drive, Front"), catalog
    )
    brake = decisions[0]

    assert brake.match_level == "AMBIGUOUS_RULE_MATCH"
    assert brake.candidate_component_ids == ("C_BRAKE_C", "C_BRAKE_P")


def test_matching_does_not_use_abc_or_commercial_name() -> None:
    identity = _identity()
    baseline = match_component_priors_vnext(identity, _catalog())
    altered = [
        ReferencePrior(
            **{
                **reference.__dict__,
                "metadata": {"resolved_A_N": 999999, "commercial_model": "DIFFERENT"},
            }
        )
        for reference in _catalog()
    ]
    assert match_component_priors_vnext(identity, altered) == baseline


def _create_candidate(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA foreign_keys=ON;
        CREATE TABLE program (
            program_id TEXT PRIMARY KEY,
            commercial_make TEXT,
            commercial_model TEXT,
            model_year_from INTEGER,
            model_year_to INTEGER
        );
        CREATE TABLE vehicle_configuration (
            vehicle_configuration_id TEXT PRIMARY KEY,
            program_id TEXT REFERENCES program(program_id),
            drive_system TEXT,
            transmission_type TEXT,
            propulsion_architecture TEXT
        );
        CREATE TABLE vde (
            id INTEGER PRIMARY KEY,
            vehicle_configuration_id TEXT REFERENCES vehicle_configuration(vehicle_configuration_id),
            category TEXT
        );
        CREATE TABLE fuelcons (
            id INTEGER PRIMARY KEY,
            vde_id INTEGER REFERENCES vde(id),
            electrification TEXT
        );
        CREATE TABLE component_db (
            component_id TEXT PRIMARY KEY,
            component_domain TEXT,
            model TEXT,
            hardware_reference TEXT,
            custom_properties_json TEXT,
            provenance_json TEXT,
            source_name TEXT
        );
        CREATE TABLE component_resolution (
            component_resolution_id TEXT PRIMARY KEY,
            boundary TEXT,
            method TEXT,
            resolved_A_N REAL,
            resolved_B_N_per_kph REAL,
            resolved_C_N_per_kph2 REAL,
            conditions_json TEXT,
            provenance_json TEXT
        );
        INSERT INTO program VALUES ('P1','Example Make','Example Model',2024,2024);
        INSERT INTO vehicle_configuration VALUES (
            'VC1','P1','All Wheel Drive','Automatic','do not use this field'
        );
        INSERT INTO vde VALUES (1,'VC1','Car');
        INSERT INTO fuelcons VALUES (1,1,'ICE');
        """
    )
    for reference in _catalog():
        connection.execute(
            "INSERT INTO component_db VALUES (?,?,?,?,?,?,?)",
            (
                reference.component_id,
                reference.component_domain,
                reference.model,
                reference.hardware_reference,
                json.dumps(
                    {
                        "application_class": reference.application_class,
                        "drive_architecture": reference.drive_architecture,
                        "position": reference.position,
                    }
                ),
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


def _create_legacy(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE vde_db (make TEXT, model TEXT, year INTEGER, category TEXT);
        INSERT INTO vde_db VALUES ('EXAMPLE MAKE','EXAMPLE MODEL',2024,'SMALL SUVS');
        """
    )
    connection.commit()
    connection.close()


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_full_runner_is_read_only_deterministic_and_uses_exact_legacy_lookup(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.db"
    legacy = tmp_path / "legacy.db"
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    _create_candidate(candidate)
    _create_legacy(legacy)
    candidate_before = _hash(candidate)
    legacy_before = _hash(legacy)

    first = run(candidate, legacy, first_dir)
    second = run(candidate, legacy, second_dir)

    assert _hash(candidate) == candidate_before
    assert _hash(legacy) == legacy_before
    assert first["quick_check"] == "ok"
    assert first["foreign_key_check_issue_count"] == 0
    assert first["configuration_match_rows"] == 6
    assert first["vde_match_rows"] == 6
    assert first["unresolved_identity_rows"] == 0
    assert first["artifact_sha256"] == second["artifact_sha256"]
    summary = json.loads((first_dir / "coverage_summary.json").read_text(encoding="utf-8"))
    assert summary["v1_comparison"]["vnext_vde_slot_projections"]["STRONG_RULE_MATCH"] == 6
    assert summary["normalization_status_by_configuration"]["application_class"] == {
        "MAPPED": 1
    }
    audit = (first_dir / "normalization_audit.csv").read_text(encoding="utf-8")
    assert "EXACT_MAKE_MODEL_MODEL_YEAR_LOOKUP" in audit
    assert "SMALL SUVS" in audit
