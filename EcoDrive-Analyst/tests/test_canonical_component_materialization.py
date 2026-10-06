from __future__ import annotations

from pathlib import Path

from src.vde_core.canonical_component_materialization import (
    _gap_specs,
    canonical_candidate_path,
    group_gap_inventory,
)


def _gap_row(**overrides):
    row = {
        "vde_id": 1, "vehicle_configuration_id": "VC-1", "gap_family": "TRANSMISSION_EVIDENCE_GAP",
        "gap_domain": "TRANSMISSION", "architecture_class": "CONVENTIONAL_MULTI_SPEED",
        "drive_type": "RWD", "transmission_code": "8HP50", "engine_code": "B58",
        "make": "BMW", "model": "330i", "model_year": 2022,
        "transmission_type": "AT", "gears": 8, "application_class": "PASSENGER_STANDARD",
        "current_confidence": "SUPPORTED", "boundary_status": "BOUNDARY_UNKNOWN",
        "missing_evidence": "Transmission hardware/family and loss boundary",
    }
    row.update(overrides)
    return row


def test_candidate_path_uses_repository_staging_constant() -> None:
    root = Path("C:/repo")
    assert canonical_candidate_path(root) == root / "data/db/staging/eco_drive_canonical_candidate.db"


def test_fixed_edrive_gets_boundary_gap_not_transmission_gap() -> None:
    row = {
        "architecture_class": "EV_FIXED_GEAR", "drive_type": "RWD", "macro_status": "CONDITIONAL",
        "application_class": "PASSENGER_LIGHT", "tire_available": 1, "brake_available": 1,
        "hub_available": 1, "transmission_available": 0, "axle_available": 0,
        "boundary_status": None,
    }
    families = {item[0] for item in _gap_specs(row)}
    assert "EDRIVE_FINE_BOUNDARY_GAP" in families
    assert "TRANSMISSION_EVIDENCE_GAP" not in families
    assert "AXLE_EVIDENCE_GAP" not in families


def test_grouping_is_deterministic_and_reuses_exact_hardware_code() -> None:
    rows = [_gap_row(vde_id=2, vehicle_configuration_id="VC-2"), _gap_row()]
    first = group_gap_inventory(rows)
    second = group_gap_inventory(list(reversed(rows)))
    assert first == second
    assert len(first) == 1
    assert first[0]["identity_tier"] == 1
    assert first[0]["vde_count"] == 2
    assert first[0]["known_hardware_codes"] == "8HP50;B58"


def test_grouping_never_merges_different_physical_boundaries() -> None:
    rows = [
        _gap_row(gap_family="TRANSMISSION_EVIDENCE_GAP", gap_domain="TRANSMISSION"),
        _gap_row(vde_id=2, vehicle_configuration_id="VC-2", gap_family="AXLE_EVIDENCE_GAP", gap_domain="AXLE"),
    ]
    groups = group_gap_inventory(rows)
    assert len(groups) == 2
    assert {group["domain"] for group in groups} == {"TRANSMISSION", "AXLE"}
