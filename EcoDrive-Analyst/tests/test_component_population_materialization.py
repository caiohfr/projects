from __future__ import annotations

import json
import sqlite3

from src.vde_core.component_population_materialization import (
    _projection_decision,
    apply_projection_manifest,
)


def _macro() -> dict:
    common = {"component_resolution_id": "x", "fit_nrmse_pct": 0.0}
    return {
        "AERO": {**common, "component_resolution_id": "a", "resolved_A_N": 0.0, "resolved_B_N_per_kph": 0.0, "resolved_C_N_per_kph2": 0.031},
        "ROLLING_MINOR": {**common, "component_resolution_id": "r", "resolved_A_N": 101.0, "resolved_B_N_per_kph": 0.12, "resolved_C_N_per_kph2": 0.0},
        "DRIVETRAIN_AGGREGATE": {**common, "component_resolution_id": "d", "resolved_A_N": 22.0, "resolved_B_N_per_kph": 0.21, "resolved_C_N_per_kph2": 0.002},
    }


def _empty_row() -> dict:
    return {
        "tire_A_final": None, "tire_B_final": None, "tire_C_final": None,
        "aero_C_coef_Npkph2": None,
        "trans_A_coef_N": None, "trans_B_coef_Npkph": None, "trans_C_coef_Npkph2": None,
        "brake_A_coef_N": None, "brake_B_coef_Npkph": None, "brake_C_coef_Npkph2": None,
        "parasitic_A_coef_N": None, "parasitic_B_coef_Npkph": None, "parasitic_C_coef_Npkph2": None,
    }


def _db() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE vde (
          id INTEGER PRIMARY KEY,
          tire_A_final REAL,tire_B_final REAL,tire_C_final REAL,
          aero_C_coef_Npkph2 REAL,
          trans_A_coef_N REAL,trans_B_coef_Npkph REAL,trans_C_coef_Npkph2 REAL,
          provenance_json TEXT
        );
        CREATE TABLE component_resolution (
          component_resolution_id TEXT PRIMARY KEY,boundary TEXT NOT NULL,method TEXT NOT NULL,
          provenance_json TEXT NOT NULL,resolved_A_N REAL,resolved_B_N_per_kph REAL,
          resolved_C_N_per_kph2 REAL,record_status TEXT,review_status TEXT,
          estimate_status TEXT,estimator_version TEXT
        );
        CREATE TABLE vde_component_resolution (
          vde_id INTEGER NOT NULL,component_resolution_id TEXT NOT NULL,boundary TEXT NOT NULL,
          adoption_role TEXT NOT NULL,ordinal INTEGER NOT NULL,
          PRIMARY KEY(vde_id,component_resolution_id,boundary),
          FOREIGN KEY(vde_id) REFERENCES vde(id),
          FOREIGN KEY(component_resolution_id) REFERENCES component_resolution(component_resolution_id)
        );
        INSERT INTO vde(id,provenance_json) VALUES (1,'{"source":"fixture"}');
        """
    )
    return connection


def test_macro_projection_uses_aggregate_semantics_without_fine_values() -> None:
    decision = _projection_decision(_empty_row(), _macro(), {})
    assert decision["projection_mode"] == "MACRO_DECOMPOSITION"
    assert decision["tire_A_final"] == 101.0
    assert decision["trans_A_coef_N"] == 22.0
    provenance = json.loads(decision["provenance_summary"])
    assert provenance["tire_slot_semantics"] == "ROLLING_MINOR_AGGREGATE"
    assert provenance["transmission_slot_semantics"] == "DRIVETRAIN_AGGREGATE"
    assert provenance["fine_component_identification"] is False


def test_existing_value_is_never_overwritten() -> None:
    row = _empty_row()
    row["tire_A_final"] = 0.0  # explicit zero is existing evidence, not missing
    decision = _projection_decision(row, _macro(), {})
    assert decision["projection_mode"] == "UNRESOLVED"
    assert decision["write_action"] == "NO_WRITE"
    assert "EXISTING_HIGHER_QUALITY_VALUE" in decision["reason_codes"]


def test_edrive_is_not_relabelled_as_transmission() -> None:
    macro = _macro()
    macro["EDRIVE_AGGREGATE"] = macro.pop("DRIVETRAIN_AGGREGATE")
    decision = _projection_decision(_empty_row(), macro, {})
    assert decision["projection_mode"] == "UNRESOLVED"
    assert "SCHEMA_PROJECTION_UNAVAILABLE_EDRIVE_AGGREGATE" in decision["reason_codes"]


def test_temp_projection_is_idempotent_and_preserves_existing_provenance() -> None:
    connection = _db()
    decision = _projection_decision(_empty_row(), _macro(), {})
    manifest = [{"vde_id": 1, **decision}]
    first = apply_projection_manifest(connection, manifest, {}, [], {})
    second = apply_projection_manifest(connection, manifest, {}, [], {})
    assert first["vde_rows_changed"] == 1
    assert second == {
        "vde_rows_changed": 0,
        "fine_component_resolutions_inserted": 0,
        "fine_supporting_links_inserted": 0,
    }
    row = connection.execute("SELECT * FROM vde WHERE id=1").fetchone()
    assert row["tire_A_final"] == 101.0
    assert row["trans_A_coef_N"] == 22.0
    provenance = json.loads(row["provenance_json"])
    assert provenance["source"] == "fixture"
    assert provenance["component_population_projection"]["decomposition_mode"] == "MACRO_DECOMPOSITION"
    connection.close()
