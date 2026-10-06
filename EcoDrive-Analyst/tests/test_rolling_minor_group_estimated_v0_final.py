from __future__ import annotations

import csv
import json
from pathlib import Path
import sqlite3
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export_rolling_minor_group_estimated_v0_excel import sha256
from scripts.verify_rolling_minor_group_estimated_v0_excel import verify


PACKAGE = REPO_ROOT / "inputs/rollingminor_group_estimated_v0_final/EcoDrive_RollingMinor_GroupEstimated_v0_FINAL"
DB = PACKAGE / "eco_drive_canonical_sprint12_group_estimated_v0.db"
WORKBOOK = REPO_ROOT / "artifacts/rolling_minor_group_estimated_v0_final/EcoDrive_RollingMinor_GroupEstimated_v0_FINAL.xlsx"


pytestmark = pytest.mark.skipif(not DB.exists(), reason="Group Estimated v0 final package not present")


def _connection() -> sqlite3.Connection:
    resolved = DB.resolve(strict=True)
    connection = sqlite3.connect(f"file:{resolved.as_posix()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def test_package_manifest_and_database_integrity() -> None:
    manifest = json.loads((PACKAGE / "FINAL_MANIFEST_V0.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = PACKAGE / name
        assert path.stat().st_size == expected["bytes"]
        assert sha256(path).lower() == expected["sha256"]
    connection = _connection()
    try:
        assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        connection.close()


def test_group_estimates_close_and_preserve_scientific_boundaries() -> None:
    connection = _connection()
    try:
        row = connection.execute("""
            SELECT COUNT(*),COUNT(DISTINCT vde_id),
                   MIN(tire_other_residual_share_50),
                   MAX(ABS(brake_share_50+hub_share_50+tire_other_residual_share_50-1.0)),
                   MIN(residual_min_force_10_70mph_N),
                   MAX(closure_max_abs_error_10_70mph_N),
                   SUM(method<>'GROUP_ESTIMATED_ROLLING_MINOR_V0'),
                   SUM(metric_basis<>'FORCE_50_MPH')
            FROM rolling_minor_group_estimate_v0
        """).fetchone()
    finally:
        connection.close()
    assert row[0] == row[1] == 7_429
    assert row[2] > 0
    assert row[3] < 1e-12
    assert row[4] > 0
    assert row[5] < 1e-10
    assert row[6] == row[7] == 0


def test_csvs_reconcile_to_materialized_tables() -> None:
    with (PACKAGE / "ROLLING_MINOR_GROUP_ESTIMATED_V0.csv").open(encoding="utf-8-sig", newline="") as handle:
        estimate_rows = sum(1 for _ in csv.reader(handle)) - 1
    with (PACKAGE / "ROLLING_MINOR_GROUP_PRIORS_V0.csv").open(encoding="utf-8-sig", newline="") as handle:
        prior_rows = sum(1 for _ in csv.reader(handle)) - 1
    assert estimate_rows == 7_429
    assert prior_rows == 9


@pytest.mark.skipif(not WORKBOOK.exists(), reason="Group Estimated v0 final workbook not present")
def test_final_workbook_reconciles_to_database() -> None:
    result = verify(DB, WORKBOOK)
    assert result["status"] == "PASS", result["failures"]
    assert all(item["match"] for item in result["reconciliation"])
    assert len(result["chart_parts"]) == 1
