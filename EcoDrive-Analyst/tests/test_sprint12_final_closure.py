from __future__ import annotations

from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.verify_sprint12_final_excel import verify
from vde_core.sprint12_final_closure import (
    EXPECTED_FINAL_SHA256,
    audit_database,
    component_coverage_rows,
    coverage_summary,
    file_sha256,
    open_read_only,
)


FINAL_DB = REPO_ROOT / "artifacts/sprint12_final/eco_drive_canonical_sprint12_final.db"
FINAL_XLSX = REPO_ROOT / "artifacts/sprint12_final/EcoDrive_Canonical_DB_Sprint12_Final.xlsx"


pytestmark = pytest.mark.skipif(not FINAL_DB.exists(), reason="Sprint 12 final release artifact not present")


def test_final_release_audit_is_read_only_and_integral() -> None:
    before = file_sha256(FINAL_DB)
    audit = audit_database(FINAL_DB)
    after = file_sha256(FINAL_DB)
    assert before == EXPECTED_FINAL_SHA256 == after
    assert audit["quick_check"] == ["ok"]
    assert audit["foreign_key_issue_count"] == 0


def test_final_component_coverage_reconciles_to_frozen_baseline() -> None:
    connection = open_read_only(FINAL_DB)
    try:
        rows = component_coverage_rows(connection)
        metrics, _ = coverage_summary(connection, rows, research_agent_summary={"deferred_no_current_model_value": 1})
    finally:
        connection.close()
    values = {row["metric"]: row["value"] for row in metrics}
    assert len(rows) == values["TOTAL_VDE"] == 11_626
    assert values["MACRO_RESOLVED"] == 7_429
    assert values["MACRO_SUPPORTED"] == 5_720
    assert values["MACRO_CONDITIONAL"] == 1_709
    assert values["MACRO_UNRESOLVED"] == 4_197
    assert values["MACRO_PROJECTED_TO_HISTORICAL_SLOTS"] == 7_183
    assert values["EDRIVE_AGGREGATE_RETAINED_WITHOUT_TRANSMISSION_PROJECTION"] == 246
    assert values["FINE_SUPPORTING_LINKS"] == 18_646
    assert values["FINE_ADOPTED_LINKS"] == 0


@pytest.mark.skipif(not FINAL_XLSX.exists(), reason="Sprint 12 final workbook artifact not present")
def test_final_excel_reconciles_to_sqlite() -> None:
    result = verify(FINAL_DB, FINAL_XLSX)
    assert result["status"] == "PASS", result["failures"]
    assert all(item["match"] for item in result["raw_reconciliation"])
    assert all(item["match"] for item in result["derived_reconciliation"].values())
