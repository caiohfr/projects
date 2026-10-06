"""Read-only discovery gate for the Sprint 12 Component Knowledge 40 package.

The locally available repository does not contain the historical PL source
package referenced by the Component provenance documentation.  This module
therefore produces deterministic blocker/audit artifacts only.  It never
opens SQLite for writing and never promotes estimated whole-vehicle roadload
splits into component evidence.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import sqlite3
from typing import Any, Sequence
import xml.etree.ElementTree as ET
import zipfile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATE = ROOT / "data" / "db" / "staging" / "eco_drive_canonical_candidate.db"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "components"

INVENTORY_FIELDS = (
    "source_path",
    "source_type",
    "source_name",
    "component_families_observed",
    "row_count_if_applicable",
    "identity_evidence",
    "loss_evidence",
    "architecture_evidence",
    "provenance_quality",
    "raw_numeric_component_source",
    "usable_for_component_db",
    "usable_for_component_instance",
    "usable_for_component_resolution",
    "notes",
)

COVERAGE_FIELDS = (
    "dimension_number",
    "dimension_name",
    "sprint_12_scope",
    "status",
    "available_evidence",
    "blocker_or_deferred_reason",
)

SEED_AUDIT_FIELDS = (
    "entity",
    "before_count",
    "after_count",
    "package_action",
    "status",
    "reason",
)

REFERENCE_FIELDS = (
    "population_key",
    "component_family",
    "component_subtype",
    "position",
    "architecture",
    "hardware_reference_if_known",
    "method",
    "boundary",
    "n",
    "median_force_50_N",
    "p10_force_50_N",
    "p90_force_50_N",
    "median_A_N",
    "median_B_N_per_kph",
    "median_C_N_per_kph2",
    "quality_status",
    "source_scope",
)

OUTLIER_FIELDS = (
    "source_record",
    "component_family",
    "value_field",
    "value",
    "reason_flagged",
    "physical_interpretation_status",
    "review_status",
    "source_path",
)


@dataclass(frozen=True)
class DiscoveryReport:
    component_source_data_available: bool
    candidate_path: str
    candidate_sha256_before: str
    candidate_sha256_after: str
    quick_check: str
    foreign_key_issues: int
    component_db_count: int
    component_instance_count: int
    component_resolution_count: int
    vde_component_resolution_count: int
    tire_db_count: int
    inventory_rows: int
    implemented_dimensions: int
    partial_dimensions: int
    reference_population_rows: int
    outlier_review_rows: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _csv_rows(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def _xlsx_first_sheet_rows(path: Path) -> int | None:
    if not path.is_file():
        return None
    with zipfile.ZipFile(path) as archive:
        sheet = "xl/worksheets/sheet1.xml"
        if sheet not in archive.namelist():
            return None
        rows = 0
        with archive.open(sheet) as handle:
            for _event, element in ET.iterparse(handle, events=("end",)):
                if element.tag.endswith("}row"):
                    rows += 1
                element.clear()
    return max(rows - 1, 0)


def _display_count(value: int | None) -> str:
    return "" if value is None else str(value)


def _inventory_rows(root: Path, candidate_counts: dict[str, int]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(**values: str) -> None:
        row = {field: "" for field in INVENTORY_FIELDS}
        row.update(values)
        rows.append(row)

    mock_specs = (
        ("data/components/brake_components_mock.csv", "BRAKE BASELINE; BRAKE STANDARD"),
        ("data/components/transmission_components_mock.csv", "TRANSMISSION"),
        ("data/components/axle_hubs_components_mock.csv", "AXLE; HUB; AXLE_HUB_COMBINED"),
        ("data/components/parasitic_components_mock.csv", "OTHER RESIDUAL PARASITIC"),
    )
    for relative, families in mock_specs:
        path = root / relative
        add(
            source_path=relative,
            source_type="CSV_SYNTHETIC_FIXTURE",
            source_name=path.stem,
            component_families_observed=families,
            row_count_if_applicable=_display_count(_csv_rows(path)),
            identity_evidence="Fabricated mock IDs only",
            loss_evidence="Synthetic ABC values labelled SYNTHETIC_COMPONENT_FIXTURE",
            architecture_evidence="Synthetic demo labels",
            provenance_quality="EXPLICITLY_SYNTHETIC",
            raw_numeric_component_source="NO",
            usable_for_component_db="NO",
            usable_for_component_instance="NO",
            usable_for_component_resolution="NO",
            notes="QA/demo fixture; prohibited as real canonical seed.",
        )

    defaults = root / "data/standards/vde_defaults_by_category_trans_elec.csv"
    add(
        source_path="data/standards/vde_defaults_by_category_trans_elec.csv",
        source_type="CSV_ENGINEERING_PRIOR",
        source_name="VDE defaults by category/transmission/electrification",
        component_families_observed="TRANSMISSION; BRAKE",
        row_count_if_applicable=_display_count(_csv_rows(defaults)),
        identity_evidence="Category/transmission-class prior; no hardware identity",
        loss_evidence="Default trans/brake A/B priors; not observed component tests",
        architecture_evidence="Coarse vehicle category and transmission class",
        provenance_quality="ESTIMATED_DEFAULT_PRIOR_SPLIT",
        raw_numeric_component_source="NO",
        usable_for_component_db="NO",
        usable_for_component_instance="NO",
        usable_for_component_resolution="NO",
        notes="Used by the legacy notebook to split whole-vehicle residual roadload; explicitly not SOURCE/MEASURED.",
    )

    notebook = root / "notebooks/etl_epa_xlsx_to_sqlite.ipynb"
    add(
        source_path="notebooks/etl_epa_xlsx_to_sqlite.ipynb",
        source_type="JUPYTER_DERIVATION_NOTEBOOK",
        source_name="Legacy EPA SQLite ETL notebook",
        component_families_observed="TRANSMISSION; BRAKE; RESIDUAL PARASITIC",
        row_count_if_applicable="",
        identity_evidence="Vehicle-level EPA make/model/configuration only",
        loss_evidence="Derives *_est by splitting whole-vehicle coastdown residual with defaults",
        architecture_evidence="Vehicle drive/transmission descriptors",
        provenance_quality="DERIVED_WHOLE_VEHICLE_ESTIMATE",
        raw_numeric_component_source="NO",
        usable_for_component_db="NO",
        usable_for_component_instance="NO",
        usable_for_component_resolution="NO",
        notes="The code labels transmission/brake values as estimates and uses max/clamped priors; forbidden as physical Component Resolution evidence.",
    )

    legacy_db = root / "data/db/archive/eco_drive_legacy_pre_sprint12.db"
    add(
        source_path="data/db/archive/eco_drive_legacy_pre_sprint12.db",
        source_type="SQLITE_LEGACY_DERIVED",
        source_name="Legacy pre-Sprint-12 runtime database",
        component_families_observed="TRANSMISSION; BRAKE; RESIDUAL PARASITIC",
        row_count_if_applicable="5003 VDE; 2958 transmission estimates; 2959 brake estimates; 2961 residual estimates",
        identity_evidence="No reusable component master rows (component_db=0)",
        loss_evidence="Persisted outputs of the legacy estimated prior split",
        architecture_evidence="Vehicle-level descriptors only",
        provenance_quality="ESTIMATED_NOT_OBSERVED",
        raw_numeric_component_source="NO",
        usable_for_component_db="NO",
        usable_for_component_instance="NO",
        usable_for_component_resolution="NO",
        notes="Numeric values exist but are not component test evidence and cannot be promoted.",
    )

    jrc = root / "etl/data/raw/wltp_jrc/Data_PV_fleet_2021_EU_PYCSIS.xlsx"
    add(
        source_path="etl/data/raw/wltp_jrc/Data_PV_fleet_2021_EU_PYCSIS.xlsx",
        source_type="XLSX_PUBLIC_VEHICLE_SOURCE",
        source_name="JRC PYCSIS 2021",
        component_families_observed="ENGINE; TRANSMISSION; EMOTOR; BATTERY; TIRE",
        row_count_if_applicable=_display_count(_xlsx_first_sheet_rows(jrc)),
        identity_evidence="Application-level descriptors; no defensible reusable hardware key",
        loss_evidence="No component ABC or PL Dyno force curve",
        architecture_evidence="Gearbox, gears, engine, motor, battery and tire descriptors",
        provenance_quality="PUBLIC_SOURCE_APPLICATION_CONTEXT",
        raw_numeric_component_source="NO",
        usable_for_component_db="NO",
        usable_for_component_instance="YES",
        usable_for_component_resolution="NO",
        notes="Already materialized as 843 unresolved component_instance rows with component_id NULL.",
    )

    epa = root / "etl/data/raw/epa_testcar/epa_testcar_2026_raw.xlsx"
    add(
        source_path="etl/data/raw/epa_testcar/epa_testcar_2026_raw.xlsx",
        source_type="XLSX_PUBLIC_VEHICLE_SOURCE",
        source_name="EPA Test Car MY2026",
        component_families_observed="TRANSMISSION; DRIVELINE descriptors",
        row_count_if_applicable=_display_count(_xlsx_first_sheet_rows(epa)),
        identity_evidence="Tested transmission type/gears; no reusable hardware identity",
        loss_evidence="Target/Set ABC are whole-vehicle roadload, not component ABC",
        architecture_evidence="Drive-system description, axle ratio and N/V ratio",
        provenance_quality="PUBLIC_SOURCE_WHOLE_VEHICLE",
        raw_numeric_component_source="NO",
        usable_for_component_db="NO",
        usable_for_component_instance="PARTIAL",
        usable_for_component_resolution="NO",
        notes="May support future application context only; cannot support causal component loss.",
    )

    legacy_epa = root / "data/vehicles/testcar-2025-2020-EPA.xlsx"
    add(
        source_path="data/vehicles/testcar-2025-2020-EPA.xlsx",
        source_type="XLSX_PUBLIC_VEHICLE_SOURCE",
        source_name="EPA Test Car MY2020-2025",
        component_families_observed="TRANSMISSION; DRIVELINE descriptors",
        row_count_if_applicable=_display_count(_xlsx_first_sheet_rows(legacy_epa)),
        identity_evidence="Vehicle/test configuration only",
        loss_evidence="Whole-vehicle coastdown; no PL component subtraction evidence",
        architecture_evidence="Vehicle transmission/drive descriptors",
        provenance_quality="PUBLIC_SOURCE_WHOLE_VEHICLE",
        raw_numeric_component_source="NO",
        usable_for_component_db="NO",
        usable_for_component_instance="PARTIAL",
        usable_for_component_resolution="NO",
        notes="Source of legacy whole-vehicle VDE population, not a component test source.",
    )

    add(
        source_path="data/db/staging/eco_drive_canonical_candidate.db",
        source_type="SQLITE_CANONICAL_CANDIDATE",
        source_name="Normalized canonical staging candidate",
        component_families_observed="ENGINE; TRANSMISSION; EMOTOR; BATTERY; TIRE application instances",
        row_count_if_applicable=(
            f"component_db={candidate_counts['component_db']}; "
            f"component_instance={candidate_counts['component_instance']}; "
            f"component_resolution={candidate_counts['component_resolution']}; "
            f"vde_component_resolution={candidate_counts['vde_component_resolution']}; "
            f"tire_db={candidate_counts['tire_db']}"
        ),
        identity_evidence="843 application instances; all reusable component_id values unresolved",
        loss_evidence="No Component Resolution rows",
        architecture_evidence="JRC application properties and roles",
        provenance_quality="CANONICAL_PUBLIC_SOURCE_CONTEXT",
        raw_numeric_component_source="NO",
        usable_for_component_db="NO",
        usable_for_component_instance="YES",
        usable_for_component_resolution="NO",
        notes="Read-only baseline. Existing instances remain valid but do not satisfy the PL loss-behavior gate.",
    )

    observability = root / "etl/data/processed/sprint_12b_source_to_contract/component_observability.csv"
    add(
        source_path="etl/data/processed/sprint_12b_source_to_contract/component_observability.csv",
        source_type="CSV_PRIOR_AUDIT",
        source_name="Sprint 12B component observability audit",
        component_families_observed="ENGINE; TRANSMISSION; DRIVELINE; TIRE; EMOTOR; BATTERY",
        row_count_if_applicable=_display_count(_csv_rows(observability)),
        identity_evidence="Documents partial instance observability and absent reusable identity",
        loss_evidence="Documents absence of component ABC",
        architecture_evidence="Documents fields available per public source",
        provenance_quality="AUDITED",
        raw_numeric_component_source="NO",
        usable_for_component_db="NO",
        usable_for_component_instance="NO",
        usable_for_component_resolution="NO",
        notes="Audit evidence, not an underlying numeric component source.",
    )

    coverage = root / "etl/data/processed/sprint_12c1_dataset_delta_audit/engineering_coverage.csv"
    add(
        source_path="etl/data/processed/sprint_12c1_dataset_delta_audit/engineering_coverage.csv",
        source_type="CSV_PRIOR_AUDIT",
        source_name="Sprint 12C.1 engineering coverage audit",
        component_families_observed="TRANSMISSION; BRAKE",
        row_count_if_applicable=_display_count(_csv_rows(coverage)),
        identity_evidence="None beyond vehicle population",
        loss_evidence="Explicitly classifies legacy component ABC as ESTIMATED and public-source ABC as absent",
        architecture_evidence="Population-level descriptors",
        provenance_quality="AUDITED",
        raw_numeric_component_source="NO",
        usable_for_component_db="NO",
        usable_for_component_instance="NO",
        usable_for_component_resolution="NO",
        notes="Confirms that legacy estimates must never be upgraded to measured/source evidence.",
    )

    add(
        source_path="docs/component_provenance_metadata.md",
        source_type="MARKDOWN_METHOD_INTERFACE",
        source_name="Component provenance metadata",
        component_families_observed="BRAKE; TRANSMISSION; AXLE; HUB; RESIDUAL PARASITIC",
        row_count_if_applicable="",
        identity_evidence="Defines expected metadata only",
        loss_evidence="No numeric records",
        architecture_evidence="Defines expected boundary/context fields",
        provenance_quality="METHODOLOGY_SUMMARY_ONLY",
        raw_numeric_component_source="NO",
        usable_for_component_db="NO",
        usable_for_component_instance="NO",
        usable_for_component_resolution="NO",
        notes="References an upstream PL Excel, PL notebooks and methodology file that are not present locally.",
    )

    missing_method = root / "Parasitic_Loss_Testing_Methodology(1).md"
    add(
        source_path="Parasitic_Loss_Testing_Methodology(1).md",
        source_type="MISSING_REFERENCED_SOURCE",
        source_name="Parasitic Loss Testing Methodology",
        component_families_observed="Expected BRAKE; TRANSMISSION; AXLE; HUB boundaries",
        row_count_if_applicable="",
        identity_evidence="UNAVAILABLE",
        loss_evidence="UNAVAILABLE",
        architecture_evidence="UNAVAILABLE",
        provenance_quality="MISSING",
        raw_numeric_component_source="NO",
        usable_for_component_db="NO",
        usable_for_component_instance="NO",
        usable_for_component_resolution="NO",
        notes=(
            "Referenced by docs/component_provenance_metadata.md but not present in the working tree or Git history."
            if not missing_method.exists()
            else "Unexpectedly present; discovery classification must be reviewed."
        ),
    )

    return sorted(rows, key=lambda row: (row["source_path"].casefold(), row["source_name"].casefold()))


def _coverage_rows() -> list[dict[str, str | int]]:
    now = "Deterministic Sprint 12 baseline"
    later = "Future Research/RAG layer"
    return [
        {
            "dimension_number": 1,
            "dimension_name": "Component identity / taxonomy",
            "sprint_12_scope": now,
            "status": "PARTIAL_NOW",
            "available_evidence": "Canonical application instances have domain/role taxonomy; reusable hardware IDs remain unresolved.",
            "blocker_or_deferred_reason": "No local part/hardware source supports component_db masters.",
        },
        {
            "dimension_number": 2,
            "dimension_name": "Application / physical context",
            "sprint_12_scope": now,
            "status": "PARTIAL_NOW",
            "available_evidence": "843 JRC-derived application instances retain vehicle configuration, role and source properties.",
            "blocker_or_deferred_reason": "PL configuration subtraction, position and physical-boundary source package is absent.",
        },
        {
            "dimension_number": 3,
            "dimension_name": "Engineering loss behavior",
            "sprint_12_scope": now,
            "status": "NOT_SUPPORTED_BY_CURRENT_SOURCE",
            "available_evidence": "Only synthetic mock ABC and vehicle-roadload-derived estimates exist.",
            "blocker_or_deferred_reason": "No raw PL component ABC/force curve with established units and boundary is local.",
        },
        {
            "dimension_number": 4,
            "dimension_name": "Methodology / provenance / quality",
            "sprint_12_scope": now,
            "status": "PARTIAL_NOW",
            "available_evidence": "Existing instance provenance and a high-level provenance-interface document are present.",
            "blocker_or_deferred_reason": "Referenced PL methodology, raw source versions and analysis notebooks are missing.",
        },
        {
            "dimension_number": 5,
            "dimension_name": "Supplier / manufacturer / part-number enrichment",
            "sprint_12_scope": later,
            "status": "DEFERRED_RAG",
            "available_evidence": "",
            "blocker_or_deferred_reason": "Explicitly deferred; no identity is fabricated.",
        },
        {
            "dimension_number": 6,
            "dimension_name": "Detailed specifications / maps / datasheets",
            "sprint_12_scope": later,
            "status": "DEFERRED_RAG",
            "available_evidence": "",
            "blocker_or_deferred_reason": "Explicitly deferred to sourced research.",
        },
        {
            "dimension_number": 7,
            "dimension_name": "Document evidence / chunks / quotations / links",
            "sprint_12_scope": later,
            "status": "DEFERRED_RAG",
            "available_evidence": "",
            "blocker_or_deferred_reason": "Explicitly deferred; no retrieval pipeline implemented.",
        },
        {
            "dimension_number": 8,
            "dimension_name": "Cross-source matching / aliases / equivalence",
            "sprint_12_scope": later,
            "status": "DEFERRED_RAG",
            "available_evidence": "",
            "blocker_or_deferred_reason": "Explicitly deferred; no fuzzy identity matching allowed here.",
        },
        {
            "dimension_number": 9,
            "dimension_name": "Enriched compatibility / topology / applicability",
            "sprint_12_scope": later,
            "status": "DEFERRED_RAG",
            "available_evidence": "",
            "blocker_or_deferred_reason": "Explicitly deferred to reviewed enrichment.",
        },
        {
            "dimension_number": 10,
            "dimension_name": "Retrieval / extraction / AI confidence metadata",
            "sprint_12_scope": later,
            "status": "DEFERRED_RAG",
            "available_evidence": "",
            "blocker_or_deferred_reason": "Explicitly deferred; RAG is not implemented in Sprint 12.",
        },
    ]


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _candidate_audit(candidate: Path) -> tuple[dict[str, int], str, int]:
    uri = "file:" + candidate.resolve().as_posix() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        tables = (
            "component_db",
            "component_instance",
            "component_resolution",
            "vde_component_resolution",
            "tire_db",
        )
        counts = {
            table: int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            for table in tables
        }
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        foreign_key_issues = len(connection.execute("PRAGMA foreign_key_check").fetchall())
        return counts, quick_check, foreign_key_issues
    finally:
        connection.close()


def build_discovery_artifacts(
    root: Path | str = ROOT,
    candidate: Path | str = DEFAULT_CANDIDATE,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> DiscoveryReport:
    root_path = Path(root).resolve()
    candidate_path = Path(candidate).resolve(strict=True)
    output_path = Path(output_dir).resolve()

    before_hash = sha256_file(candidate_path)
    counts, quick_check, foreign_key_issues = _candidate_audit(candidate_path)
    inventory = _inventory_rows(root_path, counts)
    coverage = _coverage_rows()

    source_available = any(
        row["raw_numeric_component_source"] == "YES"
        and row["usable_for_component_resolution"] == "YES"
        for row in inventory
    )
    if source_available:
        raise RuntimeError(
            "A raw numeric Component source was detected; review discovery before any population step."
        )

    seed_audit = [
        {
            "entity": "component_db",
            "before_count": counts["component_db"],
            "after_count": counts["component_db"],
            "package_action": "NO_WRITE",
            "status": "BLOCKED_NO_REUSABLE_HARDWARE_SOURCE",
            "reason": "No defensible reusable hardware identity is locally available.",
        },
        {
            "entity": "component_instance",
            "before_count": counts["component_instance"],
            "after_count": counts["component_instance"],
            "package_action": "NO_WRITE",
            "status": "UNCHANGED_EXISTING_PARTIAL_BASELINE",
            "reason": "843 public-source application instances remain valid; no PL source supports additional population.",
        },
        {
            "entity": "component_resolution",
            "before_count": counts["component_resolution"],
            "after_count": counts["component_resolution"],
            "package_action": "NO_WRITE",
            "status": "BLOCKED_NO_RAW_COMPONENT_LOSS_SOURCE",
            "reason": "Synthetic fixtures, default priors and whole-vehicle roadload splits are prohibited.",
        },
        {
            "entity": "vde_component_resolution",
            "before_count": counts["vde_component_resolution"],
            "after_count": counts["vde_component_resolution"],
            "package_action": "NO_WRITE",
            "status": "UNCHANGED_NO_EXACT_ADOPTION_EVIDENCE",
            "reason": "No exact component-resolution-to-VDE adoption evidence exists.",
        },
        {
            "entity": "tire_db",
            "before_count": counts["tire_db"],
            "after_count": counts["tire_db"],
            "package_action": "NO_WRITE",
            "status": "UNCHANGED_SEPARATE_DOMAIN",
            "reason": "Tire knowledge remains separate from component_db.",
        },
    ]

    _write_csv(output_path / "COMPONENT_SOURCE_INVENTORY.csv", INVENTORY_FIELDS, inventory)
    _write_csv(output_path / "COMPONENT_KNOWLEDGE_COVERAGE.csv", COVERAGE_FIELDS, coverage)
    _write_csv(output_path / "COMPONENT_SEED_AUDIT.csv", SEED_AUDIT_FIELDS, seed_audit)
    _write_csv(output_path / "COMPONENT_REFERENCE_POPULATIONS.csv", REFERENCE_FIELDS, [])
    _write_csv(output_path / "COMPONENT_OUTLIER_REVIEW.csv", OUTLIER_FIELDS, [])

    after_hash = sha256_file(candidate_path)
    if before_hash != after_hash:
        raise RuntimeError("Canonical candidate changed during read-only Component discovery.")

    return DiscoveryReport(
        component_source_data_available=source_available,
        candidate_path=str(candidate_path),
        candidate_sha256_before=before_hash,
        candidate_sha256_after=after_hash,
        quick_check=quick_check,
        foreign_key_issues=foreign_key_issues,
        component_db_count=counts["component_db"],
        component_instance_count=counts["component_instance"],
        component_resolution_count=counts["component_resolution"],
        vde_component_resolution_count=counts["vde_component_resolution"],
        tire_db_count=counts["tire_db"],
        inventory_rows=len(inventory),
        implemented_dimensions=sum(row["status"] == "IMPLEMENTED_NOW" for row in coverage),
        partial_dimensions=sum(row["status"] == "PARTIAL_NOW" for row in coverage),
        reference_population_rows=0,
        outlier_review_rows=0,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_discovery_artifacts(args.root, args.candidate, args.output_dir)
    for key, value in asdict(report).items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
