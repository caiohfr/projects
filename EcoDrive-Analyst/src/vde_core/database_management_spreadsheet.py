from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
import hashlib
import html
import io
import math
from pathlib import Path
import tempfile
from typing import Any
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

from src.vde_core import db as db_module
from src.vde_core.database_management_contract import (
    ChangeCommand,
    EntityType,
    ValidationIssue,
    normalize_entity_type,
)
from src.vde_core.database_management_service import get_record, preview_change
from src.vde_core.vde_request_parser import _read_xlsx_workbook


TEMPLATE_VERSION = "7E.1"


@dataclass(frozen=True)
class SpreadsheetSheetContract:
    name: str
    domain: str | None = None


@dataclass(frozen=True)
class SpreadsheetTemplateContract:
    entity_type: EntityType
    filename: str
    columns: tuple[str, ...]
    required_on_create: tuple[str, ...]
    numeric_fields: frozenset[str] = frozenset()
    integer_fields: frozenset[str] = frozenset()
    boolean_fields: frozenset[str] = frozenset()
    sheets: tuple[SpreadsheetSheetContract, ...] = ()


@dataclass(frozen=True)
class SpreadsheetImportRow:
    sheet: str
    row_number: int
    entity_type: str
    domain: str | None
    action: str
    match_method: str
    record_id: str | None
    status: str
    payload: dict[str, Any] = field(default_factory=dict)
    current_record: dict[str, Any] = field(default_factory=dict)
    field_diff: tuple[dict[str, Any], ...] = ()
    issues: tuple[ValidationIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpreadsheetImportPreview:
    batch_id: str
    entity_type: str
    filename: str
    template_version: str
    rows: tuple[SpreadsheetImportRow, ...]
    issues: tuple[ValidationIssue, ...] = ()
    unknown_columns: tuple[str, ...] = ()

    @property
    def counts(self) -> dict[str, int]:
        counts = {"inserted": 0, "updated": 0, "skipped": 0, "invalid": 0}
        for row in self.rows:
            if row.status == "READY" and row.action == "CREATE":
                counts["inserted"] += 1
            elif row.status == "READY" and row.action == "UPDATE":
                counts["updated"] += 1
            elif row.status == "SKIPPED":
                counts["skipped"] += 1
            else:
                counts["invalid"] += 1
        return counts

    @property
    def can_stage(self) -> bool:
        return any(row.status == "READY" for row in self.rows) and not any(
            issue.severity == "ERROR" for issue in self.issues
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["counts"] = self.counts
        payload["can_stage"] = self.can_stage
        return payload


_VDE_FIELDS = (
    "source_name", "source_record_id", "legislation", "category", "make", "model", "year", "notes",
    "engine_type", "engine_model", "engine_size_l", "engine_aspiration", "transmission_type",
    "transmission_model", "drive_type", "cycle_name", "cycle_source", "mass_kg", "test_mass_kg",
    "test_mass_low_kg", "test_mass_high_kg", "test_mass_basis", "inertia_class", "cda_m2",
    "weight_dist_fr_pct", "payload_kg", "gvwr_kg", "gcwr_kg", "trailer_mass_kg", "front_pressure_psi",
    "rear_pressure_psi", "coast_A_N", "coast_B_N_per_kph", "coast_C_N_per_kph2", "trans_A_coef_N",
    "trans_B_coef_Npkph", "trans_C_coef_Npkph2", "brake_A_coef_N", "brake_B_coef_Npkph",
    "brake_C_coef_Npkph2", "aero_C_coef_Npkph2", "rrc_N_per_kN", "front_tire_id", "rear_tire_id",
    "tire_load_mass_basis",
)
_FUEL_FIELDS = (
    "source_name", "source_record_id", "vde_id", "electrification", "fuel_type", "method_note",
    "eta_pt_est", "bev_eff_drive", "utility_factor_pct", "engine_max_power_kw", "engine_rpm_max_power",
    "engine_max_torque_nm", "engine_rpm_max_torque", "gear_count", "final_drive_ratio",
    "battery_capacity_kwh", "battery_usable_kwh", "bms_discharge_limit_kw", "bms_regen_limit_kw",
    "bms_note", "ambient_temp_c", "ac_on", "tire_front_psi", "tire_rear_psi", "scenario_payload_kg",
    "energy_basis", "engine_method", "engine_version", "assumptions_json", "provenance_json",
    "label_program", "label_version_year", "label_vehicle_category", "label_cycle_set", "label_class",
)
_TIRE_FIELDS = (
    "source_name", "source_record_id", "tire_test_code", "manufacturer", "model", "size_code",
    "load_index", "speed_rating", "notes", "standard_family", "standard_version", "test_method",
    "test_source", "test_date", "test_mileage_km", "is_tested_value", "is_estimated_value",
    "break_in_distance_km", "is_broken_in", "test_temperature_c", "reference_temperature_c",
    "temperature_correction_applied", "rr_n_per_kn", "rr_value_source_note", "sae_a", "sae_b", "sae_c",
    "sae_alpha", "sae_beta", "pressure_unit", "load_unit", "speed_unit", "force_unit", "iso_rrc_n_per_kn",
    "iso_test_pressure_kpa", "iso_test_load_n", "iso_test_speed_kph", "iso_rolling_resistance_force_n",
    "iso_condition_notes", "calculation_mode", "smerf",
)
_COMPONENT_FIELDS = (
    "source_name", "source_record_id", "component_code", "component_name", "source_reference",
    "hardware_reference", "component_type", "component_position", "driveline_architecture", "physical_boundary",
    "configuration_from", "configuration_to", "test_condition_type", "test_method", "net_bridge_eligible",
    "equivalent_A_N", "equivalent_B_N_per_kph", "equivalent_C_N_per_kph2", "loss_pct",
    "residual_torque_front_nm", "residual_torque_rear_nm", "wheel_radius_m", "notes",
)

_VDE_NUMERIC = frozenset(
    field for field in _VDE_FIELDS if field not in {
        "source_name", "source_record_id", "legislation", "category", "make", "model", "notes", "engine_type",
        "engine_model", "engine_aspiration", "transmission_type", "transmission_model", "drive_type", "cycle_name",
        "cycle_source", "test_mass_basis", "tire_load_mass_basis",
    }
)
_FUEL_NUMERIC = frozenset(
    field for field in _FUEL_FIELDS if field not in {
        "source_name", "source_record_id", "electrification", "fuel_type", "method_note", "bms_note",
        "energy_basis", "engine_method", "engine_version", "assumptions_json", "provenance_json", "label_program",
        "label_vehicle_category", "label_cycle_set", "label_class",
    }
)
_TIRE_NUMERIC = frozenset(
    field for field in _TIRE_FIELDS if field not in {
        "source_name", "source_record_id", "tire_test_code", "manufacturer", "model", "size_code", "load_index",
        "speed_rating", "notes", "standard_family", "standard_version", "test_method", "test_source", "test_date",
        "rr_value_source_note", "pressure_unit", "load_unit", "speed_unit", "force_unit", "iso_condition_notes",
        "calculation_mode",
    }
)
_COMPONENT_NUMERIC = frozenset(
    {
        "equivalent_A_N", "equivalent_B_N_per_kph", "equivalent_C_N_per_kph2", "loss_pct",
        "residual_torque_front_nm", "residual_torque_rear_nm", "wheel_radius_m",
    }
)

_CONTRACTS = {
    EntityType.VDE: SpreadsheetTemplateContract(
        EntityType.VDE,
        "VDE template.xlsx",
        ("internal_id", "record_origin", *_VDE_FIELDS),
        ("record_origin", "legislation", "category", "make", "model", "mass_kg"),
        numeric_fields=_VDE_NUMERIC,
        integer_fields=frozenset({"internal_id", "year", "front_tire_id", "rear_tire_id"}),
        sheets=(SpreadsheetSheetContract("VDE"),),
    ),
    EntityType.FUEL_CONSUMPTION: SpreadsheetTemplateContract(
        EntityType.FUEL_CONSUMPTION,
        "Fuel Consumption template.xlsx",
        ("internal_id", "record_origin", *_FUEL_FIELDS),
        ("record_origin", "vde_id", "electrification"),
        numeric_fields=_FUEL_NUMERIC,
        integer_fields=frozenset({"internal_id", "vde_id", "engine_rpm_max_power", "engine_rpm_max_torque", "gear_count", "label_version_year"}),
        boolean_fields=frozenset({"ac_on"}),
        sheets=(SpreadsheetSheetContract("Fuel Consumption"),),
    ),
    EntityType.TIRE: SpreadsheetTemplateContract(
        EntityType.TIRE,
        "Tire template.xlsx",
        ("internal_id", "record_origin", *_TIRE_FIELDS),
        ("record_origin", "tire_test_code", "manufacturer", "model", "standard_family", "rr_n_per_kn"),
        numeric_fields=_TIRE_NUMERIC,
        integer_fields=frozenset({"internal_id"}),
        boolean_fields=frozenset({"is_tested_value", "is_estimated_value", "is_broken_in", "temperature_correction_applied"}),
        sheets=(SpreadsheetSheetContract("Tires"),),
    ),
    EntityType.COMPONENT: SpreadsheetTemplateContract(
        EntityType.COMPONENT,
        "Components template.xlsx",
        ("internal_id", "record_origin", *_COMPONENT_FIELDS),
        ("record_origin", "component_code", "component_name", "source_name", "notes"),
        numeric_fields=_COMPONENT_NUMERIC,
        integer_fields=frozenset({"internal_id"}),
        boolean_fields=frozenset({"net_bridge_eligible"}),
        sheets=(
            SpreadsheetSheetContract("Transmission", "transmission"),
            SpreadsheetSheetContract("Brake", "brake"),
            SpreadsheetSheetContract("Axle & Hubs", "axle_hubs"),
            SpreadsheetSheetContract("Parasitic", "parasitic"),
        ),
    ),
}

_TABLES = {
    EntityType.VDE: "vde_db",
    EntityType.FUEL_CONSUMPTION: "fuelcons_db",
    EntityType.TIRE: "tire_roadload_db",
    EntityType.COMPONENT: "component_db",
}
_NATURAL_KEYS = {
    EntityType.TIRE: "tire_test_code",
    EntityType.COMPONENT: "component_code",
}
_SIMILARITY_FIELDS = {
    EntityType.VDE: ("make", "model"),
    EntityType.FUEL_CONSUMPTION: ("source_name", "source_record_id"),
    EntityType.TIRE: ("tire_test_code",),
    EntityType.COMPONENT: ("component_code", "component_name"),
}


def spreadsheet_template_contract(entity_type: EntityType | str) -> SpreadsheetTemplateContract:
    return _CONTRACTS[normalize_entity_type(entity_type)]


def generate_controlled_template(
    entity_type: EntityType | str,
    *,
    rows_by_sheet: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[str, bytes]:
    contract = spreadsheet_template_contract(entity_type)
    supplied_rows = dict(rows_by_sheet or {})
    workbook_sheets: list[tuple[str, list[list[Any]]]] = [
        (
            "Instructions",
            [
                ["EcoDrive Database Management controlled template", contract.entity_type.value],
                ["Template version", TEMPLATE_VERSION],
                ["Workflow", "Fill or update rows, upload, review the diff, then explicitly commit."],
                ["Matching", "internal_id first; otherwise exact source_name + source_record_id."],
                ["Blank update cells", "No change. Explicit zero is preserved."],
                ["Deletion", "Rows absent from this workbook are never deleted."],
                ["Derived values", "Spreadsheet formulas are not a source of truth."],
            ],
        )
    ]
    for sheet in contract.sheets:
        rows = [list(contract.columns)]
        for payload in supplied_rows.get(sheet.name, []):
            rows.append([dict(payload or {}).get(column) for column in contract.columns])
        workbook_sheets.append((sheet.name, rows))
    return contract.filename, _build_xlsx(workbook_sheets)


def preview_spreadsheet_import(
    source: bytes | bytearray | memoryview | str | Path,
    entity_type: EntityType | str,
    *,
    filename: str = "upload.xlsx",
) -> SpreadsheetImportPreview:
    entity = normalize_entity_type(entity_type)
    contract = spreadsheet_template_contract(entity)
    batch_id = f"{entity.value.lower()}-{_source_digest(source)[:16]}"
    issues: list[ValidationIssue] = []
    try:
        workbook = _read_workbook_source(source)
    except (BadZipFile, KeyError, OSError, ValueError) as exc:
        return SpreadsheetImportPreview(
            batch_id,
            entity.value,
            str(filename or "upload.xlsx"),
            "",
            (),
            (ValidationIssue("ERROR", "workbook_invalid", f"Unable to read workbook: {exc}"),),
        )

    if _workbook_contains_formulas(source):
        issues.append(
            ValidationIssue(
                "ERROR",
                "spreadsheet_formulas_not_supported",
                "Spreadsheet formulas cannot be used as database source values. Replace them with explicit values before upload.",
            )
        )

    version = _template_version(workbook)
    if version and version != TEMPLATE_VERSION:
        issues.append(
            ValidationIssue(
                "WARNING",
                "template_version_mismatch",
                f"Workbook template version {version} differs from expected {TEMPLATE_VERSION}.",
            )
        )
    expected_sheets = {sheet.name for sheet in contract.sheets}
    available_sheets = set(workbook.get("sheet_names") or ())
    missing_sheets = sorted(expected_sheets - available_sheets)
    if missing_sheets:
        issues.append(
            ValidationIssue("ERROR", "required_sheet_missing", "Missing required sheet(s): " + ", ".join(missing_sheets))
        )

    parsed_rows: list[dict[str, Any]] = []
    unknown_columns: set[str] = set()
    for sheet in contract.sheets:
        if sheet.name not in available_sheets:
            continue
        rows, sheet_unknown, sheet_issues = _parse_sheet_rows(
            workbook["sheets"].get(sheet.name, []), contract, sheet
        )
        parsed_rows.extend(rows)
        unknown_columns.update(sheet_unknown)
        issues.extend(sheet_issues)

    duplicate_keys = _duplicate_upload_keys(parsed_rows)
    previews = tuple(
        _preview_import_row(entity, contract, row, duplicate_keys)
        for row in parsed_rows
    )
    return SpreadsheetImportPreview(
        batch_id=batch_id,
        entity_type=entity.value,
        filename=str(filename or "upload.xlsx"),
        template_version=version,
        rows=previews,
        issues=tuple(issues),
        unknown_columns=tuple(sorted(unknown_columns)),
    )


def stage_commands_from_import(
    preview: SpreadsheetImportPreview,
    *,
    confirm_unknown_columns: bool = False,
) -> tuple[dict[str, Any], ...]:
    if preview.unknown_columns and not confirm_unknown_columns:
        raise ValueError("Unknown columns must be explicitly acknowledged before staging the import.")
    if any(issue.severity == "ERROR" for issue in preview.issues):
        raise ValueError("Workbook-level validation errors must be resolved before staging.")
    commands: list[dict[str, Any]] = []
    for row in preview.rows:
        if row.status != "READY":
            continue
        command = {
            "entity_type": row.entity_type,
            "action": row.action,
            "record_id": row.record_id,
            "record_origin": row.current_record.get("record_origin") or row.payload.get("record_origin"),
            "current_record": deepcopy(row.current_record),
            "payload": deepcopy(row.payload),
            "component_domain": row.domain,
            "import_batch_id": preview.batch_id,
            "import_filename": preview.filename,
            "import_sheet": row.sheet,
            "import_row": row.row_number,
            "import_counts": deepcopy(preview.counts),
        }
        commands.append(command)
    return tuple(commands)


def _parse_sheet_rows(
    worksheet_rows: list[list[object]],
    contract: SpreadsheetTemplateContract,
    sheet: SpreadsheetSheetContract,
) -> tuple[list[dict[str, Any]], set[str], list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    if not worksheet_rows:
        return [], set(), [ValidationIssue("ERROR", "header_missing", f"Sheet {sheet.name} is empty.")]
    header_index = next((index for index, row in enumerate(worksheet_rows[:10]) if any(not _is_blank(value) for value in row)), None)
    if header_index is None:
        return [], set(), [ValidationIssue("ERROR", "header_missing", f"Sheet {sheet.name} has no header row.")]
    raw_headers = worksheet_rows[header_index]
    canonical_by_key = {_header_key(column): column for column in contract.columns}
    header_columns: list[str | None] = []
    unknown: set[str] = set()
    seen: set[str] = set()
    for raw_header in raw_headers:
        text = str(raw_header or "").strip()
        if not text:
            header_columns.append(None)
            continue
        canonical = canonical_by_key.get(_header_key(text))
        if canonical is None:
            unknown.add(text)
            header_columns.append(None)
            continue
        if canonical in seen:
            issues.append(ValidationIssue("ERROR", "duplicate_column", f"Column {canonical!r} appears more than once in {sheet.name}.", canonical))
        seen.add(canonical)
        header_columns.append(canonical)
    missing_headers = [column for column in ("internal_id", "record_origin") if column not in seen]
    if missing_headers:
        issues.append(
            ValidationIssue("ERROR", "required_column_missing", f"Sheet {sheet.name} is missing column(s): {', '.join(missing_headers)}")
        )

    parsed: list[dict[str, Any]] = []
    for offset, row in enumerate(worksheet_rows[header_index + 1 :], start=header_index + 2):
        raw_payload = {
            column: row[index] if index < len(row) else None
            for index, column in enumerate(header_columns)
            if column is not None
        }
        if all(_is_blank(value) for value in raw_payload.values()):
            continue
        normalized: dict[str, Any] = {}
        row_issues: list[ValidationIssue] = []
        for column, value in raw_payload.items():
            if _is_blank(value):
                normalized[column] = None
                continue
            try:
                normalized[column] = _normalize_cell(contract, column, value)
            except ValueError as exc:
                normalized[column] = None
                row_issues.append(ValidationIssue("ERROR", "cell_invalid", str(exc), column))
        parsed.append(
            {
                "sheet": sheet.name,
                "row_number": offset,
                "domain": sheet.domain,
                "values": normalized,
                "issues": row_issues,
            }
        )
    return parsed, unknown, issues


def _preview_import_row(
    entity: EntityType,
    contract: SpreadsheetTemplateContract,
    parsed_row: dict[str, Any],
    duplicate_keys: set[tuple[Any, ...]],
) -> SpreadsheetImportRow:
    values = dict(parsed_row.get("values") or {})
    row_issues = list(parsed_row.get("issues") or [])
    domain = parsed_row.get("domain")
    identity_key = _upload_identity_key(values, domain)
    if identity_key and identity_key in duplicate_keys:
        row_issues.append(
            ValidationIssue("ERROR", "duplicate_upload_identity", "The same internal/source identity appears more than once in this upload.")
        )

    current, match_method, match_issues = _match_existing_record(entity, values, domain)
    row_issues.extend(match_issues)
    action = "UPDATE" if current else "CREATE"
    record_id = str(current["id"]) if current else None

    if action == "CREATE":
        for required in _required_fields_for_create(contract, domain):
            if _is_blank(values.get(required)):
                row_issues.append(
                    ValidationIssue("ERROR", "required_value_missing", f"{required} is required for insert.", required)
                )
    payload = {
        field: value
        for field, value in values.items()
        if field != "internal_id" and not _is_blank(value)
    }
    if entity is EntityType.COMPONENT:
        payload["domain"] = domain
    if current:
        uploaded_origin = payload.pop("record_origin", None)
        if uploaded_origin and uploaded_origin != current.get("record_origin"):
            row_issues.append(
                ValidationIssue("WARNING", "record_origin_ignored", "record_origin is immutable during update and was ignored.", "record_origin")
            )
        payload = {field: value for field, value in payload.items() if current.get(field) != value}
    else:
        natural_issue = _natural_key_conflict(entity, payload, domain)
        if natural_issue:
            row_issues.append(natural_issue)
        row_issues.extend(_similarity_warnings(entity, payload, domain))

    command_preview = preview_change(
        ChangeCommand(
            entity_type=entity,
            action=action,
            record_id=record_id,
            record_origin=current.get("record_origin") if current else payload.get("record_origin"),
            current_record=current,
            payload=payload,
            reason="Spreadsheet import preview" if action == "UPDATE" else None,
        )
    )
    row_issues.extend(command_preview.validation_issues)
    field_diff = tuple(asdict(item) for item in command_preview.field_diff)
    has_error = any(issue.severity == "ERROR" for issue in row_issues)
    status = "INVALID" if has_error else ("SKIPPED" if action == "UPDATE" and not field_diff else "READY")
    return SpreadsheetImportRow(
        sheet=str(parsed_row.get("sheet") or ""),
        row_number=int(parsed_row.get("row_number") or 0),
        entity_type=entity.value,
        domain=domain,
        action=action,
        match_method=match_method,
        record_id=record_id,
        status=status,
        # Keep the caller payload here. ``preview_change`` adds lifecycle defaults
        # such as record_status/is_active; feeding those defaults back through a
        # second preview would incorrectly treat them as direct immutable edits.
        payload=payload,
        current_record=current,
        field_diff=field_diff,
        issues=tuple(row_issues),
    )


def _match_existing_record(
    entity: EntityType,
    values: dict[str, Any],
    domain: str | None,
) -> tuple[dict[str, Any], str, list[ValidationIssue]]:
    internal_id = values.get("internal_id")
    if not _is_blank(internal_id):
        current = get_record(entity, int(internal_id), component_domain=domain)
        if current is None:
            return {}, "INTERNAL_ID", [ValidationIssue("ERROR", "internal_id_not_found", f"internal_id {internal_id} was not found.", "internal_id")]
        return current, "INTERNAL_ID", []

    source_name = values.get("source_name")
    source_record_id = values.get("source_record_id")
    if _is_blank(source_name) or _is_blank(source_record_id):
        return {}, "INSERT", []
    table = _TABLES[entity]
    clauses = ["source_name=?", "source_record_id=?"]
    params: list[Any] = [source_name, source_record_id]
    if entity is EntityType.COMPONENT:
        clauses.append("domain=?")
        params.append(domain)
    rows = db_module.fetchall(
        f"SELECT id FROM {table} WHERE {' AND '.join(clauses)} ORDER BY id",
        tuple(params),
    )
    if len(rows) > 1:
        return {}, "SOURCE_IDENTITY", [
            ValidationIssue("ERROR", "source_identity_ambiguous", "Exact source identity matches more than one existing record.")
        ]
    if rows:
        return get_record(entity, rows[0]["id"], component_domain=domain) or {}, "SOURCE_IDENTITY", []
    return {}, "INSERT", []


def _duplicate_upload_keys(rows: list[dict[str, Any]]) -> set[tuple[Any, ...]]:
    keys = [_upload_identity_key(dict(row.get("values") or {}), row.get("domain")) for row in rows]
    return {key for key in keys if key is not None and keys.count(key) > 1}


def _upload_identity_key(values: dict[str, Any], domain: str | None) -> tuple[Any, ...] | None:
    if not _is_blank(values.get("internal_id")):
        return ("id", domain, values["internal_id"])
    if not _is_blank(values.get("source_name")) and not _is_blank(values.get("source_record_id")):
        return ("source", domain, values["source_name"], values["source_record_id"])
    return None


def _natural_key_conflict(entity: EntityType, payload: dict[str, Any], domain: str | None) -> ValidationIssue | None:
    field = _NATURAL_KEYS.get(entity)
    value = payload.get(field) if field else None
    if not field or _is_blank(value):
        return None
    table = _TABLES[entity]
    clauses = [f"{field}=?"]
    params: list[Any] = [value]
    if entity is EntityType.COMPONENT:
        clauses.append("domain=?")
        params.append(domain)
    match = db_module.fetchone(f"SELECT id FROM {table} WHERE {' AND '.join(clauses)} LIMIT 1", tuple(params))
    if not match:
        return None
    return ValidationIssue(
        "ERROR",
        "natural_key_conflict",
        f"{field} already exists. Supply internal_id or the exact source identity to update it.",
        field,
    )


def _similarity_warnings(entity: EntityType, payload: dict[str, Any], domain: str | None) -> list[ValidationIssue]:
    fields = _SIMILARITY_FIELDS[entity]
    proposed = " ".join(str(payload.get(field) or "").strip() for field in fields).strip().lower()
    if len(proposed) < 4:
        return []
    table = _TABLES[entity]
    clauses = []
    params: tuple[Any, ...] = ()
    if entity is EntityType.COMPONENT:
        clauses.append("domain=?")
        params = (domain,)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    rows = db_module.fetchall(f"SELECT id, {', '.join(fields)} FROM {table}{where} LIMIT 500", params)
    best_ratio = 0.0
    best_id = None
    for row in rows:
        candidate = " ".join(str(row.get(field) or "").strip() for field in fields).strip().lower()
        ratio = SequenceMatcher(None, proposed, candidate).ratio() if candidate else 0.0
        if ratio > best_ratio:
            best_ratio, best_id = ratio, row.get("id")
    if best_ratio < 0.88:
        return []
    return [
        ValidationIssue(
            "WARNING",
            "similar_record",
            f"Text is similar to existing record {best_id} ({best_ratio:.0%}); matching remains insert-only without exact identity.",
        )
    ]


def _required_fields_for_create(contract: SpreadsheetTemplateContract, domain: str | None) -> tuple[str, ...]:
    required = list(contract.required_on_create)
    if contract.entity_type is EntityType.COMPONENT:
        required.extend(("equivalent_A_N", "equivalent_B_N_per_kph", "equivalent_C_N_per_kph2"))
        if domain == "transmission":
            required.append("loss_pct")
        if domain == "brake":
            required.extend(("residual_torque_front_nm", "residual_torque_rear_nm", "wheel_radius_m"))
    return tuple(required)


def _normalize_cell(contract: SpreadsheetTemplateContract, field: str, value: Any) -> Any:
    if field == "record_origin":
        return str(value).strip().upper()
    if field in contract.boolean_fields:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)) and value in (0, 1):
            return int(value)
        text = str(value).strip().lower()
        if text in {"true", "yes", "y", "sim", "1"}:
            return 1
        if text in {"false", "no", "n", "nao", "não", "0"}:
            return 0
        raise ValueError(f"{field} must be a boolean (0/1 or yes/no).")
    if field in contract.numeric_fields or field in contract.integer_fields:
        numeric = _to_finite_number(value, field)
        if field in contract.integer_fields:
            if not float(numeric).is_integer():
                raise ValueError(f"{field} must be an integer.")
            return int(numeric)
        return float(numeric)
    return str(value).strip()


def _to_finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric.")
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip()
        if "," in text and "." not in text:
            text = text.replace(",", ".")
        try:
            number = float(text)
        except ValueError as exc:
            raise ValueError(f"{field} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite.")
    return number


def _read_workbook_source(source: bytes | bytearray | memoryview | str | Path) -> dict[str, Any]:
    if isinstance(source, (str, Path)):
        return _read_xlsx_workbook(source)
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as handle:
            handle.write(bytes(source))
            path = Path(handle.name)
        return _read_xlsx_workbook(path)
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def _source_digest(source: bytes | bytearray | memoryview | str | Path) -> str:
    return hashlib.sha256(_source_bytes(source)).hexdigest()


def _source_bytes(source: bytes | bytearray | memoryview | str | Path) -> bytes:
    return Path(source).read_bytes() if isinstance(source, (str, Path)) else bytes(source)


def _workbook_contains_formulas(source: bytes | bytearray | memoryview | str | Path) -> bool:
    with ZipFile(io.BytesIO(_source_bytes(source))) as archive:
        for name in archive.namelist():
            if name.startswith("xl/worksheets/") and name.endswith(".xml") and b"<f" in archive.read(name):
                return True
    return False


def _template_version(workbook: dict[str, Any]) -> str:
    rows = list(dict(workbook.get("sheets") or {}).get("Instructions") or [])
    for row in rows:
        if row and str(row[0] or "").strip().lower() == "template version":
            return str(row[1] if len(row) > 1 else "").strip()
    return ""


def _header_key(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _column_letter(index: int) -> str:
    letters: list[str] = []
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return "".join(reversed(letters))


def _cell_xml(reference: str, value: Any, style_id: int) -> str:
    style = f' s="{style_id}"' if style_id else ""
    if value is None:
        return f'<c r="{reference}"{style}/>'
    if isinstance(value, bool):
        return f'<c r="{reference}" t="b"{style}><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{reference}"{style}><v>{value}</v></c>'
    return f'<c r="{reference}" t="inlineStr"{style}><is><t>{html.escape(str(value))}</t></is></c>'


def _worksheet_xml(rows: list[list[Any]], *, data_sheet: bool) -> str:
    row_xml: list[str] = []
    max_columns = max((len(row) for row in rows), default=1)
    widths = [12] * max_columns
    for row_index, row in enumerate(rows, start=1):
        cells: list[str] = []
        for column_index, value in enumerate(row):
            widths[column_index] = min(40, max(widths[column_index], len(str(value or "")) + 2))
            style_id = 1 if data_sheet and row_index == 1 else (2 if not data_sheet and column_index == 0 else 3)
            cells.append(_cell_xml(f"{_column_letter(column_index)}{row_index}", value, style_id))
        row_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    columns = "".join(
        f'<col min="{index + 1}" max="{index + 1}" width="{width}" customWidth="1"/>'
        for index, width in enumerate(widths)
    )
    freeze = '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>' if data_sheet else ""
    auto_filter = f'<autoFilter ref="A1:{_column_letter(max_columns - 1)}{max(len(rows), 1)}"/>' if data_sheet else ""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"{freeze}<cols>{columns}</cols><sheetData>{''.join(row_xml)}</sheetData>{auto_filter}</worksheet>"
    )


def _styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="3">
    <font><sz val="11"/><color rgb="FF101828"/><name val="Calibri"/></font>
    <font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/></font>
    <font><b/><sz val="11"/><color rgb="FF1F3B63"/><name val="Calibri"/></font>
  </fonts>
  <fills count="4">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F3B63"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFEFF6FF"/></patternFill></fill>
  </fills>
  <borders count="2"><border><left/><right/><top/><bottom/><diagonal/></border><border><left style="thin"><color rgb="FFD0D5DD"/></left><right style="thin"><color rgb="FFD0D5DD"/></right><top style="thin"><color rgb="FFD0D5DD"/></top><bottom style="thin"><color rgb="FFD0D5DD"/></bottom><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="4">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="2" fillId="3" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1"/>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""


def _build_xlsx(sheets: list[tuple[str, list[list[Any]]]]) -> bytes:
    sheet_entries: list[str] = []
    relationships: list[str] = []
    content_overrides: list[str] = []
    worksheet_payloads: dict[str, str] = {}
    for index, (name, rows) in enumerate(sheets, start=1):
        escaped_name = html.escape(str(name), quote=True)
        sheet_entries.append(f'<sheet name="{escaped_name}" sheetId="{index}" r:id="rId{index}"/>')
        relationships.append(
            f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>'
        )
        content_overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
        worksheet_payloads[f"xl/worksheets/sheet{index}.xml"] = _worksheet_xml(rows, data_sheet=index > 1)

    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            + "".join(content_overrides)
            + "</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f'<bookViews><workbookView activeTab="0"/></bookViews><sheets>{"".join(sheet_entries)}</sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(relationships)
            + "</Relationships>",
        )
        archive.writestr("xl/styles.xml", _styles_xml())
        for path, payload in worksheet_payloads.items():
            archive.writestr(path, payload)
    return buffer.getvalue()


__all__ = [
    "SpreadsheetImportPreview",
    "SpreadsheetImportRow",
    "SpreadsheetTemplateContract",
    "TEMPLATE_VERSION",
    "generate_controlled_template",
    "preview_spreadsheet_import",
    "spreadsheet_template_contract",
    "stage_commands_from_import",
]
