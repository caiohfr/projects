from __future__ import annotations

from datetime import datetime, timezone
import re
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

from src.vde_core.vde_request_contract import (
    FIELD_KEY_ALIASES,
    VDE_REQUEST_SCHEMA_VERSION,
    is_blank,
    normalize_domain,
    normalize_template_proposal_type,
    resolve_effective_baseline,
)


_XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_REQUESTED_COLUMN_RE = re.compile(r"^Requested\s*#\s*(\d+)$", re.IGNORECASE)
_PROPOSAL_DOMAINS = {"mass", "aero", "tire", "transmission", "brake", "axle_hubs", "parasitic"}
_DOMAIN_PARENT_MAP = {
    "trailer": "mass",
}


def _issue(
    severity: str,
    code: str,
    message: str,
    *,
    sheet: str | None = None,
    row: int | None = None,
    field_key: str | None = None,
    source_column: str | None = None,
) -> dict:
    payload = {
        "severity": severity,
        "code": code,
        "message": message,
    }
    if sheet is not None:
        payload["sheet"] = sheet
    if row is not None:
        payload["row"] = row
    if field_key is not None:
        payload["field_key"] = field_key
    if source_column is not None:
        payload["source_column"] = source_column
    return payload


def _normalize_label(value) -> str:
    if is_blank(value):
        return ""
    text = str(value).strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[\u2010-\u2015]+", "-", text)
    text = re.sub(r"[_/]+", " ", text)
    text = re.sub(r"[^a-z0-9#\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _column_letters_to_index(letters: str) -> int:
    value = 0
    for char in letters.upper():
        value = (value * 26) + (ord(char) - ord("A") + 1)
    return value - 1


def _cell_ref_to_index(ref: str) -> tuple[int, int]:
    match = re.fullmatch(r"([A-Z]+)(\d+)", str(ref or "").upper())
    if not match:
        raise ValueError(f"Unsupported cell reference '{ref}'.")
    return int(match.group(2)) - 1, _column_letters_to_index(match.group(1))


def _parse_numeric(raw: str):
    numeric = float(raw)
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _read_cell_value(cell, shared_strings: list[str]):
    cell_type = cell.attrib.get("t")
    value_node = cell.find(f"{_XML_NS}v")
    inline_node = cell.find(f"{_XML_NS}is")
    if cell_type == "inlineStr" and inline_node is not None:
        return "".join(node.text or "" for node in inline_node.iter(f"{_XML_NS}t"))
    if cell_type == "s" and value_node is not None and value_node.text is not None:
        index = int(value_node.text)
        return shared_strings[index] if 0 <= index < len(shared_strings) else value_node.text
    if cell_type == "b" and value_node is not None and value_node.text is not None:
        return value_node.text == "1"
    if value_node is None or value_node.text is None:
        return None
    raw = value_node.text
    if cell_type == "str":
        return raw
    try:
        return _parse_numeric(raw)
    except Exception:
        return raw


def _sheet_rows_from_xml(xml_bytes: bytes, shared_strings: list[str]) -> list[list[object]]:
    root = ET.fromstring(xml_bytes)
    rows_by_index: dict[int, dict[int, object]] = {}
    max_col = -1
    for row in root.findall(f".//{_XML_NS}sheetData/{_XML_NS}row"):
        row_index = int(row.attrib.get("r", "1")) - 1
        row_values = rows_by_index.setdefault(row_index, {})
        for cell in row.findall(f"{_XML_NS}c"):
            _, column_index = _cell_ref_to_index(cell.attrib.get("r", "A1"))
            row_values[column_index] = _read_cell_value(cell, shared_strings)
            max_col = max(max_col, column_index)
    if max_col < 0:
        return []
    max_row = max(rows_by_index.keys(), default=-1)
    rows: list[list[object]] = []
    for row_index in range(max_row + 1):
        source = rows_by_index.get(row_index, {})
        rows.append([source.get(column_index) for column_index in range(max_col + 1)])
    return rows


def _read_xlsx_workbook(path: str | Path) -> dict:
    workbook_path = Path(path)
    with ZipFile(workbook_path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{_XML_NS}si"):
                shared_strings.append("".join(node.text or "" for node in item.iter(f"{_XML_NS}t")))

        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        rels_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels_root
        }
        sheets: dict[str, list[list[object]]] = {}
        sheet_order: list[str] = []
        for sheet in workbook_root.find(f"{_XML_NS}sheets") or []:
            name = str(sheet.attrib.get("name") or "")
            rel_id = sheet.attrib.get(f"{_REL_NS}id")
            target = rel_map.get(rel_id or "")
            if not target:
                continue
            if not target.startswith("xl/"):
                target = f"xl/{target}"
            sheet_order.append(name)
            sheets[name] = _sheet_rows_from_xml(archive.read(target), shared_strings)
    return {
        "path": workbook_path,
        "sheet_names": sheet_order,
        "sheets": sheets,
    }


def _find_header_row(rows: list[list[object]], required_headers: list[str], *, request_mode: bool = False):
    normalized_required = {_normalize_label(item): item for item in required_headers}
    for row_index, row in enumerate(rows):
        header_map: dict[str, int] = {}
        for column_index, value in enumerate(row):
            normalized = _normalize_label(value)
            if normalized:
                header_map.setdefault(normalized, column_index)
        if not all(key in header_map for key in normalized_required):
            continue
        requested_columns = []
        if request_mode:
            for column_index, value in enumerate(row):
                match = _REQUESTED_COLUMN_RE.match(str(value or "").strip())
                if match:
                    requested_columns.append(
                        {
                            "column_index": column_index,
                            "source_column": str(value).strip(),
                            "source_index": int(match.group(1)),
                        }
                    )
            if not requested_columns:
                continue
        return {
            "row_index": row_index,
            "header_map": header_map,
            "requested_columns": requested_columns,
            "raw_headers": list(row),
        }
    return None


def _cell(row: list[object], column_index: int | None):
    if column_index is None or column_index < 0 or column_index >= len(row):
        return None
    return row[column_index]


def _extract_template_version(request_rows: list[list[object]]) -> str:
    for row in request_rows[:5]:
        for value in row[:3]:
            text = str(value or "").strip()
            match = re.search(r"v(\d+(?:\.\d+)*)", text, re.IGNORECASE)
            if match:
                return match.group(1)
    return ""


def _domain_bucket(domain_key: str) -> str:
    return _DOMAIN_PARENT_MAP.get(domain_key, domain_key)


def _load_field_map(workbook: dict, issues: list[dict]) -> dict:
    field_map_rows = workbook["sheets"].get("FIELD_MAP", [])
    header = _find_header_row(field_map_rows, ["section", "field_label", "field_key", "domain", "role"])
    if header is None:
        issues.append(_issue("error", "missing_field_map_header", "FIELD_MAP is missing one or more required header columns.", sheet="FIELD_MAP"))
        return {}

    required = header["header_map"]
    by_key: dict[str, dict] = {}
    by_tuple: dict[tuple[str, str], dict] = {}
    entries: list[dict] = []

    for row_offset, row in enumerate(field_map_rows[header["row_index"] + 1 :], start=header["row_index"] + 2):
        section = _cell(row, required.get("section"))
        field_label = _cell(row, required.get("field label"))
        field_key = _cell(row, required.get("field key"))
        domain = _cell(row, required.get("domain"))
        role = _cell(row, required.get("role"))
        unit = _cell(row, required.get("unit"))
        compact_or_advanced = _cell(row, required.get("compact or advanced"))
        validation_list = _cell(row, required.get("validation list"))
        notes = _cell(row, required.get("notes"))

        if all(is_blank(item) for item in (section, field_label, field_key, domain, role, unit, compact_or_advanced, validation_list, notes)):
            continue

        normalized_section = _normalize_label(section)
        normalized_label = _normalize_label(field_label)
        normalized_field_key = str(field_key or "").strip()
        if not normalized_section or not normalized_label or not normalized_field_key:
            issues.append(_issue("warning", "incomplete_field_map_row", "FIELD_MAP row is incomplete and will be ignored.", sheet="FIELD_MAP", row=row_offset))
            continue

        tuple_key = (normalized_section, normalized_label)
        if normalized_field_key in by_key:
            issues.append(_issue("error", "duplicate_field_key", f"Duplicate field_key '{normalized_field_key}' found in FIELD_MAP.", sheet="FIELD_MAP", row=row_offset, field_key=normalized_field_key))
            continue
        if tuple_key in by_tuple:
            issues.append(_issue("error", "duplicate_section_field_label", f"Duplicate FIELD_MAP entry for section '{section}' and field '{field_label}'.", sheet="FIELD_MAP", row=row_offset, field_key=normalized_field_key))
            continue

        entry = {
            "row_index": row_offset,
            "section": str(section).strip(),
            "field_label": str(field_label).strip(),
            "field_key": normalized_field_key,
            "domain": normalize_domain(domain),
            "role": str(role or "").strip(),
            "unit": unit,
            "compact_or_advanced": compact_or_advanced,
            "validation_list": validation_list,
            "notes": notes,
            "tuple_key": tuple_key,
            "aliases": list(FIELD_KEY_ALIASES.get(normalized_field_key, (normalized_field_key,))),
        }
        by_key[normalized_field_key] = entry
        by_tuple[tuple_key] = entry
        entries.append(entry)

    return {
        "header": header,
        "entries": entries,
        "by_key": by_key,
        "by_tuple": by_tuple,
    }


def _load_request_rows(workbook: dict, field_map: dict, issues: list[dict]) -> dict:
    request_rows = workbook["sheets"].get("REQUEST", [])
    header = _find_header_row(
        request_rows,
        ["Section", "Field / Parameter", "Unit", "Baseline / Printed", "Baseline Correction"],
        request_mode=True,
    )
    if header is None:
        issues.append(_issue("error", "missing_request_header", "REQUEST is missing one or more required header columns, or no Requested #N columns were found.", sheet="REQUEST"))
        return {}

    header_map = header["header_map"]
    records: list[dict] = []
    matched_tuples: set[tuple[str, str]] = set()

    for row_offset, row in enumerate(request_rows[header["row_index"] + 1 :], start=header["row_index"] + 2):
        section = _cell(row, header_map.get("section"))
        field_label = _cell(row, header_map.get("field parameter"))
        unit = _cell(row, header_map.get("unit"))
        baseline_printed = _cell(row, header_map.get("baseline printed"))
        baseline_correction = _cell(row, header_map.get("baseline correction"))
        requested_values = {
            column["source_column"]: _cell(row, column["column_index"])
            for column in header["requested_columns"]
        }

        tuple_key = (_normalize_label(section), _normalize_label(field_label))
        row_has_request_data = any(not is_blank(value) for value in requested_values.values())
        row_has_baseline_data = not is_blank(baseline_printed) or not is_blank(baseline_correction)
        row_has_label = bool(tuple_key[0] or tuple_key[1])

        if not row_has_label and not row_has_request_data and not row_has_baseline_data:
            continue

        mapped_entry = field_map.get("by_tuple", {}).get(tuple_key)
        if mapped_entry is None:
            severity = "warning" if not row_has_request_data and not row_has_baseline_data else "error"
            issues.append(
                _issue(
                    severity,
                    "request_row_not_mapped",
                    f"REQUEST row '{section} / {field_label}' has no matching FIELD_MAP entry.",
                    sheet="REQUEST",
                    row=row_offset,
                )
            )
            continue

        matched_tuples.add(tuple_key)
        records.append(
            {
                "row_index": row_offset,
                "section": str(section or "").strip(),
                "field_label": str(field_label or "").strip(),
                "unit": unit,
                "baseline_printed": baseline_printed,
                "baseline_correction": baseline_correction,
                "requested_values": requested_values,
                "field_map": mapped_entry,
            }
        )

    for tuple_key, entry in field_map.get("by_tuple", {}).items():
        if tuple_key not in matched_tuples:
            issues.append(
                _issue(
                    "warning",
                    "field_map_without_request_row",
                    f"FIELD_MAP entry '{entry['section']} / {entry['field_label']}' has no matching REQUEST row.",
                    sheet="FIELD_MAP",
                    row=entry["row_index"],
                    field_key=entry["field_key"],
                )
            )

    return {
        "header": header,
        "records": records,
    }


def _requested_column_is_active(source_column: str, request_records: list[dict]) -> bool:
    for record in request_records:
        if not is_blank(record["requested_values"].get(source_column)):
            return True
    return False


def _proposal_issue(proposal: dict, severity: str, code: str, message: str, *, field_key: str | None = None, domain: str | None = None):
    payload = {
        "severity": severity,
        "code": code,
        "message": message,
    }
    if field_key is not None:
        payload["field_key"] = field_key
    if domain is not None:
        payload["domain"] = domain
    proposal["issues"].append(payload)


def validate_vde_request_workbook(path: str | Path) -> dict:
    workbook = _read_xlsx_workbook(path)
    issues: list[dict] = []
    available_sheets = set(workbook["sheet_names"])

    if "REQUEST" not in available_sheets:
        issues.append(_issue("error", "missing_request_sheet", "Workbook is missing the REQUEST sheet.", sheet="REQUEST"))
    if "FIELD_MAP" not in available_sheets:
        issues.append(_issue("error", "missing_field_map_sheet", "Workbook is missing the FIELD_MAP sheet.", sheet="FIELD_MAP"))
    if "LISTS" not in available_sheets:
        issues.append(_issue("warning", "missing_lists_sheet", "Workbook is missing the LISTS sheet.", sheet="LISTS"))
    if "RULES" not in available_sheets:
        issues.append(_issue("warning", "missing_rules_sheet", "Workbook is missing the RULES sheet.", sheet="RULES"))

    field_map = _load_field_map(workbook, issues) if "FIELD_MAP" in available_sheets else {}
    request = _load_request_rows(workbook, field_map, issues) if "REQUEST" in available_sheets and field_map else {}

    template_version = _extract_template_version(workbook["sheets"].get("REQUEST", [])) if "REQUEST" in available_sheets else ""
    if template_version and template_version != VDE_REQUEST_SCHEMA_VERSION:
        issues.append(
            _issue(
                "warning",
                "template_version_mismatch",
                f"Workbook template version '{template_version}' differs from parser contract version '{VDE_REQUEST_SCHEMA_VERSION}'.",
                sheet="REQUEST",
            )
        )

    requested_columns = []
    if request:
        requested_columns = request["header"]["requested_columns"]

    errors = [item for item in issues if item["severity"] == "error"]
    warnings = [item for item in issues if item["severity"] == "warning"]
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "issues": issues,
        "template_version": template_version,
        "sheet_names": list(workbook["sheet_names"]),
        "request_header_row": (request.get("header", {}).get("row_index", -1) + 1) if request else None,
        "field_map_header_row": (field_map.get("header", {}).get("row_index", -1) + 1) if field_map else None,
        "requested_columns": [
            {
                "source_column": item["source_column"],
                "source_index": item["source_index"],
                "column_index": item["column_index"],
                "active": _requested_column_is_active(item["source_column"], request.get("records", [])) if request else False,
            }
            for item in requested_columns
        ],
    }


def parse_vde_request_workbook(path: str | Path) -> dict:
    validation = validate_vde_request_workbook(path)
    if not validation["ok"]:
        raise ValueError("Workbook validation failed: " + "; ".join(item["message"] for item in validation["errors"]))

    workbook = _read_xlsx_workbook(path)
    issues = list(validation["issues"])
    field_map = _load_field_map(workbook, issues)
    request = _load_request_rows(workbook, field_map, issues)
    request_records = request["records"]

    baseline_printed: dict[str, object] = {}
    baseline_corrections: dict[str, object] = {}
    effective_baseline: dict[str, object] = {}

    for record in request_records:
        field_key = record["field_map"]["field_key"]
        printed = record["baseline_printed"]
        correction = record["baseline_correction"]
        baseline_printed[field_key] = printed
        baseline_corrections[field_key] = correction
        effective_baseline[field_key] = resolve_effective_baseline(printed, correction)

    requested_headers = sorted(request["header"]["requested_columns"], key=lambda item: item["source_index"])
    active_requested_headers = [item for item in requested_headers if _requested_column_is_active(item["source_column"], request_records)]

    proposals: list[dict] = []
    proposal_lookup: dict[str, dict] = {}
    for display_index, header_info in enumerate(active_requested_headers, start=1):
        proposal_id = f"proposal_req_{header_info['source_index']}"
        proposal = {
            "proposal_id": proposal_id,
            "display_index": display_index,
            "source_column": header_info["source_column"],
            "source_index": header_info["source_index"],
            "name": None,
            "walk_from": {
                "kind": "baseline",
                "proposal_id": None,
                "source_column": "Baseline",
            },
            "raw_values": {},
            "normalized_values": {},
            "domain_requests": {},
            "issues": [],
        }
        proposals.append(proposal)
        proposal_lookup[header_info["source_column"]] = proposal

    for proposal in proposals:
        source_column = proposal["source_column"]
        domain_raw_values: dict[str, dict[str, object]] = {}
        domain_map_rows: dict[str, list[dict]] = {}
        proposal_type_rows: dict[str, dict] = {}

        for record in request_records:
            field_key = record["field_map"]["field_key"]
            raw_value = record["requested_values"].get(source_column)
            if not is_blank(raw_value):
                proposal["raw_values"][field_key] = raw_value
                proposal["normalized_values"][field_key] = raw_value

            mapped_domain = record["field_map"]["domain"]
            domain_bucket = _domain_bucket(mapped_domain)
            domain_map_rows.setdefault(domain_bucket, []).append(record)
            if not is_blank(raw_value):
                domain_raw_values.setdefault(domain_bucket, {})[field_key] = raw_value

            if str(record["field_map"]["role"] or "").strip().lower() == "proposal_type":
                proposal_type_rows[domain_bucket] = record

        proposal["name"] = proposal["raw_values"].get("notes")
        if is_blank(proposal["name"]):
            proposal["name"] = None

        original_walk_from = proposal["raw_values"].get("walk_from")
        if is_blank(original_walk_from):
            proposal["walk_from"] = {
                "kind": "baseline",
                "proposal_id": None,
                "source_column": "Baseline",
            }
        else:
            walk_text = str(original_walk_from).strip()
            if _normalize_label(walk_text) == "baseline":
                proposal["walk_from"] = {
                    "kind": "baseline",
                    "proposal_id": None,
                    "source_column": "Baseline",
                }
            else:
                match = _REQUESTED_COLUMN_RE.match(walk_text)
                if not match:
                    proposal["walk_from"] = {
                        "kind": "unknown",
                        "proposal_id": None,
                        "source_column": walk_text,
                    }
                    _proposal_issue(proposal, "error", "invalid_walk_from", f"Walk From '{walk_text}' is not recognized.")
                else:
                    requested_index = int(match.group(1))
                    if requested_index >= proposal["source_index"]:
                        _proposal_issue(proposal, "error", "future_walk_from", f"Walk From '{walk_text}' points to the current or a future proposal.")
                    referenced_column = next((item for item in requested_headers if item["source_index"] == requested_index), None)
                    if referenced_column is None:
                        _proposal_issue(proposal, "error", "missing_walk_from_target", f"Walk From '{walk_text}' does not exist in the workbook.")
                        proposal["walk_from"] = {
                            "kind": "proposal",
                            "proposal_id": None,
                            "source_column": walk_text,
                        }
                    elif referenced_column["source_column"] not in proposal_lookup:
                        _proposal_issue(proposal, "error", "empty_walk_from_target", f"Walk From '{walk_text}' points to a Requested column that was ignored because it is empty.")
                        proposal["walk_from"] = {
                            "kind": "proposal",
                            "proposal_id": None,
                            "source_column": referenced_column["source_column"],
                        }
                    else:
                        target_proposal = proposal_lookup[referenced_column["source_column"]]
                        proposal["walk_from"] = {
                            "kind": "proposal",
                            "proposal_id": target_proposal["proposal_id"],
                            "source_column": referenced_column["source_column"],
                        }

        for domain_bucket, mapped_rows in domain_map_rows.items():
            if domain_bucket in {"scenario", "roadload"}:
                continue

            raw_values = dict(domain_raw_values.get(domain_bucket) or {})
            proposal_type_record = proposal_type_rows.get(domain_bucket)
            raw_proposal_type = None
            normalized_proposal = None

            if proposal_type_record is not None:
                raw_proposal_type = proposal_type_record["requested_values"].get(source_column)
                normalized_proposal = normalize_template_proposal_type(domain_bucket, raw_proposal_type)
            elif domain_bucket in _PROPOSAL_DOMAINS:
                normalized_proposal = normalize_template_proposal_type(domain_bucket, None)

            domain_request = {
                "domain": domain_bucket,
                "raw_proposal_type": raw_proposal_type,
                "normalized_proposal": normalized_proposal,
                "raw_values": raw_values,
                "normalized_values": dict(raw_values),
                "aliases": {
                    row["field_map"]["field_key"]: list(row["field_map"]["aliases"])
                    for row in mapped_rows
                },
                "issues": [],
            }

            if normalized_proposal and normalized_proposal.get("ok"):
                domain_request["proposal_type"] = normalized_proposal.get("proposal_type")
                domain_request["selection_mode"] = normalized_proposal.get("selection_mode")
                domain_request["has_internal_equivalent"] = normalized_proposal.get("has_internal_equivalent")
                domain_request["proposal_details_seed"] = dict(normalized_proposal.get("details") or {})
                if normalized_proposal.get("notes"):
                    domain_request["issues"].append(
                        _issue(
                            "warning",
                            "proposal_contract_note",
                            str(normalized_proposal["notes"]),
                            field_key=proposal_type_record["field_map"]["field_key"] if proposal_type_record else None,
                            source_column=source_column,
                        )
                    )
            elif normalized_proposal:
                domain_request["proposal_type"] = None
                domain_request["has_internal_equivalent"] = False
                domain_request["issues"].append(
                    _issue(
                        "review",
                        normalized_proposal.get("error", "unknown_proposal_type"),
                        normalized_proposal.get("message", "Proposal type could not be normalized."),
                        field_key=proposal_type_record["field_map"]["field_key"] if proposal_type_record else None,
                        source_column=source_column,
                    )
                )

            if domain_bucket in _PROPOSAL_DOMAINS:
                non_type_values = {
                    key: value
                    for key, value in raw_values.items()
                    if key != (proposal_type_record["field_map"]["field_key"] if proposal_type_record else None)
                }
                if non_type_values and is_blank(raw_proposal_type):
                    domain_request["issues"].append(
                        _issue(
                            "review",
                            "missing_proposal_type",
                            "Domain contains requested values but proposal type is missing.",
                            source_column=source_column,
                        )
                    )

            proposal["domain_requests"][domain_bucket] = domain_request

        if proposal["name"] is None:
            _proposal_issue(proposal, "warning", "missing_name", "Proposal name/description is empty and will need to be generated later.")

        for domain_request in proposal["domain_requests"].values():
            proposal["issues"].extend(domain_request["issues"])

    template_version = _extract_template_version(workbook["sheets"].get("REQUEST", []))

    original_rows = []
    for record in request_records:
        original_rows.append(
            {
                "row_index": record["row_index"],
                "section": record["section"],
                "field_label": record["field_label"],
                "field_key": record["field_map"]["field_key"],
                "domain": record["field_map"]["domain"],
                "role": record["field_map"]["role"],
                "baseline_printed": record["baseline_printed"],
                "baseline_correction": record["baseline_correction"],
                "requested_values": dict(record["requested_values"]),
            }
        )

    return {
        "schema_version": VDE_REQUEST_SCHEMA_VERSION,
        "template_version": template_version,
        "source": {
            "filename": str(Path(path).name),
            "imported_at": datetime.now(timezone.utc).isoformat(),
        },
        "baseline_printed": baseline_printed,
        "baseline_corrections": baseline_corrections,
        "effective_baseline": effective_baseline,
        "proposals": proposals,
        "issues": issues,
        "original_request": {
            "sheet_names": list(workbook["sheet_names"]),
            "request_rows": original_rows,
            "requested_columns": [
                {
                    "source_column": item["source_column"],
                    "source_index": item["source_index"],
                    "active": item["source_column"] in proposal_lookup,
                }
                for item in requested_headers
            ],
        },
    }
