from __future__ import annotations

from copy import deepcopy
import io
from pathlib import Path
import re
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree import ElementTree as ET

from src.vde_core.vde_request_contract import (
    FIELD_KEY_ALIASES,
    TEMPLATE_PROPOSAL_MAP,
    VDE_REQUEST_SCHEMA_VERSION,
    is_blank,
)
from src.vde_core.vde_request_parser import (
    _extract_template_version,
    _load_field_map,
    _load_request_rows,
    _read_xlsx_workbook,
)


_XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PKG_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_CUSTOM_PROP_NS = "http://schemas.openxmlformats.org/officeDocument/2006/custom-properties"
_VT_NS = "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"
_CUSTOM_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/custom-properties"
_CUSTOM_FMTID = "{D5CDD505-2E9C-101B-9397-08002B2CF9AE}"


def sanitize_request_filename_token(value, fallback: str = "request") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return text or fallback


def build_prefilled_ppe_template_filename(
    baseline_id,
    vehicle,
    date_text: str,
) -> str:
    baseline_token = sanitize_request_filename_token(baseline_id, "baseline")
    vehicle_token = sanitize_request_filename_token(vehicle, "vehicle")
    date_token = sanitize_request_filename_token(date_text, "date")
    return f"EcoDrive_VDE_PPE_Request_{baseline_token}_{vehicle_token}_{date_token}.xlsx"


def extract_referenced_baseline_id(request_draft: dict | None):
    draft = dict(request_draft or {})
    for source in (
        dict(draft.get("effective_baseline") or {}),
        dict(draft.get("baseline_printed") or {}),
        dict(draft.get("baseline_corrections") or {}),
    ):
        for alias in FIELD_KEY_ALIASES.get("selected_baseline_vde_id", ("selected_baseline_vde_id",)):
            value = source.get(alias)
            if not is_blank(value):
                try:
                    return int(float(value))
                except Exception:
                    return value
    return None


def resolve_imported_baseline_status(current_baseline_id, imported_baseline_id, found_row: dict | None) -> dict:
    if is_blank(imported_baseline_id):
        return {
            "status": "missing_reference",
            "requires_confirmation": False,
            "blocking": True,
            "message": "The imported request does not include a recoverable baseline VDE ID.",
        }
    if not found_row:
        return {
            "status": "unresolved",
            "requires_confirmation": False,
            "blocking": True,
            "message": f"The imported request references baseline VDE ID {imported_baseline_id}, but it was not found in the current VDE database.",
        }
    if is_blank(current_baseline_id):
        return {
            "status": "ready_to_load",
            "requires_confirmation": False,
            "blocking": False,
            "message": f"Baseline VDE ID {imported_baseline_id} was found and will be loaded when the import is applied.",
        }
    if str(current_baseline_id) == str(imported_baseline_id):
        return {
            "status": "matched_current",
            "requires_confirmation": False,
            "blocking": False,
            "message": f"Imported request baseline VDE ID {imported_baseline_id} matches the current page baseline.",
        }
    return {
        "status": "mismatch",
        "requires_confirmation": True,
        "blocking": False,
        "message": (
            f"The imported request references baseline VDE ID {imported_baseline_id}, "
            f"while the current page is using baseline VDE ID {current_baseline_id}."
        ),
    }


def _is_number_like(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        compact = value.strip().replace(",", ".")
        return bool(re.fullmatch(r"[+-]?\d+(?:\.\d+)?", compact))
    return False


def _coerce_number(value):
    if isinstance(value, bool):
        raise ValueError("Boolean is not numeric here.")
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip().replace(",", "."))


def _same_snapshot_value(left, right) -> bool:
    if is_blank(left) and is_blank(right):
        return True
    if _is_number_like(left) and _is_number_like(right):
        try:
            return abs(_coerce_number(left) - _coerce_number(right)) < 1e-9
        except Exception:
            pass
    return str(left) == str(right)


def _value_from_aliases(payload: dict | None, field_key: str):
    data = dict(payload or {})
    for alias in FIELD_KEY_ALIASES.get(field_key, (field_key,)):
        if alias in data and not is_blank(data.get(alias)):
            return data.get(alias)
    return None


def compare_printed_snapshot(imported_printed: dict | None, current_baseline: dict | None) -> dict:
    imported = dict(imported_printed or {})
    current = dict(current_baseline or {})
    divergent_fields: list[dict] = []
    compared_fields = 0
    for field_key, imported_value in imported.items():
        if is_blank(imported_value):
            continue
        compared_fields += 1
        current_value = _value_from_aliases(current, field_key)
        if _same_snapshot_value(imported_value, current_value):
            continue
        divergent_fields.append(
            {
                "field_key": field_key,
                "printed_value": imported_value,
                "baseline_value": current_value,
            }
        )
    if divergent_fields:
        return {
            "status": "Review",
            "ok": False,
            "compared_fields": compared_fields,
            "divergent_fields": divergent_fields,
            "message": "Baseline / Printed differs from the current database baseline. Use Baseline Correction for intentional changes.",
        }
    return {
        "status": "OK",
        "ok": True,
        "compared_fields": compared_fields,
        "divergent_fields": [],
        "message": "Printed snapshot integrity: OK",
    }


def _template_source_bytes(template_source) -> bytes:
    if isinstance(template_source, (bytes, bytearray)):
        return bytes(template_source)
    if hasattr(template_source, "read"):
        return template_source.read()
    return Path(template_source).read_bytes()


def _workbook_from_source(template_source) -> dict:
    if isinstance(template_source, (str, Path)):
        return _read_xlsx_workbook(template_source)
    payload = _template_source_bytes(template_source)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as handle:
        handle.write(payload)
        temp_path = Path(handle.name)
    try:
        return _read_xlsx_workbook(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _request_structure(template_source) -> tuple[dict, dict, dict]:
    workbook = _workbook_from_source(template_source)
    issues: list[dict] = []
    field_map = _load_field_map(workbook, issues)
    request = _load_request_rows(workbook, field_map, issues)
    if not request or not field_map:
        raise ValueError("Template REQUEST/FIELD_MAP structure could not be loaded.")
    return workbook, field_map, request


def _sheet_path_map(archive: ZipFile) -> dict[str, str]:
    workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
    rels_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels_root}
    mapping: dict[str, str] = {}
    for sheet in workbook_root.find(f"{_XML_NS}sheets") or []:
        name = str(sheet.attrib.get("name") or "")
        rel_id = sheet.attrib.get(f"{_REL_NS}id") or ""
        target = rel_map.get(rel_id)
        if not target:
            continue
        mapping[name] = target if target.startswith("xl/") else f"xl/{target}"
    return mapping


def _column_name(index: int) -> str:
    value = index + 1
    letters = []
    while value:
        value, remainder = divmod(value - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return "".join(reversed(letters))


def _cell_ref(row_index: int, column_index: int) -> str:
    return f"{_column_name(column_index)}{row_index + 1}"


def _set_cell_value(cell: ET.Element, value) -> None:
    for child in list(cell):
        cell.remove(child)
    if is_blank(value):
        cell.attrib.pop("t", None)
        return
    if isinstance(value, bool):
        cell.set("t", "b")
        value_node = ET.SubElement(cell, f"{_XML_NS}v")
        value_node.text = "1" if value else "0"
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        cell.attrib.pop("t", None)
        value_node = ET.SubElement(cell, f"{_XML_NS}v")
        value_node.text = str(value)
        return
    cell.set("t", "inlineStr")
    is_node = ET.SubElement(cell, f"{_XML_NS}is")
    text_node = ET.SubElement(is_node, f"{_XML_NS}t")
    text_node.text = str(value)


def _cell_column_index(ref: str) -> int:
    letters = "".join(char for char in str(ref or "") if char.isalpha()).upper()
    value = 0
    for char in letters:
        value = (value * 26) + (ord(char) - ord("A") + 1)
    return value - 1


def _get_or_create_row(sheet_root: ET.Element, row_number: int) -> ET.Element:
    sheet_data = sheet_root.find(f"{_XML_NS}sheetData")
    if sheet_data is None:
        sheet_data = ET.SubElement(sheet_root, f"{_XML_NS}sheetData")
    for row in sheet_data.findall(f"{_XML_NS}row"):
        if int(row.attrib.get("r", "0")) == row_number:
            return row
    row = ET.SubElement(sheet_data, f"{_XML_NS}row", {"r": str(row_number)})
    rows = sheet_data.findall(f"{_XML_NS}row")
    rows.sort(key=lambda node: int(node.attrib.get("r", "0")))
    sheet_data[:] = rows
    return row


def _get_or_create_cell(row: ET.Element, ref: str) -> ET.Element:
    for cell in row.findall(f"{_XML_NS}c"):
        if cell.attrib.get("r") == ref:
            return cell
    cell = ET.Element(f"{_XML_NS}c", {"r": ref})
    cells = list(row.findall(f"{_XML_NS}c"))
    cells.append(cell)
    cells.sort(key=lambda node: _cell_column_index(node.attrib.get("r", "")))
    row[:] = cells
    return cell


def _canonical_request_field_rows(request_draft: dict | None) -> tuple[dict[str, dict], dict[str, dict], list[str]]:
    draft = dict(request_draft or {})
    record_map: dict[str, dict] = {}
    original = dict(draft.get("original_request") or {})
    requested_columns = []
    for item in list(original.get("requested_columns") or []):
        source_column = str(dict(item or {}).get("source_column") or "").strip()
        if source_column:
            requested_columns.append(source_column)
    for record in list(original.get("request_rows") or []):
        payload = deepcopy(dict(record or {}))
        field_key = str(payload.get("field_key") or "").strip()
        if field_key:
            record_map[field_key] = payload
    proposal_rows: dict[str, dict] = {}
    for proposal in list(draft.get("proposals") or []):
        payload = dict(proposal or {})
        source_column = str(payload.get("source_column") or f"Requested #{int(payload.get('display_index') or 0) or 1}").strip()
        if source_column and source_column not in requested_columns:
            requested_columns.append(source_column)
        proposal_rows.setdefault("notes", {})[source_column] = payload.get("name")
        walk_from = dict(payload.get("walk_from") or {})
        proposal_rows.setdefault("walk_from", {})[source_column] = walk_from.get("source_column") or "Baseline"
        for domain_key, domain_request in dict(payload.get("domain_requests") or {}).items():
            domain_payload = dict(domain_request or {})
            proposal_type_key = f"{domain_key}_proposal_type"
            proposal_rows.setdefault(proposal_type_key, {})[source_column] = _resolve_template_proposal_label(domain_key, domain_payload)
            raw_values = dict(domain_payload.get("raw_values") or {})
            for field_key, value in raw_values.items():
                proposal_rows.setdefault(field_key, {})[source_column] = value
    return record_map, proposal_rows, requested_columns


def _resolve_template_proposal_label(domain_key: str, domain_payload: dict) -> str | None:
    raw_value = domain_payload.get("raw_proposal_type")
    if not is_blank(raw_value):
        return str(raw_value)
    selection_mode = domain_payload.get("selection_mode")
    if not is_blank(selection_mode) and selection_mode != domain_payload.get("proposal_type"):
        return str(selection_mode)
    proposal_type = str(domain_payload.get("proposal_type") or "").strip()
    details = dict(domain_payload.get("proposal_details_seed") or {})
    if not proposal_type:
        return None
    for label, entry in TEMPLATE_PROPOSAL_MAP.get(domain_key, {}).items():
        if str(entry.get("proposal_type") or "") != proposal_type:
            continue
        expected = dict(entry.get("details") or {})
        if all(details.get(key) == value for key, value in expected.items()):
            return label
    return proposal_type


def _baseline_printed_value(baseline_snapshot: dict | None, field_key: str):
    return _value_from_aliases(baseline_snapshot, field_key)


def build_canonical_baseline_payload(source: dict | None) -> dict:
    payload: dict[str, object] = {}
    for field_key in FIELD_KEY_ALIASES:
        value = _value_from_aliases(source, field_key)
        if not is_blank(value):
            payload[field_key] = value
    return payload


def _baseline_correction_value(request_draft: dict | None, field_key: str, fallback_record: dict | None = None):
    corrections = dict(dict(request_draft or {}).get("baseline_corrections") or {})
    value = corrections.get(field_key)
    if not is_blank(value):
        return value
    if fallback_record:
        return fallback_record.get("baseline_correction")
    return None


def _requested_value_for_field(
    field_key: str,
    source_column: str,
    canonical_rows: dict[str, dict],
    proposal_rows: dict[str, dict],
):
    record = dict(canonical_rows.get(field_key) or {})
    requested_values = dict(record.get("requested_values") or {})
    if source_column in requested_values:
        return requested_values.get(source_column)
    return dict(proposal_rows.get(field_key) or {}).get(source_column)


def _update_request_sheet_xml(
    xml_bytes: bytes,
    request: dict,
    baseline_snapshot: dict,
    request_draft: dict | None,
) -> bytes:
    root = ET.fromstring(xml_bytes)
    canonical_rows, proposal_rows, requested_columns = _canonical_request_field_rows(request_draft)
    header = dict(request.get("header") or {})
    requested_headers = [item["source_column"] for item in list(header.get("requested_columns") or [])]
    for source_column in requested_columns:
        if source_column not in requested_headers:
            requested_headers.append(source_column)
    row_by_field_key = {
        str(record["field_map"]["field_key"]): dict(record)
        for record in list(request.get("records") or [])
    }
    header_map = dict(header.get("header_map") or {})
    for field_key, record in row_by_field_key.items():
        row_index = int(record["row_index"]) - 1
        row = _get_or_create_row(root, row_index + 1)

        baseline_printed_ref = _cell_ref(row_index, header_map.get("baseline printed"))
        baseline_printed_cell = _get_or_create_cell(row, baseline_printed_ref)
        _set_cell_value(baseline_printed_cell, _baseline_printed_value(baseline_snapshot, field_key))

        correction_ref = _cell_ref(row_index, header_map.get("baseline correction"))
        correction_cell = _get_or_create_cell(row, correction_ref)
        _set_cell_value(correction_cell, _baseline_correction_value(request_draft, field_key, canonical_rows.get(field_key)))

        for requested_meta in list(header.get("requested_columns") or []):
            source_column = str(requested_meta.get("source_column") or "")
            column_index = int(requested_meta["column_index"])
            requested_ref = _cell_ref(row_index, column_index)
            requested_cell = _get_or_create_cell(row, requested_ref)
            _set_cell_value(requested_cell, _requested_value_for_field(field_key, source_column, canonical_rows, proposal_rows))
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _update_root_relationships(xml_bytes: bytes) -> bytes:
    root = ET.fromstring(xml_bytes)
    for rel in root.findall(f"{_PKG_REL_NS}Relationship"):
        if rel.attrib.get("Type") == _CUSTOM_REL_TYPE:
            return ET.tostring(root, encoding="utf-8", xml_declaration=True)
    next_index = 1
    seen = {rel.attrib.get("Id") for rel in root.findall(f"{_PKG_REL_NS}Relationship")}
    while f"rId{next_index}" in seen:
        next_index += 1
    ET.SubElement(
        root,
        f"{_PKG_REL_NS}Relationship",
        {
            "Id": f"rId{next_index}",
            "Type": _CUSTOM_REL_TYPE,
            "Target": "docProps/custom.xml",
        },
    )
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _update_content_types(xml_bytes: bytes) -> bytes:
    root = ET.fromstring(xml_bytes)
    for override in root.findall("{http://schemas.openxmlformats.org/package/2006/content-types}Override"):
        if override.attrib.get("PartName") == "/docProps/custom.xml":
            return ET.tostring(root, encoding="utf-8", xml_declaration=True)
    ET.SubElement(
        root,
        "{http://schemas.openxmlformats.org/package/2006/content-types}Override",
        {
            "PartName": "/docProps/custom.xml",
            "ContentType": "application/vnd.openxmlformats-officedocument.custom-properties+xml",
        },
    )
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _custom_props_xml(existing: bytes | None, props: dict[str, object]) -> bytes:
    if existing:
        root = ET.fromstring(existing)
    else:
        root = ET.Element(
            "{http://schemas.openxmlformats.org/officeDocument/2006/custom-properties}Properties",
            {
                "xmlns": _CUSTOM_PROP_NS,
                "xmlns:vt": _VT_NS,
            },
        )
    existing_nodes = {
        prop.attrib.get("name"): prop
        for prop in root.findall(f"{{{_CUSTOM_PROP_NS}}}property")
    }
    used_pids = {
        int(prop.attrib.get("pid", "1"))
        for prop in root.findall(f"{{{_CUSTOM_PROP_NS}}}property")
        if str(prop.attrib.get("pid") or "").isdigit()
    }
    next_pid = max(used_pids or {1}) + 1
    for name, value in props.items():
        text = "" if is_blank(value) else str(value)
        prop = existing_nodes.get(name)
        if prop is None:
            prop = ET.SubElement(
                root,
                f"{{{_CUSTOM_PROP_NS}}}property",
                {
                    "fmtid": _CUSTOM_FMTID,
                    "pid": str(next_pid),
                    "name": name,
                },
            )
            next_pid += 1
        for child in list(prop):
            prop.remove(child)
        vt = ET.SubElement(prop, f"{{{_VT_NS}}}lpwstr")
        vt.text = text
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def build_prefilled_ppe_template(
    template_source,
    baseline_snapshot,
    request_draft=None,
    metadata=None,
) -> bytes:
    baseline_snapshot = deepcopy(dict(baseline_snapshot or {}))
    if not baseline_snapshot:
        raise ValueError("A loaded baseline snapshot is required to prefill the PPE template.")
    request_draft = deepcopy(dict(request_draft or {}))
    metadata = deepcopy(dict(metadata or {}))
    workbook, field_map, request = _request_structure(template_source)
    template_version = _extract_template_version(workbook["sheets"].get("REQUEST", [])) or VDE_REQUEST_SCHEMA_VERSION
    template_bytes = _template_source_bytes(template_source)
    baseline_id = extract_referenced_baseline_id(
        {
            "effective_baseline": baseline_snapshot,
            "baseline_printed": baseline_snapshot,
            "baseline_corrections": {},
        }
    )
    props = {
        "baseline_vde_id": baseline_id if baseline_id is not None else "",
        "baseline_source": metadata.get("baseline_source") or baseline_snapshot.get("line_source") or "",
        "request_schema_version": metadata.get("request_schema_version") or VDE_REQUEST_SCHEMA_VERSION,
        "template_version": metadata.get("template_version") or template_version,
        "source_type": metadata.get("source_type") or dict(request_draft.get("source") or {}).get("source_type") or "UI",
    }

    output = io.BytesIO()
    with ZipFile(io.BytesIO(template_bytes)) as source_archive, ZipFile(output, "w", compression=ZIP_DEFLATED) as target_archive:
        sheet_paths = _sheet_path_map(source_archive)
        request_path = sheet_paths.get("REQUEST")
        custom_xml = source_archive.read("docProps/custom.xml") if "docProps/custom.xml" in source_archive.namelist() else None
        for name in source_archive.namelist():
            data = source_archive.read(name)
            if request_path and name == request_path:
                data = _update_request_sheet_xml(data, request, baseline_snapshot, request_draft)
            elif name == "_rels/.rels":
                data = _update_root_relationships(data)
            elif name == "[Content_Types].xml":
                data = _update_content_types(data)
            elif name == "docProps/custom.xml":
                data = _custom_props_xml(custom_xml, props)
            target_archive.writestr(name, data)
        if "docProps/custom.xml" not in source_archive.namelist():
            target_archive.writestr("docProps/custom.xml", _custom_props_xml(None, props))
    return output.getvalue()


__all__ = [
    "build_canonical_baseline_payload",
    "build_prefilled_ppe_template",
    "build_prefilled_ppe_template_filename",
    "compare_printed_snapshot",
    "extract_referenced_baseline_id",
    "resolve_imported_baseline_status",
    "sanitize_request_filename_token",
]
