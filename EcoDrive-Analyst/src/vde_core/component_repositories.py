from __future__ import annotations

from copy import deepcopy
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
from pathlib import Path
import sqlite3

from src.vde_core import db as db_module
from src.vde_core.database_management_contract import EntityType, normalize_record_origin
from src.vde_core.tire_roadload_service import get_tire_by_code, get_tire_by_id
from src.vde_core.vde_request_contract import is_blank, normalize_domain


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data" / "components"
_BASE_FIELDS = ("component_id", "component_name", "status", "source", "notes")
_DB_METADATA_FIELDS = (
    "id",
    "created_at",
    "updated_at",
    "record_origin",
    "record_status",
    "source_record_id",
)
_PROVENANCE_FIELDS = (
    "component_type",
    "component_position",
    "driveline_architecture",
    "physical_boundary",
    "configuration_from",
    "configuration_to",
    "test_condition_type",
    "test_method",
    "hardware_reference",
    "source_reference",
    "net_bridge_eligible",
)
COMPONENT_PROVENANCE_FIELDS = _PROVENANCE_FIELDS
_DOMAIN_FIELD_MAP = {
    "transmission": ("trans_A", "trans_B", "trans_C", "loss_pct"),
    "brake": ("brake_A", "brake_B", "brake_C", "residual_torque_front_nm", "residual_torque_rear_nm", "wheel_radius_m"),
    "axle_hubs": ("axle_hubs_A", "axle_hubs_B", "axle_hubs_C"),
    "parasitic": ("parasitic_A", "parasitic_B", "parasitic_C"),
}
_DOMAIN_FILENAME_MAP = {
    "transmission": "transmission_components_mock.csv",
    "brake": "brake_components_mock.csv",
    "axle_hubs": "axle_hubs_components_mock.csv",
    "parasitic": "parasitic_components_mock.csv",
}
_COMPONENT_DOMAINS = tuple(_DOMAIN_FILENAME_MAP)
_QA_SEED_TIMESTAMP = "2026-07-16T00:00:00Z"
_CANONICAL_STORAGE_FIELDS = {
    "record_origin",
    "record_status",
    "domain",
    "component_code",
    "component_name",
    "source_name",
    "source_record_id",
    "source_reference",
    "hardware_reference",
    "component_type",
    "component_position",
    "driveline_architecture",
    "physical_boundary",
    "configuration_from",
    "configuration_to",
    "test_condition_type",
    "test_method",
    "net_bridge_eligible",
    "equivalent_A_N",
    "equivalent_B_N_per_kph",
    "equivalent_C_N_per_kph2",
    "loss_pct",
    "residual_torque_front_nm",
    "residual_torque_rear_nm",
    "wheel_radius_m",
    "notes",
}


def _issue(code: str, severity: str, message: str, *, domain: str, component_id: str | None = None, field_key: str | None = None) -> dict:
    return {
        "code": code,
        "severity": severity,
        "domain": domain,
        "component_id": component_id,
        "field_key": field_key,
        "message": message,
    }


def _clean_text(value) -> str:
    return str(value or "").strip()


def _to_float(value):
    if is_blank(value):
        return None
    return float(value)


def _copy_component(component: dict | None) -> dict | None:
    if component is None:
        return None
    return deepcopy(component)


@dataclass
class ComponentRepository:
    domain: str
    source: str
    _components: list[dict] = field(default_factory=list)
    _issues: list[dict] = field(default_factory=list)
    _by_id: dict[str, dict] = field(default_factory=dict)

    def list_components(self) -> list[dict]:
        return deepcopy(self._components)

    def get_by_id(self, component_id: str) -> dict | None:
        return _copy_component(self._by_id.get(str(component_id or "").strip()))

    def search(self, query: str) -> list[dict]:
        needle = str(query or "").strip().lower()
        if not needle:
            return self.list_components()
        matches: list[dict] = []
        for component in self._components:
            haystack = " ".join(
                str(component.get(key) or "").lower()
                for key in (*_BASE_FIELDS, *_PROVENANCE_FIELDS, *_DB_METADATA_FIELDS)
            )
            if needle in haystack:
                matches.append(deepcopy(component))
        return matches

    def validate_component(self, component: dict) -> list[dict]:
        return _validate_component(self.domain, component)

    @property
    def issues(self) -> list[dict]:
        return deepcopy(self._issues)


def _validate_component(domain: str, component: dict) -> list[dict]:
    domain_key = normalize_domain(domain)
    payload = dict(component or {})
    component_id = _clean_text(payload.get("component_id"))
    issues: list[dict] = []
    for field_name in _BASE_FIELDS:
        if is_blank(payload.get(field_name)):
            issues.append(
                _issue(
                    "missing_required_field",
                    "error",
                    f"Component is missing required field '{field_name}'.",
                    domain=domain_key,
                    component_id=component_id or None,
                    field_key=field_name,
                )
            )
    for field_name in _DOMAIN_FIELD_MAP.get(domain_key, ()):
        try:
            value = _to_float(payload.get(field_name))
        except Exception:
            issues.append(
                _issue(
                    "invalid_numeric_field",
                    "error",
                    f"Field '{field_name}' must be numeric.",
                    domain=domain_key,
                    component_id=component_id or None,
                    field_key=field_name,
                )
            )
            continue
        if value is None:
            issues.append(
                _issue(
                    "missing_technical_field",
                    "error",
                    f"Component is missing technical field '{field_name}'.",
                    domain=domain_key,
                    component_id=component_id or None,
                    field_key=field_name,
                )
            )
    return issues


def _normalize_component_row(domain: str, row: dict) -> dict:
    domain_key = normalize_domain(domain)
    payload = {key: value for key, value in dict(row or {}).items()}
    component = {
        "domain": domain_key,
        "component_id": _clean_text(payload.get("component_id")),
        "component_name": _clean_text(payload.get("component_name")),
        "status": _clean_text(payload.get("status")),
        "source": _clean_text(payload.get("source")),
        "notes": _clean_text(payload.get("notes")),
    }
    for field_name in _DB_METADATA_FIELDS:
        component[field_name] = payload.get(field_name)
    for field_name in _DOMAIN_FIELD_MAP.get(domain_key, ()):
        component[field_name] = _to_float(payload.get(field_name))
    for field_name in _PROVENANCE_FIELDS:
        component[field_name] = _clean_text(payload.get(field_name))
    return component


def _normalize_domain_key(domain: str) -> str:
    domain_key = normalize_domain(domain)
    if domain_key not in _COMPONENT_DOMAINS:
        raise ValueError(f"Unsupported component repository domain '{domain}'.")
    return domain_key


def _bridge_value(value) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if int(value) else 0
    normalized = _clean_text(value).upper()
    if normalized in {"TRUE", "YES", "Y", "1"}:
        return 1
    if normalized in {"FALSE", "NO", "N", "0"}:
        return 0
    if normalized in {"UNKNOWN", "N/A", "NA", "NONE"}:
        return None
    raise ValueError("net_bridge_eligible must be TRUE, FALSE, or UNKNOWN.")


def _bridge_label(value) -> str:
    if value in (None, ""):
        return "UNKNOWN"
    return "TRUE" if bool(int(value)) else "FALSE"


def _technical_storage_fields(domain: str) -> dict[str, str]:
    domain_key = _normalize_domain_key(domain)
    fields = _DOMAIN_FIELD_MAP[domain_key]
    mapping = {
        fields[0]: "equivalent_A_N",
        fields[1]: "equivalent_B_N_per_kph",
        fields[2]: "equivalent_C_N_per_kph2",
    }
    for field_name in fields[3:]:
        mapping[field_name] = field_name
    return mapping


def _db_row_to_component(row: dict) -> dict:
    payload = dict(row or {})
    domain_key = _normalize_domain_key(payload.get("domain"))
    component = {
        "id": payload.get("id"),
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
        "record_origin": payload.get("record_origin"),
        "record_status": payload.get("record_status"),
        "source_record_id": payload.get("source_record_id"),
        "domain": domain_key,
        "component_code": payload.get("component_code"),
        "component_id": payload.get("component_code"),
        "component_name": payload.get("component_name"),
        "source_name": payload.get("source_name"),
        "status": payload.get("record_status"),
        "source": payload.get("source_name"),
        "notes": payload.get("notes") or "",
    }
    for field_name in _PROVENANCE_FIELDS:
        if field_name == "net_bridge_eligible":
            component[field_name] = _bridge_label(payload.get(field_name))
        else:
            component[field_name] = payload.get(field_name) or ""
    for adapter_field, storage_field in _technical_storage_fields(domain_key).items():
        value = payload.get(storage_field)
        component[adapter_field] = value
        component[storage_field] = value
    return component


def _component_to_storage(domain: str, component: dict, *, default_origin: str) -> dict:
    domain_key = _normalize_domain_key(domain)
    payload = dict(component or {})
    component_code = _clean_text(payload.get("component_code") or payload.get("component_id"))
    source_name = _clean_text(payload.get("source_name") or payload.get("source"))
    status = _clean_text(payload.get("record_status") or payload.get("status")).upper()
    if status not in {"ACTIVE", "ARCHIVED"}:
        status = "ACTIVE"
    origin = normalize_record_origin(EntityType.COMPONENT, payload.get("record_origin") or default_origin)
    storage = {
        "record_origin": origin,
        "record_status": status,
        "domain": domain_key,
        "component_code": component_code,
        "component_name": _clean_text(payload.get("component_name")),
        "source_name": source_name,
        "source_record_id": _clean_text(payload.get("source_record_id")) or None,
        "source_reference": _clean_text(payload.get("source_reference")) or None,
        "hardware_reference": _clean_text(payload.get("hardware_reference")) or None,
        "component_type": _clean_text(payload.get("component_type")) or None,
        "component_position": _clean_text(payload.get("component_position")) or None,
        "driveline_architecture": _clean_text(payload.get("driveline_architecture")) or None,
        "physical_boundary": _clean_text(payload.get("physical_boundary")) or None,
        "configuration_from": _clean_text(payload.get("configuration_from")) or None,
        "configuration_to": _clean_text(payload.get("configuration_to")) or None,
        "test_condition_type": _clean_text(payload.get("test_condition_type")) or None,
        "test_method": _clean_text(payload.get("test_method")) or None,
        "net_bridge_eligible": _bridge_value(payload.get("net_bridge_eligible")),
        "notes": _clean_text(payload.get("notes")),
    }
    for adapter_field, storage_field in _technical_storage_fields(domain_key).items():
        value = payload.get(storage_field) if storage_field in payload else payload.get(adapter_field)
        storage[storage_field] = _to_float(value)
    return storage


def _build_repository(domain: str, rows: list[dict], *, source: str) -> ComponentRepository:
    domain_key = normalize_domain(domain)
    issues: list[dict] = []
    components: list[dict] = []
    by_id: dict[str, dict] = {}
    for row in rows:
        component = _normalize_component_row(domain_key, row)
        component_id = component["component_id"]
        component_issues = _validate_component(domain_key, component)
        issues.extend(component_issues)
        if component_id in by_id:
            issues.append(
                _issue(
                    "duplicate_component_id",
                    "error",
                    f"Duplicate component_id '{component_id}' detected in {source}.",
                    domain=domain_key,
                    component_id=component_id,
                    field_key="component_id",
                )
            )
        else:
            by_id[component_id] = deepcopy(component)
        components.append(component)
    return ComponentRepository(domain=domain_key, source=source, _components=components, _issues=issues, _by_id=by_id)


def _csv_path_for_domain(domain: str) -> Path:
    domain_key = normalize_domain(domain)
    filename = _DOMAIN_FILENAME_MAP.get(domain_key)
    if not filename:
        raise ValueError(f"Unsupported component repository domain '{domain}'.")
    return _DATA_DIR / filename


@lru_cache(maxsize=None)
def load_mock_component_repository(domain: str) -> ComponentRepository:
    domain_key = normalize_domain(domain)
    path = _csv_path_for_domain(domain_key)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return _build_repository(domain_key, rows, source="mock_csv")


def list_mock_component_domains() -> list[str]:
    return list(_DOMAIN_FILENAME_MAP)


def load_component_repository(domain: str, *, include_archived: bool = False) -> ComponentRepository:
    """Load the operational component repository from the active SQLite DB."""
    domain_key = _normalize_domain_key(domain)
    db_module.ensure_db()
    where = "domain=?" if include_archived else "domain=? AND record_status='ACTIVE'"
    rows = db_module.fetchall(
        f"SELECT * FROM component_db WHERE {where} ORDER BY component_code ASC",
        (domain_key,),
    )
    adapter_rows = [_db_row_to_component(row) for row in rows]
    return _build_repository(domain_key, adapter_rows, source="sqlite_component_db")


def find_component_by_source_identity(
    domain: str,
    source_name: str,
    source_record_id: str,
    *,
    include_archived: bool = True,
) -> dict | None:
    domain_key = _normalize_domain_key(domain)
    source = _clean_text(source_name)
    source_id = _clean_text(source_record_id)
    if not source or not source_id:
        return None
    status_clause = "" if include_archived else " AND record_status='ACTIVE'"
    row = db_module.fetchone(
        "SELECT * FROM component_db "
        "WHERE domain=? AND source_name=? AND source_record_id=?"
        f"{status_clause} ORDER BY id ASC LIMIT 1",
        (domain_key, source, source_id),
    )
    return _db_row_to_component(row) if row else None


def component_repository_signature(domain: str) -> str:
    """Return a small cache signature that changes after repository mutations."""
    domain_key = _normalize_domain_key(domain)
    row = db_module.fetchone(
        "SELECT COUNT(*) AS row_count, MAX(id) AS max_id, "
        "MAX(COALESCE(updated_at, created_at, '')) AS last_change "
        "FROM component_db WHERE domain=?",
        (domain_key,),
    ) or {}
    return f"{row.get('row_count', 0)}:{row.get('max_id') or 0}:{row.get('last_change') or ''}"


def lookup_component(domain: str, component_id: str, repositories: dict[str, ComponentRepository] | None = None) -> dict:
    domain_key = normalize_domain(domain)
    requested_id = _clean_text(component_id)
    issues: list[dict] = []
    if domain_key == "tire":
        component = None
        if requested_id:
            try:
                component = get_tire_by_id(int(requested_id))
            except Exception:
                component = None
            if not component:
                try:
                    component = get_tire_by_code(requested_id)
                except Exception:
                    component = None
        found = bool(component)
        if not found:
            issues.append(
                _issue(
                    "component_not_found",
                    "missing",
                    f"Tire component '{requested_id}' was not found.",
                    domain=domain_key,
                    component_id=requested_id or None,
                )
            )
        return {
            "found": found,
            "domain": domain_key,
            "component_id": requested_id,
            "component": deepcopy(component) if component else None,
            "issues": issues,
            "source": "tire_service",
        }

    repo = dict(repositories or {}).get(domain_key) or load_component_repository(domain_key)
    component = repo.get_by_id(requested_id)
    found = component is not None
    if not found:
        issues.append(
            _issue(
                "component_not_found",
                "missing",
                f"Component '{requested_id}' was not found in the {domain_key} repository.",
                domain=domain_key,
                component_id=requested_id or None,
            )
        )
    else:
        issues.extend(repo.validate_component(component))
    return {
        "found": found,
        "domain": domain_key,
        "component_id": requested_id,
        "component": component,
        "issues": issues,
        "source": repo.source,
    }


def _deterministic_component_id(domain: str, payload: dict) -> str:
    domain_key = normalize_domain(domain)
    seed_fields = _DOMAIN_FIELD_MAP.get(domain_key, ())
    seed = "|".join(repr(dict(payload or {}).get(field_name)) for field_name in seed_fields)
    digest = hashlib.sha1(f"{domain_key}|{seed}".encode("utf-8")).hexdigest()[:10].upper()
    prefix = {
        "transmission": "TRANS",
        "brake": "BRAKE",
        "axle_hubs": "AXLE",
        "parasitic": "PARA",
    }.get(domain_key, domain_key.upper())
    return f"{prefix}-USER-{digest}"


def create_component(domain: str, payload: dict) -> dict:
    domain_key = _normalize_domain_key(domain)
    data = dict(payload or {})
    data.setdefault("component_code", data.get("component_id") or _deterministic_component_id(domain_key, data))
    data.setdefault("component_name", str(data.get("component_code") or f"{domain_key.title()} component"))
    data.setdefault("record_status", "ACTIVE")
    data.setdefault("source_name", data.get("source") or "manual_request")
    data.setdefault("notes", "Manual request component")
    storage = _component_to_storage(domain_key, data, default_origin="MANUAL")
    component = _db_row_to_component(storage)
    issues = _validate_component(domain_key, component)
    if issues:
        first = issues[0]
        raise ValueError(str(first.get("message") or f"Invalid {domain_key} component payload."))
    db_module.ensure_db()
    columns = [field for field in storage if field in _CANONICAL_STORAGE_FIELDS]
    placeholders = ", ".join("?" for _ in columns)
    try:
        with db_module._con() as con:
            cursor = con.execute(
                f"INSERT INTO component_db ({', '.join(columns)}) VALUES ({placeholders})",
                [storage[column] for column in columns],
            )
            row_id = int(cursor.lastrowid)
    except sqlite3.IntegrityError as exc:
        if "component_code" in str(exc):
            raise ValueError(f"Duplicate component_id '{component['component_id']}'.") from exc
        raise ValueError(f"Unable to create {domain_key} component: {exc}") from exc
    return get_component(domain_key, row_id, include_archived=True)


def get_component(domain: str, component_id: str | int, *, include_archived: bool = False) -> dict | None:
    domain_key = _normalize_domain_key(domain)
    status_clause = "" if include_archived else " AND record_status='ACTIVE'"
    identifier = _clean_text(component_id)
    row = db_module.fetchone(
        "SELECT * FROM component_db WHERE domain=? AND (component_code=? OR CAST(id AS TEXT)=?)"
        f"{status_clause} ORDER BY id ASC LIMIT 1",
        (domain_key, identifier, identifier),
    )
    return _db_row_to_component(row) if row else None


def update_component(domain: str, component_id: str | int, updates: dict) -> dict:
    domain_key = _normalize_domain_key(domain)
    current = get_component(domain_key, component_id, include_archived=True)
    if current is None:
        raise ValueError(f"Component '{component_id}' was not found in the {domain_key} repository.")
    merged = {**current, **dict(updates or {})}
    if "component_code" in updates:
        merged["component_id"] = updates["component_code"]
    for adapter_field, storage_field in _technical_storage_fields(domain_key).items():
        if adapter_field in updates:
            merged[storage_field] = updates[adapter_field]
        elif storage_field in updates:
            merged[adapter_field] = updates[storage_field]
    storage = _component_to_storage(domain_key, merged, default_origin=current.get("record_origin") or "LEGACY")
    component = _db_row_to_component(storage)
    issues = _validate_component(domain_key, component)
    if issues:
        raise ValueError(str(issues[0].get("message") or f"Invalid {domain_key} component payload."))
    storage["updated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    columns = [field for field in storage if field in _CANONICAL_STORAGE_FIELDS or field == "updated_at"]
    set_clause = ", ".join(f"{field}=?" for field in columns)
    try:
        with db_module._con() as con:
            con.execute(
                f"UPDATE component_db SET {set_clause} WHERE id=?",
                [storage[field] for field in columns] + [int(current["id"])],
            )
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"Unable to update {domain_key} component: {exc}") from exc
    return get_component(domain_key, int(current["id"]), include_archived=True)


def archive_component(domain: str, component_id: str | int) -> dict:
    return _set_component_status(domain, component_id, "ARCHIVED")


def restore_component(domain: str, component_id: str | int) -> dict:
    return _set_component_status(domain, component_id, "ACTIVE")


def _set_component_status(domain: str, component_id: str | int, status: str) -> dict:
    domain_key = _normalize_domain_key(domain)
    current = get_component(domain_key, component_id, include_archived=True)
    if current is None:
        raise ValueError(f"Component '{component_id}' was not found in the {domain_key} repository.")
    with db_module._con() as con:
        con.execute(
            "UPDATE component_db SET record_status=?, updated_at=? WHERE id=?",
            (
                status,
                datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                int(current["id"]),
            ),
        )
    return get_component(domain_key, int(current["id"]), include_archived=True)


def duplicate_component(
    domain: str,
    component_id: str | int,
    *,
    component_code: str | None = None,
    overrides: dict | None = None,
) -> dict:
    domain_key = _normalize_domain_key(domain)
    current = get_component(domain_key, component_id, include_archived=True)
    if current is None:
        raise ValueError(f"Component '{component_id}' was not found in the {domain_key} repository.")
    payload = deepcopy(current)
    for field in ("id", "created_at", "updated_at"):
        payload.pop(field, None)
    payload["component_id"] = component_code or _next_duplicate_code(domain_key, current["component_id"])
    payload["component_code"] = payload["component_id"]
    payload["record_status"] = "ACTIVE"
    payload.update(dict(overrides or {}))
    return create_component(domain_key, payload)


def _next_duplicate_code(domain: str, component_code: str) -> str:
    base = f"{_clean_text(component_code)}-COPY"
    candidate = base
    suffix = 2
    while get_component(domain, candidate, include_archived=True) is not None:
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def build_mock_component_seed_rows() -> list[dict]:
    """Convert CSV fixtures to canonical rows for an explicitly selected QA DB."""
    rows: list[dict] = []
    for domain_key in _COMPONENT_DOMAINS:
        for component in load_mock_component_repository(domain_key).list_components():
            payload = deepcopy(component)
            payload["record_origin"] = "QA"
            payload["record_status"] = "ACTIVE"
            payload["source_name"] = payload.get("source") or "mock_csv"
            payload["source_record_id"] = payload.get("component_id")
            storage = _component_to_storage(domain_key, payload, default_origin="QA")
            storage["created_at"] = _QA_SEED_TIMESTAMP
            storage["updated_at"] = _QA_SEED_TIMESTAMP
            rows.append(storage)
    return rows


__all__ = [
    "COMPONENT_PROVENANCE_FIELDS",
    "ComponentRepository",
    "archive_component",
    "build_mock_component_seed_rows",
    "component_repository_signature",
    "create_component",
    "duplicate_component",
    "find_component_by_source_identity",
    "get_component",
    "list_mock_component_domains",
    "load_component_repository",
    "load_mock_component_repository",
    "lookup_component",
    "restore_component",
    "update_component",
]
