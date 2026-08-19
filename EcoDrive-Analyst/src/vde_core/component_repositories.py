from __future__ import annotations

from copy import deepcopy
import csv
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
from pathlib import Path
import tempfile

from src.vde_core.tire_roadload_service import get_tire_by_code, get_tire_by_id
from src.vde_core.vde_request_contract import is_blank, normalize_domain


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data" / "components"
_BASE_FIELDS = ("component_id", "component_name", "status", "source", "notes")
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
                for key in (*_BASE_FIELDS, *_PROVENANCE_FIELDS)
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
    for field_name in _DOMAIN_FIELD_MAP.get(domain_key, ()):
        component[field_name] = _to_float(payload.get(field_name))
    for field_name in _PROVENANCE_FIELDS:
        component[field_name] = _clean_text(payload.get(field_name))
    return component


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

    repo = dict(repositories or {}).get(domain_key) or load_mock_component_repository(domain_key)
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


def _component_csv_fieldnames(domain: str) -> list[str]:
    domain_key = normalize_domain(domain)
    return [*_BASE_FIELDS, *_PROVENANCE_FIELDS, *_DOMAIN_FIELD_MAP.get(domain_key, ())]


def _component_storage_row(domain: str, component: dict) -> dict:
    domain_key = normalize_domain(domain)
    payload = _normalize_component_row(domain_key, component)
    row = {
        "component_id": payload.get("component_id"),
        "component_name": payload.get("component_name"),
        "status": payload.get("status"),
        "source": payload.get("source"),
        "notes": payload.get("notes"),
    }
    for field_name in _PROVENANCE_FIELDS:
        row[field_name] = payload.get(field_name) or ""
    for field_name in _DOMAIN_FIELD_MAP.get(domain_key, ()):
        value = payload.get(field_name)
        row[field_name] = "" if value is None else value
    return row


def _deterministic_component_id(domain: str, payload: dict) -> str:
    domain_key = normalize_domain(domain)
    seed_fields = _DOMAIN_FIELD_MAP.get(domain_key, ())
    seed = "|".join(str(dict(payload or {}).get(field_name) or "") for field_name in seed_fields)
    digest = hashlib.sha1(f"{domain_key}|{seed}".encode("utf-8")).hexdigest()[:10].upper()
    prefix = {
        "transmission": "TRANS",
        "brake": "BRAKE",
        "axle_hubs": "AXLE",
        "parasitic": "PARA",
    }.get(domain_key, domain_key.upper())
    return f"{prefix}-USER-{digest}"


def create_component(domain: str, payload: dict) -> dict:
    domain_key = normalize_domain(domain)
    if domain_key not in _DOMAIN_FILENAME_MAP:
        raise ValueError(f"Unsupported component repository domain '{domain}'.")

    data = dict(payload or {})
    data.setdefault("component_id", _deterministic_component_id(domain_key, data))
    data.setdefault("component_name", str(data.get("component_id") or f"{domain_key.title()} component"))
    data.setdefault("status", "user_created")
    data.setdefault("source", "manual_request")
    data.setdefault("notes", "Manual request component")
    component = _normalize_component_row(domain_key, data)
    issues = _validate_component(domain_key, component)
    if issues:
        first = issues[0]
        raise ValueError(str(first.get("message") or f"Invalid {domain_key} component payload."))

    path = _csv_path_for_domain(domain_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))

    existing_ids = {
        _clean_text(dict(row or {}).get("component_id"))
        for row in rows
        if _clean_text(dict(row or {}).get("component_id"))
    }
    if component["component_id"] in existing_ids:
        raise ValueError(f"Duplicate component_id '{component['component_id']}' detected in {path.name}.")

    fieldnames = _component_csv_fieldnames(domain_key)
    rows.append(_component_storage_row(domain_key, component))
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=str(path.parent), suffix=".tmp") as handle:
            temp_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({field_name: dict(row or {}).get(field_name, "") for field_name in fieldnames})
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)

    load_mock_component_repository.cache_clear()
    return _normalize_component_row(domain_key, component)


__all__ = [
    "COMPONENT_PROVENANCE_FIELDS",
    "ComponentRepository",
    "create_component",
    "list_mock_component_domains",
    "load_mock_component_repository",
    "lookup_component",
]
