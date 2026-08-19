from __future__ import annotations

from copy import deepcopy

from src.vde_core.vde_request_compact_state import normalize_v22_state, resolve_v22_metadata_contexts
from src.vde_core.vde_request_contract import is_blank


_DOMAIN_LABELS = {
    "mass": "Mass",
    "aero": "Aero",
    "tire": "Tire",
    "transmission": "Transmission",
    "brake": "Brake",
    "axle_hubs": "Axle/Hubs",
    "parasitic": "Parasitics",
}
_COMPONENT_ID_FIELDS = {
    "transmission": ("transmission_component_name", "transmission_component_code", "transmission_component_db_id", "transmission_vde_db_id"),
    "brake": ("brake_component_name", "brake_component_code", "brake_component_db_id", "brake_vde_db_id"),
    "axle_hubs": ("axle_hubs_component_name", "axle_hubs_component_code", "axle_hubs_component_db_id", "axle_hubs_vde_db_id"),
    "parasitic": ("parasitic_component_name", "parasitic_component_code", "parasitic_component_db_id", "parasitic_vde_db_id"),
}
_MASS_LABELS = {
    "EPA_CURB_TO_TWC": "Curb mass",
    "MASS_TWC_SHIFT": "TWC Shift",
    "PERFORMANCE_CURB_MASS": "Performance loaded mass",
    "WLTP_MASS_LINE": "WLTP mass line",
    "GVWR": "GVWR loaded mass",
    "GCWR": "GCWR / trailer mass",
    "CUSTOM_MASS": "Custom test mass",
}


def build_scenario_configuration_summaries(state: dict | None) -> list[dict]:
    """Return concise, display-only finalization summaries for requested scenarios."""
    normalized = normalize_v22_state(state)
    contexts = resolve_v22_metadata_contexts(normalized)
    summaries: list[dict] = []
    for proposal in list(normalized.get("proposals") or []):
        proposal_id = str(proposal.get("proposal_id") or "")
        context = dict(contexts.get(proposal_id) or {})
        metadata = deepcopy(dict(context.get("effective_metadata") or {}))
        display_index = int(proposal.get("display_index") or len(summaries) + 1)
        changes = _direct_change_labels(proposal)
        program_label = _program_label(metadata, fallback=f"Requested #{display_index}")
        engineering_summary = _compact_change_summary(changes)
        suggested_name = _suggested_name(program_label, changes)
        summaries.append(
            {
                "proposal_id": proposal_id,
                "proposal_label": f"Requested #{display_index}",
                "program_label": program_label,
                "engineering_summary": engineering_summary,
                "based_on": _walk_from_label(normalized, proposal.get("walk_from")),
                "effective_metadata": metadata,
                "metadata_source": str(proposal.get("metadata_source") or "inherit"),
                "suggested_name": suggested_name,
                "direct_changes": changes,
            }
        )
    return summaries


def suggested_scenario_name(summary: dict, current_name=None) -> str:
    """Use a concise generated name only until a user supplies a meaningful one."""
    current = str(current_name or "").strip()
    proposal_label = str(dict(summary or {}).get("proposal_label") or "").strip().lower()
    if current and current.lower() != proposal_label.lower():
        return current
    return str(dict(summary or {}).get("suggested_name") or current)


def _direct_change_labels(proposal: dict) -> list[str]:
    labels = []
    for domain, payload in dict(proposal.get("domains") or {}).items():
        details = dict(payload or {})
        proposal_type = str(details.get("proposal_type") or "INHERIT").strip().upper()
        if proposal_type == "INHERIT":
            continue
        inputs = dict(dict(proposal.get("inputs") or {}).get(domain) or {})
        labels.append(_domain_change_label(str(domain), proposal_type, inputs))
    return labels


def _domain_change_label(domain: str, proposal_type: str, inputs: dict) -> str:
    domain_label = _DOMAIN_LABELS.get(domain, domain.replace("_", " ").title())
    if proposal_type.endswith("METADATA_ONLY"):
        return f"{domain_label} not used"
    if domain == "mass":
        return _MASS_LABELS.get(proposal_type, "Mass Proposal")
    if domain == "aero":
        return "Aero Proposal"
    if domain == "tire":
        tire_code = _short_identifier(inputs.get("tire_code") or inputs.get("new_tire_code"))
        return f"Tire {tire_code}" if tire_code else "Tire Proposal"
    if domain in _COMPONENT_ID_FIELDS:
        identifier = _first_text(inputs, _COMPONENT_ID_FIELDS[domain])
        return f"{domain_label} {_short_identifier(identifier)}" if identifier else f"{domain_label} Proposal"
    return f"{domain_label} Proposal"


def _program_label(metadata: dict, *, fallback: str) -> str:
    make = _short_identifier(metadata.get("make"))
    model = _short_identifier(metadata.get("model"))
    year = _model_year_label(metadata.get("model_year"))
    identity = " ".join(part for part in (make, model) if part)
    if not identity:
        identity = _short_identifier(metadata.get("name")) or fallback
    return f"{identity} · {year}" if year else identity


def _model_year_label(value) -> str:
    text = str(value or "").strip()
    if len(text) == 4 and text.isdigit():
        return f"MY{text[-2:]}"
    return f"MY{text}" if text else ""


def _compact_change_summary(changes: list[str]) -> str:
    if not changes:
        return "No direct engineering changes"
    if len(changes) <= 4:
        return " + ".join(changes)
    return " + ".join(changes[:4]) + f" + {len(changes) - 4} more"


def _suggested_name(program_label: str, changes: list[str]) -> str:
    identity = str(program_label or "").replace(" · ", " ").strip()
    change_text = " + ".join(changes[:3]) if changes else "Scenario"
    return f"{identity} - {change_text}".strip(" -")


def _walk_from_label(state: dict, walk_from) -> str:
    value = str(walk_from or "baseline")
    if value == "baseline":
        return "Baseline"
    for proposal in list(state.get("proposals") or []):
        if str(proposal.get("proposal_id") or "") == value:
            return f"Requested #{int(proposal.get('display_index') or 0)}"
    return value


def _first_text(values: dict, field_names: tuple[str, ...]) -> str:
    for field_name in field_names:
        value = values.get(field_name)
        if not is_blank(value):
            return str(value).strip()
    return ""


def _short_identifier(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    for prefix in ("TIRE-", "COMPONENT-"):
        if text.upper().startswith(prefix):
            text = text[len(prefix):]
    return text[:40]


__all__ = ["build_scenario_configuration_summaries", "suggested_scenario_name"]
