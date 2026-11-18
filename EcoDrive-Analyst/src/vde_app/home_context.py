from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


VDE_SETUP_STATE_KEY = "vde_setup_v22"
LOAD_CASE_VDE_KEY = "roadload_load_case_vde_id"
PWT_BASELINE_KEY = "pwt_ss:current_baseline"
PWT_DRAFTS_KEY = "pwt_ss_drafts"
COMPARISON_SELECTION_KEY = "comparison_selection"
COMPARISON_DIRECT_VDE_KEY = "comparison_direct_vde_selection"


@dataclass(frozen=True)
class HomeContextItem:
    kind: str
    title: str
    detail: str
    source: str
    destination: str
    vde_id: int | None = None
    fuelcons_id: int | None = None


@dataclass(frozen=True)
class HomeContext:
    baseline: HomeContextItem | None
    proposal: HomeContextItem | None
    fuelcons: HomeContextItem | None
    recent: tuple[HomeContextItem, ...]
    warnings: tuple[str, ...]

    @property
    def has_active_analysis(self) -> bool:
        return any((self.baseline, self.proposal, self.fuelcons))


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _safe_lookup(lookup: Callable[[int], Mapping[str, Any] | None], record_id: int | None) -> dict[str, Any]:
    if record_id is None:
        return {}
    try:
        return dict(lookup(record_id) or {})
    except Exception:
        return {}


def _vehicle_label(row: Mapping[str, Any], record_id: int) -> str:
    identity = " ".join(
        str(row.get(field) or "").strip()
        for field in ("make", "model", "year")
        if str(row.get(field) or "").strip()
    )
    return identity or f"VDE #{record_id}"


def _proposal_is_configured(proposal: Mapping[str, Any]) -> bool:
    if str(proposal.get("name") or "").strip() or dict(proposal.get("inputs") or {}):
        return True
    for domain in dict(proposal.get("domains") or {}).values():
        if str(dict(domain or {}).get("proposal_type") or "INHERIT").upper() != "INHERIT":
            return True
    return False


def _selection_ids(selection: Any) -> tuple[int, ...]:
    if selection is None:
        return ()
    reference = _positive_int(getattr(selection, "reference_fuelcons_id", None))
    comparisons = getattr(selection, "comparison_fuelcons_ids", ()) or ()
    values = [reference, *(_positive_int(value) for value in comparisons)]
    return tuple(value for value in values if value is not None)


def resolve_home_context(
    session_state: Mapping[str, Any],
    *,
    fetch_vde: Callable[[int], Mapping[str, Any] | None] | None = None,
    fetch_fuelcons: Callable[[int], Mapping[str, Any] | None] | None = None,
) -> HomeContext:
    """Build Home state from recoverable application state without computing physics."""

    if fetch_vde is None or fetch_fuelcons is None:
        from src.vde_core.repositories import fetch_fuelcons_by_id, fetch_vde_by_id

        fetch_vde = fetch_vde or fetch_vde_by_id
        fetch_fuelcons = fetch_fuelcons or fetch_fuelcons_by_id

    warnings: list[str] = []
    recent: list[HomeContextItem] = []
    seen: set[tuple[str, int]] = set()

    def vde_item(vde_id: int | None, *, kind: str, source: str, destination: str) -> HomeContextItem | None:
        if vde_id is None:
            return None
        row = _safe_lookup(fetch_vde, vde_id)
        if not row:
            warnings.append(f"{source}: VDE #{vde_id} is no longer available in the active database.")
            return None
        return HomeContextItem(
            kind=kind,
            title=_vehicle_label(row, vde_id),
            detail=f"Persisted VDE #{vde_id}",
            source=source,
            destination="pages/VDE_Setup.py" if destination == "vde" else destination,
            vde_id=vde_id,
        )

    def fuelcons_item(
        fuelcons_id: int | None,
        *,
        source: str,
        destination: str = "pages/Powertrain_Scenario.py",
    ) -> HomeContextItem | None:
        if fuelcons_id is None:
            return None
        row = _safe_lookup(fetch_fuelcons, fuelcons_id)
        if not row:
            warnings.append(f"{source}: FuelCons #{fuelcons_id} is no longer available in the active database.")
            return None
        vde_id = _positive_int(row.get("vde_id"))
        vde_row = _safe_lookup(fetch_vde, vde_id)
        label = _vehicle_label(vde_row, vde_id) if vde_id and vde_row else f"FuelCons #{fuelcons_id}"
        return HomeContextItem(
            kind="fuelcons",
            title=label,
            detail=f"Persisted FuelCons #{fuelcons_id}",
            source=source,
            destination=destination,
            vde_id=vde_id,
            fuelcons_id=fuelcons_id,
        )

    v22 = session_state.get(VDE_SETUP_STATE_KEY)
    v22 = dict(v22) if isinstance(v22, Mapping) else {}
    baseline_id = _positive_int(dict(v22.get("baseline") or {}).get("selected_vde_id"))
    baseline = vde_item(
        baseline_id,
        kind="baseline",
        source="VDE Setup session",
        destination="pages/VDE_Setup.py",
    )

    configured = [dict(item) for item in list(v22.get("proposals") or []) if _proposal_is_configured(dict(item or {}))]
    proposal = None
    if configured:
        preview_status = str(dict(v22.get("preview") or {}).get("status") or "not run").replace("_", " ")
        proposal = HomeContextItem(
            kind="proposal",
            title=f"{len(configured)} temporary proposal{'s' if len(configured) != 1 else ''}",
            detail=f"Preview status: {preview_status}",
            source="Current Streamlit session · not persisted",
            destination="pages/VDE_Setup.py",
            vde_id=baseline.vde_id if baseline else None,
        )

    save_result = dict(dict(v22.get("save") or {}).get("result") or {})
    for row in list(save_result.get("saved_proposals") or []):
        saved_id = _positive_int(dict(row or {}).get("vde_row_id"))
        item = vde_item(saved_id, kind="saved_vde", source="VDE Setup save result", destination="pages/VDE_Setup.py")
        if item and (item.kind, int(item.vde_id or 0)) not in seen:
            recent.append(item)
            seen.add((item.kind, int(item.vde_id or 0)))

    selected_fuelcons_id = _positive_int(session_state.get(PWT_BASELINE_KEY))
    if selected_fuelcons_id is None:
        for draft in tuple(session_state.get(PWT_DRAFTS_KEY) or ()):
            identity = getattr(getattr(draft, "identity", None), "role", None)
            if str(getattr(identity, "value", identity) or "").upper() == "CURRENT":
                selected_fuelcons_id = _positive_int(getattr(draft, "fuelcons_id", None))
                break
    fuelcons = fuelcons_item(selected_fuelcons_id, source="Powertrain Current selection")

    lab_id = _positive_int(session_state.get(LOAD_CASE_VDE_KEY))
    lab = vde_item(lab_id, kind="load_case", source="Load Case Lab session", destination="pages/Roadload_Load_Case_Lab.py")
    if lab:
        recent.append(lab)

    for fuel_id in _selection_ids(session_state.get(COMPARISON_SELECTION_KEY)):
        item = fuelcons_item(
            fuel_id,
            source="Comparison selection",
            destination="pages/Comparison_Report.py",
        )
        key = ("fuelcons", fuel_id)
        if item and key not in seen:
            recent.append(item)
            seen.add(key)

    for vde_id in _selection_ids(session_state.get(COMPARISON_DIRECT_VDE_KEY)):
        item = vde_item(vde_id, kind="comparison_vde", source="Comparison direct VDE selection", destination="pages/Comparison_Report.py")
        key = ("vde", vde_id)
        if item and key not in seen:
            recent.append(item)
            seen.add(key)

    if fuelcons:
        key = ("fuelcons", int(fuelcons.fuelcons_id or 0))
        if key not in seen:
            recent.append(fuelcons)

    return HomeContext(
        baseline=baseline,
        proposal=proposal,
        fuelcons=fuelcons,
        recent=tuple(recent[:6]),
        warnings=tuple(dict.fromkeys(warnings)),
    )
