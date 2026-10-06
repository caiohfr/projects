"""Physical contract for applying an absolute component lookup to a VDE.

Component Database ABC values are absolute component contributions.  They are
never interpreted as deltas.  A lookup may either annotate the existing total
or replace a known baseline component contribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


REQUIRES_USER_BASELINE_INPUT = "REQUIRES_USER_BASELINE_INPUT"


def _abc(payload: Mapping[str, Any] | None) -> dict[str, float | None]:
    data = dict(payload or {})
    return {
        key: None if data.get(key) is None else float(data[key])
        for key in ("A", "B", "C")
    }


def _complete(payload: Mapping[str, Any] | None) -> bool:
    values = _abc(payload)
    return all(values[key] is not None for key in ("A", "B", "C"))


@dataclass(frozen=True)
class ComponentLookupApplication:
    status: str
    recalculate_total_abc: bool
    old_total_abc: dict[str, float | None]
    new_component_abc: dict[str, float | None]
    baseline_component_abc: dict[str, float | None] | None
    delta_component_abc: dict[str, float | None] | None
    new_total_abc: dict[str, float | None]
    baseline_source: str | None


def apply_absolute_component_lookup(
    *,
    old_total_abc: Mapping[str, Any],
    new_component_abc: Mapping[str, Any],
    recalculate_total_abc: bool,
    baseline_component_abc: Mapping[str, Any] | None = None,
    baseline_source: str | None = None,
) -> ComponentLookupApplication:
    """Apply the approved lookup semantics without mutating any input.

    When recalculation is disabled, the total is byte-for-value preserved and
    no baseline component is required.  When enabled, the only permitted
    calculation is ``old_total + (new_component - baseline_component)``.
    """
    old_total = _abc(old_total_abc)
    new_component = _abc(new_component_abc)
    if not _complete(old_total):
        raise ValueError("Lookup application requires a complete existing TOTAL ABC.")
    if not _complete(new_component):
        raise ValueError("Component Database lookup requires complete absolute component ABC.")

    if not recalculate_total_abc:
        return ComponentLookupApplication(
            status="OK",
            recalculate_total_abc=False,
            old_total_abc=old_total,
            new_component_abc=new_component,
            baseline_component_abc=None,
            delta_component_abc=None,
            new_total_abc=dict(old_total),
            baseline_source=None,
        )

    if not _complete(baseline_component_abc):
        return ComponentLookupApplication(
            status=REQUIRES_USER_BASELINE_INPUT,
            recalculate_total_abc=True,
            old_total_abc=old_total,
            new_component_abc=new_component,
            baseline_component_abc=None,
            delta_component_abc=None,
            new_total_abc=dict(old_total),
            baseline_source=None,
        )

    baseline = _abc(baseline_component_abc)
    delta = {
        key: float(new_component[key]) - float(baseline[key])
        for key in ("A", "B", "C")
    }
    new_total = {
        key: float(old_total[key]) + delta[key]
        for key in ("A", "B", "C")
    }
    return ComponentLookupApplication(
        status="OK",
        recalculate_total_abc=True,
        old_total_abc=old_total,
        new_component_abc=new_component,
        baseline_component_abc=baseline,
        delta_component_abc=delta,
        new_total_abc=new_total,
        baseline_source=baseline_source,
    )
