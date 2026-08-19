from __future__ import annotations

import numpy as np


def canonical_cycle_segments(cycle_frame) -> dict[str, object]:
    """Return physical, selectable cycle segments without exposing aggregates."""
    if cycle_frame is None or getattr(cycle_frame, "empty", True):
        return {}
    frame = cycle_frame.copy()
    if "phase" not in frame.columns:
        return {"Cycle": frame}
    phases = frame["phase"].astype(str).str.strip()
    lowered = phases.str.lower()
    segments: dict[str, object] = {}
    if lowered.str.startswith("bag").any():
        segments["FTP-75"] = frame[lowered.str.startswith("bag")].copy()
    if (lowered == "hwfet").any():
        segments["HWFET"] = frame[lowered == "hwfet"].copy()
    if not segments:
        labels = {
            "low": "Low",
            "mid": "Medium",
            "medium": "Medium",
            "high": "High",
            "xhigh": "Extra High",
            "extra high": "Extra High",
        }
        for phase in phases.drop_duplicates():
            mask = phases == phase
            segments[labels.get(str(phase).lower(), str(phase).title())] = frame[mask].copy()
    return segments or {"Cycle": frame}


def build_cycle_power_analysis(cycle_frame, scenarios: list[dict]) -> dict:
    """Build ready-to-render TOTAL/NET demanded-power series from resolved ABC.

    Component attribution is deliberately omitted: measured coastdown may include
    residual roadload that cannot be assigned to a named physical component.
    """
    if cycle_frame is None or getattr(cycle_frame, "empty", True):
        return {"time_s": [], "speed_kph": [], "series": [], "decomposition_available": False}
    time_s = np.asarray(cycle_frame["t"], dtype=float)
    speed_mps = np.asarray(cycle_frame["v"], dtype=float)
    acceleration_mps2 = np.gradient(speed_mps, time_s) if len(time_s) > 1 else np.zeros_like(speed_mps)
    series = []
    for scenario in list(scenarios or []):
        mass_kg = _positive_float(scenario.get("mass_kg"))
        inertial_power_kw = (mass_kg * acceleration_mps2 * speed_mps / 1000.0) if mass_kg is not None else None
        for boundary in ("TOTAL", "NET"):
            abc = dict(scenario.get(boundary.lower()) or {})
            if not _abc_complete(abc):
                continue
            force_n = np.asarray(roadload_force_N(abc["A"], abc["B"], abc["C"], speed_mps * 3.6), dtype=float)
            roadload_power_kw = force_n * speed_mps / 1000.0
            demanded_power_kw = roadload_power_kw + inertial_power_kw if inertial_power_kw is not None else roadload_power_kw
            series.append(
                {
                    "scenario_id": str(scenario.get("id") or ""),
                    "scenario_label": str(scenario.get("label") or "Scenario"),
                    "boundary": boundary,
                    "roadload_power_kw": roadload_power_kw.tolist(),
                    "inertial_power_kw": inertial_power_kw.tolist() if inertial_power_kw is not None else None,
                    "demanded_power_kw": demanded_power_kw.tolist(),
                    "mass_kg": mass_kg,
                }
            )
    return {
        "time_s": time_s.tolist(),
        "speed_kph": (speed_mps * 3.6).tolist(),
        "acceleration_mps2": acceleration_mps2.tolist(),
        "series": series,
        "decomposition_available": False,
        "decomposition_note": "Component attribution is unavailable because measured roadload may include residual or other unassigned losses.",
    }


def roadload_force_N(A, B, C, speed_kph):
    a_value = float(A)
    b_value = float(B)
    c_value = float(C)
    if isinstance(speed_kph, (int, float)):
        speed_value = float(speed_kph)
        return a_value + (b_value * speed_value) + (c_value * (speed_value ** 2))
    return [
        a_value + (b_value * float(speed_value)) + (c_value * (float(speed_value) ** 2))
        for speed_value in speed_kph
    ]


def build_roadload_curve(
    abc,
    speed_min_kph: int = 0,
    speed_max_kph: int = 140,
    step_kph: int = 1,
) -> dict:
    triplet = dict(abc or {})
    start = int(speed_min_kph)
    stop = int(speed_max_kph)
    step = max(int(step_kph), 1)
    if stop < start:
        start, stop = stop, start
    speed_kph = list(range(start, stop + 1, step))
    return {
        "speed_kph": speed_kph,
        "force_N": roadload_force_N(
            triplet.get("A"),
            triplet.get("B"),
            triplet.get("C"),
            speed_kph,
        ),
    }


def _abc_complete(abc: dict) -> bool:
    try:
        return all(abc.get(key) not in (None, "") for key in ("A", "B", "C"))
    except AttributeError:
        return False


def _positive_float(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


__all__ = ["build_cycle_power_analysis", "build_roadload_curve", "canonical_cycle_segments", "roadload_force_N"]
