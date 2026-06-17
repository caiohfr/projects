"""
Compatibility shim for older cycle-loader imports.

Historically, this module provided lightweight CSV helpers. The active cycle
loading path is now exposed through ``src.vde_core.cycles`` and the underlying
implementations still live in ``src.vde_core.services``.
"""

from pathlib import Path

import pandas as pd

from src.vde_core.cycles import (
    cycle_summary,
    default_cycle_for_legislation,
    load_cycle_csv,
    use_standard_cycle,
)


def list_cycles():
    cycles_dir = Path("data/cycles")
    return [path.stem for path in cycles_dir.glob("*.csv")]


def load_cycle(name: str):
    return load_cycle_csv(name)


def load_tire_size_reference(csv_path: str | Path = "data/reference/tire_size_reference.csv") -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "size_code",
                "width_mm",
                "aspect_ratio_pct",
                "rim_in",
                "unloaded_diameter_mm",
                "unloaded_radius_mm",
                "unloaded_circumference_mm",
                "dynamic_factor",
                "expected_rolling_radius_mm",
                "expected_effective_circumference_mm",
                "source",
                "notes",
            ]
        )
    return pd.read_csv(path)


def lookup_tire_size_reference(size_code: str, csv_path: str | Path = "data/reference/tire_size_reference.csv") -> dict:
    df = load_tire_size_reference(csv_path)
    if df.empty or not size_code:
        return {}
    match = df[df["size_code"].astype(str).str.upper() == str(size_code).upper()]
    if match.empty:
        return {}
    return match.iloc[0].to_dict()


__all__ = [
    "cycle_summary",
    "default_cycle_for_legislation",
    "list_cycles",
    "load_cycle",
    "load_cycle_csv",
    "load_tire_size_reference",
    "lookup_tire_size_reference",
    "use_standard_cycle",
]
