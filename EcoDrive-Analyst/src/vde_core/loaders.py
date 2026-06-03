"""
Compatibility shim for older cycle-loader imports.

Historically, this module provided lightweight CSV helpers. The active cycle
loading path is now exposed through ``src.vde_core.cycles`` and the underlying
implementations still live in ``src.vde_core.services``.
"""

from src.vde_core.cycles import (
    cycle_summary,
    default_cycle_for_legislation,
    load_cycle_csv,
    use_standard_cycle,
)


def list_cycles():
    from pathlib import Path

    cycles_dir = Path("data/cycles")
    return [path.stem for path in cycles_dir.glob("*.csv")]


def load_cycle(name: str):
    return load_cycle_csv(name)


__all__ = [
    "cycle_summary",
    "default_cycle_for_legislation",
    "list_cycles",
    "load_cycle",
    "load_cycle_csv",
    "use_standard_cycle",
]
