"""
Compatibility shim for experimental technology-effect helpers.

The canonical location for these heuristics is now:
    src.vde_core.experimental.tech_effects
"""

from src.vde_core.experimental.tech_effects import apply_tech_effects, estimate_eta_pt

__all__ = ["apply_tech_effects", "estimate_eta_pt"]
