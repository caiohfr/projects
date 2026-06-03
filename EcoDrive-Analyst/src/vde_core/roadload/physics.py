"""
Compatibility shim for legacy RoadLoad physical helpers.

The previous experimental implementation relied on model classes that no longer
belong to the active RoadLoad domain. The preserved, self-contained reference
implementation now lives in:
    src.vde_core.roadload.physics_legacy
"""

from src.vde_core.roadload.physics_legacy import *  # noqa: F401,F403
