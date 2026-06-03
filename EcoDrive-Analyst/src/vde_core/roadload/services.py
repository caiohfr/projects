"""
Compatibility shim for legacy imports.

Historically, this module duplicated much of ``src.vde_core.services`` while
the RoadLoad package was being explored. The active application path now uses:
    models.py -> adapters.py -> engine.py -> app_service.py

To avoid maintaining two copies of the same core helpers, legacy imports from
``src.vde_core.roadload.services`` are redirected to ``src.vde_core.services``.
"""

from src.vde_core.services import *  # noqa: F401,F403
