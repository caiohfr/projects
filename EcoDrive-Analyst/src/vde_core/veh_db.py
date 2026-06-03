"""
Compatibility shim for the experimental CSV-based vehicle repository.

The canonical location for these helpers is now:
    src.vde_core.experimental.vehicle_csv_repo
"""

from src.vde_core.experimental.vehicle_csv_repo import (
    list_models,
    list_size_classes,
    list_standards,
    load_vehicle_db,
    pick_vehicle_row,
)

__all__ = [
    "list_models",
    "list_size_classes",
    "list_standards",
    "load_vehicle_db",
    "pick_vehicle_row",
]
