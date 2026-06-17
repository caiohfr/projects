from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


_VDE_REPORT_SQL = "SELECT * FROM vde_db ORDER BY COALESCE(updated_at, created_at) DESC;"


def load_vde_report_frame(db_path: str | Path) -> pd.DataFrame:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"DB not found: {path}")

    with sqlite3.connect(str(path), timeout=30, isolation_level=None) as con:
        frame = pd.read_sql_query(_VDE_REPORT_SQL, con)

    if frame.empty:
        return frame

    frame = ensure_report_abc_aliases(frame)
    frame["veh_label"] = frame.apply(build_vehicle_label, axis=1)
    return frame


def ensure_report_abc_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "A" not in out.columns and "coast_A_N" in out.columns:
        out["A"] = out["coast_A_N"]
    if "B" not in out.columns and "coast_B_N_per_kph" in out.columns:
        out["B"] = out["coast_B_N_per_kph"]
    if "C" not in out.columns and "coast_C_N_per_kph2" in out.columns:
        out["C"] = out["coast_C_N_per_kph2"]
    return out


def build_vehicle_label(row: pd.Series) -> str:
    parts = [str(row.get("make", "")).strip(), str(row.get("model", "")).strip()]
    year = row.get("year", None)
    if pd.notna(year):
        parts.append(str(int(year)))
    return " ".join(part for part in parts if part)
