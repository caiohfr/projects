import pandas as pd
from pathlib import Path

CYCLES_DIR = Path("data/cycles")

def list_cycles():
    """List available cycle CSVs in data/cycles/ (without extension)."""
    return [p.stem for p in CYCLES_DIR.glob("*.csv")]

def load_cycle(name: str) -> pd.DataFrame:
    """Load a cycle by name (without extension). Requires columns: t, v."""
    path = CYCLES_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cycle '{name}' not found in {CYCLES_DIR}")
    df = pd.read_csv(path)
    if not {"t","v"} <= set(df.columns):
        raise ValueError("Cycle CSV must have columns: t, v")
    return df
