# scripts/init_db.py
from src.vde_core.db import ensure_db, fetchall

if __name__ == "__main__":
    ensure_db()
    print("OK: schema criado/checado.")
    # sanity: lista tabelas
    rows = fetchall("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    print("Tabelas:", [r["name"] for r in rows])
