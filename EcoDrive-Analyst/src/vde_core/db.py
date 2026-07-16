# src/vde_core/db.py
# -----------------------------------------------------------------------------
# EN: Simple SQLite helpers for the EcoDrive project (CS50-style).
#     This module creates the database (if missing), defines two main tables:
#       - vde_db:   physical setup + results for a single VDE snapshot
#       - fuelcons_db: fuel/energy scenarios tied to one VDE snapshot (via vde_id)
#     It also exposes small helpers to insert/update/select rows.
#
# PT: UtilitÃ¡rios simples de SQLite para o projeto EcoDrive (estilo CS50).
#     Este mÃ³dulo cria o banco (se nÃ£o existir) e define duas tabelas principais:
#       - vde_db:   setup fÃ­sico + resultados de um snapshot VDE
#       - fuelcons_db: cenÃ¡rios de consumo/energia ligados a um VDE (via vde_id)
#     TambÃ©m fornece funÃ§Ãµes para inserir/atualizar/buscar linhas.
# -----------------------------------------------------------------------------

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime
from .services import autoresolve_test_mass
# -----------------------------------------------------------------------------
# EN: Database file path. We make sure the folder "data/db" exists.
# PT: Caminho do arquivo do banco. Garantimos que a pasta "data/db" exista.
# -----------------------------------------------------------------------------
DEFAULT_DB_PATH = Path("data/db/eco_drive.db")
DB_PATH_ENV_VAR = "ECO_DRIVE_DB_PATH"


def _normalize_db_path(path_like) -> Path:
    path = Path(path_like).expanduser() if path_like not in (None, "") else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _db_path_from_env() -> Path:
    return _normalize_db_path(os.environ.get(DB_PATH_ENV_VAR) or DEFAULT_DB_PATH)


DB_PATH = _db_path_from_env()


def configure_db_path(path_like=None) -> Path:
    """Configure the active SQLite path used by repository helpers."""
    global DB_PATH
    DB_PATH = _normalize_db_path(path_like if path_like not in (None, "") else _db_path_from_env())
    return DB_PATH


def current_db_path() -> Path:
    return Path(DB_PATH)


@contextmanager
def using_db_path(path_like):
    original = Path(DB_PATH)
    configure_db_path(path_like)
    try:
        yield Path(DB_PATH)
    finally:
        configure_db_path(original)


def _con():
    """
    EN: Open a SQLite connection and enable foreign key constraints.
    PT: Abre uma conexÃ£o SQLite e habilita foreign keys (ON DELETE CASCADE etc.).
    """
    con = sqlite3.connect(DB_PATH)
    # Very important for REFERENCES ... ON DELETE CASCADE to work in SQLite
    # Muito importante para REFERENCES ... ON DELETE CASCADE funcionar no SQLite
    con.execute("PRAGMA foreign_keys = ON")
    return con

# --- Lightweight, idempotent migrations -------------------------------------
def ensure_columns(table: str, spec: dict[str, str]) -> list[str]:
    """
    Cria colunas que nÃ£o existirem em 'table'.
    spec: {col_name: "SQL_TYPE"}  -> ex.: {"vde_urb_mj_per_km": "REAL"}
    Retorna a lista de colunas adicionadas.
    """
    with _con() as con:
        cur = con.cursor()
        cur.execute(f"PRAGMA table_info({table});")
        existing = {row[1] for row in cur.fetchall()}
        missing = [(c, t) for c, t in spec.items() if c not in existing]
        for col, typ in missing:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ};")
        con.commit()
    return [c for c, _ in missing]

def ensure_migrations() -> None:
    """Aplica migracoes idempotentes necessarias ao schema atual."""
    added = []
    added += ensure_columns("vde_db", {
        "drive_type": "TEXT",
        "mro_kg": "REAL",
        "options_kg": "REAL",
        "wltp_category": "INT",
        "test_mass_low_kg": "REAL",
        "test_mass_high_kg": "REAL",
        "test_mass_basis": "TEXT",
        "vde_urb_mj_per_km": "REAL",
        "vde_hw_mj_per_km": "REAL",
        "parasitic_A_coef_N": "REAL",
        "parasitic_B_coef_Npkph": "REAL",
        "parasitic_C_coef_Npkph2": "REAL",
        "rrc_N_per_kN": "REAL",
        "crr1_frac_at_120kph": "REAL",
    })
    added += ensure_columns("fuelcons_db", {
        "energy_basis": "TEXT",
        "engine_method": "TEXT",
        "engine_version": "TEXT",
        "source_vde_revision": "TEXT",
        "assumptions_json": "TEXT",
        "provenance_json": "TEXT",
    })
    added += ensure_columns("vde_db", {
        "test_mass_kg": "REAL",
        "vde_id_parent": "INTEGER",
        "baseline_A_N": "REAL",
        "baseline_B_N_per_kph": "REAL",
        "baseline_C_N_per_kph2": "REAL",
        "baseline_mass_kg": "REAL",
        "delta_rr_N": "REAL",
        "delta_brake_N": "REAL",
        "delta_parasitics_N": "REAL",
        "delta_aero_Npkph2": "REAL",
        "delta_mass_kg": "REAL",
    })
    added += ensure_columns("vde_db", {
        "front_tire_id": "INTEGER",
        "rear_tire_id": "INTEGER",
        "tire_improvement_pct": "REAL",
        "tire_load_mass_basis": "TEXT",
        "tire_load_mass_used_kg": "REAL",
        "tire_A_final": "REAL",
        "tire_B_final": "REAL",
        "tire_C_final": "REAL",
        "tire_calc_source": "TEXT",
        "tire_calc_notes": "TEXT",
    })
    added += ensure_columns("vde_db", {
        "gvwr_kg": "REAL",
        "gcwr_kg": "REAL",
        "trailer_mass_kg": "REAL",
        "trailer_code": "TEXT",
        "trailer_roadload_source": "TEXT",
        "trailer_A_coef_N": "REAL",
        "trailer_B_coef_Npkph": "REAL",
        "trailer_C_coef_Npkph2": "REAL",
        "mass_rule_status": "TEXT",
        "mass_rule_notes": "TEXT",
    })
    added += ensure_columns("tire_roadload_db", {
        "calculation_mode": "TEXT",
        "effective_circumference_override_mm": "REAL",
        "sae_reference_pressure_kpa": "REAL",
        "sae_reference_load_n": "REAL",
        "sae_pressure_unit": "TEXT",
        "sae_load_unit": "TEXT",
        "sae_speed_unit": "TEXT",
        "sae_force_unit": "TEXT",
        "rr_method": "TEXT",
        "rr_source": "TEXT",
        "rr_quality": "TEXT",
        "conditioning_notes": "TEXT",
        "test_pressure_value": "REAL",
        "test_load_value": "REAL",
        "test_speed_value": "REAL",
        "smerf": "REAL",
    })
    if added:
        print("[db] migracoes aplicadas:", added)

def ensure_db():
    """
    EN: Create tables and indexes if they do not exist.
    PT: Cria tabelas e Ã­ndices caso nÃ£o existam.
    """
    with _con() as con:
        cur = con.cursor()

        # ---------------------------------------------------------------------
        # EN: Main VDE table
        #     Holds a single test "snapshot": inputs (mass, tires, road-load),
        #     powertrain descriptors, and VDE outputs (NET/TOTAL, WLTP phases).
        #
        # PT: Tabela principal VDE
        #     Guarda um "snapshot" de teste: entradas (massa, pneus, road-load),
        #     descriÃ§Ã£o do powertrain e saÃ­das do VDE (NET/TOTAL, fases WLTP).
        # ---------------------------------------------------------------------
        cur.execute("""
        CREATE TABLE IF NOT EXISTS vde_db (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at           TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at           TEXT,

            -- meta / classificaÃ§Ã£o
            legislation          TEXT NOT NULL,             -- 'EPA' | 'WLTP' | 'BRA'
            category             TEXT NOT NULL,             -- classe oficial (EPA/WLTP)
            make                 TEXT NOT NULL,
            model                TEXT NOT NULL,
            year                 INTEGER,
            notes                TEXT,

            -- powertrain bÃ¡sico (seu padrÃ£o)
            engine_type          TEXT,                      -- 'SI' | 'CI' | 'HEV' | 'BEV'
            engine_model         TEXT,                      -- ex.: 'Firefly 1.3', 'Pentastar V6'
            engine_size_l        REAL,                      -- ex.: 1.3, 2.0
            engine_aspiration    TEXT,                      -- 'NA','Turbo','Supercharged'
            transmission_type    TEXT,                      -- 'AT','DCT','CVT','MT'
            transmission_model   TEXT,                      -- ex.: 'Aisin AWF8F35','Getrag DCT250'
            
            -- massa e aero
            mass_kg              REAL NOT NULL,
            test_mass_kg         REAL,
            test_mass_low_kg     REAL,
            test_mass_high_kg    REAL,
            test_mass_basis      TEXT,
            gvwr_kg              REAL,
            gcwr_kg              REAL,
            trailer_mass_kg      REAL,
            trailer_code         TEXT,
            trailer_roadload_source TEXT,
            trailer_A_coef_N     REAL,
            trailer_B_coef_Npkph REAL,
            trailer_C_coef_Npkph2 REAL,
            mass_rule_status     TEXT,
            mass_rule_notes      TEXT,
            inertia_class        REAL,                      -- ETW / InÃ©rcia WLTP / NBR
            cda_m2               REAL,                      -- Cd*Af_FE
            weight_dist_fr_pct   REAL,                      -- distribuiÃ§Ã£o F/R [%]
            payload_kg           REAL,                      -- carga Ãºtil

            -- pneus (snapshot do ensaio)
            tire_size            TEXT,
            tire_rr_note         TEXT,
            smerf                REAL,
            front_pressure_psi   REAL,
            rear_pressure_psi    REAL,

            -- coeficientes coastdown principais
            coast_A_N            REAL,
            coast_B_N_per_kph    REAL,
            coast_C_N_per_kph2   REAL,

            -- coeficientes adicionais (transmissÃ£o/freios/aero) - opcionais
            trans_A_coef_N       REAL,
            trans_B_coef_Npkph   REAL,
            trans_C_coef_Npkph2  REAL,
            brake_A_coef_N       REAL,
            brake_B_coef_Npkph   REAL,
            brake_C_coef_Npkph2  REAL,
            aero_C_coef_Npkph2   REAL,

            -- modelo RR avanÃ§ado (opcional)
            rr_alpha_N           REAL,
            rr_beta_Npkph        REAL,
            rr_a_Npkph2          REAL,
            rr_b_N               REAL,
            rr_c_Npkph           REAL,
            rr_load_kpa          REAL,

            -- ciclo
            cycle_name           TEXT,                      -- 'FTP-75_HWFET','WLTC Class 3','custom'
            cycle_source         TEXT,                      -- 'standard:EPA','standard:WLTP','custom:upload'

            -- resultados agregados
            vde_urb_mj           REAL,                      -- energia total ciclo urbano (quando aplicÃ¡vel)
            vde_hw_mj            REAL,                      -- energia total ciclo rodoviÃ¡rio (quando aplicÃ¡vel)

            vde_net_mj_per_km    REAL,                      -- NET (comparabilidade normativa)
            vde_total_mj_per_km  REAL,                      -- TOTAL (NET + perdas transmissÃ£o)

            -- quatro fases WLTP em MJ/km
            vde_low_mj_per_km        REAL,
            vde_mid_mj_per_km        REAL,
            vde_high_mj_per_km       REAL,
            vde_extra_high_mj_per_km REAL
        );
        """)

        # Helpful indexes for faster filtering in dashboards/queries
        # Ãndices Ãºteis para acelerar filtros em dashboards/consultas
        cur.execute("CREATE INDEX IF NOT EXISTS idx_vde_cat_leg ON vde_db(category, legislation);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_vde_make_model ON vde_db(make, model, year);")

        # ---------------------------------------------------------------------
        # EN: Fuel/Energy scenarios linked to a given VDE (via vde_id).
        #     We store complementary scenario info, outputs per WLTP/EPA phase,
        #     and labeling/off-cycle fields. We avoid duplicating what's in vde_db.
        #
        # PT: CenÃ¡rios de consumo/energia vinculados a um VDE (via vde_id).
        #     Guardamos informaÃ§Ãµes de cenÃ¡rio complementares, saÃ­das por fase
        #     WLTP/EPA e campos de rotulagem/ajustes off-cycle. Evitamos duplicar
        #     o que jÃ¡ estÃ¡ no vde_db.
        # ---------------------------------------------------------------------
        cur.execute("""
        CREATE TABLE IF NOT EXISTS fuelcons_db (
            id                           INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at                   TEXT DEFAULT CURRENT_TIMESTAMP,

            -- vÃ­nculo com VDE
            vde_id                       INTEGER NOT NULL REFERENCES vde_db(id) ON DELETE CASCADE,

            -- snapshot complementar (fÃ¡cil de obter / nÃ£o duplicar vde_db)
            electrification              TEXT NOT NULL,        -- 'None','MHEV','HEV','PHEV','BEV'
            fuel_type                    TEXT,                 -- 'Gasoline','Ethanol','Flex','Diesel','Electric'
            eta_pt_est                   REAL,                 -- rendimento efetivo mÃ©dio (ICE/HEV/PHEV)
            bev_eff_drive                REAL,                 -- eficiÃªncia BEV
            utility_factor_pct           REAL,                 -- UF (PHEV), se aplicÃ¡vel

            -- performance simples (nÃ£o existem em vde_db)
            engine_max_power_kw          REAL,
            engine_rpm_max_power         INTEGER,
            engine_max_torque_nm         REAL,
            engine_rpm_max_torque        INTEGER,

            -- transmissÃ£o (complementos simples; tipo/modelo jÃ¡ estÃ¡ em vde_db)
            gear_count                   INTEGER,
            final_drive_ratio            REAL,

            -- BMS / pack (simples e Ãºteis)
            battery_capacity_kwh         REAL,
            battery_usable_kwh           REAL,
            bms_discharge_limit_kw       REAL,
            bms_regen_limit_kw           REAL,
            bms_note                     TEXT,

            -- ambiente/uso do cenÃ¡rio (podem diferir do VDE base)
            ambient_temp_c               REAL,
            ac_on                        INTEGER,              -- 0/1
            tire_front_psi               REAL,
            tire_rear_psi                REAL,
            scenario_payload_kg          REAL,

            method_note                  TEXT,
            energy_basis                 TEXT,
            engine_method                TEXT,
            engine_version               TEXT,
            source_vde_revision          TEXT,
            assumptions_json             TEXT,
            provenance_json              TEXT,

            -- saÃ­das agregadas (cache)
            energy_Wh_per_km             REAL,
            fuel_km_per_l                REAL,
            fuel_l_per_100km             REAL,
            gco2_per_km                  REAL,

            -- saÃ­das por fase WLTP (normalizadas)
            energy_low_Wh_per_km         REAL,
            energy_mid_Wh_per_km         REAL,
            energy_high_Wh_per_km        REAL,
            energy_xhigh_Wh_per_km       REAL,

            fuel_low_l_per_100km         REAL,
            fuel_mid_l_per_100km         REAL,
            fuel_high_l_per_100km        REAL,
            fuel_xhigh_l_per_100km       REAL,

            gco2_low_per_km              REAL,
            gco2_mid_per_km              REAL,
            gco2_high_per_km             REAL,
            gco2_xhigh_per_km            REAL,

            -- saÃ­das por ciclo EPA (normalizadas)
            energy_ftp75_Wh_per_km       REAL,
            energy_hwfet_Wh_per_km       REAL,
            energy_us06_Wh_per_km        REAL,
            energy_sc03_Wh_per_km        REAL,
            energy_coldftp_Wh_per_km     REAL,

            fuel_ftp75_l_per_100km       REAL,
            fuel_hwfet_l_per_100km       REAL,
            fuel_us06_l_per_100km        REAL,
            fuel_sc03_l_per_100km        REAL,
            fuel_coldftp_l_per_100km     REAL,

            gco2_ftp75_per_km            REAL,
            gco2_hwfet_per_km            REAL,
            gco2_us06_per_km             REAL,
            gco2_sc03_per_km             REAL,
            gco2_coldftp_per_km          REAL,

            -- labeling / off-cycle (genÃ©ricos e fÃ¡ceis de preencher)
            label_program                TEXT,                 -- 'INMETRO/PBEV','EPA','EU-WLTP',...
            label_version_year           INTEGER,              -- ex.: 2025
            label_vehicle_category       TEXT,                 -- ex.: 'Compacto B','SUV C'
            label_cycle_set              TEXT,                 -- '2-cycle','5-cycle','WLTP'
            label_class                  TEXT,                 -- ex.: 'A'..'E' (PBEV) ou 'N/A'
            label_offcycle_method        TEXT,                 -- 'EPA 5-cycle adj','N/A'
            label_offcycle_energy_factor REAL,                 -- fator sobre energia (se houver)
            label_offcycle_fuel_factor   REAL,                 -- fator sobre combustÃ­vel (se houver)
            label_fuel_l_per_100km       REAL,                 -- valor "de etiqueta"
            label_gco2_per_km            REAL,                 -- valor "de etiqueta"
            label_range_km               REAL                  -- alcance (BEV/PHEV), se aplicÃ¡vel
        );
        """)

        cur.execute("CREATE INDEX IF NOT EXISTS idx_fc_vde ON fuelcons_db(vde_id);")

        # ---------------------------------------------------------------------
        # EN: Technical tire roadload library.
        #     Stores tire identity plus SAE/ISO roadload-relevant metadata.
        #
        # PT: Biblioteca tÃ©cnica de pneus para roadload.
        #     Guarda identificaÃ§Ã£o do pneu e metadados SAE/ISO relevantes
        #     ao cÃ¡lculo da contribuiÃ§Ã£o equivalente em A/B/C.
        # ---------------------------------------------------------------------
        cur.execute("""
        CREATE TABLE IF NOT EXISTS tire_roadload_db (
            id                             INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at                     TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at                     TEXT,

            -- identity
            tire_test_code                 TEXT NOT NULL UNIQUE,
            manufacturer                   TEXT NOT NULL,
            model                          TEXT NOT NULL,
            size_code                      TEXT,
            load_index                     TEXT,
            speed_rating                   TEXT,
            is_active                      INTEGER DEFAULT 1,
            notes                          TEXT,

            -- test / source metadata
            standard_family                TEXT NOT NULL,      -- SAE | ISO | CUSTOM
            standard_version               TEXT,
            test_method                    TEXT,
            test_source                    TEXT,
            test_date                      TEXT,
            test_mileage_km                REAL,
            is_tested_value                INTEGER DEFAULT 0,
            is_estimated_value             INTEGER DEFAULT 0,
            break_in_distance_km           REAL,
            is_broken_in                   INTEGER DEFAULT 0,
            test_temperature_c             REAL,
            reference_temperature_c        REAL,
            temperature_correction_applied INTEGER DEFAULT 0,

            -- common RR field
            rr_n_per_kn                    REAL NOT NULL,
            rr_value_source_note           TEXT,

            -- SAE model fields
            sae_a                          REAL,
            sae_b                          REAL,
            sae_c                          REAL,
            sae_alpha                      REAL,
            sae_beta                       REAL,
            pressure_unit                  TEXT,
            load_unit                      TEXT,
            speed_unit                     TEXT,
            force_unit                     TEXT,

            -- ISO profile fields
            iso_rrc_n_per_kn               REAL,
            iso_test_pressure_kpa          REAL,
            iso_test_load_n                REAL,
            iso_test_speed_kph             REAL,
            iso_rolling_resistance_force_n REAL,
            iso_corrected_rrc_n_per_kn     REAL,
            iso_condition_notes            TEXT
        );
        """)

        cur.execute("CREATE INDEX IF NOT EXISTS idx_tire_rl_code ON tire_roadload_db(tire_test_code);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_tire_rl_active ON tire_roadload_db(is_active);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_tire_rl_family ON tire_roadload_db(standard_family);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_tire_rl_make_model ON tire_roadload_db(manufacturer, model, size_code);")

        con.commit()

        con.commit()

        # --- migrations idempotentes (sem engolir erro) ---
        ensure_migrations()



# -----------------------------------------------------------------------------
# INSERT / UPDATE helpers (CS50 style)
# -----------------------------------------------------------------------------

def insert_vde(row: dict) -> int:
    """
    EN: Insert a row into vde_db. 'row' is a dict with column names.
        You can pass only the columns you have; others will be NULL/default.
    PT: Insere uma linha em vde_db. 'row' Ã© um dict com nomes de colunas.
        VocÃª pode passar sÃ³ as colunas que tiver; o resto vira NULL/default.
    """
    ensure_db()
    row = autoresolve_test_mass(row)  # << NEW: autofill mass if needed
    cols = list(row.keys())
    vals = [row[c] for c in cols]
    placeholders = ",".join(["?"] * len(cols))
    with _con() as con:
        cur = con.cursor()
        cur.execute(f"INSERT INTO vde_db ({','.join(cols)}) VALUES ({placeholders})", vals)
        return cur.lastrowid


def update_vde(vde_id: int, updates: dict) -> None:
    """
    EN: Update selected columns of vde_db by id.
        Also touches updated_at automatically (UTC ISO string).
    PT: Atualiza colunas selecionadas de vde_db pelo id.
        TambÃ©m atualiza updated_at automaticamente (UTC em ISO).
    """
    ensure_db()
    # Add/Atualiza carimbo de atualizaÃ§Ã£o
    updates = dict(updates)
    updates["updated_at"] = datetime.utcnow().isoformat()
    updates = autoresolve_test_mass(updates)  # << NEW: autofill mass if needed

    # Build "SET col1=?, col2=?, ..." and value list
    # Monta "SET col1=?, col2=?, ..." e lista de valores
    set_clause = ", ".join([f"{k}=?" for k in updates.keys()])
    vals = list(updates.values()) + [vde_id]

    with _con() as con:
        cur = con.cursor()
        cur.execute(f"UPDATE vde_db SET {set_clause} WHERE id=?", vals)


def insert_fuelcons(row: dict) -> int:
    """
    EN: Insert a row into fuelcons_db. Must include vde_id and electrification.
    PT: Insere uma linha em fuelcons_db. Precisa ter vde_id e electrification.
    """
    ensure_db()
    cols = list(row.keys())
    print(cols)
    vals = [row[c] for c in cols]
    print(vals)
    placeholders = ",".join(["?"] * len(cols))
    with _con() as con:
        cur = con.cursor()
        cur.execute(f"INSERT INTO fuelcons_db ({','.join(cols)}) VALUES ({placeholders})", vals)
        return cur.lastrowid


# -----------------------------------------------------------------------------
# READ helpers (return dicts, like CS50 Row â†’ dict)
# -----------------------------------------------------------------------------

def fetchone(sql: str, params=()):
    """
    EN: Run a SELECT that returns only one row (or None). Returns a dict.
    PT: Executa um SELECT que retorna uma linha (ou None). Retorna dict.
    """
    ensure_db()
    with _con() as con:
        con.row_factory = sqlite3.Row
        row = con.execute(sql, params).fetchone()
    return dict(row) if row else None


def fetchall(sql: str, params=()):
    """
    EN: Run a SELECT that returns many rows. Returns a list of dicts.
    PT: Executa um SELECT que retorna vÃ¡rias linhas. Retorna lista de dicts.
    """
    ensure_db()
    with _con() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


# -----------------------------------------------------------------------------
# (Optional) Triggers and views
# PT/EN: Se quiser automatizar updated_at via trigger ou criar views,
#        dÃ¡ para adicionar no ensure_db() depois, com CREATE TRIGGER/VIEW.
# -----------------------------------------------------------------------------
# --- Utils genÃ©ricos para consultas dinÃ¢micas (cole no db.py) ---
import pandas as _pd

def df_query(sql: str, params=()):
    """SELECT -> DataFrame (Ãºtil no Streamlit)."""
    ensure_db()
    with _con() as con:
        return _pd.read_sql_query(sql, con, params=params)

def table_columns(table: str) -> list[str]:
    """Lista as colunas de uma tabela usando PRAGMA table_info."""
    ensure_db()
    with _con() as con:
        cur = con.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        return [row[1] for row in cur.fetchall()]  # row[1] = name

def select_where(
    table: str,
    columns="*",
    filters: dict[str, tuple[str, object]] | None = None,
    order_by: str | None = None,
    limit: int | None = None,
):
    """
    Consulta genÃ©rica com validaÃ§Ã£o de tabela/colunas.
    - table: 'vde_db' | 'fuelcons_db'
    - columns: '*' ou lista de colunas
    - filters: dict {col: (op, value)}; op âˆˆ {'=','LIKE','>','>=','<','<='}
      Ex.: {'make': ('LIKE','%Fiat%'), 'year': ('=', 2027)}
    - order_by: string (valida coluna)
    - limit: int

    Retorna: DataFrame
    """
    ensure_db()
    allowed_tables = {"vde_db", "fuelcons_db", "tire_roadload_db"}
    if table not in allowed_tables:
        raise ValueError("Tabela nÃ£o permitida.")

    cols = table_columns(table)

    # Normaliza colunas do SELECT
    if columns == "*":
        sel = "*"
    else:
        # validaÃ§Ã£o: mantÃ©m sÃ³ colunas existentes
        safe_cols = [c for c in columns if c in cols]
        if not safe_cols:
            safe_cols = ["*"]
        sel = ", ".join(safe_cols)

    # WHERE dinÃ¢mico com placeholders
    where_clauses, params = [], []
    if filters:
        for col, (op, val) in filters.items():
            if col not in cols:
                continue
            if op not in {"=", "LIKE", ">", ">=", "<", "<="}:
                continue
            where_clauses.append(f"{col} {op} ?")
            params.append(val)
    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    # ORDER BY validado
    order_sql = ""
    if order_by:
        # permite algo como "year DESC" -> separa nome e direÃ§Ã£o
        parts = order_by.split()
        colname = parts[0]
        direction = parts[1].upper() if len(parts) > 1 else ""
        if colname in cols and direction in {"", "ASC", "DESC"}:
            order_sql = "ORDER BY " + colname + (" " + direction if direction else "")

    limit_sql = f"LIMIT {int(limit)}" if isinstance(limit, int) and limit > 0 else ""

    sql = f"SELECT {sel} FROM {table} {where_sql} {order_sql} {limit_sql};"

    return df_query(sql, params)

def delete_row(table: str, row_id: int) -> int:
    """
    Delete one row from a given table by id.
    - table: 'vde_db' or 'fuelcons_db'
    - row_id: integer id
    """
    ensure_db()
    if table not in {"vde_db", "fuelcons_db", "tire_roadload_db"}:
        raise ValueError("Table not allowed.")
    rid = int(row_id)
    with _con() as con:
        cur = con.execute(f"DELETE FROM {table} WHERE id=?", (rid,))
        return int(cur.rowcount or 0)

def update_row(table: str, row_id: int, updates: dict) -> None:
    """
    Atualiza colunas de uma linha em 'table' pelo id.
    Exemplo: update_row("vde_db", 5, {"make": "Fiat", "year": 2025})
    """
    ensure_db()
    if table not in {"vde_db", "fuelcons_db", "tire_roadload_db"}:
        raise ValueError("Tabela nÃ£o permitida.")
    set_clause = ", ".join([f"{k}=?" for k in updates.keys()])
    vals = list(updates.values()) + [row_id]
    with _con() as con:
        con.execute(f"UPDATE {table} SET {set_clause} WHERE id=?", vals)

# --- Dangerous helpers: truncate or delete the DB file -----------------------
import os

# --- Dangerous helpers (explicit db_path): truncate or delete the DB file ----
import os
from typing import Union

PathLike = Union[str, os.PathLike]

def truncate_db(db_path: PathLike) -> None:
    """
    Apaga TODAS as linhas das tabelas (mantÃ©m o arquivo .db), zera AUTOINCREMENT
    e executa VACUUM. Requer que o schema jÃ¡ exista.
    """
    db_path = Path(db_path)
    with sqlite3.connect(str(db_path), timeout=30) as con:
        cur = con.cursor()
        cur.execute("PRAGMA foreign_keys=ON;")
        cur.execute("PRAGMA busy_timeout=5000;")
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("BEGIN IMMEDIATE;")  # toma lock de escrita

        # limpa tabelas (cascata apaga fuelcons ligados)
        cur.execute("DELETE FROM fuelcons_db;")
        cur.execute("DELETE FROM vde_db;")
        # zera autoincrement
        cur.execute("DELETE FROM sqlite_sequence WHERE name IN ('vde_db','fuelcons_db');")

        con.commit()
        cur.execute("VACUUM;")  # compacta arquivo
    print(f"âœ”ï¸ DB truncado: {db_path}")

def delete_db_file(db_path: PathLike) -> None:
    """
    Deleta o arquivo do banco e arquivos auxiliares (-wal/-shm).
    TODAS as conexÃµes devem estar fechadas.
    """
    db_path = Path(db_path)
    for ext in ("", "-wal", "-shm"):
        p = Path(f"{db_path}{ext}")
        if p.exists():
            try:
                os.remove(p)
                print(f"âœ”ï¸ Removido: {p}")
            except PermissionError:
                raise RuntimeError(f"Arquivo em uso/bloqueado: {p}. Feche processos e tente novamente.")
