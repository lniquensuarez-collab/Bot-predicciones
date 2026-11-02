"""Capa de persistencia SQLite (data/futbol.db)."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS partidos (
    id TEXT PRIMARY KEY, tipo TEXT NOT NULL, competicion TEXT, temporada TEXT,
    fecha TEXT NOT NULL, local TEXT NOT NULL, visita TEXT NOT NULL,
    goles_l INTEGER, goles_v INTEGER, resultado TEXT,
    tiros_l REAL, tiros_v REAL, tiros_p_l REAL, tiros_p_v REAL, corners_l REAL, corners_v REAL,
    cuota_l REAL, cuota_e REAL, cuota_v REAL,
    faltas_l REAL, faltas_v REAL, amarillas_l REAL, amarillas_v REAL, rojas_l REAL, rojas_v REAL
);
CREATE INDEX IF NOT EXISTS idx_partidos_comp_fecha ON partidos(competicion, fecha);
CREATE TABLE IF NOT EXISTS elo_actual (
    equipo TEXT, competicion TEXT, rating REAL, partidos INTEGER, actualizado TEXT,
    PRIMARY KEY (equipo, competicion)
);
CREATE TABLE IF NOT EXISTS ingestas (
    id INTEGER PRIMARY KEY AUTOINCREMENT, fuente TEXT, detalle TEXT, filas INTEGER, fecha TEXT
);
"""

COLS = ["id", "tipo", "competicion", "temporada", "fecha", "local", "visita", "goles_l", "goles_v",
        "resultado", "tiros_l", "tiros_v", "tiros_p_l", "tiros_p_v", "corners_l", "corners_v",
        "cuota_l", "cuota_e", "cuota_v", "faltas_l", "faltas_v", "amarillas_l", "amarillas_v",
        "rojas_l", "rojas_v"]

MAPA = {"fecha": "Date", "local": "HomeTeam", "visita": "AwayTeam", "goles_l": "FTHG", "goles_v": "FTAG",
        "resultado": "FTR", "tiros_l": "HS", "tiros_v": "AS", "tiros_p_l": "HST", "tiros_p_v": "AST",
        "corners_l": "HC", "corners_v": "AC", "faltas_l": "HF", "faltas_v": "AF", "amarillas_l": "HY",
        "amarillas_v": "AY", "rojas_l": "HR", "rojas_v": "AR", "temporada": "Season", "competicion": "Competicion"}


def conectar(path=None):
    path = path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.executescript(SCHEMA)
    _migrar(con)
    return con


def _migrar(con):
    """Bases creadas con versiones anteriores: añade las columnas que falten (sin perder datos)."""
    existentes = {r[1] for r in con.execute("PRAGMA table_info(partidos)")}
    for c in COLS:
        if c not in existentes:
            con.execute(f"ALTER TABLE partidos ADD COLUMN {c} REAL")
    con.commit()


@contextmanager
def sesion(path=None):
    con = conectar(path)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def _py(v):
    if v is None:
        return None
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def guardar_partidos(con, filas):
    """Upsert por id. Los datos ya guardados no se pisan con valores vacíos."""
    registros = []
    for f in filas:
        f = dict(f)
        gl, gv = f.get("goles_l"), f.get("goles_v")
        if f.get("resultado") is None and gl is not None and gv is not None:
            f["resultado"] = "H" if gl > gv else ("D" if gl == gv else "A")
        registros.append(tuple(_py(f.get(c)) for c in COLS))
    if not registros:
        return 0
    sets = ",".join(f"{c}=COALESCE(excluded.{c}, partidos.{c})" for c in COLS if c != "id")
    con.executemany(
        f"INSERT INTO partidos ({','.join(COLS)}) VALUES ({','.join('?' * len(COLS))}) "
        f"ON CONFLICT(id) DO UPDATE SET {sets}", registros)
    con.commit()
    return len(registros)


def registrar_ingesta(con, fuente, detalle, filas):
    con.execute("INSERT INTO ingestas (fuente, detalle, filas, fecha) VALUES (?,?,?,?)",
                (fuente, detalle, int(filas), datetime.now().isoformat(timespec="seconds")))
    con.commit()


def desmargenar_cuotas(cuotas):
    """Elimina el margen de la casa de apuestas (overround) mediante el método exponencial (Power Method),
    que corrige el sesgo favorito-longshot mejor que la normalización proporcional."""
    c = np.asarray(cuotas, dtype=float)
    unidim = c.ndim == 1
    if unidim:
        c = c[np.newaxis, :]
    out = np.full_like(c, np.nan, dtype=float)
    ok = np.all(np.isfinite(c) & (c > 1.0), axis=1)
    if np.any(ok):
        q = 1.0 / c[ok]
        log_q = np.log(q)
        k = np.ones((q.shape[0], 1), dtype=float)
        for _ in range(5):
            qk = q ** k
            f = qk.sum(axis=1, keepdims=True) - 1.0
            df = (qk * log_q).sum(axis=1, keepdims=True)
            k = np.clip(k - f / np.where(np.abs(df) > 1e-12, df, -1.0), 0.1, 5.0)
        p = q ** k
        out[ok] = p / p.sum(axis=1, keepdims=True)
    return out[0] if unidim else out


def cargar_partidos(competicion, con=None):
    """DataFrame con nombres estándar del motor (HomeTeam, FTHG, ...) de UNA liga, en orden cronológico."""
    propio = con is None
    con = con or conectar()
    df = pd.read_sql_query("SELECT * FROM partidos WHERE competicion=? AND goles_l IS NOT NULL "
                           "AND goles_v IS NOT NULL ORDER BY fecha, id", con, params=[competicion])
    if propio:
        con.close()
    if df.empty:
        return df
    cuotas = df[["cuota_l", "cuota_e", "cuota_v"]].apply(pd.to_numeric, errors="coerce").to_numpy()
    prob = desmargenar_cuotas(cuotas)
    df = df.rename(columns=MAPA)
    df[["mH", "mD", "mA"]] = prob
    df["Date"] = pd.to_datetime(df["Date"])
    for c in ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def guardar_elo(con, competicion, ratings, partidos):
    ahora = datetime.now().isoformat(timespec="seconds")
    con.executemany(
        "INSERT OR REPLACE INTO elo_actual (equipo, competicion, rating, partidos, actualizado) VALUES (?,?,?,?,?)",
        [(t, competicion, float(r), int(partidos.get(t, 0)), ahora) for t, r in ratings.items()])
    con.commit()


def total_partidos(con):
    return con.execute("SELECT COUNT(*) FROM partidos").fetchone()[0]


def resumen_por_liga(con):
    return pd.read_sql_query(
        "SELECT competicion AS Liga, COUNT(*) AS Partidos, MIN(fecha) AS Desde, MAX(fecha) AS Hasta, "
        "COUNT(tiros_p_l) AS 'Con tiros a puerta', COUNT(faltas_l) AS 'Con faltas', COUNT(amarillas_l) AS 'Con tarjetas', COUNT(cuota_l) AS 'Con cuotas' "
        "FROM partidos GROUP BY competicion ORDER BY competicion", con)
