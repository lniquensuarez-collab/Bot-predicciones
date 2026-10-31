"""Ingesta de datos -> SQLite. Única fuente: football-data.co.uk (sin APIs, sin claves).
  · CSV por temporada: Premier, Championship, Bundesliga, La Liga, Serie A  (/mmz4281/{temporada}/{liga}.csv)
  · XLSX con todas las temporadas: Brasil, Argentina, Japón, China, EE. UU.  (/new/{liga}.xlsx)
"""
import io

import numpy as np
import pandas as pd
import requests

from . import db
from .config import LIGAS, URL_CSV, URL_XLSX, temporadas_recientes

HEADERS = {"User-Agent": "Mozilla/5.0 (futbol_ai)"}
OBLIGATORIAS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
ODDS = [("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA"), ("B365CH", "B365CD", "B365CA"),
        ("PH", "PD", "PA"), ("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A")]


def _num(x):
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _descargar(url):
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.content


def _fecha(serie):
    if pd.api.types.is_datetime64_any_dtype(serie):
        return serie
    try:
        return pd.to_datetime(serie, dayfirst=True, format="mixed", errors="coerce")
    except TypeError:  # pandas antiguo
        return pd.to_datetime(serie, dayfirst=True, errors="coerce")


def _estandarizar(d):
    """Unifica los nombres de columnas de los CSV y los XLSX. Devuelve None si faltan columnas clave."""
    d = d.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam", "HG": "FTHG", "AG": "FTAG", "Res": "FTR"})
    if any(c not in d.columns for c in OBLIGATORIAS):
        return None
    d = d.copy()
    d["Date"] = _fecha(d["Date"])
    d["FTHG"] = pd.to_numeric(d["FTHG"], errors="coerce")
    d["FTAG"] = pd.to_numeric(d["FTAG"], errors="coerce")
    d = d.dropna(subset=OBLIGATORIAS)
    if "FTR" not in d.columns:
        d["FTR"] = np.where(d["FTHG"] > d["FTAG"], "H", np.where(d["FTHG"] == d["FTAG"], "D", "A"))
    return d[d["FTR"].isin(["H", "D", "A"])]


def _season(v, fallback):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return fallback
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v)


def _filas(d, codigo, temporada=None):
    d = _estandarizar(d)
    if d is None or d.empty:
        return []

    def col(c):
        return pd.to_numeric(d[c], errors="coerce") if c in d.columns else pd.Series(np.nan, index=d.index)

    cuotas = pd.DataFrame(np.nan, index=d.index, columns=["l", "e", "v"])
    for cs in ODDS:                       # prioriza cierre de Pinnacle; rellena huecos con las siguientes
        if all(c in d.columns for c in cs):
            cand = pd.DataFrame({"l": col(cs[0]), "e": col(cs[1]), "v": col(cs[2])})
            vacio = cuotas.isna().any(axis=1)
            cuotas.loc[vacio] = cand.loc[vacio]

    cols = {c: col(c) for c in ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"]}
    filas = []
    for i, r in d.iterrows():
        f = r["Date"].strftime("%Y-%m-%d")
        filas.append(dict(
            id=f"fd:{codigo}:{f}:{r['HomeTeam']}:{r['AwayTeam']}", tipo="club", competicion=codigo,
            temporada=temporada or _season(r.get("Season"), f[:4]), fecha=f,
            local=r["HomeTeam"], visita=r["AwayTeam"], goles_l=int(r["FTHG"]), goles_v=int(r["FTAG"]),
            resultado=r["FTR"],
            tiros_l=_num(cols["HS"][i]), tiros_v=_num(cols["AS"][i]),
            tiros_p_l=_num(cols["HST"][i]), tiros_p_v=_num(cols["AST"][i]),
            corners_l=_num(cols["HC"][i]), corners_v=_num(cols["AC"][i]),
            faltas_l=_num(cols["HF"][i]), faltas_v=_num(cols["AF"][i]),
            amarillas_l=_num(cols["HY"][i]), amarillas_v=_num(cols["AY"][i]),
            rojas_l=_num(cols["HR"][i]), rojas_v=_num(cols["AR"][i]),
            cuota_l=_num(cuotas.at[i, "l"]), cuota_e=_num(cuotas.at[i, "e"]), cuota_v=_num(cuotas.at[i, "v"])))
    return filas


def filas_xlsx(d, codigo, anios):
    """Archivo con todas las temporadas: conserva solo los últimos `anios` años."""
    d = _estandarizar(d)
    if d is None or d.empty:
        return []
    return _filas(d[d["Date"] >= d["Date"].max() - pd.DateOffset(years=anios)], codigo)


def ingestar_ligas(con, nombres, extra_temporadas=0, anios_xlsx=3, progreso=None):
    """Descarga y guarda las ligas indicadas. extra_temporadas añade temporadas europeas anteriores
    (por defecto solo las 2 últimas, p. ej. 2526 y 2627). Devuelve (filas, avisos)."""
    tareas = []
    for nombre in nombres:
        cfg = LIGAS[nombre]
        if cfg["fuente"] == "csv":
            tareas += [(nombre, cfg, t) for t in temporadas_recientes(2 + extra_temporadas)]
        else:
            tareas.append((nombre, cfg, None))

    filas, avisos = [], []
    for k, (nombre, cfg, t) in enumerate(tareas):
        try:
            if t:
                d = pd.read_csv(io.BytesIO(_descargar(URL_CSV.format(t=t, c=cfg["codigo"]))),
                                encoding="latin-1", on_bad_lines="skip")
                nuevas = _filas(d, cfg["codigo"], t)
            else:
                d = pd.read_excel(io.BytesIO(_descargar(URL_XLSX.format(c=cfg["codigo"]))))
                nuevas = filas_xlsx(d, cfg["codigo"], anios_xlsx)
            if not nuevas:
                avisos.append(f"{nombre} {t or ''}: el archivo no tiene el formato esperado.")
            filas += nuevas
        except ImportError as e:
            avisos.append(f"{nombre}: falta una librería ({e}). Ejecuta: pip install openpyxl")
        except Exception as e:  # se informa, no se esconde
            avisos.append(f"{nombre} {t or ''}: {type(e).__name__} - {e}")
        if progreso:
            progreso((k + 1) / len(tareas), f"{nombre} {t or ''}")
    n = db.guardar_partidos(con, filas)
    db.registrar_ingesta(con, "football-data.co.uk", ", ".join(cfg_n for cfg_n in nombres), n)
    return n, avisos

def obtener_equipos_vigentes(con, competicion):
    """Obtiene la lista de equipos activos en la temporada más reciente para la interfaz."""
    query = """
        SELECT DISTINCT local AS equipo 
        FROM partidos 
        WHERE competicion = ? AND temporada = (
            SELECT MAX(temporada) FROM partidos WHERE competicion = ?
        )
    """
    df = pd.read_sql_query(query, con, params=(competicion, competicion))
    return sorted(df['equipo'].tolist())
