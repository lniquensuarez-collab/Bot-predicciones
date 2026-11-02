"""Ingesta de datos -> SQLite. Única fuente: football-data.co.uk (sin APIs, sin claves).
  · CSV por temporada: Premier, Championship, Bundesliga, La Liga, Serie A  (/mmz4281/{temporada}/{liga}.csv)
  · XLSX con todas las temporadas: Brasil, Argentina, Japón, China, EE. UU.  (/new/{liga}.xlsx)
"""
import difflib
import io
import unicodedata

import numpy as np
import pandas as pd
import requests

from . import db
from .config import (
    ESPN_SLUGS,
    LIGAS,
    URL_CSV,
    URL_ESPN_SCOREBOARD,
    URL_FIXTURES_CSV,
    URL_FIXTURES_NEW,
    URL_XLSX,
    temporadas_recientes,
)

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

    cols = {c: col(c).to_numpy() for c in ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"]}
    fechas = d["Date"].dt.strftime("%Y-%m-%d").tolist()
    homes = d["HomeTeam"].tolist()
    aways = d["AwayTeam"].tolist()
    fthg = d["FTHG"].astype(int).tolist()
    ftag = d["FTAG"].astype(int).tolist()
    ftr = d["FTR"].tolist()
    seasons = d["Season"].tolist() if "Season" in d.columns else [None] * len(d)
    refs = d["Referee"].tolist() if "Referee" in d.columns else [None] * len(d)
    cl, ce, cv = cuotas["l"].to_numpy(), cuotas["e"].to_numpy(), cuotas["v"].to_numpy()

    def _txt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        s = str(v).strip()
        return s if s and s.lower() != "nan" else None

    filas = []
    for i in range(len(d)):
        f, h, a = fechas[i], homes[i], aways[i]
        filas.append(dict(
            id=f"fd:{codigo}:{f}:{h}:{a}", tipo="club", competicion=codigo,
            temporada=temporada or _season(seasons[i], f[:4]), fecha=f,
            local=h, visita=a, goles_l=fthg[i], goles_v=ftag[i],
            resultado=ftr[i],
            tiros_l=_num(cols["HS"][i]), tiros_v=_num(cols["AS"][i]),
            tiros_p_l=_num(cols["HST"][i]), tiros_p_v=_num(cols["AST"][i]),
            corners_l=_num(cols["HC"][i]), corners_v=_num(cols["AC"][i]),
            faltas_l=_num(cols["HF"][i]), faltas_v=_num(cols["AF"][i]),
            amarillas_l=_num(cols["HY"][i]), amarillas_v=_num(cols["AY"][i]),
            rojas_l=_num(cols["HR"][i]), rojas_v=_num(cols["AR"][i]),
            cuota_l=_num(cl[i]), cuota_e=_num(ce[i]), cuota_v=_num(cv[i]),
            arbitro=_txt(refs[i])))
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
    """Obtiene todos los equipos (locales y visitantes) de la temporada más reciente por fecha."""
    query = """
        WITH ult_temp AS (
            SELECT temporada FROM partidos 
            WHERE competicion = ? 
            ORDER BY fecha DESC LIMIT 1
        )
        SELECT DISTINCT local AS equipo FROM partidos 
        WHERE competicion = ? AND temporada = (SELECT temporada FROM ult_temp)
        UNION
        SELECT DISTINCT visita AS equipo FROM partidos 
        WHERE competicion = ? AND temporada = (SELECT temporada FROM ult_temp)
    """
    df = pd.read_sql_query(query, con, params=(competicion, competicion, competicion))
    return sorted(df["equipo"].tolist())


ALIAS_EQUIPOS = {
    "manchester united": "Man United",
    "manchester city": "Man City",
    "newcastle united": "Newcastle",
    "nottingham forest": "Nott'm Forest",
    "tottenham hotspur": "Tottenham",
    "wolverhampton wanderers": "Wolves",
    "west ham united": "West Ham",
    "brighton hove albion": "Brighton",
    "leicester city": "Leicester",
    "ipswich town": "Ipswich",
    "leeds united": "Leeds",
    "sheffield united": "Sheffield United",
    "sheffield wednesday": "Sheffield Weds",
    "west bromwich albion": "West Brom",
    "queens park rangers": "QPR",
    "blackburn rovers": "Blackburn",
    "atletico madrid": "Ath Madrid",
    "athletic club": "Ath Bilbao",
    "athletic bilbao": "Ath Bilbao",
    "real betis": "Betis",
    "celta vigo": "Celta",
    "rcd espanyol": "Espanol",
    "deportivo alaves": "Alaves",
    "rayo vallecano": "Vallecano",
    "real sociedad": "Sociedad",
    "real valladolid": "Valladolid",
    "bayern munich": "Bayern Munich",
    "bayer leverkusen": "Leverkusen",
    "borussia dortmund": "Dortmund",
    "borussia monchengladbach": "M'gladbach",
    "eintracht frankfurt": "Ein Frankfurt",
    "rb leipzig": "RB Leipzig",
    "vfb stuttgart": "Stuttgart",
    "vfl wolfsburg": "Wolfsburg",
    "fsv mainz 05": "Mainz",
    "fc augsburg": "Augsburg",
    "sc freiburg": "Freiburg",
    "tsg hoffenheim": "Hoffenheim",
    "fc heidenheim": "Heidenheim",
    "st pauli": "St Pauli",
    "holstein kiel": "Holstein Kiel",
    "internazionale": "Inter",
    "inter milan": "Inter",
    "ac milan": "Milan",
    "hellas verona": "Verona",
    "parma calcio 1913": "Parma",
    "como 1907": "Como",
    "paris saint germain": "Paris SG",
    "olympique marseille": "Marseille",
    "olympique lyonnais": "Lyon",
    "as monaco": "Monaco",
    "stade rennais": "Rennes",
    "stade brestois": "Brest",
    "rc lens": "Lens",
    "losc lille": "Lille",
    "saint etienne": "St Etienne",
    "sporting cp": "Sp Lisbon",
    "sporting braga": "Sp Braga",
    "vitoria guimaraes": "Guimaraes",
}


def _norm_nombre(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii").lower()
    for tok in (" fc", " cf", " sc", " ac", "fk ", "fc ", "cd ", "ud ", "rcd ", "1. ", "sv "):
        s = s.replace(tok, " ")
    return " ".join(
        "".join(ch if ch.isalnum() else " " for ch in s).split()
    )


def _emparejar_equipo(nombre_ext, equipos_db):
    """Empareja el nombre de un equipo externo (fixtures/ESPN) con el nombre exacto en nuestra BD."""
    if not nombre_ext or not equipos_db:
        return None
    if nombre_ext in equipos_db:
        return nombre_ext
    n_ext = _norm_nombre(nombre_ext)
    if n_ext in ALIAS_EQUIPOS and ALIAS_EQUIPOS[n_ext] in equipos_db:
        return ALIAS_EQUIPOS[n_ext]
    mapa_norm = {_norm_nombre(e): e for e in equipos_db}
    if n_ext in mapa_norm:
        return mapa_norm[n_ext]
    # Subcadena o token principal
    for k_norm, orig in mapa_norm.items():
        if k_norm in n_ext or n_ext in k_norm:
            return orig
    cercanos = difflib.get_close_matches(n_ext, list(mapa_norm.keys()), n=1, cutoff=0.58)
    return mapa_norm[cercanos[0]] if cercanos else None


def buscar_partidos_del_dia(con, fecha_obj, nombres_ligas=None, ventana_dias=0):
    """Busca los partidos programados en la fecha indicada (y hasta +ventana_dias) para las ligas de la app.
    Combina: 1) football-data.co.uk fixtures, 2) calendario diario público de ESPN y 3) base local SQLite."""
    nombres_ligas = nombres_ligas or list(LIGAS)
    cod_a_nombre = {LIGAS[n]["codigo"]: n for n in nombres_ligas if n in LIGAS}
    codigos = set(cod_a_nombre.keys())
    equipos_por_liga = {c: obtener_equipos_vigentes(con, c) for c in codigos}

    f_ini = pd.Timestamp(fecha_obj).normalize()
    f_fin = f_ini + pd.Timedelta(days=int(max(0, ventana_dias)))
    encontrados, vistos = [], set()

    def _registrar(cod, f_str, h_raw, a_raw, cl=None, ce=None, cv=None, fuente="web"):
        if cod not in codigos:
            return
        eqs = equipos_por_liga.get(cod, [])
        h_db = _emparejar_equipo(h_raw, eqs)
        a_db = _emparejar_equipo(a_raw, eqs)
        if not h_db or not a_db or h_db == a_db:
            return
        clave = (cod, f_str, h_db, a_db)
        if clave in vistos:
            return
        vistos.add(clave)
        encontrados.append({
            "liga": cod_a_nombre[cod],
            "codigo": cod,
            "fecha": f_str,
            "local": h_db,
            "visita": a_db,
            "cuota_l": _num(cl),
            "cuota_e": _num(ce),
            "cuota_v": _num(cv),
            "fuente": fuente,
        })

    # 1. Consultar fixtures oficiales de football-data.co.uk (CSV europeo y ligas extra)
    for url_fx, col_div in ((URL_FIXTURES_CSV, "Div"), (URL_FIXTURES_NEW, "League")):
        try:
            raw = _descargar(url_fx)
            df_fx = pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip")
            df_fx = df_fx.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam"})
            if "Date" not in df_fx.columns or "HomeTeam" not in df_fx.columns:
                continue
            df_fx["Date"] = _fecha(df_fx["Date"])
            df_fx = df_fx.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
            sub = df_fx[(df_fx["Date"] >= f_ini) & (df_fx["Date"] <= f_fin)]
            for _, r in sub.iterrows():
                cod = str(r.get("Div") or r.get("League") or "").strip()
                # En new_league_fixtures Country/League puede venir como Brazil/BRA
                if cod not in codigos and "Country" in r:
                    mapa_pais = {"Brazil": "BRA", "Argentina": "ARG", "Japan": "JPN", "China": "CHN", "USA": "USA"}
                    cod = mapa_pais.get(str(r.get("Country")).strip(), cod)
                cl = r.get("B365H") or r.get("AvgH") or r.get("PSCH")
                ce = r.get("B365D") or r.get("AvgD") or r.get("PSCD")
                cv = r.get("B365A") or r.get("AvgA") or r.get("PSCA")
                _registrar(cod, r["Date"].strftime("%Y-%m-%d"), r["HomeTeam"], r["AwayTeam"], cl, ce, cv, "football-data")
        except Exception:
            pass

    # 2. Consultar calendario diario público de ESPN para cubrir cualquier fecha del calendario
    fechas_rango = pd.date_range(f_ini, f_fin, freq="D")
    for f_dia in fechas_rango:
        ymd = f_dia.strftime("%Y%m%d")
        f_iso = f_dia.strftime("%Y-%m-%d")
        for cod in codigos:
            slug = ESPN_SLUGS.get(cod)
            if not slug:
                continue
            try:
                r = requests.get(URL_ESPN_SCOREBOARD.format(slug=slug, ymd=ymd), headers=HEADERS, timeout=10)
                if r.status_code != 200:
                    continue
                eventos = r.json().get("events", [])
                for ev in eventos:
                    comps = ev.get("competitions", [{}])[0].get("competitors", [])
                    h_name = next((c.get("team", {}).get("name") for c in comps if c.get("homeAway") == "home"), None)
                    a_name = next((c.get("team", {}).get("name") for c in comps if c.get("homeAway") == "away"), None)
                    if h_name and a_name:
                        _registrar(cod, f_iso, h_name, a_name, fuente="calendario-vivo")
            except Exception:
                pass

    # 3. Respaldo en base de datos local SQLite (útil sin red o al consultar una fecha ya registrada)
    placeholders = ",".join("?" * len(codigos))
    q_local = f"""
        SELECT competicion, fecha, local, visita, cuota_l, cuota_e, cuota_v
        FROM partidos
        WHERE competicion IN ({placeholders}) AND fecha >= ? AND fecha <= ?
        ORDER BY fecha, id
    """
    df_loc = pd.read_sql_query(
        q_local, con, params=[*codigos, f_ini.strftime("%Y-%m-%d"), f_fin.strftime("%Y-%m-%d")]
    )
    for _, r in df_loc.iterrows():
        _registrar(r["competicion"], r["fecha"], r["local"], r["visita"],
                   r["cuota_l"], r["cuota_e"], r["cuota_v"], "base-local")

    return encontrados