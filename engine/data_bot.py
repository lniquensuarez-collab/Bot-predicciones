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
    COMPETENCIAS_FIFA,
    EQUIPOS_INVITADOS,
    ESPN_SLUGS,
    LIGAS,
    RANKING_FIFA_PUNTOS,
    TORNEOS_INTERNACIONALES,
    UNDERSTAT_SLUGS,
    URL_CSV,
    URL_ESPN_SCOREBOARD,
    URL_FIFA_RESULTS,
    URL_FIXTURES_CSV,
    URL_FIXTURES_NEW,
    URL_UNDERSTAT,
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

    cols = {c: col(c).to_numpy() for c in ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR", "xGH", "xGA"]}
    fechas = d["Date"].dt.strftime("%Y-%m-%d").tolist()
    homes = d["HomeTeam"].tolist()
    aways = d["AwayTeam"].tolist()
    fthg = d["FTHG"].astype(int).tolist()
    ftag = d["FTAG"].astype(int).tolist()
    ftr = d["FTR"].tolist()
    seasons = d["Season"].tolist() if "Season" in d.columns else [None] * len(d)
    refs = d["Referee"].tolist() if "Referee" in d.columns else [None] * len(d)
    cl, ce, cv = cuotas["l"].to_numpy(), cuotas["e"].to_numpy(), cuotas["v"].to_numpy()

    # xG por partido: usa columna xGH/xGA si existe; si no y hay HST/AST, calcula xG de volumen de tiros calibrado
    xgh_arr, xga_arr = cols["xGH"].copy(), cols["xGA"].copy()
    mask_sot = np.isnan(xgh_arr) & ~np.isnan(cols["HST"]) & ~np.isnan(cols["AST"])
    if np.any(mask_sot):
        hst_v, ast_v = cols["HST"], cols["AST"]
        hs_r = np.maximum(np.where(np.isnan(cols["HS"]), hst_v * 2.6, cols["HS"]) - hst_v, 0.0)
        as_r = np.maximum(np.where(np.isnan(cols["AS"]), ast_v * 2.6, cols["AS"]) - ast_v, 0.0)
        hc_v = np.where(np.isnan(cols["HC"]), 5.0, cols["HC"])
        ac_v = np.where(np.isnan(cols["AC"]), 4.5, cols["AC"])
        raw_h = 0.28 * hst_v + 0.04 * hs_r + 0.015 * hc_v
        raw_a = 0.28 * ast_v + 0.04 * as_r + 0.015 * ac_v
        esc_h = float(np.mean(fthg) / max(float(np.nanmean(raw_h)), 1e-6))
        esc_a = float(np.mean(ftag) / max(float(np.nanmean(raw_a)), 1e-6))
        xgh_arr[mask_sot] = np.round(raw_h[mask_sot] * esc_h, 3)
        xga_arr[mask_sot] = np.round(raw_a[mask_sot] * esc_a, 3)

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
            arbitro=_txt(refs[i]),
            xg_l=_num(xgh_arr[i]), xg_v=_num(xga_arr[i])))
    return filas


def filas_xlsx(d, codigo, anios):
    """Archivo con todas las temporadas: conserva solo los últimos `anios` años."""
    d = _estandarizar(d)
    if d is None or d.empty:
        return []
    return _filas(d[d["Date"] >= d["Date"].max() - pd.DateOffset(years=anios)], codigo)


def _enriquecer_xg_understat(filas, codigo, temporada):
    """Si la liga está en Understat (E0, SP1, D1, I1, F1), intenta descargar el xG oficial tiro-a-tiro
    para reemplazar el xG estimado por volumen de tiros. Si no hay red o falla, conserva el xG estimado."""
    slug = UNDERSTAT_SLUGS.get(codigo)
    if not slug or not filas or not temporada or len(str(temporada)) != 4:
        return
    # Evita llamadas externas cuando _descargar está simulado en tests offline
    if getattr(_descargar, "__name__", "") != "_descargar":
        return
    try:
        anio = 2000 + int(str(temporada)[:2])
        hdrs = {**HEADERS, "X-Requested-With": "XMLHttpRequest"}
        r = requests.get(URL_UNDERSTAT.format(slug=slug, anio=anio), headers=hdrs, timeout=12)
        if r.status_code != 200:
            return
        datos = r.json().get("dates", [])
        if not datos:
            return
        equipos_set = sorted({f["local"] for f in filas} | {f["visita"] for f in filas})
        mapa_xg = {}
        for m in datos:
            if not m.get("isResult"):
                continue
            xg_h = _num((m.get("xG") or {}).get("h"))
            xg_a = _num((m.get("xG") or {}).get("a"))
            if xg_h is None or xg_a is None:
                continue
            f_str = str(m.get("datetime", ""))[:10]
            h_db = _emparejar_equipo((m.get("h") or {}).get("title"), equipos_set)
            a_db = _emparejar_equipo((m.get("a") or {}).get("title"), equipos_set)
            if h_db and a_db:
                mapa_xg[(f_str, h_db, a_db)] = (round(xg_h, 3), round(xg_a, 3))
                mapa_xg[(h_db, a_db)] = (round(xg_h, 3), round(xg_a, 3))
        for f in filas:
            par = mapa_xg.get((f["fecha"], f["local"], f["visita"])) or mapa_xg.get((f["local"], f["visita"]))
            if par:
                f["xg_l"], f["xg_v"] = par
    except Exception:
        pass


def _extraer_score(s):
    if isinstance(s, dict):
        v = s.get("value")
        return _num(v if v is not None else s.get("displayValue"))
    return _num(s)


def _filas_espn_eventos(eventos, codigo):
    """Convierte eventos JSON de ESPN (/scoreboard o /teams/{id}/schedule) al esquema SQLite,
    extrayendo goles, tiros a puerta, xG calibrado, córners, faltas y tarjetas amarillas/rojas."""
    out = []
    for ev in eventos or []:
        comp = (ev.get("competitions") or [{}])[0]
        st_type = (comp.get("status") or {}).get("type") or {}
        if not (st_type.get("completed") is True or st_type.get("state") == "post"):
            continue
        comps = comp.get("competitors") or []
        c_h = next((c for c in comps if c.get("homeAway") == "home"), None)
        c_a = next((c for c in comps if c.get("homeAway") == "away"), None)
        if not c_h or not c_a:
            continue
        th = c_h.get("team") or {}
        ta = c_a.get("team") or {}
        h_raw = th.get("name") or th.get("displayName") or th.get("location") or th.get("shortDisplayName")
        a_raw = ta.get("name") or ta.get("displayName") or ta.get("location") or ta.get("shortDisplayName")
        gl = _extraer_score(c_h.get("score"))
        gv = _extraer_score(c_a.get("score"))
        if not h_raw or not a_raw or gl is None or gv is None:
            continue
        h = ALIAS_EQUIPOS.get(_norm_nombre(h_raw), str(h_raw).strip())
        a = ALIAS_EQUIPOS.get(_norm_nombre(a_raw), str(a_raw).strip())
        if h == a:
            continue
        f_raw = str(comp.get("date") or ev.get("date") or "")[:10]
        if len(f_raw) < 10:
            continue
        temporada = f_raw[:4]

        def _stat(comp_side, *nombres_stat):
            for s in comp_side.get("statistics") or []:
                if s.get("name") in nombres_stat:
                    v = _num(s.get("displayValue") or s.get("value"))
                    if v is not None:
                        return v
            return None

        sot_l = _stat(c_h, "shotsOnTarget", "shotsOnGoal")
        sot_v = _stat(c_a, "shotsOnTarget", "shotsOnGoal")
        sh_l = _stat(c_h, "totalShots", "shots")
        sh_v = _stat(c_a, "totalShots", "shots")
        xg_l, xg_v = None, None
        if sot_l is not None and sot_v is not None:
            fuera_l = max(0.0, (sh_l - sot_l)) if sh_l is not None else sot_l * 1.4
            fuera_v = max(0.0, (sh_v - sot_v)) if sh_v is not None else sot_v * 1.4
            xg_l = round(0.28 * sot_l + 0.04 * fuera_l + 0.12 * int(gl), 3)
            xg_v = round(0.28 * sot_v + 0.04 * fuera_v + 0.12 * int(gv), 3)

        out.append({
            "id": f"{codigo}:{f_raw}:{h}:{a}",
            "tipo": "club",
            "competicion": codigo,
            "temporada": temporada,
            "fecha": f_raw,
            "local": h,
            "visita": a,
            "goles_l": int(gl),
            "goles_v": int(gv),
            "tiros_p_l": sot_l,
            "tiros_p_v": sot_v,
            "corners_l": _stat(c_h, "wonCorners", "cornerKicks"),
            "corners_v": _stat(c_a, "wonCorners", "cornerKicks"),
            "faltas_l": _stat(c_h, "foulsCommitted", "fouls"),
            "faltas_v": _stat(c_a, "foulsCommitted", "fouls"),
            "amarillas_l": _stat(c_h, "yellowCards"),
            "amarillas_v": _stat(c_a, "yellowCards"),
            "rojas_l": _stat(c_h, "redCards"),
            "rojas_v": _stat(c_a, "redCards"),
            "cuota_l": None,
            "cuota_e": None,
            "cuota_v": None,
            "arbitro": None,
            "xg_l": xg_l,
            "xg_v": xg_v,
        })
    return out


def _descargar_historial_espn(slug, codigo, anios=3):
    """Descarga los últimos `anios` años de una liga desde los endpoints públicos de ESPN (/teams y /schedule)."""
    from concurrent.futures import ThreadPoolExecutor
    from datetime import date as _date

    try:
        r_t = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/teams",
                           headers=HEADERS, timeout=12)
        if r_t.status_code != 200:
            return []
        equipos_raw = (
            r_t.json()
            .get("sports", [{}])[0]
            .get("leagues", [{}])[0]
            .get("teams", [])
        )
        team_ids = [str(t.get("team", {}).get("id")) for t in equipos_raw if t.get("team", {}).get("id")]
    except Exception:
        return []

    if not team_ids:
        return []

    anio_actual = _date.today().year
    temporadas = list(range(anio_actual - max(2, int(anios)) + 1, anio_actual + 1))
    tareas = [(tid, y) for y in temporadas for tid in team_ids]

    def _pedir_schedule(par):
        tid, y = par
        try:
            url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/teams/{tid}/schedule?season={y}"
            r = requests.get(url, headers=HEADERS, timeout=12)
            if r.status_code == 200:
                return _filas_espn_eventos(r.json().get("events", []), codigo)
        except Exception:
            pass
        return []

    filas = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        for res in ex.map(_pedir_schedule, tareas):
            filas.extend(res)

    # Deduplicar por id (si una versión trae estadísticas completas, conservarla)
    vistos = {}
    for f in filas:
        cid = f["id"]
        if cid not in vistos or (vistos[cid]["tiros_p_l"] is None and f["tiros_p_l"] is not None):
            vistos[cid] = f
    return sorted(vistos.values(), key=lambda x: x["fecha"])


def importar_archivo_usuario(con, contenido_bytes, nombre_archivo, codigo_liga):
    """Permite al usuario subir un CSV o Excel propio (ej. Liga 1 de Perú con cuotas de Betsafe/Betano)."""
    if str(nombre_archivo).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(contenido_bytes))
    else:
        df = pd.read_csv(io.BytesIO(contenido_bytes), encoding="latin-1", on_bad_lines="skip")
    # Soportar tanto nombres en inglés (Date, HomeTeam, AwayTeam, FTHG, FTAG) como español (Fecha, Local, Visita, GolesL, GolesV)
    mapa_cols = {
        "Fecha": "Date", "fecha": "Date",
        "Local": "HomeTeam", "local": "HomeTeam", "Home": "HomeTeam",
        "Visita": "AwayTeam", "visita": "AwayTeam", "Visitante": "AwayTeam", "Away": "AwayTeam",
        "GolesL": "FTHG", "goles_l": "FTHG", "HG": "FTHG",
        "GolesV": "FTAG", "goles_v": "FTAG", "AG": "FTAG",
        "Cuota1": "B365H", "cuota_l": "B365H",
        "CuotaX": "B365D", "cuota_e": "B365D",
        "Cuota2": "B365A", "cuota_v": "B365A",
    }
    df = df.rename(columns={k: v for k, v in mapa_cols.items() if k in df.columns and v not in df.columns})
    temporada = str(pd.Timestamp.today().year)
    filas = _filas(df, codigo_liga, temporada)
    if not filas:
        filas = filas_xlsx(df, codigo_liga, anios=10)
    n = db.guardar_partidos(con, filas)
    db.registrar_ingesta(con, f"archivo:{nombre_archivo}", codigo_liga, n)
    return n


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
            if cfg["fuente"] == "csv" and t:
                d = pd.read_csv(io.BytesIO(_descargar(URL_CSV.format(t=t, c=cfg["codigo"]))),
                                encoding="latin-1", on_bad_lines="skip")
                nuevas = _filas(d, cfg["codigo"], t)
                _enriquecer_xg_understat(nuevas, cfg["codigo"], t)
            elif cfg["fuente"] == "espn":
                slug = cfg.get("slug") or ESPN_SLUGS.get(cfg["codigo"], "per.1")
                nuevas = _descargar_historial_espn(slug, cfg["codigo"], anios=max(2, anios_xlsx))
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
    db.registrar_ingesta(con, "football-data.co.uk+espn", ", ".join(cfg_n for cfg_n in nombres), n)
    return n, avisos

def _filas_fifa_csv(df, anios=4):
    """Convierte el repositorio abierto de resultados internacionales FIFA (martj42/international_results)
    al esquema SQLite bajo competicion='FIFA' y tipo='seleccion', calibrando estadísticas de tiros, xG,
    córners, faltas y tarjetas según el marcador, jerarquía FIFA y rigor del torneo."""
    req = {"date", "home_team", "away_team", "home_score", "away_score"}
    if not req <= set(df.columns):
        return []
    d = df.copy()
    d["date_dt"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date_dt", "home_team", "away_team", "home_score", "away_score"])
    if d.empty:
        return []
    max_anio = int(d["date_dt"].dt.year.max())
    d = d[d["date_dt"].dt.year >= max_anio - max(1, int(anios)) + 1].sort_values("date_dt")

    out = []
    for r in d.to_dict("records"):
        h = str(r["home_team"]).strip()
        a = str(r["away_team"]).strip()
        if not h or not a or h == a:
            continue
        gl = int(float(r["home_score"]))
        gv = int(float(r["away_score"]))
        f_iso = r["date_dt"].strftime("%Y-%m-%d")
        temp = str(r["date_dt"].year)
        torneo = str(r.get("tournament") or "Friendly").strip()
        t_low = torneo.lower()

        if "friendly" in t_low:
            cat_torneo = "Torneo: Amistoso FIFA"
            f_rigor = 0.72
        elif any(k in t_low for k in ("world cup", "copa am", "euro", "african cup", "asian cup", "gold cup")):
            cat_torneo = "Torneo: Fase Final / Eliminatoria"
            f_rigor = 1.15
        else:
            cat_torneo = "Torneo: Nations League / Clasificatoria"
            f_rigor = 1.00

        rk_h = float(RANKING_FIFA_PUNTOS.get(h, 1420))
        rk_a = float(RANKING_FIFA_PUNTOS.get(a, 1420))
        vent_h = np.clip((rk_h - rk_a) / 250.0, -1.2, 1.2)

        sot_l = round(max(float(gl), 2.8 + 1.35 * gl + 0.65 * max(0.0, vent_h)), 1)
        sot_v = round(max(float(gv), 2.3 + 1.35 * gv + 0.65 * max(0.0, -vent_h)), 1)
        xg_l = round(0.32 * sot_l + 0.14 * gl, 3)
        xg_v = round(0.32 * sot_v + 0.14 * gv, 3)
        cor_l = round(max(1.0, 4.4 + 0.65 * gl + 0.8 * max(0.0, vent_h)), 1)
        cor_v = round(max(1.0, 3.8 + 0.65 * gv + 0.8 * max(0.0, -vent_h)), 1)
        fal_l = round(11.5 * f_rigor + 0.4 * gv, 1)
        fal_v = round(12.2 * f_rigor + 0.4 * gl, 1)
        ama_l = round(max(0.0, (1.65 + 0.15 * gv) * f_rigor), 1)
        ama_v = round(max(0.0, (1.85 + 0.15 * gl) * f_rigor), 1)
        roj_l = 1.0 if (f_rigor > 1.05 and (gl + gv) >= 5 and gl < gv) else 0.0
        roj_v = 1.0 if (f_rigor > 1.05 and (gl + gv) >= 5 and gv < gl) else 0.0

        out.append({
            "id": f"FIFA:{f_iso}:{h}:{a}",
            "tipo": "seleccion",
            "competicion": "FIFA",
            "temporada": temp,
            "fecha": f_iso,
            "local": h,
            "visita": a,
            "goles_l": gl,
            "goles_v": gv,
            "tiros_p_l": sot_l,
            "tiros_p_v": sot_v,
            "corners_l": cor_l,
            "corners_v": cor_v,
            "faltas_l": fal_l,
            "faltas_v": fal_v,
            "amarillas_l": ama_l,
            "amarillas_v": ama_v,
            "rojas_l": roj_l,
            "rojas_v": roj_v,
            "cuota_l": None,
            "cuota_e": None,
            "cuota_v": None,
            "arbitro": cat_torneo,
            "xg_l": xg_l,
            "xg_v": xg_v,
        })
    return out


def ingestar_selecciones_fifa(con, anios=4, progreso=None):
    """Descarga el historial global de Selecciones FIFA (Mundial, Copa América, Euro, Nations League,
    Eliminatorias de todas las confederaciones, Copa Oro, Copa África, Copa Asia y Amistosos)."""
    avisos = []
    filas = []
    try:
        if progreso:
            progreso(0.2, "Descargando historial global de Selecciones FIFA...")
        raw = _descargar(URL_FIFA_RESULTS)
        df_int = pd.read_csv(io.BytesIO(raw), encoding="utf-8", on_bad_lines="skip")
        filas = _filas_fifa_csv(df_int, anios=anios)
        if progreso:
            progreso(0.8, f"Procesando {len(filas)} partidos internacionales FIFA...")
    except Exception as e:
        avisos.append(f"Selecciones FIFA: {type(e).__name__} - {e}")
    n = db.guardar_partidos(con, filas)
    db.registrar_ingesta(con, "international_results+espn", "FIFA", n)
    if progreso:
        progreso(1.0, "Selecciones FIFA listas")
    return n, avisos


def obtener_equipos_vigentes(con, competicion):
    """Obtiene todos los equipos (locales y visitantes) vigentes de la competición."""
    if competicion == "FIFA":
        query_fifa = """
            WITH max_t AS (
                SELECT CAST(MAX(temporada) AS INTEGER) AS mt FROM partidos WHERE competicion = 'FIFA'
            )
            SELECT DISTINCT local AS equipo FROM partidos
            WHERE competicion = 'FIFA' AND CAST(temporada AS INTEGER) >= (SELECT mt - 2 FROM max_t)
            UNION
            SELECT DISTINCT visita AS equipo FROM partidos
            WHERE competicion = 'FIFA' AND CAST(temporada AS INTEGER) >= (SELECT mt - 2 FROM max_t)
        """
        df = pd.read_sql_query(query_fifa, con)
        eqs = sorted(df["equipo"].tolist())
        return eqs if eqs else sorted(RANKING_FIFA_PUNTOS.keys())

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
    "psv eindhoven": "PSV Eindhoven",
    "feyenoord rotterdam": "Feyenoord",
    "az alkmaar": "AZ Alkmaar",
    "twente": "Twente",
    "ajax amsterdam": "Ajax",
    "club brugge": "Club Brugge",
    "union saint gilloise": "St. Gilloise",
    "royale union sg": "St. Gilloise",
    "anderlecht": "Anderlecht",
    "krc genk": "Genk",
    "kaa gent": "Gent",
    "olympiacos": "Olympiakos",
    "paok salonika": "PAOK",
    "aek athens": "AEK",
    "panathinaikos": "Panathinaikos",
    "glasgow rangers": "Rangers",
    "celtic glasgow": "Celtic",
    "red bull salzburg": "RB Salzburg",
    "salzburg": "RB Salzburg",
    "red star belgrade": "Crvena Zvezda",
    "universitario de deportes": "Universitario",
    "club universitario de deportes": "Universitario",
    "alianza lima": "Alianza Lima",
    "sporting cristal": "Sporting Cristal",
    "fbc melgar": "Melgar",
    "melgar": "Melgar",
    "asociacion deportiva tarma": "ADT",
    "adt tarma": "ADT",
    "deportivo garcilaso": "Deportivo Garcilaso",
    "cusco": "Cusco FC",
    "cienciano": "Cienciano",
    "sport huancayo": "Sport Huancayo",
    "chankas cyc": "Los Chankas",
    "los chankas": "Los Chankas",
    "universidad tecnica de cajamarca": "UTC",
    "utc cajamarca": "UTC",
    "comerciantes unidos": "Comerciantes Unidos",
    "alianza atletico": "Alianza Atletico",
    "atletico grau": "Atletico Grau",
    "sport boys": "Sport Boys",
    "universidad cesar vallejo": "Cesar Vallejo",
    "cesar vallejo": "Cesar Vallejo",
    "carlos a mannucci": "Mannucci",
    "deportivo binacional": "Binacional",
}


def _norm_nombre(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii").lower()
    for tok in (" fc", " cf", " sc", " ac", "fk ", "fc ", "cd ", "ud ", "rcd ", "1. ", "sv "):
        s = s.replace(tok, " ")
    return " ".join(
        "".join(ch if ch.isalnum() else " " for ch in s).split()
    )


def _emparejar_equipo(nombre_ext, equipos_db, cutoff=0.58):
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
        if len(k_norm) >= 4 and (k_norm in n_ext or n_ext in k_norm):
            return orig
    cercanos = difflib.get_close_matches(n_ext, list(mapa_norm.keys()), n=1, cutoff=cutoff)
    return mapa_norm[cercanos[0]] if cercanos else None


def resolver_equipo_internacional(nombre_ext, equipos_por_liga_total, elo_defecto=1400):
    """Resuelve un equipo de Champions/Europa/Conference/Libertadores contra:
    1) Todas las ligas domésticas cargadas en SQLite, 2) El catálogo EQUIPOS_INVITADOS, o 3) Arquetipo por defecto."""
    n_ext = _norm_nombre(nombre_ext)
    # 1. Buscar coincidencia exacta o por alias en todas las ligas domésticas
    for cod, eqs in equipos_por_liga_total.items():
        if not eqs:
            continue
        if nombre_ext in eqs:
            return {"equipo": nombre_ext, "codigo": cod, "invitado": False, "elo_ext": None}
        if n_ext in ALIAS_EQUIPOS and ALIAS_EQUIPOS[n_ext] in eqs:
            return {"equipo": ALIAS_EQUIPOS[n_ext], "codigo": cod, "invitado": False, "elo_ext": None}
        mapa_norm = {_norm_nombre(e): e for e in eqs}
        if n_ext in mapa_norm:
            return {"equipo": mapa_norm[n_ext], "codigo": cod, "invitado": False, "elo_ext": None}
    # 2. Buscar subcadena o similitud alta (>=0.72) en ligas domésticas
    for cod, eqs in equipos_por_liga_total.items():
        m = _emparejar_equipo(nombre_ext, eqs, cutoff=0.72)
        if m:
            return {"equipo": m, "codigo": cod, "invitado": False, "elo_ext": None}
    # 3. Buscar en el catálogo de EQUIPOS_INVITADOS (ligas pequeñas europeas y sudamericanas)
    m_inv = _emparejar_equipo(nombre_ext, list(EQUIPOS_INVITADOS.keys()), cutoff=0.65)
    if m_inv:
        info = EQUIPOS_INVITADOS[m_inv]
        return {"equipo": m_inv, "codigo": "EXT", "invitado": True,
                "elo_ext": int(info["elo"]), "pais": info["pais"]}
    # 4. Comodín automático para cualquier club nuevo o de fase previa
    limpio = str(nombre_ext).strip()
    return {"equipo": limpio, "codigo": "EXT", "invitado": True,
            "elo_ext": int(elo_defecto), "pais": "Internacional"}


def buscar_partidos_del_dia(con, fecha_obj, nombres_ligas=None, ventana_dias=0):
    """Busca los partidos programados en la fecha indicada (y hasta +ventana_dias) para las ligas, torneos internacionales y selecciones FIFA.
    Combina: 1) football-data.co.uk fixtures, 2) calendario diario público de ESPN (ligas + UEFA/CONMEBOL + Selecciones FIFA) y 3) base local SQLite."""
    nombres_ligas = nombres_ligas or (list(LIGAS) + list(TORNEOS_INTERNACIONALES))
    cod_a_nombre = {LIGAS[n]["codigo"]: n for n in nombres_ligas if n in LIGAS}
    torneos_sel = {n: TORNEOS_INTERNACIONALES[n] for n in nombres_ligas if n in TORNEOS_INTERNACIONALES}
    fifa_sel = {n: COMPETENCIAS_FIFA[n] for n in nombres_ligas if n in COMPETENCIAS_FIFA}
    codigos = set(cod_a_nombre.keys())

    todos_cods = {v["codigo"] for v in LIGAS.values()}
    equipos_por_liga_total = {c: obtener_equipos_vigentes(con, c) for c in todos_cods}
    equipos_por_liga = {c: equipos_por_liga_total.get(c, []) for c in codigos}
    equipos_fifa = obtener_equipos_vigentes(con, "FIFA") if fifa_sel else []

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
        clave = (cod, cod, f_str, h_db, a_db)
        if clave in vistos:
            return
        vistos.add(clave)
        encontrados.append({
            "liga": cod_a_nombre[cod],
            "codigo": cod,
            "codigo_a": cod,
            "fecha": f_str,
            "local": h_db,
            "visita": a_db,
            "cuota_l": _num(cl),
            "cuota_e": _num(ce),
            "cuota_v": _num(cv),
            "fuente": fuente,
        })

    def _registrar_internacional(nom_torneo, cfg_t, f_str, h_raw, a_raw):
        ih = resolver_equipo_internacional(h_raw, equipos_por_liga_total, elo_defecto=cfg_t["elo_base"])
        ia = resolver_equipo_internacional(a_raw, equipos_por_liga_total, elo_defecto=cfg_t["elo_base"])
        if ih["equipo"] == ia["equipo"]:
            return
        clave = (ih["codigo"], ia["codigo"], f_str, ih["equipo"], ia["equipo"])
        if clave in vistos:
            return
        vistos.add(clave)
        encontrados.append({
            "liga": nom_torneo,
            "codigo": ih["codigo"],
            "codigo_a": ia["codigo"],
            "fecha": f_str,
            "local": ih["equipo"],
            "visita": ia["equipo"],
            "elo_ext_h": ih["elo_ext"],
            "elo_ext_a": ia["elo_ext"],
            "cuota_l": None,
            "cuota_e": None,
            "cuota_v": None,
            "fuente": "torneo-int",
        })

    def _registrar_fifa(nom_comp, f_str, h_raw, a_raw, cl=None, ce=None, cv=None, fuente="fifa-vivo"):
        h_db = _emparejar_equipo(h_raw, equipos_fifa) or normalizar_equipo(h_raw)
        a_db = _emparejar_equipo(a_raw, equipos_fifa) or normalizar_equipo(a_raw)
        if not h_db or not a_db or h_db == a_db:
            return
        clave = ("FIFA", "FIFA", f_str, h_db, a_db)
        if clave in vistos:
            return
        vistos.add(clave)
        encontrados.append({
            "liga": nom_comp,
            "codigo": "FIFA",
            "codigo_a": "FIFA",
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

    # 2. Consultar calendario diario público de ESPN (ligas domésticas + torneos internacionales UEFA/CONMEBOL)
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

        for nom_torneo, cfg_t in torneos_sel.items():
            try:
                r = requests.get(URL_ESPN_SCOREBOARD.format(slug=cfg_t["slug"], ymd=ymd), headers=HEADERS, timeout=10)
                if r.status_code != 200:
                    continue
                eventos = r.json().get("events", [])
                for ev in eventos:
                    comps = ev.get("competitions", [{}])[0].get("competitors", [])
                    h_name = next((c.get("team", {}).get("name") for c in comps if c.get("homeAway") == "home"), None)
                    a_name = next((c.get("team", {}).get("name") for c in comps if c.get("homeAway") == "away"), None)
                    if h_name and a_name:
                        _registrar_internacional(nom_torneo, cfg_t, f_iso, h_name, a_name)
            except Exception:
                pass

        for nom_fifa, cfg_f in fifa_sel.items():
            try:
                r = requests.get(URL_ESPN_SCOREBOARD.format(slug=cfg_f["slug"], ymd=ymd), headers=HEADERS, timeout=10)
                if r.status_code != 200:
                    continue
                eventos = r.json().get("events", [])
                for ev in eventos:
                    comps = ev.get("competitions", [{}])[0].get("competitors", [])
                    h_name = next((c.get("team", {}).get("name") for c in comps if c.get("homeAway") == "home"), None)
                    a_name = next((c.get("team", {}).get("name") for c in comps if c.get("homeAway") == "away"), None)
                    if h_name and a_name:
                        _registrar_fifa(nom_fifa, f_iso, h_name, a_name, fuente="fifa-vivo")
            except Exception:
                pass

    # 3. Respaldo en base de datos local SQLite (útil sin red o al consultar una fecha ya registrada)
    if codigos:
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

    if fifa_sel:
        nom_def_fifa = next(iter(fifa_sel.keys()))
        df_f = pd.read_sql_query(
            """SELECT fecha, local, visita, cuota_l, cuota_e, cuota_v, arbitro
               FROM partidos
               WHERE competicion = 'FIFA' AND fecha >= ? AND fecha <= ?
               ORDER BY fecha, id""",
            con, params=[f_ini.strftime("%Y-%m-%d"), f_fin.strftime("%Y-%m-%d")],
        )
        for _, r in df_f.iterrows():
            nom_c = r["arbitro"] if r.get("arbitro") in COMPETENCIAS_FIFA else nom_def_fifa
            _registrar_fifa(nom_c, r["fecha"], r["local"], r["visita"],
                            r["cuota_l"], r["cuota_e"], r["cuota_v"], "fifa-local")

    return encontrados