"""Pruebas sin red (datos sintéticos). Ejecutar:  python -m tests.test_engine"""
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import check_grad

from engine import backtest as bt
from engine import data_bot, db, models as md
from engine.config import ELO_A_LOG, LIGAS
from engine.elo_system import EloSystem, esperado_a_1x2, prob_cruzada
from engine.features import FEATS, capacidades, construir_features, esperado_conteo_cruzado


def sinteticos(codigo, n=700, seed=0, c0=0.30, ha0=0.20):
    rng = np.random.default_rng(seed)
    eq = [f"{codigo}_Eq{i}" for i in range(10)]
    att, dfn = rng.normal(0, .3, 10), rng.normal(0, .3, 10)
    filas, base = [], pd.Timestamp("2025-08-01")
    for k in range(n):
        h, a = rng.choice(10, 2, replace=False)
        gl = rng.poisson(np.exp(c0 + ha0 + att[h] - dfn[a]))
        gv = rng.poisson(np.exp(c0 + att[a] - dfn[h]))
        f = (base + pd.Timedelta(days=int(k * 0.9))).strftime("%Y-%m-%d")
        filas.append(dict(id=f"{codigo}:{k}", tipo="club", competicion=codigo, temporada="2526", fecha=f,
                          local=eq[h], visita=eq[a], goles_l=int(gl), goles_v=int(gv),
                          tiros_p_l=float(rng.poisson(4)), tiros_p_v=float(rng.poisson(3)),
                          corners_l=float(rng.poisson(5)), corners_v=float(rng.poisson(4)),
                          faltas_l=float(rng.poisson(11)), faltas_v=float(rng.poisson(12)),
                          amarillas_l=float(rng.poisson(1.8)), amarillas_v=float(rng.poisson(2.1)),
                          rojas_l=float(rng.poisson(.05)), rojas_v=float(rng.poisson(.06)),
                          cuota_l=2.1, cuota_e=3.3, cuota_v=3.4,
                          arbitro="Ref_Severo" if k % 2 == 0 else "Ref_Suave"))
    return filas


def test_catalogo():
    assert {c["codigo"] for c in LIGAS.values()} == {
        "E0", "E1", "D1", "SP1", "I1", "F1", "P1", "T1", "BRA", "ARG", "JPN", "CHN", "USA"
    }
    assert all(c["fuente"] in ("csv", "xlsx") for c in LIGAS.values())


def test_parsers():
    # CSV estilo /mmz4281/
    csv = pd.DataFrame({"Date": ["15/08/2025", "16/08/2025"], "HomeTeam": ["A", "C"], "AwayTeam": ["B", "D"],
                        "FTHG": [2, 0], "FTAG": [1, 0], "FTR": ["H", "D"], "HST": [5, 3], "HC": [6, 4],
                        "AC": [3, 2], "HF": [11, 9], "AF": [13, 10], "HY": [2, 1], "AY": [3, 0], "HR": [0, 0],
                        "AR": [1, 0], "PSCH": [1.8, 2.5], "PSCD": [3.6, 3.2], "PSCA": [4.5, 3.0]})
    f = data_bot._filas(csv, "E0", "2526")
    assert len(f) == 2 and f[0]["fecha"] == "2025-08-15" and f[0]["tiros_p_l"] == 5 and f[0]["cuota_l"] == 1.8
    assert f[0]["faltas_l"] == 11 and f[0]["amarillas_v"] == 3 and f[0]["rojas_v"] == 1
    # XLSX estilo /new/ (Home, Away, HG, AG, Res, Season numérico, fechas reales, sin estadísticas)
    x = pd.DataFrame({"Country": "Brazil", "League": "Serie A", "Season": [2022, 2026, 2026],
                      "Date": pd.to_datetime(["2022-04-10", "2026-04-12", "2026-05-01"]),
                      "Home": ["X", "Y", "Z"], "Away": ["Y", "Z", "X"], "HG": [1, 2, 0], "AG": [0, 2, 3],
                      "Res": ["H", "D", "A"], "AvgH": [2.0, 2.2, 3.1], "AvgD": [3.3, 3.2, 3.3], "AvgA": [3.8, 3.4, 2.3]})
    fx = data_bot.filas_xlsx(x, "BRA", anios=3)
    assert len(fx) == 2 and fx[0]["temporada"] == "2026" and fx[0]["tiros_p_l"] is None and fx[0]["cuota_v"] == 3.4
    assert fx[0]["faltas_l"] is None and fx[0]["amarillas_l"] is None
    assert len(data_bot.filas_xlsx(x, "BRA", anios=10)) == 3
    # Sin columna Res: se calcula
    assert data_bot.filas_xlsx(x.drop(columns="Res"), "BRA", 10)[2]["resultado"] == "A"
    # Lectura real de xlsx (si hay openpyxl)
    try:
        import openpyxl  # noqa: F401
        buf = io.BytesIO()
        x.to_excel(buf, index=False)
        assert len(data_bot.filas_xlsx(pd.read_excel(io.BytesIO(buf.getvalue())), "BRA", 10)) == 3
    except ImportError:
        pass


def test_migracion_y_capacidades():
    import sqlite3
    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / "vieja.db"
        con = sqlite3.connect(str(ruta))          # esquema de la versión anterior (sin faltas/tarjetas)
        con.execute("CREATE TABLE partidos (id TEXT PRIMARY KEY, tipo TEXT NOT NULL, competicion TEXT, temporada TEXT, "
                    "fecha TEXT NOT NULL, local TEXT NOT NULL, visita TEXT NOT NULL, goles_l INTEGER, goles_v INTEGER, "
                    "resultado TEXT, tiros_l REAL, tiros_v REAL, tiros_p_l REAL, tiros_p_v REAL, corners_l REAL, "
                    "corners_v REAL, cuota_l REAL, cuota_e REAL, cuota_v REAL)")
        con.execute("INSERT INTO partidos (id,tipo,competicion,temporada,fecha,local,visita,goles_l,goles_v) "
                    "VALUES ('x','club','E0','2526','2025-08-01','A','B',1,0)")
        con.commit()
        con.close()
        con = db.conectar(ruta)
        cols = {r[1] for r in con.execute("PRAGMA table_info(partidos)")}
        assert {"faltas_l", "amarillas_v", "rojas_l"} <= cols
        assert db.total_partidos(con) == 1                       # no se pierde nada
        con.close()

    full = pd.DataFrame(sinteticos("E0", 300))
    d = pd.DataFrame({"HC": full["corners_l"], "AC": full["corners_v"], "HST": full["tiros_p_l"],
                      "AST": full["tiros_p_v"], "HF": full["faltas_l"], "AF": full["faltas_v"],
                      "HY": full["amarillas_l"], "AY": full["amarillas_v"], "HR": full["rojas_l"], "AR": full["rojas_v"]})
    assert all(capacidades(d).values())
    vacio = pd.DataFrame({c: np.nan for c in d.columns}, index=range(300))      # Brasil, Argentina, ...
    assert not any(capacidades(vacio).values())
    solo_corners = d.assign(HF=np.nan, AF=np.nan, HY=np.nan, AY=np.nan)
    cap = capacidades(solo_corners)
    assert cap["corners"] and not cap["faltas"] and not cap["tarjetas"]


def test_motor():
    with tempfile.TemporaryDirectory() as tmp:
        con = db.conectar(Path(tmp) / "t.db")
        assert db.guardar_partidos(con, sinteticos("T1")) == 700
        db.guardar_partidos(con, sinteticos("T1"))                       # idempotente
        db.guardar_partidos(con, sinteticos("T2", 500, seed=3, c0=0.1))
        assert db.total_partidos(con) == 1200
        assert set(db.resumen_por_liga(con)["Liga"]) == {"T1", "T2"}
        df = db.cargar_partidos("T1", con)
        assert len(df) == 700 and abs(df[["mH", "mD", "mA"]].sum(axis=1).iloc[0] - 1) < 1e-9

        e = EloSystem()
        e.update("A", "B", 3, 0)
        assert e.rating("A") > 1500 > e.rating("B")
        r_prev = e.rating("A")
        e.regresion_temporada(0.82)
        assert 1500 < e.rating("A") < r_prev
        assert abs(esperado_a_1x2(0.7).sum() - 1) < 1e-12
        # Verifica método Power para desmargenar cuotas
        p_mkt = db.desmargenar_cuotas([1.5, 4.0, 6.5])
        assert abs(p_mkt.sum() - 1.0) < 1e-12 and p_mkt[0] > 0.60

        feats, est = construir_features(df)
        assert feats["n_prev"].iloc[0] == 0 and feats["elo_diff"].iloc[0] == 0
        db.guardar_elo(con, "T1", est.elo.ratings, est.n)

        fun, p0, _, _ = md.preparar_dc(df, df["Date"].max(), 365)
        p = p0 + np.random.default_rng(1).normal(0, .1, len(p0))
        assert check_grad(lambda q: fun(q)[0], lambda q: fun(q)[1], p) < 1e-3 * max(1, np.linalg.norm(fun(p)[1]))

        # El nivel de goles debe coincidir con los datos (verifica el intercepto)
        dc, lr = md.entrenar_produccion(df, feats, 10_000)
        # Verifica warm-start en ajustar_dc
        dc_ws = md.ajustar_dc(df, df["Date"].max() + pd.Timedelta(days=1), 10_000, p0_prev=dc)
        assert abs(dc_ws["c"] - dc["c"]) < 1e-4
        # Verifica no-negatividad de la matriz en escenarios extremos
        M_ext = md.matriz(4.5, 4.5, 0.3)
        assert (M_ext >= 0).all() and abs(M_ext.sum() - 1.0) < 1e-12
        tasas = [md.tasas_dc(dc, h, a) for h, a in zip(df["HomeTeam"], df["AwayTeam"])]
        assert abs(np.mean([t[0] for t in tasas]) / df["FTHG"].mean() - 1) < 0.08
        assert abs(np.mean([t[1] for t in tasas]) / df["FTAG"].mean() - 1) < 0.08

        lam, mu = md.tasas_dc(dc, "T1_Eq1", "T1_Eq2")
        M = md.matriz(lam, mu, dc["rho"])
        assert abs(md.probs_1x2(M).sum() - 1) < 1e-9 and md.tasas_dc(dc, "T1_Eq1", "T1_Eq2", True)[0] < lam
        p_lr = md.predecir_lr(lr, pd.DataFrame([est.features("T1_Eq1", "T1_Eq2", df["Date"].max())])[FEATS])
        assert abs(p_lr.sum() - 1) < 1e-9
        # Verifica resumen de selecciones de bajo riesgo (2 a 3 opciones de distintas categorías)
        mk = md.mercados_goles(M)
        s_1x2_test = np.tile(md.probs_1x2(M), (50, 1))
        picks = md.selecciones_bajo_riesgo("T1_Eq1", "T1_Eq2", md.probs_1x2(M), s_1x2_test, mk, [mk] * 50)
        assert 2 <= len(picks) <= 3 and all(p["Probabilidad"] >= 0.58 for p in picks)

        # Verifica las 6 mejoras de apostador (asiáticos/DNB, Kelly/EV, consenso/trampa, árbitro, escáner, backtest financiero)
        asi = md.mercados_asiaticos(M, md.probs_1x2(M), "T1_Eq1", "T1_Eq2")
        assert "T1_Eq1 DNB (AH 0.0)" in asi["handicap"].index and "Over 2 goles (asiático)" in asi["goles"].index
        kelly = md.calcular_kelly_ev(0.60, 0.55, 1.90)
        assert kelly["EV %"] > 10.0 and kelly["Stake Kelly 1/4 %"] > 0.0 and "Valor" in kelly["Señal"]
        cons_ok = md.diagnosticar_consenso([0.50, 0.28, 0.22], [0.49, 0.28, 0.23], [0.51, 0.27, 0.22], 25, 25)
        cons_trampa = md.diagnosticar_consenso([0.60, 0.20, 0.20], [0.35, 0.30, 0.35], [0.40, 0.25, 0.35], 25, 25)
        assert not cons_ok["es_trampa"] and cons_trampa["es_trampa"]
        assert "Ref_Severo" in est.arbitros_disponibles()
        f_arb = est.factor_arbitro("Ref_Severo")
        assert f_arb["partidos"] > 10 and 0.65 <= f_arb["amarillas"] <= 1.50
        scan = md.escanear_jornada([("T1_Eq1", "T1_Eq2"), ("T1_Eq3", "T1_Eq4")],
                                   {"estado": est, "dc": dc, "lr": lr, "df": df}, df["Date"].max())
        assert len(scan["ranking"]) == 2 and scan["combinada"] is not None
        # Verifica búsqueda de partidos del día y escaneo multiliga (Doble + Triple)
        assert data_bot._emparejar_equipo("Manchester United", ["Man United", "Arsenal"]) == "Man United"
        f_muestra = df["Date"].iloc[-1].strftime("%Y-%m-%d")
        partidos_dia = data_bot.buscar_partidos_del_dia(con, f_muestra, ["🇹🇷 Turquía · Süper Lig"], ventana_dias=5)
        assert len(partidos_dia) >= 1 and partidos_dia[0]["codigo"] == "T1"
        scan_multi = md.escanear_jornada(
            [{"liga": "Turquía", "codigo": "T1", "local": "T1_Eq1", "visita": "T1_Eq2"},
             {"liga": "Turquía", "codigo": "T1", "local": "T1_Eq3", "visita": "T1_Eq4"},
             {"liga": "Turquía", "codigo": "T1", "local": "T1_Eq5", "visita": "T1_Eq6"}],
            {"T1": {"estado": est, "dc": dc, "lr": lr, "df": df}},
            df["Date"].max(),
        )
        assert scan_multi["combinada"] is not None and scan_multi["triple"] is not None

        # Modo europeo entre ligas
        df2 = db.cargar_partidos("T2", con)
        f2, est2 = construir_features(df2)
        dc2, _ = md.entrenar_produccion(df2, f2, 365)
        l0, m0, _ = md.tasas_cruzadas(dc, dc, "T1_Eq1", "T1_Eq2", 0.0, 0.0)
        assert abs(l0 - lam) < 1e-9 and abs(m0 - mu) < 1e-9                  # misma liga = igual que dominio local
        lc, mc, rho = md.tasas_cruzadas(dc, dc2, "T1_Eq1", "T2_Eq4", 0.0, -250 * ELO_A_LOG)
        ls, ms, _ = md.tasas_cruzadas(dc, dc2, "T1_Eq1", "T2_Eq4", 0.0, 0.0)
        assert lc > ls and mc < ms                                            # liga más débil → visita rinde menos
        assert prob_cruzada(1500, 1250)[0] > prob_cruzada(1500, 1500)[0]
        eh, ea = esperado_conteo_cruzado(est, est2, "T1_Eq1", "T2_Eq4", "C", "CA")
        assert eh > 0 and ea > 0
        for fav, con_ in (("F", "FA"), ("Y", "YA"), ("R", "RA")):             # faltas, amarillas, rojas
            a, b = esperado_conteo_cruzado(est, est, "T1_Eq1", "T1_Eq2", fav, con_)
            assert a >= 0 and b >= 0
        fh, fa = esperado_conteo_cruzado(est, est, "T1_Eq1", "T1_Eq2", "F", "FA")
        assert 7 < fh < 15 and 8 < fa < 16                                  # ~Poisson(11) y Poisson(12)
        yh, ya = esperado_conteo_cruzado(est, est, "T1_Eq1", "T1_Eq2", "Y", "YA")
        assert 0.8 < yh < 3.2 and 0.8 < ya < 3.5

        R = bt.correr_backtest(df, feats, 365, 0.6, 150, 28)
        for k in ["base", "elo", "dc", "lr", "ens"]:
            assert 0.5 < bt.metricas(R[k], R["y"])["Log Loss"] < 1.6, k
        assert len(bt.tabla_calibracion(R["ens"], R["y"])) > 3
        sim_fin = bt.simular_apuestas(R["ens"], R["cuotas"], R["y"], ev_min=0.01, prob_min=0.45)
        assert "Yield / ROI %" in sim_fin["resumen"] and len(sim_fin["curva"]) >= 1
        con.close()


def test_ingesta_simulada():
    """ingestar_ligas completo con descargas simuladas (sin red)."""
    csv = ("Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HST,AST,HC,AC,B365H,B365D,B365A\n"
           "15/08/2025,A,B,2,1,H,5,2,6,3,1.8,3.6,4.5\n")
    x = pd.DataFrame({"Season": [2026], "Date": pd.to_datetime(["2026-04-12"]), "Home": ["Y"], "Away": ["Z"],
                      "HG": [2], "AG": [2], "Res": ["D"]})
    buf = io.BytesIO()
    orig_read_excel = pd.read_excel
    try:
        x.to_excel(buf, index=False)
    except ImportError:
        pd.read_excel = lambda *_a, **_k: x.copy()
    original = data_bot._descargar
    data_bot._descargar = lambda url: buf.getvalue() if url.endswith(".xlsx") else csv.encode("latin-1")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            con = db.conectar(Path(tmp) / "i.db")
            n, av = data_bot.ingestar_ligas(con, ["🇪🇸 España · La Liga", "🇧🇷 Brasil"], extra_temporadas=0)
            assert not av, av
            assert n == 3                    # 2 temporadas CSV + 1 xlsx
            con.close()
    finally:
        data_bot._descargar = original
        pd.read_excel = orig_read_excel


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for t in (test_catalogo, test_parsers, test_migracion_y_capacidades, test_motor, test_ingesta_simulada):
        t()
        print("✅", t.__name__)
