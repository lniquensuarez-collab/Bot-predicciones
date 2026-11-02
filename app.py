# Actualizacion de imports
# ==============================================================================
# 📱 ULTIMATE AI V14 - Ligas de clubes (football-data.co.uk) + modo europeo
# Ejecutar:  streamlit run app.py
# ==============================================================================
import importlib
import math
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

import engine. config as _cfg
import engine.elo_system as _elo
from engine import backtest as bt
from engine import data_bot, db
from engine import features as ft
from engine import models as md

# En Streamlit Cloud, al actualizar por Git sin reiniciar contenedor, recarga submódulos si quedaron en caché
if not hasattr(ft.Estado, "arbitros_disponibles") or not hasattr(md, "mercados_asiaticos") or not hasattr(bt, "simular_apuestas"):
    for _mod in (_cfg, _elo, db, data_bot, ft, md, bt):
        importlib.reload(_mod)

from engine.config import ELO_A_LOG, LIGAS
from engine.elo_system import prob_cruzada

st.set_page_config(page_title="Ultimate AI V14", page_icon="⚽", layout="centered")
st.title("⚽ ULTIMATE AI V14")
st.caption("Dixon-Coles + Elo + clasificador · Solo ligas de clubes · Datos: football-data.co.uk")

LIGAS_EU = [n for n, c in LIGAS.items() if c["europa"]]


# ------------------------------ utilidades -----------------------------------
def ejecutar(fn, *args, **kw):
    barra = st.progress(0.0, text="Iniciando...")
    try:
        res = fn(*args, progreso=lambda f, t="": barra.progress(min(max(f, 0.0), 1.0), text=t), **kw)
    except Exception as e:
        barra.empty()
        st.error(f"{type(e).__name__}: {e}")
        return None
    barra.empty()
    st.cache_data.clear()
    return res


@st.cache_data(show_spinner="Preparando liga (Elo, features, modelos)...")
def preparar_liga(codigo, half_life, _version="v14.2"):
    df = db.cargar_partidos(codigo)
    if len(df) < 200:
        return None
    feats, est = ft.construir_features(df)
    dc, lr = md.entrenar_produccion(df, feats, half_life)
    return {"df": df, "feats": feats, "estado": est, "dc": dc, "lr": lr}


def esperados(est_h, est_a, home, away, favor, contra, dh, ch, da, ca):
    """Conteo esperado de local y visita (EWMA propio + lo que el rival suele provocar); si falta, media de liga."""
    eh, ea = ft.esperado_conteo_cruzado(est_h, est_a, home, away, favor, contra)
    return (dh[ch].mean() if np.isnan(eh) else eh), (da[ca].mean() if np.isnan(ea) else ea)


def lineas(df_tab):
    return df_tab.round(1).fillna("-")


# ------------------------------- sidebar --------------------------------------
with st.sidebar:
    st.header("⚙️ Parámetros")
    half_life = st.slider("Vida media del peso temporal (días)", 90, 900, 365, step=30)
    w_dc = st.slider("Peso de Dixon-Coles en el ensemble", 0.0, 1.0, 0.6, step=0.1)
    liga_principal = st.selectbox("🏆 Liga (ranking y backtest)", list(LIGAS))

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔮 Predicción", "🏅 Ranking Elo", "📈 Backtest", "🗄️ Datos", "ℹ️ Metodología"])

# ------------------------------- primera carga --------------------------------
with db.sesion() as con:
    if db.total_partidos(con) == 0:
        with st.spinner("Primera ejecución: descargando todas las ligas de football-data.co.uk..."):
            n, av = data_bot.ingestar_ligas(con, list(LIGAS), extra_temporadas=0, anios_xlsx=3)
        for a in av:
            st.warning(a)
        st.cache_data.clear()

# Bases creadas con una versión anterior no tienen faltas/tarjetas/árbitros: se completan una sola vez por sesión.
if not st.session_state.get("refresco_faltas_v2"):
    st.session_state["refresco_faltas_v2"] = True
    with db.sesion() as con:
        cods_csv = tuple(c["codigo"] for c in LIGAS.values() if c["fuente"] == "csv")
        placeholders = ",".join("?" * len(cods_csv))
        r = con.execute(
            f"SELECT COUNT(*), COUNT(faltas_l), COUNT(arbitro) FROM partidos WHERE competicion IN ({placeholders})",
            cods_csv,
        ).fetchone()
        if r[0] > 0 and (r[1] == 0 or r[2] == 0):
            pend = [n for n, c in LIGAS.items() if c["fuente"] == "csv"]
            with st.spinner("Actualizando datos para incluir faltas, tarjetas y árbitros..."):
                data_bot.ingestar_ligas(con, pend, extra_temporadas=0)
            st.cache_data.clear()

# ----------------------------------- DATOS ------------------------------------
with tab4:
    with db.sesion() as con:
        st.dataframe(db.resumen_por_liga(con), use_container_width=True, hide_index=True)
    sel = st.multiselect("Ligas a descargar / actualizar", list(LIGAS), default=list(LIGAS))
    c1, c2 = st.columns(2)
    extra = c1.slider("Temporadas europeas anteriores (además de las 2 últimas)", 0, 4, 0,
                      help="Más historial mejora la estabilidad del modelo en las ligas con CSV por temporada.")
    anios = c2.slider("Años de historial (Brasil, Argentina, Japón, China, EE. UU.)", 1, 8, 3)
    if st.button("🔄 Descargar / actualizar", use_container_width=True):
        with db.sesion() as con:
            r = ejecutar(lambda progreso: data_bot.ingestar_ligas(con, sel, extra, anios, progreso))
        if r:
            st.success(f"{r[0]} partidos procesados")
            for a in r[1]:
                st.warning(a)
    if st.button("🧹 Limpiar caché de cálculos"):
        st.cache_data.clear()
        st.rerun()

# ------------------------------ liga principal --------------------------------
P_main = preparar_liga(LIGAS[liga_principal]["codigo"], half_life)

# ------------------------------- PREDICCIÓN -----------------------------------
with tab1:
    cruzado = st.checkbox("🌍 Partido europeo entre ligas distintas (Champions / Europa / Conference League)")
    neutral = st.checkbox("Cancha neutral (ej. una final)", value=False)

    if cruzado:
        l1, l2 = st.columns(2)
        lig_h = l1.selectbox("Liga del local", LIGAS_EU, index=0, key="lig_h")
        lig_a = l2.selectbox("Liga de la visita", LIGAS_EU, index=min(3, len(LIGAS_EU) - 1), key="lig_a")
        with st.expander("⚖️ Fuerza relativa de las ligas (supuesto editable)"):
            st.caption("Los archivos solo contienen partidos de cada liga contra sí misma, así que la diferencia de "
                       "nivel entre ligas NO se puede estimar con los datos. Estos valores (puntos Elo, Premier = 0) "
                       "son un supuesto inicial: ajústalos según tu criterio; los resultados dependen de ellos.")
            fuerza = {n: st.slider(n, -400, 100, int(LIGAS[n]["fuerza"]), step=5, key=f"fz_{n}") for n in LIGAS_EU}
    else:
        lig_h = lig_a = liga_principal
        fuerza = {}

    P_h = P_main if lig_h == liga_principal else preparar_liga(LIGAS[lig_h]["codigo"], half_life)
    P_a = P_h if lig_a == lig_h else preparar_liga(LIGAS[lig_a]["codigo"], half_life)
    if P_h is None or P_a is None:
        st.info("Una de las ligas no tiene datos suficientes (mínimo 200 partidos). Ve a **🗄️ Datos**.")
        st.stop()
    cross = cruzado and lig_h != lig_a

    with db.sesion() as con:
        eq_h = data_bot.obtener_equipos_vigentes(con, LIGAS[lig_h]["codigo"])
        eq_a = data_bot.obtener_equipos_vigentes(con, LIGAS[lig_a]["codigo"])

    c1, c2 = st.columns(2)
    home = c1.selectbox("🏠 Local:", eq_h, index=0)
    away = c2.selectbox("✈️ Visita:", eq_a, index=min(1, len(eq_a) - 1) if eq_a else 0)
    fecha = pd.Timestamp(st.date_input("📅 Fecha del partido", value=date.today()))

    fn_arbs = getattr(P_h["estado"], "arbitros_disponibles", None)
    arbs_disp = fn_arbs() if (not cross and callable(fn_arbs)) else []
    with st.expander("🎛️ Escenario (ajustes manuales y árbitro)"):
        st.caption("Si modificas los % de ataque/defensa se usa solo Dixon-Coles. El árbitro ajusta faltas y tarjetas.")
        arbitro_sel = st.selectbox("⚖️ Árbitro designado (opcional, ajusta faltas y tarjetas)",
                                   ["Promedio de la liga (sin especificar)"] + arbs_disp, index=0)
        s1, s2 = st.columns(2)
        att_h = s1.slider("Ataque local %", -40, 40, 0)
        att_a = s2.slider("Ataque visita %", -40, 40, 0)
        def_h = s1.slider("Debilidad defensa local %", -40, 40, 0)
        def_a = s2.slider("Debilidad defensa visita %", -40, 40, 0)
        sigma = st.slider("Incertidumbre del modelo (σ)", 0.0, 0.30, 0.12, step=0.02)

    with st.expander("💱 Comparar con cuotas y Gestión de Banca (Kelly / EV)"):
        q1, q2, q3 = st.columns(3)
        o_h = q1.number_input("Cuota 1", 0.0, 100.0, 0.0, step=0.05)
        o_d = q2.number_input("Cuota X", 0.0, 100.0, 0.0, step=0.05)
        o_a = q3.number_input("Cuota 2", 0.0, 100.0, 0.0, step=0.05)

    if st.button("🚀 CALCULAR PARTIDO", use_container_width=True):
        if home == away and not cross:
            st.error("⚠️ Selecciona equipos distintos.")
            st.stop()
        est_h, est_a = P_h["estado"], P_a["estado"]
        dh, da = P_h["df"], P_a["df"]
        fz_h, fz_a = fuerza.get(lig_h, 0), fuerza.get(lig_a, 0)

        cap_h, cap_a = ft.capacidades(dh), ft.capacidades(da)
        disp = {k: cap_h[k] and cap_a[k] for k in cap_h}
        basico = not any(disp.values())

        if cross:
            tasa_emp = float(((dh["FTR"] == "D").mean() + (da["FTR"] == "D").mean()) / 2.0)
            lam, mu, rho = md.tasas_cruzadas(P_h["dc"], P_a["dc"], home, away,
                                             fz_h * ELO_A_LOG, fz_a * ELO_A_LOG, neutral)
            p_elo = prob_cruzada(est_h.elo.rating(home) + fz_h, est_a.elo.rating(away) + fz_a,
                                 neutral, tasa_empate=tasa_emp)
        else:
            tasa_emp = float((dh["FTR"] == "D").mean())
            lam, mu = md.tasas_dc(P_h["dc"], home, away, neutral)
            rho = P_h["dc"]["rho"]
            p_elo = est_h.elo.prob_1x2(home, away, neutral, tasa_empate=tasa_emp)
        lam *= math.exp((att_h + def_a) / 100.0)
        mu *= math.exp((att_a + def_h) / 100.0)
        M = md.matriz(lam, mu, rho)
        p_dc = md.probs_1x2(M)

        p_lr = np.full(3, np.nan)
        if not cross:
            f = est_h.features(home, away, fecha)
            p_lr = md.predecir_lr(P_h["lr"], pd.DataFrame([f])[ft.FEATS])[0]

        escenario = any([att_h, att_a, def_h, def_a])
        if escenario:
            p_fin, usado = p_dc, "solo Dixon-Coles (escenario manual)"
        elif cross or neutral:
            p_fin = w_dc * p_dc + (1 - w_dc) * p_elo
            usado = "Dixon-Coles + Elo (el clasificador se entrenó solo con partidos de liga en cancha local)"
        else:
            p_fin, usado = w_dc * p_dc + (1 - w_dc) * p_lr, "Dixon-Coles + clasificador"

        # --- INCERTIDUMBRE ADAPTATIVA (Monte Carlo sin sesgo log-normal) ---
        n_h = est_h.n.get(home, 0)
        n_a = est_a.n.get(away, 0)

        N_REF = 38.0
        f_h = float(np.clip(math.sqrt(N_REF / max(n_h, 5)), 0.7, 2.0))
        f_a = float(np.clip(math.sqrt(N_REF / max(n_a, 5)), 0.7, 2.0))

        # Incertidumbre asimétrica: pondera 70% al equipo atacante y 30% al defensor
        sigma_lam = sigma * math.sqrt(0.7 * f_h**2 + 0.3 * f_a**2)
        sigma_mu  = sigma * math.sqrt(0.7 * f_a**2 + 0.3 * f_h**2)
        drift_lam = -0.5 * sigma_lam**2
        drift_mu  = -0.5 * sigma_mu**2

        rng = np.random.default_rng(0)
        s_1x2, s_o25, s_mk, mult_tot = [], [], [], []
        p_sec = p_elo if (cross or neutral) else p_lr
        for _ in range(500):
            m_l = math.exp(rng.normal(drift_lam, sigma_lam))
            m_v = math.exp(rng.normal(drift_mu, sigma_mu))
            pert_lam = lam * m_l
            pert_mu  = mu  * m_v
            mult_tot.append(0.5 * (m_l + m_v))
            Ms = md.matriz(pert_lam, pert_mu, rho)
            p_dc_s = md.probs_1x2(Ms)
            if escenario:
                s_1x2.append(p_dc_s)
            else:
                s_1x2.append(w_dc * p_dc_s + (1.0 - w_dc) * p_sec)
            mk_s = md.mercados_goles(Ms)
            s_mk.append(mk_s)
            s_o25.append(mk_s["Over 2.5"])

        s_1x2 = np.array(s_1x2)
        mult_tot = np.array(mult_tot)
        lo, hi = np.percentile(s_1x2, 10, axis=0), np.percentile(s_1x2, 90, axis=0)
        p25 = np.percentile(s_1x2, 25, axis=0)
        consenso = md.diagnosticar_consenso(p_dc, p_elo, None if (cross or escenario) else p_lr, n_h, n_a)

        st.subheader(f"{home} vs {away}")
        if basico:
            st.caption(f"Liga sin stats detalladas (solo 1X2 y goles) | {n_h} PJ vs {n_a} PJ · {consenso['estado']}")
        else:
            nota_incert = f" ⚠️ (incertidumbre alta: {n_h if n_h < 15 else n_a} PJ)" if min(n_h, n_a) < 15 else ""
            st.caption(f"Elo interno: {est_h.elo.rating(home):.0f} ({n_h} PJ) vs {est_a.elo.rating(away):.0f} ({n_a} PJ)"
                       + (" · cancha neutral" if neutral else "") + f" · Modelo: {usado}{nota_incert}")
            if consenso["es_trampa"]:
                st.warning(f"{consenso['estado']} — {consenso['mensaje']}")
            else:
                st.caption(f"{consenso['estado']} — {consenso['mensaje']}")

        if cross:
            st.warning("Partido entre ligas: el resultado depende del supuesto de fuerza relativa de cada liga "
                       "(ver el panel ⚖️). No hay datos propios para validarlo.")

        # ---- Ganador (todas las ligas) ----
        for col, nombre, i in zip(st.columns(3), ["Local", "Empate", "Visita"], range(3)):
            col.metric(nombre, f"{p_fin[i] * 100:.1f}%")
            col.caption(f"rango {lo[i] * 100:.0f}–{hi[i] * 100:.0f}%")
            col.progress(float(p_fin[i]))

        # ---- Goles (todas las ligas) ----
        st.markdown("#### ⚽ Goles")
        g1, g2, g3 = st.columns(3)
        g1.metric(home[:14], f"{lam:.2f}")
        g2.metric(away[:14], f"{mu:.2f}")
        g3.metric("Total esperado", f"{lam + mu:.2f}")
        mk = md.mercados_goles(M)

        if basico:
            solo_goles = ["Over 1.5", "Under 1.5", "Over 2.5", "Under 2.5", "Over 3.5", "Under 3.5"]
            st.dataframe(pd.DataFrame({"Probabilidad %": {k: mk[k] * 100 for k in solo_goles}}).round(1),
                         use_container_width=True)
        else:
            comp = pd.DataFrame({"Local %": [p_elo[0], p_dc[0], p_lr[0]], "Empate %": [p_elo[1], p_dc[1], p_lr[1]],
                                 "Visita %": [p_elo[2], p_dc[2], p_lr[2]]},
                                index=["Solo Elo", "Dixon-Coles", "Clasificador"]) * 100
            if min(o_h, o_d, o_a) > 1:
                p_mkt = db.desmargenar_cuotas([o_h, o_d, o_a])
                comp.loc["Mercado (sin margen)"] = p_mkt * 100
                comp.loc["Diferencia vs mercado (pp)"] = (p_fin - p_mkt) * 100
            st.markdown("#### 📊 Comparación de modelos")
            st.dataframe(comp.round(1).fillna("-"), use_container_width=True)

            if min(o_h, o_d, o_a) > 1:
                st.markdown("#### 💹 Valor Esperado (EV%) y Gestión de Banca (Kelly 1/4 Conservador)")
                filas_ev = {}
                for nom_m, p_m, p25_m, c_m in zip(["Local (1)", "Empate (X)", "Visita (2)"], p_fin, p25, [o_h, o_d, o_a]):
                    res_k = md.calcular_kelly_ev(p_m, p25_m, c_m)
                    filas_ev[nom_m] = {"Cuota": c_m, "Prob. Modelo %": round(p_m * 100, 1), **res_k}
                st.dataframe(pd.DataFrame(filas_ev).T, use_container_width=True)

            st.dataframe(pd.DataFrame({"Probabilidad %": {k: v * 100 for k, v in mk.items()}}).round(1),
                         use_container_width=True)
            st.caption(f"Over 2.5: {np.median(s_o25) * 100:.1f}% "
                       f"(rango {np.percentile(s_o25, 10) * 100:.0f}–{np.percentile(s_o25, 90) * 100:.0f}%)")
            doble = {"1X (local o empate)": p_fin[0] + p_fin[1], "X2 (visita o empate)": p_fin[1] + p_fin[2],
                     "12 (sin empate)": p_fin[0] + p_fin[2]}
            st.dataframe(pd.DataFrame({"Probabilidad %": {k: v * 100 for k, v in doble.items()}}).round(1),
                         use_container_width=True)

            asiaticos = md.mercados_asiaticos(M, p_fin=p_fin, home=home[:14], away=away[:14])
            with st.expander("🌏 Coberturas Asiáticas y Draw No Bet (Empate No Acción)"):
                st.caption("Draw No Bet (AH 0.0) y las líneas enteras/.25/.75 devuelven total o parcialmente el stake en caso de empate.")
                st.write("**Hándicap Asiático y Draw No Bet**")
                st.dataframe(asiaticos["handicap"].round(1), use_container_width=True)
                st.write("**Líneas de Goles Asiáticas**")
                st.dataframe(asiaticos["goles"].round(1), use_container_width=True)

            st.markdown("#### 🎯 Marcadores más probables")
            ts = md.top_marcadores(M, 5)
            for col, (sc, pr) in zip(st.columns(len(ts)), ts):
                col.metric(sc, f"{pr * 100:.1f}%")

        # ---- Funciones para aislar varianza con decaimiento temporal y sin duplicar partidos ----
        def _pesos_temp(fechas_s):
            f_ref = max(fecha, fechas_s.max()) if len(fechas_s) else fecha
            dias = (f_ref - fechas_s).dt.days.to_numpy(float).clip(min=0)
            return np.exp(-np.log(2) / half_life * dias)

        def disp_partido(df_h, df_a, eq_h, eq_a, c_h, c_a):
            m_h = (df_h["HomeTeam"] == eq_h) | (df_h["AwayTeam"] == eq_h)
            m_a = (df_a["HomeTeam"] == eq_a) | (df_a["AwayTeam"] == eq_a)
            if not cross:
                m = m_h | m_a
                sub = df_h.loc[m, [c_h, c_a, "Date"]].dropna()
                tot = sub[c_h] + sub[c_a]
                w = _pesos_temp(sub["Date"])
            else:
                sub_h = df_h.loc[m_h, [c_h, c_a, "Date"]].dropna()
                sub_a = df_a.loc[m_a, [c_h, c_a, "Date"]].dropna()
                tot = pd.concat([sub_h[c_h] + sub_h[c_a], sub_a[c_h] + sub_a[c_a]], ignore_index=True)
                w = np.concatenate([_pesos_temp(sub_h["Date"]), _pesos_temp(sub_a["Date"])])
            if len(tot) >= 15:
                return md.dispersion_nb(tot.reset_index(drop=True), weights=w, min_muestras=15)
            sub_l = df_h[[c_h, c_a, "Date"]].dropna()
            return md.dispersion_nb((sub_l[c_h] + sub_l[c_a]).reset_index(drop=True),
                                    weights=_pesos_temp(sub_l["Date"]))

        def disp_equipo(df_t, eq_t, c_h, c_a):
            m = (df_t["HomeTeam"] == eq_t) | (df_t["AwayTeam"] == eq_t)
            sub = df_t.loc[m, ["HomeTeam", c_h, c_a, "Date"]].dropna()
            s = pd.Series(np.where(sub["HomeTeam"] == eq_t, sub[c_h], sub[c_a]))
            if len(s) >= 10:
                return md.dispersion_nb(s, weights=_pesos_temp(sub["Date"]), min_muestras=10)
            sub_l = df_t[[c_h, c_a, "Date"]].dropna()
            tot_l = pd.concat([sub_l[c_h], sub_l[c_a]], ignore_index=True)
            w_l = np.tile(_pesos_temp(sub_l["Date"]), 2)
            return md.dispersion_nb(tot_l, weights=w_l)

        # ---- Córners y tiros a puerta ----
        filas, sec_cands = [], []
        for etiqueta, clave, favor, contra, ch, ca, lns in (
            ("Córners", "corners", "C", "CA", "HC", "AC", [8.5, 9.5, 10.5, 11.5]),
            ("Tiros a puerta", "tiros", "ST", "STA", "HST", "AST", [6.5, 7.5, 8.5, 9.5]),
        ):
            if not disp[clave]:
                continue
            eh, ea = esperados(est_h, est_a, home, away, favor, contra, dh, ch, da, ca)
            tot_esp = eh + ea

            r_tot = disp_partido(dh, da, home, away, ch, ca)
            po = md.prob_over_nb(tot_esp, r_tot, lns)
            mu_sim = tot_esp * mult_tot[::5]  # 100 muestras para cuantiles de conteos
            for l in (lns[0], lns[-1]):
                p_ov = po[l]
                sim_ov = [md.prob_over_nb(float(m_s), r_tot, [l])[l] for m_s in mu_sim]
                sec_cands.append({"familia": etiqueta, "nombre": f"Over {l} {etiqueta.lower()}",
                                  "prob": p_ov, "muestras": sim_ov})
                sec_cands.append({"familia": etiqueta, "nombre": f"Under {l} {etiqueta.lower()}",
                                  "prob": 1.0 - p_ov, "muestras": [1.0 - x for x in sim_ov]})

            fila = {"Mercado": etiqueta, "Esp. local": eh, "Esp. visita": ea, "Total": tot_esp}
            fila.update({f"Over {l} %": v * 100 for l, v in po.items()}) 
            filas.append(fila)
            
        if filas:
            st.markdown("#### 🚩 Córners y 🎯 tiros a puerta")
            st.caption("Mercados muy ruidosos: úsalos como orientación, no como señal fuerte.")
            df_ct = pd.DataFrame(filas).set_index("Mercado")
            formato_ct = {col: "{:.1f}%" for col in df_ct.columns if "Over" in col}
            formato_ct.update({"Esp. local": "{:.1f}", "Esp. visita": "{:.1f}", "Total": "{:.1f}"})
            st.dataframe(df_ct.style.format(formato_ct, na_rep="-"), use_container_width=True)

        # ---- Faltas y tarjetas por equipo (con factor árbitro bayesiano) ----
        if disp["faltas"] or disp["tarjetas"]:
            st.markdown("#### 🟨 Faltas y tarjetas por equipo")
            fn_farb = getattr(est_h, "factor_arbitro", None)
            f_arb = (fn_farb(None if arbitro_sel.startswith("Promedio") else arbitro_sel)
                     if callable(fn_farb) else {"faltas": 1.0, "amarillas": 1.0, "rojas": 1.0, "partidos": 0})
            if f_arb["partidos"] > 0:
                st.caption(f"Árbitro activo: **{arbitro_sel}** ({f_arb['partidos']} PJ en liga) · "
                           f"Ajuste bayesiano: Faltas ×{f_arb['faltas']:.2f}, Amarillas ×{f_arb['amarillas']:.2f}, Rojas ×{f_arb['rojas']:.2f}")
            else:
                st.caption("Usando rigor arbitral promedio de la liga (puedes elegir un árbitro específico en 🎛️ Escenario).")
            equipo = pd.DataFrame(index=[home[:18], away[:18], "Total"])
            tot_tablas, eq_tablas = {}, {}

            if disp["faltas"]:
                fh, fa = esperados(est_h, est_a, home, away, "F", "FA", dh, "HF", da, "AF")
                fh, fa = fh * f_arb["faltas"], fa * f_arb["faltas"]
                equipo["Faltas"] = [fh, fa, fh + fa]
                
                r_tot = disp_partido(dh, da, home, away, "HF", "AF")
                r_eq_h = disp_equipo(dh, home, "HF", "AF")
                r_eq_a = disp_equipo(da, away, "HF", "AF")
                
                tot_tablas["Faltas totales"] = {f"Over {l} %": v * 100 for l, v in
                                                md.prob_over_nb(fh + fa, r_tot, [20.5, 22.5, 24.5, 26.5]).items()}
                eq_tablas["Faltas"] = pd.DataFrame(
                    {f"Over {l} %": [md.prob_over_nb(fh, r_eq_h, [l])[l] * 100,
                                     md.prob_over_nb(fa, r_eq_a, [l])[l] * 100]
                     for l in [10.5, 12.5, 14.5]}, index=[home[:18], away[:18]])

            if disp["tarjetas"]:
                yh, ya = esperados(est_h, est_a, home, away, "Y", "YA", dh, "HY", da, "AY")
                yh, ya = yh * f_arb["amarillas"], ya * f_arb["amarillas"]
                equipo["Amarillas"] = [yh, ya, yh + ya]
                
                r_tot = disp_partido(dh, da, home, away, "HY", "AY")
                r_eq_h = disp_equipo(dh, home, "HY", "AY")
                r_eq_a = disp_equipo(da, away, "HY", "AY")
                
                tot_tablas["Amarillas totales"] = {f"Over {l} %": v * 100 for l, v in
                                                   md.prob_over_nb(yh + ya, r_tot, [2.5, 3.5, 4.5, 5.5]).items()}
                eq_tablas["Amarillas"] = pd.DataFrame(
                    {f"Over {l} %": [md.prob_over_nb(yh, r_eq_h, [l])[l] * 100,
                                     md.prob_over_nb(ya, r_eq_a, [l])[l] * 100]
                     for l in [0.5, 1.5, 2.5, 3.5]}, index=[home[:18], away[:18]])

            if disp["tarjetas"] and disp["rojas"]:
                rh, ra = esperados(est_h, est_a, home, away, "R", "RA", dh, "HR", da, "AR")
                rh = (0.5 * rh + 0.5 * dh["HR"].mean()) * f_arb["rojas"]
                ra = (0.5 * ra + 0.5 * da["AR"].mean()) * f_arb["rojas"]
                equipo["Rojas"] = [rh, ra, rh + ra]
                equipo["P(≥1 roja) %"] = [(1 - math.exp(-m)) * 100 for m in (rh, ra, rh + ra)]

            st.dataframe(equipo.round(2).fillna("-"), use_container_width=True)
            
            for nombre, probs in tot_tablas.items():
                st.dataframe(pd.DataFrame(probs, index=[nombre]).style.format("{:.1f}%", na_rep="-"), use_container_width=True)
            
            with st.expander("📋 Líneas por equipo"):
                for nombre, tabla in eq_tablas.items():
                    st.write(f"**{nombre}**")
                    st.dataframe(tabla.style.format("{:.1f}%", na_rep="-"), use_container_width=True)

        # ---- Resumen: Selecciones de mayor probabilidad y bajo riesgo (Top 2-3) ----
        top_riesgo = md.selecciones_bajo_riesgo(home, away, p_fin, s_1x2, mk, s_mk,
                                                secundarios=sec_cands, basico=basico, max_picks=3,
                                                penalizacion_consenso=consenso["penalizacion"])
        if top_riesgo:
            st.markdown("#### 🛡️ Resumen: Opciones más probables y de bajo riesgo")
            st.caption("Seleccionadas combinando alta probabilidad esperada, piso de seguridad en Monte Carlo (P10), "
                       "consenso entre modelos y diversificación entre categorías.")
            for col, pick in zip(st.columns(len(top_riesgo)), top_riesgo):
                col.metric(pick["Selección"], f"{pick['Probabilidad'] * 100:.1f}%")
                col.caption(f"{pick['Riesgo']} · Piso–Techo: {pick['P10'] * 100:.0f}–{pick['P90'] * 100:.0f}% "
                            f"· Cuota mín.: {pick['Cuota justa']:.2f}")
            df_riesgo = pd.DataFrame([
                {
                    "Selección": p["Selección"],
                    "Categoría": p["Categoría"],
                    "Probabilidad %": round(p["Probabilidad"] * 100, 1),
                    "Rango MC (10–90%)": f"{p['P10'] * 100:.0f}% – {p['P90'] * 100:.0f}%",
                    "Cuota mín. justa": p["Cuota justa"],
                    "Cuota con valor (EV≥2%)": round(1.02 / max(p["Probabilidad"], 1e-6), 2),
                    "Nivel de riesgo": p["Riesgo"],
                }
                for p in top_riesgo
            ]).set_index("Selección")
            st.dataframe(df_riesgo, use_container_width=True)

    # ---- Escáner de Jornada / Armador de Combinada Doble ----
    if not cross and len(eq_h) >= 4:
        with st.expander("🔍 Escáner de Jornada y Armador de Combinada Doble (2 partidos)"):
            st.caption("Selecciona hasta 5 partidos de la jornada para ordenarlos de menor a mayor riesgo "
                       "y obtener la Mejor Combinada Doble sin incluir partidos trampa.")
            n_cruces = min(4, len(eq_h) // 2)
            partidos_scan = []
            for idx_c in range(n_cruces):
                sc1, sc2 = st.columns(2)
                eh_i = sc1.selectbox(f"Local #{idx_c + 1}", eq_h, index=(idx_c * 2) % len(eq_h), key=f"sc_h_{idx_c}")
                ea_i = sc2.selectbox(f"Visita #{idx_c + 1}", eq_h, index=(idx_c * 2 + 1) % len(eq_h), key=f"sc_a_{idx_c}")
                if eh_i != ea_i:
                    partidos_scan.append((eh_i, ea_i))
            if st.button("📊 Escanear jornada y armar combinada doble", use_container_width=True):
                res_scan = md.escanear_jornada(partidos_scan, P_h, fecha, w_dc=w_dc, sigma=sigma, basico=basico)
                if not res_scan["ranking"].empty:
                    st.dataframe(res_scan["ranking"], use_container_width=True, hide_index=True)
                if res_scan["combinada"]:
                    cb = res_scan["combinada"]
                    st.success(f"🤝 **Mejor Combinada Doble:** {cb['Selección 1']}  ➕  {cb['Selección 2']}  \n"
                               f"Probabilidad conjunta: **{cb['Probabilidad conjunta %']:.1f}%** "
                               f"(Piso P10: {cb['Piso conjunto (P10) %']:.1f}%) · "
                               f"Cuota mínima justa: **{cb['Cuota justa combinada']:.2f}**")

# --------------------------------- RANKING ------------------------------------
with tab2:
    if P_main is None:
        st.info("Esta liga no tiene datos suficientes. Ve a **🗄️ Datos**.")
    else:
        est = P_main["estado"]
        with db.sesion() as con:
            eqs = data_bot.obtener_equipos_vigentes(con, LIGAS[liga_principal]["codigo"])
        rk = pd.DataFrame({"Equipo": eqs, "Elo": [round(est.elo.rating(t)) for t in eqs],
                           "Partidos": [est.n.get(t, 0) for t in eqs],
                           "Forma (últ. 5, pts)": [sum(est.pts.get(t, [])[-5:]) for t in eqs]})
        rk = rk.sort_values("Elo", ascending=False).reset_index(drop=True)
        rk.index += 1
        st.caption("El Elo solo es comparable dentro de la misma liga.")
        st.dataframe(rk, use_container_width=True)
        st.bar_chart(rk.head(15).set_index("Equipo")["Elo"])
        with db.sesion() as con:
            db.guardar_elo(con, LIGAS[liga_principal]["codigo"], est.elo.ratings, est.n)

# --------------------------------- BACKTEST -----------------------------------
with tab3:
    if P_main is None:
        st.info("Esta liga no tiene datos suficientes. Ve a **🗄️ Datos**.")
    else:
        df, feats = P_main["df"], P_main["feats"]
        st.write("En cada bloque el modelo se reentrena **solo con partidos anteriores** y predice el siguiente. "
                 "Se compara contra Elo, cuotas de mercado (si existen) y una línea base.")
        max_t = max(100, min(800, len(df) - 300))
        b1, b2 = st.columns(2)
        n_test = b1.slider("Partidos de prueba", 100, max_t, min(380, max_t), step=20)
        bloque = b2.slider("Reentrenar cada (días)", 7, 90, 21, step=7)
        clave = (liga_principal, half_life, w_dc, n_test, bloque, len(df))
        if st.button("▶️ Ejecutar backtest", use_container_width=True):
            barra = st.progress(0.0)
            st.session_state["bt"] = (clave, bt.correr_backtest(df, feats, half_life, w_dc, n_test, bloque,
                                                                progreso=barra.progress))
            barra.empty()
        if "bt" in st.session_state and st.session_state["bt"][0] == clave:
            R = st.session_state["bt"][1]
            y = R["y"]
            mask = ~np.isnan(R["mercado"]).any(axis=1)
            usar_m = mask.sum() >= 50
            sel_ = mask if usar_m else np.ones(len(y), bool)
            mods = {"Línea base": R["base"], "Solo Elo": R["elo"], "Dixon-Coles": R["dc"],
                    "Clasificador": R["lr"], "Ensemble": R["ens"]}
            if usar_m:
                mods["Cuotas de mercado"] = R["mercado"]
            tabla = pd.DataFrame({k: bt.metricas(v[sel_], y[sel_]) for k, v in mods.items()}).T
            st.caption(f"Partidos evaluados: {int(sel_.sum())}" + ("" if usar_m else " · sin cuotas para comparar"))
            st.dataframe(tabla.style.format({"Log Loss": "{:.4f}", "Brier": "{:.4f}", "Accuracy": "{:.1%}"}),
                         use_container_width=True)
            st.caption("Menor Log Loss y Brier = mejor. Si no superas a las cuotas, no tienes ventaja sobre el mercado.")
            cal = bt.tabla_calibracion(R["ens"][sel_], y[sel_])
            st.markdown("#### Curva de calibración (Ensemble)")
            st.line_chart(cal.set_index("Predicho")[["Observado", "Ideal"]])
            st.dataframe(cal[["Predicho", "Observado", "N"]].round(3), use_container_width=True)

            if usar_m and "cuotas" in R:
                st.markdown("#### 💰 Simulación Financiera (ROI / Yield, Unidades y Drawdown)")
                st.caption("Simula apostar 1 unidad (o Kelly 1/4) cuando la probabilidad del modelo supera el umbral "
                           "y tiene Valor Esperado positivo (EV = P × Cuota - 1) frente a las cuotas históricas.")
                f1, f2 = st.columns(2)
                ev_min_pct = f1.slider("EV mínimo para apostar (%)", 0.0, 15.0, 2.0, step=0.5)
                prob_min_pct = f2.slider("Probabilidad mínima del modelo (%)", 40.0, 80.0, 55.0, step=5.0)
                res_fin, curvas_fin = {}, {}
                for nom_mod, p_mod in (("Ensemble", R["ens"]), ("Dixon-Coles", R["dc"]), ("Clasificador", R["lr"])):
                    sim = bt.simular_apuestas(p_mod[sel_], R["cuotas"][sel_], y[sel_],
                                              ev_min=ev_min_pct / 100.0, prob_min=prob_min_pct / 100.0)
                    res_fin[nom_mod] = sim["resumen"]
                    curvas_fin[nom_mod] = sim["curva"]
                st.dataframe(pd.DataFrame(res_fin).T, use_container_width=True)
                max_len = max(len(c) for c in curvas_fin.values())
                df_curvas = pd.DataFrame({
                    k: np.pad(c, (0, max_len - len(c)), mode="edge") for k, c in curvas_fin.items()
                })
                st.markdown("##### 📈 Evolución de Ganancias Acumuladas (Unidades)")
                st.line_chart(df_curvas)

# -------------------------------- METODOLOGÍA ---------------------------------
with tab5:
    st.markdown(f"""
**Fuente única:** football-data.co.uk, sin APIs ni claves. Premier, Championship, Bundesliga, La Liga y Serie A
(CSV por temporada) y Brasil, Argentina, Japón, China y EE. UU. (XLSX con todas las temporadas). No hay selecciones.

**Dixon-Coles:** ataque y defensa por equipo, nivel de goles de la liga, ventaja de local, `ρ` para marcadores bajos y
decaimiento temporal (vida media {half_life} días). De su matriz salen 1X2, Over/Under, BTTS y marcadores.

**Elo + clasificador logístico:** features sin fuga de información (EWMA local/visita, forma, descanso).
Ensemble: {w_dc:.0%} Dixon-Coles y {1 - w_dc:.0%} del otro modelo.

**Partido europeo entre ligas:** cada equipo conserva sus parámetros de su liga y se suma un ajuste por la fuerza
relativa de la liga (puntos Elo). Ese ajuste es un **supuesto tuyo**, no un dato: los archivos no tienen partidos
entre ligas. Úsalo como escenario, no como pronóstico validado.

**Límites:** sin alineaciones, lesiones ni xG real, lo realista es acercarse a las cuotas de cierre, no batirlas. Brasil,
Argentina, Japón, China y EE. UU. no traen córners ni tiros en estos archivos.
""")