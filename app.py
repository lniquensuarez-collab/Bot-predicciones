# ==============================================================================
# 📱 ULTIMATE AI V14 - Ligas de clubes (football-data.co.uk) + modo europeo
# Ejecutar:  streamlit run app.py
# ==============================================================================
import math
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from engine import backtest as bt
from engine import data_bot, db
from engine import features as ft
from engine import models as md
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
def preparar_liga(codigo, half_life):
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

# Bases creadas con una versión anterior no tienen faltas/tarjetas: se completan una sola vez por sesión.
if not st.session_state.get("refresco_faltas"):
    st.session_state["refresco_faltas"] = True
    with db.sesion() as con:
        r = con.execute("SELECT COUNT(*), COUNT(faltas_l) FROM partidos WHERE competicion IN ('E0','E1','D1','SP1','I1')").fetchone()
        if r[0] > 0 and r[1] == 0:
            pend = [n for n, c in LIGAS.items() if c["fuente"] == "csv"]
            with st.spinner("Actualizando datos para incluir faltas y tarjetas..."):
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

    with st.expander("🎛️ Escenario (ajustes manuales)"):
        st.caption("Si modificas algo se usa solo Dixon-Coles (Elo y clasificador no admiten ajustes manuales).")
        s1, s2 = st.columns(2)
        att_h = s1.slider("Ataque local %", -40, 40, 0)
        att_a = s2.slider("Ataque visita %", -40, 40, 0)
        def_h = s1.slider("Debilidad defensa local %", -40, 40, 0)
        def_a = s2.slider("Debilidad defensa visita %", -40, 40, 0)
        sigma = st.slider("Incertidumbre del modelo (σ)", 0.0, 0.30, 0.12, step=0.02)

    with st.expander("💱 Comparar con cuotas (opcional)"):
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

        # ¿Qué permite el archivo de cada liga? Si no hay ninguna estadística -> modo básico.
        cap_h, cap_a = ft.capacidades(dh), ft.capacidades(da)
        disp = {k: cap_h[k] and cap_a[k] for k in cap_h}
        basico = not any(disp.values())

        if cross:
            lam, mu, rho = md.tasas_cruzadas(P_h["dc"], P_a["dc"], home, away,
                                             fz_h * ELO_A_LOG, fz_a * ELO_A_LOG, neutral)
            p_elo = prob_cruzada(est_h.elo.rating(home) + fz_h, est_a.elo.rating(away) + fz_a, neutral)
        else:
            lam, mu = md.tasas_dc(P_h["dc"], home, away, neutral)
            rho = P_h["dc"]["rho"]
            p_elo = est_h.elo.prob_1x2(home, away, neutral)
        lam *= (1 + att_h / 100) * (1 + def_a / 100)
        mu *= (1 + att_a / 100) * (1 + def_h / 100)
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

        # --- Incertidumbre adaptativa según historial de partidos ---
        n_h = est_h.n.get(home, 0)
        n_a = est_a.n.get(away, 0)
        
        # Factor base: 1.0 si tiene ~38 partidos (1 temporada). Máx 2.0x si tiene pocos, mín 0.7x si tiene muchos.
        N_REF = 38.0
        f_h = float(np.clip(math.sqrt(N_REF / max(n_h, 5)), 0.7, 2.0))
        f_a = float(np.clip(math.sqrt(N_REF / max(n_a, 5)), 0.7, 2.0))
        
        # Cada tasa combina ataque de uno y defensa del otro
        sigma_lam = sigma * math.sqrt((f_h**2 + f_a**2) / 2.0)
        sigma_mu  = sigma * math.sqrt((f_a**2 + f_h**2) / 2.0)

# --- INCERTIDUMBRE ADAPTATIVA (Monte Carlo) ---
        n_h = est_h.n.get(home, 0)
        n_a = est_a.n.get(away, 0)
        
        # Factor base: 1.0 si tiene ~38 partidos. Máx 2.0x si tiene pocos, mín 0.7x si tiene muchos.
        N_REF = 38.0
        f_h = float(np.clip(math.sqrt(N_REF / max(n_h, 5)), 0.7, 2.0))
        f_a = float(np.clip(math.sqrt(N_REF / max(n_a, 5)), 0.7, 2.0))
        
        # Cada tasa combina el ataque de uno y la defensa del otro
        sigma_lam = sigma * math.sqrt((f_h**2 + f_a**2) / 2.0)
        sigma_mu  = sigma * math.sqrt((f_a**2 + f_h**2) / 2.0)

        rng = np.random.default_rng(0)
        s_1x2, s_o25 = [], []
        for _ in range(500):
            pert_lam = lam * math.exp(rng.normal(0, sigma_lam))
            pert_mu  = mu  * math.exp(rng.normal(0, sigma_mu))
            Ms = md.matriz(pert_lam, pert_mu, rho)
            s_1x2.append(md.probs_1x2(Ms))
            s_o25.append(md.mercados_goles(Ms)["Over 2.5"])
        
        s_1x2 = np.array(s_1x2)
        lo, hi = np.percentile(s_1x2, 10, axis=0), np.percentile(s_1x2, 90, axis=0)

        st.subheader(f"{home} vs {away}")
        if basico:
            st.caption(f"Liga sin stats detalladas (solo 1X2 y goles) | {n_h} PJ vs {n_a} PJ")
        else:
            nota_incert = f" ⚠️ (incertidumbre alta: {n_h if n_h < 15 else n_a} PJ)" if min(n_h, n_a) < 15 else ""
            st.caption(f"Elo interno: {est_h.elo.rating(home):.0f} ({n_h} PJ) vs {est_a.elo.rating(away):.0f} ({n_a} PJ)"
                       + (" · cancha neutral" if neutral else "") + f" · Modelo: {usado}{nota_incert}")
                       
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
                inv = 1 / np.array([o_h, o_d, o_a])
                comp.loc["Mercado (sin margen)"] = inv / inv.sum() * 100
                comp.loc["Diferencia vs mercado (pp)"] = p_fin * 100 - inv / inv.sum() * 100
            st.markdown("#### 📊 Comparación de modelos")
            st.dataframe(comp.round(1).fillna("-"), use_container_width=True)

            st.dataframe(pd.DataFrame({"Probabilidad %": {k: v * 100 for k, v in mk.items()}}).round(1),
                         use_container_width=True)
            st.caption(f"Over 2.5: {np.median(s_o25) * 100:.1f}% "
                       f"(rango {np.percentile(s_o25, 10) * 100:.0f}–{np.percentile(s_o25, 90) * 100:.0f}%)")
            doble = {"1X (local o empate)": p_fin[0] + p_fin[1], "X2 (visita o empate)": p_fin[1] + p_fin[2],
                     "12 (sin empate)": p_fin[0] + p_fin[2]}
            st.dataframe(pd.DataFrame({"Probabilidad %": {k: v * 100 for k, v in doble.items()}}).round(1),
                         use_container_width=True)

            st.markdown("#### 🎯 Marcadores más probables")
            ts = md.top_marcadores(M, 5)
            for col, (sc, pr) in zip(st.columns(len(ts)), ts):
                col.metric(sc, f"{pr * 100:.1f}%")

# ---- Córners y tiros a puerta ----
# ---- Funciones para aislar varianza (evitan el sesgo de liga) ----
            def disp_partido(df_h, df_a, eq_h, eq_a, c_h, c_a):
                m_h = (df_h["HomeTeam"] == eq_h) | (df_h["AwayTeam"] == eq_h)
                m_a = (df_a["HomeTeam"] == eq_a) | (df_a["AwayTeam"] == eq_a)
                tot = pd.concat([df_h.loc[m_h, c_h] + df_h.loc[m_h, c_a], 
                                 df_a.loc[m_a, c_h] + df_a.loc[m_a, c_a]]).dropna()
                if len(tot) >= 15:
                    mu, var = tot.mean(), tot.var()
                    return mu * mu / (var - mu) if var > mu else 1e6
                return md.dispersion_nb(df_h[c_h] + df_h[c_a]) * 1.5

            def disp_equipo(df_t, eq_t, c_h, c_a):
                m = (df_t["HomeTeam"] == eq_t) | (df_t["AwayTeam"] == eq_t)
                s = np.where(df_t.loc[m, "HomeTeam"] == eq_t, df_t.loc[m, c_h], df_t.loc[m, c_a])
                tot = pd.Series(s).dropna()
                if len(tot) >= 10:
                    mu, var = tot.mean(), tot.var()
                    return mu * mu / (var - mu) if var > mu else 1e6
                return md.dispersion_nb(pd.concat([df_t[c_h], df_t[c_a]])) * 1.5

            # ---- Córners y tiros a puerta ----
            filas = []
            for etiqueta, clave, favor, contra, ch, ca, lns in (
                ("Córners", "corners", "C", "CA", "HC", "AC", [8.5, 9.5, 10.5, 11.5]),
                ("Tiros a puerta", "tiros", "ST", "STA", "HST", "AST", [6.5, 7.5, 8.5, 9.5]),
            ):
                if not disp[clave]:
                    continue
                eh, ea = esperados(est_h, est_a, home, away, favor, contra, dh, ch, da, ca)
                
                # Usar varianza filtrada solo para estos equipos
                r_tot = disp_partido(dh, da, home, away, ch, ca)
                po = md.prob_over_nb(eh + ea, r_tot, lns)
                
                fila = {"Mercado": etiqueta, "Esp. local": eh, "Esp. visita": ea, "Total": eh + ea}
                fila.update({f"Over {l}": v * 100 for l, v in po.items()}) 
                filas.append(fila)
                
            if filas:
                st.markdown("#### 🚩 Córners y 🎯 tiros a puerta")
                st.caption("Mercados muy ruidosos: úsalos como orientación, no como señal fuerte.")
                df_ct = pd.DataFrame(filas).set_index("Mercado")
                formato_ct = {col: "{:.1f}%" for col in df_ct.columns if "Over" in col}
                formato_ct.update({"Esp. local": "{:.1f}", "Esp. visita": "{:.1f}", "Total": "{:.1f}"})
                st.dataframe(df_ct.style.format(formato_ct, na_rep="-"), use_container_width=True)

            # ---- Faltas y tarjetas por equipo ----
            if disp["faltas"] or disp["tarjetas"]:
                st.markdown("#### 🟨 Faltas y tarjetas por equipo")
                st.caption("Orientativo: el árbitro influye mucho en las tarjetas y este modelo no lo conoce.")
                equipo = pd.DataFrame(index=[home[:18], away[:18], "Total"])
                tot_tablas, eq_tablas = {}, {}

                if disp["faltas"]:
                    fh, fa = esperados(est_h, est_a, home, away, "F", "FA", dh, "HF", da, "AF")
                    equipo["Faltas"] = [fh, fa, fh + fa]
                    
                    r_tot = disp_partido(dh, da, home, away, "HF", "AF")
                    r_eq_h = disp_equipo(dh, home, "HF", "AF")
                    r_eq_a = disp_equipo(da, away, "HF", "AF")
                    
                    tot_tablas["Faltas totales"] = {f"Over {l} ": v * 100 for l, v in
                                                    md.prob_over_nb(fh + fa, r_tot, [20.5, 22.5, 24.5, 26.5]).items()}
                    eq_tablas["Faltas"] = pd.DataFrame(
                        {f"Over {l}": [md.prob_over_nb(fh, r_eq_h, [l])[l] * 100,
                                       md.prob_over_nb(fa, r_eq_a, [l])[l] * 100]
                         for l in [10.5, 12.5, 14.5]}, index=[home[:18], away[:18]])

                if disp["tarjetas"]:
                    yh, ya = esperados(est_h, est_a, home, away, "Y", "YA", dh, "HY", da, "AY")
                    equipo["Amarillas"] = [yh, ya, yh + ya]
                    
                    r_tot = disp_partido(dh, da, home, away, "HY", "AY")
                    r_eq_h = disp_equipo(dh, home, "HY", "AY")
                    r_eq_a = disp_equipo(da, away, "HY", "AY")
                    
                    tot_tablas["Amarillas totales"] = {f"Over {l}": v * 100 for l, v in
                                                       md.prob_over_nb(yh + ya, r_tot, [2.5, 3.5, 4.5, 5.5]).items()}
                    eq_tablas["Amarillas"] = pd.DataFrame(
                        {f"Over {l}": [md.prob_over_nb(yh, r_eq_h, [l])[l] * 100,
                                       md.prob_over_nb(ya, r_eq_a, [l])[l] * 100]
                         for l in [0.5, 1.5, 2.5, 3.5]}, index=[home[:18], away[:18]])
                if disp["tarjetas"] and disp["rojas"]:
                    rh, ra = esperados(est_h, est_a, home, away, "R", "RA", dh, "HR", da, "AR")
                    # Las rojas son raras: se suaviza 50 % hacia la media de la liga para no sobreajustar
                    rh = 0.5 * rh + 0.5 * dh["HR"].mean()
                    ra = 0.5 * ra + 0.5 * da["AR"].mean()
                    equipo["Rojas"] = [rh, ra, rh + ra]
                    equipo["P(≥1 roja) %"] = [(1 - math.exp(-m)) * 100 for m in (rh, ra, rh + ra)]

                st.dataframe(equipo.round(2).fillna("-"), use_container_width=True)
                
                for nombre, probs in tot_tablas.items():
                    st.dataframe(pd.DataFrame(probs, index=[nombre]).style.format("{:.1f}%", na_rep="-"), use_container_width=True)
                
                with st.expander("📋 Líneas por equipo"):
                    for nombre, tabla in eq_tablas.items():
                        st.write(f"**{nombre}**")
                        st.dataframe(tabla.style.format("{:.1f}%", na_rep="-"), use_container_width=True)

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

#Mejora Visual Opcional en el Subheader de la App
st.subheader(f"{home} vs {away}")
        st.subheader(f"{home} vs {away}")
        if basico:
            st.caption(f"Liga sin stats detalladas (solo 1X2 y goles) | {n_h} PJ vs {n_a} PJ")
        else:
            nota_incert = f" ⚠️ (incertidumbre alta: {n_h if n_h < 15 else n_a} PJ)" if min(n_h, n_a) < 15 else ""
            st.caption(f"Elo interno: {est_h.elo.rating(home):.0f} ({n_h} PJ) vs {est_a.elo.rating(away):.0f} ({n_a} PJ)"
                       + (" · cancha neutral" if neutral else "") + f" · Modelo: {usado}{nota_incert}")
                       
        if cross:
            st.warning("Partido entre ligas: el resultado depende del supuesto de fuerza relativa de cada liga "
                       "(ver el panel ⚖️). No hay datos propios para validarlo.")
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