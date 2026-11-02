"""Modelos: Dixon-Coles (con gradiente analítico), clasificador logístico y mercados derivados."""
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import nbinom, poisson
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATS

MAX_GOLES = 8
ETIQ = {"H": 0, "D": 1, "A": 2}


# ----------------------------- Dixon-Coles -----------------------------------
def preparar_dc(df, fecha_ref, half_life, reg=0.5, peso_xg=0.35):
    """Función objetivo (valor + gradiente analítico). Parámetros: ataque(n), defensa(n), ventaja local,
    rho (marcadores bajos) y c (intercepto = nivel de goles de la liga).
    Si existen xGH/xGA post-partido, suaviza los goles observados (65% goles + 35% xG) para reducir el factor suerte."""
    dias = (fecha_ref - df["Date"]).dt.days.to_numpy(float).clip(min=0)
    w = np.exp(-np.log(2) / half_life * dias)
    d = df[w > 1e-4]
    w = w[w > 1e-4]

    equipos = sorted(set(d["HomeTeam"]) | set(d["AwayTeam"]))
    idx = {t: i for i, t in enumerate(equipos)}
    n = len(equipos)
    hi, ai = d["HomeTeam"].map(idx).to_numpy(), d["AwayTeam"].map(idx).to_numpy()
    x_gol, y_gol = d["FTHG"].to_numpy(float), d["FTAG"].to_numpy(float)
    m00, m01 = (x_gol == 0) & (y_gol == 0), (x_gol == 0) & (y_gol == 1)
    m10, m11 = (x_gol == 1) & (y_gol == 0), (x_gol == 1) & (y_gol == 1)

    if peso_xg > 0 and "xGH" in d.columns and "xGA" in d.columns:
        xgh = pd.to_numeric(d["xGH"], errors="coerce").to_numpy(float)
        xga = pd.to_numeric(d["xGA"], errors="coerce").to_numpy(float)
        ok_xg = ~np.isnan(xgh) & ~np.isnan(xga)
        x = np.where(ok_xg, (1.0 - peso_xg) * x_gol + peso_xg * xgh, x_gol)
        y = np.where(ok_xg, (1.0 - peso_xg) * y_gol + peso_xg * xga, y_gol)
    else:
        x, y = x_gol, y_gol

    lgx, lgy = gammaln(x + 1), gammaln(y + 1)

    def fun(p):
        att = p[:n] - p[:n].mean()
        dfn = p[n:2 * n] - p[n:2 * n].mean()
        ha, rho, c = p[2 * n], p[2 * n + 1], p[2 * n + 2]
        lam = np.exp(c + ha + att[hi] - dfn[ai])
        mu = np.exp(c + att[ai] - dfn[hi])
        tau = np.ones_like(lam)
        tau[m00] = 1 - lam[m00] * mu[m00] * rho
        tau[m01] = 1 + lam[m01] * rho
        tau[m10] = 1 + mu[m10] * rho
        tau[m11] = 1 - rho
        tau = np.clip(tau, 1e-6, None)
        ll = np.log(tau) + x * np.log(lam) - lam - lgx + y * np.log(mu) - mu - lgy
        val = -(w * ll).sum() + reg * (att @ att + dfn @ dfn)

        tl, tm, tr = np.zeros_like(lam), np.zeros_like(lam), np.zeros_like(lam)
        tl[m00] = -lam[m00] * mu[m00] * rho / tau[m00]
        tm[m00] = tl[m00]
        tr[m00] = -lam[m00] * mu[m00] / tau[m00]
        tl[m01] = lam[m01] * rho / tau[m01]
        tr[m01] = lam[m01] / tau[m01]
        tm[m10] = mu[m10] * rho / tau[m10]
        tr[m10] = mu[m10] / tau[m10]
        tr[m11] = -1 / tau[m11]
        gl, gm = w * (x - lam + tl), w * (y - mu + tm)
        g_att = -(np.bincount(hi, gl, n) + np.bincount(ai, gm, n)) + 2 * reg * att
        g_dfn = (np.bincount(ai, gl, n) + np.bincount(hi, gm, n)) + 2 * reg * dfn
        g_att -= g_att.mean()
        g_dfn -= g_dfn.mean()
        grad = np.concatenate([g_att, g_dfn, [-gl.sum(), -(w * tr).sum(), -(gl.sum() + gm.sum())]])
        return val, grad

    p0 = np.zeros(2 * n + 3)
    p0[2 * n] = 0.2
    p0[2 * n + 2] = math.log(max(0.3, float(np.average((x + y) / 2, weights=w))))
    bounds = [(None, None)] * (2 * n + 1) + [(-0.3, 0.3), (None, None)]
    return fun, p0, bounds, equipos


def ajustar_dc(df, fecha_ref, half_life, reg=0.5, p0_prev=None):
    fun, p0, bounds, equipos = preparar_dc(df, fecha_ref, half_life, reg)
    n = len(equipos)
    if p0_prev is not None:
        p0[:n] = [p0_prev["att"].get(t, 0.0) for t in equipos]
        p0[n:2 * n] = [p0_prev["dfn"].get(t, 0.0) for t in equipos]
        p0[2 * n] = float(p0_prev.get("ha", p0[2 * n]))
        p0[2 * n + 1] = float(np.clip(p0_prev.get("rho", 0.0), -0.29, 0.29))
        p0[2 * n + 2] = float(p0_prev.get("c", p0[2 * n + 2]))
    p = minimize(fun, p0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 400}).x
    att, dfn = p[:n] - p[:n].mean(), p[n:2 * n] - p[n:2 * n].mean()
    return {"att": dict(zip(equipos, att)), "dfn": dict(zip(equipos, dfn)),
            "ha": float(p[2 * n]), "rho": float(p[2 * n + 1]), "c": float(p[2 * n + 2])}


def tasas_dc(m, h, a, neutral=False):
    """Equipos sin historial usan 0 = media de la liga. En cancha neutral no hay ventaja de local."""
    ha = 0.0 if neutral else m["ha"]
    lam = math.exp(m["c"] + ha + m["att"].get(h, 0.0) - m["dfn"].get(a, 0.0))
    mu = math.exp(m["c"] + m["att"].get(a, 0.0) - m["dfn"].get(h, 0.0))
    return lam, mu


def tasas_cruzadas(m_h, m_a, h, a, delta_h, delta_a, neutral=False):
    """Partido entre equipos de ligas distintas (Champions/Europa/Conference).
    delta_* = fuerza de la liga en escala log (puntos Elo * ELO_A_LOG), sumada al ataque y a la defensa.
    Si ambos equipos fueran de la misma liga, los delta se cancelan y da lo mismo que tasas_dc."""
    c = (m_h["c"] + m_a["c"]) / 2
    ha = 0.0 if neutral else m_h["ha"]
    att_h, dfn_h = m_h["att"].get(h, 0.0) + delta_h, m_h["dfn"].get(h, 0.0) + delta_h
    att_a, dfn_a = m_a["att"].get(a, 0.0) + delta_a, m_a["dfn"].get(a, 0.0) + delta_a
    return math.exp(c + ha + att_h - dfn_a), math.exp(c + att_a - dfn_h), (m_h["rho"] + m_a["rho"]) / 2


def matriz(lam, mu, rho, k=MAX_GOLES):
    g = np.arange(k + 1)
    M = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    M[0, 0] *= max(1e-6, 1.0 - lam * mu * rho)
    M[0, 1] *= max(1e-6, 1.0 + lam * rho)
    M[1, 0] *= max(1e-6, 1.0 + mu * rho)
    M[1, 1] *= max(1e-6, 1.0 - rho)
    M = np.clip(M, 0.0, None)
    return M / M.sum()


def probs_1x2(M):
    return np.array([np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()])


def mercados_goles(M):
    G = np.add.outer(np.arange(M.shape[0]), np.arange(M.shape[1]))
    btts = M[1:, 1:].sum()
    return {"Over 1.5": M[G > 1].sum(), "Under 1.5": M[G <= 1].sum(),
            "Over 2.5": M[G > 2].sum(), "Under 2.5": M[G <= 2].sum(),
            "Over 3.5": M[G > 3].sum(), "Under 3.5": M[G <= 3].sum(),
            "Ambos marcan (BTTS)": btts, "No BTTS": 1 - btts}


def top_marcadores(M, n=5):
    flat = [(f"{i} - {j}", M[i, j]) for i in range(M.shape[0]) for j in range(M.shape[1])]
    return sorted(flat, key=lambda t: t[1], reverse=True)[:n]


# ------------------------ Clasificador y conteos -----------------------------
def entrenar_lr(X, y, sample_weight=None):
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=1000)).fit(X, y, logisticregression__sample_weight=sample_weight)

def predecir_lr(pipe, X):
    out = np.zeros((len(X), 3))
    out[:, pipe.classes_] = pipe.predict_proba(X)
    return out

def entrenar_produccion(df, feats, half_life):
    fecha_ref = df["Date"].max() + pd.Timedelta(days=1)
    dc = ajustar_dc(df, fecha_ref, half_life)
    y = df["FTR"].map(ETIQ).to_numpy()
    ok = (feats["n_prev"] >= 5).to_numpy()
    
    # Decaimiento exponencial idéntico a Dixon-Coles
    dias = (fecha_ref - df.loc[ok, "Date"]).dt.days.to_numpy(float).clip(min=0)
    w = np.exp(-np.log(2) / half_life * dias)
    
    return dc, entrenar_lr(feats.loc[ok, FEATS], y[ok], sample_weight=w)

def dispersion_nb(total, weights=None, min_muestras=15):
    s = pd.Series(total).dropna()
    if len(s) < min_muestras:
        return 1e6
    vals = s.to_numpy(float)
    if weights is not None:
        w = np.asarray(weights, dtype=float)[s.index] if hasattr(s, "index") and len(weights) == len(total) else np.asarray(weights, dtype=float)
        w = w / w.sum()
        mu = float(np.sum(w * vals))
        denom = 1.0 - float(np.sum(w ** 2))
        var = float(np.sum(w * (vals - mu) ** 2)) / max(denom, 1e-6)
    else:
        mu, var = float(vals.mean()), float(vals.var(ddof=1))
    return mu * mu / (var - mu) if var > mu else 1e6


def prob_over_nb(mu, r, lineas):
    p = r / (r + mu)
    return {l: float(1 - nbinom.cdf(math.floor(l), r, p)) for l in lineas}


def _resolver_linea_asiatica(M, diff_mat, linea):
    """Evalúa una línea asiática (entera, .5, .25 o .75) sobre una matriz de diferencias diff_mat."""
    frac = round((linea % 1.0 + 1.0) % 1.0, 2)
    if frac in (0.25, 0.75):
        g1, n1, p1 = _resolver_linea_asiatica(M, diff_mat, linea - 0.25)[:3]
        g2, n2, p2 = _resolver_linea_asiatica(M, diff_mat, linea + 0.25)[:3]
        gana, nula, pierde = 0.5 * (g1 + g2), 0.5 * (n1 + n2), 0.5 * (p1 + p2)
    else:
        s = diff_mat + linea
        gana = float(M[s > 1e-9].sum())
        nula = float(M[np.abs(s) <= 1e-9].sum())
        pierde = float(M[s < -1e-9].sum())
    exito_ef = gana / max(gana + pierde, 1e-6)
    no_pierde = gana + nula
    cuota = 1.0 + pierde / max(gana, 1e-6)
    return gana, nula, pierde, exito_ef, no_pierde, cuota


def mercados_asiaticos(M, p_fin=None, home="Local", away="Visita"):
    """Calcula Draw No Bet (Empate No Acción), Hándicaps Asiáticos de resultado y Líneas de Goles Asiáticas."""
    p = probs_1x2(M) if p_fin is None else np.asarray(p_fin, dtype=float)
    idx = np.arange(M.shape[0])
    D = np.subtract.outer(idx, idx)  # goles_local - goles_visita
    G = np.add.outer(idx, idx)       # goles totales

    filas_ah = []
    # DNB (AH 0.0) usando probabilidades finales del ensemble
    for eq, p_g, p_p in ((f"{home} DNB (AH 0.0)", p[0], p[2]), (f"{away} DNB (AH 0.0)", p[2], p[0])):
        exito = float(p_g / max(p_g + p_p, 1e-6))
        filas_ah.append({
            "Mercado": eq,
            "Gana %": float(p_g) * 100,
            "Reembolso %": float(p[1]) * 100,
            "No pierde %": float(p_g + p[1]) * 100,
            "Éxito sin empate %": exito * 100,
            "Cuota justa": round(1.0 / max(exito, 1e-6), 2),
        })

    # Hándicaps asiáticos de resultado desde la matriz M
    for eq, mat_d, lns in ((home, D, [-1.0, -0.75, -0.5, -0.25, 0.25, 0.5]),
                           (away, -D, [-0.5, -0.25, 0.25, 0.5, 0.75, 1.0])):
        for h in lns:
            gana, nula, _, exito_ef, no_pierde, cuota = _resolver_linea_asiatica(M, mat_d, h)
            signo = f"+{h:g}" if h > 0 else f"{h:g}"
            filas_ah.append({
                "Mercado": f"{eq} AH {signo}",
                "Gana %": gana * 100,
                "Reembolso %": nula * 100,
                "No pierde %": no_pierde * 100,
                "Éxito sin empate %": exito_ef * 100,
                "Cuota justa": round(min(cuota, 99.0), 2),
            })

    filas_goles = []
    for tipo, mat_g, lns in (("Over", G, [1.75, 2.0, 2.25, 2.75, 3.0]),
                             ("Under", -G, [2.0, 2.25, 2.75, 3.0, 3.25])):
        for l in lns:
            linea_eval = -l if tipo == "Over" else l
            gana, nula, _, exito_ef, no_pierde, cuota = _resolver_linea_asiatica(M, mat_g, linea_eval)
            filas_goles.append({
                "Mercado": f"{tipo} {l:g} goles (asiático)",
                "Gana %": gana * 100,
                "Reembolso %": nula * 100,
                "No pierde %": no_pierde * 100,
                "Éxito efectivo %": exito_ef * 100,
                "Cuota justa": round(min(cuota, 99.0), 2),
            })

    return {"handicap": pd.DataFrame(filas_ah).set_index("Mercado"),
            "goles": pd.DataFrame(filas_goles).set_index("Mercado")}


def calcular_kelly_ev(prob_media, prob_p25, cuota, fraccion=0.25, max_stake=5.0):
    """Calcula Valor Esperado (EV%) y Stake sugerido (% de banca) con Kelly Fraccionario Conservador."""
    p = float(prob_media)
    p25 = float(prob_p25)
    o = float(cuota)
    if o <= 1.0 or np.isnan(p) or np.isnan(o):
        return {"EV %": 0.0, "EV conservador %": 0.0, "Stake Kelly 1/4 %": 0.0, "Señal": "—"}
    ev = (p * o - 1.0) * 100.0
    ev_c = (p25 * o - 1.0) * 100.0
    # Probabilidad prudente (50% estimación central + 50% percentil 25 de Monte Carlo)
    p_prud = 0.5 * p + 0.5 * p25
    kelly_puro = max(0.0, (p_prud * o - 1.0) / max(o - 1.0, 1e-6))
    stake = min(float(max_stake), float(fraccion) * kelly_puro * 100.0)
    if ev >= 5.0 and ev_c > 0.0:
        senal = "🔥 Valor alto"
    elif ev >= 1.5 and stake > 0.1:
        senal = "✅ Valor positivo"
    elif ev > -1.5:
        senal = "⚪ Cuota justa (sin ventaja)"
    else:
        senal = "❌ Sin valor (EV negativo)"
    return {"EV %": round(ev, 2), "EV conservador %": round(ev_c, 2),
            "Stake Kelly 1/4 %": round(stake, 2), "Señal": senal}


def diagnosticar_consenso(p_dc, p_elo, p_lr, n_h, n_a):
    """Evalúa el acuerdo entre Dixon-Coles, Elo y el Clasificador para detectar 'Partidos Trampa'."""
    modelos = [np.asarray(m, dtype=float) for m in (p_dc, p_elo, p_lr)
               if m is not None and not np.any(np.isnan(m))]
    if len(modelos) < 2:
        return {"estado": "🟡 Modelo único", "spread_pp": 0.0, "es_trampa": False,
                "mensaje": "Un solo modelo activo.", "penalizacion": 0.0}
    mat = np.vstack(modelos)
    spread_pp = float((mat.max(axis=0) - mat.min(axis=0)).max() * 100.0)
    favs = {int(np.argmax(m)) for m in modelos}
    pocos_datos = min(n_h, n_a) < 10

    if spread_pp >= 13.0 or (len(favs) > 1 and spread_pp >= 9.0) or pocos_datos:
        motivo = []
        if spread_pp >= 9.0:
            motivo.append(f"discrepancia de {spread_pp:.1f} pp entre modelos")
        if len(favs) > 1:
            motivo.append("distinto favorito según el modelo")
        if pocos_datos:
            motivo.append(f"muestra corta ({min(n_h, n_a)} PJ)")
        return {
            "estado": "🔴 Alerta: Partido Trampa / Alta Discrepancia",
            "spread_pp": round(spread_pp, 1),
            "es_trampa": True,
            "mensaje": "⚠️ Precaución (No-Bet o reducir stake): " + ", ".join(motivo) + ".",
            "penalizacion": 0.04,
        }
    if spread_pp <= 6.5 and len(favs) == 1:
        return {
            "estado": "🟢 Alto Consenso entre Modelos",
            "spread_pp": round(spread_pp, 1),
            "es_trampa": False,
            "mensaje": f"✅ Dixon-Coles, Elo y el Clasificador coinciden estrechamente (dif. máx. {spread_pp:.1f} pp).",
            "penalizacion": 0.0,
        }
    return {
        "estado": "🟡 Consenso Moderado",
        "spread_pp": round(spread_pp, 1),
        "es_trampa": False,
        "mensaje": f"Acuerdo normal entre modelos (diferencia máx. {spread_pp:.1f} pp).",
        "penalizacion": 0.01,
    }


def selecciones_bajo_riesgo(home, away, p_fin, s_1x2, mk, s_mk, secundarios=None, basico=False, max_picks=3, penalizacion_consenso=0.0):
    """Selecciona entre 2 y 3 mercados independientes con alta probabilidad y bajo riesgo (alto piso P10 en Monte Carlo).
    Descarta cuotas triviales (>90%) salvo que no exista alternativa en esa familia, y diversifica por categoría."""
    candidatos = []

    def agregar(familia, nombre, prob, muestras, penalizacion=0.0):
        prob = float(prob)
        p10, p90 = float(np.percentile(muestras, 10)), float(np.percentile(muestras, 90))
        # Penaliza opciones fuera del rango ideal [65%, 89%] para evitar mercados triviales (>90%) o muy riesgosos (<65%)
        pen_rango = 0.0
        if prob > 0.89:
            pen_rango = (prob - 0.89) * 1.8
        elif prob < 0.65:
            pen_rango = (0.65 - prob) * 1.5
        score = p10 - 0.25 * (p90 - p10) - penalizacion - penalizacion_consenso - pen_rango
        if 0.58 <= prob <= 0.94:
            if prob >= 0.76 and p10 >= 0.68 and penalizacion_consenso < 0.03:
                riesgo = "🟢 Muy bajo"
            elif prob >= 0.68 and p10 >= 0.60:
                riesgo = "🟢 Bajo"
            else:
                riesgo = "🟡 Moderado-bajo"
            candidatos.append({
                "Selección": nombre,
                "Categoría": familia,
                "Probabilidad": prob,
                "P10": p10,
                "P90": p90,
                "Cuota justa": round(1.0 / max(prob, 1e-6), 2),
                "Riesgo": riesgo,
                "_score": score,
            })

    # 1. Familia Resultado (1X2, Doble Oportunidad y Draw No Bet / Empate No Acción)
    p_h, p_d, p_a = float(p_fin[0]), float(p_fin[1]), float(p_fin[2])
    agregar("Resultado", f"Gana {home}", p_h, s_1x2[:, 0])
    agregar("Resultado", f"Gana {away}", p_a, s_1x2[:, 2])
    etq_1x = f"{home} o Empate" if basico else f"1X ({home} o Empate)"
    etq_x2 = f"{away} o Empate" if basico else f"X2 ({away} o Empate)"
    etq_12 = f"{home} o {away} (sin empate)" if basico else f"12 ({home} o {away})"
    # Si ya hay un claro favorito (>=72%), penalizar 1X/X2 para priorizar la victoria directa con cuota atractiva
    pen_1x = 0.08 if p_h >= 0.72 else 0.0
    pen_x2 = 0.08 if p_a >= 0.72 else 0.0
    agregar("Resultado", etq_1x, p_h + p_d, s_1x2[:, 0] + s_1x2[:, 1], pen_1x)
    agregar("Resultado", etq_x2, p_d + p_a, s_1x2[:, 1] + s_1x2[:, 2], pen_x2)
    # 12 (Local o Visita) solo tiene sentido táctico en duelos abiertos y parejos donde el empate es el menos probable
    if min(p_h, p_a) >= 0.24 and max(p_h, p_a) <= 0.55 and p_d < min(p_h, p_a):
        agregar("Resultado", etq_12, p_h + p_a, s_1x2[:, 0] + s_1x2[:, 2], 0.025)
    # Draw No Bet (solo útil para favoritos moderados <70% donde cubrir el empate aporta valor real)
    denom_dnb = np.maximum(s_1x2[:, 0] + s_1x2[:, 2], 1e-6)
    if 0.45 <= p_h < 0.70:
        p_dnb_h = float(p_h / max(p_h + p_a, 1e-6))
        agregar("Resultado", f"{home} DNB (Empate No Acción)", p_dnb_h, s_1x2[:, 0] / denom_dnb, 0.015)
    if 0.45 <= p_a < 0.70:
        p_dnb_a = float(p_a / max(p_h + p_a, 1e-6))
        agregar("Resultado", f"{away} DNB (Empate No Acción)", p_dnb_a, s_1x2[:, 2] / denom_dnb, 0.015)

    # 2. Familia Goles (Over / Under)
    for k, etq in (("Over 1.5", "Over 1.5 goles"), ("Over 2.5", "Over 2.5 goles"),
                   ("Under 2.5", "Under 2.5 goles"), ("Under 3.5", "Under 3.5 goles")):
        agregar("Goles", etq, mk[k], [m[k] for m in s_mk])

    # 3. Familia Ambos Marcan (solo modo completo)
    if not basico:
        agregar("Ambos marcan", "Ambos marcan (Sí)", mk["Ambos marcan (BTTS)"], [m["Ambos marcan (BTTS)"] for m in s_mk], 0.01)
        agregar("Ambos marcan", "Ambos marcan (No)", mk["No BTTS"], [m["No BTTS"] for m in s_mk], 0.01)

    # 4. Familia Estadísticas secundarias (córners, tiros, amarillas - solo modo completo)
    if not basico and secundarios:
        for item in secundarios:
            agregar(item["familia"], item["nombre"], item["prob"], item["muestras"], penalizacion=0.03)

    # Elegir máximo 1 por categoría (el de mayor _score) para no repetir mercados correlacionados
    mejores_por_cat = {}
    for c in candidatos:
        cat = c["Categoría"]
        if cat not in mejores_por_cat or c["_score"] > mejores_por_cat[cat]["_score"]:
            mejores_por_cat[cat] = c

    seleccionados = sorted(mejores_por_cat.values(), key=lambda x: x["_score"], reverse=True)[:max_picks]
    # Si solo hay 1 categoría (ej. modo básico tiene Resultado y Goles, o si falta un 3ro), permitir un 2do de otra línea distinta
    if len(seleccionados) < 2 and len(candidatos) >= 2:
        usados = {s["Selección"] for s in seleccionados}
        for c in sorted(candidatos, key=lambda x: x["_score"], reverse=True):
            if c["Selección"] not in usados:
                seleccionados.append(c)
                if len(seleccionados) >= 2:
                    break

    for s in seleccionados:
        s.pop("_score", None)
    return seleccionados


def escanear_jornada(partidos, P_liga, fecha, w_dc=0.6, sigma=0.12, basico=False):
    """Evalúa múltiples cruces [(local, visita), ...] o dicts multiliga [{'codigo', 'liga', 'local', 'visita'}, ...],
    los ordena de menor a mayor riesgo y construye las mejores Combinadas Doble (2) y Triple (3)."""
    es_multiliga = isinstance(P_liga, dict) and "estado" not in P_liga
    rng = np.random.default_rng(42)
    filas = []

    for item in partidos:
        if isinstance(item, dict):
            h, a = item["local"], item["visita"]
            cod = item.get("codigo")
            nom_liga = item.get("liga", cod or "")
            P_actual = P_liga.get(cod) if es_multiliga else P_liga
            f_partido = pd.Timestamp(item.get("fecha", fecha))
        else:
            h, a = item
            nom_liga = ""
            P_actual = P_liga
            f_partido = pd.Timestamp(fecha)

        if h == a or P_actual is None:
            continue
        est, dc, lr, df = P_actual["estado"], P_actual["dc"], P_actual["lr"], P_actual["df"]
        tasa_emp = float((df["FTR"] == "D").mean())
        lam, mu = tasas_dc(dc, h, a, False)
        rho = dc["rho"]
        M = matriz(lam, mu, rho)
        p_dc = probs_1x2(M)
        p_elo = est.elo.prob_1x2(h, a, False, tasa_empate=tasa_emp)
        f = est.features(h, a, f_partido)
        p_lr = predecir_lr(lr, pd.DataFrame([f])[FEATS])[0]
        p_fin = w_dc * p_dc + (1.0 - w_dc) * p_lr

        n_h, n_a = est.n.get(h, 0), est.n.get(a, 0)
        cons = diagnosticar_consenso(p_dc, p_elo, p_lr, n_h, n_a)
        f_h = float(np.clip(math.sqrt(38.0 / max(n_h, 5)), 0.7, 2.0))
        f_a = float(np.clip(math.sqrt(38.0 / max(n_a, 5)), 0.7, 2.0))
        s_lam = sigma * math.sqrt(0.7 * f_h**2 + 0.3 * f_a**2)
        s_mu = sigma * math.sqrt(0.7 * f_a**2 + 0.3 * f_h**2)

        s_1x2, s_mk = [], []
        for _ in range(200):
            pl = lam * math.exp(rng.normal(-0.5 * s_lam**2, s_lam))
            pm = mu * math.exp(rng.normal(-0.5 * s_mu**2, s_mu))
            Ms = matriz(pl, pm, rho)
            s_1x2.append(w_dc * probs_1x2(Ms) + (1.0 - w_dc) * p_lr)
            s_mk.append(mercados_goles(Ms))
        s_1x2 = np.array(s_1x2)
        mk = mercados_goles(M)
        picks = selecciones_bajo_riesgo(h, a, p_fin, s_1x2, mk, s_mk, basico=basico,
                                        max_picks=2, penalizacion_consenso=cons["penalizacion"])
        if not picks:
            continue
        mejor = picks[0]
        fila = {}
        if nom_liga:
            fila["Liga"] = nom_liga
        fila.update({
            "Partido": f"{h} vs {a}",
            "Mejor selección": mejor["Selección"],
            "Categoría": mejor["Categoría"],
            "Probabilidad %": round(mejor["Probabilidad"] * 100, 1),
            "Piso MC (P10) %": round(mejor["P10"] * 100, 1),
            "Cuota justa": mejor["Cuota justa"],
            "Cuota valor (EV≥2%)": round(1.02 / max(mejor["Probabilidad"], 1e-6), 2),
            "Riesgo": mejor["Riesgo"],
            "Consenso": cons["estado"],
            "_p": mejor["Probabilidad"],
            "_p10": mejor["P10"],
            "_trampa": cons["es_trampa"],
        })
        filas.append(fila)

    filas.sort(key=lambda x: (not x["_trampa"], x["_p10"], x["_p"]), reverse=True)
    combinada, triple = None, None
    no_trampa = [x for x in filas if not x["_trampa"]]
    aptos_doble = no_trampa if len(no_trampa) >= 2 else filas
    if len(aptos_doble) >= 2:
        a1, a2 = aptos_doble[0], aptos_doble[1]
        p_comb = a1["_p"] * a2["_p"]
        p10_comb = a1["_p10"] * a2["_p10"]
        combinada = {
            "Selección 1": f"{a1['Partido']}: {a1['Mejor selección']} ({a1['Probabilidad %']:.1f}%)",
            "Selección 2": f"{a2['Partido']}: {a2['Mejor selección']} ({a2['Probabilidad %']:.1f}%)",
            "Probabilidad conjunta %": round(p_comb * 100.0, 1),
            "Piso conjunto (P10) %": round(p10_comb * 100.0, 1),
            "Cuota justa combinada": round(1.0 / max(p_comb, 1e-6), 2),
            "Cuota con valor (EV≥2%)": round(1.02 / max(p_comb, 1e-6), 2),
        }
    aptos_triple = no_trampa if len(no_trampa) >= 3 else filas
    if len(aptos_triple) >= 3:
        a1, a2, a3 = aptos_triple[0], aptos_triple[1], aptos_triple[2]
        p_tri = a1["_p"] * a2["_p"] * a3["_p"]
        p10_tri = a1["_p10"] * a2["_p10"] * a3["_p10"]
        triple = {
            "Selección 1": f"{a1['Partido']}: {a1['Mejor selección']} ({a1['Probabilidad %']:.1f}%)",
            "Selección 2": f"{a2['Partido']}: {a2['Mejor selección']} ({a2['Probabilidad %']:.1f}%)",
            "Selección 3": f"{a3['Partido']}: {a3['Mejor selección']} ({a3['Probabilidad %']:.1f}%)",
            "Probabilidad conjunta %": round(p_tri * 100.0, 1),
            "Piso conjunto (P10) %": round(p10_tri * 100.0, 1),
            "Cuota justa combinada": round(1.0 / max(p_tri, 1e-6), 2),
            "Cuota con valor (EV≥2%)": round(1.02 / max(p_tri, 1e-6), 2),
        }
    for x in filas:
        for k in ("_p", "_p10", "_trampa"):
            x.pop(k, None)
    return {"ranking": pd.DataFrame(filas), "combinada": combinada, "triple": triple}
