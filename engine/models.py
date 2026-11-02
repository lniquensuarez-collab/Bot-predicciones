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
def preparar_dc(df, fecha_ref, half_life, reg=0.5):
    """Función objetivo (valor + gradiente analítico). Parámetros: ataque(n), defensa(n), ventaja local,
    rho (marcadores bajos) y c (intercepto = nivel de goles de la liga). Separada para poder testear el gradiente."""
    dias = (fecha_ref - df["Date"]).dt.days.to_numpy(float).clip(min=0)
    w = np.exp(-np.log(2) / half_life * dias)
    d = df[w > 1e-4]
    w = w[w > 1e-4]

    equipos = sorted(set(d["HomeTeam"]) | set(d["AwayTeam"]))
    idx = {t: i for i, t in enumerate(equipos)}
    n = len(equipos)
    hi, ai = d["HomeTeam"].map(idx).to_numpy(), d["AwayTeam"].map(idx).to_numpy()
    x, y = d["FTHG"].to_numpy(float), d["FTAG"].to_numpy(float)
    m00, m01 = (x == 0) & (y == 0), (x == 0) & (y == 1)
    m10, m11 = (x == 1) & (y == 0), (x == 1) & (y == 1)
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


def selecciones_bajo_riesgo(home, away, p_fin, s_1x2, mk, s_mk, secundarios=None, basico=False, max_picks=3):
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
        score = p10 - 0.25 * (p90 - p10) - penalizacion - pen_rango
        if 0.58 <= prob <= 0.94:
            if prob >= 0.76 and p10 >= 0.68:
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

    # 1. Familia Resultado (1X2 y Doble Oportunidad)
    agregar("Resultado", f"Gana {home}", p_fin[0], s_1x2[:, 0])
    agregar("Resultado", f"Gana {away}", p_fin[2], s_1x2[:, 2])
    etq_1x = f"{home} o Empate" if basico else f"1X ({home} o Empate)"
    etq_x2 = f"{away} o Empate" if basico else f"X2 ({away} o Empate)"
    etq_12 = f"{home} o {away} (sin empate)" if basico else f"12 ({home} o {away})"
    agregar("Resultado", etq_1x, p_fin[0] + p_fin[1], s_1x2[:, 0] + s_1x2[:, 1])
    agregar("Resultado", etq_x2, p_fin[1] + p_fin[2], s_1x2[:, 1] + s_1x2[:, 2])
    agregar("Resultado", etq_12, p_fin[0] + p_fin[2], s_1x2[:, 0] + s_1x2[:, 2])

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
