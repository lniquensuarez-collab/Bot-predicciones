"""Ingeniería de características SIN fuga de información: las features de cada partido
se calculan con el estado previo y recién después se actualiza el estado."""
import numpy as np
import pandas as pd

from .elo_system import EloSystem

FEATS = ["elo_diff", "att_diff", "def_diff", "sot_diff", "cor_diff", "form_diff", "rest_diff"]


def _f(r, k):
    v = r.get(k, np.nan)
    try:
        return np.nan if v is None else float(v)
    except (TypeError, ValueError):
        return np.nan


def _media(vals):
    v = [x for x in vals if not np.isnan(x)]
    return float(np.mean(v)) if v else np.nan


class Estado:
    """Estado acumulado por equipo de UNA liga (EWMA híbrida local/visita + global, Elo, forma, descanso)."""

    def __init__(self, alpha=0.25, peso_lado=0.70, elo=None):
        self.alpha = alpha
        self.peso_lado = peso_lado
        self.elo = elo or EloSystem()
        self.ewma, self.n, self.pts, self.last = {}, {}, {}, {}
        self.g_sum, self.g_count = {}, {}  # Para promedios globales de la liga
        self.ref_stats = {}                # Estadísticas acumuladas por árbitro
        self.temporada_actual = None

    def arbitros_disponibles(self, min_partidos=3):
        return sorted(k for k, v in self.ref_stats.items() if v["n"] >= min_partidos)

    def factor_arbitro(self, arbitro, prior_n=10.0):
        """Multiplicador bayesiano de rigor arbitral (faltas, amarillas, rojas) respecto a la media de la liga."""
        if not arbitro or arbitro not in self.ref_stats:
            return {"faltas": 1.0, "amarillas": 1.0, "rojas": 1.0, "partidos": 0}
        st = self.ref_stats[arbitro]
        out = {"partidos": int(st["n"])}
        for clave, stat in (("faltas", "F"), ("amarillas", "Y"), ("rojas", "R")):
            n_s = st[f"n_{stat}"]
            tot_liga = self.g_sum.get(("H", stat), 0.0) + self.g_sum.get(("A", stat), 0.0)
            cnt_liga = max(1, min(self.g_count.get(("H", stat), 0), self.g_count.get(("A", stat), 0)))
            mu_liga = tot_liga / cnt_liga if cnt_liga > 0 else 0.0
            if n_s > 0 and mu_liga > 1e-6:
                media_arb = (st[stat] + prior_n * mu_liga) / (n_s + prior_n)
                out[clave] = float(np.clip(media_arb / mu_liga, 0.65, 1.50))
            else:
                out[clave] = 1.0
        return out

    def valor(self, t, lado, stat):
        """Combina EWMA específica de lado (H/A) con EWMA general del equipo (ALL) para reducir varianza."""
        v_lado = self.ewma.get((t, lado, stat), np.nan)
        v_all = self.ewma.get((t, "ALL", stat), np.nan)
        if not np.isnan(v_lado) and not np.isnan(v_all):
            return self.peso_lado * v_lado + (1.0 - self.peso_lado) * v_all
        return v_lado if not np.isnan(v_lado) else v_all

    def media_liga(self, lado, stat):
        c = self.g_count.get((lado, stat), 0)
        return self.g_sum.get((lado, stat), 0.0) / c if c > 0 else np.nan

    def features(self, h, a, fecha):
        def diff(s):
            vh = self.valor(h, "H", s)
            va = self.valor(a, "A", s)

            # Imputación: si no hay datos del equipo, usar el promedio histórico de la liga para ese lado
            if np.isnan(vh):
                vh = self.g_sum.get(("H", s), 0.0) / max(1, self.g_count.get(("H", s), 0))
            if np.isnan(va):
                va = self.g_sum.get(("A", s), 0.0) / max(1, self.g_count.get(("A", s), 0))

            return vh - va

        def descanso(t):
            l = self.last.get(t)
            return min((fecha - l).days, 14) if l is not None else 7

        return {
            "elo_diff": self.elo.rating(h) - self.elo.rating(a),
            "att_diff": diff("GF"), "def_diff": diff("GC"), "sot_diff": diff("ST"), "cor_diff": diff("C"),
            "form_diff": sum(self.pts.get(h, [])[-5:]) - sum(self.pts.get(a, [])[-5:]),
            "rest_diff": descanso(h) - descanso(a),
            "n_prev": min(self.n.get(h, 0), self.n.get(a, 0)),
        }

    def esperado_conteo(self, h, a, favor, contra):
        return esperado_conteo_cruzado(self, self, h, a, favor, contra)

    def update(self, r):
        temp = r.get("Season") or r.get("temporada")
        if temp is not None:
            if self.temporada_actual is not None and temp != self.temporada_actual:
                self.elo.regresion_temporada()
            self.temporada_actual = temp

        h, a = r["HomeTeam"], r["AwayTeam"]
        gl, gv = float(r["FTHG"]), float(r["FTAG"])
        vh = {"GF": gl, "GC": gv, "ST": _f(r, "HST"), "STA": _f(r, "AST"), "C": _f(r, "HC"), "CA": _f(r, "AC"),
              "F": _f(r, "HF"), "FA": _f(r, "AF"), "Y": _f(r, "HY"), "YA": _f(r, "AY"), "R": _f(r, "HR"), "RA": _f(r, "AR")}
        va = {"GF": gv, "GC": gl, "ST": _f(r, "AST"), "STA": _f(r, "HST"), "C": _f(r, "AC"), "CA": _f(r, "HC"),
              "F": _f(r, "AF"), "FA": _f(r, "HF"), "Y": _f(r, "AY"), "YA": _f(r, "HY"), "R": _f(r, "AR"), "RA": _f(r, "HR")}

        for t, lado, vals in ((h, "H", vh), (a, "A", va)):
            for s, v in vals.items():
                if np.isnan(v):
                    continue
                old = self.ewma.get((t, lado, s), np.nan)
                self.ewma[(t, lado, s)] = v if np.isnan(old) else self.alpha * v + (1 - self.alpha) * old

                old_all = self.ewma.get((t, "ALL", s), np.nan)
                self.ewma[(t, "ALL", s)] = v if np.isnan(old_all) else self.alpha * v + (1 - self.alpha) * old_all

                # Actualizar promedios globales
                self.g_sum[(lado, s)] = self.g_sum.get((lado, s), 0.0) + v
                self.g_count[(lado, s)] = self.g_count.get((lado, s), 0) + 1

        ref = r.get("Referee") or r.get("arbitro")
        if isinstance(ref, str) and ref.strip() and ref.strip().lower() != "nan":
            ref = ref.strip()
            rs = self.ref_stats.setdefault(ref, {"n": 0, "F": 0.0, "n_F": 0, "Y": 0.0, "n_Y": 0, "R": 0.0, "n_R": 0})
            rs["n"] += 1
            for s_k, c_h, c_a in (("F", "HF", "AF"), ("Y", "HY", "AY"), ("R", "HR", "AR")):
                v_tot = _f(r, c_h) + _f(r, c_a)
                if not np.isnan(v_tot):
                    rs[s_k] += v_tot
                    rs[f"n_{s_k}"] += 1

        self.elo.update(h, a, gl, gv)
        ph, pa = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}[r["FTR"]]
        self.pts.setdefault(h, []).append(ph)
        self.pts.setdefault(a, []).append(pa)
        self.last[h] = self.last[a] = r["Date"]
        self.n[h] = self.n.get(h, 0) + 1
        self.n[a] = self.n.get(a, 0) + 1


def esperado_conteo_cruzado(est_h, est_a, h, a, favor, contra):
    """Conteo esperado combinando EWMA de creación propia y concesión del rival."""
    eh = _media([est_h.valor(h, "H", favor), est_a.valor(a, "A", contra)])
    ea = _media([est_a.valor(a, "A", favor), est_h.valor(h, "H", contra)])
    return eh, ea


def construir_features(df, alpha=0.25):
    """Recorre los partidos en orden cronológico: extrae features ANTES de actualizar el estado."""
    est = Estado(alpha=alpha)
    filas = []
    for r in df.to_dict("records"):
        filas.append(est.features(r["HomeTeam"], r["AwayTeam"], r["Date"]))
        est.update(r)
    return pd.DataFrame(filas, index=df.index), est


def capacidades(df, min_partidos=30):
    """Detecta qué mercados estadísticos secundarios tienen suficientes datos reales en la liga."""
    def tiene(c1, c2):
        return (c1 in df.columns and c2 in df.columns
                and int(df[c1].notna().sum()) >= min_partidos
                and int(df[c2].notna().sum()) >= min_partidos)

    return {
        "corners": tiene("HC", "AC"),
        "tiros": tiene("HST", "AST"),
        "faltas": tiene("HF", "AF"),
        "tarjetas": tiene("HY", "AY"),
        "rojas": tiene("HR", "AR"),
    }
