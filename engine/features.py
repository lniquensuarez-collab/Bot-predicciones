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
    """Estado acumulado por equipo de UNA liga (EWMA local/visita, Elo, forma, descanso)."""

    def __init__(self, alpha=0.25, elo=None):
        self.alpha = alpha
        self.elo = elo or EloSystem()
        self.ewma, self.n, self.pts, self.last = {}, {}, {}, {}

    def valor(self, t, lado, stat):
        return self.ewma.get((t, lado, stat), np.nan)

    def features(self, h, a, fecha):
        def diff(s):
            x = self.valor(h, "H", s) - self.valor(a, "A", s)
            return 0.0 if np.isnan(x) else x

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
        return (_media([self.valor(h, "H", favor), self.valor(a, "A", contra)]),
                _media([self.valor(a, "A", favor), self.valor(h, "H", contra)]))

    def update(self, r):
        h, a = r["HomeTeam"], r["AwayTeam"]
        gl, gv = float(r["FTHG"]), float(r["FTAG"])
        # F/Y/R = faltas, amarillas y rojas del propio equipo; FA/YA/RA = las del rival en esos partidos
        vh = {"GF": gl, "GC": gv, "ST": _f(r, "HST"), "STA": _f(r, "AST"), "C": _f(r, "HC"), "CA": _f(r, "AC"),
              "F": _f(r, "HF"), "FA": _f(r, "AF"), "Y": _f(r, "HY"), "YA": _f(r, "AY"),
              "R": _f(r, "HR"), "RA": _f(r, "AR")}
        va = {"GF": gv, "GC": gl, "ST": _f(r, "AST"), "STA": _f(r, "HST"), "C": _f(r, "AC"), "CA": _f(r, "HC"),
              "F": _f(r, "AF"), "FA": _f(r, "HF"), "Y": _f(r, "AY"), "YA": _f(r, "HY"),
              "R": _f(r, "AR"), "RA": _f(r, "HR")}
        for t, lado, vals in ((h, "H", vh), (a, "A", va)):
            for s, v in vals.items():
                if np.isnan(v):
                    continue
                old = self.valor(t, lado, s)
                self.ewma[(t, lado, s)] = v if np.isnan(old) else self.alpha * v + (1 - self.alpha) * old
        self.elo.update(h, a, gl, gv)
        ph, pa = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}[r["FTR"]]
        self.pts.setdefault(h, []).append(ph)
        self.pts.setdefault(a, []).append(pa)
        self.last[h] = self.last[a] = r["Date"]
        self.n[h] = self.n.get(h, 0) + 1
        self.n[a] = self.n.get(a, 0) + 1


def esperado_conteo_cruzado(est_h, est_a, h, a, favor, contra):
    """Conteo esperado (córners, tiros) cuando local y visita vienen de ligas distintas."""
    return (_media([est_h.valor(h, "H", favor), est_a.valor(a, "A", contra)]),
            _media([est_a.valor(a, "A", favor), est_h.valor(h, "H", contra)]))


def capacidades(df, minimo=100):
    """Qué estadísticas permite el archivo de la liga (datos reales, no suposiciones).
    Brasil, Argentina, Japón, China y EE. UU. no traen ninguna -> modo básico (ganador y goles)."""
    def ok(*cols):
        return all(c in df.columns and df[c].notna().sum() >= minimo for c in cols)
    return {"corners": ok("HC", "AC"), "tiros": ok("HST", "AST"), "faltas": ok("HF", "AF"),
            "tarjetas": ok("HY", "AY"), "rojas": ok("HR", "AR")}


def construir_features(df):
    """Recorre el historial en orden cronológico. Devuelve (DataFrame de features, Estado final)."""
    est, filas = Estado(), []
    for r in df.to_dict("records"):
        filas.append(est.features(r["HomeTeam"], r["AwayTeam"], r["Date"]))  # ANTES de actualizar
        est.update(r)
    return pd.DataFrame(filas, index=df.index), est
