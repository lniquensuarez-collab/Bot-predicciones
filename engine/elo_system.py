"""Sistema Elo: ventaja de local, margen de goles y probabilidades 1X2 aproximadas."""
import numpy as np


def prob_esperada(ra, rb, ventaja=0.0):
    """Expectativa de resultado del equipo A (victoria = 1, empate = 0.5)."""
    return 1.0 / (1.0 + 10 ** (-(ra + ventaja - rb) / 400.0))


def esperado_a_1x2(E):
    """Convierte la expectativa Elo en probabilidades [local, empate, visita] (aproximación)."""
    E = np.asarray(E, dtype=float)
    p_emp = 1.2 * E * (1 - E)                 # 30 % cuando E = 0.5; baja con la desigualdad
    return np.stack([E - p_emp / 2, p_emp, 1 - E - p_emp / 2], axis=-1)


def prob_cruzada(ra, rb, neutral=False, ventaja=60.0):
    """1X2 entre equipos de ligas distintas. ra y rb YA incluyen el ajuste de fuerza de su liga."""
    return esperado_a_1x2(prob_esperada(ra, rb, 0.0 if neutral else ventaja))


def _margen(dif):
    return 1.0 if dif <= 1 else (1.5 if dif == 2 else (11 + dif) / 8.0)


class EloSystem:
    def __init__(self, k_base=20.0, ventaja=60.0, inicial=1500.0, usar_margen=True):
        self.k_base, self.ventaja, self.inicial, self.usar_margen = k_base, ventaja, inicial, usar_margen
        self.ratings = {}

    def rating(self, t):
        return self.ratings.get(t, self.inicial)

    def esperado(self, h, a, neutral=False):
        return prob_esperada(self.rating(h), self.rating(a), 0.0 if neutral else self.ventaja)

    def prob_1x2(self, h, a, neutral=False):
        return esperado_a_1x2(self.esperado(h, a, neutral))

    def update(self, h, a, goles_l, goles_v):
        e = self.esperado(h, a)
        w = 1.0 if goles_l > goles_v else (0.5 if goles_l == goles_v else 0.0)
        g = _margen(abs(goles_l - goles_v)) if self.usar_margen else 1.0
        delta = self.k_base * g * (w - e)
        self.ratings[h] = self.rating(h) + delta
        self.ratings[a] = self.rating(a) - delta
        return delta
