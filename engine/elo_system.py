"""Sistema Elo: ventaja de local, margen de goles, regresión entre temporadas y probabilidades 1X2 (Davidson)."""
import numpy as np


def prob_esperada(ra, rb, ventaja=0.0):
    """Expectativa de resultado del equipo A (victoria = 1, empate = 0.5)."""
    return 1.0 / (1.0 + 10 ** (-(ra + ventaja - rb) / 400.0))


def esperado_a_1x2(E, tasa_empate=0.27):
    """Convierte la expectativa Elo en probabilidades [local, empate, visita] preservando E = P(H) + 0.5*P(D).
    Usa parametrización tipo Davidson calibrada para que en partidos parejos (E=0.5) P(D) ≈ tasa_empate."""
    E = np.clip(np.asarray(E, dtype=float), 1e-6, 1.0 - 1e-6)
    tasa_empate = float(np.clip(tasa_empate, 0.10, 0.45))
    nu = tasa_empate / max(1.0 - tasa_empate, 1e-6)
    raiz = np.sqrt(E * (1.0 - E))
    # Forma de Davidson modulada por 4*E*(1-E) para garantizar P(H) >= 0 y P(A) >= 0 en las colas
    p_emp = (2.0 * nu * raiz / (1.0 + 2.0 * nu * raiz)) * (4.0 * E * (1.0 - E))
    p_emp = np.minimum(p_emp, 2.0 * np.minimum(E, 1.0 - E))
    return np.stack([E - p_emp / 2.0, p_emp, 1.0 - E - p_emp / 2.0], axis=-1)


def prob_cruzada(ra, rb, neutral=False, ventaja=60.0, tasa_empate=0.27):
    """1X2 entre equipos de ligas distintas. ra y rb YA incluyen el ajuste de fuerza de su liga."""
    return esperado_a_1x2(prob_esperada(ra, rb, 0.0 if neutral else ventaja), tasa_empate=tasa_empate)


def _margen(dif):
    return 1.0 if dif <= 1 else (1.5 if dif == 2 else (11 + dif) / 8.0)


class EloSystem:
    def __init__(self, k_base=20.0, ventaja=60.0, inicial=1500.0, usar_margen=True, carry_over=0.82):
        self.k_base, self.ventaja, self.inicial = k_base, ventaja, inicial
        self.usar_margen, self.carry_over = usar_margen, carry_over
        self.ratings = {}

    def rating(self, t):
        return self.ratings.get(t, self.inicial)

    def regresion_temporada(self, factor=None):
        """Contrae los ratings hacia la media inicial al comenzar una nueva temporada."""
        f = self.carry_over if factor is None else float(factor)
        for t in list(self.ratings):
            self.ratings[t] = f * self.ratings[t] + (1.0 - f) * self.inicial

    def esperado(self, h, a, neutral=False):
        return prob_esperada(self.rating(h), self.rating(a), 0.0 if neutral else self.ventaja)

    def prob_1x2(self, h, a, neutral=False, tasa_empate=0.27):
        return esperado_a_1x2(self.esperado(h, a, neutral), tasa_empate=tasa_empate)

    def update(self, h, a, goles_l, goles_v):
        e = self.esperado(h, a)
        w = 1.0 if goles_l > goles_v else (0.5 if goles_l == goles_v else 0.0)
        g = _margen(abs(goles_l - goles_v)) if self.usar_margen else 1.0
        delta = self.k_base * g * (w - e)
        self.ratings[h] = self.rating(h) + delta
        self.ratings[a] = self.rating(a) - delta
        return delta
