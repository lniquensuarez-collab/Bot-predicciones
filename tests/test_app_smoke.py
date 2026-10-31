"""Ejecuta app.py de punta a punta con un Streamlit simulado y una base sintética (sin red, sin Streamlit real).
Verifica la lógica de la interfaz: modo básico (Brasil...), modo completo (Premier) y partido entre ligas.
Ejecutar:  python -m tests.test_app_smoke"""
import os
import runpy
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
TMP = tempfile.mkdtemp()
os.environ["FUTBOL_DB"] = str(Path(TMP) / "smoke.db")

from engine import db  # noqa: E402
from tests.test_engine import sinteticos  # noqa: E402


class Parar(Exception):
    pass


class FakeSt:
    """Simula la API de Streamlit que usa app.py y registra lo que se pinta."""

    def __init__(self, override):
        self.override, self.log, self.session_state = override, [], {}
        self.sidebar = self
        self.cache_data = self._cache()

    def _cache(self):
        outer = self

        class C:
            def __call__(self, *a, **k):
                return (lambda f: f) if not (a and callable(a[0])) else a[0]

            def clear(self):
                outer.log.append(("cache_clear", ""))
        return C()

    def __enter__(self): return self
    def __exit__(self, *a): return False

    def _rec(self, nombre, *a):
        self.log.append((nombre, str(a[0]) if a else ""))

    def columns(self, n): return [self] * (n if isinstance(n, int) else len(n))
    def tabs(self, nombres): return [self] * len(nombres)
    def expander(self, *a, **k): return self
    def spinner(self, *a, **k): return self
    def progress(self, *a, **k): return self
    def empty(self): pass
    def stop(self): raise Parar()
    def rerun(self): raise Parar()

    def selectbox(self, label, options, index=0, **k):
        options = list(options)
        return self.override.get(label, options[min(index, len(options) - 1)])

    def checkbox(self, label, value=False, **k):
        return next((v for p, v in self.override.items() if label.startswith(p)), value)

    def slider(self, label, mn=None, mx=None, value=None, *a, **k):
        return value if value is not None else mn

    def number_input(self, label, mn=None, mx=None, value=0.0, *a, **k): return value
    def date_input(self, label, value=None, **k): return value
    def multiselect(self, label, options, default=None, **k): return default or []
    def button(self, label, **k): return label.startswith("🚀")
    def metric(self, label, value, *a, **k): self._rec("metric", label)
    def markdown(self, t, *a, **k): self._rec("markdown", t)
    def dataframe(self, d, *a, **k): self._rec("dataframe", getattr(d, "to_string", lambda: str(d))())

    def __getattr__(self, nombre):
        return lambda *a, **k: self._rec(nombre, *a)


def correr(override):
    st = FakeSt(override)
    mod = types.ModuleType("streamlit")
    mod.__dict__.update({k: getattr(st, k) for k in dir(FakeSt) if not k.startswith("__")})
    mod.__getattr__ = st.__getattr__
    for k in ("sidebar", "session_state", "cache_data", "log", "override"):
        setattr(mod, k, getattr(st, k))
    sys.modules["streamlit"] = mod
    try:
        runpy.run_path(str(RAIZ / "app.py"), run_name="__main__")
    except Parar:
        pass
    # Solo lo que pinta la predicción: desde el título del partido hasta el inicio de la pestaña Ranking
    ini = next((i for i, (n, _) in enumerate(st.log) if n == "subheader"), 0)
    fin = next((i for i, (n, x) in enumerate(st.log) if n == "caption" and x.startswith("El Elo solo")), len(st.log))
    texto = "\n".join(f"{n}|{x}" for n, x in st.log[ini:fin])
    return st, texto


def preparar_db():
    with db.sesion() as con:
        db.guardar_partidos(con, sinteticos("E0", 450, 1))
        db.guardar_partidos(con, sinteticos("D1", 450, 2, c0=0.28))
        brasil = sinteticos("BRA", 450, 3)
        for f in brasil:                                    # los xlsx no traen estadísticas
            for k in ("tiros_p_l", "tiros_p_v", "corners_l", "corners_v", "faltas_l", "faltas_v",
                      "amarillas_l", "amarillas_v", "rojas_l", "rojas_v"):
                f[k] = None
        db.guardar_partidos(con, brasil)


PREMIER = "🇬🇧 Inglaterra · Premier League"
BRASIL = "🇧🇷 Brasil"


def test_modo_completo():
    _, t = correr({"🏆 Liga (ranking y backtest)": PREMIER})
    for esperado in ["Goles", "Comparación de modelos", "Marcadores más probables", "Córners y", "Faltas y tarjetas por equipo",
                     "Faltas", "Amarillas", "Rojas", "P(≥1 roja)", "Over 22.5", "Over 3.5 %"]:
        assert esperado in t, f"falta en modo completo: {esperado}"


def test_modo_basico():
    _, t = correr({"🏆 Liga (ranking y backtest)": BRASIL})
    assert "metric|Local" in t and "metric|Visita" in t and "Goles" in t and "Over 2.5" in t
    for prohibido in ["Comparación de modelos", "Marcadores", "Córners", "tiros a puerta", "Faltas", "Amarillas",
                      "Rojas", "BTTS", "Ambos marcan", "1X (local"]:
        assert prohibido not in t, f"no debería aparecer en modo básico: {prohibido}"


def test_partido_entre_ligas():
    _, t = correr({"🌍": True, "🏆 Liga (ranking y backtest)": PREMIER,
                   "Liga del local": PREMIER, "Liga de la visita": "🇩🇪 Alemania · Bundesliga"})
    assert "Partido entre ligas" in t and "Faltas y tarjetas por equipo" in t and "Over 2.5" in t


if __name__ == "__main__":
    preparar_db()
    for f in (test_modo_completo, test_modo_basico, test_partido_entre_ligas):
        f()
        print("✅", f.__name__)
