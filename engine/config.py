"""Configuración central: rutas y catálogo de ligas (solo football-data.co.uk, sin APIs)."""
import os
from datetime import date
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("FUTBOL_DB", BASE_DIR / "data" / "futbol.db"))

URL_CSV = "https://football-data.co.uk/mmz4281/{t}/{c}.csv"
URL_XLSX = "https://football-data.co.uk/new/{c}.xlsx"

# fuente: csv (una temporada por archivo) | xlsx (un archivo con todas las temporadas)
# europa: puede participar en el modo "partido europeo entre ligas"
# fuerza: fuerza relativa de la liga en puntos Elo (Premier = 0). SUPUESTO EDITABLE en la app:
#         no sale de los datos porque estos archivos no contienen partidos entre ligas.
LIGAS = {
    "🇬🇧 Inglaterra · Premier League": dict(codigo="E0", fuente="csv", europa=True, fuerza=0),
    "🇬🇧 Inglaterra · Championship": dict(codigo="E1", fuente="csv", europa=True, fuerza=-250),
    "🇩🇪 Alemania · Bundesliga": dict(codigo="D1", fuente="csv", europa=True, fuerza=-45),
    "🇪🇸 España · La Liga": dict(codigo="SP1", fuente="csv", europa=True, fuerza=-25),
    "🇮🇹 Italia · Serie A": dict(codigo="I1", fuente="csv", europa=True, fuerza=-55),
    "🇧🇷 Brasil": dict(codigo="BRA", fuente="xlsx", europa=False, fuerza=0),
    "🇦🇷 Argentina": dict(codigo="ARG", fuente="xlsx", europa=False, fuerza=0),
    "🇯🇵 Japón": dict(codigo="JPN", fuente="xlsx", europa=False, fuerza=0),
    "🇨🇳 China": dict(codigo="CHN", fuente="xlsx", europa=False, fuerza=0),
    "🇺🇸 Estados Unidos": dict(codigo="USA", fuente="xlsx", europa=False, fuerza=0),
}

# 1 punto Elo de diferencia entre ligas ≈ 0.00115 en escala log de ataque/defensa (aproximación)
ELO_A_LOG = 0.00115


def temporadas_recientes(n):
    """Últimas n temporadas europeas calculadas desde hoy. Con n=2 hoy: ['2526', '2627']."""
    hoy = date.today()
    inicio = hoy.year if hoy.month >= 8 else hoy.year - 1
    return [f"{(inicio - k) % 100:02d}{(inicio - k + 1) % 100:02d}" for k in range(n - 1, -1, -1)]
