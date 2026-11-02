# Ultimate AI V14 — Ligas de clubes (sin APIs, sin selecciones)

URLs canónicas: `https://football-data.co.uk/mmz4281/2627/I1.csv` (CSV) y `https://football-data.co.uk/new/BRA.xlsx` (XLSX).

Fuente única: football-data.co.uk

| Liga | Archivo |
|---|---|
| Premier League, Championship, Bundesliga, La Liga, Serie A, Ligue 1, Primeira Liga, Süper Lig | CSV por temporada (`/mmz4281/<temporada>/<código>.csv`; por defecto las 2 últimas, hoy 2526 y 2627) |
| Brasil, Argentina, Japón, China, EE. UU. | XLSX con todas las temporadas (`/new/<código>.xlsx`; se conservan los últimos N años) |

```
futbol_ai/
├── app.py                 # Interfaz Streamlit
├── engine/
│   ├── config.py          # Catálogo de ligas y fuerza relativa entre ligas
│   ├── db.py              # SQLite (data/futbol.db)
│   ├── data_bot.py        # Ingesta CSV/XLSX -> SQLite
│   ├── elo_system.py      # Elo
│   ├── features.py        # EWMA / forma / descanso SIN fuga de información
│   ├── models.py          # Dixon-Coles (con intercepto), clasificador, mercados, modo entre ligas
│   └── backtest.py        # Walk-forward, Log Loss, Brier, calibración
├── tests/test_engine.py   # Pruebas offline del motor: python -m tests.test_engine
├── tests/test_app_smoke.py# Prueba de la interfaz con Streamlit simulado: python -m tests.test_app_smoke
└── data/                  # futbol.db se crea sola
```

## Uso
```
pip install -r requirements.txt
python -m tests.test_engine
streamlit run app.py
```
Si vienes de la V13, empieza con una carpeta `data/` vacía (o borra `futbol.db`) para no arrastrar datos de selecciones.

## Qué muestra la predicción según la liga
El modo se decide con los datos reales de cada liga (no con una lista fija):

| Liga | Salida |
|---|---|
| Brasil, Argentina, Japón, China, EE. UU. (sin estadísticas de partido) | Probabilidad de ganador + goles (esperados por equipo, total y Over/Under 1.5 / 2.5 / 3.5) |
| Premier, Championship, Bundesliga, La Liga, Serie A, Ligue 1, Primeira Liga, Süper Lig | Análisis completo: comparación de modelos, mercados de goles, BTTS, doble oportunidad, marcadores, córners, tiros a puerta y **faltas y tarjetas por equipo** (amarillas, rojas, P(≥1 roja) y líneas Over) |

Si vienes de una base `futbol.db` anterior, la app añade sola las columnas de faltas/tarjetas y vuelve a descargar los CSV europeos una vez.
