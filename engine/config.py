"""Configuración central: rutas y catálogo de ligas (solo football-data.co.uk, sin APIs)."""
import os
from datetime import date
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("FUTBOL_DB", BASE_DIR / "data" / "futbol.db"))

URL_CSV = "https://football-data.co.uk/mmz4281/{t}/{c}.csv"
URL_XLSX = "https://football-data.co.uk/new/{c}.xlsx"
URL_FIXTURES_CSV = "https://www.football-data.co.uk/fixtures.csv"
URL_FIXTURES_NEW = "https://www.football-data.co.uk/new_league_fixtures.csv"
URL_ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard?dates={ymd}"
URL_UNDERSTAT = "https://understat.com/getLeagueData/{slug}/{anio}"

# Ligas con xG tiro-a-tiro en Understat
UNDERSTAT_SLUGS = {
    "E0": "EPL",
    "SP1": "La_liga",
    "D1": "Bundesliga",
    "I1": "Serie_A",
    "F1": "Ligue_1",
}

# Slugs públicos de ESPN (sin API key) para consultar el calendario diario de nuestras 17 ligas
ESPN_SLUGS = {
    "E0": "eng.1",
    "E1": "eng.2",
    "D1": "ger.1",
    "SP1": "esp.1",
    "I1": "ita.1",
    "F1": "fra.1",
    "P1": "por.1",
    "N1": "ned.1",
    "B1": "bel.1",
    "T1": "tur.1",
    "SC0": "sco.1",
    "G1": "gre.1",
    "BRA": "bra.1",
    "ARG": "arg.1",
    "PER": "per.1",
    "JPN": "jpn.1",
    "CHN": "chn.1",
    "USA": "usa.1",
}

# Altitud de estadios en la Liga 1 de Perú (metros sobre el nivel del mar) para el ajuste Altura vs Llano
ALTITUD_PERU = {
    "Universitario": 150,
    "Alianza Lima": 150,
    "Sporting Cristal": 150,
    "Sport Boys": 30,
    "Deportivo Municipal": 150,
    "Cantolao": 30,
    "Universidad Cesar Vallejo": 35,
    "Cesar Vallejo": 35,
    "Carlos A. Mannucci": 35,
    "Mannucci": 35,
    "Alianza Atletico": 60,
    "Atletico Grau": 30,
    "Juan Pablo II": 30,
    "Melgar": 2335,
    "FBC Melgar": 2335,
    "Alianza Universidad": 1894,
    "UTC": 2750,
    "UTC Cajamarca": 2750,
    "Comerciantes Unidos": 2637,
    "Ayacucho FC": 2761,
    "Ayacucho": 2761,
    "Los Chankas": 2926,
    "Chankas CYC": 2926,
    "ADT": 3053,
    "ADT Tarma": 3053,
    "Sport Huancayo": 3259,
    "Cienciano": 3400,
    "Cusco FC": 3400,
    "Deportivo Garcilaso": 3400,
    "Binacional": 3825,
}
CALOR_NORTE_PERU = {"Alianza Atletico", "Atletico Grau", "Juan Pablo II"}

# Torneos internacionales para búsqueda automática de partidos del día y cruces entre ligas
TORNEOS_INTERNACIONALES = {
    "🇪🇺 UEFA Champions League": dict(codigo="UCL", slug="uefa.champions", region="europa", elo_base=1445, fuerza=-110),
    "🇪🇺 UEFA Europa League": dict(codigo="UEL", slug="uefa.europa", region="europa", elo_base=1400, fuerza=-145),
    "🇪🇺 UEFA Conference League": dict(codigo="UECL", slug="uefa.europa.conf", region="europa", elo_base=1350, fuerza=-185),
    "🌎 CONMEBOL Libertadores": dict(codigo="LIB", slug="conmebol.libertadores", region="sudamerica", elo_base=1425, fuerza=-85),
    "🌎 CONMEBOL Sudamericana": dict(codigo="SUD", slug="conmebol.sudamericana", region="sudamerica", elo_base=1375, fuerza=-125),
}

# Catálogo de clubes recurrentes en torneos internacionales cuyas ligas domésticas no están en SQLite.
# 'elo' representa su Elo efectivo internacional (donde Premier League media = 1500).
EQUIPOS_INVITADOS = {
    # Europa (Ucrania, Chequia, Austria, Suiza, Dinamarca, Noruega, Suecia, Croacia, Serbia, Polonia, etc.)
    "Shakhtar Donetsk": dict(pais="Ucrania", elo=1485),
    "Dynamo Kyiv": dict(pais="Ucrania", elo=1445),
    "Slavia Prague": dict(pais="Chequia", elo=1485),
    "Sparta Prague": dict(pais="Chequia", elo=1475),
    "Viktoria Plzen": dict(pais="Chequia", elo=1445),
    "RB Salzburg": dict(pais="Austria", elo=1485),
    "Sturm Graz": dict(pais="Austria", elo=1450),
    "LASK": dict(pais="Austria", elo=1415),
    "Rapid Wien": dict(pais="Austria", elo=1425),
    "Young Boys": dict(pais="Suiza", elo=1450),
    "FC Basel": dict(pais="Suiza", elo=1435),
    "Lugano": dict(pais="Suiza", elo=1415),
    "Servette": dict(pais="Suiza", elo=1410),
    "FC Copenhagen": dict(pais="Dinamarca", elo=1475),
    "Midtjylland": dict(pais="Dinamarca", elo=1465),
    "Brondby": dict(pais="Dinamarca", elo=1420),
    "Bodo/Glimt": dict(pais="Noruega", elo=1475),
    "Molde": dict(pais="Noruega", elo=1430),
    "Brann": dict(pais="Noruega", elo=1415),
    "Malmo FF": dict(pais="Suecia", elo=1435),
    "Djurgarden": dict(pais="Suecia", elo=1405),
    "Elfsborg": dict(pais="Suecia", elo=1405),
    "Dinamo Zagreb": dict(pais="Croacia", elo=1465),
    "Hajduk Split": dict(pais="Croacia", elo=1410),
    "Rijeka": dict(pais="Croacia", elo=1400),
    "Crvena Zvezda": dict(pais="Serbia", elo=1460),
    "Partizan": dict(pais="Serbia", elo=1405),
    "TSC Backa Topola": dict(pais="Serbia", elo=1365),
    "Slovan Bratislava": dict(pais="Eslovaquia", elo=1400),
    "Qarabag": dict(pais="Azerbaiyán", elo=1440),
    "Ferencvaros": dict(pais="Hungría", elo=1430),
    "Ludogorets": dict(pais="Bulgaria", elo=1415),
    "FCSB": dict(pais="Rumanía", elo=1425),
    "CFR Cluj": dict(pais="Rumanía", elo=1410),
    "Legia Warsaw": dict(pais="Polonia", elo=1435),
    "Lech Poznan": dict(pais="Polonia", elo=1420),
    "Jagiellonia": dict(pais="Polonia", elo=1415),
    "Maccabi Tel Aviv": dict(pais="Israel", elo=1430),
    "Maccabi Haifa": dict(pais="Israel", elo=1420),
    "APOEL": dict(pais="Chipre", elo=1400),
    "Omonia Nicosia": dict(pais="Chipre", elo=1390),
    "Pafos": dict(pais="Chipre", elo=1410),
    "Astana": dict(pais="Kazajistán", elo=1375),
    "Sheriff Tiraspol": dict(pais="Moldavia", elo=1380),
    "Olimpija Ljubljana": dict(pais="Eslovenia", elo=1385),
    "Celje": dict(pais="Eslovenia", elo=1375),
    # Sudamérica (Libertadores / Sudamericana fuera de Brasil y Argentina)
    "Independiente del Valle": dict(pais="Ecuador", elo=1475),
    "LDU Quito": dict(pais="Ecuador", elo=1470),
    "Barcelona SC": dict(pais="Ecuador", elo=1435),
    "Penarol": dict(pais="Uruguay", elo=1460),
    "Nacional": dict(pais="Uruguay", elo=1445),
    "Colo-Colo": dict(pais="Chile", elo=1450),
    "Universidad de Chile": dict(pais="Chile", elo=1440),
    "Palestino": dict(pais="Chile", elo=1410),
    "Olimpia": dict(pais="Paraguay", elo=1445),
    "Cerro Porteno": dict(pais="Paraguay", elo=1440),
    "Libertad": dict(pais="Paraguay", elo=1450),
    "Atletico Nacional": dict(pais="Colombia", elo=1450),
    "Millonarios": dict(pais="Colombia", elo=1435),
    "Junior": dict(pais="Colombia", elo=1435),
    "Bolivar": dict(pais="Bolivia", elo=1455),
    "The Strongest": dict(pais="Bolivia", elo=1435),
    "Universitario": dict(pais="Perú", elo=1425),
    "Alianza Lima": dict(pais="Perú", elo=1410),
    "Sporting Cristal": dict(pais="Perú", elo=1415),
}

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
    "🇫🇷 Francia · Ligue 1": dict(codigo="F1", fuente="csv", europa=True, fuerza=-40),
    "🇵🇹 Portugal · Primeira Liga": dict(codigo="P1", fuente="csv", europa=True, fuerza=-100),
    "🇳🇱 Países Bajos · Eredivisie": dict(codigo="N1", fuente="csv", europa=True, fuerza=-85),
    "🇧🇪 Bélgica · Pro League": dict(codigo="B1", fuente="csv", europa=True, fuerza=-95),
    "🇹🇷 Turquía · Süper Lig": dict(codigo="T1", fuente="csv", europa=True, fuerza=-150),
    "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Escocia · Premiership": dict(codigo="SC0", fuente="csv", europa=True, fuerza=-160),
    "🇬🇷 Grecia · Super League": dict(codigo="G1", fuente="csv", europa=True, fuerza=-145),
    "🇵🇪 Perú · Liga 1": dict(codigo="PER", fuente="espn", slug="per.1", europa=False, fuerza=-115),
    "🇧🇷 Brasil": dict(codigo="BRA", fuente="xlsx", europa=False, fuerza=0),
    "🇦🇷 Argentina": dict(codigo="ARG", fuente="xlsx", europa=False, fuerza=0),
    "🇯🇵 Japón": dict(codigo="JPN", fuente="xlsx", europa=False, fuerza=0),
    "🇨🇳 China": dict(codigo="CHN", fuente="xlsx", europa=False, fuerza=0),
    "🇺🇸 Estados Unidos": dict(codigo="USA", fuente="xlsx", europa=False, fuerza=0),
}

# 1 punto Elo de diferencia entre ligas ≈ 0.00115 en escala log de ataque/defensa (aproximación)
ELO_A_LOG = 0.00115

# ======================= MÓDULO DE SELECCIONES FIFA ===========================
URL_FIFA_RESULTS = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

COMPETENCIAS_FIFA = {
    "🏆 Copa Mundial de la FIFA": dict(
        slug="fifa.world", peso=1.60, importancia=60.0,
        factor_goles=1.00, mult_goles=1.00, factor_tarjetas=1.10, mult_tarjetas=1.10,
        neutral=True, neutral_defecto=True,
    ),
    "🌎 Eliminatorias CONMEBOL (Sudamérica)": dict(
        slug="fifa.worldq.conmebol", peso=1.35, importancia=25.0,
        factor_goles=0.95, mult_goles=0.95, factor_tarjetas=1.22, mult_tarjetas=1.22,
        neutral=False, neutral_defecto=False,
    ),
    "🇪🇺 Eliminatorias UEFA (Europa)": dict(
        slug="fifa.worldq.uefa", peso=1.30, importancia=25.0,
        factor_goles=1.03, mult_goles=1.03, factor_tarjetas=1.00, mult_tarjetas=1.00,
        neutral=False, neutral_defecto=False,
    ),
    "🌎 Eliminatorias CONCACAF (Norte/Centroamérica)": dict(
        slug="fifa.worldq.concacaf", peso=1.25, importancia=25.0,
        factor_goles=1.00, mult_goles=1.00, factor_tarjetas=1.10, mult_tarjetas=1.10,
        neutral=False, neutral_defecto=False,
    ),
    "🌍 Eliminatorias CAF (África)": dict(
        slug="fifa.worldq.caf", peso=1.25, importancia=25.0,
        factor_goles=0.93, mult_goles=0.93, factor_tarjetas=1.12, mult_tarjetas=1.12,
        neutral=False, neutral_defecto=False,
    ),
    "🌏 Eliminatorias AFC (Asia)": dict(
        slug="fifa.worldq.afc", peso=1.25, importancia=25.0,
        factor_goles=1.02, mult_goles=1.02, factor_tarjetas=0.98, mult_tarjetas=0.98,
        neutral=False, neutral_defecto=False,
    ),
    "🏆 Copa América (CONMEBOL)": dict(
        slug="conmebol.america", peso=1.45, importancia=50.0,
        factor_goles=0.95, mult_goles=0.95, factor_tarjetas=1.25, mult_tarjetas=1.25,
        neutral=True, neutral_defecto=True,
    ),
    "🏆 Eurocopa (UEFA Euro)": dict(
        slug="uefa.euro", peso=1.45, importancia=50.0,
        factor_goles=0.98, mult_goles=0.98, factor_tarjetas=1.05, mult_tarjetas=1.05,
        neutral=True, neutral_defecto=True,
    ),
    "🛡️ UEFA Nations League": dict(
        slug="uefa.nations", peso=1.05, importancia=15.0,
        factor_goles=1.00, mult_goles=1.00, factor_tarjetas=1.02, mult_tarjetas=1.02,
        neutral=False, neutral_defecto=False,
    ),
    "🏆 Copa Oro (CONCACAF Gold Cup)": dict(
        slug="concacaf.gold", peso=1.30, importancia=35.0,
        factor_goles=1.02, mult_goles=1.02, factor_tarjetas=1.08, mult_tarjetas=1.08,
        neutral=True, neutral_defecto=True,
    ),
    "🛡️ CONCACAF Nations League": dict(
        slug="concacaf.nations.league", peso=1.00, importancia=15.0,
        factor_goles=1.02, mult_goles=1.02, factor_tarjetas=1.05, mult_tarjetas=1.05,
        neutral=False, neutral_defecto=False,
    ),
    "🏆 Copa Africana de Naciones (AFCON)": dict(
        slug="caf.nations", peso=1.40, importancia=35.0,
        factor_goles=0.91, mult_goles=0.91, factor_tarjetas=1.15, mult_tarjetas=1.15,
        neutral=True, neutral_defecto=True,
    ),
    "🏆 Copa Asiática (AFC Asian Cup)": dict(
        slug="afc.cup", peso=1.35, importancia=35.0,
        factor_goles=0.98, mult_goles=0.98, factor_tarjetas=1.00, mult_tarjetas=1.00,
        neutral=True, neutral_defecto=True,
    ),
    "🤝 Amistoso Internacional FIFA": dict(
        slug="fifa.friendly", peso=0.50, importancia=10.0,
        factor_goles=1.05, mult_goles=1.05, factor_tarjetas=0.65, mult_tarjetas=0.65,
        neutral=False, neutral_defecto=False,
    ),
}

# Altitud en metros sobre el nivel del mar de las sedes habituales de Eliminatorias FIFA
ALTITUD_SELECCIONES = {
    "Bolivia": 3850,       # El Alto (4,150 m) / La Paz (3,640 m)
    "Ecuador": 2850,       # Quito (Rodrigo Paz Delgado)
    "Mexico": 2240,        # Ciudad de México (Estadio Azteca)
    "Colombia": 1300,      # Barranquilla (calor extremo 30 m) / Bogotá (2,640 m)
    "Ethiopia": 2355,      # Adís Abeba
    "Kenya": 1795,         # Nairobi
    "South Africa": 1753,  # Johannesburgo
    "Guatemala": 1500,     # Ciudad de Guatemala
    "Costa Rica": 1172,    # San José
    "Honduras": 1120,      # Tegucigalpa / San Pedro Sula
    "Peru": 500,           # Lima (plantel parcialmente aclimatado a altura andina)
}
CALOR_SELECCIONES = {
    "Colombia", "Honduras", "Panama", "Saudi Arabia", "Qatar",
    "United Arab Emirates", "Nigeria", "Ivory Coast", "Senegal", "Cameroon",
}

# Puntos de referencia del Ranking FIFA oficial (escala SUM de la FIFA) para anclaje de jerarquía
RANKING_FIFA_PUNTOS = {
    "Argentina": 1889, "France": 1851, "Spain": 1857, "England": 1813, "Brazil": 1772,
    "Belgium": 1768, "Portugal": 1752, "Netherlands": 1747, "Italy": 1726, "Colombia": 1738,
    "Germany": 1713, "Uruguay": 1701, "Croatia": 1699, "Morocco": 1676, "Japan": 1645,
    "United States": 1632, "Mexico": 1640, "Senegal": 1627, "Iran": 1622, "Switzerland": 1625,
    "Denmark": 1612, "South Korea": 1589, "Australia": 1552, "Ukraine": 1548, "Turkey": 1545,
    "Ecuador": 1536, "Austria": 1532, "Sweden": 1522, "Poland": 1525, "Hungary": 1518,
    "Egypt": 1515, "Canada": 1512, "Nigeria": 1503, "Serbia": 1498, "Algeria": 1496,
    "Ivory Coast": 1495, "Scotland": 1490, "Peru": 1488, "Russia": 1486, "Chile": 1485,
    "Venezuela": 1482, "Czech Republic": 1480, "Romania": 1479, "Paraguay": 1475, "Norway": 1473,
    "Greece": 1471, "Panama": 1470, "Cameroon": 1468, "Costa Rica": 1465, "Slovakia": 1460,
    "Mali": 1455, "Tunisia": 1450, "Qatar": 1448, "Wales": 1446, "Iraq": 1442,
    "Saudi Arabia": 1435, "Uzbekistan": 1432, "South Africa": 1428, "Slovenia": 1425,
    "Jamaica": 1418, "Jordan": 1415, "Georgia": 1412, "Republic of Ireland": 1410,
    "Bolivia": 1408, "Finland": 1402, "Albania": 1398, "Burkina Faso": 1395, "DR Congo": 1392,
    "Ghana": 1390, "Honduras": 1385, "Bosnia and Herzegovina": 1380, "North Macedonia": 1375,
    "Montenegro": 1370, "Iceland": 1365, "Northern Ireland": 1360, "Guinea": 1358,
    "Cape Verde": 1355, "Israel": 1352, "Oman": 1348, "Bahrain": 1342, "El Salvador": 1338,
    "Bulgaria": 1335, "Haiti": 1330, "Curacao": 1326, "New Zealand": 1325, "China PR": 1320,
    "Equatorial Guinea": 1318, "Zambia": 1315, "Uganda": 1310, "Syria": 1305, "Guatemala": 1300,
    "Trinidad and Tobago": 1295, "Palestine": 1290, "Thailand": 1285, "Tajikistan": 1280,
}


def temporadas_recientes(n):
    """Últimas n temporadas europeas calculadas desde hoy. Con n=2 hoy: ['2526', '2627']."""
    hoy = date.today()
    inicio = hoy.year if hoy.month >= 8 else hoy.year - 1
    return [f"{(inicio - k) % 100:02d}{(inicio - k + 1) % 100:02d}" for k in range(n - 1, -1, -1)]
