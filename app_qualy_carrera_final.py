import os
import pickle
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(layout="wide")

PONDERACIONES = [0, 25, 18, 15, 12, 10, 8, 6, 4]
PUNTOS_F1 = [10, 8, 6, 5, 4, 3, 2, 1]
URL_GPT = "https://chatgpt.com/g/g-6824ad0751a481918919c57574db29b3-narrador-f1"

# --- Cargar datos y modelos ---

def arreglar_texto_mojibake(txt):
    if pd.isna(txt):
        return txt
    txt = str(txt).strip()
    # Intenta reparar textos mal decodificados tipo "AutÃ³dromo" o "NÃ¼rburgring"
    try:
        txt = txt.encode("latin1").decode("utf-8")
    except UnicodeError:
        pass
    # Normaliza los caracteres Unicode para evitar rarezas visuales
    txt = unicodedata.normalize("NFC", txt)
    # Por si alguna variante sigue viniendo rota
    reemplazos = {
        "AutÃ³dromo Hermanos RodrÃ­guez": "Autódromo Hermanos Rodríguez",
        "AutÃ³dromo Internacional Nelson Piquet": "Autódromo Internacional Nelson Piquet",
        "AutÃ³dromo Internacional do Algarve": "Autódromo Internacional do Algarve",
        "AutÃ³dromo JosÃ© Carlos Pace": "Autódromo José Carlos Pace",
        "AutÃ³dromo Juan y Oscar GÃ¡lvez": "Autódromo Juan y Oscar Gálvez",
        "AutÃ³dromo do Estoril": "Autódromo do Estoril",
        "NÃ¼rburgring": "Nürburgring",
    }
    txt = reemplazos.get(txt, txt)
    # Limpieza final de espacios raros
    txt = " ".join(txt.split())
    return txt

@st.cache_data
def cargar_datos():
    df_race = pd.read_csv("df2.csv", parse_dates=["RACE_DATE"])
    df_qualy = pd.read_csv("df_qualy2.csv", parse_dates=["GP_DATE"])
    df_drivers = pd.read_csv("df_drivers.csv")
    df_constructors = pd.read_csv("df_constructors.csv")
    df_race["CIRCUIT_NAME"] = df_race["CIRCUIT_NAME"].apply(arreglar_texto_mojibake)
    df_qualy["CIRCUIT_NAME"] = df_qualy["CIRCUIT_NAME"].apply(arreglar_texto_mojibake)
    # Año precalculado para no repetir .dt.year en cada filtro
    df_race["YEAR"] = df_race["RACE_DATE"].dt.year
    df_qualy["YEAR"] = df_qualy["GP_DATE"].dt.year
    return df_race, df_qualy, df_drivers, df_constructors

@st.cache_resource
def cargar_modelos():
    with open("DetGrid2.pkl", "rb") as f:
        model_qualy = pickle.load(f)
    with open("formula1_top88_model.pkl", "rb") as f:
        model_carrera = pickle.load(f)
    return model_qualy, model_carrera

df_race, df_qualy, df_drivers, df_constructors = cargar_datos()
model_qualy, model_carrera = cargar_modelos()

driver_id_por_ref = df_drivers.drop_duplicates("DRIVERREF").set_index("DRIVERREF")["DRIVER_ID"].to_dict()
constructor_id_por_nombre = (
    df_constructors.dropna(subset=["NAME#"])
    .drop_duplicates("NAME#")
    .set_index("NAME#")["CONSTRUCTOR_ID"]
    .to_dict()
)

@st.cache_data
def calcular_anios_validos(col_id):
    """
    {id: [años]} con los años en los que hay datos tanto de carrera como de qualy.
    col_id: "DRIVER_ID" o "CONSTRUCTOR_ID".
    """
    anios_race = df_race.groupby(col_id)["YEAR"].agg(set)
    anios_qualy = df_qualy.groupby(col_id)["YEAR"].agg(set)
    comunes = anios_race.index.intersection(anios_qualy.index)
    return {k: sorted(int(a) for a in anios_race[k] & anios_qualy[k]) for k in comunes}

@st.cache_data
def preparar_tablas():
    """
    Tablas de consulta precalculadas para no filtrar los DataFrames completos en cada simulación.
    """
    race = df_race.sort_values("RACE_DATE", kind="stable")
    qualy = df_qualy.sort_values("GP_DATE", kind="stable")
    tablas = {}
    # Última carrera/qualy de cada piloto y escudería en cada año
    for nombre, df, col_id in [
        ("race_piloto", race, "DRIVER_ID"),
        ("race_ct", race, "CONSTRUCTOR_ID"),
        ("qualy_piloto", qualy, "DRIVER_ID"),
        ("qualy_ct", qualy, "CONSTRUCTOR_ID"),
    ]:
        tablas[nombre] = df.groupby([col_id, "YEAR"]).tail(1).set_index([col_id, "YEAR"])
    # Afinidad piloto-circuito: primera aparición en el fichero
    for nombre, df, col in [("afinidad_race", df_race, "AFINIDAD3"), ("afinidad_qualy", df_qualy, "AFINIDAD_QUALY_3")]:
        df = df.assign(CIRCUIT_LOWER=df["CIRCUIT_NAME"].str.lower())
        tablas[nombre] = df.drop_duplicates(["DRIVER_ID", "CIRCUIT_LOWER"]).set_index(["DRIVER_ID", "CIRCUIT_LOWER"])[col]
    # Afinidad en lluvia real: primera carrera con lluvia de ese piloto en ese año
    lluvia = df_race[
        (df_race["LLUVIA"] == 1) &
        (df_race["AFINIDAD_LLUVIA_REAL"].notna()) &
        (df_race["AFINIDAD_LLUVIA_REAL"] > 0)
    ]
    tablas["afinidad_lluvia"] = lluvia.drop_duplicates(["DRIVER_ID", "YEAR"]).set_index(["DRIVER_ID", "YEAR"])["AFINIDAD_LLUVIA_REAL"]
    return tablas

tablas = preparar_tablas()

def buscar_fila(tabla, clave, descripcion):
    try:
        return tabla.loc[clave]
    except KeyError:
        raise KeyError(f"no hay datos de {descripcion}") from None

# --- Inicializar estado ---
if 'poles' not in st.session_state:
    st.session_state.poles = {}  # piloto: número de poles
if 'victorias' not in st.session_state:
    st.session_state.victorias = {}  # piloto: número de victorias
if 'pilotos_config' not in st.session_state:
    st.session_state.pilotos_config = {}
if 'calendario' not in st.session_state:
    st.session_state.calendario = []
if 'resultados_mundial' not in st.session_state:
    st.session_state.resultados_mundial = {}
if 'carrera_actual' not in st.session_state:
    st.session_state.carrera_actual = 0
if 'historial_carreras' not in st.session_state:
    st.session_state.historial_carreras = []

def generar_prompt_presentacion(pilotos_config):
    texto = "### Presentación del Campeonato de F1 Simulado\n\n"
    texto += "Resumen de los 8 binomios seleccionados y sus expectativas:\n\n"
    for piloto, config in pilotos_config.items():
        anio_piloto = config["anio_piloto"]
        constructor = config["constructor"]
        anio_constructor = config["anio_constructor"]
        nombre_formateado = piloto.replace("_", " ").title()
        texto += f"- {nombre_formateado} ({anio_piloto}) con el {constructor} del {anio_constructor}\n"
    texto += "\nEscribe una presentación con tono narrativo deportivo: destaca quiénes parten como favoritos, quiénes podrían sorprender, y qué combinaciones parecen más arriesgadas o desequilibradas. Añade emoción y expectativas para la temporada."
    return texto

# --- Función: generar features para clasificación (Qualy) ---

def generar_features_qualy(pilotos_config, circuito):
    rows = []
    for piloto, config in pilotos_config.items():
        if not piloto or not config['constructor']:
            continue
        try:
            driver_id = driver_id_por_ref[piloto]
            constructor_id = constructor_id_por_nombre[config['constructor']]
            driver_data = buscar_fila(tablas["qualy_piloto"], (driver_id, config['anio_piloto']),
                                      f"qualy de {piloto} en {config['anio_piloto']}")
            constructor_data = buscar_fila(tablas["qualy_ct"], (constructor_id, config['anio_constructor']),
                                           f"qualy de {config['constructor']} en {config['anio_constructor']}")
            afinidad_qualy = tablas["afinidad_qualy"].get((driver_id, circuito.lower()), 0.5)
            rows.append({
                'PILOTO': piloto,
                'QUALY_PERFORMANCE': driver_data['QUALY_PERFORMANCE'],
                'CT_QUALY_PERFORMANCE': constructor_data['CT_QUALY_PERFORMANCE'],
                'AFINIDAD_QUALY_3': afinidad_qualy
            })
        except Exception as e:
            st.warning(f"⚠️ Error al procesar {piloto}: {e}")
            continue
    return pd.DataFrame(rows)

def generar_prompt_para_gpt(carrera):
    circuito = carrera["circuito"]
    lluvia = carrera["lluvia"]
    resultados = carrera["resultados"]
    clasif_antes = carrera["clasificacion_antes"]
    clasif_despues = carrera["clasificacion_despues"]
    poleman = sorted(resultados, key=lambda x: x["salida"])[0]["piloto"]
    ganador = sorted(resultados, key=lambda x: x["llegada"])[0]["piloto"]
    resumen = f"Resumen del GP de {circuito}\n\n"
    resumen += f"Condiciones: {'Lluvia' if lluvia else 'Seco'}\n"
    resumen += f"Pole position: {poleman}\n"
    resumen += f"Ganador: {ganador}\n\n"
    resumen += "Resultados:\n"
    for r in resultados:
        resumen += f"- {r['piloto']}: salía {r['salida']}º, terminó {r['llegada']}º, +{r['puntos_f1']} pts\n"
    resumen += "\nClasificación antes de la carrera:\n"
    for i, (piloto, pts) in enumerate(sorted(clasif_antes.items(), key=lambda x: x[1], reverse=True), 1):
        resumen += f"{i}. {piloto} — {pts} pts\n"
    resumen += "\nClasificación después de la carrera:\n"
    for i, (piloto, pts) in enumerate(sorted(clasif_despues.items(), key=lambda x: x[1], reverse=True), 1):
        resumen += f"{i}. {piloto} — {pts} pts\n"
    resumen += "\nEscribe una crónica deportiva con estilo narrativo profesional, destacando lo más relevante y comentando los cambios en la clasificación."
    return resumen

# --- Función: simular carrera con ajustes ---

def simular_carrera(pilotos_config, circuito, lluvia, grid_ordenado, neutro=False):
    rows = []
    for pos, piloto in enumerate(grid_ordenado, start=1):
        try:
            config = pilotos_config[piloto]
            anio_piloto = int(config['anio_piloto'])  # 🔐 asegurar tipo
            anio_constructor = int(config['anio_constructor'])
            driver_id = driver_id_por_ref[piloto]
            constructor_id = constructor_id_por_nombre[config['constructor']]
            driver_data = buscar_fila(tablas["race_piloto"], (driver_id, anio_piloto),
                                      f"carrera de {piloto} en {anio_piloto}")
            constructor_data = buscar_fila(tablas["race_ct"], (constructor_id, anio_constructor),
                                           f"carrera de {config['constructor']} en {anio_constructor}")
            afinidad = tablas["afinidad_race"].get((driver_id, circuito.lower()), 0.5)
            # Afinidad con lluvia EN ESE AÑO
            afinidad_lluvia_real = tablas["afinidad_lluvia"].get((driver_id, anio_piloto), 0)
            rows.append({
                'PILOTO': piloto,
                'GRID': pos,
                'PREV_POSITIONS': 10,
                'ADJUSTED_PERFORMANCE': driver_data['ADJUSTED_PERFORMANCE'],
                'GENERAL_PERFORMANCE': driver_data['GENERAL_PERFORMANCE'],
                'PERFORMANCE_RATIO': driver_data['PERFORMANCE_RATIO'],
                'CONSTRUCTOR_PERFORMANCE': constructor_data['CONSTRUCTOR_PERFORMANCE'],
                'AFINIDAD3': afinidad,
                'AFINIDAD_LLUVIA_REAL': afinidad_lluvia_real,
                'LLUVIA': lluvia
            })
        except Exception as e:
            st.warning(f"⚠️ Error con {piloto}: {e}")
            continue
    if not rows:
        return []
    df_features = pd.DataFrame(rows)
    X = df_features.drop(columns="PILOTO")
    probs = model_carrera.predict_proba(X)
    pesos_grid_por_circuito = {
        "Circuit de Monaco": 5, "Hungaroring": 1, "Autodromo Enzo e Dino Ferrari": 0.5,
        "Circuit de Barcelona-Catalunya": 0.5, "Albert Park Grand Prix Circuit": 0.5,
        "Circuit Gilles Villeneuve": 0.5
    }
    peso_grid = pesos_grid_por_circuito.get(circuito, 0.1)
    factor_global_grid = 1.55
    resultados = []
    for i, row in df_features.iterrows():
        base_score = sum(PONDERACIONES[j] * probs[i][j] for j in range(len(PONDERACIONES)))
        if neutro:
            factor = 1
        else:
            if row['LLUVIA']:
                factor = (
                    1 + 0.05 * row['CONSTRUCTOR_PERFORMANCE']
                    + 0.05 * row['AFINIDAD3']
                    + 0.13 * row['AFINIDAD_LLUVIA_REAL']
                    + 0.05 * row['PERFORMANCE_RATIO']
                )
            else:
                factor = (
                    1 + 0.12 * row['CONSTRUCTOR_PERFORMANCE']
                    + 0.05 * row['AFINIDAD3']
                    + 0.065 * row['PERFORMANCE_RATIO']
                    + factor_global_grid * (peso_grid / row['GRID'])
                )
        final_score = base_score * factor
        resultados.append((final_score, row['PILOTO']))
    resultados.sort(reverse=True)
    return resultados

st.markdown("""
<style>
/* Si pruebas una fuente nueva en las variables de abajo, añádela también a este @import */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Jura:wght@400;600;700&family=Oxanium:wght@400;600;700&family=Bebas+Neue&family=Space+Grotesk:wght@400;500;700&display=swap');
/* =========================================================
   CAMBIA SOLO ESTAS 3 LÍNEAS PARA PROBAR FUENTES
   ========================================================= */
:root {
    --font_titulos: 'Orbitron', sans-serif;
    --font_texto: 'Jura', sans-serif;
    --font_labels: 'Oxanium', sans-serif;
}
/* =========================================================
   ANCHO GENERAL DE LA APP
   ========================================================= */
.block-container {
    max-width: 76% !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
}
/* Fondo general */
.stApp {
    background: #0f172a;
}
/* Texto base */
html, body, [class*="css"] {
    color: #e2e8f0 !important;
    font-family: var(--font_texto) !important;
}
/* Títulos principales */
h1, h2, h3 {
    color: #f1f5f9 !important;
    font-family: var(--font_titulos) !important;
    letter-spacing: 0.3px;
}
/* Texto normal */
p, li, div[data-testid="stMarkdownContainer"] p, div[data-testid="stMarkdownContainer"] li {
    color: #f8fafc !important;
    font-family: var(--font_texto) !important;
}
/* Labels de widgets */
label, .stRadio label, .stSelectbox label, .stNumberInput label, .stTextInput label {
    color: #f8fafc !important;
    font-family: var(--font_labels) !important;
    font-weight: 600 !important;
}
/* Caption / texto secundario */
small, span, .stCaption {
    color: #cbd5e1 !important;
    font-family: var(--font_texto) !important;
}
/* Botones */
.stButton > button {
    background: linear-gradient(90deg, #ff4b4b 0%, #ff7b72 100%);
    color: #ffffff !important;
    border-radius: 10px;
    border: none !important;
    font-weight: 700 !important;
    font-family: var(--font_labels) !important;
    padding: 0.55rem 1rem !important;
}
.stButton > button:hover {
    filter: brightness(1.05);
}
/* SELECTBOX CERRADO */
div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background-color: #f1f5f9 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    min-height: 44px !important;
}

/* Texto y valor seleccionado */
div[data-testid="stSelectbox"] [data-baseweb="select"],
div[data-testid="stSelectbox"] [data-baseweb="select"] *,
div[data-testid="stSelectbox"] [data-baseweb="select"] span {
    color: #1e293b !important;
    -webkit-text-fill-color: #1e293b !important;
    fill: #1e293b !important;
    opacity: 1 !important;
}

/* Campo de búsqueda del selector */
div[data-testid="stSelectbox"] [data-baseweb="select"] input {
    color: #1e293b !important;
    -webkit-text-fill-color: #1e293b !important;
    caret-color: #1e293b !important;
}

/* Single value / valor seleccionado */
div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="singleValue"],
div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="ValueContainer"],
div[data-testid="stSelectbox"] [data-baseweb="select"] [data-testid="stMarkdownContainer"],
div[data-testid="stSelectbox"] [data-baseweb="select"] span {
    color: #1e293b !important;
    -webkit-text-fill-color: #1e293b !important;
    opacity: 1 !important;
}

/* =========================================================
   DROPDOWN ABIERTO / MENÚ DE OPCIONES
   ========================================================= */
/* Popover raíz */
body [data-baseweb="popover"] {
    background: transparent !important;
}
/* Caja interna del popover */
body [data-baseweb="popover"] > div {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35) !important;
}
/* Menú */
body [data-baseweb="menu"] {
    background-color: #1e293b !important;
}
/* Listbox */
body ul[role="listbox"] {
    background-color: #1e293b !important;
    color: #ffffff !important;
    border-radius: 10px !important;
}
/* Opciones */
body li[role="option"],
body ul[role="listbox"] li,
body [role="listbox"] [role="option"] {
    background-color: #1e293b !important;
    color: #f8fafc !important;
    font-family: var(--font_texto) !important;
}
/* Todo lo de dentro de cada opción */
body li[role="option"] *,
body ul[role="listbox"] li *,
body [role="listbox"] [role="option"] * {
    color: #f8fafc !important;
    fill: #f8fafc !important;
    font-family: var(--font_texto) !important;
}
/* Hover */
body li[role="option"]:hover,
body ul[role="listbox"] li:hover,
body [role="listbox"] [role="option"]:hover {
    background-color: #334155 !important;
    color: #ffffff !important;
}
/* Seleccionado */
body li[role="option"][aria-selected="true"],
body ul[role="listbox"] li[aria-selected="true"],
body [role="listbox"] [role="option"][aria-selected="true"] {
    background-color: #ff4b4b !important;
    color: #ffffff !important;
}
/* Texto del piloto, año y escudería seleccionados */
div[data-testid="stSelectbox"] input[role="combobox"] {
    color: #1e293b !important;
    -webkit-text-fill-color: #1e293b !important;
    caret-color: #1e293b !important;
    opacity: 1 !important;
}
/* =========================================================
   NUMBER INPUT
   ========================================================= */
.stNumberInput input {
    background-color: #1e293b !important;
    color: #ffffff !important;
    border: 1px solid #334155 !important;
    font-family: var(--font_texto) !important;
}
/* Text input / text area */
.stTextInput input,
.stTextArea textarea {
    background-color: #1e293b !important;
    color: #ffffff !important;
    border: 1px solid #334155 !important;
    font-family: var(--font_texto) !important;
}
/* Dataframes */
[data-testid="stDataFrame"] {
    background-color: #1e293b !important;
    border-radius: 10px !important;
}
/* Alerts */
.stAlert {
    background-color: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
    font-family: var(--font_texto) !important;
}
/* Si quieres una zona concreta con otra fuente */
.zona_especial h1,
.zona_especial h2,
.zona_especial h3 {
    font-family: 'Bebas Neue', sans-serif !important;
}
.zona_especial p,
.zona_especial span,
.zona_especial div {
    font-family: 'Space Grotesk', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* TEXTO DE RESULTADOS (CLAVE) */
div[data-testid="stMarkdownContainer"] p {
    color: #f8fafc !important;
    font-size: 16px;
}
/* LISTAS (P1, P2, etc) */
div[data-testid="stMarkdownContainer"] li {
    color: #f8fafc !important;
    font-size: 16px;
}
/* TEXTO DENTRO DE st.write */
div[data-testid="stText"] {
    color: #f8fafc !important;
}
/* Ajuste para textos secundarios (tiempos) */
small, span {
    color: #cbd5e1 !important;
}
/* Resultado destacado (ganador) */
strong {
    color: #ffffff !important;
}
div[data-testid="stSelectbox"] {
    max-width: 260px !important;
}
h2 {
    font-size: 26px !important;
}
.frase-senna {
    font-family: 'Orbitron', sans-serif;
    font-size: 37px;
    text-align: center;
    color: #f5a627;
    font-style: italic;
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

def cargar_8_pilotos_aleatorios():
    rng = np.random.default_rng()
    anios_por_piloto = calcular_anios_validos("DRIVER_ID")
    anios_por_constructor = calcular_anios_validos("CONSTRUCTOR_ID")
    # Pilotos y escuderías con datos válidos tanto en race como en qualy
    pilotos_validos = sorted(df_drivers.loc[df_drivers['DRIVER_ID'].isin(anios_por_piloto), 'DRIVERREF'].dropna().unique().tolist())
    escuderias_validas = sorted(df_constructors.loc[df_constructors['CONSTRUCTOR_ID'].isin(anios_por_constructor), 'NAME#'].dropna().unique().tolist())
    if len(pilotos_validos) < 8:
        st.error("No hay suficientes pilotos válidos para generar una selección aleatoria.")
        return
    pilotos_elegidos = rng.choice(pilotos_validos, size=8, replace=False)
    for i, piloto in enumerate(pilotos_elegidos):
        # Guardar piloto en el widget
        st.session_state[f"piloto_select_{i}"] = piloto
        anios_piloto_validos = anios_por_piloto.get(driver_id_por_ref[piloto], [])
        if not anios_piloto_validos:
            continue
        anio_piloto = int(rng.choice(anios_piloto_validos))
        st.session_state[f"anio_piloto_select_{i}"] = anio_piloto
        # Escudería aleatoria válida
        constructor = rng.choice(escuderias_validas)
        st.session_state[f"constructor_select_{i}"] = constructor
        anios_constructor_validos = anios_por_constructor.get(constructor_id_por_nombre[constructor], [])
        if not anios_constructor_validos:
            continue
        anio_constructor = int(rng.choice(anios_constructor_validos))
        st.session_state[f"anio_constructor_select_{i}"] = anio_constructor

@st.cache_data
def construir_pool_equilibrado():
    """
    Pool piloto-año con métricas agregadas.
    """
    claves = ["DRIVER_ID", "YEAR"]
    race_last = tablas["race_piloto"]
    qualy_last = tablas["qualy_piloto"]
    race_g = df_race.groupby(claves)
    qualy_g = df_qualy.groupby(claves)
    lluvia_validas = df_race[df_race['AFINIDAD_LLUVIA_REAL'].notna() & (df_race['AFINIDAD_LLUVIA_REAL'] > 0)]
    pool = pd.DataFrame({
        "GENERAL_PERFORMANCE": race_last['GENERAL_PERFORMANCE'],
        "ADJUSTED_PERFORMANCE": race_last['ADJUSTED_PERFORMANCE'],
        "PERFORMANCE_RATIO": race_last['PERFORMANCE_RATIO'],
        "media_afinidad_race": race_g['AFINIDAD3'].mean(),
        "media_afinidad_lluvia": lluvia_validas.groupby(claves)['AFINIDAD_LLUVIA_REAL'].mean(),
        "n_circuitos_race": race_g['CIRCUIT_NAME'].nunique(),
    }).join(pd.DataFrame({
        "QUALY_PERFORMANCE": qualy_last['QUALY_PERFORMANCE'],
        "media_afinidad_qualy": qualy_g['AFINIDAD_QUALY_3'].mean(),
        "n_circuitos_qualy": qualy_g['CIRCUIT_NAME'].nunique(),
    }), how="inner")
    pool = pool.reset_index().rename(columns={"DRIVER_ID": "driver_id", "YEAR": "anio_piloto"})
    ref_por_driver_id = df_drivers.drop_duplicates("DRIVER_ID").set_index("DRIVER_ID")["DRIVERREF"]
    pool = pool[pool["driver_id"].isin(ref_por_driver_id.index)].copy()
    pool["piloto"] = pool["driver_id"].map(ref_por_driver_id)
    pool["anio_piloto"] = pool["anio_piloto"].astype(int)
    medias = ["media_afinidad_race", "media_afinidad_qualy", "media_afinidad_lluvia"]
    pool[medias] = pool[medias].fillna(0.0)
    pool[["n_circuitos_race", "n_circuitos_qualy"]] = pool[["n_circuitos_race", "n_circuitos_qualy"]].astype(int)
    return pool[[
        "piloto", "driver_id", "anio_piloto",
        "GENERAL_PERFORMANCE", "ADJUSTED_PERFORMANCE", "PERFORMANCE_RATIO", "QUALY_PERFORMANCE",
        "media_afinidad_race", "media_afinidad_qualy", "media_afinidad_lluvia",
        "n_circuitos_race", "n_circuitos_qualy"
    ]].reset_index(drop=True)

@st.cache_data
def construir_pool_constructores():
    """
    Pool constructor-año con métricas agregadas.
    """
    pool = tablas["race_ct"][['CONSTRUCTOR_PERFORMANCE']].join(
        tablas["qualy_ct"][['CT_QUALY_PERFORMANCE']], how="inner"
    ).reset_index()
    nombre_por_ct_id = df_constructors.drop_duplicates("CONSTRUCTOR_ID").set_index("CONSTRUCTOR_ID")["NAME#"]
    pool = pool[pool["CONSTRUCTOR_ID"].isin(nombre_por_ct_id.index)].copy()
    pool["constructor"] = pool["CONSTRUCTOR_ID"].map(nombre_por_ct_id)
    pool["anio_constructor"] = pool["YEAR"].astype(int)
    return pool[["constructor", "anio_constructor", "CONSTRUCTOR_PERFORMANCE", "CT_QUALY_PERFORMANCE"]].reset_index(drop=True)

def construir_combinaciones(pool_pilotos, pool_ct):
    """
    Todas las combinaciones piloto-coche con su score equilibrado.
    """
    df_combo = pool_pilotos[['piloto', 'anio_piloto', 'score_piloto']].merge(
        pool_ct[['constructor', 'anio_constructor', 'score_coche']], how="cross"
    )
    sp = df_combo['score_piloto']
    sc = df_combo['score_coche']
    # Score base combinación
    score_combo = 0.65 * sp + 0.35 * sc
    # Penalización si ambos están demasiado arriba a la vez
    penalizacion_top = 0.18 * ((sp - 0.75).clip(lower=0) + (sc - 0.75).clip(lower=0))
    # Penalización si ambos están demasiado abajo a la vez
    penalizacion_bajo = 0.10 * ((0.45 - sp).clip(lower=0) + (0.45 - sc).clip(lower=0))
    # Penalización por desequilibrio exagerado entre piloto y coche
    penalizacion_gap = 0.12 * ((sp - sc).abs() - 0.22).clip(lower=0)
    df_combo['score_combo'] = score_combo - penalizacion_top - penalizacion_bajo - penalizacion_gap
    return df_combo

def normalizar_columna(df, col):
    min_v = df[col].min()
    max_v = df[col].max()
    if max_v == min_v:
        return pd.Series([0.5] * len(df), index=df.index)
    return (df[col] - min_v) / (max_v - min_v)

def elegir_sin_repetir(candidatos, seleccion, pilotos_usados, constructores_usados, n=8):
    for _, row in candidatos.iterrows():
        if row['piloto'] in pilotos_usados:
            continue
        if row['constructor'] in constructores_usados:
            continue
        seleccion.append(row)
        pilotos_usados.add(row['piloto'])
        constructores_usados.add(row['constructor'])
        if len(seleccion) == n:
            break

def cargar_8_pilotos_equilibrados():
    rng = np.random.default_rng()
    pool_pilotos = construir_pool_equilibrado()
    pool_ct = construir_pool_constructores()
    if pool_pilotos.empty or pool_ct.empty:
        st.error("No se ha podido construir el pool equilibrado.")
        return
    # =========================================================
    # 1) FILTRO PILOTOS: ÉLITE EQUILIBRADA
    # =========================================================
    # Capar outliers absurdos
    pool_pilotos['PERFORMANCE_RATIO_CLIP'] = pool_pilotos['PERFORMANCE_RATIO'].clip(lower=0, upper=22)
    pool_pilotos = pool_pilotos[
        (pool_pilotos['n_circuitos_race'] >= 10) &
        (pool_pilotos['n_circuitos_qualy'] >= 10) &
        (pool_pilotos['GENERAL_PERFORMANCE'] >= 2.5) &
        (pool_pilotos['GENERAL_PERFORMANCE'] <= 6.5) &
        (pool_pilotos['ADJUSTED_PERFORMANCE'] >= 0.9) &
        (pool_pilotos['ADJUSTED_PERFORMANCE'] <= 1.7) &
        (pool_pilotos['PERFORMANCE_RATIO_CLIP'] >= 10) &
        (pool_pilotos['PERFORMANCE_RATIO_CLIP'] <= 22) &
        (pool_pilotos['QUALY_PERFORMANCE'] >= 30) &
        (pool_pilotos['QUALY_PERFORMANCE'] <= 110) &
        (pool_pilotos['media_afinidad_race'] >= 1.5) &
        (pool_pilotos['media_afinidad_qualy'] >= 2.0)
    ].copy()
    if len(pool_pilotos) < 8:
        st.error("No hay suficientes pilotos en la banda élite equilibrada.")
        return
    # Normalización piloto
    pool_pilotos['gp_norm'] = normalizar_columna(pool_pilotos, 'GENERAL_PERFORMANCE')
    pool_pilotos['adj_norm'] = normalizar_columna(pool_pilotos, 'ADJUSTED_PERFORMANCE')
    pool_pilotos['pr_norm'] = normalizar_columna(pool_pilotos, 'PERFORMANCE_RATIO_CLIP')
    pool_pilotos['qp_norm'] = normalizar_columna(pool_pilotos, 'QUALY_PERFORMANCE')
    pool_pilotos['af_race_norm'] = normalizar_columna(pool_pilotos, 'media_afinidad_race')
    pool_pilotos['af_qualy_norm'] = normalizar_columna(pool_pilotos, 'media_afinidad_qualy')
    pool_pilotos['score_piloto'] = (
        0.30 * pool_pilotos['gp_norm'] +
        0.20 * pool_pilotos['qp_norm'] +
        0.15 * pool_pilotos['adj_norm'] +
        0.15 * pool_pilotos['pr_norm'] +
        0.10 * pool_pilotos['af_race_norm'] +
        0.10 * pool_pilotos['af_qualy_norm']
    )
    # Quedarnos con 1 año por piloto: el más cercano al centro
    centro_piloto = pool_pilotos['score_piloto'].median()
    pool_pilotos['dist_centro_piloto'] = (pool_pilotos['score_piloto'] - centro_piloto).abs()
    pool_pilotos = (
        pool_pilotos
        .sort_values(['dist_centro_piloto', 'score_piloto'])
        .drop_duplicates(subset=['piloto'], keep='first')
        .copy()
    )
    if len(pool_pilotos) < 8:
        st.error("No hay suficientes pilotos distintos tras filtrar.")
        return
    # =========================================================
    # 2) FILTRO COCHES: BUENOS PERO NO ROTOS
    # =========================================================
    pool_ct = pool_ct[
        (pool_ct['CONSTRUCTOR_PERFORMANCE'] >= 8) &
        (pool_ct['CONSTRUCTOR_PERFORMANCE'] <= 22) &
        (pool_ct['CT_QUALY_PERFORMANCE'] >= 50) &
        (pool_ct['CT_QUALY_PERFORMANCE'] <= 170)
    ].copy()
    if len(pool_ct) < 8:
        st.error("No hay suficientes coches en la banda buena equilibrada.")
        return
    # Normalización coche
    pool_ct['ct_race_norm'] = normalizar_columna(pool_ct, 'CONSTRUCTOR_PERFORMANCE')
    pool_ct['ct_qualy_norm'] = normalizar_columna(pool_ct, 'CT_QUALY_PERFORMANCE')
    pool_ct['score_coche'] = (
        0.60 * pool_ct['ct_race_norm'] +
        0.40 * pool_ct['ct_qualy_norm']
    )
    # Quedarnos con 1 año por constructor: el más cercano al centro
    centro_coche = pool_ct['score_coche'].median()
    pool_ct['dist_centro_coche'] = (pool_ct['score_coche'] - centro_coche).abs()
    pool_ct = (
        pool_ct
        .sort_values(['dist_centro_coche', 'score_coche'])
        .drop_duplicates(subset=['constructor'], keep='first')
        .copy()
    )
    if len(pool_ct) < 8:
        st.error("No hay suficientes constructores distintos tras filtrar.")
        return
    # =========================================================
    # 3) SCORE DE COMBINACIÓN PILOTO + COCHE
    # =========================================================
    todas_combinaciones = construir_combinaciones(pool_pilotos, pool_ct)
    if todas_combinaciones.empty:
        st.error("No se pudieron construir combinaciones piloto-coche.")
        return
    # =========================================================
    # 4) BANDA EQUILIBRADA DE COMBINACIONES
    # =========================================================
    # Queremos combinaciones fuertes pero compactas
    df_combo = todas_combinaciones[
        (todas_combinaciones['score_combo'] >= 0.52) &
        (todas_combinaciones['score_combo'] <= 0.68)
    ].copy()
    if len(df_combo) < 8:
        # Si quedó demasiado estrecho, abrimos un poco
        df_combo = todas_combinaciones[
            (todas_combinaciones['score_combo'] >= 0.50) &
            (todas_combinaciones['score_combo'] <= 0.72)
        ].copy()
    if len(df_combo) < 8:
        st.error("No hay suficientes combinaciones equilibradas tras filtrar.")
        return
    # Ordenar por cercanía al centro de equilibrio
    centro_combo = df_combo['score_combo'].median()
    df_combo['dist_centro_combo'] = (df_combo['score_combo'] - centro_combo).abs()
    # Penalizar ligeramente pilotos que aparecen demasiadas veces en combinaciones válidas
    freq_piloto = df_combo['piloto'].value_counts()
    df_combo['penalizacion_repeticion_piloto'] = 0.025 * (df_combo['piloto'].map(freq_piloto) / freq_piloto.max())
    # Score ajustado: mantenemos equilibrio, pero favorecemos variedad de nombres
    df_combo['score_variedad'] = (
        df_combo['dist_centro_combo'] + df_combo['penalizacion_repeticion_piloto']
    )
    df_combo = df_combo.sort_values(['score_variedad', 'score_combo']).reset_index(drop=True)
    # =========================================================
    # 5) ELEGIR 8 BINOMIOS SIN REPETIR PILOTO NI CONSTRUCTOR
    # =========================================================
    seleccion_final = []
    pilotos_usados = set()
    constructores_usados = set()
    top_n = min(90, len(df_combo))
    candidatos = df_combo.iloc[:top_n].copy()
    # Evitar que un mismo piloto domine demasiado el pool final de candidatos
    candidatos = (
        candidatos
        .groupby('piloto', group_keys=False)
        .head(3)
        .reset_index(drop=True)
    )
    candidatos = candidatos.sample(frac=1, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
    elegir_sin_repetir(candidatos, seleccion_final, pilotos_usados, constructores_usados)
    # Fallback si no salió suficiente
    if len(seleccion_final) < 8:
        elegir_sin_repetir(df_combo, seleccion_final, pilotos_usados, constructores_usados)
    if len(seleccion_final) < 8:
        st.error("No se pudieron seleccionar 8 combinaciones únicas piloto-coche.")
        return
    seleccion_final = pd.DataFrame(seleccion_final).reset_index(drop=True)
    # Orden opcional: del más fuerte al menos fuerte dentro del bloque equilibrado
    seleccion_final = seleccion_final.sort_values('score_combo', ascending=False).reset_index(drop=True)
    # =========================================================
    # 6) VOLCAR A SESSION_STATE
    # =========================================================
    for i in range(8):
        row = seleccion_final.iloc[i]
        st.session_state[f"piloto_select_{i}"] = row['piloto']
        st.session_state[f"anio_piloto_select_{i}"] = int(row['anio_piloto'])
        st.session_state[f"constructor_select_{i}"] = row['constructor']
        st.session_state[f"anio_constructor_select_{i}"] = int(row['anio_constructor'])
    st.success("Mundial equilibrado generado con equilibrio real de combinación piloto-coche.")

def cargar_10_circuitos_aleatorios():
    rng = np.random.default_rng()
    circuitos_validos = sorted(df_race['CIRCUIT_NAME'].dropna().unique().tolist())
    if len(circuitos_validos) < 10:
        st.error("No hay suficientes circuitos válidos para generar un calendario aleatorio.")
        return
    circuitos_elegidos = rng.choice(circuitos_validos, size=10, replace=False)
    for i, circuito in enumerate(circuitos_elegidos):
        st.session_state[f"circuito_{i}"] = circuito
        st.session_state[f"lluvia_{i}"] = int(rng.choice([0, 1]))

def simular_siguiente_carrera(circuito, lluvia, neutro):
    pilotos_config = st.session_state.pilotos_config
    df_qualy_features = generar_features_qualy(pilotos_config, circuito)
    if df_qualy_features.empty:
        st.error("No se ha podido generar la clasificación: ningún piloto tiene datos válidos.")
        return
    X_qualy = df_qualy_features.drop(columns="PILOTO")
    preds = model_qualy.predict_proba(X_qualy)
    # Factor de peso para escudería
    factor_ct = 0.2  # más peso => mayor efecto del rendimiento de escudería
    scores = []
    for prob, piloto, ct_perf in zip(preds, df_qualy_features["PILOTO"], df_qualy_features["CT_QUALY_PERFORMANCE"]):
        base = sum(p * w for p, w in zip(prob, PONDERACIONES))
        ajuste = base * (1 + factor_ct * ct_perf)
        scores.append((ajuste, piloto))
    scores_ordenados = sorted(scores, reverse=True)
    grid = [p for _, p in scores_ordenados]  # lista final ordenada por rendimiento
    resultados = simular_carrera(pilotos_config, circuito, lluvia, grid, neutro)
    if not resultados:
        st.error("No se ha podido simular la carrera: ningún piloto tiene datos válidos.")
        return

    st.subheader(f"🔢 Clasificación en {circuito}")
    base_ajuste = scores_ordenados[0][0]
    for i, (ajuste, piloto) in enumerate(scores_ordenados, start=1):
        tiempo_dif = (base_ajuste - ajuste) * 0.001  # cada punto = 0.001 segundos
        if i == 1:
            st.write(f"**P{i}**: {piloto} — 0.000 s")
        else:
            st.write(f"**P{i}**: {piloto} — +{tiempo_dif:.3f} s")
    # Registrar pole
    poleman = grid[0]
    st.session_state.poles[poleman] = st.session_state.poles.get(poleman, 0) + 1

    st.subheader(f"🏆 Resultado carrera en {circuito}")
    clasificacion_antes = dict(st.session_state.resultados_mundial)
    base_score = resultados[0][0]  # puntuación del ganador (referencia)
    resultados_dict = []
    for i, (score, piloto) in enumerate(resultados, start=1):
        puntos_asignados = PUNTOS_F1[i-1] if i <= len(PUNTOS_F1) else 0
        tiempo = (base_score - score) * 0.5  # conversión a segundos
        if i == 1:
            st.write(f"{i}. {piloto} — 0.00 s")
        else:
            st.write(f"{i}. {piloto} — +{tiempo:.2f} s")
        st.session_state.resultados_mundial[piloto] = st.session_state.resultados_mundial.get(piloto, 0) + puntos_asignados
        resultados_dict.append({
            "piloto": piloto,
            "salida": grid.index(piloto) + 1,
            "llegada": i,
            "puntos_f1": puntos_asignados,
            "score_modelo": score
        })
    ganador = resultados[0][1]
    st.session_state.victorias[ganador] = st.session_state.victorias.get(ganador, 0) + 1
    # === Guardar resultados carrera + estado del mundial ===
    st.session_state.historial_carreras.append({
        "circuito": circuito,
        "lluvia": bool(lluvia),
        "resultados": resultados_dict,
        "clasificacion_antes": clasificacion_antes,
        "clasificacion_despues": dict(st.session_state.resultados_mundial)
    })
    st.session_state.carrera_actual += 1

# --- Interfaz de usuario ---

st.title("🏁 Simulador Histórico Mini-Mundial F1")
st.markdown("""
<p class="frase-senna">
If you no longer go for a gap that exists, you are no longer a racing driver
</p>
""", unsafe_allow_html=True)
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

st.header("1. Selección de pilotos y escuderías")
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown('<div style="display:flex; align-items:center; gap:10px;">', unsafe_allow_html=True)
    if st.button("🎲"):
        cargar_8_pilotos_aleatorios()
        st.rerun()
    st.markdown('<span style="font-size:14px; color:#cbd5e1;">Si te da pereza seleccionar pilotos o no tienes preferencias, pulsa aquí para una selección completamente aleatoria</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div style="display:flex; align-items:center; gap:10px;">', unsafe_allow_html=True)
    if st.button("⚖️"):
        cargar_8_pilotos_equilibrados()
        st.rerun()
    st.markdown('<span style="font-size:14px; color:#cbd5e1;">Esta selección también es aleatoria, es la opción RECOMENDADA si quieres un mundial equilibrado en el que cualquiera pueda ganar</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
driverrefs = sorted(df_drivers['DRIVERREF'].dropna().unique())
constructorrefs = sorted(df_constructors['NAME#'].dropna().unique())
anios_por_piloto = calcular_anios_validos("DRIVER_ID")
anios_por_constructor = calcular_anios_validos("CONSTRUCTOR_ID")
st.session_state.pilotos_config = {}
pilotos_seleccionados = []
for i in range(8):
    st.markdown(f"### Piloto {i+1}")
    # PILOTO y AÑO
    col_piloto, col_gap, col_anio, col_rest = st.columns([1.35, 0.05, 0.78, 2.10])
    with col_piloto:
        piloto = st.selectbox(f"Piloto {i+1}", driverrefs, key=f"piloto_select_{i}")
    with col_anio:
        years_piloto = anios_por_piloto.get(driver_id_por_ref[piloto]) or [2012]
        anio_piloto = st.selectbox(f"Año piloto {i+1}", years_piloto, key=f"anio_piloto_select_{i}")
    # ESCUDERÍA y AÑO
    col_esc, col_gap2, col_anio_esc, col_rest2 = st.columns([1.35, 0.05, 0.78, 2.10])
    with col_esc:
        constructor = st.selectbox(f"Escudería {i+1}", constructorrefs, key=f"constructor_select_{i}")
    with col_anio_esc:
        years_constructor = anios_por_constructor.get(constructor_id_por_nombre[constructor]) or [2012]
        anio_constructor = st.selectbox(f"Año escudería {i+1}", years_constructor, key=f"anio_constructor_select_{i}")
    pilotos_seleccionados.append(piloto)
    st.session_state.pilotos_config[piloto] = {
        "anio_piloto": int(anio_piloto),
        "constructor": constructor,
        "anio_constructor": int(anio_constructor)
    }
    st.markdown("---")
pilotos_repetidos = sorted({p for p in pilotos_seleccionados if pilotos_seleccionados.count(p) > 1})
if pilotos_repetidos:
    st.warning(f"⚠️ Hay pilotos repetidos: {', '.join(pilotos_repetidos)}. Cada piloto solo puede aparecer una vez para poder simular.")

st.header("2. Define los 10 circuitos y si llueve")
col_cal1, col_cal2 = st.columns([1, 4])
with col_cal1:
    if st.button("🎲 Circuitos y lluvia aleatorios"):
        cargar_10_circuitos_aleatorios()
        st.rerun()
with col_cal2:
    st.caption("Rellena automáticamente los 10 circuitos y asigna lluvia aleatoria a cada carrera.")
circuitos = sorted(df_race['CIRCUIT_NAME'].dropna().unique())
calendario_temporal = []
for i in range(10):
    st.markdown(f"### Carrera {i+1}")
    col_circuito, col_gap, col_lluvia, col_rest = st.columns([1.35, 0.04, 0.60, 2.01])
    with col_circuito:
        circuito = st.selectbox(
            f"Circuito {i+1}",
            circuitos,
            key=f"circuito_{i}"
        )
    with col_lluvia:
        lluvia = st.radio(
            "¿Llueve?",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Sí",
            horizontal=True,
            key=f"lluvia_{i}"
        )
    calendario_temporal.append((circuito, lluvia))
    st.markdown("---")
col_guardar, col_reset = st.columns([3, 1])
with col_guardar:
    guardar_calendario = st.button("📅 Guardar calendario")
with col_reset:
    resetear_mundial = st.button("🔄 Resetear Mundial")
if guardar_calendario:
    st.session_state.calendario = calendario_temporal
    st.success("Calendario guardado correctamente")
if resetear_mundial:
    st.session_state.calendario = []
    st.session_state.resultados_mundial = {}
    st.session_state.carrera_actual = 0
    st.session_state.poles = {}
    st.session_state.victorias = {}
    st.session_state.historial_carreras = []  # 🔁 limpia prompts anteriores
    st.success("Estado reiniciado correctamente")
if st.button("🎙️ Generar presentación de pilotos y equipos"):
    prompt_presentacion = generar_prompt_presentacion(st.session_state.pilotos_config)
    st.text_area("Prompt para GPT (Presentación previa al mundial)", prompt_presentacion, height=400)
simulacion_neutra = st.checkbox("Simular mundial sin ajustes", value=False)
if st.button("Simular siguiente carrera"):
    if not st.session_state.calendario:
        st.warning("Primero guarda el calendario con el botón '📅 Guardar calendario'")
    elif pilotos_repetidos:
        st.error("No se puede simular con pilotos repetidos. Cambia la selección en el apartado 1.")
    elif st.session_state.carrera_actual < len(st.session_state.calendario):
        circuito, lluvia = st.session_state.calendario[st.session_state.carrera_actual]
        simular_siguiente_carrera(circuito, lluvia, simulacion_neutra)
    else:
        st.info("El mundial ya ha terminado. Pulsa '🔄 Resetear Mundial' para empezar otro.")
if st.session_state.historial_carreras:
    ultima = st.session_state.historial_carreras[-1]
    prompt = generar_prompt_para_gpt(ultima)
    st.text_area("📝 Prompt generado", prompt, height=400)
    st.markdown(f"""
        👉 [Abrir mi GPT personalizado y pegar el prompt]({URL_GPT})
        ⚠️ Asegúrate de haber copiado el prompt de arriba antes de ir.
    """)

st.header("📄 Resumen del Mundial (Pilotos y Escuderías)")
# No mostrar tabla si aún no se ha simulado ninguna carrera
if st.session_state.carrera_actual == 0 or not st.session_state.resultados_mundial:
    st.info("ℹ️ El resumen del mundial aparecerá cuando se simule la primera carrera.")
else:
    resumen_mundial = []
    for piloto, puntos in st.session_state.resultados_mundial.items():
        config = st.session_state.pilotos_config.get(piloto, {})
        resumen_mundial.append({
            "Piloto": piloto,
            "Año Piloto": config.get("anio_piloto", "❓"),
            "Escudería": config.get("constructor", "❓"),
            "Año Escudería": config.get("anio_constructor", "❓"),
            "Puntos": puntos,
            "Poles": st.session_state.poles.get(piloto, 0),
            "Victorias": st.session_state.victorias.get(piloto, 0)
        })
    df_resumen = pd.DataFrame(resumen_mundial).astype({"Año Piloto": "string", "Año Escudería": "string"})
    st.dataframe(df_resumen.sort_values("Puntos", ascending=False))
if st.session_state.calendario and st.session_state.carrera_actual >= len(st.session_state.calendario):
    st.success("🏁 Mundial finalizado")
    clasificacion_final = sorted(
        st.session_state.resultados_mundial.items(),
        key=lambda x: x[1],
        reverse=True
    )
    if clasificacion_final:  # Solo si hay pilotos clasificados
        campeon = clasificacion_final[0][0]
        campeon_config = st.session_state.pilotos_config.get(campeon, {})
        escuderia = campeon_config.get("constructor", "unknown")
        st.markdown(f"🎉 **¡El campeón del mundial es {campeon.replace('_', ' ').title()} con {escuderia.title()}!** 🏆")
        ruta_piloto = os.path.join("images", f"{campeon}.jpg")
        ruta_logo = os.path.join("logos", f"{escuderia}.jpg")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(ruta_piloto):
                st.image(Image.open(ruta_piloto), caption=campeon.replace('_', ' ').title(), width=300)
            else:
                st.info("📷 Imagen del campeón no disponible.")
        with col2:
            if os.path.exists(ruta_logo):
                st.image(Image.open(ruta_logo), caption=escuderia.title(), width=400)
            else:
                st.info("🏎️ Logo de la escudería no disponible.")
