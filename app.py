"""
=============================================================================
 Simulador de la Ley de Enfriamiento de Newton  (v3: tema "hielo")
=============================================================================
 Aplicación web hecha con Streamlit + NumPy + Pandas + Plotly.

 Ejecución:
     pip install streamlit numpy pandas plotly
     streamlit run app.py

 Qué hace:
   * Resuelve numéricamente dT/dt = -k (T - Ta) con tres métodos
     (Euler, Heun/RK2 y Runge-Kutta 4).
   * Calcula la solución analítica T(t) = Ta + (T0 - Ta) e^(-k t).
   * Compara ambas, muestra el error y permite consultar la temperatura
     en cualquier instante o el tiempo necesario para alcanzar una
     temperatura objetivo.
   * Incluye ejemplos rápidos y permite descargar los datos en CSV.

 Diseño: fondo blanco, paleta celeste/azul y stickers (cubos de hielo,
 tazas de café y copos de nieve) dibujados como SVG incrustado en el CSS,
 por lo que no se necesita ningún archivo de imagen ni conexión a internet.
=============================================================================
"""

from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN GENERAL
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Ley de Enfriamiento de Newton",
    page_icon="🌡️",
    layout="wide",
)

# Opacidad de los stickers (0 = invisibles, 1 = sólidos)
OPACIDAD_STICKERS = 0.45

# ---- Paleta celeste / azul --------------------------------------------------
TINTA = "#12304D"            # azul marino para textos
COLOR_ANALITICA = "#1565C0"  # azul intenso
COLOR_NUMERICA = "#29B6F6"   # celeste
COLOR_AMBIENTE = "#90A4AE"   # gris azulado
COLOR_ERROR = "#7986CB"      # índigo suave
COLOR_ACENTO = "#26C6DA"     # cian
COLOR_GRILLA = "rgba(18,48,77,0.10)"
FUENTE = "Nunito, 'Source Sans Pro', sans-serif"

# Tema de Streamlit (se escribe en .streamlit/config.toml si no existe)
TEMA = {
    "base": "light",
    "primaryColor": "#1E88E5",
    "backgroundColor": "#FFFFFF",
    "secondaryBackgroundColor": "#EAF4FC",
    "textColor": TINTA,
}


def asegurar_tema():
    """
    Streamlit solo lee el tema (colores de sliders, tablas, pestañas...) desde
    .streamlit/config.toml. Si el archivo no existe en la carpeta desde la que se
    ejecuta la app, se crea. Devuelve 'creado', 'existe' o 'error'.
    """
    ruta = Path.cwd() / ".streamlit" / "config.toml"
    if ruta.exists():
        return "existe"
    try:
        ruta.parent.mkdir(exist_ok=True)
        lineas = ["[theme]"] + [f'{clave} = "{valor}"' for clave, valor in TEMA.items()]
        ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
        return "creado"
    except OSError:
        return "error"


ESTADO_TEMA = asegurar_tema()

# -----------------------------------------------------------------------------
# 2. STICKERS (SVG dibujados en código)
# -----------------------------------------------------------------------------
def _sticker(contenido, giro=0, opacidad=OPACIDAD_STICKERS):
    """Envuelve un dibujo en un SVG con sombra suave y una rotación opcional."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="140" height="140" viewBox="-20 -20 140 140">'
        '<defs><filter id="s" x="-25%" y="-25%" width="150%" height="150%">'
        '<feDropShadow dx="0" dy="3" stdDeviation="3" flood-color="#1E88E5" flood-opacity="0.15"/>'
        "</filter></defs>"
        f'<g filter="url(#s)" opacity="{opacidad}" transform="rotate({giro} 50 50)">{contenido}</g>'
        "</svg>"
    )


def sticker_cubo(giro=0):
    """Cubo de hielo isométrico."""
    borde = 'stroke="#7CC4EE" stroke-width="2.5" stroke-linejoin="round"'
    return _sticker(
        f'<polygon points="50,10 88,30 50,50 12,30" fill="#E6F5FE" {borde}/>'
        f'<polygon points="12,30 50,50 50,92 12,72" fill="#BDE3F8" {borde}/>'
        f'<polygon points="88,30 50,50 50,92 88,72" fill="#94D0F2" {borde}/>'
        '<path d="M30 27 L47 19" stroke="#fff" stroke-width="4" stroke-linecap="round" opacity=".9"/>'
        '<path d="M21 44 L21 60" stroke="#fff" stroke-width="4" stroke-linecap="round" opacity=".8"/>',
        giro,
    )


def sticker_taza(giro=0):
    """Taza de café humeante sobre un platillo."""
    azul = "#4A90D9"
    return _sticker(
        '<ellipse cx="48" cy="86" rx="38" ry="8" fill="#D6EBFA" stroke="#7CB8E8" stroke-width="2.5"/>'
        f'<path d="M78 44 C98 44 98 70 76 70" fill="none" stroke="{azul}" stroke-width="6" stroke-linecap="round"/>'
        f'<path d="M16 38 H80 V58 C80 76 66 84 48 84 C30 84 16 76 16 58 Z" fill="#FFFFFF" '
        f'stroke="{azul}" stroke-width="3" stroke-linejoin="round"/>'
        f'<ellipse cx="48" cy="38" rx="32" ry="6.5" fill="#8D6E63" stroke="{azul}" stroke-width="3"/>'
        '<path d="M26 52 C26 62 32 70 40 74" stroke="#CFE8FA" stroke-width="4" fill="none" stroke-linecap="round"/>'
        '<g fill="none" stroke="#7CB8E8" stroke-width="3.5" stroke-linecap="round" opacity=".9">'
        '<path d="M34 26 C28 19 40 14 34 5"/><path d="M50 26 C44 19 56 14 50 5"/>'
        '<path d="M66 26 C60 19 72 14 66 5"/></g>',
        giro,
    )


def sticker_copo(giro=0):
    """Copo de nieve de seis brazos."""
    brazo = (
        '<path d="M50 8 V92 M50 24 L40 14 M50 24 L60 14 M50 76 L40 86 M50 76 L60 86"/>'
    )
    return _sticker(
        '<g stroke="#7FC4F0" stroke-width="4" stroke-linecap="round" fill="none">'
        f'{brazo}<g transform="rotate(60 50 50)">{brazo}</g>'
        f'<g transform="rotate(120 50 50)">{brazo}</g></g>',
        giro,
    )


def _uri(svg):
    """Convierte un SVG en una URI de datos utilizable dentro de CSS."""
    return "data:image/svg+xml;utf8," + quote(svg)


def _capas_css(capas):
    """De una lista [(svg, tamaño, posición)] devuelve las tres listas CSS."""
    imagenes = ", ".join(f'url("{_uri(svg)}")' for svg, _, _ in capas)
    tamanos = ", ".join(t for _, t, _ in capas)
    posiciones = ", ".join(p for _, _, p in capas)
    return imagenes, tamanos, posiciones


# Stickers del fondo de la página (posición fija respecto a la ventana)
CAPAS_FONDO = [
    (sticker_taza(-8), "74px", "right 1.5% top 14%"),
    (sticker_cubo(12), "58px", "right 3% bottom 6%"),
    (sticker_copo(0), "50px", "right 16% top 2%"),
    (sticker_cubo(-14), "46px", "left 38% bottom 3%"),
    (sticker_taza(9), "58px", "left 62% bottom 2%"),
    (sticker_copo(15), "56px", "right 1% top 52%"),
]
# Stickers del panel lateral
CAPAS_LATERAL = [
    (sticker_copo(10), "36px", "right 12px top 18px"),
    (sticker_cubo(-10), "46px", "right 10px bottom 8px"),
]

# -----------------------------------------------------------------------------
# 3. ESTILOS (CSS)
# -----------------------------------------------------------------------------
_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

    :root {
        --tinta: #12304D;
        --tinta-suave: #4A6A8A;
        --azul: #1E88E5;
        --azul-osc: #1565C0;
        --hielo: #EAF4FC;
        --borde: rgba(30,136,229,0.22);
    }

    /* ---- Fondo blanco con stickers ---- */
    .stApp {
        background-color: #FFFFFF;
        background-image: __FONDO_IMG__;
        background-repeat: no-repeat;
        background-size: __FONDO_TAM__;
        background-position: __FONDO_POS__;
        background-attachment: fixed;
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stHeader"] {background-color: transparent !important;}
    .block-container {padding-top: 2.2rem; max-width: 1300px;}

    /* ---- Panel lateral celeste con stickers ---- */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div:first-child {background-color: var(--hielo);}
    section[data-testid="stSidebar"] > div:first-child {
        background-image: __LAT_IMG__;
        background-repeat: no-repeat;
        background-size: __LAT_TAM__;
        background-position: __LAT_POS__;
    }

    /* ---- Tipografía y colores de texto ---- */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {
        color: var(--azul-osc); font-family: 'Nunito', 'Source Sans Pro', sans-serif;
    }
    .stApp h1 {font-size: 2.5rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.1rem;}
    .stApp h3 {font-weight: 800;}
    .stApp h4 {font-size: 1.1rem; font-weight: 800; margin-bottom: 0.2rem;}
    .stApp p, .stApp li, .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stWidgetLabel"] p {color: var(--tinta);}
    .stApp [data-testid="stCaptionContainer"], .stApp small {color: var(--tinta-suave);}
    .stApp button p {color: inherit !important;}

    /* ---- Métricas (evita que los valores se corten) ---- */
    [data-testid="stMetricLabel"] p {color: var(--tinta-suave);}
    [data-testid="stMetricValue"] {color: var(--azul-osc); font-size: 1.5rem;}
    [data-testid="stMetricValue"] > div {
        overflow: visible !important; text-overflow: clip !important; white-space: normal !important;
    }

    /* ---- Tarjetas de resumen ---- */
    .tarjeta {
        border: 1px solid var(--borde);
        border-left: 5px solid var(--acento);
        border-radius: 14px;
        padding: 0.85rem 1.1rem;
        background: rgba(255,255,255,0.88);
        box-shadow: 0 2px 10px rgba(30,136,229,0.08);
        backdrop-filter: blur(3px);
        height: 100%;
        font-family: 'Nunito', 'Source Sans Pro', sans-serif;
    }
    .tarjeta-titulo  {font-size: 0.85rem; color: var(--tinta-suave);}
    .tarjeta-valor   {font-size: 1.75rem; font-weight: 800; line-height: 1.25; color: var(--azul-osc);}
    .tarjeta-detalle {font-size: 0.8rem; color: var(--tinta-suave);}

    /* ---- Paneles con borde: efecto vidrio sobre los stickers ---- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,0.82);
        border-color: var(--borde) !important;
        border-radius: 14px;
        backdrop-filter: blur(3px);
        box-shadow: 0 2px 10px rgba(30,136,229,0.07);
    }
    .panel-titulo {font-weight: 800; color: var(--azul-osc); margin-bottom: 0.2rem;}

    /* ---- Campos de entrada: fondo blanco y texto azul marino ---- */
    .stApp {color-scheme: light;}
    .stApp [data-baseweb="input"],
    .stApp [data-baseweb="base-input"],
    .stApp [data-baseweb="input"] > div,
    .stApp [data-baseweb="select"] > div,
    .stApp [data-baseweb="select"] > div > div {
        background-color: #FFFFFF !important; border-color: var(--borde) !important;
    }
    .stApp input, .stApp textarea {
        background-color: #FFFFFF !important; color: var(--tinta) !important;
        -webkit-text-fill-color: var(--tinta) !important; caret-color: var(--azul);
    }
    .stApp [data-baseweb="select"] div,
    .stApp [data-baseweb="select"] span {color: var(--tinta) !important;}
    .stApp [data-baseweb="select"] svg {fill: var(--azul-osc);}
    .stApp [data-testid="stNumberInputStepDown"],
    .stApp [data-testid="stNumberInputStepUp"] {background-color: var(--hielo); color: var(--azul-osc);}

    /* Listas desplegables de los selectores */
    [data-baseweb="popover"] ul, [data-baseweb="popover"] [data-baseweb="menu"] {background-color: #FFFFFF !important;}
    [data-baseweb="popover"] li, [data-baseweb="popover"] li * {color: var(--tinta) !important;}
    [data-baseweb="popover"] li:hover {background-color: var(--hielo) !important;}

    /* Iconos de ayuda (?), barra superior y botón de plegar el panel */
    [data-testid="stTooltipIcon"] svg {color: var(--tinta-suave) !important;}
    [data-testid="stHeader"], [data-testid="stToolbar"],
    [data-testid="stToolbar"] button, [data-testid="stToolbar"] svg,
    [data-testid="stSidebarCollapseButton"] svg {color: var(--tinta-suave) !important;}

    /* ---- Botones ---- */
    .stApp button[kind="secondary"],
    .stApp button[data-testid="stBaseButton-secondary"] {
        background-color: #FFFFFF; color: var(--azul-osc); border: 1px solid var(--borde);
    }
    .stApp button[kind="secondary"]:hover,
    .stApp button[data-testid="stBaseButton-secondary"]:hover {
        background-color: #E3F2FD; border-color: var(--azul); color: var(--azul-osc);
    }
    .stApp button[kind="primary"],
    .stApp button[data-testid="stBaseButton-primary"] {
        background-color: var(--azul); border: none; color: #FFFFFF;
    }
    .stApp button[kind="primary"]:hover,
    .stApp button[data-testid="stBaseButton-primary"]:hover {
        background-color: var(--azul-osc); color: #FFFFFF;
    }

    /* ---- Pestañas ---- */
    .stApp button[data-baseweb="tab"] {color: var(--tinta-suave);}
    .stApp button[data-baseweb="tab"][aria-selected="true"] {color: var(--azul) !important;}
    .stApp [data-baseweb="tab-highlight"] {background-color: var(--azul) !important;}
    .stApp [data-baseweb="tab-border"] {background-color: var(--borde) !important;}

    /* ---- Avisos ---- */
    div[data-testid="stAlert"] {border-radius: 12px;}
</style>
"""

_img, _tam, _pos = _capas_css(CAPAS_FONDO)
_limg, _ltam, _lpos = _capas_css(CAPAS_LATERAL)
st.markdown(
    _CSS.replace("__FONDO_IMG__", _img).replace("__FONDO_TAM__", _tam).replace("__FONDO_POS__", _pos)
        .replace("__LAT_IMG__", _limg).replace("__LAT_TAM__", _ltam).replace("__LAT_POS__", _lpos),
    unsafe_allow_html=True,
)

# Aviso si el tema azul aún no está activo (hay que reiniciar Streamlit una vez)
if (st.get_option("theme.primaryColor") or "").lower() != TEMA["primaryColor"].lower():
    if ESTADO_TEMA == "creado":
        st.info(
            "Se creó `.streamlit/config.toml` con el tema azul. Reinicia la app una vez "
            "(Ctrl+C y `streamlit run app.py`) para aplicarlo a sliders, tablas y pestañas."
        )
    else:
        st.info(
            "Para ver la paleta azul completa, agrega esto a tu `.streamlit/config.toml` y "
            "reinicia la app:\n\n```toml\n[theme]\nbase = \"light\"\nprimaryColor = \"#1E88E5\"\n"
            "backgroundColor = \"#FFFFFF\"\nsecondaryBackgroundColor = \"#EAF4FC\"\n"
            f"textColor = \"{TINTA}\"\n```"
        )

# -----------------------------------------------------------------------------
# 4. CONSTANTES DE LA APLICACIÓN
# -----------------------------------------------------------------------------
# Unidades de tiempo: nombre -> (símbolo, segundos que contiene)
UNIDADES_TIEMPO = {
    "segundos": ("s", 1.0),
    "minutos": ("min", 60.0),
    "horas": ("h", 3600.0),
}
METODOS = ["Euler", "Heun (RK2)", "Runge-Kutta 4"]
ORDEN_METODO = {"Euler": 1, "Heun (RK2)": 2, "Runge-Kutta 4": 4}

# Límites de validación de las entradas
LIM_T = (-100.0, 1500.0)      # °C
LIM_K = (0.0001, 10.0)        # 1/unidad de tiempo
LIM_TIEMPO = (0.1, 100000.0)  # unidades de tiempo

# Ejemplos rápidos (valores ilustrativos, en minutos): etiqueta corta, ayuda, valores
EJEMPLOS = {
    "cafe": ("☕\ufe0f Café", "Café recién servido (85 °C) en una habitación a 22 °C",
             dict(T0=85.0, Ta=22.0, k=0.05, t_total=90.0)),
    "agua": ("💧 Agua", "Agua caliente (100 °C) enfriándose a 25 °C",
             dict(T0=100.0, Ta=25.0, k=0.03, t_total=120.0)),
    "frio": ("🧊 Frío", "Bebida fría (4 °C) calentándose al sol a 30 °C",
             dict(T0=4.0, Ta=30.0, k=0.06, t_total=60.0)),
    "pan": ("🍞 Pan", "Pan recién horneado (100 °C) a 22 °C",
            dict(T0=100.0, Ta=22.0, k=0.04, t_total=90.0)),
}

# Valores iniciales guardados en la sesión (los widgets leen de aquí)
VALORES_INICIALES = {
    "unidad": "minutos", "unidad_prev": "minutos",
    "T0": 90.0, "Ta": 22.0, "k": 0.10, "t_total": 60.0,
    "metodo": "Euler", "n_pasos": 60,
}
for _clave, _valor in VALORES_INICIALES.items():
    st.session_state.setdefault(_clave, _valor)


# -----------------------------------------------------------------------------
# 5. FUNCIONES DE FÍSICA Y CÁLCULO
# -----------------------------------------------------------------------------
def solucion_analitica(t, T0, Ta, k):
    """Solución exacta: T(t) = Ta + (T0 - Ta) * exp(-k t). Acepta escalares o arrays."""
    return Ta + (T0 - Ta) * np.exp(-k * t)


def derivada(T, Ta, k):
    """Lado derecho de la EDO: dT/dt = -k (T - Ta)."""
    return -k * (T - Ta)


@st.cache_data(show_spinner=False)
def resolver_numericamente(T0, Ta, k, t_total, n_pasos, metodo):
    """
    Integra la EDO con paso fijo h = t_total / n_pasos.
    Devuelve (t, T), arrays de longitud n_pasos + 1.
    """
    t = np.linspace(0.0, t_total, n_pasos + 1)
    h = t_total / n_pasos
    T = np.empty_like(t)
    T[0] = T0

    for i in range(n_pasos):
        Ti = T[i]
        if metodo == "Euler":
            # Orden 1: pendiente al inicio del intervalo
            T[i + 1] = Ti + h * derivada(Ti, Ta, k)
        elif metodo == "Heun (RK2)":
            # Orden 2: promedio de la pendiente inicial y la del extremo estimado
            k1 = derivada(Ti, Ta, k)
            k2 = derivada(Ti + h * k1, Ta, k)
            T[i + 1] = Ti + 0.5 * h * (k1 + k2)
        else:
            # Orden 4: Runge-Kutta clásico
            k1 = derivada(Ti, Ta, k)
            k2 = derivada(Ti + 0.5 * h * k1, Ta, k)
            k3 = derivada(Ti + 0.5 * h * k2, Ta, k)
            k4 = derivada(Ti + h * k3, Ta, k)
            T[i + 1] = Ti + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return t, T


def tiempo_para_alcanzar(T_objetivo, T0, Ta, k):
    """
    Despeja t de la solución analítica:  t = -ln((Tobj - Ta)/(T0 - Ta)) / k
    Devuelve None si la temperatura objetivo nunca se alcanza.
    """
    if T0 == Ta:
        return None
    cociente = (T_objetivo - Ta) / (T0 - Ta)
    if cociente <= 0 or cociente > 1:
        return None
    return -np.log(cociente) / k


# -----------------------------------------------------------------------------
# 6. FUNCIONES AUXILIARES DE INTERFAZ
# -----------------------------------------------------------------------------
def aplicar_ejemplo(clave):
    """Carga un ejemplo rápido en los controles (se ejecuta al pulsar el botón)."""
    st.session_state.update(EJEMPLOS[clave][2])
    st.session_state["unidad"] = "minutos"
    st.session_state["unidad_prev"] = "minutos"


def cambiar_unidad():
    """
    Al cambiar la unidad de tiempo se convierten k y el tiempo total para que
    la física del problema no cambie (solo cambian las unidades en que se expresa).
    """
    vieja = st.session_state["unidad_prev"]
    nueva = st.session_state["unidad"]
    factor = UNIDADES_TIEMPO[nueva][1] / UNIDADES_TIEMPO[vieja][1]
    st.session_state["t_total"] = float(np.clip(st.session_state["t_total"] / factor, *LIM_TIEMPO))
    st.session_state["k"] = float(np.clip(st.session_state["k"] * factor, *LIM_K))
    st.session_state["unidad_prev"] = nueva


def tarjeta(titulo, valor, detalle, color):
    """Dibuja una tarjeta de resumen con una barra de color a la izquierda."""
    st.markdown(
        f"""
        <div class="tarjeta" style="--acento:{color}">
            <div class="tarjeta-titulo">{titulo}</div>
            <div class="tarjeta-valor">{valor}</div>
            <div class="tarjeta-detalle">{detalle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def estilo_figura(fig, eje_x, eje_y):
    """Estilo común de los gráficos: fondo transparente, texto azul marino y grilla discreta."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TINTA, family=FUENTE),
        hoverlabel=dict(bgcolor="white", font=dict(color=TINTA)),
    )
    fig.update_xaxes(title=eje_x, gridcolor=COLOR_GRILLA, linecolor=COLOR_GRILLA, zeroline=False)
    fig.update_yaxes(title=eje_y, gridcolor=COLOR_GRILLA, linecolor=COLOR_GRILLA, zeroline=False)


# -----------------------------------------------------------------------------
# 7. PANEL LATERAL DE CONTROL
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Panel de control")

    # --- Ejemplos rápidos ---
    with st.container(border=True):
        st.markdown('<div class="panel-titulo">Ejemplos rápidos</div>', unsafe_allow_html=True)
        fila1 = st.columns(2)
        fila2 = st.columns(2)
        for columna, clave in zip(fila1 + fila2, EJEMPLOS):
            columna.button(
                EJEMPLOS[clave][0], key=f"btn_{clave}", help=EJEMPLOS[clave][1],
                on_click=aplicar_ejemplo, args=(clave,), use_container_width=True,
            )

    # --- Objeto y entorno ---
    with st.container(border=True):
        st.markdown('<div class="panel-titulo">Objeto y entorno</div>', unsafe_allow_html=True)
        st.number_input(
            "Temperatura inicial T₀ (°C)", key="T0",
            min_value=LIM_T[0], max_value=LIM_T[1], step=1.0,
            help="Temperatura del objeto en t = 0.",
        )
        st.number_input(
            "Temperatura ambiente Tₐ (°C)", key="Ta",
            min_value=LIM_T[0], max_value=1000.0, step=1.0,
            help="Temperatura del entorno, que se supone constante.",
        )

    # --- Tiempo y constante k ---
    with st.container(border=True):
        st.markdown('<div class="panel-titulo">Tiempo y constante k</div>', unsafe_allow_html=True)
        st.selectbox(
            "Unidad de tiempo", list(UNIDADES_TIEMPO.keys()), key="unidad",
            on_change=cambiar_unidad,
            help="Al cambiarla, k y el tiempo total se convierten automáticamente.",
        )
        u, seg_por_unidad = UNIDADES_TIEMPO[st.session_state["unidad"]]
        st.number_input(
            f"Constante de enfriamiento k (1/{u})", key="k",
            min_value=LIM_K[0], max_value=LIM_K[1], step=0.001, format="%.4f",
            help="Mayor k → el objeto se enfría (o calienta) más rápido. Debe ser positiva.",
        )
        st.number_input(
            f"Tiempo total de simulación ({u})", key="t_total",
            min_value=LIM_TIEMPO[0], max_value=LIM_TIEMPO[1], step=1.0,
            help="Duración de la simulación en la unidad elegida.",
        )

    # --- Método numérico ---
    with st.container(border=True):
        st.markdown('<div class="panel-titulo">Método numérico</div>', unsafe_allow_html=True)
        st.selectbox("Método de integración", METODOS, key="metodo")
        st.slider(
            "Número de pasos", min_value=5, max_value=5000, step=5, key="n_pasos",
            help="Más pasos → paso h más pequeño → menor error numérico.",
        )

# Lectura de los valores actuales
T0 = st.session_state["T0"]
Ta = st.session_state["Ta"]
k = st.session_state["k"]
t_total = st.session_state["t_total"]
metodo = st.session_state["metodo"]
n_pasos = int(st.session_state["n_pasos"])
h = t_total / n_pasos

# Avisos de coherencia (en el panel lateral)
if T0 == Ta:
    st.sidebar.warning("T₀ = Tₐ: el objeto ya está en equilibrio y la temperatura no cambia.")
if metodo == "Euler" and k * h > 1:
    st.sidebar.warning(
        f"k·h = {k * h:.2f} > 1: Euler oscila o se vuelve inestable. "
        "Aumenta los pasos o usa RK4."
    )

# -----------------------------------------------------------------------------
# 8. CÁLCULOS PRINCIPALES
# -----------------------------------------------------------------------------
t_num, T_num = resolver_numericamente(T0, Ta, k, t_total, n_pasos, metodo)
T_ana_en_malla = solucion_analitica(t_num, T0, Ta, k)
error_abs = np.abs(T_num - T_ana_en_malla)

# Curva analítica "suave" (muchos puntos) solo para dibujar
t_denso = np.linspace(0, t_total, 1000)
T_ana_densa = solucion_analitica(t_denso, T0, Ta, k)

# Datos para descarga
df = pd.DataFrame(
    {
        f"tiempo_{u}": t_num,
        "T_numerica_C": T_num,
        "T_analitica_C": T_ana_en_malla,
        "error_absoluto_C": error_abs,
    }
)

# -----------------------------------------------------------------------------
# 9. ENCABEZADO Y TARJETAS DE RESUMEN
# -----------------------------------------------------------------------------
modo = "enfriamiento" if T0 > Ta else ("calentamiento" if T0 < Ta else "equilibrio")
st.title("🌡️ Ley de Enfriamiento de Newton")
st.caption(
    f"Modo actual: {modo}. Observa cómo el objeto se acerca a la temperatura del entorno "
    "y compara la solución numérica con la analítica."
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    tarjeta("Temperatura final", f"{T_ana_en_malla[-1]:.2f} °C",
            f"En t = {t_total:g} {u}, con Tₐ = {Ta:g} °C", COLOR_NUMERICA)
with c2:
    tarjeta("Constante de tiempo τ = 1/k", f"{1 / k:.3g} {u}",
            "La diferencia con Tₐ cae a 37 %", COLOR_ANALITICA)
with c3:
    tarjeta("Vida media (ln 2 / k)", f"{np.log(2) / k:.3g} {u}",
            "La diferencia con Tₐ se reduce a la mitad", COLOR_ACENTO)
with c4:
    tarjeta("Error numérico máximo", f"{error_abs.max():.2e} °C",
            f"{metodo}, h = {h:.3g} {u}", COLOR_ERROR)

st.write("")  # pequeño espacio vertical

# -----------------------------------------------------------------------------
# 10. PESTAÑAS
# -----------------------------------------------------------------------------
tab_sim, tab_err, tab_datos, tab_teoria = st.tabs(
    ["Simulación", "Error numérico", "Datos y descarga", "Teoría"]
)

# ------------------------- 10.1 SIMULACIÓN -----------------------------------
with tab_sim:
    col_graf, col_consulta = st.columns([2, 1], gap="medium")

    # --- Columna derecha: consultas ---
    with col_consulta:
        with st.container(border=True):
            st.markdown("#### Temperatura en un instante")
            t_consulta = st.slider(
                f"Tiempo t ({u})",
                min_value=0.0,
                max_value=float(t_total),
                value=float(t_total) / 2,
                step=float(t_total) / 1000,
            )
            T_cons_ana = float(solucion_analitica(t_consulta, T0, Ta, k))
            T_cons_num = float(np.interp(t_consulta, t_num, T_num))

            m1, m2 = st.columns(2)
            m1.metric("Analítica", f"{T_cons_ana:.2f} °C",
                      delta=f"{T_cons_ana - Ta:+.1f} °C vs ambiente", delta_color="off")
            m2.metric("Numérica", f"{T_cons_num:.2f} °C")

            # Avance hacia el equilibrio: 0 % = T₀, 100 % = Tₐ
            avance = 1.0 if T0 == Ta else float(np.clip(1 - (T_cons_ana - Ta) / (T0 - Ta), 0, 1))
            st.progress(avance, text=f"{avance * 100:.0f} % del camino hacia el equilibrio")

        with st.container(border=True):
            st.markdown("#### Tiempo para alcanzar una temperatura")
            T_obj_defecto = float(np.clip((T0 + Ta) / 2, min(T0, Ta), max(T0, Ta)))
            T_obj = st.number_input(
                "Temperatura objetivo (°C)",
                min_value=LIM_T[0], max_value=LIM_T[1], value=T_obj_defecto, step=1.0,
            )
            t_obj = tiempo_para_alcanzar(T_obj, T0, Ta, k)
            if t_obj is None:
                st.warning(
                    "Esa temperatura nunca se alcanza: debe estar entre T₀ y Tₐ, "
                    "y T = Tₐ solo se logra en el límite t → ∞."
                )
            else:
                st.info(f"Se alcanza en **{t_obj:.3f} {u}** ({t_obj * seg_por_unidad:,.1f} s).", icon="⏱️")

    # --- Columna izquierda: gráfico principal ---
    with col_graf:
        with st.container(border=True):
            st.markdown("#### Temperatura del objeto en función del tiempo")
            fig = go.Figure()

            # Línea de temperatura ambiente (se dibuja primero para rellenar hasta ella)
            fig.add_trace(go.Scatter(
                x=[0, t_total], y=[Ta, Ta], mode="lines", name=f"Ambiente ({Ta:g} °C)",
                line=dict(color=COLOR_AMBIENTE, width=1.5, dash="dash"), hoverinfo="skip",
            ))
            # Solución analítica, con sombreado suave entre la curva y Tₐ
            fig.add_trace(go.Scatter(
                x=t_denso, y=T_ana_densa, mode="lines", name="Analítica",
                line=dict(color=COLOR_ANALITICA, width=3),
                fill="tonexty", fillcolor="rgba(41,182,246,0.13)",
                hovertemplate="t = %{x:.3f} " + u + "<br>T = %{y:.3f} °C<extra>Analítica</extra>",
            ))
            # Solución numérica (con marcadores si hay pocos pasos)
            fig.add_trace(go.Scatter(
                x=t_num, y=T_num,
                mode="lines+markers" if n_pasos <= 200 else "lines",
                name=f"Numérica ({metodo})",
                line=dict(color=COLOR_NUMERICA, width=2, dash="dot"),
                marker=dict(size=6),
                hovertemplate="t = %{x:.3f} " + u + "<br>T = %{y:.3f} °C<extra>Numérica</extra>",
            ))
            # Instante consultado
            fig.add_vline(x=t_consulta, line=dict(color="rgba(21,101,192,0.5)", width=1, dash="dot"))
            fig.add_trace(go.Scatter(
                x=[t_consulta], y=[T_cons_ana], mode="markers", name="Instante consultado",
                marker=dict(color="white", size=14, line=dict(color="#0D47A1", width=3)),
                hovertemplate="t = %{x:.3f} " + u + "<br>T = %{y:.3f} °C<extra></extra>",
            ))

            estilo_figura(fig, f"Tiempo ({u})", "Temperatura (°C)")
            fig.update_layout(
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                height=500,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            # theme=None: los colores del gráfico los define estilo_figura()
            st.plotly_chart(fig, use_container_width=True, theme=None)

# ------------------------- 10.2 ERROR NUMÉRICO -------------------------------
with tab_err:
    col_e1, col_e2 = st.columns([1.5, 1], gap="medium")

    with col_e1:
        with st.container(border=True):
            st.markdown("#### Error absoluto en cada paso")
            st.caption("Diferencia entre la solución numérica y la analítica.")
            fig_e = go.Figure()
            fig_e.add_trace(go.Scatter(
                x=t_num, y=error_abs, mode="lines", name="|T_num − T_ana|",
                line=dict(color=COLOR_ERROR, width=2.5),
                fill="tozeroy", fillcolor="rgba(121,134,203,0.15)",
                hovertemplate="t = %{x:.3f} " + u + "<br>error = %{y:.3e} °C<extra></extra>",
            ))
            estilo_figura(fig_e, f"Tiempo ({u})", "Error absoluto (°C)")
            fig_e.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig_e, use_container_width=True, theme=None)

    with col_e2:
        with st.container(border=True):
            st.markdown("#### Comparación de métodos")
            st.caption(f"Con {n_pasos} pasos (h = {h:.4g} {u}).")
            filas = []
            for m in METODOS:
                _, T_m = resolver_numericamente(T0, Ta, k, t_total, n_pasos, m)
                e = np.abs(T_m - T_ana_en_malla)
                filas.append({
                    "Método": m,
                    "Orden": ORDEN_METODO[m],
                    "Error máx. (°C)": float(e.max()),
                    "Error RMS (°C)": float(np.sqrt(np.mean(e ** 2))),
                })
            st.dataframe(
                pd.DataFrame(filas).style.format(
                    {"Error máx. (°C)": "{:.2e}", "Error RMS (°C)": "{:.2e}"}
                ),
                hide_index=True, use_container_width=True,
            )
            st.info(
                "Un método de orden p reduce su error aproximadamente 2ᵖ veces "
                "al duplicar el número de pasos."
            )

# ------------------------- 10.3 DATOS Y DESCARGA -----------------------------
with tab_datos:
    col_d1, col_d2 = st.columns([2, 1], gap="medium")

    with col_d1:
        with st.container(border=True):
            st.markdown("#### Tabla de la simulación")
            st.dataframe(df, use_container_width=True, height=420, hide_index=True)

    with col_d2:
        with st.container(border=True):
            st.markdown("#### Exportar")
            st.write(f"**{len(df):,}** filas · paso h = {h:.4g} {u} · {metodo}")
            # utf-8-sig para que Excel reconozca bien los caracteres
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Descargar CSV", data=csv, file_name="enfriamiento_newton.csv",
                mime="text/csv", type="primary", use_container_width=True,
            )
        with st.container(border=True):
            st.markdown("#### Parámetros usados")
            st.markdown(
                f"- **T₀:** {T0:g} °C\n"
                f"- **Tₐ:** {Ta:g} °C\n"
                f"- **k:** {k:.4g} 1/{u}\n"
                f"- **Tiempo total:** {t_total:g} {u}\n"
                f"- **Pasos:** {n_pasos}"
            )

# ------------------------- 10.4 TEORÍA ---------------------------------------
with tab_teoria:
    col_t1, col_t2 = st.columns(2, gap="medium")

    with col_t1:
        with st.container(border=True):
            st.markdown("#### ¿Qué dice la ley?")
            st.write(
                "Isaac Newton observó que la rapidez con la que un cuerpo cambia de temperatura "
                "es proporcional a la diferencia entre su temperatura y la del medio que lo rodea. "
                "Cuanto mayor es esa diferencia, más rápido cambia la temperatura; al acercarse al "
                "equilibrio, el cambio se hace cada vez más lento."
            )
            st.markdown("**Ecuación diferencial**")
            st.latex(r"\frac{dT}{dt} = -k\,(T - T_a)")
            st.markdown("**Solución analítica** (separando variables e integrando)")
            st.latex(r"T(t) = T_a + (T_0 - T_a)\,e^{-kt}")
            st.markdown("**Tiempo para alcanzar una temperatura T₁**")
            st.latex(r"t = -\frac{1}{k}\,\ln\!\left(\frac{T_1 - T_a}{T_0 - T_a}\right)")

    with col_t2:
        with st.container(border=True):
            st.markdown("#### Cuándo es válido este modelo")
            st.markdown(
                "- Diferencias de temperatura moderadas (transferencia de calor sobre todo por convección).\n"
                "- Temperatura uniforme dentro del objeto (número de Biot pequeño).\n"
                "- Temperatura ambiente constante.\n"
                "- Si T₀ < Tₐ, el mismo modelo describe un **calentamiento**."
            )
        with st.container(border=True):
            st.markdown("#### Regla práctica")
            st.write(
                "Tras 1 constante de tiempo (t = τ) la diferencia con el ambiente se reduce "
                "al 36,8 %; tras 3τ, al 5 %; tras 5τ, a menos del 1 %."
            )

    with st.container(border=True):
        st.markdown("#### Parámetros y unidades")
        st.table(pd.DataFrame(
            {
                "Símbolo": ["T(t)", "T₀", "Tₐ", "k", "t", "τ = 1/k"],
                "Significado físico": [
                    "Temperatura del objeto en el instante t",
                    "Temperatura del objeto en t = 0",
                    "Temperatura del ambiente (se supone constante)",
                    "Constante de enfriamiento: depende del material, la superficie y el medio. "
                    "Mayor k → equilibrio más rápido",
                    "Tiempo transcurrido",
                    "Constante de tiempo: en ese lapso la diferencia (T − Tₐ) cae a 1/e ≈ 37 %",
                ],
                "Unidad": ["°C", "°C", "°C", f"1/{u} (1/s en el SI)", u, u],
            }
        ))

st.caption("Simulación de la Ley de Enfriamiento de Newton.")

#streamlit run app.py
