"""
=============================================================================
 Simulador de la Ley de Enfriamiento de Newton  (v4: dashboard por pestañas)
=============================================================================
 Aplicación web hecha con Streamlit + NumPy + Pandas + Plotly.

 Ejecución:
     pip install -r requirements.txt
     streamlit run app.py

 Estructura (sin panel lateral; todo se navega desde las pestañas superiores):
   * Inicio ........ resumen, acceso a cada sección y ejemplos rápidos.
   * Simulación .... datos del problema, gráfica didáctica y consultas.
   * Constante k ... calcula k a partir de datos básicos (1 medición,
                     2 mediciones o una tabla de medidas con ajuste).
   * Error numérico  comparación de Euler, Heun (RK2) y Runge-Kutta 4.
   * Datos ......... tabla completa y descarga en CSV.
   * Teoría ........ ecuaciones, supuestos y cómo se obtiene k.

 Diseño: fondo blanco, paleta celeste/azul y stickers (cubos de hielo,
 tazas de café y copos de nieve) dibujados como SVG incrustado en el CSS,
 por lo que no se necesita ningún archivo de imagen ni conexión a internet.
=============================================================================
"""

from pathlib import Path
from types import SimpleNamespace
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
    initial_sidebar_state="collapsed",
)

# Opacidad de los stickers del fondo (0 = invisibles, 1 = sólidos)
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
    """Crea .streamlit/config.toml con el tema azul si no existe. Devuelve 'creado', 'existe' o 'error'."""
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


def sticker_cubo(giro=0, opacidad=OPACIDAD_STICKERS):
    """Cubo de hielo isométrico."""
    borde = 'stroke="#7CC4EE" stroke-width="2.5" stroke-linejoin="round"'
    return _sticker(
        f'<polygon points="50,10 88,30 50,50 12,30" fill="#E6F5FE" {borde}/>'
        f'<polygon points="12,30 50,50 50,92 12,72" fill="#BDE3F8" {borde}/>'
        f'<polygon points="88,30 50,50 50,92 88,72" fill="#94D0F2" {borde}/>'
        '<path d="M30 27 L47 19" stroke="#fff" stroke-width="4" stroke-linecap="round" opacity=".9"/>'
        '<path d="M21 44 L21 60" stroke="#fff" stroke-width="4" stroke-linecap="round" opacity=".8"/>',
        giro, opacidad,
    )


def sticker_taza(giro=0, opacidad=OPACIDAD_STICKERS):
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
        giro, opacidad,
    )


def sticker_copo(giro=0, opacidad=OPACIDAD_STICKERS):
    """Copo de nieve de seis brazos."""
    brazo = '<path d="M50 8 V92 M50 24 L40 14 M50 24 L60 14 M50 76 L40 86 M50 76 L60 86"/>'
    return _sticker(
        '<g stroke="#7FC4F0" stroke-width="4" stroke-linecap="round" fill="none">'
        f'{brazo}<g transform="rotate(60 50 50)">{brazo}</g>'
        f'<g transform="rotate(120 50 50)">{brazo}</g></g>',
        giro, opacidad,
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
    (sticker_copo(-12), "44px", "left 1% top 40%"),
    (sticker_cubo(8), "40px", "left 1.5% bottom 8%"),
]
# Stickers del encabezado (más opacos para que se vean sobre el degradado azul)
CAPAS_HERO = [
    (sticker_taza(-8, 0.95), "92px", "right 24px center"),
    (sticker_cubo(12, 0.95), "70px", "right 130px bottom 6px"),
    (sticker_copo(10, 0.9), "62px", "right 215px top 8px"),
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
        --marino: #0D3B66;
        --hielo: #EAF4FC;
        --borde: rgba(30,136,229,0.22);
    }

    /* ---- Sin panel lateral ---- */
    section[data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {display: none !important;}

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
    .block-container {padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1280px;}

    /* ---- Tipografía y colores de texto ---- */
    .stApp, .stApp button, .stApp input {font-family: 'Nunito', 'Source Sans Pro', sans-serif;}
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {
        color: var(--azul-osc); font-family: 'Nunito', 'Source Sans Pro', sans-serif;
    }
    .stApp h3 {font-weight: 800;}
    .stApp h4 {font-size: 1.08rem; font-weight: 800; margin-bottom: 0.2rem;}
    .stApp p, .stApp li, .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stWidgetLabel"] p {color: var(--tinta);}
    .stApp [data-testid="stCaptionContainer"], .stApp small {color: var(--tinta-suave);}
    .stApp button p {color: inherit !important;}

    /* ---- Encabezado (hero) ---- */
    .hero {
        background-image: __HERO_IMG__, linear-gradient(115deg, #0D3B66 0%, #1565C0 55%, #29B6F6 100%);
        background-repeat: no-repeat;
        background-size: __HERO_TAM__, cover;
        background-position: __HERO_POS__, center;
        border-radius: 18px;
        padding: 1.5rem 1.9rem 1.45rem 1.9rem;
        box-shadow: 0 8px 24px rgba(21,101,192,0.25);
        margin-bottom: 0.9rem;
    }
    .stApp .hero h1 {
        color: #FFFFFF; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em;
        margin: 0 0 0.25rem 0; padding: 0;
    }
    .stApp .hero p {color: #DCEEFB; margin: 0; font-size: 1.02rem; max-width: 62%;}
    .hero-etiqueta {
        display: inline-block; font-size: 0.72rem; font-weight: 800; letter-spacing: 0.14em;
        text-transform: uppercase; color: #BFE4FB; margin-bottom: 0.35rem;
    }

    /* ---- Navegación superior (radio con aspecto de pestañas) ---- */
    .st-key-pagina div[role="radiogroup"] {
        gap: 0.35rem; flex-wrap: wrap; background: rgba(255,255,255,0.92);
        padding: 0.35rem; border: 1px solid var(--borde); border-radius: 14px;
        box-shadow: 0 2px 10px rgba(30,136,229,0.08);
    }
    .st-key-pagina label {
        padding: 0.5rem 1.15rem; border-radius: 10px; cursor: pointer; margin: 0;
        transition: background-color .15s ease;
    }
    .st-key-pagina label > div:first-child {display: none;}
    .st-key-pagina label p {font-weight: 700; color: var(--tinta-suave) !important;}
    .st-key-pagina label:hover {background-color: var(--hielo);}
    .st-key-pagina label:has(input:checked) {
        background: linear-gradient(120deg, #1565C0, #29B6F6);
        box-shadow: 0 3px 10px rgba(30,136,229,0.35);
    }
    .st-key-pagina label:has(input:checked) p {color: #FFFFFF !important;}

    /* ---- Fichas de parámetros actuales ---- */
    .chips {display: flex; flex-wrap: wrap; gap: 0.45rem; margin: 0.7rem 0 0.2rem 0;}
    .chip {
        background: rgba(255,255,255,0.92); border: 1px solid var(--borde); border-radius: 999px;
        padding: 0.18rem 0.8rem; font-size: 0.84rem; color: var(--tinta);
    }
    .chip b {color: var(--azul-osc); margin-right: 0.3rem;}

    /* ---- Métricas ---- */
    [data-testid="stMetricLabel"] p {color: var(--tinta-suave);}
    [data-testid="stMetricValue"] {color: var(--azul-osc); font-size: 1.5rem;}
    [data-testid="stMetricValue"] > div {
        overflow: visible !important; text-overflow: clip !important; white-space: normal !important;
    }

    /* ---- Tarjetas de resumen ---- */
    .tarjeta {
        border: 1px solid var(--borde); border-left: 5px solid var(--acento); border-radius: 14px;
        padding: 0.85rem 1.1rem; background: rgba(255,255,255,0.9);
        box-shadow: 0 2px 10px rgba(30,136,229,0.08); backdrop-filter: blur(3px); height: 100%;
    }
    .tarjeta-titulo  {font-size: 0.85rem; color: var(--tinta-suave);}
    .tarjeta-valor   {font-size: 1.7rem; font-weight: 800; line-height: 1.25; color: var(--azul-osc);}
    .tarjeta-detalle {font-size: 0.8rem; color: var(--tinta-suave);}

    /* ---- Paneles con borde: efecto vidrio sobre los stickers ---- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,0.86); border-color: var(--borde) !important;
        border-radius: 14px; backdrop-filter: blur(3px); box-shadow: 0 2px 10px rgba(30,136,229,0.07);
    }
    .panel-titulo {font-weight: 800; color: var(--azul-osc); margin-bottom: 0.2rem;}
    .paso {
        display: inline-block; background: var(--azul-osc); color: #fff !important; font-weight: 800;
        border-radius: 999px; width: 1.55rem; height: 1.55rem; line-height: 1.55rem;
        text-align: center; font-size: 0.85rem; margin-right: 0.5rem;
    }
    .nota {
        background: var(--hielo); border-left: 4px solid var(--azul); border-radius: 10px;
        padding: 0.6rem 0.9rem; font-size: 0.92rem; color: var(--tinta);
    }

    /* ---- Campos de entrada: fondo blanco y texto azul marino ---- */
    .stApp {color-scheme: light;}
    .stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"],
    .stApp [data-baseweb="input"] > div, .stApp [data-baseweb="select"] > div,
    .stApp [data-baseweb="select"] > div > div {
        background-color: #FFFFFF !important; border-color: var(--borde) !important;
    }
    .stApp input, .stApp textarea {
        background-color: #FFFFFF !important; color: var(--tinta) !important;
        -webkit-text-fill-color: var(--tinta) !important; caret-color: var(--azul);
    }
    .stApp [data-baseweb="select"] div, .stApp [data-baseweb="select"] span {color: var(--tinta) !important;}
    .stApp [data-baseweb="select"] svg {fill: var(--azul-osc);}
    .stApp [data-testid="stNumberInputStepDown"],
    .stApp [data-testid="stNumberInputStepUp"] {background-color: var(--hielo); color: var(--azul-osc);}
    [data-baseweb="popover"] ul, [data-baseweb="popover"] [data-baseweb="menu"] {background-color: #FFFFFF !important;}
    [data-baseweb="popover"] li, [data-baseweb="popover"] li * {color: var(--tinta) !important;}
    [data-baseweb="popover"] li:hover {background-color: var(--hielo) !important;}
    [data-testid="stTooltipIcon"] svg {color: var(--tinta-suave) !important;}
    [data-testid="stHeader"], [data-testid="stToolbar"],
    [data-testid="stToolbar"] button, [data-testid="stToolbar"] svg {color: var(--tinta-suave) !important;}

    /* ---- Botones ---- */
    .stApp button[kind="secondary"], .stApp button[data-testid="stBaseButton-secondary"] {
        background-color: #FFFFFF; color: var(--azul-osc); border: 1px solid var(--borde); border-radius: 10px;
    }
    .stApp button[kind="secondary"]:hover, .stApp button[data-testid="stBaseButton-secondary"]:hover {
        background-color: #E3F2FD; border-color: var(--azul); color: var(--azul-osc);
    }
    .stApp button[kind="primary"], .stApp button[data-testid="stBaseButton-primary"] {
        background-color: var(--azul); border: none; color: #FFFFFF; border-radius: 10px;
    }
    .stApp button[kind="primary"]:hover, .stApp button[data-testid="stBaseButton-primary"]:hover {
        background-color: var(--azul-osc); color: #FFFFFF;
    }

    /* ---- Pestañas internas (st.tabs) ---- */
    .stApp button[data-baseweb="tab"] {color: var(--tinta-suave);}
    .stApp button[data-baseweb="tab"][aria-selected="true"] {color: var(--azul) !important;}
    .stApp [data-baseweb="tab-highlight"] {background-color: var(--azul) !important;}
    .stApp [data-baseweb="tab-border"] {background-color: var(--borde) !important;}

    div[data-testid="stAlert"] {border-radius: 12px;}
    .pie {text-align: center; color: var(--tinta-suave); font-size: 0.82rem; margin-top: 1.5rem;}
</style>
"""

_img, _tam, _pos = _capas_css(CAPAS_FONDO)
_himg, _htam, _hpos = _capas_css(CAPAS_HERO)
st.markdown(
    _CSS.replace("__FONDO_IMG__", _img).replace("__FONDO_TAM__", _tam).replace("__FONDO_POS__", _pos)
        .replace("__HERO_IMG__", _himg).replace("__HERO_TAM__", _htam).replace("__HERO_POS__", _hpos),
    unsafe_allow_html=True,
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

# Tabla de medidas de ejemplo para el ajuste de k (café: T0 = 85, Ta = 22, k ≈ 0,05 1/min)
_T_MEDIDAS = [0, 5, 10, 15, 20, 30, 40, 60]
_RUIDO = [0.0, 0.3, -0.2, 0.4, -0.3, 0.2, -0.1, 0.1]
MEDIDAS_EJEMPLO = pd.DataFrame({
    "Tiempo": [float(t) for t in _T_MEDIDAS],
    "Temperatura (°C)": [round(22 + 63 * float(np.exp(-0.05 * t)) + r, 1) for t, r in zip(_T_MEDIDAS, _RUIDO)],
})

PAGINAS = ["Inicio", "Simulación", "Constante k", "Error numérico", "Datos", "Teoría"]

# Valores que deben sobrevivir al cambiar de pestaña (los widgets que no se dibujan
# en una pestaña pierden su estado en Streamlit; abajo se re-guardan en cada ejecución).
VALORES_INICIALES = {
    "pagina": "Inicio",
    "unidad": "minutos", "unidad_prev": "minutos",
    "T0": 90.0, "Ta": 22.0, "k": 0.10, "t_total": 60.0,
    "metodo": "Euler", "n_pasos": 60,
    # Pestaña "Constante k"
    "kc_unidad": "minutos", "kc_modo": "Una medición",
    "kc_T0": 85.0, "kc_Ta": 22.0, "kc_t1": 10.0, "kc_T1": 60.0,
    "kc2_Ta": 22.0, "kc2_t1": 5.0, "kc2_T1": 70.0, "kc2_t2": 20.0, "kc2_T2": 45.0,
    "kc3_Ta": 22.0,
}
for _clave, _valor in VALORES_INICIALES.items():
    st.session_state.setdefault(_clave, _valor)
    st.session_state[_clave] = st.session_state[_clave]
st.session_state.setdefault("medidas", MEDIDAS_EJEMPLO.copy())


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
    """Integra la EDO con paso fijo h = t_total / n_pasos. Devuelve (t, T)."""
    t = np.linspace(0.0, t_total, n_pasos + 1)
    h = t_total / n_pasos
    T = np.empty_like(t)
    T[0] = T0
    for i in range(n_pasos):
        Ti = T[i]
        if metodo == "Euler":
            T[i + 1] = Ti + h * derivada(Ti, Ta, k)
        elif metodo == "Heun (RK2)":
            k1 = derivada(Ti, Ta, k)
            k2 = derivada(Ti + h * k1, Ta, k)
            T[i + 1] = Ti + 0.5 * h * (k1 + k2)
        else:
            k1 = derivada(Ti, Ta, k)
            k2 = derivada(Ti + 0.5 * h * k1, Ta, k)
            k3 = derivada(Ti + 0.5 * h * k2, Ta, k)
            k4 = derivada(Ti + h * k3, Ta, k)
            T[i + 1] = Ti + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, T


def tiempo_para_alcanzar(T_objetivo, T0, Ta, k):
    """Despeja t:  t = -ln((Tobj - Ta)/(T0 - Ta)) / k.  None si nunca se alcanza."""
    if T0 == Ta:
        return None
    cociente = (T_objetivo - Ta) / (T0 - Ta)
    if cociente <= 0 or cociente > 1:
        return None
    return -np.log(cociente) / k


def k_una_medicion(T0, Ta, t1, T1):
    """k a partir de T0, Ta y una medición (t1, T1). Devuelve (k, error)."""
    if t1 <= 0:
        return None, "El tiempo de la medición debe ser mayor que 0."
    if T0 == Ta:
        return None, "T₀ y Tₐ son iguales: no hay enfriamiento que medir."
    cociente = (T1 - Ta) / (T0 - Ta)
    if cociente <= 0:
        return None, "T₁ debe quedar entre T₀ y Tₐ (sin cruzar la temperatura ambiente)."
    if cociente >= 1:
        return None, "T₁ debe estar más cerca de Tₐ que T₀: el objeto tiene que acercarse al ambiente."
    return float(-np.log(cociente) / t1), None


def k_dos_mediciones(Ta, t1, T1, t2, T2):
    """k a partir de Tₐ y dos mediciones (t1, T1), (t2, T2). Devuelve (k, T0_estimada, error)."""
    if t2 <= t1:
        return None, None, "El segundo instante debe ser posterior al primero (t₂ > t₁)."
    if T1 == Ta:
        return None, None, "T₁ no puede ser igual a Tₐ."
    cociente = (T2 - Ta) / (T1 - Ta)
    if cociente <= 0:
        return None, None, "Las dos temperaturas deben estar del mismo lado de Tₐ."
    if cociente >= 1:
        return None, None, "T₂ debe estar más cerca de Tₐ que T₁."
    k = float(-np.log(cociente) / (t2 - t1))
    return k, float(Ta + (T1 - Ta) * np.exp(k * t1)), None


def k_por_ajuste(df, Ta):
    """
    Ajuste por mínimos cuadrados de ln|T - Ta| = ln|T0 - Ta| - k t.
    Devuelve un diccionario con k, T0 estimada, R², puntos usados... o {'error': ...}.
    """
    d = df.dropna()
    if len(d) < 2:
        return {"error": "Escribe al menos 2 mediciones completas (tiempo y temperatura)."}
    t = d.iloc[:, 0].to_numpy(float)
    T = d.iloc[:, 1].to_numpy(float)
    dif = T - Ta
    signo = 1.0 if np.sum(dif > 0) >= np.sum(dif < 0) else -1.0
    ok = dif * signo > 0
    if ok.sum() < 2 or len(np.unique(t[ok])) < 2:
        return {"error": "Necesito al menos 2 mediciones con tiempos distintos y temperatura distinta de Tₐ."}
    t, T, y = t[ok], T[ok], np.log(np.abs(dif[ok]))
    pendiente, ordenada = np.polyfit(t, y, 1)
    if pendiente >= 0:
        return {"error": "Los datos no se acercan a Tₐ con el tiempo (la pendiente no es negativa). Revisa las medidas."}
    ss_res = float(np.sum((y - (pendiente * t + ordenada)) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "k": float(-pendiente), "T0": float(Ta + signo * np.exp(ordenada)),
        "r2": 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot,
        "t": t, "T": T, "y": y, "pendiente": float(pendiente), "ordenada": float(ordenada),
        "usados": int(ok.sum()), "descartados": int(len(ok) - ok.sum()),
    }


# -----------------------------------------------------------------------------
# 6. FUNCIONES AUXILIARES DE INTERFAZ
# -----------------------------------------------------------------------------
def ir_a(pagina):
    """Callback: cambia de pestaña principal."""
    st.session_state["pagina"] = pagina


def aplicar_ejemplo(clave, ir=False):
    """Carga un ejemplo rápido en los controles (se ejecuta al pulsar el botón)."""
    st.session_state.update(EJEMPLOS[clave][2])
    st.session_state["unidad"] = "minutos"
    st.session_state["unidad_prev"] = "minutos"
    if ir:
        st.session_state["pagina"] = "Simulación"


def cambiar_unidad():
    """Al cambiar la unidad se convierten k y el tiempo total para que la física no cambie."""
    vieja = st.session_state["unidad_prev"]
    nueva = st.session_state["unidad"]
    factor = UNIDADES_TIEMPO[nueva][1] / UNIDADES_TIEMPO[vieja][1]
    st.session_state["t_total"] = float(np.clip(st.session_state["t_total"] / factor, *LIM_TIEMPO))
    st.session_state["k"] = float(np.clip(st.session_state["k"] * factor, *LIM_K))
    st.session_state["unidad_prev"] = nueva


def usar_k_en_simulacion(k, unidad, T0, Ta):
    """Pasa la k calculada (y T0, Tₐ) a la pestaña Simulación."""
    ss = st.session_state
    ss["unidad"] = unidad
    ss["unidad_prev"] = unidad
    ss["k"] = float(np.clip(k, *LIM_K))
    ss["Ta"] = float(np.clip(Ta, LIM_T[0], 1000.0))
    if T0 is not None:
        ss["T0"] = float(np.clip(T0, *LIM_T))
    ss["t_total"] = float(np.clip(round(5.0 / k, 1), *LIM_TIEMPO))
    ss["pagina"] = "Simulación"


def tarjeta(titulo, valor, detalle, color):
    """Tarjeta de resumen con una barra de color a la izquierda."""
    st.markdown(
        f'<div class="tarjeta" style="--acento:{color}">'
        f'<div class="tarjeta-titulo">{titulo}</div>'
        f'<div class="tarjeta-valor">{valor}</div>'
        f'<div class="tarjeta-detalle">{detalle}</div></div>',
        unsafe_allow_html=True,
    )


def encabezado_seccion(numero, titulo, ayuda=None):
    """Título de un bloque numerado (①, ②...) con ayuda opcional."""
    st.markdown(f'<span class="paso">{numero}</span><b style="color:#1565C0;font-size:1.08rem">{titulo}</b>',
                unsafe_allow_html=True)
    if ayuda:
        st.caption(ayuda)


def estilo_figura(fig, eje_x, eje_y):
    """Estilo común de los gráficos: fondo transparente, texto azul marino y grilla discreta."""
    fig.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TINTA, family=FUENTE), hoverlabel=dict(bgcolor="white", font=dict(color=TINTA)),
    )
    fig.update_xaxes(title=eje_x, gridcolor=COLOR_GRILLA, linecolor=COLOR_GRILLA, zeroline=False)
    fig.update_yaxes(title=eje_y, gridcolor=COLOR_GRILLA, linecolor=COLOR_GRILLA, zeroline=False)


def par(x):
    """Número entre paréntesis si es negativo (para mostrar sustituciones en LaTeX)."""
    return f"({x:g})" if x < 0 else f"{x:g}"


def contexto():
    """Lee los parámetros guardados y calcula todo lo que usan las pestañas."""
    ss = st.session_state
    T0, Ta, k, t_total = ss["T0"], ss["Ta"], ss["k"], ss["t_total"]
    metodo, n_pasos = ss["metodo"], int(ss["n_pasos"])
    u, seg = UNIDADES_TIEMPO[ss["unidad"]]
    t_num, T_num = resolver_numericamente(T0, Ta, k, t_total, n_pasos, metodo)
    T_ana = solucion_analitica(t_num, T0, Ta, k)
    t_denso = np.linspace(0, t_total, 1000)
    return SimpleNamespace(
        T0=T0, Ta=Ta, k=k, t_total=t_total, metodo=metodo, n_pasos=n_pasos, u=u, seg=seg,
        unidad=ss["unidad"], h=t_total / n_pasos, t_num=t_num, T_num=T_num, T_ana=T_ana,
        err=np.abs(T_num - T_ana), t_denso=t_denso,
        T_denso=solucion_analitica(t_denso, T0, Ta, k), tau=1.0 / k, t_half=np.log(2) / k,
        modo="enfriamiento" if T0 > Ta else ("calentamiento" if T0 < Ta else "equilibrio"),
    )


def fichas(c):
    """Franja con los parámetros actuales (se muestra en las pestañas que no los editan)."""
    st.markdown(
        '<div class="chips">'
        f'<span class="chip"><b>T₀</b>{c.T0:g} °C</span>'
        f'<span class="chip"><b>Tₐ</b>{c.Ta:g} °C</span>'
        f'<span class="chip"><b>k</b>{c.k:.4g} 1/{c.u}</span>'
        f'<span class="chip"><b>Tiempo</b>{c.t_total:g} {c.u}</span>'
        f'<span class="chip"><b>Método</b>{c.metodo}, {c.n_pasos} pasos</span>'
        "</div>",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 7. ENCABEZADO Y NAVEGACIÓN
# -----------------------------------------------------------------------------
st.markdown(
    '<div class="hero"><div class="hero-etiqueta">Simulador interactivo</div>'
    "<h1>Ley de Enfriamiento de Newton</h1>"
    "<p>Explora cómo cambia la temperatura de un objeto, calcula la constante de enfriamiento "
    "k a partir de tus mediciones y compara métodos numéricos con la solución exacta.</p></div>",
    unsafe_allow_html=True,
)
st.radio("Navegación", PAGINAS, key="pagina", horizontal=True, label_visibility="collapsed")

# Aviso si el tema azul aún no está activo (hay que reiniciar Streamlit una vez)
if (st.get_option("theme.primaryColor") or "").lower() != TEMA["primaryColor"].lower():
    if ESTADO_TEMA == "creado":
        st.info("Se creó `.streamlit/config.toml` con el tema azul. Reinicia la app una vez "
                "(Ctrl+C y `streamlit run app.py`) para aplicarlo a sliders, tablas y pestañas.")
    else:
        st.info("Para ver la paleta azul completa, agrega esto a tu `.streamlit/config.toml` y reinicia la app:\n\n"
                '```toml\n[theme]\nbase = "light"\nprimaryColor = "#1E88E5"\nbackgroundColor = "#FFFFFF"\n'
                f'secondaryBackgroundColor = "#EAF4FC"\ntextColor = "{TINTA}"\n```')


# -----------------------------------------------------------------------------
# 8. PÁGINA: INICIO
# -----------------------------------------------------------------------------
def pagina_inicio():
    col_a, col_b = st.columns([1.7, 1], gap="medium")
    with col_a:
        with st.container(border=True):
            st.markdown("#### ¿De qué trata?")
            st.write(
                "Un objeto más caliente que su entorno se enfría rápido al principio y cada vez más "
                "despacio al acercarse al equilibrio. Newton lo describió con una sola ecuación: la "
                "rapidez de cambio es proporcional a la diferencia de temperatura con el ambiente."
            )
            st.latex(r"\frac{dT}{dt} = -k\,(T - T_a)\qquad\Longrightarrow\qquad T(t) = T_a + (T_0 - T_a)\,e^{-kt}")
    with col_b:
        with st.container(border=True):
            st.markdown("#### Cómo empezar")
            st.markdown(
                "1. Elige un **ejemplo rápido** o escribe tus propios datos en *Simulación*.\n"
                "2. Si no conoces **k**, calcúlala en *Constante k* con unas pocas mediciones.\n"
                "3. Revisa el **error**, la **tabla** y la **teoría** en las demás pestañas."
            )

    st.write("")
    secciones = [
        ("Simulación", "Ingresa T₀, Tₐ y k, y observa la curva con τ, vida media y tangente inicial. "
                       "Consulta la temperatura en cualquier instante o el tiempo para llegar a una temperatura."),
        ("Constante k", "¿Solo tienes datos básicos? Obtén k con una medición, con dos mediciones o ajustando "
                        "una tabla de medidas, con el procedimiento paso a paso."),
        ("Error numérico", "Compara Euler, Heun (RK2) y Runge-Kutta 4 con la solución analítica y mira "
                           "cómo el error baja al aumentar los pasos."),
        ("Datos", "Tabla completa de la simulación, parámetros usados y descarga en CSV."),
        ("Teoría", "Ecuación diferencial, solución, supuestos del modelo y deducción de la fórmula de k."),
    ]
    columnas = st.columns(3, gap="medium") + st.columns(3, gap="medium")
    for col, (nombre, texto) in zip(columnas, secciones):
        with col:
            with st.container(border=True):
                st.markdown(f"#### {nombre}")
                st.caption(texto)
                st.button("Abrir →", key=f"ir_{nombre}", on_click=ir_a, args=(nombre,), width="stretch")
    with columnas[5]:
        with st.container(border=True):
            st.markdown("#### Ejemplos rápidos")
            st.caption("Carga un caso típico y ve directo a la simulación.")
            f1, f2 = st.columns(2), st.columns(2)
            for col, clave in zip(f1 + f2, EJEMPLOS):
                col.button(EJEMPLOS[clave][0], key=f"inicio_{clave}", help=EJEMPLOS[clave][1],
                           on_click=aplicar_ejemplo, args=(clave, True), width="stretch")


# -----------------------------------------------------------------------------
# 9. PÁGINA: SIMULACIÓN
# -----------------------------------------------------------------------------
def figura_principal(c, t_consulta, T_consulta, mostrar):
    """Gráfica didáctica: curva exacta, numérica y marcas que explican cada concepto."""
    fig = go.Figure()
    enfria = c.T0 >= c.Ta
    signo_ay = -1 if enfria else 1

    # Temperatura ambiente (primero, para poder rellenar hasta ella)
    fig.add_trace(go.Scatter(
        x=[0, c.t_total], y=[c.Ta, c.Ta], mode="lines", name=f"Ambiente Tₐ = {c.Ta:g} °C",
        line=dict(color=COLOR_AMBIENTE, width=1.8, dash="dash"), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=c.t_denso, y=c.T_denso, mode="lines", name="Solución analítica (exacta)",
        line=dict(color=COLOR_ANALITICA, width=3.5),
        fill="tonexty" if mostrar["sombreado"] else None, fillcolor="rgba(41,182,246,0.14)",
        hovertemplate="t = %{x:.3f} " + c.u + "<br>T = %{y:.3f} °C<extra>Analítica</extra>",
    ))
    if mostrar["numerica"]:
        fig.add_trace(go.Scatter(
            x=c.t_num, y=c.T_num, mode="lines+markers" if c.n_pasos <= 200 else "lines",
            name=f"Solución numérica ({c.metodo})", line=dict(color=COLOR_NUMERICA, width=2, dash="dot"),
            marker=dict(size=6),
            hovertemplate="t = %{x:.3f} " + c.u + "<br>T = %{y:.3f} °C<extra>Numérica</extra>",
        ))
    if mostrar["tangente"] and c.T0 != c.Ta:
        x_fin = min(c.tau, c.t_total)
        fig.add_trace(go.Scatter(
            x=[0, x_fin], y=[c.T0, c.T0 - c.k * (c.T0 - c.Ta) * x_fin], mode="lines",
            name="Tangente en t = 0 (llegaría a Tₐ en t = τ)",
            line=dict(color=COLOR_ACENTO, width=2, dash="dashdot"), hoverinfo="skip",
        ))

    # Punto inicial
    fig.add_trace(go.Scatter(
        x=[0], y=[c.T0], mode="markers", name="Inicio (t = 0)", showlegend=False,
        marker=dict(color=COLOR_ANALITICA, size=11, line=dict(color="white", width=2)),
        hovertemplate=f"Inicio: T₀ = {c.T0:g} °C<extra></extra>",
    ))
    fig.add_annotation(x=0, y=c.T0, text=f"<b>Inicio</b><br>T₀ = {c.T0:g} °C", showarrow=True, arrowhead=2,
                       arrowcolor=COLOR_ANALITICA, ax=55, ay=35 * signo_ay,
                       font=dict(color=TINTA, size=12), bgcolor="rgba(255,255,255,0.85)",
                       bordercolor=COLOR_ANALITICA, borderwidth=1, borderpad=3)
    fig.add_annotation(x=c.t_total, y=c.Ta, text="Equilibrio: la curva se acerca a Tₐ", showarrow=False,
                       xanchor="right", yanchor="top" if enfria else "bottom", yshift=-6 if enfria else 6,
                       font=dict(color="#607D8B", size=11))

    # τ y vida media
    if mostrar["tau"] and c.T0 != c.Ta and c.tau <= c.t_total:
        T_tau = float(solucion_analitica(c.tau, c.T0, c.Ta, c.k))
        fig.add_trace(go.Scatter(
            x=[c.tau], y=[T_tau], mode="markers", showlegend=False,
            marker=dict(symbol="diamond", size=12, color="#FFFFFF", line=dict(color=COLOR_ACENTO, width=3)),
            hovertemplate=f"τ = {c.tau:.3g} {c.u}<br>T = {T_tau:.2f} °C<extra></extra>",
        ))
        fig.add_annotation(
            x=c.tau, y=T_tau, showarrow=True, arrowhead=2, arrowcolor=COLOR_ACENTO, ax=70, ay=60 * signo_ay,
            text=f"<b>τ = 1/k = {c.tau:.3g} {c.u}</b><br>T = {T_tau:.1f} °C<br>queda el 36,8 % de la diferencia",
            font=dict(color=TINTA, size=11), bgcolor="rgba(255,255,255,0.88)",
            bordercolor=COLOR_ACENTO, borderwidth=1, borderpad=3, align="left")
    if mostrar["mitad"] and c.T0 != c.Ta and c.t_half <= c.t_total:
        T_h = (c.T0 + c.Ta) / 2
        fig.add_trace(go.Scatter(
            x=[c.t_half], y=[T_h], mode="markers", showlegend=False,
            marker=dict(symbol="square", size=10, color="#FFFFFF", line=dict(color=COLOR_ERROR, width=3)),
            hovertemplate=f"Vida media = {c.t_half:.3g} {c.u}<br>T = {T_h:.2f} °C<extra></extra>",
        ))
        fig.add_annotation(
            x=c.t_half, y=T_h, showarrow=True, arrowhead=2, arrowcolor=COLOR_ERROR, ax=-85, ay=-50 * signo_ay,
            text=f"<b>Vida media = {c.t_half:.3g} {c.u}</b><br>T = {T_h:.1f} °C<br>la diferencia se reduce a la mitad",
            font=dict(color=TINTA, size=11), bgcolor="rgba(255,255,255,0.88)",
            bordercolor=COLOR_ERROR, borderwidth=1, borderpad=3, align="left")

    # Instante consultado con guías hacia los ejes
    fig.add_shape(type="line", x0=0, x1=t_consulta, y0=T_consulta, y1=T_consulta,
                  line=dict(color="rgba(21,101,192,0.45)", width=1.2, dash="dot"))
    fig.add_shape(type="line", x0=t_consulta, x1=t_consulta, y0=0, y1=1, yref="y domain",
                  line=dict(color="rgba(21,101,192,0.45)", width=1.2, dash="dot"))
    fig.add_trace(go.Scatter(
        x=[t_consulta], y=[T_consulta], mode="markers", name="Instante consultado",
        marker=dict(color="white", size=14, line=dict(color="#0D47A1", width=3)),
        hovertemplate="t = %{x:.3f} " + c.u + "<br>T = %{y:.3f} °C<extra></extra>",
    ))

    span = max(abs(c.T0 - c.Ta), 1.0)
    estilo_figura(fig, f"Tiempo ({c.u})", "Temperatura (°C)")
    fig.update_yaxes(range=[min(c.T0, c.Ta) - 0.12 * span, max(c.T0, c.Ta) + 0.14 * span])
    fig.update_layout(
        hovermode="x unified", height=540, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def pagina_simulacion():
    ss = st.session_state
    # ---- ① Datos del problema ----------------------------------------------
    with st.container(border=True):
        encabezado_seccion("1", "Datos del problema",
                           "Escribe los datos o carga un ejemplo; todo lo demás se actualiza solo.")
        ej = st.columns([1.1, 1, 1, 1, 1, 2.2], gap="small")
        ej[0].markdown("**Ejemplos:**")
        for col, clave in zip(ej[1:5], EJEMPLOS):
            col.button(EJEMPLOS[clave][0], key=f"btn_{clave}", help=EJEMPLOS[clave][1],
                       on_click=aplicar_ejemplo, args=(clave,), width="stretch")

        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            st.markdown('<div class="panel-titulo">Objeto y entorno</div>', unsafe_allow_html=True)
            st.number_input("Temperatura inicial T₀ (°C)", key="T0", min_value=LIM_T[0], max_value=LIM_T[1],
                            step=1.0, help="Temperatura del objeto en t = 0.")
            st.number_input("Temperatura ambiente Tₐ (°C)", key="Ta", min_value=LIM_T[0], max_value=1000.0,
                            step=1.0, help="Temperatura del entorno, que se supone constante.")
        with c2:
            st.markdown('<div class="panel-titulo">Tiempo y constante k</div>', unsafe_allow_html=True)
            st.selectbox("Unidad de tiempo", list(UNIDADES_TIEMPO), key="unidad", on_change=cambiar_unidad,
                         help="Al cambiarla, k y el tiempo total se convierten automáticamente.")
            u, seg = UNIDADES_TIEMPO[ss["unidad"]]
            st.number_input(f"Constante de enfriamiento k (1/{u})", key="k", min_value=LIM_K[0],
                            max_value=LIM_K[1], step=0.001, format="%.4f",
                            help="Mayor k → el objeto se enfría (o calienta) más rápido. Debe ser positiva.")
            st.number_input(f"Tiempo total de simulación ({u})", key="t_total", min_value=LIM_TIEMPO[0],
                            max_value=LIM_TIEMPO[1], step=1.0, help="Duración de la simulación en la unidad elegida.")
            st.button("¿No conoces k? Calcúlala →", key="ir_k", on_click=ir_a, args=("Constante k",), width="stretch")
        with c3:
            st.markdown('<div class="panel-titulo">Método numérico</div>', unsafe_allow_html=True)
            st.selectbox("Método de integración", METODOS, key="metodo")
            st.slider("Número de pasos", min_value=5, max_value=5000, step=5, key="n_pasos",
                      help="Más pasos → paso h más pequeño → menor error numérico.")
            st.caption(f"Paso de integración h = {ss['t_total'] / ss['n_pasos']:.4g} {u}")

    c = contexto()
    if c.T0 == c.Ta:
        st.warning("T₀ = Tₐ: el objeto ya está en equilibrio y la temperatura no cambia.")
    if c.metodo == "Euler" and c.k * c.h > 1:
        st.warning(f"k·h = {c.k * c.h:.2f} > 1: Euler oscila o se vuelve inestable. Aumenta los pasos o usa RK4.")

    # ---- ② Resultados clave ---------------------------------------------------
    st.write("")
    encabezado_seccion("2", f"Resultados clave · modo {c.modo}")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        tarjeta("Temperatura final", f"{c.T_ana[-1]:.2f} °C", f"En t = {c.t_total:g} {c.u}, con Tₐ = {c.Ta:g} °C", COLOR_NUMERICA)
    with t2:
        tarjeta("Constante de tiempo τ = 1/k", f"{c.tau:.3g} {c.u}", "La diferencia con Tₐ cae a 37 %", COLOR_ANALITICA)
    with t3:
        tarjeta("Vida media (ln 2 / k)", f"{c.t_half:.3g} {c.u}", "La diferencia con Tₐ se reduce a la mitad", COLOR_ACENTO)
    with t4:
        tarjeta("Error numérico máximo", f"{c.err.max():.2e} °C", f"{c.metodo}, h = {c.h:.3g} {c.u}", COLOR_ERROR)

    # ---- ③ Gráfica -------------------------------------------------------------
    st.write("")
    with st.container(border=True):
        encabezado_seccion("3", "Gráfica: temperatura en función del tiempo",
                           "Activa o apaga las ayudas visuales para entender cada concepto de la curva.")
        o = st.columns(5)
        mostrar = {
            "numerica": o[0].checkbox("Solución numérica", value=True),
            "tau": o[1].checkbox("Constante de tiempo τ", value=True),
            "mitad": o[2].checkbox("Vida media", value=False),
            "tangente": o[3].checkbox("Tangente inicial", value=False),
            "sombreado": o[4].checkbox("Sombrear diferencia", value=True),
        }
        t_consulta = st.slider(f"Instante a observar: t ({c.u})", min_value=0.0, max_value=float(c.t_total),
                               value=float(c.t_total) / 2, step=float(c.t_total) / 1000, key="t_consulta_graf")
        T_cons_ana = float(solucion_analitica(t_consulta, c.T0, c.Ta, c.k))
        T_cons_num = float(np.interp(t_consulta, c.t_num, c.T_num))
        st.plotly_chart(figura_principal(c, t_consulta, T_cons_ana, mostrar), width="stretch", theme=None)

        with st.expander("¿Cómo leer esta gráfica?"):
            st.markdown(
                "- **Línea azul gruesa:** temperatura exacta del objeto. Empieza en **T₀** y se acerca a **Tₐ** sin llegar nunca a ella.\n"
                "- **Línea gris punteada:** la temperatura del ambiente (Tₐ), hacia donde tiende el objeto.\n"
                "- **Zona sombreada:** la diferencia *T − Tₐ*. Se achica con el tiempo: eso es lo que gobierna la rapidez de cambio.\n"
                "- **Rombo (τ):** tras un tiempo τ = 1/k la diferencia se reduce a 36,8 % de la inicial.\n"
                "- **Tangente inicial:** la rapidez de enfriamiento al empezar (la máxima). Si siguiera igual, llegaría a Tₐ justo en t = τ.\n"
                "- **Puntos celestes:** solución numérica; mientras más pasos, más se pegan a la curva exacta."
            )

    # ---- ④ Consultas -----------------------------------------------------------
    st.write("")
    encabezado_seccion("4", "Consultas sobre la simulación")
    q1, q2 = st.columns(2, gap="medium")
    with q1:
        with st.container(border=True):
            st.markdown("#### Temperatura en el instante elegido")
            st.caption("Usa el control “Instante a observar” de la gráfica.")
            m1, m2 = st.columns(2)
            m1.metric(f"Analítica en t = {t_consulta:.3g} {c.u}", f"{T_cons_ana:.2f} °C",
                      delta=f"{T_cons_ana - c.Ta:+.1f} °C vs ambiente", delta_color="off")
            m2.metric("Numérica", f"{T_cons_num:.2f} °C")
            avance = 1.0 if c.T0 == c.Ta else float(np.clip(1 - (T_cons_ana - c.Ta) / (c.T0 - c.Ta), 0, 1))
            st.progress(avance, text=f"{avance * 100:.0f} % del camino hacia el equilibrio")
            st.latex(rf"T({t_consulta:.3g}) = {par(c.Ta)} + ({par(c.T0)} - {par(c.Ta)})\,e^{{-{c.k:.4g}\cdot {t_consulta:.3g}}} = {T_cons_ana:.2f}\ ^\circ\mathrm{{C}}")
    with q2:
        with st.container(border=True):
            st.markdown("#### Tiempo para alcanzar una temperatura")
            T_def = float(np.clip((c.T0 + c.Ta) / 2, min(c.T0, c.Ta), max(c.T0, c.Ta)))
            T_obj = st.number_input("Temperatura objetivo (°C)", min_value=LIM_T[0], max_value=LIM_T[1],
                                    value=T_def, step=1.0, key="T_obj")
            t_obj = tiempo_para_alcanzar(T_obj, c.T0, c.Ta, c.k)
            if t_obj is None:
                st.warning("Esa temperatura nunca se alcanza: debe estar entre T₀ y Tₐ, y T = Tₐ solo se logra en el límite t → ∞.")
            else:
                st.info(f"Se alcanza en **{t_obj:.3f} {c.u}** ({t_obj * c.seg:,.1f} s).", icon="⏱️")
                st.latex(rf"t = -\frac{{1}}{{{c.k:.4g}}}\ln\!\left(\frac{{{par(T_obj)} - {par(c.Ta)}}}{{{par(c.T0)} - {par(c.Ta)}}}\right) = {t_obj:.3f}\ \mathrm{{{c.u}}}")

    # ---- ⑤ Hitos ---------------------------------------------------------------
    st.write("")
    with st.container(border=True):
        encabezado_seccion("5", "Hitos del enfriamiento",
                           "Cuánto falta para el equilibrio en múltiplos de la constante de tiempo τ.")
        mult = [("Vida media", np.log(2)), ("1 τ", 1), ("2 τ", 2), ("3 τ", 3), ("5 τ", 5)]
        filas = []
        for nombre, m in mult:
            t = m * c.tau
            T = float(solucion_analitica(t, c.T0, c.Ta, c.k))
            resta = float(np.exp(-m) * 100)
            filas.append({"Hito": nombre, f"Tiempo ({c.u})": f"{t:.4g}", "Temperatura (°C)": f"{T:.2f}",
                          "Diferencia restante": f"{resta:.1f} %", "Camino recorrido": f"{100 - resta:.1f} %"})
        st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")


# -----------------------------------------------------------------------------
# 10. PÁGINA: CONSTANTE k
# -----------------------------------------------------------------------------
MODOS_K = ["Una medición", "Dos mediciones", "Tabla de medidas"]


def figura_medidas(t_pts, T_pts, T0, Ta, k, u, etiqueta_pts="Mediciones"):
    """Curva ajustada con los puntos medidos: comprueba visualmente que k es razonable."""
    t_max = max(5.0 / k, float(np.max(t_pts)) * 1.15)
    t = np.linspace(0, t_max, 400)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, t_max], y=[Ta, Ta], mode="lines", name=f"Ambiente Tₐ = {Ta:g} °C",
                             line=dict(color=COLOR_AMBIENTE, width=1.8, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=solucion_analitica(t, T0, Ta, k), mode="lines",
                             name=f"Curva con k = {k:.4g} 1/{u}", line=dict(color=COLOR_ANALITICA, width=3.2),
                             fill="tonexty", fillcolor="rgba(41,182,246,0.12)"))
    fig.add_trace(go.Scatter(x=t_pts, y=T_pts, mode="markers", name=etiqueta_pts,
                             marker=dict(size=12, color="white", line=dict(color="#0D47A1", width=3)),
                             hovertemplate="t = %{x:g}<br>T = %{y:g} °C<extra></extra>"))
    estilo_figura(fig, f"Tiempo ({u})", "Temperatura (°C)")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10), hovermode="closest",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return fig


def pagina_constante_k():
    ss = st.session_state
    st.markdown(
        '<div class="nota"><b>¿Cómo se usa?</b> Elige cuántos datos tienes, escríbelos y obtendrás la constante '
        "de enfriamiento <b>k</b> con su procedimiento paso a paso. Luego puedes enviarla a la simulación.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    with st.container(border=True):
        encabezado_seccion("1", "Datos que tienes")
        a, b = st.columns([1.4, 1], gap="large")
        with a:
            st.radio("Método", MODOS_K, key="kc_modo", horizontal=True,
                     help="Una medición: conoces T₀, Tₐ y una temperatura posterior. "
                          "Dos mediciones: no conoces T₀. Tabla: varias medidas (ajuste por mínimos cuadrados).")
        with b:
            st.selectbox("Unidad de tiempo de tus medidas", list(UNIDADES_TIEMPO), key="kc_unidad")
        unidad = ss["kc_unidad"]
        u, seg = UNIDADES_TIEMPO[unidad]
        modo = ss["kc_modo"]

        resultado = None   # (k, T0, Ta, figura, pasos_latex, extra)
        if modo == "Una medición":
            st.caption("Datos básicos: temperatura inicial, temperatura ambiente y una temperatura medida después de un tiempo.")
            c1, c2, c3, c4 = st.columns(4)
            T0 = c1.number_input("T₀ inicial (°C)", key="kc_T0", min_value=LIM_T[0], max_value=LIM_T[1], step=1.0)
            Ta = c2.number_input("Tₐ ambiente (°C)", key="kc_Ta", min_value=LIM_T[0], max_value=1000.0, step=1.0)
            t1 = c3.number_input(f"Tiempo t₁ ({u})", key="kc_t1", min_value=0.0, step=1.0)
            T1 = c4.number_input("Temperatura T₁ en t₁ (°C)", key="kc_T1", min_value=LIM_T[0], max_value=LIM_T[1], step=1.0)
            k, err = k_una_medicion(T0, Ta, t1, T1)
            if err:
                st.error(err)
            else:
                q = (T1 - Ta) / (T0 - Ta)
                pasos = [
                    r"T(t)=T_a+(T_0-T_a)\,e^{-kt}\ \Rightarrow\ \frac{T_1-T_a}{T_0-T_a}=e^{-kt_1}",
                    rf"\frac{{T_1-T_a}}{{T_0-T_a}}=\frac{{{par(T1)}-{par(Ta)}}}{{{par(T0)}-{par(Ta)}}}={q:.4f}",
                    r"k=-\frac{1}{t_1}\,\ln\!\left(\frac{T_1-T_a}{T_0-T_a}\right)",
                    rf"k=-\frac{{1}}{{{t1:g}}}\ln({q:.4f})={k:.5f}\ \mathrm{{{u}}}^{{-1}}",
                ]
                resultado = dict(k=k, T0=T0, Ta=Ta, pasos=pasos, t=[0, t1], T=[T0, T1])
        elif modo == "Dos mediciones":
            st.caption("No necesitas T₀: basta la temperatura ambiente y dos mediciones en instantes distintos.")
            Ta = st.number_input("Tₐ ambiente (°C)", key="kc2_Ta", min_value=LIM_T[0], max_value=1000.0, step=1.0)
            c1, c2, c3, c4 = st.columns(4)
            t1 = c1.number_input(f"Tiempo t₁ ({u})", key="kc2_t1", min_value=0.0, step=1.0)
            T1 = c2.number_input("Temperatura T₁ (°C)", key="kc2_T1", min_value=LIM_T[0], max_value=LIM_T[1], step=1.0)
            t2 = c3.number_input(f"Tiempo t₂ ({u})", key="kc2_t2", min_value=0.0, step=1.0)
            T2 = c4.number_input("Temperatura T₂ (°C)", key="kc2_T2", min_value=LIM_T[0], max_value=LIM_T[1], step=1.0)
            k, T0e, err = k_dos_mediciones(Ta, t1, T1, t2, T2)
            if err:
                st.error(err)
            else:
                q = (T2 - Ta) / (T1 - Ta)
                pasos = [
                    r"\frac{T_2-T_a}{T_1-T_a}=\frac{e^{-kt_2}}{e^{-kt_1}}=e^{-k(t_2-t_1)}",
                    rf"\frac{{T_2-T_a}}{{T_1-T_a}}=\frac{{{par(T2)}-{par(Ta)}}}{{{par(T1)}-{par(Ta)}}}={q:.4f}",
                    r"k=-\frac{1}{t_2-t_1}\,\ln\!\left(\frac{T_2-T_a}{T_1-T_a}\right)",
                    rf"k=-\frac{{1}}{{{t2:g}-{par(t1)}}}\ln({q:.4f})={k:.5f}\ \mathrm{{{u}}}^{{-1}}",
                    rf"T_0=T_a+(T_1-T_a)\,e^{{kt_1}}={T0e:.2f}\ ^\circ\mathrm{{C}}\ \text{{(estimada)}}",
                ]
                resultado = dict(k=k, T0=T0e, Ta=Ta, pasos=pasos, t=[t1, t2], T=[T1, T2])
        else:
            st.caption("Escribe tus mediciones (puedes agregar filas con el + de la tabla). Con 3 o más puntos "
                       "el ajuste promedia los errores de medición.")
            tcol, gcol = st.columns([1.2, 1], gap="large")
            with tcol:
                st.session_state["medidas"] = st.data_editor(
                    st.session_state["medidas"], num_rows="dynamic", hide_index=True, width="stretch",
                    column_config={
                        "Tiempo": st.column_config.NumberColumn(f"Tiempo ({u})", min_value=0.0),
                        "Temperatura (°C)": st.column_config.NumberColumn("Temperatura (°C)", format="%.2f"),
                    },
                )
            with gcol:
                Ta = st.number_input("Tₐ ambiente (°C)", key="kc3_Ta", min_value=LIM_T[0], max_value=1000.0, step=1.0)
                st.button("Restaurar datos de ejemplo", key="kc_reset", width="stretch",
                          on_click=lambda: st.session_state.update(medidas=MEDIDAS_EJEMPLO.copy()))
            aj = k_por_ajuste(st.session_state["medidas"], Ta)
            if "error" in aj:
                st.error(aj["error"])
            else:
                pasos = [
                    r"\ln|T-T_a|=\ln|T_0-T_a|-k\,t\quad\text{(es una recta: pendiente }=-k\text{)}",
                    rf"\text{{pendiente}}={aj['pendiente']:.5f},\quad \text{{ordenada}}={aj['ordenada']:.4f}",
                    rf"k=-\text{{pendiente}}={aj['k']:.5f}\ \mathrm{{{u}}}^{{-1}},\qquad R^2={aj['r2']:.4f}",
                    rf"T_0=T_a\pm e^{{\text{{ordenada}}}}={aj['T0']:.2f}\ ^\circ\mathrm{{C}}\ \text{{(estimada)}}",
                ]
                resultado = dict(k=aj["k"], T0=aj["T0"], Ta=Ta, pasos=pasos, t=aj["t"], T=aj["T"], aj=aj)

    if resultado is None:
        return

    k, T0r, Tar = resultado["k"], resultado["T0"], resultado["Ta"]
    st.write("")
    encabezado_seccion("2", "Resultado")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        tarjeta("Constante k", f"{k:.5g} 1/{u}", f"= {k / seg:.3e} 1/s en el SI", COLOR_ANALITICA)
    with r2:
        tarjeta("Constante de tiempo τ = 1/k", f"{1 / k:.4g} {u}", "La diferencia cae a 37 %", COLOR_NUMERICA)
    with r3:
        tarjeta("Vida media (ln 2 / k)", f"{np.log(2) / k:.4g} {u}", "La diferencia se reduce a la mitad", COLOR_ACENTO)
    with r4:
        if "aj" in resultado:
            aj = resultado["aj"]
            tarjeta("Calidad del ajuste R²", f"{aj['r2']:.4f}", f"{aj['usados']} puntos usados"
                    + (f", {aj['descartados']} descartados" if aj["descartados"] else ""), COLOR_ERROR)
        else:
            tarjeta("T₀ usada", f"{T0r:.2f} °C", "Dato ingresado" if modo == "Una medición" else "Estimada con k", COLOR_ERROR)

    st.write("")
    g1, g2 = st.columns([1.25, 1], gap="medium")
    with g1:
        with st.container(border=True):
            st.markdown("#### Comprobación visual")
            st.caption("Si los puntos medidos caen sobre la curva, la constante k es coherente con tus datos.")
            st.plotly_chart(figura_medidas(resultado["t"], resultado["T"], T0r, Tar, k, u),
                            width="stretch", theme=None)
    with g2:
        with st.container(border=True):
            st.markdown("#### Procedimiento paso a paso")
            for paso in resultado["pasos"]:
                st.latex(paso)
        with st.container(border=True):
            st.markdown("#### ¿Y ahora?")
            st.caption("Envía k (y las temperaturas) a la simulación para ver la curva completa.")
            st.button("Usar en la simulación →", key="kc_usar", type="primary", width="stretch",
                      on_click=usar_k_en_simulacion, args=(k, unidad, T0r, Tar))
            if not (LIM_K[0] <= k <= LIM_K[1]):
                st.warning(f"k está fuera del rango permitido en la simulación ({LIM_K[0]} a {LIM_K[1]}); se ajustará al límite.")


# -----------------------------------------------------------------------------
# 11. PÁGINA: ERROR NUMÉRICO
# -----------------------------------------------------------------------------
def pagina_error():
    c = contexto()
    fichas(c)
    st.caption("Estos parámetros se editan en la pestaña Simulación.")
    col_e1, col_e2 = st.columns([1.5, 1], gap="medium")
    with col_e1:
        with st.container(border=True):
            st.markdown("#### Error absoluto en cada paso")
            st.caption("Diferencia entre la solución numérica y la analítica.")
            fig_e = go.Figure(go.Scatter(
                x=c.t_num, y=c.err, mode="lines", name="|T_num − T_ana|", line=dict(color=COLOR_ERROR, width=2.5),
                fill="tozeroy", fillcolor="rgba(121,134,203,0.15)",
                hovertemplate="t = %{x:.3f} " + c.u + "<br>error = %{y:.3e} °C<extra></extra>"))
            estilo_figura(fig_e, f"Tiempo ({c.u})", "Error absoluto (°C)")
            fig_e.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig_e, width="stretch", theme=None)
    with col_e2:
        with st.container(border=True):
            st.markdown("#### Comparación de métodos")
            st.caption(f"Con {c.n_pasos} pasos (h = {c.h:.4g} {c.u}).")
            filas = []
            for m in METODOS:
                _, T_m = resolver_numericamente(c.T0, c.Ta, c.k, c.t_total, c.n_pasos, m)
                e = np.abs(T_m - c.T_ana)
                filas.append({"Método": m, "Orden": ORDEN_METODO[m], "Error máx. (°C)": float(e.max()),
                              "Error RMS (°C)": float(np.sqrt(np.mean(e ** 2)))})
            st.dataframe(pd.DataFrame(filas).style.format({"Error máx. (°C)": "{:.2e}", "Error RMS (°C)": "{:.2e}"}),
                         hide_index=True, width="stretch")
            st.info("Un método de orden p reduce su error aproximadamente 2ᵖ veces al duplicar el número de pasos.")

    with st.container(border=True):
        st.markdown("#### Cómo baja el error al aumentar los pasos")
        st.caption("Escala logarítmica: mientras más inclinada la recta, más rápido converge el método.")
        ns = [5, 10, 20, 40, 80, 160, 320, 640, 1280]
        fig_c = go.Figure()
        for m, color in zip(METODOS, [COLOR_ERROR, COLOR_NUMERICA, COLOR_ANALITICA]):
            errores = []
            for n in ns:
                tt, TT = resolver_numericamente(c.T0, c.Ta, c.k, c.t_total, n, m)
                errores.append(max(float(np.max(np.abs(TT - solucion_analitica(tt, c.T0, c.Ta, c.k)))), 1e-17))
            fig_c.add_trace(go.Scatter(x=[c.t_total / n for n in ns], y=errores, mode="lines+markers",
                                       name=f"{m} (orden {ORDEN_METODO[m]})", line=dict(color=color, width=2.5)))
        estilo_figura(fig_c, f"Paso h ({c.u})", "Error máximo (°C)")
        fig_c.update_xaxes(type="log")
        fig_c.update_yaxes(type="log")
        fig_c.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        st.plotly_chart(fig_c, width="stretch", theme=None)


# -----------------------------------------------------------------------------
# 12. PÁGINA: DATOS
# -----------------------------------------------------------------------------
def pagina_datos():
    c = contexto()
    fichas(c)
    st.caption("Estos parámetros se editan en la pestaña Simulación.")
    df = pd.DataFrame({
        f"Tiempo ({c.u})": c.t_num, "T numérica (°C)": c.T_num,
        "T analítica (°C)": c.T_ana, "Error absoluto (°C)": c.err,
    })
    col_d1, col_d2 = st.columns([2, 1], gap="medium")
    with col_d1:
        with st.container(border=True):
            st.markdown("#### Tabla de la simulación")
            st.dataframe(df, width="stretch", height=430, hide_index=True)
    with col_d2:
        with st.container(border=True):
            st.markdown("#### Exportar")
            st.write(f"**{len(df):,}** filas · paso h = {c.h:.4g} {c.u} · {c.metodo}")
            csv = df.to_csv(index=False).encode("utf-8-sig")  # utf-8-sig: Excel reconoce los acentos
            st.download_button("Descargar CSV", data=csv, file_name="enfriamiento_newton.csv",
                               mime="text/csv", type="primary", width="stretch")
        with st.container(border=True):
            st.markdown("#### Parámetros usados")
            st.markdown(f"- **T₀:** {c.T0:g} °C\n- **Tₐ:** {c.Ta:g} °C\n- **k:** {c.k:.4g} 1/{c.u}\n"
                        f"- **Tiempo total:** {c.t_total:g} {c.u}\n- **Pasos:** {c.n_pasos}")


# -----------------------------------------------------------------------------
# 13. PÁGINA: TEORÍA
# -----------------------------------------------------------------------------
def pagina_teoria():
    u = UNIDADES_TIEMPO[st.session_state["unidad"]][0]
    col_t1, col_t2 = st.columns(2, gap="medium")
    with col_t1:
        with st.container(border=True):
            st.markdown("#### ¿Qué dice la ley?")
            st.write(
                "Isaac Newton observó que la rapidez con la que un cuerpo cambia de temperatura es "
                "proporcional a la diferencia entre su temperatura y la del medio que lo rodea. Cuanto mayor "
                "es esa diferencia, más rápido cambia la temperatura; al acercarse al equilibrio, el cambio "
                "se hace cada vez más lento."
            )
            st.markdown("**Ecuación diferencial**")
            st.latex(r"\frac{dT}{dt} = -k\,(T - T_a)")
            st.markdown("**Solución analítica** (separando variables e integrando)")
            st.latex(r"T(t) = T_a + (T_0 - T_a)\,e^{-kt}")
            st.markdown("**Tiempo para alcanzar una temperatura T₁**")
            st.latex(r"t = -\frac{1}{k}\,\ln\!\left(\frac{T_1 - T_a}{T_0 - T_a}\right)")
        with st.container(border=True):
            st.markdown("#### Cómo se obtiene k de las mediciones")
            st.write("Se despeja k de la solución analítica. Con una medición (t₁, T₁) y T₀ conocida:")
            st.latex(r"k = -\frac{1}{t_1}\,\ln\!\left(\frac{T_1 - T_a}{T_0 - T_a}\right)")
            st.write("Con dos mediciones, T₀ se cancela:")
            st.latex(r"k = -\frac{1}{t_2 - t_1}\,\ln\!\left(\frac{T_2 - T_a}{T_1 - T_a}\right)")
            st.write("Con varias, se grafica ln|T − Tₐ| contra t: es una recta de pendiente −k.")
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
            st.write("Tras 1 constante de tiempo (t = τ) la diferencia con el ambiente se reduce al 36,8 %; "
                     "tras 3τ, al 5 %; tras 5τ, a menos del 1 %.")
    with st.container(border=True):
        st.markdown("#### Parámetros y unidades")
        st.table(pd.DataFrame({
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
        }))


# -----------------------------------------------------------------------------
# 14. DESPACHO DE PÁGINAS
# -----------------------------------------------------------------------------
{
    "Inicio": pagina_inicio,
    "Simulación": pagina_simulacion,
    "Constante k": pagina_constante_k,
    "Error numérico": pagina_error,
    "Datos": pagina_datos,
    "Teoría": pagina_teoria,
}[st.session_state["pagina"]]()

st.markdown('<div class="pie">Simulación de la Ley de Enfriamiento de Newton</div>', unsafe_allow_html=True)

# streamlit run app.py