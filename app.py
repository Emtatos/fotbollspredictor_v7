# app.py
import streamlit as st
from pathlib import Path
import logging

# Importera nödvändiga funktioner från våra moduler
from main import run_pipeline, get_current_season_code
from model_handler import load_model
from xgboost import XGBClassifier

# Sätt upp sidans konfiguration och titel
st.set_page_config(page_title="Fotbollsmodellen V7", layout="wide")
st.title("⚽ Fotbollsmodellen V7")

# Konfigurera logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Modell-laddning (med cache) ---
@st.cache_resource(show_spinner="Laddar maskininlärningsmodell...")
def load_cached_model(model_path: Path) -> XGBClassifier | None:
    """
    En wrapper-funktion för att ladda modellen som kan cachas av Streamlit.
    Cache-nyckeln baseras på funktionen och dess argument (dvs. model_path).
    """
    if not model_path.exists():
        st.error(f"Modellfilen hittades inte på sökvägen: {model_path}")
        return None
    model = load_model(model_path)
    return model

# Bestäm vilken modell som ska laddas
SEASON = get_current_season_code()
MODEL_FILENAME = f"xgboost_model_v7_{SEASON}.joblib"
model_path = Path("models") / MODEL_FILENAME

# Ladda modellen
model = load_cached_model(model_path)

# --- Sidebar ---
with st.sidebar:
    st.header("Systemstatus")
    if model:
        st.success(f"Modell laddad: `{MODEL_FILENAME}`")
    else:
        st.warning("Ingen modell laddad. Träna modellen för att börja.")

    st.divider()
    st.header("Åtgärder")
    if st.button(
        "Kör omträning av modell",
        help="Detta kör hela pipelinen: laddar ner ny data, skapar features och tränar om modellen. Kan ta en stund."
    ):
        with st.spinner("Pipeline körs... Detta kan ta en stund."):
            try:
                run_pipeline()
                st.success("Pipelinen är färdig! Modellen har tränats om.")
                st.info("Appen laddas om för att använda den nya modellen...")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Ett fel inträffade i pipelinen: {e}")

# --- Huvud-gränssnitt ---
st.header("Prediktera Matcher")
if not model:
    st.info("Träna en modell med knappen i sidomenyn för att kunna göra prediktioner.")
else:
    st.markdown("Modellen är redo! Nästa steg är att bygga inmatning och resultattabell här.")
    # TODO:
    # 1) Text-area för att klistra in matcher.
    # 2) Knapp för att starta prediktion.
    # 3) Tabell för att visa resultat.
    # 4) (Ev.) XAI/feature-importance.
