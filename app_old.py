import streamlit as st
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Importera nödvändiga funktioner
from main import run_pipeline, get_current_season_code
from model_handler import load_model
from xgboost import XGBClassifier
from ui_utils import get_halfguard_sign # Vi behöver inte längre alla ui_utils
from utils import normalize_team_name

# Sätt upp sidans konfiguration och titel
st.set_page_config(page_title="Fotbollsmodellen V7", layout="wide")
st.title("⚽ Fotbollsmodellen V7")

# Konfigurera logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Modell- och Data-laddning (med cache) ---
@st.cache_resource(show_spinner="Laddar maskininlärningsmodell...")
def load_cached_model(model_path: Path) -> XGBClassifier | None:
    if not model_path.exists(): return None
    return load_model(model_path)

@st.cache_data(show_spinner="Laddar historisk data för lag...")
def load_feature_data(path: Path) -> pd.DataFrame | None:
    if not path.exists(): return None
    try: return pd.read_parquet(path)
    except Exception as e: st.error(f"Kunde inte ladda feature-data: {e}"); return None

# --- NYTT: Funktion för att hämta alla kända lagnamn ---
@st.cache_data
def get_all_teams(_df_features: pd.DataFrame) -> list[str]:
    """ Extraherar en unik, sorterad lista av alla lagnamn från feature-datan. """
    if _df_features is None or _df_features.empty:
        return []
    unique_teams = pd.unique(_df_features[['HomeTeam', 'AwayTeam']].values.ravel('K'))
    return sorted([str(team) for team in unique_teams])


# --- Generella hjälpfunktioner ---
def get_team_snapshot(team_name: str, df: pd.DataFrame) -> pd.Series | None:
    team_matches = df[(df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)]
    if team_matches.empty: return None
    return team_matches.iloc[-1]

# --- Ladda in nödvändiga resurser ---
MODEL_FILENAME = f"xgboost_model_v7_{get_current_season_code()}.joblib"
model_path = Path("models") / MODEL_FILENAME
model = load_cached_model(model_path)
df_features = load_feature_data(Path("data") / "features.parquet")

# --- Sidebar ---
with st.sidebar:
    st.header("Systemstatus")
    if model: st.success(f"Modell laddad: `{MODEL_FILENAME}`")
    else: st.warning(f"Ingen modell laddad.")
    st.divider()
    st.header("Åtgärder")
    if st.button("Kör omträning av modell", help="Kör hela pipelinen."):
        with st.spinner("Pipeline körs..."):
            try:
                run_pipeline()
                st.success("Pipelinen är färdig!")
                st.cache_resource.clear()
                st.cache_data.clear()
                st.rerun()
            except Exception as e: st.error(f"Ett fel inträffade: {e}")

# ==============================================================================
#  HUVUD-GRÄNSSNITT - NU MED MANUELLT VAL
# ==============================================================================
st.header("Prediktera en enskild match")

if not model or df_features is None:
    st.warning("Modell eller feature-data saknas. Kör en omträning med knappen i sidomenyn.")
else:
    all_teams = get_all_teams(df_features)

    if not all_teams:
        st.error("Kunde inte ladda några lagnamn från datan. Kör omträning.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            home_team_selection = st.selectbox(
                "Välj hemmalag:",
                options=all_teams,
                index=None,
                placeholder="Skriv för att söka..."
            )
        with col2:
            away_team_selection = st.selectbox(
                "Välj bortalag:",
                options=all_teams,
                index=None,
                placeholder="Skriv för att söka..."
            )
        
        # Halvgardering är antingen på eller av för en enskild match
        use_halfguard = st.toggle("Visa halvgardering?")

        if st.button("Tippa Match", type="primary", use_container_width=True):
            if not home_team_selection or not away_team_selection:
                st.error("Du måste välja både ett hemmalag och ett bortalag.")
            elif home_team_selection == away_team_selection:
                st.error("Hemmalag och bortalag kan inte vara samma.")
            else:
                home_stats = get_team_snapshot(home_team_selection, df_features)
                away_stats = get_team_snapshot(away_team_selection, df_features)

                if home_stats is None or away_stats is None:
                    # Detta bör inte hända eftersom vi väljer från listan, men som en säkerhetsåtgärd
                    st.error("Kunde inte hitta historisk data för ett av de valda lagen.")
                else:
                    h_form_pts, h_form_gd, h_elo = (home_stats['HomeFormPts'], home_stats['HomeFormGD'], home_stats['HomeElo']) if home_stats['HomeTeam'] == home_team_selection else (home_stats['AwayFormPts'], home_stats['AwayFormGD'], home_stats['AwayElo'])
                    a_form_pts, a_form_gd, a_elo = (away_stats['HomeFormPts'], away_stats['HomeFormGD'], away_stats['HomeElo']) if away_stats['HomeTeam'] == away_team_selection else (away_stats['AwayFormPts'], away_stats['AwayFormGD'], away_stats['AwayElo'])
                    
                    feature_vector = np.array([[h_form_pts, h_form_gd, a_form_pts, a_form_gd, h_elo, a_elo]])
                    probs = model.predict_proba(feature_vector)[0]

                    if use_halfguard:
                        sign = get_halfguard_sign(probs)
                    else:
                        sign = ['1', 'X', '2'][np.argmax(probs)]
                    
                    result = {
                        "Match": f"{home_team_selection} - {away_team_selection}",
                        "1": f"{probs[0]:.1%}", "X": f"{probs[1]:.1%}", "2": f"{probs[2]:.1%}",
                        "Tips": sign,
                        "ELO-skillnad": f"{(h_elo - a_elo):+.0f}",
                        "Form-skillnad (Poäng)": f"{(h_form_pts - a_form_pts):+.1f}"
                    }
                    df_result = pd.DataFrame([result])

                    st.subheader("Resultat")
                    st.dataframe(df_result, use_container_width=True, hide_index=True)
                    st.subheader("Tipsrad för kopiering")
                    st.code(sign, language=None)

# Felsökningsverktyget (kan vara kvar, dolt bakom URL-parameter)
if st.query_params.get("debug") == "true":
    # ... (innehållet är detsamma, men kan vara bra att ha kvar)
    pass
