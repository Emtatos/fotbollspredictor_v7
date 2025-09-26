import streamlit as st
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Importera nödvändiga funktioner från våra moduler
from main import run_pipeline, get_current_season_code
from model_handler import load_model
from xgboost import XGBClassifier
from ui_utils import parse_match_input
# VIKTIGT: Vi importerar INTE halvgarderingsfunktionerna i denna version
# from utils import normalize_team_name (importeras via ui_utils)

# Sätt upp sidans konfiguration och titel
st.set_page_config(page_title="Fotbollsmodellen V7", layout="wide")
st.title("⚽ Fotbollsmodellen V7")

# Konfigurera logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Modell-laddning (med cache) ---
@st.cache_resource(show_spinner="Laddar maskininlärningsmodell...")
def load_cached_model(model_path: Path) -> XGBClassifier | None:
    if not model_path.exists(): return None
    return load_model(model_path)

SEASON = get_current_season_code()
MODEL_FILENAME = f"xgboost_model_v7_{SEASON}.joblib"
model_path = Path("models") / MODEL_FILENAME
model = load_cached_model(model_path)

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

# --- Data-laddning och hjälpfunktioner ---
@st.cache_data(show_spinner="Laddar historisk data för lag...")
def load_feature_data(path: Path) -> pd.DataFrame | None:
    if not path.exists(): return None
    try: return pd.read_parquet(path)
    except Exception as e: st.error(f"Kunde inte ladda feature-data: {e}"); return None

def get_team_snapshot(team_name: str, df: pd.DataFrame) -> pd.Series | None:
    team_matches = df[(df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)]
    if team_matches.empty: return None
    return team_matches.iloc[-1]

df_features = None
st.header("Prediktera Matcher")

if not model:
    st.info("Träna en modell med knappen i sidomenyn för att kunna göra prediktioner.")
else:
    features_path = Path("data") / "features.parquet"
    df_features = load_feature_data(features_path)

    if df_features is None:
        st.warning("Feature-data (`features.parquet`) saknas. Kör en omträning.")
    else:
        st.subheader("Mata in din tipsrad")
        default_matches = ("Crystal Palace - Liverpool\nManchester City - Burnley\nLeeds - Bournemouth\nCharlton - Blackburn\nOxford - Sheffield United\nPreston - Bristol City\nSheffield Wednesday - Queens Park Rangers\nSouthampton - Middlesbrough\nStoke City - Norwich City\nWatford - Hull")
        match_input = st.text_area("Klistra in dina matcher, en per rad.", value=default_matches, height=250)
        
        if st.button("Tippa Matcher", type="primary", use_container_width=True):
            parsed_matches = parse_match_input(match_input)
            
            if not parsed_matches:
                st.error("Kunde inte tolka några matcher. Kontrollera formatet.")
            else:
                results = []
                for home_team, away_team in parsed_matches:
                    home_stats = get_team_snapshot(home_team, df_features)
                    away_stats = get_team_snapshot(away_team, df_features)

                    if home_stats is None or away_stats is None:
                        results.append({"Match": f"{home_team} - {away_team}", "Tips": "Data saknas"})
                    else:
                        h_form_pts, h_form_gd, h_elo = (home_stats['HomeFormPts'], home_stats['HomeFormGD'], home_stats['HomeElo']) if home_stats['HomeTeam'] == home_team else (home_stats['AwayFormPts'], home_stats['AwayFormGD'], home_stats['AwayElo'])
                        a_form_pts, a_form_gd, a_elo = (away_stats['HomeFormPts'], away_stats['HomeFormGD'], away_stats['HomeElo']) if away_stats['HomeTeam'] == away_team else (away_stats['AwayFormPts'], away_stats['AwayFormGD'], away_stats['AwayElo'])
                        feature_vector = np.array([[h_form_pts, h_form_gd, a_form_pts, a_form_gd, h_elo, a_elo]])
                        probs = model.predict_proba(feature_vector)[0]
                        prediction = np.argmax(probs)
                        sign = ['1', 'X', '2'][prediction]
                        results.append({"Match": f"{home_team} - {away_team}", "1": f"{probs[0]:.1%}", "X": f"{probs[1]:.1%}", "2": f"{probs[2]:.1%}", "Tips": sign})
                
                df_results = pd.DataFrame(results)
                st.subheader("Resultat")
                st.dataframe(df_results, use_container_width=True, hide_index=True)
                st.subheader("Tipsrad för kopiering")
                tips_string = " ".join(df_results['Tips'].tolist())
                st.code(tips_string, language=None)

# ==============================================================================
#  FELSÖKNINGSVERKTYG (Endast synligt för admin)
# ==============================================================================
if st.query_params.get("debug") == "true":
    st.divider()
    with st.expander("DEBUG: Inspektera Lagnamn i Dataset", expanded=True):
        if df_features is not None and not df_features.empty:
            try:
                unique_teams = pd.unique(df_features[['HomeTeam', 'AwayTeam']].values.ravel('K'))
                sorted_teams = sorted([str(team) for team in unique_teams])
                st.write(f"Hittade **{len(sorted_teams)}** unika lagnamn i `features.parquet`:")
                selected_teams = st.multiselect("Sök bland lagnamn...", options=sorted_teams)
                if selected_teams:
                    st.write("Visar all data för valda lag:")
                    st.dataframe(df_features[(df_features['HomeTeam'].isin(selected_teams)) | (df_features['AwayTeam'].isin(selected_teams))])
            except Exception as e:
                st.error(f"Kunde inte bearbeta lagnamn för felsökning: {e}")
        else:
            st.info("Datafilen (features.parquet) är inte laddad än.")
