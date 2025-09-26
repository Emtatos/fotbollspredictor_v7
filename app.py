import streamlit as st
from pathlib import Path
import logging
import pandas as pd
import numpy as np

# Importera nödvändiga funktioner från våra moduler
from main import run_pipeline, get_current_season_code
from model_handler import load_model
from xgboost import XGBClassifier
from ui_utils import parse_match_input

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
    """
    if not model_path.exists():
        # Detta är inte ett fel, utan ett normalt tillstånd vid första körning
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
        st.warning(f"Ingen modell laddad. Träna modellen för att börja.")
        st.info(f"Appen letade efter: `{model_path}`")


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
                # Rensa alla cachar för att tvinga omladdning av både modell och data
                st.cache_resource.clear()
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Ett fel inträffade i pipelinen: {e}")

# --- Data-laddning och Huvud-gränssnitt ---
@st.cache_data(show_spinner="Laddar historisk data för lag...")
def load_feature_data(path: Path) -> pd.DataFrame | None:
    """Laddar och cachar den sparade filen med features."""
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        st.error(f"Kunde inte ladda feature-data: {e}")
        return None

def get_team_snapshot(team_name: str, df_features: pd.DataFrame) -> pd.Series | None:
    """Hämtar den senaste raden med statistik för ett specifikt lag."""
    team_matches = df_features[(df_features['HomeTeam'] == team_name) | (df_features['AwayTeam'] == team_name)]
    if team_matches.empty:
        return None
    return team_matches.iloc[-1]

# **KORRIGERINGEN:** Initiera df_features till None för att garantera att variabeln alltid finns
df_features = None

st.header("Prediktera Matcher")

if not model:
    st.info("Träna en modell med knappen i sidomenyn för att kunna göra prediktioner.")
else:
    features_path = Path("data") / "features.parquet"
    df_features = load_feature_data(features_path)

    if df_features is None:
        st.warning("Feature-data (`features.parquet`) saknas. Kör en omträning för att skapa filen.")
    else:
        st.subheader("Mata in din tipsrad")
        default_matches = (
            "Crystal Palace - Liverpool\n"
            "Manchester City - Burnley\n"
            "Leeds - Bournemouth\n"
            "Charlton - Blackburn\n"
            "Oxford - Sheffield United\n"
            "Preston - Bristol City\n"
            "Sheffield Wednesday - Queens Park Rangers\n"
            "Southampton - Middlesbrough\n"
            "Stoke City - Norwich City\n"
            "Watford - Hull"
        )
        match_input = st.text_area(
            "Klistra in dina matcher, en per rad (t.ex. 'Heerenveen - Heracles').",
            value=default_matches,
            height=300
        )

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
                        results.append({
                            "Match": f"{home_team} - {away_team}", "1": "-", "X": "-", "2": "-",
                            "Tips": "Data saknas", "Sannolikhet": 0
                        })
                        continue

                    h_form_pts = home_stats['HomeFormPts'] if home_stats['HomeTeam'] == home_team else home_stats['AwayFormPts']
                    h_form_gd = home_stats['HomeFormGD'] if home_stats['HomeTeam'] == home_team else home_stats['AwayFormGD']
                    h_elo = home_stats['HomeElo'] if home_stats['HomeTeam'] == home_team else home_stats['AwayElo']
                    
                    a_form_pts = away_stats['HomeFormPts'] if away_stats['HomeTeam'] == away_team else away_stats['AwayFormPts']
                    a_form_gd = away_stats['HomeFormGD'] if away_stats['HomeTeam'] == away_team else away_stats['AwayFormGD']
                    a_elo = away_stats['AwayElo'] if away_stats['AwayTeam'] == away_team else away_stats['HomeElo']

                    feature_vector = np.array([[
                        h_form_pts, h_form_gd, a_form_pts, a_form_gd, h_elo, a_elo
                    ]])

                    probs = model.predict_proba(feature_vector)[0]
                    prediction = np.argmax(probs)
                    sign = ['1', 'X', '2'][prediction]
                    
                    results.append({
                        "Match": f"{home_team} - {away_team}", "1": f"{probs[0]:.1%}",
                        "X": f"{probs[1]:.1%}", "2": f"{probs[2]:.1%}",
                        "Tips": sign, "Sannolikhet": probs[prediction]
                    })
                
                df_results = pd.DataFrame(results)
                
                st.subheader("Resultat")
                st.dataframe(df_results[['Match', '1', 'X', '2', 'Tips']], use_container_width=True, hide_index=True)
                
                st.subheader("Tipsrad för kopiering")
                tips_string = " ".join(df_results['Tips'].tolist())
                st.code(tips_string, language=None)

# ==============================================================================
#  FELSÖKNINGSVERKTYG
# ==============================================================================
if st.query_params.get("debug") == "true":
st.divider()
with st.expander("DEBUG: Inspektera Lagnamn i Dataset"):
    if df_features is not None and not df_features.empty:
        try:
            unique_teams = pd.unique(df_features[['HomeTeam', 'AwayTeam']].values.ravel('K'))
            sorted_teams = sorted([str(team) for team in unique_teams])

            st.write(f"Hittade **{len(sorted_teams)}** unika lagnamn i `features.parquet`:")
            
            selected_teams = st.multiselect(
                "Sök bland lagnamn i den bearbetade datan för att verifiera stavning:",
                options=sorted_teams
            )
            
            if selected_teams:
                st.write("Visar all data för valda lag:")
                st.dataframe(df_features[(df_features['HomeTeam'].isin(selected_teams)) | (df_features['AwayTeam'].isin(selected_teams))])

        except Exception as e:
            st.error(f"Kunde inte bearbeta lagnamn för felsökning: {e}")
    else:
        st.info("Datafilen (features.parquet) är inte laddad. Kör pipelinen eller se till att filen finns.")
