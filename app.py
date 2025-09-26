# app.py

import pandas as pd
import numpy as np
from ui_utils import parse_match_input
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

@st.cache_data(show_spinner="Laddar historisk data (features)...")
def load_feature_data(path: Path) -> pd.DataFrame | None:
    """Laddar och cachar den sparade filen med features."""
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        # Säkerställ sortering efter datum om inte redan
        if "Date" in df.columns:
            df = df.sort_values("Date").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Kunde inte ladda feature-data: {e}")
        return None

def get_team_snapshot(team_name: str, df_features: pd.DataFrame) -> pd.Series | None:
    """Hämtar den senaste raden (oavsett hemma/borta) för ett lag."""
    team_matches = df_features[(df_features['HomeTeam'] == team_name) | (df_features['AwayTeam'] == team_name)]
    if team_matches.empty:
        return None
    return team_matches.iloc[-1]  # senaste

if not model:
    st.info("Träna en modell med knappen i sidomenyn för att kunna göra prediktioner.")
else:
    features_path = Path("data") / "features.parquet"
    df_features = load_feature_data(features_path)

    if df_features is None or df_features.empty:
        st.warning("Feature-data (`data/features.parquet`) saknas. Kör omträning i sidomenyn för att skapa filen.")
    else:
        st.subheader("Mata in din tipsrad")
        default_matches = (
            "Arsenal - Aston Villa\n"
            "Bournemouth - Man United\n"
            "Liverpool - Crystal Palace\n"
            "Man City - Luton\n"
            "West Ham - Fulham\n"
            "Coventry - Birmingham\n"
            "Middlesbrough - Leeds\n"
            "Norwich - Bristol City\n"
            "Stoke - Plymouth\n"
            "Sunderland - Millwall\n"
            "West Brom - Leicester\n"
            "Bolton - Portsmouth\n"
            "Derby - Leyton Orient"
        )

        match_input = st.text_area(
            "Klistra in dina matcher, en per rad (t.ex. 'Arsenal - Chelsea').",
            value=default_matches,
            height=280
        )

        if st.button("Tippa Matcher", type="primary", use_container_width=True):
            parsed_matches = parse_match_input(match_input)

            if not parsed_matches:
                st.error("Kunde inte tolka några matcher. Kontrollera att du använder t.ex. 'Lag A - Lag B'.")
            else:
                rows = []
                tips = []
                for home_team, away_team in parsed_matches:
                    hs = get_team_snapshot(home_team, df_features)
                    as_ = get_team_snapshot(away_team, df_features)

                    if hs is None or as_ is None:
                        rows.append({
                            "Match": f"{home_team} - {away_team}",
                            "1": "-", "X": "-", "2": "-",
                            "Tips": "Data saknas"
                        })
                        tips.append("(X)")
                        continue

                    # Plocka form/elo utifrån om laget var hemma eller borta i sin senaste rad
                    h_form_pts = hs['HomeFormPts'] if hs['HomeTeam'] == home_team else hs['AwayFormPts']
                    h_form_gd  = hs['HomeFormGD'] if hs['HomeTeam'] == home_team else hs['AwayFormGD']
                    h_elo      = hs['HomeElo']     if hs['HomeTeam'] == home_team else hs['AwayElo']

                    a_form_pts = as_['HomeFormPts'] if as_['HomeTeam'] == away_team else as_['AwayFormPts']
                    a_form_gd  = as_['HomeFormGD'] if as_['HomeTeam'] == away_team else as_['AwayFormGD']
                    a_elo      = as_['AwayElo']     if as_['AwayTeam'] == away_team else as_['HomeElo']

                    X = np.array([[h_form_pts, h_form_gd, a_form_pts, a_form_gd, h_elo, a_elo]])
                    probs = model.predict_proba(X)[0]
                    pred_idx = int(np.argmax(probs))
                    sign = ['1', 'X', '2'][pred_idx]

                    rows.append({
                        "Match": f"{home_team} - {away_team}",
                        "1": f"{probs[0]:.1%}",
                        "X": f"{probs[1]:.1%}",
                        "2": f"{probs[2]:.1%}",
                        "Tips": sign
                    })
                    tips.append(f"({sign})")

                df_out = pd.DataFrame(rows)
                st.subheader("Resultat")
                st.dataframe(df_out[['Match', '1', 'X', '2', 'Tips']], use_container_width=True, hide_index=True)

                st.subheader("Tipsrad för kopiering")
                st.code(" ".join(tips), language=None)
