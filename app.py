import streamlit as st
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Importera ALLA nödvändiga funktioner
from main import run_pipeline, get_current_season_code
from model_handler import load_model
from xgboost import XGBClassifier
from data_loader import get_api_fixtures
from ui_utils import pick_half_guards, get_halfguard_sign
from utils import normalize_team_name

# Sätt upp sidans konfiguration och titel
st.set_page_config(page_title="Fotbollsmodellen V7", layout="wide")
st.title("⚽ Fotbollsmodellen V7")

# Konfigurera logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Konstanter för API-anrop ---
LEAGUE_IDS = {
    "Premier League (England)": 39,
    "Championship (England)": 40,
    "League One (England)": 41,
}

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

@st.cache_data(ttl=6 * 3600, show_spinner="Hämtar kommande matcher från API...")
def fetch_and_parse_fixtures(_league_id: int, _season: int) -> list[tuple[str, str]] | None:
    fixtures_json = get_api_fixtures(league_id=_league_id, season=_season)
    if not fixtures_json:
        st.error("Kunde inte hämta matcher från api-football. Kontrollera din API-nyckel och API-status.")
        return None
    
    parsed_matches = []
    for fixture in fixtures_json:
        home_raw = fixture.get('teams', {}).get('home', {}).get('name', '')
        away_raw = fixture.get('teams', {}).get('away', {}).get('name', '')
        home_team = normalize_team_name(home_raw)
        away_team = normalize_team_name(away_raw)
        if home_team and away_team:
            parsed_matches.append((home_team, away_team))
    return parsed_matches

# --- Generella hjälpfunktioner ---
def get_team_snapshot(team_name: str, df: pd.DataFrame) -> pd.Series | None:
    team_matches = df[(df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)]
    if team_matches.empty: return None
    return team_matches.iloc[-1]

# --- Ladda in nödvändiga resurser ---
SEASON_YEAR = int(get_current_season_code()[:2]) + 2000
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
#  HUVUD-GRÄNSSNITT
# ==============================================================================
st.header("Prediktera Matcher")

if not model or df_features is None:
    st.warning("Modell eller feature-data saknas. Kör en omträning med knappen i sidomenyn.")
else:
    league_selection = st.selectbox(
        "Välj en liga för att hämta kommande matcher:",
        options=list(LEAGUE_IDS.keys()),
        index=None,
        placeholder="Välj en liga..."
    )

    if league_selection:
        league_id = LEAGUE_IDS[league_selection]
        parsed_matches = fetch_and_parse_fixtures(league_id, SEASON_YEAR)

        if parsed_matches:
            st.subheader(f"Kommande matcher för {league_selection}")
            match_display_df = pd.DataFrame(parsed_matches, columns=["Hemmalag", "Bortalag"])
            match_display_df.index = np.arange(1, len(match_display_df) + 1)
            st.dataframe(match_display_df, use_container_width=True)

            num_guards = st.number_input("Antal halvgarderingar att föreslå:", min_value=0, max_value=len(parsed_matches), value=3, step=1)

            if st.button("Tippa Matcher", type="primary", use_container_width=True):
                match_probs_list = []
                # NYTT: Lista för att spara feature-data för XAI
                feature_details_list = []

                for home_team, away_team in parsed_matches:
                    home_stats = get_team_snapshot(home_team, df_features)
                    away_stats = get_team_snapshot(away_team, df_features)

                    if home_stats is None or away_stats is None:
                        match_probs_list.append(None)
                        feature_details_list.append(None) # Lägg till None även här
                        continue
                    
                    h_form_pts, h_form_gd, h_elo = (home_stats['HomeFormPts'], home_stats['HomeFormGD'], home_stats['HomeElo']) if home_stats['HomeTeam'] == home_team else (home_stats['AwayFormPts'], home_stats['AwayFormGD'], home_stats['AwayElo'])
                    a_form_pts, a_form_gd, a_elo = (away_stats['HomeFormPts'], away_stats['HomeFormGD'], away_stats['HomeElo']) if away_stats['HomeTeam'] == away_team else (away_stats['AwayFormPts'], away_stats['AwayFormGD'], away_stats['AwayElo'])
                    
                    # NYTT: Spara feature-värdena för XAI
                    feature_details_list.append({
                        'elo_diff': h_elo - a_elo,
                        'form_pts_diff': h_form_pts - a_form_pts
                    })

                    feature_vector = np.array([[h_form_pts, h_form_gd, a_form_pts, a_form_gd, h_elo, a_elo]])
                    probs = model.predict_proba(feature_vector)[0]
                    match_probs_list.append(probs)

                guard_indices = pick_half_guards(match_probs_list, num_guards)
                results = []

                for i, (home_team, away_team) in enumerate(parsed_matches):
                    probs = match_probs_list[i]
                    details = feature_details_list[i]
                    sign = "Data saknas"
                    
                    if probs is not None:
                        if i in guard_indices: sign = get_halfguard_sign(probs)
                        else: sign = ['1', 'X', '2'][np.argmax(probs)]
                    
                    results.append({
                        "Match": f"{home_team} - {away_team}",
                        "1": f"{probs[0]:.1%}" if probs is not None else "-",
                        "X": f"{probs[1]:.1%}" if probs is not None else "-",
                        "2": f"{probs[2]:.1%}" if probs is not None else "-",
                        "Tips": sign,
                        # NYTT: Lägg till de råa skillnaderna i resultatet
                        "elo_diff": details['elo_diff'] if details else 0,
                        "form_pts_diff": details['form_pts_diff'] if details else 0
                    })
                
                df_results = pd.DataFrame(results)
                
                # NYTT: Formatera XAI-kolumnerna för snyggare visning
                if not df_results.empty:
                    df_results['ELO-skillnad'] = df_results['elo_diff'].apply(lambda x: f"{x:+.0f}")
                    df_results['Form-skillnad (Poäng)'] = df_results['form_pts_diff'].apply(lambda x: f"{x:+.1f}")

                st.subheader("Resultat")
                # NYTT: Visa de nya kolumnerna i tabellen
                st.dataframe(
                    df_results[['Match', 'Tips', '1', 'X', '2', 'ELO-skillnad', 'Form-skillnad (Poäng)']],
                    use_container_width=True, 
                    hide_index=True
                )
                
                st.subheader("Tipsrad för kopiering")
                tips_string = " ".join(df_results['Tips'].tolist())
                st.code(tips_string, language=None)

# Felsökningsverktyget (kan vara kvar, dolt bakom URL-parameter)
if st.query_params.get("debug") == "true":
    # ... (kvarstår oförändrad)
