"""
Fotbollspredictor v7 - Konsoliderad Streamlit-applikation

Denna app kombinerar det bästa från app.py och streamlit_app_v7.py:
- Modulär arkitektur med pipeline från main.py
- Avancerade funktioner som halvgarderingar och OpenAI-analys
- Förbättrad användarvänlighet och kodkvalitet
"""

import streamlit as st
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Optional, List, Tuple

# Importera nödvändiga funktioner från moduler
from main import run_pipeline, get_current_season_code
from model_handler import load_model
from xgboost import XGBClassifier
from ui_utils import get_halfguard_sign, pick_half_guards, parse_match_input
from utils import normalize_team_name, set_canonical_teams, get_canonical_teams

# News scraper för AI-analys
try:
    from news_scraper_v2 import get_match_context, IntelligentFootballAnalyzer
    HAS_NEWS_SCRAPER = True
except ImportError:
    HAS_NEWS_SCRAPER = False

# Injury scraper
try:
    from injury_scraper import InjuryDataFetcher, update_injury_data, get_injury_features_for_match
    HAS_INJURY_SCRAPER = True
except ImportError:
    HAS_INJURY_SCRAPER = False

# OpenAI (valfritt)
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# ============================================================================
# KONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Fotbollspredictor v7",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Konfigurera logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# HJÄLPFUNKTIONER
# ============================================================================

@st.cache_resource(show_spinner="Laddar maskininlärningsmodell...")
def load_cached_model(model_path: Path) -> Optional[XGBClassifier]:
    """Laddar modellen med caching för prestanda"""
    if not model_path.exists():
        return None
    return load_model(model_path)


@st.cache_data(show_spinner="Laddar historisk data för lag...")
def load_feature_data(path: Path) -> Optional[pd.DataFrame]:
    """Laddar feature-data från parquet-fil"""
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        st.error(f"Kunde inte ladda feature-data: {e}")
        return None


@st.cache_data
def get_all_teams(_df_features: pd.DataFrame) -> List[str]:
    """Extraherar en unik, sorterad lista av alla lagnamn från feature-datan"""
    if _df_features is None or _df_features.empty:
        return []
    unique_teams = pd.unique(_df_features[['HomeTeam', 'AwayTeam']].values.ravel('K'))
    return sorted([str(team) for team in unique_teams])


def get_team_snapshot(team_name: str, df: pd.DataFrame) -> Optional[pd.Series]:
    """Hämtar senaste matchdata för ett lag"""
    team_matches = df[(df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)]
    if team_matches.empty:
        return None
    return team_matches.iloc[-1]


def get_team_features(team_name: str, snapshot: pd.Series, df: pd.DataFrame) -> dict:
    """Extraherar alla features för ett lag från snapshot"""
    if snapshot['HomeTeam'] == team_name:
        return {
            'FormPts': snapshot['HomeFormPts'],
            'FormGD': snapshot['HomeFormGD'],
            'FormHome': snapshot['HomeFormHome'],
            'GoalsFor': snapshot['HomeGoalsFor'],
            'GoalsAgainst': snapshot['HomeGoalsAgainst'],
            'Streak': snapshot['HomeStreak'],
            'Position': snapshot['HomePosition'],
            'Elo': snapshot['HomeElo']
        }
    else:
        return {
            'FormPts': snapshot['AwayFormPts'],
            'FormGD': snapshot['AwayFormGD'],
            'FormHome': snapshot['AwayFormAway'],
            'GoalsFor': snapshot['AwayGoalsFor'],
            'GoalsAgainst': snapshot['AwayGoalsAgainst'],
            'Streak': snapshot['AwayStreak'],
            'Position': snapshot['AwayPosition'],
            'Elo': snapshot['AwayElo']
        }


def predict_match(
    model: XGBClassifier,
    home_team: str,
    away_team: str,
    df_features: pd.DataFrame
) -> Optional[Tuple[np.ndarray, dict]]:
    """
    Gör en prediktion för en match
    
    Returns:
        Tuple med (sannolikheter, statistik) eller None om data saknas
    """
    home_stats = get_team_snapshot(home_team, df_features)
    away_stats = get_team_snapshot(away_team, df_features)
    
    if home_stats is None or away_stats is None:
        return None
    
    home_features = get_team_features(home_team, home_stats, df_features)
    away_features = get_team_features(away_team, away_stats, df_features)
    
    # Beräkna H2H features från snapshot
    h2h_home_wins = home_stats.get('H2H_HomeWins', 0)
    h2h_draws = home_stats.get('H2H_Draws', 0)
    h2h_away_wins = home_stats.get('H2H_AwayWins', 0)
    h2h_home_goal_diff = home_stats.get('H2H_HomeGoalDiff', 0)
    position_diff = home_features['Position'] - away_features['Position']
    
    # Hämta skade-features om tillgängligt
    injury_features = {'InjuredPlayers_Home': 0, 'InjuredPlayers_Away': 0, 
                      'KeyPlayersOut_Home': 0, 'KeyPlayersOut_Away': 0,
                      'InjurySeverity_Home': 0, 'InjurySeverity_Away': 0}
    
    if HAS_INJURY_SCRAPER:
        try:
            injury_features = get_injury_features_for_match(home_team, away_team)
        except Exception as e:
            logger.warning(f"Kunde inte hämta skade-features: {e}")
    
    # Skapa feature vector med alla 27 features (21 original + 6 skade-features)
    feature_vector = np.array([[
        home_features['FormPts'],
        home_features['FormGD'],
        away_features['FormPts'],
        away_features['FormGD'],
        home_features['FormHome'],
        away_features['FormHome'],
        home_features['GoalsFor'],
        home_features['GoalsAgainst'],
        away_features['GoalsFor'],
        away_features['GoalsAgainst'],
        home_features['Streak'],
        away_features['Streak'],
        h2h_home_wins,
        h2h_draws,
        h2h_away_wins,
        h2h_home_goal_diff,
        home_features['Position'],
        away_features['Position'],
        position_diff,
        home_features['Elo'],
        away_features['Elo'],
        injury_features['InjuredPlayers_Home'],
        injury_features['InjuredPlayers_Away'],
        injury_features['KeyPlayersOut_Home'],
        injury_features['KeyPlayersOut_Away'],
        injury_features['InjurySeverity_Home'],
        injury_features['InjurySeverity_Away']
    ]])
    
    probs = model.predict_proba(feature_vector)[0]
    
    stats = {
        "home_form_pts": home_features['FormPts'],
        "home_form_gd": home_features['FormGD'],
        "home_elo": home_features['Elo'],
        "away_form_pts": away_features['FormPts'],
        "away_form_gd": away_features['FormGD'],
        "away_elo": away_features['Elo'],
        "home_goals_for": home_features['GoalsFor'],
        "away_goals_for": away_features['GoalsFor']
    }
    
    return probs, stats


def get_openai_analysis(
    home: str,
    away: str,
    probs: np.ndarray,
    stats: dict
) -> Optional[str]:
    """Genererar AI-analys av matchen med OpenAI"""
    if not HAS_OPENAI:
        return None
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
        except:
            pass
    
    if not api_key:
        return None
    
    try:
        client = OpenAI(api_key=api_key)
        
        prompt = f"""Du är en sportanalytiker. Ge en kort briefing inför matchen {home} - {away}.
Använd endast siffrorna nedan (inga påhittade nyheter eller skador):
- Hemma form (5): poäng {stats['home_form_pts']:.2f}, målskillnad {stats['home_form_gd']:.2f}
- Borta form (5): poäng {stats['away_form_pts']:.2f}, målskillnad {stats['away_form_gd']:.2f}
- ELO: {home} {stats['home_elo']:.1f}, {away} {stats['away_elo']:.1f}
- Modellens sannolikheter: 1={probs[0]:.1%}, X={probs[1]:.1%}, 2={probs[2]:.1%}

Svara med 3 korta punkter:
1) Styrkebalans (ELO) och hemmaprofil.
2) Formkurvor (5 matcher) och vad det antyder.
3) Kort riskbedömning (t.ex. hög osäkerhet om 2 utfall ligger nära)."""
        
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Du skriver kort, sakligt och utan spekulationer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=220,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI-analys misslyckades: {e}")
        return None


# ============================================================================
# LADDA RESURSER
# ============================================================================

MODEL_FILENAME = f"xgboost_model_v7_{get_current_season_code()}.joblib"
model_path = Path("models") / MODEL_FILENAME
model = load_cached_model(model_path)
df_features = load_feature_data(Path("data") / "features.parquet")

# Sätt kanoniska lagnamn om data finns
if df_features is not None and not df_features.empty:
    try:
        canon = set(df_features["HomeTeam"].dropna().astype(str)) | set(df_features["AwayTeam"].dropna().astype(str))
        set_canonical_teams(canon)
    except Exception as e:
        logger.warning(f"Kunde inte sätta kanoniska lagnamn: {e}")

# ============================================================================
# ANVÄNDARGRÄNSSNITT
# ============================================================================

st.title("⚽ Fotbollspredictor v7")
st.markdown("Prediktera matcher från Premier League (E0), Championship (E1), League One (E2) och League Two (E3)")

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("📊 Systemstatus")
    
    if model:
        st.success(f"✅ Modell laddad: `{MODEL_FILENAME}`")
    else:
        st.warning("⚠️ Ingen modell laddad")
    
    if df_features is not None:
        st.success(f"✅ Data laddad: {len(df_features)} matcher")
        all_teams = get_all_teams(df_features)
        st.info(f"📋 {len(all_teams)} lag tillgängliga")
    else:
        st.warning("⚠️ Ingen data laddad")
        all_teams = []
    
    # OpenAI-status
    if HAS_OPENAI and (os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)):
        st.success("✅ AI-analys tillgänglig")
    else:
        st.info("ℹ️ AI-analys ej tillgänglig")
    
    # Skade-data status
    if HAS_INJURY_SCRAPER:
        injury_file = Path('data/injuries_latest.json')
        if injury_file.exists():
            fetcher = InjuryDataFetcher()
            if fetcher.is_data_stale():
                st.warning("⚠️ Skadedata är gammal (>24h)")
            else:
                # Läs tidpunkt
                import json
                with open(injury_file, 'r') as f:
                    data = json.load(f)
                    last_update = data.get('last_updated', 'Okänd')
                    if last_update != 'Okänd':
                        last_update = datetime.fromisoformat(last_update).strftime("%Y-%m-%d %H:%M")
                st.success(f"✅ Skadedata uppdaterad: {last_update}")
        else:
            st.info("ℹ️ Skadedata saknas - klicka 'Uppdatera skador'")
    
    st.divider()
    
    st.header("🔧 Åtgärder")
    
    # Knapp för att uppdatera skadedata
    if HAS_INJURY_SCRAPER:
        if st.button("🎪 Uppdatera skador & form", help="Hämtar senaste skador och matchresultat", use_container_width=True):
            with st.spinner("🔄 Hämtar färsk data..."):
                try:
                    # Uppdatera skadedata
                    success = update_injury_data()
                    
                    if success:
                        st.success("✅ Skadedata uppdaterad!")
                        st.rerun()
                    else:
                        st.error("❌ Kunde inte uppdatera skadedata. Kontrollera API-nyckel.")
                except Exception as e:
                    st.error(f"❌ Fel vid uppdatering: {e}")
                    logger.error(f"Injury update failed: {e}", exc_info=True)
    
    if st.button("🔄 Kör omträning av modell", help="Tränar om modellen med alla 27 features (inkl. skador)", use_container_width=True):
        with st.spinner("🔄 Tränar modell med 27 features..."):
            try:
                # Använd quick-fix för att träna med 27 features
                from retrain_model_27features import retrain_with_injury_features
                model_path = retrain_with_injury_features()
                st.success(f"✅ Modell omtränad med 27 features!")
                st.info(f"💾 Sparad som: {model_path.name}")
                st.cache_resource.clear()
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Ett fel inträffade: {e}")
                logger.error(f"Retrain failed: {e}", exc_info=True)
                # Fallback till full pipeline
                try:
                    st.info("🔄 Försöker med full pipeline...")
                    run_pipeline()
                    st.success("✅ Pipelinen är färdig!")
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e2:
                    st.error(f"❌ Pipeline misslyckades också: {e2}")
                    logger.error(f"Pipeline fallback failed: {e2}", exc_info=True)

# ============================================================================
# HUVUDINNEHÅLL
# ============================================================================

if not model or df_features is None or not all_teams:
    st.warning("⚠️ Modell eller feature-data saknas. Tränar modellen automatiskt...")
    st.info("Detta kan ta 30-60 sekunder första gången. Var god vänta...")
    
    try:
        with st.spinner("Kör pipeline för att hämta data och träna modell..."):
            run_pipeline()
        st.success("✅ Modellen är tränad! Laddar om sidan...")
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
    except Exception as e:
        st.error(f"❌ Kunde inte träna modellen: {e}")
        st.info("Försök köra omträning manuellt med knappen i sidomenyn.")
        logger.error(f"Auto-training misslyckades: {e}", exc_info=True)
        st.stop()

# Skapa flikar för olika funktioner
tab1, tab2, tab3 = st.tabs(["🎯 Enskild Match", "📋 Flera Matcher", "ℹ️ Om Appen"])

# ============================================================================
# FLIK 1: ENSKILD MATCH
# ============================================================================

with tab1:
    st.header("Prediktera en enskild match")
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_team_selection = st.selectbox(
            "Välj hemmalag:",
            options=all_teams,
            index=None,
            placeholder="Skriv för att söka...",
            key="single_home"
        )
    
    with col2:
        away_team_selection = st.selectbox(
            "Välj bortalag:",
            options=all_teams,
            index=None,
            placeholder="Skriv för att söka...",
            key="single_away"
        )
    
    col3, col4 = st.columns(2)
    
    with col3:
        use_halfguard = st.toggle("Visa halvgardering?", key="single_halfguard")
    
    with col4:
        use_ai_analysis = st.toggle("Visa AI-analys?", key="single_ai", 
                                    disabled=not (HAS_OPENAI and (os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None))))
    
    if st.button("⚽ Tippa Match", type="primary", use_container_width=True):
        if not home_team_selection or not away_team_selection:
            st.error("❌ Du måste välja både ett hemmalag och ett bortalag.")
        elif home_team_selection == away_team_selection:
            st.error("❌ Hemmalag och bortalag kan inte vara samma.")
        else:
            result = predict_match(model, home_team_selection, away_team_selection, df_features)
            
            if result is None:
                st.error("❌ Kunde inte hitta historisk data för ett av de valda lagen.")
            else:
                probs, stats = result
                
                # Bestäm tips
                if use_halfguard:
                    sign = get_halfguard_sign(probs)
                else:
                    sign = ['1', 'X', '2'][np.argmax(probs)]
                
                # Visa resultat
                st.subheader("📊 Resultat")
                
                result_data = {
                    "Match": f"{home_team_selection} - {away_team_selection}",
                    "1 (Hemma)": f"{probs[0]:.1%}",
                    "X (Oavgjort)": f"{probs[1]:.1%}",
                    "2 (Borta)": f"{probs[2]:.1%}",
                    "Tips": sign,
                    "ELO-skillnad": f"{(stats['home_elo'] - stats['away_elo']):+.0f}",
                    "Form-skillnad": f"{(stats['home_form_pts'] - stats['away_form_pts']):+.1f}"
                }
                df_result = pd.DataFrame([result_data])
                st.dataframe(df_result, use_container_width=True, hide_index=True)
                
                # Visa tipsrad
                st.subheader("📝 Tipsrad för kopiering")
                st.code(sign, language=None)
                
                # AI-analys
                if use_ai_analysis:
                    with st.spinner("Genererar AI-analys..."):
                        analysis = get_openai_analysis(
                            home_team_selection,
                            away_team_selection,
                            probs,
                            stats
                        )
                        if analysis:
                            st.subheader("🤖 AI-analys")
                            st.info(analysis)
                        else:
                            st.warning("Kunde inte generera AI-analys")

# ============================================================================
# FLIK 2: FLERA MATCHER
# ============================================================================

with tab2:
    st.header("Prediktera flera matcher samtidigt")
    st.markdown("Skriv in matcher, en per rad. Format: `Hemmalag - Bortalag`")
    
    match_input = st.text_area(
        "Matcher:",
        height=200,
        placeholder="Arsenal - Chelsea\nLiverpool - Manchester United\nTottenham - Newcastle",
        key="multi_matches"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_halfguards = st.number_input(
            "Antal halvgarderingar:",
            min_value=0,
            max_value=10,
            value=0,
            key="multi_halfguards"
        )
    
    if st.button("⚽ Tippa Alla Matcher", type="primary", use_container_width=True):
        if not match_input.strip():
            st.error("❌ Skriv in minst en match.")
        else:
            # Säkerställ att kanoniska lag är satta innan parsing
            if df_features is not None and not df_features.empty:
                canon = set(df_features["HomeTeam"].dropna().astype(str)) | set(df_features["AwayTeam"].dropna().astype(str))
                set_canonical_teams(canon)
            
            matches = parse_match_input(match_input)
            
            if not matches:
                st.error("❌ Kunde inte tolka några matcher. Kontrollera formatet.")
                with st.expander("🔍 Felsökning"):
                    st.write("Antal rader i input:", len(match_input.strip().split('\n')))
                    st.write("Första raden:", match_input.strip().split('\n')[0] if match_input.strip() else "Tom")
                    st.write("Antal kanoniska lag:", len(get_canonical_teams()))
                    st.write("Exempel på kanoniska lag:", list(get_canonical_teams())[:10])
            else:
                st.subheader(f"📊 Resultat för {len(matches)} matcher")
                
                results = []
                all_probs = []
                
                for home, away in matches:
                    result = predict_match(model, home, away, df_features)
                    
                    if result is None:
                        results.append({
                            "Match": f"{home} - {away}",
                            "1": "N/A",
                            "X": "N/A",
                            "2": "N/A",
                            "Tips": "?",
                            "Status": "❌ Ingen data"
                        })
                        all_probs.append(None)
                    else:
                        probs, stats = result
                        all_probs.append(probs)
                        
                        sign = ['1', 'X', '2'][np.argmax(probs)]
                        
                        results.append({
                            "Match": f"{home} - {away}",
                            "1": f"{probs[0]:.1%}",
                            "X": f"{probs[1]:.1%}",
                            "2": f"{probs[2]:.1%}",
                            "Tips": sign,
                            "Status": "✅"
                        })
                
                # Applicera halvgarderingar
                if num_halfguards > 0:
                    guard_indices = pick_half_guards(all_probs, num_halfguards)
                    for idx in guard_indices:
                        if all_probs[idx] is not None:
                            results[idx]["Tips"] = get_halfguard_sign(all_probs[idx])
                            results[idx]["Status"] = "✅ (½)"
                
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True, hide_index=True)
                
                # Visa tipsrad
                st.subheader("📝 Tipsrad för kopiering")
                tipsrad = "".join([r["Tips"] for r in results if r["Tips"] != "?"])
                st.code(tipsrad, language=None)

# ============================================================================
# FLIK 3: OM APPEN
# ============================================================================

with tab3:
    st.header("Om Fotbollspredictor v7.6")
    
    st.markdown("""
    Fotbollspredictor v7.6 är en avancerad maskininlärningsapplikation designad för att prediktera fotbollsmatcher 
    med hög noggrannhet. Appen kombinerar statistisk analys med realtidsdata för att ge insiktsfulla och datadrivna förutsägelser.
    """)
    
    st.divider()
    
    st.subheader("🧠 Hur fungerar modellen?")
    
    st.markdown("""
    Modellen använder en **XGBoost-algoritm** (Extreme Gradient Boosting), en kraftfull och beprövad metod för 
    prediktiv modellering. Den tränas på tusentals historiska matcher från Premier League, Championship och League One.
    """)
    
    st.markdown("#### Features (27 totalt)")
    st.markdown("Modellen analyserar **27 olika features** för varje match. Dessa kan delas in i sex huvudkategorier:")
    
    feature_data = {
        "Kategori": ["Form", "Målstatistik", "Momentum", "Head-to-Head", "Styrka & Position", "Mänsklig påverkan"],
        "Antal": [6, 4, 2, 4, 5, 6],
        "Exempel på features": [
            "Genomsnittlig poäng, målskillnad (senaste 5 matcher)",
            "Genomsnitt gjorda/insläppta mål",
            "Vinst/förlust-streak",
            "Tidigare möten mellan lagen",
            "ELO-rating, ligaposition",
            "Skador, suspenderingar, nyckelspelare borta"
        ]
    }
    st.dataframe(feature_data, use_container_width=True, hide_index=True)
    
    st.info("""
    **Nytt i v7.6: Mänsklig påverkan**
    
    Den senaste versionen integrerar **skador och suspenderingar** via API-Football. Detta ger en mer realistisk bild 
    av lagens aktuella styrka.
    
    - **Datakälla:** API-Football (uppdateras dagligen)
    - **Nya features:** Antal skadade, antal nyckelspelare borta, allvarlighetsgrad (0-10)
    - **Användning:** Klicka "Uppdatera skador & form" i sidomenyn för att hämta färsk data.
    """)
    
    st.divider()
    
    st.subheader("🎯 Funktioner i appen")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Enskild match-prediktion:** Analysera en specifik match i detalj.
        - **Flera matcher:** Tippa en hel omgång samtidigt.
        - **Halvgarderingar:** Få förslag på vilka matcher som är mest osäkra.
        """)
    
    with col2:
        st.markdown("""
        - **AI-analys (valfritt):** OpenAI-driven textanalys av matchen.
        - **On-demand data-uppdatering:** Hämta färsk skadedata med en knapptryckning.
        - **Automatisk omträning:** Träna om modellen med den senaste datan.
        """)
    
    st.divider()
    
    st.subheader("🚀 Framtida förbättringsmöjligheter")
    
    st.markdown("För att ytterligare förbättra noggrannheten finns flera spännande möjligheter:")
    
    improvements_data = {
        "Förbättring": ["Tränarbyte", "Spelarbetyg", "Vilodagar", "Väder", "Historisk skadedata", "Live-odds", "Avancerad H2H"],
        "Beskrivning": [
            "Implementera 'new manager bounce'-effekten.",
            "Använd individuell spelarform istället för bara lagform.",
            "Analysera hur tätt matchschema påverkar prestation.",
            "Ta hänsyn till väderförhållanden (regn, vind, etc.).",
            "Träna modellen på historisk skadedata, inte bara aktuell.",
            "Jämför modellens prediktioner med live-odds från spelbolag.",
            "Analysera taktiska mönster i tidigare möten."
        ],
        "Potentiell påverkan": ["🔴 Hög", "🔴 Hög", "🟡 Medel", "🟡 Låg-Medel", "🔴 Hög", "🟡 Medel", "🟡 Medel"]
    }
    st.dataframe(improvements_data, use_container_width=True, hide_index=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Teknisk Stack")
        st.markdown("""
        - **Frontend:** Streamlit
        - **Backend:** Python
        - **ML-modell:** XGBoost, scikit-learn
        - **Datahantering:** pandas, numpy, pyarrow
        - **API-integration:** requests, python-dotenv
        - **Testning:** pytest, pytest-cov (46 tester)
        - **Deployment:** Render, Docker
        """)
    
    with col2:
        st.subheader("🔧 Utveckling & Kvalitet")
        st.markdown("""
        Projektet följer moderna best practices:
        - **Modulär arkitektur:** Lätt att underhålla och bygga ut.
        - **Automatiserad testning:** 42 enhetstester och 4 integrationstester.
        - **Prestandaoptimering:** 5-10x snabbare feature engineering.
        - **CI/CD-redo:** Automatisk deployment via GitHub och Render.
        - **Säkerhet:** API-nycklar hanteras via miljövariabler.
        """)
    
    st.divider()
    
    st.subheader("📝 Version")
    st.success("**v7.6.0** - 'Human Impact' Edition")
    
    st.subheader("🐛 Felsökning")
    
    st.markdown("""
    Om du stöter på problem:
    1. **Uppdatera skadedata:** Klicka "Uppdatera skador & form" i sidomenyn.
    2. **Kör omträning:** Klicka "Kör omträning av modell".
    3. **Kontrollera API-nyckel:** Verifiera att `API_FOOTBALL_KEY` är korrekt i Render.
    4. **Se loggar:** Kolla loggarna i Render Dashboard för felmeddelanden.
    """)
    
    st.divider()
    
    st.caption("Utvecklad av **Manus AI** på uppdrag av **Emtatos**.")
