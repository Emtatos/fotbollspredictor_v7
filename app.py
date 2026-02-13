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
from ui_utils import get_halfguard_sign, pick_half_guards, parse_match_input, calculate_match_entropy
from utils import normalize_team_name, set_canonical_teams, get_canonical_teams

# Konsistenta inference-moduler
from state import build_current_team_states
from features import compute_h2h
from inference import build_feature_row, predict_match as infer_predict_match
from schema import FEATURE_COLUMNS
from trust import compute_trust_features, trust_score
from metadata import load_metadata


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
    """Extraherar alla features för ett lag från snapshot (legacy).

    OBS: Denna funktion används inte längre för prediktion (vi använder state replay),
    men behålls för kompatibilitet med övrig UI-kod.
    """
    if snapshot['HomeTeam'] == team_name:
        return {
            'FormPts': snapshot.get('HomeFormPts', 0),
            'FormGD': snapshot.get('HomeFormGD', 0),
            'FormHome': snapshot.get('HomeFormHome', 0),
            'GoalsFor': snapshot.get('HomeGoalsFor', 0),
            'GoalsAgainst': snapshot.get('HomeGoalsAgainst', 0),
            'Streak': snapshot.get('HomeStreak', 0),
            'Position': snapshot.get('HomePosition', 0),
            'Elo': snapshot.get('HomeElo', 1500),
        }
    else:
        return {
            'FormPts': snapshot.get('AwayFormPts', 0),
            'FormGD': snapshot.get('AwayFormGD', 0),
            'FormHome': snapshot.get('AwayFormAway', 0),
            'GoalsFor': snapshot.get('AwayGoalsFor', 0),
            'GoalsAgainst': snapshot.get('AwayGoalsAgainst', 0),
            'Streak': snapshot.get('AwayStreak', 0),
            'Position': snapshot.get('AwayPosition', 0),
            'Elo': snapshot.get('AwayElo', 1500),
        }


def predict_match(
    model: XGBClassifier,
    home_team: str,
    away_team: str,
    df_features: pd.DataFrame
) -> Optional[Tuple[np.ndarray, dict]]:
    """
    Gör en prediktion för en match med konsekvent feature-kontrakt.

    Fixar:
      - använder current state (replay) i stället för senaste matchrad
      - räknar H2H för rätt lagpar
      - garanterar att predict_proba tolkas som [1, X, 2]
    """
    if df_features is None or df_features.empty:
        return None

    # Normalisera lagnamn för lookup
    home_team_n = normalize_team_name(home_team)
    away_team_n = normalize_team_name(away_team)

    hist_need = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    if any(c not in df_features.columns for c in hist_need):
        logger.error("df_features saknar nödvändiga kolumner för prediktion: %s",
                     [c for c in hist_need if c not in df_features.columns])
        return None

    hist = df_features.copy()
    hist["HomeTeam"] = hist["HomeTeam"].apply(normalize_team_name)
    hist["AwayTeam"] = hist["AwayTeam"].apply(normalize_team_name)

    states = build_current_team_states(hist)
    hs = states.get(home_team_n)
    as_ = states.get(away_team_n)
    if hs is None or as_ is None:
        return None

    h2h_hw, h2h_d, h2h_aw, h2h_gd = compute_h2h(hist, home_team_n, away_team_n)

    # Compute trust score
    trust_features = compute_trust_features(
        home_state=hs,
        away_state=as_,
        h2h_home_wins=h2h_hw,
        h2h_draws=h2h_d,
        h2h_away_wins=h2h_aw,
        league_code=hs.get("League", -1),
    )
    trust_score_val, trust_label = trust_score(trust_features)

    # Injury features (default 0; uppdateras om scraper finns)
    injury_features = {
        "InjuredPlayers_Home": 0,
        "InjuredPlayers_Away": 0,
        "KeyPlayersOut_Home": 0,
        "KeyPlayersOut_Away": 0,
        "InjurySeverity_Home": 0.0,
        "InjurySeverity_Away": 0.0,
    }
    if HAS_INJURY_SCRAPER:
        try:
            injury_features = get_injury_features_for_match(home_team_n, away_team_n)
        except Exception as e:
            logger.warning(f"Kunde inte hämta skade-features: {e}")

    feature_dict = {
        "HomeFormPts": hs["FormPts"],
        "HomeFormGD": hs["FormGD"],
        "AwayFormPts": as_["FormPts"],
        "AwayFormGD": as_["FormGD"],
        "HomeFormHome": hs["FormHome"],
        "AwayFormAway": as_["FormAway"],
        "HomeGoalsFor": hs["GoalsFor"],
        "HomeGoalsAgainst": hs["GoalsAgainst"],
        "AwayGoalsFor": as_["GoalsFor"],
        "AwayGoalsAgainst": as_["GoalsAgainst"],
        "HomeStreak": hs["Streak"],
        "AwayStreak": as_["Streak"],
        "H2H_HomeWins": h2h_hw,
        "H2H_Draws": h2h_d,
        "H2H_AwayWins": h2h_aw,
        "H2H_HomeGoalDiff": h2h_gd,
        "HomePosition": hs["Position"],
        "AwayPosition": as_["Position"],
        "PositionDiff": as_["Position"] - hs["Position"],  # samma definition som i feature_engineering
        "HomeElo": hs["Elo"],
        "AwayElo": as_["Elo"],
        "League": hs.get("League", -1),
        **injury_features,
    }

    row_df = build_feature_row(feature_dict)
    probs_map = infer_predict_match(model, row_df)  # {"1","X","2"}
    probs = np.array([probs_map["1"], probs_map["X"], probs_map["2"]], dtype=float)

    stats = {
        "home_form_pts": hs["FormPts"],
        "home_form_gd": hs["FormGD"],
        "home_elo": hs["Elo"],
        "away_form_pts": as_["FormPts"],
        "away_form_gd": as_["FormGD"],
        "away_elo": as_["Elo"],
        "home_goals_for": hs["GoalsFor"],
        "away_goals_for": as_["GoalsFor"],
        "trust_score": trust_score_val,
        "trust_label": trust_label,
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
model_metadata = load_metadata(Path("models"))

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
    
    if model_metadata:
        with st.expander("📋 Model Card"):
            st.markdown(f"**Version:** {model_metadata.get('model_version', 'N/A')}")
            st.markdown(f"**Tränad:** {model_metadata.get('train_date', 'N/A')[:10]}")
            cov = model_metadata.get("dataset_coverage", {})
            dr = cov.get("date_range", {})
            if dr.get("min") and dr.get("max"):
                st.markdown(f"**Dataperiod:** {dr['min'][:10]} – {dr['max'][:10]}")
            leagues = cov.get("leagues", [])
            if leagues:
                st.markdown(f"**Ligor:** {', '.join(leagues)}")
            feats = model_metadata.get("features", [])
            st.markdown(f"**Features:** {len(feats)}")
            groups = model_metadata.get("ablation_groups", [])
            if groups:
                st.markdown(f"**Ablation:** {', '.join(groups)}")
            use_odds = model_metadata.get("use_odds_features", False)
            st.markdown(f"**Odds-features:** {'Ja' if use_odds else 'Nej'}")
            st.markdown(f"**Kalibrering:** {model_metadata.get('calibration_method', 'N/A')}")
            splits = model_metadata.get("splits", {})
            if splits:
                st.markdown(f"**Split:** train={splits.get('train',0)}, cal={splits.get('calibrate',0)}, test={splits.get('test',0)}")
    
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
                
                trust_display = stats.get('trust_label', 'N/A')
                if trust_display == "LOW":
                    trust_display = "LOW (varning)"
                
                result_data = {
                    "Match": f"{home_team_selection} - {away_team_selection}",
                    "1 (Hemma)": f"{probs[0]:.1%}",
                    "X (Oavgjort)": f"{probs[1]:.1%}",
                    "2 (Borta)": f"{probs[2]:.1%}",
                    "Tips": sign,
                    "Trust": trust_display,
                    "TrustScore": stats.get('trust_score', 0),
                    "ELO-skillnad": f"{(stats['home_elo'] - stats['away_elo']):+.0f}",
                    "Form-skillnad": f"{(stats['home_form_pts'] - stats['away_form_pts']):+.1f}"
                }
                df_result = pd.DataFrame([result_data])
                st.dataframe(df_result, use_container_width=True, hide_index=True)
                
                # Feature contributions (top 5 via XGB)
                if hasattr(model, "feature_names_in_") or hasattr(model, "estimator"):
                    try:
                        from schema import get_expected_feature_columns
                        feat_names = get_expected_feature_columns(model)
                        base = model.estimator if hasattr(model, "estimator") else model
                        if hasattr(base, "get_booster"):
                            booster = base.get_booster()
                            importance = booster.get_score(importance_type="gain")
                            pairs = []
                            for fname, gain in importance.items():
                                idx = int(fname.replace("f", "")) if fname.startswith("f") and fname[1:].isdigit() else -1
                                name = feat_names[idx] if 0 <= idx < len(feat_names) else fname
                                pairs.append((name, float(gain)))
                            pairs.sort(key=lambda x: x[1], reverse=True)
                            top5 = pairs[:5]
                            with st.expander("Top 5 feature contributions (gain)"):
                                for rank, (name, gain) in enumerate(top5, 1):
                                    st.markdown(f"{rank}. **{name}** — gain {gain:.1f}")
                    except Exception:
                        pass

                if model_metadata:
                    use_odds = model_metadata.get("use_odds_features", False)
                    if not use_odds:
                        st.caption("Modellen tr\u00e4nades utan odds-features.")

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
                            "Entropy": "N/A",
                            "Trust": "N/A",
                            "Tips": "?",
                            "HALV": ""
                        })
                        all_probs.append(None)
                    else:
                        probs, stats = result
                        all_probs.append(probs)
                        
                        sign = ['1', 'X', '2'][np.argmax(probs)]
                        entropy = calculate_match_entropy(probs)
                        trust_lbl = stats.get('trust_label', 'N/A')
                        if trust_lbl == "LOW":
                            trust_lbl = "LOW (varning)"
                        
                        results.append({
                            "Match": f"{home} - {away}",
                            "1": f"{probs[0]:.1%}",
                            "X": f"{probs[1]:.1%}",
                            "2": f"{probs[2]:.1%}",
                            "Entropy": f"{entropy:.2f}" if entropy is not None else "N/A",
                            "Trust": trust_lbl,
                            "Tips": sign,
                            "HALV": ""
                        })
                
                # Applicera halvgarderingar
                if num_halfguards > 0:
                    guard_indices = pick_half_guards(all_probs, num_halfguards)
                    for idx in guard_indices:
                        if all_probs[idx] is not None:
                            results[idx]["Tips"] = get_halfguard_sign(all_probs[idx])
                            results[idx]["HALV"] = "HALV"
                
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
    st.header("Om Fotbollspredictor v8")
    
    st.markdown("""
    Fotbollspredictor v8 bygger vidare med en professionell ML-pipeline:
    unified FeatureBuilder, walk-forward validering, kalibrering, och stod for odds som valbar feature-grupp.
    """)
    
    st.divider()
    
    st.subheader("🧠 Hur fungerar modellen?")
    
    st.markdown("""
    Modellen anvander **XGBoost** med **CalibratedClassifierCV** (sigmoid/isotonic).
    Traning sker med tidssorterad split (train/calibrate/test) och walk-forward cross-validation (3 folds).
    Hyperparametrar optimeras via RandomizedSearchCV pa train-only data.
    """)
    
    if model_metadata:
        feats = model_metadata.get("features", [])
        n_feats = len(feats)
    else:
        n_feats = len(FEATURE_COLUMNS)
    
    st.markdown(f"#### Feature-grupper ({n_feats} features)")
    
    feature_data = {
        "Grupp": ["Base (form, ELO, H2H, position)", "Matchstats (shots, SOT, corners, cards)", "Odds (implied probs)", "Skador"],
        "Antal": [22, 24, 4, 6],
        "Beskrivning": [
            "FormPts, FormGD, Streak, Elo, Position, H2H, League",
            "Rolling shots/SOT (5,10), conversion, corner share, cards rate + has_matchstats",
            "has_odds, ImpliedHome, ImpliedDraw, ImpliedAway (valbar via USE_ODDS_FEATURES)",
            "InjuredPlayers, KeyPlayersOut, InjurySeverity (Home/Away)",
        ],
    }
    st.dataframe(feature_data, use_container_width=True, hide_index=True)
    
    st.info("""
    **Nytt i v8:**
    - **Unified FeatureBuilder** — samma logik for traning och inference, eliminerar mismatch
    - **Walk-forward validering** — 3 folds over tid, rapporterar mean + std
    - **Kalibrering** — sigmoid/isotonic pa separat calibration split
    - **Odds som valbar feature** — trana med/utan odds, jamfor i backtest-rapporten
    - **Matchstats** — rolling shots, SOT, conversion, corner share, cards rate
    - **Backtest-rapport** — logloss, brier, accuracy, F1, per liga/sasong, reliability bins
    """)
    
    st.divider()
    
    st.subheader("🎯 Funktioner i appen")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Enskild match-prediktion** med feature contributions (top 5 gain)
        - **Flera matcher** — tippa en hel omgang samtidigt
        - **Halvgarderingar** — forslag pa ossakra matcher
        """)
    
    with col2:
        st.markdown("""
        - **Model Card** i sidomenyn: version, traning, features, ablation
        - **Trust score** — datatackning per prediktion
        - **AI-analys (valfritt)** via OpenAI
        """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Teknisk Stack")
        st.markdown("""
        - **Frontend:** Streamlit
        - **ML:** XGBoost + scikit-learn (CalibratedClassifierCV)
        - **Data:** pandas, numpy, pyarrow
        - **Testning:** pytest (118+ tester, CI pa Python 3.9/3.10/3.11)
        - **Deployment:** Render, Docker
        """)
    
    with col2:
        st.subheader("🔧 Pipeline")
        st.markdown("""
        1. **Data ingestion** — download + normalize + Season extraction
        2. **Feature engineering** — unified FeatureBuilder (replay engine)
        3. **Training** — hyperparam search + walk-forward + calibration
        4. **Report** — backtest_report.md med alla metriker
        5. **Metadata** — metadata.json med full reproducerbarhet
        """)
    
    st.divider()
    
    st.subheader("📝 Version")
    if model_metadata:
        ver = model_metadata.get("model_version", "v8.0")
        st.success(f"**{ver}**")
    else:
        st.success("**v8.0**")
    
    st.subheader("🐛 Felsökning")
    
    st.markdown("""
    1. **Kör omträning** via sidomenyn.
    2. **Kontrollera API-nyckel** for skadedata (`API_FOOTBALL_KEY`).
    3. **Se loggar** i Render Dashboard.
    """)
    
    st.divider()
    
    st.caption("Utvecklad av **Emtatos**.")
