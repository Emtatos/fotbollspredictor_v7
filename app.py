"""
Odds & Value-verktyg — Streamlit-applikation

Huvudflöde: odds → fair probabilities → value (edge) → streckjämförelse → ranking.
Innehåller även modellbaserad prediktion (XGBoost) som komplement.
"""

import streamlit as st
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, Optional, List, Tuple

# Importera nödvändiga funktioner från moduler
from main import run_pipeline, get_current_season_code
from model_handler import load_model
from xgboost import XGBClassifier
from ui_utils import get_halfguard_sign, pick_half_guards, parse_match_input, calculate_match_entropy
from utils import normalize_team_name, set_canonical_teams, get_canonical_teams

# Konsistenta inference-moduler
from feature_builder import FeatureBuilder
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
    page_title="Odds & Value-verktyg",
    page_icon="📈",
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
    return load_model(model_path)


@st.cache_resource(show_spinner="Bygger FeatureBuilder...")
def _get_feature_builder(_df_features: pd.DataFrame) -> Optional[FeatureBuilder]:
    """Skapar och cachar en FeatureBuilder-instans baserad på historisk data.

    Cache-livscykel:
    Parametern har underscore-prefix (_df_features) så att Streamlit inte
    försöker hasha DataFrame:n.  Detta är säkert eftersom:
      1. df_features laddas en gång vid appstart via load_feature_data() och
         muteras aldrig under sessionens livstid.
      2. Vid omträning (retrain / pipeline) anropas st.cache_resource.clear()
         och st.cache_data.clear() följt av st.rerun(), vilket tvingar
         en ny FeatureBuilder att skapas med uppdaterad data.
    Resultatet är att exakt en FeatureBuilder-instans lever per Streamlit-
    appkörning, och den speglar alltid den senast inlästa datan.
    """
    if _df_features is None or _df_features.empty:
        return None
    builder = FeatureBuilder()
    builder.fit(_df_features)
    return builder


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


def predict_match(
    model: XGBClassifier,
    home_team: str,
    away_team: str,
    df_features: pd.DataFrame
) -> Optional[Tuple[np.ndarray, dict]]:
    """
    Gör en prediktion för en match via FeatureBuilder.features_for_match().

    Använder samma feature-logik som träningsvägen för att undvika
    train/inference-mismatch.
    """
    if df_features is None or df_features.empty:
        return None

    builder = _get_feature_builder(df_features)
    if builder is None:
        return None

    home_team_n = normalize_team_name(home_team)
    away_team_n = normalize_team_name(away_team)

    feature_dict = builder.features_for_match(home_team_n, away_team_n)
    if feature_dict is None:
        return None

    # Defensiv komplettering av skade-features om scraper finns
    if HAS_INJURY_SCRAPER:
        try:
            inj = get_injury_features_for_match(home_team_n, away_team_n)
            feature_dict.update(inj)
        except Exception as e:
            logger.warning(f"Kunde inte hämta skade-features: {e}")

    row_df = build_feature_row(feature_dict)
    probs_map = infer_predict_match(model, row_df)  # {"1","X","2"}
    probs = np.array([probs_map["1"], probs_map["X"], probs_map["2"]], dtype=float)

    # Compute trust score from builder state
    hs = builder.get_team_state(home_team_n)
    as_ = builder.get_team_state(away_team_n)
    h2h_hw = int(feature_dict.get("H2H_HomeWins", 0))
    h2h_d = int(feature_dict.get("H2H_Draws", 0))
    h2h_aw = int(feature_dict.get("H2H_AwayWins", 0))
    trust_features = compute_trust_features(
        home_state=hs or {},
        away_state=as_ or {},
        h2h_home_wins=h2h_hw,
        h2h_draws=h2h_d,
        h2h_away_wins=h2h_aw,
        league_code=int(feature_dict.get("League", -1)),
    )
    trust_score_val, trust_label = trust_score(trust_features)

    stats = {
        "home_form_pts": feature_dict.get("HomeFormPts", 0),
        "home_form_gd": feature_dict.get("HomeFormGD", 0),
        "home_elo": feature_dict.get("HomeElo", 1500),
        "away_form_pts": feature_dict.get("AwayFormPts", 0),
        "away_form_gd": feature_dict.get("AwayFormGD", 0),
        "away_elo": feature_dict.get("AwayElo", 1500),
        "home_goals_for": feature_dict.get("HomeGoalsFor", 0),
        "away_goals_for": feature_dict.get("AwayGoalsFor", 0),
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

st.title("📈 Odds & Value-verktyg")
st.markdown("Analysera odds, fair probabilities, value och streckjämförelse för engelska ligor")

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("📊 Status")
    
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
tab_odds, tab1, tab2, tab3 = st.tabs(["📈 Odds & Value", "🎯 Enskild Match", "📋 Flera Matcher", "ℹ️ Om Appen"])

# ============================================================================
# FLIK 1: ENSKILD MATCH
# ============================================================================

with tab1:
    st.header("Enskild match — modellprediktion")
    st.caption("Använder den tränade XGBoost-modellen för att prediktera utfall. Se Odds & Value-fliken för oddsanalys.")
    
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
    st.header("Flera matcher — modellprediktion")
    st.caption("Använder den tränade modellen. Se Odds & Value-fliken för oddsanalys.")
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
# FLIK: ODDSVERKTYG
# ============================================================================

with tab_odds:
    st.header("Odds, Value & Streckjämförelse")
    st.markdown(
        "Mata in odds → se fair probabilities → jämför value och streck → hitta intressanta utfall."
    )

    from odds_tool import (
        OddsEntry,
        build_match_report,
        build_reports_from_dataframe,
        extract_odds_from_row,
    )
    from value_analysis import (
        build_value_report,
        rank_outcomes_by_edge,
        rank_matches_by_interest,
    )
    from streck_analysis import (
        build_streck_report,
        build_streck_report_from_odds_report,
        rank_outcomes_by_streck_delta,
        rank_matches_by_streck_interest,
    )
    from streck_import import auto_load_streck

    with st.expander("Ordlista: edge, value, streck", expanded=False):
        st.markdown(
            "**Positiv edge** — jämförelsesannolikheten är högre än "
            "marknadens fair probability. Oddsen kan vara generösa.\n\n"
            "**Negativ edge** — marknaden ger högre sannolikhet än jämförelsen.\n\n"
            "**Expected return** — förväntad avkastning givet jämförelsesannolikheten och oddsen.\n\n"
            "**Understreckad** — streckandelen är lägre än fair probability (färre spelar än marknaden antyder).\n\n"
            "**Överstreckad** — streckandelen är högre än fair probability (fler spelar än marknaden antyder).\n\n"
            "*Beslutsstöd, inte garanti.*"
        )

    odds_mode = st.radio(
        "Inmatning:",
        ["Aktuell omgång (importera)", "Manuell (skriv in odds)", "Från data (historiska matcher)"],
        key="odds_mode",
        horizontal=True,
    )

    if odds_mode == "Aktuell omgång (importera)":
        from matchday_import import (
            parse_fixtures_csv,
            parse_odds_csv,
            parse_streck_csv,
            match_matchday_data,
            MatchdayImportStatus,
            MatchdayMatch,
            generate_fixtures_template,
            generate_odds_template,
            generate_streck_template,
            FIXTURES_TEMPLATE_CSV,
            ODDS_TEMPLATE_CSV,
            ODDS_MULTI_BM_TEMPLATE_CSV,
            ODDS_SIMPLE_TEMPLATE_CSV,
            STRECK_TEMPLATE_CSV,
        )

        st.subheader("Aktuell omgang — importera fixtures, odds & streck")
        st.markdown(
            "Ladda in aktuell omgangs matcher, odds och streckdata i tre steg. "
            "Appen matchar automatiskt och visar analys."
        )

        # ----- CSV-mallar for nedladdning -----
        with st.expander("Ladda ner CSV-mallar", expanded=False):
            st.markdown(
                "Anvand dessa mallar for att forbereda din data. "
                "Fyll i med aktuella varden och ladda upp nedan."
            )
            tcol1, tcol2, tcol3 = st.columns(3)
            with tcol1:
                st.download_button(
                    "Fixtures-mall (.csv)",
                    FIXTURES_TEMPLATE_CSV,
                    "fixtures_mall.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.caption("Kolumner: HomeTeam, AwayTeam")
            with tcol2:
                st.download_button(
                    "Odds-mall (.csv)",
                    ODDS_TEMPLATE_CSV,
                    "odds_mall.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.caption("Kolumner: HomeTeam, AwayTeam, B365H, B365D, B365A")
            with tcol3:
                st.download_button(
                    "Streck-mall (.csv)",
                    STRECK_TEMPLATE_CSV,
                    "streck_mall.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.caption("Kolumner: HomeTeam, AwayTeam, Streck1, StreckX, Streck2")

            st.markdown("---")
            st.markdown("**Oddsformat som stods:**")
            st.markdown(
                "- **football-data.co.uk-format:** B365H/B365D/B365A, PSH/PSD/PSA, BWH/BWD/BWA, etc.\n"
                "- **Enkelt format:** Home/Draw/Away\n"
                "- Flera bookmakers per rad stods (t.ex. B365 + Pinnacle)."
            )
            st.download_button(
                "Odds-mall med flera bookmakers (.csv)",
                ODDS_MULTI_BM_TEMPLATE_CSV,
                "odds_multi_bm_mall.csv",
                mime="text/csv",
            )
            st.download_button(
                "Odds-mall enkelt format (.csv)",
                ODDS_SIMPLE_TEMPLATE_CSV,
                "odds_simple_mall.csv",
                mime="text/csv",
            )

        st.markdown("---")

        # ----- Steg 1: Fixtures -----
        st.markdown("### Steg 1: Ladda in matcher (fixtures)")
        fixtures_file = st.file_uploader(
            "CSV med matcher (HomeTeam, AwayTeam)",
            type=["csv"],
            key="matchday_fixtures_upload",
        )

        # ----- Steg 2: Odds -----
        st.markdown("### Steg 2: Ladda in odds")
        odds_file = st.file_uploader(
            "CSV med odds (HomeTeam, AwayTeam + oddskolumner)",
            type=["csv"],
            key="matchday_odds_upload",
        )

        # ----- Steg 3: Streck -----
        st.markdown("### Steg 3: Ladda in streckprocent")
        streck_file = st.file_uploader(
            "CSV med streck (HomeTeam, AwayTeam, Streck1, StreckX, Streck2)",
            type=["csv"],
            key="matchday_streck_upload",
        )

        st.markdown("---")

        # ----- Bearbeta import -----
        if fixtures_file is not None or odds_file is not None:
            all_import_errors: List[str] = []
            fixtures_list = []
            odds_by_key: Dict = {}
            streck_by_key: Dict = {}
            odds_rows_loaded = 0
            streck_rows_loaded = 0

            # Parse fixtures
            if fixtures_file is not None:
                try:
                    fixtures_df = pd.read_csv(fixtures_file)
                    fixtures_list, fix_errors = parse_fixtures_csv(fixtures_df)
                    all_import_errors.extend(fix_errors)
                except Exception as e:
                    all_import_errors.append(f"Kunde inte lasa fixtures-CSV: {e}")
            else:
                st.info("Ladda upp en fixtures-fil for att starta.")

            # Parse odds
            if odds_file is not None:
                try:
                    odds_df = pd.read_csv(odds_file)
                    odds_by_key, odds_rows_loaded, odds_errors = parse_odds_csv(odds_df)
                    all_import_errors.extend(odds_errors)
                except Exception as e:
                    all_import_errors.append(f"Kunde inte lasa odds-CSV: {e}")

                # If no fixtures file, create fixtures from odds
                if not fixtures_list and odds_by_key:
                    st.info(
                        "Ingen separat fixtures-fil uppladdad — "
                        "anvander matcher fran odds-filen."
                    )
                    from matchday_import import MatchdayFixture
                    for key, entries in odds_by_key.items():
                        parts = key.split("_", 1)
                        if len(parts) == 2:
                            fixtures_list.append(MatchdayFixture(
                                home_team=parts[0],
                                away_team=parts[1],
                                match_key=key,
                            ))

            # Parse streck
            if streck_file is not None:
                try:
                    streck_df = pd.read_csv(streck_file)
                    streck_by_key, streck_rows_loaded, streck_errors = parse_streck_csv(streck_df)
                    all_import_errors.extend(streck_errors)
                except Exception as e:
                    all_import_errors.append(f"Kunde inte lasa streck-CSV: {e}")

            # Match data
            if fixtures_list:
                matchday_matches, import_status = match_matchday_data(
                    fixtures_list, odds_by_key, streck_by_key,
                )

                # ---- Importstatus-dashboard ----
                st.subheader("Importstatus")
                scol1, scol2, scol3, scol4 = st.columns(4)
                with scol1:
                    st.metric("Fixtures", import_status.fixtures_count)
                with scol2:
                    st.metric("Med odds", import_status.odds_matched)
                with scol3:
                    st.metric("Med streck", import_status.streck_matched)
                with scol4:
                    st.metric("Komplett", import_status.fully_matched)

                # Visa omatchade
                has_unmatched = (
                    import_status.fixtures_without_odds
                    or import_status.fixtures_without_streck
                    or import_status.unmatched_odds
                    or import_status.unmatched_streck
                )
                if has_unmatched:
                    with st.expander("Omatchade rader", expanded=False):
                        if import_status.fixtures_without_odds:
                            st.markdown("**Fixtures utan odds:**")
                            for item in import_status.fixtures_without_odds:
                                st.caption(f"- {item}")
                        if import_status.fixtures_without_streck:
                            st.markdown("**Fixtures utan streck:**")
                            for item in import_status.fixtures_without_streck:
                                st.caption(f"- {item}")
                        if import_status.unmatched_odds:
                            st.markdown("**Oddsrader utan matchande fixture:**")
                            for item in import_status.unmatched_odds:
                                st.caption(f"- {item}")
                        if import_status.unmatched_streck:
                            st.markdown("**Streckrader utan matchande fixture:**")
                            for item in import_status.unmatched_streck:
                                st.caption(f"- {item}")

                # Visa felmeddelanden
                if all_import_errors:
                    with st.expander(f"Felmeddelanden ({len(all_import_errors)})", expanded=False):
                        for err in all_import_errors:
                            st.caption(f"- {err}")

                st.markdown("---")

                # ---- Analys for matchade matcher ----
                matches_with_odds = [m for m in matchday_matches if m.has_odds and m.odds_report]

                if matches_with_odds:
                    # --- Ranking: snabboversikt ---
                    value_reports_list = [
                        m.value_report for m in matches_with_odds
                        if m.value_report is not None
                    ]

                    if value_reports_list:
                        from value_analysis import rank_outcomes_by_edge, rank_matches_by_interest
                        st.subheader("Snabboversikt — mest intressanta utfall")
                        ranked_outcomes = rank_outcomes_by_edge(value_reports_list)

                        positive_outcomes = [r for r in ranked_outcomes if r[2].edge > 0.001]
                        negative_outcomes = [r for r in ranked_outcomes if r[2].edge < -0.001]

                        if positive_outcomes:
                            st.markdown("**Hogst positiv edge:**")
                            pos_rows = []
                            for match_label, outcome_label, ov in positive_outcomes[:10]:
                                pos_rows.append({
                                    "Match": match_label,
                                    "Utfall": outcome_label,
                                    "Odds": f"{ov.odds:.2f} ({ov.bookmaker})",
                                    "Edge": f"{ov.edge:+.1%}",
                                    "Exp. return": f"{ov.expected_return:+.1%}",
                                    "Bedomning": ov.edge_label,
                                })
                            st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

                        if negative_outcomes:
                            st.markdown("**Mest negativa edge:**")
                            neg_rows = []
                            for match_label, outcome_label, ov in reversed(negative_outcomes[-5:]):
                                neg_rows.append({
                                    "Match": match_label,
                                    "Utfall": outcome_label,
                                    "Odds": f"{ov.odds:.2f} ({ov.bookmaker})",
                                    "Edge": f"{ov.edge:+.1%}",
                                    "Exp. return": f"{ov.expected_return:+.1%}",
                                    "Bedomning": ov.edge_label,
                                })
                            st.dataframe(pd.DataFrame(neg_rows), use_container_width=True, hide_index=True)

                        if not positive_outcomes and not negative_outcomes:
                            st.info("Ingen tydlig edge hittades bland aktuella matcher.")

                    # --- Streckranking ---
                    streck_reports_list = [
                        m.streck_report for m in matches_with_odds
                        if m.streck_report is not None
                    ]
                    if streck_reports_list:
                        from streck_analysis import (
                            rank_outcomes_by_streck_delta,
                            rank_matches_by_streck_interest,
                        )
                        st.divider()
                        st.subheader("Streckjamforelse — over- & understreckade")

                        ranked_streck_outcomes = rank_outcomes_by_streck_delta(streck_reports_list)

                        understreck = [r for r in ranked_streck_outcomes if r[2].delta < -0.001]
                        if understreck:
                            st.markdown("**Mest understreckade utfall:**")
                            us_rows = []
                            for match_label, outcome_label, os_item in understreck[:10]:
                                us_rows.append({
                                    "Match": match_label,
                                    "Utfall": outcome_label,
                                    "Fair prob": f"{os_item.fair_prob:.1%}",
                                    "Streck": f"{os_item.streck_pct:.1%}",
                                    "Delta": f"{os_item.delta:+.1%}",
                                    "Bedomning": "understreckad",
                                })
                            st.dataframe(pd.DataFrame(us_rows), use_container_width=True, hide_index=True)

                        overstreck = [r for r in reversed(ranked_streck_outcomes) if r[2].delta > 0.001]
                        if overstreck:
                            st.markdown("**Mest overstreckade utfall:**")
                            os_rows = []
                            for match_label, outcome_label, os_item in overstreck[:10]:
                                os_rows.append({
                                    "Match": match_label,
                                    "Utfall": outcome_label,
                                    "Fair prob": f"{os_item.fair_prob:.1%}",
                                    "Streck": f"{os_item.streck_pct:.1%}",
                                    "Delta": f"{os_item.delta:+.1%}",
                                    "Bedomning": "overstreckad",
                                })
                            st.dataframe(pd.DataFrame(os_rows), use_container_width=True, hide_index=True)

                        ranked_streck_matches = rank_matches_by_streck_interest(streck_reports_list)
                        if ranked_streck_matches:
                            st.markdown("**Mest intressanta matcher (storst streckavvikelse):**")
                            mi_rows = []
                            for sr in ranked_streck_matches[:10]:
                                mi_rows.append({
                                    "Match": f"{sr.home_team} vs {sr.away_team}",
                                    "Storsta avvikelse": f"{sr.max_abs_delta:+.1%}",
                                })
                            st.dataframe(pd.DataFrame(mi_rows), use_container_width=True, hide_index=True)

                    st.divider()

                    # --- Detaljvy per match ---
                    st.subheader("Detaljvy per match")

                    for match in matchday_matches:
                        if not match.odds_report:
                            # Match without odds — show minimal info
                            with st.expander(
                                f"{match.home_team} vs {match.away_team}  |  Inga odds"
                            ):
                                st.caption("Ingen oddsdata tillganglig for denna match.")
                                if match.has_streck and match.streck:
                                    st.markdown(
                                        f"**Streck:** 1={match.streck['1']:.0f}% / "
                                        f"X={match.streck['X']:.0f}% / "
                                        f"2={match.streck['2']:.0f}%"
                                    )
                            continue

                        rpt = match.odds_report
                        vr = match.value_report

                        # Build expander label
                        if vr is not None and vr.outcomes:
                            max_edge = max(ov.edge for ov in vr.outcomes)
                            edge_hint = f"  |  Hogsta edge: {max_edge:+.1%}" if abs(max_edge) > 0.001 else ""
                        else:
                            edge_hint = ""

                        status_parts = []
                        if match.has_odds:
                            status_parts.append("odds")
                        if match.has_streck:
                            status_parts.append("streck")
                        status_tag = " + ".join(status_parts) if status_parts else "partiell"

                        with st.expander(
                            f"{match.home_team} vs {match.away_team}  |  "
                            f"Overround: {rpt.overround:.1f}%{edge_hint}  [{status_tag}]"
                        ):
                            num_bm = len(rpt.bookmaker_odds)

                            # Bookmaker odds table
                            bm_rows = []
                            for e in rpt.bookmaker_odds:
                                bm_rows.append({
                                    "Bookmaker": e.bookmaker,
                                    "1 (Hemma)": f"{e.home:.2f}",
                                    "X (Oavgjort)": f"{e.draw:.2f}",
                                    "2 (Borta)": f"{e.away:.2f}",
                                })
                            st.dataframe(pd.DataFrame(bm_rows), use_container_width=True, hide_index=True)

                            # Value analysis
                            if vr is not None:
                                st.markdown("**Value-analys:**")
                                st.caption(f"Jamforelsekalla: {vr.comparison_source}")
                                value_rows = []
                                for ov in vr.outcomes:
                                    label = {"1": "Hemma (1)", "X": "Oavgjort (X)", "2": "Borta (2)"}[ov.outcome]
                                    value_rows.append({
                                        "Utfall": label,
                                        "Odds": f"{ov.odds:.2f} ({ov.bookmaker})",
                                        "Market fair prob": f"{ov.market_fair_prob:.1%}",
                                        "Comparison prob": f"{ov.comparison_prob:.1%}",
                                        "Edge": f"{ov.edge:+.1%}",
                                        "Expected return": f"{ov.expected_return:+.1%}",
                                        "Bedomning": ov.edge_label,
                                    })
                                st.dataframe(pd.DataFrame(value_rows), use_container_width=True, hide_index=True)
                            else:
                                # Fallback: show probabilities only
                                prob_table = {
                                    "Utfall": ["Hemma (1)", "Oavgjort (X)", "Borta (2)"],
                                    "Implicit sannol.": [
                                        f"{rpt.implied_probs['1']:.1%}",
                                        f"{rpt.implied_probs['X']:.1%}",
                                        f"{rpt.implied_probs['2']:.1%}",
                                    ],
                                    "Fair sannol.": [
                                        f"{rpt.fair_probs['1']:.1%}",
                                        f"{rpt.fair_probs['X']:.1%}",
                                        f"{rpt.fair_probs['2']:.1%}",
                                    ],
                                }
                                st.dataframe(pd.DataFrame(prob_table), use_container_width=True, hide_index=True)

                            # Best odds
                            if num_bm > 1 and rpt.best_odds:
                                st.markdown("**Basta odds per utfall:**")
                                for outcome in ["1", "X", "2"]:
                                    if outcome in rpt.best_odds:
                                        oval, obm = rpt.best_odds[outcome]
                                        label = {"1": "Hemma", "X": "Oavgjort", "2": "Borta"}[outcome]
                                        st.write(f"- {label} ({outcome}): **{oval:.2f}** ({obm})")

                            # Streck comparison
                            if match.streck_report is not None:
                                st.markdown("**Streckjamforelse:**")
                                streck_rows = []
                                for os_item in match.streck_report.outcomes:
                                    olabel = {"1": "Hemma (1)", "X": "Oavgjort (X)", "2": "Borta (2)"}[os_item.outcome]
                                    if os_item.label == "understreckad":
                                        badge = "understreckad"
                                    elif os_item.label == "overstreckad":
                                        badge = "overstreckad"
                                    else:
                                        badge = "neutral"
                                    streck_rows.append({
                                        "Utfall": olabel,
                                        "Fair market prob": f"{os_item.fair_prob:.1%}",
                                        "Streckprocent": f"{os_item.streck_pct:.1%}",
                                        "Delta": f"{os_item.delta:+.1%}",
                                        "Bedomning": badge,
                                    })
                                st.dataframe(pd.DataFrame(streck_rows), use_container_width=True, hide_index=True)
                            elif match.has_streck and match.streck:
                                st.markdown(
                                    f"**Streck (ra):** 1={match.streck['1']:.0f}% / "
                                    f"X={match.streck['X']:.0f}% / "
                                    f"2={match.streck['2']:.0f}%"
                                )

                    st.caption(
                        "Value-analysen bygger pa marknadskonsensus mellan bookmakers. "
                        "Streckjamforelsen visar skillnaden mellan folkets streck och "
                        "marknadens fair probability."
                    )

                elif not matches_with_odds and matchday_matches:
                    st.warning(
                        "Inga matcher har giltiga odds. "
                        "Ladda upp en odds-fil for att se analys."
                    )

                    # Show fixtures without odds as a list
                    st.markdown("**Laddade fixtures:**")
                    for m in matchday_matches:
                        streck_info = ""
                        if m.has_streck and m.streck:
                            streck_info = (
                                f" — Streck: 1={m.streck['1']:.0f}% / "
                                f"X={m.streck['X']:.0f}% / "
                                f"2={m.streck['2']:.0f}%"
                            )
                        st.caption(f"- {m.home_team} vs {m.away_team}{streck_info}")
            else:
                st.info(
                    "Ladda upp minst en fil (fixtures eller odds) for att borja. "
                    "Ladda ner CSV-mallarna ovan for att se ratt format."
                )
        else:
            st.info(
                "Ladda upp filer for att importera aktuell omgangs data. "
                "Borja med att ladda ner mallarna ovan."
            )

    elif odds_mode == "Manuell (skriv in odds)":
        st.subheader("1. Ange 1X2-odds")
        ocol1, ocol2, ocol3 = st.columns(3)
        with ocol1:
            manual_home = st.number_input("Hemma (1)", min_value=1.01, value=2.10, step=0.05, key="m_home")
        with ocol2:
            manual_draw = st.number_input("Oavgjort (X)", min_value=1.01, value=3.40, step=0.05, key="m_draw")
        with ocol3:
            manual_away = st.number_input("Borta (2)", min_value=1.01, value=3.60, step=0.05, key="m_away")

        manual_bm = st.text_input("Bookmaker (valfritt)", value="Manuell", key="m_bm")

        # Valfri comparison probability for manuellt lage
        use_manual_comp = st.checkbox(
            "Ange egen jamforelsesannolikhet (valfritt)",
            value=False,
            key="use_manual_comp",
        )
        manual_comp_probs = None
        if use_manual_comp:
            st.caption(
                "Ange dina egna sannolikheter for varje utfall. "
                "De normaliseras automatiskt till 100%."
            )
            ccol1, ccol2, ccol3 = st.columns(3)
            with ccol1:
                comp_home = st.number_input("Hemma (1) %", min_value=0.0, max_value=100.0, value=45.0, step=1.0, key="c_home")
            with ccol2:
                comp_draw = st.number_input("Oavgjort (X) %", min_value=0.0, max_value=100.0, value=28.0, step=1.0, key="c_draw")
            with ccol3:
                comp_away = st.number_input("Borta (2) %", min_value=0.0, max_value=100.0, value=27.0, step=1.0, key="c_away")
            total_comp = comp_home + comp_draw + comp_away
            if total_comp > 0:
                manual_comp_probs = {
                    "1": comp_home / total_comp,
                    "X": comp_draw / total_comp,
                    "2": comp_away / total_comp,
                }

        # -----------------------------------------------------------
        # Streckprocent-inmatning (manuellt lage)
        # -----------------------------------------------------------
        use_streck = st.checkbox(
            "Ange streckprocent for jamforelse",
            value=False,
            key="use_streck_manual",
        )
        manual_streck = None
        if use_streck:
            st.caption(
                "Ange folkets streckprocent for varje utfall (1 / X / 2). "
                "Ange i procent (t.ex. 45 for 45%). Normaliseras automatiskt."
            )
            scol1, scol2, scol3 = st.columns(3)
            with scol1:
                streck_home = st.number_input(
                    "Streck 1 (%)", min_value=0.0, max_value=100.0,
                    value=40.0, step=1.0, key="s_home",
                )
            with scol2:
                streck_draw = st.number_input(
                    "Streck X (%)", min_value=0.0, max_value=100.0,
                    value=30.0, step=1.0, key="s_draw",
                )
            with scol3:
                streck_away = st.number_input(
                    "Streck 2 (%)", min_value=0.0, max_value=100.0,
                    value=30.0, step=1.0, key="s_away",
                )
            total_streck = streck_home + streck_draw + streck_away
            if total_streck > 0:
                manual_streck = {
                    "1": streck_home,
                    "X": streck_draw,
                    "2": streck_away,
                }

        if st.button("Berakna", key="odds_calc", type="primary", use_container_width=True):
            entry = OddsEntry(bookmaker=manual_bm or "Manuell", home=manual_home, draw=manual_draw, away=manual_away)
            report = build_match_report("Hemmalag", "Bortalag", [entry])
            if report is None:
                st.error("Ogiltiga odds — alla varden maste vara > 1.0")
            else:
                st.subheader("2. Odds & Fair Probabilities")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Overround", f"{report.overround:.1f}%")
                with col_b:
                    num_bm = len(report.bookmaker_odds)
                    st.metric("Bookmakers", str(num_bm))

                odds_table = {
                    "Utfall": ["Hemma (1)", "Oavgjort (X)", "Borta (2)"],
                    "Odds": [f"{entry.home:.2f}", f"{entry.draw:.2f}", f"{entry.away:.2f}"],
                    "Implicit sannol.": [
                        f"{report.implied_probs['1']:.1%}",
                        f"{report.implied_probs['X']:.1%}",
                        f"{report.implied_probs['2']:.1%}",
                    ],
                    "Fair sannol.": [
                        f"{report.fair_probs['1']:.1%}",
                        f"{report.fair_probs['X']:.1%}",
                        f"{report.fair_probs['2']:.1%}",
                    ],
                }
                st.dataframe(pd.DataFrame(odds_table), use_container_width=True, hide_index=True)

                # Value-analys for manuellt lage
                value_rpt = build_value_report(report, comparison_probs=manual_comp_probs)
                if value_rpt is not None and manual_comp_probs is not None:
                    st.subheader("3. Value-analys")
                    st.caption(f"Jamforelsekalla: {value_rpt.comparison_source}")
                    value_rows = []
                    for ov in value_rpt.outcomes:
                        label = {"1": "Hemma (1)", "X": "Oavgjort (X)", "2": "Borta (2)"}[ov.outcome]
                        value_rows.append({
                            "Utfall": label,
                            "Odds": f"{ov.odds:.2f}",
                            "Market fair prob": f"{ov.market_fair_prob:.1%}",
                            "Comparison prob": f"{ov.comparison_prob:.1%}",
                            "Edge": f"{ov.edge:+.1%}",
                            "Expected return": f"{ov.expected_return:+.1%}",
                            "Bedomning": ov.edge_label,
                        })
                    st.dataframe(pd.DataFrame(value_rows), use_container_width=True, hide_index=True)
                elif manual_comp_probs is None:
                    st.info(
                        "Ange en egen jamforelsesannolikhet ovan for att se value-analys. "
                        "Med en enda bookmaker finns ingen marknadskonsensus att jamfora mot."
                    )

                # Streckjamforelse for manuellt lage
                if manual_streck is not None:
                    streck_rpt = build_streck_report_from_odds_report(
                        report, manual_streck,
                    )
                    if streck_rpt is not None:
                        st.subheader("4. Streckjämförelse")
                        streck_rows = []
                        for os_item in streck_rpt.outcomes:
                            olabel = {"1": "Hemma (1)", "X": "Oavgjort (X)", "2": "Borta (2)"}[os_item.outcome]
                            # Color-code the label
                            if os_item.label == "understreckad":
                                badge = "🟢 understreckad"
                            elif os_item.label == "overstreckad":
                                badge = "🔴 overstreckad"
                            else:
                                badge = "⚪ neutral"
                            streck_rows.append({
                                "Utfall": olabel,
                                "Fair market prob": f"{os_item.fair_prob:.1%}",
                                "Streckprocent": f"{os_item.streck_pct:.1%}",
                                "Delta": f"{os_item.delta:+.1%}",
                                "Bedomning": badge,
                            })
                        st.dataframe(pd.DataFrame(streck_rows), use_container_width=True, hide_index=True)
                    else:
                        st.warning("Kunde inte berakna streckjamforelse — kontrollera streckvardena.")

                st.caption(
                    "Fair-sannolikheter ar overround-justerade (normaliserade till 100%)."
                )

    else:
        st.subheader("Odds & Value från historisk data")
        if df_features is None or df_features.empty:
            st.warning("Ingen feature-data laddad. Kor pipeline forst.")
        else:
            # Check if odds columns exist
            odds_cols_present = any(
                all(c in df_features.columns for c in (h, d, a))
                for h, d, a, _ in [
                    ("B365H", "B365D", "B365A", "Bet365"),
                    ("PSH", "PSD", "PSA", "Pinnacle"),
                ]
            )
            if not odds_cols_present:
                st.info(
                    "Ingen odds-data hittades i den laddade datan. "
                    "CSV-filer fran football-data.co.uk innehaller normalt odds-kolumner "
                    "(B365H, B365D, B365A etc). Kor pipeline med ratt data for att se odds."
                )
            else:
                n_rows = min(len(df_features), 50)
                st.info(f"Visar de senaste {n_rows} matcherna med tillgangliga odds.")

                # --------------------------------------------------
                # Streckdata-inmatning for historiskt lage (CSV)
                # --------------------------------------------------
                # -----------------------------------------------
                # Automatisk streckinlasning
                # -----------------------------------------------
                st.markdown("---")
                streck_lookup: Dict[str, Dict[str, float]] = {}
                streck_import_status = None

                # Forsok automatisk inlasning fran data/streck_data.csv
                try:
                    auto_lookup, auto_status = auto_load_streck()
                    streck_import_status = auto_status
                    if auto_status.loaded and auto_lookup:
                        streck_lookup.update(auto_lookup)
                        st.success(
                            f"Streckdata laddad automatiskt fran `{auto_status.source_label}` "
                            f"({auto_status.valid_rows} rader, "
                            f"{auto_status.matched_count} matcher)."
                        )
                        if auto_status.skipped_rows > 0:
                            st.warning(
                                f"{auto_status.skipped_rows} rad(er) hoppades over vid validering."
                            )
                    if auto_status.errors:
                        with st.expander("Detaljer om streckinlasning"):
                            for err in auto_status.errors:
                                st.caption(f"- {err}")
                except Exception as _auto_exc:
                    logger.warning("Automatisk streckinlasning misslyckades: %s", _auto_exc)

                # Manuell CSV-fallback
                use_streck_csv = st.checkbox(
                    "Ladda in streckprocent fran CSV (manuell)",
                    value=False,
                    key="use_streck_csv",
                    help=(
                        "CSV med kolumner: HomeTeam, AwayTeam, Streck1, StreckX, Streck2. "
                        "Overskriver automatiskt laddad streckdata."
                    ),
                )
                if use_streck_csv:
                    uploaded_streck = st.file_uploader(
                        "Valj CSV-fil med streckprocent",
                        type=["csv"],
                        key="streck_csv_upload",
                    )
                    if uploaded_streck is not None:
                        try:
                            streck_df = pd.read_csv(uploaded_streck)
                            required_cols = {"HomeTeam", "AwayTeam", "Streck1", "StreckX", "Streck2"}
                            if not required_cols.issubset(set(streck_df.columns)):
                                st.error(
                                    f"CSV maste innehalla kolumnerna: {', '.join(sorted(required_cols))}. "
                                    f"Hittade: {', '.join(streck_df.columns.tolist())}"
                                )
                            else:
                                # Manuell CSV overskriver auto-laddad data
                                streck_lookup.clear()
                                for _, srow in streck_df.iterrows():
                                    key = f"{srow['HomeTeam']}_{srow['AwayTeam']}"
                                    streck_lookup[key] = {
                                        "1": float(srow["Streck1"]),
                                        "X": float(srow["StreckX"]),
                                        "2": float(srow["Streck2"]),
                                    }
                                st.success(
                                    f"Laddade streckdata fran uppladdad CSV for {len(streck_lookup)} matcher. "
                                    "(Overskriver automatisk data.)"
                                )
                        except Exception as e:
                            st.error(f"Kunde inte lasa CSV: {e}")

                if not streck_lookup and streck_import_status is None:
                    st.info(
                        "Ingen streckdata hittades. "
                        "Placera en CSV-fil som `data/streck_data.csv` med kolumner "
                        "HomeTeam, AwayTeam, Streck1, StreckX, Streck2 "
                        "for automatisk inlasning, eller ladda upp manuellt ovan."
                    )
                st.markdown("---")

                recent = df_features.tail(n_rows).iloc[::-1]
                reports = build_reports_from_dataframe(recent)

                if not reports:
                    st.warning("Inga giltiga odds hittades i de valda matcherna.")
                else:
                    # Bygg value-rapporter for alla matcher
                    value_reports = []
                    for rpt in reports:
                        vr = build_value_report(rpt)
                        if vr is not None:
                            value_reports.append((rpt, vr))

                    # Rangordna matcher efter intresse (hogst edge forst)
                    if value_reports:
                        vr_list = [vr for _, vr in value_reports]
                        ranked_vr = rank_matches_by_interest(vr_list)
                        # Bygg lookup for att matcha tillbaka
                        vr_to_rpt = {id(vr): rpt for rpt, vr in value_reports}

                        # -----------------------------------------------
                        # Ranking-tabell: mest intressanta utfall
                        # -----------------------------------------------
                        st.subheader("Snabböversikt — mest intressanta utfall")
                        ranked_outcomes = rank_outcomes_by_edge(vr_list)

                        # Visa topp-10 positiva och topp-5 negativa
                        positive_outcomes = [r for r in ranked_outcomes if r[2].edge > 0.001]
                        negative_outcomes = [r for r in ranked_outcomes if r[2].edge < -0.001]

                        if positive_outcomes:
                            st.markdown("**Hogst positiv edge:**")
                            pos_rows = []
                            for match_label, outcome_label, ov in positive_outcomes[:10]:
                                pos_rows.append({
                                    "Match": match_label,
                                    "Utfall": outcome_label,
                                    "Odds": f"{ov.odds:.2f} ({ov.bookmaker})",
                                    "Edge": f"{ov.edge:+.1%}",
                                    "Exp. return": f"{ov.expected_return:+.1%}",
                                    "Bedomning": ov.edge_label,
                                })
                            st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

                        if negative_outcomes:
                            st.markdown("**Mest negativa edge:**")
                            neg_rows = []
                            # Sortera negativa fran mest negativ forst
                            for match_label, outcome_label, ov in reversed(negative_outcomes[-5:]):
                                neg_rows.append({
                                    "Match": match_label,
                                    "Utfall": outcome_label,
                                    "Odds": f"{ov.odds:.2f} ({ov.bookmaker})",
                                    "Edge": f"{ov.edge:+.1%}",
                                    "Exp. return": f"{ov.expected_return:+.1%}",
                                    "Bedomning": ov.edge_label,
                                })
                            st.dataframe(pd.DataFrame(neg_rows), use_container_width=True, hide_index=True)

                        if not positive_outcomes and not negative_outcomes:
                            st.info("Ingen tydlig edge hittades bland aktuella matcher.")

                        st.divider()

                    # -----------------------------------------------
                    # Detaljvy per match (rankad efter intresse)
                    # -----------------------------------------------
                    st.subheader("Detaljvy per match")

                    # Visa matcher i intresseordning om ranking finns
                    if value_reports:
                        display_order = [(vr_to_rpt.get(id(vr), rpt), vr) for vr in ranked_vr for rpt, _ in value_reports if id(_) == id(vr)]
                        if not display_order:
                            display_order = value_reports
                    else:
                        display_order = [(rpt, None) for rpt in reports]

                    for rpt, vr in display_order:
                        # Bygg expander-etikett med edge-info
                        if vr is not None and vr.outcomes:
                            max_edge = max(ov.edge for ov in vr.outcomes)
                            edge_hint = f"  |  Hogsta edge: {max_edge:+.1%}" if abs(max_edge) > 0.001 else ""
                        else:
                            edge_hint = ""

                        with st.expander(
                            f"{rpt.home_team} vs {rpt.away_team}  |  "
                            f"Overround: {rpt.overround:.1f}%{edge_hint}"
                        ):
                            num_bm = len(rpt.bookmaker_odds)

                            # Bookmaker odds table
                            bm_rows = []
                            for e in rpt.bookmaker_odds:
                                bm_rows.append({
                                    "Bookmaker": e.bookmaker,
                                    "1 (Hemma)": f"{e.home:.2f}",
                                    "X (Oavgjort)": f"{e.draw:.2f}",
                                    "2 (Borta)": f"{e.away:.2f}",
                                })
                            st.dataframe(pd.DataFrame(bm_rows), use_container_width=True, hide_index=True)

                            # Value-analys per utfall
                            if vr is not None:
                                st.markdown("**Value-analys:**")
                                st.caption(f"Jamforelsekalla: {vr.comparison_source}")
                                value_rows = []
                                for ov in vr.outcomes:
                                    label = {"1": "Hemma (1)", "X": "Oavgjort (X)", "2": "Borta (2)"}[ov.outcome]
                                    value_rows.append({
                                        "Utfall": label,
                                        "Odds": f"{ov.odds:.2f} ({ov.bookmaker})",
                                        "Market fair prob": f"{ov.market_fair_prob:.1%}",
                                        "Comparison prob": f"{ov.comparison_prob:.1%}",
                                        "Edge": f"{ov.edge:+.1%}",
                                        "Expected return": f"{ov.expected_return:+.1%}",
                                        "Bedomning": ov.edge_label,
                                    })
                                st.dataframe(pd.DataFrame(value_rows), use_container_width=True, hide_index=True)
                            else:
                                # Fallback: visa bara sannolikheter
                                prob_table = {
                                    "Utfall": ["Hemma (1)", "Oavgjort (X)", "Borta (2)"],
                                    "Implicit sannol.": [
                                        f"{rpt.implied_probs['1']:.1%}",
                                        f"{rpt.implied_probs['X']:.1%}",
                                        f"{rpt.implied_probs['2']:.1%}",
                                    ],
                                    "Fair sannol.": [
                                        f"{rpt.fair_probs['1']:.1%}",
                                        f"{rpt.fair_probs['X']:.1%}",
                                        f"{rpt.fair_probs['2']:.1%}",
                                    ],
                                }
                                st.dataframe(pd.DataFrame(prob_table), use_container_width=True, hide_index=True)

                            # Best odds
                            if num_bm > 1 and rpt.best_odds:
                                st.markdown("**Basta odds per utfall:**")
                                for outcome in ["1", "X", "2"]:
                                    if outcome in rpt.best_odds:
                                        oval, obm = rpt.best_odds[outcome]
                                        label = {"1": "Hemma", "X": "Oavgjort", "2": "Borta"}[outcome]
                                        st.write(f"- {label} ({outcome}): **{oval:.2f}** ({obm})")
                            elif num_bm == 1:
                                st.caption(
                                    f"Endast en bookmaker ({rpt.bookmaker_odds[0].bookmaker}) — "
                                    "value-analys kraver fler bookmakers for marknadskonsensus."
                                )

                    # --------------------------------------------------
                    # Streckjamforelse: ranking for historiskt lage
                    # --------------------------------------------------
                    if streck_lookup:
                        streck_reports_list = []
                        for rpt in reports:
                            skey = f"{rpt.home_team}_{rpt.away_team}"
                            if skey in streck_lookup:
                                sr = build_streck_report_from_odds_report(
                                    rpt, streck_lookup[skey],
                                )
                                if sr is not None:
                                    streck_reports_list.append(sr)

                        if streck_reports_list:
                            st.divider()
                            st.subheader("Streckjämförelse — över- & understreckade")

                            ranked_streck_outcomes = rank_outcomes_by_streck_delta(streck_reports_list)

                            # Mest understreckade
                            understreck = [r for r in ranked_streck_outcomes if r[2].delta < -0.001]
                            if understreck:
                                st.markdown("**Mest understreckade utfall:**")
                                us_rows = []
                                for match_label, outcome_label, os_item in understreck[:10]:
                                    us_rows.append({
                                        "Match": match_label,
                                        "Utfall": outcome_label,
                                        "Fair prob": f"{os_item.fair_prob:.1%}",
                                        "Streck": f"{os_item.streck_pct:.1%}",
                                        "Delta": f"{os_item.delta:+.1%}",
                                        "Bedomning": "🟢 understreckad",
                                    })
                                st.dataframe(pd.DataFrame(us_rows), use_container_width=True, hide_index=True)

                            # Mest overstreckade
                            overstreck = [r for r in reversed(ranked_streck_outcomes) if r[2].delta > 0.001]
                            if overstreck:
                                st.markdown("**Mest overstreckade utfall:**")
                                os_rows = []
                                for match_label, outcome_label, os_item in overstreck[:10]:
                                    os_rows.append({
                                        "Match": match_label,
                                        "Utfall": outcome_label,
                                        "Fair prob": f"{os_item.fair_prob:.1%}",
                                        "Streck": f"{os_item.streck_pct:.1%}",
                                        "Delta": f"{os_item.delta:+.1%}",
                                        "Bedomning": "🔴 overstreckad",
                                    })
                                st.dataframe(pd.DataFrame(os_rows), use_container_width=True, hide_index=True)

                            # Mest intressanta matcher
                            ranked_streck_matches = rank_matches_by_streck_interest(streck_reports_list)
                            if ranked_streck_matches:
                                st.markdown("**Mest intressanta matcher (storst streckavvikelse):**")
                                mi_rows = []
                                for sr in ranked_streck_matches[:10]:
                                    mi_rows.append({
                                        "Match": f"{sr.home_team} vs {sr.away_team}",
                                        "Storsta avvikelse": f"{sr.max_abs_delta:+.1%}",
                                    })
                                st.dataframe(pd.DataFrame(mi_rows), use_container_width=True, hide_index=True)

                            if not understreck and not overstreck:
                                st.info("Ingen tydlig streckavvikelse hittades.")

                    st.caption(
                        "Value-analysen bygger pa marknadskonsensus mellan bookmakers. "
                        "Streckjamforelsen visar skillnaden mellan folkets streck och "
                        "marknadens fair probability."
                    )

# ============================================================================
# FLIK 3: OM APPEN
# ============================================================================

with tab3:
    st.header("Om appen")

    st.markdown("""
    **Odds & Value-verktyget** hjälper dig analysera 1X2-odds, beräkna fair
    probabilities, hitta value (edge) och jämföra mot folkets streckprocent.

    Appen innehåller också en **modellbaserad prediktion** (XGBoost) som
    komplement — se flikarna *Enskild Match* och *Flera Matcher*.
    """)

    st.divider()

    st.subheader("Huvudfunktioner")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Odds & Value (huvudflöde)**
        - Fair probabilities från bookmaker-odds
        - Value-analys: edge & expected return
        - Streckjämförelse: under-/överstreckade utfall
        - Ranking av mest intressanta utfall & matcher
        """)

    with col2:
        st.markdown("""
        **Modellprediktion (komplement)**
        - Enskild match eller hel omgång
        - Halvgarderingsförslag
        - Trust score & AI-analys (valfritt)
        """)

    st.divider()

    with st.expander("Teknisk information"):
        st.markdown("""
        - **Frontend:** Streamlit
        - **Modell:** XGBoost + CalibratedClassifierCV
        - **Data:** football-data.co.uk (engelska ligor)
        - **Testning:** pytest, CI på Python 3.9/3.10/3.11
        - **Deployment:** Render / Docker
        """)

        if model_metadata:
            feats = model_metadata.get("features", [])
            n_feats = len(feats)
        else:
            n_feats = len(FEATURE_COLUMNS)
        st.caption(f"Modellen använder {n_feats} features.")

    st.divider()

    if model_metadata:
        ver = model_metadata.get("model_version", "v8.0")
        st.caption(f"Version: **{ver}**")
    else:
        st.caption("Version: **v8.0**")

    st.caption("Utvecklad av **Emtatos**.")
