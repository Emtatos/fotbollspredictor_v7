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
tab1, tab2, tab_odds, tab3 = st.tabs(["🎯 Enskild Match", "📋 Flera Matcher", "📈 Oddsverktyg", "ℹ️ Om Appen"])

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
# FLIK: ODDSVERKTYG
# ============================================================================

with tab_odds:
    st.header("Oddsverktyg — 1X2 Sannolikheter, Value & Ranking")
    st.markdown(
        "Beraknar implicita sannolikheter, overround-justerade fair-sannolikheter, "
        "value-analys (edge & expected return) och rankar utfall efter edge."
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

    # -------------------------------------------------------------------
    # Forklaring av value-analys
    # -------------------------------------------------------------------
    with st.expander("Vad betyder value-analysen?", expanded=False):
        st.markdown(
            "**Positiv edge** innebar att jamforelsesannolikheten ar hogre an "
            "marknadens fair probability for ett utfall. Det kan tyda pa att "
            "oddsen ar generosa relativt marknaden.\n\n"
            "**Negativ edge** innebar det motsatta — marknaden ger hogre "
            "sannolikhet an jamforelsesannolikheten.\n\n"
            "**Expected return** visar forvantad avkastning baserat pa "
            "jamforelsesannolikheten och oddsen. Positivt varde antyder "
            "att oddsen ar generosa.\n\n"
            "*Detta ar ett beslutsstod, inte en garanti for vinst. "
            "Value-analysen i denna version bygger pa marknadskonsensus "
            "mellan bookmakers som jamforelsekalla.*"
        )

    odds_mode = st.radio(
        "Valj inmatningsmetod:",
        ["Manuell (skriv in odds)", "Fran data (historiska matcher)"],
        key="odds_mode",
        horizontal=True,
    )

    if odds_mode == "Manuell (skriv in odds)":
        st.subheader("Ange 1X2-odds")
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

        if st.button("Berakna", key="odds_calc", type="primary", use_container_width=True):
            entry = OddsEntry(bookmaker=manual_bm or "Manuell", home=manual_home, draw=manual_draw, away=manual_away)
            report = build_match_report("Hemmalag", "Bortalag", [entry])
            if report is None:
                st.error("Ogiltiga odds — alla varden maste vara > 1.0")
            else:
                st.subheader("Resultat")

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
                    st.subheader("Value-analys")
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

                st.caption(
                    "Fair-sannolikheter ar overround-justerade (normaliserade till 100%)."
                )

    else:
        st.subheader("Analysera odds fran historisk data")
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
                        st.subheader("Ranking — mest intressanta utfall")
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
                    st.subheader("Detaljerad analys per match")

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

                    st.caption(
                        "Value-analysen i denna version bygger pa marknadskonsensus "
                        "mellan bookmakers. Framtida tillagg: modellsannolikheter, "
                        "streckjamforelse."
                    )

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
