"""
app_helpers.py — Gemensamma hjälpfunktioner för alla Streamlit-sidor.

Innehåller modell-laddning, feature-data, prediktion och AI-analys.
Importeras av app.py och alla sidor i pages/.
"""

import streamlit as st
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import os
from typing import Optional, List, Tuple

from main import run_pipeline, get_current_season_code
from model_handler import load_model
from xgboost import XGBClassifier
from ui_utils import get_halfguard_sign, pick_half_guards, parse_match_input, calculate_match_entropy
from utils import normalize_team_name, set_canonical_teams, get_canonical_teams

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

logger = logging.getLogger(__name__)


# ============================================================================
# CACHED LOADERS
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


# ============================================================================
# PREDIKTION
# ============================================================================

def _predict_match_core(
    model: XGBClassifier,
    home_team: str,
    away_team: str,
    df_features: pd.DataFrame,
    builder_fn,
) -> Optional[Tuple[np.ndarray, dict]]:
    """
    Kärn-logik för matchprediktion.

    builder_fn: callable som returnerar en FeatureBuilder-instans.
    Separerad så att app.py kan skicka sin egen (patchbar) referens.
    """
    if df_features is None or df_features.empty:
        return None

    builder = builder_fn(df_features)
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
    return _predict_match_core(model, home_team, away_team, df_features, _get_feature_builder)


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
# MODELL OCH DATA
# ============================================================================

def get_model_and_data():
    """Returnerar (model, df_features, model_metadata, all_teams, MODEL_FILENAME) med caching."""
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

    all_teams = get_all_teams(df_features) if df_features is not None else []

    return model, df_features, model_metadata, all_teams, MODEL_FILENAME


def ensure_model_ready(model, df_features, all_teams):
    """Kontrollerar att modell och data finns, annars kör pipeline automatiskt.
    
    Returnerar True om redo, False om stoppad.
    """
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
        return False
    return True
