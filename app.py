"""
Odds & Value-verktyg — Streamlit-applikation (huvudsida).

Konfiguration, sidebar och resursladdning.
Sidinnehållet finns i pages/-katalogen:
  - 1_📈_Odds_och_Value.py  (odds, value, streck)
  - 2_🎯_Enskild_Match.py   (enskild matchprediktion)
  - 3_📋_Flera_Matcher.py   (batchprediktion)
  - 4_ℹ️_Om_Appen.py        (info & hjälp)
"""

import streamlit as st
from pathlib import Path
import logging
import os
from datetime import datetime

from app_helpers import (  # noqa: F401 — re-export for backward compatibility
    get_model_and_data,
    ensure_model_ready,
    _predict_match_core,
    get_openai_analysis,
    load_cached_model,
    _get_feature_builder,
    load_feature_data,
    get_all_teams,
    HAS_INJURY_SCRAPER,
    HAS_OPENAI,
)


def predict_match(model, home_team, away_team, df_features):
    """Backward-compatible predict_match.

    Defined here (not just re-exported) so that tests can patch
    ``app._get_feature_builder`` and have the patched version picked up,
    since Python resolves the name from this module's globals at call time.
    """
    return _predict_match_core(model, home_team, away_team, df_features, _get_feature_builder)
from main import run_pipeline

# Injury scraper (behövs för sidebar-knappar)
try:
    from injury_scraper import InjuryDataFetcher, update_injury_data
except ImportError:
    pass

# ============================================================================
# KONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Odds & Value-verktyg",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# LADDA RESURSER
# ============================================================================

model, df_features, model_metadata, all_teams, MODEL_FILENAME = get_model_and_data()

# ============================================================================
# ANVÄNDARGRÄNSSNITT
# ============================================================================

st.title("📈 Odds & Value-verktyg")
st.markdown("Analysera odds, fair probabilities, value och streckjämförelse för engelska ligor")

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("⚽ Fotbollspredictor")

    # Kort, tydlig status
    if model is not None:
        st.success("Modell laddad")
    else:
        st.warning("Ingen modell — kör pipeline först")

    if df_features is not None:
        st.caption(f"{len(df_features)} matcher · {len(all_teams)} lag")

    # Detaljerad teknisk status under expander
    with st.expander("Teknisk status", expanded=False):
        if df_features is not None:
            st.write(f"Matcher i data: {len(df_features)}")
            st.write(f"Antal lag: {len(all_teams)}")
        else:
            st.write("Ingen feature-data laddad")

        if model_metadata:
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
            st.write("AI-analys: tillgänglig")
        else:
            st.write("AI-analys: ej tillgänglig")

        # Skade-data status
        if HAS_INJURY_SCRAPER:
            injury_file = Path("data/injuries_latest.json")
            if injury_file.exists():
                fetcher = InjuryDataFetcher()
                if fetcher.is_data_stale():
                    st.write("Skadedata: gammal (>24h)")
                else:
                    import json
                    with open(injury_file, "r") as f:
                        inj_data = json.load(f)
                        last_update = inj_data.get("last_updated", "Okänd")
                        if last_update != "Okänd":
                            last_update = datetime.fromisoformat(last_update).strftime("%Y-%m-%d %H:%M")
                    st.write(f"Skadedata: {last_update}")
            else:
                st.write("Skadedata: saknas")

        st.divider()

        # Pipeline-knappar under teknisk status
        if HAS_INJURY_SCRAPER:
            if st.button("Uppdatera skador & form", help="Hämtar senaste skador och matchresultat", use_container_width=True):
                with st.spinner("Hämtar färsk data..."):
                    try:
                        success = update_injury_data()
                        if success:
                            st.success("Skadedata uppdaterad!")
                            st.rerun()
                        else:
                            st.error("Kunde inte uppdatera skadedata. Kontrollera API-nyckel.")
                    except Exception as e:
                        st.error(f"Fel vid uppdatering: {e}")
                        logger.error(f"Injury update failed: {e}", exc_info=True)

        if st.button("Kör omträning av modell", help="Tränar om modellen med alla 27 features (inkl. skador)", use_container_width=True):
            with st.spinner("Tränar modell med 27 features..."):
                try:
                    from retrain_model_27features import retrain_with_injury_features
                    new_model_path = retrain_with_injury_features()
                    st.success(f"Modell omtränad med 27 features!")
                    st.info(f"Sparad som: {new_model_path.name}")
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Ett fel inträffade: {e}")
                    logger.error(f"Retrain failed: {e}", exc_info=True)
                    try:
                        st.info("Försöker med full pipeline...")
                        run_pipeline()
                        st.success("Pipelinen är färdig!")
                        st.cache_resource.clear()
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e2:
                        st.error(f"Pipeline misslyckades också: {e2}")
                        logger.error(f"Pipeline fallback failed: {e2}", exc_info=True)

# ============================================================================
# AUTO-TRÄNING OM MODELL/DATA SAKNAS
# ============================================================================

if not ensure_model_ready(model, df_features, all_teams):
    st.stop()

st.info("Välj en sida i vänstermenyn för att börja analysera.")
