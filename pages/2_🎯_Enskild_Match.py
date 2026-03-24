"""
Sida: Enskild Match — modellprediktion med XGBoost.

Låter användaren välja hemma- och bortalag, och visar sannolikheter,
halvgardering, trust score och valfri AI-analys.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

from app_helpers import (
    get_model_and_data,
    ensure_model_ready,
    predict_match,
    get_openai_analysis,
    HAS_OPENAI,
)
from ui_utils import get_halfguard_sign
from schema import FEATURE_COLUMNS

# Ladda modell och data via gemensam helper
model, df_features, model_metadata, all_teams, MODEL_FILENAME = get_model_and_data()

# Kontrollera att modell är redo
if not ensure_model_ready(model, df_features, all_teams):
    st.stop()

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
                    st.caption("Modellen tränades utan odds-features.")

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
