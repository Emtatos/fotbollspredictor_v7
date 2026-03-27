"""
Sida: Flera Matcher — batchprediktion med XGBoost.

Användaren klistrar in flera matcher (en per rad) och får sannolikheter,
halvgarderingsförslag och en sammanslagen tipsrad.
"""

import streamlit as st
import pandas as pd
import numpy as np

from app_helpers import (
    get_model_and_data,
    ensure_model_ready,
    predict_match,
)
from ui_utils import (
    get_halfguard_sign,
    pick_half_guards,
    pick_half_guards_combined,
    get_halfguard_sign_combined,
    parse_match_input,
    parse_match_input_with_errors,
    calculate_match_entropy,
)
from combined_probability import build_combined_match
from utils import set_canonical_teams, get_canonical_teams

# Ladda modell och data via gemensam helper
model, df_features, model_metadata, all_teams, MODEL_FILENAME = get_model_and_data()

# Kontrollera att modell är redo
if not ensure_model_ready(model, df_features, all_teams):
    st.stop()

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

        matches, parse_errors = parse_match_input_with_errors(match_input)

        # Visa eventuella tolkningsfel
        if parse_errors:
            with st.expander(f"⚠️ {len(parse_errors)} rad(er) kunde inte tolkas", expanded=True):
                for err in parse_errors:
                    st.warning(err)

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

            # Bygg kombinerade sannolikheter för alla matcher
            # (Flera Matcher har bara modell-probs, inga odds/streck)
            combined_matches = []
            for i, (home, away) in enumerate(matches):
                cm = build_combined_match(
                    home_team=home,
                    away_team=away,
                    model_probs=all_probs[i],
                )
                combined_matches.append(cm)

            # Applicera halvgarderingar
            if num_halfguards > 0:
                # Fallback: Flera Matcher har bara modell → använd enbart modell
                guard_indices = pick_half_guards(all_probs, num_halfguards)
                for idx in guard_indices:
                    if all_probs[idx] is not None:
                        results[idx]["Tips"] = get_halfguard_sign(all_probs[idx])
                        results[idx]["HALV"] = "HALV"

                # Visa vilka signaler som användes
                sources_used = []
                if any(cm.sources["odds"] for cm in combined_matches):
                    sources_used.append("odds (50%)")
                if any(cm.sources["model"] for cm in combined_matches):
                    sources_used.append("modell (35%)")
                if any(cm.sources["streck"] for cm in combined_matches):
                    sources_used.append("streck (15%)")
                if sources_used:
                    st.caption(f"Halvgarderingar baserade på: {', '.join(sources_used)}")

            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True, hide_index=True)

            # Visa tipsrad
            st.subheader("📝 Tipsrad för kopiering")
            tipsrad = "".join([r["Tips"] for r in results if r["Tips"] != "?"])
            st.code(tipsrad, language=None)
