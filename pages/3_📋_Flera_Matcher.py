"""
Sida: Flera Matcher — batchprediktion med XGBoost.

Användaren klistrar in flera matcher (en per rad) och får sannolikheter,
halvgarderingsförslag och en sammanslagen tipsrad.

Om modellen saknar data för ett lag (t.ex. lag utanför E0–E2) men odds/streck
finns tillgängliga från en tidigare import (current_round), används dessa som
fallback så att matchen inte visas som N/A utan förklaring.
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
from combined_probability import build_combined_match, odds_to_fair_probs
from utils import normalize_team_name, set_canonical_teams, get_canonical_teams
from matchday_import import _make_key

# Ladda modell och data via gemensam helper
model, df_features, model_metadata, all_teams, MODEL_FILENAME = get_model_and_data()

# Kontrollera att modell är redo
if not ensure_model_ready(model, df_features, all_teams):
    st.stop()

st.header("Flera matcher — modellprediktion")
st.caption(
    "Använder den tränade modellen. "
    "Om ett lag saknas i modellens data men odds/streck finns från "
    "senaste import visas en odds-baserad fallback istället för N/A."
)
st.markdown("Skriv in matcher, en per rad. Format: `Hemmalag - Bortalag`")

# --- Importera från senaste scanning (current_round) ---
current_round = st.session_state.get("current_round")
if current_round and current_round.get("matches"):
    source_label = current_round.get("source", "okänd källa")
    timestamp = current_round.get("timestamp", "")
    ts_short = timestamp[:16].replace("T", " ") if timestamp else ""
    num_matches = len(current_round["matches"])

    with st.container():
        imp_col1, imp_col2 = st.columns([3, 1])
        with imp_col1:
            st.info(
                f"ℹ️ {num_matches} matcher tillgängliga från senaste import "
                f"({source_label}{', ' + ts_short if ts_short else ''}). "
                f"Klicka för att fylla i automatiskt."
            )
        with imp_col2:
            if st.button(
                "Importera matcher",
                key="btn_import_from_round",
                use_container_width=True,
            ):
                lines = [
                    f"{home} - {away}"
                    for home, away in current_round["matches"]
                ]
                st.session_state["multi_matches"] = "\n".join(lines)
                st.rerun()

match_input = st.text_area(
    "Matcher:",
    height=200,
    placeholder="Arsenal - Chelsea\nLiverpool - Manchester United\nTottenham - Newcastle",
    key="multi_matches"
)

col1, col2 = st.columns(2)


# ---- Hjälpfunktion: hämta odds/streck från current_round ----

def _lookup_round_odds(home: str, away: str):
    """Returnerar (odds_entries, streck_dict) från current_round om tillgängligt."""
    cr = st.session_state.get("current_round")
    if not cr:
        return None, None
    odds_by_key = cr.get("odds") or {}
    streck_by_key = cr.get("streck") or {}
    key = _make_key(home, away)
    odds_entries = odds_by_key.get(key)
    streck_dict = streck_by_key.get(key)
    # Fallback: case-insensitive
    if odds_entries is None:
        for k, v in odds_by_key.items():
            if k.lower() == key.lower():
                odds_entries = v
                break
    if streck_dict is None:
        for k, v in streck_by_key.items():
            if k.lower() == key.lower():
                streck_dict = v
                break
    return odds_entries, streck_dict

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
            data_sources = []  # "modell", "odds (fallback)", "N/A"

            for home, away in matches:
                result = predict_match(model, home, away, df_features)

                if result is not None:
                    probs, stats = result
                    all_probs.append(probs)
                    data_sources.append("modell")

                    sign = ['1', 'X', '2'][np.argmax(probs)]
                    trust_lbl = stats.get('trust_label', 'N/A')
                    if trust_lbl == "LOW":
                        trust_lbl = "LOW (varning)"

                    results.append({
                        "Match": f"{home} - {away}",
                        "1": f"{probs[0]:.1%}",
                        "X": f"{probs[1]:.1%}",
                        "2": f"{probs[2]:.1%}",
                        "Källa": "modell",
                        "Trust": trust_lbl,
                        "Tips": sign,
                        "HALV": ""
                    })
                else:
                    # --- Fallback: odds/streck från current_round ---
                    odds_entries, streck_dict = _lookup_round_odds(home, away)

                    fallback_probs = None
                    if odds_entries:
                        e = odds_entries[0]
                        fallback_probs = odds_to_fair_probs(e.home, e.draw, e.away)

                    if fallback_probs is not None:
                        all_probs.append(fallback_probs)
                        data_sources.append("odds (fallback)")

                        sign = ['1', 'X', '2'][np.argmax(fallback_probs)]

                        results.append({
                            "Match": f"{home} - {away}",
                            "1": f"{fallback_probs[0]:.1%}",
                            "X": f"{fallback_probs[1]:.1%}",
                            "2": f"{fallback_probs[2]:.1%}",
                            "Källa": "odds (fallback)",
                            "Trust": "—",
                            "Tips": sign,
                            "HALV": ""
                        })
                    else:
                        all_probs.append(None)
                        data_sources.append("N/A")
                        results.append({
                            "Match": f"{home} - {away}",
                            "1": "N/A",
                            "X": "N/A",
                            "2": "N/A",
                            "Källa": "saknas",
                            "Trust": "N/A",
                            "Tips": "?",
                            "HALV": ""
                        })

            # Bygg kombinerade sannolikheter med odds/streck från current_round
            combined_matches = []
            for i, (home, away) in enumerate(matches):
                odds_entries, streck_dict = _lookup_round_odds(home, away)

                odds_1_val = odds_x_val = odds_2_val = None
                streck_1_val = streck_x_val = streck_2_val = None

                if odds_entries:
                    e = odds_entries[0]
                    odds_1_val = e.home
                    odds_x_val = e.draw
                    odds_2_val = e.away

                if streck_dict:
                    streck_1_val = streck_dict.get("1")
                    streck_x_val = streck_dict.get("X")
                    streck_2_val = streck_dict.get("2")

                cm = build_combined_match(
                    home_team=home,
                    away_team=away,
                    odds_1=odds_1_val,
                    odds_x=odds_x_val,
                    odds_2=odds_2_val,
                    model_probs=all_probs[i],
                    streck_1=streck_1_val,
                    streck_x=streck_x_val,
                    streck_2=streck_2_val,
                )
                combined_matches.append(cm)

            # Applicera halvgarderingar
            if num_halfguards > 0:
                # Använd kombinerade sannolikheter för halvgardering
                guard_indices = pick_half_guards_combined(combined_matches, num_halfguards)
                for idx in guard_indices:
                    cm = combined_matches[idx]
                    results[idx]["Tips"] = get_halfguard_sign_combined(cm)
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
                st.caption(
                    "Urval av halvgarderingar styrs av **gain** "
                    "(näst högsta sannolikheten = marginalnytta av en halvgardering)."
                )

            # Visa fallback-information om den användes
            n_fallback = sum(1 for ds in data_sources if ds == "odds (fallback)")
            n_missing = sum(1 for ds in data_sources if ds == "N/A")
            if n_fallback > 0 or n_missing > 0:
                parts = []
                if n_fallback > 0:
                    parts.append(
                        f"{n_fallback} match(er) saknar modelldata — "
                        f"odds-baserad fallback används"
                    )
                if n_missing > 0:
                    parts.append(
                        f"{n_missing} match(er) saknar all data (varken modell eller odds)"
                    )
                st.warning(". ".join(parts) + ".")

            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True, hide_index=True)

            # Visa tipsrad
            st.subheader("📝 Tipsrad för kopiering")
            tipsrad = "".join([r["Tips"] for r in results if r["Tips"] != "?"])
            st.code(tipsrad, language=None)
