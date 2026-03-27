"""
Sida: Odds & Value — huvudflödet för oddsanalys, value och streckjämförelse.

Stödjer fyra inmatningslägen:
  1. Aktuell omgång (importera via text eller CSV)
  2. Kupongbild (screenshot → AI-tolkning → kontrolltabell)
  3. Manuell (skriv in odds)
  4. Från data (historiska matcher)
"""

import streamlit as st
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List

from app_helpers import get_model_and_data, ensure_model_ready

logger = logging.getLogger(__name__)

# --- Session state: current_round (5B) ---
if "current_round" not in st.session_state:
    st.session_state["current_round"] = None


def _save_current_round(matches, odds, streck, source: str):
    """Spara aktuell omgång i session state oavsett input-metod (5C)."""
    st.session_state["current_round"] = {
        "matches": matches,
        "odds": odds,
        "streck": streck,
        "source": source,
        "timestamp": datetime.now().isoformat(),
    }

# Ladda modell och data via gemensam helper
model, df_features, model_metadata, all_teams, MODEL_FILENAME = get_model_and_data()

# Kontrollera att modell är redo
if not ensure_model_ready(model, df_features, all_teams):
    st.stop()

st.header("Odds, Value & Streckjämförelse")
st.markdown(
    "Mata in odds → se fair probabilities → jämför value och streck → hitta intressanta utfall."
)


def _show_analysis_summary(value_reports_list, streck_reports_list, n_matches):
    """Visa en sammanfattning med metrics innan detaljerna."""
    from value_analysis import rank_outcomes_by_edge
    from streck_analysis import rank_outcomes_by_streck_delta

    top_value_match = "—"
    top_overstreck = "—"

    if value_reports_list:
        ranked = rank_outcomes_by_edge(value_reports_list)
        positive = [r for r in ranked if r[2].edge > 0.001]
        if positive:
            match_label, outcome_label, ov = positive[0]
            top_value_match = f"{match_label}: {outcome_label} ({ov.edge:+.1%})"

    if streck_reports_list:
        ranked_streck = rank_outcomes_by_streck_delta(streck_reports_list)
        overstreck = [r for r in reversed(ranked_streck) if r[2].delta > 0.001]
        if overstreck:
            match_label, outcome_label, os_item = overstreck[0]
            top_overstreck = f"{match_label}: {outcome_label} ({os_item.delta:+.1%})"

    st.subheader("Sammanfattning")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Matcher analyserade", n_matches)
    with col2:
        st.metric("Mest intressanta value", top_value_match)
    with col3:
        st.metric("Mest överstreckat utfall", top_overstreck)
    st.divider()


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
from combined_probability import build_combined_match
from ui_utils import pick_half_guards_combined, get_halfguard_sign_combined

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
    ["Aktuell omgång (importera)", "Kupongbild (screenshot)", "Manuell (skriv in odds)", "Från data (historiska matcher)"],
    key="odds_mode",
    horizontal=True,
)

if odds_mode == "Aktuell omgång (importera)":
    from matchday_import import (
        parse_fixtures_csv,
        parse_fixture_lines,
        fetch_odds_for_fixtures,
        ParseFixtureLinesResult,
        parse_odds_csv,
        parse_streck_csv,
        match_matchday_data,
        MatchdayImportStatus,
        MatchdayMatch,
        MatchdayFixture,
        generate_fixtures_template,
        generate_odds_template,
        generate_streck_template,
        FIXTURES_TEMPLATE_CSV,
        ODDS_TEMPLATE_CSV,
        ODDS_MULTI_BM_TEMPLATE_CSV,
        ODDS_SIMPLE_TEMPLATE_CSV,
        STRECK_TEMPLATE_CSV,
    )
    from matchday_storage import (
        save_matchday_data,
        load_saved_matchday_data,
        clear_saved_matchday_data,
        get_saved_matchday_status,
    )

    st.subheader("Aktuell omgang — klistra in matcher & hamta odds")
    st.markdown(
        "Klistra in matchlistan nedan. Appen hamtar odds automatiskt "
        "fran tillganglig data och visar analys."
    )

    # ----- Sparad omgangsdata: status och atgarder -----
    saved_status = get_saved_matchday_status()

    # Initiera session state for datakalla-sparning
    if "matchday_data_source" not in st.session_state:
        st.session_state["matchday_data_source"] = None

    if saved_status.exists:
        st.info(
            f"Sparad omgang finns ({saved_status.match_count} matcher, "
            f"sparad {saved_status.saved_at or 'okant datum'}). "
            f"Odds: {'ja' if saved_status.has_odds else 'nej'}, "
            f"Streck: {'ja' if saved_status.has_streck else 'nej'}."
        )
        saved_col1, saved_col2, saved_col3 = st.columns(3)
        with saved_col1:
            use_saved = st.button(
                "Anvand sparad omgang",
                use_container_width=True,
                key="btn_use_saved",
            )
        with saved_col2:
            replace_saved = st.button(
                "Ersatt med ny import",
                use_container_width=True,
                key="btn_replace_saved",
            )
        with saved_col3:
            clear_saved = st.button(
                "Rensa sparad omgang",
                use_container_width=True,
                key="btn_clear_saved",
            )

        if clear_saved:
            if clear_saved_matchday_data():
                st.session_state["matchday_data_source"] = None
                st.success("Sparad omgangsdata har rensats.")
                st.rerun()
            else:
                st.error("Kunde inte rensa sparad data.")

        if replace_saved:
            st.session_state["matchday_data_source"] = None
            st.info("Ladda upp nya filer nedan for att ersatta sparad data.")

        if use_saved:
            st.session_state["matchday_data_source"] = "saved"
    else:
        st.caption("Ingen sparad omgang.")

    # ----- Ladda sparad data om anvandaren valde det -----
    _use_saved_data = st.session_state.get("matchday_data_source") == "saved"
    _saved_loaded = False
    _saved_fixtures: list = []
    _saved_odds: Dict = {}
    _saved_streck: Dict = {}

    if _use_saved_data and saved_status.exists:
        loaded = load_saved_matchday_data()
        if loaded is not None:
            _saved_fixtures, _saved_odds, _saved_streck, _saved_meta = loaded
            _saved_loaded = True
        else:
            st.warning("Sparad data kunde inte lasas. Ladda upp pa nytt.")
            st.session_state["matchday_data_source"] = None

    st.markdown("---")

    # =====================================================================
    # PRIMÄR VÄG: Klistra in matchlista
    # =====================================================================
    st.markdown("### Klistra in matcher")
    st.caption(
        "Skriv en match per rad. Stodda separatorer: `-`, `–`, `vs`  \n"
        "Exempel: `Leeds United - Brentford`"
    )
    match_text = st.text_area(
        "Matcher (en per rad)",
        height=180,
        placeholder="Leeds United - Brentford\nEverton – Chelsea\nFulham vs Burnley",
        key="matchday_text_input",
    )
    analyze_btn = st.button(
        "Analysera omgang",
        type="primary",
        use_container_width=True,
        key="btn_analyze_matchday",
    )

    # ----- Valfritt: streck-CSV -----
    st.markdown("### Streckdata (valfritt)")
    streck_file = st.file_uploader(
        "CSV med streck (HomeTeam, AwayTeam, Streck1, StreckX, Streck2)",
        type=["csv"],
        key="matchday_streck_upload",
    )

    # ----- Sekundar fallback: CSV-import -----
    with st.expander("Alternativ: importera via CSV-filer", expanded=False):
        st.markdown(
            "Om du vill ladda upp fixtures och/eller odds som CSV istallet. "
            "Anvand mallarna nedan."
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
        fixtures_file = st.file_uploader(
            "CSV med matcher (HomeTeam, AwayTeam)",
            type=["csv"],
            key="matchday_fixtures_upload",
        )
        odds_file = st.file_uploader(
            "CSV med odds (HomeTeam, AwayTeam + oddskolumner)",
            type=["csv"],
            key="matchday_odds_upload",
        )

    st.markdown("---")

    # ----- Bearbeta import -----
    matchday_matches = None
    import_status = None
    _matchday_data_ready = False

    if _saved_loaded:
        # -- Anvander sparad omgangsdata --
        st.success(
            f"Anvander sparad omgangsdata "
            f"(sparad {_saved_meta.get('saved_at', 'okant datum')}, "
            f"{_saved_meta.get('match_count', 0)} matcher)."
        )
        fixtures_list = _saved_fixtures
        odds_by_key = _saved_odds
        streck_by_key = _saved_streck

        if fixtures_list:
            matchday_matches, import_status = match_matchday_data(
                fixtures_list, odds_by_key, streck_by_key,
            )
            st.session_state["matchday_data_source"] = "saved"

            # Spara i current_round (enhetligt flöde, 5B/5C)
            _save_current_round(
                matches=[(f.home_team, f.away_team) for f in fixtures_list],
                odds=odds_by_key,
                streck=streck_by_key,
                source="saved",
            )

            _matchday_data_ready = True

    elif analyze_btn and match_text and match_text.strip():
        # ============================================================
        # PRIMÄR VÄG: text-paste + automatisk oddshämtning
        # ============================================================
        parse_result = parse_fixture_lines(match_text)

        # -- Visa parsingstatus --
        st.markdown("#### Parsingstatus")
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            st.metric("Giltiga matcher", parse_result.valid_count)
        with pcol2:
            st.metric("Ogiltiga rader", parse_result.invalid_count)
        with pcol3:
            st.metric("Tomma rader", parse_result.blank_lines)

        if parse_result.invalid_lines:
            with st.expander(
                f"Kunde inte tolka {len(parse_result.invalid_lines)} rad(er)",
                expanded=True,
            ):
                for bad_line in parse_result.invalid_lines:
                    st.caption(f"- `{bad_line}`")

        if not parse_result.fixtures:
            st.warning("Inga giltiga matcher hittades. Kontrollera formatet.")
        else:
            fixtures_list = parse_result.fixtures

            # -- Hamta odds automatiskt --
            odds_by_key, odds_matched, odds_unmatched, unmatched_labels = (
                fetch_odds_for_fixtures(fixtures_list)
            )

            # -- Visa oddshämtningsstatus --
            st.markdown("#### Oddsstatus")
            ocol1, ocol2 = st.columns(2)
            with ocol1:
                st.metric("Matcher med odds", odds_matched)
            with ocol2:
                st.metric("Matcher utan odds", odds_unmatched)

            if unmatched_labels:
                with st.expander(
                    f"Matcher utan odds ({len(unmatched_labels)})",
                    expanded=False,
                ):
                    for label in unmatched_labels:
                        st.caption(f"- {label}")

            # -- Streck (valfri CSV) --
            streck_by_key: Dict = {}
            if streck_file is not None:
                try:
                    streck_df = pd.read_csv(streck_file)
                    streck_by_key, _srl, streck_errors = parse_streck_csv(streck_df)
                    if streck_errors:
                        with st.expander(f"Streckfel ({len(streck_errors)})"):
                            for err in streck_errors:
                                st.caption(f"- {err}")
                except Exception as e:
                    st.warning(f"Kunde inte lasa streck-CSV: {e}")

            # -- Match + analysera --
            matchday_matches, import_status = match_matchday_data(
                fixtures_list, odds_by_key, streck_by_key,
            )

            # Spara automatiskt
            if save_matchday_data(fixtures_list, odds_by_key, streck_by_key):
                st.session_state["matchday_data_source"] = "textpaste"
                st.caption("Omgangsdata sparad automatiskt for ateranvandning.")

            # Spara i current_round (enhetligt flöde, 5B/5C)
            _save_current_round(
                matches=[(f.home_team, f.away_team) for f in fixtures_list],
                odds=odds_by_key,
                streck=streck_by_key,
                source="textpaste",
            )

            _matchday_data_ready = True

    elif fixtures_file is not None or odds_file is not None:
        # ============================================================
        # SEKUNDÄR VÄG: CSV-import (fallback)
        # ============================================================
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

            # -- Spara importerad data automatiskt --
            if save_matchday_data(fixtures_list, odds_by_key, streck_by_key):
                st.session_state["matchday_data_source"] = "nyimporterad"
                st.caption("Omgangsdata sparad automatiskt for ateranvandning.")

            # Spara i current_round (enhetligt flöde, 5B/5C)
            _save_current_round(
                matches=[(f.home_team, f.away_team) for f in fixtures_list],
                odds=odds_by_key,
                streck=streck_by_key,
                source="csv_import",
            )

            _matchday_data_ready = True

        # Visa felmeddelanden fran import
        if all_import_errors:
            with st.expander(f"Felmeddelanden ({len(all_import_errors)})", expanded=False):
                for err in all_import_errors:
                    st.caption(f"- {err}")

    else:
        st.info(
            "Klistra in matcher i textfaltet ovan och klicka **Analysera omgang** "
            "for att starta."
        )

    # ---- Gemensam rendering: importstatus + analys ----
    if _matchday_data_ready and matchday_matches is not None and import_status is not None:
        # ---- Importstatus-dashboard ----
        st.subheader("Importstatus")
        # Visa datakalla
        _data_src = st.session_state.get("matchday_data_source", "")
        if _data_src == "textpaste":
            st.caption("Datakalla: inklistrad matchlista med automatisk oddshämtning.")
        elif _data_src == "nyimporterad":
            st.caption("Datakalla: nyimporterad (fran uppladdade filer).")
        elif _data_src == "saved":
            st.caption("Datakalla: aterlast fran sparad omgang.")
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

        st.markdown("---")

        # ---- Analys for matchade matcher ----
        matches_with_odds = [m for m in matchday_matches if m.has_odds and m.odds_report]

        if matches_with_odds:
            # Samla value- och streckrapporter
            value_reports_list = [
                m.value_report for m in matches_with_odds
                if m.value_report is not None
            ]
            streck_reports_list = [
                m.streck_report for m in matches_with_odds
                if m.streck_report is not None
            ]

            # --- Sammanfattning ---
            _show_analysis_summary(value_reports_list, streck_reports_list, len(matches_with_odds))

            # --- Ranking: snabboversikt ---
            if value_reports_list:
                from value_analysis import rank_outcomes_by_edge, rank_matches_by_interest
                with st.expander("Fullständig analys", expanded=True):
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

            # --- Kombinerade halvgarderingar (odds + streck + modell) ---
            combined_matches_for_hg = []
            for match in matchday_matches:
                odds_1_val = None
                odds_x_val = None
                odds_2_val = None
                if match.odds_report and match.odds_report.fair_probs:
                    # Hämta genomsnittliga odds från bookmaker-odds
                    if match.odds_report.bookmaker_odds:
                        bm = match.odds_report.bookmaker_odds[0]
                        odds_1_val = bm.home
                        odds_x_val = bm.draw
                        odds_2_val = bm.away

                streck_1_val = None
                streck_x_val = None
                streck_2_val = None
                if match.has_streck and match.streck:
                    streck_1_val = match.streck.get("1")
                    streck_x_val = match.streck.get("X")
                    streck_2_val = match.streck.get("2")

                cm = build_combined_match(
                    home_team=match.home_team,
                    away_team=match.away_team,
                    odds_1=odds_1_val,
                    odds_x=odds_x_val,
                    odds_2=odds_2_val,
                    streck_1=streck_1_val,
                    streck_x=streck_x_val,
                    streck_2=streck_2_val,
                )
                combined_matches_for_hg.append(cm)

            # Visa halvgarderingsförslag
            num_hg = st.number_input(
                "Antal halvgarderingar:",
                min_value=0,
                max_value=len(matchday_matches),
                value=min(3, len(matchday_matches)),
                key="matchday_num_halfguards",
            )

            if num_hg > 0 and combined_matches_for_hg:
                guard_indices = pick_half_guards_combined(combined_matches_for_hg, num_hg)

                st.subheader("Halvgarderingar (kombinerad analys)")
                hg_rows = []
                for idx in guard_indices:
                    cm = combined_matches_for_hg[idx]
                    sign = get_halfguard_sign_combined(cm)
                    hg_rows.append({
                        "Match": f"{cm.home_team} vs {cm.away_team}",
                        "Tips": sign,
                        "Entropy": f"{cm.entropy:.3f}",
                        "1": f"{cm.prob_1:.1%}",
                        "X": f"{cm.prob_x:.1%}",
                        "2": f"{cm.prob_2:.1%}",
                    })
                st.dataframe(pd.DataFrame(hg_rows), use_container_width=True, hide_index=True)

                # Visa vilka signaler som användes
                sources_used = []
                if any(cm.sources["odds"] for cm in combined_matches_for_hg):
                    sources_used.append("odds (50%)")
                if any(cm.sources["model"] for cm in combined_matches_for_hg):
                    sources_used.append("modell (35%)")
                if any(cm.sources["streck"] for cm in combined_matches_for_hg):
                    sources_used.append("streck (15%)")
                if sources_used:
                    st.caption(f"Halvgarderingar baserade på: {', '.join(sources_used)}")

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

elif odds_mode == "Kupongbild (screenshot)":
    # ==================================================================
    # KUPONGBILD-FLODE: bild -> tolkning -> kontrolltabell -> analys
    # ==================================================================
    from coupon_image_parser import (
        parse_coupon_image,
        coupon_rows_to_dataframe,
        dataframe_to_coupon_rows,
        confirmed_rows_to_matchday_data,
        is_supported_image,
        CouponRow,
    )
    from matchday_import import (
        match_matchday_data,
        fetch_odds_for_fixtures,
        MatchdayFixture,
    )
    from matchday_storage import save_matchday_data

    st.subheader("Kupongbild — ladda upp skarmbilden")
    st.markdown(
        "Ladda upp en skärmbild av kupongen. Appen extraherar matcher, "
        "streckprocent och odds. Du granskar och rättar i en kontrolltabell "
        "innan analysen körs."
    )

    # -- Bilduppladdning --
    coupon_file = st.file_uploader(
        "Välj kupongbild",
        type=["png", "jpg", "jpeg", "webp"],
        key="coupon_image_upload",
        help="Stödda format: PNG, JPG, JPEG, WEBP",
    )

    if coupon_file is not None:
        # Visa bilden
        st.image(coupon_file, caption="Uppladdad kupongbild", use_container_width=True)

        # -- Extrahera data fran bilden --
        if "coupon_extraction_df" not in st.session_state:
            st.session_state["coupon_extraction_df"] = None
            st.session_state["coupon_extraction_status"] = None

        extract_btn = st.button(
            "Extrahera matcher, streck & odds",
            type="primary",
            use_container_width=True,
            key="btn_extract_coupon",
        )

        if extract_btn:
            with st.spinner("Tolkar kupongbilden med AI..."):
                image_bytes = coupon_file.getvalue()
                result = parse_coupon_image(image_bytes, coupon_file.name)

            if result.error:
                st.error(f"Tolkningsfel: {result.error}")
                if result.raw_response:
                    with st.expander("Ratt API-svar (debug)"):
                        st.code(result.raw_response)
            elif not result.rows:
                st.warning("Inga matcher kunde tolkas fran bilden.")
            else:
                df = coupon_rows_to_dataframe(result.rows)
                st.session_state["coupon_extraction_df"] = df
                st.session_state["coupon_extraction_status"] = {
                    "total": result.total_rows,
                    "complete": result.complete_rows,
                    "uncertain": result.uncertain_rows,
                    "incomplete": result.incomplete_rows,
                }

        # -- Visa kontrolltabell om extraktion ar klar --
        if st.session_state.get("coupon_extraction_df") is not None:
            ext_status = st.session_state.get("coupon_extraction_status", {})

            # -- Extraktionsstatus --
            st.markdown("---")
            st.subheader("Extraktionsstatus")
            ecol1, ecol2, ecol3, ecol4 = st.columns(4)
            with ecol1:
                st.metric("Rader tolkade", ext_status.get("total", 0))
            with ecol2:
                st.metric("Fullstandiga", ext_status.get("complete", 0))
            with ecol3:
                st.metric("Osakra", ext_status.get("uncertain", 0))
            with ecol4:
                st.metric("Ofullstandiga", ext_status.get("incomplete", 0))

            # -- Redigerbar kontrolltabell --
            st.markdown("---")
            st.subheader("Kontrolltabell — granska och ratta")
            st.caption(
                "Ratta feltolkningar, ta bort trasiga rader, komplettera tomma varden. "
                "Analysen kors forst nar du bekraftar tabellen nedan."
            )

            edited_df = st.data_editor(
                st.session_state["coupon_extraction_df"],
                num_rows="dynamic",
                use_container_width=True,
                key="coupon_data_editor",
                column_config={
                    "HomeTeam": st.column_config.TextColumn("Hemmalag", required=True),
                    "AwayTeam": st.column_config.TextColumn("Bortalag", required=True),
                    "Streck1": st.column_config.NumberColumn("Streck 1 (%)", min_value=0, max_value=100, format="%.1f"),
                    "StreckX": st.column_config.NumberColumn("Streck X (%)", min_value=0, max_value=100, format="%.1f"),
                    "Streck2": st.column_config.NumberColumn("Streck 2 (%)", min_value=0, max_value=100, format="%.1f"),
                    "Odds1": st.column_config.NumberColumn("Odds 1", min_value=1.01, format="%.2f"),
                    "OddsX": st.column_config.NumberColumn("Odds X", min_value=1.01, format="%.2f"),
                    "Odds2": st.column_config.NumberColumn("Odds 2", min_value=1.01, format="%.2f"),
                    "Status": st.column_config.SelectboxColumn("Status", options=["ok", "uncertain", "incomplete"]),
                    "Notes": st.column_config.TextColumn("Anteckningar"),
                },
            )

            # -- Bekrafta och kor analys --
            confirm_btn = st.button(
                "Bekrafta tabell och kor analys",
                type="primary",
                use_container_width=True,
                key="btn_confirm_coupon",
            )

            if confirm_btn:
                # Konvertera redigerad tabell tillbaka till CouponRows
                confirmed_rows = dataframe_to_coupon_rows(edited_df)

                if not confirmed_rows:
                    st.error("Inga giltiga rader i tabellen.")
                else:
                    # Bygg matchday-data fran bekraftade rader
                    fixtures, odds_by_key, streck_by_key, keys_with_odds, keys_missing_odds = (
                        confirmed_rows_to_matchday_data(confirmed_rows)
                    )

                    # -- Fallback-odds for rader utan bildodds --
                    fallback_keys: List[str] = []
                    if keys_missing_odds:
                        st.markdown("---")
                        st.info(
                            f"{len(keys_missing_odds)} match(er) saknar odds fran bilden. "
                            f"Forsoker hamta fallback-odds fran historisk data..."
                        )
                        fallback_fixtures = [
                            f for f in fixtures if f.match_key in keys_missing_odds
                        ]
                        if fallback_fixtures:
                            fb_odds, fb_matched, fb_unmatched, fb_labels = (
                                fetch_odds_for_fixtures(fallback_fixtures)
                            )
                            for fkey, fentries in fb_odds.items():
                                odds_by_key[fkey] = fentries
                                fallback_keys.append(fkey)
                            if fb_matched > 0:
                                st.success(f"Fallback-odds hittades for {fb_matched} match(er).")
                            if fb_unmatched > 0:
                                st.warning(
                                    f"{fb_unmatched} match(er) saknar fortfarande odds."
                                )
                                if fb_labels:
                                    with st.expander("Matcher utan odds"):
                                        for lbl in fb_labels:
                                            st.caption(f"- {lbl}")

                    # -- Oddskallstatus --
                    st.markdown("---")
                    st.subheader("Oddskallor")
                    total_with_odds = len(keys_with_odds) + len(fallback_keys)
                    total_without_odds = len(keys_missing_odds) - len(fallback_keys)
                    ocol1, ocol2, ocol3 = st.columns(3)
                    with ocol1:
                        st.metric("Odds fran bild", len(keys_with_odds))
                    with ocol2:
                        st.metric("Odds fran fallback", len(fallback_keys))
                    with ocol3:
                        st.metric("Saknar odds", max(0, total_without_odds))

                    # -- Koppla till befintlig analys --
                    matchday_matches, import_status = match_matchday_data(
                        fixtures, odds_by_key, streck_by_key,
                    )

                    # Spara data
                    save_matchday_data(fixtures, odds_by_key, streck_by_key)
                    st.session_state["matchday_data_source"] = "kupongbild"

                    # Spara i current_round (enhetligt flöde, 5B/5C)
                    _save_current_round(
                        matches=[(f.home_team, f.away_team) for f in fixtures],
                        odds=odds_by_key,
                        streck=streck_by_key,
                        source="kupongbild",
                    )

                    # -- Importstatus --
                    st.markdown("---")
                    st.subheader("Importstatus")
                    st.caption("Datakalla: kupongbild (screenshot).")
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

                    st.markdown("---")

                    # -- Analys (ateranvand existerande rendering) --
                    matches_with_odds = [
                        m for m in matchday_matches if m.has_odds and m.odds_report
                    ]

                    if matches_with_odds:
                        # Samla rapporter for sammanfattning
                        value_reports_list = [
                            m.value_report for m in matches_with_odds
                            if m.value_report is not None
                        ]
                        streck_reports_list_coupon = [
                            m.streck_report for m in matches_with_odds
                            if m.streck_report is not None
                        ]

                        # --- Sammanfattning ---
                        _show_analysis_summary(value_reports_list, streck_reports_list_coupon, len(matches_with_odds))

                        # --- Ranking: snabboversikt ---
                        if value_reports_list:
                            st.subheader("Snabboversikt — mest intressanta utfall")
                            ranked_outcomes = rank_outcomes_by_edge(value_reports_list)

                            positive_outcomes = [r for r in ranked_outcomes if r[2].edge > 0.001]
                            negative_outcomes = [r for r in ranked_outcomes if r[2].edge < -0.001]

                            if positive_outcomes:
                                st.markdown("**Hogst positiv edge:**")
                                pos_rows = []
                                for match_label, outcome_label, ov in positive_outcomes[:10]:
                                    # Markera oddskalla
                                    odds_source = "bild"
                                    for m in matches_with_odds:
                                        mkey = f"{m.home_team} vs {m.away_team}"
                                        if mkey == match_label and m.match_key in fallback_keys:
                                            odds_source = "fallback"
                                            break
                                    pos_rows.append({
                                        "Match": match_label,
                                        "Utfall": outcome_label,
                                        "Odds": f"{ov.odds:.2f} ({ov.bookmaker})",
                                        "Edge": f"{ov.edge:+.1%}",
                                        "Exp. return": f"{ov.expected_return:+.1%}",
                                        "Oddskalla": odds_source,
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

                        st.divider()

                        # --- Detaljvy per match ---
                        st.subheader("Detaljvy per match")

                        for match in matchday_matches:
                            if not match.odds_report:
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

                            if vr is not None and vr.outcomes:
                                max_edge = max(ov.edge for ov in vr.outcomes)
                                edge_hint = f"  |  Hogsta edge: {max_edge:+.1%}" if abs(max_edge) > 0.001 else ""
                            else:
                                edge_hint = ""

                            # Oddskalla-markering
                            if match.match_key in keys_with_odds:
                                src_tag = "bild"
                            elif match.match_key in fallback_keys:
                                src_tag = "fallback"
                            else:
                                src_tag = "okand"

                            status_parts = []
                            if match.has_odds:
                                status_parts.append(f"odds ({src_tag})")
                            if match.has_streck:
                                status_parts.append("streck")
                            status_tag = " + ".join(status_parts) if status_parts else "partiell"

                            with st.expander(
                                f"{match.home_team} vs {match.away_team}  |  "
                                f"Overround: {rpt.overround:.1f}%{edge_hint}  [{status_tag}]"
                            ):
                                # Bookmaker odds
                                bm_rows = []
                                for e in rpt.bookmaker_odds:
                                    bm_rows.append({
                                        "Bookmaker": e.bookmaker,
                                        "1 (Hemma)": f"{e.home:.2f}",
                                        "X (Oavgjort)": f"{e.draw:.2f}",
                                        "2 (Borta)": f"{e.away:.2f}",
                                    })
                                st.dataframe(pd.DataFrame(bm_rows), use_container_width=True, hide_index=True)

                                # Value
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

                                # Streck
                                if match.streck_report is not None:
                                    st.markdown("**Streckjamforelse:**")
                                    streck_rows = []
                                    for os_item in match.streck_report.outcomes:
                                        olabel = {"1": "Hemma (1)", "X": "Oavgjort (X)", "2": "Borta (2)"}[os_item.outcome]
                                        streck_rows.append({
                                            "Utfall": olabel,
                                            "Fair market prob": f"{os_item.fair_prob:.1%}",
                                            "Streckprocent": f"{os_item.streck_pct:.1%}",
                                            "Delta": f"{os_item.delta:+.1%}",
                                            "Bedomning": os_item.label,
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
                            "marknadens fair probability. "
                            "Oddskalla (bild/fallback) visas per match."
                        )

                    elif not matches_with_odds and matchday_matches:
                        st.warning(
                            "Inga matcher har giltiga odds. "
                            "Ratta oddsen i kontrolltabellen ovan, "
                            "eller lat appen anvanda fallback-odds."
                        )
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
            "Ladda upp en kupongbild ovan for att borja. "
            "Stodda format: PNG, JPG, JPEG, WEBP."
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

                # Samla streck-rapporter for sammanfattning
                streck_reports_for_summary = []
                if streck_lookup:
                    for rpt in reports:
                        skey = f"{rpt.home_team}_{rpt.away_team}"
                        if skey in streck_lookup:
                            sr = build_streck_report_from_odds_report(
                                rpt, streck_lookup[skey],
                            )
                            if sr is not None:
                                streck_reports_for_summary.append(sr)

                # --- Sammanfattning ---
                _show_analysis_summary(
                    [vr for _, vr in value_reports] if value_reports else [],
                    streck_reports_for_summary,
                    len(reports),
                )

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
