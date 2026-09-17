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
from combined_probability import combined_from_current_round, describe_sources_used
from utils import set_canonical_teams, get_canonical_teams
from matchday_import import _make_key
from archive.db import MATCH_COUNT
from archive.fetch import (
    IncompleteRowError,
    SnapshotRequiredError,
    register_played_system,
)
from archive_ui import (
    default_draw_number,
    engine_or_error,
    ensure_snapshot_for_current_round,
    remembered_snapshot_id,
    snapshot_fingerprint,
    snapshot_rows_from_current_round,
)

PLAYED_ROW_STATE_KEY = "played_row_for_archive"


def _prob_or_none(cm, index):
    if cm is None or cm.probs is None:
        return None
    return float(cm.probs[index])


def _gain_or_none(cm):
    """gain = nast hogsta kombinerade sannolikheten (samma som ui_utils)."""
    if cm is None or cm.probs is None:
        return None
    return float(sorted(cm.probs, reverse=True)[1])


def _row_is_complete(played) -> bool:
    positions = {int(r["position"]) for r in played["rows"]}
    return (
        len(played["rows"]) == MATCH_COUNT
        and positions == set(range(1, MATCH_COUNT + 1))
    )


def _render_register_played_system():
    """Knappen `Registrera spelad rad`: sparar raden i arkivet."""
    played = st.session_state.get(PLAYED_ROW_STATE_KEY)
    if not played or not played.get("rows"):
        return

    st.markdown("---")
    st.subheader("Registrera spelad rad i arkivet")
    engine = engine_or_error()
    if engine is None:
        return

    cr = st.session_state.get("current_round") or {}
    suggested = default_draw_number(engine)
    col1, col2 = st.columns([1, 2])
    with col1:
        draw_raw = st.text_input(
            "Omgangsnummer",
            value=str(suggested) if suggested else "",
            key="played_row_draw",
            help="Forslaget ar den oppna omgangen i arkivet.",
        )
    with col2:
        note = st.text_input("Notering (valfritt)", key="played_row_note")

    complete = _row_is_complete(played)
    if not complete:
        st.error(
            f"Raden ar inte komplett: {len(played['rows'])} av {MATCH_COUNT} "
            "matcher har tips. En Stryktipsrad med farre an 13 matcher kan "
            "inte registreras."
        )
    st.caption(
        f"Rad: `{played['tipsrad']}` · {played['n_halfguards']} halvgarderingar"
    )

    has_market_data = bool(cr.get("matches"))
    if draw_raw.strip().isdigit() and has_market_data:
        fingerprint = snapshot_fingerprint(snapshot_rows_from_current_round(cr))
        known = remembered_snapshot_id(int(draw_raw.strip()), fingerprint)
        if known is None:
            st.caption(
                "Inget snapshot av exakt dessa odds/streck finns i sessionen: "
                f"ett snapshot skapas automatiskt fran "
                f"{cr.get('source', 'importen')} innan raden registreras."
            )
    elif not has_market_data:
        st.warning(
            "Inga marknadsdata (odds/streck) finns i sessionen. Importera "
            "omgangen under Odds & Value forst; en spelad rad utan snapshot "
            "kan inte registreras."
        )

    if st.button(
        "Registrera spelad rad", type="primary", use_container_width=True,
        key="played_row_register_btn",
        disabled=not (complete and has_market_data),
    ):
        if not draw_raw.strip().isdigit():
            st.error("Omgangsnummer maste anges (heltal).")
            return
        draw_number = int(draw_raw.strip())
        try:
            snapshot_id = ensure_snapshot_for_current_round(
                draw_number, cr, engine=engine,
            )
            system_id = register_played_system(
                draw_number, played["rows"], snapshot_id,
                note=note.strip(), engine=engine,
            )
        except (IncompleteRowError, SnapshotRequiredError) as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001 -- visa alla fel i UI:t
            st.error(f"Kunde inte registrera raden: {exc}")
        else:
            st.success(
                f"Spelad rad #{system_id} registrerad for omgang {draw_number} "
                f"(snapshot #{snapshot_id}, {played['n_halfguards']} "
                "halvgarderingar)."
            )


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

def _safe_odds_values(entry):
    """Extrahera (home, draw, away) från OddsEntry-objekt eller dict utan krasch."""
    if entry is None:
        return None, None, None
    try:
        if hasattr(entry, "home"):
            return float(entry.home), float(entry.draw), float(entry.away)
        if isinstance(entry, dict):
            return (
                float(entry["home"]),
                float(entry["draw"]),
                float(entry["away"]),
            )
    except (KeyError, TypeError, ValueError):
        pass
    return None, None, None


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

            # Separera tydligt:
            #   model_probs_list  — riktiga modellprobabiliteter (None om fallback)
            #   data_sources      — "modell", "odds (fallback)", "N/A"
            #   trust_labels      — trust-etikett per match
            model_probs_list = []   # None för fallback-matcher
            data_sources = []       # "modell", "odds (fallback)", "N/A"
            trust_labels = []       # trust-etikett per match

            for home, away in matches:
                result = predict_match(model, home, away, df_features)

                if result is not None:
                    probs, stats = result
                    model_probs_list.append(probs)
                    data_sources.append("modell")
                    trust_lbl = stats.get('trust_label', 'N/A')
                    if trust_lbl == "LOW":
                        trust_lbl = "LOW (varning)"
                    trust_labels.append(trust_lbl)
                else:
                    # Fallback: modell saknas — skicka INTE odds som model_probs
                    model_probs_list.append(None)

                    # Kolla om odds finns (för att skilja fallback från N/A)
                    odds_entries, _ = _lookup_round_odds(home, away)
                    has_odds = False
                    if odds_entries:
                        o1, ox, o2 = _safe_odds_values(odds_entries[0])
                        has_odds = o1 is not None

                    if has_odds:
                        data_sources.append("odds (fallback)")
                    else:
                        data_sources.append("N/A")
                    trust_labels.append("—" if has_odds else "N/A")

            # Kombinerade sannolikheter via den gemensamma buildern
            # (samma som Odds & Value).
            combined_matches = combined_from_current_round(
                matches=matches,
                current_round=st.session_state.get("current_round"),
                model_probs=model_probs_list,
                make_key=_make_key,
            )

            # Bygg resultat-tabell.
            # UI visar *kombinerade* sannolikheter i 1/X/2 — samma som
            # Tips och HALV bygger på — så att tabellen är sanningsenlig.
            results = []
            for i, (home, away) in enumerate(matches):
                cm = combined_matches[i]
                ds = data_sources[i]

                if ds == "N/A":
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
                else:
                    # Bygg källbeskrivning från combined-signaler
                    source_parts = []
                    if cm.sources["model"]:
                        source_parts.append("modell")
                    if cm.sources["odds"]:
                        source_parts.append("odds")
                    if cm.sources["streck"]:
                        source_parts.append("streck")
                    if ds == "odds (fallback)":
                        source_label = "odds fallback"
                        if cm.sources["streck"]:
                            source_label += " + streck"
                    else:
                        source_label = " + ".join(source_parts) if source_parts else ds

                    c_probs = cm.probs
                    sign = ['1', 'X', '2'][np.argmax(c_probs)]

                    results.append({
                        "Match": f"{home} - {away}",
                        "1": f"{c_probs[0]:.1%}",
                        "X": f"{c_probs[1]:.1%}",
                        "2": f"{c_probs[2]:.1%}",
                        "Källa": source_label,
                        "Trust": trust_labels[i],
                        "Tips": sign,
                        "HALV": ""
                    })

            # Applicera halvgarderingar
            if num_halfguards > 0:
                # Använd kombinerade sannolikheter för halvgardering
                guard_indices = pick_half_guards_combined(combined_matches, num_halfguards)
                for idx in guard_indices:
                    cm = combined_matches[idx]
                    results[idx]["Tips"] = get_halfguard_sign_combined(cm)
                    results[idx]["HALV"] = "HALV"

                sources_used = describe_sources_used(combined_matches)
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

            st.caption(
                "Visade sannolikheter (1/X/2) är de kombinerade sannolikheter "
                "som Tips och halvgarderingar bygger på."
            )

            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True, hide_index=True)

            # Visa tipsrad
            st.subheader("📝 Tipsrad för kopiering")
            tipsrad = "".join([r["Tips"] for r in results if r["Tips"] != "?"])
            st.code(tipsrad, language=None)

            # Spara raden for registrering i arkivet (knappen nedan lever
            # utanfor klick-blocket sa att den overlever Streamlits rerun).
            st.session_state[PLAYED_ROW_STATE_KEY] = {
                "rows": [
                    {
                        "position": i + 1,
                        "played_signs": r["Tips"],
                        "is_halfguard": r["HALV"] == "HALV",
                        "combined_p1": _prob_or_none(combined_matches[i], 0),
                        "combined_px": _prob_or_none(combined_matches[i], 1),
                        "combined_p2": _prob_or_none(combined_matches[i], 2),
                        "gain": _gain_or_none(combined_matches[i]),
                        "sources_used": r["Källa"],
                    }
                    for i, r in enumerate(results)
                    if r["Tips"] != "?"
                ],
                "n_halfguards": int(num_halfguards),
                "tipsrad": tipsrad,
                "match_count": len(results),
            }

_render_register_played_system()
