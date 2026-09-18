"""
Sida: Arkiv — omgangens status direkt fran databasen (Render Postgres).

Anvandaren ser aktuell och tidigare omgangar utan att kanna till filnamn,
JSON eller draw-id:n. Knapparna anropar de rena funktionerna i
`archive.fetch`; ingen fil-fallback finns.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from archive.fetch import (
    ArchiveFetchError,
    ResultConflictError,
    capture_market_snapshot,
    fetch_current_round,
    fetch_result,
)
from archive.legacy import (
    DrawNumberRequired,
    DuplicateImport,
    LegacyImportError,
    import_legacy_json,
)
from archive.repair import RoundNotFound, repair_round
from archive.status import RoundStatus, list_round_status
from archive_ui import engine_or_error
from svenskaspel_results import ResultFetchError

st.header("Arkiv: omgangens status")
st.caption(
    "Allt nedan lases fran databasen. Hamtningar gor exakt ett API-anrop "
    "per knapptryck; inget skrivs over."
)

engine = engine_or_error()
if engine is None:
    st.stop()

FLASH_KEY = "archive_flash"


def _flash(kind: str, text: str) -> None:
    """Visa meddelandet efter nasta st.rerun()."""
    st.session_state[FLASH_KEY] = (kind, text)


_pending = st.session_state.pop(FLASH_KEY, None)
if _pending:
    {"success": st.success, "info": st.info, "error": st.error}[_pending[0]](
        _pending[1]
    )


def _fmt_time(moment) -> str:
    if moment is None:
        return "—"
    return moment.strftime("%Y-%m-%d %H:%M UTC")


def _status_label(status: RoundStatus) -> str:
    return {
        "open": "Oppen",
        "closed": "Stangd",
        "finalized": "Avslutad",
    }.get(status.status, status.status)


def _render_actions(current: Optional[RoundStatus]) -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(
            "Hamta aktuell omgang", use_container_width=True,
            key="archive_fetch_round",
        ):
            try:
                rnd = fetch_current_round(engine=engine)
            except ArchiveFetchError as exc:
                st.error(f"Hamtningen misslyckades: {exc}")
            else:
                if rnd is None:
                    st.info("Ingen oppen omgang just nu.")
                else:
                    _flash(
                        "success",
                        f"Omgang {rnd.draw_number} ({rnd.week_label or '?'}) "
                        f"identifierad med {len(rnd.matches)} matcher.",
                    )
                    st.rerun()
    with col2:
        disabled = current is None
        if st.button(
            "Ta marknadssnapshot", use_container_width=True,
            key="archive_capture_snapshot", disabled=disabled,
            help="Sparar aktuella streck och odds som en ny observation.",
        ):
            try:
                snapshot_id = capture_market_snapshot(
                    current.draw_number, engine=engine,
                )
            except ArchiveFetchError as exc:
                st.error(f"Snapshot misslyckades: {exc}")
            else:
                _flash(
                    "success",
                    f"Snapshot #{snapshot_id} sparat for omgang "
                    f"{current.draw_number}.",
                )
                st.rerun()
    with col3:
        disabled = current is None
        if st.button(
            "Hamta resultat", use_container_width=True,
            key="archive_fetch_result", disabled=disabled,
            help="Hamtar ratt rad och utdelning nar omgangen ar avgjord.",
        ):
            try:
                fetch_result(current.draw_number, engine=engine)
            except ResultConflictError as exc:
                st.error(str(exc))
            except (ResultFetchError, ArchiveFetchError) as exc:
                st.error(f"Resultathamtning misslyckades: {exc}")
            else:
                _flash(
                    "success",
                    f"Resultat sparat for omgang {current.draw_number}.",
                )
                st.rerun()


def _render_round(status: RoundStatus, *, expanded: bool) -> None:
    title = (
        f"Omgang {status.draw_number}"
        f"{' · ' + status.week_label if status.week_label else ''}"
        f" · {_status_label(status)}"
    )
    with st.expander(title, expanded=expanded):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Spelstopp", _fmt_time(status.reg_close_time))
        m2.metric(
            "Omgang identifierad",
            "Ja" if status.round_identified else "Nej",
            help=f"{status.match_count} matcher i databasen.",
        )
        m3.metric(
            "Spelad rad",
            (
                f"{len(status.played_systems)} st"
                if status.has_played_system else "Saknas"
            ),
            help=(
                ", ".join(
                    f"#{p.id}: {p.n_halfguards} halvgarderingar"
                    for p in status.played_systems
                ) or None
            ),
        )
        m4.metric(
            "Resultat",
            "Klart" if status.result_state == "finalized" else "Vantar",
        )

        st.markdown("**Marknadssnapshots**")
        if not status.snapshots:
            st.caption("Inga snapshots annu.")
        else:
            st.dataframe(
                pd.DataFrame([
                    {
                        "Id": s.id,
                        "Tidpunkt": _fmt_time(s.captured_at),
                        "Precision": s.captured_at_precision,
                        "Kalla": s.source,
                        "Parser": s.parser_version,
                    }
                    for s in status.snapshots
                ]),
                use_container_width=True, hide_index=True,
            )

        if status.played_systems:
            st.markdown("**Spelade rader**")
            st.dataframe(
                pd.DataFrame([
                    {
                        "Id": p.id,
                        "Registrerad": _fmt_time(p.created_at),
                        "Snapshot": p.snapshot_id if p.snapshot_id else "—",
                        "Halvgarderingar": p.n_halfguards,
                        "Notering": p.note,
                    }
                    for p in status.played_systems
                ]),
                use_container_width=True, hide_index=True,
            )

        if status.result is not None:
            r = status.result
            st.markdown("**Resultat**")
            st.code(r.correct_row, language=None)
            st.dataframe(
                pd.DataFrame([
                    {
                        "Ratt": tier,
                        "Utdelning": r.payouts.get(tier),
                        "Vinnare": r.winners.get(tier),
                    }
                    for tier in ("13", "12", "11", "10")
                ]),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                f"Omsattning: {r.turnover if r.turnover is not None else '—'}"
                f" · hamtat {_fmt_time(r.fetched_at)} · kalla {r.source}"
            )

        if st.button(
            "Reparera omgang", key=f"archive_repair_{status.draw_number}",
            help=(
                "Fyller pa saknade matcher och spelstopp fran omgangens redan "
                "sparade snapshots och resultat. Inget natverksanrop, inget "
                "nytt snapshot eller resultat."
            ),
        ):
            try:
                outcome = repair_round(status.draw_number, engine=engine)
            except RoundNotFound as exc:
                st.error(str(exc))
            else:
                if outcome.changed:
                    _flash(
                        "success",
                        f"Omgang {status.draw_number} reparerad: "
                        f"{outcome.matches_inserted} matcher tillagda "
                        f"({outcome.match_count} totalt)"
                        + (", spelstopp ifyllt" if outcome.reg_close_time_filled else "")
                        + (
                            f", {len(outcome.conflicts)} lagkonflikter behallna"
                            if outcome.conflicts else ""
                        )
                        + ".",
                    )
                else:
                    _flash(
                        "info",
                        f"Omgang {status.draw_number}: inget att reparera "
                        f"({outcome.match_count} matcher, "
                        f"{outcome.snapshots_scanned} snapshots genomsokta).",
                    )
                st.rerun()


def _render_legacy_import() -> None:
    st.markdown("---")
    with st.expander("Importera aldre JSON", expanded=False):
        st.caption(
            "Snapshot- och resultatfiler fran det gamla filarkivet "
            "(`data/snapshots/*.json`, `data/results/*.json`). Saknar filen "
            "omgangsnummer (`unknown_...json`) anger du det nedan."
        )
        uploads = st.file_uploader(
            "JSON-filer", type=["json"], accept_multiple_files=True,
            key="archive_legacy_files",
        )
        draw_raw = st.text_input(
            "Omgangsnummer (kravs bara om filen saknar draw)",
            key="archive_legacy_draw",
        )
        if st.button("Importera", key="archive_legacy_import_btn"):
            if not uploads:
                st.warning("Valj minst en fil.")
                return
            override = int(draw_raw) if draw_raw.strip().isdigit() else None
            for upload in uploads:
                try:
                    outcome = import_legacy_json(
                        upload.getvalue(), draw_number=override, engine=engine,
                    )
                except DrawNumberRequired:
                    st.error(
                        f"{upload.name}: filen saknar omgangsnummer, ange det "
                        "i faltet ovan och importera igen."
                    )
                except DuplicateImport as exc:
                    st.warning(f"{upload.name}: {exc}")
                except LegacyImportError as exc:
                    st.error(f"{upload.name}: {exc}")
                else:
                    label = (
                        f"snapshot #{outcome.snapshot_id}"
                        if outcome.kind == "snapshot" else "resultat"
                    )
                    st.success(
                        f"{upload.name}: importerat som {label} for omgang "
                        f"{outcome.draw_number}."
                    )


statuses = list_round_status(limit=20, engine=engine)
current = next((s for s in statuses if s.is_open), statuses[0] if statuses else None)

if current is None:
    st.info(
        "Arkivet ar tomt. Tryck `Hamta aktuell omgang` for att identifiera "
        "veckans omgang."
    )
else:
    st.subheader(
        f"Aktuell omgang: {current.draw_number}"
        f"{' (' + current.week_label + ')' if current.week_label else ''}"
    )

_render_actions(current)

st.markdown("---")
for status in statuses:
    _render_round(status, expanded=(status is current))

_render_legacy_import()
