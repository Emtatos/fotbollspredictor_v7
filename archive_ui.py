"""
archive_ui.py -- Streamlit-hjalpare for omgangsarkivet (archive/).

Har finns allt som kraver Streamlit: felmeddelande nar DATABASE_URL saknas,
och kopplingen mellan session state (current_round) och snapshot i databasen.
De rena hamtfunktionerna ligger i `archive.fetch` och ar Streamlit-fria.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

import streamlit as st
from sqlalchemy.engine import Engine

from archive.db import (
    DATABASE_URL_ENV,
    SOURCE_IMAGE_SCAN,
    SOURCE_MANUAL,
    SOURCE_PASTE,
    DatabaseNotConfigured,
    get_engine,
)
from archive.fetch import SnapshotRow, save_snapshot
from archive.status import current_round_status
from matchday_import import _make_key

SNAPSHOT_STATE_KEY = "archive_snapshot"

DATABASE_MISSING_MESSAGE = (
    f"Miljovariabeln `{DATABASE_URL_ENV}` saknas. Arkivet kraver Render "
    "Postgres (Internal Database URL) och faller inte tillbaka pa filer. "
    "Satt variabeln pa webbtjansten i Render och starta om."
)

_SOURCE_MAP = {
    "kupongbild": SOURCE_IMAGE_SCAN,
    "image_scan": SOURCE_IMAGE_SCAN,
    "textpaste": SOURCE_PASTE,
    "nyimporterad": SOURCE_PASTE,
    "saved": SOURCE_PASTE,
    "paste": SOURCE_PASTE,
    "manual": SOURCE_MANUAL,
}


def engine_or_error() -> Optional[Engine]:
    """Delad engine, eller None efter att ett tydligt fel visats."""
    try:
        return get_engine()
    except DatabaseNotConfigured:
        st.error(DATABASE_MISSING_MESSAGE)
        return None


def snapshot_source_for(source_label: Optional[str]) -> str:
    """Oversatter UI-kallor (kupongbild/textpaste/...) till arkivets kallor."""
    return _SOURCE_MAP.get((source_label or "").lower(), SOURCE_PASTE)


def _lookup(mapping: Dict[str, Any], key: str) -> Any:
    if key in mapping:
        return mapping[key]
    lowered = key.lower()
    for candidate, value in mapping.items():
        if candidate.lower() == lowered:
            return value
    return None


def _odds_values(entries: Any):
    entry = entries[0] if isinstance(entries, list) and entries else entries
    if entry is None:
        return None, None, None
    try:
        if hasattr(entry, "home"):
            return float(entry.home), float(entry.draw), float(entry.away)
        if isinstance(entry, dict):
            return (
                float(entry["home"]), float(entry["draw"]), float(entry["away"]),
            )
    except (KeyError, TypeError, ValueError):
        pass
    return None, None, None


def _streck_values(streck: Any):
    if not isinstance(streck, dict):
        return None, None, None
    try:
        return (
            float(streck["1"]), float(streck["X"]), float(streck["2"]),
        )
    except (KeyError, TypeError, ValueError):
        return None, None, None


def snapshot_rows_from_current_round(
    current_round: Dict[str, Any],
) -> List[SnapshotRow]:
    """Bygger snapshotrader fran `st.session_state["current_round"]`."""
    odds_by_key = current_round.get("odds") or {}
    streck_by_key = current_round.get("streck") or {}
    rows: List[SnapshotRow] = []
    for position, (home, away) in enumerate(
        current_round.get("matches") or [], start=1,
    ):
        key = _make_key(home, away)
        o1, ox, o2 = _odds_values(_lookup(odds_by_key, key))
        s1, sx, s2 = _streck_values(_lookup(streck_by_key, key))
        rows.append(SnapshotRow(
            position=position,
            streck_1=s1, streck_x=sx, streck_2=s2,
            odds_1=o1, odds_x=ox, odds_2=o2,
            home_team=str(home), away_team=str(away),
        ))
    return rows


def snapshot_rows_from_snapshot_matches(matches: List[Any]) -> List[SnapshotRow]:
    """Fran `snapshot_storage.SnapshotMatch` (parsning av kupong/paste)."""
    return [
        SnapshotRow(
            position=int(m.position),
            streck_1=m.streck_1, streck_x=m.streck_x, streck_2=m.streck_2,
            odds_1=m.odds_1, odds_x=m.odds_x, odds_2=m.odds_2,
            home_team=m.home_team, away_team=m.away_team, league=m.league,
        )
        for m in matches
    ]


def snapshot_fingerprint(rows: List[SnapshotRow]) -> str:
    """
    Hash av marknadsdatan (lag, streck, odds per match). Anvands for att
    binda ett ihagkommet snapshot till exakt den data raden byggdes pa.
    """
    payload = [
        [
            r.position, r.home_team, r.away_team,
            r.streck_1, r.streck_x, r.streck_2,
            r.odds_1, r.odds_x, r.odds_2,
        ]
        for r in sorted(rows, key=lambda r: r.position)
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def current_round_fingerprint() -> Optional[str]:
    """Fingerprint av `st.session_state["current_round"]`, eller None."""
    current_round = st.session_state.get("current_round")
    if not current_round or not current_round.get("matches"):
        return None
    return snapshot_fingerprint(snapshot_rows_from_current_round(current_round))


def remember_snapshot(
    draw_number: int,
    snapshot_id: int,
    source: str,
    fingerprint: Optional[str] = None,
) -> None:
    st.session_state[SNAPSHOT_STATE_KEY] = {
        "draw_number": int(draw_number),
        "snapshot_id": int(snapshot_id),
        "source": source,
        "fingerprint": fingerprint,
    }


def remembered_snapshot_id(
    draw_number: int, fingerprint: Optional[str] = None,
) -> Optional[int]:
    """
    Ihagkommet snapshot for omgangen. Anges `fingerprint` returneras id bara
    om snapshotet togs pa exakt samma data.
    """
    stored = st.session_state.get(SNAPSHOT_STATE_KEY)
    if not stored or int(stored.get("draw_number", -1)) != int(draw_number):
        return None
    if fingerprint is not None and stored.get("fingerprint") != fingerprint:
        return None
    return int(stored["snapshot_id"])


def default_draw_number(engine: Engine) -> Optional[int]:
    """Oppen (eller senaste) omgang i arkivet, som forslag i formular."""
    status = current_round_status(engine=engine)
    return status.draw_number if status else None


def ensure_snapshot_for_current_round(
    draw_number: int,
    current_round: Dict[str, Any],
    *,
    engine: Engine,
) -> int:
    """
    Snapshot-id for raden som byggs fran current_round. Ett ihagkommet
    snapshot ateranvands bara om dess fingerprint matchar aktuell data;
    annars skapas ett nytt (image_scan for kupongbild, annars paste) sa att
    raden aldrig kopplas till ett snapshot med annan marknadsdata.
    """
    rows = snapshot_rows_from_current_round(current_round)
    fingerprint = snapshot_fingerprint(rows)
    existing = remembered_snapshot_id(draw_number, fingerprint)
    if existing is not None:
        return existing
    source = snapshot_source_for(current_round.get("source"))
    snapshot_id = save_snapshot(
        int(draw_number), rows, source=source, engine=engine,
    )
    remember_snapshot(draw_number, snapshot_id, source, fingerprint)
    return snapshot_id
