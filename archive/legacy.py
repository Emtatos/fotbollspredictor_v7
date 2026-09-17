"""
archive/legacy.py -- import av aldre JSON-filer fran `snapshot_storage.py`.

Snapshot-JSON  -> market_snapshots (source=legacy_import) + matchrader.
Resultat-JSON  -> results + result_matches (source=legacy_import).

Saknar filen draw (t.ex. `unknown_...json`) maste anroparen ange draw.
Ett snapshot med samma draw och captured_at som redan finns avvisas.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import select
from sqlalchemy.engine import Engine

from archive.db import (
    CAPTURED_AT_PRECISIONS,
    PRECISION_UNKNOWN,
    SOURCE_LEGACY_IMPORT,
    market_snapshots,
    session_scope,
)
from archive.fetch import (
    SnapshotRow,
    parse_timestamp,
    save_result,
    save_snapshot,
)
from svenskaspel_results import FetchedMatch, FetchedResult

logger = logging.getLogger(__name__)

LEGACY_PARSER_VERSION = 0

KIND_SNAPSHOT = "snapshot"
KIND_RESULT = "result"


class LegacyImportError(ValueError):
    """Filen kunde inte importeras. Inget har skrivits."""


class DrawNumberRequired(LegacyImportError):
    """Filen saknar draw; anroparen maste ange omgangsnummer."""


class DuplicateImport(LegacyImportError):
    """Samma draw + captured_at finns redan i arkivet."""


@dataclass
class LegacyImportOutcome:
    kind: str
    draw_number: int
    snapshot_id: Optional[int] = None
    written: bool = True


def load_legacy_json(data: Union[bytes, str, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(data, dict):
        return data
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise LegacyImportError(f"Ogiltig JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LegacyImportError("JSON-filen ar inte ett objekt.")
    return payload


def detect_kind(payload: Dict[str, Any]) -> str:
    """`snapshot` om filen har `matches`, `result` om den har `correct_row`."""
    if "correct_row" in payload:
        return KIND_RESULT
    if "matches" in payload or "captured_at" in payload:
        return KIND_SNAPSHOT
    raise LegacyImportError(
        "Kande inte igen filen: varken snapshot (matches) eller "
        "resultat (correct_row)."
    )


def _resolve_draw(payload: Dict[str, Any], draw_number: Optional[int]) -> int:
    raw = payload.get("draw")
    if raw not in (None, "", 0):
        try:
            file_draw = int(raw)
        except (TypeError, ValueError) as exc:
            raise LegacyImportError(f"Ogiltigt draw i filen: {raw!r}") from exc
        if draw_number is not None and int(draw_number) != file_draw:
            raise LegacyImportError(
                f"Filen anger omgang {file_draw} men {draw_number} angavs."
            )
        return file_draw
    if draw_number is None:
        raise DrawNumberRequired(
            "Filen saknar omgangsnummer (draw=null). Ange omgang manuellt."
        )
    return int(draw_number)


def _float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return None


def _int(value: Any) -> Optional[int]:
    number = _float(value)
    return None if number is None else int(number)


def import_snapshot_payload(
    payload: Dict[str, Any],
    *,
    draw_number: Optional[int] = None,
    engine: Optional[Engine] = None,
) -> LegacyImportOutcome:
    draw = _resolve_draw(payload, draw_number)
    precision = payload.get("captured_at_precision") or PRECISION_UNKNOWN
    if precision not in CAPTURED_AT_PRECISIONS:
        raise LegacyImportError(
            f"Ogiltig captured_at_precision: {precision!r}"
        )
    captured_at = parse_timestamp(payload.get("captured_at"))
    if captured_at is None:
        raise LegacyImportError(
            f"Ogiltig captured_at: {payload.get('captured_at')!r}"
        )

    rows: List[SnapshotRow] = []
    for index, item in enumerate(payload.get("matches") or [], start=1):
        if not isinstance(item, dict):
            continue
        rows.append(SnapshotRow(
            position=_int(item.get("position")) or index,
            streck_1=_float(item.get("streck_1")),
            streck_x=_float(item.get("streck_x")),
            streck_2=_float(item.get("streck_2")),
            odds_1=_float(item.get("odds_1")),
            odds_x=_float(item.get("odds_x")),
            odds_2=_float(item.get("odds_2")),
            home_team=str(item.get("home_team") or "").strip() or None,
            away_team=str(item.get("away_team") or "").strip() or None,
            league=item.get("league") or None,
        ))
    if not rows:
        raise LegacyImportError("Snapshot-filen innehaller inga matcher.")

    with session_scope(engine) as session:
        duplicate = session.execute(
            select(market_snapshots.c.id).where(
                market_snapshots.c.draw_number == draw,
                market_snapshots.c.captured_at == captured_at,
            )
        ).first()
    if duplicate is not None:
        raise DuplicateImport(
            f"Snapshot for omgang {draw} med captured_at "
            f"{captured_at.isoformat()} finns redan (id={duplicate[0]})."
        )

    snapshot_id = save_snapshot(
        draw, rows,
        source=SOURCE_LEGACY_IMPORT,
        captured_at=captured_at,
        captured_at_precision=precision,
        raw_payload=payload,
        parser_version=LEGACY_PARSER_VERSION,
        engine=engine,
    )
    return LegacyImportOutcome(KIND_SNAPSHOT, draw, snapshot_id=snapshot_id)


def import_result_payload(
    payload: Dict[str, Any],
    *,
    draw_number: Optional[int] = None,
    engine: Optional[Engine] = None,
) -> LegacyImportOutcome:
    draw = _resolve_draw(payload, draw_number)
    correct_row = [str(s).strip().upper() for s in payload.get("correct_row") or []]
    payouts = payload.get("payouts") or {}
    winners = payload.get("winners") or {}
    fetched = FetchedResult(
        draw=draw,
        correct_row=correct_row,
        turnover=_float(payload.get("turnover")),
        payouts={t: _float(payouts.get(t)) for t in ("13", "12", "11", "10")},
        winners={t: _float(winners.get(t)) for t in ("13", "12", "11", "10")},
        matches=[
            FetchedMatch(position=i, description="", sign=sign)
            for i, sign in enumerate(correct_row, start=1)
        ],
        reg_close_time=payload.get("reg_close_time"),
        draw_state=payload.get("draw_state"),
    )
    if not fetched.is_complete:
        raise LegacyImportError(
            f"Resultatfilen har ogiltig rad ({len(correct_row)} tecken)."
        )
    written = save_result(
        fetched,
        source=SOURCE_LEGACY_IMPORT,
        raw_payload=payload,
        fetched_at=parse_timestamp(payload.get("entered_at")),
        parser_version=LEGACY_PARSER_VERSION,
        engine=engine,
    )
    if not written:
        raise DuplicateImport(
            f"Resultat for omgang {draw} finns redan i arkivet."
        )
    return LegacyImportOutcome(KIND_RESULT, draw)


def import_legacy_json(
    data: Union[bytes, str, Dict[str, Any]],
    *,
    draw_number: Optional[int] = None,
    engine: Optional[Engine] = None,
) -> LegacyImportOutcome:
    """Importerar en aldre snapshot- eller resultatfil."""
    payload = load_legacy_json(data)
    kind = detect_kind(payload)
    if kind == KIND_RESULT:
        return import_result_payload(
            payload, draw_number=draw_number, engine=engine,
        )
    return import_snapshot_payload(
        payload, draw_number=draw_number, engine=engine,
    )
