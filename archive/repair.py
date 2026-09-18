"""
archive/repair.py -- reparerar en omgangs metadata ur data som redan finns
i databasen. Ingen natverkstrafik, inga nya snapshots eller resultat.

Fallet: resultatet importerades fore snapshotet, `rounds` skapades utan
`round_matches`, och snapshotimporten fyllde inte pa matcherna. Reparationen
laser `market_snapshots.raw_payload` (legacy `matches[]` eller API-svarets
`draw.drawEvents`) och `results.raw_payload` och kompletterar det som saknas.
Befintliga rader raderas eller skrivs aldrig over; funktionen ar idempotent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.engine import Engine

from archive.db import (
    ROUND_STATUS_FINALIZED,
    market_snapshots,
    results,
    round_matches,
    rounds,
    session_scope,
)
from archive.fetch import (
    ArchiveFetchError,
    RoundMatch,
    _parse_events,
    ensure_round_row,
    merge_round_matches,
    parse_timestamp,
)
from archive.legacy import verifiable_matches

logger = logging.getLogger(__name__)


class RoundNotFound(LookupError):
    """Omgangen finns inte i `rounds`."""


@dataclass
class RepairOutcome:
    draw_number: int
    snapshots_scanned: int = 0
    snapshots_with_matches: int = 0
    matches_inserted: int = 0
    conflicts: List[int] = field(default_factory=list)
    reg_close_time_filled: bool = False
    status_finalized: bool = False
    match_count: int = 0

    @property
    def changed(self) -> bool:
        return bool(
            self.matches_inserted
            or self.reg_close_time_filled
            or self.status_finalized
        )


def _matches_from_snapshot_payload(payload: Any) -> List[RoundMatch]:
    """Legacy `matches[]` eller API-payloadens `draw.drawEvents`; annars []."""
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("matches"), list):
        return verifiable_matches(payload)
    draw = payload.get("draw")
    if isinstance(draw, dict) and draw.get("drawEvents"):
        try:
            return _parse_events(draw.get("drawEvents"))
        except ArchiveFetchError as exc:
            logger.warning("drawEvents kunde inte tolkas: %s", exc)
    return []


def repair_round(
    draw_number: int, *, engine: Optional[Engine] = None,
) -> RepairOutcome:
    """
    Fyller pa saknade `round_matches` fran omgangens sparade snapshots och
    kompletterar `reg_close_time`/status fran sparat resultat. Laser bara
    Postgres; skapar inget nytt snapshot och inget nytt resultat.
    """
    draw = int(draw_number)
    outcome = RepairOutcome(draw_number=draw)
    with session_scope(engine) as session:
        round_row = session.execute(
            select(rounds).where(rounds.c.draw_number == draw)
        ).mappings().first()
        if round_row is None:
            raise RoundNotFound(f"Omgang {draw} finns inte i arkivet.")

        snapshot_payloads = session.execute(
            select(market_snapshots.c.raw_payload)
            .where(market_snapshots.c.draw_number == draw)
            .order_by(market_snapshots.c.captured_at, market_snapshots.c.id)
        ).scalars().all()
        for payload in snapshot_payloads:
            outcome.snapshots_scanned += 1
            matches = _matches_from_snapshot_payload(payload)
            if not matches:
                continue
            outcome.snapshots_with_matches += 1
            merged = merge_round_matches(session, draw, matches)
            outcome.matches_inserted += merged.inserted
            for position in merged.conflicts:
                if position not in outcome.conflicts:
                    outcome.conflicts.append(position)

        result_row = session.execute(
            select(results.c.raw_payload).where(results.c.draw_number == draw)
        ).first()
        if result_row is not None:
            raw: Dict[str, Any] = (
                result_row[0] if isinstance(result_row[0], dict) else {}
            )
            outcome.reg_close_time_filled = ensure_round_row(
                session, draw,
                reg_close_time=parse_timestamp(raw.get("reg_close_time")),
            )
            if round_row["status"] != ROUND_STATUS_FINALIZED:
                session.execute(
                    rounds.update()
                    .where(rounds.c.draw_number == draw)
                    .values(status=ROUND_STATUS_FINALIZED)
                )
                outcome.status_finalized = True

        outcome.match_count = int(session.execute(
            select(func.count()).where(round_matches.c.draw_number == draw)
        ).scalar() or 0)

    logger.info(
        "Reparation av omgang %s: %d matcher tillagda, %d konflikter, "
        "reg_close_time %s, %d matcher totalt.",
        draw, outcome.matches_inserted, len(outcome.conflicts),
        "ifylld" if outcome.reg_close_time_filled else "oforandrad",
        outcome.match_count,
    )
    return outcome
