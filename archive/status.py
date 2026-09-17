"""
archive/status.py -- statusvy per omgang, harledd enbart fran databasen.

Anvandaren ska se omgangens status utan att kanna till filnamn eller
draw-id:n. Ingen natverkstrafik har.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.engine import Engine

from archive.db import (
    ROUND_STATUS_OPEN,
    market_snapshots,
    played_systems,
    results,
    round_matches,
    rounds,
    session_scope,
)


@dataclass
class SnapshotInfo:
    id: int
    captured_at: datetime
    captured_at_precision: str
    source: str
    parser_version: int


@dataclass
class PlayedSystemInfo:
    id: int
    created_at: datetime
    snapshot_id: Optional[int]
    n_halfguards: int
    note: str


@dataclass
class ResultInfo:
    correct_row: str
    fetched_at: datetime
    turnover: Optional[float]
    payouts: Dict[str, Optional[float]]
    winners: Dict[str, Optional[int]]
    source: str


@dataclass
class RoundStatus:
    draw_number: int
    week_label: Optional[str]
    reg_close_time: Optional[datetime]
    status: str
    match_count: int
    snapshots: List[SnapshotInfo] = field(default_factory=list)
    played_systems: List[PlayedSystemInfo] = field(default_factory=list)
    result: Optional[ResultInfo] = None

    @property
    def round_identified(self) -> bool:
        return self.match_count > 0

    @property
    def has_snapshot(self) -> bool:
        return bool(self.snapshots)

    @property
    def has_played_system(self) -> bool:
        return bool(self.played_systems)

    @property
    def result_state(self) -> str:
        """`finalized` nar resultat finns, annars `waiting`."""
        return "finalized" if self.result is not None else "waiting"

    @property
    def is_open(self) -> bool:
        return self.status == ROUND_STATUS_OPEN

    @property
    def latest_snapshot(self) -> Optional[SnapshotInfo]:
        return self.snapshots[-1] if self.snapshots else None


def _utc(moment: Optional[datetime]) -> Optional[datetime]:
    """SQLite ger naiva tidsstamplar; allt i arkivet ar lagrat i UTC."""
    if moment is None:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def list_round_status(
    *,
    limit: int = 20,
    engine: Optional[Engine] = None,
) -> List[RoundStatus]:
    """Senaste `limit` omgangarna, nyast forst, med all status harledd."""
    with session_scope(engine) as session:
        round_rows = session.execute(
            select(rounds).order_by(rounds.c.draw_number.desc()).limit(limit)
        ).mappings().all()
        if not round_rows:
            return []
        draws = [r["draw_number"] for r in round_rows]

        match_counts = dict(session.execute(
            select(round_matches.c.draw_number, func.count())
            .where(round_matches.c.draw_number.in_(draws))
            .group_by(round_matches.c.draw_number)
        ).all())

        snaps: Dict[int, List[SnapshotInfo]] = {d: [] for d in draws}
        for row in session.execute(
            select(market_snapshots)
            .where(market_snapshots.c.draw_number.in_(draws))
            .order_by(market_snapshots.c.captured_at, market_snapshots.c.id)
        ).mappings():
            snaps[row["draw_number"]].append(SnapshotInfo(
                id=row["id"],
                captured_at=_utc(row["captured_at"]),
                captured_at_precision=row["captured_at_precision"],
                source=row["source"],
                parser_version=row["parser_version"],
            ))

        systems: Dict[int, List[PlayedSystemInfo]] = {d: [] for d in draws}
        for row in session.execute(
            select(played_systems)
            .where(played_systems.c.draw_number.in_(draws))
            .order_by(played_systems.c.created_at, played_systems.c.id)
        ).mappings():
            systems[row["draw_number"]].append(PlayedSystemInfo(
                id=row["id"],
                created_at=_utc(row["created_at"]),
                snapshot_id=row["snapshot_id"],
                n_halfguards=row["n_halfguards"],
                note=row["note"] or "",
            ))

        result_map: Dict[int, ResultInfo] = {}
        for row in session.execute(
            select(results).where(results.c.draw_number.in_(draws))
        ).mappings():
            result_map[row["draw_number"]] = ResultInfo(
                correct_row=row["correct_row"],
                fetched_at=_utc(row["fetched_at"]),
                turnover=row["turnover"],
                payouts={t: row[f"payout_{t}"] for t in ("13", "12", "11", "10")},
                winners={t: row[f"winners_{t}"] for t in ("13", "12", "11", "10")},
                source=row["source"],
            )

    return [
        RoundStatus(
            draw_number=r["draw_number"],
            week_label=r["week_label"],
            reg_close_time=_utc(r["reg_close_time"]),
            status=r["status"],
            match_count=int(match_counts.get(r["draw_number"], 0)),
            snapshots=snaps[r["draw_number"]],
            played_systems=systems[r["draw_number"]],
            result=result_map.get(r["draw_number"]),
        )
        for r in round_rows
    ]


def current_round_status(
    *, engine: Optional[Engine] = None,
) -> Optional[RoundStatus]:
    """Den oppna omgangen om en finns, annars den senaste."""
    statuses = list_round_status(limit=50, engine=engine)
    for status in statuses:
        if status.is_open:
            return status
    return statuses[0] if statuses else None


def get_round_status(
    draw_number: int, *, engine: Optional[Engine] = None,
) -> Optional[RoundStatus]:
    for status in list_round_status(limit=10_000, engine=engine):
        if status.draw_number == int(draw_number):
            return status
    return None
