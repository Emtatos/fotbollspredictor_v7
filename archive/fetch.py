"""
archive/fetch.py -- hamtfunktioner for omgangsarkivet. Ren Python.

Inga Streamlit-beroenden: samma funktioner anropas fran UI:t (knappar) och
fran ett framtida cron-jobb (PR 2). Varje funktion gor exakt ett API-anrop
(tva for `fetch_current_round` om listan skulle sakna matcher), ingen loop,
ingen retry, ingen backfill. Fel loggas och kastas.

Faltnamn i Svenska Spels svar (verifierade mot riktiga svar 2026-09-17):

    /draw/1/stryktipset/draws
        draws[]: drawNumber, drawState ("Open"), drawComment
                 ("Stryktipset v. 2026-38"), regCloseTime, drawEvents[]

    /draw/1/stryktipset/draws/{draw}
        draw.drawEvents[]: eventNumber, eventDescription,
            match.participants[{type: home|away, name}], match.league.name,
            odds{one,x,two}            -- aktuella odds (svenskt decimalkomma)
            startOdds{one,x,two}       -- startodds
            favouriteOdds{one,x,two}   -- favoritskap i procent
            svenskaFolket{one,x,two}   -- aktuella streck i procent
            (refOne/refX/refTwo ar referensvarden och anvands INTE)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests
from sqlalchemy import func, select
from sqlalchemy.engine import Engine

from archive.db import (
    MATCH_COUNT,
    PARSER_VERSION,
    PRECISION_EXACT,
    ROUND_STATUS_CLOSED,
    ROUND_STATUS_FINALIZED,
    ROUND_STATUS_OPEN,
    SNAPSHOT_SOURCES,
    SOURCE_API,
    SOURCE_LEGACY_IMPORT,
    market_snapshot_matches,
    market_snapshots,
    played_system_matches,
    played_systems,
    result_matches,
    results,
    round_matches,
    rounds,
    session_scope,
)
from svenskaspel_results import (
    ResultFetchError,
    FetchedResult,
    fetch_result_payload,
    parse_amount,
    parse_result_payload,
)

logger = logging.getLogger(__name__)

API_BASE = "https://api.spela.svenskaspel.se/draw/1/stryktipset"
DRAWS_ENDPOINT = f"{API_BASE}/draws"
DRAW_ENDPOINT = f"{API_BASE}/draws/{{draw}}"

USER_AGENT = (
    "fotbollspredictor_v7 (private analysis tool; "
    "contact: emtatos@gmail.com)"
)
REQUEST_TIMEOUT_SECONDS = 10.0

VALID_SIGNS = ("1", "X", "2")
VALID_PLAYED_SIGNS = ("1", "X", "2", "1X", "12", "X2", "1X2")

_WEEK_RE = re.compile(r"v\.?\s*(?:\d{4}-)?(\d{1,2})", re.IGNORECASE)


class ArchiveFetchError(Exception):
    """Hamtning eller tolkning misslyckades. Inget har skrivits."""


class ResultConflictError(Exception):
    """Ett annat resultat finns redan for omgangen. Inget skrivs over."""


class IncompleteRowError(ValueError):
    """Spelad rad ar inte en komplett 13-matchersrad."""


class SnapshotRequiredError(ValueError):
    """`snapshot_id` saknas for en kalla dar det kravs."""


# ---------------------------------------------------------------------------
# Dataklasser
# ---------------------------------------------------------------------------

@dataclass
class RoundMatch:
    position: int
    home_team: str
    away_team: str
    league: Optional[str] = None


@dataclass
class Round:
    draw_number: int
    week_label: Optional[str]
    reg_close_time: Optional[datetime]
    status: str
    matches: List[RoundMatch] = field(default_factory=list)


@dataclass
class SnapshotRow:
    """En matchrad i ett snapshot (streck i procent, decimalodds)."""
    position: int
    streck_1: Optional[float] = None
    streck_x: Optional[float] = None
    streck_2: Optional[float] = None
    odds_1: Optional[float] = None
    odds_x: Optional[float] = None
    odds_2: Optional[float] = None
    startodds_1: Optional[float] = None
    startodds_x: Optional[float] = None
    startodds_2: Optional[float] = None
    favoritskap_1: Optional[float] = None
    favoritskap_x: Optional[float] = None
    favoritskap_2: Optional[float] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    league: Optional[str] = None


@dataclass
class PlayedRow:
    """En match i en spelad rad, som Flera Matcher byggt den."""
    position: int
    played_signs: str
    is_halfguard: bool
    combined_p1: Optional[float] = None
    combined_px: Optional[float] = None
    combined_p2: Optional[float] = None
    gain: Optional[float] = None
    sources_used: Optional[str] = None


# ---------------------------------------------------------------------------
# Hjalpare
# ---------------------------------------------------------------------------

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_timestamp(value: Any) -> Optional[datetime]:
    """ISO-8601 (inkl. `Z` och offset) -> tz-medveten datetime i UTC."""
    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            moment = datetime.fromisoformat(text)
        except ValueError:
            return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def week_label_from_comment(comment: Any) -> Optional[str]:
    """`"Stryktipset v. 2026-38"` -> `"v38"`."""
    if not comment:
        return None
    found = _WEEK_RE.search(str(comment))
    if not found:
        return None
    return f"v{int(found.group(1))}"


def _canon(name: str) -> str:
    from utils import normalize_team_name
    try:
        return normalize_team_name(name)
    except Exception:  # noqa: BLE001 -- normalisering far aldrig stoppa arkivet
        return name.strip()


def _get(url: str, *, timeout: float) -> Dict[str, Any]:
    """Exakt ett GET-anrop. Kastar ArchiveFetchError vid varje fel."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
    except requests.Timeout as exc:
        logger.error("Timeout mot %s: %s", url, exc)
        raise ArchiveFetchError(
            f"Anropet tog for lang tid (timeout {timeout:g} s). "
            "Inget omforsok gors automatiskt."
        ) from exc
    except requests.RequestException as exc:
        logger.error("Natverksfel mot %s: %s", url, exc)
        raise ArchiveFetchError(f"Natverksfel vid hamtning: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise ArchiveFetchError(
            f"Svaret kunde inte tolkas som JSON: {exc}"
        ) from exc
    if response.status_code != 200:
        raise ArchiveFetchError(
            f"Ovantad HTTP-status {response.status_code} fran {url}."
        )
    if not isinstance(payload, dict):
        raise ArchiveFetchError("Svaret var inte ett JSON-objekt.")
    error = payload.get("error")
    if error:
        code = error.get("code") if isinstance(error, dict) else error
        raise ArchiveFetchError(f"API:t svarade med fel (code={code}).")
    return payload


def _event_teams(event: Dict[str, Any]) -> tuple:
    """(home, away, league) fran ett drawEvent."""
    match = event.get("match") if isinstance(event.get("match"), dict) else {}
    home = away = ""
    for participant in match.get("participants") or []:
        if not isinstance(participant, dict):
            continue
        name = str(participant.get("name") or "").strip()
        if participant.get("type") == "home":
            home = name
        elif participant.get("type") == "away":
            away = name
    if not home or not away:
        description = str(event.get("eventDescription") or "")
        if " - " in description:
            left, right = description.split(" - ", 1)
            home = home or left.strip()
            away = away or right.strip()
    league = match.get("league") if isinstance(match.get("league"), dict) else {}
    league_name = league.get("name") if league else None
    return home, away, (str(league_name).strip() if league_name else None)


def _parse_events(events: Any) -> List[RoundMatch]:
    if not isinstance(events, list) or len(events) != MATCH_COUNT:
        count = len(events) if isinstance(events, list) else 0
        raise ArchiveFetchError(
            f"Svaret innehaller {count} matcher, forvantat {MATCH_COUNT}."
        )
    ordered = sorted(
        (e for e in events if isinstance(e, dict)),
        key=lambda e: int(e.get("eventNumber") or 0),
    )
    matches: List[RoundMatch] = []
    for index, event in enumerate(ordered, start=1):
        position = int(event.get("eventNumber") or index)
        home, away, league = _event_teams(event)
        if not home or not away:
            raise ArchiveFetchError(
                f"Match {position} saknar lagnamn i svaret."
            )
        matches.append(RoundMatch(position, home, away, league))
    return matches


def _triple(block: Any) -> tuple:
    if not isinstance(block, dict):
        return None, None, None
    return (
        parse_amount(block.get("one")),
        parse_amount(block.get("x")),
        parse_amount(block.get("two")),
    )


def parse_snapshot_rows(draw_payload: Dict[str, Any]) -> List[SnapshotRow]:
    """Tolkar `/draws/{draw}`-svaret till 13 snapshotrader."""
    draw = draw_payload.get("draw")
    if not isinstance(draw, dict):
        raise ArchiveFetchError("Svaret innehaller ingen draw-post.")
    matches = _parse_events(draw.get("drawEvents"))
    events = sorted(
        (e for e in draw.get("drawEvents") if isinstance(e, dict)),
        key=lambda e: int(e.get("eventNumber") or 0),
    )
    rows: List[SnapshotRow] = []
    for match, event in zip(matches, events):
        s1, sx, s2 = _triple(event.get("svenskaFolket"))
        o1, ox, o2 = _triple(event.get("odds"))
        so1, sox, so2 = _triple(event.get("startOdds"))
        f1, fx, f2 = _triple(event.get("favouriteOdds"))
        rows.append(SnapshotRow(
            position=match.position,
            streck_1=s1, streck_x=sx, streck_2=s2,
            odds_1=o1, odds_x=ox, odds_2=o2,
            startodds_1=so1, startodds_x=sox, startodds_2=so2,
            favoritskap_1=f1, favoritskap_x=fx, favoritskap_2=f2,
            home_team=match.home_team,
            away_team=match.away_team,
            league=match.league,
        ))
    return rows


def _round_from_draw(draw: Dict[str, Any]) -> Round:
    draw_number = draw.get("drawNumber")
    if draw_number is None:
        raise ArchiveFetchError("Svaret saknar drawNumber.")
    state = str(draw.get("drawState") or "").lower()
    if state == "open":
        status = ROUND_STATUS_OPEN
    elif state in ("finalized", "final", "finished"):
        status = ROUND_STATUS_FINALIZED
    else:
        status = ROUND_STATUS_CLOSED
    return Round(
        draw_number=int(draw_number),
        week_label=week_label_from_comment(draw.get("drawComment")),
        reg_close_time=parse_timestamp(draw.get("regCloseTime")),
        status=status,
        matches=_parse_events(draw.get("drawEvents"))
        if draw.get("drawEvents") else [],
    )


# ---------------------------------------------------------------------------
# Skrivning: rounds / round_matches (upsert, raderar aldrig)
# ---------------------------------------------------------------------------

def _upsert_round(session, rnd: Round) -> None:
    existing = session.execute(
        select(rounds.c.draw_number).where(
            rounds.c.draw_number == rnd.draw_number
        )
    ).first()
    values = {
        "week_label": rnd.week_label,
        "reg_close_time": rnd.reg_close_time,
        "status": rnd.status,
    }
    if existing is None:
        session.execute(rounds.insert().values(
            draw_number=rnd.draw_number, created_at=now_utc(), **values,
        ))
    else:
        session.execute(
            rounds.update()
            .where(rounds.c.draw_number == rnd.draw_number)
            .values(**{k: v for k, v in values.items() if v is not None})
        )
    for match in rnd.matches:
        row = {
            "home_team": match.home_team,
            "away_team": match.away_team,
            "home_team_canon": _canon(match.home_team),
            "away_team_canon": _canon(match.away_team),
            "league": match.league,
        }
        present = session.execute(
            select(round_matches.c.position).where(
                round_matches.c.draw_number == rnd.draw_number,
                round_matches.c.position == match.position,
            )
        ).first()
        if present is None:
            session.execute(round_matches.insert().values(
                draw_number=rnd.draw_number, position=match.position, **row,
            ))
        else:
            session.execute(
                round_matches.update()
                .where(
                    round_matches.c.draw_number == rnd.draw_number,
                    round_matches.c.position == match.position,
                )
                .values(**row)
            )


def ensure_round(
    draw_number: int,
    *,
    engine: Optional[Engine] = None,
    week_label: Optional[str] = None,
    reg_close_time: Optional[datetime] = None,
    status: str = ROUND_STATUS_CLOSED,
    matches: Sequence[RoundMatch] = (),
) -> None:
    """Skapar `rounds`-raden om den saknas (for import/manuella kallor)."""
    with session_scope(engine) as session:
        _upsert_round(session, Round(
            draw_number=int(draw_number),
            week_label=week_label,
            reg_close_time=reg_close_time,
            status=status,
            matches=list(matches),
        ))


# ---------------------------------------------------------------------------
# Publikt API
# ---------------------------------------------------------------------------

def fetch_current_round(
    *,
    engine: Optional[Engine] = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> Optional[Round]:
    """
    Hamtar `/draws`, hittar den oppna omgangen och upsertar `rounds` +
    `round_matches`. Returnerar None (med loggning) om ingen omgang ar oppen.
    """
    payload = _get(DRAWS_ENDPOINT, timeout=timeout)
    draws = payload.get("draws")
    if not isinstance(draws, list) or not draws:
        logger.info("Ingen oppen omgang: /draws gav tom lista.")
        return None

    open_draws = [
        d for d in draws
        if isinstance(d, dict)
        and str(d.get("drawState") or "").lower() == "open"
    ]
    if not open_draws:
        logger.info(
            "Ingen oppen omgang: drawState=%s.",
            [d.get("drawState") for d in draws if isinstance(d, dict)],
        )
        return None
    chosen = min(open_draws, key=lambda d: int(d.get("drawNumber") or 0))

    if not chosen.get("drawEvents"):
        detail = _get(
            DRAW_ENDPOINT.format(draw=int(chosen["drawNumber"])),
            timeout=timeout,
        )
        chosen = detail.get("draw") if isinstance(detail.get("draw"), dict) else chosen

    rnd = _round_from_draw(chosen)
    with session_scope(engine) as session:
        _upsert_round(session, rnd)
    logger.info(
        "Omgang %s (%s) identifierad, spelstopp %s.",
        rnd.draw_number, rnd.week_label, rnd.reg_close_time,
    )
    return rnd


def save_snapshot(
    draw_number: int,
    rows: Iterable[SnapshotRow],
    *,
    source: str,
    captured_at: Optional[datetime] = None,
    captured_at_precision: str = PRECISION_EXACT,
    raw_payload: Optional[Dict[str, Any]] = None,
    parser_version: int = PARSER_VERSION,
    engine: Optional[Engine] = None,
) -> int:
    """
    Skriver ETT nytt snapshot (header + matchrader). Skriver aldrig over.
    Returnerar headerns id. Saknas `rounds`-raden skapas den.
    """
    if source not in SNAPSHOT_SOURCES:
        raise ValueError(
            f"source maste vara en av {SNAPSHOT_SOURCES}, fick {source!r}"
        )
    row_list = list(rows)
    if not row_list:
        raise ValueError("Snapshot maste innehalla minst en match.")
    moment = captured_at or now_utc()
    with session_scope(engine) as session:
        present = session.execute(
            select(rounds.c.draw_number).where(
                rounds.c.draw_number == int(draw_number)
            )
        ).first()
        if present is None:
            round_rows = [
                RoundMatch(r.position, r.home_team, r.away_team, r.league)
                for r in row_list if r.home_team and r.away_team
            ]
            _upsert_round(session, Round(
                draw_number=int(draw_number), week_label=None,
                reg_close_time=None, status=ROUND_STATUS_CLOSED,
                matches=round_rows,
            ))
        result = session.execute(market_snapshots.insert().values(
            draw_number=int(draw_number),
            captured_at=moment,
            captured_at_precision=captured_at_precision,
            source=source,
            parser_version=parser_version,
            raw_payload=raw_payload,
        ))
        snapshot_id = int(result.inserted_primary_key[0])
        session.execute(market_snapshot_matches.insert(), [
            {
                "snapshot_id": snapshot_id,
                "position": int(r.position),
                "streck_1": r.streck_1, "streck_x": r.streck_x,
                "streck_2": r.streck_2,
                "odds_1": r.odds_1, "odds_x": r.odds_x, "odds_2": r.odds_2,
                "startodds_1": r.startodds_1, "startodds_x": r.startodds_x,
                "startodds_2": r.startodds_2,
                "favoritskap_1": r.favoritskap_1,
                "favoritskap_x": r.favoritskap_x,
                "favoritskap_2": r.favoritskap_2,
            }
            for r in row_list
        ])
    logger.info(
        "Snapshot %s sparat for omgang %s (%d rader, source=%s).",
        snapshot_id, draw_number, len(row_list), source,
    )
    return snapshot_id


def capture_market_snapshot(
    draw_number: int,
    *,
    engine: Optional[Engine] = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> int:
    """
    Hamtar `/draws/{draw}` och sparar aktuella streck/odds som ETT nytt
    snapshot med source=api. Varje anrop ar en ny observation.
    """
    payload = _get(DRAW_ENDPOINT.format(draw=int(draw_number)), timeout=timeout)
    draw = payload.get("draw")
    if not isinstance(draw, dict):
        raise ArchiveFetchError(
            f"Omgang {draw_number} finns inte (draw saknas i svaret)."
        )
    rows = parse_snapshot_rows(payload)
    rnd = _round_from_draw(draw)
    with session_scope(engine) as session:
        _upsert_round(session, rnd)
    return save_snapshot(
        rnd.draw_number, rows,
        source=SOURCE_API,
        captured_at=now_utc(),
        captured_at_precision=PRECISION_EXACT,
        raw_payload=payload,
        engine=engine,
    )


def _result_values(fetched: FetchedResult, *, source: str) -> Dict[str, Any]:
    def _int(value: Optional[float]) -> Optional[int]:
        return None if value is None else int(value)
    return {
        "correct_row": "".join(fetched.correct_row),
        "turnover": fetched.turnover,
        "payout_13": fetched.payouts.get("13"),
        "payout_12": fetched.payouts.get("12"),
        "payout_11": fetched.payouts.get("11"),
        "payout_10": fetched.payouts.get("10"),
        "winners_13": _int(fetched.winners.get("13")),
        "winners_12": _int(fetched.winners.get("12")),
        "winners_11": _int(fetched.winners.get("11")),
        "winners_10": _int(fetched.winners.get("10")),
        "source": source,
    }


_COMPARED_RESULT_FIELDS = (
    "correct_row", "turnover", "payout_13", "payout_12", "payout_11",
    "payout_10", "winners_13", "winners_12", "winners_11", "winners_10",
)


def save_result(
    fetched: FetchedResult,
    *,
    source: str,
    raw_payload: Optional[Dict[str, Any]] = None,
    fetched_at: Optional[datetime] = None,
    parser_version: int = PARSER_VERSION,
    engine: Optional[Engine] = None,
) -> bool:
    """
    Skriver `results` + `result_matches`.

    Finns identiskt resultat -> returnerar False utan att skriva. Finns ett
    resultat som skiljer sig -> ResultConflictError, inget skrivs over.
    Returnerar True nar ett nytt resultat skrevs.
    """
    if not fetched.is_complete:
        raise ResultFetchError(
            f"Resultatet ar inte komplett ({MATCH_COUNT} giltiga utfall "
            "kravs); det far inte sparas."
        )
    values = _result_values(fetched, source=source)
    with session_scope(engine) as session:
        existing = session.execute(
            select(results).where(results.c.draw_number == fetched.draw)
        ).mappings().first()
        if existing is not None:
            diffs = {
                key: (existing[key], values[key])
                for key in _COMPARED_RESULT_FIELDS
                if existing[key] != values[key]
            }
            if diffs:
                logger.error(
                    "Resultatkonflikt for omgang %s: %s", fetched.draw, diffs,
                )
                raise ResultConflictError(
                    f"Omgang {fetched.draw} har redan ett resultat som "
                    f"skiljer sig: {diffs}. Inget skrivs over."
                )
            logger.info(
                "Identiskt resultat finns redan for omgang %s.", fetched.draw,
            )
            return False

        present = session.execute(
            select(rounds.c.draw_number).where(
                rounds.c.draw_number == fetched.draw
            )
        ).first()
        if present is None:
            _upsert_round(session, Round(
                draw_number=fetched.draw, week_label=None,
                reg_close_time=parse_timestamp(fetched.reg_close_time),
                status=ROUND_STATUS_FINALIZED, matches=[],
            ))
        else:
            session.execute(
                rounds.update()
                .where(rounds.c.draw_number == fetched.draw)
                .values(status=ROUND_STATUS_FINALIZED)
            )
        session.execute(results.insert().values(
            draw_number=fetched.draw,
            fetched_at=fetched_at or now_utc(),
            parser_version=parser_version,
            raw_payload=raw_payload,
            **values,
        ))
        session.execute(result_matches.insert(), [
            {
                "draw_number": fetched.draw,
                "position": match.position,
                "home_score": match.home_goals,
                "away_score": match.away_goals,
                "outcome": match.sign,
            }
            for match in fetched.matches
        ])
    logger.info("Resultat sparat for omgang %s.", fetched.draw)
    return True


def fetch_result(
    draw_number: int,
    *,
    engine: Optional[Engine] = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> None:
    """
    Hamtar `/draws/{draw}/result` (logik fran PR #50) och skriver till
    `results`/`result_matches`. Identiskt befintligt resultat -> ingen
    skrivning. Avvikande befintligt resultat -> ResultConflictError.
    """
    payload = fetch_result_payload(int(draw_number), timeout=timeout)
    fetched = parse_result_payload(payload)
    save_result(
        fetched, source=SOURCE_API, raw_payload=payload, engine=engine,
    )


def _normalize_played_rows(rows: Iterable[Any]) -> List[PlayedRow]:
    normalized: List[PlayedRow] = []
    for index, item in enumerate(rows, start=1):
        if isinstance(item, PlayedRow):
            row = item
        elif isinstance(item, dict):
            row = PlayedRow(
                position=int(item.get("position") or index),
                played_signs=str(item.get("played_signs") or ""),
                is_halfguard=bool(item.get("is_halfguard")),
                combined_p1=item.get("combined_p1"),
                combined_px=item.get("combined_px"),
                combined_p2=item.get("combined_p2"),
                gain=item.get("gain"),
                sources_used=item.get("sources_used"),
            )
        else:
            raise TypeError(f"Ogiltig radtyp: {type(item).__name__}")
        signs = row.played_signs.strip().upper()
        ordered = "".join(sign for sign in VALID_SIGNS if sign in signs)
        if not signs or any(ch not in VALID_SIGNS for ch in signs) \
                or ordered not in VALID_PLAYED_SIGNS:
            raise ValueError(
                f"Match {row.position}: ogiltiga tecken {row.played_signs!r}."
            )
        row.played_signs = ordered
        normalized.append(row)
    validate_complete_row(normalized)
    return sorted(normalized, key=lambda r: r.position)


def validate_complete_row(rows: List[PlayedRow]) -> None:
    """
    En Stryktipsrad ar exakt 13 matcher med unika positioner 1-13. Ofullstandiga
    rader (saknade tips, luckor, dubbletter) hor inte hemma i arkivet.
    """
    if len(rows) != MATCH_COUNT:
        raise IncompleteRowError(
            f"Raden maste ha exakt {MATCH_COUNT} matcher, fick {len(rows)}."
        )
    positions = [int(r.position) for r in rows]
    duplicates = sorted({p for p in positions if positions.count(p) > 1})
    if duplicates:
        raise IncompleteRowError(f"Dubblettpositioner i raden: {duplicates}.")
    missing = sorted(set(range(1, MATCH_COUNT + 1)) - set(positions))
    if missing:
        raise IncompleteRowError(
            f"Raden saknar match {missing} "
            f"(positioner maste vara 1-{MATCH_COUNT})."
        )


def register_played_system(
    draw_number: int,
    rows: Iterable[Any],
    snapshot_id: Optional[int],
    *,
    source: str = SOURCE_API,
    note: str = "",
    engine: Optional[Engine] = None,
) -> int:
    """
    Sparar raden fran Flera Matcher i `played_systems` +
    `played_system_matches`.

    `snapshot_id` maste peka pa den observation raden byggdes fran. NULL
    tillats ENDAST for source=legacy_import. Finns exakt samma system redan
    (samma draw, samma tecken per match) returneras befintligt id.
    """
    if snapshot_id is None and source != SOURCE_LEGACY_IMPORT:
        raise SnapshotRequiredError(
            "snapshot_id kravs: en spelad rad utan kant snapshot kan inte "
            f"analyseras (source={source!r})."
        )
    played = _normalize_played_rows(rows)
    signature = {row.position: row.played_signs for row in played}
    n_halfguards = sum(1 for row in played if row.is_halfguard)

    with session_scope(engine) as session:
        if snapshot_id is not None:
            snap = session.execute(
                select(market_snapshots.c.draw_number).where(
                    market_snapshots.c.id == int(snapshot_id)
                )
            ).first()
            if snap is None:
                raise ValueError(f"Snapshot {snapshot_id} finns inte.")
            if int(snap[0]) != int(draw_number):
                raise ValueError(
                    f"Snapshot {snapshot_id} tillhor omgang {snap[0]}, "
                    f"inte {draw_number}."
                )
            snap_rows = session.execute(
                select(func.count()).select_from(market_snapshot_matches).where(
                    market_snapshot_matches.c.snapshot_id == int(snapshot_id)
                )
            ).scalar()
            if int(snap_rows or 0) != MATCH_COUNT:
                raise IncompleteRowError(
                    f"Snapshot {snapshot_id} har {snap_rows} matchrader, "
                    f"kravs {MATCH_COUNT}."
                )

        existing_ids = session.execute(
            select(played_systems.c.id).where(
                played_systems.c.draw_number == int(draw_number)
            )
        ).scalars().all()
        for system_id in existing_ids:
            existing_rows = session.execute(
                select(
                    played_system_matches.c.position,
                    played_system_matches.c.played_signs,
                ).where(played_system_matches.c.played_system_id == system_id)
            ).all()
            if {int(p): s for p, s in existing_rows} == signature:
                logger.info(
                    "Identiskt system finns redan (id=%s) for omgang %s.",
                    system_id, draw_number,
                )
                return int(system_id)

        present = session.execute(
            select(rounds.c.draw_number).where(
                rounds.c.draw_number == int(draw_number)
            )
        ).first()
        if present is None:
            _upsert_round(session, Round(
                draw_number=int(draw_number), week_label=None,
                reg_close_time=None, status=ROUND_STATUS_CLOSED, matches=[],
            ))

        inserted = session.execute(played_systems.insert().values(
            draw_number=int(draw_number),
            created_at=now_utc(),
            snapshot_id=snapshot_id,
            n_halfguards=n_halfguards,
            note=note or "",
        ))
        system_id = int(inserted.inserted_primary_key[0])
        session.execute(played_system_matches.insert(), [
            {
                "played_system_id": system_id,
                "position": int(row.position),
                "played_signs": row.played_signs,
                "is_halfguard": bool(row.is_halfguard),
                "combined_p1": row.combined_p1,
                "combined_px": row.combined_px,
                "combined_p2": row.combined_p2,
                "gain": row.gain,
                "sources_used": row.sources_used,
            }
            for row in played
        ])
    logger.info(
        "Spelad rad %s registrerad for omgang %s (%d halvgarderingar, "
        "snapshot=%s).",
        system_id, draw_number, n_halfguards, snapshot_id,
    )
    return system_id
