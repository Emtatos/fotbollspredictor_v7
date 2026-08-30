"""
snapshot_storage.py -- Append-only arkiv for kupongomgangar.

Varje importerad eller scannad kupong kan sparas som ett tidsstamplat
snapshot. Snapshots skrivs aldrig over: streck och odds som de sag ut fore
spelstopp kan inte aterskapas i efterhand.

Lagringsformat:
    data/snapshots/<draw>_<captured_at>.json   (append-only)
    data/results/<draw>.json                   (efterhandsdata, separat fil)

Efterhandsdata (radresultat, utdelning) skrivs ALDRIG in i snapshot-filen.
Det skulle bryta append-only-garantin.

OBS: appen kors pa Render dar filsystemet nollstalls vid omstart och deploy.
Diskskrivning ar darfor bara beständig vid lokal korning -- anvand
`snapshot_json()` for att erbjuda nedladdning som primar lagringsvag.

Anvandning:
    from snapshot_storage import (
        build_snapshot,
        save_snapshot,
        snapshot_json,
        list_snapshots,
        save_result,
    )

    snapshot = build_snapshot(matches, draw=4966, source="image_scan")
    payload = snapshot_json(snapshot)      # identisk med filens innehall
    path = save_snapshot(snapshot)         # skriver utan att skriva over
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanter
# ---------------------------------------------------------------------------

DEFAULT_SNAPSHOT_DIR = Path("data") / "snapshots"
DEFAULT_RESULTS_DIR = Path("data") / "results"

SNAPSHOT_VERSION = 1

# captured_at_precision ar obligatoriskt och beskriver hur exakt tidpunkten
# ar kand. En framtida backtest ska kunna filtrera pa hur langt fore
# spelstopp observationen gjordes; snapshots utan exakt tidpunkt far inte
# tyst blandas in i den analysen.
PRECISION_EXACT = "exact"
PRECISION_DATE_ONLY = "date_only"
PRECISION_UNKNOWN = "unknown"
CAPTURED_AT_PRECISIONS = (
    PRECISION_EXACT,
    PRECISION_DATE_ONLY,
    PRECISION_UNKNOWN,
)

SOURCE_IMAGE_SCAN = "image_scan"
SOURCE_PASTE = "paste"
SOURCE_MANUAL = "manual"
SNAPSHOT_SOURCES = (SOURCE_IMAGE_SCAN, SOURCE_PASTE, SOURCE_MANUAL)

VALID_SIGNS = ("1", "X", "2")
PAYOUT_TIERS = ("13", "12", "11", "10")

# Kallor for efterhandsdata. Aldre resultatfiler saknar faltet och lases som
# manuellt inmatade.
RESULT_SOURCE_MANUAL = "manual"
RESULT_SOURCE_API = "svenskaspel_api"
RESULT_SOURCES = (RESULT_SOURCE_MANUAL, RESULT_SOURCE_API)

_TIMESTAMP_UNSAFE = re.compile(r"[^0-9A-Za-z]")


# ---------------------------------------------------------------------------
# Dataklasser
# ---------------------------------------------------------------------------

@dataclass
class SnapshotMatch:
    """En match i ett snapshot, med streck och odds som de sag ut."""
    position: int
    home_team: str
    away_team: str
    league: Optional[str] = None
    streck_1: Optional[float] = None
    streck_x: Optional[float] = None
    streck_2: Optional[float] = None
    odds_1: Optional[float] = None
    odds_x: Optional[float] = None
    odds_2: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-kompatibel representation."""
        return {
            "position": int(self.position),
            "home_team": self.home_team,
            "away_team": self.away_team,
            "league": self.league,
            "streck_1": self.streck_1,
            "streck_x": self.streck_x,
            "streck_2": self.streck_2,
            "odds_1": self.odds_1,
            "odds_x": self.odds_x,
            "odds_2": self.odds_2,
        }


@dataclass
class Snapshot:
    """Ett tidsstamplat snapshot av en kupongomgang."""
    draw: Optional[int]
    captured_at: str
    captured_at_precision: str
    source: str
    matches: List[SnapshotMatch]
    reg_close_time: Optional[str] = None
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """JSON-kompatibel representation."""
        return {
            "version": SNAPSHOT_VERSION,
            "draw": self.draw,
            "captured_at": self.captured_at,
            "captured_at_precision": self.captured_at_precision,
            "reg_close_time": self.reg_close_time,
            "source": self.source,
            "note": self.note,
            "matches": [match.to_dict() for match in self.matches],
        }


@dataclass
class SnapshotInfo:
    """Lattviktig beskrivning av en sparad snapshot-fil."""
    path: Path
    draw: Optional[int] = None
    captured_at: str = ""
    captured_at_precision: str = ""
    source: str = ""
    note: str = ""
    match_count: int = 0
    readable: bool = True


@dataclass
class RoundResult:
    """Efterhandsdata for en omgang. Skrivs i egen fil, aldrig i snapshot."""
    draw: int
    correct_row: List[str] = field(default_factory=list)
    turnover: Optional[float] = None
    payouts: Dict[str, Optional[float]] = field(default_factory=dict)
    winners: Dict[str, Optional[float]] = field(default_factory=dict)
    entered_at: str = ""
    entered_manually: bool = True
    source: str = RESULT_SOURCE_MANUAL
    draw_state: Optional[str] = None
    reg_close_time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-kompatibel representation."""
        return {
            "draw": int(self.draw),
            "correct_row": list(self.correct_row),
            "turnover": self.turnover,
            "payouts": {
                tier: self.payouts.get(tier) for tier in PAYOUT_TIERS
            },
            "winners": {
                tier: self.winners.get(tier) for tier in PAYOUT_TIERS
            },
            "entered_at": self.entered_at,
            "entered_manually": bool(self.entered_manually),
            "source": self.source,
            "draw_state": self.draw_state,
            "reg_close_time": self.reg_close_time,
        }


# ---------------------------------------------------------------------------
# Intern: hjalpare
# ---------------------------------------------------------------------------

def _optional_float(value: Any) -> Optional[float]:
    """Konverterar till float, eller None for tomma/ogiltiga varden."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().replace(",", ".")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def _optional_int(value: Any) -> Optional[int]:
    """Konverterar till int, eller None for tomma/ogiltiga varden."""
    number = _optional_float(value)
    if number is None:
        return None
    return int(number)


def utc_timestamp(moment: Optional[datetime] = None) -> str:
    """Aktuell tidpunkt som `YYYY-MM-DDTHH:MM:SSZ`."""
    if moment is None:
        moment = datetime.now(timezone.utc)
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def date_only_timestamp(day: Any) -> str:
    """Datum utan klockslag som `YYYY-MM-DDT00:00:00Z`."""
    if isinstance(day, datetime):
        return day.strftime("%Y-%m-%dT00:00:00Z")
    if hasattr(day, "strftime"):
        return day.strftime("%Y-%m-%dT00:00:00Z")
    text = str(day).strip()[:10]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise ValueError(f"Ogiltigt datum: {day!r}")
    return f"{text}T00:00:00Z"


# ---------------------------------------------------------------------------
# Publikt API: bygga snapshots
# ---------------------------------------------------------------------------

def build_snapshot(
    matches: Iterable[Any],
    *,
    source: str,
    captured_at_precision: str = PRECISION_EXACT,
    draw: Optional[int] = None,
    captured_at: Optional[str] = None,
    reg_close_time: Optional[str] = None,
    note: str = "",
) -> Snapshot:
    """
    Bygger ett snapshot av matchrader.

    `matches` kan vara SnapshotMatch-objekt eller dicts med nycklarna
    home_team/away_team och valfria streck_*/odds_*/league/position.

    Kastar ValueError vid ogiltig `captured_at_precision`, ogiltig `source`
    eller tom matchlista.
    """
    if captured_at_precision not in CAPTURED_AT_PRECISIONS:
        raise ValueError(
            "captured_at_precision maste vara en av "
            f"{CAPTURED_AT_PRECISIONS}, fick {captured_at_precision!r}"
        )
    if source not in SNAPSHOT_SOURCES:
        raise ValueError(
            f"source maste vara en av {SNAPSHOT_SOURCES}, fick {source!r}"
        )

    normalized = normalize_matches(matches)
    if not normalized:
        raise ValueError("Snapshot maste innehalla minst en match.")

    return Snapshot(
        draw=_optional_int(draw),
        captured_at=captured_at or utc_timestamp(),
        captured_at_precision=captured_at_precision,
        source=source,
        matches=normalized,
        reg_close_time=reg_close_time,
        note=note,
    )


def normalize_matches(matches: Iterable[Any]) -> List[SnapshotMatch]:
    """Konverterar matchrader till SnapshotMatch med lopande position."""
    normalized: List[SnapshotMatch] = []
    for index, item in enumerate(matches, start=1):
        if isinstance(item, SnapshotMatch):
            match = SnapshotMatch(
                position=index,
                home_team=item.home_team,
                away_team=item.away_team,
                league=item.league,
                streck_1=item.streck_1,
                streck_x=item.streck_x,
                streck_2=item.streck_2,
                odds_1=item.odds_1,
                odds_x=item.odds_x,
                odds_2=item.odds_2,
            )
        elif isinstance(item, dict):
            home = str(item.get("home_team", "")).strip()
            away = str(item.get("away_team", "")).strip()
            league = item.get("league")
            match = SnapshotMatch(
                position=index,
                home_team=home,
                away_team=away,
                league=str(league).strip() if league else None,
                streck_1=_optional_float(item.get("streck_1")),
                streck_x=_optional_float(item.get("streck_x")),
                streck_2=_optional_float(item.get("streck_2")),
                odds_1=_optional_float(item.get("odds_1")),
                odds_x=_optional_float(item.get("odds_x")),
                odds_2=_optional_float(item.get("odds_2")),
            )
        else:
            match = SnapshotMatch(
                position=index,
                home_team=str(getattr(item, "home_team", "")).strip(),
                away_team=str(getattr(item, "away_team", "")).strip(),
                league=getattr(item, "league", None),
                streck_1=_optional_float(getattr(item, "streck_1", None)),
                streck_x=_optional_float(getattr(item, "streck_x", None)),
                streck_2=_optional_float(getattr(item, "streck_2", None)),
                odds_1=_optional_float(getattr(item, "odds_1", None)),
                odds_x=_optional_float(getattr(item, "odds_x", None)),
                odds_2=_optional_float(getattr(item, "odds_2", None)),
            )

        if not match.home_team or not match.away_team:
            continue
        normalized.append(match)

    # Positionerna ska vara lopande aven om tomma rader hoppats over.
    for position, match in enumerate(normalized, start=1):
        match.position = position
    return normalized


def matches_from_dataframe(df: Any) -> List[SnapshotMatch]:
    """
    Konverterar kontrolltabellen i UI:t till SnapshotMatch-rader.

    Forvantar kolumnerna HomeTeam, AwayTeam, Streck1/StreckX/Streck2 och
    Odds1/OddsX/Odds2 (kupongbildens format).
    """
    if df is None or len(df) == 0:
        return []

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rows.append({
            "home_team": row.get("HomeTeam", ""),
            "away_team": row.get("AwayTeam", ""),
            "league": row.get("League"),
            "streck_1": row.get("Streck1"),
            "streck_x": row.get("StreckX"),
            "streck_2": row.get("Streck2"),
            "odds_1": row.get("Odds1"),
            "odds_x": row.get("OddsX"),
            "odds_2": row.get("Odds2"),
        })
    return normalize_matches(rows)


def matches_from_matchday_matches(matchday_matches: Iterable[Any]) -> List[
    SnapshotMatch
]:
    """
    Konverterar MatchdayMatch-objekt till SnapshotMatch-rader.

    Odds tas fran forsta oddsposten (den bookmaker som importen listar
    forst); snapshotet lagrar radens odds, inte konsensusberakningar.
    """
    rows: List[Dict[str, Any]] = []
    for item in matchday_matches:
        entries = list(getattr(item, "odds_entries", []) or [])
        first = entries[0] if entries else None
        streck = getattr(item, "streck", None) or {}
        rows.append({
            "home_team": getattr(item, "home_team", ""),
            "away_team": getattr(item, "away_team", ""),
            "streck_1": streck.get("1"),
            "streck_x": streck.get("X"),
            "streck_2": streck.get("2"),
            "odds_1": getattr(first, "home", None),
            "odds_x": getattr(first, "draw", None),
            "odds_2": getattr(first, "away", None),
        })
    return normalize_matches(rows)


def snapshot_json(snapshot: Snapshot) -> str:
    """
    Snapshotens JSON-text.

    Samma strang anvands bade for nedladdningsknappen och for filen pa disk,
    sa att nedladdad kopia och sparad fil ar identiska.
    """
    return json.dumps(
        snapshot.to_dict(), ensure_ascii=False, indent=2,
    ) + "\n"


# ---------------------------------------------------------------------------
# Publikt API: filnamn och skrivning
# ---------------------------------------------------------------------------

def snapshot_filename(snapshot: Snapshot) -> str:
    """Filnamn `<draw>_<captured_at>.json`, `unknown_...` utan draw."""
    draw_part = "unknown" if snapshot.draw is None else str(snapshot.draw)
    stamp = _TIMESTAMP_UNSAFE.sub("-", snapshot.captured_at)
    return f"{draw_part}_{stamp}.json"


def _next_free_path(path: Path) -> Path:
    """Forsta lediga sokvagen: befintliga filer far aldrig skrivas over."""
    if not path.exists():
        return path
    suffix = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def save_snapshot(
    snapshot: Snapshot,
    directory: Optional[Path] = None,
) -> Path:
    """
    Skriver snapshotet till disk utan att nagon befintlig fil andras.

    Vid namnkollision far filen ett suffix. Skrivningen sker med lage "x"
    sa att en samtidig skrivare inte kan skriva over en befintlig fil.
    Returnerar sokvagen som skrevs.
    """
    target_dir = Path(directory) if directory is not None else (
        DEFAULT_SNAPSHOT_DIR
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    payload = snapshot_json(snapshot)
    candidate = _next_free_path(target_dir / snapshot_filename(snapshot))
    while True:
        try:
            with open(candidate, "x", encoding="utf-8") as handle:
                handle.write(payload)
            break
        except FileExistsError:
            candidate = _next_free_path(candidate)

    logger.info(
        "Snapshot sparat: %s (%d matcher, precision=%s).",
        candidate,
        len(snapshot.matches),
        snapshot.captured_at_precision,
    )
    return candidate


def load_snapshot(path: Path) -> Snapshot:
    """Laser ett sparat snapshot. Kastar ValueError vid ogiltigt format."""
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    precision = raw.get("captured_at_precision")
    if precision not in CAPTURED_AT_PRECISIONS:
        raise ValueError(
            f"Ogiltig captured_at_precision i {path}: {precision!r}"
        )

    matches = [
        SnapshotMatch(
            position=_optional_int(item.get("position")) or index,
            home_team=str(item.get("home_team", "")),
            away_team=str(item.get("away_team", "")),
            league=item.get("league"),
            streck_1=_optional_float(item.get("streck_1")),
            streck_x=_optional_float(item.get("streck_x")),
            streck_2=_optional_float(item.get("streck_2")),
            odds_1=_optional_float(item.get("odds_1")),
            odds_x=_optional_float(item.get("odds_x")),
            odds_2=_optional_float(item.get("odds_2")),
        )
        for index, item in enumerate(raw.get("matches", []), start=1)
    ]

    return Snapshot(
        draw=_optional_int(raw.get("draw")),
        captured_at=str(raw.get("captured_at", "")),
        captured_at_precision=precision,
        source=str(raw.get("source", "")),
        matches=matches,
        reg_close_time=raw.get("reg_close_time"),
        note=str(raw.get("note", "")),
    )


def list_snapshots(directory: Optional[Path] = None) -> List[SnapshotInfo]:
    """
    Beskriver sparade snapshots, nyast forst. Endast lasning.

    Trasiga filer listas med `readable=False` istallet for att kasta fel.
    """
    target_dir = Path(directory) if directory is not None else (
        DEFAULT_SNAPSHOT_DIR
    )
    if not target_dir.exists():
        return []

    infos: List[SnapshotInfo] = []
    for path in sorted(target_dir.glob("*.json")):
        try:
            snapshot = load_snapshot(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Kunde inte lasa snapshot %s: %s", path, exc)
            infos.append(SnapshotInfo(path=path, readable=False))
            continue
        infos.append(SnapshotInfo(
            path=path,
            draw=snapshot.draw,
            captured_at=snapshot.captured_at,
            captured_at_precision=snapshot.captured_at_precision,
            source=snapshot.source,
            note=snapshot.note,
            match_count=len(snapshot.matches),
        ))

    infos.sort(
        key=lambda info: (info.captured_at, info.path.name), reverse=True,
    )
    return infos


# ---------------------------------------------------------------------------
# Publikt API: resultatfiler
# ---------------------------------------------------------------------------

def build_result(
    draw: int,
    correct_row: Iterable[Any],
    *,
    turnover: Optional[float] = None,
    payouts: Optional[Dict[str, Any]] = None,
    winners: Optional[Dict[str, Any]] = None,
    entered_at: Optional[str] = None,
    entered_manually: bool = True,
    source: str = RESULT_SOURCE_MANUAL,
    draw_state: Optional[str] = None,
    reg_close_time: Optional[str] = None,
) -> RoundResult:
    """
    Bygger efterhandsdata for en omgang.

    Kastar ValueError om raden innehaller annat an `1`, `X` eller `2`, eller
    om `source` inte ar en kand kalla.
    """
    if source not in RESULT_SOURCES:
        raise ValueError(
            f"source maste vara en av {RESULT_SOURCES}, fick {source!r}"
        )
    row: List[str] = []
    for value in correct_row:
        sign = str(value).strip().upper()
        if sign not in VALID_SIGNS:
            raise ValueError(
                f"Ogiltigt tecken i raden: {value!r} (tillatna: {VALID_SIGNS})"
            )
        row.append(sign)

    return RoundResult(
        draw=int(draw),
        correct_row=row,
        turnover=_optional_float(turnover),
        payouts={
            tier: _optional_float((payouts or {}).get(tier))
            for tier in PAYOUT_TIERS
        },
        winners={
            tier: _optional_float((winners or {}).get(tier))
            for tier in PAYOUT_TIERS
        },
        entered_at=entered_at or utc_timestamp(),
        entered_manually=entered_manually,
        source=source,
        draw_state=draw_state,
        reg_close_time=reg_close_time,
    )


def result_json(result: RoundResult) -> str:
    """Resultatfilens JSON-text."""
    return json.dumps(
        result.to_dict(), ensure_ascii=False, indent=2,
    ) + "\n"


def result_filename(result: RoundResult) -> str:
    """Filnamn `<draw>.json`."""
    return f"{int(result.draw)}.json"


def save_result(
    result: RoundResult,
    directory: Optional[Path] = None,
) -> Path:
    """
    Skriver efterhandsdata till `data/results/<draw>.json`.

    Resultatdata halls i egen fil: snapshot-filerna ror vi aldrig igen.
    Filen far uppdateras nar anvandaren kompletterar utdelningar.
    """
    target_dir = Path(directory) if directory is not None else (
        DEFAULT_RESULTS_DIR
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    path = target_dir / result_filename(result)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(result_json(result))

    logger.info("Resultat sparat: %s.", path)
    return path


def load_result(
    draw: int,
    directory: Optional[Path] = None,
) -> Optional[RoundResult]:
    """Laser efterhandsdata for en omgang, eller None om den saknas."""
    target_dir = Path(directory) if directory is not None else (
        DEFAULT_RESULTS_DIR
    )
    path = target_dir / f"{int(draw)}.json"
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Kunde inte lasa resultat %s: %s", path, exc)
        return None

    return RoundResult(
        draw=_optional_int(raw.get("draw")) or int(draw),
        correct_row=[str(sign) for sign in raw.get("correct_row", [])],
        turnover=_optional_float(raw.get("turnover")),
        payouts={
            tier: _optional_float((raw.get("payouts") or {}).get(tier))
            for tier in PAYOUT_TIERS
        },
        winners={
            tier: _optional_float((raw.get("winners") or {}).get(tier))
            for tier in PAYOUT_TIERS
        },
        entered_at=str(raw.get("entered_at", "")),
        entered_manually=bool(raw.get("entered_manually", True)),
        source=str(raw.get("source") or RESULT_SOURCE_MANUAL),
        draw_state=raw.get("draw_state"),
        # Additivt falt: aldre resultatfiler saknar det och lases som None.
        reg_close_time=raw.get("reg_close_time"),
    )
