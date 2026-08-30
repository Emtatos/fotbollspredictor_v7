"""
svenskaspel_results.py -- manuell engangshamtning av omgangsresultat.

Endpointen ar publik men **odokumenterad**, och atkomststatusen bedomdes som
oklar i RESULTS_DATA_AVAILABILITY.md. Modulen gor darfor exakt ett anrop per
anrop av `fetch_draw_payload()`: ingen loop, ingen backfill, ingen
bakgrundskorning, ingen automatisk retry. Anroparen (UI:t) triggar hamtningen
pa knapptryckning och far trycka om sjalv vid fel.

Hamtningen sparar aldrig nagot. `parse_draw_payload()` returnerar data for
granskning; skrivning sker forst nar anvandaren valjer att spara via
`snapshot_storage.save_result()`.

Anvandning:
    payload = fetch_draw_payload(4966)
    fetched = parse_draw_payload(payload)
    result = fetched.to_round_result()   # skrivs inte till disk
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from snapshot_storage import (
    PAYOUT_TIERS,
    RESULT_SOURCE_API,
    RoundResult,
    build_result,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanter
# ---------------------------------------------------------------------------

DRAW_ENDPOINT = (
    "https://api.spela.svenskaspel.se/draw/1/stryktipset/draws/{draw}"
)

# Identifierande User-Agent: anonym hamtning ar inte acceptabelt.
USER_AGENT = (
    "fotbollspredictor_v7 (private analysis tool; "
    "contact: emtatos@gmail.com)"
)

REQUEST_TIMEOUT_SECONDS = 10.0

DRAW_STATE_FINALIZED = "Finalized"

_FULLTIME_RESULT_TYPE = "Fulltime"


class ResultFetchError(Exception):
    """Hamtningen eller tolkningen misslyckades. Ingen fil ska skapas."""


# ---------------------------------------------------------------------------
# Dataklasser
# ---------------------------------------------------------------------------

@dataclass
class FetchedMatch:
    """En match i det hamtade svaret, for granskningsvyn."""
    position: int
    description: str
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    sign: Optional[str] = None
    league: Optional[str] = None


@dataclass
class FetchedResult:
    """Hamtad efterhandsdata, ej sparad."""
    draw: int
    draw_state: str
    correct_row: List[str]
    turnover: Optional[float] = None
    payouts: Dict[str, Optional[float]] = field(default_factory=dict)
    winners: Dict[str, Optional[float]] = field(default_factory=dict)
    matches: List[FetchedMatch] = field(default_factory=list)
    draw_comment: str = ""
    reg_close_time: Optional[str] = None

    @property
    def is_finalized(self) -> bool:
        """True nar API:t rapporterar omgangen som avslutad."""
        return self.draw_state == DRAW_STATE_FINALIZED

    @property
    def missing_fields(self) -> List[str]:
        """Falt som saknas i svaret och darfor lamnas null."""
        missing: List[str] = []
        if self.turnover is None:
            missing.append("turnover")
        for tier in PAYOUT_TIERS:
            if self.payouts.get(tier) is None:
                missing.append(f"payouts.{tier}")
        for tier in PAYOUT_TIERS:
            if self.winners.get(tier) is None:
                missing.append(f"winners.{tier}")
        return missing

    def to_round_result(self) -> RoundResult:
        """Bygger resultatobjektet i `data/results/<draw>.json`-schemat."""
        return build_result(
            self.draw,
            self.correct_row,
            turnover=self.turnover,
            payouts=self.payouts,
            winners=self.winners,
            entered_manually=False,
            source=RESULT_SOURCE_API,
            draw_state=self.draw_state,
        )


# ---------------------------------------------------------------------------
# Intern: hjalpare
# ---------------------------------------------------------------------------

def _parse_amount(value: Any) -> Optional[float]:
    """Tolkar belopp som `"14740820,00"` till float, annars None."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().replace("\xa0", "").replace(" ", "")
    if not text:
        return None
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _parse_goals(value: Any) -> Optional[int]:
    """Tolkar malantal till int, annars None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _fulltime_goals(match: Dict[str, Any]) -> tuple:
    """Fulltidsresultatet som `(home, away)`, `(None, None)` om det saknas."""
    for entry in match.get("result") or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("sportEventResultType") == _FULLTIME_RESULT_TYPE:
            return (
                _parse_goals(entry.get("home")),
                _parse_goals(entry.get("away")),
            )
    return (None, None)


def _sign_from_goals(home: int, away: int) -> str:
    """Tecknet 1/X/2 for ett matchresultat."""
    if home > away:
        return "1"
    if home < away:
        return "2"
    return "X"


def _distribution_tiers(draw: Dict[str, Any]) -> tuple:
    """
    Utdelning och vinnarantal per vinstgrupp om svaret innehaller dem.

    Draw-endpointen levererar i praktiken ingen `distribution`; falten lamnas
    darfor null istallet for att gissas.
    """
    payouts: Dict[str, Optional[float]] = {
        tier: None for tier in PAYOUT_TIERS
    }
    winners: Dict[str, Optional[float]] = {
        tier: None for tier in PAYOUT_TIERS
    }

    entries = draw.get("distribution")
    if not isinstance(entries, list):
        return payouts, winners

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("name", entry.get("description", "")))
        tier = next(
            (t for t in PAYOUT_TIERS if label.strip().startswith(t)), None,
        )
        if tier is None:
            continue
        payouts[tier] = _parse_amount(entry.get("amount"))
        winners[tier] = _parse_amount(entry.get("winners"))
    return payouts, winners


# ---------------------------------------------------------------------------
# Publikt API
# ---------------------------------------------------------------------------

def fetch_draw_payload(
    draw: int,
    *,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """
    Gor EXAKT ett GET-anrop mot draw-endpointen och returnerar JSON-svaret.

    Inget sparas och inget forsok upprepas. Kastar ResultFetchError vid
    natverksfel, timeout, HTTP-fel, ogiltig JSON, okant omgangsnummer
    (`draw: null`) eller fel i payloaden (`error`).
    """
    draw_number = int(draw)
    url = DRAW_ENDPOINT.format(draw=draw_number)
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
    except requests.Timeout as exc:
        raise ResultFetchError(
            f"Anropet tog for lang tid (timeout {timeout:g} s): {exc}. "
            "Inget omforsok gors automatiskt."
        ) from exc
    except requests.RequestException as exc:
        raise ResultFetchError(f"Natverksfel vid hamtning: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise ResultFetchError(
            f"Svaret kunde inte tolkas som JSON: {exc}"
        ) from exc

    if response.status_code != 200:
        raise ResultFetchError(
            f"Ovantad HTTP-status {response.status_code} for omgang "
            f"{draw_number}."
        )

    if not isinstance(payload, dict):
        raise ResultFetchError("Svaret var inte ett JSON-objekt.")

    error = payload.get("error")
    if error:
        code = error.get("code") if isinstance(error, dict) else error
        raise ResultFetchError(
            f"API:t svarade med fel (code={code}) for omgang {draw_number}."
        )

    if payload.get("draw") is None:
        raise ResultFetchError(
            f"Omgang {draw_number} finns inte (draw saknas i svaret)."
        )

    logger.info("Hamtade omgang %s fran draw-endpointen.", draw_number)
    return payload


def parse_draw_payload(payload: Dict[str, Any]) -> FetchedResult:
    """
    Tolkar draw-svaret till granskningsbar efterhandsdata.

    Kastar ResultFetchError om ratta raden inte kan harledas ur
    matchresultaten; ingenting gissas.
    """
    draw = payload.get("draw")
    if not isinstance(draw, dict):
        raise ResultFetchError("Svaret innehaller ingen draw-post.")

    draw_number = _parse_goals(draw.get("drawNumber"))
    if draw_number is None:
        raise ResultFetchError("Svaret saknar drawNumber.")

    events = draw.get("drawEvents")
    if not isinstance(events, list) or not events:
        raise ResultFetchError("Svaret saknar matcher (drawEvents).")

    matches: List[FetchedMatch] = []
    correct_row: List[str] = []
    for position, event in enumerate(
        sorted(
            events,
            key=lambda item: _parse_goals(item.get("eventNumber")) or 0,
        ),
        start=1,
    ):
        match = event.get("match")
        match = match if isinstance(match, dict) else {}
        league = match.get("league")
        home_goals, away_goals = _fulltime_goals(match)

        if event.get("cancelled"):
            raise ResultFetchError(
                f"Match {position} ar installd; ratta raden kan inte "
                "harledas."
            )
        if home_goals is None or away_goals is None:
            raise ResultFetchError(
                f"Match {position} saknar fulltidsresultat; ratta raden kan "
                "inte harledas."
            )

        sign = _sign_from_goals(home_goals, away_goals)
        correct_row.append(sign)
        matches.append(FetchedMatch(
            position=position,
            description=str(event.get("eventDescription", "")),
            home_goals=home_goals,
            away_goals=away_goals,
            sign=sign,
            league=(
                str(league.get("name")) if isinstance(league, dict)
                and league.get("name") else None
            ),
        ))

    payouts, winners = _distribution_tiers(draw)

    return FetchedResult(
        draw=draw_number,
        draw_state=str(draw.get("drawState", "")),
        correct_row=correct_row,
        turnover=_parse_amount(draw.get("currentNetSale")),
        payouts=payouts,
        winners=winners,
        matches=matches,
        draw_comment=str(draw.get("drawComment", "")),
        reg_close_time=draw.get("regCloseTime"),
    )


def fetch_result(
    draw: int,
    *,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> FetchedResult:
    """Ett anrop + tolkning. Sparar ingenting."""
    return parse_draw_payload(fetch_draw_payload(draw, timeout=timeout))
