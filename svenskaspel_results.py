"""
svenskaspel_results.py -- manuell engangshamtning av omgangsresultat.

Endpointen ar publik men **odokumenterad**, och atkomststatusen bedomdes som
oklar i RESULTS_DATA_AVAILABILITY.md. Modulen gor darfor exakt ett anrop per
anrop av `fetch_result_payload()`: ingen loop, ingen backfill, ingen
bakgrundskorning, ingen automatisk retry. Anroparen (UI:t) triggar hamtningen
pa knapptryckning och far trycka om sjalv vid fel.

Hamtningen sparar aldrig nagot. `parse_result_payload()` returnerar data for
granskning; skrivning sker forst nar anvandaren valjer att spara via
`snapshot_storage.save_result()`.

Resultatendpointen levererar ratt rad, omsattning, utdelning och vinnarantal
i ett och samma svar. Den ger ingen `drawState`, sa `draw_state` blir null:
sparspaerren bygger i stallet pa strukturell kompletthet (13 events med
giltiga utfall).

Anvandning:
    payload = fetch_result_payload(4968)
    fetched = parse_result_payload(payload)
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
    VALID_SIGNS,
    RoundResult,
    build_result,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanter
# ---------------------------------------------------------------------------

RESULT_ENDPOINT = (
    "https://api.spela.svenskaspel.se/draw/1/stryktipset/draws/{draw}/result"
)

# Identifierande User-Agent: anonym hamtning ar inte acceptabelt.
USER_AGENT = (
    "fotbollspredictor_v7 (private analysis tool; "
    "contact: emtatos@gmail.com)"
)

REQUEST_TIMEOUT_SECONDS = 10.0

# En Stryktipsetkupong har alltid 13 matcher.
EXPECTED_EVENT_COUNT = 13

# distribution[].winDiv 0..3 motsvarar 13, 12, 11 och 10 ratt.
_WIN_DIV_TIERS = {index: tier for index, tier in enumerate(PAYOUT_TIERS)}


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
    sign: str
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None


@dataclass
class FetchedResult:
    """Hamtad efterhandsdata, ej sparad."""
    draw: int
    correct_row: List[str]
    turnover: Optional[float] = None
    payouts: Dict[str, Optional[float]] = field(default_factory=dict)
    winners: Dict[str, Optional[float]] = field(default_factory=dict)
    matches: List[FetchedMatch] = field(default_factory=list)
    reg_close_time: Optional[str] = None
    # Resultatendpointen rapporterar ingen drawState; falten gissas inte.
    draw_state: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        """
        True nar raden ar strukturellt komplett och far sparas.

        Endpointen ger ingen omgangsstatus, sa kompletthet -- 13 matcher med
        giltiga utfall -- ar det som avgor om resultatet far skrivas.
        """
        return (
            len(self.matches) == EXPECTED_EVENT_COUNT
            and len(self.correct_row) == EXPECTED_EVENT_COUNT
            and all(sign in VALID_SIGNS for sign in self.correct_row)
        )

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
        if not self.is_complete:
            raise ResultFetchError(
                "Resultatet ar inte strukturellt komplett "
                f"({EXPECTED_EVENT_COUNT} giltiga utfall kravs); "
                "det far inte sparas."
            )
        return build_result(
            self.draw,
            self.correct_row,
            turnover=self.turnover,
            payouts=self.payouts,
            winners=self.winners,
            entered_manually=False,
            source=RESULT_SOURCE_API,
            draw_state=self.draw_state,
            reg_close_time=self.reg_close_time,
        )


# ---------------------------------------------------------------------------
# Intern: hjalpare
# ---------------------------------------------------------------------------

def parse_amount(value: Any) -> Optional[float]:
    """
    Tolkar belopp som `"590909,00"` eller `"29 625 572,00"` till float.

    Beloppen kommer som strangar med svenskt decimalkomma och kan innehalla
    tusentalsavgransare; de castas darfor aldrig rakt av. None nar vardet
    saknas eller inte gar att tolka.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("\xa0", "").replace(" ", "")
    if not text:
        return None
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _parse_int(value: Any) -> Optional[int]:
    """Tolkar heltal, annars None."""
    number = parse_amount(value)
    if number is None:
        return None
    return int(number)


def _distribution_tiers(result: Dict[str, Any]) -> tuple:
    """
    Utdelning och vinnarantal per vinstgrupp.

    Falten forblir null nar `distribution` saknas -- de raknas aldrig fram.
    Vinstgruppen tas fran `name` ("13 ratt" osv.) med `winDiv` som reserv.
    """
    payouts: Dict[str, Optional[float]] = {
        tier: None for tier in PAYOUT_TIERS
    }
    winners: Dict[str, Optional[float]] = {
        tier: None for tier in PAYOUT_TIERS
    }

    entries = result.get("distribution")
    if not isinstance(entries, list):
        return payouts, winners

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("name") or "").strip()
        tier = next(
            (candidate for candidate in PAYOUT_TIERS
             if label.startswith(candidate)),
            _WIN_DIV_TIERS.get(_parse_int(entry.get("winDiv"))),
        )
        if tier is None:
            continue
        payouts[tier] = parse_amount(entry.get("amount"))
        winners[tier] = parse_amount(entry.get("winners"))
    return payouts, winners


# ---------------------------------------------------------------------------
# Publikt API
# ---------------------------------------------------------------------------

def fetch_result_payload(
    draw: int,
    *,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """
    Gor EXAKT ett GET-anrop mot resultatendpointen och returnerar svaret.

    Inget sparas och inget forsok upprepas. Kastar ResultFetchError vid
    natverksfel, timeout, HTTP-fel, ogiltig JSON, okant omgangsnummer
    (`result: null`) eller fel i payloaden (`error`).
    """
    draw_number = int(draw)
    url = RESULT_ENDPOINT.format(draw=draw_number)
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

    if payload.get("result") is None:
        raise ResultFetchError(
            f"Omgang {draw_number} finns inte (result saknas i svaret)."
        )

    logger.info("Hamtade omgang %s fran resultatendpointen.", draw_number)
    return payload


def parse_result_payload(payload: Dict[str, Any]) -> FetchedResult:
    """
    Tolkar resultatsvaret till granskningsbar efterhandsdata.

    Kastar ResultFetchError om svaret inte innehaller exakt
    `EXPECTED_EVENT_COUNT` matcher med giltiga utfall; ingenting gissas.
    """
    result = payload.get("result")
    if not isinstance(result, dict):
        raise ResultFetchError("Svaret innehaller ingen result-post.")

    draw_number = _parse_int(result.get("drawNumber"))
    if draw_number is None:
        raise ResultFetchError("Svaret saknar drawNumber.")

    events = result.get("events")
    if not isinstance(events, list):
        raise ResultFetchError("Svaret saknar matcher (events).")
    if len(events) != EXPECTED_EVENT_COUNT:
        raise ResultFetchError(
            f"Svaret innehaller {len(events)} matcher, forvantat "
            f"{EXPECTED_EVENT_COUNT}."
        )

    matches: List[FetchedMatch] = []
    correct_row: List[str] = []
    ordered = sorted(
        events,
        key=lambda item: (
            _parse_int(item.get("eventNumber"))
            if isinstance(item, dict) else None
        ) or 0,
    )
    for position, event in enumerate(ordered, start=1):
        event = event if isinstance(event, dict) else {}
        outcome = str(event.get("outcome") or "").strip().upper()
        if outcome not in VALID_SIGNS:
            raise ResultFetchError(
                f"Match {position} har ogiltigt utfall "
                f"{event.get('outcome')!r} (tillatna: {VALID_SIGNS})."
            )

        score = event.get("outcomeScore")
        score = score if isinstance(score, dict) else {}
        correct_row.append(outcome)
        matches.append(FetchedMatch(
            position=position,
            description=str(event.get("eventDescription", "")),
            sign=outcome,
            home_goals=_parse_int(score.get("home")),
            away_goals=_parse_int(score.get("away")),
        ))

    payouts, winners = _distribution_tiers(result)

    return FetchedResult(
        draw=draw_number,
        correct_row=correct_row,
        turnover=parse_amount(result.get("currentNetSale")),
        payouts=payouts,
        winners=winners,
        matches=matches,
        reg_close_time=result.get("regCloseTime"),
    )


def fetch_result(
    draw: int,
    *,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> FetchedResult:
    """Ett anrop + tolkning. Sparar ingenting."""
    return parse_result_payload(fetch_result_payload(draw, timeout=timeout))
