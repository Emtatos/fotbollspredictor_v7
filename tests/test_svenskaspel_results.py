"""Tester for manuell resultathamtning fran Svenska Spels draw-endpoint."""

import json
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import svenskaspel_results  # noqa: E402
from snapshot_storage import (  # noqa: E402
    PAYOUT_TIERS,
    RESULT_SOURCE_API,
    load_result,
    save_result,
)
from svenskaspel_results import (  # noqa: E402
    DRAW_ENDPOINT,
    REQUEST_TIMEOUT_SECONDS,
    USER_AGENT,
    ResultFetchError,
    fetch_draw_payload,
    fetch_result,
    parse_draw_payload,
)

# 13 matchresultat: hemmavinst, bortavinst och oavgjort ska ge 1/2/X.
SCORES = [
    (2, 0), (0, 2), (1, 1), (3, 1), (0, 0), (1, 2), (2, 2),
    (4, 0), (0, 1), (1, 0), (2, 1), (0, 3), (1, 1),
]
EXPECTED_ROW = [
    "1", "2", "X", "1", "X", "2", "X", "1", "2", "1", "1", "2", "X",
]


def _draw_event(number, home_goals, away_goals):
    return {
        "cancelled": False,
        "eventNumber": number,
        "eventDescription": f"Hem {number} - Borta {number}",
        "match": {
            "matchId": 1000 + number,
            "league": {"name": "Championship"},
            "result": [
                {
                    "sportEventResultType": "Halftime",
                    "home": "0",
                    "away": "0",
                },
                {
                    "sportEventResultType": "Fulltime",
                    "home": str(home_goals),
                    "away": str(away_goals),
                },
            ],
        },
    }


def _payload(
    *,
    draw_number=4966,
    draw_state="Finalized",
    net_sale="14740820,00",
    scores=None,
):
    scores = SCORES if scores is None else scores
    return {
        "draw": {
            "drawNumber": draw_number,
            "drawState": draw_state,
            "drawComment": "Stryktipset v. 2026-33",
            "currentNetSale": net_sale,
            "regCloseTime": "2026-08-15T15:59:00+02:00",
            "drawEvents": [
                _draw_event(index, home, away)
                for index, (home, away) in enumerate(scores, start=1)
            ],
        },
        "error": None,
    }


class FakeResponse:
    """Minimal requests-liknande respons for mockade anrop."""

    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self._text = text

    def json(self):
        if self._payload is None:
            raise ValueError(f"Ogiltig JSON: {self._text!r}")
        return self._payload


@pytest.fixture
def call_log(monkeypatch):
    """Mockar requests.get och raknar antal anrop."""
    calls = []

    def _install(behaviour):
        def fake_get(url, **kwargs):
            calls.append({"url": url, **kwargs})
            outcome = behaviour(len(calls))
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(svenskaspel_results.requests, "get", fake_get)
        return calls

    return _install


def test_successful_fetch_maps_to_result_schema(tmp_path, call_log):
    """Test 1: lyckad hamtning mappas till resultatschemat."""
    call_log(lambda _n: FakeResponse(payload=_payload()))

    fetched = fetch_result(4966)
    result = fetched.to_round_result()
    path = save_result(result, directory=tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    assert path.name == "4966.json"
    assert raw["draw"] == 4966
    assert raw["correct_row"] == EXPECTED_ROW
    assert raw["turnover"] == pytest.approx(14740820.0)
    assert set(raw["payouts"]) == set(PAYOUT_TIERS)
    assert set(raw["winners"]) == set(PAYOUT_TIERS)
    assert raw["entered_manually"] is False
    assert raw["source"] == RESULT_SOURCE_API
    assert raw["draw_state"] == "Finalized"
    assert raw["entered_at"]

    reloaded = load_result(4966, directory=tmp_path)
    assert reloaded.correct_row == EXPECTED_ROW
    assert reloaded.source == RESULT_SOURCE_API


def test_correct_row_includes_draws_as_x():
    """Test 2: raden harleds ur matchresultaten, oavgjort blir X."""
    fetched = parse_draw_payload(_payload())

    assert fetched.correct_row == EXPECTED_ROW
    assert [match.sign for match in fetched.matches] == EXPECTED_ROW
    assert [
        (match.home_goals, match.away_goals) for match in fetched.matches
    ] == SCORES


def test_events_are_ordered_by_event_number():
    """Raden foljer eventNumber aven om svaret kommer i annan ordning."""
    payload = _payload()
    payload["draw"]["drawEvents"].reverse()

    assert parse_draw_payload(payload).correct_row == EXPECTED_ROW


def test_http_200_with_draw_null_is_error(tmp_path, call_log):
    """Test 3: HTTP 200 med draw: null ger fel och ingen fil."""
    calls = call_log(lambda _n: FakeResponse(
        payload={"draw": None, "error": None},
    ))

    with pytest.raises(ResultFetchError, match="finns inte"):
        fetch_result(999999)

    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []


def test_http_200_with_error_404_is_error(tmp_path, call_log):
    """Test 4: HTTP 200 med error.code 404 ger fel och ingen fil."""
    calls = call_log(lambda _n: FakeResponse(payload={
        "draw": None,
        "error": {"code": 404, "message": "Resource Not Found"},
    }))

    with pytest.raises(ResultFetchError, match="code=404"):
        fetch_result(999999)

    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []


def test_http_error_status_is_error(tmp_path, call_log):
    """Icke-200-status ger fel och ingen fil."""
    calls = call_log(lambda _n: FakeResponse(
        status_code=500, payload={"error": None, "draw": None},
    ))

    with pytest.raises(ResultFetchError):
        fetch_result(4966)

    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []


def test_network_error_is_error_without_retry(tmp_path, call_log):
    """Test 5a: natverksfel ger fel, ingen fil, inget omforsok."""
    calls = call_log(
        lambda _n: requests.ConnectionError("namnuppslagning misslyckades"),
    )

    with pytest.raises(ResultFetchError, match="Natverksfel"):
        fetch_result(4966)

    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []


def test_timeout_is_error_without_retry(tmp_path, call_log):
    """Test 5b: timeout ger fel, ingen fil, inget omforsok."""
    calls = call_log(lambda _n: requests.Timeout("tiden gick ut"))

    with pytest.raises(ResultFetchError, match="timeout"):
        fetch_result(4966)

    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []


def test_invalid_json_is_error(tmp_path, call_log):
    """Test 6: ogiltig JSON ger fel och ingen fil."""
    calls = call_log(lambda _n: FakeResponse(payload=None, text="<html>"))

    with pytest.raises(ResultFetchError, match="JSON"):
        fetch_result(4966)

    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []


def test_unfinalized_draw_is_flagged_but_savable(tmp_path, call_log):
    """Test 7: drawState != Finalized flaggas och kan sparas med status."""
    call_log(lambda _n: FakeResponse(
        payload=_payload(draw_state="Ongoing"),
    ))

    fetched = fetch_result(4966)
    assert fetched.is_finalized is False
    assert fetched.draw_state == "Ongoing"

    path = save_result(fetched.to_round_result(), directory=tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["draw_state"] == "Ongoing"


def test_missing_payouts_stay_null():
    """Test 8: saknad utdelning ger null-falt, inget gissas."""
    fetched = parse_draw_payload(_payload(net_sale=None))

    assert fetched.turnover is None
    assert all(fetched.payouts[tier] is None for tier in PAYOUT_TIERS)
    assert all(fetched.winners[tier] is None for tier in PAYOUT_TIERS)
    assert "turnover" in fetched.missing_fields
    assert "payouts.13" in fetched.missing_fields
    assert "winners.10" in fetched.missing_fields

    raw = json.loads(json.dumps(fetched.to_round_result().to_dict()))
    assert raw["turnover"] is None
    assert raw["payouts"]["13"] is None


def test_distribution_is_used_when_present():
    """Utdelning tas fran svaret nar den finns; annars null."""
    payload = _payload()
    payload["draw"]["distribution"] = [
        {"name": "13 ratt", "winners": "5", "amount": "1119407,00"},
        {"name": "10 ratt", "winners": "25703", "amount": "136,00"},
    ]

    fetched = parse_draw_payload(payload)
    assert fetched.payouts["13"] == pytest.approx(1119407.0)
    assert fetched.winners["13"] == pytest.approx(5.0)
    assert fetched.payouts["12"] is None
    assert fetched.winners["11"] is None


def test_fetch_alone_creates_no_file(tmp_path, call_log):
    """Test 9: hamtning ensam skapar ingen fil."""
    call_log(lambda _n: FakeResponse(payload=_payload()))

    fetched = fetch_result(4966)
    fetched.to_round_result()
    assert list(tmp_path.iterdir()) == []

    save_result(fetched.to_round_result(), directory=tmp_path)
    assert [path.name for path in tmp_path.iterdir()] == ["4966.json"]


def test_user_agent_and_timeout_are_set(call_log):
    """Test 11: identifierande User-Agent och timeout satts i anropet."""
    calls = call_log(lambda _n: FakeResponse(payload=_payload()))

    fetch_draw_payload(4966)

    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == DRAW_ENDPOINT.format(draw=4966)
    assert call["headers"]["User-Agent"] == (
        "fotbollspredictor_v7 (private analysis tool; "
        "contact: emtatos@gmail.com)"
    )
    assert call["headers"]["User-Agent"] == USER_AGENT
    assert call["timeout"] == REQUEST_TIMEOUT_SECONDS


def test_cancelled_or_missing_result_is_error(tmp_path, call_log):
    """Installd match eller saknat fulltidsresultat ger fel, ingen fil."""
    payload = _payload()
    payload["draw"]["drawEvents"][3]["cancelled"] = True
    calls = call_log(lambda _n: FakeResponse(payload=payload))

    with pytest.raises(ResultFetchError, match="installd"):
        fetch_result(4966)

    payload_missing = _payload()
    payload_missing["draw"]["drawEvents"][2]["match"]["result"] = [
        {"sportEventResultType": "Halftime", "home": "0", "away": "0"},
    ]
    with pytest.raises(ResultFetchError, match="fulltidsresultat"):
        parse_draw_payload(payload_missing)

    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []


def test_payload_without_events_is_error():
    """Svar utan matcher ger fel istallet for en tom rad."""
    payload = _payload()
    payload["draw"]["drawEvents"] = []

    with pytest.raises(ResultFetchError, match="drawEvents"):
        parse_draw_payload(payload)
