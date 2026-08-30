"""Tester for manuell resultathamtning fran Svenska Spels resultatendpoint."""

import copy
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
    REQUEST_TIMEOUT_SECONDS,
    RESULT_ENDPOINT,
    USER_AGENT,
    ResultFetchError,
    fetch_result,
    fetch_result_payload,
    parse_amount,
    parse_result_payload,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "svenskaspel_result_4968.json"
)

# Omgang 4968 (vecka 35 2026) enligt det faktiska API-svaret.
DRAW_4968 = 4968
ROW_4968 = [
    "2", "X", "2", "2", "2", "1", "X", "1", "1", "1", "X", "2", "2",
]
TURNOVER_4968 = 29625572.0
REG_CLOSE_4968 = "2026-08-29T15:59:00+02:00"
PAYOUTS_4968 = {"13": 590909.0, "12": 5919.0, "11": 357.0, "10": 91.0}
WINNERS_4968 = {"13": 22.0, "12": 488.0, "11": 6468.0, "10": 52666.0}


def _payload():
    """Fixturens svar for omgang 4968 (mockat, inget natverksanrop)."""
    with open(FIXTURE_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


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


def test_draw_4968_maps_to_result_schema(tmp_path, call_log):
    """Test 1: fixturen for 4968 mappas exakt till resultatschemat."""
    calls = call_log(lambda _n: FakeResponse(payload=_payload()))

    fetched = fetch_result(DRAW_4968)
    path = save_result(fetched.to_round_result(), directory=tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    assert len(calls) == 1
    assert path.name == "4968.json"
    assert raw["draw"] == DRAW_4968
    assert raw["correct_row"] == ROW_4968
    assert raw["turnover"] == pytest.approx(TURNOVER_4968)
    assert raw["payouts"] == pytest.approx(PAYOUTS_4968)
    assert raw["winners"] == pytest.approx(WINNERS_4968)
    assert raw["reg_close_time"] == REG_CLOSE_4968
    assert raw["entered_manually"] is False
    assert raw["source"] == RESULT_SOURCE_API
    assert raw["draw_state"] is None
    assert raw["entered_at"]
    assert fetched.missing_fields == []

    reloaded = load_result(DRAW_4968, directory=tmp_path)
    assert reloaded.correct_row == ROW_4968
    assert reloaded.source == RESULT_SOURCE_API
    assert reloaded.reg_close_time == REG_CLOSE_4968


def test_correct_row_comes_from_event_outcomes():
    """Raden lases direkt ur events[].outcome."""
    fetched = parse_result_payload(_payload())

    assert fetched.correct_row == ROW_4968
    assert [match.sign for match in fetched.matches] == ROW_4968
    assert fetched.matches[0].description == "Tottenham - Newcastle"
    assert (
        fetched.matches[0].home_goals,
        fetched.matches[0].away_goals,
    ) == (0, 2)


@pytest.mark.parametrize("text, expected", [
    ("590909,00", 590909.0),
    ("29625572,00", 29625572.0),
    ("29 625 572,00", 29625572.0),
    ("91,50", 91.5),
    ("22", 22.0),
    ("", None),
    (None, None),
    ("okant", None),
])
def test_swedish_decimal_comma_is_parsed(text, expected):
    """Test 2: belopp med svenskt decimalkomma tolkas explicit."""
    if expected is None:
        assert parse_amount(text) is None
    else:
        assert parse_amount(text) == pytest.approx(expected)


def test_events_are_ordered_by_event_number():
    """Test 3: raden foljer eventNumber aven vid omkastad ordning."""
    payload = _payload()
    payload["result"]["events"].reverse()

    assert parse_result_payload(payload).correct_row == ROW_4968


def test_wrong_event_count_is_error(tmp_path):
    """Test 4: fler eller farre an 13 matcher ger fel och ingen fil."""
    too_few = _payload()
    too_few["result"]["events"] = too_few["result"]["events"][:12]
    with pytest.raises(ResultFetchError, match="12 matcher"):
        parse_result_payload(too_few)

    too_many = _payload()
    extra = copy.deepcopy(too_many["result"]["events"][0])
    extra["eventNumber"] = 14
    too_many["result"]["events"].append(extra)
    with pytest.raises(ResultFetchError, match="14 matcher"):
        parse_result_payload(too_many)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("outcome", ["", None, "3", "x1", "-"])
def test_invalid_outcome_is_error(tmp_path, outcome):
    """Test 5: utfall utanfor 1/X/2 ger fel och ingen fil."""
    payload = _payload()
    payload["result"]["events"][4]["outcome"] = outcome

    with pytest.raises(ResultFetchError, match="ogiltigt utfall"):
        parse_result_payload(payload)

    assert list(tmp_path.iterdir()) == []


def test_missing_distribution_leaves_null_values():
    """Test 6: saknad distribution ger null-falt, inget raknas fram."""
    payload = _payload()
    del payload["result"]["distribution"]

    fetched = parse_result_payload(payload)
    assert all(fetched.payouts[tier] is None for tier in PAYOUT_TIERS)
    assert all(fetched.winners[tier] is None for tier in PAYOUT_TIERS)
    assert "payouts.13" in fetched.missing_fields
    assert "winners.10" in fetched.missing_fields
    # Omsattningen finns fortfarande och gissas inte fram ur utdelningen.
    assert fetched.turnover == pytest.approx(TURNOVER_4968)

    raw = fetched.to_round_result().to_dict()
    assert raw["payouts"]["13"] is None
    assert raw["winners"]["10"] is None


def test_distribution_maps_by_win_div_when_name_is_missing():
    """Vinstgruppen kan tas fran winDiv nar name saknas."""
    payload = _payload()
    for entry in payload["result"]["distribution"]:
        entry.pop("name")

    fetched = parse_result_payload(payload)
    assert fetched.payouts == pytest.approx(PAYOUTS_4968)
    assert fetched.winners == pytest.approx(WINNERS_4968)


def test_error_field_is_error(tmp_path, call_log):
    """Test 7: ifylld error ger fel och ingen fil."""
    payload = _payload()
    payload["error"] = {"code": 404, "message": "Resource Not Found"}
    calls = call_log(lambda _n: FakeResponse(payload=payload))

    with pytest.raises(ResultFetchError, match="code=404"):
        fetch_result(DRAW_4968)

    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []


def test_result_null_is_error(tmp_path, call_log):
    """Test 8: result: null (okand omgang) ger fel och ingen fil."""
    calls = call_log(lambda _n: FakeResponse(
        payload={"result": None, "error": None},
    ))

    with pytest.raises(ResultFetchError, match="finns inte"):
        fetch_result(999999)

    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []


def test_http_error_status_is_error(tmp_path, call_log):
    """Icke-200-status ger fel och ingen fil."""
    calls = call_log(lambda _n: FakeResponse(
        status_code=500, payload={"result": None, "error": None},
    ))

    with pytest.raises(ResultFetchError, match="HTTP-status 500"):
        fetch_result(DRAW_4968)

    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []


def test_network_error_is_error_without_retry(tmp_path, call_log):
    """Test 9a: natverksfel ger fel, ingen fil, inget omforsok."""
    calls = call_log(
        lambda _n: requests.ConnectionError("namnuppslagning misslyckades"),
    )

    with pytest.raises(ResultFetchError, match="Natverksfel"):
        fetch_result(DRAW_4968)

    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []


def test_timeout_is_error_without_retry(tmp_path, call_log):
    """Test 9b: timeout ger fel, ingen fil, inget omforsok."""
    calls = call_log(lambda _n: requests.Timeout("tiden gick ut"))

    with pytest.raises(ResultFetchError, match="timeout"):
        fetch_result(DRAW_4968)

    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []


def test_invalid_json_is_error(tmp_path, call_log):
    """Ogiltig JSON ger fel och ingen fil."""
    calls = call_log(lambda _n: FakeResponse(payload=None, text="<html>"))

    with pytest.raises(ResultFetchError, match="JSON"):
        fetch_result(DRAW_4968)

    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []


def test_exactly_one_request_per_fetch(call_log):
    """Test 10: exakt ett anrop per hamtning, mot /result-endpointen."""
    calls = call_log(lambda _n: FakeResponse(payload=_payload()))

    fetch_result(DRAW_4968)
    assert len(calls) == 1
    assert calls[0]["url"] == RESULT_ENDPOINT.format(draw=DRAW_4968)
    assert calls[0]["url"].endswith("/draws/4968/result")

    fetch_result(DRAW_4968)
    assert len(calls) == 2


def test_user_agent_and_timeout_are_set(call_log):
    """Identifierande User-Agent och timeout satts i anropet."""
    calls = call_log(lambda _n: FakeResponse(payload=_payload()))

    fetch_result_payload(DRAW_4968)

    assert len(calls) == 1
    call = calls[0]
    assert call["headers"]["User-Agent"] == (
        "fotbollspredictor_v7 (private analysis tool; "
        "contact: emtatos@gmail.com)"
    )
    assert call["headers"]["User-Agent"] == USER_AGENT
    assert call["timeout"] == REQUEST_TIMEOUT_SECONDS


def test_fetch_alone_creates_no_file(tmp_path, call_log):
    """Test 11: hamtning ensam skapar ingen fil; bara save_result skriver."""
    call_log(lambda _n: FakeResponse(payload=_payload()))

    fetched = fetch_result(DRAW_4968)
    fetched.to_round_result()
    assert list(tmp_path.iterdir()) == []

    save_result(fetched.to_round_result(), directory=tmp_path)
    assert [path.name for path in tmp_path.iterdir()] == ["4968.json"]


def test_null_draw_state_does_not_block_save(tmp_path, call_log):
    """
    Test 13: draw_state = null gissas inte och kraver ingen bekraftelse.

    Sparspaerren ar strukturell kompletthet, inte omgangsstatus, sa den
    gamla kryssrutan for icke-avslutade omgangar ar borta.
    """
    call_log(lambda _n: FakeResponse(payload=_payload()))

    fetched = fetch_result(DRAW_4968)
    assert fetched.draw_state is None
    assert fetched.is_complete is True
    assert not hasattr(fetched, "is_finalized")
    assert not hasattr(svenskaspel_results, "DRAW_STATE_FINALIZED")

    raw = json.loads(
        save_result(
            fetched.to_round_result(), directory=tmp_path,
        ).read_text(encoding="utf-8")
    )
    assert raw["draw_state"] is None


def test_incomplete_result_cannot_be_saved(tmp_path):
    """Strukturellt inkompletta resultat kan inte sparas."""
    fetched = parse_result_payload(_payload())
    fetched.correct_row = fetched.correct_row[:12]

    assert fetched.is_complete is False
    with pytest.raises(ResultFetchError, match="komplett"):
        fetched.to_round_result()

    assert list(tmp_path.iterdir()) == []


def test_payload_without_events_is_error():
    """Svar utan matcher ger fel istallet for en tom rad."""
    payload = _payload()
    del payload["result"]["events"]

    with pytest.raises(ResultFetchError, match="events"):
        parse_result_payload(payload)
