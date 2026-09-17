"""Tester for databasarkivet (archive/). Inga riktiga natverksanrop."""

import copy
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest
import requests
from sqlalchemy import inspect, select, text

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from archive import db, fetch, legacy, status  # noqa: E402
from archive.db import (  # noqa: E402
    SOURCE_API,
    SOURCE_IMAGE_SCAN,
    SOURCE_LEGACY_IMPORT,
    SOURCE_PASTE,
    DatabaseNotConfigured,
)
from archive.fetch import (  # noqa: E402
    PlayedRow,
    ResultConflictError,
    SnapshotRequiredError,
    SnapshotRow,
    capture_market_snapshot,
    fetch_current_round,
    fetch_result,
    register_played_system,
    save_snapshot,
)

FIXTURES = ROOT / "tests" / "fixtures"
DRAWS_OPEN = json.loads(
    (FIXTURES / "svenskaspel_draws_open_4971.json").read_text("utf-8")
)
DRAW_4971 = json.loads(
    (FIXTURES / "svenskaspel_draw_4971.json").read_text("utf-8")
)
RESULT_4968 = json.loads(
    (FIXTURES / "svenskaspel_result_4968.json").read_text("utf-8")
)
ROW_4968 = "2X2221X111X22"


# ---------------------------------------------------------------------------
# Hjalpare
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    eng = db.make_engine("sqlite:///:memory:")
    db.create_schema(eng)
    yield eng
    eng.dispose()


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch):
    """Alla tester har maste mocka natverket explicit."""
    def _forbidden(*args, **kwargs):
        raise AssertionError(f"Riktigt natverksanrop forsoktes: {args}")
    monkeypatch.setattr(requests, "get", _forbidden)
    monkeypatch.setattr(requests, "request", _forbidden)
    yield


def _response(payload, status_code=200):
    resp = mock.Mock()
    resp.status_code = status_code
    resp.json.return_value = payload
    return resp


def _api(mapping):
    """`mapping`: url-suffix -> payload."""
    def _get(url, headers=None, timeout=None):
        for suffix, payload in mapping.items():
            if url.endswith(suffix):
                return _response(payload)
        raise AssertionError(f"Ovantad URL: {url}")
    return _get


def _played_rows(n_half=7):
    return [
        {
            "position": i,
            "played_signs": "1X" if i <= n_half else "1",
            "is_halfguard": i <= n_half,
            "combined_p1": 0.5, "combined_px": 0.3, "combined_p2": 0.2,
            "gain": 0.3, "sources_used": "modell + odds + streck",
        }
        for i in range(1, 14)
    ]


def _snapshot_rows(draw=4971):
    return [
        SnapshotRow(
            position=i, streck_1=50, streck_x=30, streck_2=20,
            odds_1=1.5, odds_x=4.0, odds_2=6.0,
            home_team=f"Hem{i}", away_team=f"Bort{i}",
        )
        for i in range(1, 14)
    ]


def _count(engine, table):
    with db.session_scope(engine) as s:
        return s.execute(text(f"select count(*) from {table}")).scalar()


# ---------------------------------------------------------------------------
# 1-2. Schema och rundtur
# ---------------------------------------------------------------------------

def test_schema_creates_all_tables_with_foreign_keys(engine):
    insp = inspect(engine)
    tables = set(insp.get_table_names())
    assert {
        "rounds", "round_matches", "market_snapshots",
        "market_snapshot_matches", "played_systems",
        "played_system_matches", "results", "result_matches",
    } <= tables
    fks = {
        t: {fk["referred_table"] for fk in insp.get_foreign_keys(t)}
        for t in tables
    }
    assert fks["round_matches"] == {"rounds"}
    assert fks["market_snapshots"] == {"rounds"}
    assert fks["market_snapshot_matches"] == {"market_snapshots"}
    assert fks["played_systems"] == {"rounds", "market_snapshots"}
    assert fks["played_system_matches"] == {"played_systems"}
    assert fks["results"] == {"rounds"}
    assert fks["result_matches"] == {"results"}
    assert insp.get_pk_constraint("round_matches")["constrained_columns"] == [
        "draw_number", "position",
    ]


def test_sqlite_enforces_foreign_keys(engine):
    with pytest.raises(Exception):
        with db.session_scope(engine) as s:
            s.execute(db.market_snapshots.insert().values(
                draw_number=1, captured_at=fetch.now_utc(),
                captured_at_precision="exact", source="api",
                parser_version=1,
            ))


def test_get_engine_requires_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    db.set_engine(None)
    with pytest.raises(DatabaseNotConfigured):
        db.get_engine()


def test_round_trip_all_tables(engine):
    fetch.ensure_round(
        4971, engine=engine, week_label="v38",
        reg_close_time=datetime(2026, 9, 19, 13, 59, tzinfo=timezone.utc),
        status="open",
        matches=[fetch.RoundMatch(i, f"H{i}", f"A{i}", "Liga")
                 for i in range(1, 14)],
    )
    sid = save_snapshot(4971, _snapshot_rows(), source=SOURCE_PASTE,
                        engine=engine)
    pid = register_played_system(4971, _played_rows(), sid, engine=engine)
    from svenskaspel_results import parse_result_payload
    fetched = parse_result_payload(RESULT_4968)
    fetch.save_result(fetched, source=SOURCE_API, raw_payload=RESULT_4968,
                      engine=engine)

    statuses = {s.draw_number: s for s in status.list_round_status(engine=engine)}
    r4971 = statuses[4971]
    assert r4971.week_label == "v38"
    assert r4971.match_count == 13
    assert [s.id for s in r4971.snapshots] == [sid]
    assert r4971.played_systems[0].id == pid
    assert r4971.played_systems[0].n_halfguards == 7
    assert r4971.result_state == "waiting"
    r4968 = statuses[4968]
    assert r4968.result.correct_row == ROW_4968
    assert r4968.result.payouts["13"] == 590_909.0
    assert r4968.result_state == "finalized"
    assert r4968.status == "finalized"
    assert _count(engine, "result_matches") == 13


# ---------------------------------------------------------------------------
# 3. Flera snapshots for samma omgang
# ---------------------------------------------------------------------------

def test_multiple_snapshots_for_one_draw_are_separate(engine):
    a = save_snapshot(4971, _snapshot_rows(), source=SOURCE_PASTE, engine=engine)
    b = save_snapshot(4971, _snapshot_rows(), source=SOURCE_IMAGE_SCAN,
                      engine=engine)
    assert a != b
    st = status.get_round_status(4971, engine=engine)
    assert [s.source for s in st.snapshots] == [SOURCE_PASTE, SOURCE_IMAGE_SCAN]
    assert _count(engine, "market_snapshot_matches") == 26


# ---------------------------------------------------------------------------
# 4. fetch_current_round
# ---------------------------------------------------------------------------

def test_fetch_current_round_open(engine, monkeypatch):
    getter = mock.Mock(side_effect=_api({"/draws": DRAWS_OPEN}))
    monkeypatch.setattr(fetch.requests, "get", getter)
    rnd = fetch_current_round(engine=engine)
    assert rnd is not None
    assert rnd.draw_number == 4971
    assert rnd.week_label == "v38"
    assert rnd.status == "open"
    assert rnd.reg_close_time == datetime(2026, 9, 19, 13, 59, tzinfo=timezone.utc)
    assert len(rnd.matches) == 13
    assert rnd.matches[0].home_team and rnd.matches[0].away_team
    assert getter.call_count == 1
    _, kwargs = getter.call_args
    assert kwargs["timeout"] == 10.0
    assert kwargs["headers"]["User-Agent"] == fetch.USER_AGENT
    st = status.current_round_status(engine=engine)
    assert st.draw_number == 4971 and st.round_identified

    # Idempotent: andra anropet uppdaterar, raderar inget.
    fetch_current_round(engine=engine)
    assert _count(engine, "rounds") == 1
    assert _count(engine, "round_matches") == 13


def test_fetch_current_round_empty_list_returns_none(engine, monkeypatch, caplog):
    monkeypatch.setattr(fetch.requests, "get", _api({"/draws": {"draws": []}}))
    with caplog.at_level("INFO"):
        assert fetch_current_round(engine=engine) is None
    assert "Ingen oppen omgang" in caplog.text
    assert _count(engine, "rounds") == 0


def test_fetch_current_round_network_error_is_logged_and_raised(
    engine, monkeypatch, caplog,
):
    def _boom(*a, **k):
        raise requests.ConnectionError("nere")
    monkeypatch.setattr(fetch.requests, "get", _boom)
    with pytest.raises(fetch.ArchiveFetchError):
        fetch_current_round(engine=engine)
    assert "Natverksfel" in caplog.text


# ---------------------------------------------------------------------------
# 5. capture_market_snapshot
# ---------------------------------------------------------------------------

def test_capture_market_snapshot_writes_13_rows_and_raw_payload(
    engine, monkeypatch,
):
    getter = mock.Mock(side_effect=_api({"/draws/4971": DRAW_4971}))
    monkeypatch.setattr(fetch.requests, "get", getter)
    sid = capture_market_snapshot(4971, engine=engine)
    assert getter.call_count == 1
    with db.session_scope(engine) as s:
        header = s.execute(
            select(db.market_snapshots).where(db.market_snapshots.c.id == sid)
        ).mappings().one()
        rows = s.execute(
            select(db.market_snapshot_matches)
            .where(db.market_snapshot_matches.c.snapshot_id == sid)
            .order_by(db.market_snapshot_matches.c.position)
        ).mappings().all()
    assert header["source"] == SOURCE_API
    assert header["parser_version"] == 1
    assert header["raw_payload"] == DRAW_4971
    assert len(rows) == 13
    first = rows[0]
    ev = DRAW_4971["draw"]["drawEvents"][0]
    assert first["streck_1"] == float(ev["svenskaFolket"]["one"].replace(",", "."))
    assert first["odds_1"] == float(ev["odds"]["one"].replace(",", "."))
    assert first["startodds_x"] == float(ev["startOdds"]["x"].replace(",", "."))
    assert first["favoritskap_2"] == float(
        ev["favouriteOdds"]["two"].replace(",", ".")
    )
    assert all(r["streck_1"] is not None and r["odds_1"] is not None for r in rows)


def test_capture_market_snapshot_rejects_wrong_event_count(engine, monkeypatch):
    broken = copy.deepcopy(DRAW_4971)
    broken["draw"]["drawEvents"].pop()
    monkeypatch.setattr(fetch.requests, "get", _api({"/draws/4971": broken}))
    with pytest.raises(fetch.ArchiveFetchError):
        capture_market_snapshot(4971, engine=engine)
    assert _count(engine, "market_snapshots") == 0


# ---------------------------------------------------------------------------
# 6, 15, 16. fetch_result
# ---------------------------------------------------------------------------

def test_fetch_result_from_fixture(engine, monkeypatch):
    getter = mock.Mock(side_effect=_api({"/draws/4968/result": RESULT_4968}))
    monkeypatch.setattr(fetch.requests, "get", getter)
    import svenskaspel_results
    monkeypatch.setattr(svenskaspel_results.requests, "get", getter)
    fetch_result(4968, engine=engine)
    assert getter.call_count == 1
    st = status.get_round_status(4968, engine=engine)
    assert st.result.correct_row == ROW_4968
    assert st.result.winners["13"] == 22
    assert st.result.turnover is not None
    with db.session_scope(engine) as s:
        outcomes = s.execute(
            select(db.result_matches.c.outcome)
            .where(db.result_matches.c.draw_number == 4968)
            .order_by(db.result_matches.c.position)
        ).scalars().all()
    assert "".join(outcomes) == ROW_4968


def test_fetch_result_identical_is_idempotent(engine, monkeypatch):
    getter = mock.Mock(side_effect=_api({"/draws/4968/result": RESULT_4968}))
    import svenskaspel_results
    monkeypatch.setattr(svenskaspel_results.requests, "get", getter)
    fetch_result(4968, engine=engine)
    fetch_result(4968, engine=engine)
    assert _count(engine, "results") == 1
    assert _count(engine, "result_matches") == 13


def test_fetch_result_conflict_does_not_overwrite(engine, monkeypatch):
    import svenskaspel_results
    monkeypatch.setattr(
        svenskaspel_results.requests, "get",
        _api({"/draws/4968/result": RESULT_4968}),
    )
    fetch_result(4968, engine=engine)

    changed = copy.deepcopy(RESULT_4968)
    changed["result"]["distribution"][0]["amount"] = "1,00"
    monkeypatch.setattr(
        svenskaspel_results.requests, "get",
        _api({"/draws/4968/result": changed}),
    )
    with pytest.raises(ResultConflictError):
        fetch_result(4968, engine=engine)
    st = status.get_round_status(4968, engine=engine)
    assert st.result.payouts["13"] == 590_909.0


# ---------------------------------------------------------------------------
# 7, 9, 17. register_played_system
# ---------------------------------------------------------------------------

def test_register_played_system_with_seven_halfguards(engine):
    sid = save_snapshot(4971, _snapshot_rows(), source=SOURCE_PASTE, engine=engine)
    pid = register_played_system(4971, _played_rows(7), sid, engine=engine)
    with db.session_scope(engine) as s:
        header = s.execute(
            select(db.played_systems).where(db.played_systems.c.id == pid)
        ).mappings().one()
        rows = s.execute(
            select(db.played_system_matches)
            .where(db.played_system_matches.c.played_system_id == pid)
            .order_by(db.played_system_matches.c.position)
        ).mappings().all()
    assert header["n_halfguards"] == 7
    assert header["snapshot_id"] == sid
    assert len(rows) == 13
    assert sum(r["is_halfguard"] for r in rows) == 7
    assert rows[0]["played_signs"] == "1X"
    assert rows[12]["played_signs"] == "1"
    assert rows[0]["gain"] == 0.3
    assert rows[0]["sources_used"] == "modell + odds + streck"


def test_register_played_system_rejects_null_snapshot_except_legacy(engine):
    with pytest.raises(SnapshotRequiredError):
        register_played_system(4971, _played_rows(), None, engine=engine)
    with pytest.raises(SnapshotRequiredError):
        register_played_system(
            4971, _played_rows(), None, source=SOURCE_IMAGE_SCAN, engine=engine,
        )
    pid = register_played_system(
        4971, _played_rows(), None, source=SOURCE_LEGACY_IMPORT, engine=engine,
    )
    assert pid >= 1


def test_register_played_system_duplicate_returns_same_id(engine):
    sid = save_snapshot(4971, _snapshot_rows(), source=SOURCE_PASTE, engine=engine)
    a = register_played_system(4971, _played_rows(), sid, engine=engine)
    shuffled = [
        dict(r, played_signs="X1" if r["played_signs"] == "1X" else r["played_signs"])
        for r in _played_rows()
    ]
    b = register_played_system(4971, shuffled, sid, engine=engine)
    assert a == b
    assert _count(engine, "played_systems") == 1
    other = [PlayedRow(i, "2", False) for i in range(1, 14)]
    c = register_played_system(4971, other, sid, engine=engine)
    assert c != a


def test_register_played_system_rejects_snapshot_of_other_draw(engine):
    sid = save_snapshot(4970, _snapshot_rows(), source=SOURCE_PASTE, engine=engine)
    with pytest.raises(ValueError):
        register_played_system(4971, _played_rows(), sid, engine=engine)


# ---------------------------------------------------------------------------
# 8. Automatiskt snapshot fran scan/paste innan registrering (archive_ui)
# ---------------------------------------------------------------------------

def test_ensure_snapshot_for_current_round_creates_scan_or_paste_snapshot(
    engine, monkeypatch,
):
    import archive_ui
    monkeypatch.setattr(archive_ui.st, "session_state", {})

    class Odds:
        def __init__(self, h, d, a):
            self.home, self.draw, self.away = h, d, a

    from matchday_import import _make_key
    matches = [(f"Hem{i}", f"Bort{i}") for i in range(1, 14)]
    cr = {
        "matches": matches,
        "odds": {_make_key(h, a): [Odds(1.5, 4.0, 6.0)] for h, a in matches},
        "streck": {_make_key(h, a): {"1": 55, "X": 25, "2": 20} for h, a in matches},
        "source": "kupongbild",
    }
    sid = archive_ui.ensure_snapshot_for_current_round(4971, cr, engine=engine)
    assert archive_ui.remembered_snapshot_id(4971) == sid
    st = status.get_round_status(4971, engine=engine)
    assert st.snapshots[0].source == SOURCE_IMAGE_SCAN
    with db.session_scope(engine) as s:
        row = s.execute(
            select(db.market_snapshot_matches)
            .where(db.market_snapshot_matches.c.snapshot_id == sid)
        ).mappings().first()
    assert (row["odds_1"], row["streck_x"]) == (1.5, 25.0)

    # Samma id ateranvands, inget nytt snapshot.
    assert archive_ui.ensure_snapshot_for_current_round(
        4971, cr, engine=engine,
    ) == sid
    assert _count(engine, "market_snapshots") == 1

    pid = register_played_system(4971, _played_rows(), sid, engine=engine)
    assert pid >= 1

    archive_ui.st.session_state.clear()
    cr["source"] = "textpaste"
    sid2 = archive_ui.ensure_snapshot_for_current_round(4971, cr, engine=engine)
    assert status.get_round_status(4971, engine=engine).snapshots[1].source == (
        SOURCE_PASTE
    )
    assert sid2 != sid


# ---------------------------------------------------------------------------
# 10-11. Legacy import
# ---------------------------------------------------------------------------

def _legacy_snapshot(draw=4968):
    return {
        "version": 1,
        "draw": draw,
        "captured_at": "2026-08-29T10:00:00Z",
        "captured_at_precision": "exact",
        "source": "image_scan",
        "note": "",
        "matches": [
            {
                "position": i, "home_team": f"H{i}", "away_team": f"A{i}",
                "league": None, "streck_1": 40, "streck_x": 30, "streck_2": 30,
                "odds_1": 2.0, "odds_x": 3.3, "odds_2": 3.5,
            }
            for i in range(1, 14)
        ],
    }


def _legacy_result(draw=4968):
    return {
        "draw": draw,
        "correct_row": list(ROW_4968),
        "turnover": 59_836_398.0,
        "payouts": {"13": 4207374.0, "12": 20000.0, "11": 900.0, "10": 80.0},
        "winners": {"13": 1, "12": 100, "11": 3000, "10": 40000},
        "entered_at": "2026-09-01T08:00:00Z",
        "entered_manually": False,
        "source": "svenskaspel_api",
    }


def test_legacy_import_requires_draw_when_null(engine):
    payload = _legacy_snapshot(draw=None)
    with pytest.raises(legacy.DrawNumberRequired):
        legacy.import_legacy_json(payload, engine=engine)
    out = legacy.import_legacy_json(payload, draw_number=4968, engine=engine)
    assert out.kind == "snapshot" and out.draw_number == 4968
    st = status.get_round_status(4968, engine=engine)
    assert st.snapshots[0].source == SOURCE_LEGACY_IMPORT
    assert st.snapshots[0].captured_at == datetime(
        2026, 8, 29, 10, 0, tzinfo=timezone.utc,
    )
    assert st.match_count == 13

    res = _legacy_result(draw=None)
    with pytest.raises(legacy.DrawNumberRequired):
        legacy.import_legacy_json(json.dumps(res), engine=engine)
    out = legacy.import_legacy_json(json.dumps(res), draw_number=4968, engine=engine)
    assert out.kind == "result"
    st = status.get_round_status(4968, engine=engine)
    assert st.result.correct_row == ROW_4968
    assert st.result.source == SOURCE_LEGACY_IMPORT


def test_legacy_import_rejects_duplicates(engine):
    legacy.import_legacy_json(_legacy_snapshot(), engine=engine)
    with pytest.raises(legacy.DuplicateImport):
        legacy.import_legacy_json(_legacy_snapshot(), engine=engine)
    assert _count(engine, "market_snapshots") == 1

    legacy.import_legacy_json(_legacy_result(), engine=engine)
    with pytest.raises(legacy.DuplicateImport):
        legacy.import_legacy_json(_legacy_result(), engine=engine)
    assert _count(engine, "results") == 1


def test_legacy_import_for_4968_and_4969_generic(engine):
    for draw in (4968, 4969):
        legacy.import_legacy_json(_legacy_snapshot(draw), engine=engine)
        legacy.import_legacy_json(
            dict(_legacy_result(draw), correct_row=list("1" * 13)), engine=engine,
        )
    draws = {s.draw_number for s in status.list_round_status(engine=engine)}
    assert draws == {4968, 4969}


def test_legacy_import_rejects_draw_mismatch(engine):
    with pytest.raises(legacy.LegacyImportError):
        legacy.import_legacy_json(
            _legacy_snapshot(4968), draw_number=4969, engine=engine,
        )


# ---------------------------------------------------------------------------
# 12. Statusharledning
# ---------------------------------------------------------------------------

def test_status_derivation(engine):
    fetch.ensure_round(4971, engine=engine, status="open")
    st = status.get_round_status(4971, engine=engine)
    assert not st.round_identified
    assert not st.has_snapshot and not st.has_played_system
    assert st.result_state == "waiting"
    assert st.is_open

    fetch.ensure_round(
        4971, engine=engine, status="open",
        matches=[fetch.RoundMatch(i, f"H{i}", f"A{i}") for i in range(1, 14)],
    )
    sid = save_snapshot(4971, _snapshot_rows(), source=SOURCE_PASTE, engine=engine)
    register_played_system(4971, _played_rows(3), sid, engine=engine)
    st = status.get_round_status(4971, engine=engine)
    assert st.round_identified and st.has_snapshot and st.has_played_system
    assert st.played_systems[0].n_halfguards == 3
    assert st.latest_snapshot.id == sid

    fetch.ensure_round(4970, engine=engine, status="closed")
    assert status.current_round_status(engine=engine).draw_number == 4971


# ---------------------------------------------------------------------------
# 14. Streamlit-fritt
# ---------------------------------------------------------------------------

def test_fetch_modules_do_not_import_streamlit():
    code = (
        "import sys; import archive.fetch, archive.db, archive.status, "
        "archive.legacy; "
        "assert 'streamlit' not in sys.modules, 'streamlit importerad'"
    )
    subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, check=True,
        capture_output=True,
    )


# ---------------------------------------------------------------------------
# 18. Alembic-migration mot tom SQLite
# ---------------------------------------------------------------------------

def test_alembic_upgrade_head_builds_schema(tmp_path):
    url = f"sqlite:///{tmp_path / 'archive.sqlite'}"
    from alembic import command
    from alembic.config import Config
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", url)
    with mock.patch.dict("os.environ", {"DATABASE_URL": url}):
        command.upgrade(cfg, "head")

    eng = db.make_engine(url)
    insp = inspect(eng)
    assert {
        "rounds", "round_matches", "market_snapshots",
        "market_snapshot_matches", "played_systems",
        "played_system_matches", "results", "result_matches",
        "alembic_version",
    } <= set(insp.get_table_names())
    migrated = {
        t: {c["name"] for c in insp.get_columns(t)}
        for t in db.metadata.tables
    }
    declared = {
        t: {c.name for c in table.columns}
        for t, table in db.metadata.tables.items()
    }
    assert migrated == declared
    sid = save_snapshot(4971, _snapshot_rows(), source=SOURCE_PASTE, engine=eng)
    assert sid == 1
    eng.dispose()


def test_normalize_database_url_handles_render_scheme():
    assert db.normalize_database_url("postgres://u:p@h/db").startswith(
        "postgresql://"
    )
    assert db.normalize_database_url("sqlite:///x.db") == "sqlite:///x.db"


# ---------------------------------------------------------------------------
# Tillagg: provenance (fingerprint) och komplett rad
# ---------------------------------------------------------------------------

def _current_round(streck_1, odds_1, source="textpaste"):
    from matchday_import import _make_key

    class Odds:
        def __init__(self, h, d, a):
            self.home, self.draw, self.away = h, d, a

    matches = [(f"Hem{i}", f"Bort{i}") for i in range(1, 14)]
    return {
        "matches": matches,
        "odds": {_make_key(h, a): [Odds(odds_1, 4.0, 6.0)] for h, a in matches},
        "streck": {
            _make_key(h, a): {"1": streck_1, "X": 100 - streck_1 - 20, "2": 20}
            for h, a in matches
        },
        "source": source,
    }


def _snapshot_market(engine, sid):
    with db.session_scope(engine) as s:
        return s.execute(
            select(
                db.market_snapshot_matches.c.streck_1,
                db.market_snapshot_matches.c.odds_1,
            ).where(db.market_snapshot_matches.c.snapshot_id == sid)
        ).all()


def test_new_market_data_never_links_row_to_old_snapshot(engine, monkeypatch):
    """Regressionstest: current_round A -> snapshot A -> current_round B ->
    registrera rad utan manuellt snapshot => nytt snapshot B, aldrig A."""
    import archive_ui
    monkeypatch.setattr(archive_ui.st, "session_state", {})

    round_a = _current_round(streck_1=55, odds_1=1.5)
    archive_ui.st.session_state["current_round"] = round_a
    sid_a = save_snapshot(
        4971, archive_ui.snapshot_rows_from_current_round(round_a),
        source=SOURCE_PASTE, engine=engine,
    )
    archive_ui.remember_snapshot(
        4971, sid_a, SOURCE_PASTE, archive_ui.current_round_fingerprint(),
    )
    assert archive_ui.remembered_snapshot_id(4971) == sid_a

    round_b = _current_round(streck_1=40, odds_1=2.2)
    archive_ui.st.session_state["current_round"] = round_b

    sid_b = archive_ui.ensure_snapshot_for_current_round(4971, round_b, engine=engine)
    assert sid_b != sid_a
    pid = register_played_system(4971, _played_rows(), sid_b, engine=engine)
    with db.session_scope(engine) as s:
        linked = s.execute(
            select(db.played_systems.c.snapshot_id)
            .where(db.played_systems.c.id == pid)
        ).scalar()
    assert linked == sid_b
    assert all((s1, o1) == (40.0, 2.2) for s1, o1 in _snapshot_market(engine, sid_b))
    assert all((s1, o1) == (55.0, 1.5) for s1, o1 in _snapshot_market(engine, sid_a))

    # Samma data igen => samma snapshot, inget nytt.
    assert archive_ui.ensure_snapshot_for_current_round(
        4971, round_b, engine=engine,
    ) == sid_b
    assert _count(engine, "market_snapshots") == 2


def test_remembered_snapshot_without_fingerprint_is_not_reused(
    engine, monkeypatch,
):
    """Snapshot sparat utan kand fingerprint (t.ex. fran Arkiv-sidan) ateranvands
    inte for en rad byggd pa sessionsdata."""
    import archive_ui
    monkeypatch.setattr(archive_ui.st, "session_state", {})
    sid = save_snapshot(4971, _snapshot_rows(), source=SOURCE_API, engine=engine)
    archive_ui.remember_snapshot(4971, sid, SOURCE_API)
    rnd = _current_round(streck_1=50, odds_1=1.8)
    assert archive_ui.remembered_snapshot_id(4971) == sid
    assert archive_ui.ensure_snapshot_for_current_round(4971, rnd, engine=engine) != sid


def test_snapshot_fingerprint_changes_with_market_data():
    import archive_ui
    a = archive_ui.snapshot_rows_from_current_round(_current_round(55, 1.5))
    b = archive_ui.snapshot_rows_from_current_round(_current_round(56, 1.5))
    c = archive_ui.snapshot_rows_from_current_round(_current_round(55, 1.51))
    fa = archive_ui.snapshot_fingerprint(a)
    assert fa == archive_ui.snapshot_fingerprint(list(reversed(a)))
    assert fa != archive_ui.snapshot_fingerprint(b)
    assert fa != archive_ui.snapshot_fingerprint(c)


@pytest.mark.parametrize("bad_rows, message", [
    (_played_rows()[:12], "exakt 13"),
    (_played_rows()[:12] + [dict(_played_rows()[11])], "Dubblettpositioner"),
    ([dict(r, position=r["position"] + 1) for r in _played_rows()], "saknar match"),
])
def test_register_played_system_rejects_incomplete_rows(engine, bad_rows, message):
    sid = save_snapshot(4971, _snapshot_rows(), source=SOURCE_PASTE, engine=engine)
    with pytest.raises(fetch.IncompleteRowError, match=message):
        register_played_system(4971, bad_rows, sid, engine=engine)
    assert _count(engine, "played_systems") == 0


def test_register_played_system_rejects_position_gap(engine):
    sid = save_snapshot(4971, _snapshot_rows(), source=SOURCE_PASTE, engine=engine)
    rows = _played_rows()
    rows[12]["position"] = 14
    with pytest.raises(fetch.IncompleteRowError, match="saknar match \\[13\\]"):
        register_played_system(4971, rows, sid, engine=engine)


@pytest.mark.parametrize("signs", ["", "?", "3", "H", "1-X"])
def test_register_played_system_rejects_invalid_signs(engine, signs):
    sid = save_snapshot(4971, _snapshot_rows(), source=SOURCE_PASTE, engine=engine)
    rows = _played_rows()
    rows[4]["played_signs"] = signs
    with pytest.raises(ValueError, match="ogiltiga tecken"):
        register_played_system(4971, rows, sid, engine=engine)
    assert _count(engine, "played_systems") == 0


def test_register_played_system_rejects_snapshot_with_12_rows(engine):
    sid = save_snapshot(4971, _snapshot_rows()[:12], source=SOURCE_PASTE, engine=engine)
    with pytest.raises(fetch.IncompleteRowError, match="12 matchrader"):
        register_played_system(4971, _played_rows(), sid, engine=engine)
    assert _count(engine, "played_systems") == 0
