"""Tester for append-only snapshotarkivet."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snapshot_storage import (  # noqa: E402
    CAPTURED_AT_PRECISIONS,
    PAYOUT_TIERS,
    build_result,
    build_snapshot,
    date_only_timestamp,
    list_snapshots,
    load_result,
    load_snapshot,
    matches_from_dataframe,
    matches_from_matchday_matches,
    result_json,
    save_result,
    save_snapshot,
    snapshot_filename,
    snapshot_json,
)

# Retroaktiv testomgang: draw okant, endast datum kant, kalla manual.
RETROACTIVE_COUPON = [
    ("Blackpool", "Burton", 63, 24, 23, 2.32, 3.35, 2.95),
    ("Exeter", "Leyton Orient", 38, 25, 37, 2.38, 3.40, 2.88),
    ("Reading", "Wigan", 59, 22, 19, 2.17, 3.30, 3.35),
    ("Stockport", "Wimbledon", 67, 17, 16, 1.88, 4.00, 3.60),
    ("Wycombe", "Port Vale", 66, 18, 16, 1.60, 3.90, 5.50),
    ("Barnet", "Cambridge", 34, 28, 38, 2.55, 3.10, 2.70),
    ("Bristol Rovers", "Accrington", 61, 20, 19, 1.85, 3.40, 3.95),
    ("Colchester", "Walsall", 43, 28, 29, 2.20, 3.05, 3.25),
    ("Crawley Town", "Gillingham", 42, 28, 30, 2.35, 3.35, 2.75),
    ("Crewe", "Oldham", 41, 27, 32, 2.35, 3.20, 2.85),
    ("Newport", "Shrewsbury", 42, 26, 32, 2.43, 3.05, 2.85),
    ("Salford", "MK Dons", 35, 26, 39, 2.70, 3.25, 2.43),
    ("Swindon", "Fleetwood", 63, 21, 16, 1.98, 3.35, 3.50),
]


def _retroactive_rows():
    return [
        {
            "home_team": home,
            "away_team": away,
            "streck_1": s1,
            "streck_x": sx,
            "streck_2": s2,
            "odds_1": o1,
            "odds_x": ox,
            "odds_2": o2,
        }
        for home, away, s1, sx, s2, o1, ox, o2 in RETROACTIVE_COUPON
    ]


def _sample_matches(count: int = 2):
    return [
        {
            "home_team": f"Home{index}",
            "away_team": f"Away{index}",
            "streck_1": 50.0,
            "streck_x": 25.0,
            "streck_2": 25.0,
            "odds_1": 2.0,
            "odds_x": 3.4,
            "odds_2": 3.6,
        }
        for index in range(1, count + 1)
    ]


def test_snapshot_written_with_all_fields_including_precision(tmp_path):
    snapshot = build_snapshot(
        _sample_matches(),
        source="image_scan",
        draw=4966,
        captured_at="2026-08-24T18:30:00Z",
        note="T-60",
    )
    path = save_snapshot(snapshot, tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    assert path.name == "4966_2026-08-24T18-30-00Z.json"
    assert raw["draw"] == 4966
    assert raw["captured_at"] == "2026-08-24T18:30:00Z"
    assert raw["captured_at_precision"] == "exact"
    assert raw["reg_close_time"] is None
    assert raw["source"] == "image_scan"
    assert raw["note"] == "T-60"
    assert len(raw["matches"]) == 2
    assert raw["matches"][0] == {
        "position": 1,
        "home_team": "Home1",
        "away_team": "Away1",
        "league": None,
        "streck_1": 50.0,
        "streck_x": 25.0,
        "streck_2": 25.0,
        "odds_1": 2.0,
        "odds_x": 3.4,
        "odds_2": 3.6,
    }


def test_captured_at_precision_is_mandatory_and_never_dropped(tmp_path):
    snapshot = build_snapshot(
        _sample_matches(), source="manual", captured_at_precision="unknown",
    )
    written = json.loads(
        save_snapshot(snapshot, tmp_path).read_text(encoding="utf-8")
    )

    assert "captured_at_precision" in written
    assert written["captured_at_precision"] == "unknown"


def test_existing_snapshot_is_never_overwritten_collision_gets_suffix(
    tmp_path,
):
    first = build_snapshot(
        _sample_matches(2),
        source="image_scan",
        draw=4966,
        captured_at="2026-08-24T18:30:00Z",
        note="forst",
    )
    second = build_snapshot(
        _sample_matches(3),
        source="image_scan",
        draw=4966,
        captured_at="2026-08-24T18:30:00Z",
        note="andra",
    )

    first_path = save_snapshot(first, tmp_path)
    second_path = save_snapshot(second, tmp_path)
    third_path = save_snapshot(second, tmp_path)

    assert first_path != second_path != third_path
    assert second_path.name == "4966_2026-08-24T18-30-00Z_2.json"
    assert third_path.name == "4966_2026-08-24T18-30-00Z_3.json"
    assert json.loads(first_path.read_text(encoding="utf-8"))["note"] == (
        "forst"
    )
    assert len(list(tmp_path.glob("*.json"))) == 3


def test_multiple_snapshots_per_draw_are_kept_side_by_side(tmp_path):
    for stamp, note in (
        ("2026-08-24T17:30:00Z", "T-60"),
        ("2026-08-24T18:20:00Z", "T-10"),
    ):
        save_snapshot(
            build_snapshot(
                _sample_matches(),
                source="image_scan",
                draw=4966,
                captured_at=stamp,
                note=note,
            ),
            tmp_path,
        )

    infos = list_snapshots(tmp_path)

    assert [info.note for info in infos] == ["T-10", "T-60"]
    assert {info.draw for info in infos} == {4966}
    assert all(info.readable for info in infos)


def test_no_delete_function_is_exposed():
    import snapshot_storage

    assert not [
        name
        for name in dir(snapshot_storage)
        if any(word in name for word in ("delete", "remove", "clear"))
    ]


def test_result_file_is_separate_and_leaves_snapshot_untouched(tmp_path):
    snapshot_dir = tmp_path / "snapshots"
    results_dir = tmp_path / "results"
    snapshot_path = save_snapshot(
        build_snapshot(
            _sample_matches(),
            source="image_scan",
            draw=4966,
            captured_at="2026-08-24T18:30:00Z",
        ),
        snapshot_dir,
    )
    before = snapshot_path.read_bytes()

    result = build_result(
        4966,
        ["1", "X", "2"] * 4 + ["1"],
        turnover=1000.0,
        payouts={"13": 250000.0},
        winners={"13": 4},
        entered_at="2026-08-26T09:00:00Z",
    )
    result_path = save_result(result, results_dir)
    raw = json.loads(result_path.read_text(encoding="utf-8"))

    assert result_path.name == "4966.json"
    assert result_path.parent != snapshot_path.parent
    assert snapshot_path.read_bytes() == before
    assert "correct_row" not in json.loads(before.decode("utf-8"))
    assert raw["entered_manually"] is True
    assert raw["turnover"] == 1000.0
    assert set(raw["payouts"]) == set(PAYOUT_TIERS)
    assert raw["payouts"]["12"] is None
    assert load_result(4966, results_dir).correct_row == result.correct_row


def test_result_row_rejects_signs_outside_1x2():
    with pytest.raises(ValueError, match="Ogiltigt tecken"):
        build_result(4966, ["1", "X", "3"])


def test_precision_accepts_only_the_three_allowed_values():
    assert CAPTURED_AT_PRECISIONS == ("exact", "date_only", "unknown")

    for precision in CAPTURED_AT_PRECISIONS:
        snapshot = build_snapshot(
            _sample_matches(), source="manual",
            captured_at_precision=precision,
        )
        assert snapshot.captured_at_precision == precision

    with pytest.raises(ValueError, match="captured_at_precision"):
        build_snapshot(
            _sample_matches(), source="manual",
            captured_at_precision="approximate",
        )


def test_source_is_restricted_and_matches_must_be_non_empty():
    with pytest.raises(ValueError, match="source"):
        build_snapshot(_sample_matches(), source="svenskaspel")

    with pytest.raises(ValueError, match="minst en match"):
        build_snapshot([], source="manual")


def test_retroactive_manual_entry_with_date_only(tmp_path):
    snapshot = build_snapshot(
        _retroactive_rows(),
        source="manual",
        captured_at_precision="date_only",
        captured_at=date_only_timestamp("2026-08-22"),
        note="retroaktiv inmatning",
    )
    path = save_snapshot(snapshot, tmp_path)
    reloaded = load_snapshot(path)

    assert path.name == "unknown_2026-08-22T00-00-00Z.json"
    assert reloaded.draw is None
    assert reloaded.captured_at == "2026-08-22T00:00:00Z"
    assert reloaded.captured_at_precision == "date_only"
    assert reloaded.source == "manual"
    assert len(reloaded.matches) == 13
    assert [match.position for match in reloaded.matches] == list(
        range(1, 14)
    )
    assert reloaded.matches[0].home_team == "Blackpool"
    assert reloaded.matches[0].streck_1 == 63
    assert reloaded.matches[12].odds_2 == 3.50


def test_download_payload_is_identical_to_the_written_file(tmp_path):
    snapshot = build_snapshot(
        _retroactive_rows(),
        source="manual",
        captured_at_precision="date_only",
        captured_at=date_only_timestamp("2026-08-22"),
    )
    payload = snapshot_json(snapshot)
    path = save_snapshot(snapshot, tmp_path)

    assert path.name == snapshot_filename(snapshot)
    assert path.read_text(encoding="utf-8") == payload


def test_result_payload_helper_matches_written_result_file(tmp_path):
    result = build_result(4966, ["1"] * 13, entered_at="2026-08-26T09:00:00Z")
    path = save_result(result, tmp_path)

    assert path.read_text(encoding="utf-8") == result_json(result)


def test_matches_from_dataframe_skips_incomplete_rows_and_renumbers():
    frame = pd.DataFrame([
        {
            "HomeTeam": "Blackpool", "AwayTeam": "Burton",
            "Streck1": 63, "StreckX": 24, "Streck2": 23,
            "Odds1": 2.32, "OddsX": 3.35, "Odds2": 2.95,
        },
        {
            "HomeTeam": "", "AwayTeam": "Wigan",
            "Streck1": None, "StreckX": None, "Streck2": None,
            "Odds1": None, "OddsX": None, "Odds2": None,
        },
        {
            "HomeTeam": "Reading", "AwayTeam": "Wigan",
            "Streck1": float("nan"), "StreckX": 22, "Streck2": 19,
            "Odds1": 2.17, "OddsX": 3.30, "Odds2": 3.35,
        },
    ])

    matches = matches_from_dataframe(frame)

    assert [match.position for match in matches] == [1, 2]
    assert [match.home_team for match in matches] == ["Blackpool", "Reading"]
    assert matches[1].streck_1 is None


def test_matches_from_matchday_matches_uses_first_odds_entry():
    class FakeOdds:
        def __init__(self, home, draw, away):
            self.home = home
            self.draw = draw
            self.away = away

    class FakeMatch:
        def __init__(self, home, away, entries, streck):
            self.home_team = home
            self.away_team = away
            self.odds_entries = entries
            self.streck = streck

    matches = matches_from_matchday_matches([
        FakeMatch(
            "Reading", "Wigan",
            [FakeOdds(2.17, 3.30, 3.35), FakeOdds(2.20, 3.20, 3.40)],
            {"1": 59.0, "X": 22.0, "2": 19.0},
        ),
        FakeMatch("Crewe", "Oldham", [], None),
    ])

    assert matches[0].odds_1 == 2.17
    assert matches[0].streck_x == 22.0
    assert matches[1].odds_1 is None
    assert matches[1].streck_2 is None


def test_saved_matchday_functionality_is_untouched(tmp_path):
    from matchday_storage import (
        DEFAULT_STORAGE_PATH,
        get_saved_matchday_status,
        load_saved_matchday_data,
        save_matchday_data,
    )
    import snapshot_storage

    assert DEFAULT_STORAGE_PATH == Path("data") / "saved_matchday.json"
    assert snapshot_storage.DEFAULT_SNAPSHOT_DIR != DEFAULT_STORAGE_PATH.parent

    path = tmp_path / "saved_matchday.json"
    fixtures = [
        {"home_team": "Reading", "away_team": "Wigan",
         "match_key": "Reading_Wigan"},
    ]
    streck = {"Reading_Wigan": {"1": 59.0, "X": 22.0, "2": 19.0}}

    assert save_matchday_data(fixtures, {}, streck, path=path) is True

    loaded = load_saved_matchday_data(path=path)
    assert loaded is not None
    status = get_saved_matchday_status(path=path)
    assert status.exists is True
    assert status.match_count == 1

    # Snapshot-skrivning ror inte den sparade omgangen.
    before = path.read_bytes()
    save_snapshot(
        build_snapshot(_sample_matches(), source="paste"),
        tmp_path / "snapshots",
    )
    assert path.read_bytes() == before


def test_list_snapshots_reports_unreadable_files_without_raising(tmp_path):
    (tmp_path / "4966_broken.json").write_text("{not json", encoding="utf-8")
    save_snapshot(
        build_snapshot(
            _sample_matches(), source="paste",
            captured_at="2026-08-24T18:30:00Z",
        ),
        tmp_path,
    )

    infos = list_snapshots(tmp_path)

    assert len(infos) == 2
    assert sorted(info.readable for info in infos) == [False, True]


def test_list_snapshots_returns_empty_list_for_missing_directory(tmp_path):
    assert list_snapshots(tmp_path / "missing") == []
