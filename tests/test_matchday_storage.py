"""
Tester for matchday_storage.py -- spara, ladda, rensa och status for omgangsdata.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from matchday_storage import (
    SavedMatchdayStatus,
    save_matchday_data,
    load_saved_matchday_data,
    clear_saved_matchday_data,
    get_saved_matchday_status,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_fixtures():
    """Returnerar exempelfixtures som dicts (serialiserade)."""
    return [
        {"home_team": "Arsenal", "away_team": "Liverpool", "match_key": "arsenal_liverpool"},
        {"home_team": "Man City", "away_team": "Chelsea", "match_key": "man city_chelsea"},
    ]


def _sample_odds_by_key():
    """Returnerar exempelodds som dicts (serialiserade)."""
    return {
        "arsenal_liverpool": [
            {"bookmaker": "B365", "home": 2.10, "draw": 3.40, "away": 3.60},
        ],
        "man city_chelsea": [
            {"bookmaker": "B365", "home": 1.50, "draw": 4.50, "away": 6.50},
        ],
    }


def _sample_streck_by_key():
    """Returnerar exempelstreck."""
    return {
        "arsenal_liverpool": {"1": 45.0, "X": 28.0, "2": 27.0},
        "man city_chelsea": {"1": 55.0, "X": 25.0, "2": 20.0},
    }


# ---------------------------------------------------------------------------
# test: save
# ---------------------------------------------------------------------------

class TestSaveMatchdayData:
    def test_save_creates_file(self, tmp_path):
        path = tmp_path / "saved.json"
        result = save_matchday_data(
            _sample_fixtures(), _sample_odds_by_key(), _sample_streck_by_key(),
            path=path,
        )
        assert result is True
        assert path.exists()

    def test_save_writes_valid_json(self, tmp_path):
        path = tmp_path / "saved.json"
        save_matchday_data(
            _sample_fixtures(), _sample_odds_by_key(), _sample_streck_by_key(),
            path=path,
        )
        with open(path) as f:
            data = json.load(f)
        assert data["version"] == 1
        assert len(data["fixtures"]) == 2
        assert "saved_at" in data
        assert data["meta"]["match_count"] == 2
        assert data["meta"]["odds_count"] == 2
        assert data["meta"]["streck_count"] == 2

    def test_save_without_odds_or_streck(self, tmp_path):
        path = tmp_path / "saved.json"
        result = save_matchday_data(
            _sample_fixtures(), {}, {},
            path=path,
        )
        assert result is True
        with open(path) as f:
            data = json.load(f)
        assert data["meta"]["odds_count"] == 0
        assert data["meta"]["streck_count"] == 0


# ---------------------------------------------------------------------------
# test: load
# ---------------------------------------------------------------------------

class TestLoadMatchdayData:
    def test_load_returns_none_when_no_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        assert load_saved_matchday_data(path=path) is None

    def test_load_after_save(self, tmp_path):
        path = tmp_path / "saved.json"
        save_matchday_data(
            _sample_fixtures(), _sample_odds_by_key(), _sample_streck_by_key(),
            path=path,
        )
        loaded = load_saved_matchday_data(path=path)
        assert loaded is not None
        fixtures, odds, streck, meta = loaded
        assert len(fixtures) == 2
        assert len(odds) == 2
        assert len(streck) == 2
        assert meta["match_count"] == 2
        assert "saved_at" in meta

    def test_load_corrupted_json(self, tmp_path):
        path = tmp_path / "corrupted.json"
        path.write_text("not valid json {{{", encoding="utf-8")
        assert load_saved_matchday_data(path=path) is None

    def test_load_wrong_version(self, tmp_path):
        path = tmp_path / "wrong_version.json"
        path.write_text(json.dumps({"version": 999}), encoding="utf-8")
        assert load_saved_matchday_data(path=path) is None

    def test_load_missing_fields(self, tmp_path):
        path = tmp_path / "missing.json"
        path.write_text(json.dumps({"version": 1}), encoding="utf-8")
        assert load_saved_matchday_data(path=path) is None

    def test_load_invalid_fixtures_type(self, tmp_path):
        path = tmp_path / "bad_fixtures.json"
        path.write_text(json.dumps({
            "version": 1,
            "fixtures": "not a list",
            "odds_by_key": {},
            "streck_by_key": {},
        }), encoding="utf-8")
        assert load_saved_matchday_data(path=path) is None


# ---------------------------------------------------------------------------
# test: clear
# ---------------------------------------------------------------------------

class TestClearMatchdayData:
    def test_clear_removes_file(self, tmp_path):
        path = tmp_path / "saved.json"
        save_matchday_data(
            _sample_fixtures(), _sample_odds_by_key(), _sample_streck_by_key(),
            path=path,
        )
        assert path.exists()
        result = clear_saved_matchday_data(path=path)
        assert result is True
        assert not path.exists()

    def test_clear_when_no_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        result = clear_saved_matchday_data(path=path)
        assert result is True


# ---------------------------------------------------------------------------
# test: status
# ---------------------------------------------------------------------------

class TestGetMatchdayStatus:
    def test_status_no_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        status = get_saved_matchday_status(path=path)
        assert status.exists is False
        assert status.match_count == 0

    def test_status_after_save(self, tmp_path):
        path = tmp_path / "saved.json"
        save_matchday_data(
            _sample_fixtures(), _sample_odds_by_key(), _sample_streck_by_key(),
            path=path,
        )
        status = get_saved_matchday_status(path=path)
        assert status.exists is True
        assert status.match_count == 2
        assert status.has_odds is True
        assert status.has_streck is True
        assert status.odds_count == 2
        assert status.streck_count == 2
        assert status.saved_at is not None
        assert status.source == "sparad"

    def test_status_after_clear(self, tmp_path):
        path = tmp_path / "saved.json"
        save_matchday_data(
            _sample_fixtures(), _sample_odds_by_key(), _sample_streck_by_key(),
            path=path,
        )
        clear_saved_matchday_data(path=path)
        status = get_saved_matchday_status(path=path)
        assert status.exists is False

    def test_status_corrupted_file(self, tmp_path):
        path = tmp_path / "corrupted.json"
        path.write_text("not json", encoding="utf-8")
        status = get_saved_matchday_status(path=path)
        assert status.exists is False

    def test_status_without_odds_or_streck(self, tmp_path):
        path = tmp_path / "saved.json"
        save_matchday_data(_sample_fixtures(), {}, {}, path=path)
        status = get_saved_matchday_status(path=path)
        assert status.exists is True
        assert status.has_odds is False
        assert status.has_streck is False


# ---------------------------------------------------------------------------
# test: replace (ny import ersatter gammal)
# ---------------------------------------------------------------------------

class TestReplaceMatchdayData:
    def test_new_save_replaces_old(self, tmp_path):
        path = tmp_path / "saved.json"

        # Forsta sparning
        save_matchday_data(
            _sample_fixtures(), _sample_odds_by_key(), _sample_streck_by_key(),
            path=path,
        )
        status1 = get_saved_matchday_status(path=path)
        assert status1.match_count == 2

        # Ny sparning med andra data
        new_fixtures = [
            {"home_team": "Brighton", "away_team": "Newcastle", "match_key": "brighton_newcastle"},
        ]
        save_matchday_data(new_fixtures, {}, {}, path=path)
        status2 = get_saved_matchday_status(path=path)
        assert status2.match_count == 1
        assert status2.has_odds is False

        # Ladda och verifiera
        loaded = load_saved_matchday_data(path=path)
        assert loaded is not None
        fixtures, odds, streck, meta = loaded
        assert len(fixtures) == 1
