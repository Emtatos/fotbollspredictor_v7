"""
Tester for matchday_import.py -- importflode for aktuell omgang.

Testar:
- fixtures-import: validering, saknade kolumner, tomma lagnamn
- odds-import: validering, flera bookmaker-format, ogiltiga varden
- streck-import: validering, saknade kolumner, negativa varden
- matchning mellan fixtures, odds och streck
- ofullstandig data hanteras defensivt
- importstatus byggs korrekt
- fallback-vagar (manuell input, historik) inte bryts
"""

import pytest
import pandas as pd

from matchday_import import (
    parse_fixtures_csv,
    parse_fixture_lines,
    fetch_odds_for_fixtures,
    ParseFixtureLinesResult,
    parse_odds_csv,
    parse_streck_csv,
    match_matchday_data,
    MatchdayFixture,
    MatchdayImportStatus,
    MatchdayMatch,
    generate_fixtures_template,
    generate_odds_template,
    generate_streck_template,
    FIXTURES_TEMPLATE_CSV,
    ODDS_TEMPLATE_CSV,
    STRECK_TEMPLATE_CSV,
)
from odds_tool import OddsEntry


# ---------------------------------------------------------------------------
# parse_fixtures_csv
# ---------------------------------------------------------------------------

class TestParseFixturesCsv:
    def test_valid_fixtures(self):
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal", "Man City"],
            "AwayTeam": ["Liverpool", "Chelsea"],
        })
        fixtures, errors = parse_fixtures_csv(df)
        assert len(fixtures) == 2
        assert len(errors) == 0
        assert fixtures[0].home_team == "Arsenal"
        assert fixtures[0].away_team == "Liverpool"
        assert "_" in fixtures[0].match_key

    def test_missing_columns(self):
        df = pd.DataFrame({
            "Home": ["Arsenal"],
            "Away": ["Liverpool"],
        })
        fixtures, errors = parse_fixtures_csv(df)
        assert len(fixtures) == 0
        assert any("Saknade kolumner" in e for e in errors)

    def test_empty_team_names_skipped(self):
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal", "", "Brighton"],
            "AwayTeam": ["Liverpool", "Chelsea", "Newcastle"],
        })
        fixtures, errors = parse_fixtures_csv(df)
        assert len(fixtures) == 2
        assert any("tomt lagnamn" in e for e in errors)

    def test_empty_dataframe(self):
        df = pd.DataFrame({"HomeTeam": [], "AwayTeam": []})
        fixtures, errors = parse_fixtures_csv(df)
        assert len(fixtures) == 0
        assert any("Inga giltiga" in e for e in errors)

    def test_with_extra_columns(self):
        """Extra kolumner ignoreras utan fel."""
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Liverpool"],
            "Date": ["2025-03-15"],
            "League": ["E0"],
        })
        fixtures, errors = parse_fixtures_csv(df)
        assert len(fixtures) == 1
        assert len(errors) == 0


# ---------------------------------------------------------------------------
# parse_odds_csv
# ---------------------------------------------------------------------------

class TestParseOddsCsv:
    def test_valid_b365_odds(self):
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal", "Man City"],
            "AwayTeam": ["Liverpool", "Chelsea"],
            "B365H": [2.10, 1.50],
            "B365D": [3.40, 4.50],
            "B365A": [3.60, 6.50],
        })
        odds_by_key, valid_rows, errors = parse_odds_csv(df)
        assert valid_rows == 2
        assert len(odds_by_key) == 2

    def test_simple_format(self):
        """Home/Draw/Away-format stods."""
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Liverpool"],
            "Home": [2.10],
            "Draw": [3.40],
            "Away": [3.60],
        })
        odds_by_key, valid_rows, errors = parse_odds_csv(df)
        assert valid_rows == 1
        assert len(odds_by_key) == 1

    def test_multiple_bookmakers(self):
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Liverpool"],
            "B365H": [2.10],
            "B365D": [3.40],
            "B365A": [3.60],
            "PSH": [2.15],
            "PSD": [3.35],
            "PSA": [3.65],
        })
        odds_by_key, valid_rows, errors = parse_odds_csv(df)
        assert valid_rows == 1
        # Should have 2 OddsEntry per match
        key = list(odds_by_key.keys())[0]
        assert len(odds_by_key[key]) == 2

    def test_missing_team_columns(self):
        df = pd.DataFrame({
            "B365H": [2.10],
            "B365D": [3.40],
            "B365A": [3.60],
        })
        odds_by_key, valid_rows, errors = parse_odds_csv(df)
        assert len(odds_by_key) == 0
        assert any("Saknade kolumner" in e for e in errors)

    def test_no_odds_columns(self):
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Liverpool"],
            "SomeCol": [42],
        })
        odds_by_key, valid_rows, errors = parse_odds_csv(df)
        assert len(odds_by_key) == 0
        assert any("Inga kanda oddskolumner" in e for e in errors)

    def test_invalid_odds_values(self):
        """Ogiltiga odds (<=1.0) genererar felmeddelande."""
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Liverpool"],
            "B365H": [0.5],
            "B365D": [3.40],
            "B365A": [3.60],
        })
        odds_by_key, valid_rows, errors = parse_odds_csv(df)
        assert valid_rows == 0
        assert any("inga giltiga odds" in e for e in errors)

    def test_empty_team_in_odds(self):
        df = pd.DataFrame({
            "HomeTeam": [""],
            "AwayTeam": ["Liverpool"],
            "B365H": [2.10],
            "B365D": [3.40],
            "B365A": [3.60],
        })
        odds_by_key, valid_rows, errors = parse_odds_csv(df)
        assert valid_rows == 0
        assert any("tomt lagnamn" in e for e in errors)


# ---------------------------------------------------------------------------
# parse_streck_csv
# ---------------------------------------------------------------------------

class TestParseStreckCsv:
    def test_valid_streck(self):
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal", "Man City"],
            "AwayTeam": ["Liverpool", "Chelsea"],
            "Streck1": [45, 55],
            "StreckX": [28, 25],
            "Streck2": [27, 20],
        })
        streck_by_key, valid_rows, errors = parse_streck_csv(df)
        assert valid_rows == 2
        assert len(streck_by_key) == 2

    def test_missing_columns(self):
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Liverpool"],
            "Streck1": [45],
        })
        streck_by_key, valid_rows, errors = parse_streck_csv(df)
        assert len(streck_by_key) == 0
        assert any("Saknade kolumner" in e for e in errors)

    def test_negative_values_skipped(self):
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal", "Man City"],
            "AwayTeam": ["Liverpool", "Chelsea"],
            "Streck1": [45, -5],
            "StreckX": [28, 55],
            "Streck2": [27, 50],
        })
        streck_by_key, valid_rows, errors = parse_streck_csv(df)
        assert valid_rows == 1
        assert any("negativa" in e for e in errors)

    def test_non_numeric_values_skipped(self):
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal", "Man City"],
            "AwayTeam": ["Liverpool", "Chelsea"],
            "Streck1": [45, "abc"],
            "StreckX": [28, 25],
            "Streck2": [27, 20],
        })
        streck_by_key, valid_rows, errors = parse_streck_csv(df)
        assert valid_rows == 1
        assert any("ogiltiga" in e for e in errors)

    def test_zero_sum_skipped(self):
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Liverpool"],
            "Streck1": [0],
            "StreckX": [0],
            "Streck2": [0],
        })
        streck_by_key, valid_rows, errors = parse_streck_csv(df)
        assert valid_rows == 0
        assert any("strecksumma" in e for e in errors)

    def test_empty_team_in_streck(self):
        df = pd.DataFrame({
            "HomeTeam": [""],
            "AwayTeam": ["Liverpool"],
            "Streck1": [45],
            "StreckX": [28],
            "Streck2": [27],
        })
        streck_by_key, valid_rows, errors = parse_streck_csv(df)
        assert valid_rows == 0
        assert any("tomt lagnamn" in e for e in errors)


# ---------------------------------------------------------------------------
# match_matchday_data
# ---------------------------------------------------------------------------

class TestMatchMatchdayData:
    def test_full_match(self):
        """Alla tre dataset matchar korrekt."""
        fixtures = [
            MatchdayFixture("Arsenal", "Liverpool", "Arsenal_Liverpool"),
            MatchdayFixture("Man City", "Chelsea", "Man City_Chelsea"),
        ]
        odds_by_key = {
            "Arsenal_Liverpool": [OddsEntry("Bet365", 2.10, 3.40, 3.60)],
            "Man City_Chelsea": [OddsEntry("Bet365", 1.50, 4.50, 6.50)],
        }
        streck_by_key = {
            "Arsenal_Liverpool": {"1": 45, "X": 28, "2": 27},
            "Man City_Chelsea": {"1": 55, "X": 25, "2": 20},
        }
        matches, status = match_matchday_data(fixtures, odds_by_key, streck_by_key)
        assert status.fixtures_count == 2
        assert status.odds_matched == 2
        assert status.streck_matched == 2
        assert status.fully_matched == 2
        assert len(status.fixtures_without_odds) == 0
        assert len(status.fixtures_without_streck) == 0

    def test_partial_odds(self):
        """Bara en fixture har odds."""
        fixtures = [
            MatchdayFixture("Arsenal", "Liverpool", "Arsenal_Liverpool"),
            MatchdayFixture("Man City", "Chelsea", "Man City_Chelsea"),
        ]
        odds_by_key = {
            "Arsenal_Liverpool": [OddsEntry("Bet365", 2.10, 3.40, 3.60)],
        }
        matches, status = match_matchday_data(fixtures, odds_by_key, {})
        assert status.odds_matched == 1
        assert status.streck_matched == 0
        assert status.fully_matched == 0
        assert len(status.fixtures_without_odds) == 1
        assert "Man City vs Chelsea" in status.fixtures_without_odds[0]

    def test_partial_streck(self):
        """Bara en fixture har streck."""
        fixtures = [
            MatchdayFixture("Arsenal", "Liverpool", "Arsenal_Liverpool"),
            MatchdayFixture("Man City", "Chelsea", "Man City_Chelsea"),
        ]
        streck_by_key = {
            "Arsenal_Liverpool": {"1": 45, "X": 28, "2": 27},
        }
        matches, status = match_matchday_data(fixtures, {}, streck_by_key)
        assert status.streck_matched == 1
        assert len(status.fixtures_without_streck) == 1

    def test_unmatched_odds_detected(self):
        """Odds-rader som inte matchar nagon fixture rapporteras."""
        fixtures = [
            MatchdayFixture("Arsenal", "Liverpool", "Arsenal_Liverpool"),
        ]
        odds_by_key = {
            "Arsenal_Liverpool": [OddsEntry("Bet365", 2.10, 3.40, 3.60)],
            "Brighton_Newcastle": [OddsEntry("Bet365", 2.60, 3.30, 2.80)],
        }
        matches, status = match_matchday_data(fixtures, odds_by_key, {})
        assert "Brighton_Newcastle" in status.unmatched_odds

    def test_empty_fixtures(self):
        """Tom fixtures-lista hanteras gracefully."""
        matches, status = match_matchday_data([], {}, {})
        assert status.fixtures_count == 0
        assert status.fully_matched == 0

    def test_odds_report_built(self):
        """MatchOddsReport byggs for matcher med odds."""
        fixtures = [
            MatchdayFixture("Arsenal", "Liverpool", "Arsenal_Liverpool"),
        ]
        odds_by_key = {
            "Arsenal_Liverpool": [OddsEntry("Bet365", 2.10, 3.40, 3.60)],
        }
        matches, status = match_matchday_data(fixtures, odds_by_key, {})
        assert len(matches) == 1
        assert matches[0].odds_report is not None
        assert matches[0].odds_report.home_team == "Arsenal"

    def test_streck_report_built_when_both_present(self):
        """StreckReport byggs nar bade odds och streck matchar."""
        fixtures = [
            MatchdayFixture("Arsenal", "Liverpool", "Arsenal_Liverpool"),
        ]
        odds_by_key = {
            "Arsenal_Liverpool": [OddsEntry("Bet365", 2.10, 3.40, 3.60)],
        }
        streck_by_key = {
            "Arsenal_Liverpool": {"1": 45, "X": 28, "2": 27},
        }
        matches, status = match_matchday_data(fixtures, odds_by_key, streck_by_key)
        assert matches[0].streck_report is not None

    def test_value_report_built(self):
        """ValueReport byggs for matcher med odds."""
        fixtures = [
            MatchdayFixture("Arsenal", "Liverpool", "Arsenal_Liverpool"),
        ]
        odds_by_key = {
            "Arsenal_Liverpool": [
                OddsEntry("Bet365", 2.10, 3.40, 3.60),
                OddsEntry("Pinnacle", 2.15, 3.35, 3.65),
            ],
        }
        matches, status = match_matchday_data(fixtures, odds_by_key, {})
        # With 2 bookmakers, value report should be built
        assert matches[0].value_report is not None


# ---------------------------------------------------------------------------
# Import status
# ---------------------------------------------------------------------------

class TestImportStatus:
    def test_status_counts_correct(self):
        fixtures = [
            MatchdayFixture("A", "B", "A_B"),
            MatchdayFixture("C", "D", "C_D"),
            MatchdayFixture("E", "F", "E_F"),
        ]
        odds_by_key = {
            "A_B": [OddsEntry("Bet365", 2.10, 3.40, 3.60)],
            "C_D": [OddsEntry("Bet365", 1.50, 4.50, 6.50)],
        }
        streck_by_key = {
            "A_B": {"1": 45, "X": 28, "2": 27},
        }
        matches, status = match_matchday_data(fixtures, odds_by_key, streck_by_key)
        assert status.fixtures_count == 3
        assert status.odds_matched == 2
        assert status.streck_matched == 1
        assert status.fully_matched == 1
        assert len(status.fixtures_without_odds) == 1
        assert len(status.fixtures_without_streck) == 2

    def test_status_default_values(self):
        status = MatchdayImportStatus()
        assert status.fixtures_count == 0
        assert status.odds_matched == 0
        assert status.streck_matched == 0
        assert status.fully_matched == 0
        assert status.errors == []
        assert status.unmatched_odds == []
        assert status.unmatched_streck == []


# ---------------------------------------------------------------------------
# CSV-mallar
# ---------------------------------------------------------------------------

class TestCsvTemplates:
    def test_fixtures_template_parseable(self):
        """Fixtures-mallen kan parsas till giltig DataFrame."""
        import io
        df = pd.read_csv(io.StringIO(FIXTURES_TEMPLATE_CSV))
        assert "HomeTeam" in df.columns
        assert "AwayTeam" in df.columns
        assert len(df) > 0

    def test_odds_template_parseable(self):
        """Odds-mallen kan parsas till giltig DataFrame."""
        import io
        df = pd.read_csv(io.StringIO(ODDS_TEMPLATE_CSV))
        assert "HomeTeam" in df.columns
        assert "AwayTeam" in df.columns
        assert "B365H" in df.columns
        assert len(df) > 0

    def test_streck_template_parseable(self):
        """Streck-mallen kan parsas till giltig DataFrame."""
        import io
        df = pd.read_csv(io.StringIO(STRECK_TEMPLATE_CSV))
        assert "HomeTeam" in df.columns
        assert "AwayTeam" in df.columns
        assert "Streck1" in df.columns
        assert "StreckX" in df.columns
        assert "Streck2" in df.columns
        assert len(df) > 0

    def test_generate_functions_return_strings(self):
        assert isinstance(generate_fixtures_template(), str)
        assert isinstance(generate_odds_template(), str)
        assert isinstance(generate_streck_template(), str)

    def test_templates_round_trip(self):
        """Mallar kan parsas och sedan valideras korrekt."""
        import io

        # Fixtures round-trip
        fix_df = pd.read_csv(io.StringIO(FIXTURES_TEMPLATE_CSV))
        fixtures, errors = parse_fixtures_csv(fix_df)
        assert len(fixtures) > 0
        assert len(errors) == 0

        # Odds round-trip
        odds_df = pd.read_csv(io.StringIO(ODDS_TEMPLATE_CSV))
        odds_by_key, valid_rows, errors = parse_odds_csv(odds_df)
        assert valid_rows > 0

        # Streck round-trip
        streck_df = pd.read_csv(io.StringIO(STRECK_TEMPLATE_CSV))
        streck_by_key, valid_rows, errors = parse_streck_csv(streck_df)
        assert valid_rows > 0


# ---------------------------------------------------------------------------
# Defensiv hantering
# ---------------------------------------------------------------------------

class TestDefensiveHandling:
    def test_incomplete_data_handled(self):
        """Matcher utan odds eller streck ska inte krascha."""
        fixtures = [
            MatchdayFixture("Arsenal", "Liverpool", "Arsenal_Liverpool"),
        ]
        matches, status = match_matchday_data(fixtures, {}, {})
        assert len(matches) == 1
        assert matches[0].odds_report is None
        assert matches[0].value_report is None
        assert matches[0].streck_report is None
        assert not matches[0].has_odds
        assert not matches[0].has_streck

    def test_nan_team_names_filtered(self):
        """NaN-lagnamn i DataFrame ska filtreras bort."""
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal", None],
            "AwayTeam": ["Liverpool", "Chelsea"],
        })
        fixtures, errors = parse_fixtures_csv(df)
        assert len(fixtures) == 1
        assert any("tomt lagnamn" in e for e in errors)

    def test_case_insensitive_matching(self):
        """Matchning ska fungera case-insensitive."""
        fixtures = [
            MatchdayFixture("Arsenal", "Liverpool", "Arsenal_Liverpool"),
        ]
        # Odds key with different casing
        odds_by_key = {
            "arsenal_liverpool": [OddsEntry("Bet365", 2.10, 3.40, 3.60)],
        }
        matches, status = match_matchday_data(fixtures, odds_by_key, {})
        assert status.odds_matched == 1


# ---------------------------------------------------------------------------
# Fallback-vagar
# ---------------------------------------------------------------------------

class TestFallbackPaths:
    def test_existing_streck_import_not_broken(self):
        """Befintlig auto_load_streck ska fortfarande fungera."""
        from streck_import import auto_load_streck
        # Should return gracefully when file doesn't exist
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            lookup, status = auto_load_streck(path=Path(tmpdir) / "missing.csv")
            assert lookup is None
            assert not status.loaded

    def test_existing_odds_tool_not_broken(self):
        """Befintlig odds_tool-logik ska fortfarande fungera."""
        from odds_tool import build_match_report, OddsEntry
        entries = [OddsEntry("Bet365", 2.10, 3.40, 3.60)]
        report = build_match_report("Arsenal", "Liverpool", entries)
        assert report is not None
        assert abs(sum(report.fair_probs.values()) - 1.0) < 1e-9

    def test_existing_value_analysis_not_broken(self):
        """Befintlig value_analysis-logik ska fortfarande fungera."""
        from value_analysis import build_value_report
        from odds_tool import build_match_report, OddsEntry
        entries = [
            OddsEntry("Bet365", 2.10, 3.40, 3.60),
            OddsEntry("Pinnacle", 2.15, 3.35, 3.65),
        ]
        report = build_match_report("Arsenal", "Liverpool", entries)
        value_rpt = build_value_report(report)
        assert value_rpt is not None

    def test_existing_streck_analysis_not_broken(self):
        """Befintlig streck_analysis-logik ska fortfarande fungera."""
        from streck_analysis import build_streck_report
        fair_probs = {"1": 0.45, "X": 0.28, "2": 0.27}
        streck_pcts = {"1": 40, "X": 30, "2": 30}
        report = build_streck_report("Arsenal", "Liverpool", fair_probs, streck_pcts)
        assert report is not None


# ---------------------------------------------------------------------------
# parse_fixture_lines (text-paste parser)
# ---------------------------------------------------------------------------

class TestParseFixtureLines:
    def test_hyphen_separator(self):
        """Parser hanterar vanligt bindestreck."""
        result = parse_fixture_lines("Leeds United - Brentford")
        assert result.valid_count == 1
        assert result.fixtures[0].home_team == "Leeds United"
        assert result.fixtures[0].away_team == "Brentford"

    def test_en_dash_separator(self):
        """Parser hanterar tankstreck (en-dash)."""
        result = parse_fixture_lines("Everton \u2013 Chelsea")
        assert result.valid_count == 1
        assert result.fixtures[0].home_team == "Everton"
        assert result.fixtures[0].away_team == "Chelsea"

    def test_em_dash_separator(self):
        """Parser hanterar em-dash."""
        result = parse_fixture_lines("Fulham \u2014 Burnley")
        assert result.valid_count == 1
        assert result.fixtures[0].home_team == "Fulham"
        assert result.fixtures[0].away_team == "Burnley"

    def test_vs_separator(self):
        """Parser hanterar 'vs' separator."""
        result = parse_fixture_lines("Arsenal vs Liverpool")
        assert result.valid_count == 1
        assert result.fixtures[0].home_team == "Arsenal"
        assert result.fixtures[0].away_team == "Liverpool"

    def test_vs_dot_separator(self):
        """Parser hanterar 'vs.' separator."""
        result = parse_fixture_lines("Arsenal vs. Liverpool")
        assert result.valid_count == 1
        assert result.fixtures[0].home_team == "Arsenal"
        assert result.fixtures[0].away_team == "Liverpool"

    def test_multiple_matches(self):
        """Parser hanterar flera matcher."""
        text = "Leeds United - Brentford\nEverton \u2013 Chelsea\nFulham vs Burnley"
        result = parse_fixture_lines(text)
        assert result.valid_count == 3
        assert result.invalid_count == 0

    def test_empty_lines_ignored(self):
        """Tomma rader ska ignoreras."""
        text = "Arsenal - Liverpool\n\n\nChelsea - Everton\n"
        result = parse_fixture_lines(text)
        assert result.valid_count == 2
        assert result.blank_lines >= 2
        assert result.invalid_count == 0

    def test_invalid_lines_reported(self):
        """Ogiltiga rader rapporteras."""
        text = "Arsenal - Liverpool\nDetta ar ingen match\nChelsea - Everton"
        result = parse_fixture_lines(text)
        assert result.valid_count == 2
        assert result.invalid_count == 1
        assert "Detta ar ingen match" in result.invalid_lines

    def test_whitespace_trimmed(self):
        """Whitespace runt lagnamn trimmas."""
        result = parse_fixture_lines("  Arsenal  -  Liverpool  ")
        assert result.valid_count == 1
        assert result.fixtures[0].home_team == "Arsenal"
        assert result.fixtures[0].away_team == "Liverpool"

    def test_empty_input(self):
        """Tom input ger tomma resultat."""
        result = parse_fixture_lines("")
        assert result.valid_count == 0
        assert result.invalid_count == 0

    def test_match_key_built(self):
        """match_key byggs korrekt fran lagnamn."""
        result = parse_fixture_lines("Arsenal - Liverpool")
        assert result.fixtures[0].match_key
        assert "_" in result.fixtures[0].match_key

    def test_result_is_parse_fixture_lines_result(self):
        """Returnerar ParseFixtureLinesResult."""
        result = parse_fixture_lines("Arsenal - Liverpool")
        assert isinstance(result, ParseFixtureLinesResult)

    def test_only_separator_line_invalid(self):
        """Rad med bara separator ar ogiltig."""
        result = parse_fixture_lines(" - ")
        assert result.valid_count == 0
        assert result.invalid_count == 1


# ---------------------------------------------------------------------------
# fetch_odds_for_fixtures
# ---------------------------------------------------------------------------

class TestFetchOddsForFixtures:
    def test_no_data_dir_returns_all_unmatched(self, tmp_path):
        """Utan data-mapp returneras alla som omatchade."""
        fixtures = [MatchdayFixture("Arsenal", "Liverpool", "Arsenal_Liverpool")]
        odds, matched, unmatched, labels = fetch_odds_for_fixtures(
            fixtures, data_dir=tmp_path / "nonexistent",
        )
        assert matched == 0
        assert unmatched == 1
        assert len(labels) == 1

    def test_empty_data_dir_returns_all_unmatched(self, tmp_path):
        """Tom data-mapp returnerar alla som omatchade."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        fixtures = [MatchdayFixture("Arsenal", "Liverpool", "Arsenal_Liverpool")]
        odds, matched, unmatched, labels = fetch_odds_for_fixtures(
            fixtures, data_dir=data_dir,
        )
        assert matched == 0
        assert unmatched == 1

    def test_matches_from_csv(self, tmp_path):
        """Odds matchas korrekt fran CSV-fil."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        csv_path = data_dir / "E0_2526.csv"
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal", "Man City"],
            "AwayTeam": ["Liverpool", "Chelsea"],
            "B365H": [2.10, 1.50],
            "B365D": [3.40, 4.50],
            "B365A": [3.60, 6.50],
        })
        df.to_csv(csv_path, index=False)

        fixtures = [
            MatchdayFixture("Arsenal", "Liverpool", "Arsenal_Liverpool"),
            MatchdayFixture("Man City", "Chelsea", "Man City_Chelsea"),
        ]
        odds, matched, unmatched, labels = fetch_odds_for_fixtures(
            fixtures, data_dir=data_dir,
        )
        assert matched == 2
        assert unmatched == 0
        assert len(labels) == 0

    def test_partial_match(self, tmp_path):
        """Bara en del av fixtures hittas i CSV."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        csv_path = data_dir / "E0_2526.csv"
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Liverpool"],
            "B365H": [2.10],
            "B365D": [3.40],
            "B365A": [3.60],
        })
        df.to_csv(csv_path, index=False)

        fixtures = [
            MatchdayFixture("Arsenal", "Liverpool", "Arsenal_Liverpool"),
            MatchdayFixture("Brighton", "Newcastle", "Brighton_Newcastle"),
        ]
        odds, matched, unmatched, labels = fetch_odds_for_fixtures(
            fixtures, data_dir=data_dir,
        )
        assert matched == 1
        assert unmatched == 1
        assert "Brighton vs Newcastle" in labels[0]

    def test_missing_odds_handled_defensively(self, tmp_path):
        """CSV utan oddskolumner kraschar inte."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        csv_path = data_dir / "E0_2526.csv"
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Liverpool"],
        })
        df.to_csv(csv_path, index=False)

        fixtures = [MatchdayFixture("Arsenal", "Liverpool", "Arsenal_Liverpool")]
        odds, matched, unmatched, labels = fetch_odds_for_fixtures(
            fixtures, data_dir=data_dir,
        )
        assert matched == 0
        assert unmatched == 1

    def test_status_object_built_correctly(self, tmp_path):
        """Status fran parse + fetch kan anvandas med match_matchday_data."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        csv_path = data_dir / "E0_2526.csv"
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Liverpool"],
            "B365H": [2.10],
            "B365D": [3.40],
            "B365A": [3.60],
        })
        df.to_csv(csv_path, index=False)

        # Parse text -> fetch odds -> match
        parse_result = parse_fixture_lines("Arsenal - Liverpool\nBrighton - Newcastle")
        assert parse_result.valid_count == 2

        odds_by_key, matched, unmatched, labels = fetch_odds_for_fixtures(
            parse_result.fixtures, data_dir=data_dir,
        )
        assert matched == 1
        assert unmatched == 1

        # Feed into match_matchday_data
        matches, status = match_matchday_data(parse_result.fixtures, odds_by_key, {})
        assert status.fixtures_count == 2
        assert status.odds_matched == 1
        assert len(status.fixtures_without_odds) == 1
