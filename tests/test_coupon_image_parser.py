"""
Tester for coupon_image_parser.py -- kupongbildstolkning och pipeline.

Testar:
- Bildformatvalidering
- Extraktionsresultat -> DataFrame-konvertering
- Defensiv hantering av osakra/ofullstandiga rader
- Kontrolltabell -> matchday-datastrukturer
- Bekraftat dataflode bygger fixtures/streck/odds korrekt
- Fallback-odds anvands endast nar bildodds saknas
- Fallback-vagar inte bryts
"""

import pytest
import pandas as pd
import json
from unittest.mock import patch, MagicMock

from coupon_image_parser import (
    CouponRow,
    CouponExtractionResult,
    is_supported_image,
    coupon_rows_to_dataframe,
    dataframe_to_coupon_rows,
    confirmed_rows_to_matchday_data,
    parse_coupon_image,
    _parse_extraction_response,
    _safe_float,
    _safe_odds,
    _revalidate_confidence,
    _build_result,
)
from odds_tool import OddsEntry
from matchday_import import (
    match_matchday_data,
    MatchdayFixture,
)


# ---------------------------------------------------------------------------
# Bildformatvalidering
# ---------------------------------------------------------------------------

class TestImageFormatValidation:
    """test att bildinputvagen accepterar giltiga bildformat"""

    def test_png_accepted(self):
        assert is_supported_image("coupon.png") is True

    def test_jpg_accepted(self):
        assert is_supported_image("coupon.jpg") is True

    def test_jpeg_accepted(self):
        assert is_supported_image("coupon.jpeg") is True

    def test_webp_accepted(self):
        assert is_supported_image("coupon.webp") is True

    def test_uppercase_accepted(self):
        assert is_supported_image("COUPON.PNG") is True
        assert is_supported_image("photo.JPG") is True

    def test_gif_rejected(self):
        assert is_supported_image("coupon.gif") is False

    def test_pdf_rejected(self):
        assert is_supported_image("coupon.pdf") is False

    def test_bmp_rejected(self):
        assert is_supported_image("coupon.bmp") is False

    def test_no_extension_rejected(self):
        assert is_supported_image("coupon") is False

    def test_unsupported_format_returns_error(self):
        """parse_coupon_image returnerar fel for ej stodda format."""
        result = parse_coupon_image(b"fake", "test.gif", api_key="fake-key")
        assert result.error is not None
        assert "stods inte" in result.error

    def test_no_api_key_returns_error(self):
        """parse_coupon_image returnerar fel om API-nyckel saknas."""
        with patch.dict("os.environ", {}, clear=True):
            result = parse_coupon_image(b"fake", "test.png")
            assert result.error is not None
            assert "API-nyckel" in result.error


# ---------------------------------------------------------------------------
# Extraktionsresultat -> DataFrame-konvertering
# ---------------------------------------------------------------------------

class TestCouponRowsToDataframe:
    """test att extraktionsresultat kan omvandlas till korrekt tabellstruktur"""

    def test_basic_conversion(self):
        rows = [
            CouponRow("Arsenal", "Liverpool", 45, 28, 27, 2.10, 3.40, 3.60),
            CouponRow("Man City", "Chelsea", 55, 25, 20, 1.50, 4.50, 6.50),
        ]
        df = coupon_rows_to_dataframe(rows)
        assert len(df) == 2
        assert list(df.columns) == [
            "HomeTeam", "AwayTeam", "Streck1", "StreckX", "Streck2",
            "Odds1", "OddsX", "Odds2", "Status", "Notes",
        ]
        assert df.iloc[0]["HomeTeam"] == "Arsenal"
        assert df.iloc[0]["AwayTeam"] == "Liverpool"
        assert df.iloc[0]["Streck1"] == 45
        assert df.iloc[0]["Odds1"] == 2.10

    def test_none_values_preserved(self):
        rows = [
            CouponRow("Arsenal", "Liverpool", None, None, None, None, None, None,
                       confidence="incomplete", notes="Kunde inte tolka"),
        ]
        df = coupon_rows_to_dataframe(rows)
        assert pd.isna(df.iloc[0]["Streck1"])
        assert pd.isna(df.iloc[0]["Odds1"])
        assert df.iloc[0]["Status"] == "incomplete"
        assert df.iloc[0]["Notes"] == "Kunde inte tolka"

    def test_empty_rows_returns_empty_df(self):
        df = coupon_rows_to_dataframe([])
        assert len(df) == 0

    def test_status_and_notes_included(self):
        rows = [
            CouponRow("A", "B", confidence="uncertain", notes="Svarlast"),
        ]
        df = coupon_rows_to_dataframe(rows)
        assert df.iloc[0]["Status"] == "uncertain"
        assert df.iloc[0]["Notes"] == "Svarlast"


class TestDataframeToCouponRows:
    """test att DataFrame kan konverteras tillbaka till CouponRows."""

    def test_roundtrip(self):
        original = [
            CouponRow("Arsenal", "Liverpool", 45, 28, 27, 2.10, 3.40, 3.60, "ok", ""),
        ]
        df = coupon_rows_to_dataframe(original)
        result = dataframe_to_coupon_rows(df)
        assert len(result) == 1
        assert result[0].home_team == "Arsenal"
        assert result[0].away_team == "Liverpool"
        assert result[0].streck_1 == 45
        assert result[0].odds_1 == 2.10

    def test_empty_rows_filtered(self):
        """Rader med tomma lagnamn filtreras bort."""
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal", "", ""],
            "AwayTeam": ["Liverpool", "", ""],
            "Streck1": [45, None, None],
            "StreckX": [28, None, None],
            "Streck2": [27, None, None],
            "Odds1": [2.10, None, None],
            "OddsX": [3.40, None, None],
            "Odds2": [3.60, None, None],
            "Status": ["ok", "incomplete", "incomplete"],
            "Notes": ["", "", ""],
        })
        result = dataframe_to_coupon_rows(df)
        assert len(result) == 1
        assert result[0].home_team == "Arsenal"


# ---------------------------------------------------------------------------
# Defensiv hantering av osakra/ofullstandiga rader
# ---------------------------------------------------------------------------

class TestDefensiveHandling:
    """test att osakra/ofullstandiga rader hanteras defensivt"""

    def test_complete_row_stays_ok(self):
        row = CouponRow("A", "B", 45, 28, 27, 2.10, 3.40, 3.60, "ok", "")
        result = _revalidate_confidence(row)
        assert result.confidence == "ok"

    def test_missing_teams_marked_incomplete(self):
        row = CouponRow("", "", 45, 28, 27, 2.10, 3.40, 3.60, "ok", "")
        result = _revalidate_confidence(row)
        assert result.confidence == "incomplete"
        assert "Lagnamn" in result.notes

    def test_no_data_marked_incomplete(self):
        row = CouponRow("A", "B", None, None, None, None, None, None, "ok", "")
        result = _revalidate_confidence(row)
        assert result.confidence == "incomplete"

    def test_partial_streck_marked_uncertain(self):
        row = CouponRow("A", "B", 45, None, 27, 2.10, 3.40, 3.60, "ok", "")
        result = _revalidate_confidence(row)
        assert result.confidence == "uncertain"
        assert "streck" in result.notes

    def test_partial_odds_marked_uncertain(self):
        row = CouponRow("A", "B", 45, 28, 27, 2.10, None, 3.60, "ok", "")
        result = _revalidate_confidence(row)
        assert result.confidence == "uncertain"
        assert "odds" in result.notes

    def test_already_uncertain_stays_uncertain(self):
        row = CouponRow("A", "B", 45, 28, 27, 2.10, None, 3.60, "uncertain", "Svarlast")
        result = _revalidate_confidence(row)
        assert result.confidence == "uncertain"
        # Original notes preserved when already set
        assert result.notes == "Svarlast"

    def test_safe_float_handles_none(self):
        assert _safe_float(None) is None

    def test_safe_float_handles_invalid(self):
        assert _safe_float("abc") is None

    def test_safe_float_handles_negative(self):
        assert _safe_float(-5) is None

    def test_safe_float_handles_valid(self):
        assert _safe_float(45.5) == 45.5

    def test_safe_odds_rejects_low_value(self):
        assert _safe_odds(0.5) is None
        assert _safe_odds(1.0) is None

    def test_safe_odds_accepts_valid(self):
        assert _safe_odds(2.10) == 2.10


# ---------------------------------------------------------------------------
# JSON-parsning (mocka API-svar)
# ---------------------------------------------------------------------------

class TestParseExtractionResponse:
    """test att API-svar parsas korrekt."""

    def test_valid_json_array(self):
        raw = json.dumps([
            {
                "HomeTeam": "Arsenal",
                "AwayTeam": "Liverpool",
                "Streck1": 45,
                "StreckX": 28,
                "Streck2": 27,
                "Odds1": 2.10,
                "OddsX": 3.40,
                "Odds2": 3.60,
                "confidence": "ok",
                "notes": "",
            }
        ])
        result = _parse_extraction_response(raw)
        assert result.error is None
        assert result.total_rows == 1
        assert result.rows[0].home_team == "Arsenal"
        assert result.rows[0].odds_1 == 2.10

    def test_json_in_markdown_code_block(self):
        raw = '```json\n[{"HomeTeam":"A","AwayTeam":"B","Streck1":45,"StreckX":28,"Streck2":27,"Odds1":2.10,"OddsX":3.40,"Odds2":3.60,"confidence":"ok","notes":""}]\n```'
        result = _parse_extraction_response(raw)
        assert result.error is None
        assert result.total_rows == 1

    def test_invalid_json_returns_error(self):
        result = _parse_extraction_response("this is not json")
        assert result.error is not None
        assert "JSON" in result.error

    def test_non_array_json_returns_error(self):
        result = _parse_extraction_response('{"key": "value"}')
        assert result.error is not None
        assert "lista" in result.error

    def test_null_values_handled(self):
        raw = json.dumps([
            {
                "HomeTeam": "Arsenal",
                "AwayTeam": "Liverpool",
                "Streck1": None,
                "StreckX": None,
                "Streck2": None,
                "Odds1": None,
                "OddsX": None,
                "Odds2": None,
                "confidence": "incomplete",
                "notes": "Kunde inte lasa",
            }
        ])
        result = _parse_extraction_response(raw)
        assert result.total_rows == 1
        assert result.rows[0].streck_1 is None
        assert result.rows[0].odds_1 is None
        assert result.incomplete_rows == 1

    def test_multiple_rows_counted_correctly(self):
        raw = json.dumps([
            {"HomeTeam": "A", "AwayTeam": "B", "Streck1": 45, "StreckX": 28, "Streck2": 27,
             "Odds1": 2.10, "OddsX": 3.40, "Odds2": 3.60, "confidence": "ok", "notes": ""},
            {"HomeTeam": "C", "AwayTeam": "D", "Streck1": 55, "StreckX": 25, "Streck2": 20,
             "Odds1": None, "OddsX": None, "Odds2": None, "confidence": "uncertain", "notes": "Odds saknas"},
            {"HomeTeam": "", "AwayTeam": "", "Streck1": None, "StreckX": None, "Streck2": None,
             "Odds1": None, "OddsX": None, "Odds2": None, "confidence": "incomplete", "notes": ""},
        ])
        result = _parse_extraction_response(raw)
        # Empty teams row filtered out
        assert result.total_rows == 2
        assert result.complete_rows == 1
        # Second row has teams + streck but no odds -> uncertain
        assert result.uncertain_rows >= 1


# ---------------------------------------------------------------------------
# Kontrolltabell -> matchday-data
# ---------------------------------------------------------------------------

class TestConfirmedRowsToMatchdayData:
    """test att bekraftat dataflode bygger fixtures/streck/odds korrekt"""

    def test_full_rows_build_correct_structures(self):
        rows = [
            CouponRow("Arsenal", "Liverpool", 45, 28, 27, 2.10, 3.40, 3.60, "ok", ""),
            CouponRow("Man City", "Chelsea", 55, 25, 20, 1.50, 4.50, 6.50, "ok", ""),
        ]
        fixtures, odds_by_key, streck_by_key, with_odds, missing_odds = (
            confirmed_rows_to_matchday_data(rows)
        )

        assert len(fixtures) == 2
        assert len(odds_by_key) == 2
        assert len(streck_by_key) == 2
        assert len(with_odds) == 2
        assert len(missing_odds) == 0

        # Verify fixture structure
        assert fixtures[0].home_team == "Arsenal"
        assert fixtures[0].away_team == "Liverpool"
        assert "_" in fixtures[0].match_key

        # Verify odds structure
        key = fixtures[0].match_key
        assert len(odds_by_key[key]) == 1
        assert odds_by_key[key][0].bookmaker == "Kupongbild"
        assert odds_by_key[key][0].home == 2.10
        assert odds_by_key[key][0].draw == 3.40
        assert odds_by_key[key][0].away == 3.60

        # Verify streck structure
        assert streck_by_key[key]["1"] == 45
        assert streck_by_key[key]["X"] == 28
        assert streck_by_key[key]["2"] == 27

    def test_missing_odds_tracked(self):
        rows = [
            CouponRow("Arsenal", "Liverpool", 45, 28, 27, 2.10, 3.40, 3.60, "ok", ""),
            CouponRow("Man City", "Chelsea", 55, 25, 20, None, None, None, "uncertain", "Odds saknas"),
        ]
        fixtures, odds_by_key, streck_by_key, with_odds, missing_odds = (
            confirmed_rows_to_matchday_data(rows)
        )

        assert len(with_odds) == 1
        assert len(missing_odds) == 1
        assert fixtures[1].match_key in missing_odds

    def test_partial_odds_treated_as_missing(self):
        """Om bara nagra odds finns, raknas de som saknande."""
        rows = [
            CouponRow("A", "B", 45, 28, 27, 2.10, None, 3.60, "uncertain", ""),
        ]
        fixtures, odds_by_key, streck_by_key, with_odds, missing_odds = (
            confirmed_rows_to_matchday_data(rows)
        )
        assert len(missing_odds) == 1
        assert len(with_odds) == 0

    def test_empty_team_names_skipped(self):
        rows = [
            CouponRow("", "", 45, 28, 27, 2.10, 3.40, 3.60, "incomplete", ""),
            CouponRow("Arsenal", "Liverpool", 45, 28, 27, 2.10, 3.40, 3.60, "ok", ""),
        ]
        fixtures, odds_by_key, streck_by_key, with_odds, missing_odds = (
            confirmed_rows_to_matchday_data(rows)
        )
        assert len(fixtures) == 1

    def test_streck_with_zero_sum_excluded(self):
        rows = [
            CouponRow("A", "B", 0, 0, 0, 2.10, 3.40, 3.60, "ok", ""),
        ]
        fixtures, odds_by_key, streck_by_key, with_odds, missing_odds = (
            confirmed_rows_to_matchday_data(rows)
        )
        assert len(streck_by_key) == 0

    def test_partial_streck_excluded(self):
        rows = [
            CouponRow("A", "B", 45, None, 27, 2.10, 3.40, 3.60, "uncertain", ""),
        ]
        fixtures, odds_by_key, streck_by_key, with_odds, missing_odds = (
            confirmed_rows_to_matchday_data(rows)
        )
        assert len(streck_by_key) == 0


# ---------------------------------------------------------------------------
# Integration med match_matchday_data
# ---------------------------------------------------------------------------

class TestIntegrationWithMatchdayData:
    """test att kontrolltabellen kan anvandas som kalla till vidare analys"""

    def test_coupon_data_feeds_analysis(self):
        """Bekraftad kupongdata ger korrekt matchday-analys."""
        rows = [
            CouponRow("Arsenal", "Liverpool", 45, 28, 27, 2.10, 3.40, 3.60, "ok", ""),
        ]
        fixtures, odds_by_key, streck_by_key, _, _ = (
            confirmed_rows_to_matchday_data(rows)
        )

        matches, status = match_matchday_data(fixtures, odds_by_key, streck_by_key)

        assert status.fixtures_count == 1
        assert status.odds_matched == 1
        assert status.streck_matched == 1
        assert status.fully_matched == 1
        assert len(matches) == 1
        assert matches[0].has_odds is True
        assert matches[0].has_streck is True
        assert matches[0].odds_report is not None

    def test_coupon_data_without_odds_partial_analysis(self):
        """Kupongdata utan odds ger partiell analys."""
        rows = [
            CouponRow("Arsenal", "Liverpool", 45, 28, 27, None, None, None, "uncertain", ""),
        ]
        fixtures, odds_by_key, streck_by_key, _, _ = (
            confirmed_rows_to_matchday_data(rows)
        )

        matches, status = match_matchday_data(fixtures, odds_by_key, streck_by_key)

        assert status.fixtures_count == 1
        assert status.odds_matched == 0
        assert status.streck_matched == 1
        assert matches[0].has_odds is False
        assert matches[0].has_streck is True


# ---------------------------------------------------------------------------
# Fallback-odds logik
# ---------------------------------------------------------------------------

class TestFallbackOddsLogic:
    """test att fallback-odds anvands endast nar bildodds saknas"""

    def test_image_odds_used_when_present(self):
        rows = [
            CouponRow("Arsenal", "Liverpool", 45, 28, 27, 2.10, 3.40, 3.60, "ok", ""),
        ]
        fixtures, odds_by_key, streck_by_key, with_odds, missing_odds = (
            confirmed_rows_to_matchday_data(rows)
        )

        # Simulera fallback-odds som INTE ska anvandas
        key = fixtures[0].match_key
        assert key in odds_by_key
        assert key in with_odds
        assert key not in missing_odds

        # Bildodds ska vara "Kupongbild"
        assert odds_by_key[key][0].bookmaker == "Kupongbild"

    def test_fallback_only_for_missing(self):
        rows = [
            CouponRow("Arsenal", "Liverpool", 45, 28, 27, 2.10, 3.40, 3.60, "ok", ""),
            CouponRow("Man City", "Chelsea", 55, 25, 20, None, None, None, "uncertain", ""),
        ]
        fixtures, odds_by_key, streck_by_key, with_odds, missing_odds = (
            confirmed_rows_to_matchday_data(rows)
        )

        # Arsenal har bildodds, Man City saknar
        assert len(with_odds) == 1
        assert len(missing_odds) == 1
        assert fixtures[0].match_key in with_odds
        assert fixtures[1].match_key in missing_odds

        # Simulera fallback: lagg till odds for den saknade matchen
        fb_key = fixtures[1].match_key
        odds_by_key[fb_key] = [OddsEntry("Fallback", 1.50, 4.50, 6.50)]

        # Nu ska bada ha odds
        matches, status = match_matchday_data(fixtures, odds_by_key, streck_by_key)
        assert status.odds_matched == 2


# ---------------------------------------------------------------------------
# Fallback-vagar inte bryts
# ---------------------------------------------------------------------------

class TestFallbackPathsPreserved:
    """test att fallback-vagar inte bryts"""

    def test_existing_matchday_import_works(self):
        """Befintlig matchday_import-pipeline fungerar som vanligt."""
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
        assert status.fixtures_count == 1
        assert status.fully_matched == 1

    def test_existing_odds_tool_works(self):
        """Befintlig odds_tool fungerar som vanligt."""
        from odds_tool import build_match_report
        entries = [
            OddsEntry("Bet365", 2.10, 3.40, 3.60),
            OddsEntry("Pinnacle", 2.15, 3.35, 3.65),
        ]
        report = build_match_report("Arsenal", "Liverpool", entries)
        assert report.home_team == "Arsenal"
        assert report.overround > 0

    def test_existing_value_analysis_works(self):
        """Befintlig value_analysis fungerar som vanligt."""
        from odds_tool import build_match_report
        from value_analysis import build_value_report
        entries = [
            OddsEntry("Bet365", 2.10, 3.40, 3.60),
            OddsEntry("Pinnacle", 2.15, 3.35, 3.65),
        ]
        report = build_match_report("Arsenal", "Liverpool", entries)
        vr = build_value_report(report)
        assert vr is not None

    def test_existing_streck_analysis_works(self):
        """Befintlig streck_analysis fungerar som vanligt."""
        from odds_tool import build_match_report
        from streck_analysis import build_streck_report_from_odds_report
        entries = [OddsEntry("Bet365", 2.10, 3.40, 3.60)]
        report = build_match_report("Arsenal", "Liverpool", entries)
        streck = {"1": 45.0, "X": 28.0, "2": 27.0}
        sr = build_streck_report_from_odds_report(report, streck)
        assert sr is not None


# ---------------------------------------------------------------------------
# Build result
# ---------------------------------------------------------------------------

class TestBuildResult:
    """test att _build_result raeknar korrekt."""

    def test_counts_correct(self):
        rows = [
            CouponRow("A", "B", confidence="ok"),
            CouponRow("C", "D", confidence="uncertain"),
            CouponRow("E", "F", confidence="incomplete"),
            CouponRow("G", "H", confidence="ok"),
        ]
        result = _build_result(rows, "raw")
        assert result.total_rows == 4
        assert result.complete_rows == 2
        assert result.uncertain_rows == 1
        assert result.incomplete_rows == 1
        assert result.error is None

    def test_empty_rows(self):
        result = _build_result([], "raw")
        assert result.total_rows == 0
        assert result.complete_rows == 0
