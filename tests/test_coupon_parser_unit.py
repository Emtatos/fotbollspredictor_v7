"""
Enhetstester for coupon_image_parser.

Testar parsning av JSON-svar, validering av streck-summor,
confidence-omvalidering och felhantering utan att anropa OpenAI.
"""
import pytest
from coupon_image_parser import (
    _parse_extraction_response,
    _json_to_coupon_rows,
    _revalidate_confidence,
    _safe_float,
    _safe_odds,
    _build_result,
    is_supported_image,
    _detect_mime_type,
    _encode_image_to_base64,
    CouponRow,
    CouponExtractionResult,
)


# ---------------------------------------------------------------------------
# _parse_extraction_response
# ---------------------------------------------------------------------------

class TestParseExtractionResponse:
    """Tester for _parse_extraction_response."""

    def test_parse_valid_json_response(self):
        """Testar att valformaterat JSON-svar parseas korrekt."""
        raw = '''[
            {
                "HomeTeam": "Arsenal",
                "AwayTeam": "Chelsea",
                "Streck1": 55.0,
                "StreckX": 25.0,
                "Streck2": 20.0,
                "Odds1": 1.85,
                "OddsX": 3.50,
                "Odds2": 4.20,
                "confidence": "ok",
                "notes": ""
            }
        ]'''
        result = _parse_extraction_response(raw)
        assert len(result.rows) == 1
        assert result.rows[0].home_team == "Arsenal"
        assert result.rows[0].away_team == "Chelsea"
        assert result.rows[0].odds_1 == 1.85
        assert result.rows[0].odds_x == 3.50
        assert result.rows[0].odds_2 == 4.20
        assert result.rows[0].streck_1 == 55.0
        assert result.rows[0].streck_x == 25.0
        assert result.rows[0].streck_2 == 20.0
        assert result.complete_rows == 1
        assert result.error is None

    def test_parse_multiple_rows(self):
        """Testar parsning av flera matcher."""
        raw = '''[
            {"HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
             "Streck1": 50, "StreckX": 25, "Streck2": 25,
             "Odds1": 2.0, "OddsX": 3.5, "Odds2": 3.5},
            {"HomeTeam": "Liverpool", "AwayTeam": "Everton",
             "Streck1": 60, "StreckX": 22, "Streck2": 18,
             "Odds1": 1.60, "OddsX": 4.0, "Odds2": 5.5}
        ]'''
        result = _parse_extraction_response(raw)
        assert len(result.rows) == 2
        assert result.rows[0].home_team == "Arsenal"
        assert result.rows[1].home_team == "Liverpool"
        assert result.total_rows == 2

    def test_parse_incomplete_row(self):
        """Rad utan odds ska flaggas som uncertain (har streck men inte odds)."""
        raw = '''[
            {
                "HomeTeam": "Liverpool",
                "AwayTeam": "Everton",
                "Streck1": 60.0,
                "StreckX": 22.0,
                "Streck2": 18.0
            }
        ]'''
        result = _parse_extraction_response(raw)
        assert len(result.rows) == 1
        assert result.rows[0].odds_1 is None
        assert result.rows[0].odds_x is None
        assert result.rows[0].odds_2 is None
        assert result.rows[0].confidence == "uncertain"

    def test_parse_malformed_json(self):
        """Ogiltig JSON ska ge tom result med error."""
        result = _parse_extraction_response("detta ar inte json alls")
        assert len(result.rows) == 0
        assert result.error is not None

    def test_parse_json_in_markdown_code_block(self):
        """JSON inbaddat i markdown code fences ska parseas korrekt."""
        raw = '''```json
[
    {"HomeTeam": "Malmo FF", "AwayTeam": "AIK",
     "Streck1": 45, "StreckX": 28, "Streck2": 27,
     "Odds1": 2.10, "OddsX": 3.40, "Odds2": 3.60}
]
```'''
        result = _parse_extraction_response(raw)
        assert len(result.rows) == 1
        assert result.rows[0].home_team == "Malmo FF"

    def test_parse_json_with_leading_text(self):
        """JSON med text fore arraystart ska parseas."""
        raw = '''Here are the results:
[
    {"HomeTeam": "IFK Goteborg", "AwayTeam": "Djurgarden",
     "Streck1": 35, "StreckX": 30, "Streck2": 35,
     "Odds1": 2.80, "OddsX": 3.20, "Odds2": 2.60}
]'''
        result = _parse_extraction_response(raw)
        assert len(result.rows) == 1
        assert result.rows[0].home_team == "IFK Goteborg"

    def test_parse_not_a_list(self):
        """JSON som inte ar en lista ska ge error."""
        raw = '{"HomeTeam": "Arsenal", "AwayTeam": "Chelsea"}'
        result = _parse_extraction_response(raw)
        assert len(result.rows) == 0
        assert result.error is not None

    def test_parse_empty_list(self):
        """Tom JSON-lista ska ge tomt resultat utan error."""
        result = _parse_extraction_response("[]")
        assert len(result.rows) == 0
        assert result.total_rows == 0
        assert result.error is None


# ---------------------------------------------------------------------------
# _revalidate_confidence
# ---------------------------------------------------------------------------

class TestRevalidateConfidence:
    """Tester for _revalidate_confidence."""

    def test_revalidate_streck_sum_too_low(self):
        """Streck som inte summerar till ~100% ska ge uncertain."""
        row = CouponRow(
            home_team="A", away_team="B",
            streck_1=30.0, streck_x=30.0, streck_2=30.0,  # = 90%
            odds_1=2.0, odds_x=3.0, odds_2=4.0,
        )
        validated = _revalidate_confidence(row)
        assert validated.confidence == "uncertain"
        assert "90.0%" in validated.notes

    def test_revalidate_streck_sum_too_high(self):
        """Streck > 105% ska ge uncertain."""
        row = CouponRow(
            home_team="A", away_team="B",
            streck_1=50.0, streck_x=40.0, streck_2=20.0,  # = 110%
            odds_1=2.0, odds_x=3.0, odds_2=4.0,
        )
        validated = _revalidate_confidence(row)
        assert validated.confidence == "uncertain"
        assert "110.0%" in validated.notes

    def test_revalidate_streck_sum_within_tolerance(self):
        """Streck som summerar till 98% (inom +-5%) ska vara ok."""
        row = CouponRow(
            home_team="A", away_team="B",
            streck_1=45.0, streck_x=28.0, streck_2=25.0,  # = 98%
            odds_1=2.0, odds_x=3.0, odds_2=4.0,
        )
        validated = _revalidate_confidence(row)
        assert validated.confidence == "ok"

    def test_revalidate_streck_sum_exactly_100(self):
        """Streck som summerar exakt till 100% ska vara ok."""
        row = CouponRow(
            home_team="A", away_team="B",
            streck_1=45.0, streck_x=28.0, streck_2=27.0,
            odds_1=2.0, odds_x=3.0, odds_2=4.0,
        )
        validated = _revalidate_confidence(row)
        assert validated.confidence == "ok"

    def test_missing_teams(self):
        """Rad utan lagnamn ska vara incomplete."""
        row = CouponRow(
            home_team="", away_team="",
            streck_1=45.0, streck_x=28.0, streck_2=27.0,
            odds_1=2.0, odds_x=3.0, odds_2=4.0,
        )
        validated = _revalidate_confidence(row)
        assert validated.confidence == "incomplete"
        assert "Lagnamn" in validated.notes

    def test_no_streck_no_odds(self):
        """Rad utan streck eller odds ska vara incomplete."""
        row = CouponRow(
            home_team="A", away_team="B",
        )
        validated = _revalidate_confidence(row)
        assert validated.confidence == "incomplete"

    def test_partial_streck_missing_odds(self):
        """Rad med delvis streck men inga odds ska vara uncertain."""
        row = CouponRow(
            home_team="A", away_team="B",
            streck_1=45.0, streck_x=None, streck_2=27.0,
        )
        validated = _revalidate_confidence(row)
        assert validated.confidence == "uncertain"

    def test_streck_only_no_odds_valid_sum(self):
        """Streck utan odds, giltig summa, ska vara uncertain (saknar odds)."""
        row = CouponRow(
            home_team="A", away_team="B",
            streck_1=45.0, streck_x=28.0, streck_2=27.0,
        )
        validated = _revalidate_confidence(row)
        assert validated.confidence == "uncertain"

    def test_streck_only_no_odds_bad_sum(self):
        """Streck utan odds, dalig summa, ska vara uncertain med streck-note."""
        row = CouponRow(
            home_team="A", away_team="B",
            streck_1=30.0, streck_x=20.0, streck_2=20.0,  # = 70%
        )
        validated = _revalidate_confidence(row)
        assert validated.confidence == "uncertain"
        assert "70.0%" in validated.notes


# ---------------------------------------------------------------------------
# _safe_float / _safe_odds
# ---------------------------------------------------------------------------

class TestSafeConversions:
    """Tester for _safe_float och _safe_odds."""

    def test_safe_float_valid(self):
        assert _safe_float(3.14) == 3.14
        assert _safe_float("2.5") == 2.5
        assert _safe_float(0) == 0.0
        assert _safe_float(100) == 100.0

    def test_safe_float_none(self):
        assert _safe_float(None) is None

    def test_safe_float_invalid(self):
        assert _safe_float("abc") is None
        assert _safe_float("") is None

    def test_safe_float_negative(self):
        assert _safe_float(-1.0) is None

    def test_safe_odds_valid(self):
        assert _safe_odds(2.10) == 2.10
        assert _safe_odds("3.50") == 3.50

    def test_safe_odds_too_low(self):
        """Odds <= 1.0 ska vara None."""
        assert _safe_odds(1.0) is None
        assert _safe_odds(0.5) is None

    def test_safe_odds_none(self):
        assert _safe_odds(None) is None


# ---------------------------------------------------------------------------
# _json_to_coupon_rows
# ---------------------------------------------------------------------------

class TestJsonToCouponRows:
    """Tester for _json_to_coupon_rows."""

    def test_skips_non_dict_items(self):
        data = [42, "string", None, {"HomeTeam": "A", "AwayTeam": "B"}]
        rows = _json_to_coupon_rows(data)
        assert len(rows) == 1
        assert rows[0].home_team == "A"

    def test_skips_empty_teams(self):
        data = [{"HomeTeam": "", "AwayTeam": ""}]
        rows = _json_to_coupon_rows(data)
        assert len(rows) == 0

    def test_extracts_all_fields(self):
        data = [{
            "HomeTeam": "Team A",
            "AwayTeam": "Team B",
            "Streck1": 45,
            "StreckX": 28,
            "Streck2": 27,
            "Odds1": 2.10,
            "OddsX": 3.40,
            "Odds2": 3.60,
            "confidence": "ok",
            "notes": "bra",
        }]
        rows = _json_to_coupon_rows(data)
        assert len(rows) == 1
        r = rows[0]
        assert r.home_team == "Team A"
        assert r.away_team == "Team B"
        assert r.streck_1 == 45.0
        assert r.streck_x == 28.0
        assert r.streck_2 == 27.0
        assert r.odds_1 == 2.10
        assert r.odds_x == 3.40
        assert r.odds_2 == 3.60


# ---------------------------------------------------------------------------
# Hjalp-funktioner
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    """Tester for hjalp-funktioner."""

    def test_is_supported_image(self):
        assert is_supported_image("test.png") is True
        assert is_supported_image("test.jpg") is True
        assert is_supported_image("test.jpeg") is True
        assert is_supported_image("test.webp") is True
        assert is_supported_image("test.gif") is False
        assert is_supported_image("test.bmp") is False
        assert is_supported_image("test.PDF") is False

    def test_detect_mime_type(self):
        assert _detect_mime_type("test.png") == "image/png"
        assert _detect_mime_type("test.jpg") == "image/jpeg"
        assert _detect_mime_type("test.jpeg") == "image/jpeg"
        assert _detect_mime_type("test.webp") == "image/webp"
        assert _detect_mime_type("test.unknown") == "image/png"

    def test_encode_image_to_base64(self):
        data = b"hello"
        encoded = _encode_image_to_base64(data)
        assert isinstance(encoded, str)
        import base64
        assert base64.b64decode(encoded) == data

    def test_build_result_counts(self):
        rows = [
            CouponRow(home_team="A", away_team="B", confidence="ok"),
            CouponRow(home_team="C", away_team="D", confidence="uncertain"),
            CouponRow(home_team="E", away_team="F", confidence="incomplete"),
            CouponRow(home_team="G", away_team="H", confidence="ok"),
        ]
        result = _build_result(rows, "raw")
        assert result.total_rows == 4
        assert result.complete_rows == 2
        assert result.uncertain_rows == 1
        assert result.incomplete_rows == 1
        assert result.raw_response == "raw"
