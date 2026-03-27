"""
Golden-style tester for scanner_pipeline.py.

Dessa tester mockar AI-delen (_call_openai_vision) och validerar att
postprocessing-pipelinen (parsning, validering, team mapping, confidence)
ger forvantad output givet kanda AI-svar.
"""

import json
import os
import pytest
from unittest.mock import patch

from scanner_pipeline import (
    ScannedRow,
    ScannerResult,
    run_scanner_pipeline,
    parse_row_fields,
    apply_team_mapping,
    validate_rows,
    _parse_ai_response,
    _build_scanner_result,
    scanned_rows_to_coupon_rows,
    scanner_result_to_extraction_result,
)


# ---------------------------------------------------------------------------
# Mock AI responses for each coupon image
# ---------------------------------------------------------------------------

MOCK_AI_RESPONSE_COUPON_1 = json.dumps([
    {"HomeTeam": "Arsenal", "AwayTeam": "Liverpool", "Streck1": 42, "StreckX": 28, "Streck2": 30, "Odds1": 2.15, "OddsX": 3.40, "Odds2": 3.50, "row_confidence": "high", "field_issues": []},
    {"HomeTeam": "Man Utd", "AwayTeam": "Chelsea", "Streck1": 35, "StreckX": 30, "Streck2": 35, "Odds1": 2.80, "OddsX": 3.30, "Odds2": 2.65, "row_confidence": "high", "field_issues": []},
    {"HomeTeam": "Wolves", "AwayTeam": "Brighton", "Streck1": 38, "StreckX": 30, "Streck2": 32, "Odds1": 2.50, "OddsX": 3.20, "Odds2": 2.90, "row_confidence": "high", "field_issues": []},
    {"HomeTeam": "Newcastle", "AwayTeam": "Tottenham", "Streck1": 45, "StreckX": 27, "Streck2": 28, "Odds1": 2.10, "OddsX": 3.50, "Odds2": 3.40, "row_confidence": "high", "field_issues": []},
    {"HomeTeam": "Aston Villa", "AwayTeam": "West Ham", "Streck1": 50, "StreckX": 26, "Streck2": 24, "Odds1": 1.85, "OddsX": 3.60, "Odds2": 4.20, "row_confidence": "high", "field_issues": []},
    {"HomeTeam": "Everton", "AwayTeam": "Nott'm Forest", "Streck1": 33, "StreckX": 30, "Streck2": 37, "Odds1": 3.00, "OddsX": 3.30, "Odds2": 2.40, "row_confidence": "high", "field_issues": []},
    {"HomeTeam": "Bournemouth", "AwayTeam": "Crystal Palace", "Streck1": 40, "StreckX": 29, "Streck2": 31, "Odds1": 2.30, "OddsX": 3.35, "Odds2": 3.10, "row_confidence": "high", "field_issues": []},
    {"HomeTeam": "Fulham", "AwayTeam": "Brentford", "Streck1": 36, "StreckX": 31, "Streck2": 33, "Odds1": 2.70, "OddsX": 3.25, "Odds2": 2.70, "row_confidence": "high", "field_issues": []},
    {"HomeTeam": "Leicester", "AwayTeam": "Ipswich", "Streck1": 48, "StreckX": 27, "Streck2": 25, "Odds1": 1.95, "OddsX": 3.50, "Odds2": 3.90, "row_confidence": "high", "field_issues": []},
    {"HomeTeam": "Southampton", "AwayTeam": "Leeds", "Streck1": 34, "StreckX": 29, "Streck2": 37, "Odds1": 2.85, "OddsX": 3.35, "Odds2": 2.55, "row_confidence": "high", "field_issues": []},
    {"HomeTeam": "Sheff Utd", "AwayTeam": "QPR", "Streck1": 52, "StreckX": 26, "Streck2": 22, "Odds1": 1.75, "OddsX": 3.60, "Odds2": 4.50, "row_confidence": "high", "field_issues": []},
    {"HomeTeam": "Derby", "AwayTeam": "Preston", "Streck1": 41, "StreckX": 29, "Streck2": 30, "Odds1": 2.25, "OddsX": 3.40, "Odds2": 3.20, "row_confidence": "high", "field_issues": []},
    {"HomeTeam": "Luton", "AwayTeam": "Plymouth", "Streck1": 39, "StreckX": 30, "Streck2": 31, "Odds1": 2.40, "OddsX": 3.30, "Odds2": 3.00, "row_confidence": "high", "field_issues": []},
])

_COUPON_2_DATA = [
    {"HomeTeam": "AIK", "AwayTeam": "Djurgarden", "Streck1": 38, "StreckX": 29, "Streck2": 33, "Odds1": 2.55, "OddsX": 3.30, "Odds2": 2.80, "row_confidence": "high", "field_issues": []},
    {"HomeTeam": "Hammarby", "AwayTeam": "Malmo FF", "Streck1": 42, "StreckX": 28, "Streck2": 30, "Odds1": 2.20, "OddsX": 3.40, "Odds2": 3.30, "row_confidence": "high", "field_issues": []},
    {"HomeTeam": "IFK Goteborg", "AwayTeam": "IFK Norrkoping", "Streck1": 45, "StreckX": 27, "Streck2": 28, "Odds1": 2.10, "OddsX": 3.50, "Odds2": 3.40, "row_confidence": "high", "field_issues": []},
    {"HomeTeam": "IF Elfsborg", "AwayTeam": "Hacken", "Streck1": 40, "StreckX": 29, "Streck2": 31, "Odds1": 2.35, "OddsX": 3.30, "Odds2": 3.05, "row_confidence": "high", "field_issues": []},
    {"HomeTeam": "Kalmar FF", "AwayTeam": "Mjallby", "Streck1": 44, "StreckX": 28, "Streck2": 28, "row_confidence": "medium", "field_issues": ["Odds1", "OddsX", "Odds2"]},
    {"HomeTeam": "Sirius", "AwayTeam": "Varnamo", "Streck1": 50, "StreckX": 26, "Streck2": 24, "row_confidence": "medium", "field_issues": ["Odds1", "OddsX", "Odds2"]},
    {"HomeTeam": "Brommapojkarna", "AwayTeam": "Halmstad", "Streck1": 35, "StreckX": 30, "Streck2": 35, "Odds1": 2.75, "OddsX": 3.25, "Odds2": 2.65, "row_confidence": "high", "field_issues": []},
    {"HomeTeam": "Degerfors", "AwayTeam": "Sundsvall", "Streck1": 43, "StreckX": 28, "Streck2": 29, "Odds1": 2.20, "OddsX": 3.40, "Odds2": 3.30, "row_confidence": "high", "field_issues": []},
]
# Rows 5-6 have no odds -- set them to None explicitly
for _r in _COUPON_2_DATA:
    _r.setdefault("Odds1", None)
    _r.setdefault("OddsX", None)
    _r.setdefault("Odds2", None)
MOCK_AI_RESPONSE_COUPON_2 = json.dumps(_COUPON_2_DATA)

MOCK_AI_RESPONSE_COUPON_3 = json.dumps([
    {"HomeTeam": "Man City", "AwayTeam": "Nott'm Forest", "Streck1": 65, "StreckX": 18, "Streck2": 17, "Odds1": 1.35, "OddsX": 5.00, "Odds2": 8.50, "row_confidence": "high", "field_issues": []},
    {"HomeTeam": "Sheff Wed", "AwayTeam": "West Brom", "Streck1": 36, "StreckX": 30, "Streck2": 34, "Odds1": 2.70, "OddsX": 3.25, "Odds2": 2.70, "row_confidence": "high", "field_issues": []},
    {"HomeTeam": "Brighton", "AwayTeam": "Wolverhampton", "Streck1": 48, "StreckX": 27, "Streck2": 25, "Odds1": 1.95, "OddsX": 3.50, "Odds2": 3.90, "row_confidence": "high", "field_issues": []},
    {"HomeTeam": "Spurs", "AwayTeam": "Bournemouth", "Streck1": 52, "StreckX": 26, "Streck2": 22, "Odds1": 1.80, "OddsX": 3.55, "Odds2": 4.30, "row_confidence": "high", "field_issues": []},
    {"HomeTeam": "Blackburn", "AwayTeam": "Stockport", "Streck1": 44, "StreckX": 28, "Streck2": 28, "Odds1": 2.15, "OddsX": 3.40, "Odds2": 3.40, "row_confidence": "high", "field_issues": []},
    {"HomeTeam": "Cambridge Utd", "AwayTeam": "Charlton", "Streck1": 37, "StreckX": 30, "Streck2": 33, "Odds1": 2.60, "OddsX": 3.30, "Odds2": 2.80, "row_confidence": "high", "field_issues": []},
])

_BAD_DATA = [
    {"HomeTeam": "Arsenal", "AwayTeam": "Liverpool", "Streck1": 42, "StreckX": 28, "Streck2": 30, "Odds1": 2.15, "OddsX": 3.40, "Odds2": 3.50, "row_confidence": "high", "field_issues": []},
    {"HomeTeam": "", "AwayTeam": "", "Streck1": 35, "StreckX": 30, "Streck2": 35, "Odds1": 2.80, "OddsX": 3.30, "Odds2": 2.65, "row_confidence": "low", "field_issues": ["HomeTeam", "AwayTeam"]},
    {"HomeTeam": "Man Utd", "AwayTeam": "Chelsea", "Streck1": 20, "StreckX": 20, "Streck2": 20, "Odds1": 2.80, "OddsX": 3.30, "Odds2": 2.65, "row_confidence": "medium", "field_issues": []},
    {"HomeTeam": "Wolves", "AwayTeam": "Brighton", "Streck1": 2.50, "StreckX": 3.20, "Streck2": 2.90, "Odds1": 38, "OddsX": 30, "Odds2": 32, "row_confidence": "medium", "field_issues": []},
    {"HomeTeam": "Newcastle", "AwayTeam": "Newcastle", "Streck1": 45, "StreckX": 27, "Streck2": 28, "Odds1": 2.10, "OddsX": 3.50, "Odds2": 3.40, "row_confidence": "high", "field_issues": []},
    {"HomeTeam": "Everton", "AwayTeam": "Forest", "Streck1": 33, "StreckX": 30, "Streck2": 37, "Odds1": 200.0, "OddsX": 3.30, "Odds2": 2.40, "row_confidence": "medium", "field_issues": ["Odds1"]},
]
MOCK_AI_RESPONSE_BAD_DATA = json.dumps(_BAD_DATA)


def _fixture_path(name):
    return os.path.join(os.path.dirname(__file__), "fixtures", "coupons", name)


def _read_fixture(name):
    path = _fixture_path(name)
    if not os.path.exists(path):
        pytest.skip(f"Fixture not found: {path}")
    with open(path, "rb") as f:
        return f.read()


def _run_pipeline_with_mock(fixture_name, mock_response):
    """Run full pipeline with mocked AI vision call."""
    image_bytes = _read_fixture(fixture_name)
    with patch(
        "scanner_pipeline._call_openai_vision",
        return_value=(mock_response, None),
    ):
        result = run_scanner_pipeline(
            image_bytes, fixture_name, api_key="test-key"
        )
    return result


# ===========================================================================
# Golden Test 1: Clean 13-match stryktipset
# ===========================================================================

class TestGoldenCoupon1Clean:

    @pytest.fixture
    def pipeline_result(self):
        return _run_pipeline_with_mock(
            "coupon_clean_13matches.png", MOCK_AI_RESPONSE_COUPON_1
        )

    def test_extracts_all_13_rows(self, pipeline_result):
        assert pipeline_result.total_rows == 13

    def test_all_rows_ok(self, pipeline_result):
        assert pipeline_result.ok_rows == 13
        assert pipeline_result.uncertain_rows == 0
        assert pipeline_result.failed_rows == 0

    def test_team_names_mapped(self, pipeline_result):
        rows = pipeline_result.rows
        assert rows[1].home_team == "Manchester United"
        assert rows[1].home_team_mapped is True
        assert rows[1].home_team_raw == "Man Utd"
        assert rows[2].home_team == "Wolverhampton Wanderers"
        assert rows[2].away_team == "Brighton & Hove Albion"
        assert rows[3].home_team == "Newcastle United"
        assert rows[3].away_team == "Tottenham Hotspur"
        assert rows[5].away_team == "Nottingham Forest"
        assert rows[10].home_team == "Sheffield United"
        assert rows[10].away_team == "Queens Park Rangers"

    def test_streck_values_correct(self, pipeline_result):
        row0 = pipeline_result.rows[0]
        assert row0.streck_1 == 42.0
        assert row0.streck_x == 28.0
        assert row0.streck_2 == 30.0

    def test_odds_values_correct(self, pipeline_result):
        row0 = pipeline_result.rows[0]
        assert row0.odds_1 == 2.15
        assert row0.odds_x == 3.40
        assert row0.odds_2 == 3.50

    def test_all_streck_present(self, pipeline_result):
        for row in pipeline_result.rows:
            assert row.streck_1 is not None
            assert row.streck_x is not None
            assert row.streck_2 is not None

    def test_all_odds_present(self, pipeline_result):
        for row in pipeline_result.rows:
            assert row.odds_1 is not None
            assert row.odds_x is not None
            assert row.odds_2 is not None

    def test_preprocessing_applied(self, pipeline_result):
        assert len(pipeline_result.preprocessing_applied) > 0

    def test_backward_compat_conversion(self, pipeline_result):
        coupon_rows = scanned_rows_to_coupon_rows(pipeline_result.rows)
        assert len(coupon_rows) == 13
        for cr in coupon_rows:
            assert cr.confidence == "ok"

    def test_extraction_result_conversion(self, pipeline_result):
        extraction = scanner_result_to_extraction_result(pipeline_result)
        assert extraction.total_rows == 13
        assert extraction.complete_rows == 13
        assert extraction.error is None


# ===========================================================================
# Golden Test 2: Allsvenskan partial (missing odds)
# ===========================================================================

class TestGoldenCoupon2Partial:

    @pytest.fixture
    def pipeline_result(self):
        return _run_pipeline_with_mock(
            "coupon_allsvenskan_partial.png", MOCK_AI_RESPONSE_COUPON_2
        )

    def test_extracts_all_8_rows(self, pipeline_result):
        assert pipeline_result.total_rows == 8

    def test_rows_5_6_marked_uncertain(self, pipeline_result):
        row5 = pipeline_result.rows[4]
        row6 = pipeline_result.rows[5]
        assert row5.row_status == "uncertain"
        assert row6.row_status == "uncertain"
        assert row5.odds_1 is None
        assert row6.odds_1 is None

    def test_rows_with_odds_are_ok(self, pipeline_result):
        for i in [0, 1, 2, 3, 6, 7]:
            assert pipeline_result.rows[i].row_status == "ok", f"Row {i} should be ok"

    def test_missing_odds_flagged_in_issues(self, pipeline_result):
        row5 = pipeline_result.rows[4]
        assert any("AI-rapporterade" in issue for issue in row5.issues)

    def test_streck_present_for_all(self, pipeline_result):
        for row in pipeline_result.rows:
            assert row.streck_1 is not None

    def test_uncertain_count(self, pipeline_result):
        assert pipeline_result.uncertain_rows == 2
        assert pipeline_result.ok_rows == 6


# ===========================================================================
# Golden Test 3: Noisy/rotated with alias-heavy names
# ===========================================================================

class TestGoldenCoupon3Noisy:

    @pytest.fixture
    def pipeline_result(self):
        return _run_pipeline_with_mock(
            "coupon_noisy_rotated.png", MOCK_AI_RESPONSE_COUPON_3
        )

    def test_extracts_all_6_rows(self, pipeline_result):
        assert pipeline_result.total_rows == 6

    def test_all_rows_ok(self, pipeline_result):
        assert pipeline_result.ok_rows == 6

    def test_canonical_mapping_applied(self, pipeline_result):
        rows = pipeline_result.rows
        assert rows[0].home_team == "Manchester City"
        assert rows[0].away_team == "Nottingham Forest"
        assert rows[1].home_team == "Sheffield Wednesday"
        assert rows[1].away_team == "West Bromwich Albion"
        assert rows[2].home_team == "Brighton & Hove Albion"
        assert rows[2].away_team == "Wolverhampton Wanderers"
        assert rows[3].home_team == "Tottenham Hotspur"
        assert rows[4].home_team == "Blackburn Rovers"
        assert rows[4].away_team == "Stockport County"
        assert rows[5].home_team == "Cambridge United"
        assert rows[5].away_team == "Charlton Athletic"

    def test_mapping_count(self, pipeline_result):
        mapped = sum(1 for r in pipeline_result.rows if r.home_team_mapped)
        mapped += sum(1 for r in pipeline_result.rows if r.away_team_mapped)
        assert mapped >= 10

    def test_raw_names_preserved(self, pipeline_result):
        assert pipeline_result.rows[0].home_team_raw == "Man City"
        assert pipeline_result.rows[3].home_team_raw == "Spurs"


# ===========================================================================
# Golden Test 4: Bad/problematic data validation
# ===========================================================================

class TestGoldenBadData:

    @pytest.fixture
    def pipeline_result(self):
        return _run_pipeline_with_mock(
            "coupon_clean_13matches.png", MOCK_AI_RESPONSE_BAD_DATA
        )

    def test_extracts_rows(self, pipeline_result):
        """Rows with empty teams are filtered out by _parse_ai_response."""
        assert pipeline_result.total_rows == 5

    def test_good_row_is_ok(self, pipeline_result):
        assert pipeline_result.rows[0].home_team == "Arsenal"
        assert pipeline_result.rows[0].row_status == "ok"

    def test_bad_streck_sum_flagged(self, pipeline_result):
        row = pipeline_result.rows[1]
        assert row.home_team == "Manchester United"
        has_streck_issue = any("streck" in i.lower() or "60" in i for i in row.issues)
        assert has_streck_issue, f"Expected streck issue, got: {row.issues}"

    def test_streck_odds_confusion_detected(self, pipeline_result):
        row = pipeline_result.rows[2]
        confusion_issues = [i for i in row.issues if "forvirr" in i.lower()]
        assert len(confusion_issues) > 0, f"Expected confusion issue, got: {row.issues}"

    def test_identical_teams_flagged(self, pipeline_result):
        row = pipeline_result.rows[3]
        assert any(
            "identiska" in i.lower() for i in row.issues
        ), f"Expected identical-teams issue, got: {row.issues}"

    def test_unreasonable_odds_flagged(self, pipeline_result):
        row = pipeline_result.rows[4]
        assert any(
            "orimligt" in i.lower() or "200" in i for i in row.issues
        ), f"Expected unreasonable odds issue, got: {row.issues}"

    def test_has_failed_or_uncertain_rows(self, pipeline_result):
        non_ok = pipeline_result.uncertain_rows + pipeline_result.failed_rows
        assert non_ok >= 2, f"Expected >=2 non-ok rows, got {non_ok}"


# ===========================================================================
# Test: Postprocessing pipeline stages in isolation
# ===========================================================================

class TestPostprocessingStages:

    def test_parse_then_map_then_validate(self):
        raw = MOCK_AI_RESPONSE_COUPON_1
        rows, error = _parse_ai_response(raw)
        assert error is None
        assert len(rows) == 13
        rows = apply_team_mapping(rows)
        assert rows[1].home_team == "Manchester United"
        rows = validate_rows(rows)
        for row in rows:
            assert row.row_status == "ok"

    def test_parse_partial_rows_have_issues(self):
        raw = MOCK_AI_RESPONSE_COUPON_2
        rows, error = _parse_ai_response(raw)
        assert error is None
        assert len(rows) == 8
        row5 = rows[4]
        row6 = rows[5]
        # Odds are None for rows 5-6
        assert row5.odds_1 is None
        assert row6.odds_1 is None
        # These rows have AI-reported field_issues in their issues list
        assert any("AI-rapporterade" in i for i in row5.issues)
        assert any("AI-rapporterade" in i for i in row6.issues)

    def test_build_result_aggregates_correctly(self):
        rows = [
            ScannedRow(row_status="ok"),
            ScannedRow(row_status="ok"),
            ScannedRow(row_status="uncertain"),
            ScannedRow(row_status="failed"),
        ]
        result = _build_scanner_result(rows, "raw", ["step1", "step2"])
        assert result.total_rows == 4
        assert result.ok_rows == 2
        assert result.uncertain_rows == 1
        assert result.failed_rows == 1
        assert result.preprocessing_applied == ["step1", "step2"]
