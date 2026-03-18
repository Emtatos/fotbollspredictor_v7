"""
Tester for odds_tool.py -- karnlogik for oddsverktyget.

Testar:
- implicita sannolikheter fran odds
- overround-berakning
- overround-normalisering (fair probs summerar till 1)
- defensiv hantering av ogiltiga odds
- best-odds-logik
- rapport-generering
"""

import pytest
from odds_tool import (
    OddsEntry,
    MatchOddsReport,
    odds_to_implied_probs,
    compute_overround,
    remove_overround,
    find_best_odds,
    combined_best_overround,
    build_match_report,
    extract_odds_from_row,
    format_report,
)


# ---------------------------------------------------------------------------
# odds_to_implied_probs
# ---------------------------------------------------------------------------

class TestOddsToImpliedProbs:
    def test_basic_conversion(self):
        result = odds_to_implied_probs(2.0, 3.0, 4.0)
        assert abs(result["1"] - 0.50) < 1e-9
        assert abs(result["X"] - 1.0 / 3.0) < 1e-9
        assert abs(result["2"] - 0.25) < 1e-9

    def test_even_odds(self):
        result = odds_to_implied_probs(3.0, 3.0, 3.0)
        for key in ["1", "X", "2"]:
            assert abs(result[key] - 1.0 / 3.0) < 1e-9

    def test_heavy_favourite(self):
        result = odds_to_implied_probs(1.10, 8.0, 15.0)
        assert result["1"] > result["X"] > result["2"]
        # Implied sum should be > 1 (bookmaker margin)
        assert sum(result.values()) > 1.0

    def test_raises_on_odds_equal_one(self):
        with pytest.raises(ValueError):
            odds_to_implied_probs(1.0, 3.0, 4.0)

    def test_raises_on_odds_less_than_one(self):
        with pytest.raises(ValueError):
            odds_to_implied_probs(0.5, 3.0, 4.0)

    def test_raises_on_negative_odds(self):
        with pytest.raises(ValueError):
            odds_to_implied_probs(-2.0, 3.0, 4.0)

    def test_raises_on_non_numeric(self):
        with pytest.raises(TypeError):
            odds_to_implied_probs("abc", 3.0, 4.0)

    def test_raises_on_none(self):
        with pytest.raises(TypeError):
            odds_to_implied_probs(None, 3.0, 4.0)

    def test_integer_odds_accepted(self):
        result = odds_to_implied_probs(2, 3, 4)
        assert abs(result["1"] - 0.50) < 1e-9


# ---------------------------------------------------------------------------
# compute_overround
# ---------------------------------------------------------------------------

class TestComputeOverround:
    def test_typical_overround(self):
        implied = odds_to_implied_probs(2.10, 3.40, 3.60)
        overround = compute_overround(implied)
        assert overround > 0  # bookmaker margin is positive
        assert overround < 20  # sanity check

    def test_no_margin(self):
        # Perfectly fair odds: 1/sum = 1
        fair_probs = {"1": 0.5, "X": 0.3, "2": 0.2}
        overround = compute_overround(fair_probs)
        assert abs(overround) < 1e-9

    def test_known_overround(self):
        # If implied probs sum to 1.05, overround is 5%
        implied = {"1": 0.50, "X": 0.30, "2": 0.25}
        overround = compute_overround(implied)
        assert abs(overround - 5.0) < 1e-9


# ---------------------------------------------------------------------------
# remove_overround
# ---------------------------------------------------------------------------

class TestRemoveOverround:
    def test_sums_to_one(self):
        implied = odds_to_implied_probs(2.10, 3.40, 3.60)
        fair = remove_overround(implied)
        total = sum(fair.values())
        assert abs(total - 1.0) < 1e-9

    def test_preserves_relative_order(self):
        implied = odds_to_implied_probs(1.50, 4.00, 6.00)
        fair = remove_overround(implied)
        assert fair["1"] > fair["X"] > fair["2"]

    def test_already_fair(self):
        fair_probs = {"1": 0.5, "X": 0.3, "2": 0.2}
        result = remove_overround(fair_probs)
        assert abs(result["1"] - 0.5) < 1e-9
        assert abs(result["X"] - 0.3) < 1e-9
        assert abs(result["2"] - 0.2) < 1e-9

    def test_raises_on_zero_sum(self):
        with pytest.raises(ValueError):
            remove_overround({"1": 0.0, "X": 0.0, "2": 0.0})

    def test_with_high_margin(self):
        # Odds with ~10% overround
        implied = odds_to_implied_probs(1.80, 3.20, 4.00)
        fair = remove_overround(implied)
        total = sum(fair.values())
        assert abs(total - 1.0) < 1e-9
        # All fair probs should be lower than implied
        for key in ["1", "X", "2"]:
            assert fair[key] <= implied[key]


# ---------------------------------------------------------------------------
# find_best_odds
# ---------------------------------------------------------------------------

class TestFindBestOdds:
    def test_single_bookmaker(self):
        entries = [OddsEntry("Bet365", 2.10, 3.40, 3.60)]
        best = find_best_odds(entries)
        assert best["1"] == (2.10, "Bet365")
        assert best["X"] == (3.40, "Bet365")
        assert best["2"] == (3.60, "Bet365")

    def test_multiple_bookmakers(self):
        entries = [
            OddsEntry("Bet365", 2.10, 3.40, 3.60),
            OddsEntry("Pinnacle", 2.15, 3.30, 3.70),
            OddsEntry("bwin", 2.05, 3.50, 3.55),
        ]
        best = find_best_odds(entries)
        assert best["1"] == (2.15, "Pinnacle")
        assert best["X"] == (3.50, "bwin")
        assert best["2"] == (3.70, "Pinnacle")

    def test_empty_list(self):
        best = find_best_odds([])
        assert best == {}

    def test_skips_invalid_odds(self):
        entries = [
            OddsEntry("Bad", 0.5, 3.00, 3.00),
            OddsEntry("Good", 2.00, 3.00, 3.00),
        ]
        best = find_best_odds(entries)
        # Home odds from "Bad" (0.5) should be skipped
        assert best["1"] == (2.00, "Good")


# ---------------------------------------------------------------------------
# combined_best_overround
# ---------------------------------------------------------------------------

class TestCombinedBestOverround:
    def test_lower_than_single_bookmaker(self):
        entries = [
            OddsEntry("A", 2.10, 3.40, 3.60),
            OddsEntry("B", 2.20, 3.30, 3.70),
        ]
        best = find_best_odds(entries)
        combined_or = combined_best_overround(best)

        # Single bookmaker overround for A
        implied_a = odds_to_implied_probs(2.10, 3.40, 3.60)
        single_or = compute_overround(implied_a)

        assert combined_or < single_or

    def test_empty_best_odds(self):
        assert combined_best_overround({}) == 0.0


# ---------------------------------------------------------------------------
# build_match_report
# ---------------------------------------------------------------------------

class TestBuildMatchReport:
    def test_basic_report(self):
        entries = [OddsEntry("Bet365", 2.10, 3.40, 3.60)]
        report = build_match_report("Arsenal", "Liverpool", entries)
        assert report is not None
        assert report.home_team == "Arsenal"
        assert report.away_team == "Liverpool"
        assert abs(sum(report.fair_probs.values()) - 1.0) < 1e-9
        assert report.overround > 0

    def test_returns_none_for_no_valid_odds(self):
        entries = [OddsEntry("Bad", 0.5, 0.5, 0.5)]
        report = build_match_report("A", "B", entries)
        assert report is None

    def test_returns_none_for_empty_entries(self):
        report = build_match_report("A", "B", [])
        assert report is None

    def test_multi_bookmaker_report(self):
        entries = [
            OddsEntry("Bet365", 2.10, 3.40, 3.60),
            OddsEntry("Pinnacle", 2.15, 3.35, 3.65),
        ]
        report = build_match_report("Home", "Away", entries)
        assert report is not None
        assert len(report.bookmaker_odds) == 2
        assert "1" in report.best_odds
        assert "X" in report.best_odds
        assert "2" in report.best_odds


# ---------------------------------------------------------------------------
# extract_odds_from_row
# ---------------------------------------------------------------------------

class TestExtractOddsFromRow:
    def test_extracts_bet365(self):
        import pandas as pd
        row = pd.Series({"B365H": 2.10, "B365D": 3.40, "B365A": 3.60})
        entries = extract_odds_from_row(row, row.index)
        assert len(entries) == 1
        assert entries[0].bookmaker == "Bet365"
        assert entries[0].home == 2.10

    def test_extracts_multiple_bookmakers(self):
        import pandas as pd
        row = pd.Series({
            "B365H": 2.10, "B365D": 3.40, "B365A": 3.60,
            "PSH": 2.15, "PSD": 3.35, "PSA": 3.65,
        })
        entries = extract_odds_from_row(row, row.index)
        assert len(entries) == 2
        bookmakers = {e.bookmaker for e in entries}
        assert "Bet365" in bookmakers
        assert "Pinnacle" in bookmakers

    def test_skips_missing_values(self):
        import pandas as pd
        row = pd.Series({"B365H": 2.10, "B365D": None, "B365A": 3.60})
        entries = extract_odds_from_row(row, row.index)
        assert len(entries) == 0

    def test_empty_row(self):
        import pandas as pd
        row = pd.Series({"SomeCol": 42})
        entries = extract_odds_from_row(row, row.index)
        assert len(entries) == 0


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------

class TestFormatReport:
    def test_format_produces_string(self):
        entries = [OddsEntry("Bet365", 2.10, 3.40, 3.60)]
        report = build_match_report("Arsenal", "Liverpool", entries)
        text = format_report(report)
        assert "Arsenal" in text
        assert "Liverpool" in text
        assert "Overround" in text
        assert "Bet365" in text
