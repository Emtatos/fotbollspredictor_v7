"""
Tester for streck_analysis.py -- streckjamforelse ovanpa oddsverktyget.

Testar:
- delta-berakning
- klassificering (understreckad/neutral/overstreckad)
- neutralzon
- input-validering (procent- vs decimalformat)
- rapport-generering
- ranking (utfall och matcher)
- defensiv hantering av ogiltiga/ofullstandiga varden
"""

import pytest
from odds_tool import OddsEntry, build_match_report
from streck_analysis import (
    compute_streck_delta,
    classify_streck_value,
    build_streck_report,
    build_streck_report_from_odds_report,
    rank_outcomes_by_streck_delta,
    rank_matches_by_streck_interest,
    format_streck_report,
    OutcomeStreck,
    MatchStreckReport,
    STRECK_NEUTRAL_THRESHOLD,
    _validate_streck_input,
)


# ---------------------------------------------------------------------------
# compute_streck_delta
# ---------------------------------------------------------------------------

class TestComputeStreckDelta:
    def test_positive_delta_overstreckad(self):
        # streck 50%, fair 40% -> overstreckad, delta = +0.10
        delta = compute_streck_delta(0.50, 0.40)
        assert abs(delta - 0.10) < 1e-9

    def test_negative_delta_understreckad(self):
        # streck 30%, fair 45% -> understreckad, delta = -0.15
        delta = compute_streck_delta(0.30, 0.45)
        assert abs(delta - (-0.15)) < 1e-9

    def test_zero_delta(self):
        delta = compute_streck_delta(0.40, 0.40)
        assert abs(delta) < 1e-9

    def test_extreme_understreck(self):
        delta = compute_streck_delta(0.05, 0.60)
        assert abs(delta - (-0.55)) < 1e-9

    def test_extreme_overstreck(self):
        delta = compute_streck_delta(0.80, 0.20)
        assert abs(delta - 0.60) < 1e-9

    def test_small_delta(self):
        delta = compute_streck_delta(0.331, 0.330)
        assert abs(delta - 0.001) < 1e-9


# ---------------------------------------------------------------------------
# classify_streck_value
# ---------------------------------------------------------------------------

class TestClassifyStreckValue:
    def test_overstreckad(self):
        assert classify_streck_value(0.05) == "overstreckad"

    def test_understreckad(self):
        assert classify_streck_value(-0.05) == "understreckad"

    def test_neutral_zero(self):
        assert classify_streck_value(0.0) == "neutral"

    def test_neutral_within_threshold(self):
        assert classify_streck_value(0.01) == "neutral"
        assert classify_streck_value(-0.01) == "neutral"

    def test_exactly_at_threshold(self):
        # Exactly at threshold is still neutral
        assert classify_streck_value(STRECK_NEUTRAL_THRESHOLD) == "neutral"
        assert classify_streck_value(-STRECK_NEUTRAL_THRESHOLD) == "neutral"

    def test_just_above_threshold(self):
        assert classify_streck_value(STRECK_NEUTRAL_THRESHOLD + 0.001) == "overstreckad"

    def test_just_below_negative_threshold(self):
        assert classify_streck_value(-STRECK_NEUTRAL_THRESHOLD - 0.001) == "understreckad"

    def test_custom_threshold(self):
        assert classify_streck_value(0.03, threshold=0.05) == "neutral"
        assert classify_streck_value(0.06, threshold=0.05) == "overstreckad"
        assert classify_streck_value(-0.06, threshold=0.05) == "understreckad"


# ---------------------------------------------------------------------------
# _validate_streck_input
# ---------------------------------------------------------------------------

class TestValidateStreckInput:
    def test_percent_format_normalized(self):
        """Procent-format (summa ~100) ska normaliseras till decimalform."""
        result = _validate_streck_input({"1": 45.0, "X": 28.0, "2": 27.0})
        assert result is not None
        assert abs(sum(result.values()) - 1.0) < 1e-9
        assert abs(result["1"] - 0.45) < 1e-9
        assert abs(result["X"] - 0.28) < 1e-9
        assert abs(result["2"] - 0.27) < 1e-9

    def test_decimal_format_passthrough(self):
        """Decimalform (summa ~1) ska passera igenom."""
        result = _validate_streck_input({"1": 0.40, "X": 0.30, "2": 0.30})
        assert result is not None
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_percent_format_not_exactly_100(self):
        """Procentvarden som summerar till nara men inte exakt 100."""
        result = _validate_streck_input({"1": 40.0, "X": 30.0, "2": 28.0})
        assert result is not None
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_missing_key_returns_none(self):
        result = _validate_streck_input({"1": 40.0, "X": 30.0})
        assert result is None

    def test_negative_value_returns_none(self):
        result = _validate_streck_input({"1": -10.0, "X": 60.0, "2": 50.0})
        assert result is None

    def test_all_zero_returns_none(self):
        result = _validate_streck_input({"1": 0.0, "X": 0.0, "2": 0.0})
        assert result is None

    def test_non_numeric_returns_none(self):
        result = _validate_streck_input({"1": "abc", "X": 30.0, "2": 30.0})
        assert result is None

    def test_integer_values_accepted(self):
        result = _validate_streck_input({"1": 50, "X": 25, "2": 25})
        assert result is not None
        assert abs(result["1"] - 0.50) < 1e-9


# ---------------------------------------------------------------------------
# build_streck_report
# ---------------------------------------------------------------------------

class TestBuildStreckReport:
    def test_basic_report(self):
        fair = {"1": 0.45, "X": 0.28, "2": 0.27}
        streck = {"1": 50.0, "X": 25.0, "2": 25.0}
        sr = build_streck_report("Arsenal", "Liverpool", fair, streck)
        assert sr is not None
        assert sr.home_team == "Arsenal"
        assert sr.away_team == "Liverpool"
        assert len(sr.outcomes) == 3

    def test_delta_calculated_correctly(self):
        fair = {"1": 0.40, "X": 0.30, "2": 0.30}
        streck = {"1": 50.0, "X": 25.0, "2": 25.0}
        sr = build_streck_report("A", "B", fair, streck)
        assert sr is not None
        # streck normalized: 1=0.50, X=0.25, 2=0.25
        # delta: 1=+0.10, X=-0.05, 2=-0.05
        for o in sr.outcomes:
            if o.outcome == "1":
                assert abs(o.delta - 0.10) < 1e-9
                assert o.label == "overstreckad"
            elif o.outcome == "X":
                assert abs(o.delta - (-0.05)) < 1e-9
                assert o.label == "understreckad"
            elif o.outcome == "2":
                assert abs(o.delta - (-0.05)) < 1e-9
                assert o.label == "understreckad"

    def test_max_abs_delta(self):
        fair = {"1": 0.40, "X": 0.30, "2": 0.30}
        streck = {"1": 55.0, "X": 25.0, "2": 20.0}
        sr = build_streck_report("A", "B", fair, streck)
        assert sr is not None
        assert sr.max_abs_delta == max(abs(o.delta) for o in sr.outcomes)

    def test_all_neutral(self):
        fair = {"1": 0.40, "X": 0.30, "2": 0.30}
        streck = {"1": 40.0, "X": 30.0, "2": 30.0}
        sr = build_streck_report("A", "B", fair, streck)
        assert sr is not None
        for o in sr.outcomes:
            assert o.label == "neutral"

    def test_returns_none_for_missing_fair_prob_key(self):
        fair = {"1": 0.50, "X": 0.30}  # missing "2"
        streck = {"1": 40.0, "X": 30.0, "2": 30.0}
        sr = build_streck_report("A", "B", fair, streck)
        assert sr is None

    def test_returns_none_for_invalid_streck(self):
        fair = {"1": 0.45, "X": 0.28, "2": 0.27}
        streck = {"1": -10.0, "X": 60.0, "2": 50.0}
        sr = build_streck_report("A", "B", fair, streck)
        assert sr is None

    def test_decimal_streck_input(self):
        """Streck i decimalform ska fungera lika bra."""
        fair = {"1": 0.40, "X": 0.30, "2": 0.30}
        streck = {"1": 0.50, "X": 0.25, "2": 0.25}
        sr = build_streck_report("A", "B", fair, streck)
        assert sr is not None
        for o in sr.outcomes:
            if o.outcome == "1":
                assert abs(o.delta - 0.10) < 1e-9


# ---------------------------------------------------------------------------
# build_streck_report_from_odds_report
# ---------------------------------------------------------------------------

class TestBuildStreckReportFromOddsReport:
    def test_uses_fair_probs_from_odds_report(self):
        entries = [OddsEntry("Bet365", 2.10, 3.40, 3.60)]
        report = build_match_report("Arsenal", "Liverpool", entries)
        streck = {"1": 50.0, "X": 25.0, "2": 25.0}
        sr = build_streck_report_from_odds_report(report, streck)
        assert sr is not None
        assert sr.home_team == "Arsenal"
        # Verify fair probs come from the odds report
        for o in sr.outcomes:
            assert abs(o.fair_prob - report.fair_probs[o.outcome]) < 1e-9

    def test_returns_none_for_empty_fair_probs(self):
        from odds_tool import MatchOddsReport
        report = MatchOddsReport(
            home_team="A", away_team="B",
            bookmaker_odds=[OddsEntry("Test", 2.0, 3.0, 4.0)],
            fair_probs={},
        )
        sr = build_streck_report_from_odds_report(report, {"1": 40, "X": 30, "2": 30})
        assert sr is None


# ---------------------------------------------------------------------------
# rank_outcomes_by_streck_delta
# ---------------------------------------------------------------------------

class TestRankOutcomesByStreckDelta:
    def _make_sr(self, home, away, deltas):
        outcomes = []
        for key, d in zip(("1", "X", "2"), deltas):
            outcomes.append(OutcomeStreck(
                outcome=key,
                fair_prob=0.33,
                streck_pct=0.33 + d,
                delta=d,
                label=classify_streck_value(d),
            ))
        return MatchStreckReport(
            home_team=home,
            away_team=away,
            outcomes=outcomes,
            max_abs_delta=max(abs(d) for d in deltas),
        )

    def test_sorted_most_understreckad_first(self):
        sr1 = self._make_sr("A", "B", [0.05, -0.10, 0.02])
        sr2 = self._make_sr("C", "D", [-0.15, 0.08, -0.01])
        ranked = rank_outcomes_by_streck_delta([sr1, sr2])
        deltas = [item[2].delta for item in ranked]
        assert deltas == sorted(deltas)

    def test_empty_list(self):
        ranked = rank_outcomes_by_streck_delta([])
        assert ranked == []

    def test_single_match(self):
        sr = self._make_sr("A", "B", [0.05, 0.00, -0.08])
        ranked = rank_outcomes_by_streck_delta([sr])
        assert len(ranked) == 3
        assert ranked[0][2].delta == -0.08  # most understreckad first
        assert ranked[-1][2].delta == 0.05  # most overstreckad last

    def test_labels_in_output(self):
        sr = self._make_sr("A", "B", [0.05, 0.00, -0.08])
        ranked = rank_outcomes_by_streck_delta([sr])
        outcome_labels = {item[1] for item in ranked}
        assert "Hemma" in outcome_labels
        assert "Oavgjort" in outcome_labels
        assert "Borta" in outcome_labels


# ---------------------------------------------------------------------------
# rank_matches_by_streck_interest
# ---------------------------------------------------------------------------

class TestRankMatchesByStreckInterest:
    def _make_sr(self, home, away, max_delta):
        outcomes = [OutcomeStreck(
            outcome="1",
            fair_prob=0.40,
            streck_pct=0.40 + max_delta,
            delta=max_delta,
            label=classify_streck_value(max_delta),
        )]
        return MatchStreckReport(
            home_team=home,
            away_team=away,
            outcomes=outcomes,
            max_abs_delta=abs(max_delta),
        )

    def test_sorted_by_max_abs_delta(self):
        sr1 = self._make_sr("A", "B", 0.03)
        sr2 = self._make_sr("C", "D", -0.12)
        sr3 = self._make_sr("E", "F", 0.07)
        ranked = rank_matches_by_streck_interest([sr1, sr2, sr3])
        assert ranked[0].home_team == "C"  # abs(delta) = 0.12
        assert ranked[1].home_team == "E"  # abs(delta) = 0.07
        assert ranked[2].home_team == "A"  # abs(delta) = 0.03

    def test_empty_list(self):
        ranked = rank_matches_by_streck_interest([])
        assert ranked == []

    def test_single_match(self):
        sr = self._make_sr("A", "B", 0.05)
        ranked = rank_matches_by_streck_interest([sr])
        assert len(ranked) == 1


# ---------------------------------------------------------------------------
# format_streck_report
# ---------------------------------------------------------------------------

class TestFormatStreckReport:
    def test_format_includes_key_info(self):
        fair = {"1": 0.45, "X": 0.28, "2": 0.27}
        streck = {"1": 50.0, "X": 25.0, "2": 25.0}
        sr = build_streck_report("Arsenal", "Liverpool", fair, streck)
        text = format_streck_report(sr)
        assert "Arsenal" in text
        assert "Liverpool" in text
        assert "Delta" in text
        assert "beslutsstod" in text


# ---------------------------------------------------------------------------
# Defensiv hantering
# ---------------------------------------------------------------------------

class TestDefensiveHandling:
    def test_streck_with_missing_outcome_key(self):
        """build_streck_report returnerar None om streck saknar nyckel."""
        fair = {"1": 0.45, "X": 0.28, "2": 0.27}
        streck = {"1": 50.0, "X": 30.0}  # missing "2"
        sr = build_streck_report("A", "B", fair, streck)
        assert sr is None

    def test_streck_all_zero(self):
        fair = {"1": 0.45, "X": 0.28, "2": 0.27}
        streck = {"1": 0.0, "X": 0.0, "2": 0.0}
        sr = build_streck_report("A", "B", fair, streck)
        assert sr is None

    def test_streck_negative_values(self):
        fair = {"1": 0.45, "X": 0.28, "2": 0.27}
        streck = {"1": -5.0, "X": 55.0, "2": 50.0}
        sr = build_streck_report("A", "B", fair, streck)
        assert sr is None

    def test_delta_with_zero_values(self):
        delta = compute_streck_delta(0.0, 0.0)
        assert delta == 0.0

    def test_mixed_percent_and_decimal_not_confused(self):
        """Procentformat (summa > 1.5) ska inte forvaxlas med decimalformat."""
        # 40 + 30 + 30 = 100 -> procentformat
        result_pct = _validate_streck_input({"1": 40.0, "X": 30.0, "2": 30.0})
        assert result_pct is not None
        assert abs(result_pct["1"] - 0.40) < 1e-9

        # 0.40 + 0.30 + 0.30 = 1.0 -> decimalformat
        result_dec = _validate_streck_input({"1": 0.40, "X": 0.30, "2": 0.30})
        assert result_dec is not None
        assert abs(result_dec["1"] - 0.40) < 1e-9
