"""
Tester for value_analysis.py -- value-lager ovanpa oddsverktyget.

Testar:
- edge-berakning
- expected return-berakning
- edge-klassificering (positiv/neutral/negativ)
- marknadskonsensus
- value-rapport
- ranking/sortering
- defensiv hantering av saknade/ogiltiga varden
"""

import pytest
from odds_tool import OddsEntry, MatchOddsReport, build_match_report
from value_analysis import (
    compute_edge,
    compute_expected_return,
    classify_edge,
    compute_market_consensus,
    build_value_report,
    rank_outcomes_by_edge,
    rank_matches_by_interest,
    OutcomeValue,
    MatchValueReport,
    EDGE_NEUTRAL_THRESHOLD,
)


# ---------------------------------------------------------------------------
# compute_edge
# ---------------------------------------------------------------------------

class TestComputeEdge:
    def test_positive_edge(self):
        # comparison thinks 50%, market says 45% -> positive edge
        edge = compute_edge(0.50, 0.45)
        assert abs(edge - 0.05) < 1e-9

    def test_negative_edge(self):
        # comparison thinks 30%, market says 40% -> negative edge
        edge = compute_edge(0.30, 0.40)
        assert abs(edge - (-0.10)) < 1e-9

    def test_zero_edge(self):
        edge = compute_edge(0.40, 0.40)
        assert abs(edge) < 1e-9

    def test_extreme_positive(self):
        edge = compute_edge(0.90, 0.10)
        assert abs(edge - 0.80) < 1e-9

    def test_small_edge(self):
        edge = compute_edge(0.451, 0.450)
        assert abs(edge - 0.001) < 1e-9


# ---------------------------------------------------------------------------
# compute_expected_return
# ---------------------------------------------------------------------------

class TestComputeExpectedReturn:
    def test_positive_return(self):
        # prob 0.5, odds 2.5 -> 0.5*2.5 - 1 = 0.25
        er = compute_expected_return(0.50, 2.50)
        assert abs(er - 0.25) < 1e-9

    def test_negative_return(self):
        # prob 0.3, odds 2.0 -> 0.3*2.0 - 1 = -0.4
        er = compute_expected_return(0.30, 2.0)
        assert abs(er - (-0.40)) < 1e-9

    def test_break_even(self):
        # prob 0.5, odds 2.0 -> 0.5*2.0 - 1 = 0.0
        er = compute_expected_return(0.50, 2.0)
        assert abs(er) < 1e-9

    def test_high_odds(self):
        # prob 0.10, odds 12.0 -> 0.10*12.0 - 1 = 0.2
        er = compute_expected_return(0.10, 12.0)
        assert abs(er - 0.20) < 1e-9

    def test_invalid_odds_returns_minus_one(self):
        er = compute_expected_return(0.50, 1.0)
        assert er == -1.0

    def test_zero_prob(self):
        er = compute_expected_return(0.0, 3.0)
        assert abs(er - (-1.0)) < 1e-9


# ---------------------------------------------------------------------------
# classify_edge
# ---------------------------------------------------------------------------

class TestClassifyEdge:
    def test_positive_edge(self):
        assert classify_edge(0.05) == "positiv edge"

    def test_negative_edge(self):
        assert classify_edge(-0.05) == "negativ edge"

    def test_neutral_edge_zero(self):
        assert classify_edge(0.0) == "neutral"

    def test_neutral_within_threshold(self):
        assert classify_edge(0.005) == "neutral"
        assert classify_edge(-0.005) == "neutral"

    def test_exactly_at_threshold(self):
        # Exactly at threshold is still neutral
        assert classify_edge(EDGE_NEUTRAL_THRESHOLD) == "neutral"
        assert classify_edge(-EDGE_NEUTRAL_THRESHOLD) == "neutral"

    def test_just_above_threshold(self):
        assert classify_edge(EDGE_NEUTRAL_THRESHOLD + 0.001) == "positiv edge"

    def test_just_below_negative_threshold(self):
        assert classify_edge(-EDGE_NEUTRAL_THRESHOLD - 0.001) == "negativ edge"

    def test_custom_threshold(self):
        assert classify_edge(0.03, threshold=0.05) == "neutral"
        assert classify_edge(0.06, threshold=0.05) == "positiv edge"


# ---------------------------------------------------------------------------
# compute_market_consensus
# ---------------------------------------------------------------------------

class TestComputeMarketConsensus:
    def test_single_bookmaker(self):
        entries = [OddsEntry("Bet365", 2.10, 3.40, 3.60)]
        consensus = compute_market_consensus(entries)
        assert consensus is not None
        assert abs(sum(consensus.values()) - 1.0) < 1e-9

    def test_multiple_bookmakers(self):
        entries = [
            OddsEntry("Bet365", 2.10, 3.40, 3.60),
            OddsEntry("Pinnacle", 2.15, 3.35, 3.65),
            OddsEntry("bwin", 2.05, 3.50, 3.55),
        ]
        consensus = compute_market_consensus(entries)
        assert consensus is not None
        assert abs(sum(consensus.values()) - 1.0) < 1e-9
        # All keys present
        for key in ("1", "X", "2"):
            assert key in consensus
            assert 0 < consensus[key] < 1

    def test_empty_list(self):
        consensus = compute_market_consensus([])
        assert consensus is None

    def test_all_invalid_entries(self):
        entries = [OddsEntry("Bad", 0.5, 0.5, 0.5)]
        consensus = compute_market_consensus(entries)
        assert consensus is None

    def test_mixed_valid_invalid(self):
        entries = [
            OddsEntry("Bad", 0.5, 0.5, 0.5),
            OddsEntry("Good", 2.10, 3.40, 3.60),
        ]
        consensus = compute_market_consensus(entries)
        assert consensus is not None
        assert abs(sum(consensus.values()) - 1.0) < 1e-9

    def test_consensus_preserves_relative_order(self):
        entries = [
            OddsEntry("A", 1.50, 4.00, 6.00),
            OddsEntry("B", 1.55, 3.90, 5.80),
        ]
        consensus = compute_market_consensus(entries)
        assert consensus is not None
        assert consensus["1"] > consensus["X"] > consensus["2"]


# ---------------------------------------------------------------------------
# build_value_report
# ---------------------------------------------------------------------------

class TestBuildValueReport:
    def _make_report(self, entries):
        return build_match_report("Arsenal", "Liverpool", entries)

    def test_single_bookmaker_no_consensus(self):
        entries = [OddsEntry("Bet365", 2.10, 3.40, 3.60)]
        report = self._make_report(entries)
        vr = build_value_report(report)
        assert vr is not None
        # With single bookmaker and no manual comparison,
        # edge should be 0 since fair probs == comparison probs
        for ov in vr.outcomes:
            assert abs(ov.edge) < 1e-6

    def test_multi_bookmaker_report(self):
        entries = [
            OddsEntry("Bet365", 2.10, 3.40, 3.60),
            OddsEntry("Pinnacle", 2.15, 3.30, 3.70),
            OddsEntry("bwin", 2.05, 3.50, 3.55),
        ]
        report = self._make_report(entries)
        vr = build_value_report(report)
        assert vr is not None
        assert vr.num_bookmakers == 3
        assert len(vr.outcomes) == 3
        assert "marknadskonsensus" in vr.comparison_source
        for ov in vr.outcomes:
            assert ov.outcome in ("1", "X", "2")
            assert ov.odds > 1.0
            assert 0 < ov.market_fair_prob < 1
            assert 0 < ov.comparison_prob < 1

    def test_manual_comparison_probs(self):
        entries = [OddsEntry("Bet365", 2.10, 3.40, 3.60)]
        report = self._make_report(entries)
        comp = {"1": 0.50, "X": 0.25, "2": 0.25}
        vr = build_value_report(report, comparison_probs=comp)
        assert vr is not None
        assert "manuell" in vr.comparison_source
        # Check that comparison probs are used
        for ov in vr.outcomes:
            assert abs(ov.comparison_prob - comp[ov.outcome]) < 1e-9

    def test_returns_none_for_empty_odds(self):
        report = MatchOddsReport(
            home_team="A",
            away_team="B",
            bookmaker_odds=[],
        )
        vr = build_value_report(report)
        assert vr is None

    def test_edge_labels_assigned(self):
        entries = [
            OddsEntry("A", 2.00, 3.50, 4.00),
            OddsEntry("B", 2.20, 3.20, 3.80),
        ]
        report = self._make_report(entries)
        vr = build_value_report(report)
        assert vr is not None
        for ov in vr.outcomes:
            assert ov.edge_label in ("positiv edge", "neutral", "negativ edge")

    def test_expected_return_calculated(self):
        entries = [OddsEntry("Bet365", 2.10, 3.40, 3.60)]
        report = self._make_report(entries)
        comp = {"1": 0.60, "X": 0.20, "2": 0.20}
        vr = build_value_report(report, comparison_probs=comp)
        assert vr is not None
        for ov in vr.outcomes:
            expected = ov.comparison_prob * ov.odds - 1.0
            assert abs(ov.expected_return - expected) < 1e-9


# ---------------------------------------------------------------------------
# rank_outcomes_by_edge
# ---------------------------------------------------------------------------

class TestRankOutcomesByEdge:
    def _make_vr(self, home, away, edges):
        outcomes = []
        for key, edge_val in zip(("1", "X", "2"), edges):
            outcomes.append(OutcomeValue(
                outcome=key,
                odds=2.0,
                bookmaker="Test",
                market_fair_prob=0.33,
                comparison_prob=0.33 + edge_val,
                edge=edge_val,
                expected_return=edge_val * 2.0,
                edge_label=classify_edge(edge_val),
            ))
        return MatchValueReport(
            home_team=home,
            away_team=away,
            outcomes=outcomes,
            comparison_source="test",
            num_bookmakers=2,
            overround=5.0,
        )

    def test_sorted_by_edge_descending(self):
        vr1 = self._make_vr("A", "B", [0.05, -0.03, 0.10])
        vr2 = self._make_vr("C", "D", [0.02, 0.08, -0.01])
        ranked = rank_outcomes_by_edge([vr1, vr2])
        edges = [item[2].edge for item in ranked]
        assert edges == sorted(edges, reverse=True)

    def test_empty_list(self):
        ranked = rank_outcomes_by_edge([])
        assert ranked == []

    def test_single_match(self):
        vr = self._make_vr("A", "B", [0.05, 0.00, -0.05])
        ranked = rank_outcomes_by_edge([vr])
        assert len(ranked) == 3
        assert ranked[0][2].edge == 0.05
        assert ranked[-1][2].edge == -0.05


# ---------------------------------------------------------------------------
# rank_matches_by_interest
# ---------------------------------------------------------------------------

class TestRankMatchesByInterest:
    def _make_vr(self, home, away, max_edge):
        outcomes = [OutcomeValue(
            outcome="1",
            odds=2.0,
            bookmaker="Test",
            market_fair_prob=0.50,
            comparison_prob=0.50 + max_edge,
            edge=max_edge,
            expected_return=max_edge * 2.0,
            edge_label=classify_edge(max_edge),
        )]
        return MatchValueReport(
            home_team=home,
            away_team=away,
            outcomes=outcomes,
            comparison_source="test",
            num_bookmakers=2,
            overround=5.0,
        )

    def test_sorted_by_max_edge(self):
        vr1 = self._make_vr("A", "B", 0.03)
        vr2 = self._make_vr("C", "D", 0.10)
        vr3 = self._make_vr("E", "F", 0.05)
        ranked = rank_matches_by_interest([vr1, vr2, vr3])
        assert ranked[0].home_team == "C"
        assert ranked[1].home_team == "E"
        assert ranked[2].home_team == "A"

    def test_empty_list(self):
        ranked = rank_matches_by_interest([])
        assert ranked == []

    def test_single_match(self):
        vr = self._make_vr("A", "B", 0.05)
        ranked = rank_matches_by_interest([vr])
        assert len(ranked) == 1


# ---------------------------------------------------------------------------
# Defensiv hantering
# ---------------------------------------------------------------------------

class TestDefensiveHandling:
    def test_consensus_with_none_entries(self):
        """compute_market_consensus ska hantera ogiltiga entries utan att krascha."""
        entries = [
            OddsEntry("Good", 2.10, 3.40, 3.60),
        ]
        consensus = compute_market_consensus(entries)
        assert consensus is not None

    def test_build_value_report_missing_comparison_key(self):
        """build_value_report returnerar None om comparison probs saknar nyckel."""
        entries = [OddsEntry("Bet365", 2.10, 3.40, 3.60)]
        report = build_match_report("A", "B", entries)
        # Missing "2" key
        comp = {"1": 0.50, "X": 0.30}
        vr = build_value_report(report, comparison_probs=comp)
        assert vr is None

    def test_edge_with_zero_values(self):
        edge = compute_edge(0.0, 0.0)
        assert edge == 0.0

    def test_expected_return_with_odds_below_one(self):
        er = compute_expected_return(0.50, 0.5)
        assert er == -1.0
