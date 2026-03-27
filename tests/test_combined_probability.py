# tests/test_combined_probability.py
"""Tester för combined_probability.py"""
import pytest
import numpy as np
from combined_probability import (
    odds_to_fair_probs,
    combine_probabilities,
    build_combined_match,
    CombinedMatchProbability,
)


class TestOddsToFairProbs:
    def test_basic_conversion(self):
        """Odds 2.0/3.0/4.0 ska ge fair probs som summerar till 1.0."""
        probs = odds_to_fair_probs(2.0, 3.0, 4.0)
        assert abs(probs.sum() - 1.0) < 1e-9
        assert probs[0] > probs[1] > probs[2]  # favorit → outsider

    def test_equal_odds(self):
        """Lika odds ska ge ~33% vardera."""
        probs = odds_to_fair_probs(3.0, 3.0, 3.0)
        assert abs(probs[0] - 1/3) < 1e-9
        assert abs(probs[1] - 1/3) < 1e-9

    def test_invalid_odds_fallback(self):
        """Ogiltiga odds (<=1.0) ska ge uniform."""
        probs = odds_to_fair_probs(0.5, 3.0, 4.0)
        assert abs(probs[0] - 1/3) < 1e-9


class TestCombineProbabilities:
    def test_only_odds(self):
        """Med bara odds ska resultatet vara odds-probs."""
        odds = np.array([0.5, 0.3, 0.2])
        result = combine_probabilities(odds_probs=odds)
        np.testing.assert_array_almost_equal(result, odds)

    def test_only_model(self):
        """Med bara modell ska resultatet vara modell-probs."""
        model = np.array([0.4, 0.35, 0.25])
        result = combine_probabilities(model_probs=model)
        np.testing.assert_array_almost_equal(result, model)

    def test_all_three_sources(self):
        """Med alla tre ska odds väga mest."""
        odds = np.array([0.50, 0.30, 0.20])
        model = np.array([0.40, 0.35, 0.25])
        streck = np.array([0.60, 0.20, 0.20])
        result = combine_probabilities(
            odds_probs=odds, model_probs=model, streck_pcts=streck
        )
        # Odds säger stark hemma, streck ännu starkare, modell jämnare
        # Resultatet ska ligga närmast odds (50% vikt)
        assert result[0] > 0.45  # hemma ska vara hög
        assert abs(result.sum() - 1.0) < 1e-9

    def test_no_sources_gives_uniform(self):
        """Utan data → uniform fördelning."""
        result = combine_probabilities()
        np.testing.assert_array_almost_equal(result, [1/3, 1/3, 1/3])

    def test_weight_redistribution(self):
        """Om en källa saknas fördelas vikten på övriga."""
        odds = np.array([0.50, 0.30, 0.20])
        model = np.array([0.40, 0.35, 0.25])
        # Utan streck → odds 50/(50+35)=58.8%, modell 41.2%
        result = combine_probabilities(odds_probs=odds, model_probs=model)
        # Resultatet ska ligga närmare odds
        assert result[0] > 0.45


class TestBuildCombinedMatch:
    def test_full_data(self):
        """Test med alla tre signaler."""
        cm = build_combined_match(
            home_team="Arsenal",
            away_team="Chelsea",
            odds_1=2.0, odds_x=3.5, odds_2=4.0,
            model_probs=np.array([0.45, 0.30, 0.25]),
            streck_1=55, streck_x=25, streck_2=20,
        )
        assert cm.home_team == "Arsenal"
        assert cm.sources["odds"] is True
        assert cm.sources["model"] is True
        assert cm.sources["streck"] is True
        assert abs(cm.prob_1 + cm.prob_x + cm.prob_2 - 1.0) < 1e-9
        assert 0 < cm.entropy < 1

    def test_only_odds_and_streck(self):
        """Utan modell ska det fortfarande fungera."""
        cm = build_combined_match(
            home_team="Liverpool",
            away_team="Everton",
            odds_1=1.60, odds_x=3.90, odds_2=5.50,
            streck_1=66, streck_x=18, streck_2=16,
        )
        assert cm.sources["model"] is False
        assert cm.prob_1 > cm.prob_x > cm.prob_2

    def test_streck_delta_calculated(self):
        """Streck-delta ska visa överstreckat korrekt."""
        cm = build_combined_match(
            home_team="Wycombe",
            away_team="Port Vale",
            odds_1=1.60, odds_x=3.90, odds_2=5.50,
            streck_1=66, streck_x=18, streck_2=16,
        )
        # Odds implicerar ~62.5% hemma, streck säger 66% → överstreckat hemma
        assert cm.streck_delta_1 > 0  # positivt = överstreckat

    def test_no_data(self):
        """Utan allt → uniform, max entropy."""
        cm = build_combined_match(
            home_team="A", away_team="B",
        )
        assert abs(cm.prob_1 - 1/3) < 1e-6
        assert cm.entropy > 0.99


class TestHalfGuardIntegration:
    """Testar att halvgarderingarna väljer rätt med kombinerade probs."""

    def test_uncertain_match_gets_guard(self):
        """Mest osäker match ska halvgarderas."""
        from ui_utils import pick_half_guards_combined, get_halfguard_sign_combined

        certain = build_combined_match(
            "Stockport", "Wimbledon",
            odds_1=1.88, odds_x=4.00, odds_2=3.60,
            streck_1=67, streck_x=17, streck_2=16,
        )
        uncertain = build_combined_match(
            "Exeter", "Leyton Orient",
            odds_1=2.38, odds_x=3.40, odds_2=2.88,
            streck_1=38, streck_x=25, streck_2=37,
        )

        guards = pick_half_guards_combined([certain, uncertain], n_guards=1)
        assert guards == [1]  # Exeter-matchen är mest osäker

    def test_halfguard_sign_uses_combined(self):
        """Halvgarderingstecknet ska baseras på kombinerad prob."""
        from ui_utils import get_halfguard_sign_combined

        cm = build_combined_match(
            "Barnet", "Cambridge",
            odds_1=2.55, odds_x=3.10, odds_2=2.70,
            streck_1=34, streck_x=28, streck_2=38,
        )
        sign = get_halfguard_sign_combined(cm)
        # Oavgjort minst sannolikt → ska strykas → "12"
        assert "X" not in sign or cm.prob_x > min(cm.prob_1, cm.prob_2)
