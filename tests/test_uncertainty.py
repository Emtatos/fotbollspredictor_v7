"""
Enhetstester för uncertainty.py - entropy-baserad osäkerhetsberäkning
"""
import pytest
import math
from uncertainty import entropy_norm


class TestEntropyNorm:
    """Tester för entropy_norm-funktionen"""

    def test_fully_certain_home_win(self):
        """Testar att entropy är nära 0 för helt säker prediktion (hemmavinst)"""
        result = entropy_norm(1.0, 0.0, 0.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_fully_certain_draw(self):
        """Testar att entropy är nära 0 för helt säker prediktion (oavgjort)"""
        result = entropy_norm(0.0, 1.0, 0.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_fully_certain_away_win(self):
        """Testar att entropy är nära 0 för helt säker prediktion (bortavinst)"""
        result = entropy_norm(0.0, 0.0, 1.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_maximally_uncertain_uniform(self):
        """Testar att entropy är 1 för uniform fördelning (1/3, 1/3, 1/3)"""
        result = entropy_norm(1/3, 1/3, 1/3)
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_two_equal_outcomes(self):
        """Testar att entropy för (0.5, 0.5, 0.0) är mellan 0 och 1"""
        result = entropy_norm(0.5, 0.5, 0.0)
        assert 0.0 < result < 1.0
        # Specifikt: H = -2*(0.5*log(0.5)) = log(2), normaliserat = log(2)/log(3) ≈ 0.63
        expected = math.log(2) / math.log(3)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_entropy_in_valid_range(self):
        """Testar att entropy alltid är i intervallet [0, 1]"""
        test_cases = [
            (0.6, 0.3, 0.1),
            (0.4, 0.35, 0.25),
            (0.7, 0.2, 0.1),
            (0.33, 0.34, 0.33),
            (0.9, 0.05, 0.05),
            (0.1, 0.1, 0.8),
        ]
        for p1, px, p2 in test_cases:
            result = entropy_norm(p1, px, p2)
            assert 0.0 <= result <= 1.0, f"Entropy {result} out of range for ({p1}, {px}, {p2})"

    def test_higher_entropy_for_more_uncertain(self):
        """Testar att mer osäkra fördelningar ger högre entropy"""
        # Tydlig favorit
        certain = entropy_norm(0.7, 0.2, 0.1)
        # Osäker match
        uncertain = entropy_norm(0.4, 0.35, 0.25)
        # Uniform
        uniform = entropy_norm(1/3, 1/3, 1/3)

        assert certain < uncertain < uniform

    def test_renormalization(self):
        """Testar att funktionen hanterar icke-normaliserade sannolikheter"""
        # Sannolikheter som inte summerar till 1
        result = entropy_norm(0.6, 0.6, 0.6)
        # Ska renormaliseras till (1/3, 1/3, 1/3) och ge entropy ≈ 1
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_handles_small_probabilities(self):
        """Testar att funktionen hanterar mycket små sannolikheter"""
        result = entropy_norm(0.99, 0.005, 0.005)
        assert 0.0 <= result <= 1.0
        # Ska vara nära 0 eftersom en klass dominerar
        assert result < 0.2
