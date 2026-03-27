"""
Enhetstester för ui_utils.py
"""
import pytest
import numpy as np
from ui_utils import (
    parse_match_input, parse_match_input_with_errors,
    pick_half_guards, get_halfguard_sign, calculate_match_entropy,
    pick_half_guards_combined, get_halfguard_sign_combined,
)
from combined_probability import CombinedMatchProbability
from utils import set_canonical_teams


class TestParseMatchInput:
    """Tester för parse_match_input-funktionen"""
    
    def setup_method(self):
        """Sätt upp kanoniska lagnamn före varje test"""
        teams = ["Arsenal", "Chelsea", "Liverpool", "Manchester United", "Tottenham Hotspur"]
        set_canonical_teams(teams)
    
    def test_single_match_parsing(self):
        """Testar parsning av en enskild match"""
        input_text = "Arsenal - Chelsea"
        result = parse_match_input(input_text)
        
        assert len(result) == 1
        assert result[0] == ("Arsenal", "Chelsea")
    
    def test_multiple_matches_parsing(self):
        """Testar parsning av flera matcher"""
        input_text = """Arsenal - Chelsea
Liverpool - Manchester United
Tottenham Hotspur - Arsenal"""
        result = parse_match_input(input_text)
        
        assert len(result) == 3
        assert result[0] == ("Arsenal", "Chelsea")
        assert result[1] == ("Liverpool", "Manchester United")
    
    def test_vs_separator(self):
        """Testar parsning med 'vs' som separator"""
        input_text = "Arsenal vs Chelsea"
        result = parse_match_input(input_text)
        
        assert len(result) == 1
        assert result[0] == ("Arsenal", "Chelsea")
    
    def test_mot_separator(self):
        """Testar parsning med 'mot' som separator"""
        input_text = "Arsenal mot Chelsea"
        result = parse_match_input(input_text)
        
        assert len(result) == 1
        assert result[0] == ("Arsenal", "Chelsea")
    
    def test_empty_lines_ignored(self):
        """Testar att tomma rader ignoreras"""
        input_text = """Arsenal - Chelsea

Liverpool - Manchester United"""
        result = parse_match_input(input_text)
        
        assert len(result) == 2
    
    def test_whitespace_handling(self):
        """Testar hantering av extra whitespace"""
        input_text = "  Arsenal   -   Chelsea  "
        result = parse_match_input(input_text)
        
        assert len(result) == 1
        assert result[0] == ("Arsenal", "Chelsea")
    
    def test_invalid_format_ignored(self):
        """Testar att ogiltiga format ignoreras"""
        input_text = """Arsenal - Chelsea
Invalid line without separator
Liverpool - Manchester United"""
        result = parse_match_input(input_text)
        
        assert len(result) == 2


class TestParseMatchInputNumberedLines:
    """Tester för parse_match_input med radnumrerade rader (STEG 5A)"""

    def setup_method(self):
        teams = ["Arsenal", "Chelsea", "Liverpool", "Manchester United", "Tottenham Hotspur"]
        set_canonical_teams(teams)

    def test_numbered_dot_format(self):
        """Testar 1. Arsenal - Chelsea"""
        result = parse_match_input("1. Arsenal - Chelsea")
        assert len(result) == 1
        assert result[0] == ("Arsenal", "Chelsea")

    def test_numbered_paren_format(self):
        """Testar 1) Arsenal - Chelsea"""
        result = parse_match_input("1) Arsenal - Chelsea")
        assert len(result) == 1
        assert result[0] == ("Arsenal", "Chelsea")

    def test_numbered_colon_format(self):
        """Testar 1: Arsenal - Chelsea"""
        result = parse_match_input("1: Arsenal - Chelsea")
        assert len(result) == 1
        assert result[0] == ("Arsenal", "Chelsea")

    def test_multi_digit_number(self):
        """Testar 13. Arsenal - Chelsea"""
        result = parse_match_input("13. Arsenal - Chelsea")
        assert len(result) == 1
        assert result[0] == ("Arsenal", "Chelsea")

    def test_en_dash_separator(self):
        """Testar en-dash: Arsenal – Chelsea"""
        result = parse_match_input("Arsenal \u2013 Chelsea")
        assert len(result) == 1
        assert result[0] == ("Arsenal", "Chelsea")

    def test_em_dash_separator(self):
        """Testar em-dash: Arsenal — Chelsea"""
        result = parse_match_input("Arsenal \u2014 Chelsea")
        assert len(result) == 1
        assert result[0] == ("Arsenal", "Chelsea")

    def test_vs_dot_separator(self):
        """Testar vs. med punkt"""
        result = parse_match_input("Arsenal vs. Chelsea")
        assert len(result) == 1
        assert result[0] == ("Arsenal", "Chelsea")

    def test_mixed_numbered_and_plain(self):
        """Testar blandning av numrerade och vanliga rader"""
        text = "1. Arsenal - Chelsea\nLiverpool - Manchester United\n3) Tottenham Hotspur - Arsenal"
        result = parse_match_input(text)
        assert len(result) == 3


class TestParseMatchInputWithErrors:
    """Tester för parse_match_input_with_errors (STEG 5A)"""

    def setup_method(self):
        teams = ["Arsenal", "Chelsea", "Liverpool", "Manchester United"]
        set_canonical_teams(teams)

    def test_returns_tuple(self):
        """Testar att funktionen returnerar en tuple (matches, errors)"""
        matches, errors = parse_match_input_with_errors("Arsenal - Chelsea")
        assert isinstance(matches, list)
        assert isinstance(errors, list)

    def test_valid_input_no_errors(self):
        """Inga felmeddelanden för korrekt input"""
        matches, errors = parse_match_input_with_errors("Arsenal - Chelsea")
        assert len(matches) == 1
        assert len(errors) == 0

    def test_invalid_line_produces_error(self):
        """Ogiltiga rader ger felmeddelanden"""
        matches, errors = parse_match_input_with_errors("Arsenal - Chelsea\nBadLine\nLiverpool - Manchester United")
        assert len(matches) == 2
        assert len(errors) == 1
        assert "Rad 2" in errors[0]

    def test_empty_input_no_errors(self):
        """Tom input ger tomma listor"""
        matches, errors = parse_match_input_with_errors("")
        assert len(matches) == 0
        assert len(errors) == 0


class TestPickHalfGuards:
    """Tester för pick_half_guards-funktionen (gain-baserad)"""

    def test_pick_highest_gain_match(self):
        """Testar att matchen med högst gain (second_best) väljs"""
        probs = [
            np.array([0.7, 0.2, 0.1]),   # gain=0.2, top2=0.9
            np.array([0.55, 0.40, 0.05]), # gain=0.40, top2=0.95
            np.array([0.6, 0.3, 0.1]),    # gain=0.3, top2=0.9
        ]
        result = pick_half_guards(probs, n_guards=1)
        assert result == [1]

    def test_gain_beats_higher_entropy(self):
        """Testar att match med högst gain väljs framför match med högre entropy men sämre gain"""
        probs = [
            np.array([0.34, 0.33, 0.33]),  # gain=0.33, top2=0.67 (hög entropy)
            np.array([0.55, 0.40, 0.05]),   # gain=0.40, top2=0.95 (lägre entropy)
        ]
        result = pick_half_guards(probs, n_guards=1)
        assert result == [1]

    def test_none_not_prioritized(self):
        """Testar att matcher utan data (None) INTE prioriteras för gardering"""
        probs = [
            np.array([0.55, 0.40, 0.05]),  # gain=0.40
            None,                            # gain=0.0 (lägst prioritet)
            np.array([0.6, 0.3, 0.1]),      # gain=0.3
        ]
        result = pick_half_guards(probs, n_guards=1)
        assert result == [0]

    def test_none_gets_lowest_priority(self):
        """Testar att None hamnar sist i urvalslistan"""
        probs = [
            None,
            np.array([0.55, 0.40, 0.05]),  # gain=0.40
        ]
        result = pick_half_guards(probs, n_guards=2)
        assert result[0] == 1
        assert result[1] == 0

    def test_tiebreak_by_top2(self):
        """Testar att vid lika gain, högst top2 väljs"""
        probs = [
            np.array([0.50, 0.35, 0.15]),  # gain=0.35, top2=0.85
            np.array([0.55, 0.35, 0.10]),  # gain=0.35, top2=0.90
        ]
        result = pick_half_guards(probs, n_guards=1)
        assert result == [1]

    def test_tiebreak_by_index(self):
        """Testar att vid lika gain och top2, lägst index väljs"""
        probs = [
            np.array([0.55, 0.35, 0.10]),  # gain=0.35, top2=0.90
            np.array([0.55, 0.35, 0.10]),  # gain=0.35, top2=0.90
        ]
        result = pick_half_guards(probs, n_guards=1)
        assert result == [0]

    def test_selection_order_by_gain(self):
        """Testar att urvalet sorteras efter gain (högst först)"""
        probs = [
            np.array([0.7, 0.2, 0.1]),   # gain=0.2
            np.array([0.55, 0.40, 0.05]), # gain=0.40
            np.array([0.5, 0.3, 0.2]),    # gain=0.3
        ]
        result = pick_half_guards(probs, n_guards=2)
        assert result[0] == 1  # gain=0.40
        assert result[1] == 2  # gain=0.3

    def test_zero_guards_requested(self):
        """Testar när inga garderingar begärs"""
        probs = [
            np.array([0.7, 0.2, 0.1]),
            np.array([0.4, 0.35, 0.25])
        ]
        result = pick_half_guards(probs, n_guards=0)
        assert len(result) == 0

    def test_more_guards_than_matches(self):
        """Testar när fler garderingar begärs än det finns matcher"""
        probs = [
            np.array([0.7, 0.2, 0.1]),
            np.array([0.4, 0.35, 0.25])
        ]
        result = pick_half_guards(probs, n_guards=5)
        assert len(result) <= 2


class TestCalculateMatchEntropy:
    """Tester för calculate_match_entropy-funktionen"""

    def test_returns_none_for_none_probs(self):
        """Testar att None returneras om probs är None"""
        result = calculate_match_entropy(None)
        assert result is None

    def test_returns_float_for_valid_probs(self):
        """Testar att ett float-värde returneras för giltiga sannolikheter"""
        probs = np.array([0.5, 0.3, 0.2])
        result = calculate_match_entropy(probs)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_high_entropy_for_uniform(self):
        """Testar att entropy är hög för uniform fördelning"""
        probs = np.array([1/3, 1/3, 1/3])
        result = calculate_match_entropy(probs)
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_low_entropy_for_certain(self):
        """Testar att entropy är låg för säker prediktion"""
        probs = np.array([0.9, 0.05, 0.05])
        result = calculate_match_entropy(probs)
        assert result < 0.5


class TestGetHalfguardSign:
    """Tester för get_halfguard_sign-funktionen"""
    
    def test_remove_least_likely_outcome(self):
        """Testar att minst sannolika utfallet tas bort"""
        probs = np.array([0.6, 0.3, 0.1])
        result = get_halfguard_sign(probs)
        assert result == "1X"
    
    def test_draw_least_likely(self):
        """Testar när oavgjort är minst sannolikt"""
        probs = np.array([0.5, 0.1, 0.4])
        result = get_halfguard_sign(probs)
        assert result == "12"
    
    def test_home_win_least_likely(self):
        """Testar när hemmavinst är minst sannolikt"""
        probs = np.array([0.1, 0.4, 0.5])
        result = get_halfguard_sign(probs)
        assert result == "X2"
    
    def test_equal_probabilities(self):
        """Testar med lika sannolikheter"""
        probs = np.array([0.33, 0.33, 0.34])
        result = get_halfguard_sign(probs)
        assert result in ["1X", "X2", "12"]
        assert len(result) == 2


class TestPickHalfGuardsCombined:
    """Tester för pick_half_guards_combined (gain-baserad)"""

    def _make_cm(self, p1, px, p2, home="A", away="B"):
        """Hjälpmetod för att skapa CombinedMatchProbability."""
        from uncertainty import entropy_norm
        return CombinedMatchProbability(
            home_team=home,
            away_team=away,
            prob_1=p1,
            prob_x=px,
            prob_2=p2,
            entropy=entropy_norm(p1, px, p2),
            sources={"odds": True, "model": True, "streck": True},
        )

    def test_combined_picks_highest_gain(self):
        """Testar att combined-logiken väljer matchen med högst gain"""
        cms = [
            self._make_cm(0.7, 0.2, 0.1),   # gain=0.2
            self._make_cm(0.55, 0.40, 0.05), # gain=0.40
        ]
        result = pick_half_guards_combined(cms, n_guards=1)
        assert result == [1]

    def test_combined_tiebreak_by_top2(self):
        """Testar att vid lika gain, högst top2 väljs i combined"""
        cms = [
            self._make_cm(0.50, 0.35, 0.15),  # gain=0.35, top2=0.85
            self._make_cm(0.55, 0.35, 0.10),  # gain=0.35, top2=0.90
        ]
        result = pick_half_guards_combined(cms, n_guards=1)
        assert result == [1]

    def test_combined_zero_guards(self):
        """Testar att 0 garderingar returnerar tom lista"""
        cms = [self._make_cm(0.5, 0.3, 0.2)]
        result = pick_half_guards_combined(cms, n_guards=0)
        assert result == []


class TestGetHalfguardSignCombined:
    """Tester för get_halfguard_sign_combined-funktionen"""

    def _make_cm(self, p1, px, p2):
        from uncertainty import entropy_norm
        return CombinedMatchProbability(
            home_team="A",
            away_team="B",
            prob_1=p1,
            prob_x=px,
            prob_2=p2,
            entropy=entropy_norm(p1, px, p2),
            sources={"odds": True, "model": True, "streck": True},
        )

    def test_removes_least_likely(self):
        """Testar att minst sannolika utfallet tas bort"""
        cm = self._make_cm(0.6, 0.3, 0.1)
        result = get_halfguard_sign_combined(cm)
        assert result == "1X"

    def test_removes_home_when_least_likely(self):
        """Testar att hemmavinst tas bort om minst sannolik"""
        cm = self._make_cm(0.1, 0.4, 0.5)
        result = get_halfguard_sign_combined(cm)
        assert result == "X2"
