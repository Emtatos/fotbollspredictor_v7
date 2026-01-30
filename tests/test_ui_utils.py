"""
Enhetstester för ui_utils.py
"""
import math
import pytest
import numpy as np
from ui_utils import (
    parse_match_input, 
    pick_half_guards, 
    compute_half_guard_gain,
    get_halfguard_sign, 
    calculate_match_entropy
)
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


class TestPickHalfGuards:
    """Tester för pick_half_guards-funktionen (entropy-baserad)"""
    
    def test_pick_most_uncertain_matches_by_entropy(self):
        """Testar att matcher med högst entropy väljs för halvgardering"""
        # Skapa sannolikheter där andra matchen har högst entropy (mest osäker)
        probs = [
            np.array([0.7, 0.2, 0.1]),  # Tydlig favorit, låg entropy
            np.array([0.34, 0.33, 0.33]),  # Nära uniform, hög entropy
            np.array([0.6, 0.3, 0.1])   # Ganska tydlig, medel entropy
        ]
        
        result = pick_half_guards(probs, n_guards=1)
        
        # Index 1 (andra matchen med högst entropy) ska väljas
        assert 1 in result
    
    def test_entropy_based_selection_order(self):
        """Testar att urvalet sorteras efter entropy (högst först)"""
        probs = [
            np.array([0.7, 0.2, 0.1]),  # Låg entropy
            np.array([0.34, 0.33, 0.33]),  # Högst entropy (nära uniform)
            np.array([0.5, 0.3, 0.2])   # Medel entropy
        ]
        
        result = pick_half_guards(probs, n_guards=2)
        
        # Index 1 ska komma först (högst entropy), sedan index 2
        assert result[0] == 1
        assert result[1] == 2
    
    def test_none_values_prioritized(self):
        """Testar att matcher utan data prioriteras för gardering"""
        probs = [
            np.array([0.7, 0.2, 0.1]),
            None,  # Saknar data
            np.array([0.6, 0.3, 0.1])
        ]
        
        result = pick_half_guards(probs, n_guards=1)
        
        # Index 1 (None) ska prioriteras
        assert 1 in result
    
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
        
        # Ska returnera max 2 (antalet matcher)
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
        # Hemmavinst mest sannolik, bortavinst minst sannolik
        probs = np.array([0.6, 0.3, 0.1])
        result = get_halfguard_sign(probs)
        
        # Ska returnera "1X" (ta bort 2)
        assert result == "1X"
    
    def test_draw_least_likely(self):
        """Testar när oavgjort är minst sannolikt"""
        probs = np.array([0.5, 0.1, 0.4])
        result = get_halfguard_sign(probs)
        
        # Ska returnera "12" (ta bort X)
        assert result == "12"
    
    def test_home_win_least_likely(self):
        """Testar när hemmavinst är minst sannolikt"""
        probs = np.array([0.1, 0.4, 0.5])
        result = get_halfguard_sign(probs)
        
        # Ska returnera "X2" (ta bort 1)
        assert result == "X2"
    
    def test_equal_probabilities(self):
        """Testar med lika sannolikheter"""
        probs = np.array([0.33, 0.33, 0.34])
        result = get_halfguard_sign(probs)
        
        # Ska ta bort ett av de två första (minst sannolika)
        assert result in ["1X", "X2", "12"]
        assert len(result) == 2


class TestComputeHalfGuardGain:
    """Tester för compute_half_guard_gain-funktionen"""
    
    def test_gain_calculation(self):
        """Testar att gain beräknas korrekt"""
        # probs = [0.6, 0.3, 0.1] -> p_spik = 0.6, p_half = 0.6 + 0.3 = 0.9
        # gain = log(0.9) - log(0.6) = log(0.9/0.6) = log(1.5)
        probs = np.array([0.6, 0.3, 0.1])
        gain = compute_half_guard_gain(probs)
        expected = math.log(0.9) - math.log(0.6)
        assert gain == pytest.approx(expected, abs=1e-10)
    
    def test_gain_higher_for_uncertain_match(self):
        """Testar att gain är högre för osäkra matcher"""
        # Säker match: [0.8, 0.15, 0.05] -> gain = log(0.95) - log(0.8)
        # Osäker match: [0.4, 0.35, 0.25] -> gain = log(0.75) - log(0.4)
        certain_probs = np.array([0.8, 0.15, 0.05])
        uncertain_probs = np.array([0.4, 0.35, 0.25])
        
        gain_certain = compute_half_guard_gain(certain_probs)
        gain_uncertain = compute_half_guard_gain(uncertain_probs)
        
        assert gain_uncertain > gain_certain
    
    def test_gain_zero_for_perfect_prediction(self):
        """Testar att gain är nära 0 för perfekt prediktion"""
        # Om p_spik = 1.0, p_half = 1.0, gain = 0
        probs = np.array([1.0, 0.0, 0.0])
        gain = compute_half_guard_gain(probs)
        assert gain == pytest.approx(0.0, abs=1e-10)


class TestGainBasedHalfGuardSelection:
    """Tester för gain-baserat urval av halvgarderingar"""
    
    def test_n_half_zero_returns_empty(self):
        """Testar att N_HALF=0 ger inga halvgarderingar"""
        probs = [
            np.array([0.6, 0.3, 0.1]),
            np.array([0.4, 0.35, 0.25]),
            np.array([0.5, 0.3, 0.2]),
        ]
        
        result = pick_half_guards(probs, n_guards=0)
        
        assert result == []
    
    def test_n_half_two_returns_exactly_two(self):
        """Testar att N_HALF=2 ger exakt 2 halvgarderingar"""
        probs = [
            np.array([0.6, 0.3, 0.1]),
            np.array([0.4, 0.35, 0.25]),
            np.array([0.5, 0.3, 0.2]),
        ]
        
        result = pick_half_guards(probs, n_guards=2)
        
        assert len(result) == 2
    
    def test_selects_highest_gain_match(self):
        """Testar att matchen med högst gain väljs när N_HALF=1"""
        # Match 0: säker [0.8, 0.15, 0.05] -> låg gain (log(0.95/0.8))
        # Match 1: osäker [0.4, 0.35, 0.25] -> hög gain (log(0.75/0.4))
        # Match 2: medel [0.6, 0.25, 0.15] -> medel gain (log(0.85/0.6))
        probs = [
            np.array([0.8, 0.15, 0.05]),
            np.array([0.4, 0.35, 0.25]),
            np.array([0.6, 0.25, 0.15]),
        ]
        
        result = pick_half_guards(probs, n_guards=1)
        
        # Match 1 ska väljas (högst gain)
        assert result == [1]
    
    def test_tie_breaker_entropy(self):
        """Testar tie-breaker: högre entropy först vid samma gain"""
        # Två matcher med samma probs (samma gain) men olika entropy
        probs = [
            np.array([0.5, 0.3, 0.2]),
            np.array([0.5, 0.3, 0.2]),
        ]
        entropy_values = [0.7, 0.9]  # Match 1 har högre entropy
        
        result = pick_half_guards(probs, n_guards=1, entropy_values=entropy_values)
        
        # Match 1 ska väljas (högre entropy)
        assert result == [1]
    
    def test_tie_breaker_trust(self):
        """Testar tie-breaker: lägre trust först vid samma gain och entropy"""
        probs = [
            np.array([0.5, 0.3, 0.2]),
            np.array([0.5, 0.3, 0.2]),
        ]
        entropy_values = [0.8, 0.8]  # Samma entropy
        trust_scores = [80, 40]  # Match 1 har lägre trust
        
        result = pick_half_guards(
            probs, n_guards=1, 
            entropy_values=entropy_values,
            trust_scores=trust_scores
        )
        
        # Match 1 ska väljas (lägre trust = sämre data = mer osäker)
        assert result == [1]
    
    def test_none_probs_prioritized(self):
        """Testar att matcher utan data prioriteras"""
        probs = [
            np.array([0.6, 0.3, 0.1]),
            None,  # Saknar data -> inf gain
            np.array([0.5, 0.3, 0.2]),
        ]
        
        result = pick_half_guards(probs, n_guards=1)
        
        # Match 1 (None) ska prioriteras (inf gain)
        assert result == [1]


class TestHalfGuardSignIsTop2:
    """Tester för att verifiera att HALV-tecken alltid är top2"""
    
    def test_top2_is_1x(self):
        """Testar att top2 blir 1X när 2 är minst sannolik"""
        probs = np.array([0.5, 0.4, 0.1])  # top2 = 1, X
        result = get_halfguard_sign(probs)
        assert result == "1X"
    
    def test_top2_is_12(self):
        """Testar att top2 blir 12 när X är minst sannolik"""
        probs = np.array([0.5, 0.1, 0.4])  # top2 = 1, 2
        result = get_halfguard_sign(probs)
        assert result == "12"
    
    def test_top2_is_x2(self):
        """Testar att top2 blir X2 när 1 är minst sannolik"""
        probs = np.array([0.1, 0.4, 0.5])  # top2 = X, 2
        result = get_halfguard_sign(probs)
        assert result == "X2"
    
    def test_halfguard_sign_always_two_outcomes(self):
        """Testar att halvgarderingstecknet alltid har exakt 2 utfall"""
        test_cases = [
            np.array([0.6, 0.3, 0.1]),
            np.array([0.4, 0.35, 0.25]),
            np.array([0.33, 0.34, 0.33]),
            np.array([0.1, 0.1, 0.8]),
        ]
        
        for probs in test_cases:
            result = get_halfguard_sign(probs)
            assert len(result) == 2
            assert all(c in "1X2" for c in result)
