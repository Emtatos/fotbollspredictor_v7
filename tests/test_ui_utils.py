"""
Enhetstester för ui_utils.py
"""
import pytest
import numpy as np
from ui_utils import parse_match_input, pick_half_guards, get_halfguard_sign
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
    """Tester för pick_half_guards-funktionen"""
    
    def test_pick_most_uncertain_matches(self):
        """Testar att de mest osäkra matcherna väljs"""
        # Skapa sannolikheter där andra matchen är mest osäker
        probs = [
            np.array([0.7, 0.2, 0.1]),  # Tydlig favorit
            np.array([0.4, 0.35, 0.25]),  # Mycket osäker
            np.array([0.6, 0.3, 0.1])   # Ganska tydlig
        ]
        
        result = pick_half_guards(probs, n_guards=1)
        
        # Index 1 (andra matchen) ska väljas
        assert 1 in result
    
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
