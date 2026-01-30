"""
Enhetstester för utils.py
"""
import pytest
from utils import normalize_team_name, set_canonical_teams, get_canonical_teams


class TestNormalizeTeamName:
    """Tester för normalize_team_name-funktionen"""
    
    def setup_method(self):
        """Körs före varje test - sätt upp kanoniska lagnamn"""
        canonical = [
            "Manchester United", "Manchester City", "Arsenal", "Chelsea",
            "Liverpool", "Tottenham Hotspur", "Newcastle United",
            "Brighton & Hove Albion", "West Ham United",
            "Nottingham Forest", "Wolverhampton Wanderers",
            "Sheffield United", "Sheffield Wednesday",
            "Queens Park Rangers", "Birmingham City",
            "Leicester City", "Norwich City", "Derby County"
        ]
        set_canonical_teams(canonical)
    
    def test_exact_match(self):
        """Testar exakt matchning av lagnamn"""
        assert normalize_team_name("Arsenal") == "Arsenal"
        assert normalize_team_name("Chelsea") == "Chelsea"
    
    def test_case_insensitive(self):
        """Testar case-insensitiv matchning"""
        assert normalize_team_name("arsenal") == "Arsenal"
        assert normalize_team_name("CHELSEA") == "Chelsea"
        assert normalize_team_name("ArSeNaL") == "Arsenal"
    
    def test_manual_aliases(self):
        """Testar manuella alias"""
        assert normalize_team_name("Man United") == "Manchester United"
        assert normalize_team_name("Man City") == "Manchester City"
        assert normalize_team_name("Spurs") == "Tottenham Hotspur"
        assert normalize_team_name("Wolves") == "Wolverhampton Wanderers"
        assert normalize_team_name("QPR") == "Queens Park Rangers"
    
    def test_dash_normalization(self):
        """Testar normalisering av olika typer av bindestreck"""
        # En-dash, em-dash och minus ska bli vanligt bindestreck
        set_canonical_teams(["Brighton & Hove Albion"])
        assert normalize_team_name("Brighton & Hove Albion") == "Brighton & Hove Albion"
    
    def test_whitespace_normalization(self):
        """Testar normalisering av whitespace"""
        assert normalize_team_name("  Arsenal  ") == "Arsenal"
        assert normalize_team_name("Manchester   United") == "Manchester United"
    
    def test_fuzzy_matching(self):
        """Testar fuzzy matching för liknande namn"""
        # Nottingham Forest varianter
        assert normalize_team_name("Nott'm Forest") == "Nottingham Forest"
        assert normalize_team_name("Nottm Forest") == "Nottingham Forest"
        
        # Sheffield varianter
        assert normalize_team_name("Sheff Utd") == "Sheffield United"
        assert normalize_team_name("Sheff Wed") == "Sheffield Wednesday"
    
    def test_empty_and_invalid_input(self):
        """Testar hantering av tomma och ogiltiga inputs"""
        assert normalize_team_name("") == ""
        # Whitespace-only strängar returneras som de är efter normalisering
        result = normalize_team_name("   ")
        assert result.strip() == ""
        assert normalize_team_name(None) == None
    
    def test_unknown_team_returns_original(self):
        """Testar att okända lag returnerar originalnamnet"""
        unknown = "Completely Unknown FC"
        result = normalize_team_name(unknown)
        # Ska returnera det normaliserade namnet (trimmat)
        assert result == unknown
    
    def test_apostrophe_normalization(self):
        """Testar normalisering av olika apostrofer"""
        set_canonical_teams(["Nottingham Forest"])
        # Olika varianter ska normaliseras till det kanoniska namnet
        assert normalize_team_name("Nott'm Forest") == "Nottingham Forest"
        assert normalize_team_name("Nottm Forest") == "Nottingham Forest"


class TestCanonicalTeams:
    """Tester för hantering av kanoniska lagnamn"""
    
    def test_set_and_get_canonical_teams(self):
        """Testar att sätta och hämta kanoniska lagnamn"""
        teams = ["Arsenal", "Chelsea", "Liverpool"]
        set_canonical_teams(teams)
        canonical = get_canonical_teams()
        
        assert len(canonical) == 3
        assert "Arsenal" in canonical
        assert "Chelsea" in canonical
        assert "Liverpool" in canonical
    
    def test_canonical_teams_cleaned(self):
        """Testar att kanoniska lagnamn rensas från whitespace"""
        teams = ["  Arsenal  ", "Chelsea", "  Liverpool  "]
        set_canonical_teams(teams)
        canonical = get_canonical_teams()
        
        # Ska vara trimmade
        assert "Arsenal" in canonical
        assert "  Arsenal  " not in canonical
    
    def test_empty_strings_filtered(self):
        """Testar att tomma strängar filtreras bort"""
        teams = ["Arsenal", "", "   ", "Chelsea"]
        set_canonical_teams(teams)
        canonical = get_canonical_teams()
        
        assert len(canonical) == 2
        assert "Arsenal" in canonical
        assert "Chelsea" in canonical
