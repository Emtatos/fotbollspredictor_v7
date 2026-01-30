"""
Enhetstester för feature_engineering.py
"""
import pytest
import pandas as pd
import numpy as np
from feature_engineering import create_features


class TestCreateFeatures:
    """Tester för create_features-funktionen"""
    
    def test_basic_feature_creation(self):
        """Testar grundläggande feature-skapande"""
        # Skapa minimal testdata
        data = {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "HomeTeam": ["Arsenal", "Chelsea", "Arsenal"],
            "AwayTeam": ["Chelsea", "Liverpool", "Liverpool"],
            "FTHG": [2, 1, 3],
            "FTAG": [1, 1, 0],
            "FTR": ["H", "D", "H"]
        }
        df = pd.DataFrame(data)
        
        result = create_features(df)
        
        # Verifiera att nya kolumner skapats
        expected_cols = ["HomeFormPts", "HomeFormGD", "AwayFormPts", "AwayFormGD", "HomeElo", "AwayElo"]
        for col in expected_cols:
            assert col in result.columns
    
    def test_empty_dataframe(self):
        """Testar hantering av tom DataFrame"""
        df = pd.DataFrame()
        result = create_features(df)
        assert result.empty
    
    def test_elo_initialization(self):
        """Testar att ELO initialiseras till 1500"""
        data = {
            "Date": pd.to_datetime(["2024-01-01"]),
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
            "FTHG": [2],
            "FTAG": [1],
            "FTR": ["H"]
        }
        df = pd.DataFrame(data)
        
        result = create_features(df)
        
        # Första matchen ska ha ELO 1500 för båda lagen
        assert result.iloc[0]["HomeElo"] == 1500.0
        assert result.iloc[0]["AwayElo"] == 1500.0
    
    def test_form_calculation(self):
        """Testar beräkning av form över flera matcher"""
        # Skapa data där Arsenal vinner alla matcher
        dates = pd.date_range("2024-01-01", periods=6, freq="D")
        data = {
            "Date": dates,
            "HomeTeam": ["Arsenal"] * 6,
            "AwayTeam": ["Chelsea", "Liverpool", "Tottenham", "Man City", "Newcastle", "Brighton"],
            "FTHG": [3] * 6,
            "FTAG": [0] * 6,
            "FTR": ["H"] * 6
        }
        df = pd.DataFrame(data)
        
        result = create_features(df)
        
        # Efter 5 vinster ska form-poängen vara 3.0 (genomsnitt av senaste 5)
        last_row = result.iloc[-1]
        assert last_row["HomeFormPts"] == 3.0
    
    def test_elo_updates_after_win(self):
        """Testar att ELO uppdateras korrekt efter vinst"""
        data = {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "HomeTeam": ["Arsenal", "Arsenal"],
            "AwayTeam": ["Chelsea", "Liverpool"],
            "FTHG": [3, 2],
            "FTAG": [0, 1],
            "FTR": ["H", "H"]
        }
        df = pd.DataFrame(data)
        
        result = create_features(df)
        
        # Efter vinst ska hemmalaget ha högre ELO än 1500
        assert result.iloc[1]["HomeElo"] > 1500.0
    
    def test_data_sorting_by_date(self):
        """Testar att data sorteras efter datum"""
        # Skapa data i fel ordning
        data = {
            "Date": pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"]),
            "HomeTeam": ["Arsenal", "Chelsea", "Liverpool"],
            "AwayTeam": ["Chelsea", "Liverpool", "Arsenal"],
            "FTHG": [2, 1, 3],
            "FTAG": [1, 1, 0],
            "FTR": ["H", "D", "H"]
        }
        df = pd.DataFrame(data)
        
        result = create_features(df)
        
        # Verifiera att resultatet är sorterat
        assert result["Date"].is_monotonic_increasing
    
    def test_goal_difference_calculation(self):
        """Testar beräkning av målskillnad i form"""
        data = {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "HomeTeam": ["Arsenal", "Arsenal"],
            "AwayTeam": ["Chelsea", "Liverpool"],
            "FTHG": [3, 2],
            "FTAG": [1, 1],
            "FTR": ["H", "H"]
        }
        df = pd.DataFrame(data)
        
        result = create_features(df)
        
        # Andra matchen ska ha form-GD baserat på första matchens +2
        second_match = result.iloc[1]
        assert second_match["HomeFormGD"] == 2.0  # Från första matchen (3-1)
