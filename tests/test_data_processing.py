"""
Enhetstester för data_processing.py
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from data_processing import normalize_csv_data


class TestNormalizeCSVData:
    """Tester för normalize_csv_data-funktionen"""
    
    def test_valid_csv_processing(self, tmp_path):
        """Testar att giltig CSV-data bearbetas korrekt"""
        # Skapa en temporär CSV-fil med testdata
        csv_content = """Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR
01/01/2024,Arsenal,Chelsea,2,1,H
02/01/2024,Liverpool,Man City,1,1,D
03/01/2024,Tottenham,Newcastle,0,2,A"""
        
        csv_file = tmp_path / "test_data.csv"
        csv_file.write_text(csv_content)
        
        # Bearbeta filen
        df = normalize_csv_data([csv_file])
        
        # Verifiera resultat
        assert len(df) == 3
        assert list(df.columns) == ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "League"]
        assert df["FTHG"].dtype == np.int64
        assert df["FTAG"].dtype == np.int64
        assert pd.api.types.is_datetime64_any_dtype(df["Date"])
    
    def test_missing_columns(self, tmp_path):
        """Testar hantering av CSV med saknade kolumner"""
        csv_content = """Date,HomeTeam,AwayTeam
01/01/2024,Arsenal,Chelsea"""
        
        csv_file = tmp_path / "incomplete.csv"
        csv_file.write_text(csv_content)
        
        df = normalize_csv_data([csv_file])
        
        # Ska returnera tom DataFrame när nödvändiga kolumner saknas
        assert df.empty
    
    def test_invalid_date_handling(self, tmp_path):
        """Testar hantering av ogiltiga datum"""
        csv_content = """Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR
invalid_date,Arsenal,Chelsea,2,1,H
02/01/2024,Liverpool,Man City,1,1,D"""
        
        csv_file = tmp_path / "bad_dates.csv"
        csv_file.write_text(csv_content)
        
        df = normalize_csv_data([csv_file])
        
        # Rader med ogiltiga datum ska tas bort
        assert len(df) == 1
        assert df.iloc[0]["HomeTeam"] == "Liverpool"
    
    def test_invalid_goals_handling(self, tmp_path):
        """Testar hantering av ogiltiga målsiffror"""
        csv_content = """Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR
01/01/2024,Arsenal,Chelsea,invalid,1,H
02/01/2024,Liverpool,Man City,1,1,D"""
        
        csv_file = tmp_path / "bad_goals.csv"
        csv_file.write_text(csv_content)
        
        df = normalize_csv_data([csv_file])
        
        # Rader med ogiltiga mål ska tas bort
        assert len(df) == 1
        assert df.iloc[0]["HomeTeam"] == "Liverpool"
    
    def test_multiple_files_concatenation(self, tmp_path):
        """Testar sammanslagning av flera CSV-filer"""
        csv1_content = """Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR
01/01/2024,Arsenal,Chelsea,2,1,H"""
        
        csv2_content = """Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR
02/01/2024,Liverpool,Man City,1,1,D"""
        
        csv1 = tmp_path / "file1.csv"
        csv2 = tmp_path / "file2.csv"
        csv1.write_text(csv1_content)
        csv2.write_text(csv2_content)
        
        df = normalize_csv_data([csv1, csv2])
        
        # Båda filerna ska slås samman
        assert len(df) == 2
    
    def test_empty_file_list(self):
        """Testar hantering av tom fillista"""
        df = normalize_csv_data([])
        assert df.empty
    
    def test_nonexistent_file(self, tmp_path):
        """Testar hantering av icke-existerande fil"""
        fake_file = tmp_path / "does_not_exist.csv"
        df = normalize_csv_data([fake_file])
        assert df.empty
    
    def test_missing_values_handling(self, tmp_path):
        """Testar hantering av saknade värden"""
        csv_content = """Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR
01/01/2024,Arsenal,,2,1,H
02/01/2024,Liverpool,Man City,1,1,D
03/01/2024,,Newcastle,0,2,A"""
        
        csv_file = tmp_path / "missing_values.csv"
        csv_file.write_text(csv_content)
        
        df = normalize_csv_data([csv_file])
        
        # Rader med saknade lagnamn ska tas bort
        assert len(df) == 1
        assert df.iloc[0]["HomeTeam"] == "Liverpool"
